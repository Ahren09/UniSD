"""GSM8K evaluation script using vLLM. Pass@1 with greedy decoding."""

import argparse
import gc
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.utils.env_utils import disable_thinking_mode, get_model_dtype  # noqa: F401 — env_utils import patches transformers for custom-code models
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.eval.eval_args import add_common_eval_args
from src.utils.lora_utils import prepare_model_for_vllm
from src.data.gsm8k import load_gsm8k
from src.prompts.math import build_gsm8k_prompt
from src.utils.eval_utils import (
    compute_accuracy,
    compute_extraction_success_rate,
)
from src.utils.path_utils import get_eval_results_path
from src.analysis.resource_consumption_utils import (
    append_resource_record,
    compute_resource_consumption_record,
    detect_num_gpus,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GSM8K evaluation (pass@1)")
    add_common_eval_args(parser, max_new_tokens_default=512)
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"])
    return parser.parse_args()


# ── Answer extraction ────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def _normalize_number(s: str) -> str | None:
    """Strip commas, $, %, and trailing punctuation. Return canonical numeric string or None."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.rstrip(".").strip()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    if f.is_integer():
        return str(int(f))
    # Round to a stable representation (avoid 6.000000001)
    return f"{f:.6f}".rstrip("0").rstrip(".")


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract numeric answer from a GSM8K generation.

    Order of attempts:
      1) Last '####' marker → token after it.
      2) 'The answer is X' / 'final answer is X' phrasing.
      3) Last number in the text.
    """
    if not text:
        return None

    # 1) #### sentinel
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        m = _NUM_RE.search(tail)
        if m:
            return _normalize_number(m.group(0))

    # 2) "answer is X"
    for phrase in ["the answer is", "The answer is", "final answer is", "Final answer is", "answer:", "Answer:"]:
        if phrase in text:
            tail = text.rsplit(phrase, 1)[1]
            m = _NUM_RE.search(tail)
            if m:
                return _normalize_number(m.group(0))

    # 3) Last number anywhere
    matches = _NUM_RE.findall(text)
    if matches:
        return _normalize_number(matches[-1])
    return None


def _answers_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    pred_n = _normalize_number(pred)
    gold_n = _normalize_number(gold)
    if pred_n is None or gold_n is None:
        return False
    if pred_n == gold_n:
        return True
    # Numeric tolerance fallback (handles 6.0 vs 6.00000001 type issues)
    try:
        return abs(float(pred_n) - float(gold_n)) < 1e-4
    except ValueError:
        return False


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    t0 = time.time()

    if "/" in args.model_name_or_path and not os.path.exists(args.model_name_or_path):
        pass  # HF model id
    else:
        assert os.path.exists(args.model_name_or_path), f"Model {args.model_name_or_path} does not exist"

    print(f"Loading GSM8K (split={args.split})...")
    problems = load_gsm8k(args.split, args.max_samples)
    print(f"Loaded {len(problems)} problems")

    prompts_raw = [build_gsm8k_prompt(p) for p in problems]
    task_ids = [p["task_id"] for p in problems]

    _clean_path = args.model_name_or_path.rstrip("/")
    if _clean_path.endswith("_merged"):
        _clean_path = os.path.dirname(_clean_path)
    if "checkpoint-" in os.path.basename(_clean_path):
        model_basename = os.path.basename(os.path.dirname(_clean_path))
    else:
        model_basename = os.path.basename(_clean_path)

    model_path = prepare_model_for_vllm(
        args.model_name_or_path, force_remerge=getattr(args, "force_remerge", False),
    )
    max_model_len = args.max_prompt_length + args.max_new_tokens
    print(f"Initializing vLLM (max_model_len={max_model_len})...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=get_model_dtype(),
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    disable_thinking_mode(tokenizer)
    prompts = []
    for p in prompts_raw:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted)
    print(f"Sample prompt (first):\n{prompts[0][:600]}...")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )
    print(f"Generating with temperature={args.temperature}, max_tokens={args.max_new_tokens}...")
    outputs = llm.generate(prompts, sampling_params)
    generated_texts = [o.outputs[0].text for o in outputs]

    # Free vLLM before file I/O.
    del llm
    gc.collect()

    print("Evaluating GSM8K answers...")
    results = []
    for i, (problem, gen_text) in enumerate(zip(problems, generated_texts)):
        extracted = extract_gsm8k_answer(gen_text)
        correct = _answers_equal(extracted, problem["gold_answer"])
        results.append({
            "task_id": problem["task_id"],
            "question": problem["question"][:200],
            "gold_answer": problem["gold_answer"],
            "extracted_answer": extracted,
            "correct": correct,
            "generated_text": gen_text,
        })
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(problems)}] running accuracy={compute_accuracy(results):.3f}")

    accuracy = compute_accuracy(results)
    extraction_rate = compute_extraction_success_rate(results)

    print(f"\n{'=' * 60}")
    print(f"Results: gsm8k {args.split} | {model_basename}")
    print(f"  accuracy: {accuracy:.4f} ({sum(r['correct'] for r in results)}/{len(results)})")
    print(f"  extraction_success_rate: {extraction_rate:.4f}")

    jsonl_path = get_eval_results_path(
        args.output_dir, args.mode, "gsm8k", args.model_name_or_path,
        args.split, "jsonl", subset=None, hparam_tag=args.hparam_tag,
        suffix=args.suffix,
    )
    os.makedirs(jsonl_path.parent, exist_ok=True)
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
        summary = {
            "__summary__": True,
            "dataset": "gsm8k",
            "split": args.split,
            "model": args.model_name_or_path,
            "model_basename": model_basename,
            "total": len(results),
            "accuracy": accuracy,
            "extraction_success_rate": extraction_rate,
        }
        f.write(json.dumps(summary) + "\n")
    print(f"JSONL written to: {jsonl_path}")

    try:
        peak_gb = None
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        record = compute_resource_consumption_record(
            method=getattr(args, "mode", "eval"),
            model=getattr(args, "model_name_or_path", "unknown"),
            dataset="gsm8k",
            phase="eval",
            wall_time_sec=time.time() - t0,
            num_gpus=detect_num_gpus(),
            peak_gpu_mem_gb=peak_gb,
            subset=None,
            split=getattr(args, "split", None),
            max_samples=getattr(args, "max_samples", None),
            max_new_tokens=getattr(args, "max_new_tokens", None),
            score=accuracy,
        )
        append_resource_record(record)
    except Exception as e:
        print(f"[resource_consumption] WARNING: failed to log record: {e}")


if __name__ == "__main__":
    main()
