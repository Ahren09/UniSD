"""Phase 3: Code generation evaluation script using vLLM."""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.utils.env_utils import get_model_dtype, get_torch_dtype, load_project_env
load_project_env()  # must run before vllm import

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from openpyxl import Workbook

from src.eval.eval_args import parse_args
from src.utils.lora_utils import (
    prepare_model_for_vllm,
    is_lora_checkpoint,
    get_base_model_path,
    load_model_for_inference,
)
from src.data.mbpp import load_mbpp
from src.data.humaneval import load_humaneval
from src.prompts.codegen import build_code_generation_prompt
from src.utils.eval_utils import (
    build_humaneval_test_code,
    build_mbpp_test_code,
    compute_pass_at_1,
    extract_code,
    run_code_test,
    score_completions_under_model,
)
from src.utils.path_utils import get_eval_results_path
from src.analysis.resource_consumption_utils import (
    append_resource_record,
    compute_resource_consumption_record,
    detect_num_gpus,
)


_RETENTION_RECORD_KEYS: tuple[str, ...] = (
    "retention_logp_mean_avg",
    "retention_logp_mean_std",
    "retention_ppl_mean_avg",
    "retention_num_examples",
    "retention_num_empty_completions",
    "retention_total_tokens",
)


def _run_retention_pass(args, prompts: list[str], generated_texts: list[str]) -> list[dict]:
    """Score generated completions under the BASE model and return per-example logp records.

    Loads the base model fresh on a single device (NOT the LoRA-merged FT model),
    runs HF teacher forcing via score_completions_under_model, frees the model,
    and returns the per-example list. Mirrors the helper in eval_mcqa.py.
    """
    if is_lora_checkpoint(args.model_name_or_path):
        base_path = get_base_model_path(args.model_name_or_path)
    elif args.base_model_name_or_path:
        base_path = args.base_model_name_or_path
    else:
        raise ValueError(
            "--compute_retention requires --base_model_name_or_path when the "
            "checkpoint is not a LoRA adapter."
        )

    print(f"[retention] Loading BASE model for scoring: {base_path}")
    base_tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    base_model = load_model_for_inference(
        base_path, torch_dtype=get_torch_dtype(), device_map=None,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model.to(device)
    base_model.eval()

    max_length = args.retention_max_length or (args.max_prompt_length + args.max_new_tokens)
    print(
        f"[retention] Scoring {len(prompts)} completions under base model "
        f"(batch_size={args.retention_batch_size}, max_length={max_length})..."
    )
    records = score_completions_under_model(
        base_model,
        base_tok,
        prompts,
        generated_texts,
        batch_size=args.retention_batch_size,
        max_length=max_length,
    )

    del base_model, base_tok
    gc.collect()

    return records


def main():
    args = parse_args()
    t0 = time.time()



    # Auto-merge LoRA checkpoint if needed
    model_path = prepare_model_for_vllm(args.model_name_or_path, force_remerge=getattr(args, "force_remerge", False))

    # Load dataset
    print(f"Loading {args.dataset} dataset (split={args.split})...")
    if args.dataset == "mbpp":
        problems = load_mbpp(args.split, args.max_samples)
    else:
        problems = load_humaneval(args.max_samples)
    print(f"Loaded {len(problems)} problems")

    # Build prompts
    prompts_raw = [build_code_generation_prompt(p, args.dataset) for p in problems]

    # Apply chat template
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

    prompts = []
    for p in prompts_raw:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted)

    print(f"Sample prompt (first):\n{prompts[0][:300]}...")

    # Check generation cache (vLLM is expensive; code testing is cheap)
    # Strip _merged suffix before extracting model basename
    _clean_path = args.model_name_or_path.rstrip("/")
    if _clean_path.endswith("_merged"):
        _clean_path = os.path.dirname(_clean_path)
    if "checkpoint-" in os.path.basename(_clean_path):
        model_basename = os.path.basename(os.path.dirname(_clean_path))
    else:
        model_basename = os.path.basename(_clean_path)
    
    cache_dir = os.path.dirname(get_eval_results_path(args.output_dir, args.mode, args.dataset, args.model_name_or_path, args.split, "jsonl", hparam_tag=args.hparam_tag, suffix=getattr(args, "suffix", "")))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"generations_{args.dataset}_{args.split}_{model_basename}.json")

    vllm_used = False
    if os.path.exists(cache_path):
        print(f"Loading cached generations from {cache_path}")
        with open(cache_path) as f:
            generated_texts = json.load(f)
    else:
        # Init vLLM


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
        vllm_used = True

        # Generate
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        print(f"Generating with temperature={args.temperature}, max_tokens={args.max_new_tokens}...")
        outputs = llm.generate(prompts, sampling_params)
        generated_texts = [o.outputs[0].text for o in outputs]

        # Save generation cache
        os.makedirs(args.output_dir, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(generated_texts, f)
        print(f"Cached generations to {cache_path}")

    # Optional: retention-to-base scoring under the BASE model.
    # Cache-aware: only release vLLM if it was actually instantiated this run.
    retention_records = None
    if args.compute_retention:
        if vllm_used:
            print("[retention] Releasing vLLM before loading HF base model...")
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            vllm_used = False
        retention_records = _run_retention_pass(args, prompts, generated_texts)

    # Evaluate
    print("Evaluating generated code...")
    results = []
    num_timeouts = 0
    num_exec_errors = 0

    for i, (problem, gen_text) in enumerate(zip(problems, generated_texts)):
        extracted = extract_code(gen_text)

        if args.dataset == "mbpp":
            test_code = build_mbpp_test_code(problem)
            full_code = extracted
        else:
            test_code = build_humaneval_test_code(problem)
            full_code = extracted

        passed, error = run_code_test(full_code, test_code, timeout_s=10)

        if "Timeout" in error:
            num_timeouts += 1
        elif error:
            num_exec_errors += 1

        record = {
            "task_id": problem["task_id"],
            "prompt": problem["prompt"] if args.dataset == "mbpp" else problem["prompt"][:200],
            "generated_text": gen_text,
            "extracted_code": extracted,
            "passed": passed,
            "error": error,
        }
        if retention_records is not None:
            rec = retention_records[i]
            record["retention_logp_mean"] = rec["logp_mean"]
            record["retention_logp_sum"] = rec["logp_sum"]
            record["retention_num_tokens"] = rec["n_tokens"]
        results.append(record)

        if (i + 1) % 20 == 0:
            running_pass = compute_pass_at_1([r["passed"] for r in results])
            print(f"  [{i+1}/{len(problems)}] running pass@1={running_pass:.3f}")

    # Compute metrics
    passed_list = [r["passed"] for r in results]
    pass_at_1 = compute_pass_at_1(passed_list)
    format_success = sum(1 for r in results if r["extracted_code"] != r["generated_text"]) / len(results)

    print(f"\n{'=' * 60}")
    print(f"Results: {args.dataset} | {model_basename}")
    print(f"  pass@1: {pass_at_1:.4f} ({sum(passed_list)}/{len(passed_list)})")
    print(f"  format_success_rate: {format_success:.4f}")
    print(f"  timeouts: {num_timeouts}, exec_errors: {num_exec_errors}")

    # Write outputs
    
    
    results_path = get_eval_results_path(
        output_dir=args.output_dir, mode=args.mode, dataset=args.dataset, model_name=args.model_name_or_path, split=args.split, file_type="jsonl", hparam_tag=args.hparam_tag, suffix=getattr(args, "suffix", "")
    )
    
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
        summary = {
            "__summary__": True,
            "dataset": args.dataset,
            "model": args.model_name_or_path,
            "model_basename": model_basename,
            "total": len(results),
            "pass_at_1": pass_at_1,
            "num_passed": sum(passed_list),
            "format_success_rate": format_success,
            "num_timeouts": num_timeouts,
            "num_exec_errors": num_exec_errors,
        }
        if retention_records is not None:
            valid = [r["logp_mean"] for r in retention_records if r["n_tokens"] > 0]
            empty = sum(1 for r in retention_records if r["n_tokens"] == 0)
            summary["retention_logp_mean_avg"] = float(np.mean(valid)) if valid else None
            summary["retention_logp_mean_std"] = float(np.std(valid)) if valid else None
            ppl_values = [float(np.exp(-v)) for v in valid] if valid else []
            summary["retention_ppl_mean_avg"] = float(np.mean(ppl_values)) if ppl_values else None
            summary["retention_num_examples"] = len(valid)
            summary["retention_num_empty_completions"] = empty
            summary["retention_total_tokens"] = int(sum(r["n_tokens"] for r in retention_records))
            print(
                f"  retention_logp_mean_avg: {summary['retention_logp_mean_avg']} "
                f"(ppl={summary['retention_ppl_mean_avg']}, "
                f"scored={summary['retention_num_examples']}, "
                f"empty={summary['retention_num_empty_completions']}, "
                f"tokens={summary['retention_total_tokens']})"
            )
        f.write(json.dumps(summary) + "\n")

    print(f"JSONL written to: {results_path}")

    
    try:
        xlsx_path = get_eval_results_path(
            output_dir=args.output_dir, mode=args.mode, dataset=args.dataset, model_name=args.model_name_or_path, split=args.split, file_type="xlsx", hparam_tag=args.hparam_tag, suffix=getattr(args, "suffix", "")
        )
        wb = Workbook()
        ws = wb.active
        ws.title = "Summary"
        ws.append(["Metric", "Value"])
        for k, v in summary.items():
            if k == "__summary__":
                continue
            ws.append([k, str(v)])
        wb.save(xlsx_path)
        print(f"XLSX written to: {xlsx_path}")

    except ImportError:
        print("openpyxl not available, skipping XLSX output")

    try:
        peak_gb = None
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        retention_extras = {
            k: summary[k] for k in _RETENTION_RECORD_KEYS if k in summary
        }
        record = compute_resource_consumption_record(
            method=getattr(args, "mode", "eval"),
            model=getattr(args, "model_name_or_path", "unknown"),
            dataset=getattr(args, "dataset", "unknown"),
            phase="eval",
            wall_time_sec=time.time() - t0,
            num_gpus=detect_num_gpus(),
            peak_gpu_mem_gb=peak_gb,
            split=getattr(args, "split", None),
            max_samples=getattr(args, "max_samples", None),
            max_new_tokens=getattr(args, "max_new_tokens", None),
            score=locals().get("pass_at_1"),
            **retention_extras,
        )
        append_resource_record(record)
    except Exception as e:
        print(f"[resource_consumption] WARNING: failed to log record: {e}")


if __name__ == "__main__":
    main()
