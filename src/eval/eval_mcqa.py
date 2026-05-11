"""Unified MCQA evaluation script (GPQA, ScienceQA) using vLLM."""

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.utils.env_utils import disable_thinking_mode, get_model_dtype  # noqa: F401 — env_utils import patches transformers for custom-code models
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from openpyxl import Workbook

from src.eval.eval_args import add_common_eval_args
from src.utils.lora_utils import prepare_model_for_vllm
from src.data.gpqa import load_gpqa
from src.data.scienceqa import load_scienceqa
from src.data.cos_e import load_cos_e
from src.data.medmcqa import load_medmcqa
from src.prompts.mcqa import build_mcqa_prompt
from src.utils.eval_utils import (
    extract_answer,
    compute_accuracy,
    compute_per_domain_accuracy,
    compute_extraction_success_rate,
)
from src.utils.path_utils import get_eval_results_path
from src.eval.eval_retention import (
    RETENTION_SUMMARY_KEYS,
    load_generations_by_task_id,
    run_all_retention_passes,
    load_retention_from_results,
    flatten_retention_into_record,
    add_retention_to_summary,
)
from src.analysis.resource_consumption_utils import (
    append_resource_record,
    compute_resource_consumption_record,
    detect_num_gpus,
)


GPQA_SUBSETS = ["gpqa_main", "gpqa_diamond", "gpqa_extended"]
SCIENCEQA_SUBSETS = ["all", "natural_science", "language_science", "social_science"]

GPQA_SPLITS = ["all", "train", "test"]
SCIENCEQA_SPLITS = ["train", "validation", "test"]


def parse_args():
    parser = argparse.ArgumentParser(description="Unified MCQA evaluation (GPQA, ScienceQA)")
    parser.add_argument("--dataset", required=True, choices=["gpqa", "scienceqa", "cos_e", "medmcqa"],
                        help="Dataset to evaluate on")
    add_common_eval_args(parser, max_new_tokens_default=512)
    parser.add_argument("--subset", type=str, default=None,
                        help="Dataset subset (gpqa: gpqa_main/gpqa_diamond/gpqa_extended; scienceqa: all/natural_science/...)")
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split (gpqa: all/train/test; scienceqa: train/validation/test)")
    parser.add_argument("--structured_output", type=str, default=None,
                        choices=[None, "letter", "json"],
                        help="Output mode: none (free-form CoT), letter (single char), json")
    parser.add_argument("--include_lecture", action=argparse.BooleanOptionalAction, default=True,
                        help="Include lecture text as context in prompts (ScienceQA only)")
    parser.add_argument("--disable_thinking_mode", action="store_true", default=False,
                        help="Disable Qwen3-style thinking mode in the chat template (matches training).")
    return parser.parse_args()


def _apply_defaults_and_validate(args):
    """Apply dataset-specific defaults and validate subset/split choices."""
    if args.dataset == "gpqa":
        if args.subset is None:
            args.subset = "gpqa_main"
        if args.split is None:
            args.split = "all"
        if args.subset not in GPQA_SUBSETS:
            raise ValueError(f"Invalid GPQA subset '{args.subset}'. Choose from: {GPQA_SUBSETS}")
        if args.split not in GPQA_SPLITS:
            raise ValueError(f"Invalid GPQA split '{args.split}'. Choose from: {GPQA_SPLITS}")
    elif args.dataset == "scienceqa":
        if args.subset is None:
            args.subset = "all"
        if args.split is None:
            args.split = "test"
        if args.subset not in SCIENCEQA_SUBSETS:
            raise ValueError(f"Invalid ScienceQA subset '{args.subset}'. Choose from: {SCIENCEQA_SUBSETS}")
        if args.split not in SCIENCEQA_SPLITS:
            raise ValueError(f"Invalid ScienceQA split '{args.split}'. Choose from: {SCIENCEQA_SPLITS}")
    elif args.dataset == "cos_e":
        if args.subset is None:
            args.subset = None  # CoS-E has no subsets
        if args.split is None:
            args.split = "validation"
    elif args.dataset == "medmcqa":
        if args.subset is None:
            args.subset = None  # MedMCQA has no subsets
        if args.split is None:
            args.split = "validation"  # test labels are hidden


def main():
    args = parse_args()
    t0 = time.time()
    _apply_defaults_and_validate(args)
    # Allow HF model IDs (e.g. "Qwen/Qwen2.5-7B-Instruct") — vLLM resolves them
    if "/" in args.model_name_or_path and not os.path.exists(args.model_name_or_path):
        pass  # Assume it's a HuggingFace model ID
    else:
        assert os.path.exists(args.model_name_or_path), f"Model {args.model_name_or_path} does not exist"
    # Auto-set max_new_tokens for letter mode
    if args.structured_output == "letter" and args.max_new_tokens == 512:
        args.max_new_tokens = 1
        print(f"[{args.dataset}] Letter mode: auto-setting max_new_tokens=1")


    # Load dataset
    print(f"Loading {args.dataset} dataset (subset={args.subset}, split={args.split})...")
    if args.dataset == "gpqa":
        problems = load_gpqa(args.subset, args.max_samples, split=args.split)
    elif args.dataset == "cos_e":
        problems = load_cos_e(args.split, args.max_samples)
    elif args.dataset == "medmcqa":
        problems = load_medmcqa(args.split, args.max_samples)
    else:
        problems = load_scienceqa(args.split, args.max_samples, subset=args.subset)
    print(f"Loaded {len(problems)} problems")

    # Build prompts from dataset
    prompts_raw = []
    for p in problems:
        context = None
        if args.dataset == "scienceqa" and args.include_lecture and p.get("lecture"):
            context = p["lecture"]
        prompts_raw.append(build_mcqa_prompt(p, structured_output=args.structured_output, context=context))

    # Model basename for output paths
    _clean_path = args.model_name_or_path.rstrip("/")
    if _clean_path.endswith("_merged"):
        _clean_path = os.path.dirname(_clean_path)
    if "checkpoint-" in os.path.basename(_clean_path):
        model_basename = os.path.basename(os.path.dirname(_clean_path))
    else:
        model_basename = os.path.basename(_clean_path)

    # --- Generation phase: load from cache or run vLLM ---
    task_ids = [p["task_id"] for p in problems]
    vllm_used = False

    # Auto-merge LoRA checkpoint (needed for tokenizer chat template; reused by vLLM).
    model_path = prepare_model_for_vllm(
        args.model_name_or_path,
        force_remerge=getattr(args, "force_remerge", False),
    )

    if args.ft_generations_path:
        # Load FT completions from cached JSONL — no vLLM needed.
        generated_texts = load_generations_by_task_id(
            args.ft_generations_path, task_ids,
        )
        print(f"Loaded {len(generated_texts)} FT generations from cache: {args.ft_generations_path}")
    else:
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

    # Apply chat template (needed for both cached and fresh generation paths)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if args.disable_thinking_mode:
        disable_thinking_mode(tokenizer)
    prompts = []
    for p in prompts_raw:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted)
    print(f"Sample prompt (first):\n{prompts[0][:500]}...")

    
    if vllm_used:
        # Build sampling params — guided decoding for letter mode
        sampling_kwargs = dict(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        if args.structured_output == "letter":
            letter_ids = list({
                tid
                for letter in "ABCDE"
                for tid in tokenizer.encode(letter, add_special_tokens=False)
            })
            sampling_kwargs["allowed_token_ids"] = letter_ids
            print(f"[{args.dataset}] Letter mode: restricting to token IDs {letter_ids}")

        sampling_params = SamplingParams(**sampling_kwargs)
        print(f"Generating with temperature={args.temperature}, max_tokens={args.max_new_tokens}...")
        outputs = llm.generate(prompts, sampling_params)
        generated_texts = [o.outputs[0].text for o in outputs]

    # --- Retention scoring ---
    retention = None
    if args.compute_retention:
        # Check if existing results JSONL already has retention scores
        results_path = str(get_eval_results_path(
            args.output_dir, args.mode, args.dataset, args.model_name_or_path,
            args.split, "jsonl", subset=args.subset, hparam_tag=args.hparam_tag,
            suffix=args.suffix,
        ))
        retention = load_retention_from_results(results_path)

        if retention is None:
            if vllm_used:
                print("[retention] Releasing vLLM before loading HF model(s)...")
                del llm
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                vllm_used = False

            base_completions = None
            if args.base_generations_path:
                base_completions = load_generations_by_task_id(
                    args.base_generations_path, task_ids, strict=False,
                )
                print(
                    f"[retention] Loaded {len(base_completions)} base completions "
                    f"from {args.base_generations_path}"
                )

            retention = run_all_retention_passes(
                args, prompts, generated_texts, base_completions,
            )

    # Evaluate
    print(f"Evaluating {args.dataset} answers...")
    results = []

    for i, (problem, gen_text) in enumerate(zip(problems, generated_texts)):
        extracted = extract_answer(
            gen_text,
            structured_output=args.structured_output,
            num_choices=problem.get("num_choices", 4),
        )
        correct = extracted == problem["correct_letter"] if extracted else False

        record = {
            "task_id": problem["task_id"],
            "question": problem["question"][:200],
            "correct_letter": problem["correct_letter"],
            "extracted_answer": extracted,
            "correct": correct,
            "generated_text": gen_text,
            "domain": problem.get("domain", ""),
            "subdomain": problem.get("subdomain", ""),
            "num_choices": problem.get("num_choices", 4),
        }
        if retention is not None:
            flatten_retention_into_record(record, retention, i)
        results.append(record)

        if (i + 1) % 50 == 0:
            running_acc = compute_accuracy(results)
            print(f"  [{i+1}/{len(problems)}] running accuracy={running_acc:.3f}")

    # Compute metrics
    accuracy = compute_accuracy(results)
    extraction_rate = compute_extraction_success_rate(results)
    domain_acc = compute_per_domain_accuracy(results)

    print(f"\n{'=' * 60}")
    print(f"Results: {args.dataset} {args.subset} {args.split} | {model_basename} | mode={args.structured_output}")
    print(f"  accuracy: {accuracy:.4f} ({sum(r['correct'] for r in results)}/{len(results)})")
    print(f"  extraction_success_rate: {extraction_rate:.4f}")
    print(f"\n  Per-domain accuracy:")
    for domain, acc in domain_acc.items():
        n_domain = sum(1 for r in results if r.get("domain") == domain)
        print(f"    {domain}: {acc:.4f} ({int(acc * n_domain)}/{n_domain})")

    # Per-subdomain metrics (if any results have subdomain)
    has_subdomain = any(r.get("subdomain") for r in results)
    subdomain_acc = {}
    if has_subdomain:
        subdomain_correct = defaultdict(list)
        for r in results:
            subdomain_correct[r.get("subdomain", "unknown")].append(r["correct"])
        subdomain_acc = {
            sd: sum(vals) / len(vals)
            for sd, vals in sorted(subdomain_correct.items())
        }
        print(f"\n  Per-topic accuracy (top 10):")
        sorted_topics = sorted(subdomain_acc.items(), key=lambda x: -len(subdomain_correct[x[0]]))
        for topic, acc in sorted_topics[:10]:
            n_topic = len(subdomain_correct[topic])
            print(f"    {topic}: {acc:.4f} ({int(acc * n_topic)}/{n_topic})")

    # Write outputs
    jsonl_path = get_eval_results_path(
        args.output_dir, args.mode, args.dataset, args.model_name_or_path,
        args.split, "jsonl", subset=args.subset, hparam_tag=args.hparam_tag,
        suffix=args.suffix,
    )
    os.makedirs(jsonl_path.parent, exist_ok=True)

    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
        summary = {
            "__summary__": True,
            "dataset": args.dataset,
            "subset": args.subset,
            "split": args.split,
            "model": args.model_name_or_path,
            "model_basename": model_basename,
            "structured_output": args.structured_output,
            "total": len(results),
            "accuracy": accuracy,
            "extraction_success_rate": extraction_rate,
            "per_domain_accuracy": domain_acc,
        }
        if subdomain_acc:
            summary["per_subdomain_accuracy"] = subdomain_acc
        if retention is not None:
            add_retention_to_summary(summary, retention)
        f.write(json.dumps(summary) + "\n")

    print(f"JSONL written to: {jsonl_path}")


    try:
        xlsx_path = get_eval_results_path(
            args.output_dir, args.mode, args.dataset, args.model_name_or_path,
            args.split, "xlsx", subset=args.subset, hparam_tag=args.hparam_tag,
            suffix=args.suffix,
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
            k: summary[k] for k in RETENTION_SUMMARY_KEYS if k in summary
        }
        record = compute_resource_consumption_record(
            method=getattr(args, "mode", "eval"),
            model=getattr(args, "model_name_or_path", "unknown"),
            dataset=getattr(args, "dataset", "unknown"),
            phase="eval",
            wall_time_sec=time.time() - t0,
            num_gpus=detect_num_gpus(),
            peak_gpu_mem_gb=peak_gb,
            subset=getattr(args, "subset", None),
            split=getattr(args, "split", None),
            max_samples=getattr(args, "max_samples", None),
            max_new_tokens=getattr(args, "max_new_tokens", None),
            score=locals().get("accuracy"),
            **retention_extras,
        )
        append_resource_record(record)
    except Exception as e:
        print(f"[resource_consumption] WARNING: failed to log record: {e}")


if __name__ == "__main__":
    main()
