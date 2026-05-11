#!/usr/bin/env python3
"""Generation diagnostics: length stats, EOS rate, Action detection across a grid of models × max_tokens.

Usage (grid mode):
    python scripts/generation_diagnostic.py \
        --models Qwen/Qwen2.5-0.5B-Instruct outputs/E4/context_disagreement_sanity/checkpoint-910 \
        --max_tokens_list 256 512 \
        --temperature 0.7 --repetition_penalty 1.2 --gpu 1

Usage (single model, backward-compatible):
    python scripts/generation_diagnostic.py \
        --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
        --max_tokens 256 --temperature 0.7 --repetition_penalty 1.2 --gpu 1
"""
import argparse
import json
import re
import os
import statistics


def parse_tool_calls(text):
    """Extract (Action, Action_Input) pairs — copied from eval/eval_tooluse.py."""
    pattern = r"Action\s*\d*\s*:\s*(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:\s*(.*?)(?=\nThought|\nAction\s*\d*\s*:|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    calls = []
    for action, action_input in matches:
        calls.append({"Action": action.strip(), "Action_Input": action_input.strip().strip('"')})
    if not calls:
        pattern_single = r"Action\s*\d*\s*:\s*(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:\s*(.*)"
        match = re.search(pattern_single, text, re.DOTALL)
        if match:
            calls.append({"Action": match.group(1).strip(), "Action_Input": match.group(2).strip().strip('"')})
    return calls


def analyze_outputs(outputs, tokenizer):
    """Analyze a batch of vLLM outputs. Returns dict of aggregate stats."""
    lengths = []
    eos_count = 0
    eos_by_token = 0
    has_action = 0
    has_parsed_call = 0
    has_complete_call = 0  # parsed call with valid JSON Action_Input
    finish_reasons = {}

    eos_token_id = tokenizer.eos_token_id

    for out in outputs:
        o = out.outputs[0]
        text = o.text
        token_ids = list(o.token_ids)
        length = len(token_ids)
        lengths.append(length)

        # finish_reason distribution
        fr = str(getattr(o, "finish_reason", "unknown"))
        finish_reasons[fr] = finish_reasons.get(fr, 0) + 1

        # EOS detection: primary = finish_reason, secondary = token check
        if fr == "stop":
            eos_count += 1
            eos_by_token += 1
        elif eos_token_id is not None and token_ids and token_ids[-1] == eos_token_id:
            eos_by_token += 1
            eos_count += 1

        # Action line detection
        if "Action:" in text or re.search(r"Action\s*\d*\s*:", text):
            has_action += 1

        # Complete call detection: at least one parsed call
        calls = parse_tool_calls(text)
        if calls:
            has_parsed_call += 1
            # Check if at least one call has valid JSON Action_Input
            for c in calls:
                try:
                    json.loads(c["Action_Input"])
                    has_complete_call += 1
                    break
                except (json.JSONDecodeError, TypeError):
                    continue

    n = len(outputs)
    return {
        "n": n,
        "lengths": lengths,
        "eos_count": eos_count,
        "eos_by_token": eos_by_token,
        "has_action": has_action,
        "has_parsed_call": has_parsed_call,
        "has_complete_call": has_complete_call,
        "finish_reasons": finish_reasons,
    }


def print_single_report(model_label, max_tokens, stats):
    """Print detailed report for a single (model, max_tokens) combo."""
    n = stats["n"]
    lengths = stats["lengths"]

    print(f"\n{'─'*70}")
    print(f"  {model_label}  |  max_tokens={max_tokens}")
    print(f"{'─'*70}")

    print(f"\n  Completion Length (tokens):")
    print(f"    min:    {min(lengths)}")
    print(f"    median: {statistics.median(lengths):.0f}")
    print(f"    mean:   {statistics.mean(lengths):.1f}")
    print(f"    max:    {max(lengths)}")

    print(f"\n  EOS / Truncation:")
    print(f"    % EOS reached:                 {100*stats['eos_count']/n:.1f}%  ({stats['eos_count']}/{n})")
    print(f"    % truncated (hit max_tokens):  {100*(n-stats['eos_count'])/n:.1f}%  ({n-stats['eos_count']}/{n})")

    print(f"\n  finish_reason distribution:")
    for reason, count in sorted(stats["finish_reasons"].items(), key=lambda x: -x[1]):
        print(f"    {reason:20s}  {count:4d}  ({100*count/n:.1f}%)")

    print(f"\n  Content Quality:")
    print(f"    % with Action: line:            {100*stats['has_action']/n:.1f}%  ({stats['has_action']}/{n})")
    print(f"    % with parsed call (>=1):        {100*stats['has_parsed_call']/n:.1f}%  ({stats['has_parsed_call']}/{n})")
    print(f"    % with complete call (valid JSON):{100*stats['has_complete_call']/n:.1f}%  ({stats['has_complete_call']}/{n})")

    # Length histogram
    buckets = [0, 50, 100, 150, 200, 250, 256, 300, 400, 512, 768, 1024, 99999]
    print(f"\n  Length histogram:")
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        count = sum(1 for l in lengths if lo <= l < hi)
        if count > 0:
            bar = "#" * max(1, count * 40 // n)
            label = f"{lo}-{hi-1}" if hi < 99999 else f"{lo}+"
            print(f"    {label:>10s}:  {count:3d}  ({100*count/n:.1f}%)  {bar}")


def print_consolidated_table(all_results):
    """Print consolidated table across all (model, max_tokens) combos."""
    print(f"\n{'='*120}")
    print(f"  CONSOLIDATED GENERATION DIAGNOSTICS")
    print(f"{'='*120}")

    header = (f"{'Model':>40s} | {'max_tk':>6s} | {'%EOS':>5s} | {'%trunc':>6s} | "
              f"{'%parsed>=1':>10s} | {'%complete':>9s} | {'%Action':>7s} | "
              f"{'len_min':>7s} | {'len_med':>7s} | {'len_mean':>8s} | {'len_max':>7s}")
    print(f"\n{header}")
    print(f"{'-'*len(header)}")

    for model_label, max_tokens, stats in all_results:
        n = stats["n"]
        lengths = stats["lengths"]
        eos_pct = 100 * stats["eos_count"] / n
        trunc_pct = 100 * (n - stats["eos_count"]) / n
        parsed_pct = 100 * stats["has_parsed_call"] / n
        complete_pct = 100 * stats["has_complete_call"] / n
        action_pct = 100 * stats["has_action"] / n
        print(f"{model_label:>40s} | {max_tokens:>6d} | {eos_pct:>5.1f} | {trunc_pct:>6.1f} | "
              f"{parsed_pct:>10.1f} | {complete_pct:>9.1f} | {action_pct:>7.1f} | "
              f"{min(lengths):>7d} | {statistics.median(lengths):>7.0f} | "
              f"{statistics.mean(lengths):>8.1f} | {max(lengths):>7d}")

    # finish_reason breakdown
    print(f"\n  finish_reason breakdown:")
    fr_header = f"{'Model':>40s} | {'max_tk':>6s} | {'stop':>8s} | {'length':>8s} | {'other':>8s}"
    print(f"  {fr_header}")
    print(f"  {'-'*len(fr_header)}")
    for model_label, max_tokens, stats in all_results:
        n = stats["n"]
        fr = stats["finish_reasons"]
        stop_pct = 100 * fr.get("stop", 0) / n
        length_pct = 100 * fr.get("length", 0) / n
        other = n - fr.get("stop", 0) - fr.get("length", 0)
        other_pct = 100 * other / n
        print(f"  {model_label:>40s} | {max_tokens:>6d} | {stop_pct:>7.1f}% | {length_pct:>7.1f}% | {other_pct:>7.1f}%")


def main():
    from src.utils.lora_utils import prepare_model_for_vllm
    from src.utils.env_utils import get_model_dtype

    parser = argparse.ArgumentParser()
    # Grid mode args
    parser.add_argument("--models", type=str, nargs="+", help="Model paths (grid mode)")
    parser.add_argument("--max_tokens_list", type=int, nargs="+", help="Max token values (grid mode)")
    # Single-model backward-compatible args
    parser.add_argument("--model_name_or_path", type=str, help="Single model path (backward compat)")
    parser.add_argument("--max_tokens", type=int, default=256, help="Single max_tokens (backward compat)")
    # Common args
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.3)
    args = parser.parse_args()

    # Resolve model list and max_tokens list
    if args.models:
        model_paths = args.models
    elif args.model_name_or_path:
        model_paths = [args.model_name_or_path]
    else:
        parser.error("Must specify --models or --model_name_or_path")

    if args.max_tokens_list:
        max_tokens_list = args.max_tokens_list
    else:
        max_tokens_list = [args.max_tokens]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # Load eval data once from HuggingFace dataset
    from datasets import load_dataset
    eval_data = load_dataset("Ahren09/ToolAlpaca", split="test").to_list()
    print(f"Loaded {len(eval_data)} eval examples from Ahren09/ToolAlpaca [test split]")

    all_results = []  # list of (model_label, max_tokens, stats)

    for model_path in model_paths:
        model_label = model_path.rstrip("/").split("/")[-1]
        resolved_path = prepare_model_for_vllm(model_path)
        print(f"\n{'='*70}")
        print(f"  Loading model: {model_label}")
        print(f"  Path: {model_path}" + (f" (merged: {resolved_path})" if resolved_path != model_path else ""))
        print(f"{'='*70}")

        # Load tokenizer + model once per model
        tokenizer = AutoTokenizer.from_pretrained(resolved_path)
        prompts = []
        for ex in eval_data:
            messages = [{"role": "user", "content": ex["prompt"]}]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        # Use the largest max_tokens in the list for max_model_len
        max_max_tokens = max(max_tokens_list)
        llm = LLM(model=resolved_path, gpu_memory_utilization=args.gpu_memory_utilization,
                   max_model_len=1024 + max_max_tokens, dtype=get_model_dtype())

        # Inner loop: generate for each max_tokens setting
        for max_tokens in max_tokens_list:
            print(f"\n  Generating with max_tokens={max_tokens}...")
            sampling_params = SamplingParams(
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                max_tokens=max_tokens,
            )
            outputs = llm.generate(prompts, sampling_params)
            stats = analyze_outputs(outputs, tokenizer)
            print_single_report(model_label, max_tokens, stats)
            all_results.append((model_label, max_tokens, stats))

        # Free model memory before loading next
        del llm
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Consolidated table at the end
    if len(all_results) > 1:
        print_consolidated_table(all_results)


if __name__ == "__main__":
    main()
