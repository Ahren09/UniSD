"""Standalone tool-use evaluation script for trained checkpoints.

Eval data is loaded from the HuggingFace dataset Ahren09/ToolAlpaca (test split).

Usage (colocate mode — default):
    python -m eval.eval_tooluse \
        --model_name_or_path <checkpoint_dir> \
        --max_new_tokens 1024 \
        --output_dir outputs

Usage (server mode — requires running vLLM server):
    python -m eval.eval_tooluse \
        --model_name_or_path <checkpoint_dir> \
        --vllm_mode server \
        --vllm_server_url http://localhost:2507
"""

import json
import re
import os
import time
from typing import Any

from src.utils.env_utils import get_model_dtype, load_project_env
load_project_env()  # must run before vllm import (sets VLLM_ATTENTION_BACKEND, VLLM_USE_V1, etc.)

from pydantic import BaseModel
from transformers import AutoTokenizer
from openai import OpenAI
from vllm import LLM, SamplingParams
from openpyxl import Workbook

from .eval_args import parse_args
from src.utils.lora_utils import prepare_model_for_vllm
from src.utils.tooluse_utils import (
    compare_action_inputs,
    compare_action_names,
    compute_argument_accuracy,
    parse_tool_calls,
    rewrite_prompt_for_structured,
)
from src.analysis.resource_consumption_utils import (
    append_resource_record,
    compute_resource_consumption_record,
    detect_num_gpus,
)
from src.utils.path_utils import get_eval_results_path

class ToolCall(BaseModel):
    Thought: str
    Action: str
    Action_Input: dict[str, Any]


class ToolCallResponse(BaseModel):
    tool_calls: list[ToolCall]


def parse_generated_text(generated_text, structured_output):
    """Parse a single generated text into predicted_calls list.
    Returns (predicted_calls, success) tuple."""
    if structured_output:
        try:
            parsed = json.loads(generated_text)
            predicted_calls = [
                {"Action": tc["Action"],
                 "Action_Input": json.dumps(tc["Action_Input"], ensure_ascii=False)}
                for tc in parsed.get("tool_calls", [])
            ]
        except (json.JSONDecodeError, TypeError):
            predicted_calls = []
        # Structured mode fallback: try regex if JSON parse gave nothing
        if not predicted_calls:
            predicted_calls = parse_tool_calls(generated_text)
    else:
        predicted_calls = parse_tool_calls(generated_text)
    success = len(predicted_calls) > 0
    return predicted_calls, success


def main():
    args = parse_args()
    t0 = time.time()
    assert args.dataset == "tooluse", "Dataset must be tooluse"
    
    model_name = os.path.basename(args.model_name_or_path.rstrip("/"))
    
    args.output_path = get_eval_results_path(args.output_dir, args.mode, args.dataset, args.model_name_or_path, args.split, "jsonl", hparam_tag=getattr(args, "hparam_tag", ""), suffix=getattr(args, "suffix", ""))
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Auto-merge LoRA checkpoint if needed (colocate mode only)
    if args.vllm_mode == "colocate":
        model_path = prepare_model_for_vllm(args.model_name_or_path, force_remerge=getattr(args, "force_remerge", False))
    else:
        model_path = args.model_name_or_path

    # Load eval data from HuggingFace dataset
    from datasets import load_dataset
    eval_data = load_dataset("Ahren09/ToolAlpaca", split="test").to_list()
    print(f"Loaded {len(eval_data)} eval examples for ToolAlpaca [test split]")
    
    if args.max_samples is not None:
        eval_data = eval_data[:args.max_samples]

    # Rewrite prompts for structured output mode
    if args.structured_output:
        json_instruction = (
            'Return your answer as a JSON object with this exact structure:\n'
            '{"tool_calls": [{"Action": "<tool_name>", '
            '"Action_Input": {"<param>": "<value>", ...}}]}\n'
            'Output ONLY the JSON object, no other text.\n\nBegin!'
        )
        for i, ex in enumerate(eval_data):
            new_prompt, _ = rewrite_prompt_for_structured(ex["prompt"], json_instruction)
            eval_data[i] = {**ex, "prompt": new_prompt}

    # Prepare prompts
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = []
    for example in eval_data:
        messages = [{"role": "user", "content": example["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    # Log prompt token lengths for truncation diagnostics
    tokenized = tokenizer(prompts, add_special_tokens=False).input_ids
    token_lengths = [len(ids) for ids in tokenized]
    print(f"Prompt token lengths: min={min(token_lengths)}, max={max(token_lengths)}, "
          f"mean={sum(token_lengths)/len(token_lengths):.0f}")
    n_over = sum(1 for l in token_lengths if l > args.max_prompt_length)
    if n_over > 0:
        print(f"WARNING: {n_over}/{len(token_lengths)} prompts exceed max_prompt_length={args.max_prompt_length}")

    # Generate completions via vLLM
    if args.vllm_mode == "colocate":
        print(f"Loading model with vLLM (colocate, tp={args.tensor_parallel_size})...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_prompt_length + args.max_new_tokens,
            dtype=get_model_dtype(),
            enforce_eager=args.enforce_eager,
            trust_remote_code=True,
        )
        if args.structured_output:
            from vllm.sampling_params import StructuredOutputsParams
            structured_params = StructuredOutputsParams(json=ToolCallResponse.model_json_schema())
            sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                structured_outputs=structured_params,
            )
        else:
            sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
            )
        outputs = llm.generate(prompts, sampling_params)
        generated_texts = [o.outputs[0].text for o in outputs]
    else:  # server mode
        if args.structured_output:
            print("WARNING: --structured_output is only supported in colocate mode. Ignoring flag.")
            args.structured_output = False
        print(f"Connecting to vLLM server at {args.vllm_server_url}...")
        client = OpenAI(base_url=args.vllm_server_url + "/v1", api_key="dummy")
        response = client.completions.create(
            model=args.model_name_or_path,
            prompt=prompts,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        generated_texts = [c.text for c in response.choices]

    print(f"Generated {len(generated_texts)} completions")

    # First pass: parse all generated texts
    parsed_results = {}  # idx -> (generated_text, predicted_calls)
    retry_indices = []

    for idx, (example, generated_text) in enumerate(zip(eval_data, generated_texts)):
        predicted_calls, success = parse_generated_text(generated_text, args.structured_output)
        parsed_results[idx] = (generated_text, predicted_calls)
        if not success:
            retry_indices.append(idx)

    # Snapshot first-pass failures for retry accounting
    first_attempt_failures = len(retry_indices)

    # Retry loop — use higher temperature for diversity on re-generation
    if args.vllm_mode == "colocate":
        retry_temp = max(args.temperature, 0.7)
        if args.structured_output:
            from vllm.sampling_params import StructuredOutputsParams as _SOP
            retry_sampling_params = SamplingParams(
                temperature=retry_temp,
                max_tokens=args.max_new_tokens,
                structured_outputs=_SOP(json=ToolCallResponse.model_json_schema()),
            )
        else:
            retry_sampling_params = SamplingParams(
                temperature=retry_temp,
                max_tokens=args.max_new_tokens,
            )

    for retry_round in range(args.max_retries):
        if not retry_indices:
            break
        print(f"Retry round {retry_round + 1}: re-generating {len(retry_indices)} failed examples...")
        retry_prompts = [prompts[i] for i in retry_indices]

        if args.vllm_mode == "colocate":
            retry_outputs = llm.generate(retry_prompts, retry_sampling_params)
            retry_texts = [o.outputs[0].text for o in retry_outputs]
        else:
            retry_resp = client.completions.create(
                model=args.model_name_or_path,
                prompt=retry_prompts,
                max_tokens=args.max_new_tokens,
                temperature=max(args.temperature, 0.7),
            )
            retry_texts = [c.text for c in retry_resp.choices]

        still_failed = []
        for idx, gen_text in zip(retry_indices, retry_texts):
            predicted_calls, success = parse_generated_text(gen_text, args.structured_output)
            parsed_results[idx] = (gen_text, predicted_calls)
            if not success:
                still_failed.append(idx)
        retry_indices = still_failed

    num_still_failed = len(retry_indices)
    num_recovered = first_attempt_failures - num_still_failed
    retry_temp = max(args.temperature, 0.7) if args.max_retries > 0 else None
    print(f"Retry accounting: {first_attempt_failures} failed first pass, "
          f"{num_recovered} recovered, {num_still_failed} still failed")
    if num_still_failed:
        print(f"WARNING: {num_still_failed} examples still failed after {args.max_retries} retries")

    # Score all results
    results = []
    action_correct = 0
    argument_acc_sum = 0.0
    full_correct = 0

    for idx, example in enumerate(eval_data):
        generated_text, predicted_calls = parsed_results[idx]
        golden_calls = example["golden_answer"]

        action_score = compare_action_names(predicted_calls, golden_calls)
        arg_score = compute_argument_accuracy(predicted_calls, golden_calls)
        full_score = 1.0 if (action_score == 1.0 and arg_score == 1.0) else 0.0

        action_correct += action_score
        argument_acc_sum += arg_score
        full_correct += full_score

        results.append({
            "instruction": example.get("instruction", ""),
            "name": example.get("name", ""),
            "generated_text": generated_text,
            "predicted_calls": predicted_calls,
            "golden_calls": golden_calls,
            "action_correct": action_score,
            "argument_accuracy": arg_score,
            "full_match": full_score,
        })

    n = len(eval_data)
    summary = {
        
        "action_accuracy": action_correct / n,
        "argument_accuracy": argument_acc_sum / n,
        "full_accuracy": full_correct / n,
        "total": n,
        "num_failed_first_pass": first_attempt_failures,
        "num_recovered": num_recovered,
        "num_still_failed": num_still_failed,
        "retry_temperature": retry_temp,
        "structured_output": args.structured_output,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps({"__summary__": True, **summary}, ensure_ascii=False) + "\n")

    # Write summary to xlsx
    xlsx_path = get_eval_results_path(
        args.output_dir, args.mode, args.dataset, args.model_name_or_path,
        args.split, "xlsx", hparam_tag=getattr(args, "hparam_tag", ""),
        suffix=getattr(args, "suffix", ""),
    )
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["Metric", "Value"])
    for key, val in summary.items():
        ws.append([key, val])
    wb.save(xlsx_path)

    print(f"\nResults saved to {args.output_path}")
    print(f"Summary saved to {xlsx_path}")
    print(f"  action_accuracy:   {summary['action_accuracy']:.4f}")
    print(f"  argument_accuracy: {summary['argument_accuracy']:.4f}")
    print(f"  full_accuracy:     {summary['full_accuracy']:.4f}")
    print(f"  total:             {summary['total']}")

    try:
        peak_gb = None
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                peak_gb = _torch.cuda.max_memory_allocated() / (1024**3)
        except Exception:
            pass
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
            score=summary.get("full_accuracy"),
        )
        append_resource_record(record)
    except Exception as e:
        print(f"[resource_consumption] WARNING: failed to log record: {e}")


if __name__ == "__main__":
    main()
