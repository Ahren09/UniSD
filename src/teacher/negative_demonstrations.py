"""Generate hard-negative demonstrations for contrastive self-distillation.

Produces N negative demos per training example using vLLM structured output mode.
Each negative is plausible but semantically wrong — designed as hard negatives for
contrastive learning alongside the standard KL loss.

Usage:
    python -m src.teacher.negative_demonstrations \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --dataset gpqa \
        --max_samples 5
"""

import argparse
import json
import os
from pathlib import Path

from pydantic import BaseModel
from src.utils.env_utils import get_model_dtype  # noqa: F401 — env_utils import patches transformers for custom-code models
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from src.const import GPQA, SCIENCEQA, MBPP, TOOLUSE, COS_E, MEDMCQA
from src.utils.path_utils import get_neg_demo_cache_path

# ---------------------------------------------------------------------------
# Pydantic schema for structured vLLM output
# ---------------------------------------------------------------------------

class NegativeDemonstration(BaseModel):
    reasoning: str
    answer: str


# ---------------------------------------------------------------------------
# Corruption prompt builders
# ---------------------------------------------------------------------------

def build_mcqa_corruption_prompt(problem: dict, wrong_letter: str) -> str:
    """Build a prompt asking the model to argue for a specific wrong MCQA answer."""
    options = problem["options"]
    sorted_keys = sorted(options.keys())
    options_text = "\n".join(f"{k}. {options[k].strip()}" for k in sorted_keys)
    wrong_text = options[wrong_letter].strip()

    prompt = f"""You are simulating a confident student who made a mistake on this question.
Argue convincingly that the answer is {wrong_letter} ({wrong_text}).
Question: {problem['question']}
{options_text}

Rules:
- You MUST conclude that the answer is {wrong_letter}.
- Your reasoning must sound plausible and knowledgeable.
- Use domain-appropriate terminology.
- Do NOT hint that you might be wrong or that another answer could be better.
- Keep reasoning to 2-4 sentences.
"""

    prompt += f"""Return format:
{{
    "reasoning": "BRIEF EXPLANATION",
    "answer": "{wrong_letter}"
}}
"""
    return prompt


def build_code_corruption_prompt(problem: dict, bug_type: str) -> str:
    """Build a prompt asking the model to write subtly buggy code."""
    function_name = problem.get("test_function_name") or problem.get("code_function_name") or "solution"
    test_cases = "\n".join(problem.get("test_list", []))

    prompt = f"""You are simulating a programmer who makes a subtle mistake.
        f"You are simulating a programmer who makes a subtle mistake.\n"
Write a Python solution for this task that LOOKS correct but has a {bug_type}.
Task: {problem['prompt']}
Function name: {function_name}
Test cases (your code should fail at least one):
{test_cases}
Rules:
- Use the correct function name and signature.
- The bug must be subtle (not a syntax error or obvious typo).
- Include necessary imports.
- Return ONLY the function code.
"""
    prompt += f"""Return format: Return ONLY the function code:
```python
def {function_name}:
    ...
```
"""
    return prompt


_TOOLUSE_CORRUPTION_DESCRIPTIONS = {
    "wrong_action": (
        "Call a plausible but INCORRECT function name (not {correct_action}). "
        "Pick a function that sounds related but does the wrong thing."
    ),
    "wrong_args": (
        "Use the correct function '{correct_action}' but with WRONG parameter values "
        "— e.g., wrong URL, wrong method, swapped or missing fields."
    ),
    "wrong_interpretation": (
        "Misunderstand the user's request. Call '{correct_action}' but for a "
        "subtly different purpose than what was asked."
    ),
    "extra_tool_calls": (
        "Include an unnecessary extra tool call in addition to the main one, "
        "as if you're over-interpreting the request."
    ),
}


def _get_correct_action(problem: dict) -> str:
    """Extract the correct action name from golden_answer or golden_response."""
    # Try golden_answer first (list of dicts)
    ga = problem.get("golden_answer")
    if ga and isinstance(ga, list) and len(ga) > 0:
        first = ga[0]
        if isinstance(first, dict):
            return first.get("Action", "unknown_action")
        if isinstance(first, str):
            try:
                parsed = json.loads(first)
                return parsed.get("Action", "unknown_action")
            except (json.JSONDecodeError, TypeError):
                pass
    # Fallback: parse golden_response text
    gr = problem.get("golden_response", "")
    if isinstance(gr, list):
        gr = "\n".join(gr)
    import re
    m = re.search(r"Action\s*\d*\s*:\s*(.+)", gr)
    if m:
        return m.group(1).strip()
    return "unknown_action"


def build_tooluse_corruption_prompt(problem: dict, corruption_type: str) -> str:
    """Build a prompt asking the model to generate a wrong tool call."""
    correct_action = _get_correct_action(problem)
    desc = _TOOLUSE_CORRUPTION_DESCRIPTIONS[corruption_type].format(correct_action=correct_action)

    instruction = problem.get("instruction", "")
    prompt_text = problem.get("prompt", "")

    return (
        f"You are simulating an AI assistant that makes a mistake when using tools.\n"
        f"Given the user's request and API documentation, produce a tool call response "
        f"that contains a {corruption_type}.\n\n"
        f"User instruction: {instruction}\n\n"
        f"API documentation:\n{prompt_text}\n\n"
        f"Corruption type: {desc}\n\n"
        f"Rules:\n"
        f"- Your response must look like a genuine ReAct-style attempt (Thought/Action/Action_Input).\n"
        f"- The mistake should be subtle and plausible.\n"
        f"- Use valid JSON for Action_Input.\n"
        f"- Do NOT mention that you are making a mistake."
    )


# ---------------------------------------------------------------------------
# Assignment helpers
# ---------------------------------------------------------------------------

def assign_wrong_options(correct_letter: str, all_letters: list[str], num_negatives: int = 4) -> list[str]:
    """Assign wrong option letters round-robin for num_negatives negatives."""
    wrong_letters = [l for l in all_letters if l != correct_letter]
    if not wrong_letters:
        raise ValueError(f"No wrong options available (correct={correct_letter}, all={all_letters})")
    return [wrong_letters[i % len(wrong_letters)] for i in range(num_negatives)]


CODE_BUG_TYPES = ["off-by-one error", "wrong operator", "missing edge case", "wrong return value"]
TOOLUSE_CORRUPTION_TYPES = ["wrong_action", "wrong_args", "wrong_interpretation", "extra_tool_calls"]


# ---------------------------------------------------------------------------
# Generation pipeline
# ---------------------------------------------------------------------------

def generate_negative_demos(
    prompts: list[str],
    llm: LLM,
    max_new_tokens: int,
    temperature: float,
    max_retries: int,
) -> list[dict]:
    """Generate negative demonstrations via vLLM with structured output + retry loop.

    Returns a list of dicts with keys: reasoning, answer.
    Failed entries get empty strings.
    """
    structured_params = StructuredOutputsParams(json=NegativeDemonstration.model_json_schema())
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        structured_outputs=structured_params,
    )

    results: list[dict | None] = [None] * len(prompts)
    pending = list(range(len(prompts)))

    for attempt in range(max_retries + 1):
        if not pending:
            break
        batch_prompts = [prompts[i] for i in pending]
        outputs = llm.generate(batch_prompts, sampling_params)

        still_pending = []
        for j, orig_idx in enumerate(pending):
            text = outputs[j].outputs[0].text.strip()
            try:
                parsed = NegativeDemonstration.model_validate_json(text)
                results[orig_idx] = parsed.model_dump()
            except Exception:
                still_pending.append(orig_idx)

        print(f"  [NegDemo] Attempt {attempt + 1}: {len(pending) - len(still_pending)}/{len(pending)} valid")
        pending = still_pending

    # Fallback for anything still failing
    for idx in pending:
        results[idx] = {"reasoning": "", "answer": ""}
    if pending:
        print(f"  [NegDemo] WARNING: {len(pending)} entries failed all retries, using empty fallback")

    return results


def load_or_generate_negative_demos(
    problems: list[dict],
    model_name: str,
    dataset_name: str,
    cache_dir: str,
    num_negatives: int = 4,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    max_retries: int = 2,
    gpu_memory_utilization: float = 0.85,
    seed: int = 42,
    max_samples: int | None = None,
) -> dict[str, list[dict]]:
    """Load cached negative demos or generate them via vLLM.

    Returns dict keyed by task_id, each value is a list of num_negatives dicts.
    """
    
    
    cache_path = get_neg_demo_cache_path(cache_dir, dataset_name, model_name, max_samples)

    # Check cache
    if cache_path.exists():
        print(f"[NegDemo] Loading cached negative demos from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"[NegDemo] Loaded {len(cached)} cached entries")
        return cached

    # Build prompts + metadata
    prompts: list[str] = []
    metadata: list[dict] = []  # track task_id, neg_idx, extra info

    for problem in problems:
        task_id = problem["task_id"]
        # Multiple-choice QA tasks
        if dataset_name in (GPQA, SCIENCEQA, COS_E, MEDMCQA):
            all_letters = sorted(problem["options"].keys())
            wrong_letters = assign_wrong_options(problem["correct_letter"], all_letters, num_negatives)
            for i, wl in enumerate(wrong_letters):
                prompts.append(build_mcqa_corruption_prompt(problem, wl))
                metadata.append({"task_id": task_id, "neg_idx": i, "target_wrong_letter": wl})

        elif dataset_name == MBPP:
            for i in range(num_negatives):
                bug_type = CODE_BUG_TYPES[i % len(CODE_BUG_TYPES)]
                prompts.append(build_code_corruption_prompt(problem, bug_type))
                metadata.append({"task_id": task_id, "neg_idx": i, "strategy": bug_type})

        elif dataset_name == TOOLUSE:
            for i in range(num_negatives):
                corruption_type = TOOLUSE_CORRUPTION_TYPES[i % len(TOOLUSE_CORRUPTION_TYPES)]
                prompts.append(build_tooluse_corruption_prompt(problem, corruption_type))
                metadata.append({"task_id": task_id, "neg_idx": i, "strategy": corruption_type})

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    print(f"[NegDemo] Built {len(prompts)} prompts for {len(problems)} problems × {num_negatives} negatives")

    # Apply chat template
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted)

    print(f"[NegDemo] Sample prompt (first):\n{formatted_prompts[0][:500]}...")

    # vLLM generation
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        dtype=get_model_dtype(),
        trust_remote_code=True,
    )

    results = generate_negative_demos(
        formatted_prompts, llm,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        max_retries=max_retries,
    )

    # Reshape flat list -> dict keyed by task_id
    output: dict[str, list[dict]] = {}
    for result, meta in zip(results, metadata):
        task_id = meta["task_id"]
        entry = {**result}
        # Attach metadata
        for key in ("target_wrong_letter", "strategy"):
            if key in meta:
                entry[key] = meta[key]

        if task_id not in output:
            output[task_id] = []
        output[task_id].append(entry)

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[NegDemo] Saved {len(output)} entries to {cache_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate negative demonstrations for contrastive training")
    parser.add_argument("--model_name", type=str, required=True, help="HF model name or path")
    parser.add_argument("--dataset", type=str, required=True, choices=[GPQA, SCIENCEQA, MBPP, TOOLUSE, COS_E, MEDMCQA])
    parser.add_argument("--num_negatives", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default="outputs/cache/demonstration")
    # Dataset-specific
    parser.add_argument("--gpqa_subset", type=str, default="gpqa_extended")
    parser.add_argument("--scienceqa_subset", type=str, default="all")
    parser.add_argument("--mbpp_config", type=str, default="sanitized")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    if args.dataset == GPQA:
        from src.data.gpqa import load_gpqa
        problems = load_gpqa(
            subset=args.gpqa_subset,
            seed=args.seed,
            split=args.split,
            max_samples=args.max_samples,
        )

    elif args.dataset == SCIENCEQA:
        from src.data.scienceqa import load_scienceqa
        problems = load_scienceqa(
            split=args.split,
            seed=args.seed,
            max_samples=args.max_samples,
        )

    elif args.dataset == MBPP:
        from src.data.mbpp import load_mbpp
        problems = load_mbpp(
            split=args.split,
            max_samples=args.max_samples,
            config=args.mbpp_config,
        )

    elif args.dataset == COS_E:
        from src.data.cos_e import load_cos_e
        problems = load_cos_e(
            split=args.split,
            seed=args.seed,
            max_samples=args.max_samples,
        )

    elif args.dataset == MEDMCQA:
        from src.data.medmcqa import load_medmcqa
        problems = load_medmcqa(
            split=args.split,
            seed=args.seed,
            max_samples=args.max_samples,
        )

    elif args.dataset == TOOLUSE:
        from datasets import load_dataset
        raw_data = load_dataset("Ahren09/ToolAlpaca", split="train").to_list()
        # Add task_id if missing
        for i, ex in enumerate(raw_data):
            if "task_id" not in ex:
                ex["task_id"] = f"tooluse_{i}"
        if args.max_samples is not None:
            raw_data = raw_data[:args.max_samples]
        problems = raw_data

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(f"[NegDemo] Loaded {len(problems)} problems from {args.dataset}")

    # Generate or load from cache
    output = load_or_generate_negative_demos(
        problems=problems,
        model_name=args.model_name,
        dataset_name=args.dataset,
        cache_dir=args.cache_dir,
        num_negatives=args.num_negatives,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        max_samples=args.max_samples,
    )

    # Summary
    total_demos = sum(len(v) for v in output.values())
    non_empty = sum(1 for v in output.values() for d in v if d.get("reasoning"))
    print(f"\n[NegDemo] Done: {total_demos} negative demos for {len(output)} problems "
          f"({non_empty} non-empty)")


if __name__ == "__main__":
    main()
