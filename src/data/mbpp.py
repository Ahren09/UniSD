"""MBPP dataset adapter with normalized keys."""

import json
import os
import re
import torch
from typing import List, Dict, Any

from src.data.retrieval import compute_or_load_embeddings, retrieve_topk_donors, get_embedding_text
from datasets import Dataset, load_dataset
from src.prompts.codegen import build_code_generation_prompt
from src.const import *
from src.data.teacher_text import build_teacher_text
from src.teacher.auxiliary_context import build_induction_auxiliary_context


def extract_function_name_from_code(code: str) -> str | None:
    """Extract def function name from reference code."""
    m = re.search(r'def\s+([a-zA-Z_]\w*)\s*\(', code)
    return m.group(1) if m else None


def load_instructions_from_cache(
    cache_file: str, N: int, num_contexts: int,
) -> list[str]:
    """Load induced instructions from a pre-generated cache file.

    Raises FileNotFoundError if cache doesn't exist.
    Returns a flat list of instruction strings (at least num_contexts long).
    """
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"Induction cache file not found: {cache_file}\n"
            "Run `python -m src.teacher.instruction_induction` first to generate the cache."
        )
    with open(cache_file) as f:
        instructions = json.load(f)

    if not isinstance(instructions, list):
        raise ValueError(f"Expected list in cache, got {type(instructions)}")

    if len(instructions) < num_contexts:
        raise ValueError(
            f"Cache has {len(instructions)} instructions but "
            f"num_contexts={num_contexts} requested."
        )
    # Extract instruction string from dicts if structured format
    instructions = [
        entry["instruction"] if isinstance(entry, dict) else entry
        for entry in instructions
    ]

    return instructions



def extract_function_name_from_tests(test_list: list[str]) -> str | None:
    """Extract expected function name from test assertions.
    Handles: assert func(...), assert math.isclose(func(...)),
    assert set(func(...)), assert sorted(func(...)), etc.
    """
    WRAPPERS = {'math', 'set', 'sorted', 'len', 'list', 'tuple', 'str',
                'int', 'float', 'type', 'isinstance', 'round', 'max', 'min'}
    for test in test_list:
        # Direct: assert func_name(
        m = re.search(r'assert\s+([a-zA-Z_]\w*)\s*\(', test)
        if m and m.group(1) not in WRAPPERS:
            return m.group(1)
        # Nested: assert wrapper(func_name(  or  assert math.isclose(func_name(
        m = re.search(r'assert\s+\w+(?:\.\w+)?\s*\(\s*([a-zA-Z_]\w*)\s*\(', test)
        if m and m.group(1) not in WRAPPERS:
            return m.group(1)
    return None


def load_mbpp(split: str = "test", max_samples: int | None = None, config: str = "sanitized") -> list[dict]:
    """Load MBPP dataset and return list of dicts with normalized keys.

    Returns dicts with keys:
        task_id, prompt, code, test_imports_or_setup, test_list
    Handles both 'sanitized' (fields: prompt, test_imports) and 'full' (fields: text, test_setup_code).
    """
    ds = load_dataset("google-research-datasets/mbpp", config, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    results = []
    for ex in ds:
        # prompt: sanitized uses 'prompt', full uses 'text'
        prompt = ex.get("prompt") or ex.get("text", "")

        # test_imports_or_setup: sanitized uses 'test_imports' (list), full uses 'test_setup_code' (str)
        if "test_imports" in ex:
            test_imports_or_setup = ex["test_imports"]
        elif "test_setup_code" in ex:
            setup = ex["test_setup_code"]
            test_imports_or_setup = setup if setup else ""
        else:
            test_imports_or_setup = ""

        code_fn = extract_function_name_from_code(ex["code"])
        test_fn = extract_function_name_from_tests(ex["test_list"])

        results.append({
            "task_id": ex["task_id"],
            "prompt": prompt,
            "code": ex["code"],
            "test_imports_or_setup": test_imports_or_setup,
            "test_list": ex["test_list"],
            "code_function_name": code_fn,
            "test_function_name": test_fn,
        })

    # Diagnostic summary
    n_test_fn = sum(1 for r in results if r["test_function_name"])
    n_code_fn = sum(1 for r in results if r["code_function_name"])
    mismatches = [
        r for r in results
        if r["test_function_name"] and r["code_function_name"]
        and r["test_function_name"] != r["code_function_name"]
    ]
    mismatch_strs = [
        f"task_id={r['task_id']} (code={r['code_function_name']}, test={r['test_function_name']})"
        for r in mismatches
    ]
    print(f"[MBPP] Loaded {len(results)} problems: "
          f"{n_test_fn} test_fn found, {n_code_fn} code_fn found, "
          f"{len(mismatches)} mismatches")
    if mismatches:
        print(f"[MBPP] Mismatches: {', '.join(mismatch_strs)}")

    return results


def _fix_function_name_in_code(code: str, code_fn: str | None, test_fn: str | None) -> str:
    """Rewrite the def line in reference code to match test_function_name if they differ.

    This avoids NameError at eval time when tests call a different function name
    than the reference solution defines.
    """
    if not test_fn or not code_fn or test_fn == code_fn:
        return code
    # Replace the first occurrence of `def code_fn(` with `def test_fn(`
    return re.sub(
        rf'\bdef\s+{re.escape(code_fn)}\s*\(',
        f'def {test_fn}(',
        code,
        count=1,
    )




def build_fewshot_auxiliary_context(
    few_shot_samples: List[Dict[str, Any]],
    current_prompt: str,
) -> str:
    """
    Build an augmented teacher prompt by placing reference (few-shot) solved examples
    BEFORE the actual task prompt, with strong boundaries to avoid task confusion.

    IMPORTANT:
    - Reference examples are explicitly marked as NOT the task to solve.
    - Each reference solution is fenced in ```python for consistency.
    - The actual task (current_prompt) comes LAST and remains unchanged
      (so existing demo markers like drop_demo still work).
    """
    if not few_shot_samples:
        return current_prompt

    header = (
        "## Reference Samples (DO NOT SOLVE THESE)\n"
        "The following are solved examples from OTHER tasks, provided only as reference.\n"
        "They are NOT the task you should solve now.\n"
        "Do NOT copy their function names. Do NOT mention these examples in your final answer.\n\n"
    )

    refs: List[str] = []
    for idx, s in enumerate(few_shot_samples, 1):
        ref_task = s.get("prompt", "").strip()
        ref_code = s.get("code", "").strip()
        
        ref_test_cases = '\n'.join(s.get("test_list", []))

        # Use safer labels than "Problem/Solution" to reduce instruction confusion.
        refs.append(f"""
### [REFERENCE ONLY #{idx}]
Reference Task
{ref_task}

Reference Test Cases
{ref_test_cases}

Reference Solution:
```python
{ref_code}
```
""")

    separator = """
-----
## Task to Solve (ONLY ANSWER THIS TASK)
Now solve ONLY the following task. Output ONLY the final code for THIS task.

"""

    prompt = header + "\n".join(refs) + separator + current_prompt

    return prompt




def load_mbpp_training_dataset(
    seed: int = 42,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    config: str = "sanitized",
    num_auxiliary_contexts: int = 0,
    fewshot_method: str | None = None,
    retrieval_model: str = "all-MiniLM-L6-v2",
    embedding_cache_dir: str | None = None,
    demo_k: int = 1,
    induction_cache_file: str | None = None,
):
    """Load MBPP for UniSD training. Returns (train_dataset, eval_dataset) as HF Datasets.

    Uses official train/validation splits (NOT random split).
    Each example has:
      - prompt: chat format [{"role": "user", "content": <codegen_prompt>}]
      - teacher_prompt: chat format [{"role": "user", "content": <codegen_prompt + demo>}]
      - task_id, code_function_name, test_function_name (metadata)

    Optional:
      - If num_auxiliary_contexts > 0, add extra fields:
            fewshot_teacher_prompt_0, ..., fewshot_teacher_prompt_{K-1}
        Each is a chat-format prompt that prepends reference solved examples (few-shot)
        BEFORE the original teacher_prompt, while keeping the original teacher_prompt intact.
    """
    train_problems = load_mbpp("train", max_samples=max_train_samples, config=config)
    eval_problems = load_mbpp("validation", max_samples=max_eval_samples, config=config)

    def _format_problems(problems: list[dict]) -> list[dict]:
        formatted: list[dict] = []
        for p in problems:
            student_text = build_code_generation_prompt(p, "mbpp")

            # Fix function name mismatch: rewrite def line to match test expectations
            fixed_code = _fix_function_name_in_code(
                p["code"], p.get("code_function_name"), p.get("test_function_name")
            )

            teacher_text = build_teacher_text(student_text, fixed_code)

            formatted.append(
                {
                    "prompt": [{"role": "user", "content": student_text}],
                    "teacher_prompt": [{"role": "user", "content": teacher_text}],
                    "task_id": p["task_id"],
                    "code_function_name": p.get("code_function_name"),
                    "test_function_name": p.get("test_function_name"),
                    # Store fixed code for downstream few-shot pool use (optional debug/useful)
                    "_fixed_code": fixed_code,
                }
            )
        return formatted

    train_data = _format_problems(train_problems)
    eval_data = _format_problems(eval_problems)

    # --- Optional few-shot teacher prompts branch (keeps backward compatibility) ---
    if num_auxiliary_contexts > 0 and len(train_data) > 1:
        import random

        # Build a donor pool from the formatted train_data (reuse fixed code and prompt text).
        # Enrich with test_list from train_problems for richer few-shot context.
        donor_pool: list[dict] = []
        for idx, ex in enumerate(train_data):
            donor_pool.append(
                {
                    "prompt": ex["prompt"][0]["content"],
                    "code": ex["_fixed_code"],
                    "test_list": train_problems[idx].get("test_list", []) if idx < len(train_problems) else [],
                }
            )

        if fewshot_method == RETRIEVAL:
            # --- Retrieval-based fewshot ---
            texts = [get_embedding_text(ex) for ex in train_data]
            embeddings = compute_or_load_embeddings(
                texts,
                model_name=retrieval_model,
                cache_path=embedding_cache_dir,
                device="cuda",
            )
            self_indices = torch.arange(len(train_data), device=embeddings.device)
            topk_indices = retrieve_topk_donors(
                embeddings, embeddings,
                k=num_auxiliary_contexts * demo_k,
                self_indices=self_indices,
            )
            topk_list = topk_indices.cpu().tolist()

            # Diagnostic log
            print(f"[MBPP Retrieval] Pool={len(donor_pool)}, embedding_dim={embeddings.shape[1]}, "
                  f"model={retrieval_model}")
            if len(train_data) > 0:
                sample_id = train_data[0].get("task_id", 0)
                sample_retrieved = topk_list[0][:min(3, len(topk_list[0]))]
                print(f"[MBPP Retrieval] Example: task_id={sample_id} -> retrieved {sample_retrieved}")

            for i, example in enumerate(train_data):
                current_student_text = example["prompt"][0]["content"]
                for k in range(num_auxiliary_contexts):
                    demos = [donor_pool[topk_list[i][k * demo_k + d]] for d in range(demo_k)]  # Length: number of demos per teacher
                    augmented = build_fewshot_auxiliary_context(demos, current_student_text)
                    example[f"fewshot_teacher_prompt_{k}"] = [{"role": "user", "content": augmented}]

            print(f"[MBPP Training] Added {num_auxiliary_contexts} retrieval-based fewshot teacher prompts per example")

        elif fewshot_method == INDUCTION:
            # --- Instruction induction fewshot (loads from pre-generated cache) ---
            if not induction_cache_file:
                raise ValueError(
                    f"fewshot_method={INDUCTION!r} requires induction_cache_file. "
                    "Run `python -m src.teacher.instruction_induction` first to generate the cache."
                )
            instructions = load_instructions_from_cache(
                induction_cache_file, len(train_data), num_auxiliary_contexts,
            )
            
            random_induction_indices = random.choices(range(len(instructions)), k=len(train_data) * num_auxiliary_contexts)
            for i, example in enumerate(train_data):
                student_text = example["prompt"][0]["content"]
                for k in range(num_auxiliary_contexts):
                    augmented = build_induction_auxiliary_context(student_text, instructions[random_induction_indices[i * num_auxiliary_contexts + k]])
                    example[f"fewshot_teacher_prompt_{k}"] = [{"role": "user", "content": augmented}]

            print(f"[MBPP Training] Added {num_auxiliary_contexts} induction-based fewshot teacher prompts per example")

        elif fewshot_method == RANDOM:
            # --- Random fewshot (default, original behavior) ---
            rng = random.Random(seed)

            for i, example in enumerate(train_data):
                current_student_text = example["prompt"][0]["content"]

                # Exclude self to avoid leaking the same task as "reference"
                candidate_indices = [j for j in range(len(donor_pool)) if j != i]

                # Prefer sampling without replacement; fallback to with-replacement if pool is too small.
                if len(candidate_indices) >= num_auxiliary_contexts:
                    chosen = rng.sample(candidate_indices, k=num_auxiliary_contexts)
                else:
                    # Very small pool edge case
                    chosen = [rng.choice(candidate_indices) for _ in range(num_auxiliary_contexts)]

                for k, j in enumerate(chosen):
                    few_shot_sample = donor_pool[j]
                    augmented = build_fewshot_auxiliary_context([few_shot_sample], current_student_text)
                    example[f"fewshot_teacher_prompt_{k}"] = [{"role": "user", "content": augmented}]

            print(f"[MBPP Training] Added {num_auxiliary_contexts} random fewshot teacher prompts per example")

        else:
            raise ValueError(f"Unsupported fewshot_method: {fewshot_method}")


    # Keep `_fixed_code` in the final dataset so instruction induction
    # (src/teacher/instruction_induction.py) can read the canonical solution.

    print(f"[MBPP Training] train={len(train_data)}, eval={len(eval_data)}")

    return Dataset.from_list(train_data), Dataset.from_list(eval_data)