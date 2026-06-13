"""GSM8K dataset adapter with normalized keys."""

import json
import os
import re
from typing import List, Dict, Any

from datasets import Dataset, load_dataset

from src.const import *
from src.prompts.math import build_gsm8k_prompt
from src.data.teacher_text import build_teacher_text


def _strip_gold_answer(raw: str) -> str:
    """Normalize a GSM8K gold answer ("18", "1,000", "$5") to a canonical numeric string."""
    s = raw.strip().replace(",", "").replace("$", "")
    return s


def load_gsm8k(
    split: str = "test",
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load GSM8K and return a list of normalized dicts.

    Splits: train (7473), test (1319). Source: HF dataset 'gsm8k', config 'main'.

    Returns dicts with keys:
        task_id, question, reference_solution (full CoT before '####'),
        gold_answer (string after '####', commas/$ stripped).
    """
    cache_dir = os.path.join("outputs", "cache", "negative_demonstrations", "problems")
    cache_file = os.path.join(cache_dir, f"gsm8k_{split}_seed{seed}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            results = json.load(f)
        if max_samples is not None:
            results = results[:max_samples]
        print(f"[GSM8K] Loaded {len(results)} problems from cache ({cache_file})")
        return results

    ds = load_dataset("gsm8k", "main", split=split)

    results = []
    for idx, ex in enumerate(ds):
        full = ex["answer"]
        if "####" not in full:
            continue
        cot, gold = full.rsplit("####", 1)
        results.append({
            "task_id": f"gsm8k_{split}_{idx}",
            "question": ex["question"].strip(),
            "reference_solution": cot.strip(),
            "gold_answer": _strip_gold_answer(gold),
        })

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[GSM8K] Cached {len(results)} problems to {cache_file}")

    if max_samples is not None:
        results = results[:max_samples]

    print(f"[GSM8K] Loaded {len(results)} problems (split={split})")
    return results


def load_gsm8k_training_dataset(
    seed: int = 42,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
):
    """Load GSM8K for UniSD training. Returns (train_dataset, eval_dataset) as HF Datasets.

    Each example has:
      - prompt: chat format [{"role": "user", "content": <gsm8k_prompt>}]
      - teacher_prompt: chat format [{"role": "user", "content": <gsm8k_prompt + reference solution + gold answer>}]
      - task_id, gold_answer (metadata)
    """
    train_problems = load_gsm8k("train", seed=seed)
    eval_problems = load_gsm8k("test", seed=seed)

    if max_train_samples is not None:
        train_problems = train_problems[:max_train_samples]
    if max_eval_samples is not None:
        eval_problems = eval_problems[:max_eval_samples]

    def _format_problems(problems: list[dict]) -> list[dict]:
        formatted: list[dict] = []
        for p in problems:
            student_text = build_gsm8k_prompt(p)
            # Reference solution as the demonstration body; gold answer in canonical format.
            reasoning = p["reference_solution"]
            answer = f"#### {p['gold_answer']}"
            teacher_text = build_teacher_text(student_text, answer, reasoning)

            formatted.append({
                "prompt": [{"role": "user", "content": student_text}],
                "teacher_prompt": [{"role": "user", "content": teacher_text}],
                "task_id": p["task_id"],
                "gold_answer": p["gold_answer"],
                "_question": p["question"],
                "_reference_solution": p["reference_solution"],
            })
        return formatted

    train_data = _format_problems(train_problems)
    eval_data = _format_problems(eval_problems)

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    print(f"[GSM8K Training] Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    return train_dataset, eval_dataset
