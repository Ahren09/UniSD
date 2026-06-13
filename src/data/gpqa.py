"""GPQA dataset adapter with normalized keys."""

import json
import random as _random

from datasets import load_dataset

from src.const import *

def load_gpqa(
    subset: str = "gpqa_diamond",
    max_samples: int | None = None,
    seed: int = 42,
    split: str = "test",
) -> list[dict]:
    """Load GPQA dataset and return list of dicts with normalized keys.

    Assigns shuffled A/B/C/D labels per-example using a deterministic
    per-example seed (so option order is fixed for a given seed+index).
    Dataset order is NEVER shuffled.

    Args:
        subset: HF subset name.
        max_samples: Cap on total examples (applied before split).
        seed: Seed for per-example option assignment.
        split: "all" (default), "train" (first train_ratio), or "test" (rest).
        train_ratio: Fraction of data used for train (default 0.8).

    Returns dicts with keys:
        task_id, question, options (dict A-D), correct_letter, correct_answer,
        explanation, domain, subdomain
    """
    ds = load_dataset("Idavidrein/gpqa", subset, split="train")

    results = []

    for idx, ex in enumerate(ds):
        # Collect all 4 answers
        answers = [
            ex["Correct Answer"],
            ex["Incorrect Answer 1"],
            ex["Incorrect Answer 2"],
            ex["Incorrect Answer 3"],
        ]
        # Shuffle options with per-example seed for reproducibility
        rng_local = _random.Random(seed + idx)
        shuffled = list(range(4))
        rng_local.shuffle(shuffled)

        labels = ["A", "B", "C", "D"]
        options = {labels[i]: answers[shuffled[i]] for i in range(4)}
        # Find which label the correct answer (index 0 before shuffle) got
        correct_label = labels[shuffled.index(0)]

        results.append({
            "task_id": f"gpqa_{idx}",
            "question": ex["Question"],
            "options": options,
            "correct_letter": correct_label,
            "correct_answer": ex["Correct Answer"],
            "explanation": ex.get("Explanation", ""),
            "domain": ex.get("High-level domain", ""),
            "subdomain": ex.get("Subdomain", ""),
        })

    if max_samples is not None:
        results = results[:max_samples]

    print(f"[GPQA] Loaded {len(results)} problems from {subset} (split={split})")
    return results


