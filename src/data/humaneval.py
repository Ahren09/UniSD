"""HumanEval dataset adapter."""

from datasets import load_dataset


def load_humaneval(max_samples: int | None = None) -> list[dict]:
    """Load HumanEval dataset and return list of dicts.

    Returns dicts with keys:
        task_id, prompt, canonical_solution, test, entry_point
    """
    ds = load_dataset("openai/openai_humaneval", split="test")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    results = []
    for ex in ds:
        results.append({
            "task_id": ex["task_id"],
            "prompt": ex["prompt"],
            "canonical_solution": ex["canonical_solution"],
            "test": ex["test"],
            "entry_point": ex["entry_point"],
        })

    return results
