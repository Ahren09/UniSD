"""ScienceQA dataset adapter with normalized keys."""

import json
import os
import random as _random
from typing import List, Dict, Any

import torch
from datasets import Dataset, load_dataset

from src.const import *
from src.data.retrieval import compute_or_load_embeddings, retrieve_topk_donors, get_embedding_text
from src.prompts.mcqa import build_mcqa_prompt
from src.data.teacher_text import build_teacher_text
from src.teacher.auxiliary_context import build_induction_auxiliary_context

# Subject name normalization (HF field uses spaces, CLI uses underscores)
_SUBJECT_MAP = {
    "natural_science": "natural science",
    "language_science": "language science",
    "social_science": "social science",
}


def load_scienceqa(
    split: str = "test",
    max_samples: int | None = None,
    seed: int = 42,
    text_only: bool = True,
    subset: str = "all",
) -> list[dict]:
    """Load ScienceQA dataset and return list of dicts with normalized keys.

    Assigns shuffled option labels per-example using a deterministic
    per-example seed (so option order is fixed for a given seed+index).
    Dataset order is NEVER shuffled.

    Args:
        split: HF split name ("train", "validation", "test").
        max_samples: Cap on total examples.
        seed: Seed for per-example option shuffle.
        text_only: If True, filter to text-only examples (image is None).
        subset: "all", "natural_science", "language_science", "social_science".

    Returns dicts with keys:
        task_id, question, options (dict A/B/... -> text), correct_letter,
        correct_answer, explanation, lecture, domain, subdomain, num_choices,
        grade, hint
    """
    # Try cache first
    cache_dir = os.path.join("outputs", "cache", "negative_demonstrations", "problems")
    if text_only:
        setting = "text"
        
    else:
        setting = "visual"
    
    cache_file = os.path.join(cache_dir, f"scienceqa_{split}_seed{seed}_{setting}_{subset}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            results = json.load(f)
        if max_samples is not None:
            results = results[:max_samples]
        print(f"[ScienceQA] Loaded {len(results)} problems from cache ({cache_file})")
        return results

    ds = load_dataset("derek-thomas/ScienceQA", split=split)
    if text_only:
        ds = ds.filter(lambda x: x["image"] is None)

    # Filter by subject
    if subset != "all":
        target_subject = _SUBJECT_MAP.get(subset, subset)
        ds = ds.filter(lambda x: (x.get("subject", "") or "").lower() == target_subject.lower())

    results = []
    for idx, ex in enumerate(ds):
        choices = ex["choices"]
        num_choices = len(choices)
        correct_idx = ex["answer"]

        # Shuffle options with per-example seed for reproducibility
        rng_local = _random.Random(seed + idx)
        shuffled = list(range(num_choices))
        rng_local.shuffle(shuffled)

        labels = [chr(ord('A') + i) for i in range(num_choices)]
        options = {labels[i]: choices[shuffled[i]] for i in range(num_choices)}
        # Find which label the correct answer got
        correct_label = labels[shuffled.index(correct_idx)]

        results.append({
            "task_id": f"scienceqa_{split}_{idx}",
            "question": ex["question"],
            "options": options,
            "correct_letter": correct_label,
            "correct_answer": choices[correct_idx],
            "explanation": ex.get("solution", ""),
            "lecture": ex.get("lecture", ""),
            "domain": ex.get("subject", ""),
            "subdomain": ex.get("topic", ""),
            "num_choices": num_choices,
            "grade": str(ex.get("grade", "")),
            "hint": ex.get("hint", ""),
        })

    # Save to cache (before max_samples truncation)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[ScienceQA] Cached {len(results)} problems to {cache_file}")

    if max_samples is not None:
        results = results[:max_samples]

    print(f"[ScienceQA] Loaded {len(results)} problems (split={split}, text_only={text_only}, subset={subset})")
    return results


def build_teacher_text_scienceqa(
    student_text: str,
    correct_letter: str,
    explanation: str,
    structured_output: str | None = None,
) -> str:
    """Build teacher prompt with demonstration for ScienceQA.

    Uses same demo markers as gpqa.py / mbpp.py.
    """
    if structured_output == "letter":
        demo_content = correct_letter
    elif structured_output == "json":
        demo_content = json.dumps({
            "reasoning": explanation,
            "answer": correct_letter,
        })
    else:
        demo_content = ""
        if explanation:
            demo_content += f"Reasoning: {explanation}\n"
        demo_content += f"Answer: {correct_letter}."

    return (
        f"{student_text}\n\n"
        f"This is an example for a response to the question:\n"
        f"{demo_content}\n"
        f"\n{SELF_DISTILLATION_INSTRUCTION}"
    )


def build_fewshot_auxiliary_context_scienceqa(
    few_shot_samples: List[Dict[str, Any]],
    current_student_text: str,
    structured_output: str | None = None,
) -> str:
    """Prepend reference ScienceQA examples before the student prompt.
    """
    if not few_shot_samples:
        return current_student_text

    header = (
        "## Reference Samples (DO NOT SOLVE THESE)\n"
        "The following are solved examples from OTHER questions, provided only as reference.\n"
        "Do NOT copy their answers. Do NOT mention these examples in your final answer.\n\n"
    )

    refs: List[str] = []
    for idx, s in enumerate(few_shot_samples, 1):
        options_text = "\n".join(
            f"{k}) {v}" for k, v in sorted(s["options"].items())
        )

        if structured_output == "letter":
            answer_text = s["correct_letter"]
        elif structured_output == "json":
            answer_text = json.dumps({
                "answer": s["correct_letter"],
                "reasoning": s.get("explanation", ""),
            })
        else:
            answer_text = f"Answer: {s['correct_letter']}."
            if s.get("explanation"):
                answer_text += f"\nExplanation: {s['explanation']}"

        refs.append(f"""### [REFERENCE ONLY #{idx}]
Reference Question:
{s['question']}

{options_text}

Reference Answer:
{answer_text}
""")

    separator = """
-----
## Task to Solve (ONLY ANSWER THIS TASK)
Now solve ONLY the following task. Output ONLY the answer for THIS task.

"""

    return header + "\n".join(refs) + separator + current_student_text


def load_scienceqa_training_dataset(
    seed: int = 42,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    num_auxiliary_contexts: int = 0,
    fewshot_method: str | None = None,
    structured_output: str | None = None,
    retrieval_model: str = "all-MiniLM-L6-v2",
    embedding_cache_dir: str | None = None,
    demo_k: int = 1,
    induction_cache_file: str | None = None,
    subset: str = "all",
    text_only: bool = True,
    include_lecture: bool = True,
    keep_extra_columns: list[str] | None = None,
):
    """Load ScienceQA for UniSD training. Returns (train_dataset, eval_dataset) as HF Datasets.

    Uses HF's official train/validation splits directly (no manual split).
    Each example has:
      - prompt: chat format [{"role": "user", "content": <mcqa_prompt>}]
      - teacher_prompt: chat format [{"role": "user", "content": <mcqa_prompt + demo>}]
      - task_id, correct_letter, domain, subdomain (metadata)
    """
    train_problems = load_scienceqa("train", seed=seed, text_only=text_only, subset=subset)
    eval_problems = load_scienceqa("validation", seed=seed, text_only=text_only, subset=subset)

    if max_train_samples is not None:
        train_problems = train_problems[:max_train_samples]
    if max_eval_samples is not None:
        eval_problems = eval_problems[:max_eval_samples]

    def _format_problems(problems: list[dict]) -> list[dict]:
        formatted: list[dict] = []
        for p in problems:
            context = p.get("lecture") if include_lecture and p.get("lecture") else None
            student_text = build_mcqa_prompt(p, structured_output=structured_output, context=context)
            teacher_text = build_teacher_text(
                student_text, p["correct_letter"], p["explanation"],
            )

            formatted.append({
                "prompt": [{"role": "user", "content": student_text}],
                "teacher_prompt": [{"role": "user", "content": teacher_text}],
                "task_id": p["task_id"],
                "correct_letter": p["correct_letter"],
                "domain": p["domain"],
                "subdomain": p["subdomain"],
                # Keep raw fields for fewshot donor pool
                "_question": p["question"],
                "_options": p["options"],
                "_explanation": p["explanation"],
                "_lecture": p.get("lecture", ""),
            })
        return formatted

    train_data = _format_problems(train_problems)
    eval_data = _format_problems(eval_problems)

    # --- Optional fewshot teacher prompts ---
    if num_auxiliary_contexts > 0 and len(train_data) > 1:
        donor_pool = train_problems

        if fewshot_method == RETRIEVAL:
            texts = [p["question"] for p in train_problems]
            cache_dir = embedding_cache_dir or f"data/.cache/scienceqa_{retrieval_model.replace('/', '_')}"
            embeddings = compute_or_load_embeddings(
                texts,
                model_name=retrieval_model,
                cache_path=cache_dir,
                device="cuda",
            )
            self_indices = torch.arange(len(train_data), device=embeddings.device)
            topk_indices = retrieve_topk_donors(
                embeddings, embeddings,
                k=num_auxiliary_contexts * demo_k,
                self_indices=self_indices,
            )
            topk_list = topk_indices.cpu().tolist()

            print(f"[ScienceQA Retrieval] Pool={len(donor_pool)}, embedding_dim={embeddings.shape[1]}, "
                  f"model={retrieval_model}")

            for i, example in enumerate(train_data):
                current_student_text = example["prompt"][0]["content"]
                for k in range(num_auxiliary_contexts):
                    demos = [donor_pool[topk_list[i][k * demo_k + d]] for d in range(demo_k)]
                    augmented = build_fewshot_auxiliary_context_scienceqa(
                        demos, current_student_text,
                        structured_output=structured_output,
                    )
                    example[f"fewshot_teacher_prompt_{k}"] = [{"role": "user", "content": augmented}]

            print(f"[ScienceQA Training] Added {num_auxiliary_contexts} retrieval-based fewshot teacher prompts per example")

        elif fewshot_method == INDUCTION:
            from src.data.mbpp import load_instructions_from_cache

            if not induction_cache_file:
                raise ValueError(
                    f"fewshot_method={INDUCTION!r} requires induction_cache_file. "
                    "Run `python -m src.teacher.instruction_induction --dataset scienceqa` first."
                )
            instructions = load_instructions_from_cache(
                induction_cache_file, len(train_data), num_auxiliary_contexts,
            )

            rng_induction = _random.Random(seed)
            random_induction_indices = rng_induction.choices(
                range(len(instructions)), k=len(train_data) * num_auxiliary_contexts
            )
            for i, example in enumerate(train_data):
                student_text = example["prompt"][0]["content"]
                for k in range(num_auxiliary_contexts):
                    augmented = build_induction_auxiliary_context(
                        student_text, instructions[random_induction_indices[i * num_auxiliary_contexts + k]]
                    )
                    example[f"fewshot_teacher_prompt_{k}"] = [{"role": "user", "content": augmented}]

            print(f"[ScienceQA Training] Added {num_auxiliary_contexts} induction-based fewshot teacher prompts per example")

        elif fewshot_method == RANDOM:
            rng_fs = _random.Random(seed)

            for i, example in enumerate(train_data):
                current_student_text = example["prompt"][0]["content"]
                candidate_indices = [j for j in range(len(donor_pool)) if j != i]

                if len(candidate_indices) >= num_auxiliary_contexts:
                    chosen = rng_fs.sample(candidate_indices, k=num_auxiliary_contexts)
                else:
                    chosen = [rng_fs.choice(candidate_indices) for _ in range(num_auxiliary_contexts)]

                for k, j in enumerate(chosen):
                    augmented = build_fewshot_auxiliary_context_scienceqa(
                        [donor_pool[j]], current_student_text,
                        structured_output=structured_output,
                    )
                    example[f"fewshot_teacher_prompt_{k}"] = [{"role": "user", "content": augmented}]

            print(f"[ScienceQA Training] Added {num_auxiliary_contexts} random fewshot teacher prompts per example")

        else:
            raise ValueError(f"Unsupported fewshot_method: {fewshot_method}")

    # Clean up internal fields (keep columns listed in keep_extra_columns)
    keep_extra = set(keep_extra_columns or [])
    for ex in train_data + eval_data:
        for key in ["_question", "_options", "_explanation", "_lecture"]:
            if key not in keep_extra:
                ex.pop(key, None)

    print(f"[ScienceQA Training] train={len(train_data)}, eval={len(eval_data)}")

    return Dataset.from_list(train_data), Dataset.from_list(eval_data)
