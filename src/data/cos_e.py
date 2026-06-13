"""CoS-E (Commonsense Explanations) dataset adapter with normalized keys."""

import json
import os
import random as _random
from typing import List, Dict, Any

import torch
from datasets import Dataset, load_dataset

from src.const import *
from src.data.retrieval import compute_or_load_embeddings, retrieve_topk_donors
from src.data.scienceqa import build_fewshot_auxiliary_context_scienceqa
from src.teacher.auxiliary_context import build_induction_auxiliary_context
from src.prompts.mcqa import build_mcqa_prompt
from src.data.teacher_text import build_teacher_text


def load_cos_e(
    split: str = "validation",
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load CoS-E v1.11 dataset and return list of dicts with normalized keys.

    Assigns shuffled option labels per-example using a deterministic
    per-example seed (so option order is fixed for a given seed+index).
    Dataset order is NEVER shuffled.

    Args:
        split: HF split name ("train", "validation"). CoS-E has no test split.
        max_samples: Cap on total examples.
        seed: Seed for per-example option shuffle.

    Returns dicts with keys:
        task_id, question, options (dict A/B/C -> text), correct_letter,
        correct_answer, explanation, num_choices
    """
    # Try cache first
    cache_dir = os.path.join("outputs", "cache", "negative_demonstrations", "problems")
    cache_file = os.path.join(cache_dir, f"cos_e_{split}_seed{seed}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            results = json.load(f)
        if max_samples is not None:
            results = results[:max_samples]
        print(f"[CoS-E] Loaded {len(results)} problems from cache ({cache_file})")
        return results

    ds = load_dataset("Salesforce/cos_e", "v1.11", split=split)

    results = []
    for idx, ex in enumerate(ds):
        choices = ex["choices"]
        answer_text = ex["answer"]
        num_choices = len(choices)

        # Find correct index by matching answer text
        correct_idx = choices.index(answer_text)

        # Shuffle options with per-example seed for reproducibility
        rng_local = _random.Random(seed + idx)
        shuffled = list(range(num_choices))
        rng_local.shuffle(shuffled)

        labels = [chr(ord('A') + i) for i in range(num_choices)]
        options = {labels[i]: choices[shuffled[i]] for i in range(num_choices)}
        # Find which label the correct answer got
        correct_label = labels[shuffled.index(correct_idx)]

        results.append({
            "task_id": f"cos_e_{split}_{idx}",
            "question": ex["question"],
            "options": options,
            "correct_letter": correct_label,
            "correct_answer": answer_text,
            "explanation": ex.get("abstractive_explanation", ""),
            "num_choices": num_choices,
        })

    # Save to cache (before max_samples truncation)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[CoS-E] Cached {len(results)} problems to {cache_file}")

    if max_samples is not None:
        results = results[:max_samples]

    print(f"[CoS-E] Loaded {len(results)} problems (split={split})")
    return results


def load_cos_e_training_dataset(
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
    keep_extra_columns: list[str] | None = None,
):
    """Load CoS-E for UniSD training. Returns (train_dataset, eval_dataset) as HF Datasets.

    Uses HF's official train/validation splits directly.
    Each example has:
      - prompt: chat format [{"role": "user", "content": <mcqa_prompt>}]
      - teacher_prompt: chat format [{"role": "user", "content": <mcqa_prompt + demo>}]
      - task_id, correct_letter (metadata)
    """
    train_problems = load_cos_e("train", seed=seed)
    eval_problems = load_cos_e("validation", seed=seed)

    if max_train_samples is not None:
        train_problems = train_problems[:max_train_samples]
    if max_eval_samples is not None:
        eval_problems = eval_problems[:max_eval_samples]

    def _format_problems(problems: list[dict]) -> list[dict]:
        formatted: list[dict] = []
        for p in problems:
            student_text = build_mcqa_prompt(p, structured_output=structured_output)
            teacher_text = build_teacher_text(
                student_text, p["correct_letter"], p["explanation"],
            )

            formatted.append({
                "prompt": [{"role": "user", "content": student_text}],
                "teacher_prompt": [{"role": "user", "content": teacher_text}],
                "task_id": p["task_id"],
                "correct_letter": p["correct_letter"],
                "domain": "",
                "subdomain": "",
                # Keep raw fields for fewshot donor pool
                "_question": p["question"],
                "_options": p["options"],
                "_explanation": p["explanation"],
            })
        return formatted

    train_data = _format_problems(train_problems)
    eval_data = _format_problems(eval_problems)

    # --- Optional fewshot teacher prompts ---
    if num_auxiliary_contexts > 0 and len(train_data) > 1:
        donor_pool = train_problems

        if fewshot_method == RETRIEVAL:
            texts = [p["question"] for p in train_problems]
            cache_dir = embedding_cache_dir or f"data/.cache/cos_e_{retrieval_model.replace('/', '_')}"
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

            print(f"[CoS-E Retrieval] Pool={len(donor_pool)}, embedding_dim={embeddings.shape[1]}, "
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

            print(f"[CoS-E Training] Added {num_auxiliary_contexts} retrieval-based fewshot teacher prompts per example")

        elif fewshot_method == INDUCTION:
            from src.data.mbpp import load_instructions_from_cache

            if not induction_cache_file:
                raise ValueError(
                    f"fewshot_method={INDUCTION!r} requires induction_cache_file. "
                    "Run `python -m src.teacher.instruction_induction --dataset cos_e` first."
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

            print(f"[CoS-E Training] Added {num_auxiliary_contexts} induction-based fewshot teacher prompts per example")

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

            print(f"[CoS-E Training] Added {num_auxiliary_contexts} random fewshot teacher prompts per example")

        else:
            raise ValueError(f"Unsupported fewshot_method: {fewshot_method}")

    # Clean up internal fields (keep columns named in keep_extra_columns).
    _keep = set(keep_extra_columns or [])
    for ex in train_data + eval_data:
        for key in ["_question", "_options", "_explanation"]:
            if key not in _keep:
                ex.pop(key, None)

    print(f"[CoS-E Training] train={len(train_data)}, eval={len(eval_data)}")

    return Dataset.from_list(train_data), Dataset.from_list(eval_data)
