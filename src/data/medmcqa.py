"""MedMCQA dataset adapter with normalized keys."""

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


def load_medmcqa(
    split: str = "validation",
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load MedMCQA dataset and return list of dicts with normalized keys.

    Assigns shuffled option labels per-example using a deterministic
    per-example seed (so option order is fixed for a given seed+index).
    Dataset order is NEVER shuffled.

    Args:
        split: HF split name ("train", "validation", "test").
            Note: test split has hidden labels (cop=-1); those examples are skipped.
        max_samples: Cap on total examples.
        seed: Seed for per-example option shuffle.

    Returns dicts with keys:
        task_id, question, options (dict A/B/C/D -> text), correct_letter,
        correct_answer, explanation, num_choices, domain, subdomain
    """
    cache_dir = os.path.join("outputs", "cache", "negative_demonstrations", "problems")
    cache_file = os.path.join(cache_dir, f"medmcqa_{split}_seed{seed}.json")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            results = json.load(f)
        if max_samples is not None:
            results = results[:max_samples]
        print(f"[MedMCQA] Loaded {len(results)} problems from cache ({cache_file})")
        return results

    ds = load_dataset("openlifescienceai/medmcqa", split=split)

    results = []
    skipped = 0
    for idx, ex in enumerate(ds):
        cop = ex["cop"]
        if cop < 0 or cop > 3:
            skipped += 1
            continue

        original_options = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        num_choices = 4

        # Shuffle options with per-example seed for reproducibility
        rng_local = _random.Random(seed + idx)
        shuffled = list(range(num_choices))
        rng_local.shuffle(shuffled)

        labels = [chr(ord('A') + i) for i in range(num_choices)]
        options = {labels[i]: original_options[shuffled[i]] for i in range(num_choices)}
        correct_label = labels[shuffled.index(cop)]

        results.append({
            "task_id": f"medmcqa_{split}_{idx}",
            "question": ex["question"],
            "options": options,
            "correct_letter": correct_label,
            "correct_answer": original_options[cop],
            "explanation": ex.get("exp") or "",
            "num_choices": num_choices,
            "domain": ex.get("subject_name") or "",
            "subdomain": ex.get("topic_name") or "",
        })

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[MedMCQA] Cached {len(results)} problems to {cache_file}"
          + (f" (skipped {skipped} with hidden labels)" if skipped else ""))

    if max_samples is not None:
        results = results[:max_samples]

    print(f"[MedMCQA] Loaded {len(results)} problems (split={split})")
    return results


def load_medmcqa_training_dataset(
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
):
    """Load MedMCQA for UniSD training. Returns (train_dataset, eval_dataset) as HF Datasets.

    Each example has:
      - prompt: chat format [{"role": "user", "content": <mcqa_prompt>}]
      - teacher_prompt: chat format [{"role": "user", "content": <mcqa_prompt + demo>}]
      - task_id, correct_letter, domain, subdomain (metadata)
    """
    train_problems = load_medmcqa("train", seed=seed)
    eval_problems = load_medmcqa("validation", seed=seed)

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
                "domain": p.get("domain", ""),
                "subdomain": p.get("subdomain", ""),
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
            # Cap retrieval pool to bound similarity-matrix memory (full pool ~183k -> 134GB).
            MAX_POOL_SIZE = 10000
            if len(train_problems) > MAX_POOL_SIZE:
                rng_pool = _random.Random(seed)
                pool_indices = rng_pool.sample(range(len(train_problems)), MAX_POOL_SIZE)
                pool_problems = [train_problems[i] for i in pool_indices]
                print(f"[MedMCQA Retrieval] Subsampled donor pool: {len(train_problems)} -> {MAX_POOL_SIZE}")
            else:
                pool_problems = train_problems
            donor_pool = pool_problems

            query_texts = [p["question"] for p in train_problems]
            pool_texts = [p["question"] for p in pool_problems]
            cache_dir = embedding_cache_dir or f"data/.cache/medmcqa_{retrieval_model.replace('/', '_')}"
            query_embeddings = compute_or_load_embeddings(
                query_texts,
                model_name=retrieval_model,
                cache_path=cache_dir,
                device="cuda",
            )
            if len(pool_texts) == len(query_texts):
                pool_embeddings = query_embeddings
            else:
                pool_embeddings = compute_or_load_embeddings(
                    pool_texts,
                    model_name=retrieval_model,
                    cache_path=cache_dir,
                    device="cuda",
                )
            # Self-exclusion only applies when query[i] sits at a known pool index.
            topk_indices = retrieve_topk_donors(
                query_embeddings, pool_embeddings,
                k=num_auxiliary_contexts * demo_k,
                self_indices=None,
            )
            topk_list = topk_indices.cpu().tolist()

            print(f"[MedMCQA Retrieval] Pool={len(donor_pool)}, embedding_dim={pool_embeddings.shape[1]}, "
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

            print(f"[MedMCQA Training] Added {num_auxiliary_contexts} retrieval-based fewshot teacher prompts per example")

        elif fewshot_method == INDUCTION:
            from src.data.mbpp import load_instructions_from_cache

            if not induction_cache_file:
                raise ValueError(
                    f"fewshot_method={INDUCTION!r} requires induction_cache_file. "
                    "Run `python -m src.teacher.instruction_induction --dataset medmcqa` first."
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

            print(f"[MedMCQA Training] Added {num_auxiliary_contexts} induction-based fewshot teacher prompts per example")

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

            print(f"[MedMCQA Training] Added {num_auxiliary_contexts} random fewshot teacher prompts per example")

        else:
            raise ValueError(f"Unsupported fewshot_method: {fewshot_method}")

    # Clean up internal fields
    for ex in train_data + eval_data:
        for key in ["_question", "_options", "_explanation"]:
            ex.pop(key, None)

    print(f"[MedMCQA Training] train={len(train_data)}, eval={len(eval_data)}")

    return Dataset.from_list(train_data), Dataset.from_list(eval_data)
