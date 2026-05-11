"""Shared tooluse utility functions for dataset loading and prompt formatting."""

import glob
import re
import json
import random
import sys
from string import Template

import torch
from datasets import Dataset, load_dataset

from src.const import *
from src.data.retrieval import compute_or_load_embeddings, retrieve_topk_donors


# --------------------------------------------------------------------------- #
# Tool-call parsing & scoring (shared by eval/* and analysis/* scripts)        #
# --------------------------------------------------------------------------- #


def parse_tool_calls(text: str) -> list[dict]:
    """Extract (Action, Action_Input) pairs from model output.

    Matches the ToolAlpaca ReAct format:
        Thought: ...
        Action: <tool_name>
        Action Input: <json_args>
    """
    pattern = r"Action\s*\d*\s*:\s*(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:\s*(.*?)(?=\nThought|\nAction\s*\d*\s*:|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    calls = []
    for action, action_input in matches:
        action = action.strip()
        action_input = action_input.strip().strip('"')
        calls.append({"Action": action, "Action_Input": action_input})
    if not calls:
        pattern_single = r"Action\s*\d*\s*:\s*(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:\s*(.*)"
        match = re.search(pattern_single, text, re.DOTALL)
        if match:
            action = match.group(1).strip()
            action_input = match.group(2).strip().strip('"')
            calls.append({"Action": action, "Action_Input": action_input})
    return calls


def compare_action_names(predicted_calls: list[dict], golden_calls: list[dict]) -> float:
    """Check if all predicted action names match golden in order. Returns 1.0 or 0.0."""
    if len(predicted_calls) != len(golden_calls):
        return 0.0
    for pred, gold in zip(predicted_calls, golden_calls):
        if pred["Action"].strip() != gold["Action"].strip():
            return 0.0
    return 1.0


def compare_action_inputs(pred_input_str, gold_input_str) -> float:
    """Compare Action_Input JSON dicts. Returns fraction of matching key-value pairs."""
    try:
        pred = json.loads(pred_input_str)
    except (json.JSONDecodeError, TypeError):
        pred = {}
    try:
        gold = json.loads(gold_input_str)
    except (json.JSONDecodeError, TypeError):
        gold = {}
    if not isinstance(pred, dict):
        pred = {}
    if not isinstance(gold, dict):
        gold = {}

    if not gold:
        return 1.0 if not pred else 0.0

    matches = 0
    total = 0
    for key, gold_val in gold.items():
        if gold_val == "" or gold_val is None:
            continue
        total += 1
        pred_val = pred.get(key)
        if str(pred_val).strip() == str(gold_val).strip():
            matches += 1

    if total == 0:
        return 1.0
    return matches / total


def compute_argument_accuracy(predicted_calls: list[dict], golden_calls: list[dict]) -> float:
    """Average argument accuracy across all call steps."""
    if len(predicted_calls) != len(golden_calls):
        return 0.0
    if not golden_calls:
        return 1.0
    scores = [
        compare_action_inputs(pred["Action_Input"], gold["Action_Input"])
        for pred, gold in zip(predicted_calls, golden_calls)
    ]
    return sum(scores) / len(scores)


def classify_failure(predicted_calls: list[dict], golden_calls: list[dict]) -> str:
    """Classify failure type (first match wins).

    Returns one of: "parse_failure", "under_generation", "extra_tool_calls",
    "wrong_action_name", "invalid_json_trailing", "wrong_arg_values", "correct".
    """
    if not predicted_calls:
        return "parse_failure"
    if len(predicted_calls) < len(golden_calls):
        return "under_generation"
    if len(predicted_calls) > len(golden_calls):
        return "extra_tool_calls"
    for pred, gold in zip(predicted_calls, golden_calls):
        if pred["Action"].strip() != gold["Action"].strip():
            return "wrong_action_name"
    for pred, gold in zip(predicted_calls, golden_calls):
        ai = pred["Action_Input"]
        if "\n" in ai:
            return "invalid_json_trailing"
        try:
            json.loads(ai)
        except (json.JSONDecodeError, TypeError):
            return "invalid_json_trailing"
    for pred, gold in zip(predicted_calls, golden_calls):
        if compare_action_inputs(pred["Action_Input"], gold["Action_Input"]) < 1.0:
            return "wrong_arg_values"
    return "correct"


def score_first_call(
    pred_calls: list[dict],
    gold_calls: list[dict],
    truncate_newline: bool = False,
) -> tuple[float, float, float]:
    """Score using only the first predicted call vs first golden call.

    Returns (action_score, arg_score, full_score), each in {0.0, 1.0} except
    arg_score which can be a fraction. full_score is 1.0 iff action and args
    both match exactly.

    truncate_newline: if True and the predicted Action_Input contains a newline,
    only the part before the first newline is scored. Used by aggregate_decode_grid
    to ignore trailing JSON/commentary that the model may emit after the args.
    """
    if not gold_calls:
        return (1.0, 1.0, 1.0)
    if not pred_calls:
        return (0.0, 0.0, 0.0)

    pred = pred_calls[0]
    gold = gold_calls[0]

    action_ok = 1.0 if pred["Action"].strip() == gold["Action"].strip() else 0.0

    pred_input = pred["Action_Input"]
    if truncate_newline and "\n" in pred_input:
        pred_input = pred_input[: pred_input.index("\n")]

    arg_score = compare_action_inputs(pred_input, gold["Action_Input"])
    full_score = 1.0 if (action_ok == 1.0 and arg_score == 1.0) else 0.0
    return (action_ok, arg_score, full_score)


def build_teacher_text_tooluse(
    student_text: str,
    answer: str,
    reasoning: str = "",
) -> str:
    """Build teacher prompt with demonstration for ToolUse.

    Mirrors the teacher_prompt format used in load_tooluse_dataset.
    """
    demo_content = answer
    if reasoning:
        demo_content = f"Thought: {reasoning}\n{answer}"
    suffix = f"\n{SELF_DISTILLATION_INSTRUCTION} Including your thought process (the 'Thought' field in your response)."
    return (
        f"\n{student_text}\n\n"
        f"This is an example for a response to the question:\n"
        f"{demo_content}\n"
        f"{suffix}"
    )


STRUCTURED_JSON_INSTRUCTION = (
    'Return your answer as a JSON object with this exact structure:\n'
    '{"tool_calls": [{"Thought": "<your_reasoning>", "Action": "<tool_name>", '
    '"Action_Input": {"<param>": "<value>", ...}}]}\n'
    'Output ONLY the JSON object, no other text.\n\nBegin!'
)


def rewrite_prompt_for_structured(prompt: str, json_instruction: str = STRUCTURED_JSON_INSTRUCTION) -> tuple[str, bool]:
    """Replace the 'Use the following format:...Begin!' block with JSON output instructions."""
    s = prompt.find("Use the following format:")
    if s < 0:
        return prompt, False
    e = prompt.find("Begin!", s)
    if e < 0:
        return prompt, False
    return prompt[:s] + json_instruction + prompt[e + len("Begin!"):], True


def _parse_tool_calls_from_text(text: str) -> list[dict]:
    """Extract (Thought, Action, Action_Input) from ReAct-format text.

    Assumes: thought text comes first, followed by a single Action/Action Input pair.
    """
    match = re.search(
        r"Action\s*\d*\s*:\s*(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:\s*(.*)",
        text, re.DOTALL,
    )
    assert match, f"No Action/Action Input found in: {text[:200]}"
    thought = text[:match.start()].strip()
    return [{
        "Thought": thought,
        "Action": match.group(1).strip(),
        "Action_Input": match.group(2).strip().strip('"'),
    }]


def _extract_json_value(raw: str):
    """Extract the first valid JSON object/value from a potentially noisy string."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try to find a JSON object at the start
    if raw.startswith('{'):
        depth = 0
        for i, c in enumerate(raw):
            if c == '{': depth += 1
            elif c == '}': depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[:i + 1])
                except json.JSONDecodeError:
                    break
    return raw


def golden_response_to_json(golden_response) -> str:
    """Convert a ReAct-format golden response to structured JSON string."""
    text = '\n'.join(golden_response) if isinstance(golden_response, list) else golden_response
    calls = _parse_tool_calls_from_text(text)
    tool_calls = []
    for call in calls:
        action_input = _extract_json_value(call["Action_Input"])
        assert isinstance(action_input, (dict, None))
        if action_input is None:
            action_input = {}
        tool_calls.append({"Thought": call["Thought"], "Action": call["Action"], "Action_Input": action_input})

    return json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)


def build_auxiliary_context_tooluse(student_text: str, answer: str, reasoning: str = "") -> str:
    """Build teacher prompt with demonstration for tooluse (positive or negative).

    Uses the same demo markers as other datasets
    """
    demo_content = answer
    if reasoning:
        demo_content = f"{reasoning}\n{answer}"

    teacher_suffix = f"\n{SELF_DISTILLATION_INSTRUCTION} Include your thought process (the 'Thought' field in your response)."

    return (
        f"\n{student_text}\n\n"
        f"This is an example for a response to the question:\n"
        f"{demo_content}\n"
        f"{teacher_suffix}"
    )


def build_fewshot_auxiliary_context_tooluse(
    few_shot_samples: list[dict],
    current_teacher_text: str,
    structured_output: bool = False,
) -> str:
    """Prepend compact tooluse reference examples before the original teacher prompt.

    Each reference shows only instruction + golden_response (no API docs) to keep
    the overhead per reference at ~100-120 tokens.
    """
    if not few_shot_samples:
        return current_teacher_text

    header = (
        "## Reference Samples (DO NOT SOLVE THESE)\n"
        "The following are solved examples from OTHER tasks, provided only as reference.\n"
        "Do NOT copy their tool names or actions. Do NOT mention these examples in your final answer.\n-----\n"
    )

    refs = []
    for idx, s in enumerate(few_shot_samples, 1):
        if structured_output:
            response_text = golden_response_to_json(s['golden_response'])
        else:
            response_text = ' '.join(s['golden_response']).strip()
        refs.append(
            f"### [REFERENCE ONLY #{idx}]\n"
            f"#### Reference Question\n{s['instruction'].strip()}\n\n"
            f"#### Reference Response\n{response_text}\n"
        )

    separator = (
        "\n-----\n"
        "## Task to Solve (ONLY ANSWER THIS TASK)\n"
        "Now solve ONLY the following task.\n\n"
    )

    prompt = header + "\n".join(refs) + separator + current_teacher_text
    return prompt


def load_tooluse_dataset(
    seed=42,
    num_auxiliary_contexts=0,
    structured_output=False,
    fewshot_method: str | None = None,
    retrieval_model: str = "all-MiniLM-L6-v2",
    output_dir: str = "outputs/",
    demo_k: int = 1,
    induction_cache_file: str | None = None,
) -> tuple[Dataset, Dataset]:
    """Load and prepare tooluse dataset with formatted prompts.

    Args:
        seed: Random seed for train/test split and fewshot sampling.
        num_auxiliary_contexts: If >0, add fewshot_teacher_prompt_{k} fields
            with compact reference examples prepended to each teacher prompt.
        structured_output: If True, rewrite prompts to request JSON output and
            convert golden responses in demos to JSON format.
        fewshot_method: Fewshot strategy (random / retrieval / induction). None = no fewshot context.
        retrieval_model: Sentence-transformers model for retrieval-based fewshot.
        output_dir: Output directory for caching embeddings.
        demo_k: Number of retrieved demos per fewshot teacher context.
        induction_cache_file: Path to pre-generated induction cache.
    """
    hf_dataset = load_dataset("Ahren09/ToolAlpaca")

    raw_train_dataset = hf_dataset["train"].map(
        lambda _, idx: {"task_id": f"tooluse_{idx}"},
        with_indices=True,
    )
    raw_train_list = raw_train_dataset.to_list()

    raw_eval_dataset = hf_dataset["test"].map(
        lambda _, idx: {"task_id": f"tooluse_eval_{idx}"},
        with_indices=True,
    )
    raw_eval_list = raw_eval_dataset.to_list()

    def _format_one(example):
        prompt_text = example['prompt']
        golden = example.get('golden_response')
        # HF test split exposes golden_response=[''] (empty); treat as absent and fall back to golden_answer
        if isinstance(golden, list) and not any(isinstance(s, str) and s.strip() for s in golden):
            golden = None
        if golden is None:
            # Eval data has golden_answer (list of dicts) instead of golden_response (list of strings)
            golden_answer = example.get('golden_answer', [])
            if golden_answer and isinstance(golden_answer[0], dict):
                golden = '\n'.join(
                    f"Action: {a['Action']}\nAction Input: {a.get('Action_Input', '{}')}"
                    for a in golden_answer
                )
            else:
                golden = golden_answer
        golden_text = '\n'.join(golden) if isinstance(golden, list) else golden

        if structured_output:
            prompt_text, _ = rewrite_prompt_for_structured(prompt_text)
            golden_text = golden_response_to_json(golden)

        teacher_suffix = f"\n{SELF_DISTILLATION_INSTRUCTION} Include your thought process (the 'Thought' field in your response)."

        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text
""" + "$suffix")

        return {
            "task_id": example.get("task_id", ""),
            "prompt": [{"role": "user", "content": prompt_text}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(orig_content=prompt_text, output_text=golden_text, suffix=teacher_suffix)}],
        }

    train_data = [_format_one(ex) for ex in raw_train_list]
    eval_data = [_format_one(ex) for ex in raw_eval_list]

    # --- Optional fewshot teacher prompts (3-way mode dispatch) ---
    if num_auxiliary_contexts > 0 and len(train_data) > 1:
        # Donor pool: raw train examples with instruction + golden_response
        donor_pool = raw_train_list

        if fewshot_method == RETRIEVAL:
            # Embed the short `instruction` field (NOT full prompt with tool docs)
            texts = [ex["instruction"] for ex in raw_train_list]
            embedding_cache_dir = f"{output_dir}/cache/embed/tooluse_{retrieval_model.replace('/', '_')}"
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

            print(f"[ToolUse Retrieval] Pool={len(donor_pool)}, embedding_dim={embeddings.shape[1]}, "
                  f"model={retrieval_model}")

            for i, example in enumerate(train_data):
                current_teacher_text = example["teacher_prompt"][0]["content"]
                for k in range(num_auxiliary_contexts):
                    demos = [donor_pool[topk_list[i][k * demo_k + d]] for d in range(demo_k)]
                    augmented = build_fewshot_auxiliary_context_tooluse(
                        demos, current_teacher_text,
                        structured_output=structured_output,
                    )
                    example[f"fewshot_teacher_prompt_{k}"] = [{"role": "user", "content": augmented}]

            print(f"[ToolUse Training] Added {num_auxiliary_contexts} retrieval-based fewshot teacher prompts per example")

        elif fewshot_method == INDUCTION:
            from src.data.mbpp import load_instructions_from_cache
            from src.teacher.auxiliary_context import build_induction_auxiliary_context

            if not induction_cache_file:
                raise ValueError(
                    f"fewshot_method={INDUCTION!r} requires induction_cache_file. "
                    "Run `python -m src.teacher.instruction_induction --dataset tooluse` first."
                )
            instructions = load_instructions_from_cache(
                induction_cache_file, len(train_data), num_auxiliary_contexts,
            )

            rng = random.Random(seed)
            random_induction_indices = rng.choices(range(len(instructions)), k=len(train_data) * num_auxiliary_contexts)
            for i, example in enumerate(train_data):
                student_text = example["prompt"][0]["content"]
                for k in range(num_auxiliary_contexts):
                    augmented = build_induction_auxiliary_context(
                        student_text, instructions[random_induction_indices[i * num_auxiliary_contexts + k]]
                    )
                    example[f"fewshot_teacher_prompt_{k}"] = [{"role": "user", "content": augmented}]

            print(f"[ToolUse Training] Added {num_auxiliary_contexts} induction-based fewshot teacher prompts per example")

        elif fewshot_method == RANDOM:
            rng = random.Random(seed)

            for i, example in enumerate(train_data):
                current_text = example["prompt"][0]["content"]

                candidate_indices = [j for j in range(len(donor_pool)) if j != i]
                if len(candidate_indices) >= num_auxiliary_contexts:
                    chosen = rng.sample(candidate_indices, k=num_auxiliary_contexts)
                else:
                    chosen = [rng.choice(candidate_indices) for _ in range(num_auxiliary_contexts)]

                for k, j in enumerate(chosen):
                    augmented = build_fewshot_auxiliary_context_tooluse(
                        [donor_pool[j]], current_text,
                        structured_output=structured_output,
                    )
                    example[f"fewshot_teacher_prompt_{k}"] = [{"role": "user", "content": augmented}]

            print(f"[ToolUse Training] Added {num_auxiliary_contexts} random fewshot teacher prompts per example")

        else:
            raise ValueError(f"Unsupported fewshot_method: {fewshot_method}")

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data)

    return train_ds, eval_ds
