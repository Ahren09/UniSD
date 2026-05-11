"""Prompt builder for multiple-choice QA tasks (e.g., GPQA, ScienceQA)."""
from src.const import *
import json

def build_mcqa_prompt(problem: dict, structured_output: str | None = None, context: str | None = None, reasoning: bool = True) -> str:
    """Build a multiple-choice QA prompt.

    Args:
        problem: dict with keys 'question' and 'options' (dict A/B/C/... -> text).
        structured_output: None for free-form CoT, "letter" for single-letter,
            "json" for JSON output.
        context: Optional context text (e.g., lecture) to insert before options.
    """
    options = problem["options"]
    sorted_keys = sorted(options.keys())

    options_text = ""
    for k in sorted_keys:
        v = options[k].strip().strip('\n')
        options_text += f"{k}. {v}\n"
    options_text = options_text.strip()

    # Build dynamic letter list string: "A or B", "A, B, or C", "A, B, C, or D", etc.
    if len(sorted_keys) == 1:
        letters_str = sorted_keys[0]
    elif len(sorted_keys) == 2:
        letters_str = f"{sorted_keys[0]} or {sorted_keys[1]}"
    else:
        letters_str = ", ".join(sorted_keys[:-1]) + f", or {sorted_keys[-1]}"

    # Slash-separated for format line: "A/B/C/D"
    letters_slash = "/".join(sorted_keys)

    if structured_output == "letter":
        instruction = f"Output ONLY the letter of your answer ({letters_str}). No other text."
    elif structured_output == "json":
        if reasoning:
            instruction = """Return your answer as JSON, nothing else. Example:
{
    "answer": "X",
    "reasoning": "BRIEF EXPLANATION"
}
"""
        else:
            instruction = """Return your answer as JSON, nothing else. Example:
{
    "answer": "X"
}
"""
    else:
        # Free-form (default)
        if reasoning:
            # Build "Answer: A, Answer: B, ..." list
            answer_examples = ", ".join(f"Answer: {k}" for k in sorted_keys)
            instruction = f"""Reason briefly, then provide the final answer (only the option letter) in the last line. Your final answer must be one of {letters_str}. There is only one correct answer. Keep your reasoning concise and to the point. Do not output anything after the answer line.

## Output format
Reasoning: <at most 3 short sentences>
Answer: <{letters_slash}>

"""
        else:
            instruction = f"""Output only the answer letter. Do not include any reasoning or explanation.

## Output format
Answer: <{letters_slash}>
"""

    context_block = f"Context: {context}\n\n" if context else ""

    prompt = (
        f"Answer the following multiple-choice question.\n\n"
        f"Question: {problem['question']}\n\n"
        f"{context_block}"
        f"{options_text}\n\n"
        f"{instruction}"
    )

    return prompt

def build_mcqa_answer(problem: dict, structured_output: str | None, dataset_name: str, reasoning: bool = True) -> str:
    """Build the reference answer string for an MCQA problem.

    Mirrors the demo_content logic from build_auxiliary_text_mcqa/gpqa.
    """
    letter = problem["correct_letter"]
    explanation = problem.get("explanation", "")

    if structured_output == "letter":
        return letter
    elif structured_output == "json":
        if reasoning:
            return json.dumps({"answer": letter, "reasoning": explanation})
        else:
            return json.dumps({"answer": letter})
    else:
        if not reasoning:
            return f"Answer: {letter}."
        # Free-form: match existing teacher text patterns
        if dataset_name == GPQA:
            # GPQA: explanation first, then answer (matches build_auxiliary_context_gpqa)
            parts = []
            if explanation:
                parts.append(f"Explanation: {explanation}")
            parts.append(f"Answer: {letter}.")
            return "\n".join(parts)
        else:
            # ScienceQA, CoS-E: answer first, then explanation
            answer = f"Answer: {letter}."
            if explanation:
                answer += f"\nExplanation: {explanation}"
            return answer

