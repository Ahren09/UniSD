"""Prompt builder for math word problems (GSM8K)."""


def build_gsm8k_prompt(problem: dict) -> str:
    """Build a CoT-style prompt that ends with '#### <answer>' on its own line.

    The '####' sentinel matches the canonical GSM8K format and is what the eval
    extractor looks for first.
    """
    question = problem["question"].strip()
    return (
        "Solve the following math word problem. "
        "Reason step by step. On the last line, write '#### ' followed by the "
        "final numeric answer (no units, no extra text).\n\n"
        f"Question: {question}\n"
    )
