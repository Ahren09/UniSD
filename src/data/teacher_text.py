from src.const import *

def build_teacher_text(student_text: str, answer: str, reasoning: str = None) -> str:
    f"""Build teacher prompt with demonstration using verbatim drop_demo markers.

    Markers match src/trainers/unisd_trainer.py (drop_demo augmentation):
      start: "This is an example for a response to the question:\\n"
      end:   "\\n {SELF_DISTILLATION_INSTRUCTION}"
    """
    if reasoning:
        reference = f"""Reasoning: {reasoning}
Answer: {answer}
"""
    else:
        reference = f"{answer}"

    teacher_prompt = """
{ORIGINAL_CONTENT}

This is an example for a response to the question:
{reference}

{SELF_DISTILLATION_INSTRUCTION}
"""
    return teacher_prompt.format(ORIGINAL_CONTENT=student_text, reference=reference, SELF_DISTILLATION_INSTRUCTION=SELF_DISTILLATION_INSTRUCTION)
