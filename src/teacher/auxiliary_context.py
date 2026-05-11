"""Pure-string helpers for building teacher auxiliary-context prompts.

This module has NO imports from other src.* packages so it can be
imported freely without circular-import risk.
"""

from src.const import *


def build_induction_auxiliary_context(student_text: str, induced_instruction: str) -> str:
    """Format an induced instruction + student task as a teacher prompt.

    Matching the existing marker in unisd_trainer.py and mbpp.py.
    """
    return f"""## General Instruction for Your Answer
{induced_instruction}

{student_text}

{SELF_DISTILLATION_INSTRUCTION}"""
