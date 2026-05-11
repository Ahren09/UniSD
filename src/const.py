DEBUG = False
"""Training mode constants for UniSDTrainer."""
# The 6 training modes
AGREEMENT_SEQ_RANDOM = "agreement_seq_random"    # K pre-built fewshot prompts, random demo selection
AGREEMENT_SEQ_RETRIEVAL = "agreement_seq_retrieval"  # K pre-built fewshot prompts, retrieval-based demos
AGREEMENT_SEQ_INDUCTION = "agreement_seq_induction"              # K pre-built fewshot prompts, induced instructions
CONTRASTIVE = "contrastive"               # Token-level contrastive learning with positive/negative demonstrations
EMA = "ema"          # Temporal stabilization via EMA teacher (exponential moving average of student parameters)
MATCH_JOINT = "match_joint"  # Joint logit distillation + final-layer feature matching
MATCH_REPR = "match_repr"                  # Final-layer feature matching only (no logit distillation)
UNISD_STAR = "unisd_star"                          # Combines: EMA + MATCH_JOINT + CONTRASTIVE + fewshot agreement
CLIP = "clip"                # Per-token JSD divergence clipping (caps high-divergence tokens)
# Token-level agreement variants of fewshot/induction modes
AGREEMENT_TOK_RANDOM = "agreement_tok_random"
AGREEMENT_TOK_RETRIEVAL = "agreement_tok_retrieval"
AGREEMENT_TOK_INDUCTION = "agreement_tok_induction"

ALL_MODES = {AGREEMENT_SEQ_RANDOM, AGREEMENT_SEQ_RETRIEVAL, AGREEMENT_SEQ_INDUCTION, CONTRASTIVE, EMA,
             MATCH_JOINT, MATCH_REPR, UNISD_STAR,
             CLIP,
             AGREEMENT_TOK_RANDOM, AGREEMENT_TOK_RETRIEVAL, AGREEMENT_TOK_INDUCTION}
AGREEMENT_MODES = {AGREEMENT_SEQ_RANDOM, AGREEMENT_SEQ_RETRIEVAL, AGREEMENT_SEQ_INDUCTION,
                        AGREEMENT_TOK_RANDOM, AGREEMENT_TOK_RETRIEVAL, AGREEMENT_TOK_INDUCTION,
                        UNISD_STAR}

# Fewshot context-selection strategies (orthogonal to seq/tok granularity).
RANDOM = "random"
RETRIEVAL = "retrieval"
INDUCTION = "induction"
FEWSHOT_METHODS = {RANDOM, RETRIEVAL, INDUCTION}

MODE_TO_FEWSHOT = {
    AGREEMENT_SEQ_RANDOM: RANDOM,
    AGREEMENT_TOK_RANDOM: RANDOM,
    AGREEMENT_SEQ_RETRIEVAL: RETRIEVAL,
    AGREEMENT_TOK_RETRIEVAL: RETRIEVAL,
    AGREEMENT_SEQ_INDUCTION: INDUCTION,
    AGREEMENT_TOK_INDUCTION: INDUCTION,
}
METHOD_DISPLAY_NAME = {
    AGREEMENT_SEQ_RETRIEVAL: "Agree",
    CONTRASTIVE: "Contrast",
    EMA: "EMA",
    MATCH_REPR: "Match",
    CLIP: "Clip",
}

# Method family constants (for color grouping).
# CONTRASTIVE and EMA are defined above as mode constants — they double as
# family identifiers since the values match. Only AGREEMENT/MATCHING/CLIPPING
# need their own definitions here.
AGREEMENT = "agreement"
MATCHING = "matching"
CLIPPING = "clipping"
# Token-level agreement granularity modes
TOKEN_GRANULARITY_MODES = {AGREEMENT_TOK_RANDOM, AGREEMENT_TOK_RETRIEVAL, AGREEMENT_TOK_INDUCTION, UNISD_STAR}


SELF_DISTILLATION_INSTRUCTION = "Carefully study the reference solution, then solve the problem below with your own approach."

# DATASET NAMES
MBPP = "mbpp"
TOOLUSE = "tooluse"
GPQA = "gpqa"
SCIENCEQA = "scienceqa"
COS_E = "cos_e"
MEDMCQA = "medmcqa"
HUMANEVAL = "humaneval"
GSM8K = "gsm8k"

DISPLAY_DATASET = {
    "scienceqa-all": "ScienceQA",
    "gpqa_main": "GPQA",
    "cos_e": "CoS-E",
    "mbpp": "MBPP",
    "humaneval": "HumanEval",
}