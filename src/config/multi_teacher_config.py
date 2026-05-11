"""
UniSDConfig: extends DistilConfig with multi-teacher agreement fields.
All new fields default to no-op values, so this config is a drop-in replacement for DistilConfig.
"""
from dataclasses import dataclass, field
from typing import Optional

from src.config.base_config import DistilConfig
from src.const import *


@dataclass
class UniSDConfig(DistilConfig):
    """
    Configuration for UniSD agreement-weighted distillation.

    When ``mode=UNISD_STAR`` ("unisd_star", default), the trainer runs the full UniSD recipe
    (fewshot agreement + contrastive + EMA + final-layer matching).
    """

    mode: str = field(
        default=UNISD_STAR,
        metadata={"help": "UniSD training mode: unisd_star (default), ema, contrastive, "
                          "match_joint, match_repr, agreement_seq_random, "
                          "agreement_seq_retrieval, agreement_seq_induction, clip, "
                          "agreement_tok_random, agreement_tok_retrieval, "
                          "agreement_tok_induction."},
    )
    num_auxiliary_contexts: int = field(
        default=0,
        metadata={"help": "Number of auxiliary contexts (e.g. fewshot examples)."},
    )
    gamma_agreement: float = field(
        default=0.0,
        metadata={
            "help": "Agreement weighting strength. 0.0 disables weighting (baseline). "
            "Higher values downweight tokens where teachers disagree more."
        },
    )
    agreement_stat: str = field(
        default="var",
        metadata={
            "help": "Disagreement statistic: 'var' (variance) or 'range' (max - min)."
        },
    )
    log_agreement_stats: bool = field(
        default=True,
        metadata={
            "help": "Whether to log per-step agreement weight statistics."
        },
    )

    
    context_augment: str = field(
        default="none",
        metadata={
            "help": "Context augmentation strategy: "
            "'none' (no augmentation), 'drop_demo' (remove demonstration section), "
            "'token_drop' (drop a contiguous span of tokens from teacher prompt)."
        },
    )
    context_drop_prob: float = field(
        default=0.5,
        metadata={
            "help": "Fraction of teacher prompt tokens to drop in 'token_drop' augmentation."
        },
    )

    # Contrastive loss fields
    contrastive_weight: float = field(
        default=0.0,
        metadata={
            "help": "Weight for bad-teacher contrastive auxiliary loss. "
            "0.0 disables contrastive loss (default)."
        },
    )
    contrastive_margin: float = field(
        default=0.5,
        metadata={
            "help": "Margin for contrastive hinge loss terms."
        },
    )

    # Final-layer hidden-state distillation
    final_layer_distill_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for auxiliary final-layer hidden-state MSE loss. "
            "0.0 disables it (default). Only active in distillation_final_layer mode."
        },
    )

    # Per-token JSD divergence clipping (caps high-divergence tokens)
    token_clip: Optional[float] = field(
        default=None,
        metadata={
            "help": "Per-token KL/JSD clip ceiling (static). None (default) = no clipping. "
            "When set (e.g. 10.0), per-token loss is clamp(max=token_clip) "
            "after vocab-reduction and before masking/weighting/final reduction. "
            "Mutually exclusive with token_clip_quantile."
        },
    )
    token_clip_quantile: Optional[float] = field(
        default=None,
        metadata={
            "help": "Per-token KL/JSD clip ceiling, computed as an empirical quantile "
            "(e.g. 0.99) of masked completion-token losses per batch. Mutually "
            "exclusive with token_clip. Mirrors the agreement_quantile pattern."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        if self.mode not in ALL_MODES:
            raise ValueError(
                f"mode must be one of {sorted(ALL_MODES)}, got '{self.mode}'"
            )
        if self.agreement_stat not in ("var", "range"):
            raise ValueError(
                f"agreement_stat must be 'var' or 'range', got '{self.agreement_stat}'"
            )
        if self.context_augment not in ("none", "drop_demo", "token_drop"):
            raise ValueError(
                f"context_augment must be 'none', 'drop_demo', or 'token_drop', "
                f"got '{self.context_augment}'"
            )
        if self.token_clip is not None and self.token_clip_quantile is not None:
            raise ValueError(
                "Set either token_clip or token_clip_quantile, not both."
            )
        if self.token_clip_quantile is not None and not (
            0.0 < self.token_clip_quantile <= 1.0
        ):
            raise ValueError(
                f"token_clip_quantile must be in (0, 1], "
                f"got {self.token_clip_quantile}"
            )
