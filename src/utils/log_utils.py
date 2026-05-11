"""Logging utilities: colored stdout printing and HF Trainer log_history parsing.

`trainer.state.log_history` is a list of dicts: one entry per `logging_steps`
plus a final summary appended by HF Trainer with keys like `train_runtime`,
`train_samples_per_second`, `train_steps_per_second`, `train_loss`.

`extract_final_train_metrics` surfaces the most recent value seen for each of
a curated key set so the post-training resource record can include them
directly, instead of having the analysis pipeline regex-scrape stdout logs.

Slash-style metric keys (e.g. ``agreement/weight_mean``) are flattened to
underscores so they are friendly downstream (Excel column names, dataframes).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def print_colored(text, color):
    # Define ANSI color codes
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m"  # Reset to default color
    }

    # Get the color code from the dictionary (default to reset if not found)
    color_code = colors.get(color.lower(), colors["reset"])

    # Print the colored text
    print(f"{color_code}{text}{colors['reset']}")


TRAIN_METRIC_KEYS: tuple[str, ...] = (
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "train_loss",
    "num_tokens",
    "loss",
    "grad_norm",
    "learning_rate",
    "entropy",
    "mean_token_accuracy",
    "kl_approx",
    "kl_to_base",
    "agreement/weight_mean",
    "agreement/weight_min",
    "agreement/weight_max",
    "agreement/disagreement_mean",
    "contrastive/loss",
    "contrastive/pos_dist_mean",
    "contrastive/neg_dist_mean",
    "contrastive/margin_violation_mean",
    "final_layer/loss",
)


def _flatten_key(k: str) -> str:
    return k.replace("/", "_")


def extract_final_train_metrics(
    log_history: Iterable[Mapping[str, Any]] | None,
    keys: Iterable[str] = TRAIN_METRIC_KEYS,
) -> dict[str, Any]:
    """Walk log_history once; return the most recent non-None value per key.

    Keys absent from every entry (e.g. agreement/* in single-teacher mode) are
    simply omitted from the result — no special-casing needed downstream.
    """
    out: dict[str, Any] = {}
    if not log_history:
        return out
    for entry in log_history:
        if not isinstance(entry, Mapping):
            continue
        for k in keys:
            v = entry.get(k)
            if v is not None:
                out[_flatten_key(k)] = v
    return out
