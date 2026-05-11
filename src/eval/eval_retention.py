"""Retention-scoring helpers shared across MCQA / code / GSM8K eval scripts.

All logic in this module was extracted verbatim from ``src/eval/eval_mcqa.py``
to keep the per-task eval scripts focused on accuracy and to give the project
one canonical home for retention scoring (forward/reverse log-probability and
JSD between the BASE and FT models).

Public surface used by callers:

- ``RETENTION_SUMMARY_KEYS`` — summary-level retention keys (used by
  ``compute_resource_consumption_record`` for resource logging).
- ``resolve_base_path(args)`` — pick the BASE model path (LoRA auto-resolve
  or explicit ``--base_model_name_or_path``).
- ``load_generations_by_task_id(jsonl, task_ids, *, strict=True)`` — load
  cached completions matched by ``task_id``.
- ``run_all_retention_passes(args, prompts, ft_completions, base_completions)``
  — load BASE+FT, run forward/reverse logp+JSD, free models. Returns the
  retention dict consumed by the helpers below.
- ``load_retention_from_results(jsonl_path)`` — reconstruct the retention dict
  from a previously-written results JSONL (skip rescoring on re-runs).
- ``flatten_retention_into_record(record, retention, index)`` — add per-example
  retention columns to a result record.
- ``add_retention_to_summary(summary, retention)`` — add aggregate logp / PPL /
  JSD / paired-delta stats to a summary dict.
"""

from __future__ import annotations

import gc
import json
import os

import numpy as np
import torch

from src.utils.env_utils import get_torch_dtype  # noqa: F401 — env_utils import patches transformers for custom-code models
from transformers import AutoTokenizer
from src.utils.lora_utils import (
    is_lora_checkpoint,
    get_base_model_path,
    load_model_for_inference,
)
from src.utils.eval_utils import (
    score_completions_under_model,
    score_completions_jsd_between_models,
)


RETENTION_SUMMARY_KEYS: tuple[str, ...] = (
    "extraction_success_rate",
    # Forward retention: log p_base(y_ft | x)
    "retention_logp_mean_avg",
    "retention_logp_mean_std",
    "retention_ppl_mean_avg",
    "retention_num_examples",
    "retention_total_tokens",
    "retention_jsd_mean_avg",
    "retention_jsd_mean_std",
    "retention_jsd_total_tokens",
    # Reverse retention: log p_ft(y_base | x)
    "reverse_logp_mean_avg",
    "reverse_logp_mean_std",
    "reverse_ppl_mean_avg",
    "reverse_num_examples",
    "reverse_total_tokens",
    "reverse_jsd_mean_avg",
    "reverse_jsd_mean_std",
    "reverse_jsd_total_tokens",
    # Paired forward-vs-reverse deltas
    "paired_delta_logp_mean",
    "paired_delta_logp_median",
    "paired_delta_logp_win_rate",
    "paired_delta_logp_num_pairs",
    "paired_delta_jsd_mean",
    "paired_delta_jsd_median",
    "paired_delta_jsd_num_pairs",
)


_RETENTION_KEYS = ("forward_logp", "forward_jsd", "reverse_logp", "reverse_jsd")

_QUANTILES = [(0.10, "p10"), (0.25, "p25"), (0.50, "median"), (0.75, "p75"), (0.90, "p90")]

# (retention dict key, {raw_field: output_column}) for flattening into per-example records
_RETENTION_RECORD_MAP = [
    ("forward_logp", {"logp_mean": "retention_logp_mean", "logp_sum": "retention_logp_sum", "n_tokens": "retention_num_tokens"}),
    ("forward_jsd", {"jsd_mean": "retention_jsd_mean", "jsd_sum": "retention_jsd_sum", "n_tokens": "retention_jsd_num_tokens"}),
    ("reverse_logp", {"logp_mean": "reverse_logp_mean", "logp_sum": "reverse_logp_sum", "n_tokens": "reverse_num_tokens"}),
    ("reverse_jsd", {"jsd_mean": "reverse_jsd_mean", "jsd_sum": "reverse_jsd_sum", "n_tokens": "reverse_jsd_num_tokens"}),
]


def resolve_base_path(args) -> str:
    """Resolve the BASE model path for retention-style scoring.

    Prefers LoRA auto-resolution via ``get_base_model_path``; otherwise falls back
    to the explicit ``--base_model_name_or_path`` flag.
    """
    if is_lora_checkpoint(args.model_name_or_path):
        return get_base_model_path(args.model_name_or_path)
    if args.base_model_name_or_path:
        return args.base_model_name_or_path
    raise ValueError(
        "--compute_retention requires --base_model_name_or_path when the "
        "checkpoint is not a LoRA adapter."
    )


def load_generations_by_task_id(
    jsonl_path: str, task_ids: list[str], strict: bool = True,
) -> list[str | None]:
    """Load ``generated_text`` from a JSONL file, matched by ``task_id``.

    Reads all non-summary records, builds a ``{task_id: generated_text}`` map,
    and returns a list aligned with ``task_ids``.

    Args:
        strict: If True (default, for FT completions), raise ValueError when any
            task_id is missing. If False (for base completions), return None for
            missing entries and print a warning.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Generations JSONL not found: {jsonl_path}")

    gen_map: dict[str, str] = {}
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("__summary__"):
                continue
            tid = record.get("task_id")
            if tid is not None:
                gen_map[tid] = record.get("generated_text", "")

    missing = [tid for tid in task_ids if tid not in gen_map]
    if missing:
        if strict:
            raise ValueError(
                f"load_generations_by_task_id: {len(missing)} task_ids not found in "
                f"{jsonl_path}. First 5 missing: {missing[:5]}"
            )
        print(
            f"[load_generations_by_task_id] WARNING: {len(missing)}/{len(task_ids)} "
            f"task_ids not in {os.path.basename(jsonl_path)} (filtered dataset?). "
            f"Reverse scoring will skip those examples."
        )
    return [gen_map.get(tid) for tid in task_ids]


def run_all_retention_passes(
    args,
    prompts: list[str],
    ft_completions: list[str],
    base_completions: list[str] | None,
) -> dict:
    """Load both models once, run all retention scoring passes, then free them.

    Forward (on y_ft):
      - logp: log p_base(y_ft | x) — how natural y_ft looks to the base model.
      - jsd:  JSD(p_base, p_ft) on y_ft — symmetric token-level divergence.

    Reverse (on y_base, only when ``base_completions`` is not None):
      - logp: log p_ft(y_base | x) — how natural y_base looks to the FT model.
      - jsd:  JSD(p_base, p_ft) on y_base — symmetric divergence on base sequences.

    Returns a dict with keys ``"forward_logp"``, ``"forward_jsd"``,
    ``"reverse_logp"``, ``"reverse_jsd"``, each either a ``list[dict]`` or ``None``.
    """
    base_path = resolve_base_path(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = get_torch_dtype()
    max_length = args.retention_max_length or (args.max_prompt_length + args.max_new_tokens)

    # --- Optionally limit GPU memory ---
    gpu_mem_frac = getattr(args, "gpu_memory_fraction", None)
    if gpu_mem_frac is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_mem_frac)
        print(f"[retention] GPU memory fraction set to {gpu_mem_frac}")

    # --- Load both models once ---
    print(f"[retention] Loading BASE model: {base_path}")
    base_tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    base_model = load_model_for_inference(
        base_path, torch_dtype=torch_dtype, device_map=None,
    )
    base_model.to(device).eval()

    print(f"[retention] Loading EVAL (FT) model: {args.model_name_or_path}")
    try:
        ft_tok = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True,
        )
    except (OSError, ValueError):
        ft_tok = base_tok
    ft_model = load_model_for_inference(
        args.model_name_or_path, torch_dtype=torch_dtype, device_map=None,
    )
    ft_model.to(device).eval()

    # --- Forward passes (on y_ft) ---
    print(
        f"[retention/forward] Scoring {len(prompts)} FT completions "
        f"(batch={args.retention_batch_size}, max_len={max_length})..."
    )
    forward_logp = score_completions_under_model(
        base_model, base_tok, prompts, ft_completions,
        batch_size=args.retention_batch_size, max_length=max_length,
    )
    forward_jsd = score_completions_jsd_between_models(
        base_model, base_tok, ft_model, ft_tok, prompts, ft_completions,
        batch_size=args.retention_batch_size, max_length=max_length,
    )

    # --- Reverse passes (on y_base) ---
    reverse_logp = None
    reverse_jsd = None
    if base_completions is not None:
        # Replace None entries (missing task_ids) with "" so scorers treat them
        # as empty completions (n_tokens=0, logp/jsd=nan).
        base_filled = [c if c is not None else "" for c in base_completions]
        n_avail = sum(1 for c in base_completions if c is not None)
        print(
            f"[retention/reverse] Scoring {n_avail}/{len(prompts)} base completions "
            f"(batch={args.retention_batch_size}, max_len={max_length})..."
        )
        reverse_logp = score_completions_under_model(
            ft_model, ft_tok, prompts, base_filled,
            batch_size=args.retention_batch_size, max_length=max_length,
        )
        reverse_jsd = score_completions_jsd_between_models(
            base_model, base_tok, ft_model, ft_tok, prompts, base_filled,
            batch_size=args.retention_batch_size, max_length=max_length,
        )

    # --- Free both models ---
    del ft_model, base_model
    if ft_tok is not base_tok:
        del ft_tok
    del base_tok
    gc.collect()

    return {
        "forward_logp": forward_logp,
        "forward_jsd": forward_jsd,
        "reverse_logp": reverse_logp,
        "reverse_jsd": reverse_jsd,
    }


def load_retention_from_results(path: str) -> dict | None:
    """Try to load retention scores from an existing results JSONL.

    Returns a retention dict (same shape as run_all_retention_passes output)
    if the file exists and has retention columns, else None.
    """
    if not os.path.exists(path):
        return None

    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("__summary__"):
                continue
            records.append(rec)

    if not records or "retention_logp_mean" not in records[0]:
        return None

    # Reconstruct retention dict from flattened columns
    # Invert _RETENTION_RECORD_MAP: {output_col: (key, src_field)}
    inv = {}
    for key, col_map in _RETENTION_RECORD_MAP:
        for src, col in col_map.items():
            inv[col] = (key, src)

    retention = {k: [] for k in _RETENTION_KEYS}
    active_keys = set()
    for rec in records:
        per_key: dict[str, dict] = {k: {} for k in _RETENTION_KEYS}
        for col, (key, src) in inv.items():
            if col in rec and rec[col] is not None:
                per_key[key][src] = rec[col]
                active_keys.add(key)
        for k in _RETENTION_KEYS:
            retention[k].append(per_key[k] if per_key[k] else None)

    # Convert inactive keys to None
    for k in _RETENTION_KEYS:
        if k not in active_keys:
            retention[k] = None

    print(f"[retention] Loaded {len(records)} cached retention scores from: {path}")
    return retention


def flatten_retention_into_record(record: dict, retention: dict, index: int) -> None:
    """Add per-example retention columns to ``record`` (in place).

    Expands each pass in ``retention`` (forward_logp, forward_jsd, reverse_logp,
    reverse_jsd) into the columns defined in ``_RETENTION_RECORD_MAP``, and
    derives ``*_ppl_mean`` from any logp_mean entry with non-zero token count.
    """
    for key, col_map in _RETENTION_RECORD_MAP:
        if not retention[key]:
            continue
        rec = retention[key][index]
        record.update({col: rec[src] for src, col in col_map.items()})
        if "logp_mean" in rec and rec["n_tokens"] > 0:
            prefix = col_map["logp_mean"].rsplit("_logp_mean", 1)[0]
            record[f"{prefix}_ppl_mean"] = float(np.exp(-rec["logp_mean"]))


def _add_logp_summary(summary: dict, records: list[dict], prefix: str) -> None:
    valid = [r["logp_mean"] for r in records if r["n_tokens"] > 0]
    empty = sum(1 for r in records if r["n_tokens"] == 0)
    summary[f"{prefix}_logp_mean_avg"] = float(np.mean(valid)) if valid else None
    summary[f"{prefix}_logp_mean_std"] = float(np.std(valid)) if valid else None
    summary[f"{prefix}_num_examples"] = len(valid)
    summary[f"{prefix}_num_empty_completions"] = empty
    summary[f"{prefix}_total_tokens"] = int(sum(r["n_tokens"] for r in records))
    # Perplexity
    ppl_values = [np.exp(-v) for v in valid] if valid else []
    summary[f"{prefix}_ppl_mean_avg"] = float(np.mean(ppl_values)) if ppl_values else None
    # Quantiles
    if valid:
        arr = np.array(valid)
        ppl_arr = np.array(ppl_values)
        for q, label in _QUANTILES:
            summary[f"{prefix}_logp_mean_{label}"] = float(np.quantile(arr, q))
            summary[f"{prefix}_ppl_mean_{label}"] = float(np.quantile(ppl_arr, q))
    print(
        f"  {prefix}_logp_mean_avg: {summary[f'{prefix}_logp_mean_avg']} "
        f"(ppl={summary[f'{prefix}_ppl_mean_avg']}, "
        f"scored={len(valid)}, empty={empty}, "
        f"tokens={summary[f'{prefix}_total_tokens']})"
    )


def _add_jsd_summary(summary: dict, records: list[dict], prefix: str) -> None:
    valid = [r["jsd_mean"] for r in records if r["n_tokens"] > 0]
    empty = sum(1 for r in records if r["n_tokens"] == 0)
    summary[f"{prefix}_jsd_mean_avg"] = float(np.mean(valid)) if valid else None
    summary[f"{prefix}_jsd_mean_std"] = float(np.std(valid)) if valid else None
    summary[f"{prefix}_jsd_num_examples"] = len(valid)
    summary[f"{prefix}_jsd_num_empty_completions"] = empty
    summary[f"{prefix}_jsd_total_tokens"] = int(sum(r["n_tokens"] for r in records))
    # Quantiles
    if valid:
        arr = np.array(valid)
        for q, label in _QUANTILES:
            summary[f"{prefix}_jsd_mean_{label}"] = float(np.quantile(arr, q))
    print(
        f"  {prefix}_jsd_mean_avg: {summary[f'{prefix}_jsd_mean_avg']} nats "
        f"(scored={len(valid)}, empty={empty}, "
        f"tokens={summary[f'{prefix}_jsd_total_tokens']})"
    )


def _add_paired_logp_summary(
    summary: dict, forward_records: list[dict], reverse_records: list[dict],
    prefix: str = "paired",
) -> None:
    deltas = []
    for fwd, rev in zip(forward_records, reverse_records):
        if fwd["n_tokens"] > 0 and rev["n_tokens"] > 0:
            deltas.append(fwd["logp_mean"] - rev["logp_mean"])
    if not deltas:
        return
    arr = np.array(deltas)
    summary[f"{prefix}_delta_logp_mean"] = float(np.mean(arr))
    summary[f"{prefix}_delta_logp_median"] = float(np.median(arr))
    summary[f"{prefix}_delta_logp_win_rate"] = float(np.mean(arr > 0))
    summary[f"{prefix}_delta_logp_num_pairs"] = len(deltas)
    print(
        f"  {prefix}_delta_logp: mean={summary[f'{prefix}_delta_logp_mean']:.4f}, "
        f"median={summary[f'{prefix}_delta_logp_median']:.4f}, "
        f"win_rate={summary[f'{prefix}_delta_logp_win_rate']:.3f} "
        f"(n={len(deltas)})"
    )


def _add_paired_jsd_summary(
    summary: dict, forward_records: list[dict], reverse_records: list[dict],
    prefix: str = "paired",
) -> None:
    deltas = []
    for fwd, rev in zip(forward_records, reverse_records):
        if fwd["n_tokens"] > 0 and rev["n_tokens"] > 0:
            deltas.append(fwd["jsd_mean"] - rev["jsd_mean"])
    if not deltas:
        return
    arr = np.array(deltas)
    summary[f"{prefix}_delta_jsd_mean"] = float(np.mean(arr))
    summary[f"{prefix}_delta_jsd_median"] = float(np.median(arr))
    summary[f"{prefix}_delta_jsd_num_pairs"] = len(deltas)
    print(
        f"  {prefix}_delta_jsd: mean={summary[f'{prefix}_delta_jsd_mean']:.4f}, "
        f"median={summary[f'{prefix}_delta_jsd_median']:.4f} "
        f"(n={len(deltas)})"
    )


def add_retention_to_summary(summary: dict, retention: dict) -> None:
    """Populate ``summary`` (in place) with aggregate retention statistics.

    Adds logp/PPL/JSD means, stds, quantiles (p10/p25/median/p75/p90), and
    paired forward-vs-reverse deltas (mean / median / win_rate / num_pairs)
    for whichever passes are present in ``retention``.
    """
    if retention["forward_logp"]:
        _add_logp_summary(summary, retention["forward_logp"], "retention")
    if retention["forward_jsd"]:
        _add_jsd_summary(summary, retention["forward_jsd"], "retention")
    if retention["reverse_logp"]:
        _add_logp_summary(summary, retention["reverse_logp"], "reverse")
    if retention["reverse_jsd"]:
        _add_jsd_summary(summary, retention["reverse_jsd"], "reverse")
    if retention["forward_logp"] and retention["reverse_logp"]:
        _add_paired_logp_summary(summary, retention["forward_logp"], retention["reverse_logp"])
    if retention["forward_jsd"] and retention["reverse_jsd"]:
        _add_paired_jsd_summary(summary, retention["forward_jsd"], retention["reverse_jsd"])
