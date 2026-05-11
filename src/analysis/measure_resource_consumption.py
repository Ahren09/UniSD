"""Aggregate resource consumption + carbon footprint into one XLSX summary.

Merges three previously-separate scripts:
  - log scanning (backfill `train_runtime` / peak-mem / `num_tokens` from .log files)
  - resource record aggregation (GPU-hours, kWh, kgCO2e via resource_consumption_utils)
  - paper-ready Ours breakdown

Output: a single multi-sheet `outputs/resource_consumption.xlsx` summarizing
every (method, model, dataset, phase) combination found in the live JSONL
records and/or training logs.

Numbers are *estimates* from runtime + GPU TDP, not measured datacenter values.

Usage:
    python -m src.analysis.measure_resource_consumption \\
        [--input_jsonl outputs/resource_consumption_records.jsonl] \\
        [--logs_glob "logs/**/*.log,outputs/**/train.log"] \\
        [--no-scan-logs] \\
        [--output_xlsx outputs/resource_consumption.xlsx]
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

# Allow running both as a module (-m src.analysis.measure_resource_consumption)
# and as a script.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.resource_consumption_utils import compute_resource_consumption_record


# ── Family tagging (Ours vs Other) ──
OUR_METHOD = {
    "all",
    "induction", "agreement_tok_induction",
    "fewshot_random", "agreement_tok_random",
    "fewshot_retrieval", "agreement_tok_retrieval",
    "teacher_ema",
    "token_cl",
    "distillation_final_layer",
    "mse_final_layer",
    "clip", "jsd_clip",
}
DROP_DATASETS = {"bak_scienceqa-all_before_debug_token_cl", "unknown"}
DROP_METHODS = {"unknown"}


# ── Log-scanning regexes (HF Trainer / TRL summary lines) ──
TRAIN_RUNTIME_RE = re.compile(r"\{'train_runtime':\s*[\d.]+,.*?\}")
PEAK_MEM_RE = re.compile(r"Peak GPU memory allocated:\s*([\d.]+)\s*GB")
OUTPUT_DIR_RE = re.compile(r"Output dir(?:ectory)?:\s*(outputs/\S+)")
NUM_TOKENS_RE = re.compile(r"'num_tokens':\s*([\d.]+)")


# ── Display column orders ──
PER_RUN_COLS = [
    "family", "method", "model", "dataset", "phase",
    "num_gpus", "wall_time_hours", "gpu_hours",
    "estimated_kwh", "estimated_kgco2e",
    "peak_gpu_mem_gb",
    "total_tokens", "tokens_per_sec", "kwh_per_1k_tokens",
]

# Direct-logged columns surfaced in PerRun when present (training/eval write
# these via extra_fields at the end of each run; see src/utils/log_utils.py
# and the eval scripts' _RETENTION_RECORD_KEYS). Absent for older records.
DIRECT_LOG_COLS = [
    # Training summary (from trainer.state.log_history)
    "train_runtime", "train_samples_per_second", "train_steps_per_second",
    "train_loss", "train_ppl", "entropy", "kl_approx",
    # UniSD component metrics
    "agreement_weight_mean", "agreement_disagreement_mean",
    "contrastive_loss", "final_layer_loss",
    # Eval retention PPL
    "retention_logp_mean_avg", "retention_ppl_mean_avg", "retention_total_tokens",
    "reverse_logp_mean_avg", "reverse_ppl_mean_avg",
    "paired_delta_logp_mean", "paired_delta_logp_win_rate",
    # Eval task score
    "score",
]


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_jsonl", default="outputs/resource_consumption_records.jsonl")
    p.add_argument(
        "--logs_glob",
        default="logs/**/*.log,outputs/**/train.log",
        help="comma-separated recursive globs to scan when --scan_logs is set",
    )
    p.add_argument(
        "--scan-logs", dest="scan_logs", action=argparse.BooleanOptionalAction, default=False,
        help="Backfill records by regex-scanning training stdout logs. OFF by default: "
             "training and eval scripts now log directly via append_resource_record. "
             "Enable for archival runs that predate direct logging.",
    )
    p.add_argument("--output_xlsx", default="outputs/resource_consumption.xlsx")
    p.add_argument("--gpu_type", default="H200-SXM")
    p.add_argument("--gpu_tdp_w", type=float, default=700)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--gpu_utilization", type=float, default=0.7)
    p.add_argument("--pue", type=float, default=1.2)
    p.add_argument("--carbon_intensity_g_per_kwh", type=float, default=475)
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Live JSONL records
# ────────────────────────────────────────────────────────────────────────────

def _load_live_records(jsonl_path: str) -> list[dict]:
    if not os.path.exists(jsonl_path):
        print(f"[resource] live JSONL does not exist: {jsonl_path}")
        return []
    records: list[dict] = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[resource] skipping malformed line {i} in {jsonl_path}: {e}")
    print(f"[resource] live records loaded: {len(records)}  (from {jsonl_path})")
    return records


# ────────────────────────────────────────────────────────────────────────────
# Log-scanning backfill
# ────────────────────────────────────────────────────────────────────────────

def _parse_path(rel_path: str) -> dict:
    """Parse 'outputs/<method>/<dataset>/[<hparam>/]<model>' into components."""
    parts = rel_path.strip().rstrip("/").split("/")
    if parts and parts[0] == "outputs":
        parts = parts[1:]
    out = {"method": "unknown", "dataset": "unknown", "model": "unknown", "hparam_tag": None}
    if len(parts) == 3:
        out["method"], out["dataset"], out["model"] = parts
    elif len(parts) == 4:
        out["method"], out["dataset"], out["hparam_tag"], out["model"] = parts
    elif len(parts) == 5:
        out["method"], out["dataset"], out["hparam_tag"], _, out["model"] = parts
    elif len(parts) >= 2:
        out["method"], out["dataset"] = parts[0], parts[1]
        out["model"] = parts[-1]
        if len(parts) > 3:
            out["hparam_tag"] = "/".join(parts[2:-1])
    return out


def _path_components_from_train_log_path(path: str) -> dict | None:
    """For files at outputs/<method>/<dataset>/[<hparam>/]<model>/train.log."""
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)
    try:
        idx = parts.index("outputs")
    except ValueError:
        return None
    if parts[-1] != "train.log":
        return None
    inner = parts[idx + 1 : -1]  # method, dataset, [hparam], model
    out = {"method": "unknown", "dataset": "unknown", "model": "unknown", "hparam_tag": None}
    if len(inner) == 3:
        out["method"], out["dataset"], out["model"] = inner
    elif len(inner) == 4:
        out["method"], out["dataset"], out["hparam_tag"], out["model"] = inner
    elif len(inner) >= 5:
        out["method"], out["dataset"] = inner[0], inner[1]
        out["model"] = inner[-1]
        out["hparam_tag"] = "/".join(inner[2:-1])
    elif len(inner) == 2:
        out["method"], out["model"] = inner
    else:
        return None
    return out


def _parse_runtime_dict(line: str) -> dict | None:
    m = TRAIN_RUNTIME_RE.search(line)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(0))
    except Exception:
        return None


def _scan_one_log(path: str, default_num_gpus: int) -> list[dict]:
    """Extract one record per completed run in a single .log file."""
    try:
        with open(path, errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[resource] cannot read {path}: {e}")
        return []

    path_comp = _path_components_from_train_log_path(path)

    records: list[dict] = []
    for i, line in enumerate(lines):
        rt = _parse_runtime_dict(line)
        if rt is None:
            continue

        # Look ahead ≤15 lines for peak mem and Output dir.
        peak_mem = None
        output_dir = None
        window = "".join(lines[i + 1 : i + 1 + 15])
        m = PEAK_MEM_RE.search(window)
        if m:
            peak_mem = float(m.group(1))
        m = OUTPUT_DIR_RE.search(window)
        if m:
            output_dir = m.group(1)

        # Look behind ≤5 lines for the most recent num_tokens (per-step log).
        total_tokens = None
        behind = "".join(lines[max(0, i - 5) : i])
        m = NUM_TOKENS_RE.search(behind)
        if m:
            try:
                total_tokens = int(float(m.group(1)))
            except Exception:
                total_tokens = None

        comp = path_comp if path_comp is not None else (
            _parse_path(output_dir) if output_dir else {
                "method": "unknown", "dataset": "unknown", "model": "unknown", "hparam_tag": None,
            }
        )
        records.append({
            "method": comp["method"],
            "model": comp["model"],
            "dataset": comp["dataset"],
            "phase": "train",
            "wall_time_sec": float(rt["train_runtime"]),
            "num_gpus": default_num_gpus,
            "peak_gpu_mem_gb": peak_mem,
            "total_tokens": total_tokens,
            "hparam_tag": comp["hparam_tag"],
            "train_samples_per_second": rt.get("train_samples_per_second"),
            "train_steps_per_second": rt.get("train_steps_per_second"),
            "train_loss": rt.get("train_loss"),
            "source_log": os.path.relpath(path),
        })
    return records


def _scan_logs(globs_csv: str, default_num_gpus: int) -> list[dict]:
    globs = [g.strip() for g in globs_csv.split(",") if g.strip()]
    paths: list[str] = []
    for g in globs:
        paths.extend(glob.glob(g, recursive=True))
    paths = sorted(set(paths))
    print(f"[resource] scanning {len(paths)} log file(s) (globs: {globs})")

    all_records: list[dict] = []
    for p in paths:
        recs = _scan_one_log(p, default_num_gpus=default_num_gpus)
        if recs:
            all_records.extend(recs)
    print(f"[resource] backfilled records from logs: {len(all_records)}")
    return all_records


# ────────────────────────────────────────────────────────────────────────────
# Dedup + DataFrame build
# ────────────────────────────────────────────────────────────────────────────

def _record_key(r: dict) -> tuple:
    return (
        r.get("method", ""),
        r.get("model", ""),
        r.get("dataset", ""),
        r.get("phase", ""),
        r.get("hparam_tag", ""),
    )


def _dedup_prefer_live(live: list[dict], imported: list[dict]) -> list[dict]:
    """Live records win; imported records fill gaps for keys not in live."""
    live_keys = {_record_key(r) for r in live}
    extras = [r for r in imported if _record_key(r) not in live_keys]
    return live + extras


def _row_value(row: dict, key: str, default):
    v = row.get(key)
    return v if v is not None else default


def _build_dataframe(records: list[dict], cli_args) -> pd.DataFrame:
    rows = []
    for raw in records:
        method = raw.get("method", "unknown")
        if method in DROP_METHODS or raw.get("dataset") in DROP_DATASETS:
            continue

        wall_time_sec = float(raw.get("wall_time_sec", 0) or 0)
        num_gpus = int(_row_value(raw, "num_gpus", cli_args.num_gpus))
        gpu_type = _row_value(raw, "gpu_type", cli_args.gpu_type)
        gpu_tdp_w = float(_row_value(raw, "gpu_tdp_w", cli_args.gpu_tdp_w))
        gpu_utilization = float(_row_value(raw, "gpu_utilization", cli_args.gpu_utilization))
        pue = float(_row_value(raw, "pue", cli_args.pue))
        carbon = float(_row_value(raw, "carbon_intensity_g_per_kwh", cli_args.carbon_intensity_g_per_kwh))

        # Reserved keys are popped; the rest flow through **extra_fields.
        reserved = {
            "method", "model", "dataset", "phase", "wall_time_sec", "num_gpus",
            "gpu_type", "gpu_tdp_w", "gpu_utilization", "pue",
            "carbon_intensity_g_per_kwh", "peak_gpu_mem_gb", "total_tokens",
        }
        extras = {k: v for k, v in raw.items() if k not in reserved}

        rec = compute_resource_consumption_record(
            method=method,
            model=raw.get("model", "unknown"),
            dataset=raw.get("dataset", "unknown"),
            phase=raw.get("phase", "unknown"),
            wall_time_sec=wall_time_sec,
            num_gpus=num_gpus,
            gpu_type=gpu_type, gpu_tdp_w=gpu_tdp_w,
            gpu_utilization=gpu_utilization, pue=pue,
            carbon_intensity_g_per_kwh=carbon,
            peak_gpu_mem_gb=raw.get("peak_gpu_mem_gb"),
            total_tokens=raw.get("total_tokens"),
            **extras,
        )
        rec["family"] = (
            "Ours" if method in OUR_METHOD
            else "Other"
        )
        rows.append(rec)

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# Aggregations
# ────────────────────────────────────────────────────────────────────────────

def _by_method(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method totals across all datasets/runs, with per-token efficiency."""
    g = df.groupby(["family", "method"], as_index=False).agg(
        n_runs=("method", "size"),
        wall_time_hours=("wall_time_hours", "sum"),
        gpu_hours=("gpu_hours", "sum"),
        estimated_kwh=("estimated_kwh", "sum"),
        estimated_kgco2e=("estimated_kgco2e", "sum"),
        peak_gpu_mem_gb_max=("peak_gpu_mem_gb", "max"),
    )
    valid = df.dropna(subset=["total_tokens"])
    valid = valid[valid["total_tokens"] > 0]
    if len(valid):
        per_tok = valid.groupby(["family", "method"], as_index=False).agg(
            tokens_sum=("total_tokens", "sum"),
            kwh_sum=("estimated_kwh", "sum"),
            kgco2e_sum=("estimated_kgco2e", "sum"),
            gpu_hours_sum_v=("gpu_hours", "sum"),
        )
        per_tok["kwh_per_1m_tokens"] = per_tok["kwh_sum"] / per_tok["tokens_sum"] * 1e6
        per_tok["kgco2e_per_1m_tokens"] = per_tok["kgco2e_sum"] / per_tok["tokens_sum"] * 1e6
        per_tok["tokens_per_gpu_hour"] = per_tok["tokens_sum"] / per_tok["gpu_hours_sum_v"]
        per_tok = per_tok[["family", "method", "kwh_per_1m_tokens",
                           "kgco2e_per_1m_tokens", "tokens_per_gpu_hour"]]
        g = g.merge(per_tok, on=["family", "method"], how="left")
    return g.sort_values(["family", "gpu_hours"], ascending=[True, False]).reset_index(drop=True)


def _pivot_method_model(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df.pivot_table(index="method", columns="model", values=value_col, aggfunc="sum")
        .fillna(0.0)
        .round(4)
    )


# ────────────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    live = _load_live_records(args.input_jsonl)
    imported = _scan_logs(args.logs_glob, default_num_gpus=args.num_gpus) if args.scan_logs else []
    records = _dedup_prefer_live(live, imported)
    print(f"[resource] total deduped records: {len(records)}")

    if not records:
        print("[resource] no records found; nothing to write.")
        return

    df = _build_dataframe(records, args)
    if df.empty:
        print("[resource] all records were dropped (DROP_METHODS / DROP_DATASETS); nothing to write.")
        return

    by_method = _by_method(df).round({
        "wall_time_hours": 2, "gpu_hours": 2,
        "estimated_kwh": 2, "estimated_kgco2e": 3,
        "peak_gpu_mem_gb_max": 1,
        "kwh_per_1m_tokens": 4, "kgco2e_per_1m_tokens": 4, "tokens_per_gpu_hour": 0,
    })
    ours_by = by_method[by_method["family"] == "Ours"].drop(columns=["family"]).reset_index(drop=True)

    per_run_cols = [c for c in (PER_RUN_COLS + DIRECT_LOG_COLS) if c in df.columns]
    per_run = (
        df[per_run_cols]
        .sort_values(["family", "method", "model", "dataset"])
        .round({
            "wall_time_hours": 2, "gpu_hours": 2,
            "estimated_kwh": 2, "estimated_kgco2e": 3,
            "peak_gpu_mem_gb": 1,
            "tokens_per_sec": 1, "kwh_per_1k_tokens": 6,
        })
        .reset_index(drop=True)
    )

    out_path = args.output_xlsx
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        per_run.to_excel(w, sheet_name="PerRun", index=False)
        by_method.to_excel(w, sheet_name="ByMethod", index=False)
        ours_by.to_excel(w, sheet_name="OursByMethod", index=False)
        _pivot_method_model(df, "gpu_hours").to_excel(w, sheet_name="GPU-hours pivot")
        _pivot_method_model(df, "estimated_kwh").to_excel(w, sheet_name="kWh pivot")
        _pivot_method_model(df, "estimated_kgco2e").to_excel(w, sheet_name="kgCO2e pivot")

    print(
        f"[resource] wrote {out_path}  "
        f"({len(per_run)} runs, {len(by_method)} methods, "
        f"total {df['gpu_hours'].sum():.2f} GPU-h / "
        f"{df['estimated_kwh'].sum():.2f} kWh / "
        f"{df['estimated_kgco2e'].sum():.3f} kgCO2e)"
    )


if __name__ == "__main__":
    main()
