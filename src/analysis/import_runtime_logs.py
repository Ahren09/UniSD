"""Backfill resource_consumption records from existing training .log files.

Many older training runs already wrote ``train_runtime`` and ``Peak GPU memory
allocated`` to their stdout logs (HuggingFace Trainer's final summary).
This script scans ``logs/**/*.log`` and emits one JSONL record per completed
run, so we can build the resource-consumption table without re-running anything.

Usage:
    python -m src.analysis.import_runtime_logs \
        --logs_glob "logs/**/*.log" \
        --output_jsonl outputs/resource_consumption_records_imported.jsonl

Then aggregate:
    python -m src.analysis.measure_resource_consumption \
        --input_log_jsonl outputs/resource_consumption_records_imported.jsonl
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
from typing import Optional


# Final summary line printed by HF Trainer / TRL trainer.
TRAIN_RUNTIME_RE = re.compile(r"\{'train_runtime':\s*[\d.]+,.*?\}")
PEAK_MEM_RE = re.compile(r"Peak GPU memory allocated:\s*([\d.]+)\s*GB")
# "Output dir: outputs/method/dataset[/hparam]/model"

OUTPUT_DIR_RE = re.compile(r"Output dir(?:ectory)?:\s*(outputs/\S+)")


def _parse_path(rel_path: str) -> dict:
    """Parse 'outputs/<method>/<dataset>/[<hparam>/]<model>' into components."""
    parts = rel_path.strip().rstrip("/").split("/")
    # drop leading 'outputs'
    if parts and parts[0] == "outputs":
        parts = parts[1:]
    out = {"method": "unknown", "dataset": "unknown", "model": "unknown", "hparam_tag": None}
    if len(parts) == 3:
        out["method"], out["dataset"], out["model"] = parts
    elif len(parts) == 4:
        out["method"], out["dataset"], out["hparam_tag"], out["model"] = parts
    elif len(parts) == 5:
        # e.g. outputs/method/dataset/hparam/extra/model
        out["method"], out["dataset"], out["hparam_tag"], _, out["model"] = parts
    elif len(parts) >= 2:
        out["method"], out["dataset"] = parts[0], parts[1]
        out["model"] = parts[-1]
        if len(parts) > 3:
            out["hparam_tag"] = "/".join(parts[2:-1])
    return out


def _parse_runtime_dict(line: str) -> Optional[dict]:
    """Parse the {'train_runtime': ...} line as a Python literal."""
    m = TRAIN_RUNTIME_RE.search(line)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(0))
    except Exception:
        return None


def _path_components_from_train_log_path(path: str) -> dict | None:
    """For files at outputs/<method>/<dataset>/[<hparam>/]<model>/train.log, derive components."""
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


def _scan_log(path: str, default_num_gpus: int) -> list[dict]:
    """Extract one record per completed run in a single .log file."""
    try:
        with open(path, errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[import] WARNING: cannot read {path}: {e}")
        return []

    # Path-based components (used as fallback / preferred for outputs/**/train.log).
    path_comp = _path_components_from_train_log_path(path)

    records: list[dict] = []
    for i, line in enumerate(lines):
        rt = _parse_runtime_dict(line)
        if rt is None:
            continue

        # Look ahead up to 15 lines for peak mem and output dir.
        peak_mem = None
        output_dir = None
        window = "".join(lines[i + 1 : i + 1 + 15])
        m = PEAK_MEM_RE.search(window)
        if m:
            peak_mem = float(m.group(1))
        m = OUTPUT_DIR_RE.search(window)
        if m:
            output_dir = m.group(1)

        # Look behind up to 5 lines for the most recent num_tokens (per-step log).
        total_tokens = None
        behind = "".join(lines[max(0, i - 5) : i])
        m = re.search(r"'num_tokens':\s*([\d.]+)", behind)
        if m:
            try:
                total_tokens = int(float(m.group(1)))
            except Exception:
                total_tokens = None

        # Prefer path-based components when available; else fall back to "Output dir:" parse.
        comp = path_comp if path_comp is not None else (
            _parse_path(output_dir) if output_dir else {
                "method": "unknown", "dataset": "unknown", "model": "unknown", "hparam_tag": None,
            }
        )
        rec = {
            "method": comp["method"],
            "model": comp["model"],
            "dataset": comp["dataset"],
            "phase": "train",
            "wall_time_sec": float(rt["train_runtime"]),
            "num_gpus": default_num_gpus,
            "peak_gpu_mem_gb": peak_mem,
            "total_tokens": total_tokens,
            # carry-through context (lands in **extra_fields)
            "hparam_tag": comp["hparam_tag"],
            "train_samples_per_second": rt.get("train_samples_per_second"),
            "train_steps_per_second": rt.get("train_steps_per_second"),
            "train_loss": rt.get("train_loss"),
            "source_log": os.path.relpath(path),
        }
        records.append(rec)
    return records


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs_glob", default="logs/**/*.log",
                   help="comma-separated glob patterns (recursive). "
                        "Default scans logs/**/*.log AND outputs/**/train.log.")
    p.add_argument("--output_jsonl", default="outputs/resource_consumption_records_imported.jsonl")
    p.add_argument("--num_gpus", type=int, default=1, help="default num_gpus for imported records")
    return p.parse_args()


def main():
    args = parse_args()
    # Always include outputs/**/train.log (where the runner writes per-job logs).
    globs = [g.strip() for g in args.logs_glob.split(",") if g.strip()]
    if "outputs/**/train.log" not in globs:
        globs.append("outputs/**/train.log")
    paths = []
    for g in globs:
        paths.extend(glob.glob(g, recursive=True))
    paths = sorted(set(paths))
    print(f"[import] Scanning {len(paths)} log files (globs: {globs})")

    all_records: list[dict] = []
    for p in paths:
        recs = _scan_log(p, default_num_gpus=args.num_gpus)
        if recs:
            print(f"  {p}: {len(recs)} run(s)")
            all_records.extend(recs)

    if not all_records:
        print("[import] No completed runs found in logs.")
        return

    parent = os.path.dirname(args.output_jsonl)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output_jsonl, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"\n[import] Wrote {len(all_records)} records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
