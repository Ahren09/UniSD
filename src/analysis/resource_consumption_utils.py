"""Lightweight resource consumption *estimates* for training and inference runs.

These are estimates derived from runtime + GPU TDP assumptions, not measured
datacenter values. Primary metrics: GPU-hours and estimated kWh. Secondary:
estimated kgCO2e.

Core metric functions are pure (no I/O). The only I/O helper in this module is
``append_resource_record``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional


def compute_gpu_hours(wall_time_sec: float, num_gpus: int) -> float:
    return wall_time_sec / 3600.0 * num_gpus


def estimate_energy_kwh(
    wall_time_sec: float,
    num_gpus: int,
    gpu_tdp_w: float = 700,
    gpu_utilization: float = 0.7,
    pue: float = 1.2,
) -> float:
    return (
        wall_time_sec / 3600.0
        * num_gpus
        * gpu_tdp_w / 1000.0
        * gpu_utilization
        * pue
    )


def estimate_kgco2e(energy_kwh: float, carbon_intensity_g_per_kwh: float = 475) -> float:
    return energy_kwh * carbon_intensity_g_per_kwh / 1000.0


def detect_num_gpus() -> int:
    """WORLD_SIZE > torch.cuda.device_count() > 1. Ignore non-positive / non-numeric WORLD_SIZE."""
    raw = os.environ.get("WORLD_SIZE")
    if raw is not None:
        try:
            n = int(raw)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if n > 0:
                return n
    except Exception:
        pass
    return 1


def compute_resource_consumption_record(
    method: str,
    model: str,
    dataset: str,
    phase: str,
    wall_time_sec: float,
    num_gpus: int,
    gpu_type: str = "H200-SXM",
    gpu_tdp_w: float = 700,
    gpu_utilization: float = 0.7,
    pue: float = 1.2,
    carbon_intensity_g_per_kwh: float = 475,
    peak_gpu_mem_gb: Optional[float] = None,
    total_tokens: Optional[int] = None,
    **extra_fields: Any,
) -> dict:
    """Build one resource-consumption record. Pure: no I/O.

    On wall_time_sec==0: gpu_hours/kwh/kgCO2e are 0 (formulas yield 0). Per-token
    ratios become None to avoid div-by-zero.
    On total_tokens==0 or None: per-token ratios are None; absolute metrics
    unaffected. Never raises.
    """
    wall_time_hours = wall_time_sec / 3600.0
    gpu_hours = compute_gpu_hours(wall_time_sec, num_gpus)
    estimated_kwh = estimate_energy_kwh(
        wall_time_sec, num_gpus, gpu_tdp_w, gpu_utilization, pue,
    )
    estimated_kgco2e = estimate_kgco2e(estimated_kwh, carbon_intensity_g_per_kwh)

    tokens_per_sec: Optional[float] = None
    kwh_per_1k_tokens: Optional[float] = None
    gco2e_per_1k_tokens: Optional[float] = None
    if total_tokens is not None and total_tokens > 0 and wall_time_sec > 0:
        tokens_per_sec = total_tokens / wall_time_sec
        kwh_per_1k_tokens = estimated_kwh / total_tokens * 1000.0
        # gCO2e per 1k tokens = kgCO2e * 1000 (g/kg) / total_tokens * 1000 (per 1k)
        gco2e_per_1k_tokens = estimated_kgco2e * 1000.0 / total_tokens * 1000.0

    computed = {
        "method": method,
        "model": model,
        "dataset": dataset,
        "phase": phase,
        "wall_time_sec": wall_time_sec,
        "wall_time_hours": wall_time_hours,
        "num_gpus": num_gpus,
        "gpu_type": gpu_type,
        "gpu_tdp_w": gpu_tdp_w,
        "gpu_utilization": gpu_utilization,
        "pue": pue,
        "carbon_intensity_g_per_kwh": carbon_intensity_g_per_kwh,
        "gpu_hours": gpu_hours,
        "estimated_kwh": estimated_kwh,
        "estimated_kgco2e": estimated_kgco2e,
        "peak_gpu_mem_gb": peak_gpu_mem_gb,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "kwh_per_1k_tokens": kwh_per_1k_tokens,
        "gco2e_per_1k_tokens": gco2e_per_1k_tokens,
    }
    # extra_fields first, computed wins on collision
    return {**extra_fields, **computed}


def append_resource_record(
    record: dict,
    jsonl_path: str = "outputs/resource_consumption_records.jsonl",
) -> None:
    """Append one JSON line. Caller wraps in try/except; this function may raise."""
    parent = os.path.dirname(jsonl_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
