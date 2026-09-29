"""Standardized EdgeAIBus evaluation KPIs.

Loads a per-step ``states.csv`` written by the runner (or by the legacy
``eval_sim.py``) and computes the metrics used in the paper comparison table:

- ``moves`` / ``moves_per_step``           container migrations
- ``model_switches`` / ``switches_per_step``  YOLO model version switches
- ``server_faults`` / ``fault_steps_frac``  nodes > 85% CPU (requests-based)
- ``mean_accuracy``                        mean YOLO accuracy over steps
- ``mean_cluster_cpu_util``                mean active-node CPU utilization
- ``conserved_cores`` / ``conserved_per_step``  sum of allocatable CPU on empty hosts
- ``consolidated_count`` / ``consolidated_per_step`` cumulative+per-step empty hosts
- ``sla_violations_frac`` / ``sla_violations_pct`` per-step SLA violation fraction
- ``oversub_cores`` / ``oversub_per_step``
- reward components (sla / accuracy / illegal / consolidation / total)

It writes ``metrics.json`` (all scalars, units documented) and ``metrics.csv``
(one row per step for the drift-check columns).
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import pandas as pd


def _col(df: pd.DataFrame, names) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute scalar KPIs from a per-step states DataFrame."""
    out: Dict[str, float] = {}

    n = float(len(df))
    out["steps"] = n

    def summed(names, scale=1.0):
        c = _col(df, names)
        return 0.0 if c is None else float(df[c].sum()) * scale

    def meaned(names, scale=1.0):
        c = _col(df, names)
        return 0.0 if c is None else float(df[c].astype(float).mean()) * scale

    # Migrations / model switching
    out["moves"] = summed(["num_moves"])
    out["moves_per_step"] = out["moves"] / n if n else 0.0
    out["model_switches"] = summed(["num_model_switches"])
    out["switches_per_step"] = out["model_switches"] / n if n else 0.0

    # Overload / server fault (>85% request-based CPU utilization)
    out["server_faults"] = summed(["num_overloaded"])
    overload_col = _col(df, ["num_overloaded"])
    out["fault_steps_frac"] = float((df[overload_col].astype(float) > 0).mean()) if (overload_col and n) else 0.0

    # Accuracy and utilization
    out["mean_accuracy"] = meaned(["mean_accuracy"])
    out["mean_cluster_cpu_util"] = meaned(["mean_cluster_cpu_util"])
    out["mean_cluster_mem_util"] = meaned(["mean_cluster_mem_util"])

    # Consolidation / conserved cores
    out["conserved_cores"] = summed(["cpu_conserved_cost"])
    out["conserved_per_step"] = out["conserved_cores"] / n if n else 0.0
    out["consolidated_count"] = summed(["num_consolidated"])
    out["consolidated_per_step"] = out["consolidated_count"] / n if n else 0.0

    # SLA violations (fraction of containers violating 800 ms, 0..1 per step)
    sla_col = _col(df, ["num_slav", "sla_violations_frac"])
    sla_mean = float(df[sla_col].astype(float).mean()) if sla_col else 0.0
    out["sla_violations_frac"] = sla_mean
    out["sla_violations_pct"] = sla_mean * 100.0

    # Oversubscription
    out["oversub_cores"] = summed(["oversub_cores"])
    out["oversub_per_step"] = out["oversub_cores"] / n if n else 0.0

    # Rewards
    out["reward_total"] = summed(["reward"])
    out["reward_sla"] = summed(["reward_sla"])
    out["reward_accuracy"] = summed(["reward_accuracy"])
    out["reward_illegal"] = summed(["reward_illegal"])
    out["reward_consolidation"] = summed(["reward_consolidation"])

    # Inference-time latency when recorded (per step mean)
    if _col(df, ["inference_time_ms"]):
        out["mean_inference_time_ms"] = meaned(["inference_time_ms"])
    return out


def save_metrics(results_dir: str, df: pd.DataFrame, prefix: str = "run") -> Dict[str, float]:
    """Compute + persist metrics.json/metrics.csv under results_dir (created)."""
    os.makedirs(results_dir, exist_ok=True)
    metrics = compute_metrics(df)
    with open(os.path.join(results_dir, f"{prefix}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    df.to_csv(os.path.join(results_dir, f"{prefix}_states.csv"), index=False)
    return metrics
