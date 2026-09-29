# -*- coding: utf-8 -*-
"""Forecaster verification (Phase 1).

Reproduces the DLinear vs PatchTST forecaster comparison from the paper's
Table(s) using the same pipeline as experiments/train_predict.py, but:

- runs over multiple seeds (default 42, 43, 44, 45, 46),
- aggregates RMSE / MAE / R2 / RMSE% / MAE% / latency across seeds,
- compares every seed against the paper-reported numbers,
- writes results/phase1/forecast_verify/forecast_verify_<seed>.json and
  forecast_verify_summary.json.

Expected paper values (from EdgeAIBus paper / benchmark_results.json):
    PatchTST  RMSE 8.51  MAE 5.62  R2 0.67
    DLinear   RMSE 9.00  MAE 6.14  R2 0.63

Usage:
    python experiments/forecast_verify.py [--seeds 42 43 44 45 46]
        [--models dlinear|patchtst|both] [--device auto|cpu|cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(__file__))

from train_predict import run_model  # noqa: E402
from typing import Dict

PAPER = {
    "PatchTST": {"rmse": 8.51, "mae": 5.62, "r2": 0.67},
    "DLinear": {"rmse": 9.00, "mae": 6.14, "r2": 0.63},
}
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def finalize(model_res: dict, paper: dict) -> dict:
    """Annotate a run_model result entry with per-metric deltas vs paper."""
    out = {k: v for k, v in model_res.items() if k != "per_horizon"}
    out["per_horizon"] = model_res.get("per_horizon", [])
    out["paper_rmse"] = paper["rmse"]
    out["paper_mae"] = paper["mae"]
    out["paper_r2"] = paper["r2"]
    out["delta_rmse"] = model_res["rmse"] - paper["rmse"]
    out["delta_mae"] = model_res["mae"] - paper["mae"]
    out["delta_r2"] = model_res["r2"] - paper["r2"]
    return out


def summarize_across_seeds(all_seeds: list) -> dict:
    by_model: Dict[str, list] = {}
    for seed_res in all_seeds:
        for m in seed_res["models"]:
            by_model.setdefault(m["model"], []).append(m)
    summary = {}
    for model_name, entries in by_model.items():
        vals = {k: [e[k] for e in entries] for k in ("rmse", "mae", "r2", "rmse_pct", "mae_pct")}
        lats = [e["latency_ms"]["mean"] for e in entries]
        p95s = [e["latency_ms"]["p95"] for e in entries]
        summary[model_name] = {
            "n_seeds": len(entries),
            "rmse_mean": float(np.mean(vals["rmse"])),
            "rmse_std": float(np.std(vals["rmse"])),
            "mae_mean": float(np.mean(vals["mae"])),
            "mae_std": float(np.std(vals["mae"])),
            "r2_mean": float(np.mean(vals["r2"])),
            "r2_std": float(np.std(vals["r2"])),
            "rmse_pct_mean": float(np.mean(vals["rmse_pct"])),
            "mae_pct_mean": float(np.mean(vals["mae_pct"])),
            "latency_ms_mean_mean": float(np.mean(lats)),
            "latency_ms_p95_mean": float(np.mean(p95s)),
            "num_params": int(entries[0].get("num_params", 0)),
            "paper": PAPER.get(model_name, {}),
        }
    return summary


def main():
    ap = argparse.ArgumentParser(description="Forecaster verification (Phase 1)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--models", choices=["dlinear", "patchtst", "both"], default="both")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--h", type=int, default=6)
    ap.add_argument("--input-size", type=int, default=48)
    ap.add_argument("--results-root", default=os.path.join("results", "phase1"))
    args = ap.parse_args()

    import torch

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    out_root = os.path.join(args.results_root, "forecast_verify")
    os.makedirs(out_root, exist_ok=True)
    all_seeds = []
    for seed in args.seeds:
        from experiments.seeding import seed_everything
        seed_everything(seed)
        torch.manual_seed(seed)
        res = run_model(
            argparse.Namespace(
                model=args.models, h=args.h, input_size=args.input_size,
                moving_avg=25, epochs=30, batch_size=32, lr=1e-3,
                patchtst_steps=100, dlinear_impl="custom", device=str(device),
                bitbrains_path="datasets/bitbrains/rnd",
                verbose=False,
            ),
            device=device,
        )
        res = {k: v for k, v in res.items() if k != "device"}
        res["seed"] = seed
        for i, m in enumerate(res["models"]):
            paper = PAPER.get(m.get("model"), {})
            res["models"][i] = finalize(m, paper)
        all_seeds.append(res)

        with open(os.path.join(out_root, f"forecast_verify_{seed}.json"), "w") as f:
            json.dump(res, f, indent=2)

    summary = summarize_across_seeds(all_seeds)
    payload = {
        "seeds": args.seeds,
        "h": args.h,
        "input_size": args.input_size,
        "device": str(device),
        "per_seed": all_seeds,
        "summary": summary,
        "paper": PAPER,
    }
    with open(os.path.join(out_root, "forecast_verify_summary.json"), "w") as f:
        json.dump(payload, f, indent=2)

    print("=" * 80)
    print("Forecast verification summary (RMSE/MAE% are percentages of mean y)")
    print("=" * 80)
    for model, s in summary.items():
        p = s["paper"]
        print(f"{model}: RMSE {s['rmse_mean']:.4f}+-{s['rmse_std']:.4f} "
              f"(paper {p.get('rmse','-'):.2f})  MAE {s['mae_mean']:.4f} "
              f"(paper {p.get('mae','-'):.2f})  R2 {s['r2_mean']:.4f} "
              f"(paper {p.get('r2','-'):.2f})  latency {s['latency_ms_mean_mean']:.2f}ms")
    print(f"\n[saved] {os.path.join(out_root, 'forecast_verify_summary.json')}")


if __name__ == "__main__":
    main()
