# -*- coding: utf-8 -*-
"""Training-run diagnostics for Phase 1 reporting.

Reads each archived IMPALA trial (data/trainresults/IMPALA_{6,12,18}/*/)
and summarizes the training trajectory from progress.csv:

- wall-clock/hours, final + best episode_reward_mean (env_runners/episode_reward_mean)
- final policy_loss, grad_gnorm, entropy, vf_explained_var
- convergence iteration (first iter where reward_mean >= 95% of final)
- throughput (env steps / second), total timesteps
- per-iteration series exported to CSV for plotting
- run metadata from params.json (algorithm, num_containers, model_type, seed)

Writes:
  results/diagnostics/diagnostics.json       summary table (all trials)
  results/diagnostics/progress_<trial>.csv   normalized per-iteration series
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.utils.constants import TRAIN_RESULTS_PATH
from experiments.eval_verify import checkpoint_model_type


REWARD_COLS = ("env_runners/episode_reward_mean", "episode_reward_mean")
POLICY_LOSS_COLS = (
    "info/learner/default_policy/learner_stats/policy_loss",
    "info/learner/default_policy/policy_loss",
)
ENTROPY_COLS = (
    "info/learner/default_policy/learner_stats/entropy",
    "info/learner/default_policy/entropy",
)
GRAD_COLS = (
    "info/learner/default_policy/learner_stats/grad_gnorm",
    "info/learner/default_policy/grad_gnorm",
)
ITER_COLS = ("training_iteration", "iterations_since_restore")
TOTAL_TIME_COLS = ("time_total_s",)
STEPS_SAMPLED_COLS = ("num_env_steps_sampled", "info/num_env_steps_sampled")


def _pick(df, cols, default=None):
    for c in cols:
        if c in df.columns:
            return df[c]
    return default


def summarize_trial(trial_dir, num_containers, trial_name):
    row = {"num_containers": num_containers, "trial": trial_name}
    row["model_type"] = checkpoint_model_type(os.path.dirname(trial_dir), os.path.basename(trial_dir))
    params_path = os.path.join(trial_dir, "params.json")
    if os.path.exists(params_path):
        try:
            cfg = json.load(open(params_path, encoding="utf-8"))
            row["algorithm"] = cfg.get("algorithm", cfg.get("run_or_experiment"))
        except Exception as exc:
            row["algorithm"] = None
            row["params_error"] = str(exc)[:120]

    progress = os.path.join(trial_dir, "progress.csv")
    if not os.path.exists(progress):
        row["error"] = "no progress.csv"
        return row
    df = pd.read_csv(progress)
    n = len(df)
    total_time_s = float(_pick(df, TOTAL_TIME_COLS).iloc[-1]) if _pick(df, TOTAL_TIME_COLS) is not None and n else 0.0
    total_steps = float(_pick(df, STEPS_SAMPLED_COLS).iloc[-1]) if _pick(df, STEPS_SAMPLED_COLS) is not None and n else 0.0

    reward = _pick(df, REWARD_COLS)
    policy_loss = _pick(df, POLICY_LOSS_COLS)
    entropy = _pick(df, ENTROPY_COLS)
    grad = _pick(df, GRAD_COLS)
    iters = _pick(df, ITER_COLS)

    if reward is not None and n:
        rew = reward.astype(float)
        final_rew = float(rew.iloc[-1])
        best_rew = float(rew.max())
        best_iter = int(rew.idxmax()) + 1
        target = final_rew * 0.95
        conv = None
        for i, v in enumerate(rew):
            if v >= target:
                conv = i
                break
        row.update({"final_episode_reward_mean": round(final_rew, 4),
                    "best_episode_reward_mean": round(best_rew, 4),
                    "best_iteration": best_iter,
                    "converge_95pct_iteration": conv})
    if policy_loss is not None and n:
        row["final_policy_loss"] = round(float(policy_loss.astype(float).iloc[-1]), 6)
    if entropy is not None and n:
        row["final_entropy"] = round(float(entropy.astype(float).iloc[-1]), 6)
    if grad is not None and n:
        row["final_grad_gnorm"] = round(float(grad.astype(float).iloc[-1]), 6)

    row.update({
        "iterations": n,
        "total_steps_sampled": int(total_steps),
        "wall_clock_hours": round(total_time_s / 3600.0, 3),
        "throughput_steps_per_sec": round(total_steps / total_time_s, 2) if total_time_s else None,
    })

    # normalized series for plotting
    out_cols = {}
    if iters is not None:
        out_cols["training_iteration"] = iters.astype(int).values
    if reward is not None:
        out_cols["episode_reward_mean"] = reward.astype(float).values
    if policy_loss is not None:
        out_cols["policy_loss"] = policy_loss.astype(float).values
    if entropy is not None:
        out_cols["entropy"] = entropy.astype(float).values
    if TOTAL_TIME_COLS[0] in df.columns:
        out_cols["wall_clock_s"] = df[TOTAL_TIME_COLS[0]].astype(float).values
    return row, (out_cols if out_cols else None)


def main():
    out_root = os.path.join("results", "diagnostics")
    os.makedirs(out_root, exist_ok=True)
    all_rows = []
    for n in (6, 12, 18):
        exp_dir = os.path.join(TRAIN_RESULTS_PATH, f"IMPALA_{n}")
        if not os.path.isdir(exp_dir):
            continue
        for entry in sorted(os.listdir(exp_dir)):
            td = os.path.join(exp_dir, entry)
            if not os.path.isdir(td) or not entry.startswith("IMPALA_"):
                continue
            res = summarize_trial(td, n, entry)
            if isinstance(res, tuple):
                row, series = res
                if series:
                    short = entry[len("IMPALA_SimEdgeEnv_"):].split("_00000_0_")[0]
                    sname = "progress_" + short + ".csv"
                    pd.DataFrame({k: (v if v is not None else []) for k, v in series.items()}).to_csv(
                        os.path.join(out_root, sname), index=False)
            else:
                row = res
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(out_root, "diagnostics.csv"), index=False)
    json.dump(df.fillna("").to_dict(orient="records"),
              open(os.path.join(out_root, "diagnostics.json"), "w"), indent=2, ensure_ascii=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
