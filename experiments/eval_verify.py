"""Phase-1 verification evaluation driver.

Runs a trained IMPALA checkpoint under the correct prediction provenance
(PatchTST / DLinear / NO-PREDICTOR) and writes:

  results/<phase>/<arm>/<run_id>/
      run_states.csv          per-step state
      run_metrics.json        standardized KPIs (metrics.compute_metrics)
      metadata.json           full reproducibility metadata

Checkpoint selection
--------------------
--checkpoint final        = last checkpoint_* folder
--checkpoint best         = checkpoint from progress.csv with max episode_reward_mean
--checkpoint N            = checkpoint_%06d (as in the author archives)

Prediction provenance
---------------------
The model_type is resolved in this order:
  1. --model-type flag (patchtst|dlinear|none) if given
  2. the checkpoint run's own params.json env_config.model_type if present
  3. 'patchtst' (repo default)

Avoids the legacy trap where the shared data/configs or archived
data/trainresults configs contained a stale model_type (fixed in the
hygiene commit).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datacenter.Datacenter import DatacenterGeneration  # noqa: E402
from workload.Workload_v2 import WorkloadGenerator  # noqa: E402

from utils.constants import (  # noqa: E402
    TRAIN_RESULTS_PATH,
    TESTS_RESULTS_PATH,
    DATASETS_PATH,
    SCHEDULER_PATH,
    CONFIGS_PATH,
    ENVSMAP,
)
from metrics import compute_metrics  # noqa: E402


MODEL_TYPES = ("patchtst", "dlinear", "none")
ARM_MAP = {"patchtst": "patchtst", "dlinear": "dlinear", "none": "no_predictor"}


def git_revision():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.getcwd(), capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return out or None
    except Exception:
        return None


def git_dirty():
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=os.getcwd(), capture_output=True, text=True, timeout=10,
        ).stdout
        return bool(out.strip())
    except Exception:
        return None


def find_trial(experiment_folder, trial=None, year=None):
    folders = [f for f in os.listdir(experiment_folder)
               if os.path.isdir(os.path.join(experiment_folder, f)) and ("checkpoint" not in f)]
    if trial:
        matches = [f for f in folders if trial in f]
        if not matches:
            raise SystemExit(f"--trial '{trial}' matched no folder in {experiment_folder}")
        return matches[0]
    if year:
        for f in folders:
            if str(year) in f:
                return f
    return folders[0]


def checkpoint_model_type(experiment_folder, trial_folder):
    """Provenance: read the run's own params.json (authoritative)."""
    params = os.path.join(experiment_folder, trial_folder, "params.json")
    if os.path.exists(params):
        try:
            cfg = json.load(open(params, encoding="utf-8"))
            ec = cfg.get("env_config", {})
            mt = ec.get("model_type") if isinstance(ec, dict) else None
            if mt:
                return str(mt)
        except Exception:
            pass
    return "patchtst"


def resolve_checkpoint(base_dir, checkpoint):
    checkpoints = sorted(
        s for s in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, s)) and s.startswith("checkpoint_")
    )
    if not checkpoints:
        raise SystemExit(f"no checkpoint folders in {base_dir}")
    if isinstance(checkpoint, int) or checkpoint.isdigit():
        name = f"checkpoint_{int(checkpoint):06d}"
        if name not in checkpoints:
            raise SystemExit(f"{name} not found in {base_dir} (have {checkpoints})")
        return os.path.join(base_dir, name)
    if checkpoint == "final":
        return os.path.join(base_dir, checkpoints[-1])
    if checkpoint == "best":
        progress = os.path.join(base_dir, "progress.csv")
        if os.path.exists(progress):
            df = pd.read_csv(progress)
            reward_col = next((c for c in ("env_runners/episode_reward_mean", "episode_reward_mean")
                               if c in df.columns), None)
            iter_col = next((c for c in ("training_iteration", "iterations_since_restore")
                             if c in df.columns), None)
            if reward_col is not None and iter_col is not None and len(df):
                idx = int(df[reward_col].idxmax())
                it = int(df[iter_col].iloc[idx])
                name = f"checkpoint_{it:06d}" if it > 0 else "checkpoint_000000"
                if name in checkpoints:
                    return os.path.join(base_dir, name)
                # fall back to nearest earlier saved checkpoint
                nums = sorted(int(c.split("_")[1]) for c in checkpoints)
                near = [n for n in nums if n <= it]
                pick = max(near) if near else nums[0] if nums else None
                if pick is not None:
                    return os.path.join(base_dir, f"checkpoint_{pick:06d}")
        raise SystemExit("could not resolve 'best' from progress.csv; pass an explicit N")
    raise SystemExit(f"unknown --checkpoint '{checkpoint}'")


def load_config(num_containers, model_type, config_file):
    """Pick the correct datacenter config for the arm."""
    if model_type == "dlinear":
        path = os.path.join(CONFIGS_PATH, f"datacenter_test_{num_containers}.json")
    else:
        # patchtst / none: use the archived (PT) config restored by hygiene commit
        path = os.path.join(TRAIN_RESULTS_PATH, f"IMPALA_{num_containers}", f"{config_file}.json")
    if not os.path.exists(path):
        raise SystemExit(f"config not found: {path}")
    return json.load(open(path, encoding="utf-8")), path


def recommend_config(model_type, num_containers):
    """Suggest which config file a user should pass."""
    if model_type == "dlinear":
        return f"datacenter_test_{num_containers}.json"
    return "datacenter_test.json"


def main():
    ap = argparse.ArgumentParser(description="Phase-1 verification eval driver")
    ap.add_argument("--num-containers", type=int, required=True, choices=[5, 6, 12, 18])
    ap.add_argument("--config-file", type=str, default=None,
                    help="datacenter config name inside the config dir (default: arm-derived)")
    ap.add_argument("--model-type", type=str, choices=list(MODEL_TYPES), default=None,
                    help="patchtst|dlinear|none; if unset read from the checkpoint params.json")
    ap.add_argument("--checkpoint", type=str, default="final",
                    help="final | best | N (checkpoint index, zero-padded to 6 digits)")
    ap.add_argument("--trial", type=str, default=None, help="substring of trial folder")
    ap.add_argument("--year", type=str, default=None, help="fallback marker for trial selection")
    ap.add_argument("--explore", type=str, choices=["deterministic", "sampled"], default="deterministic")
    ap.add_argument("--steps", type=int, default=7000, help="episode length in steps")
    ap.add_argument("--num-episodes", type=int, default=1)
    ap.add_argument("--tag", type=str, default=None, help="run_id tag (defaults to arm_checkpoint)")
    ap.add_argument("--phase", type=str, default="phase1")
    ap.add_argument("--results-root", type=str, default=None)
    args = ap.parse_args()

    import ray
    from gymnasium import make as gym_make
    from ray.rllib.algorithms.algorithm import Algorithm

    experiment_folder = os.path.join(TRAIN_RESULTS_PATH, f"IMPALA_{args.num_containers}")
    trial_folder = find_trial(experiment_folder, args.trial, args.year)
    trial_dir = os.path.join(experiment_folder, trial_folder)

    model_type = args.model_type or checkpoint_model_type(experiment_folder, trial_folder)
    if model_type not in MODEL_TYPES:
        model_type = "patchtst"
    arm = ARM_MAP[model_type]

    config_file = args.config_file or recommend_config(model_type, args.num_containers)
    config_full, config_path = load_config(args.num_containers, model_type, config_file)

    ckpt_path = resolve_checkpoint(trial_dir, args.checkpoint)
    ckpt_num = os.path.basename(ckpt_path).replace("checkpoint_", "")

    # Deterministic seeding BEFORE datacenter construction (all arms symmetric).
    from seeding import seed_everything
    seed = int(config_full.get("seed", 42))
    seed_everything(seed)

    generator_config = deepcopy(config_full)
    generator_config.pop("notes", None)
    generator_config.update({"type_env": "sim-edge",
                             "model_type": None if model_type == "none" else model_type,
                             "no_predictor": model_type == "none"})

    datacenter = DatacenterGeneration(generator_config)
    workload = WorkloadGenerator()
    env_config = deepcopy(generator_config["env_config_base"])
    env_config.update({
        "episode_length": args.steps,
        "datacenter": datacenter,
        "datasets": {"bitbrains_path": os.path.join(DATASETS_PATH, "bitbrains/rnd"),
                     "yolo_path": os.path.join(DATASETS_PATH, "yolo")},
        "scheduler_path": SCHEDULER_PATH,
        "overload_threshold": workload,
        "model_type": None if model_type == "none" else model_type,
        "no_predictor": model_type == "none",
    })

    ray.init(local_mode=True)
    try:
        env = gym_make(ENVSMAP["sim-edge"], config=env_config)
        agent = Algorithm.from_checkpoint(ckpt_path)
        print(f"trial       : {trial_folder}")
        print(f"model_type  : {model_type}")
        print(f"checkpoint  : {os.path.basename(ckpt_path)}")
        print(f"explore     : {args.explore}")

        rows = []
        for ep in range(args.num_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                t0 = time.perf_counter()
                action = agent.compute_single_action(obs, explore=(args.explore == "sampled"))
                obs, reward, done, trunc, info = env.step(action)
                latency = (time.perf_counter() - t0) * 1000
                rows.append(_flatten(action, reward, info, latency))
                if args.num_episodes == 1 and len(rows) >= args.steps:
                    break

        df = pd.DataFrame(rows)
        metrics = compute_metrics(df)

        results_root = args.results_root or os.path.join("results", args.phase)
        run_root = os.path.join(results_root, arm)
        os.makedirs(run_root, exist_ok=True)
        run_id = args.tag or f"{arm}_ckpt{ckpt_num}"

        json.dump(metrics, open(os.path.join(run_root, f"{run_id}_metrics.json"), "w"), indent=2,
                  ensure_ascii=False)
        df.to_csv(os.path.join(run_root, f"{run_id}_states.csv"), index=False)

        metadata = {
            "run_id": run_id,
            "arm": arm,
            "model_type": model_type,
            "num_containers": args.num_containers,
            "explore": args.explore,
            "checkpoint": os.path.basename(ckpt_path),
            "checkpoint_index": ckpt_num,
            "trial": trial_folder,
            "config_file": config_file,
            "config_path": config_path,
            "seed": seed,
            "git_rev": git_revision(),
            "git_dirty": git_dirty(),
            "steps": len(df),
            "num_episodes": args.num_episodes,
            "episode_length": args.steps,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0],
            "ray": ray.__version__,
        }
        json.dump(metadata, open(os.path.join(run_root, f"{run_id}_metadata.json"), "w"), indent=2,
                  ensure_ascii=False)

        print("---- metrics ----")
        for k in ("steps", "moves", "moves_per_step", "model_switches", "switches_per_step",
                  "server_faults", "fault_steps_frac", "mean_accuracy",
                  "mean_cluster_cpu_util", "conserved_cores", "conserved_per_step",
                  "consolidated_count", "consolidated_per_step", "sla_violations_frac",
                  "sla_violations_pct", "oversub_cores"):
            print(f"{k:24s} {metrics.get(k, float('nan')):.6f}")
        print("results written to", run_root)
    finally:
        ray.shutdown()


def _flatten(action, reward, info, latency_ms):
    return {
        "action": str(list(np.asarray(action).reshape(-1))),
        "num_moves": int(info["num_moves"]),
        "num_model_switches": int(info["num_model_switches"]),
        "num_overloaded": int(info["num_overloaded"]),
        "server_faults": int(info["server_faults"]),
        "num_consolidated": int(info["num_consolidated"]),
        "mean_accuracy": float(info["mean_accuracy"]),
        "mean_cluster_cpu_util": float(info["mean_cluster_cpu_util"]),
        "mean_cluster_mem_util": float(info["mean_cluster_mem_util"]),
        "cpu_conserved_cost": float(info["cpu_conserved_cost"]),
        "oversub_cores": float(info["oversub_cores"]),
        "num_slav": float(info["num_slav"]),
        "node_cpu_util_frac": str(np.asarray(info["node_cpu_util_frac"]).round(4).tolist()),
        "reward": float(reward),
        "reward_sla": float(info["rewards"]["reward_sla"]),
        "reward_accuracy": float(info["rewards"]["reward_accuracy"]),
        "reward_illegal": float(info["rewards"]["reward_illegal"]),
        "reward_consolidation": float(info["rewards"]["reward_consolidation"]),
        "inference_time_ms": latency_ms,
        "timestep": int(info["timestep"]),
        "global_timestep": int(info["global_timestep"]),
    }


if __name__ == "__main__":
    main()
