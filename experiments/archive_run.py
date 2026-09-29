"""
Archive a finished training run from data/ray_logs into data/trainresults/IMPALA_<N>.

Usage:
  python experiments/archive_run.py --num_containers 6
  python experiments/archive_run.py --num_containers 18

Picks the newest IMPALA_<timestamp> run folder in data/ray_logs, copies its trial
subfolder into data/trainresults/IMPALA_<N>/, and refreshes datacenter_sim.json /
datacenter_test.json from data/configs/datacenter_sim_<N>.json and
datacenter_test_<N>.json (so eval loads the DLinear config).
"""
import os
import shutil
import sys

import click

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.constants import RAY_LOGS_PATH, TRAIN_RESULTS_PATH, CONFIGS_PATH  # noqa: E402


def newest_2026_run():
    runs = []
    for d in os.listdir(RAY_LOGS_PATH):
        p = os.path.join(RAY_LOGS_PATH, d)
        if not os.path.isdir(p):
            continue
        if not d.startswith('IMPALA_'):
            continue
        # trial subfolder(s) live directly inside the run dir
        trials = [x for x in os.listdir(p) if os.path.isdir(os.path.join(p, x))]
        if not trials:
            continue
        runs.append((p, trials))
    # newest first by dir name timestamp
    runs.sort(key=lambda r: r[0], reverse=True)
    return runs


def detect_num_containers(run_dir, trials):
    """Load a checkpoint's action space and return the number of containers (= nvec length)."""
    from ray.rllib.algorithms.algorithm import Algorithm
    for trial in trials:
        trial_dir = os.path.join(run_dir, trial)
        cps = sorted(
            s for s in filter(
                lambda x: 'checkpoint' in x,
                os.listdir(trial_dir)))
        if not cps:
            continue
        try:
            agent = Algorithm.from_checkpoint(os.path.join(trial_dir, cps[-1]))
            nvec = agent.get_policy().action_space.nvec
            return int(len(nvec)), trial
        except Exception as e:
            print(f"  detect failed for {trial}: {str(e)[:80]}")
            continue
    return None, None


@click.command()
@click.option('--num_containers', required=True, type=int)
@click.option('--run-index', required=False, type=int, default=None,
              help='0-based index into newest-first run list; default: auto-match by container count')
def main(num_containers: int, run_index: int):
    import ray
    ray.init(local_mode=True, log_to_driver=False, ignore_reinit_error=True)
    runs = newest_2026_run()
    if not runs:
        raise RuntimeError("no IMPALA_ runs found in data/ray_logs")

    if run_index is not None:
        if run_index >= len(runs):
            raise RuntimeError(f"—run-index {run_index} out of range (have {len(runs)})")
        run_dir, trials = runs[run_index]
        print(f"using --run-index {run_index}: {run_dir}")
    else:
        matched = None
        for run_dir, trials in runs:
            print(f"detecting container count in {run_dir} ...")
            found, trial = detect_num_containers(run_dir, trials)
            if found == num_containers:
                matched = (run_dir, trials, trial)
                break
        if matched is None:
            raise RuntimeError(f"no run with {num_containers} containers found in data/ray_logs")
        run_dir, trials, trial = matched
        print(f"matched run {run_dir} -> {num_containers} containers")

    if len(trials) != 1:
        print(f"WARNING: {len(trials)} trials in {run_dir}, taking first")
    trial = trials[0]

    dest_folder = os.path.join(TRAIN_RESULTS_PATH, f"IMPALA_{num_containers}")
    os.makedirs(dest_folder, exist_ok=True)

    src_trial = os.path.join(run_dir, trial)
    dst_trial = os.path.join(dest_folder, trial)
    if os.path.exists(dst_trial):
        print(f"destination trial already exists, removing: {dst_trial}")
        shutil.rmtree(dst_trial)
    print(f"copying {src_trial}")
    print(f"      -> {dst_trial}")
    shutil.copytree(src_trial, dst_trial)

    for suffix in ('sim', 'test'):
        src_cfg = os.path.join(CONFIGS_PATH, f"datacenter_{suffix}_{num_containers}.json")
        if os.path.exists(src_cfg):
            dst_cfg = os.path.join(dest_folder, f"datacenter_{suffix}.json")
            print(f"copying {src_cfg} -> {dst_cfg}")
            shutil.copy2(src_cfg, dst_cfg)
        else:
            print(f"WARNING: {src_cfg} not found, leaving existing archive config")

    print(f"\narchived into {dest_folder}")


if __name__ == "__main__":
    main()