# Reproducibility & Diff Notes (Phase 1)

Purpose: document every deliberate difference between the "as-published" code
and the verification/optimization baseline so re-runs remain arm-neutral.

## 1. Config-provenance hygiene (BUG FIX, not a behavior change)
- **Finding:** the 6 archived configs under
  `data/trainresults/IMPALA_{6,12,18}/datacenter_{sim,test}.json` were mutated
  to `model_type: dlinear` during the DLinear integration, but the real
  archived runs (c3861/da1af/7f87e, ft beat64df, 2024-10) used PatchTST.
  Naive re-eval of those checkpoints would have silently consumed the
  DLinear prediction pickles.
- **Fix:** `git restore` brought all 6 archived configs back to HEAD (PatchTST,
  `model_type` absent). Base `data/configs/datacenter_{sim,test}.json`
  (12-node, PatchTST) also restored. Explicit DLinear configs now live at
  `data/configs/datacenter_{sim,test}_{6,12,18}.json` (n_cont 12 for `_12`,
  `model_type: dlinear`, seed 42, ep 100).
- **Impact:** none on historical test folders; only affects future eval routing.

## 2. Deterministic seeding (NEW capability, symmetric across arms)
- `experiments/seeding.py::seed_everything(seed)` seeds Python `random`,
  `numpy`, `PYTHONHASHSEED`, and torch (+ CUDA). Call BEFORE
  `DatacenterGeneration` because:
  - `Datacenter.py` lines 44-45 have `np.random.seed` / `random.seed`
    commented out;
  - line 89 `containers_model = np.random.randint(2, ...)`;
  - env step uses global `random.sample` for YOLO request sampling.
- Historical numbers are frozen: compare seeded<->seeded,
  historical<->historical only.

## 3. No-predictor arm (NEW flag)
- `sim_edge_env.py`: `no_predictor` config zeros the `cpu_predictions` slice in
  `observation()` BEFORE preprocessing. Obs dims identical (144, etc.);
  prediction pipeline and pickles untouched.

## 4. Evaluation reproducibility metadata (NEW)
- `eval_verify.py` records git_rev, git_dirty, model_type, checkpoint
  (final/best/N), trial folder, config path, both seeds (42 generator,
  203 RL), Ray/python versions, wall-clock. Fixes the latent bug where
  `--checkpoint best` could not be resolved for trials whose progress.csv
  used Ray 2.32 column names (`env_runners/episode_reward_mean`) and fell
  back incorrectly; now it picks the highest-saved checkpoint from
  `progress.csv` iteration when the best iteration has no checkpoint dir.

## 5. Checkpoint loading caveat (documented)
- Archived PatchTST checkpoints (Oct 2024) were pickled with an older Python.
  Under Python 3.12 + Ray 2.32, `Algorithm.from_checkpoint` fails with
  `TypeError: code() argument 13 must be str, not int` (loader mismatch).
  DLinear checkpoints (Sep 2026) load cleanly with the same toolchain.
  => Run re-evals for PT on the original Python/cloud image, or retrain;
  DLinear re-eval is runnable locally (Python 3.12/Ray 2.32).
- `progress.csv`/`params.json` compatible across the bucket.

## 6. Metrics layer (NEW)
- `metrics.py` computes SOC/CPU/conserved/SLA (%) in one consistent way for
  both historical states.csv and new evals (identical units as the paper
  table). `train_diagnostics.py` summarizes archived training curves
  (final/best reward, best/converged iteration, wall-clock, throughput).

## 7. Result storage convention
- New outputs go under `results/phase1/<arm>/<run_id>{_metrics,_states,_metadata}.json|csv`
  (gitignored). Forecast verification under `results/phase1/forecast_verify/`.
  Nothing in `data/` is rewritten except config provenance fixes described
  above (those are committed.)

## 8. Known differences vs published paper pipeline
- Original `test.py` picked `folders[0]` as the trial and still routed config
  from the (mutated) experiment folder; `eval_verify.py` instead resolves
  provenance from the checkpoint's own params.json + explicit config files and
  allows trial selection (`--trial`, `--year`).
- The archived reward decomposition already lives in states.csv, so historical
  KPIs are recoverable without a checkpoint load.
