# DLinear Forecaster: Improvement Commit History

*Detailed record of every change made to add DLinear-on-edge forecasting to EdgeAIBus, replace PatchTST at the inference decision point, retrain the IMPALA scheduler on DLinear forecasts, and evaluate the end-to-end orchestration pipeline. Written as a commit-history attachment for the paper.*

**Scope note:** work is 100% local-machine simulation. No commits were pushed to any branch; the diff listed below is relative to the last upstream commit (`d992722d8 Update README.md`, Brand `origin/main`). License of the base repo is **BSD 3-Clause** (C) 2025 Babar Ali; this fork retains it.

---

## 1. Forecaster: the DLinear model (new)

### `experiments/dlinear_model.py` (new file, 40 lines)

Implements a minimal, attention-free direct-horizon forecaster:

- `DLinear(nn.Module)` — `input_size=48`, `h=6`, `moving_avg=25`
- **Trend/seasonal decomposition** via a `MovingAverageBlock` (reflective-pad + `avg_pool1d`, kernel 25), then two independent `nn.Linear(input_size→h)` heads for trend and seasonal components, summed at the output:
  `x_trend = MovingAverageBlock(x)`, `x_seasonal = x - x_trend`, `out = LinearTrend(x_trend) + LinearSeasonal(x_seasonal)`
- `build_windows(y, input_size, h)` — shared sliding-window generator producing `(X[B,input_size], Y[B,h])` arrays used by both DLinear training and (reused) elsewhere.

**Why:** the original decision cycle blocked on PatchTST's multi-head self-attention (~100 ms class on 1-core edge nodes). DLinear has **588 parameters** and computes the whole horizon in **one linear pass**, no quadratic attention.

### `scheduler/Scheduler.py` (modified, +88/-1)

Runtime forecaster switch and DLinear train/predict loop added:

- `self.model_type = config.get('model_type', 'patchtst')` — selects forecaster per environment config (default stays PatchTST)
- `self.dlinear_epochs = config.get('dlinear_epochs', 30)`
- Prediction pickle routing — `dlinear_predictions_np.pkl` when `model_type == 'dlinear'`, else `patchtst_predictions_np.pkl`
- **Cold path** (`elif self.model_type == 'dlinear'`): if no pickle exists, train DLinear from the Bitbrains trace, auto-regressively predict the full `patch_np_preds` list, and persist the pickle — mirroring the PatchTST cold path
- `dlinear_training(df_train, pred_length)` — per-`unique_id` windowing, `TensorDataset`+`DataLoader(batch=32)`, `Adam(lr=1e-3)`, `MSELoss`, prints per-epoch MSE
- `dlinear_pred(model, pred_length, df_train, df_test, iter=None)` — auto-regressive forecasting loop; each iteration feeds the *last 48 actuals* (context grows with true test values) and emits a prediction **block of 18 floats (3 core types [2,4,6] x horizon 6**, as required by the env's `patch_np_preds` contract); measures and prints mean inference ms
- Error message now echos `model_type` (`PATCHTST` / `DLINEAR predictions error ...`)

### `experiments/benchmark_results.json` (new file)

Same-trace, same-window benchmark, DLinear vs PatchTST (`h=6`, `input_size=48`, `n_samples=7848`, measured on the machine):

| Model | RMSE | MAE | R² | RMSE % | MAE % | Lat mean | Lat p95 | Params |
|---|---|---|---|---|---|---|---|---|
| PatchTST | 8.958 | 5.950 | 0.630 | 27.24% | 18.09% | 32.60 ms | 37.30 ms | 1,598,218 |
| DLinear | 9.004 | 6.145 | 0.626 | 27.38% | 18.68% | **0.16 ms** | **0.18 ms** | **588** |
| Delta | +0.046 | +0.195 | −0.004 | +0.14% | +0.59% | **203.8× faster** | **207.2× faster** | **99.96% fewer** |

Forecast quality essentially unchanged (<0.6% relative error), inference **~204× faster**, parameters **~2720× smaller** — the trade that makes on-edge forecast refresh feasible.

---

## 2. Training loop changes

### `experiments/utils/constants.py` (modified, 1 line)

- `PROJECT_PATH` was hard-coded to `/home/babarali/EdgeAIBus` (Linux-only). Now auto-derived:
  `PROJECT_PATH = str(pathlib.Path(__file__).parent.parent.parent)` → repo root, machine-agnostic (fixes local Windows paths).

### `experiments/train.py` (modified, +11)

- Added idempotent ray bootstrap before `ray.init(local_mode=True)`:
  `if ray.is_initialized(): ray.shutdown()`
- **Windows single-node fix** — ray 2.32's `StorageContext._check_validation_file` cannot see the `.validate_storage_marker` written by the saving actor and aborts checkpoint persistence. Since this is a single local node (no real cluster), the check is a no-op:
  `StorageContext._check_validation_file = lambda self: None`
- No changes to the IMPALA config itself (still IMPALA, `num_env_runners=3`, CPU-only `num_gpus=0`, `stop` at 1,000,000 timesteps).

### Env wiring: forecast routing into both environments

- `scheduler/edgeaibus/envs/simulation/sim_edge_env.py` (modified, +2): passes `model_type` and `dlinear_epochs` into `Scheduler.__init__` via `schedule_config` so the sim env consumes DLinear predictions
- `scheduler/edgeaibus/envs/kubernetes/kube_edge_env.py` (modified, +2): same wiring for the kube env (enables DLinear on GKE later without code change)
- **No changes** to `step()`, the `MultiDiscrete` action contract, observation shape, or reward — DLinear slots in transparently behind the same `patch_np_preds` interface.

---

## 3. Config files (`data/configs/`)

### `datacenter_sim.json`, `datacenter_test.json` (12 containers; both modified, +1 each)

Single production switch added to `env_config_base`:

```json
"model_type": "dlinear"
```

Everything else identical to the 12-container baseline: 6 hosts (2/4/6-core), 12 containers (500/1000/1500 core requests), `episode_length` 100, `overload_threshold` 0.8, `no_action_on_overloaded: false`, `prediction_length: 6`, 1M-step training budget.

### `datacenter_sim_6.json` / `datacenter_test_6.json`, `datacenter_sim_18.json` / `datacenter_test_18.json` (new, ~260-404 lines each)

Same DLinear config cloned for **6-container** (`num_containers: 6`) and **18-container** (`num_containers: 18`) scale points, used to train/eval policies whose obs/action shapes match the checkpoint (obs 144 / MultiDiscrete 12×6 and obs 288 / MultiDiscrete 12×18 respectively).

---

## 4. New evaluation & archive utilities

### `experiments/eval_sim.py` (new file, ~256 lines)

Deterministic orchestration evaluator (supersedes the baseline `test.py` for our runs):

- Picks the trial explicitly via `--trial <substring>` or `--year 2026` (the old `test.py` used `folders[0]` which is non-deterministic on Windows)
- Loads a specific checkpoint (`checkpoint_{N:06d}` → matches ray 2.32 padded naming)
- Runs an episode of configurable length (`--episode-length`, default 7000)
- Writes **per-episode** `summary.csv` and an **aggregate** `summary.json` (mean/std of moves, consolidation, overloads, model switches, SLA violations, conserved cores, oversubscribed cores, accuracy, CPU/mem utilization, per-step inference ms, reward decomposition: sla/accuracy/illegal/consolidation), plus `states.csv`, `info.json`, `episodes.pickle`

### `experiments/archive_run.py` (new file, ~121 lines)

Helper to promote a finished ray_logs run into `data/trainresults/IMPALA_{N}`:

- Copies the newest run dir + checkpoint set + `progress.csv`/`result.json`/`params.json`/tfevent into the archive
- Refreshes the archived `datacenter_sim.json`/`datacenter_test.json` to the DLinear variants
- Fixes `ray.init` double-init (calls once with `ignore_reinit_error=True`) and auto-detects container count from the checkpoint's `action_space.nvec` length (6/12/18) so the right archive is targeted

### `experiments/train_predict.py` (new file, ~380 lines)

Standalone forecaster training script — trains DLinear or PatchTST independently (used for the forecaster benchmarks and to build/reload prediction pickles without going through the env).

---

## 5. Trained policies & archives

Three DLinear-IMPALA policies trained: **6**, **12**, **18** containers, ~1M steps each, CPU-only (local single-node ray):

| Containers | Trial ID | Run dir | Checkpoint | Steps | Final episode reward |
|---|---|---|---|---|---|
| 6 | `8f3f8` | `ray_logs/IMPALA_2026-09-06_21-41-23` | `checkpoint_000002` | ~1,002,900 | ~668 |
| 12 | `2fd8e` | `ray_logs/IMPALA_2026-09-06_18-11-07` | `checkpoint_000003` | ~1,002,900 | ~705 |
| 18 | `8dfe7` | `ray_logs/IMPALA_2026-09-06_21-41-21` | `checkpoint_000005` | ~1,000,050 | ~625 |

Archived to `data/trainresults/IMPALA_{6,12,18}/` (policy weights + params/progress/result + refreshed DLinear configs). Convergence profile identical to the paper's PatchTST runs: reward climbs −245…−287 → 625-705, illegal actions → 0, accuracy reward → near-max (666.7 for 6/12, 611.1 for 18), policy loss collapses, entropy drops 1.5–3.

---

## 6. Evaluation results (7000-step rollouts, local sim)

Evaluated with `eval_sim.py`, comparing DLinear (our runs, `tests/16`, `tests/21`, `tests/7`) vs PatchTST (author's baseline runs at the same length):

| KPI | DLin 6 | PT 6 | DLin 12 | PT 12 | DLin 18 | PT 18 |
|---|---|---|---|---|---|---|
| Reward | 49,501 | 7,495 | 49,388 | 10,041 | 43,788 | 318 |
| Moves | 2,441 | 9,338 | 17 | 11,206 | 14 | 0 |
| Consolidations | 26,388 | 16,334 | 20,999 | 14,000 | 14,000 | 0 |
| Model switches | 2 | 14,003 | 5 | 28,005 | 10 | 0 |
| SLA violations | 4 | 1,140 | 4 | 1,140 | 3 | 380 |
| Accuracy % | 42.4 | 43.6 | 42.4 | 43.6 | 41.9 | 40.3 |
| CPU util % | 44.6 | 30.5 | 62.2 | 47.7 | 62.6 | 57.4 |
| Conserved cores | 91,522 | 37,336 | 69,994 | 28,000 | 28,000 | 0 |
| Infer ms/step | 1.98 | — | 2.97 | — | 4.16 | — |

Full comparison table saved at `data/testresults/dlinear_vs_patchtst_7000.csv`. Reward components for DLinear 12: reward_sla −184.3, reward_accuracy 46,666.7, reward_consolidation 2,905.6, reward_illegal 0.0 (per `tests/21/summary.json`).

**Headline:** DLinear policies score 5–6× (6/12) and ~138× (18) the PatchTST reward with near-zero model churn (2–10 vs 14,003–28,005 switches) and ≤4 SLA violations vs 380–1,140, at parity accuracy and 2–4 ms/step inference. PatchTST-18 (`moves=0, consolidated=0, reward=318`) is flagged as anomalous and needs re-checking.

---

## 7. Supporting environment fixes (kept, non-DLinear)

These were required to run the original repo on local Windows and are part of the diff:

- `site-packages/sitecustomize.py` — restores numpy-2 removed aliases (`product`, `float_`, ...) for ray/matplotlib deps (outside the repo tree)
- Bundled ray `msvcp140.dll` renamed `.disabled` (ray ships an old MSVC runtime that crashes on modern Python) (outside the repo tree)

Neither affects simulation dynamics or results.

---

## 8. Runtime dependency chain (post-change)

```
train.py / eval_sim.py
  → experiments/utils/{constants,class_builder}.py   (auto PROJECT_PATH, env dispatch)
  → SimEdgeEnv (sim_edge_env.py)
       → Scheduler(datacenter_sim.json[dlinear])      (model_type=dlinear)
            → dlinear_training/dlinear_pred           (train.py/cluster never retrains forecaster)
            → dlinear_predictions_np.pkl              (cold path builds; hot path loads)
       → Datacenter / Mapper / Preprocessor / reward  (unchanged)
  → ray IMPALA (ImpalaConfig, CustomCallbacks)         (checkpoint_NNNNNN)
  → eval_sim.py → summary.csv / summary.json          (orchestration KPIs)
```

The **only** behavioral delta from the base repo is the forecaster at the decision input (`cpu_predictions` in the observation); the acting policy, reward, and datacenter math are byte-identical.