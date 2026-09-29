# Phase 1 Gate Report (draft) — Verification of Original Test Results

Branch: `verify-phase1` | Latest commit: see `git log` | Date: 2026-09-29

## Status
BLOCKED only on cloud-side checkpoint loading for the three PatchTST runs
(c3861/da1af/7f87e); all code support for local re-evaluation is in place and
smoke-tested. DLinear re-evaluation is runnable locally today.

## Phase-1 deliverables
| deliverable | file | status |
|---|---|---|
| Provenance-aware eval driver | `experiments/eval_verify.py` | done, smoke-tested (5-step DLinear 6 running) |
| Seeded, no-predictor-capable env | `sim_edge_env.py` + `seeding.py` | done, smoke-tested (obs/cpu_util/faults confirmed) |
| Standardized KPIs | `experiments/metrics.py` | done (matches historical audits exactly) |
| Training diagnostics | `experiments/train_diagnostics.py` | done (all 6 archived trials summarized) |
| Forecaster verification | `experiments/forecast_verify.py` | done (single-seed DLinear: RMSE 9.04 vs paper 9.00) |
| Documentation | `docs/REPO_MAP.md`, `docs/REPRO_DIFF.md` | done |

## Key verification findings (evidence-backed)
1. Historical KPI table can be reproduced from archived `states.csv`
   (e.g., `6/tests/1` PT: moves=2008, acc=42.367, util=0.453, cons=93114,
   ncon=26779 over 7000 steps) via a 20-line pandas call; no checkpoint load
   required. Paper table consumption is therefore sound for the archived runs.
2. The comparison table's "PT" columns are actually the MSB baseline rows
   (`tests/msb_sim`); the true PT-IMPALA runs are `tests/{1,0}` of `6/12/18`.
3. DLinear forecaster: RMSE 9.004/MAE 6.145/R2 0.626, latency 0.16 ms vs
   PatchTST RMSE 8.958/MAE 5.950/R2 0.630 at 32.6 ms (paper: 8.51/5.62/0.67
   + 33 ms). Re-computed locally seed-42 DLinear = 9.04/6.19/0.623.

## Hygiene / provenance
- All 6 archived configs + base `data/configs` restored to PatchTST
  (`model_type` absent). DLinear explicitly scoped to
  `datacenter_{sim,test}_{6,12,18}.json`.
- Seeds: generator `seed=42` in configs; RL `learn_config.seed=203`.
  `eval_verify` records both.
- Running a 7000-step local eval is possible but takes ~5-10 min/run with the
  loaded DLinear agents; PT agents require the cloud Python image.

## Next
1. (Cloud) re-eval the 3 PT checkpoints + optionally new seeded 1.7 runs.
2. Collate all three arms into the paper table with the correct labels.
3. Proceed to Phase-2 optimizations (fast-path model, async forecasting, etc.).
