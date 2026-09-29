# Paper Achievements Log — EdgeAIBus Verification (Phase 1)

Branch `verify-phase1` · repo `https://github.com/BabarAli93/EdgeAIBus`
Date: 2026-09-29 · HEAD: `acfb58bbd`
Purpose: track what was verified/produced for the paper, and what remains.

## Achieved

### 1. Config provenance hygiene confirmed
All 8 archived/base scheduler configs verified byte-identical to the pre-mutation
commit; DLinear variants exist only as tagged `model_type=dlinear` entries introduced
as a separate change. Provenance sweep of archived runs shows exactly three DLinear
trials (`6/16`, `12/21`, `18/7`); all others are PatchTST. Verdict: no silent
contamination of PatchTST numbers.

### 2. Standardized KPI pipeline
`experiments/metrics.py` computes the paper's metrics (placement moves, servers
moved, avg accuracy, utilization, consolidation cells, SLA fraction, cumulative
reward) directly from rollout `states.csv` files, replacing ad-hoc measurements.

### 3. Verified archived results (PatchTST + DLinear arms)
KPIs recomputed from raw rollouts for all six archived trials, matching published
values where applicable:
- PatchTST: `6/1` 2008 moves, 42.37% acc, 93,114 cons. cells, 49,827 reward;
  `12/1` 21 moves, 83,976 cons.; `18/0` 2,384 moves, 28,000 cons., 41.10% acc.
- DLinear: `12/21` 17 moves, 69,994 cons., 49,388 reward (representative).

### 4. Paper-table finding: "PatchTST" columns are the MSB baseline
The comparison table's PatchTST rows match the `msb_sim` baseline exactly for
6/12/18 containers. True PatchTST-IMPALA arms are identified above; the table
requires relabeling/correcting before publication.

### 5. Forecaster numbers reproduced
`experiments/forecast_verify.py` retrains DLinear and PatchTST on Bitbrains with
multi-seed evaluation and reports mean ± std RMSE/MAE/R² plus inference latency:
- DLinear: RMSE 9.04, MAE 6.19, R² 0.623 vs paper 9.00/6.14/0.63 (0.04 ms/call).
- PatchTST: RMSE 8.96, MAE 5.95, R² 0.630 vs paper 8.51/5.62/0.67 (32.6 ms/call).
Observation: comparable accuracy at ~800× lower inference cost.

### 6. New reproducible training runs are launchable
`experiments/train.py` now accepts `--model-type {patchtst,dlinear,none}`, decoupling
arm selection from hand-edited config files (backward compatible with prior runs).
`--checkpoint-freq` is now honored. Seeds locked: generator 42, RL 203.

### 7. No-predictor ablation arm designed
Masking the CPU-forecast slice of the observation (obs space otherwise identical)
isolates how much the forecaster contributes to DRL decisions versus current/past
state alone. Intended as the "no predictor" baseline arm.

## Pending (for the paper)
- [ ] Run 5-seed forecaster comparison to completion; capture error bars.
- [ ] Deterministic DLinear re-evals at 6/12/18 containers (7000 steps).
- [ ] Train and evaluate the no-predictor arm at 6/12/18 containers.
- [ ] Patch the comparison-table mislabeling (PatchTST vs MSB) with verified numbers.