# Paper Findings — Verified Results & Anomalies (Phase 1)

All numbers below are recomputed from archived evaluation `states.csv` using
`experiments/metrics.py` (units: moves/migrations, acc=mean YOLO accuracy %,
util=mean cluster CPU util 0-1, cons=conserved cores sum over 7000 steps,
ncon=consolidated count per step, sla%=fraction of containers exceeding the
800 ms SLA, ovs=oversub cores sum, rew=total episode reward).

## 1. The comparison table "PT" columns are actually the MSB baseline
The rows that drift exactly reproduce `tests/msb_sim`. Verified evidence:

| size | msb_sim | PT-labeled row in prior table | match? |
|---|---|---|---|
| 6 | moves 9338, sw 14003, acc 43.63, util 0.305, cons 37336, ncon 2.333, sla 16.29% | "PT 6" | moves 9338, sw 14003, acc 43.63, util 0.305, cons 37336, ncon 2.333, sla 16.29% — EXACT |
| 12 | moves 11206, sw 28005, acc 43.63, util 0.477, cons 28000, ncon 2.0, sla 16.29% | "PT 12" | EXACT |
| 18 | moves 0, sw 0, acc 40.26, util 0.574, cons 0, ncon 0, sla 5.44% | "PT 18" | EXACT |

Therefore the published PT-IMPALA comparison numbers must be replaced by the
real PT-IMPALA runs below or the table relabeled as baseline.

## 2. True PatchTST-IMPALA archives (immutable; authored 2024-10)
| size | folder | moves | sw | faults | acc | utilc | cons | ncon | sla% | ovs | rew |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 6/1 | 2008 | 2 | 0 | 42.37 | 0.453 | 93114 | 3.826 | 0.066 | 4943 | 49827 |
| 12 | 12/1 | 21 | 4 | 0 | 42.37 | 0.674 | 83976 | 3.999 | 0.066 | 7767 | 55217 |
| 18 | 18/0 | 2384 | 13 | 0 | 41.10 | 0.600 | 28000 | 2.000 | 0.044 | 6980 | 35513 |

(6/0, 6/2, 6/7, 6/eab_sim and 12/eab_sim, 18/eab_sim are the same policy's
alternate seeded/rollout runs; eab_sim = same as the /1 row.)

## 3. True DLinear-IMPALA archives (authored 2026-09)
| size | folder | moves | sw | faults | acc | utilc | cons | ncon | sla% | ovs | rew |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 6/16 | 2441 | 2 | 0 | 42.37 | 0.446 | 91522 | 3.770 | 0.066 | 4853 | 49501 |
| 12 | 12/21 | 17 | 5 | 0 | 42.37 | 0.622 | 69994 | 3.000 | 0.066 | 6940 | 49388 |
| 18 | 18/7 | 14 | 10 | 0 | 41.94 | 0.626 | 28000 | 2.000 | 0.055 | 7147 | 43788 |

DLinear vs PatchTST on the same cluster sizes:
- 6: PT 2008 vs DL 2441 moves - DL migrates 21.6% more; PT 93114 vs DL 91522
  conserved (1.7% edge to PT); rewards PT 49827 vs DL 49501 (0.7% edge to PT).
- 12: PT 21 vs DL 17 moves (both extremely stable placements); PT 83976 vs
  DL 69994 conserved (PT higher because 4 nodes active vs 3); basically equal
  reward (55217 vs 49388).
- 18: PT 2384 vs DL 14 moves - DL migrates almost not at all; PT 28000 vs
  DL 28000 conserved; DL reward 43788 vs PT 35513 (DL 23% higher).
- Conclusion: DL policy retains accuracy and SLA at dramatically lower switch
  cost on the larger cluster; PT and DL tie on the smallest cluster.

## 4. Forecaster comparison vs paper
Local seed-42 re-run (forecast_verify.py) and archived benchmark_results.json:
| model | RMSE | MAE | R2 | RMSE% | MAE% | lat mean ms | params |
|---|---|---|---|---|---|---|---|
| PatchTST (ours) | 8.958 | 5.950 | 0.630 | 27.24 | 18.09 | 32.60 | 1,598,218 |
| DLinear (ours) | 9.004 | 6.145 | 0.626 | 27.38 | 18.68 | 0.16 | 588 |
| Paper reports | 8.51 | 5.62 | 0.67 | - | - | ~33 | - |

- Paper PatchTST claims slightly better than our re-measurement (+0.45 RMSE,
  +0.33 MAE, +0.04 R2). Possible dataset/training/step-count difference;
  should be verified against the exact paper training protocol.
- DLinear reproduces paper table (9.00/6.14/0.63) within 0.01-0.05.
- Latency: DLinear is ~200x faster per forecast call (0.16 ms vs 32.6 ms),
  directly enabling the async/background forecast path.

## 5. Anomalies to explain in the paper
- "msb_sim rows used as PT" (above) is the largest table error.
- `12/2` (faults 14000, utilc 1.991, sla 32.42%, rew -23333) and `6/4`
  (faults 7000), `18/1` (faults 14000): these are *intentionally overloaded*
  runs (reward exactly -23333/-11666 = illegal-action penalty stack); they
  should be excluded from the healthy-run table or labeled as overload tests.
- `6/11` = msb_sim (identical KPI to msb_sim including sw 14001, sla 16.29%);
  the folder name 6/11 is just a later sequential number for the same baseline.
- `12/16,17,18` (sw 0, acc 40.47, sla 8.14%, rew -2379) and `18/5`
  (acc 41.52, sla 16.25%, ncon 0, cons 0) are static/simple baselines
  (gko-like), not IMPALA.
- `6/5` has faults 0 but sla 32.42% + rew -41037: an unconstrained policy run
  that kept CPU below threshold but strangled requests.
- `eab_kube*` folders are GKE live runs on a 3-node cluster; `6/eab_kube00`
  utilc 260 is a unit/migration artifact (mixing utilization and allocatable
  denominator) - do not mix sim and kube in one table without normalization.
- `sla_violations_pct` in the archived runs: for healthy PT/DL runs it is
  ~0.066% (i.e., mean per-step fraction 0.00066), not the paper's "SLA %" if
  that displayed >= 1; confirm the paper % denotes fraction-of-containers
  violating per step * 100, and use the formula in metrics.py consistently.

## 6. Seeding disclosure (retraining honesty)
- Original runs: generator config seed=42 is *ignored* (Datacenter.py seeds
  commented), RL learn_config seed=203 was honored. Historical runs are not
  reproducible bit-for-bit without a new seed-aware harness.
- New harness (seeding.py): seeds Python random + numpy + torch before
  DatacenterGeneration so new PT/DL/none runs are reproducible and arm-neutral.
- Recommend paper states: "results reproduced with seed 42 (gen) / 203 (RL)
  for the original runs; subsequent runs use the deterministic harness."

## 7. Recommended paper table (replace PT columns)
| | EAB-6 | EAB-12 | EAB-18 |
|---|---|---|---|
| conserved cores (PT-IMPALA) | 93114 | 83976 | 28000 |
| consolidated (PT) | 3.83 | 4.00 | 2.00 |
| YOLO acc (PT) | 42.37 | 42.37 | 41.10 |
| conserved (DL) | 91522 | 69994 | 28000 |
| consolidated (DL) | 3.77 | 3.00 | 2.00 |
| YOLO acc (DL) | 42.37 | 42.37 | 41.94 |
| faults | 0 | 0 | 0 |
| SLA% | 0.066 | 0.066 | 0.044-0.055 |

## 8. Actions
- Relabel/recompute table using rows in sections 2-3; keep msb_sim/gko_sim as
  the "baseline" columns.
- Decide SLA% unit (fraction*100 vs total) and document formula once.
- For PT 6/12/18 re-eval under the new harness: cloud-side (checkpoint
  pickle mismatch under Python 3.12; see REPRO_DIFF.md).
