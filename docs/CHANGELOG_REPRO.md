# Reproducibility Change Log (Phase 1)

Branch `verify-phase1`. Each item is a deliberate, documented change.

## Commit 9c70d27f2 — DLinear integration baseline + hygiene
- Restore 6 archived IMPALA configs + base `data/configs` to PatchTST.
- Add DLinear 12-node config variants (`_12`).
- Add .gitignore (ray_logs, lightning_logs, pycache, results/).
- Commit DLinear archived runs, configs, docs (AGENTS.md, Article-to-Code Map,
  DLinear_Improvements_Commit_History.md, VerificationTask.md).

## Commit 25802fd52 — repro-eval tooling
- experiments/seeding.py (seed_everything)
- sim_edge_env.py: no_predictor flag, server_faults + node_cpu_util_frac info
- experiments/metrics.py (standardized KPIs)
- experiments/eval_verify.py (provenance-aware eval driver)

## Commit 829c7b5a3 — training diagnostics
- experiments/train_diagnostics.py (progress.csv summaries)
- eval_verify metadata now records generator seed 42 + RL seed 203

## Commit c8d2e95e3 — forecaster verification
- experiments/forecast_verify.py (multi-seed DLinear vs PatchTST)

## Commit 411b3428e — docs
- docs/REPO_MAP.md, docs/REPRO_DIFF.md, docs/gate_report_1.md

## This commit — paper findings
- docs/PAPER_FINDINGS.md (verified tables + anomaly catalog + recommendations)

## Repro rules of thumb
1. Never overwrite `data/trainresults/IMPALA_*/datacenter_*.json` when changing
   a forecaster; version arm-configs by suffix (`_6/_12/_18`) instead.
2. Always call seed_everything() before DatacenterGeneration for new runs.
3. Never merge sim and kube KPIs in one table (unit mismatch proven above).
4. Re-verify any paper number against a states.csv row, not a label.
