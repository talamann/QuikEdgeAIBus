# EdgeAIBus Repository Map (verification/optimization baseline)

Generated during Phase-1 verification work on branch `verify-phase1`.

## Repository purpose
EdgeAIBus is a proactive framework that jointly manages container placement
(consolidation / oversubscription) and ML/DL model switching for YOLO object
detection services on heterogeneous edge nodes. A PatchTST CPU forecaster feeds
an IMPALA (off-policy Actor-Critic, V-trace) policy that maps each container to
(node, model-version) each timestep. A risk-check layer prevents Kubernetes
OutOfCPU / OutOfMemory admission failures.

## Top-level layout
```
AGENTS.md                  operational/architectural reference
VerificationTask.md        Phase 1 verification task (source of truth)
data/
  configs/                 datacenter generation configs (PT 12-node base,
                           DLinear variants datacenter_*_{6,12,18}.json)
  trainresults/IMPALA_{6,12,18}/   archived Ray trials + config provenance
  testresults/IMPALA/containers/{6,12,18}/tests/   eval KPI evidence
  datasets/bitbrains/rnd   Bitbrains trace data (383/392/386.csv, 3 months)
datacenter/                DatacenterGeneration (node/container simulation)
workload/                  WorkloadGenerator (overload threshold, SLA 800 ms)
scheduler/
  edgeaibus/envs/simulation/sim_edge_env.py   Gymnasium env
  edgeaibus/scheduler/Scheduler.py            prediction routing
  scheduler.py / msb / gko                    baselines
  patchtst_predictions_np.pkl, dlinear_predictions_np.pkl
experiments/
  train.py                 training driver (Ray IMPALA)
  test.py                  evaluation driver (legacy)
  eval_sim.py              eval emitting states.csv
  eval_verify.py           provenance-aware verification eval driver (new)
  train_predict.py         forecaster training/eval (DLinear + PatchTST)
  forecast_verify.py       multi-seed forecaster comparison (new)
  train_diagnostics.py     progress.csv summarizer (new)
  metrics.py               standardized KPI computation (new)
  seeding.py               seed_everything() for determinism (new)
  constants.py (utils/)    path/constant helpers
docs/                      this file + REPRO_DIFF.md
```

## Namespace / config layout
- Generator config files in `data/configs/` are the arm selector:
  `model_type: patchtst|dlinear` (absent = patchtst), `no_predictor: true` for
  the no-prediction arm.
- `env_config_base` keys drive the env: `episode_length`, penalties,
  obs_elements; `learn_config` holds RL hyper-params and RL seed (203).

## Core flows (Phase-1 relevant)
1. Forecaster: `train_predict.load_bitbrains -> split_dataset ->
       train_{dlinear,patchtst} -> evaluate_* -> summarize`.
2. Env step: `SimEdgeEnv.step(action) -> Scheduler (routing) ->
       Datacenter hosts usage / SLA / reward -> info`.
3. Eval: `eval_verify.py (checkpoint + config provenance) -> env.rollout ->
       metrics.compute_metrics -> metadata.json`.

## KPI definitions (see metrics.py)
- moves, moves_per_step                       migrations
- model_switches, switches_per_step           YOLO model changes
- server_faults, fault_steps_frac             nodes > 85% CPU (requests-based)
- mean_accuracy, mean_cluster_cpu_util        utilization
- conserved_cores, consolidated_count         consolidation (energy)
- sla_violations_frac, sla_violations_pct     fraction of containers > 800 ms
- oversub_cores                               oversubscription magnitude
