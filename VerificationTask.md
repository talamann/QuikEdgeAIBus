# ROLE
You are a research engineer helping with a master's/PhD thesis. The codebase is a fork of EdgeAIBus
(https://github.com/BabarAli93/EdgeAIBus), from the paper "EdgeAIBus: AI-Driven Joint Container
Management and Model Selection Framework for Heterogeneous Edge Computing" (IEEE TPDS, 2025).
Your job is to run, verify, and extend the experiments in a reproducible, scientifically honest way.
Do NOT tune things until numbers look good. If something looks wrong, investigate and report it.

# BACKGROUND
EdgeAIBus uses a forecaster to predict CPU usage of Edge servers, feeds the predictions into a Gymnasium
state, and trains an IMPALA (RLlib) agent to jointly place containers and switch YOLOv8 model versions
(Nano/Small). Original forecaster: PatchTST. My fork replaces it with DLinear. Baselines: MSB (balanced
scheduler + model switching) and GKO (consolidation scheduler, static model).

Paper setup to respect:
- Cluster: 6 nodes (2x 2-core, 2x 4-core, 2x 6-core). Scenarios: 6, 12, 18 containers (EAB-6/12/18).
- Training: 1M steps per scenario. Testing: 7000 simulated steps. 50 YOLO samples drawn per container per step.
- SLA threshold: 800 ms. Server-fault threshold: 85% real CPU usage (illegal action = -10 reward).
- Forecast horizon: 6 steps, Bitbrains VMs 383 (2-core), 392 (4-core), 386 (6-core), 5-min resample,
  forward-fill, 90/10 train/test split, autoregressive rolling evaluation.
- Reported paper results (single run, use as approximate bands, not exact targets):
  * Training, per step: conserved cores ~14 (EAB-6), ~12 (EAB-12), ~4 (EAB-18); consolidated servers 4 / (2-3) / 2
    (the paper's "12 servers" for EAB-12 is impossible with 6 nodes, so judge by conserved cores);
    train YOLO accuracy 42.22 / 42.01 / 41.25 %; zero server faults.
  * Testing: ~110% oversubscription (incl. K8s overhead), CU% < 85 (about 70), SLA violations about 1% or less,
    near-zero model switches, few migrations, MSB has higher accuracy but more SLA violations.
  * PatchTST forecast: RMSE 8.51%, MAE 5.62%, R2 0.67.

# CURRENT OBSERVATIONS (from my own runs, believed to be cumulative over 7000 test steps, UNVERIFIED)
KPI               DLin6  PT6    DLin12 PT12   DLin18 PT18
Reward            49501  7495   49388  10041  43788  318
Moves             2441   9338   17     11206  14     0
Consolid. steps   26388  16334  20999  14000  14000  0
Model switches    2      14003  5      28005  10     0
SLA violations    4      1140   4      1140   3      380
Accuracy %        42.4   43.6   42.4   43.6   41.9   40.3
CPU util %        44.6   30.5   62.2   47.7   62.6   57.4
Conserved cores   91522  37336  69994  28000  28000  0
Infer ms/step     1.98   -      2.97   -      4.16   -
Suspicious: PT switches ~2x/4x per step (14003, 28005 = 2x, 4x of 7000); PT-18 is all zeros; PT-6 and PT-12
have identical SLA (1140) and accuracy (43.6); DLinear barely acts (17, 14 moves). OC% and server-fault
counts are missing from my table.

# GLOBAL RULES
1. Work in a new git branch. Never overwrite existing results; write every run to results/<phase>/<run_id>/.
2. Every run must log: git commit hash, full config, seed, library versions, checkpoint path, wall-clock time.
3. Use explicit seeds everywhere (numpy, torch, python, gymnasium, RLlib). Default seed set: {0,1,2,3,4}.
4. Same training script, steps, evaluation mode (deterministic vs sampled actions), and checkpoint rule
   (report final AND best checkpoint separately) across all arms being compared.
5. Normalize counts to per-step averages and percentages. Store raw cumulative counts too.
6. Never modify baseline (PatchTST, MSB, GKO) logic except to fix a bug, and document every such fix in
   CHANGELOG_REPRO.md with a before/after result.
7. Check for train/test leakage in every forecaster pipeline (the rolling evaluation appends test data step by step;
   the model must only see data up to time t when predicting t+1..t+6).
8. If a compute budget is a problem, use fewer training steps for a smoke test first (e.g., 50k) and clearly
   label it as a smoke test. Do not report smoke-test numbers as results.
9. At each GATE below, stop and write a short report (gate_report_<n>.md) before continuing.

# PHASE 1: VERIFY EXISTING RESULTS  (highest priority)
1.1 Read the repo. Produce REPO_MAP.md: entry points, config files, where the state (Eq. 9), reward (Eq. 11-13),
    action mapping (Eq. 10), risk check, and the forecaster are implemented, and where each KPI is computed.
1.2 Metric audit: determine exactly how each KPI in my table is computed (cumulative vs per-step, units, what
    "moves", "consolid. steps", "SLA violations" count). Add missing metrics to the evaluation code:
    OC% and CU% (formulas from the paper, Section V-B), max node utilization per step, explicit server-fault
    count (steps where any node's real CPU > 85%), migrations/step, switches/step, consolidated servers/step,
    conserved cores/step, SV% per step. Output a standardized metrics JSON + CSV.
1.3 Training diagnostics: plot reward curves (total, accuracy, SLA, energy/consolidation, illegal-action) for all
    six existing runs (PT and DLin x 6/12/18), plus illegal-action rate, model-switch rate, and migration rate
    over training. Investigate PT-18 (all zeros): is the policy collapsed, was training truncated, is evaluation
    loading the wrong checkpoint/config, or is there a bug? Same for PT switching every step, and for identical
    PT-6/PT-12 SLA and accuracy values (verify each scenario loads its own checkpoint and config).
1.4 Evaluation-mode check: re-evaluate every existing checkpoint with deterministic actions and with sampled
    actions, and with final vs best checkpoint. Report how much the KPIs change.
1.5 Reproduce the paper's PatchTST result: run unmodified PatchTST + IMPALA with the repo defaults, 5 seeds,
    for 6, 12, 18 containers (full 1M steps if budget allows; otherwise 6 containers first).
    Compare against the paper bands above. Define "reproduced" as: most seeds land within roughly
    12-14 conserved cores/step, SLA <= ~1%, accuracy ~42%, near-zero switching, zero server faults at 6 containers.
    If it does not reproduce, diff the repo against the paper (reward factors and weights, Table IV hyperparameters,
    state layout, risk check settings, sample counts) and document each mismatch in REPRO_DIFF.md. Do not silently
    change them.
1.6 Forecaster-only evaluation: on the same 3 Bitbrains VMs, same 90/10 split, same 6-step horizon and rolling
    protocol, compute RMSE, MAE, R2 for PatchTST and DLinear (5 seeds where training is stochastic), plus
    inference latency. Confirm my DLinear numbers and the paper's PatchTST numbers (8.51 / 5.62 / 0.67).
1.7 Controlled comparison with three arms, same script, 5 seeds, 6/12/18 containers: PatchTST, DLinear,
    NO-PREDICTOR (zero out or remove the forecast features from the state). Report mean +/- std and 95% CI, and a
    paired test across seeds (e.g., Welch t-test or Mann-Whitney) for the key KPIs.
GATE 1: Summarize whether (a) PatchTST baseline reproduces, (b) DLinear vs PatchTST vs no-predictor differences
are larger than seed noise, (c) bugs found and fixed. Stop here if the baseline is broken and propose fixes
instead of moving on.

# PHASE 2: ADD MORE TEST ENVIRONMENTS
Only start after Gate 1 is passed or explicitly waved through. Keep baselines, seeds, traffic, and metrics
identical across environments. Save the same standardized metrics JSON for every environment.
2.1 Second workload trace (simulation): add one more CPU trace beyond Bitbrains (Google cluster trace, Alibaba
    trace, or Azure VM trace; pick whichever can be downloaded with a script). Re-run forecaster-only evaluation
    and the 3-arm scheduling comparison on it. Do NOT retrain on Bitbrains and test on the new trace unless you
    label it as a transfer experiment; report both if you do both.
2.2 Second simulator (if feasible): port the environment interface to EdgeSimPy (Python) or another simulator and run
    MSB, GKO, and the agent there. If porting is too large, write a design doc (SIM2_PLAN.md) with the effort
    estimate instead.
2.3 Local Kubernetes testbed: script a k3s or kind cluster with heterogeneous nodes (2/4/6 CPU limits via
    VMs or Docker/cgroups). Deploy YOLOv8 Nano/Small containers with a Flask inference API. Use k6 or Locust for
    repeatable traffic. Implement MSB-like (spread/balanced) and GKO-like (bin-packing/most-allocated) baselines with
    the Kubernetes scheduler framework or equivalent, and state clearly that these are re-implementations, not
    GKE's proprietary schedulers. Run 6 and 12 containers. Collect real latency, CPU usage, and node counts.
    Provide docker-compose/k3s setup scripts and a one-command runner.
2.4 Scale emulation: use KWOK or Kubemark (or the simulator) to test 50, 100, 200 containers for decision-time and
    scalability only. Do not report accuracy/SLA from emulated pods.
2.5 (Optional) Constrained-device profiling: re-run the paper's cgroup-limited setup (0.3, 0.6, 1 core, 1 GB RAM)
    for the forecasters, the scheduler, MSB, and GKO. Record inference time, CPU, memory, and energy (RAPL where
    available). Fill in PatchTST's missing inference time.
GATE 2: Report per-environment results side by side and note where conclusions change between environments.
# PHASE 3: ADD NEW MODELS
Only start after Gate 2. All new models go through the same pipeline, same protocol, 5 seeds.
3.1 New forecasters, implemented behind a common interface (fit, predict(horizon=6), save/load, latency): 
    GRU and HRF-ExGB (paper baselines), LSTM, a naive last-value / moving-average forecaster, NLinear,
    plus one more recent model (e.g., TimesNet, iTransformer, or N-BEATS) if a maintained library is available.
    Report RMSE/MAE/R2, inference time, memory, and downstream scheduling KPIs for each.
3.2 Forecast-quality ablations: horizon in {1, 3, 6, 12}, lookback window in {24, 48, 96}, and injected forecast noise
    levels, to test how sensitive the scheduler is to forecast quality. Include the no-predictor arm.
3.3 (Optional) Uncertainty-aware risk check: produce prediction intervals (quantile heads or conformal prediction) and
    use the upper bound in the 85% risk threshold / illegal-action check instead of the point forecast. Compare
    server-fault counts and consolidation against the point-forecast version.
3.4 (Optional) Second inference model family: add another application besides YOLOv8 (e.g., an image classifier with
    several size variants). Profile its accuracy, latency, and CPU/memory per core setting the way the paper profiles
    YOLO, and run the scheduling comparison.
GATE 3: Summarize which models help, which do not, and the cost/accuracy trade-off.

# DELIVERABLES
- results/ with raw logs, standardized metrics CSV/JSON per run, and checkpoints (or checkpoint hashes)
- figures/: training curves, KPI bar charts with error bars, forecast error vs downstream KPI scatter, latency/energy table
- REPO_MAP.md, CHANGELOG_REPRO.md, REPRO_DIFF.md, gate_report_1..3.md
- A single script per phase to reproduce everything (run_phase1.sh, run_phase2.sh, run_phase3.sh) with seeds and configs
- A final SUMMARY.md: what was verified, what failed, what is uncertain, and what claims are and are not supported

# REPORTING STYLE
Be concrete and skeptical. Report negative results. Give tables with mean +/- std across seeds. Flag anything that
looks like a bug, leakage, or a degenerate policy. If you are unsure whether a metric is cumulative or per-step,
say so instead of guessing. Ask me before making any change that alters the baseline's behavior.