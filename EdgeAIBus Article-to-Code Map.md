# EdgeAIBus: Article → Code Mapping

*A brief context document mapping each AGENTS.md bullet point to the concrete code that implements it, and *why* the design works the way it does.*

---

## 1. System Overview

- **Joint container management + model switching** → `scheduler/edgeaibus/envs/simulation/sim_edge_env.py`
  - The Gym environment's action space is a `MultiDiscrete` where **each container chooses both a target node AND a model version** (`MultiDiscrete(num_containers × num_hosts*2)`). This single-action design *is* the "joint" part — the RL agent picks container placement and model version in one decision, because optimizing them independently (place first, then pick model) ignores how they interact: a heavier model only fits on a node with enough spare CPU, so the two choices are coupled and must be chosen together.
- **Heterogeneous edge nodes (2-, 4-, 6-core)** → `datacenter/Datacenter.py` + config files
  - The cluster is built with `num_cores = {2, 4, 6}` and nodes are created per core-size from `data/configs/datacenter_sim.json`. Heterogeneity matters because a decision that is optimal on a 6-core node (e.g., run YOLO-Small) is *illegal* on a 2-core node under the 85% limit — so the scheduler must reason about each node type differently, not assume identical hardware.
- **Optimization objective $\min \sum (E + S - A)$** → `scheduler/edgeaibus/envs/rewards/reward.py`
  - The objective is expressed as a **reward** the agent maximizes (which is the inverse of minimization): `reward_energy` (E, usage + consolidation), `reward_sla` (S, *negatively* weighted ×-4 so violations are penalized), and `reward_accuracy` (A, maximized). Energy and accuracy get ×10 weights while SLA gets ×-4, encoding the paper's priority: aggressiveness on energy/consolidation is rewarded the most, and accuracy is nearly as valuable, but every SLA violation subtracts.

---

## 2. Key Mechanisms

- **Multivariate CPU forecasting via PatchTST** → `scheduler/Scheduler.py`
  - The `Scheduler` base class trains a `NeuralForecast` `PatchTST(h=6, input_size=48, ...)` on the Bitbrains trace. It's "multivariate" because it predicts all node types (2-, 4-, 6-core) in parallel from a shared model. Predictions are **pre-computed once and cached in `scheduler/patchtst_predictions_np.pkl`** (a list indexed by timestep), so at runtime a step just does `patch_np_preds[global_timesteps]` — a lookup instead of re-running the transformer.
- **IMPALA with V-trace** → `experiments/train.py`
  - EdgeAIBus does **not** implement a custom agent — it uses Ray RLlib's built-in IMPALA configured with `vtrace=True, vtrace_clip_rho_threshold=2`. V-trace is an *off-policy* correction because many actor workers are each running a slightly different (stale) policy, and their experiences are replayed to one learner; V-trace re-weights those off-policy samples so the learner can still learn from them without bias.
- **Oversubscription** → `datacenter/Datacenter.py :: oversubscribed_cores`
  - This property sums all container **requested** CPU + base system CPU and divides by allocatable CPU. A ratio > 1.0 means more CPU is requested than physically exists — this is safe *only because* containers almost never use 100% of their requests simultaneously, so overbooking reclaims idle CPU. Memory is deliberately excluded because RAM overallocation causes fatal OOM (hard-killed) whereas CPU overallocation just slows things.
- **Risk-aware enforcement** → `datacenter/Datacenter.py :: minimum_availability_check()`
  - Before applying a migration on GKE, this cancels any `gke_action` that would push a node's requested resources over its allocatable — preventing the exact case Kubernetes rejects with `OutOfCPU`/`OutOfMemory`. It exists because the RL action is just "which (node, model)"; nothing guarantees that node actually has room, so a separate check must validate feasibility before the orchestrator acts.

---

## 3. Architectural Components

- **Gymnasium environment** → `scheduler/edgeaibus/envs/`
  - `SimEdgeEnv` (simulation) and `KubeEdgeEnv` (real GKE) both subclass `gym.Env` *and* `Scheduler`, giving the environment direct access to the PatchTST predictions for building the state vector.
- **DRL Engine (IMPALA)** → Ray RLlib (see `train.py`)
  - Actor-Learner split is provided by RLlib: `num_rollout_workers=2` actors run local policies and report experiences; the central learner updates the shared policy. The actor is what runs *during serving/inference* at each timestep.
- **Serving workload (Flask + YOLO)** → `yolov8/app.py`, `yolov8/Dockerfile`
  - Each edge node hosts a stateless Flask container running YOLO. The model selection action (`Nano` vs `Small`) maps to which `.pt` weights the container serves. Statelessness (documented in §3) makes migrations trivial — no dirty memory or persistent volumes to sync when a pod moves, which is why `migrate()` can just create a new pod and delete the old one.

---

## 4. State, Action, Reward

- **State $s(t)$** → `SimEdgeEnv.observation` + `Preprocessor`
  - The observation dict holds exactly the paper's components: `cpu_predictions` (PatchTST Ψ), `hosts_resources_alloc/req/usage` (N), `containers_request/usage` (P), `containers_accuracy` (ACL), `sla_violations` (S), one-hot `containers_hosts` (C). The `Preprocessor` normalizes each to [0,1] (dividing resource values by capacity, predictions by 100, and one-hot encoding placement). Normalization is required because the neural network inputs must be on comparable scales or the larger-magnitude features would dominate the gradient.
- **Action space $a_i = (n, m)$** → `edgeaibus/utils/mappers.py :: Mapper`
  - An integer action per container is decoded via `action2host_mapper[n] = (host_id, model_version)`. Encoding `(node, model)` as a single integer (`n = host*2 + model`) keeps the RL problem discrete and compact (6 hosts × 2 models = 12 values instead of a large combinatorial space), which is easier for the agent to explore.
- **Reward $r(t)$** → `rewards/reward.py`
  - **Illegal action (usage > 85%)** → returns `-10` (scaled by overload fraction) and zeroes the other three terms — a hard cliff that matches the spec exactly.
  - **Otherwise** → `reward_energy + reward_accuracy + reward_sla`, each rescaled to [0,1] and multiplied by its weight (10, 10, -4).
  - The 85%-cliff design explains the *known limitation* in AGENTS.md §6: because crossing 85% is catastrophic (the node fails/evicts), the agent becomes *defensively conservative* near 75–80% and drops models early to stay far from the cliff — motivating the proposed smooth log-barrier penalty.

---

## 5. Operational Rules

- **800 ms SLA threshold** → `datacenter/Datacenter.py :: yolo_sla()`
  - Counts, per container, how many of the 50 sampled YOLO processing-delay measurements exceed `0.8` seconds, then divides by 50 for the violation fraction. This 800ms is the strict delay SLA; exceeding it means the user-perceived latency bound is broken and triggers the negative SLA reward.
- **YOLO Medium excluded** → `yolov8/` + `datacenter` config
  - Only Nano and Small are set as `model_versions = 2` in the environment. Medium is dropped because the profiled YOLO data shows it exceeds 800ms for *every* core allocation — so letting the agent choose it would always lose (either SLA-fail or fail to fit), adding an action that can never be optimal.
- **Nano on ≥0.5 cores, Small on ≥1.0 cores** → YOLO profile CSVs (`datasets/yolo/`)
  - This "permissible allocation" rule emerges from the **profiled** YOLO data, not from hardcoded logic: `yolo_reading()` loads 6 CSVs (Nano/Small × 0.5/1.0/1.5 cores), each containing measured processing delays. Small at 0.5 cores fails SLA in the profile data, so the environment (via SLA check) makes it an unattractive/illegal choice.
- **85% fault threshold** → `Datacenter.num_overloaded`
  - In simulation, a node is overloaded when actual usage fraction `> 0.85`. In GKE it's `> 1.0` (requested vs allocatable), because on a real cluster Kubernetes enforces the hard limit at 100% request. The 85% simulation guard is *more conservative* than the 100% hardware limit — a safety margin because real usage is noisy and spiky.
- **Soft/hard resets** → `SimEdgeEnv.done`
  - Soft reset every `episode_length` (100) steps: resets `timesteps` but keeps placements (fresh RL episode, same world). Hard reset at `global_timesteps >= 7800`: the PatchTST trace sequence has expired, so the whole cluster/container config must be re-initialized and `global_timesteps` reset to 0.

---

## 6. Known Bottlenecks (found in code)

- **~100 ms transformer overhead** → `Scheduler.patchtst_pred()`
  - The auto-regressive loop re-runs `model.predict()` every timestep — multi-head attention is quadratic, ~100ms on a 1-core node. This is why predictions are **cached head-of-time into the pickle** for the sim, but a *live* runtime would still pay the cost.
- **Synchronous pipeline** → the env awaits PatchTST predictions every step
  - The proposed fix (async background loop refreshing every 30–60s while the agent acts on cached values) directly removes this per-step dependency.
- **Redundant actor inference for <1.0-core pods** → in `SimEdgeEnv`, the actor evaluates *all* containers, even those whose YOLO profile guarantees Small fails SLA. The proposed fast-path (route <1.0-core pods straight to Nano) avoids the wasted NN forward pass.
- **Coarse model degradation (2.54–3.81% accuracy loss)** → only two binary options (Nano/Small); the proposed dynamic resolution/quantization would give a finer, smoother accuracy-vs-latency tradeoff.
- **Missing queue metrics** → the `obs_elements` config has no request-backlog feature, so the agent can't see traffic building up — it only reacts *after* SLA violations appear.
- **Reward step cliff** → `reward.py` hard -10 at 85%; proposed smooth interior barrier `-μ·ln(0.85 − U)` would make the agent glide down instead of snapping.

---

## 7. Testing Hierarchy

- **Unit tests** (tensor shapes, $s(t) \in [0,1]$, rule cascades) → notebook checks in `experiments/*.ipynb`, `edgeaibus/utils/preprocessors.py` rounding decorator
- **Micro-benchmarks** (sub-15ms latency, thread contention) → `experiments/tpds_eab.sh` / `tpds_patchtst.sh` use `systemd-run` cgroups + RAPL for physical, controlled benchmarking
- **Integration rollouts** (1,000 steps) → `experiments/test.py` runs rollouts and logs 12 metrics per step (consolidation, moves, overloads, SLA, accuracy, conserved cores, reward components) to `states.csv`
- **End-to-end ablation (500k steps)** → `experiments/train.py` via Ray Tune with `stop={"timesteps_total": 1_000_000}`; results in `data/trainresults/`, compared against `MSB` / `GKO` baselines

---

### One-line TL;DR per bullet
- **Placement+model are joint** because they're coupled by node capacity → one `MultiDiscrete` action.
- **V-trace** corrects for the fact that many actors run stale policies.
- **Oversubscription is safe** because containers rarely jointly-hit their requests.
- **Risk check** pre-validates that a node actually has room, since the RL action doesn't.
- **Normalized state** so large-magnitude features don't dominate the network.
- **(node, model) encoded as one int** to keep exploration tractable.
- **-10 cliff at 85%** is why the agent gets defensive near 75–80%.
- **800ms SLA** is measured from profiled YOLO delays, not assumed.
- **Medium excluded** because it fails SLA at every allocation.
- **Soft reset** = new episode, same world; **hard reset** = trace exhausted.
