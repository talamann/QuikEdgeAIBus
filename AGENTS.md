```markdown
# Context: EdgeAIBus Repository Analysis & Optimization

This context document serves as the operational and architectural reference for an AI coding agent analyzing, debugging, or optimizing the **EdgeAIBus** framework repository[cite: 1].

---

## 1. System Overview & Core Objectives

**EdgeAIBus** is an AI-driven, proactive framework for joint container management (placement, consolidation, oversubscription) and ML/DL model switching targeting heterogeneous edge computing environments[cite: 1, 3].

* **Repository**: `https://github.com/BabarAli93/EdgeAIBus`[cite: 1, 3]
* **Target Environment**: Heterogeneous edge nodes (2-, 4-, 6-core servers) hosting containerized object detection services[cite: 1, 3].
* **Core Optimization Balance**:
  $$\min \sum_{t=1}^T (E(t) + S(t) - A(t))$$
  Subject to real usage remaining below the failure threshold ($U_{\text{node}} \le 85\%$)[cite: 1, 3].
* **Key Mechanisms**:
  1. **Multivariate CPU Forecasting**: Time-series prediction of node-level utilization via PatchTST[cite: 1, 3].
  2. **Joint Reinforcement Learning Policy**: Off-policy distributed Actor-Critic (IMPALA with V-trace) selecting both container node placement and hosted model version per timestep[cite: 1, 3].
  3. **Oversubscription & Consolidation**: Aggressive container request-based overbooking paired with migration to empty nodes for energy savings[cite: 1, 3].
  4. **Risk-Aware Enforcement**: Admission risk check modules preventing Kubernetes `OutOfCPU` / `OutOfMemory` errors[cite: 1, 3].

---

## 2. Architectural Components


```

```
            +------------------------------------+
            |        PatchTST Predictor          | (Multivariate CPU forecast
            |     (Bitbrains trace data)         |  horizon τ = 6)
            +-----------------+------------------+
                              |
                              v

```

+------------------+     +-------------------+     +-------------------------+
| Profiled Data /  | --> | OpenAI Gymnasium  | <-- | Heterogeneous Edge Nodes|
| YOLO Inference   |     |    Environment    |     | (Simulation or GKE)     |
+------------------+     +---------+---------+     +-------------------------+
|
State s(t)
v
+-------------------+
|  IMPALA Agent     |
|  (Actor-Learner)  |
+---------+---------+
| Joint Action a_i(n, m)
v
+---------------------+---------------------+
|                                           |
v                                           v
[Container Placement / Migration]          [Model Switching]
(Target: Consolidate / Oversubscribe)      (YOLOv8 Nano vs. Small)

```

### Component Details
* **PatchTST Predictor**: Auto-regressive transformer predicting CPU usage over a horizon $\tau = 6$ using the Bitbrains dataset[cite: 1, 3]. Achieves 8.51% RMSE and 5.62% MAE[cite: 1, 3].
* **Gymnasium Environment**: Custom reinforcement learning environment modeling edge nodes, containers, resource constraints, and latency violations[cite: 1, 3].
* **DRL Engine (IMPALA)**: Distributed actors running local policies $\mu$ reporting experiences to a central learner running policy $\pi_\theta$ using V-trace importance sampling correction[cite: 1, 3].
* **Serving Workload**: Stateless Flask API instances serving Ultralytics YOLOv8[cite: 1, 3].

---

## 3. Operational Rules & Constraints

* **Strict Delay SLA**: The processing time threshold $sv_{th}$ is fixed at **800 ms**[cite: 1, 3].
* **Model Selection Space**:
  * **YOLOv8 Medium**: Excluded from active scheduling because it fails the 800 ms SLA under all edge allocations (0.5, 1.0, 1.5 cores)[cite: 1, 3].
  * **YOLOv8 Small**: Permissible on $\ge 1.0$ core allocations[cite: 1, 3].
  * **YOLOv8 Nano**: Permissible across all core allocations ($\ge 0.5$ cores)[cite: 1, 3].
* **Server Fault Threshold**: Real cluster node utilization must stay below **85%**[cite: 1, 3].
  * Actions exceeding 85% real usage are marked **illegal**, immediately triggering a $-10$ reward penalty while setting objective rewards to 0[cite: 1, 3].
* **Workload Statelessness**: Containers are strictly stateless; persistent volumes and dirty memory synchronization are not modeled during migrations[cite: 1, 3].
* **Environment Resets**:
  * **Soft Reset**: Occurs every 100 timesteps; resets step count while preserving agent environment state[cite: 1, 3].
  * **Hard Reset**: Occurs upon expiration of the PatchTST dataset sequence; fully reinitializes cluster and container configurations[cite: 1, 3].

---

## 4. State, Action, and Reward Specifications

### State Space $s(t)$
Vector normalized to $[0, 1]$ aggregating[cite: 1, 3]:
1. $\Psi(t)$: Multi-step ahead PatchTST CPU predictions for all nodes[cite: 1, 3].
2. $N_{alloc}, N_{req}, N_{util}$: Allocatable, requested, and utilized CPU/RAM across edge servers[cite: 1, 3].
3. $P_{req}, P_{util}$: Container-level requested and utilized resources[cite: 1, 3].
4. $ACL(t-1)$: Accuracy of active YOLO models from previous step[cite: 1, 3].
5. $S(t-1)$: SLA violations incurred at previous step[cite: 1, 3].
6. $C$: One-hot encoded placement matrix of containers on nodes[cite: 1, 3].

### Action Space $a(t)$
Multi-discrete joint action vector mapping each container $p \in P$ to a tuple:
$$a_i = (n, m)$$
where $n \in N$ is the target node ID and $m \in \{\text{Nano}, \text{Small}\}$ is the model version[cite: 1, 3].

### Reward Function $r(t)$
$$r(t) = \begin{cases} -10 & \text{if } U_{\text{node}} > 85\% \text{ (Illegal Action)} \\ r_A(t) + r_S(t) + r_E(t) & \text{otherwise} \end{cases}$$
[cite: 1, 3]

* $r_A(t) = rf_A \cdot A(t)$: Weighted by mean system accuracy[cite: 1, 3].
* $r_S(t) = rf_S \cdot S(t)$: Penalizes fraction of requests exceeding 800 ms[cite: 1, 3].
* $r_E(t) = rf_E \cdot (w_1 \cdot U(t) + w_2 \cdot cr(t))$: Encourages consolidation ratio $cr(t) = \kappa / |N|$ and active node utilization $U(t)$[cite: 1, 3].

---

## 5. Directory & Codebase Navigation Guide

When exploring the codebase, prioritize these functional boundaries:
* **Forecasting Pipeline (`models/` or `forecasting/`)**: Contains PatchTST implementation, Bitbrains dataset loaders, rolling window auto-regressive inference loops, and baseline GRU/HRF-ExGB implementations[cite: 1, 3].
* **Gymnasium Environment (`envs/` or `gym_edge/`)**: Contains custom OpenAI/Farama Gymnasium environments, observation space normalization routines, illegal action checks, and reward penalty calculations[cite: 1, 3].
* **DRL Agents (`agents/` or `rl/`)**: Contains IMPALA actor/learner architectures, V-trace off-policy corrections, and experience replay buffer management[cite: 1, 3].
* **Cluster Orchestration & Risk Check (`orchestrator/` or `k8s/`)**: Implements the GKE Python client API integration, pod migration logic, Flask endpoint interfaces, and the **Risk Check** module preventing `OutOfCPU`/`OutOfMemory` admission failures[cite: 1, 3].

---

## 6. Optimization Analysis & Known Bottlenecks

### Speed & Resource Bottlenecks
* **Transformer Overhead**: Auto-regressive multi-head attention in PatchTST causes the decision cycle to take ~100 ms on 1-core edge nodes[cite: 1, 3].
  * *Target Improvement*: Replace with direct-horizon DLinear or quantized ONNX models to compute predictions in a single pass without quadratic attention complexity[cite: 1, 2].
* **Synchronous Pipeline Execution**: IMPALA awaits PatchTST forecasting at every step[cite: 1].
  * *Target Improvement*: Decouple forecasting into an asynchronous background loop updating every 30–60 seconds while IMPALA acts on cached predictions[cite: 1, 2].
* **Redundant DRL Inferences**: Evaluating actor neural networks for pods with allocations $< 1.0$ core where YOLO Small deterministically fails SLA[cite: 1, 3].
  * *Target Improvement*: Fast-path rule evaluation mapping $< 1.0$ core pods directly to Nano[cite: 1, 2].

### Accuracy & SLA Bottlenecks
* **Coarse Model Degradation**: Binary switching between Nano and Small introduces a 2.54%–3.81% accuracy loss[cite: 1, 3].
  * *Target Improvement*: Dynamic input resolution scaling ($640 \to 480 \to 320$) and dynamic quantization (FP16/INT8)[cite: 1, 2].
* **Missing Queue Metrics**: State vector $s(t)$ lacks request backlog or payload complexity features, delaying model downgrades during traffic spikes[cite: 1, 3].
  * *Target Improvement*: Ingest container request queue depth into $s(t)$[cite: 1, 2].
* **Reward Step Cliff**: The hard $-10$ penalty at 85% node usage causes defensive model drops near 75%–80%[cite: 1, 3].
  * *Target Improvement*: Smooth interior barrier penalty (e.g., $-\mu \ln(0.85 - U_{\text{node}})$)[cite: 1, 2].

---

## 7. Testing & Validation Hierarchy

When developing or modifying components in this repository, follow this execution ladder:

1. **Unit Tests**:
   * Verify input/output tensor shapes (e.g., DLinear outputting $[B, \tau, C]$)[cite: 1].
   * Verify state normalization boundaries ($s(t) \in [0, 1]$)[cite: 1, 3].
   * Verify rule cascades for constrained containers[cite: 1].
2. **Micro-benchmarks**:
   * Profile single-thread inference time using `taskset -c 0` to verify sub-15 ms latency[cite: 1].
   * Benchmark thread contention on asynchronous observation queues[cite: 1].
3. **Gymnasium Integration Tests**:
   * Run 1,000-step simulated rollouts to confirm action validity, soft/hard reset handling, and absence of NaN values in V-trace updates[cite: 1, 3].
4. **End-to-End Ablation Training**:
   * Execute 500k-step IMPALA training runs comparing learning curves, consolidation core count, SLA violation rates, and cumulative rewards against the paper baselines[cite: 1, 3].

```/s