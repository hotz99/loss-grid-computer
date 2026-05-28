# THESIS MENTAL MAP (canonical-overview)

Companion to `main.tex`. The thesis as a logical chain:
gap → RQ → method → experiment, threaded so each layer
inherits its reason from the layer above. Bullets and
fragments, not prose. Trace any claim in the paper back
here for its upstream justification.

---

# 1. WHY

- Loss-grid construction dominates latency in interactive
  checkpoint comparison [xie2025losslens, losslens_repo].
- Canonical loss-grid implementation is MPI/cluster
  [li2018losslandscape, goldstein_loss_landscape_repo];
  cluster queue and setup latency conflict with the
  interactive workflow.
- → Evaluate single-machine optimization mechanisms for
  loss-grid construction under one GPU, available CPU
  cores, finite RAM and VRAM.

---

# 2. STRUCTURE

- Algorithm = loop over independent grid points →
  embarrassingly parallel.
- Two optimization axes (neither runs the other per grid
  point):
  - **A. intra-device per-grid-point**: compose grid
    points into a single device-side call via `vmap` and
    `torch.compile` → RQ1.
  - **B. inter-device hybrid scheduling**: split grid
    points across CPU and GPU workers under a shared queue
    → RQ2.
- One meta-axis tying them at the selection level:
  - **C. calibration cache**: cache the per-(machine,
    workload) hetero-config tuple, reuse across related
    checkpoints → RQ3.

---

# 3. AXIS A — intra-device (RQ1)

- **Gap**: PyTorch documents `functional_call`+`vmap` for
  model ensembling [pytorch_ensembling] (multiple models)
  and `torch.compile` [pytorch_compile] for general
  programs. No canonical applicability claim on the
  loss-grid pattern (one model, many perturbations) or on
  their composition.
- **Claim**: applicability is architecture-conditional;
  convolutional, sequential, and dense workloads give
  different per-row verdicts.
- **Method**: three candidates against the vanilla Li
  baseline [li2018losslandscape]: `vmapped` (sweeps
  point-chunk $K\in\{32,64\}$), `compiled`, and
  `compiled_vmapped` (intra-axis composition). Per-row
  Kalibera CI verdict on $T_{\text{grid}}$ speedup
  [kalibera2020effect]. Composition reported as
  descriptive CI ratios ($q_{\text{best}}$,
  $q_{\text{prod}}$).
- **Experiment**: per-workload speedup CI tables;
  section-level decomposition of vanilla $T_{\text{grid}}$
  motivates which mechanism has headroom on which section.

---

# 4. AXIS B — hybrid scheduling (RQ2)

- **Gap**: Gallet & Gowanlock [gallet2021heterogeneous]
  use the CPU/GPU throughput ratio as the partition-split
  parameter inside a static cost model on epsilon grid
  joins; non-ML workload, different question (how to
  split, not whether to combine). The applicability
  question for ML-inference loss-grid is ours. The
  first-principles argument: a CPU adder with throughput
  negligible against the GPU cannot reduce grid runtime
  under any scheduling policy; near parity, it can. Native
  $r$ on our available hardware does not span the regime
  where the question is interesting, so $r$ must be tipped
  via a controllable instrument.
- **Claim**: heterocompute applicability is
  $r$-conditional; row pattern across achieved $r$ reveals
  the regime boundary qualitatively (direction, no
  numeric threshold).
- **Method**: shared-queue dynamic self-scheduling at
  chunk size one. A task is one grid-point evaluation of
  the canonical algorithm; per-task work is uniform, so
  the imbalance term of the chunk-size trade-off
  [kruskal1985selfscheduling] is none. Richer schedulers
  exist (StarPU [augonnet2011starpu], HPC/scientific
  workloads) but uniform per-task work makes the simpler
  primitive sufficient. Slowdown instrument: added latency
  on the GPU worker path scaling by $s$ that tips the
  achieved $r$ toward parity. The queue-based scheduler
  sees only task-completion timing, so the sleep is
  sufficient at this granularity.
- **Experiment**: per-workload row of native $r$, slowdown
  factor $s$, achieved $r$, speedup CI versus
  slowdown-matched vanilla, CPU task share.

---

# 5. AXIS C — calibration cache (RQ3)

- **Gap**: in our shared-queue hybrid, effective $T_{cpu}$
  is a function of (CPU worker count $W$, CPU batch size
  $\text{BS}_{cpu}$) via two contention effects:
  - count vs. contention: more workers cause super-linear
    per-worker throughput drop past some $W^*$.
  - batch size vs. memory pressure: larger
    $\text{BS}_{cpu}$ amortizes dispatch but inflates
    per-worker working set, accelerating contention at
    the same $W$.
  ML inference per-task work is forward-pass tensor
  evaluation, memory-bandwidth-bound at small per-task
  batches, so the contention surface is steeper than for
  general parallel work and is per-(machine, workload)
  specific. Prior work supplies parts but not the
  signature: Polychronopoulos [polychronopoulos1987gss]
  treats worker count as a deployment input; ATLAS
  [whaley1998atlas] establishes tune-cache-reuse at
  BLAS-op signature; OpenTuner [ansel2014opentuner]
  supplies bounded-search discipline. → Gap: how to pick
  $(W, \text{BS}_{cpu})$ per (machine, workload) on
  ML-inference loss-grid.
- **Claim**: one bounded calibration plus cached
  hetero-config tuple reuse across $N=4$ related
  checkpoints can reduce cumulative $T_{\text{session}}$
  vs. per-checkpoint vanilla in a regime with hybrid affinity,
  native or slowed.
- **Method**: calibrate hetero-config tuple (policy, $W$,
  $\text{BS}_{cpu}$). Autotune-and-cache pattern
  [whaley1998atlas] with signature lifted from BLAS-op to
  (machine, workload). Bounded
  $k$-consecutive-non-improvement patience sweep
  [ansel2014opentuner]. Conditioned on a fixed
  `rq3_config` that is validated by RQ1 on the
  selected workload and platform.
- **Experiment**: regime selected by filtering Experiment 2
  for hybrid affinity (native preferred, slowed fallback);
  the inherited slowdown is the operating point. Per-row:
  operating point, selected tuple, calibration cost,
  vanilla and cached session times, cumulative ratio,
  break-even $N^*$, three-way amortization label
  (supported, asymptotic-only, refuted). `gpu_only`
  selection in the tuple is a sweep result, not a default.

---

# 6. SHARED

- **Measurement**: `time.perf_counter_ns` with backend
  completion synchronized [python_perf_counter_ns,
  pep_418]. Paired repeats with rotated trial order;
  Kalibera CI verdict for RQ1 and RQ2
  [kalibera2013rigorous, kalibera2020effect]. RQ3 is
  single-measurement and descriptive. Correctness gate:
  `torch.allclose` defaults [pytorch_allclose] on finite
  losses, paired NaNs at matching coordinates count as
  agreement, grid count and coordinates must match
  baseline exactly. Surface failure suppresses the timing
  claim.
- **Instantiation**: two platforms (Apple M4 unified
  memory, 10 logical CPU cores; NVIDIA T4 discrete VRAM,
  2 logical CPU cores). Four workloads
  (`cifar10_resnet20`, `cifar10_row_gru`, `california_mlp`,
  `mnist_mlp`) for RQ1 and RQ2. Grid $8\times 8$ for RQ1
  and RQ2; $40\times 40$ session grid for RQ3 (RQ3
  excludes `cifar10_resnet20` for budget). $R=5$ paired
  repeats for RQ1 and RQ2; single-measurement for RQ3.

---

# 7. CONTRIBUTIONS

- **RQ1**: per-candidate applicability of
  `functional_call`+`vmap` and `torch.compile` to Li's
  loss-grid algorithm across the workload set, plus the
  intra-axis composition verdict on
  `vmap × torch.compile`.
- **RQ2**: dynamic self-scheduling hybrid for CPU/GPU
  loss-grid; applicability gated by the CPU/GPU
  inference-throughput ratio on ML-inference loss-grid.
  The applicability framing is ours; Gallet supplies a
  bounded related-work pointer for the variable on a
  different workload.
- **RQ3**: calibrate-once-cache for the hetero scheduler
  configured on top of a controlled A config validated by
  RQ1, reused across related checkpoint sessions.
  One-time calibration treated as part of the measured
  workflow; cumulative session speedup ratio plus
  break-even $N^*$ as practical guide.
- Baseline-wins reported as primary findings alongside
  optimization wins.
