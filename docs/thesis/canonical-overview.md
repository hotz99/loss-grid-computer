# WHY

- Loss-grid is the bottleneck in interactive checkpoint comparison
  [xie2025losslens; losslens_repo]
- Canonical loss-grid implementation is MPI/cluster
  [li2018losslandscape, goldstein_loss_landscape_repo]
- Interactive use requires single-machine deployment
  (cluster queue + setup latency conflicts with interactive workflow)
- → Optimize loss-grid construction under single-machine constraint

# PROBLEM

- Empirical contract (shared across A, B, C):
  - measure host-observed elapsed caller-side wall time of a
    loss-grid computation on one machine — T_grid (one grid after
    inputs are ready); T_session (multi-checkpoint session,
    including any one-time calibration the measured arm performs)
  - compare methods by speedup with uncertainty, not by isolated
    raw times
  - preserve the computed loss surface relative to the baseline
- Surface budget (correctness gate; precondition for any speedup
  claim) = pre-specified relative surface-equivalence on the
  computed loss values (definition under MECHANISMS/MEASUREMENT;
  stricter than the visual-equivalence convention used by
  [li2018losslandscape, goldstein_loss_landscape_repo])
- Constraints: 1 GPU, CPU cores, RAM/VRAM

# STRUCTURE (no cite — structural)

- Algorithm = loop over independent grid points → embarrassingly parallel
- Two optimization axes (independent at the mechanism level — neither
  runs the other per grid point; C ties them at the selection level):
  - A. intra-device per-grid-point optimization (compose grid points
    into a single device-side call: batched transform via vmap,
    and/or graph capture via torch.compile) → RQ1
  - B. inter-device hybrid scheduling (split grid points across CPU
    and GPU workers under a shared queue) → RQ2
- Both axes have free parameters that depend on (machine, workload):
  - A: chunk size K, compile mode
  - B: CPU worker count, CPU batch size, policy choice
    (gpu_only vs gpu_cpu_hybrid)
- → C. meta-axis: cache the per-(machine, workload)
  hetero-config tuple calibration (configured on top of A's RQ1
  winner for that workload) and reuse across same-family checkpoints
  → RQ3.
  Composition with A is at the deployed-stack level: the session
  runtime assembles A's RQ1 winner + C's cached hetero-config
  tuple. C does not re-sweep A.

# MECHANISMS (methods-only; direct ancestors)

- A → PyTorch program-transformation APIs (intra-device, per-grid-point)
  - stateless invocation → pytorch_functional_call (prerequisite
    primitive; not a standalone candidate — no independent
    optimization claim, and the candidates that build on it subsume
    its observable runtime contribution)
  - batched transform → pytorch_vmap (auto-batched function
    transform; documented composition with functional_call is
    pytorch_ensembling — stacked-state evaluation)
  - graph capture → pytorch_compile (ahead-of-time graph
    capture, kernel fusion, and dispatch elision; orthogonal
    intra-device optimization on the same per-grid-point work,
    composable with vmap)
  - target algorithm → li2018losslandscape (2D filter-
    normalized loss grid)
- B → Eager shared-queue hybrid scheduler (same code both devices)
  - applicability variable → gallet2021heterogeneous (throughput ratio
    r = T_cpu/T_gpu as core variable of heterocompute applicability)
  - scheduler policy → polychronopoulos1987gss (dynamic
    self-scheduling — work pulled by whichever device is free)
  - task granularity → grid-point-as-task (the natural unit of
    the embarrassingly-parallel algorithm; no aggregation needed
    because per-point work is uniform)
  - sweep instrument → mei2017gpudvfs (controllable GPU
    throughput knob for sensitivity sweep; their mechanism is DVFS,
    ours is the software analog — post-kernel delay injection — used
    to sweep r across regimes the native workload set does not span.
    This is a methodological device, not a scheduling mechanism; the
    variable being swept is Gallet's r)
- C → Calibration cache across same-family sessions
  - same-family predicate → same workload signature (architecture +
    dataset + loss + grid spec), different checkpoint contents;
    analogous to ATLAS's same-op-signature reuse condition
    (op = BLAS routine like GEMM)
  - search space → hetero-config tuple only: {gpu_only,
    gpu_cpu_hybrid}, with GPU batch size, CPU worker count, and
    CPU batch size as tuple fields. Device asymmetry under
    hybrid scheduling: the CPU worker runs the baseline sequential path;
    A's choices apply to GPU only (vmap/compile semantics on small
    per-task CPU slices are out of scope). A is not in the sweep —
    its per-(machine, workload) winner is established by RQ1 and
    consumed here as a configuration input.
  - composition with A → at the deployed-stack level, not the
    sweep level: the session runtime assembles (A's RQ1 winner) +
    (C's calibrated hetero-config tuple). Calibrating the
    hetero-config tuple on top of A's winner is non-trivially
    different from calibrating it on top of vanilla — once A
    shrinks GPU time, the break-even for adding CPU workers shifts,
    so C's calibrated tuple is conditioned on which A is configured.
    This is why C's cache is keyed on
    (machine, workload) — the same key A is keyed on.
  - autotune+cache pattern → whaley1998atlas (the canonical
    tune/cache/reuse pattern; RQ3 adapts it from BLAS-op reuse to
    workload-family reuse)
  - bounded autotuning search → ansel2014opentuner (search budget
    with stopping criterion as the canonical departure from
    ATLAS-style exhaustive search; our implementation is a
    k-consecutive-non-improvement patience rule over the hetero-config
    sub-sweep — one instance of OpenTuner's bounded-search
    discipline)
- MEASUREMENT (shared across A, B, C):
  - clock primitive → python_perf_counter_ns / pep_418
    (portable monotonic high-resolution elapsed time; host-observed
    wall time including all caller-visible costs — Python
    orchestration, framework dispatch, backend execution, parameter
    binding, dataloader iteration, result collection, and
    synchronization. Asynchronous backend completion is forced
    through the framework backend abstraction before stopping the
    timer; no CUDA-event-specific dependency. PyTorch Timer and
    CUDA events are not root canon for this thesis.)
  - runtime targets → T_grid (one loss-grid after
    inputs are ready) and T_session (multi-checkpoint session,
    including any one-time calibration the measured arm performs;
    framed at the experiment level — the artifacts are a means to
    conduct experiments, not productized — see WHERE THIS SITS /
    RQ3 stance on install-time-vs-experiment). Every speedup claim
    names the target it uses.
  - section decomposition → T_grid = T_perturb + T_bind +
    T_eval + T_collect + residual; same wall-time primitive on
    each section; explanatory only, not a separate research
    method.
  - repeats + uncertainty → kalibera2013rigorous (repeated
    executions, variation estimates, and repetitions/stopping
    budget — the discipline for getting from raw timings to a
    defensible mean with uncertainty)
  - effect-size CI → kalibera2020effect (speedup as
    a ratio / effect-size with confidence interval — the verdict
    statistic for RQ1 and RQ2; RQ3 inherits it only if full
    sessions are repeated, else RQ3 is descriptive session
    evidence)
  - trial-order + pairing → baseline and candidate evaluated
    within the same repeat under identical workload, seed, and
    grid; trial order rotates across repeats; no fixed-order
    benchmarks. Convention disclosed per experiment.
  - verdict rule → CI_low > 1.0 ⇒ supported
    speedup; CI_high < 1.0 ⇒ supported regression; otherwise
    inconclusive. Same rule across A, B, and (when repeated) C.
  - correctness gate → surface invalid ⇒ no speedup claim
    regardless of timing. The gate confirms that a candidate
    implementation produces the same loss surface as the
    baseline on the same (workload × backend × checkpoint).
    Different implementations of the same nominal float32
    computation produce different numeric outputs in practice
    (op reordering, kernel/fusion choice, hybrid CPU/GPU
    split), so the gate is a relative-equivalence criterion
    rather than bitwise equality. Three parts: (i) grid point
    count and coordinates match the baseline exactly;
    (ii) paired NaNs at matching coordinates count as
    agreement; (iii) finite losses satisfy
    `|a − b| ≤ 1e-5 · max(|a|, |b|)` (rtol = 1e-5 follows
    torch.allclose's default [pytorch_allclose]; atol = 0, so
    equivalence requires proportionally small drift rather
    than an absolute floor).

# CHARACTER

- Each axis's payoff workload-dependent
- → Applicability verdict per axis × workload from MECHANISMS/MEASUREMENT:
  - {applies, regression, inconclusive} from the effect-size CI rule
    [kalibera2020effect] over paired trials on the named runtime
    target (T_grid for A/B, T_session for C when repeated)
  - {invalid-surface, insufficient-data} as preconditions
    (surface budget = pre-specified relative-equivalence gate
    defined in MECHANISMS/MEASUREMENT; preconditions gate the
    timing categories — see PROBLEM)
- For axis A specifically, the per-candidate verdict is paired with
  an **intra-axis composition verdict** ∈ {multiplicative, redundant,
  destructive, inconclusive}, derived by comparing the composed
  candidate's speedup to the product of the individual candidates'
  speedups under the same effect-size CI rule on the ratio.
- For axis C specifically, the per-axis verdict asks whether the
  cached hetero-config tuple amortizes: one bounded calibration on
  checkpoint 0 plus cached reuse across same-family checkpoints
  must reduce cumulative T_session versus the vanilla session
  reference. Break-even N reports the session size needed to pay
  for calibration.
- Baseline-wins (regression) reported as primary findings

# EXPERIMENTS (one design per RQ; verdict from CHARACTER)

Each experiment instantiates the shared MECHANISMS/MEASUREMENT method.
The experiment-specific surface is workload set, platforms, pairing
scheme, trial order, repetitions/stopping budget, and surface-
validation threshold. The statistical rule, clock primitive, runtime-
target vocabulary, and correctness gate are shared.

- preflight (no RQ): platform inventory + functional-API probe

- motivation (substantiates WHY): section-level decomposition of
  T_grid on the vanilla baseline per workload (T_perturb + T_bind +
  T_eval + T_collect + residual). Uses the shared wall-time
  primitive; not a separate timing mechanism. Shows which section
  dominates per workload and produces the shared baseline T_grid
  artifact reused by RQ2.

- RQ1 (axis A, intra-device per-grid-point optimization):
  three candidates against the vanilla baseline per workload on
  T_grid, paired repeats:
  - `vmapped` — functional_call + vmap [pytorch_ensembling], with a
    point-chunk sweep K for breadth
  - `compiled` — torch.compile on the baseline eval path
    [pytorch_compile]
  - `compiled_vmapped` — torch.compile applied over the vmapped
    candidate (intra-axis composition)
    Per-candidate verdict from CHARACTER on
    the T_grid speedup effect size [kalibera2020effect]. Intra-axis
    composition verdict from CHARACTER: do vmap and torch.compile
    multiply, are they redundant on this workload pattern, or do they
    interfere destructively?

- RQ2 (axis B, inter-device):
  measure r_native per workload (unslowed CPU and GPU throughputs);
  sweep r across regimes via the GPU-slowdown instrument
  [mei2017gpudvfs]; at each swept r, run the dynamic self-scheduling
  hybrid and compare against the slowdown-matched vanilla GPU
  baseline on T_grid. Verdict from CHARACTER at each r; predictor
  falsification (r ≥ 1 but hybrid CI fails) reported separately.
  r_native is the diagnostic that locates each workload on the
  swept axis — it is a predictor, not the speedup statistic.

- RQ3 (axis C, meta on B; composes with A at use time):
  configure the GPU-side implementation as A's RQ1 winner for this
  (machine, workload); calibrate once on one checkpoint by sweeping
  hetero-config tuples with patience-based termination; cache the resulting
  (machine, workload) → hetero-config tuple selection; reuse across the
  remaining same-family checkpoints; compare against per-checkpoint
  vanilla on T_session.
  - **Cumulative T_session speedup** at the LossLens-scoped session
    size — headline; descriptive evidence when sessions are measured
    once, CI-promoted when repeated under the shared rule
    [kalibera2020effect].
  - **Break-even N** — sessions needed to amortize the
    hetero-config calibration cost; diagnostic.

- synthesis: applicability table per axis × workload using the
  CHARACTER taxonomy.

# RELATED WORK MAP

## Loss-landscape visualization (rw-losslandscape)

- Goodfellow 1D linear interpolation [goodfellow2015qualitative]
- Li et al. 2D filter-normalized loss grid (baseline) [li2018losslandscape]
- Goldstein MPI reference implementation [goldstein_loss_landscape_repo]
- LossLens visual-analytics workflow + released code [xie2025losslens,
  losslens_repo]

## Adaptive sampling (rw-adaptive-sampling)

- Prakash, Wang, Balestriero — adaptive vs. uniform [prakash2024adaptive]

## Heterogeneous CPU/GPU scheduling (rw-scheduling)

- Para 1 — task-parallel axis:
  - Polychronopoulos & Kuck 1987 [polychronopoulos1987gss]
    canonical dynamic self-scheduling family (this thesis's policy)
  - Gallet & Gowanlock 2021 [gallet2021heterogeneous]
    empirical identification of throughput ratio as the core
    variable of heterocompute applicability (this thesis's variable)
  - Augonnet et al. 2011 (StarPU) [augonnet2011starpu]
    canonical heterocompute scheduling system; this thesis's
    scheduler is the simpler dynamic-self-scheduling variant of
    the StarPU class, sufficient because loss-grid is embarrassingly
    parallel with uniform per-task work
- Para 2 — model-parallel/pipeline axis (orthogonal):
  - OpenVINO HETERO [openvino_hetero]

## Autotuning (rw-autotuning)

- Whaley & Dongarra — ATLAS [whaley1998atlas]
  canonical autotune-and-cache pattern: at install time, empirically
  tunes the inner loops of BLAS routines (GEMM, DAXPY, etc.) for the
  target machine's cache and register hierarchy; caches the selection
  by (machine, BLAS-op) signature; reuses across every subsequent
  matching call. RQ3 inherits the four-part pattern shape (tune once
  per machine, cache by signature, reuse across matching invocations,
  per-machine scoping) and lifts the signature from op-level (BLAS op)
  to workload-level (architecture + dataset + loss + grid spec).
- Ansel et al. — OpenTuner [ansel2014opentuner]
  canonical framework for bounded autotuning search with explicit
  stopping criteria; the discipline-level departure from
  ATLAS-style exhaustive search. RQ3 inherits an instance of that
  discipline (implementation details in methods).

# WHERE THIS THESIS SITS (closing of related work)

- task-parallel axis on ML-inference loss-grid
- same-domain orthogonal axis: OpenVINO HETERO (out of scope)
- RQ1 stance:
  PyTorch documents the functional_call+vmap idiom on model
  ensembling (independently-trained models). Loss-grid has the same
  outer-loop shape but a different semantic (one model, many
  perturbations). torch.compile is an orthogonal intra-device
  optimization on the same per-grid-point work, documented for
  general PyTorch programs but with no canonical composition
  alongside vmap on the loss-grid pattern. RQ1 tests three
  candidates: vmap alone, torch.compile alone, and the composition.
  The semantic-gap test applies to each candidate independently;
  the intra-axis composition test asks whether the two mechanisms
  multiply, are redundant, or destructively interfere on this
  workload pattern.
- RQ2 stance:
  Gallet on set ops identifies the throughput ratio as the
  applicability variable for heterocompute hybrid scheduling. This
  thesis inherits the variable and pairs it with dynamic self-
  scheduling [polychronopoulos1987gss] as the policy — the simpler
  variant of the StarPU-class heterocompute scheduling system
  [augonnet2011starpu], sufficient because loss-grid is embarrassingly
  parallel with uniform per-task work. The RQ2 test is whether r
  gates applicability on ML-inference loss-grid. Gallet's policies,
  device-specific algorithms, and workload domain are not inherited.
- RQ3 stance:
  ATLAS [whaley1998atlas] is the pattern source for RQ3, not an
  implementation model. RQ3 keeps the tune/cache/reuse idea and
  per-machine scoping, but changes the unit of reuse from a BLAS op
  to a same-family workload signature (architecture + dataset + loss
  + grid spec). The calibrated object is also different: a
  hetero-config tuple configured on top of RQ1's selected
  intra-device policy, found with a bounded OpenTuner-style search
  rule [ansel2014opentuner] and evaluated as experiment-time
  calibration rather than install-time library generation.

  RQ3's falsifiable claim: the autotune-and-cache pattern's
  amortization survives the signature lift — the cached hetero-config tuple
  (on top of A's RQ1 winner) stays valid across same-family
  checkpoints (cumulative T_session speedup positive at the
  LossLens-scoped session size) and break-even N falls within that
  scope.

- The empirical study exercises a commodity single-machine envelope
  (free-tier cloud GPU at one end, consumer laptop at the other) so
  applicability findings sit on a realistic deployment surface for
  interactive use [xie2025losslens]. The specific hardware tiers and
  per-tier verdicts are reported in the discussion; the canon makes
  no hardware-pinned claims.

# CONTRIBUTIONS (intro)

- RQ1: PyTorch intra-device transforms (functional_call+vmap and
  torch.compile) applied to Li's loss-grid algorithm —
  per-candidate applicability across the workload set and
  intra-axis composition verdict on vmap × torch.compile
- RQ2: dynamic self-scheduling hybrid for CPU/GPU loss-grid —
  applicability gated by Gallet's throughput-ratio variable
- RQ3: calibrate-once-cache for the hetero scheduler (configured on
  top of A's RQ1 winner) across same-family sessions — cumulative
  session speedup headline; break-even N as context. Composition with
  A is at use time, not at sweep time.
- Baseline-wins reported as primary findings alongside optimization wins
