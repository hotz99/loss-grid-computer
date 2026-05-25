# Experiment B: Hybrid Scheduler Applicability (RQ2)

## RQ2 connection

> **RQ2.** Under what conditions do idle CPU cores reduce total
> loss-grid time, and how is the CPU/GPU inference throughput ratio
> associated with those conditions?

Test whether the eager shared-queue hybrid CPU/GPU scheduler reduces
T_grid versus a vanilla GPU-only baseline, and whether Gallet's
throughput-ratio variable [gallet2021heterogeneous] predicts where
hybrid applies.

## Method (knobs only; protocol shared from canon)

See `canonical-overview.md` MECHANISMS/MEASUREMENT for the shared
protocol (clock primitive, runtime-target vocabulary, verdict-rule
template, trial-order / pairing convention, surface-budget
definition, no-CUDA-event canon).

This experiment's knobs:

- **Verdict statistic.** T_grid speedup
  `speedup_i = T_vanilla_i / T_hybrid_i` at the achieved ratio;
  canon CI rule [kalibera2020effect].
- **Repetitions.** R = 5 paired repeats per (workload × platform × r).
  Smallest R yielding df = R − 1 = 4 for a usable paired t-CI on
  speedup ratios; canon repetition discipline applies above this
  floor.
- **Pairing.** At each swept r, the hybrid and the slowdown-matched
  vanilla baseline are evaluated within the same repeat under the
  same workload, grid, seed, and slowdown factor s.
- **Trial-order rotation.** Hybrid/vanilla order alternates across
  repeats.
- **Surface gate.** Per canon MEASUREMENT/correctness gate:
  point count + coordinates match exactly; paired NaNs count as
  agreement; finite losses satisfy
  `|a − b| ≤ 1e-5 · max(|a|, |b|)` (rtol = 1e-5, atol = 0). Zero
  mismatches required across the grid.

The CPU/GPU throughput ratio r is a **predictor**, not the speedup
statistic [gallet2021heterogeneous]. It locates each workload on the
swept axis but does not appear in the verdict rule. Gallet's result
transfers the variable, not a universal threshold; the threshold is
workload-specific.

**B's GPU-side implementation in this experiment is vanilla**, not
RQ1's winner. RQ2 measures B's per-axis ground truth — the
contribution of CPU workers around the vanilla GPU path. Composition
with A is RQ3's at-use-time concern (canon MECHANISMS/C); pre-mixing
A's winner here would conflate B's effect with A's. The per-r best
B-cell reported by RQ2 is descriptive context — RQ3 runs its own
bounded calibration on top of A's RQ1 winner and does not consume
RQ2's cell at execution time.

## Experiment-specific surface

- **Workload set.** cifar10_resnet20, cifar10_row_gru, california_mlp,
  mnist_mlp.
- **Platforms.** Apple M4 (mps), NVIDIA Tesla T4 (cuda).
- **Grid.** 8×8 (64 queued tasks), scale 1.0. 64 tasks is enough
  for the dynamic self-scheduling queue to expose CPU/GPU work
  placement (multiple tasks per worker) while keeping the
  (workload × platform × r × R) sweep bounded.

## Sweep instrument (Gallet's r, controlled by the DVFS analog)

`r_native = throughput_cpu / throughput_gpu` is measured per workload
without slowdown. To sweep r across regimes the native workload set
does not span, the GPU work path receives a controlled post-kernel
delay — the software analog of the DVFS-sweep methodology
[mei2017gpudvfs]. The slowdown factor s targets parity:
`s = 1 / r_native` when `r_native < 1`, else `s = 1.0`.

The instrument controls the swept parameter (Gallet's r); it does not
model native execution and is not a competitor for hardware-faithful
simulators. The claim at `s > 1` is scoped to that constructed
operating point.

## Conditions to compare

| Condition | Role | Configuration |
| --- | --- | --- |
| `vanilla_baseline` | comparator at the achieved ratio | `VanillaMode(gpu_batch_size)`, GPU slowdown = s |
| `hybrid` | proposed scheduler at the achieved ratio | calibrated `HybridMode(gpu_batch_size, cpu_batch_size, cpu_workers)`, GPU slowdown = s |

The unslowed measurement (`r_native`, CPU throughput, GPU throughput)
is a **diagnostic** that locates each workload on the swept axis. It
is not a comparison condition and must not contribute to the headline
speedup column.

## Diagnostics (no verdict role)

- **`r_native`** — regime predictor [gallet2021heterogeneous].
- **`achieved_ratio`** — realized CPU/GPU ratio under slowdown.
  `slowdown_distance_from_unity = |achieved_ratio − 1.0|` quantifies
  how close the parity setup actually reached parity.
- **`selected_policy`** — `gpu_only` vs `gpu_cpu_hybrid` chosen by
  calibration. `gpu_only` at the achieved ratio is a negative result
  for the predictor.
- **`worker_throughput_split.cpu_fraction`** — fraction of grid
  points actually executed on CPU.

## Falsification (when the claim fails honestly)

- CI contains 1.0 at the achieved ratio ⇒ inconclusive.
- CI_high < 1.0 ⇒ regression; report and explain via r_native and
  the worker split.
- `selected_policy = gpu_only` at the achieved ratio ⇒ calibration
  rejected hybrid even with `achieved_ratio ≈ 1.0`; negative result
  for the predictor at that workload.
- `slowdown_distance_from_unity > 0.25` ⇒ the parity interpretation
  is weak even if hybrid wins; scope the claim away from "parity" or
  report at the achieved ratio.
- `r_native ≥ 1.0` and the native (`s = 1`) hybrid CI does not exceed
  1.0 ⇒ the predicted-favorable direction is not confirmed for that
  workload; report and propose a mechanism (memory bandwidth, queue
  overhead, etc.).
- Surface validation fails ⇒ row suppressed from the headline column.

## Out of scope

- `r` sweep beyond {`1.0`, `r_native`}.
- Multi-GPU; NUMA-aware CPU pinning.
- Real GPU contention (CUDA MPS, NVIDIA MIG).
- Heterogeneous CPU types treated as separate workers.
- Composition with axis A — RQ3 exercises A's RQ1 winner on top of
  B's cached cell at deployment time, not RQ2's concern. RQ2 keeps
  the GPU path vanilla so B's per-axis ground truth is uncontaminated
  by A.

## Implementation spec (`experiments/hybrid_applicability.py`)

Per (workload × platform × r × repeat):

- `regime_predictor`: `cpu_throughput`, `gpu_throughput`, `r_native`
  (diagnostic block; not a comparison condition)
- `vanilla_baseline.T_grid`, `vanilla_baseline.derived_from`
  (`measured` vs `unslowed_multiplied_by_s` — when the slowed baseline
  is derived rather than re-executed, the threat-to-validity flag is
  explicit)
- `hybrid.T_grid`, `hybrid.selected_policy`,
  `hybrid.worker_throughput_split`
- `achieved_ratio`, `slowdown_distance_from_unity`
- `hybrid.surface_validation`
- `trial_order` for this repeat

Per (workload × platform × r), aggregated across R repeats:

- `speedup_mean`, `speedup_ci_low`, `speedup_ci_high`
  [kalibera2020effect]
- `claim_status` ∈ {`hybrid_wins`, `inconclusive`,
  `hybrid_loses_at_parity`, `predictor_invalid`, `invalid_surface`}
  derived in one pass from CI bounds, surface gate, and
  `selected_policy`.
- `per_r_best_b_cell`: the hybrid configuration (CPU worker count,
  CPU batch size, GPU batch size, selected_policy) RQ2 measured at
  this r for this workload × platform. Descriptive context only;
  RQ3 calibrates its own cell on top of A's RQ1 winner and does
  not consume this field at execution time.

Student-t critical value: df = R − 1 = 4 (R = 5, unified across
platforms).

CPU worker count is a fixed system property of the host, not an
experimental treatment; it lives outside the `control` block to avoid
conflation with the calibration sweep's `cpu_worker_candidates`.
