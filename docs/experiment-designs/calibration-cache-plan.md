# Experiment C: Calibration Cache for B, on top of A's RQ1 Winner (RQ3)

## RQ3 connection

> **RQ3.** Does reusing the calibrated B-cell across a related
> checkpoint session amortize the upfront calibration cost? Does
> the autotune-and-cache pattern's amortization survive the
> signature lift from ATLAS's op-level signature (machine, BLAS op)
> to this thesis's workload-level signature (machine, architecture
> + dataset + loss + grid spec)?

Test whether one-time **B-cell** calibration on one same-family
checkpoint, configured on top of **A's RQ1 winner** for the
workload and reused across the remaining N − 1 checkpoints, reduces
T_session versus running per-checkpoint vanilla.

## Method (knobs only; protocol shared from canon)

See `canonical-overview.md` MECHANISMS/MEASUREMENT for the shared
protocol (clock primitive, runtime-target vocabulary, verdict-rule
template, trial-order / pairing convention, surface-budget
definition, no-CUDA-event canon).

This experiment's knobs:

- **Verdict statistic.** Descriptive cumulative T_session evidence
  at N = 4 plus break-even N diagnostic — sessions are measured
  once each, not repeated, so no effect-size CI claim. A follow-up
  that repeats full sessions could promote the verdict to the canon
  CI rule [kalibera2020effect].
- **Target.** T_session is the headline; per-checkpoint T_grid is
  recorded as a diagnostic.
- **Repetitions.** Each session measured once; T_session is
  descriptive evidence, not a CI claim. A follow-up that repeats
  full sessions could promote the verdict to the canon CI rule.
- **Pairing.** Both sessions run on the same machine in a fixed,
  disclosed order; cache state reset between sessions.
- **Surface gate.** Per canon MEASUREMENT/correctness gate:
  point count + coordinates match exactly; paired NaNs count as
  agreement; finite losses satisfy
  `|a − b| ≤ 1e-5 · max(|a|, |b|)` (rtol = 1e-5, atol = 0). Zero
  mismatches required per checkpoint.

**Composition with A.** A is consumed from RQ1's winner per
(workload × platform), not re-swept here (canon STRUCTURE,
MECHANISMS/C). The composed-with-cache session uses (A's RQ1 winner)
as the GPU-side implementation with (cached B-cell) as the
scheduler; the vanilla reference uses vanilla on both axes.

## Native operating regime (no artificial slowdown)

Experiment C runs natively and inherits no slowdown from Experiment B.
The claim is about whether the bounded calibration amortizes on the
real machine across a same-family checkpoint session.

If C's bounded calibration picks `gpu_only` (the hetero arm did not
beat the GPU-only baseline within the patience budget), the composed
session reduces to (A-winner + GPU-only) with `calibration_s`
overhead, and `break_even_n = ∞` is reported as the honest negative
result. No design-time RQ2 dependency: C runs natively per workload
in the set and the calibration outcome speaks for itself.

## Session size N = 4

LossLens [xie2025losslens] reports a participant study in which
model-comparison sessions typically involve a small number of related
trained variants compared side-by-side; their 100-model case study
finds: *"we were able to draw similar conclusions compared to cases
with only 4 models."* The headline is read at N = 4. The experiment
does not measure or claim at N > 4.

Per-variant cost unit is the dense 2D loss surface
[li2018losslandscape].

## Configuration consumed from RQ1

- **A's winner** (from RQ1's `winner_candidate` field): the GPU-side
  per-grid-point implementation with the highest CI_low for the
  selected workload × platform. Held fixed across both sessions in
  this experiment. If RQ1 reports `winner_candidate = baseline` (no
  A optimization applies), the composed-with-cache session
  degenerates to (vanilla GPU + cached B-cell), which is the
  original B-only calibration; the plan handles this without
  modification.

No RQ2 dependency at execution time. RQ3 calibrates its own B-cell
under the bounded-patience discipline [ansel2014opentuner]; RQ2's
per-r findings are reported alongside RQ3 in the projection layer
as context, not consumed as a search reference.

## Experiment-specific surface

- **Workload.** One (cifar10_row_gru in the documented case).
- **Platform.** One (Apple M4 in the documented case).
- **Checkpoints.** N = 4 same-family checkpoints. Same-family
  predicate: identical architecture + dataset + loss + grid spec;
  only trained weights differ (e.g., different seeds).
- **Session grid.** 40×40, scale 1.0 (1600 grid points). Denser
  than RQ1/RQ2's 8×8 systems-benchmark control so the headline
  reads at a realistic session-time scale for the LossLens-scoped
  workflow [xie2025losslens]; the specific 40 is one denser-than-
  benchmark grid that stays tractable for session repetition.
- **B-cell calibration probe grid.** Square grid of side m,
  smallest m where m² ≥ 4 × (1 + p_max) and p_max is the largest
  CPU worker count probed. 4 tasks per (1 GPU + p_max CPU workers)
  is the minimum queue depth for the dynamic self-scheduler to
  expose work-placement signal during calibration. (Side variable
  renamed m to avoid collision with Gallet's CPU/GPU throughput
  ratio r.)
- **Sessions measured.** 2 (no repeats; descriptive session
  evidence).

## Sessions compared

| Session | Role | How obtained |
| --- | --- | --- |
| `vanilla_reference` | RQ3 denominator | N vanilla 40×40 grid runs, one per checkpoint, native |
| `composed_with_cache` | RQ3 numerator | 1 B-cell calibration on checkpoint 0 (with A's RQ1 winner configured GPU-side) + N grid runs under (A's RQ1 winner + cached B-cell) for each of the N checkpoints, native |

The system has no per-checkpoint recalibration mode, so there is no
third "no-cache" session to measure; the cache's value is read off
the comparison between the two real sessions.

**What the cache stores.** A B-cell calibration result — the
scheduler policy (`gpu_only` or a specific `gpu_cpu_hybrid`
configuration with CPU worker count and CPU batch size) selected
by the sweep for the current (machine, workload), given A's RQ1
winner. A is not in the cache value because A is established
upstream by RQ1 on the same key. The autotune-and-cache pattern
is inherited from ATLAS [whaley1998atlas], lifted from op-level
(BLAS routine signature) to workload-level signature.

## Why B's calibration is conditioned on A (not independent of it)

Calibrating B on top of A's RQ1 winner is non-trivially different
from calibrating B on top of vanilla. Once A shrinks GPU time, B's
break-even for adding CPU workers shifts (canon MECHANISMS/C
composition-with-A note). The B-cell C calibrates is therefore
conditioned on A; this is why C's cache is keyed on (machine,
workload) — the same key A is keyed on — rather than on B
parameters alone.

## Calibration cost is the design lever

`break_even_n = ⌈ calibration_s / (T_v − T_p) ⌉` when `T_v > T_p`,
else absent — where `T_v` is per-checkpoint T_grid under the vanilla
reference and `T_p` is per-checkpoint T_grid under the composed-
with-cache stack. Once the workload × platform fix `T_v` and `T_p`,
the only knob is `calibration_s`. Three levers, in increasing
impact:

1. **Probe grid resolution.** Already small (worker-depth probe
   grid, not the 40×40 session grid).
2. **Per-measurement retry count.** Trades noise floor for
   `calibration_s` linearly.
3. **Patience-based termination.** The sweep stops after P = 3
   consecutive non-improvements — bounded autotuning-search
   discipline [ansel2014opentuner], the departure from ATLAS-style
   exhaustive search. P = 3 absorbs one noise hit before declaring
   convergence on the small B-cell surface; P is the explicit
   cost/exhaustiveness knob. (Named P, not R, to avoid collision
   with the repetition count R used by RQ1/RQ2.)

## Headline metric (descriptive)

```
session_speedup_vs_vanilla =
    vanilla_reference_session_total_s
    / composed_with_cache_session_total_s
    (measured at N = 4)

break_even_n = ⌈ calibration_s / (T_v − T_p) ⌉   when T_v > T_p
             = absent                            otherwise
    (diagnostic; crossover of the cumulative session curves)
```

| Outcome at N = 4 | Condition | Interpretation |
| --- | --- | --- |
| Speedup at LossLens scale | `session_speedup_vs_vanilla > 1` | Cache amortizes B's calibration within the LossLens workflow scope, in the composed (A-winner + cached-B) stack |
| No speedup at LossLens scale | `session_speedup_vs_vanilla ≤ 1` | Negative for the documented workflow; `break_even_n` characterizes how far the boundary sits from the target |

No claim is made at N > 4.

Under bounded-autotuning framing [ansel2014opentuner] the question
is not whether C's sweep finds the global best hetero-config — it
is whether the bounded calibration produces enough benefit to
amortize over the session. Near-optimality vs. a denser sweep
(an oracle-gap claim) is therefore out of scope for RQ3.

## Validation figure

Cumulative T_session vs n ∈ [1..N], two lines (`vanilla_reference`,
`composed_with_cache`) with σ bands from their N measured
per-variant times. The intersection is `break_even_n`. Caption
states workload, platform, native CPU/GPU ratio,
`source_exp_a_record`, `a_winner`,
`session_speedup_vs_vanilla` at N = 4, and `break_even_n`.
Adjacent table lists `T_v`, `T_p`, `calibration_s`, and
`break_even_n`.

## Falsification (when the claim fails honestly)

- `T_v ≤ T_p` (composed stack not faster per-checkpoint than
  vanilla) ⇒ `break_even_n` is absent; report as no amortization
  for this workload × platform. Do not retry calibration with
  looser stopping to force a hybrid selection.
- `session_speedup_vs_vanilla ≤ 1` at N = 4 ⇒ negative for the
  LossLens-scoped workflow; report `break_even_n` as context.
- Surface validation fails on any checkpoint ⇒ session suppressed
  from the headline; failure mode reported separately.

## Out of scope

- Re-sweeping A — A's winner is consumed from RQ1 (canon
  MECHANISMS/C); C's sweep is over B-cell only. Profiling of A
  candidates is RQ1's domain and is not repeated here.
- Effect-size CI claims on T_session — sessions are not repeated
  here; a follow-up could promote the verdict to the shared CI rule
  [kalibera2020effect].
- Speedup claims at N > 4.
- Artificial slowdown — Exp C runs natively; the slowdown is an
  Exp B instrument and is not inherited here.
- Hardware-variation validation across platforms.
- Cache-invalidation edge cases (model family switch, batch shape
  change).

## Implementation spec (`experiments/calibration_cache.py`)

Input: N same-family checkpoints on the selected workload × platform;
the 40×40 session grid; the worker-count-sized probe grid for
B-cell calibration; the RQ1 record (for `winner_candidate` =
A-winner). No RQ2 input is consumed at execution time.

Execution (all native, no artificial slowdown):

1. **Load A's RQ1 winner.** Read `winner_candidate` and its K (if
   applicable) from the RQ1 record for this workload × platform.
   If `winner_candidate = baseline`, configure the GPU path as
   vanilla.
2. **Calibrate B-cell** once on checkpoint 0 using the probe grid,
   with A's RQ1 winner configured on the GPU side →
   `calibration_s`, `selected_b_cell` (policy + worker count +
   batch size). Sweep uses patience-based termination (stop after
   P = 3 consecutive non-improvements).
3. **Composed-with-cache session.** Run (A's RQ1 winner) +
   (`selected_b_cell`) on the 40×40 session grid for each of the
   N checkpoints → per-variant T_grid_p, `T_p = mean`,
   `T_p_sigma_rel`. Then
   `composed_with_cache_session_total_s = calibration_s +
   sum(per-variant T_grid_p)` per canon's T_session definition
   (multi-checkpoint session including any one-time calibration the
   measured arm performs; the framing is experiment-level, since
   the artifacts are a means to conduct experiments rather than a
   productized runtime — canon WHERE THIS SITS / RQ3 stance). The
   vanilla reference excludes calibration because that arm performs
   none.
4. **Vanilla reference session.** Run `VanillaMode(gpu_batch_size)`
   on the 40×40 session grid for each of the N checkpoints →
   per-variant T_grid_v, `T_v = mean`, `T_v_sigma_rel`,
   `vanilla_reference_session_total_s`.

Derived (no extra executions):

- `session_speedup_vs_vanilla` (headline; descriptive)
- `break_even_n`
- `break_even_meets_lossLens_target = (break_even_n is not None
   and break_even_n ≤ 4)` (diagnostic only; the success condition
   is `session_speedup_vs_vanilla > 1`)

Record fields:

- `status`: `completed` or `skipped` (skip on missing checkpoint
  assets, unsupported workload, or missing RQ1 input)
- `session_regime`: workload, platform, `cpu_pool_size`,
  `unslowed_cpu_gpu_ratio`, `source_exp_a_record`, `a_winner`
- `n_checkpoints`, `session_grid_resolution`,
  `calibration_grid_resolution`
- `selected_b_cell`
- `calibration_s`
- `T_v`, `T_v_sigma_rel`, `vanilla_per_variant_times_s`
- `T_p`, `T_p_sigma_rel`, `composed_per_variant_times_s`
- `vanilla_reference_session_total_s`,
  `composed_with_cache_session_total_s`
- `session_speedup_vs_vanilla` (headline; descriptive)
- `break_even_n`, `break_even_meets_lossLens_target` (diagnostic)
- `surface_validation` per checkpoint (gate)
