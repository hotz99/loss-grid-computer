# Experiment A: Intra-Device Per-Grid-Point Optimization (RQ1)

## RQ1 connection

> **RQ1.** Do PyTorch's intra-device program-transformation APIs
> reduce perturbation-based loss-grid runtime, and for which
> architecture types? Do the two mechanisms — `functional_call+vmap`
> and `torch.compile` — multiply, are they redundant, or do they
> interfere destructively on this workload pattern?

Test whether each documented intra-device transform reduces T_grid
versus the in-place mutation baseline, and test the intra-axis
composition of the two.

## Method (knobs only; protocol shared from canon)

See `canonical-overview.md` MECHANISMS/MEASUREMENT for the shared
protocol (clock primitive, runtime-target vocabulary, section-
decomposition definition, verdict-rule template, trial-order /
pairing convention, surface-budget definition, no-CUDA-event canon).

This experiment's knobs:

- **Verdict statistic.** Per-candidate T_grid speedup
  `speedup_i = T_baseline_i / T_candidate_i`; intra-axis composition
  verdict on the per-repeat ratio
  `composition_i = speedup(compiled_vmapped)_i /
(speedup(vmapped)_i × speedup(compiled)_i)`. Both apply the canon
  CI rule [kalibera2020effect].
- **Repetitions.** R = 5 paired repeats per (workload × platform).
  Smallest R yielding df = R − 1 = 4 for a usable paired t-CI on
  speedup ratios; canon repetition discipline applies above this
  floor.
- **Pairing.** Baseline and each candidate evaluated within the same
  repeat against an identical workload, data subset, grid, and seed.
- **Trial-order rotation.** Cycle through permutations of
  {baseline, vmapped, compiled, compiled_vmapped} across repeats.
- **Surface gate.** Per canon MEASUREMENT/correctness gate:
  point count + coordinates match exactly; paired NaNs count as
  agreement; finite losses satisfy
  `|a − b| ≤ 1e-5 · max(|a|, |b|)` (rtol = 1e-5, atol = 0). Zero
  mismatches required across the grid.

## Experiment-specific surface

- **Workload set.** cifar10_resnet20, cifar10_row_gru, california_mlp,
  mnist_mlp.
- **Platforms.** Apple M4 (mps), NVIDIA Tesla T4 (cuda). Per-platform
  CIs; backends are not pooled.
- **Grid.** 8×8 (64 grid points), scale 1.0; 1024 evaluation
  samples per point, batch size 64. 64 grid points spans the
  K = 32 and K = 64 chunk paths (2 chunks and 1 chunk
  respectively). 1024 samples / 64 batch = 16 batches per point
  is enough for a stable per-point loss without letting eval
  dominate runtime over the orchestration sections the candidates
  attack.

## Candidates

| Role                 | Configuration                                                                |
| -------------------- | ---------------------------------------------------------------------------- |
| Baseline (reference) | in-place mutation + sequential variant loop                                  |
| `vmapped`            | `functional_call` + `vmap` over K grid points per chunk [pytorch_ensembling] |
| `compiled`           | `torch.compile` on the baseline eval path [pytorch_compile]                  |
| `compiled_vmapped`   | `torch.compile` applied over `vmapped` (intra-axis composition)              |

`functional_call` alone is not a candidate: prerequisite primitive
with no independent optimization claim; `vmap` subsumes its
observable contribution.

Chunk size K ∈ {32, 64} is a sub-parameter swept on the vmap-bearing
candidates, not a separate candidate. The per-candidate headline
reports the K with the highest CI_low; the full curve is retained
in the supporting record.

## Diagnostics (no verdict role)

- **Baseline section share** — T_perturb / T_bind / T_eval /
  T_collect / residual shares; localizes which section the
  candidates can attack and explains workload-dependent payoff.
- **Per-K speedup curve** (vmap-bearing candidates) — identifies
  the chunk-size optimum and exposes failure modes (OOM at large K,
  dispatch-dominated at small K).
- **Peak CPU and CUDA memory** (per candidate) — chunk size and
  graph capture both trade memory for dispatch amortization.
- **Compile cold-start cost** (compiled candidates) —
  `compile_cold_start_s` measured once per repeat as wall-time delta
  before/after the triggering call; reported separately from T_grid
  because the baseline pays no equivalent cost. Repeats within a
  measurement reuse the compiled graph.
- **Recompile counter** (compiled candidates) — `torch._dynamo`
  recompile events during the measured region; a non-zero count
  invalidates the steady-state T_grid assumption for that row.

## Falsification (when the claim fails honestly)

- Per-candidate CI contains 1.0 ⇒ inconclusive for that candidate.
  Do not increase R post hoc to narrow the CI.
- Per-candidate CI_high < 1.0 ⇒ regression. Report and explain via
  section share (where the candidate touches vs where the workload
  spends time).
- Composition CI on the ratio
  `speedup(compiled_vmapped) / (speedup(vmapped) × speedup(compiled))`,
  classified into the canon CHARACTER 4-category taxonomy
  {`multiplicative`, `redundant`, `destructive`, `inconclusive`}:
  - CI not bounded above by 1.0 (CI contains 1.0, or CI_low > 1.0)
    with both individuals also applicable ⇒ `multiplicative`
    (composition delivers at least the product of the individuals;
    CI_low > 1.0 flagged in the supporting record as a
    super-multiplicative outcome for follow-up discussion within the
    multiplicative category)
  - CI_high < 1.0, composed candidate still beats baseline ⇒
    `redundant` (composed underdelivers vs product)
  - CI_high < 1.0, composed candidate itself regresses below
    baseline ⇒ `destructive`
  - Otherwise (per-candidate inconclusives propagate or composition
    CI uninformative) ⇒ `inconclusive`
- Surface validation fails ⇒ row suppressed from the headline column;
  failure mode reported separately. Do not relax tolerances.
- `recompile_count > 0` within the measured region for a compiled
  candidate ⇒ steady-state assumption violated; row flagged and
  excluded from the composition verdict.
- Workloads with small T_eval / T_grid share are expected to be
  inconclusive or regress; do not exclude them.

## Out of scope

- Full-test-set sweep unless `INCLUDE_FULL_TEST_SET` is set.
- Broader chunk-size sweep unless `INCLUDE_VMAP_REPRODUCTION` is set.
- Composition with axis B — RQ3 exercises A's winner on top of B's
  cached cell at deployment time; not RQ1's concern.
- CPU-side application of these transforms — A's choices apply to
  GPU only in this thesis (see canon MECHANISMS/A and MECHANISMS/C
  device-asymmetry note).

## Implementation spec (`experiments/functional_eval_experiments.py`)

Per (workload × platform × repeat):

- `baseline.T_grid`, `vmapped.T_grid`, `compiled.T_grid`,
  `compiled_vmapped.T_grid` (paired within repeat)
- `baseline.section_timings`: T_perturb, T_bind, T_eval, T_collect,
  residual
- `{candidate}.peak_cpu_memory_bytes`,
  `{candidate}.peak_cuda_memory_bytes` (per candidate)
- `compiled.compile_cold_start_s`, `compiled.recompile_count`,
  `compiled_vmapped.compile_cold_start_s`,
  `compiled_vmapped.recompile_count`
- `{candidate}.surface_validation`: mismatch_count, RMSE,
  max_abs_error
- `trial_order` for this repeat (permutation index)

Per (workload × platform × K, where K applies), aggregated across R
repeats:

- `speedup_mean`, `speedup_ci_low`, `speedup_ci_high` from the paired
  trial ratios [kalibera2020effect]
- `claim_status` ∈ {`speedup`, `inconclusive`, `regression`,
  `invalid_surface`} per candidate
- `headline_K` (vmap-bearing): the K with the highest CI_low among
  rows where CI_low > 1.0 (else `null`)

Per (workload × platform), aggregated across candidates:

- `composition_ratio_mean`, `composition_ratio_ci_low`,
  `composition_ratio_ci_high` from per-repeat ratios
- `composition_status` ∈ {`multiplicative`, `redundant`,
  `destructive`, `inconclusive`} per the canon CHARACTER taxonomy
  (CI_low > 1.0 on the composition ratio is recorded as a
  super-multiplicative flag inside the `multiplicative` status)
- `winner_candidate`: the candidate with the highest CI_low among
  those with CI_low > 1.0; `baseline` if no candidate qualifies.
  **RQ3 consumes this field as A's winner for the composed stack.**

Student-t critical value: df = R − 1 = 4 (R = 5, unified across
platforms).
