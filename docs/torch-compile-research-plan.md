# Torch Compile Research Plan

Date: 2026-05-19

## Framing

This effort evaluates `torch.compile` as a setup-heavy optimization for
repeated loss-grid comparison sessions. It is not framed as a general
claim that compiled execution is faster. The question is whether the
cost of compilation can be amortized when several same-family checkpoint
variants reuse the same architecture, input shape, loss path, grid
structure, and execution backend.

Research question:

> Does `torch.compile` reduce end-to-end comparison-session compute time
> relative to vanilla eager loss-grid evaluation when compilation and
> calibration overhead are included?

The candidate is useful only if total comparison-session time improves:

```text
T_compile + T_calibration + N * T_compiled_grid
<
N * T_vanilla_eager_grid
```

Break-even variant count:

```text
N* = ceil((T_compile + T_calibration)
          / (T_vanilla_eager_grid - T_compiled_grid))
```

If `T_compiled_grid >= T_vanilla_eager_grid`, there is no amortization
point.

## Compilation Target Motivation

The compilation target must be derived from the original loss-grid
algorithm and the measured section profile, not selected as an
implementation convenience.

Canonical loss-grid structure:

```text
for grid point (alpha, beta):
  1. construct perturbed weights
  2. bind weights to model
  3. for batch in dataloader:
       y = model(x)
       loss = criterion(y, target)
  4. aggregate scalar loss
  5. write point result
```

The thesis section profiles show that forward/loss batch evaluation is
the dominant section, accounting for roughly 95--99% of total grid time
across the measured workloads and platforms. That makes the repeated
model-forward/loss path the scientifically justified compilation target.

`torch.compile` is therefore applied to the stable tensor-dominant
evaluation path:

- model forward,
- loss computation,
- optionally a fixed-shape batch evaluation function,
- optionally a functional or vmapped evaluator if it remains graph-stable.

The following regions should remain eager in the main experiment:

- grid-coordinate loop,
- dataloader iteration,
- CPU/GPU scheduler,
- calibration sweep,
- checkpoint/session orchestration,
- JSON or result writing.

Rationale:

- These regions are control-heavy rather than tensor-heavy.
- They contribute little to the measured runtime compared with
  forward/loss evaluation.
- They include Python iteration, queues, timing, device branching, file
  I/O, or policy search.
- They are more likely to cause graph breaks or recompilation.
- Compiling them would make negative results hard to interpret, because
  failure could come from orchestration rather than the repeated tensor
  path.

This keeps the experiment falsifiable. If the compiled evaluation path
does not amortize, the conclusion is clear: compiler setup and
steady-state gains are insufficient for the dominant repeated tensor
section. If the whole system were compiled and failed, the failure mode
would be ambiguous.

Paper-ready phrasing:

> Compilation targets are selected according to the original loss-grid
> algorithm's section structure and measured runtime shares. Since
> forward/loss evaluation dominates total grid time, `torch.compile` is
> applied only to the stable tensor evaluation path. Orchestration,
> scheduling, calibration, and result writing remain eager because they
> are control-heavy, low-arithmetic-intensity regions whose compilation
> would obscure the measurement and likely introduce graph breaks or
> recompilation.

## Scope

Initial scope:

- Same-family checkpoint variants.
- Fixed architecture.
- Fixed input shape.
- Fixed loss function.
- Fixed batch size.
- Fixed device/backend.
- Fixed grid resolution and perturbation scale.
- Inference only.
- Existing four-workload thesis set:
  - `cifar10_resnet20`
  - `cifar10_row_gru`
  - `california_mlp`
  - `mnist_mlp`

Start with CUDA/T4 if available because `torch.compile` is primarily
optimized around the PyTorch 2 compiler stack and CUDA/Triton path. Treat
MPS as optional or exploratory unless local behavior is stable.

## Candidate Paths

Evaluate only candidates that preserve the same output contract as the
vanilla eager baseline.

### `vanilla_eager`

Current baseline. Full session cost:

```text
T_session = N * T_vanilla_eager_grid
```

### `compiled_model_forward`

Compile the model forward/loss path used after parameter mutation.

This is the lowest-risk compilation target because it keeps the original
loss-grid algorithm mostly unchanged while accelerating the hottest
tensor section.

Compiled section:

```text
model_forward(inputs) -> predictions
loss(predictions, targets) -> batch_loss
```

Eager sections:

```text
grid loop
perturbation construction
parameter mutation/binding
dataloader iteration
loss aggregation
result writing
```

### `compiled_functional_eval`

Compile the stateless functional evaluator if the function remains
trace-stable.

This candidate tests whether compilation composes with the functional
evaluation path already studied in Experiment A.

Compiled section:

```text
functional_call(model, params_i, buffers, inputs)
loss(predictions, targets)
```

Eager sections:

```text
grid loop
perturbed parameter dictionary construction
dataloader iteration
loss aggregation
result writing
```

### `compiled_vmapped_eval`

Compile the vmapped chunk evaluator.

This candidate repeats the semantics of the existing `vmap` redesign
experiment while adding compilation around the repeated chunked
functional evaluation path.

Compiled section:

```text
vmap(functional_loss_for_params)(batched_parameters, inputs, targets)
```

Eager sections:

```text
grid loop
chunk construction
batched parameter materialization
dataloader iteration
loss aggregation
result writing
```

Because implementation effort is low relative to the existing vmapped
path, this should be treated as a normal candidate rather than optional.

## Code Module Sketch

Mirror the existing functional-evaluation modules so the compile
experiment remains comparable to Experiment A.

Suggested files:

```text
src/functional_eval/compiled.py
experiments/compile_eval_experiments.py
experiments/test_compile_eval_experiments.py
```

Suggested public functions:

```text
run_compiled_forward_surface(...)
run_compiled_functional_surface(...)
run_compiled_vmapped_surface(...)
evaluate_compiled_forward_points(...)
evaluate_compiled_functional_points(...)
evaluate_compiled_vmapped_points(...)
```

Suggested result shape:

```text
CompiledEvalResult
  candidate
  records
  timings
  compile_s
  first_grid_s
  steady_grid_s
  graph_break_count
  recompile_count
  peak_cpu_memory_bytes
  peak_cuda_memory_bytes
  metadata
  error
```

Candidate naming should follow the existing redesign candidates:

```text
compiled_forward
compiled_functional
compiled_vmapped_chunk_32
compiled_vmapped_chunk_64
```

The module should preserve the same surface output contract as the
baseline and existing functional/vmapped evaluators.

## Metrics

For each workload, platform, candidate, and variant count:

- `T_compile`: first-call compile overhead.
- `T_first_grid`: first grid including compilation.
- `T_steady_grid`: subsequent grid runtime after compilation.
- `T_calibration`: calibration overhead if policy selection is used.
- `T_session_N`: full comparison-session time for `N` variants.
- `N_break_even`: minimum variant count needed to beat vanilla eager.
- `recompile_count`: number of recompilations, if observable.
- `graph_break_count`: number of graph breaks, if observable.
- Surface validation: mismatch count and RMSE against vanilla eager.

Variant counts:

- `N = 4`

## Protocol

1. Sweep the existing four workloads:
   - `cifar10_resnet20`
   - `cifar10_row_gru`
   - `california_mlp`
   - `mnist_mlp`
2. Run vanilla eager full-grid evaluation for each checkpoint variant.
3. Run the existing Experiment A candidates where needed for comparison:
   - `functional_sequential`
   - `vmapped_chunk_32`
   - `vmapped_chunk_64`
4. Compile each compilation candidate using the first variant.
5. Run full-grid evaluation for all variants under the compiled path.
6. Separate first-call compile overhead from steady-state grid runtime.
7. Check for graph breaks and recompilations.
8. Validate every compiled surface against vanilla eager.
9. Compute `T_session_N` and `N_break_even`.
10. Compare runtime against:
    - vanilla eager,
    - corresponding eager redesign candidate,
    - best existing non-compiled candidate.
11. Report whether the compiled candidate beats vanilla eager for
    realistic comparison-session sizes.

The analysis should repeat the same redesign-candidate semantics as
Experiment A:

- each candidate targets a named algorithm section,
- section timings explain why the candidate helps or fails,
- speedup is reported relative to vanilla eager,
- compiled variants also report speedup relative to their corresponding
  eager candidate,
- numerical surface validation is required before runtime claims are
  accepted.

## MVP Implementation

The quick-confirmation MVP is intentionally self-contained:

```text
experiments/compile_mvp/run_compile_mvp.py
```

It reuses the existing workload, vanilla backend, validation, functional
sequential, and vmapped modules. It adds the compiled candidate
implementations in:

```text
src/functional_eval/compiled.py
```

MVP constraints:

- 8x8 grid by default.
- Existing four workloads by default.
- Stdout-only JSON-line records.
- No experiment result files.
- Python bytecode writing disabled inside the runner.
- Graph-break and recompile counters logged when PyTorch exposes them.
- Optional `--calibration-s` input included in compiled session-time
  accounting.
- Compilation candidates compared against vanilla eager and the matching
  eager redesign candidate where applicable.

The runner itself does not write logs or result artifacts. PyTorch's
compiler stack may still use its own internal cache directories; that is
outside the MVP's experiment-recording path and should be noted when
reporting the protocol.

Example command:

```text
venv/bin/python experiments/compile_mvp/run_compile_mvp.py \
  --device cuda \
  --variants 4
```

For a quick local smoke test:

```text
venv/bin/python experiments/compile_mvp/run_compile_mvp.py \
  --workload mnist_mlp_classification \
  --candidate compiled_forward \
  --device cpu \
  --sample-count 16 \
  --variants 1
```

## Decision Rules

The candidate supports the thesis only if:

```text
T_session_compiled_N < T_session_vanilla_eager_N
```

for `N <= 4`.

For candidate-level interpretation:

- `compiled_forward` is useful only if it beats vanilla eager after
  compile and calibration overhead are included.
- `compiled_functional` is useful only if it beats both vanilla eager and
  eager `functional_sequential` at the session level.
- `compiled_vmapped_chunk_K` is useful only if it beats vanilla eager and
  eager `vmapped_chunk_K` at the session level.
- If a compiled candidate improves steady-state runtime but loses after
  amortized overhead, report it as a steady-state compiler effect but not
  a practical comparison-session win.

The result is inconclusive if:

- steady-state execution is faster but break-even requires an unrealistic
  number of variants,
- results hold for only one workload,
- graph breaks make the actually compiled region unclear,
- recompilation occurs but the diagnostic reason is unresolved.

The candidate should be rejected as practically useful if:

- compiled steady-state runtime is not faster,
- compilation overhead dominates all realistic sessions,
- recompilation occurs per checkpoint variant,
- numerical validation fails,
- the candidate only beats eager after excluding compilation or
  calibration overhead.

## Thesis Value

This effort enriches the thesis if it remains an amortization experiment.
It tests whether the existing calibration/cache-reuse idea generalizes to
compiler setup cost: pay once, reuse across same-family variants, and
measure whether total comparison-session time improves.

The strongest possible positive result is not "compiled loss-grid
generation is faster." It is:

> For specific same-family comparison sessions, compiling the repeated
> tensor evaluation path amortizes after a measured number of variants and
> reduces total session time while preserving surface equivalence.

The strongest useful negative result is:

> Even when targeting the dominant repeated tensor section, compilation
> overhead, graph breaks, recompilation, or insufficient steady-state
> speedup prevent `torch.compile` from beating vanilla eager for realistic
> comparison-session sizes.
