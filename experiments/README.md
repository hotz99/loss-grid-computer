Runner Dataflow
===============

Config Variables
----------------

Variables are grouped by scope. Changing a variable only affects the
experiments that consume it.

### Infra (no effect on results)

| Variable | Default | Purpose |
|---|---|---|
| `DEVICE` | `"auto"` | Target device (`"cpu"`, `"cuda"`, `"mps"`, or `"auto"`) |
| `OUTPUT_DIR` | `None` | Output root; auto-timestamped under `outputs/` if `None` |
| `RUN_LABEL` | `None` | Optional prefix on output filenames |
| `FAIL_FAST` | `False` | Abort suite on first experiment error |
| `VERBOSE_EXPERIMENT_LOGS` | `False` | Suppress stdout/stderr redirect during runs |

### Shared compute defaults (`GRID_RESOLUTION` applies to A/B)

| Variable | Default | Purpose |
|---|---|---|
| `SEED` | `1337` | RNG seed passed to all backends |
| `SAMPLE_COUNT` | `1024` | Dataset samples per workload |
| `GRID_RESOLUTION` | `8` | Bounded benchmark/probe grid resolution (n×n points) for A/B |
| `GRID_SCALE` | `1.0` | Loss landscape perturbation scale |
| `GPU_BATCH_SIZE` | `64` | Batch size for GPU execution paths |

### Validation (A candidates, B)

| Variable | Default | Purpose |
|---|---|---|
| `ATOL` | `1e-6` | Absolute tolerance for surface equivalence check |
| `RTOL` | `1e-5` | Relative tolerance for surface equivalence check |

### Workload list (A profiling, A candidates, B)

| Variable | Default | Purpose |
|---|---|---|
| `MLTASK_WORKLOADS` | all four workloads | Which workloads to run; each resolved to its default checkpoint via `WORKLOADS` registry |

### Exp A candidates only

| Variable | Default | Purpose |
|---|---|---|
| `FUNCTIONAL_EVAL_REPEATS` | `5` | Paired-repeat count for CI estimation |
| `FUNCTIONAL_EVAL_SAMPLE_COUNTS` | `[1024]` | Sample-count sweep |
| `FUNCTIONAL_EVAL_BATCH_SIZE` | `GPU_BATCH_SIZE` | Override batch size for functional eval |
| `POINT_CHUNK_SIZES` | `[32, 64]` | vmap chunk sizes to benchmark |
| `MAX_MEMORY_FRACTION` | `0.85` | OOM guard for vmap candidates |
| `INCLUDE_VMAP_REPRODUCTION` | `False` | Include vmap reproduction check runs |
| `INCLUDE_FULL_TEST_SET` | `False` | Run full test-set variants alongside sampled |
| `FUNCTIONAL_EVAL_WORKLOADS` | `None` | Override workload list for functional eval only; falls back to `MLTASK_WORKLOADS` if `None` |

### Exp B and C

| Variable | Default | Purpose |
|---|---|---|
| `CALIBRATION_RETRY` | `3` | Calibration retry count on noisy measurements |
| `MAX_CPU_WORKER_CANDIDATE` | `None` | Cap on CPU worker sweep; `None` = no cap |

### Exp C only

| Variable | Default | Purpose |
|---|---|---|
| `MEASURE_WITHOUT_CACHE` | `True` | Empirically measures the recalibrate-each-checkpoint diagnostic required by the headline claim |
| `EXPERIMENT_C_SESSION_GRID_RESOLUTION` | `40` | Full session grid resolution (n×n points); calibration still uses its worker-depth probe grid |

Research-facing resolution policy: Experiments A/B use `GRID_RESOLUTION=8`
for bounded implementation and scheduler probes; Experiment C uses
`EXPERIMENT_C_SESSION_GRID_RESOLUTION=40` for the full amortized session while
calibration keeps its smaller worker-depth probe grid. The motivation and
citations live in `docs/algorithm-redesign-plan.md`,
`docs/hybrid-scheduler-plan.md`, and `docs/calibration-cache-plan.md`.

### Checkpoint dimensionality

Experiments differ in how many checkpoints they consume per workload:

| Experiment | Workload scope | Checkpoints per workload | How resolved |
|---|---|---|---|
| A profiling | `MLTASK_WORKLOADS` | 1 | `WORKLOADS[name].spec.checkpoint_path` (registry default) |
| A candidates | `MLTASK_WORKLOADS` (or `FUNCTIONAL_EVAL_WORKLOADS`) | 1 | same |
| B | `MLTASK_WORKLOADS` | 1 | same |
| C | all four workloads | 4 variant paths per workload | defined in `calibration_cache.VARIANT_PATHS`; each workload runs as one independent session — not a product across variants |

Exp C is not a cartesian product over checkpoints. It runs a single
calibration on variant 0, then reuses the cached policy across all four
variants. The variant list is a session input, not an iteration axis.

---

The runner is an execution harness. It may reuse parameterized compute
artifacts across research-facing experiments, but that reuse does not change
the experiment descriptions in `docs/thesis/experiments-outline.md`.

Notebook / local Python
-----------------------

1. Configure globals:
   - `DEVICE`, `OUTPUT_DIR`, `RUN_LABEL`
   - `SAMPLE_COUNT`, `GRID_RESOLUTION`, `GRID_SCALE`, `GPU_BATCH_SIZE`
   - `EXPERIMENT_C_SESSION_GRID_RESOLUTION`
   - `FUNCTIONAL_EVAL_REPEATS`, `FUNCTIONAL_EVAL_SAMPLE_COUNTS`
   - `EXPERIMENT_REGISTRY[*].enabled`
2. Call `experiments.runner.run_aio_suite()`.
3. The runner creates one output directory, executes enabled registry entries
   in order, writes and verifies one child JSON payload per entry, and refreshes
   `summary.json` after each entry. Long composed entries may also write a
   `*-partial.json` payload while still running.

Record Contract
---------------

Persistence has one owner: `experiments.runner`.

Experiment modules compose primitives and return plain dictionaries:

- `status`
- `result`: full experiment data
- `record`: minimal summary record
- `child_stem`: optional filename stem for the runner child JSON

The runner adds `duration_s` and `output_path` to each record. Raw in-process
artifact objects live under private `shared_state` keys and are omitted from
the persisted summary.

Minimal Compute Graph
---------------------

The runner should compute each unique parameterized node once per suite run.
The shared key includes workload, checkpoint, sample count, grid resolution and
scale, requested device, seed, GPU batch size, and slowdown factor.

1. `vanilla_full_grid(workload, checkpoint, full_grid, slowdown=1.0)`
   - computed by Experiment A profiling
   - reused by Experiment B for the GPU side of the unslowed CPU/GPU ratio
   - not reused for Experiment B's slowdown-adjusted parity baseline
   - not reused for Experiment C because calibration uses a smaller
     worker-depth grid

2. `section_profile(workload, checkpoint, full_grid, slowdown=1.0)`
   - computed by Experiment A profiling
   - consumed only by Experiment A

3. `functional_sequential(workload, checkpoint, full_grid, chunk_config)`
   - computed by Experiment A candidates
   - consumed only by Experiment A

4. `vmapped_functional(workload, checkpoint, full_grid, point_chunk_size)`
   - computed by Experiment A candidates
   - consumed only by Experiment A

5. `cpu_probe(workload, checkpoint, full_grid, cpu_workers, cpu_batch_size)`
   - computed by Experiment B
   - combined with the shared vanilla full-grid node to report the unslowed
     CPU/GPU throughput ratio

6. `parity_slowdown = f(unslowed CPU/GPU ratio)`
   - derived by Experiment B without model execution

7. `vanilla_full_grid(workload, checkpoint, full_grid, slowdown=parity_slowdown)`
   - computed by Experiment B
   - used only as the parity comparison baseline

8. `hybrid_parity(workload, checkpoint, full_grid, slowdown=parity_slowdown)`
   - computed by Experiment B
   - used only for the parity probe

9. `vanilla_calibration_grid(variant_1, calibration_grid, slowdown=1.0)`
   - computed by Experiment C
   - not reused from Experiments A or B because the grid differs

10. `calibration_sweep(variant_1, calibration_grid, cpu_worker_candidates,
    cpu_batch_candidates, slowdown=1.0)`
    - computed by Experiment C
    - selects the policy reused in the cache condition

11. `variant_full_grid_execution(variant_i, full_grid, selected_policy)`
    - computed by Experiment C for the same-family session

Experiment Chain
----------------

`e0_platform_inventory`
: Records host, Python, torch, CPU, GPU, memory, and workload asset metadata.

`functional_eval_api_probe`
: Preflights `torch.func.functional_call`, `torch.vmap`, chunking behavior,
  and the basic workload functional-call path.

`experiment_a_profiling`
: Runs the vanilla full-grid baseline per configured workload, records section
  timings, and publishes the shared `vanilla_full_grid` artifact.

`experiment_a_candidates`
: Runs `baseline_original`, `functional_sequential`, and vmapped chunk
  candidates and reports all results. Candidate-repeat baselines remain local
  to this experiment because they support the paired-repeat comparison.
  Comparison and candidate selection are left to analysis.

`experiment_b_hybrid_applicability`
: Reuses the shared `vanilla_full_grid` artifact when the key matches,
  measures CPU throughput, derives the unslowed CPU/GPU ratio, then performs
  the slowdown-adjusted parity probe.

`experiment_c_calibration_cache`
: Uses an unslowed runtime condition. It measures calibration on the
  worker-depth grid, selects a policy, and compares same-family sessions with
  and without cache reuse. No Experiment A/B full-grid artifact is reused here.

`experiment_d_merged_stack`
: Disabled placeholder; enabled after analysis of Experiment A results confirms a beneficial candidate.

`progressive_visualization_deferred`
: Disabled placeholder outside the implemented A-D experiment chain.
