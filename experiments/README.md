Runner Dataflow
===============

The runner is an execution harness. It may reuse parameterized compute
artifacts across research-facing experiments, but that reuse does not change
the experiment descriptions in `docs/thesis/experiments-outline.md`.

Notebook / local Python
-----------------------

1. Configure globals:
   - `DEVICE`, `OUTPUT_DIR`, `RUN_LABEL`
   - `SAMPLE_COUNT`, `GRID_RESOLUTION`, `GRID_SCALE`, `GPU_BATCH_SIZE`
   - `FUNCTIONAL_EVAL_REPEATS`, `FUNCTIONAL_EVAL_SAMPLE_COUNTS`
   - `EXPERIMENT_REGISTRY[*].enabled`
2. Call `experiments.runner.run_aio_suite()`.
3. The runner creates one output directory, executes enabled registry entries
   in order, writes one child JSON payload per entry, then writes
   `summary.json`.

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
