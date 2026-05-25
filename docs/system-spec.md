## 1. System Overview and End-to-End Workflow

The system is built around a single-GPU baseline and a hybrid execution path that adds CPU helpers only when the inference throughput ratio suggests they are useful.

<!-- TODO this reads rough -->

The latter preserves the same loss-grid semantics as the canonical baseline. GPU and CPU workers consume work from the same shared queue. This design is consistent with prior heterogeneous shared-queue systems: dynamic dequeueing lets faster devices consume more work while keeping slower devices productive, rather than forcing a static partition that risks imbalance [Augonnet et al., 2010].
Take a workload definition, calibrate on the current machine and workload regime, execute either vanilla or hybrid path to produce one loss surface.

### High-Level Architecture

```text
run_orchestration(RunRequest, device, seed, gpu_slowdown_factor)
  |
  +--> request.mode is None ?
       |
       +-- no --> resolved_mode = request.mode
       |
       +-- yes --> build calibration request:
                   HybridMode(calibration_gpu_batch_size)
                   + dynamic candidates:
                     - cpu_worker_candidates
                     - cpu_batch_size_candidates
                   |
                   v
                 build_calibration_cache_key_payload(...)
                   |
                   v
                 load_calibration_cache(path)
                   |
                   +-- hit  --> resolved_mode = cached resolved_mode
                   |
                   +-- miss --> run_backend(vanilla baseline)
                               -> baseline_total_s
                               -> run_calibration(...)
                               -> write_calibration_cache(
                                    baseline_total_s,
                                    created_at,
                                    resolved_mode
                                  )
  |
  v
run_backend(SchedulerRequest(task, grid, resolved_mode, device))
  |
  v
print_json(ExperimentResult)
```

The key decision boundary is calibration:

- if no tested hybrid candidate improves on GPU-only runtime, the system selects the vanilla path
- otherwise the system selects the best tested hybrid configuration

## 2. Task-Agnostic Workload Interface

In principle, abstracting over ML tasks should be feasible because the scheduler does not depend on classification-specific semantics. At the scheduler boundary, the unit of work is a `task`. In the current system, each task is one iteration of the perturbation-based loss-grid procedure: sequential task logic prepares the perturbed model state and batch-level evaluation; followed by inference on either GPU or CPU [Li et al., 2018]. If some ML task can be expressed under this procedure, it is eligible for heterogeneous scheduling.

At the software level, this is achieved by abstracting over a task specification contract:

- machine learning task category and model family
- a dataset builder
- a model builder
- a batch-level `compute_loss(model, batch, device) -> (loss, batch_size)` function

## 3. Calibration

### Motivation and Procedure

The calibration mechanism is motivated by the runtime-reduction goal. It decides whether adding CPU workers in parallel to the GPU reduces total runtime with the current hardware and workload; and if so, which tested configuration has lowest runtime.

1. Measure and record GPU-only baseline `R_gpu`.
2. Enumerate hybrid candidates as the Cartesian product of:
   - CPU worker count `p`
   - CPU batch size
3. Evaluate each candidate:
   - run one GPU worker with added `p` CPU workers in parallel
   - use the same scheduler structure and policy as for the final hybrid system
   - record total runtime and throughput for each candidate
4. Apply a stopping rule:
   - evaluate candidate configurations in loop order
   - after each candidate run, update the best runtime seen so far if the new runtime is lower
   - when a candidate improves the best runtime, reset the non-improvement counter to `0`
   - when a candidate does not improve the best runtime, increment the non-improvement counter by `1`
   - stop the sweep once the non-improvement counter reaches `retry`
5. Select configuration:
   - sort candidate records by lowest runtime
   - choose `gpu_only` if no hybrid record improves on `R_gpu`
   - otherwise choose the first hybrid record in the sorted list

CPU assistance is never measured as a separate CPU-only cohort. Candidate usefulness is judged only under parallel GPU+CPU execution, because memory-bus contention, scheduler overhead, queue behavior, and other shared-resource effects differ from isolated device measurements [Hugo et al., 2014].

## 4. Cache

### Motivation and Reuse

Calibration is expensive. We amortize the time cost by caching over model family and hardware configuration, such that calibration results remain valid across runs, even when trained weights differ.

Under typical loss-grid use cases, users compute landscapes for multiple related model variants in order to compare them, described by both LossLens and related loss-landscape work that treats the landscape as a relation among multiple trained solutions rather than a one-model-only diagnostic [Xie et al., 2025; Yang et al., 2021]. For that workload the cache mechanism turns the one-time calibration into an amortized cost across the comparison session.

### Cache-Key Definition (WIP)

The execution-policy cache key should answer one question:

`could this change the selected execution mode or winning hybrid configuration?`

If no, it should not invalidate reuse.

```text
CalibrationCacheKey {
  // local compute regime under which the policy was measured
  resolved_hardware: {
    backend_class: "cuda" | "mps" | "cpu"
    gpu_name: string
    cpu_name: string
    cpu_worker_capacity: int
  }

  // workload-family and calibration inputs that can change the winning policy
  input_payload: {
    scheduler_policy_id: string
    preload_enabled: bool

    model_family: string
    model_param_shape_signature: tuple[int, ...] | string

    task_family: string
    dataset_family: string
    input_shape: tuple[int, ...]

    data_batch_size: int
    gpu_batch_size: int | None
    gpu_slowdown_factor: float

    subset_size: int
    grid_resolution: int
    num_batches: int | None

    cpu_worker_values: tuple[int, ...]
    cpu_batch_sizes: tuple[int, ...]
    retry: int
  }
}
```

Fields intentionally excluded from the key:

- exact checkpoint path / trained weights
- `experiment_name`
- `runtime.output_root`
- `runtime.verbose`
- `seed`
- `grid.scale`

These exclusions follow the LossLens-oriented reuse case: the cache must remain valid across repeated comparisons of related trained model variants, not just exact repeats of one experiment snapshot [Xie et al., 2025].

## 5. Mode Selection

Calibration dictates whether the system should use the hybrid path or fall back to the vanilla GPU-only path.

If the measured inference regime does not justify CPU assistance, the system runs the vanilla path.

If the measured inference regime qualifies for hybridization, the system enters the hybrid path, with inputs:

- model and dataset subset
- grid definition
- CPU worker count `p*` resolved from calibration
- queue chunk size: currently one grid point per task, matching the unit of work used by the vanilla baseline

NOTE: This choice may increase scheduler overhead because workers return to the shared queue after every grid point. That overhead should be measured separately and can motivate a later `(chunk_size, cpu_batch_size, gpu_batch_size)` tuning experiment under the same pickup-as-you-finish policy.

## 6. Scheduler Architecture and Policy

The hybrid scheduler uses a simple master-worker design.

```text
Hybrid Runner
  -> resolve GPU device
  -> build grid points
  -> chunk grid points
  -> create shared task queue
  -> create result queue
  -> spawn workers
     -> CPU worker processes 0..N-1
     -> GPU worker process
  -> collect worker payloads
  -> summarize payloads
  -> return loss surface + runtime metrics
```

```text
Worker Process
  -> prepare model, data loader, and perturbation tensors on its device
  -> pull one chunk from the shared task queue
  -> evaluate chunk points on its device
  -> synchronize and apply GPU slowdown if needed
  -> append local records
  -> repeat until sentinel is received
  -> return worker payload
```

The scheduling policy is dynamic shared-queue dequeueing rather than a static CPU/GPU partition. Faster workers naturally consume more tasks, while slower workers continue to contribute useful work when the throughput regime allows it [Augonnet et al., 2010].

Heterogeneity here is not a separate scheduler layer. It means the same logical work function is performed by workers bound to different devices, with one GPU worker and `N` CPU workers consuming from the same queue.

The implementation uses multiple processes rather than multiple threads under one process. CPU helpers need real parallel execution across physical cores, and CPython threads are a poor fit for CPU-bound worker execution because only one thread can execute Python bytecode at a time; the `multiprocessing` module is explicitly designed to side-step that limitation by using subprocesses instead of threads [Python threading docs; Python multiprocessing docs]. Scheduler behavior is therefore shaped not only by worker count and queue dynamics, but also by the device affinity of the core inference task itself. The research narrative evaluates this explicitly through a workload-affinity comparison between convolution-heavy ResNet20 inference and row-GRU-style sequential inference.

- Coordinator process:
  - builds the global task queue and result queue
  - collects completed results
  - assembles the final surface
  - summarizes runtime metrics from worker payloads

- GPU worker:
  - spawned as its own process
  - first to claim work from the shared queue

- `N` CPU workers:
  - claim remaining work from the same queue as they become available

## References

- Augonnet, C., Thibault, S., Namyst, R., & Wacrenier, P.-A. (2010). StarPU: a unified platform for task scheduling on heterogeneous multicore architectures. _Concurrency and Computation: Practice and Experience, 23_(2), 187-198. https://doi.org/10.1002/cpe.1631
- Hugo, A., Guermouche, A., Wacrenier, P.-A., & Namyst, R. (2014). Composing multiple StarPU applications over heterogeneous machines: A supervised approach. _The International Journal of High Performance Computing Applications, 28_(3). https://doi.org/10.1177/1094342014527575
- Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). Visualizing the Loss Landscape of Neural Nets. _Advances in Neural Information Processing Systems 31_. https://mlanthology.org/neurips/2018/li2018neurips-visualizing/
- Python Software Foundation. `multiprocessing` — Process-based parallelism. https://docs.python.org/3/library/multiprocessing.html
- Python Software Foundation. `threading` — Thread-based parallelism. https://docs.python.org/3/library/threading.html
- Xie, T., Chen, J., Yang, Y., Geniesse, C., Shi, G., Chaudhari, A. J., Cava, J. K., Mahoney, M. W., Perciano, T., Weber, G. H., & Maciejewski, R. (2025). LossLens: Diagnostics for Machine Learning Through Loss Landscape Visual Analytics. _IEEE Computer Graphics and Applications, 45_(2), 112-125. https://doi.org/10.1109/MCG.2024.3509374
- Yang, Y., Hodgkinson, L., Theisen, R., & Zou, J. (2021). Taxonomizing local versus global structure in neural network loss landscapes. https://arxiv.org/abs/2107.11228
