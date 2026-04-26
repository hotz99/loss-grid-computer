# System Specification

## 1. Calibration Procedure

The calibration mechanism is motivated by the runtime-reduction goal. Its purpose is to decide whether adding CPU workers in parallel to the GPU reduces total grid runtime on the current machine and workload, and if so, which tested configuration performs best:

1. Measure the GPU-only baseline
   - evaluate a representative calibration workload with the GPU alone
   - record baseline runtime `R_gpu`

2. Enumerate hybrid candidates
   - define the Cartesian product of:
     - CPU worker count `p`
     - CPU batch size
   - keep GPU batch size fixed for this experiment

3. Evaluate each candidate under the real mixed-device regime
   - run one GPU worker and `p` CPU workers in parallel
   - use the same shared queue and pickup policy intended for the final hybrid system
   - record total runtime and throughput for each candidate

4. Qualify hybrid execution
   - hybrid qualifies if there exists some tested candidate with runtime lower than `R_gpu`
   - CPU workers therefore do not need to match GPU throughput in isolation; they only need to reduce total wall-clock runtime when added in parallel to the GPU

5. Select configuration
   - sort candidate records by lowest runtime
   - choose `gpu_only` if no hybrid record improves on `R_gpu`
   - otherwise choose the first hybrid record in the sorted list
   - this is a best-tested selection rule, not a claim of global optimality over all possible parameter settings

6. Apply a stopping rule
   - after evaluating all batch-size candidates for a given `p`, compare the best runtime at that `p` against the current best runtime
   - stop after `retry` consecutive non-improvements once added CPU workers no longer reduce runtime

7. Interpretation rule
   - CPU assistance is never measured as a separate CPU-only cohort
   - candidate usefulness is judged only under parallel GPU+CPU execution, because memory-bus contention, scheduler overhead, queue behavior, and other shared-resource effects differ from isolated device measurements

### Calibration Reuse and Cache Design

Calibration introduces extra work, so its utility depends on amortization across repeated runs. This matters especially because a full worker-count sweep is more expensive than a single static choice.
We amortize the calibration cost via caching over model and hardware configuration; sucht that calibration results remain valid across runs, even when trained weights differ.
If the same model family and hardware are used across many repeated landscape computations, one calibration sweep can be reused across those runs.

This aligns with workflows such as LossLens, where many related trained models under a shared model configuration are analyzed offline.
It performs repeated runs over related variants of the same model architecture. Hence, our caching mechanism design is validated by this usecase.

TODO: add exact source references from the LossLens paper supporting this use-case framing.

Useful cache classes:

- execution-policy cache:
  stores the outcome of calibration, such as GPU-only versus hybrid and the chosen CPU worker count
- calibration-measurement cache:
  stores measured throughput probes for matched workload and hardware conditions

Cache key for execution-policy reuse:

- primary key:
  - hardware and backend identity
  - model architecture and parameter-shape signature
- workload key:
  - input shape
  - dataset or subset regime
  - batch size and `num_batches`
- experiment key:
  - artificial GPU slowdown setting
  - runtime knobs that substantially affect scheduling behavior

TODO: revisit whether workload size, including grid resolution, should become part of the cache key once the small-workload fallback rule is formally defined.

## 2. Mode Selection

calibration dictates whether the system should use the hybrid path or fall back to the vanilla GPU-only path
if the measured inference regime does not justify CPU assistance, the system runs the vanilla path
if the measured inference regime qualifies for hybridization, the system enters the hybrid path, with inputs:

- model and dataset subset
- grid definition
- CPU worker count `p*` (resolved from calibration)
- queue chunk size: currently one grid point per task, matching the unit of work used by the vanilla baseline
  NOTE: This choice may increase scheduler overhead because workers return to the shared queue after every grid point. That overhead should be measured separately and can motivate a later (chunk-size, cpu_batch_size, gpu_batch_size) tuning experiment under the same pickup-as-you-finish policy.

### Heterogeneous Scheduling Policy

- Coordinator Process:
  - owns the global task queue
  - collects completed results
  - assembles the final surface
  - validates against the vanilla baseline

- GPU worker:
  - starts first and claims work from the shared queue
  - provides the main high-throughput execution path when GPU throughput is dominant
  - is a logical worker owned by the coordinator process, not a separate spawned CPU process

- `N` CPU workers:
  - claim remaining work from the same queue as they become available
  - provide auxiliary throughput when the effective GPU advantage is not too large

TODO: add small motivation for using multiprocessing over multithreading for this system
