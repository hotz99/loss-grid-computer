# INTRODUCTION

## Motivation: Loss-Grid Runtime as Bottleneck

The broader target software was LossLess, whose analysis pipeline exposes many comparison metrics across trained models. The original goal was therefore broad: reduce end-to-end time of that comparison pipeline. Early profiling work showed that loss-grid construction dominated runtime, so the research scope was narrowed to the loss-grid compute subsystem.

## Scope Narrowing from Broader Model-Comparison Runtime

The first optimization line was a vectorized redesign of the local-loss-landscape backend inherited from LossLens. Inspection of the codebase showed that the loss grid was being computed through an embarrassingly parallel iterator over grid points; we investigated whether GPU-friendly inference parallelization APIs could evaluate many points at once. That search led to `torch.vmap`, which fits the workload because it offers batched evaluation over many perturbation points without Python loops around each inference call.

The second line was heterogeneous scheduling: rather than confining execution to a single GPU, delegate a fraction of the embarrassingly parallel grid workload to available CPU workers. Because GPU and CPU inference throughputs can differ substantially across hardware and model classes, applicability is not uniform.

## Problem Statement

The problem is to reduce loss-grid compute time on a single machine without access to multi-GPU clusters or stronger hardware, while preserving the numerical fidelity of the resulting loss surface. Two sub-problems emerge from this framing: how to evaluate individual grid points more efficiently on a single device, and how to spread independent grid work across the available CPU and GPU resources on the same machine.

## Research Questions

`RQ1.` Can section-level redesign of the perturbation-based loss-grid algorithm reduce runtime while matching the resulting surface?

`RQ2.` Can a hybrid CPU+GPU execution system reduce perturbation-based loss-grid runtime under single-GPU constraints while matching the resulting surface?

`RQ3.` How does workload and hardware device affinity affect the usefulness of heterogeneous CPU+GPU scheduling?

`RQ4.` Can calibration and cache reuse make adaptive scheduling practical across repeated loss-grid sessions?

`RQ5.` _(deferred; excluded from current evaluation scope)_ Can progressive visualization expose useful loss-landscape structure before a full fixed-resolution grid is complete?

## Contributions

This work studies loss-grid runtime reduction across four connected but separable efforts:

- profiling the vanilla core algorithm to identify the sections that dominate runtime
- redesigning profiled algorithm sections through functional evaluation and vectorized point evaluation
- maximizing multi-device efficiency through heterogeneous CPU/GPU scheduling
- framing progressive visualization as an adaptive-mesh-refinement problem above the grid-computation layer

## Report Structure

Section 2 establishes the problem context: what a loss grid is, how the canonical single-device algorithm constructs one, and what single-machine hardware constraints bound the design space. Section 3 profiles the vanilla algorithm, defines the validation contract, and maps each measured section to an optimization candidate. Section 4 describes the four optimization methods in design terms: functional evaluation, vectorized point evaluation, heterogeneous scheduling, and progressive visualization. Section 5 specifies the experimental design for each method across the common MLTask set. Sections 6 and 7 are reserved for discussion and conclusions once results are available. Appendices record the system specification, platform metadata, and full experiment harness details.

---

# BACKGROUND AND PROBLEM CONTEXT

## Loss Landscapes and Perturbation-Based Grids

For a given model and task, the loss is the scalar quantity used to evaluate model quality on data. In loss-landscape analysis, that loss evaluation is repeated across an array of variants of the original trained model, formed by moving the parameter vector along chosen directions. The resulting array of loss values is the loss grid, which is used to render the loss landscape. Each grid point is a model-evaluation event on a fixed workload rather than a training step, consistent with standard machine-learning separation between fitting and evaluation.

## Canonical Vanilla Loss-Grid Algorithm

We take Li et al., _Visualizing the Loss Landscape of Neural Nets_ (NeurIPS 2018), as the canonical method for deriving loss landscapes:

- select one grid point
- perturb model parameters for that point using  
   `w(α, β) = w₀ + α·d₁ + β·d₂`
- run forward/loss evaluation over the dataset subset
- write one scalar loss value into the surface
- repeat until the grid is complete

Where `w₀` is the original trained model (a vector of parameters); `d₁` and `d₂` are the chosen perturbation directions; `α`, `β` set the displacement along those directions.

This baseline is intentionally:

- single process
- single GPU
- free of CPU helper execution

The only intentional omission relative to the public MPI-oriented implementation pattern is distributed execution. This leads to the practical claim: when a user does not have access to a more powerful compute environment—whether a stronger single GPU or a multi-GPU cluster—our system may reduce total time-to-results by delegating some work to available CPU resources.

The `vanilla` backend implements this algorithm; all runtime and surface comparisons are relative to this baseline.

## Single-Machine Compute Constraints

The experiments are framed around practical single-machine constraints: one available GPU, available CPU cores, finite RAM and VRAM, and backend-specific numerical behavior. These constraints matter because loss-grid rendering repeats the same inference workload many times across perturbed model states, so both compute throughput and memory pressure can determine which optimization is viable.

---

# SYSTEM AND BASELINE PROFILING

## Vanilla Core Algorithm

The core algorithm has a nested structure:

- inference on one input sample
- repeated inference across a dataset subset
- accumulation of those results into one loss value for the current model state
- repetition of that full loss evaluation across many perturbed model states

This structure makes the workload embarrassingly parallel at the outermost level: each grid point is independent, and each point's evaluation is computationally identical.

## Profiling Decomposition

Profiling decomposes the vanilla core algorithm into the sections that later optimization methods target:

- perturbation construction
- parameter binding or model mutation
- dataset forward/loss evaluation
- result collection

This decomposition is the bridge between the baseline algorithm and any proposed optimization: an optimization is meaningful only if it targets a measured section of the computation and preserves the resulting loss surface.

## Validation Contract

The measurement target for all runtime comparisons is grid computation alone:

- constructing and evaluating all grid points
- excluding startup and setup that are not part of point evaluation and result collection
- while still logging calibration and scheduler overhead separately for analysis

Performance comparisons are valid only if they preserve the resulting loss surface:

- sort both surfaces by grid coordinate
- check point-count and coordinate alignment
- treat paired `NaN` values as matching
- apply pointwise `math.isclose` with `(atol, rtol)`
- report mismatch count and RMSE

## Platform and Artifact Recording

The system records platform and artifact metadata so runtime claims can be interpreted under the hardware and data conditions that produced them. The full system-facing specification is maintained in [System Specification](system-spec.md).

Backend-specific numerical behavior is tracked explicitly. CPU, CUDA, and MPS runs are labeled by platform, and validation claims are scoped to the backend on which they are produced.

## Workload Definition

The experiments use a common `MLTaskSpec` workload contract: a task specifies the dataset, model family, loss, checkpoint, model builder, dataset builder, and batch-level loss computation. This keeps the perturbation-based grid algorithm independent of any one model implementation.

The MLTask set is used as a workload-diversity axis where each method is applicable. Workload affinity names the per-task dimension along which optimization value varies: it is the relationship between a workload's execution structure and the runtime change produced by a given optimization candidate. A single task would conflate API validity, model architecture, backend behavior, and hardware memory pressure; the task set lets the experiment distinguish general task-contract support from workload-specific speedup.

- ResNet20 on CIFAR-10 as the reference convolutional workload
- row-GRU on CIFAR-10 as a sequential model-execution contrast
- California Housing MLP regression as a task, modality, and loss-function contrast
- MNIST wide MLP classification as an expected favorable `vmap` workload

## Optimization Opportunities Identified by Profiling

The profiling output should be read as an optimization map rather than only as a runtime breakdown. Each section has a candidate redesign, a baseline fallback, and a decision rule:

| Vanilla subtask | Measured role | Candidate optimization | Decision rule |
| --- | --- | --- | --- |
| perturbation construction | creates one perturbed parameter vector per grid point | batch perturbation construction for chunked point evaluation | pursue only if perturbation time is non-negligible or if batching is required by a larger vectorized candidate |
| parameter binding or model mutation | copies the perturbed vector into the model before evaluation | `torch.func.functional_call` with explicit parameter dictionaries | pursue when mutation/binding cost is measurable, or when a stateless call surface is needed for `vmap` |
| dataset forward/loss evaluation | runs model inference and loss accumulation for each perturbed state | `torch.vmap` over perturbed parameter states, implemented through functional evaluation | pursue when repeated point evaluation dominates and memory permits chunked vectorization |
| result collection | stores the scalar loss for each grid coordinate | keep the baseline path unless profiling shows collection overhead is material | treat as a non-target when its measured cost is small relative to grid evaluation |

This mapping also defines negative outcomes. If a profiled section contributes little to total grid time, the correct optimization candidate for that section is the baseline implementation. For example, if dataset inference or result collection is measured as a small fraction of total runtime for a platform/workload pair, the experiment should report that the baseline section is already adequate rather than force a redesign that cannot materially reduce total grid time.

Functional evaluation and vectorized point evaluation are not claims about CIFAR-10 or ResNet20 specifically; they are claims about the perturbation-based loss-grid execution pattern. The task-set rationale is defined in §3.5.

---

# OPTIMIZATION METHODS

## Section-Targeted Algorithm Redesign

The vanilla algorithm exposes two different optimization opportunities after profiling: the sequential parameter-binding step and the parallel repetition across grid points. These should be evaluated separately even though the vectorized candidate depends on the functional-evaluation form.

The section-targeted candidates follow directly from the profiling map. Functional evaluation targets parameter binding and mutation semantics; `vmap` targets repeated point evaluation by stacking independent perturbation states; point chunking bounds the memory footprint introduced by that vectorization. Sections whose profiling share is small remain assigned to the baseline implementation.

### Functional Evaluation with `torch.func.functional_call`

The parameter-binding section of the vanilla algorithm mutates a model instance to one perturbed parameter vector at a time before running dataset evaluation. A functional-evaluation redesign instead treats the model call as a function of supplied parameters, buffers, and inputs. This targets the sequential mutation/binding section while preserving the original one-grid-point-at-a-time execution order.

This method is not itself a parallelization strategy. Its role is to remove in-place model mutation from the candidate path, make parameter layout explicit, and create a stateless call surface that later vectorized evaluation can use.

### Vectorized Point Evaluation with `torch.vmap`

The forward/loss section repeats the same dataset evaluation across many independent perturbed model states. `torch.vmap` targets this parallel grid-point dimension by evaluating a chunk of perturbed parameter states for the same data batch.

`vmap` must be coupled to functional evaluation because the mapped function needs to accept the parameter state as an explicit input. Without the stateless function created by `torch.func.functional_call`, the implementation would still rely on mutating a single model instance, which is incompatible with mapping many parameter states through the same call.

The two research questions remain decoupled:

- whether stateless functional evaluation improves or preserves sequential point execution
- whether vectorizing multiple parameter states improves parallel point execution once functional evaluation is available

### Decoupled Candidate Taxonomy

The local redesign candidates are evaluated as separable components before any system-level composition:

- `functional_sequential`: replace in-place mutation with `torch.func.functional_call` while preserving the sequential grid-point loop
- `vmapped_chunk_K`: stack `vmap` over functional evaluation to evaluate chunks of independent perturbation states
- baseline/no-op sections: retain original behavior where profiling shows insufficient runtime share to justify redesign

Vectorizing over grid points increases parameter, activation, and loss-memory pressure. Point chunking therefore acts as a memory-control mechanism for `vmapped_chunk_K`: the chunk size determines how many perturbed parameter states are evaluated together. The appropriate chunk size is hardware-dependent and must be swept under recorded platform constraints.

## Multi-Device Efficiency Maximizing

Inference is the dominant repeated operation, and it can in principle be executed on both GPU and CPU. The relative inference throughput of the single GPU and the aggregate CPU worker set is therefore the primary factor determining whether hybrid execution accelerates total runtime.

The central claim is conditional: hybridization is not expected to help universally, but should help for some regimes of relative GPU and CPU inference throughput, and that relative throughput can be characterized experimentally through controlled artificial GPU latency while keeping the algorithm, workload definition, and numerical validation fixed.

A fixed execution mode—always GPU-only, or always hybrid with a fixed CPU count—will be wrong for some hardware and workload combinations. This motivates an adaptive system: one that measures the applicable regime for the current machine and workload, then selects the execution mode accordingly. The scheduling mechanism is eager scheduling: both GPU and CPU workers pull independent grid points from a shared queue on demand, with unit chunk granularity. The primary systems contribution is the measurement and caching mechanism that selects the right execution mode, not a claim that hybrid execution universally reduces runtime.

## Merging Beneficial Components onto the Heterogeneous Scheduler

The local algorithm redesign and the heterogeneous scheduler should not be merged by default. A redesigned component should first pass numerical validation and demonstrate positive runtime value in isolation. Only then should it be composed with the heterogeneous scheduler.

The merged system is a later composition step:

- local redesign improves how one device evaluates grid work
- heterogeneous scheduling decides where independent grid work should execute
- merged candidates test whether those benefits remain positive when combined

## Progressive Visualization as AMR (TODO — deferred; excluded from §5 evaluation scope)

Progressive visualization treats the loss surface as something that may become useful before every point in a fixed-resolution grid has been evaluated. Under this framing, adaptive mesh refinement is a visualization-layer strategy above the grid-computation layer: it can consume faster point evaluation from either local algorithm redesign or heterogeneous scheduling, but it should be evaluated with visualization-specific criteria.

The method therefore separates raw time-to-render-grid from progressive usefulness. Runtime optimizations reduce the cost of computing grid points; AMR asks whether the system can reveal useful loss-landscape structure earlier by refining the surface progressively.

---

# EVALUATION

## Common Setup

**Hardware and platform.** Hardware platform is recorded at runtime; functional-evaluation and scheduler experiments are platform-agnostic and should be run on CUDA, MPS, and any additional available GPU/CPU pair where feasible.

**Result bundle schema.** Each experiment produces a result bundle with required fields for cross-machine comparability:

- platform identity: GPU model, CPU model, CUDA/MPS/CPU backend, driver version
- workload identity: MLTask name, dataset subset size, batch size, num_batches, grid resolution
- runtime measurements: total grid time, per-section timings where applicable
- surface validity: point count, coordinate alignment, mismatch count, RMSE vs vanilla baseline
- experiment config: candidate name, chunk size, slowdown factor, worker count (where applicable)

**MLTask set.** The common MLTask set used across experiments where the method is applicable:

| MLTask | Model family | Role in the experiment |
| --- | --- | --- |
| `cifar10_resnet20_classification` | ResNet20 convolutional classifier | reference workload and BatchNorm/buffer-handling stress case |
| `cifar10_row_gru_classification` | row-wise GRU classifier | same dataset/task/loss as ResNet20 but sequential execution structure; included to probe lower GPU-affinity regimes where CPU/GPU throughput ratios may approach parity |
| `california_mlp_regression` | tabular MLP regressor | task, modality, and loss-function contrast |
| `mnist_mlp_classification` | wide dense MLP classifier | expected favorable `vmap` workload because it uses stateless dense layers without BatchNorm or recurrent state |

**Grid resolution.** 8×8 for all full-grid evaluation runs; calibration grid uses the smallest square grid satisfying the worker-depth heuristic (see §5.5). For full-grid evaluation and all RQ comparisons, this resolution is smaller than typical end-use landscape computations but sufficient for the tested worker counts to participate in the shared-queue workload rather than terminating before mixed-device scheduling behavior emerges.

**Runtime parameters.** Subset size, batch size, `num_batches`, and numerical validation tolerance are fixed across workload runs.

Experiments run across the MLTask set where the method is applicable. Unsupported combinations are reported as skips or scoped limitations rather than omitted silently.

## Experiment A: Vanilla Profiling

### Design

This experiment measures the vanilla core algorithm by section across the MLTask set: perturbation construction, parameter binding or mutation, dataset forward/loss evaluation, and result collection. The purpose is to establish the baseline cost structure that motivates later optimization methods.

The output of this experiment is a per-workload section-to-candidate map. A later candidate is justified only when it targets a measured section whose runtime share could affect total grid time, or when it is a required enabler for another candidate. This prevents the evaluation from treating all implementation changes as equally relevant.

## Experiment B: Functional Evaluation and Vmap Redesign

### Design

This experiment evaluates section-level redesign candidates against the original algorithm over the MLTask set. The goal is not to prove that one transform wins everywhere; it is to map workload structure to the candidate that should be composed into the larger system.

Functional evaluation is evaluated independently where possible: the candidate preserves sequential grid-point execution while replacing model mutation with explicit parameter binding through `torch.func.functional_call`.

`vmap` is evaluated as a separate vectorization question, but implemented on top of functional evaluation (structural dependency defined in §4.1.2). The evaluation decouples the questions while coupling the implementation:

- does functional evaluation improve or preserve sequential execution?
- does vectorizing over multiple parameter states improve point-level parallel execution?

Point chunk size is treated as the memory-control parameter for the vectorized candidate.

The candidate taxonomy is therefore:

| Candidate | Components | Targeted sections |
| --- | --- | --- |
| baseline original | in-place mutation and sequential point loop | reference for all sections |
| functional sequential | `torch.func.functional_call` | parameter binding or mutation |
| vmapped functional chunks | `functional_call` + `vmap` + point chunking | perturbation construction, parameter binding, and repeated forward/loss evaluation |
| baseline/no-op section | unchanged original implementation | any section whose profiled cost is too small to justify redesign |

For each MLTask, the experiment records total grid runtime, section timings, peak RAM/VRAM, numerical equivalence to the baseline surface, candidate speedup, and the best valid `vmap` chunk size.

## Experiment C: Hybrid Scheduler Applicability

### Design

Experiment C tests the relative-throughput hypothesis across the MLTask set by varying the effective GPU/CPU throughput ratio via `gpu_slowdown_factor` and observing whether a crossover regime exists where hybrid execution surpasses the vanilla baseline.

**Slowdown control.** `gpu_slowdown_factor` is an experimental control used to vary the GPU/CPU throughput ratio by adding latency to the GPU workload routine. Injected latency is acceptable here as an application-level control for changing effective GPU service time while keeping the algorithm, workload definition, and surface-validation procedure fixed. It is not a hardware-faithful simulation of a weaker GPU. Architecture papers on scaled GPU studies emphasize validated workload/silicon simulation precisely because simplistic scaling can miss contention effects or relevant code regions [Avalos et al., 2021].

**CPU/GPU inference ratio as the experimental variable.** Experiment C reports the effective CPU/GPU inference ratio as the primary regime variable. The artificial slowdown is only the control used to move the current platform into a desired ratio regime.

Let:

```
r_native = throughput_cpu / throughput_gpu
```

where both throughputs are measured on the same workload, dataset subset, batch policy, and platform. If the GPU evaluation path is slowed by a factor `s`, then the first-order expected ratio is:

```
r_slowed = s * r_native
```

The initial slowdown used to target a desired ratio is therefore:

```
s_target = r_target / r_native
```

This calculation is only an initial estimate. Runtime overhead, scheduling contention, memory behavior, and backend effects can make achieved throughput differ from the linear estimate. The experiment therefore resolves each ratio regime empirically:

1. measure the native CPU and GPU throughput for the workload
2. compute the initial slowdown estimate for the target ratio
3. run the scheduler measurement under that slowdown
4. recompute the achieved CPU/GPU inference ratio
5. refine the slowdown with a bounded local search until the achieved ratio is within tolerance, or mark the target ratio unresolved

The experiment reports both `target_ratio` and `achieved_ratio`, along with the slowdown required to reach that regime.

For example, a workload may be summarized across regimes such as:

| Target CPU/GPU inference ratio | Interpretation | Expected scheduler behavior |
| --- | --- | --- |
| 0.25 | GPU-dominant | GPU-only usually wins |
| 0.50 | CPU begins to contribute materially | hybrid may become useful |
| 1.00 | CPU and GPU are near parity | CPU contribution should increase |

The exact thresholds are not assumed to be universal. They are reported as platform- and workload-specific measurements.

**Speedup metric.** The comparison quantity is:

```
speedup = t_vanilla_scaled / t_hybrid
t_vanilla_scaled = slowdown × t_vanilla
```

where `slowdown` is the artificial GPU slowdown factor, `t_vanilla` is the measured vanilla runtime under unslowed GPU conditions, and `t_hybrid` is the measured hybrid runtime under the same slowdown. `t_vanilla_scaled` is a synthetic baseline representing the vanilla runtime one would observe if the GPU were `slowdown`-times slower. This comparison isolates the scheduling benefit: it asks whether CPU assistance reduces grid time given an effective GPU throughput of 1/`slowdown` of baseline.

**Crossover protocol.** The "crossover region" is `[s_low, s_high]`, where:

- `s_low`: largest tested slowdown with speedup ≤ 1.0
- `s_high`: smallest tested slowdown with speedup > 1.0

This interval contains the applicability threshold for the current workload and hardware. To locate it efficiently, an adaptive bracketing heuristic with midpoint refinement is used:

1. evaluate hybrid at slowdown = 1.0
2. increase slowdown multiplicatively until a losing/winning pair is bracketed
3. refine that bracket with two midpoint tests

After resolving the crossover interval, slowdown values inside that interval are linearly sampled to obtain a local view of how hybrid speedup varies around the applicability threshold. The final linear-sampling phase uses three repeats per slowdown point, enabling speedup standard deviation to be reported alongside mean estimates. Bracketing-phase measurements remain single-repeat; resolved crossover boundaries should be reported as approximate.

For each MLTask, the experiment records:

- selected execution mode: GPU-only or GPU+CPU hybrid
- selected worker count and CPU batch size
- CPU/GPU point split
- isolated or recorded per-device throughput
- numerical fidelity of the selected execution mode's surface against that workload's vanilla surface

## Experiment D: Calibration and Cache Amortization

### Design

This experiment evaluates the calibration mechanism that follows from the throughput-ratio insight. If applicability is regime-dependent, a usable system should automate the applicability determination for the current machine and workload, and should make that automation cheap enough to justify in the intended workflow.

Experiment D does not depend on the artificial control from Experiment C. It evaluates the calibration mechanism under the platform's recorded runtime condition. The purpose is to specify the system behavior a user would actually invoke: measure candidate configurations, select the lowest-runtime valid configuration, and reuse that decision where cache semantics permit.

**Experiment D1: Calibration Selection.** Calibration evaluates execution mode selection for the current platform and workload. Rather than assuming a fixed execution mode, the system measures candidate configurations under the recorded runtime condition and resolves whether GPU-only or hybrid execution is preferable and, if hybrid, which tested configuration minimizes runtime.

Calibration searches the candidate space as follows:

1. measure the GPU-only baseline runtime under the platform's recorded runtime condition
2. evaluate each candidate from the Cartesian product of tested `p` and CPU batch size values under parallel GPU+CPU
3. record total runtime and throughput for each candidate
4. stop after `retry` consecutive non-improvements in best runtime as `p` increases
5. sort the tested records by lowest runtime

CPU batch sizes are chosen dynamically as a powers-of-two ladder, bounded above by the smaller of the workload batch size and subset size. This keeps the calibration sweep compact while restricting candidates to feasible batch granularities.

- if no hybrid candidate has lower runtime than the GPU-only baseline, calibration selects `gpu_only`
- otherwise calibration selects the lowest-runtime tested hybrid configuration

CPU assistance is never measured as a separate CPU-only cohort. Candidate usefulness is judged only under parallel GPU+CPU execution, because memory-bus contention, scheduler overhead, queue behavior, and other shared-resource effects differ from isolated device measurements [Hugo et al., 2014].

**Experiment D2: Calibration Cache Amortization.** The practical use case is not one isolated landscape computation, but a comparison session over multiple related model variants from the same model family. In that setting, calibration is necessary to choose a good hybrid configuration, but calibration is also the main extra cost introduced by the method. The cache mechanism determines whether the hybrid system improves total session runtime rather than only isolated-run runtime.

Experiment D2 changes the unit of analysis from one calibrated run to a short comparison session. Calibration does not use the full evaluation grid. Instead, it uses the smallest square grid whose queue is comfortably larger than the tested worker set: if `p_max` is the largest tested CPU worker count, we require at least `4 × (1 + p_max)` grid points and choose the smallest resolution `r` such that `r²` meets that bound. This ties calibration size to queue depth rather than an arbitrary wall-time threshold.

The cache-amortization claim is validated with a minimal sequential session over four model variants from the same model family:

- **with cache**: the first variant pays the one-time baseline and calibration cost, then executes the selected grid computation; later variants reuse the cached execution mode selection and execute grid computation only
- **without cache**: each variant is charged the same baseline and calibration overhead again before grid computation; per-variant calibration decisions are recorded separately, capturing variance in which execution mode is selected when calibration runs in isolation for each variant

The comparison is session-level: the question is not whether the first cached run is cheaper in isolation, but whether reuse lowers total time-to-results across the full comparison workflow.

**Cache reuse scope.** Valid reuse case: same model family, same task and dataset, same hardware. Invalid reuse case: same task and dataset, different model families (e.g., ResNet versus AugViT). The reason is that calibration is driven by execution behavior, not task label alone. Two model families solving the same ML task can have very different compute structure, memory-access patterns, and GPU/CPU affinity, leading to different winning hybrid configurations.

Additional invalidation risks:

- scheduler or backend code changes may silently invalidate old calibration; this must currently be handled manually
- hardware changes are only safe if the resolved hardware identity changes with them; backend class alone is not enough, so the cache key includes concrete GPU/CPU identity fields
- very small workloads may rank policies differently because startup and synchronization overhead dominate inference time
- if `runtime.num_batches` or dataset selection become more operationally important later, they must be promoted in the key and old cache entries invalidated

## Experiment E: Merged Optimization Stack on the Heterogeneous Scheduler

### Design (TODO — WIP; gated on Experiment B producing confirmed beneficial components)

This experiment is reserved for components that first prove useful in isolation. The goal is to test whether a section-level runtime improvement remains beneficial after the heterogeneous scheduler controls work placement.

The experiment should distinguish:

- local improvements in how a device evaluates one or more grid points
- global improvements in how grid points are assigned to available devices
- interaction effects where a local optimization changes the scheduler's selected execution mode

---

# DISCUSSION (TODO)

Discussion should interpret which optimization avenues worked, under which hardware constraints, and whether isolated gains composed. It should also discuss backend numerical behavior, memory limits, and threats to validity.

---

# CONCLUSIONS (TODO)

Conclusions should answer the research questions and summarize the practical recommendations for reducing perturbation-based loss-grid runtime under constrained hardware.

---

# REFERENCES

- Li, H., Xu, Z., Taylor, G., Studer, C., Goldstein, T. _Visualizing the Loss Landscape of Neural Nets_. NeurIPS 2018. https://papers.nips.cc/paper/7875-visualizing-the-loss-landscape-of-neural-nets
- Xie, T., Chen, J., Yang, Y., Geniesse, C., Shi, G., Chaudhari, A. J., Cava, J. K., Mahoney, M. W., Perciano, T., Weber, G. H., Maciejewski, R. _LossLens: Diagnostics for Machine Learning Through Loss Landscape Visual Analytics_. IEEE Computer Graphics and Applications, 2025. https://doi.org/10.1109/MCG.2024.3509374
- Yang, Y., Hodgkinson, L., Theisen, R., Zou, J., Gonzalez, J. E., Ramchandran, K., Mahoney, M. W. _Taxonomizing local versus global structure in neural network loss landscapes_. CoRR abs/2107.11228, 2021. https://dblp.org/rec/journals/corr/abs-2107-11228
- Augonnet, C., Thibault, S., Namyst, R., Wacrenier, P.-A. _StarPU: A Unified Platform for Task Scheduling on Heterogeneous Multicore Architectures_. Concurrency and Computation: Practice and Experience, 23(2), 187–198, 2011. https://doi.org/10.1002/cpe.1631
- Hugo, A., Guermouche, A., Wacrenier, P.-A., Namyst, R. _Composing multiple StarPU applications over heterogeneous machines: A supervised approach_. International Journal of High Performance Computing Applications, 28(3), 2014. https://doi.org/10.1177/1094342014527575
- Gallet, B., Gowanlock, M. _Heterogeneous CPU-GPU Epsilon Grid Joins: Static and Dynamic Work Partitioning Strategies_. Data Science and Engineering, 2021. https://doi.org/10.1007/s41019-020-00145-x
- Avalos, C., Khairy, M., Green, R., Payer, M., Rogers, T. _Principal Kernel Analysis: A Tractable Methodology to Simulate Scaled GPU Workloads_. MICRO 2021. https://engineering.purdue.edu/tgrogers/publication/avalos-micro-2021/
- Gabbay, F., Lev Aharoni, R., Schweitzer, O. _Deep Neural Network Memory Performance and Throughput Modeling and Simulation Framework_. Mathematics, 2022. https://doi.org/10.3390/math10214144
- NVIDIA. _CUDA C++ Programming Guide_, section 1.1, "The Benefits of Using GPUs". https://docs.nvidia.com/cuda/archive/11.1.1/cuda-c-programming-guide/index.html
- Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., Bengio, Y. _Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation_. EMNLP 2014. https://aclanthology.org/D14-1179/
- Chung, J., Gulcehre, C., Cho, K., Bengio, Y. _Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling_. arXiv:1412.3555, 2014. https://arxiv.org/abs/1412.3555
