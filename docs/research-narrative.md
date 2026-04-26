# INTRODUCTION

## Problem Context

For a given model and task, the loss is the scalar quantity used to evaluate model quality on data. In loss-landscape analysis, that loss evaluation is repeated across an array of variants of the original trained model, formed by moving the parameter vector along chosen directions. The resulting array of loss values is the loss grid, which is used to render the loss landscape.

The optimization target is the runtime required to compute this loss grid while preserving the resulting surface.

The project is motivated by a constrained local setting: one available GPU and additional CPU capacity. The central question is not whether a hybrid CPU+GPU system can beat unconstrained multi-GPU execution, but whether it can reduce loss-grid runtime under realistic single-GPU conditions while preserving the computed loss surface.

## Canonical Baseline

We take Li et al., _Visualizing the Loss Landscape of Neural Nets_ (arXiv:1712.09913), as the canonical method for deriving loss landscapes. The canonical local baseline is the vanilla backend. It is the local, non-MPI reconstruction of the standard loss-landscape evaluation pattern:

- select one grid point
- perturb model parameters for that point using  
   w(alpha, beta) = w0 + alpha \* d1 + beta \* d2
- run forward/loss evaluation over the dataset subset
- write one scalar loss value into the surface
- repeat until the grid is complete

Here, w0 is the original trained model (a vector of parameters), while d1 and d2 are the chosen perturbation directions. For each perturbed model variant, the system evaluates the loss on the common dataset and writes the resulting loss-value scalar into the grid.

This baseline is intentionally:

- single process
- single GPU
- eager
- free of CPU helper execution
- used as the reference surface and the reference runtime for hybrid comparisons

The only intentional omission relative to the public MPI-oriented implementation pattern is distributed execution.  
The vanilla backend implements this algorithm, and runtime comparisons are relative to this baseline.

## Domain Insight

The core algorithm has a nested structure:

- inference on one input sample
- repeated inference across a dataset subset
- accumulation of those results into one loss value for the current model state
- repetition of that full loss evaluation across many perturbed model states

This structure motivates our system's design: inference is the dominant repeated operation, and it can in principle be executed on both GPU and CPU. Then, the relative inference capability of the single GPU and the aggregate CPU worker set is the main factor determining if hybrid execution accelerates runtime.

The claim to test is conditional:

- hybridization is not expected to help universally
- hybridization should help for some regimes of relative GPU and CPU inference capability
- that relative capability can be changed experimentally through controlled artificial GPU latency while keeping the algorithm, workload definition, and numerical validation fixed

## Research Question

Can a hybrid CPU+GPU execution system reduce loss-grid runtime under single-GPU constraints while preserving the computed loss surface?

# METHODS

## Heterogeneous System Overview

The system is built around a single-GPU baseline and a hybrid execution path that adds CPU helpers only when the effective throughput regime suggests they are useful.

Hybrid execution preserves the same loss-grid semantics as the canonical baseline, but changes how grid points are assigned to compute resources. GPU and CPU workers consume work from the same shared queue, and usefulness is judged by total wall-clock runtime of the complete grid rather than by isolated device throughput alone.

TODO link to existing heterogeneous compute systems designs; worker queue multiprocessing

## Calibration Mechanism

After the core throughput hypothesis is validated, it motivates a calibration procedure for the current workload and hardware, aiming to find the lowest-runtime configuration among:

The algorithm is:

1\. Measure the GPU-only baseline on the calibration workload  
This yields the runtime B_gpu

2\. Enumerate candidate hybrid configurations over the Cartesian product of:

- - CPU worker count p ; bounded by available CPU worker counts on the machine
    - CPU batch size ; bounded by the input dataset regime
    - fixed GPU batch-size of 64, optimizing it is outside the scope of this experiment

3\. For each candidate, run the actual hybrid system:

- - start one GPU worker
    - add p CPU workers in parallel

4\. Record one calibration record for each candidate's run:

- - CPU worker count p
    - CPU batch size
    - measured total runtime
    - measured throughput (points/s)

5\. Apply a stopping rule over CPU worker counts:

- - stop after retry consecutive non-improvements once added CPU workers no longer reduce runtime

6\. Output the list of records sorted by lowest runtime

7\. Mode selection is then simple:

- - choose gpu_only if no tested hybrid record improves on the GPU-only baseline
    - otherwise choose the first hybrid record in the sorted list

CPU assistance is always measured in parallel with the GPU, not as a separate CPU-only cohort. This is crucial because memory-bus contention, scheduler overhead, queue behavior, and other shared-resource effects only appear under heterogeneous compute context; then this calibration routine mimics the real workload correctly.

All workers consume the same shared queue, and the same scheduling policy for the final system is used: pick-up-as-you-finish.

#### Caching and Reuse (WIP)

Calibration introduces extra work, so its utility depends on amortization across repeated runs. This matters especially because a full worker-count sweep is more expensive than a single static choice.  
We amortize the calibration cost via caching over model and hardware configuration, such that calibration results remain valid across runs, even when trained weights differ.  
If the same model family and hardware are used across many repeated landscape computations, one calibration result can be reused across those runs.

## Validation Methodology

The measurement target for all runtime comparisons is grid computation alone:

- constructing and evaluating all grid points
- excluding startup and setup that are not part of point evaluation and result collection
- while still logging calibration and scheduler overhead separately for analysis

Performance comparisons are valid only if they preserve the resulting loss surface.

NOTE: On the Mac/MPS platform, repeated perturbed-model evaluation is not numerically stable enough to serve as the primary validation oracle: repeated vanilla runs can disagree on the same grid point, whereas the same validation procedure is stable on cpu and cuda. Therefore, all numerical validation was done on the stable CUDA backend.

## Statistical Analysis

Runtime and throughput summaries are currently reported as averages over repeated scalar measurements. WIP: revise for rigor later.

# EXPERIMENTS

## Experimental Setup

- hardware platform: Google Colab T4-GPU
- dataset: CIFAR-10 (image classification ML task)
- models: ResNet20 PLUS at least one more mobile-oriented image classifier with expected higher CPU affinity TODO!!
- runtime parameters: subset size (instead of full dataset to reduce test wait time, but does not reduce arithmetic intensity of operations, so does not invalidate accuracy of results wrt real scenarios), batch size, num_batches, and numerical validation tolerance

For the applicability and calibration experiments, we use an 8x8 grid. This resolution is smaller than typical end-use landscape computations, but it is sufficient for the tested worker counts to participate in the shared-queue workload rather than terminating before mixed-device scheduling behavior emerges.

NOTE: The model and workload choices reflect realistic loss-landscape analysis use cases, consistent with systems such as LossLens.  
TODO: add source references validating/supporting this use-case framing from the LossLens paper.

TODO: insert the exact evaluated setup and motivate the final model, dataset, and grid choices

## Experiment 1: Hybrid Applicability

### Goal

Recall the central claim that hybrid usefulness is determined by the relative inference capability of GPU to CPU.  
Hybrid applicability is investigated by varying this power ratio, via controlled addition of latency to GPU's workload routine, simulating longer inference times for this device.

The expected result is a threshold behavior:

- when GPU throughput is sufficiently dominant, vanilla GPU-only should be faster
- as GPU throughput is reduced relative to CPU throughput, hybridization should become beneficial
- the positive case is therefore a conditional speedup case against the vanilla single-GPU baseline

If the relative-throughput hypothesis is correct, as gpu_slowdown_factor increases, the GPU/CPU throughput should move from GPU-dominant toward CPU-competitive regimes, and the amount of useful CPU contribution should increase.

### Protocol

The "crossover region" is \[s_low, s_high\], where:

- s_low: largest tested slowdown with speedup <= 1.0
- s_high: smallest tested slowdown with speedup > 1.0

This interval contains the applicability threshold for the current workload and hardware.

To find it quickly, we use an adaptive search:

1\. evaluate hybrid at slowdown = 1.0

2\. increase slowdown multiplicatively until a losing/winning pair is bracketed

3\. refine that bracket with midpoint tests

The comparison quantity is:

- speedup = t_vanilla_scaled / t_hybrid
- t_vanilla_scaled = slowdown \* t_vanilla

where:

- slowdown is the artificial GPU slowdown factor
- t_vanilla is the measured vanilla runtime on the same workload
- t_hybrid is the measured hybrid runtime

This adaptive search is specific to the RQ1 experiment. Its only purpose is to locate the applicability threshold efficiently.

### Follow-up Sampling

After locating the crossover interval, we linearly sample slowdown values inside that interval to obtain a local view of how hybrid speedup changes around applicability onset.

### Recorded Metrics

For each run of this sweep, we record:

- vanilla/hybrid total grid-processing runtime
- vanilla/hybrid throughput in grid points per second
- GPU/CPU-worker-set throughput inside hybrid in grid points per second
- hybrid speedup over vanilla, computed as vanilla_time / hybrid_time
- numerical equivalence of the resulting surfaces inside the validated backend-stable region

## Experiment 2: Calibration Evaluation

#### System Consequence

This experiment validates the calibration mechanism that follows from the throughput-ratio insight.

### Goal

Its motivation is:

- if hybrid usefulness depends on the effective GPU/CPU throughput regime
- then the system should not hard-code a fixed hybrid configuration
- it should measure candidate hybrid settings and select the best tested one for the current machine and workload

The hypothesis is:

- as effective GPU throughput decreases relative to the available CPU worker set, adding more CPU helpers is beneficial, until resource contention becomes limiting

### Fixed Regime

This experiment follows Experiment 1 by fixing one slowdown value inside a chosen GPU/CPU throughput regime. Again, slowdown control is useful to test the calibration hypothesis by varying the inference ratio.

### Search Space

The calibration search space is:

- CPU worker count p; with upper bound as the number of available CPU cores
- CPU batch size
- fixed GPU batch size of 64

### Protocol

1\. measure the GPU-only baseline runtime under that fixed slowdown

2\. evaluate each candidate from the Cartesian product of tested p and CPU batch size values under parallel GPU+CPU

3\. record total runtime and throughput for each candidate

4\. stop after retry consecutive non-improvements in best runtime as p increases

5\. sort the tested records by lowest runtime

### Recorded Metrics

The result is interpreted as follows:

- if no hybrid candidate has lower runtime than GPU-only baseline, calibration selects gpu_only
- otherwise calibration selects the lowest-runtime tested hybrid configuration

# RESULTS

## Result 1: Hybrid Applicability

T4/CUDA adaptive-search output (colab-t4-gpu-crossover-region-results.txt) reports:

## Result 2: Calibration Evaluation

Using the present search space with p in {1, 2, 4} and CPU batch size in {4, 8, 16}, the calibration showcase supports the hypothesis:

- at slowdown = 2.4, calibration selects p = 1, CPU batch size 8
- at slowdown = 4.2, calibration selects p = 2, CPU batch size 8
- at slowdown = 8.2, calibration selects p = 4, CPU batch size 8

# DISCUSSION

## Interpretation of Hybrid Applicability

The initial experiment supports the central question by testing the condition under which hybrid execution becomes useful: as the effective GPU advantage decreases relative to the available CPU worker set, CPU assistance should become more likely to reduce total grid runtime.

## Interpretation of Calibration Behavior

Across the calibration showcase runs, increasing slowdown reduces GPU-only throughput and shifts the selected hybrid configuration from p = 1 to p = 2 and then to p = 4. This is consistent with the hypothesis that weaker effective GPU dominance makes larger CPU helper sets more beneficial.

## FUTURE WORK

Generalization across architectures and hardware settings remains future work.

Memory bandwidth is a known bottleneck for inference workloads, and a larger memory subsystem with more aggregate bandwidth may mitigate this.  
TODO: add source(s) supporting this claim of 'common inference bottleneck'

Compilation-oriented optimizations may be revisited later once the hybrid scheduling claim, calibration procedure, and validation methodology are stable;
