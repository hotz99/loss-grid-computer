# Loss-Grid Kernel Benchmark Framework

Reproducible benchmarking harness for the canonical loss-grid computation from Li et al. across:

- GPU-only single node
- Hybrid CPU-GPU single node
- MPI multi-GPU distributed execution

The framework is intentionally incremental: the same kernel, dataset, model factory, and instrumentation are reused from single-device runs through distributed scaling studies.

## What Is Implemented

- Filter-wise normalized random directions with deterministic seeds
- CIFAR-10 dataset loading from local Python batch files in `assets/`
- ResNet20 and ResNet20-without-skip model loading from local checkpoints
- Configurable loss-surface evaluation over an `(alpha, beta)` grid
- Three primary backends behind a common `LossGridExecutor` interface
- Decomposition strategies: `row`, `block`, `cyclic`
- MPI communication modes: `gather`, `allreduce`, `overlap`
- Per-stage timers for perturbation, transfer, forward/loss, and communication
- Throughput and optional scaling-efficiency calculations
- Output artifacts: CSV, optional Parquet, surface tensor dump, runtime logs, environment capture
- Sweep expansion from YAML/JSON

## CLI

Run a single experiment:

```bash
python3 run_experiment.py run --config configs/gpu_single.yaml
```

Run a sweep:

```bash
python3 run_experiment.py sweep --config configs/experiment_matrix.yaml
```

Run a purpose-built single-node `gpu` vs `hybrid` sweep:

```bash
python3 run_experiment.py sweep --config configs/hybrid_vs_gpu_matrix.yaml
```

Compare two completed runs:

```bash
python3 run_experiment.py compare --lhs outputs/gpu-run --rhs outputs/hybrid-run
```

The dedicated `hybrid_vs_gpu_matrix.yaml` sweep is intended to evaluate three points for matched runs:

- Surface equality: `allclose`, `max_abs_diff`, `rmse`, `nan_mismatch_count`
- Total runtime: `lhs_total_s` vs `rhs_total_s`
- Hybrid speedup over GPU baseline: `speedup_rhs_vs_lhs_baseline`

## MPI Launch Examples

Single-node, 4 ranks / 4 GPUs:

```bash
mpirun -np 4 python3 run_experiment.py run --config configs/mpi_multi_gpu.yaml
```

Open MPI with CUDA-aware transport enabled:

```bash
OMPI_MCA_opal_cuda_support=true mpirun -np 4 python3 run_experiment.py run --config configs/mpi_multi_gpu.yaml
```

SLURM example:

```bash
srun --nodes=1 --ntasks=4 --gpus-per-task=1 python3 run_experiment.py run --config configs/mpi_multi_gpu.yaml
```

## Experiment Matrix Covering RQ1-RQ4

`configs/experiment_matrix.yaml` defines a sweep that covers:

- RQ1: backend × decomposition strategy × hybrid split ratio
- RQ1: backend × decomposition strategy × CPU worker count × chunk sizing
- RQ2: CPU worker count × chunk sizing × batch size × grid resolution
- RQ3: MPI ranks × decomposition in the default matrix, with communication mode varied from `configs/mpi_multi_gpu.yaml`
- RQ4: all of the above with per-stage timers enabled

Recommended matrix:

- Backend: `gpu`, `hybrid`, `mpi`
- Decomposition: `row`, `block`, `cyclic`
- Grid resolution: `9`, `17`, `33`
- Batch size: `16`, `32`, `64`
- CPU workers: `1`, `2`, `4`, `8`
- CPU chunk size: `1`, `2`
- GPU chunk size cap: `4`, `8`, `16`
- MPI communication: `gather`, `allreduce`, `overlap`
- MPI ranks: `1`, `2`, `4`, `8`

## Performance Metric Definitions

- Throughput: `grid_points / total_runtime_seconds`
- Perturbation construction time: time spent forming `theta + alpha*d1 + beta*d2`
- Model transfer time: time spent moving a staged parameter vector and applying it to the model
- Forward/loss time: end-to-end inference and loss accumulation over the selected subset
- GPU kernel time: CUDA event elapsed time during forward/loss execution
- Communication time: time in gather/allreduce/send/receive operations
- Overlap efficiency:

```text
overlap_efficiency = max(0, 1 - total_runtime / serial_stage_sum)
serial_stage_sum = perturb + transfer + forward + communication
```

- Strong scaling efficiency:

```text
E_strong(p) = T1 / (p * Tp)
```

- Weak scaling efficiency:

```text
E_weak(p) = T1 / Tp
```

Where `T1` is the baseline runtime and `Tp` is the runtime at `p` workers under constant work per worker.

## Hybrid Scheduling Policy

High-level pseudocode:

```text
run short calibration on GPU and CPU
estimate points/sec for one GPU worker and one CPU worker
choose a larger chunk size for the GPU
start N CPU worker processes, each with its own model replica

while grid points remain:
    GPU claims next chunk from shared counter and runs forward/loss
    each CPU worker claims a small chunk from the same counter and runs forward/loss
    aggregate results as workers finish
```

Practical policy used here:

- `backend=hybrid` is true heterogeneous inference: GPU and CPU workers both evaluate loss-grid points
- CPU workers are separate processes so PyTorch CPU inference does not oversubscribe a single interpreter
- `calibration_points` estimates per-device service rate before the main schedule starts
- `cpu_chunk_size` should stay small; `gpu_chunk_size_max` caps how aggressively the GPU can pull ahead
- All workers share the same deterministic directions and dataset ordering

## Extending To Multi-Node Clusters

- Bind one rank per GPU and pin CPU affinity close to the PCIe/NVLink locality domain
- Use rank-local dataset shards or preloaded shared filesystems to avoid startup skew
- Replace root-only collection with hierarchical node-local reduction if the surface grows large
- Add topology-aware partitioning so 2-D tiles remain mostly intra-node before cross-node exchange
- Record node hostname, CUDA driver, MPI runtime, and NCCL environment in the captured metadata
- For production runs, store outputs on a shared filesystem with unique run IDs instead of local timestamps

## Reproducibility Checklist

- Fixed random seeds for directions
- `shuffle=False` and deterministic data loader generator
- `torch.use_deterministic_algorithms(True)` where supported
- Environment capture written to each run directory
- Config snapshot saved with the results

## Notes

- The default configs assume local assets under `assets/cifar-10-batches-py` and a checkpoint such as `assets/cifar10-resnet20-0.pkl`.
- If `runtime.device: auto` resolves to CPU, you can validate surface equality but not GPU-assisted speedup.
- `mpi4py`, `numpy`, `pandas`, `pyarrow`, and `matplotlib` are optional extras; features that require them fail with explicit messages.
- Host-staging-only experiments were removed from the active codepath; the prior design is documented in `FUTURE_WORK.md`.
