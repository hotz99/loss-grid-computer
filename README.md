# Loss Grid Computer

Benchmark code for my thesis, *Single-Machine Optimization of Perturbation-Based
Loss Grid Computation*.

A loss landscape is a trained model's loss evaluated on a grid of parameter
perturbations. Visual comparison tools build these grids for many model variants
in one session. On a single-GPU baseline, the perturbed batch evaluations take
over 95% of grid construction time. This project measures when three
single-machine mechanisms speed up that evaluation:

- **RQ1, vectorization and compilation.** `torch.vmap` and `torch.compile`,
  alone and combined, against an in-place baseline.
- **RQ2, CPU/GPU hybrid.** A CPU worker pool alongside the GPU, swept over a
  GPU-slowdown ladder to find the CPU/GPU throughput ratio where it pays off.
- **RQ3, composition.** Choosing between GPU-only and hybrid per machine, and
  reusing compiled evaluators across a multi-variant session.

## Results in brief

Measured on an Apple M4 (MPS) and an NVIDIA T4 across four workloads, with
paired repeats and 95% confidence intervals:

- Vectorization combined with compilation was faster than the baseline in every
  case. Compilation alone stayed near parity and sometimes regressed.
- The hybrid CPU pool helped only when the CPU/GPU throughput ratio was near 1.
  With the optimized GPU evaluator, no measured machine came close, so hybrid
  scheduling was never selected.
- A four-variant session ran 3.2x to 44.6x faster than the baseline (a single
  measurement, no confidence interval).

## Workloads

| Model | Dataset |
| --- | --- |
| ResNet-20 | CIFAR-10 |
| Row-GRU | CIFAR-10 |
| MLP regressor | California housing |
| MLP classifier | MNIST |

## Repository layout

```
experiments/
  runner.py             pipeline entry point
  inventory.py          platform inventory
  exp_1_algorithm.py    RQ1
  exp_2_hybrid.py       RQ2
  exp_3_composition.py  RQ3
  candidates/           evaluators: baseline, vmapped, compiled, hybrid
  models/               model definitions
  canonical-results/    result JSON used in the thesis (MPS and T4)
  scripts/              profiling, compile-cost, and plotting utilities
  tests/                regression tests
```

## Getting started

See [`experiments/README.md`](experiments/README.md) for installation, running
the pipeline, and the helper scripts.

Datasets and trained checkpoints are not included in this repository. The code
expects them under `experiments/assets/`, with checkpoints named
`{family}-{seed}.pkl` (for example `cifar10-resnet20-0.pkl`). Tests that need
missing assets are skipped.

## AI use

Parts of this codebase were generated with AI assistance to speed up
implementation. The research design is my own.
