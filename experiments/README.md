# Loss Grid Computer: Experiments

Benchmark suite behind the thesis.

**DISCLAIMER**: For academic integrity, I state some sections of this codebase were generated using AI, to accelerate implementation. Design rationale is my own.

## Installation

Requires Python 3.12. This directory is self-contained; run everything from
here.

```bash
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` lists only the direct dependencies (torch, torchvision,
numpy, matplotlib, scikit-learn); pip resolves the rest.

## Running the pipeline

```bash
python runner.py
```

`runner.py` owns the fixed pipeline and is configured by the module-level
constants near the top of the file (device, seed, grid resolution, repeats,
workload selection, per-step toggles, and reuse sources for posthoc runs).
Each run writes a timestamped directory under
`outputs/platform_experiment_suite/` containing `config.json`, one JSON file per
step, `projection.json`, and `suite.json`. Set `OUTPUT_DIR` in the runner to
write to a fixed path instead.

## Pipeline overview

The runner executes five steps in order; each consumes the prior results:

1. `inventory.run` records the platform (device, library versions, memory).
2. `exp_1_algorithm.run` (RQ1) profiles each candidate evaluator against the
   in-place baseline and reports paired speedups with 95% intervals.
3. `exp_2_hybrid.run` (RQ2) sweeps a GPU-slowdown ladder to find where adding
   CPU workers makes the hybrid path beat GPU-only.
4. `exp_3_composition.run` (RQ3) runs native composition selection (GPU-only vs
   hybrid) with the GPU worker fixed to the RQ1-selected evaluator, then measures
   compiled-evaluator reuse over an N=4 same-family seed-variant session, folding
   in cold compile cost.
5. `project.project` filters and composes the A/B/C results into the
   paper-facing `projection.json` consumed by the thesis figures.

A relative-only surface gate (`surface_gate.py`) validates that each candidate
reproduces the baseline loss surface before its timing is trusted.

### Core modules

- `schemas.py` defines the local dataclasses (specs, configs, result types).
- `workloads.py` is the workload registry: ResNet-20 and row-GRU on CIFAR-10,
  an MLP regressor on California housing, and an MLP classifier on MNIST.
- `candidates/` holds the evaluators (baseline, vmapped, compiled, hybrid).
- `stats.py` provides means, paired speedups, t-intervals, and claim status.
- `project.py` builds the projection consumed by figure generation.

### Helper scripts

Offline utilities under `scripts/`. Run them from this directory
(`python scripts/<name>.py --help`); their path defaults resolve from here:

- `generate_vanilla_profile.py` / `persist_vanilla_profile.py` produce the
  baseline section-decomposition snapshot for the profiling table.
- `remeasure_compile_cost.py` re-times cold compile cost without rerunning grids.
- `plot_result_figures.py` renders the RQ1/RQ2 result figures.

## Tests and linting

```bash
python -m pytest   # regression suite
ruff check .       # lint
```

Tests that need dataset assets (MNIST, CIFAR-10) skip when those assets are
absent.
