# experiment flow

`runner.py` owns the fixed thesis pipeline:

1. `inventory.run`
2. `exp_1_algorithm.run`
3. `exp_2_hybrid.run`
4. `exp_3_cache.run`
5. `project.project`

The runner writes `config.json`, one JSON file per experiment, `projection.json`,
and `suite.json`. There is no experiment registry, context bus, suite config
ADT, artifact module, or table wrapper.

## Core Files

- `schemas.py` defines the minimal local ADTs: `DatasetSpec`, `MLTaskSpec`,
  `GridSpec`, experiment configs, `TrialSpec`, `CandidateRunResult`,
  `CandidateAggregate`, and experiment result types.
- `workloads.py` is the only registry: `WORKLOADS: dict[str, MLTaskSpec]`.
- `surface_gate.py` implements the canon relative-only surface gate.
- `stats.py` provides mean, paired speedups, t-intervals, and claim status
  helpers.
- `exp_1_algorithm.py`, `exp_2_hybrid.py`, and `exp_3_cache.py` define the
  RQ-specific planning/result shape.
- `project.py` filters and composes A/B/C result objects into the paper-facing
  projection consumed by TeX generation.
- `tests/` contains the minimal regression suite for the experiment layer.
