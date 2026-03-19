from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, List

from loss_grid.config import ExperimentConfig, experiment_config_from_dict


def _set_dotted(raw: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target = raw
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def expand_sweep_configs(config: ExperimentConfig) -> List[ExperimentConfig]:
    if config.cases:
        raw = config.to_dict()
        expanded = []
        for case in config.cases:
            instance = experiment_config_from_dict(raw)
            instance_raw = instance.to_dict()
            for key, value in case.items():
                _set_dotted(instance_raw, key, value)
            instance = experiment_config_from_dict(instance_raw)
            instance.sweep = {}
            instance.cases = []
            expanded.append(instance)
        return expanded

    if not config.sweep:
        return [config]

    raw = config.to_dict()
    keys = list(config.sweep.keys())
    values = [config.sweep[key] for key in keys]
    expanded = []
    for combination in itertools.product(*values):
        instance = experiment_config_from_dict(raw)
        instance_raw = instance.to_dict()
        for key, value in zip(keys, combination):
            _set_dotted(instance_raw, key, value)
        instance = experiment_config_from_dict(instance_raw)
        instance.sweep = {}
        instance.cases = []
        expanded.append(instance)
    return expanded
