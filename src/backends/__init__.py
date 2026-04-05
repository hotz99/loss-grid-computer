from src.backends import hybrid, vanilla
from src.config import ExperimentConfig


def run_backend(config: ExperimentConfig):
    backend = config.backend.lower()
    if backend == "vanilla":
        return vanilla.run(config)
    if backend == "vanilla_compiled":
        return vanilla.run_compiled(config)
    if backend == "hybrid":
        return hybrid.run(config)
    raise ValueError(f"Unsupported backend: {config.backend}")


__all__ = ["run_backend"]
