from src.backends import hybrid, vanilla
from src.config import HybridExecutionConfig, VanillaExecutionConfig


def run_backend(config: VanillaExecutionConfig | HybridExecutionConfig):
    if config._tag == "vanilla":
        return vanilla.run(config)
    if config._tag == "hybrid":
        return hybrid.run(config)


__all__ = ["run_backend"]
