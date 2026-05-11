from src.backends import hybrid, vanilla
from src.schemas import SchedulerRequest


def run_backend(
    request: SchedulerRequest,
    seed: int = 1337,
    gpu_slowdown_factor: float = 1.0,
):
    if request.mode._tag == "vanilla":
        return vanilla.run(
            request,
            seed,
            gpu_slowdown_factor,
        )
    if request.mode._tag == "hybrid":
        return hybrid.run(
            request,
            seed,
            gpu_slowdown_factor,
        )
    raise TypeError(f"Unsupported run mode: {type(request.mode)!r}")


__all__ = ["run_backend"]
