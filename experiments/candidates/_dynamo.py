from __future__ import annotations


def reset() -> None:
    # Clear the compiled-graph cache and guards so each candidate compiles from
    # cold. Without this the shared code object accumulates cache variants across
    # workloads and trips the recompile limit on later workloads.
    try:
        import torch._dynamo as dynamo

        dynamo.reset()
    except Exception:
        return


def reset_counters() -> None:
    try:
        from torch._dynamo.utils import counters

        counters.clear()
    except Exception:
        return


def recompile_count() -> int:
    try:
        from torch._dynamo.utils import counters

        recompiles = counters.get("recompiles", {})
        return int(sum(recompiles.values()))
    except Exception:
        return 0
