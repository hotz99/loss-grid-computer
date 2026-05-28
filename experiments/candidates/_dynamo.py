from __future__ import annotations


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
