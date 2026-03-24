from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StageBreakdown:
    transfer_s: float = 0.0
    total_s: float = 0.0

    def finalize(self, total_runtime_s: float) -> "StageBreakdown":
        self.total_s = total_runtime_s
        return self
