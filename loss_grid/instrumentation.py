from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StageBreakdown:
    perturbation_s: float = 0.0
    transfer_s: float = 0.0
    forward_s: float = 0.0
    communication_s: float = 0.0
    gpu_kernel_s: float = 0.0
    host_preprocessing_s: float = 0.0
    total_s: float = 0.0
    overlap_efficiency: float = 0.0

    def finalize(self, total_runtime_s: float) -> "StageBreakdown":
        self.total_s = total_runtime_s
        serial_stage_sum = (
            self.perturbation_s
            + self.transfer_s
            + self.forward_s
            + self.communication_s
        )
        if serial_stage_sum > 0:
            self.overlap_efficiency = max(0.0, 1.0 - (total_runtime_s / serial_stage_sum))
        else:
            self.overlap_efficiency = 0.0
        return self
