from __future__ import annotations

from typing import Any

from experiments.schemas import Experiment1Result, Experiment2Result, Experiment3Result


def project(
    experiment_1: Experiment1Result,
    experiment_2: Experiment2Result,
    experiment_3: Experiment3Result,
) -> dict[str, Any]:
    return {
        "schema_version": "paper-projection-v1",
        "status": "planned",
        "rq1": _project_rq1(experiment_1),
        "rq2": _project_rq2(experiment_2),
        "rq3": _project_rq3(experiment_3),
        "claims": _compose_claims(experiment_1, experiment_2, experiment_3),
    }


def _project_rq1(experiment_1: Experiment1Result) -> dict[str, Any]:
    return {
        "rq3_config": experiment_1.rq3_config,
        "composition": experiment_1.composition,
        "candidates": [
            {
                "workload_name": aggregate.workload_name,
                "candidate": aggregate.candidate,
                "role": aggregate.role,
                "speedup_mean": aggregate.speedup_mean,
                "speedup_ci_low": aggregate.speedup_ci_low,
                "speedup_ci_high": aggregate.speedup_ci_high,
                "claim_status": aggregate.claim_status,
                "repeats": aggregate.repeats,
            }
            for aggregate in experiment_1.aggregates
        ],
    }


def _project_rq2(experiment_2: Experiment2Result) -> dict[str, Any]:
    return {
        "status": experiment_2.status,
        "workloads": experiment_2.record.get("workloads"),
    }


def _project_rq3(experiment_3: Experiment3Result) -> dict[str, Any]:
    return {
        "status": experiment_3.status,
        "headline": (experiment_3.result.get("headline") or {}),
        "record": experiment_3.record,
    }


def _compose_claims(
    experiment_1: Experiment1Result,
    experiment_2: Experiment2Result,
    experiment_3: Experiment3Result,
) -> dict[str, Any]:
    return {
        "rq1_ready": experiment_1.status != "planned",
        "rq2_ready": experiment_2.status != "planned",
        "rq3_ready": experiment_3.status != "planned",
        "implementation_status": "pending_backend_wiring",
    }
