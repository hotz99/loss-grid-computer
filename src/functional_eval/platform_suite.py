from __future__ import annotations

from src.functional_eval.t4_suite import (
    DEFAULT_SCENARIOS,
    FULL_TEST_SET_SCENARIO,
    LEGACY_DEFAULT_SCENARIOS,
    PRD_VMAP_REPRODUCTION_SCENARIO,
    PlatformSuiteScenario,
    default_scenarios,
    main,
    run_platform_suite,
)

__all__ = [
    "DEFAULT_SCENARIOS",
    "FULL_TEST_SET_SCENARIO",
    "LEGACY_DEFAULT_SCENARIOS",
    "PRD_VMAP_REPRODUCTION_SCENARIO",
    "PlatformSuiteScenario",
    "default_scenarios",
    "main",
    "run_platform_suite",
]


if __name__ == "__main__":
    main()
