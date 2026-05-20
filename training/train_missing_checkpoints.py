#!/usr/bin/env python3
"""Train missing same-family checkpoints for loss-grid workloads.

Iterates over (workload, seed) pairs and invokes the corresponding existing
training script via subprocess with --seed and --output. Skips pairs whose
target file already exists. Intended for assembling N>=4 same-family
checkpoints needed by the torch.compile amortization experiment.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Workload:
    label: str
    script: Path
    checkpoint_pattern: str


WORKLOADS: tuple[Workload, ...] = (
    Workload(
        label="california_mlp",
        script=REPO_ROOT / "training" / "train_california_mlp.py",
        checkpoint_pattern="assets/california-mlp-{seed}.pkl",
    ),
    Workload(
        label="mnist_mlp",
        script=REPO_ROOT / "training" / "train_mnist_mlp.py",
        checkpoint_pattern="assets/mnist-mlp-{seed}.pkl",
    ),
    Workload(
        label="cifar10_row_gru",
        script=REPO_ROOT / "training" / "train_row_gru.py",
        checkpoint_pattern="assets/cifar10-row-gru-{seed}.pkl",
    ),
)

DEFAULT_SEEDS: tuple[int, ...] = (42, 99, 1337)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Seeds to train for each workload (default: %(default)s).",
    )
    parser.add_argument(
        "--workload",
        dest="workloads",
        action="append",
        choices=[w.label for w in WORKLOADS],
        help="Restrict to specific workloads. Repeat to select multiple. Defaults to all.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional --device value forwarded to each training script (default: auto).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the commands that would run without executing them.",
    )
    args = parser.parse_args()

    selected_labels = set(args.workloads) if args.workloads else {w.label for w in WORKLOADS}
    selected = [w for w in WORKLOADS if w.label in selected_labels]

    plan: list[tuple[Workload, int, Path]] = []
    skipped: list[tuple[Workload, int, Path]] = []
    for workload in selected:
        for seed in args.seeds:
            output = REPO_ROOT / workload.checkpoint_pattern.format(seed=seed)
            if output.exists():
                skipped.append((workload, seed, output))
            else:
                plan.append((workload, seed, output))

    print(f"workloads: {[w.label for w in selected]}")
    print(f"seeds: {args.seeds}")
    print(f"to_train: {len(plan)}  already_present: {len(skipped)}")
    for workload, seed, output in skipped:
        print(f"  skip {workload.label} seed={seed} -> {output.relative_to(REPO_ROOT)}")
    for workload, seed, output in plan:
        print(f"  plan {workload.label} seed={seed} -> {output.relative_to(REPO_ROOT)}")

    if args.dry_run:
        return 0

    failures: list[tuple[Workload, int, int]] = []
    for index, (workload, seed, output) in enumerate(plan, start=1):
        cmd = [
            sys.executable,
            str(workload.script),
            "--seed",
            str(seed),
            "--output",
            str(output),
        ]
        if args.device is not None:
            cmd += ["--device", args.device]
        print(
            f"\n[{index}/{len(plan)}] train {workload.label} seed={seed} "
            f"-> {output.relative_to(REPO_ROOT)}",
            flush=True,
        )
        print("$ " + " ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            failures.append((workload, seed, result.returncode))
            print(
                f"FAILED {workload.label} seed={seed} exit={result.returncode}",
                flush=True,
            )

    print(f"\ndone: trained={len(plan) - len(failures)} failed={len(failures)}")
    for workload, seed, code in failures:
        print(f"  failed {workload.label} seed={seed} exit={code}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
