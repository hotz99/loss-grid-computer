#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/profile_hybrid_linux.sh <output-dir> -- <command> [args...]

Example:
  scripts/profile_hybrid_linux.sh profiles/hybrid -- \
    python3 run_experiment.py sweep --config configs/hybrid_vs_gpu_resnet_equivalent.yaml

This wrapper captures:
  - perf stat summary counters for the wrapped command
  - nvidia-smi dmon GPU telemetry once per second
  - nvidia-smi pmon per-process GPU telemetry once per second
  - the wrapped command output
EOF
}

if [[ $# -lt 3 ]]; then
  usage
  exit 1
fi

OUTPUT_DIR=$1
shift

if [[ $1 != "--" ]]; then
  usage
  exit 1
fi
shift

mkdir -p "$OUTPUT_DIR"
printf '%s\n' "$*" > "$OUTPUT_DIR/command.txt"

cleanup() {
  if [[ -n "${DMON_PID:-}" ]]; then
    kill "$DMON_PID" 2>/dev/null || true
    wait "$DMON_PID" 2>/dev/null || true
  fi
  if [[ -n "${PMON_PID:-}" ]]; then
    kill "$PMON_PID" 2>/dev/null || true
    wait "$PMON_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi dmon -s pucmt -d 1 > "$OUTPUT_DIR/nvidia_dmon.log" 2>&1 &
  DMON_PID=$!
  nvidia-smi pmon -s um -d 1 > "$OUTPUT_DIR/nvidia_pmon.log" 2>&1 &
  PMON_PID=$!
else
  echo "nvidia-smi unavailable; skipping GPU telemetry" > "$OUTPUT_DIR/nvidia_dmon.log"
  echo "nvidia-smi unavailable; skipping GPU telemetry" > "$OUTPUT_DIR/nvidia_pmon.log"
fi

perf stat \
  -d -d -d \
  -o "$OUTPUT_DIR/perf_stat.txt" \
  "$@" 2>&1 | tee "$OUTPUT_DIR/command.log"
