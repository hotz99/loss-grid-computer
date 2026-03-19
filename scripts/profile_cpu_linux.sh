#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/profile_cpu_linux.sh <output-dir> -- <command> [args...]

Example:
  scripts/profile_cpu_linux.sh profiles/cpu2 -- \
    python3 run_experiment.py run --config configs/hybrid_cpu2.yaml

This wrapper captures:
  - perf stat summary counters
  - perf record call-stack sample profile
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

perf stat \
  -d -d -d \
  -o "$OUTPUT_DIR/perf_stat.txt" \
  "$@" 2>&1 | tee "$OUTPUT_DIR/command.log"

perf record \
  -F 99 \
  -g \
  -o "$OUTPUT_DIR/perf.data" \
  "$@" > "$OUTPUT_DIR/perf_record.stdout.log" 2> "$OUTPUT_DIR/perf_record.stderr.log"

perf report \
  --stdio \
  -i "$OUTPUT_DIR/perf.data" \
  > "$OUTPUT_DIR/perf_report.txt"
