#!/bin/bash
# Quick memory bandwidth profiling using macOS powermetrics
#
# This script runs powermetrics in parallel with your experiment to capture
# memory bandwidth, GPU utilization, and thermal data.
#
# Usage:
#   ./scripts/profile_with_powermetrics.sh configs/profiling_test.yaml

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

CONFIG_FILE="$1"
EXPERIMENT_NAME=$(basename "$CONFIG_FILE" .yaml)
OUTPUT_DIR="outputs/powermetrics-${EXPERIMENT_NAME}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "====================================="
echo "Profiling with powermetrics"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "====================================="

# Note: powermetrics requires sudo
if ! sudo -n true 2>/dev/null; then
    echo "This script requires sudo access for powermetrics."
    echo "You may be prompted for your password."
    sudo -v
fi

# Start powermetrics in background
echo "Starting powermetrics..."
sudo powermetrics \
    --samplers cpu_power,gpu_power,thermal,network,disk \
    --show-process-coalition \
    --show-process-gpu \
    -i 500 \
    -o "$OUTPUT_DIR/powermetrics.log" &

POWERMETRICS_PID=$!
echo "powermetrics PID: $POWERMETRICS_PID"

# Give powermetrics time to start
sleep 2

# Run the experiment
echo ""
echo "Running experiment..."
export TMPDIR=/tmp
source venv/bin/activate
python3 run_experiment.py run --config "$CONFIG_FILE" 2>&1 | tee "$OUTPUT_DIR/experiment.log"

# Stop powermetrics
echo ""
echo "Stopping powermetrics..."
sudo kill -INT $POWERMETRICS_PID 2>/dev/null || true
wait $POWERMETRICS_PID 2>/dev/null || true

# Parse key metrics
echo ""
echo "====================================="
echo "Parsing metrics..."
echo "====================================="

# Extract GPU metrics
if [ -f "$OUTPUT_DIR/powermetrics.log" ]; then
    echo ""
    echo "GPU Active Residency (%):"
    grep -A 1 "GPU HW active residency" "$OUTPUT_DIR/powermetrics.log" | grep "%" | head -5
    
    echo ""
    echo "GPU Power (mW):"
    grep "GPU Power" "$OUTPUT_DIR/powermetrics.log" | head -10
    
    echo ""
    echo "CPU Package Power (mW):"
    grep "CPU Power" "$OUTPUT_DIR/powermetrics.log" | head -10
    
    echo ""
    echo "Thermal Pressure:"
    grep -i "thermal" "$OUTPUT_DIR/powermetrics.log" | grep -i "pressure\|level" | head -10
    
    echo ""
    echo "Memory Bandwidth (Read/Write):"
    grep -i "bandwidth" "$OUTPUT_DIR/powermetrics.log" | head -10
fi

echo ""
echo "====================================="
echo "Full results saved to:"
echo "  $OUTPUT_DIR/powermetrics.log"
echo "  $OUTPUT_DIR/experiment.log"
echo "====================================="
echo ""
echo "To analyze further:"
echo "  # View GPU utilization"
echo "  grep 'GPU HW active residency' $OUTPUT_DIR/powermetrics.log"
echo ""
echo "  # View memory bandwidth"
echo "  grep -i 'bandwidth' $OUTPUT_DIR/powermetrics.log"
echo ""
echo "  # View thermal throttling"
echo "  grep -i 'thermal' $OUTPUT_DIR/powermetrics.log"
