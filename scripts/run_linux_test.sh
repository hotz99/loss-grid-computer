#!/bin/bash
# Quick test script for Linux machine with discrete GPU
# Tests the hypothesis: unified memory contention on MPS vs separate VRAM on CUDA

set -e

echo "=========================================="
echo "Linux Hybrid Scheduler Test"
echo "=========================================="
echo ""
echo "This test compares GPU-only vs Hybrid (GPU+4CPU) performance"
echo "on Linux with discrete GPU (separate VRAM)."
echo ""
echo "Expected outcome:"
echo "  - If MPS unified memory was the bottleneck:"
echo "    → Hybrid should be FASTER or EQUAL to GPU-only on Linux"
echo "  - If scheduling/orchestration is the bottleneck:"
echo "    → Hybrid will still be SLOWER on Linux"
echo ""
echo "=========================================="
echo ""

# Check if CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Running comparison..."
python3 scripts/profile_comparison.py --config configs/linux_hybrid_test.yaml

echo ""
echo "=========================================="
echo "Additional Analysis"
echo "=========================================="
echo ""
echo "To monitor GPU utilization during execution:"
echo "  watch -n 0.5 nvidia-smi"
echo ""
echo "To test with different CPU worker counts:"
echo "  for w in 1 2 4 8; do"
echo "    sed \"s/cpu_workers: 4/cpu_workers: \$w/\" configs/linux_hybrid_test.yaml > /tmp/test_\$w.yaml"
echo "    python3 run_experiment.py run --config /tmp/test_\$w.yaml"
echo "  done"
