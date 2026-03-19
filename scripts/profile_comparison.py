#!/usr/bin/env python3
"""
Profile comparison tool for analyzing hybrid vs GPU-only performance.

Usage:
    python scripts/profile_comparison.py --config configs/profiling_test.yaml
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def run_profiled_experiment(config_path: Path, cpu_workers: int) -> dict:
    """Run experiment and extract profiling data."""
    print(f"\n{'=' * 60}")
    print(f"Running with cpu_workers={cpu_workers}")
    print('=' * 60)
    
    # Load and modify config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    config['resources'] = config.get('resources', {})
    config['resources']['cpu_workers'] = cpu_workers
    config['experiment_name'] = f"{config['experiment_name']}-workers{cpu_workers}"
    
    # Write temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        yaml.dump(config, tmp)
        tmp_path = tmp.name
    
    try:
        # Run experiment
        result = subprocess.run(
            [sys.executable, 'run_experiment.py', 'run', '--config', tmp_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        if result.returncode != 0:
            print(f"Error running experiment: {result.stderr}")
            return None
        
        # Extract JSON output
        lines = result.stdout.split('\n')
        json_start = None
        for i, line in enumerate(lines):
            if line.strip() == '{':
                json_start = i
                break
        
        if json_start is None:
            print("Could not find JSON output")
            return None
        
        # Find matching closing brace
        brace_count = 0
        json_end = None
        for i in range(json_start, len(lines)):
            for char in lines[i]:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i
                        break
            if json_end is not None:
                break
        
        if json_end is None:
            print("Could not find complete JSON output")
            return None
        
        json_text = '\n'.join(lines[json_start:json_end + 1])
        summary = json.loads(json_text)
        
        # Extract profiling section
        profiling_start = None
        for i, line in enumerate(lines):
            if '=== Profiling Summary ===' in line:
                profiling_start = i
                break
        
        profiling_text = '\n'.join(lines[profiling_start:]) if profiling_start else ""
        
        return {
            'cpu_workers': cpu_workers,
            'summary': summary,
            'profiling_text': profiling_text,
            'stdout': result.stdout,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def compare_results(gpu_only: dict, hybrid: dict) -> None:
    """Compare GPU-only vs hybrid results."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    gpu_time = gpu_only['summary']['total_s']
    hybrid_time = hybrid['summary']['total_s']
    speedup = gpu_time / hybrid_time if hybrid_time > 0 else 0
    overhead_pct = ((hybrid_time - gpu_time) / gpu_time * 100) if gpu_time > 0 else 0
    
    print(f"\n⏱️  Wall Time:")
    print(f"  GPU-only:  {gpu_time:.4f}s")
    print(f"  Hybrid:    {hybrid_time:.4f}s")
    print(f"  Speedup:   {speedup:.3f}x")
    print(f"  Overhead:  {overhead_pct:+.1f}%")
    
    gpu_pps = gpu_only['summary']['throughput_points_per_s']
    hybrid_pps = hybrid['summary']['throughput_points_per_s']
    pps_ratio = hybrid_pps / gpu_pps if gpu_pps > 0 else 0
    
    print(f"\n📊 Throughput:")
    print(f"  GPU-only:  {gpu_pps:.2f} points/s")
    print(f"  Hybrid:    {hybrid_pps:.2f} points/s")
    print(f"  Ratio:     {pps_ratio:.3f}x")
    
    print(f"\n🔬 Stage Breakdown:")
    for stage in ['forward_s', 'transfer_s', 'perturbation_s']:
        gpu_val = gpu_only['summary'].get(stage, 0)
        hybrid_val = hybrid['summary'].get(stage, 0)
        delta = hybrid_val - gpu_val
        pct = (delta / gpu_val * 100) if gpu_val > 0 else 0
        print(f"  {stage:15s}: GPU={gpu_val:.4f}s  Hybrid={hybrid_val:.4f}s  Δ={delta:+.4f}s ({pct:+.1f}%)")
    
    print("\n" + "=" * 60)
    print("GPU-ONLY PROFILING")
    print("=" * 60)
    print(gpu_only['profiling_text'])
    
    print("\n" + "=" * 60)
    print("HYBRID PROFILING")
    print("=" * 60)
    print(hybrid['profiling_text'])


def main():
    parser = argparse.ArgumentParser(
        description="Compare GPU-only vs hybrid performance with profiling"
    )
    parser.add_argument('--config', required=True, help='Base config file')
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    # Run GPU-only
    gpu_only_result = run_profiled_experiment(config_path, cpu_workers=0)
    if gpu_only_result is None:
        print("Failed to run GPU-only experiment")
        return 1
    
    # Run hybrid
    hybrid_result = run_profiled_experiment(config_path, cpu_workers=2)
    if hybrid_result is None:
        print("Failed to run hybrid experiment")
        return 1
    
    # Compare
    compare_results(gpu_only_result, hybrid_result)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
