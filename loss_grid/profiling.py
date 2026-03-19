"""Lightweight profiling utilities for bandwidth and timing analysis."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ProfilingSnapshot:
    """Single point-in-time measurement."""
    timestamp: float
    label: str
    rss_gb: Optional[float] = None
    cpu_percent: Optional[float] = None
    thread_count: Optional[int] = None


@dataclass
class ProfilingSection:
    """Timing for a labeled code section."""
    label: str
    start_time: float
    end_time: Optional[float] = None
    duration_s: Optional[float] = None
    rss_start_gb: Optional[float] = None
    rss_end_gb: Optional[float] = None
    rss_delta_gb: Optional[float] = None
    count: int = 1


class Profiler:
    """Lightweight profiler for hybrid execution analysis."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.snapshots: List[ProfilingSnapshot] = []
        self.sections: Dict[str, List[ProfilingSection]] = defaultdict(list)
        self.active_sections: Dict[str, ProfilingSection] = {}
        
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
        else:
            self.process = None
    
    def snapshot(self, label: str) -> None:
        """Record a point-in-time measurement."""
        if not self.enabled:
            return
        
        rss_gb = None
        cpu_percent = None
        thread_count = None
        
        if self.process:
            try:
                mem_info = self.process.memory_info()
                rss_gb = mem_info.rss / (1024 ** 3)
                cpu_percent = self.process.cpu_percent()
                thread_count = self.process.num_threads()
            except Exception:
                pass
        
        self.snapshots.append(ProfilingSnapshot(
            timestamp=time.perf_counter(),
            label=label,
            rss_gb=rss_gb,
            cpu_percent=cpu_percent,
            thread_count=thread_count,
        ))
    
    def section_start(self, label: str) -> None:
        """Start timing a named section."""
        if not self.enabled:
            return
        
        rss_gb = None
        if self.process:
            try:
                rss_gb = self.process.memory_info().rss / (1024 ** 3)
            except Exception:
                pass
        
        section = ProfilingSection(
            label=label,
            start_time=time.perf_counter(),
            rss_start_gb=rss_gb,
        )
        self.active_sections[label] = section
    
    def section_end(self, label: str) -> None:
        """End timing a named section."""
        if not self.enabled:
            return
        
        end_time = time.perf_counter()
        section = self.active_sections.pop(label, None)
        if section is None:
            return
        
        rss_gb = None
        if self.process:
            try:
                rss_gb = self.process.memory_info().rss / (1024 ** 3)
            except Exception:
                pass
        
        section.end_time = end_time
        section.duration_s = end_time - section.start_time
        section.rss_end_gb = rss_gb
        if section.rss_start_gb is not None and rss_gb is not None:
            section.rss_delta_gb = rss_gb - section.rss_start_gb
        
        self.sections[label].append(section)
    
    @contextmanager
    def section(self, label: str):
        """Context manager for timing a section."""
        self.section_start(label)
        try:
            yield
        finally:
            self.section_end(label)
    
    def summarize(self) -> Dict:
        """Generate summary statistics."""
        summary = {
            "snapshots": [],
            "sections": {},
        }
        
        # Snapshot summary
        for snap in self.snapshots:
            entry = {
                "label": snap.label,
                "timestamp": snap.timestamp,
            }
            if snap.rss_gb is not None:
                entry["rss_gb"] = round(snap.rss_gb, 3)
            if snap.cpu_percent is not None:
                entry["cpu_percent"] = round(snap.cpu_percent, 1)
            if snap.thread_count is not None:
                entry["thread_count"] = snap.thread_count
            summary["snapshots"].append(entry)
        
        # Section summary
        for label, sections in self.sections.items():
            if not sections:
                continue
            
            durations = [s.duration_s for s in sections if s.duration_s is not None]
            rss_deltas = [s.rss_delta_gb for s in sections if s.rss_delta_gb is not None]
            
            section_summary = {
                "count": len(sections),
                "total_s": round(sum(durations), 6) if durations else None,
                "mean_s": round(sum(durations) / len(durations), 6) if durations else None,
                "min_s": round(min(durations), 6) if durations else None,
                "max_s": round(max(durations), 6) if durations else None,
            }
            
            if rss_deltas:
                section_summary["mean_rss_delta_gb"] = round(
                    sum(rss_deltas) / len(rss_deltas), 3
                )
                section_summary["max_rss_delta_gb"] = round(max(rss_deltas), 3)
            
            summary["sections"][label] = section_summary
        
        return summary
    
    def print_summary(self) -> None:
        """Print a human-readable summary."""
        summary = self.summarize()
        
        print("\n=== Profiling Summary ===")
        
        if summary["sections"]:
            print("\nSections:")
            for label, stats in sorted(summary["sections"].items()):
                print(f"  {label}:")
                if stats.get("total_s") is not None:
                    print(f"    total: {stats['total_s']:.6f}s")
                if stats.get("mean_s") is not None:
                    print(f"    mean:  {stats['mean_s']:.6f}s  ({stats['count']} calls)")
                if stats.get("mean_rss_delta_gb") is not None:
                    print(f"    mem delta: {stats['mean_rss_delta_gb']:.3f} GB (mean)")
        
        if summary["snapshots"]:
            print("\nSnapshots:")
            for snap in summary["snapshots"]:
                parts = [snap["label"]]
                if "rss_gb" in snap:
                    parts.append(f"RSS={snap['rss_gb']:.3f}GB")
                if "thread_count" in snap:
                    parts.append(f"threads={snap['thread_count']}")
                print(f"  {', '.join(parts)}")


# Global instance for convenience
_global_profiler: Optional[Profiler] = None


def get_profiler() -> Profiler:
    """Get or create the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = Profiler(enabled=True)
    return _global_profiler


def enable_profiling() -> Profiler:
    """Enable global profiling and return profiler instance."""
    global _global_profiler
    _global_profiler = Profiler(enabled=True)
    return _global_profiler


def disable_profiling() -> None:
    """Disable global profiling."""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.enabled = False
