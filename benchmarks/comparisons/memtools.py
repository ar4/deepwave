"""Peak resident-memory measurement helper (stdlib only).

Mirrors `benchmarks/harness.py`'s `read_vmhwm_bytes` so peak-memory numbers
are collected the same way for every tool in this comparison: a
whole-process, monotonically-increasing high-water mark read from
`/proc/self/status`, read once at the end of a run that has this one tool
(and nothing else heavy) in its own dedicated process.
"""

from __future__ import annotations

from pathlib import Path


def read_vmhwm_bytes() -> int | None:
    """Return the current process's peak RSS (VmHWM) in bytes, or None."""
    try:
        with Path("/proc/self/status").open(encoding="ascii") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    return None
