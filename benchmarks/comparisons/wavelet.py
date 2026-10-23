"""Ricker wavelet generator (stdlib + numpy only).

Reproduces `deepwave.wavelets.ricker`'s formula exactly (a plain
numpy/math re-implementation, since the third-party runner scripts in this
directory run in a throwaway venv without deepwave/torch installed) so the
source wavelet used by every tool in the comparison is numerically
identical, not just similarly-shaped.
"""

from __future__ import annotations

import math

import numpy as np


def ricker(freq: float, length: int, dt: float, peak_time: float) -> np.ndarray:
    """Return a Ricker wavelet with the specified central frequency.

    Args:
        freq: The central frequency (Hz).
        length: The number of time samples.
        dt: The time sample spacing (s).
        peak_time: The time (s) of the peak amplitude.

    Returns:
        A 1-D float64 numpy array of length `length`.
    """
    t = np.arange(float(length), dtype=np.float64) * dt - peak_time
    return (1 - 2 * math.pi**2 * freq**2 * t**2) * np.exp(
        -(math.pi**2) * freq**2 * t**2
    )
