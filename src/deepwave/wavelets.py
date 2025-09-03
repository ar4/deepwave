"""Common seismic wavelets."""

import math
from typing import Optional

import torch


def ricker(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return a Ricker wavelet with the specified central frequency.

    Args:
        freq: The central frequency.
        length: The number of time samples.
        dt: The time sample spacing.
        peak_time: The time (in secs) of the peak amplitude.
        dtype: The PyTorch datatype to use. Optional, defaults to PyTorch's
            default (float32).

    Returns:
        A PyTorch tensor representing the Ricker wavelet.

    """
    if dt == 0:
        raise ValueError("dt cannot be zero.")

    t: torch.Tensor = torch.arange(float(length), dtype=dtype) * dt - peak_time
    y: torch.Tensor = (1 - 2 * math.pi**2 * freq**2 * t**2) * torch.exp(
        -(math.pi**2) * freq**2 * t**2,
    )
    if dtype is not None:
        return y.to(dtype)
    return y
