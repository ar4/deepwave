"""Common seismic wavelets."""

import math
from typing import Optional, Union
import torch
from torch import Tensor


def ricker(
    freq: Union[int, float],
    length: int,
    dt: Union[int, float],
    peak_time: Union[int, float],
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Return a Ricker wavelet with the specified central frequency.

    Args:
        freq:
            A number (int or float) specifying the central frequency
        length:
            An int specifying the number of time samples
        dt:
            A number (int or float) specifying the time sample spacing
        peak_time:
            A number (int or float) specifying the time (in secs) of the peak amplitude
        dtype:
            The PyTorch datatype to use. Optional, defaults to PyTorch's
            default (float32).
    """
    t: Tensor = torch.arange(float(length), dtype=dtype) * dt - peak_time
    y: Tensor = (1 - 2 * math.pi**2 * freq**2 * t**2) * torch.exp(
        -(math.pi**2) * freq**2 * t**2
    )
    if dtype is not None:
        return y.to(dtype)
    return y
