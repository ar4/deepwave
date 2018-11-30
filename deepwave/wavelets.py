"""Common seismic wavelets."""
import math
import torch


def ricker(freq, length, dt, peak_time, dtype=None):
    """Return a Ricker wavelet with the specified central frequency.

    Args:
        freq: A float specifying the central frequency
        length: An int specifying the number of time samples
        dt: A float specifying the time sample spacing
        peak_time: A float specifying the time (in secs) of the peak amplitude
        dtype: The PyTorch datatype to use. Optional, defaults to default.
    """
    t = torch.arange(float(length), dtype=dtype) * dt - peak_time
    y = (1 - 2 * math.pi**2 * freq**2 * t**2) \
        * torch.exp(-math.pi**2 * freq**2 * t**2)
    return y.to(dtype)
