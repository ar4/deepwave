"""Common seismic wavelets."""
import math
import torch


def ricker(freq, length, dt, peak_time):
    """Return a Ricker wavelet with the specified central frequency."""
    t = torch.arange(length, dtype=torch.float) * dt - peak_time
    y = (1 - 2 * math.pi**2 * freq**2 * t**2) \
        * torch.exp(-math.pi**2 * freq**2 * t**2)
    return y
