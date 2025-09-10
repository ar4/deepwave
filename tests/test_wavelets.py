"""Tests for deepwave.wavelets."""

import math

import pytest
import torch

from deepwave.wavelets import ricker


def test_ricker_basic_functionality() -> None:
    """Test the basic functionality of the ricker wavelet generation."""
    freq = 25
    length = 100
    dt = 0.001
    peak_time = 0.05
    wavelet = ricker(freq, length, dt, peak_time)

    assert isinstance(wavelet, torch.Tensor)
    assert wavelet.shape == (length,)
    assert wavelet.dtype == torch.float32  # Default dtype

    # Check peak value (should be 1.0 at peak_time)
    # The peak time is 0.05, which corresponds to index 50 (0.05 / 0.001)
    assert torch.isclose(wavelet[50], torch.tensor(1.0))

    # Check values at start and end (should be close to zero)
    assert torch.isclose(wavelet[0], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(wavelet[-1], torch.tensor(0.0), atol=1e-4)  # Increased atol


def test_ricker_different_freq() -> None:
    """Test ricker wavelet generation with different frequencies."""
    length = 100
    dt = 0.001
    peak_time = 0.05

    wavelet_low_freq = ricker(10, length, dt, peak_time)
    wavelet_high_freq = ricker(50, length, dt, peak_time)

    # Higher frequency wavelet should be narrower
    assert torch.argmax(wavelet_low_freq) == torch.argmax(
        wavelet_high_freq,
    )  # Peak time should be the same
    # Check that values away from the peak decay faster for higher freq
    assert abs(wavelet_high_freq[40]) < abs(wavelet_low_freq[40])


def test_ricker_different_length() -> None:
    """Test ricker wavelet generation with different lengths."""
    freq = 25
    dt = 0.001

    wavelet_short = ricker(freq, 50, dt, 0.02)  # Adjusted peak_time to be within bounds
    wavelet_long = ricker(freq, 200, dt, 0.05)

    assert wavelet_short.shape == (50,)
    assert wavelet_long.shape == (200,)
    assert torch.isclose(wavelet_short[20], torch.tensor(1.0))  # 0.02 / 0.001 = 20
    assert torch.isclose(wavelet_long[50], torch.tensor(1.0))


def test_ricker_different_dt() -> None:
    """Test ricker wavelet generation with different time steps (dt)."""
    freq = 25
    length = 100

    wavelet_small_dt = ricker(freq, length, 0.0005, 0.04)  # Adjusted peak_time
    wavelet_large_dt = ricker(freq, length, 0.002, 0.05)

    # Peak index should change with dt (peak_time / dt)
    assert torch.isclose(wavelet_small_dt[80], torch.tensor(1.0))  # 0.04 / 0.0005 = 80
    assert torch.isclose(wavelet_large_dt[25], torch.tensor(1.0))  # 0.05 / 0.002 = 25


def test_ricker_different_peak_time() -> None:
    """Test ricker wavelet generation with different peak times."""
    freq = 25
    length = 100
    dt = 0.001

    wavelet_early_peak = ricker(freq, length, dt, 0.02)
    wavelet_late_peak = ricker(freq, length, dt, 0.08)

    assert torch.isclose(wavelet_early_peak[20], torch.tensor(1.0))  # 0.02 / 0.001 = 20
    assert torch.isclose(wavelet_late_peak[80], torch.tensor(1.0))  # 0.08 / 0.001 = 80


def test_ricker_different_dtype() -> None:
    """Test ricker wavelet generation with different dtypes."""
    freq = 25
    length = 100
    dt = 0.001
    peak_time = 0.05

    wavelet_float32 = ricker(freq, length, dt, peak_time, dtype=torch.float32)
    wavelet_float64 = ricker(freq, length, dt, peak_time, dtype=torch.float64)

    assert wavelet_float32.dtype == torch.float32
    assert wavelet_float64.dtype == torch.float64
    assert torch.isclose(wavelet_float32[50], torch.tensor(1.0, dtype=torch.float32))
    assert torch.isclose(wavelet_float64[50], torch.tensor(1.0, dtype=torch.float64))


def test_ricker_length_one() -> None:
    """Test ricker wavelet generation with a length of one."""
    freq = 25
    length = 1
    dt = 0.001
    peak_time = 0.05
    wavelet = ricker(freq, length, dt, peak_time)

    assert wavelet.shape == (1,)
    # The single sample should be the value of the Ricker wavelet at t = -0.05
    t = torch.tensor([-0.05])
    expected_val = (1 - 2 * math.pi**2 * freq**2 * t**2) * torch.exp(
        -(math.pi**2) * freq**2 * t**2,
    )
    assert torch.isclose(wavelet[0], expected_val[0])


def test_ricker_freq_zero() -> None:
    """Test ricker wavelet generation when frequency is zero."""
    # Ricker wavelet formula becomes 1 * exp(0) = 1 when freq is 0
    freq = 0
    length = 10
    dt = 0.001
    peak_time = 0.005
    wavelet = ricker(freq, length, dt, peak_time)

    assert torch.allclose(wavelet, torch.ones(length))


def test_ricker_dt_zero_raises_error() -> None:
    """Test that ricker raises an error when dt is zero."""
    freq = 25
    length = 10
    dt = 0  # dt cannot be zero
    peak_time = 0.005
    with pytest.raises(ValueError, match="dt cannot be zero"):  # Expect ValueError
        ricker(freq, length, dt, peak_time)
