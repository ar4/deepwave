"""Tests for deepwave.resample."""

import math
import re
from typing import Any, Callable

import pytest
import torch

from deepwave.discretisation import downsample, upsample


def test_upsample_invalid_signal_type() -> None:
    """Test that upsample raises TypeError if signal is not a torch.Tensor."""
    with pytest.raises(TypeError, match=r"signal must be a torch.Tensor."):
        upsample([1, 2, 3], 2)


def test_downsample_invalid_signal_type() -> None:
    """Test that downsample raises TypeError if signal is not a torch.Tensor."""
    with pytest.raises(TypeError, match=r"signal must be a torch.Tensor."):
        downsample([1, 2, 3], 2)


@pytest.mark.parametrize(
    ("func", "arg_name", "arg_value", "expected_error_match", "expected_error_type"),
    [
        (upsample, "step_ratio", 0, "step_ratio must be positive.", ValueError),
        (upsample, "step_ratio", -1, "step_ratio must be positive.", ValueError),
        (upsample, "step_ratio", 1.5, "step_ratio must be an int.", TypeError),
        (
            upsample,
            "freq_taper_frac",
            -0.1,
            r"freq_taper_frac must be in [0, 1], got -0.1.",
            ValueError,
        ),
        (
            upsample,
            "freq_taper_frac",
            1.1,
            r"freq_taper_frac must be in [0, 1], got 1.1.",
            ValueError,
        ),
        (
            upsample,
            "freq_taper_frac",
            "invalid",
            "freq_taper_frac must be a float.",
            TypeError,
        ),
        (
            upsample,
            "time_pad_frac",
            -0.1,
            r"time_pad_frac must be in [0, 1], got -0.1.",
            ValueError,
        ),
        (
            upsample,
            "time_pad_frac",
            1.1,
            r"time_pad_frac must be in [0, 1], got 1.1.",
            ValueError,
        ),
        (
            upsample,
            "time_pad_frac",
            "invalid",
            "time_pad_frac must be a float.",
            TypeError,
        ),
        (upsample, "time_taper", "invalid", "time_taper must be a bool.", TypeError),
        (downsample, "step_ratio", 0, "step_ratio must be positive.", ValueError),
        (downsample, "step_ratio", -1, "step_ratio must be positive.", ValueError),
        (downsample, "step_ratio", 1.5, "step_ratio must be an int.", TypeError),
        (
            downsample,
            "freq_taper_frac",
            -0.1,
            r"freq_taper_frac must be in [0, 1], got -0.1.",
            ValueError,
        ),
        (
            downsample,
            "freq_taper_frac",
            1.1,
            r"freq_taper_frac must be in [0, 1], got 1.1.",
            ValueError,
        ),
        (
            downsample,
            "freq_taper_frac",
            "invalid",
            "freq_taper_frac must be a float.",
            TypeError,
        ),
        (
            downsample,
            "time_pad_frac",
            -0.1,
            r"time_pad_frac must be in [0, 1], got -0.1.",
            ValueError,
        ),
        (
            downsample,
            "time_pad_frac",
            1.1,
            r"time_pad_frac must be in [0, 1], got 1.1.",
            ValueError,
        ),
        (
            downsample,
            "time_pad_frac",
            "invalid",
            "time_pad_frac must be a float.",
            TypeError,
        ),
        (downsample, "time_taper", "invalid", "time_taper must be a bool.", TypeError),
        (downsample, "shift", "invalid", "shift must be a float.", TypeError),
    ],
)
def test_resample_invalid_args(
    func: Callable[..., torch.Tensor],
    arg_name: str,
    arg_value: Any,
    expected_error_match: str,
    expected_error_type: Any,
) -> None:
    """Test that upsample and downsample raise errors for invalid arguments."""
    signal = torch.randn(10)

    if arg_name == "step_ratio":
        step_ratio_val = arg_value
        kwargs = {}
    else:
        step_ratio_val = 2
        kwargs = {arg_name: arg_value}

    if expected_error_type is UserWarning:
        with pytest.warns(expected_error_type, match=re.escape(expected_error_match)):
            func(signal, step_ratio_val, **kwargs)
    else:
        with pytest.raises(expected_error_type, match=re.escape(expected_error_match)):
            func(signal, step_ratio_val, **kwargs)


def test_spike_upsample(n=128, step_ratio=2, dtype=torch.double, device=None):
    """Test that padding mitigates wraparound."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.zeros(n, dtype=dtype, device=device)
    x[0] = 1
    y = upsample(x, step_ratio, freq_taper_frac=0.25, time_pad_frac=1.0)
    assert y[len(y) // 2 :].abs().max() < 1e-4


def test_spike_downsample(n=128, step_ratio=2, dtype=torch.double, device=None):
    """Test that padding mitigates wraparound."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.zeros(n, dtype=dtype, device=device)
    x[1] = 1
    y = downsample(x, step_ratio, freq_taper_frac=0.25, time_pad_frac=1.0)
    assert y[len(y) // 2 :].abs().max() < 2e-4


def test_shift(n=128, dtype=torch.double, device=None):
    """Test the shift functionality of downsample."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # sine wave
    x = torch.sin(torch.arange(n, dtype=dtype, device=device) * 2 * math.pi / n * 6)
    y = downsample(x, 2, shift=-0.23456)
    expected = torch.sin(
        (torch.arange(n, dtype=dtype, device=device) + 0.23456) * 2 * math.pi / n * 6,
    )[::2]
    assert torch.allclose(y, expected)


@pytest.mark.parametrize("step_ratio", [2, 3])
def test_downsample_preserves_constant_for_divisible_odd_and_even_lengths(
    step_ratio: int,
) -> None:
    """Orthonormal FFT scaling preserves DC for supported coarse lengths."""
    for coarse_length in (8, 9):
        signal = torch.ones(coarse_length * step_ratio, dtype=torch.double)
        result = downsample(signal, step_ratio)
        assert torch.allclose(result, torch.ones(coarse_length, dtype=torch.double))


def test_downsample_rejects_nondivisible_time_length() -> None:
    """Downsampling requires an integer number of complete output periods."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "signal time length must be divisible by step_ratio for downsampling; "
            "pad or trim the signal explicitly."
        ),
    ):
        downsample(torch.ones(5, dtype=torch.double), 2)


@pytest.mark.parametrize("freq_taper_frac", [0.0, 1e-12])
def test_upsample_zeros_even_length_nyquist(
    freq_taper_frac: float,
) -> None:
    """An even input's Nyquist component is removed even for tiny tapers."""
    signal = (-1.0) ** torch.arange(8, dtype=torch.double)
    result = upsample(signal, 2, freq_taper_frac=freq_taper_frac)
    assert result.abs().max() < 1e-12


@pytest.mark.parametrize("freq_taper_frac", [0.0, 1e-12])
def test_downsample_zeros_even_output_nyquist(
    freq_taper_frac: float,
) -> None:
    """An even output's Nyquist component is removed even for tiny tapers."""
    coarse_length = 8
    step_ratio = 2
    sample = torch.arange(coarse_length * step_ratio, dtype=torch.double)
    phase = 2 * math.pi * (coarse_length // 2) * sample / (coarse_length * step_ratio)
    signal = torch.cos(phase)
    result = downsample(signal, step_ratio, freq_taper_frac=freq_taper_frac)
    assert result.abs().max() < 1e-12


def test_downsample_preserves_highest_bin_for_odd_output() -> None:
    """Odd outputs have no Nyquist bin, so their highest bin is retained."""
    coarse_length = 9
    step_ratio = 2
    sample = torch.arange(coarse_length * step_ratio, dtype=torch.double)
    phase = 2 * math.pi * (coarse_length // 2) * sample / (coarse_length * step_ratio)
    signal = torch.cos(phase)
    expected_sample = torch.arange(coarse_length, dtype=torch.double)
    expected = torch.cos(
        2 * math.pi * (coarse_length // 2) * expected_sample / coarse_length
    )
    result = downsample(signal, step_ratio)
    assert torch.allclose(result, expected, atol=1e-12, rtol=1e-12)
