"""Tests for deepwave.resample."""

import math
import re
from typing import Any, Callable

import pytest
import torch

from deepwave.common import downsample, upsample


def test_upsample_invalid_signal_type() -> None:
    """Test that upsample raises TypeError if signal is not a torch.Tensor."""
    with pytest.raises(TypeError, match="signal must be a torch.Tensor."):
        upsample([1, 2, 3], 2)


def test_downsample_invalid_signal_type() -> None:
    """Test that downsample raises TypeError if signal is not a torch.Tensor."""
    with pytest.raises(TypeError, match="signal must be a torch.Tensor."):
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
