"""Resolved wavelet samples are invariant to reciprocal time/frequency units."""

import math

import pytest
import torch

from deepwave import wavelets


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extreme_reciprocal_units_match_dimensionless_wavelets(dtype):
    """Changing finite units preserves the sampled pulse and four-corner wavelet."""
    if dtype is torch.float64:
        scales = [(1e200, 1e-202), (1e150, 1e-152), (1e-200, 1e198)]
    else:
        scales = [(1e20, 1e-22), (1e30, 1e-32), (1e-30, 1e28)]
    ricker_reference = (
        1 - 2 * (torch.pi * (torch.arange(7, dtype=torch.float64) - 3.5) * 0.01) ** 2
    ) * torch.exp(
        -((torch.pi * (torch.arange(7, dtype=torch.float64) - 3.5) * 0.01) ** 2)
    )
    for freq, dt in scales:
        ricker_actual = wavelets.ricker(freq, 7, dt, dtype=dtype)
        assert torch.equal(
            wavelets.ricker(freq, 7, dt, dtype=dtype, shape=()), ricker_actual
        )
        torch.testing.assert_close(
            ricker_actual.to(torch.float64),
            ricker_reference,
            rtol=2e-6 if dtype is torch.float32 else 1e-14,
            atol=2e-7 if dtype is torch.float32 else 1e-15,
        )
        u = (torch.arange(9, dtype=torch.float64) - 4.5) * 0.01
        gaussian_reference = torch.exp(
            -(math.pi**2 / (4 * math.log(2))) * (0.5 * u) ** 2
        ) * torch.cos(2 * torch.pi * u)
        gaussian_actual = wavelets.gaussian_pulse(freq, 0.5, nt=9, dt=dt, dtype=dtype)
        assert torch.equal(
            wavelets.gaussian_pulse(freq, 0.5, nt=9, dt=dt, dtype=dtype, shape=()),
            gaussian_actual,
        )
        torch.testing.assert_close(
            gaussian_actual.to(torch.float64),
            gaussian_reference,
            rtol=2e-6 if dtype is torch.float32 else 1e-14,
            atol=2e-7 if dtype is torch.float32 else 1e-15,
        )

        corners = (freq, 2 * freq, 3 * freq, 4 * freq)
        actual_tensor = wavelets.ormsby(*corners, nt=9, dt=dt, dtype=dtype)
        assert torch.equal(
            wavelets.ormsby(*corners, nt=9, dt=dt, dtype=dtype, shape=()),
            actual_tensor,
        )
        actual = actual_tensor.to(torch.float64)
        expected = wavelets.ormsby(
            1.0, 2.0, 3.0, 4.0, nt=9, dt=0.01, dtype=torch.float64
        )
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-6 if dtype is torch.float32 else 1e-13,
            atol=2e-7 if dtype is torch.float32 else 1e-14,
        )


@pytest.mark.parametrize(
    ("dtype", "scales"),
    [
        (torch.float64, [(1e308, 1e-310), (1e200, 1e-202)]),
        (torch.float32, [(1e38, 1e-40), (1e20, 1e-22)]),
    ],
)
def test_tone_burst_extreme_finite_frequency_product(dtype, scales):
    """Resolved carrier samples and tapers survive an overflowing angular frequency."""
    for window in ("hamming", "blackman"):
        reference = wavelets.tone_burst(
            1.0, 4, nt=405, dt=0.01, dtype=torch.float64, window=window
        )
        for freq, dt in scales:
            wavelet = wavelets.tone_burst(
                freq, 4, nt=405, dt=dt, dtype=dtype, window=window
            )
            assert torch.isfinite(wavelet).all()
            assert torch.equal(
                wavelets.tone_burst(
                    freq, 4, nt=405, dt=dt, dtype=dtype, shape=(), window=window
                ),
                wavelet,
            )
            torch.testing.assert_close(
                wavelet.to(torch.float64),
                reference,
                rtol=2e-6 if dtype is torch.float32 else 1e-12,
                atol=2e-7 if dtype is torch.float32 else 1e-14,
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gaussian_narrow_center_and_ricker_far_tail_are_finite(dtype):
    """A narrow centred Gaussian peaks at one and remote Ricker tails vanish."""
    narrow = wavelets.gaussian_pulse(
        1e200, 1e200, nt=5, dt=1e-202, peak_time=2e-202, dtype=dtype
    )
    assert torch.equal(narrow, torch.tensor([0, 0, 1, 0, 0], dtype=dtype))
    tail = wavelets.ricker(1e200, 5, 1e-202, peak_time=1.0, dtype=dtype)
    assert torch.equal(tail, torch.zeros(5, dtype=dtype))
    assert wavelets.ricker(1e200, 0, 1e-202, dtype=dtype).shape == (0,)


def test_explicit_peak_cancellation_keeps_centered_gaussian_peak():
    """An explicit grid-aligned peak survives cancellation of scaled coordinates."""
    freq = 1e200
    dt = 3e-202
    peak_time = 3 * dt
    result = wavelets.gaussian_pulse(
        freq, 1e200, nt=7, dt=dt, peak_time=peak_time, dtype=torch.float64
    )
    assert result[3] == 1
    assert torch.equal(result, torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.float64))


def test_negative_dt_orientation_is_preserved_in_fallback():
    """Reversing time preserves even pulses and reverses the sine burst."""
    dtype = torch.float64
    dt = 1e-202
    freq = 1e200
    assert torch.equal(
        wavelets.ricker(freq, 7, -dt, dtype=dtype),
        wavelets.ricker(freq, 7, dt, dtype=dtype),
    )
    assert torch.equal(
        wavelets.gaussian_pulse(freq, 0.5, nt=9, dt=-dt, dtype=dtype),
        wavelets.gaussian_pulse(freq, 0.5, nt=9, dt=dt, dtype=dtype),
    )
    assert torch.equal(
        wavelets.tone_burst(freq, 4, nt=405, dt=-dt, dtype=dtype),
        -wavelets.tone_burst(freq, 4, nt=405, dt=dt, dtype=dtype),
    )
    corners = (freq, 2 * freq, 3 * freq, 4 * freq)
    assert torch.equal(
        wavelets.ormsby(*corners, nt=9, dt=-dt, dtype=dtype),
        wavelets.ormsby(*corners, nt=9, dt=dt, dtype=dtype),
    )


def test_constructor_specific_prefactor_overflow_guards():
    """Full prefactors can overflow before a representable frequency square does."""
    dt = 1e-21
    u = (torch.arange(5, dtype=torch.float64) - 2.5) * 0.01
    expected_ricker = (1 - 2 * (math.pi * u) ** 2) * torch.exp(-((math.pi * u) ** 2))
    actual_ricker = wavelets.ricker(1e19, 5, dt, dtype=torch.float32)
    assert torch.isfinite(actual_ricker).all()
    torch.testing.assert_close(
        actual_ricker.double(), expected_ricker, rtol=2e-6, atol=2e-7
    )

    corners = (2.75e18, 5.5e18, 8.25e18, 1.1e19)
    actual_ormsby = wavelets.ormsby(*corners, nt=9, dt=dt, dtype=torch.float32)
    assert torch.isfinite(actual_ormsby).all()
    x = [(torch.arange(9, dtype=torch.float64) - 4.5) * (f * dt) for f in corners]
    ratios = tuple(f / corners[3] for f in corners[:3])
    expected = (torch.sinc(x[3]) ** 2 - ratios[2] ** 2 * torch.sinc(x[2]) ** 2) / (
        1 - ratios[2]
    ) - (
        ratios[1] ** 2 * torch.sinc(x[1]) ** 2 - ratios[0] ** 2 * torch.sinc(x[0]) ** 2
    ) / (ratios[1] - ratios[0])
    expected /= expected.abs().max()
    torch.testing.assert_close(actual_ormsby.double(), expected, rtol=2e-6, atol=2e-7)


def test_large_physical_time_extent_uses_dimensionless_fallback():
    """Reciprocal large time units cannot erase a resolved low-frequency pulse."""
    freq, dt = 1e-200, 1e200
    actual = wavelets.ricker(freq, 5, dt, dtype=torch.float64)
    u = (torch.arange(5, dtype=torch.float64) - 2.5) * (freq * dt)
    expected = (1 - 2 * (math.pi * u) ** 2) * torch.exp(-((math.pi * u) ** 2))
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
def test_ordinary_output_dtype_and_expanded_shape(dtype):
    """Every builder keeps its requested dtype and shared leading-dimension view."""
    expected_dtype = torch.get_default_dtype() if dtype is None else dtype
    calls = [
        (wavelets.ricker, (25.0, 100, 0.004), {}),
        (wavelets.tone_burst, (1e6, 5), {"nt": 510, "dt": 1e-8}),
        (wavelets.gaussian_pulse, (1e6, 0.5), {"nt": 500, "dt": 1e-8}),
        (wavelets.ormsby, (5.0, 10.0, 40.0, 60.0), {"nt": 1000, "dt": 0.001}),
    ]
    for builder, args, kwargs in calls:
        result = builder(*args, **kwargs, dtype=dtype)
        assert result.shape == (kwargs.get("nt", args[1]),)
        assert result.dtype is expected_dtype
        expanded = builder(*args, **kwargs, dtype=dtype, shape=(2, 3))
        assert expanded.shape == (2, 3, result.shape[0])
        assert torch.equal(expanded, result.expand(2, 3, result.shape[0]))


def test_explicit_peak_on_subnormal_grid_keeps_centered_gaussian_peak():
    """A grid-aligned peak stays exact when physical sample times are subnormal."""
    dt = 1e-310
    result = wavelets.gaussian_pulse(
        1e308, 1e200, nt=11, dt=dt, peak_time=5 * dt, dtype=torch.float64
    )
    expected = torch.zeros(11, dtype=torch.float64)
    expected[5] = 1
    assert torch.equal(result, expected)


def test_ormsby_extremely_separated_bands_preserve_resolved_high_band():
    """A negligible low band cannot turn the finite normalised high band into NaNs."""
    result = wavelets.ormsby(
        1e-200, 2e-200, 1e200, 2e200, nt=9, dt=1e-202, dtype=torch.float64
    )
    # The high band resolves to corners one and two in scaled sample units.
    # This oracle drops only a low-band contribution below float64's range.
    t = (torch.arange(9, dtype=torch.float64) - 4.5) * 0.01
    expected = 4 * torch.sinc(2 * t) ** 2 - torch.sinc(t) ** 2
    expected /= expected.abs().max()
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, rtol=1e-14, atol=1e-15)
