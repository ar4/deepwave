"""Independent, high-precision checks of complete PML coefficient profiles."""

import math
from decimal import Decimal, localcontext

import pytest
import torch

from deepwave._common import setup_pml


def _reference(widths, starts, spacing, dt, n, speed, frequency, power=2):
    """Evaluate the defining exponential integral in decimal arithmetic."""
    a, b = [], []
    with localcontext() as context:
        context.prec = 400
        d = Decimal.from_float
        for index in range(n):
            fractions = [
                min(Decimal(1), max(Decimal(0), (d(start) - index) / width))
                if side == 0 and width
                else min(Decimal(1), max(Decimal(0), (index - d(start)) / width))
                if width
                else Decimal(0)
                for side, (width, start) in enumerate(zip(widths, starts))
            ]
            if not any(fractions):
                a.append(0.0)
                b.append(0.0)
                continue
            sigma = sum(
                -(1 + power)
                * d(speed)
                * d(0.001).ln()
                * fraction**power
                / (2 * width * d(spacing))
                for width, fraction in zip(widths, fractions)
                if width and fraction
            )
            alpha = d(math.pi) * d(frequency) * (1 - max(fractions))
            total = sigma + alpha
            argument = total * abs(d(dt))
            exponential = Decimal(0) if argument > 1000 else (-argument).exp()
            a.append(float(exponential))
            b.append(float(sigma * (exponential - 1) / total) if total else 0.0)
    return a, b


def _assert_profiles_close(actual, expected, extreme=False):
    """Allow rounding and the known conditioning of exp, without a unit floor."""
    finfo = torch.finfo(actual[0].dtype)
    rtol = max(8 * finfo.eps, 2e-12 if extreme else 0)
    atol = 2 * finfo.tiny * finfo.eps
    # For a=exp(-z), a relative rounding error in z becomes about z times
    # that relative error in a. This matters for strongly damped, tiny a;
    # imposing a fixed relative bound would demand excess precision in z.
    exponent_size = expected[0].clamp_min(finfo.tiny).log().abs()
    a_error_bound = atol + rtol * expected[0] * (1 + exponent_size)
    relative_to_bound = (actual[0] - expected[0]).abs() / a_error_bound
    assert bool((relative_to_bound <= 1).all()), relative_to_bound
    torch.testing.assert_close(actual[1], expected[1], rtol=rtol, atol=atol)


_CASES = [
    pytest.param(5.0, 0.001, 1500.0, 25.0, False, id="ordinary"),
    pytest.param(5.0, 1e-20, 1500.0, 25.0, False, id="tiny-step"),
    pytest.param(5.0, 1e100, 1e-100, 1e-100, False, id="tiny-rates"),
    pytest.param(5.0, 1e-100, 1e100, 1e100, False, id="huge-rates"),
    pytest.param(5.0, 1e308, 1e-320, 1e-320, False, id="subnormal-rates"),
    pytest.param(1e308, 0.001, 1e308, 25.0, False, id="overflowing-products"),
    pytest.param(1.0, 1.0, 1e308, 1e308, True, id="huge-damping"),
    pytest.param(1e-308, 1e308, 1e308, 1e308, True, id="overflowing-rates"),
    pytest.param(1e308, 5e-324, 5e-324, 5e-324, True, id="underflowing-rates"),
    pytest.param(1.0, 1e308, 0.0, 1e308, True, id="huge-shift-zero-speed"),
    pytest.param(1e-308, 1e308, 1e308, 0.0, True, id="huge-speed-zero-shift"),
    pytest.param(1.0, 1.0, 1.0, 1e308, True, id="unequal-rates"),
    pytest.param(5.0, 0.0, 1500.0, 25.0, False, id="zero-step"),
    pytest.param(5.0, 0.001, 0.0, 0.0, False, id="zero-rates"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("power", [0, 2])
@pytest.mark.parametrize("widths", [[2, 3], [0, 3], [5, 5], [0, 0]])
@pytest.mark.parametrize(("spacing", "dt", "speed", "frequency", "extreme"), _CASES)
def test_complete_profiles_match_high_precision(
    dtype, sign, power, widths, spacing, dt, speed, frequency, extreme, numerics_device
):
    """All cells, including interfaces and overlapping faces, satisfy the formula."""
    n = 8
    starts = [float(widths[0]), float(n - 1 - widths[1])]
    actual = setup_pml(
        widths,
        starts,
        spacing,
        sign * dt,
        n,
        speed,
        dtype,
        torch.device(numerics_device),
        frequency,
        n_power=power,
    )
    expected = tuple(
        torch.tensor(values, dtype=dtype, device=numerics_device)
        for values in _reference(
            widths, starts, spacing, sign * dt, n, speed, frequency, power
        )
    )
    # The logarithmic extreme-scale path incurs error in log/exp round trips.
    # Normal cases retain a rounding-scale bound; no absolute unit-size floor
    # may conceal a small but representable damping coefficient.
    _assert_profiles_close(actual, expected, extreme)
    assert all(bool(torch.isfinite(value).all()) for value in actual)
    assert bool(((actual[0] >= 0) & (actual[0] <= 1)).all())
    assert bool(((actual[1] >= -1) & (actual[1] <= 0)).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("frequency", [1e-6, 1.0, 25.0, 1e6])
@pytest.mark.parametrize("widths", [[0, 0], [1, 1], [10, 0], [20, 20]])
def test_ordinary_profiles_match_high_precision(
    dtype, frequency, widths, numerics_device
):
    """Check the equation, rather than reproducing the old rounding bit-for-bit."""
    n = 60
    starts = [float(widths[0]), float(n - 1 - widths[1])]
    actual = setup_pml(
        widths,
        starts,
        5.0,
        0.001,
        n,
        1500.0,
        dtype,
        torch.device(numerics_device),
        frequency,
    )
    expected = tuple(
        torch.tensor(values, dtype=dtype, device=numerics_device)
        for values in _reference(widths, starts, 5.0, 0.001, n, 1500.0, frequency)
    )
    _assert_profiles_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_large_power_does_not_lose_the_profile_before_scaling(dtype, numerics_device):
    """A tiny polynomial and huge rate can have an ordinary, finite product."""
    widths, starts, n = [2, 0], [2.0, 4.0], 5
    actual = setup_pml(
        widths,
        starts,
        1.0,
        1e308,
        n,
        1e305,
        dtype,
        torch.device(numerics_device),
        0.0,
        n_power=2048,
    )
    expected = tuple(
        torch.tensor(values, dtype=dtype, device=numerics_device)
        for values in _reference(widths, starts, 1.0, 1e308, n, 1e305, 0.0, 2048)
    )
    torch.testing.assert_close(
        actual, expected, rtol=max(8 * torch.finfo(dtype).eps, 2e-12), atol=0
    )
    assert 0 < expected[0][1] < 1
    assert expected[1][1] != 0


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("grid_spacing", 0.0),
        ("grid_spacing", -1.0),
        ("grid_spacing", math.inf),
        ("grid_spacing", math.nan),
        ("dt", math.inf),
        ("dt", math.nan),
        ("max_vel", -1.0),
        ("max_vel", math.inf),
        ("max_vel", math.nan),
        ("pml_freq", -1.0),
        ("pml_freq", math.inf),
        ("pml_freq", math.nan),
        ("r_val", 0.0),
        ("r_val", 2.0),
        ("r_val", math.nan),
        ("n_power", -1),
    ],
)
def test_pml_rejects_configuration_outside_absorbing_profile_domain(argument, value):
    """Invalid damping parameters fail explicitly instead of yielding NaN profiles."""
    kwargs = {"grid_spacing": 5.0, "dt": 0.001, "max_vel": 1500.0, "pml_freq": 25.0}
    kwargs[argument] = value
    with pytest.raises(ValueError, match="PML"):
        setup_pml(
            [2, 2],
            [2.0, 5.0],
            n=8,
            dtype=torch.float64,
            device=torch.device("cpu"),
            **kwargs,
        )


@pytest.mark.parametrize("dt", [0.001, 1e308])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unit_reflection_has_zero_damping_coefficient(dt, dtype, numerics_device):
    """R=1 gives zero sigma even in the logarithmic calculation."""
    a, b = setup_pml(
        [2, 2],
        [2.0, 5.0],
        5.0,
        dt,
        8,
        1500.0,
        dtype,
        torch.device(numerics_device),
        25.0,
        r_val=1.0,
    )
    assert bool(torch.isfinite(a).all())
    assert bool((b == 0).all())
    # alpha vanishes at the exterior endpoint; its memory multiplier stays one.
    assert a[0] == 1
