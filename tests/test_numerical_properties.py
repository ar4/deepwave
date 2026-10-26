"""Reproducible generated checks with bounded, one-dimensional allocations."""

import math

import pytest
import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st

from deepwave._common import cfl_condition_n, setup_pml


@pytest.mark.parametrize(
    ("spacing_factors", "numerator_factor", "denominator"),
    [([1], 1, 512), ([4, 3], 5, 6144), ([2, 1, 1], 3, 1024)],
)
@settings(max_examples=256, derandomize=True, database=None, deadline=None)
@example(length_exponent=-1000, time_choice=1000, numerator=6143, sign=-1)
@example(length_exponent=-1000, time_choice=1000, numerator=6144, sign=-1)
@example(length_exponent=-1000, time_choice=1000, numerator=6145, sign=-1)
@example(length_exponent=1000, time_choice=0, numerator=6144, sign=1)
@given(
    length_exponent=st.integers(-1000, 1000),
    time_choice=st.integers(0, 2000),
    numerator=st.integers(0, 8192),
    sign=st.sampled_from([-1, 1]),
)
def test_cfl_matches_exact_rational_oracle(
    spacing_factors,
    numerator_factor,
    denominator,
    length_exponent,
    time_choice,
    numerator,
    sign,
):
    """Rational spatial norms give an integer oracle across extreme unit choices.

    The norms of reciprocal spacings are 1, 5/12 and 3/2 respectively.
    Powers of two change length/time units exactly, without changing the
    expected substep count. All generated inputs are finite normal doubles.
    """
    lower = max(-1000, length_exponent - 1000)
    upper = min(1000, length_exponent + 1000)
    time_exponent = lower + time_choice % (upper - lower + 1)
    spacings = [math.ldexp(factor, length_exponent) for factor in spacing_factors]
    dt = sign * math.ldexp(numerator, time_exponent - 10)
    speed = math.ldexp(1, length_exponent - time_exponent)
    expected_ratio = max(
        1, (numerator * numerator_factor + denominator - 1) // denominator
    )
    inner_dt, ratio = cfl_condition_n(spacings, dt, speed, courant_number=0.5)
    assert ratio == expected_ratio
    assert inner_dt == dt / expected_ratio


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@settings(max_examples=256, derandomize=True, database=None, deadline=None)
@given(
    exponent=st.integers(-800, 800),
    widths=st.tuples(st.integers(0, 8), st.integers(0, 8)),
    speed=st.sampled_from([0.0, 1e-20, 1500.0, 1e20]),
    frequency=st.sampled_from([0.0, 1e-20, 25.0, 1e20]),
    dt=st.sampled_from([0.0, -1e-20, 1e-20, -0.001, 0.001, 1e20]),
    start=st.sampled_from([0.0, 0.5]),
    power=st.sampled_from([0, 1, 2, 4, 8]),
)
def test_pml_is_invariant_to_exact_time_unit_changes(
    dtype,
    exponent,
    widths,
    speed,
    frequency,
    dt,
    start,
    power,
):
    """Changing time units preserves every coefficient, including zero damping.

    Complement the independent Decimal oracle in test_pml_numerics with
    generated combinations of geometry, scale, signs and inactive faces.
    Each profile has at most 24 elements; no wavefield is allocated.
    """
    n = sum(widths) + 8
    starts = [float(widths[0]), float(n - 1 - widths[1])]
    profiles = [
        setup_pml(
            widths,
            starts,
            5.0,
            math.ldexp(dt, shift),
            n,
            math.ldexp(speed, -shift),
            dtype,
            torch.device("cpu"),
            math.ldexp(frequency, -shift),
            start=start,
            n_power=power,
        )
        for shift in (0, exponent)
    ]
    for reference, transformed in zip(*profiles):
        assert torch.isfinite(transformed).all()
        torch.testing.assert_close(transformed, reference, rtol=0, atol=0)
    a, b = profiles[0]
    assert ((a >= 0) & (a <= 1)).all()
    assert ((b >= -1) & (b <= 0)).all()
