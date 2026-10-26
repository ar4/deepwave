"""Independent equation checks of multiaxial PML at ordinary and extreme scales."""

import functools
import itertools
import math
from decimal import Decimal, localcontext

import pytest
import torch

from deepwave import _staggered_grid


@functools.lru_cache(maxsize=32)
def _reference(shape, widths, spacing, dt, speed, frequency, fraction, cells=None):
    """Integrate each cell's combined damping with 400-digit arithmetic."""
    result = []
    d = Decimal.from_float
    with localcontext() as context:
        context.prec = 400
        sigma_scale = -3 * d(speed) * d(0.001).ln() / 2
        profiles = {}
        for axis, n in enumerate(shape):
            for half in (0.0, 0.5):
                values = []
                for index in range(n):
                    position = d(index + half)
                    fractions = []
                    sigma = Decimal(0)
                    for side, width in enumerate(widths[2 * axis : 2 * axis + 2]):
                        interface = width if side == 0 else n - 1 - width
                        distance = (
                            interface - position if side == 0 else position - interface
                        )
                        x = (
                            min(Decimal(1), max(Decimal(0), distance / width))
                            if width
                            else Decimal(0)
                        )
                        fractions.append(x)
                        if width:
                            sigma += sigma_scale * x**2 / (width * d(spacing[axis]))
                    alpha = d(math.pi) * d(frequency) * (1 - max(fractions))
                    values.append((sigma, alpha, any(fractions)))
                profiles[axis, half] = values
        for axis in range(len(shape)):
            for half in (0.0, 0.5):
                a, b = [], []
                coordinates = (
                    itertools.product(*(range(n) for n in shape))
                    if cells is None
                    else cells
                )
                for cell in coordinates:
                    own, alpha, active = profiles[axis, half][cell[axis]]
                    sigma = own
                    for other in range(len(shape)):
                        if other != axis:
                            cross, _, cross_active = profiles[other, 0.0][cell[other]]
                            sigma += d(fraction) * cross
                            active = active or cross_active
                    total = sigma + alpha
                    argument = total * abs(d(dt))
                    exponential = Decimal(0) if argument > 1000 else (-argument).exp()
                    a.append(float(exponential) if active else 0.0)
                    b.append(float(sigma / total * (exponential - 1)) if total else 0.0)
                result.extend([a, b])
    return result


_CASES = [
    pytest.param(5.0, 0.001, 1500.0, 25.0, 0.15, id="ordinary"),
    pytest.param(5.0, 1e-20, 1500.0, 25.0, 0.15, id="tiny-step"),
    pytest.param(5.0, 1e100, 1e-100, 1e-100, 0.15, id="tiny-rates"),
    pytest.param(5.0, 1e-100, 1e100, 1e100, 0.15, id="huge-rates"),
    pytest.param(5.0, 1e308, 1e-320, 1e-320, 0.15, id="subnormal-rates"),
    pytest.param(1e308, 0.001, 1e308, 25.0, 0.15, id="overflowing-products"),
    pytest.param(1e-308, 1e308, 1e308, 1e308, 0.15, id="overflowing-rates"),
    pytest.param(1e308, 5e-324, 5e-324, 5e-324, 0.15, id="underflowing-rates"),
    pytest.param(1.0, 1.0, 1.0, 1e308, 0.15, id="unequal-rates"),
    pytest.param(5.0, 0.001, 0.0, 25.0, 0.15, id="zero-speed"),
    pytest.param(5.0, 0.0, 1500.0, 25.0, 0.15, id="zero-step"),
    pytest.param(5.0, 0.001, 0.0, 0.0, 0.15, id="zero-rates"),
    pytest.param(1.0, 1.0, 1e308, 0.0, 5e-324, id="tiny-cross-fraction"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(5, 6), (4, 5, 6)])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize(("dx", "dt", "speed", "frequency", "fraction"), _CASES)
def test_mpml_profiles_match_defining_equation(
    dtype, shape, sign, dx, dt, speed, frequency, fraction, numerics_device
):
    """Check every coefficient, including corners, absent faces and half cells."""
    ndim = len(shape)
    widths = [2, 0, 1, 2, 0, 2][: 2 * ndim]
    spacings = [dx / (axis + 1) for axis in range(ndim)]
    actual = _staggered_grid.set_pml_profiles(
        widths,
        2,
        [0] * (2 * ndim),
        sign * dt,
        spacings,
        speed,
        dtype,
        torch.device(numerics_device),
        frequency,
        shape,
        mpml_fraction=fraction,
    )
    expected = _reference(
        shape, tuple(widths), tuple(spacings), abs(dt), speed, frequency, fraction
    )
    finfo = torch.finfo(dtype)
    # Three-axis summation plus log/exp round trips in exceptional ranges.
    rtol = max(16 * finfo.eps, 2e-12)
    atol = 2 * finfo.tiny * finfo.eps
    for index, (value, reference) in enumerate(zip(actual, expected)):
        reference = torch.tensor(
            reference, dtype=dtype, device=numerics_device
        ).reshape(1, *shape)
        assert value.shape == reference.shape
        assert value.dtype == dtype
        assert torch.isfinite(value).all()
        if index % 2 == 0:
            bound = atol + rtol * reference * (
                1 + reference.clamp_min(finfo.tiny).log().abs()
            )
            assert ((value - reference).abs() <= bound).all()
        else:
            torch.testing.assert_close(value, reference, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mpml_chunk_boundaries_and_empty_faces(dtype):
    """Exercise a second chunk without a large wavefield or Decimal grid."""
    shape = (257, 257)
    profiles = _staggered_grid.set_pml_profiles(
        [0, 0, 0, 0],
        2,
        [0] * 4,
        1e308,
        [1.0, 1.0],
        1e308,
        dtype,
        torch.device("cpu"),
        0.0,
        shape,
        mpml_fraction=0.15,
    )
    assert len(profiles) == 8
    for profile in profiles:
        assert profile.shape == (1, *shape)
        assert torch.equal(profile, torch.zeros_like(profile))


def test_mpml_chunk_boundaries_preserve_nonzero_coefficients():
    """Cross-damping remains accurate on both sides of the 65536-cell boundary."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class BoundedDoubleTemporaries(TorchDispatchMode):
        """Reject a full-grid promotion in this exceptional float32 calculation."""

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            values = result if isinstance(result, (tuple, list)) else (result,)
            for value in values:
                if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                    assert value.numel() <= 65536
            return result

    shape = (257, 257)
    widths = (0, 0, 2, 3)
    cells = ((0, 0), (255, 0), (255, 1), (255, 2), (256, 256))
    with BoundedDoubleTemporaries():
        profiles = _staggered_grid.set_pml_profiles(
            list(widths),
            2,
            [0] * 4,
            0.001,
            [1.0, 1.0],
            1e308,
            torch.float32,
            torch.device("cpu"),
            0.0,
            shape,
            mpml_fraction=5e-324,
        )
    expected = _reference(shape, widths, (1.0, 1.0), 0.001, 1e308, 0.0, 5e-324, cells)
    for profile, reference in zip(profiles, expected):
        values = torch.stack([profile[(0, *cell)] for cell in cells])
        torch.testing.assert_close(values, torch.tensor(reference), rtol=2e-6, atol=0)
    assert profiles[1][0, 255, 1] != 0
