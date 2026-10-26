"""Correctness of the Hicks **dipole** (``monopole=False``) weights.

Mutation testing found that flipping the sign in the dipole weight formula --
``(cos(pi x) - sinc(x))`` -> ``(cos(pi x) + sinc(x))``
(``py_hicks_dipole_sign``) -- was not caught by any test file in the suite. The
monopole path is well covered; the dipole path had no correctness check at
all, only shape/plumbing coverage.

**What a dipole weight is.** Hicks interpolation reconstructs a band-limited
field from its samples with a Kaiser-windowed sinc:
``f(x0) ~= sum_n sinc(n - x0) K(n - x0) f(n)``. The dipole weights are the
same construction with ``sinc`` replaced by its derivative (the Kaiser window
``K`` is *not* differentiated -- that is the standard Hicks form), so they
interpolate the field's spatial *derivative* instead of its value.

Deriving ``sinc'`` independently of the implementation, with
``sinc(x) = sin(pi x) / (pi x)``::

    sinc'(x) = [pi x . pi cos(pi x) - pi sin(pi x)] / (pi x)^2
             = [cos(pi x) - sinc(x)] / x

which is what the implementation must compute (it writes it as
``(cos(pi x) - sinc(x)) / (x^2 + eps) * x`` to stay finite at ``x = 0``).

**Sign.** The implementation evaluates its weights at ``x = node - location``
(``_get_hicks_for_one_location_dim``), and

    d/dx0 sum_n sinc(n - x0) f(n) = -sum_n sinc'(n - x0) f(n)

so the dipole-weighted sum is **minus** the derivative at the location. That
sign is fixed by the derivation above, not read off the implementation, which
is what makes the sign check meaningful.
"""

import math

import pytest
import torch

from deepwave import _location_interpolation as li

DTYPE = torch.float64
HALFWIDTH = 4
BETA = torch.tensor(6.31, dtype=DTYPE)
EPS = 1e-10


def _window(x: torch.Tensor) -> torch.Tensor:
    """The Kaiser window factor, written out independently of the module."""
    u = 1 - (x / HALFWIDTH) ** 2
    return torch.i0(BETA * u.sqrt()) / torch.i0(BETA)


def _sinc_derivative(x: torch.Tensor) -> torch.Tensor:
    """``d/dx sinc(x)``, from the closed form derived in the module docstring."""
    safe = torch.where(x == 0, torch.ones_like(x), x)
    exact = (torch.cos(math.pi * x) - torch.sinc(x)) / safe
    # sinc is even and smooth, so sinc'(0) = 0.
    return torch.where(x == 0, torch.zeros_like(x), exact)


def _offsets(location: float) -> torch.Tensor:
    """The ``x = node - location`` values the implementation uses."""
    floor_location = math.floor(location)
    nodes = torch.arange(-HALFWIDTH + 1, HALFWIDTH + 1, dtype=DTYPE)
    return nodes + floor_location - location


@pytest.mark.parametrize("location", [10.0, 10.1, 10.37, 10.5, 10.75, 10.99])
def test_dipole_weights_match_the_closed_form(location: float) -> None:
    """The dipole weights are ``sinc'(x) * KaiserWindow(x)``, sign included."""
    x = _offsets(location)
    actual = li._hicks_window_weights(x, BETA, HALFWIDTH, False, EPS)
    expected = _sinc_derivative(x) * _window(x)
    # The implementation divides by ``x**2 + eps`` rather than ``x`` to stay
    # finite at ``x = 0``, so it differs from the exact closed form by a
    # relative ``eps / x**2`` -- around 1e-9 at these locations. The
    # tolerance below accommodates that and is still many orders of
    # magnitude tighter than any sign or formula error, which change the
    # weights by O(1).
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6), (
        f"location={location}: dipole weights differ from sinc'(x)*K(x); "
        f"max abs difference {(actual - expected).abs().max().item():.3e}"
    )
    # Non-vacuity: the weights are not all zero, and not symmetric-positive
    # (a monopole-shaped result would pass a sloppier comparison).
    assert actual.abs().max() > 0.1
    assert actual.min() < 0 < actual.max()


@pytest.mark.parametrize("location", [10.1, 10.37, 10.5, 10.75])
def test_dipole_weights_annihilate_a_constant(location: float) -> None:
    """A derivative operator returns zero on a constant field.

    The monopole weights sum to 1 (they reproduce the constant); the dipole
    weights must sum to ~0. A sign flip inside the bracket destroys this: the
    terms stop cancelling.
    """
    x = _offsets(location)
    monopole = li._hicks_window_weights(x, BETA, HALFWIDTH, True, EPS)
    dipole = li._hicks_window_weights(x, BETA, HALFWIDTH, False, EPS)
    assert abs(float(monopole.sum()) - 1.0) < 1e-2
    assert abs(float(dipole.sum())) < 1e-2, (
        f"location={location}: dipole weights sum to "
        f"{float(dipole.sum()):.3e}, so they do not annihilate a constant"
    )


@pytest.mark.parametrize("location", [10.1, 10.37, 10.5, 10.75])
@pytest.mark.parametrize("wavelength", [8.0, 12.0, 20.0])
def test_dipole_recovers_minus_the_derivative(
    location: float, wavelength: float
) -> None:
    """On a band-limited field, the dipole-weighted sum is ``-f'(location)``.

    This is the property a dipole source/receiver actually provides, and it
    pins the sign: with the bracket's sign flipped the result does not
    approximate ``-f'`` (or ``+f'``) at all.
    """
    x = _offsets(location)
    dipole = li._hicks_window_weights(x, BETA, HALFWIDTH, False, EPS)
    nodes = torch.arange(-HALFWIDTH + 1, HALFWIDTH + 1, dtype=DTYPE) + math.floor(
        location
    )
    k = 2 * math.pi / wavelength
    values = torch.sin(k * nodes)
    true_derivative = k * math.cos(k * location)
    interpolated = float((dipole * values).sum())
    assert abs(true_derivative) > 1e-3, "degenerate sample point; test is vacuous"
    ratio = interpolated / true_derivative
    assert abs(ratio - (-1.0)) < 0.05, (
        f"location={location}, wavelength={wavelength}: the dipole-weighted "
        f"sum is {interpolated:.6f} but -f'(location) is "
        f"{-true_derivative:.6f} (ratio {ratio:.4f}, expected about -1)"
    )
