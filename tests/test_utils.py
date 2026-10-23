"""Tests for the small test-only helper functions in tests/helpers.py.

The helper *implementations* (``_set_sources``, ``_set_coords``,
``direct_2d_approx``, ``scattered``, ``_second_deriv``, etc.) live in
``tests/helpers.py`` so ``test_acoustic.py``, ``test_elastic.py``,
``test_scalar.py``, ``test_scalar_born.py``, and
``test_location_interpolation.py`` can share one copy; this file holds only
the tests that exercise them.
"""

import pytest
import torch
from helpers import (
    _second_deriv,
    _set_coords,
    _set_sources,
    direct_2d_approx,
    scattered,
)


# Unit tests for _set_sources
def test_set_sources_shape() -> None:
    """Test the shape of the output of _set_sources."""
    x_s = torch.zeros(2, 3, 2)
    freq = 25
    dt = 0.004
    nt = 100
    sources = _set_sources(x_s, freq, dt, nt)
    assert sources["amplitude"].shape == (2, 3, 100)
    assert sources["locations"].shape == (2, 3, 2)


def test_set_sources_ricker_values() -> None:
    """Test that _set_sources generates Ricker wavelet values."""
    x_s = torch.zeros(1, 1, 2)
    freq = 25
    dt = 0.004
    nt = 100
    sources = _set_sources(x_s, freq, dt, nt, dpeak_time=0.0)
    # Check if ricker wavelet is generated (simple check, not full validation)
    assert sources["amplitude"][0, 0, 0] != 0


def test_set_sources_invalid_xs_type() -> None:
    """Test that _set_sources raises TypeError for invalid x_s type."""
    with pytest.raises(TypeError, match=r"x_s must be a torch.Tensor."):
        _set_sources([1, 2], 25, 0.004, 100)


def test_set_sources_invalid_xs_ndim() -> None:
    """Test that _set_sources raises RuntimeError for invalid x_s dimensions."""
    x_s = torch.zeros(2, 3)  # Should be 3D
    with pytest.raises(RuntimeError, match=r"x_s must have three dimensions."):
        _set_sources(x_s, 25, 0.004, 100)


def test_set_sources_invalid_freq_type() -> None:
    """Test that _set_sources raises TypeError for invalid freq type."""
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(TypeError, match=r"freq must be a number."):
        _set_sources(x_s, "invalid", 0.004, 100)


def test_set_sources_non_positive_freq() -> None:
    """Test that _set_sources raises ValueError for non-positive freq."""
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(ValueError, match=r"freq must be positive."):
        _set_sources(x_s, 0, 0.004, 100)
    with pytest.raises(ValueError, match=r"freq must be positive."):
        _set_sources(x_s, -1, 0.004, 100)


def test_set_sources_invalid_dt_type() -> None:
    """Test that _set_sources raises TypeError for invalid dt type."""
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(TypeError, match=r"dt must be a number."):
        _set_sources(x_s, 25, "invalid", 100)


def test_set_sources_non_positive_dt() -> None:
    """Test that _set_sources raises ValueError for non-positive dt."""
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(ValueError, match=r"dt must be positive."):
        _set_sources(x_s, 25, 0, 100)
    with pytest.raises(ValueError, match=r"dt must be positive."):
        _set_sources(x_s, 25, -0.004, 100)


def test_set_sources_invalid_nt_type() -> None:
    """Test that _set_sources raises TypeError for invalid nt type."""
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(TypeError, match=r"nt must be an int."):
        _set_sources(x_s, 25, 0.004, 100.0)


def test_set_sources_non_positive_nt() -> None:
    """Test that _set_sources raises ValueError for non-positive nt."""
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(ValueError, match=r"nt must be positive."):
        _set_sources(x_s, 25, 0.004, 0)
    with pytest.raises(ValueError, match=r"nt must be positive."):
        _set_sources(x_s, 25, 0.004, -1)


def test_set_sources_invalid_dtype() -> None:
    """Test that _set_sources raises TypeError for invalid dtype."""
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(TypeError, match=r"dtype must be a torch.dtype."):
        _set_sources(x_s, 25, 0.004, 100, dtype="invalid")


def test_set_sources_negative_dpeak_time() -> None:
    """Test that _set_sources raises ValueError for negative dpeak_time."""
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(ValueError, match=r"dpeak_time must be non-negative."):
        _set_sources(x_s, 25, 0.004, 100, dpeak_time=-0.1)


# Unit tests for _set_coords
def test_set_coords_shape() -> None:
    """Test the shape of the output of _set_coords."""
    num_shots = 2
    num_per_shot = 3
    nx = (50, 50)
    coords = _set_coords(num_shots, num_per_shot, nx)
    assert coords.shape == (num_shots, num_per_shot, len(nx))


def test_set_coords_location_top() -> None:
    """Test _set_coords with 'top' location."""
    num_shots = 1
    num_per_shot = 1
    nx = (10, 10)
    coords = _set_coords(num_shots, num_per_shot, nx, location="top")
    assert coords[0, 0, 0] == 0
    assert coords[0, 0, 1] == 5  # nx[1] / 2


def test_set_coords_location_bottom() -> None:
    """Test _set_coords with 'bottom' location."""
    num_shots = 1
    num_per_shot = 1
    nx = (10, 10)
    coords = _set_coords(num_shots, num_per_shot, nx, location="bottom")
    assert coords[0, 0, 0] == 8  # nx[0] - 2
    assert coords[0, 0, 1] == 5


def test_set_coords_location_middle() -> None:
    """Test _set_coords with 'middle' location."""
    num_shots = 1
    num_per_shot = 1
    nx = (10, 10)
    coords = _set_coords(num_shots, num_per_shot, nx, location="middle")
    assert coords[0, 0, 0] == 5  # int(nx[0] / 2)
    assert coords[0, 0, 1] == 5


def test_set_coords_invalid_location() -> None:
    """Test that _set_coords raises ValueError for invalid location."""
    num_shots = 1
    num_per_shot = 1
    nx = (10, 10)
    with pytest.raises(ValueError, match="unsupported location"):
        _set_coords(num_shots, num_per_shot, nx, location="invalid")


def test_set_coords_invalid_num_shots_type() -> None:
    """Test that _set_coords raises TypeError for invalid num_shots type."""
    with pytest.raises(TypeError, match=r"num_shots must be an int."):
        _set_coords(1.0, 1, (10, 10))


def test_set_coords_non_positive_num_shots() -> None:
    """Test that _set_coords raises ValueError for non-positive num_shots."""
    with pytest.raises(ValueError, match=r"num_shots must be positive."):
        _set_coords(0, 1, (10, 10))
    with pytest.raises(ValueError, match=r"num_shots must be positive."):
        _set_coords(-1, 1, (10, 10))


def test_set_coords_invalid_num_per_shot_type() -> None:
    """Test that _set_coords raises TypeError for invalid num_per_shot type."""
    with pytest.raises(TypeError, match=r"num_per_shot must be an int."):
        _set_coords(1, 1.0, (10, 10))


def test_set_coords_non_positive_num_per_shot() -> None:
    """Test that _set_coords raises ValueError for non-positive num_per_shot."""
    with pytest.raises(ValueError, match=r"num_per_shot must be positive."):
        _set_coords(1, 0, (10, 10))
    with pytest.raises(ValueError, match=r"num_per_shot must be positive."):
        _set_coords(1, -1, (10, 10))


def test_set_coords_invalid_nx_type() -> None:
    """Test that _set_coords raises TypeError for invalid nx type."""
    with pytest.raises(TypeError, match=r"nx must be a list or tuple."):
        _set_coords(1, 1, "invalid")


def test_set_coords_empty_nx() -> None:
    """Test that _set_coords raises ValueError for empty nx."""
    with pytest.raises(ValueError, match=r"nx must not be empty."):
        _set_coords(1, 1, ())


def test_set_coords_invalid_nx_elements_type() -> None:
    """Test that _set_coords raises TypeError for invalid nx elements type."""
    with pytest.raises(TypeError, match=r"nx elements must be integers."):
        _set_coords(1, 1, (10, 10.0))


def test_set_coords_non_positive_nx_elements() -> None:
    """Test that _set_coords raises ValueError for non-positive nx elements."""
    with pytest.raises(ValueError, match=r"nx elements must be positive."):
        _set_coords(1, 1, (10, 0))
    with pytest.raises(ValueError, match=r"nx elements must be positive."):
        _set_coords(1, 1, (10, -1))


# Unit tests for direct_2d_approx
def test_direct_2d_approx_output_shape() -> None:
    """Test the output shape of direct_2d_approx."""
    x = torch.tensor([0, 0])
    x_s = torch.tensor([1, 1])
    dx = torch.tensor([1.0, 1.0])
    dt = 0.001
    c = 1500
    f = torch.randn(100)
    result = direct_2d_approx(x, x_s, dx, dt, c, f)
    assert result.shape == (100,)


# Unit tests for scattered
def test_scattered_output_shape() -> None:
    """Test the output shape of scattered."""
    x = torch.tensor([0, 0])
    x_s = torch.tensor([1, 1])
    x_p = torch.tensor([0, 0])
    dx = torch.tensor([1.0, 1.0])
    dt = 0.001
    c = 1500
    dc = 100
    f = torch.randn(100)
    result = scattered(x, x_s, x_p, dx, dt, c, dc, f, 2)
    assert result.shape == (100,)


# Unit tests for _second_deriv
def test_second_deriv_output_shape() -> None:
    """Test the output shape of _second_deriv."""
    arr = torch.randn(100)
    dt = 0.001
    result = _second_deriv(arr, dt)
    assert result.shape == (100,)


def test_second_deriv_values() -> None:
    """Test the values computed by _second_deriv."""
    arr = torch.tensor([0.0, 1.0, 4.0, 9.0, 16.0])  # y = x^2
    dt = 1.0
    # Expected second derivative of x^2 is 2
    result = _second_deriv(arr, dt)
    # The central difference approximation will be exact for a quadratic
    assert torch.allclose(result[1:-1], torch.tensor([2.0, 2.0, 2.0]))
    # Check boundary conditions
    assert torch.allclose(result[0], result[1])
    assert torch.allclose(result[-1], result[-2])
