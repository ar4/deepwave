import math

import pytest
import scipy.special
import torch

from deepwave.wavelets import ricker


def _set_sources(x_s, freq, dt, nt, dtype=None, dpeak_time=0.3):
    """Create sources with amplitudes that have randomly shifted start times."""
    if not isinstance(x_s, torch.Tensor):
        raise TypeError("x_s must be a torch.Tensor.")
    if x_s.ndim != 3:
        raise RuntimeError("x_s must have three dimensions.")
    if not isinstance(freq, (int, float)):
        raise TypeError("freq must be a number.")
    if freq <= 0:
        raise ValueError("freq must be positive.")
    if not isinstance(dt, (int, float)):
        raise TypeError("dt must be a number.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if not isinstance(nt, int):
        raise TypeError("nt must be an int.")
    if nt <= 0:
        raise ValueError("nt must be positive.")
    if dtype is not None and not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype.")
    if dpeak_time < 0:
        raise ValueError("dpeak_time must be non-negative.")

    num_shots, num_sources_per_shot = x_s.shape[:2]
    sources = {}
    sources["amplitude"] = torch.zeros(num_shots, num_sources_per_shot, nt, dtype=dtype)

    sources["locations"] = x_s

    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            peak_time = 0.05 + torch.rand(1).item() * dpeak_time
            sources["amplitude"][shot, source, :] = ricker(
                freq,
                nt,
                dt,
                peak_time,
                dtype=dtype,
            )
    return sources


def _set_coords(num_shots, num_per_shot, nx, location="top"):
    """Create an array of coordinates at the specified location."""
    if not isinstance(num_shots, int):
        raise TypeError("num_shots must be an int.")
    if num_shots <= 0:
        raise ValueError("num_shots must be positive.")
    if not isinstance(num_per_shot, int):
        raise TypeError("num_per_shot must be an int.")
    if num_per_shot <= 0:
        raise ValueError("num_per_shot must be positive.")
    if not isinstance(nx, (list, tuple)):
        raise TypeError("nx must be a list or tuple.")
    if not nx:
        raise ValueError("nx must not be empty.")
    for dim_size in nx:
        if not isinstance(dim_size, int):
            raise TypeError("nx elements must be integers.")
        if dim_size <= 0:
            raise ValueError("nx elements must be positive.")

    ndim = len(nx)
    coords = torch.zeros(num_shots, num_per_shot, ndim)
    coords[..., 0] = torch.arange(num_shots * num_per_shot).reshape(
        num_shots,
        num_per_shot,
    )
    if location == "top":
        pass
    elif location == "bottom":
        coords[..., 0] = float(nx[0] - 1) - coords[..., 0]
    elif location == "middle":
        coords[..., 0] += int(float(nx[0]) / 2)
    else:
        raise ValueError("unsupported location")

    for dim in range(1, ndim):
        coords[..., dim] = torch.round(torch.tensor(float(nx[dim]) / 2))

    return coords.long()


# Unit tests for _set_sources
def test_set_sources_shape():
    x_s = torch.zeros(2, 3, 2)
    freq = 25
    dt = 0.004
    nt = 100
    sources = _set_sources(x_s, freq, dt, nt)
    assert sources["amplitude"].shape == (2, 3, 100)
    assert sources["locations"].shape == (2, 3, 2)


def test_set_sources_ricker_values():
    x_s = torch.zeros(1, 1, 2)
    freq = 25
    dt = 0.004
    nt = 100
    sources = _set_sources(x_s, freq, dt, nt, dpeak_time=0.0)
    # Check if ricker wavelet is generated (simple check, not full validation)
    assert sources["amplitude"][0, 0, 0] != 0


def test_set_sources_invalid_xs_type():
    with pytest.raises(TypeError, match="x_s must be a torch.Tensor."):
        _set_sources([1, 2], 25, 0.004, 100)


def test_set_sources_invalid_xs_ndim():
    x_s = torch.zeros(2, 3)  # Should be 3D
    with pytest.raises(RuntimeError, match="x_s must have three dimensions."):
        _set_sources(x_s, 25, 0.004, 100)


def test_set_sources_invalid_freq_type():
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(TypeError, match="freq must be a number."):
        _set_sources(x_s, "invalid", 0.004, 100)


def test_set_sources_non_positive_freq():
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(ValueError, match="freq must be positive."):
        _set_sources(x_s, 0, 0.004, 100)
    with pytest.raises(ValueError, match="freq must be positive."):
        _set_sources(x_s, -1, 0.004, 100)


def test_set_sources_invalid_dt_type():
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(TypeError, match="dt must be a number."):
        _set_sources(x_s, 25, "invalid", 100)


def test_set_sources_non_positive_dt():
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(ValueError, match="dt must be positive."):
        _set_sources(x_s, 25, 0, 100)
    with pytest.raises(ValueError, match="dt must be positive."):
        _set_sources(x_s, 25, -0.004, 100)


def test_set_sources_invalid_nt_type():
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(TypeError, match="nt must be an int."):
        _set_sources(x_s, 25, 0.004, 100.0)


def test_set_sources_non_positive_nt():
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(ValueError, match="nt must be positive."):
        _set_sources(x_s, 25, 0.004, 0)
    with pytest.raises(ValueError, match="nt must be positive."):
        _set_sources(x_s, 25, 0.004, -1)


def test_set_sources_invalid_dtype():
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(TypeError, match="dtype must be a torch.dtype."):
        _set_sources(x_s, 25, 0.004, 100, dtype="invalid")


def test_set_sources_negative_dpeak_time():
    x_s = torch.zeros(2, 3, 2)
    with pytest.raises(ValueError, match="dpeak_time must be non-negative."):
        _set_sources(x_s, 25, 0.004, 100, dpeak_time=-0.1)


# Unit tests for _set_coords
def test_set_coords_shape():
    num_shots = 2
    num_per_shot = 3
    nx = (50, 50)
    coords = _set_coords(num_shots, num_per_shot, nx)
    assert coords.shape == (num_shots, num_per_shot, len(nx))


def test_set_coords_location_top():
    num_shots = 1
    num_per_shot = 1
    nx = (10, 10)
    coords = _set_coords(num_shots, num_per_shot, nx, location="top")
    assert coords[0, 0, 0] == 0
    assert coords[0, 0, 1] == 5  # nx[1] / 2


def test_set_coords_location_bottom():
    num_shots = 1
    num_per_shot = 1
    nx = (10, 10)
    coords = _set_coords(num_shots, num_per_shot, nx, location="bottom")
    assert coords[0, 0, 0] == 9  # nx[0] - 1
    assert coords[0, 0, 1] == 5


def test_set_coords_location_middle():
    num_shots = 1
    num_per_shot = 1
    nx = (10, 10)
    coords = _set_coords(num_shots, num_per_shot, nx, location="middle")
    assert coords[0, 0, 0] == 5  # int(nx[0] / 2)
    assert coords[0, 0, 1] == 5


def test_set_coords_invalid_location():
    num_shots = 1
    num_per_shot = 1
    nx = (10, 10)
    with pytest.raises(ValueError, match="unsupported location"):
        _set_coords(num_shots, num_per_shot, nx, location="invalid")


def test_set_coords_invalid_num_shots_type():
    with pytest.raises(TypeError, match="num_shots must be an int."):
        _set_coords(1.0, 1, (10, 10))


def test_set_coords_non_positive_num_shots():
    with pytest.raises(ValueError, match="num_shots must be positive."):
        _set_coords(0, 1, (10, 10))
    with pytest.raises(ValueError, match="num_shots must be positive."):
        _set_coords(-1, 1, (10, 10))


def test_set_coords_invalid_num_per_shot_type():
    with pytest.raises(TypeError, match="num_per_shot must be an int."):
        _set_coords(1, 1.0, (10, 10))


def test_set_coords_non_positive_num_per_shot():
    with pytest.raises(ValueError, match="num_per_shot must be positive."):
        _set_coords(1, 0, (10, 10))
    with pytest.raises(ValueError, match="num_per_shot must be positive."):
        _set_coords(1, -1, (10, 10))


def test_set_coords_invalid_nx_type():
    with pytest.raises(TypeError, match="nx must be a list or tuple."):
        _set_coords(1, 1, "invalid")


def test_set_coords_empty_nx():
    with pytest.raises(ValueError, match="nx must not be empty."):
        _set_coords(1, 1, ())


def test_set_coords_invalid_nx_elements_type():
    with pytest.raises(TypeError, match="nx elements must be integers."):
        _set_coords(1, 1, (10, 10.0))


def test_set_coords_non_positive_nx_elements():
    with pytest.raises(ValueError, match="nx elements must be positive."):
        _set_coords(1, 1, (10, 0))
    with pytest.raises(ValueError, match="nx elements must be positive."):
        _set_coords(1, 1, (10, -1))


def direct_2d_approx(x, x_s, dx, dt, c, f):
    """Use an approximation of the 2D Green's function to calculate the
    wavefield at a given location due to the given source.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor.")
    if x.ndim != 1 or x.shape[0] != 2:
        raise RuntimeError("x must be a 1D Tensor of length 2.")
    if not isinstance(x_s, torch.Tensor):
        raise TypeError("x_s must be a torch.Tensor.")
    if x_s.ndim != 1 or x_s.shape[0] != 2:
        raise RuntimeError("x_s must be a 1D Tensor of length 2.")
    if not isinstance(dx, torch.Tensor):
        raise TypeError("dx must be a torch.Tensor.")
    if dx.ndim != 1 or dx.shape[0] != 2:
        raise RuntimeError("dx must be a 1D Tensor of length 2.")
    if not isinstance(dt, (int, float)):
        raise TypeError("dt must be a number.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if not isinstance(c, (int, float)):
        raise TypeError("c must be a number.")
    if c <= 0:
        raise ValueError("c must be positive.")
    if not isinstance(f, torch.Tensor):
        raise TypeError("f must be a torch.Tensor.")
    if f.ndim != 1:
        raise RuntimeError("f must be a 1D Tensor.")

    r = torch.norm(x * dx - x_s * dx).item()
    nt = len(f)
    w = torch.fft.rfftfreq(nt, dt)
    fw = torch.fft.rfft(f)
    h = scipy.special.hankel1(0, -2 * math.pi * w.numpy() * r / c)
    G = 1j / 4 * torch.tensor(h)
    G[0] = 0
    s = G * fw * torch.prod(dx).item()
    u = torch.fft.irfft(s, nt)
    return u


def scattered_2d(x, x_s, x_p, dx, dt, c, dc, f):
    """Calculate the scattered wavefield at a given location."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor.")
    if x.ndim != 1 or x.shape[0] != 2:
        raise RuntimeError("x must be a 1D Tensor of length 2.")
    if not isinstance(x_s, torch.Tensor):
        raise TypeError("x_s must be a torch.Tensor.")
    if x_s.ndim != 1 or x_s.shape[0] != 2:
        raise RuntimeError("x_s must be a 1D Tensor of length 2.")
    if not isinstance(x_p, torch.Tensor):
        raise TypeError("x_p must be a torch.Tensor.")
    if x_p.ndim != 1 or x_p.shape[0] != 2:
        raise RuntimeError("x_p must be a 1D Tensor of length 2.")
    if not isinstance(dx, torch.Tensor):
        raise TypeError("dx must be a torch.Tensor.")
    if dx.ndim != 1 or dx.shape[0] != 2:
        raise RuntimeError("dx must be a 1D Tensor of length 2.")
    if not isinstance(dt, (int, float)):
        raise TypeError("dt must be a number.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if not isinstance(c, (int, float)):
        raise TypeError("c must be a number.")
    if c <= 0:
        raise ValueError("c must be positive.")
    if not isinstance(dc, (int, float)):
        raise TypeError("dc must be a number.")
    if not isinstance(f, torch.Tensor):
        raise TypeError("f must be a torch.Tensor.")
    if f.ndim != 1:
        raise RuntimeError("f must be a 1D Tensor.")

    u_p = direct_2d_approx(x_p, x_s, dx, dt, c, f)
    du_pdt2 = _second_deriv(u_p, dt)
    u = 2 * dc / c**3 * direct_2d_approx(x, x_p, dx, dt, c, du_pdt2)
    return u


def _second_deriv(arr, dt):
    """Calculate the second derivative."""
    if not isinstance(arr, torch.Tensor):
        raise TypeError("arr must be a torch.Tensor.")
    if arr.ndim != 1:
        raise RuntimeError("arr must be a 1D Tensor.")
    if arr.numel() < 3:
        raise ValueError(
            "arr must have at least 3 elements for second derivative calculation.",
        )
    if not isinstance(dt, (int, float)):
        raise TypeError("dt must be a number.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    d2dt2 = torch.zeros_like(arr)
    d2dt2[1:-1] = (arr[2:] - 2 * arr[1:-1] + arr[:-2]) / dt**2
    d2dt2[0] = d2dt2[1]
    d2dt2[-1] = d2dt2[-2]
    return d2dt2


# Unit tests for direct_2d_approx
def test_direct_2d_approx_output_shape():
    x = torch.tensor([0, 0])
    x_s = torch.tensor([1, 1])
    dx = torch.tensor([1.0, 1.0])
    dt = 0.001
    c = 1500
    f = torch.randn(100)
    result = direct_2d_approx(x, x_s, dx, dt, c, f)
    assert result.shape == (100,)


# Unit tests for scattered_2d
def test_scattered_2d_output_shape():
    x = torch.tensor([0, 0])
    x_s = torch.tensor([1, 1])
    x_p = torch.tensor([0, 0])
    dx = torch.tensor([1.0, 1.0])
    dt = 0.001
    c = 1500
    dc = 100
    f = torch.randn(100)
    result = scattered_2d(x, x_s, x_p, dx, dt, c, dc, f)
    assert result.shape == (100,)


# Unit tests for _second_deriv
def test_second_deriv_output_shape():
    arr = torch.randn(100)
    dt = 0.001
    result = _second_deriv(arr, dt)
    assert result.shape == (100,)


def test_second_deriv_values():
    arr = torch.tensor([0.0, 1.0, 4.0, 9.0, 16.0])  # y = x^2
    dt = 1.0
    # Expected second derivative of x^2 is 2
    result = _second_deriv(arr, dt)
    # The central difference approximation will be exact for a quadratic
    assert torch.allclose(result[1:-1], torch.tensor([2.0, 2.0, 2.0]))
    # Check boundary conditions
    assert torch.allclose(result[0], result[1])
    assert torch.allclose(result[-1], result[-2])
