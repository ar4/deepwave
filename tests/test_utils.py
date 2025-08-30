import pytest
import torch
from deepwave.wavelets import ricker
import math
import scipy.special

def _set_sources(x_s, freq, dt, nt, dtype=None, dpeak_time=0.3):
    """Create sources with amplitudes that have randomly shifted start times.
    """
    num_shots, num_sources_per_shot = x_s.shape[:2]
    sources = {}
    sources['amplitude'] = torch.zeros(num_shots,
                                       num_sources_per_shot,
                                       nt,
                                       dtype=dtype)

    sources['locations'] = x_s

    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            peak_time = 0.05 + torch.rand(1).item() * dpeak_time
            sources['amplitude'][shot, source, :] = \
                ricker(freq, nt, dt, peak_time, dtype=dtype)
    return sources


def _set_coords(num_shots, num_per_shot, nx, location='top'):
    """Create an array of coordinates at the specified location."""
    ndim = len(nx)
    coords = torch.zeros(num_shots, num_per_shot, ndim)
    coords[..., 0] = torch.arange(num_shots * num_per_shot)\
                          .reshape(num_shots, num_per_shot)
    if location == 'top':
        pass
    elif location == 'bottom':
        coords[..., 0] = float(nx[0] - 1) - coords[..., 0]
    elif location == 'middle':
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
    assert sources['amplitude'].shape == (2, 3, 100)
    assert sources['locations'].shape == (2, 3, 2)

def test_set_sources_ricker_values():
    x_s = torch.zeros(1, 1, 2)
    freq = 25
    dt = 0.004
    nt = 100
    sources = _set_sources(x_s, freq, dt, nt, dpeak_time=0.0)
    # Check if ricker wavelet is generated (simple check, not full validation)
    assert sources['amplitude'][0, 0, 0] != 0

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
    coords = _set_coords(num_shots, num_per_shot, nx, location='top')
    assert coords[0, 0, 0] == 0
    assert coords[0, 0, 1] == 5 # nx[1] / 2

def test_set_coords_location_bottom():
    num_shots = 1
    num_per_shot = 1
    nx = (10, 10)
    coords = _set_coords(num_shots, num_per_shot, nx, location='bottom')
    assert coords[0, 0, 0] == 9 # nx[0] - 1
    assert coords[0, 0, 1] == 5

def test_set_coords_location_middle():
    num_shots = 1
    num_per_shot = 1
    nx = (10, 10)
    coords = _set_coords(num_shots, num_per_shot, nx, location='middle')
    assert coords[0, 0, 0] == 5 # int(nx[0] / 2)
    assert coords[0, 0, 1] == 5

def test_set_coords_invalid_location():
    num_shots = 1
    num_per_shot = 1
    nx = (10, 10)
    with pytest.raises(ValueError, match="unsupported location"):
        _set_coords(num_shots, num_per_shot, nx, location='invalid')

def direct_2d_approx(x, x_s, dx, dt, c, f):
    """Use an approximation of the 2D Green's function to calculate the
    wavefield at a given location due to the given source.
    """
    r = torch.norm(x * dx - x_s * dx).item()
    nt = len(f)
    w = torch.fft.rfftfreq(nt, dt)
    fw = torch.fft.rfft(f)
    G = 1j / 4 * scipy.special.hankel1(0, -2 * math.pi * w * r / c)
    G[0] = 0
    s = G * fw * torch.prod(dx).item()
    u = torch.fft.irfft(s, nt)
    return u


def scattered_2d(x, x_s, x_p, dx, dt, c, dc, f):
    """Calculate the scattered wavefield at a given location."""
    u_p = direct_2d_approx(x_p, x_s, dx, dt, c, f)
    du_pdt2 = _second_deriv(u_p, dt)
    u = 2 * dc / c**3 * direct_2d_approx(x, x_p, dx, dt, c, du_pdt2)
    return u

def _second_deriv(arr, dt):
    """Calculate the second derivative."""
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
    arr = torch.tensor([0.0, 1.0, 4.0, 9.0, 16.0]) # y = x^2
    dt = 1.0
    # Expected second derivative of x^2 is 2
    result = _second_deriv(arr, dt)
    # The central difference approximation will be exact for a quadratic
    assert torch.allclose(result[1:-1], torch.tensor([2.0, 2.0, 2.0]))
    # Check boundary conditions
    assert torch.allclose(result[0], result[1])
    assert torch.allclose(result[-1], result[-2])