"""Create constant and point scatterer models."""
import pytest
import torch
import numpy as np
import scipy.special
from scipy.ndimage.interpolation import shift
from deepwave.scalar import Propagator
from deepwave.scalar import BornPropagator
from deepwave.wavelets import ricker


def test_pml_width_tensor():
    """Verify that PML width supplied as int and Tensor give same result."""
    _, actual_int = run_direct_2d(propagator=scalarprop,
                                  prop_kwargs={'pml_width': 20})
    pml_width_tensor = torch.Tensor([20, 20, 20, 20, 0, 0]).long()
    _, actual_tensor = run_direct_2d(propagator=scalarprop,
                                     prop_kwargs={'pml_width':
                                                  pml_width_tensor})
    assert np.allclose(actual_int.cpu().numpy(), actual_tensor.cpu().numpy())


def test_zero_vp():
    """Verify that a 0 element in vp Tensor raises error."""
    model = torch.ones(2, 3) * 1500
    model[1, 1] = 0.0
    dx = 5.0
    with pytest.raises(RuntimeError):
        prop = Propagator({'vp': model}, dx)


def test_negative_vp():
    """Verify that a < 0 element in vp Tensor raises error."""
    model = torch.ones(2, 3) * 1500
    model[1, 1] *= -1
    dx = 5.0
    with pytest.raises(RuntimeError):
        prop = Propagator({'vp': model}, dx)


def test_direct_1d():
    """Test propagation in a constant 1D model."""
    expected, actual = run_direct_1d(propagator=scalarprop)
    diff = (expected - actual.cpu()).numpy().ravel()
    assert np.linalg.norm(diff) < 47


def test_direct_1d_double():
    """Test propagation in a constant 1D model."""
    expected, actual = run_direct_1d(propagator=scalarprop, dtype=torch.double,
                                     device=torch.device('cpu'))
    diff = (expected - actual.cpu()).numpy().ravel()
    assert np.linalg.norm(diff) < 47


def test_direct_2d():
    """Test propagation in a constant 2D model."""
    expected, actual = run_direct_2d(propagator=scalarprop)
    diff = (expected - actual.cpu()).numpy().ravel()
    assert np.linalg.norm(diff) < 1.46


def test_direct_3d():
    """Test propagation in a constant 3D model."""
    expected, actual = run_direct_3d(propagator=scalarprop,
                                     dt=0.002,
                                     prop_kwargs={'pml_width': 20})
    diff = (expected - actual.cpu()).numpy().ravel()
    assert np.linalg.norm(diff) < 0.33


def test_scatter_1d():
    """Test propagation in a 1D model with a point scatterer."""
    expected, actual = run_scatter_1d(propagator=scalarprop)
    diff = (expected - actual.cpu()).numpy().ravel()
    assert np.linalg.norm(diff) < 1.86


def test_scatter_2d():
    """Test propagation in a 2D model with a point scatterer."""
    expected, actual = run_scatter_2d(propagator=scalarprop,
                                      dt=0.001,
                                      prop_kwargs={'pml_width': 30})
    diff = (expected - actual.cpu()).numpy().ravel()
    assert np.linalg.norm(diff) < 0.017


def test_scatter_3d():
    """Test propagation in a 3D model with a point scatterer."""
    expected, actual = run_scatter_3d(propagator=scalarprop,
                                      dt=0.002,
                                      prop_kwargs={'pml_width': 30})
    diff = (expected - actual.cpu()).numpy().ravel()
    assert np.linalg.norm(diff) < 0.0015


def test_born_scatter_1d():
    """Test Born propagation in a 1D model with a point scatterer."""
    expected, actual = run_born_scatter_1d(propagator=scalarbornprop)
    diff = (expected - actual.cpu()).numpy().ravel()
    assert np.linalg.norm(diff) < 1.86


def test_born_scatter_2d():
    """Test Born propagation in a 2D model with a point scatterer."""
    expected, actual = run_born_scatter_2d(propagator=scalarbornprop,
                                           dt=0.001,
                                           prop_kwargs={'pml_width': 30})
    diff = (expected - actual.cpu()).numpy().ravel()
    assert np.linalg.norm(diff) < 0.0025


def test_born_scatter_3d():
    """Test Born propagation in a 3D model with a point scatterer."""
    expected, actual = run_born_scatter_3d(propagator=scalarbornprop,
                                           dt=0.002,
                                           prop_kwargs={'pml_width': 30})
    diff = (expected - actual.cpu()).numpy().ravel()
    assert np.linalg.norm(diff) < 6e-5


def test_gradcheck_1d():
    """Test gradcheck in a 1D model."""
    run_gradcheck_1d(propagator=scalarprop)


def test_gradcheck_2d():
    """Test gradcheck in a 2D model."""
    run_gradcheck_2d(propagator=scalarprop)


def test_gradcheck_3d():
    """Test gradcheck in a 3D model."""
    run_gradcheck_3d(propagator=scalarprop)


def test_born_gradcheck_1d():
    """Test gradcheck in a 1D model with Born propagator."""
    run_born_gradcheck_1d(propagator=scalarbornprop)


def test_born_gradcheck_2d():
    """Test gradcheck in a 2D model with Born propagator."""
    run_born_gradcheck_2d(propagator=scalarbornprop)


def test_born_gradcheck_3d():
    """Test gradcheck in a 3D model with Born propagator."""
    run_born_gradcheck_3d(propagator=scalarbornprop)


def scalarprop(model, dx, dt, source_amplitude, source_locations,
               receiver_locations, prop_kwargs=None, pml_width=None):
    """Wraps the scalar propagator."""

    if prop_kwargs is None:
        prop_kwargs = {}
    # For consistency when actual max speed changes
    prop_kwargs['vpmax'] = 2000

    # Workaround for gradcheck not accepting prop_kwargs dictionary
    if pml_width is not None:
        prop_kwargs['pml_width'] = pml_width

    device = model.device
    source_amplitude = source_amplitude.to(device)
    source_locations = source_locations.to(device)
    receiver_locations = receiver_locations.to(device)

    prop = Propagator({'vp': model}, dx, **prop_kwargs)
    receiver_amplitudes = prop(source_amplitude,
                               source_locations,
                               receiver_locations, dt)

    return receiver_amplitudes


def scalarbornprop(model, scatter, dx, dt, source_amplitude, source_locations,
                   receiver_locations, prop_kwargs=None, pml_width=None):
    """Wraps the scalar propagator."""

    if prop_kwargs is None:
        prop_kwargs = {}
    # For consistency when actual max speed changes
    prop_kwargs['vpmax'] = 2000

    # Workaround for gradcheck not accepting prop_kwargs dictionary
    if pml_width is not None:
        prop_kwargs['pml_width'] = pml_width

    device = model.device
    source_amplitude = source_amplitude.to(device)
    source_locations = source_locations.to(device)
    receiver_locations = receiver_locations.to(device)

    prop = BornPropagator({'vp': model, 'scatter': scatter}, dx, **prop_kwargs)
    receiver_amplitudes = prop(source_amplitude,
                               source_locations,
                               receiver_locations, dt)

    return receiver_amplitudes


def direct_1d(x, x_s, dx, dt, c, f):
    """Use the 1D Green's function to determine the wavefield at a given
    location due to the given source.
    """
    r = torch.abs(x - x_s).item()
    t_shift = (r / c) / dt
    u = dx * dt * c / 2 * torch.Tensor(np.cumsum(shift(f, t_shift)))
    return u


def direct_2d_approx(x, x_s, dx, dt, c, f):
    """Use an approximation of the 2D Green's function to calculate the
    wavefield at a given location due to the given source.
    """
    r = torch.norm(x - x_s).item()
    nt = len(f)
    w = np.fft.rfftfreq(nt, dt)
    fw = np.fft.rfft(f)
    G = 1j / 4 * scipy.special.hankel1(0, -2 * np.pi * w * r / c)
    G[0] = 0
    s = G * fw * np.prod(dx.numpy())
    u = torch.Tensor(np.fft.irfft(s, nt))
    return u


def direct_3d(x, x_s, dx, dt, c, f):
    """Use the 3D Green's function to determine the wavefield at a given
    location due to the given source.
    """
    r = torch.norm(x - x_s).item()
    t_shift = (r / c) / dt
    u = torch.prod(dx) / 4 / np.pi / r * torch.Tensor(shift(f, t_shift))
    return u


def scattered_1d(x, x_s, x_p, dx, dt, c, dc, f):
    """Calculate the scattered wavefield at a given location."""
    u_p = direct_1d(x_p, x_s, dx, dt, c, f)
    du_pdt2 = _second_deriv(u_p, dt)
    u = 2 * dc / c**3 * direct_1d(x, x_p, dx, dt, c, du_pdt2)
    return u


def scattered_2d(x, x_s, x_p, dx, dt, c, dc, f):
    """Calculate the scattered wavefield at a given location."""
    u_p = direct_2d_approx(x_p, x_s, dx, dt, c, f)
    du_pdt2 = _second_deriv(u_p, dt)
    u = 2 * dc / c**3 * direct_2d_approx(x, x_p, dx, dt, c, du_pdt2)
    return u


def scattered_3d(x, x_s, x_p, dx, dt, c, dc, f):
    """Calculate the scattered wavefield at a given location."""
    u_p = direct_3d(x_p, x_s, dx, dt, c, f)
    du_pdt2 = _second_deriv(u_p, dt)
    u = 2 * dc / c**3 * direct_3d(x, x_p, dx, dt, c, du_pdt2)
    return u


def _second_deriv(arr, dt):
    """Calculate the second derivative."""
    d2dt2 = torch.zeros_like(arr)
    d2dt2[1:-1] = (arr[2:] - 2 * arr[1:-1] + arr[:-2]) / dt**2
    d2dt2[0] = d2dt2[1]
    d2dt2[-1] = d2dt2[-2]
    return d2dt2


def _set_sources(x_s, freq, dt, nt, dtype=None, dpeak_time=0.3):
    """Create sources with amplitudes that have randomly shifted start times.
    """
    num_shots, num_sources_per_shot = x_s.shape[:2]
    sources = {}
    sources['amplitude'] = torch.zeros(nt, num_shots, num_sources_per_shot,
                                       dtype=dtype)

    sources['locations'] = x_s

    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            peak_time = 0.05 + torch.rand(1).item() * dpeak_time
            sources['amplitude'][:, shot, source] = \
                ricker(freq, nt, dt, peak_time, dtype=dtype)
    return sources


def _set_coords(num_shots, num_per_shot, nx, dx, location='top'):
    """Create an array of coordinates at the specified location."""
    ndim = len(nx)
    coords = torch.zeros(num_shots, num_per_shot, ndim)
    coords[..., 0] = torch.arange(num_shots * num_per_shot)\
                          .reshape(num_shots, num_per_shot)
    if location == 'top':
        pass
    elif location == 'bottom':
        coords[..., 0] = (nx[0] - 1).float() - coords[..., 0]
    elif location == 'middle':
        coords[..., 0] += int(nx[0] / 2)
    else:
        raise ValueError("unsupported location")

    for dim in range(1, ndim):
        coords[..., dim] = torch.round(nx[dim].float() / 2)

    return coords * dx


def run_direct(c, freq, dx, dt, nx,
               num_shots, num_sources_per_shot,
               num_receivers_per_shot,
               propagator, prop_kwargs, device=None,
               dtype=None):
    """Create a constant model, and the expected waveform at point,
       and the forward propagated wave.
    """
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.ones(*nx, device=device, dtype=dtype) * c

    nx = torch.Tensor(nx).long()
    dx = torch.Tensor(dx)

    nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx, dx)
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx, dx, 'bottom')
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    if len(nx) == 1:
        direct = direct_1d
    elif len(nx) == 2:
        direct = direct_2d_approx
    elif len(nx) == 3:
        direct = direct_3d
    else:
        raise ValueError("unsupported nx")

    expected = torch.zeros(nt, num_shots, num_receivers_per_shot, dtype=dtype)
    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            for receiver in range(num_receivers_per_shot):
                expected[:, shot, receiver] += \
                    direct(x_r[shot, receiver], x_s[shot, source],
                           dx, dt, c,
                           sources['amplitude'][:, shot, source]).to(dtype)

    actual = propagator(model, dx, dt, sources['amplitude'],
                        sources['locations'], x_r,
                        prop_kwargs=prop_kwargs)

    return expected, actual


def run_direct_1d(c=1500, freq=25, dx=(5,), dt=0.0001, nx=(80,),
                  num_shots=2, num_sources_per_shot=2,
                  num_receivers_per_shot=2,
                  propagator=None, prop_kwargs=None, device=None,
                  dtype=None):
    """Runs run_direct with default parameters for 1D."""

    return run_direct(c, freq, dx, dt, nx,
                      num_shots, num_sources_per_shot,
                      num_receivers_per_shot,
                      propagator, prop_kwargs, device, dtype)


def run_direct_2d(c=1500, freq=25, dx=(5, 5), dt=0.0001, nx=(50, 50),
                  num_shots=2, num_sources_per_shot=2,
                  num_receivers_per_shot=2,
                  propagator=None, prop_kwargs=None, device=None,
                  dtype=None):
    """Runs run_direct with default parameters for 2D."""

    return run_direct(c, freq, dx, dt, nx,
                      num_shots, num_sources_per_shot,
                      num_receivers_per_shot,
                      propagator, prop_kwargs, device, dtype)


def run_direct_3d(c=1500, freq=25, dx=(5, 5, 5), dt=0.0001, nx=(20, 10, 20),
                  num_shots=2, num_sources_per_shot=2,
                  num_receivers_per_shot=2,
                  propagator=None, prop_kwargs=None, device=None,
                  dtype=None):
    """Runs run_direct with default parameters for 3D."""

    return run_direct(c, freq, dx, dt, nx,
                      num_shots, num_sources_per_shot,
                      num_receivers_per_shot,
                      propagator, prop_kwargs, device, dtype)


def run_scatter(c, dc, freq, dx, dt, nx,
                num_shots, num_sources_per_shot,
                num_receivers_per_shot,
                propagator, prop_kwargs, device=None, dtype=None):
    """Create a point scatterer model, and the expected waveform at point,
       and the forward propagated wave.
    """
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.ones(*nx, device=device, dtype=dtype) * c
    model_const = model.clone()

    nx = torch.Tensor(nx).long()
    dx = torch.Tensor(dx)

    nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx, dx)
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx, dx)
    x_p = _set_coords(1, 1, nx, dx, 'middle')[0, 0]
    model[torch.split((x_p / dx).long(), 1)] += dc
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    if len(nx) == 1:
        scattered = scattered_1d
    elif len(nx) == 2:
        scattered = scattered_2d
    elif len(nx) == 3:
        scattered = scattered_3d
    else:
        raise ValueError("unsupported nx")

    expected = torch.zeros(nt, num_shots, num_receivers_per_shot, dtype=dtype)
    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            for receiver in range(num_receivers_per_shot):
                expected[:, shot, receiver] += \
                    scattered(x_r[shot, receiver], x_s[shot, source], x_p,
                              dx, dt, c, dc,
                              sources['amplitude'][:, shot, source]).to(dtype)

    y_const = propagator(model_const, dx, dt, sources['amplitude'],
                         sources['locations'], x_r,
                         prop_kwargs=prop_kwargs)
    y = propagator(model, dx, dt, sources['amplitude'], sources['locations'],
                   x_r, prop_kwargs=prop_kwargs)

    actual = y - y_const

    return expected, actual


def run_scatter_1d(c=1500, dc=50, freq=25, dx=(5,), dt=0.0001, nx=(100,),
                   num_shots=2, num_sources_per_shot=2,
                   num_receivers_per_shot=2,
                   propagator=None, prop_kwargs=None, device=None,
                   dtype=None):
    """Runs run_scatter with default parameters for 1D."""

    return run_scatter(c, dc, freq, dx, dt, nx,
                       num_shots, num_sources_per_shot,
                       num_receivers_per_shot,
                       propagator, prop_kwargs, device, dtype)


def run_scatter_2d(c=1500, dc=150, freq=25, dx=(5, 5), dt=0.0001,
                   nx=(50, 50),
                   num_shots=2, num_sources_per_shot=2,
                   num_receivers_per_shot=2,
                   propagator=None, prop_kwargs=None, device=None,
                   dtype=None):
    """Runs run_scatter with default parameters for 2D."""

    return run_scatter(c, dc, freq, dx, dt, nx,
                       num_shots, num_sources_per_shot,
                       num_receivers_per_shot,
                       propagator, prop_kwargs, device, dtype)


def run_scatter_3d(c=1500, dc=100, freq=25, dx=(5, 5, 5), dt=0.0001,
                   nx=(15, 5, 10),
                   num_shots=2, num_sources_per_shot=2,
                   num_receivers_per_shot=2,
                   propagator=None, prop_kwargs=None, device=None,
                   dtype=None):
    """Runs run_scatter with default parameters for 3D."""

    return run_scatter(c, dc, freq, dx, dt, nx,
                       num_shots, num_sources_per_shot,
                       num_receivers_per_shot,
                       propagator, prop_kwargs, device, dtype)


def run_born_scatter(c, dc, freq, dx, dt, nx,
                     num_shots, num_sources_per_shot,
                     num_receivers_per_shot,
                     propagator, prop_kwargs, device=None, dtype=None):
    """Create a point scatterer model, and the expected waveform at point,
       and the forward propagated wave.
    """
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.ones(*nx, device=device, dtype=dtype) * c

    nx = torch.Tensor(nx).long()
    dx = torch.Tensor(dx)

    nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx, dx)
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx, dx)
    x_p = _set_coords(1, 1, nx, dx, 'middle')[0, 0]
    scatter = torch.zeros_like(model)
    scatter[torch.split((x_p / dx).long(), 1)] = dc
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    if len(nx) == 1:
        scattered = scattered_1d
    elif len(nx) == 2:
        scattered = scattered_2d
    elif len(nx) == 3:
        scattered = scattered_3d
    else:
        raise ValueError("unsupported nx")

    expected = torch.zeros(nt, num_shots, num_receivers_per_shot, dtype=dtype)
    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            for receiver in range(num_receivers_per_shot):
                expected[:, shot, receiver] += \
                    scattered(x_r[shot, receiver], x_s[shot, source], x_p,
                              dx, dt, c, dc,
                              sources['amplitude'][:, shot, source]).to(dtype)

    actual = propagator(model, scatter, dx, dt, sources['amplitude'],
                        sources['locations'], x_r, prop_kwargs=prop_kwargs)

    return expected, actual


def run_born_scatter_1d(c=1500, dc=50, freq=25, dx=(5,), dt=0.0001, nx=(100,),
                        num_shots=2, num_sources_per_shot=2,
                        num_receivers_per_shot=2,
                        propagator=None, prop_kwargs=None, device=None,
                        dtype=None):
    """Runs run_born_scatter with default parameters for 1D."""

    return run_born_scatter(c, dc, freq, dx, dt, nx,
                            num_shots, num_sources_per_shot,
                            num_receivers_per_shot,
                            propagator, prop_kwargs, device, dtype)


def run_born_scatter_2d(c=1500, dc=150, freq=25, dx=(5, 5), dt=0.0001,
                        nx=(50, 50),
                        num_shots=2, num_sources_per_shot=2,
                        num_receivers_per_shot=2,
                        propagator=None, prop_kwargs=None, device=None,
                        dtype=None):
    """Runs run_born_scatter with default parameters for 2D."""

    return run_born_scatter(c, dc, freq, dx, dt, nx,
                            num_shots, num_sources_per_shot,
                            num_receivers_per_shot,
                            propagator, prop_kwargs, device, dtype)


def run_born_scatter_3d(c=1500, dc=100, freq=25, dx=(5, 5, 5), dt=0.0001,
                        nx=(15, 5, 10),
                        num_shots=2, num_sources_per_shot=2,
                        num_receivers_per_shot=2,
                        propagator=None, prop_kwargs=None, device=None,
                        dtype=None):
    """Runs run_born_scatter with default parameters for 3D."""

    return run_born_scatter(c, dc, freq, dx, dt, nx,
                            num_shots, num_sources_per_shot,
                            num_receivers_per_shot,
                            propagator, prop_kwargs, device, dtype)


def run_gradcheck(c, dc, freq, dx, dt, nx,
                  num_shots, num_sources_per_shot,
                  num_receivers_per_shot,
                  propagator, prop_kwargs,
                  device=None, dtype=None, atol=1e-5, eps=1e-6):
    """Create a point scatterer model, and the gradient."""
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        # double-precision not currently enabled for GPUs
        dtype = torch.float
        eps = 10
    model = (torch.ones(*nx, device=device, dtype=dtype) * c +
             torch.rand(*nx, device=device, dtype=dtype) * dc)

    nx = torch.Tensor(nx).long()
    dx = torch.Tensor(dx)

    nt = int((2 * torch.norm(nx.float() * dx) / c + 0.1 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx, dx)
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx, dx)
    sources = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)

    model.requires_grad_()
    sources['amplitude'].requires_grad_()

    pml_width = 3

    torch.autograd.gradcheck(propagator, (model, dx, dt, sources['amplitude'],
                                          sources['locations'], x_r,
                                          prop_kwargs, pml_width),
                             atol=atol, eps=eps)


def run_gradcheck_1d(c=1500, dc=100, freq=25, dx=(5,), dt=0.001, nx=(10,),
                     num_shots=2, num_sources_per_shot=2,
                     num_receivers_per_shot=2,
                     propagator=None, prop_kwargs=None,
                     device=None, dtype=torch.double):
    """Runs run_gradcheck with default parameters for 1D."""

    return run_gradcheck(c, dc, freq, dx, dt, nx,
                         num_shots, num_sources_per_shot,
                         num_receivers_per_shot,
                         propagator, prop_kwargs, device=device,
                         dtype=dtype)


def run_gradcheck_2d(c=1500, dc=100, freq=25, dx=(5, 5), dt=0.001,
                     nx=(4, 3),
                     num_shots=2, num_sources_per_shot=2,
                     num_receivers_per_shot=2,
                     propagator=None, prop_kwargs=None,
                     device=None, dtype=torch.double):
    """Runs run_gradcheck with default parameters for 2D."""

    return run_gradcheck(c, dc, freq, dx, dt, nx,
                         num_shots, num_sources_per_shot,
                         num_receivers_per_shot,
                         propagator, prop_kwargs, device=device,
                         dtype=dtype)


def run_gradcheck_3d(c=1500, dc=100, freq=25, dx=(5, 5, 5), dt=0.0005,
                     nx=(4, 3, 3),
                     num_shots=2, num_sources_per_shot=2,
                     num_receivers_per_shot=2,
                     propagator=None, prop_kwargs=None,
                     device=None, dtype=torch.double):
    """Runs run_gradcheck with default parameters for 3D."""

    # Reduced precision (atol=3e-4) required because of approximation in
    # 3D imaging condition
    return run_gradcheck(c, dc, freq, dx, dt, nx,
                         num_shots, num_sources_per_shot,
                         num_receivers_per_shot,
                         propagator, prop_kwargs, device=device,
                         dtype=dtype, atol=3e-4)


def run_born_gradcheck(c, dc, freq, dx, dt, nx,
                  num_shots, num_sources_per_shot,
                  num_receivers_per_shot,
                  propagator, prop_kwargs,
                  device=None, dtype=None, atol=1e-5, rtol=1e-3, eps=1e-6):
    """Create a point scatterer model, and the gradient."""
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        # double-precision not currently enabled for GPUs
        dtype = torch.float
        eps = 10
    model = torch.ones(*nx, device=device, dtype=dtype) * c
    scatter = torch.rand(*nx, device=device, dtype=dtype) * dc

    nx = torch.Tensor(nx).long()
    dx = torch.Tensor(dx)

    nt = int((2 * torch.norm(nx.float() * dx) / c + 0.1 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx, dx)
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx, dx)
    sources = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)

    scatter.requires_grad_()

    pml_width = 3

    torch.autograd.gradcheck(propagator, (model, scatter, dx, dt, sources['amplitude'],
                                          sources['locations'], x_r,
                                          prop_kwargs, pml_width),
                             atol=atol, eps=eps)


def run_born_gradcheck_1d(c=1500, dc=100, freq=25, dx=(5,), dt=0.001, nx=(10,),
                     num_shots=2, num_sources_per_shot=2,
                     num_receivers_per_shot=2,
                     propagator=None, prop_kwargs=None,
                     device=None, dtype=torch.double):
    """Runs run_gradcheck with default parameters for 1D."""

    return run_born_gradcheck(c, dc, freq, dx, dt, nx,
                         num_shots, num_sources_per_shot,
                         num_receivers_per_shot,
                         propagator, prop_kwargs, device=device,
                         dtype=dtype, atol=6e-6, rtol=1e-3, eps=1)


def run_born_gradcheck_2d(c=1500, dc=100, freq=25, dx=(5, 5), dt=0.001,
                     nx=(4, 3),
                     num_shots=2, num_sources_per_shot=2,
                     num_receivers_per_shot=2,
                     propagator=None, prop_kwargs=None,
                     device=None, dtype=torch.double):
    """Runs run_gradcheck with default parameters for 2D."""

    return run_born_gradcheck(c, dc, freq, dx, dt, nx,
                         num_shots, num_sources_per_shot,
                         num_receivers_per_shot,
                         propagator, prop_kwargs, device=device,
                         dtype=dtype, atol=1e-7, rtol=1e-4, eps=1)


def run_born_gradcheck_3d(c=1500, dc=100, freq=25, dx=(5, 5, 5), dt=0.0005,
                     nx=(4, 3, 3),
                     num_shots=2, num_sources_per_shot=2,
                     num_receivers_per_shot=2,
                     propagator=None, prop_kwargs=None,
                     device=None, dtype=torch.double):
    """Runs run_gradcheck with default parameters for 3D."""

    return run_born_gradcheck(c, dc, freq, dx, dt, nx,
                         num_shots, num_sources_per_shot,
                         num_receivers_per_shot,
                         propagator, prop_kwargs, device=device,
                         dtype=dtype, atol=1e-8, rtol=1e-5, eps=1)
