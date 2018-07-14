"""Create constant and point scatterer models."""
import torch
import numpy as np
import scipy.special
import scipy.integrate
from scipy.ndimage.interpolation import shift
from deepwave.scalar import Propagator
from wavelets import ricker


def test_direct_1d():
    """Test propagation in a constant 1D model."""
    expected, actual = model_direct_1d(propagator=scalarprop)
    diff = (expected - actual).numpy().ravel()
    assert np.linalg.norm(diff) < 41


def test_direct_2d():
    """Test propagation in a constant 2D model."""
    expected, actual = model_direct_2d(propagator=scalarprop)
    diff = (expected - actual).numpy().ravel()
    assert np.linalg.norm(diff) < 1.3


def test_direct_3d():
    """Test propagation in a constant 3D model."""
    expected, actual = model_direct_3d(propagator=scalarprop,
                                       dt=0.002,
                                       prop_kwargs={'pml_width': 20})
    diff = (expected - actual).numpy().ravel()
    assert np.linalg.norm(diff) < 0.25


def test_scatter_1d():
    """Test propagation in a 1D model with a point scatterer."""
    expected, actual = model_scatter_1d(propagator=scalarprop)
    diff = (expected - actual).numpy().ravel()
    assert np.linalg.norm(diff) < 1.8


def test_scatter_2d():
    """Test propagation in a 2D model with a point scatterer."""
    expected, actual = model_scatter_2d(propagator=scalarprop,
                                        prop_kwargs={'pml_width': 30})
    diff = (expected - actual).numpy().ravel()
    assert np.linalg.norm(diff) < 0.051


def test_scatter_3d():
    """Test propagation in a 3D model with a point scatterer."""
    expected, actual = model_scatter_3d(propagator=scalarprop,
                                        dt=0.002,
                                        prop_kwargs={'pml_width': 30})
    diff = (expected - actual).numpy().ravel()
    assert np.linalg.norm(diff) < 0.005


def test_grad_1d():
    """Test the gradient calculation in a 1D model with a point scatterer."""
    expected, actual = model_grad_1d(propagator=scalarprop)
    diff = (expected - actual).numpy().ravel()
    assert np.linalg.norm(diff) < 5e-5


def test_grad_2d():
    """Test the gradient calculation in a 2D model with a point scatterer."""
    expected, actual = model_grad_2d(propagator=scalarprop,
                                     prop_kwargs={'pml_width': 30})
    diff = (expected - actual).numpy().ravel()
    assert np.linalg.norm(diff[np.where(~np.isnan(diff))]) < 7e-8


def test_grad_3d():
    """Test the gradient calculation in a 3D model with a point scatterer."""
    expected, actual = model_grad_3d(propagator=scalarprop,
                                     dt=0.004,
                                     prop_kwargs={'pml_width': 20})
    diff = (expected - actual).numpy().ravel()
    assert np.linalg.norm(diff[np.where(~np.isnan(diff))]) < 2e-9


def scalarprop(model, dx, dt, sources, receiver_locations, grad=False,
               loss=False, forward_true=None, prop_kwargs=None):
    """Wraps the scalar propagator."""

    if prop_kwargs is None:
        prop_kwargs = {}

    prop = Propagator(model, dx, **prop_kwargs)
    receiver_amplitudes = prop.forward(sources['amplitude'],
                                       sources['locations'],
                                       receiver_locations, dt)

    if loss:
        l = torch.nn.MSELoss()(receiver_amplitudes, torch.Tensor(forward_true))
        return l.detach().item()
    if grad:
        l = torch.nn.MSELoss()(receiver_amplitudes, torch.Tensor(forward_true))
        l.backward()
        return model.grad.detach()
    return receiver_amplitudes.detach()


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


def grad_1d(nx, x_r, x_ss, x_p, dx, dt, c, dc, f):
    """Calculate the expected model gradient."""
    d = []
    for i, x_s in enumerate(x_ss):
        d.append(-flip(scattered_1d(x_r, x_s, x_p, dx, dt, c, dc, f[:, i]), 0))
    d = sum(d)
    grad = torch.zeros(torch.split(nx, 1))
    for x_idx in range(nx[0]):
        x = x_idx * dx[0]
        u_r = flip(direct_1d(x, x_r, dx, dt, c, d), 0)
        u_0 = []
        for i, x_s in enumerate(x_ss):
            u_0.append(direct_1d(x, x_s, dx, dt, c, f[:, i]))
        u_0 = sum(u_0)
        du_0dt2 = _second_deriv(u_0, dt)
        grad[x_idx] = 2 / len(f) * 2 / c**3 * torch.sum(u_r * du_0dt2)
    return grad


def grad_2d(nx, x_r, x_ss, x_p, dx, dt, c, dc, f):
    """Calculate the expected model gradient."""
    d = []
    for i, x_s in enumerate(x_ss):
        d.append(-flip(scattered_2d(x_r, x_s, x_p, dx, dt, c, dc, f[:, i]), 0))
    d = sum(d)
    grad = torch.zeros(torch.split(nx, 1))
    for z_idx in range(nx[0]):
        for y_idx in range(nx[1]):
            x = torch.Tensor([z_idx * dx[0], y_idx * dx[1]])
            u_r = flip(direct_2d_approx(x, x_r, dx, dt, c, d), 0)
            u_0 = []
            for i, x_s in enumerate(x_ss):
                u_0.append(direct_2d_approx(x, x_s, dx, dt, c, f[:, i]))
            u_0 = sum(u_0)
            du_0dt2 = _second_deriv(u_0, dt)
            grad[z_idx, y_idx] = (2 / len(f) * 2 / c**3 *
                                  torch.sum(u_r * du_0dt2))
    return grad


def grad_3d(nx, x_r, x_ss, x_p, dx, dt, c, dc, f):
    """Calculate the expected model gradient."""
    d = []
    for i, x_s in enumerate(x_ss):
        d.append(-flip(scattered_3d(x_r, x_s, x_p, dx, dt, c, dc, f[:, i]), 0))
    d = sum(d)
    grad = torch.zeros(torch.split(nx, 1))
    for z_idx in range(nx[0]):
        for y_idx in range(nx[1]):
            for x_idx in range(nx[2]):
                x = torch.Tensor([z_idx * dx[0], y_idx * dx[1], x_idx * dx[2]])
                u_r = flip(direct_3d(x, x_r, dx, dt, c, d), 0)
                u_0 = []
                for i, x_s in enumerate(x_ss):
                    u_0.append(direct_3d(x, x_s, dx, dt, c, f[:, i]))
                u_0 = sum(u_0)
                du_0dt2 = _second_deriv(u_0, dt)
                grad[z_idx, y_idx, x_idx] = (2 / len(f) * 2 / c**3 *
                                             torch.sum(u_r * du_0dt2))
    return grad


def flip(x, dim):
    """Reverses the order of element in one dimension.

    This is necessary as PyTorch does not support negative steps, so [::-1]
    is not allowed.
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def _second_deriv(arr, dt):
    """Calculate the second derivative."""
    return torch.Tensor(np.gradient(np.gradient(arr)) / dt**2)


def _set_sources(x_s, freq, dt, nt):
    """Create sources with amplitudes that have randomly shifted start times.
    """
    num_shots, num_sources_per_shot = x_s.shape[:2]
    sources = {}
    sources['amplitude'] = torch.zeros(nt, num_shots, num_sources_per_shot)

    sources['locations'] = x_s

    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            peak_time = 0.05 + torch.rand(1).item() * 0.3
            sources['amplitude'][:, shot, source] = \
                ricker(freq, nt, dt, peak_time)
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
        coords[..., 0] = nx[0].float() - coords[..., 0]
    elif location == 'middle':
        coords[..., 0] += int(nx[0] / 2)
    else:
        raise ValueError("unsupported location")

    for dim in range(1, ndim):
        coords[..., dim] = torch.round(nx[dim].float() / 2)

    return coords * dx


def model_direct(c, freq, dx, dt, nx,
                 num_shots, num_sources_per_shot,
                 num_receivers_per_shot,
                 propagator, prop_kwargs):
    """Create a constant model, and the expected waveform at point,
       and the forward propagated wave.
    """
    torch.manual_seed(1)
    model = torch.ones(1, *nx) * c

    nx = torch.Tensor(nx).long()
    dx = torch.Tensor(dx)

    nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx, dx)
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx, dx, 'bottom')
    sources = _set_sources(x_s, freq, dt, nt)

    if len(nx) == 1:
        direct = direct_1d
    elif len(nx) == 2:
        direct = direct_2d_approx
    elif len(nx) == 3:
        direct = direct_3d
    else:
        raise ValueError("unsupported nx")

    expected = torch.zeros(nt, num_shots, num_receivers_per_shot)
    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            for receiver in range(num_receivers_per_shot):
                expected[:, shot, receiver] += \
                    direct(x_r[shot, receiver], x_s[shot, source],
                           dx, dt, c,
                           sources['amplitude'][:, shot, source])

    actual = propagator(model, dx, dt, sources, x_r,
                        prop_kwargs=prop_kwargs)

    return expected, actual


def model_direct_1d(c=1500, freq=25, dx=(5,), dt=0.0001, nx=(80,),
                    num_shots=2, num_sources_per_shot=2,
                    num_receivers_per_shot=2,
                    propagator=None, prop_kwargs=None):
    """Runs model_direct with default parameters for 1D."""

    return model_direct(c, freq, dx, dt, nx,
                        num_shots, num_sources_per_shot,
                        num_receivers_per_shot,
                        propagator, prop_kwargs)


def model_direct_2d(c=1500, freq=25, dx=(5, 5), dt=0.0001, nx=(50, 50),
                    num_shots=2, num_sources_per_shot=2,
                    num_receivers_per_shot=2,
                    propagator=None, prop_kwargs=None):
    """Runs model_direct with default parameters for 2D."""

    return model_direct(c, freq, dx, dt, nx,
                        num_shots, num_sources_per_shot,
                        num_receivers_per_shot,
                        propagator, prop_kwargs)


def model_direct_3d(c=1500, freq=25, dx=(5, 5, 5), dt=0.0001, nx=(20, 10, 20),
                    num_shots=2, num_sources_per_shot=2,
                    num_receivers_per_shot=2,
                    propagator=None, prop_kwargs=None):
    """Runs model_direct with default parameters for 3D."""

    return model_direct(c, freq, dx, dt, nx,
                        num_shots, num_sources_per_shot,
                        num_receivers_per_shot,
                        propagator, prop_kwargs)


def model_scatter(c, dc, freq, dx, dt, nx,
                  num_shots, num_sources_per_shot,
                  num_receivers_per_shot,
                  propagator, prop_kwargs):
    """Create a point scatterer model, and the expected waveform at point,
       and the forward propagated wave.
    """
    torch.manual_seed(1)
    model = torch.ones(1, *nx) * c
    model_const = model.clone()

    nx = torch.Tensor(nx).long()
    dx = torch.Tensor(dx)

    nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx, dx)
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx, dx)
    x_p = _set_coords(1, 1, nx, dx, 'middle')[0, 0]
    model[(0,) + torch.split((x_p / dx).long(), 1)] += dc
    sources = _set_sources(x_s, freq, dt, nt)

    if len(nx) == 1:
        scattered = scattered_1d
    elif len(nx) == 2:
        scattered = scattered_2d
    elif len(nx) == 3:
        scattered = scattered_3d
    else:
        raise ValueError("unsupported nx")

    expected = torch.zeros(nt, num_shots, num_receivers_per_shot)
    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            for receiver in range(num_receivers_per_shot):
                expected[:, shot, receiver] += \
                    scattered(x_r[shot, receiver], x_s[shot, source], x_p,
                              dx, dt, c, dc,
                              sources['amplitude'][:, shot, source])

    y_const = propagator(model_const, dx, dt, sources, x_r,
                         prop_kwargs=prop_kwargs)
    y = propagator(model, dx, dt, sources, x_r, prop_kwargs=prop_kwargs)

    actual = y - y_const

    return expected, actual


def model_scatter_1d(c=1500, dc=50, freq=25, dx=(5,), dt=0.0001, nx=(100,),
                     num_shots=2, num_sources_per_shot=2,
                     num_receivers_per_shot=2,
                     propagator=None, prop_kwargs=None):
    """Runs model_scatter with default parameters for 1D."""

    return model_scatter(c, dc, freq, dx, dt, nx,
                         num_shots, num_sources_per_shot,
                         num_receivers_per_shot,
                         propagator, prop_kwargs)


def model_scatter_2d(c=1500, dc=150, freq=25, dx=(5, 5), dt=0.0001,
                     nx=(50, 50),
                     num_shots=2, num_sources_per_shot=2,
                     num_receivers_per_shot=2,
                     propagator=None, prop_kwargs=None):
    """Runs model_scatter with default parameters for 2D."""

    return model_scatter(c, dc, freq, dx, dt, nx,
                         num_shots, num_sources_per_shot,
                         num_receivers_per_shot,
                         propagator, prop_kwargs)


def model_scatter_3d(c=1500, dc=100, freq=25, dx=(5, 5, 5), dt=0.0001,
                     nx=(15, 5, 10),
                     num_shots=2, num_sources_per_shot=2,
                     num_receivers_per_shot=2,
                     propagator=None, prop_kwargs=None):
    """Runs model_scatter with default parameters for 3D."""

    return model_scatter(c, dc, freq, dx, dt, nx,
                         num_shots, num_sources_per_shot,
                         num_receivers_per_shot,
                         propagator, prop_kwargs)


def model_grad(c, dc, freq, dx, dt, nx,
               num_shots, num_sources_per_shot,
               num_receivers_per_shot,
               propagator, prop_kwargs, calc_true_grad=False):
    """Create a point scatterer model, and the gradient."""
    torch.manual_seed(1)
    model_init = torch.ones(1, *nx) * c
    model_true = model_init.clone()
    model_init.requires_grad_()

    nx = torch.Tensor(nx).long()
    dx = torch.Tensor(dx)

    nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx, dx)
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx, dx)
    x_p = _set_coords(1, 1, nx, dx, 'middle')[0, 0]
    model_true[(0,) + torch.split((x_p / dx).long(), 1)] += dc
    sources = _set_sources(x_s, freq, dt, nt)

    if len(nx) == 1:
        grad = grad_1d
    elif len(nx) == 2:
        grad = grad_2d
    elif len(nx) == 3:
        grad = grad_3d
    else:
        raise ValueError("unsupported nx")

    expected = torch.zeros(1, *nx)
    for shot in range(num_shots):
        for receiver in range(num_receivers_per_shot):
            expected[0] += \
                grad(nx, x_r[shot, receiver], x_s[shot], x_p,
                     dx, dt, c, dc,
                     sources['amplitude'][:, shot])
    expected /= (num_shots * num_receivers_per_shot)

    forward_true = propagator(model_true, dx, dt, sources, x_r,
                              prop_kwargs=prop_kwargs)
    actual = propagator(model_init, dx, dt, sources, x_r, grad=True,
                        forward_true=forward_true, prop_kwargs=prop_kwargs)

    if calc_true_grad:
        true_grad = torch.zeros(1, *nx)
        for idx, _ in np.ndenumerate(model_init):
            tmp_model = model_init.clone()
            tmp_model[(0, ) + idx] += dc
            lossp = propagator(tmp_model, dx, dt, sources, x_r,
                               loss=True, forward_true=forward_true,
                               prop_kwargs=prop_kwargs)
            tmp_model = model_init.clone()
            tmp_model[(0, ) + idx] -= dc
            lossm = propagator(tmp_model, dx, dt, sources, x_r,
                               loss=True, forward_true=forward_true,
                               prop_kwargs=prop_kwargs)
            true_grad[(0, ) + idx] = (lossp.double() -
                                      lossm.double()) / (2 * dc)

        return expected, actual, true_grad

    return expected, actual


def model_grad_1d(c=1500, dc=100, freq=25, dx=(5,), dt=0.0001, nx=(100,),
                  num_shots=2, num_sources_per_shot=2,
                  num_receivers_per_shot=2,
                  propagator=None, prop_kwargs=None,
                  calc_true_grad=False):
    """Runs model_grad with default parameters for 1D."""

    return model_grad(c, dc, freq, dx, dt, nx,
                      num_shots, num_sources_per_shot,
                      num_receivers_per_shot,
                      propagator, prop_kwargs, calc_true_grad)


def model_grad_2d(c=1500, dc=100, freq=25, dx=(5, 5), dt=0.0001, nx=(20, 20),
                  num_shots=2, num_sources_per_shot=2,
                  num_receivers_per_shot=2,
                  propagator=None, prop_kwargs=None,
                  calc_true_grad=False):
    """Runs model_grad with default parameters for 2D."""

    return model_grad(c, dc, freq, dx, dt, nx,
                      num_shots, num_sources_per_shot,
                      num_receivers_per_shot,
                      propagator, prop_kwargs, calc_true_grad)


def model_grad_3d(c=1500, dc=200, freq=25, dx=(5, 5, 5), dt=0.0001,
                  nx=(15, 5, 10),
                  num_shots=2, num_sources_per_shot=2,
                  num_receivers_per_shot=2,
                  propagator=None, prop_kwargs=None,
                  calc_true_grad=False):
    """Runs model_grad with default parameters for 3D."""

    return model_grad(c, dc, freq, dx, dt, nx,
                      num_shots, num_sources_per_shot,
                      num_receivers_per_shot,
                      propagator, prop_kwargs, calc_true_grad)
