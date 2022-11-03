import math
import torch
import scipy.special
from deepwave import Scalar, scalar
from deepwave.wavelets import ricker
from deepwave.common import cfl_condition, upsample, downsample


def scalarprop(model, dx, dt, source_amplitudes, source_locations,
               receiver_locations, prop_kwargs=None, pml_width=None,
               survey_pad=None, origin=None, wavefield_0=None,
               wavefield_m1=None,
               psiy_m1=None, psix_m1=None,
               zetay_m1=None, zetax_m1=None, nt=None,
               model_gradient_sampling_interval=1,
               functional=True):
    """Wraps the scalar propagator."""

    if prop_kwargs is None:
        prop_kwargs = {}
    # For consistency when actual max speed changes
    prop_kwargs['max_vel'] = 2000

    # Workaround for gradcheck not accepting prop_kwargs dictionary
    if pml_width is not None:
        prop_kwargs['pml_width'] = pml_width
    prop_kwargs['survey_pad'] = survey_pad
    prop_kwargs['origin'] = origin

    device = model.device
    if source_amplitudes is not None:
        source_amplitudes = source_amplitudes.to(device)
        source_locations = source_locations.to(device)
    if receiver_locations is not None:
        receiver_locations = receiver_locations.to(device)

    if functional:
        return scalar(model, dx, dt,
                      source_amplitudes=source_amplitudes,
                      source_locations=source_locations,
                      receiver_locations=receiver_locations,
                      wavefield_0=wavefield_0, wavefield_m1=wavefield_m1,
                      psiy_m1=psiy_m1, psix_m1=psix_m1,
                      zetay_m1=zetay_m1, zetax_m1=zetax_m1, nt=nt,
                      model_gradient_sampling_interval=
                      model_gradient_sampling_interval,
                      **prop_kwargs)

    prop = Scalar(model, dx)
    return prop(dt, source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                wavefield_0=wavefield_0, wavefield_m1=wavefield_m1,
                psiy_m1=psiy_m1, psix_m1=psix_m1,
                zetay_m1=zetay_m1, zetax_m1=zetax_m1, nt=nt,
                model_gradient_sampling_interval=
                model_gradient_sampling_interval,
                **prop_kwargs)


def scalarpropchained(model, dx, dt, source_amplitudes, source_locations,
                      receiver_locations, prop_kwargs=None, pml_width=None,
                      survey_pad=None, origin=None, wavefield_0=None,
                      wavefield_m1=None,
                      psiy_m1=None, psix_m1=None,
                      zetay_m1=None, zetax_m1=None, nt=None,
                      model_gradient_sampling_interval=1,
                      functional=True, n_chained=2):
    """Wraps multiple scalar propagators chained sequentially."""
    if prop_kwargs is None:
        prop_kwargs = {}
    # For consistency when actual max speed changes
    prop_kwargs['max_vel'] = 2000

    # Workaround for gradcheck not accepting prop_kwargs dictionary
    if pml_width is not None:
        prop_kwargs['pml_width'] = pml_width
    prop_kwargs['survey_pad'] = survey_pad
    prop_kwargs['origin'] = origin

    device = model.device
    if source_amplitudes is not None:
        source_amplitudes = source_amplitudes.to(device)
        source_locations = source_locations.to(device)
    if receiver_locations is not None:
        receiver_locations = receiver_locations.to(device)

    max_vel = 2000
    dt, step_ratio = cfl_condition(dx[0], dx[1], dt, max_vel)
    if source_amplitudes is not None:
        source_amplitudes = upsample(source_amplitudes, step_ratio)
        nt_per_segment = ((((source_amplitudes.shape[-1] + n_chained - 1) //
                            n_chained + step_ratio - 1) // step_ratio)
                          * step_ratio)
    else:
        nt *= step_ratio
        nt_per_segment = (((nt + n_chained - 1) // n_chained + step_ratio - 1)
                          // step_ratio) * step_ratio

    wfc = wavefield_0
    wfp = wavefield_m1
    psiy = psiy_m1
    psix = psix_m1
    zetay = zetay_m1
    zetax = zetax_m1

    if receiver_locations is not None:
        if source_amplitudes is not None:
            receiver_amplitudes = torch.zeros(receiver_locations.shape[0],
                                              receiver_locations.shape[1],
                                              source_amplitudes.shape[-1],
                                              dtype=model.dtype,
                                              device=model.device)
        else:
            receiver_amplitudes = torch.zeros(receiver_locations.shape[0],
                                              receiver_locations.shape[1],
                                              nt, dtype=model.dtype,
                                              device=model.device)

    for segment_idx in range(n_chained):
        if source_amplitudes is not None:
            segment_source_amplitudes = \
                source_amplitudes[...,
                                  nt_per_segment * segment_idx:
                                  min(nt_per_segment * (segment_idx+1),
                                      source_amplitudes.shape[-1])]
            segment_nt = None
        else:
            segment_source_amplitudes = None
            segment_nt = (min(nt_per_segment * (segment_idx+1),
                              source_amplitudes.shape[-1]) -
                          nt_per_segment * segment_idx)
        wfc, wfp, psiy, psix, zetay, zetax, segment_receiver_amplitudes = \
            scalar(model, dx, dt,
                   source_amplitudes=segment_source_amplitudes,
                   source_locations=source_locations,
                   receiver_locations=receiver_locations,
                   wavefield_0=wfc, wavefield_m1=wfp,
                   psiy_m1=psiy, psix_m1=psix,
                   zetay_m1=zetay, zetax_m1=zetax,
                   nt=segment_nt,
                   model_gradient_sampling_interval=step_ratio,
                   **prop_kwargs)
        if receiver_locations is not None:
            receiver_amplitudes[...,
                                nt_per_segment * segment_idx:
                                min(nt_per_segment * (segment_idx+1),
                                    receiver_amplitudes.shape[-1])] = \
                                    segment_receiver_amplitudes

    if receiver_locations is not None:
        receiver_amplitudes = downsample(receiver_amplitudes, step_ratio)

    return wfc, wfp, psiy, psix, zetay, zetax, receiver_amplitudes


def test_pml_width_list():
    """Verify that PML width supplied as int and list give same result."""
    _, actual_int = run_direct_2d(propagator=scalarprop,
                                  prop_kwargs={'pml_width': 20})
    _, actual_list = run_direct_2d(propagator=scalarprop,
                                   prop_kwargs={'pml_width':
                                                [20, 20, 20, 20]})
    assert torch.allclose(actual_int, actual_list)


def test_direct_2d():
    """Test propagation in a constant 2D model."""
    expected, actual = run_direct_2d(propagator=scalarprop)
    diff = (expected - actual.cpu()).flatten()
    assert diff.norm().item() < 1.48


def test_direct_2d_2nd_order():
    """Test propagation with a 2nd order accurate propagator."""
    expected, actual = run_direct_2d(propagator=scalarprop,
                                     prop_kwargs={'accuracy': 2})
    diff = (expected - actual.cpu()).flatten()
    assert diff.norm().item() < 16


def test_direct_2d_6th_order():
    """Test propagation with a 6th order accurate propagator."""
    expected, actual = run_direct_2d(propagator=scalarprop,
                                     prop_kwargs={'accuracy': 6})
    diff = (expected - actual.cpu()).flatten()
    assert diff.norm().item() < 0.22


def test_direct_2d_8th_order():
    """Test propagation with a 8th order accurate propagator."""
    expected, actual = run_direct_2d(propagator=scalarprop,
                                     prop_kwargs={'accuracy': 8})
    diff = (expected - actual.cpu()).flatten()
    assert diff.norm().item() < 0.048


def test_wavefield_decays():
    """Test that the PML causes the wavefield amplitude to decay."""
    out = run_forward_2d(propagator=scalarprop, nt=2000)
    for outi in out[:-1]:
        assert outi.norm() < 1e-5


def test_forward_cpu_gpu_match():
    """Test propagation on CPU and GPU produce the same result."""
    if torch.cuda.is_available():
        actual_cpu = run_forward_2d(propagator=scalarprop,
                                    device=torch.device("cpu"))
        actual_gpu = run_forward_2d(propagator=scalarprop,
                                    device=torch.device("cuda"))
        for cpui, gpui in zip(actual_cpu[:-1], actual_gpu[:-1]):
            assert torch.allclose(cpui, gpui.cpu(), atol=2e-6)
        cpui = actual_cpu[-1]
        gpui = actual_cpu[-1]
        assert torch.allclose(cpui, gpui.cpu(), atol=5e-5)


def test_direct_2d_module():
    """Test propagation with the Module interface."""
    expected, actual = run_direct_2d(propagator=scalarprop,
                                     functional=False)
    diff = (expected - actual.cpu()).flatten()
    assert diff.norm().item() < 1.48


def test_scatter_2d():
    """Test propagation in a 2D model with a point scatterer."""
    expected, actual = run_scatter_2d(propagator=scalarprop,
                                      dt=0.001,
                                      prop_kwargs={'pml_width': 30})
    diff = (expected - actual.cpu()).flatten()
    assert diff.norm().item() < 0.008


def test_gradcheck_2d():
    """Test gradcheck in a 2D model."""
    run_gradcheck_2d(propagator=scalarprop)


def test_gradcheck_2d_2nd_order():
    """Test gradcheck with a 2nd order accurate propagator."""
    run_gradcheck_2d(propagator=scalarprop,
                     prop_kwargs={'accuracy': 2}, atol=1.2e-8)


def test_gradcheck_2d_6th_order():
    """Test gradcheck with a 6th order accurate propagator."""
    run_gradcheck_2d(propagator=scalarprop,
                     prop_kwargs={'accuracy': 6})


def test_gradcheck_2d_8th_order():
    """Test gradcheck with a 8th order accurate propagator."""
    run_gradcheck_2d(propagator=scalarprop,
                     prop_kwargs={'accuracy': 8})


def test_gradcheck_2d_cfl():
    """Test gradcheck with a timestep greater than the CFL limit."""
    run_gradcheck_2d(propagator=scalarprop, dt=0.002, atol=1.5e-6)


def test_gradcheck_2d_odd_timesteps():
    """Test gradcheck with one more timestep."""
    run_gradcheck_2d(propagator=scalarprop, nt_add=1)


def test_gradcheck_2d_one_shot():
    """Test gradcheck with one shot."""
    run_gradcheck_2d(propagator=scalarprop, num_shots=1)


def test_gradcheck_2d_no_sources():
    """Test gradcheck with no sources."""
    run_gradcheck_2d(propagator=scalarprop, num_sources_per_shot=0)


def test_gradcheck_2d_no_receivers():
    """Test gradcheck with no receivers."""
    run_gradcheck_2d(propagator=scalarprop, num_receivers_per_shot=0)


def test_gradcheck_2d_survey_pad():
    """Test gradcheck with survey_pad."""
    run_gradcheck_2d(propagator=scalarprop, survey_pad=0,
                     provide_wavefields=False)


def test_gradcheck_2d_partial_wavefields():
    """Test gradcheck with wavefields that do not cover the model."""
    run_gradcheck_2d(propagator=scalarprop, origin=(0, 2),
                     wavefield_size=(4+6, 1+6))


def test_gradcheck_2d_chained():
    """Test gradcheck when two propagators are chained."""
    run_gradcheck_2d(propagator=scalarpropchained)


def test_gradcheck_2d_negative():
    """Test gradcheck with negative velocity."""
    run_gradcheck_2d(c=-1500, propagator=scalarprop)


def test_gradcheck_2d_zero():
    """Test gradcheck with zero velocity."""
    run_gradcheck_2d(c=0, dc=0, propagator=scalarprop)


def test_gradcheck_2d_different_pml():
    """Test gradcheck with different pml widths."""
    run_gradcheck_2d(propagator=scalarprop, pml_width=[0, 1, 5, 10],
                     atol=5e-8)


def test_gradcheck_2d_no_pml():
    """Test gradcheck with no pml."""
    run_gradcheck_2d(propagator=scalarprop, pml_width=0)


def test_gradcheck_2d_different_dx():
    """Test gradcheck with different dx values."""
    run_gradcheck_2d(propagator=scalarprop, dx=(4, 5), dt=0.0005, atol=2e-8)


def test_gradcheck_2d_single_cell():
    """Test gradcheck with a single model cell."""
    run_gradcheck_2d(propagator=scalarprop, nx=(1, 1),
                     num_shots=1,
                     num_sources_per_shot=1,
                     num_receivers_per_shot=1)


def test_gradcheck_2d_big():
    """Test gradcheck with a big model."""
    run_gradcheck_2d(propagator=scalarprop, nx=(5+2*(3+3*2), 4+2*(3+3*2)),
                     atol=2e-8)


def test_negative_vel(c=1500, dc=100, freq=25, dx=(5, 5), dt=0.005,
                      nx=(50, 50), num_shots=2, num_sources_per_shot=2,
                      num_receivers_per_shot=2,
                      propagator=scalarprop, prop_kwargs=None, device=None,
                      dtype=None):
    """Test propagation with a zero or negative velocity or dt"""
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nx = torch.Tensor(nx).long()
    dx = torch.Tensor(dx)

    nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx)
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx, 'bottom')
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    # Positive velocity
    model = (torch.ones(*nx, device=device, dtype=dtype) * c +
             torch.randn(*nx, device=device, dtype=dtype) * dc)
    out_positive = propagator(model, dx, dt, sources['amplitude'],
                              sources['locations'], x_r,
                              prop_kwargs=prop_kwargs)

    # Negative velocity
    out = propagator(-model, dx, dt, sources['amplitude'],
                     sources['locations'], x_r,
                     prop_kwargs=prop_kwargs)
    assert torch.allclose(out_positive[0], out[0])
    assert torch.allclose(out_positive[-1], out[-1])

    # Negative dt
    out = propagator(model, dx, -dt, sources['amplitude'],
                     sources['locations'], x_r,
                     prop_kwargs=prop_kwargs)
    assert torch.allclose(out_positive[0], out[0])
    assert torch.allclose(out_positive[-1], out[-1])

    # Zero velocity
    out = propagator(torch.zeros_like(model), dx, -dt, sources['amplitude'],
                     sources['locations'], x_r,
                     prop_kwargs=prop_kwargs)
    assert torch.allclose(out[0], torch.zeros_like(out[0]))


def test_gradcheck_only_v_2d():
    """Test gradcheck with only v requiring gradient."""
    run_gradcheck_2d(propagator=scalarprop,
                     source_requires_grad=False,
                     wavefield_0_requires_grad=False,
                     wavefield_m1_requires_grad=False,
                     psiy_m1_requires_grad=False,
                     psix_m1_requires_grad=False,
                     zetay_m1_requires_grad=False,
                     zetax_m1_requires_grad=False,
                     )


def test_gradcheck_only_source_2d():
    """Test gradcheck with only source requiring gradient."""
    run_gradcheck_2d(propagator=scalarprop,
                     v_requires_grad=False,
                     wavefield_0_requires_grad=False,
                     wavefield_m1_requires_grad=False,
                     psiy_m1_requires_grad=False,
                     psix_m1_requires_grad=False,
                     zetay_m1_requires_grad=False,
                     zetax_m1_requires_grad=False,
                     )


def test_gradcheck_only_wavefield_0_2d():
    """Test gradcheck with only wavefield_0 requiring gradient."""
    run_gradcheck_2d(propagator=scalarprop,
                     v_requires_grad=False,
                     source_requires_grad=False,
                     wavefield_m1_requires_grad=False,
                     psiy_m1_requires_grad=False,
                     psix_m1_requires_grad=False,
                     zetay_m1_requires_grad=False,
                     zetax_m1_requires_grad=False,
                     )


def test_jit():
    """Test that the propagator can be JIT compiled"""
    torch.jit.script(Scalar(torch.ones(1, 1), 5.0))(
        0.001, source_amplitudes=torch.ones(1, 1, 1),
        source_locations=torch.zeros(1, 1, 2)
    )
    torch.jit.script(scalar)(torch.ones(1, 1), 5.0, 0.001,
                             source_amplitudes=torch.ones(1, 1, 1),
                             source_locations=torch.zeros(1, 1, 2).long())


def direct_2d_approx(x, x_s, dx, dt, c, f):
    """Use an approximation of the 2D Green's function to calculate the
    wavefield at a given location due to the given source.
    """
    r = torch.norm(x*dx - x_s*dx).item()
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


def _set_sources(x_s, freq, dt, nt, dtype=None, dpeak_time=0.3):
    """Create sources with amplitudes that have randomly shifted start times.
    """
    num_shots, num_sources_per_shot = x_s.shape[:2]
    sources = {}
    sources['amplitude'] = torch.zeros(num_shots, num_sources_per_shot, nt,
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
        coords[..., 0] = (nx[0] - 1).float() - coords[..., 0]
    elif location == 'middle':
        coords[..., 0] += int(nx[0] / 2)
    else:
        raise ValueError("unsupported location")

    for dim in range(1, ndim):
        coords[..., dim] = torch.round(nx[dim].float() / 2)

    return coords.long()


def run_direct(c, freq, dx, dt, nx,
               num_shots, num_sources_per_shot,
               num_receivers_per_shot,
               propagator, prop_kwargs, device=None,
               dtype=None, **kwargs):
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
    x_s = _set_coords(num_shots, num_sources_per_shot, nx)
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx, 'bottom')
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    if len(nx) == 2:
        direct = direct_2d_approx
    else:
        raise ValueError("unsupported nx")

    expected = torch.zeros(num_shots, num_receivers_per_shot, nt, dtype=dtype)
    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            for receiver in range(num_receivers_per_shot):
                expected[shot, receiver, :] += \
                    direct(x_r[shot, receiver], x_s[shot, source],
                           dx, dt, c,
                           -sources['amplitude'][shot, source, :]).to(dtype)

    actual = propagator(model, dx, dt, sources['amplitude'],
                        sources['locations'], x_r,
                        prop_kwargs=prop_kwargs, **kwargs)[6]

    return expected, actual


def run_direct_2d(c=1500, freq=25, dx=(5, 5), dt=0.0001, nx=(50, 50),
                  num_shots=2, num_sources_per_shot=2,
                  num_receivers_per_shot=2,
                  propagator=None, prop_kwargs=None, device=None,
                  dtype=None, **kwargs):
    """Runs run_direct with default parameters for 2D."""

    return run_direct(c, freq, dx, dt, nx,
                      num_shots, num_sources_per_shot,
                      num_receivers_per_shot,
                      propagator, prop_kwargs, device, dtype, **kwargs)


def run_forward(c, freq, dx, dt, nx,
                num_shots, num_sources_per_shot,
                num_receivers_per_shot,
                propagator, prop_kwargs, device=None,
                dtype=None, dc=100, nt=None, **kwargs):
    """Create a random model and forward propagate.
    """
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (torch.ones(*nx, device=device, dtype=dtype) * c +
             torch.randn(*nx, dtype=dtype).to(device) * dc)

    nx = torch.Tensor(nx).long()
    dx = torch.Tensor(dx)

    if nt is None:
        nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx)
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx, 'bottom')
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    return propagator(model, dx, dt, sources['amplitude'],
                      sources['locations'], x_r,
                      prop_kwargs=prop_kwargs, **kwargs)


def run_forward_2d(c=1500, freq=25, dx=(5, 5), dt=0.004, nx=(50, 50),
                   num_shots=2, num_sources_per_shot=2,
                   num_receivers_per_shot=2,
                   propagator=None, prop_kwargs=None, device=None,
                   dtype=None, **kwargs):
    """Runs run_forward with default parameters for 2D."""

    return run_forward(c, freq, dx, dt, nx,
                       num_shots, num_sources_per_shot,
                       num_receivers_per_shot,
                       propagator, prop_kwargs, device=device,
                       dtype=dtype, **kwargs)


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
    x_s = _set_coords(num_shots, num_sources_per_shot, nx)
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx)
    x_p = _set_coords(1, 1, nx, 'middle')[0, 0]
    model[torch.split((x_p).long(), 1)] += dc
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    if len(nx) == 2:
        scattered = scattered_2d
    else:
        raise ValueError("unsupported nx")

    expected = torch.zeros(num_shots, num_receivers_per_shot, nt, dtype=dtype)
    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            for receiver in range(num_receivers_per_shot):
                expected[shot, receiver, :] += \
                    scattered(x_r[shot, receiver], x_s[shot, source], x_p,
                              dx, dt, c, dc,
                              -sources['amplitude'][shot, source, :]).to(dtype)

    y_const = propagator(model_const, dx, dt, sources['amplitude'],
                         sources['locations'], x_r,
                         prop_kwargs=prop_kwargs)[6]
    y = propagator(model, dx, dt, sources['amplitude'], sources['locations'],
                   x_r, prop_kwargs=prop_kwargs)[6]

    actual = y - y_const

    return expected, actual


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


def run_gradcheck(c, dc, freq, dx, dt, nx,
                  num_shots, num_sources_per_shot,
                  num_receivers_per_shot,
                  propagator, prop_kwargs,
                  pml_width=3, survey_pad=None,
                  origin=None, wavefield_size=None,
                  device=None, dtype=None, v_requires_grad=True,
                  source_requires_grad=True,
                  provide_wavefields=True,
                  wavefield_0_requires_grad=True,
                  wavefield_m1_requires_grad=True,
                  psiy_m1_requires_grad=True,
                  psix_m1_requires_grad=True,
                  zetay_m1_requires_grad=True,
                  zetax_m1_requires_grad=True,
                  atol=1e-8, rtol=1e-5, nt_add=0):
    """Run PyTorch's gradcheck to test the gradient."""
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (torch.ones(*nx, device=device, dtype=dtype) * c +
             torch.rand(*nx, device=device, dtype=dtype) * dc)

    nx = torch.Tensor(nx).long()
    dx = torch.Tensor(dx)

    if c != 0:
        nt = int((2 * torch.norm(nx.float() * dx) / abs(c) + 0.1 + 2 / freq)
                 / dt)
    else:
        nt = int((2 * torch.norm(nx.float() * dx) / 1500 + 0.1 + 2 / freq)
                 / dt)
    nt += nt_add
    if num_sources_per_shot > 0:
        x_s = _set_coords(num_shots, num_sources_per_shot, nx)
        sources = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)
        sources['amplitude'].requires_grad_(source_requires_grad)
        nt = None
    else:
        sources = {'amplitude': None, 'locations': None}
    if num_receivers_per_shot > 0:
        x_r = _set_coords(num_shots, num_receivers_per_shot, nx)
    else:
        x_r = None
    if isinstance(pml_width, int):
        pml_width = [pml_width for _ in range(4)]
    if provide_wavefields:
        if wavefield_size is None:
            wavefield_size = (nx[0] + pml_width[0] + pml_width[1],
                              nx[1] + pml_width[2] + pml_width[3])
        wavefield_0 = torch.zeros(num_shots, *wavefield_size, dtype=dtype,
                                  device=device)
        wavefield_m1 = torch.zeros_like(wavefield_0)
        psiy_m1 = torch.zeros_like(wavefield_0)
        psix_m1 = torch.zeros_like(wavefield_0)
        zetay_m1 = torch.zeros_like(wavefield_0)
        zetax_m1 = torch.zeros_like(wavefield_0)
        wavefield_0.requires_grad_(wavefield_0_requires_grad)
        wavefield_m1.requires_grad_(wavefield_m1_requires_grad)
        psiy_m1.requires_grad_(psiy_m1_requires_grad)
        psix_m1.requires_grad_(psix_m1_requires_grad)
        zetay_m1.requires_grad_(zetay_m1_requires_grad)
        zetax_m1.requires_grad_(zetax_m1_requires_grad)
    else:
        wavefield_0 = None
        wavefield_m1 = None
        psiy_m1 = None
        psix_m1 = None
        zetay_m1 = None
        zetax_m1 = None

    model.requires_grad_(v_requires_grad)

    torch.autograd.gradcheck(propagator, (model, dx, dt, sources['amplitude'],
                                          sources['locations'], x_r,
                                          prop_kwargs, pml_width, survey_pad,
                                          origin,
                                          wavefield_0, wavefield_m1,
                                          psiy_m1, psix_m1, zetay_m1,
                                          zetax_m1, nt, 1, True),
                             nondet_tol=1e-3, check_grad_dtypes=True,
                             atol=atol, rtol=rtol)


def run_gradcheck_2d(c=1500, dc=100, freq=25, dx=(5, 5), dt=0.001,
                     nx=(4, 3),
                     num_shots=2, num_sources_per_shot=2,
                     num_receivers_per_shot=2,
                     propagator=None, prop_kwargs=None,
                     pml_width=3,
                     survey_pad=None,
                     device=None, dtype=torch.double, **kwargs):
    """Runs run_gradcheck with default parameters for 2D."""

    return run_gradcheck(c, dc, freq, dx, dt, nx,
                         num_shots, num_sources_per_shot,
                         num_receivers_per_shot,
                         propagator, prop_kwargs, pml_width=pml_width,
                         survey_pad=survey_pad,
                         device=device, dtype=dtype, **kwargs)
