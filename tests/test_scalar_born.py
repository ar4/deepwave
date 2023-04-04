import torch
from deepwave import ScalarBorn, scalar_born
from deepwave.common import cfl_condition, upsample, downsample
from test_scalar import (scattered_2d, _set_sources, _set_coords)


def test_born_scatter_2d():
    """Test Born propagation in a 2D model with a point scatterer."""
    expected, actual = run_born_scatter_2d(propagator=scalarbornprop,
                                           dt=0.001,
                                           prop_kwargs={'pml_width': 30})
    diff = (expected - actual.cpu()).flatten()
    assert diff.norm() < 0.0025


def test_born_scatter_2d_2nd_order():
    """Test Born propagation with a 2nd order accurate propagator."""
    expected, actual = run_born_scatter_2d(propagator=scalarbornprop,
                                           dt=0.0001,
                                           prop_kwargs={'pml_width': 30,
                                                        'accuracy': 2})
    diff = (expected - actual.cpu()).flatten()
    assert diff.norm() < 0.14


def test_born_scatter_2d_6th_order():
    """Test Born propagation with a 6th order accurate propagator."""
    expected, actual = run_born_scatter_2d(propagator=scalarbornprop,
                                           dt=0.0001,
                                           prop_kwargs={'pml_width': 30,
                                                        'accuracy': 6})
    diff = (expected - actual.cpu()).flatten()
    assert diff.norm() < 0.003


def test_born_scatter_2d_8th_order():
    """Test Born propagation with a 8th order accurate propagator."""
    expected, actual = run_born_scatter_2d(propagator=scalarbornprop,
                                           dt=0.0001,
                                           prop_kwargs={'pml_width': 30,
                                                        'accuracy': 8})
    diff = (expected - actual.cpu()).flatten()
    assert diff.norm() < 0.00061


def test_born_scatter_2d_module():
    """Test Born propagation using the Module interface."""
    expected, actual = run_born_scatter_2d(propagator=scalarbornprop,
                                           dt=0.001,
                                           prop_kwargs={'pml_width': 30},
                                           functional=False)
    diff = (expected - actual.cpu()).flatten()
    assert diff.norm() < 0.0025


def scalarbornprop(model, scatter, dx, dt, source_amplitudes,
                   source_locations,
                   receiver_locations, bg_receiver_locations=None,
                   prop_kwargs=None, pml_width=None,
                   survey_pad=None, origin=None, wavefield_0=None,
                   wavefield_m1=None,
                   psiy_m1=None, psix_m1=None,
                   zetay_m1=None, zetax_m1=None,
                   wavefield_sc_0=None, wavefield_sc_m1=None,
                   psiy_sc_m1=None, psix_sc_m1=None,
                   zetay_sc_m1=None, zetax_sc_m1=None, nt=None,
                   model_gradient_sampling_interval=1,
                   functional=True):
    """Wraps the scalar born propagator."""

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
    if bg_receiver_locations is not None:
        bg_receiver_locations = bg_receiver_locations.to(device)

    if functional:
        return scalar_born(model, scatter, dx, dt,
                           source_amplitudes=source_amplitudes,
                           source_locations=source_locations,
                           receiver_locations=receiver_locations,
                           bg_receiver_locations=bg_receiver_locations,
                           wavefield_0=wavefield_0,
                           wavefield_m1=wavefield_m1,
                           psiy_m1=psiy_m1, psix_m1=psix_m1,
                           zetay_m1=zetay_m1, zetax_m1=zetax_m1,
                           wavefield_sc_0=wavefield_sc_0,
                           wavefield_sc_m1=wavefield_sc_m1,
                           psiy_sc_m1=psiy_sc_m1, psix_sc_m1=psix_sc_m1,
                           zetay_sc_m1=zetay_sc_m1,
                           zetax_sc_m1=zetax_sc_m1, nt=nt,
                           model_gradient_sampling_interval=
                           model_gradient_sampling_interval,
                           **prop_kwargs)
    prop = ScalarBorn(model, scatter, dx)
    return prop(dt, source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                bg_receiver_locations=bg_receiver_locations,
                wavefield_0=wavefield_0,
                wavefield_m1=wavefield_m1,
                psiy_m1=psiy_m1, psix_m1=psix_m1,
                zetay_m1=zetay_m1, zetax_m1=zetax_m1,
                wavefield_sc_0=wavefield_sc_0,
                wavefield_sc_m1=wavefield_sc_m1,
                psiy_sc_m1=psiy_sc_m1, psix_sc_m1=psix_sc_m1,
                zetay_sc_m1=zetay_sc_m1, zetax_sc_m1=zetax_sc_m1, nt=nt,
                model_gradient_sampling_interval=
                model_gradient_sampling_interval,
                **prop_kwargs)


def scalarbornpropchained(model, scatter, dx, dt, source_amplitudes,
                          source_locations,
                          receiver_locations, bg_receiver_locations=None,
                          prop_kwargs=None,
                          pml_width=None,
                          survey_pad=None, origin=None, wavefield_0=None,
                          wavefield_m1=None,
                          psiy_m1=None, psix_m1=None,
                          zetay_m1=None, zetax_m1=None,
                          wavefield_sc_0=None, wavefield_sc_m1=None,
                          psiy_sc_m1=None, psix_sc_m1=None,
                          zetay_sc_m1=None, zetax_sc_m1=None, nt=None,
                          model_gradient_sampling_interval=1,
                          functional=True, n_chained=2):
    """Wraps multiple scalar born propagators chained sequentially."""

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
    if bg_receiver_locations is not None:
        bg_receiver_locations = bg_receiver_locations.to(device)

    max_vel = 2000
    dt, step_ratio = cfl_condition(dx[0], dx[1], dt, max_vel)
    if source_amplitudes is not None:
        source_amplitudes = upsample(source_amplitudes, step_ratio)
        nt_per_segment = ((((source_amplitudes.shape[-1] + n_chained - 1)
                            // n_chained + step_ratio - 1) // step_ratio) *
                          step_ratio)
    else:
        nt *= step_ratio
        nt_per_segment = (((nt + n_chained - 1) // n_chained +
                           step_ratio - 1) // step_ratio) * step_ratio

    wfc = wavefield_0
    wfp = wavefield_m1
    psiy = psiy_m1
    psix = psix_m1
    zetay = zetay_m1
    zetax = zetax_m1
    wfcsc = wavefield_sc_0
    wfpsc = wavefield_sc_m1
    psiysc = psiy_sc_m1
    psixsc = psix_sc_m1
    zetaysc = zetay_sc_m1
    zetaxsc = zetax_sc_m1

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

    if bg_receiver_locations is not None:
        if source_amplitudes is not None:
            bg_receiver_amplitudes = (
                torch.zeros(bg_receiver_locations.shape[0],
                            bg_receiver_locations.shape[1],
                            source_amplitudes.shape[-1],
                            dtype=model.dtype,
                            device=model.device)
            )
        else:
            bg_receiver_amplitudes = (
                torch.zeros(bg_receiver_locations.shape[0],
                            bg_receiver_locations.shape[1],
                            nt, dtype=model.dtype,
                            device=model.device)
            )

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
        (wfc, wfp, psiy, psix, zetay, zetax, wfcsc, wfpsc, psiysc, psixsc,
         zetaysc, zetaxsc, segment_bg_receiver_amplitudes,
         segment_receiver_amplitudes) = \
            scalar_born(model, scatter, dx, dt,
                        source_amplitudes=segment_source_amplitudes,
                        source_locations=source_locations,
                        receiver_locations=receiver_locations,
                        bg_receiver_locations=bg_receiver_locations,
                        wavefield_0=wfc, wavefield_m1=wfp,
                        psiy_m1=psiy, psix_m1=psix,
                        zetay_m1=zetay, zetax_m1=zetax,
                        wavefield_sc_0=wfc, wavefield_sc_m1=wfp,
                        psiy_sc_m1=psiy, psix_sc_m1=psix,
                        zetay_sc_m1=zetay, zetax_sc_m1=zetax,
                        nt=segment_nt,
                        model_gradient_sampling_interval=step_ratio,
                        **prop_kwargs)
        if receiver_locations is not None:
            receiver_amplitudes[...,
                                nt_per_segment * segment_idx:
                                min(nt_per_segment * (segment_idx+1),
                                    receiver_amplitudes.shape[-1])] = \
                                    segment_receiver_amplitudes
        if bg_receiver_locations is not None:
            bg_receiver_amplitudes[...,
                                   nt_per_segment * segment_idx:
                                   min(nt_per_segment * (segment_idx+1),
                                       bg_receiver_amplitudes.shape[-1])] = \
                                       segment_bg_receiver_amplitudes

    if receiver_locations is not None:
        receiver_amplitudes = downsample(receiver_amplitudes, step_ratio)
    if bg_receiver_locations is not None:
        bg_receiver_amplitudes = downsample(bg_receiver_amplitudes, step_ratio)

    return (wfc, wfp, psiy, psix, zetay, zetax, wfcsc, wfpsc, psiysc, psixsc,
            zetaysc, zetaxsc, bg_receiver_amplitudes, receiver_amplitudes)


def run_born_scatter(c, dc, freq, dx, dt, nx,
                     num_shots, num_sources_per_shot,
                     num_receivers_per_shot,
                     propagator, prop_kwargs, device=None, dtype=None,
                     **kwargs):
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
    x_s = _set_coords(num_shots, num_sources_per_shot, nx)
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx)
    x_p = _set_coords(1, 1, nx, 'middle')[0, 0]
    scatter = torch.zeros_like(model)
    scatter[torch.split((x_p).long(), 1)] = dc
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

    actual = propagator(model, scatter, dx, dt, sources['amplitude'],
                        sources['locations'], x_r, x_r,
                        prop_kwargs=prop_kwargs, **kwargs)[-1]

    return expected, actual


def run_born_scatter_2d(c=1500, dc=150, freq=25, dx=(5, 5), dt=0.0001,
                        nx=(50, 50),
                        num_shots=2, num_sources_per_shot=2,
                        num_receivers_per_shot=2,
                        propagator=None, prop_kwargs=None, device=None,
                        dtype=None, **kwargs):
    """Runs run_born_scatter with default parameters for 2D."""

    return run_born_scatter(c, dc, freq, dx, dt, nx,
                            num_shots, num_sources_per_shot,
                            num_receivers_per_shot,
                            propagator, prop_kwargs, device, dtype, **kwargs)


def run_born_forward(c, freq, dx, dt, nx,
                     num_shots, num_sources_per_shot,
                     num_receivers_per_shot,
                     propagator, prop_kwargs, device=None,
                     dtype=None, dc=100, **kwargs):
    """Create a random model and forward propagate.
    """
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (torch.ones(*nx, device=device, dtype=dtype) * c +
             torch.randn(*nx, dtype=dtype).to(device) * dc)
    scatter = torch.randn(*nx, dtype=dtype).to(device) * dc

    nx = torch.Tensor(nx).long()
    dx = torch.Tensor(dx)

    nt = int((2 * torch.norm(nx.float() * dx) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx)
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx, 'bottom')
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    return propagator(model, scatter, dx, dt, sources['amplitude'],
                      sources['locations'], x_r, x_r,
                      prop_kwargs=prop_kwargs, **kwargs)


def run_born_forward_2d(c=1500, freq=25, dx=(5, 5), dt=0.004, nx=(50, 50),
                        num_shots=2, num_sources_per_shot=2,
                        num_receivers_per_shot=2,
                        propagator=None, prop_kwargs=None, device=None,
                        dtype=None, **kwargs):
    """Runs run_forward with default parameters for 2D."""

    return run_born_forward(c, freq, dx, dt, nx,
                            num_shots, num_sources_per_shot,
                            num_receivers_per_shot,
                            propagator, prop_kwargs, device, dtype, **kwargs)


def test_forward_cpu_gpu_match():
    """Test propagation on CPU and GPU produce the same result."""
    if torch.cuda.is_available():
        actual_cpu = run_born_forward_2d(propagator=scalarbornprop,
                                         device=torch.device("cpu"))
        actual_gpu = run_born_forward_2d(propagator=scalarbornprop,
                                         device=torch.device("cuda"))
        for cpui, gpui in zip(actual_cpu, actual_gpu):
            assert torch.allclose(cpui, gpui.cpu(), atol=1e-5)


def test_born_gradcheck_2d():
    """Test gradcheck in a 2D model with Born propagator."""
    run_born_gradcheck_2d(propagator=scalarbornprop)


def test_born_gradcheck_2d_2nd_order():
    """Test gradcheck with a 2nd order accurate propagator."""
    run_born_gradcheck_2d(propagator=scalarbornprop,
                          prop_kwargs={'accuracy': 2})


def test_born_gradcheck_2d_6th_order():
    """Test gradcheck with a 6th order accurate propagator."""
    run_born_gradcheck_2d(propagator=scalarbornprop,
                          prop_kwargs={'accuracy': 6})


def test_born_gradcheck_2d_8th_order():
    """Test gradcheck with a 8th order accurate propagator."""
    run_born_gradcheck_2d(propagator=scalarbornprop,
                          prop_kwargs={'accuracy': 8})


def test_born_gradcheck_2d_cfl():
    """Test gradcheck with a timestep greater than the CFL limit."""
    run_born_gradcheck_2d(propagator=scalarbornprop, dt=0.002, atol=2e-7,
                          rtol=1e-8, nt_add=100)


def test_born_gradcheck_2d_odd_timesteps():
    """Test gradcheck with one more timestep."""
    run_born_gradcheck_2d(propagator=scalarbornprop, nt_add=1)


def test_born_gradcheck_2d_one_shot():
    """Test gradcheck with one shot."""
    run_born_gradcheck_2d(propagator=scalarbornprop, num_shots=1)


def test_born_gradcheck_2d_no_sources():
    """Test gradcheck with no sources."""
    run_born_gradcheck_2d(propagator=scalarbornprop, num_sources_per_shot=0)


def test_born_gradcheck_2d_no_receivers():
    """Test gradcheck with no receivers."""
    run_born_gradcheck_2d(propagator=scalarbornprop, num_receivers_per_shot=0)


def test_born_gradcheck_2d_survey_pad():
    """Test gradcheck with survey_pad."""
    run_born_gradcheck_2d(propagator=scalarbornprop, survey_pad=0,
                          provide_wavefields=False)


def test_born_gradcheck_2d_partial_wavefields():
    """Test gradcheck with wavefields that do not cover the model."""
    run_born_gradcheck_2d(propagator=scalarbornprop, origin=(0, 2),
                          wavefield_size=(4+6, 1+6))


def test_born_gradcheck_2d_chained():
    """Test gradcheck when two propagators are chained."""
    run_born_gradcheck_2d(propagator=scalarbornpropchained)


def test_born_gradcheck_2d_negative():
    """Test gradcheck with negative velocity."""
    run_born_gradcheck_2d(c=-1500, propagator=scalarbornprop)


def test_born_gradcheck_2d_zero():
    """Test gradcheck with zero velocity."""
    run_born_gradcheck_2d(c=0, dc=0, propagator=scalarbornprop)


def test_born_gradcheck_2d_different_pml():
    """Test gradcheck with different pml widths."""
    run_born_gradcheck_2d(propagator=scalarbornprop, pml_width=[0, 1, 5, 10],
                          atol=5e-8)


def test_born_gradcheck_2d_no_pml():
    """Test gradcheck with no pml."""
    run_born_gradcheck_2d(propagator=scalarbornprop, pml_width=0, atol=2e-8)


def test_born_gradcheck_2d_different_dx():
    """Test gradcheck with different dx values."""
    run_born_gradcheck_2d(propagator=scalarbornprop, dx=(4, 5), dt=0.0005,
                          atol=2e-8)


def test_born_gradcheck_2d_single_cell():
    """Test gradcheck with a single model cell."""
    run_born_gradcheck_2d(propagator=scalarbornprop, nx=(1, 1),
                          num_shots=1,
                          num_sources_per_shot=1,
                          num_receivers_per_shot=1)


def test_born_gradcheck_2d_big():
    """Test gradcheck with a big model."""
    run_born_gradcheck_2d(propagator=scalarbornprop,
                          nx=(5+2*(3+3*2), 4+2*(3+3*2)))


def test_negative_vel(c=1500, dc=100, freq=25, dx=(5, 5), dt=0.005,
                      nx=(50, 50), num_shots=2, num_sources_per_shot=2,
                      num_receivers_per_shot=2,
                      propagator=scalarbornprop, prop_kwargs=None, device=None,
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
    scatter = torch.randn(*nx, device=device, dtype=dtype)
    out_positive = propagator(model, scatter, dx, dt, sources['amplitude'],
                              sources['locations'], x_r, x_r,
                              prop_kwargs=prop_kwargs)

    # Negative velocity
    out = propagator(-model, scatter, dx, dt, sources['amplitude'],
                     sources['locations'], x_r, x_r,
                     prop_kwargs=prop_kwargs)
    assert torch.allclose(out_positive[0], out[0])
    assert torch.allclose(out_positive[6], -out[6])
    assert torch.allclose(out_positive[-1], -out[-1])

    # Negative dt
    out = propagator(model, scatter, dx, -dt, sources['amplitude'],
                     sources['locations'], x_r, x_r,
                     prop_kwargs=prop_kwargs)
    assert torch.allclose(out_positive[0], out[0])
    assert torch.allclose(out_positive[6], out[6])
    assert torch.allclose(out_positive[-1], out[-1])

    # Zero velocity
    out = propagator(torch.zeros_like(model), scatter, dx, -dt,
                     sources['amplitude'],
                     sources['locations'], x_r, x_r,
                     prop_kwargs=prop_kwargs)
    assert torch.allclose(out[0], torch.zeros_like(out[0]))
    assert torch.allclose(out[6], torch.zeros_like(out[6]))


def test_born_gradcheck_only_v_2d():
    """Test gradcheck with only v requiring gradient."""
    run_born_gradcheck_2d(propagator=scalarbornprop,
                          scatter_requires_grad=False,
                          source_requires_grad=False,
                          wavefield_0_requires_grad=False,
                          wavefield_m1_requires_grad=False,
                          psiy_m1_requires_grad=False,
                          psix_m1_requires_grad=False,
                          zetay_m1_requires_grad=False,
                          zetax_m1_requires_grad=False,
                          wavefieldsc_0_requires_grad=False,
                          wavefieldsc_m1_requires_grad=False,
                          psiysc_m1_requires_grad=False,
                          psixsc_m1_requires_grad=False,
                          zetaysc_m1_requires_grad=False,
                          zetaxsc_m1_requires_grad=False,
                          )


def test_born_gradcheck_only_scatter_2d():
    """Test gradcheck with only scatter requiring gradient."""
    run_born_gradcheck_2d(propagator=scalarbornprop,
                          v_requires_grad=False,
                          source_requires_grad=False,
                          wavefield_0_requires_grad=False,
                          wavefield_m1_requires_grad=False,
                          psiy_m1_requires_grad=False,
                          psix_m1_requires_grad=False,
                          zetay_m1_requires_grad=False,
                          zetax_m1_requires_grad=False,
                          wavefieldsc_0_requires_grad=False,
                          wavefieldsc_m1_requires_grad=False,
                          psiysc_m1_requires_grad=False,
                          psixsc_m1_requires_grad=False,
                          zetaysc_m1_requires_grad=False,
                          zetaxsc_m1_requires_grad=False,
                          )


def test_born_gradcheck_only_source_2d():
    """Test gradcheck with only source requiring gradient."""
    run_born_gradcheck_2d(propagator=scalarbornprop,
                          v_requires_grad=False,
                          scatter_requires_grad=False,
                          wavefield_0_requires_grad=False,
                          wavefield_m1_requires_grad=False,
                          psiy_m1_requires_grad=False,
                          psix_m1_requires_grad=False,
                          zetay_m1_requires_grad=False,
                          zetax_m1_requires_grad=False,
                          wavefieldsc_0_requires_grad=False,
                          wavefieldsc_m1_requires_grad=False,
                          psiysc_m1_requires_grad=False,
                          psixsc_m1_requires_grad=False,
                          zetaysc_m1_requires_grad=False,
                          zetaxsc_m1_requires_grad=False,
                          )


def test_born_gradcheck_only_wavefield_0_2d():
    """Test gradcheck with only wavefield_0 requiring gradient."""
    run_born_gradcheck_2d(propagator=scalarbornprop,
                          v_requires_grad=False,
                          scatter_requires_grad=False,
                          source_requires_grad=False,
                          wavefield_m1_requires_grad=False,
                          psiy_m1_requires_grad=False,
                          psix_m1_requires_grad=False,
                          zetay_m1_requires_grad=False,
                          zetax_m1_requires_grad=False,
                          wavefieldsc_0_requires_grad=False,
                          wavefieldsc_m1_requires_grad=False,
                          psiysc_m1_requires_grad=False,
                          psixsc_m1_requires_grad=False,
                          zetaysc_m1_requires_grad=False,
                          zetaxsc_m1_requires_grad=False,
                          )


def test_born_gradcheck_only_wavefieldsc_0_2d():
    """Test gradcheck with only wavefieldsc_0 requiring gradient."""
    run_born_gradcheck_2d(propagator=scalarbornprop,
                          v_requires_grad=False,
                          scatter_requires_grad=False,
                          source_requires_grad=False,
                          wavefield_0_requires_grad=False,
                          wavefield_m1_requires_grad=False,
                          psiy_m1_requires_grad=False,
                          psix_m1_requires_grad=False,
                          zetay_m1_requires_grad=False,
                          zetax_m1_requires_grad=False,
                          wavefieldsc_m1_requires_grad=False,
                          psiysc_m1_requires_grad=False,
                          psixsc_m1_requires_grad=False,
                          zetaysc_m1_requires_grad=False,
                          zetaxsc_m1_requires_grad=False,
                          )


def test_jit():
    """Test that the propagator can be JIT compiled"""
    torch.jit.script(ScalarBorn(torch.ones(1, 1), torch.ones(1, 1), 5.0))(
        0.001, source_amplitudes=torch.ones(1, 1, 1),
        source_locations=torch.zeros(1, 1, 2)
    )
    torch.jit.script(scalar_born)(
        torch.ones(1, 1), torch.ones(1, 1), 5.0,
        0.001, source_amplitudes=torch.ones(1, 1, 1),
        source_locations=torch.zeros(1, 1, 2)
    )


def run_born_gradcheck(c, dc, freq, dx, dt, nx,
                       num_shots, num_sources_per_shot,
                       num_receivers_per_shot,
                       propagator, prop_kwargs,
                       pml_width=3, survey_pad=None,
                       origin=None, wavefield_size=None,
                       device=None, dtype=None, v_requires_grad=True,
                       scatter_requires_grad=True,
                       source_requires_grad=True,
                       provide_wavefields=True,
                       wavefield_0_requires_grad=True,
                       wavefield_m1_requires_grad=True,
                       psiy_m1_requires_grad=True,
                       psix_m1_requires_grad=True,
                       zetay_m1_requires_grad=True,
                       zetax_m1_requires_grad=True,
                       wavefieldsc_0_requires_grad=True,
                       wavefieldsc_m1_requires_grad=True,
                       psiysc_m1_requires_grad=True,
                       psixsc_m1_requires_grad=True,
                       zetaysc_m1_requires_grad=True,
                       zetaxsc_m1_requires_grad=True,
                       atol=1e-8, rtol=1e-5, nt_add=0):
    """Run PyTorch's gradcheck."""
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (torch.ones(*nx, device=device, dtype=dtype) * c +
             torch.rand(*nx, device=device, dtype=dtype) * dc)
    scatter = torch.rand(*nx, device=device, dtype=dtype) * dc

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
        wavefield_sc_0 = torch.zeros_like(wavefield_0)
        wavefield_sc_m1 = torch.zeros_like(wavefield_0)
        psiy_sc_m1 = torch.zeros_like(wavefield_0)
        psix_sc_m1 = torch.zeros_like(wavefield_0)
        zetay_sc_m1 = torch.zeros_like(wavefield_0)
        zetax_sc_m1 = torch.zeros_like(wavefield_0)
        wavefield_0.requires_grad_(wavefield_0_requires_grad)
        wavefield_m1.requires_grad_(wavefield_m1_requires_grad)
        psiy_m1.requires_grad_(psiy_m1_requires_grad)
        psix_m1.requires_grad_(psix_m1_requires_grad)
        zetay_m1.requires_grad_(zetay_m1_requires_grad)
        zetax_m1.requires_grad_(zetax_m1_requires_grad)
        wavefield_sc_0.requires_grad_(wavefieldsc_0_requires_grad)
        wavefield_sc_m1.requires_grad_(wavefieldsc_m1_requires_grad)
        psiy_sc_m1.requires_grad_(psiysc_m1_requires_grad)
        psix_sc_m1.requires_grad_(psixsc_m1_requires_grad)
        zetay_sc_m1.requires_grad_(zetaysc_m1_requires_grad)
        zetax_sc_m1.requires_grad_(zetaxsc_m1_requires_grad)
    else:
        wavefield_0 = None
        wavefield_m1 = None
        psiy_m1 = None
        psix_m1 = None
        zetay_m1 = None
        zetax_m1 = None
        wavefield_sc_0 = None
        wavefield_sc_m1 = None
        psiy_sc_m1 = None
        psix_sc_m1 = None
        zetay_sc_m1 = None
        zetax_sc_m1 = None

    model.requires_grad_(v_requires_grad)
    scatter.requires_grad_(scatter_requires_grad)

    torch.autograd.gradcheck(propagator, (model, scatter, dx, dt,
                                          sources['amplitude'],
                                          sources['locations'], x_r, x_r,
                                          prop_kwargs, pml_width, survey_pad,
                                          origin,
                                          wavefield_0,
                                          wavefield_m1,
                                          psiy_m1, psix_m1,
                                          zetay_m1, zetax_m1,
                                          wavefield_sc_0,
                                          wavefield_sc_m1,
                                          psiy_sc_m1, psix_sc_m1,
                                          zetay_sc_m1, zetax_sc_m1, nt, 1,
                                          True
                                          ),
                             nondet_tol=1e-3, check_grad_dtypes=True,
                             atol=atol, rtol=rtol)


def run_born_gradcheck_2d(c=1500, dc=100, freq=25, dx=(5, 5), dt=0.001,
                          nx=(4, 3),
                          num_shots=2, num_sources_per_shot=2,
                          num_receivers_per_shot=2,
                          propagator=None, prop_kwargs=None,
                          pml_width=3,
                          survey_pad=None,
                          device=None, dtype=torch.double, **kwargs):
    """Runs run_gradcheck with default parameters for 2D."""

    return run_born_gradcheck(c, dc, freq, dx, dt, nx,
                              num_shots, num_sources_per_shot,
                              num_receivers_per_shot,
                              propagator, prop_kwargs, pml_width=pml_width,
                              survey_pad=survey_pad,
                              device=device, dtype=dtype, **kwargs)
