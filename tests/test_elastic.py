import torch
import pytest
from deepwave import Elastic, elastic
from deepwave.wavelets import ricker
from deepwave.common import (cfl_condition, upsample, downsample,
                             lambmubuoyancy_to_vpvsrho,
                             vpvsrho_to_lambmubuoyancy, IGNORE_LOCATION)

DEFAULT_LAMB = 550000000
DEFAULT_MU = 2200000000
DEFAULT_BUOYANCY = 1 / 2200


def elasticprop(lamb,
                mu,
                buoyancy,
                dx,
                dt,
                source_amplitudes_y,
                source_amplitudes_x,
                source_locations_y,
                source_locations_x,
                receiver_locations_y,
                receiver_locations_x,
                receiver_locations_p,
                prop_kwargs=None,
                pml_width=None,
                survey_pad=None,
                origin=None,
                vy0=None,
                vx0=None,
                sigmayy0=None,
                sigmaxy0=None,
                sigmaxx0=None,
                m_vyy0=None,
                m_vyx0=None,
                m_vxy0=None,
                m_vxx0=None,
                m_sigmayyy0=None,
                m_sigmaxyy0=None,
                m_sigmaxyx0=None,
                m_sigmaxxx0=None,
                nt=None,
                model_gradient_sampling_interval=1,
                functional=True):
    """Wraps the scalar propagator."""

    if prop_kwargs is None:
        prop_kwargs = {}
    # For consistency when actual max speed changes
    prop_kwargs['max_vel'] = 2000

    prop_kwargs['pml_freq'] = 25

    # Workaround for gradcheck not accepting prop_kwargs dictionary
    if pml_width is not None:
        prop_kwargs['pml_width'] = pml_width
    prop_kwargs['survey_pad'] = survey_pad
    prop_kwargs['origin'] = origin

    device = lamb.device
    if source_amplitudes_y is not None:
        source_amplitudes_y = source_amplitudes_y.to(device)
        source_locations_y = source_locations_y.to(device)
    if source_amplitudes_x is not None:
        source_amplitudes_x = source_amplitudes_x.to(device)
        source_locations_x = source_locations_x.to(device)
    if receiver_locations_y is not None:
        receiver_locations_y = receiver_locations_y.to(device)
    if receiver_locations_x is not None:
        receiver_locations_x = receiver_locations_x.to(device)
    if receiver_locations_p is not None:
        receiver_locations_p = receiver_locations_p.to(device)

    if functional:
        return elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_y=source_amplitudes_y,
            source_amplitudes_x=source_amplitudes_x,
            source_locations_y=source_locations_y,
            source_locations_x=source_locations_x,
            receiver_locations_y=receiver_locations_y,
            receiver_locations_x=receiver_locations_x,
            receiver_locations_p=receiver_locations_p,
            vy_0=vy0,
            vx_0=vx0,
            sigmayy_0=sigmayy0,
            sigmaxy_0=sigmaxy0,
            sigmaxx_0=sigmaxx0,
            m_vyy_0=m_vyy0,
            m_vyx_0=m_vyx0,
            m_vxy_0=m_vxy0,
            m_vxx_0=m_vxx0,
            m_sigmayyy_0=m_sigmayyy0,
            m_sigmaxyy_0=m_sigmaxyy0,
            m_sigmaxyx_0=m_sigmaxyx0,
            m_sigmaxxx_0=m_sigmaxxx0,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            **prop_kwargs)

    prop = Elastic(lamb, mu, buoyancy, dx)
    return prop(
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_amplitudes_x=source_amplitudes_x,
        source_locations_y=source_locations_y,
        source_locations_x=source_locations_x,
        receiver_locations_y=receiver_locations_y,
        receiver_locations_x=receiver_locations_x,
        receiver_locations_p=receiver_locations_p,
        vy_0=vy0,
        vx_0=vx0,
        sigmayy_0=sigmayy0,
        sigmaxy_0=sigmaxy0,
        sigmaxx_0=sigmaxx0,
        m_vyy_0=m_vyy0,
        m_vyx_0=m_vyx0,
        m_vxy_0=m_vxy0,
        m_vxx_0=m_vxx0,
        m_sigmayyy_0=m_sigmayyy0,
        m_sigmaxyy_0=m_sigmaxyy0,
        m_sigmaxyx_0=m_sigmaxyx0,
        m_sigmaxxx_0=m_sigmaxxx0,
        nt=nt,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        **prop_kwargs)


def elasticpropchained(lamb,
                       mu,
                       buoyancy,
                       dx,
                       dt,
                       source_amplitudes_y,
                       source_amplitudes_x,
                       source_locations_y,
                       source_locations_x,
                       receiver_locations_y,
                       receiver_locations_x,
                       receiver_locations_p,
                       prop_kwargs=None,
                       pml_width=None,
                       survey_pad=None,
                       origin=None,
                       vy0=None,
                       vx0=None,
                       sigmayy0=None,
                       sigmaxy0=None,
                       sigmaxx0=None,
                       m_vyy0=None,
                       m_vyx0=None,
                       m_vxy0=None,
                       m_vxx0=None,
                       m_sigmayyy0=None,
                       m_sigmaxyy0=None,
                       m_sigmaxyx0=None,
                       m_sigmaxxx0=None,
                       nt=None,
                       model_gradient_sampling_interval=1,
                       functional=True,
                       n_chained=2):
    """Wraps multiple scalar propagators chained sequentially."""
    if prop_kwargs is None:
        prop_kwargs = {}
    # For consistency when actual max speed changes
    prop_kwargs['max_vel'] = 2000

    prop_kwargs['pml_freq'] = 25

    # Workaround for gradcheck not accepting prop_kwargs dictionary
    if pml_width is not None:
        prop_kwargs['pml_width'] = pml_width
    prop_kwargs['survey_pad'] = survey_pad
    prop_kwargs['origin'] = origin

    device = lamb.device
    if source_amplitudes_y is not None:
        source_amplitudes_y = source_amplitudes_y.to(device)
        source_locations_y = source_locations_y.to(device)
    if source_amplitudes_x is not None:
        source_amplitudes_x = source_amplitudes_x.to(device)
        source_locations_x = source_locations_x.to(device)
    if receiver_locations_y is not None:
        receiver_locations_y = receiver_locations_y.to(device)
    if receiver_locations_x is not None:
        receiver_locations_x = receiver_locations_x.to(device)
    if receiver_locations_p is not None:
        receiver_locations_p = receiver_locations_p.to(device)

    max_vel = 2000
    dt, step_ratio = cfl_condition(dx[0], dx[1], dt, max_vel)
    source_nt = None
    if source_amplitudes_y is not None or source_amplitudes_x is not None:
        if source_amplitudes_y is not None:
            source_amplitudes_y = upsample(source_amplitudes_y, step_ratio)
            source_nt = source_amplitudes_y.shape[-1]
            nt_per_segment = (((
                (source_nt + n_chained - 1) // n_chained + step_ratio - 1) //
                               step_ratio) * step_ratio)
        if source_amplitudes_x is not None:
            source_amplitudes_x = upsample(source_amplitudes_x, step_ratio)
            source_nt = source_amplitudes_x.shape[-1]
            nt_per_segment = (((
                (source_nt + n_chained - 1) // n_chained + step_ratio - 1) //
                               step_ratio) * step_ratio)
    else:
        nt *= step_ratio
        nt_per_segment = ((
            (nt + n_chained - 1) // n_chained + step_ratio - 1) //
                          step_ratio) * step_ratio

    vy = vx0
    vx = vy0
    sigmayy = sigmayy0
    sigmaxy = sigmaxy0
    sigmaxx = sigmaxx0
    m_vyy = m_vyy0
    m_vyx = m_vyx0
    m_vxy = m_vxy0
    m_vxx = m_vxx0
    m_sigmayyy = m_sigmayyy0
    m_sigmaxyy = m_sigmaxyy0
    m_sigmaxyx = m_sigmaxyx0
    m_sigmaxxx = m_sigmaxxx0

    if receiver_locations_y is not None:
        if source_nt is not None:
            receiver_amplitudes_y = torch.zeros(receiver_locations_y.shape[0],
                                                receiver_locations_y.shape[1],
                                                source_nt,
                                                dtype=lamb.dtype,
                                                device=lamb.device)
        else:
            receiver_amplitudes_y = torch.zeros(receiver_locations_y.shape[0],
                                                receiver_locations_y.shape[1],
                                                nt,
                                                dtype=lamb.dtype,
                                                device=lamb.device)

    if receiver_locations_x is not None:
        if source_nt is not None:
            receiver_amplitudes_x = torch.zeros(receiver_locations_x.shape[0],
                                                receiver_locations_x.shape[1],
                                                source_nt,
                                                dtype=lamb.dtype,
                                                device=lamb.device)
        else:
            receiver_amplitudes_x = torch.zeros(receiver_locations_x.shape[0],
                                                receiver_locations_x.shape[1],
                                                nt,
                                                dtype=lamb.dtype,
                                                device=lamb.device)

    if receiver_locations_p is not None:
        if source_nt is not None:
            receiver_amplitudes_p = torch.zeros(receiver_locations_p.shape[0],
                                                receiver_locations_p.shape[1],
                                                source_nt,
                                                dtype=lamb.dtype,
                                                device=lamb.device)
        else:
            receiver_amplitudes_p = torch.zeros(receiver_locations_p.shape[0],
                                                receiver_locations_p.shape[1],
                                                nt,
                                                dtype=lamb.dtype,
                                                device=lamb.device)

    for segment_idx in range(n_chained):
        if (source_amplitudes_y is not None
                or source_amplitudes_x is not None):
            if source_amplitudes_y is not None:
                segment_source_amplitudes_y = \
                    source_amplitudes_y[...,
                                        nt_per_segment * segment_idx:
                                        min(nt_per_segment * (segment_idx+1),
                                            source_nt)]
            if source_amplitudes_x is not None:
                segment_source_amplitudes_x = \
                    source_amplitudes_x[...,
                                        nt_per_segment * segment_idx:
                                        min(nt_per_segment * (segment_idx+1),
                                            source_nt)]
            segment_nt = None
        else:
            segment_source_amplitudes_y = None
            segment_source_amplitudes_x = None
            segment_nt = (nt_per_segment * (segment_idx + 1) -
                          nt_per_segment * segment_idx)
        (vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,
         m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx,
         segment_receiver_amplitudes_p,
         segment_receiver_amplitudes_y, segment_receiver_amplitudes_x) = \
            elastic(lamb, mu, buoyancy, dx, dt,
                    source_amplitudes_y=segment_source_amplitudes_y,
                    source_amplitudes_x=segment_source_amplitudes_x,
                    source_locations_y=source_locations_y,
                    source_locations_x=source_locations_x,
                    receiver_locations_y=receiver_locations_y,
                    receiver_locations_x=receiver_locations_x,
                    receiver_locations_p=receiver_locations_p,
                    vy_0=vx,
                    vx_0=vy,
                    sigmayy_0=sigmayy, sigmaxy_0=sigmaxy,
                    sigmaxx_0=sigmaxx, m_vyy_0=m_vyy, m_vyx_0=m_vyx,
                    m_vxy_0=m_vxy, m_vxx_0=m_vxx, m_sigmayyy_0=m_sigmayyy,
                    m_sigmaxyy_0=m_sigmaxyy, m_sigmaxyx_0=m_sigmaxyx,
                    m_sigmaxxx_0=m_sigmaxxx,
                    nt=segment_nt,
                    model_gradient_sampling_interval=step_ratio,
                    **prop_kwargs)
        if receiver_locations_y is not None:
            receiver_amplitudes_y[...,
                                  nt_per_segment * segment_idx:
                                  min(nt_per_segment * (segment_idx+1),
                                      receiver_amplitudes_y.shape[-1])] = \
                                      segment_receiver_amplitudes_y
        if receiver_locations_x is not None:
            receiver_amplitudes_x[...,
                                  nt_per_segment * segment_idx:
                                  min(nt_per_segment * (segment_idx+1),
                                      receiver_amplitudes_x.shape[-1])] = \
                                      segment_receiver_amplitudes_x
        if receiver_locations_p is not None:
            receiver_amplitudes_p[...,
                                  nt_per_segment * segment_idx:
                                  min(nt_per_segment * (segment_idx+1),
                                      receiver_amplitudes_p.shape[-1])] = \
                                      segment_receiver_amplitudes_p

    if receiver_locations_y is not None:
        receiver_amplitudes_y = downsample(receiver_amplitudes_y, step_ratio)
    if receiver_locations_x is not None:
        receiver_amplitudes_x = downsample(receiver_amplitudes_x, step_ratio)
    if receiver_locations_p is not None:
        receiver_amplitudes_p = downsample(receiver_amplitudes_p, step_ratio)

    return (vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,
            m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx,
            receiver_amplitudes_p, receiver_amplitudes_y,
            receiver_amplitudes_x)


def test_forward():
    """Compare the recorded data with that generated by another code."""
    with open('tests/Uz_file_ascii', 'r') as f:
        d = f.read().split()
        expected_vy = torch.tensor([float(di) for di in d[:3000]])
    with open('tests/Ux_file_ascii', 'r') as f:
        d = f.read().split()
        expected_vx = torch.tensor([float(di) for di in d[3000:]])
    expected_scale = expected_vy.abs().max().item()
    expected_vy /= expected_scale
    expected_vx /= expected_scale
    for accuracy, target_err in zip([2, 4], [0.75, 0.23]):
        for orientation in range(4):
            out = run_forward_lamb(orientation,
                                   prop_kwargs={'accuracy': accuracy})
            if orientation < 2:
                vy = -out[-2].cpu().flatten()
                vx = out[-1].cpu().flatten()
            else:
                vx = out[-2].cpu().flatten()
                vy = -out[-1].cpu().flatten()
            scale = vy.abs().max().item()
            vy /= scale
            vx /= scale
            assert (vy[1:] - expected_vy[:-1]).norm().item() < target_err
            assert (vx[1:] - expected_vx[:-1]).norm().item() < target_err


def test_wavefield_decays():
    """Test that the PML causes the wavefield amplitude to decay."""
    out = run_forward_2d(propagator=elasticprop, nt=10000)
    for outi in out[:-3]:
        assert outi.norm() < 2e-6


def test_model_too_small():
    """Test that an error is raised when the model is too small."""
    with pytest.raises(RuntimeError):
        run_forward_2d(propagator=elasticprop, nx=(3, 3), pml_width=3)


def test_forward_cpu_gpu_match():
    """Test propagation on CPU and GPU produce the same result."""
    if torch.cuda.is_available():
        actual_cpu = run_forward_2d(propagator=elasticprop,
                                    device=torch.device("cpu"))
        actual_gpu = run_forward_2d(propagator=elasticprop,
                                    device=torch.device("cuda"))
        for cpui, gpui in zip(actual_cpu[:-1], actual_gpu[:-1]):
            assert torch.allclose(cpui, gpui.cpu(), atol=1e-6)
        cpui = actual_cpu[-1]
        gpui = actual_cpu[-1]
        assert torch.allclose(cpui, gpui.cpu(), atol=5e-5)


def test_unused_source_receiver(mlamb=DEFAULT_LAMB,
                   mmu=DEFAULT_MU,
                   mbuoyancy=DEFAULT_BUOYANCY,
                   freq=25,
                   dx=(5, 5),
                   dt=0.004,
                   nx=(50, 50),
                   num_shots=2,
                   num_sources_per_shot=3,
                   num_receivers_per_shot=4,
                   propagator=elasticprop,
                   prop_kwargs=None,
                   device=None,
                   dtype=None,
                   dlamb=DEFAULT_LAMB / 10,
                   dmu=DEFAULT_MU / 10,
                   dbuoyancy=DEFAULT_BUOYANCY / 10,
                   nt=None,
                   dpeak_time=0.3):
    """Test forward and backward propagation when a source and receiver are unused."""

    assert num_shots > 1
    assert num_sources_per_shot > num_shots
    assert num_receivers_per_shot > num_shots + 1

    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nx = torch.Tensor(nx).long()
    dx = torch.Tensor(dx)

    lamb = (torch.ones(*nx, device=device, dtype=dtype) * mlamb +
            torch.randn(*nx, dtype=dtype).to(device) * dlamb)
    mu = (torch.ones(*nx, device=device, dtype=dtype) * mmu +
          torch.randn(*nx, dtype=dtype).to(device) * dmu)
    buoyancy = (torch.ones(*nx, device=device, dtype=dtype) * mbuoyancy +
                torch.randn(*nx, dtype=dtype).to(device) * dbuoyancy)

    vp, vs, rho = lambmubuoyancy_to_vpvsrho(lamb.abs(), mu.abs(),
                                            buoyancy.abs())
    vmin = min(vp.abs().min(), vs.abs().min())

    if nt is None:
        nt = int(
            (2 * torch.norm(nx.float() * dx) / vmin + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, 2 * num_sources_per_shot, nx)
    x_s_y = x_s[:, :num_sources_per_shot]
    x_s_x = x_s[:, num_sources_per_shot:]
    x_r = _set_coords(num_shots, 3 * num_receivers_per_shot, nx, 'bottom')
    x_r_y = x_r[:, :num_receivers_per_shot]
    x_r_x = x_r[:, num_receivers_per_shot:2*num_receivers_per_shot]
    x_r_p = x_r[:, 2*num_receivers_per_shot:]
    sources_y = _set_sources(x_s_y, freq, dt, nt, dtype, dpeak_time=dpeak_time)
    sources_x = _set_sources(x_s_x, freq, dt, nt, dtype, dpeak_time=dpeak_time)

    # Forward with source and receiver ignored
    lambi = lamb.clone()
    lambi.requires_grad_()
    mui = mu.clone()
    mui.requires_grad_()
    buoyancyi = buoyancy.clone()
    buoyancyi.requires_grad_()
    for i in range(num_shots):
        x_s_y[i, i, :] = IGNORE_LOCATION
        x_s_x[i, i+1, :] = IGNORE_LOCATION
        x_r_y[i, i, :] = IGNORE_LOCATION
        x_r_x[i, i+1, :] = IGNORE_LOCATION
        x_r_p[i, i+2, :] = IGNORE_LOCATION
    sources_y['locations'] = x_s_y
    sources_x['locations'] = x_s_x
    out_ignored = propagator(lambi,
                      mui,
                      buoyancyi,
                      dx,
                      dt,
                      sources_y['amplitude'],
                      sources_x['amplitude'],
                      sources_y['locations'],
                      sources_x['locations'],
                      x_r_y,
                      x_r_x,
                      x_r_p,
                      prop_kwargs=prop_kwargs)

    ((out_ignored[-1]**2).sum() + (out_ignored[-2]**2).sum() + (out_ignored[-3]**2).sum()).backward()

    # Forward with amplitudes of sources that will be ignored set to zero
    lambf = lamb.clone()
    lambf.requires_grad_()
    muf = mu.clone()
    muf.requires_grad_()
    buoyancyf = buoyancy.clone()
    buoyancyf.requires_grad_()
    x_s = _set_coords(num_shots, 2 * num_sources_per_shot, nx)
    x_s_y = x_s[:, :num_sources_per_shot]
    x_s_x = x_s[:, num_sources_per_shot:]
    x_r = _set_coords(num_shots, 3 * num_receivers_per_shot, nx, 'bottom')
    x_r_y = x_r[:, :num_receivers_per_shot]
    x_r_x = x_r[:, num_receivers_per_shot:2*num_receivers_per_shot]
    x_r_p = x_r[:, 2*num_receivers_per_shot:]
    for i in range(num_shots):
        sources_y['amplitude'][i, i].fill_(0)
        sources_x['amplitude'][i, i+1].fill_(0)
    sources_y['locations'] = x_s_y
    sources_x['locations'] = x_s_x
    out_filled = propagator(lambf,
                      muf,
                      buoyancyf,
                      dx,
                      dt,
                      sources_y['amplitude'],
                      sources_x['amplitude'],
                      sources_y['locations'],
                      sources_x['locations'],
                      x_r_y,
                      x_r_x,
                      x_r_p,
                      prop_kwargs=prop_kwargs)
    # Set receiver amplitudes of receiver that will be ignored to zero
    for i in range(num_shots):
        out_filled[-2][i, i].fill_(0)
        out_filled[-1][i, i+1].fill_(0)
        out_filled[-3][i, i+2].fill_(0)

    ((out_filled[-1]**2).sum() + (out_filled[-2]**2).sum() + (out_filled[-3]**2).sum()).backward()

    for ofi, oii in zip(out_filled, out_ignored):
        assert torch.allclose(ofi, oii)

    assert torch.allclose(lambf.grad, lambi.grad)
    assert torch.allclose(muf.grad, mui.grad)
    assert torch.allclose(buoyancyf.grad, buoyancyi.grad)


def run_elasticfunc(nt=3):
    from deepwave.elastic import elastic_func
    torch.manual_seed(1)
    ny = 11
    nx = 12
    n_batch = 2
    dt = 0.0005
    dy = dx = 5
    pml_width = [3, 3, 3, 3]
    n_sources_y_per_shot = 1
    n_sources_x_per_shot = 2
    n_receivers_y_per_shot = 3
    n_receivers_x_per_shot = 4
    n_receivers_p_per_shot = 5
    step_ratio = 1
    accuracy = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    vp = 1500 + 100*torch.rand(ny, nx, dtype=dtype, device=device)
    vs = 1000 + 100*torch.rand(ny, nx, dtype=dtype, device=device)
    rho = 2200 + 100*torch.rand(ny, nx, dtype=dtype, device=device)

    vy = torch.randn(n_batch, ny, nx, dtype=torch.double, device=device)
    vx = torch.randn_like(vy)
    sigmayy = torch.randn_like(vy)
    sigmaxy = torch.randn_like(vy)
    sigmaxx = torch.randn_like(vy)
    m_vyy = torch.randn_like(vy)
    m_vyx = torch.randn_like(vy)
    m_vxy = torch.randn_like(vy)
    m_vxx = torch.randn_like(vy)
    m_sigmayyy = torch.randn_like(vy)
    m_sigmaxyy = torch.randn_like(vy)
    m_sigmaxyx = torch.randn_like(vy)
    m_sigmaxxx = torch.randn_like(vy)
    source_amplitudes_y = torch.randn(nt,
                                      n_batch,
                                      n_sources_y_per_shot,
                                      dtype=torch.double,
                                      device=device)
    source_amplitudes_x = torch.randn(nt,
                                      n_batch,
                                      n_sources_x_per_shot,
                                      dtype=torch.double,
                                      device=device)
    sources_y_i = torch.tensor([[1 * nx + 1], [2 * nx + 2]]).long().to(device)
    sources_x_i = torch.tensor([[3 * nx + 3, 4 * nx + 4],
                                [2 * nx + 2, 4 * nx + 5]]).long().to(device)
    receivers_y_i = torch.tensor([[1 * nx + 1, 2 * nx + 2, 3 * nx + 3],
                                  [4 * nx + 4, 2 * nx + 3,
                                   2 * nx + 1]]).long().to(device)
    receivers_x_i = torch.tensor(
        [[1 * nx + 1, 2 * nx + 2, 3 * nx + 3, 3 * nx + 2],
         [4 * nx + 4, 2 * nx + 3, 2 * nx + 1, 3 * nx + 3]]).long().to(device)
    receivers_p_i = torch.tensor(
        [[1 * nx + 1, 2 * nx + 2, 3 * nx + 3, 3 * nx + 2, 3 * nx + 4],
         [4 * nx + 4, 2 * nx + 3, 2 * nx + 1, 3 * nx + 3,
          2 * nx + 2]]).long().to(device)
    vp.requires_grad_()
    vs.requires_grad_()
    rho.requires_grad_()
    source_amplitudes_y.requires_grad_()
    source_amplitudes_x.requires_grad_()
    vy.requires_grad_()
    vx.requires_grad_()
    sigmayy.requires_grad_()
    sigmaxy.requires_grad_()
    sigmaxx.requires_grad_()
    m_vyy.requires_grad_()
    m_vyx.requires_grad_()
    m_vxy.requires_grad_()
    m_vxx.requires_grad_()
    m_sigmayyy.requires_grad_()
    m_sigmaxyy.requires_grad_()
    m_sigmaxyx.requires_grad_()
    m_sigmaxxx.requires_grad_()
    ay = torch.randn(ny, dtype=torch.double, device=device)
    ayh = torch.randn(ny, dtype=torch.double, device=device)
    ax = torch.randn(nx, dtype=torch.double, device=device)
    axh = torch.randn(nx, dtype=torch.double, device=device)
    by = torch.randn(ny, dtype=torch.double, device=device)
    byh = torch.randn(ny, dtype=torch.double, device=device)
    bx = torch.randn(nx, dtype=torch.double, device=device)
    bxh = torch.randn(nx, dtype=torch.double, device=device)
    ay[1 + pml_width[0]:ny - pml_width[1]].fill_(0)
    by[1 + pml_width[0]:ny - pml_width[1]].fill_(0)
    ax[pml_width[2]:nx - pml_width[3]].fill_(0)
    bx[pml_width[2]:nx - pml_width[3]].fill_(0)
    ayh[pml_width[0]:ny - pml_width[1]].fill_(0)
    byh[pml_width[0]:ny - pml_width[1]].fill_(0)
    axh[pml_width[2]:nx - 1 - pml_width[3]].fill_(0)
    bxh[pml_width[2]:nx - 1 - pml_width[3]].fill_(0)

    def wrap(vp, vs, rho, *args):
        o = list(elastic_func(*vpvsrho_to_lambmubuoyancy(vp, vs, rho), *args))
        for i in [2, 3, 4, 9, 10, 11, 12, 13]:
            o[i] /= 1e6
        return o

    torch.autograd.gradcheck(
        wrap,
        (vp, vs, rho, source_amplitudes_y, source_amplitudes_x, vy, vx,
         sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx, m_sigmayyy,
         m_sigmaxyy, m_sigmaxyx, m_sigmaxxx, ay, ayh, ax, axh, by, byh, bx,
         bxh, sources_y_i, sources_x_i, receivers_y_i, receivers_x_i,
         receivers_p_i, dy, dx, dt, nt, step_ratio, accuracy, pml_width,
         n_batch))


def test_elasticfunc():
    run_elasticfunc(nt=1)
    run_elasticfunc(nt=2)
    run_elasticfunc(nt=3)


def test_gradcheck_2d():
    """Test gradcheck in a 2D model."""
    run_gradcheck_2d(propagator=elasticprop)


def test_gradcheck_2d_2nd_order():
    """Test gradcheck with a 2nd order accurate propagator."""
    run_gradcheck_2d(propagator=elasticprop, prop_kwargs={'accuracy': 2})


def test_gradcheck_2d_cfl():
    """Test gradcheck with a timestep greater than the CFL limit."""
    run_gradcheck_2d(propagator=elasticprop,
                     dt=0.002,
                     prop_kwargs={'time_pad_frac': 0.2})


def test_gradcheck_2d_odd_timesteps():
    """Test gradcheck with one more timestep."""
    run_gradcheck_2d(propagator=elasticprop, nt_add=1)


def test_gradcheck_2d_one_shot():
    """Test gradcheck with one shot."""
    run_gradcheck_2d(propagator=elasticprop, num_shots=1)


def test_gradcheck_2d_survey_pad():
    """Test gradcheck with survey_pad."""
    run_gradcheck_2d(propagator=elasticprop, nx=(12, 12), survey_pad=4)


def test_gradcheck_2d_chained():
    """Test gradcheck when two propagators are chained."""
    run_gradcheck_2d(propagator=elasticpropchained)


def test_gradcheck_2d_different_pml():
    """Test gradcheck with different pml widths."""
    run_gradcheck_2d(propagator=elasticprop, pml_width=[0, 1, 5, 10])


def test_gradcheck_2d_no_pml():
    """Test gradcheck with no pml."""
    run_gradcheck_2d(propagator=elasticprop, pml_width=0, nx=(11, 11))


def test_gradcheck_2d_different_dx():
    """Test gradcheck with different dx values."""
    run_gradcheck_2d(propagator=elasticprop, dx=(4, 5))


def test_gradcheck_only_lamb_2d():
    """Test gradcheck with only lamb requiring gradient."""
    run_gradcheck_2d(
        propagator=elasticprop,
        lamb_requires_grad=True,
        mu_requires_grad=False,
        buoyancy_requires_grad=False,
        source_requires_grad=False,
    )


def test_gradcheck_only_mu_2d():
    """Test gradcheck with only mu requiring gradient."""
    run_gradcheck_2d(
        propagator=elasticprop,
        lamb_requires_grad=False,
        mu_requires_grad=True,
        buoyancy_requires_grad=False,
        source_requires_grad=False,
    )

def test_gradcheck_batched_lamb_2d():
    """Test gradcheck in a 2D model with batched lamb."""
    run_gradcheck_2d(propagator=elasticprop,
                     lamb=torch.tensor([[[DEFAULT_LAMB]],[[DEFAULT_LAMB*1.2]]]))

def test_gradcheck_batched_mu_2d():
    """Test gradcheck in a 2D model with batched mu."""
    run_gradcheck_2d(propagator=elasticprop,
                     mu=torch.tensor([[[DEFAULT_MU]],[[DEFAULT_MU*1.2]]]))

def test_gradcheck_batched_buoyancy_2d():
    """Test gradcheck in a 2D model with batched buoyancy."""
    run_gradcheck_2d(propagator=elasticprop,
                     buoyancy=torch.tensor([[[DEFAULT_BUOYANCY]],[[DEFAULT_BUOYANCY*1.2]]]))

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
        coords[..., 0] = (nx[0] - 1).float() - coords[..., 0]
    elif location == 'middle':
        coords[..., 0] += int(nx[0] / 2)
    else:
        raise ValueError("unsupported location")

    for dim in range(1, ndim):
        coords[..., dim] = torch.div(nx[dim], 2)

    return coords.long()


def run_forward_lamb(orientation=0,
                     prop_kwargs=None,
                     device=None,
                     dtype=None,
                     pml_width=20,
                     **kwargs):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny = 20
    nx = 20
    nt = 3000
    dt = 0.0005
    dx = 4
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(
        1500 * torch.ones(1, dtype=dtype, device=device),
        1000.0 * torch.ones(1, dtype=dtype, device=device),
        2200 * torch.ones(1, dtype=dtype, device=device))
    lamb = torch.ones(ny, nx, dtype=dtype, device=device) * lamb
    mu = torch.ones(ny, nx, dtype=dtype, device=device) * mu
    buoyancy = torch.ones(ny, nx, dtype=dtype, device=device) * buoyancy
    freq = 14.5
    peak_time = 0.08
    source_amplitudes = ricker(freq, nt, dt, peak_time,
                               dtype).reshape(1, 1, -1)
    if orientation == 0:
        x_s_y = torch.tensor([[[15, 0]]])
        x_s_x = None
        sa_y = source_amplitudes
        sa_x = None
        x_r_y = torch.tensor([[[5, 15]]])
        x_r_x = torch.tensor([[[6, 16]]])
        pml_width = [0, pml_width, pml_width, pml_width]
    elif orientation == 1:
        x_s_y = torch.tensor([[[ny - 1 - 15, nx - 2 - 0]]])
        x_s_x = None
        sa_y = source_amplitudes
        sa_x = None
        x_r_y = torch.tensor([[[ny - 1 - 5, nx - 2 - 15]]])
        x_r_x = torch.tensor([[[ny - 1 - 5, nx - 2 - 15]]])
        pml_width = [pml_width, 0, pml_width, pml_width]
    elif orientation == 2:
        x_s_y = None
        x_s_x = torch.tensor([[[1, 15]]])
        sa_y = None
        sa_x = source_amplitudes
        x_r_y = torch.tensor([[[16, 5]]])
        x_r_x = torch.tensor([[[16, 5]]])
        pml_width = [pml_width, pml_width, 0, pml_width]
    else:
        x_s_y = None
        x_s_x = torch.tensor([[[ny - 1 - 1, nx - 1 - 15]]])
        sa_y = None
        sa_x = source_amplitudes
        x_r_y = torch.tensor([[[ny - 1 - 17, nx - 1 - 6]]])
        x_r_x = torch.tensor([[[ny - 1 - 16, nx - 1 - 5]]])
        pml_width = [pml_width, pml_width, pml_width, 0]
    return elasticprop(lamb,
                       mu,
                       buoyancy,
                       dx,
                       dt,
                       sa_y,
                       sa_x,
                       x_s_y,
                       x_s_x,
                       x_r_y,
                       x_r_x,
                       x_r_x,
                       prop_kwargs=prop_kwargs,
                       pml_width=pml_width,
                       **kwargs)


def run_forward(mlamb,
                mmu,
                mbuoyancy,
                freq,
                dx,
                dt,
                nx,
                num_shots,
                num_sources_per_shot,
                num_receivers_per_shot,
                propagator,
                prop_kwargs,
                device=None,
                dtype=None,
                dlamb=DEFAULT_LAMB / 10,
                dmu=DEFAULT_MU / 10,
                dbuoyancy=DEFAULT_BUOYANCY / 10,
                nt=None,
                dpeak_time=0.3,
                **kwargs):
    """Create a random model and forward propagate.
    """
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lamb = (torch.ones(*nx, device=device, dtype=dtype) * mlamb +
            torch.randn(*nx, dtype=dtype).to(device) * dlamb)
    mu = (torch.ones(*nx, device=device, dtype=dtype) * mmu +
          torch.randn(*nx, dtype=dtype).to(device) * dmu)
    buoyancy = (torch.ones(*nx, device=device, dtype=dtype) * mbuoyancy +
                torch.randn(*nx, dtype=dtype).to(device) * dbuoyancy)

    nx = torch.Tensor(nx).long()
    dx = torch.Tensor(dx)

    vp, vs, rho = lambmubuoyancy_to_vpvsrho(lamb.abs(), mu.abs(),
                                            buoyancy.abs())
    vmin = min(vp.abs().min(), vs.abs().min())

    if nt is None:
        nt = int(
            (2 * torch.norm(nx.float() * dx) / vmin + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, 2 * num_sources_per_shot, nx)
    x_s_y = x_s[:, :num_sources_per_shot]
    x_s_x = x_s[:, num_sources_per_shot:]
    x_r = _set_coords(num_shots, 2 * num_receivers_per_shot, nx, 'bottom')
    x_r_y = x_r[:, :num_receivers_per_shot]
    x_r_x = x_r[:, num_receivers_per_shot:]
    sources_y = _set_sources(x_s_y, freq, dt, nt, dtype, dpeak_time=dpeak_time)
    sources_x = _set_sources(x_s_x, freq, dt, nt, dtype, dpeak_time=dpeak_time)

    return propagator(lamb,
                      mu,
                      buoyancy,
                      dx,
                      dt,
                      sources_y['amplitude'],
                      sources_x['amplitude'],
                      sources_y['locations'],
                      sources_x['locations'],
                      x_r_y,
                      x_r_x,
                      x_r_x,
                      prop_kwargs=prop_kwargs,
                      **kwargs)


def run_forward_2d(lamb=DEFAULT_LAMB,
                   mu=DEFAULT_MU,
                   buoyancy=DEFAULT_BUOYANCY,
                   freq=25,
                   dx=(5, 5),
                   dt=0.004,
                   nx=(50, 50),
                   num_shots=2,
                   num_sources_per_shot=2,
                   num_receivers_per_shot=2,
                   propagator=None,
                   prop_kwargs=None,
                   device=None,
                   dtype=None,
                   **kwargs):
    """Runs run_forward with default parameters for 2D."""

    return run_forward(lamb,
                       mu,
                       buoyancy,
                       freq,
                       dx,
                       dt,
                       nx,
                       num_shots,
                       num_sources_per_shot,
                       num_receivers_per_shot,
                       propagator,
                       prop_kwargs,
                       device=device,
                       dtype=dtype,
                       **kwargs)


def run_gradcheck(mlamb,
                  mmu,
                  mbuoyancy,
                  freq,
                  dx,
                  dt,
                  nx,
                  num_shots,
                  num_sources_per_shot,
                  num_receivers_per_shot,
                  propagator,
                  prop_kwargs,
                  pml_width=3,
                  survey_pad=None,
                  device=None,
                  dtype=None,
                  dlamb=DEFAULT_LAMB / 10,
                  dmu=DEFAULT_MU / 10,
                  dbuoyancy=DEFAULT_BUOYANCY / 10,
                  lamb_requires_grad=True,
                  mu_requires_grad=True,
                  buoyancy_requires_grad=True,
                  source_requires_grad=True,
                  nt_add=0,
                  atol=1e-5,
                  rtol=1e-3):
    """Run PyTorch's gradcheck to test the gradient."""
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(mlamb, torch.Tensor):
        mlamb = mlamb.to(device)
    if isinstance(mmu, torch.Tensor):
        mmu = mmu.to(device)
    if isinstance(mbuoyancy, torch.Tensor):
        mbuoyancy = mbuoyancy.to(device)
    lamb = torch.ones(*nx, device=device, dtype=dtype) * mlamb
    lamb += torch.randn(*lamb.shape, dtype=dtype).to(device) * dlamb
    mu = torch.ones(*nx, device=device, dtype=dtype) * mmu
    mu += torch.randn(*mu.shape, dtype=dtype).to(device) * dmu
    buoyancy = torch.ones(*nx, device=device, dtype=dtype) * mbuoyancy
    buoyancy += torch.randn(*buoyancy.shape, dtype=dtype).to(device) * dbuoyancy

    nx = torch.Tensor(nx).long()
    dx = torch.Tensor(dx)

    vp, vs, rho = lambmubuoyancy_to_vpvsrho(lamb.abs(), mu.abs(),
                                            buoyancy.abs())
    vmin = min(vp.abs().min(), vs.abs().min())

    if vmin != 0:
        nt = int(
            (2 * torch.norm(nx.float() * dx) / vmin + 0.1 + 2 / freq) / dt)
    else:
        nt = int(
            (2 * torch.norm(nx.float() * dx) / 1500 + 0.1 + 2 / freq) / dt)
    nt += nt_add
    if num_sources_per_shot > 0:
        x_s = _set_coords(num_shots, 2 * num_sources_per_shot, nx)
        x_s_y = x_s[:, :num_sources_per_shot]
        x_s_x = x_s[:, num_sources_per_shot:]
        sources_y = _set_sources(x_s_y, freq, dt, nt, dtype, dpeak_time=0.05)
        sources_x = _set_sources(x_s_x, freq, dt, nt, dtype, dpeak_time=0.05)
        sources_y['amplitude'] = sources_y['amplitude'].to(device)
        sources_x['amplitude'] = sources_x['amplitude'].to(device)
        sources_y['amplitude'].requires_grad_(source_requires_grad)
        sources_x['amplitude'].requires_grad_(source_requires_grad)
        nt = None
    else:
        sources_y = {'amplitude': None, 'locations': None}
        sources_x = {'amplitude': None, 'locations': None}
    if num_receivers_per_shot > 0:
        x_r = _set_coords(num_shots, 2 * num_receivers_per_shot, nx)
        x_r_y = x_r[:, :num_receivers_per_shot]
        x_r_x = x_r[:, num_receivers_per_shot:]
    else:
        x_r_y = None
        x_r_x = None
    if isinstance(pml_width, int):
        pml_width = [pml_width for _ in range(4)]

    if isinstance(mlamb, torch.Tensor):
        mlamb = mlamb.to(device)
        min_mlamb = mlamb.abs().min().item()
    else:
        min_mlamb = mlamb
    if isinstance(mmu, torch.Tensor):
        mmu = mmu.to(device)
        min_mmu = mmu.abs().min().item()
    else:
        min_mmu = mmu
    if isinstance(mbuoyancy, torch.Tensor):
        mbuoyancy = mbuoyancy.to(device)
        min_mbuoyancy = mbuoyancy.abs().min().item()
    else:
        min_mbuoyancy = mbuoyancy

    if min_mlamb != 0:
        lamb /= mlamb
    if min_mmu != 0:
        mu /= mmu
    if min_mbuoyancy != 0:
        buoyancy /= mbuoyancy

    lamb.requires_grad_(lamb_requires_grad)
    mu.requires_grad_(mu_requires_grad)
    buoyancy.requires_grad_(buoyancy_requires_grad)

    def wrap(lamb, mu, buoyancy, sources_y_amplitude, sources_x_amplitude):
        if sources_y_amplitude is not None and min_mbuoyancy != 0:
            sources_y_amplitude = sources_y_amplitude / mbuoyancy / dt
        if sources_x_amplitude is not None and min_mbuoyancy != 0:
            sources_x_amplitude = sources_x_amplitude / mbuoyancy / dt
        out = propagator(lamb * mlamb,
                         mu * mmu,
                         buoyancy * mbuoyancy,
                         dx,
                         dt,
                         sources_y_amplitude,
                         sources_x_amplitude,
                         sources_y['locations'],
                         sources_x['locations'],
                         x_r_y,
                         x_r_x,
                         x_r_x,
                         prop_kwargs,
                         pml_width,
                         survey_pad,
                         nt=nt)
        return (out[-3] / 1e6, out[-2], out[-1])

    torch.autograd.gradcheck(
        wrap,
        (lamb, mu, buoyancy, sources_y['amplitude'], sources_x['amplitude']),
        nondet_tol=1e-3,
        check_grad_dtypes=True,
        atol=atol,
        rtol=rtol)


def run_gradcheck_2d(lamb=DEFAULT_LAMB,
                     mu=DEFAULT_MU,
                     buoyancy=DEFAULT_BUOYANCY,
                     freq=25,
                     dx=(5, 5),
                     dt=0.001,
                     nx=(10, 10),
                     num_shots=2,
                     num_sources_per_shot=2,
                     num_receivers_per_shot=2,
                     propagator=None,
                     prop_kwargs=None,
                     pml_width=3,
                     survey_pad=None,
                     device=None,
                     dtype=torch.double,
                     **kwargs):
    """Runs run_gradcheck with default parameters for 2D."""

    return run_gradcheck(lamb,
                         mu,
                         buoyancy,
                         freq,
                         dx,
                         dt,
                         nx,
                         num_shots,
                         num_sources_per_shot,
                         num_receivers_per_shot,
                         propagator,
                         prop_kwargs,
                         pml_width=pml_width,
                         survey_pad=survey_pad,
                         device=device,
                         dtype=dtype,
                         **kwargs)
