"""Front-end for the spike propagators.

The spike replaces the *compiled kernel layer* only.  Everything above it --
survey extraction, model padding, CFL time-step resampling, source/receiver
index arithmetic, PML profile construction -- is Deepwave's own Python
front-end, imported and reused unchanged, so a parity comparison isolates the
kernel rather than measuring front-end differences.  Every front-end step that
touches a differentiable input (the ``-v^2 dt^2`` source scaling, the model
padding) is an ordinary torch op, so autograd handles it.
"""

from __future__ import annotations

from typing import Optional

import torch

from deepwave import _common, _engine, _regular_grid

from . import ops


def _flat(profiles: list[torch.Tensor]) -> list[torch.Tensor]:
    return [p.reshape(-1).contiguous() for p in profiles]


def scalar(
    v: torch.Tensor,
    grid_spacing,
    dt: float,
    source_amplitudes: torch.Tensor,
    source_locations: torch.Tensor,
    receiver_locations: torch.Tensor,
    *,
    accuracy: int = 4,
    pml_width=20,
    pml_freq: Optional[float] = None,
    max_vel: Optional[float] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    n_threads: int = 1,
    ndim: Optional[int] = None,
):
    if ndim is None:
        ndim = v.dim()
    dtype_tag = {torch.float32: "f32", torch.float64: "f64"}[v.dtype]
    device_tag = "cpu" if v.device.type == "cpu" else "cuda"

    min_nz, max_mv = _common.velocity_extremes(v, max_vel)
    fd_pad = [accuracy // 2] * 2 * ndim
    (
        models,
        src_amp_out,
        wavefields,
        sources_i,
        receivers_i,
        grid_spacing_r,
        dt_i,
        nt_i,
        n_shots,
        step_ratio,
        mgsi,
        accuracy,
        pml_width,
        pml_freq,
        max_vel,
        _ftf,
        _tpf,
        _tt,
        device,
        dtype,
        _courant,
        _origin,
        _extent,
    ) = _common.setup_propagator(
        [v],
        ["replicate"],
        ["v"],
        grid_spacing,
        dt,
        [source_amplitudes],
        [source_locations],
        [receiver_locations],
        accuracy,
        fd_pad,
        pml_width,
        pml_freq,
        max_vel,
        min_nz,
        max_mv,
        None,
        [None] * (2 + 2 * ndim),
        None,
        nt,
        model_gradient_sampling_interval,
        0.0,
        0.0,
        False,
        ndim,
    )
    model_shape = tuple(int(s) for s in models[0].shape[-ndim:])
    src_amp_out[0] = (
        -src_amp_out[0]
        * _common.gather_model_at_sources(models[0], sources_i[0], model_shape, n_shots)
        ** 2
        * dt_i**2
    )
    profiles = _flat(
        _regular_grid.set_pml_profiles(
            pml_width,
            accuracy,
            fd_pad,
            dt_i,
            grid_spacing_r,
            max_vel,
            dtype,
            device,
            pml_freq,
            model_shape,
        )
    )

    key = f"scalar|{ndim}|{accuracy}|{dtype_tag}|{device_tag}"
    spec = ops._model(key).spec
    state0 = [
        torch.zeros((n_shots, *model_shape), device=device, dtype=dtype)
        for _ in spec.state
    ]
    pml_b, pml_e = _engine._pml_region(
        False, "forward", model_shape, pml_width, accuracy // 2, ndim
    )
    pml_bb, pml_eb = _engine._pml_region(
        False, "backward", model_shape, pml_width, accuracy // 2, ndim
    )
    flts = [float(dt_i**2)]
    flts += [1.0 / float(g) for g in grid_spacing_r]
    flts += [1.0 / float(g) ** 2 for g in grid_spacing_r]
    need_store = bool(models[0].requires_grad)

    def _ints(pb, pe):
        return [
            int(nt_i),
            int(n_shots),
            *model_shape,
            1 if models[0].shape[0] > 1 else 0,
            int(step_ratio * mgsi),
            1,
            int(sources_i[0].shape[1]),
            int(receivers_i[0].shape[1]),
            *[int(x) for x in pb],
            *[int(x) for x in pe],
            int(need_store),
            int(n_threads),
        ]

    ints = _ints(pml_b, pml_e)
    ints_bwd = _ints(pml_bb, pml_eb)
    out = torch.ops.dwspike.fwd(
        [models[0].contiguous()],
        profiles,
        state0,
        [src_amp_out[0].contiguous()],
        [sources_i[0].contiguous()],
        [receivers_i[0].contiguous()],
        key,
        ints,
        ints_bwd,
        flts,
    )
    return _common.downsample_and_movedim(out[0], step_ratio, 0.0, 0.0, False)


def elastic(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    grid_spacing,
    dt: float,
    *,
    source_amplitudes_y=None,
    source_amplitudes_x=None,
    source_amplitudes_p=None,
    source_locations_y=None,
    source_locations_x=None,
    source_locations_p=None,
    receiver_locations_y=None,
    receiver_locations_x=None,
    receiver_locations_p=None,
    accuracy: int = 4,
    pml_width=20,
    pml_freq: Optional[float] = None,
    max_vel: Optional[float] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    n_threads: int = 1,
):
    from deepwave import _staggered_grid
    from deepwave.elastic import prepare_parameters

    ndim = 2
    dtype_tag = {torch.float32: "f32", torch.float64: "f64"}[lamb.dtype]
    device_tag = "cpu" if lamb.device.type == "cpu" else "cuda"

    vp, vs, _rho = _common.lambmubuoyancy_to_vpvsrho(lamb, mu, buoyancy)
    max_model_vel = max(vp.abs().max().item(), vs.abs().max().item())
    nz = vp[vp != 0]
    min_nz = float(nz.abs().min().item()) if nz.numel() else 0.0
    fd_pad = [accuracy // 2, accuracy // 2 - 1] * ndim
    (
        models,
        src_out,
        wavefields,
        sources_i,
        receivers_i,
        gs,
        dt_i,
        nt_i,
        n_shots,
        step_ratio,
        mgsi,
        accuracy,
        pml_width,
        pml_freq,
        max_vel,
        _a,
        _b,
        _c,
        device,
        dtype,
        _d,
        _e,
        _f,
    ) = _common.setup_propagator(
        [lamb, mu, buoyancy],
        ["replicate"] * 3,
        ["lamb", "mu", "buoyancy"],
        grid_spacing,
        dt,
        [source_amplitudes_y, source_amplitudes_x, source_amplitudes_p],
        [source_locations_y, source_locations_x, source_locations_p],
        [receiver_locations_y, receiver_locations_x, receiver_locations_p],
        accuracy,
        fd_pad,
        pml_width,
        pml_freq,
        max_vel,
        min_nz,
        max_model_vel,
        None,
        [None] * 13,
        None,
        nt,
        model_gradient_sampling_interval,
        0.0,
        0.0,
        False,
        ndim,
        staggered=True,
    )
    model_shape = tuple(int(s) for s in models[0].shape[-ndim:])
    mu_yx, buoy_y, buoy_x = prepare_parameters(models[1], models[2])
    params = [models[0], models[1], mu_yx, buoy_y, buoy_x]

    if src_out[0] is not None and src_out[0].numel():
        src_out[0] = (
            src_out[0]
            * _common.gather_model_at_sources(
                buoy_y, sources_i[0], model_shape, n_shots
            )
            * dt_i
        )
    if src_out[1] is not None and src_out[1].numel():
        src_out[1] = (
            src_out[1]
            * _common.gather_model_at_sources(
                buoy_x, sources_i[1], model_shape, n_shots
            )
            * dt_i
        )
    if src_out[2] is not None and src_out[2].numel():
        src_out[2] = -src_out[2] * dt_i

    profiles = _flat(
        _staggered_grid.set_pml_profiles(
            pml_width,
            accuracy,
            fd_pad,
            dt_i,
            gs,
            max_vel,
            dtype,
            device,
            pml_freq,
            model_shape,
        )
    )
    key = f"elastic|2|{accuracy}|{dtype_tag}|{device_tag}"
    spec = ops._model(key).spec
    state0 = [
        torch.zeros((n_shots, *model_shape), device=device, dtype=dtype)
        for _ in spec.state
    ]
    pb, pe = _engine._pml_region(
        True, "forward", model_shape, pml_width, accuracy // 2, ndim
    )
    pbb, peb = _engine._pml_region(
        True, "backward", model_shape, pml_width, accuracy // 2, ndim
    )
    flts = [float(dt_i), 1.0 / float(gs[0]), 1.0 / float(gs[1])]
    need_store = any(mm.requires_grad for mm in models)

    def _count(t):
        return 0 if t is None or t.numel() == 0 else int(t.shape[1])

    def _ints(b, e):
        return [
            int(nt_i),
            int(n_shots),
            *model_shape,
            1 if models[0].shape[0] > 1 else 0,
            int(step_ratio * mgsi),
            1,
            *[_count(sources_i[k]) for k in range(3)],
            *[_count(receivers_i[k]) for k in range(3)],
            *[int(x) for x in b],
            *[int(x) for x in e],
            int(need_store),
            int(n_threads),
        ]

    def _t(t, dt_):
        if t is None or t.numel() == 0:
            return torch.zeros(0, device=device, dtype=dt_)
        return t.contiguous()

    out = torch.ops.dwspike.fwd(
        [p.contiguous() for p in params],
        profiles,
        state0,
        [_t(src_out[k], dtype) for k in range(3)],
        [_t(sources_i[k], torch.int64) for k in range(3)],
        [_t(receivers_i[k], torch.int64) for k in range(3)],
        key,
        _ints(pb, pe),
        _ints(pbb, peb),
        flts,
    )
    ry, rx, rp = (
        _common.downsample_and_movedim(out[k], step_ratio, 0.0, 0.0, False)
        for k in range(3)
    )
    return -rp / ndim, ry, rx
