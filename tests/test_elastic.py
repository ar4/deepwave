"""Tests for deepwave.elastic."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest
import torch
from test_utils import (
    _set_coords,
    _set_sources,
    direct_1d,
    direct_2d_approx,
    direct_3d,
    run_reciprocity_check,
    scattered,
)

import deepwave
from deepwave import IGNORE_LOCATION, Elastic, elastic
from deepwave.backend_utils import USE_OPENMP
from deepwave.common import (
    cfl_condition,
    downsample,
    lambmubuoyancy_to_vpvsrho,
    upsample,
    vpvsrho_to_lambmubuoyancy,
)
from deepwave.wavelets import ricker

torch._dynamo.config.cache_size_limit = 256  # noqa: SLF001
torch.autograd.set_detect_anomaly(True)

DEFAULT_LAMB = 550000000
DEFAULT_MU = 2200000000
DEFAULT_BUOYANCY = 1 / 2200


def elasticprop(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    dx: Union[float, List[float]],
    dt: float,
    source_amplitudes_z: Optional[torch.Tensor] = None,
    source_amplitudes_y: Optional[torch.Tensor] = None,
    source_amplitudes_x: Optional[torch.Tensor] = None,
    source_amplitudes_p: Optional[torch.Tensor] = None,
    source_locations_z: Optional[torch.Tensor] = None,
    source_locations_y: Optional[torch.Tensor] = None,
    source_locations_x: Optional[torch.Tensor] = None,
    source_locations_p: Optional[torch.Tensor] = None,
    receiver_locations_z: Optional[torch.Tensor] = None,
    receiver_locations_y: Optional[torch.Tensor] = None,
    receiver_locations_x: Optional[torch.Tensor] = None,
    receiver_locations_p: Optional[torch.Tensor] = None,
    prop_kwargs: Optional[Dict[str, Any]] = None,
    pml_width: Optional[Union[int, List[int]]] = None,
    survey_pad: Optional[Union[int, List[int]]] = None,
    vz_0: Optional[torch.Tensor] = None,
    vy_0: Optional[torch.Tensor] = None,
    vx_0: Optional[torch.Tensor] = None,
    sigmazz_0: Optional[torch.Tensor] = None,
    sigmayz_0: Optional[torch.Tensor] = None,
    sigmaxz_0: Optional[torch.Tensor] = None,
    sigmayy_0: Optional[torch.Tensor] = None,
    sigmaxy_0: Optional[torch.Tensor] = None,
    sigmaxx_0: Optional[torch.Tensor] = None,
    m_vzz_0: Optional[torch.Tensor] = None,
    m_vzy_0: Optional[torch.Tensor] = None,
    m_vzx_0: Optional[torch.Tensor] = None,
    m_vyz_0: Optional[torch.Tensor] = None,
    m_vxz_0: Optional[torch.Tensor] = None,
    m_vyy_0: Optional[torch.Tensor] = None,
    m_vyx_0: Optional[torch.Tensor] = None,
    m_vxy_0: Optional[torch.Tensor] = None,
    m_vxx_0: Optional[torch.Tensor] = None,
    m_sigmazzz_0: Optional[torch.Tensor] = None,
    m_sigmayzy_0: Optional[torch.Tensor] = None,
    m_sigmaxzx_0: Optional[torch.Tensor] = None,
    m_sigmayzz_0: Optional[torch.Tensor] = None,
    m_sigmaxzz_0: Optional[torch.Tensor] = None,
    m_sigmayyy_0: Optional[torch.Tensor] = None,
    m_sigmaxyy_0: Optional[torch.Tensor] = None,
    m_sigmaxyx_0: Optional[torch.Tensor] = None,
    m_sigmaxxx_0: Optional[torch.Tensor] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    functional: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """Wraps the elastic propagator for testing purposes."""
    if prop_kwargs is None:
        prop_kwargs = {}
    # For consistency when actual max speed changes
    prop_kwargs["max_vel"] = 2000

    # Workaround for gradcheck not accepting prop_kwargs dictionary
    if pml_width is not None:
        prop_kwargs["pml_width"] = pml_width
    prop_kwargs["survey_pad"] = survey_pad

    device = lamb.device
    if source_amplitudes_z is not None:
        source_amplitudes_z = source_amplitudes_z.to(device)
        source_locations_z = source_locations_z.to(device)
    if source_amplitudes_y is not None:
        source_amplitudes_y = source_amplitudes_y.to(device)
        source_locations_y = source_locations_y.to(device)
    if source_amplitudes_x is not None:
        source_amplitudes_x = source_amplitudes_x.to(device)
        source_locations_x = source_locations_x.to(device)
    if source_amplitudes_p is not None:
        source_amplitudes_p = source_amplitudes_p.to(device)
        source_locations_p = source_locations_p.to(device)
    if receiver_locations_z is not None:
        receiver_locations_z = receiver_locations_z.to(device)
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
            source_amplitudes_z=source_amplitudes_z,
            source_amplitudes_y=source_amplitudes_y,
            source_amplitudes_x=source_amplitudes_x,
            source_amplitudes_p=source_amplitudes_p,
            source_locations_z=source_locations_z,
            source_locations_y=source_locations_y,
            source_locations_x=source_locations_x,
            source_locations_p=source_locations_p,
            receiver_locations_z=receiver_locations_z,
            receiver_locations_y=receiver_locations_y,
            receiver_locations_x=receiver_locations_x,
            receiver_locations_p=receiver_locations_p,
            vz_0=vz_0,
            vy_0=vy_0,
            vx_0=vx_0,
            sigmazz_0=sigmazz_0,
            sigmayz_0=sigmayz_0,
            sigmaxz_0=sigmaxz_0,
            sigmayy_0=sigmayy_0,
            sigmaxy_0=sigmaxy_0,
            sigmaxx_0=sigmaxx_0,
            m_vzz_0=m_vzz_0,
            m_vzy_0=m_vzy_0,
            m_vzx_0=m_vzx_0,
            m_vyz_0=m_vyz_0,
            m_vxz_0=m_vxz_0,
            m_vyy_0=m_vyy_0,
            m_vyx_0=m_vyx_0,
            m_vxy_0=m_vxy_0,
            m_vxx_0=m_vxx_0,
            m_sigmazzz_0=m_sigmazzz_0,
            m_sigmayzy_0=m_sigmayzy_0,
            m_sigmaxzx_0=m_sigmaxzx_0,
            m_sigmayzz_0=m_sigmayzz_0,
            m_sigmaxzz_0=m_sigmaxzz_0,
            m_sigmayyy_0=m_sigmayyy_0,
            m_sigmaxyy_0=m_sigmaxyy_0,
            m_sigmaxyx_0=m_sigmaxyx_0,
            m_sigmaxxx_0=m_sigmaxxx_0,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            **prop_kwargs,
        )

    prop = Elastic(lamb, mu, buoyancy, dx)
    return prop(
        dt,
        source_amplitudes_z=source_amplitudes_z,
        source_amplitudes_y=source_amplitudes_y,
        source_amplitudes_x=source_amplitudes_x,
        source_amplitudes_p=source_amplitudes_p,
        source_locations_z=source_locations_z,
        source_locations_y=source_locations_y,
        source_locations_x=source_locations_x,
        source_locations_p=source_locations_p,
        receiver_locations_z=receiver_locations_z,
        receiver_locations_y=receiver_locations_y,
        receiver_locations_x=receiver_locations_x,
        receiver_locations_p=receiver_locations_p,
        vz_0=vz_0,
        vy_0=vy_0,
        vx_0=vx_0,
        sigmazz_0=sigmazz_0,
        sigmayz_0=sigmayz_0,
        sigmaxz_0=sigmaxz_0,
        sigmayy_0=sigmayy_0,
        sigmaxy_0=sigmaxy_0,
        sigmaxx_0=sigmaxx_0,
        m_vzz_0=m_vzz_0,
        m_vzy_0=m_vzy_0,
        m_vzx_0=m_vzx_0,
        m_vyz_0=m_vyz_0,
        m_vxz_0=m_vxz_0,
        m_vyy_0=m_vyy_0,
        m_vyx_0=m_vyx_0,
        m_vxy_0=m_vxy_0,
        m_vxx_0=m_vxx_0,
        m_sigmazzz_0=m_sigmazzz_0,
        m_sigmayzy_0=m_sigmayzy_0,
        m_sigmaxzx_0=m_sigmaxzx_0,
        m_sigmayzz_0=m_sigmayzz_0,
        m_sigmaxzz_0=m_sigmaxzz_0,
        m_sigmayyy_0=m_sigmayyy_0,
        m_sigmaxyy_0=m_sigmaxyy_0,
        m_sigmaxyx_0=m_sigmaxyx_0,
        m_sigmaxxx_0=m_sigmaxxx_0,
        nt=nt,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        **prop_kwargs,
    )


def elasticpropchained(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    dx: Union[float, List[float]],
    dt: float,
    source_amplitudes_z: Optional[torch.Tensor] = None,
    source_amplitudes_y: Optional[torch.Tensor] = None,
    source_amplitudes_x: Optional[torch.Tensor] = None,
    source_amplitudes_p: Optional[torch.Tensor] = None,
    source_locations_z: Optional[torch.Tensor] = None,
    source_locations_y: Optional[torch.Tensor] = None,
    source_locations_x: Optional[torch.Tensor] = None,
    source_locations_p: Optional[torch.Tensor] = None,
    receiver_locations_z: Optional[torch.Tensor] = None,
    receiver_locations_y: Optional[torch.Tensor] = None,
    receiver_locations_x: Optional[torch.Tensor] = None,
    receiver_locations_p: Optional[torch.Tensor] = None,
    prop_kwargs: Optional[Dict[str, Any]] = None,
    pml_width: Optional[Union[int, List[int]]] = None,
    survey_pad: Optional[Union[int, List[int]]] = None,
    vz_0: Optional[torch.Tensor] = None,
    vy_0: Optional[torch.Tensor] = None,
    vx_0: Optional[torch.Tensor] = None,
    sigmazz_0: Optional[torch.Tensor] = None,
    sigmayz_0: Optional[torch.Tensor] = None,
    sigmaxz_0: Optional[torch.Tensor] = None,
    sigmayy_0: Optional[torch.Tensor] = None,
    sigmaxy_0: Optional[torch.Tensor] = None,
    sigmaxx_0: Optional[torch.Tensor] = None,
    m_vzz_0: Optional[torch.Tensor] = None,
    m_vzy_0: Optional[torch.Tensor] = None,
    m_vzx_0: Optional[torch.Tensor] = None,
    m_vyz_0: Optional[torch.Tensor] = None,
    m_vxz_0: Optional[torch.Tensor] = None,
    m_vyy_0: Optional[torch.Tensor] = None,
    m_vyx_0: Optional[torch.Tensor] = None,
    m_vxy_0: Optional[torch.Tensor] = None,
    m_vxx_0: Optional[torch.Tensor] = None,
    m_sigmazzz_0: Optional[torch.Tensor] = None,
    m_sigmayzy_0: Optional[torch.Tensor] = None,
    m_sigmaxzx_0: Optional[torch.Tensor] = None,
    m_sigmayzz_0: Optional[torch.Tensor] = None,
    m_sigmaxzz_0: Optional[torch.Tensor] = None,
    m_sigmayyy_0: Optional[torch.Tensor] = None,
    m_sigmaxyy_0: Optional[torch.Tensor] = None,
    m_sigmaxyx_0: Optional[torch.Tensor] = None,
    m_sigmaxxx_0: Optional[torch.Tensor] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    functional: bool = True,
    n_chained: int = 2,
) -> Tuple[torch.Tensor, ...]:
    """Wraps multiple elastic propagators chained sequentially for testing purposes."""
    if prop_kwargs is None:
        prop_kwargs = {}
    # For consistency when actual max speed changes
    prop_kwargs["max_vel"] = 2000

    prop_kwargs["pml_freq"] = 25

    # Workaround for gradcheck not accepting prop_kwargs dictionary
    if pml_width is not None:
        prop_kwargs["pml_width"] = pml_width
    prop_kwargs["survey_pad"] = survey_pad

    device = lamb.device
    if source_amplitudes_y is not None:
        source_amplitudes_y = source_amplitudes_y.to(device)
        source_locations_y = source_locations_y.to(device)
    if source_amplitudes_x is not None:
        source_amplitudes_x = source_amplitudes_x.to(device)
        source_locations_x = source_locations_x.to(device)
    if source_amplitudes_p is not None:
        source_amplitudes_p = source_amplitudes_p.to(device)
        source_locations_p = source_locations_p.to(device)
    if receiver_locations_y is not None:
        receiver_locations_y = receiver_locations_y.to(device)
    if receiver_locations_x is not None:
        receiver_locations_x = receiver_locations_x.to(device)
    if receiver_locations_p is not None:
        receiver_locations_p = receiver_locations_p.to(device)

    max_vel = 2000
    dt, step_ratio = cfl_condition(dx[0], dx[1], dt, max_vel)
    source_nt = None
    if (
        source_amplitudes_y is not None
        or source_amplitudes_x is not None
        or source_amplitudes_p is not None
    ):
        if source_amplitudes_y is not None:
            source_amplitudes_y = upsample(source_amplitudes_y, step_ratio)
            source_nt = source_amplitudes_y.shape[-1]
            nt_per_segment = (
                ((source_nt + n_chained - 1) // n_chained + step_ratio - 1)
                // step_ratio
            ) * step_ratio
        if source_amplitudes_x is not None:
            source_amplitudes_x = upsample(source_amplitudes_x, step_ratio)
            source_nt = source_amplitudes_x.shape[-1]
            nt_per_segment = (
                ((source_nt + n_chained - 1) // n_chained + step_ratio - 1)
                // step_ratio
            ) * step_ratio
        if source_amplitudes_p is not None:
            source_amplitudes_p = upsample(source_amplitudes_p, step_ratio)
            source_nt = source_amplitudes_p.shape[-1]
            nt_per_segment = (
                ((source_nt + n_chained - 1) // n_chained + step_ratio - 1)
                // step_ratio
            ) * step_ratio
    else:
        nt *= step_ratio
        nt_per_segment = (
            ((nt + n_chained - 1) // n_chained + step_ratio - 1) // step_ratio
        ) * step_ratio

    vy = vx_0
    vx = vy_0
    sigmayy = sigmayy_0
    sigmaxy = sigmaxy_0
    sigmaxx = sigmaxx_0
    m_vyy = m_vyy_0
    m_vyx = m_vyx_0
    m_vxy = m_vxy_0
    m_vxx = m_vxx_0
    m_sigmayyy = m_sigmayyy_0
    m_sigmaxyy = m_sigmaxyy_0
    m_sigmaxyx = m_sigmaxyx_0
    m_sigmaxxx = m_sigmaxxx_0

    if receiver_locations_y is not None:
        if source_nt is not None:
            receiver_amplitudes_y = torch.zeros(
                receiver_locations_y.shape[0],
                receiver_locations_y.shape[1],
                source_nt,
                dtype=lamb.dtype,
                device=lamb.device,
            )
        else:
            receiver_amplitudes_y = torch.zeros(
                receiver_locations_y.shape[0],
                receiver_locations_y.shape[1],
                nt,
                dtype=lamb.dtype,
                device=lamb.device,
            )

    if receiver_locations_x is not None:
        if source_nt is not None:
            receiver_amplitudes_x = torch.zeros(
                receiver_locations_x.shape[0],
                receiver_locations_x.shape[1],
                source_nt,
                dtype=lamb.dtype,
                device=lamb.device,
            )
        else:
            receiver_amplitudes_x = torch.zeros(
                receiver_locations_x.shape[0],
                receiver_locations_x.shape[1],
                nt,
                dtype=lamb.dtype,
                device=lamb.device,
            )

    if receiver_locations_p is not None:
        if source_nt is not None:
            receiver_amplitudes_p = torch.zeros(
                receiver_locations_p.shape[0],
                receiver_locations_p.shape[1],
                source_nt,
                dtype=lamb.dtype,
                device=lamb.device,
            )
        else:
            receiver_amplitudes_p = torch.zeros(
                receiver_locations_p.shape[0],
                receiver_locations_p.shape[1],
                nt,
                dtype=lamb.dtype,
                device=lamb.device,
            )

    for segment_idx in range(n_chained):
        if (
            source_amplitudes_y is not None
            or source_amplitudes_x is not None
            or source_amplitudes_p is not None
        ):
            if source_amplitudes_y is not None:
                segment_source_amplitudes_y = source_amplitudes_y[
                    ...,
                    nt_per_segment * segment_idx : min(
                        nt_per_segment * (segment_idx + 1),
                        source_nt,
                    ),
                ]
            if source_amplitudes_x is not None:
                segment_source_amplitudes_x = source_amplitudes_x[
                    ...,
                    nt_per_segment * segment_idx : min(
                        nt_per_segment * (segment_idx + 1),
                        source_nt,
                    ),
                ]
            if source_amplitudes_p is not None:
                segment_source_amplitudes_p = source_amplitudes_p[
                    ...,
                    nt_per_segment * segment_idx : min(
                        nt_per_segment * (segment_idx + 1),
                        source_nt,
                    ),
                ]
            segment_nt = None
        else:
            segment_source_amplitudes_y = None
            segment_source_amplitudes_x = None
            segment_source_amplitudes_p = None
            segment_nt = (
                nt_per_segment * (segment_idx + 1) - nt_per_segment * segment_idx
            )
        (
            vy,
            vx,
            sigmayy,
            sigmaxy,
            sigmaxx,
            m_vyy,
            m_vyx,
            m_vxy,
            m_vxx,
            m_sigmayyy,
            m_sigmaxyy,
            m_sigmaxyx,
            m_sigmaxxx,
            segment_receiver_amplitudes_p,
            segment_receiver_amplitudes_y,
            segment_receiver_amplitudes_x,
        ) = elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_y=segment_source_amplitudes_y,
            source_amplitudes_x=segment_source_amplitudes_x,
            source_amplitudes_p=segment_source_amplitudes_p,
            source_locations_y=source_locations_y,
            source_locations_x=source_locations_x,
            source_locations_p=source_locations_p,
            receiver_locations_y=receiver_locations_y,
            receiver_locations_x=receiver_locations_x,
            receiver_locations_p=receiver_locations_p,
            vy_0=vx,
            vx_0=vy,
            sigmayy_0=sigmayy,
            sigmaxy_0=sigmaxy,
            sigmaxx_0=sigmaxx,
            m_vyy_0=m_vyy,
            m_vyx_0=m_vyx,
            m_vxy_0=m_vxy,
            m_vxx_0=m_vxx,
            m_sigmayyy_0=m_sigmayyy,
            m_sigmaxyy_0=m_sigmaxyy,
            m_sigmaxyx_0=m_sigmaxyx,
            m_sigmaxxx_0=m_sigmaxxx,
            nt=segment_nt,
            model_gradient_sampling_interval=step_ratio,
            **prop_kwargs,
        )
        if receiver_locations_y is not None:
            receiver_amplitudes_y[
                ...,
                nt_per_segment * segment_idx : min(
                    nt_per_segment * (segment_idx + 1),
                    receiver_amplitudes_y.shape[-1],
                ),
            ] = segment_receiver_amplitudes_y
        if receiver_locations_x is not None:
            receiver_amplitudes_x[
                ...,
                nt_per_segment * segment_idx : min(
                    nt_per_segment * (segment_idx + 1),
                    receiver_amplitudes_x.shape[-1],
                ),
            ] = segment_receiver_amplitudes_x
        if receiver_locations_p is not None:
            receiver_amplitudes_p[
                ...,
                nt_per_segment * segment_idx : min(
                    nt_per_segment * (segment_idx + 1),
                    receiver_amplitudes_p.shape[-1],
                ),
            ] = segment_receiver_amplitudes_p

    if receiver_locations_y is not None:
        receiver_amplitudes_y = downsample(receiver_amplitudes_y, step_ratio)
    if receiver_locations_x is not None:
        receiver_amplitudes_x = downsample(receiver_amplitudes_x, step_ratio)
    if receiver_locations_p is not None:
        receiver_amplitudes_p = downsample(receiver_amplitudes_p, step_ratio)

    return (
        vy,
        vx,
        sigmayy,
        sigmaxy,
        sigmaxx,
        m_vyy,
        m_vyx,
        m_vxy,
        m_vxx,
        m_sigmayyy,
        m_sigmaxyy,
        m_sigmaxyx,
        m_sigmaxxx,
        receiver_amplitudes_p,
        receiver_amplitudes_y,
        receiver_amplitudes_x,
    )


def test_python_backends() -> None:
    """Verify that Python backends can be called without error."""
    if USE_OPENMP:
        run_forward(propagator=elasticprop, prop_kwargs={"python_backend": "jit"})
        run_forward(propagator=elasticprop, prop_kwargs={"python_backend": "compile"})


def test_forward():
    """Compare the recorded data with that generated by another code."""
    with Path("tests/Uz_file_ascii").open() as f:
        d = f.read().split()
        expected_vy = torch.tensor([float(di) for di in d[:3000]])
    with Path("tests/Ux_file_ascii").open() as f:
        d = f.read().split()
        expected_vx = torch.tensor([float(di) for di in d[3000:]])
    expected_scale = expected_vy.abs().max().item()
    expected_vy /= expected_scale
    expected_vx /= expected_scale
    for accuracy, target_err in zip([2, 4, 6, 8], [0.55, 0.27, 0.34, 0.37]):
        for orientation in range(4):
            actuals_vy = {}
            actuals_vx = {}
            for python in [True, False]:
                out = run_forward_lamb(
                    orientation,
                    prop_kwargs={"python_backend": python, "accuracy": accuracy},
                )
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
                actuals_vy[python] = vy
                actuals_vx[python] = vx
            assert torch.allclose(actuals_vy[True], actuals_vy[False], atol=5e-6)
            assert torch.allclose(actuals_vx[True], actuals_vx[False], atol=5e-6)


def run_direct(
    mlamb: float = DEFAULT_LAMB,
    mmu: float = DEFAULT_MU,
    mbuoyancy: float = DEFAULT_BUOYANCY,
    freq: float = 25,
    dx: Union[float, List[float]] = (5, 5),
    dt: float = 0.0001,
    nx: Tuple[int, ...] = (50, 50),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = elasticprop,
    prop_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create the expected and actual direct pressure waveform at a point."""
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lamb = torch.ones(*nx, device=device, dtype=dtype) * mlamb
    mu = torch.ones(*nx, device=device, dtype=dtype) * mmu
    buoyancy = torch.ones(*nx, device=device, dtype=dtype) * mbuoyancy

    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)
    ndim = len(nx)

    vp, vs, _rho = lambmubuoyancy_to_vpvsrho(lamb.abs(), mu.abs(), buoyancy.abs())
    vmin = min(vp.abs().min().item(), vs.abs().min().item())
    c = vp.abs().min().item()

    if vmin != 0:
        nt = int((2 * torch.norm(nx.float() * dx) / vmin + 0.1 + 2 / freq) / dt)
    else:
        nt = int((2 * torch.norm(nx.float() * dx) / 1500 + 0.1 + 2 / freq) / dt)

    x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist(), "bottom")
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    if len(nx) == 1:
        direct = direct_1d
    elif len(nx) == 2:
        direct = direct_2d_approx
    elif len(nx) == 3:
        direct = direct_3d
    else:
        raise ValueError("unsupported nx")

    expected = torch.zeros(num_shots, num_receivers_per_shot, nt, dtype=dtype)
    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            for receiver in range(num_receivers_per_shot):
                expected[shot, receiver] += direct(
                    x_r[shot, receiver],
                    x_s[shot, source],
                    dx,
                    dt,
                    c,
                    sources["amplitude"][shot, source, :],
                ).to(dtype)
    expected = ((mlamb + 2 / ndim * mmu) * mbuoyancy / c**4) * torch.gradient(
        expected, spacing=dt, dim=-1
    )[0]

    actual = propagator(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_p=sources["amplitude"],
        source_locations_p=sources["locations"],
        receiver_locations_p=x_r,
        prop_kwargs=prop_kwargs,
        **kwargs,
    )[-(ndim + 1)]

    return expected, actual


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((50,), (5,)),
        ((50, 50), (5, 5)),
        ((10, 10, 10), (5, 5, 5)),
    ],
)
def test_direct(nx, dx):
    """Test propagation in a constant medium."""
    actuals = {}
    for python in [True, False]:
        expected, actual = run_direct(
            nx=nx,
            dx=dx,
            propagator=elasticprop,
            dtype=torch.double,
            prop_kwargs={"python_backend": python},
        )
        assert torch.allclose(
            expected, actual.cpu(), atol=0.1 * expected.abs().max().item()
        )
        actuals[python] = actual
    assert torch.allclose(
        actuals[True], actuals[False], atol=1e-14 * actuals[True].abs().max().item()
    )


def run_scatter_acoustic(
    c: float,
    dc: float,
    freq: float,
    dx: Union[float, List[float]],
    dt: float,
    nx: Tuple[int, ...],
    num_shots: int,
    num_sources_per_shot: int,
    num_receivers_per_shot: int,
    propagator: Any,
    prop_kwargs: Optional[Dict[str, Any]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create the expected and actual point scattered acoustic waveform at point."""
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v_const = torch.ones(*nx, device=device, dtype=dtype) * c

    v = v_const.clone()
    nx_tensor = torch.tensor(nx, dtype=torch.long)
    if isinstance(dx, (float, int)):
        dx = [dx]
    dx_tensor = torch.tensor(dx, dtype=dtype)
    ndim = len(nx)

    nt = int((2 * torch.norm(nx_tensor.float() * dx_tensor) / c + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, num_sources_per_shot, nx)
    x_r = _set_coords(num_shots, num_receivers_per_shot, nx)
    x_p = _set_coords(1, 1, nx, "middle")[0, 0]
    v[torch.split((x_p).long(), 1)] += dc
    sources = _set_sources(x_s, freq, dt, nt, dtype)

    expected = torch.zeros(num_shots, num_receivers_per_shot, nt, dtype=dtype)
    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            for receiver in range(num_receivers_per_shot):
                expected[shot, receiver, :] += scattered(
                    x_r[shot, receiver],
                    x_s[shot, source],
                    x_p,
                    dx_tensor,
                    dt,
                    c,
                    dc,
                    -sources["amplitude"][shot, source, :],
                    ndim,
                ).to(dtype)

    lamb_const = v_const**2
    lamb = v**2
    mu = torch.zeros_like(lamb)
    buoyancy = torch.ones_like(lamb)

    k = lamb + 2 / ndim * mu
    indices = [sources["locations"][..., i] for i in range(ndim)]
    source_amplitudes_p = (
        -torch.cumsum(sources["amplitude"], dim=-1)
        * (k.cpu()[tuple(indices)].unsqueeze(-1))
    ) * dt

    y_const = propagator(
        lamb_const,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=sources["locations"],
        receiver_locations_p=x_r,
        prop_kwargs=prop_kwargs,
    )[-(ndim + 1)]

    y = propagator(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=sources["locations"],
        receiver_locations_p=x_r,
        prop_kwargs=prop_kwargs,
    )[-(ndim + 1)]

    actual = y - y_const

    return expected, actual


def run_scatter_2d_acoustic(
    c=1500,
    dc=150,
    freq=25,
    dx=(5, 5),
    dt=0.0001,
    nx=(50, 50),
    num_shots=2,
    num_sources_per_shot=2,
    num_receivers_per_shot=2,
    propagator=None,
    prop_kwargs=None,
    device=None,
    dtype=None,
):
    """Runs run_scatter_acoustic with default parameters for 2D."""
    return run_scatter_acoustic(
        c,
        dc,
        freq,
        dx,
        dt,
        nx,
        num_shots,
        num_sources_per_shot,
        num_receivers_per_shot,
        propagator,
        prop_kwargs,
        device,
        dtype,
    )


def test_scatter_2d_acoustic() -> None:
    """Test propagation in a 2D acoustic model with a point scatterer."""
    expected, actual = run_scatter_2d_acoustic(
        propagator=elasticprop,
        dt=0.001,
        prop_kwargs={"pml_width": 30},
    )
    diff = (expected - actual.cpu()).flatten()
    assert diff.norm().item() < 0.008


@pytest.mark.parametrize(
    ("nx", "dx", "nt"),
    [
        ((10,), (5,), 10000),
        ((10, 4), (5, 5), 10000),
        ((10, 4, 4), (6, 6, 6), 2000),
    ],
)
def test_wavefield_decays(nx, dx, nt) -> None:
    """Test that the PML causes the wavefield amplitude to decay."""
    out = run_forward(propagator=elasticprop, nt=nt, nx=nx, dx=dx)
    for outi in out[: -(len(dx) + 1)]:
        assert outi.norm() < 2e-6


def test_model_too_small() -> None:
    """Test that an error is raised when the model is too small."""
    with pytest.raises(RuntimeError):
        run_forward(propagator=elasticprop, nx=(3, 3), pml_width=3)


def test_forward_cpu_gpu_match() -> None:
    """Test propagation on CPU and GPU produce the same result."""
    if torch.cuda.is_available():
        for python in [True, False]:
            actual_cpu = run_forward(
                propagator=elasticprop,
                device=torch.device("cpu"),
                prop_kwargs={"python_backend": python},
            )
            actual_gpu = run_forward(
                propagator=elasticprop,
                device=torch.device("cuda"),
                prop_kwargs={"python_backend": python},
            )
            # Check wavefields
            for cpui, gpui in zip(actual_cpu[:-1], actual_gpu[:-1]):
                assert torch.allclose(cpui, gpui.cpu(), atol=1e-6)
            # Check receiver amplitudes
            cpui = actual_cpu[-1]
            gpui = actual_gpu[-1]
            assert torch.allclose(cpui, gpui.cpu(), atol=5e-5)


def test_unused_source_receiver(
    mlamb=DEFAULT_LAMB,
    mmu=DEFAULT_MU,
    mbuoyancy=DEFAULT_BUOYANCY,
    freq=25,
    dx=(5, 5),
    dt=0.004,
    nx=(25, 10),
    num_shots=2,
    num_sources_per_shot=3,
    num_receivers_per_shot=4,
    propagator=elasticprop,
    prop_kwargs=None,
    device=None,
    dtype=torch.double,
    dlamb=DEFAULT_LAMB / 10,
    dmu=DEFAULT_MU / 10,
    dbuoyancy=DEFAULT_BUOYANCY / 10,
    nt=None,
    dpeak_time=0.3,
):
    """Test forward and backward propagation when a source and receiver are unused."""
    assert num_shots > 1
    assert num_sources_per_shot > num_shots
    assert num_receivers_per_shot > num_shots + 1
    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)

    for python in [True, False]:
        torch.manual_seed(1)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if prop_kwargs is None:
            prop_kwargs = {}
        prop_kwargs["python_backend"] = python

        lamb = (
            torch.ones(*nx, device=device, dtype=dtype) * mlamb
            + torch.randn(*nx, dtype=dtype).to(device) * dlamb
        )
        mu = (
            torch.ones(*nx, device=device, dtype=dtype) * mmu
            + torch.randn(*nx, dtype=dtype).to(device) * dmu
        )
        buoyancy = (
            torch.ones(*nx, device=device, dtype=dtype) * mbuoyancy
            + torch.randn(*nx, dtype=dtype).to(device) * dbuoyancy
        )

        vp, vs, _rho = lambmubuoyancy_to_vpvsrho(lamb.abs(), mu.abs(), buoyancy.abs())
        vmin = min(vp.abs().min(), vs.abs().min())

        if nt is None:
            nt = int((2 * torch.norm(nx.float() * dx) / vmin + 0.35 + 2 / freq) / dt)
        x_s = _set_coords(num_shots, 3 * num_sources_per_shot, nx.tolist())
        x_s_y = x_s[:, :num_sources_per_shot]
        x_s_x = x_s[:, num_sources_per_shot : 2 * num_sources_per_shot]
        x_s_p = x_s[:, 2 * num_sources_per_shot :]
        x_r = _set_coords(num_shots, 3 * num_receivers_per_shot, nx.tolist(), "bottom")
        x_r_y = x_r[:, :num_receivers_per_shot]
        x_r_x = x_r[:, num_receivers_per_shot : 2 * num_receivers_per_shot]
        x_r_p = x_r[:, 2 * num_receivers_per_shot :]
        sources_y = _set_sources(x_s_y, freq, dt, nt, dtype, dpeak_time=dpeak_time)
        sources_x = _set_sources(x_s_x, freq, dt, nt, dtype, dpeak_time=dpeak_time)
        sources_p = _set_sources(x_s_p, freq, dt, nt, dtype, dpeak_time=dpeak_time)

        # Forward with source and receiver ignored
        lambi = lamb.clone()
        lambi.requires_grad_()
        mui = mu.clone()
        mui.requires_grad_()
        buoyancyi = buoyancy.clone()
        buoyancyi.requires_grad_()
        for i in range(num_shots):
            x_s_y[i, i, :] = IGNORE_LOCATION
            x_s_x[i, i + 1, :] = IGNORE_LOCATION
            x_s_p[i, i + 1, :] = IGNORE_LOCATION
            x_r_y[i, i, :] = IGNORE_LOCATION
            x_r_x[i, i + 1, :] = IGNORE_LOCATION
            x_r_p[i, i + 2, :] = IGNORE_LOCATION
        sources_y["locations"] = x_s_y
        sources_x["locations"] = x_s_x
        sources_p["locations"] = x_s_p
        out_ignored = propagator(
            lambi,
            mui,
            buoyancyi,
            dx.tolist(),
            dt,
            source_amplitudes_y=sources_y["amplitude"],
            source_amplitudes_x=sources_x["amplitude"],
            source_amplitudes_p=sources_p["amplitude"],
            source_locations_y=sources_y["locations"],
            source_locations_x=sources_x["locations"],
            source_locations_p=sources_p["locations"],
            receiver_locations_y=x_r_y,
            receiver_locations_x=x_r_x,
            receiver_locations_p=x_r_p,
            prop_kwargs=prop_kwargs,
        )

        (
            (out_ignored[-1] ** 2).sum()
            + (out_ignored[-2] ** 2).sum()
            + (out_ignored[-3] ** 2).sum()
        ).backward()

        # Forward with amplitudes of sources that will be ignored set to zero
        lambf = lamb.clone()
        lambf.requires_grad_()
        muf = mu.clone()
        muf.requires_grad_()
        buoyancyf = buoyancy.clone()
        buoyancyf.requires_grad_()
        x_s = _set_coords(num_shots, 3 * num_sources_per_shot, nx.tolist())
        x_s_y = x_s[:, :num_sources_per_shot]
        x_s_x = x_s[:, num_sources_per_shot : 2 * num_sources_per_shot]
        x_s_p = x_s[:, 2 * num_sources_per_shot :]
        x_r = _set_coords(num_shots, 3 * num_receivers_per_shot, nx.tolist(), "bottom")
        x_r_y = x_r[:, :num_receivers_per_shot]
        x_r_x = x_r[:, num_receivers_per_shot : 2 * num_receivers_per_shot]
        x_r_p = x_r[:, 2 * num_receivers_per_shot :]
        for i in range(num_shots):
            sources_y["amplitude"][i, i].fill_(0)
            sources_x["amplitude"][i, i + 1].fill_(0)
            sources_p["amplitude"][i, i + 1].fill_(0)
        sources_y["locations"] = x_s_y
        sources_x["locations"] = x_s_x
        sources_p["locations"] = x_s_p
        out_filled = propagator(
            lambf,
            muf,
            buoyancyf,
            dx.tolist(),
            dt,
            source_amplitudes_y=sources_y["amplitude"],
            source_amplitudes_x=sources_x["amplitude"],
            source_amplitudes_p=sources_p["amplitude"],
            source_locations_y=sources_y["locations"],
            source_locations_x=sources_x["locations"],
            source_locations_p=sources_p["locations"],
            receiver_locations_y=x_r_y,
            receiver_locations_x=x_r_x,
            receiver_locations_p=x_r_p,
            prop_kwargs=prop_kwargs,
        )
        # Set receiver amplitudes of receiver that will be ignored to zero
        for i in range(num_shots):
            out_filled[-2][i, i].fill_(0)
            out_filled[-1][i, i + 1].fill_(0)
            out_filled[-3][i, i + 2].fill_(0)

        (
            (out_filled[-1] ** 2).sum()
            + (out_filled[-2] ** 2).sum()
            + (out_filled[-3] ** 2).sum()
        ).backward()

        for ofi, oii in zip(out_filled, out_ignored):
            assert torch.allclose(ofi, oii)

        assert torch.allclose(lambf.grad, lambi.grad)
        assert torch.allclose(muf.grad, mui.grad)
        assert torch.allclose(buoyancyf.grad, buoyancyi.grad)


def run_elasticfunc(nt: int = 3) -> None:
    """Run elastic_func for testing purposes."""
    from deepwave.elastic import elastic_func, prepare_parameters

    torch.manual_seed(1)
    ny = 10
    nx = 11
    n_batch = 2
    dt = 0.0005
    dy = dx = 5
    grid_spacing = [dy, dx]
    pml_width = [3, 3, 3, 3]
    n_sources_y_per_shot = 1
    n_sources_x_per_shot = 2
    n_sources_p_per_shot = 2

    step_ratio = 1
    accuracy = 4
    fd_pad = accuracy // 2
    fd_pad_list = [accuracy // 2, accuracy // 2 - 1, accuracy // 2, accuracy // 2 - 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    vp = 1500 + 100 * torch.rand(1, ny, nx, dtype=dtype, device=device)
    vs = 1000 + 100 * torch.rand(1, ny, nx, dtype=dtype, device=device)
    rho = 2200 + 100 * torch.rand(1, ny, nx, dtype=dtype, device=device)

    lamb, mu, buoyancy = list(vpvsrho_to_lambmubuoyancy(vp, vs, rho))
    mu_yx, buoyancy_y, buoyancy_x = prepare_parameters(mu, buoyancy)

    vy = torch.randn(
        n_batch,
        ny - 2 * fd_pad + 1,
        nx - 2 * fd_pad + 1,
        dtype=torch.double,
        device=device,
    )
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
    source_amplitudes_y = torch.randn(
        nt,
        n_batch,
        n_sources_y_per_shot,
        dtype=torch.double,
        device=device,
    )
    source_amplitudes_x = torch.randn(
        nt,
        n_batch,
        n_sources_x_per_shot,
        dtype=torch.double,
        device=device,
    )
    source_amplitudes_p = torch.randn(
        nt,
        n_batch,
        n_sources_p_per_shot,
        dtype=torch.double,
        device=device,
    )
    sources_y_i = torch.tensor([[2 * nx + 2], [3 * nx + 3]]).long().to(device)
    sources_x_i = (
        torch.tensor([[6 * nx + 6, 7 * nx + 7], [5 * nx + 5, 7 * nx + 8]])
        .long()
        .to(device)
    )
    sources_p_i = (
        torch.tensor([[6 * nx + 6, 7 * nx + 7], [5 * nx + 5, 7 * nx + 8]])
        .long()
        .to(device)
    )
    receivers_y_i = (
        torch.tensor(
            [
                [4 * nx + 4, 5 * nx + 5, 6 * nx + 6],
                [7 * nx + 7, 5 * nx + 6, 5 * nx + 4],
            ],
        )
        .long()
        .to(device)
    )
    receivers_x_i = (
        torch.tensor(
            [
                [4 * nx + 4, 5 * nx + 5, 6 * nx + 6, 6 * nx + 7],
                [7 * nx + 7, 5 * nx + 6, 5 * nx + 4, 6 * nx + 6],
            ],
        )
        .long()
        .to(device)
    )
    receivers_p_i = (
        torch.tensor(
            [
                [4 * nx + 4, 5 * nx + 5, 6 * nx + 6, 6 * nx + 5, 6 * nx + 7],
                [7 * nx + 7, 5 * nx + 6, 5 * nx + 4, 6 * nx + 6, 5 * nx + 5],
            ],
        )
        .long()
        .to(device)
    )
    lamb.requires_grad_()
    mu.requires_grad_()
    mu_yx.requires_grad_()
    buoyancy_y.requires_grad_()
    buoyancy_x.requires_grad_()
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
    max_vel = 2000
    pml_freq = 25
    pml_profiles = deepwave.staggered_grid.set_pml_profiles(
        pml_width,
        accuracy,
        fd_pad_list,
        dt,
        grid_spacing,
        max_vel,
        dtype,
        device,
        pml_freq,
        [ny, nx],
    )
    for i, profile in enumerate(pml_profiles):
        mask = profile != 0
        pml_profiles[i] = torch.randn_like(profile) * mask

    def wrap(python):
        inputs = (
            python,
            pml_profiles,
            [
                dy,
                dx,
            ],
            dt,
            nt,
            step_ratio,
            accuracy,
            pml_width,
            n_batch,
            None,
            None,
            1,
            lamb,
            mu,
            mu_yx,
            buoyancy_y,
            buoyancy_x,
            source_amplitudes_y,
            source_amplitudes_x,
            source_amplitudes_p,
            sources_y_i,
            sources_x_i,
            sources_p_i,
            receivers_y_i,
            receivers_x_i,
            receivers_p_i,
            vy,
            vx,
            sigmayy,
            sigmaxy,
            sigmaxx,
            m_vyy,
            m_vyx,
            m_vxy,
            m_vxx,
            m_sigmayyy,
            m_sigmaxyy,
            m_sigmaxyx,
            m_sigmaxxx,
        )
        out = elastic_func(*inputs)
        v = [torch.randn_like(o.contiguous()) for o in out]
        grads = torch.autograd.grad(
            out,
            [p for p in inputs if isinstance(p, torch.Tensor) and p.requires_grad],
            grad_outputs=v,
        )
        return out + grads

    torch.manual_seed(1)
    out_compiled = wrap(False)
    torch.manual_seed(1)
    out_python = wrap(True)
    for oc, op in zip(out_compiled, out_python):
        assert torch.allclose(oc, op)


def test_elasticfunc():
    """Test elastic_func with different time steps."""
    run_elasticfunc(nt=1)
    run_elasticfunc(nt=2)
    run_elasticfunc(nt=3)


def test_gradcheck():
    """Test gradcheck in a 2D model."""
    run_gradcheck(propagator=elasticprop)


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((5,), (5,)),
        ((5, 4), (5, 5)),
        ((5, 4, 4), (6, 6, 6)),
    ],
)
def test_gradcheck_2nd_order(nx, dx):
    """Test gradcheck with a 2nd order accurate propagator."""
    run_gradcheck(propagator=elasticprop, dx=dx, nx=nx, prop_kwargs={"accuracy": 2})


def test_gradcheck_cfl():
    """Test gradcheck with a timestep greater than the CFL limit."""
    run_gradcheck(
        propagator=elasticprop,
        dt=0.002,
        prop_kwargs={"time_taper": True},
        only_receivers_out=True,
    )


def test_gradcheck_odd_timesteps():
    """Test gradcheck with one more timestep."""
    run_gradcheck(propagator=elasticprop, nt_add=1)


def test_gradcheck_one_shot():
    """Test gradcheck with one shot."""
    run_gradcheck(propagator=elasticprop, num_shots=1)


def test_gradcheck_survey_pad():
    """Test gradcheck with survey_pad."""
    # Location range: y: [0, 3], x: [10]
    # With survey pad: y: [0, 3+4=7], x: [10-4=6, 10+4=14]
    run_gradcheck(
        propagator=elasticprop,
        nx=(21, 20),
        pml_width=5,
        wavefield_size=[(7 - 0 + 1) + 2 * 5, (14 - 6 + 1) + 2 * 5],
        survey_pad=4,
    )


def test_gradcheck_chained():
    """Test gradcheck when two propagators are chained."""
    run_gradcheck(propagator=elasticpropchained)


def test_gradcheck_different_pml():
    """Test gradcheck with different pml widths."""
    run_gradcheck(propagator=elasticprop, pml_width=[0, 1, 5, 10])


def test_gradcheck_no_pml():
    """Test gradcheck with no pml."""
    run_gradcheck(propagator=elasticprop, pml_width=0, nx=(11, 11))


def test_gradcheck_different_dx():
    """Test gradcheck with different dx values."""
    run_gradcheck(propagator=elasticprop, dx=(4, 5), dt=0.0005)


def test_gradcheck_only_lamb_2d():
    """Test gradcheck with only lamb requiring gradient."""
    run_gradcheck(
        propagator=elasticprop,
        lamb_requires_grad=True,
        mu_requires_grad=False,
        buoyancy_requires_grad=False,
        source_requires_grad=False,
    )


def test_gradcheck_only_mu_2d():
    """Test gradcheck with only mu requiring gradient."""
    run_gradcheck(
        propagator=elasticprop,
        lamb_requires_grad=False,
        mu_requires_grad=True,
        buoyancy_requires_grad=False,
        source_requires_grad=False,
    )


def test_gradcheck_batched_lamb_2d():
    """Test gradcheck in a 2D model with batched lamb."""
    run_gradcheck(
        propagator=elasticprop,
        mlamb=torch.tensor([[[DEFAULT_LAMB]], [[DEFAULT_LAMB * 1.2]]]),
    )


def test_gradcheck_batched_mu_2d():
    """Test gradcheck in a 2D model with batched mu."""
    run_gradcheck(
        propagator=elasticprop,
        mmu=torch.tensor([[[DEFAULT_MU]], [[DEFAULT_MU * 1.2]]]),
    )


def test_gradcheck_batched_buoyancy_2d():
    """Test gradcheck in a 2D model with batched buoyancy."""
    run_gradcheck(
        propagator=elasticprop,
        mbuoyancy=torch.tensor([[[DEFAULT_BUOYANCY]], [[DEFAULT_BUOYANCY * 1.2]]]),
    )


def test_gradcheck_zeros_in_properties():
    """Test gradcheck when the property models contain zeros."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    nx = (10, 10)
    lamb = torch.ones(*nx, device=device, dtype=dtype) * DEFAULT_LAMB
    mu = torch.ones(*nx, device=device, dtype=dtype) * DEFAULT_MU
    buoyancy = torch.ones(*nx, device=device, dtype=dtype) * DEFAULT_BUOYANCY
    lamb[:2] = 0
    mu[:4] = 0
    buoyancy[:6] = 0
    lamb[-1] = 0
    mu[-2] = 0
    buoyancy[-3] = 0
    run_gradcheck(
        propagator=elasticprop,
        lamb=lamb,
        mu=mu,
        buoyancy=buoyancy,
    )


def run_forward_lamb(
    orientation=0,
    prop_kwargs=None,
    device=None,
    dtype=None,
    pml_width=20,
    **kwargs,
):
    """Runs a forward elastic propagation with a specific source orientation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny = 20
    nx = 20
    nt = 3000
    dt = 0.0005
    dx = 4
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(
        1500 * torch.ones(1, dtype=dtype, device=device),
        1000 * torch.ones(1, dtype=dtype, device=device),
        2200 * torch.ones(1, dtype=dtype, device=device),
    )
    lamb = torch.ones(ny, nx, dtype=dtype, device=device) * lamb
    mu = torch.ones(ny, nx, dtype=dtype, device=device) * mu
    buoyancy = torch.ones(ny, nx, dtype=dtype, device=device) * buoyancy
    freq = 14.5
    peak_time = 0.08
    source_amplitudes = ricker(freq, nt, dt, peak_time, dtype).reshape(1, 1, -1)
    if orientation == 0:
        x_s_y = torch.tensor([[[15, 0]]])
        x_s_x = None
        sa_y = source_amplitudes
        sa_x = None
        x_r_y = torch.tensor([[[5, 15]]])
        x_r_x = torch.tensor([[[6, 15]]])
        pml_width = [0, pml_width, pml_width, pml_width]
    elif orientation == 1:
        x_s_y = torch.tensor([[[ny - 1 - 15, nx - 1 - 0]]])
        x_s_x = None
        sa_y = source_amplitudes
        sa_x = None
        x_r_y = torch.tensor([[[ny - 1 - 5, nx - 1 - 15]]])
        x_r_x = torch.tensor([[[ny - 1 - 5, nx - 1 - 16]]])
        pml_width = [pml_width, 0, pml_width, pml_width]
    elif orientation == 2:
        x_s_y = None
        x_s_x = torch.tensor([[[0, 15]]])
        sa_y = None
        sa_x = source_amplitudes
        x_r_y = torch.tensor([[[15, 6]]])
        x_r_x = torch.tensor([[[15, 5]]])
        pml_width = [pml_width, pml_width, 0, pml_width]
    else:
        x_s_y = None
        x_s_x = torch.tensor([[[ny - 1, nx - 1 - 15]]])
        sa_y = None
        sa_x = source_amplitudes
        x_r_y = torch.tensor([[[ny - 1 - 16, nx - 1 - 5]]])
        x_r_x = torch.tensor([[[ny - 1 - 15, nx - 1 - 5]]])
        pml_width = [pml_width, pml_width, pml_width, 0]
    pad = (
        int(pml_width[2] == 0),
        int(pml_width[3] == 0),
        int(pml_width[0] == 0),
        int(pml_width[1] == 0),
    )
    lambp = torch.nn.functional.pad(lamb, pad)
    mup = torch.nn.functional.pad(mu, pad)
    buoyancyp = torch.nn.functional.pad(buoyancy, pad)
    return elasticprop(
        lambp,
        mup,
        buoyancyp,
        dx,
        dt,
        source_amplitudes_y=sa_y,
        source_amplitudes_x=sa_x,
        source_amplitudes_p=None,
        source_locations_y=x_s_y,
        source_locations_x=x_s_x,
        source_locations_p=None,
        receiver_locations_y=x_r_y,
        receiver_locations_x=x_r_x,
        receiver_locations_p=x_r_x,
        prop_kwargs=prop_kwargs,
        pml_width=pml_width,
        **kwargs,
    )


def run_forward(
    mlamb=DEFAULT_LAMB,
    mmu=DEFAULT_MU,
    mbuoyancy=DEFAULT_BUOYANCY,
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
    dlamb=DEFAULT_LAMB / 10,
    dmu=DEFAULT_MU / 10,
    dbuoyancy=DEFAULT_BUOYANCY / 10,
    nt=None,
    dpeak_time=0.3,
    use_pressure_sources=True,
    **kwargs,
):
    """Create a random model and forward propagate."""
    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lamb = (
        torch.ones(*nx, device=device, dtype=dtype) * mlamb
        + torch.randn(*nx, dtype=dtype).to(device) * dlamb
    )
    mu = (
        torch.ones(*nx, device=device, dtype=dtype) * mmu
        + torch.randn(*nx, dtype=dtype).to(device) * dmu
    )
    buoyancy = (
        torch.ones(*nx, device=device, dtype=dtype) * mbuoyancy
        + torch.randn(*nx, dtype=dtype).to(device) * dbuoyancy
    )

    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)
    ndim = len(dx)

    vp, vs, _rho = lambmubuoyancy_to_vpvsrho(lamb.abs(), mu.abs(), buoyancy.abs())
    vmin = min(vp.abs().min(), vs.abs().min())

    if nt is None:
        nt = int((2 * torch.norm(nx.float() * dx) / vmin + 0.35 + 2 / freq) / dt)
    x_s = _set_coords(num_shots, 2 * num_sources_per_shot, nx.tolist())
    x_s_y = x_s[:, :num_sources_per_shot]
    x_s_x = x_s[:, num_sources_per_shot:]
    x_r = _set_coords(num_shots, 2 * num_receivers_per_shot, nx.tolist(), "bottom")
    x_r_y = x_r[:, :num_receivers_per_shot]
    x_r_x = x_r[:, num_receivers_per_shot:]
    sources_y = _set_sources(x_s_y, freq, dt, nt, dtype, dpeak_time=dpeak_time)
    sources_x = _set_sources(x_s_x, freq, dt, nt, dtype, dpeak_time=dpeak_time)

    source_amplitudes = {
        "source_amplitudes_x": sources_x["amplitude"],
    }
    source_locations = {
        "source_locations_x": sources_x["locations"],
    }
    receiver_locations = {
        "receiver_locations_x": x_r_x,
        "receiver_locations_p": x_r_x,
    }
    if use_pressure_sources:
        source_amplitudes["source_amplitudes_p"] = sources_x["amplitude"]
        source_locations["source_locations_p"] = sources_x["locations"]
    if ndim >= 2:
        source_amplitudes["source_amplitudes_y"] = sources_y["amplitude"]
        source_locations["source_locations_y"] = sources_y["locations"]
        receiver_locations["receiver_locations_y"] = x_r_y
    if ndim >= 3:
        source_amplitudes["source_amplitudes_z"] = sources_y["amplitude"]
        source_locations["source_locations_z"] = sources_y["locations"]
        receiver_locations["receiver_locations_z"] = x_r_y

    return propagator(
        lamb,
        mu,
        buoyancy,
        dx.tolist(),
        dt,
        **source_amplitudes,
        **source_locations,
        **receiver_locations,
        prop_kwargs=prop_kwargs,
        **kwargs,
    )


def run_gradcheck(
    mlamb: float = DEFAULT_LAMB,
    mmu: float = DEFAULT_MU,
    mbuoyancy: float = DEFAULT_BUOYANCY,
    freq: float = 25,
    dx: Tuple[float, float] = (5, 5),
    dt: float = 0.001,
    nx: Tuple[int, int] = (10, 10),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = elasticprop,
    prop_kwargs: Optional[Dict[str, Any]] = None,
    pml_width: Union[int, List[int]] = 4,
    survey_pad: Optional[Union[int, List[int]]] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.double,
    dlamb=DEFAULT_LAMB / 10,
    dmu=DEFAULT_MU / 10,
    dbuoyancy=DEFAULT_BUOYANCY / 10,
    lamb_requires_grad=True,
    mu_requires_grad=True,
    buoyancy_requires_grad=True,
    source_requires_grad=True,
    provide_wavefields: bool = True,
    only_receivers_out: bool = False,
    wavefield_size: Optional[List[int]] = None,
    nt_add=0,
    lamb: Optional[torch.Tensor] = None,
    mu: Optional[torch.Tensor] = None,
    buoyancy: Optional[torch.Tensor] = None,
    atol=1e-5,
    rtol=1e-3,
):
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
    if lamb is None:
        lamb = torch.ones(*nx, device=device, dtype=dtype) * mlamb
        lamb += torch.randn(*lamb.shape, dtype=dtype).to(device) * dlamb
    if mu is None:
        mu = torch.ones(*nx, device=device, dtype=dtype) * mmu
        mu += torch.randn(*mu.shape, dtype=dtype).to(device) * dmu
    if buoyancy is None:
        buoyancy = torch.ones(*nx, device=device, dtype=dtype) * mbuoyancy
        buoyancy += torch.randn(*buoyancy.shape, dtype=dtype).to(device) * dbuoyancy

    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)
    ndim = len(nx)

    vp, vs, _rho = lambmubuoyancy_to_vpvsrho(lamb.abs(), mu.abs(), buoyancy.abs())
    vmin = min(vp.abs().min(), vs.abs().min())

    if vmin != 0:
        nt = int((2 * torch.norm(nx.float() * dx) / vmin + 0.1 + 2 / freq) / dt)
    else:
        nt = int((2 * torch.norm(nx.float() * dx) / 1500 + 0.1 + 2 / freq) / dt)
    nt += nt_add

    source_amplitudes = {
        "source_amplitudes_z": None,
        "source_amplitudes_y": None,
        "source_amplitudes_x": None,
        "source_amplitudes_p": None,
    }
    source_locations = {
        "source_locations_z": None,
        "source_locations_y": None,
        "source_locations_x": None,
        "source_locations_p": None,
    }
    receiver_locations = {
        "receiver_locations_z": None,
        "receiver_locations_y": None,
        "receiver_locations_x": None,
        "receiver_locations_p": None,
    }
    if num_sources_per_shot > 0:
        x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
        sources_x = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)
        sources_x["amplitude"] = sources_x["amplitude"].to(device)
        sources_x["amplitude"].requires_grad_(source_requires_grad)
        source_amplitudes["source_amplitudes_x"] = sources_x["amplitude"]
        source_locations["source_locations_x"] = sources_x["locations"]
        sources_p = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)
        sources_p["amplitude"] = sources_p["amplitude"].to(device)
        sources_p["amplitude"].requires_grad_(source_requires_grad)
        source_amplitudes["source_amplitudes_p"] = sources_p["amplitude"]
        source_locations["source_locations_p"] = sources_p["locations"]
        if ndim >= 2:
            sources_y = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)
            sources_y["amplitude"] = sources_y["amplitude"].to(device)
            sources_y["amplitude"].requires_grad_(source_requires_grad)
            source_amplitudes["source_amplitudes_y"] = sources_y["amplitude"]
            source_locations["source_locations_y"] = sources_y["locations"]
        if ndim >= 3:
            sources_z = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)
            sources_z["amplitude"] = sources_z["amplitude"].to(device)
            sources_z["amplitude"].requires_grad_(source_requires_grad)
            source_amplitudes["source_amplitudes_z"] = sources_z["amplitude"]
            source_locations["source_locations_z"] = sources_z["locations"]
        nt = None
    if num_receivers_per_shot > 0:
        x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist())
        receiver_locations["receiver_locations_x"] = x_r
        receiver_locations["receiver_locations_p"] = x_r
        if ndim >= 2:
            receiver_locations["receiver_locations_y"] = x_r
        if ndim >= 3:
            receiver_locations["receiver_locations_z"] = x_r
    if isinstance(pml_width, int):
        pml_width = [pml_width for _ in range(2 * ndim)]

    wavefields = {}
    for name in [
        "vz_0",
        "vy_0",
        "vx_0",
        "sigmazz_0",
        "sigmayz_0",
        "sigmaxz_0",
        "sigmayy_0",
        "sigmaxy_0",
        "sigmaxx_0",
        "m_vzz_0",
        "m_vzy_0",
        "m_vzx_0",
        "m_vyz_0",
        "m_vxz_0",
        "m_vyy_0",
        "m_vyx_0",
        "m_vxy_0",
        "m_vxx_0",
        "m_sigmazzz_0",
        "m_sigmayzy_0",
        "m_sigmaxzx_0",
        "m_sigmayzz_0",
        "m_sigmaxzz_0",
        "m_sigmayyy_0",
        "m_sigmaxyy_0",
        "m_sigmaxyx_0",
        "m_sigmaxxx_0",
    ]:
        wavefields[name] = None

    wavefield_names = []
    if ndim >= 3:
        wavefield_names.extend(
            [
                "vz_0",
                "sigmazz_0",
                "sigmayz_0",
                "sigmaxz_0",
                "m_vzz_0",
                "m_vzy_0",
                "m_vzx_0",
                "m_vyz_0",
                "m_vxz_0",
                "m_sigmazzz_0",
                "m_sigmayzy_0",
                "m_sigmaxzx_0",
                "m_sigmayzz_0",
                "m_sigmaxzz_0",
            ]
        )
    if ndim >= 2:
        wavefield_names.extend(
            [
                "vy_0",
                "sigmayy_0",
                "sigmaxy_0",
                "m_vyy_0",
                "m_vyx_0",
                "m_vxy_0",
                "m_sigmayyy_0",
                "m_sigmaxyy_0",
                "m_sigmaxyx_0",
            ]
        )
    wavefield_names.extend(
        [
            "vx_0",
            "sigmaxx_0",
            "m_vxx_0",
            "m_sigmaxxx_0",
        ]
    )

    if provide_wavefields:
        if wavefield_size is None:
            wavefield_size = tuple(
                nx[i] + pml_width[2 * i] + pml_width[2 * i + 1] for i in range(ndim)
            )
        for name in wavefield_names:
            wavefields[name] = torch.zeros(
                num_shots,
                *wavefield_size,
                dtype=dtype,
                device=device,
                requires_grad=True,
            )

    lamb.requires_grad_(lamb_requires_grad)
    mu.requires_grad_(mu_requires_grad)
    buoyancy.requires_grad_(buoyancy_requires_grad)

    if prop_kwargs is None:
        prop_kwargs = {}

    out_idxs = []

    def wrap(python):
        prop_kwargs["python_backend"] = python
        inputs = (
            lamb,
            mu,
            buoyancy,
            dx.tolist(),
            dt,
            *source_amplitudes.values(),
            *source_locations.values(),
            *receiver_locations.values(),
            prop_kwargs,
            pml_width,
            survey_pad,
            *wavefields.values(),
            nt,
        )
        out = propagator(*inputs)
        if only_receivers_out:
            out = out[-3:]
        if len(out_idxs) == 0:
            for i, o in enumerate(out):
                if o.requires_grad:
                    out_idxs.append(i)
        out = tuple([out[i] for i in out_idxs])
        v = [torch.randn_like(o.contiguous()) for o in out]
        return out + torch.autograd.grad(
            out,
            [p for p in inputs if isinstance(p, torch.Tensor) and p.requires_grad],
            grad_outputs=v,
        )

    torch.manual_seed(1)
    out_python = wrap(True)
    torch.manual_seed(1)
    out_compiled = wrap(False)
    for oc, op in zip(out_compiled, out_python):
        assert torch.allclose(oc, op, atol=atol, rtol=rtol)


# Tests for Elastic.__init__
def test_elastic_init_lamb_non_tensor() -> None:
    """Test that Elastic.__init__ raises TypeError if lamb is not a tensor."""
    with pytest.raises(TypeError, match=re.escape("lamb must be a torch.Tensor.")):
        Elastic([1, 2, 3], torch.ones(10, 10), torch.ones(10, 10), 1.0)


def test_elastic_init_mu_non_tensor() -> None:
    """Test that Elastic.__init__ raises TypeError if mu is not a tensor."""
    with pytest.raises(TypeError, match=re.escape("mu must be a torch.Tensor.")):
        Elastic(torch.ones(10, 10), [1, 2, 3], torch.ones(10, 10), 1.0)


def test_elastic_init_buoyancy_non_tensor() -> None:
    """Test that Elastic.__init__ raises TypeError if buoyancy is not a tensor."""
    with pytest.raises(TypeError, match=re.escape("buoyancy must be a torch.Tensor.")):
        Elastic(torch.ones(10, 10), torch.ones(10, 10), [1, 2, 3], 1.0)


def test_elastic_init_lamb_requires_grad_non_bool() -> None:
    """Test Elastic.__init__ raises TypeError if lamb_requires_grad is not bool."""
    lamb = torch.ones(10, 10)
    mu = torch.ones(10, 10)
    buoyancy = torch.ones(10, 10)
    with pytest.raises(
        TypeError, match=re.escape("lamb_requires_grad must be bool, got int")
    ):
        Elastic(lamb, mu, buoyancy, 1.0, lamb_requires_grad=1)


def test_elastic_init_mu_requires_grad_non_bool() -> None:
    """Test that Elastic.__init__ raises TypeError if mu_requires_grad is not a bool."""
    lamb = torch.ones(10, 10)
    mu = torch.ones(10, 10)
    buoyancy = torch.ones(10, 10)
    with pytest.raises(
        TypeError, match=re.escape("mu_requires_grad must be bool, got float")
    ):
        Elastic(lamb, mu, buoyancy, 1.0, mu_requires_grad=0.5)


def test_elastic_init_buoyancy_requires_grad_non_bool() -> None:
    """Test Elastic.__init__ raises TypeError if buoyancy_requires_grad is not bool."""
    lamb = torch.ones(10, 10)
    mu = torch.ones(10, 10)
    buoyancy = torch.ones(10, 10)
    with pytest.raises(
        TypeError, match=re.escape("buoyancy_requires_grad must be bool, got str")
    ):
        Elastic(lamb, mu, buoyancy, 1.0, buoyancy_requires_grad="True")


# Tests for elastic function (location bounds)
def test_elastic_source_locations_x_out_of_bounds() -> None:
    """Test raises for source locations out of x bounds."""
    lamb = torch.ones(10, 10)
    mu = torch.ones(10, 10)
    buoyancy = torch.ones(10, 10)
    grid_spacing = 1.0
    dt = 0.001
    source_amplitudes_x = torch.randn(1, 1, 10)
    source_locations_x = torch.tensor(
        [[[1, 9]]], dtype=torch.long
    )  # x-coord is 9, max is 8 (shape[1]-1)
    nt = 10
    wavefield_0 = torch.zeros(1, 10 + 2 * 20, 10 + 2 * 20)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "With the provided model, the maximum x source location in the "
            "x dimension must be less than 9."
        ),
    ):
        elastic(
            lamb,
            mu,
            buoyancy,
            grid_spacing,
            dt,
            source_amplitudes_x=source_amplitudes_x,
            source_locations_x=source_locations_x,
            nt=nt,
            sigmayy_0=wavefield_0,
        )


def test_elastic_receiver_locations_x_out_of_bounds() -> None:
    """Test raises for receiver locations out of x bounds."""
    lamb = torch.ones(10, 10)
    mu = torch.ones(10, 10)
    buoyancy = torch.ones(10, 10)
    grid_spacing = 1.0
    dt = 0.001
    receiver_locations_x = torch.tensor(
        [[[1, 9]]], dtype=torch.long
    )  # x-coord is 9, max is 8 (shape[1]-1)
    nt = 10
    wavefield_0 = torch.zeros(1, 10 + 2 * 20, 10 + 2 * 20)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "With the provided model, the maximum x receiver location in the "
            "x dimension must be less than 9."
        ),
    ):
        elastic(
            lamb,
            mu,
            buoyancy,
            grid_spacing,
            dt,
            receiver_locations_x=receiver_locations_x,
            nt=nt,
            sigmayy_0=wavefield_0,
        )


def test_elastic_source_locations_y_out_of_bounds() -> None:
    """Test raises for source locations out of y bounds."""
    lamb = torch.ones(10, 10)
    mu = torch.ones(10, 10)
    buoyancy = torch.ones(10, 10)
    grid_spacing = 1.0
    dt = 0.001
    source_amplitudes_y = torch.randn(1, 1, 10)
    source_locations_y = torch.tensor(
        [[[9, 1]]], dtype=torch.long
    )  # y-coord is 9, max is 8
    nt = 10
    wavefield_0 = torch.zeros(1, 10 + 2 * 20, 10 + 2 * 20)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "With the provided model, the maximum y source location in the "
            "y dimension must be less than 9."
        ),
    ):
        elastic(
            lamb,
            mu,
            buoyancy,
            grid_spacing,
            dt,
            source_amplitudes_y=source_amplitudes_y,
            source_locations_y=source_locations_y,
            nt=nt,
            sigmayy_0=wavefield_0,
        )


def test_elastic_receiver_locations_y_out_of_bounds() -> None:
    """Test raises for receiver locations out of y bounds."""
    lamb = torch.ones(10, 10)
    mu = torch.ones(10, 10)
    buoyancy = torch.ones(10, 10)
    grid_spacing = 1.0
    dt = 0.001
    receiver_locations_y = torch.tensor(
        [[[9, 1]]], dtype=torch.long
    )  # y-coord is 9, max is 8
    nt = 10
    wavefield_0 = torch.zeros(1, 10 + 2 * 20, 10 + 2 * 20)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "With the provided model, the maximum y receiver location in the "
            "y dimension must be less than 9."
        ),
    ):
        elastic(
            lamb,
            mu,
            buoyancy,
            grid_spacing,
            dt,
            receiver_locations_y=receiver_locations_y,
            nt=nt,
            sigmayy_0=wavefield_0,
        )


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((10,), (5,)),
        ((10, 10), (5, 5)),
        ((10, 10, 10), (6, 6, 6)),
    ],
)
def test_reciprocity(nx, dx):
    """Test source-receiver reciprocity."""
    dtype = torch.double
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freq = 25
    nt = 250
    dt = 0.001
    peak_time = 0.05
    ndim = len(nx)

    torch.manual_seed(1)
    lamb = (
        torch.ones(*nx, device=device, dtype=dtype) * DEFAULT_LAMB
        + torch.randn(*nx, device=device, dtype=dtype) * DEFAULT_LAMB / 10
    )
    mu = (
        torch.ones(*nx, device=device, dtype=dtype) * DEFAULT_MU
        + torch.randn(*nx, device=device, dtype=dtype) * DEFAULT_MU / 10
    )
    buoyancy = (
        torch.ones(*nx, device=device, dtype=dtype) * DEFAULT_BUOYANCY
        + torch.randn(*nx, device=device, dtype=dtype) * DEFAULT_BUOYANCY / 10
    )

    source_amplitudes = ricker(freq, nt, dt, peak_time, dtype=dtype).reshape(1, 1, -1)

    loc_a = torch.tensor([d // 4 for d in nx], device=device).reshape(1, 1, -1)
    loc_b = torch.tensor([d - d // 4 for d in nx], device=device).reshape(1, 1, -1)

    force_components = ["x"]
    if ndim >= 2:
        force_components.append("y")
    if ndim == 3:
        force_components.append("z")

    prop_kwargs = {"accuracy": 2}
    rec_idxs = {"p": -(ndim + 1), "z": -3, "y": -2, "x": -1}

    for i in range(len(force_components)):
        for j in range(i, len(force_components)):
            src_comp = force_components[i]
            rec_comp = force_components[j]
            run_reciprocity_check(
                [lamb, mu, buoyancy],
                dx,
                dt,
                source_amplitudes,
                "_" + src_comp,
                "_" + rec_comp,
                rec_idxs[src_comp],
                rec_idxs[rec_comp],
                loc_a,
                loc_b,
                prop_kwargs,
                elasticprop,
            )

    # Test pressure reciprocity
    run_reciprocity_check(
        [lamb, mu, buoyancy],
        dx,
        dt,
        source_amplitudes,
        "_p",
        "_p",
        rec_idxs["p"],
        rec_idxs["p"],
        loc_a,
        loc_b,
        prop_kwargs,
        elasticprop,
    )


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((51,), (1,)),
        ((51, 41), (5, 5)),
        ((51, 41, 31), (5, 5, 5)),
    ],
)
def test_travel_time(nx, dx):
    """Test travel time difference between two receivers."""
    dtype = torch.double
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loc_s = torch.tensor([d - 1 for d in nx], device=device).reshape(1, 1, -1)
    loc_r1 = torch.tensor([0 for _ in nx], device=device).reshape(1, 1, -1)
    loc_r2 = torch.tensor([d // 2 for d in nx], device=device).reshape(1, 1, -1)

    vp = 2000.0
    # Use an acoustic model to simplify wave arrivals
    vs = 0.0
    rho = 2200.0
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(
        torch.tensor(vp, device=device, dtype=dtype),
        torch.tensor(vs, device=device, dtype=dtype),
        torch.tensor(rho, device=device, dtype=dtype),
    )

    lamb = torch.ones(*nx, device=device, dtype=dtype) * lamb
    mu = torch.ones(*nx, device=device, dtype=dtype) * mu
    buoyancy = torch.ones(*nx, device=device, dtype=dtype) * buoyancy

    # Calculate theoretical travel time difference
    dx_tensor = torch.tensor(dx, device=device, dtype=dtype)
    dist_s_r1 = torch.norm((loc_s.float() - loc_r1.float()) * dx_tensor).item()
    dist_s_r2 = torch.norm((loc_s.float() - loc_r2.float()) * dx_tensor).item()
    dt_theory = (dist_s_r2 - dist_s_r1) / vp

    freq = 25
    dt = 0.0005
    peak_time = 0.05
    nt = int((max(dist_s_r1, dist_s_r2) / vp + 2 * peak_time) / dt)
    ndim = len(dx)

    source_amplitudes = ricker(freq, nt, dt, peak_time, dtype=dtype).reshape(1, 1, -1)

    # Use pressure source and receivers for P-waves
    kwargs = {
        "source_amplitudes_p": source_amplitudes,
        "source_locations_p": loc_s,
        "receiver_locations_p": torch.cat([loc_r1, loc_r2], dim=1),
    }

    outputs = elasticprop(
        lamb,
        mu,
        buoyancy,
        list(dx),
        dt,
        **kwargs,
    )

    recs = outputs[-(ndim + 1)]
    rec1 = recs[0, 0, :]
    rec2 = recs[0, 1, :]

    # Find observed travel time difference
    t1_obs_idx = torch.argmax(rec1.abs()).item()
    t2_obs_idx = torch.argmax(rec2.abs()).item()
    dt_obs = (t2_obs_idx - t1_obs_idx) * dt

    assert abs(dt_obs - dt_theory) <= dt


@pytest.mark.parametrize(
    ("shape", "dx", "stress_indices"),
    [
        ((5,), (4.0,), [1]),
        ((5, 5), (4.0, 4.0), [2, 4]),
        ((5, 5, 5), (4.0, 4.0, 4.0), [3, 6, 8]),
    ],
)
def test_stress_symmetry(shape, dx, stress_indices):
    """Test that normal stress is symmetric for a central pressure source."""
    pml_width = 5
    accuracy = 2
    dtype = torch.double
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nt = 100
    dt = 0.004
    torch.manual_seed(1)

    # Model
    lamb = torch.tensor(DEFAULT_LAMB, dtype=dtype, device=device)
    mu = torch.tensor(DEFAULT_MU, dtype=dtype, device=device)
    buoyancy = torch.tensor(DEFAULT_BUOYANCY, dtype=dtype, device=device)
    model_lamb = torch.full(shape, lamb.item(), dtype=dtype, device=device)
    model_mu = torch.full(shape, mu.item(), dtype=dtype, device=device)
    model_buoyancy = torch.full(shape, buoyancy.item(), dtype=dtype, device=device)

    # Source
    source_amplitudes = torch.randn(nt, dtype=dtype).reshape(1, 1, -1)
    source_location = torch.tensor([s // 2 for s in shape], device=device).reshape(
        1, 1, -1
    )

    # Propagate
    outputs = elasticprop(
        model_lamb,
        model_mu,
        model_buoyancy,
        list(dx),
        dt,
        source_amplitudes_p=source_amplitudes,
        source_locations_p=source_location,
        pml_width=pml_width,
        prop_kwargs={"accuracy": accuracy},
        nt=nt,
    )

    # Check symmetry
    normal_stresses = [outputs[i] for i in stress_indices]
    sum_normal_stresses = torch.sum(torch.stack(normal_stresses), dim=0)

    for dim in range(len(shape)):
        # Dimension in tensor is dim + 1 due to batch dimension
        flipped = torch.flip(sum_normal_stresses, dims=[dim + 1])
        assert torch.allclose(sum_normal_stresses, flipped, atol=1e-9)
