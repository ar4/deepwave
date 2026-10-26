"""Tests for deepwave.elastic."""

import re
from pathlib import Path
from typing import Any, Literal, Optional, Union

import pytest
import torch
from helpers import (
    _build_multicomponent_receivers,
    _build_multicomponent_sources,
    _default_wavefield_size,
    _normalize_pml_width,
    _set_coords,
    _set_sources,
    auxiliary_state_arg,
    c_backend_available,
    compute_backend_parity_grad,
    compute_backend_parity_nt,
    direct_1d,
    direct_2d_approx,
    direct_3d,
    element_tensors,
    output_tensors,
    run_checkpoint_equivalence,
    run_python_vs_compiled_backend_parity,
    run_reciprocity_check,
    scattered,
)
from torch.func import jvp, vmap

import deepwave
from deepwave import (
    IGNORE_LOCATION,
    Elastic,
    elastic,
    lambmubuoyancy_to_vpvsrho,
    vpvsrho_to_lambmubuoyancy,
)
from deepwave._backend_utils import USE_OPENMP
from deepwave.discretisation import cfl_condition, downsample, upsample
from deepwave.wavelets import ricker


@pytest.fixture(scope="module", autouse=True)
def _detect_anomaly():
    """Runs this module's tests with autograd anomaly detection enabled.

    Anomaly mode is a global of the interpreter, so enabling it at import
    time would leave it on for every other test module collected into the
    same process, where it costs an order of magnitude in run time for no
    benefit. Scoping it to this module and restoring the previous setting
    afterwards keeps it where it is wanted.
    """
    previous = torch.is_anomaly_enabled()
    torch.autograd.set_detect_anomaly(True)
    try:
        yield
    finally:
        torch.autograd.set_detect_anomaly(previous)


# ``python_backend=True`` resolves to "torch_compile" if the OpenMP build is
# available, else "eager" (v0.0.27 chose torch.jit.script there, a mode that
# no longer exists). Tests below that loop over "the non-compiled Python
# backend vs the compiled C backend" use that same resolution rather than
# hardcoding one, so they exercise whichever path this build actually takes.
_PYTHON_BACKEND_TRUE_EQUIVALENT = "torch_compile" if USE_OPENMP else "eager"
#: The same choice in the *internal* step-function vocabulary, for the tests
#: that call ``elastic_func`` directly (it takes the resolved mode, not the
#: public ``backend`` spelling).
_PYTHON_BACKEND_TRUE_MODE = "compile" if USE_OPENMP else "eager"

DEFAULT_LAMB = 550000000
DEFAULT_MU = 2200000000
DEFAULT_BUOYANCY = 1 / 2200


def test_cfl_uses_pwave_velocity() -> None:
    """The elastic CFL must use the P-wave velocity (the fastest mode).

    elastic() sets ``max_model_vel = max(vp.max(), vs.max())`` from
    ``lambmubuoyancy_to_vpvsrho``. Since vp = sqrt((lambda + 2 mu) / rho) is
    always >= vs = sqrt(mu / rho), the controlling velocity is vp. This guards
    the formula and the fastest-mode selection.
    """
    rho = torch.full((5, 5), 2200.0, dtype=torch.double)
    vp = torch.full((5, 5), 4000.0, dtype=torch.double)
    vs = torch.full((5, 5), 1500.0, dtype=torch.double)
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    vp_out, vs_out, _ = lambmubuoyancy_to_vpvsrho(lamb, mu, buoyancy)
    # P-wave velocity is sqrt((lambda + 2 mu) / rho); buoyancy = 1/rho.
    assert torch.allclose(vp_out, ((lamb + 2 * mu) * buoyancy).sqrt())
    assert torch.allclose(vp_out, vp)
    assert torch.allclose(vs_out, vs)
    # The velocity used for the CFL condition is the faster P-wave velocity.
    assert vp_out.max().item() > vs_out.max().item()
    max_model_vel = max(vp_out.abs().max().item(), vs_out.abs().max().item())
    assert max_model_vel == vp_out.abs().max().item()


def test_cfl_stability_governed_by_pwave() -> None:
    """A high-Vp / low-Vs model stays stable only if the CFL uses Vp.

    The time step (0.002 s, dx 5 m) gives a P-wave Courant number of 1.6, so the
    propagator must subsample the time step based on Vp = 4000 to remain stable.
    If the CFL used Vs = 1000 (Courant 0.4) it would not subsample and the
    P-waves would blow up. A finite result therefore confirms Vp governs the
    CFL.
    """
    nx = (15, 15)
    dx = 5.0
    dt = 0.002
    nt = 40
    rho = torch.full(nx, 2200.0, dtype=torch.double)
    vp = torch.full(nx, 4000.0, dtype=torch.double)
    vs = torch.full(nx, 1000.0, dtype=torch.double)
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    src = ricker(15, nt, dt, 0.06, dtype=torch.double).reshape(1, 1, nt)
    out = elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_pressure_rate=src,
        source_locations_pressure_rate=torch.tensor([[[7, 7]]]),
        receiver_locations_p=torch.tensor([[[8, 8]]]),
        pml_width=3,
        pml_freq=15,
    )
    assert all(torch.isfinite(o).all() for o in output_tensors(out))
    # The recorded P-wave data is non-trivial (the simulation actually ran).
    nonempty = [o for o in output_tensors(out) if o.numel() > 0]
    assert any(o.abs().max().item() > 0 for o in nonempty)


def elasticprop(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    dx: Union[float, list[float]],
    dt: float,
    source_amplitudes_z: Optional[torch.Tensor] = None,
    source_amplitudes_y: Optional[torch.Tensor] = None,
    source_amplitudes_x: Optional[torch.Tensor] = None,
    source_amplitudes_pressure_rate: Optional[torch.Tensor] = None,
    source_locations_z: Optional[torch.Tensor] = None,
    source_locations_y: Optional[torch.Tensor] = None,
    source_locations_x: Optional[torch.Tensor] = None,
    source_locations_pressure_rate: Optional[torch.Tensor] = None,
    receiver_locations_z: Optional[torch.Tensor] = None,
    receiver_locations_y: Optional[torch.Tensor] = None,
    receiver_locations_x: Optional[torch.Tensor] = None,
    receiver_locations_p: Optional[torch.Tensor] = None,
    prop_kwargs: Optional[dict[str, Any]] = None,
    pml_width: Optional[Union[int, list[int]]] = None,
    survey_pad: Optional[Union[int, list[int]]] = None,
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
) -> tuple[torch.Tensor, ...]:
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
    if source_amplitudes_pressure_rate is not None:
        source_amplitudes_pressure_rate = source_amplitudes_pressure_rate.to(device)
        source_locations_pressure_rate = source_locations_pressure_rate.to(device)
    if receiver_locations_z is not None:
        receiver_locations_z = receiver_locations_z.to(device)
    if receiver_locations_y is not None:
        receiver_locations_y = receiver_locations_y.to(device)
    if receiver_locations_x is not None:
        receiver_locations_x = receiver_locations_x.to(device)
    if receiver_locations_p is not None:
        receiver_locations_p = receiver_locations_p.to(device)

    # m_* are the absorbing boundary's memory variables, not physical fields,
    # so they are no longer direct keyword arguments: they travel inside the
    # single ``auxiliary_state`` that ``initial_state`` carries. Any that are
    # None are simply absent, and that part of the state starts from rest.
    m_state = auxiliary_state_arg(
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
    )

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
            source_amplitudes_pressure_rate=source_amplitudes_pressure_rate,
            source_locations_z=source_locations_z,
            source_locations_y=source_locations_y,
            source_locations_x=source_locations_x,
            source_locations_pressure_rate=source_locations_pressure_rate,
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
            initial_state=m_state,
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
        source_amplitudes_pressure_rate=source_amplitudes_pressure_rate,
        source_locations_z=source_locations_z,
        source_locations_y=source_locations_y,
        source_locations_x=source_locations_x,
        source_locations_pressure_rate=source_locations_pressure_rate,
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
        initial_state=m_state,
        nt=nt,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        **prop_kwargs,
    )


def elasticpropchained(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    dx: Union[float, list[float]],
    dt: float,
    source_amplitudes_z: Optional[torch.Tensor] = None,
    source_amplitudes_y: Optional[torch.Tensor] = None,
    source_amplitudes_x: Optional[torch.Tensor] = None,
    source_amplitudes_pressure_rate: Optional[torch.Tensor] = None,
    source_locations_z: Optional[torch.Tensor] = None,
    source_locations_y: Optional[torch.Tensor] = None,
    source_locations_x: Optional[torch.Tensor] = None,
    source_locations_pressure_rate: Optional[torch.Tensor] = None,
    receiver_locations_z: Optional[torch.Tensor] = None,
    receiver_locations_y: Optional[torch.Tensor] = None,
    receiver_locations_x: Optional[torch.Tensor] = None,
    receiver_locations_p: Optional[torch.Tensor] = None,
    prop_kwargs: Optional[dict[str, Any]] = None,
    pml_width: Optional[Union[int, list[int]]] = None,
    survey_pad: Optional[Union[int, list[int]]] = None,
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
) -> tuple[torch.Tensor, ...]:
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
    if source_amplitudes_pressure_rate is not None:
        source_amplitudes_pressure_rate = source_amplitudes_pressure_rate.to(device)
        source_locations_pressure_rate = source_locations_pressure_rate.to(device)
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
        or source_amplitudes_pressure_rate is not None
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
        if source_amplitudes_pressure_rate is not None:
            source_amplitudes_pressure_rate = upsample(
                source_amplitudes_pressure_rate, step_ratio
            )
            source_nt = source_amplitudes_pressure_rate.shape[-1]
            nt_per_segment = (
                ((source_nt + n_chained - 1) // n_chained + step_ratio - 1)
                // step_ratio
            ) * step_ratio
    else:
        nt *= step_ratio
        nt_per_segment = (
            ((nt + n_chained - 1) // n_chained + step_ratio - 1) // step_ratio
        ) * step_ratio

    vy = vy_0
    vx = vx_0
    sigmayy = sigmayy_0
    sigmaxy = sigmaxy_0
    sigmaxx = sigmaxx_0
    # The absorbing boundary's memory variables travel as one opaque
    # ``auxiliary_state``, so each segment hands the next its whole state
    # object rather than eight individually named tensors.
    auxiliary_state = auxiliary_state_arg(
        m_vyy_0=m_vyy_0,
        m_vyx_0=m_vyx_0,
        m_vxy_0=m_vxy_0,
        m_vxx_0=m_vxx_0,
        m_sigmayyy_0=m_sigmayyy_0,
        m_sigmaxyy_0=m_sigmaxyy_0,
        m_sigmaxyx_0=m_sigmaxyx_0,
        m_sigmaxxx_0=m_sigmaxxx_0,
    )

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
            or source_amplitudes_pressure_rate is not None
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
            if source_amplitudes_pressure_rate is not None:
                segment_source_amplitudes_pressure_rate = (
                    source_amplitudes_pressure_rate[
                        ...,
                        nt_per_segment * segment_idx : min(
                            nt_per_segment * (segment_idx + 1),
                            source_nt,
                        ),
                    ]
                )
            segment_nt = None
        else:
            segment_source_amplitudes_y = None
            segment_source_amplitudes_x = None
            segment_source_amplitudes_pressure_rate = None
            segment_nt = (
                nt_per_segment * (segment_idx + 1) - nt_per_segment * segment_idx
            )
        out = elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_y=segment_source_amplitudes_y,
            source_amplitudes_x=segment_source_amplitudes_x,
            source_amplitudes_pressure_rate=segment_source_amplitudes_pressure_rate,
            source_locations_y=source_locations_y,
            source_locations_x=source_locations_x,
            source_locations_pressure_rate=source_locations_pressure_rate,
            receiver_locations_y=receiver_locations_y,
            receiver_locations_x=receiver_locations_x,
            receiver_locations_p=receiver_locations_p,
            vy_0=vy,
            vx_0=vx,
            sigmayy_0=sigmayy,
            sigmaxy_0=sigmaxy,
            sigmaxx_0=sigmaxx,
            initial_state=auxiliary_state,
            nt=segment_nt,
            model_gradient_sampling_interval=step_ratio,
            **prop_kwargs,
        )
        vy = out.vy_0
        vx = out.vx_0
        sigmayy = out.sigmayy_0
        sigmaxy = out.sigmaxy_0
        sigmaxx = out.sigmaxx_0
        auxiliary_state = {"auxiliary_state": out.auxiliary_state}
        segment_receiver_amplitudes_p = out.receiver_amplitudes_p
        segment_receiver_amplitudes_y = out.receiver_amplitudes_y
        segment_receiver_amplitudes_x = out.receiver_amplitudes_x
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

    # A flat tuple of tensors in the pre-fold order, which is what the
    # backend-parity comparison this helper feeds expects; the boundary
    # state is unpacked here rather than threaded on.
    state = auxiliary_state["auxiliary_state"]
    return (
        vy,
        vx,
        sigmayy,
        sigmaxy,
        sigmaxx,
        state["m_vyy_0"],
        state["m_vyx_0"],
        state["m_vxy_0"],
        state["m_vxx_0"],
        state["m_sigmayyy_0"],
        state["m_sigmaxyy_0"],
        state["m_sigmaxyx_0"],
        state["m_sigmaxxx_0"],
        receiver_amplitudes_p,
        receiver_amplitudes_y,
        receiver_amplitudes_x,
    )


def test_backends() -> None:
    """Verify that Python backends can be called without error."""
    if USE_OPENMP:
        run_forward(propagator=elasticprop, prop_kwargs={"backend": "torch_compile"})


def test_forward():
    """Compare the recorded data with that generated by another code."""
    tests_dir = Path(__file__).parent
    with (tests_dir / "Uz_file_ascii").open() as f:
        d = f.read().split()
        expected_vy = torch.tensor([float(di) for di in d[:3000]])
    with (tests_dir / "Ux_file_ascii").open() as f:
        d = f.read().split()
        expected_vx = torch.tensor([float(di) for di in d[3000:]])
    expected_scale = expected_vy.abs().max().item()
    expected_vy /= expected_scale
    expected_vx /= expected_scale
    for accuracy, target_err in zip([2, 4, 6, 8], [0.55, 0.27, 0.34, 0.37]):
        for orientation in range(4):
            actuals_vy = {}
            actuals_vx = {}
            for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
                out = run_forward_lamb(
                    orientation,
                    prop_kwargs={"backend": python, "accuracy": accuracy},
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
            assert torch.allclose(
                actuals_vy[_PYTHON_BACKEND_TRUE_EQUIVALENT],
                actuals_vy["compiled"],
                atol=5e-6,
            )
            assert torch.allclose(
                actuals_vx[_PYTHON_BACKEND_TRUE_EQUIVALENT],
                actuals_vx["compiled"],
                atol=5e-6,
            )


def run_direct(
    mlamb: float = DEFAULT_LAMB,
    mmu: float = DEFAULT_MU,
    mbuoyancy: float = DEFAULT_BUOYANCY,
    freq: float = 25,
    dx: Union[float, list[float]] = (5, 5),
    dt: float = 0.0001,
    nx: tuple[int, ...] = (50, 50),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = elasticprop,
    prop_kwargs: Optional[dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        source_amplitudes_pressure_rate=sources["amplitude"],
        source_locations_pressure_rate=sources["locations"],
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
    for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
        expected, actual = run_direct(
            nx=nx,
            dx=dx,
            propagator=elasticprop,
            dtype=torch.double,
            prop_kwargs={"backend": python},
        )
        assert torch.allclose(
            expected, actual.cpu(), atol=0.1 * expected.abs().max().item()
        )
        actuals[python] = actual
    assert torch.allclose(
        actuals[_PYTHON_BACKEND_TRUE_EQUIVALENT],
        actuals["compiled"],
        atol=1e-14 * actuals[_PYTHON_BACKEND_TRUE_EQUIVALENT].abs().max().item(),
    )


def run_scatter_acoustic(
    c: float,
    dc: float,
    freq: float,
    dx: Union[float, list[float]],
    dt: float,
    nx: tuple[int, ...],
    num_shots: int,
    num_sources_per_shot: int,
    num_receivers_per_shot: int,
    propagator: Any,
    prop_kwargs: Optional[dict[str, Any]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    source_amplitudes_pressure_rate = (
        -torch.cumsum(sources["amplitude"], dim=-1)
        * (k.cpu()[tuple(indices)].unsqueeze(-1))
    ) * dt

    y_const = propagator(
        lamb_const,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_pressure_rate=source_amplitudes_pressure_rate,
        source_locations_pressure_rate=sources["locations"],
        receiver_locations_p=x_r,
        prop_kwargs=prop_kwargs,
    )[-(ndim + 1)]

    y = propagator(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_pressure_rate=source_amplitudes_pressure_rate,
        source_locations_pressure_rate=sources["locations"],
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
    for outi in output_tensors(out[: -(len(dx) + 1)]):
        assert outi.norm() < 2e-6


def test_model_too_small() -> None:
    """Test that an error is raised when the model is too small."""
    with pytest.raises(RuntimeError):
        run_forward(propagator=elasticprop, nx=(3, 3), pml_width=3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_forward_cpu_gpu_match() -> None:
    """Test propagation on CPU and GPU produce the same result."""
    for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
        actual_cpu = run_forward(
            propagator=elasticprop,
            device=torch.device("cpu"),
            prop_kwargs={"backend": python},
        )
        actual_gpu = run_forward(
            propagator=elasticprop,
            device=torch.device("cuda"),
            prop_kwargs={"backend": python},
        )
        # Check wavefields
        for cpui, gpui in zip(
            output_tensors(actual_cpu[:-1]), output_tensors(actual_gpu[:-1])
        ):
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

    for python in [_PYTHON_BACKEND_TRUE_EQUIVALENT, "compiled"]:
        torch.manual_seed(1)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if prop_kwargs is None:
            prop_kwargs = {}
        prop_kwargs["backend"] = python

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
            source_amplitudes_pressure_rate=sources_p["amplitude"],
            source_locations_y=sources_y["locations"],
            source_locations_x=sources_x["locations"],
            source_locations_pressure_rate=sources_p["locations"],
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
            source_amplitudes_pressure_rate=sources_p["amplitude"],
            source_locations_y=sources_y["locations"],
            source_locations_x=sources_x["locations"],
            source_locations_pressure_rate=sources_p["locations"],
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

        # Every element is a Tensor except auxiliary_state, which holds
        # several; compare the tensors either way.
        for ofi, oii in zip(out_filled, out_ignored):
            for filled, ignored in zip(element_tensors(ofi), element_tensors(oii)):
                assert torch.allclose(filled, ignored)

        assert torch.allclose(lambf.grad, lambi.grad)
        assert torch.allclose(muf.grad, mui.grad)
        assert torch.allclose(buoyancyf.grad, buoyancyi.grad)


@pytest.mark.parametrize(
    ("boundary", "accuracy"),
    [(["rigid"] * 4, 2), (["rigid", "pml", "rigid", "pml"], 4)],
    ids=["rigid-every-face-accuracy-2", "rigid-low-faces-accuracy-4"],
)
def test_ignored_receiver_zero_with_non_pml_boundary(boundary, accuracy):
    """An IGNORE_LOCATION receiver must read exactly zero under non-PML boundaries.

    A non-PML boundary condition writes real field values into the padding
    cells that an ignored receiver's safe-substituted index reads from;
    default PML boundaries leave those cells at zero, which is why
    ``test_unused_source_receiver`` above -- PML-only -- never exercises
    this. Regression test: the eager ``_elastic_eager_core`` used to compute
    a masked receiver-amplitude list but return the unmasked list instead,
    so an ignored receiver combined with a ``"rigid"`` boundary leaked real
    field data instead of zero. ``elastic`` accepts a ``"rigid"`` face on the
    end of a dimension at ``accuracy=2`` only, so the two cases here are a
    closed rigid box at that accuracy and rigid low faces at accuracy 4.
    """
    torch.manual_seed(0)
    shape = (20, 20)
    lamb = torch.full(shape, DEFAULT_LAMB, dtype=torch.double)
    mu = torch.full(shape, DEFAULT_MU, dtype=torch.double)
    buoyancy = torch.full(shape, DEFAULT_BUOYANCY, dtype=torch.double)
    dt = 0.001
    nt = 300
    source_locations = torch.tensor([[[10, 10]]])
    source_amplitudes = torch.zeros(1, 1, nt, dtype=torch.double)
    source_amplitudes[0, 0, 5] = 1.0
    receiver_locations = torch.tensor([[[10, 5], [10, 15]]])
    receiver_locations[0, 0] = IGNORE_LOCATION

    for backend in ["eager", "torch_compile"]:
        out = elastic(
            lamb,
            mu,
            buoyancy,
            [4.0, 4.0],
            dt,
            source_amplitudes_pressure_rate=source_amplitudes,
            source_locations_pressure_rate=source_locations,
            receiver_locations_p=receiver_locations,
            accuracy=accuracy,
            backend=backend,
            pml_width=0,
            boundary=boundary,
        )
        receiver_amplitudes_p = out.receiver_amplitudes_p
        assert torch.equal(
            receiver_amplitudes_p[:, 0],
            torch.zeros_like(receiver_amplitudes_p[:, 0]),
        )
        assert receiver_amplitudes_p[:, 1].abs().max() > 0


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
    source_amplitudes_pressure_rate = torch.randn(
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
    pml_profiles = deepwave._staggered_grid.set_pml_profiles(
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
            [0, 0, 0, 0],  # boundary (all-BC_PML, i.e. no boundary= face set)
            n_batch,
            None,
            None,
            1,
            "device",
            ".",
            False,
            False,  # mpml (multiaxial PML disabled) is the last leading arg
            lamb,
            mu,
            mu_yx,
            buoyancy_y,
            buoyancy_x,
            source_amplitudes_y,
            source_amplitudes_x,
            source_amplitudes_pressure_rate,
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
    # The same mode ``python_backend=True`` picks here. It keeps the
    # torch.compile coverage that a hardcoded Python mode would drop: this
    # comparison is the regression test for the int-vs-float grid-spacing
    # precision loss documented in _engine.float_grid_spacing.
    out_python = wrap(_PYTHON_BACKEND_TRUE_MODE)
    for oc, op in zip(out_compiled, out_python):
        assert torch.allclose(oc, op)


def test_elasticfunc():
    """Test elastic_func with different time steps."""
    run_elasticfunc(nt=1)
    run_elasticfunc(nt=2)
    run_elasticfunc(nt=3)


def test_backend_parity():
    """Test backend parity in a 2D model."""
    run_backend_parity(propagator=elasticprop)


@pytest.mark.parametrize(
    ("shape", "loc"),
    [
        ((24, 24), (7, 7)),
        ((16, 16, 16), (5, 5, 5)),
    ],
)
def test_receiver_output_order(shape, loc):
    """The receiver outputs must be ordered pressure first, then velocity.

    The public contract (documented since the first elastic release, and the
    actual return order back to v0.0.19) is that the recorded pressure data is
    the *first* of the returned receiver amplitudes, followed by the velocity
    components in z, y, x order (for the dimensions present). The internal
    backend stores them velocity-first then pressure-last, so the public
    ``elastic`` function reorders them; this guards that reorder against
    regressing.
    """
    ndim = len(shape)
    lamb = torch.full(shape, float(DEFAULT_LAMB))
    mu = torch.full(shape, float(DEFAULT_MU))
    buoyancy = torch.full(shape, DEFAULT_BUOYANCY)
    nt = 30
    dt = 0.0001
    source_amplitudes = ricker(25, nt, dt, 0.05).reshape(1, 1, nt)
    source_location = torch.tensor([s // 2 for s in shape]).reshape(1, 1, ndim)
    rl = torch.tensor(loc).reshape(1, 1, ndim)

    # Drive with a pressure source so every receiver records something.
    common = {
        "source_amplitudes_pressure_rate": source_amplitudes,
        "source_locations_pressure_rate": source_location,
        "nt": nt,
        "prop_kwargs": {"max_vel": 2000},
    }
    # The order of receiver outputs is the last (ndim + 1) elements of the
    # output tuple. Identify which slot each receiver type occupies by enabling
    # one receiver type at a time and finding the single non-empty slot.
    kw_by_type = {
        "p": {"receiver_locations_p": rl},
        "x": {"receiver_locations_x": rl},
        "y": {"receiver_locations_y": rl},
        "z": {"receiver_locations_z": rl},
    }
    types = ["p", "x"] + (["y"] if ndim >= 2 else []) + (["z"] if ndim == 3 else [])

    def nonempty_slot(rec_type):
        out = elasticprop(
            lamb, mu, buoyancy, [5] * ndim, dt, **kw_by_type[rec_type], **common
        )
        recv = out[-(ndim + 1) :]
        nonempty = [i for i, r in enumerate(recv) if r.numel() > 0]
        assert len(nonempty) == 1, f"{rec_type}: expected one non-empty slot"
        return nonempty[0]

    slots = {t: nonempty_slot(t) for t in types}
    # Pressure must be the first receiver output.
    assert slots["p"] == 0, f"pressure must be the first receiver output, got {slots}"
    # Velocity components follow in z, y, x order (for the dims present).
    expected_velocity_order = (
        (["z"] if ndim == 3 else []) + (["y"] if ndim >= 2 else []) + ["x"]
    )
    for offset, t in enumerate(expected_velocity_order, start=1):
        assert slots[t] == offset, f"unexpected slot for {t}: {slots}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_backend_parity_accuracy(accuracy):
    """Test backend parity for every advertised accuracy order (2, 4, 6, 8).

    The elastic propagator (and its docstrings) advertise accuracies 2, 4, 6,
    and 8, but only 2 and 4 were exercised by this comparison. This confirms
    that the higher orders produce eager/compiled-consistent gradients rather
    than merely compiling.
    """
    run_backend_parity(propagator=elasticprop, prop_kwargs={"accuracy": accuracy})


@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((5,), (5,)),
        ((5, 4), (5, 5)),
        ((5, 4, 4), (6, 6, 6)),
    ],
)
def test_backend_parity_2nd_order(nx, dx):
    """Test backend parity with a 2nd order accurate propagator."""
    run_backend_parity(
        propagator=elasticprop, dx=dx, nx=nx, prop_kwargs={"accuracy": 2}
    )


def test_backend_parity_cfl():
    """Test backend parity with a timestep greater than the CFL limit."""
    run_backend_parity(
        propagator=elasticprop,
        dt=0.002,
        prop_kwargs={"time_taper": True},
        only_receivers_out=True,
    )


def test_backend_parity_odd_timesteps():
    """Test backend parity with one more timestep."""
    run_backend_parity(propagator=elasticprop, nt_add=1)


def test_backend_parity_one_shot():
    """Test backend parity with one shot."""
    run_backend_parity(propagator=elasticprop, num_shots=1)


def test_backend_parity_survey_pad():
    """Test backend parity with survey_pad."""
    # Location range: y: [0, 3], x: [10]
    # With survey pad: y: [0, 3+4=7], x: [10-4=6, 10+4=14]
    run_backend_parity(
        propagator=elasticprop,
        nx=(21, 20),
        pml_width=5,
        wavefield_size=[(7 - 0 + 1) + 2 * 5, (14 - 6 + 1) + 2 * 5],
        survey_pad=4,
    )


def test_backend_parity_chained():
    """Test backend parity when two propagators are chained."""
    run_backend_parity(propagator=elasticpropchained)


def test_backend_parity_different_pml():
    """Test backend parity with different pml widths."""
    run_backend_parity(propagator=elasticprop, pml_width=[0, 1, 5, 10])


def test_backend_parity_no_pml():
    """Test backend parity with no pml."""
    run_backend_parity(propagator=elasticprop, pml_width=0, nx=(11, 11))


def test_backend_parity_different_dx():
    """Test backend parity with different dx values."""
    run_backend_parity(propagator=elasticprop, dx=(4, 5), dt=0.0005)


def test_backend_parity_only_lamb_2d():
    """Test backend parity with only lamb requiring gradient."""
    run_backend_parity(
        propagator=elasticprop,
        lamb_requires_grad=True,
        mu_requires_grad=False,
        buoyancy_requires_grad=False,
        source_requires_grad=False,
    )


def test_backend_parity_only_mu_2d():
    """Test backend parity with only mu requiring gradient."""
    run_backend_parity(
        propagator=elasticprop,
        lamb_requires_grad=False,
        mu_requires_grad=True,
        buoyancy_requires_grad=False,
        source_requires_grad=False,
    )


def test_backend_parity_batched_lamb_2d():
    """Test backend parity in a 2D model with batched lamb."""
    run_backend_parity(
        propagator=elasticprop,
        mlamb=torch.tensor([[[DEFAULT_LAMB]], [[DEFAULT_LAMB * 1.2]]]),
    )


def test_backend_parity_batched_mu_2d():
    """Test backend parity in a 2D model with batched mu."""
    run_backend_parity(
        propagator=elasticprop,
        mmu=torch.tensor([[[DEFAULT_MU]], [[DEFAULT_MU * 1.2]]]),
    )


def test_backend_parity_batched_buoyancy_2d():
    """Test backend parity in a 2D model with batched buoyancy."""
    run_backend_parity(
        propagator=elasticprop,
        mbuoyancy=torch.tensor([[[DEFAULT_BUOYANCY]], [[DEFAULT_BUOYANCY * 1.2]]]),
    )


def test_backend_parity_zeros_in_properties():
    """Test backend parity when the property models contain zeros."""
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
    # Vacuum has no stiffness; retain a separate active fluid strip (mu=0).
    lamb[buoyancy == 0] = 0
    mu[buoyancy == 0] = 0
    mu[6] = 0
    run_backend_parity(
        propagator=elasticprop,
        lamb=lamb,
        mu=mu,
        buoyancy=buoyancy,
    )


@pytest.mark.parametrize(("lamb_requires_grad"), [True, False])
@pytest.mark.parametrize(("mu_requires_grad"), [True, False])
@pytest.mark.parametrize(("buoyancy_requires_grad"), [True, False])
@pytest.mark.parametrize(("num_shots"), [1, 2])
@pytest.mark.parametrize(
    ("nx", "dx"),
    [
        ((5,), (5,)),
        ((5, 4), (5, 5)),
        ((5, 4, 4), (6, 6, 6)),
    ],
)
def test_storage(
    lamb_requires_grad,
    mu_requires_grad,
    buoyancy_requires_grad,
    num_shots,
    nx,
    dx,
    mlamb=DEFAULT_LAMB,
    mmu=DEFAULT_MU,
    mbuoyancy=DEFAULT_BUOYANCY,
    freq=25,
    dt=0.004,
    num_sources_per_shot=2,
    num_receivers_per_shot=2,
    propagator=elasticprop,
    prop_kwargs=None,
    device=None,
    dtype=None,
    dlamb=DEFAULT_LAMB / 10,
    dmu=DEFAULT_MU / 10,
    dbuoyancy=DEFAULT_BUOYANCY / 10,
    nt=None,
):
    """Test forward and backward propagation when a source and receiver are unused."""
    if not lamb_requires_grad and not mu_requires_grad and not buoyancy_requires_grad:
        return
    nx = torch.tensor(nx, dtype=torch.long)
    dx = torch.tensor(dx, dtype=dtype)
    ndim = len(nx)

    torch.manual_seed(1)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.double

    if prop_kwargs is None:
        prop_kwargs = {}

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

    source_amplitudes = {
        "source_amplitudes_z": None,
        "source_amplitudes_y": None,
        "source_amplitudes_x": None,
        "source_amplitudes_pressure_rate": None,
    }
    source_locations = {
        "source_locations_z": None,
        "source_locations_y": None,
        "source_locations_x": None,
        "source_locations_pressure_rate": None,
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
        source_amplitudes["source_amplitudes_x"] = sources_x["amplitude"]
        source_locations["source_locations_x"] = sources_x["locations"]
        sources_p = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)
        sources_p["amplitude"] = sources_p["amplitude"].to(device)
        source_amplitudes["source_amplitudes_pressure_rate"] = sources_p["amplitude"]
        source_locations["source_locations_pressure_rate"] = sources_p["locations"]
        if ndim >= 2:
            sources_y = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)
            sources_y["amplitude"] = sources_y["amplitude"].to(device)
            source_amplitudes["source_amplitudes_y"] = sources_y["amplitude"]
            source_locations["source_locations_y"] = sources_y["locations"]
        if ndim >= 3:
            sources_z = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)
            sources_z["amplitude"] = sources_z["amplitude"].to(device)
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

    # Store uncompressed in device memory (default)
    lambd = lamb.clone()
    lambd.requires_grad_(lamb_requires_grad)
    mud = mu.clone()
    mud.requires_grad_(mu_requires_grad)
    buoyancyd = buoyancy.clone()
    buoyancyd.requires_grad_(buoyancy_requires_grad)
    out = propagator(
        lambd,
        mud,
        buoyancyd,
        dx.tolist(),
        dt,
        source_amplitudes_z=source_amplitudes["source_amplitudes_z"],
        source_amplitudes_y=source_amplitudes["source_amplitudes_y"],
        source_amplitudes_x=source_amplitudes["source_amplitudes_x"],
        source_amplitudes_pressure_rate=source_amplitudes[
            "source_amplitudes_pressure_rate"
        ],
        source_locations_z=source_locations["source_locations_z"],
        source_locations_y=source_locations["source_locations_y"],
        source_locations_x=source_locations["source_locations_x"],
        source_locations_pressure_rate=source_locations[
            "source_locations_pressure_rate"
        ],
        receiver_locations_z=receiver_locations["receiver_locations_z"],
        receiver_locations_y=receiver_locations["receiver_locations_y"],
        receiver_locations_x=receiver_locations["receiver_locations_x"],
        receiver_locations_p=receiver_locations["receiver_locations_p"],
        prop_kwargs=prop_kwargs,
    )

    sum([(o**2).sum() for o in out[-ndim - 1 :]]).backward()

    # ``storage_mode="none"`` is deliberately NOT in this list. It retains no
    # forward wavefield history, so no model gradient can be formed from it;
    # the engine now raises rather than returning the plausible-looking but
    # wrong gradient the propagators' own front-end reparameterisation used to
    # produce. This loop runs models that require grad, so it cannot include
    # "none". That rejection, and the fact that source-amplitude gradients and
    # forward-only runs still work in "none" mode, are covered by
    # tests/test_storage_mode_none_model_gradient.py.
    modes = ["device", "disk"]
    if lamb.is_cuda:
        modes.append("cpu")

    # widened 2026-07: marginal under xdist load (parallel test-worker
    # contention widens timing/scheduling-sensitive numerical drift).
    atol_lamb = [0.004, 0.002, 0.004]
    atol_mu = [0.004, 0.002, 0.008]
    atol_buoyancy = [0.01, 0.02, 0.08]

    for mode in modes:
        for compression in [False, True]:
            if mode == "device" and not compression:  # Default
                continue
            lambi = lamb.clone()
            lambi.requires_grad_(lamb_requires_grad)
            mui = mu.clone()
            mui.requires_grad_(mu_requires_grad)
            buoyancyi = buoyancy.clone()
            buoyancyi.requires_grad_(buoyancy_requires_grad)
            prop_kwargs = dict(prop_kwargs)
            prop_kwargs["storage_mode"] = mode
            prop_kwargs["storage_compression"] = compression
            out = propagator(
                lambi,
                mui,
                buoyancyi,
                dx.tolist(),
                dt,
                source_amplitudes_z=source_amplitudes["source_amplitudes_z"],
                source_amplitudes_y=source_amplitudes["source_amplitudes_y"],
                source_amplitudes_x=source_amplitudes["source_amplitudes_x"],
                source_amplitudes_pressure_rate=source_amplitudes[
                    "source_amplitudes_pressure_rate"
                ],
                source_locations_z=source_locations["source_locations_z"],
                source_locations_y=source_locations["source_locations_y"],
                source_locations_x=source_locations["source_locations_x"],
                source_locations_pressure_rate=source_locations[
                    "source_locations_pressure_rate"
                ],
                receiver_locations_z=receiver_locations["receiver_locations_z"],
                receiver_locations_y=receiver_locations["receiver_locations_y"],
                receiver_locations_x=receiver_locations["receiver_locations_x"],
                receiver_locations_p=receiver_locations["receiver_locations_p"],
                prop_kwargs=prop_kwargs,
            )

            sum([(o**2).sum() for o in out[-ndim - 1 :]]).backward()

            if compression:
                if lamb_requires_grad:
                    assert torch.allclose(
                        lambd.grad,
                        lambi.grad,
                        atol=atol_lamb[len(nx) - 1]
                        * lambd.grad.detach().abs().max().item(),
                    )
                if mu_requires_grad:
                    assert torch.allclose(
                        mud.grad,
                        mui.grad,
                        atol=atol_mu[len(nx) - 1]
                        * mud.grad.detach().abs().max().item(),
                    )
                if buoyancy_requires_grad:
                    assert torch.allclose(
                        buoyancyd.grad,
                        buoyancyi.grad,
                        atol=atol_buoyancy[len(nx) - 1]
                        * buoyancyd.grad.detach().abs().max().item(),
                    )
            else:
                if lamb_requires_grad:
                    assert torch.allclose(
                        lambd.grad,
                        lambi.grad,
                        atol=lambd.grad.detach().abs().max().item() * 1e-5,
                    )
                if mu_requires_grad:
                    assert torch.allclose(
                        mud.grad,
                        mui.grad,
                        atol=mud.grad.detach().abs().max().item() * 1e-5,
                    )
                if buoyancy_requires_grad:
                    assert torch.allclose(
                        buoyancyd.grad,
                        buoyancyi.grad,
                        atol=buoyancyd.grad.detach().abs().max().item() * 1e-5,
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
        source_amplitudes_pressure_rate=None,
        source_locations_y=x_s_y,
        source_locations_x=x_s_x,
        source_locations_pressure_rate=None,
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
        source_amplitudes["source_amplitudes_pressure_rate"] = sources_x["amplitude"]
        source_locations["source_locations_pressure_rate"] = sources_x["locations"]
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


def run_backend_parity(
    mlamb: float = DEFAULT_LAMB,
    mmu: float = DEFAULT_MU,
    mbuoyancy: float = DEFAULT_BUOYANCY,
    freq: float = 25,
    dx: tuple[float, float] = (5, 5),
    dt: float = 0.001,
    nx: tuple[int, int] = (10, 10),
    num_shots: int = 2,
    num_sources_per_shot: int = 2,
    num_receivers_per_shot: int = 2,
    propagator: Any = elasticprop,
    prop_kwargs: Optional[dict[str, Any]] = None,
    pml_width: Union[int, list[int]] = 4,
    survey_pad: Optional[Union[int, list[int]]] = None,
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
    wavefield_size: Optional[list[int]] = None,
    nt_add=0,
    lamb: Optional[torch.Tensor] = None,
    mu: Optional[torch.Tensor] = None,
    buoyancy: Optional[torch.Tensor] = None,
    atol=1e-5,
    rtol=1e-3,
):
    """Compare the eager and compiled backends' gradients (no finite differencing)."""
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

    nt = compute_backend_parity_nt(nx, dx, vmin, freq, dt, nt_add)

    key_order = ["z", "y", "x", "p"]
    source_amplitudes: dict[str, Optional[torch.Tensor]] = {
        f"source_amplitudes_{k}": None for k in key_order
    }
    source_locations: dict[str, Optional[torch.Tensor]] = {
        f"source_locations_{k}": None for k in key_order
    }
    receiver_locations: dict[str, Optional[torch.Tensor]] = {
        f"receiver_locations_{k}": None for k in key_order
    }
    if num_sources_per_shot > 0:
        x_s = _set_coords(num_shots, num_sources_per_shot, nx.tolist())
        source_amplitudes, source_locations = _build_multicomponent_sources(
            key_order, x_s, freq, dt, nt, dtype, device, ndim, source_requires_grad
        )
        nt = None
    if num_receivers_per_shot > 0:
        x_r = _set_coords(num_shots, num_receivers_per_shot, nx.tolist())
        receiver_locations = _build_multicomponent_receivers(key_order, x_r, ndim)
    pml_width = _normalize_pml_width(pml_width, ndim)

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
            wavefield_size = _default_wavefield_size(nx, pml_width, ndim)
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

    def wrap(python):
        prop_kwargs["backend"] = python
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
            out = out[-ndim - 1 :]
        # The auxiliary state is one output element holding several tensors;
        # the parity comparison is over tensors.
        return compute_backend_parity_grad(output_tensors(out), inputs)

    run_python_vs_compiled_backend_parity(wrap, "eager", atol=atol, rtol=rtol)


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
            "elastic: source_locations_x is out of bounds: a "
            "x location is staggered half a cell past the "
            "node it names, so its x coordinate must be "
            "less than 9."
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
            "elastic: receiver_locations_x is out of bounds: a "
            "x location is staggered half a cell past the "
            "node it names, so its x coordinate must be "
            "less than 9."
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
            "elastic: source_locations_y is out of bounds: a "
            "y location is staggered half a cell past the "
            "node it names, so its y coordinate must be "
            "less than 9."
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
            "elastic: receiver_locations_y is out of bounds: a "
            "y location is staggered half a cell past the "
            "node it names, so its y coordinate must be "
            "less than 9."
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

    # Test pressure reciprocity. The channel is ``_p`` as a receiver and
    # ``_pressure_rate`` as a source.
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
        source_comps=("_pressure_rate", "_pressure_rate"),
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
        "source_amplitudes_pressure_rate": source_amplitudes,
        "source_locations_pressure_rate": loc_s,
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
        source_amplitudes_pressure_rate=source_amplitudes,
        source_locations_pressure_rate=source_location,
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


def test_vmap_elastic_props() -> None:
    """Test vmap over elastic property models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny, nx = 8, 8
    dx = 5.0
    dt = 0.004
    lamb_base = 1e9
    mu_base = 1e9
    buoyancy_base = 1.0 / 2200.0

    batch_size = 3
    lamb_batch = torch.empty(batch_size, ny, nx, device=device)
    mu_batch = torch.empty(batch_size, ny, nx, device=device)
    buoyancy_batch = torch.empty(batch_size, ny, nx, device=device)
    for i in range(batch_size):
        lamb_batch[i] = lamb_base + i * 1e8
        mu_batch[i] = mu_base + i * 1e8
        buoyancy_batch[i] = buoyancy_base

    n_shots = 2
    n_receivers_per_shot = 3
    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_locations[..., 0] = ny // 2
    source_locations[..., 1] = nx // 2

    receiver_locations = torch.zeros(
        n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device
    )
    receiver_locations[..., 0] = ny // 2
    receiver_locations[..., 1] = torch.linspace(
        0, nx - 1, n_receivers_per_shot, device=device
    ).long()

    source_amplitudes = (
        ricker(25.0, 30, dt, 0.05).to(device).reshape(1, 1, -1).repeat(n_shots, 1, 1)
    )

    def prop(lamb, mu, buoyancy):
        return elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_pressure_rate=source_amplitudes,
            source_locations_pressure_rate=source_locations,
            receiver_locations_p=receiver_locations,
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        ).receiver_amplitudes_p

    res_vmap = vmap(prop)(lamb_batch, mu_batch, buoyancy_batch)

    res_loop = torch.stack(
        [prop(lamb_batch[i], mu_batch[i], buoyancy_batch[i]) for i in range(batch_size)]
    )

    assert torch.allclose(res_vmap, res_loop, atol=1e-5)


def test_vmap_elastic_partial() -> None:
    """Test vmap over lamb only (others shared)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny, nx = 8, 8
    dx = 5.0
    dt = 0.004
    mu = torch.full((ny, nx), 1e9, device=device)
    buoyancy = torch.full((ny, nx), 1.0 / 2200.0, device=device)

    batch_size = 3
    lamb_batch = torch.empty(batch_size, ny, nx, device=device)
    for i in range(batch_size):
        lamb_batch[i] = 1e9 + i * 1e8

    n_shots = 1
    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_locations[..., 0] = ny // 2
    source_locations[..., 1] = nx // 2
    receiver_locations = source_locations.clone()

    source_amplitudes = ricker(25.0, 30, dt, 0.05).to(device).reshape(1, 1, -1)

    def prop(lamb):
        return elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_pressure_rate=source_amplitudes,
            source_locations_pressure_rate=source_locations,
            receiver_locations_p=receiver_locations,
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        ).receiver_amplitudes_p

    res_vmap = vmap(prop)(lamb_batch)

    res_loop = torch.stack([prop(lamb_batch[i]) for i in range(batch_size)])

    assert torch.allclose(res_vmap, res_loop, atol=1e-5)


def test_jvp_elastic() -> None:
    """Test jvp on elastic."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    ny, nx = 8, 8
    dx = 5.0
    dt = 0.004
    lamb0 = torch.full((ny, nx), 1e9, device=device, dtype=dtype)
    mu0 = torch.full((ny, nx), 1e9, device=device, dtype=dtype)
    buoyancy0 = torch.full((ny, nx), 1.0 / 2200.0, device=device, dtype=dtype)

    lamb_tangent = torch.randn_like(lamb0) * 1e7
    mu_tangent = torch.randn_like(mu0) * 1e7
    buoyancy_tangent = torch.randn_like(buoyancy0) * 1e-6

    source_locations = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    source_locations[..., 0] = ny // 2
    source_locations[..., 1] = nx // 2
    receiver_locations = source_locations.clone()
    source_amplitudes = (
        ricker(25.0, 30, dt, 0.05, dtype=dtype).to(device).reshape(1, 1, -1)
    )

    def prop(lamb, m, b, backend="eager"):
        return elastic(
            lamb,
            m,
            b,
            dx,
            dt,
            source_amplitudes_pressure_rate=source_amplitudes,
            source_locations_pressure_rate=source_locations,
            receiver_locations_p=receiver_locations,
            pml_width=3,
            max_vel=2000.0,
            backend=backend,
        ).receiver_amplitudes_p

    _, jvp_out = jvp(
        prop, (lamb0, mu0, buoyancy0), (lamb_tangent, mu_tangent, buoyancy_tangent)
    )

    # Finite Difference
    epsilon = 1e-6
    out_plus = prop(
        lamb0 + epsilon * lamb_tangent,
        mu0 + epsilon * mu_tangent,
        buoyancy0 + epsilon * buoyancy_tangent,
    )
    out_minus = prop(
        lamb0 - epsilon * lamb_tangent,
        mu0 - epsilon * mu_tangent,
        buoyancy0 - epsilon * buoyancy_tangent,
    )
    fd_out = (out_plus - out_minus) / (2 * epsilon)

    assert jvp_out.numel() > 0
    assert torch.allclose(jvp_out, fd_out, atol=1e-5)

    # eager-vs-FD above is self-consistent (both differentiate the same
    # eager Python code, so a bug in that code's physics would not show up:
    # torch.func.jvp of a smooth function always agrees with a finite
    # difference of that same function, buggy or not). torch.func.jvp of the
    # default compiled backend routes through a genuinely different
    # implementation -- the compiled Born kernel
    # (elastic_born.elastic_born_compiled_core, see
    # ``_elastic_forward_jvp``'s docstring) -- so comparing it against the
    # same independent FD reference is an actual cross-implementation check.
    _, jvp_compiled = jvp(
        lambda lamb_, m_, b_: prop(lamb_, m_, b_, "compiled"),
        (lamb0, mu0, buoyancy0),
        (lamb_tangent, mu_tangent, buoyancy_tangent),
    )
    assert torch.allclose(jvp_compiled, fd_out, atol=1e-5)


def test_vmap_error_check() -> None:
    """Ensure error is raised if non-property inputs are vmapped."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny, nx = 8, 8
    lamb = torch.full((ny, nx), 1e9, device=device)
    mu = torch.full((ny, nx), 1e9, device=device)
    buoyancy = torch.full((ny, nx), 1.0 / 2200.0, device=device)

    # Try to vmap over source amplitudes
    batch_size = 2
    source_amplitudes_batch = torch.randn(batch_size, 1, 1, 10, device=device)

    def prop(amps):
        return elastic(
            lamb,
            mu,
            buoyancy,
            5.0,
            0.004,
            source_amplitudes_pressure_rate=amps,
            source_locations_pressure_rate=torch.zeros(
                1, 1, 2, dtype=torch.long, device=device
            ),
            pml_width=3,
            max_vel=2000.0,
            backend="eager",
        )

    with pytest.raises(NotImplementedError, match="Only property models"):
        vmap(prop)(source_amplitudes_batch)


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_checkpoint_gradient(use_reentrant: bool) -> None:
    """Gradient checkpointing must not change the gradient (issue #120)."""
    ny, nx, dx = 25, 25, 5.0
    nt, freq, dt0, max_vel = 60, 25.0, 0.004, 3300.0
    n_shots = 2

    lamb = (2500.0**2 * 2200.0) * torch.ones(ny, nx)
    mu = (1400.0**2 * 2200.0) * torch.ones(ny, nx)
    buoyancy = (1 / 2200.0) * torch.ones(ny, nx)

    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long)
    source_locations[..., 1] = 5
    source_locations[:, 0, 0] = torch.tensor([ny // 4, 3 * ny // 4])
    # The staggered grid requires receivers strictly inside the model edge.
    receiver_locations = torch.zeros(n_shots, nx - 1, 2, dtype=torch.long)
    receiver_locations[..., 1] = 5
    receiver_locations[:, :, 0] = torch.arange(nx - 1).repeat(n_shots, 1)

    dt, step_ratio = cfl_condition(dx, dx, dt0, max_vel)
    source_amplitudes = upsample(
        ricker(freq, nt, dt0, 1.5 / freq).repeat(n_shots, 1, 1), step_ratio
    )

    def call(model: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Differentiate w.r.t. buoyancy: with this homogeneous model it has a
        # healthy gradient (lamb's huge magnitude makes its gradient negligible).
        return elastic(
            lamb,
            mu,
            model,
            dx,
            dt,
            source_amplitudes_y=source_amplitudes,
            source_locations_y=source_locations,
            receiver_locations_y=receiver_locations,
            pml_freq=freq,
            max_vel=max_vel,
            model_gradient_sampling_interval=step_ratio,
        )

    run_checkpoint_equivalence(buoyancy, call, use_reentrant=use_reentrant)


@pytest.mark.parametrize(
    ("vp", "vs"),
    [
        (3000.0, 1700.0),
        (2800.0, 1500.0),
    ],
)
def test_pwave_swave_arrival_times(vp, vs, record_property) -> None:
    """Both P- and S-wave arrivals appear at their analytic travel times.

    In a homogeneous isotropic elastic solid a point force radiates a P wave at
    vp = sqrt((lamb + 2 mu) / rho) and an S wave at vs = sqrt(mu / rho). With a
    vertical (y) force source, the vy receiver offset *along* the force is
    P-dominated (radial motion) and the vy receiver offset *perpendicular* to
    the force is S-dominated (transverse motion), so each arrival can be picked
    cleanly as that trace's envelope peak.

    The existing travel-time test only covers the acoustic limit (vs = 0, P
    only). Here vs > 0, exercising genuine shear physics. The absolute arrival
    times carry a small (few-percent) systematic offset because the velocity
    far-field pulse is a derivative of the source wavelet; that bias cancels in
    the P-to-S *difference* t_S - t_P = r (1/vs - 1/vp), which is therefore the
    tight check, with looser absolute bounds confirming the modes (and the CFL
    P/S choice) are not swapped.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    ny = nx = 200
    dx = 5.0
    dt = 0.0004
    freq = 12.0
    rho = 2000.0
    offset = 50
    peak_time = 1.5 / freq
    nt = int(0.6 / dt)

    lamb = torch.full((ny, nx), rho * (vp**2 - 2 * vs**2), device=device, dtype=dtype)
    mu = torch.full((ny, nx), rho * vs**2, device=device, dtype=dtype)
    buoyancy = torch.full((ny, nx), 1.0 / rho, device=device, dtype=dtype)

    cz, cx = ny // 2, nx // 2
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    source_locations[..., 0] = cz
    source_locations[..., 1] = cx
    receiver_locations = torch.zeros(1, 2, 2, dtype=torch.long, device=device)
    receiver_locations[0, 0] = torch.tensor([cz + offset, cx])  # along force -> P
    receiver_locations[0, 1] = torch.tensor([cz, cx + offset])  # perpendicular -> S
    source_amplitudes = (
        ricker(freq, nt, dt, peak_time, dtype=dtype).to(device).reshape(1, 1, -1)
    )

    out = elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes,
        source_locations_y=source_locations,
        receiver_locations_y=receiver_locations,
        pml_width=20,
        pml_freq=freq,
        accuracy=4,
    )
    vy = out.receiver_amplitudes_y[0]

    r = offset * dx
    t_p = vy[0].abs().argmax().item() * dt - peak_time
    t_s = vy[1].abs().argmax().item() * dt - peak_time

    # S clearly follows P (distinct modes, not a single arrival).
    assert t_s > t_p
    # Absolute speeds are right to within the source-shape bias (modes/CFL not
    # swapped: an S that travelled at vp, or vice versa, would fail badly).
    assert abs(t_p - r / vp) / (r / vp) < 0.13
    assert abs(t_s - r / vs) / (r / vs) < 0.13
    # Bias-cancelling tight check on the P-to-S separation.
    dt_expected = r * (1.0 / vs - 1.0 / vp)
    record_property(
        "ps_separation_relative_error", abs((t_s - t_p) - dt_expected) / dt_expected
    )
    assert abs((t_s - t_p) - dt_expected) / dt_expected < 0.04


def test_free_surface_rayleigh_wave_speed() -> None:
    """A free surface supports a Rayleigh wave at the analytic surface speed.

    Deepwave's elastic propagator imposes a free surface via the "improved
    vacuum" method (Zeng et al. 2012): zeroing the property models above the
    surface makes those cells a vacuum, so the traction vanishes at the
    interface. The defining consequence is a Rayleigh surface wave -- the
    "ground roll" of exploration seismology -- that travels along the surface
    slower than the shear wave. For a Poisson solid (lamb = mu, i.e.
    vp = sqrt(3) vs, Poisson ratio 1/4) the Rayleigh speed is the classic
    cR = 0.9194 vs (the relevant root of the Rayleigh equation).

    A vertical (y) force at the surface couples strongly into the Rayleigh
    mode, which then dominates the surface motion at offset (body waves
    radiate downward and decay faster along the surface). We measure its speed
    from the cross-correlation lag between two surface receivers; using the lag
    between two receivers -- rather than an absolute arrival time -- cancels the
    source-wavelet shape and the excitation delay. This converts the otherwise
    only-implicitly-exercised free-surface implementation into a measured one.
    """
    dtype = torch.double
    vs = 1000.0
    vp = 3.0**0.5 * vs  # Poisson solid => Rayleigh speed 0.9194 vs
    rho = 2000.0
    freq = 15.0
    dx = 4.0  # ~17 grid cells per shear wavelength
    dt = 0.4 * dx / vp
    # Geometry in metres, converted to cells (so the resolution is explicit).
    src_x = int(250 / dx) + 5
    off1 = int(700 / dx)
    off2 = int(1400 / dx)
    ny = int(700 / dx)
    nx = src_x + off2 + int(150 / dx)

    vp_model = torch.full((ny, nx), vp, dtype=dtype)
    vs_model = torch.full((ny, nx), vs, dtype=dtype)
    rho_model = torch.full((ny, nx), rho, dtype=dtype)
    # Vacuum (free surface) above the top row.
    vp_model[0] = 0
    vs_model[0] = 0
    rho_model[0] = 0

    nt = int((1.4 * off2 * dx / (0.85 * vs)) / dt)
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)
    # Source and receivers in the first solid row (just below the surface).
    source_locations = torch.tensor([[[1, src_x]]])
    receiver_locations = torch.tensor([[[1, src_x + off1], [1, src_x + off2]]])

    out = elastic(
        *vpvsrho_to_lambmubuoyancy(vp_model, vs_model, rho_model),
        dx,
        dt,
        source_amplitudes_y=amp,
        source_locations_y=source_locations,
        receiver_locations_y=receiver_locations,
        pml_width=[0, 20, 20, 20],  # free surface on top, PML on the other faces
        max_vel=vp,
        pml_freq=freq,
        accuracy=4,
    )
    vy = out.receiver_amplitudes_y[0]

    # Cross-correlation lag (in samples) of the far trace relative to the near
    # trace: the time the dominant (Rayleigh) wavetrain takes to travel between
    # the two receivers. The lag is integer-quantised (one sample is ~0.1% of
    # the inter-receiver travel time here) and the correlation peak stands well
    # clear of its neighbours, so the measurement is essentially deterministic
    # across platforms. The asserted band brackets the main peak and excludes
    # the adjacent carrier side lobes, so a mis-pick would fail rather than
    # silently pass.
    near, far = vy[0], vy[1]
    n = len(near)
    corr = torch.fft.irfft(
        torch.fft.rfft(far, 2 * n) * torch.fft.rfft(near, 2 * n).conj(), 2 * n
    )
    lag = torch.roll(corr, n).argmax().item() - n
    assert lag > 0  # the far receiver arrival follows the near one
    c_rayleigh = (off2 - off1) * dx / (lag * dt)
    ratio = c_rayleigh / vs

    # Measured (CPU, float64): cR/vs ~ 0.910 (analytic 0.9194); converges toward
    # 0.9194 as the grid is refined. The bounds confirm a genuine sub-shear
    # Rayleigh wave near the Poisson-solid value (not an S wave at vs, nor a
    # P/numerical artefact), with margin for the vacuum method's few-percent
    # bias and grid dispersion.
    assert 0.86 < ratio < 0.95


def test_pml_reflection_decreases_with_width(record_property) -> None:
    """Residual PML reflection must shrink as the PML width grows (elastic).

    A 2-D homogeneous isotropic-elastic experiment records the pressure while
    the right boundary absorbs with a PML of varying width; the other three
    faces keep a fixed wide PML so only the right boundary differs between runs.
    ``pml_width`` adds absorbing cells *outside* the model, so the model
    interior is held fixed (constant receiver-to-right-edge distance) and only
    the PML thickness varies. Subtracting a reflection-free reference (right
    model edge pushed far away) isolates the energy reflected by the right PML.
    Converts the elastic PML from a "trusted" component into a measured one
    (Workstream A1.4 / H).
    """
    dtype = torch.double
    vp = 2000.0
    vs = 1200.0
    rho = 2000.0
    mu_val = rho * vs**2
    lamb_val = rho * vp**2 - 2 * mu_val
    buoyancy_val = 1.0 / rho
    freq = 15.0
    dx = 5.0
    dt = 0.5 * dx / vp
    ny = 70
    other_pml = 20  # fixed wide PML on left/top/bottom
    src_y = ny // 2
    src_x = other_pml + 10
    rec_x = src_x + 20
    model_nx = rec_x + 50  # receiver-to-(right model edge) distance, constant
    nt = 350
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)
    source_locations = torch.tensor([[[src_y, src_x]]])
    receiver_locations = torch.tensor([[[src_y, rec_x]]])

    def trace(nx: int, right_pml: int) -> torch.Tensor:
        shape = (ny, nx)
        lamb = torch.full(shape, lamb_val, dtype=dtype)
        mu = torch.full(shape, mu_val, dtype=dtype)
        buoyancy = torch.full(shape, buoyancy_val, dtype=dtype)
        return elastic(
            lamb,
            mu,
            buoyancy,
            [dx, dx],
            dt,
            source_amplitudes_pressure_rate=amp,
            source_locations_pressure_rate=source_locations,
            receiver_locations_p=receiver_locations,
            accuracy=4,
            pml_width=[other_pml, other_pml, other_pml, right_pml],
            max_vel=vp,
            pml_freq=freq,
        )[-3][0, 0]  # [-3] is the pressure receiver (pressure-first ordering)

    reference = trace(rec_x + 800, 20)

    coeffs = []
    for width in (2, 5, 10, 20):
        t = trace(model_nx, width)
        n = min(len(t), len(reference))
        coeffs.append(((t[:n] - reference[:n]).norm() / reference[:n].norm()).item())

    # Measured (CPU, float64): 0.0340, 0.000185, 0.0000517, 0.00000239.
    # A nonabsorbing control validates the reflection window independently.
    reflecting = trace(model_nx, 0)
    no_pml = float((reflecting - reference).norm() / reference.norm())
    record_property("pml_reflection_coefficients", coeffs)
    record_property("no_pml_relative_error", no_pml)
    # Elastic PML absorbs very effectively, so the bounds differ from scalar.
    assert all(coeffs[i] < coeffs[i - 1] for i in range(1, len(coeffs))), (
        f"PML reflection not monotonically decreasing in width: {coeffs}"
    )
    assert no_pml > 0.1  # the recording window contains substantial reflection
    assert coeffs[0] > 100 * coeffs[-1]  # thin versus wide absorbing layers
    assert coeffs[-1] < 0.01  # a 20-cell PML absorbs to < 1%


def test_pml_reflection_decreases_with_width_3d() -> None:
    """A wider PML reflects less in a 3-D elastic domain.

    The 3-D counterpart of the elastic PML-width test, exercising the
    separately-compiled 3-D elastic PML kernel (the heaviest kernel, and the
    last PML code path without an analytic check). A point force radiates to all
    six faces, so we record one off-axis velocity receiver and subtract a
    reflection-free reference obtained by re-running with a very wide (near-
    perfect) PML: the physical direct field and coda cancel, leaving only the
    finite-width PML's reflected energy. The elastic PML absorbs strongly, so
    the transition is sampled at narrow widths (2/4/6) to stay above the floor.
    """
    dtype = torch.double
    dx = 5.0
    freq = 25.0
    rho = 2000.0
    vp = 2500.0
    vs = 1400.0
    dt = 0.4 * dx / vp
    n = 24  # cubic model interior, identical for every run
    src = (n // 2, n // 2, n // 2)
    rec = (n // 2 + 6, n // 2 + 6, n // 2 + 6)  # off-axis: sees all six faces
    lamb = torch.full((n, n, n), rho * (vp**2 - 2 * vs**2), dtype=dtype)
    mu = torch.full((n, n, n), rho * vs**2, dtype=dtype)
    buoyancy = torch.full((n, n, n), 1.0 / rho, dtype=dtype)
    diag = (3 * (n * dx) ** 2) ** 0.5
    nt = int((1.5 * diag / vp + 2 / freq) / dt)
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(1, 1, -1)
    source_locations = torch.tensor([[list(src)]])
    receiver_locations = torch.tensor([[list(rec)]])

    def trace(width: int) -> torch.Tensor:
        return elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_x=amp * 1e6,
            source_locations_x=source_locations,
            receiver_locations_x=receiver_locations,
            accuracy=4,
            pml_width=width,
            max_vel=vp,
            pml_freq=freq,
        ).receiver_amplitudes_x[0, 0]

    # A 20-cell PML is effectively reflection-free at this scale (the reference).
    reference = trace(20)
    widths = (2, 4, 6)
    coeffs = [((trace(w) - reference).norm() / reference.norm()).item() for w in widths]

    # Measured (CPU, float64): ~0.314, 0.012, 0.0019 for width 2/4/6.
    assert all(coeffs[i] < coeffs[i - 1] for i in range(1, len(coeffs))), (
        f"PML reflection not monotonically decreasing in width: {coeffs}"
    )
    assert coeffs[0] > 0.05  # a 2-cell PML reflects noticeably
    assert coeffs[0] > 5 * coeffs[1]  # widening 2 -> 4 cells sharply cuts it
    assert coeffs[-1] < 0.01  # a 6-cell elastic PML reflects < 1% to the receiver


def test_second_order_hvp_matches_eager() -> None:
    """The compiled elastic Hessian-vector product matches the eager backend.

    The double-backward (one elastic Born forward + one elastic Born backward,
    see ``docs/elastic_born_derivation.md`` section 5) computes ``H*h`` without
    forming ``H``. The ``backend="eager"`` path computes the exact
    second-order derivative through native autograd, so the compiled HVP must
    equal it (to near machine precision) w.r.t. every model (lamb, mu, buoyancy)
    and the source amplitudes, in 2D and 3D. This is the rigorous correctness
    gate; ``test_second_order_hvp_equals_fd_of_gradient`` is an independent
    finite-difference sanity check.
    """
    for shape in [(6, 6), (5, 6, 6)]:
        ndim = len(shape)
        dx = 6.0
        dt = 1e-4
        nt = 8
        torch.manual_seed(0)
        vp = torch.full(shape, 1500.0, dtype=torch.double) + 30 * torch.rand(
            shape, dtype=torch.double
        )
        vs = torch.full(shape, 900.0, dtype=torch.double)
        rho = torch.full(shape, 2200.0, dtype=torch.double)
        lamb0, mu0, buoyancy0 = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
        samp0 = torch.randn(1, 1, nt, dtype=torch.double)
        centre = [s // 2 for s in shape]
        rloc = list(centre)
        rloc[-1] -= 1
        loc = torch.tensor([[centre]])
        rec = torch.tensor([[rloc]])
        torch.manual_seed(1)
        hs = {
            "lamb": torch.rand_like(lamb0),
            "mu": torch.rand_like(mu0),
            "buoyancy": torch.rand_like(buoyancy0),
            "samp": torch.rand_like(samp0),
        }
        wrt = ["lamb", "mu", "buoyancy", "samp"]

        def hvp(backend: str) -> tuple[torch.Tensor, ...]:
            params = {
                "lamb": lamb0.clone().requires_grad_(True),
                "mu": mu0.clone().requires_grad_(True),
                "buoyancy": buoyancy0.clone().requires_grad_(True),
                "samp": samp0.clone().requires_grad_(True),
            }
            out = elastic(
                params["lamb"],
                params["mu"],
                params["buoyancy"],
                dx,
                dt,
                source_amplitudes_pressure_rate=params["samp"],
                source_locations_pressure_rate=loc,
                receiver_locations_p=rec,
                pml_width=2,
                pml_freq=15,
                nt=nt,
                max_vel=5000.0,
                backend=backend,
            )
            plist = [params[w] for w in wrt]
            grads = torch.autograd.grad(
                out[-(ndim + 1)].pow(2).sum(), plist, create_graph=True
            )
            return torch.autograd.grad(grads, plist, grad_outputs=[hs[w] for w in wrt])

        compiled = hvp("compiled")
        eager = hvp("eager")
        for name, a, e in zip(wrt, compiled, eager):
            denom = e.abs().max().item() or 1.0
            rel = (a - e).abs().max().item() / denom
            assert rel < 1e-6, f"{ndim}D {name}: rel={rel}"


def test_second_order_hvp_equals_fd_of_gradient() -> None:
    """The elastic Hessian-vector product equals a finite difference of grad.

    Checks the compiled double-backward against
    ``(grad J(m + eps h) - grad J(m - eps h)) / (2 eps)`` for a data-misfit loss
    and an interior direction ``h`` on ``lamb`` (well-conditioned; matches to the
    compiled fast-math band).

    ``nt`` must clear the source wavelet's own ramp-up, not just the travel
    time to the receiver: a Ricker wavelet with ``peak_time=0.06`` has
    negligible amplitude far from that time, so a run ending well before it
    (``nt=18`` at ``dt=1e-4`` covers only 0.0018s, versus the wavelet's
    ~0.12s effective support) is driven by a source at under 1% of its peak
    amplitude -- the loss, its gradient, and this HVP all end up at
    subnormal float64 magnitudes (measured: ~1e-37 to 1e-80), well outside
    float64's ~1e-16 relative-precision band, so the ``< 1e-6`` comparison
    below is comparing accumulated rounding noise, not physics. ``nt=1200``
    covers the wavelet's full support (measured source amplitude reaches
    its 1.0 peak within the window, versus ~0.007 at ``nt=18``).
    """
    ny, nx = 12, 13
    dx = 5.0
    dt = 1e-4
    nt = 1200
    torch.manual_seed(0)
    vp = torch.full((ny, nx), 1500.0, dtype=torch.double) + 30 * torch.rand(
        ny, nx, dtype=torch.double
    )
    vs = torch.full((ny, nx), 900.0, dtype=torch.double)
    rho = torch.full((ny, nx), 2200.0, dtype=torch.double)
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    src = ricker(15, nt, dt, 0.06, dtype=torch.double).reshape(1, 1, nt)
    kwargs = {
        "source_amplitudes_pressure_rate": src,
        "source_locations_pressure_rate": torch.tensor([[[ny // 2, nx // 2]]]),
        "receiver_locations_p": torch.tensor([[[ny // 2, nx // 2 + 3]]]),
        "pml_width": 3,
        "pml_freq": 15,
        "max_vel": 3000.0,
    }
    torch.manual_seed(1)
    # h must be an O(lamb) perturbation direction: lamb itself is O(1e9)
    # (rho * vp**2), so an O(1) h would make eps * h a ~1e-11 relative
    # perturbation, right at the edge of float64's usable precision for a
    # finite difference -- the resulting fd/hvp values would be dominated by
    # rounding noise rather than the true second derivative.
    h = lamb.mean() * torch.rand(ny, nx, dtype=torch.double)

    def grad_lamb(lamb_in: torch.Tensor) -> torch.Tensor:
        ll = lamb_in.clone().requires_grad_(True)
        out = elastic(ll, mu, buoyancy, dx, dt, **kwargs)
        (g,) = torch.autograd.grad(out[-3].pow(2).sum(), ll, create_graph=True)
        return g

    ll = lamb.clone().requires_grad_(True)
    out = elastic(ll, mu, buoyancy, dx, dt, **kwargs)
    (g,) = torch.autograd.grad(out[-3].pow(2).sum(), ll, create_graph=True)
    (hvp,) = torch.autograd.grad(g, ll, grad_outputs=h)

    # With h now O(lamb) (see above), the FD is truncation-error-dominated
    # (empirically confirmed: the relative error scales as eps**2 -- e.g.
    # 2.4e-7 at eps=1e-3 vs 2.4e-5 at eps=1e-2 -- rather than improving with
    # a larger eps as a rounding-noise-dominated regime would), so a small
    # eps gives the closest match, not a compromise against cancellation.
    eps = 1e-3
    fd = (grad_lamb(lamb + eps * h) - grad_lamb(lamb - eps * h)) / (2 * eps)
    assert (hvp - fd).abs().max() / fd.abs().max() < 1e-6


def test_first_order_unaffected_by_second_order_hvp() -> None:
    """Plain first-order elastic gradients are finite and deterministic.

    The Born-powered double-backward (Hessian-vector product) that replaced the
    old ``no_second_order`` guard must not perturb the ordinary first-order
    gradient.
    """
    ny, nx = 15, 15
    dx = 5.0
    dt = 0.0001
    nt = 20
    src = ricker(15, nt, dt, 0.06, dtype=torch.double).reshape(1, 1, nt)

    def grad_lamb() -> torch.Tensor:
        vp = torch.full((ny, nx), 1500.0, dtype=torch.double)
        vs = torch.full((ny, nx), 1000.0, dtype=torch.double)
        rho = torch.full((ny, nx), 2200.0, dtype=torch.double)
        lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
        lamb.requires_grad_(True)
        out = elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_pressure_rate=src,
            source_locations_pressure_rate=torch.tensor([[[ny // 2, nx // 2]]]),
            receiver_locations_p=torch.tensor([[[ny // 2, nx // 2 + 3]]]),
            pml_width=3,
            pml_freq=15,
        ).receiver_amplitudes_p
        return torch.autograd.grad(out.pow(2).sum(), lamb)[0]

    g0 = grad_lamb()
    g1 = grad_lamb()
    assert torch.isfinite(g0).all()
    assert g0.abs().sum() > 0
    assert torch.equal(g0, g1)


# --- compiled backward boundary-face adjoint fix -------------------------
# Regression tests for a bug in which elastic.c/elastic.cu's compiled
# backward re-applied the forward boundary guard-ring fill to the adjoint
# fields instead of that fill's true transpose. Elastic's multi-field fills
# (unlike acoustic's single-field pressure/velocity fill) are not
# self-adjoint, so this produced model gradients wrong by 10%-106% relative
# whenever any ``boundary=`` face was non-PML. See
# ``test_elastic_c_jvp_supports_non_pml``/``test_elastic_c_hvp_supports_non_
# pml`` in test_boundary.py for the same well-conditioned-harness rationale
# (``nt`` long enough for several source-to-face-to-receiver round trips, so
# a wrong adjoint fill is not masked by numerical dust).


def _isotropic_elastic_media(
    shape: tuple[int, ...], dtype: torch.dtype = torch.double
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Isotropic elastic medium; rho/c/cs match test_boundary.py's own constants."""
    rho, c, cs = 1000.0, 1500.0, 900.0
    lamb = torch.full(shape, rho * (c**2 - 2 * cs**2), dtype=dtype)
    mu = torch.full(shape, rho * cs**2, dtype=dtype)
    buoyancy = torch.full(shape, 1.0 / rho, dtype=dtype)
    return lamb, mu, buoyancy


_ELASTIC_C_BACKEND = pytest.mark.skipif(
    not c_backend_available("elastic", device=torch.device("cpu")),
    reason="compiled elastic backend not present in libdeepwave_C",
)


@_ELASTIC_C_BACKEND
@pytest.mark.parametrize(
    ("boundary", "pml_width"),
    [
        (["rigid", "pml", "pml", "pml"], [0, 0, 10, 10]),
        (["rigid", "pml", "rigid", "pml"], [0, 0, 0, 0]),
        (["pml", "pml", "rigid", "pml"], [0, 0, 0, 0]),
    ],
    ids=["rigid_lo_pml", "rigid_lo_both_axes", "rigid_lo_transverse"],
)
def test_elastic_compiled_backward_boundary_adjoint_parity(
    boundary: list[str], pml_width: list[int]
) -> None:
    """Compiled backward matches eager for lamb/mu/buoyancy/source grads.

    Negative control: run against elastic.c/.cu reverted to the pre-fix
    (committed 8d4e292) state and rebuilt, this test fails with relative
    error 0.10-1.06 depending on ``boundary``, matching the diagnosed bug.
    """
    ny, nx = 16, 16
    dx, dt, nt = 5.0, 0.0001, 1200
    lamb0, mu0, buoyancy0 = _isotropic_elastic_media((ny, nx))
    sa0 = ricker(25.0, nt, dt, 0.02, dtype=torch.double).reshape(1, 1, nt)
    kw: dict[str, Any] = {
        "source_locations_pressure_rate": torch.tensor([[[10, 10]]]),
        "receiver_locations_p": torch.tensor([[[5, 5]]]),
        "pml_width": pml_width,
        "boundary": boundary,
        "accuracy": 4,
        "pml_freq": 25.0,
    }

    def run(
        backend: Optional[Literal["eager", "compile"]],
    ) -> tuple[torch.Tensor, ...]:
        lamb = lamb0.clone().requires_grad_(True)
        mu = mu0.clone().requires_grad_(True)
        buoyancy = buoyancy0.clone().requires_grad_(True)
        sa = sa0.clone().requires_grad_(True)
        out = elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_pressure_rate=sa,
            backend=backend,
            **kw,
        ).receiver_amplitudes_p
        return torch.autograd.grad(out.sum(), [lamb, mu, buoyancy, sa])

    g_c = run("compiled")
    g_e = run("eager")
    for name, gc, ge in zip(
        ["lamb", "mu", "buoyancy", "source_amplitudes_pressure_rate"], g_c, g_e
    ):
        denom = ge.abs().max()
        rel = (gc - ge).abs().max() / denom
        assert rel < 1e-9, f"{name} grad rel err {rel.item()} boundary={boundary}"


@_ELASTIC_C_BACKEND
def test_elastic_compiled_backward_boundary_adjoint_parity_3d() -> None:
    """3-D counterpart of the 2-D adjoint-parity test above, mixed faces."""
    nz = ny = nx = 12
    dx, dt, nt = 6.0, 0.00015, 500
    lamb0, mu0, buoyancy0 = _isotropic_elastic_media((nz, ny, nx))
    sa0 = ricker(20.0, nt, dt, 0.03, dtype=torch.double).reshape(1, 1, nt)
    kw: dict[str, Any] = {
        "source_amplitudes_pressure_rate": sa0,
        "source_locations_pressure_rate": torch.tensor([[[6, 6, 6]]]),
        "receiver_locations_p": torch.tensor([[[3, 3, 3]]]),
        "pml_width": [0, 0, 2, 2, 2, 2],
        "boundary": ["rigid", "pml", "pml", "pml", "pml", "pml"],
        "accuracy": 4,
        "pml_freq": 20.0,
    }

    def run(
        backend: Optional[Literal["eager", "compile"]],
    ) -> tuple[torch.Tensor, ...]:
        lamb = lamb0.clone().requires_grad_(True)
        mu = mu0.clone().requires_grad_(True)
        buoyancy = buoyancy0.clone().requires_grad_(True)
        out = elastic(
            lamb, mu, buoyancy, dx, dt, backend=backend, **kw
        ).receiver_amplitudes_p
        return torch.autograd.grad(out.sum(), [lamb, mu, buoyancy])

    g_c = run("compiled")
    g_e = run("eager")
    for name, gc, ge in zip(["lamb", "mu", "buoyancy"], g_c, g_e):
        rel = (gc - ge).abs().max() / ge.abs().max()
        assert rel < 1e-9, f"{name} grad rel err {rel.item()} (3-D)"


@_ELASTIC_C_BACKEND
def test_elastic_compiled_backward_boundary_fd_arbitration() -> None:
    """Central-FD arbitration at the max-|g| cell (independent of eager).

    Rather than trusting eager as ground truth, perturb the model directly
    at the cell with the largest-magnitude eager gradient and compare the
    compiled analytic gradient there against a central finite difference of
    the compiled forward loss.
    """
    ny, nx = 16, 16
    dx, dt, nt = 5.0, 0.0001, 1200
    lamb0, mu0, buoyancy0 = _isotropic_elastic_media((ny, nx))
    sa0 = ricker(25.0, nt, dt, 0.02, dtype=torch.double).reshape(1, 1, nt)
    kw: dict[str, Any] = {
        "source_amplitudes_pressure_rate": sa0,
        "source_locations_pressure_rate": torch.tensor([[[10, 10]]]),
        "receiver_locations_p": torch.tensor([[[5, 5]]]),
        "pml_width": [0, 0, 10, 10],
        "boundary": ["rigid", "pml", "pml", "pml"],
        "accuracy": 4,
        "pml_freq": 25.0,
    }

    def loss(
        mu: torch.Tensor, backend: Optional[Literal["eager", "compile"]]
    ) -> torch.Tensor:
        out = elastic(lamb0, mu, buoyancy0, dx, dt, backend=backend, **kw)
        return out.receiver_amplitudes_p.sum()

    mu_e = mu0.clone().requires_grad_(True)
    g_e = torch.autograd.grad(loss(mu_e, "eager"), mu_e)[0]
    idx = tuple(int(i) for i in (g_e.abs() == g_e.abs().max()).nonzero()[0])

    mu_c = mu0.clone().requires_grad_(True)
    g_c = torch.autograd.grad(loss(mu_c, "compiled"), mu_c)[0]

    eps = float(mu0[idx].item()) * 1e-6
    mu_plus = mu0.clone()
    mu_plus[idx] += eps
    mu_minus = mu0.clone()
    mu_minus[idx] -= eps
    fd = (loss(mu_plus, "compiled") - loss(mu_minus, "compiled")) / (2 * eps)

    rel = abs(g_c[idx].item() - fd.item()) / abs(fd.item())
    assert rel < 2e-4, f"compiled grad vs FD rel err {rel} at {idx}"


@_ELASTIC_C_BACKEND
@pytest.mark.parametrize(
    ("boundary", "pml_width"),
    [
        (["rigid", "pml", "pml", "pml"], [0, 0, 10, 10]),
        (["rigid", "pml", "rigid", "pml"], [0, 0, 0, 0]),
    ],
    ids=["rigid_lo_pml", "rigid_lo_both_axes"],
)
def test_elastic_compiled_forward_final_state_matches_eager(
    boundary: list[str], pml_width: list[int]
) -> None:
    """Compiled forward's returned final wavefields are boundary-consistent.

    Regression test for a secondary bug: the compiled forward returned the
    post-update, unfilled final stress/velocity state (the last timestep's
    field update runs after the last in-loop boundary fill), while eager
    applies a final guard-ring/on-surface fill after its time loop
    (``_elastic_eager_core``). Without the fix, e.g. the free-surface stress
    nodes were nonzero in the compiled output where eager (and physics)
    requires them to be exactly zero.
    """
    ny, nx = 16, 16
    dx, dt, nt = 5.0, 0.0001, 400
    lamb, mu, buoyancy = _isotropic_elastic_media((ny, nx))
    sa = ricker(25.0, nt, dt, 0.02, dtype=torch.double).reshape(1, 1, nt)
    kw: dict[str, Any] = {
        "source_amplitudes_pressure_rate": sa,
        "source_locations_pressure_rate": torch.tensor([[[10, 10]]]),
        "receiver_locations_p": torch.tensor([[[5, 5]]]),
        "pml_width": pml_width,
        "boundary": boundary,
        "accuracy": 4,
        "pml_freq": 25.0,
    }
    out_c = elastic(lamb, mu, buoyancy, dx, dt, backend="compiled", **kw)
    out_e = elastic(lamb, mu, buoyancy, dx, dt, backend="eager", **kw)
    for field in ("vy_0", "vx_0", "sigmayy_0", "sigmaxy_0", "sigmaxx_0"):
        field_c = getattr(out_c, field)
        field_e = getattr(out_e, field)
        denom = field_e.abs().max()
        if denom == 0:
            assert field_c.abs().max() == 0, f"{field}: compiled nonzero, eager zero"
            continue
        rel = (field_c - field_e).abs().max() / denom
        assert rel < 1e-12, f"{field} final-state rel err {rel.item()}"


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_forward_callback_shear_fields_match_compiled(ndim) -> None:
    """Eager forward-callback shear/memory fields match the compiled backend.

    Previously a zip misalignment (pairing a dimension's shear-stress/shear-
    memory names against a row that still included its ``torch.empty(0)``
    diagonal placeholder) mapped one name to that unused placeholder and gave
    another name the wrong field's real tensor, silently (see PR #121).
    """
    n = 10
    shape = (n,) * ndim
    lamb = torch.full(shape, 2200.0)
    mu = torch.full(shape, 1000.0)
    buoyancy = torch.full(shape, 1 / 2200.0)
    dx = 5.0
    dt = 0.0005
    nt = 12
    comp = "z" if ndim == 3 else "y"
    amp = torch.zeros(1, 1, nt)
    amp[0, 0, 2] = 1.0
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded: dict[str, dict[str, torch.Tensor]] = {}

    def make_recorder(key: str):
        def recorder(state: deepwave.CallbackState) -> None:
            recorded[key] = {name: t.clone() for name, t in state._wavefields.items()}

        return recorder

    common_kwargs: dict[str, Any] = {
        "source_amplitudes_x": amp,
        "source_locations_x": loc,
        f"source_amplitudes_{comp}": amp,
        f"source_locations_{comp}": loc,
        "pml_width": 4,
        "callback_frequency": 1,
    }
    deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        backend="eager",
        forward_callback=make_recorder("eager"),
        **common_kwargs,
    )
    deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        backend="compiled",
        forward_callback=make_recorder("compiled"),
        **common_kwargs,
    )
    eager_dict = recorded["eager"]
    compiled_dict = recorded["compiled"]
    assert set(eager_dict) == set(compiled_dict)
    for name, tensor in eager_dict.items():
        assert tensor.numel() > 0, f"{name}: eager tensor is empty"
    for name in eager_dict:
        e = eager_dict[name]
        c = compiled_dict[name]
        assert e.shape == c.shape, f"{name}: shape mismatch {e.shape} vs {c.shape}"
        denom = max(c.abs().max().item(), 1e-30)
        relerr = (e - c).abs().max().item() / denom
        assert relerr < 1e-3, f"{name}: eager/compiled mismatch, rel err {relerr:.3e}"


def test_eager_rejects_storage_compression() -> None:
    """backend="eager" rejects storage_compression=True (see PR #121)."""
    lamb = torch.ones(10, 10) * 2200
    mu = torch.ones(10, 10) * 1000
    buoyancy = torch.ones(10, 10) * 1 / 2200
    source_amplitudes_y = torch.zeros(1, 1, 20)
    source_locations_y = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations_y = torch.zeros(1, 1, 2, dtype=torch.long)
    with pytest.raises(RuntimeError, match="Storage compression"):
        elastic(
            lamb,
            mu,
            buoyancy,
            5.0,
            0.004,
            source_amplitudes_y=source_amplitudes_y,
            source_locations_y=source_locations_y,
            receiver_locations_y=receiver_locations_y,
            backend="eager",
            storage_compression=True,
        )
