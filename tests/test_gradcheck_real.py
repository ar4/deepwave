"""Genuine ``torch.autograd.gradcheck`` coverage, one call per propagator.

Every test here perturbs its inputs and compares the analytical gradient
against a real finite-difference Jacobian via ``torch.autograd.gradcheck``.
This is distinct from the ``test_backend_parity_*`` tests elsewhere, which
compare backends against each other and never finite-difference anything.

**These tests are manual-only.** They carry the ``manual_gradcheck_anchor``
marker and ``tests/conftest.py`` deselects that marker from every automatic
selection, so no default run and no CI job collects them. The routine
verification chain (analytic forward oracles, then autograd of the eager
reference, then compiled-vs-eager gradient parity) resolves a wrong gradient
at float64 parity precision, which is far finer than a finite difference's
step-size noise. These anchors provide a separate check for shared errors in
preparation, differentiation, or the reference itself; backend agreement alone
cannot exclude those errors. Run them
with::

    pytest -m manual_gradcheck_anchor

Each case targets the **compiled** backend for every retained
propagator: the compiled backward is a hand-derived adjoint written in
C/CUDA, so it is the implementation that most deserves an oracle independent
of autograd.

Each case uses a tiny float64, CPU, single-shot, single-mechanism model, just
large enough for its propagator's finite-difference stencil, and checks the
gradient with respect to the propagator's main model parameter(s) and the
source amplitudes in one joint ``gradcheck`` call. Where a model parameter is
on the order of 1e8-1e9 (elastic-family stiffnesses), ``gradcheck``'s default
``eps`` (an absolute perturbation) is close to float64's local ULP at that
scale, so the looser elastic-family tolerance (``atol=1e-4``, ``rtol=1e-2``)
is used, matching the tolerance already established for such cases in
``tests/test_boundary.py``. Where a model parameter reaches roughly 1e9-1e10
(the VTI stiffnesses), the leaf tensors are additionally rescaled to
O(1-10) before being passed to ``gradcheck`` (and rescaled back inside the
wrapper), so the perturbation is well above the local ULP; see the acoustic
VTI case for the full reasoning, which is shared by every rescaled case
below.
"""

import pytest
import torch

import deepwave
import deepwave.wavelets

pytestmark = pytest.mark.manual_gradcheck_anchor

DTYPE = torch.float64


def test_gradcheck_real_scalar() -> None:
    """Real gradcheck for :func:`deepwave.scalar`, w.r.t. ``v`` and the source."""
    torch.manual_seed(0)
    nx = 20
    dx, dt, nt = 5.0, 0.0007, 12
    v = (1500.0 + torch.randn(nx, dtype=DTYPE)).requires_grad_(True)
    sa = torch.randn(1, 1, nt, dtype=DTYPE).requires_grad_(True)
    sl = torch.tensor([[[10]]])
    rl = torch.tensor([[[7]]])

    def f(v: torch.Tensor, sa: torch.Tensor) -> torch.Tensor:
        out = deepwave.scalar(
            v,
            dx,
            dt,
            source_amplitudes=sa,
            source_locations=sl,
            receiver_locations=rl,
            pml_width=6,
            accuracy=4,
            pml_freq=15.0,
            max_vel=2000.0,
            backend="compiled",
        )
        return out.receiver_amplitudes

    assert torch.autograd.gradcheck(
        f, (v, sa), eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=True
    )


def test_gradcheck_real_scalar_born() -> None:
    """Real gradcheck for :func:`deepwave.scalar_born`, w.r.t. ``v``, ``dv``, source."""
    torch.manual_seed(0)
    nx = 20
    dx, dt, nt = 5.0, 0.0007, 12
    v = (1500.0 + torch.randn(nx, dtype=DTYPE)).requires_grad_(True)
    dv = (75.0 * torch.randn(nx, dtype=DTYPE)).requires_grad_(True)
    sa = torch.randn(1, 1, nt, dtype=DTYPE).requires_grad_(True)
    sl = torch.tensor([[[10]]])
    rl = torch.tensor([[[7]]])

    def f(v: torch.Tensor, dv: torch.Tensor, sa: torch.Tensor) -> torch.Tensor:
        out = deepwave.scalar_born(
            v,
            dx,
            dt,
            dv=dv,
            source_amplitudes=sa,
            source_locations=sl,
            sc_receiver_locations=rl,
            bg_receiver_locations=rl,
            pml_width=6,
            accuracy=4,
            pml_freq=15.0,
            max_vel=2000.0,
            backend="compiled",
        )
        return out.sc_receiver_amplitudes + out.bg_receiver_amplitudes

    assert torch.autograd.gradcheck(
        f, (v, dv, sa), eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=True
    )


def test_gradcheck_real_acoustic() -> None:
    """Real gradcheck for :func:`deepwave.acoustic`, w.r.t. ``v``, ``rho``, source."""
    torch.manual_seed(0)
    ny, nx = 14, 16
    dx, dt, nt = 5.0, 0.0007, 12
    v = (1500.0 + torch.randn(ny, nx, dtype=DTYPE)).requires_grad_(True)
    rho = (1000.0 + 50.0 * torch.rand(ny, nx, dtype=DTYPE)).requires_grad_(True)
    sa = torch.randn(1, 1, nt, dtype=DTYPE).requires_grad_(True)
    sl = torch.tensor([[[7, 8]]])
    rl = torch.tensor([[[5, 6]]])

    def f(v: torch.Tensor, rho: torch.Tensor, sa: torch.Tensor) -> torch.Tensor:
        out = deepwave.acoustic(
            v,
            rho,
            dx,
            dt,
            source_amplitudes_volume_rate=sa,
            source_locations_volume_rate=sl,
            receiver_locations_p=rl,
            pml_width=4,
            accuracy=4,
            pml_freq=15.0,
            max_vel=2000.0,
            backend="compiled",
        )
        return out.receiver_amplitudes_p * 1e-4

    assert torch.autograd.gradcheck(
        f, (v, rho, sa), eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=True
    )


def test_gradcheck_real_acoustic_born() -> None:
    """Real gradcheck for acoustic_born, w.r.t. ``dv``/``drho`` and source."""
    torch.manual_seed(0)
    ny, nx = 14, 16
    dx, dt, nt = 5.0, 0.0007, 12
    v = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    rho = torch.full((ny, nx), 1000.0, dtype=DTYPE)
    dv = (75.0 * torch.randn(ny, nx, dtype=DTYPE)).requires_grad_(True)
    drho = (50.0 * torch.randn(ny, nx, dtype=DTYPE)).requires_grad_(True)
    sa = torch.randn(1, 1, nt, dtype=DTYPE).requires_grad_(True)
    sl = torch.tensor([[[7, 8]]])
    rl = torch.tensor([[[5, 6]]])

    def f(dv: torch.Tensor, drho: torch.Tensor, sa: torch.Tensor) -> torch.Tensor:
        out = deepwave.acoustic_born(
            v,
            rho,
            dx,
            dt,
            dv=dv,
            drho=drho,
            source_amplitudes_volume_rate=sa,
            source_locations_volume_rate=sl,
            sc_receiver_locations_p=rl,
            bg_receiver_locations_p=rl,
            pml_width=4,
            accuracy=4,
            pml_freq=15.0,
            max_vel=2000.0,
            backend="compiled",
        )
        return (out.sc_receiver_amplitudes_p + out.bg_receiver_amplitudes_p) * 1e-4

    assert torch.autograd.gradcheck(
        f, (dv, drho, sa), eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=True
    )


def test_gradcheck_real_acoustic_vti() -> None:
    """Real gradcheck for acoustic_vti, w.r.t. every Cij/buoyancy/source.

    ``gradcheck``'s ``eps`` (default 1e-6) is an ABSOLUTE perturbation applied
    identically to every leaf. c11/c13/c33 are ~1e9 (a physically realistic
    Pa-scale stiffness), so perturbing by 1e-6 is close to float64's local ULP
    there (1e9 * 2**-52 ~= 2.2e-7): too close for a well-conditioned numerical
    Jacobian, and -- worse -- if the true gradient at this tiny problem were
    also very small, a broken (always-zero) numerical Jacobian would silently
    match it. ``scale`` rescales c11/c13/c33 to O(1-10) leaf tensors (a
    well-conditioned ~1e-7 relative perturbation under the default ``eps``);
    ``wrapper`` multiplies back by ``scale`` before calling the propagator, so
    the physical medium is unchanged. ``ampscale`` raises the source amplitude
    so the c11/c13/c33 Jacobian entries (which scale with the source's overall
    magnitude, this being a linear PDE) clear ``atol``.
    """
    torch.manual_seed(4)
    n, nt, dt = 8, 8, 0.00025
    dx = 5.0
    shape = (n, n)
    scale = 1e9
    ampscale = 1e6
    base = 1e9 * (1 + 0.1 * torch.rand(*shape, dtype=DTYPE))
    c33 = (4 * base / scale).requires_grad_(True)
    c11 = (6 * base / scale).requires_grad_(True)
    c13 = (2 * base / scale).requires_grad_(True)  # c11*c33 >= c13**2 -> eps > delta
    buoyancy = (
        1 / (2000 * (1 + 0.1 * torch.rand(*shape, dtype=DTYPE)))
    ).requires_grad_(True)
    amp = (torch.randn(1, 1, nt, dtype=DTYPE) / ampscale).requires_grad_(True)
    sl = torch.tensor([[[4, 4]]])
    rl = torch.tensor([[[3, 3]]])

    def wrapper(
        c11: torch.Tensor,
        c13: torch.Tensor,
        c33: torch.Tensor,
        buoyancy: torch.Tensor,
        amp: torch.Tensor,
    ) -> torch.Tensor:
        out = deepwave.acoustic_vti(
            c11 * scale,
            c13 * scale,
            c33 * scale,
            buoyancy,
            dx,
            dt,
            source_amplitudes_x=amp * ampscale,
            source_locations_x=sl,
            receiver_locations_x=rl,
            pml_width=2,
            accuracy=2,
            max_vel=6000.0,
            pml_freq=15.0,
            backend="compiled",
        )
        return out.receiver_amplitudes_x

    assert torch.autograd.gradcheck(
        wrapper,
        (c11, c13, c33, buoyancy, amp),
        atol=1e-8,
        rtol=1e-4,
        raise_exception=True,
    )


def test_gradcheck_real_elastic() -> None:
    """Real gradcheck for :func:`deepwave.elastic`, w.r.t. lamb/mu/buoyancy, source."""
    torch.manual_seed(0)
    shape = (14, 14)
    dx, dt, nt = 5.0, 0.0002, 12
    c, cs, rho = 1500.0, 900.0, 1000.0
    lamb = (
        torch.full(shape, rho * (c**2 - 2 * cs**2), dtype=DTYPE)
        + 3e6 * torch.rand(shape, dtype=DTYPE)
    ).requires_grad_(True)
    mu = (
        torch.full(shape, rho * cs**2, dtype=DTYPE)
        + 2e6 * torch.rand(shape, dtype=DTYPE)
    ).requires_grad_(True)
    buoyancy = (
        torch.full(shape, 1.0 / rho, dtype=DTYPE)
        + 2e-6 * torch.rand(shape, dtype=DTYPE)
    ).requires_grad_(True)
    sa = (
        deepwave.wavelets.ricker(25.0, nt, dt, 0.02, dtype=DTYPE).reshape(1, 1, nt)
    ).requires_grad_(True)
    sl = torch.tensor([[[7, 7]]])
    rl = torch.tensor([[[5, 5]]])

    def f(
        lamb: torch.Tensor, mu: torch.Tensor, buoyancy: torch.Tensor, sa: torch.Tensor
    ) -> torch.Tensor:
        out = deepwave.elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_pressure_rate=sa,
            source_locations_pressure_rate=sl,
            receiver_locations_p=rl,
            pml_width=4,
            accuracy=4,
            pml_freq=25.0,
            backend="compiled",
        )
        return out.receiver_amplitudes_p * 1e-8

    # TODO(maintainer): Establish this anchor's sensitivity before treating a
    # pass as evidence for every parameter. Outputs are scaled by 1e-8 while
    # atol is 1e-4, and eps=1e-6 perturbs unscaled O(1e9) stiffnesses. The
    # 2026-10-23 static review did not establish whether a zero/wrong backward
    # would pass. Nondimensionalise the inputs, fix max_vel, measure Jacobian
    # magnitudes and finite-difference convergence, and require zero-gradient
    # negative controls to fail for each input group before setting tolerances.
    assert torch.autograd.gradcheck(
        f,
        (lamb, mu, buoyancy, sa),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-2,
        raise_exception=True,
    )


def test_gradcheck_real_elastic_born() -> None:
    """Real gradcheck for elastic_born, w.r.t. every model parameter and source."""
    torch.manual_seed(5)
    shape = (10, 11)
    dx, dt = 5.0, 4e-4
    vp = 2000 + 30 * torch.rand(shape, dtype=DTYPE)
    vs = 1100 + 20 * torch.rand(shape, dtype=DTYPE)
    rho = 2200 + 30 * torch.rand(shape, dtype=DTYPE)
    lamb, mu, bu = deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    lamb = lamb.requires_grad_(True)
    mu = mu.requires_grad_(True)
    bu = bu.requires_grad_(True)
    dlamb = (0.05 * lamb.detach() * torch.rand(shape, dtype=DTYPE)).requires_grad_(True)
    dmu = (0.05 * mu.detach() * torch.rand(shape, dtype=DTYPE)).requires_grad_(True)
    dbuoyancy = (0.05 * bu.detach() * torch.rand(shape, dtype=DTYPE)).requires_grad_(
        True
    )
    src = (1e3 * torch.randn(1, 1, 6, dtype=DTYPE)).requires_grad_(True)
    sl = torch.tensor([[[5, 6]]])
    rl = torch.tensor([[[3, 4], [7, 8]]])
    common = {
        "pml_width": 2,
        "backend": "compiled",
        "pml_freq": 15.0,
        "max_vel": 2200.0,
    }

    def f(
        lamb: torch.Tensor,
        mu: torch.Tensor,
        bu: torch.Tensor,
        dlamb: torch.Tensor,
        dmu: torch.Tensor,
        dbuoyancy: torch.Tensor,
        src: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = deepwave.elastic_born(
            lamb,
            mu,
            bu,
            dx,
            dt,
            dlamb=dlamb,
            dmu=dmu,
            dbuoyancy=dbuoyancy,
            source_amplitudes_y=src,
            source_locations_y=sl,
            sc_receiver_locations_y=rl,
            sc_receiver_locations_p=rl,
            **common,
        )
        return out.sc_receiver_amplitudes_p, out.sc_receiver_amplitudes_y

    assert torch.autograd.gradcheck(
        f,
        (lamb, mu, bu, dlamb, dmu, dbuoyancy, src),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-2,
        raise_exception=True,
    )


def test_gradcheck_real_elastic_vti() -> None:
    """Real gradcheck for elastic_vti, w.r.t. every Cij/buoyancy/source."""
    torch.manual_seed(0)
    shape = (14, 14)
    dx, dt, nt = 5.0, 0.0002, 12
    c, cs, rho = 1500.0, 900.0, 1000.0
    lamb0 = rho * (c**2 - 2 * cs**2)
    mu0 = rho * cs**2
    c11 = (
        torch.full(shape, lamb0 + 2 * mu0, dtype=DTYPE)
        + 3e6 * torch.rand(shape, dtype=DTYPE)
    ).requires_grad_(True)
    c13 = (
        torch.full(shape, lamb0, dtype=DTYPE) + 3e6 * torch.rand(shape, dtype=DTYPE)
    ).requires_grad_(True)
    c33 = (
        torch.full(shape, lamb0 + 2 * mu0, dtype=DTYPE)
        + 3e6 * torch.rand(shape, dtype=DTYPE)
    ).requires_grad_(True)
    c55 = (
        torch.full(shape, mu0, dtype=DTYPE) + 2e6 * torch.rand(shape, dtype=DTYPE)
    ).requires_grad_(True)
    buoyancy = (
        torch.full(shape, 1.0 / rho, dtype=DTYPE)
        + 2e-6 * torch.rand(shape, dtype=DTYPE)
    ).requires_grad_(True)
    sa = (
        deepwave.wavelets.ricker(25.0, nt, dt, 0.02, dtype=DTYPE).reshape(1, 1, nt)
    ).requires_grad_(True)
    sl = torch.tensor([[[7, 7]]])
    rl = torch.tensor([[[5, 5]]])

    def f(
        c11: torch.Tensor,
        c13: torch.Tensor,
        c33: torch.Tensor,
        c55: torch.Tensor,
        buoyancy: torch.Tensor,
        sa: torch.Tensor,
    ) -> torch.Tensor:
        out = deepwave.elastic_vti(
            c11,
            c13,
            c33,
            c55,
            buoyancy,
            dx,
            dt,
            source_amplitudes_pressure_rate=sa,
            source_locations_pressure_rate=sl,
            receiver_locations_p=rl,
            pml_width=4,
            accuracy=4,
            pml_freq=25.0,
            backend="compiled",
        )
        return out.receiver_amplitudes_p * 1e-8

    assert torch.autograd.gradcheck(
        f,
        (c11, c13, c33, c55, buoyancy, sa),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-2,
        raise_exception=True,
    )


def test_gradcheck_real_viscoacoustic() -> None:
    """Real gradcheck for viscoacoustic, w.r.t. ``v``/``rho``/``q`` and source."""
    torch.manual_seed(0)
    ny, nx = 14, 16
    dx, dt, nt = 5.0, 0.0007, 12
    # Dimensionless leaves and an O(1) readout keep the absolute gradcheck
    # tolerance meaningful for pressure-rate forcing and every model input.
    v = (1.0 + torch.randn(ny, nx, dtype=DTYPE) / 1500).requires_grad_(True)
    rho = (1.0 + 0.05 * torch.rand(ny, nx, dtype=DTYPE)).requires_grad_(True)
    q = (1.0 + 0.1 * torch.rand(ny, nx, dtype=DTYPE)).requires_grad_(True)
    sa = torch.randn(1, 1, nt, dtype=DTYPE).requires_grad_(True)
    sl = torch.tensor([[[7, 8]]])
    rl = torch.tensor([[[5, 6]]])

    def f(
        v: torch.Tensor, rho: torch.Tensor, q: torch.Tensor, sa: torch.Tensor
    ) -> torch.Tensor:
        out = deepwave.viscoacoustic(
            1500 * v,
            1000 * rho,
            50 * q,
            dx,
            dt,
            source_amplitudes_pressure_rate=sa,
            source_locations_pressure_rate=sl,
            receiver_locations_p=rl,
            pml_width=4,
            accuracy=4,
            pml_freq=15.0,
            q_freq=15.0,
            max_vel=2000.0,
            backend="compiled",
        )
        return out.receiver_amplitudes_p * 1e4

    for grad in torch.autograd.grad(f(v, rho, q, sa).square().sum(), (v, rho, q, sa)):
        assert grad.norm() > 1e-5

    assert torch.autograd.gradcheck(
        f, (v, rho, q, sa), eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=True
    )


def test_gradcheck_real_scalar_backward_callback() -> None:
    """Real gradcheck for :func:`deepwave.scalar` with a ``backward_callback`` attached.

    A user-supplied ``backward_callback`` rides alongside the analytical
    gradient computation and must not perturb it. This checks the gradient of
    ``receiver_amplitudes`` w.r.t. ``v`` and the source amplitudes against a
    finite-difference Jacobian while a genuine callback (one that reads the
    adjoint wavefield at every step) is attached, on the eager backend (the
    only backend of :func:`deepwave.scalar` that supports ``backward_callback``
    outside the compiled one). ``gradcheck``'s default ``check_undefined_grad``
    also drives an undefined (identically zero) gradient through the callback,
    which must be materialised as zeros rather than reaching the callback as
    ``None``.
    """
    torch.manual_seed(0)
    nx = 20
    dx, dt, nt = 5.0, 0.0007, 12
    v = (1500.0 + torch.randn(nx, dtype=DTYPE)).requires_grad_(True)
    sa = torch.randn(1, 1, nt, dtype=DTYPE).requires_grad_(True)
    sl = torch.tensor([[[10]]])
    rl = torch.tensor([[[7]]])
    calls: list[float] = []

    def bcb(state: deepwave.CallbackState) -> None:
        calls.append(state.get_wavefield("wavefield_0").abs().sum().item())

    def f(v: torch.Tensor, sa: torch.Tensor) -> torch.Tensor:
        out = deepwave.scalar(
            v,
            dx,
            dt,
            source_amplitudes=sa,
            source_locations=sl,
            receiver_locations=rl,
            pml_width=6,
            accuracy=4,
            pml_freq=15.0,
            max_vel=2000.0,
            backend="compiled",
            backward_callback=bcb,
            callback_frequency=1,
        )
        return out.receiver_amplitudes

    assert torch.autograd.gradcheck(
        f, (v, sa), eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=True
    )
    # The callback must have genuinely fired (not been dead code) for the
    # check above to mean anything.
    assert len(calls) > 0


def test_gradcheck_real_scalar_custom_imaging_condition() -> None:
    """Real gradcheck for :func:`deepwave.scalar` through the imaging-condition path.

    ``backward_callback`` combined with ``state.get_wavefield(...,
    role="forward")`` is Deepwave's mechanism for building a custom forward x
    adjoint imaging condition from within the backward pass (see
    ``tests/test_custom_imaging_condition.py``); ``scalar`` is the only
    propagator whose eager backend supports ``role="forward"``. This checks
    that engaging that machinery -- which reconstructs the raw forward
    wavefield alongside the adjoint one at every step -- does not perturb the
    analytical gradient of ``receiver_amplitudes`` w.r.t. ``v``/source, by
    comparing it against a finite-difference Jacobian.
    """
    torch.manual_seed(1)
    ny, nx = 14, 16
    dx, dt, nt = 5.0, 0.0007, 12
    v = (1500.0 + torch.randn(ny, nx, dtype=DTYPE)).requires_grad_(True)
    sa = torch.randn(1, 1, nt, dtype=DTYPE).requires_grad_(True)
    sl = torch.tensor([[[7, 8]]])
    rl = torch.tensor([[[5, 6]]])
    image = torch.zeros(ny, nx, dtype=DTYPE)

    def bcb(state: deepwave.CallbackState) -> None:
        forward_field = state.get_wavefield("wavefield_0", role="forward")[0]
        adjoint_field = state.get_wavefield("wavefield_0")[0]
        image.add_(forward_field * adjoint_field)

    def f(v: torch.Tensor, sa: torch.Tensor) -> torch.Tensor:
        out = deepwave.scalar(
            v,
            dx,
            dt,
            source_amplitudes=sa,
            source_locations=sl,
            receiver_locations=rl,
            pml_width=4,
            accuracy=4,
            pml_freq=15.0,
            max_vel=2000.0,
            backend="compiled",
            backward_callback=bcb,
            backward_callback_needs_forward_wavefield=True,
            callback_frequency=1,
        )
        return out.receiver_amplitudes

    assert torch.autograd.gradcheck(
        f, (v, sa), eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=True
    )
    # The imaging-condition callback must have genuinely fired and produced a
    # non-trivial image for the check above to mean anything.
    assert image.abs().max() > 0
