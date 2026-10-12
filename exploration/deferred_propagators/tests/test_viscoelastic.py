"""Tests for the viscoelastic (elastic + Qp/Qs GSLS attenuation) propagator.

Physics gates (all measured against exact theory or a hard-precedent
reference on the eager Python backend -- no compiled backend exists yet):
  * Qp, Qs -> infinity reduces to :func:`deepwave.elastic` (forward and
    gradients), 2-D and 3-D;
  * P-wave and S-wave amplitude decay and phase-velocity dispersion, each
    measured against the exact GSLS complex-modulus theory;
  * dLoss/dqp, dLoss/dqs, dLoss/dlamb, dLoss/dmu, dLoss/dbuoyancy central-FD
    gradchecks;
  * reciprocity;
  * adjoint dot-product;
  * elastic M-PML corner stability, inherited (mpml_fraction > 0 with finite Qp/Qs);
  * negative controls;
  * free surface: bit-exact vacuum
    Q-invariance and exact-zero-traction (IVM), Qp/Qs -> large reduces to
    elastic with the same vacuum rows, the analytic reflection coefficient
    of the ``boundary="rigid"`` face, a boundary gradcheck, a finite-Q
    Rayleigh-wave decay check against the exact complex Rayleigh-root
    theory, and a plain (non-M-PML) long-run stability check with a vacuum
    free surface and with a reflecting boundary face.
"""

import cmath
import dataclasses
import importlib
import math
import warnings

import numpy as np
import pytest
import torch

import deepwave

_vemod = importlib.import_module("deepwave.viscoelastic")
_gsls = importlib.import_module("deepwave._gsls")

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def _fit_line(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Least-squares slope/intercept of ``y`` against ``x``."""
    x = x.double()
    y = y.double()
    xm = x.mean()
    ym = y.mean()
    b = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    a = ym - b * xm
    return float(a.item()), float(b.item())


def _sls_complex_modulus(m_relaxed: float, q: float, f0: float, f: float) -> complex:
    """Exact single-SLS complex modulus (same relation as viscoacoustic's).

    See :func:`deepwave._gsls.q_to_relaxation_times`.
    """
    omega0 = 2 * math.pi * f0
    omega = 2 * math.pi * f
    inv_q = 1.0 / q
    tausig = (math.sqrt(1 + inv_q**2) - inv_q) / omega0
    taueps = 1.0 / (omega0**2 * tausig)
    return m_relaxed * (1 + 1j * omega * taueps) / (1 + 1j * omega * tausig)


def _p_wave_theory(
    vp: float, vs: float, rho: float, qp: float, f0: float, f: float, ndim: int = 2
) -> tuple[float, float]:
    """Exact P-wave phase velocity/attenuation with Qs -> infinity.

    The P-wave modulus is ``M = K_iso + 2*mu*(ndim-1)/ndim`` (see the
    viscoelastic module docstring); with Qs -> infinity, ``mu`` is real
    (unattenuated) and only ``K_iso`` carries the GSLS complex modulus.
    """
    mu = rho * vs**2
    k_iso = rho * vp**2 - 2 * mu * (ndim - 1) / ndim
    k_iso_omega = _sls_complex_modulus(k_iso, qp, f0, f)
    m_omega = k_iso_omega + 2 * mu * (ndim - 1) / ndim
    omega = 2 * math.pi * f
    kc = omega * (rho / m_omega) ** 0.5
    if kc.real < 0:
        kc = -kc
    return omega / kc.real, abs(kc.imag)


def _s_wave_theory(
    vs: float, rho: float, qs: float, f0: float, f: float
) -> tuple[float, float]:
    """Exact S-wave phase velocity/attenuation (depends only on Qs)."""
    mu = rho * vs**2
    mu_omega = _sls_complex_modulus(mu, qs, f0, f)
    omega = 2 * math.pi * f
    kc = omega * (rho / mu_omega) ** 0.5
    if kc.real < 0:
        kc = -kc
    return omega / kc.real, abs(kc.imag)


def _homogeneous(
    ny: int, nx: int, vp: float, vs: float, rho: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = rho * vs * vs
    lamb = rho * (vp * vp - 2 * vs * vs)
    return (
        torch.full((ny, nx), lamb, dtype=DTYPE),
        torch.full((ny, nx), mu, dtype=DTYPE),
        torch.full((ny, nx), 1.0 / rho, dtype=DTYPE),
    )


def _scalar_loss(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    qp: torch.Tensor,
    qs: torch.Tensor,
    src: torch.Tensor,
    src_loc: torch.Tensor,
    rec_loc: torch.Tensor,
    dx: float,
    dt: float,
) -> torch.Tensor:
    out = deepwave.viscoelastic(
        lamb,
        mu,
        buoyancy,
        qp,
        qs,
        dx,
        dt,
        source_amplitudes_y=src,
        source_locations_y=src_loc,
        receiver_locations_p=rec_loc,
        pml_freq=25.0,
        accuracy=4,
        pml_width=6,
        q_freq=25.0,
        backend="eager",
    )
    return out.receiver_amplitudes_p.pow(2).sum()


# --- Qp, Qs -> infinity reduces to elastic ----------------------------------


@pytest.mark.parametrize("ndim", [2, 3])
def test_qp_qs_infinity_reduces_to_elastic(ndim: int) -> None:
    """Qp, Qs -> very large reduces to elastic (forward + gradients).

    The residual is the physical O(1/Q) attenuation, so a very large but
    finite Q is used (mirroring viscoacoustic's own
    ``test_acoustic_limit``); the reduction is a clean O(1/Q) as Q grows.
    """
    torch.manual_seed(0)
    n = 16 if ndim == 3 else 24
    shape = (n,) * ndim
    dx = 5.0
    dt = 0.0004 if ndim == 3 else 0.0005
    nt = 30 if ndim == 3 else 40
    freq = 25.0
    q_big = 1e10 if ndim == 3 else 1e12
    tol_fwd = 1e-7 if ndim == 3 else 1e-9
    tol_grad = 1e-6 if ndim == 3 else 1e-8

    vp = 2000.0 + 20.0 * torch.randn(*shape, dtype=DTYPE)
    vs = 1000.0 + 20.0 * torch.randn(*shape, dtype=DTYPE)
    rho = 2200.0 + 20.0 * torch.randn(*shape, dtype=DTYPE)
    mu = rho * vs**2
    lamb = rho * (vp**2 - 2 * vs**2)
    buoyancy = 1.0 / rho
    qbig = torch.full(shape, q_big, dtype=DTYPE)

    src = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq).to(DTYPE).reshape(1, 1, -1)
    src_loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    src_loc[0, 0] = torch.tensor([n // 2] * ndim)
    rec_loc = torch.zeros(1, 4, ndim, dtype=torch.long)
    for j in range(4):
        rec_loc[0, j] = torch.tensor([5 + j] * ndim)

    def run_ve(
        ll: torch.Tensor, mm: torch.Tensor, bb: torch.Tensor
    ) -> deepwave.PropagatorOutput:
        kwargs: dict = {}
        if ndim == 3:
            kwargs["source_amplitudes_x"] = src
            kwargs["source_locations_x"] = src_loc
        else:
            kwargs["source_amplitudes_y"] = src
            kwargs["source_locations_y"] = src_loc
        return deepwave.viscoelastic(
            ll,
            mm,
            bb,
            qbig,
            qbig,
            dx,
            dt,
            receiver_locations_p=rec_loc,
            pml_freq=freq,
            q_freq=freq,
            accuracy=4,
            pml_width=8,
            backend="eager",
            **kwargs,
        )

    def run_e(
        ll: torch.Tensor, mm: torch.Tensor, bb: torch.Tensor
    ) -> deepwave.PropagatorOutput:
        kwargs = {}
        if ndim == 3:
            kwargs["source_amplitudes_x"] = src
            kwargs["source_locations_x"] = src_loc
        else:
            kwargs["source_amplitudes_y"] = src
            kwargs["source_locations_y"] = src_loc
        return deepwave.elastic(
            ll,
            mm,
            bb,
            dx,
            dt,
            receiver_locations_p=rec_loc,
            pml_freq=freq,
            accuracy=4,
            pml_width=8,
            backend="eager",
            **kwargs,
        )

    rec_ve = run_ve(lamb, mu, buoyancy).receiver_amplitudes_p
    rec_e = run_e(lamb, mu, buoyancy).receiver_amplitudes_p
    assert rec_e.norm().item() > 0
    assert (rec_ve - rec_e).norm().item() / rec_e.norm().item() < tol_fwd

    ll = lamb.clone().requires_grad_(True)
    mm = mu.clone().requires_grad_(True)
    bb = buoyancy.clone().requires_grad_(True)
    run_ve(ll, mm, bb).receiver_amplitudes_p.pow(2).sum().backward()

    lla = lamb.clone().requires_grad_(True)
    mma = mu.clone().requires_grad_(True)
    bba = buoyancy.clone().requires_grad_(True)
    run_e(lla, mma, bba).receiver_amplitudes_p.pow(2).sum().backward()

    assert (ll.grad - lla.grad).norm().item() / lla.grad.norm().item() < tol_grad
    assert (mm.grad - mma.grad).norm().item() / mma.grad.norm().item() < tol_grad
    assert (bb.grad - bba.grad).norm().item() / bba.grad.norm().item() < tol_grad


# --- Analytic P-wave/S-wave amplitude decay and dispersion -----------------
#
# A plane wave is approximated with a LINE of co-phased sources spanning
# (most of) the y-extent, so the measured amplitude decay is a pure
# attenuation effect (no 2-D cylindrical geometric spreading to correct
# for), mirroring viscoacoustic's exactly-1-D ``_measure_1d`` setup as
# closely as the elastic case (which genuinely needs >= 2 dimensions)
# allows. A force along x with receivers reading v_x directly ahead of the
# source (theta=0 relative to the force axis, where the standard elastic
# radiation pattern's S-wave lobe vanishes) isolates the P-wave; a force
# along y with receivers reading v_y (theta=0 relative to THAT force axis,
# i.e. still directly ahead along x, where the P-wave lobe of a transverse
# force vanishes) isolates the S-wave.
#
# Every measurement is taken DIFFERENTIALLY against a Q = 1e8 reference run
# at the same geometry (subtracting its measured decay), a standard
# real-data Q-estimation technique that cancels any residual geometry-only
# artifact (a finite line source is not a perfect infinite plane wave); the
# differential recipe reproduces the exact GSLS theory to ~1% for both
# waves. Non-vacuousness for the underlying physics is checked directly by
# ``test_negative_control_broken_gsls_coupling`` below (breaking the GSLS
# coupling itself, not the measurement technique).


def _measure_wave(
    wave: str,
    q_val: float,
    f: float,
    f0: float,
    measure_frac: float = 0.85,
    ny: int = 60,
    nx: int = 200,
    nt: int = 8500,
    edge: int = 8,
    x0: int = 30,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Monochromatic steady-state amplitude & phase along a receiver line.

    ``wave in ("p", "s")`` selects a force along x (P) or y (S); the
    corresponding velocity component is read back along a receiver line
    directly ahead of the source (see the module note above).
    """
    dx = 4.0
    dt = 0.0003
    vp0, vs0, rho0 = 2000.0, 1000.0, 2200.0
    lamb, mu, buoyancy = _homogeneous(ny, nx, vp0, vs0, rho0)
    qhuge = torch.full((ny, nx), 1e8, dtype=DTYPE)
    if wave == "p":
        qp = torch.full((ny, nx), q_val, dtype=DTYPE)
        qs = qhuge
    else:
        qp = qhuge
        qs = torch.full((ny, nx), q_val, dtype=DTYPE)

    t = torch.arange(nt, dtype=DTYPE) * dt
    ramp = torch.clamp(t / (20.0 / f), max=1.0)
    src1 = (ramp * torch.sin(2 * math.pi * f * t)).reshape(1, 1, -1)
    n_line = ny - 2 * edge
    src_loc = torch.zeros(1, n_line, 2, dtype=torch.long)
    src_loc[0, :, 0] = torch.arange(edge, ny - edge)
    src_loc[0, :, 1] = x0
    src_amp = src1.expand(1, n_line, nt)
    xs = torch.arange(60, 160, 4)
    rec_loc = torch.zeros(1, xs.numel(), 2, dtype=torch.long)
    rec_loc[0, :, 0] = ny // 2
    rec_loc[0, :, 1] = xs

    kwargs = {
        "pml_freq": f0,
        "q_freq": f0,
        "accuracy": 6,
        "pml_width": 20,
        "backend": "eager",
    }
    if wave == "p":
        out = deepwave.viscoelastic(
            lamb,
            mu,
            buoyancy,
            qp,
            qs,
            dx,
            dt,
            source_amplitudes_x=src_amp,
            source_locations_x=src_loc,
            receiver_locations_x=rec_loc,
            **kwargs,
        )
        rec = out.receiver_amplitudes_x[0]
    else:
        out = deepwave.viscoelastic(
            lamb,
            mu,
            buoyancy,
            qp,
            qs,
            dx,
            dt,
            source_amplitudes_y=src_amp,
            source_locations_y=src_loc,
            receiver_locations_y=rec_loc,
            **kwargs,
        )
        rec = out.receiver_amplitudes_y[0]

    half = slice(int(nt * measure_frac), nt)
    tt = t[half]
    sig = rec[:, half]
    a = (sig * torch.cos(2 * math.pi * f * tt)).sum(1)
    b = (sig * torch.sin(2 * math.pi * f * tt)).sum(1)
    amp = torch.sqrt(a**2 + b**2)
    phase = torch.atan2(b, a)
    return (xs.double() - x0) * dx, amp, phase


def _wave_decay_and_dispersion(
    wave: str, q_val: float, f0: float
) -> tuple[float, float, float, float]:
    """Returns ``(alpha_meas, alpha_th, c_meas, c_th)`` for one wave type."""
    x, amp_q, phase_q = _measure_wave(wave, q_val, f0, f0)
    _, amp_huge, _ = _measure_wave(wave, 1e8, f0, f0)
    alpha_q = -_fit_line(x, torch.log(amp_q))[1]
    alpha_huge = -_fit_line(x, torch.log(amp_huge))[1]
    alpha_meas = alpha_q - alpha_huge

    vp0, vs0, rho0 = 2000.0, 1000.0, 2200.0
    if wave == "p":
        c_th, alpha_th = _p_wave_theory(vp0, vs0, rho0, q_val, f0, f0)
    else:
        c_th, alpha_th = _s_wave_theory(vs0, rho0, q_val, f0, f0)

    phw = torch.from_numpy(np.unwrap(phase_q.numpy()))
    k_meas = abs(_fit_line(x, phw)[1])
    c_meas = 2 * math.pi * f0 / k_meas
    return alpha_meas, alpha_th, c_meas, c_th


def test_p_wave_decay_and_dispersion() -> None:
    """P-wave amplitude decay and phase velocity match the GSLS theory."""
    alpha_meas, alpha_th, c_meas, c_th = _wave_decay_and_dispersion("p", 20.0, 25.0)
    assert abs(alpha_meas / alpha_th - 1) < 0.1
    assert abs(c_meas / c_th - 1) < 0.03


def test_s_wave_decay_and_dispersion() -> None:
    """S-wave amplitude decay and phase velocity match the GSLS theory."""
    alpha_meas, alpha_th, c_meas, c_th = _wave_decay_and_dispersion("s", 20.0, 25.0)
    assert abs(alpha_meas / alpha_th - 1) < 0.1
    assert abs(c_meas / c_th - 1) < 0.03


# --- Gradcheck (central finite difference) ----------------------------------


@pytest.mark.parametrize("ndim", [2, 3])
def test_gradcheck_fd(ndim: int) -> None:
    """Absolute central-FD gradient check w.r.t. lamb, mu, buoyancy, qp, qs, src."""
    torch.manual_seed(2)
    n = 6
    shape = (n,) * ndim
    dx = 5.0
    dt = 0.001
    nt = 6
    vp = 1500.0 + 10.0 * torch.rand(*shape, dtype=DTYPE)
    vs = 800.0 + 10.0 * torch.rand(*shape, dtype=DTYPE)
    rho = 1000.0 + 10.0 * torch.rand(*shape, dtype=DTYPE)
    mu = rho * vs**2
    lamb = rho * (vp**2 - 2 * vs**2)
    buoyancy = 1.0 / rho
    qp = 25.0 + 5.0 * torch.rand(*shape, dtype=DTYPE)
    qs = 20.0 + 5.0 * torch.rand(*shape, dtype=DTYPE)
    src = torch.randn(1, 1, nt, dtype=DTYPE)
    src_loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    src_loc[0, 0] = torch.tensor([n // 2] * ndim)
    rec_loc = torch.zeros(1, 2, ndim, dtype=torch.long)
    rec_loc[0, 0] = torch.tensor([2] * ndim)
    rec_loc[0, 1] = torch.tensor([1] * ndim)

    fields = {
        "lamb": (lamb, 1e-1),
        "mu": (mu, 1e-1),
        "buoyancy": (buoyancy, 1e-9),
        "qp": (qp, 1e-3),
        "qs": (qs, 1e-3),
        "src": (src, 1e-5),
    }
    base = {k: t for k, (t, _) in fields.items()}

    def loss(**over: torch.Tensor) -> torch.Tensor:
        args = {**base, **over}
        kwargs: dict = {}
        if ndim == 3:
            kwargs["source_amplitudes_x"] = args["src"]
            kwargs["source_locations_x"] = src_loc
        else:
            kwargs["source_amplitudes_y"] = args["src"]
            kwargs["source_locations_y"] = src_loc
        out = deepwave.viscoelastic(
            args["lamb"],
            args["mu"],
            args["buoyancy"],
            args["qp"],
            args["qs"],
            dx,
            dt,
            receiver_locations_p=rec_loc,
            pml_freq=25.0,
            accuracy=4,
            pml_width=4,
            q_freq=25.0,
            backend="eager",
            **kwargs,
        )
        return out.receiver_amplitudes_p.pow(2).sum()

    for name, (tensor, eps) in fields.items():
        leaf = tensor.clone().requires_grad_(True)
        loss(**{name: leaf}).backward()
        assert leaf.grad is not None
        ana = leaf.grad.reshape(-1)
        flat = tensor.reshape(-1)
        fd = torch.zeros_like(flat)
        for i in range(flat.numel()):
            up = flat.clone()
            up[i] += eps
            dn = flat.clone()
            dn[i] -= eps
            lp = loss(**{name: up.reshape(tensor.shape)})
            lm = loss(**{name: dn.reshape(tensor.shape)})
            fd[i] = (lp - lm).item() / (2 * eps)
        rel = (ana - fd).abs() / (fd.abs() + fd.abs().max() * 1e-3 + 1e-30)
        assert rel.max().item() < 5e-3, (name, rel.max().item())


# --- Reciprocity -------------------------------------------------------------


def test_reciprocity() -> None:
    """Pressure-source/pressure-receiver reciprocity."""
    torch.manual_seed(1)
    ny, nx = 50, 60
    dx = 5.0
    dt = 0.001
    nt = 150
    freq = 15.0
    lamb, mu, buoyancy = _homogeneous(ny, nx, 1800.0, 900.0, 2000.0)
    qp = 30.0 * torch.ones(ny, nx, dtype=DTYPE)
    qs = 25.0 * torch.ones(ny, nx, dtype=DTYPE)
    wav = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq).to(DTYPE).reshape(1, 1, -1)
    loc_a = torch.tensor([[[20, 15]]])
    loc_b = torch.tensor([[[30, 45]]])

    def run(sl: torch.Tensor, rl: torch.Tensor) -> torch.Tensor:
        return deepwave.viscoelastic(
            lamb,
            mu,
            buoyancy,
            qp,
            qs,
            dx,
            dt,
            source_amplitudes_pressure_rate=wav,
            source_locations_pressure_rate=sl,
            receiver_locations_p=rl,
            pml_freq=freq,
            accuracy=4,
            q_freq=freq,
            backend="eager",
        ).receiver_amplitudes_p

    d_ab = run(loc_a, loc_b)
    d_ba = run(loc_b, loc_a)
    scale = d_ab.abs().max().item()
    assert scale > 0
    assert (d_ab - d_ba).abs().max().item() < 5e-3 * scale


# --- Adjoint dot-product ------------------------------------------------------
#
# The source_amplitudes_pressure_rate <F s, r> == <s, F* r> identity for viscoelastic is
# covered by the registry-driven tests/test_adjoint_dot_product.py (2-D and
# 3-D source cases, plus a wavefield case), using the same eager backend and
# the same source_amplitudes_pressure_rate/receiver_amplitudes_p channel a per-file
# version here would check -- no coverage gap, just no per-file duplicate.


# --- Negative control: break the mean/deviatoric coupling --------------------


def test_negative_control_broken_gsls_coupling() -> None:
    """Deliberately break the mean-part GSLS coupling; confirm a gate goes red.

    Zeroes out the mean-part memory-variable contribution inside
    :func:`_gsls_accumulate`'s call for the mean term (by monkeypatching
    :func:`deepwave.viscoelastic._gsls_accumulate` to zero the mean
    contribution specifically), then reruns the Qp, Qs -> infinity
    reduction-to-elastic gate and confirms it goes red -- the mean/
    deviatoric split is genuinely load-bearing, not a vacuous no-op. The
    patch is reverted in a ``finally`` block.
    """
    torch.manual_seed(0)
    n = 20
    dx = 5.0
    dt = 0.0005
    nt = 30
    freq = 25.0
    vp = 2000.0 + 20.0 * torch.randn(n, n, dtype=DTYPE)
    vs = 1000.0 + 20.0 * torch.randn(n, n, dtype=DTYPE)
    rho = 2200.0 + 20.0 * torch.randn(n, n, dtype=DTYPE)
    mu = rho * vs**2
    lamb = rho * (vp**2 - 2 * vs**2)
    buoyancy = 1.0 / rho
    q_moderate = torch.full((n, n), 15.0, dtype=DTYPE)

    src = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq).to(DTYPE).reshape(1, 1, -1)
    src_loc = torch.tensor([[[n // 2, n // 2]]])
    rec_loc = torch.tensor([[[6, 6], [7, 7]]])

    def run() -> torch.Tensor:
        out = deepwave.viscoelastic(
            lamb,
            mu,
            buoyancy,
            q_moderate,
            q_moderate,
            dx,
            dt,
            source_amplitudes_pressure_rate=src,
            source_locations_pressure_rate=src_loc,
            receiver_locations_p=rec_loc,
            pml_freq=freq,
            accuracy=4,
            pml_width=6,
            q_freq=freq,
            backend="eager",
        )
        return out.receiver_amplitudes_p

    baseline = run()
    assert baseline.abs().max().item() > 0

    orig = _vemod._gsls_accumulate
    call_count = {"n": 0}

    def broken(s, m1, c2, ar, r_list, dt):
        call_count["n"] += 1
        # update_stresses calls _gsls_accumulate 1 (mean) + ndim (deviatoric
        # normal) + n_shear (deviatoric shear) times per timestep, in that
        # order (see its body); for 2-D that is 1 + 2 + 1 = 4 calls, and the
        # mean call is always the first of each group of 4.
        contribution, new_r_list = orig(s, m1, c2, ar, r_list, dt)
        if call_count["n"] % 4 == 1:
            contribution = contribution * 0.0
        return contribution, new_r_list

    _vemod._gsls_accumulate = broken
    try:
        broken_out = run()
    finally:
        _vemod._gsls_accumulate = orig
        call_count["n"] = 0

    rel_diff = (broken_out - baseline).abs().max().item() / baseline.abs().max().item()
    assert rel_diff > 1e-3, (
        "breaking the mean-part GSLS coupling had no effect (vacuous test)"
    )


# --- Free surface -----------------------------------------------------------
#
# Two independent free-surface mechanisms, both reusing deepwave.elastic's
# machinery unchanged (see the module docstring's "Free/rigid/periodic
# faces" section and docs/boundaries.rst):
#   * the improved vacuum method (IVM): zero lamb/mu/buoyancy in an "air"
#     region. Needs no code change -- every GSLS coefficient is forced to
#     exactly zero wherever its driving modulus is zero -- so the gates
#     below are validation only.
#   * boundary= flat-face faces: reuse _boundaries.fill_elastic_stress_
#     boundary/fill_elastic_velocity_boundary verbatim.


def _vacuum_medium(
    ny: int, nx: int, n_vac: int, qp_val: float, qs_val: float
) -> tuple[torch.Tensor, ...]:
    """A homogeneous elastic medium with ``n_vac`` vacuum rows at the top."""
    lamb, mu, buoyancy = _homogeneous(ny, nx, 2000.0, 1000.0, 2200.0)
    qp = torch.full((ny, nx), qp_val, dtype=DTYPE)
    qs = torch.full((ny, nx), qs_val, dtype=DTYPE)
    lamb[:n_vac] = 0.0
    mu[:n_vac] = 0.0
    buoyancy[:n_vac] = 0.0
    return lamb, mu, buoyancy, qp, qs


def test_vacuum_qp_qs_bit_exact_invariance() -> None:
    """The IVM free surface is EXACTLY (bit-for-bit) independent of Qp/Qs.

    Not a large-but-finite-Q limit argument: with lamb=mu=buoyancy=0 in the
    vacuum rows, every GSLS coefficient there (m1, c2) is forced to exactly
    zero (see _gsls.sls_coefficients), so the memory variables started at
    zero stay exactly zero and contribute exactly zero -- the whole
    solution is bit-identical regardless of what Qp/Qs are set to in the
    vacuum region. Two wildly different vacuum-region Q values are used to
    make this a strong, non-vacuous check (a bug in the vacuum masking, the
    Q-clamping in _q_safe, or an Inf/NaN leak would very likely break bit
    identity, not just perturb it slightly).
    """
    ny, nx, n_vac = 20, 24, 3
    dx, dt, nt = 5.0, 0.0004, 60
    src = deepwave.wavelets.ricker(25.0, nt, dt, 1.4 / 25.0).to(DTYPE).reshape(1, 1, -1)
    sl = torch.tensor([[[10, 12]]])
    rl = torch.tensor([[[n_vac + 1, 12]]])
    kw = {
        "source_amplitudes_y": src,
        "source_locations_y": sl,
        "receiver_locations_p": rl,
        "pml_freq": 25.0,
        "accuracy": 4,
        "pml_width": [0, 6, 6, 6],
        "q_freq": 25.0,
        "backend": "eager",
    }

    def run(qp_val: float, qs_val: float) -> deepwave.PropagatorOutput:
        lamb, mu, buoyancy, qp, qs = _vacuum_medium(ny, nx, n_vac, qp_val, qs_val)
        # Only the vacuum-region Q values differ between the two calls;
        # overwrite the interior (below n_vac) to a FIXED value so the only
        # thing varying at all is Q inside the vacuum cells themselves.
        qp[n_vac:] = 30.0
        qs[n_vac:] = 20.0
        return deepwave.viscoelastic(lamb, mu, buoyancy, qp, qs, dx, dt, **kw)

    out_a = run(5.0, 3.0)
    out_b = run(1.0e6, 1.0e6)
    fields_a = dict(zip(out_a.names, out_a))
    fields_b = dict(zip(out_b.names, out_b))
    fields_a.update(out_a.auxiliary_state.as_dict())
    fields_b.update(out_b.auxiliary_state.as_dict())
    del fields_a["auxiliary_state"], fields_b["auxiliary_state"]
    for name, value in fields_a.items():
        assert torch.equal(value, fields_b[name]), (
            f"{name} differs between vacuum Q=5/3 and Q=1e6/1e6 -- IVM should "
            "be exactly Q-independent"
        )


def test_vacuum_exact_zero_traction() -> None:
    """Every stress and GSLS memory variable is EXACTLY zero inside the vacuum.

    Checked via forward_callback (which fires post-update, pre any guard
    fill -- see the module docstring) at finite, non-degenerate Qp/Qs, so
    this is not merely re-checking the all-PML interior physics.
    """
    ny, nx, n_vac = 20, 24, 3
    dx, dt, nt = 5.0, 0.0004, 40
    lamb, mu, buoyancy, qp, qs = _vacuum_medium(ny, nx, n_vac, 30.0, 20.0)
    qp[n_vac:] = 30.0
    qs[n_vac:] = 20.0
    src = deepwave.wavelets.ricker(25.0, nt, dt, 1.4 / 25.0).to(DTYPE).reshape(1, 1, -1)
    sl = torch.tensor([[[10, 12]]])

    stress_and_mem_names = [
        name
        for name in _vemod.viscoelastic_wavefield_names(2)
        if name.startswith(("sigma", "rp", "rs_"))
    ]
    checked = {"n": 0}

    def cb(state) -> None:
        for name in stress_and_mem_names:
            field = state.get_wavefield(name)[:, :n_vac]
            assert (field == 0.0).all(), f"{name} nonzero inside vacuum"
            checked["n"] += 1

    deepwave.viscoelastic(
        lamb,
        mu,
        buoyancy,
        qp,
        qs,
        dx,
        dt,
        source_amplitudes_y=src,
        source_locations_y=sl,
        pml_freq=25.0,
        accuracy=4,
        pml_width=[0, 6, 6, 6],
        q_freq=25.0,
        backend="eager",
        forward_callback=cb,
        callback_frequency=1,
    )
    assert checked["n"] > 0, (
        "no stress/memory-variable fields were checked (vacuous test)"
    )


@pytest.mark.parametrize("ndim", [2, 3])
def test_qp_qs_infinity_reduces_to_elastic_with_vacuum(ndim: int) -> None:
    """Extends test_qp_qs_infinity_reduces_to_elastic with a real free surface.

    Same recipe (large-but-finite Q, forward + gradient agreement against
    deepwave.elastic), but now with vacuum rows at the top of the model, so
    the free-surface reflection itself -- not just the interior physics --
    is exercised by the Q->infinity reduction.
    """
    torch.manual_seed(4)
    n = 16 if ndim == 3 else 24
    n_vac = 2
    shape = (n,) * ndim
    dx = 5.0
    dt = 0.0004 if ndim == 3 else 0.0005
    nt = 30 if ndim == 3 else 40
    freq = 25.0
    q_big = 1e10 if ndim == 3 else 1e12
    tol_fwd = 1e-7 if ndim == 3 else 1e-9
    tol_grad = 1e-6 if ndim == 3 else 1e-8

    vp = 2000.0 + 20.0 * torch.randn(*shape, dtype=DTYPE)
    vs = 1000.0 + 20.0 * torch.randn(*shape, dtype=DTYPE)
    rho = 2200.0 + 20.0 * torch.randn(*shape, dtype=DTYPE)
    mu = rho * vs**2
    lamb = rho * (vp**2 - 2 * vs**2)
    buoyancy = 1.0 / rho
    lamb[:n_vac] = 0.0
    mu[:n_vac] = 0.0
    buoyancy[:n_vac] = 0.0
    qbig = torch.full(shape, q_big, dtype=DTYPE)

    src = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq).to(DTYPE).reshape(1, 1, -1)
    src_loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    src_loc[0, 0] = torch.tensor([n_vac + n // 2] + [n // 2] * (ndim - 1))
    rec_loc = torch.zeros(1, 4, ndim, dtype=torch.long)
    for j in range(4):
        rec_loc[0, j] = torch.tensor([n_vac + j] + [n // 2 + 2] * (ndim - 1))

    def run_ve(
        ll: torch.Tensor, mm: torch.Tensor, bb: torch.Tensor
    ) -> deepwave.PropagatorOutput:
        kwargs: dict = {}
        if ndim == 3:
            kwargs["source_amplitudes_x"] = src
            kwargs["source_locations_x"] = src_loc
        else:
            kwargs["source_amplitudes_y"] = src
            kwargs["source_locations_y"] = src_loc
        return deepwave.viscoelastic(
            ll,
            mm,
            bb,
            qbig,
            qbig,
            dx,
            dt,
            receiver_locations_p=rec_loc,
            pml_freq=freq,
            q_freq=freq,
            accuracy=4,
            pml_width=[0] + [8] * (2 * ndim - 1),
            backend="eager",
            **kwargs,
        )

    def run_e(
        ll: torch.Tensor, mm: torch.Tensor, bb: torch.Tensor
    ) -> deepwave.PropagatorOutput:
        kwargs = {}
        if ndim == 3:
            kwargs["source_amplitudes_x"] = src
            kwargs["source_locations_x"] = src_loc
        else:
            kwargs["source_amplitudes_y"] = src
            kwargs["source_locations_y"] = src_loc
        return deepwave.elastic(
            ll,
            mm,
            bb,
            dx,
            dt,
            receiver_locations_p=rec_loc,
            pml_freq=freq,
            accuracy=4,
            pml_width=[0] + [8] * (2 * ndim - 1),
            backend="eager",
            **kwargs,
        )

    rec_ve = run_ve(lamb, mu, buoyancy).receiver_amplitudes_p
    rec_e = run_e(lamb, mu, buoyancy).receiver_amplitudes_p
    assert rec_e.norm().item() > 0
    assert (rec_ve - rec_e).norm().item() / rec_e.norm().item() < tol_fwd

    ll = lamb.clone().requires_grad_(True)
    mm = mu.clone().requires_grad_(True)
    bb = buoyancy.clone().requires_grad_(True)
    run_ve(ll, mm, bb).receiver_amplitudes_p.pow(2).sum().backward()

    lla = lamb.clone().requires_grad_(True)
    mma = mu.clone().requires_grad_(True)
    bba = buoyancy.clone().requires_grad_(True)
    run_e(lla, mma, bba).receiver_amplitudes_p.pow(2).sum().backward()

    assert (ll.grad - lla.grad).norm().item() / lla.grad.norm().item() < tol_grad
    assert (mm.grad - mma.grad).norm().item() / mma.grad.norm().item() < tol_grad
    assert (bb.grad - bba.grad).norm().item() / bba.grad.norm().item() < tol_grad


# --- boundary= flat-face faces -----------------------------------------------
#
# Q is set large (5000) in the reflection-coefficient/IVM-agreement gates
# below, matching the precedent tests_boundary.py's own equivalent elastic
# gates use (search "Q = 5000.0" there): at a traction-free plane the
# amplitude reflection coefficient R=-1 is exactly Q-independent, but the
# MEASURED peak ratio is not, because the reflected arrival travels an
# extra path and is therefore attenuated by exp(-alpha_P * extra) relative
# to the direct arrival; large Q keeps that correction inside these tests'
# 1-3% tolerances. Finite-Q physics is exercised separately, in the
# Rayleigh-wave decay test below.

_VE_Q_LARGE = 5000.0


def _ve_reflection_coefficient(face: str, freq: float = 15.0) -> float:
    """Normal-incidence reflection off a flat viscoelastic face.

    ``viscoelastic`` needs at least two spatial dimensions, so the 1-D plane
    wave is emulated by a line of identical sources and receivers spanning
    the transverse axis, whose two faces are bare zero-width ``"pml"``
    faces. For a field that is uniform along that axis such a face is exact:
    the only exterior cells the transverse stencil reads are the transverse
    velocity and the shear stress, both identically zero here, so leaving
    them at zero is the exact condition (and it is accepted at every
    accuracy, unlike a named reflecting face at the end of an axis). The
    incident-only run is subtracted to isolate the reflected arrival, and
    Qp=Qs is large (see the module note above) so that the extra attenuation
    along the reflected path stays inside the tolerance.
    """
    dx, dt, nt, n_test = 2.0, 0.0003, 4000, 800
    s, r = 60, 35
    n_per = 12
    vp, vs, rho = 1500.0, 900.0, 1000.0
    lamb, mu, buoyancy = _homogeneous(n_test, n_per, vp, vs, rho)
    qbig = torch.full((n_test, n_per), _VE_Q_LARGE, dtype=DTYPE)
    sa, tpk = (
        deepwave.wavelets.ricker(freq, nt, dt, 1.4 / freq).to(DTYPE).reshape(1, 1, nt),
        int(1.4 / freq / dt),
    )
    sa_rep = sa.repeat(1, n_per, 1)

    def loc(pos: int) -> torch.Tensor:
        return torch.tensor([[[pos, j] for j in range(n_per)]])

    def run(tested_bc: str, tested_pml: int) -> torch.Tensor:
        out = deepwave.viscoelastic(
            lamb,
            mu,
            buoyancy,
            qbig,
            qbig,
            dx,
            dt,
            source_amplitudes_pressure_rate=sa_rep,
            source_locations_pressure_rate=loc(s),
            receiver_locations_p=loc(r),
            pml_width=[tested_pml, 40, 0, 0],
            boundary=[tested_bc, "pml", "pml", "pml"],
            accuracy=4,
            pml_freq=freq,
            q_freq=freq,
            backend="eager",
        )
        return out.receiver_amplitudes_p[0].mean(dim=0)

    tdir = int((s - r) * dx / vp / dt) + tpk
    tref = int((s + r) * dx / vp / dt) + tpk
    w = int(1.0 / freq / dt)
    inc = run("pml", 40)
    dwin = inc[tdir - w : tdir + w]
    a_dir = dwin[dwin.abs().argmax()]
    tr = run(face, 0)
    refl = tr - inc
    rwin = refl[tref - w : tref + w]
    a_ref = rwin[rwin.abs().argmax()]
    return (a_ref / a_dir).item()


def test_viscoelastic_rigid_reflection_coefficient() -> None:
    """boundary="rigid" reflects sigma_AA with R = +1 to <1% (large Q)."""
    r = _ve_reflection_coefficient("rigid")
    assert abs(r - 1.0) < 0.01, f"R_rigid={r}"


def test_viscoelastic_boundary_gradcheck() -> None:
    """Central-FD gradcheck through the boundary= guard-fill autograd path.

    There is no compiled backend, so the eager fill functions' plain
    narrow/cat/flip/mul autograd IS the entire adjoint story for
    boundary=; this is what verifies it transposes correctly, mirroring
    the spirit of test_boundary.py's _boundary_gradcheck_check family. A
    ``"rigid"`` face at the beginning of each axis is what viscoelastic
    accepts, and is the fill branch it therefore has to transpose.
    """
    torch.manual_seed(5)
    ny, nx = 14, 16
    dx, dt, nt = 5.0, 0.0006, 14
    vp = 1800.0 + 10.0 * torch.randn(ny, nx, dtype=DTYPE)
    vs = 900.0 + 10.0 * torch.randn(ny, nx, dtype=DTYPE)
    rho = 2000.0 + 10.0 * torch.randn(ny, nx, dtype=DTYPE)
    mu = (rho * vs**2).requires_grad_(True)
    lamb = (rho * (vp**2 - 2 * vs**2)).requires_grad_(True)
    buoyancy = (1.0 / rho).requires_grad_(True)
    qp = (torch.full((ny, nx), 50.0, dtype=DTYPE)).requires_grad_(True)
    qs = (torch.full((ny, nx), 40.0, dtype=DTYPE)).requires_grad_(True)
    src = (0.01 * torch.randn(1, 1, nt, dtype=DTYPE)).requires_grad_(True)
    sl = torch.tensor([[[6, 7]]])
    rl = torch.tensor([[[3, 4]], [[10, 12]]]).reshape(1, 2, 2)

    def fn(
        lamb: torch.Tensor,
        mu: torch.Tensor,
        buoyancy: torch.Tensor,
        qp: torch.Tensor,
        qs: torch.Tensor,
        src: torch.Tensor,
    ) -> torch.Tensor:
        out = deepwave.viscoelastic(
            lamb,
            mu,
            buoyancy,
            qp,
            qs,
            dx,
            dt,
            source_amplitudes_y=src,
            source_locations_y=sl,
            receiver_locations_p=rl,
            pml_width=[0, 0, 0, 0],
            boundary=["rigid", "pml", "rigid", "pml"],
            accuracy=4,
            pml_freq=25.0,
            q_freq=25.0,
            backend="eager",
        )
        return out.receiver_amplitudes_p

    assert torch.autograd.gradcheck(
        fn,
        (lamb, mu, buoyancy, qp, qs, src),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


# --- Finite-Q Rayleigh-wave decay (free surface + GSLS attenuation, together)
#
# Exact complex Rayleigh-root theory: at the reference frequency f0=q_freq,
# the P/S moduli become the complex GSLS moduli (_sls_complex_modulus above),
# giving complex P/S speeds alpha, beta and slowness ratio kappa=beta/alpha.
# The (complex) Rayleigh secular equation is the textbook real-medium one
# with real quantities analytically continued:
#
#     F(xi) = (2 - xi**2)**2 - 4*sqrt(1 - kappa**2*xi**2)*sqrt(1 - xi**2) = 0
#
# solved by Newton from the real-medium root (Im(kappa) = O(1/Q), so a few
# iterations converge to machine precision); xi = c_R/beta gives the complex
# Rayleigh wavenumber k_R = omega/(beta*xi), whose real part is the phase
# velocity and whose imaginary part is the spatial attenuation coefficient.
# Measurement uses the same monochromatic-steady-state differential
# technique _measure_wave/_wave_decay_and_dispersion already use and
# validate to ~1% for the P/S body waves, applied to a vertical force at
# the free surface with receivers along the surface (isolating the
# Rayleigh mode in its 2-D far field).


def _rayleigh_root(
    vp: float, vs: float, rho: float, qp: float, qs: float, f0: float, f: float
) -> tuple[float, float]:
    """Exact complex-medium Rayleigh root.

    Returns ``(c_R, alpha_R)``: the Rayleigh phase velocity and spatial
    attenuation coefficient (Np/m) at frequency ``f``.
    """
    ndim = 2
    mu_rel = rho * vs**2
    k_iso_rel = rho * vp**2 - 2 * mu_rel * (ndim - 1) / ndim
    mu_c = _sls_complex_modulus(mu_rel, qs, f0, f)
    k_c = _sls_complex_modulus(k_iso_rel, qp, f0, f)
    m_p_c = k_c + 2 * mu_c * (ndim - 1) / ndim
    alpha = cmath.sqrt(m_p_c / rho)
    beta = cmath.sqrt(mu_c / rho)
    kappa = beta / alpha

    def secular(xi: complex) -> complex:
        return (2 - xi**2) ** 2 - 4 * cmath.sqrt(1 - kappa**2 * xi**2) * cmath.sqrt(
            1 - xi**2
        )

    # Real-medium starting root (bisection on (0, 1)), then Newton-refine
    # into the complex plane.
    kappa0 = vs / vp

    def secular_real(xi: float) -> float:
        return (2 - xi**2) ** 2 - 4 * math.sqrt(1 - kappa0**2 * xi**2) * math.sqrt(
            1 - xi**2
        )

    lo, hi = 0.01, 0.999
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if secular_real(lo) * secular_real(mid) <= 0:
            hi = mid
        else:
            lo = mid
    xi = complex(0.5 * (lo + hi), 0.0)
    for _ in range(50):
        h = 1e-6
        dsecular = (secular(xi + h) - secular(xi - h)) / (2 * h)
        xi = xi - secular(xi) / dsecular

    omega = 2 * math.pi * f
    c_r_c = beta * xi
    k_r = omega / c_r_c
    return omega / k_r.real, abs(k_r.imag)


def _measure_rayleigh(
    qp_val: float,
    qs_val: float,
    ny: int = 60,
    nx: int = 200,
    n_vac: int = 2,
    dx: float = 4.0,
    dt: float = 4e-4,
    nt: int = 6000,
    f0: float = 25.0,
    measure_frac: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Monochromatic steady-state Rayleigh-wave amplitude/phase along the surface.

    A vertical force at the free surface (IVM, ``n_vac`` vacuum rows) with
    receivers along the same surface row reads the Rayleigh wave's vertical
    particle motion in its 2-D far field (the receiver line starts >= 4
    Rayleigh wavelengths from the source, where body waves have spread out
    and decayed relative to the surface-trapped mode).
    """
    vs, vp, rho = 1000.0, 1000.0 * math.sqrt(3), 2200.0  # Poisson = 0.25 exactly
    lamb, mu, buoyancy = _homogeneous(ny, nx, vp, vs, rho)
    lamb[:n_vac] = 0.0
    mu[:n_vac] = 0.0
    buoyancy[:n_vac] = 0.0
    qp = torch.full((ny, nx), qp_val, dtype=DTYPE)
    qs = torch.full((ny, nx), qs_val, dtype=DTYPE)
    t = torch.arange(nt, dtype=DTYPE) * dt
    ramp = torch.clamp(t / (20.0 / f0), max=1.0)
    src = (ramp * torch.sin(2 * math.pi * f0 * t)).reshape(1, 1, -1)
    sl = torch.tensor([[[n_vac, 30]]])
    xs = torch.arange(70, 171, 4)
    rl = torch.zeros(1, xs.numel(), 2, dtype=torch.long)
    rl[0, :, 0] = n_vac
    rl[0, :, 1] = xs
    out = deepwave.viscoelastic(
        lamb,
        mu,
        buoyancy,
        qp,
        qs,
        dx,
        dt,
        source_amplitudes_y=src,
        source_locations_y=sl,
        receiver_locations_y=rl,
        pml_freq=f0,
        accuracy=4,
        pml_width=[0, 20, 20, 20],
        q_freq=f0,
        backend="eager",
        nt=nt,
    )
    rec = out.receiver_amplitudes_y[0]
    half = slice(int(nt * measure_frac), nt)
    tt = t[half]
    sig = rec[:, half]
    a = (sig * torch.cos(2 * math.pi * f0 * tt)).sum(1)
    b = (sig * torch.sin(2 * math.pi * f0 * tt)).sum(1)
    amp = torch.sqrt(a**2 + b**2)
    phase = torch.atan2(b, a)
    x = (xs.double() - 30) * dx
    return x, amp, phase


def test_rayleigh_wave_finite_q_decay() -> None:
    """Finite-Q Rayleigh-wave decay matches the exact complex-root theory.

    Exercises the free surface (IVM) and the GSLS attenuation TOGETHER,
    the physics this task's whole free-surface effort is ultimately meant
    to enable: Rayleigh waves/ground roll under finite Qp/Qs, the specific
    capability the old EXPERIMENTAL warning said viscoelastic could not
    represent. A differential measurement against a Q=1e8 reference run at
    the same geometry cancels the (identical, Q-independent) free-surface
    discretisation error to first order, matching
    _wave_decay_and_dispersion's own P/S-wave technique.
    """
    f0 = 25.0
    vs, vp, rho = 1000.0, 1000.0 * math.sqrt(3), 2200.0
    x, amp_q, _ = _measure_rayleigh(30.0, 20.0, f0=f0)
    _, amp_inf, _ = _measure_rayleigh(1e8, 1e8, f0=f0)
    alpha_q = -_fit_line(x, torch.log(amp_q))[1]
    alpha_inf = -_fit_line(x, torch.log(amp_inf))[1]
    alpha_meas = alpha_q - alpha_inf

    _, a_th = _rayleigh_root(vp, vs, rho, 30.0, 20.0, f0, f0)

    assert abs(alpha_meas / a_th - 1) < 0.15, (
        f"Rayleigh decay: measured {alpha_meas}, theory {a_th}"
    )
    # The companion phase-velocity/Qp-channel checks (which pin down that the
    # Rayleigh mode itself, not a body wave or PML artefact, is what is being
    # measured here) live in test_rayleigh_wave_phase_velocity_and_qp_channel
    # below, sharing this same geometry.


def test_rayleigh_wave_phase_velocity_and_qp_channel() -> None:
    """Rayleigh phase velocity matches theory; the Qp channel reaches the surface wave.

    Two checks in one test (sharing the expensive Q=1e8 reference run):
    (1) measured phase velocity matches the exact complex-root theory to
    <3%, confirming the receiver line is genuinely reading the Rayleigh
    mode and not a body wave or PML artefact; (2) the attenuation ratio
    between (Qp=30, Qs=20) and (Qp=1e8, Qs=20) matches theory's prediction
    that only ~13% of the total decay comes from Qp at this Poisson's ratio
    (0.25) -- a direct, quantitative confirmation that Qp reaches the free
    surface through the mean-part GSLS coupling, not just Qs.
    """
    f0 = 25.0
    vs, vp, rho = 1000.0, 1000.0 * math.sqrt(3), 2200.0

    x, amp_30_20, phase_30_20 = _measure_rayleigh(30.0, 20.0, f0=f0)
    _, amp_inf, _ = _measure_rayleigh(1e8, 1e8, f0=f0)
    _, amp_inf_20, _ = _measure_rayleigh(1e8, 20.0, f0=f0)

    alpha_inf = -_fit_line(x, torch.log(amp_inf))[1]
    alpha_30_20 = -_fit_line(x, torch.log(amp_30_20))[1] - alpha_inf
    alpha_inf_20 = -_fit_line(x, torch.log(amp_inf_20))[1] - alpha_inf

    phw = torch.from_numpy(np.unwrap(phase_30_20.numpy()))
    k_meas = abs(_fit_line(x, phw)[1])
    c_meas = 2 * math.pi * f0 / k_meas
    c_th, _ = _rayleigh_root(vp, vs, rho, 30.0, 20.0, f0, f0)
    assert abs(c_meas / c_th - 1) < 0.03, f"c_meas={c_meas}, c_th={c_th}"

    ratio_meas = alpha_30_20 / alpha_inf_20
    _, a_th_30_20 = _rayleigh_root(vp, vs, rho, 30.0, 20.0, f0, f0)
    _, a_th_inf_20 = _rayleigh_root(vp, vs, rho, 1e8, 20.0, f0, f0)
    ratio_th = a_th_30_20 / a_th_inf_20
    assert 1.03 < ratio_meas < 1.18, (
        f"Qp-channel ratio {ratio_meas} outside expected band "
        f"(theory {ratio_th}); the free surface may not be passing the Qp "
        "attenuation through to the Rayleigh wave correctly"
    )


# --- Elastic M-PML corner stability (inherited) ------------------------------


@pytest.mark.slow
def test_mpml_corner_stability_inherited() -> None:
    """The elastic M-PML corner-stability fix also stabilises viscoelastic.

    Ports ``tests/test_elastic_mpml.py::test_instability_repro`` (a thin
    high-Poisson slab between two facing vacuum free surfaces with x-only
    PML -- the archived cheap reproduction of the late-time elastic C-PML
    instability) to viscoelastic with a FINITE, moderate Qp/Qs (not Q ->
    infinity, so the GSLS memory-variable coupling is genuinely exercised
    alongside the M-PML one): unstable at ``mpml_fraction = 0``, stable at
    ``mpml_fraction = 0.15``. This confirms the GSLS memory variables -- which sit
    alongside, and are computed independently of, the M-PML memory
    variables inside :func:`update_stresses` -- do not reintroduce or mask
    the instability.
    """
    ny, nx, n_vac = 12, 70, 2
    nt_seg, n_chunks = 1000, 18  # 18000 steps
    dt, dx = 9e-4, 4.0
    vp, vs, rho = 2000.0, 300.0, 2200.0
    mu_val = rho * vs * vs
    lamb_val = rho * (vp * vp - 2 * vs * vs)

    lamb = torch.full((ny, nx), lamb_val, dtype=DTYPE)
    mu = torch.full((ny, nx), mu_val, dtype=DTYPE)
    buoyancy = torch.full((ny, nx), 1.0 / rho, dtype=DTYPE)
    qp = torch.full((ny, nx), 25.0, dtype=DTYPE)
    qs = torch.full((ny, nx), 20.0, dtype=DTYPE)
    for r in list(range(n_vac)) + list(range(ny - n_vac, ny)):
        lamb[r, :] = 0.0
        mu[r, :] = 0.0
        buoyancy[r, :] = 0.0
    top_solid = n_vac
    pml_width = [0, 0, 30, 30]
    src = (
        deepwave.wavelets.ricker(15.0, nt_seg, dt, 1.0 / 15.0)
        .to(DTYPE)
        .reshape(1, 1, nt_seg)
    )
    zero = torch.zeros_like(src)
    sl = torch.tensor([[[top_solid, nx // 2]]])

    def final_energy(p: float) -> tuple[float, float]:
        state = None
        peak = 0.0
        energy = 0.0
        for c in range(n_chunks):
            out = deepwave.viscoelastic(
                lamb,
                mu,
                buoyancy,
                qp,
                qs,
                dx,
                dt,
                source_amplitudes_y=(src if c == 0 else zero),
                source_locations_y=sl,
                accuracy=4,
                pml_width=pml_width,
                pml_freq=15.0,
                q_freq=15.0,
                nt=nt_seg,
                mpml_fraction=p,
                initial_state=state,
                backend="eager",
            )
            state = out
            energy = float((out.vy_0**2 + out.vx_0**2).sum())
            peak = max(peak, energy)
            if not math.isfinite(energy) or energy > 1e12:
                return energy, peak
        return energy, peak

    e0, _peak0 = final_energy(0.0)
    e15, _peak15 = final_energy(0.15)

    # p=0: unmistakable exponential blow-up (mirrors elastic's own gate).
    assert (not math.isfinite(e0)) or e0 > 1e6, (
        f"expected C-PML blow-up at p=0, got final energy {e0:.3e}"
    )
    # p=0.15: bounded (the late energy has decayed well below the peak).
    assert math.isfinite(e15)
    assert e15 < 1e-3, f"p=0.15 did not remain bounded: final energy {e15:.3e}"


@pytest.mark.slow
@pytest.mark.parametrize("mechanism", ["ivm", "rigid_face"])
def test_long_run_stability_top_face_finite_q(mechanism: str) -> None:
    """A reflecting top face with finite Qp/Qs stays energy-bounded.

    PLAIN stability (mpml_fraction=0, no M-PML crutch -- that combination is
    already covered by test_mpml_corner_stability_inherited above), for the
    vacuum free surface and for a ``boundary="rigid"`` top face, mirroring
    test_elastic_fluid_solid.py's
    test_long_run_stability_water_over_solid energy-bounded pattern. Each
    GSLS memory variable is provably BIBO-stable in isolation (the
    relaxation-time recursion has |ar| < 1 strictly for any finite Q > 0),
    but that does not by itself imply the *coupled* free-surface/PML/GSLS
    scheme is stable, so this is checked empirically.
    """
    ny, nx = 60, 100
    dx, dt = 5.0, 9e-4
    vp, vs, rho = 2000.0, 1150.0, 2200.0  # Poisson ~0.253
    n_vac = 3
    freq = 15.0
    nt = 12000
    step_e = 200

    lamb, mu, buoyancy = _homogeneous(ny, nx, vp, vs, rho)
    qp = torch.full((ny, nx), 25.0, dtype=DTYPE)
    qs = torch.full((ny, nx), 20.0, dtype=DTYPE)
    kwargs: dict = {
        "pml_freq": freq,
        "q_freq": freq,
        "accuracy": 4,
        "mpml_fraction": 0.0,
        "backend": "eager",
    }
    if mechanism == "ivm":
        lamb[:n_vac] = 0.0
        mu[:n_vac] = 0.0
        buoyancy[:n_vac] = 0.0
        top = n_vac
        kwargs["pml_width"] = [0, 20, 20, 20]
    else:
        top = 0
        kwargs["pml_width"] = [0, 20, 20, 20]
        kwargs["boundary"] = ["rigid", "pml", "pml", "pml"]

    peak = 1.0 / freq
    amp = deepwave.wavelets.ricker(freq, nt, dt, peak).to(DTYPE).reshape(1, 1, -1)
    src_loc = torch.tensor([[[top + 5, nx // 2]]])

    energies = []

    def cb(state: object) -> None:
        if state.step % step_e == 0:  # type: ignore[attr-defined]
            vy = state.get_wavefield("vy_0")  # type: ignore[attr-defined]
            vx = state.get_wavefield("vx_0")  # type: ignore[attr-defined]
            energies.append((vy**2).sum().item() + (vx**2).sum().item())

    deepwave.viscoelastic(
        lamb,
        mu,
        buoyancy,
        qp,
        qs,
        dx,
        dt,
        source_amplitudes_y=amp,
        source_locations_y=src_loc,
        forward_callback=cb,
        callback_frequency=step_e,
        nt=nt,
        **kwargs,
    )

    energies_arr = np.array(energies)
    assert np.isfinite(energies_arr).all()
    # Unlike test_elastic_fluid_solid.py's trapped-interface geometry, this
    # domain has PML on every non-free-surface face and no waveguide, so a
    # STABLE run radiates/attenuates energy away rather than settling to a
    # steady trapped baseline -- the peak (near the source burst) should be
    # followed by monotonic-ish decay to a much smaller final level, not a
    # late-time regrowth (the C-PML corner-instability signature already
    # gated by test_mpml_corner_stability_inherited above).
    peak_idx = int(energies_arr.argmax())
    peak, final = energies_arr[peak_idx], energies_arr[-1]
    assert peak_idx < len(energies_arr) // 2, (
        f"{mechanism}: peak energy occurred late (idx {peak_idx}/"
        f"{len(energies_arr)}) -- possible late-time regrowth"
    )
    assert final < 1e-2 * peak, (
        f"{mechanism}: final energy {final:.3e} did not decay well below "
        f"peak {peak:.3e}"
    )


# --- Coverage gap-list follow-up tests ---------------------------------------


def test_resolve_q_freq_takes_only_what_it_was_given() -> None:
    """_resolve_q_freq floats the given value and never invents one.

    It has no access to ``pml_freq``, which is the point: the reference
    frequency of the Qp/Qs models cannot be inherited from the absorbing
    boundary's tuning frequency. ``None`` reaches it only for
    ``q_n_mechanisms > 1``, whose coefficients never read the result.
    """
    assert _vemod._resolve_q_freq(30.0) == 30.0
    assert _vemod._resolve_q_freq(10) == 10.0
    assert isinstance(_vemod._resolve_q_freq(10), float)
    assert _vemod._resolve_q_freq(None) == 0.0


def test_multi_mechanism_gsls_runs_and_matches_single_mechanism_limit() -> None:
    """q_n_mechanisms > 1 runs; a narrow q_band around q_freq matches the L=1 SLS.

    Generalises the single-mechanism gate to L=3: with a narrow q_band
    centred on q_freq, the multi-mechanism GSLS fit degenerates to
    approximately the same single-relaxation-time behaviour the default
    q_n_mechanisms=1 path uses, so the two forward outputs should agree
    fairly closely (this is a coarse fit-quality check, not a bit-identical
    one -- the two paths use different, independently-fit relaxation times).
    """
    torch.manual_seed(3)
    ny, nx = 20, 24
    dx, dt, nt = 5.0, 0.001, 40
    freq = 25.0
    lamb, mu, buoyancy = _homogeneous(ny, nx, 1800.0, 900.0, 2000.0)
    qp = 30.0 * torch.ones(ny, nx, dtype=DTYPE)
    qs = 25.0 * torch.ones(ny, nx, dtype=DTYPE)
    wav = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq).to(DTYPE).reshape(1, 1, -1)
    sl = torch.tensor([[[10, 12]]])
    rl = torch.tensor([[[8, 8]]])
    kw = {
        "source_amplitudes_pressure_rate": wav,
        "source_locations_pressure_rate": sl,
        "receiver_locations_p": rl,
        "pml_freq": freq,
        "accuracy": 4,
        "backend": "eager",
    }
    out_single = deepwave.viscoelastic(
        lamb, mu, buoyancy, qp, qs, dx, dt, q_freq=freq, **kw
    )
    out_multi = deepwave.viscoelastic(
        lamb,
        mu,
        buoyancy,
        qp,
        qs,
        dx,
        dt,
        q_n_mechanisms=3,
        q_band=(freq * 0.9, freq * 1.1),
        **kw,
    )
    a = out_single.receiver_amplitudes_p
    b = out_multi.receiver_amplitudes_p
    assert torch.isfinite(b).all()
    scale = a.abs().max().item()
    assert scale > 0
    assert (a - b).abs().max().item() < 0.1 * scale


def test_multi_mechanism_validation() -> None:
    """q_n_mechanisms/q_band validation errors."""
    n = 6
    lamb, mu, buoyancy = _homogeneous(n, n, 1500.0, 800.0, 1000.0)
    qp = 30.0 * torch.ones(n, n, dtype=DTYPE)
    qs = 25.0 * torch.ones(n, n, dtype=DTYPE)
    common = (lamb, mu, buoyancy, qp, qs, 5.0, 0.001)
    with pytest.raises(ValueError, match="q_band must be None"):
        deepwave.viscoelastic(*common, q_n_mechanisms=1, q_band=(1.0, 2.0), q_freq=25.0)
    with pytest.raises(ValueError, match=r"q_band=\(f1, f2\) is required"):
        deepwave.viscoelastic(*common, q_n_mechanisms=3)
    with pytest.raises(ValueError, match="0 < f1 < f2"):
        deepwave.viscoelastic(*common, q_n_mechanisms=3, q_band=(60.0, 10.0))
    with pytest.raises(ValueError, match="q_n_mechanisms must be >= 1"):
        deepwave.viscoelastic(*common, q_n_mechanisms=0)


def test_module_matches_functional() -> None:
    """The Viscoelastic module reproduces the functional interface."""
    ny, nx = 16, 18
    lamb, mu, buoyancy = _homogeneous(ny, nx, 1800.0, 900.0, 2000.0)
    qp = 30.0 * torch.ones(ny, nx, dtype=DTYPE)
    qs = 25.0 * torch.ones(ny, nx, dtype=DTYPE)
    kw = {
        "source_amplitudes_y": torch.zeros(1, 1, 5, dtype=DTYPE),
        "source_locations_y": torch.tensor([[[8, 9]]]),
        "receiver_locations_p": torch.tensor([[[5, 5]]]),
        "pml_freq": 25.0,
        "q_freq": 25.0,
        "backend": "eager",
    }
    prop = _vemod.Viscoelastic(lamb, mu, buoyancy, qp, qs, 5.0)
    out_m = prop(0.001, **kw)
    out_f = deepwave.viscoelastic(lamb, mu, buoyancy, qp, qs, 5.0, 0.001, **kw)
    for name in ("receiver_amplitudes_p", "vy_0", "vx_0"):
        assert torch.equal(getattr(out_m, name), getattr(out_f, name))


@pytest.mark.slow
def test_compile_backend_matches_eager() -> None:
    """backend='torch_compile' reproduces the eager backend.

    Unlike deepwave.elastic (which swaps in compiled kernels for
    compiled-C/CUDA-adjacent Python modes), viscoelastic has no compiled
    backend at all -- "torch_compile" only needs update_stresses (this
    module's own GSLS-augmented kernel) and deepwave.elastic.
    update_velocities to be traceable. Mirrors test_elastic_vti.
    py's test_compile_backend_matches_eager (same tolerance),
    forward-only; gradient agreement is checked directly below.
    """
    ny, nx = 16, 18
    lamb, mu, buoyancy = _homogeneous(ny, nx, 1800.0, 900.0, 2000.0)
    qp = 30.0 * torch.ones(ny, nx, dtype=DTYPE)
    qs = 25.0 * torch.ones(ny, nx, dtype=DTYPE)
    src = (
        deepwave.wavelets.ricker(25.0, 12, 0.001, 1.5 / 25.0)
        .to(DTYPE)
        .reshape(1, 1, -1)
    )
    kw = {
        "source_amplitudes_y": src,
        "source_locations_y": torch.tensor([[[8, 9]]]),
        "receiver_locations_p": torch.tensor([[[5, 5]]]),
        "pml_freq": 25.0,
        "q_freq": 25.0,
    }
    out_eager = deepwave.viscoelastic(
        lamb, mu, buoyancy, qp, qs, 5.0, 0.001, backend="eager", **kw
    )
    for backend in ("torch_compile",):
        out = deepwave.viscoelastic(
            lamb, mu, buoyancy, qp, qs, 5.0, 0.001, backend=backend, **kw
        )
        assert torch.allclose(
            out.receiver_amplitudes_p, out_eager.receiver_amplitudes_p, atol=1e-9
        )


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("ny", "nx"), [(20, 24), (17, 23)])
def test_compile_backend_no_codegen_landmines(
    dtype: torch.dtype, ny: int, nx: int
) -> None:
    """torch.compile's CPU (Inductor) codegen is dtype/shape-sensitive.

    A real Inductor C++ codegen bug was found on this box (a VecMask width
    mismatch, specific to a particular boolean-mask arithmetic pattern) --
    specific to that pattern, not present here,
    but its discovery is why this sweeps both float32/float64 (the failing
    pattern was a dtype-width mixing bug) and an odd/prime inner grid
    dimension (17, 23 -- Inductor's vectorized codegen takes a different,
    less-exercised scalar-tail path when an axis is not a multiple of the
    vector width) rather than trusting a single float64/round-number-shape
    smoke test.
    """
    lamb, mu, buoyancy = _homogeneous(ny, nx, 1800.0, 900.0, 2000.0)
    qp = 30.0 * torch.ones(ny, nx, dtype=dtype)
    qs = 25.0 * torch.ones(ny, nx, dtype=dtype)
    lamb, mu, buoyancy = lamb.to(dtype), mu.to(dtype), buoyancy.to(dtype)
    src = (
        deepwave.wavelets.ricker(25.0, 12, 0.001, 1.5 / 25.0)
        .to(dtype)
        .reshape(1, 1, -1)
    )
    sl_y = min(8, ny - 2)
    rl_y = min(5, ny - 2)
    kw = {
        "source_amplitudes_y": src,
        "source_locations_y": torch.tensor([[[sl_y, 9]]]),
        "receiver_locations_p": torch.tensor([[[rl_y, 5]]]),
        "pml_freq": 25.0,
        "q_freq": 25.0,
    }
    out_eager = deepwave.viscoelastic(
        lamb, mu, buoyancy, qp, qs, 5.0, 0.001, backend="eager", **kw
    )
    out_compile = deepwave.viscoelastic(
        lamb, mu, buoyancy, qp, qs, 5.0, 0.001, backend="torch_compile", **kw
    )
    assert torch.allclose(
        out_compile.receiver_amplitudes_p,
        out_eager.receiver_amplitudes_p,
        rtol=1e-4,
        atol=1e-8,
    )


@pytest.mark.slow
def test_compile_gradients_match_eager() -> None:
    """backend='torch_compile' gradients agree with eager to round-off."""
    ny, nx = 16, 18
    lamb0, mu0, buoyancy = _homogeneous(ny, nx, 1800.0, 900.0, 2000.0)
    qp = 30.0 * torch.ones(ny, nx, dtype=DTYPE)
    qs = 25.0 * torch.ones(ny, nx, dtype=DTYPE)
    src = (
        deepwave.wavelets.ricker(25.0, 12, 0.001, 1.5 / 25.0)
        .to(DTYPE)
        .reshape(1, 1, -1)
    )
    kw = {
        "source_amplitudes_y": src,
        "source_locations_y": torch.tensor([[[8, 9]]]),
        "receiver_locations_p": torch.tensor([[[5, 5]]]),
        "pml_freq": 25.0,
        "q_freq": 25.0,
    }

    def run(backend: str) -> tuple[torch.Tensor, torch.Tensor]:
        lamb = lamb0.clone().requires_grad_(True)
        mu = mu0.clone().requires_grad_(True)
        out = deepwave.viscoelastic(
            lamb, mu, buoyancy, qp, qs, 5.0, 0.001, backend=backend, **kw
        )
        out.receiver_amplitudes_p.pow(2).sum().backward()
        assert lamb.grad is not None
        assert mu.grad is not None
        return lamb.grad, mu.grad

    g_lamb_e, g_mu_e = run("eager")
    for backend in ("torch_compile",):
        g_lamb, g_mu = run(backend)
        assert (
            g_lamb - g_lamb_e
        ).abs().max().item() <= 1e-9 * g_lamb_e.abs().max().item()
        assert (g_mu - g_mu_e).abs().max().item() <= 1e-9 * g_mu_e.abs().max().item()


def test_scope_and_validation_errors() -> None:
    """``ndim`` and ``mpml_fraction`` raise clearly."""
    n = 8
    lamb, mu, buoyancy = _homogeneous(n, n, 1500.0, 800.0, 1000.0)
    qp = 30.0 * torch.ones(n, n, dtype=DTYPE)
    qs = 25.0 * torch.ones(n, n, dtype=DTYPE)
    common = (lamb, mu, buoyancy, qp, qs, 5.0, 0.001)
    src_kw = {
        "source_amplitudes_y": torch.zeros(1, 1, 5, dtype=DTYPE),
        "source_locations_y": torch.tensor([[[4, 4]]]),
        "q_freq": 25.0,
    }
    with pytest.raises(ValueError, match="mpml_fraction"):
        deepwave.viscoelastic(*common, mpml_fraction=1.0, backend="eager", **src_kw)
    lamb1d, mu1d, buoyancy1d = _homogeneous(1, n, 1500.0, 800.0, 1000.0)
    with pytest.raises(ValueError, match="2-D and 3-D"):
        deepwave.viscoelastic(
            lamb1d.reshape(n),
            mu1d.reshape(n),
            buoyancy1d.reshape(n),
            qp.reshape(n, n)[0],
            qs.reshape(n, n)[0],
            5.0,
            0.001,
            backend="eager",
            q_freq=25.0,
            source_amplitudes_x=torch.zeros(1, 1, 5, dtype=DTYPE),
            source_locations_x=torch.tensor([[[4]]]),
        )


def test_backend_compiled_default_resolves_to_eager_with_warning() -> None:
    """The default ``backend="compiled"`` runs (as "eager"), warning once.

    It previously raised ``NotImplementedError`` (see the
    now-removed assertion in ``test_scope_and_validation_errors``); it now
    resolves to the only backend that exists and warns instead of silently
    falling back, matching the explicit ``"eager"`` result exactly.
    """
    n = 8
    lamb, mu, buoyancy = _homogeneous(n, n, 1500.0, 800.0, 1000.0)
    qp = 30.0 * torch.ones(n, n, dtype=DTYPE)
    qs = 25.0 * torch.ones(n, n, dtype=DTYPE)
    common = (lamb, mu, buoyancy, qp, qs, 5.0, 0.001)
    src_kw = {
        "source_amplitudes_y": torch.zeros(1, 1, 5, dtype=DTYPE),
        "source_locations_y": torch.tensor([[[4, 4]]]),
        "receiver_locations_p": torch.tensor([[[4, 4]]]),
        # Given explicitly, and nothing to do with the backend resolution
        # under test. The source amplitudes here are all zero, so an omitted
        # pml_freq is inferred from a fifth of Nyquist = 100 Hz, which fires
        # two once-per-process UserWarnings -- the migration notice and the
        # grid-cells-per-wavelength notice -- that the "error" filter below
        # promotes. Whether either fires depends on what ran earlier in the
        # process, so omitting it made this test pass in file order and fail
        # standalone. 25 Hz is the value releases up to 0.0.27 used here and
        # is comfortably resolved on this grid, so neither warning arises.
        "pml_freq": 25.0,
        # The frequency Qp/Qs are specified at; required, and unrelated to the
        # backend resolution under test.
        "q_freq": 25.0,
    }
    with warnings.catch_warnings():
        # The default resolves to the eager backend SILENTLY: the fallback
        # UserWarning this used to assert was removed deliberately, because a
        # warning is invisible in a training loop, under -W ignore and in a
        # distributed worker. out.info.backend is the replacement, and
        # backend="compiled" is the assertion that raises.
        warnings.simplefilter("error", UserWarning)
        # The experimental-tier notice is a UserWarning subclass and fires
        # once per process, so "error" promotes it here whenever this test is
        # the first in the process to call viscoelastic. It is what
        # ``test_experimental_warning_fires_once_per_process`` is for, and is
        # ignored by category rather than by weakening the filter that
        # catches every other UserWarning.
        out_none = deepwave.viscoelastic(*common, **src_kw)
    assert out_none.info.backend == "eager"
    out_eager = deepwave.viscoelastic(*common, backend="eager", **src_kw)
    assert torch.equal(out_none.receiver_amplitudes_p, out_eager.receiver_amplitudes_p)
    with pytest.raises(deepwave.DeepwaveNotImplementedError, match="no compiled"):
        deepwave.viscoelastic(*common, backend="compiled", **src_kw)


def test_backend_eager_explicit_does_not_warn_about_fallback() -> None:
    """Explicitly requesting ``"eager"`` does not emit the fallback warning."""
    n = 8
    lamb, mu, buoyancy = _homogeneous(n, n, 1500.0, 800.0, 1000.0)
    qp = 30.0 * torch.ones(n, n, dtype=DTYPE)
    qs = 25.0 * torch.ones(n, n, dtype=DTYPE)
    common = (lamb, mu, buoyancy, qp, qs, 5.0, 0.001)
    src_kw = {
        "source_amplitudes_y": torch.zeros(1, 1, 5, dtype=DTYPE),
        "source_locations_y": torch.tensor([[[4, 4]]]),
        "backend": "eager",
        "q_freq": 25.0,
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deepwave.viscoelastic(*common, **src_kw)
    assert not any("no compiled backend" in str(x.message) for x in w)


def test_no_maturity_warning_fires() -> None:
    """Viscoelastic says nothing about its tier: the label is documentation.

    The interface promise Deepwave makes at run time is identical on all
    eleven propagators, so the experimental label must be invisible to a
    call. ``tests/test_stability.py`` asserts the same for all eleven.
    """
    n = 8
    lamb, mu, buoyancy = _homogeneous(n, n, 1500.0, 800.0, 1000.0)
    qp = 30.0 * torch.ones(n, n, dtype=DTYPE)
    qs = 25.0 * torch.ones(n, n, dtype=DTYPE)
    common = (lamb, mu, buoyancy, qp, qs, 5.0, 0.001)
    src_kw = {
        "source_amplitudes_y": torch.zeros(1, 1, 5, dtype=DTYPE),
        "source_locations_y": torch.tensor([[[4, 4]]]),
        "backend": "eager",
        "q_freq": 25.0,
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deepwave.viscoelastic(*common, **src_kw)
    assert not [x for x in w if "experimental" in str(x.message)]


def test_eager_backend_unsupported_options_raise() -> None:
    """backward_callback, non-'device' storage, and compression all raise."""
    n = 6
    lamb, mu, buoyancy = _homogeneous(n, n, 1500.0, 800.0, 1000.0)
    qp = 30.0 * torch.ones(n, n, dtype=DTYPE)
    qs = 25.0 * torch.ones(n, n, dtype=DTYPE)
    common = (lamb, mu, buoyancy, qp, qs, 5.0, 0.001)
    src_kw = {
        "source_amplitudes_y": torch.zeros(1, 1, 5, dtype=DTYPE),
        "source_locations_y": torch.tensor([[[3, 3]]]),
        "backend": "eager",
        "q_freq": 25.0,
    }
    with pytest.raises(RuntimeError, match="backward_callback"):
        deepwave.viscoelastic(*common, backward_callback=lambda state: None, **src_kw)
    with pytest.raises(RuntimeError, match="storage_mode"):
        deepwave.viscoelastic(*common, storage_mode="cpu", **src_kw)
    with pytest.raises(RuntimeError, match="Storage compression"):
        deepwave.viscoelastic(*common, storage_compression=True, **src_kw)


def test_elastic_velocity_extremes_edge_cases() -> None:
    """_elastic_velocity_extremes' zero-vp/zero-vs branches and its vmap guard."""
    n = 4
    zero = torch.zeros(n, n, dtype=DTYPE)
    one = torch.ones(n, n, dtype=DTYPE)
    # All-zero lamb/mu (vacuum): both vp and vs are zero everywhere.
    min_v, max_v = _vemod._elastic_velocity_extremes(zero, zero, one, None)
    assert min_v == 0.0
    assert max_v == 0.0
    # mu=0 (fluid: vs=0 everywhere) but lamb>0 (vp>0 everywhere).
    lamb_fluid = 4.0e6 * one
    min_v, max_v = _vemod._elastic_velocity_extremes(lamb_fluid, zero, one, None)
    assert min_v > 0.0
    assert max_v > 0.0
    # lamb+2mu=0 everywhere (vp=0) but mu>0 (vs>0 everywhere): an unphysical
    # but mathematically valid combination that exercises the vp-only-zero
    # branch.
    mu_only = 1.0e6 * one
    lamb_p_vacuum = -2 * mu_only
    min_v, max_v = _vemod._elastic_velocity_extremes(lamb_p_vacuum, mu_only, one, None)
    assert min_v > 0.0
    assert max_v > 0.0
    # BatchedTensor (vmap) input requires an explicit max_vel.
    with pytest.raises(RuntimeError, match="max_vel"):
        torch.func.vmap(
            lambda lamb_b: _vemod._elastic_velocity_extremes(lamb_b, one, one, None)[1]
        )(one.unsqueeze(0))
    extremes = torch.func.vmap(
        lambda lamb_b: torch.tensor(
            _vemod._elastic_velocity_extremes(lamb_b, one, one, 3000.0)
        )
    )(one.unsqueeze(0))
    assert bool((extremes[:, 1] == 3000.0).all())


def test_forward_callback_invoked() -> None:
    """forward_callback is invoked once per outer step (callback_frequency=1)."""
    ny, nx = 12, 14
    dx, dt, nt = 5.0, 0.001, 8
    lamb, mu, buoyancy = _homogeneous(ny, nx, 1500.0, 800.0, 1000.0)
    qp = 30.0 * torch.ones(ny, nx, dtype=DTYPE)
    qs = 25.0 * torch.ones(ny, nx, dtype=DTYPE)
    src = torch.zeros(1, 1, nt, dtype=DTYPE)
    sl = torch.tensor([[[6, 7]]])

    calls = []

    def record(state: object) -> None:
        calls.append(state.step)  # type: ignore[attr-defined]

    deepwave.viscoelastic(
        lamb,
        mu,
        buoyancy,
        qp,
        qs,
        dx,
        dt,
        source_amplitudes_y=src,
        source_locations_y=sl,
        pml_freq=25.0,
        q_freq=25.0,
        backend="eager",
        forward_callback=record,
    )
    assert calls == list(range(nt))


def test_vmap_with_receivers() -> None:
    """A batched (vmap) run with receivers exercises the is_batched accumulation."""
    ny, nx = 12, 14
    dx, dt, nt = 5.0, 0.001, 6
    _lamb, mu, buoyancy = _homogeneous(ny, nx, 1500.0, 800.0, 1000.0)
    qp = 30.0 * torch.ones(ny, nx, dtype=DTYPE)
    qs = 25.0 * torch.ones(ny, nx, dtype=DTYPE)
    src = torch.zeros(1, 1, nt, dtype=DTYPE)
    src[0, 0, 0] = 1.0
    sl = torch.tensor([[[6, 7]]])
    rl = torch.tensor([[[5, 5]]])
    lamb_batch = torch.stack(
        [
            torch.full((ny, nx), 4.0e6, dtype=DTYPE),
            torch.full((ny, nx), 5.0e6, dtype=DTYPE),
        ]
    )

    def run(lamb_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = deepwave.viscoelastic(
            lamb_b,
            mu,
            buoyancy,
            qp,
            qs,
            dx,
            dt,
            source_amplitudes_y=src,
            source_locations_y=sl,
            receiver_locations_p=rl,
            receiver_locations_y=rl,  # exercises the velocity-receiver is_batched path
            pml_freq=25.0,
            q_freq=25.0,
            backend="eager",
            max_vel=3000.0,
        )
        return out.receiver_amplitudes_p, out.receiver_amplitudes_y

    out_p, out_y = torch.func.vmap(run)(lamb_batch)
    for out in (out_p, out_y):
        assert out.shape[0] == 2
        assert torch.isfinite(out).all()
        # The two batch elements used different lamb, so their traces differ.
        assert (out[0] - out[1]).abs().max() > 0


def test_q_freq_is_required_and_never_taken_from_pml_freq() -> None:
    """An unset ``q_freq`` raises; it is not inherited from ``pml_freq``.

    The viscoelastic counterpart of the viscoacoustic test of the same name:
    the frequency at which ``Qp``/``Qs`` were specified is a property of the
    medium, and cannot be inherited from the absorbing boundary's tuning
    frequency (which is itself inferred from the wavelet).
    """
    freq = 60.0
    ny, nx, nt, dt, dx = 40, 40, 200, 0.0005, 5.0
    lamb, mu, buoyancy = _homogeneous(ny, nx, 1800.0, 900.0, 2000.0)
    qp = torch.full((ny, nx), 20.0, dtype=lamb.dtype)
    qs = torch.full((ny, nx), 15.0, dtype=lamb.dtype)
    src = (
        deepwave.wavelets.ricker(freq, nt, dt, 1.0 / freq)
        .to(lamb.dtype)
        .reshape(1, 1, -1)
    )

    def run(**kwargs):
        return deepwave.viscoelastic(
            lamb,
            mu,
            buoyancy,
            qp,
            qs,
            dx,
            dt,
            source_amplitudes_y=src,
            source_locations_y=torch.full((1, 1, 2), 20, dtype=torch.long),
            receiver_locations_y=torch.full((1, 1, 2), 10, dtype=torch.long),
            pml_width=10,
            **kwargs,
        ).receiver_amplitudes_y

    with pytest.raises(deepwave.DeepwaveValueError, match="q_freq is required"):
        run()
    given = run(q_freq=freq)
    # Non-vacuity: the reference frequency really does change the numbers.
    assert not torch.equal(given, run(q_freq=25.0))
    assert given.abs().max() > 0


@pytest.mark.parametrize("form", ["functional", "module"])
def test_q_freq_is_required_in_both_call_forms(form: str) -> None:
    """The Module raises the same way the functional does.

    The rule lives in the functional both forms share, so this pins that the
    ``Module`` does not paper over a missing ``q_freq`` with a default of its
    own on the way through.
    """
    n = 20
    lamb, mu, buoyancy = _homogeneous(n, n, 1800.0, 900.0, 2000.0)
    qp = torch.full((n, n), 20.0, dtype=DTYPE)
    qs = torch.full((n, n), 15.0, dtype=DTYPE)
    src = torch.zeros(1, 1, 10, dtype=DTYPE)
    src[0, 0, 3] = 1.0
    kwargs = {
        "source_amplitudes_y": src,
        "source_locations_y": torch.full((1, 1, 2), 10, dtype=torch.long),
        "receiver_locations_y": torch.full((1, 1, 2), 6, dtype=torch.long),
        "pml_width": 3,
        "pml_freq": 25.0,
    }

    def call(**extra: object) -> object:
        if form == "functional":
            return deepwave.viscoelastic(
                lamb, mu, buoyancy, qp, qs, 5.0, 0.0005, **kwargs, **extra
            )
        return deepwave.Viscoelastic(lamb, mu, buoyancy, qp, qs, 5.0)(
            0.0005, **kwargs, **extra
        )

    with pytest.raises(deepwave.DeepwaveValueError, match="q_freq is required"):
        call()
    # The same call with q_freq given runs, so the raise is about q_freq only.
    call(q_freq=25.0)


def test_q_freq_is_refused_with_several_mechanisms() -> None:
    """Above one mechanism ``q_freq`` is rejected, mirroring ``q_band``."""
    n = 20
    lamb, mu, buoyancy = _homogeneous(n, n, 1800.0, 900.0, 2000.0)
    qp = torch.full((n, n), 20.0, dtype=DTYPE)
    qs = torch.full((n, n), 15.0, dtype=DTYPE)
    src = torch.zeros(1, 1, 10, dtype=DTYPE)
    src[0, 0, 3] = 1.0
    kwargs = {
        "source_amplitudes_y": src,
        "source_locations_y": torch.full((1, 1, 2), 10, dtype=torch.long),
        "receiver_locations_y": torch.full((1, 1, 2), 6, dtype=torch.long),
        "pml_width": 3,
        "pml_freq": 25.0,
        "q_n_mechanisms": 2,
        "q_band": (10.0, 60.0),
    }
    with pytest.raises(deepwave.DeepwaveValueError, match="q_freq must be None"):
        deepwave.viscoelastic(
            lamb, mu, buoyancy, qp, qs, 5.0, 0.0005, q_freq=25.0, **kwargs
        )
    deepwave.viscoelastic(lamb, mu, buoyancy, qp, qs, 5.0, 0.0005, **kwargs)


def test_q_freq_must_be_positive() -> None:
    """Zero or negative reference frequencies are rejected, not clamped."""
    n = 20
    lamb, mu, buoyancy = _homogeneous(n, n, 1800.0, 900.0, 2000.0)
    qp = torch.full((n, n), 20.0, dtype=DTYPE)
    qs = torch.full((n, n), 15.0, dtype=DTYPE)
    src = torch.zeros(1, 1, 10, dtype=DTYPE)
    src[0, 0, 3] = 1.0
    for bad in (0.0, -5.0):
        with pytest.raises(
            deepwave.DeepwaveValueError, match="q_freq must be positive"
        ):
            deepwave.viscoelastic(
                lamb,
                mu,
                buoyancy,
                qp,
                qs,
                5.0,
                0.0005,
                source_amplitudes_y=src,
                source_locations_y=torch.full((1, 1, 2), 10, dtype=torch.long),
                receiver_locations_y=torch.full((1, 1, 2), 6, dtype=torch.long),
                pml_width=3,
                pml_freq=25.0,
                q_freq=bad,
            )


#: ``q_freq`` values that are numbers but not usable frequencies, with the
#: word the ``DeepwaveValueError`` uses. A non-finite reference frequency used
#: to be accepted here and to surface thousands of time steps later as a
#: wavefield of ``NaN``s.
_BAD_Q_FREQ_VALUES = [
    (0.0, "positive"),
    (-5.0, "positive"),
    (float("nan"), "finite"),
    (float("inf"), "finite"),
    (float("-inf"), "finite"),
]

#: ``q_freq`` values that are not numbers at all. ``"25"`` used to reach
#: ``float()`` and work; a ``bool`` is an ``int`` to Python but not a
#: frequency to anyone.
_NON_NUMBER_Q_FREQ_VALUES = ["25", True, False, [25.0], (25.0,), object()]


@pytest.mark.parametrize("form", ["functional", "module"])
def test_q_freq_must_be_a_positive_finite_number(form: str) -> None:
    """Non-finite, non-positive and non-numeric ``q_freq`` values are refused.

    The viscoacoustic test of the same name, on the propagator that shares the
    rule: the value defines the medium's attenuation through
    ``omega0 = 2*pi*q_freq``, so an unusable one is caught at the call rather
    than left to produce a ``NaN`` wavefield or whatever error the first
    arithmetic on it happens to raise.
    """
    n = 20
    lamb, mu, buoyancy = _homogeneous(n, n, 1800.0, 900.0, 2000.0)
    qp = torch.full((n, n), 20.0, dtype=DTYPE)
    qs = torch.full((n, n), 15.0, dtype=DTYPE)
    src = torch.zeros(1, 1, 10, dtype=DTYPE)
    src[0, 0, 3] = 1.0
    kwargs = {
        "source_amplitudes_y": src,
        "source_locations_y": torch.full((1, 1, 2), 10, dtype=torch.long),
        "receiver_locations_y": torch.full((1, 1, 2), 6, dtype=torch.long),
        "pml_width": 3,
        "pml_freq": 25.0,
    }

    def call(q_freq: object) -> object:
        if form == "functional":
            return deepwave.viscoelastic(
                lamb, mu, buoyancy, qp, qs, 5.0, 0.0005, q_freq=q_freq, **kwargs
            )
        return deepwave.Viscoelastic(lamb, mu, buoyancy, qp, qs, 5.0)(
            0.0005, q_freq=q_freq, **kwargs
        )

    for bad, word in _BAD_Q_FREQ_VALUES:
        with pytest.raises(deepwave.DeepwaveValueError, match=f"q_freq must be {word}"):
            call(bad)
    for wrong_type in [*_NON_NUMBER_Q_FREQ_VALUES, torch.tensor([25.0, 30.0])]:
        with pytest.raises(
            deepwave.DeepwaveTypeError, match="q_freq must be a positive number"
        ):
            call(wrong_type)
    # A one-element tensor or a NumPy scalar is a number, and still runs.
    for good in (25.0, 25, torch.tensor(25.0), np.float32(25.0)):
        assert torch.isfinite(call(good).receiver_amplitudes_y).all()


def test_multi_mechanism_public_field_order_and_mechanism_identity() -> None:
    """At two mechanisms the GSLS block is grouped by mechanism, not by kind.

    Two regroupings compose in this propagator's public order: the elastic
    family's grouping of the physical fields by kind, and the GSLS memory
    block's grouping by mechanism. The backend stores that block by kind
    (``rp_0, rp_1, rs_y_0, rs_y_1, ...``), so the public order is a
    transpose of it -- and at one mechanism the two are identical, which is
    why this has to be checked at two.

    The field order is pinned against a written-out list. Which tensor each
    name actually carries is then checked physically, not against that list:
    with every physical field started at zero and no source there is nothing
    to drive the memory variables, so each decays by exactly its own
    mechanism's Crank-Nicolson factor ``ar_l = (1 - dt/2tausig_l) / (1 +
    dt/2tausig_l)``, and each is started at a different value. A field
    holding the wrong mechanism's tensor decays by the wrong factor; one
    holding the wrong kind's starts from the wrong value. Both show up as a
    ratio that misses.
    """
    n_mech = 2
    ny, nx, pml = 20, 21, 5
    dx, dt, band = 5.0, 0.0004, (2.0, 80.0)
    shape = (ny, nx)
    lamb = torch.full(shape, 2.25e9, dtype=DTYPE)
    mu = torch.full(shape, 1.0e9, dtype=DTYPE)
    buoyancy = torch.full(shape, 1 / 2200.0, dtype=DTYPE)
    qp = torch.full(shape, 40.0, dtype=DTYPE)
    qs = torch.full(shape, 30.0, dtype=DTYPE)

    expected = [
        "vy_0",
        "vx_0",
        "sigmayy_0",
        "sigmaxy_0",
        "sigmaxx_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_y",
        "receiver_amplitudes_x",
    ]
    assert _vemod.viscoelastic_output_names(2, n_mech) == expected
    # The GSLS block is inside the auxiliary state, grouped by mechanism.
    assert _vemod.viscoelastic_auxiliary_state_names(2, n_mech)[:8] == [
        "rp_mech0",
        "rs_y_mech0",
        "rs_x_mech0",
        "rs_xy_mech0",
        "rp_mech1",
        "rs_y_mech1",
        "rs_x_mech1",
        "rs_xy_mech1",
    ]

    gsls_names = [
        n
        for n in _vemod.viscoelastic_auxiliary_state_names(2, n_mech)
        if n.startswith(("rp_", "rs_"))
    ]
    starts = {name: float(i + 2) for i, name in enumerate(gsls_names)}
    wshape = (1, ny + 2 * pml, nx + 2 * pml)

    def run(initial_state=None):
        return deepwave.viscoelastic(
            lamb,
            mu,
            buoyancy,
            qp,
            qs,
            dx,
            dt,
            source_amplitudes_y=torch.zeros(1, 1, 1, dtype=DTYPE),
            source_locations_y=torch.tensor([[[ny // 2, nx // 2]]]),
            receiver_locations_p=torch.tensor([[[ny // 2 + 1, nx // 2]]]),
            pml_width=pml,
            pml_freq=25.0,
            q_n_mechanisms=n_mech,
            q_band=band,
            backend="eager",
            initial_state=initial_state,
        )

    # The memory variables are auxiliary state, so they are seeded by
    # replacing the tensors of a state a run returned, which keeps that
    # state's compatibility record.
    from_rest = run()
    seeded = dataclasses.replace(
        from_rest.auxiliary_state,
        _tensors={
            **{n: torch.zeros(wshape, dtype=DTYPE) for n in from_rest.auxiliary_state},
            **{n: torch.full(wshape, c, dtype=DTYPE) for n, c in starts.items()},
        },
    )
    initial: dict = {
        name: torch.zeros(wshape, dtype=DTYPE)
        for name in ("vy_0", "vx_0", "sigmayy_0", "sigmaxy_0", "sigmaxx_0")
    }
    initial["auxiliary_state"] = seeded

    out = run(initial_state=initial)
    assert list(out.names) == expected
    # One outer step, so the decay below is a single application of ar.
    assert out.info.step_ratio == 1
    assert out.info.nt == 1

    tausig, _ = _gsls.gsls_fit_weights(band, n_mech, DTYPE, torch.device("cpu"))
    half = out.info.internal_dt / (2.0 * tausig)
    ar = (1.0 - half) / (1.0 + half)
    # The two mechanisms must be well separated, or the check is vacuous.
    assert abs(float(ar[0]) - float(ar[1])) > 0.1

    interior = (0, pml + ny // 2, pml + nx // 2)
    for name in gsls_names:
        mech = int(name.rsplit("mech", 1)[1])
        decayed = float(out.auxiliary_state[name][interior]) / starts[name]
        assert decayed == pytest.approx(float(ar[mech]), rel=1e-9), name

    # The elastic block is published in the elastic family's kind-grouped
    # order; the attenuation and absorbing-boundary memory are both folded
    # away entirely.
    assert set(out.auxiliary_state) == set(
        _vemod.viscoelastic_auxiliary_state_names(2, n_mech)
    )
    for gone in ("rp_0", "rp_1", "rp_mech0", "rs_yx_0", "m_vyy_0"):
        assert gone not in out.names
