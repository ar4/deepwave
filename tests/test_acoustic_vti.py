"""Tests for the pseudo-acoustic VTI propagator (deepwave.acoustic_vti).

The validation strategy is:

1. **The elastic_vti oracle** (the strongest test): acoustic_vti IS
   ``elastic_vti(c11, c13, c33, c55=0(, c66=0), buoyancy)``, so their forwards
   agree to machine precision (``sigma_v == sigmazz``,
   ``sigma_h == sigmaxx == sigmayy``, velocities equal) and their model/source
   gradients agree under a matched velocity-receiver loss, in 2-D and 3-D. This
   pins the constitutive coupling, the axis convention, and (in 3-D) the
   ``sigma_h`` duplication onto both horizontal stresses. One case is chained
   through ``thomsen_to_cij(vp, vs=0, ...)``.
2. **Isotropic limit** ``eps = delta = 0`` (``c11 = c13 = c33``): reduces to the
   variable-density acoustic (velocity-pressure) system and matches
   ``deepwave.acoustic``.
3. **Analytic Christoffel** phase velocities for the ``V_S = 0`` qP surface in a
   homogeneous strongly-anisotropic medium (slowness-decomposition of a delayed
   line source), plus Grechka stability, reciprocity, PML absorption, an
   absolute finite-difference gradcheck through the Thomsen converters, the
   pseudo-shear source taper, and the compiled-backend guard.
"""

import importlib
import math
import warnings

import pytest
import torch
from helpers import c_backend_available, element_tensors, output_tensors

import deepwave
import deepwave.common

# ``deepwave.acoustic_vti`` (the top-level package attribute) is the
# re-exported *function*, not the submodule; ``import deepwave.acoustic_vti as
# x`` would resolve to that same function (it reads the now-overwritten
# package attribute, not sys.modules directly), so reach the submodule's
# non-re-exported helpers (e.g. output names) via importlib instead.
acoustic_vti_module = importlib.import_module("deepwave.acoustic_vti")

DEVICE = torch.device("cpu")
DTYPE = torch.float64

C_BACKEND = pytest.mark.skipif(
    not c_backend_available("acoustic_vti", device=DEVICE),
    reason="compiled acoustic_vti backend not present in libdeepwave_C",
)

# Backends to run the (task 7.G) physics/gradient battery on (task 7.H
# milestone H3): the eager Python reference and, when built, the compiled
# C backend (backend="compiled"). Mirrors test_elastic_vti.py's BACKENDS.
BACKENDS = [
    pytest.param("eager", id="eager"),
    pytest.param("compiled", id="compiled", marks=C_BACKEND),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_media(ndim, n=12, eps=0.2, delta=0.08, batch=False):
    """Heterogeneous, genuinely anisotropic (eps > delta) VTI stiffnesses."""
    torch.manual_seed(1)
    shape = (n,) * ndim
    if batch:
        shape = (2, *shape)
    vp = 1500 + 300 * torch.rand(*shape, dtype=DTYPE)
    rho = 2000 + 300 * torch.rand(*shape, dtype=DTYPE)
    c33 = rho * vp**2
    c11 = c33 * (1 + 2 * eps)
    c13 = c33 * math.sqrt(1 + 2 * delta)
    buoyancy = 1 / rho
    return c11.detach(), c13.detach(), c33.detach(), buoyancy.detach()


def _survey(ndim, nt, n=12, n_shots=1, dt=0.001, stress_source=False):
    """Velocity (and optionally stress) sources; velocity + stress receivers."""
    torch.manual_seed(2)
    freq = 15.0
    amp = (
        deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq)
        .to(DTYPE)
        .reshape(1, 1, -1)
        .repeat(n_shots, 1, 1)
    )
    src = torch.tensor([[[n // 2] * ndim]], dtype=torch.long).repeat(n_shots, 1, 1)
    rec = torch.tensor([[[n // 2 - 2] * ndim]], dtype=torch.long).repeat(n_shots, 1, 1)
    kwargs = {
        "dt": dt,
        "source_amplitudes_x": amp,
        "source_locations_x": src,
        "receiver_locations_x": rec,
        "receiver_locations_y": rec,
        "pml_width": 4,
        "accuracy": 4,
        "max_vel": 3500.0,
        "pml_freq": freq,
    }
    if ndim >= 3:
        kwargs["receiver_locations_z"] = rec
    if stress_source:
        kwargs["source_amplitudes_sigma_h"] = amp
        kwargs["source_locations_sigma_h"] = src
        kwargs["source_amplitudes_sigma_v"] = amp
        kwargs["source_locations_sigma_v"] = src
    return kwargs, amp, src, rec


def _velocity_receivers(out, ndim):
    """The velocity receiver amplitudes in a fixed order, from a dataclass out."""
    recs = [out.receiver_amplitudes_y, out.receiver_amplitudes_x]
    if ndim >= 3:
        recs = [out.receiver_amplitudes_z, *recs]
    return recs


# ---------------------------------------------------------------------------
# The elastic_vti oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("nt", [24, 25])  # even and odd
@pytest.mark.parametrize("stress_source", [False, True])
def test_oracle_forward(ndim, nt, stress_source, backend):
    """Forward is bit-identical to elastic_vti with c55 = c66 = 0.

    The two normal stresses map as sigma_v == sigmazz and
    sigma_h == sigmaxx == sigmayy, and the velocities are equal. An explosive
    (sigma_h and sigma_v, same wavelet) acoustic_vti source equals a pressure
    source of the negated amplitude in elastic_vti (both add +a*dt to every
    normal stress).

    Parametrized over ``backend`` (task 7.H milestone H3): this is the
    strongest available physics oracle, so re-running it against the
    compiled backend is an independent check of the C/CUDA forward beyond
    ``test_compiled_forward_matches_eager``'s direct eager-vs-compiled
    parity (here the reference is a *different* propagator, elastic_vti's
    own eager path).
    """
    c11, c13, c33, buoyancy = _make_media(ndim)
    kwargs, amp, src, _ = _survey(ndim, nt)
    av = deepwave.acoustic_vti(
        c11,
        c13,
        c33,
        buoyancy,
        5.0,
        backend=backend,
        **kwargs,
    )
    zero = torch.zeros_like(c11)
    ekwargs = {**kwargs}
    if stress_source:
        # Move the explosive source over to elastic_vti's pressure channel.
        del ekwargs["source_amplitudes_x"], ekwargs["source_locations_x"]
        for k in list(kwargs):
            if k.startswith(("source_amplitudes_sigma", "source_locations_sigma")):
                del ekwargs[k]
        av = deepwave.acoustic_vti(
            c11,
            c13,
            c33,
            buoyancy,
            5.0,
            source_amplitudes_sigma_h=amp,
            source_amplitudes_sigma_v=amp,
            source_locations_sigma_h=src,
            source_locations_sigma_v=src,
            backend=backend,
            **ekwargs,
        )
        ev = deepwave.elastic_vti(
            c11,
            c13,
            c33,
            zero,
            buoyancy,
            5.0,
            c66=zero if ndim == 3 else None,
            source_amplitudes_pressure_rate=-amp,
            source_locations_pressure_rate=src,
            **ekwargs,
        )
    else:
        ev = deepwave.elastic_vti(
            c11,
            c13,
            c33,
            zero,
            buoyancy,
            5.0,
            c66=zero if ndim == 3 else None,
            **kwargs,
        )

    pairs = [
        (av.sigma_v_0, ev.sigmazz_0 if ndim == 3 else ev.sigmayy_0),
        (av.sigma_h_0, ev.sigmaxx_0),
        (av.sigma_h_0, ev.sigmayy_0 if ndim == 3 else ev.sigmaxx_0),
        (av.vy_0, ev.vy_0),
        (av.vx_0, ev.vx_0),
        (av.receiver_amplitudes_x, ev.receiver_amplitudes_x),
        (av.receiver_amplitudes_y, ev.receiver_amplitudes_y),
    ]
    if ndim == 3:
        pairs += [
            (av.vz_0, ev.vz_0),
            (av.receiver_amplitudes_z, ev.receiver_amplitudes_z),
        ]
    for a, b in pairs:
        scale = max(a.abs().max().item(), 1e-30)
        assert (a - b).abs().max().item() <= scale * 1e-12


# NOTE ON THE GRADIENT ORACLE. The derivation's section 6.3 proposes checking
# gradients against ``elastic_vti(c55=c66=0)`` too. That reference is NOT usable
# for a machine-precision *gradient* check: elastic_vti's eager autograd gradient
# is unreliable at exactly c55 = 0 for genuinely anisotropic media (c11 != c33) --
# it disagrees with the finite-difference gradient of its own (bit-identical)
# forward, while it is correct once c55 > 0 and at the isotropic limit c11 = c33.
# (Reported to the maintainer as an elastic_vti finding. Task 7.J re-investigated
# on 2026-07-12: five independent reproduction attempts -- eager-vs-compiled and
# eager-vs-directional-FD sweeps over ndim/accuracy/c55, at exactly c55 = 0 with
# genuinely anisotropic media -- all agreed to ~1e-15/FD-floor, so the report is
# treated as an unreproduced methodology artifact of the original ad hoc script;
# this file's self-contained FD validation below stands either way.) The eager
# acoustic_vti
# backend has no hand-coded adjoint -- gradients come from native autograd, so the
# 3-D ``sigma_h = sigmaxx + sigmayy`` summation hazard cannot arise here -- so the
# gradients are instead validated directly and self-containedly against finite
# differences below (which is the ground truth), in 2-D and 3-D. The forward
# oracle above already pins the 3-D duplication (sigma_h == sigmaxx == sigmayy).


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("ndim", [2, 3])
def test_gradients_match_finite_difference(ndim, backend):
    """Every model and the source gradient match a converged directional FD.

    A scale-robust check: a random direction in each input is projected onto the
    autograd gradient and compared with a central finite difference of the same
    scalar loss along that direction (which is exact for the ~1e-9 relative
    agreement asserted). This validates the full 3-D gradient (including the
    horizontal/axial coupling that the isotropic limit cannot exercise) without
    relying on any second propagator.

    Parametrized over ``backend`` (task 7.H milestone H3): at ``backend="compiled"``
    this validates the compiled hand-adjoint directly against finite
    differences -- an oracle independent of ``test_compiled_backward_matches_
    eager``'s eager-vs-compiled parity check. ``dt`` is deliberately
    courant-safe (see ``_COURANT_SAFE_DT``'s module comment on the
    2-D-vs-3-D CFL trap) so ``step_ratio == 1``, making the compiled model
    gradients exact rather than merely bounded -- required for the tight
    ``rel < 1e-5`` tolerance asserted below.

    ``nt=120`` (previously 16, with a mismatched ``_survey`` call that built
    the wavelet at ``_survey``'s default ``dt=0.001`` while the simulation
    itself ran at ``_COURANT_SAFE_DT=0.0003`` -- fixed here by passing
    ``dt=_COURANT_SAFE_DT`` through). At n=10, source and receiver are 2
    cells apart on every axis (Euclidean distance ``2*sqrt(ndim)*dx`` =
    14.1 m in 2-D / 17.3 m in 3-D). At the slowest relevant phase velocity
    (the axial ``vp`` ~= 1500 m/s; the horizontal/coupling direction is
    faster), a wavefront needs 14.1/1500 = 9.4 ms (2-D) / 17.3/1500 = 11.5 ms
    (3-D) to arrive; the previous ``nt=16`` window (16*0.0003 = 4.8 ms) was
    physically causally impossible to receive anything on, and the source
    wavelet's own peak (``peak_time=1.5/15=0.1``s) was also outside that
    window, so the injected source amplitude itself was ~3e-6 (a Ricker's
    early tail), not a genuine pulse. ``nt=120`` (120*0.0003 = 36 ms) gives
    ~3-4x the causal travel time and a source amplitude of ~1.8e-3 (600x
    larger). Verified: re-running this test's own FD/autograd check at
    nt=120 (unbugged) still agrees to ~1e-10-1e-9 relative (well inside the
    asserted 1e-5), and a from-scratch bug that disconnects c13 from the
    autograd graph on one of its two roles in the stress update -- not
    caught by any parametrization of this test at nt=16 -- is caught at
    nt=120 with rel diff 1.8 (2-D) / 0.13 (3-D) on the c13 direction.
    """
    c11, c13, c33, buoyancy = _make_media(ndim, n=10)
    _, amp, src, rec = _survey(ndim, 120, n=10, dt=_COURANT_SAFE_DT)
    torch.manual_seed(0)
    w = torch.randn(1, 1, 120, dtype=DTYPE)
    torch.manual_seed(3)
    inputs = [c11, c13, c33, buoyancy, amp]
    directions = [torch.randn_like(t) for t in inputs]

    def loss(vals):
        c11v, c13v, c33v, bv, ampv = vals
        out = deepwave.acoustic_vti(
            c11v,
            c13v,
            c33v,
            bv,
            5.0,
            _COURANT_SAFE_DT,
            source_amplitudes_x=ampv,
            source_locations_x=src,
            receiver_locations_x=rec,
            receiver_locations_y=rec,
            pml_width=4,
            accuracy=4,
            max_vel=3500.0,
            pml_freq=15.0,
            backend=backend,
        )
        return (out.receiver_amplitudes_x * w).sum() + (
            out.receiver_amplitudes_y * w
        ).sum()

    grad_inputs = [t.clone().requires_grad_(True) for t in inputs]
    loss(grad_inputs).backward()
    for i, name in enumerate(["c11", "c13", "c33", "buoyancy", "source"]):
        gdot = float((grad_inputs[i].grad * directions[i]).sum())
        h = 1e-6 * inputs[i].abs().mean().item()
        plus = list(inputs)
        minus = list(inputs)
        plus[i] = inputs[i] + h * directions[i]
        minus[i] = inputs[i] - h * directions[i]
        fd = (loss(plus).item() - loss(minus).item()) / (2 * h)
        rel = abs(gdot - fd) / max(abs(fd), 1e-300)
        assert rel < 1e-5, f"{name}: autograd {gdot} vs FD {fd} (rel {rel})"


@pytest.mark.parametrize("backend", BACKENDS)
def test_oracle_forward_through_thomsen(backend):
    """Cij built via thomsen_to_cij(vp, vs=0, ...) match elastic_vti forward.

    Confirms the differentiable Thomsen path (vs = 0 gives c55 = c66 = 0
    exactly) feeds acoustic_vti a medium bit-identical to elastic_vti's. The
    gradient through this path is covered by test_gradcheck_through_thomsen.

    Parametrized over ``backend`` (task 7.H milestone H3).
    """
    torch.manual_seed(9)
    n, nt, dt, ndim = 10, 20, 0.001, 2
    shape = (n,) * ndim
    vp = 3000 * (1 + 0.05 * torch.rand(*shape, dtype=DTYPE))
    rho = 2000 * (1 + 0.05 * torch.rand(*shape, dtype=DTYPE))
    eps = 0.2 * torch.ones(*shape, dtype=DTYPE)
    delta = 0.1 * torch.ones(*shape, dtype=DTYPE)
    amp = deepwave.wavelets.ricker(15.0, nt, dt, 0.1).to(DTYPE).reshape(1, 1, -1)
    src = torch.tensor([[[n // 2] * ndim]])
    rec = torch.tensor([[[n // 2 - 2] * ndim]])
    vs = torch.zeros_like(vp)
    kw = {
        "source_amplitudes_x": amp,
        "source_locations_x": src,
        "receiver_locations_x": rec,
        "pml_width": 4,
        "accuracy": 4,
        "max_vel": 4000.0,
        "pml_freq": 15.0,
    }
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        vp, vs, rho, eps, delta
    )
    assert float(c55.abs().max()) == 0.0
    av = deepwave.acoustic_vti(
        c11,
        c13,
        c33,
        1 / rho,
        5.0,
        dt,
        backend=backend,
        **kw,
    )
    ev = deepwave.elastic_vti(c11, c13, c33, c55, 1 / rho, 5.0, dt, c66=None, **kw)
    a = av.receiver_amplitudes_x
    scale = max(a.abs().max().item(), 1e-30)
    assert (a - ev.receiver_amplitudes_x).abs().max().item() <= scale * 1e-11


# ---------------------------------------------------------------------------
# Isotropic limit vs the acoustic (velocity-pressure) propagator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("ndim", [2, 3])
def test_isotropic_limit_matches_acoustic(ndim, backend):
    """Eps = delta = 0 (c11 = c13 = c33) reduces to variable-density acoustic.

    The two normal stresses collapse onto a single pressure-like field, so the
    velocity-receiver traces match ``deepwave.acoustic`` (which shares the
    staggered velocity-pressure discretisation) to machine precision; the level
    is pinned here rather than assumed.

    Parametrized over ``backend`` (task 7.H milestone H3).
    """
    torch.manual_seed(3)
    n, nt, dt = 14, 30, 0.0005
    shape = (n,) * ndim
    vp = 1500 + 200 * torch.rand(*shape, dtype=DTYPE)
    rho = 2000 + 200 * torch.rand(*shape, dtype=DTYPE)
    c33 = rho * vp**2
    freq = 20.0
    amp = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq).to(DTYPE).reshape(1, 1, -1)
    src = torch.tensor([[[n // 2] * ndim]])
    rec = torch.tensor([[[n // 2 - 2] * ndim]])
    recs = {"receiver_locations_x": rec, "receiver_locations_y": rec}
    if ndim == 3:
        recs["receiver_locations_z"] = rec
    kw = {"pml_width": 6, "accuracy": 4, "max_vel": 2200.0, "pml_freq": freq}
    av = deepwave.acoustic_vti(
        c33,
        c33.clone(),
        c33.clone(),
        1 / rho,
        5.0,
        dt,
        source_amplitudes_x=amp,
        source_locations_x=src,
        backend=backend,
        **recs,
        **kw,
    )
    ac = deepwave.acoustic(
        vp,
        rho,
        5.0,
        dt,
        source_amplitudes_x=amp,
        source_locations_x=src,
        **recs,
        **kw,
    )
    names = ["receiver_amplitudes_x", "receiver_amplitudes_y"] + (
        ["receiver_amplitudes_z"] if ndim == 3 else []
    )
    for name in names:
        a = getattr(av, name)
        c = getattr(ac, name)
        scale = max(a.abs().max().item(), 1e-30)
        assert (a - c).abs().max().item() <= scale * 1e-11


# ---------------------------------------------------------------------------
# Analytic Christoffel phase velocities (exact V_S = 0 qP surface)
# ---------------------------------------------------------------------------

VP0, RHO0, EPS_T, DELTA_T = 3000.0, 2000.0, 0.24, 0.1


def _cij():
    c33 = RHO0 * VP0**2
    return c33 * (1 + 2 * EPS_T), c33 * math.sqrt(1 + 2 * DELTA_T), c33


def _qp_phase_speed(theta, c11, c13, c33):
    """Exact V_S = 0 qP phase speed at angle ``theta`` from the symmetry axis."""
    s2, c2 = math.sin(theta) ** 2, math.cos(theta) ** 2
    g11, g33 = c11 * s2, c33 * c2
    g13 = c13 * math.sin(theta) * math.cos(theta)
    tr, det = g11 + g33, g11 * g33 - g13 * g13
    disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
    return math.sqrt((tr / 2 + disc) / RHO0)


def _q_of_p(p, orient, c11, c13, c33):
    """Transverse slowness from the exact V_S = 0 qP dispersion relation."""
    if orient == "h":  # p is transverse slowness, q axial
        num = RHO0 * (c11 * p * p - RHO0)
        den = (c11 * c33 - c13 * c13) * p * p - RHO0 * c33
    else:  # p is axial slowness, q transverse
        num = RHO0 * (c33 * p * p - RHO0)
        den = (c11 * c33 - c13 * c13) * p * p - RHO0 * c11
    return math.sqrt(num / den)


def _window_at(tr, dt, freq, t_center):
    period = 1.0 / freq
    nt = tr.shape[0]
    t = torch.arange(nt, dtype=DTYPE) * dt
    w = torch.zeros(nt, dtype=DTYPE)
    inside = (t >= t_center - 1.5 * period) & (t <= t_center + 1.5 * period)
    w[inside] = torch.hann_window(int(inside.sum()), periodic=False, dtype=DTYPE)
    return tr * w


def _measure_delay(x1, x2, dt, freq):
    nt = x1.shape[0]
    x1_f = torch.fft.rfft(x1)
    x2_f = torch.fft.rfft(x2)
    freqs = torch.fft.rfftfreq(nt, dt)
    cross = x2_f * x1_f.conj()
    corr = torch.fft.irfft(cross, n=nt)
    lag = int(corr.argmax().item())
    if lag > nt // 2:
        lag -= nt
    coarse = lag * dt
    omega_all = 2 * math.pi * freqs
    cross_res = cross * torch.exp(1j * omega_all * coarse)
    band = (freqs > 0.5 * freq) & (freqs < 1.5 * freq)
    w = cross_res.abs()[band]
    omega = omega_all[band]
    phase = torch.angle(cross_res[band])
    residual = -(w * phase * omega).sum() / (w * omega * omega).sum()
    return coarse + float(residual)


def _run_plane_wave(p_slow, orient, backend="eager"):
    """Measure (theta, v) of a qP plane wave with along-line slowness p_slow."""
    c11c, c13c, c33c = _cij()
    # ``across`` must keep both receivers (transverse coordinates 70 and 100)
    # clear of the pml_width=20 absorbing band, or the far receiver sits in the
    # PML and the measured slowness is corrupted (worst at oblique angles).
    across, along = 150, 260
    ny, nx = (across, along) if orient == "h" else (along, across)
    dx, dt, freq = 5.0, 0.0005, 14.0
    line_extent = (along // 2 - 30) * dx
    peak_time = 1.5 / freq + p_slow * line_extent
    q_exp = _q_of_p(p_slow, orient, c11c, c13c, c33c)
    t_far = peak_time + q_exp * (100 - 10) * dx + 3.0 / freq
    nt = math.ceil(t_far / dt)

    shape = (ny, nx)
    one = torch.ones(shape, dtype=DTYPE)
    c11, c13, c33 = c11c * one, c13c * one, c33c * one
    buoyancy = one / RHO0
    wavelet = deepwave.wavelets.ricker(freq, nt, dt, peak_time).to(DTYPE)
    line = torch.arange(30, along - 30, dtype=torch.long)
    src_loc = torch.zeros(1, line.shape[0], 2, dtype=torch.long)
    center = along // 2
    if orient == "h":
        src_loc[0, :, 0] = 10
        src_loc[0, :, 1] = line
        rec_loc = torch.tensor([[[70, center], [100, center]]], dtype=torch.long)
    else:
        src_loc[0, :, 0] = line
        src_loc[0, :, 1] = 10
        rec_loc = torch.tensor([[[center, 70], [center, 100]]], dtype=torch.long)
    delays = p_slow * (line.to(DTYPE) - float(center)) * dx
    amps = deepwave.transducers.apply_delays(wavelet, delays, dt)
    n_line = line.shape[0]
    n_tap = n_line // 4
    taper = torch.ones(n_line, dtype=DTYPE)
    ramp = torch.hann_window(2 * n_tap, periodic=False, dtype=DTYPE)
    taper[:n_tap] = ramp[:n_tap]
    taper[-n_tap:] = ramp[n_tap:]
    amps = (amps * taper.unsqueeze(-1)).unsqueeze(0)

    # Explosive (isotropic pressure) source excites qP; record sigma_v.
    out = deepwave.acoustic_vti(
        c11,
        c13,
        c33,
        buoyancy,
        dx,
        dt,
        source_amplitudes_sigma_h=amps,
        source_amplitudes_sigma_v=amps,
        source_locations_sigma_h=src_loc,
        source_locations_sigma_v=src_loc,
        receiver_locations_sigma_v=rec_loc,
        pml_width=20,
        accuracy=8,
        pml_freq=freq,
        backend=backend,
    )
    rec = out.receiver_amplitudes_sigma_v
    tr1 = _window_at(rec[0, 0, :], dt, freq, peak_time + q_exp * (70 - 10) * dx)
    tr2 = _window_at(rec[0, 1, :], dt, freq, peak_time + q_exp * (100 - 10) * dx)
    dy = (100 - 70) * dx
    q = _measure_delay(tr1, tr2, dt, freq) / dy
    v_meas = 1.0 / math.sqrt(p_slow**2 + q**2)
    theta = (
        math.atan2(p_slow, q) if orient == "h" else math.pi / 2 - math.atan2(p_slow, q)
    )
    return theta, v_meas, _qp_phase_speed(theta, c11c, c13c, c33c)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("p_slow", "orient", "angle"),
    [
        (0.0, "h", 0.0),
        (1.3e-4, "h", 21.0),
        (1.0e-4, "v", 73.0),
        (0.0, "v", 90.0),
    ],
)
def test_phase_velocity_christoffel(p_slow, orient, angle, backend):
    """Measured qP phase speeds match the exact Christoffel solution (<3%).

    Homogeneous strongly-anisotropic medium (eps=0.24, delta=0.1). This detects
    axis-convention or constitutive-coupling errors independently of the
    elastic_vti oracle (which would give the wrong angle or a many-percent speed
    error). The 3% tolerance is the grid-dispersion + slowness-measurement
    budget of this deliberately modest eager-backend grid; the forward is
    bit-identical to elastic_vti (verified separately), whose own Christoffel
    test resolves <0.2% on a larger grid.

    Parametrized over ``backend`` (task 7.H milestone H3): re-running this
    independent analytic oracle against the compiled forward is a stronger
    check than plain eager-vs-compiled parity, since it does not merely prove
    the two backends agree with each other.
    """
    theta, v_meas, v_ref = _run_plane_wave(p_slow, orient, backend)
    assert abs(math.degrees(theta) - angle) < 8.0
    assert abs(v_meas - v_ref) / v_ref < 0.03


# ---------------------------------------------------------------------------
# Grechka stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_grechka_stable_run_bounded(backend):
    """A long run with eps > delta everywhere stays bounded.

    Parametrized over ``backend`` (task 7.H milestone H3).
    """
    c11, c13, c33, buoyancy = _make_media(2, n=30, eps=0.25, delta=0.05)
    nt = 600
    amp = deepwave.wavelets.ricker(15.0, nt, 0.001, 0.1).to(DTYPE).reshape(1, 1, -1)
    src = torch.tensor([[[15, 15]]])
    out = deepwave.acoustic_vti(
        c11,
        c13,
        c33,
        buoyancy,
        5.0,
        0.001,
        source_amplitudes_sigma_v=amp,
        source_locations_sigma_v=src,
        source_amplitudes_sigma_h=amp,
        source_locations_sigma_h=src,
        pml_width=10,
        accuracy=4,
        max_vel=3000.0,
        pml_freq=15.0,
        backend=backend,
    )
    # A stable run stays finite and, the source having long finished and the
    # PML having absorbed the field, decays far below the ~1e-3-per-step source
    # injection; an eps < delta instability would instead grow without bound.
    assert torch.isfinite(out.sigma_v_0).all()
    assert out.sigma_v_0.abs().max().item() < 1e-2


def test_grechka_guard_raises():
    """Eps < delta anywhere (c11 c33 < c13**2) raises a physics-naming error."""
    n = 10
    c33 = torch.full((n, n), 2000.0**2 * 2000.0, dtype=DTYPE)
    c11 = c33.clone()
    c13 = c33 * 1.2  # c13**2 > c11 c33 -> delta > eps
    b = torch.full((n, n), 1 / 2000.0, dtype=DTYPE)
    amp = torch.zeros(1, 1, 5, dtype=DTYPE)
    src = torch.tensor([[[5, 5]]])
    with pytest.raises(ValueError, match=r"epsilon >= delta|c11 \* c33"):
        deepwave.acoustic_vti(
            c11,
            c13,
            c33,
            b,
            5.0,
            0.001,
            source_amplitudes_x=amp,
            source_locations_x=src,
        )


# ---------------------------------------------------------------------------
# Reciprocity and PML absorption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_reciprocity_2d(backend):
    """Source/receiver reciprocity for a same-component velocity pair (vx-vx).

    Parametrized over ``backend`` (task 7.H milestone H3).
    """
    ny, nx, dx, dt, nt, freq = 60, 70, 5.0, 0.001, 200, 15.0
    one = torch.ones(ny, nx, dtype=DTYPE)
    c33 = 2000.0**2 * 2000.0 * one
    c11 = c33 * (1 + 2 * 0.25)
    c13 = c33 * math.sqrt(1 + 2 * 0.1)
    buoyancy = one / 2000.0
    amp = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq).to(DTYPE).reshape(1, 1, -1)
    loc_a = torch.tensor([[[25, 20]]], dtype=torch.long)
    loc_b = torch.tensor([[[35, 50]]], dtype=torch.long)

    def run(src, rec):
        return deepwave.acoustic_vti(
            c11,
            c13,
            c33,
            buoyancy,
            dx,
            dt,
            source_amplitudes_x=amp,
            source_locations_x=src,
            receiver_locations_x=rec,
            pml_width=20,
            accuracy=4,
            pml_freq=freq,
            backend=backend,
        ).receiver_amplitudes_x

    d_ab = run(loc_a, loc_b)
    d_ba = run(loc_b, loc_a)
    scale = d_ab.abs().max().item()
    assert (d_ab - d_ba).abs().max().item() < 5e-3 * scale


@pytest.mark.parametrize("backend", BACKENDS)
def test_pml_absorption(backend, record_property):
    """Boundary reflections are small: a thin-PML run matches a padded reference.

    Parametrized over ``backend`` (task 7.H milestone H3).
    """
    dx, dt, nt, freq = 5.0, 0.0005, 600, 18.0
    pad = 40
    n = 60

    def run(n_full, off, width=10):
        shape = (n_full, n_full)
        one = torch.ones(shape, dtype=DTYPE)
        c33 = 2000.0**2 * 2000.0 * one
        c11 = c33 * (1 + 2 * 0.2)
        c13 = c33 * math.sqrt(1 + 2 * 0.1)
        buoyancy = one / 2000.0
        amp = (
            deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq)
            .to(DTYPE)
            .reshape(1, 1, -1)
        )
        src = torch.tensor([[[off + 30, off + 30]]], dtype=torch.long)
        rec = torch.tensor([[[off + 45, off + 45]]], dtype=torch.long)
        # Named access (not positional [-1]/[-2]) because A4.2 made the
        # receiver-output channel order sigma_v-then-sigma_h (previously
        # sigma_h-then-sigma_v), and only receiver_locations_sigma_v is
        # provided here.
        return deepwave.acoustic_vti(
            c11,
            c13,
            c33,
            buoyancy,
            dx,
            dt,
            source_amplitudes_sigma_v=amp,
            source_amplitudes_sigma_h=amp,
            source_locations_sigma_v=src,
            source_locations_sigma_h=src,
            receiver_locations_sigma_v=rec,
            pml_width=width,
            accuracy=4,
            pml_freq=freq,
            backend=backend,
        ).receiver_amplitudes_sigma_v

    small = run(n, 0)
    large = run(n + 2 * pad, pad)
    scale = large.abs().max().item()
    # The record includes the reflected main pulse, not just the direct tail.
    reflecting = run(n, 0, width=0)
    assert scale > 0
    absorbed_error = (small - large).abs().max().item() / scale
    reflecting_error = (reflecting - large).abs().max().item() / scale
    record_property("pml_peak_relative_error", absorbed_error)
    record_property("no_pml_peak_relative_error", reflecting_error)
    assert reflecting_error > 0.2
    assert absorbed_error < 5e-3


# ---------------------------------------------------------------------------
# Absolute finite-difference gradcheck through the Thomsen converters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_gradcheck_through_thomsen(backend):
    """FD gradcheck chained through thomsen_to_cij(vs=0) (Thomsen-space FWI).

    Parametrized over ``backend`` (task 7.H milestone H3). ``dt`` is
    courant-safe (step_ratio == 1 at max_vel=5000.0) so the compiled
    gradient is exact, matching gradcheck's tight tolerances.

    ``vp``/``rho``/``eps``/``delta`` are already reasonably scaled for
    ``torch.autograd.gradcheck``'s default ``eps=1e-6``: no float64-ULP
    underflow the way raw ~1e9-scale Cij leaf tensors would have, where an
    absolute 1e-6 perturbation sits below the value's own local ULP
    (1e9 * 2**-52 ~= 2.2e-7) and the numerical Jacobian comes back exactly
    zero. At this tiny grid/nt the analytical
    Jacobian was still measured at only ~1e-12 to 1e-15 (e.g. grad_delta),
    far below ``atol=1e-8``: a from-scratch bug that disconnects c13 from
    the graph (delta's only route to a nonzero gradient in this
    parametrization: c13 is the only Cij depending on delta) was NOT
    caught -- ``delta.grad`` came back ``None`` and gradcheck still
    reported success. Since this is a linear PDE, the model-parameter
    Jacobian scales with the source's overall amplitude; ``ampscale``
    raises the (originally O(1) ``randn``) source amplitude so this
    Jacobian clears ``atol`` without needing a larger, more expensive
    ``nt``/grid. Verified: with ``ampscale``, nt UNCHANGED at 8, the
    c13-disconnection bug is caught (delta.grad is no longer silently
    masked), while the unmodified (correct) implementation still
    gradchecks cleanly at several tested ``ampscale`` values (1e6-1e10).
    """
    torch.manual_seed(5)
    n, nt, dt = 8, 8, 0.0003
    ampscale = 1e6
    shape = (n,) * 2
    vp = (3000 * (1 + 0.05 * torch.rand(*shape, dtype=DTYPE))).requires_grad_(True)
    rho = (2000 * (1 + 0.05 * torch.rand(*shape, dtype=DTYPE))).requires_grad_(True)
    eps = (0.2 * torch.ones(*shape, dtype=DTYPE)).requires_grad_(True)
    delta = (0.1 * torch.ones(*shape, dtype=DTYPE)).requires_grad_(True)
    amp = torch.randn(1, 1, nt, dtype=DTYPE) * ampscale
    src = torch.tensor([[[n // 2] * 2]], dtype=torch.long)
    rec = torch.tensor([[[n // 2 - 1] * 2]], dtype=torch.long)

    def wrapper(vp_t, rho_t, eps_t, delta_t):
        vs = torch.zeros_like(vp_t)
        c11, c13, c33, _, _, _ = deepwave.anisotropy.thomsen_to_cij(
            vp_t, vs, rho_t, eps_t, delta_t
        )
        out = deepwave.acoustic_vti(
            c11,
            c13,
            c33,
            1 / rho_t,
            5.0,
            dt,
            source_amplitudes_sigma_v=amp,
            source_locations_sigma_v=src,
            source_amplitudes_sigma_h=amp,
            source_locations_sigma_h=src,
            receiver_locations_x=rec,
            pml_width=2,
            accuracy=2,
            max_vel=5000.0,
            pml_freq=15.0,
            backend=backend,
        )
        # NOT ``out[-2]`` (``receiver_amplitudes_sigma_v``): no
        # ``receiver_locations_sigma_v`` is set above, so that channel is an
        # empty (shape ``(0,)``) tensor -- gradcheck of an empty output
        # trivially "passes" with zero elements checked, regardless of any
        # bug (found while reconditioning this test; see the docstring).
        # ``receiver_locations_x=rec`` is the channel actually populated.
        return out.receiver_amplitudes_x

    assert torch.autograd.gradcheck(
        wrapper, (vp, rho, eps, delta), atol=1e-8, rtol=1e-4
    )


# ---------------------------------------------------------------------------
# Pseudo-shear source taper
# ---------------------------------------------------------------------------


def test_source_taper_closed_form():
    """The raised-cosine c13 blend hits its closed-form values exactly."""
    n = 21
    c11 = torch.full((n, n), 3.0, dtype=DTYPE)
    c33 = torch.full((n, n), 12.0, dtype=DTYPE)  # sqrt(c11 c33) = 6
    c13 = torch.full((n, n), 4.0, dtype=DTYPE)
    src = torch.tensor([[[10, 10]]])
    radius = 5.0
    out = deepwave.anisotropy.source_taper(c11, c13, c33, src, radius)
    ell = math.sqrt(3.0 * 12.0)
    assert abs(out[10, 10].item() - ell) < 1e-12  # r = 0, w = 1
    assert abs(out[10, 15].item() - 4.0) < 1e-12  # r = radius, w = 0
    assert abs(out[0, 0].item() - 4.0) < 1e-12  # far, unchanged
    r = 3.0
    w = 0.5 * (1 + math.cos(math.pi * r / radius))
    assert abs(out[10, 13].item() - (w * ell + (1 - w) * 4.0)) < 1e-12


def test_source_taper_integrates_and_perturbs_near_source():
    """The taper feeds a stable, locally-elliptical model into the propagator.

    The raised-cosine blend is validated exactly by test_source_taper_closed_form;
    here we check that the tapered c13 (i) makes the near-source medium elliptical
    so it satisfies the Grechka guard and runs to a finite result, and (ii)
    genuinely changes the wavefield relative to the untapered eps != delta medium
    (i.e. the mitigation is actually applied inside the propagator). A single
    receiver's amplitude is not a monotone proxy for the diamond-shaped
    pseudo-shear emission, so we assert a stable, non-trivial effect rather than a
    directional decrease.
    """
    ny, nx, dx, dt, nt, freq = 80, 80, 5.0, 0.0006, 220, 16.0
    one = torch.ones(ny, nx, dtype=DTYPE)
    c33 = 2000.0**2 * 2000.0 * one
    c11 = c33 * (1 + 2 * 0.3)
    c13 = c33 * math.sqrt(1 + 2 * 0.02)  # delta << eps: strong artifact
    buoyancy = one / 2000.0
    amp = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq).to(DTYPE).reshape(1, 1, -1)
    src = torch.tensor([[[40, 40]]])
    rec = torch.tensor([[[42, 60]]])  # off-axis receiver
    kw = {
        "source_amplitudes_sigma_v": amp,
        "source_amplitudes_sigma_h": amp,
        "source_locations_sigma_v": src,
        "source_locations_sigma_h": src,
        "receiver_locations_y": rec,
        "pml_width": 15,
        "accuracy": 4,
        "pml_freq": freq,
    }
    base = deepwave.acoustic_vti(
        c11, c13, c33, buoyancy, dx, dt, **kw
    ).receiver_amplitudes_y
    c13_t = deepwave.anisotropy.source_taper(c11, c13, c33, src, 6.0)
    # At the source the tapered c13 is the elliptical sqrt(c11 c33).
    assert abs(c13_t[40, 40].item() - math.sqrt(c11[40, 40] * c33[40, 40])) < 1e-6
    tapered = deepwave.acoustic_vti(
        c11, c13_t, c33, buoyancy, dx, dt, **kw
    ).receiver_amplitudes_y
    assert torch.isfinite(tapered).all()
    scale = base.abs().max().item()
    diff = (tapered - base).abs().max().item()
    assert diff > 1e-3 * scale  # the mitigation actually changed the wavefield


# ---------------------------------------------------------------------------
# Guards, ergonomics, and the step_ratio (CFL) machinery
# ---------------------------------------------------------------------------


def test_ndim_1_raises():
    """1-D inputs are rejected (VTI degenerates to the scalar equation)."""
    c = torch.ones(10, dtype=DTYPE)
    amp = torch.zeros(1, 1, 10, dtype=DTYPE)
    loc = torch.tensor([[[5]]], dtype=torch.long)
    with pytest.raises(ValueError, match=r"at least 2|degenerates"):
        deepwave.acoustic_vti(
            c, c, c, c, 5.0, 0.001, source_amplitudes_x=amp, source_locations_x=loc
        )


# ---------------------------------------------------------------------------
# Compiled and torch.compile backends: forward-only parity vs eager
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ndim", "pml_width", "free_surface"),
    [
        (2, 0, False),
        (2, 4, False),
        (3, 4, False),
        (2, 4, True),
    ],
)
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("compiled", id="compiled"),
        pytest.param("torch_compile", id="torch_compile"),
    ],
)
def test_compiled_forward_matches_eager(ndim, pml_width, free_surface, backend):
    """Compiled and torch.compile forward match eager to near machine precision.

    Covers 2-D and 3-D, ``pml_width`` 0 and > 0, and one free-surface
    (vacuum) boundary configuration. ``backend``
    additionally covers the ``torch_compile`` Python mode, which runs
    the same eager time loop with :func:`update_velocities_acoustic_vti`/
    :func:`update_stresses_acoustic_vti` ``torch.compile``d, so
    it is held to the same tolerance as the compiled C/CUDA backend.
    """
    c11, c13, c33, buoyancy = _make_media(ndim, n=14)
    if free_surface:
        # Vacuum the first two cells along the (slowest) axis dimension --
        # improved-vacuum free surface, as in deepwave.elastic's tests.
        c11 = c11.clone()
        c13 = c13.clone()
        c33 = c33.clone()
        buoyancy = buoyancy.clone()
        c11[:2] = 0.0
        c13[:2] = 0.0
        c33[:2] = 0.0
        buoyancy[:2] = 0.0
        pml_width_arg = [0, pml_width] * ndim
    else:
        pml_width_arg = pml_width
    kwargs, _, _, _ = _survey(ndim, nt=32, n=14, stress_source=True)
    kwargs["pml_width"] = pml_width_arg
    common = {
        "c11": c11,
        "c13": c13,
        "c33": c33,
        "buoyancy": buoyancy,
        "grid_spacing": 5.0,
    }
    out_eager = deepwave.acoustic_vti(**common, backend="eager", **kwargs)
    out_other = deepwave.acoustic_vti(**common, backend=backend, **kwargs)
    assert len(out_eager) == len(out_other)
    # The auxiliary state holds several tensors under one output name, so
    # expand it and label each with its own field name.
    for name, a, b in zip(
        acoustic_vti_module.acoustic_vti_output_names(ndim),
        out_eager,
        out_other,
    ):
        for i, (x, y) in enumerate(zip(element_tensors(a), element_tensors(b))):
            label = name if isinstance(a, torch.Tensor) else a.names[i]
            torch.testing.assert_close(
                x,
                y,
                atol=1e-10,
                rtol=1e-10,
                msg=lambda m, label=label: f"field {label!r}: {m}",
            )


@C_BACKEND
@pytest.mark.parametrize("ndim", [2, 3])
def test_c_threading_matches_single_thread(ndim):
    """Intra-shot OpenMP threading (B2) does not change the compiled result.

    ``acoustic_vti.c`` was ported from one-shot-per-thread parallelism to
    intra-shot spatial parallelism (see ``acoustic_vti.c``'s ``FUNC(forward)``/
    ``FUNC(backward)``, mirroring ``elastic.c`` and ``viscoacoustic.c``'s B2
    port): each time step's velocity-update loops (vz/vy/vx) and the combined
    sigma_v/sigma_h update loop are now split across threads via ``#pragma omp
    for simd ... nowait``, with shots processed serially. Every grid cell --
    including the (grad_c11, grad_c13, grad_c33) gradient accumulation, which
    for this propagator is a single combined update touching all three
    stiffness gradients together (unlike a scalar propagator's single
    gradient) -- is written by exactly one thread per parallel region (the
    PML-region tiling partitions the grid disjointly) and time steps/shots are
    barrier-separated, so the arithmetic performed for any given cell does not
    depend on how many threads are used -- forward and gradients should be
    identical (to measurement precision) whether the compiled backend runs
    with 1 or 4 threads. This is a single-shot (``n_shots=1``, the default
    from ``_survey``) problem, so under the old one-shot-per-thread scheme
    extra threads would have sat idle; here they should be put to work within
    the single shot instead (this is what ``spatial_parallel=True`` in
    ``acoustic_vti.py``'s ``SPEC`` enables).

    Uses ``torch.set_num_threads`` (read by the backend via
    ``torch.get_num_threads()`` at call time, exactly as ``OMP_NUM_THREADS``
    would set the default) rather than an env var, so the comparison runs
    in-process; the tolerance mirrors
    ``test_compiled_forward_matches_eager``/``test_compiled_backward_matches_eager``
    above (the existing cross-backend-parity patterns in this file), not a
    new one.
    """
    saved_threads = torch.get_num_threads()
    try:
        c11, c13, c33, buoyancy = _make_media(ndim, n=14)
        kwargs, _, _, _ = _survey(ndim, nt=32, n=14, stress_source=True)

        def run(n_threads):
            torch.set_num_threads(n_threads)
            c11_ = c11.clone().requires_grad_(True)
            c13_ = c13.clone().requires_grad_(True)
            c33_ = c33.clone().requires_grad_(True)
            b_ = buoyancy.clone().requires_grad_(True)
            out = deepwave.acoustic_vti(
                c11_,
                c13_,
                c33_,
                b_,
                5.0,
                backend="compiled",
                **kwargs,
            )
            loss = sum(o.pow(2).sum() for o in output_tensors(out) if o.numel() > 0)
            loss.backward()
            return out, (c11_.grad, c13_.grad, c33_.grad, b_.grad)

        out_1, g_1 = run(1)
        out_4, g_4 = run(4)
        for a, b in zip(output_tensors(out_1), output_tensors(out_4)):
            if a.numel() == 0:
                continue
            scale = max(a.abs().max().item(), 1.0)
            assert (a - b).abs().max().item() <= scale * 1e-6
        for g1, g4 in zip(g_1, g_4):
            scale = max(g1.abs().max().item(), 1.0)
            assert (g1 - g4).abs().max().item() <= scale * 1e-6
    finally:
        torch.set_num_threads(saved_threads)


# ---------------------------------------------------------------------------
# Compiled backend: adjoint (backward) gradients (task 7.H milestone H2)
# ---------------------------------------------------------------------------
#
# Gradient parity vs the verified eager backend (task G). The compiled
# backend's model gradients (c11/c13/c33/buoyancy) are stored/sampled once
# per ``step_ratio`` block and rescaled by ``step_ratio`` (derivation doc
# Sec. 5; the elastic_born-escalation-#7 pattern) -- this is *exact* only
# when the sampled substep is representative of its whole block, i.e. at
# ``step_ratio == 1``. At ``step_ratio > 1`` this is a genuine, small,
# BOUNDED approximation shared with (not introduced by) the rest of the
# compiled-backend family: ``deepwave.elastic_vti`` shows the *same order of
# magnitude* of eager-vs-compiled gradient discrepancy under the identical
# CFL-forced ``step_ratio > 1`` condition (verified empirically while
# debugging this task; not a regression specific to acoustic_vti). So:
#   * ``step_ratio == 1`` (courant-safe dt, model_gradient_sampling_interval
#     == 1): near machine precision.
#   * ``step_ratio in (2, 3)`` (via ``model_gradient_sampling_interval``,
#     with a courant-safe dt so this is the *only* source of step_ratio >
#     1): bounded but non-machine-precision agreement (empirically a few
#     percent worst-case here); the mathematically strict, step_ratio-
#     independent check of the compiled adjoint itself is
#     tests/test_adjoint_dot_product.py's ``<F s, r> == <s, F* r>`` identity
#     (~1e-15), which this test complements rather than duplicates, and the
#     compiled-vs-compiled oracle
#     ``test_compiled_gradients_match_elastic_vti_compiled`` below, which pins
#     the model-gradient sampling/rescaling machinery to ~1e-12 at every
#     sampling interval against elastic_vti's independently verified compiled
#     adjoint (both compiled backends implement the SAME sampling scheme, so
#     no approximation gap exists between them).
# step_ratio == 1 from CFL alone at max_vel=3500, BOTH 2-D and 3-D: the CFL
# sum (1/dx**2 + 1/dy**2 [+ 1/dz**2]) has one more term in 3-D, so 3-D's
# critical dt is smaller than 2-D's -- 0.0005 (safe in 2-D) actually gives
# step_ratio == 2 in 3-D, an easy trap when eyeballing this (verified
# empirically via deepwave._common.setup_propagator's returned step_ratio
# while chasing a spurious "3-D-only" gradient mismatch that was really this
# test using an insufficiently small dt, not a bug in the propagator).
_COURANT_SAFE_DT = 0.0003


def _grad_parity_setup(ndim, model_gradient_sampling_interval):
    # ``nt=40`` (the previous value) is a vacuous gate for anything specific
    # to the PML region: n=14 with pml_width=4 leaves an interior of only
    # 14 - 2*4 = 6 cells, so the source/receiver pair (2 cells apart, both
    # interior) sits just 2-3 cells from the nearest PML band. At
    # CFL = v*dt/dx with v up to ~2130 (c11-direction, eps=0.2) and
    # dt=_COURANT_SAFE_DT=0.0003, dx=5.0: 2130*0.0003/5 = 0.128 cells/step,
    # so 40 steps only advances the front ~5 cells -- it barely grazes the
    # PML band and never develops meaningful absorption dynamics there.
    # Verified directly: injecting a sign flip into the eager PML
    # memory-variable recursion (``update_velocities_acoustic_vti``'s
    # ``mem = ah*s_mem_vars + bh*d``) was NOT caught by this test at
    # nt=40 for any of the six (ndim, model_gradient_sampling_interval)
    # cases (all six passed at rel diff ~1e-12, machine precision, despite
    # the bug), yet the same bug WAS caught by
    # test_compiled_forward_matches_eager's PML-active configs (rel diff
    # ~1e-1) -- a real vacuous-gate gap in this file's own primary
    # compiled-vs-eager gradient parity gate, the same "dust" pattern
    # documented in 84fd036 for the (separate, boundary=-specific) adjoint
    # bug. ``nt=1200`` (30x) advances the front ~150 cells, many multiples
    # of the interior width, giving the wave sustained residence inside the
    # PML band: re-measured with the same injected bug at nt=1200, the
    # eager-vs-compiled c11/c13/c33/buoyancy relative differences are
    # 0.09-0.51 (ndim=2) and 0.19-0.40 (ndim=3) at every
    # model_gradient_sampling_interval in {1, 2, 3} -- comfortably above
    # both the mgsi=1 (1e-8) and mgsi>1 (0.1) thresholds asserted below,
    # with several-fold margin. n=14/pml_width=4/dt are otherwise unchanged.
    c11, c13, c33, buoyancy = _make_media(ndim, n=14)
    kwargs, _, _, _ = _survey(
        ndim, nt=1200, n=14, dt=_COURANT_SAFE_DT, stress_source=True
    )
    kwargs["pml_width"] = 4
    kwargs["model_gradient_sampling_interval"] = model_gradient_sampling_interval
    return c11, c13, c33, buoyancy, kwargs


def _run_grad(c11, c13, c33, buoyancy, kwargs, backend):
    c11_ = c11.clone().requires_grad_()
    c13_ = c13.clone().requires_grad_()
    c33_ = c33.clone().requires_grad_()
    b_ = buoyancy.clone().requires_grad_()
    out = deepwave.acoustic_vti(
        c11_,
        c13_,
        c33_,
        b_,
        5.0,
        backend=backend,
        **kwargs,
    )
    loss = sum(
        (getattr(out, f"receiver_amplitudes_{c}") ** 2).sum() for c in ("x", "y")
    )
    loss.backward()
    return (
        loss.item(),
        c11_.grad.clone(),
        c13_.grad.clone(),
        c33_.grad.clone(),
        b_.grad.clone(),
    )


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("model_gradient_sampling_interval", [1, 2, 3])
def test_compiled_backward_matches_eager(ndim, model_gradient_sampling_interval):
    """Compiled model gradients match eager (task 7.H milestone H2 gate).

    2-D and 3-D (3-D is mandatory: the ``sigma_h = sigma_xx + sigma_yy``
    summation hazard, derivation doc Sec. 2/7a, is invisible in 2-D), at
    ``model_gradient_sampling_interval`` (the user-facing ``step_ratio``
    knob) in ``{1, 2, 3}`` with a courant-safe ``dt`` so this is the only
    source of ``step_ratio > 1`` (isolated from any CFL-forced internal
    substepping). See the module comment above for the tolerance rationale.
    """
    c11, c13, c33, buoyancy, kwargs = _grad_parity_setup(
        ndim, model_gradient_sampling_interval
    )
    le, g11e, g13e, g33e, gbe = _run_grad(c11, c13, c33, buoyancy, kwargs, "eager")
    lc, g11c, g13c, g33c, gbc = _run_grad(c11, c13, c33, buoyancy, kwargs, "compiled")
    torch.testing.assert_close(torch.tensor(lc), torch.tensor(le), rtol=1e-9, atol=0)
    rtol = 1e-8 if model_gradient_sampling_interval == 1 else 0.1
    for name, ge, gc in [("c11", g11e, g11c), ("c13", g13e, g13c), ("c33", g33e, g33c)]:
        denom = ge.abs().max().clamp(min=1e-300)
        rel = ((ge - gc).abs() / denom).max().item()
        assert rel < rtol, (
            f"{name} (ndim={ndim}, "
            f"mgsi={model_gradient_sampling_interval}): rel diff {rel} >= {rtol}"
        )
    denom_b = gbe.abs().max().clamp(min=1e-300)
    rel_b = ((gbe - gbc).abs() / denom_b).max().item()
    assert rel_b < rtol, (
        f"buoyancy (ndim={ndim}, mgsi={model_gradient_sampling_interval}): "
        f"rel diff {rel_b} >= {rtol}"
    )


@C_BACKEND
@pytest.mark.skipif(
    not c_backend_available("elastic_vti", device=DEVICE),
    reason="compiled elastic_vti backend not present in libdeepwave_C",
)
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("model_gradient_sampling_interval", [1, 2, 3])
def test_compiled_gradients_match_elastic_vti_compiled(
    ndim, model_gradient_sampling_interval
):
    """Compiled model gradients match compiled elastic_vti(c55=c66=0) tightly.

    The strict step_ratio gate the derivation doc (Secs. 2/5/6) calls for,
    complementing ``test_compiled_backward_matches_eager``: the eager
    reference there is exact autograd, so a bounded tolerance is inherent at
    ``model_gradient_sampling_interval > 1``, whereas elastic_vti's
    hand-written compiled adjoint implements the SAME once-per-block gradient
    sampling and ``(DW_DTYPE)step_ratio`` rescaling as acoustic_vti's, so the
    two compiled backends must agree near machine precision at EVERY sampling
    interval. A missing or wrong step_ratio gating/scaling, and in 3-D any
    ``s_h = s_xx + s_yy`` summation error (derivation doc Sec. 2), breaks
    this parity directly. Velocity
    sources and receivers only: those states and pullbacks are identical
    under the Sec. 2 reduction map (stress receivers pull back differently).
    grad_c11 needs no folding correction here because autograd folds
    elastic_vti's ``c12 = c11 - 2*c66`` dependence into its returned c11
    gradient, which at ``c66 = 0`` equals acoustic_vti's grad_c11 exactly
    (derivation doc Sec. 2). Added by the task 7.J adversarial review
    (2026-07-12), which measured ~1e-15 relative agreement across all cases.
    """
    n = 14
    c11, c13, c33, buoyancy = _make_media(ndim, n=n)
    # nt must be divisible by every model_gradient_sampling_interval this test
    # parametrises over (1, 2, 3): the history layout requires complete
    # sampling blocks. 42 is the smallest such value near the original 40, which
    # left a remainder of 1 at mgsi=3 -- both propagators were then truncated
    # identically and agreed with each other about a one-step-short
    # simulation. Deepwave now rejects that combination outright.
    kwargs, _, _, _ = _survey(ndim, nt=42, n=n, dt=_COURANT_SAFE_DT)
    kwargs["pml_width"] = 4
    kwargs["model_gradient_sampling_interval"] = model_gradient_sampling_interval

    def run_grads(propagator):
        c11_ = c11.clone().requires_grad_()
        c13_ = c13.clone().requires_grad_()
        c33_ = c33.clone().requires_grad_()
        b_ = buoyancy.clone().requires_grad_()
        amp_ = kwargs["source_amplitudes_x"].clone().requires_grad_()
        kw = dict(kwargs)
        kw["source_amplitudes_x"] = amp_
        if propagator == "acoustic_vti":
            out = deepwave.acoustic_vti(
                c11_,
                c13_,
                c33_,
                b_,
                5.0,
                backend="compiled",
                **kw,
            )
        else:
            zero = torch.zeros_like(c11)
            out = deepwave.elastic_vti(
                c11_,
                c13_,
                c33_,
                zero,
                b_,
                5.0,
                c66=zero if ndim == 3 else None,
                backend="compiled",
                **kw,
            )
        loss = sum((r**2).sum() for r in _velocity_receivers(out, ndim))
        loss.backward()
        return [c11_.grad, c13_.grad, c33_.grad, b_.grad, amp_.grad]

    grads_av = run_grads("acoustic_vti")
    grads_ev = run_grads("elastic_vti")
    for name, av, ev in zip(
        ("c11", "c13", "c33", "buoyancy", "source"), grads_av, grads_ev
    ):
        denom = ev.abs().max().clamp(min=1e-300)
        rel = ((av - ev).abs() / denom).max().item()
        assert rel < 1e-12, (
            f"{name} (ndim={ndim}, "
            f"mgsi={model_gradient_sampling_interval}): rel diff {rel} >= 1e-12"
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_named_output_and_initial_state_roundtrip(backend):
    """PropagatorOutput field names match, and outputs feed back as initial_state.

    Parametrized over ``backend`` (task 7.H milestone H3), mirroring
    test_elastic_vti.py's own version of this test.
    """
    c11, c13, c33, buoyancy = _make_media(2)
    kwargs, _, _, _ = _survey(2, 20)
    out = deepwave.acoustic_vti(
        c11,
        c13,
        c33,
        buoyancy,
        5.0,
        backend=backend,
        **kwargs,
    )
    assert hasattr(out, "vy_0")
    assert hasattr(out, "sigma_v_0")
    assert hasattr(out, "receiver_amplitudes_sigma_h")
    out2 = deepwave.acoustic_vti(
        c11,
        c13,
        c33,
        buoyancy,
        5.0,
        backend=backend,
        initial_state=out,
        **kwargs,
    )
    assert out2[-1].shape == out.receiver_amplitudes_sigma_v.shape


@pytest.mark.parametrize("backend", BACKENDS)
def test_module_matches_functional(backend):
    """The AcousticVti module reproduces the functional interface.

    Parametrized over ``backend`` (task 7.H milestone H3).
    """
    c11, c13, c33, buoyancy = _make_media(2)
    kwargs, _, _, _ = _survey(2, 20)
    prop = deepwave.AcousticVti(c11, c13, c33, buoyancy, 5.0)
    out_m = prop(backend=backend, **kwargs)
    out_f = deepwave.acoustic_vti(
        c11, c13, c33, buoyancy, 5.0, backend=backend, **kwargs
    )
    for a, b in zip(output_tensors(out_m), output_tensors(out_f)):
        assert torch.equal(a, b)


def test_batched_models_eager():
    """Per-shot batched models run and differ across shots.

    Eager only, pinned explicitly with ``backend="eager"`` (the
    compiled backend's batched-model path is untested here, matching
    test_elastic_vti.py's own eager-only version of this test -- not
    something task 7.H claims coverage of; the default flip to compiled
    (task 7.H milestone H3) must not silently change what this test
    exercises).
    """
    c11, c13, c33, buoyancy = _make_media(2, batch=True)
    c11 = c11.clone()
    c11[1] *= 1.4
    kwargs, _, _, _ = _survey(2, 20, n_shots=2)
    # Record a stress receiver (well above the ~1e-13 velocity scale, so the
    # per-shot difference is not masked by allclose's absolute tolerance).
    src = kwargs["source_locations_x"]
    kwargs["receiver_locations_sigma_v"] = src.clone()
    out = deepwave.acoustic_vti(
        c11,
        c13,
        c33,
        buoyancy,
        5.0,
        backend="eager",
        **kwargs,
    )
    rec = out.receiver_amplitudes_sigma_v
    assert rec.shape[0] == 2
    assert (rec[0] - rec[1]).abs().max() > 1e-6 * rec.abs().max()


@pytest.mark.parametrize("backend", BACKENDS)
def test_step_ratio_matches_elastic_vti(backend):
    """A coarse dt (forcing internal step_ratio > 1) still tracks elastic_vti.

    The step_ratio / dt upsampling is the shared setup_propagator machinery; a
    dt above the CFL limit must trigger internal upsampling identically for
    acoustic_vti and elastic_vti(c55=0), keeping the two bit-identical.

    Parametrized over ``backend`` (task 7.H milestone H3): the compiled
    forward is exact regardless of ``step_ratio`` (only the compiled
    backward's model gradients are a bounded approximation at
    ``step_ratio > 1``), so this forward-only oracle should hold at the same
    near-machine-precision tolerance for both backends.
    """
    c11, c13, c33, buoyancy = _make_media(2, n=12)
    nt, dt = 40, 0.004  # deliberately coarse -> step_ratio > 1
    amp = deepwave.wavelets.ricker(12.0, nt, dt, 1.5 / 12.0).to(DTYPE).reshape(1, 1, -1)
    src = torch.tensor([[[6, 6]]])
    rec = torch.tensor([[[4, 4]]])
    kw = {
        "source_amplitudes_x": amp,
        "source_locations_x": src,
        "receiver_locations_x": rec,
        "pml_width": 4,
        "accuracy": 4,
        "pml_freq": 12.0,
    }
    av = deepwave.acoustic_vti(
        c11,
        c13,
        c33,
        buoyancy,
        5.0,
        dt,
        backend=backend,
        **kw,
    ).receiver_amplitudes_x
    zero = torch.zeros_like(c11)
    ev = deepwave.elastic_vti(
        c11, c13, c33, zero, buoyancy, 5.0, dt, c66=None, **kw
    ).receiver_amplitudes_x
    assert torch.isfinite(av).all()
    scale = max(av.abs().max().item(), 1e-30)
    assert (av - ev).abs().max().item() <= scale * 1e-12


def test_step_ratio_gt_1_model_gradient_is_the_documented_quadrature():
    """Pins the ``step_ratio > 1`` compiled-vs-eager model-gradient gap.

    docs/what_deepwave_calculates.rst states that the compiled backend's
    model-parameter gradient samples the per-internal-step correlation once
    per ``step_ratio`` block and rescales by ``step_ratio``. For the finite
    record and initial-field objective in this test, that differs from eager
    autograd through every internal step. The sampled inner product is still
    exact for ideal factors strictly inside the data Nyquist band; finite
    starts and endpoints here need not satisfy that condition. This is
    DOCUMENTED, EXPECTED behaviour for this fixture, not a kernel bug -- a real
    ``step_ratio >= 2`` compiled-vs-eager mismatch was once misdiagnosed as
    one, costing a full diagnostic cycle before being root-caused (see
    ``adjoint_fix_prototype/session_8c/`` for that investigation's scripts;
    this test is a standing, lower-cost distillation of its decisive
    arbitration so the same mismatch can never again reach that point).

    Rigor achieved: this reconstructs the EXACT per-internal-step adjoint
    gradient integrand ``g_t`` via a pure-Python replica of the eager
    update loop with a SEPARATE model leaf per internal step (so autograd
    gives each step's contribution individually, which the propagator's own
    public API cannot expose). It then verifies, at ``step_ratio == 2``,
    BOTH sides of the documented relationship to near machine precision:

    * ``sum_t g_t`` (every step's contribution) equals the eager gradient.
    * ``step_ratio * sum_{t % step_ratio == 0} g_t`` (the sampled correlation
      rule the documentation describes) equals the compiled gradient.

    and, as a contrasting control, that at ``step_ratio == 1`` compiled and
    eager already agree to near machine precision (the "clean" case both
    formulas above reduce to when ``step_ratio == 1``). If the compiled
    kernel's gradient sampling ever changes, the ``step_ratio == 2``
    quadrature check here is the one that should catch it -- if this test
    starts failing there, that is a genuine kernel-behaviour change worth
    investigating, not evidence to loosen the tolerance.
    """
    import torch.nn.functional as nn_functional

    from deepwave import _engine, _staggered_grid
    from deepwave.acoustic_vti import (
        _prepare_buoyancy,
        _zero_edges_and_interiors_acoustic_vti,
        update_stresses_acoustic_vti,
        update_velocities_acoustic_vti,
    )

    torch.manual_seed(4)
    n = 8
    ndim = 3
    nt_user = 4
    accuracy = 2
    fd_pad = 1
    fd_pad_list = [1, 0] * ndim
    pml_width = [0] * (2 * ndim)
    gs = [5.0] * ndim
    max_vel = 2000.0
    pml_freq = 15.0

    shape = (n,) * ndim
    base = 1e9 * (1 + 0.1 * torch.rand(*shape, dtype=DTYPE))
    c11_u = 5 * base
    c13_u = 2 * base
    c33_u = 4 * base
    buoy_u = 1 / (2000 * (1 + 0.1 * torch.rand(*shape, dtype=DTYPE)))
    sv0 = torch.zeros(1, *shape, dtype=DTYPE)
    sv0[0, n // 2, n // 2, n // 2] = 1.0
    sv0[0, n // 2 - 1, n // 2, n // 2 + 1] = -0.5
    vx0 = 1e-4 * torch.randn(1, *shape, dtype=DTYPE)

    def loss_of(fields):
        return sum(f.pow(2).sum() for f in fields) * 1e-3

    def run_backend(dt_user, backend, which):
        models = {
            "c11": c11_u.clone(),
            "c13": c13_u.clone(),
            "c33": c33_u.clone(),
            "buoyancy": buoy_u.clone(),
        }
        p = models[which].requires_grad_(True)
        out = deepwave.acoustic_vti(
            models["c11"],
            models["c13"],
            models["c33"],
            models["buoyancy"],
            gs,
            dt_user,
            sigma_v_0=sv0,
            vx_0=vx0,
            nt=nt_user,
            pml_width=0,
            accuracy=accuracy,
            max_vel=max_vel,
            backend=backend,
            pml_freq=pml_freq,
        )
        # Every final state field, with the auxiliary state expanded back into
        # the individual PML memory variables the reference loop below holds.
        loss = loss_of(output_tensors(list(out.final_state.values())))
        (g,) = torch.autograd.grad(loss, (p,))
        return g

    def pad_model(m):
        return nn_functional.pad(m[None, None], (1, 0, 1, 0, 1, 0), mode="replicate")[
            0, 0
        ]

    def per_step_grads(dt_user, step_ratio):
        """Exact per-internal-step adjoint integrand for every model leaf.

        A pure-Python replica of the eager update loop, driven by initial
        wavefields only (no source/receiver resampling), with a SEPARATE
        leaf tensor per internal step so ``torch.autograd`` gives each
        step's contribution to the gradient individually -- the exact
        integrand ``g_t`` of the adjoint time integral that the compiled
        kernel's block-sampling only ever approximates for ``t`` not a
        multiple of ``step_ratio``.
        """
        dt_i = dt_user / step_ratio
        n_int = nt_user * step_ratio
        base_models = {
            "c11": c11_u,
            "c13": c13_u,
            "c33": c33_u,
            "buoyancy": buoy_u,
        }
        leaves = {
            w: [m.clone().requires_grad_(True) for _ in range(n_int)]
            for w, m in base_models.items()
        }
        size_with_batch = (1, *(s + 1 for s in shape))
        names = ["vz", "vy", "vx", "sigma_v", "sigma_h"]
        init = {"sigma_v": sv0, "vx": vx0}
        wfs = [
            _engine.create_or_pad(
                init.get(nm, torch.zeros(0)),
                fd_pad_list,
                torch.device("cpu"),
                DTYPE,
                size_with_batch,
            )
            for nm in names
        ] + [
            _engine.create_or_pad(
                torch.zeros(0), fd_pad_list, torch.device("cpu"), DTYPE, size_with_batch
            )
            for _ in range(2 * ndim)
        ]
        _zero_edges_and_interiors_acoustic_vti(
            wfs, ndim, fd_pad, fd_pad_list, pml_width, mpml=False
        )
        velocities = wfs[:ndim]
        stresses = [wfs[ndim], wfs[ndim + 1]]
        v_mem = wfs[ndim + 2 : 2 * ndim + 2]
        s_mem = wfs[2 * ndim + 2 : 3 * ndim + 2]
        profiles = _staggered_grid.set_pml_profiles(
            pml_width,
            accuracy,
            fd_pad_list,
            dt_i,
            gs,
            max_vel,
            DTYPE,
            torch.device("cpu"),
            pml_freq,
            tuple(s + 1 for s in shape),
        )
        for t in range(n_int):
            models_t = [
                pad_model(leaves["c11"][t]),
                pad_model(leaves["c13"][t]),
                pad_model(leaves["c33"][t]),
                *_prepare_buoyancy(pad_model(leaves["buoyancy"][t])[None]),
            ]
            velocities, s_mem = update_velocities_acoustic_vti(
                models_t,
                stresses,
                velocities,
                s_mem,
                profiles,
                gs,
                dt_i,
                accuracy,
            )
            stresses, v_mem = update_stresses_acoustic_vti(
                models_t,
                velocities,
                stresses,
                v_mem,
                profiles,
                gs,
                dt_i,
                accuracy,
            )
        s = (slice(None), *(slice(fd_pad, sh + 1 - (fd_pad - 1)) for sh in shape))
        finals = [f[s] for f in [*velocities, *stresses, *v_mem, *s_mem]]
        loss = loss_of(finals)
        all_leaves = [leaf for w in leaves for leaf in leaves[w]]
        grads = torch.autograd.grad(loss, all_leaves, allow_unused=True)
        per_step = {}
        i = 0
        for w in leaves:
            per_step[w] = [
                g if g is not None else torch.zeros_like(leaves[w][0])
                for g in grads[i : i + n_int]
            ]
            i += n_int
        return per_step

    def rel(a, b):
        return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # --- step_ratio == 2: the documented approximation ---
        step_ratio = 2
        dt_user = 0.001  # gives step_ratio == 2 at max_vel=2000, dx=5.0
        per_step = per_step_grads(dt_user, step_ratio)
        for which in ("c11", "c13", "c33", "buoyancy"):
            g_eager = run_backend(dt_user, "eager", which)
            g_compiled = run_backend(dt_user, "compiled", which)

            # This finite-record fixture has deliberately non-bandlimited
            # endpoints, so its compiled and eager gradients differ. If they
            # match too closely, investigate a change in the test or kernel.
            assert rel(g_compiled, g_eager) > 1e-3, (
                f"{which}: compiled and eager model gradients matched too "
                "closely at step_ratio=2 -- if the compiled kernel now "
                "computes the exact adjoint here, that is a genuine "
                "behaviour change, not a reason to tighten this guard"
            )

            g_full = sum(per_step[which])
            g_quad = step_ratio * sum(
                per_step[which][t] for t in range(0, len(per_step[which]), step_ratio)
            )
            assert rel(g_full, g_eager) < 1e-10, (
                f"{which}: full per-step sum should equal the exact eager "
                f"adjoint gradient (got rel diff {rel(g_full, g_eager):.2e})"
            )
            assert rel(g_quad, g_compiled) < 1e-10, (
                f"{which}: step_ratio * strided sum should equal the "
                f"compiled gradient (got rel diff {rel(g_quad, g_compiled):.2e})"
            )

        # --- step_ratio == 1: the clean control ---
        dt_safe = 0.0003  # courant-safe at max_vel=2000, dx=5.0 in 3-D
        for which in ("c11", "c13", "c33", "buoyancy"):
            g_eager = run_backend(dt_safe, "eager", which)
            g_compiled = run_backend(dt_safe, "compiled", which)
            assert rel(g_compiled, g_eager) < 1e-10, (
                f"{which}: compiled and eager should match to near machine "
                f"precision at step_ratio == 1 (got rel diff "
                f"{rel(g_compiled, g_eager):.2e})"
            )


# --- compiled backward boundary-face adjoint fix --------------------------
# Regression tests for a bug in which acoustic_vti.c/acoustic_vti.cu's
# compiled backward re-applied the forward boundary guard-ring fill to the
# adjoint fields instead of that fill's true transpose, and the backward's
# per-cell gather kernels only ever ran over the grid interior -- the same
# bug class fixed for elastic.c/elastic.cu and elastic_vti.c/elastic_vti.cu
# (see those files' equivalent tests). Uses a long enough ``nt`` for several
# source-to-face-to-receiver round trips, so a wrong adjoint fill is not
# masked by numerical dust (the short-``nt`` gradient checks elsewhere in
# this file never exercise a non-PML face at all).


@C_BACKEND
@pytest.mark.parametrize(
    ("boundary", "pml_width"),
    [
        (["rigid", "rigid", "pml", "pml"], [0, 0, 10, 10]),
        (["rigid", "pml", "pml", "pml"], [0, 10, 10, 10]),
        (["pml", "rigid", "pml", "pml"], [10, 0, 10, 10]),
        (["pml", "pml", "rigid", "pml"], [10, 10, 0, 10]),
        (["pml", "pml", "pml", "rigid"], [10, 10, 10, 0]),
        (["rigid", "rigid", "rigid", "rigid"], [0, 0, 0, 0]),
    ],
    ids=[
        "rigid_axis_both",
        "rigid_lo_only_axis",
        "rigid_hi_only_axis",
        "rigid_lo_only_transverse",
        "rigid_hi_only_transverse",
        "rigid_every_face",
    ],
)
def test_acoustic_vti_compiled_backward_boundary_adjoint_parity(boundary, pml_width):
    """Compiled backward matches eager for c11/c13/c33/buoyancy/source grads.

    ``acoustic_vti`` accepts ``"rigid"`` and ``"pml"`` faces, so those are
    the combinations covered here.
    Negative control: run against acoustic_vti.c/.cu reverted to the pre-fix
    (committed bd87fd8) state and rebuilt, the mixed rigid/pml
    configuration of this test fails (measured: c11 grad relative error well
    above 1e-9, consistent with the ~1.73% c11-grad baseline this repository's
    B5-fix task notes document for acoustic_vti).
    """
    ny = 16
    dx, dt, nt = 5.0, 0.0001, 1200
    c11_0, c13_0, c33_0, buoyancy0 = _make_media(2, n=ny, eps=0.15, delta=0.05)
    sa0 = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).to(DTYPE).reshape(1, 1, nt)
    kw = {
        "source_locations_y": torch.tensor([[[10, 10]]]),
        "receiver_locations_y": torch.tensor([[[5, 5]]]),
        "pml_width": pml_width,
        "boundary": boundary,
        "accuracy": 4,
        "pml_freq": 25.0,
    }

    def run(backend):
        c11 = c11_0.clone().requires_grad_(True)
        c13 = c13_0.clone().requires_grad_(True)
        c33 = c33_0.clone().requires_grad_(True)
        buoyancy = buoyancy0.clone().requires_grad_(True)
        sa = sa0.clone().requires_grad_(True)
        out = deepwave.acoustic_vti(
            c11,
            c13,
            c33,
            buoyancy,
            dx,
            dt,
            source_amplitudes_y=sa,
            backend=backend,
            **kw,
        ).receiver_amplitudes_y
        return torch.autograd.grad(out.sum(), [c11, c13, c33, buoyancy, sa])

    g_c = run("compiled")
    g_e = run("eager")
    for name, gc, ge in zip(
        ["c11", "c13", "c33", "buoyancy", "source_amplitudes_y"], g_c, g_e
    ):
        denom = ge.abs().max()
        rel = (gc - ge).abs().max() / denom
        assert rel < 1e-9, f"{name} grad rel err {rel.item()} boundary={boundary}"


@C_BACKEND
def test_acoustic_vti_compiled_backward_boundary_adjoint_parity_3d():
    """3-D counterpart of the 2-D adjoint-parity test above, mixed faces."""
    nz = 10
    dx, dt, nt = 6.0, 0.00012, 400
    c11_0, c13_0, c33_0, buoyancy0 = _make_media(3, n=nz, eps=0.12, delta=0.04)
    sa0 = deepwave.wavelets.ricker(20.0, nt, dt, 0.02).to(DTYPE).reshape(1, 1, nt)
    kw = {
        "source_amplitudes_z": sa0,
        "source_locations_z": torch.tensor([[[5, 5, 5]]]),
        "receiver_locations_z": torch.tensor([[[3, 3, 3]]]),
        "pml_width": [0, 0, 0, 0, 5, 5],
        "boundary": ["rigid", "rigid", "rigid", "rigid", "pml", "pml"],
        "accuracy": 4,
        "pml_freq": 20.0,
    }

    def run(backend):
        c11 = c11_0.clone().requires_grad_(True)
        c13 = c13_0.clone().requires_grad_(True)
        c33 = c33_0.clone().requires_grad_(True)
        buoyancy = buoyancy0.clone().requires_grad_(True)
        out = deepwave.acoustic_vti(
            c11, c13, c33, buoyancy, dx, dt, backend=backend, **kw
        ).receiver_amplitudes_z
        return torch.autograd.grad(out.sum(), [c11, c13, c33, buoyancy])

    g_c = run("compiled")
    g_e = run("eager")
    for name, gc, ge in zip(["c11", "c13", "c33", "buoyancy"], g_c, g_e):
        rel = (gc - ge).abs().max() / ge.abs().max()
        assert rel < 1e-9, f"{name} grad rel err {rel.item()} (3-D)"


@C_BACKEND
def test_acoustic_vti_compiled_backward_boundary_fd_arbitration():
    """Central-FD arbitration at the max-|g| cell (independent of eager).

    Rather than trusting eager as ground truth, perturb the model directly
    at the cell with the largest-magnitude eager gradient and compare the
    compiled analytic gradient there against a central finite difference of
    the compiled forward loss.
    """
    ny = 16
    dx, dt, nt = 5.0, 0.0001, 1200
    c11_0, c13_0, c33_0, buoyancy0 = _make_media(2, n=ny, eps=0.15, delta=0.05)
    sa0 = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).to(DTYPE).reshape(1, 1, nt)
    kw = {
        "source_amplitudes_y": sa0,
        "source_locations_y": torch.tensor([[[10, 10]]]),
        "receiver_locations_y": torch.tensor([[[5, 5]]]),
        "pml_width": [0, 0, 10, 10],
        "boundary": ["rigid", "rigid", "pml", "pml"],
        "accuracy": 4,
        "pml_freq": 25.0,
    }

    def loss(c11, backend):
        out = deepwave.acoustic_vti(
            c11, c13_0, c33_0, buoyancy0, dx, dt, backend=backend, **kw
        )
        return out.receiver_amplitudes_y.sum()

    c11_e = c11_0.clone().requires_grad_(True)
    g_e = torch.autograd.grad(loss(c11_e, "eager"), c11_e)[0]
    idx = tuple(int(i) for i in (g_e.abs() == g_e.abs().max()).nonzero()[0])

    c11_c = c11_0.clone().requires_grad_(True)
    g_c = torch.autograd.grad(loss(c11_c, "compiled"), c11_c)[0]

    eps = float(c11_0[idx].item()) * 1e-6
    c11_plus = c11_0.clone()
    c11_plus[idx] += eps
    c11_minus = c11_0.clone()
    c11_minus[idx] -= eps
    fd = (loss(c11_plus, "compiled") - loss(c11_minus, "compiled")) / (2 * eps)

    rel = abs(g_c[idx].item() - fd.item()) / abs(fd.item())
    assert rel < 2e-3, f"compiled grad vs FD rel err {rel} at {idx}"


@C_BACKEND
@pytest.mark.parametrize(
    ("boundary", "pml_width"),
    [
        (["rigid", "rigid", "pml", "pml"], [0, 0, 10, 10]),
        (["rigid", "rigid", "rigid", "rigid"], [0, 0, 0, 0]),
    ],
    ids=["rigid_axis_both", "rigid_every_face"],
)
def test_acoustic_vti_compiled_forward_final_state_matches_eager(boundary, pml_width):
    """Compiled forward's returned final wavefields are boundary-consistent.

    Regression test for a secondary bug: the compiled forward returned the
    post-update, unfilled final sigma_v/sigma_h/velocity state (the last
    time step's field update runs after the last in-loop boundary fill),
    while eager applies a final guard-ring/on-surface fill after its time
    loop (``_acoustic_vti_eager_core``). Without the fix, e.g. the
    on-surface sigma_v/sigma_h nodes disagreed between the two backends
    where eager (and physics) fixes their value.
    """
    ny = 16
    dx, dt, nt = 5.0, 0.0001, 400
    c11, c13, c33, buoyancy = _make_media(2, n=ny, eps=0.15, delta=0.05)
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).to(DTYPE).reshape(1, 1, nt)
    kw = {
        "source_amplitudes_y": sa,
        "source_locations_y": torch.tensor([[[10, 10]]]),
        "receiver_locations_y": torch.tensor([[[5, 5]]]),
        "pml_width": pml_width,
        "boundary": boundary,
        "accuracy": 4,
        "pml_freq": 25.0,
    }
    out_c = deepwave.acoustic_vti(
        c11, c13, c33, buoyancy, dx, dt, backend="compiled", **kw
    )
    out_e = deepwave.acoustic_vti(
        c11, c13, c33, buoyancy, dx, dt, backend="eager", **kw
    )
    for field in ("vy_0", "vx_0", "sigma_v_0", "sigma_h_0"):
        field_c = getattr(out_c, field)
        field_e = getattr(out_e, field)
        denom = field_e.abs().max()
        if denom == 0:
            assert field_c.abs().max() == 0, f"{field}: compiled nonzero, eager zero"
            continue
        rel = (field_c - field_e).abs().max() / denom
        assert rel < 1e-9, f"{field} rel err {rel.item()} boundary={boundary}"


@pytest.mark.parametrize("ndim", [2, 3])
def test_acoustic_vti_backward_callback_dict_keys(ndim) -> None:
    """The backward-callback dict's key set matches acoustic_vti_wavefield_names.

    Previously ``callback_drop_last=1`` dropped the last wavefield name from
    the dict a user's ``backward_callback`` receives (see PR #121).
    """
    from deepwave.acoustic_vti import acoustic_vti_wavefield_names

    n = 8
    shape = (n,) * ndim
    base = torch.full(shape, 1e9)
    c33 = (4 * base).requires_grad_()
    c11 = (6 * base).requires_grad_()
    c13 = (2 * base).requires_grad_()
    buoyancy = torch.full(shape, 1 / 2000.0).requires_grad_()
    amp = torch.zeros(1, 1, 8)
    amp[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded_keys: set[str] = set()

    def recorder(state: deepwave.CallbackState) -> None:
        recorded_keys.update(state._wavefields.keys())

    out = deepwave.acoustic_vti(
        c11,
        c13,
        c33,
        buoyancy,
        5.0,
        0.00025,
        source_amplitudes_x=amp,
        source_locations_x=loc,
        receiver_locations_x=loc,
        pml_width=3,
        backward_callback=recorder,
        callback_frequency=1,
    )
    out.receiver_amplitudes_x.sum().backward()
    assert recorded_keys == set(acoustic_vti_wavefield_names(ndim))


def test_eager_rejects_storage_compression() -> None:
    """backend="eager" rejects storage_compression=True (see PR #121)."""
    n = 8
    base = torch.full((n, n), 1e9)
    c33, c11, c13 = 4 * base, 6 * base, 2 * base
    buoyancy = torch.full((n, n), 1 / 2000.0)
    amp = torch.zeros(1, 1, 8)
    loc = torch.zeros(1, 1, 2, dtype=torch.long)
    with pytest.raises(RuntimeError, match="Storage compression"):
        deepwave.acoustic_vti(
            c11,
            c13,
            c33,
            buoyancy,
            5.0,
            0.00025,
            source_amplitudes_x=amp,
            source_locations_x=loc,
            receiver_locations_x=loc,
            backend="eager",
            storage_compression=True,
        )
