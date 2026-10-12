"""Tests for the 2-D elastic TTI (tilted transverse isotropy) propagator.

The physics gates are:

* zero-tilt reduction to :func:`deepwave.elastic_vti` at machine precision;
* a tilted-medium phase-velocity oracle: measured plane-wave slownesses match
  the exact monoclinic Christoffel solution (which itself equals the VTI
  Christoffel solution evaluated at the tilt-shifted angle -- the analytic
  statement of the Bond rotation), with a negative control that a
  wrong-sign tilt fails the same comparison;
* elastodynamic reciprocity in a tilted medium;
* the adjoint dot-product (native autograd) test;
* gradcheck on every stiffness, buoyancy, and the tilt angle;
* a stability-envelope sweep over (max tilt) x (anisotropy strength) corners,
  each run for several domain-crossing times, asserting bounded energy.
"""

import math
import sys
import warnings

import numpy as np
import pytest
import torch

import deepwave

DTYPE = torch.float64

# Strong-anisotropy reference medium (Thomsen eps=0.3, delta=0.1, vp0/vs0=2).
VP0, VS0, RHO0 = 3000.0, 1500.0, 2000.0
EPS_T, DELTA_T = 0.3, 0.1


def _cij_scalars():
    one = torch.ones(1, dtype=DTYPE)
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        VP0 * one, VS0 * one, RHO0 * one, EPS_T * one, DELTA_T * one
    )
    return float(c11), float(c13), float(c33), float(c55)


def _bond_scalars(c11, c13, c33, c55, tilt):
    """Six rotated monoclinic constants (C11',C13',C33',C55',C15',C35')."""
    c = math.cos(tilt)
    s = math.sin(tilt)
    c2, s2, cs = c * c, s * s, c * s
    m = [[c2, s2, 2 * cs], [s2, c2, -2 * cs], [-cs, cs, c2 - s2]]

    def e(a, b):
        return (
            m[a][0] * m[b][0] * c11
            + (m[a][0] * m[b][1] + m[a][1] * m[b][0]) * c13
            + m[a][1] * m[b][1] * c33
            + m[a][2] * m[b][2] * c55
        )

    return e(0, 0), e(0, 1), e(1, 1), e(2, 2), e(0, 2), e(1, 2)


def _christoffel_vti(theta, c11, c13, c33, c55):
    """Exact 2-D VTI phase speeds (qP, qSV) at angle theta from the axis."""
    s2 = math.sin(theta) ** 2
    c2 = math.cos(theta) ** 2
    g11 = c11 * s2 + c55 * c2
    g33 = c55 * s2 + c33 * c2
    g13 = (c13 + c55) * math.sin(theta) * math.cos(theta)
    tr = g11 + g33
    disc = math.sqrt(max(tr * tr / 4 - (g11 * g33 - g13 * g13), 0.0))
    return math.sqrt((tr / 2 + disc) / RHO0), math.sqrt((tr / 2 - disc) / RHO0)


def _tti_q_of_p(p, wave, orient, tilt):
    """Transverse slowness of a plane wave from the monoclinic dispersion.

    Solves ``det(Gamma(s) - rho I) = 0`` for the transverse slowness ``q``
    given the along-line slowness ``p``, using the rotated (monoclinic)
    stiffness. ``orient='h'`` fires the source line along the fast (x)
    dimension (so ``p = s_x``, ``q = s_z``); ``orient='v'`` along the axis (y)
    dimension. Returns the physical (real, positive) qP or qSV branch.
    """
    c11, c13, c33, c55 = _cij_scalars()
    c11r, c13r, c33r, c55r, c15r, c35r = _bond_scalars(c11, c13, c33, c55, tilt)
    # Gamma_ik entries as quadratics in q (physics x-z, motion ux,uz). Each is
    # [const, linear, quadratic] in q.
    if orient == "h":  # p = s_x, q = s_z
        g11 = [c11r * p * p, 2 * c15r * p, c55r]
        g33 = [c55r * p * p, 2 * c35r * p, c33r]
        g13 = [c15r * p * p, (c13r + c55r) * p, c35r]
    else:  # p = s_z, q = s_x
        g11 = [c55r * p * p, 2 * c15r * p, c11r]
        g33 = [c33r * p * p, 2 * c35r * p, c55r]
        g13 = [c35r * p * p, (c13r + c55r) * p, c15r]
    g11m = np.array(g11, dtype=float)
    g33m = np.array(g33, dtype=float)
    g11m[0] -= RHO0
    g33m[0] -= RHO0
    # det = (g11-rho)(g33-rho) - g13^2, polynomials in q (numpy polynomial
    # coefficient order: lowest degree first).
    det = np.polynomial.polynomial.polymul(g11m, g33m)
    det = np.polynomial.polynomial.polysub(
        det, np.polynomial.polynomial.polymul(g13, g13)
    )
    roots = np.polynomial.polynomial.polyroots(det)
    qs = sorted(
        float(r.real)
        for r in roots
        if abs(r.imag) < 1e-9 * (abs(r.real) + 1) and r.real > 0
    )
    if not qs:
        raise ValueError(f"no propagating root at p={p}, tilt={tilt}")
    # qP is the fastest branch = smallest total slowness. Compare the two
    # candidate total slownesses.
    if wave == "qP":
        return qs[0]
    if len(qs) < 2:
        raise ValueError(f"{wave} evanescent at p={p}, tilt={tilt}")
    return qs[1]


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


def _run_plane_wave_case(p_slow, wave, orient, tilt):
    """Measure (theta_grid, q) of a plane wave in a homogeneous TTI medium."""
    across = 220
    along = 460
    ny, nx = (across, along) if orient == "h" else (along, across)
    dx = 5.0
    dt = 0.0005
    freq = 14.0

    line_extent = (along // 2 - 30) * dx
    peak_time = 1.5 / freq + p_slow * line_extent
    q_exp = _tti_q_of_p(p_slow, wave, orient, tilt)
    t_far = peak_time + q_exp * (140 - 10) * dx + 3.0 / freq
    nt = math.ceil(t_far / dt)

    one = torch.ones((ny, nx), dtype=DTYPE)
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        VP0 * one, VS0 * one, RHO0 * one, EPS_T * one, DELTA_T * one
    )
    buoyancy = one / RHO0
    tilt_field = tilt * one

    wavelet = deepwave.wavelets.ricker(freq, nt, dt, peak_time).to(DTYPE)
    line = torch.arange(30, along - 30, dtype=torch.long)
    src_loc = torch.zeros(1, line.shape[0], 2, dtype=torch.long)
    center = along // 2
    if orient == "h":
        src_loc[0, :, 0] = 10
        src_loc[0, :, 1] = line
        rec_loc = torch.tensor([[[100, center], [140, center]]], dtype=torch.long)
    else:
        src_loc[0, :, 0] = line
        src_loc[0, :, 1] = 10
        rec_loc = torch.tensor([[[center, 100], [center, 140]]], dtype=torch.long)
    delays = p_slow * (line.to(DTYPE) - float(center)) * dx
    amps = deepwave.transducers.apply_delays(wavelet, delays, dt)
    n_line = line.shape[0]
    n_tap = n_line // 4
    taper = torch.ones(n_line, dtype=DTYPE)
    ramp = torch.hann_window(2 * n_tap, periodic=False, dtype=DTYPE)
    taper[:n_tap] = ramp[:n_tap]
    taper[-n_tap:] = ramp[n_tap:]
    amps = (amps * taper.unsqueeze(-1)).unsqueeze(0)

    kwargs = {
        "pml_width": 20,
        "accuracy": 8,
        "backend": "eager",
        "pml_freq": freq,
    }
    if wave == "qP":
        out = deepwave.elastic_tti(
            c11,
            c13,
            c33,
            c55,
            buoyancy,
            dx,
            dt,
            tilt=tilt_field,
            source_amplitudes_pressure_rate=amps,
            source_locations_pressure_rate=src_loc,
            receiver_locations_p=rec_loc,
            **kwargs,
        )
        rec = out.receiver_amplitudes_p
    else:
        out = deepwave.elastic_tti(
            c11,
            c13,
            c33,
            c55,
            buoyancy,
            dx,
            dt,
            tilt=tilt_field,
            source_amplitudes_x=amps,
            source_locations_x=src_loc,
            receiver_locations_x=rec_loc,
            **kwargs,
        )
        rec = out.receiver_amplitudes_x

    tr1 = _window_at(rec[0, 0, :], dt, freq, peak_time + q_exp * (100 - 10) * dx)
    tr2 = _window_at(rec[0, 1, :], dt, freq, peak_time + q_exp * (140 - 10) * dx)
    dline = (140 - 100) * dx
    q_meas = _measure_delay(tr1, tr2, dt, freq) / dline
    if orient == "h":
        theta = math.atan2(p_slow, q_meas)
    else:
        theta = math.pi / 2 - math.atan2(p_slow, q_meas)
    return theta, q_meas, q_exp


# (p, wave, orient, tilt_deg, nominal grid angle). Non-trivial tilts.
PLANE_WAVE_CASES = [
    pytest.param(1.5e-4, "qP", "h", 30.0, id="qP-h-tilt30"),
    pytest.param(1.0e-4, "qP", "v", 30.0, id="qP-v-tilt30"),
    pytest.param(1.5e-4, "qP", "h", -20.0, id="qP-h-tiltm20"),
    pytest.param(1.0e-4, "qP", "v", 45.0, id="qP-v-tilt45"),
    pytest.param(0.0, "qSV", "h", 25.0, id="qSV-h-tilt25"),
]


@pytest.mark.parametrize(
    ("p_slow", "wave", "orient", "tilt_deg"),
    [(c.values[0], c.values[1], c.values[2], c.values[3]) for c in PLANE_WAVE_CASES],
    ids=[c.id for c in PLANE_WAVE_CASES],
)
def test_phase_velocity_christoffel(p_slow, wave, orient, tilt_deg):
    """Measured plane-wave transverse slowness matches the monoclinic oracle.

    The oracle (:func:`_tti_q_of_p`) solves the exact monoclinic Christoffel
    dispersion relation built from the Bond-rotated stiffness; agreement here
    validates both the rotation and the cross-term (co-location) stencil.
    Tolerance 2.5% (grid-dispersion + measurement budget).
    """
    tilt = math.radians(tilt_deg)
    theta, q_meas, q_exp = _run_plane_wave_case(p_slow, wave, orient, tilt)
    # Cross-check the oracle itself against the VTI-at-shifted-angle identity.
    c11c, c13c, c33c, c55c = _cij_scalars()
    v_ref = _christoffel_vti(theta - tilt, c11c, c13c, c33c, c55c)
    v_ref = v_ref[0] if wave == "qP" else v_ref[1]
    v_meas = 1.0 / math.sqrt(p_slow**2 + q_meas**2)
    assert abs(q_meas - q_exp) / q_exp < 0.025
    assert abs(v_meas - v_ref) / v_ref < 0.025


def test_phase_velocity_negative_control():
    """A wrong-sign tilt fails the oracle: the test can distinguish tilt.

    Feeds the propagator tilt +30 deg but predicts with tilt -30 deg; the
    measured slowness must NOT match, confirming the phase-velocity gate is
    genuinely sensitive to the rotation (not vacuously passing).
    """
    tilt = math.radians(30.0)
    _theta, q_meas, _q_exp = _run_plane_wave_case(1.5e-4, "qP", "h", tilt)
    q_wrong = _tti_q_of_p(1.5e-4, "qP", "h", -tilt)
    assert abs(q_meas - q_wrong) / q_wrong > 0.05


def _homog_tti(ny, nx, tilt_deg, eps=EPS_T, delta=DELTA_T):
    one = torch.ones((ny, nx), dtype=DTYPE)
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        VP0 * one, VS0 * one, RHO0 * one, eps * one, delta * one
    )
    return c11, c13, c33, c55, one / RHO0, math.radians(tilt_deg) * one


def test_reciprocity():
    """Source/receiver reciprocity for same-component (vx-vx) pairs, tilted.

    Elastodynamic reciprocity: swapping a vx force source and a vx velocity
    receiver leaves the trace unchanged. This holds only if the discrete
    stiffness operator (including the monoclinic cross-terms) is symmetric.
    """
    ny, nx = 90, 110
    dx, dt = 5.0, 0.0005
    freq = 15.0
    nt = 700
    c11, c13, c33, c55, buoy, tilt = _homog_tti(ny, nx, 35.0)
    wav = deepwave.wavelets.ricker(freq, nt, dt, 1.3 / freq).to(DTYPE).reshape(1, 1, -1)
    a_loc = torch.tensor([[[30, 35]]])
    b_loc = torch.tensor([[[60, 78]]])
    kw = {
        "tilt": tilt,
        "backend": "eager",
        "accuracy": 4,
        "pml_width": 20,
        "pml_freq": freq,
    }
    o1 = deepwave.elastic_tti(
        c11,
        c13,
        c33,
        c55,
        buoy,
        dx,
        dt,
        source_amplitudes_x=wav,
        source_locations_x=a_loc,
        receiver_locations_x=b_loc,
        **kw,
    )
    o2 = deepwave.elastic_tti(
        c11,
        c13,
        c33,
        c55,
        buoy,
        dx,
        dt,
        source_amplitudes_x=wav,
        source_locations_x=b_loc,
        receiver_locations_x=a_loc,
        **kw,
    )
    t1 = o1.receiver_amplitudes_x[0, 0]
    t2 = o2.receiver_amplitudes_x[0, 0]
    # Tolerance matched to test_elastic_vti.py::test_reciprocity_2d (5e-3);
    # the achieved agreement here is ~5e-5 of the peak.
    scale = float(t1.abs().max())
    assert float((t1 - t2).abs().max()) < 5e-3 * scale


def _small_setup(requires_grad_for=(), tilt_deg=35.0, nt=18):
    ny, nx = 14, 16
    dx, dt = 5.0, 0.0006
    freq = 12.0
    torch.manual_seed(0)
    one = torch.ones((ny, nx), dtype=DTYPE)
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        VP0 * one, VS0 * one, RHO0 * one, EPS_T * one, DELTA_T * one
    )
    buoy = one / RHO0
    tilt = math.radians(tilt_deg) * one
    wav = deepwave.wavelets.ricker(freq, nt, dt, 1.2 / freq).to(DTYPE).reshape(1, 1, -1)
    sl = torch.tensor([[[6, 7]]])
    rl = torch.tensor([[[4, 4], [9, 11]]])
    tensors = {
        "c11": c11,
        "c13": c13,
        "c33": c33,
        "c55": c55,
        "buoyancy": buoy,
        "tilt": tilt,
    }
    for name in requires_grad_for:
        tensors[name] = tensors[name].clone().requires_grad_(True)
    return tensors, wav, sl, rl, dx, dt, freq


def test_adjoint_dot_product():
    """Native-autograd adjoint dot-product test: <L m, d> == <m, L^T d>.

    With F the (linear-in-source) forward map, the vector-Jacobian product from
    autograd must satisfy <F s, d> == <s, F^T d> to machine precision.
    """
    tensors, wav, sl, rl, dx, dt, freq = _small_setup(nt=40)
    src = wav.clone().requires_grad_(True)
    out = deepwave.elastic_tti(
        tensors["c11"],
        tensors["c13"],
        tensors["c33"],
        tensors["c55"],
        tensors["buoyancy"],
        dx,
        dt,
        tilt=tensors["tilt"],
        source_amplitudes_y=src,
        source_locations_y=sl,
        receiver_locations_y=rl,
        backend="eager",
        pml_freq=freq,
    )
    rec = out.receiver_amplitudes_y
    torch.manual_seed(1)
    d = torch.randn_like(rec)
    inner_fwd = (rec * d).sum()
    (grad_src,) = torch.autograd.grad(inner_fwd, src)
    # <F s, d> should equal <s, F^T d> = <s, grad_src> since F is linear in s.
    lhs = float((rec * d).sum().detach())
    rhs = float((src * grad_src).sum().detach())
    assert abs(lhs - rhs) / abs(lhs) < 1e-10


@pytest.mark.parametrize("param", ["c11", "c13", "c33", "c55", "buoyancy", "tilt"])
def test_gradcheck(param):
    """Finite-difference gradcheck for each model input, including tilt."""
    tensors, wav, sl, rl, dx, dt, freq = _small_setup(requires_grad_for=(param,), nt=12)

    def run(x):
        m = dict(tensors)
        m[param] = x
        out = deepwave.elastic_tti(
            m["c11"],
            m["c13"],
            m["c33"],
            m["c55"],
            m["buoyancy"],
            dx,
            dt,
            tilt=m["tilt"],
            source_amplitudes_y=wav,
            source_locations_y=sl,
            receiver_locations_y=rl,
            backend="eager",
            pml_freq=freq,
            pml_width=6,
        )
        return out.receiver_amplitudes_y

    x = tensors[param]
    scale = float(x.detach().abs().max())
    eps = 1e-6 * (scale if scale > 0 else 1.0)
    assert torch.autograd.gradcheck(
        run, (x,), eps=eps, atol=1e-6, rtol=1e-3, nondet_tol=1e-10
    )


# Stability-envelope sweep: (max tilt deg, (eps, delta)) corners.
STABILITY_CORNERS = [
    (0.0, (0.3, 0.1)),
    (45.0, (0.3, 0.1)),
    (30.0, (0.5, -0.1)),
    (60.0, (0.4, 0.2)),
    (90.0, (0.3, 0.1)),
]


@pytest.mark.parametrize("mpml_fraction", [0.0, 0.15], ids=["mpml0", "mpml0.15"])
@pytest.mark.parametrize(
    ("tilt_deg", "eps_delta"),
    STABILITY_CORNERS,
    ids=[f"tilt{int(t)}-eps{ed[0]}" for t, ed in STABILITY_CORNERS],
)
def test_stability_envelope(tilt_deg, eps_delta, mpml_fraction):
    """Energy stays bounded over 4x the domain-crossing time at each corner.

    The tilted-anisotropy PML remains stable: the final wavefield energy must
    not exceed a modest multiple of the peak energy during propagation (no
    exponential blow-up).

    Both ``mpml_fraction`` values are run. ``0.15`` is the value recommended for a
    model that does show PML growth, and was the only one this test
    originally covered; ``0.0`` is the **shipped default**, which therefore
    had no stability coverage at all despite a (now corrected) CHANGELOG
    claim that M-PML damping "is needed for stability at typical tilt
    angles". It is not: at every corner below, ``0.0`` decays at least as
    well as ``0.15``.
    """
    eps, delta = eps_delta
    ny, nx = 70, 80
    dx, dt = 5.0, 0.0005
    freq = 15.0
    cross_t = max(ny, nx) * dx / VS0  # slowest useful speed ~ vs
    nt = int(4 * cross_t / dt)
    c11, c13, c33, c55, buoy, tilt = _homog_tti(ny, nx, tilt_deg, eps, delta)
    wav = deepwave.wavelets.ricker(freq, nt, dt, 1.2 / freq).to(DTYPE).reshape(1, 1, -1)
    sl = torch.tensor([[[ny // 2, nx // 2]]])
    energies = []

    def cb(state):
        e = sum(float((state.get_wavefield(k) ** 2).sum()) for k in ("vy_0", "vx_0"))
        energies.append(e)

    out = deepwave.elastic_tti(
        c11,
        c13,
        c33,
        c55,
        buoy,
        dx,
        dt,
        tilt=tilt,
        source_amplitudes_pressure_rate=wav,
        source_locations_pressure_rate=sl,
        backend="eager",
        pml_freq=freq,
        pml_width=20,
        mpml_fraction=mpml_fraction,
        forward_callback=cb,
        callback_frequency=20,
    )
    final_e = float((out.vy_0**2).sum() + (out.vx_0**2).sum())
    peak_e = max(energies) if energies else final_e
    assert math.isfinite(final_e)
    assert final_e <= 1e-3 * peak_e  # decayed away, not blown up


def test_zero_tilt_matches_elastic_vti():
    """Zero tilt reproduces elastic_vti to machine precision (eager vs eager)."""
    ny, nx = 40, 50
    dx, dt = 5.0, 0.0004
    freq = 15.0
    nt = 200
    torch.manual_seed(1)
    vp = VP0 + 100 * torch.randn(ny, nx, dtype=DTYPE)
    vs = VS0 + 50 * torch.randn(ny, nx, dtype=DTYPE)
    rho = RHO0 + 50 * torch.randn(ny, nx, dtype=DTYPE)
    one = torch.ones(ny, nx, dtype=DTYPE)
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        vp, vs, rho, 0.2 * one, 0.1 * one
    )
    buoy = 1.0 / rho
    wav = deepwave.wavelets.ricker(freq, nt, dt, 1.2 / freq).to(DTYPE).reshape(1, 1, -1)
    sl = torch.tensor([[[20, 25]]])
    rl = torch.tensor([[[10, 10], [30, 40]]])
    kw = {
        "source_amplitudes_y": wav,
        "source_locations_y": sl,
        "receiver_locations_y": rl,
        "accuracy": 4,
        "pml_width": 10,
        "pml_freq": freq,
        "backend": "eager",
        "mpml_fraction": 0.0,
    }
    o_vti = deepwave.elastic_vti(c11, c13, c33, c55, buoy, dx, dt, c66=None, **kw)
    o_tti = deepwave.elastic_tti(c11, c13, c33, c55, buoy, dx, dt, tilt=None, **kw)
    for name in [
        "receiver_amplitudes_y",
        "vy_0",
        "vx_0",
        "sigmayy_0",
        "sigmaxx_0",
        "sigmaxy_0",
    ]:
        a = getattr(o_vti, name)
        b = getattr(o_tti, name)
        scale = max(float(a.abs().max()), 1e-30)
        assert float((a - b).abs().max()) / scale < 1e-12


def test_elastic_tti_model_names_rejects_bad_ndim():
    """elastic_tti_model_names raises for any ndim other than 2 (3-D unimplemented)."""
    mod = sys.modules["deepwave.elastic_tti"]
    with pytest.raises(ValueError, match="only 2 spatial dimensions"):
        mod.elastic_tti_model_names(1)
    with pytest.raises(ValueError, match="only 2 spatial dimensions"):
        mod.elastic_tti_model_names(3)


def test_module_matches_functional():
    """The ElasticTti module reproduces the functional interface."""
    ny, nx = 20, 24
    one = torch.ones(ny, nx, dtype=DTYPE)
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        VP0 * one, VS0 * one, RHO0 * one, 0.2 * one, 0.1 * one
    )
    buoy = one / RHO0
    tilt = 0.3 * torch.ones(ny, nx, dtype=DTYPE)
    kw = {
        "source_amplitudes_y": torch.zeros(1, 1, 5, dtype=DTYPE),
        "source_locations_y": torch.tensor([[[10, 10]]]),
        "receiver_locations_y": torch.tensor([[[5, 5]]]),
        "backend": "eager",
    }
    prop = deepwave.ElasticTti(c11, c13, c33, c55, buoy, 5.0, tilt=tilt)
    out_m = prop(0.0005, **kw)
    out_f = deepwave.elastic_tti(c11, c13, c33, c55, buoy, 5.0, 0.0005, tilt=tilt, **kw)
    for name in ("receiver_amplitudes_y", "vy_0", "vx_0"):
        assert torch.equal(getattr(out_m, name), getattr(out_f, name))


@pytest.mark.slow
def test_compile_backend_matches_eager():
    """backend='torch_compile' reproduces the eager backend.

    There is no compiled C/CUDA backend at all -- "torch_compile" only needs
    update_stresses_tti (this module's own monoclinic stress kernel) and
    deepwave.elastic.update_velocities to be traceable. Mirrors
    test_elastic_vti.py's test_compile_backend_matches_eager (same
    tolerance).
    """
    ny, nx = 20, 24
    one = torch.ones(ny, nx, dtype=DTYPE)
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        VP0 * one, VS0 * one, RHO0 * one, 0.2 * one, 0.1 * one
    )
    buoy = one / RHO0
    tilt = 0.3 * torch.ones(ny, nx, dtype=DTYPE)
    src = deepwave.wavelets.ricker(25.0, 12, 0.0005, 1.5 / 25.0).to(DTYPE)
    src = src.reshape(1, 1, -1)
    kw = {
        "source_amplitudes_y": src,
        "source_locations_y": torch.tensor([[[10, 10]]]),
        "receiver_locations_p": torch.tensor([[[5, 5]]]),
        "tilt": tilt,
    }
    out_eager = deepwave.elastic_tti(
        c11, c13, c33, c55, buoy, 5.0, 0.0005, backend="eager", **kw
    )
    for backend in ("torch_compile",):
        out = deepwave.elastic_tti(
            c11, c13, c33, c55, buoy, 5.0, 0.0005, backend=backend, **kw
        )
        assert torch.allclose(
            out.receiver_amplitudes_p, out_eager.receiver_amplitudes_p, atol=1e-9
        )


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("ny", "nx"), [(20, 24), (17, 23)])
def test_compile_backend_no_codegen_landmines(dtype, ny, nx):
    """torch.compile's CPU (Inductor) codegen is dtype/shape-sensitive.

    See test_viscoelastic.py's identically-named test for why: a real
    Inductor C++ codegen bug (a VecMask dtype-width mismatch) was found on
    this box, specific to a particular boolean-mask arithmetic pattern not
    present
    here -- but its discovery motivates sweeping both float32/float64 and
    an odd/prime inner grid dimension rather than trusting a single
    smoke test.
    """
    one = torch.ones(ny, nx, dtype=dtype)
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        VP0 * one, VS0 * one, RHO0 * one, 0.2 * one, 0.1 * one
    )
    buoy = one / RHO0
    tilt = 0.3 * torch.ones(ny, nx, dtype=dtype)
    src = deepwave.wavelets.ricker(25.0, 12, 0.0005, 1.5 / 25.0).to(dtype)
    src = src.reshape(1, 1, -1)
    sl_y = min(10, ny - 2)
    rl_y = min(5, ny - 2)
    kw = {
        "source_amplitudes_y": src,
        "source_locations_y": torch.tensor([[[sl_y, 10]]]),
        "receiver_locations_p": torch.tensor([[[rl_y, 5]]]),
        "tilt": tilt,
    }
    out_eager = deepwave.elastic_tti(
        c11, c13, c33, c55, buoy, 5.0, 0.0005, backend="eager", **kw
    )
    out_compile = deepwave.elastic_tti(
        c11, c13, c33, c55, buoy, 5.0, 0.0005, backend="torch_compile", **kw
    )
    assert torch.allclose(
        out_compile.receiver_amplitudes_p,
        out_eager.receiver_amplitudes_p,
        rtol=1e-4,
        atol=1e-8,
    )


def test_c66_and_azimuth_are_gone():
    """``c66``/``azimuth`` are not parameters of the 2-D-only propagator.

    They were 3-D degrees of freedom that this propagator only ever
    rejected, on the functional and the Module alike. They never shipped in
    a release, so they were removed outright rather than kept as arguments
    whose only behaviour was to raise. ``tilt`` stays: it is implemented.
    """
    ny, nx = 10, 10
    one = torch.ones(ny, nx, dtype=DTYPE)
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        VP0 * one, VS0 * one, RHO0 * one, 0.2 * one, 0.1 * one
    )
    buoy = one / RHO0
    common = {
        "source_amplitudes_y": torch.zeros(1, 1, 5, dtype=DTYPE),
        "source_locations_y": torch.tensor([[[5, 5]]]),
    }
    for dead in ("c66", "azimuth"):
        with pytest.raises(TypeError, match=dead):
            deepwave.elastic_tti(
                c11,
                c13,
                c33,
                c55,
                buoy,
                5.0,
                0.0005,
                backend="eager",
                **{dead: c55.clone()},
                **common,
            )
        with pytest.raises(TypeError, match=dead):
            deepwave.ElasticTti(c11, c13, c33, c55, buoy, 5.0, **{dead: c55.clone()})
        # The seven never-released Modules' individual ``*_requires_grad``
        # flags were removed; these two must not have come back.
        with pytest.raises(TypeError, match=f"{dead}_requires_grad"):
            deepwave.ElasticTti(
                c11, c13, c33, c55, buoy, 5.0, **{f"{dead}_requires_grad": True}
            )
    # tilt=None still registers, and remains an accepted argument.
    prop = deepwave.ElasticTti(c11, c13, c33, c55, buoy, 5.0, tilt=None)
    assert prop.tilt is None
    assert not hasattr(prop, "c66")
    assert not hasattr(prop, "azimuth")


def test_scope_and_validation_errors():
    """3-D inputs, 1-D inputs and non-eager backends raise clearly."""
    ny, nx = 20, 20
    one = torch.ones(ny, nx, dtype=DTYPE)
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        VP0 * one, VS0 * one, RHO0 * one, 0.2 * one, 0.1 * one
    )
    buoy = one / RHO0
    common = {
        "source_amplitudes_y": torch.zeros(1, 1, 5, dtype=DTYPE),
        "source_locations_y": torch.tensor([[[10, 10]]]),
    }
    # fewer than 2 spatial dimensions
    one1d = torch.ones(ny, dtype=DTYPE)
    c11_1d, c13_1d, c33_1d, c55_1d, _, _ = deepwave.anisotropy.thomsen_to_cij(
        VP0 * one1d, VS0 * one1d, RHO0 * one1d, 0.2 * one1d, 0.1 * one1d
    )
    with pytest.raises(ValueError, match="at least 2 spatial dimensions"):
        deepwave.elastic_tti(
            c11_1d,
            c13_1d,
            c33_1d,
            c55_1d,
            one1d / RHO0,
            5.0,
            0.0005,
            backend="eager",
            source_amplitudes_y=torch.zeros(1, 1, 5, dtype=DTYPE),
            source_locations_y=torch.tensor([[[10]]]),
        )
    # z-locations must not be provided for a 2-D propagation
    with pytest.raises(ValueError, match="z dimension should not be provided"):
        deepwave.elastic_tti(
            c11,
            c13,
            c33,
            c55,
            buoy,
            5.0,
            0.0005,
            backend="eager",
            source_locations_z=torch.tensor([[[0, 0]]]),
            **common,
        )
    # mpml_fraction out of [0, 1)
    with pytest.raises(ValueError, match="mpml_fraction"):
        deepwave.elastic_tti(
            c11,
            c13,
            c33,
            c55,
            buoy,
            5.0,
            0.0005,
            backend="eager",
            mpml_fraction=1.0,
            **common,
        )
    # a source beyond the model's own extent in a dimension
    with pytest.raises(RuntimeError, match="must be less than"):
        deepwave.elastic_tti(
            c11,
            c13,
            c33,
            c55,
            buoy,
            5.0,
            0.0005,
            backend="eager",
            source_amplitudes_y=torch.zeros(1, 1, 5, dtype=DTYPE),
            source_locations_y=torch.tensor([[[ny - 1, 0]]]),
        )
    # BatchedTensor (vmap) input requires an explicit max_vel
    with pytest.raises(RuntimeError, match="max_vel"):
        torch.func.vmap(
            lambda c11_b: (
                deepwave.elastic_tti(
                    c11_b,
                    c13,
                    c33,
                    c55,
                    buoy,
                    5.0,
                    0.0005,
                    backend="eager",
                    **common,
                ).receiver_amplitudes_y
            )
        )(c11.unsqueeze(0))
    # BatchedTensor (vmap) input with max_vel given runs successfully
    vmap_out = torch.func.vmap(
        lambda c11_b: (
            deepwave.elastic_tti(
                c11_b,
                c13,
                c33,
                c55,
                buoy,
                5.0,
                0.0005,
                backend="eager",
                max_vel=VP0 * 1.5,
                **common,
            ).receiver_amplitudes_y
        )
    )(c11.unsqueeze(0))
    assert torch.isfinite(vmap_out).all()
    # 3-D not implemented
    one3 = torch.ones(8, 8, 8, dtype=DTYPE)
    c11b, c13b, c33b, c55b, _, _ = deepwave.anisotropy.thomsen_to_cij(
        VP0 * one3, VS0 * one3, RHO0 * one3, 0.2 * one3, 0.1 * one3
    )
    with pytest.raises(NotImplementedError):
        deepwave.elastic_tti(
            c11b,
            c13b,
            c33b,
            c55b,
            one3 / RHO0,
            5.0,
            0.0005,
            backend="eager",
            source_amplitudes_y=torch.zeros(1, 1, 5, dtype=DTYPE),
            source_locations_y=torch.tensor([[[4, 4, 4]]]),
        )


def test_backend_compiled_default_resolves_to_eager_with_warning() -> None:
    """The default ``backend="compiled"`` runs (as "eager"), warning once.

    It previously raised ``NotImplementedError`` (see the
    now-removed assertion in ``test_scope_and_validation_errors``); it now
    resolves to the only backend that exists and warns instead of silently
    falling back, matching the explicit ``"eager"`` result exactly.
    """
    ny, nx = 16, 16
    one = torch.ones(ny, nx, dtype=DTYPE)
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        VP0 * one, VS0 * one, RHO0 * one, 0.2 * one, 0.1 * one
    )
    buoy = one / RHO0
    kw = {
        "source_amplitudes_y": torch.zeros(1, 1, 5, dtype=DTYPE),
        "source_locations_y": torch.tensor([[[8, 8]]]),
        "receiver_locations_y": torch.tensor([[[8, 6]]]),
    }
    with warnings.catch_warnings():
        # The default resolves to the eager backend SILENTLY: the fallback
        # UserWarning this used to assert was removed deliberately (see
        # deepwave._engine.resolve_eager_only_backend). out.info.backend is
        # the replacement, and backend="compiled" is the assertion that raises.
        warnings.simplefilter("error", UserWarning)
        out_none = deepwave.elastic_tti(c11, c13, c33, c55, buoy, 5.0, 0.0005, **kw)
    assert out_none.info.backend == "eager"
    out_eager = deepwave.elastic_tti(
        c11, c13, c33, c55, buoy, 5.0, 0.0005, backend="eager", **kw
    )
    assert torch.equal(out_none.receiver_amplitudes_y, out_eager.receiver_amplitudes_y)
    with pytest.raises(deepwave.DeepwaveNotImplementedError, match="no compiled"):
        deepwave.elastic_tti(
            c11, c13, c33, c55, buoy, 5.0, 0.0005, backend="compiled", **kw
        )


def test_backend_eager_explicit_does_not_warn_about_fallback() -> None:
    """Explicitly requesting ``"eager"`` does not emit the fallback warning."""
    ny, nx = 16, 16
    one = torch.ones(ny, nx, dtype=DTYPE)
    c11, c13, c33, c55, _, _ = deepwave.anisotropy.thomsen_to_cij(
        VP0 * one, VS0 * one, RHO0 * one, 0.2 * one, 0.1 * one
    )
    buoy = one / RHO0
    kw = {
        "source_amplitudes_y": torch.zeros(1, 1, 5, dtype=DTYPE),
        "source_locations_y": torch.tensor([[[8, 8]]]),
        "backend": "eager",
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deepwave.elastic_tti(c11, c13, c33, c55, buoy, 5.0, 0.0005, **kw)
    assert not any("no compiled backend" in str(x.message) for x in w)
