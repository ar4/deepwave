"""Tests for the viscoacoustic (single-mechanism SLS) propagator.

Physics gates (all measured against the exact single-SLS complex modulus):
  * acoustic limit: Q -> very large reduces to ``deepwave.acoustic`` (forward
    and gradients), on both the eager and C backends;
  * amplitude decay: a monochromatic wave at omega0 decays as
    exp(-omega0 x / (2 c Q)) -- measured over propagation distance at Q=20,100;
  * dispersion: measured phase velocity at several frequencies matches the SLS
    phase-velocity relation;
  * reciprocity;
  * cross-backend agreement of the forward and of every gradient, including
    dLoss/dQ.
"""

import cmath
import dataclasses
import importlib
import math

import numpy
import pytest
import torch
from helpers import c_backend_available, output_tensors

import deepwave

# ``deepwave.viscoacoustic`` (the package attribute) is the propagator function;
# reach the module for its helpers via importlib.
_vmod = importlib.import_module("deepwave.viscoacoustic")

DEVICE = torch.device("cpu")
DTYPE = torch.float64

C_BACKEND = pytest.mark.skipif(
    not c_backend_available("viscoacoustic", device=DEVICE),
    reason="compiled viscoacoustic backend not present in libdeepwave_C",
)

BACKENDS = [
    pytest.param("eager", id="eager"),
    pytest.param("compiled", id="c", marks=C_BACKEND),
]


# --- SLS reference theory (exact single-SLS complex modulus) ---------------
def _sls_theory(v, rho, q, f0, f):
    """Exact single-SLS phase velocity, attenuation and Q(omega)."""
    omega0 = 2 * math.pi * f0
    omega = 2 * math.pi * f
    inv_q = 1.0 / q
    tausig = (math.sqrt(1 + inv_q**2) - inv_q) / omega0
    taueps = 1.0 / (omega0**2 * tausig)
    k_mod = rho * v**2
    m = k_mod * (1 + 1j * omega * taueps) / (1 + 1j * omega * tausig)
    kc = omega * cmath.sqrt(rho / m)
    if kc.real < 0:
        kc = -kc
    c_phase = omega / kc.real
    alpha = abs(kc.imag)
    q_omega = m.real / m.imag
    return c_phase, alpha, q_omega


# --------------------------------------------------------------------------
def test_q_mapping_exact_at_omega0():
    """The Q -> (taueps, tausig) mapping gives Q(omega0) == Q exactly."""
    for q_ref in [5.0, 20.0, 100.0, 1000.0]:
        _, _, q_omega = _sls_theory(1500.0, 1000.0, q_ref, 25.0, 25.0)
        assert abs(q_omega - q_ref) < 1e-9 * q_ref
    # The q_to_relaxation_times helper matches the reference.
    q = torch.tensor([20.0, 100.0], dtype=DTYPE)
    omega0 = 2 * math.pi * 25.0
    taueps, tausig = _vmod.q_to_relaxation_times(q, omega0)
    assert torch.allclose(taueps * tausig, torch.full_like(q, 1.0 / omega0**2))


@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("q_value", [1e12, float("inf")])
def test_acoustic_limit(ndim, backend, q_value):
    """Q -> very large reduces to acoustic (forward + gradients) to <1e-10 rel.

    The residual is the physical O(1/Q) attenuation, so Q=1e12 is used (giving
    ~6e-12); the reduction is a clean O(1/Q) as Q grows.
    """
    torch.manual_seed(0)
    n = 24
    shape = (n,) * ndim
    dx = 5.0
    dt = 0.0005
    nt = 40
    freq = 25.0
    v = 1500.0 + 20.0 * torch.randn(*shape, dtype=DTYPE)
    rho = 1000.0 + 30.0 * torch.randn(*shape, dtype=DTYPE)
    q_big = torch.full(shape, q_value, dtype=DTYPE)

    src = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq).to(DTYPE).reshape(1, 1, -1)
    src_loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    src_loc[0, 0] = torch.tensor([n // 2] * ndim)
    rec_loc = torch.zeros(1, 4, ndim, dtype=torch.long)
    for j in range(4):
        rec_loc[0, j] = torch.tensor([5 + j] * ndim)

    def run_visco(vv, rr, qq):
        source_index = tuple(src_loc[0, 0].tolist())
        source_k = rr[source_index] * vv[source_index] ** 2
        return deepwave.viscoacoustic(
            vv,
            rr,
            qq,
            dx,
            dt,
            source_amplitudes_pressure_rate=source_k * src,
            source_locations_pressure_rate=src_loc,
            receiver_locations_p=rec_loc,
            pml_freq=freq,
            q_freq=freq,
            accuracy=4,
            backend=backend,
        )

    def run_acoustic(vv, rr):
        return deepwave.acoustic(
            vv,
            rr,
            dx,
            dt,
            source_amplitudes_volume_rate=src,
            source_locations_volume_rate=src_loc,
            receiver_locations_p=rec_loc,
            pml_freq=freq,
            accuracy=4,
            backend=("eager" if backend == "eager" else "compiled"),
        )

    rec_v = run_visco(v, rho, q_big).receiver_amplitudes_p
    rec_a = run_acoustic(v, rho).receiver_amplitudes_p
    assert (rec_v - rec_a).norm().item() / rec_a.norm().item() < 1e-10

    # gradients w.r.t v and rho match acoustic.
    vv = v.clone().requires_grad_(True)
    rr = rho.clone().requires_grad_(True)
    run_visco(vv, rr, q_big).receiver_amplitudes_p.pow(2).sum().backward()
    va = v.clone().requires_grad_(True)
    ra = rho.clone().requires_grad_(True)
    run_acoustic(va, ra).receiver_amplitudes_p.pow(2).sum().backward()
    assert (vv.grad - va.grad).norm().item() / va.grad.norm().item() < 1e-9
    assert (rr.grad - ra.grad).norm().item() / ra.grad.norm().item() < 1e-9


def test_pressure_rate_adds_dt_times_amplitude_without_model_scaling():
    """A one-step zero-state oracle checks direct pressure-rate forcing."""
    n = 16
    v = torch.full((n,), 2.0, dtype=DTYPE, requires_grad=True)
    rho = torch.linspace(2.0, 4.0, n, dtype=DTYPE, requires_grad=True)
    q = torch.linspace(10.0, 20.0, n, dtype=DTYPE, requires_grad=True)
    amplitude = torch.tensor([[[12.5]]], dtype=DTYPE, requires_grad=True)
    location = torch.tensor([[[n // 2]]])
    dt = 0.05
    out = deepwave.viscoacoustic(
        v,
        rho,
        q,
        1.0,
        dt,
        source_amplitudes_pressure_rate=amplitude,
        source_locations_pressure_rate=location,
        q_freq=0.1,
        max_vel=3.0,
        pml_freq=0.1,
        pml_width=0,
        boundary="rigid",
        accuracy=2,
        backend="eager",
        location_interpolation="none",
    )

    pressure = out.pressure_0.squeeze()
    expected = out.info.internal_dt * amplitude.item()
    assert out.info.internal_dt == pytest.approx(dt)
    assert pressure[n // 2].item() == pytest.approx(expected)
    assert pressure.sum().item() == pytest.approx(expected)
    assert all(
        torch.count_nonzero(memory) == 0 for memory in out.auxiliary_state.values()
    )
    gradients = torch.autograd.grad(
        pressure.sum(), (amplitude, v, rho, q), allow_unused=True
    )
    assert gradients[0] is not None
    assert gradients[0].item() == pytest.approx(out.info.internal_dt)
    for gradient in gradients[1:]:
        assert gradient is None or torch.count_nonzero(gradient) == 0


def _measure_1d(
    q_val,
    f,
    f0,
    backend,
    v=1500.0,
    rho=1000.0,
    q_n_mechanisms=1,
    q_band=None,
    measure_frac=0.7,
):
    """Monochromatic 1-D steady-state amplitude & phase along a receiver line.

    ``q_n_mechanisms``/``q_band`` default to the single-mechanism SLS (bit-
    identical to the historical behaviour of this helper); passing
    ``q_n_mechanisms > 1`` with a ``q_band`` measures the multi-mechanism
    GSLS propagator instead (eager backend only, see :func:`deepwave.
    viscoacoustic`'s ``q_n_mechanisms`` docs).

    ``measure_frac`` (default 0.7, matching every existing single-mechanism
    caller's historical window exactly) is the fraction of ``nt`` after which
    the steady-state amplitude/phase measurement window starts. Multi-
    mechanism GSLS callers measuring OFF an individual relaxation
    mechanism's own peak frequency (the single-mechanism callers always
    measure exactly ``f0 == f``, i.e. AT the peak) need a later window: the
    single-frequency source's finite ramp-up bandwidth interacts with the
    medium's frequency-dependent dispersion away from a peak, and that
    transient measurably pollutes a 0.7-window amplitude fit (verified with
    the historical single-mechanism formula: off-peak measurement at
    ``pml_freq`` correctly matched to the propagating frequency still shows a
    20-70% amplitude-decay error at ``measure_frac=0.7``, dropping to <2% at
    ``measure_frac=0.85`` -- a measurement-window artifact, not a physics or
    implementation bug; see the C1 commit body for the diagnostic).
    """
    dx = 2.0
    dt = 0.0004
    nx = 1600
    nt = 6000
    vt = v * torch.ones(nx, dtype=DTYPE)
    rhot = rho * torch.ones(nx, dtype=DTYPE)
    qt = q_val * torch.ones(nx, dtype=DTYPE)
    t = torch.arange(nt, dtype=DTYPE) * dt
    ramp = torch.clamp(t / (20.0 / f), max=1.0)
    src = (ramp * torch.sin(2 * math.pi * f * t)).reshape(1, 1, -1)
    src_loc = torch.tensor([[[100]]])
    xs = torch.arange(300, 900)
    rec_loc = torch.zeros(1, xs.numel(), 1, dtype=torch.long)
    rec_loc[0, :, 0] = xs
    out = deepwave.viscoacoustic(
        vt,
        rhot,
        qt,
        dx,
        dt,
        source_amplitudes_pressure_rate=src,
        source_locations_pressure_rate=src_loc,
        receiver_locations_p=rec_loc,
        pml_freq=f0,
        q_freq=f0 if q_n_mechanisms == 1 else None,
        q_n_mechanisms=q_n_mechanisms,
        q_band=q_band,
        pml_width=40,
        accuracy=6,
        backend=backend,
    )
    rec = out.receiver_amplitudes_p[0]
    half = slice(int(nt * measure_frac), nt)
    tt = t[half]
    sig = rec[:, half]
    a = (sig * torch.cos(2 * math.pi * f * tt)).sum(1)
    b = (sig * torch.sin(2 * math.pi * f * tt)).sum(1)
    amp = torch.sqrt(a**2 + b**2)
    phase = torch.atan2(b, a)
    return xs.double() * dx, amp, phase


def _fit_line(x, y):
    mat = torch.stack([torch.ones_like(x), x], 1)
    return torch.linalg.lstsq(mat, y).solution


@pytest.mark.parametrize("q_val", [20.0, 100.0])
def test_amplitude_decay(q_val, record_property):
    """Measured amplitude decay matches exp(-omega0 x/(2 c Q)) to a few %."""
    f0 = 25.0
    x, amp, _ = _measure_1d(q_val, f0, f0, "eager")
    i0, i1 = 50, 500
    xw = x[i0:i1]
    alpha_meas = -_fit_line(xw, torch.log(amp[i0:i1]))[1].item()
    _, alpha_th, _ = _sls_theory(1500.0, 1000.0, q_val, f0, f0)
    record_property("attenuation_relative_error", abs(alpha_meas / alpha_th - 1))
    assert abs(alpha_meas / alpha_th - 1) < 0.05


@pytest.mark.parametrize("f", [15.0, 25.0, 40.0])
def test_dispersion(f, record_property):
    """Measured phase velocity matches the SLS relation to ~1%."""
    f0 = 25.0
    q_val = 100.0
    x, _, phase = _measure_1d(q_val, f, f0, "eager")
    i0, i1 = 50, 500
    xw = x[i0:i1]
    phw = torch.from_numpy(__import__("numpy").unwrap(phase[i0:i1].numpy()))
    k_meas = abs(_fit_line(xw, phw)[1].item())
    c_meas = 2 * math.pi * f / k_meas
    c_th, _, _ = _sls_theory(1500.0, 1000.0, q_val, f0, f)
    record_property("phase_speed_relative_error", abs(c_meas / c_th - 1))
    assert abs(c_meas / c_th - 1) < 0.01


# --- Multi-mechanism GSLS constant-Q fit (q_n_mechanisms > 1) --------------
# q_n_mechanisms > 1 is supported on the eager backend and the default
# compiled C/CUDA backend (the DW_GSLS kernel variant -- see
# deepwave.viscoacoustic's docstring); torch_compile raises
# NotImplementedError (tested in test_multi_mechanism_validation below), as
# do the compiled backend's unsupported combinations (M-PML, storage
# compression, per-shot batched models).

_GSLS_BAND = (10.0, 60.0)
_GSLS_L = 3


def test_multi_mechanism_validation():
    """q_n_mechanisms/q_band validation errors, and the unsupported gates."""
    n = 8
    v = torch.full((n,), 1500.0, dtype=DTYPE)
    rho = torch.full((n,), 1000.0, dtype=DTYPE)
    q = torch.full((n,), 30.0, dtype=DTYPE)

    with pytest.raises(ValueError, match="q_band must be None"):
        deepwave.viscoacoustic(
            v, rho, q, 5.0, 0.001, q_n_mechanisms=1, q_band=(1.0, 2.0), q_freq=25.0
        )
    with pytest.raises(ValueError, match=r"q_band=\(f1, f2\) is required"):
        deepwave.viscoacoustic(v, rho, q, 5.0, 0.001, q_n_mechanisms=3)
    with pytest.raises(ValueError, match="0 < f1 < f2"):
        deepwave.viscoacoustic(
            v, rho, q, 5.0, 0.001, q_n_mechanisms=3, q_band=(60.0, 10.0)
        )
    with pytest.raises(ValueError, match="q_n_mechanisms must be >= 1"):
        deepwave.viscoacoustic(v, rho, q, 5.0, 0.001, q_n_mechanisms=0)
    kw = {"q_n_mechanisms": _GSLS_L, "q_band": _GSLS_BAND}
    with pytest.raises(NotImplementedError, match='backend="torch_compile"'):
        deepwave.viscoacoustic(v, rho, q, 5.0, 0.001, backend="torch_compile", **kw)
    # Compiled-backend (DW_GSLS) unsupported combinations.
    with pytest.raises(NotImplementedError, match="storage_compression"):
        deepwave.viscoacoustic(v, rho, q, 5.0, 0.001, storage_compression=True, **kw)
    with pytest.raises(NotImplementedError, match="batched"):
        deepwave.viscoacoustic(
            v.expand(2, n).clone(),
            rho,
            q,
            5.0,
            0.001,
            source_amplitudes_pressure_rate=torch.zeros(2, 1, 4, dtype=DTYPE),
            source_locations_pressure_rate=torch.full(
                (2, 1, 1), n // 2, dtype=torch.long
            ),
            **kw,
        )


def test_gsls_fit_effective_q_within_band(record_property):
    """The GSLS fit's effective Q(f) is within 5% of the target Q (L=3).

    Checked across q_band at 3 probe frequencies spanning it. Pure
    fit-quality check (no propagation): the analytic multi-mechanism
    ``1/Q(omega) = sum_l tau_l * (omega*tausig_l)/(1+(omega*tausig_l)**2)``,
    using the SAME weights :func:`deepwave.viscoacoustic.gsls_fit_weights`
    the propagator computes, evaluated against the constant target Q. Since
    ``tau_l = w_l / q`` (see ``gsls_fit_weights``), the effective-Q error
    (a ratio) is independent of the target Q value.
    """
    tausig, weights = _vmod.gsls_fit_weights(_GSLS_BAND, _GSLS_L, torch.float64, DEVICE)
    q_target = 50.0
    tau = weights / q_target
    errs = []
    for f in [15.0, 25.0, 40.0]:
        omega = 2 * math.pi * f
        x = omega * tausig
        inv_q_eff = (tau * x / (1 + x**2)).sum().item()
        q_eff = 1.0 / inv_q_eff
        errs.append(abs(q_eff / q_target - 1))
        assert errs[-1] < 0.05, (f, q_eff, q_target)
    # Sanity: this is a real, non-vacuous fit -- a naive single-frequency
    # (L=1-style) approximation would be far worse than 5% away from flat
    # over this ~6x band (verified once, kept as an inline sensitivity check
    # rather than a second permanent test): using ONLY the band's lowest
    # frequency's tausig for all three probe points is measurably worse.
    tausig_lo = tausig[0]
    naive_errs = []
    for f in [15.0, 25.0, 40.0]:
        omega = 2 * math.pi * f
        x = omega * tausig_lo
        inv_q_eff = (weights[0] / q_target * x / (1 + x**2)).item()
        naive_errs.append(abs((1.0 / inv_q_eff) / q_target - 1))
    assert max(naive_errs) > 5 * max(errs)
    record_property("effective_q_relative_errors", errs)


@pytest.mark.parametrize("q_value", [1e10, float("inf")])
def test_acoustic_limit_multi_mechanism(q_value):
    """Q -> very large reduces to acoustic (forward + gradients), L=3 GSLS.

    Generalises test_acoustic_limit to q_n_mechanisms=3 (eager backend only).
    """
    torch.manual_seed(0)
    ndim = 2
    n = 24
    shape = (n,) * ndim
    dx = 5.0
    dt = 0.0005
    nt = 40
    freq = 25.0
    v = 1500.0 + 20.0 * torch.randn(*shape, dtype=DTYPE)
    rho = 1000.0 + 30.0 * torch.randn(*shape, dtype=DTYPE)
    q_big = torch.full(shape, q_value, dtype=DTYPE)

    src = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq).to(DTYPE).reshape(1, 1, -1)
    src_loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    src_loc[0, 0] = torch.tensor([n // 2] * ndim)
    rec_loc = torch.zeros(1, 4, ndim, dtype=torch.long)
    for j in range(4):
        rec_loc[0, j] = torch.tensor([5 + j] * ndim)

    def run_visco(vv, rr, qq):
        source_index = tuple(src_loc[0, 0].tolist())
        source_k = rr[source_index] * vv[source_index] ** 2
        return deepwave.viscoacoustic(
            vv,
            rr,
            qq,
            dx,
            dt,
            source_amplitudes_pressure_rate=source_k * src,
            source_locations_pressure_rate=src_loc,
            receiver_locations_p=rec_loc,
            pml_freq=freq,
            accuracy=4,
            q_n_mechanisms=_GSLS_L,
            q_band=_GSLS_BAND,
            backend="eager",
        ).receiver_amplitudes_p

    def run_acoustic(vv, rr):
        return deepwave.acoustic(
            vv,
            rr,
            dx,
            dt,
            source_amplitudes_volume_rate=src,
            source_locations_volume_rate=src_loc,
            receiver_locations_p=rec_loc,
            pml_freq=freq,
            accuracy=4,
            backend="eager",
        ).receiver_amplitudes_p

    rec_v = run_visco(v, rho, q_big)
    rec_a = run_acoustic(v, rho)
    assert (rec_v - rec_a).norm().item() / rec_a.norm().item() < 1e-7

    vv = v.clone().requires_grad_(True)
    rr = rho.clone().requires_grad_(True)
    run_visco(vv, rr, q_big).pow(2).sum().backward()
    va = v.clone().requires_grad_(True)
    ra = rho.clone().requires_grad_(True)
    run_acoustic(va, ra).pow(2).sum().backward()
    assert (vv.grad - va.grad).norm().item() / va.grad.norm().item() < 1e-6
    assert (rr.grad - ra.grad).norm().item() / ra.grad.norm().item() < 1e-6


@pytest.mark.parametrize("f", [15.0, 25.0, 40.0])
def test_amplitude_decay_multi_mechanism(f, record_property):
    """Multi-mechanism (L=3) measured decay matches the target Q within 5%.

    Generalises test_amplitude_decay to 3 probe frequencies spanning
    q_band, each independently measured end-to-end (real propagation, not
    just the fit): amplitude decay AND phase velocity are measured from the
    simulated wavefield and combined into an effective Q via alpha = omega /
    (2 * c_meas * Q), then compared to the target Q. Uses a later
    measurement window than the single-mechanism test (see
    ``_measure_1d``'s ``measure_frac`` docs): none of the 3 probe
    frequencies sits exactly at one of the 3 mechanisms' own relaxation
    peaks (the single-mechanism test always measures exactly at its one
    peak), so the window needs to be later for the off-peak transient to
    have genuinely settled.
    """
    q_val = 50.0
    x, amp, phase = _measure_1d(
        q_val,
        f,
        f,
        "eager",
        q_n_mechanisms=_GSLS_L,
        q_band=_GSLS_BAND,
        measure_frac=0.85,
    )
    i0, i1 = 50, 500
    xw = x[i0:i1]
    alpha_meas = -_fit_line(xw, torch.log(amp[i0:i1]))[1].item()
    phw = torch.from_numpy(__import__("numpy").unwrap(phase[i0:i1].numpy()))
    k_meas = abs(_fit_line(xw, phw)[1].item())
    c_meas = 2 * math.pi * f / k_meas
    omega = 2 * math.pi * f
    q_eff_meas = omega / (2 * c_meas * alpha_meas)
    record_property("measured_effective_q", q_eff_meas)
    record_property("effective_q_relative_error", abs(q_eff_meas / q_val - 1))
    assert abs(q_eff_meas / q_val - 1) < 0.05, (f, q_eff_meas, alpha_meas, c_meas)


def test_adjoint_dot_product_multi_mechanism():
    """``<F s, r> == <s, F* r>`` for source amplitudes, L=3 GSLS (eager only).

    Mirrors tests/test_adjoint_dot_product.py's pattern (see that file's
    docstring for the identity's derivation); the compiled backend's
    equivalent is test_multi_mechanism_c_adjoint_dot_product below.
    """
    torch.manual_seed(7)
    n = 16
    nt = 24
    dx = 5.0
    dt = 0.0006
    shape = (n, n)
    v = 1500.0 + 20.0 * torch.randn(shape, dtype=DTYPE)
    rho = 1000.0 + 30.0 * torch.randn(shape, dtype=DTYPE)
    q = torch.full(shape, 40.0, dtype=DTYPE)
    src_loc = torch.tensor([[[n // 2, n // 2]]], dtype=torch.long)
    nrec = 6
    rec_loc = torch.zeros(1, nrec, 2, dtype=torch.long)
    for j in range(nrec):
        rec_loc[0, j, 0] = 3 + j
        rec_loc[0, j, 1] = n // 2

    def forward(x):
        out = deepwave.viscoacoustic(
            v,
            rho,
            q,
            dx,
            dt,
            source_amplitudes_pressure_rate=x,
            source_locations_pressure_rate=src_loc,
            receiver_locations_p=rec_loc,
            pml_freq=25.0,
            accuracy=4,
            pml_width=4,
            q_n_mechanisms=_GSLS_L,
            q_band=_GSLS_BAND,
            backend="eager",
        )
        return out.receiver_amplitudes_p

    s = torch.randn(1, 1, nt, dtype=DTYPE)
    leaf = s.clone().requires_grad_(True)
    out = forward(leaf)
    assert out.abs().max().item() > 0
    r = torch.randn_like(out)
    loss = (out * r).sum()
    loss.backward()
    assert leaf.grad is not None
    lhs = loss.item()
    rhs = (s * leaf.grad).sum().item()
    denom = max(abs(lhs), abs(rhs), 1e-300)
    assert abs(lhs - rhs) / denom < 1e-10, (lhs, rhs)


def test_forward_callback_dict_keys_multi_mechanism():
    """forward_callback's wavefield-dict keys match r_0..r_{L-1}, L=3."""
    n = 10
    ndim = 2
    shape = (n,) * ndim
    v = torch.full(shape, 1500.0, dtype=DTYPE)
    rho = torch.full(shape, 1000.0, dtype=DTYPE)
    q = torch.full(shape, 30.0, dtype=DTYPE)
    amp = torch.zeros(1, 1, 8, dtype=DTYPE)
    amp[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded: dict[str, torch.Tensor] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded.update(state._wavefields)

    deepwave.viscoacoustic(
        v,
        rho,
        q,
        5.0,
        0.0005,
        source_amplitudes_pressure_rate=amp,
        source_locations_pressure_rate=loc,
        receiver_locations_p=loc,
        pml_freq=15.0,
        pml_width=3,
        q_n_mechanisms=_GSLS_L,
        q_band=_GSLS_BAND,
        forward_callback=recorder,
        callback_frequency=1,
        backend="eager",
    )
    assert set(recorded) == set(_vmod.viscoacoustic_wavefield_names(ndim, _GSLS_L))
    for name, tensor in recorded.items():
        assert tensor.numel() > 0, f"{name}: empty tensor"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_multi_mechanism_eager_cpu_gpu_match() -> None:
    """Eager backend: CPU and GPU forward output and dLoss/dq gradients match.

    The eager-backend analogue of test_c_forward_and_gradient_cpu_gpu_match
    -- verifies the multi-mechanism GSLS path (gsls_fit_weights, the
    per-mechanism update_pressure loop) runs correctly on a real GPU, not
    just CPU. Runs only when CUDA is available. The compiled (DW_GSLS)
    backend has its own CPU/CUDA coverage in the
    test_multi_mechanism_c_* tests below.
    """
    torch.manual_seed(3)
    n = 12
    shape = (n, n)
    v = 1500.0 + 20.0 * torch.randn(shape, dtype=DTYPE)
    rho = 1000.0 + 30.0 * torch.randn(shape, dtype=DTYPE)
    q = 25.0 + 5.0 * torch.rand(shape, dtype=DTYPE)
    src = torch.randn(1, 1, 20, dtype=DTYPE)
    src_loc = torch.zeros(1, 1, 2, dtype=torch.long)
    src_loc[0, 0] = torch.tensor([n // 2, n // 2])
    rec_loc = torch.zeros(1, 3, 2, dtype=torch.long)
    for j in range(3):
        rec_loc[0, j] = torch.tensor([3 + j, 3 + j])

    def run(device):
        vv = v.to(device).clone().requires_grad_(True)
        rr = rho.to(device).clone().requires_grad_(True)
        qq = q.to(device).clone().requires_grad_(True)
        ss = src.to(device).clone().requires_grad_(True)
        out = deepwave.viscoacoustic(
            vv,
            rr,
            qq,
            5.0,
            0.0006,
            source_amplitudes_pressure_rate=ss,
            source_locations_pressure_rate=src_loc.to(device),
            receiver_locations_p=rec_loc.to(device),
            pml_width=5,
            accuracy=4,
            pml_freq=15.0,
            q_n_mechanisms=_GSLS_L,
            q_band=_GSLS_BAND,
            backend="eager",
        )
        loss = out.receiver_amplitudes_p.pow(2).sum()
        loss.backward()
        return (
            out.receiver_amplitudes_p.detach().cpu(),
            [g.grad.detach().cpu() for g in (vv, rr, qq, ss)],
        )

    out_cpu, g_cpu = run(torch.device("cpu"))
    out_gpu, g_gpu = run(torch.device("cuda"))
    scale = max(out_cpu.abs().max().item(), 1.0)
    assert (out_cpu - out_gpu).abs().max().item() <= scale * 1e-6
    for gc, gg in zip(g_cpu, g_gpu):
        scale = max(gc.abs().max().item(), 1.0)
        assert (gc - gg).abs().max().item() <= scale * 1e-6


# --- Compiled (DW_GSLS) backend, q_n_mechanisms > 1 ------------------------
# All correctness gates compare the compiled C/CUDA backend against the eager
# reference (autograd through the machine-identical physics), which the
# eager-only tests above validated analytically. The boundary-config matrix
# reuses the proven well-conditioned recipe from tests/test_boundary.py's
# wrapping/fr-combination parity tests (nt chosen so the front crosses the
# 16-cell interior > 2x, so the periodic wrap and the reflecting faces are
# genuinely exercised -- asserted numerically), because vacuous
# never-reaches-the-boundary configs are exactly how earlier compiled
# guard-fill adjoint bugs shipped unnoticed.

_GSLS_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.gpu,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
        ],
    ),
]


def _gsls_c_run(v, rho, q, sa, sl, rl, boundary, backend, dev, n_shots=1):
    pml_width = (
        [0 if code != "pml" else 10 for code in boundary] if boundary is not None else 4
    )
    return deepwave.viscoacoustic(
        v.to(dev),
        rho.to(dev),
        q.to(dev),
        5.0,
        0.0001,
        source_amplitudes_pressure_rate=sa.to(dev),
        source_locations_pressure_rate=sl.to(dev),
        receiver_locations_p=rl.to(dev),
        pml_width=pml_width,
        boundary=boundary,
        accuracy=4,
        pml_freq=25.0,
        q_n_mechanisms=_GSLS_L,
        q_band=_GSLS_BAND,
        backend=backend,
    )


@C_BACKEND
@pytest.mark.parametrize("device", _GSLS_DEVICES)
@pytest.mark.parametrize(
    "boundary",
    [
        None,  # all-PML (the default path)
        ["rigid", "pml", "free", "pml"],
        ["free", "pml", "free", "pml"],
        ["free", "rigid", "free", "rigid"],  # historical re-fill pairing pin
    ],
    ids=lambda b: "pml" if b is None else "-".join(b),
)
def test_multi_mechanism_c_parity_boundaries(device, boundary):
    """Compiled L=3 GSLS: forward and every gradient match eager (2-D).

    Covers the all-PML default plus several free/rigid combinations
    (including the mixed ones that take the exact fold/gather adjoint path,
    and the (free-low, rigid-high) pairing that keeps the historical re-fill
    path). ``viscoacoustic`` accepts a ``"free"`` face at the beginning of a
    dimension at every accuracy, and ``"rigid"`` on either side, so every
    combination here puts its ``"free"`` faces on the low side.
    """
    torch.manual_seed(0)
    shape = (16, 16)
    nt = 1200
    # The front must cross the interior at least twice (5.0 = dx,
    # 0.0001 = dt, both fixed inside _gsls_c_run).
    assert 1500.0 * 0.0001 / 5.0 * nt > 2 * shape[0]
    sa = deepwave.wavelets.ricker(25.0, nt, 0.0001, 0.02).reshape(1, 1, nt)
    sa = sa.to(DTYPE).requires_grad_(True)
    v = (1500.0 + 30 * torch.rand(shape, dtype=DTYPE)).requires_grad_(True)
    rho = (1000.0 + 20 * torch.rand(shape, dtype=DTYPE)).requires_grad_(True)
    q = (50.0 + 5 * torch.rand(shape, dtype=DTYPE)).requires_grad_(True)
    sl = torch.tensor([[[10, 10]]])
    rl = torch.tensor([[[5, 5]]])

    args = (v, rho, q, sa)
    out_c = _gsls_c_run(*args, sl, rl, boundary, "compiled", device)
    rec_c = out_c.receiver_amplitudes_p
    g_c = torch.autograd.grad(rec_c.sum(), args)
    args2 = tuple(a.detach().requires_grad_(True) for a in args)
    out_e = _gsls_c_run(*args2, sl, rl, boundary, "eager", "cpu")
    rec_e = out_e.receiver_amplitudes_p
    g_e = torch.autograd.grad(rec_e.sum(), args2)
    assert rec_e.abs().max() > 0
    assert (rec_c.cpu() - rec_e).abs().max() <= 1e-6 * rec_e.abs().max()
    # Final wavefields (including every r_l memory variable) also match --
    # except on free-face configurations other than the historical
    # (free-low, rigid-high) pairing, where the raw returned snapshots differ between
    # the backends for q_n_mechanisms == 1 already (the eager backend
    # applies a final boundary-consistency fill to the returned state; the
    # compiled kernel applies the identical fill at the start of the next
    # step instead, so continuation physics agrees while the raw
    # surface/guard node values do not). That pre-existing single-mechanism
    # end-state representation difference is out of this comparison's scope;
    # receiver amplitudes and every gradient are compared for ALL configs
    # above/below.
    compare_wavefields = boundary is None or boundary == [
        "free",
        "rigid",
        "free",
        "rigid",
    ]
    if compare_wavefields:
        # Every state field by name, with the auxiliary state expanded into
        # the individual absorbing-boundary variables it carries.
        pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for name, vc, ve in zip(
            _vmod.viscoacoustic_output_names(2, _GSLS_L), out_c, out_e
        ):
            if name.startswith("receiver_"):
                continue
            if name == "auxiliary_state":
                pairs.extend((key, vc[key], ve[key]) for key in vc)
            else:
                pairs.append((name, vc, ve))
        for name, wc_t, we_t in pairs:
            wc = wc_t.detach().cpu()
            we = we_t.detach()
            scale = we.abs().max().clamp_min(rec_e.abs().max() * 1e-9)
            assert (wc - we).abs().max() <= 1e-6 * scale, name
    for a, b in zip(g_c, g_e):
        assert (a.cpu() - b).abs().max() <= 1e-6 * b.abs().max().clamp_min(1e-300)


@C_BACKEND
@pytest.mark.parametrize("device", _GSLS_DEVICES)
@pytest.mark.parametrize("ndim", [1, 3])
def test_multi_mechanism_c_parity_1d_3d(device, ndim):
    """Compiled L=3 GSLS: forward/gradient parity with eager in 1-D and 3-D."""
    torch.manual_seed(1)
    n = 12 if ndim == 3 else 32
    shape = (n,) * ndim
    nt = 60
    sa = torch.randn(1, 1, nt, dtype=DTYPE).requires_grad_(True)
    v = (1500.0 + 30 * torch.rand(shape, dtype=DTYPE)).requires_grad_(True)
    rho = (1000.0 + 20 * torch.rand(shape, dtype=DTYPE)).requires_grad_(True)
    q = (50.0 + 5 * torch.rand(shape, dtype=DTYPE)).requires_grad_(True)
    sl = torch.full((1, 1, ndim), n // 2, dtype=torch.long)
    rl = torch.full((1, 2, ndim), 3, dtype=torch.long)
    rl[0, 1] = n // 2 - 2

    def run(vv, rr, qq, ss, backend, dev):
        return deepwave.viscoacoustic(
            vv.to(dev),
            rr.to(dev),
            qq.to(dev),
            5.0,
            0.0005,
            source_amplitudes_pressure_rate=ss.to(dev),
            source_locations_pressure_rate=sl.to(dev),
            receiver_locations_p=rl.to(dev),
            pml_width=4,
            accuracy=4,
            pml_freq=25.0,
            q_n_mechanisms=_GSLS_L,
            q_band=_GSLS_BAND,
            backend=backend,
        ).receiver_amplitudes_p

    args = (v, rho, q, sa)
    out_c = run(*args, "compiled", device)
    g_c = torch.autograd.grad(out_c.sum(), args)
    args2 = tuple(a.detach().requires_grad_(True) for a in args)
    out_e = run(*args2, "eager", "cpu")
    g_e = torch.autograd.grad(out_e.sum(), args2)
    assert out_e.abs().max() > 0
    assert (out_c.cpu() - out_e).abs().max() <= 1e-6 * out_e.abs().max()
    for a, b in zip(g_c, g_e):
        assert (a.cpu() - b).abs().max() <= 1e-6 * b.abs().max().clamp_min(1e-300)


@C_BACKEND
@pytest.mark.parametrize("device", _GSLS_DEVICES)
def test_multi_mechanism_c_multishot_parity(device):
    """Compiled L=3 GSLS, n_shots=2: gradients match eager.

    Exercises the multi-shot gradient-accumulation path (per-shot scratch
    plus the per-mechanism shot-combine on CUDA; barrier-separated direct
    accumulation on CPU) for the L-times-bigger grad_c2/grad_ar buffers.
    """
    torch.manual_seed(2)
    n = 16
    shape = (n, n)
    nt = 60
    sa = torch.randn(2, 1, nt, dtype=DTYPE).requires_grad_(True)
    v = (1500.0 + 30 * torch.rand(shape, dtype=DTYPE)).requires_grad_(True)
    rho = (1000.0 + 20 * torch.rand(shape, dtype=DTYPE)).requires_grad_(True)
    q = (50.0 + 5 * torch.rand(shape, dtype=DTYPE)).requires_grad_(True)
    sl = torch.tensor([[[8, 8]], [[9, 6]]])
    rl = torch.tensor([[[5, 5]], [[4, 10]]])

    def run(vv, rr, qq, ss, backend, dev):
        return deepwave.viscoacoustic(
            vv.to(dev),
            rr.to(dev),
            qq.to(dev),
            5.0,
            0.0005,
            source_amplitudes_pressure_rate=ss.to(dev),
            source_locations_pressure_rate=sl.to(dev),
            receiver_locations_p=rl.to(dev),
            pml_width=4,
            accuracy=4,
            pml_freq=25.0,
            q_n_mechanisms=_GSLS_L,
            q_band=_GSLS_BAND,
            backend=backend,
        ).receiver_amplitudes_p

    args = (v, rho, q, sa)
    out_c = run(*args, "compiled", device)
    g_c = torch.autograd.grad(out_c.sum(), args)
    args2 = tuple(a.detach().requires_grad_(True) for a in args)
    out_e = run(*args2, "eager", "cpu")
    g_e = torch.autograd.grad(out_e.sum(), args2)
    assert out_e.abs().max() > 0
    assert (out_c.cpu() - out_e).abs().max() <= 1e-6 * out_e.abs().max()
    for a, b in zip(g_c, g_e):
        assert (a.cpu() - b).abs().max() <= 1e-6 * b.abs().max().clamp_min(1e-300)


@C_BACKEND
@pytest.mark.parametrize("device", _GSLS_DEVICES)
def test_multi_mechanism_c_storage_modes(device):
    """Compiled L=3 GSLS: disk (and CUDA cpu) storage match device storage.

    The L-times-bigger r_old checkpoint blobs use their own byte-scaled
    save/load path (see viscoacoustic.c/.cu); this verifies the disk
    seek/append arithmetic (and the CUDA pinned-host staging) against the
    in-memory device-storage gradients.
    """
    torch.manual_seed(3)
    n = 12
    shape = (n, n)
    nt = 30
    sa = torch.randn(1, 1, nt, dtype=DTYPE)
    v0 = 1500.0 + 30 * torch.rand(shape, dtype=DTYPE)
    rho0 = 1000.0 + 20 * torch.rand(shape, dtype=DTYPE)
    q0 = 50.0 + 5 * torch.rand(shape, dtype=DTYPE)
    sl = torch.tensor([[[6, 6]]])
    rl = torch.tensor([[[3, 3]]])

    def grads(storage_mode, tmp_path="."):
        vv = v0.clone().requires_grad_(True)
        rr = rho0.clone().requires_grad_(True)
        qq = q0.clone().requires_grad_(True)
        out = deepwave.viscoacoustic(
            vv.to(device),
            rr.to(device),
            qq.to(device),
            5.0,
            0.0005,
            source_amplitudes_pressure_rate=sa.to(device),
            source_locations_pressure_rate=sl.to(device),
            receiver_locations_p=rl.to(device),
            pml_width=4,
            accuracy=4,
            pml_freq=25.0,
            q_n_mechanisms=_GSLS_L,
            q_band=_GSLS_BAND,
            storage_mode=storage_mode,
            storage_path=tmp_path,
        ).receiver_amplitudes_p
        out.pow(2).sum().backward()
        return [g.grad.detach().cpu() for g in (vv, rr, qq)]

    ref = grads("device")
    modes = ["disk"] if device == "cpu" else ["cpu", "disk"]
    for mode in modes:
        got = grads(mode)
        for a, b in zip(got, ref):
            close = (a - b).abs().max() <= 1e-12 * b.abs().max().clamp_min(1e-300)
            assert torch.equal(a, b) or close, mode


@C_BACKEND
def test_multi_mechanism_c_adjoint_dot_product():
    """``<F s, r> == <s, F* r>`` on the compiled backend, L=3 GSLS.

    The compiled-backend counterpart of
    test_adjoint_dot_product_multi_mechanism above (same construction,
    backend="compiled").
    """
    torch.manual_seed(7)
    n = 16
    nt = 24
    shape = (n, n)
    v = 1500.0 + 20.0 * torch.randn(shape, dtype=DTYPE)
    rho = 1000.0 + 30.0 * torch.randn(shape, dtype=DTYPE)
    q = torch.full(shape, 40.0, dtype=DTYPE)
    src_loc = torch.tensor([[[n // 2, n // 2]]], dtype=torch.long)
    nrec = 6
    rec_loc = torch.zeros(1, nrec, 2, dtype=torch.long)
    for j in range(nrec):
        rec_loc[0, j, 0] = 3 + j
        rec_loc[0, j, 1] = n // 2

    def forward(x):
        out = deepwave.viscoacoustic(
            v,
            rho,
            q,
            5.0,
            0.0006,
            source_amplitudes_pressure_rate=x,
            source_locations_pressure_rate=src_loc,
            receiver_locations_p=rec_loc,
            pml_freq=25.0,
            accuracy=4,
            pml_width=4,
            q_n_mechanisms=_GSLS_L,
            q_band=_GSLS_BAND,
        )
        return out.receiver_amplitudes_p

    s = torch.randn(1, 1, nt, dtype=DTYPE)
    leaf = s.clone().requires_grad_(True)
    out = forward(leaf)
    assert out.abs().max().item() > 0
    r = torch.randn_like(out)
    loss = (out * r).sum()
    loss.backward()
    assert leaf.grad is not None
    lhs = loss.item()
    rhs = (s * leaf.grad).sum().item()
    denom = max(abs(lhs), abs(rhs), 1e-300)
    assert abs(lhs - rhs) / denom < 1e-10, (lhs, rhs)


@C_BACKEND
@pytest.mark.parametrize("device", _GSLS_DEVICES)
def test_multi_mechanism_c_grad_ar_parity(device, monkeypatch):
    """Compiled L=3 GSLS: dLoss/dar (per-mechanism) matches eager.

    The public GSLS parameterisation's per-mechanism decay factors ``ar_l``
    depend only on the fixed relaxation times (not on ``q`` -- see
    ``prepare_models``), so ``ar`` never requires grad through the public
    API and the kernel's grad_ar path (including its n_mechanisms-times
    -bigger per-time-step r_old checkpoint storage, the one consumer of the
    second StorageManager) would otherwise go completely unexercised. This
    re-attaches ``ar`` as a differentiable leaf at the ``prepare_models``
    seam, so autograd routes a real cotangent through the compiled
    grad_ar/r_old machinery (and, on the eager side, through autograd's own
    ar edge), and compares them. Two shots, so the CUDA per-shot
    scratch/per-mechanism combine path is exercised too.
    """
    leaves = []
    orig_prepare = _vmod.prepare_models

    def patched(*a, **k):
        models = orig_prepare(*a, **k)
        leaf = models[2].detach().clone().requires_grad_(True)
        leaves.append(leaf)
        models[2] = leaf
        return models

    monkeypatch.setattr(_vmod, "prepare_models", patched)

    torch.manual_seed(6)
    n = 14
    shape = (n, n)
    nt = 40
    sa = torch.randn(2, 1, nt, dtype=DTYPE)
    v = 1500.0 + 30 * torch.rand(shape, dtype=DTYPE)
    rho = 1000.0 + 20 * torch.rand(shape, dtype=DTYPE)
    q = 50.0 + 5 * torch.rand(shape, dtype=DTYPE)
    sl = torch.tensor([[[7, 7]], [[8, 5]]])
    rl = torch.tensor([[[4, 4]], [[3, 9]]])

    def grad_ar(backend):
        leaves.clear()
        out = deepwave.viscoacoustic(
            v.to(device if backend == "compiled" else "cpu"),
            rho.to(device if backend == "compiled" else "cpu"),
            q.to(device if backend == "compiled" else "cpu"),
            5.0,
            0.0005,
            source_amplitudes_pressure_rate=sa.to(
                device if backend == "compiled" else "cpu"
            ),
            source_locations_pressure_rate=sl,
            receiver_locations_p=rl,
            pml_width=4,
            accuracy=4,
            pml_freq=25.0,
            q_n_mechanisms=_GSLS_L,
            q_band=_GSLS_BAND,
            backend=backend,
        ).receiver_amplitudes_p
        out.pow(2).sum().backward()
        (leaf,) = leaves
        assert leaf.grad is not None
        return leaf.grad.detach().cpu()

    g_c = grad_ar("compiled")
    g_e = grad_ar("eager")
    assert g_e.abs().max() > 0
    # The per-mechanism gradients must genuinely differ (a shared value
    # across mechanisms would mean the mechanism axis is not really being
    # resolved).
    assert (g_e[0] - g_e[1]).abs().max() > 1e-6 * g_e.abs().max()
    assert (g_c - g_e).abs().max() <= 1e-6 * g_e.abs().max()


@C_BACKEND
def test_multi_mechanism_c_l1_bit_identical():
    """q_n_mechanisms=1 on the compiled backend is unaffected by the port.

    The GSLS compiled path dispatches to separate _gsls symbols; the default
    single-mechanism call must remain byte-identical, so an L=1 run must be
    bitwise-equal to the same call made through the historical
    single-mechanism code path (they are the same call -- this pins that
    q_n_mechanisms=1 never routes to the _gsls variant).
    """
    torch.manual_seed(4)
    n = 12
    shape = (n, n)
    nt = 20
    sa = torch.randn(1, 1, nt, dtype=DTYPE)
    v = 1500.0 + 30 * torch.rand(shape, dtype=DTYPE)
    rho = 1000.0 + 20 * torch.rand(shape, dtype=DTYPE)
    q = 50.0 + 5 * torch.rand(shape, dtype=DTYPE)
    sl = torch.tensor([[[6, 6]]])
    rl = torch.tensor([[[3, 3]]])

    def run(**extra):
        return deepwave.viscoacoustic(
            v,
            rho,
            q,
            5.0,
            0.0005,
            source_amplitudes_pressure_rate=sa,
            source_locations_pressure_rate=sl,
            receiver_locations_p=rl,
            pml_width=4,
            accuracy=4,
            pml_freq=25.0,
            q_freq=25.0,
            **extra,
        ).receiver_amplitudes_p

    assert torch.equal(run(), run(q_n_mechanisms=1))


@pytest.mark.parametrize("backend", BACKENDS)
def test_reciprocity(backend):
    """Pressure-source/pressure-receiver reciprocity."""
    torch.manual_seed(1)
    ny, nx = 60, 70
    dx = 5.0
    dt = 0.001
    nt = 200
    freq = 15.0
    v = 1500.0 * torch.ones(ny, nx, dtype=DTYPE)
    rho = 1000.0 * torch.ones(ny, nx, dtype=DTYPE)
    q = 30.0 * torch.ones(ny, nx, dtype=DTYPE)
    wav = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq).to(DTYPE).reshape(1, 1, -1)
    loc_a = torch.tensor([[[25, 20]]])
    loc_b = torch.tensor([[[35, 50]]])

    def run(sl, rl):
        return deepwave.viscoacoustic(
            v,
            rho,
            q,
            dx,
            dt,
            source_amplitudes_pressure_rate=wav,
            source_locations_pressure_rate=sl,
            receiver_locations_p=rl,
            pml_freq=freq,
            q_freq=freq,
            accuracy=4,
            backend=backend,
        ).receiver_amplitudes_p

    d_ab = run(loc_a, loc_b)
    d_ba = run(loc_b, loc_a)
    scale = d_ab.abs().max().item()
    assert (d_ab - d_ba).abs().max().item() < 5e-3 * scale


@C_BACKEND
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_c_matches_eager(ndim):
    """The C backend forward and gradients match the eager reference tightly."""
    torch.manual_seed(3)
    n = 12
    shape = (n,) * ndim
    dx = 5.0
    dt = 0.0006
    nt = 20
    v = 1500.0 + 20.0 * torch.randn(*shape, dtype=DTYPE)
    rho = 1000.0 + 30.0 * torch.randn(*shape, dtype=DTYPE)
    q = 25.0 + 5.0 * torch.rand(*shape, dtype=DTYPE)
    src = torch.randn(1, 1, nt, dtype=DTYPE)
    src_loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    src_loc[0, 0] = torch.tensor([n // 2] * ndim)
    rec_loc = torch.zeros(1, 3, ndim, dtype=torch.long)
    for j in range(3):
        rec_loc[0, j] = torch.tensor([3 + j] * ndim)

    def run(backend):
        vv = v.clone().requires_grad_(True)
        rr = rho.clone().requires_grad_(True)
        qq = q.clone().requires_grad_(True)
        ss = src.clone().requires_grad_(True)
        out = deepwave.viscoacoustic(
            vv,
            rr,
            qq,
            dx,
            dt,
            source_amplitudes_pressure_rate=ss,
            source_locations_pressure_rate=src_loc,
            receiver_locations_p=rec_loc,
            pml_width=5,
            accuracy=4,
            pml_freq=15.0,
            q_freq=15.0,
            backend=backend,
        )
        loss = sum(o.pow(2).sum() for o in output_tensors(out) if o.numel() > 0)
        loss.backward()
        return out, (vv.grad, rr.grad, qq.grad, ss.grad)

    out_e, g_e = run("eager")
    out_c, g_c = run("compiled")
    for a, b in zip(output_tensors(out_e), output_tensors(out_c)):
        if a.numel() == 0:
            continue
        scale = max(a.abs().max().item(), 1.0)
        assert (a - b).abs().max().item() <= scale * 1e-6
    for ge, gc in zip(g_e, g_c):
        # The model gradients have different units and can be much smaller
        # than one. A unit floor allowed an entirely zero v/rho gradient to
        # pass; measure each error against that gradient's actual magnitude.
        scale = ge.abs().max().item()
        assert scale > 0, "the reference must exercise every requested gradient"
        torch.testing.assert_close(gc, ge, rtol=0, atol=scale * 1e-6)


@C_BACKEND
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_c_threading_matches_single_thread(ndim):
    """Intra-shot OpenMP threading (B2) does not change the compiled result.

    ``viscoacoustic.c`` was ported from one-shot-per-thread parallelism to
    intra-shot spatial parallelism (see ``viscoacoustic.c``'s ``FUNC(forward)``/
    ``FUNC(backward)``): each time step's velocity/pressure stencil loops are
    now split across threads via ``#pragma omp for simd ... nowait``, with
    shots processed serially, mirroring ``elastic.c``. Every grid cell is
    written by exactly one thread per parallel region (the PML-region tiling
    partitions the grid disjointly) and time steps/shots are barrier-separated,
    so the arithmetic performed for any given cell does not depend on how many
    threads are used -- forward and gradients should be identical (to
    measurement precision) whether the compiled backend runs with 1 or 4
    threads. This is a single-shot (``n_shots=1``) problem, so under the old
    one-shot-per-thread scheme extra threads would have sat idle; here they
    should be put to work within the single shot instead (this is what
    ``spatial_parallel=True`` in ``viscoacoustic.py``'s ``SPEC`` enables).

    Uses ``torch.set_num_threads`` (read by the backend via
    ``torch.get_num_threads()`` at call time, exactly as ``OMP_NUM_THREADS``
    would set the default) rather than an env var, so the comparison runs
    in-process; the tolerance mirrors ``test_c_matches_eager`` above (the only
    existing cross-backend-parity pattern in this file), not a new one.
    """
    saved_threads = torch.get_num_threads()
    try:
        torch.manual_seed(3)
        n = 12
        shape = (n,) * ndim
        dx = 5.0
        dt = 0.0006
        nt = 20
        v = 1500.0 + 20.0 * torch.randn(*shape, dtype=DTYPE)
        rho = 1000.0 + 30.0 * torch.randn(*shape, dtype=DTYPE)
        q = 25.0 + 5.0 * torch.rand(*shape, dtype=DTYPE)
        src = torch.randn(1, 1, nt, dtype=DTYPE)
        src_loc = torch.zeros(1, 1, ndim, dtype=torch.long)
        src_loc[0, 0] = torch.tensor([n // 2] * ndim)
        rec_loc = torch.zeros(1, 3, ndim, dtype=torch.long)
        for j in range(3):
            rec_loc[0, j] = torch.tensor([3 + j] * ndim)

        def run(n_threads):
            torch.set_num_threads(n_threads)
            vv = v.clone().requires_grad_(True)
            rr = rho.clone().requires_grad_(True)
            qq = q.clone().requires_grad_(True)
            ss = src.clone().requires_grad_(True)
            out = deepwave.viscoacoustic(
                vv,
                rr,
                qq,
                dx,
                dt,
                source_amplitudes_pressure_rate=ss,
                source_locations_pressure_rate=src_loc,
                receiver_locations_p=rec_loc,
                pml_width=5,
                accuracy=4,
                pml_freq=15.0,
                q_freq=15.0,
                backend="compiled",
            )
            loss = sum(o.pow(2).sum() for o in output_tensors(out) if o.numel() > 0)
            loss.backward()
            return out, (vv.grad, rr.grad, qq.grad, ss.grad)

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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_c_forward_and_gradient_cpu_gpu_match(ndim):
    """Compiled backend: CPU and GPU forward output and gradients match.

    Mirrors ``test_c_matches_eager`` above (and ``test_elastic.py``'s
    ``test_forward_cpu_gpu_match`` pattern), comparing CPU against GPU rather
    than eager against compiled. viscoacoustic has a full compiled CUDA
    adjoint, so gradients -- not just the forward output -- should agree.
    Runs only on the maintainer's GPU (skipped here, no CUDA on this box).
    """
    torch.manual_seed(3)
    n = 12
    shape = (n,) * ndim
    dx = 5.0
    dt = 0.0006
    nt = 20
    v = 1500.0 + 20.0 * torch.randn(*shape, dtype=DTYPE)
    rho = 1000.0 + 30.0 * torch.randn(*shape, dtype=DTYPE)
    q = 25.0 + 5.0 * torch.rand(*shape, dtype=DTYPE)
    src = torch.randn(1, 1, nt, dtype=DTYPE)
    src_loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    src_loc[0, 0] = torch.tensor([n // 2] * ndim)
    rec_loc = torch.zeros(1, 3, ndim, dtype=torch.long)
    for j in range(3):
        rec_loc[0, j] = torch.tensor([3 + j] * ndim)

    def run(device):
        vv = v.to(device).clone().requires_grad_(True)
        rr = rho.to(device).clone().requires_grad_(True)
        qq = q.to(device).clone().requires_grad_(True)
        ss = src.to(device).clone().requires_grad_(True)
        out = deepwave.viscoacoustic(
            vv,
            rr,
            qq,
            dx,
            dt,
            source_amplitudes_pressure_rate=ss,
            source_locations_pressure_rate=src_loc.to(device),
            receiver_locations_p=rec_loc.to(device),
            pml_width=5,
            accuracy=4,
            pml_freq=15.0,
            q_freq=15.0,
            backend="compiled",
        )
        loss = sum(o.pow(2).sum() for o in output_tensors(out) if o.numel() > 0)
        loss.backward()
        outputs = [o.detach().cpu() for o in output_tensors(out)]
        grads = [g.grad.detach().cpu() for g in (vv, rr, qq, ss)]
        return outputs, grads

    out_cpu, g_cpu = run(torch.device("cpu"))
    out_gpu, g_gpu = run(torch.device("cuda"))
    for a, b in zip(out_cpu, out_gpu):
        if a.numel() == 0:
            continue
        scale = max(a.abs().max().item(), 1.0)
        assert (a - b).abs().max().item() <= scale * 1e-6
    for ge, gc in zip(g_cpu, g_gpu):
        scale = max(ge.abs().max().item(), 1.0)
        assert (ge - gc).abs().max().item() <= scale * 1e-6


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_viscoacoustic_backward_callback_dict_keys(ndim) -> None:
    """The backward-callback dict's key set matches viscoacoustic_wavefield_names.

    ``backward_callback`` is supported only on the default compiled backend
    (``backend="compiled"``); the eager Python backend rejects it outright
    (``viscoacoustic_python`` raises ``RuntimeError`` on a non-``None``
    ``backward_callback``), so this checks the compiled backend only.
    """
    n = 10
    shape = (n,) * ndim
    v = torch.full(shape, 1500.0, dtype=DTYPE).requires_grad_()
    rho = torch.full(shape, 1000.0, dtype=DTYPE).requires_grad_()
    q = torch.full(shape, 20.0, dtype=DTYPE).requires_grad_()
    amp = torch.zeros(1, 1, 8, dtype=DTYPE)
    amp[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded: dict[str, torch.Tensor] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded.update(state._wavefields)

    out = deepwave.viscoacoustic(
        v,
        rho,
        q,
        5.0,
        0.0005,
        source_amplitudes_pressure_rate=amp,
        source_locations_pressure_rate=loc,
        receiver_locations_p=loc,
        pml_freq=15.0,
        q_freq=15.0,
        pml_width=3,
        backward_callback=recorder,
        callback_frequency=1,
    )
    out.receiver_amplitudes_p.sum().backward()
    assert set(recorded) == set(_vmod.viscoacoustic_wavefield_names(ndim))
    for name, tensor in recorded.items():
        assert tensor.numel() > 0, f"{name}: empty tensor"


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_viscoacoustic_forward_callback_dict_keys_and_backend_agreement(ndim) -> None:
    """forward_callback keys match on both backends and their values agree.

    Asserts the key set equals ``viscoacoustic_wavefield_names`` on both the
    eager and compiled backends, and that the eager and compiled per-name
    tensors agree numerically. Unlike ``backward_callback``, ``forward_callback``
    IS supported on the eager Python backend (``viscoacoustic_python`` builds
    ``dict(zip(wf_names, current_wavefields))`` every ``callback_frequency``
    steps), so both backends are compared here. Tolerance mirrors
    ``test_c_matches_eager``'s ``scale * 1e-6`` convention (the only existing
    cross-backend numeric comparison in this file).
    """
    n = 10
    shape = (n,) * ndim
    v = torch.full(shape, 1500.0, dtype=DTYPE)
    rho = torch.full(shape, 1000.0, dtype=DTYPE)
    q = torch.full(shape, 20.0, dtype=DTYPE)
    amp = torch.zeros(1, 1, 8, dtype=DTYPE)
    amp[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded: dict[str, dict[str, torch.Tensor]] = {}

    def make_recorder(key: str):
        def recorder(state: deepwave.CallbackState) -> None:
            if key not in recorded:
                recorded[key] = {
                    name: t.clone() for name, t in state._wavefields.items()
                }

        return recorder

    common_kwargs: dict[str, object] = {
        "source_amplitudes_pressure_rate": amp,
        "source_locations_pressure_rate": loc,
        "receiver_locations_p": loc,
        "pml_freq": 15.0,
        "q_freq": 15.0,
        "pml_width": 3,
        "callback_frequency": 1,
    }
    deepwave.viscoacoustic(
        v,
        rho,
        q,
        5.0,
        0.0005,
        backend="eager",
        forward_callback=make_recorder("eager"),
        **common_kwargs,
    )
    deepwave.viscoacoustic(
        v,
        rho,
        q,
        5.0,
        0.0005,
        backend="compiled",
        forward_callback=make_recorder("compiled"),
        **common_kwargs,
    )
    expected = set(_vmod.viscoacoustic_wavefield_names(ndim))
    assert set(recorded["eager"]) == expected
    assert set(recorded["compiled"]) == expected
    for name, tensor in recorded["eager"].items():
        assert tensor.numel() > 0, f"{name}: eager tensor is empty"
    for name in recorded["eager"]:
        e = recorded["eager"][name]
        c = recorded["compiled"][name]
        assert e.shape == c.shape, f"{name}: shape mismatch {e.shape} vs {c.shape}"
        scale = max(c.abs().max().item(), 1.0)
        assert (e - c).abs().max().item() <= scale * 1e-6, f"{name}: mismatch"


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_viscoacoustic_model_dict_keys(ndim) -> None:
    """backward_callback's model/gradient dicts are keyed by model names.

    Compared against ``viscoacoustic_model_names``.
    """
    n = 10
    shape = (n,) * ndim
    v = torch.full(shape, 1500.0, dtype=DTYPE).requires_grad_()
    rho = torch.full(shape, 1000.0, dtype=DTYPE).requires_grad_()
    q = torch.full(shape, 20.0, dtype=DTYPE).requires_grad_()
    amp = torch.zeros(1, 1, 8, dtype=DTYPE)
    amp[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded_models: dict[str, torch.Tensor] = {}
    recorded_grads: dict[str, torch.Tensor] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded_models.update(state._models)
        recorded_grads.update(state._gradients)

    out = deepwave.viscoacoustic(
        v,
        rho,
        q,
        5.0,
        0.0005,
        source_amplitudes_pressure_rate=amp,
        source_locations_pressure_rate=loc,
        receiver_locations_p=loc,
        pml_freq=15.0,
        q_freq=15.0,
        pml_width=3,
        backward_callback=recorder,
        callback_frequency=1,
    )
    out.receiver_amplitudes_p.sum().backward()
    expected = set(_vmod.viscoacoustic_model_names(ndim))
    assert set(recorded_models) == expected
    assert set(recorded_grads) == expected


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_viscoacoustic_output_names(ndim) -> None:
    """PropagatorOutput field names == viscoacoustic_output_names, in order."""
    n = 10
    shape = (n,) * ndim
    v = torch.full(shape, 1500.0, dtype=DTYPE)
    rho = torch.full(shape, 1000.0, dtype=DTYPE)
    q = torch.full(shape, 20.0, dtype=DTYPE)
    amp = torch.zeros(1, 1, 8, dtype=DTYPE)
    amp[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    out = deepwave.viscoacoustic(
        v,
        rho,
        q,
        5.0,
        0.0005,
        source_amplitudes_pressure_rate=amp,
        source_locations_pressure_rate=loc,
        receiver_locations_p=loc,
        pml_freq=15.0,
        q_freq=15.0,
        pml_width=3,
    )
    assert out._names == tuple(_vmod.viscoacoustic_output_names(ndim))


def test_eager_rejects_storage_compression() -> None:
    """backend="eager" rejects storage_compression=True (see PR #121)."""
    nx = 16
    v = torch.full((nx,), 1500.0, dtype=DTYPE)
    rho = torch.full((nx,), 1000.0, dtype=DTYPE)
    q = torch.full((nx,), 20.0, dtype=DTYPE)
    source_amplitudes = torch.zeros(1, 1, 8, dtype=DTYPE)
    source_locations = torch.zeros(1, 1, 1, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 1, dtype=torch.long)
    with pytest.raises(RuntimeError, match="Storage compression"):
        deepwave.viscoacoustic(
            v,
            rho,
            q,
            5.0,
            0.001,
            source_amplitudes_pressure_rate=source_amplitudes,
            source_locations_pressure_rate=source_locations,
            receiver_locations_p=receiver_locations,
            pml_freq=15.0,
            q_freq=15.0,
            backend="eager",
            storage_compression=True,
        )


def test_q_freq_is_required_and_never_taken_from_pml_freq() -> None:
    """An unset ``q_freq`` raises; it is not inherited from ``pml_freq``.

    ``Q`` without the frequency it was specified at is not an attenuation
    model, and ``pml_freq`` -- itself inferred from the source amplitudes --
    would have made the attenuation model a function of the wavelet. Both
    spellings of the old default are checked: the resolved ``pml_freq``
    (the wavelet's own frequency here) and the fixed 25.0 of Deepwave
    0.0.27, either of which a silent default could have produced.
    """
    freq = 60.0
    ny, nx, nt, dt, dx = 40, 40, 200, 0.001, 5.0
    v = torch.full((ny, nx), 1500.0, dtype=torch.float64)
    rho = torch.full((ny, nx), 2200.0, dtype=torch.float64)
    q = torch.full((ny, nx), 20.0, dtype=torch.float64)
    src = (
        deepwave.wavelets.ricker(freq, nt, dt, 1.0 / freq)
        .to(torch.float64)
        .reshape(1, 1, -1)
    )

    def run(**kwargs):
        return deepwave.viscoacoustic(
            v,
            rho,
            q,
            dx,
            dt,
            source_amplitudes_pressure_rate=src,
            source_locations_pressure_rate=torch.full((1, 1, 2), 20, dtype=torch.long),
            receiver_locations_p=torch.full((1, 1, 2), 10, dtype=torch.long),
            pml_width=10,
            **kwargs,
        ).receiver_amplitudes_p

    with pytest.raises(deepwave.DeepwaveValueError, match="q_freq is required"):
        run()
    given = run(q_freq=freq)
    # Non-vacuity: the reference frequency really does change the numbers, so
    # a default (of either the inferred pml_freq or the old fixed 25.0) would
    # have silently chosen a different attenuation model.
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
    v = torch.full((n, n), 1500.0, dtype=DTYPE)
    rho = torch.full((n, n), 2200.0, dtype=DTYPE)
    q = torch.full((n, n), 20.0, dtype=DTYPE)
    src = torch.zeros(1, 1, 10, dtype=DTYPE)
    src[0, 0, 3] = 1.0
    kwargs = {
        "source_amplitudes_pressure_rate": src,
        "source_locations_pressure_rate": torch.full((1, 1, 2), 10, dtype=torch.long),
        "receiver_locations_p": torch.full((1, 1, 2), 6, dtype=torch.long),
        "pml_width": 3,
        "pml_freq": 25.0,
    }

    def call(**extra: object) -> object:
        if form == "functional":
            return deepwave.viscoacoustic(v, rho, q, 5.0, 0.001, **kwargs, **extra)
        return deepwave.Viscoacoustic(v, rho, q, 5.0)(0.001, **kwargs, **extra)

    with pytest.raises(deepwave.DeepwaveValueError, match="q_freq is required"):
        call()
    # The same call with q_freq given runs, so the raise is about q_freq only.
    call(q_freq=25.0)


def test_q_freq_is_refused_with_several_mechanisms() -> None:
    """Above one mechanism ``q_freq`` is rejected, mirroring ``q_band``.

    A multi-mechanism run fits a constant ``Q`` across ``q_band``; there is no
    single reference frequency, so a ``q_freq`` passed there would have been
    silently ignored.
    """
    n = 20
    v = torch.full((n, n), 1500.0, dtype=DTYPE)
    rho = torch.full((n, n), 2200.0, dtype=DTYPE)
    q = torch.full((n, n), 20.0, dtype=DTYPE)
    src = torch.zeros(1, 1, 10, dtype=DTYPE)
    src[0, 0, 3] = 1.0
    kwargs = {
        "source_amplitudes_pressure_rate": src,
        "source_locations_pressure_rate": torch.full((1, 1, 2), 10, dtype=torch.long),
        "receiver_locations_p": torch.full((1, 1, 2), 6, dtype=torch.long),
        "pml_width": 3,
        "pml_freq": 25.0,
        "q_n_mechanisms": 2,
        "q_band": (10.0, 60.0),
    }
    with pytest.raises(deepwave.DeepwaveValueError, match="q_freq must be None"):
        deepwave.viscoacoustic(v, rho, q, 5.0, 0.001, q_freq=25.0, **kwargs)
    # Without it the same call runs: q_band, not q_freq, carries the frequency
    # information there.
    deepwave.viscoacoustic(v, rho, q, 5.0, 0.001, **kwargs)


def test_q_freq_must_be_positive() -> None:
    """Zero or negative reference frequencies are rejected, not clamped."""
    n = 20
    v = torch.full((n, n), 1500.0, dtype=DTYPE)
    rho = torch.full((n, n), 2200.0, dtype=DTYPE)
    q = torch.full((n, n), 20.0, dtype=DTYPE)
    src = torch.zeros(1, 1, 10, dtype=DTYPE)
    src[0, 0, 3] = 1.0
    for bad in (0.0, -5.0):
        with pytest.raises(
            deepwave.DeepwaveValueError, match="q_freq must be positive"
        ):
            deepwave.viscoacoustic(
                v,
                rho,
                q,
                5.0,
                0.001,
                source_amplitudes_pressure_rate=src,
                source_locations_pressure_rate=torch.full(
                    (1, 1, 2), 10, dtype=torch.long
                ),
                receiver_locations_p=torch.full((1, 1, 2), 6, dtype=torch.long),
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

    The value defines the medium's attenuation through
    ``omega0 = 2*pi*q_freq``, so an unusable one has to be caught at the call
    rather than left to produce a ``NaN`` wavefield (non-finite) or whatever
    error the first arithmetic on it happens to raise (wrong type). A
    one-element tensor is still accepted: the check is on what the value is,
    not on which library made it.
    """
    n = 20
    v = torch.full((n, n), 1500.0, dtype=DTYPE)
    rho = torch.full((n, n), 2200.0, dtype=DTYPE)
    q = torch.full((n, n), 20.0, dtype=DTYPE)
    src = torch.zeros(1, 1, 10, dtype=DTYPE)
    src[0, 0, 3] = 1.0
    kwargs = {
        "source_amplitudes_pressure_rate": src,
        "source_locations_pressure_rate": torch.full((1, 1, 2), 10, dtype=torch.long),
        "receiver_locations_p": torch.full((1, 1, 2), 6, dtype=torch.long),
        "pml_width": 3,
        "pml_freq": 25.0,
    }

    def call(q_freq: object) -> object:
        if form == "functional":
            return deepwave.viscoacoustic(
                v, rho, q, 5.0, 0.001, q_freq=q_freq, **kwargs
            )
        return deepwave.Viscoacoustic(v, rho, q, 5.0)(0.001, q_freq=q_freq, **kwargs)

    for bad, word in _BAD_Q_FREQ_VALUES:
        with pytest.raises(deepwave.DeepwaveValueError, match=f"q_freq must be {word}"):
            call(bad)
    for wrong_type in [*_NON_NUMBER_Q_FREQ_VALUES, torch.tensor([25.0, 30.0])]:
        with pytest.raises(
            deepwave.DeepwaveTypeError, match="q_freq must be a positive number"
        ):
            call(wrong_type)
    # A one-element tensor or a NumPy scalar is a number, and still runs.
    for good in (25.0, 25, torch.tensor(25.0), numpy.float32(25.0)):
        assert torch.isfinite(call(good).receiver_amplitudes_p).all()


def test_multi_mechanism_public_field_order_and_mechanism_identity() -> None:
    """At two mechanisms the public fields are named and ordered per mechanism.

    The mechanism index moves from a bare suffix (which reads as the time
    index every other name carries) into an explicit ``_mech`` marker, and
    the count of those fields follows ``q_n_mechanisms``. The identity of
    each one is checked physically rather than against the name list this
    test also asserts: with every field but the memory variables started at
    zero and no source, the GSLS update reduces to ``r <- ar * r``, and each
    mechanism's own decay factor ``ar`` differs, so the ratio a field decays
    by says which mechanism it belongs to.
    """
    ndim, n_mech = 2, 2
    ny, nx, pml = 20, 21, 5
    dx, dt, band = 4.0, 0.001, (5.0, 50.0)
    shape = (ny, nx)
    v = torch.full(shape, 1500.0, dtype=DTYPE)
    rho = torch.full(shape, 1000.0, dtype=DTYPE)
    q = torch.full(shape, 40.0, dtype=DTYPE)

    expected = [
        "pressure_0",
        "vy_0",
        "vx_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_y",
        "receiver_amplitudes_x",
    ]
    assert _vmod.viscoacoustic_output_names(ndim, n_mech) == expected

    def run(initial_state=None):
        return deepwave.viscoacoustic(
            v,
            rho,
            q,
            dx,
            dt,
            source_amplitudes_pressure_rate=torch.zeros(1, 1, 1, dtype=DTYPE),
            source_locations_pressure_rate=torch.tensor([[[ny // 2, nx // 2]]]),
            receiver_locations_p=torch.tensor([[[ny // 2 + 1, nx // 2]]]),
            pml_width=pml,
            pml_freq=25.0,
            q_n_mechanisms=n_mech,
            q_band=band,
            backend="eager",
            initial_state=initial_state,
        )

    # Distinct constants in the two mechanisms' memory variables, everything
    # else zero, and a zero source: nothing drives the memory variables, so
    # each simply decays by its own mechanism's factor. The memory variables
    # are auxiliary state, so they are seeded by replacing the tensors of a
    # state a run returned, which keeps that state's compatibility record.
    wshape = (1, ny + 2 * pml, nx + 2 * pml)
    starts = {"r_mech0": 3.0, "r_mech1": 7.0}
    from_rest = run()
    seeded = dataclasses.replace(
        from_rest.auxiliary_state,
        _tensors={
            **{n: torch.zeros(wshape, dtype=DTYPE) for n in from_rest.auxiliary_state},
            **{n: torch.full(wshape, c, dtype=DTYPE) for n, c in starts.items()},
        },
    )
    initial = {
        "pressure_0": torch.zeros(wshape, dtype=DTYPE),
        "vy_0": torch.zeros(wshape, dtype=DTYPE),
        "vx_0": torch.zeros(wshape, dtype=DTYPE),
        "auxiliary_state": seeded,
    }
    out = run(initial_state=initial)
    assert list(out.names) == expected

    # The decay factors the propagator's own model preparation derives, in
    # mechanism order -- the independent statement of which is which.
    ar = _vmod.prepare_models(v, rho, q, dt, 2.0 * math.pi * 25.0, n_mech, band)[2]
    assert ar.shape[0] == n_mech
    centre = (ny // 2, nx // 2)
    expected_ar = [float(ar[mech][centre]) for mech in range(n_mech)]
    # A real, wide separation, or the check below could not tell them apart.
    assert abs(expected_ar[0] - expected_ar[1]) > 0.1

    interior = (0, pml + ny // 2, pml + nx // 2)
    for mech, name in enumerate(("r_mech0", "r_mech1")):
        decayed = float(out.auxiliary_state[name][interior]) / starts[name]
        assert decayed == pytest.approx(expected_ar[mech], rel=1e-9), name

    # The attenuation and absorbing-boundary memory are both folded away.
    assert set(out.auxiliary_state) == set(
        _vmod.viscoacoustic_auxiliary_state_names(ndim, n_mech)
    )
    for gone in ("r_0", "r_1", "r_mech0", "r_mech1", "phi_y_0", "psi_x_0"):
        assert gone not in out.names
