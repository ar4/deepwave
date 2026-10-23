"""Degenerate-equivalence matrix: each propagator's limiting special cases.

Every propagator has parameter limits in which it must reduce to a simpler,
independently-tested propagator (or to zero). This table-driven matrix collects
those reductions and checks them on both the eager and compiled backends:

* ``scalar_born``/``acoustic_born``/``elastic_born`` with **zero scattering**
  reduce to their background propagator (``scalar``/``acoustic``/``elastic``),
  with identically-zero scattered receivers.
* ``acoustic`` with **constant density** obeys the same wave equation as
  ``scalar`` (same ``v``); the two agree only up to their different spatial
  discretisations (a staggered first-order velocity-stress system vs. a
  second-order scalar stencil), a level we *measure and pin* rather than assert
  as machine equality.
* ``elastic`` driven by an isotropic **pressure-rate** source in a homogeneous
  medium radiates only P, and its pressure matches ``scalar`` run at ``v_P``
  once the source-amplitude conversion documented in ``docs/units.rst`` is
  applied; this case is the numerical proof of that calibration.
* ``viscoacoustic`` in the **large-Q limit** reduces to ``acoustic``; the
  residual is the physical O(1/Q) attenuation, whose clean ``1/Q`` scaling this
  measures via a log-log fit.
* ``scalar`` at increasing **spatial accuracy order** converges to a
  fine-grid reference on a smooth model.

Two reductions already have dedicated tests, so they are referenced here and
**not** duplicated:

* ``elastic_vti`` with isotropic ``Cij`` == ``elastic`` -- see
  ``test_elastic_vti.py::test_isotropic_limit`` (forward + gradients, 2-D/3-D,
  both backends, to machine precision).
* ``viscoacoustic`` at a single very large ``Q`` == ``acoustic`` (forward +
  gradients) -- see ``test_viscoacoustic.py::test_acoustic_limit``. The
  ``viscoacoustic_q_scaling`` case below covers the *complementary* gap: the
  rate at which the residual vanishes as ``Q`` grows.
"""

import math

import numpy as np
import pytest
import torch
from helpers import c_backend_available

import deepwave
from deepwave import vpvsrho_to_lambmubuoyancy
from deepwave.acoustic import acoustic_output_names
from deepwave.acoustic_born import acoustic_born_output_names
from deepwave.elastic import elastic_output_names
from deepwave.elastic_born import elastic_born_output_names
from deepwave.viscoacoustic import viscoacoustic_output_names
from deepwave.wavelets import ricker

DTYPE = torch.float64
DEVICE = torch.device("cpu")
DX = 5.0

BACKENDS = [
    pytest.param("eager", id="eager"),
    pytest.param("compiled", id="c"),
]


def _require(backend, *props) -> None:
    """Skip a compiled-backend case if any needed C propagator is absent."""
    if backend == "compiled":
        for prop in props:
            if not c_backend_available(prop, device=DEVICE):
                pytest.skip(f"compiled {prop} backend not present in libdeepwave_C")


def _survey(n, nt, dt, freq, ndim=2):
    """A central source and a short line of near-surface receivers."""
    src = ricker(freq, nt, dt, 1.5 / freq).to(DTYPE).reshape(1, 1, -1)
    src_loc = torch.tensor([[[n // 2] * ndim]], dtype=torch.long)
    rec_loc = torch.zeros(1, 5, ndim, dtype=torch.long)
    for j in range(5):
        rec_loc[0, j, 0] = 3 + j
        rec_loc[0, j, 1:] = n // 2
    return src, src_loc, rec_loc


def _iso_media(n):
    """Deterministic isotropic elastic Lame/buoyancy models."""
    torch.manual_seed(7)
    shape = (n, n)
    vp = 1600.0 + 50.0 * torch.rand(shape, dtype=DTYPE)
    vs = 800.0 + 50.0 * torch.rand(shape, dtype=DTYPE)
    rho = 2000.0 + 100.0 * torch.rand(shape, dtype=DTYPE)
    lamb, mu, buoyancy = (t.detach() for t in vpvsrho_to_lambmubuoyancy(vp, vs, rho))
    return lamb, mu, buoyancy


def _case_scalar_born_zero_scatter(backend) -> None:
    """scalar_born with zero scatter == scalar background, zero scattered.

    With ``scatter == 0`` the Born scattered source vanishes, so the background
    field must equal the plain ``scalar`` output and every scattered receiver
    must be identically zero.
    """
    _require(backend, "scalar", "scalar_born")
    n, nt, dt, freq = 24, 40, 0.0006, 25.0
    torch.manual_seed(0)
    v = 1500.0 + 20.0 * torch.randn(n, n, dtype=DTYPE)
    src, sl, rl = _survey(n, nt, dt, freq)
    common = {
        "pml_width": 4,
        "pml_freq": freq,
        "accuracy": 4,
        "backend": backend,
    }
    born = deepwave.scalar_born(
        v,
        DX,
        dt,
        dv=torch.zeros_like(v),
        source_amplitudes=src,
        source_locations=sl,
        sc_receiver_locations=rl,
        bg_receiver_locations=rl,
        **common,
    )
    ref = deepwave.scalar(
        v,
        DX,
        dt,
        source_amplitudes=src,
        source_locations=sl,
        receiver_locations=rl,
        **common,
    )
    bg_rec, sc_rec = born[-2], born[-1]
    scale = ref[-1].abs().max().item()
    assert (bg_rec - ref[-1]).abs().max().item() <= 1e-10 * scale
    assert sc_rec.abs().max().item() <= 1e-12 * scale


def _case_acoustic_born_zero_scatter(backend) -> None:
    """acoustic_born with zero dv/drho == acoustic background, zero scattered."""
    _require(backend, "acoustic", "acoustic_born")
    n, nt, dt, freq = 24, 40, 0.0006, 25.0
    torch.manual_seed(0)
    v = 1500.0 + 20.0 * torch.randn(n, n, dtype=DTYPE)
    rho = 1000.0 + 30.0 * torch.randn(n, n, dtype=DTYPE)
    src, sl, rl = _survey(n, nt, dt, freq)
    common = {
        "pml_width": 4,
        "pml_freq": freq,
        "accuracy": 4,
        "backend": backend,
    }
    born = deepwave.acoustic_born(
        v,
        rho,
        DX,
        dt,
        dv=torch.zeros_like(v),
        drho=torch.zeros_like(rho),
        source_amplitudes_volume_rate=src,
        source_locations_volume_rate=sl,
        sc_receiver_locations_p=rl,
        bg_receiver_locations_p=rl,
        **common,
    )
    ref = deepwave.acoustic(
        v,
        rho,
        DX,
        dt,
        source_amplitudes_volume_rate=src,
        source_locations_volume_rate=sl,
        receiver_locations_p=rl,
        **common,
    )
    names = acoustic_born_output_names(2)
    bg = born[names.index("bg_receiver_amplitudes_p")]
    sc = born[names.index("sc_receiver_amplitudes_p")]
    ref_p = ref[acoustic_output_names(2).index("receiver_amplitudes_p")]
    scale = ref_p.abs().max().item()
    assert (bg - ref_p).abs().max().item() <= 1e-10 * scale
    assert sc.abs().max().item() <= 1e-12 * scale


def _case_elastic_born_zero_scatter(backend) -> None:
    """elastic_born with zero dlamb/dmu/dbuoyancy == elastic, zero scattered."""
    _require(backend, "elastic", "elastic_born")
    n, nt, dt, freq = 24, 40, 0.0006, 25.0
    lamb, mu, buoyancy = _iso_media(n)
    src, sl, rl = _survey(n, nt, dt, freq)
    common = {
        "pml_width": 4,
        "pml_freq": freq,
        "accuracy": 4,
        "backend": backend,
    }
    born = deepwave.elastic_born(
        lamb,
        mu,
        buoyancy,
        DX,
        dt,
        dlamb=torch.zeros_like(lamb),
        dmu=torch.zeros_like(mu),
        dbuoyancy=torch.zeros_like(buoyancy),
        source_amplitudes_pressure_rate=src,
        source_locations_pressure_rate=sl,
        sc_receiver_locations_p=rl,
        bg_receiver_locations_p=rl,
        **common,
    )
    ref = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        DX,
        dt,
        source_amplitudes_pressure_rate=src,
        source_locations_pressure_rate=sl,
        receiver_locations_p=rl,
        **common,
    )
    names = elastic_born_output_names(2)
    bg = born[names.index("bg_receiver_amplitudes_p")]
    sc = born[names.index("sc_receiver_amplitudes_p")]
    ref_p = ref[elastic_output_names(2).index("receiver_amplitudes_p")]
    scale = ref_p.abs().max().item()
    assert (bg - ref_p).abs().max().item() <= 1e-10 * scale
    assert sc.abs().max().item() <= 1e-12 * scale


def _case_acoustic_const_rho_equals_scalar(backend) -> None:
    """Constant-density acoustic == scalar (same v), up to discretisation.

    A constant-density acoustic pressure field satisfies the same second-order
    scalar wave equation as ``scalar``. The source correspondence is exact only
    up to the source-injection convention: an acoustic pressure source of
    amplitude ``f`` matches a scalar source of ``-rho * df/dt`` (the acoustic
    pressure response equals the scalar Green's function driven by that scaled,
    differentiated source; see ``test_acoustic.py::run_direct`` and
    ``test_scalar.py::run_direct``). The two propagators use *different* spatial
    schemes (staggered first-order velocity-stress vs. a second-order scalar
    stencil), so they agree only to the spatial-discretisation level. That level
    is measured here (~3% relative on this geometry) and pinned from both sides:
    below 5% (they really do model the same physics) and above 1e-3 (they are
    genuinely different schemes, not a machine-precision restatement).
    """
    _require(backend, "acoustic", "scalar")
    n, dt, freq, c = 70, 0.0004, 20.0, 1500.0
    nt = 400
    v = torch.full((n, n), c, dtype=DTYPE)
    rho_val = 2000.0
    rho = torch.full((n, n), rho_val, dtype=DTYPE)
    src = ricker(freq, nt, dt, 1.5 / freq).to(DTYPE).reshape(1, 1, -1)
    sl = torch.tensor([[[n // 2, n // 2]]])
    rl = torch.tensor([[[n // 2 + 12, n // 2]]])
    common = {"pml_width": 20, "pml_freq": freq, "accuracy": 8, "max_vel": c}
    ac = deepwave.acoustic(
        v,
        rho,
        DX,
        dt,
        source_amplitudes_volume_rate=src,
        source_locations_volume_rate=sl,
        receiver_locations_p=rl,
        backend=backend,
        **common,
    )[acoustic_output_names(2).index("receiver_amplitudes_p")][0, 0]
    src_diff = torch.gradient(src, dim=-1, spacing=dt)[0]
    sc = deepwave.scalar(
        v,
        DX,
        dt,
        source_amplitudes=-rho_val * src_diff,
        source_locations=sl,
        receiver_locations=rl,
        backend=backend,
        **common,
    )[-1][0, 0]
    m = min(len(ac), len(sc))
    rel = ((ac[:m] - sc[:m]).norm() / sc[:m].norm()).item()
    assert 1e-3 < rel < 5e-2, rel


def _case_viscoacoustic_q_scaling(backend) -> None:
    """Large-Q viscoacoustic -> acoustic residual scales as O(1/Q).

    ``test_viscoacoustic.py::test_acoustic_limit`` already pins the single-point
    equality at ``Q = 1e12``. This complementary case measures the *rate*: the
    relative residual between ``viscoacoustic`` and ``acoustic`` must fall as
    ``1/Q`` (physical single-SLS attenuation), i.e. a log-log slope near -1 and
    a strictly decreasing residual as ``Q`` doubles.
    """
    _require(backend, "acoustic", "viscoacoustic")
    n, nt, dt, freq = 24, 40, 0.0005, 25.0
    torch.manual_seed(0)
    v = 1500.0 + 20.0 * torch.randn(n, n, dtype=DTYPE)
    rho = 1000.0 + 30.0 * torch.randn(n, n, dtype=DTYPE)
    src, sl, rl = _survey(n, nt, dt, freq)
    common = {
        "pml_width": 4,
        "pml_freq": freq,
        "accuracy": 4,
        "backend": backend,
    }
    ac_idx = acoustic_output_names(2).index("receiver_amplitudes_p")
    vo_idx = viscoacoustic_output_names(2).index("receiver_amplitudes_p")
    ac = deepwave.acoustic(
        v,
        rho,
        DX,
        dt,
        source_amplitudes_volume_rate=src,
        source_locations_volume_rate=sl,
        receiver_locations_p=rl,
        **common,
    )[ac_idx]
    qs = [50.0, 100.0, 200.0, 400.0, 800.0]
    res = []
    for q_val in qs:
        q = torch.full((n, n), q_val, dtype=DTYPE)
        vo = deepwave.viscoacoustic(
            v,
            rho,
            q,
            DX,
            dt,
            source_amplitudes_pressure_rate=(
                rho[sl[..., 0], sl[..., 1]] * v[sl[..., 0], sl[..., 1]] ** 2
            ).unsqueeze(-1)
            * src,
            source_locations_pressure_rate=sl,
            receiver_locations_p=rl,
            q_freq=freq,
            **common,
        )[vo_idx]
        res.append(((vo - ac).norm() / ac.norm()).item())
    # Strictly decreasing, and a clean 1/Q log-log slope (measured ~ -1.02).
    assert all(res[i + 1] < res[i] for i in range(len(res) - 1)), res
    slope = float(np.polyfit(np.log(qs), np.log(res), 1)[0])
    assert -1.15 < slope < -0.85, slope


def _case_scalar_accuracy_convergence(backend) -> None:
    """scalar: higher spatial accuracy converges to a fine-grid reference.

    On a smooth (depth-varying) velocity model, the coarse-grid ``scalar``
    waveform approaches a 3x-refined ``accuracy=8`` reference as the coarse
    accuracy order rises (2 -> 4 -> 8). Waveforms are peak-independent-normalised
    before comparison because a point source's receiver amplitude scales with
    the grid cell size (``dx**ndim``), so the *shape* (dispersion) is the
    order-dependent quantity, not the absolute amplitude. This complements
    ``test_scalar.py::test_direct`` (constant model, analytic Green's-function
    reference) by using a heterogeneous model and a numerical fine-grid
    reference. Measured (float64): 2 -> ~0.32, 4 -> ~0.036, 8 -> ~0.0018.
    """
    _require(backend, "scalar")
    freq, dt = 15.0, 0.0003
    n, dx, nt = 50, 10.0, 800
    src = ricker(freq, nt, dt, 1.5 / freq).to(DTYPE).reshape(1, 1, -1)

    def vmodel(nn, dxx):
        depth = torch.arange(nn, dtype=DTYPE).reshape(-1, 1) * dxx
        return (1500.0 + 0.35 * depth).repeat(1, nn)

    def run(nn, dxx, src_i, rec_i, acc):
        v = vmodel(nn, dxx)
        sl = torch.tensor([[list(src_i)]])
        rl = torch.tensor([[list(rec_i)]])
        return deepwave.scalar(
            v,
            dxx,
            dt,
            source_amplitudes=src,
            source_locations=sl,
            receiver_locations=rl,
            accuracy=acc,
            pml_width=20,
            pml_freq=freq,
            max_vel=3500.0,
            backend=backend,
        )[-1][0, 0]

    refine = 3
    fine = run(
        refine * (n - 1) + 1,
        dx / refine,
        (refine * 12, refine * 25),
        (refine * 34, refine * 25),
        8,
    )
    fine_n = fine / fine.norm()
    misfit = {}
    for acc in (2, 4, 8):
        coarse = run(n, dx, (12, 25), (34, 25), acc)
        coarse_n = coarse / coarse.norm()
        m = min(len(coarse_n), len(fine_n))
        misfit[acc] = (coarse_n[:m] - fine_n[:m]).norm().item()
    assert misfit[4] < 0.3 * misfit[2], misfit
    assert misfit[8] < 0.3 * misfit[4], misfit


def _case_elastic_p_source_equals_scalar(backend) -> None:
    r"""Elastic pressure source == scalar source, after the documented conversion.

    This is the numerical proof of the source-amplitude calibration table in
    ``docs/units.rst``. An isotropic (pressure-rate) source in a *homogeneous*
    elastic medium radiates only P, and the resulting pressure obeys the same
    second-order wave equation as ``scalar`` run with ``v = v_P``. The table's
    conversion between the two source arguments is

        source_amplitudes(scalar) = -(K_n / (rho v_P^4)) d/dt
                                        source_amplitudes_pressure_rate(elastic)

    with ``K_n = lamb + 2 mu / ndim`` the modulus that multiplies the
    dilatation rate in the trace of the stress (``lamb + mu`` in 2-D). ``K_n``
    is not a detail: at ``v_P = 1500``, ``v_S = 800`` it is a 40% amplitude
    change, and using the 3-D form in 2-D is a 13% one.

    The elastic pressure source is added to the update from ``p(t dt)`` to
    ``p((t+1) dt)``, so its sample ``t`` acts at ``(t + 1/2) dt`` -- half a
    step later than a ``scalar`` source sample, which acts at ``t dt``. The
    elastic wavelet is therefore evaluated at ``(t + 1/2) dt``. Both
    propagators' pressure receivers report ``t dt``, so no receiver-side
    correction is needed. Leaving the half step out costs
    ``omega dt / 2 = 1.9%`` of relative waveform error here, an order of
    magnitude above the tolerance below.

    The time derivative is applied analytically rather than numerically: the
    Ricker wavelet ``(1 - 2 pi^2 f^2 tau^2) exp(-pi^2 f^2 tau^2)`` is exactly
    the derivative of ``tau exp(-pi^2 f^2 tau^2)``, so the elastic source is
    that antiderivative and no finite-difference error enters the comparison.

    Tolerance: the two schemes discretise the same continuum problem
    differently (second-order-in-time regular grid vs. staggered
    velocity-stress), so they agree only to their discretisation error. Their
    temporal dispersion relations coincide, and at the 25 points per
    wavelength used here the difference in *spatial* dispersion (a 4th-order
    central second derivative, phase error ``theta^4/180``, vs. 4th-order
    staggered first derivatives, ``3 theta^4/640``) predicts only ~4e-5 over
    the 1.6-wavelength path; the rest comes from the two schemes' different
    discrete point-source and PML representations, giving an expected
    residual of a few times 1e-4. Measured (float64, both backends
    identical): peak amplitudes 0.9778309 vs. 0.9776793, ratio 0.99984500,
    relative L2 1.84e-4. Halving the grid spacing and time step takes that to
    2.3e-5, confirming it is discretisation error. The assertions below are
    set at 1e-3, above the measurement but far below the 1.9e-2 of a missing
    half step or the 4e-1 of a missing ``K_n``.
    """
    _require(backend, "scalar", "elastic")
    n, dx, dt, nt = 160, 4.0, 0.0004, 600
    freq, peak_time = 15.0, 0.08
    vp_val, vs_val, rho_val = 1500.0, 800.0, 1000.0
    ndim = 2

    shape = (n, n)
    lamb, mu, buoyancy = (
        t.detach()
        for t in vpvsrho_to_lambmubuoyancy(
            torch.full(shape, vp_val, dtype=DTYPE),
            torch.full(shape, vs_val, dtype=DTYPE),
            torch.full(shape, rho_val, dtype=DTYPE),
        )
    )
    k_n = lamb[0, 0].item() + 2 * mu[0, 0].item() / ndim

    src_scalar = ricker(freq, nt, dt, peak_time).to(DTYPE)
    # Antiderivative of the Ricker, sampled at the (t + 1/2) dt times at which
    # the elastic pressure-source samples act.
    tau = (torch.arange(float(nt), dtype=DTYPE) + 0.5) * dt - peak_time
    src_elastic = (
        -(rho_val * vp_val**4 / k_n) * tau * torch.exp(-(math.pi**2) * freq**2 * tau**2)
    )

    sl = torch.tensor([[[n // 2, n // 2]]])
    rl = torch.tensor([[[n // 2, n // 2 + 40]]])
    common = {
        "pml_width": 20,
        "pml_freq": freq,
        "accuracy": 4,
        "max_vel": vp_val,
        "backend": backend,
    }
    sc = deepwave.scalar(
        torch.full(shape, vp_val, dtype=DTYPE),
        dx,
        dt,
        source_amplitudes=src_scalar.reshape(1, 1, -1),
        source_locations=sl,
        receiver_locations=rl,
        **common,
    )[-1][0, 0]
    el = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_pressure_rate=src_elastic.reshape(1, 1, -1),
        source_locations_pressure_rate=sl,
        receiver_locations_p=rl,
        **common,
    )[elastic_output_names(ndim).index("receiver_amplitudes_p")][0, 0]

    peak_ratio = (el.abs().max() / sc.abs().max()).item()
    rel = ((el - sc).norm() / sc.norm()).item()
    assert abs(peak_ratio - 1.0) < 1e-3, peak_ratio
    assert rel < 1e-3, rel


_CASES = {
    "scalar_born_zero_scatter": _case_scalar_born_zero_scatter,
    "acoustic_born_zero_scatter": _case_acoustic_born_zero_scatter,
    "elastic_born_zero_scatter": _case_elastic_born_zero_scatter,
    "acoustic_const_rho_equals_scalar": _case_acoustic_const_rho_equals_scalar,
    "elastic_p_source_equals_scalar": _case_elastic_p_source_equals_scalar,
    "viscoacoustic_q_scaling": _case_viscoacoustic_q_scaling,
    "scalar_accuracy_convergence": _case_scalar_accuracy_convergence,
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", list(_CASES))
def test_equivalence(case, backend) -> None:
    """Run one degenerate-equivalence reduction on one backend.

    Each case asserts that a propagator, in a stated parameter limit, reduces to
    a simpler independently-tested propagator (or to zero) at the documented
    tolerance. See the per-case helper docstrings for the physics and the pinned
    residual levels.
    """
    _CASES[case](backend)
