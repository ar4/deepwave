"""Tests for the parameter-modified (PM) free surface of the elastic propagator.

The PM method (Cao and Chen, 2018, *Geophysics* 83(6), T313-T332,
https://doi.org/10.1190/geo2018-0098.1) realises an arbitrary staircase
free-surface topography by modifying the medium parameters once, before time
stepping: the per-component normal-stress stiffnesses at the discrete surface
cells are replaced by the stress-free-reduced values (with the modified
P-modulus ``alpha = 2 mu (lamb + mu) / (lamb + 2 mu)``) and the surface-plane
densities are halved. It is selected with ``surface_method="pm"`` and is
implemented in 2-D on both the eager backend and the default compiled C/CUDA
backend, and in 3-D on the eager backend (see
``docs/dev/boundary_conditions/CAO_CHEN_PM_SUMMARY.md`` for the derivation
record, including the 3-D Table 2 re-derivation).

The tests cover: the input validation; the interior-unchanged identity (PM
reduces exactly to the stock scheme where there is no vacuum); the flat-surface
reduction to the analytic horizontal-surface coefficients; a Rayleigh-wave
analytic oracle showing PM is at least as accurate as the improved vacuum method
(IVM) at large Poisson's ratio; a topography benchmark reproducing the paper's
finding that PM is markedly more accurate than IVM on rugged surfaces across the
Poisson range; reciprocity; the adjoint dot-product identity and a
finite-difference gradcheck through the modified model preparation; a negative
control confirming the surface modification is actually exercised; and the
compiled-backend gates (compiled-vs-eager forward and gradient parity, the
compiled hand-derived adjoint's dot-product identity, and a compiled gradcheck)
on CPU and GPU.
"""

import copy
import math

import pytest
import torch

from deepwave import elastic, vpvsrho_to_lambmubuoyancy
from deepwave.elastic import (
    _pm_modify_buoyancy,
    _pm_modify_shear,
    _pm_normal_stiffness,
    _pm_surface_masks,
    prepare_parameters,
    update_stresses,
    update_stresses_pm,
    update_velocities,
    update_velocities_pm,
)
from deepwave.wavelets import ricker

DTYPE = torch.float64


def _rayleigh_velocity(cl: float, ct: float) -> float:
    """Rayleigh-wave phase velocity: the root of the Rayleigh function in (0, ct).

    ``(2 - (c/ct)^2)^2 - 4 sqrt(1 - (c/cl)^2) sqrt(1 - (c/ct)^2) = 0``. The
    Rayleigh wave of a homogeneous half-space is non-dispersive, so this single
    root is the analytic phase velocity at every frequency.
    """

    def rayleigh(c: float) -> float:
        return (2 - (c / ct) ** 2) ** 2 - 4 * math.sqrt(
            max(1 - (c / cl) ** 2, 0.0)
        ) * math.sqrt(max(1 - (c / ct) ** 2, 0.0))

    lo, hi = 0.1 * ct, 0.9999 * ct
    flo = rayleigh(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fmid = rayleigh(mid)
        if flo * fmid <= 0:
            hi = mid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def _solid_model(ny: int = 20, nx: int = 24):
    vp = torch.full((ny, nx), 2000.0, dtype=DTYPE)
    vs = torch.full((ny, nx), 1150.0, dtype=DTYPE)
    rho = torch.full((ny, nx), 2000.0, dtype=DTYPE)
    return vpvsrho_to_lambmubuoyancy(vp, vs, rho)


def _free_surface_model(ny: int = 30, nx: int = 40, vac: int = 5, vs: float = 1150.0):
    vp_t = torch.zeros(ny, nx, dtype=DTYPE)
    vs_t = torch.zeros(ny, nx, dtype=DTYPE)
    rho_t = torch.zeros(ny, nx, dtype=DTYPE)
    vp_t[vac:] = 2000.0
    vs_t[vac:] = vs
    rho_t[vac:] = 2000.0
    return vpvsrho_to_lambmubuoyancy(vp_t, vs_t, rho_t)


def _tiny_source_2d():
    amp = torch.zeros(1, 1, 3, dtype=DTYPE)
    loc = torch.tensor([[[4, 4]]])
    return amp, loc


def _tiny_source_3d():
    amp = torch.zeros(1, 1, 3, dtype=DTYPE)
    loc = torch.tensor([[[4, 4, 4]]])
    return amp, loc


def test_pm_requires_vacuum_raises() -> None:
    """``surface_method="pm"`` on a purely solid model raises ``ValueError``."""
    lamb, mu, buoyancy = _solid_model()
    amp, loc = _tiny_source_2d()
    with pytest.raises(ValueError, match="requires vacuum cells"):
        elastic(
            lamb,
            mu,
            buoyancy,
            5.0,
            0.001,
            source_amplitudes_y=amp,
            source_locations_y=loc,
            surface_method="pm",
            backend="eager",
        )


def test_pm_fluid_layer_raises() -> None:
    """PM refuses models containing fluid (``mu == 0``, ``lamb != 0``) cells.

    The classifier defines vacuum by ``mu == 0`` alone, so a water layer would
    be indistinguishable from air and the solid beneath it would wrongly be
    modified as a free surface; the validation must reject this loudly rather
    than produce a silently wrong fluid-solid interface.
    """
    ny, nx, vac, water = 24, 20, 3, 4
    vp = torch.zeros(ny, nx, dtype=DTYPE)
    vs = torch.zeros(ny, nx, dtype=DTYPE)
    rho = torch.zeros(ny, nx, dtype=DTYPE)
    vp[vac:] = 1500.0
    rho[vac:] = 1000.0
    vp[vac + water :] = 2000.0
    vs[vac + water :] = 1150.0
    rho[vac + water :] = 2000.0
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    amp, loc = _tiny_source_2d()
    with pytest.raises(ValueError, match="fluid"):
        elastic(
            lamb,
            mu,
            buoyancy,
            5.0,
            0.001,
            source_amplitudes_y=amp,
            source_locations_y=loc,
            surface_method="pm",
            backend="eager",
        )


def test_pm_invalid_surface_method_raises() -> None:
    """An unrecognised ``surface_method`` string raises ``ValueError``."""
    lamb, mu, buoyancy = _free_surface_model()
    amp, loc = _tiny_source_2d()
    with pytest.raises(ValueError, match='surface_method must be "ivm" or "pm"'):
        elastic(
            lamb,
            mu,
            buoyancy,
            5.0,
            0.001,
            source_amplitudes_y=amp,
            source_locations_y=loc,
            surface_method="vacuum",  # type: ignore[arg-type]
            backend="eager",
        )


@pytest.mark.parametrize("backend", ["torch_compile"])
def test_pm_compile_backend_raises(backend) -> None:
    """PM on the compiled Python mode raises ``NotImplementedError``.

    The eager backend and the default compiled C/CUDA backend
    (``backend="compiled"``) both support PM; only
    ``backend="torch_compile"`` does not.
    """
    lamb, mu, buoyancy = _free_surface_model()
    amp, loc = _tiny_source_2d()
    with pytest.raises(NotImplementedError):
        elastic(
            lamb,
            mu,
            buoyancy,
            5.0,
            0.001,
            source_amplitudes_y=amp,
            source_locations_y=loc,
            surface_method="pm",
            backend=backend,
        )


def test_pm_3d_compiled_raises() -> None:
    """3D PM is eager-only; the compiled backend raises ``NotImplementedError``."""
    n = 12
    vp = torch.zeros(n, n, n, dtype=DTYPE)
    vs = torch.zeros(n, n, n, dtype=DTYPE)
    rho = torch.zeros(n, n, n, dtype=DTYPE)
    vp[3:] = 2000.0
    vs[3:] = 1150.0
    rho[3:] = 2000.0
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    amp, loc = _tiny_source_3d()
    with pytest.raises(NotImplementedError, match="eager"):
        elastic(
            lamb,
            mu,
            buoyancy,
            5.0,
            0.0005,
            source_amplitudes_y=amp,
            source_locations_y=loc,
            surface_method="pm",
        )


# --------------------------------------------------------------------------
# Interior-unchanged identity (gate-0 verify_topo.py check A)
# --------------------------------------------------------------------------


def _fake_pml(ny: int, nx: int):
    ones = torch.ones(1, ny, nx, dtype=DTYPE)
    zeros = torch.zeros(1, ny, nx, dtype=DTYPE)
    return [ones, zeros, ones, zeros] * 2  # a, b, ah, bh per dim


def test_pm_interior_unchanged_kernels() -> None:
    """With no vacuum, the PM update kernels reproduce the stock kernels.

    The parameter-modified surface cells are the only difference between PM and
    the interior scheme; where there is no adjacent vacuum, the classifier marks
    nothing, so ``_pm_normal_stiffness`` returns the interior coefficients and
    ``_pm_modify_buoyancy`` returns the buoyancies unchanged. The PM kernels must
    then match the stock kernels to rounding (they differ only in the order of a
    couple of floating-point operations). This is the permanent form of gate-0's
    verify_topo.py check A (which drove the public API, but that now raises for a
    vacuum-free model, so the identity is checked at the kernel level here).
    """
    torch.manual_seed(0)
    ny, nx = 20, 24
    lamb, mu, buoyancy = _solid_model(ny, nx)
    models = [lamb.unsqueeze(0), mu.unsqueeze(0)]
    models = models + prepare_parameters(mu.unsqueeze(0), buoyancy.unsqueeze(0))
    grid_spacing = [5.0, 5.0]
    dt = 0.001

    velocities = [torch.randn(1, ny, nx, dtype=DTYPE) for _ in range(2)]
    normal = [torch.randn(1, ny, nx, dtype=DTYPE) for _ in range(2)]
    shear_val = torch.randn(1, ny, nx, dtype=DTYPE)
    shear = [[torch.empty(0), shear_val], [shear_val, torch.empty(0)]]

    def zmem_normal():
        return [torch.zeros(1, ny, nx, dtype=DTYPE) for _ in range(2)]

    def zmem_shear():
        z = torch.zeros(1, ny, nx, dtype=DTYPE)
        return [[torch.empty(0), z.clone()], [z.clone(), torch.empty(0)]]

    stock_s = update_stresses(
        models,
        velocities,
        [x.clone() for x in normal],
        copy.deepcopy(shear),
        zmem_normal(),
        zmem_shear(),
        _fake_pml(ny, nx),
        grid_spacing,
        dt,
        4,
    )
    pm_s = update_stresses_pm(
        models,
        velocities,
        [x.clone() for x in normal],
        copy.deepcopy(shear),
        zmem_normal(),
        zmem_shear(),
        _fake_pml(ny, nx),
        grid_spacing,
        dt,
        4,
    )
    for i in range(2):
        scale = stock_s[0][i].abs().max().item()
        assert (stock_s[0][i] - pm_s[0][i]).abs().max().item() / scale < 1e-12

    stock_v = update_velocities(
        models,
        [x.clone() for x in normal],
        copy.deepcopy(shear),
        [x.clone() for x in velocities],
        zmem_normal(),
        zmem_shear(),
        _fake_pml(ny, nx),
        grid_spacing,
        dt,
        4,
    )
    pm_v = update_velocities_pm(
        models,
        [x.clone() for x in normal],
        copy.deepcopy(shear),
        [x.clone() for x in velocities],
        zmem_normal(),
        zmem_shear(),
        _fake_pml(ny, nx),
        grid_spacing,
        dt,
        4,
    )
    for i in range(2):
        scale = stock_v[0][i].abs().max().item()
        assert (stock_v[0][i] - pm_v[0][i]).abs().max().item() / scale < 1e-12


# --------------------------------------------------------------------------
# Flat-surface reduction to the analytic horizontal-surface coefficients
# (gate-0 verify_topo.py check B)
# --------------------------------------------------------------------------


def test_pm_flat_reduction_to_horizontal_coefficients() -> None:
    """On a flat surface the classifier and coefficients equal the H-cell values.

    A flat free surface (vacuum rows above a solid half-space) must be classified
    as ``H`` (horizontal) on exactly the first solid row and nowhere else, and
    the surface-modified coefficients must equal the analytic Cao-Chen Table 1
    horizontal-surface values there: the ``sigma_xx`` stiffness reduces to
    ``alpha`` on ``dvx/dx`` with the other three entries frozen to zero, and the
    ``x``-face buoyancy is doubled (the halved surface density) while the
    ``y``-face buoyancy is unchanged. Interior rows keep the isotropic
    ``(lamb+2mu, lamb)`` stiffness and unmodified buoyancy exactly.
    """
    ny, nx, vac = 24, 20, 5
    lamb, mu, buoyancy = _free_surface_model(ny, nx, vac)
    lamb = lamb.unsqueeze(0)
    mu = mu.unsqueeze(0)
    _mu_yx, buoyancy_y, buoyancy_x = prepare_parameters(mu, buoyancy.unsqueeze(0))

    face, transition = _pm_surface_masks(mu, 2)
    horizontal = face[0]
    # H exactly on the first solid row; no vertical or corner cells on a flat
    # surface.
    assert bool(horizontal[0, vac].all())
    assert int(horizontal.sum().item()) == nx
    assert int((face[1] | transition).sum().item()) == 0

    c = _pm_normal_stiffness(lamb, mu, (face, transition))
    lp2m = (lamb + 2.0 * mu)[0, vac]
    alpha = (2.0 * mu * (lamb + mu) / (lamb + 2.0 * mu))[0, vac]
    zero = torch.zeros_like(alpha)
    # Surface row: sigma_yy frozen, sigma_xx = alpha * dvx/dx.
    assert torch.allclose(c[0][0][0, vac], zero)
    assert torch.allclose(c[0][1][0, vac], zero)
    assert torch.allclose(c[1][0][0, vac], zero)
    assert torch.allclose(c[1][1][0, vac], alpha)
    # Interior row (well below the surface): unchanged isotropic coefficients.
    interior = vac + 6
    assert torch.allclose(c[0][0][0, interior], (lamb + 2.0 * mu)[0, interior])
    assert torch.allclose(c[1][1][0, interior], (lamb + 2.0 * mu)[0, interior])
    assert torch.allclose(c[0][1][0, interior], lamb[0, interior])
    assert torch.allclose(c[1][0][0, interior], lamb[0, interior])
    # Sanity: alpha is a genuine reduction of the P-modulus (not equal to it).
    assert (alpha < 0.999 * lp2m).all()

    by, bx = _pm_modify_buoyancy(
        [buoyancy_y.clone(), buoyancy_x.clone()], (face, transition)
    )
    assert torch.allclose(bx[0, vac], 2.0 * buoyancy_x[0, vac])
    assert torch.allclose(by[0, vac], buoyancy_y[0, vac])
    assert torch.allclose(bx[0, interior], buoyancy_x[0, interior])


# --------------------------------------------------------------------------
# Rayleigh-wave analytic oracle (Lamb's problem): PM error <= IVM error
# --------------------------------------------------------------------------


def _measure_rayleigh_velocity(method: str, vp: float, vs: float, rho: float) -> float:
    """Phase velocity of the Rayleigh wave on a flat half-space (m/s).

    A vertical force below a flat free surface launches a Rayleigh wave; its
    phase velocity is read from the phase difference of the ``vy`` component at
    two surface receivers a fraction of a wavelength apart, via a
    single-frequency DFT at the source's dominant frequency (the same phase
    measurement as ``tests/test_elastic_lamb.py``).
    """
    dx = 5.0
    ppw = 15.0
    cr = _rayleigh_velocity(vp, vs)
    f0 = vs / (ppw * dx)
    lam_r = cr / f0
    vac = 6
    surf = vac
    ny = vac + 120
    nx = 420
    dt = 0.5 * dx / vp
    nt = min(round(2.2 * (nx * dx) / cr / dt), 4000)
    omega = 2 * math.pi * f0

    vp_t = torch.zeros(ny, nx, dtype=DTYPE)
    vs_t = torch.zeros(ny, nx, dtype=DTYPE)
    rho_t = torch.zeros(ny, nx, dtype=DTYPE)
    vp_t[vac:] = vp
    vs_t[vac:] = vs
    rho_t[vac:] = rho
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp_t, vs_t, rho_t)

    from deepwave.wavelets import tone_burst

    wavelet = tone_burst(f0, 4, nt=nt, dt=dt, peak_time=4 / f0 / 2, dtype=DTYPE)
    src_amp = wavelet.reshape(1, 1, -1)
    src_loc = torch.tensor([[[surf + 3, 60]]])
    sep = max(2, round(0.3 * lam_r / dx))
    rx = (260, 260 + sep)
    rec_loc = torch.tensor([[[surf, rx[0]], [surf, rx[1]]]])

    out = elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=src_amp,
        source_locations_y=src_loc,
        receiver_locations_y=rec_loc,
        pml_width=[0, 20, 20, 20],
        pml_freq=f0,
        accuracy=4,
        max_vel=vp,
        surface_method=method,
        backend="eager",
    )
    rec = out.receiver_amplitudes_y[0]
    n = torch.arange(nt, dtype=DTYPE)
    phasor = torch.exp(-1j * omega * n * dt)
    spec = (rec * phasor).sum(dim=1)
    dphi = (torch.angle(spec[1]) - torch.angle(spec[0])).item()
    dphi = (dphi + math.pi) % (2 * math.pi) - math.pi
    return omega * (rx[1] - rx[0]) * dx / (-dphi)


def test_pm_rayleigh_oracle_high_poisson(record_property) -> None:
    """At large Poisson's ratio PM is at least as accurate as IVM (Rayleigh).

    The paper's headline: the improved vacuum method degrades badly at high
    Poisson's ratio while PM stays accurate. On a flat half-space with the
    paper's crest-model parameters (Vp=4500, Vs=1357, sigma~0.45) the measured
    Rayleigh-wave phase velocity of PM must be no further from the analytic
    Rayleigh velocity than IVM's, and within about 1% of it (well within twice
    the paper's sub-percent ppw-15 misfit claim, Table 4).
    """
    vp, vs, rho = 4500.0, 1357.0, 1500.0
    cr = _rayleigh_velocity(vp, vs)
    c_ivm = _measure_rayleigh_velocity("ivm", vp, vs, rho)
    c_pm = _measure_rayleigh_velocity("pm", vp, vs, rho)
    err_ivm = abs(c_ivm - cr) / cr
    err_pm = abs(c_pm - cr) / cr
    record_property("pm_relative_phase_error", err_pm)
    record_property("ivm_relative_phase_error", err_ivm)
    assert err_pm <= err_ivm + 1e-3, (
        f"PM Rayleigh error {err_pm:.4f} should be <= IVM error {err_ivm:.4f} "
        f"(cR={cr:.1f}, PM={c_pm:.1f}, IVM={c_ivm:.1f})"
    )
    assert err_pm < 0.01, f"PM Rayleigh phase-velocity error {err_pm:.4f} too large"


# --------------------------------------------------------------------------
# Topography benchmark across Poisson's ratio: PM markedly beats IVM
# --------------------------------------------------------------------------


def _run_trough(method: str, vp: float, vs: float, dx: float, dt: float, nt: int):
    """A staircase trough half-space; returns stacked (vy, vx) receiver traces.

    The vacuum region defines a symmetric trough (40% grade flanks) so the H,
    VL, VR and corner cell types are all exercised. Receivers sit just below the
    surface on both flanks.
    """
    width, depth, pml_m, freq = 600.0, 360.0, 100.0, 12.0
    rho = 1500.0

    def topo(x: float) -> float:
        r = max(min(x, width - x), 0.0)
        return 48.0 + min(0.4 * r, 96.0)

    p = round(pml_m / dx)
    ny = round((depth + pml_m) / dx)
    nx = round(width / dx) + 2 * p
    iy = torch.arange(ny, dtype=DTYPE).reshape(-1, 1) * dx
    xs = (torch.arange(nx, dtype=DTYPE) - p) * dx
    ys = torch.tensor([topo(float(x)) for x in xs], dtype=DTYPE)
    vac = iy < ys.reshape(1, -1) - 1e-9
    vp_t = torch.full((ny, nx), vp, dtype=DTYPE)
    vs_t = torch.full((ny, nx), vs, dtype=DTYPE)
    rho_t = torch.full((ny, nx), rho, dtype=DTYPE)
    vp_t[vac] = 0.0
    vs_t[vac] = 0.0
    rho_t[vac] = 0.0
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp_t, vs_t, rho_t)

    def col(x: float) -> int:
        return p + round(x / dx)

    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=DTYPE).reshape(1, 1, -1)
    src_loc = torch.tensor([[[round((topo(width / 2) + 16.0) / dx), col(width / 2)]]])
    rec_xs = [width * 0.25, width * 0.35, width * 0.65, width * 0.75]
    rec_loc = torch.tensor(
        [[[round((topo(rx) + 12.0) / dx), col(rx)] for rx in rec_xs]]
    )
    out = elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=amp,
        source_locations_y=src_loc,
        receiver_locations_y=rec_loc,
        receiver_locations_x=rec_loc,
        pml_width=[0, p, p, p],
        max_vel=vp,
        pml_freq=freq,
        accuracy=4,
        surface_method=method,
        backend="eager",
    )
    return torch.stack([out.receiver_amplitudes_y[0], out.receiver_amplitudes_x[0]])


def _misfit(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten()
    b = b.flatten()
    return float((a / a.norm() - b / b.norm()).norm())


@pytest.mark.parametrize(
    ("vp", "vs", "name"),
    [(4500.0, 2598.0, "sigma0.25"), (4500.0, 1357.0, "sigma0.45")],
)
def test_pm_topography_beats_ivm(vp, vs, name, record_property) -> None:
    """On staircase topography PM converges markedly faster than IVM.

    Reproduces the paper's central result (Table 4): on a rugged surface the PM
    solution on a coarse grid is much closer to the fine-grid (converged)
    solution than the IVM solution is, across the Poisson range. The comparison
    is made reference-robust by requiring PM to win whether the fine-grid
    reference is computed with PM *or* with IVM (both converge to the same
    physical solution, so neither choice can manufacture PM's advantage).
    """
    dx_coarse, dx_fine = 8.0, 8.0 / 3
    dt = 0.4 * dx_fine / vp
    nt = round(0.9 / dt)
    coarse_ivm = _run_trough("ivm", vp, vs, dx_coarse, dt, nt)
    coarse_pm = _run_trough("pm", vp, vs, dx_coarse, dt, nt)
    fine_ivm = _run_trough("ivm", vp, vs, dx_fine, dt, nt)
    fine_pm = _run_trough("pm", vp, vs, dx_fine, dt, nt)
    for label, reference in (("pm", fine_pm), ("ivm", fine_ivm)):
        err_ivm = _misfit(coarse_ivm, reference)
        err_pm = _misfit(coarse_pm, reference)
        record_property(f"{label}_reference_pm_misfit", err_pm)
        record_property(f"{label}_reference_ivm_misfit", err_ivm)
        assert err_pm < err_ivm, (
            f"{name}: PM misfit {err_pm:.4f} should be < IVM misfit "
            f"{err_ivm:.4f} (reference-robust)"
        )


# --------------------------------------------------------------------------
# Reciprocity, adjoint dot-product, gradcheck, negative control
# --------------------------------------------------------------------------


def _staircase_free_surface_model(ny: int, nx: int, vac: int):
    """A half-space with a one-step staircase surface (H, VL/VR and corner)."""
    vp_t = torch.zeros(ny, nx, dtype=DTYPE)
    vs_t = torch.zeros(ny, nx, dtype=DTYPE)
    rho_t = torch.zeros(ny, nx, dtype=DTYPE)
    vp_t[vac:] = 2200.0
    vs_t[vac:] = 1200.0
    rho_t[vac:] = 2000.0
    # A single downward step on the right half raises the surface by one cell.
    vp_t[: vac + 1, nx // 2 :] = 0.0
    vs_t[: vac + 1, nx // 2 :] = 0.0
    rho_t[: vac + 1, nx // 2 :] = 0.0
    return vpvsrho_to_lambmubuoyancy(vp_t, vs_t, rho_t)


def test_pm_reciprocity() -> None:
    """A PM topography configuration is source/receiver-swap reciprocal.

    For a like-component (vy force, vy receiver) pair, swapping the source and
    receiver positions leaves the recorded trace invariant. This is a property
    of the discrete elastic operator (the staggered-grid update is symmetric),
    so the parameter-modified surface must preserve it.
    """
    ny, nx, vac = 34, 44, 6
    lamb, mu, buoyancy = _staircase_free_surface_model(ny, nx, vac)
    dx = 5.0
    dt = 0.4 * dx / 2200.0
    nt = 240
    freq = 15.0
    amp = ricker(freq, nt, dt, 1.5 / freq, dtype=DTYPE).reshape(1, 1, -1)
    a = torch.tensor([[[vac + 6, nx // 4]]])
    b = torch.tensor([[[vac + 8, 3 * nx // 4]]])

    def shot(src, rec):
        out = elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_y=amp,
            source_locations_y=src,
            receiver_locations_y=rec,
            pml_width=[0, 15, 15, 15],
            max_vel=2200.0,
            pml_freq=freq,
            accuracy=4,
            surface_method="pm",
            backend="eager",
        )
        return out.receiver_amplitudes_y[0, 0]

    ab = shot(a, b)
    ba = shot(b, a)
    rel = (ab - ba).abs().max().item() / ab.abs().max().item()
    assert rel < 1e-6, f"PM reciprocity residual {rel:.2e}"


def _gradcheck_free_geometry(ny: int, nx: int, vac: int):
    """A fixed staircase vacuum mask for the gradcheck / adjoint models."""
    mask = torch.zeros(ny, nx, dtype=torch.bool)
    for j in range(nx):
        top = vac + (1 if j < nx // 2 else 0)
        mask[:top, j] = True
    return mask


def test_pm_adjoint_dot_product() -> None:
    """The eager PM backward pass is the exact transpose of its forward pass.

    The source amplitude enters the propagator linearly, so for the linear
    operator ``F`` (source amplitudes -> receiver amplitudes) the discrete
    adjoint identity ``<F s, r> == <s, F* r>`` must hold to machine precision,
    where ``F* r`` is obtained by reverse-mode autograd. A wrong PM adjoint would
    break it by orders of magnitude.
    """
    ny, nx, vac = 26, 28, 4
    mask = _gradcheck_free_geometry(ny, nx, vac)
    l0, m0, b0 = vpvsrho_to_lambmubuoyancy(
        torch.tensor(2200.0, dtype=DTYPE),
        torch.tensor(1200.0, dtype=DTYPE),
        torch.tensor(2000.0, dtype=DTYPE),
    )
    dx = 6.0
    dt = 0.4 * dx / 2200.0
    nt = 22
    freq = 15.0
    src = torch.tensor([[[vac + 9, nx // 2]]])
    rec = torch.tensor([[[vac + 3, 6], [vac + 3, 20]]])

    def forward(source_amplitudes):
        lamb = torch.full((ny, nx), l0.item(), dtype=DTYPE).masked_fill(mask, 0.0)
        mu = torch.full((ny, nx), m0.item(), dtype=DTYPE).masked_fill(mask, 0.0)
        buoyancy = torch.full((ny, nx), b0.item(), dtype=DTYPE).masked_fill(mask, 0.0)
        out = elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_y=source_amplitudes,
            source_locations_y=src,
            receiver_locations_y=rec,
            pml_width=[0, 8, 8, 8],
            max_vel=2200.0,
            pml_freq=freq,
            accuracy=2,
            surface_method="pm",
            backend="eager",
        )
        return out.receiver_amplitudes_y

    torch.manual_seed(0)
    s = torch.randn(1, 1, nt, dtype=DTYPE, requires_grad=True)
    fs = forward(s)
    r = torch.randn_like(fs)
    lhs = (fs * r).sum()
    lhs.backward()
    rhs = (s * s.grad).sum()
    assert abs(lhs.item() - rhs.item()) <= 1e-10 * abs(lhs.item())


def test_pm_gradcheck_through_model_prep() -> None:
    """Finite-difference gradcheck of PM w.r.t. vp/vs/rho through model prep.

    The surface-slab coefficients (``alpha``, the halved densities) are computed
    differentiably from the model, so the full path
    ``(vp, vs, rho) -> lamb/mu/buoyancy -> PM coefficients -> receivers`` must be
    differentiable end-to-end. The (fixed) vacuum geometry is held constant so
    the classifier's non-differentiable mask does not move under the
    finite-difference perturbation; only the solid-cell material values vary.
    """
    ny, nx, vac = 22, 24, 4
    mask = _gradcheck_free_geometry(ny, nx, vac)
    dx = 6.0
    dt = 0.4 * dx / 2200.0
    nt = 16
    freq = 15.0
    amp = ricker(freq, nt, dt, 0.08, dtype=DTYPE).reshape(1, 1, -1)
    src = torch.tensor([[[vac + 8, nx // 2]]])
    rec = torch.tensor([[[vac + 2, 6], [vac + 2, 18]]])
    solid_shape = (ny - vac, nx)
    vp_s = torch.full(solid_shape, 2200.0, dtype=DTYPE)
    vs_s = torch.full(solid_shape, 1200.0, dtype=DTYPE)
    rho_s = torch.full(solid_shape, 2000.0, dtype=DTYPE)

    def forward(vp_solid, vs_solid, rho_solid):
        vp_t = torch.zeros(ny, nx, dtype=DTYPE)
        vs_t = torch.zeros(ny, nx, dtype=DTYPE)
        rho_t = torch.zeros(ny, nx, dtype=DTYPE)
        vp_t[vac:] = vp_solid
        vs_t[vac:] = vs_solid
        rho_t[vac:] = rho_solid
        vp_t = vp_t.masked_fill(mask, 0.0)
        vs_t = vs_t.masked_fill(mask, 0.0)
        rho_t = rho_t.masked_fill(mask, 0.0)
        lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp_t, vs_t, rho_t)
        out = elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_y=amp,
            source_locations_y=src,
            receiver_locations_y=rec,
            pml_width=[0, 8, 8, 8],
            # Keep the discretisation fixed while covering both signs of every
            # finite-difference perturbation of the material parameters.
            max_vel=2300.0,
            pml_freq=freq,
            accuracy=2,
            surface_method="pm",
            backend="eager",
        )
        assert out.info.step_ratio == 1
        return out.receiver_amplitudes_y

    inputs = (
        vp_s.clone().requires_grad_(True),
        vs_s.clone().requires_grad_(True),
        rho_s.clone().requires_grad_(True),
    )
    assert torch.autograd.gradcheck(forward, inputs, eps=1e-4, atol=1e-4, rtol=1e-3)


def test_pm_negative_control_surface_modification_active() -> None:
    """The surface stiffness reduction demonstrably changes the solution.

    A guard against a silently inactive PM modification: disabling the
    ``alpha`` reduction on the horizontal-surface stiffness (restoring the
    interior P-modulus there) must change the recorded traces. If it did not,
    the surface treatment would be a no-op and every accuracy test above would be
    meaningless. This is the permanent form of the development-time negative
    control (break the coefficient, confirm the tests go red, revert).
    """
    import sys

    import deepwave.elastic  # noqa: F401

    elastic_mod = sys.modules["deepwave.elastic"]

    ny, nx, vac = 40, 50, 6
    vp, vs, rho = 2000.0, 1155.0, 2000.0
    vp_t = torch.zeros(ny, nx, dtype=DTYPE)
    vs_t = torch.zeros(ny, nx, dtype=DTYPE)
    rho_t = torch.zeros(ny, nx, dtype=DTYPE)
    vp_t[vac:] = vp
    vs_t[vac:] = vs
    rho_t[vac:] = rho
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp_t, vs_t, rho_t)
    dx = 5.0
    dt = 0.4 * dx / vp
    nt = 120
    freq = 15.0
    amp = ricker(freq, nt, dt, 0.08, dtype=DTYPE).reshape(1, 1, -1)
    src = torch.tensor([[[vac + 14, nx // 2]]])
    rec = torch.tensor([[[vac + 10, 15], [vac + 10, 40]]])
    kw = {
        "source_amplitudes_y": amp,
        "source_locations_y": src,
        "receiver_locations_y": rec,
        "pml_width": [0, 12, 12, 12],
        "max_vel": vp,
        "pml_freq": freq,
        "accuracy": 4,
        "surface_method": "pm",
        "backend": "eager",
    }
    good = elastic(lamb, mu, buoyancy, dx, dt, **kw).receiver_amplitudes_y[0].clone()

    original = elastic_mod._pm_normal_stiffness

    def broken(lamb_, mu_, masks):
        c = original(lamb_, mu_, masks)
        horizontal = masks[0][0]
        c[1][1] = torch.where(horizontal, lamb_ + 2.0 * mu_, c[1][1])
        return c

    elastic_mod._pm_normal_stiffness = broken
    try:
        bad = elastic(lamb, mu, buoyancy, dx, dt, **kw).receiver_amplitudes_y[0].clone()
    finally:
        elastic_mod._pm_normal_stiffness = original

    rel = (bad - good).abs().max().item() / good.abs().max().item()
    assert rel > 1e-3, (
        "Disabling the PM surface stiffness reduction did not change the "
        f"solution (rel {rel:.2e}); the surface treatment may be inactive."
    )


# --------------------------------------------------------------------------
# Compiled C/CUDA backend (DW_PM) gates: the compiled parameter-modified free
# surface must reproduce the eager reference (forward and gradients) to
# rounding, its hand-derived adjoint must be an exact transpose, and both must
# hold on CPU and the real GPU.
# --------------------------------------------------------------------------

_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="no CUDA device"
        ),
    ),
]


def _pm_compiled_config(
    ny: int, nx: int, vac: int, nt: int, accuracy: int, n_shots: int = 1
):
    """A staircase-trough half-space exercising H/VL/VR/corner surface cells.

    Returns ``(lamb, mu, buoyancy, kwargs)`` where ``kwargs`` is the shared
    ``elastic`` keyword set (source/receivers just below the topography, a PML
    on every side except the top free surface). The trough flanks are stepped
    so every surface-cell type is present. ``lamb``/``mu``/``buoyancy`` are
    always returned unbatched (shape ``(ny, nx)``, no leading shot dimension).
    With ``n_shots > 1`` the sources are spread to distinct locations with
    distinct wavelets (different peak time and amplitude scale per shot) so a
    multi-shot run cannot accidentally degenerate to a repeated single shot;
    the receivers are shared across shots.
    """
    vp, vs, rho = 2200.0, 1200.0, 2000.0
    vp_t = torch.full((ny, nx), vp, dtype=DTYPE)
    vs_t = torch.full((ny, nx), vs, dtype=DTYPE)
    rho_t = torch.full((ny, nx), rho, dtype=DTYPE)
    # A symmetric stepped trough: the surface rises towards the edges in
    # one-cell steps, producing horizontal, both vertical, and corner cells.
    for j in range(nx):
        d = min(j, nx - 1 - j)
        top = vac + min(d // 3, vac - 1)
        vp_t[:top, j] = 0.0
        vs_t[:top, j] = 0.0
        rho_t[:top, j] = 0.0
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp_t, vs_t, rho_t)
    dx = 5.0
    dt = 0.4 * dx / vp
    freq = 15.0
    if n_shots == 1:
        amp = ricker(freq, nt, dt, 1.5 / freq, dtype=DTYPE).reshape(1, 1, -1)
        src = torch.tensor([[[ny - 8, nx // 2]]])
    else:
        spread = max(1, nx // 6)
        offsets = torch.linspace(-spread, spread, n_shots).round().long()
        src = torch.tensor([[[ny - 8, int(nx // 2 + off)]] for off in offsets])
        amp = torch.stack(
            [
                (1.0 + 0.25 * s)
                * ricker(freq, nt, dt, (1.3 + 0.2 * s) / freq, dtype=DTYPE)
                for s in range(n_shots)
            ]
        ).reshape(n_shots, 1, nt)
    rec = torch.tensor([[[vac + 2, nx // 4], [vac + 2, 3 * nx // 4]]]).repeat(
        n_shots, 1, 1
    )
    kwargs = {
        "grid_spacing": dx,
        "dt": dt,
        "source_amplitudes_y": amp,
        "source_locations_y": src,
        "receiver_locations_y": rec,
        "receiver_locations_x": rec,
        "pml_width": [0, 12, 12, 12],
        "max_vel": vp,
        "pml_freq": freq,
        "accuracy": accuracy,
        "surface_method": "pm",
    }
    return lamb, mu, buoyancy, kwargs


@pytest.mark.parametrize("device", _DEVICES)
def test_pm_compiled_forward_matches_eager(device) -> None:
    """The compiled PM forward equals the eager PM forward to rounding.

    The deferred ``C == eager`` gate: on a genuine staircase topography (all
    surface-cell types present) the default compiled C/CUDA backend's traces
    must match the eager backend's to floating-point rounding, confirming the
    ``DW_PM`` kernel reproduces the reference update exactly.
    """
    lamb, mu, buoyancy, kwargs = _pm_compiled_config(30, 40, 6, nt=400, accuracy=4)
    lamb, mu, buoyancy = (t.to(device) for t in (lamb, mu, buoyancy))
    kwargs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in kwargs.items()
    }
    eager = elastic(lamb, mu, buoyancy, backend="eager", **kwargs)
    compiled = elastic(lamb, mu, buoyancy, backend="compiled", **kwargs)
    for name in ("receiver_amplitudes_y", "receiver_amplitudes_x"):
        e = getattr(eager, name)
        c = getattr(compiled, name)
        scale = e.abs().max().item()
        rel = (e - c).abs().max().item() / scale
        assert rel < 1e-6, f"{name} on {device}: compiled-vs-eager rel {rel:.2e}"


@pytest.mark.parametrize("device", _DEVICES)
def test_pm_compiled_adjoint_dot_product(device) -> None:
    """The compiled PM backward is the exact transpose of its forward.

    ``<F s, r> == <s, F* r>`` to machine precision for the source-amplitude ->
    receiver linear operator ``F``, with ``F* r`` obtained by reverse-mode
    autograd through the compiled ``DW_PM`` adjoint (CPU and real GPU).
    """
    ny, nx, vac = 26, 30, 5
    lamb, mu, buoyancy, kwargs = _pm_compiled_config(ny, nx, vac, nt=60, accuracy=2)
    lamb, mu, buoyancy = (t.to(device) for t in (lamb, mu, buoyancy))
    kwargs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in kwargs.items()
    }
    nt = kwargs["source_amplitudes_y"].shape[-1]
    kwargs = {k: v for k, v in kwargs.items() if k != "source_amplitudes_y"}

    def forward(source_amplitudes):
        out = elastic(
            lamb,
            mu,
            buoyancy,
            source_amplitudes_y=source_amplitudes,
            backend="compiled",
            **kwargs,
        )
        return out.receiver_amplitudes_y

    torch.manual_seed(0)
    s = torch.randn(1, 1, nt, dtype=DTYPE, device=device, requires_grad=True)
    fs = forward(s)
    r = torch.randn_like(fs)
    lhs = (fs * r).sum()
    lhs.backward()
    rhs = (s * s.grad).sum()
    assert abs(lhs.item() - rhs.item()) <= 1e-10 * abs(lhs.item()), (
        f"compiled PM adjoint dot-product mismatch on {device}"
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_pm_compiled_gradients_match_eager(device) -> None:
    """Compiled PM model gradients match the eager reference to a tight tol.

    The cross-backend correctness gate for the hand-derived adjoint: the
    gradients of a scalar misfit w.r.t. lamb, mu and buoyancy computed by the
    compiled ``DW_PM`` backend must match the eager (native-autograd) backend
    on a genuinely-reflecting staircase-topography configuration. A wrong
    ``C[i][j]`` transpose or buoyancy fold would break this well beyond the
    tolerance.
    """
    lamb0, mu0, buoyancy0, kwargs = _pm_compiled_config(30, 40, 6, nt=400, accuracy=4)
    lamb0, mu0, buoyancy0 = (t.to(device) for t in (lamb0, mu0, buoyancy0))
    kwargs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in kwargs.items()
    }

    def grads(backend):
        lamb = lamb0.clone().requires_grad_(True)
        mu = mu0.clone().requires_grad_(True)
        buoyancy = buoyancy0.clone().requires_grad_(True)
        out = elastic(lamb, mu, buoyancy, backend=backend, **kwargs)
        loss = (
            out.receiver_amplitudes_y.pow(2).sum()
            + out.receiver_amplitudes_x.pow(2).sum()
        )
        loss.backward()
        return lamb.grad, mu.grad, buoyancy.grad

    ge = grads("eager")
    gc = grads("compiled")
    for name, e, c in zip(("lamb", "mu", "buoyancy"), ge, gc):
        scale = e.abs().max().item()
        rel = (e - c).abs().max().item() / scale
        assert rel < 1e-6, f"grad_{name} on {device}: compiled-vs-eager rel {rel:.2e}"


def test_pm_compiled_gradients_match_eager_at_step_ratio_gt_1() -> None:
    """The PM ``C[i][j]`` gradient terms keep their ``step_ratio`` factor.

    Mutation testing's one surviving compiled-kernel mutant
    (``ek_grad_c00_step_ratio``) drops
    the ``step_ratio`` factor from the first of the four ``DW_PM`` stiffness
    gradients accumulated together in ``elastic_kernels.h``::

        grad_c00[i] += sigmayy[i] * dvydy_store[i] * (DW_DTYPE)step_ratio;
        ->
        grad_c00[i] += sigmayy[i] * dvydy_store[i];

    It survived the whole suite because every PM gradient test runs at
    ``step_ratio == 1``, where the factor is 1 and the mutation is a no-op.
    ``test_pm_compiled_gradients_match_eager`` above is exactly this check at
    ``step_ratio == 1``; this is its ``step_ratio > 1`` companion.

    At ``step_ratio > 1`` the compiled backend's model gradient is a strided
    rectangle-rule quadrature of the exact adjoint integral while the eager
    backend is the exact discrete adjoint, so the two differ by a documented
    O(dt) error (``docs/what_deepwave_calculates.rst``) rather than agreeing
    to rounding. Measured on this configuration at ``step_ratio == 2`` that
    error is about 9.9e-3 (``lamb``), 3.9e-3 (``mu``), 1.0e-2 (``buoyancy``).
    With
    the mutation it is **1.7 (172%)** for ``lamb`` and **0.40** for ``mu`` --
    two orders of magnitude larger, because a missing ``step_ratio`` is a
    structural error, not a quadrature one. The bound below sits between them.

    CPU only: the mutation is in the shared kernel header, so the CPU build
    exercises it, and this keeps the test cheap.
    """
    lamb0, mu0, buoyancy0, kwargs = _pm_compiled_config(26, 30, 5, nt=120, accuracy=2)
    # Double dt to push the CFL step_ratio to 2 (the helper picks a dt at 0.4
    # of the limit, i.e. step_ratio 1).
    kwargs = dict(kwargs)
    kwargs["dt"] = kwargs["dt"] * 2

    def grads(backend):
        lamb = lamb0.clone().requires_grad_(True)
        mu = mu0.clone().requires_grad_(True)
        buoyancy = buoyancy0.clone().requires_grad_(True)
        out = elastic(lamb, mu, buoyancy, backend=backend, **kwargs)
        (
            out.receiver_amplitudes_y.pow(2).sum()
            + out.receiver_amplitudes_x.pow(2).sum()
        ).backward()
        return (lamb.grad, mu.grad, buoyancy.grad), int(out.info.step_ratio)

    eager_grads, step_ratio = grads("eager")
    compiled_grads, _ = grads("compiled")
    assert step_ratio == 2, (
        f"expected step_ratio 2, got {step_ratio}; this test is pointless at 1"
    )
    for name, e, c in zip(("lamb", "mu", "buoyancy"), eager_grads, compiled_grads):
        scale = e.abs().max().item()
        assert scale > 0, f"grad_{name} is zero; this test would be vacuous"
        rel = (e - c).abs().max().item() / scale
        assert rel < 0.1, (
            f"grad_{name} at step_ratio=2: compiled-vs-eager rel {rel:.2e}, "
            "far beyond the documented O(dt) strided-quadrature error (about "
            "1.0e-2 here). Check that every DW_PM gradient term in "
            "elastic_kernels.h still carries its step_ratio factor."
        )


@pytest.mark.parametrize("device", _DEVICES)
def test_pm_compiled_multishot_gradients_match_eager(device) -> None:
    """Compiled PM model gradients match eager for an unbatched, multi-shot run.

    ``elastic.cu``'s ``DW_PM`` backward accumulates each shot's ``grad_c00``/
    ``grad_c01``/``grad_c10``/``grad_c11`` into a per-shot temporary and then
    reduces it into the final gradient with ``combine_grad``, but only when the
    model is not itself shot-batched and ``n_shots > 1``; a single-shot run
    aliases the per-shot temporary to the final gradient and never touches the
    reduction. This test uses an unbatched model (``lamb``/``mu``/``buoyancy``
    with no leading shot dimension) and two shots with distinct source
    locations and wavelets on the staircase-trough topography used by the
    other compiled PM gates, which is exactly the configuration that exercises
    ``combine_grad`` for the ``c00``/``c01``/``c10``/``c11`` arrays.
    """
    n_shots = 2
    lamb0, mu0, buoyancy0, kwargs = _pm_compiled_config(
        30, 40, 6, nt=400, accuracy=4, n_shots=n_shots
    )
    assert lamb0.dim() == 2, "model must be unbatched to trigger combine_grad"
    lamb0, mu0, buoyancy0 = (t.to(device) for t in (lamb0, mu0, buoyancy0))
    kwargs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in kwargs.items()
    }
    assert kwargs["source_amplitudes_y"].shape[0] == n_shots

    def grads(backend):
        lamb = lamb0.clone().requires_grad_(True)
        mu = mu0.clone().requires_grad_(True)
        buoyancy = buoyancy0.clone().requires_grad_(True)
        out = elastic(lamb, mu, buoyancy, backend=backend, **kwargs)
        loss = (
            out.receiver_amplitudes_y.pow(2).sum()
            + out.receiver_amplitudes_x.pow(2).sum()
        )
        loss.backward()
        return lamb.grad, mu.grad, buoyancy.grad

    ge = grads("eager")
    gc = grads("compiled")
    for name, e, c in zip(("lamb", "mu", "buoyancy"), ge, gc):
        scale = e.abs().max().item()
        rel = (e - c).abs().max().item() / scale
        assert rel < 1e-6, (
            f"grad_{name} on {device} (n_shots={n_shots}, unbatched model): "
            f"compiled-vs-eager rel {rel:.2e}"
        )


@pytest.mark.parametrize("device", _DEVICES)
def test_pm_compiled_gradcheck(device) -> None:
    """Finite-difference gradcheck of the compiled PM path w.r.t. vp/vs/rho.

    Exercises the full differentiable chain
    ``(vp, vs, rho) -> lamb/mu/buoyancy -> DW_PM stiffness arrays -> receivers``
    through the compiled backend. The vacuum geometry is fixed so the
    classifier's mask does not move under the finite-difference perturbation.
    """
    ny, nx, vac = 22, 24, 4
    mask = _gradcheck_free_geometry(ny, nx, vac).to(device)
    dx = 6.0
    dt = 0.4 * dx / 2200.0
    nt = 16
    freq = 15.0
    amp = ricker(freq, nt, dt, 0.08, dtype=DTYPE).reshape(1, 1, -1).to(device)
    src = torch.tensor([[[vac + 8, nx // 2]]])
    rec = torch.tensor([[[vac + 2, 6], [vac + 2, 18]]])
    solid_shape = (ny - vac, nx)
    vp_s = torch.full(solid_shape, 2200.0, dtype=DTYPE, device=device)
    vs_s = torch.full(solid_shape, 1200.0, dtype=DTYPE, device=device)
    rho_s = torch.full(solid_shape, 2000.0, dtype=DTYPE, device=device)

    def forward(vp_solid, vs_solid, rho_solid):
        vp_t = torch.zeros(ny, nx, dtype=DTYPE, device=device)
        vs_t = torch.zeros(ny, nx, dtype=DTYPE, device=device)
        rho_t = torch.zeros(ny, nx, dtype=DTYPE, device=device)
        vp_t[vac:] = vp_solid
        vs_t[vac:] = vs_solid
        rho_t[vac:] = rho_solid
        vp_t = vp_t.masked_fill(mask, 0.0)
        vs_t = vs_t.masked_fill(mask, 0.0)
        rho_t = rho_t.masked_fill(mask, 0.0)
        lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp_t, vs_t, rho_t)
        out = elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_y=amp,
            source_locations_y=src,
            receiver_locations_y=rec,
            pml_width=[0, 8, 8, 8],
            # Keep CFL/PML setup fixed for both finite-difference sides.
            max_vel=2300.0,
            pml_freq=freq,
            accuracy=2,
            surface_method="pm",
            backend="compiled",
        )
        assert out.info.step_ratio == 1
        return out.receiver_amplitudes_y

    inputs = (
        vp_s.clone().requires_grad_(True),
        vs_s.clone().requires_grad_(True),
        rho_s.clone().requires_grad_(True),
    )
    assert torch.autograd.gradcheck(forward, inputs, eps=1e-4, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("device", _DEVICES)
def test_pm_compiled_differs_from_ivm(device) -> None:
    """The compiled PM surface treatment is demonstrably active.

    A permanent guard against a silently-inactive compiled path: on the same
    topography the compiled ``surface_method="pm"`` traces must differ from the
    compiled default ``surface_method="ivm"`` traces (the parameter
    modification actually changes the recorded wavefield), while still matching
    the eager PM reference (checked separately). If PM collapsed to IVM the
    accuracy gates would be meaningless.
    """
    lamb, mu, buoyancy, kwargs = _pm_compiled_config(30, 40, 6, nt=400, accuracy=4)
    lamb, mu, buoyancy = (t.to(device) for t in (lamb, mu, buoyancy))
    kwargs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in kwargs.items()
    }
    kwargs_ivm = {**kwargs, "surface_method": "ivm"}
    pm = elastic(lamb, mu, buoyancy, backend="compiled", **kwargs)
    ivm = elastic(lamb, mu, buoyancy, backend="compiled", **kwargs_ivm)
    a = pm.receiver_amplitudes_y
    b = ivm.receiver_amplitudes_y
    rel = (a - b).abs().max().item() / a.abs().max().item()
    assert rel > 1e-3, (
        f"compiled PM did not differ from IVM on {device} (rel {rel:.2e}); "
        "the surface treatment may be inactive."
    )


# --------------------------------------------------------------------------
# 3-D eager PM: Cao-Chen Table 2 / eq 25 coefficient checks, exact frozen-
# surface invariants, lateral-axis symmetry, the interior-unchanged identity,
# and the adjoint/gradient identities. (The compiled backend is 2-D only; its
# 3-D port is scheduled after the Q1.5(c)/(d) kernel-variant restructure.)
# --------------------------------------------------------------------------


def _mesa_model_3d(
    nz: int = 14, ny: int = 12, nx: int = 12, vac: int = 4, block: int = 2
):
    """A half-space with a square mesa on the free surface.

    The flat surface is at depth index ``vac``; a raised block of height
    ``block`` covers the footprint ``[4, 8) x [4, 8)``, producing every
    3-D surface-cell category: flat and mesa-top ``H`` faces, walls facing
    both lateral axes, the mesa-top rim (horizontal edges) and the vertical
    edges (both eq 25 transition cells). The footprint is symmetric under
    swapping the two lateral axes.
    """
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    solid[vac:] = True
    solid[vac - block : vac, 4:8, 4:8] = True
    vp = torch.where(solid, 2000.0, 0.0).to(DTYPE)
    vs = torch.where(solid, 1150.0, 0.0).to(DTYPE)
    rho = torch.where(solid, 2000.0, 0.0).to(DTYPE)
    return vpvsrho_to_lambmubuoyancy(vp, vs, rho)


def test_pm_3d_coefficients_match_table2() -> None:
    """The 3-D classifier and coefficients equal the Cao-Chen Table 2 values.

    Spot-checks every surface-cell category of the mesa model against the
    symbolically verified table (see
    ``docs/dev/boundary_conditions/verify_topo_tables.py``,
    checks [D] and [F]): a face cell with surface normal along axis ``k``
    freezes the ``sigma_kk`` stiffness row and keeps ``alpha`` on the
    surviving diagonals with ``beta`` cross-coupling, doubles the two in-plane
    staggered buoyancies, and halves the in-plane shear modulus; a transition
    (edge/corner) cell freezes all rows, doubles all buoyancies and halves all
    shear moduli (eq 25); interior cells keep the stock isotropic
    coefficients exactly.
    """
    vac = 4
    lamb, mu, buoyancy = _mesa_model_3d(vac=vac)
    lamb = lamb.unsqueeze(0)
    mu = mu.unsqueeze(0)
    prepared = prepare_parameters(mu, buoyancy.unsqueeze(0))
    mu_zy, mu_zx, mu_yx = prepared[0], prepared[1], prepared[2]
    b_z, b_y, b_x = prepared[3], prepared[4], prepared[5]

    face, transition = _pm_surface_masks(mu, 3)
    # The masks are pairwise disjoint by construction.
    assert not bool((transition & (face[0] | face[1] | face[2])).any())
    assert not bool((face[0] & face[1]).any())
    assert not bool((face[0] & face[2]).any())
    assert not bool((face[1] & face[2]).any())
    # One representative cell of each category (batch, z, y, x).
    h = (0, vac, 1, 1)  # flat horizontal surface
    h_top = (0, vac - 2, 5, 5)  # mesa top, also H
    w_y = (0, vac - 1, 4, 5)  # mesa wall facing -y
    w_x = (0, vac - 1, 5, 4)  # mesa wall facing -x
    rim = (0, vac - 2, 4, 5)  # mesa-top rim: horizontal edge
    v_edge = (0, vac - 1, 4, 4)  # vertical edge of the mesa
    interior = (0, vac + 4, 6, 6)
    for cell in (h, h_top):
        assert bool(face[0][cell])
        assert not bool(transition[cell])
    assert bool(face[1][w_y])
    assert not bool(transition[w_y])
    assert bool(face[2][w_x])
    assert not bool(transition[w_x])
    for cell in (rim, v_edge):
        assert bool(transition[cell])
    assert not bool((face[0] | face[1] | face[2] | transition)[interior])

    c = _pm_normal_stiffness(lamb, mu, (face, transition))
    lp2m = lamb + 2.0 * mu
    alpha = 2.0 * mu * (lamb + mu) / lp2m.clamp(min=1.0)
    beta = mu * lamb / lp2m.clamp(min=1.0)

    def entry(i, j, cell):
        return c[i][j][cell].item()

    # H cell: sigma_zz row frozen; tau_yy = alpha eyy + beta exx etc.
    for cell in (h, h_top):
        for j in range(3):
            assert entry(0, j, cell) == 0.0
        assert entry(1, 1, cell) == pytest.approx(alpha[cell].item(), rel=1e-14)
        assert entry(1, 2, cell) == pytest.approx(beta[cell].item(), rel=1e-14)
        assert entry(1, 0, cell) == 0.0
        assert entry(2, 2, cell) == pytest.approx(alpha[cell].item(), rel=1e-14)
        assert entry(2, 1, cell) == pytest.approx(beta[cell].item(), rel=1e-14)
        assert entry(2, 0, cell) == 0.0
    # Wall facing -y: sigma_yy row frozen; zz and xx rows keep alpha/beta.
    for j in range(3):
        assert entry(1, j, w_y) == 0.0
    assert entry(0, 0, w_y) == pytest.approx(alpha[w_y].item(), rel=1e-14)
    assert entry(0, 2, w_y) == pytest.approx(beta[w_y].item(), rel=1e-14)
    assert entry(0, 1, w_y) == 0.0
    assert entry(2, 2, w_y) == pytest.approx(alpha[w_y].item(), rel=1e-14)
    assert entry(2, 0, w_y) == pytest.approx(beta[w_y].item(), rel=1e-14)
    assert entry(2, 1, w_y) == 0.0
    # Wall facing -x: sigma_xx row frozen (the axis-2 mirror image).
    for j in range(3):
        assert entry(2, j, w_x) == 0.0
    assert entry(0, 0, w_x) == pytest.approx(alpha[w_x].item(), rel=1e-14)
    assert entry(0, 1, w_x) == pytest.approx(beta[w_x].item(), rel=1e-14)
    assert entry(0, 2, w_x) == 0.0
    # Transition cells: every row frozen (eq 25; exact at 3-face corners,
    # the paper's approximation along edges).
    for cell in (rim, v_edge):
        for i in range(3):
            for j in range(3):
                assert entry(i, j, cell) == 0.0
    # Interior: the stock isotropic stiffness exactly.
    for i in range(3):
        for j in range(3):
            expected = lp2m if i == j else lamb
            assert entry(i, j, interior) == expected[interior].item()

    # Buoyancy doubling: the two in-plane components at a face cell, all
    # three at a transition cell, none in the interior.
    bz_m, by_m, bx_m = _pm_modify_buoyancy(
        [b_z.clone(), b_y.clone(), b_x.clone()], (face, transition)
    )
    assert b_z[h].item() != 0.0
    assert b_y[h].item() != 0.0
    assert b_x[h].item() != 0.0
    assert bz_m[h] == b_z[h]
    assert by_m[h] == 2 * b_y[h]
    assert bx_m[h] == 2 * b_x[h]
    assert b_z[w_y].item() != 0.0
    assert b_x[w_y].item() != 0.0
    assert bz_m[w_y] == 2 * b_z[w_y]
    assert by_m[w_y] == b_y[w_y]
    assert bx_m[w_y] == 2 * b_x[w_y]
    assert bz_m[v_edge] == 2 * b_z[v_edge]
    assert by_m[v_edge] == 2 * b_y[v_edge]
    assert bx_m[v_edge] == 2 * b_x[v_edge]
    for b_mod, b_base in ((bz_m, b_z), (by_m, b_y), (bx_m, b_x)):
        assert b_mod[interior] == b_base[interior]

    # Shear halving: the in-surface-plane pair at a face cell, all pairs at a
    # transition cell, none in the interior.
    empty = torch.empty(0)
    shear = [
        [empty, mu_zy, mu_zx],
        [mu_zy, empty, mu_yx],
        [mu_zx, mu_yx, empty],
    ]
    shear_m = _pm_modify_shear(shear, (face, transition))
    assert mu_yx[h].item() != 0.0
    assert mu_zy[h].item() != 0.0
    assert shear_m[1][2][h] == 0.5 * mu_yx[h]
    assert shear_m[0][1][h] == mu_zy[h]
    assert shear_m[0][2][h] == mu_zx[h]
    assert mu_zx[w_y].item() != 0.0
    assert shear_m[0][2][w_y] == 0.5 * mu_zx[w_y]
    assert shear_m[0][1][w_y] == mu_zy[w_y]
    assert shear_m[1][2][w_y] == mu_yx[w_y]
    assert mu_zy[rim].item() != 0.0
    assert shear_m[0][1][rim] == 0.5 * mu_zy[rim]
    assert shear_m[0][2][rim] == 0.5 * mu_zx[rim]
    assert shear_m[1][2][rim] == 0.5 * mu_yx[rim]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert shear_m[i][j][interior] == shear[i][j][interior]
        # The symmetric counterpart is the same tensor.
        assert shear_m[j][i] is shear_m[i][j]


def test_pm_3d_frozen_surface_stress_stays_zero() -> None:
    """The traction-frozen normal-stress components stay exactly zero.

    The PM stiffness places exact zeros in the frozen rows, so after any
    number of eager 3-D steps the surface-normal stress component must be
    identically zero at every face cell of its axis (and all components at
    transition cells) -- not small, but bitwise zero -- while the wave is
    demonstrably present elsewhere. This is the discrete traction-free
    condition holding by construction.
    """
    vac = 4
    lamb, mu, buoyancy = _mesa_model_3d(vac=vac)
    dx = 5.0
    dt = 0.4 * dx / 2000.0
    nt = 30
    amp = ricker(15.0, nt, dt, 0.05, dtype=DTYPE).reshape(1, 1, -1)
    src = torch.tensor([[[vac + 5, 6, 6]]])
    out = elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_z=amp,
        source_locations_z=src,
        pml_width=0,
        max_vel=2000.0,
        pml_freq=15.0,
        accuracy=2,
        surface_method="pm",
        backend="eager",
    )
    face, transition = _pm_surface_masks(mu.unsqueeze(0), 3)
    normal = [out.sigmazz_0, out.sigmayy_0, out.sigmaxx_0]
    for field in [*normal, out.vz_0, out.vy_0, out.vx_0]:
        assert field.isfinite().all()
        assert field.shape[-3:] == mu.shape
    for k in range(3):
        assert bool((normal[k][face[k]] == 0).all())
        assert bool((normal[k][transition] == 0).all())
    # Sanity: the wave reached the surface region and is nonzero elsewhere.
    assert out.sigmazz_0.abs().max().item() > 0.0
    assert out.vz_0[face[0]].abs().max().item() > 0.0


def test_pm_3d_lateral_axis_symmetry() -> None:
    """A lateral-axis-symmetric problem produces a symmetric wavefield.

    The mesa model, source and boundaries are invariant under swapping the two
    lateral axes, so the propagated wavefields must be too: ``vz`` and
    ``sigmazz`` symmetric under transposing the lateral dims, and ``vy``/``vx``
    (and ``sigmayy``/``sigmaxx``) mapping onto each other. An asymmetry between
    the axis-1 and axis-2 surface classification or coefficients would break
    this immediately.
    """
    vac = 4
    lamb, mu, buoyancy = _mesa_model_3d(vac=vac)
    dx = 5.0
    dt = 0.4 * dx / 2000.0
    nt = 24
    amp = ricker(15.0, nt, dt, 0.05, dtype=DTYPE).reshape(1, 1, -1)
    src = torch.tensor([[[vac + 4, 6, 6]]])
    out = elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_z=amp,
        source_locations_z=src,
        pml_width=0,
        max_vel=2000.0,
        pml_freq=15.0,
        accuracy=2,
        surface_method="pm",
        backend="eager",
    )

    def sym(a, b):
        scale = max(a.abs().max().item(), 1e-30)
        return (a - b.transpose(-2, -1)).abs().max().item() / scale

    assert sym(out.vz_0, out.vz_0) < 1e-12
    assert sym(out.sigmazz_0, out.sigmazz_0) < 1e-12
    assert sym(out.vy_0, out.vx_0) < 1e-12
    assert sym(out.sigmayy_0, out.sigmaxx_0) < 1e-12
    assert out.vz_0.abs().max().item() > 0.0


def _fake_pml_3d(nz: int, ny: int, nx: int):
    ones = torch.ones(1, nz, ny, nx, dtype=DTYPE)
    zeros = torch.zeros(1, nz, ny, nx, dtype=DTYPE)
    return [ones, zeros, ones, zeros] * 3  # a, b, ah, bh per dim


def test_pm_3d_interior_unchanged_kernels() -> None:
    """With no vacuum, the 3-D PM update kernels reproduce the stock kernels.

    The 3-D analogue of the 2-D interior identity: with nothing classified,
    the stiffness, buoyancy and shear modifications are all inactive and the
    PM kernels must match the stock kernels to rounding.
    """
    torch.manual_seed(0)
    nz, ny, nx = 10, 12, 14
    vp = torch.full((nz, ny, nx), 2000.0, dtype=DTYPE)
    vs = torch.full((nz, ny, nx), 1150.0, dtype=DTYPE)
    rho = torch.full((nz, ny, nx), 2000.0, dtype=DTYPE)
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    models = [lamb.unsqueeze(0), mu.unsqueeze(0)]
    models = models + prepare_parameters(mu.unsqueeze(0), buoyancy.unsqueeze(0))
    grid_spacing = [5.0, 5.0, 5.0]
    dt = 0.001

    velocities = [torch.randn(1, nz, ny, nx, dtype=DTYPE) for _ in range(3)]
    normal = [torch.randn(1, nz, ny, nx, dtype=DTYPE) for _ in range(3)]
    empty = torch.empty(0)
    s_zy = torch.randn(1, nz, ny, nx, dtype=DTYPE)
    s_zx = torch.randn(1, nz, ny, nx, dtype=DTYPE)
    s_yx = torch.randn(1, nz, ny, nx, dtype=DTYPE)
    shear = [[empty, s_zy, s_zx], [s_zy, empty, s_yx], [s_zx, s_yx, empty]]

    def zmem_normal():
        return [torch.zeros(1, nz, ny, nx, dtype=DTYPE) for _ in range(3)]

    def zmem_shear():
        def z():
            return torch.zeros(1, nz, ny, nx, dtype=DTYPE)

        return [
            [empty, z(), z()],
            [z(), empty, z()],
            [z(), z(), empty],
        ]

    args = (
        _fake_pml_3d(nz, ny, nx),
        grid_spacing,
        dt,
        4,
    )
    stock_s = update_stresses(
        models,
        velocities,
        [x.clone() for x in normal],
        copy.deepcopy(shear),
        zmem_normal(),
        zmem_shear(),
        *args,
    )
    pm_s = update_stresses_pm(
        models,
        velocities,
        [x.clone() for x in normal],
        copy.deepcopy(shear),
        zmem_normal(),
        zmem_shear(),
        *args,
    )
    for i in range(3):
        scale = stock_s[0][i].abs().max().item()
        assert (stock_s[0][i] - pm_s[0][i]).abs().max().item() / scale < 1e-12

    stock_v = update_velocities(
        models,
        [x.clone() for x in normal],
        copy.deepcopy(shear),
        [x.clone() for x in velocities],
        zmem_normal(),
        zmem_shear(),
        *args,
    )
    pm_v = update_velocities_pm(
        models,
        [x.clone() for x in normal],
        copy.deepcopy(shear),
        [x.clone() for x in velocities],
        zmem_normal(),
        zmem_shear(),
        *args,
    )
    for i in range(3):
        scale = stock_v[0][i].abs().max().item()
        assert (stock_v[0][i] - pm_v[0][i]).abs().max().item() / scale < 1e-12


def _step_free_geometry_3d(nz: int, ny: int, nx: int, vac: int):
    """A fixed 3-D staircase vacuum mask (a step along the y axis)."""
    mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
    for j in range(ny):
        top = vac + (1 if j < ny // 2 else 0)
        mask[:top, j, :] = True
    return mask


def test_pm_3d_adjoint_dot_product() -> None:
    """The 3-D eager PM backward pass is the exact transpose of the forward.

    The 3-D analogue of the 2-D identity: for the linear operator ``F``
    (source amplitudes -> receiver amplitudes) over a stepped topography that
    exercises the H, wall and transition coefficient paths,
    ``<F s, r> == <s, F* r>`` must hold to machine precision.
    """
    nz, ny, nx, vac = 16, 12, 12, 4
    mask = _step_free_geometry_3d(nz, ny, nx, vac)
    l0, m0, b0 = vpvsrho_to_lambmubuoyancy(
        torch.tensor(2200.0, dtype=DTYPE),
        torch.tensor(1200.0, dtype=DTYPE),
        torch.tensor(2000.0, dtype=DTYPE),
    )
    dx = 6.0
    dt = 0.4 * dx / 2200.0
    nt = 16
    src = torch.tensor([[[vac + 6, 6, 6]]])
    rec = torch.tensor([[[vac + 2, 3, 3], [vac + 2, 8, 8]]])

    def forward(source_amplitudes):
        lamb = torch.full((nz, ny, nx), l0.item(), dtype=DTYPE).masked_fill(mask, 0.0)
        mu = torch.full((nz, ny, nx), m0.item(), dtype=DTYPE).masked_fill(mask, 0.0)
        buoyancy = torch.full((nz, ny, nx), b0.item(), dtype=DTYPE).masked_fill(
            mask, 0.0
        )
        out = elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_z=source_amplitudes,
            source_locations_z=src,
            receiver_locations_z=rec,
            pml_width=[0, 4, 4, 4, 4, 4],
            max_vel=2200.0,
            pml_freq=15.0,
            accuracy=2,
            surface_method="pm",
            backend="eager",
        )
        return out.receiver_amplitudes_z

    torch.manual_seed(0)
    s = torch.randn(1, 1, nt, dtype=DTYPE, requires_grad=True)
    fs = forward(s)
    r = torch.randn_like(fs)
    lhs = (fs * r).sum()
    lhs.backward()
    rhs = (s * s.grad).sum()
    assert abs(lhs.item() - rhs.item()) <= 1e-10 * abs(lhs.item())


def test_pm_3d_gradcheck_scalar_materials() -> None:
    """Finite-difference gradcheck of 3-D PM w.r.t. the material values.

    The 3-D coefficient construction (``alpha``/``beta`` stiffness, buoyancy
    doubling and shear halving) must be differentiable end-to-end through
    ``(vp, vs, rho) -> lamb/mu/buoyancy -> PM coefficients -> receivers``.
    Scalar material values (broadcast over the fixed solid geometry) keep the
    finite-difference cost small while still exercising every coefficient
    path; the per-cell gradient placement is shared with (and covered by) the
    2-D per-cell gradcheck.
    """
    nz, ny, nx, vac = 12, 10, 10, 3
    mask = _step_free_geometry_3d(nz, ny, nx, vac)
    solid = ~mask
    dx = 6.0
    dt = 0.4 * dx / 2400.0
    nt = 12
    amp = ricker(15.0, nt, dt, 0.06, dtype=DTYPE).reshape(1, 1, -1)
    src = torch.tensor([[[vac + 5, 5, 5]]])
    rec = torch.tensor([[[vac + 2, 3, 3], [vac + 2, 7, 7]]])

    def forward(vp0, vs0, rho0):
        vp_t = solid * vp0
        vs_t = solid * vs0
        rho_t = solid * rho0
        lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp_t, vs_t, rho_t)
        out = elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_z=amp,
            source_locations_z=src,
            receiver_locations_z=rec,
            pml_width=[0, 4, 4, 4, 4, 4],
            max_vel=2400.0,
            pml_freq=15.0,
            accuracy=2,
            surface_method="pm",
            backend="eager",
        )
        return out.receiver_amplitudes_z

    inputs = (
        torch.tensor(2200.0, dtype=DTYPE, requires_grad=True),
        torch.tensor(1200.0, dtype=DTYPE, requires_grad=True),
        torch.tensor(2000.0, dtype=DTYPE, requires_grad=True),
    )
    assert torch.autograd.gradcheck(forward, inputs, eps=1e-4, atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------------------
# 1-D eager PM: the degenerate (exact) reduction. The traction-free
# condition eliminates the only strain, so the surface cell's stiffness
# freezes and no density or shear modification arises.
# --------------------------------------------------------------------------


def _free_surface_model_1d(nx: int = 50, vac: int = 5):
    vp = torch.zeros(nx, dtype=DTYPE)
    vs = torch.zeros(nx, dtype=DTYPE)
    rho = torch.zeros(nx, dtype=DTYPE)
    vp[vac:] = 2000.0
    vs[vac:] = 1150.0
    rho[vac:] = 2000.0
    return vpvsrho_to_lambmubuoyancy(vp, vs, rho)


def test_pm_1d_coefficients() -> None:
    """In 1-D the classifier marks one surface cell and freezes only it.

    The single traction-free condition eliminates the single strain, so the
    surface cell's normal-stress stiffness is exactly zero (the 1-D analogue
    of the eq 25 joint elimination, which is exact), the interior keeps
    ``lamb + 2 mu`` exactly, no transition cells exist, and the staggered
    buoyancy is unmodified (no velocity node lies in the surface plane).
    """
    nx, vac = 50, 5
    lamb, mu, buoyancy = _free_surface_model_1d(nx, vac)
    lamb_b = lamb.unsqueeze(0)
    mu_b = mu.unsqueeze(0)
    face, transition = _pm_surface_masks(mu_b, 1)
    assert bool(face[0][0, vac])
    assert int(face[0].sum().item()) == 1
    assert not bool(transition.any())
    c = _pm_normal_stiffness(lamb_b, mu_b, (face, transition))
    assert c[0][0][0, vac].item() == 0.0
    lp2m = lamb + 2.0 * mu
    assert c[0][0][0, vac + 5].item() == lp2m[vac + 5].item()
    (buoyancy_x,) = prepare_parameters(mu_b, buoyancy.unsqueeze(0))
    (bx_m,) = _pm_modify_buoyancy([buoyancy_x.clone()], (face, transition))
    assert torch.equal(bx_m, buoyancy_x)


def test_pm_1d_frozen_surface_stress_stays_zero() -> None:
    """The 1-D surface stress stays bitwise zero while the wave reflects."""
    nx, vac = 60, 5
    lamb, mu, buoyancy = _free_surface_model_1d(nx, vac)
    dx = 5.0
    dt = 0.4 * dx / 2000.0
    nt = 60
    amp = ricker(15.0, nt, dt, 0.05, dtype=DTYPE).reshape(1, 1, -1)
    src = torch.tensor([[[vac + 20]]])
    out = elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_x=amp,
        source_locations_x=src,
        pml_width=0,
        max_vel=2000.0,
        pml_freq=15.0,
        accuracy=2,
        surface_method="pm",
        backend="eager",
    )
    assert out.sigmaxx_0.isfinite().all()
    assert out.sigmaxx_0[0, vac].item() == 0.0
    assert out.sigmaxx_0.abs().max().item() > 0.0
    assert out.vx_0.abs().max().item() > 0.0


def test_pm_1d_adjoint_dot_product() -> None:
    """The 1-D eager PM backward pass is the exact transpose of the forward."""
    nx, vac = 40, 4
    lamb, mu, buoyancy = _free_surface_model_1d(nx, vac)
    dx = 6.0
    dt = 0.4 * dx / 2000.0
    nt = 24
    src = torch.tensor([[[vac + 16]]])
    rec = torch.tensor([[[vac + 4], [vac + 26]]])

    def forward(source_amplitudes):
        out = elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_x=source_amplitudes,
            source_locations_x=src,
            receiver_locations_x=rec,
            pml_width=[0, 8],
            max_vel=2000.0,
            pml_freq=15.0,
            accuracy=2,
            surface_method="pm",
            backend="eager",
        )
        return out.receiver_amplitudes_x

    torch.manual_seed(0)
    s = torch.randn(1, 1, nt, dtype=DTYPE, requires_grad=True)
    fs = forward(s)
    r = torch.randn_like(fs)
    lhs = (fs * r).sum()
    lhs.backward()
    rhs = (s * s.grad).sum()
    assert abs(lhs.item() - rhs.item()) <= 1e-10 * abs(lhs.item())


def test_pm_1d_compiled_raises() -> None:
    """1-D PM is eager-only; the compiled backend raises ``NotImplementedError``."""
    lamb, mu, buoyancy = _free_surface_model_1d()
    amp = torch.zeros(1, 1, 3, dtype=DTYPE)
    loc = torch.tensor([[[20]]])
    with pytest.raises(NotImplementedError, match="eager"):
        elastic(
            lamb,
            mu,
            buoyancy,
            5.0,
            0.001,
            source_amplitudes_x=amp,
            source_locations_x=loc,
            surface_method="pm",
        )
