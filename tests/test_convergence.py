"""Taylor / linearisation convergence-rate tests (compiled backend).

For a nonlinear forward map ``f(m)`` (receivers as a function of the model) and
a fixed perturbation direction ``dm``, a correct linearisation ``J`` must give

    ||f(m + h dm) - f(m)||            = O(h)     (first order)
    ||f(m + h dm) - f(m) - h J(dm)||  = O(h^2)   (second order)

as ``h -> 0``. These tests assert the *measured log-log slope* of each residual
against ``h`` -- not a single-point tolerance -- so they verify the convergence
*rate*, which is what actually certifies that ``J`` is the exact derivative
(the first-order term alone only needs continuity; the second-order term
pins the linearisation). ``J(dm)`` is taken from the propagator's forward-mode
rule (``torch.func.jvp``) where one exists on the compiled backend
(``scalar``/``acoustic`` route it through their Born machinery -- see
``test_scalar_jvp.py``/``test_acoustic_jvp.py``); for ``elastic`` there is no
compiled ``jvp``, so the directional derivative is probed through a fixed random
receiver-space projection ``<r, f(m)>`` whose model gradient ``g`` gives
``<r, J dm> = <g, dm>`` by the adjoint identity.

The Born pairs are additionally checked against *their own* linearisation: the
Born scattered field must match the finite difference
``(f_nonlinear(m + h dm) - f_nonlinear(m)) / h`` of the corresponding nonlinear
propagator with an O(h) (forward-difference) residual.

All runs use the compiled backend (``backend="compiled"``) -- the point is to
certify the hand-written C forward/adjoint/Born kernels. The perturbation is
smooth and confined to the interior (zero in the PML ring), both because the
scalar/acoustic ``jvp`` is only defined for interior model tangents and because
a boundary perturbation would not linearise cleanly. ``h`` spans
``{0.3, 0.1, 0.03, 0.01}`` times a perturbation that is already a small fraction
of the model (so every sample is deep in the asymptotic regime yet above the
float64 rounding floor).

A second, independent family of tests refines the *grid* instead of a
perturbation, and so measures the propagators' spatial order of accuracy. The
same physical problem -- a smooth Gaussian pressure pulse released from rest in
a smoothly varying medium -- is solved on grids refined by ``REFINEMENTS`` and
compared against a ``REFERENCE_REFINEMENT``-times refined solution; the
measured log-log slope of the receiver misfit against the grid spacing must
match the requested ``accuracy``. Three details make that slope the spatial
order and nothing else:

* Only normal-stress/pressure receivers are used. Those sit on the integer
  nodes, so their physical position is identical on every grid, whereas a
  staggered velocity node sits half a cell away and therefore *moves* as the
  grid is refined, contributing a spurious O(dx) term.
* The time step is fixed at a value that satisfies the CFL condition on the
  finest grid, so every run takes exactly one internal step per output step and
  the time-discretisation error is common to all of them and cancels in the
  differences.
* The recording window ends well before the pulse reaches the absorbing
  boundary (verified: enlarging the domain leaves the traces bit-identical), so
  what is measured is the accuracy of the interior stencils.

The models are smooth and the density is constant: the stiffnesses live on the
integer nodes and are used directly, but buoyancy is arithmetically averaged
onto the staggered velocity nodes, which is only second-order accurate and
would otherwise cap the observed order at two.
"""

import warnings

import numpy as np
import pytest
import torch
from helpers import c_backend_available
from torch.func import jvp

import deepwave
from deepwave import vpvsrho_to_lambmubuoyancy
from deepwave.acoustic import acoustic_output_names
from deepwave.acoustic_born import acoustic_born_output_names
from deepwave.acoustic_vti import acoustic_vti_output_names
from deepwave.discretisation import cfl_condition_n
from deepwave.elastic import elastic_output_names
from deepwave.elastic_born import elastic_born_output_names
from deepwave.viscoacoustic import viscoacoustic_output_names
from deepwave.wavelets import ricker

DTYPE = torch.float64
DEVICE = torch.device("cpu")
DX = 5.0
FREQ = 25.0
PML = 5
N, NT, DT = 40, 40, 0.0006
MAX_VEL = 2500.0
# Perturbation multipliers spanning ~1.5 decades in the asymptotic regime.
H = (0.3, 0.1, 0.03, 0.01)

# Grid-refinement (spatial-order) settings; see the module docstring. The
# coarse grid is GRID_N points of GRID_DX with a GRID_PML-cell absorbing frame,
# and a refinement by ``r`` scales all three so that the physical geometry --
# domain, boundary thickness, source pulse, receiver positions -- is unchanged
# and every coarse node ``i`` is the refined node ``r * i``.
GRID_N = 81
GRID_DX = 8.0
GRID_PML = 8
GRID_DT = 1.5e-4
GRID_NT = 500
GRID_FREQ = 20.0  # ~the dominant frequency of the pulse below
GRID_SIGMA = 16.0  # standard deviation of the initial Gaussian pulse
GRID_RHO = 2200.0
GRID_VP = 2000.0
GRID_MAX_VEL = 2650.0  # >= max qP speed; sets the CFL step and the PML profile
GRID_ACCURACY = 4
REFINEMENTS = (1, 2, 3)
REFERENCE_REFINEMENT = 8
# Receiver offsets (in coarse cells) from the pulse centre, sampling the
# symmetry axis, the transverse direction, and two oblique directions.
GRID_RECEIVER_OFFSETS = ((-14, 0), (0, 14), (10, 10), (-7, -12))


def _slope(residuals) -> float:
    """Least-squares log-log slope of ``residuals`` against ``H``."""
    return float(np.polyfit(np.log(H), np.log(residuals), 1)[0])


def _spatial_order(errors) -> float:
    """Least-squares log-log slope of ``errors`` against the grid spacing."""
    spacings = [GRID_DX / refinement for refinement in REFINEMENTS]
    return float(np.polyfit(np.log(spacings), np.log(errors), 1)[0])


def _refined_grid(refinement):
    """``(n, dx, pml_width, node coordinates)`` at the given refinement."""
    n = refinement * (GRID_N - 1) + 1
    dx = GRID_DX / refinement
    return n, dx, GRID_PML * refinement, torch.arange(n, dtype=DTYPE) * dx


def _grid_kwargs(refinement):
    """Propagator arguments shared by the two grid-refinement tests."""
    _, dx, pml_width, _ = _refined_grid(refinement)
    return {
        "grid_spacing": dx,
        "dt": GRID_DT,
        "nt": GRID_NT,
        "accuracy": GRID_ACCURACY,
        "pml_width": pml_width,
        "pml_freq": GRID_FREQ,
        "max_vel": GRID_MAX_VEL,
        "origin": (0, 0),
        "backend": "compiled",
    }


def _gaussian_pulse(refinement):
    """The same physical Gaussian pressure pulse, sampled on a refined grid.

    A Gaussian this wide is resolved to well below the float64 floor even on
    the coarsest grid (its amplitude at the coarse Nyquist wavenumber is
    ~1e-9), so the representation of the initial data never limits the
    measured order.
    """
    n, _, _, coord = _refined_grid(refinement)
    centre = (GRID_N - 1) / 2 * GRID_DX  # a node on every grid
    squared_radius = (coord.reshape(-1, 1) - centre) ** 2 + (
        coord.reshape(1, -1) - centre
    ) ** 2
    return torch.exp(-squared_radius / (2 * GRID_SIGMA**2)).reshape(1, n, n)


def _smooth_vp(refinement):
    """A smooth 10% velocity variation, the same physical field on every grid."""
    _, _, _, coord = _refined_grid(refinement)
    span = (GRID_N - 1) * GRID_DX
    return GRID_VP * (
        1
        + 0.1
        * torch.sin(2 * torch.pi * coord.reshape(-1, 1) / span)
        * torch.cos(2 * torch.pi * coord.reshape(1, -1) / span)
    )


def _grid_receivers(refinement):
    """Integer-node receivers at fixed physical positions on every grid."""
    centre = (GRID_N - 1) // 2
    return refinement * torch.tensor(
        [[[centre + dy, centre + dx] for dy, dx in GRID_RECEIVER_OFFSETS]],
        dtype=torch.long,
    )


def _measure_spatial_order(run) -> tuple[list[float], float]:
    """Relative misfits against the fine-grid reference, and their log-log slope.

    ``run(refinement)`` returns the receiver traces on that grid; all grids
    share ``GRID_DT`` and ``GRID_NT``, so the traces are directly comparable.
    """
    # GRID_DT must already be stable on the finest grid, so that every run takes
    # exactly one internal step per output sample and no grid gets a different
    # (finer) time discretisation than the others.
    finest = GRID_DX / REFERENCE_REFINEMENT
    assert cfl_condition_n([finest, finest], GRID_DT, GRID_MAX_VEL)[1] == 1
    reference = run(REFERENCE_REFINEMENT)
    assert reference.abs().max().item() > 0
    errors = [
        ((run(refinement) - reference).norm() / reference.norm()).item()
        for refinement in REFINEMENTS
    ]
    return errors, _spatial_order(errors)


def _smooth_interior(scale, seed):
    """A smooth random field of the given max-abs ``scale``, zero in the PML.

    Confining the perturbation to the interior keeps it inside the region where
    the compiled forward-mode (``jvp``) rule is defined and avoids the
    ill-conditioned PML edge.
    """
    torch.manual_seed(seed)
    f = torch.randn(N, N, dtype=DTYPE)
    for _ in range(3):  # a few box blurs -> smooth
        f = 0.25 * (
            f + torch.roll(f, 1, 0) + torch.roll(f, 1, 1) + torch.roll(f, -1, 0)
        )
    mask = torch.zeros(N, N, dtype=DTYPE)
    mask[PML:-PML, PML:-PML] = 1.0
    f = f * mask
    return f / f.abs().max() * scale


def _survey():
    """A central source and a line of interior near-surface receivers."""
    src = ricker(FREQ, NT, DT, 1.5 / FREQ).to(DTYPE).reshape(1, 1, -1)
    src_loc = torch.tensor([[[N // 2, N // 2]]])
    rec_loc = torch.zeros(1, 5, 2, dtype=torch.long)
    for j in range(5):
        rec_loc[0, j] = torch.tensor([PML + 1 + j, N // 2])
    return src, src_loc, rec_loc


def _skip_if_no_c(*props) -> None:
    for prop in props:
        if not c_backend_available(prop, device=DEVICE):
            pytest.skip(f"compiled {prop} backend not present in libdeepwave_C")


@pytest.mark.parametrize("prop", ["scalar", "acoustic"])
def test_taylor_convergence_jvp(prop) -> None:
    """First-/second-order Taylor convergence via the compiled ``jvp`` rule.

    ``scalar`` and ``acoustic`` expose a compiled forward-mode derivative, so
    ``J(dm)`` is the exact ``jvp`` tangent. The first-order residual must fall as
    O(h) (slope >= 0.95) and the ``jvp``-corrected second-order residual as
    O(h^2) (slope >= 1.9).
    """
    _skip_if_no_c(prop)
    src, sl, rl = _survey()
    torch.manual_seed(1)
    v = 1500.0 + 30.0 * torch.rand(N, N, dtype=DTYPE)
    dv = _smooth_interior(1.0, 3)
    common = {
        "pml_width": PML,
        "pml_freq": FREQ,
        "accuracy": 4,
        "max_vel": MAX_VEL,
        "backend": "compiled",
    }
    if prop == "scalar":

        def f(vv):
            return deepwave.scalar(
                vv,
                DX,
                DT,
                source_amplitudes=src,
                source_locations=sl,
                receiver_locations=rl,
                **common,
            )[-1]
    else:
        rho = 2000.0 + 50.0 * torch.rand(N, N, dtype=DTYPE)
        ridx = acoustic_output_names(2).index("receiver_amplitudes_p")

        def f(vv):
            return deepwave.acoustic(
                vv,
                rho,
                DX,
                DT,
                source_amplitudes_volume_rate=src,
                source_locations_volume_rate=sl,
                receiver_locations_p=rl,
                **common,
            )[ridx]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # interior tangent: no PML warning
        primal, tangent = jvp(f, (v,), (dv,))
    first, second = [], []
    for h in H:
        perturbed = f(v + h * dv)
        first.append((perturbed - primal).norm().item())
        second.append((perturbed - primal - h * tangent).norm().item())
    assert _slope(first) >= 0.95, (first, _slope(first))
    assert _slope(second) >= 1.9, (second, _slope(second))


def test_taylor_convergence_elastic() -> None:
    """First-/second-order Taylor convergence for ``elastic`` via projection.

    ``elastic`` has no compiled forward-mode rule, so the directional derivative
    is probed through a fixed random receiver-space projection: with
    ``phi(m) = <r, f(m)>`` and its model gradient ``g``, the exact linear term is
    ``<r, J dm> = <g, dm>``. The velocity (y) receiver carries the signal here
    (the 2-D pressure receiver of an explosive source is ~0). The perturbation is
    a small (0.5%) fraction of ``lamb`` so every ``h`` is in the linear regime.
    """
    _skip_if_no_c("elastic")
    src, sl, rl = _survey()
    torch.manual_seed(1)
    vp = 1600.0 + 50.0 * torch.rand(N, N, dtype=DTYPE)
    vs = 800.0 + 50.0 * torch.rand(N, N, dtype=DTYPE)
    rho = 2000.0 + 100.0 * torch.rand(N, N, dtype=DTYPE)
    lamb, mu, buoyancy = (t.detach() for t in vpvsrho_to_lambmubuoyancy(vp, vs, rho))
    dlamb = _smooth_interior(0.005 * lamb.abs().mean().item(), 3)
    ridx = elastic_output_names(2).index("receiver_amplitudes_y")
    common = {
        "source_amplitudes_y": src,
        "source_locations_y": sl,
        "receiver_locations_y": rl,
        "pml_width": PML,
        "pml_freq": FREQ,
        "accuracy": 4,
        "max_vel": MAX_VEL,
        "backend": "compiled",
    }

    def phi(la, r):
        return (deepwave.elastic(la, mu, buoyancy, DX, DT, **common)[ridx] * r).sum()

    torch.manual_seed(9)
    out0 = deepwave.elastic(lamb, mu, buoyancy, DX, DT, **common)[ridx]
    assert out0.abs().max().item() > 0
    r = torch.randn_like(out0)
    leaf = lamb.clone().requires_grad_(True)
    phi0 = phi(leaf, r)
    (g,) = torch.autograd.grad(phi0, leaf)
    jdm = (g * dlamb).sum().item()
    phi0v = phi0.item()
    first, second = [], []
    for h in H:
        with torch.no_grad():
            ph = phi(lamb + h * dlamb, r).item()
        first.append(abs(ph - phi0v))
        second.append(abs(ph - phi0v - h * jdm))
    assert _slope(first) >= 0.95, (first, _slope(first))
    assert _slope(second) >= 1.9, (second, _slope(second))


@pytest.mark.parametrize("prop", ["scalar_born", "acoustic_born", "elastic_born"])
def test_born_linearization(prop) -> None:
    """The Born scattered field is the linearisation of the nonlinear forward.

    For each Born pair, the scattered receivers computed for a model perturbation
    ``dm`` must match the forward finite difference
    ``(f_nl(m + h dm) - f_nl(m)) / h`` of the corresponding nonlinear propagator,
    with an O(h) residual (slope >= 0.95) as ``h -> 0``.
    """
    src, sl, rl = _survey()
    common = {
        "pml_width": PML,
        "pml_freq": FREQ,
        "accuracy": 4,
        "max_vel": MAX_VEL,
        "backend": "compiled",
    }
    if prop == "scalar_born":
        _skip_if_no_c("scalar", "scalar_born")
        torch.manual_seed(1)
        v = 1500.0 + 30.0 * torch.rand(N, N, dtype=DTYPE)
        dv = _smooth_interior(1.0, 3)
        sc = {"source_amplitudes": src, "source_locations": sl}

        def nl(step):
            return deepwave.scalar(
                v + step * dv, DX, DT, receiver_locations=rl, **sc, **common
            )[-1]

        scattered = deepwave.scalar_born(
            v, DX, DT, dv=dv, sc_receiver_locations=rl, **sc, **common
        )[-1]
    elif prop == "acoustic_born":
        _skip_if_no_c("acoustic", "acoustic_born")
        torch.manual_seed(1)
        v = 1500.0 + 30.0 * torch.rand(N, N, dtype=DTYPE)
        rho = 2000.0 + 50.0 * torch.rand(N, N, dtype=DTYPE)
        dv = _smooth_interior(1.0, 3)
        drho = _smooth_interior(1.0, 4)
        sc = {"source_amplitudes_volume_rate": src, "source_locations_volume_rate": sl}
        idx = acoustic_output_names(2).index("receiver_amplitudes_p")
        sidx = acoustic_born_output_names(2).index("sc_receiver_amplitudes_p")

        def nl(step):
            return deepwave.acoustic(
                v + step * dv,
                rho + step * drho,
                DX,
                DT,
                receiver_locations_p=rl,
                **sc,
                **common,
            )[idx]

        scattered = deepwave.acoustic_born(
            v, rho, DX, DT, dv=dv, drho=drho, sc_receiver_locations_p=rl, **sc, **common
        )[sidx]
    else:  # elastic_born
        _skip_if_no_c("elastic", "elastic_born")
        torch.manual_seed(1)
        vp = 1600.0 + 50.0 * torch.rand(N, N, dtype=DTYPE)
        vs = 800.0 + 50.0 * torch.rand(N, N, dtype=DTYPE)
        rho = 2000.0 + 100.0 * torch.rand(N, N, dtype=DTYPE)
        lamb, mu, buoy = (t.detach() for t in vpvsrho_to_lambmubuoyancy(vp, vs, rho))
        dlamb = _smooth_interior(0.005 * lamb.abs().mean().item(), 3)
        dmu = _smooth_interior(0.005 * mu.abs().mean().item(), 4)
        dbuoy = _smooth_interior(0.005 * buoy.abs().mean().item(), 5)
        sc = {"source_amplitudes_y": src, "source_locations_y": sl}
        idx = elastic_output_names(2).index("receiver_amplitudes_y")
        sidx = elastic_born_output_names(2).index("sc_receiver_amplitudes_y")

        def nl(step):
            return deepwave.elastic(
                lamb + step * dlamb,
                mu + step * dmu,
                buoy + step * dbuoy,
                DX,
                DT,
                receiver_locations_y=rl,
                **sc,
                **common,
            )[idx]

        scattered = deepwave.elastic_born(
            lamb,
            mu,
            buoy,
            DX,
            DT,
            dlamb=dlamb,
            dmu=dmu,
            dbuoyancy=dbuoy,
            sc_receiver_locations_y=rl,
            **sc,
            **common,
        )[sidx]

    assert scattered.abs().max().item() > 0
    f0 = nl(0.0)
    residuals = [(((nl(h) - f0) / h) - scattered).norm().item() for h in H]
    assert _slope(residuals) >= 0.95, (residuals, _slope(residuals))


def test_spatial_order_acoustic_vti() -> None:
    """``acoustic_vti`` converges at its requested spatial order under refinement.

    A genuinely anisotropic medium (Thomsen ``eps = 0.2``, ``delta = 0.08``,
    so ``eps > delta`` satisfies the Grechka/Duveneck well-posedness condition
    ``c11 c33 >= c13**2`` strictly) carries a smooth pressure pulse released
    from rest on both normal stresses. Refining the grid must reduce the misfit
    against the fine-grid reference as O(dx**accuracy). Measured (float64,
    ``accuracy=4``): errors ~5.2e-3, 3.3e-4, 6.5e-5, giving an observed order of
    ~3.99 against the 3.8 acceptance floor (0.95 of the design order, the same
    discipline as the Taylor tests above).
    """
    _skip_if_no_c("acoustic_vti")
    eps, delta = 0.2, 0.08

    def run(refinement):
        n, _, _, _ = _refined_grid(refinement)
        c33 = GRID_RHO * _smooth_vp(refinement) ** 2
        c11 = c33 * (1 + 2 * eps)
        c13 = c33 * (1 + 2 * delta) ** 0.5
        assert (c11 * c33 >= c13**2).all()  # Grechka/Duveneck stability
        buoyancy = torch.full((n, n), 1 / GRID_RHO, dtype=DTYPE)
        receivers = _grid_receivers(refinement)
        pulse = _gaussian_pulse(refinement)
        out = deepwave.acoustic_vti(
            c11,
            c13,
            c33,
            buoyancy,
            sigma_v_0=pulse,
            sigma_h_0=pulse,
            receiver_locations_sigma_v=receivers,
            receiver_locations_sigma_h=receivers,
            **_grid_kwargs(refinement),
        )
        names = acoustic_vti_output_names(2)
        return torch.cat(
            [
                out[names.index("receiver_amplitudes_sigma_v")],
                out[names.index("receiver_amplitudes_sigma_h")],
            ],
            dim=1,
        )

    errors, order = _measure_spatial_order(run)
    assert order >= 0.95 * GRID_ACCURACY, (errors, order)


def test_spatial_order_viscoacoustic() -> None:
    """``viscoacoustic`` converges at its requested spatial order under refinement.

    The attenuating counterpart of ``test_spatial_order_acoustic_vti``: the same
    pulse, medium and refinement scheme, with a strongly attenuating constant
    ``Q = 15`` at ``q_freq``, which changes the recorded traces by ~16% relative
    to a lossless run, so the relaxation mechanism is genuinely active rather
    than a perturbation of the acoustic case. Because ``Q`` and the time step
    are held fixed, the standard-linear-solid relaxation times are the same
    physical quantities on every grid and the memory-variable update contributes
    no grid-dependent error, leaving the spatial stencils. Measured (float64,
    ``accuracy=4``): errors ~4.7e-3, 3.0e-4, 5.9e-5, giving an observed order of
    ~3.98 against the 3.8 acceptance floor.
    """
    _skip_if_no_c("viscoacoustic")

    def run(refinement):
        n, _, _, _ = _refined_grid(refinement)
        rho = torch.full((n, n), GRID_RHO, dtype=DTYPE)
        q = torch.full((n, n), 15.0, dtype=DTYPE)
        out = deepwave.viscoacoustic(
            _smooth_vp(refinement),
            rho,
            q,
            pressure_0=_gaussian_pulse(refinement),
            receiver_locations_p=_grid_receivers(refinement),
            q_freq=GRID_FREQ,
            **_grid_kwargs(refinement),
        )
        return out[viscoacoustic_output_names(2).index("receiver_amplitudes_p")]

    errors, order = _measure_spatial_order(run)
    assert order >= 0.95 * GRID_ACCURACY, (errors, order)
