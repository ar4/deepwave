#!/usr/bin/env python
"""Boundary face-fill energy, spectral-radius and adjointness instrument.

A reflecting face (``boundary="free"``, ``"rigid"``, ``"periodic"``, or a
zero-width ``"pml"`` face) replaces the interior finite-difference stencil on
the rows next to that face. Whether the resulting scheme still conserves a
discrete energy -- and is therefore stable up to the CFL limit -- is decided
by one algebraic property of those replacement rows, which this script
measures directly on the propagators' own kernels. It is not part of the test
suite; it is a standalone diagnostic, run by hand.

Everything below is computed in float64 on the CPU, in a closed PML-free box
(``pml_width=0``), with no sources and no receivers.

The criterion
-------------
Every velocity-stress propagator advances

    v^(n+1/2) = v^(n-1/2) + dt B A_v sigma^n
    sigma^(n+1) = sigma^n + dt C A_sigma v^(n+1/2)

where ``B`` is the staggered buoyancy (diagonal, one entry per velocity node),
``C`` the per-node discrete stiffness, and ``A_v``/``A_sigma`` the spatial
stencils *including* the face-fill rows. The time-staggered energy

    E^n = 1/2 v^(n+1/2) . W_v B^-1 v^(n-1/2) + 1/2 sigma^n . W_sigma C^-1 sigma^n

is exactly conserved, for a diagonal weighting ``W_v``, ``W_sigma`` > 0, if and
only if

    W_v A_v = (W_sigma A_sigma)^T.

The centred staggered interior stencils satisfy this with ``W = I``; only
face-fill rows can break it. Writing ``G_v = W_v B^-1`` and
``G_sigma = W_sigma C^-1`` for the two halves of the energy metric, and noting
that the off-diagonal blocks of the one-step operator ``M`` are exactly
``M_v,sigma = dt B A_v`` and ``M_sigma,v = dt C A_sigma`` whichever of the two
half-step updates the kernel applies first, the criterion becomes the residual

    R = G_sigma M_sigma,v + M_v,sigma^T G_v     (zero when adjoint)

which needs no separate knowledge of ``A_v`` and ``A_sigma``. Each nonzero row
of ``R`` names a stress node, and each nonzero column a velocity node, whose
face-fill row is not the transpose of its partner. ``--adjointness`` prints
those rows with the face they belong to, their distance from it, and their
largest coefficients.

For the regular-grid propagators (``scalar``, ``scalar_born``), which advance
``u^(n+1) = 2 u^n - u^(n-1) + dt^2 c^2 L u^n``, the criterion is that ``W L``
is symmetric for a diagonal ``W > 0``, and the residual printed is
``L - L^T`` (with ``W = I``).

Uniform weights (``W = I``) are the target, so the metric this script uses is
``G_v = B^-1`` and ``G_sigma = C^-1``, built from the kernels' own staggered
model parameters:

* the staggered density is the arithmetic mean of the two neighbouring cell
  densities, so ``G_v`` is that mean;
* the normal-stress nodes carry the constitutive matrix of the propagator
  (``[[lambda + 2 mu, lambda], [lambda, lambda + 2 mu]]`` isotropic,
  ``[[c33, c13], [c13, c11]]`` for the VTI families), and ``G_sigma`` is its
  inverse;
* the shear-stress nodes carry the harmonically averaged shear modulus, and
  each shear term is counted once;
* a singular node (a vacuum cell, or a fluid cell with no shear modulus) uses
  a pseudo-inverse, which sends its weight to zero.

A face that conserves energy only with position-dependent (but
medium-independent) weights still proves stability; ``--solve-weights`` solves
the feasibility problem for such a weighting and reports it.

What each mode measures
-----------------------
``--adjointness``
    Builds ``M`` and prints ``max|R|`` relative to the size of its two terms,
    together with the table of offending rows. A correct fill gives round-off
    (about 1e-15) on a homogeneous *and* on a heterogeneous medium: a fill can
    be symmetric for constant parameters alone, for instance by using the
    node's own modulus where the transpose needs its neighbour's.

    ``R`` decides a case only where the operator left by the ghost elimination
    still has the leapfrog block form the derivation above assumes. That is
    measured per case and printed as ``structure``: the relative distance of the
    ghost-eliminated operator from the nearer of the two leapfrog forms (the
    velocity half step first, or the stress half step first). A fill whose ghost
    value is consumed one step before the value it leaves stored breaks the
    form, and then ``R`` can be zero although the scheme grows, so a case whose
    ``structure`` is not round-off is marked ``R provisional (ghost rows:
    energy/spectral gates decide)``: its ``R`` is reported but never counted as
    a pass, and with no other gate requested the case does not pass at all.

``--energy``
    Reports the worst relative change of ``E`` over a long run of the
    propagator itself, started from a smooth velocity pulse with zero stress
    (a smooth displacement pulse at rest for the regular-grid family), plus
    the worst per-step relative change over random states. The sum runs with
    uniform weights over every node except dead nodes (whose row of ``M`` is
    exactly ``e_i``, so the node is a decoupled constant) and ghost rows
    (whose diagonal in ``M`` is zero although the row is not, so the fill
    rewrites the node from other nodes every step). A row held at zero is
    neither: it contributes nothing.

``--spectral``
    Reports ``rho(M) - 1`` on the live subspace, and lists every eigenvalue
    with ``|lambda| - 1 > 1e-12`` (the twenty largest, then the number of the
    rest). A correct scheme has ``rho(M) = 1`` exactly, but a defective unit
    eigenvalue can be computed up to about 1e-8 off, so this is a screen rather
    than the authoritative check: it catches per-step defects of 1e-5 and pins
    their dependence on the medium, while a non-normal ``M`` with linear growth
    passes it and is caught by the energy gate instead. On nodes with no shear
    modulus only the isotropic normal stress is kept, since the stress there
    must lie in the range of ``C``.

    A unit eigenvalue of multiplicity two is computed with an error of order
    ``sqrt(eps)``, about 1.5e-8, which is just above the 1e-8 pass condition, so
    a band immediately above it is reported separately rather than as a failure:
    a ``rho(M) - 1`` in (1e-8, 1e-6] on a case whose adjointness residual is
    round-off, and whose energy gate passes or was not requested, is labelled
    ``S~``, the defective-unit-eigenvalue class. It is informational, counted in
    its own tally, and never a plain pass nor a plain fail; the energy gate is
    the authoritative verdict for it, and where it did not run the label says so.
    Every ``rho(M) - 1`` above 1e-6, and every case whose adjointness or energy
    gate fails, stays a hard ``S`` failure. The class is a reading of the pass
    condition, not a loosened tolerance: the tolerance itself is unchanged, and
    the smallest defect the faces measured here exhibit is 1e-5, well above the
    band.

``--controls``
    Runs all three gates on the faces that are expected to conserve: every
    face a zero-width ``"pml"``, a vacuum layer against a zero-width box,
    acoustic ``"rigid"``, and elastic ``"rigid"`` at accuracy 2. These must
    pass: they are the evidence that the metric and the classification above
    are the kernels' own. A control failure means this script is wrong until
    shown otherwise, and is never to be answered by loosening a tolerance.

The readings chosen where a definition could go either way
----------------------------------------------------------
Each was settled by requiring the conserving controls to pass, and each is
checked per case rather than assumed.

* A node whose row of ``M`` is exactly ``e_i`` is a decoupled constant (dead);
  one whose row is zero is held at zero; one whose diagonal is zero although
  its row is not is a ghost, an image of the live state. Ghosts are eliminated
  algebraically before the residual is formed, which is what distinguishes a
  fill that is adjoint from one that computes a ghost and then applies the
  interior stencil to it.
* The staggered density and the harmonically averaged shear modulus have no
  neighbour beyond the outermost half-integer node; the kernels' model padding
  replicates the edge cell, so the edge value is used. Those nodes turn out to
  carry no state in every case measured, so the choice does not reach the
  result.
* The attenuating propagators run with Q = 1e30 and the Born propagators with
  zero perturbations, so that the operator under test is the lossless or
  background one. The state left out (the attenuation memory, the scattered
  fields) is checked to stay at zero, and reported as ``leak``; a nonzero
  ``leak`` means the operator is not closed and the numbers beside it do not
  mean what they say.
* ``acoustic_vti``'s two stresses share the constitutive matrix
  ``[[c33, c13], [c13, c11]]`` acting on the axial and the transverse strain,
  in 2-D and 3-D alike.
* Deepwave's own Courant number (``dt v sqrt(sum_i 1/dx_i^2)``) and the
  per-axis one (``dt v / dx``) differ by ``sqrt(ndim)`` on a cubic grid, so
  both are offered (``--courant-convention``) and the one in force is printed
  with every case (``C=`` or ``Cax=``).

Not covered
-----------


Known limitations
-----------------
* A dead node -- one whose row of ``M`` is exactly ``e_i`` -- can still have a
  nonzero *column*: nothing writes it, but it drives the live state every step.
  All three gates leave such nodes out (the initial pulse is zeroed there, the
  energy sum skips them, the spectral basis drops them), so a dead node that
  forces the live state is outside everything this script measures. It is not
  hypothetical: acoustic ``free`` at accuracy 2 on a 12x12 grid has 44 dead
  nodes whose columns reach 2.0e-1. Started from a random state that does excite
  them, the live state neither grew nor decayed over 50,000 steps in five
  configurations (2-D accuracy 2, 4 and 8, and 3-D accuracy 2), so the exclusion
  hid no growth there; that is evidence, not a proof that it cannot elsewhere.
* ``--selftest`` checks the gate functions against synthetic operators whose
  verdict is known by construction. It says nothing about the kernels.

Usage
-----
::

    python benchmarks/boundary_energy.py --selftest
    python benchmarks/boundary_energy.py --controls
    python benchmarks/boundary_energy.py --quick
    python benchmarks/boundary_energy.py --adjointness \
        --propagator elastic --faces free --accuracy 4
    python benchmarks/boundary_energy.py --spectral --spectral-media full \
        --propagator elastic --faces rigid

``--quick`` shrinks every dimension of the work (one small grid, two
accuracies, a short run) for smoke use. ``--spectral-media`` applies whenever
the spectral gate runs, whether or not the other gates run with it.
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

import deepwave
from deepwave._common import unchecked_boundary_faces

# ---------------------------------------------------------------------------
# Tolerances and fixed choices
# ---------------------------------------------------------------------------

#: Relative size below which the adjointness residual is round-off.
ADJOINT_TOL = 1e-12

#: Pass condition for the energy gate.
ENERGY_TOL = 1e-10

#: Pass condition for the spectral gate.
SPECTRAL_TOL = 1e-8

#: Eigenvalues further than this from the unit circle are listed individually.
SPECTRAL_REPORT_TOL = 1e-12

#: How many of those eigenvalues are listed per case before only the remaining
#: count is printed.
SPECTRAL_REPORT_LIMIT = 20

#: The top of the band in which a spectral excess is read as a defective unit
#: eigenvalue (the ``S~`` class) rather than as growth. A double unit eigenvalue
#: is computed with an error of order ``sqrt(eps)``, about 1.5e-8; this leaves
#: two decades of margin above that and stays two decades below the smallest
#: defect the faces measured here exhibit.
SPECTRAL_DEFECTIVE_CEILING = 1e-6

#: Default number of propagator steps in the energy gate.
ENERGY_STEPS = 50_000

#: Number of checkpoints at which the energy gate samples the run.
ENERGY_CHECKPOINTS = 50

#: Number of random states used for the per-step energy defect.
DEFECT_PROBES = 20

#: Anisotropy of the VTI/TTI families' media, as Thomsen parameters.
EPSILON, DELTA = 0.2, 0.1

#: Quality factors that switch attenuation off (Q = infinity is rejected).
Q_INFINITE = 1e30

#: Width, in cells, of the vacuum layer used as a conserving control.
VACUUM_WIDTH = 3

FACE_CODES = ("pml", "free", "rigid", "periodic")

#: The families whose fastest wavespeed is not ``vp`` but the horizontal P
#: velocity of the VTI stiffnesses below.
ANISOTROPIC = (
    "acoustic_vti",
    "elastic_vti",
)

VELOCITY_STRESS = (
    "acoustic",
    "acoustic_born",
    "acoustic_vti",
    "elastic",
    "elastic_born",
    "elastic_vti",
    "viscoacoustic",
)
REGULAR_GRID = ("scalar", "scalar_born")
ALL_PROPAGATORS = (
    "scalar",
    "acoustic",
    "elastic",
    "acoustic_vti",
    "elastic_vti",
    "scalar_born",
    "acoustic_born",
    "elastic_born",
    "viscoacoustic",
)

#: The propagators whose state this script reduces to a background operator by
#: setting every model perturbation to zero.
BORN = ("scalar_born", "acoustic_born", "elastic_born")

#: The propagators whose attenuation this script switches off with Q = 1e30.
VISCOUS = ("viscoacoustic",)


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Medium:
    """A homogeneous or heterogeneous medium on an ``n**ndim`` grid.

    ``rho``, ``vp`` and ``vs`` are cell-centred models in nondimensional units
    (``dx = 1``, ``vp`` of order 1), which keeps the one-step operator well
    scaled. ``label`` names the draw for the report.
    """

    rho: torch.Tensor
    vp: torch.Tensor
    vs: torch.Tensor
    label: str

    @property
    def ndim(self) -> int:
        """The number of spatial dimensions."""
        return self.rho.ndim

    def max_vel(self, propagator: str) -> float:
        """The fastest wavespeed ``propagator`` propagates in this medium.

        For the anisotropic families that is the horizontal P velocity, which
        exceeds ``vp`` by the Thomsen ``epsilon`` factor; for the others it is
        ``vp`` itself. It is what the propagator is told as ``max_vel``, and
        what the Courant number below is measured against, so the two agree.
        """
        fastest = float(self.vp.max())
        if propagator in ANISOTROPIC:
            fastest *= math.sqrt(1 + 2 * EPSILON)
        return fastest


def make_medium(
    ndim: int, n: int, kind: str, *, vs_vp: float = 0.5, seed: int = 7
) -> Medium:
    """Builds one medium.

    Args:
        ndim: The number of spatial dimensions.
        n: The number of cells along every axis.
        kind: ``"homogeneous"``, ``"heterogeneous"``, ``"vacuum"`` (a
            homogeneous medium whose first ``VACUUM_WIDTH`` rows along the
            slowest axis are empty) or ``"fluid"`` (a homogeneous medium whose
            first third has no shear modulus).
        vs_vp: The shear-to-P velocity ratio of a homogeneous medium.
        seed: The random seed of a heterogeneous draw.

    Returns:
        The medium.
    """
    shape = (n,) * ndim
    dtype = torch.float64
    if kind == "heterogeneous":
        generator = torch.Generator().manual_seed(seed)

        def draw() -> torch.Tensor:
            return torch.rand(*shape, dtype=dtype, generator=generator)

        rho = 1 + draw()
        vp = 0.7 + 0.3 * draw()
        vs = vp * (0.25 + 0.2 * draw())
        return Medium(rho, vp, vs, f"heterogeneous(seed={seed})")
    rho = torch.ones(shape, dtype=dtype)
    vp = torch.ones(shape, dtype=dtype)
    vs = vs_vp * vp
    if kind == "homogeneous":
        return Medium(rho, vp, vs, f"homogeneous(vs/vp={vs_vp})")
    if kind == "vacuum":
        rho = rho.clone()
        vp = vp.clone()
        vs = vs.clone()
        rho[:VACUUM_WIDTH] = 0
        vp[:VACUUM_WIDTH] = 0
        vs[:VACUUM_WIDTH] = 0
        return Medium(rho, vp, vs, f"vacuum({VACUUM_WIDTH} rows)")
    if kind == "fluid":
        vs = vs.clone()
        vs[: max(1, n // 3)] = 0
        return Medium(rho, vp, vs, f"fluid(top third, vs/vp={vs_vp})")
    raise ValueError(f"unknown medium kind {kind!r}")


def model_arguments(propagator: str, medium: Medium) -> dict[str, torch.Tensor]:
    """The model keyword arguments ``propagator`` takes for ``medium``.

    The Born propagators are given zero perturbations, which makes their
    scattered state decoupled and their background state the operator under
    test. The attenuating propagators are given Q = 1e30, the largest quality
    factor they accept (Q = infinity raises).
    """
    rho, vp, vs = medium.rho, medium.vp, medium.vs
    zero = torch.zeros_like(rho)
    buoyancy = torch.where(rho > 0, 1 / torch.where(rho > 0, rho, 1), zero)
    lamb = rho * (vp**2 - 2 * vs**2)
    mu = rho * vs**2
    q = torch.full_like(rho, Q_INFINITE)
    if propagator == "scalar":
        return {"v": vp}
    if propagator == "scalar_born":
        return {"v": vp, "dv": zero}
    if propagator == "acoustic":
        return {"v": vp, "rho": rho}
    if propagator == "acoustic_born":
        return {"v": vp, "rho": rho, "dv": zero, "drho": zero}
    if propagator == "viscoacoustic":
        return {"v": vp, "rho": rho, "q": q}
    if propagator == "elastic":
        return {"lamb": lamb, "mu": mu, "buoyancy": buoyancy}
    if propagator == "elastic_born":
        return {
            "lamb": lamb,
            "mu": mu,
            "buoyancy": buoyancy,
            "dlamb": zero,
            "dmu": zero,
            "dbuoyancy": zero,
        }
    stiffnesses = vti_stiffnesses(rho, vp, vs)
    if propagator == "acoustic_vti":
        return {
            "c11": stiffnesses["c11"],
            "c13": stiffnesses["c13_acoustic"],
            "c33": stiffnesses["c33"],
            "buoyancy": buoyancy,
        }
    if propagator == "elastic_vti":
        arguments = {
            "c11": stiffnesses["c11"],
            "c13": stiffnesses["c13"],
            "c33": stiffnesses["c33"],
            "c55": stiffnesses["c55"],
            "buoyancy": buoyancy,
        }
        if medium.ndim == 3:
            arguments["c66"] = stiffnesses["c55"]
        return arguments
    raise ValueError(f"unknown propagator {propagator!r}")


def vti_stiffnesses(
    rho: torch.Tensor, vp: torch.Tensor, vs: torch.Tensor
) -> dict[str, torch.Tensor]:
    """The VTI stiffnesses of a medium with Thomsen ``EPSILON``/``DELTA``.

    ``c33 = rho vp^2`` along the symmetry axis, ``c11 = c33 (1 + 2 epsilon)``
    across it, ``c55 = rho vs^2``, and ``c13`` from the Thomsen ``delta``
    definition. The pseudo-acoustic propagator has no ``c55``, so its ``c13``
    is the shear-free limit of the same definition; ``epsilon > delta`` keeps
    ``[[c33, c13], [c13, c11]]`` positive definite.
    """
    c33 = rho * vp**2
    c11 = c33 * (1 + 2 * EPSILON)
    c55 = rho * vs**2
    gap = c33 - c55
    c13 = torch.sqrt(torch.clamp(2 * c33 * gap * DELTA + gap**2, min=0.0))
    c13_acoustic = c33 * math.sqrt(1 + 2 * DELTA)
    return {
        "c11": c11,
        "c13": c13,
        "c13_acoustic": c13_acoustic,
        "c33": c33,
        "c55": c55,
    }


# ---------------------------------------------------------------------------
# Cases and calls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One (propagator, grid, medium, faces, accuracy, time step) point."""

    propagator: str
    medium: Medium
    faces: tuple[str, ...]
    accuracy: int
    courant: float = 0.2
    dx: float = 1.0
    convention: str = "deepwave"

    @property
    def ndim(self) -> int:
        """The number of spatial dimensions."""
        return self.medium.ndim

    @property
    def n(self) -> int:
        """The number of cells along every axis."""
        return int(self.medium.rho.shape[0])

    @property
    def dt(self) -> float:
        """The time step giving the requested Courant number.

        Two conventions are in use and they differ by ``sqrt(ndim)`` on a cubic
        grid, so both are available and the one in force is printed with every
        case:

        * ``"deepwave"``: ``dt v_max sqrt(sum_i 1/dx_i^2)``, which is the
          quantity Deepwave itself limits and reports in ``RunInfo``;
        * ``"per-axis"``: ``dt v_max / dx``, the per-axis ratio.

        Deepwave sub-steps internally when its own Courant number exceeds its
        limit, and one call with ``nt=1`` is then not one step of the scheme,
        which the one-step operator depends on; a case that asks for such a
        time step is refused rather than measured.
        """
        fastest = self.medium.max_vel(self.propagator)
        axes = math.sqrt(self.ndim) if self.convention == "deepwave" else 1.0
        return self.courant * self.dx / (fastest * axes)

    @property
    def face_label(self) -> str:
        """A compact name for the face combination."""
        unique = set(self.faces)
        if len(unique) == 1:
            return next(iter(unique))
        return ",".join(self.faces)

    def label(self) -> str:
        """A one-line name for the report."""
        marker = "C" if self.convention == "deepwave" else "Cax"
        return (
            f"{self.propagator} {self.ndim}-D n={self.n} acc={self.accuracy} "
            f"faces={self.face_label} {marker}={self.courant} {self.medium.label}"
        )


def state_fields(propagator: str, ndim: int) -> list[str]:
    """The public state field names this script treats as the state vector.

    The auxiliary state (the absorbing boundary's memory variables and the
    attenuation memory) is excluded: with ``pml_width=0`` the absorbing
    memory is identically zero, and Q = 1e30 leaves the attenuation memory
    identically zero, so the physical fields close on themselves. The
    scattered fields of a Born propagator are excluded because the operator
    under test is the background one, and a zero perturbation decouples them.
    Both exclusions are verified per case (see :func:`build_one_step`).
    """
    names = [
        name
        for name in deepwave.wavefield_names(propagator, ndim)
        if name != "auxiliary_state"
    ]
    if propagator in BORN:
        names = [name for name in names if "_sc_" not in name]
    return names


def call(
    case: Case,
    state: dict[str, torch.Tensor],
    *,
    nt: int,
    auxiliary_state: Any | None = None,
) -> Any:
    """Runs ``case``'s propagator for ``nt`` steps from ``state``.

    There is no one spelling of an initial state that all nine propagators
    accept. Most take either explicit per-field keyword arguments or an
    ``initial_state`` mapping; ``acoustic_born`` infers
    the number of spatial dimensions before merging an ``initial_state``, so
    they need the explicit fields, while ``elastic_born`` infers it from
    ``initial_state`` alone and ignores the explicit fields when it does, so
    it needs the mapping. Hence the split below.
    """
    medium = case.medium
    fastest = medium.max_vel(case.propagator)
    keywords: dict[str, Any] = {
        "nt": nt,
        "accuracy": case.accuracy,
        "pml_width": 0,
        "pml_freq": fastest / (10 * case.dx),
        "max_vel": fastest,
    }
    if case.propagator in VISCOUS:
        # Required with one relaxation mechanism; irrelevant at Q = 1e30.
        keywords["q_freq"] = keywords["pml_freq"]
    if set(case.faces) != {"pml"}:
        keywords["boundary"] = list(case.faces)
    bundle: dict[str, Any] = {}
    if auxiliary_state is not None:
        bundle["auxiliary_state"] = auxiliary_state
    if case.propagator == "elastic_born":
        bundle.update(state)
        fields: dict[str, Any] = {}
    else:
        fields = dict(state)
    if bundle:
        keywords["initial_state"] = bundle
    propagator = getattr(deepwave, case.propagator)
    # Deepwave itself refuses the faces whose fills do not conserve energy
    # (that is what the measurements below decided), so this instrument, whose
    # whole purpose is to measure those fills and say whether they could be
    # offered, asks for them through the private bypass.
    with warnings.catch_warnings(), unchecked_boundary_faces():
        warnings.simplefilter("ignore")
        return propagator(
            **model_arguments(case.propagator, medium),
            grid_spacing=case.dx,
            dt=case.dt,
            **fields,
            **keywords,
        )


# ---------------------------------------------------------------------------
# The one-step operator
# ---------------------------------------------------------------------------


@dataclass
class Operator:
    """The one-step operator of a case, and the bookkeeping that goes with it.

    Attributes:
        case: The case it belongs to.
        names: The state field names, in the order they occupy ``matrix``.
        matrix: ``M``, with ``M[:, j]`` the image of basis vector ``j``.
        dead: Nodes whose row of ``M`` is exactly ``e_i``: a decoupled
            constant, carrying no dynamics.
        held: Nodes whose row of ``M`` is zero: held at zero by the fill.
        ghost: Nodes whose diagonal in ``M`` is zero although their row is
            not: rewritten from other nodes every step.
        leak: The largest value that appeared in a state field excluded from
            the state vector (the auxiliary state, or a Born scattered
            field). It has to be zero for ``matrix`` to be the whole
            operator.
        seconds: Wall-clock time spent building it.
    """

    case: Case
    names: list[str]
    matrix: npt.NDArray[Any]
    dead: npt.NDArray[Any]
    held: npt.NDArray[Any]
    ghost: npt.NDArray[Any]
    leak: float
    seconds: float

    @property
    def n_cells(self) -> int:
        """The number of cells per field."""
        return int(self.case.n**self.case.ndim)

    @property
    def size(self) -> int:
        """The dimension of the state vector."""
        return len(self.names) * self.n_cells

    def slice_of(self, name: str) -> slice:
        """The entries of the state vector belonging to field ``name``."""
        i = self.names.index(name)
        return slice(i * self.n_cells, (i + 1) * self.n_cells)

    def describe(self, index: int) -> str:
        """``field[node]`` for one entry of the state vector."""
        name = self.names[index // self.n_cells]
        cell = np.unravel_index(index % self.n_cells, (self.case.n,) * self.case.ndim)
        return f"{name}{tuple(int(c) for c in cell)}"

    def live_indices(self) -> npt.NDArray[Any]:
        """The state-vector entries that carry dynamics.

        Dead nodes (a decoupled constant), rows held at zero and ghost rows
        are all excluded: none of them is a degree of freedom of the scheme.
        """
        return np.nonzero(~(self.dead | self.held | self.ghost))[0]

    def effective(self) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """The one-step operator on the live subspace, ghost rows eliminated.

        A ghost row is an image of the live state, so it defines a map ``P``
        from live nodes to the full state; the effective operator is
        ``M[live] P``. That folds the ghost's contribution back on to the
        interior nodes that read it, which is the difference between a fill
        that is adjoint and one that computes a ghost and then applies the
        interior stencil to it. With no ghost rows this is just the live
        submatrix.

        Returns:
            The effective operator, and the state-vector entries its rows and
            columns correspond to.
        """
        live = self.live_indices()
        prolong = np.zeros((self.size, len(live)))
        prolong[live, np.arange(len(live))] = 1
        for ghost in np.nonzero(self.ghost)[0]:
            prolong[ghost] = self.matrix[ghost, live]
        return self.matrix[live] @ prolong, live

    def face_of(self, index: int) -> str:
        """The nearest face of the box, and the distance to it, for an entry."""
        n = self.case.n
        cell = np.unravel_index(index % self.n_cells, (n,) * self.case.ndim)
        best = (n, -1)
        for axis, position in enumerate(cell):
            for side, distance in ((0, int(position)), (1, n - 1 - int(position))):
                if distance < best[0]:
                    best = (distance, 2 * axis + side)
        face = best[1]
        return f"{self.case.faces[face]}@face{face}+{best[0]}"


def build_one_step(case: Case, *, chunk: int = 512) -> Operator:
    """Builds the exact one-step operator of ``case``.

    One basis vector per shot is propagated through the initial state with
    ``nt=1``, in batches of ``chunk`` shots, which is the operator the kernel
    actually applies (no differentiation of any kind is involved).

    Raises:
        RuntimeError: If the propagator sub-steps internally, so that one
            call is not one step of the scheme.
    """
    started = time.perf_counter()
    names = state_fields(case.propagator, case.ndim)
    n_cells = case.n**case.ndim
    size = len(names) * n_cells
    shape = (case.n,) * case.ndim
    matrix = np.zeros((size, size))
    leak = 0.0
    for start in range(0, size, chunk):
        stop = min(size, start + chunk)
        width = stop - start
        basis = torch.zeros(width, size, dtype=torch.float64)
        basis[torch.arange(width), torch.arange(start, stop)] = 1
        basis = basis.reshape(width, len(names), *shape)
        state = {name: basis[:, i] for i, name in enumerate(names)}
        out = call(case, state, nt=1)
        if out.info.step_ratio != 1:
            raise RuntimeError(
                f"{case.label()}: the propagator took {out.info.step_ratio} "
                "internal steps per output step, so one call is not one step "
                "of the scheme. Lower the Courant number."
            )
        values = out.as_dict()
        columns = torch.stack([values[name] for name in names], 1).reshape(width, size)
        matrix[:, start:stop] = columns.T.numpy()
        leak = max(leak, excluded_state_size(out, names))
    dead, held, ghost = classify_rows(matrix, case.propagator)
    return Operator(
        case=case,
        names=names,
        matrix=matrix,
        dead=dead,
        held=held,
        ghost=ghost,
        leak=leak,
        seconds=time.perf_counter() - started,
    )


def classify_rows(
    matrix: npt.NDArray[Any], propagator: str
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Sorts the rows of a one-step operator into dead, held and ghost rows.

    Args:
        matrix: The one-step operator.
        propagator: The propagator it belongs to, since the regular-grid family
            has a zero diagonal on half its rows by construction.

    Returns:
        Three boolean masks: the rows that are exactly ``e_i`` (a decoupled
        constant), the rows that are zero (held at zero), and the rows whose
        diagonal is zero although the row is not (a ghost, rewritten from other
        nodes every step).
    """
    size = matrix.shape[0]
    dead = np.all(matrix == np.eye(size), axis=1)
    held = np.asarray(~matrix.any(axis=1))
    if propagator in REGULAR_GRID:
        # The second-order form keeps the previous step as a state field, so
        # every one of those rows has a zero diagonal by construction. A fill
        # that rewrites a node from its neighbours shows up in the residual of
        # L instead.
        ghost = np.zeros(size, dtype=bool)
    else:
        ghost = (np.diag(matrix) == 0) & ~held & ~dead
    return dead, held, ghost


def excluded_state_size(out: Any, names: list[str]) -> float:
    """The largest absolute value in a state field left out of the operator."""
    worst = 0.0
    for name, value in out.as_dict().items():
        if name in names or "receiver" in name:
            continue
        if name == "auxiliary_state":
            for tensor in value.values():
                if tensor.numel():
                    worst = max(worst, float(tensor.abs().max()))
        elif isinstance(value, torch.Tensor) and value.numel():
            worst = max(worst, float(value.abs().max()))
    return worst


# ---------------------------------------------------------------------------
# The energy metric
# ---------------------------------------------------------------------------


@dataclass
class Metric:
    """The uniform-weight energy metric of a case.

    Attributes:
        velocity: ``G_v`` as a length-``size`` vector, zero off the velocity
            fields. For the regular-grid family it is ``1 / (c^2 dt^2)`` on
            the wavefield entries instead.
        stress: ``G_sigma`` as a ``size`` by ``size`` matrix, zero off the
            stress fields and block-diagonal within them.
        velocity_names: The fields ``velocity`` covers.
        stress_names: The fields ``stress`` covers.
    """

    velocity: npt.NDArray[Any]
    stress: npt.NDArray[Any]
    velocity_names: list[str] = field(default_factory=list)
    stress_names: list[str] = field(default_factory=list)
    coupled: list[list[str]] = field(default_factory=list)


def _stagger_density(rho: npt.NDArray[Any], axis: int) -> npt.NDArray[Any]:
    """The arithmetic mean of neighbouring cell densities along ``axis``.

    The outermost half-integer node has no neighbour beyond the box; the
    kernels' model padding replicates the edge cell there, so the mean is the
    edge value itself. Where the mean is zero the node is a vacuum node and
    carries no state.
    """
    lower = np.take(rho, range(rho.shape[axis] - 1), axis=axis)
    upper = np.take(rho, range(1, rho.shape[axis]), axis=axis)
    mean = (lower + upper) / 2
    edge = np.take(rho, [rho.shape[axis] - 1], axis=axis)
    return np.concatenate([mean, edge], axis=axis)


def _harmonic_corner(mu: npt.NDArray[Any], axes: tuple[int, int]) -> npt.NDArray[Any]:
    """The harmonic mean of the four cells around a staggered shear node.

    Any vacuum cell among the four sends the average to zero, exactly as the
    kernels' improved-vacuum averaging does. The outermost row along either
    axis has no such node; the edge value is used there and the node turns out
    to carry no state.
    """
    slices: list[Any] = [slice(None)] * mu.ndim
    out = np.array(mu, dtype=float, copy=True)
    first, second = axes
    lo_first = [slice(None)] * mu.ndim
    lo_first[first] = slice(None, -1)
    hi_first = [slice(None)] * mu.ndim
    hi_first[first] = slice(1, None)
    corners = []
    for outer in (lo_first, hi_first):
        for inner_start, inner_stop in ((None, -1), (1, None)):
            key = list(outer)
            key[second] = slice(inner_start, inner_stop)
            corners.append(mu[tuple(key)])
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = sum(1 / np.where(corner > 0, corner, np.inf) for corner in corners)
        value = np.where(inverse > 0, 4 / np.where(inverse > 0, inverse, 1), 0.0)
    positive = np.logical_and.reduce([corner > 0 for corner in corners])
    value = np.where(positive, value, 0.0)
    slices[first] = slice(None, -1)
    slices[second] = slice(None, -1)
    out[tuple(slices)] = value
    return out


def _block_inverse(blocks: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """The pseudo-inverse of each per-node constitutive matrix."""
    return np.linalg.pinv(blocks, hermitian=True)


def build_metric(operator: Operator) -> Metric:
    """Builds the uniform-weight metric of ``operator``'s case.

    Raises:
        NotImplementedError: If the propagator's metric is not implemented
            here.
    """
    case = operator.case
    medium = case.medium
    size = operator.size
    n_cells = operator.n_cells
    velocity = np.zeros(size)
    stress = np.zeros((size, size))
    metric = Metric(velocity, stress)
    if case.propagator in REGULAR_GRID:
        c2 = (medium.vp**2).numpy().ravel()
        wavefield = operator.slice_of("wavefield_0")
        with np.errstate(divide="ignore"):
            weight = np.where(c2 > 0, 1 / np.where(c2 > 0, c2, 1), 0.0) / case.dt**2
        velocity[wavefield] = weight
        metric.velocity_names = ["wavefield_0"]
        return metric
    rho = medium.rho.numpy()
    axis_names = "zyx"[3 - case.ndim :]
    for axis, name in enumerate(axis_names):
        staggered = _stagger_density(rho, axis)
        velocity[operator.slice_of(f"v{name}_0")] = staggered.ravel()
        metric.velocity_names.append(f"v{name}_0")
    normal, shear = _stress_blocks(operator)
    for names, blocks in normal:
        indices = np.array([operator.slice_of(name).start for name in names], dtype=int)
        inverse = _block_inverse(blocks)
        for cell in range(n_cells):
            rows = indices + cell
            stress[np.ix_(rows, rows)] = inverse[cell]
        metric.stress_names.extend(names)
        if len(names) > 1:
            metric.coupled.append(list(names))
    for name, modulus in shear:
        block = operator.slice_of(name)
        flat = modulus.ravel()
        with np.errstate(divide="ignore"):
            stress[block, block] = np.diag(
                np.where(flat > 0, 1 / np.where(flat > 0, flat, 1), 0.0)
            )
        metric.stress_names.append(name)
    return metric


def _stress_blocks(
    operator: Operator,
) -> tuple[
    list[tuple[list[str], npt.NDArray[Any]]], list[tuple[str, npt.NDArray[Any]]]
]:
    """The constitutive matrices of the stress nodes.

    Returns:
        A pair of

        * a list of ``(field names, per-cell matrices)`` for the groups of
          co-located stress components that share a constitutive matrix, and
        * a list of ``(field name, per-cell modulus)`` for the staggered shear
          stresses, each counted once.
    """
    case = operator.case
    medium = case.medium
    ndim = case.ndim
    rho, vp, vs = medium.rho.numpy(), medium.vp.numpy(), medium.vs.numpy()
    n_cells = operator.n_cells
    propagator = case.propagator
    if propagator in ("acoustic", "acoustic_born", "viscoacoustic"):
        bulk = (rho * vp**2).ravel()
        return [(["pressure_0"], bulk.reshape(n_cells, 1, 1))], []
    stiff = {k: v.numpy() for k, v in vti_stiffnesses(*medium_tensors(medium)).items()}
    if propagator == "acoustic_vti":
        c11, c13, c33 = stiff["c11"], stiff["c13_acoustic"], stiff["c33"]
        blocks = np.zeros((n_cells, 2, 2))
        blocks[:, 0, 0] = c33.ravel()
        blocks[:, 1, 1] = c11.ravel()
        blocks[:, 0, 1] = blocks[:, 1, 0] = c13.ravel()
        return [(["sigma_v_0", "sigma_h_0"], blocks)], []
    if propagator in (
        "elastic",
        "elastic_born",
    ):
        lamb = rho * (vp**2 - 2 * vs**2)
        mu = rho * vs**2
        blocks = np.zeros((n_cells, ndim, ndim))
        for i in range(ndim):
            for j in range(ndim):
                blocks[:, i, j] = lamb.ravel() + (2 * mu.ravel() if i == j else 0.0)
        normal_names = [f"sigma{a}{a}_0" for a in "zyx"[3 - ndim :]]
        shear = [
            (f"sigma{a}{b}_0", _harmonic_corner(mu, axes))
            for a, b, axes in _shear_axes(ndim)
        ]
        return [(normal_names, blocks)], shear
    if propagator in ("elastic_vti",):
        c11, c13, c33, c55 = (
            stiff["c11"],
            stiff["c13"],
            stiff["c33"],
            stiff["c55"],
        )
        blocks = np.zeros((n_cells, ndim, ndim))
        if ndim == 2:
            blocks[:, 0, 0] = c33.ravel()
            blocks[:, 1, 1] = c11.ravel()
            blocks[:, 0, 1] = blocks[:, 1, 0] = c13.ravel()
        else:
            c12 = c11 - 2 * c55
            blocks[:, 0, 0] = c33.ravel()
            blocks[:, 1, 1] = blocks[:, 2, 2] = c11.ravel()
            blocks[:, 0, 1] = blocks[:, 1, 0] = c13.ravel()
            blocks[:, 0, 2] = blocks[:, 2, 0] = c13.ravel()
            blocks[:, 1, 2] = blocks[:, 2, 1] = c12.ravel()
        normal_names = [f"sigma{a}{a}_0" for a in "zyx"[3 - ndim :]]
        shear = [
            (f"sigma{a}{b}_0", _harmonic_corner(c55, axes))
            for a, b, axes in _shear_axes(ndim)
        ]
        return [(normal_names, blocks)], shear
    raise NotImplementedError(f"no energy metric implemented for {propagator}")


def medium_tensors(medium: Medium) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(rho, vp, vs)`` of a medium, for the stiffness helper."""
    return medium.rho, medium.vp, medium.vs


def _shear_axes(ndim: int) -> list[tuple[str, str, tuple[int, int]]]:
    """The staggered shear stresses and the two axes each is staggered on."""
    if ndim == 2:
        return [("x", "y", (0, 1))]
    return [("y", "z", (0, 1)), ("x", "z", (0, 2)), ("x", "y", (1, 2))]


# ---------------------------------------------------------------------------
# Gate 1: adjointness
# ---------------------------------------------------------------------------


@dataclass
class Adjointness:
    """The adjointness residual of one case."""

    case: Case
    residual: float
    interior_residual: float
    structure: float
    rows: list[tuple[str, str, float]]
    columns: list[tuple[str, str, float]]
    dead: int
    held: int
    ghost: int
    leak: float
    seconds: float

    @property
    def analysable(self) -> bool:
        """Whether ``R`` is a verdict on this case at all.

        It is one only where the operator left by the ghost elimination has the
        leapfrog block form the residual is derived from; see
        :func:`leapfrog_residual`. Where it has not, ``R`` is reported but
        decides nothing, and the energy and spectral gates are the verdict.
        """
        return self.structure <= ADJOINT_TOL

    @property
    def passed(self) -> bool:
        """Whether the residual is round-off, on a case ``R`` can decide."""
        return self.analysable and self.residual <= ADJOINT_TOL


def _blame(
    operator: Operator,
    residual: npt.NDArray[Any],
    keys: npt.NDArray[Any],
    reference: float,
    axis: int,
    top: int,
) -> list[tuple[str, str, float]]:
    """The nodes along one axis of ``R`` whose entries are not round-off.

    A nonzero row of ``R`` names a stress node and a nonzero column a velocity
    node, so both are reported: the stress row and the velocity row that should
    be its transpose are equally implicated, and which of the two the fill got
    wrong is not decided by ``R`` alone.
    """
    if reference <= 0 or residual.size == 0:
        return []
    magnitudes = np.abs(residual).max(axis=axis) / reference
    blamed = []
    for position in np.argsort(-magnitudes)[:top]:
        if magnitudes[position] <= ADJOINT_TOL:
            break
        index = int(keys[position])
        blamed.append(
            (
                operator.describe(index),
                operator.face_of(index),
                float(magnitudes[position]),
            )
        )
    return blamed


def leapfrog_residual(operator: Operator, metric: Metric) -> float:
    """How far the ghost-eliminated operator is from the leapfrog block form.

    ``R`` is derived from a step that is the composition of the two half-step
    updates, so that on the live subspace, with the state ordered as velocity
    then stress,

        ``M = [[I, K], [Nm, I + Nm K]]``

    when the velocity half step comes first, or the same with the two roles
    exchanged when the stress half step does. Only then are the off-diagonal
    blocks exactly ``dt B A_v`` and ``dt C A_sigma``, which is what lets ``R``
    be read as a property of the fill.

    A fill that leaves a ghost row can break that form: :meth:`Operator.effective`
    eliminates a ghost by taking its row of ``M`` as an image of the *stored*
    live state, which is the value the ghost holds at the end of the step. If the
    kernel instead consumes the value the ghost held at the start of the step,
    the eliminated operator is not the one the kernel applies, and ``R`` can be
    exactly zero for a scheme that grows geometrically. The distance from the
    nearer of the two block forms detects that: it is round-off when the
    elimination reconstructed the step correctly.

    Returns:
        The relative residual, or zero for the regular-grid family, whose block
        form :func:`_scalar_operator` already asserts exactly.
    """
    if operator.case.propagator in REGULAR_GRID:
        return 0.0
    effective, _ = operator.effective()
    velocity, stress = _velocity_stress_indices(operator, metric)
    if not len(velocity) or not len(stress):
        return 0.0
    to_velocity = effective[np.ix_(velocity, stress)]
    to_stress = effective[np.ix_(stress, velocity)]
    velocity_block = effective[np.ix_(velocity, velocity)]
    stress_block = effective[np.ix_(stress, stress)]
    velocity_identity = np.eye(len(velocity))
    stress_identity = np.eye(len(stress))
    velocity_first = max(
        float(np.abs(velocity_block - velocity_identity).max()),
        float(np.abs(stress_block - (stress_identity + to_stress @ to_velocity)).max()),
    )
    stress_first = max(
        float(np.abs(stress_block - stress_identity).max()),
        float(
            np.abs(velocity_block - (velocity_identity + to_velocity @ to_stress)).max()
        ),
    )
    reference = max(1.0, float(np.abs(effective).max()))
    return min(velocity_first, stress_first) / reference


def adjointness(operator: Operator, metric: Metric, *, top: int = 8) -> Adjointness:
    """Measures ``R = G_sigma M_sigma,v + M_v,sigma^T G_v``.

    For the regular-grid family the residual is ``L - L^T`` instead, with
    ``L`` recovered from the one-step operator.

    ``interior_residual`` is the same quantity restricted to rows and columns
    at least ``accuracy // 2 + 1`` cells from every face. It is a self-test of
    the metric rather than a property of the fill: the interior stencils are
    centred and adjoint whatever the faces do, so a nonzero interior residual
    means the metric this script built is not the kernel's.

    ``structure`` is :func:`leapfrog_residual`, which says whether ``R`` decides
    the case at all.
    """
    started = time.perf_counter()
    case = operator.case
    if case.propagator in REGULAR_GRID:
        left, right, rows_index, columns_index = _scalar_terms(operator, metric)
    else:
        left, right, rows_index, columns_index = _staggered_terms(operator, metric)
    residual_matrix = left - right
    scale = np.abs(residual_matrix).max() if residual_matrix.size else 0.0
    reference = max(float(np.abs(left).max()), float(np.abs(right).max()))
    relative = float(scale / reference) if reference > 0 else float(scale)
    interior = _interior_mask(operator)
    inner = residual_matrix[np.ix_(interior[rows_index], interior[columns_index])]
    interior_relative = (
        float(np.abs(inner).max() / reference) if reference > 0 and inner.size else 0.0
    )
    rows = _blame(operator, residual_matrix, rows_index, reference, 1, top)
    columns = _blame(operator, residual_matrix, columns_index, reference, 0, top)
    return Adjointness(
        case=case,
        residual=relative,
        interior_residual=interior_relative,
        structure=leapfrog_residual(operator, metric),
        rows=rows,
        columns=columns,
        dead=int(operator.dead.sum()),
        held=int(operator.held.sum()),
        ghost=int(operator.ghost.sum()),
        leak=operator.leak,
        seconds=time.perf_counter() - started,
    )


def _velocity_stress_indices(
    operator: Operator, metric: Metric
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Which live entries belong to the velocity and which to the stresses."""
    live = operator.live_indices()
    belongs = np.zeros(operator.size, dtype=bool)
    for name in metric.velocity_names:
        belongs[operator.slice_of(name)] = True
    return np.nonzero(belongs[live])[0], np.nonzero(~belongs[live])[0]


def _staggered_terms(
    operator: Operator, metric: Metric
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """The two terms of ``R``, and the state entries they are indexed by."""
    effective, live = operator.effective()
    velocity, stress = _velocity_stress_indices(operator, metric)
    to_velocity = effective[np.ix_(velocity, stress)]
    to_stress = effective[np.ix_(stress, velocity)]
    g_velocity = metric.velocity[live[velocity]]
    g_stress = metric.stress[np.ix_(live[stress], live[stress])]
    return (
        g_stress @ to_stress,
        -(to_velocity.T * g_velocity),
        live[stress],
        live[velocity],
    )


def _scalar_operator(operator: Operator, metric: Metric) -> npt.NDArray[Any]:
    """``L``, the regular-grid family's spatial operator, recovered from ``M``.

    With the state ``(u^n, u^(n-1))`` the one-step operator is
    ``[[2 I + dt^2 c^2 L, -I], [I, 0]]``, so ``L`` follows from its leading
    block. The structure is asserted, not assumed.
    """
    case = operator.case
    n_cells = operator.n_cells
    matrix = operator.matrix
    lead = matrix[:n_cells, :n_cells]
    if not (
        np.allclose(matrix[:n_cells, n_cells : 2 * n_cells], -np.eye(n_cells))
        and np.allclose(matrix[n_cells : 2 * n_cells, :n_cells], np.eye(n_cells))
    ):
        raise RuntimeError(
            f"{case.label()}: the one-step operator is not in the expected "
            "second-order leapfrog form, so L cannot be recovered from it."
        )
    inverse_c2dt2 = metric.velocity[operator.slice_of("wavefield_0")]
    recovered = (lead - 2 * np.eye(n_cells)) * inverse_c2dt2[:, None]
    return np.asarray(recovered)


def _scalar_terms(
    operator: Operator, metric: Metric
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """``L`` and ``L^T`` for the regular-grid family, on the live nodes.

    A node the fill holds at zero is a Dirichlet node: its row of ``L`` is
    zero, it is not a degree of freedom, and it is dropped from both the rows
    and the columns.
    """
    lap = _scalar_operator(operator, metric)
    alive = ~(operator.dead | operator.held)
    live = np.nonzero(alive[operator.slice_of("wavefield_0")])[0]
    reduced = lap[np.ix_(live, live)]
    return reduced, reduced.T, live, live


def _interior_mask(operator: Operator) -> npt.NDArray[Any]:
    """Whether each state-vector entry is far enough from every face.

    Far enough is half the stencil width plus one cell, which is the deepest
    a face fill can reach.
    """
    case = operator.case
    margin = case.accuracy // 2 + 1
    n = case.n
    axes = np.indices((n,) * case.ndim).reshape(case.ndim, -1)
    inside = np.all((axes >= margin) & (axes < n - margin), axis=0)
    return np.tile(inside, len(operator.names))


# ---------------------------------------------------------------------------
# Gate 2: energy
# ---------------------------------------------------------------------------


@dataclass
class Energy:
    """The energy-gate result of one case."""

    case: Case
    drift: float
    per_step: float
    steps: int
    excluded: int
    initial: float
    seconds: float

    @property
    def passed(self) -> bool:
        """Whether the drift is within the pass condition.

        A case whose initial energy is zero has nothing to conserve -- the
        pulse landed entirely on nodes the sum excludes -- and never counts as
        a pass, however small its drift.
        """
        if self.initial == 0 or not np.isfinite(self.drift):
            return False
        return self.drift <= ENERGY_TOL


def energy_form(operator: Operator, metric: Metric) -> npt.NDArray[Any]:
    """The matrix ``H`` with ``E(x) = x^T H x`` for the stored state ``x``.

    The energy is time-staggered: the velocity half of it pairs the velocity
    at the two half steps either side of the stress's integer step, so it
    reads the velocity one step on, which is ``(M x)``. The regular-grid
    family's energy pairs ``u^n`` with ``u^(n-1)`` in the same way. Dead
    nodes and ghost rows are dropped from the sum by zeroing their weights.
    """
    size = operator.size
    keep = ~(operator.dead | operator.ghost)
    velocity = metric.velocity * keep
    if operator.case.propagator in REGULAR_GRID:
        # E = 1/2 |u^n - u^(n-1)|^2 / (c^2 dt^2) - 1/2 u^n . (L u^(n-1)); both
        # terms are quadratic in the stored state alone, so neither needs the
        # step-on field the velocity-stress form below reads from M.
        n_cells = operator.n_cells
        lap = _scalar_operator(operator, metric)
        lap = lap * keep[:n_cells][:, None] * keep[:n_cells][None, :]
        weight = np.diag(velocity[:n_cells])
        form = np.zeros((size, size))
        form[:n_cells, :n_cells] = 0.5 * weight
        form[n_cells : 2 * n_cells, n_cells : 2 * n_cells] = 0.5 * weight
        form[:n_cells, n_cells : 2 * n_cells] = -0.5 * weight - 0.25 * lap
        form[n_cells : 2 * n_cells, :n_cells] = -0.5 * weight - 0.25 * lap.T
        return form
    stress = metric.stress * keep[:, None] * keep[None, :]
    diagonal = np.diag(velocity)
    form = 0.25 * (operator.matrix.T @ diagonal + diagonal @ operator.matrix)
    return np.asarray(form + 0.5 * stress)


def initial_pulse(operator: Operator) -> dict[str, torch.Tensor]:
    """A smooth pulse to start the energy run from.

    A velocity pulse with zero stress for the velocity-stress propagators, a
    displacement pulse at rest for the regular-grid family. Dead and held
    nodes are zeroed so the pulse lives entirely in the live state.
    """
    case = operator.case
    n = case.n
    shape = (n,) * case.ndim
    grid = np.indices(shape).astype(float)
    centre = [n / 2 + 0.3 * (axis + 1) for axis in range(case.ndim)]
    radius2 = sum((grid[axis] - centre[axis]) ** 2 for axis in range(case.ndim))
    pulse = np.exp(-radius2 / (n / 5.0) ** 2)
    live = ~(operator.dead | operator.held | operator.ghost)
    state: dict[str, torch.Tensor] = {}
    for name in operator.names:
        block = operator.slice_of(name)
        if case.propagator in REGULAR_GRID:
            values = pulse if name in ("wavefield_0", "wavefield_m1") else pulse * 0
        else:
            values = pulse if name.startswith("v") else pulse * 0
        values = values.ravel() * live[block]
        state[name] = torch.tensor(values.reshape(1, *shape), dtype=torch.float64)
    return state


def energy(
    operator: Operator,
    metric: Metric,
    *,
    steps: int = ENERGY_STEPS,
    checkpoints: int = ENERGY_CHECKPOINTS,
    probes: int = DEFECT_PROBES,
) -> Energy:
    """Runs the energy gate on ``operator``'s case.

    ``drift`` is the worst ``|E - E0| / |E0|`` over ``steps`` steps of the
    propagator itself, sampled at ``checkpoints`` points. ``per_step`` is the
    worst relative change of ``E`` in a single application of ``M`` over
    ``probes`` random states, which is the sharper statistic: it does not
    depend on the initial pulse exciting the offending mode.
    """
    started = time.perf_counter()
    form = energy_form(operator, metric)
    per_step = per_step_defect(operator, form, probes=probes)
    state = initial_pulse(operator)
    reference = _energy_of(state, operator, form)
    worst = 0.0
    auxiliary: Any | None = None
    done = 0
    for target in np.linspace(0, steps, checkpoints + 1).astype(int)[1:]:
        span = int(target) - done
        if span <= 0:
            continue
        out = call(operator.case, state, nt=span, auxiliary_state=auxiliary)
        done = int(target)
        values = out.as_dict()
        state = {name: values[name] for name in operator.names}
        auxiliary = out.auxiliary_state
        current = _energy_of(state, operator, form)
        if reference != 0:
            worst = max(worst, abs(current - reference) / abs(reference))
    return Energy(
        case=operator.case,
        drift=worst,
        per_step=per_step,
        steps=steps,
        excluded=int((operator.dead | operator.ghost).sum()),
        initial=reference,
        seconds=time.perf_counter() - started,
    )


def per_step_defect(
    operator: Operator, form: npt.NDArray[Any], *, probes: int = DEFECT_PROBES
) -> float:
    """The worst relative change of ``E`` in one application of ``M``.

    Measured over ``probes`` random states, each first advanced two steps so
    that the time-staggered energy is evaluated on a state the scheme has
    reached. It is the sharper of the two energy statistics: unlike the run from
    the initial pulse it does not depend on that pulse exciting the offending
    mode.
    """
    generator = np.random.default_rng(0)
    live = ~(operator.dead | operator.held)
    worst = 0.0
    for _ in range(probes):
        vector = generator.standard_normal(operator.size) * live
        vector = operator.matrix @ (operator.matrix @ vector)
        before = float(vector @ form @ vector)
        after = operator.matrix @ vector
        value = float(after @ form @ after)
        if before != 0:
            worst = max(worst, abs(value - before) / abs(before))
    return worst


def _energy_of(
    state: dict[str, torch.Tensor], operator: Operator, form: npt.NDArray[Any]
) -> float:
    """``E`` of a state held as the propagator's fields."""
    vector = np.concatenate(
        [state[name][0].detach().numpy().ravel() for name in operator.names]
    )
    return float(vector @ form @ vector)


# ---------------------------------------------------------------------------
# Gate 3: spectral radius
# ---------------------------------------------------------------------------


@dataclass
class Spectral:
    """The spectral-gate result of one case."""

    case: Case
    radius_excess: float
    outside: int
    outside_values: list[float]
    worst: list[float]
    dropped: int
    seconds: float

    @property
    def passed(self) -> bool:
        """Whether the spectral radius is within the pass condition.

        A case that fails it may still fall in the defective-unit-eigenvalue
        band, which needs the other two gates to decide and so is classified by
        :meth:`Result.spectral_class` rather than here.
        """
        return bool(np.isfinite(self.radius_excess)) and (
            self.radius_excess <= SPECTRAL_TOL
        )


def spectral(operator: Operator, metric: Metric) -> Spectral:
    """Measures ``rho(M) - 1`` on the live subspace.

    Dead states are dropped. On nodes with no shear modulus only the isotropic
    normal stress is kept: the deviatoric normal stress there is outside the
    range of the constitutive matrix, so it is not a state the scheme can
    reach, and keeping it adds a spurious defective eigenvalue.

    ``outside`` counts every eigenvalue with ``|lambda| - 1`` above
    ``SPECTRAL_REPORT_TOL`` and ``outside_values`` holds the largest
    ``SPECTRAL_REPORT_LIMIT`` of them, largest first, so that the values and not
    only their number reach the report.
    """
    started = time.perf_counter()
    basis = _live_basis(operator, metric)
    reduced = basis.T @ operator.matrix @ basis
    eigenvalues = np.linalg.eigvals(reduced)
    magnitudes = np.abs(eigenvalues)
    excess = float(magnitudes.max() - 1)
    reported = sorted(
        (float(m - 1) for m in magnitudes if m > 1 + SPECTRAL_REPORT_TOL), reverse=True
    )
    worst = sorted((float(m - 1) for m in magnitudes), reverse=True)[:5]
    return Spectral(
        case=operator.case,
        radius_excess=excess,
        outside=len(reported),
        outside_values=reported[:SPECTRAL_REPORT_LIMIT],
        worst=worst,
        dropped=operator.size - basis.shape[1],
        seconds=time.perf_counter() - started,
    )


def _live_basis(operator: Operator, metric: Metric) -> npt.NDArray[Any]:
    """An orthonormal basis of the subspace the scheme can reach."""
    size = operator.size
    keep = ~operator.dead
    shear_free = np.zeros(size, dtype=bool)
    columns: list[npt.NDArray[Any]] = []
    for normal in metric.coupled:
        starts = [operator.slice_of(name).start for name in normal]
        for cell in range(operator.n_cells):
            rows = [start + cell for start in starts]
            block = metric.stress[np.ix_(rows, rows)]
            values, vectors = np.linalg.eigh(block)
            rank = int((values > 1e-12 * max(values.max(), 1e-300)).sum())
            if rank == len(rows):
                continue
            shear_free[rows] = True
            if not keep[rows].all():
                continue
            for column in range(len(rows) - rank, len(rows)):
                vector = np.zeros(size)
                vector[rows] = vectors[:, column]
                columns.append(vector)
    identity = np.eye(size)
    columns.extend(
        identity[:, index]
        for index in range(size)
        if keep[index] and not shear_free[index]
    )
    return np.stack(columns, axis=1) if columns else np.zeros((size, 0))


# ---------------------------------------------------------------------------
# Non-uniform weights
# ---------------------------------------------------------------------------


def solve_weights(operator: Operator, metric: Metric) -> npt.NDArray[Any] | None:
    """Solves for a diagonal weighting that makes the fill adjoint.

    The condition ``W_v A_v = (W_sigma A_sigma)^T`` is linear in the diagonal
    entries of ``W``, and multiplicative in each matched pair of coefficients,
    so it is solved in the logarithm of the weights by least squares over the
    pairs that could match at all. Returns the weights, normalised to a median
    of one, or ``None`` if some pair has coefficients of opposite sign (in
    which case no positive weighting exists).
    """
    if operator.case.propagator in REGULAR_GRID:
        left, right, _, _ = _scalar_terms(operator, metric)
        symmetric = True
    else:
        left, right, _, _ = _staggered_terms(operator, metric)
        symmetric = False
    rows, columns = np.nonzero((np.abs(left) > 0) | (np.abs(right) > 0))
    n_rows, n_columns = left.shape
    offset = 0 if symmetric else n_columns
    unknowns = n_rows if symmetric else n_rows + n_columns
    design = np.zeros((len(rows), unknowns))
    target = np.zeros(len(rows))
    usable = np.ones(len(rows), dtype=bool)
    for k, (i, j) in enumerate(zip(rows, columns)):
        if left[i, j] == 0 or right[i, j] == 0 or left[i, j] * right[i, j] <= 0:
            usable[k] = False
            continue
        design[k, offset + i] = 1
        design[k, j] = -1
        target[k] = math.log(abs(right[i, j] / left[i, j]))
    if not usable.any():
        return None
    solution = np.linalg.lstsq(design[usable], target[usable], rcond=None)[0]
    weights = np.exp(solution - np.median(solution))
    scaled_left = weights[offset:, None] * left
    scaled_right = right * weights[None, :n_columns]
    if np.abs(scaled_left - scaled_right).max() > ADJOINT_TOL * max(
        1.0, np.abs(scaled_left).max()
    ):
        return None
    return np.asarray(weights)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """Everything measured about one case."""

    case: Case
    adjointness: Adjointness | None = None
    energy: Energy | None = None
    spectral: Spectral | None = None
    weights: tuple[float, float] | None = None
    error: str | None = None

    def line(self) -> str:
        """One line summarising the case."""
        parts = [self.case.label()]
        if self.error is not None:
            parts.append(f"ERROR {self.error}")
            return "  ".join(parts)
        if self.adjointness is not None:
            if self.adjointness.analysable:
                marker = "ok" if self.adjointness.passed else "FAIL"
            else:
                marker = "R provisional (ghost rows: energy/spectral gates decide)"
            parts.append(
                f"|R|={self.adjointness.residual:.1e} ({marker}) "
                f"interior={self.adjointness.interior_residual:.1e} "
                f"structure={self.adjointness.structure:.1e} "
                f"dead={self.adjointness.dead} held={self.adjointness.held} "
                f"ghost={self.adjointness.ghost} leak={self.adjointness.leak:.1e}"
            )
        if self.energy is not None:
            marker = "ok" if self.energy.passed else "FAIL"
            parts.append(
                f"drift={self.energy.drift:.1e} ({marker}) "
                f"per_step={self.energy.per_step:.1e} "
                f"[{self.energy.steps} steps, {self.energy.seconds:.1f}s]"
            )
        if self.spectral is not None:
            classification = self.spectral_class
            if classification == "pass":
                marker = "ok"
            elif classification == "defective":
                marker = "S~ defective unit eigenvalue, informational"
                if self.energy is None:
                    marker += "; energy gate not run, rerun with --energy"
            else:
                marker = "FAIL"
            parts.append(
                f"rho-1={self.spectral.radius_excess:+.1e} ({marker}) "
                f"outside={self.spectral.outside}"
            )
        if self.weights is not None:
            parts.append(f"weights in [{self.weights[0]:.4f},{self.weights[1]:.4f}]")
        return "  ".join(parts)

    @property
    def spectral_class(self) -> str:
        """``"pass"``, ``"defective"`` or ``"FAIL"`` for the spectral gate.

        ``"defective"`` is the ``S~`` class of the module docstring: an excess in
        (``SPECTRAL_TOL``, ``SPECTRAL_DEFECTIVE_CEILING``] on a case whose
        adjointness residual is round-off and whose energy gate passed or was not
        requested. It is read as a defective unit eigenvalue, computed with an
        error of order ``sqrt(eps)``, rather than as growth. Anything larger, and
        any case whose adjointness or energy gate failed, is a failure.
        """
        if self.spectral is None:
            raise ValueError("no spectral result to classify")
        if self.spectral.passed:
            return "pass"
        adjoint_decides = self.adjointness is not None and self.adjointness.passed
        energy_allows = self.energy is None or self.energy.passed
        if (
            adjoint_decides
            and energy_allows
            and np.isfinite(self.spectral.radius_excess)
            and self.spectral.radius_excess <= SPECTRAL_DEFECTIVE_CEILING
        ):
            return "defective"
        return "FAIL"

    @property
    def verdict(self) -> str:
        """``"pass"``, ``"defective"`` or ``"FAIL"`` for the case as a whole.

        ``"defective"`` is a case whose only reservation is the ``S~`` class: it
        is neither a plain pass nor a plain fail. A case whose ``R`` cannot
        decide it (see :attr:`Adjointness.analysable`) is left to the energy and
        spectral gates, and does not pass when neither of them ran.
        """
        if self.error is not None:
            return "FAIL"
        if self.adjointness is not None:
            if self.adjointness.analysable:
                if not self.adjointness.passed:
                    return "FAIL"
            elif self.energy is None and self.spectral is None:
                return "FAIL"
        if self.energy is not None and not self.energy.passed:
            return "FAIL"
        if self.spectral is not None:
            classification = self.spectral_class
            if classification != "pass":
                return "FAIL" if classification == "FAIL" else "defective"
        return "pass"

    @property
    def passed(self) -> bool:
        """Whether every gate that ran passed outright."""
        return self.verdict == "pass"

    @property
    def defective(self) -> bool:
        """Whether the case passed but for the ``S~`` class."""
        return self.verdict == "defective"


def run_case(
    case: Case,
    *,
    gates: tuple[str, ...],
    steps: int,
    checkpoints: int,
    want_weights: bool,
    verbose: bool = True,
) -> Result:
    """Runs the requested gates on one case."""
    result = Result(case=case)
    try:
        operator = build_one_step(case)
        metric = build_metric(operator)
        if "adjointness" in gates:
            result.adjointness = adjointness(operator, metric)
        if "energy" in gates:
            result.energy = energy(
                operator, metric, steps=steps, checkpoints=checkpoints
            )
        if "spectral" in gates:
            result.spectral = spectral(operator, metric)
        if want_weights:
            weights = solve_weights(operator, metric)
            if weights is not None:
                result.weights = (float(weights.min()), float(weights.max()))
    except (NotImplementedError, RuntimeError, ValueError) as exception:
        result.error = f"{type(exception).__name__}: {exception}"
    if verbose:
        print(result.line(), flush=True)
        if result.adjointness is not None:
            for kind, blamed in (
                ("row", result.adjointness.rows),
                ("col", result.adjointness.columns),
            ):
                for name, face, magnitude in blamed:
                    print(
                        f"      R {kind} {name:24s} {face:22s} {magnitude:.2e}",
                        flush=True,
                    )
        if result.spectral is not None:
            for line in spectral_detail(result.spectral):
                print(line, flush=True)
    return result


def spectral_detail(result: Spectral) -> list[str]:
    """The eigenvalues off the unit circle, for the case's detail output.

    Every ``|lambda| - 1`` above ``SPECTRAL_REPORT_TOL`` is reported, largest
    first, up to ``SPECTRAL_REPORT_LIMIT`` values and then the number of the
    rest, so that a saved log holds the values and not only their count.
    """
    if not result.outside:
        return []
    lines = []
    values = result.outside_values
    per_line = 8
    for start in range(0, len(values), per_line):
        chunk = " ".join(f"{value:+.2e}" for value in values[start : start + per_line])
        if start == 0:
            plural = "" if result.outside == 1 else "s"
            lines.append(
                f"      |lambda|-1 > {SPECTRAL_REPORT_TOL:.0e} "
                f"({result.outside} eigenvalue{plural}): {chunk}"
            )
        else:
            lines.append(f"        {chunk}")
    remaining = result.outside - len(values)
    if remaining > 0:
        lines.append(f"        ... and {remaining} more, all smaller")
    return lines


def control_cases(
    propagators: tuple[str, ...],
    accuracies: tuple[int, ...],
    n: int,
    ndim: int,
    courant: float,
    convention: str,
) -> list[Case]:
    """The cases that must conserve, and so validate the instrument."""
    cases: list[Case] = []

    def case(propagator: str, kind: str, code: str, accuracy: int) -> Case:
        medium = make_medium(ndim, n, kind)
        return Case(
            propagator,
            medium,
            (code,) * (2 * ndim),
            accuracy,
            courant,
            convention=convention,
        )

    for propagator in propagators:
        cases.extend(
            case(propagator, kind, "pml", accuracy)
            for accuracy in accuracies
            for kind in ("homogeneous", "heterogeneous")
        )
        if propagator in ("acoustic", "acoustic_born", "viscoacoustic"):
            cases.extend(
                case(propagator, "homogeneous", "rigid", accuracy)
                for accuracy in accuracies
            )
        if propagator in (
            "elastic",
            "elastic_born",
        ):
            cases.append(case(propagator, "homogeneous", "rigid", 2))
        if propagator in ("elastic", "elastic_vti", "elastic_born"):
            cases.extend(
                case(propagator, "vacuum", "pml", accuracy) for accuracy in accuracies
            )
    return cases


def spectral_media(ndim: int, n: int, level: str) -> list[tuple[Medium, float]]:
    """The (medium, Courant number) pairs of the spectral gate."""
    if level == "full":
        ratios = [0.0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        courants = [0.05, 0.1, 0.2, 0.4]
        seeds = [1, 2, 3]
    elif level == "screen":
        ratios = [0.0, 0.3, 0.5, 0.7]
        courants = [0.1, 0.4]
        seeds = [1]
    else:
        ratios = [0.5]
        courants = [0.2]
        seeds = [1]
    pairs = [
        (make_medium(ndim, n, "homogeneous", vs_vp=ratio), courant)
        for ratio in ratios
        for courant in courants
    ]
    pairs += [(make_medium(ndim, n, "heterogeneous", seed=seed), 0.2) for seed in seeds]
    return pairs


def parse_faces(values: list[str], ndim: int) -> list[tuple[str, ...]]:
    """Turns ``--faces`` values into per-face tuples.

    A single code means that code on every face; a comma-separated list is
    taken face by face, in Deepwave's order (the low then the high face of
    each axis, slowest axis first).
    """
    combinations = []
    for value in values:
        parts = value.split(",")
        if len(parts) == 1:
            combinations.append(tuple(parts * (2 * ndim)))
        elif len(parts) == 2 * ndim:
            combinations.append(tuple(parts))
        else:
            raise ValueError(
                f"--faces {value!r} names {len(parts)} faces; a "
                f"{ndim}-D box has {2 * ndim}"
            )
    return combinations


# ---------------------------------------------------------------------------
# Self-test on synthetic operators
# ---------------------------------------------------------------------------

#: Nodes per field in the synthetic operators of the self-test.
SELFTEST_NODES = 8

#: Above this, a per-step energy defect of a synthetic operator is a real defect
#: rather than round-off. Every defect the self-test's broken operators produce
#: is above 1e-3 and every one its correct operators produce is below 1e-15.
SELFTEST_DEFECT_FLOOR = 1e-6


@dataclass(frozen=True)
class Synthetic:
    """One synthetic operator of the self-test, with the verdicts it must get.

    Attributes:
        name: What the operator is.
        matrix: Its one-step operator.
        metric: The energy metric of the medium it was built on.
        adjoint: Whether ``max|R|`` must be round-off.
        analysable: Whether the leapfrog-form check must accept it, so that
            ``R`` decides the case.
        spectral: Whether the spectral gate must pass it.
        defect: Whether the per-step energy defect must be a real defect.
        weights: ``"identity"`` if ``--solve-weights`` must return uniform
            weights, ``"non-uniform"`` if it must return a position-dependent
            weighting, ``"none"`` if no positive weighting exists.
    """

    name: str
    matrix: npt.NDArray[Any]
    metric: Metric
    adjoint: bool
    analysable: bool
    spectral: bool
    defect: bool
    weights: str


def _synthetic_case() -> Case:
    """The case the synthetic operators stand in for: a 1-D closed box."""
    ones = torch.ones(SELFTEST_NODES, dtype=torch.float64)
    return Case(
        "elastic",
        Medium(ones, ones.clone(), 0.5 * ones, "synthetic"),
        ("rigid", "rigid"),
        2,
    )


def _synthetic_operator(matrix: npt.NDArray[Any]) -> Operator:
    """A synthetic operator, classified exactly as a measured one would be."""
    dead, held, ghost = classify_rows(matrix, "elastic")
    return Operator(
        case=_synthetic_case(),
        names=["vx_0", "sigmaxx_0"],
        matrix=matrix,
        dead=dead,
        held=held,
        ghost=ghost,
        leak=0.0,
        seconds=0.0,
    )


def _synthetic_metric(buoyancy: npt.NDArray[Any], bulk: npt.NDArray[Any]) -> Metric:
    """``G_v = B^-1``, ``G_sigma = C^-1`` for a one-velocity, one-stress box."""
    nodes = SELFTEST_NODES
    velocity = np.zeros(2 * nodes)
    velocity[:nodes] = 1 / buoyancy
    stress = np.zeros((2 * nodes, 2 * nodes))
    stress[nodes:, nodes:] = np.diag(1 / bulk)
    return Metric(velocity, stress, ["vx_0"], ["sigmaxx_0"], [])


def _synthetic_stencils() -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """A staggered forward difference and its exact negative transpose."""
    forward = np.zeros((SELFTEST_NODES, SELFTEST_NODES))
    for node in range(SELFTEST_NODES - 1):
        forward[node, node] = -1.0
        forward[node, node + 1] = 1.0
    return forward, -forward.T


def _synthetic_leapfrog(
    to_velocity: npt.NDArray[Any], to_stress: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """``M`` of a leapfrog step with those two off-diagonal blocks."""
    nodes = SELFTEST_NODES
    matrix = np.zeros((2 * nodes, 2 * nodes))
    matrix[:nodes, :nodes] = np.eye(nodes)
    matrix[:nodes, nodes:] = to_velocity
    matrix[nodes:, :nodes] = to_stress
    matrix[nodes:, nodes:] = np.eye(nodes) + to_stress @ to_velocity
    return matrix


def _synthetic_ghost(
    buoyancy: npt.NDArray[Any],
    bulk: npt.NDArray[Any],
    factor: float,
    *,
    lagged: bool,
    dt: float = 0.05,
) -> npt.NDArray[Any]:
    """``M`` of a box whose first velocity node is a ghost.

    The fill refills the ghost as ``v_0 := (b_1 / b_0) v_1`` and the stress rows
    read it, so eliminating the ghost puts its contribution on to node 1. The
    velocity row at node 1 uses ``factor`` in place of that ratio, which makes
    the fill the exact transpose of the eliminated contribution when ``factor``
    is ``b_1 / b_0`` and not otherwise; on a uniform buoyancy both spellings
    coincide, which is how a fill can be adjoint for constant parameters alone.

    ``lagged`` moves the refill to the end of the step, so the stress update
    reads the value the ghost held a step earlier while the stored value is the
    refilled one. The stored state is then not the state the step consumed, the
    ghost elimination reconstructs the wrong operator, and ``R`` stops being a
    verdict -- which is what :func:`leapfrog_residual` is for.
    """
    nodes = SELFTEST_NODES
    forward, backward = _synthetic_stencils()
    ratio = buoyancy[1] / buoyancy[0]
    velocity_rows = forward.copy()
    velocity_rows[1, 0] = -factor
    velocity_rows[1, 1] = factor - 1.0
    size = 2 * nodes
    velocity_step = np.eye(size)
    velocity_step[:nodes, nodes:] = dt * np.diag(buoyancy) @ velocity_rows
    stress_step = np.eye(size)
    stress_step[nodes:, :nodes] = dt * np.diag(bulk) @ backward
    refill = np.eye(size)
    refill[0] = 0
    refill[0, 1] = ratio
    if lagged:
        return np.asarray(refill @ stress_step @ velocity_step)
    return np.asarray(stress_step @ refill @ velocity_step)


def _synthetic_summation_by_parts() -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """A first-derivative pair that is adjoint only in the weighting ``H``.

    ``Q + Q^T = diag(-1, 0, ..., 0, 1)`` and ``A_v = H^-1 Q``,
    ``A_sigma = -H^-1 Q^T`` with ``H = diag(1/2, 1, ..., 1, 1/2)``: the pair
    conserves the ``H``-weighted energy and no uniformly weighted one, so
    ``--solve-weights`` must recover a position-dependent weighting.
    """
    nodes = SELFTEST_NODES
    weight = np.ones(nodes)
    weight[0] = weight[-1] = 0.5
    difference = np.zeros((nodes, nodes))
    for node in range(nodes):
        if node + 1 < nodes:
            difference[node, node + 1] = 0.5
        if node - 1 >= 0:
            difference[node, node - 1] = -0.5
    difference[0, 0] = -0.5
    difference[-1, -1] = 0.5
    inverse = np.diag(1 / weight)
    return inverse @ difference, -inverse @ difference.T


def _synthetic_systems() -> list[Synthetic]:
    """The synthetic operators, each with the verdict its construction implies."""
    nodes = SELFTEST_NODES
    uniform_buoyancy = np.ones(nodes)
    graded_buoyancy = 1.0 + 0.5 * np.arange(nodes) / nodes
    bulk = np.ones(nodes)
    forward, backward = _synthetic_stencils()
    dt = 0.2
    uniform = _synthetic_metric(uniform_buoyancy, bulk)
    to_velocity = dt * np.diag(uniform_buoyancy) @ forward
    to_stress = dt * np.diag(bulk) @ backward
    systems = [
        Synthetic(
            "exactly adjoint fill",
            _synthetic_leapfrog(to_velocity, to_stress),
            uniform,
            adjoint=True,
            analysable=True,
            spectral=True,
            defect=False,
            weights="identity",
        )
    ]
    wrong = backward.copy()
    wrong[2, 2] += 0.3
    systems.append(
        Synthetic(
            "one wrong stencil coefficient",
            _synthetic_leapfrog(to_velocity, dt * np.diag(bulk) @ wrong),
            uniform,
            adjoint=False,
            analysable=True,
            spectral=True,
            defect=True,
            weights="non-uniform",
        )
    )
    for lagged in (False, True):
        when = "stored a step behind" if lagged else "refilled inside the step"
        for medium, buoyancy in (
            ("uniform B", uniform_buoyancy),
            ("graded B", graded_buoyancy),
        ):
            uniform_medium = medium == "uniform B"
            metric = _synthetic_metric(buoyancy, bulk)
            for spelling, factor in (
                ("ratio the wrong way round", buoyancy[0] / buoyancy[1]),
                ("exact transpose", buoyancy[1] / buoyancy[0]),
            ):
                adjoint = uniform_medium or spelling == "exact transpose"
                systems.append(
                    Synthetic(
                        f"ghost fill, {spelling}, {medium}, {when}",
                        _synthetic_ghost(buoyancy, bulk, factor, lagged=lagged),
                        metric,
                        adjoint=adjoint,
                        analysable=not lagged,
                        spectral=not lagged,
                        defect=lagged or not adjoint,
                        weights="identity" if adjoint else "none",
                    )
                )
    # A rank-one perturbation of the velocity block that the stress block
    # annihilates: every eigenvalue of M is untouched, so the operator is
    # spectrally stable while it is not adjoint and grows linearly.
    direction = np.zeros(nodes)
    direction[1] = 0.5
    systems.append(
        Synthetic(
            "non-normal, spectrally stable",
            _synthetic_leapfrog(to_velocity + np.outer(direction, 1 / bulk), to_stress),
            uniform,
            adjoint=False,
            analysable=True,
            spectral=True,
            defect=True,
            weights="none",
        )
    )
    sbp_velocity, sbp_stress = _synthetic_summation_by_parts()
    systems.append(
        Synthetic(
            "adjoint only in a position-dependent weighting",
            _synthetic_leapfrog(
                0.1 * np.diag(uniform_buoyancy) @ sbp_velocity,
                0.1 * np.diag(bulk) @ sbp_stress,
            ),
            uniform,
            adjoint=False,
            analysable=True,
            spectral=True,
            defect=True,
            weights="non-uniform",
        )
    )
    return systems


def _weights_kind(weights: npt.NDArray[Any] | None) -> str:
    """Whether a solved weighting is uniform, position-dependent, or absent."""
    if weights is None:
        return "none"
    smallest = float(weights.min())
    if smallest <= 0:
        return "none"
    spread = float(weights.max()) / smallest
    return "identity" if spread <= 1 + 1e-9 else "non-uniform"


def selftest(*, verbose: bool = True) -> int:
    """Runs the gate functions on synthetic operators of known verdict.

    Each operator is a small one-dimensional leapfrog step built so that the
    verdict of every gate follows from its construction: an exactly adjoint
    fill, a single wrong coefficient, a ghost fill that is adjoint for a uniform
    buoyancy alone, the same ghost fill with the ghost stored a step behind what
    the step consumed, a non-normal but spectrally stable operator, and a pair
    that conserves only a position-dependent energy. A mismatch is a defect in
    this script, not in a kernel.

    Returns:
        0 if every verdict matched, 1 otherwise.
    """
    systems = _synthetic_systems()
    wrong_systems = 0
    for system in systems:
        operator = _synthetic_operator(system.matrix)
        residual = adjointness(operator, system.metric)
        spectrum = spectral(operator, system.metric)
        defect = per_step_defect(operator, energy_form(operator, system.metric))
        weights = solve_weights(operator, system.metric)
        on_r_alone = Result(case=operator.case, adjointness=residual)
        measured = {
            "R round-off": residual.residual <= ADJOINT_TOL,
            "R decides the case": residual.analysable,
            "spectral gate passes": spectrum.passed,
            "per-step energy defect": defect > SELFTEST_DEFECT_FLOOR,
            "solved weights": _weights_kind(weights),
            "passes on the adjointness gate alone": on_r_alone.passed,
        }
        expected = {
            "R round-off": system.adjoint,
            "R decides the case": system.analysable,
            "spectral gate passes": system.spectral,
            "per-step energy defect": system.defect,
            "solved weights": system.weights,
            "passes on the adjointness gate alone": system.adjoint
            and system.analysable,
        }
        mismatched = [key for key in expected if measured[key] != expected[key]]
        wrong_systems += bool(mismatched)
        if verbose:
            print(
                f"  {'FAIL' if mismatched else 'ok  '} {system.name}: "
                f"|R|={residual.residual:.2e} structure={residual.structure:.2e} "
                f"rho-1={spectrum.radius_excess:+.2e} defect={defect:.2e} "
                f"weights={measured['solved weights']}",
                flush=True,
            )
            for key in mismatched:
                print(
                    f"        {key}: expected {expected[key]!r}, "
                    f"measured {measured[key]!r}",
                    flush=True,
                )
    print(
        f"\n{len(systems) - wrong_systems}/{len(systems)} synthetic operators gave "
        "the verdicts their construction implies",
        flush=True,
    )
    return 1 if wrong_systems else 0


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether a propagator's boundary face fills conserve a "
            "discrete energy, are adjoint, and keep the one-step operator's "
            "spectral radius at one."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes (any combination; with none of them, all three gates "
            "run):\n"
            "  --adjointness   the R = G_sigma M_sigma,v + M_v,sigma^T G_v "
            "residual, and the rows it blames\n"
            "  --energy        the long-run energy drift of the propagator, "
            "and the per-step defect\n"
            "  --spectral      rho(M) - 1 on the live subspace\n"
            "  --controls      all three, on the faces that must conserve "
            "(the instrument's own check)\n"
            "  --selftest      the gates on synthetic operators whose verdict "
            "follows from their construction\n"
            "\nWith no restriction, every face code of every propagator at every "
            "accuracy is measured.\n"
        ),
    )
    parser.add_argument("--adjointness", action="store_true", help="run that gate")
    parser.add_argument("--energy", action="store_true", help="run that gate")
    parser.add_argument("--spectral", action="store_true", help="run that gate")
    parser.add_argument(
        "--controls", action="store_true", help="run the conserving controls"
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help=(
            "check the gates against synthetic operators of known verdict and "
            "exit (no propagator is run, and every case selection is ignored)"
        ),
    )
    parser.add_argument(
        "--propagator",
        action="append",
        default=[],
        choices=list(ALL_PROPAGATORS),
        help="restrict to this propagator (repeatable)",
    )
    parser.add_argument(
        "--faces",
        action="append",
        default=[],
        help=(
            "a face code for every face, or a comma-separated code per face "
            "(repeatable)"
        ),
    )
    parser.add_argument(
        "--accuracy",
        action="append",
        type=int,
        default=[],
        choices=[2, 4, 6, 8],
        help="restrict to this accuracy (repeatable)",
    )
    parser.add_argument("--ndim", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--n", type=int, default=0, help="cells per axis (0: default)")
    parser.add_argument(
        "--medium",
        action="append",
        default=[],
        choices=["homogeneous", "heterogeneous", "vacuum", "fluid"],
        help="restrict to this medium (repeatable)",
    )
    parser.add_argument(
        "--steps", type=int, default=ENERGY_STEPS, help="steps in the energy gate"
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        default=ENERGY_CHECKPOINTS,
        help="times the energy gate samples the run",
    )
    parser.add_argument(
        "--spectral-media",
        choices=["one", "screen", "full"],
        default="one",
        help=(
            "how many media the spectral gate crosses in (a medium, Courant) "
            "sweep; every gate requested runs on each of them. It needs the "
            "spectral gate, so it cannot be combined with --controls, which "
            "fixes the media itself"
        ),
    )
    parser.add_argument(
        "--courant", type=float, default=0.2, help="Courant number of the runs"
    )
    parser.add_argument(
        "--courant-convention",
        choices=["deepwave", "per-axis"],
        default="deepwave",
        help=(
            "what the Courant number means: Deepwave's own "
            "dt v sqrt(sum 1/dx^2), or the per-axis dt v / dx (they differ by "
            "sqrt(ndim) on a cubic grid)"
        ),
    )
    parser.add_argument(
        "--solve-weights",
        action="store_true",
        help="also solve for a position-dependent weighting that would conserve",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="a small grid, two accuracies and a short run, for smoke use",
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="CPU threads (keep small on a busy box)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runs the instrument.

    Returns:
        0 if no case that ran failed. A case in the ``S~`` class is not a
        failure and not a plain pass: it is tallied and listed on its own, and
        leaves the exit code at 0, so the summary and not the exit code carries
        that verdict.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)
    torch.set_num_threads(max(1, arguments.threads))
    if arguments.selftest:
        return selftest()
    ndim = arguments.ndim
    n = arguments.n or (8 if ndim == 3 else 12)
    steps = arguments.steps
    accuracies = tuple(arguments.accuracy) or (2, 4, 6, 8)
    propagators = tuple(arguments.propagator) or ALL_PROPAGATORS
    media = tuple(arguments.medium) or ("homogeneous", "heterogeneous")
    if arguments.quick:
        n = arguments.n or (6 if ndim == 3 else 8)
        accuracies = tuple(arguments.accuracy) or (2, 4)
        steps = min(steps, 200)
        media = tuple(arguments.medium) or ("homogeneous",)
    gates = tuple(
        name
        for name, wanted in (
            ("adjointness", arguments.adjointness),
            ("energy", arguments.energy),
            ("spectral", arguments.spectral),
        )
        if wanted
    ) or ("adjointness", "energy", "spectral")
    convention = arguments.courant_convention
    if arguments.spectral_media != "one" and (
        arguments.controls or "spectral" not in gates
    ):
        parser.error(
            "--spectral-media sweeps the media the spectral gate crosses, so it "
            "needs that gate and its own choice of media: it cannot be combined "
            "with --controls, whose media are the fixed conserving ones. Drop it, "
            "or run it in a separate invocation that includes the spectral gate."
        )
    if arguments.controls:
        cases = control_cases(
            propagators, accuracies, n, ndim, arguments.courant, convention
        )
        gates = ("adjointness", "energy", "spectral")
    else:
        faces = arguments.faces or list(FACE_CODES)
        combinations = parse_faces(faces, ndim)
        cases = []
        for propagator, accuracy, combination, kind in itertools.product(
            propagators, accuracies, combinations, media
        ):
            medium = make_medium(ndim, n, kind)
            cases.append(
                Case(
                    propagator,
                    medium,
                    combination,
                    accuracy,
                    arguments.courant,
                    convention=convention,
                )
            )
        if arguments.spectral_media != "one":
            cases = []
            for propagator, accuracy, combination in itertools.product(
                propagators, accuracies, combinations
            ):
                for medium, courant in spectral_media(
                    ndim, n, arguments.spectral_media
                ):
                    cases.append(
                        Case(
                            propagator,
                            medium,
                            combination,
                            accuracy,
                            courant,
                            convention=convention,
                        )
                    )
    print(
        f"{len(cases)} cases, gates: {', '.join(gates)}, "
        f"{ndim}-D n={n}, float64, {torch.get_num_threads()} threads",
        flush=True,
    )
    results = [
        run_case(
            case,
            gates=gates,
            steps=steps,
            checkpoints=arguments.checkpoints,
            want_weights=arguments.solve_weights,
        )
        for case in cases
    ]
    defective = [result for result in results if result.defective]
    failed = [result for result in results if not (result.passed or result.defective)]
    passed = len(results) - len(defective) - len(failed)
    print(
        f"\n{passed} passed, {len(defective)} passed with S~, {len(failed)} failed",
        flush=True,
    )
    for result in defective:
        print(f"  S~ {result.line()}", flush=True)
    for result in failed:
        print(f"  FAILED {result.line()}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
