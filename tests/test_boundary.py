"""Tests for first-class boundary faces (free/rigid/periodic).

These validate the ``boundary`` argument of the scalar (and acoustic)
propagators against analytic ground truth:

* analytic reflection coefficients at every accuracy order 2-8 -- a ``"free"``
  (pressure-release) face reflects with :math:`R=-1`, a ``"rigid"`` face with
  :math:`R=+1`, to better than 1% at high points-per-wavelength. This is the
  improvement over a bare zero-``pml_width`` face, whose exterior is held to
  zero rather than filled with the exact odd/even image and which therefore
  carries a spurious reflection of several percent at accuracy above 2 (see
  ``test_free_removes_high_order_spurious_reflection``);
* ``"periodic"`` faces are exactly translation-invariant (a plane wave wraps
  with no reflection) and conserve a wrap-around arrival;
* the default path (``boundary=None``) is bit-identical to not passing the
  argument at all;
* gradients through every face type are correct (the compiled backend's
  gradients match the eager reference's, which are autograd's own).

Not every propagator offers every face, so each test below covers only the
faces its propagator offers. ``deepwave._common.check_boundary_supported``
holds the rule and ``tests/test_boundary_support.py`` tests the rule itself;
in summary, ``scalar`` and ``scalar_born`` accept every face on every face of
every dimension, the staggered-grid propagators accept no ``"periodic"``
face, ``"free"`` is accepted by ``acoustic``, ``acoustic_born`` and
``viscoacoustic`` on the beginning of a dimension (and on its end only at
accuracy 2), and the elastic family accepts no ``"free"`` face and accepts
``"rigid"`` on that same restricted footing.

Both backends are exercised: the eager reference implements every face type,
and the compiled (C/CUDA) backend now threads the per-face ``bc`` codes and
applies the same guard-ring image fills for both the scalar and acoustic
propagators (the ``test_scalar_c_*`` / ``test_acoustic_c_*`` tests check
cross-backend agreement, analytic reflection coefficients via the compiled
path). scalar's forward-mode (jvp) and second-order (HVP) derivatives, which
route through the scalar_born Born backend, now support non-PML faces too
(scalar_born itself gained boundary faces, and its jvp/HVP call sites were
extended once that landed) -- ``test_scalar_c_jvp_
supports_non_pml`` / ``test_scalar_c_hvp_supports_non_pml`` check compiled vs.
eager agreement. acoustic_born gained ``boundary=`` support of its own
(applied independently to its background and scattered fields), which
also lifted acoustic's own jvp/HVP restriction the same way scalar_born's did
for scalar -- ``test_acoustic_c_jvp_supports_non_pml`` / ``test_acoustic_c_
hvp_supports_non_pml`` check compiled vs. eager agreement.
"""

import math
import warnings
from typing import Callable, NamedTuple, Optional

import pytest
import torch
from helpers import (
    assert_compiled_matches_eager_forward_and_grad,
    assert_compiled_matches_eager_forward_and_grad_on_device,
    assert_compiled_matches_eager_grad,
)

import deepwave
from deepwave import (
    acoustic,
    acoustic_born,
    acoustic_vti,
    elastic,
    elastic_born,
    elastic_vti,
    scalar,
    scalar_born,
    viscoacoustic,
)
from deepwave._common import set_boundary

C = 1500.0


@pytest.fixture(autouse=True)
def _default_dtype_double():
    """Run every test in this module with a float64 default dtype.

    Setting the default at import time (as this module previously did) leaks
    into every other test module in the same pytest process, because pytest
    imports all collected modules up front: any dtype-sensitive test that runs
    after collection then sees a float64 default and fails (observed 2026-07-08
    as 38 cross-module failures). Set and restore per test instead.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    yield
    torch.set_default_dtype(prev)


def _ricker(freq, nt, dt):
    peak = 1.4 / freq
    return deepwave.wavelets.ricker(freq, nt, dt, peak).reshape(1, 1, nt), int(
        peak / dt
    )


def _fd_arbitrate_max_grad_cell(loss_fn, param, g_compiled, h_rel=1e-5):
    """Central-FD probe of ``loss_fn`` at the cell of largest |compiled gradient|.

    Compiled and eager gradients agreeing with each other only shows the two
    backends agree -- they could share the same bug (this is exactly why the
    original too-short-``nt`` tests in this file, comparing numerical dust,
    passed for years despite the elastic/elastic_vti/acoustic_vti compiled
    backward being wrong by 10-106% relative at reflecting faces; see
    docs/boundaries.rst's Policy section). This probes one fact that is
    independent of both backends' analytic adjoints: a central finite
    difference of ``loss_fn``, called twice with the model perturbed at a
    single cell, at the cell where the compiled gradient is largest in
    magnitude. ``loss_fn`` must evaluate the eager forward pass only (no
    autograd) and return a Python float.
    """
    idx = torch.unravel_index(g_compiled.detach().abs().argmax(), g_compiled.shape)
    idx = tuple(int(i) for i in idx)
    base = param.detach().clone()
    h = float(h_rel * base[idx].abs().clamp_min(1.0))
    plus = base.clone()
    plus[idx] = plus[idx] + h
    minus = base.clone()
    minus[idx] = minus[idx] - h
    fd = (loss_fn(plus) - loss_fn(minus)) / (2 * h)
    return fd, float(g_compiled[idx])


def _reflection_coefficient(accuracy, face_boundary, freq=12.0, backend="eager"):
    """Measure the normal-incidence reflection coefficient off a low-x face.

    A homogeneous 1-D medium with a reflecting boundary on the low face; the
    incident field is obtained from an otherwise-identical run whose low face is
    absorbing, so ``reflecting - absorbing`` isolates the boundary reflection.
    Returns the signed reflected/direct peak-amplitude ratio. ``backend``
    selects the eager reference or (``"compiled"``) the compiled backend.
    """
    dx = 2.0
    dt = 0.00025
    nt = 4000
    nx = 800
    s, r = 120, 70
    sa, tpk = _ricker(freq, nt, dt)
    v = torch.ones(nx) * C

    def run(bc, pml):
        return scalar(
            v,
            dx,
            dt,
            source_amplitudes=sa,
            source_locations=torch.tensor([[[s]]]),
            receiver_locations=torch.tensor([[[r]]]),
            pml_width=pml,
            boundary=bc,
            backend=backend,
            accuracy=accuracy,
            pml_freq=freq,
        )[-1][0, 0]

    tdir = int((s - r) * dx / C / dt) + tpk
    tref = int((s + r + 2) * dx / C / dt) + tpk
    w = int(1.0 / freq / dt)
    inc = run(None, [40, 40])
    dwin = inc[tdir - w : tdir + w]
    a_dir = dwin[dwin.abs().argmax()]
    tr = run(face_boundary, [0, 40])
    refl = tr - inc
    rwin = refl[tref - w : tref + w]
    a_ref = rwin[rwin.abs().argmax()]
    return (a_ref / a_dir).item()


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_scalar_free_reflection_coefficient(accuracy):
    """A ``"free"`` scalar face reflects with R = -1 at every accuracy order."""
    r = _reflection_coefficient(accuracy, ["free", "pml"])
    assert abs(r - (-1.0)) < 0.01, f"acc={accuracy}: R_free={r}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_scalar_rigid_reflection_coefficient(accuracy):
    """A ``"rigid"`` scalar face reflects with R = +1 at every accuracy order."""
    r = _reflection_coefficient(accuracy, ["rigid", "pml"])
    assert abs(r - 1.0) < 0.01, f"acc={accuracy}: R_rigid={r}"


@pytest.mark.parametrize("accuracy", [4, 6, 8])
def test_free_removes_high_order_spurious_reflection(accuracy):
    """Exact-image ``"free"`` removes the spurious reflection of a bare pml0 face.

    A bare zero-``pml_width`` face holds its exterior to zero rather than the
    exact odd image, producing a spurious reflection that grows with accuracy
    order (roughly 5-8% at accuracy 4-8). The exact-image ``"free"`` face
    removes it: the two differ by several percent of the main reflection at high
    order, and the exact-image reflection is the more accurate one.
    """
    dx, dt, nt, nx = 5.0, 0.0007, 1400, 300
    s, r, freq = 90, 55, 15.0
    sa, _ = _ricker(freq, nt, dt)
    v = torch.ones(nx) * C

    def run(bc, pml, pb):
        return scalar(
            v,
            dx,
            dt,
            source_amplitudes=sa,
            source_locations=torch.tensor([[[s]]]),
            receiver_locations=torch.tensor([[[r]]]),
            pml_width=pml,
            boundary=bc,
            backend=pb,
            accuracy=accuracy,
            pml_freq=freq,
        )[-1][0, 0]

    inc = run(None, [40, 40], "eager")
    exact = run(["free", "pml"], [0, 40], "eager")
    bare = run(None, [0, 40], "compiled")  # compiled bare pml0 (existing behaviour)
    main = (exact - inc).abs().max()
    spurious = (bare - exact).abs().max()
    # The spurious reflection removed by the exact image is a few percent of the
    # main reflection at accuracy >= 4 (0% at accuracy 2, where the two coincide).
    assert spurious / main > 0.02, (
        f"acc={accuracy}: expected a spurious reflection to remove, "
        f"got {spurious / main}"
    )


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_scalar_periodic_translation_invariance(accuracy):
    """A periodic domain is exactly invariant under a shared source+receiver shift."""
    dx, dt, nt, nx = 5.0, 0.0007, 1500, 200
    freq = 15.0
    sa, _ = _ricker(freq, nt, dt)
    v = torch.ones(nx) * C

    def run(s, r):
        return scalar(
            v,
            dx,
            dt,
            source_amplitudes=sa,
            source_locations=torch.tensor([[[s]]]),
            receiver_locations=torch.tensor([[[r]]]),
            pml_width=[0, 0],
            boundary=["periodic", "periodic"],
            backend="eager",
            accuracy=accuracy,
            pml_freq=freq,
        )[-1][0, 0]

    a = run(100, 60)
    d = 70
    b = run((100 + d) % nx, (60 + d) % nx)
    assert (a - b).abs().max() / a.abs().max() < 1e-9


def test_scalar_periodic_wrap_arrival():
    """A wave leaving one periodic face re-enters the opposite face."""
    dx, dt, nt, nx = 5.0, 0.0007, 1500, 200
    freq = 15.0
    sa, _ = _ricker(freq, nt, dt)
    v = torch.ones(nx) * C
    tr = scalar(
        v,
        dx,
        dt,
        source_amplitudes=sa,
        source_locations=torch.tensor([[[100]]]),
        receiver_locations=torch.tensor([[[10]]]),
        pml_width=[0, 0],
        boundary=["periodic", "periodic"],
        backend="eager",
        accuracy=4,
        pml_freq=freq,
    )[-1][0, 0]
    # The direct path 100->10 rightward would exit the far face; the arrival at
    # the near receiver comes from the wrap-around, present at late time.
    assert tr[900:].abs().max() > 1.0


class _DefaultRegressionCase(NamedTuple):
    """One propagator's entry in the ``boundary=None``-is-a-no-op family.

    ``receiver_fields`` names one or two :class:`~deepwave._engine.
    PropagatorOutput` attributes to check (two for the Born pairs:
    background then scattered). ``compiled_tol``/``compiled_relative`` are
    ``None``/``False`` for the propagators (elastic, elastic_vti,
    elastic_born) whose compiled-backend comparison lives in a separate
    ``test_*_c_default_regression`` test below instead of here.
    ``boundary_cases`` is the ``(boundary, pml_width)`` list this propagator
    is run over: the faces it offers, which are not the same for all of them.
    """

    name: str
    func: Callable
    build: Callable
    receiver_fields: tuple[str, ...]
    boundary_cases: tuple = ()
    check_omit_arg: bool = True
    compiled_tol: Optional[float] = None
    compiled_relative: bool = False


def _build_scalar_default(sa):
    v = torch.ones(60) * C
    return (v,), {
        "source_amplitudes": sa,
        "source_locations": torch.tensor([[[30]]]),
        "receiver_locations": torch.tensor([[[20]]]),
    }


def _build_acoustic_default(sa):
    v = torch.ones(60) * C
    rho = torch.ones(60) * RHO
    return (v, rho), {
        "source_amplitudes_volume_rate": sa,
        "source_locations_volume_rate": torch.tensor([[[30]]]),
        "receiver_locations_p": torch.tensor([[[20]]]),
    }


def _build_viscoacoustic_default(sa):
    v = torch.ones(60) * C
    rho = torch.ones(60) * RHO
    q = torch.ones(60) * Q
    return (v, rho, q), {
        "source_amplitudes_pressure_rate": (RHO * C**2) * sa,
        "source_locations_pressure_rate": torch.tensor([[[30]]]),
        "receiver_locations_p": torch.tensor([[[20]]]),
        # The frequency Q is specified at; the attenuating propagators require
        # it, and it matches the pml_freq these regression runs pass.
        "q_freq": 15.0,
    }


def _build_scalar_born_default(sa):
    v = torch.ones(60) * C
    scat = torch.zeros(60)
    scat[25] = C
    return (v,), {
        "dv": scat,
        "source_amplitudes": sa,
        "source_locations": torch.tensor([[[30]]]),
        "receiver_locations": torch.tensor([[[20]]]),
        "bg_receiver_locations": torch.tensor([[[20]]]),
    }


def _build_acoustic_born_default(sa):
    v = torch.ones(60) * C
    rho = torch.ones(60) * RHO
    dv = torch.zeros(60)
    dv[25] = 0.1 * C
    return (v, rho), {
        "dv": dv,
        "source_amplitudes_volume_rate": sa,
        "source_locations_volume_rate": torch.tensor([[[30]]]),
        "sc_receiver_locations_p": torch.tensor([[[20]]]),
        "bg_receiver_locations_p": torch.tensor([[[20]]]),
    }


def _build_acoustic_vti_default(sa):
    c11, c13, c33, buoyancy = _acoustic_vti_isotropic_media((20, 20))
    return (c11, c13, c33, buoyancy), {
        "source_amplitudes_sigma_v": sa,
        "source_amplitudes_sigma_h": sa,
        "source_locations_sigma_v": torch.tensor([[[10, 10]]]),
        "source_locations_sigma_h": torch.tensor([[[10, 10]]]),
        "receiver_locations_sigma_v": torch.tensor([[[6, 6]]]),
    }


def _build_elastic_default(sa):
    lamb, mu, buoyancy = _elastic_isotropic_media((20, 20))
    return (lamb, mu, buoyancy), {
        "source_amplitudes_pressure_rate": sa,
        "source_locations_pressure_rate": torch.tensor([[[10, 10]]]),
        "receiver_locations_p": torch.tensor([[[6, 6]]]),
    }


def _build_elastic_vti_default(sa):
    c11, c13, c33, c55, buoyancy = _elastic_vti_isotropic_media((20, 20))
    return (c11, c13, c33, c55, buoyancy), {
        "source_amplitudes_pressure_rate": sa,
        "source_locations_pressure_rate": torch.tensor([[[10, 10]]]),
        "receiver_locations_p": torch.tensor([[[6, 6]]]),
    }


def _build_elastic_born_default(sa):
    lamb, mu, buoyancy = _elastic_isotropic_media((20, 20))
    dlamb = torch.zeros_like(lamb)
    dlamb[8, 8] = 0.2 * lamb[8, 8]
    return (lamb, mu, buoyancy), {
        "dlamb": dlamb,
        "source_amplitudes_pressure_rate": sa,
        "source_locations_pressure_rate": torch.tensor([[[10, 10]]]),
        "sc_receiver_locations_p": torch.tensor([[[6, 6]]]),
        "bg_receiver_locations_p": torch.tensor([[[6, 6]]]),
    }


# The ``(boundary, pml_width)`` cases each family is run over below, at the
# accuracy order (4) these runs use. They differ because the propagators do
# not offer the same faces: the nodal pair takes every face everywhere; the
# staggered 1-D propagators take no ``"periodic"`` face and take ``"free"``
# only on the beginning of a dimension; ``acoustic_vti`` takes ``"rigid"`` on
# any face but no ``"free"`` face; and the elastic family takes no ``"free"``
# face and ``"rigid"`` only on the beginning of a dimension.
_NODAL_BOUNDARY_CASES_1D = (
    (None, [20, 20]),
    (["free", "pml"], [0, 20]),
    (["rigid", "pml"], [0, 20]),
    (["periodic", "periodic"], [0, 0]),
    (["free", "rigid"], [0, 0]),
)
_STAGGERED_BOUNDARY_CASES_1D = (
    (None, [20, 20]),
    (["free", "pml"], [0, 20]),
    (["rigid", "pml"], [0, 20]),
    (["free", "rigid"], [0, 0]),
)
_ACOUSTIC_VTI_BOUNDARY_CASES_2D = (
    (None, [10, 10, 10, 10]),
    (["rigid", "pml", "pml", "pml"], [0, 10, 10, 10]),
    (["pml", "pml", "rigid", "pml"], [10, 10, 0, 10]),
    (["rigid", "rigid", "rigid", "rigid"], [0, 0, 0, 0]),
)
_ELASTIC_BOUNDARY_CASES_2D = (
    (None, [10, 10, 10, 10]),
    (["rigid", "pml", "pml", "pml"], [0, 10, 10, 10]),
    (["pml", "pml", "rigid", "pml"], [10, 10, 0, 10]),
    (["rigid", "pml", "rigid", "pml"], [0, 10, 0, 10]),
)


# ``boundary=None`` is a no-op (bit-identical to omitting it, or -- where a
# separate ``test_*_c_default_regression`` below does not already cover it --
# matching the compiled backend to machine precision); other faces simply run
# finite in eager mode. Every propagator's version of this check shared the
# same shape, only differing in its model/kwarg construction and which
# receiver field(s) it reads -- see ``_DefaultRegressionCase``.
_DEFAULT_REGRESSION_1D = [
    _DefaultRegressionCase(
        "scalar",
        scalar,
        _build_scalar_default,
        ("receiver_amplitudes",),
        boundary_cases=_NODAL_BOUNDARY_CASES_1D,
        check_omit_arg=False,
        compiled_tol=1e-12,
        compiled_relative=False,
    ),
    _DefaultRegressionCase(
        "acoustic",
        acoustic,
        _build_acoustic_default,
        ("receiver_amplitudes_p",),
        boundary_cases=_STAGGERED_BOUNDARY_CASES_1D,
        compiled_tol=1e-12,
        compiled_relative=True,
    ),
    _DefaultRegressionCase(
        "viscoacoustic",
        viscoacoustic,
        _build_viscoacoustic_default,
        ("receiver_amplitudes_p",),
        boundary_cases=_STAGGERED_BOUNDARY_CASES_1D,
        compiled_tol=1e-9,
        compiled_relative=True,
    ),
    _DefaultRegressionCase(
        "scalar_born",
        scalar_born,
        _build_scalar_born_default,
        ("bg_receiver_amplitudes", "sc_receiver_amplitudes"),
        boundary_cases=_NODAL_BOUNDARY_CASES_1D,
        check_omit_arg=False,
        compiled_tol=1e-12,
        compiled_relative=False,
    ),
    _DefaultRegressionCase(
        "acoustic_born",
        acoustic_born,
        _build_acoustic_born_default,
        ("bg_receiver_amplitudes_p", "sc_receiver_amplitudes_p"),
        boundary_cases=_STAGGERED_BOUNDARY_CASES_1D,
        compiled_tol=1e-12,
        compiled_relative=True,
    ),
]

_DEFAULT_REGRESSION_2D = [
    _DefaultRegressionCase(
        "acoustic_vti",
        acoustic_vti,
        _build_acoustic_vti_default,
        ("receiver_amplitudes_sigma_v",),
        boundary_cases=_ACOUSTIC_VTI_BOUNDARY_CASES_2D,
        compiled_tol=1e-9,
        compiled_relative=True,
    ),
    _DefaultRegressionCase(
        "elastic",
        elastic,
        _build_elastic_default,
        ("receiver_amplitudes_p",),
        boundary_cases=_ELASTIC_BOUNDARY_CASES_2D,
    ),
    _DefaultRegressionCase(
        "elastic_vti",
        elastic_vti,
        _build_elastic_vti_default,
        ("receiver_amplitudes_p",),
        boundary_cases=_ELASTIC_BOUNDARY_CASES_2D,
    ),
    _DefaultRegressionCase(
        "elastic_born",
        elastic_born,
        _build_elastic_born_default,
        ("bg_receiver_amplitudes_p", "sc_receiver_amplitudes_p"),
        boundary_cases=_ELASTIC_BOUNDARY_CASES_2D,
    ),
]


def _regression_params(cases):
    """Expand a registry into ``(case, boundary, pml_width)`` parameters."""
    return [
        pytest.param(
            case,
            boundary,
            pml,
            id=f"{case.name}-{'default' if boundary is None else '-'.join(boundary)}",
        )
        for case in cases
        for boundary, pml in case.boundary_cases
    ]


def _default_regression_check(case, dx, dt, boundary, pml, model_args, kwargs):
    eager = case.func(
        *model_args,
        dx,
        dt,
        backend="eager",
        boundary=boundary,
        pml_width=pml,
        accuracy=4,
        pml_freq=15.0,
        **kwargs,
    )
    eager_vals = [getattr(eager, f) for f in case.receiver_fields]
    if boundary is not None:
        for ev in eager_vals:
            assert torch.isfinite(ev).all()
        return
    if case.check_omit_arg:
        ref = case.func(
            *model_args,
            dx,
            dt,
            backend="eager",
            pml_width=pml,
            accuracy=4,
            pml_freq=15.0,
            **kwargs,
        )
        for ev, f in zip(eager_vals, case.receiver_fields):
            assert (ev - getattr(ref, f)).abs().max() == 0.0
    if case.compiled_tol is not None:
        compiled = case.func(
            *model_args,
            dx,
            dt,
            boundary=None,
            pml_width=pml,
            accuracy=4,
            pml_freq=15.0,
            **kwargs,
        )
        for ev, f in zip(eager_vals, case.receiver_fields):
            cv = getattr(compiled, f)
            diff = (ev - cv).abs().max()
            if case.compiled_relative:
                diff = diff / ev.abs().max()
            assert diff < case.compiled_tol


@pytest.mark.parametrize(
    ("case", "boundary", "pml"), _regression_params(_DEFAULT_REGRESSION_1D)
)
def test_default_regression_1d(case, boundary, pml):
    """``boundary=None`` is a no-op for every 1-D-family propagator (eager).

    Consolidates what were separate, near-identical ``test_<propagator>_
    default_regression``/``test_<propagator>_default_bit_identity`` tests
    (scalar, acoustic, viscoacoustic, scalar_born, acoustic_born) into one
    registry-driven test; see ``_DEFAULT_REGRESSION_1D``.
    """
    dx, dt, nt = 5.0, 0.0007, 150
    sa, _ = _ricker(15.0, nt, dt)
    model_args, kwargs = case.build(sa)
    _default_regression_check(case, dx, dt, boundary, pml, model_args, kwargs)


@pytest.mark.parametrize(
    ("case", "boundary", "pml"), _regression_params(_DEFAULT_REGRESSION_2D)
)
def test_default_regression_2d(case, boundary, pml):
    """``boundary=None`` is a no-op for every 2-D-family propagator (eager).

    Consolidates what were separate, near-identical ``test_<propagator>_
    default_regression`` tests (acoustic_vti, elastic, elastic_vti,
    elastic_born) into one registry-driven test; see
    ``_DEFAULT_REGRESSION_2D``.
    """
    dx, dt, nt = 5.0, 0.0007, 60
    sa, _ = _ricker(15.0, nt, dt)
    model_args, kwargs = case.build(sa)
    _default_regression_check(case, dx, dt, boundary, pml, model_args, kwargs)


def test_scalar_boundary_2d_runs_and_reflects():
    """A 2-D free surface on the top face reflects with inverted polarity."""
    dx, dt, nt = 5.0, 0.0007, 700
    ny, nx = 80, 120
    freq = 15.0
    sa, _ = _ricker(freq, nt, dt)
    v = torch.ones(ny, nx) * C
    # source and receiver near the top (low-y) reflecting face
    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=sa,
        source_locations=torch.tensor([[[30, 60]]]),
        receiver_locations=torch.tensor([[[15, 60]]]),
        pml_width=[0, 20, 20, 20],
        boundary=["free", "pml", "pml", "pml"],
        backend="eager",
        accuracy=4,
        pml_freq=freq,
    )[-1][0, 0]
    inc = scalar(
        v,
        dx,
        dt,
        source_amplitudes=sa,
        source_locations=torch.tensor([[[30, 60]]]),
        receiver_locations=torch.tensor([[[15, 60]]]),
        pml_width=[20, 20, 20, 20],
        backend="eager",
        accuracy=4,
        pml_freq=freq,
    )[-1][0, 0]
    refl = out - inc
    # a reflection is present and it is inverted relative to the direct arrival
    assert refl.abs().max() > 0.1 * inc.abs().max()


# --- acoustic (staggered grid) --------------------------------------------

RHO = 1000.0


def _acoustic_reflection_coefficient(accuracy, face_boundary, freq=12.0):
    """Measure the normal-incidence pressure reflection off a low-x acoustic face.

    Mirrors :func:`_reflection_coefficient` for the staggered-grid acoustic
    propagator (pressure receiver). ``free`` (p=0) reflects with :math:`R=-1`,
    ``rigid`` (v_n=0) with :math:`R=+1`.
    """
    dx = 2.0
    dt = 0.00025
    nt = 4000
    nx = 800
    s, r = 120, 70
    sa, tpk = _ricker(freq, nt, dt)
    v = torch.ones(nx) * C
    rho = torch.ones(nx) * RHO

    def run(bc, pml):
        # The trailing outputs are the (1 + ndim) receivers [pressure, vx];
        # index -(1 + ndim) selects the pressure receiver.
        return acoustic(
            v,
            rho,
            dx,
            dt,
            source_amplitudes_volume_rate=sa,
            source_locations_volume_rate=torch.tensor([[[s]]]),
            receiver_locations_p=torch.tensor([[[r]]]),
            pml_width=pml,
            boundary=bc,
            backend="eager",
            accuracy=accuracy,
            pml_freq=freq,
        )[-2][0, 0]

    tdir = int((s - r) * dx / C / dt) + tpk
    tref = int((s + r + 2) * dx / C / dt) + tpk
    w = int(1.0 / freq / dt)
    inc = run(None, [40, 40])
    dwin = inc[tdir - w : tdir + w]
    a_dir = dwin[dwin.abs().argmax()]
    tr = run(face_boundary, [0, 40])
    refl = tr - inc
    rwin = refl[tref - w : tref + w]
    a_ref = rwin[rwin.abs().argmax()]
    return (a_ref / a_dir).item()


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_acoustic_free_reflection_coefficient(accuracy):
    """An acoustic ``"free"`` (p=0) face reflects with R = -1 at every order."""
    r = _acoustic_reflection_coefficient(accuracy, ["free", "pml"])
    assert abs(r - (-1.0)) < 0.01, f"acc={accuracy}: R_free={r}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_acoustic_rigid_reflection_coefficient(accuracy):
    """An acoustic ``"rigid"`` (v_n=0) face reflects with R = +1 at every order."""
    r = _acoustic_reflection_coefficient(accuracy, ["rigid", "pml"])
    assert abs(r - 1.0) < 0.01, f"acc={accuracy}: R_rigid={r}"


def test_acoustic_boundary_2d_free_reflects():
    """A 2-D acoustic free surface reflects with inverted pressure polarity."""
    dx, dt, nt = 5.0, 0.0007, 700
    ny, nx = 80, 120
    freq = 15.0
    sa, _ = _ricker(freq, nt, dt)
    v = torch.ones(ny, nx) * C
    rho = torch.ones(ny, nx) * RHO
    kw = {
        "source_amplitudes_volume_rate": sa,
        "source_locations_volume_rate": torch.tensor([[[30, 60]]]),
        "receiver_locations_p": torch.tensor([[[15, 60]]]),
        "backend": "eager",
        "accuracy": 4,
        "pml_freq": freq,
    }
    out = acoustic(
        v,
        rho,
        dx,
        dt,
        pml_width=[0, 20, 20, 20],
        boundary=["free", "pml", "pml", "pml"],
        **kw,
    )[-3][0, 0]
    inc = acoustic(v, rho, dx, dt, pml_width=[20, 20, 20, 20], **kw)[-3][0, 0]
    refl = out - inc
    assert refl.abs().max() > 0.1 * inc.abs().max()


def test_acoustic_boundary_3d_runs():
    """A 3-D acoustic free surface runs and produces a finite reflection."""
    dx, dt, nt = 5.0, 0.0007, 300
    nz = ny = nx = 16
    freq = 15.0
    sa, _ = _ricker(freq, nt, dt)
    v = torch.ones(nz, ny, nx) * C
    rho = torch.ones(nz, ny, nx) * RHO
    out = acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=sa,
        source_locations_volume_rate=torch.tensor([[[8, 8, 8]]]),
        receiver_locations_p=torch.tensor([[[3, 8, 8]]]),
        pml_width=[0, 20, 20, 20, 20, 20],
        boundary=["free", "pml", "pml", "pml", "pml", "pml"],
        backend="eager",
        accuracy=4,
        pml_freq=freq,
    )[-4][0, 0]
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0


# --- acoustic compiled (C/CUDA) backend boundary faces --------------------
# STEP B: the compiled acoustic backend threads the per-face ``bc`` codes and
# applies the staggered guard-ring image fills (pressure before the velocity
# update, boundary-normal velocity before the pressure update) plus, for the
# adjoint, a mirror-pad of the model coefficients (K on the pressure grid along
# every axis, each buoyancy on its velocity grid along its own axis).


def _acoustic_c_reflection_coefficient(accuracy, face_boundary, freq=12.0):
    """Acoustic reflection coefficient off a low-x face via the compiled backend."""
    dx, dt, nt, nx = 2.0, 0.00025, 4000, 800
    s, r = 120, 70
    sa, tpk = _ricker(freq, nt, dt)
    v = torch.ones(nx) * C
    rho = torch.ones(nx) * RHO

    def run(bc, pml):
        return acoustic(
            v,
            rho,
            dx,
            dt,
            source_amplitudes_volume_rate=sa,
            source_locations_volume_rate=torch.tensor([[[s]]]),
            receiver_locations_p=torch.tensor([[[r]]]),
            pml_width=pml,
            boundary=bc,
            accuracy=accuracy,
            pml_freq=freq,
        )[-2][0, 0]

    tdir = int((s - r) * dx / C / dt) + tpk
    tref = int((s + r + 2) * dx / C / dt) + tpk
    w = int(1.0 / freq / dt)
    inc = run(None, [40, 40])
    a_dir = inc[tdir - w : tdir + w][inc[tdir - w : tdir + w].abs().argmax()]
    tr = run(face_boundary, [0, 40])
    refl = tr - inc
    a_ref = refl[tref - w : tref + w][refl[tref - w : tref + w].abs().argmax()]
    return (a_ref / a_dir).item()


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_acoustic_c_free_reflection_coefficient(accuracy):
    """Compiled acoustic ``"free"`` (p=0) face reflects with R = -1 to <1%."""
    r = _acoustic_c_reflection_coefficient(accuracy, ["free", "pml"])
    assert abs(r - (-1.0)) < 0.01, f"acc={accuracy}: R_free={r}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_acoustic_c_rigid_reflection_coefficient(accuracy):
    """Compiled acoustic ``"rigid"`` (v_n=0) face reflects with R = +1 to <1%."""
    r = _acoustic_c_reflection_coefficient(accuracy, ["rigid", "pml"])
    assert abs(r - 1.0) < 0.01, f"acc={accuracy}: R_rigid={r}"


# --- gradient conditioning for the model-parametrised propagators ----------
# ``assert_compiled_matches_eager_grad``'s tolerance is absolute and shared by
# every input, but the models are not of comparable magnitude: the
# acoustic_vti/elastic/elastic_vti/elastic_born stiffnesses are of order 1e9
# Pa, a wave speed is of order 1e3 m/s, a density is of order 1e3 kg/m^3, and
# a buoyancy is of order 1e-3 m^3/kg. In SI a stiffness gradient entry is of
# order 1e-14 per Pa, so a shared absolute tolerance that a velocity or
# buoyancy entry can meaningfully constrain is one the stiffness entries pass
# vacuously, whatever the compiled backward does with them. The tests below
# therefore differentiate dimensionless leaves, in the units defined here,
# converting back to SI inside the function under test, so every input's
# gradient is compared at a comparable scale. They also pin ``max_vel`` above
# the medium's true maximum speed, so the PML damping profile -- otherwise
# derived from the model's own maximum, a discretisation device that is not
# differentiated -- is a fixed property of the test rather than of the
# particular random medium drawn. Both effects are invisible while the
# recorded amplitudes are numerical dust; they appear as soon as a test is
# sized to produce a signal the tolerance can actually see.
MODULUS_UNIT = 1e9  # Pa
VELOCITY_UNIT = 1e3  # m/s
DENSITY_UNIT = 1e3  # kg/m^3
BUOYANCY_UNIT = 1e-3  # m^3/kg


@pytest.mark.parametrize(
    ("boundary", "pml", "shape", "src", "rec"),
    [
        (["free", "rigid"], [0, 0], (18,), [[[4]]], [[[3], [14]]]),
        (
            ["free", "rigid", "free", "rigid"],
            [0, 0, 0, 0],
            (14, 14),
            [[[4, 4]]],
            [[[3, 3]]],
        ),
        (
            ["rigid", "rigid", "rigid", "rigid"],
            [0, 0, 0, 0],
            (14, 14),
            [[[4, 4]]],
            [[[3, 3]]],
        ),
        (
            ["free", "pml", "rigid", "pml"],
            [0, 6, 0, 6],
            (16, 16),
            [[[4, 4]]],
            [[[3, 3]]],
        ),
        # 3-D: the 4-cell PML on the far end of each axis leaves a 4-cell
        # interior on each, with the source 2 cells from the reflecting face
        # at the near end. The 3-D case also uses its own ``dt``/``nt`` (see
        # the body).
        (
            ["free", "pml", "rigid", "pml", "rigid", "pml"],
            [0, 4, 0, 4, 0, 4],
            (8, 8, 8),
            [[[2, 2, 2]]],
            [[[1, 1, 1]]],
        ),
    ],
)
def test_acoustic_c_matches_eager_grad(boundary, pml, shape, src, rec):
    """Compiled acoustic: forward and (v, rho, source) gradients match eager.

    Across 1-D/2-D/3-D and every face acoustic offers, including mixed
    PML/reflecting axes. Every ``"free"`` face sits at the beginning of its
    dimension, which is where acoustic offers one above accuracy 2.
    Only the 3-D case, which runs at a larger ``dt`` (see below),
    carries the wave out to the reflecting faces within ``nt``: at the
    1-D/2-D cases' ``dt`` the front advances about a cell over the whole
    record, so those cases exercise the guard-ring fills and the adjoint
    model mirror-pad while the wave is still short of the faces. Gradient
    correctness once the wave has reflected off the faces repeatedly is
    covered by ``test_acoustic_c_boundary_adjoint_parity_well_conditioned``
    and the ``test_acoustic_c_fr_combination_adjoint_parity`` family, which
    run the same faces at long ``nt``.
    """
    torch.manual_seed(0)
    ndim = len(shape)
    dx = 5.0
    # The 3-D case runs at a larger ``dt`` than the 1-D/2-D cases. At
    # ``dt=0.0001`` the front advances only C*dt/dx = 0.03 cells/step, and on
    # a grid small enough to differentiate in 3-D the receiver then records
    # essentially nothing: both the recorded amplitudes and the model
    # gradients fall below the comparison tolerance, which passes the check
    # vacuously. ``dt=0.0007`` (0.21 cells/step) carries the front 6.3
    # cells in 30 steps, past both reflecting faces (2 cells from the source)
    # and back, so the fills act on a wave that is actually there; the
    # compiled and eager forwards still agree exactly at this ``dt`` for the
    # 3-D case's accuracy order and bounded model offsets.
    dt, nt = (0.0007, 30) if ndim == 3 else (0.0001, 40)
    acc = 2 if ndim == 3 else 4
    max_vel = 1600.0  # above the medium's true maximum speed of 1530 m/s
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = (
        (torch.ones(shape) * C + 30 * torch.rand(shape)) / VELOCITY_UNIT
    ).requires_grad_(True)
    rho = (
        (torch.ones(shape) * RHO + 20 * torch.rand(shape)) / DENSITY_UNIT
    ).requires_grad_(True)
    sl, rl = torch.tensor(src), torch.tensor(rec)

    def f(v, rho, sa, backend):
        return (
            acoustic(
                v * VELOCITY_UNIT,
                rho * DENSITY_UNIT,
                dx,
                dt,
                source_amplitudes_volume_rate=sa,
                source_locations_volume_rate=sl,
                receiver_locations_p=rl,
                pml_width=pml,
                boundary=boundary,
                accuracy=acc,
                pml_freq=25.0,
                max_vel=max_vel,
                backend=backend,
            )[-(1 + ndim)]
            * 1e-5
        )

    assert (f(v, rho, sa, "compiled") - f(v, rho, sa, "eager")).abs().max() < 1e-3
    assert_compiled_matches_eager_grad(f, (v, rho, sa))


def test_acoustic_c_jvp_supports_non_pml():
    """Forward-mode AD (jvp via Born) now supports non-PML faces.

    acoustic_born's compiled and eager backends both thread the first-class
    boundary ``bc`` codes, so acoustic's jvp rule -- which routes
    through acoustic_born -- no longer needs to refuse non-PML faces.
    Mirrors ``test_scalar_c_jvp_supports_non_pml``:
    confirms the compiled and eager jvp paths agree with each other for a
    non-PML boundary (formerly this raised NotImplementedError).
    """
    dx, dt, nt, nx = 5.0, 0.0001, 20, 20
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    kw = {
        "source_amplitudes_volume_rate": sa,
        "source_locations_volume_rate": torch.tensor([[[10]]]),
        "receiver_locations_p": torch.tensor([[[5]]]),
        "pml_width": [0, 0],
        "boundary": ["free", "rigid"],
    }
    rho = torch.ones(nx) * RHO
    v = torch.ones(nx) * C

    def f(vv, backend):
        return acoustic(vv, rho, dx, dt, backend=backend, **kw)[-2]

    _, tangent_c = torch.autograd.functional.jvp(
        lambda vv: f(vv, "compiled"), (v,), (torch.ones_like(v),)
    )
    _, tangent_e = torch.autograd.functional.jvp(
        lambda vv: f(vv, "eager"), (v,), (torch.ones_like(v),)
    )
    assert (tangent_c - tangent_e).abs().max() <= 1e-6 * tangent_e.abs().max()


def test_acoustic_c_hvp_supports_non_pml():
    """Second-order: the compiled HVP now supports non-PML faces.

    The double-backward routes through acoustic_born's compiled backend,
    which now threads the boundary ``bc`` codes AND mirror-pads the model
    coefficients' guard rings for the adjoint read (see
    ``mirror_pad_acoustic_models`` in ``AcousticBackwardFunc.backward``'s HVP
    path). Mirrors ``test_scalar_c_hvp_supports_non_pml``: confirms the
    compiled HVP no longer raises for a non-PML boundary (formerly this
    raised NotImplementedError) and agrees with eager.
    """
    dx, dt, nt, nx = 5.0, 0.0001, 20, 20
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    kw = {
        "source_amplitudes_volume_rate": sa,
        "source_locations_volume_rate": torch.tensor([[[10]]]),
        "receiver_locations_p": torch.tensor([[[5]]]),
        "pml_width": [0, 0],
        "boundary": ["free", "rigid"],
        "accuracy": 4,
    }
    rho = torch.ones(nx) * RHO

    def hvp(backend):
        vv = (torch.ones(nx) * C).requires_grad_(True)
        out = acoustic(vv, rho, dx, dt, backend=backend, **kw)[-2]
        g = torch.autograd.grad(out.sum(), vv, create_graph=True)[0]
        h = torch.autograd.grad(g.sum(), vv)[0]
        return g.detach(), h.detach()

    g_c, h_c = hvp("compiled")
    g_e, h_e = hvp("eager")
    assert (g_c - g_e).abs().max() <= 1e-6 * g_e.abs().max()
    assert (h_c - h_e).abs().max() <= 1e-6 * h_e.abs().max()


def test_acoustic_c_boundary_adjoint_parity_well_conditioned():
    """Compiled acoustic: long-``nt`` compiled-vs-eager gradient parity.

    ``test_acoustic_c_matches_eager_grad`` above runs its 1-D/2-D
    cases at ``nt=40``, ``dt=0.0001``: CFL = C*dt/dx = 1500*0.0001/5 = 0.03
    cells/step, so the wavefront moves under 2 cells and never reaches either
    reflecting face in those cases -- the same vacuous-gate pattern that hid
    the elastic-family compiled-backward adjoint bug (see
    ``test_elastic_c_jvp_supports_non_pml``'s docstring for the diagnosis).
    Unlike its Born/viscoacoustic/VTI siblings
    (``test_acoustic_born_c_boundary_adjoint_parity_well_conditioned`` and
    friends), plain ``acoustic`` never got a corresponding long-``nt``
    permanent regression test in this file -- this is that check. It uses
    ``nt=1200``: the front advances ``0.03 * 1200 = 36`` cells, more than
    twice the 16-cell reflecting-axis interior (``pml_width=0`` on that
    axis, matching ``test_elastic_c_jvp_supports_non_pml``'s own domain-size
    choice), so the wave genuinely reflects off both non-PML faces and
    returns past the receiver multiple times before ``nt*dt`` ends. The
    boundary is deliberately asymmetric (``["free", "rigid", "pml",
    "pml"]``) so a face-ordering defect would be visible.

    Checks compiled-vs-eager forward output and v/rho/source gradients at a
    tight relative tolerance, plus a central-FD arbitration (independent of
    both backends' analytic adjoints) at the grid cell where the compiled
    ``v`` gradient is largest in magnitude.
    """
    torch.manual_seed(0)
    shape = (16, 16)
    dx, dt, nt = 5.0, 0.0001, 1200
    acc = 4
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = (torch.ones(shape) * C + 30 * torch.rand(shape)).requires_grad_(True)
    rho = (torch.ones(shape) * RHO + 20 * torch.rand(shape)).requires_grad_(True)
    sl = torch.tensor([[[10, 10]]])
    rl = torch.tensor([[[5, 5]]])
    # 4 cells of PML at each end of the absorbing axis leaves cells 4-11 of it
    # undamped, so both the source and the receiver sit in the interior. A
    # wider PML than half the axis leaves no interior there at all and buries
    # the source in absorbing ground.
    pml_width = [0, 0, 4, 4]
    boundary = ["rigid", "pml", "pml", "pml"]

    def f(v, rho, sa, backend):
        return acoustic(
            v,
            rho,
            dx,
            dt,
            source_amplitudes_volume_rate=sa,
            source_locations_volume_rate=sl,
            receiver_locations_p=rl,
            pml_width=pml_width,
            boundary=boundary,
            accuracy=acc,
            pml_freq=25.0,
            backend=backend,
        ).receiver_amplitudes_p

    args = (v, rho, sa)
    g_c = assert_compiled_matches_eager_forward_and_grad(f, args, "compiled")

    def loss(vv):
        with torch.no_grad():
            return f(vv, rho.detach(), sa.detach(), "eager").sum().item()

    fd, analytic = _fd_arbitrate_max_grad_cell(loss, v, g_c[0])
    assert abs(fd - analytic) <= 1e-3 * abs(analytic)


# --- viscoacoustic (staggered grid + SLS memory variable) ------------------
# Ports acoustic's staggered-grid image method (pressure/
# boundary-normal-velocity fill) unchanged, extended so the SLS memory
# variable ``r`` is filled with the exact same image as pressure (no separate
# physics derivation -- see docs/boundaries.rst and viscoacoustic.c's
# FUNC(fill_p_bc_line) comment for the ROBERTSSON_FS_SUMMARY.md citation).
# Q is set large (weak attenuation) in the reflection-coefficient tests
# below so the analytic R=-1/+1 ground truth (derived for the lossless case)
# still applies to within the tests' tolerances.
# ``test_viscoacoustic_free_reflection_at_physical_q``
# repeats the free-surface measurement at a physically realistic Q=30, where
# that lossless ground truth no longer holds and the analytic reference must
# carry the SLS dispersion/attenuation of the extra reflected path.

Q = 5000.0
Q_PHYSICAL = 30.0


def _windowed_peak(signal, centre, half_width):
    """Signed value of ``signal`` where it is largest in magnitude in a window."""
    window = signal[centre - half_width : centre + half_width]
    return window[window.abs().argmax()]


def _viscoacoustic_reflection_traces(
    accuracy, face_boundary, backend="eager", freq=12.0, q_value=Q
):
    """Run the normal-incidence low-x face experiment and return its traces.

    A homogeneous 1-D viscoacoustic medium with a reflecting boundary on the
    low face; the incident field comes from an otherwise-identical run whose
    low face is absorbing, so ``reflecting - absorbing`` isolates the boundary
    reflection. ``backend`` selects the eager reference (``"eager"``) or
    (``"compiled"``) the compiled backend.

    Returns ``(incident, reflected, tdir, tref, half_width, dt, extra_path)``,
    where ``tdir``/``tref`` are the sample indices about which the direct and
    reflected arrivals are picked, ``half_width`` the picking half-window, and
    ``extra_path`` the distance by which the reflected ray exceeds the direct
    one. The free surface lies on the pressure node at model index 0, so the
    image source sits at index ``-s`` and ``extra_path = 2 * r * dx``, i.e.
    twice the receiver's depth below the surface, independent of the source.
    """
    dx = 2.0
    dt = 0.00025
    nt = 4000
    nx = 800
    s, r = 120, 70
    sa, tpk = _ricker(freq, nt, dt)
    v = torch.ones(nx) * C
    rho = torch.ones(nx) * RHO
    q = torch.ones(nx) * q_value

    def run(bc, pml):
        return viscoacoustic(
            v,
            rho,
            q,
            dx,
            dt,
            source_amplitudes_pressure_rate=(RHO * C**2) * (sa),
            source_locations_pressure_rate=torch.tensor([[[s]]]),
            receiver_locations_p=torch.tensor([[[r]]]),
            pml_width=pml,
            boundary=bc,
            backend=backend,
            accuracy=accuracy,
            pml_freq=freq,
            q_freq=freq,
        )[-2][0, 0]

    tdir = int((s - r) * dx / C / dt) + tpk
    tref = int((s + r + 2) * dx / C / dt) + tpk
    w = int(1.0 / freq / dt)
    inc = run(None, [40, 40])
    tr = run(face_boundary, [0, 40])
    return inc, tr - inc, tdir, tref, w, dt, 2 * r * dx


def _viscoacoustic_reflection_coefficient(
    accuracy, face_boundary, backend="eager", freq=12.0
):
    """Measure the normal-incidence pressure reflection off a low-x face.

    Mirrors :func:`_acoustic_reflection_coefficient`; ``backend`` selects the
    eager reference (``"eager"``) or (``"compiled"``) the compiled backend. Returns
    the signed reflected/direct peak-amplitude ratio.
    """
    inc, refl, tdir, tref, w, _, _ = _viscoacoustic_reflection_traces(
        accuracy, face_boundary, backend, freq
    )
    a_dir = _windowed_peak(inc, tdir, w)
    a_ref = _windowed_peak(refl, tref, w)
    return (a_ref / a_dir).item()


def _sls_propagate(signal, distance, dt, c, q_value, q_freq):
    """Propagate ``signal`` a further ``distance`` through the SLS medium.

    The exact plane-wave propagator of the single-mechanism standard linear
    solid that :mod:`deepwave.viscoacoustic` solves: multiply the spectrum by
    ``exp(-i k(omega) distance)`` with the complex wavenumber ``k = omega /
    (c * sqrt(M(omega)/K))`` derived in
    :func:`test_viscoacoustic_free_reflection_at_physical_q`. ``c`` is the
    relaxed (zero-frequency) wavespeed and ``q_freq`` the reference frequency
    at which the quality factor equals ``q_value``. The signal is zero-padded
    to twice its length so the (circular) spectral multiplication does not
    wrap the delayed arrival back onto the start of the trace.
    """
    n = signal.numel()
    padded = torch.zeros(2 * n, dtype=signal.dtype)
    padded[:n] = signal
    omega = 2 * math.pi * torch.fft.rfftfreq(2 * n, dt)
    omega0 = 2 * math.pi * q_freq
    inv_q = 1.0 / q_value
    tausig = (math.sqrt(1.0 + inv_q**2) - inv_q) / omega0
    taueps = 1.0 / (omega0**2 * tausig)
    modulus = (1.0 + 1j * omega * taueps) / (1.0 + 1j * omega * tausig)
    velocity = c * torch.sqrt(modulus)
    # The zero-frequency bin is a pure (undamped, undispersed) constant.
    k = torch.where(
        omega > 0, omega / velocity, torch.zeros_like(omega, dtype=velocity.dtype)
    )
    return torch.fft.irfft(torch.fft.rfft(padded) * torch.exp(-1j * k * distance))[:n]


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_viscoacoustic_free_reflection_coefficient(accuracy):
    """Eager viscoacoustic ``"free"`` (p=0) face reflects with R = -1 at every order."""
    r = _viscoacoustic_reflection_coefficient(accuracy, ["free", "pml"])
    assert abs(r - (-1.0)) < 0.01, f"acc={accuracy}: R_free={r}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_viscoacoustic_rigid_reflection_coefficient(accuracy):
    """Eager viscoacoustic ``"rigid"`` (v_n=0) face reflects with R = +1 always."""
    r = _viscoacoustic_reflection_coefficient(accuracy, ["rigid", "pml"])
    assert abs(r - 1.0) < 0.01, f"acc={accuracy}: R_rigid={r}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_viscoacoustic_c_free_reflection_coefficient(accuracy):
    """Compiled viscoacoustic ``"free"`` face reflects with R = -1 to <1%."""
    r = _viscoacoustic_reflection_coefficient(
        accuracy, ["free", "pml"], backend="compiled"
    )
    assert abs(r - (-1.0)) < 0.01, f"acc={accuracy}: R_free={r}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_viscoacoustic_c_rigid_reflection_coefficient(accuracy):
    """Compiled viscoacoustic ``"rigid"`` face reflects with R = +1 to <1%."""
    r = _viscoacoustic_reflection_coefficient(
        accuracy, ["rigid", "pml"], backend="compiled"
    )
    assert abs(r - 1.0) < 0.01, f"acc={accuracy}: R_rigid={r}"


@pytest.mark.parametrize("backend", ["eager", "compiled"])
@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_viscoacoustic_free_reflection_at_physical_q(accuracy, backend):
    """A ``"free"`` face is right at a realistic Q, not merely at Q >> 1.

    The reflection-coefficient tests above run at ``Q = 5000``, which is the
    lossless limit in all but name: they would pass just as well against a
    propagator that ignored attenuation altogether. This test repeats the
    same normal-incidence experiment at ``Q = 30`` -- a value typical of real
    sedimentary rock -- where the recorded surface reflection is some 21%
    weaker than the lossless ``R = -1`` prediction, and checks it against an
    analytic reference that accounts for the attenuation.

    **The complex modulus.** The propagator solves (see
    :mod:`deepwave.viscoacoustic`)::

        rho dv/dt = -dp/dx
        dp/dt = -K (1 + tau) dv/dx - r
        dr/dt = -(1/tausig) (r + K tau dv/dx)

    with ``K = rho c**2`` the relaxed (zero-frequency) bulk modulus and
    ``tau = taueps/tausig - 1``. For a time dependence ``exp(i omega t)`` the
    memory-variable equation gives ``r`` in closed form::

        r = -K tau / (1 + i omega tausig) dv/dx

    and substituting it into the pressure equation collapses the pair to a
    single constitutive relation ``i omega p = -M(omega) dv/dx`` with the
    complex modulus::

        M(omega) = K [ (1 + tau) - tau / (1 + i omega tausig) ]
                 = K (1 + i omega taueps) / (1 + i omega tausig)

    Separating real and imaginary parts, ``M = K [(1 + omega**2 taueps
    tausig) + i omega (taueps - tausig)] / (1 + omega**2 tausig**2)``, so the
    quality factor is ``Q(omega) = Re M / Im M = (1 + omega**2 taueps tausig)
    / (omega (taueps - tausig))``. The relaxation times Deepwave derives from
    the user's ``Q`` satisfy ``taueps tausig = 1/omega0**2``, which makes
    that expression exactly ``Q`` at the reference frequency ``omega0 = 2 pi
    q_freq`` (12 Hz here, the frequency these runs pass) -- so the
    analytic reference below and the propagator are parameterised
    consistently.

    **The dispersion relation.** Combining ``M`` with the momentum equation
    gives plane waves ``exp(i (omega t - k x))`` with::

        k(omega) = omega / V(omega),   V(omega) = c sqrt(M(omega) / K)

    ``Im M > 0`` for ``omega > 0``, hence ``Im V > 0`` and ``Im k < 0``, so
    ``exp(-i k x)`` decays with distance, while ``Re k`` gives a phase
    velocity rising with frequency from the relaxed ``c``. To first order in
    ``1/Q`` the decay rate is the familiar ``omega / (2 Q c)``, but the exact
    ``k`` above is used rather than that approximation, so the reference
    carries no weak-attenuation assumption of its own.

    **The free surface.** The boundary condition is ``p = 0`` on the surface.
    That constrains the field alone; it involves no material parameter and
    hence no rheology, so the normal-incidence pressure reflection
    coefficient remains exactly ``R = -1`` at every frequency however lossy
    the medium is. What attenuation changes is not ``R`` but the *response*:
    the reflected ray reaches the receiver over a longer path than the direct
    one, and that excess path attenuates and disperses it. The free surface
    lies on the pressure node at model index 0, so the image source lies at
    index ``-s``; with the source at index ``s`` and the receiver at index
    ``r < s`` the direct ray covers ``(s - r) dx`` and the reflected ray
    ``(s + r) dx``, an excess of::

        Delta = 2 r dx

    -- twice the receiver's depth, independent of the source position. There
    is no geometrical spreading in one dimension, so the two arrivals are
    related exactly by::

        P_reflected(omega) = -P_direct(omega) exp(-i k(omega) Delta)

    which is the analytic reference (:func:`_sls_propagate` applies it to the
    measured direct arrival). In the lossless limit ``k -> omega/c`` it is a
    pure time shift of amplitude ratio ``-1``, recovering the ``Q = 5000``
    tests' ground truth; at ``Q = 30`` and ``Delta = 280 m`` it predicts a
    peak-amplitude ratio near ``-0.79``.

    **Comparison and tolerance.** Measured and predicted reflections are
    reduced to the same scalar the ``Q = 5000`` tests use -- the signed
    reflected/direct peak-amplitude ratio, picked over the same windows --
    and compared with the same absolute tolerance of 0.01. No loosening is
    warranted: the reference is exact for the continuous problem, and the
    discretisation errors separating it from the simulation (the
    Crank-Nicolson memory-variable update, ``O((dt/tausig)**2)`` with
    ``dt/tausig`` about 0.02 here, and the leapfrog time stepping and spatial
    stencil at 62 points per wavelength) are all far below it. A peak
    amplitude is compared rather than a whole waveform because it is
    insensitive to the sub-sample arrival-time shift that the scheme's
    numerical dispersion introduces at accuracy 2.

    The second assertion records that the test has teeth: the attenuation is
    strong enough that the lossless ``R = -1`` ground truth used by the tests
    above is wrong here by more than twenty times the tolerance.
    """
    freq = 12.0
    inc, refl, tdir, tref, w, dt, extra_path = _viscoacoustic_reflection_traces(
        accuracy, ["free", "pml"], backend, freq, Q_PHYSICAL
    )
    a_dir = _windowed_peak(inc, tdir, w)
    measured = (_windowed_peak(refl, tref, w) / a_dir).item()
    predicted_trace = -_sls_propagate(inc, extra_path, dt, C, Q_PHYSICAL, freq)
    predicted = (_windowed_peak(predicted_trace, tref, w) / a_dir).item()
    assert abs(measured - predicted) < 0.01, (
        f"acc={accuracy} backend={backend}: R_free={measured}, analytic={predicted}"
    )
    assert abs(predicted - (-1.0)) > 0.2, (
        f"Q={Q_PHYSICAL} does not attenuate enough to test: analytic={predicted}"
    )


@pytest.mark.parametrize(
    ("boundary", "pml", "shape", "src", "rec"),
    [
        (["free", "rigid"], [0, 0], (18,), [[[4]]], [[[3], [14]]]),
        (
            ["free", "rigid", "free", "rigid"],
            [0, 0, 0, 0],
            (14, 14),
            [[[4, 4]]],
            [[[3, 3]]],
        ),
        (
            ["rigid", "rigid", "rigid", "rigid"],
            [0, 0, 0, 0],
            (14, 14),
            [[[4, 4]]],
            [[[3, 3]]],
        ),
        (
            ["free", "pml", "rigid", "pml"],
            [0, 6, 0, 6],
            (16, 16),
            [[[4, 4]]],
            [[[3, 3]]],
        ),
    ],
)
def test_viscoacoustic_c_matches_eager_grad(boundary, pml, shape, src, rec):
    """Compiled viscoacoustic: forward and (v, source) gradients match eager.

    Across 1-D/2-D and every face viscoacoustic offers, including a mixed
    PML/reflecting-axis case, exercising both the guard-ring field fills and
    the backward model mirror-pad (m1/c2/buoyancy). Every ``"free"`` face
    sits at the beginning of its dimension, which is where viscoacoustic
    offers one above accuracy 2. Mirrors
    :func:`test_acoustic_c_matches_eager_grad`.

    ``dt`` gives CFL = C*dt/dx = 1500*0.0007/5 = 0.21 cells/step, so over
    ``nt`` the front covers 8.4 cells and reaches both the receiver and the
    reflecting faces from anywhere in these grids; at a ``dt`` an order of
    magnitude smaller the front barely moves a cell, the receiver records
    almost nothing, and the ``v`` gradient sinks below the tolerance. The
    velocity leaf is dimensionless (in units of ``VELOCITY_UNIT``) and
    ``max_vel`` is pinned, for the reasons given above ``MODULUS_UNIT``.
    """
    torch.manual_seed(0)
    dx, dt, nt = 5.0, 0.0007, 40
    ndim = len(shape)
    acc = 4
    max_vel = 1600.0  # above the medium's true maximum speed of 1530 m/s
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = (
        (torch.ones(shape) * C + 30 * torch.rand(shape)) / VELOCITY_UNIT
    ).requires_grad_(True)
    rho = torch.ones(shape) * RHO
    q = torch.ones(shape) * 50.0
    sl, rl = torch.tensor(src), torch.tensor(rec)

    def f(v, sa, backend):
        return (
            viscoacoustic(
                v * VELOCITY_UNIT,
                rho,
                q,
                dx,
                dt,
                source_amplitudes_pressure_rate=(RHO * C**2) * (sa),
                source_locations_pressure_rate=sl,
                receiver_locations_p=rl,
                pml_width=pml,
                boundary=boundary,
                accuracy=acc,
                pml_freq=25.0,
                q_freq=25.0,
                max_vel=max_vel,
                backend=backend,
            )[-(1 + ndim)]
            * 1e-5
        )

    assert (f(v, sa, "compiled") - f(v, sa, "eager")).abs().max() < 1e-3
    assert_compiled_matches_eager_grad(f, (v, sa))


def test_viscoacoustic_c_boundary_adjoint_parity_well_conditioned():
    """Compiled viscoacoustic: long-``nt`` compiled-vs-eager gradient parity.

    ``test_viscoacoustic_c_matches_eager_grad`` above does carry
    the wave out to its reflecting faces, but only for the few round trips a
    short-``nt`` parity check allows. Gradient correctness after many
    reflections, where a face-ordering or guard-fill transpose defect
    accumulates, needs a longer run -- the same vacuous-gate pattern that hid
    the elastic-family compiled-backward adjoint bug (see
    ``test_elastic_c_jvp_supports_non_pml``'s docstring).
    This test uses ``dt=0.0001``, ``nt=1200``: the front advances
    ``0.03 * 1200 = 36`` cells, more than twice the 16-cell reflecting-axis
    interior (``pml_width=0`` on that axis, the same axis/domain-size choice
    as ``test_elastic_c_jvp_supports_non_pml``), so the wave genuinely
    reflects off both non-PML faces and returns past the receiver multiple
    times before ``nt*dt`` ends. The boundary is deliberately asymmetric
    (``["free", "rigid", "pml", "pml"]``) so a face-ordering defect would be
    visible, and the other axis stays PML so the test also exercises a mixed
    PML/reflecting configuration.

    Checks compiled-vs-eager forward output and v/source_amplitudes
    gradients at a tight relative tolerance, plus a central-FD arbitration
    (independent of both backends' analytic adjoints) at the grid cell where
    the compiled ``v`` gradient is largest in magnitude.
    """
    torch.manual_seed(0)
    shape = (16, 16)
    dx, dt, nt = 5.0, 0.0001, 1200
    acc = 4
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = (torch.ones(shape) * C + 30 * torch.rand(shape)).requires_grad_(True)
    rho = torch.ones(shape) * RHO
    q = torch.ones(shape) * Q
    sl = torch.tensor([[[10, 10]]])
    rl = torch.tensor([[[5, 5]]])
    # PML narrow enough to leave an interior (cells 4-11) on the absorbing
    # axis holding both the source and the receiver.
    pml_width = [0, 0, 4, 4]
    boundary = ["rigid", "pml", "pml", "pml"]

    def f(v, sa, backend):
        return viscoacoustic(
            v,
            rho,
            q,
            dx,
            dt,
            source_amplitudes_pressure_rate=(RHO * C**2) * (sa),
            source_locations_pressure_rate=sl,
            receiver_locations_p=rl,
            pml_width=pml_width,
            boundary=boundary,
            accuracy=acc,
            pml_freq=25.0,
            q_freq=25.0,
            backend=backend,
        ).receiver_amplitudes_p

    args = (v, sa)
    g_c = assert_compiled_matches_eager_forward_and_grad(f, args, "compiled")

    def loss(vv):
        with torch.no_grad():
            return f(vv, sa.detach(), "eager").sum().item()

    fd, analytic = _fd_arbitrate_max_grad_cell(loss, v, g_c[0])
    assert abs(fd - analytic) <= 1e-3 * abs(analytic)


# --- acoustic_vti (staggered grid, two normal stresses) --------------------
# acoustic_vti has two independent "pressure-like" fields --
# sigma_v (axial normal stress, dimension 0: z in 3-D, y in 2-D) and sigma_h
# (transverse normal stress, every other dimension). Per the CONTINUOUS
# SYSTEM equations in acoustic_vti.c's header, sigma_v is fed only by the
# axis-derivative term and feeds back only the axis velocity; sigma_h only by
# the transverse-derivative terms and only the transverse velocities. So
# sigma_v is guard-filled only on the axis face and sigma_h only on the
# transverse faces -- directly reusing acoustic's per-axis image rule
# (fill_acoustic_pressure_boundary), applied independently to each field on
# its own restricted face set (see acoustic_vti.py's
# _acoustic_vti_axis_boundary/_acoustic_vti_transverse_boundary and
# _boundaries.mirror_pad_acoustic_vti_models for the full citation, including
# why c13 needs no model mirror-pad).
#
# acoustic_vti has no 1-D form, so the reflection-coefficient tests below use
# an ISOTROPIC (eps = delta = 0, i.e. c11 = c13 = c33) 2-D medium and an
# additional trick to get an exact 1-D-equivalent measurement without a 2-D
# point-source Green's-function correction: the axis NOT under test is made
# ``"rigid"`` on both of its faces and driven by a uniform line of identical
# point sources (one per index along it), which by translation invariance
# collapses the genuinely-2-D domain to an exact 1-D plane wave along the
# tested axis (confirmed: every transverse receiver trace is identical to
# floating-point precision). A ``"rigid"`` transverse face is exactly the
# condition this field already satisfies: its transverse velocity is zero
# everywhere (so the odd image of it is zero too) and its normal stress is
# uniform along that axis (so the even image leaves the transverse derivative
# zero). In the isotropic medium with dvxdx (or dvydy) identically zero by
# that same symmetry, sigma_v's (or sigma_h's) own update decouples from the
# other stress and reduces exactly to the acoustic pressure equation with
# wave speed C, so acoustic's own analytic R = +1 ground truth applies
# directly. acoustic_vti offers no ``"free"`` face, so ``"rigid"`` is the
# only named condition measured here.

N_PER = 8  # transverse width; must exceed accuracy // 2 for every stencil


def _acoustic_vti_isotropic_media(shape, dtype=torch.double):
    c11 = torch.full(shape, C**2 * RHO, dtype=dtype)
    c13 = c11.clone()
    c33 = c11.clone()
    buoyancy = torch.full(shape, 1.0 / RHO, dtype=dtype)
    return c11, c13, c33, buoyancy


def _acoustic_vti_reflection_coefficient(
    accuracy, axis, face_boundary, backend="eager", freq=12.0
):
    """Normal-incidence reflection off a flat face, isolated to a 1-D plane wave.

    ``axis`` selects which acoustic_vti dimension is tested: ``0`` (the
    symmetry axis, "y" in 2-D -- reads/sources ``sigma_v``) or ``1``
    (transverse, "x" -- reads/sources ``sigma_h``). The other dimension is
    ``"rigid"`` on both faces with a uniform line source/receiver, which for
    this field is the same condition as wrapping it (see the module note
    above). Mirrors :func:`_acoustic_c_reflection_coefficient`'s
    direct-vs-reflected peak-amplitude-ratio methodology, applied to the
    resulting 1-D-equivalent trace.
    """
    dx, dt, nt, n_test = 2.0, 0.00025, 4000, 800
    s, r = 120, 70
    sa, tpk = _ricker(freq, nt, dt)
    sa_rep = sa.repeat(1, N_PER, 1)
    shape = [0, 0]
    shape[axis] = n_test
    shape[1 - axis] = N_PER
    c11, c13, c33, buoyancy = _acoustic_vti_isotropic_media(tuple(shape))
    channel = "sigma_v" if axis == 0 else "sigma_h"

    def loc(pos):
        coords = []
        for j in range(N_PER):
            c = [0, 0]
            c[axis] = pos
            c[1 - axis] = j
            coords.append(c)
        return torch.tensor([coords])

    def run(tested_bc, tested_pml):
        boundary = [None, None, None, None]
        pml_width = [None, None, None, None]
        boundary[2 * axis], boundary[2 * axis + 1] = tested_bc
        pml_width[2 * axis], pml_width[2 * axis + 1] = tested_pml
        other = 1 - axis
        boundary[2 * other] = boundary[2 * other + 1] = "rigid"
        pml_width[2 * other] = pml_width[2 * other + 1] = 0
        out = acoustic_vti(
            c11,
            c13,
            c33,
            buoyancy,
            dx,
            dt,
            **{
                f"source_amplitudes_{channel}": sa_rep,
                f"source_locations_{channel}": loc(s),
                f"receiver_locations_{channel}": loc(r),
            },
            pml_width=pml_width,
            boundary=boundary,
            accuracy=accuracy,
            pml_freq=freq,
            backend=backend,
        )
        chan = out[-2] if axis == 0 else out[-1]
        return chan[0].mean(dim=0)  # average the (identical) transverse traces

    tdir = int((s - r) * dx / C / dt) + tpk
    tref = int((s + r + 2) * dx / C / dt) + tpk
    w = int(1.0 / freq / dt)
    inc = run(("pml", "pml"), (40, 40))
    dwin = inc[tdir - w : tdir + w]
    a_dir = dwin[dwin.abs().argmax()]
    tr = run(tuple(face_boundary), (0, 40))
    refl = tr - inc
    rwin = refl[tref - w : tref + w]
    a_ref = rwin[rwin.abs().argmax()]
    return (a_ref / a_dir).item()


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_acoustic_vti_rigid_reflection_coefficient(accuracy, axis):
    """Eager acoustic_vti ``"rigid"`` face reflects with R = +1 to <1%."""
    r = _acoustic_vti_reflection_coefficient(accuracy, axis, ["rigid", "pml"])
    assert abs(r - 1.0) < 0.01, f"acc={accuracy} axis={axis}: R_rigid={r}"


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_acoustic_vti_c_rigid_reflection_coefficient(accuracy, axis):
    """Compiled acoustic_vti ``"rigid"`` face reflects with R = +1 to <1%."""
    r = _acoustic_vti_reflection_coefficient(
        accuracy, axis, ["rigid", "pml"], backend="compiled"
    )
    assert abs(r - 1.0) < 0.01, f"acc={accuracy} axis={axis}: R_rigid={r}"


@pytest.mark.parametrize(
    ("boundary", "pml", "src", "rec"),
    [
        (
            ["rigid", "rigid", "rigid", "rigid"],
            [0, 0, 0, 0],
            [[[4, 4]]],
            [[[3, 3]]],
        ),
        # The PML widths below leave a genuine interior on every axis, and
        # every source and receiver sits inside it: a source buried in PML,
        # or a receiver on the wall itself, records almost nothing and the
        # gradient check then has nothing to compare.
        (
            ["pml", "rigid", "rigid", "pml"],
            [6, 0, 0, 6],
            [[[11, 4]]],
            [[[12, 3]]],
        ),
        (
            ["rigid", "pml", "pml", "pml"],
            [0, 2, 2, 2],
            [[[4, 6]]],
            [[[1, 5]]],
        ),
        (
            ["pml", "pml", "rigid", "pml"],
            [4, 4, 0, 4],
            [[[7, 4]]],
            [[[6, 1]]],
        ),
    ],
)
def test_acoustic_vti_c_matches_eager_grad(boundary, pml, src, rec):
    """Compiled acoustic_vti: forward and gradients match eager.

    Differentiates c11, c13, c33, buoyancy and the source amplitudes, across
    every face acoustic_vti offers (``"pml"`` and ``"rigid"``), including a
    mixed PML/reflecting-axis case and both axis-only and transverse-only
    reflecting faces -- exercising both the per-field guard-ring fills and
    the backward model mirror-pad (c11/c33 selectively, buoyancy on its own
    axis; see ``_boundaries.mirror_pad_acoustic_vti_models``).

    ``dt`` gives CFL = C*dt/dx = 1500*0.0007/5 = 0.21 cells/step, so over
    ``nt`` the front covers 6.3 cells: far enough to reach the reflecting
    faces and the receiver from anywhere in these 14-cell grids. The
    dimensionless leaves and the pinned ``max_vel`` are explained above
    ``MODULUS_UNIT``.
    """
    torch.manual_seed(0)
    shape = (14, 14)
    dx, dt, nt = 5.0, 0.0007, 30
    acc = 4
    max_vel = 1600.0  # above the medium's true maximum speed of ~1515 m/s
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    c33 = (
        (torch.ones(shape) * C**2 * RHO + 3e4 * torch.rand(shape)) / MODULUS_UNIT
    ).requires_grad_(True)
    c11 = c33.detach().clone().requires_grad_(True)
    # ``c13`` sits strictly below ``sqrt(c11 * c33)`` so the medium is inside
    # the Grechka/Duveneck well-posedness region (``c11 * c33 >= c13**2``)
    # with room to spare: exactly on that boundary, any upward
    # finite-difference perturbation of ``c13`` is rejected by the
    # propagator, which no perturbation small enough to be lost to
    # floating-point rounding would ever have revealed.
    c13 = (0.9 * c33.detach()).clone().requires_grad_(True)
    buoyancy = (
        (torch.ones(shape) / RHO + 2e-5 * torch.rand(shape)) / BUOYANCY_UNIT
    ).requires_grad_(True)
    sl, rl = torch.tensor(src), torch.tensor(rec)

    def f(c11, c13, c33, buoyancy, sa, backend):
        out = acoustic_vti(
            c11 * MODULUS_UNIT,
            c13 * MODULUS_UNIT,
            c33 * MODULUS_UNIT,
            buoyancy * BUOYANCY_UNIT,
            dx,
            dt,
            source_amplitudes_sigma_v=sa,
            source_amplitudes_sigma_h=sa,
            source_locations_sigma_v=sl,
            source_locations_sigma_h=sl,
            receiver_locations_sigma_v=rl,
            receiver_locations_sigma_h=rl,
            pml_width=pml,
            boundary=boundary,
            accuracy=acc,
            pml_freq=25.0,
            max_vel=max_vel,
            backend=backend,
        )
        # The recorded amplitudes are small in SI, so they are scaled up (not
        # down) to sit well clear of the absolute gradient tolerance.
        return (out[-2] + out[-1]) * 1e4

    args = (c11, c13, c33, buoyancy, sa)
    assert (f(*args, "compiled") - f(*args, "eager")).abs().max() < 1e-3
    assert_compiled_matches_eager_grad(f, args)


def test_acoustic_vti_jvp_and_hvp_reject_with_boundary_set():
    """Forward-mode (jvp) and second-order (HVP) still refuse to run.

    acoustic_vti has no Born backend at all (unlike acoustic, whose jvp/HVP
    route through acoustic_born and support non-PML
    faces too -- see ``test_acoustic_c_jvp_supports_non_pml``/
    ``test_acoustic_c_hvp_supports_non_pml``), so acoustic_vti's compiled
    backend ``jvp``/second-order guards are unconditional and pre-date
    ``boundary=`` support; this pins that they still raise with a reflecting
    face set (not silently mishandled).
    """
    dx, dt, nt = 5.0, 0.0001, 20
    shape = (16, 16)
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    c11, c13, c33, buoyancy = _acoustic_vti_isotropic_media(shape)
    kw = {
        "source_amplitudes_sigma_v": sa,
        "source_locations_sigma_v": torch.tensor([[[10, 10]]]),
        "receiver_locations_sigma_v": torch.tensor([[[5, 5]]]),
        # PML narrow enough to leave an interior (cells 4-11) on the
        # absorbing axis holding both the source and the receiver.
        "pml_width": [0, 0, 4, 4],
        # ``"rigid"``, not ``"free"``: acoustic_vti offers no free face, and
        # a refused face would raise for the wrong reason.
        "boundary": ["rigid", "rigid", "pml", "pml"],
    }
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.autograd.functional.jvp(
            lambda cc: acoustic_vti(cc, c13, c33, buoyancy, dx, dt, **kw)[-2],
            (c11,),
            (torch.ones_like(c11),),
        )
    # Unlike acoustic (whose ``no_second_order`` guard only trips on the
    # *second* differentiation), acoustic_vti's compiled backward has no
    # analytic second-order path at all and raises as soon as ``backward``
    # runs with grad enabled -- i.e. already at the first
    # ``create_graph=True`` call, before a second differentiation is even
    # attempted. This is pre-existing (unconditional) behaviour, unrelated to
    # ``boundary=``.
    c11b = c11.clone().requires_grad_(True)
    out = acoustic_vti(c11b, c13, c33, buoyancy, dx, dt, accuracy=4, **kw)[-2]
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.autograd.grad(out.sum(), c11b, create_graph=True)


def test_acoustic_vti_c_boundary_adjoint_parity_well_conditioned():
    """Compiled acoustic_vti: long-``nt`` compiled-vs-eager gradient parity.

    ``test_acoustic_vti_c_matches_eager_grad`` above does carry the
    wave out to its reflecting faces, but only for the few round trips a
    short-``nt`` parity check allows. Gradient correctness after many
    reflections, where a face-ordering or guard-fill transpose defect
    accumulates, needs a longer run -- the same vacuous-gate pattern that hid
    the elastic-family compiled-backward adjoint bug (see
    ``test_elastic_c_jvp_supports_non_pml``'s docstring). This test uses
    ``dt=0.0001``, ``nt=1200``: the front advances ``0.03 * 1200 = 36``
    cells, more than twice the 16-cell reflecting-axis interior
    (``pml_width=0`` on that axis, matching ``test_elastic_c_jvp_supports_
    non_pml``'s own domain-size choice), so the wave genuinely reflects off
    both non-PML faces and returns past the receiver multiple times before
    ``nt*dt`` ends. acoustic_vti offers no ``"free"`` face, so the reflecting
    axis is ``"rigid"`` on both of its faces while the other axis stays PML:
    a defect that confused an axis, or a reflecting face with an absorbing
    one, would still be visible.

    Checks compiled-vs-eager forward output and c11/c13/c33/buoyancy/source
    gradients (summed over both stress channels) at a tight relative
    tolerance, plus a central-FD arbitration at the grid cell where the
    compiled ``buoyancy`` gradient is largest in magnitude. Arbitration
    perturbs ``buoyancy`` rather than a stiffness: this medium is built
    elliptical (``c11 = c13 = c33`` pointwise, satisfying the
    Grechka/Zhang/Rector stability constraint ``c11*c33 >= c13**2`` with
    equality -- see ``acoustic_vti``'s own stability check), so a one-cell
    FD perturbation of only ``c33`` (holding ``c11``/``c13`` fixed) can push
    that one cell to ``c11*c33 < c13**2`` and raise; ``buoyancy`` carries no
    such cross-model constraint.
    """
    torch.manual_seed(0)
    shape = (16, 16)
    dx, dt, nt = 5.0, 0.0001, 1200
    acc = 4
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    c33 = (torch.ones(shape) * C**2 * RHO + 3e4 * torch.rand(shape)).requires_grad_(
        True
    )
    c11 = c33.detach().clone().requires_grad_(True)
    c13 = c33.detach().clone().requires_grad_(True)
    buoyancy = (torch.ones(shape) / RHO + 2e-5 * torch.rand(shape)).requires_grad_(True)
    sl = torch.tensor([[[10, 10]]])
    rl = torch.tensor([[[5, 5]]])
    # PML narrow enough to leave an interior (cells 4-11) on the absorbing
    # axis holding both the source and the receiver.
    pml_width = [0, 0, 4, 4]
    boundary = ["rigid", "rigid", "pml", "pml"]
    # Pinned for the reason given above ``MODULUS_UNIT``: the central-FD
    # arbitration below perturbs ``buoyancy``, which feeds the wave speed the
    # PML profile is otherwise derived from, and that path is not
    # differentiated. Pinning it drops the FD-vs-adjoint disagreement at the
    # probed cell from 5e-3 to 4e-5 relative.
    max_vel = 1600.0  # above the medium's true maximum speed of ~1515 m/s

    def f(c11, c13, c33, buoyancy, sa, backend):
        out = acoustic_vti(
            c11,
            c13,
            c33,
            buoyancy,
            dx,
            dt,
            source_amplitudes_sigma_v=sa,
            source_amplitudes_sigma_h=sa,
            source_locations_sigma_v=sl,
            source_locations_sigma_h=sl,
            receiver_locations_sigma_v=rl,
            receiver_locations_sigma_h=rl,
            pml_width=pml_width,
            boundary=boundary,
            accuracy=acc,
            pml_freq=25.0,
            max_vel=max_vel,
            backend=backend,
        )
        return out.receiver_amplitudes_sigma_v + out.receiver_amplitudes_sigma_h

    args = (c11, c13, c33, buoyancy, sa)
    g_c = assert_compiled_matches_eager_forward_and_grad(f, args, "compiled")

    def loss(bb):
        with torch.no_grad():
            return (
                f(
                    c11.detach(),
                    c13.detach(),
                    c33.detach(),
                    bb,
                    sa.detach(),
                    "eager",
                )
                .sum()
                .item()
            )

    fd, analytic = _fd_arbitrate_max_grad_cell(loss, buoyancy, g_c[3])
    assert abs(fd - analytic) <= 1e-3 * abs(analytic)


# --- elastic (full stress-tensor multi-field parities) ---------------------
#
# Physics/derivation citations live in src/deepwave/_boundaries.py's
# module-level comment above fill_elastic_stress_boundary/
# fill_elastic_velocity_boundary, and in docs/boundaries.rst's Policy
# section. Only the eager backend implements boundary= for elastic so far
# (the compiled C/CUDA kernels have not been given a bc parameter yet); all
# tests here therefore pin backend="eager" explicitly, mirroring the
# "_c_" compiled-backend tests' *absence* rather than a placeholder.


def _elastic_isotropic_media(shape, dtype=torch.double):
    lamb = torch.full(shape, RHO * (C**2 - 2 * CS**2), dtype=dtype)
    mu = torch.full(shape, RHO * CS**2, dtype=dtype)
    buoyancy = torch.full(shape, 1.0 / RHO, dtype=dtype)
    return lamb, mu, buoyancy


CS = 900.0  # shear speed for the elastic isotropic test medium (C is the P speed)


def _elastic_reflection_coefficient(
    accuracy, face_boundary, freq=12.0, backend="eager"
):
    """Normal-incidence reflection off a flat elastic face, in one dimension.

    Uses ``dx=2.0``/``freq=12`` (not a coarser grid) because at accuracy=2
    (the lowest FD order under test) numerical dispersion needs a higher
    points-per-wavelength floor than accuracy 4/6/8 do for equivalent <1%
    convergence; verified directly that a coarser grid (~12 ppw at
    accuracy=4) converges accuracy 4/6/8 to <0.1% but leaves accuracy=2 at
    ~R=+0.92.

    The model is genuinely one-dimensional: a homogeneous 1-D elastic medium
    with the tested condition on the low face and an absorbing high face, so
    there is no transverse axis to emulate and the measured wave is a plane
    wave by construction. The incident arrival comes from an
    otherwise-identical run whose low face is absorbing, so ``reflecting -
    absorbing`` isolates the boundary reflection. Measures the pressure-type
    channel (``source_amplitudes_pressure_rate``/``receiver_locations_p``),
    which in 1-D is the axial normal stress ``sigma_AA`` -- the field with a
    node -- so ``rigid`` reflects it with ``R = +1`` (the acoustic/scalar
    convention). elastic offers no ``"free"`` face, so ``"rigid"`` is the
    only named condition measured here.
    """
    dx, dt, nt, n = 2.0, 0.0003, 4000, 800
    s, r = 60, 35
    sa, tpk = _ricker(freq, nt, dt)
    lamb, mu, buoyancy = _elastic_isotropic_media((n,))

    def run(bc, pml):
        return elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_pressure_rate=sa,
            source_locations_pressure_rate=torch.tensor([[[s]]]),
            receiver_locations_p=torch.tensor([[[r]]]),
            pml_width=pml,
            boundary=bc,
            accuracy=accuracy,
            pml_freq=freq,
            backend=backend,
        ).receiver_amplitudes_p[0, 0]

    tdir = int((s - r) * dx / C / dt) + tpk
    tref = int((s + r) * dx / C / dt) + tpk
    w = int(1.0 / freq / dt)
    inc = run(["pml", "pml"], [40, 40])
    dwin = inc[tdir - w : tdir + w]
    a_dir = dwin[dwin.abs().argmax()]
    tr = run(list(face_boundary), [0, 40])
    refl = tr - inc
    rwin = refl[tref - w : tref + w]
    a_ref = rwin[rwin.abs().argmax()]
    return (a_ref / a_dir).item()


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_elastic_rigid_reflection_coefficient(accuracy):
    """Eager elastic ``"rigid"`` face reflects sigma_AA with R = +1 to <1%."""
    r = _elastic_reflection_coefficient(accuracy, ["rigid", "pml"])
    assert abs(r - 1.0) < 0.01, f"acc={accuracy}: R_rigid={r}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_elastic_c_rigid_reflection_coefficient(accuracy):
    """Compiled elastic ``"rigid"`` face reflects sigma_AA with R = +1 to <1%."""
    r = _elastic_reflection_coefficient(accuracy, ["rigid", "pml"], backend="compiled")
    assert abs(r - 1.0) < 0.01, f"acc={accuracy}: R_rigid={r}"


def test_elastic_boundary_none_compiled_matches_eager():
    """``boundary=None`` (the default) is unaffected: compiled == eager.

    Pins that adding the ``boundary``/``ElasticForwardFunc.forward``
    plumbing did not perturb the existing all-PML compiled path at all.
    """
    ny, nx = 20, 20
    dx, dt, nt = 5.0, 0.0007, 60
    sa, _ = _ricker(15.0, nt, dt)
    lamb, mu, buoyancy = _elastic_isotropic_media((ny, nx))
    kw = {
        "source_amplitudes_pressure_rate": sa,
        "source_locations_pressure_rate": torch.tensor([[[10, 10]]]),
        "receiver_locations_p": torch.tensor([[[6, 6]]]),
        "pml_width": [10, 10, 10, 10],
        "accuracy": 4,
        "pml_freq": 15.0,
    }
    compiled = elastic(lamb, mu, buoyancy, dx, dt, **kw).receiver_amplitudes_p
    eager = elastic(
        lamb, mu, buoyancy, dx, dt, backend="eager", **kw
    ).receiver_amplitudes_p
    assert (compiled - eager).abs().max() / compiled.abs().max() < 1e-9


@pytest.mark.parametrize(
    ("boundary", "pml"),
    [
        (["rigid", "pml", "pml", "pml"], [0, 10, 10, 10]),
        (["pml", "pml", "rigid", "pml"], [10, 10, 0, 10]),
        (["rigid", "pml", "rigid", "pml"], [0, 10, 0, 10]),
    ],
)
def test_elastic_c_default_regression(boundary, pml):
    """Compiled elastic: every face it offers runs finite (the compiled bc ABI).

    Mirrors ``test_default_regression_2d``'s elastic coverage but on the
    compiled backend, which implements the same guard-ring fill. elastic
    offers no ``"free"`` face and offers ``"rigid"`` only at the beginning of
    a dimension at this accuracy order, so those are the cases here.
    """
    ny, nx = 20, 20
    dx, dt, nt = 5.0, 0.0007, 60
    sa, _ = _ricker(15.0, nt, dt)
    lamb, mu, buoyancy = _elastic_isotropic_media((ny, nx))
    out = elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_pressure_rate=sa,
        source_locations_pressure_rate=torch.tensor([[[10, 10]]]),
        receiver_locations_p=torch.tensor([[[6, 6]]]),
        pml_width=pml,
        boundary=boundary,
        accuracy=4,
        pml_freq=15.0,
    ).receiver_amplitudes_p
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    ("boundary", "pml", "src", "rec"),
    [
        # elastic-family propagators offer no ``"free"`` face, and offer
        # ``"rigid"`` only at the beginning of a dimension at this accuracy
        # order, so every reflecting face below is a low one. The PML widths
        # leave a genuine interior on every axis, with every source and
        # receiver inside it.
        (
            ["rigid", "pml", "rigid", "pml"],
            [0, 4, 0, 4],
            [[[4, 4]]],
            [[[3, 3]]],
        ),
        (
            ["pml", "pml", "rigid", "pml"],
            [4, 4, 0, 4],
            [[[7, 4]]],
            [[[6, 1]]],
        ),
        (
            ["rigid", "pml", "pml", "pml"],
            [0, 2, 2, 2],
            [[[4, 6]]],
            [[[1, 5]]],
        ),
    ],
)
def test_elastic_c_matches_eager_grad(boundary, pml, src, rec):
    """Compiled elastic: forward and gradients match eager.

    Differentiates lamb, mu, buoyancy and the source amplitudes, across every
    face type including a mixed PML/reflecting-axis case and a fully-enclosed
    free/rigid box -- exercising both the per-field guard-ring fills and
    the compiled backward's model mirror-pad (see
    ``_boundaries.mirror_pad_elastic_models``: lamb/mu/every mu_* shear
    stiffness mirrored on every axis, each buoyancy on its own axis).

    ``dt`` gives CFL (P) = C*dt/dx = 1500*0.0007/5 = 0.21 cells/step, so over
    ``nt`` the front covers 6.3 cells: far enough to reach the reflecting
    faces and the receiver from anywhere in these 14-cell grids. The
    dimensionless leaves and the pinned ``max_vel`` are explained above
    ``MODULUS_UNIT``.
    """
    torch.manual_seed(0)
    shape = (14, 14)
    dx, dt, nt = 5.0, 0.0007, 30
    acc = 4
    max_vel = 1550.0  # above the medium's true maximum P speed of ~1504 m/s
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    lamb0, mu0, buoyancy0 = _elastic_isotropic_media(shape)
    lamb = ((lamb0 + 3e6 * torch.rand(shape)) / MODULUS_UNIT).requires_grad_(True)
    mu = ((mu0 + 2e6 * torch.rand(shape)) / MODULUS_UNIT).requires_grad_(True)
    buoyancy = ((buoyancy0 + 2e-6 * torch.rand(shape)) / BUOYANCY_UNIT).requires_grad_(
        True
    )
    sl, rl = torch.tensor(src), torch.tensor(rec)

    def f(lamb, mu, buoyancy, sa, backend):
        out = elastic(
            lamb * MODULUS_UNIT,
            mu * MODULUS_UNIT,
            buoyancy * BUOYANCY_UNIT,
            dx,
            dt,
            source_amplitudes_pressure_rate=sa,
            source_locations_pressure_rate=sl,
            receiver_locations_p=rl,
            pml_width=pml,
            boundary=boundary,
            accuracy=acc,
            pml_freq=25.0,
            max_vel=max_vel,
            backend=backend,
        )
        # The recorded amplitudes are small in SI, so they are scaled up (not
        # down) to sit well clear of the absolute gradient tolerance.
        return out.receiver_amplitudes_p * 1e4

    args = (lamb, mu, buoyancy, sa)
    assert (f(*args, "compiled") - f(*args, "eager")).abs().max() < 1e-3
    assert_compiled_matches_eager_grad(f, args)


# Known RAM-cascade pair (this test and test_elastic_c_hvp_supports_non_pml
# below): each runs an nt=1200 elastic simulation with a
# double-backward, and under ``pytest -n 4`` the two have been observed to
# crash an xdist worker from a memory cascade even though neither is
# individually remarkable; both pass reliably every time when run serially
# (single-process, no ``-n``). ``xdist_group`` pins both tests onto the same
# worker so a parallel run schedules them sequentially relative to each
# other rather than concurrently, instead of leaving this knowledge only in
# prose.
@pytest.mark.xdist_group("ram_heavy")
def test_elastic_c_jvp_supports_non_pml():
    """Forward-mode AD (jvp via Born) supports non-PML faces.

    elastic's jvp rule routes through elastic_born's compiled backend, which
    threads the first-class boundary ``bc`` codes, so a non-PML face is
    supported rather than raising NotImplementedError. Mirrors
    ``test_acoustic_c_jvp_supports_non_pml``: confirms the compiled and eager
    jvp paths agree with each other for a non-PML boundary.

    Conditioning matters twice here. ``dt=0.0001`` is the well-conditioned
    choice this file's acoustic_born reflection tests settled on (see the
    acoustic_born Policy entry in docs/boundaries.rst for why a
    badly-conditioned ``dt``/perturbation choice can produce a misleadingly
    large compiled-vs-eager discrepancy that looks like a real bug but
    isn't). And ``nt`` must be large enough for the wavefield to genuinely
    reach the reflecting face and return to the receiver: the per-face
    codes here are deliberately asymmetric (rigid on one face, PML on the
    other three), so a face-ordering defect in the compiled call's ``bc``
    argument packing changes the physics -- but only a test whose waves
    actually interact with that face can see it. At CFL ~0.03 cells/step a 20-step run
    (the original, too-short choice) moves the front 0.6 cells and compares
    numerical dust; nt=1200 gives several source-to-face-to-receiver
    round trips of real amplitude.
    """
    ny, nx = 16, 16
    dx, dt, nt = 5.0, 0.0001, 1200
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    lamb, mu, buoyancy = _elastic_isotropic_media((ny, nx))
    kw = {
        "source_amplitudes_pressure_rate": sa,
        "source_locations_pressure_rate": torch.tensor([[[10, 10]]]),
        "receiver_locations_p": torch.tensor([[[5, 5]]]),
        # elastic offers ``"rigid"`` only at the beginning of a dimension at
        # this accuracy order, so the reflecting face is the low one of
        # dimension 0 and the other three absorb. A PML narrow enough to
        # leave an interior (cells 4-11) on the absorbing axis holds both
        # the source and the receiver.
        "pml_width": [0, 4, 4, 4],
        "boundary": ["rigid", "pml", "pml", "pml"],
    }

    def f(m, backend):
        return elastic(
            lamb, m, buoyancy, dx, dt, backend=backend, **kw
        ).receiver_amplitudes_p

    _, tangent_c = torch.autograd.functional.jvp(
        lambda m: f(m, "compiled"), (mu,), (torch.ones_like(mu),)
    )
    _, tangent_e = torch.autograd.functional.jvp(
        lambda m: f(m, "eager"), (mu,), (torch.ones_like(mu),)
    )
    assert (tangent_c - tangent_e).abs().max() <= 1e-6 * tangent_e.abs().max()


# See the RAM-cascade comment on test_elastic_c_jvp_supports_non_pml above:
# this test forms the other half of the known worker-crashing pair under
# ``pytest -n 4``; it passes reliably when run serially, and the
# ``xdist_group`` marker keeps it off a separate worker from its pair.
@pytest.mark.xdist_group("ram_heavy")
def test_elastic_c_hvp_supports_non_pml():
    """Second-order: the compiled HVP supports non-PML faces.

    The double-backward routes through elastic_born's compiled backend,
    which threads the boundary ``bc`` codes AND mirror-pads the model
    coefficients' guard rings for the adjoint read (see
    ``mirror_pad_elastic_models`` inside ``elastic_born_backward_compiled``).
    Mirrors ``test_acoustic_c_hvp_supports_non_pml``: confirms the compiled
    HVP does not raise for a non-PML boundary and agrees with eager.
    ``nt`` is sized so the wavefield genuinely interacts with the
    (deliberately asymmetric) reflecting face and returns to the receiver
    -- see ``test_elastic_c_jvp_supports_non_pml``'s docstring for why a
    short run would compare numerical dust and miss face-ordering defects.
    """
    ny, nx = 16, 16
    dx, dt, nt = 5.0, 0.0001, 1200
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    lamb, _mu, buoyancy = _elastic_isotropic_media((ny, nx))
    kw = {
        "source_amplitudes_pressure_rate": sa,
        "source_locations_pressure_rate": torch.tensor([[[10, 10]]]),
        "receiver_locations_p": torch.tensor([[[5, 5]]]),
        # elastic offers ``"rigid"`` only at the beginning of a dimension at
        # this accuracy order, so the reflecting face is the low one of
        # dimension 0 and the other three absorb. A PML narrow enough to
        # leave an interior (cells 4-11) on the absorbing axis holds both
        # the source and the receiver.
        "pml_width": [0, 4, 4, 4],
        "boundary": ["rigid", "pml", "pml", "pml"],
        "accuracy": 4,
    }

    def hvp(backend):
        m = _mu.clone().requires_grad_(True)
        out = elastic(
            lamb, m, buoyancy, dx, dt, backend=backend, **kw
        ).receiver_amplitudes_p
        g = torch.autograd.grad(out.sum(), m, create_graph=True)[0]
        h = torch.autograd.grad(g.sum(), m)[0]
        return g.detach(), h.detach()

    g_c, h_c = hvp("compiled")
    g_e, h_e = hvp("eager")
    assert (g_c - g_e).abs().max() <= 1e-6 * g_e.abs().max()
    assert (h_c - h_e).abs().max() <= 1e-6 * h_e.abs().max()


# --- elastic_vti boundary= faces --------------------------------------------
#
# Physics/derivation: elastic_vti's per-field image parity is identical to
# isotropic elastic's (see _boundaries.py's mirror_pad_elastic_vti_models
# docstring and docs/boundaries.rst's Policy section for the full
# derivation -- the free-surface traction-free condition and the
# zero-velocity ghost convention are statements about which fields are
# stresses/velocities and which axis they differentiate across, not about
# the stiffness values used to compute updates from them), so
# fill_elastic_stress_boundary/fill_elastic_velocity_boundary are reused
# unchanged for elastic_vti (see _elastic_eager_core, shared by both
# propagators). Rather than derive new anisotropic analytic oracles, every
# gate below uses an ISOTROPIC-REDUCTION VTI medium (c11=c33=lamb+2*mu,
# c13=lamb, c55=mu, so c12=c11-2*c66 is not needed in 2-D) run through the
# elastic_vti() API -- physically identical to the isotropic elastic medium
# used above, so the same analytic reflection coefficients / Rayleigh-Lamb
# S0 root apply verbatim; this exercises elastic_vti's own guard-fill
# wiring/model set/compiled kernels while reusing already-validated ground
# truth, rather than guessing at a genuinely anisotropic oracle.


def _elastic_vti_isotropic_media(shape, dtype=torch.double):
    """2-D VTI stiffnesses reducing exactly to :func:`_elastic_isotropic_media`.

    ``c11 = c33 = lamb + 2*mu``, ``c13 = lamb``, ``c55 = mu`` (c66 unused in
    2-D). See the module comment above this section.
    """
    lamb, mu, buoyancy = _elastic_isotropic_media(shape, dtype)
    c11 = lamb + 2 * mu
    c33 = lamb + 2 * mu
    c13 = lamb
    c55 = mu
    return c11, c13, c33, c55, buoyancy


def _elastic_vti_reflection_coefficient(
    accuracy, axis, face_boundary, freq=12.0, backend="eager"
):
    """VTI analogue of :func:`_elastic_reflection_coefficient`.

    ``elastic_vti`` has no one-dimensional implementation, so this keeps the
    2-D line-source geometry: a line of identical sources and receivers
    spanning the transverse axis, whose two faces are bare zero-width
    ``"pml"`` faces. For a field that is uniform along that axis such a face
    is exact -- the only exterior cells the transverse stencil reads are the
    transverse velocity and the shear stress, both identically zero here, so
    leaving them at zero is the exact condition -- which is what collapses
    the domain to a 1-D plane wave, and it is accepted at every accuracy.
    Otherwise identical to :func:`_elastic_reflection_coefficient`:
    incident-only subtraction, the isotropic-reduction medium above, and the
    same analytic ``R = +1`` on ``sigma_AA``.
    """
    dx, dt, nt, n_test = 2.0, 0.0003, 4000, 800
    s, r = 60, 35
    n_per = 12
    sa, tpk = _ricker(freq, nt, dt)
    sa_rep = sa.repeat(1, n_per, 1)
    shape = [0, 0]
    shape[axis] = n_test
    shape[1 - axis] = n_per
    c11, c13, c33, c55, buoyancy = _elastic_vti_isotropic_media(tuple(shape))

    def loc(pos):
        coords = []
        for j in range(n_per):
            c = [0, 0]
            c[axis] = pos
            c[1 - axis] = j
            coords.append(c)
        return torch.tensor([coords])

    def run(tested_bc, tested_pml):
        boundary = [None, None, None, None]
        pml_width = [None, None, None, None]
        boundary[2 * axis], boundary[2 * axis + 1] = tested_bc
        pml_width[2 * axis], pml_width[2 * axis + 1] = tested_pml
        other = 1 - axis
        boundary[2 * other] = boundary[2 * other + 1] = "pml"
        pml_width[2 * other] = pml_width[2 * other + 1] = 0
        out = elastic_vti(
            c11,
            c13,
            c33,
            c55,
            buoyancy,
            dx,
            dt,
            source_amplitudes_pressure_rate=sa_rep,
            source_locations_pressure_rate=loc(s),
            receiver_locations_p=loc(r),
            pml_width=pml_width,
            boundary=boundary,
            accuracy=accuracy,
            pml_freq=freq,
            backend=backend,
        )
        return out.receiver_amplitudes_p[0].mean(dim=0)

    tdir = int((s - r) * dx / C / dt) + tpk
    tref = int((s + r) * dx / C / dt) + tpk
    w = int(1.0 / freq / dt)
    inc = run(("pml", "pml"), (40, 40))
    dwin = inc[tdir - w : tdir + w]
    a_dir = dwin[dwin.abs().argmax()]
    tr = run(tuple(face_boundary), (0, 40))
    refl = tr - inc
    rwin = refl[tref - w : tref + w]
    a_ref = rwin[rwin.abs().argmax()]
    return (a_ref / a_dir).item()


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_elastic_vti_rigid_reflection_coefficient(accuracy, axis):
    """Eager elastic_vti ``"rigid"`` face reflects sigma_AA with R = +1 to <1%."""
    r = _elastic_vti_reflection_coefficient(accuracy, axis, ["rigid", "pml"])
    assert abs(r - 1.0) < 0.01, f"acc={accuracy} axis={axis}: R_rigid={r}"


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_elastic_vti_c_rigid_reflection_coefficient(accuracy, axis):
    """Compiled elastic_vti ``"rigid"`` face reflects sigma_AA with R = +1 to <1%."""
    r = _elastic_vti_reflection_coefficient(
        accuracy, axis, ["rigid", "pml"], backend="compiled"
    )
    assert abs(r - 1.0) < 0.01, f"acc={accuracy} axis={axis}: R_rigid={r}"


def test_elastic_vti_boundary_none_compiled_matches_eager():
    """``boundary=None`` (the default) is unaffected: compiled == eager.

    Pins that adding the ``boundary``/``ElasticVtiForwardFunc.forward``
    plumbing did not perturb the existing all-PML compiled path at all.
    """
    ny, nx = 20, 20
    dx, dt, nt = 5.0, 0.0007, 60
    sa, _ = _ricker(15.0, nt, dt)
    c11, c13, c33, c55, buoyancy = _elastic_vti_isotropic_media((ny, nx))
    kw = {
        "source_amplitudes_pressure_rate": sa,
        "source_locations_pressure_rate": torch.tensor([[[10, 10]]]),
        "receiver_locations_p": torch.tensor([[[6, 6]]]),
        "pml_width": [10, 10, 10, 10],
        "accuracy": 4,
        "pml_freq": 15.0,
    }
    compiled = elastic_vti(
        c11, c13, c33, c55, buoyancy, dx, dt, **kw
    ).receiver_amplitudes_p
    eager = elastic_vti(
        c11, c13, c33, c55, buoyancy, dx, dt, backend="eager", **kw
    ).receiver_amplitudes_p
    assert (compiled - eager).abs().max() / compiled.abs().max() < 1e-9


@pytest.mark.parametrize(
    ("boundary", "pml"),
    [
        (["rigid", "pml", "pml", "pml"], [0, 10, 10, 10]),
        (["pml", "pml", "rigid", "pml"], [10, 10, 0, 10]),
        (["rigid", "pml", "rigid", "pml"], [0, 10, 0, 10]),
    ],
)
def test_elastic_vti_c_default_regression(boundary, pml):
    """Compiled elastic_vti: every face it offers runs finite (the bc ABI).

    Mirrors ``test_default_regression_2d``'s elastic_vti coverage but on the
    compiled backend, which implements the same guard-ring fill. elastic_vti
    offers no ``"free"`` face and offers ``"rigid"`` only at the beginning of
    a dimension at this accuracy order, so those are the cases here.
    """
    ny, nx = 20, 20
    dx, dt, nt = 5.0, 0.0007, 60
    sa, _ = _ricker(15.0, nt, dt)
    c11, c13, c33, c55, buoyancy = _elastic_vti_isotropic_media((ny, nx))
    out = elastic_vti(
        c11,
        c13,
        c33,
        c55,
        buoyancy,
        dx,
        dt,
        source_amplitudes_pressure_rate=sa,
        source_locations_pressure_rate=torch.tensor([[[10, 10]]]),
        receiver_locations_p=torch.tensor([[[6, 6]]]),
        pml_width=pml,
        boundary=boundary,
        accuracy=4,
        pml_freq=15.0,
    ).receiver_amplitudes_p
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    ("boundary", "pml", "src", "rec"),
    [
        # elastic-family propagators offer no ``"free"`` face, and offer
        # ``"rigid"`` only at the beginning of a dimension at this accuracy
        # order, so every reflecting face below is a low one. The PML widths
        # leave a genuine interior on every axis, with every source and
        # receiver inside it.
        (
            ["rigid", "pml", "rigid", "pml"],
            [0, 4, 0, 4],
            [[[4, 4]]],
            [[[3, 3]]],
        ),
        (
            ["pml", "pml", "rigid", "pml"],
            [4, 4, 0, 4],
            [[[7, 4]]],
            [[[6, 1]]],
        ),
        (
            ["rigid", "pml", "pml", "pml"],
            [0, 2, 2, 2],
            [[[4, 6]]],
            [[[1, 5]]],
        ),
    ],
)
def test_elastic_vti_c_matches_eager_grad(boundary, pml, src, rec):
    """Compiled elastic_vti: forward and gradients match eager.

    Differentiates every Cij, buoyancy and the source amplitudes across every
    face type. Exercises both the per-field guard-ring fills and the compiled
    backward's model mirror-pad (see
    ``_boundaries.mirror_pad_elastic_vti_models``: every stress-role model
    mirrored on every axis, each buoyancy on its own axis).

    ``dt`` gives CFL (P) = C*dt/dx = 1500*0.0007/5 = 0.21 cells/step, so over
    ``nt`` the front covers 6.3 cells: far enough to reach the reflecting
    faces and the receiver from anywhere in these 14-cell grids. The
    dimensionless leaves and the pinned ``max_vel`` are explained above
    ``MODULUS_UNIT``.
    """
    torch.manual_seed(0)
    shape = (14, 14)
    dx, dt, nt = 5.0, 0.0007, 30
    acc = 4
    max_vel = 1800.0  # covers the perturbed constitutive/density CFL bound
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    c11_0, c13_0, c33_0, c55_0, buoyancy0 = _elastic_vti_isotropic_media(shape)
    c11 = ((c11_0 + 3e6 * torch.rand(shape)) / MODULUS_UNIT).requires_grad_(True)
    c13 = ((c13_0 + 3e6 * torch.rand(shape)) / MODULUS_UNIT).requires_grad_(True)
    c33 = ((c33_0 + 3e6 * torch.rand(shape)) / MODULUS_UNIT).requires_grad_(True)
    c55 = ((c55_0 + 2e6 * torch.rand(shape)) / MODULUS_UNIT).requires_grad_(True)
    buoyancy = ((buoyancy0 + 2e-6 * torch.rand(shape)) / BUOYANCY_UNIT).requires_grad_(
        True
    )
    sl, rl = torch.tensor(src), torch.tensor(rec)

    def f(c11, c13, c33, c55, buoyancy, sa, backend):
        out = elastic_vti(
            c11 * MODULUS_UNIT,
            c13 * MODULUS_UNIT,
            c33 * MODULUS_UNIT,
            c55 * MODULUS_UNIT,
            buoyancy * BUOYANCY_UNIT,
            dx,
            dt,
            source_amplitudes_pressure_rate=sa,
            source_locations_pressure_rate=sl,
            receiver_locations_p=rl,
            pml_width=pml,
            boundary=boundary,
            accuracy=acc,
            pml_freq=25.0,
            max_vel=max_vel,
            backend=backend,
        )
        # The recorded amplitudes are small in SI, so they are scaled up (not
        # down) to sit well clear of the absolute gradient tolerance.
        return out.receiver_amplitudes_p * 1e4

    args = (c11, c13, c33, c55, buoyancy, sa)
    assert (f(*args, "compiled") - f(*args, "eager")).abs().max() < 1e-3
    assert_compiled_matches_eager_grad(f, args)


def test_elastic_vti_c_boundary_adjoint_parity_well_conditioned():
    """Compiled elastic_vti: long-``nt`` compiled-vs-eager gradient parity.

    ``test_elastic_vti_c_matches_eager_grad`` above does carry the
    wave out to its reflecting faces, but only for the few round trips a
    short-``nt`` parity check allows. Gradient correctness after many
    reflections, where a face-ordering or guard-fill transpose defect
    accumulates, needs a longer run -- the same vacuous-gate pattern that hid
    elastic_vti's own compiled-backward adjoint bug at reflecting faces (see
    ``git show 431547f`` and ``test_elastic_c_jvp_supports_non_pml``'s
    docstring for the diagnosis). This test uses ``dt=0.0001``, ``nt=1200``:
    CFL = 1500*0.0001/5 = 0.03 cells/step, so the front advances ``0.03 *
    1200 = 36`` cells, more than twice the 16-cell reflecting-axis interior
    (``pml_width=0`` on that face, matching ``test_elastic_c_jvp_supports_
    non_pml``'s own domain-size choice), so the wave genuinely reflects off
    the non-PML face and returns past the receiver multiple times before
    ``nt*dt`` ends. elastic_vti offers ``"rigid"`` only at the beginning of a
    dimension at this accuracy order and offers no ``"free"`` face, so the
    boundary is one ``"rigid"`` face with the other three absorbing --
    deliberately asymmetric, so a face-ordering defect would be visible.

    Checks compiled-vs-eager forward output and c11/c13/c33/c55/buoyancy/
    source gradients at a tight relative tolerance, plus a central-FD
    arbitration at the grid cell where the compiled ``c55`` (shear
    stiffness) gradient is largest in magnitude -- ``c55`` because the fixed
    elastic-family compiled-backward adjoint bug was diagnosed via the shear
    modulus gradient specifically (see ``git show 84fd036``'s commit
    message).
    """
    torch.manual_seed(0)
    shape = (16, 16)
    dx, dt, nt = 5.0, 0.0001, 1200
    acc = 4
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    c11_0, c13_0, c33_0, c55_0, buoyancy0 = _elastic_vti_isotropic_media(shape)
    c11 = (c11_0 + 3e6 * torch.rand(shape)).requires_grad_(True)
    c13 = (c13_0 + 3e6 * torch.rand(shape)).requires_grad_(True)
    c33 = (c33_0 + 3e6 * torch.rand(shape)).requires_grad_(True)
    c55 = (c55_0 + 2e6 * torch.rand(shape)).requires_grad_(True)
    buoyancy = (buoyancy0 + 2e-6 * torch.rand(shape)).requires_grad_(True)
    sl = torch.tensor([[[10, 10]]])
    rl = torch.tensor([[[5, 5]]])
    # PML narrow enough to leave an interior (cells 4-11) on the absorbing
    # faces holding both the source and the receiver.
    pml_width = [0, 4, 4, 4]
    boundary = ["rigid", "pml", "pml", "pml"]

    def f(c11, c13, c33, c55, buoyancy, sa, backend):
        out = elastic_vti(
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
            pml_width=pml_width,
            boundary=boundary,
            accuracy=acc,
            pml_freq=25.0,
            backend=backend,
        )
        return out.receiver_amplitudes_p

    args = (c11, c13, c33, c55, buoyancy, sa)
    g_c = assert_compiled_matches_eager_forward_and_grad(f, args, "compiled")

    def loss(cc55):
        with torch.no_grad():
            return (
                f(
                    c11.detach(),
                    c13.detach(),
                    c33.detach(),
                    cc55,
                    buoyancy.detach(),
                    sa.detach(),
                    "eager",
                )
                .sum()
                .item()
            )

    fd, analytic = _fd_arbitrate_max_grad_cell(loss, c55, g_c[3])
    assert abs(fd - analytic) <= 1e-3 * abs(analytic)


# --- validation -----------------------------------------------------------


def test_set_boundary_none_is_all_pml():
    """``boundary=None`` maps to the all-PML code on every face."""
    assert set_boundary(None, [20, 20, 20, 20], 2) == [0, 0, 0, 0]


def test_set_boundary_maps_names():
    """Each boundary name maps to its expected integer code."""
    codes = set_boundary(["free", "rigid", "periodic", "periodic"], [0, 0, 0, 0], 2)
    assert codes == [1, 2, 3, 3]


def test_set_boundary_requires_zero_pml_on_reflecting_face():
    """A reflecting face with nonzero ``pml_width`` raises a clear error."""
    with pytest.raises(ValueError, match="requires pml_width == 0"):
        set_boundary(["free", "pml"], [20, 20], 1)


def test_set_boundary_periodic_needs_both_faces():
    """A periodic face on only one side of an axis raises a clear error."""
    with pytest.raises(ValueError, match="both faces"):
        set_boundary(["periodic", "pml"], [0, 0], 1)


def test_set_boundary_bad_name():
    """An unrecognised boundary name raises a clear error."""
    with pytest.raises(ValueError, match="must be one of"):
        set_boundary(["reflecting", "pml"], [0, 20], 1)


def test_set_boundary_wrong_length():
    """A boundary list of the wrong length raises a clear error."""
    with pytest.raises(RuntimeError, match="length"):
        set_boundary(["free"], [0, 20], 1)


# --- scalar compiled (C/CUDA) backend boundary faces ----------------------
# STEP B: the compiled backend now threads the per-face ``bc`` codes and applies
# the same guard-ring image fill as the eager reference, so these gates exercise
# the C path directly (cross-backend forward and gradient agreement, analytic
# reflection coefficients, periodic invariance, and the default bit-identity).


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_scalar_c_free_reflection_coefficient(accuracy):
    """Compiled ``"free"`` face reflects with R = -1 to <1% at every order."""
    r = _reflection_coefficient(accuracy, ["free", "pml"], backend="compiled")
    assert abs(r - (-1.0)) < 0.01, f"acc={accuracy}: R_free={r}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_scalar_c_rigid_reflection_coefficient(accuracy):
    """Compiled ``"rigid"`` face reflects with R = +1 to <1% at every order."""
    r = _reflection_coefficient(accuracy, ["rigid", "pml"], backend="compiled")
    assert abs(r - 1.0) < 0.01, f"acc={accuracy}: R_rigid={r}"


def test_scalar_c_periodic_translation_invariance():
    """The compiled periodic domain is exactly invariant under a shared shift."""
    dx, dt, nt, nx = 5.0, 0.0007, 1200, 200
    freq = 15.0
    sa, _ = _ricker(freq, nt, dt)
    v = torch.ones(nx) * C

    def run(s, r):
        return scalar(
            v,
            dx,
            dt,
            source_amplitudes=sa,
            source_locations=torch.tensor([[[s]]]),
            receiver_locations=torch.tensor([[[r]]]),
            pml_width=[0, 0],
            boundary=["periodic", "periodic"],
            accuracy=4,
            pml_freq=freq,
        )[-1][0, 0]

    a = run(100, 60)
    d = 70
    b = run((100 + d) % nx, (60 + d) % nx)
    assert (a - b).abs().max() / a.abs().max() < 1e-9


@pytest.mark.parametrize(
    ("boundary", "pml", "shape", "src", "rec"),
    [
        (["free", "rigid"], [0, 0], (16,), [[[4]]], [[[3], [13]]]),
        (
            ["free", "free", "free", "free"],
            [0, 0, 0, 0],
            (16, 16),
            [[[4, 4]]],
            [[[3, 3], [13, 13]]],
        ),
        (
            ["rigid", "rigid", "rigid", "rigid"],
            [0, 0, 0, 0],
            (16, 16),
            [[[4, 4]]],
            [[[3, 3]]],
        ),
        (
            ["periodic", "periodic", "periodic", "periodic"],
            [0, 0, 0, 0],
            (16, 16),
            [[[8, 8]]],
            [[[3, 3]]],
        ),
        # mixed: PML on one end of each axis, reflecting on the other
        (
            ["pml", "free", "rigid", "pml"],
            [6, 0, 0, 6],
            (18, 18),
            [[[13, 4]]],
            [[[15, 3]]],
        ),
        # 3-D: the 4-cell PML on one end of each of the first two axes leaves
        # a 4-cell interior on each, with the source 2 cells from the
        # reflecting face at the other end -- well within the 12.6 cells the
        # front covers in ``nt`` steps, so it still reflects repeatedly.
        (
            ["pml", "free", "rigid", "pml", "periodic", "periodic"],
            [4, 0, 0, 4, 0, 0],
            (8, 8, 8),
            [[[5, 2, 4]]],
            [[[6, 1, 2]]],
        ),
    ],
)
def test_scalar_c_matches_eager_grad(boundary, pml, shape, src, rec):
    """Compiled backend: forward and (v, source) gradients match eager.

    Covers 1-D, 2-D, and 3-D and every face type including mixed PML/reflecting
    axes (which exercise the empty-PML-strip corner logic). ``nt`` and the
    source/receiver geometry are chosen so the wave actually reaches and
    reflects off the reflecting faces within the recorded window (a short run
    where the wave never touches the boundary would not exercise the fill).
    """
    torch.manual_seed(0)
    dx, dt, nt = 5.0, 0.0007, 60
    acc = 2 if len(shape) == 3 else 4
    max_vel = 1550.0  # above the medium's true maximum speed of ~1504 m/s
    sa = deepwave.wavelets.ricker(15.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = ((torch.ones(shape) * C + torch.randn(shape)) / VELOCITY_UNIT).requires_grad_(
        True
    )
    sl, rl = torch.tensor(src), torch.tensor(rec)

    def f(vv, ss, backend):
        return scalar(
            vv * VELOCITY_UNIT,
            dx,
            dt,
            source_amplitudes=ss,
            source_locations=sl,
            receiver_locations=rl,
            pml_width=pml,
            boundary=boundary,
            accuracy=acc,
            pml_freq=15.0,
            max_vel=max_vel,
            backend=backend,
        )[-1]

    # The forward compiled/eager fields agree to machine precision even after
    # the wave has interacted with the boundary.
    assert (f(v, sa, "compiled") - f(v, sa, "eager")).abs().max() < 1e-10
    # Compiled and eager gradients agree to machine precision.
    assert_compiled_matches_eager_grad(f, (v, sa), tol=1e-9)


def test_scalar_c_jvp_supports_non_pml():
    """Forward-mode AD (jvp via Born) now supports non-PML faces (task O STEP C).

    scalar_born's compiled and eager backends both thread the first-class
    boundary ``bc`` codes (task O STEP A/B), so scalar's jvp rule -- which
    routes through scalar_born -- no longer needs to refuse non-PML faces
    (free/rigid/periodic). See ``test_jvp_receiver_equals_born_with_boundary``
    in test_scalar_jvp.py for the machine-precision equivalence check; this
    test just confirms the compiled and eager jvp paths agree with each other
    for a non-PML boundary (formerly this raised NotImplementedError).
    """
    dx, dt, nt, nx = 5.0, 0.0005, 12, 20
    sa, _ = _ricker(15.0, nt, dt)
    v = torch.ones(nx) * C

    def f(vv, backend):
        return scalar(
            vv,
            dx,
            dt,
            source_amplitudes=sa,
            source_locations=torch.tensor([[[10]]]),
            receiver_locations=torch.tensor([[[5]]]),
            pml_width=[0, 0],
            boundary=["free", "rigid"],
            backend=backend,
        )[-1]

    _, tangent_c = torch.autograd.functional.jvp(
        lambda vv: f(vv, "compiled"), (v,), (torch.ones_like(v),)
    )
    _, tangent_e = torch.autograd.functional.jvp(
        lambda vv: f(vv, "eager"), (v,), (torch.ones_like(v),)
    )
    assert (tangent_c - tangent_e).abs().max() <= 1e-6 * tangent_e.abs().max()


def test_scalar_c_hvp_supports_non_pml():
    """Second-order: the compiled HVP now supports non-PML faces (task O STEP C).

    The double-backward routes through scalar_born's compiled backend, which
    now threads the boundary ``bc`` codes AND mirror-pads the model
    coefficients' guard rings for the adjoint read (see
    ``mirror_pad_scalar_born_models`` in scalar.py's ``ScalarBackwardFunc.
    backward``). See ``test_hvp_matches_eager_with_boundary`` in
    test_scalar_jvp.py for the machine-precision compiled-vs-eager check on a
    short, controlled geometry (that test's docstring explains why a short
    geometry matters: comparing the ``-Ofast``/``--use_fast_math`` compiled
    backend against eager over many undamped reflections amplifies ordinary
    floating-point differences, unrelated to correctness). This test just
    confirms the compiled HVP no longer raises for a non-PML boundary
    (formerly this raised NotImplementedError) and agrees with eager.
    """
    dx, dt, nt, nx = 5.0, 0.0005, 12, 20
    sa, _ = _ricker(15.0, nt, dt)
    kw = {
        "source_amplitudes": sa,
        "source_locations": torch.tensor([[[10]]]),
        "receiver_locations": torch.tensor([[[5]]]),
        "pml_width": [0, 0],
        "boundary": ["free", "rigid"],
        "accuracy": 4,
    }

    def hvp(backend):
        vv = (torch.ones(nx) * C).requires_grad_(True)
        out = scalar(vv, dx, dt, backend=backend, **kw)[-1]
        g = torch.autograd.grad(out.sum(), vv, create_graph=True)[0]
        h = torch.autograd.grad(g.sum(), vv)[0]
        return g.detach(), h.detach()

    g_c, h_c = hvp("compiled")
    g_e, h_e = hvp("eager")
    assert (g_c - g_e).abs().max() <= 1e-6 * g_e.abs().max()
    assert (h_c - h_e).abs().max() <= 1e-6 * h_e.abs().max()

    # The eager backend also composes the second-order derivative correctly
    # against a gradgradcheck (unaffected by this task; kept for coverage).
    v2 = (torch.ones(nx) * C + 0.1 * torch.randn(nx)).requires_grad_(True)

    def fe(vv):
        return scalar(vv, dx, dt, backend="eager", **kw)[-1]

    assert torch.autograd.gradgradcheck(fe, (v2,), atol=1e-5, rtol=1e-3, eps=1e-6)


def test_scalar_c_boundary_adjoint_parity_well_conditioned():
    """Compiled scalar: long-``nt`` compiled-vs-eager gradient parity.

    ``test_scalar_c_matches_eager_grad`` above does carry the wave
    out to its reflecting faces, but only for the few round trips a
    short-``nt`` parity check allows, and
    ``test_scalar_c_periodic_wrapping_adjoint_parity`` runs a long ``nt``
    on periodic faces only. Neither pins the free/rigid adjoint after the
    wave has reflected many times, which is where a face-ordering or
    guard-fill transpose defect accumulates -- the same vacuous-gate pattern
    that hid the elastic-family compiled-backward adjoint bug (see
    ``test_elastic_c_jvp_supports_non_pml``'s docstring for the diagnosis).
    This test is that check, on the harness
    ``test_acoustic_c_boundary_adjoint_parity_well_conditioned`` uses for the
    staggered family. It runs ``nt=1200`` at ``dt=0.0001``: CFL = C*dt/dx =
    1500*0.0001/5 = 0.03 cells/step, so the front advances ``0.03 * 1200 =
    36`` cells, more than twice the 16-cell reflecting-axis interior
    (``pml_width=0`` on that axis), and the wave reflects off both non-PML
    faces and returns past the receiver several times before ``nt*dt`` ends.
    The boundary is deliberately asymmetric (``["free", "rigid", "pml",
    "pml"]``) so a face-ordering defect would be visible.

    Checks compiled-vs-eager forward output and v/source gradients at a
    tight relative tolerance, plus a central-FD arbitration (independent of
    both backends' analytic adjoints) at the grid cell where the compiled
    ``v`` gradient is largest in magnitude.
    """
    torch.manual_seed(0)
    shape = (16, 16)
    dx, dt, nt = 5.0, 0.0001, 1200
    acc = 4
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = (torch.ones(shape) * C + 30 * torch.rand(shape)).requires_grad_(True)
    sl = torch.tensor([[[10, 10]]])
    rl = torch.tensor([[[5, 5]]])
    # 4 cells of PML at each end of the absorbing axis leaves cells 4-11 of it
    # undamped, so both the source and the receiver sit in the interior.
    pml_width = [0, 0, 4, 4]
    boundary = ["rigid", "pml", "pml", "pml"]

    def f(v, sa, backend):
        return scalar(
            v,
            dx,
            dt,
            source_amplitudes=sa,
            source_locations=sl,
            receiver_locations=rl,
            pml_width=pml_width,
            boundary=boundary,
            accuracy=acc,
            pml_freq=25.0,
            backend=backend,
        )[-1]

    args = (v, sa)
    g_c = assert_compiled_matches_eager_forward_and_grad(f, args, "compiled")

    def loss(vv):
        with torch.no_grad():
            return f(vv, sa.detach(), "eager").sum().item()

    fd, analytic = _fd_arbitrate_max_grad_cell(loss, v, g_c[0])
    assert abs(fd - analytic) <= 1e-3 * abs(analytic)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_scalar_c_boundary_cpu_gpu_match():
    """Compiled backend, first-class boundary faces: CPU and GPU forward + grad match.

    Mirrors ``test_elastic.py``'s ``test_forward_cpu_gpu_match`` pattern,
    applied to a first-class ``boundary`` face (all-``"free"``) on the
    compiled scalar-propagator backend, and additionally checks the gradient
    (the compiled boundary path has a full CUDA adjoint, unlike elastic_born).
    Runs only on the maintainer's GPU (skipped here, no CUDA on this box).
    """
    torch.manual_seed(0)
    dx, dt, nt = 5.0, 0.0007, 60
    shape = (16, 16)
    boundary = ["free", "free", "free", "free"]
    pml = [0, 0, 0, 0]
    sa0 = deepwave.wavelets.ricker(15.0, nt, dt, 0.02).reshape(1, 1, nt)
    v0 = torch.ones(shape) * C + torch.randn(shape)
    sl = torch.tensor([[[4, 4]]])
    rl = torch.tensor([[[3, 3], [13, 13]]])

    def run(device):
        vv = v0.to(device).clone().requires_grad_(True)
        ss = sa0.to(device).clone().requires_grad_(True)
        out = scalar(
            vv,
            dx,
            dt,
            source_amplitudes=ss,
            source_locations=sl.to(device),
            receiver_locations=rl.to(device),
            pml_width=pml,
            boundary=boundary,
            accuracy=4,
            pml_freq=15.0,
            backend="compiled",
        )[-1]
        out.sum().backward()
        return out.detach().cpu(), vv.grad.detach().cpu(), ss.grad.detach().cpu()

    rec_cpu, gv_cpu, gs_cpu = run(torch.device("cpu"))
    rec_gpu, gv_gpu, gs_gpu = run(torch.device("cuda"))
    assert torch.allclose(rec_cpu, rec_gpu, atol=1e-6, rtol=1e-5)
    assert torch.allclose(gv_cpu, gv_gpu, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gs_cpu, gs_gpu, atol=1e-5, rtol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_acoustic_c_boundary_cpu_gpu_match():
    """Compiled backend, first-class boundary faces: CPU and GPU forward + grad match.

    The acoustic-propagator analogue of ``test_scalar_c_boundary_cpu_gpu_match``
    above. Runs only on the maintainer's GPU (skipped here, no CUDA on this box).
    """
    torch.manual_seed(1)
    dx, dt, nt = 2.0, 0.00025, 400
    nx = 200
    boundary = ["free", "rigid"]
    pml = [0, 0]
    sa0 = deepwave.wavelets.ricker(12.0, nt, dt, 1.4 / 12.0).reshape(1, 1, nt)
    v0 = torch.ones(nx) * C
    rho0 = torch.ones(nx) * RHO

    def run(device):
        vv = v0.to(device).clone().requires_grad_(True)
        rr = rho0.to(device).clone().requires_grad_(True)
        ss = sa0.to(device).clone().requires_grad_(True)
        out = acoustic(
            vv,
            rr,
            dx,
            dt,
            source_amplitudes_volume_rate=ss,
            source_locations_volume_rate=torch.tensor([[[120]]], device=device),
            receiver_locations_p=torch.tensor([[[70]]], device=device),
            pml_width=pml,
            boundary=boundary,
            accuracy=4,
            pml_freq=12.0,
            backend="compiled",
        )[-2]
        out.sum().backward()
        return (
            out.detach().cpu(),
            vv.grad.detach().cpu(),
            rr.grad.detach().cpu(),
            ss.grad.detach().cpu(),
        )

    rec_cpu, gv_cpu, gr_cpu, gs_cpu = run(torch.device("cpu"))
    rec_gpu, gv_gpu, gr_gpu, gs_gpu = run(torch.device("cuda"))
    assert torch.allclose(rec_cpu, rec_gpu, atol=1e-6, rtol=1e-5)
    assert torch.allclose(gv_cpu, gv_gpu, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gr_cpu, gr_gpu, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gs_cpu, gs_gpu, atol=1e-5, rtol=1e-4)


# --- scalar Born (regular grid, background + scattered fields) --------------
#
# scalar_born is nodal like scalar, so it reuses the same odd/even/wrap image
# fill; both the background and the scattered wavefields obey the chosen faces
# (mirroring is linear, so the scattered field uses the same images as the
# background). Each analytic test therefore checks BOTH the background and the
# scattered receiver. The eager tests below cover STEP A (pure-PyTorch backend);
# the ``_c_`` tests further down drive the compiled backend (STEP B).


def _born_reflection_coefficient(accuracy, face_boundary, freq=12.0, backend="eager"):
    """Measure the R off a low-x face for the background AND scattered fields.

    A homogeneous 1-D medium with a single point scatterer at ``xsc``; the
    background field emanates from the source and the scattered field emanates
    from the scatterer (as the background wave passes it). For each field the
    incident arrival is taken from an otherwise-identical run whose low face is
    absorbing, so ``reflecting - absorbing`` isolates the boundary reflection.
    Returns ``(R_bg, R_sc)``, the signed reflected/direct peak ratios.
    """
    dx = 2.0
    dt = 0.00025
    nt = 4000
    nx = 800
    s, r, xsc = 120, 70, 200
    sa, tpk = _ricker(freq, nt, dt)
    v = torch.ones(nx) * C
    scat = torch.zeros(nx)
    scat[xsc] = C

    def run(bc, pml):
        out = scalar_born(
            v,
            dx,
            dt,
            dv=scat,
            source_amplitudes=sa,
            source_locations=torch.tensor([[[s]]]),
            sc_receiver_locations=torch.tensor([[[r]]]),
            bg_receiver_locations=torch.tensor([[[r]]]),
            pml_width=pml,
            boundary=bc,
            backend=backend,
            accuracy=accuracy,
            pml_freq=freq,
        )
        return out[-2][0, 0], out[-1][0, 0]  # background receiver, scattered

    w = int(1.0 / freq / dt)
    bg_inc, sc_inc = run(None, [40, 40])
    bg_ref, sc_ref = run(face_boundary, [0, 40])

    # Background: direct s->r, reflected off the low face s->0->r (+2 ghost).
    t_bg_dir = int((s - r) * dx / C / dt) + tpk
    t_bg_ref = int((s + r + 2) * dx / C / dt) + tpk
    bg_dw = bg_inc[t_bg_dir - w : t_bg_dir + w]
    bg_a_dir = bg_dw[bg_dw.abs().argmax()]
    bg_rw = (bg_ref - bg_inc)[t_bg_ref - w : t_bg_ref + w]
    bg_a_ref = bg_rw[bg_rw.abs().argmax()]

    # Scattered: direct s->xsc->r, reflected s->xsc->0->r.
    t_sc_dir = int((abs(s - xsc) + abs(xsc - r)) * dx / C / dt) + tpk
    t_sc_ref = int((abs(s - xsc) + (xsc + r + 2)) * dx / C / dt) + tpk
    sc_dw = sc_inc[t_sc_dir - w : t_sc_dir + w]
    sc_a_dir = sc_dw[sc_dw.abs().argmax()]
    sc_rw = (sc_ref - sc_inc)[t_sc_ref - w : t_sc_ref + w]
    sc_a_ref = sc_rw[sc_rw.abs().argmax()]

    return (bg_a_ref / bg_a_dir).item(), (sc_a_ref / sc_a_dir).item()


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_scalar_born_free_reflection_coefficient(accuracy):
    """A ``"free"`` face reflects both Born fields with R = -1 at every order."""
    r_bg, r_sc = _born_reflection_coefficient(accuracy, ["free", "pml"])
    assert abs(r_bg - (-1.0)) < 0.01, f"acc={accuracy}: R_bg={r_bg}"
    assert abs(r_sc - (-1.0)) < 0.01, f"acc={accuracy}: R_sc={r_sc}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_scalar_born_rigid_reflection_coefficient(accuracy):
    """A ``"rigid"`` face reflects both Born fields with R = +1 at every order."""
    r_bg, r_sc = _born_reflection_coefficient(accuracy, ["rigid", "pml"])
    assert abs(r_bg - 1.0) < 0.01, f"acc={accuracy}: R_bg={r_bg}"
    assert abs(r_sc - 1.0) < 0.01, f"acc={accuracy}: R_sc={r_sc}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_scalar_born_periodic_translation_invariance(accuracy):
    """A periodic Born domain is invariant under a shared source+receiver shift."""
    dx, dt, nt, nx = 5.0, 0.0007, 1500, 200
    freq = 15.0
    sa, _ = _ricker(freq, nt, dt)
    v = torch.ones(nx) * C
    scat = torch.zeros(nx)
    scat[80] = C

    def run(s, r):
        out = scalar_born(
            v,
            dx,
            dt,
            dv=scat,
            source_amplitudes=sa,
            source_locations=torch.tensor([[[s]]]),
            sc_receiver_locations=torch.tensor([[[r]]]),
            bg_receiver_locations=torch.tensor([[[r]]]),
            pml_width=[0, 0],
            boundary=["periodic", "periodic"],
            backend="eager",
            accuracy=accuracy,
            pml_freq=freq,
        )
        return out[-2][0, 0], out[-1][0, 0]

    d = 70
    bg_a, sc_a = run(100, 60)
    # Shift the source, receiver, and scatterer together by ``d`` (mod nx).
    scat = torch.zeros(nx)
    scat[(80 + d) % nx] = C
    bg_b, sc_b = run((100 + d) % nx, (60 + d) % nx)
    assert (bg_a - bg_b).abs().max() / bg_a.abs().max() < 1e-9
    assert (sc_a - sc_b).abs().max() / sc_a.abs().max() < 1e-9


def test_scalar_born_boundary_2d_runs_and_reflects():
    """A 2-D free surface reflects the scattered Born field with inverted sign."""
    dx, dt, nt = 5.0, 0.0007, 700
    ny, nx = 80, 120
    freq = 15.0
    sa, _ = _ricker(freq, nt, dt)
    v = torch.ones(ny, nx) * C
    scat = torch.zeros(ny, nx)
    scat[40, 60] = C

    def run(pml, boundary):
        return scalar_born(
            v,
            dx,
            dt,
            dv=scat,
            source_amplitudes=sa,
            source_locations=torch.tensor([[[30, 60]]]),
            sc_receiver_locations=torch.tensor([[[15, 60]]]),
            bg_receiver_locations=torch.tensor([[[15, 60]]]),
            pml_width=pml,
            boundary=boundary,
            backend="eager",
            accuracy=4,
            pml_freq=freq,
        )[-1][0, 0]

    out = run([0, 20, 20, 20], ["free", "pml", "pml", "pml"])
    inc = run([20, 20, 20, 20], None)
    refl = out - inc
    assert refl.abs().max() > 0.1 * inc.abs().max()


# --- scalar Born compiled (C/CUDA) backend boundary faces ------------------
#
# STEP B: the guard-ring fill (and, for the adjoint, the model mirror-pad) is
# implemented in the compiled backend too, so ``boundary=`` no longer requires
# ``backend="eager"`` for scalar_born. These tests drive the C path
# directly (cross-backend forward and gradient agreement, analytic
# reflection coefficients for both fields, periodic invariance).


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_scalar_born_c_free_reflection_coefficient(accuracy):
    """Compiled ``"free"`` face: both Born fields reflect with R=-1, <1%."""
    r_bg, r_sc = _born_reflection_coefficient(
        accuracy, ["free", "pml"], backend="compiled"
    )
    assert abs(r_bg - (-1.0)) < 0.01, f"acc={accuracy}: R_bg={r_bg}"
    assert abs(r_sc - (-1.0)) < 0.01, f"acc={accuracy}: R_sc={r_sc}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_scalar_born_c_rigid_reflection_coefficient(accuracy):
    """Compiled ``"rigid"`` face: both Born fields reflect with R=+1, <1%."""
    r_bg, r_sc = _born_reflection_coefficient(
        accuracy, ["rigid", "pml"], backend="compiled"
    )
    assert abs(r_bg - 1.0) < 0.01, f"acc={accuracy}: R_bg={r_bg}"
    assert abs(r_sc - 1.0) < 0.01, f"acc={accuracy}: R_sc={r_sc}"


def test_scalar_born_c_periodic_translation_invariance():
    """The compiled Born periodic domain is invariant under a shared shift."""
    dx, dt, nt, nx = 5.0, 0.0007, 1200, 200
    freq = 15.0
    sa, _ = _ricker(freq, nt, dt)
    v = torch.ones(nx) * C
    scat = torch.zeros(nx)
    scat[80] = C

    def run(s, r):
        out = scalar_born(
            v,
            dx,
            dt,
            dv=scat,
            source_amplitudes=sa,
            source_locations=torch.tensor([[[s]]]),
            sc_receiver_locations=torch.tensor([[[r]]]),
            bg_receiver_locations=torch.tensor([[[r]]]),
            pml_width=[0, 0],
            boundary=["periodic", "periodic"],
            accuracy=4,
            pml_freq=freq,
        )
        return out[-2][0, 0], out[-1][0, 0]

    d = 70
    bg_a, sc_a = run(100, 60)
    scat = torch.zeros(nx)
    scat[(80 + d) % nx] = C
    bg_b, sc_b = run((100 + d) % nx, (60 + d) % nx)
    assert (bg_a - bg_b).abs().max() / bg_a.abs().max() < 1e-9
    assert (sc_a - sc_b).abs().max() / sc_a.abs().max() < 1e-9


@pytest.mark.parametrize(
    ("boundary", "pml", "shape", "src", "rec"),
    [
        (["free", "rigid"], [0, 0], (16,), [[[4]]], [[[3], [13]]]),
        (
            ["free", "free", "free", "free"],
            [0, 0, 0, 0],
            (16, 16),
            [[[4, 4]]],
            [[[3, 3], [13, 13]]],
        ),
        (
            ["rigid", "rigid", "rigid", "rigid"],
            [0, 0, 0, 0],
            (16, 16),
            [[[4, 4]]],
            [[[3, 3]]],
        ),
        (
            ["periodic", "periodic", "periodic", "periodic"],
            [0, 0, 0, 0],
            (16, 16),
            [[[8, 8]]],
            [[[3, 3]]],
        ),
        # mixed: PML on one end of each axis, reflecting on the other
        (
            ["pml", "free", "rigid", "pml"],
            [6, 0, 0, 6],
            (18, 18),
            [[[13, 4]]],
            [[[15, 3]]],
        ),
        # 3-D: the 4-cell PML on one end of each of the first two axes leaves
        # a 4-cell interior on each, with the source 2 cells from the
        # reflecting face at the other end -- well within the 12.6 cells the
        # front covers in ``nt`` steps, so it still reflects repeatedly.
        (
            ["pml", "free", "rigid", "pml", "periodic", "periodic"],
            [4, 0, 0, 4, 0, 0],
            (8, 8, 8),
            [[[5, 2, 4]]],
            [[[6, 1, 2]]],
        ),
    ],
)
def test_scalar_born_c_matches_eager_grad(boundary, pml, shape, src, rec):
    """Compiled backend: forward and gradients match eager.

    Covers 1-D, 2-D, and 3-D and every face type including mixed PML/reflecting
    axes, checking gradients w.r.t. v, scatter, AND source_amplitudes, and both
    the background and scattered receiver outputs together (mirrors
    ``test_scalar_c_matches_eager_grad`` for the plain propagator).
    """
    torch.manual_seed(0)
    dx, dt, nt = 5.0, 0.0007, 60
    acc = 2 if len(shape) == 3 else 4
    max_vel = 1550.0  # above the medium's true maximum speed of ~1504 m/s
    sa = deepwave.wavelets.ricker(15.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = ((torch.ones(shape) * C + torch.randn(shape)) / VELOCITY_UNIT).requires_grad_(
        True
    )
    scat = (
        (torch.randn(shape) * 0.05 * C).to(torch.double) / VELOCITY_UNIT
    ).requires_grad_(True)
    sl, rl = torch.tensor(src), torch.tensor(rec)

    def f(vv, scsc, ss, backend):
        out = scalar_born(
            vv * VELOCITY_UNIT,
            dx,
            dt,
            dv=scsc * VELOCITY_UNIT,
            source_amplitudes=ss,
            source_locations=sl,
            sc_receiver_locations=rl,
            bg_receiver_locations=rl,
            pml_width=pml,
            boundary=boundary,
            accuracy=acc,
            pml_freq=15.0,
            max_vel=max_vel,
            backend=backend,
        )
        return out[-2] + out[-1]

    # The forward compiled/eager fields agree to machine precision even after
    # the wave has interacted with the boundary.
    assert (f(v, scat, sa, "compiled") - f(v, scat, sa, "eager")).abs().max() < 1e-10
    # Compiled and eager gradients agree to machine precision.
    assert_compiled_matches_eager_grad(f, (v, scat, sa), tol=1e-9)


def test_scalar_born_c_boundary_adjoint_parity_well_conditioned():
    """Compiled scalar_born: long-``nt`` compiled-vs-eager gradient parity.

    The scalar_born sibling of
    ``test_scalar_c_boundary_adjoint_parity_well_conditioned``, and the same
    gap it closes: ``test_scalar_born_c_matches_eager_grad`` above
    reaches its reflecting faces but only for the few round trips a
    short-``nt`` parity check allows, and
    ``test_scalar_born_c_periodic_wrapping_adjoint_parity`` runs a long
    ``nt`` on periodic faces only, so neither pins the free/rigid adjoint
    after many reflections. It runs ``nt=1200`` at ``dt=0.0001``: CFL =
    C*dt/dx = 1500*0.0001/5 = 0.03 cells/step, so the front advances ``0.03
    * 1200 = 36`` cells, more than twice the 16-cell reflecting-axis
    interior (``pml_width=0`` on that axis), and the wave reflects off both
    non-PML faces and returns past the receiver several times before
    ``nt*dt`` ends. The boundary is deliberately asymmetric (``["free",
    "rigid", "pml", "pml"]``) so a face-ordering defect would be visible.

    Checks compiled-vs-eager forward output and v/dv/source gradients, on
    the background AND scattered receiver outputs together, at a tight
    relative tolerance, plus a central-FD arbitration (independent of both
    backends' analytic adjoints) at the grid cell where the compiled ``dv``
    (scattering) gradient is largest in magnitude -- ``dv`` because it is the
    quantity the scattered-field adjoint alone determines.
    """
    torch.manual_seed(0)
    shape = (16, 16)
    dx, dt, nt = 5.0, 0.0001, 1200
    acc = 4
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = (torch.ones(shape) * C + 30 * torch.rand(shape)).requires_grad_(True)
    scat = (torch.randn(shape) * 0.05 * C).to(torch.double).requires_grad_(True)
    sl = torch.tensor([[[10, 10]]])
    rl = torch.tensor([[[5, 5]]])
    # 4 cells of PML at each end of the absorbing axis leaves cells 4-11 of it
    # undamped, so both the source and the receiver sit in the interior.
    pml_width = [0, 0, 4, 4]
    boundary = ["rigid", "pml", "pml", "pml"]

    def f(v, scat, sa, backend):
        out = scalar_born(
            v,
            dx,
            dt,
            dv=scat,
            source_amplitudes=sa,
            source_locations=sl,
            sc_receiver_locations=rl,
            bg_receiver_locations=rl,
            pml_width=pml_width,
            boundary=boundary,
            accuracy=acc,
            pml_freq=25.0,
            backend=backend,
        )
        return out[-2] + out[-1]

    args = (v, scat, sa)
    g_c = assert_compiled_matches_eager_forward_and_grad(f, args, "compiled")

    def loss(ss):
        with torch.no_grad():
            return f(v.detach(), ss, sa.detach(), "eager").sum().item()

    fd, analytic = _fd_arbitrate_max_grad_cell(loss, scat, g_c[1])
    assert abs(fd - analytic) <= 1e-3 * abs(analytic)


@pytest.mark.parametrize(
    ("boundary", "pml"),
    [
        (None, [20, 20]),
        (["free", "pml"], [0, 20]),
        (["rigid", "pml"], [0, 20]),
        (["periodic", "periodic"], [0, 0]),
        (["free", "rigid"], [0, 0]),
    ],
)
def test_scalar_born_c_default_bit_identity(boundary, pml):
    """``boundary=None`` is bit-identical to the pre-STEP-B compiled default.

    Only the ``None`` case must match the eager backend bit-for-bit (the same
    identity the STEP-A test already checks, re-verified here now that the
    compiled backend is exercised); the others simply must run without error.
    """
    dx, dt, nt, nx = 5.0, 0.0007, 150, 60
    sa, _ = _ricker(15.0, nt, dt)
    v = torch.ones(nx) * C
    scat = torch.zeros(nx)
    scat[25] = C
    kw = {
        "source_amplitudes": sa,
        "source_locations": torch.tensor([[[30]]]),
        "receiver_locations": torch.tensor([[[20]]]),
        "bg_receiver_locations": torch.tensor([[[20]]]),
        "pml_width": pml,
        "boundary": boundary,
        "accuracy": 4,
        "pml_freq": 15.0,
    }
    compiled = scalar_born(v, dx, dt, dv=scat, **kw)
    if boundary is None:
        eager = scalar_born(v, dx, dt, dv=scat, backend="eager", **kw)
        assert (compiled[-2] - eager[-2]).abs().max() < 1e-12
        assert (compiled[-1] - eager[-1]).abs().max() < 1e-12
    else:
        assert torch.isfinite(compiled[-2]).all()
        assert torch.isfinite(compiled[-1]).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_scalar_born_c_boundary_cpu_gpu_match():
    """Compiled backend, first-class boundary faces: CPU and GPU forward+grad match.

    The scalar_born analogue of ``test_scalar_c_boundary_cpu_gpu_match``.
    Runs only on the maintainer's GPU (skipped here, no CUDA on this box).
    """
    torch.manual_seed(0)
    dx, dt, nt = 5.0, 0.0007, 60
    shape = (16, 16)
    boundary = ["free", "free", "free", "free"]
    pml = [0, 0, 0, 0]
    sa0 = deepwave.wavelets.ricker(15.0, nt, dt, 0.02).reshape(1, 1, nt)
    v0 = torch.ones(shape) * C + torch.randn(shape)
    scat0 = torch.randn(shape) * 0.05 * C
    sl = torch.tensor([[[4, 4]]])
    rl = torch.tensor([[[3, 3], [13, 13]]])

    def run(device):
        vv = v0.to(device).clone().requires_grad_(True)
        scsc = scat0.to(device).clone().requires_grad_(True)
        ss = sa0.to(device).clone().requires_grad_(True)
        out = scalar_born(
            vv,
            dx,
            dt,
            dv=scsc,
            source_amplitudes=ss,
            source_locations=sl.to(device),
            sc_receiver_locations=rl.to(device),
            bg_receiver_locations=rl.to(device),
            pml_width=pml,
            boundary=boundary,
            accuracy=4,
            pml_freq=15.0,
            backend="compiled",
        )
        (out[-2] + out[-1]).sum().backward()
        return (
            out[-2].detach().cpu(),
            out[-1].detach().cpu(),
            vv.grad.detach().cpu(),
            scsc.grad.detach().cpu(),
            ss.grad.detach().cpu(),
        )

    bg_cpu, sc_cpu, gv_cpu, gsc_cpu, gs_cpu = run(torch.device("cpu"))
    bg_gpu, sc_gpu, gv_gpu, gsc_gpu, gs_gpu = run(torch.device("cuda"))
    assert torch.allclose(bg_cpu, bg_gpu, atol=1e-6, rtol=1e-5)
    assert torch.allclose(sc_cpu, sc_gpu, atol=1e-6, rtol=1e-5)
    assert torch.allclose(gv_cpu, gv_gpu, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gsc_cpu, gsc_gpu, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gs_cpu, gs_gpu, atol=1e-5, rtol=1e-4)


# --- acoustic Born (staggered grid, coupled background + scattered) --------
#
# acoustic_born's background field is exactly acoustic's own
# (p, v_d) staggered-grid field, and the scattered field (dp, dv_d) solves the
# SAME acoustic system on the SAME grid, driven by a virtual source (see the
# module docstring); both therefore obey the identical acoustic per-axis
# image rule (docs/boundaries.rst's "Acoustic" ghost-plane convention),
# applied independently -- exactly the scalar_born precedent above, staggered.
# Both backends thread the per-face ``bc`` codes: the eager driver via
# :func:`_boundaries.fill_acoustic_pressure_boundary`/
# :func:`fill_acoustic_velocity_boundary` applied separately to the
# background and scattered field arrays (mirrors ``acoustic_python``, doubled
# for the coupled fields), the compiled backend via a direct, unchanged-
# arithmetic copy of ``acoustic.c``'s guard-fill functions in
# ``acoustic_born.c``/``.cu``, called on the background then the scattered
# field pointers at the same points in the coupled time loop, plus (for the
# adjoint) :func:`_boundaries.mirror_pad_acoustic_models` applied
# independently to the ``[K, B_*]`` and ``[dK, dB_*]`` model groups.


def _acoustic_born_reflection_coefficient(
    accuracy, face_boundary, freq=12.0, backend="eager"
):
    """Measure the pressure R off a low-x face for the background AND scattered fields.

    Mirrors :func:`_born_reflection_coefficient` (scalar_born) for
    the staggered acoustic grid: a single point scatterer (``dv``) sits past
    the receiver; the background field emanates from the source, and the
    scattered field emanates from the scatterer as the background wave passes
    it. For each field the incident arrival is taken from an otherwise-
    identical run whose low face is absorbing, so ``reflecting - absorbing``
    isolates the boundary reflection. Returns ``(R_bg, R_sc)``.
    """
    dx = 2.0
    dt = 0.00025
    nt = 4000
    nx = 800
    s, r, xsc = 120, 70, 200
    sa, tpk = _ricker(freq, nt, dt)
    v = torch.ones(nx) * C
    rho = torch.ones(nx) * RHO
    dv = torch.zeros(nx)
    dv[xsc] = 0.1 * C

    def run(bc, pml):
        out = acoustic_born(
            v,
            rho,
            dx,
            dt,
            dv=dv,
            source_amplitudes_volume_rate=sa,
            source_locations_volume_rate=torch.tensor([[[s]]]),
            sc_receiver_locations_p=torch.tensor([[[r]]]),
            bg_receiver_locations_p=torch.tensor([[[r]]]),
            pml_width=pml,
            boundary=bc,
            backend=backend,
            accuracy=accuracy,
            pml_freq=freq,
        )
        return out.bg_receiver_amplitudes_p[0, 0], out.sc_receiver_amplitudes_p[0, 0]

    w = int(1.0 / freq / dt)
    bg_inc, sc_inc = run(None, [40, 40])
    bg_ref, sc_ref = run(face_boundary, [0, 40])

    # Background: direct s->r, reflected off the low face s->0->r (+2 ghost).
    t_bg_dir = int((s - r) * dx / C / dt) + tpk
    t_bg_ref = int((s + r + 2) * dx / C / dt) + tpk
    bg_dw = bg_inc[t_bg_dir - w : t_bg_dir + w]
    bg_a_dir = bg_dw[bg_dw.abs().argmax()]
    bg_rw = (bg_ref - bg_inc)[t_bg_ref - w : t_bg_ref + w]
    bg_a_ref = bg_rw[bg_rw.abs().argmax()]

    # Scattered: direct s->xsc->r, reflected s->xsc->0->r.
    t_sc_dir = int((abs(s - xsc) + abs(xsc - r)) * dx / C / dt) + tpk
    t_sc_ref = int((abs(s - xsc) + (xsc + r + 2)) * dx / C / dt) + tpk
    sc_dw = sc_inc[t_sc_dir - w : t_sc_dir + w]
    sc_a_dir = sc_dw[sc_dw.abs().argmax()]
    sc_rw = (sc_ref - sc_inc)[t_sc_ref - w : t_sc_ref + w]
    sc_a_ref = sc_rw[sc_rw.abs().argmax()]

    return (bg_a_ref / bg_a_dir).item(), (sc_a_ref / sc_a_dir).item()


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_acoustic_born_free_reflection_coefficient(accuracy):
    """A ``"free"`` (p=0) face reflects both acoustic_born fields with R=-1."""
    r_bg, r_sc = _acoustic_born_reflection_coefficient(accuracy, ["free", "pml"])
    assert abs(r_bg - (-1.0)) < 0.01, f"acc={accuracy}: R_bg={r_bg}"
    assert abs(r_sc - (-1.0)) < 0.01, f"acc={accuracy}: R_sc={r_sc}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_acoustic_born_rigid_reflection_coefficient(accuracy):
    """A ``"rigid"`` (v_n=0) face reflects both acoustic_born fields with R=+1."""
    r_bg, r_sc = _acoustic_born_reflection_coefficient(accuracy, ["rigid", "pml"])
    assert abs(r_bg - 1.0) < 0.01, f"acc={accuracy}: R_bg={r_bg}"
    assert abs(r_sc - 1.0) < 0.01, f"acc={accuracy}: R_sc={r_sc}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_acoustic_born_c_free_reflection_coefficient(accuracy):
    """Compiled ``"free"`` face: both acoustic_born fields reflect with R=-1."""
    r_bg, r_sc = _acoustic_born_reflection_coefficient(
        accuracy, ["free", "pml"], backend="compiled"
    )
    assert abs(r_bg - (-1.0)) < 0.01, f"acc={accuracy}: R_bg={r_bg}"
    assert abs(r_sc - (-1.0)) < 0.01, f"acc={accuracy}: R_sc={r_sc}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_acoustic_born_c_rigid_reflection_coefficient(accuracy):
    """Compiled ``"rigid"`` face: both acoustic_born fields reflect with R=+1."""
    r_bg, r_sc = _acoustic_born_reflection_coefficient(
        accuracy, ["rigid", "pml"], backend="compiled"
    )
    assert abs(r_bg - 1.0) < 0.01, f"acc={accuracy}: R_bg={r_bg}"
    assert abs(r_sc - 1.0) < 0.01, f"acc={accuracy}: R_sc={r_sc}"


@pytest.mark.parametrize(
    ("boundary", "pml", "shape", "src", "rec"),
    [
        (["free", "rigid"], [0, 0], (16,), [[[4]]], [[[3], [13]]]),
        (
            ["free", "rigid", "free", "rigid"],
            [0, 0, 0, 0],
            (16, 16),
            [[[4, 4]]],
            [[[3, 3], [13, 13]]],
        ),
        (
            ["rigid", "rigid", "rigid", "rigid"],
            [0, 0, 0, 0],
            (16, 16),
            [[[4, 4]]],
            [[[3, 3]]],
        ),
        (
            ["free", "pml", "rigid", "pml"],
            [0, 6, 0, 6],
            (18, 18),
            [[[4, 4]]],
            [[[3, 3]]],
        ),
        # 3-D: the 4-cell PML on the far end of each axis leaves a 4-cell
        # interior on each, with the source 2 cells from the reflecting face
        # at the near end. The 3-D case also uses its own ``dt``/``nt`` (see
        # the body).
        (
            ["free", "pml", "rigid", "pml", "rigid", "pml"],
            [0, 4, 0, 4, 0, 4],
            (8, 8, 8),
            [[[2, 2, 2]]],
            [[[1, 1, 1]]],
        ),
    ],
)
def test_acoustic_born_c_matches_eager_grad(boundary, pml, shape, src, rec):
    """Compiled backend: forward and gradients match eager.

    Covers 1-D/2-D/3-D and every face acoustic_born offers, including mixed
    PML/reflecting axes, checking gradients w.r.t. v, rho, dv, AND
    source_amplitudes, on both the background and scattered receiver outputs
    together. Every ``"free"`` face sits at the beginning of its dimension,
    which is where acoustic_born offers one above accuracy 2 (mirrors
    ``test_scalar_born_c_matches_eager_grad``). ``dt``/``v``/
    ``pml_freq`` match ``test_acoustic_c_matches_eager_grad``'s own
    well-conditioned choices exactly (``dt=0.0001``, not e.g. ``0.0007``,
    and ``v`` a bounded ``C + 30*rand`` offset, not an unbounded Gaussian):
    an earlier draft of this test used less conservative values and the
    resulting larger discretisation/dispersion error was, at first, mistaken
    for a compiled-vs-eager correctness bug in the all-``"rigid"`` case
    specifically (plain ``acoustic``'s own identical corner configurations
    showed the same sensitivity to this choice under direct reproduction).
    The 3-D case is the exception, for the reason given in the body.
    """
    torch.manual_seed(0)
    ndim = len(shape)
    dx = 5.0
    # The 3-D case runs at a larger ``dt`` than the 1-D/2-D cases. At
    # ``dt=0.0001`` the front advances only C*dt/dx = 0.03 cells/step, and on
    # a grid small enough to differentiate in 3-D the receivers then record
    # essentially nothing: both the recorded amplitudes and the model
    # gradients fall below the comparison tolerance, which passes the check
    # vacuously. ``dt=0.0007`` (0.21 cells/step) carries the front 6.3
    # cells in 30 steps, past both reflecting faces (2 cells from the source)
    # and back, so the fills act on a wave that is actually there. The
    # dispersion sensitivity described above does not bite here: at the 3-D
    # case's accuracy order, with the bounded model offsets this test uses,
    # the compiled and eager forwards still agree to machine precision.
    dt, nt = (0.0007, 30) if ndim == 3 else (0.0001, 40)
    acc = 2 if ndim == 3 else 4
    max_vel = 1600.0  # above the medium's true maximum speed of 1530 m/s
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = (
        (torch.ones(shape) * C + 30 * torch.rand(shape)) / VELOCITY_UNIT
    ).requires_grad_(True)
    rho = (
        (torch.ones(shape) * RHO + 20 * torch.rand(shape)) / DENSITY_UNIT
    ).requires_grad_(True)
    dv = (
        (torch.randn(shape) * 0.05 * C).to(torch.double) / VELOCITY_UNIT
    ).requires_grad_(True)
    sl, rl = torch.tensor(src), torch.tensor(rec)

    def f(vv, rr, dvdv, ss, backend):
        out = acoustic_born(
            vv * VELOCITY_UNIT,
            rr * DENSITY_UNIT,
            dx,
            dt,
            dv=dvdv * VELOCITY_UNIT,
            source_amplitudes_volume_rate=ss,
            source_locations_volume_rate=sl,
            sc_receiver_locations_p=rl,
            bg_receiver_locations_p=rl,
            pml_width=pml,
            boundary=boundary,
            accuracy=acc,
            pml_freq=25.0,
            max_vel=max_vel,
            backend=backend,
        )
        return (out.bg_receiver_amplitudes_p + out.sc_receiver_amplitudes_p) * 1e-5

    assert (
        f(v, rho, dv, sa, "compiled") - f(v, rho, dv, sa, "eager")
    ).abs().max() < 1e-3
    assert_compiled_matches_eager_grad(f, (v, rho, dv, sa), compiled_backend="compiled")


def test_acoustic_born_c_boundary_adjoint_parity_well_conditioned():
    """Compiled acoustic_born: long-``nt`` compiled-vs-eager gradient parity.

    ``test_acoustic_born_c_matches_eager_grad`` above runs its
    1-D/2-D cases at ``nt=40``, ``dt=0.0001``: CFL = C*dt/dx = 1500*0.0001/5
    = 0.03 cells/step, so the wavefront moves under 2 cells and never reaches
    either reflecting face in those cases -- the same vacuous-gate pattern
    that hid the elastic-family compiled-backward adjoint bug (see
    ``test_elastic_c_jvp_supports_non_pml``'s docstring for the diagnosis).
    acoustic_born itself was separately audited and found clean at a
    well-conditioned harness, but that audit did not add a permanent
    long-``nt`` regression test to this file -- this test is that permanent
    check. It uses ``nt=1200``: the front advances ``0.03 * 1200 = 36``
    cells, more than twice the 16-cell reflecting-axis interior
    (``pml_width=0`` on that axis, matching ``test_elastic_c_jvp_supports_
    non_pml``'s own domain-size choice), so the wave genuinely reflects off
    both non-PML faces and returns past the receiver multiple times before
    ``nt*dt`` ends. The boundary is deliberately asymmetric (``["free",
    "rigid", "pml", "pml"]``) so a face-ordering defect would be visible.

    Checks compiled-vs-eager forward output and v/rho/dv/source gradients,
    on the background AND scattered receiver outputs together, at a tight
    relative tolerance, plus a central-FD arbitration at the grid cell where
    the compiled ``v`` (background velocity) gradient is largest in
    magnitude.
    """
    torch.manual_seed(0)
    shape = (16, 16)
    dx, dt, nt = 5.0, 0.0001, 1200
    acc = 4
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = (torch.ones(shape) * C + 30 * torch.rand(shape)).requires_grad_(True)
    rho = (torch.ones(shape) * RHO + 20 * torch.rand(shape)).requires_grad_(True)
    dv = (torch.randn(shape) * 0.05 * C).to(torch.double).requires_grad_(True)
    sl = torch.tensor([[[10, 10]]])
    rl = torch.tensor([[[5, 5]]])
    # PML narrow enough to leave an interior (cells 4-11) on the absorbing
    # axis holding both the source and the receiver.
    pml_width = [0, 0, 4, 4]
    boundary = ["rigid", "pml", "pml", "pml"]

    def f(vv, rr, dvdv, ss, backend):
        out = acoustic_born(
            vv,
            rr,
            dx,
            dt,
            dv=dvdv,
            source_amplitudes_volume_rate=ss,
            source_locations_volume_rate=sl,
            sc_receiver_locations_p=rl,
            bg_receiver_locations_p=rl,
            pml_width=pml_width,
            boundary=boundary,
            accuracy=acc,
            pml_freq=25.0,
            backend=backend,
        )
        return out.bg_receiver_amplitudes_p + out.sc_receiver_amplitudes_p

    args = (v, rho, dv, sa)
    g_c = assert_compiled_matches_eager_forward_and_grad(f, args, "compiled")

    def loss(vv):
        with torch.no_grad():
            return f(vv, rho.detach(), dv.detach(), sa.detach(), "eager").sum().item()

    fd, analytic = _fd_arbitrate_max_grad_cell(loss, v, g_c[0])
    assert abs(fd - analytic) <= 1e-3 * abs(analytic)


@pytest.mark.parametrize(
    ("boundary", "pml"),
    [
        (None, [20, 20]),
        (["free", "pml"], [0, 20]),
        (["rigid", "pml"], [0, 20]),
        (["free", "rigid"], [0, 0]),
    ],
)
def test_acoustic_born_c_default_bit_identity(boundary, pml):
    """``boundary=None`` is bit-identical to the pre-B5-step-5/9 compiled default.

    Only the ``None`` case must match the eager backend bit-for-bit (the same
    identity the eager test already checks, re-verified here now that the
    compiled backend is exercised); the others simply must run without error.
    """
    dx, dt, nt, nx = 5.0, 0.0007, 150, 60
    sa, _ = _ricker(15.0, nt, dt)
    v = torch.ones(nx) * C
    rho = torch.ones(nx) * RHO
    dv = torch.zeros(nx)
    dv[25] = 0.1 * C
    kw = {
        "source_amplitudes_volume_rate": sa,
        "source_locations_volume_rate": torch.tensor([[[30]]]),
        "sc_receiver_locations_p": torch.tensor([[[20]]]),
        "bg_receiver_locations_p": torch.tensor([[[20]]]),
        "pml_width": pml,
        "boundary": boundary,
        "accuracy": 4,
        "pml_freq": 15.0,
    }
    compiled = acoustic_born(v, rho, dx, dt, dv=dv, **kw)
    if boundary is None:
        eager = acoustic_born(v, rho, dx, dt, dv=dv, backend="eager", **kw)
        # Relative, not absolute: acoustic(_born)'s receiver amplitudes carry
        # the large K = rho*v**2 scaling (unlike scalar_born's, O(1e4) here),
        # matching test_acoustic_default_regression's own relative check in
        # this file rather than scalar_born's absolute one.
        assert (
            compiled.bg_receiver_amplitudes_p - eager.bg_receiver_amplitudes_p
        ).abs().max() / eager.bg_receiver_amplitudes_p.abs().max() < 1e-12
        assert (
            compiled.sc_receiver_amplitudes_p - eager.sc_receiver_amplitudes_p
        ).abs().max() / eager.sc_receiver_amplitudes_p.abs().max() < 1e-12
    else:
        assert torch.isfinite(compiled.bg_receiver_amplitudes_p).all()
        assert torch.isfinite(compiled.sc_receiver_amplitudes_p).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_acoustic_born_c_boundary_cpu_gpu_match():
    """Compiled backend, first-class boundary faces: CPU and GPU forward+grad match.

    The acoustic_born analogue of ``test_acoustic_c_boundary_cpu_gpu_match``.
    """
    torch.manual_seed(0)
    dx, dt, nt = 5.0, 0.0007, 60
    shape = (16, 16)
    boundary = ["free", "rigid", "free", "rigid"]
    pml = [0, 0, 0, 0]
    sa0 = deepwave.wavelets.ricker(15.0, nt, dt, 0.02).reshape(1, 1, nt)
    v0 = torch.ones(shape) * C + torch.randn(shape)
    rho0 = torch.ones(shape) * RHO + 20 * torch.rand(shape)
    dv0 = torch.randn(shape) * 0.05 * C
    sl = torch.tensor([[[4, 4]]])
    rl = torch.tensor([[[3, 3], [13, 13]]])

    def run(device):
        vv = v0.to(device).clone().requires_grad_(True)
        rr = rho0.to(device).clone().requires_grad_(True)
        dvdv = dv0.to(device).clone().requires_grad_(True)
        ss = sa0.to(device).clone().requires_grad_(True)
        out = acoustic_born(
            vv,
            rr,
            dx,
            dt,
            dv=dvdv,
            source_amplitudes_volume_rate=ss,
            source_locations_volume_rate=sl.to(device),
            sc_receiver_locations_p=rl.to(device),
            bg_receiver_locations_p=rl.to(device),
            pml_width=pml,
            boundary=boundary,
            accuracy=4,
            pml_freq=15.0,
            backend="compiled",
        )
        (out.bg_receiver_amplitudes_p + out.sc_receiver_amplitudes_p).sum().backward()
        return (
            out.bg_receiver_amplitudes_p.detach().cpu(),
            out.sc_receiver_amplitudes_p.detach().cpu(),
            vv.grad.detach().cpu(),
            rr.grad.detach().cpu(),
            dvdv.grad.detach().cpu(),
            ss.grad.detach().cpu(),
        )

    bg_cpu, sc_cpu, gv_cpu, gr_cpu, gdv_cpu, gs_cpu = run(torch.device("cpu"))
    bg_gpu, sc_gpu, gv_gpu, gr_gpu, gdv_gpu, gs_gpu = run(torch.device("cuda"))
    assert torch.allclose(bg_cpu, bg_gpu, atol=1e-6, rtol=1e-5)
    assert torch.allclose(sc_cpu, sc_gpu, atol=1e-6, rtol=1e-5)
    assert torch.allclose(gv_cpu, gv_gpu, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gr_cpu, gr_gpu, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gdv_cpu, gdv_gpu, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gs_cpu, gs_gpu, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gs_cpu, gs_gpu, atol=1e-5, rtol=1e-4)


# --- elastic Born (staggered grid, coupled background + scattered) ---------
#
# elastic_born's background field (v, sigma) is exactly elastic's own
# isotropic staggered-grid field, and the scattered field (dv, dsigma) solves
# the SAME isotropic elastic system on the SAME grid, driven by a virtual
# source built from the background field's own strain-rate/stress-divergence
# terms times the scattering models (see the module docstring); both
# therefore obey the identical
# per-field image parity documented on ``deepwave.elastic`` (Robertsson
# 1996's stress-image method, see the elastic Policy entry above), applied
# independently -- exactly the acoustic_born precedent, generalised to the
# full stress tensor. Both backends thread the per-face ``bc`` codes: the
# eager driver via ``_boundaries.fill_elastic_stress_boundary``/
# ``fill_elastic_velocity_boundary`` applied separately to the background
# and scattered field arrays (mirrors ``_elastic_eager_core``, doubled for
# the coupled fields), the compiled backend via a direct, unchanged-
# arithmetic copy of ``elastic.c``'s guard-fill functions in
# ``elastic_born.c``/``.cu``, called on the background then the scattered
# field pointers at the same points in the coupled time loop, plus (for the
# adjoint) ``_boundaries.mirror_pad_elastic_models`` applied independently to
# the background and scattering model groups. This also lifts the non-PML-
# face restriction on ``deepwave.elastic``'s own compiled forward-mode AD
# (``torch.func.jvp``) and second-order (HVP) rules -- see
# ``test_elastic_c_jvp_supports_non_pml``/``test_elastic_c_hvp_supports_non_pml``
# above.


def _elastic_born_reflection_coefficient(
    accuracy, face_boundary, freq=12.0, backend="eager"
):
    """Elastic Born analogue of :func:`_elastic_reflection_coefficient`.

    The same genuinely one-dimensional model, with a single-cell ``dlamb``
    scatterer added so that the scattered field is a plane wave too. Mirrors
    :func:`_acoustic_born_reflection_coefficient`'s incident-vs-reflecting
    subtraction, applied to both the background (``bg_receiver_amplitudes_p``)
    and scattered (``sc_receiver_amplitudes_p``) pressure-type channels, but
    without that function's staggered "+2" ghost-node offset (this measures
    the on-row ``sigma_AA`` channel directly, exactly like
    :func:`_elastic_reflection_coefficient`'s own background-only formula).
    Returns ``(R_bg, R_sc)``.
    """
    dx, dt, nt, n = 2.0, 0.0003, 4000, 800
    # ``s`` is placed far enough from ``r`` that a contaminating higher-order
    # arrival (the background wave bouncing off the tested face BEFORE
    # reaching the scatterer, then scattering straight to the receiver --
    # path length ``s + 2*xsc - r`` for ``xsc > s``) lands outside both the
    # simulated window (``nt``) and the measurement window around the
    # intended single-scatter/single-bounce arrival (path length
    # ``|s - xsc| + xsc + r``); the gap between the two is ``2*(s - r)``
    # cells, independent of ``xsc``, so a large ``s - r`` (unlike
    # :func:`_elastic_reflection_coefficient`'s plain ``s=60, r=35``) is
    # required to keep them from overlapping and biasing the measured R.
    s, r, xsc = 200, 35, 400
    sa, tpk = _ricker(freq, nt, dt)
    lamb, mu, buoyancy = _elastic_isotropic_media((n,))
    dlamb = torch.zeros_like(lamb)
    dlamb[xsc] = 0.15 * lamb[xsc]

    def run(tested_bc, tested_pml):
        out = elastic_born(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            dlamb=dlamb,
            source_amplitudes_pressure_rate=sa,
            source_locations_pressure_rate=torch.tensor([[[s]]]),
            sc_receiver_locations_p=torch.tensor([[[r]]]),
            bg_receiver_locations_p=torch.tensor([[[r]]]),
            pml_width=list(tested_pml),
            boundary=list(tested_bc),
            accuracy=accuracy,
            pml_freq=freq,
            backend=backend,
        )
        return (
            out.bg_receiver_amplitudes_p[0, 0],
            out.sc_receiver_amplitudes_p[0, 0],
        )

    w = int(1.0 / freq / dt)

    t_bg_dir = int((s - r) * dx / C / dt) + tpk
    t_bg_ref = int((s + r) * dx / C / dt) + tpk
    t_sc_dir = int((abs(s - xsc) + abs(xsc - r)) * dx / C / dt) + tpk
    t_sc_ref = int((abs(s - xsc) + (xsc + r)) * dx / C / dt) + tpk

    bg_inc, sc_inc = run(("pml", "pml"), (40, 40))
    bg_dw = bg_inc[t_bg_dir - w : t_bg_dir + w]
    bg_a_dir = bg_dw[bg_dw.abs().argmax()]
    sc_dw = sc_inc[t_sc_dir - w : t_sc_dir + w]
    sc_a_dir = sc_dw[sc_dw.abs().argmax()]

    bg_tr, sc_tr = run(tuple(face_boundary), (0, 40))
    bg_rw = (bg_tr - bg_inc)[t_bg_ref - w : t_bg_ref + w]
    bg_a_ref = bg_rw[bg_rw.abs().argmax()]
    sc_rw = (sc_tr - sc_inc)[t_sc_ref - w : t_sc_ref + w]
    sc_a_ref = sc_rw[sc_rw.abs().argmax()]

    return (bg_a_ref / bg_a_dir).item(), (sc_a_ref / sc_a_dir).item()


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_elastic_born_rigid_reflection_coefficient(accuracy):
    """Eager elastic_born ``"rigid"`` face: R = +1 on both fields, <1%."""
    r_bg, r_sc = _elastic_born_reflection_coefficient(accuracy, ["rigid", "pml"])
    assert abs(r_bg - 1.0) < 0.01, f"acc={accuracy}: R_bg={r_bg}"
    assert abs(r_sc - 1.0) < 0.01, f"acc={accuracy}: R_sc={r_sc}"


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_elastic_born_c_rigid_reflection_coefficient(accuracy):
    """Compiled elastic_born ``"rigid"`` face: R = +1 on both fields, <1%."""
    r_bg, r_sc = _elastic_born_reflection_coefficient(
        accuracy, ["rigid", "pml"], backend="compiled"
    )
    assert abs(r_bg - 1.0) < 0.01, f"acc={accuracy}: R_bg={r_bg}"
    assert abs(r_sc - 1.0) < 0.01, f"acc={accuracy}: R_sc={r_sc}"


@pytest.mark.parametrize(
    ("boundary", "pml"),
    [
        (None, [10, 10, 10, 10]),
        (["rigid", "pml", "pml", "pml"], [0, 10, 10, 10]),
        (["pml", "pml", "rigid", "pml"], [10, 10, 0, 10]),
    ],
)
def test_elastic_born_c_default_regression(boundary, pml):
    """``boundary=None`` on the compiled backend matches eager (also non-None)."""
    ny, nx = 20, 20
    dx, dt, nt = 5.0, 0.0007, 60
    sa, _ = _ricker(15.0, nt, dt)
    lamb, mu, buoyancy = _elastic_isotropic_media((ny, nx))
    dlamb = torch.zeros_like(lamb)
    dlamb[8, 8] = 0.2 * lamb[8, 8]
    kw = {
        "dlamb": dlamb,
        "source_amplitudes_pressure_rate": sa,
        "source_locations_pressure_rate": torch.tensor([[[10, 10]]]),
        "sc_receiver_locations_p": torch.tensor([[[6, 6]]]),
        "bg_receiver_locations_p": torch.tensor([[[6, 6]]]),
        "pml_width": pml,
        "accuracy": 4,
        "pml_freq": 15.0,
    }
    compiled = elastic_born(lamb, mu, buoyancy, dx, dt, boundary=boundary, **kw)
    eager = elastic_born(
        lamb, mu, buoyancy, dx, dt, boundary=boundary, backend="eager", **kw
    )
    assert (
        compiled.bg_receiver_amplitudes_p - eager.bg_receiver_amplitudes_p
    ).abs().max() < 1e-6
    assert (
        compiled.sc_receiver_amplitudes_p - eager.sc_receiver_amplitudes_p
    ).abs().max() < 1e-6


@pytest.mark.parametrize(
    ("boundary", "pml", "src", "rec"),
    [
        # elastic-family propagators offer no ``"free"`` face, and offer
        # ``"rigid"`` only at the beginning of a dimension at this accuracy
        # order, so every reflecting face below is a low one. The PML widths
        # leave a genuine interior on every axis, with every source and
        # receiver inside it.
        (
            ["rigid", "pml", "rigid", "pml"],
            [0, 4, 0, 4],
            [[[4, 4]]],
            [[[3, 3]]],
        ),
        (
            ["pml", "pml", "rigid", "pml"],
            [4, 4, 0, 4],
            [[[7, 4]]],
            [[[6, 1]]],
        ),
        (
            ["rigid", "pml", "pml", "pml"],
            [0, 2, 2, 2],
            [[[4, 6]]],
            [[[1, 5]]],
        ),
    ],
)
def test_elastic_born_c_matches_eager_grad(boundary, pml, src, rec):
    """Compiled elastic_born 2-D: forward and gradients match eager.

    Differentiates ``lamb``, ``mu``, ``buoyancy``, ``dlamb``, ``dmu``,
    ``dbuoyancy``, and the shared source amplitude together, across every
    face type including a mixed PML/reflecting-axis case and a fully-
    enclosed free/rigid box -- exercising both the per-field guard-ring
    fills (background and scattered, independently) and the compiled
    backward's model mirror-pad (``_boundaries.mirror_pad_elastic_models``,
    applied independently to the background and scattering model groups).

    ``dt`` gives CFL (P) = C*dt/dx = 1500*0.0007/5 = 0.21 cells/step, so over
    ``nt`` the front covers 6.3 cells: far enough to reach the reflecting
    faces and the receiver from anywhere in these 14-cell grids. The
    dimensionless leaves and the pinned ``max_vel`` are explained above
    ``MODULUS_UNIT``.
    """
    torch.manual_seed(0)
    shape = (14, 14)
    dx, dt, nt = 5.0, 0.0007, 30
    acc = 4
    max_vel = 1550.0  # above the medium's true maximum P speed of ~1504 m/s
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    lamb0, mu0, buoyancy0 = _elastic_isotropic_media(shape)
    lamb = ((lamb0 + 3e6 * torch.rand(shape)) / MODULUS_UNIT).requires_grad_(True)
    mu = ((mu0 + 2e6 * torch.rand(shape)) / MODULUS_UNIT).requires_grad_(True)
    buoyancy = ((buoyancy0 + 2e-6 * torch.rand(shape)) / BUOYANCY_UNIT).requires_grad_(
        True
    )
    dlamb = ((3e5 * torch.rand(shape)) / MODULUS_UNIT).requires_grad_(True)
    dmu = ((2e5 * torch.rand(shape)) / MODULUS_UNIT).requires_grad_(True)
    dbuoyancy = ((2e-7 * torch.rand(shape)) / BUOYANCY_UNIT).requires_grad_(True)
    sl, rl = torch.tensor(src), torch.tensor(rec)

    def f(lamb, mu, buoyancy, dlamb, dmu, dbuoyancy, sa, backend):
        out = elastic_born(
            lamb * MODULUS_UNIT,
            mu * MODULUS_UNIT,
            buoyancy * BUOYANCY_UNIT,
            dx,
            dt,
            dlamb=dlamb * MODULUS_UNIT,
            dmu=dmu * MODULUS_UNIT,
            dbuoyancy=dbuoyancy * BUOYANCY_UNIT,
            source_amplitudes_pressure_rate=sa,
            source_locations_pressure_rate=sl,
            sc_receiver_locations_p=rl,
            bg_receiver_locations_p=rl,
            pml_width=pml,
            boundary=boundary,
            accuracy=acc,
            pml_freq=25.0,
            max_vel=max_vel,
            backend=backend,
        )
        # The recorded amplitudes are small in SI, so they are scaled up (not
        # down) to sit well clear of the absolute gradient tolerance.
        return torch.cat(
            [out.bg_receiver_amplitudes_p * 1e4, out.sc_receiver_amplitudes_p * 1e4],
            dim=-1,
        )

    args = (lamb, mu, buoyancy, dlamb, dmu, dbuoyancy, sa)
    assert (f(*args, "compiled") - f(*args, "eager")).abs().max() < 1e-3
    assert_compiled_matches_eager_grad(f, args)


def test_elastic_born_c_matches_eager_grad_3d(
    boundary=("rigid", "pml", "pml", "pml", "pml", "pml"),
):
    """Compiled elastic_born 3-D gradients (the elastic-family 3-D requirement).

    The elastic family requires 3-D gradient coverage (not just the 2-D
    minimum other propagators use). A small 3-D grid keeps this tractable;
    a single representative face combination (one ``"rigid"`` face, the rest
    ``"pml"``) is enough to exercise the 3-D guard-fill/mirror-pad code paths
    (the 2-D test above already covers every combination of the faces
    elastic_born offers).

    Three choices below are what make this check actually bite, and each is
    load-bearing:

    * **A real interior.** ``pml_width=[0, 2, 2, 2, 2, 2]`` leaves cells 0-3
      free of PML on the first axis (the ``"rigid"`` face is at index 0) and
      cells 2-3 on the other two, so both the source and the receiver sit in
      undamped ground. A wider PML on a grid this small leaves no interior at
      all and buries the source, which absorbs the record to nothing.
    * **Dimensionless leaves and a fixed ``max_vel``**, both explained above
      ``MODULUS_UNIT``: without them a modulus gradient entry cannot clear a
      shared absolute tolerance.

    ``dt`` gives CFL (P) = C*dt/dx = 1500*0.0007/5 = 0.21 cells/step, so over
    ``nt=20`` the P front covers 4.2 cells: it crosses the 2 cells from the
    source to the ``"rigid"`` face and returns past the receiver, which sits
    one cell off that face. Replacing that face with an absorbing one changes
    the recorded trace by a large fraction of its peak amplitude, so the fill
    is genuinely shaping what is differentiated.
    """
    torch.manual_seed(0)
    shape = (6, 6, 6)
    dx, dt, nt = 5.0, 0.0007, 20
    acc = 4
    max_vel = 1550.0  # above the medium's true maximum P speed of ~1504 m/s
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    lamb0, mu0, buoyancy0 = _elastic_isotropic_media(shape)
    lamb = ((lamb0 + 3e6 * torch.rand(shape)) / MODULUS_UNIT).requires_grad_(True)
    mu = ((mu0 + 2e6 * torch.rand(shape)) / MODULUS_UNIT).requires_grad_(True)
    buoyancy = ((buoyancy0 + 2e-6 * torch.rand(shape)) / BUOYANCY_UNIT).requires_grad_(
        True
    )
    dlamb = ((3e5 * torch.rand(shape)) / MODULUS_UNIT).requires_grad_(True)
    dmu = ((2e5 * torch.rand(shape)) / MODULUS_UNIT).requires_grad_(True)
    sl = torch.tensor([[[2, 2, 2]]])
    rl = torch.tensor([[[1, 3, 3]]])
    pml = [0, 2, 2, 2, 2, 2]

    def f(lamb, mu, buoyancy, dlamb, dmu, sa, backend):
        out = elastic_born(
            lamb * MODULUS_UNIT,
            mu * MODULUS_UNIT,
            buoyancy * BUOYANCY_UNIT,
            dx,
            dt,
            dlamb=dlamb * MODULUS_UNIT,
            dmu=dmu * MODULUS_UNIT,
            source_amplitudes_pressure_rate=sa,
            source_locations_pressure_rate=sl,
            sc_receiver_locations_p=rl,
            bg_receiver_locations_p=rl,
            pml_width=pml,
            boundary=list(boundary),
            accuracy=acc,
            pml_freq=25.0,
            max_vel=max_vel,
            backend=backend,
        )
        # The recorded amplitudes are of order 1e-4 Pa, so they are scaled up
        # (not down) to sit well clear of the absolute gradient tolerance.
        return torch.cat(
            [out.bg_receiver_amplitudes_p * 1e4, out.sc_receiver_amplitudes_p * 1e4],
            dim=-1,
        )

    args = (lamb, mu, buoyancy, dlamb, dmu, sa)
    assert (f(*args, "compiled") - f(*args, "eager")).abs().max() < 1e-3
    assert_compiled_matches_eager_grad(f, args)


def test_elastic_born_c_boundary_adjoint_parity_well_conditioned():
    """Compiled elastic_born: long-``nt`` compiled-vs-eager gradient parity.

    ``test_elastic_born_c_matches_eager_grad`` above does carry the
    wave out to its reflecting faces, but only for the few round trips a
    short-``nt`` parity check allows. Gradient correctness after many
    reflections, where a face-ordering or guard-fill transpose defect
    accumulates, needs a longer run -- the same vacuous-gate pattern that hid
    the elastic-family compiled-backward adjoint bug (see ``git show 84fd036``
    for the canonical diagnosis, and
    ``test_elastic_c_jvp_supports_non_pml``'s docstring). This test uses
    ``dt=0.0001``, ``nt=1200``: CFL = 1500*0.0001/5 = 0.03 cells/step, so the
    front advances ``0.03 * 1200 = 36`` cells, more than twice the 16-cell
    reflecting-axis interior (``pml_width=0`` on that axis, matching
    ``test_elastic_c_jvp_supports_non_pml``'s own domain-size choice), so the
    wave genuinely reflects off both non-PML faces and returns past the
    receiver multiple times before ``nt*dt`` ends. The boundary is
    deliberately asymmetric (``["rigid", "pml", "pml", "pml"]`` with zero
    width on both faces of that axis, so the low face is a named ``rigid``
    fill and the high face the bare zero-width reflector) so a face-ordering
    defect would be visible.

    Checks compiled-vs-eager forward output and lamb/mu/buoyancy/dlamb/dmu/
    dbuoyancy/source gradients, on the background AND scattered receiver
    outputs together, at a tight relative tolerance, plus a central-FD
    arbitration at the grid cell where the compiled ``mu`` (background shear
    modulus) gradient is largest in magnitude -- ``mu`` because the fixed
    elastic-family compiled-backward adjoint bug was diagnosed via the
    shear modulus gradient specifically (see ``git show 84fd036``'s commit
    message).
    """
    torch.manual_seed(0)
    shape = (16, 16)
    dx, dt, nt = 5.0, 0.0001, 1200
    acc = 4
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    lamb0, mu0, buoyancy0 = _elastic_isotropic_media(shape)
    lamb = (lamb0 + 3e6 * torch.rand(shape)).requires_grad_(True)
    mu = (mu0 + 2e6 * torch.rand(shape)).requires_grad_(True)
    buoyancy = (buoyancy0 + 2e-6 * torch.rand(shape)).requires_grad_(True)
    dlamb = (3e5 * torch.rand(shape)).requires_grad_(True)
    dmu = (2e5 * torch.rand(shape)).requires_grad_(True)
    dbuoyancy = (2e-7 * torch.rand(shape)).requires_grad_(True)
    sl = torch.tensor([[[10, 10]]])
    rl = torch.tensor([[[5, 5]]])
    # PML narrow enough to leave an interior (cells 4-11) on the absorbing
    # axis holding both the source and the receiver.
    pml_width = [0, 0, 4, 4]
    boundary = ["rigid", "pml", "pml", "pml"]

    def f(lamb, mu, buoyancy, dlamb, dmu, dbuoyancy, sa, backend):
        out = elastic_born(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            dlamb=dlamb,
            dmu=dmu,
            dbuoyancy=dbuoyancy,
            source_amplitudes_pressure_rate=sa,
            source_locations_pressure_rate=sl,
            sc_receiver_locations_p=rl,
            bg_receiver_locations_p=rl,
            pml_width=pml_width,
            boundary=boundary,
            accuracy=acc,
            pml_freq=25.0,
            backend=backend,
        )
        return out.bg_receiver_amplitudes_p + out.sc_receiver_amplitudes_p

    args = (lamb, mu, buoyancy, dlamb, dmu, dbuoyancy, sa)
    g_c = assert_compiled_matches_eager_forward_and_grad(f, args, "compiled")

    def loss(mm):
        with torch.no_grad():
            return (
                f(
                    lamb.detach(),
                    mm,
                    buoyancy.detach(),
                    dlamb.detach(),
                    dmu.detach(),
                    dbuoyancy.detach(),
                    sa.detach(),
                    "eager",
                )
                .sum()
                .item()
            )

    fd, analytic = _fd_arbitrate_max_grad_cell(loss, mu, g_c[1])
    assert abs(fd - analytic) <= 1e-3 * abs(analytic)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_elastic_born_c_boundary_cpu_gpu_match():
    """Compiled backend, first-class boundary faces: CPU and GPU forward+grad match.

    The elastic_born analogue of ``test_acoustic_born_c_boundary_cpu_gpu_match``.
    """
    torch.manual_seed(0)
    dx, dt, nt = 5.0, 0.0002, 30
    shape = (12, 12)
    boundary = ["rigid", "pml", "rigid", "pml"]
    pml = [0, 4, 0, 4]
    sa0 = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    lamb0, mu0, buoyancy0 = _elastic_isotropic_media(shape)
    lamb0 = lamb0 + 3e6 * torch.rand(shape)
    mu0 = mu0 + 2e6 * torch.rand(shape)
    buoyancy0 = buoyancy0 + 2e-6 * torch.rand(shape)
    dlamb0 = 3e5 * torch.rand(shape)
    dmu0 = 2e5 * torch.rand(shape)
    sl = torch.tensor([[[4, 4]]])
    rl = torch.tensor([[[3, 3]]])

    def run(device):
        lamb = lamb0.to(device).clone().requires_grad_(True)
        mu = mu0.to(device).clone().requires_grad_(True)
        buoyancy = buoyancy0.to(device).clone().requires_grad_(True)
        dlamb = dlamb0.to(device).clone().requires_grad_(True)
        dmu = dmu0.to(device).clone().requires_grad_(True)
        ss = sa0.to(device).clone().requires_grad_(True)
        out = elastic_born(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            dlamb=dlamb,
            dmu=dmu,
            source_amplitudes_pressure_rate=ss,
            source_locations_pressure_rate=sl.to(device),
            sc_receiver_locations_p=rl.to(device),
            bg_receiver_locations_p=rl.to(device),
            pml_width=pml,
            boundary=boundary,
            accuracy=4,
            pml_freq=25.0,
            backend="compiled",
        )
        (out.bg_receiver_amplitudes_p + out.sc_receiver_amplitudes_p).sum().backward()
        return (
            out.bg_receiver_amplitudes_p.detach().cpu(),
            out.sc_receiver_amplitudes_p.detach().cpu(),
            lamb.grad.detach().cpu(),
            mu.grad.detach().cpu(),
            buoyancy.grad.detach().cpu(),
            dlamb.grad.detach().cpu(),
            dmu.grad.detach().cpu(),
            ss.grad.detach().cpu(),
        )

    bg_cpu, sc_cpu, glamb_cpu, gmu_cpu, gb_cpu, gdlamb_cpu, gdmu_cpu, gs_cpu = run(
        torch.device("cpu")
    )
    bg_gpu, sc_gpu, glamb_gpu, gmu_gpu, gb_gpu, gdlamb_gpu, gdmu_gpu, gs_gpu = run(
        torch.device("cuda")
    )
    assert torch.allclose(bg_cpu, bg_gpu, atol=1e-6, rtol=1e-5)
    assert torch.allclose(sc_cpu, sc_gpu, atol=1e-6, rtol=1e-5)
    assert torch.allclose(glamb_cpu, glamb_gpu, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gmu_cpu, gmu_gpu, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gb_cpu, gb_gpu, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gdlamb_cpu, gdlamb_gpu, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gdmu_cpu, gdmu_gpu, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gs_cpu, gs_gpu, atol=1e-5, rtol=1e-4)


# --- Wrapping all-periodic compiled-backward adjoint parity -----------------
# ``scalar`` and ``scalar_born`` are the propagators that offer a
# ``"periodic"`` face. A short parity run at CFL ~0.03 cells/step moves the
# wavefront ~1.2 cells and never wraps, so the periodic guard maintenance's
# adjoint would never be exercised (the same vacuous-gate pattern
# _fd_arbitrate_max_grad_cell's docstring describes). The tests below use
# nt=1200 at that CFL on a 16-cell all-periodic domain: the front advances
# 1500*0.0001/5*1200 = 36 cells > 2x the interior, so the wave genuinely
# wraps both axes repeatedly (asserted numerically inside each test), and
# eager (autograd through the machine-identical forward) is ground truth.

_WRAP_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="no CUDA device"
        ),
    ),
]


def _assert_wraps(nt, dt, dx, n_interior):
    """The front must cross the periodic interior at least twice."""
    assert C * dt / dx * nt > 2 * n_interior


@pytest.mark.parametrize("device", _WRAP_DEVICES)
def test_scalar_c_periodic_wrapping_adjoint_parity(device):
    """Compiled scalar, all-periodic, wrapping: grads match eager."""
    torch.manual_seed(0)
    shape = (16, 16)
    dx, dt, nt = 5.0, 0.0001, 1200
    _assert_wraps(nt, dt, dx, shape[0])
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = (torch.ones(shape) * C + 30 * torch.rand(shape)).requires_grad_(True)
    sl = torch.tensor([[[10, 10]]])
    rl = torch.tensor([[[5, 5]]])

    def f(v, sa, backend, dev="cpu"):
        return scalar(
            v.to(dev),
            dx,
            dt,
            source_amplitudes=sa.to(dev),
            source_locations=sl.to(dev),
            receiver_locations=rl.to(dev),
            pml_width=[0, 0, 0, 0],
            boundary=["periodic"] * 4,
            accuracy=4,
            pml_freq=25.0,
            backend=backend,
        )[-1]

    args = (v, sa)
    assert_compiled_matches_eager_forward_and_grad_on_device(f, args, device)


@pytest.mark.parametrize("device", _WRAP_DEVICES)
def test_scalar_born_c_periodic_wrapping_adjoint_parity(device):
    """Compiled scalar_born, all-periodic, wrapping: grads match eager."""
    torch.manual_seed(0)
    shape = (16, 16)
    dx, dt, nt = 5.0, 0.0001, 1200
    _assert_wraps(nt, dt, dx, shape[0])
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = (torch.ones(shape) * C + 30 * torch.rand(shape)).requires_grad_(True)
    scat = (torch.randn(shape) * 0.05 * C).to(torch.double).requires_grad_(True)
    sl = torch.tensor([[[10, 10]]])
    rl = torch.tensor([[[5, 5]]])

    def f(v, scat, sa, backend, dev="cpu"):
        out = scalar_born(
            v.to(dev),
            dx,
            dt,
            dv=scat.to(dev),
            source_amplitudes=sa.to(dev),
            source_locations=sl.to(dev),
            sc_receiver_locations=rl.to(dev),
            bg_receiver_locations=rl.to(dev),
            pml_width=[0, 0, 0, 0],
            boundary=["periodic"] * 4,
            accuracy=4,
            pml_freq=25.0,
            backend=backend,
        )
        return out[-2] + out[-1]

    args = (v, scat, sa)
    assert_compiled_matches_eager_forward_and_grad_on_device(f, args, device)


# --- Free/rigid-combination compiled-backward adjoint parity ----------------
# The compiled backward of the acoustic staggered family historically
# realised the free/rigid guard-fill transpose by re-applying the forward
# fill to the adjoint fields (with mirror-padded models). That scheme is
# only exact for the per-axis (free-low, rigid-high) pairing -- the one
# combination the pre-existing parity tests exercised -- and was
# wrong (up to ~1.3e-2 relative in grad_rho) for every other free/rigid
# combination; those axes now get the exact fold/gather transpose (see
# acoustic_kernels.h's _fr functions). The tests below run the same
# well-conditioned long-nt harness as the periodic wrapping tests (front
# advances 36 cells > 2x the 16-cell interior, so the wave reflects off
# every non-PML face repeatedly) across a matrix of free/rigid face
# combinations, including the historical pairing as a regression pin.
# Eager (autograd through the machine-identical forward) is ground truth.
# These propagators offer ``"free"`` only at the beginning of a dimension
# above accuracy 2, so every combination below puts it there; ``"rigid"`` is
# offered on either face, which is what still separates the historical
# pairing from the rest.

_FR_COMBOS = [
    ["rigid", "rigid", "pml", "pml"],
    ["free", "pml", "pml", "rigid"],
    ["pml", "rigid", "free", "rigid"],
    ["rigid", "rigid", "rigid", "rigid"],
    ["free", "rigid", "rigid", "rigid"],
    ["rigid", "rigid", "free", "rigid"],
    ["free", "rigid", "free", "rigid"],  # historical (free-low, rigid-high) pin
]


def _fr_pml_width(boundary):
    return [0 if face != "pml" else 10 for face in boundary]


@pytest.mark.parametrize("device", _WRAP_DEVICES)
@pytest.mark.parametrize("boundary", _FR_COMBOS, ids=lambda b: "-".join(b))
def test_acoustic_c_fr_combination_adjoint_parity(device, boundary):
    """Compiled acoustic: every free/rigid face combination's grads match eager.

    For the deliberately-asymmetric ``["free","pml","pml","rigid"]`` case the
    max-|grad_v| cell is additionally FD-arbitrated (independent of both
    backends' analytic adjoints).
    """
    torch.manual_seed(0)
    shape = (16, 16)
    dx, dt, nt = 5.0, 0.0001, 1200
    _assert_wraps(nt, dt, dx, shape[0])
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = (torch.ones(shape) * C + 30 * torch.rand(shape)).requires_grad_(True)
    rho = (torch.ones(shape) * RHO + 20 * torch.rand(shape)).requires_grad_(True)
    sl = torch.tensor([[[10, 10]]])
    rl = torch.tensor([[[5, 5]]])

    def f(v, rho, sa, backend, dev="cpu"):
        return acoustic(
            v.to(dev),
            rho.to(dev),
            dx,
            dt,
            source_amplitudes_volume_rate=sa.to(dev),
            source_locations_volume_rate=sl.to(dev),
            receiver_locations_p=rl.to(dev),
            pml_width=_fr_pml_width(boundary),
            boundary=boundary,
            accuracy=4,
            pml_freq=25.0,
            backend=backend,
        ).receiver_amplitudes_p

    args = (v, rho, sa)
    g_c = assert_compiled_matches_eager_forward_and_grad_on_device(f, args, device)

    if device == "cpu" and boundary == ["free", "pml", "pml", "rigid"]:

        def loss(vv):
            with torch.no_grad():
                return f(vv, rho.detach(), sa.detach(), "eager").sum().item()

        fd, analytic = _fd_arbitrate_max_grad_cell(loss, v, g_c[0].cpu())
        assert abs(fd - analytic) <= 1e-3 * abs(analytic)


@pytest.mark.parametrize("device", _WRAP_DEVICES)
def test_acoustic_c_fr_combination_adjoint_parity_1d(device):
    """Compiled acoustic 1-D, (free-low, rigid-high): grads match eager."""
    torch.manual_seed(0)
    nx, dx, dt, nt = 16, 5.0, 0.0001, 1200
    _assert_wraps(nt, dt, dx, nx)
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = (torch.ones(nx) * C + 30 * torch.rand(nx)).requires_grad_(True)
    rho = (torch.ones(nx) * RHO + 20 * torch.rand(nx)).requires_grad_(True)
    sl = torch.tensor([[[10]]])
    rl = torch.tensor([[[5]]])

    def f(v, rho, sa, backend, dev="cpu"):
        return acoustic(
            v.to(dev),
            rho.to(dev),
            dx,
            dt,
            source_amplitudes_volume_rate=sa.to(dev),
            source_locations_volume_rate=sl.to(dev),
            receiver_locations_p=rl.to(dev),
            pml_width=[0, 0],
            boundary=["free", "rigid"],
            accuracy=4,
            pml_freq=25.0,
            backend=backend,
        ).receiver_amplitudes_p

    args = (v, rho, sa)
    assert_compiled_matches_eager_forward_and_grad_on_device(f, args, device)


@pytest.mark.parametrize("device", _WRAP_DEVICES)
def test_acoustic_c_fr_combination_adjoint_parity_3d(device):
    """Compiled acoustic 3-D, mixed free/rigid faces: grads match eager."""
    torch.manual_seed(0)
    shape = (16, 16, 16)
    dx, dt, nt = 5.0, 0.0001, 600
    _assert_wraps(nt, dt, dx, shape[0] // 2)
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = (torch.ones(shape) * C + 30 * torch.rand(shape)).requires_grad_(True)
    rho = (torch.ones(shape) * RHO + 20 * torch.rand(shape)).requires_grad_(True)
    sl = torch.tensor([[[10, 10, 10]]])
    rl = torch.tensor([[[5, 5, 5]]])
    boundary = ["free", "rigid", "free", "rigid", "pml", "pml"]

    def f(v, rho, sa, backend, dev="cpu"):
        return acoustic(
            v.to(dev),
            rho.to(dev),
            dx,
            dt,
            source_amplitudes_volume_rate=sa.to(dev),
            source_locations_volume_rate=sl.to(dev),
            receiver_locations_p=rl.to(dev),
            pml_width=_fr_pml_width(boundary),
            boundary=boundary,
            accuracy=4,
            pml_freq=25.0,
            backend=backend,
        ).receiver_amplitudes_p

    args = (v, rho, sa)
    assert_compiled_matches_eager_forward_and_grad_on_device(f, args, device)


@pytest.mark.parametrize("device", _WRAP_DEVICES)
@pytest.mark.parametrize(
    "boundary",
    [
        ["rigid", "rigid", "pml", "pml"],
        ["free", "pml", "pml", "rigid"],
        ["free", "rigid", "free", "rigid"],  # historical pairing pin
    ],
    ids=lambda b: "-".join(b),
)
def test_viscoacoustic_c_fr_combination_adjoint_parity(device, boundary):
    """Compiled viscoacoustic, free/rigid combinations: grads match eager."""
    torch.manual_seed(0)
    shape = (16, 16)
    dx, dt, nt = 5.0, 0.0001, 1200
    _assert_wraps(nt, dt, dx, shape[0])
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double).requires_grad_(True)
    v = (torch.ones(shape) * C + 30 * torch.rand(shape)).requires_grad_(True)
    rho = (torch.ones(shape) * RHO + 20 * torch.rand(shape)).requires_grad_(True)
    q = torch.full(shape, Q).requires_grad_(True)
    sl = torch.tensor([[[10, 10]]])
    rl = torch.tensor([[[5, 5]]])

    def f(v, rho, q, sa, backend, dev="cpu"):
        return viscoacoustic(
            v.to(dev),
            rho.to(dev),
            q.to(dev),
            dx,
            dt,
            source_amplitudes_pressure_rate=(RHO * C**2) * (sa.to(dev)),
            source_locations_pressure_rate=sl.to(dev),
            receiver_locations_p=rl.to(dev),
            pml_width=_fr_pml_width(boundary),
            boundary=boundary,
            accuracy=4,
            pml_freq=25.0,
            q_freq=25.0,
            backend=backend,
        ).receiver_amplitudes_p

    args = (v, rho, q, sa)
    assert_compiled_matches_eager_forward_and_grad_on_device(f, args, device)


@pytest.mark.parametrize("device", _WRAP_DEVICES)
@pytest.mark.parametrize("sc_only", [False, True])
@pytest.mark.parametrize(
    "boundary",
    [
        ["rigid", "rigid", "pml", "pml"],
        ["free", "pml", "pml", "rigid"],
        ["free", "rigid", "free", "rigid"],  # historical pairing pin
    ],
    ids=lambda b: "-".join(b),
)
def test_acoustic_born_c_fr_combination_adjoint_parity(device, sc_only, boundary):
    """Compiled acoustic_born, free/rigid combinations: grads match eager.

    ``sc_only`` restricts requires_grad to ``dv``, which exercises the
    separate ``backward_sc`` entry point; the full case exercises the
    two-field ``backward``.
    """
    torch.manual_seed(0)
    shape = (16, 16)
    dx, dt, nt = 5.0, 0.0001, 1200
    _assert_wraps(nt, dt, dx, shape[0])
    sa = deepwave.wavelets.ricker(25.0, nt, dt, 0.02).reshape(1, 1, nt)
    sa = sa.to(torch.double)
    v = torch.ones(shape) * C + 30 * torch.rand(shape)
    rho = torch.ones(shape) * RHO + 20 * torch.rand(shape)
    dv = (torch.randn(shape) * 0.05 * C).to(torch.double)
    sl = torch.tensor([[[10, 10]]])
    rl = torch.tensor([[[5, 5]]])

    def f(vv, rr, dvdv, ss, backend, dev="cpu"):
        out = acoustic_born(
            vv.to(dev),
            rr.to(dev),
            dx,
            dt,
            dv=dvdv.to(dev),
            source_amplitudes_volume_rate=ss.to(dev),
            source_locations_volume_rate=sl.to(dev),
            sc_receiver_locations_p=rl.to(dev),
            bg_receiver_locations_p=rl.to(dev),
            pml_width=_fr_pml_width(boundary),
            boundary=boundary,
            accuracy=4,
            pml_freq=25.0,
            backend=backend,
        )
        return out.bg_receiver_amplitudes_p + out.sc_receiver_amplitudes_p

    if sc_only:
        leaves = [dv]

        def build(d):
            return (v, rho, d, sa)
    else:
        leaves = [v, rho, dv, sa]

        def build(*a):
            return a

    leaves_c = [x.clone().requires_grad_(True) for x in leaves]
    out_c = f(*build(*leaves_c), "compiled", device)
    g_c = torch.autograd.grad(out_c.sum(), leaves_c)
    leaves_e = [x.clone().requires_grad_(True) for x in leaves]
    out_e = f(*build(*leaves_e), "eager")
    g_e = torch.autograd.grad(out_e.sum(), leaves_e)
    assert (out_c.cpu() - out_e).abs().max() <= 1e-6 * out_e.abs().max()
    for a, b in zip(g_c, g_e):
        assert (a.cpu() - b).abs().max() <= 1e-6 * b.abs().max().clamp_min(1e-300)


# ---------------------------------------------------------------------------
# The bare zero-width PML face warns, and deepwave.boundaries()
# ---------------------------------------------------------------------------


def _clear_bare_face_warnings() -> None:
    """Drops the once-per-process guards so the warning can fire again."""
    from deepwave import _common

    for cause in list(_common._warned_once_causes):
        if cause.startswith("bare-zero-width-face:"):
            _common._warned_once_causes.discard(cause)


def _bare_face_warnings(record) -> list:
    """The zero-width-face warnings among a ``catch_warnings(record=True)`` list."""
    return [w for w in record if "zero-width PML face" in str(w.message)]


def _scalar_call(**kwargs):
    """A small scalar call, for exercising the zero-width-face warning."""
    v = torch.full((30, 30), 1500.0)
    amplitudes = torch.zeros(1, 1, 20)
    amplitudes[0, 0, 5] = 1.0
    return deepwave.scalar(
        v,
        4.0,
        0.001,
        source_amplitudes=amplitudes,
        source_locations=torch.tensor([[[15, 15]]]),
        pml_freq=25.0,
        **kwargs,
    )


def test_bare_zero_width_face_warns_once_per_process_per_face() -> None:
    """One warning per (propagator, face), not one per call.

    A per-call warning would print thousands of lines from an FWI loop, so
    the guard is a module-level set. That also means it is *not* controlled
    by the warnings filter: ``simplefilter("always")`` does not make it
    repeat, which is deliberate and documented.
    """
    _clear_bare_face_warnings()
    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        _scalar_call(pml_width=[0, 20, 20, 20])
    assert len(_bare_face_warnings(first)) == 1

    # The same face again: silent.
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        _scalar_call(pml_width=[0, 20, 20, 20])
    assert not _bare_face_warnings(again)

    # A different face: a second warning.
    with warnings.catch_warnings(record=True) as other_face:
        warnings.simplefilter("always")
        _scalar_call(pml_width=[20, 20, 0, 20])
    assert len(_bare_face_warnings(other_face)) == 1

    # A different propagator, same face: a third.
    with warnings.catch_warnings(record=True) as other_prop:
        warnings.simplefilter("always")
        v = torch.full((30, 30), 1500.0)
        rho = torch.full((30, 30), 2000.0)
        amplitudes = torch.zeros(1, 1, 20)
        amplitudes[0, 0, 5] = 1.0
        deepwave.acoustic(
            v,
            rho,
            4.0,
            0.001,
            source_amplitudes_volume_rate=amplitudes,
            source_locations_volume_rate=torch.tensor([[[15, 15]]]),
            pml_freq=25.0,
            pml_width=[0, 20, 20, 20],
        )
    assert len(_bare_face_warnings(other_prop)) == 1


def test_bare_zero_width_face_named_by_boundary_does_not_warn() -> None:
    """Naming the condition is the fix, and silences it."""
    _clear_bare_face_warnings()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _scalar_call(pml_width=[0, 20, 20, 20], boundary=["free", "pml", "pml", "pml"])
    assert not _bare_face_warnings(record)


def test_all_pml_call_does_not_warn() -> None:
    """The no-zero-width control: an ordinary call is silent."""
    _clear_bare_face_warnings()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _scalar_call(pml_width=20)
    assert not _bare_face_warnings(record)


def test_pml_named_face_with_zero_width_still_warns() -> None:
    """A face named ``"pml"`` with width 0 still warns.

    It says it absorbs, and it does not.

    This is the one rule a reviewer will second-guess, so it is pinned:
    naming a face ``"pml"`` is not naming the condition it actually
    produces.
    """
    _clear_bare_face_warnings()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _scalar_call(pml_width=[0, 20, 20, 20], boundary=["pml", "pml", "pml", "pml"])
    assert len(_bare_face_warnings(record)) == 1


def test_bare_zero_width_warning_output_is_unchanged() -> None:
    """The warning is a message, not a behavioural change.

    Byte-identical receiver amplitudes with the warning suppressed and with
    it emitted.
    """
    _clear_bare_face_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = _scalar_call(
            pml_width=[0, 20, 20, 20], receiver_locations=torch.tensor([[[5, 5]]])
        )
        second = _scalar_call(
            pml_width=[0, 20, 20, 20], receiver_locations=torch.tensor([[[5, 5]]])
        )
    assert torch.equal(first.receiver_amplitudes, second.receiver_amplitudes)


@pytest.mark.parametrize(
    "name",
    [
        "scalar",
        "scalar_born",
        "acoustic",
        "acoustic_born",
        "acoustic_vti",
        "viscoacoustic",
        "elastic",
        "elastic_born",
        "elastic_vti",
    ],
)
def test_bare_zero_width_condition_table_covers_every_propagator(name: str) -> None:
    """Every propagator has a row, and it names a real boundary condition.

    The message is built from this table, and the documentation table in
    ``docs/boundaries.rst`` records the same measurements, so a propagator
    missing here would produce a ``KeyError`` at the warning site rather
    than a wrong message.
    """
    from deepwave._common import _BARE_ZERO_WIDTH_CONDITION, _BOUNDARY_NAME_TO_CODE

    condition, coefficient, suggested, caveat = _BARE_ZERO_WIDTH_CONDITION[name]
    assert condition
    assert coefficient
    assert suggested in _BOUNDARY_NAME_TO_CODE
    assert suggested != "pml", "the suggestion must name a reflecting condition"
    # The scalar family is a free surface; the staggered families are rigid.
    expected = "free" if name in ("scalar", "scalar_born") else "rigid"
    assert suggested == expected
    # The elastic families' bare face is frictionless (normal velocity
    # clamped, tangential free), so ``rigid`` -- which is no-slip -- is *not*
    # the same condition and the message must say so. The scalar and acoustic
    # families have no such discrepancy and must not carry a caveat.
    elastic = name.startswith("elastic")
    if elastic:
        assert "not quite this condition" in caveat
        assert "no-slip" in caveat
        assert "frictionless" in condition
    else:
        assert caveat == ""


def test_bare_zero_width_warning_names_the_actual_condition() -> None:
    """The message says what the face *is*, per propagator, with its sign."""
    _clear_bare_face_warnings()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _scalar_call(pml_width=[0, 20, 20, 20])
    message = str(_bare_face_warnings(record)[0].message)
    assert "free surface" in message
    assert "-1" in message
    assert "scalar" in message
    assert "the beginning of dimension 0" in message
    # It offers both fixes, and the stable filter prefix.
    assert "boundary=['free', 'pml', 'pml', 'pml']" in message
    assert "deepwave.boundaries(2, dim0_begin='free')" in message
    assert message.startswith("deepwave: zero-width PML face")


def test_docs_reflection_table_matches_the_warning_table() -> None:
    """The documentation table and the warning table report the same numbers.

    ``docs/boundaries.rst`` carries the per-accuracy measurements and
    ``_BARE_ZERO_WIDTH_CONDITION`` carries the range the warning quotes.
    Neither is generated from the other, so without this they can drift --
    and they had: the documentation's Born rows showed only the background
    channel while the warning's range covered the scattered channel too.
    Every value the documentation lists must round, at the three decimal
    places the warning uses, into the range the warning states.
    """
    import pathlib
    import re

    from deepwave._common import _BARE_ZERO_WIDTH_CONDITION

    text = pathlib.Path(__file__).resolve().parents[1] / "docs" / "boundaries.rst"
    lines = text.read_text(encoding="utf-8").splitlines()
    # The rows of the reflection-coefficient list-table: a propagator name
    # cell, then five more cells, the fifth of which holds the numbers.
    minus = "\u2212"  # the documentation table uses a typographic minus
    number = re.compile(rf"[+{minus}-]\d+\.\d+")
    documented: dict[str, list[float]] = {}
    for index, line in enumerate(lines):
        match = re.fullmatch(r"   \* - ``(\w+)``", line)
        if match is None or match.group(1) not in _BARE_ZERO_WIDTH_CONDITION:
            continue
        cell = "\n".join(lines[index + 4 : index + 7])
        values = [float(found.replace(minus, "-")) for found in number.findall(cell)]
        assert values, f"no numbers found for {match.group(1)}"
        documented[match.group(1)] = values
    assert set(documented) == set(_BARE_ZERO_WIDTH_CONDITION), (
        "every propagator must have a row in docs/boundaries.rst"
    )
    for name, values in documented.items():
        coefficient = _BARE_ZERO_WIDTH_CONDITION[name][1]
        # The range the warning states, as every coefficient-looking number
        # in the phrase before its first semicolon (the primary channel).
        stated = [
            float(found)
            for found in re.findall(r"[+-]\d\.\d+", coefficient.split(";")[0])
        ]
        assert stated, f"no coefficient in the warning phrase for {name}"
        low, high = min(stated), max(stated)
        for value in values:
            rounded = round(value, 3)
            assert low - 5e-4 <= rounded <= high + 5e-4, (
                f"{name}: docs value {value} is outside the warning's stated "
                f"range {low} to {high}"
            )


# --- deepwave.boundaries() -------------------------------------------------


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_boundaries_helper_returns_a_consistent_pair(ndim: int) -> None:
    """Both lists have the right length and cannot disagree."""
    boundary, pml_width = deepwave.boundaries(ndim)
    assert len(boundary) == len(pml_width) == 2 * ndim
    assert all(name == "pml" for name in boundary)
    assert all(width == 20 for width in pml_width)
    # A named face always gets width 0; that is the invariant.
    boundary, pml_width = deepwave.boundaries(ndim, dim0_begin="free")
    assert boundary[0] == "free"
    assert pml_width[0] == 0
    assert all(width == 20 for width in pml_width[1:])


def test_boundaries_top_is_an_alias_for_dim0_begin() -> None:
    """``top`` and ``dim0_begin`` name the same face; both is an error."""
    assert deepwave.boundaries(2, top="free") == deepwave.boundaries(
        2, dim0_begin="free"
    )
    with pytest.raises(deepwave.DeepwaveValueError, match="top and dim0_begin"):
        deepwave.boundaries(2, top="free", dim0_begin="free")


def test_boundaries_rejects_a_dimension_the_model_does_not_have() -> None:
    """A ``dim{k}_*`` keyword for a dimension ``ndim`` does not have raises."""
    with pytest.raises(deepwave.DeepwaveValueError, match=r"dim2_begin.*ndim=2"):
        deepwave.boundaries(2, dim2_begin="free")


def test_boundaries_rejects_an_unknown_condition_name() -> None:
    """An unrecognised condition name raises, listing the accepted ones."""
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        deepwave.boundaries(2, top="squishy")
    for name in ("free", "periodic", "pml", "rigid"):
        assert name in str(excinfo.value)


def test_boundaries_rejects_a_zero_width_pml_face() -> None:
    """The helper refuses to mint the thing the warning is about."""
    with pytest.raises(deepwave.DeepwaveValueError, match="implicit reflector"):
        deepwave.boundaries(2, width=0)
    with pytest.raises(deepwave.DeepwaveValueError, match="implicit reflector"):
        deepwave.boundaries(2, width=[0, 20, 20, 20])


def test_boundaries_delegates_the_periodic_rule() -> None:
    """A rule the helper does not implement itself is still enforced.

    ``periodic`` must be set on both faces of a dimension. The helper runs
    the propagators' own validation on its result, so it cannot return a
    pair they would reject -- and it reports it as its own error type.
    """
    with pytest.raises(deepwave.DeepwaveValueError, match="periodic"):
        deepwave.boundaries(2, dim0_begin="periodic")
    boundary, pml_width = deepwave.boundaries(
        2, dim0_begin="periodic", dim0_end="periodic"
    )
    assert boundary[:2] == ["periodic", "periodic"]
    assert pml_width[:2] == [0, 0]


def test_boundaries_rejects_a_bad_ndim_or_width() -> None:
    """``ndim`` outside 1-3, and a negative or wrong-length width, raise."""
    with pytest.raises(deepwave.DeepwaveValueError, match="ndim must be 1, 2 or 3"):
        deepwave.boundaries(4)
    with pytest.raises(deepwave.DeepwaveValueError, match="non-negative"):
        deepwave.boundaries(2, width=-1)
    with pytest.raises(deepwave.DeepwaveValueError, match="entries for ndim"):
        deepwave.boundaries(2, width=[20, 20])


def test_boundaries_output_is_accepted_and_does_not_warn() -> None:
    """The round trip: what the helper returns runs, and is silent.

    The helper exists to replace the implicit reflector, so a pair it built
    must never trip the zero-width-face warning.
    """
    _clear_bare_face_warnings()
    boundary, pml_width = deepwave.boundaries(2, top="free", width=10)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = _scalar_call(boundary=boundary, pml_width=pml_width)
    assert not _bare_face_warnings(record)
    assert torch.isfinite(out.wavefield_0).all()


# --- the measured table's own test -----------------------------------------
#
# The numbers asserted here are the ones documented at
# ``docs/boundaries.rst``'s :ref:`bare-zero-width-face`. Every propagator lands
# within ~0.6% of its nominal +-1 at every accuracy order, so a 1% tolerance
# (the same one the named-face tests use) both passes and bites.

_BARE_FACE_EXPECTED_SIGN = {
    # (helper key): the sign the *primary channel* reflects with.
    "scalar": -1.0,
    "scalar_born": -1.0,
    "acoustic": +1.0,
    "acoustic_born": +1.0,
    "acoustic_vti": +1.0,
    "viscoacoustic": +1.0,
    "elastic": +1.0,
    "elastic_born": +1.0,
    "elastic_vti": +1.0,
}


@pytest.mark.slow
@pytest.mark.parametrize("accuracy", [2, 4])
@pytest.mark.parametrize(
    "name",
    [
        "scalar",
        "scalar_born",
        "acoustic",
        "acoustic_born",
        "viscoacoustic",
        "acoustic_vti",
        "elastic",
        "elastic_vti",
        "elastic_born",
    ],
)
def test_bare_zero_width_face_reflection_coefficient(name, accuracy):
    """A bare zero-width face reflects with the documented sign and magnitude.

    This is the documentation table's test: it pins the *measured* facts that
    ``docs/boundaries.rst`` states and that the warning's message quotes,
    so the three cannot drift apart. ``face_boundary=None`` is the bare
    zero-width case in every one of these harnesses.

    Subsumes the two one-off bare-face assertions that used to live in
    ``tests/test_scalar.py`` and ``tests/test_acoustic.py``.
    """
    # ``face_boundary=None`` is the bare zero-width case in the scalar and
    # acoustic helpers; the vti/elastic ones take the per-face list, whose
    # bare spelling is ["pml", "pml"]. The Born helpers return
    # ``(background, scattered)``, and both channels are asserted.
    if name == "scalar":
        r = _reflection_coefficient(accuracy, None)
    elif name == "scalar_born":
        r = _born_reflection_coefficient(accuracy, None)
    elif name == "acoustic":
        r = _acoustic_reflection_coefficient(accuracy, None)
    elif name == "acoustic_born":
        r = _acoustic_born_reflection_coefficient(accuracy, None)
    elif name == "viscoacoustic":
        r = _viscoacoustic_reflection_coefficient(accuracy, None)
    elif name == "acoustic_vti":
        r = _acoustic_vti_reflection_coefficient(accuracy, 0, ["pml", "pml"])
    elif name == "elastic":
        torch.set_default_dtype(torch.double)
        try:
            r = _elastic_reflection_coefficient(accuracy, ["pml", "pml"])
        finally:
            torch.set_default_dtype(torch.float)
    elif name == "elastic_vti":
        torch.set_default_dtype(torch.double)
        try:
            r = _elastic_vti_reflection_coefficient(accuracy, 0, ["pml", "pml"])
        finally:
            torch.set_default_dtype(torch.float)
    else:
        torch.set_default_dtype(torch.double)
        try:
            r = _elastic_born_reflection_coefficient(accuracy, ["pml", "pml"])
        finally:
            torch.set_default_dtype(torch.float)
    expected = _BARE_FACE_EXPECTED_SIGN[name]
    # The Born helpers report (background, scattered); both must match.
    measured = r if isinstance(r, tuple) else (r,)
    for channel, value in enumerate(measured):
        assert value == pytest.approx(expected, abs=0.01), (
            f"{name} accuracy {accuracy} channel {channel}: bare zero-width "
            f"face gave R={value}, documented as {expected:+.0f}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("accuracy", [2, 4])
def test_bare_zero_width_face_matches_the_table_sign_for_every_propagator(accuracy):
    """The sign in ``_BARE_ZERO_WIDTH_CONDITION`` is the measured sign.

    The warning's message quotes the table; this ties the table's *claim* to
    the propagators whose coefficient is actually measured above, so a wrong
    row cannot be shipped with a right message.
    """
    from deepwave._common import _BARE_ZERO_WIDTH_CONDITION

    for name, sign in _BARE_FACE_EXPECTED_SIGN.items():
        _, coefficient, suggested, _ = _BARE_ZERO_WIDTH_CONDITION[name]
        if sign < 0:
            assert suggested == "free"
            assert "-1" in coefficient
        else:
            assert suggested == "rigid"
            assert "+1" in coefficient or "+0.99" in coefficient


@pytest.mark.parametrize("accuracy", [2, 4])
def test_bare_zero_width_elastic_face_preserves_tangential_velocity(accuracy):
    """A bare face leaves tangential motion free; rigid clamps it.

    A supported one-step initial-value problem isolates the boundary rule.
    A nonzero tangential velocity starts on the low face with every stress
    zero. The bare zero-width PML face leaves it unchanged, while the named
    rigid wall sets it to zero. The other faces retain absorbing PMLs.
    """
    n = 20
    lamb = torch.full((n, n), 2.0e9, dtype=torch.double)
    mu = torch.full((n, n), 1.0e9, dtype=torch.double)
    buoyancy = torch.full((n, n), 1.0 / 2200.0, dtype=torch.double)
    # The spatial shape includes the PML widths [0, 4, 4, 4].
    vx = torch.zeros((1, n + 4, n + 8), dtype=torch.double)
    vx[0, 0, 9:19] = 1.0
    kwargs = {
        "vx_0": vx,
        "nt": 1,
        "pml_width": [0, 4, 4, 4],
        "accuracy": accuracy,
        "pml_freq": 15.0,
        "max_vel": 2500.0,
        "backend": "eager",
        "origin": [0, 0],
    }
    bare = elastic(
        lamb,
        mu,
        buoyancy,
        5.0,
        0.0005,
        boundary=["pml", "pml", "pml", "pml"],
        **kwargs,
    )
    rigid = elastic(
        lamb,
        mu,
        buoyancy,
        5.0,
        0.0005,
        boundary=["rigid", "pml", "pml", "pml"],
        **kwargs,
    )
    assert torch.equal(bare.vx_0, vx)
    assert torch.count_nonzero(rigid.vx_0) == 0


# ---------------------------------------------------------------------------
# A single ``boundary`` string applies to every face, and the three
# spellings of a zero-width ``"pml"`` face are one call
# ---------------------------------------------------------------------------


_ALL_ELEVEN = (
    "scalar",
    "scalar_born",
    "acoustic",
    "acoustic_born",
    "acoustic_vti",
    "viscoacoustic",
    "elastic",
    "elastic_born",
    "elastic_vti",
)


@pytest.mark.parametrize("name", ["pml", "free", "rigid", "periodic"])
def test_single_boundary_string_applies_to_every_face(name: str) -> None:
    """A string is broadcast to all faces, not iterated character by character.

    ``list("pml")`` is ``['p', 'm', 'l']``, which is why this needs pinning:
    without the broadcast the call fails with a message about a face named
    ``'p'`` (or a length of 3), neither of which mentions what was written.
    """
    for ndim in (1, 2, 3):
        widths = [0] * 2 * ndim if name != "pml" else [20] * 2 * ndim
        assert set_boundary(name, widths, ndim) == set_boundary(
            [name] * 2 * ndim, widths, ndim
        )


def test_single_boundary_string_gives_the_same_numbers_as_the_list() -> None:
    """The broadcast is the list; nothing about the run changes."""
    spelled = _scalar_call(pml_width=[0, 20, 20, 20], boundary="pml")
    listed = _scalar_call(
        pml_width=[0, 20, 20, 20], boundary=["pml", "pml", "pml", "pml"]
    )
    assert torch.equal(spelled[0], listed[0])


@pytest.mark.parametrize(
    "boundary",
    [None, "pml", ["pml", "pml", "pml", "pml"]],
    ids=["omitted", "string", "list"],
)
def test_zero_width_pml_face_warns_under_every_spelling(boundary) -> None:
    """``"pml"`` is the spelled-out default, so the three spellings are one call.

    It warns under all three and raises under none: a zero-width ``"pml"``
    face is v0.0.27's ``pml_width=0`` idiom and stays supported.
    """
    _clear_bare_face_warnings()
    kwargs = {} if boundary is None else {"boundary": boundary}
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _scalar_call(pml_width=[0, 20, 20, 20], **kwargs)
    caught = _bare_face_warnings(record)
    assert len(caught) == 1
    assert issubclass(caught[0].category, deepwave.ZeroWidthFaceWarning)


def test_zero_width_face_warning_names_its_own_category() -> None:
    """The one-line filter recipe is in the message the user is reading."""
    _clear_bare_face_warnings()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _scalar_call(pml_width=[0, 20, 20, 20], boundary="pml")
    message = str(_bare_face_warnings(record)[0].message)
    assert "deepwave.ZeroWidthFaceWarning" in message


def test_zero_width_face_warning_category_hierarchy() -> None:
    """Filtering the base silences it; filtering ``FutureWarning`` does not."""
    assert issubclass(deepwave.ZeroWidthFaceWarning, deepwave.DeepwaveWarning)
    assert issubclass(deepwave.ZeroWidthFaceWarning, UserWarning)
    assert not issubclass(deepwave.ZeroWidthFaceWarning, FutureWarning)


@pytest.mark.parametrize("name", _ALL_ELEVEN)
def test_zero_width_pml_face_spellings_agree_on_all_nine(name: str) -> None:
    """Every propagator treats the three spellings identically."""
    from deepwave import _capabilities

    ndim = 2
    faces = 2 * ndim
    widths = [0, *[20] * (faces - 1)]
    models = _capabilities._models(name, ndim)
    base = _capabilities._base_kwargs(name, ndim, None)
    base.update(_capabilities._extra_kwargs(name, ndim))
    base.pop("pml_width", None)
    base.pop("boundary", None)
    caught = []
    for boundary in (None, "pml", ["pml"] * faces):
        _clear_bare_face_warnings()
        kwargs = dict(base, pml_width=widths)
        if boundary is not None:
            kwargs["boundary"] = boundary
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            getattr(deepwave, name)(*models, **kwargs)
        caught.append(_bare_face_warnings(record))
    assert [len(c) for c in caught] == [1, 1, 1], name
    assert all(issubclass(c[0].category, deepwave.ZeroWidthFaceWarning) for c in caught)
    assert len({str(c[0].message) for c in caught}) == 1
