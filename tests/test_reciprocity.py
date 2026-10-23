r"""Source--receiver reciprocity, channel pairing by channel pairing.

Reciprocity says that swapping a source and a receiver leaves the recorded
trace unchanged. On the staggered-grid propagators the *mixed* pairings --
a scalar (volume-rate or pressure-rate) source recorded by a velocity
receiver, versus a force source recorded by the pressure receiver -- are not
simply equal: the pressure and velocity fields live on time levels half a
step apart, so the two traces are related by a sign, a one-sample time shift
and (on the elastic family) the bulk modulus. The relations below are exact
properties of the discrete scheme, not approximations, and they pin down the
time level of every source injection and every receiver recording.

Time levels
-----------
Within one step the propagators record the pressure receivers from
:math:`p^{\,n}` and the velocity receivers from
:math:`\mathbf w^{\,n-\frac12}`, then advance
:math:`\mathbf w^{\,n-\frac12}\to\mathbf w^{\,n+\frac12}` and add the
force sources, then advance :math:`p^{\,n}\to p^{\,n+1}` and add the scalar
sources. Sample ``t`` of a pressure receiver is therefore the field at
:math:`t\,\Delta t`, sample ``t`` of a velocity receiver the field at
:math:`(t-0.5)\Delta t`, sample ``t`` of a force source acts at
:math:`t\,\Delta t` and sample ``t`` of a scalar source at
:math:`(t+0.5)\Delta t` -- the arrangement documented in ``docs/units.rst``
and in each propagator's ``source_amplitudes_*`` docstrings.

The relations
-------------
Write ``f`` for the trace of receiver channel ``rc`` at B driven by source
channel ``sc`` at A, and ``g`` for the trace of the *dual* pairing: source
channel dual to ``rc`` firing at B, recorded by the receiver channel dual to
``sc`` at A (the scalar source channel is dual to the ``p`` receiver; each
velocity channel is dual to itself). With ``C`` the scalar source channel's
coupling to pressure -- ``1`` for a volume rate, the bulk modulus at the
scalar channel's own point for a pressure rate -- the discrete scheme obeys

* scalar source, ``p`` receiver: ``f[n] == g[n]``;
* velocity source, velocity receiver: ``f[n] == g[n]``;
* scalar source at A, velocity receiver at B: ``C * f[n] == -g[n - 1]``;
* velocity source at A, ``p`` receiver at B: ``f[n] == -C * g[n + 1]``.

These hold to machine precision (measured ~1e-15 relative in float64), for
sources and receivers anywhere on the grid, including inside the absorbing
boundary. A half-step error in any source injection or receiver recording
would leave the two traces related by a *fractional* sample shift, which no
integer shift fits; the tests below therefore also check that the neighbouring
integer shifts fail by a wide margin, so a passing result cannot be vacuous.

The Born propagators
--------------------
A Born propagator's *scattered* field obeys the derivative of the relation
its background propagator obeys. ``acoustic_born``'s ``C`` is one, a
constant, so its scattered field obeys the relations above unchanged.
``elastic_born``'s ``C`` is the bulk modulus, which the model perturbation
moves, so differentiating ``C f[n] == -g[n - 1]`` leaves an extra term:

* ``C df[n] + dC f0[n] == -dg[n - 1]``, with ``f0`` the background trace of
  the scalar-source pairing and ``dC = dlamb + 2 dmu / ndim``.

That term vanishes for a pure ``dbuoyancy`` perturbation and is of order one
otherwise.

Limits
------
Two families of propagator satisfy only the like-for-like relations exactly:

* attenuating media (``viscoacoustic``), because converting
  between a scalar source rate and a pressure needs the frequency-dependent
  modulus, not the single real constant ``C``. The departure is O(1/Q) and
  vanishes as Q grows;
* anisotropic media (``acoustic_vti``, ``elastic_vti``),
  because the isotropic scalar source is the dual of a volumetric *strain*
  rate while the ``p`` receiver records mean normal *stress*; the two differ
  by the stiffness, which is a single scalar only when it is isotropic. The
  departure vanishes in the isotropic limit.

Both limits are tested here, so the mixed pairings stay covered on those
propagators without pinning an arbitrary tolerance.
"""

import math

import pytest
import torch

import deepwave

DTYPE = torch.float64
NX = (40, 40)
DX = 5.0
DT = 4.0e-4
NT = 140
FREQ = 20.0
PEAK_TIME = 0.05
# Reciprocity is exact in the discrete scheme; this absorbs the compiled
# backend's -Ofast reordering only.
RTOL = 1e-11
# A wrong integer shift must miss by far more than this, or the test would
# pass on a propagator whose time levels are off.
VACUITY_FLOOR = 1e-2

VP = 1500.0
VS = 900.0
RHO = 2000.0
LAMB = RHO * (VP**2 - 2 * VS**2)
MU = RHO * VS**2
# The bulk modulus of the isotropic elastic model above, in 2-D.
ELASTIC_BULK_MODULUS = LAMB + MU

LOC_A = torch.tensor([[[16, 18]]])
LOC_B = torch.tensor([[[23, 26]]])

# Propagators whose every channel pairing is exactly reciprocal.
EXACT = ("acoustic", "acoustic_born", "elastic")
# Propagators whose mixed pairings need a relation of their own: the limit
# tests for the attenuating and anisotropic ones, and the linearised relation
# for ``elastic_born``. Their like-for-like pairings are exact, and are
# checked here.
LIKE_FOR_LIKE = (
    "viscoacoustic",
    "acoustic_vti",
    "elastic_vti",
    "elastic_born",
)


def _ones() -> torch.Tensor:
    return torch.ones(*NX, dtype=DTYPE)


def _models(name: str) -> tuple[torch.Tensor, ...]:
    """The positional model arguments of ``name``, homogeneous."""
    one = _ones()
    if name in ("acoustic", "acoustic_born"):
        return (VP * one, RHO * one)
    if name == "viscoacoustic":
        return (VP * one, RHO * one, 40.0 * one)
    if name == "acoustic_vti":
        c33 = RHO * VP**2
        return (1.1 * c33 * one, 0.9 * c33 * one, c33 * one, one / RHO)
    if name in (
        "elastic",
        "elastic_born",
    ):
        return (LAMB * one, MU * one, one / RHO)
    if name in ("elastic_vti",):
        c33 = RHO * VP**2
        c55 = RHO * VS**2
        return (
            1.1 * c33 * one,
            0.9 * (c33 - 2 * c55) * one,
            c33 * one,
            c55 * one,
            one / RHO,
        )
    raise AssertionError(name)


def _extra(name: str) -> dict[str, object]:
    """The keyword-only model arguments and required options of ``name``."""
    one = _ones()
    if name in ("viscoacoustic",):
        return {"q_freq": FREQ}
    if name == "acoustic_born":
        return {"dv": 30.0 * one, "drho": 5.0 * one}
    if name == "elastic_born":
        return {"dlamb": 1.0e8 * one, "dmu": 5.0e7 * one, "dbuoyancy": 1.0e-5 * one}
    return {}


def _bulk_modulus(name: str) -> float:
    """``C``: the scalar source channel's coupling to pressure.

    A volume rate already carries the modulus, so ``C`` is one; a pressure
    rate does not, so ``C`` is the bulk modulus of the (homogeneous) model at
    the point the scalar channel occupies. A model that is not one of the
    defaults built by :func:`_models` passes its own value instead: the
    stiffness-parametrised propagators have no single bulk modulus until the
    stiffnesses are given.
    """
    if deepwave.spec(name).source_channels(2)[0] == "volume_rate":
        return 1.0
    if name == "viscoacoustic":
        # Direct pressure-rate forcing approaches acoustic volume forcing
        # divided by the relaxed bulk modulus in the lossless limit.
        return RHO * VP**2
    return ELASTIC_BULK_MODULUS


def _wavelet() -> torch.Tensor:
    t = torch.arange(NT, dtype=DTYPE) * DT - PEAK_TIME
    a = (math.pi * FREQ * t) ** 2
    return ((1 - 2 * a) * torch.exp(-a)).reshape(1, 1, -1)


def _trace(
    name: str,
    source_channel: str,
    source_location: torch.Tensor,
    receiver_channel: str,
    receiver_location: torch.Tensor,
    backend: str,
    models: object = None,
    extra: object = None,
) -> torch.Tensor:
    """One shot: ``source_channel`` at one point, ``receiver_channel`` at another."""
    spec = deepwave.spec(name)
    kwargs: dict[str, object] = {
        "pml_width": 10,
        "pml_freq": FREQ,
        "max_vel": 2.0 * VP,
        "accuracy": 4,
        "backend": backend,
        "source_amplitudes_" + source_channel: _wavelet(),
        "source_locations_" + source_channel: source_location,
        spec.receiver_locations_kwarg(receiver_channel): receiver_location,
    }
    kwargs.update(_extra(name) if extra is None else extra)  # type: ignore[arg-type]
    positional = _models(name) if models is None else models
    out = getattr(deepwave, name)(*positional, DX, DT, **kwargs)  # type: ignore[misc]
    return getattr(out, spec.receiver_output_name(receiver_channel))[0, 0]


def _residual(f: torch.Tensor, g: torch.Tensor, shift: int, scale: float) -> float:
    """Relative L2 misfit of ``f[n] == scale * g[n + shift]``."""
    if shift > 0:
        f, g = f[:-shift], g[shift:]
    elif shift < 0:
        f, g = f[-shift:], g[:shift]
    return float(torch.linalg.vector_norm(f - scale * g) / torch.linalg.vector_norm(f))


def _expected(
    name: str,
    source_channel: str,
    receiver_channel: str,
    modulus: float = 0.0,
) -> tuple[int, float]:
    """The ``(shift, scale)`` of the pairing's reciprocity relation."""
    scalar_source = deepwave.spec(name).source_channels(2)[0]
    modulus = modulus or _bulk_modulus(name)
    if source_channel == scalar_source:
        if receiver_channel == "p":
            return 0, 1.0
        return -1, -1.0 / modulus
    if receiver_channel == "p":
        return 1, -modulus
    return 0, 1.0


def _dual(name: str, channel: str) -> str:
    """The source channel dual to a receiver channel, or vice versa."""
    scalar_source = deepwave.spec(name).source_channels(2)[0]
    if channel == "p":
        return scalar_source
    if channel == scalar_source:
        return "p"
    return channel


def _assert_reciprocal(
    name: str,
    source_channel: str,
    receiver_channel: str,
    backend: str,
    models: object = None,
    extra: object = None,
    rtol: float = RTOL,
    check_vacuity: bool = True,
    modulus: float = 0.0,
) -> None:
    f = _trace(
        name, source_channel, LOC_A, receiver_channel, LOC_B, backend, models, extra
    )
    g = _trace(
        name,
        _dual(name, receiver_channel),
        LOC_B,
        _dual(name, source_channel),
        LOC_A,
        backend,
        models,
        extra,
    )
    shift, scale = _expected(name, source_channel, receiver_channel, modulus)
    residual = _residual(f, g, shift, scale)
    assert residual < rtol, (
        f"{name} {source_channel}->{receiver_channel}: reciprocity residual "
        f"{residual:.3e} at shift {shift:+d}, scale {scale:.6e}"
    )
    if check_vacuity:
        # A propagator whose time levels were half a step out would not be
        # fitted by any integer shift; check the neighbours miss by a lot.
        for wrong in (shift - 1, shift + 1):
            assert _residual(f, g, wrong, scale) > VACUITY_FLOOR


def _pairings(name: str) -> list[tuple[str, str]]:
    spec = deepwave.spec(name)
    return [
        (source_channel, receiver_channel)
        for source_channel in spec.source_channels(2)
        for receiver_channel in spec.receiver_channels(2)
    ]


def _all_cases() -> list[tuple[str, str, str]]:
    return [
        (name, source_channel, receiver_channel)
        for name in EXACT
        for source_channel, receiver_channel in _pairings(name)
    ]


@pytest.mark.parametrize(("name", "source_channel", "receiver_channel"), _all_cases())
def test_every_channel_pairing_is_exactly_reciprocal(
    name: str, source_channel: str, receiver_channel: str
) -> None:
    """Every source/receiver channel pairing obeys its exact relation."""
    _assert_reciprocal(name, source_channel, receiver_channel, "compiled")


@pytest.mark.parametrize(("name", "source_channel", "receiver_channel"), _all_cases())
def test_every_channel_pairing_is_exactly_reciprocal_eager(
    name: str, source_channel: str, receiver_channel: str
) -> None:
    """The eager backend obeys the same relations as the compiled one."""
    _assert_reciprocal(name, source_channel, receiver_channel, "eager")


@pytest.mark.parametrize(
    ("name", "channel"),
    [
        (name, channel)
        for name in LIKE_FOR_LIKE
        for channel in deepwave.spec(name).receiver_channels(2)
    ],
)
def test_like_for_like_pairings_are_exactly_reciprocal(name: str, channel: str) -> None:
    """Swapping a channel with itself is exact on every propagator."""
    _assert_reciprocal(
        name,
        _dual(name, channel),
        channel,
        "auto",
        # The scalar/scalar and velocity/velocity relations carry no shift, so
        # a neighbouring shift is a meaningful miss; the stress channels of
        # acoustic_vti are covered the same way.
        check_vacuity=True,
    )


@pytest.mark.parametrize("name", ["acoustic_vti", "elastic_vti"])
def test_mixed_pairings_are_reciprocal_in_the_isotropic_limit(name: str) -> None:
    """Anisotropy, not the discretisation, breaks the mixed pairings."""
    one = _ones()
    c33 = RHO * VP**2
    c55 = RHO * VS**2
    if name == "acoustic_vti":
        # An isotropic fluid: no shear stiffness, so the bulk modulus is c33.
        isotropic = (c33 * one, c33 * one, c33 * one, one / RHO)
        anisotropic = (1.1 * c33 * one, 0.9 * c33 * one, c33 * one, one / RHO)
        modulus = c33
    else:
        c13 = c33 - 2 * c55
        isotropic = (c33 * one, c13 * one, c33 * one, c55 * one, one / RHO)
        anisotropic = (
            1.1 * c33 * one,
            0.9 * c13 * one,
            c33 * one,
            c55 * one,
            one / RHO,
        )
        # lamb + 2 mu / ndim with lamb = c13, mu = c55, in 2-D.
        modulus = c13 + c55
    scalar_source = deepwave.spec(name).source_channels(2)[0]
    _assert_reciprocal(
        name, scalar_source, "x", "auto", models=isotropic, modulus=modulus
    )
    with pytest.raises(AssertionError):
        _assert_reciprocal(
            name, scalar_source, "x", "auto", models=anisotropic, modulus=modulus
        )


@pytest.mark.parametrize(
    "name",
    [
        "viscoacoustic",
    ],
)
def test_mixed_pairings_are_reciprocal_in_the_lossless_limit(name: str) -> None:
    """The mixed-pairing departure is O(1/Q) and vanishes as Q grows."""
    one = _ones()
    scalar_source = deepwave.spec(name).source_channels(2)[0]
    residuals = []
    for q in (40.0, 400.0):
        models: tuple[torch.Tensor, ...] = (VP * one, RHO * one, q * one)
        f = _trace(name, scalar_source, LOC_A, "x", LOC_B, "auto", models=models)
        g = _trace(name, "x", LOC_B, "p", LOC_A, "auto", models=models)
        shift, scale = _expected(name, scalar_source, "x")
        residuals.append(_residual(f, g, shift, scale))
    assert residuals[0] > VACUITY_FLOOR
    assert residuals[1] < residuals[0] / 5


def _born_traces(
    perturbation: dict[str, float],
    source_channel: str,
    source_location: torch.Tensor,
    receiver_channel: str,
    receiver_location: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``elastic_born``'s scattered and background traces for one shot."""
    one = _ones()
    kwargs: dict[str, object] = {
        "pml_width": 10,
        "pml_freq": FREQ,
        "max_vel": 2.0 * VP,
        "accuracy": 4,
        "source_amplitudes_" + source_channel: _wavelet(),
        "source_locations_" + source_channel: source_location,
        "sc_receiver_locations_" + receiver_channel: receiver_location,
        "bg_receiver_locations_" + receiver_channel: receiver_location,
    }
    kwargs.update({name: value * one for name, value in perturbation.items()})
    out = deepwave.elastic_born(LAMB * one, MU * one, one / RHO, DX, DT, **kwargs)  # type: ignore[arg-type]
    scattered = getattr(out, "sc_receiver_amplitudes_" + receiver_channel)[0, 0]
    background = getattr(out, "bg_receiver_amplitudes_" + receiver_channel)[0, 0]
    return scattered, background


@pytest.mark.parametrize(
    "perturbation",
    [
        {"dlamb": 1.0e8},
        {"dmu": 5.0e7},
        {"dbuoyancy": 1.0e-5},
        {"dlamb": 1.0e8, "dmu": 5.0e7, "dbuoyancy": 1.0e-5},
    ],
    ids=["dlamb", "dmu", "dbuoyancy", "all"],
)
def test_born_scattered_field_obeys_the_linearised_relation(
    perturbation: dict[str, float],
) -> None:
    """The scattered field obeys the derivative of the mixed relation.

    ``elastic_born`` linearises ``elastic``, so differentiating that
    propagator's ``K f[n] == -g[n - 1]`` gives

        ``K df[n] + dK f0[n] == -dg[n - 1]``

    with ``f0`` the background trace of the scalar-source pairing and ``dK``
    the bulk modulus' own perturbation, ``dlamb + 2 dmu / ndim``. Dropping
    the ``dK f0`` term (which vanishes only for a pure buoyancy perturbation)
    leaves a residual of order one, so this also pins the sign and scale of
    the perturbation's contribution.
    """
    modulus_perturbation = perturbation.get("dlamb", 0.0) + perturbation.get("dmu", 0.0)
    # Scalar source at A, velocity receiver at B.
    scattered_f, background_f = _born_traces(
        perturbation, "pressure_rate", LOC_A, "x", LOC_B
    )
    # The dual pairing: force at B, pressure receiver at A.
    scattered_g, _ = _born_traces(perturbation, "x", LOC_B, "p", LOC_A)

    lhs = ELASTIC_BULK_MODULUS * scattered_f[1:]
    rhs = -scattered_g[:-1] - modulus_perturbation * background_f[1:]
    residual = float(
        torch.linalg.vector_norm(lhs - rhs) / torch.linalg.vector_norm(lhs)
    )
    assert residual < RTOL, f"linearised reciprocity residual {residual:.3e}"

    if modulus_perturbation:
        # Without the background term the relation misses by order one.
        without = float(
            torch.linalg.vector_norm(lhs + scattered_g[:-1])
            / torch.linalg.vector_norm(lhs)
        )
        assert without > VACUITY_FLOOR
