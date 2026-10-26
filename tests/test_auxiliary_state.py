"""Tests for ``deepwave.AuxiliaryState`` and the folded propagator return.

The absorbing boundary's memory variables are carried in one opaque
``auxiliary_state`` field rather than as individual public outputs. These tests
cover that object, the ``PropagatorOutput`` surface that exposes it
(``names``/``as_dict``/``final_state``/``info``), the compatibility
fingerprint that rejects a restart the state does not belong to, and
``RunInfo``.

``PROPS`` lists the propagators whose return is folded; the spec-level
invariants below run over every propagator regardless, since they only read
the ``PropagatorSpec`` and never start a simulation.
"""

import contextlib
import copy
import dataclasses
import pickle
import re
import warnings
from typing import Optional

import pytest
import torch

import deepwave
from deepwave import _engine
from deepwave.checkpointing import _get_spec, run_segments

PROPS = (
    "scalar",
    "acoustic",
    "scalar_born",
    "elastic",
    "acoustic_born",
    "elastic_born",
    "acoustic_vti",
    "elastic_vti",
    "viscoacoustic",
)

# The subset of ``PROPS`` that ``v0.0.27`` released (``status="stable"``).
# What is special about these four is their *positional* return: it still
# holds the auxiliary fields in the same slots, which is what keeps a
# ``v0.0.27`` unpacking working. Neither the individual names as output
# attributes nor the individual names as ``initial_state`` keys is accepted on
# any of the nine -- neither spelling ever shipped.
RELEASED_PROPS = ("scalar", "acoustic", "scalar_born", "elastic")

ALL_PROPS = (
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

DTYPE = torch.float64


def _spec(name: str) -> _engine.PropagatorSpec:
    return _get_spec(getattr(deepwave, name))


def _spec_ndims(name: str) -> tuple[int, ...]:
    """The spatial dimensionalities whose spec ``name`` answers for.

    ``acoustic_vti``'s names functions raise in 1-D (the ``V_S = 0`` reduction
    degenerates to the 1-D scalar equation, and the message says to use
    ``deepwave.acoustic`` instead), so the spec-level invariants below skip
    that combination. Every other propagator answers for 1, 2 and 3 -- note
    these are spec-level invariants and never start a simulation.
    """
    return (2, 3) if name == "acoustic_vti" else (1, 2, 3)


_SPEC_CASES = [(name, ndim) for name in PROPS for ndim in _spec_ndims(name)]


def _runtime_ndims(name: str) -> tuple[int, ...]:
    """The spatial dimensionalities ``name`` can actually be *run* at.

    The run tests use the shared supported dimensions of the retained
    propagators, while spec-level tests also cover one-dimensional layouts.
    """
    return (2, 3)


_RUNTIME_CASES = [(name, ndim) for name in PROPS for ndim in _runtime_ndims(name)]


# The source-amplitude argument and the receiver-amplitude output field the
# tests below drive each propagator through. Every propagator in ``PROPS``
# needs an entry.
_SOURCE_ARG = {
    "scalar": "source_amplitudes",
    "scalar_born": "source_amplitudes",
    "acoustic": "source_amplitudes_volume_rate",
    "elastic": "source_amplitudes_y",
    "acoustic_born": "source_amplitudes_volume_rate",
    "elastic_born": "source_amplitudes_y",
    "acoustic_vti": "source_amplitudes_sigma_h",
    "elastic_vti": "source_amplitudes_y",
    "viscoacoustic": "source_amplitudes_pressure_rate",
}

_RECEIVER_FIELD = {
    "scalar": "receiver_amplitudes",
    "scalar_born": "sc_receiver_amplitudes",
    "acoustic": "receiver_amplitudes_p",
    "elastic": "receiver_amplitudes_y",
    "acoustic_born": "sc_receiver_amplitudes_p",
    "elastic_born": "sc_receiver_amplitudes_y",
    "acoustic_vti": "receiver_amplitudes_sigma_h",
    "elastic_vti": "receiver_amplitudes_y",
    "viscoacoustic": "receiver_amplitudes_p",
}


def _problem(name: str, ndim: int = 2) -> dict:
    """A tiny ``ndim``-dimensional problem for ``name``, with a step_ratio of 1.

    The 2-D form is the default because almost every invariant below is
    dimension-independent and 2-D is far cheaper. The 3-D form exists for the
    handful that are not -- the chained restart above all, whose per-axis
    absorbing-boundary memory only has a third axis in 3-D -- and is
    deliberately small (a 14-cell cube with a 5-cell PML) so that nine
    propagators' worth of it stays affordable. Callers must go through
    ``_runtime_ndims``: not every propagator whose *spec* answers for 3-D can
    be *run* there.
    """
    if ndim == 3:
        shape = (14, 14, 14)
        centre, corner, offset = 7, 3, 10
    else:
        shape = (24, 24)
        centre, corner, offset = 12, 4, 18
    ones = torch.ones(shape, dtype=DTYPE)
    v = torch.full(shape, 1500.0, dtype=DTYPE)
    v[centre:] = 1700.0
    sources = torch.tensor([[[centre] * ndim]])
    receivers = torch.tensor(
        [[[corner] * ndim, [corner] * (ndim - 1) + [offset]]],
    )
    shared: dict = {"pml_freq": 15.0, "max_vel": 2000.0}
    if ndim == 3:
        # A narrower absorbing boundary than the default 20: the padded grid
        # would otherwise be 54 cells a side and dominate the run time, and
        # nothing here depends on the boundary being wide.
        shared["pml_width"] = 5

    def _scatter(value: float) -> torch.Tensor:
        """A compact anomaly below the source, wherever the model is.

        In 2-D this is the ``[14:18, 8:16]`` box these tests have always used;
        the 3-D form is the same box, extended along the extra axis.
        """
        block = torch.zeros(shape, dtype=DTYPE)
        below = slice(centre + 2, centre + 6)
        across = slice(centre - 4, centre + 4)
        block[(below, *((across,) * (ndim - 1)))] = value
        return block

    if name in ("scalar", "scalar_born"):
        models = [v]
        kwargs = {
            "source_locations": sources,
            "receiver_locations": receivers,
            **shared,
        }
        if name == "scalar_born":
            scatter = _scatter(100.0)
            kwargs["dv"] = scatter
            kwargs["bg_receiver_locations"] = receivers
            kwargs["sc_receiver_locations"] = kwargs.pop("receiver_locations")
    elif name == "acoustic":
        models = [v, 2200.0 * ones]
        kwargs = {
            "source_locations_volume_rate": sources,
            "receiver_locations_p": receivers,
            **shared,
        }
    elif name == "acoustic_born":
        scatter = _scatter(100.0)
        models = [v, 2200.0 * ones]
        kwargs = {
            "dv": scatter,
            "source_locations_volume_rate": sources,
            "bg_receiver_locations_p": receivers,
            "sc_receiver_locations_p": receivers,
            **shared,
        }
    elif name == "elastic":
        models = [2.25e9 * ones, 1.0e9 * ones, (1 / 2200) * ones]
        kwargs = {
            "source_locations_y": sources,
            "receiver_locations_y": receivers,
            **shared,
        }
    elif name == "elastic_born":
        scatter = _scatter(1.0)
        models = [2.25e9 * ones, 1.0e9 * ones, (1 / 2200) * ones]
        kwargs = {
            "dlamb": 1e7 * scatter,
            "dmu": 5e6 * scatter,
            "dbuoyancy": (1e-6) * scatter,
            "source_locations_y": sources,
            "bg_receiver_locations_y": receivers,
            "sc_receiver_locations_y": receivers,
            **shared,
        }
    elif name in (
        "acoustic_vti",
        "elastic_vti",
    ):
        # The pseudo-acoustic block requires c13**2 <= c11*c33; the 3-D
        # elastic normal block requires c13**2 < (c11-c66)*c33 as well.
        # Keep both positive, with every phase velocity under max_vel.
        c33 = 2200.0 * 1500.0**2 * ones
        c11 = 1.2 * c33
        c13 = 0.8 * (c11 * c33).sqrt()
        buoyancy = (1 / 2200) * ones
        if name == "acoustic_vti":
            models = [c11, c13, c33, buoyancy]
            kwargs = {
                "source_locations_sigma_h": sources,
                "receiver_locations_sigma_h": receivers,
                **shared,
            }
        else:
            models = [c11, c13, c33, 0.3 * c33, buoyancy]
            kwargs = {
                "source_locations_y": sources,
                "receiver_locations_y": receivers,
                **shared,
            }
            if name == "elastic_vti" and ndim == 3:
                # The horizontal-plane shear stiffness has no 2-D analogue
                # (there is no sigma_xy in the 2-D P-SV equations), so it is a
                # keyword argument that 3-D requires and 2-D refuses.
                kwargs["c66"] = 0.3 * c33
                # The 3-D Christoffel trace bound is sqrt(1.8)*1500,
                # slightly above the common 2000 fixture bound.
                kwargs["max_vel"] = 2200.0
    elif name == "viscoacoustic":
        # A finite Q, so the GSLS memory variable is actually driven. These
        # invariants run at the default one mechanism; the multi-mechanism
        # field order is pinned in the propagator's own test file, since it
        # is the only place the mechanism regrouping is observable.
        models = [v, 2200.0 * ones, 40.0 * ones]
        kwargs = {
            "source_locations_pressure_rate": sources,
            "receiver_locations_p": receivers,
            # The frequency Q is specified at; it has no default.
            "q_freq": 15.0,
            **shared,
        }
    else:  # pragma: no cover - a new propagator needs a problem here
        raise AssertionError(f"no test problem for {name}")
    dx = 5.0
    # The CFL limit tightens with the number of dimensions: 0.001 s needs two
    # internal steps per sample in 3-D, and a restart is only bit-exact across
    # a segment boundary when there is exactly one (the boundary would
    # otherwise fall mid-substep). ``_assert_step_ratio_is_one`` pins this.
    dt = 0.0007 if ndim == 3 else 0.001
    return {"models": models, "dx": dx, "dt": dt, "kwargs": kwargs}


def _assert_step_ratio_is_one(problem: dict, ndim: int) -> None:
    """The premise every bit-exact-restart assertion below rests on.

    With ``step_ratio > 1`` a segment boundary does not land on an internal
    time step and the two-segment run is only approximately the single call,
    so an equality asserted on such a problem would be testing the wrong
    thing (or silently relaxed later to make it pass).
    """
    _, step_ratio = deepwave.discretisation.cfl_condition_n(
        [problem["dx"]] * ndim,
        problem["dt"],
        problem["kwargs"]["max_vel"],
    )
    assert step_ratio == 1, step_ratio


def _wavelet(nt: int) -> torch.Tensor:
    """The tiny problem's source wavelet, ``nt`` samples long."""
    return (
        deepwave.wavelets.ricker(15.0, nt, 0.001, 1.0 / 15.0)
        .to(DTYPE)
        .reshape(1, 1, -1)
    )


def _run(name: str, nt: int = 12, **extra) -> deepwave.PropagatorOutput:
    """One call of ``name`` on the tiny problem."""
    problem = _problem(name)
    kwargs = dict(problem["kwargs"])
    kwargs.setdefault(_SOURCE_ARG[name], _wavelet(nt))
    kwargs.update(extra)
    return getattr(deepwave, name)(
        *problem["models"],
        problem["dx"],
        problem["dt"],
        **kwargs,
    )


def _aux(name: str, ndim: int = 2) -> str:
    """One folded auxiliary-state field name of ``name``."""
    return _spec(name).auxiliary_state_names(ndim)[0]


def _receivers(out: deepwave.PropagatorOutput, name: str) -> torch.Tensor:
    """The receiver amplitudes the tests compare, whatever they are called."""
    tensor: torch.Tensor = out.as_dict()[_RECEIVER_FIELD[name]]
    return tensor


def _physical_state(out: deepwave.PropagatorOutput) -> dict:
    """The public state fields other than ``auxiliary_state``."""
    return {
        field: value
        for field, value in out.final_state.items()
        if field != "auxiliary_state"
    }


# ---------------------------------------------------------------------------
# Spec-level invariants (no simulation is run)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_PROPS)
@pytest.mark.parametrize("ndim", [2, 3])
def test_auxiliary_state_names_subset_of_wavefield_names(name: str, ndim: int) -> None:
    """Every folded name is one of the propagator's public state names.

    The collapse is computed in the public order and spelling (that is where
    "the index of the first auxiliary field" is meaningful), so a folded name
    that is not a public state name would fold nothing.
    """
    spec = _spec(name)
    assert set(spec.auxiliary_state_names(ndim)) <= set(spec.public_state_names(ndim))


@pytest.mark.parametrize(("name", "ndim"), _SPEC_CASES)
def test_output_names_are_public_names_with_block_collapsed(
    name: str, ndim: int
) -> None:
    """``output_names`` is the public state order with the block collapsed."""
    spec = _spec(name)
    state = spec.public_state_names(ndim)
    collapsed = _engine.collapse_auxiliary_state_names(
        state, spec.auxiliary_state_names(ndim)
    )
    names = spec.output_names(ndim)
    assert names[: len(collapsed)] == collapsed
    # Everything after the state block is a receiver field.
    assert set(names[len(collapsed) :]) == set(
        _engine.receiver_output_names(spec)
    ) & set(names)


@pytest.mark.parametrize("name", ALL_PROPS)
@pytest.mark.parametrize("ndim", [2, 3])
def test_output_names_hold_every_state_field(name: str, ndim: int) -> None:
    """Every public state field is either an output name or folded away."""
    spec = _spec(name)
    folded = set(spec.auxiliary_state_names(ndim))
    names = set(spec.output_names(ndim))
    kept = {n for n in spec.public_state_names(ndim) if n not in folded}
    assert kept <= names
    assert not (folded & names)
    assert ("auxiliary_state" in names) == bool(folded)


@pytest.mark.parametrize(
    ("name", "ndim", "expected"),
    [
        ("scalar", 2, 6),
        ("scalar", 3, 8),
        ("scalar_born", 2, 12),
        ("scalar_born", 3, 16),
        ("acoustic", 2, 7),
        ("acoustic", 3, 10),
        ("elastic", 2, 13),
        ("elastic", 3, 27),
    ],
)
def test_estimate_memory_wavefield_count_matches_the_spec(
    name: str, ndim: int, expected: int
) -> None:
    """``estimate_memory`` counts *internal* wavefields, which the fold keeps.

    The fold changes only how the final state is presented; the number of
    wavefields the forward pass allocates is unchanged, so the memory
    estimate's per-propagator count must still equal the spec's.

    These eight numbers were a hand-maintained table in ``_common`` that the
    estimate consulted. The estimate now reads the spec directly, for all
    nine propagators rather than these four, so the table is gone -- but
    the numbers it held are written out here so that the spec cannot drift
    away from what these four propagators have always allocated.
    """
    assert _spec(name).n_wavefields(ndim) == expected


def test_boundary_condition_names_match_the_code_table() -> None:
    """The engine's face-code -> name table agrees with the parser's."""
    from deepwave._common import _BOUNDARY_NAME_TO_CODE

    for label, code in _BOUNDARY_NAME_TO_CODE.items():
        assert _engine.BOUNDARY_CONDITION_NAMES[code] == label
    assert len(_engine.BOUNDARY_CONDITION_NAMES) == len(_BOUNDARY_NAME_TO_CODE)


@pytest.mark.parametrize("name", ALL_PROPS)
def test_receiver_output_names_excludes_auxiliary_state(name: str) -> None:
    """``auxiliary_state`` is state, not a receiver output.

    ``receiver_output_names`` is passed as the set of ``initial_state`` keys to
    ignore, so classing ``auxiliary_state`` as a receiver would make
    ``initial_state=out`` silently discard the entire auxiliary state.
    """
    assert "auxiliary_state" not in _engine.receiver_output_names(_spec(name))


@pytest.mark.parametrize(("name", "ndim"), _SPEC_CASES)
def test_no_public_field_name_is_a_boundary_auxiliary(name: str, ndim: int) -> None:
    """No public output name is one of the folded auxiliary spellings."""
    for field in _spec(name).output_names(ndim):
        assert not field.startswith(("psi", "zeta", "phi_", "m_")), field


# The exact public field list of every folded propagator, hard-coded. Nothing
# else catches a wrongly-ordered collapse: every state tensor has the same
# shape, so the wrong ones landing in ``auxiliary_state`` is silent.
_SCALAR_BORN_OUTPUT_NAMES = [
    "wavefield_0",
    "wavefield_m1",
    "auxiliary_state",
    "wavefield_sc_0",
    "wavefield_sc_m1",
    "bg_receiver_amplitudes",
    "sc_receiver_amplitudes",
]

_EXPECTED_OUTPUT_NAMES = {
    ("viscoacoustic", 1): [
        "pressure_0",
        "vx_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_x",
    ],
    ("viscoacoustic", 2): [
        "pressure_0",
        "vy_0",
        "vx_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_y",
        "receiver_amplitudes_x",
    ],
    ("viscoacoustic", 3): [
        "pressure_0",
        "vz_0",
        "vy_0",
        "vx_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_z",
        "receiver_amplitudes_y",
        "receiver_amplitudes_x",
    ],
    ("scalar", 1): [
        "wavefield_0",
        "wavefield_m1",
        "auxiliary_state",
        "receiver_amplitudes",
    ],
    ("scalar", 2): [
        "wavefield_0",
        "wavefield_m1",
        "auxiliary_state",
        "receiver_amplitudes",
    ],
    ("scalar", 3): [
        "wavefield_0",
        "wavefield_m1",
        "auxiliary_state",
        "receiver_amplitudes",
    ],
    ("acoustic", 1): [
        "pressure_0",
        "vx_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_x",
    ],
    ("acoustic", 2): [
        "pressure_0",
        "vy_0",
        "vx_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_y",
        "receiver_amplitudes_x",
    ],
    ("acoustic", 3): [
        "pressure_0",
        "vz_0",
        "vy_0",
        "vx_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_z",
        "receiver_amplitudes_y",
        "receiver_amplitudes_x",
    ],
    # scalar_born's two auxiliary blocks are not contiguous: the single
    # auxiliary_state takes the position of the first of them and the
    # scattered physical fields keep their relative order after it.
    ("scalar_born", 1): _SCALAR_BORN_OUTPUT_NAMES,
    ("scalar_born", 2): _SCALAR_BORN_OUTPUT_NAMES,
    ("scalar_born", 3): _SCALAR_BORN_OUTPUT_NAMES,
    # elastic's public order groups the fields by kind while the backend
    # interleaves them by dimension, so auxiliary_state takes the position of
    # the first auxiliary field in the *public* order.
    ("elastic", 1): [
        "vx_0",
        "sigmaxx_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_x",
    ],
    ("elastic", 2): [
        "vy_0",
        "vx_0",
        "sigmayy_0",
        "sigmaxy_0",
        "sigmaxx_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_y",
        "receiver_amplitudes_x",
    ],
    ("elastic", 3): [
        "vz_0",
        "vy_0",
        "vx_0",
        "sigmazz_0",
        "sigmayz_0",
        "sigmaxz_0",
        "sigmayy_0",
        "sigmaxy_0",
        "sigmaxx_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_z",
        "receiver_amplitudes_y",
        "receiver_amplitudes_x",
    ],
    # The Born propagators' scattered physical fields are spelled publicly
    # with `sc` before the time index, and their two auxiliary blocks (the
    # background one and the scattered one) collapse into the single
    # auxiliary_state at the position of the first of them.
    ("acoustic_born", 1): [
        "pressure_0",
        "vx_0",
        "auxiliary_state",
        "pressure_sc_0",
        "vx_sc_0",
        "bg_receiver_amplitudes_p",
        "bg_receiver_amplitudes_x",
        "sc_receiver_amplitudes_p",
        "sc_receiver_amplitudes_x",
    ],
    ("acoustic_born", 2): [
        "pressure_0",
        "vy_0",
        "vx_0",
        "auxiliary_state",
        "pressure_sc_0",
        "vy_sc_0",
        "vx_sc_0",
        "bg_receiver_amplitudes_p",
        "bg_receiver_amplitudes_y",
        "bg_receiver_amplitudes_x",
        "sc_receiver_amplitudes_p",
        "sc_receiver_amplitudes_y",
        "sc_receiver_amplitudes_x",
    ],
    ("acoustic_born", 3): [
        "pressure_0",
        "vz_0",
        "vy_0",
        "vx_0",
        "auxiliary_state",
        "pressure_sc_0",
        "vz_sc_0",
        "vy_sc_0",
        "vx_sc_0",
        "bg_receiver_amplitudes_p",
        "bg_receiver_amplitudes_z",
        "bg_receiver_amplitudes_y",
        "bg_receiver_amplitudes_x",
        "sc_receiver_amplitudes_p",
        "sc_receiver_amplitudes_z",
        "sc_receiver_amplitudes_y",
        "sc_receiver_amplitudes_x",
    ],
    ("elastic_born", 1): [
        "vx_0",
        "sigmaxx_0",
        "auxiliary_state",
        "vx_sc_0",
        "sigmaxx_sc_0",
        "bg_receiver_amplitudes_p",
        "bg_receiver_amplitudes_x",
        "sc_receiver_amplitudes_p",
        "sc_receiver_amplitudes_x",
    ],
    ("elastic_born", 2): [
        "vy_0",
        "vx_0",
        "sigmayy_0",
        "sigmaxy_0",
        "sigmaxx_0",
        "auxiliary_state",
        "vy_sc_0",
        "vx_sc_0",
        "sigmayy_sc_0",
        "sigmaxy_sc_0",
        "sigmaxx_sc_0",
        "bg_receiver_amplitudes_p",
        "bg_receiver_amplitudes_y",
        "bg_receiver_amplitudes_x",
        "sc_receiver_amplitudes_p",
        "sc_receiver_amplitudes_y",
        "sc_receiver_amplitudes_x",
    ],
    ("elastic_born", 3): [
        "vz_0",
        "vy_0",
        "vx_0",
        "sigmazz_0",
        "sigmayz_0",
        "sigmaxz_0",
        "sigmayy_0",
        "sigmaxy_0",
        "sigmaxx_0",
        "auxiliary_state",
        "vz_sc_0",
        "vy_sc_0",
        "vx_sc_0",
        "sigmazz_sc_0",
        "sigmayz_sc_0",
        "sigmaxz_sc_0",
        "sigmayy_sc_0",
        "sigmaxy_sc_0",
        "sigmaxx_sc_0",
        "bg_receiver_amplitudes_p",
        "bg_receiver_amplitudes_z",
        "bg_receiver_amplitudes_y",
        "bg_receiver_amplitudes_x",
        "sc_receiver_amplitudes_p",
        "sc_receiver_amplitudes_z",
        "sc_receiver_amplitudes_y",
        "sc_receiver_amplitudes_x",
    ],
    # acoustic_vti's public order is its backend order (no permutation), and
    # its whole m_* block is contiguous at the end of the state fields. It has
    # no 1-D form.
    ("acoustic_vti", 2): [
        "vy_0",
        "vx_0",
        "sigma_v_0",
        "sigma_h_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_y",
        "receiver_amplitudes_x",
        "receiver_amplitudes_sigma_v",
        "receiver_amplitudes_sigma_h",
    ],
    ("acoustic_vti", 3): [
        "vz_0",
        "vy_0",
        "vx_0",
        "sigma_v_0",
        "sigma_h_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_z",
        "receiver_amplitudes_y",
        "receiver_amplitudes_x",
        "receiver_amplitudes_sigma_v",
        "receiver_amplitudes_sigma_h",
    ],
    # elastic_vti shares elastic's wavefield layout and fold set. Written out
    # rather than derived from elastic's entries, so a changed permutation is
    # caught here.
    ("elastic_vti", 1): [
        "vx_0",
        "sigmaxx_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_x",
    ],
    ("elastic_vti", 2): [
        "vy_0",
        "vx_0",
        "sigmayy_0",
        "sigmaxy_0",
        "sigmaxx_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_y",
        "receiver_amplitudes_x",
    ],
    ("elastic_vti", 3): [
        "vz_0",
        "vy_0",
        "vx_0",
        "sigmazz_0",
        "sigmayz_0",
        "sigmaxz_0",
        "sigmayy_0",
        "sigmaxy_0",
        "sigmaxx_0",
        "auxiliary_state",
        "receiver_amplitudes_p",
        "receiver_amplitudes_z",
        "receiver_amplitudes_y",
        "receiver_amplitudes_x",
    ],
}


@pytest.mark.parametrize(("name", "ndim"), sorted(_EXPECTED_OUTPUT_NAMES))
def test_public_output_names_are_exact(name: str, ndim: int) -> None:
    """The exact public field list of each folded propagator, hard-coded."""
    assert _spec(name).output_names(ndim) == _EXPECTED_OUTPUT_NAMES[name, ndim]


# ---------------------------------------------------------------------------
# AuxiliaryState behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPS)
def test_auxiliary_state_repr_has_no_tensor_values(name: str) -> None:
    """The repr is one line naming the run, with no tensor contents."""
    text = repr(_run(name).auxiliary_state)
    assert "\n" not in text
    assert "tensor(" not in text
    assert name in text
    assert "pml_width=" in text
    # A whole output prints on one short line too.
    assert "\n" not in repr(_run(name))


@pytest.mark.parametrize("name", PROPS)
def test_auxiliary_state_is_not_a_tensor(name: str) -> None:
    """It is a plain object, never a Tensor."""
    assert not isinstance(_run(name).auxiliary_state, torch.Tensor)


@pytest.mark.parametrize("name", PROPS)
def test_auxiliary_state_is_read_only(name: str) -> None:
    """No item assignment, no field assignment, and ``as_dict`` is a copy."""
    state = _run(name).auxiliary_state
    with pytest.raises(TypeError):
        state[_aux(name)] = torch.zeros(1)  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.accuracy = 8  # type: ignore[misc]
    copied = state.as_dict()
    copied.clear()
    assert len(state) > 0


@pytest.mark.parametrize("name", PROPS)
def test_auxiliary_state_mapping_surface(name: str) -> None:
    """names/keys/items/values/len/in/iteration all agree."""
    state = _run(name).auxiliary_state
    assert state.names == tuple(state.keys())
    assert list(iter(state)) == list(state.names)
    assert len(state) == len(state.names)
    for field in state.names:
        assert field in state
        assert state[field] is state.as_dict()[field]
    assert "not_a_field" not in state
    with pytest.raises(KeyError, match="not part of this auxiliary state"):
        _ = state["not_a_field"]


@pytest.mark.parametrize("name", PROPS)
def test_auxiliary_state_equality_does_not_raise(name: str) -> None:
    """``==`` returns a bool, so an auxiliary state works in ``if`` and ``in``.

    A generated ``__eq__`` would compare the tensors and return a Tensor,
    which would raise inside any truth test.
    """
    state = _run(name).auxiliary_state
    other = _run(name).auxiliary_state
    assert isinstance(state == other, bool)
    assert state == state
    assert state != other
    assert state in [state]


@pytest.mark.parametrize("name", PROPS)
def test_restart_tensors_are_identical_objects(name: str) -> None:
    """The auxiliary state holds the engine's own tensors, uncopied."""
    out = _run(name)
    state = out.auxiliary_state
    assert state.as_dict()[_aux(name)] is state[_aux(name)]
    # The same object survives a round trip through final_state.
    assert out.final_state["auxiliary_state"] is state


@pytest.mark.parametrize("name", PROPS)
def test_auxiliary_state_to_is_a_no_op_without_arguments(name: str) -> None:
    """``.to()`` with nothing to change returns the same object."""
    state = _run(name).auxiliary_state
    assert state.to() is state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("name", PROPS)
def test_auxiliary_state_to_device_roundtrip_restarts_bit_exactly(name: str) -> None:
    """A CPU round trip of the auxiliary state does not change the restart."""
    problem = _problem(name)
    device = torch.device("cuda:0")
    models = [m.to(device) for m in problem["models"]]
    src = _wavelet(24).to(device)
    kwargs = {
        k: (t.to(device) if torch.is_tensor(t) else t)
        for k, t in problem["kwargs"].items()
    }
    prop = getattr(deepwave, name)
    source_arg = _SOURCE_ARG[name]
    first = prop(
        *models,
        problem["dx"],
        problem["dt"],
        **{source_arg: src[..., :12]},
        **kwargs,
    )
    direct = prop(
        *models,
        problem["dx"],
        problem["dt"],
        **{source_arg: src[..., 12:]},
        initial_state=first,
        **kwargs,
    )
    moved = dict(first.final_state)
    moved["auxiliary_state"] = moved["auxiliary_state"].to("cpu").to(device)
    viacpu = prop(
        *models,
        problem["dx"],
        problem["dt"],
        **{source_arg: src[..., 12:]},
        initial_state=moved,
        **kwargs,
    )
    assert torch.equal(_receivers(direct, name), _receivers(viacpu, name))
    for field, tensor in direct.auxiliary_state.items():
        assert torch.equal(tensor, viacpu.auxiliary_state[field]), field


# ---------------------------------------------------------------------------
# The compatibility fingerprint
# ---------------------------------------------------------------------------


def _restart_with(name: str, state: deepwave.AuxiliaryState, **extra):
    """A second segment started from a (possibly doctored) auxiliary state."""
    first = _run(name)
    bundle = dict(first.final_state)
    bundle["auxiliary_state"] = state
    return _run(name, initial_state=bundle, **extra)


@pytest.mark.parametrize("name", PROPS)
def test_mismatched_accuracy_restart_raises(name: str) -> None:
    """A restart at a different accuracy names both values and raises."""
    out = _run(name)
    with pytest.raises(deepwave.DeepwaveValueError, match="accuracy") as excinfo:
        _restart_with(name, out.auxiliary_state, accuracy=8)
    assert "accuracy=4" in str(excinfo.value)
    assert "accuracy=8" in str(excinfo.value)


@pytest.mark.parametrize("name", PROPS)
def test_mismatched_pml_width_restart_raises(name: str) -> None:
    """A restart with a different PML width names both widths and raises."""
    out = _run(name)
    doctored = dataclasses.replace(out.auxiliary_state, pml_width=(10, 10, 10, 10))
    with pytest.raises(deepwave.DeepwaveValueError, match="pml_width") as excinfo:
        _restart_with(name, doctored)
    assert "(20, 20, 20, 20)" in str(excinfo.value)
    assert "(10, 10, 10, 10)" in str(excinfo.value)


@pytest.mark.parametrize("name", PROPS)
def test_actually_changing_pml_width_hits_the_shape_check_first(name: str) -> None:
    """The pre-existing shape check fires before the fingerprint.

    A different ``pml_width`` also changes the simulation domain, and the
    wavefield shape is validated while the call's own ``pml_width`` is being
    resolved -- before there is a resolved value for the fingerprint to
    compare against. The error still points at the initial wavefield.
    """
    out = _run(name)
    with pytest.raises(RuntimeError, match="initial wavefield of the wrong shape"):
        _restart_with(name, out.auxiliary_state, pml_width=10)


@pytest.mark.parametrize("name", PROPS)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("propagator", "not_a_propagator"),
        ("ndim", 3),
        ("n_mechanisms", 2),
        ("mpml", True),
        ("scheme", "other-cpml"),
    ],
)
def test_mismatched_fingerprint_field_raises(name: str, field: str, value) -> None:
    """Every checked fingerprint field rejects a restart that contradicts it."""
    out = _run(name)
    doctored = dataclasses.replace(out.auxiliary_state, **{field: value})
    with pytest.raises(deepwave.DeepwaveValueError, match=field) as excinfo:
        _restart_with(name, doctored)
    assert repr(value) in str(excinfo.value)


@pytest.mark.parametrize("name", PROPS)
def test_mismatched_boundary_restart_raises(name: str) -> None:
    """A restart with different per-face boundary conditions raises."""
    out = _run(name)
    doctored = dataclasses.replace(
        out.auxiliary_state, boundary=("free", "pml", "pml", "pml")
    )
    with pytest.raises(deepwave.DeepwaveValueError, match="boundary"):
        _restart_with(name, doctored)


@pytest.mark.parametrize("name", PROPS)
def test_mismatched_pml_freq_rejects_continuation(name: str) -> None:
    """PML memories cannot be continued under a different profile."""
    out = _run(name)
    doctored = dataclasses.replace(out.auxiliary_state, pml_freq=30.0)
    with pytest.raises(deepwave.DeepwaveValueError, match="pml_freq"):
        _restart_with(name, doctored)


@pytest.mark.parametrize("name", PROPS)
def test_dtype_and_device_mismatch_use_the_existing_messages(name: str) -> None:
    """The fingerprint does not duplicate the wavefield consistency checks."""
    out = _run(name)
    bundle = dict(out.final_state)
    field = next(iter(_physical_state(out)))
    bundle[field] = bundle[field].to(torch.float32)
    with pytest.raises(RuntimeError, match="Inconsistent dtype"):
        _run(name, initial_state=bundle)


@pytest.mark.parametrize("name", PROPS)
def test_compatible_with_reports_the_same_answer(name: str) -> None:
    """``compatible_with`` is the non-raising form of the same comparison."""
    state = _run(name).auxiliary_state
    assert state.compatible_with(state)
    assert not state.compatible_with(dataclasses.replace(state, accuracy=8))
    # An unrecorded field is neither reported nor checked.
    assert state.compatible_with(dataclasses.replace(state, accuracy=None))
    # A different bundle layout is incompatible whatever else matches, and
    # is reported as such rather than only raising from a propagator call.
    assert not state.compatible_with(
        dataclasses.replace(
            state, format_version=_engine.AUXILIARY_STATE_FORMAT_VERSION + 1
        )
    )


# ---------------------------------------------------------------------------
# initial_state acceptance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPS)
def test_final_state_is_the_initial_state_keys(name: str) -> None:
    """``final_state`` is exactly the public state fields, and is read-only."""
    out = _run(name)
    spec = _spec(name)
    receivers = _engine.receiver_output_names(spec)
    assert list(out.final_state) == [n for n in out.names if n not in receivers]
    with pytest.raises(TypeError):
        out.final_state["not_a_field"] = torch.zeros(1)  # type: ignore[index]


@pytest.mark.parametrize("name", PROPS)
def test_final_state_restart_equals_output_restart(name: str) -> None:
    """``initial_state=out`` and ``initial_state=out.final_state`` agree."""
    first = _run(name)
    a = _run(name, initial_state=first)
    b = _run(name, initial_state=dict(first.final_state))
    assert torch.equal(_receivers(a, name), _receivers(b, name))
    for field, tensor in a.auxiliary_state.items():
        assert torch.equal(tensor, b.auxiliary_state[field]), field


def _assert_chained_restart_is_bit_exact(
    name: str,
    ndim: int,
    device: str = "cpu",
    backend: Optional[str] = None,
) -> None:
    """Two segments equal one call, field by field, for one configuration."""
    problem = _problem(name, ndim)
    _assert_step_ratio_is_one(problem, ndim)
    src = _wavelet(24).to(device)
    prop = getattr(deepwave, name)
    models = [model.to(device) for model in problem["models"]]
    kwargs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in problem["kwargs"].items()
    }
    if backend is not None:
        kwargs["backend"] = backend

    def call(amplitudes, **more):
        return prop(
            *models,
            problem["dx"],
            problem["dt"],
            **{_SOURCE_ARG[name]: amplitudes},
            **kwargs,
            **more,
        )

    full = call(src)
    seg1 = call(src[..., :10])
    seg2 = call(src[..., 10:], initial_state=seg1)
    state = _physical_state(full)
    assert state, f"{name} ndim {ndim}: no physical state fields to compare"
    assert len(full.auxiliary_state) > 0, (
        f"{name} ndim {ndim}: an empty auxiliary state would make this vacuous"
    )
    for field, tensor in state.items():
        assert torch.equal(tensor, getattr(seg2, field)), field
    for field, tensor in full.auxiliary_state.items():
        assert torch.equal(tensor, seg2.auxiliary_state[field]), field
    chained = torch.cat([_receivers(seg1, name), _receivers(seg2, name)], dim=-1)
    assert torch.equal(_receivers(full, name), chained)


@pytest.mark.parametrize(("name", "ndim"), _RUNTIME_CASES)
def test_chained_restart_is_bit_exact(name: str, ndim: int) -> None:
    """Two segments equal one call, field by field, in 2-D and in 3-D.

    3-D is not a formality here: it is the only dimensionality in which the
    absorbing boundary has a third per-axis memory group, so a fold that
    mislaid an axis would still restart exactly in 2-D.
    """
    _assert_chained_restart_is_bit_exact(name, ndim)


@pytest.mark.parametrize("name", ["scalar", "acoustic", "elastic"])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chained_restart_is_bit_exact_on_cuda(name: str, ndim: int) -> None:
    """The same restart, on the GPU: a different backend threads the state."""
    _assert_chained_restart_is_bit_exact(name, ndim, device="cuda")


@pytest.mark.parametrize("name", ["scalar", "acoustic", "elastic"])
@pytest.mark.parametrize("ndim", [2, 3])
def test_chained_restart_is_bit_exact_on_the_eager_backend(
    name: str, ndim: int
) -> None:
    """The same restart with ``backend="eager"``.

    The default compiled backend and the pure-Python eager one carry the
    auxiliary state through different code, so both need pinning.
    """
    _assert_chained_restart_is_bit_exact(name, ndim, backend="eager")


@pytest.mark.parametrize("name", PROPS)
def test_legacy_initial_state_keys_raise_naming_the_bundle(name: str) -> None:
    """The pre-fold auxiliary names are not ``initial_state`` keys.

    ``v0.0.27`` took each auxiliary field as its own keyword argument and had
    no ``initial_state`` at all, so no release ever accepted these as keys. One
    error per call, listing every offending key and naming the replacement, on
    all nine.
    """
    first = _run(name)
    legacy = {
        **_physical_state(first),
        **first.auxiliary_state.as_dict(),
    }
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        _run(name, initial_state=legacy)
    message = str(excinfo.value)
    assert "initial_state=out" in message
    for field in first.auxiliary_state.names:
        assert repr(field) in message


@pytest.mark.parametrize("name", PROPS)
def test_auxiliary_state_and_legacy_key_together_raises(name: str) -> None:
    """A field given both inside auxiliary_state and separately is ambiguous."""
    first = _run(name)
    bundle = dict(first.final_state)
    bundle[_aux(name)] = first.auxiliary_state[_aux(name)]
    with pytest.raises(deepwave.DeepwaveValueError, match="both inside initial_state"):
        _run(name, initial_state=bundle)


@pytest.mark.parametrize("name", PROPS)
def test_unknown_initial_state_key_names_the_public_keys(name: str) -> None:
    """A typo names the public state keys, including auxiliary_state."""
    with pytest.raises(deepwave.DeepwaveValueError, match="unexpected key") as excinfo:
        _run(name, initial_state={"wavefield_typo": torch.zeros(1, 64, 64)})
    message = str(excinfo.value)
    assert "auxiliary_state" in message
    assert _aux(name) not in message


@pytest.mark.parametrize("name", PROPS)
def test_auxiliary_state_entry_must_be_a_auxiliary_state(name: str) -> None:
    """A wrongly-typed auxiliary_state entry says what it should have been."""
    first = _run(name)
    bundle = dict(first.final_state)
    bundle["auxiliary_state"] = {_aux(name): torch.zeros(1)}
    with pytest.raises(deepwave.DeepwaveTypeError, match="must be a"):
        _run(name, initial_state=bundle)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPS)
def test_auxiliary_state_tensor_is_differentiable(name: str) -> None:
    """A loss taken on an auxiliary-state tensor reaches the model."""
    problem = _problem(name)
    models = [m.clone() for m in problem["models"]]
    models[0].requires_grad_(True)
    out = getattr(deepwave, name)(
        *models,
        problem["dx"],
        problem["dt"],
        **{_SOURCE_ARG[name]: _wavelet(12)},
        **problem["kwargs"],
    )
    out.auxiliary_state[_aux(name)].pow(2).sum().backward()
    assert models[0].grad is not None
    assert torch.any(models[0].grad != 0)


@pytest.mark.parametrize("name", PROPS)
def test_restart_gradient_flows_through_auxiliary_state(name: str) -> None:
    """A second segment's loss reaches the first segment's source."""
    src = _wavelet(24)
    first_amp = src[..., :10].clone().requires_grad_(True)
    seg1 = _run(name, **{_SOURCE_ARG[name]: first_amp})
    seg2 = _run(name, **{_SOURCE_ARG[name]: src[..., 10:]}, initial_state=seg1)
    _receivers(seg2, name).pow(2).sum().backward()
    assert first_amp.grad is not None
    assert torch.any(first_amp.grad != 0)


# ---------------------------------------------------------------------------
# torch.utils.checkpoint composition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPS)
@pytest.mark.parametrize("use_reentrant", [True, False])
def test_output_through_torch_checkpoint(name: str, use_reentrant: bool) -> None:
    """The documented return type and auxiliary-state gradient status.

    ``use_reentrant=True`` runs the wrapped function under ``torch.no_grad()``
    and re-attaches only the top-level ``Tensor`` elements of what it returns,
    so the subclass is lost and every tensor held inside ``auxiliary_state``
    comes back detached. ``use_reentrant=False`` preserves both. This pins the
    statement made in ``PropagatorOutput``'s docstring.
    """
    problem = _problem(name)
    v = problem["models"][0].clone().requires_grad_(True)
    src = _wavelet(12)

    def call(model):
        return getattr(deepwave, name)(
            model,
            *problem["models"][1:],
            problem["dx"],
            problem["dt"],
            **{_SOURCE_ARG[name]: src},
            **problem["kwargs"],
        )

    out = torch.utils.checkpoint.checkpoint(call, v, use_reentrant=use_reentrant)
    names = _spec(name).output_names(2)
    state = out[names.index("auxiliary_state")]
    aux = _aux(name)
    if use_reentrant:
        assert type(out) is tuple
        assert not state[aux].requires_grad
    else:
        assert isinstance(out, deepwave.PropagatorOutput)
        assert state[aux].grad_fn is not None
    # The receiver gradient is correct either way.
    out[names.index(_RECEIVER_FIELD[name])].pow(2).sum().backward()
    reference = call(problem["models"][0].clone().requires_grad_(True))
    _receivers(reference, name).pow(2).sum().backward()
    assert v.grad is not None


# ---------------------------------------------------------------------------
# PropagatorOutput surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPS)
def test_output_pickles_and_deepcopies(name: str) -> None:
    """Names, info and the auxiliary state all survive a round trip."""
    out = _run(name)
    # Touch final_state first: its cache lives in the instance __dict__, which
    # is part of what pickle and copy carry.
    assert list(out.final_state)
    for restored in (
        pickle.loads(pickle.dumps(out)),  # noqa: S301 - trusted, in-process data
        copy.deepcopy(out),
    ):
        assert isinstance(restored, deepwave.PropagatorOutput)
        assert restored.names == out.names
        assert restored.info == out.info
        assert restored.auxiliary_state.names == out.auxiliary_state.names
        assert list(restored.final_state) == list(out.final_state)
        for field, tensor in out.auxiliary_state.items():
            assert torch.equal(tensor, restored.auxiliary_state[field]), field


# How many absorbing-boundary names each propagator folds into its
# ``auxiliary_state``, per ndim. Pinned as counts, so a renamed field does not
# need this table edited but a changed fold set does.
_FOLDED_COUNTS = {
    ("scalar", 2): 4,
    ("scalar", 3): 6,
    ("scalar_born", 2): 8,
    ("scalar_born", 3): 12,
    ("acoustic", 2): 4,
    ("acoustic", 3): 6,
    ("elastic", 2): 8,
    ("elastic", 3): 18,
}


@pytest.mark.parametrize(("name", "ndim"), sorted(_FOLDED_COUNTS))
def test_folded_count_matches_the_fold_set(name: str, ndim: int) -> None:
    """The names folded into ``auxiliary_state`` are exactly this many."""
    assert len(_spec(name).auxiliary_state_names(ndim)) == _FOLDED_COUNTS[name, ndim]


@pytest.mark.parametrize("name", PROPS)
def test_legacy_output_attribute_raises_naming_the_bundle(name: str) -> None:
    """``out.psiy_m1`` is not a field, on any of the nine.

    ``v0.0.27`` returned a plain tuple, so an auxiliary field was never an
    *attribute* of anything it returned. The message says where the field went
    (``out.auxiliary_state``, which is the one tuple element the whole set now
    occupies) and how to continue a simulation without naming it.
    """
    out = _run(name)
    assert out.auxiliary_state.names
    for field in out.auxiliary_state.names:
        with pytest.raises(AttributeError, match="has no field") as excinfo:
            getattr(out, field)
        message = str(excinfo.value)
        assert "out.auxiliary_state" in message
        assert "initial_state=out" in message


@pytest.mark.parametrize("name", PROPS)
def test_unknown_attribute_message_mentions_auxiliary_state(name: str) -> None:
    """The AttributeError lists the public fields and the auxiliary state."""
    out = _run(name)
    with pytest.raises(AttributeError, match="has no field 'not_a_field'") as excinfo:
        _ = out.not_a_field
    assert "out.auxiliary_state additionally holds" in str(excinfo.value)


@pytest.mark.parametrize("name", PROPS)
def test_mid_tuple_index_now_raises(name: str) -> None:
    """The tuple is shorter, so the old auxiliary positions are gone."""
    out = _run(name)
    spec = _spec(name)
    assert len(out) == len(spec.output_names(2))
    # The pre-fold tuple was longer by one element per folded field, less the
    # one the auxiliary state occupies, so those positions no longer exist.
    assert len(spec.auxiliary_state_names(2)) > 1
    with pytest.raises(IndexError, match="tuple index out of range"):
        _ = out[len(out) + len(spec.auxiliary_state_names(2)) - 2]


@pytest.mark.parametrize("name", PROPS)
def test_output_names_and_as_dict_are_public(name: str) -> None:
    """``names``/``as_dict`` are the public spellings of ``_names``/``_asdict``."""
    out = _run(name)
    assert out.names == out._names
    assert list(out.as_dict()) == list(out._asdict())
    assert out.as_dict()["auxiliary_state"] is out.auxiliary_state


def test_propagator_output_requires_info() -> None:
    """Every PropagatorOutput carries the metadata of the run that made it."""
    with pytest.raises(deepwave.DeepwaveTypeError, match="requires info="):
        deepwave.PropagatorOutput(["a"], [torch.zeros(1)])


def test_propagator_output_rejects_single_iterable_helpfully() -> None:
    """The DataParallel.gather reconstruction path names the workaround."""
    out = _run("scalar")
    with pytest.raises(deepwave.DeepwaveTypeError) as excinfo:
        type(out)(iter(out))
    message = str(excinfo.value)
    assert "out.receiver_amplitudes" in message
    assert "docs/example_distributed.rst" in message


# ---------------------------------------------------------------------------
# RunInfo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPS)
def test_runinfo_sequence_fields_are_tuples_and_frozen(name: str) -> None:
    """The sequence fields are immutable tuples, and the object is frozen."""
    info = _run(name).info
    for field in ("pml_width", "origin", "extent", "model_batched", "wavefield_names"):
        assert isinstance(getattr(info, field), tuple), field
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.nt = 1  # type: ignore[misc]


@pytest.mark.parametrize("name", PROPS)
def test_runinfo_new_fields_are_populated(name: str) -> None:
    """The fields the engine resolves are all filled in."""
    out = _run(name)
    info = out.info
    assert info.max_vel == 2000.0
    assert info.ndim == 2
    assert info.dtype == DTYPE
    assert info.device == torch.device("cpu")
    spec = _spec(name)
    assert info.model_batched == (False,) * len(spec.model_names(2))
    assert info.wavefield_names == tuple(out.final_state)
    assert info.receiver_names == tuple(
        field for field in out.names if field in _engine.receiver_output_names(spec)
    )
    assert info.location_interpolation_used == "none"
    assert info.pml_freq == 15.0


def test_runinfo_records_hicks_interpolation() -> None:
    """An off-grid float location resolves to hicks, and is recorded."""
    out = _run("scalar", source_locations=torch.tensor([[[12.5, 12.5]]]))
    assert out.info.location_interpolation_used == "hicks"


def test_runinfo_every_field_is_documented() -> None:
    """Every RunInfo field appears in the class ``Attributes:`` block."""
    doc = deepwave.RunInfo.__doc__ or ""
    for field in dataclasses.fields(deepwave.RunInfo):
        assert re.search(rf"^\s+{field.name}:", doc, re.MULTILINE), field.name


def test_runinfo_dt_internal_raises_naming_internal_dt() -> None:
    """``dt_internal`` is not an attribute, and the error names the one that is."""
    info = _run("scalar").info
    with pytest.raises(AttributeError, match="internal_dt"):
        _ = info.dt_internal
    assert "dt_internal" not in {f.name for f in dataclasses.fields(deepwave.RunInfo)}


# ---------------------------------------------------------------------------
# Propagators that build their own PropagatorOutput
# ---------------------------------------------------------------------------


def _attenuating(name: str) -> deepwave.PropagatorOutput:
    """One call of an attenuating propagator on a tiny 2-D problem."""
    shape = (16, 16)
    ones = torch.ones(shape, dtype=DTYPE)
    # The scalar source channel is named for the quantity it injects, which
    # differs between the two families; the receiver channel is ``p`` on both.
    rate = deepwave.spec(name).source_channels(2)[0]
    kwargs = {
        f"source_locations_{rate}": torch.tensor([[[8, 8]]]),
        "receiver_locations_p": torch.tensor([[[4, 4]]]),
        f"source_amplitudes_{rate}": _wavelet(8),
        "max_vel": 2000.0,
        "pml_freq": 15.0,
        "pml_width": 3,
        "q_freq": 15.0,
    }
    assert name == "viscoacoustic"
    return deepwave.viscoacoustic(
        1500.0 * ones, 2200.0 * ones, 20.0 * ones, 5.0, 0.001, **kwargs
    )


@pytest.mark.parametrize(
    "name",
    [
        "viscoacoustic",
    ],
)
def test_attenuating_outputs_record_their_state_names(name: str) -> None:
    """The propagators with a mechanism-dependent field count fill ``RunInfo``.

    Their field count depends on the number of relaxation mechanisms, which a
    ``PropagatorSpec`` cannot see, so they hand ``named_output`` their own
    name lists rather than letting it read them off the spec -- and the
    ``RunInfo`` fields it fills must follow those, or ``final_state`` has no
    state names to work from.
    """
    out = _attenuating(name)
    receivers = _engine.receiver_output_names(_spec(name))
    state = [field for field in out.names if field not in receivers]
    assert out.info.ndim == 2
    assert list(out.info.wavefield_names) == state
    assert list(out.info.receiver_names) == [
        field for field in out.names if field in receivers
    ]
    assert list(out.final_state) == state


@pytest.mark.parametrize("name", PROPS)
def test_reentrant_checkpoint_loss_on_auxiliary_state_only_raises(name: str) -> None:
    """A reentrant checkpoint detaches the auxiliary state, and says so loudly.

    ``torch.utils.checkpoint(..., use_reentrant=True)`` re-attaches only the
    top-level ``Tensor`` elements of the wrapped function's return, so a loss
    taken on nothing but an auxiliary-state tensor has no graph to walk and
    ``backward`` raises. It is a loud failure, not a wrong number, and
    ``use_reentrant=False`` is the fix -- pinned here because the pre-fold
    return had these tensors at the top level, where they were re-attached.
    """
    problem = _problem(name)
    v = problem["models"][0].clone().requires_grad_(True)
    src = _wavelet(12)
    kwargs = {
        key: value
        for key, value in problem["kwargs"].items()
        if not key.startswith(
            ("receiver_locations", "bg_receiver_locations", "sc_receiver_locations")
        )
    }
    aux = _aux(name)

    def call(model: torch.Tensor) -> deepwave.PropagatorOutput:
        return getattr(deepwave, name)(
            model,
            *problem["models"][1:],
            problem["dx"],
            problem["dt"],
            **{_SOURCE_ARG[name]: src},
            **kwargs,
        )

    out = torch.utils.checkpoint.checkpoint(call, v, use_reentrant=True)
    state = out[_spec(name).output_names(2).index("auxiliary_state")]
    assert not state[aux].requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        state[aux].pow(2).sum().backward()

    v2 = problem["models"][0].clone().requires_grad_(True)
    good = torch.utils.checkpoint.checkpoint(call, v2, use_reentrant=False)
    good.auxiliary_state[aux].pow(2).sum().backward()
    assert v2.grad is not None
    assert torch.any(v2.grad != 0)


# ---------------------------------------------------------------------------
# Deliberate detachment, and the warning for accidental detachment
# ---------------------------------------------------------------------------

# The distinguishing phrase of the accidental-detachment warning.
_DETACHED = "carry no gradient history"


def _scalar_checkpoint_state(use_reentrant: bool):
    """A ``scalar`` auxiliary state returned through ``torch.utils.checkpoint``.

    With ``use_reentrant=True`` its tensors come back detached (the reentrant
    implementation re-attaches only the top-level ``Tensor`` elements of the
    return); with ``use_reentrant=False`` they keep their ``grad_fn``.
    """
    problem = _problem("scalar")
    v = problem["models"][0].clone().requires_grad_(True)
    src = _wavelet(12)

    def call(model: torch.Tensor):
        return deepwave.scalar(
            model,
            problem["dx"],
            problem["dt"],
            source_amplitudes=src,
            **problem["kwargs"],
        )

    out = torch.utils.checkpoint.checkpoint(call, v, use_reentrant=use_reentrant)
    names = _spec("scalar").output_names(2)
    state = out[names.index("auxiliary_state")]
    return v, src, problem, state


@contextlib.contextmanager
def _detached_warnings():
    """Records warnings, with the once-per-process detachment cause reset."""
    # The cause is keyed by propagator, so clear every variant of it.
    _engine._warned_once_causes -= {
        c
        for c in _engine._warned_once_causes
        if c.startswith("auxiliary_state_detached")
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _fired(caught) -> bool:
    """Whether the accidental-detachment warning is among ``caught``."""
    return any(_DETACHED in str(w.message) for w in caught)


def _restart(v, src, problem, state) -> deepwave.PropagatorOutput:
    """A ``scalar`` continuation from ``state`` alone."""
    return deepwave.scalar(
        v,
        problem["dx"],
        problem["dt"],
        source_amplitudes=src,
        initial_state={"auxiliary_state": state},
        **problem["kwargs"],
    )


def test_detached_auxiliary_state_warns_on_restart() -> None:
    """A reentrant checkpoint's auxiliary state warns when it is fed back in.

    All four conditions hold: the state has a recorded fingerprint, every one
    of its tensors is detached, grad is enabled, and the model requires grad.
    The gradient of the continuation cannot reach the run that produced the
    state, and nothing else says so.
    """
    v, src, problem, state = _scalar_checkpoint_state(use_reentrant=True)
    assert state.propagator == "scalar"
    assert all(t.grad_fn is None and not t.requires_grad for t in state.values())
    with _detached_warnings() as caught:
        _restart(v, src, problem, state)
    assert _fired(caught)
    message = next(str(w.message) for w in caught if _DETACHED in str(w.message))
    assert "use_reentrant=False" in message
    # The recipe named must be the one that actually truncates: detaching the
    # auxiliary state alone leaves the physical wavefields attached, so the
    # graph is not cut and the gradient is still the full one.
    assert "initial_state=out.detach()" in message


def test_non_reentrant_checkpoint_auxiliary_state_does_not_warn() -> None:
    """``use_reentrant=False`` keeps the graph, so there is nothing to warn about."""
    v, src, problem, state = _scalar_checkpoint_state(use_reentrant=False)
    assert any(t.grad_fn is not None for t in state.values())
    with _detached_warnings() as caught:
        _restart(v, src, problem, state)
    assert not _fired(caught)


def test_detached_auxiliary_state_does_not_warn_without_grad() -> None:
    """Under ``no_grad`` there is no gradient to lose, so no warning."""
    v, src, problem, state = _scalar_checkpoint_state(use_reentrant=True)
    with _detached_warnings() as caught, torch.no_grad():
        _restart(v, src, problem, state)
    assert not _fired(caught)


def test_detached_auxiliary_state_does_not_warn_without_grad_inputs() -> None:
    """No model or source input requiring grad means no gradient to lose."""
    v, src, problem, state = _scalar_checkpoint_state(use_reentrant=True)
    with _detached_warnings() as caught:
        _restart(v.detach(), src, problem, state)
    assert not _fired(caught)


def test_hand_built_auxiliary_state_does_not_warn() -> None:
    """A fingerprint-less state records no run, so its detachment means nothing."""
    v, src, problem, state = _scalar_checkpoint_state(use_reentrant=True)
    hand_built = _engine.AuxiliaryState(
        {n: t.detach().clone() for n, t in state.items()}
    )
    assert hand_built.propagator is None
    with _detached_warnings() as caught:
        _restart(v, src, problem, hand_built)
    assert not _fired(caught)


def test_deliberately_detached_auxiliary_state_does_not_warn() -> None:
    """``detach()`` says the cut was intended, which is the escape hatch."""
    v, src, problem, state = _scalar_checkpoint_state(use_reentrant=True)
    with _detached_warnings() as caught:
        _restart(v, src, problem, state.detach())
    assert not _fired(caught)


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_run_segments_never_warns_about_detached_state(use_reentrant: bool) -> None:
    """No segment of ``run_segments`` trips the warning, either way round.

    ``run_segments`` threads its state as flat top-level tensors precisely so
    that a reentrant checkpoint re-attaches them, and rebuilds the boundary
    state with ``_from_flat`` for each segment. The second case is the
    regression net for the flag: when the caller starts a truncated
    backpropagation loop by passing a deliberately detached state,
    ``_from_flat`` must carry ``detached_deliberately`` across every segment
    boundary or every segment after the first nags.
    """
    problem = _problem("scalar")
    src = _wavelet(24)
    v = problem["models"][0].clone().requires_grad_(True)

    def run(initial_state) -> None:
        out = run_segments(
            deepwave.scalar,
            [v],
            problem["dx"],
            problem["dt"],
            {"source_amplitudes": src},
            n_segments=3,
            checkpoint=True,
            use_reentrant=use_reentrant,
            initial_state=initial_state,
            **problem["kwargs"],
        )
        _receivers(out, "scalar").pow(2).sum().backward()

    with _detached_warnings() as caught:
        run(None)
    assert not _fired(caught)

    first = deepwave.scalar(
        v,
        problem["dx"],
        problem["dt"],
        source_amplitudes=src,
        **problem["kwargs"],
    )
    truncated = {
        n: (t.detach() if isinstance(t, torch.Tensor) else t)
        for n, t in first.final_state.items()
    }
    truncated["auxiliary_state"] = first.auxiliary_state.detach()
    with _detached_warnings() as caught:
        run(truncated)
    assert not _fired(caught)


@pytest.mark.parametrize("name", PROPS)
def test_detach_returns_a_detached_copy_that_says_so(name: str) -> None:
    """``detach()`` detaches every tensor, records the intent, keeps the rest."""
    problem = _problem(name)
    models = [m.clone().requires_grad_(True) for m in problem["models"]]
    out = getattr(deepwave, name)(
        *models,
        problem["dx"],
        problem["dt"],
        **{_SOURCE_ARG[name]: _wavelet(12)},
        **problem["kwargs"],
    )
    state = out.auxiliary_state
    assert not state.detached_deliberately
    assert any(t.requires_grad for t in state.values())
    cut = state.detach()
    assert cut is not state
    assert cut.detached_deliberately
    assert cut.names == state.names
    assert all(t.grad_fn is None and not t.requires_grad for t in cut.values())
    for field in ("propagator", "ndim", "accuracy", "pml_width", "boundary", "shape"):
        assert getattr(cut, field) == getattr(state, field)
    # The tensors are the same numbers, just without their history.
    for key in state.names:
        assert torch.equal(cut[key], state[key])


@pytest.mark.parametrize("name", PROPS)
def test_detach_flag_is_not_part_of_compatibility(name: str) -> None:
    """Deliberate detachment says nothing about what the tensors mean."""
    state = _run(name).auxiliary_state
    cut = state.detach()
    assert cut.compatible_with(state)
    assert state.compatible_with(cut)
    assert not state._mismatches(cut)


@pytest.mark.parametrize("name", PROPS)
def test_detach_flag_survives_to_and_from_flat(name: str) -> None:
    """``.to()`` and ``_from_flat`` both carry the flag; neither invents it."""
    state = _run(name).auxiliary_state
    cut = state.detach()
    assert cut.to(dtype=torch.float32).detached_deliberately
    assert not state.to(dtype=torch.float32).detached_deliberately
    assert _engine.AuxiliaryState._from_flat(cut.as_dict(), cut).detached_deliberately
    assert not _engine.AuxiliaryState._from_flat(
        state.as_dict(), state
    ).detached_deliberately
    assert not _engine.AuxiliaryState._from_flat(cut.as_dict()).detached_deliberately


@pytest.mark.parametrize("name", PROPS)
def test_repr_names_deliberate_detachment_only_when_set(name: str) -> None:
    """The repr stays quiet about the flag unless it is set."""
    state = _run(name).auxiliary_state
    assert "detached_deliberately" not in repr(state)
    assert "detached_deliberately" in repr(state.detach())


# ---------------------------------------------------------------------------
# Saving a state and continuing from it later
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPS)
def test_torch_save_load_round_trip_restarts_bit_exactly(name: str, tmp_path) -> None:
    """A state written to disk and read back continues the same simulation.

    Continuing a long simulation across processes -- or across a machine
    restart -- means writing the join out and reading it back, so the round
    trip through ``torch.save``/``torch.load`` has to preserve everything a
    continuation reads: the tensors, and the recorded run parameters the
    compatibility check compares. The reloaded state must therefore restart
    bit-exactly, and identically to the in-memory object it came from.
    """
    src = _wavelet(24)
    prop_kwargs = {_SOURCE_ARG[name]: src[..., :10]}
    first = _run(name, **prop_kwargs)

    path = tmp_path / f"{name}_state.pt"
    torch.save(first, path)
    reloaded = torch.load(path, weights_only=False)

    assert isinstance(reloaded.auxiliary_state, deepwave.AuxiliaryState)
    assert reloaded.auxiliary_state.names == first.auxiliary_state.names
    assert (
        reloaded.auxiliary_state.format_version == first.auxiliary_state.format_version
    )
    for field, tensor in first.auxiliary_state.items():
        assert torch.equal(tensor, reloaded.auxiliary_state[field]), field

    rest = {_SOURCE_ARG[name]: src[..., 10:]}
    from_memory = _run(name, initial_state=first, **rest)
    from_disk = _run(name, initial_state=reloaded, **rest)
    assert torch.equal(_receivers(from_memory, name), _receivers(from_disk, name))
    for field, tensor in _physical_state(from_memory).items():
        assert torch.equal(tensor, getattr(from_disk, field)), field
    for field, tensor in from_memory.auxiliary_state.items():
        assert torch.equal(tensor, from_disk.auxiliary_state[field]), field


@pytest.mark.parametrize("name", PROPS)
def test_auxiliary_state_records_the_current_format_version(name: str) -> None:
    """Every run stamps the state with the layout version it wrote."""
    state = _run(name).auxiliary_state
    assert isinstance(state.format_version, int)
    assert state.format_version == _engine.AUXILIARY_STATE_FORMAT_VERSION


@pytest.mark.parametrize("name", PROPS)
def test_a_different_format_version_raises_naming_both(name: str) -> None:
    """A state from another layout is refused, not read as this one.

    The message names the version the state carries and the version this
    release reads, because those two numbers are what tells the user whether
    to re-run the first call or to pin an older Deepwave.
    """
    first = _run(name)
    stale = dataclasses.replace(
        first.auxiliary_state,
        format_version=_engine.AUXILIARY_STATE_FORMAT_VERSION + 1,
    )
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        _run(name, initial_state={**_physical_state(first), "auxiliary_state": stale})
    message = str(excinfo.value)
    assert "format_version" in message
    assert str(_engine.AUXILIARY_STATE_FORMAT_VERSION + 1) in message
    assert str(_engine.AUXILIARY_STATE_FORMAT_VERSION) in message


@pytest.mark.parametrize(
    "name",
    [
        "viscoacoustic",
    ],
)
@pytest.mark.parametrize("n_mechanisms", [1, 2])
def test_attenuation_memory_is_carried_in_the_auxiliary_state(
    name: str, n_mechanisms: int
) -> None:
    """The GSLS memory variables are auxiliary state, not public outputs.

    They are implementation state of the attenuation model -- their number
    is ``q_n_mechanisms``, which is a knob, not a physical quantity the
    caller asked for -- so they are folded away with the absorbing
    boundary's memory rather than occupying output slots whose count varies.
    """
    extra: dict = {"q_n_mechanisms": n_mechanisms}
    if n_mechanisms > 1:
        # Above one mechanism the constant-Q fit spans q_band, and q_freq --
        # which the tiny problem pins -- is refused.
        extra["q_band"] = (5.0, 60.0)
        extra["q_freq"] = None
    out = _run(name, **extra)
    memory = [field for field in out.auxiliary_state.names if _is_gsls(field)]
    assert len(memory) == n_mechanisms * (1 if name == "viscoacoustic" else 4)
    assert not [field for field in out._names if _is_gsls(field)]
    # The fold moved where these tensors are reported, not what they are:
    # continuing through the folded state still reproduces one combined run.
    # That holds bit for bit at every mechanism count. It did not always: the
    # constant-Q fit behind more than one mechanism used to be solved by a
    # least-squares driver whose rounding depended on the process's
    # allocation history, so two identical calls could differ in their last
    # bits. ``tests/test_determinism.py`` pins that down directly.
    src = _wavelet(24)
    full = _run(name, **{_SOURCE_ARG[name]: src}, **extra)
    seg1 = _run(name, **{_SOURCE_ARG[name]: src[..., :10]}, **extra)
    seg2 = _run(name, **{_SOURCE_ARG[name]: src[..., 10:]}, initial_state=seg1, **extra)
    chained = torch.cat([_receivers(seg1, name), _receivers(seg2, name)], dim=-1)
    combined = _receivers(full, name)
    assert torch.equal(combined, chained)


def _is_gsls(field: str) -> bool:
    """Whether ``field`` names a GSLS attenuation memory variable."""
    return field.startswith(("r_mech", "rp_mech", "rs_"))
