"""Drift guard for ``docs/callback_names.rst``'s generated table.

``deepwave._capabilities.internal_names`` reads every name in that table from
each propagator's declared ``PropagatorSpec`` -- its ``model_names``,
``wavefield_names`` and ``auxiliary_state_names`` callables. That is the same
source the engine and the eager backends use when they build a
``CallbackState``, but "same source" is a claim about the code, not a
measurement, and it is exactly the claim the page makes to its readers. So
the tests below re-derive the table the only way that cannot share a bug with
it: they **run real callbacks** on independently-built minimal problems and
ask each one which names it was actually handed.

Asking is possible without touching a private attribute, because
``CallbackState``'s accessors document what an unknown name does -- they raise
a ``KeyError`` naming the keys that *are* present (see
``CallbackState._lookup``'s docstring, and the "A name the accessors do not
recognise" paragraph of ``docs/callback_names.rst``). Parsing that list gives
the exact set, so these tests check set *equality*: a name in the table that
a callback does not accept fails, and so does a name a callback accepts that
the table does not list.

The fixtures come from ``test_adjoint_dot_product.py`` (the vetted
grid/model/survey set the rest of the suite shares) and the per-propagator
argument shapes are spelled out again here rather than imported from
``deepwave._capabilities``, so that a bug in that module's own call-building
helpers cannot hide from these tests.
"""

import ast
import pathlib
import re
import warnings

import pytest
import torch
from test_adjoint_dot_product import _grid, _media, _survey

import deepwave
from deepwave._capabilities import (
    PROPAGATOR_NAMES,
    callback_names_rst,
    internal_names,
)

DTYPE = torch.float64

_ROWS = {(row.name, row.ndim): row for row in internal_names()}


def _positional(prop: str, m: dict) -> tuple:
    """The model tensors ``prop`` takes, in positional order (own fixture)."""
    if prop in ("scalar", "scalar_born"):
        return (m["v"],)
    if prop in ("acoustic", "acoustic_born"):
        return (m["v"], m["rho"])
    if prop == "viscoacoustic":
        return (m["v"], m["rho"], m["q"])
    if prop in ("elastic", "elastic_born"):
        return (m["lamb"], m["mu"], m["buoyancy"])
    if prop == "acoustic_vti":
        return (m["c11"], m["c13"], m["c33"], m["buoyancy"])
    if prop in ("elastic_vti",):
        return (m["c11"], m["c13"], m["c33"], m["c55"], m["buoyancy"])
    raise AssertionError(prop)


def _kwargs(prop: str, ndim: int) -> dict:
    """A minimal valid keyword set for ``prop`` at ``ndim`` dimensions.

    Generic in ``ndim``, because the names under test are ndim-dependent and
    the table covers every dimensionality each propagator declares.
    """
    g = _grid(ndim)
    n, nt, dt = g["n"], g["nt"], g["dt"]
    src, src_loc, rec_loc = _survey(ndim, n, nt, dt)
    spec = deepwave.spec(prop)
    # A source is named for the quantity it injects and a receiver for the
    # field it records, so the two sides do not share a channel name.
    # ``acoustic_vti``'s scalar channels are both synthesised by the wrapper,
    # so its x velocity is used on both sides instead.
    if prop == "acoustic_vti":
        source_channel = receiver_channel = "x"
    else:
        source_channel = spec.source_channels(ndim)[0]
        receiver_channel = spec.receiver_channels(ndim)[0]
    kwargs: dict = {
        "grid_spacing": 5.0,
        "dt": dt,
        "accuracy": g["acc"],
        "pml_width": 3,
        # Explicit rather than data-dependent, so the models may carry
        # gradients (see deepwave._common.velocity_extremes).
        "max_vel": 2500.0,
        spec.source_amplitudes_kwarg(source_channel): src,
        spec.source_locations_kwarg(source_channel): src_loc,
        spec.receiver_locations_kwarg(receiver_channel): rec_loc,
    }
    zeros = torch.zeros((n,) * ndim, dtype=DTYPE)
    if prop == "scalar_born":
        kwargs["dv"] = zeros
        kwargs["bg_receiver_locations"] = rec_loc
    elif prop == "acoustic_born":
        kwargs["dv"] = zeros
        kwargs["drho"] = zeros
        kwargs["bg_receiver_locations_p"] = rec_loc
    elif prop == "elastic_born":
        kwargs["dlamb"] = zeros
        kwargs["dmu"] = zeros
        kwargs["dbuoyancy"] = zeros
        kwargs["bg_receiver_locations_p"] = rec_loc
    elif prop == "elastic_vti" and ndim == 3:
        # c66 is required in 3-D only, and must stay None in 2-D.
        kwargs["c66"] = torch.full((n,) * ndim, 2.0e9, dtype=DTYPE)
    if prop in ("viscoacoustic",):
        kwargs["q_freq"] = 25.0
    return kwargs


#: The list of present keys a ``CallbackState`` accessor names in the
#: ``KeyError`` it raises for an unknown name.
_PRESENT = re.compile(r"present: (\[[^\]]*\])")

#: A name no propagator can plausibly hold, used to provoke that ``KeyError``.
_UNKNOWN = "__deepwave_callback_names_test__"


def _present(state, kind: str) -> list:
    """The names ``state`` actually holds for ``kind``, read from its KeyError.

    The documented behaviour of every accessor for an unrecognised name (see
    ``CallbackState._lookup``): raise a ``KeyError`` listing the keys that are
    present. Using it here keeps this test out of ``CallbackState``'s private
    dictionaries, so it measures the public contract the docs page describes
    rather than an implementation detail.
    """
    getter = {
        "model": state.get_model,
        "wavefield": state.get_wavefield,
        "gradient": state.get_gradient,
    }[kind]
    try:
        getter(_UNKNOWN)
    except KeyError as exc:
        message = str(exc)
    else:
        raise AssertionError(f"{kind} {_UNKNOWN!r} unexpectedly resolved")
    match = _PRESENT.search(message)
    assert match is not None, f"unparsable {kind} KeyError: {message}"
    return sorted(ast.literal_eval(match.group(1)))


#: Every ``(propagator, ndim)`` pair the table claims a row for.
_CASES = sorted(_ROWS)


def test_every_propagator_and_dimensionality_has_a_row() -> None:
    """The table covers all nine propagators, at every ndim each declares.

    ``spec.ndims`` is re-read here rather than taken from the rows, so a
    propagator that gained or lost a dimensionality cannot quietly keep its
    old set of rows.
    """
    assert {name for name, _ in _ROWS} == set(PROPAGATOR_NAMES)
    assert len(PROPAGATOR_NAMES) == 9
    for name in PROPAGATOR_NAMES:
        ndims = tuple(sorted(ndim for prop, ndim in _ROWS if prop == name))
        assert ndims == tuple(sorted(deepwave.spec(name).ndims)), name


def test_auxiliary_is_a_subset_of_the_wavefields() -> None:
    """The auxiliary column is a subset of the wavefield column, in its order.

    The page presents it that way ("Of which ..."), and the subset relation is
    what makes the auxiliary names reachable through ``get_wavefield`` at all.
    The order check catches a regression in the public-to-internal spelling
    translation ``internal_names`` performs for the attenuating propagators'
    constant-Q memory (public ``r_mech0``, internal ``r_0``).
    """
    for row in _ROWS.values():
        assert set(row.auxiliary) <= set(row.wavefields), (row.name, row.ndim)
        order = [n for n in row.wavefields if n in set(row.auxiliary)]
        assert list(row.auxiliary) == order, (row.name, row.ndim)


def test_callback_names_rst_lists_every_name() -> None:
    """The rendered RST is one table naming every propagator and every name."""
    rst = callback_names_rst(list(_ROWS.values()))
    assert rst.count(".. list-table::") == 1
    for name in PROPAGATOR_NAMES:
        assert f"``{name}``" in rst
    for row in _ROWS.values():
        for entry in (*row.models, *row.wavefields, *row.auxiliary):
            assert f"``{entry}``" in rst, (row.name, row.ndim, entry)
    # No cell may be empty: a bare "- " in a list-table row silently shifts
    # every later column of that row.
    for line in rst.splitlines():
        assert line.rstrip() not in ("   * -", "     -"), line


@pytest.mark.parametrize(("name", "ndim"), _CASES)
def test_forward_callback_is_given_exactly_the_documented_names(
    name: str, ndim: int
) -> None:
    """A real ``forward_callback`` holds exactly the table's names.

    The central check: set equality, on both the model and the wavefield
    column, for all nine propagators at every dimensionality they support. It
    is what lets the page promise that these names are the ones a callback
    accepts, rather than merely the ones a spec declares.
    """
    row = _ROWS[(name, ndim)]
    seen: dict = {}

    def callback(state) -> None:
        seen.setdefault("models", _present(state, "model"))
        seen.setdefault("wavefields", _present(state, "wavefield"))

    media = _media(ndim, _grid(ndim)["n"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(deepwave, name)(
            *_positional(name, media),
            **_kwargs(name, ndim),
            forward_callback=callback,
        )
    assert seen.get("models") == sorted(row.models), name
    assert seen.get("wavefields") == sorted(row.wavefields), name


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_gradient_names_are_the_model_names(name: str) -> None:
    """The gradient accumulators a ``backward_callback`` sees are the models.

    The page's other load-bearing claim: the model column *is* the gradient
    column, so ``get_gradient("K")`` reaches the accumulator behind
    ``acoustic``'s internal ``K``. Checked by differentiating a real call with
    every model requiring a gradient, and comparing the set the callback holds
    against the row's model names.
    """
    ndim = 2
    row = _ROWS[(name, ndim)]
    seen: dict = {}

    def callback(state) -> None:
        seen.setdefault("gradients", _present(state, "gradient"))

    media = _media(ndim, _grid(ndim)["n"])
    models = tuple(t.clone().requires_grad_(True) for t in _positional(name, media))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = getattr(deepwave, name)(
            *models, **_kwargs(name, ndim), backward_callback=callback
        )
        out[-1].sum().backward()
    assert seen.get("gradients") == sorted(row.models), name


#: Internal names ``v0.0.27`` accepted from a callback, which stay accepted
#: (see ``docs/callback_names.rst`` and ``docs/stability.rst``). Written out
#: here rather than derived: freezing them is the whole promise, so a rename
#: has to be a deliberate edit of this list and not a silently regenerated
#: table. One entry per released propagator, in 2-D.
_V0027_NAMES: dict = {
    ("scalar", 2): (
        "v",
        "wavefield_0",
        "wavefield_m1",
        "psiy_m1",
        "psix_m1",
        "zetay_m1",
        "zetax_m1",
    ),
    ("scalar_born", 2): (
        "v",
        "dv",
        "wavefield_sc_0",
        "wavefield_sc_m1",
        "psiy_sc_m1",
        "zetax_sc_m1",
    ),
    ("acoustic", 2): (
        "K",
        "By",
        "Bx",
        "pressure_0",
        "vy_0",
        "vx_0",
        "phi_y_0",
        "phi_x_0",
        "psi_y_0",
        "psi_x_0",
    ),
    ("elastic", 2): (
        "lamb",
        "mu",
        "mu_yx",
        "buoyancy_y",
        "buoyancy_x",
        "vy_0",
        "sigmayy_0",
        "sigmaxy_0",
        "m_vyy_0",
        "m_sigmaxyy_0",
    ),
}


@pytest.mark.parametrize(("case", "expected"), sorted(_V0027_NAMES.items()))
def test_v0027_internal_names_are_still_in_the_table(
    case: tuple, expected: tuple
) -> None:
    """Every ``v0.0.27`` callback name still appears, on the same propagator.

    These four propagators shipped in a release, so their internal callback
    names are frozen: a caller's ``state.get_wavefield("psiy_m1")`` from 2020
    must still work. The deprecation of some of the same spellings on the
    *public* output/``initial_state`` side (see ``docs/upgrading.rst``) does
    not touch the callback side, and this test is what keeps the two apart.
    """
    row = _ROWS[case]
    available = set(row.models) | set(row.wavefields)
    missing = [name for name in expected if name not in available]
    assert not missing, (case, missing)


def test_attenuating_rows_name_their_mechanism_count() -> None:
    """Only the two attenuating propagators' rows depend on ``q_n_mechanisms``.

    And the value the table is generated for is the argument's own default,
    which the page states. Both halves matter: a row that quietly stopped
    naming a mechanism count would publish names true only for one setting.
    """
    attenuating = {
        "viscoacoustic",
    }
    for row in _ROWS.values():
        if row.name in attenuating:
            assert row.n_mechanisms == 1, row.name
        else:
            assert row.n_mechanisms is None, row.name
    rendered = callback_names_rst(list(_ROWS.values()))
    assert "``q_n_mechanisms=1``" in rendered


@pytest.mark.parametrize(
    "name",
    [
        "viscoacoustic",
    ],
)
def test_a_second_mechanism_adds_the_documented_memory_names(name: str) -> None:
    """``q_n_mechanisms=2`` adds a ``_1`` copy of each constant-Q memory field.

    The page says the memory names run from ``_0`` to ``_{N-1}`` and that
    every other name is unaffected. Checked against a real callback rather
    than against the spec, since that sentence is the only part of the page
    describing names the generated table does not itself show.
    """
    ndim = 2
    row = _ROWS[(name, ndim)]
    seen: dict = {}

    def callback(state) -> None:
        seen.setdefault("wavefields", _present(state, "wavefield"))
        seen.setdefault("models", _present(state, "model"))

    media = _media(ndim, _grid(ndim)["n"])
    kwargs = _kwargs(name, ndim)
    kwargs["q_n_mechanisms"] = 2
    # A multi-mechanism GSLS is fitted over a band rather than at one
    # frequency, so q_band is required (and q_freq refused) once N > 1.
    kwargs["q_freq"] = None
    kwargs["q_band"] = (5.0, 50.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(deepwave, name)(
            *_positional(name, media), **kwargs, forward_callback=callback
        )
    # Nothing the one-mechanism row lists disappears, ...
    assert set(row.wavefields) <= set(seen["wavefields"]), name
    # ... and every added name is the ``_1`` twin of a ``_0`` field that is
    # already there, which is what "the memory names run from _0 to _{N-1}"
    # means. (For N=1 the ``_0`` suffix is a time level on the other fields
    # too, so the twin relation, not the suffix alone, is the check.)
    added = set(seen["wavefields"]) - set(row.wavefields)
    assert added, name
    for entry in sorted(added):
        assert entry.endswith("_1"), (name, entry)
        assert entry[: -len("_1")] + "_0" in set(row.auxiliary), (name, entry)
    # The models do not depend on the mechanism count.
    assert seen["models"] == sorted(row.models), name


def test_the_docs_page_includes_the_generated_table_and_is_in_the_toctree() -> None:
    """The page exists, includes the generated fragment, and is reachable.

    A generated table nothing includes, or a page no toctree lists, is a
    silent no-op: the build succeeds and the reader never sees it.
    """
    docs = pathlib.Path(__file__).resolve().parents[1] / "docs"
    page = docs / "callback_names.rst"
    if not page.is_file():  # pragma: no cover - source checkout only
        pytest.skip("docs/callback_names.rst is not present in this checkout")
    text = page.read_text()
    assert ".. include:: _generated/callback_names_table.rst" in text
    assert "callback_names" in (docs / "index.rst").read_text()
    conf = (docs / "conf.py").read_text()
    assert "callback_names_table.rst" in conf
    assert "_write_callback_names_table" in conf
