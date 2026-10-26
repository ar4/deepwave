"""One rule generates every flat keyword, with a closed exception table.

The flat keywords are the widest part of Deepwave's surface -- nine
propagators times a channel set times three argument groups -- and they are
frozen at 0.1.0. What keeps that surface learnable is that there is nothing
to learn beyond one rule: with channel ``<ch>`` and state field ``<s>``, a
propagator takes

* ``source_amplitudes_<ch>``, ``source_locations_<ch>``, ``source_mask_<ch>``
* ``receiver_locations_<ch>``, ``receiver_mask_<ch>``
* ``<s>_0``, and ``<s>_m1`` where the scheme has a previous step

with the ``_<ch>`` suffix dropped for the scalar channel ``u``, and with the
receiver keywords spelled ``bg_receiver_*`` and ``sc_receiver_*`` on a Born
propagator, whose scattered state adds ``<s>_sc_0``.

Every departure from that rule is listed in :data:`_EXCEPTIONS` below, and
the table is **closed**: a flat keyword that is neither rule-derived nor in
the table fails the test, whether it was added by accident or on purpose.
The table has exactly two entries, both of them deprecations that predate
the rule rather than exemptions from it.

The rule is checked in both directions. Missing names matter as much as
extra ones: a channel the spec declares but the signature never grew is a
keyword users are told exists and cannot pass.
"""

import inspect
from collections.abc import Callable
from typing import Any

import pytest

import deepwave

PROPAGATORS = deepwave._engine.PROPAGATOR_NAMES

#: Each propagator's ``Module`` class, so that the rule can be checked on the
#: two entry points a survey can be passed through.
_MODULES: dict[str, Any] = {
    "scalar": deepwave.Scalar,
    "scalar_born": deepwave.ScalarBorn,
    "acoustic": deepwave.Acoustic,
    "acoustic_born": deepwave.AcousticBorn,
    "acoustic_vti": deepwave.AcousticVti,
    "viscoacoustic": deepwave.Viscoacoustic,
    "elastic": deepwave.Elastic,
    "elastic_born": deepwave.ElasticBorn,
    "elastic_vti": deepwave.ElasticVti,
}

#: The stems of every flat channel keyword, longest first so that a
#: ``bg_``/``sc_`` prefixed name is not mistaken for the bare one.
_CHANNEL_STEMS = (
    "bg_receiver_locations",
    "bg_receiver_mask",
    "sc_receiver_locations",
    "sc_receiver_mask",
    "source_amplitudes",
    "source_locations",
    "source_mask",
    "receiver_locations",
    "receiver_mask",
)

#: The closed table of flat keywords the rule does not derive. Both entries
#: are deprecated spellings that keep working until they are removed; nothing
#: here is an exemption a future propagator may copy.
_EXCEPTIONS: dict[str, frozenset[str]] = {
    "acoustic": frozenset({"source_amplitudes_p", "source_locations_p"}),
    "elastic": frozenset({"source_amplitudes_p", "source_locations_p"}),
    "scalar_born": frozenset({"receiver_locations"}),
}


def _ndims(name: str) -> list[int]:
    """The dimensionalities the rule is evaluated at for ``name``.

    Always include 3: every propagator's signature carries the same
    keyword set, and the spec's ``all_channels`` takes the union too.
    """
    return sorted(set(deepwave.spec(name).ndims) | {3})


def _kinds(name: str) -> list[str]:
    """The channel-argument groups ``name`` has."""
    if deepwave.spec(name).has_bg_receivers:
        return ["source", "bg_receiver", "sc_receiver"]
    return ["source", "receiver"]


def _rule_derived(name: str) -> set[str]:
    """Every flat keyword the naming rule derives for ``name``."""
    private = deepwave._engine.propagator_spec(name)
    names: set[str] = set()
    for ndim in _ndims(name):
        for kind in _kinds(name):
            channels = (
                private.receiver_channels(ndim)
                if kind == "sc_receiver"
                else getattr(private, f"{kind}_channels")(ndim)
            )
            for channel in channels:
                names.update(private.channel_kwargs(kind, channel))
        auxiliary = set(private.auxiliary_state_names(ndim))
        names.update(
            field
            for field in private.public_state_names(ndim)
            if field not in auxiliary
        )
    return names


def _flat_keywords(signature: Callable[..., Any]) -> set[str]:
    """The flat channel and state keywords a callable accepts.

    A flat keyword is either a channel keyword (one of the stems, optionally
    suffixed with a channel name) or an initial-state field, which is what
    the ``_0``/``_m1`` suffix marks. Everything else in the signature --
    models, the grid, the options, ``sources=``/``receivers=``,
    ``initial_state=`` -- is out of the rule's scope.
    """
    return {
        parameter
        for parameter in inspect.signature(signature).parameters
        if parameter.startswith(_CHANNEL_STEMS) or parameter.endswith(("_0", "_m1"))
    }


@pytest.mark.parametrize("name", PROPAGATORS)
def test_every_flat_keyword_is_rule_derived_or_tabled(name: str) -> None:
    """The closed table: no third source of flat keywords exists."""
    accepted = _flat_keywords(getattr(deepwave, name))
    unexplained = accepted - _rule_derived(name) - _EXCEPTIONS.get(name, frozenset())
    assert not unexplained, (
        f"{name} accepts flat keyword(s) {sorted(unexplained)} that the naming "
        "rule does not derive and that the closed exception table does not "
        "list. Either spell them by the rule or, if they really are a new "
        "exception, that is an interface decision, not a table edit."
    )


@pytest.mark.parametrize("name", PROPAGATORS)
def test_every_rule_derived_flat_keyword_exists(name: str) -> None:
    """The other direction: a declared channel is a keyword you can pass."""
    accepted = _flat_keywords(getattr(deepwave, name))
    missing = _rule_derived(name) - accepted
    assert not missing, (
        f"{name} declares channels or state fields whose flat keyword(s) "
        f"{sorted(missing)} its signature never grew."
    )


@pytest.mark.parametrize("name", PROPAGATORS)
def test_the_module_takes_the_same_flat_keywords(name: str) -> None:
    """``Module.forward`` and the function are one interface, not two.

    The ``Module`` holds the models, so it takes no model arguments; the
    survey and the initial state are per-call, so every flat keyword is the
    same on both.
    """
    assert _flat_keywords(_MODULES[name].forward) == _flat_keywords(
        getattr(deepwave, name)
    )


@pytest.mark.parametrize("name", sorted(_EXCEPTIONS))
def test_every_tabled_exception_is_really_accepted(name: str) -> None:
    """The table lists nothing that has already been removed.

    A stale entry would quietly widen the table: it would excuse a name that
    no longer exists, and would go on excusing a name reintroduced later.
    """
    accepted = _flat_keywords(getattr(deepwave, name))
    assert _EXCEPTIONS[name] <= accepted


@pytest.mark.parametrize("name", PROPAGATORS)
def test_the_table_agrees_with_the_spec(name: str) -> None:
    """The table's source-side entries are exactly the spec's deprecations."""
    deprecated = deepwave.spec(name).deprecated_source_channels()
    expected = {
        f"{stem}_{old}"
        for old in deprecated
        for stem in ("source_amplitudes", "source_locations")
    }
    tabled = {
        entry
        for entry in _EXCEPTIONS.get(name, frozenset())
        if entry.startswith("source_")
    }
    assert tabled == expected


def test_the_scalar_channel_carries_no_suffix() -> None:
    """The one shape rule that is not a name: ``u`` is unsuffixed."""
    for name in PROPAGATORS:
        private = deepwave._engine.propagator_spec(name)
        scalar_channel = private.source_channels(2) == ["u"]
        assert private.channel_suffixes is not scalar_channel
        if scalar_channel:
            assert "source_amplitudes" in _flat_keywords(getattr(deepwave, name))
            assert "source_amplitudes_u" not in _flat_keywords(getattr(deepwave, name))
