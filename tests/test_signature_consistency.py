"""Signature-consistency drift guard for the "ONE signature convention".

This is the permanent guard ensuring every propagator functional and its
``nn.Module`` wrapper's ``forward`` follow exactly the same keyword
convention:

(a) Everything after ``dt`` is keyword-only.
(b) Every keyword-argument name shared by two or more propagators has the
    same default everywhere it appears, and appears in the same relative
    order (the canonical order encoded below).
(c) Each Module's ``forward`` mirrors its functional's keyword tail exactly
    (name-for-name, default-for-default), except for the parameters the
    Module's constructor already holds (the required/optional model tensors
    and the storage_mode/storage_path/storage_compression trio).
(d) Channel groups (source_amplitudes_*, source_locations_*,
    bg_receiver_locations_*, receiver_locations_*) follow the channel order
    ``p, z, y, x`` (``acoustic_vti`` adds ``sigma_v, sigma_h`` after ``x``),
    which is the order ``deepwave.spec(...).source_channels(ndim)`` and
    ``.receiver_channels(ndim)`` declare and the returned tuple follows.
(e) Every name documented in a propagator functional's docstring ``Args:``
    section is a real parameter of its signature, and every attribute
    documented in its ``Returns:`` section exists on the real
    ``PropagatorOutput`` returned by an actual tiny propagation.
(f) The deprecated ``python_backend`` alias appears only on the four
    propagators shipped in ``v0.0.27`` and defaults to ``None``.

If a future change reorders a keyword, changes a shared default
inconsistently, lets a Module's forward drift from its functional, or lets a
docstring drift from the real signature/return value, this test fails
loudly.
"""

import inspect
import re
import warnings
from collections.abc import Callable
from typing import Any

import torch

import deepwave

_FUNCTIONALS: dict[str, Any] = {
    "scalar": deepwave.scalar,
    "scalar_born": deepwave.scalar_born,
    "acoustic": deepwave.acoustic,
    "viscoacoustic": deepwave.viscoacoustic,
    "acoustic_born": deepwave.acoustic_born,
    "elastic": deepwave.elastic,
    "elastic_born": deepwave.elastic_born,
    "elastic_vti": deepwave.elastic_vti,
    "acoustic_vti": deepwave.acoustic_vti,
}

_MODULES: dict[str, Any] = {
    "scalar": deepwave.Scalar,
    "scalar_born": deepwave.ScalarBorn,
    "acoustic": deepwave.Acoustic,
    "viscoacoustic": deepwave.Viscoacoustic,
    "acoustic_born": deepwave.AcousticBorn,
    "elastic": deepwave.Elastic,
    "elastic_born": deepwave.ElasticBorn,
    "elastic_vti": deepwave.ElasticVti,
    "acoustic_vti": deepwave.AcousticVti,
}


def _all_user_model_names(propagator: str) -> set[str]:
    """Every public model name ``propagator`` can take, over all its ndims.

    A ``Module`` constructor is not per-``ndim``, so it must offer a flag for
    a model that exists at any of the dimensionalities the propagator
    supports (``elastic_vti``'s ``c66`` is a model only in 3-D).
    """
    spec = deepwave._engine.propagator_spec(propagator)
    names: set[str] = set()
    for ndim in spec.ndims:
        names.update(deepwave._common._propagator_user_model_names(propagator, ndim))
    return names


# Parameters that a Module's __init__ already holds (as constructor
# arguments / stored attributes), and so are legitimately absent from that
# Module's forward() even though they are in the functional's keyword tail.
#
# The storage arguments used to be here: the trio was constructor-only, and
# this set was what pinned that. They are now accepted on ``forward`` as well
# (``None`` meaning "use the constructor's value"), so that one Module can
# serve a no-grad forward pass and a gradient pass without holding two copies
# of the model. The set is therefore empty of them, and
# ``test_module_forward_mirrors_functional_tail`` now *requires* all five
# storage names to be present on every ``forward``.
_STORAGE_ARGS: set[str] = set()
_CONSTRUCTOR_HELD: dict[str, set[str]] = {
    "scalar": _STORAGE_ARGS,
    "scalar_born": _STORAGE_ARGS | {"dv", "scatter"},
    "acoustic": _STORAGE_ARGS,
    "viscoacoustic": _STORAGE_ARGS,
    "acoustic_born": _STORAGE_ARGS | {"dv", "drho"},
    "elastic": _STORAGE_ARGS,
    "elastic_born": _STORAGE_ARGS | {"dlamb", "dmu", "dbuoyancy"},
    "elastic_vti": _STORAGE_ARGS | {"c66"},
    "acoustic_vti": _STORAGE_ARGS,
}

# ---------------------------------------------------------------------------
# Canonical order (see docs/dev/api_conventions.md): a rank function.
#
# Each keyword-only parameter name maps to a (group_rank, channel_rank) pair;
# a propagator's own parameter sequence must be non-decreasing under this
# key. Groups are listed in the exact order that doc specifies; channel
# sub-order within a channel group is p, z, y, x, sigma_v, sigma_h: the
# shared p/z/y/x order, with acoustic_vti's two stress channels after the
# velocity channels, which is the order its spec declares and its returned
# tuple follows.
# ---------------------------------------------------------------------------

# The long documented quantity names of the ``p`` channel rank exactly where
# ``p`` does: they are the canonical names of that one argument, of which the
# deprecated ``p`` spelling is an alias, not a second channel, so they sit
# beside it rather than opening a new group.
_CHANNEL_ORDER = [
    "",
    "p",
    "volume_rate",
    "pressure_rate",
    "z",
    "y",
    "x",
    "sigma_v",
    "sigma_h",
]


def _channel_index(suffix: str) -> int:
    """The channel's rank, with the long quantity names folded onto ``p``."""
    if suffix in ("volume_rate", "pressure_rate"):
        suffix = "p"
    return _CHANNEL_ORDER.index(suffix)


_OPTIONAL_MODEL_NAMES = {
    "dv",
    "drho",
    "dlamb",
    "dmu",
    "dbuoyancy",
    "c66",
    "tilt",
    "azimuth",
}

# (name -> rank) for the config tail, in exactly the section-0.6 order.
_CONFIG_TAIL = [
    "initial_state",
    "accuracy",
    "pml_width",
    "boundary",
    "pml_freq",
    "mpml_fraction",
    "surface_method",
    "q_freq",
    "q_n_mechanisms",
    "q_band",
    "max_vel",
    "courant_number",
    "survey_pad",
    "origin",
    "nt",
    "model_gradient_sampling_interval",
    "checkpoint_segments",
    "freq_taper_frac",
    "time_pad_frac",
    "time_taper",
    "check_output_finite",
    "forward_callback",
    "backward_callback",
    "callback_frequency",
    "backward_callback_needs_forward_wavefield",
    "backend",
    "python_backend",
    "storage",
    "storage_mode",
    "storage_path",
    "storage_compression",
    "location_interpolation",
    "location_interpolation_halfwidth",
]
# Offset the config-tail ranks past the channel-group / physical-field ranks
# below (0-6), so the whole thing sorts as one sequence.
_CONFIG_TAIL_RANK = {name: 10 + i for i, name in enumerate(_CONFIG_TAIL)}

# Group ranks (before the config tail). The channel-keyed mapping parameters
# come first: they are the canonical way to specify a survey, so they read
# before the flat keywords that remain as permanent sugar. Each locations
# group is immediately followed by its own mask group.
_RANK_OPTIONAL_MODEL = 0
_RANK_CHANNEL_MAPPING = 1
_RANK_SOURCE_AMPLITUDES = 2
_RANK_SOURCE_LOCATIONS = 3
_RANK_SOURCE_MASK = 4
_RANK_BG_RECEIVER_LOCATIONS = 5
_RANK_BG_RECEIVER_MASK = 6
_RANK_RECEIVER_LOCATIONS = 7
_RANK_RECEIVER_MASK = 8
_RANK_PHYSICAL_INITIAL_FIELD = 9


#: The channel-keyed mapping parameters, which have no channel of their own.
_MAPPING_NAMES = {"sources", "receivers", "bg_receivers"}

#: ``(parameter-name prefix, group rank)``, longest prefix first so that
#: ``bg_receiver_locations`` and ``sc_receiver_locations`` are not mistaken
#: for ``receiver_locations``.
_GROUP_PREFIXES = [
    ("bg_receiver_locations", _RANK_BG_RECEIVER_LOCATIONS),
    ("bg_receiver_mask", _RANK_BG_RECEIVER_MASK),
    ("sc_receiver_locations", _RANK_RECEIVER_LOCATIONS),
    ("sc_receiver_mask", _RANK_RECEIVER_MASK),
    ("source_amplitudes", _RANK_SOURCE_AMPLITUDES),
    ("source_locations", _RANK_SOURCE_LOCATIONS),
    ("source_mask", _RANK_SOURCE_MASK),
    ("receiver_locations", _RANK_RECEIVER_LOCATIONS),
    ("receiver_mask", _RANK_RECEIVER_MASK),
]


def _channel_rank(name: str, prefix: str) -> int:
    suffix = name[len(prefix) :]
    if suffix.startswith("_"):
        suffix = suffix[1:]
    return _channel_index(suffix)


def _rank(name: str) -> tuple[int, int]:
    """Maps a keyword-only parameter name to its canonical sort key."""
    if name in _OPTIONAL_MODEL_NAMES or name == "scatter":
        return (_RANK_OPTIONAL_MODEL, 0)
    if name in _MAPPING_NAMES:
        return (_RANK_CHANNEL_MAPPING, 0)
    for prefix, group in _GROUP_PREFIXES:
        if name == prefix or name.startswith(prefix + "_"):
            return (group, _channel_rank(name, prefix))
    if name in _CONFIG_TAIL_RANK:
        return (_CONFIG_TAIL_RANK[name], 0)
    # Everything else appearing before initial_state is a physical
    # initial-field kwarg (wavefield_0, pressure_0, vz_0, sigmazz_0, r_0,
    # wavefield_sc_0, sigmazz_sc_0, sigma_v_0, ...; see A4.3).
    return (_RANK_PHYSICAL_INITIAL_FIELD, 0)


def _kwonly_tail(sig: inspect.Signature) -> list[inspect.Parameter]:
    """Returns the parameters of ``sig`` from ``dt`` (exclusive) onward."""
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    assert "dt" in names, "every propagator signature must have a dt parameter"
    dt_index = names.index("dt")
    return params[dt_index + 1 :]


# ---------------------------------------------------------------------------
# (a) Everything after dt is keyword-only.
# ---------------------------------------------------------------------------


def test_functionals_keyword_only_after_dt() -> None:
    """A4.1: every propagator functional's parameters after ``dt`` are kw-only."""
    for name, fn in _FUNCTIONALS.items():
        sig = inspect.signature(fn)
        for p in _kwonly_tail(sig):
            assert p.kind is inspect.Parameter.KEYWORD_ONLY, (
                f"{name}: parameter {p.name!r} after dt is {p.kind.name}, "
                "expected KEYWORD_ONLY"
            )


def test_module_forwards_keyword_only_after_dt() -> None:
    """A4.1: every Module's ``forward`` parameters after ``dt`` are kw-only."""
    for name, cls in _MODULES.items():
        sig = inspect.signature(cls.forward)
        for p in _kwonly_tail(sig):
            assert p.kind is inspect.Parameter.KEYWORD_ONLY, (
                f"{name}.forward: parameter {p.name!r} after dt is "
                f"{p.kind.name}, expected KEYWORD_ONLY"
            )


# ---------------------------------------------------------------------------
# (b) Shared keyword names have the same default everywhere, and every
# propagator's own tail follows the canonical relative order (also covers
# (d), since the channel groups' sub-order is baked into _rank()).
#
# There are no exceptions: every keyword shared by two or more propagators
# (e.g. mpml_fraction) has the same default in all of them.
# ---------------------------------------------------------------------------


def test_shared_keyword_defaults_are_consistent() -> None:
    """A keyword shared by two+ propagators must default the same everywhere."""
    seen: dict[str, tuple[Any, str]] = {}
    for prop_name, fn in _FUNCTIONALS.items():
        sig = inspect.signature(fn)
        for p in _kwonly_tail(sig):
            if p.name in seen:
                prev_default, prev_prop = seen[p.name]
                assert p.default == prev_default, (
                    f"{p.name!r} has default {p.default!r} in {prop_name} but "
                    f"{prev_default!r} in {prev_prop} -- shared keyword names "
                    "must have the same default everywhere (A4 section 0.6)."
                )
            else:
                seen[p.name] = (p.default, prop_name)


def _init_params(cls: Any) -> dict[str, inspect.Parameter]:
    """Returns a Module ``__init__``'s parameters, excluding ``self``."""
    return {
        name: p
        for name, p in inspect.signature(cls.__init__).parameters.items()
        if name != "self"
    }


def test_module_init_shared_keyword_defaults_are_consistent() -> None:
    """A constructor parameter shared by two+ Modules defaults the same way.

    Extends the functional-side check above to the Module constructors (e.g.
    ``c66``/``c66_requires_grad`` across ``ElasticVti``/``ElasticTti``,
    ``storage_mode``/``storage_path``/``storage_compression`` everywhere).
    Parameters with no default (required, e.g. ``v``, ``grid_spacing``) are
    skipped, since "no default" trivially agrees with itself.
    """
    seen: dict[str, tuple[Any, str]] = {}
    for prop_name, cls in _MODULES.items():
        for p in _init_params(cls).values():
            if p.default is inspect.Parameter.empty:
                continue
            if p.name in seen:
                prev_default, prev_prop = seen[p.name]
                assert p.default == prev_default, (
                    f"{p.name!r} defaults to {p.default!r} in {prop_name}'s "
                    f"__init__ but {prev_default!r} in {prev_prop}'s -- shared "
                    "constructor parameter names must default identically."
                )
            else:
                seen[p.name] = (p.default, prop_name)


def test_functionals_follow_canonical_relative_order() -> None:
    """A4 section 0.6: each functional's kw-only tail is in canonical order."""
    for name, fn in _FUNCTIONALS.items():
        sig = inspect.signature(fn)
        tail = _kwonly_tail(sig)
        ranks = [_rank(p.name) for p in tail]
        sorted_indices = sorted(range(len(tail)), key=lambda i: ranks[i])
        assert ranks == [ranks[i] for i in sorted_indices], (
            f"{name}: keyword-only tail is not in canonical order.\n"
            f"  got:      {[p.name for p in tail]}\n"
            f"  expected: {[tail[i].name for i in sorted_indices]}"
        )


def test_module_forwards_follow_canonical_relative_order() -> None:
    """A4 section 0.6: each Module's ``forward`` kw-only tail is in canonical order."""
    for name, cls in _MODULES.items():
        sig = inspect.signature(cls.forward)
        tail = _kwonly_tail(sig)
        ranks = [_rank(p.name) for p in tail]
        sorted_indices = sorted(range(len(tail)), key=lambda i: ranks[i])
        assert ranks == [ranks[i] for i in sorted_indices], (
            f"{name}.forward: keyword-only tail is not in canonical order.\n"
            f"  got:      {[p.name for p in tail]}\n"
            f"  expected: {[tail[i].name for i in sorted_indices]}"
        )


# ---------------------------------------------------------------------------
# (c) Each Module's forward mirrors its functional's keyword tail exactly,
# minus constructor-held parameters.
# ---------------------------------------------------------------------------


def test_module_forward_mirrors_functional_tail() -> None:
    """A4.6/A4.3(c): Module.forward mirrors its functional's kw tail exactly.

    Modulo the parameters the Module's constructor already holds
    (``_CONSTRUCTOR_HELD``): required/optional model tensors and the
    storage_mode/storage_path/storage_compression trio.
    """
    for name, functional in _FUNCTIONALS.items():
        func_sig = inspect.signature(functional)
        func_tail = {p.name: p.default for p in _kwonly_tail(func_sig)}
        mod_tail = {
            p.name: p.default
            for p in _kwonly_tail(inspect.signature(_MODULES[name].forward))
        }
        held = _CONSTRUCTOR_HELD[name]
        expected = {n: d for n, d in func_tail.items() if n not in held}
        assert set(mod_tail) == set(expected), (
            f"{name}: Module.forward's keyword tail does not match the "
            f"functional's tail (minus constructor-held params {held}).\n"
            f"  Module has, functional lacks: {set(mod_tail) - set(expected)}\n"
            f"  functional has, Module lacks: {set(expected) - set(mod_tail)}"
        )
        for pname, default in expected.items():
            assert mod_tail[pname] == default, (
                f"{name}.forward's {pname!r} default {mod_tail[pname]!r} "
                f"does not match {name}'s functional default {default!r}."
            )


def test_module_init_matches_functional_for_shared_concepts() -> None:
    """A Module's ``__init__`` mirrors its functional's default for shared names.

    A parameter name appearing in BOTH a propagator's functional signature
    and that propagator's Module ``__init__`` signature (e.g. ``c66`` for
    ``elastic_vti``/``ElasticVti``; ``dv``/``scatter`` for
    ``scalar_born``/``ScalarBorn``) represents the same modelling concept in
    both, and so must agree on whether it is required and, if optional, on
    its default value. This is what would have caught ``ElasticVti``'s
    ``c66`` once being a required constructor argument while
    ``elastic_vti()``'s ``c66`` was already optional.
    """
    for name, cls in _MODULES.items():
        func_params = inspect.signature(_FUNCTIONALS[name]).parameters
        mod_params = _init_params(cls)
        for pname in set(func_params) & set(mod_params):
            fp, mp = func_params[pname], mod_params[pname]
            f_required = fp.default is inspect.Parameter.empty
            m_required = mp.default is inspect.Parameter.empty
            assert f_required == m_required, (
                f"{name}: {pname!r} is "
                f"{'required' if f_required else 'optional'} in the "
                f"functional but "
                f"{'required' if m_required else 'optional'} in the Module "
                "constructor -- shared concept parameters must agree on "
                "whether they are required."
            )
            if not f_required:
                assert fp.default == mp.default, (
                    f"{name}: {pname!r} defaults to {fp.default!r} in the "
                    f"functional but {mp.default!r} in the Module "
                    "constructor."
                )


# ---------------------------------------------------------------------------
# (f) The deprecated python_backend alias is present on exactly the four
# propagators v0.0.27 shipped, in both the functional and Module.forward.
# ---------------------------------------------------------------------------


def test_python_backend_alias_only_on_released_propagators() -> None:
    """Only the four v0.0.27 propagators retain their released alias."""
    released = {"scalar", "scalar_born", "acoustic", "elastic"}
    for name, functional in _FUNCTIONALS.items():
        for sig in (
            inspect.signature(functional),
            inspect.signature(_MODULES[name].forward),
        ):
            params = sig.parameters
            assert ("python_backend" in params) == (name in released)
            if name in released:
                assert params["python_backend"].default is None


def test_backend_is_str_defaulting_to_auto_everywhere() -> None:
    """All 22 entry points take ``backend: str = "auto"``.

    ``backend`` is deliberately typed ``str`` rather than a closed
    ``Literal``: a closed set of implementation technologies would be a
    promise to keep those technologies.
    """
    for label, sig in [
        (name, inspect.signature(fn)) for name, fn in _FUNCTIONALS.items()
    ] + [
        (f"{name}.forward", inspect.signature(cls.forward))
        for name, cls in _MODULES.items()
    ]:
        parameter = sig.parameters["backend"]
        assert parameter.default == "auto", (
            f"{label}: backend defaults to {parameter.default!r}, expected 'auto'"
        )
        assert parameter.annotation is str, (
            f"{label}: backend annotates as {parameter.annotation!r}, expected str"
        )


def test_every_module_has_one_requires_grad_flag_per_model() -> None:
    """All nine Modules carry ``<model>_requires_grad``, one per model.

    The flags are not a hand-kept list: the set each ``Module`` declares must
    be exactly the set of public model names the propagator's spec reports,
    so a propagator that gains or loses a model gains or loses its flag with
    it rather than drifting. ``scatter_requires_grad`` is the one addition,
    a deprecated alias for ``ScalarBorn``'s ``dv_requires_grad``.
    """
    for name, cls in _MODULES.items():
        flags = {
            parameter[: -len("_requires_grad")]
            for parameter in inspect.signature(cls.__init__).parameters
            if parameter.endswith("_requires_grad")
        }
        flags.discard("scatter")
        expected = set(_all_user_model_names(name))
        assert flags == expected, (
            f"{name}: declares {sorted(flags)} but its models are {sorted(expected)}."
        )


def test_constructors_are_keyword_only_after_grid_spacing() -> None:
    """Every Module constructor takes only keywords after ``grid_spacing``.

    ``grid_spacing`` is the last required parameter of all nine, so the
    rule is unambiguous even though its positional slot varies from 2 to 6.
    """
    for name, cls in _MODULES.items():
        parameters = list(inspect.signature(cls.__init__).parameters.values())
        index = next(i for i, p in enumerate(parameters) if p.name == "grid_spacing")
        positional = [
            p.name
            for p in parameters[index + 1 :]
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        assert not positional, (
            f"{name}: {positional} are still positional after grid_spacing"
        )


# ---------------------------------------------------------------------------
# (d) Channel groups follow the declared channel order (p, z, y, x, with
# acoustic_vti's sigma_v, sigma_h after them). Checked against the spec, and
# pinned explicitly for acoustic_vti too so a broken _rank() helper cannot
# silently hide a real regression.
# ---------------------------------------------------------------------------


def test_channel_groups_are_in_the_spec_order() -> None:
    """Every channel group is listed in the order the spec declares it.

    The spec's channel order is the order of the returned receiver tuple, so
    a signature that lists the same channels in another order would give a
    reader two orders to learn for one set of channels. The deprecated ``_p``
    spelling sits beside the canonical name of the channel it aliases, which
    is why it is folded onto it rather than ranked separately.
    """
    for name, functional in _FUNCTIONALS.items():
        spec = deepwave.spec(name)
        ndim = max(spec.ndims)
        for kind, channels, kwarg in (
            ("source", spec.source_channels(ndim), spec.source_amplitudes_kwarg),
            (
                "receiver",
                spec.receiver_channels(ndim),
                spec.receiver_locations_kwarg,
            ),
        ):
            expected = [kwarg(channel) for channel in channels]
            names = [p.name for p in _kwonly_tail(inspect.signature(functional))]
            found = [n for n in names if n in set(expected)]
            assert found == expected, (name, kind, found, expected)


def test_acoustic_vti_stress_channels_are_sigma_v_first() -> None:
    """acoustic_vti's two stress channels are sigma_v before sigma_h.

    The scalar channel leads the group, as on every other multi-channel
    propagator (it is the synthesised pressure-rate channel that drives both
    stresses), with the deprecated ``_p`` spelling beside it; the velocity
    channels follow in the shared z/y/x order, and the two stress channels
    come last, sigma_v (axial) before sigma_h (transverse).
    """
    sig = inspect.signature(deepwave.acoustic_vti)
    names = [p.name for p in _kwonly_tail(sig)]
    amp_channels = [n for n in names if n.startswith("source_amplitudes")]
    assert amp_channels == [
        "source_amplitudes_pressure_rate",
        "source_amplitudes_z",
        "source_amplitudes_y",
        "source_amplitudes_x",
        "source_amplitudes_sigma_v",
        "source_amplitudes_sigma_h",
    ]


def test_scalar_source_channel_before_zyx_channels() -> None:
    """A4.2: the scalar source channel comes before the z, y, x channels.

    It is spelled by the quantity it injects -- ``volume_rate`` on the
    acoustic family, ``pressure_rate`` on viscoacoustic and the elastic
    family -- with the released deprecated ``_p`` spelling immediately
    after it where applicable.
    """
    for name in ("acoustic", "viscoacoustic", "elastic", "elastic_vti"):
        rate = "volume_rate" if name == "acoustic" else "pressure_rate"
        sig = inspect.signature(_FUNCTIONALS[name])
        names = [p.name for p in _kwonly_tail(sig)]
        amp_channels = [n for n in names if n.startswith("source_amplitudes")]
        assert amp_channels[0] == f"source_amplitudes_{rate}"
        assert ("source_amplitudes_p" in amp_channels) == (
            name in {"acoustic", "elastic"}
        )


def test_born_bg_receiver_locations_before_receiver_locations() -> None:
    """A4 section 0.6: bg_receiver_locations_* precede sc_receiver_locations_*.

    The scattered receivers are spelled ``sc_`` on all three Born
    propagators, so that the keyword names match the order of the returned
    tuple (background first, scattered second).
    """
    for name in ("acoustic_born", "elastic_born"):
        sig = inspect.signature(_FUNCTIONALS[name])
        names = [p.name for p in _kwonly_tail(sig)]
        assert not any(n.startswith("receiver_locations") for n in names), (
            f"{name}: the bare receiver_locations_* spelling is gone"
        )
        bg_indices = [
            i for i, n in enumerate(names) if n.startswith("bg_receiver_locations")
        ]
        recv_indices = [
            i for i, n in enumerate(names) if n.startswith("sc_receiver_locations")
        ]
        assert bg_indices, f"{name}: expected at least one bg_receiver_locations_*"
        assert recv_indices, f"{name}: expected at least one sc_receiver_locations_*"
        assert max(bg_indices) < min(recv_indices), (
            f"{name}: bg_receiver_locations_* must all precede sc_receiver_locations_*"
        )


# ---------------------------------------------------------------------------
# (e) Docstring truth: every documented Args name is a real parameter, and
# every documented Returns attribute exists on the real PropagatorOutput.
# ---------------------------------------------------------------------------

# After ``inspect.cleandoc`` (needed on Python 3.9, where ``__doc__`` retains
# indentation), Google-style headings are at
# column 0, each entry's name at exactly 4 spaces, and continuation lines/
# nested bullets at 8+ spaces -- so an entry line is distinguished from a
# wrapped continuation line by requiring EXACTLY 4 spaces of indentation.
_ARGS_ENTRY_RE = re.compile(r"^ {4}([A-Za-z_][A-Za-z0-9_]*):", re.MULTILINE)
# Returns: blocks in this codebase (where they enumerate fields at all) use an
# 8-space-indented bullet list, "- name: description" or "- name (...): ...".
# A bulleted description that instead starts with a templated/variadic name
# (e.g. "- r_1, ..., r_{q_n_mechanisms - 1}: ...") is deliberately NOT matched
# (the identifier is not immediately followed by ":" or "("), since it does
# not claim one single fixed attribute name.
_RETURNS_ENTRY_RE = re.compile(r"^ {8}- ([A-Za-z_][A-Za-z0-9_]*)\s*[:(]", re.MULTILINE)


def _args_section(doc: str) -> str:
    """Returns the body of a docstring's ``Args:`` section (entries only)."""
    m = re.search(r"\nArgs:\n(.*?)\nReturns:\n", inspect.cleandoc(doc), re.DOTALL)
    assert m, "docstring has no Args:/Returns: section in the expected form"
    return m.group(1)


def _returns_section(doc: str) -> str:
    """Returns the body of a docstring's ``Returns:`` section."""
    m = re.search(r"\nReturns:\n(.*)", inspect.cleandoc(doc), re.DOTALL)
    assert m, "docstring has no Returns: section in the expected form"
    return m.group(1)


def _documented_args(fn: Any) -> set[str]:
    return set(_ARGS_ENTRY_RE.findall(_args_section(fn.__doc__)))


def _documented_returns(fn: Any) -> set[str]:
    return set(_RETURNS_ENTRY_RE.findall(_returns_section(fn.__doc__)))


# One tiny, real, CPU float32 eager-backend propagation per propagator,
# returning the set of real PropagatorOutput field names. Where a
# propagator's Returns: docstring documents dimension-conditional fields
# (e.g. "(if ndim == 3)"), the propagation is run at the propagator's largest
# supported number of spatial dimensions, which -- because every such
# conditional in this codebase is monotonic in ndim ("if ndim == 3" or
# "if ndim >= 2") except acoustic_vti's single axis-tied PML memory variable
# (named m_sigma_v_z_0 at ndim == 3 XOR m_sigma_v_y_0 at ndim == 2) -- makes a
# single run's field set a superset of everything documented, without needing
# every possible ndim. acoustic_vti additionally unions in a ndim == 2 run to
# cover that one exception.
_DTYPE = torch.float32


def _tiny_scalar() -> frozenset[str]:
    v = torch.full((10, 10, 10), 1500.0, dtype=_DTYPE)
    sa = torch.zeros(1, 1, 4, dtype=_DTYPE)
    out = deepwave.scalar(
        v,
        5.0,
        0.0007,
        source_amplitudes=sa,
        source_locations=torch.tensor([[[5, 5, 5]]]),
        receiver_locations=torch.tensor([[[4, 4, 4]]]),
        pml_width=3,
        accuracy=2,
        pml_freq=15.0,
        max_vel=2000.0,
        backend="eager",
    )
    return frozenset(out._names)


def _tiny_scalar_born() -> frozenset[str]:
    v = torch.full((10, 10, 10), 1500.0, dtype=_DTYPE)
    dv = torch.zeros(10, 10, 10, dtype=_DTYPE)
    sa = torch.zeros(1, 1, 4, dtype=_DTYPE)
    out = deepwave.scalar_born(
        v,
        5.0,
        0.0007,
        dv=dv,
        source_amplitudes=sa,
        source_locations=torch.tensor([[[5, 5, 5]]]),
        sc_receiver_locations=torch.tensor([[[4, 4, 4]]]),
        bg_receiver_locations=torch.tensor([[[4, 4, 4]]]),
        pml_width=3,
        accuracy=2,
        pml_freq=15.0,
        max_vel=2000.0,
        backend="eager",
    )
    return frozenset(out._names)


def _tiny_acoustic() -> frozenset[str]:
    shape = (10, 10, 10)
    v = torch.full(shape, 1500.0, dtype=_DTYPE)
    rho = torch.full(shape, 1000.0, dtype=_DTYPE)
    sa = torch.zeros(1, 1, 4, dtype=_DTYPE)
    out = deepwave.acoustic(
        v,
        rho,
        5.0,
        0.0007,
        source_amplitudes_volume_rate=sa,
        source_locations_volume_rate=torch.tensor([[[5, 5, 5]]]),
        receiver_locations_p=torch.tensor([[[4, 4, 4]]]),
        pml_width=3,
        accuracy=2,
        pml_freq=15.0,
        max_vel=2000.0,
        backend="eager",
    )
    return frozenset(out._names)


def _tiny_acoustic_born() -> frozenset[str]:
    shape = (8, 8, 8)
    v = torch.full(shape, 1500.0, dtype=_DTYPE)
    rho = torch.full(shape, 1000.0, dtype=_DTYPE)
    dv = torch.zeros(shape, dtype=_DTYPE)
    drho = torch.zeros(shape, dtype=_DTYPE)
    sa = torch.zeros(1, 1, 4, dtype=_DTYPE)
    out = deepwave.acoustic_born(
        v,
        rho,
        5.0,
        0.0007,
        dv=dv,
        drho=drho,
        source_amplitudes_volume_rate=sa,
        source_locations_volume_rate=torch.tensor([[[4, 4, 4]]]),
        sc_receiver_locations_p=torch.tensor([[[3, 3, 3]]]),
        bg_receiver_locations_p=torch.tensor([[[3, 3, 3]]]),
        pml_width=2,
        accuracy=2,
        pml_freq=15.0,
        max_vel=2000.0,
        backend="eager",
    )
    return frozenset(out._names)


def _tiny_acoustic_vti() -> frozenset[str]:
    names: set[str] = set()
    c11 = torch.full((10, 10), 6e9, dtype=_DTYPE)
    c13 = torch.full((10, 10), 2e9, dtype=_DTYPE)
    c33 = torch.full((10, 10), 4e9, dtype=_DTYPE)
    buoy = torch.full((10, 10), 1 / 2000.0, dtype=_DTYPE)
    ampx = torch.zeros(1, 1, 4, dtype=_DTYPE)
    out2d = deepwave.acoustic_vti(
        c11,
        c13,
        c33,
        buoy,
        5.0,
        0.00025,
        source_amplitudes_x=ampx,
        source_locations_x=torch.tensor([[[5, 5]]]),
        receiver_locations_x=torch.tensor([[[4, 4]]]),
        pml_width=2,
        accuracy=2,
        max_vel=6000.0,
        pml_freq=15.0,
        backend="eager",
    )
    names.update(out2d._names)
    c11_3d = torch.full((10, 10, 10), 6e9, dtype=_DTYPE)
    c13_3d = torch.full((10, 10, 10), 2e9, dtype=_DTYPE)
    c33_3d = torch.full((10, 10, 10), 4e9, dtype=_DTYPE)
    buoy_3d = torch.full((10, 10, 10), 1 / 2000.0, dtype=_DTYPE)
    ampz = torch.zeros(1, 1, 4, dtype=_DTYPE)
    out3d = deepwave.acoustic_vti(
        c11_3d,
        c13_3d,
        c33_3d,
        buoy_3d,
        5.0,
        0.00025,
        source_amplitudes_z=ampz,
        source_locations_z=torch.tensor([[[5, 5, 5]]]),
        receiver_locations_z=torch.tensor([[[4, 4, 4]]]),
        pml_width=2,
        accuracy=2,
        max_vel=6000.0,
        pml_freq=15.0,
        backend="eager",
    )
    names.update(out3d._names)
    return frozenset(names)


def _tiny_elastic() -> frozenset[str]:
    shape = (14, 14, 14)
    c, cs, rho = 1500.0, 900.0, 1000.0
    lamb = torch.full(shape, rho * (c**2 - 2 * cs**2), dtype=_DTYPE)
    mu = torch.full(shape, rho * cs**2, dtype=_DTYPE)
    buoy = torch.full(shape, 1 / rho, dtype=_DTYPE)
    sa = torch.zeros(1, 1, 4, dtype=_DTYPE)
    out = deepwave.elastic(
        lamb,
        mu,
        buoy,
        5.0,
        0.0002,
        source_amplitudes_pressure_rate=sa,
        source_locations_pressure_rate=torch.tensor([[[7, 7, 7]]]),
        receiver_locations_p=torch.tensor([[[5, 5, 5]]]),
        pml_width=4,
        accuracy=2,
        pml_freq=25.0,
        backend="eager",
    )
    return frozenset(out._names)


def _tiny_elastic_born() -> frozenset[str]:
    shape = (10, 11, 12)
    vp = torch.full(shape, 2000.0, dtype=_DTYPE)
    vs = torch.full(shape, 1100.0, dtype=_DTYPE)
    rho = torch.full(shape, 2200.0, dtype=_DTYPE)
    lamb, mu, bu = deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    dlamb = torch.zeros(shape, dtype=_DTYPE)
    dmu = torch.zeros(shape, dtype=_DTYPE)
    dbu = torch.zeros(shape, dtype=_DTYPE)
    src = torch.zeros(1, 1, 4, dtype=_DTYPE)
    rl = torch.tensor([[[3, 4, 5], [7, 8, 9]]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = deepwave.elastic_born(
            lamb,
            mu,
            bu,
            5.0,
            4e-4,
            dlamb=dlamb,
            dmu=dmu,
            dbuoyancy=dbu,
            source_amplitudes_y=src,
            source_locations_y=torch.tensor([[[5, 6, 7]]]),
            sc_receiver_locations_y=rl,
            sc_receiver_locations_p=rl,
            pml_width=2,
            backend="eager",
            pml_freq=15.0,
            max_vel=2200.0,
        )
    return frozenset(out._names)


def _tiny_elastic_vti() -> frozenset[str]:
    shape = (14, 14, 14)
    c, cs, rho = 1500.0, 900.0, 1000.0
    c11 = torch.full(shape, rho * (c**2 - 2 * cs**2) + 2 * rho * cs**2, dtype=_DTYPE)
    c13 = torch.full(shape, rho * (c**2 - 2 * cs**2), dtype=_DTYPE)
    c33 = c11.clone()
    c55 = torch.full(shape, rho * cs**2, dtype=_DTYPE)
    c66 = c55.clone()
    buoy = torch.full(shape, 1 / rho, dtype=_DTYPE)
    sa = torch.zeros(1, 1, 4, dtype=_DTYPE)
    out = deepwave.elastic_vti(
        c11,
        c13,
        c33,
        c55,
        buoy,
        5.0,
        0.0002,
        c66=c66,
        source_amplitudes_pressure_rate=sa,
        source_locations_pressure_rate=torch.tensor([[[7, 7, 7]]]),
        receiver_locations_p=torch.tensor([[[5, 5, 5]]]),
        pml_width=4,
        accuracy=2,
        pml_freq=25.0,
        backend="eager",
    )
    return frozenset(out._names)


def _tiny_viscoacoustic() -> frozenset[str]:
    shape = (10, 10, 10)
    v = torch.full(shape, 1500.0, dtype=_DTYPE)
    rho = torch.full(shape, 1000.0, dtype=_DTYPE)
    q = torch.full(shape, 50.0, dtype=_DTYPE)
    sa = torch.zeros(1, 1, 4, dtype=_DTYPE)
    out = deepwave.viscoacoustic(
        v,
        rho,
        q,
        5.0,
        0.0007,
        source_amplitudes_pressure_rate=sa,
        source_locations_pressure_rate=torch.tensor([[[5, 5, 5]]]),
        receiver_locations_p=torch.tensor([[[4, 4, 4]]]),
        pml_width=4,
        accuracy=2,
        pml_freq=15.0,
        q_freq=15.0,
        max_vel=2000.0,
        backend="eager",
    )
    return frozenset(out._names)


_TINY_REAL_OUTPUT_NAMES: dict[str, Callable[[], frozenset[str]]] = {
    "scalar": _tiny_scalar,
    "scalar_born": _tiny_scalar_born,
    "acoustic": _tiny_acoustic,
    "acoustic_born": _tiny_acoustic_born,
    "acoustic_vti": _tiny_acoustic_vti,
    "elastic": _tiny_elastic,
    "elastic_born": _tiny_elastic_born,
    "elastic_vti": _tiny_elastic_vti,
    "viscoacoustic": _tiny_viscoacoustic,
}
assert set(_TINY_REAL_OUTPUT_NAMES) == set(_FUNCTIONALS)


def test_docstring_args_are_real_parameters() -> None:
    """Every name documented in a functional's Args: is a real parameter.

    Catches phantom documented-but-nonexistent parameters (e.g. the 18
    ``m_v*_0``/``m_sigma*_0`` CPML-memory-variable entries once documented on
    :func:`deepwave.elastic` but absent from its actual signature).
    """
    for name, fn in _FUNCTIONALS.items():
        documented = _documented_args(fn)
        real = set(inspect.signature(fn).parameters)
        phantom = documented - real
        assert not phantom, (
            f"{name}: Args: documents parameter(s) {sorted(phantom)} that do "
            "not exist in the real signature"
        )


def test_docstring_returns_are_real_output_fields() -> None:
    """Every attribute documented in a functional's Returns: is real.

    Runs one tiny, real, CPU float32 propagation per propagator and checks
    every attribute name the docstring's Returns: section claims against the
    actual ``PropagatorOutput``'s field names (catches, e.g., the
    ``psiy_ntm1``/``wavefield_nt``-style fictional aliases once documented on
    :func:`deepwave.scalar`, which do not exist as real attribute names).
    """
    for name, fn in _FUNCTIONALS.items():
        documented = _documented_returns(fn)
        real = _TINY_REAL_OUTPUT_NAMES[name]()
        phantom = documented - real
        assert not phantom, (
            f"{name}: Returns: documents attribute(s) {sorted(phantom)} that "
            "are not real PropagatorOutput fields (from a tiny real "
            f"propagation, real fields were: {sorted(real)})"
        )


def _args_entries(fn: Any) -> dict[str, str]:
    """Every ``Args:`` entry of a functional, as ``{name: body}``.

    The body is the entry's text with its line breaks flattened, which is how
    a reader sees it once Sphinx has reflowed the paragraph.
    """
    section = _args_section(fn.__doc__)
    entries: dict[str, str] = {}
    current: str | None = None
    for line in section.split("\n"):
        match = _ARGS_ENTRY_RE.match(line)
        if match:
            current = match.group(1)
            entries[current] = line.split(":", 1)[1].strip()
        elif current is not None and line.strip():
            entries[current] = f"{entries[current]} {line.strip()}"
    return entries


def _documented_channels(body: str) -> list[str]:
    """The channel names a ``sources:``/``receivers:`` entry lists."""
    match = re.search(
        r"(?:channels? (?:are|is)|has one channel,) (.*?)(?:\(|, with|"
        r", and a channel|\. )",
        body,
    )
    assert match, f"no channel list found in: {body}"
    return re.findall(r"``(\w+)``", match.group(1))


def test_documented_channel_lists_match_the_spec() -> None:
    """The ``sources=``/``receivers=`` entries list this propagator's channels.

    The mapping form's whole point is that the channel names are discoverable,
    so a pasted-in list from another propagator (a ``z`` channel on a
    single-channel or 2-D-only propagator, say) is a defect the reader cannot
    detect. Compared against the spec at the largest supported ndim, where
    every channel that exists at any ndim is present.
    """
    for name, fn in _FUNCTIONALS.items():
        spec = deepwave.spec(name)
        ndim = max(spec.ndims)
        entries = _args_entries(fn)
        for parameter, expected in (
            ("sources", spec.source_channels(ndim)),
            ("receivers", spec.receiver_channels(ndim)),
        ):
            listed = _documented_channels(entries[parameter])
            assert listed == list(expected), (
                f"{name}: the {parameter}= entry lists channels {listed}, but "
                f"the propagator's are {list(expected)}"
            )


def _mask_entries(fn: Any) -> dict[str, str]:
    """The ``Args:`` entries that describe a mask in full (not the aliases)."""
    return {
        name: body
        for name, body in _args_entries(fn).items()
        if "_mask" in name and "IGNORE_LOCATION" in body
    }


def test_mask_entries_state_the_hicks_restriction() -> None:
    """A mask cannot be combined with a ``Hicks``, and must say so.

    The paired locations entry actively recommends passing a
    :class:`deepwave.Hicks`, while ``apply_location_masks`` raises for that
    combination, so a mask entry that does not mention it sends the reader
    into an avoidable error.
    """
    for name, fn in _FUNCTIONALS.items():
        for parameter, body in _mask_entries(fn).items():
            assert "Hicks" in body, (
                f"{name}: the {parameter} entry does not say that a mask "
                "cannot be combined with a Hicks object"
            )


def test_mask_entries_state_the_right_consequence() -> None:
    """A source mask injects nothing; a receiver mask records zeros.

    One shared template used to claim both of both kinds, which is wrong for
    half of every mask entry: a receiver injects nothing whatever its mask
    says, and a source records nothing whatever its mask says.
    """
    for name, fn in _FUNCTIONALS.items():
        for parameter, body in _mask_entries(fn).items():
            wanted, unwanted = (
                ("injects nothing", "records zeros")
                if "source_mask" in parameter
                else ("records zeros", "injects nothing")
            )
            assert wanted in body, (
                f"{name}: the {parameter} entry does not say that a masked-out "
                f"point {wanted}"
            )
            assert unwanted not in body, (
                f"{name}: the {parameter} entry says a masked-out point "
                f"{unwanted}, which is not what a "
                f"{'source' if 'source_mask' in parameter else 'receiver'} "
                "mask does"
            )


def test_z_channel_entries_state_their_dimensionality() -> None:
    """Every ``z``-channel entry says when the channel exists.

    The ``z`` channel exists only in 3-D, so its entry must state that
    dimensionality clearly.
    """
    for name, fn in _FUNCTIONALS.items():
        ndims = deepwave.spec(name).ndims
        for parameter, body in _args_entries(fn).items():
            if not parameter.endswith("_z"):
                continue
            if 3 in ndims:
                assert "3-D" in body, (
                    f"{name}: the {parameter} entry does not say that the "
                    "``z`` channel exists only in 3-D"
                )
            else:
                assert "2-D only" in body, (
                    f"{name}: the {parameter} entry does not say that this "
                    "propagator has no ``z`` channel"
                )
