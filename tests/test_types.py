"""Tests for the public types module.

Covers the three things that can silently rot about ``deepwave.types``:

* a ``Literal`` alias drifting away from the runtime check it mirrors;
* the ``Callback`` protocol rejecting a callback it should accept;
* the public types reporting a private module, which would put
  ``deepwave._engine.RunInfo`` into ``repr``, ``pickle`` and the API extract.
"""

import functools
import pickle
import subprocess
import sys
import typing

import pytest
import torch

import deepwave
import deepwave.types


def test_public_types_report_deepwave_module() -> None:
    """The public types name their public home, not the file they live in."""
    for name in (
        "CallbackState",
        "PropagatorOutput",
        "RunInfo",
        "MemoryEstimate",
        "Callback",
        "AuxiliaryState",
    ):
        cls = getattr(deepwave, name)
        assert cls.__module__ == "deepwave", (
            f"{name}.__module__ is {cls.__module__!r}, not 'deepwave'"
        )
        # The spelling the module claims must actually resolve to it, or a
        # pickle round trip and every "where does this come from?" lookup
        # would go somewhere wrong.
        assert getattr(deepwave, cls.__name__) is cls


#: The only ``deepwave`` modules ``deepwave.types`` may import at module
#: level. Their allowed dependency chain is checked below for cycles.
_TYPES_PERMITTED_IMPORTS = frozenset({"deepwave._exceptions", "deepwave._pytree"})
_TYPES_DEPENDENCIES = {
    "deepwave._exceptions": [],
    "deepwave._pytree": [],
}


def _module_level_deepwave_imports(module: object) -> list[str]:
    """The ``deepwave`` modules ``module`` imports at module level.

    Only module-level imports can create a cycle at import time: a
    ``TYPE_CHECKING`` block never executes, and neither does an import inside
    a function body until it is called.

    Args:
        module: The imported module to read the source of.

    Returns:
        The fully qualified names, in source order.

    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(module.__file__).read_text())  # type: ignore[attr-defined]
    imported: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            package = node.module or ""
            if package == "deepwave":
                # ``from deepwave import x`` imports ``deepwave.x``.
                imported.extend(f"deepwave.{alias.name}" for alias in node.names)
            elif package.startswith("deepwave"):
                imported.append(package)
        elif isinstance(node, ast.Import):
            imported.extend(
                alias.name for alias in node.names if alias.name.startswith("deepwave")
            )
    return imported


def test_types_module_is_an_import_leaf() -> None:
    """``deepwave.types`` imports only the ``deepwave`` modules that are leaves.

    The whole reason it is a separate file is that ``_common`` and
    ``_engine`` refer to each other, so the public types cannot live in
    either without risking a cycle. An implementer who "simplifies" by
    importing ``_common`` here gets an ``ImportError`` that only appears
    under some import orders, so the rule is checked statically -- at run
    time, importing any submodule imports the package first, which would
    hide the violation.
    """
    offenders = [
        name
        for name in _module_level_deepwave_imports(deepwave.types)
        if name not in _TYPES_PERMITTED_IMPORTS
    ]
    assert not offenders, (
        "deepwave/types.py must stay an import leaf, but it imports "
        f"{offenders} at module level"
    )


@pytest.mark.parametrize("name", sorted(_TYPES_PERMITTED_IMPORTS))
def test_the_modules_types_may_import_form_an_acyclic_chain(name: str) -> None:
    """Dependencies of ``types`` may only refer to earlier leaves in the chain."""
    import importlib

    assert (
        _module_level_deepwave_imports(importlib.import_module(name))
        == (_TYPES_DEPENDENCIES[name])
    )


# ---------------------------------------------------------------------------
# The Literal aliases match the runtime checks they mirror
# ---------------------------------------------------------------------------


def test_literal_aliases_match_their_runtime_validators() -> None:
    """Each ``Literal`` names exactly the values its validator accepts.

    This is the whole protection against an alias drifting from the check:
    widen one without the other and this fails.
    """
    from deepwave import _engine
    from deepwave._location_interpolation import _LOCATION_INTERPOLATION_MODES
    from deepwave.misfits import _REDUCTIONS

    assert typing.get_args(deepwave.StorageMode) == _engine._ALL_STORAGE_MODES
    assert typing.get_args(deepwave.LocationInterp) == _LOCATION_INTERPOLATION_MODES
    assert typing.get_args(deepwave.Reduction) == _REDUCTIONS
    assert typing.get_args(deepwave.Accuracy) == (2, 4, 6, 8)
    assert typing.get_args(deepwave.WavefieldView) == ("full", "pml", "inner")
    # ``"backward"`` is deliberately absent: it was a development-branch
    # synonym for ``"adjoint"`` that no release ever accepted, so it is not a
    # value and ``get_wavefield`` raises for it.
    assert typing.get_args(deepwave.CallbackRole) == ("adjoint", "forward")
    assert typing.get_args(deepwave.CallbackDirection) == ("forward", "backward")
    assert typing.get_args(deepwave.CallbackKind) == ("wavefield", "model", "gradient")
    assert typing.get_args(deepwave.CallbackMode) == ("read", "write")


#: A tiny ``scalar`` call's arguments, for the callback-alias checks below.
_CALLBACK_CALL = {
    "source_amplitudes": torch.zeros(1, 1, 4, dtype=torch.float64),
    "source_locations": torch.tensor([[[8, 8]]]),
    "receiver_locations": torch.tensor([[[9, 9]]]),
    "pml_freq": 25.0,
    "callback_frequency": 100,
}


def _in_a_callback(body: typing.Callable[[deepwave.CallbackState], None]) -> None:
    """Runs ``body`` inside a real forward callback of a tiny ``scalar`` call."""
    deepwave.scalar(
        torch.full((16, 16), 1500.0, dtype=torch.float64),
        4.0,
        0.001,
        forward_callback=body,
        **_CALLBACK_CALL,  # type: ignore[arg-type]
    )


def test_every_callback_kind_and_mode_literal_value_is_accepted() -> None:
    """The three callback aliases name values the accessors really take.

    ``CallbackKind`` and ``CallbackMode`` are checked at run time by
    :class:`deepwave.CallbackState`'s own accessors rather than against a
    module-level tuple, so the tie between alias and validator has to be made
    through a real call. ``CallbackDirection`` is checked by reading
    ``state.direction``, which is one of its values.
    """
    seen: list[str] = []

    def body(state: deepwave.CallbackState) -> None:
        for kind in typing.get_args(deepwave.CallbackKind):
            assert isinstance(state.writable("v", kind=kind), bool)
        for mode in typing.get_args(deepwave.CallbackMode):
            assert state.get_model("v", mode=mode) is not None
        seen.append(state.direction)

    _in_a_callback(body)
    assert seen
    assert set(seen) <= set(typing.get_args(deepwave.CallbackDirection))


def test_a_callback_kind_or_mode_outside_its_literal_is_rejected() -> None:
    """The other direction, so the check above cannot be met by a no-op."""
    bogus = "not_a_value"
    assert bogus not in typing.get_args(deepwave.CallbackKind)
    assert bogus not in typing.get_args(deepwave.CallbackMode)

    def body(state: deepwave.CallbackState) -> None:
        with pytest.raises(ValueError, match="kind must be"):
            state.writable("v", kind=bogus)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="mode must be"):
            state.get_model("v", mode=bogus)  # type: ignore[arg-type]

    _in_a_callback(body)


def test_storage_mode_literal_matches_the_private_enum_names() -> None:
    """A second tie for ``StorageMode``, to the enum behind the strings.

    ``deepwave._storage.StorageMode`` is an ``IntEnum`` whose *values* are
    0-3, so the alias cannot be compared against them directly; its member
    *names*, lowercased, are the string spellings.
    """
    from deepwave._storage import StorageMode as StorageModeEnum

    assert typing.get_args(deepwave.StorageMode) == tuple(
        member.name.lower() for member in StorageModeEnum
    )


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_every_accuracy_literal_value_is_actually_accepted(accuracy: int) -> None:
    """The ``Accuracy`` alias does not promise an order that then raises."""
    from deepwave._common import set_accuracy

    assert set_accuracy(accuracy) == accuracy


@pytest.mark.parametrize("mode", typing.get_args(deepwave.StorageMode))
def test_every_storage_mode_literal_value_resolves(mode: str) -> None:
    """Likewise for ``StorageMode``: every named value is a real mode.

    The values are *derived* from the alias rather than listed again here,
    so that widening ``StorageMode`` without teaching the checks about the
    new value fails, instead of leaving the "the alias matches" tests green
    while every call that passes the new value raises.

    Checked against the two validators a user's call actually reaches --
    the one every propagator goes through
    (``deepwave._engine.resolve_storage_mode``) and the one behind
    :func:`deepwave.estimate_memory` -- because those, not the tuple the
    error messages are built from, decide what is accepted.
    """
    from deepwave._engine import resolve_storage_mode

    assert resolve_storage_mode(mode, torch.device("cpu")) is not None
    estimate = deepwave.estimate_memory(
        "scalar", model_shape=(8, 8), nt=4, n_shots=1, storage_mode=mode
    )
    assert estimate["padded_model_elements"] > 0


def test_a_storage_mode_outside_the_literal_is_rejected() -> None:
    """The other direction: both checks refuse a value the alias omits.

    Without this, "every value resolves" could be satisfied by a validator
    that accepts anything at all.
    """
    from deepwave._engine import resolve_storage_mode

    bogus = "not_a_storage_mode"
    assert bogus not in typing.get_args(deepwave.StorageMode)
    with pytest.raises(ValueError, match="storage_mode must be"):
        resolve_storage_mode(bogus, torch.device("cpu"))
    with pytest.raises(ValueError, match="storage_mode must be"):
        deepwave.estimate_memory(
            "scalar", model_shape=(8, 8), nt=4, n_shots=1, storage_mode=bogus
        )


# ---------------------------------------------------------------------------
# The Callback protocol
# ---------------------------------------------------------------------------


def _named_parameter_callback(anything: deepwave.CallbackState) -> None:
    """A callback whose parameter is not called ``state``."""


class _CallableObject:
    def __call__(self, state: deepwave.CallbackState) -> None:
        """A callback that is an instance, not a function."""


class _WithMethod:
    def method(self, state: deepwave.CallbackState) -> None:
        """A callback that is a bound method."""


def _two_parameter(state: deepwave.CallbackState, extra: int) -> None:
    """Becomes a valid callback once ``extra`` is bound."""


def test_callback_protocol_accepts_lambda_and_named_functions() -> None:
    """Every ordinary spelling of a one-argument callable satisfies it.

    The protocol's ``__call__`` takes its argument positionally-only (the
    ``/``). Without that, the protocol would bind the parameter *name* and
    reject a bare ``lambda s: ...`` and every callback whose parameter is
    not called ``state`` -- which is most of them.
    """
    candidates = [
        lambda s: None,
        _named_parameter_callback,
        _CallableObject(),
        _WithMethod().method,
        functools.partial(_two_parameter, extra=1),
    ]
    for candidate in candidates:
        assert isinstance(candidate, deepwave.Callback), candidate


def test_callback_protocol_rejects_a_non_callable() -> None:
    """``runtime_checkable`` still distinguishes callables from data."""
    assert not isinstance(42, deepwave.Callback)
    assert not isinstance("not a callback", deepwave.Callback)


def test_callback_protocol_is_accepted_by_mypy_at_a_real_call_site() -> None:
    """A ``lambda`` passed as ``forward_callback`` type-checks.

    The runtime ``isinstance`` above cannot show this: protocol *parameter
    naming* is a static-only failure, so it needs a static check. This is
    the fixture that would have caught a protocol written without the ``/``.
    """
    pytest.importorskip("mypy")
    source = (
        "import deepwave\n"
        "def cb(anything: deepwave.CallbackState) -> None: ...\n"
        "f: deepwave.Callback = cb\n"
        "g: deepwave.Callback = lambda s: None\n"
        "def h(x: deepwave.CallbackState) -> None: ...\n"
        "i: deepwave.Callback = h\n"
    )
    result = subprocess.run(  # noqa: S603 -- the source is the literal above
        [sys.executable, "-m", "mypy", "--strict", "-c", source],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_initial_state_mapping_with_auxiliary_bundle_type_checks() -> None:
    """A typed mapping can preserve the complete state at a continuation."""
    pytest.importorskip("mypy")
    source = """from collections.abc import Mapping
from typing import Union
import torch
import deepwave

def restart(v: torch.Tensor, field: torch.Tensor,
            auxiliary: deepwave.AuxiliaryState,
            output: deepwave.PropagatorOutput) -> None:
    state: Mapping[str, Union[torch.Tensor, deepwave.AuxiliaryState]] = {
        "wavefield_0": field, "auxiliary_state": auxiliary,
    }
    initial: deepwave.InitialState = state
    final: Mapping[str, Union[torch.Tensor, deepwave.AuxiliaryState]] = (
        output.final_state
    )
    deepwave.scalar(v, 4.0, 0.001, initial_state=state, nt=5)
    deepwave.scalar(v, 4.0, 0.001, initial_state=final, nt=5)
"""
    result = subprocess.run(  # noqa: S603 -- fixed static consumer example
        [sys.executable, "-m", "mypy", "--strict", "-c", source],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_callback_protocol_is_used_on_the_propagators() -> None:
    """The propagators annotate their callbacks with the public protocol.

    ``Callback`` has to be a protocol *and used*; an unused
    protocol documents nothing.
    """
    import inspect

    signature = inspect.signature(deepwave.scalar)
    for name in ("forward_callback", "backward_callback"):
        annotation = signature.parameters[name].annotation
        assert "Callback" in str(annotation), (name, annotation)


# ---------------------------------------------------------------------------
# The types still behave as themselves after the move
# ---------------------------------------------------------------------------


def test_memory_estimate_is_exported_and_is_the_estimate_dicts_type() -> None:
    """``MemoryEstimate`` describes what ``estimate_memory`` returns."""
    estimate = deepwave.estimate_memory("scalar", [50, 50], nt=10, n_shots=1)
    assert set(estimate) == set(deepwave.MemoryEstimate.__annotations__)


def test_propagator_output_pickles_under_its_public_name() -> None:
    """Setting ``__module__`` did not break pickling of a real output.

    ``pickle`` resolves a class by ``__module__`` + ``__qualname__``, so
    claiming ``deepwave`` only works because ``deepwave.PropagatorOutput``
    really is the class.
    """
    v = torch.full((25, 25), 1500.0)
    source_amplitudes = torch.zeros(1, 1, 15)
    source_amplitudes[0, 0, 3] = 1.0
    out = deepwave.scalar(
        v,
        4.0,
        0.001,
        source_amplitudes=source_amplitudes,
        source_locations=torch.tensor([[[12, 12]]]),
        pml_freq=25.0,
        receiver_locations=torch.tensor([[[5, 5]]]),
    )
    restored = pickle.loads(pickle.dumps(out))  # noqa: S301 -- our own object
    assert type(restored) is type(out)
    assert restored.names == out.names
    assert torch.equal(restored.receiver_amplitudes, out.receiver_amplitudes)
    assert "deepwave" in repr(type(out))


def test_run_info_repr_and_fields_survive_the_move() -> None:
    """``RunInfo`` is still a frozen dataclass with its documented fields."""
    import dataclasses

    v = torch.full((25, 25), 1500.0)
    source_amplitudes = torch.zeros(1, 1, 15)
    source_amplitudes[0, 0, 3] = 1.0
    info = deepwave.scalar(
        v,
        4.0,
        0.001,
        source_amplitudes=source_amplitudes,
        source_locations=torch.tensor([[[12, 12]]]),
        pml_freq=25.0,
    ).info
    assert dataclasses.is_dataclass(info)
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.internal_dt = 1.0  # type: ignore[misc]
    assert info.location_interpolation_used in ("none", "hicks", "mixed")


_FUNCTIONALS = (
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


def test_get_type_hints_resolves_on_the_whole_public_api() -> None:
    """``typing.get_type_hints`` works on every public callable.

    ``LocationsLike`` names ``Hicks``, which ``deepwave.types`` imports only
    under ``TYPE_CHECKING`` to stay a leaf module, so the forward reference
    has to carry the module it belongs to. Without that, every functional
    and every ``Module.forward`` raises ``NameError: name 'Hicks' is not
    defined`` -- runtime introspection of the public surface is broken, and
    so is Sphinx's ``autodoc_typehints`` if it is ever switched on.
    """
    callables = []
    for name in _FUNCTIONALS:
        cls = {
            "scalar": "Scalar",
            "scalar_born": "ScalarBorn",
            "acoustic": "Acoustic",
            "acoustic_born": "AcousticBorn",
            "acoustic_vti": "AcousticVti",
            "viscoacoustic": "Viscoacoustic",
            "elastic": "Elastic",
            "elastic_born": "ElasticBorn",
            "elastic_vti": "ElasticVti",
        }[name]
        module = getattr(deepwave, cls)
        callables.append((name, getattr(deepwave, name)))
        callables.append((f"{cls}.forward", module.forward))
        callables.append((f"{cls}.__init__", module.__init__))

    def failure(label: str, func: object) -> str:
        try:
            typing.get_type_hints(func)
        except Exception as exc:  # noqa: BLE001 -- reported, not raised
            return f"{label}: {type(exc).__name__}: {exc}"
        return ""

    failures = [
        message for message in (failure(*item) for item in callables) if message
    ]
    assert not failures, "get_type_hints failed on: " + "; ".join(failures)


def test_locations_like_forward_reference_names_its_module() -> None:
    """The alias itself carries the module, not just the callables using it.

    This is the mechanism the test above depends on, pinned separately so a
    failure says *why* rather than only that introspection broke.
    """
    refs = [
        arg
        for arg in typing.get_args(deepwave.types.LocationsLike)
        if isinstance(arg, typing.ForwardRef)
    ]
    assert refs, "LocationsLike should still hold a forward reference"
    assert all(
        getattr(ref, "__forward_module__", None) == "deepwave._location_interpolation"
        for ref in refs
    )
