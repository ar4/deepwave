"""Every error Deepwave raises deliberately is a ``DeepwaveError``.

``deepwave.DeepwaveError`` is only worth catching if it really does cover
everything Deepwave decides to reject. That promise cannot be kept by review:
a single ``raise ValueError(...)`` added to a validation helper is invisible
in a diff and silently punches a hole in it. So it is checked mechanically
here, by walking the AST of every module in ``src/deepwave`` and failing on
any ``raise`` of a builtin exception type that is not in the allowlist below.

The allowlist holds the builtins that are deliberately *not* Deepwave errors,
because raising them is Python protocol rather than a message to the caller:
``AttributeError`` is what a ``__getattr__`` must raise for an unknown name
and ``KeyError`` what a mapping lookup must raise for an absent key, while
``AssertionError`` marks an internal invariant that no caller can violate --
reaching one is a bug in Deepwave, not a request to reject. Everything else
that Deepwave raises on purpose uses one of the ``Deepwave*Error`` classes,
each of which inherits both :class:`deepwave.DeepwaveError` and the builtin
that site raised historically, so ``except ValueError`` keeps catching
exactly what it caught before.

The promise is deliberately about errors Deepwave *raises deliberately*. A
``TypeError`` from Python's own argument binding (a missing or misspelled
keyword, raised before any Deepwave code runs) and any exception from torch
are outside it, and the last two tests here pin that boundary so the
documentation cannot overstate it.
"""

import ast
import builtins
import pathlib
from typing import Optional

import pytest
import torch

import deepwave

_SRC = pathlib.Path(deepwave.__file__).parent

#: Builtin exception types that ``src/deepwave`` may raise as themselves,
#: with the reason each one is not a ``DeepwaveError``. Every other builtin
#: exception must be raised through its ``Deepwave*`` counterpart. Adding a
#: row here is a deliberate, reviewable act; forgetting to convert a raise is
#: not, which is the asymmetry this table exists to create.
_ALLOWED_BUILTIN_RAISES = {
    "AssertionError": (
        "an internal invariant, unreachable through the public interface; "
        "reaching one is a bug in Deepwave rather than a rejected request"
    ),
    "AttributeError": (
        "the exception a ``__getattr__`` (module or object) is required to "
        "raise for an unknown name, so that ``hasattr`` and ``getattr`` with "
        "a default keep working"
    ),
    "KeyError": (
        "the exception a mapping lookup is required to raise for an absent "
        "key, so that ``dict``-like access and ``.get`` keep working"
    ),
}

#: The builtin exception names, resolved once.
_BUILTIN_EXCEPTIONS = frozenset(
    name
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type)
    and issubclass(getattr(builtins, name), BaseException)
)


def _modules() -> list[pathlib.Path]:
    """Every Python module of the installed ``deepwave`` package."""
    paths = sorted(_SRC.glob("*.py"))
    assert paths, f"no Python sources found under {_SRC}"
    return paths


def _raised_name(node: ast.Raise) -> Optional[str]:
    """The identifier a ``raise`` statement names, if it names a plain one."""
    if node.exc is None:
        return None  # a bare ``raise`` re-raises; it introduces no new type.
    target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
    return target.id if isinstance(target, ast.Name) else None


def test_no_unconverted_builtin_raises() -> None:
    """No module raises a builtin exception outside the allowlist."""
    offenders = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise):
                continue
            name = _raised_name(node)
            if name is None or name not in _BUILTIN_EXCEPTIONS:
                continue
            if name in _ALLOWED_BUILTIN_RAISES:
                continue
            offenders.append(f"{path.name}:{node.lineno}: raise {name}")
    assert not offenders, (
        "every error Deepwave raises deliberately must be a DeepwaveError: "
        "replace each of these with the matching class from "
        "deepwave._exceptions (DeepwaveValueError for ValueError, and so on), "
        "or, if the builtin really is required by a Python protocol, add it "
        "to _ALLOWED_BUILTIN_RAISES with the reason:\n" + "\n".join(offenders)
    )


def test_allowlist_holds_only_builtin_exceptions() -> None:
    """The allowlist cannot drift into naming something that is not one."""
    for name in _ALLOWED_BUILTIN_RAISES:
        assert name in _BUILTIN_EXCEPTIONS, f"{name} is not a builtin exception"


def test_allowlist_has_no_dead_rows() -> None:
    """Every allowlisted builtin is still raised somewhere, with a reason.

    A row that nothing uses any more is a hole held open for no reason, so it
    is removed rather than left to excuse a future raise nobody looked at.
    """
    raised = set()
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise):
                name = _raised_name(node)
                if name is not None:
                    raised.add(name)
    unused = sorted(set(_ALLOWED_BUILTIN_RAISES) - raised)
    assert not unused, f"remove these unused allowlist rows: {unused}"
    for name, reason in _ALLOWED_BUILTIN_RAISES.items():
        assert reason.strip(), f"{name} has no recorded reason"


@pytest.mark.parametrize(
    ("deepwave_name", "builtin"),
    [
        ("DeepwaveValueError", ValueError),
        ("DeepwaveTypeError", TypeError),
        ("DeepwaveRuntimeError", RuntimeError),
        ("DeepwaveNotImplementedError", NotImplementedError),
        ("DeepwaveOSError", OSError),
        ("DeepwaveFloatingPointError", FloatingPointError),
    ],
)
def test_each_class_inherits_both(deepwave_name: str, builtin: type) -> None:
    """Each class is both a ``DeepwaveError`` and its historical builtin."""
    cls = getattr(deepwave, deepwave_name)
    assert issubclass(cls, deepwave.DeepwaveError)
    assert issubclass(cls, builtin)
    assert cls.__module__ == "deepwave"


def test_a_rejected_call_raises_a_deepwave_error() -> None:
    """A real validation failure is catchable as both spellings."""
    v = torch.ones(10, 10)
    with pytest.raises(deepwave.DeepwaveError):
        deepwave.scalar(
            v,
            4.0,
            0.001,
            source_amplitudes=torch.zeros(1, 1, 5),
            source_locations=torch.tensor([[[100, 3]]]),
            pml_freq=25.0,
        )
    with pytest.raises(RuntimeError):
        deepwave.scalar(
            v,
            4.0,
            0.001,
            source_amplitudes=torch.zeros(1, 1, 5),
            source_locations=torch.tensor([[[100, 3]]]),
            pml_freq=25.0,
        )


def test_python_argument_binding_is_not_a_deepwave_error() -> None:
    """A ``TypeError`` from Python's own call binding is outside the promise.

    ``deepwave.scalar(**{"grid_spacing": ...})`` with the positional argument
    missing never reaches Deepwave's body, so nothing in Deepwave can classify
    it. The documentation says so; this is what it means.
    """
    with pytest.raises(TypeError) as excinfo:
        deepwave.scalar(torch.ones(10, 10))  # type: ignore[call-arg]
    assert not isinstance(excinfo.value, deepwave.DeepwaveError)

    with pytest.raises(TypeError) as excinfo:
        deepwave.scalar(
            torch.ones(10, 10),
            4.0,
            0.001,
            not_an_argument=1,
        )
    assert not isinstance(excinfo.value, deepwave.DeepwaveError)


def test_a_torch_error_is_not_a_deepwave_error() -> None:
    """An exception raised by torch itself stays torch's."""
    with pytest.raises(RuntimeError) as excinfo:
        torch.ones(3, 4) @ torch.ones(3, 4)
    assert not isinstance(excinfo.value, deepwave.DeepwaveError)
