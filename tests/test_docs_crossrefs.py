"""Every ``:func:``/``:class:`` target in a public docstring must resolve.

``sphinx-build -W`` does not fail on an unresolved Python cross-reference by
default, so a docstring can point at a private helper -- or at a name that is
a *function* being addressed as though it were a module -- and the published
page silently renders it as plain text. That is how the dead references this
test first found accumulated. This test is the gate that keeps them fixed.

Scoped to ``src/deepwave/*.py`` deliberately: it resolves dotted names
against the live ``deepwave`` package, which is exactly the check that
catches the "module that is really a function" case, and which nitpicky mode
does not.

Resolution is an *attribute walk*: import only the top-level ``deepwave``
package and ``getattr`` down the dotted chain. Importing the longest
importable dotted prefix instead (which an earlier version of this file did)
is wrong for the very case the file is named after: ``deepwave.acoustic``
the attribute is the propagator *function*, so
``importlib.import_module("deepwave.acoustic")`` hands back a submodule that
no reader can reach from the package, and every target beneath it appears to
resolve.
"""

import importlib
import pathlib
import re
from typing import Any

import pytest

import deepwave

#: A Sphinx Python cross-reference role and its target, e.g.
#: ``:func:`deepwave.scalar``` or ``:class:`~deepwave.Hicks```.
_ROLE = re.compile(r":(func|class|meth|attr|data|mod|exc):`~?([^`<>]+)`")

#: Targets that are deliberately not Deepwave names.
_EXTERNAL_PREFIXES = ("torch", "numpy", "np.", "scipy", "matplotlib")


def _source_files() -> list[pathlib.Path]:
    """Every module of the installed ``deepwave`` package."""
    root = pathlib.Path(deepwave.__file__).parent
    return sorted(root.glob("*.py"))


def _resolves(target: str) -> bool:
    """Whether a reader can reach ``target`` by attribute access.

    Only the top-level package is imported; every remaining component must
    be an attribute of the object before it. That is exactly what a reader
    following the documented name can do, and -- unlike importing the
    longest importable prefix -- it does not silently see through a
    submodule that a same-named public function shadows.
    """
    parts = target.split(".")
    if parts[0] != "deepwave":
        return False
    obj: Any = deepwave
    for attribute in parts[1:]:
        if not hasattr(obj, attribute):
            return False
        obj = getattr(obj, attribute)
    return True


def _targets() -> list[tuple[str, int, str]]:
    """``(file, line, target)`` for every Deepwave cross-reference found."""
    found = []
    for path in _source_files():
        text = path.read_text()
        for match in _ROLE.finditer(text):
            raw = match.group(2)
            # Sphinx normalises whitespace inside a role, so a target split
            # across source lines (with or without a backslash continuation)
            # is one name; normalise it the same way before resolving.
            target = " ".join(raw.replace("\\", " ").split()).replace(" ", "")
            if target.startswith(_EXTERNAL_PREFIXES):
                continue
            if not target.startswith("deepwave"):
                # A bare name resolved relative to the current module by
                # Sphinx; too ambiguous to check here.
                continue
            line = text.count("\n", 0, match.start()) + 1
            found.append((path.name, line, target))
    return found


def test_public_docstring_crossrefs_resolve() -> None:
    """No docstring points at a name that does not exist."""
    broken = [entry for entry in _targets() if not _resolves(entry[2])]
    listing = "\n".join(
        f"  {name}:{line}: {target}" for name, line, target in sorted(broken)
    )
    assert not broken, (
        f"{len(broken)} cross-references ({len({t for _, _, t in broken})} "
        "distinct targets) are unreachable by attribute access from the "
        "``deepwave`` package, so they render as plain text rather than "
        f"links:\n{listing}"
    )


def test_resolver_walks_attributes_rather_than_importing_submodules() -> None:
    """The resolver must not see through a shadowed submodule.

    Every propagator name is bound at the package root to a *function*, and
    that binding shadows the same-named submodule. So
    ``deepwave.elastic.elastic`` is importable but unreachable: no reader
    can get there from ``deepwave`` by attribute access. A resolver that
    imports the longest importable dotted prefix accepts such a target,
    which is what made the cross-reference test above unable to fail. This
    pins the difference between the two algorithms rather than re-checking
    the docstrings, which the test above already covers.
    """
    shadowed = []
    for name in deepwave._engine.PROPAGATOR_NAMES:
        module = importlib.import_module(f"deepwave.{name}")
        if getattr(deepwave, name, None) is not module:
            shadowed.append((name, module))
    assert shadowed, "expected propagator functions to shadow their submodules"
    for name, module in shadowed:
        # The propagator function is defined in the submodule it shadows, so
        # this target always exists for an import-based resolver to find.
        assert callable(getattr(module, name)), f"deepwave.{name}.{name}"
        assert not _resolves(f"deepwave.{name}.{name}"), (
            f"deepwave.{name}.{name} is importable but unreachable by "
            "attribute access; the resolver must reject it"
        )


@pytest.mark.parametrize(
    "target",
    [
        "deepwave.spec",
        "deepwave.wavefield_names",
        "deepwave.receiver_names",
        "deepwave.PropagatorSpec",
        "deepwave.LocationsLike",
        "deepwave.SourceSpec",
        "deepwave.ReceiverSpec",
        "deepwave.wavefields.merge_point_sources",
    ],
    ids=lambda t: t.rsplit(".", 1)[-1],
)
def test_new_public_symbols_resolve(target: str) -> None:
    """Every newly published symbol is reachable by its documented name."""
    assert _resolves(target), target
