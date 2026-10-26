"""Compatibility shim for the ``deepwave.location_interpolation`` spelling.

Deepwave 0.0.27 exposed this module, and ``Hicks`` and ``DEFAULT_EPS``
through it. The canonical spellings are now :class:`deepwave.Hicks` and
:data:`deepwave.ON_NODE_TOLERANCE`; the implementation lives in the private
``deepwave._location_interpolation``, which is private because
``location_interpolation`` is also the name of the propagators' own keyword
argument (a string mode) and the two must not be confused.

**Deprecated.** Every attribute access here emits a
:class:`deepwave.DeepwaveFutureWarning`
naming the canonical spelling. The module is removed in Deepwave 0.4.0.
Importing :mod:`deepwave` does not warn -- only touching one of these names
does.

**What is covered.** Deepwave 0.0.27's module had no ``__all__`` and held ten
names that do not start with an underscore, but eight of those were typing
imports and imported modules (``torch``, ``math``, ``deepwave``) rather than
an interface. This shim carries the two that were: ``Hicks`` and
``DEFAULT_EPS``.

``__all__`` lists exactly those two, so ``from deepwave.location_interpolation
import *`` binds them -- and warns about each, because a star-import of a name
that is not already a module global goes through the same PEP 562 hook. An
empty ``__all__`` would make that one route bind nothing and warn nothing, and
the caller's next line would fail as an unrelated ``NameError``. The
documentation still points at the canonical homes, not here.
"""

import importlib as _importlib
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

from deepwave._exceptions import (
    warn_deprecated_name as _warn_deprecated_name,
)

# name -> (attribute in deepwave._location_interpolation, canonical spelling).
# The two names of Deepwave 0.0.27's ``deepwave.location_interpolation`` that
# were part of its interface, confirmed against a 0.0.27 install.
_ALIASES: dict[str, tuple[str, str]] = {
    "Hicks": ("Hicks", "deepwave.Hicks"),
    "DEFAULT_EPS": ("_DEFAULT_EPS", "deepwave.ON_NODE_TOLERANCE"),
}

# Kept in step with ``_ALIASES`` by a test. Written out rather than computed so
# that static tooling can read the exported surface without running the module.
__all__ = ["DEFAULT_EPS", "Hicks"]

if _TYPE_CHECKING:  # pragma: no cover -- lets mypy resolve the aliases
    from deepwave._location_interpolation import (
        Hicks as Hicks,
    )

    # Declared rather than imported: the private spelling is ``_DEFAULT_EPS``,
    # and an ``import ... as`` that renames is not a re-export.
    DEFAULT_EPS: float


def __getattr__(name: str) -> _Any:
    """Resolves a ``deepwave.location_interpolation`` name, with a warning.

    Looked up lazily (PEP 562) rather than imported eagerly, so that
    ``import deepwave`` does not warn. ``stacklevel=2`` attributes the
    warning to the line that touched the name, for a plain attribute access
    and for ``from deepwave.location_interpolation import Hicks`` alike.

    ``from deepwave.location_interpolation import *`` also arrives here, once
    per name in ``__all__``, because neither is a module global; the warnings
    are attributed to the star-import's own line.
    """
    entry = _ALIASES.get(name)
    if entry is None:
        raise AttributeError(
            f"module 'deepwave.location_interpolation' has no attribute "
            f"{name!r}. It is a compatibility shim carrying the two names of "
            "Deepwave 0.0.27's deepwave.location_interpolation that were part "
            "of its interface: DEFAULT_EPS, Hicks. Use deepwave.Hicks and "
            "deepwave.ON_NODE_TOLERANCE. The other names 0.0.27 happened to "
            "expose here were typing imports and imported modules, and have "
            "no public replacement."
        )
    attribute, canonical = entry
    _warn_deprecated_name(
        f"deepwave.location_interpolation.{name}", canonical, stacklevel=2
    )
    # Deliberately not cached: every access should warn.
    module = _importlib.import_module("deepwave._location_interpolation")
    return getattr(module, attribute)


def __dir__() -> list[str]:
    """The deprecated aliases, so that tab-completion still finds them."""
    return sorted(_ALIASES)
