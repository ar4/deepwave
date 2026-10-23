"""Compatibility shim for the ``deepwave.common.*`` spelling.

Everything here has a canonical home elsewhere -- at the top level
(``deepwave.IGNORE_LOCATION``) or in :mod:`deepwave.discretisation`
(``deepwave.discretisation.upsample``). This module exists only so that code
written against Deepwave 0.0.27, which reached these helpers through
``deepwave.common``, keeps working.

**Deprecated.** Every attribute access here emits a
:class:`deepwave.DeepwaveFutureWarning`
naming the canonical spelling. The module is removed in Deepwave 0.4.0.
Importing :mod:`deepwave` does not warn -- only touching one of these names
does.

**What is covered.** Deepwave 0.0.27's ``deepwave.common`` had no ``__all__``
and held 68 names that do not start with an underscore, most of them private
helpers, imported modules and typing imports that happened to be visible. This
shim covers ten of them: the eight the 0.0.27 documentation referred to, the
``Callback`` type that goes with the documented ``CallbackState``, and the
``IGNORE_LOCATION`` sentinel, which 0.0.27 also exported from the package
root. The other 58 were never part of that interface; touching one here
raises an :exc:`AttributeError` that says so, and names a public home for the
few that have one.

``__all__`` lists exactly those ten, so ``from deepwave.common import *``
binds them -- and warns about each, because a star-import of a name that is
not already a module global goes through the same PEP 562 hook. The
documentation still points at the canonical homes, not here.

Four further names -- ``critical_courant_number``, ``to_cells``,
``estimate_memory`` and ``check_finite`` -- were added to this module during
development and never shipped from it, in 0.0.27 or any earlier release. They
have been removed rather than frozen as a second permanent spelling; use their
canonical homes.
"""

import importlib as _importlib
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

from deepwave._exceptions import (
    warn_deprecated_name as _warn_deprecated_name,
)

# name -> (module holding it, attribute name there, the spelling to suggest).
# The ten names of Deepwave 0.0.27's ``deepwave.common`` that were part of its
# documented interface, confirmed against a 0.0.27 install.
_ALIASES: dict[str, tuple[str, str, str]] = {
    "Callback": ("deepwave.types", "Callback", "deepwave.Callback"),
    "CallbackState": (
        "deepwave.types",
        "CallbackState",
        "deepwave.CallbackState",
    ),
    "IGNORE_LOCATION": (
        "deepwave._common",
        "IGNORE_LOCATION",
        "deepwave.IGNORE_LOCATION",
    ),
    "cfl_condition": (
        "deepwave.discretisation",
        "cfl_condition",
        "deepwave.discretisation.cfl_condition",
    ),
    "cfl_condition_n": (
        "deepwave.discretisation",
        "cfl_condition_n",
        "deepwave.discretisation.cfl_condition_n",
    ),
    "cosine_taper_end": (
        "deepwave.discretisation",
        "cosine_taper_end",
        "deepwave.discretisation.cosine_taper_end",
    ),
    "downsample": (
        "deepwave.discretisation",
        "downsample",
        "deepwave.discretisation.downsample",
    ),
    "lambmubuoyancy_to_vpvsrho": (
        "deepwave._common",
        "lambmubuoyancy_to_vpvsrho",
        "deepwave.lambmubuoyancy_to_vpvsrho",
    ),
    "upsample": (
        "deepwave.discretisation",
        "upsample",
        "deepwave.discretisation.upsample",
    ),
    "vpvsrho_to_lambmubuoyancy": (
        "deepwave._common",
        "vpvsrho_to_lambmubuoyancy",
        "deepwave.vpvsrho_to_lambmubuoyancy",
    ),
}

# Kept in step with ``_ALIASES`` by a test. Written out rather than computed so
# that static tooling can read the exported surface without running the module.
__all__ = [
    "IGNORE_LOCATION",
    "Callback",
    "CallbackState",
    "cfl_condition",
    "cfl_condition_n",
    "cosine_taper_end",
    "downsample",
    "lambmubuoyancy_to_vpvsrho",
    "upsample",
    "vpvsrho_to_lambmubuoyancy",
]

# Names that are *not* shimmed but do have a public home, so that the
# ``AttributeError`` can route the reader instead of leaving them stranded.
# ``StorageMode`` is deliberately worded as a replacement rather than an
# alias: 0.0.27's was an ``IntEnum``, the current one is a string ``Literal``,
# so it is not a drop-in and must not resolve silently.
_ELSEWHERE: dict[str, str] = {
    "StorageMode": (
        "deepwave.StorageMode, which is now a Literal of strings "
        "('device', 'cpu', 'disk', 'none') rather than an IntEnum"
    ),
    "check_finite": "deepwave.check_finite",
    "critical_courant_number": "deepwave.discretisation.critical_courant_number",
    "estimate_memory": "deepwave.estimate_memory",
    "to_cells": "deepwave.to_cells",
}

if _TYPE_CHECKING:  # pragma: no cover -- lets mypy resolve the aliases
    from deepwave._common import (
        IGNORE_LOCATION as IGNORE_LOCATION,
    )
    from deepwave._common import (
        lambmubuoyancy_to_vpvsrho as lambmubuoyancy_to_vpvsrho,
    )
    from deepwave._common import (
        vpvsrho_to_lambmubuoyancy as vpvsrho_to_lambmubuoyancy,
    )
    from deepwave.discretisation import (
        cfl_condition as cfl_condition,
    )
    from deepwave.discretisation import (
        cfl_condition_n as cfl_condition_n,
    )
    from deepwave.discretisation import (
        cosine_taper_end as cosine_taper_end,
    )
    from deepwave.discretisation import (
        downsample as downsample,
    )
    from deepwave.discretisation import (
        upsample as upsample,
    )
    from deepwave.types import (
        Callback as Callback,
    )
    from deepwave.types import (
        CallbackState as CallbackState,
    )


def _missing_message(name: str) -> str:
    """The ``AttributeError`` text for a name this shim does not carry."""
    message = (
        f"module 'deepwave.common' has no attribute {name!r}. "
        "deepwave.common is a compatibility shim carrying the ten names of "
        "Deepwave 0.0.27's deepwave.common that were part of its documented "
        f"interface: {', '.join(sorted(_ALIASES))}. "
    )
    elsewhere = _ELSEWHERE.get(name)
    if elsewhere is not None:
        return message + f"Use {elsewhere}."
    return message + (
        "Deepwave 0.0.27's deepwave.common had no __all__, and the other 58 "
        "names it happened to expose -- private helpers, imported modules and "
        "typing imports -- were never part of that interface and have no "
        f"public replacement, so {name!r} is not covered."
    )


def __getattr__(name: str) -> _Any:
    """Resolves a ``deepwave.common`` name, with a deprecation warning.

    Looked up lazily (PEP 562) rather than imported eagerly, so that
    ``import deepwave`` -- which imports this module -- does not warn.

    ``stacklevel=2`` attributes the warning to the line that touched the
    name, for a plain attribute access (``deepwave.common.upsample``) and
    for ``from deepwave.common import upsample`` alike: the ``IMPORT_FROM``
    bytecode performs the attribute lookup in the importing frame itself,
    so this hook's caller is user code in both cases (verified, not
    assumed), and the warning is therefore correctly placed.

    ``from deepwave.common import *`` also arrives here, once per name in
    ``__all__``, because none of them is a module global; the warnings are
    attributed to the star-import's own line.
    """
    entry = _ALIASES.get(name)
    if entry is None:
        raise AttributeError(_missing_message(name))
    module_name, attribute, canonical = entry
    _warn_deprecated_name(f"deepwave.common.{name}", canonical, stacklevel=2)
    # Deliberately not cached in the module's globals: caching would make the
    # warning fire only on the first access, and the point is to tell every
    # caller.
    return getattr(_importlib.import_module(module_name), attribute)


def __dir__() -> list[str]:
    """The deprecated aliases, so that tab-completion still finds them."""
    return sorted(_ALIASES)
