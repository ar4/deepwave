"""Compatibility shim for the ``deepwave.regular_grid`` spelling.

Deepwave 0.0.27 exposed this module at the package's top level. It holds
the regular-grid finite-difference stencils and the PML profile builder
that the ``scalar`` family's pure-PyTorch time loop uses -- internal
machinery, not a designed interface, and its contents track whatever the
eager backend currently needs. It *was*
importable, though, and Deepwave 0.1.0 removes nothing that 0.0.27 shipped,
so the spelling is kept until 0.4.0 over the private
``deepwave._regular_grid``.

**Deprecated.** Every attribute access here emits a
:class:`deepwave.DeepwaveFutureWarning`
naming the private module. The shim is removed in Deepwave 0.4.0.
Importing :mod:`deepwave` does not warn -- only touching one of these names
does.

**What is covered.** Deepwave 0.0.27's module had no ``__all__`` and
15 names that do not start with an underscore. The nine it
defined itself are carried, and are listed in ``__all__`` so that
``from deepwave.regular_grid import *`` binds them and warns about each; the rest
were ``typing`` imports and imported modules that its import statements
happened to re-export, and have their own canonical homes. Every carried
name has the same signature it had in 0.0.27.
"""

import importlib as _importlib
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

from deepwave._exceptions import (
    warn_deprecated_name as _warn_deprecated_name,
)

#: The names Deepwave 0.0.27's ``deepwave.regular_grid`` defined itself, confirmed
#: against a 0.0.27 install. Each resolves to the same attribute of
#: ``deepwave._regular_grid``, so the shim cannot drift from the private module
#: it forwards to.
_FORWARDED: tuple[str, ...] = (
    "diff1",
    "diff2",
    "diffx1",
    "diffx2",
    "diffy1",
    "diffy2",
    "diffz1",
    "diffz2",
    "set_pml_profiles",
)

# Kept in step with ``_FORWARDED`` by a test. Written out rather than computed
# from it so that static tooling can read the exported surface without running
# the module.
__all__ = [
    "diff1",
    "diff2",
    "diffx1",
    "diffx2",
    "diffy1",
    "diffy2",
    "diffz1",
    "diffz2",
    "set_pml_profiles",
]

if _TYPE_CHECKING:  # pragma: no cover -- lets mypy resolve the forwards
    from deepwave._regular_grid import (
        diff1 as diff1,
    )
    from deepwave._regular_grid import (
        diff2 as diff2,
    )
    from deepwave._regular_grid import (
        diffx1 as diffx1,
    )
    from deepwave._regular_grid import (
        diffx2 as diffx2,
    )
    from deepwave._regular_grid import (
        diffy1 as diffy1,
    )
    from deepwave._regular_grid import (
        diffy2 as diffy2,
    )
    from deepwave._regular_grid import (
        diffz1 as diffz1,
    )
    from deepwave._regular_grid import (
        diffz2 as diffz2,
    )
    from deepwave._regular_grid import (
        set_pml_profiles as set_pml_profiles,
    )


def __getattr__(name: str) -> _Any:
    """Resolves a ``deepwave.regular_grid`` name, with a warning.

    Looked up lazily (PEP 562) rather than imported eagerly, so that
    ``import deepwave`` does not warn. ``stacklevel=2`` attributes the
    warning to the line that touched the name, for a plain attribute access
    and for ``from deepwave.regular_grid import diff1`` alike.
    """
    if name not in _FORWARDED:
        raise AttributeError(
            f"module 'deepwave.regular_grid' has no attribute {name!r}. It is a "
            "compatibility shim carrying the nine names that Deepwave "
            "0.0.27's deepwave.regular_grid defined itself: "
            f"{', '.join(_FORWARDED)}. The other names 0.0.27 happened to "
            "expose here were typing imports and imported modules, which "
            "have their own canonical homes. None of this is a supported "
            "interface; it is the eager backend's internal machinery, and it "
            "now lives in deepwave._regular_grid."
        )
    _warn_deprecated_name(
        f"deepwave.regular_grid.{name}",
        f"deepwave._regular_grid.{name}",
        stacklevel=2,
        detail=(
            "deepwave.regular_grid was never a supported interface -- it is the "
            "eager backend's finite-difference machinery -- so the private "
            "module's contents may change in any release."
        ),
    )
    # Deliberately not cached: every access should warn.
    module = _importlib.import_module("deepwave._regular_grid")
    return getattr(module, name)


def __dir__() -> list[str]:
    """The deprecated names, so that tab-completion still finds them."""
    return sorted(_FORWARDED)
