"""Compatibility shim for the ``deepwave.backend_utils`` spelling.

Deepwave 0.0.27 exposed this module at the package's top level. It was never
a designed interface -- it is the ``ctypes`` plumbing that finds the compiled
library and builds argument-type lists for its entry points -- but it *was*
importable, and Deepwave does not remove in 0.1.0 anything that 0.0.27
shipped. The implementation now lives in the private
``deepwave._backend_utils``, and nothing here is a supported API: the
templates, the loaded library handle and the resolution helpers all change
shape whenever a propagator or a build option does.

**Deprecated.** Every attribute access here emits a
:class:`deepwave.DeepwaveFutureWarning`
naming the private module. The shim is removed in Deepwave 0.4.0.
Importing :mod:`deepwave` does not warn -- only touching one of these names
does.

**What is covered.** Deepwave 0.0.27's module had no ``__all__`` and 33
names that do not start with an underscore. Fourteen of those were imported
modules (``ctypes``, ``pathlib``, ``platform``, ``torch``), ``typing``
imports (``Any``, ``Callable``, ``List``, ``TypeAlias``) and ``ctypes``
scalar types re-exported by the import that brought them in (``c_bool``,
``c_double``, ``c_float``, ``c_int64``, ``c_size_t``, ``c_void_p``); they
were incidental to the import statements, have their own canonical homes,
and are not carried. The nineteen this module defined itself are, and are
listed in ``__all__``, so ``from deepwave.backend_utils import *`` binds them
and warns about each -- an empty ``__all__`` would make that route bind
nothing, warn nothing, and fail on the caller's next line as an unrelated
``NameError``.

The same rule was applied to the ``deepwave.common`` and
``deepwave.location_interpolation`` shims.
"""

import importlib as _importlib
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

from deepwave._exceptions import (
    warn_deprecated_name as _warn_deprecated_name,
)

#: The names Deepwave 0.0.27's ``deepwave.backend_utils`` defined itself,
#: confirmed against a 0.0.27 install. Each resolves to the same attribute of
#: ``deepwave._backend_utils``, so the shim cannot drift from the private
#: module it forwards to; a name that stops existing there fails this
#: module's own test rather than silently returning something else.
_FORWARDED: tuple[str, ...] = (
    "CFunctionPointer",
    "FLOAT_TYPE",
    "SO_EXT",
    "USE_OPENMP",
    "current_accuracy",
    "current_dtype",
    "current_ndim",
    "dll",
    "get_acoustic_backward_template",
    "get_acoustic_forward_template",
    "get_backend_function",
    "get_elastic_backward_template",
    "get_elastic_forward_template",
    "get_scalar_backward_template",
    "get_scalar_born_backward_sc_template",
    "get_scalar_born_backward_template",
    "get_scalar_born_forward_template",
    "get_scalar_forward_template",
    "templates",
)

# Kept in step with ``_FORWARDED`` by a test. Written out rather than computed
# from it so that static tooling can read the exported surface without running
# the module.
__all__ = [
    "FLOAT_TYPE",
    "SO_EXT",
    "USE_OPENMP",
    "CFunctionPointer",
    "current_accuracy",
    "current_dtype",
    "current_ndim",
    "dll",
    "get_acoustic_backward_template",
    "get_acoustic_forward_template",
    "get_backend_function",
    "get_elastic_backward_template",
    "get_elastic_forward_template",
    "get_scalar_backward_template",
    "get_scalar_born_backward_sc_template",
    "get_scalar_born_backward_template",
    "get_scalar_born_forward_template",
    "get_scalar_forward_template",
    "templates",
]

if _TYPE_CHECKING:  # pragma: no cover -- lets mypy resolve the forwards
    from deepwave._backend_utils import (
        FLOAT_TYPE as FLOAT_TYPE,
    )
    from deepwave._backend_utils import (
        SO_EXT as SO_EXT,
    )
    from deepwave._backend_utils import (
        USE_OPENMP as USE_OPENMP,
    )
    from deepwave._backend_utils import (
        CFunctionPointer as CFunctionPointer,
    )
    from deepwave._backend_utils import (
        current_accuracy as current_accuracy,
    )
    from deepwave._backend_utils import (
        current_dtype as current_dtype,
    )
    from deepwave._backend_utils import (
        current_ndim as current_ndim,
    )
    from deepwave._backend_utils import (
        dll as dll,
    )
    from deepwave._backend_utils import (
        get_acoustic_backward_template as get_acoustic_backward_template,
    )
    from deepwave._backend_utils import (
        get_acoustic_forward_template as get_acoustic_forward_template,
    )
    from deepwave._backend_utils import (
        get_backend_function as get_backend_function,
    )
    from deepwave._backend_utils import (
        get_elastic_backward_template as get_elastic_backward_template,
    )
    from deepwave._backend_utils import (
        get_elastic_forward_template as get_elastic_forward_template,
    )
    from deepwave._backend_utils import (
        get_scalar_backward_template as get_scalar_backward_template,
    )
    from deepwave._backend_utils import (
        get_scalar_born_backward_sc_template as get_scalar_born_backward_sc_template,
    )
    from deepwave._backend_utils import (
        get_scalar_born_backward_template as get_scalar_born_backward_template,
    )
    from deepwave._backend_utils import (
        get_scalar_born_forward_template as get_scalar_born_forward_template,
    )
    from deepwave._backend_utils import (
        get_scalar_forward_template as get_scalar_forward_template,
    )
    from deepwave._backend_utils import (
        templates as templates,
    )


def __getattr__(name: str) -> _Any:
    """Resolves a ``deepwave.backend_utils`` name, with a warning.

    Looked up lazily (PEP 562) rather than imported eagerly, so that
    ``import deepwave`` does not warn. ``stacklevel=2`` attributes the
    warning to the line that touched the name, for a plain attribute access
    and for ``from deepwave.backend_utils import dll`` alike.
    """
    if name not in _FORWARDED:
        raise AttributeError(
            f"module 'deepwave.backend_utils' has no attribute {name!r}. It "
            "is a compatibility shim carrying the nineteen names that "
            "Deepwave 0.0.27's deepwave.backend_utils defined itself: "
            f"{', '.join(_FORWARDED)}. The other names 0.0.27 happened to "
            "expose here were imported modules, typing imports and ctypes "
            "scalar types, which have their own canonical homes. None of "
            "this is a supported interface; it is the compiled backend's "
            "internal plumbing, and it now lives in "
            "deepwave._backend_utils."
        )
    _warn_deprecated_name(
        f"deepwave.backend_utils.{name}",
        f"deepwave._backend_utils.{name}",
        stacklevel=2,
        detail=(
            "deepwave.backend_utils was never a supported interface -- it is "
            "the compiled backend's ctypes plumbing -- so the private "
            "module's contents may change in any release."
        ),
    )
    # Deliberately not cached: every access should warn. Resolved from the
    # private module on each access, so that a name whose value the engine
    # rebinds (``current_ndim`` and its neighbours) reads through rather than
    # freezing the value at first touch, as reading the 0.0.27 module global
    # did.
    module = _importlib.import_module("deepwave._backend_utils")
    return getattr(module, name)


def __dir__() -> list[str]:
    """The deprecated names, so that tab-completion still finds them."""
    return sorted(_FORWARDED)
