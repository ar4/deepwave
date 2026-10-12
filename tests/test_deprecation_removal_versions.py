"""The deprecation table: one row per deprecated name, checked both ways.

Deepwave 0.1.0 removes nothing that ``v0.0.27`` shipped. Every old spelling
survives as a deprecated alias that computes exactly what it always did, warns
:class:`deepwave.DeepwaveFutureWarning` once per call per name, and names both
its replacement and ``0.4.0``, the earliest release that may remove it.
Later stable deprecations must have their own replacement, warning and
migration example, and keep the old form for two subsequent minor releases.

``_DEPRECATIONS`` is that table. Each row carries

* a trigger that uses the deprecated spelling,
* a trigger that uses the replacement, which must compute the same thing and
  warn about nothing,
* the replacement name the message has to contain, and
* either a snippet that proves the name worked in ``v0.0.27`` (the ``shipped``
  rows) or the row it was extended from (the ``extends`` rows).

Four groups of guard run over it:

1. **Every row warns, once, in the right words** -- one
   ``DeepwaveFutureWarning`` naming the replacement, ``0.4.0`` and the category
   to filter on; filterable by that category; and the replacement spelling
   gives the identical result and no warning at all.
2. **The source cannot drift from the table.** Every ``warnings.warn`` in
   ``src/deepwave`` that uses a deprecation category is required to be the one
   shared emitter, so a new deprecation cannot appear with its own wording, its
   own category, or no removal version; and no shipped text may say "no earlier
   than" about a removal, because the version is a date rather than a bound.
3. **Audibility under Python's own default filters**, in a subprocess with no
   ``-W`` and no ``PYTHONWARNINGS``. ``pytest.warns`` forces ``always``, so it
   cannot see this class of bug at all: a ``DeprecationWarning`` attributed to
   ``torch`` passes every in-process test and then emits nothing for a real
   user.
4. **The ``shipped`` rows really did ship.** Each is run under a real
   ``v0.0.27`` interpreter, named by ``DEEPWAVE_V0027_PYTHON``; the group skips
   when that variable is unset, because no such interpreter exists in CI. The
   ``extends`` rows cannot be checked that way -- the propagators they are on
   were never released -- so each is checked against the released row it was
   extended from instead.
"""

import ast
import os
import pathlib
import subprocess
import sys
import textwrap
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple, Optional

import pytest
import torch

import deepwave
import deepwave._engine
from deepwave._exceptions import DEPRECATION_REMOVAL_VERSION

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "deepwave"

#: The environment variable naming a ``v0.0.27`` interpreter. Set it to the
#: ``python`` of an environment with ``deepwave==0.0.27`` installed to run the
#: fourth group of guards; unset, they skip.
_V0027_ENV = "DEEPWAVE_V0027_PYTHON"


def _v() -> torch.Tensor:
    return 1500 * torch.ones(8, 8)


def _sources() -> dict[str, Any]:
    return {
        "source_amplitudes": torch.ones(1, 1, 4),
        "source_locations": torch.tensor([[[4, 4]]]),
        "pml_freq": 25.0,
    }


# ---------------------------------------------------------------------------
# The table. Two triggers per row: the deprecated spelling and its
# replacement. Both return a value, so that the pair can be compared.
# ---------------------------------------------------------------------------


def _scalar_born(**kwargs: Any) -> Any:
    return deepwave.scalar_born(_v(), 4.0, 0.002, **_sources(), **kwargs)


def _old_scatter_kwarg() -> Any:
    return _scalar_born(
        scatter=torch.full((8, 8), 3.0),
        sc_receiver_locations=torch.tensor([[[4, 5]]]),
    ).sc_receiver_amplitudes


def _new_dv_kwarg() -> Any:
    return _scalar_born(
        dv=torch.full((8, 8), 3.0),
        sc_receiver_locations=torch.tensor([[[4, 5]]]),
    ).sc_receiver_amplitudes


def _old_module_scatter() -> Any:
    return deepwave.ScalarBorn(_v(), 4.0, scatter=torch.full((8, 8), 3.0)).dv


def _new_module_dv() -> Any:
    return deepwave.ScalarBorn(_v(), 4.0, dv=torch.full((8, 8), 3.0)).dv


def _old_module_scatter_requires_grad() -> Any:
    module = deepwave.ScalarBorn(
        _v(),
        4.0,
        scatter=torch.full((8, 8), 3.0),
        scatter_requires_grad=True,
    )
    return module.dv.requires_grad


def _new_module_dv_requires_grad() -> Any:
    module = deepwave.ScalarBorn(
        _v(), 4.0, dv=torch.full((8, 8), 3.0), dv_requires_grad=True
    )
    return module.dv.requires_grad


def _quietly(build: Callable[[], Any]) -> Any:
    """Builds something whose construction warns, without the warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build()


def _scatter_property() -> Any:
    module = _quietly(
        lambda: deepwave.ScalarBorn(_v(), 4.0, dv=torch.full((8, 8), 3.0))
    )
    return module.scatter


def _new_dv_property() -> Any:
    module = _quietly(
        lambda: deepwave.ScalarBorn(_v(), 4.0, dv=torch.full((8, 8), 3.0))
    )
    return module.dv


def _old_state_dict_key() -> Any:
    module = _quietly(lambda: deepwave.ScalarBorn(_v(), 4.0, dv=torch.zeros(8, 8)))
    module.load_state_dict({"v": _v(), "scatter": torch.full((8, 8), 3.0)})
    return module.dv


def _new_state_dict_key() -> Any:
    module = _quietly(lambda: deepwave.ScalarBorn(_v(), 4.0, dv=torch.zeros(8, 8)))
    module.load_state_dict({"v": _v(), "dv": torch.full((8, 8), 3.0)})
    return module.dv


def _old_ricker_length() -> Any:
    return deepwave.wavelets.ricker(25.0, length=100, dt=0.002)


def _new_ricker_nt() -> Any:
    return deepwave.wavelets.ricker(25.0, nt=100, dt=0.002)


def _old_python_backend() -> Any:
    return deepwave.scalar(
        _v(), 4.0, 0.002, **_sources(), python_backend=False
    ).receiver_amplitudes


def _new_backend() -> Any:
    return deepwave.scalar(
        _v(), 4.0, 0.002, **_sources(), backend="auto"
    ).receiver_amplitudes


def _viscoacoustic(**kwargs: Any) -> Any:
    return deepwave.viscoacoustic(
        _v(),
        torch.full((8, 8), 1000.0),
        torch.full((8, 8), 50.0),
        4.0,
        0.002,
        source_amplitudes_pressure_rate=(1000.0 * 1500.0**2) * (torch.ones(1, 1, 4)),
        source_locations_pressure_rate=torch.tensor([[[4, 4]]]),
        receiver_locations_p=torch.tensor([[[4, 5]]]),
        pml_freq=25.0,
        q_freq=25.0,
        **kwargs,
    ).receiver_amplitudes_p


def _acoustic(**kwargs: Any) -> Any:
    return deepwave.acoustic(
        _v(),
        torch.full((8, 8), 1000.0),
        4.0,
        0.002,
        pml_freq=25.0,
        receiver_locations_p=torch.tensor([[[4, 5]]]),
        **kwargs,
    ).receiver_amplitudes_p


def _old_source_p_on_acoustic() -> Any:
    return _acoustic(
        source_amplitudes_p=torch.ones(1, 1, 4),
        source_locations_p=torch.tensor([[[4, 4]]]),
    )


def _new_source_volume_rate_on_acoustic() -> Any:
    return _acoustic(
        source_amplitudes_volume_rate=torch.ones(1, 1, 4),
        source_locations_volume_rate=torch.tensor([[[4, 4]]]),
    )


def _elastic_models() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lamb = torch.full((8, 8), 2.2e9)
    mu = torch.full((8, 8), 1.1e9)
    buoyancy = torch.full((8, 8), 1 / 2200.0)
    return lamb, mu, buoyancy


def _elastic(**kwargs: Any) -> Any:
    lamb, mu, buoyancy = _elastic_models()
    return deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        4.0,
        0.002,
        pml_freq=25.0,
        receiver_locations_p=torch.tensor([[[4, 5]]]),
        **kwargs,
    ).receiver_amplitudes_p


def _old_source_p_on_elastic() -> Any:
    return _elastic(
        source_amplitudes_p=torch.ones(1, 1, 4),
        source_locations_p=torch.tensor([[[4, 4]]]),
    )


def _new_source_pressure_rate_on_elastic() -> Any:
    return _elastic(
        source_amplitudes_pressure_rate=torch.ones(1, 1, 4),
        source_locations_pressure_rate=torch.tensor([[[4, 4]]]),
    )


def _old_bare_receiver_locations() -> Any:
    return _scalar_born(
        dv=torch.full((8, 8), 3.0), receiver_locations=torch.tensor([[[4, 5]]])
    ).sc_receiver_amplitudes


def _new_sc_receiver_locations() -> Any:
    return _scalar_born(
        dv=torch.full((8, 8), 3.0),
        sc_receiver_locations=torch.tensor([[[4, 5]]]),
    ).sc_receiver_amplitudes


def _old_callback_name_scatter() -> Any:
    return deepwave.CallbackState._lookup(
        {"dv": torch.full((1,), 3.0)}, "scatter", "model"
    )


def _new_callback_name_dv() -> Any:
    return deepwave.CallbackState._lookup({"dv": torch.full((1,), 3.0)}, "dv", "model")


def _old_callback_name_scatter_gradient() -> Any:
    return deepwave.CallbackState._lookup(
        {"dv": torch.full((1,), 3.0)}, "scatter", "gradient"
    )


def _new_callback_name_dv_gradient() -> Any:
    return deepwave.CallbackState._lookup(
        {"dv": torch.full((1,), 3.0)}, "dv", "gradient"
    )


def _shim(module: str, name: str) -> Callable[[], Any]:
    """A trigger that touches ``name`` on the shim module ``module``."""

    def trigger() -> Any:
        return getattr(getattr(deepwave, module), name)

    return trigger


def _private(module: str, name: str) -> Callable[[], Any]:
    """The same name on the private module the shim forwards to."""

    def trigger() -> Any:
        return getattr(__import__(f"deepwave._{module}", fromlist=[name]), name)

    return trigger


def _canonical_upsample() -> Any:
    return deepwave.discretisation.upsample


def _canonical_hicks() -> Any:
    return deepwave.Hicks


class Row(NamedTuple):
    """One deprecated name, and everything the guards need about it."""

    #: The call that uses the deprecated spelling, and returns its result.
    old: Callable[[], Any]
    #: The same call written with the replacement, which must not warn.
    new: Callable[[], Any]
    #: The replacement spelling the warning's text must name.
    replacement: str
    #: A ``v0.0.27`` snippet proving the name shipped, or ``None`` for a row
    #: that extends another to a propagator ``v0.0.27`` did not have
    #: (``extends`` then names that row).
    shipped: Optional[str] = None
    #: The label of the released row this one extends.
    extends: Optional[str] = None


#: Every deprecated name in Deepwave 0.1.0. Closed after 0.1.0: nothing new is
#: deprecated in 0.1.0. Later deprecations require new, reviewed rows.
_DEPRECATIONS: dict[str, Row] = {
    "scalar_born(scatter=)": Row(
        _old_scatter_kwarg,
        _new_dv_kwarg,
        "dv",
        shipped="""
        import torch, deepwave
        deepwave.scalar_born(
            1500 * torch.ones(8, 8), torch.full((8, 8), 3.0), 4.0, dt=0.002,
            source_amplitudes=torch.ones(1, 1, 4),
            source_locations=torch.tensor([[[4, 4]]]),
        )
        """,
    ),
    "ScalarBorn(scatter=)": Row(
        _old_module_scatter,
        _new_module_dv,
        "dv",
        shipped="""
        import torch, deepwave
        m = deepwave.ScalarBorn(1500 * torch.ones(8, 8), torch.full((8, 8), 3.0), 4.0)
        assert m.scatter.shape == (8, 8)
        """,
    ),
    "ScalarBorn(scatter_requires_grad=)": Row(
        _old_module_scatter_requires_grad,
        _new_module_dv_requires_grad,
        "dv_requires_grad",
        shipped="""
        import torch, deepwave
        m = deepwave.ScalarBorn(
            1500 * torch.ones(8, 8), torch.full((8, 8), 3.0), 4.0,
            scatter_requires_grad=True,
        )
        assert m.scatter.requires_grad
        """,
    ),
    "ScalarBorn.scatter": Row(
        _scatter_property,
        _new_dv_property,
        "dv",
        shipped="""
        import torch, deepwave
        m = deepwave.ScalarBorn(1500 * torch.ones(8, 8), torch.full((8, 8), 3.0), 4.0)
        assert m.scatter is not None
        """,
    ),
    "ScalarBorn state_dict 'scatter'": Row(
        _old_state_dict_key,
        _new_state_dict_key,
        "'dv'",
        shipped="""
        import torch, deepwave
        m = deepwave.ScalarBorn(1500 * torch.ones(8, 8), torch.zeros(8, 8), 4.0)
        assert "scatter" in m.state_dict()
        """,
    ),
    "ricker(length=)": Row(
        _old_ricker_length,
        _new_ricker_nt,
        "nt",
        shipped="""
        import deepwave
        w = deepwave.wavelets.ricker(25.0, length=100, dt=0.002, peak_time=0.05)
        assert w.shape == (100,)
        """,
    ),
    "python_backend=": Row(
        _old_python_backend,
        _new_backend,
        "backend",
        shipped="""
        import torch, deepwave
        deepwave.scalar(
            1500 * torch.ones(8, 8), 4.0, 0.002,
            source_amplitudes=torch.ones(1, 1, 4),
            source_locations=torch.tensor([[[4, 4]]]),
            python_backend=False,
        )
        """,
    ),
    "source channel 'p' on acoustic": Row(
        _old_source_p_on_acoustic,
        _new_source_volume_rate_on_acoustic,
        "source_amplitudes_volume_rate",
        shipped="""
        import torch, deepwave
        deepwave.acoustic(
            1500 * torch.ones(8, 8), torch.full((8, 8), 1000.0), 4.0, 0.002,
            source_amplitudes_p=torch.ones(1, 1, 4),
            source_locations_p=torch.tensor([[[4, 4]]]),
            receiver_locations_p=torch.tensor([[[4, 5]]]),
        )
        """,
    ),
    "source channel 'p' on elastic": Row(
        _old_source_p_on_elastic,
        _new_source_pressure_rate_on_elastic,
        "source_amplitudes_pressure_rate",
        shipped="""
        import torch, deepwave
        deepwave.elastic(
            torch.full((8, 8), 2.2e9), torch.full((8, 8), 1.1e9),
            torch.full((8, 8), 1 / 2200.0), 4.0, 0.002,
            source_amplitudes_p=torch.ones(1, 1, 4),
            source_locations_p=torch.tensor([[[4, 4]]]),
            receiver_locations_p=torch.tensor([[[4, 5]]]),
        )
        """,
    ),
    "scalar_born(receiver_locations=)": Row(
        _old_bare_receiver_locations,
        _new_sc_receiver_locations,
        "sc_receiver_locations",
        shipped="""
        import torch, deepwave
        deepwave.scalar_born(
            1500 * torch.ones(8, 8), torch.full((8, 8), 3.0), 4.0, dt=0.002,
            source_amplitudes=torch.ones(1, 1, 4),
            source_locations=torch.tensor([[[4, 4]]]),
            receiver_locations=torch.tensor([[[4, 5]]]),
        )
        """,
    ),
    "callback name 'scatter'": Row(
        _old_callback_name_scatter,
        _new_callback_name_dv,
        "'dv'",
        shipped="""
        import torch, deepwave

        seen = {}

        def cb(state):
            seen["scatter"] = state.get_model("scatter")

        deepwave.scalar_born(
            1500 * torch.ones(8, 8), torch.full((8, 8), 3.0), 4.0, dt=0.002,
            source_amplitudes=torch.ones(1, 1, 4),
            source_locations=torch.tensor([[[4, 4]]]),
            forward_callback=cb, callback_frequency=1,
        )
        assert seen["scatter"] is not None
        """,
    ),
    "callback gradient name 'scatter'": Row(
        _old_callback_name_scatter_gradient,
        _new_callback_name_dv_gradient,
        "'dv'",
        shipped="""
        import torch, deepwave

        seen = {}

        def cb(state):
            seen["scatter"] = state.get_gradient("scatter")

        dv = torch.full((8, 8), 3.0, requires_grad=True)
        out = deepwave.scalar_born(
            1500 * torch.ones(8, 8), dv, 4.0, dt=0.002,
            source_amplitudes=torch.ones(1, 1, 4),
            source_locations=torch.tensor([[[4, 4]]]),
            receiver_locations=torch.tensor([[[4, 5]]]),
            backward_callback=cb, callback_frequency=1,
        )
        out[-1].sum().backward()
        assert seen["scatter"] is not None
        """,
    ),
    "deepwave.common.* shim": Row(
        _shim("common", "upsample"),
        _canonical_upsample,
        "deepwave.discretisation.upsample",
        shipped="""
        import deepwave.common
        assert deepwave.common.upsample is not None
        """,
    ),
    "deepwave.location_interpolation.* shim": Row(
        _shim("location_interpolation", "Hicks"),
        _canonical_hicks,
        "deepwave.Hicks",
        shipped="""
        import deepwave.location_interpolation
        assert deepwave.location_interpolation.Hicks is not None
        """,
    ),
    "deepwave.backend_utils.* shim": Row(
        _shim("backend_utils", "SO_EXT"),
        _private("backend_utils", "SO_EXT"),
        "deepwave._backend_utils.SO_EXT",
        shipped="""
        import deepwave.backend_utils
        assert deepwave.backend_utils.SO_EXT is not None
        """,
    ),
    "deepwave.regular_grid.* shim": Row(
        _shim("regular_grid", "set_pml_profiles"),
        _private("regular_grid", "set_pml_profiles"),
        "deepwave._regular_grid.set_pml_profiles",
        shipped="""
        import deepwave.regular_grid
        assert deepwave.regular_grid.set_pml_profiles is not None
        """,
    ),
    "deepwave.staggered_grid.* shim": Row(
        _shim("staggered_grid", "set_pml_profiles"),
        _private("staggered_grid", "set_pml_profiles"),
        "deepwave._staggered_grid.set_pml_profiles",
        shipped="""
        import deepwave.staggered_grid
        assert deepwave.staggered_grid.set_pml_profiles is not None
        """,
    ),
}


def _equal(a: Any, b: Any) -> bool:
    """Whether two trigger results are the same thing."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return bool(torch.equal(a, b))
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return bool(a == b)


# ---------------------------------------------------------------------------
# 1. Every row warns, once, in the right words, and the replacement does not.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", sorted(_DEPRECATIONS))
def test_row_warns_once_naming_the_replacement_and_the_release(label: str) -> None:
    """One ``DeepwaveFutureWarning`` per call, naming replacement and 0.4.0."""
    row = _DEPRECATIONS[label]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        row.old()
    deprecations = [
        w for w in record if issubclass(w.category, deepwave.DeepwaveFutureWarning)
    ]
    assert deprecations, f"{label}: no DeepwaveFutureWarning was emitted"
    assert len(deprecations) == _expected_warning_count(label), (
        f"{label}: expected one warning per deprecated name, got "
        f"{[str(w.message) for w in deprecations]}"
    )
    for warning in deprecations:
        message = str(warning.message)
        assert DEPRECATION_REMOVAL_VERSION in message, message
        assert "deepwave.DeepwaveFutureWarning" in message, message
    joined = " ".join(str(w.message) for w in deprecations)
    assert row.replacement in joined, (label, row.replacement, joined)


#: The rows that legitimately warn about more than one name in a single call:
#: each passes two deprecated spellings, and each spelling gets its own
#: message.
_MULTI_NAME_ROWS = {
    "ScalarBorn(scatter_requires_grad=)": 2,  # scatter= comes with it
    "source channel 'p' on acoustic": 2,  # amplitudes and locations
    "source channel 'p' on elastic": 2,
}


def _expected_warning_count(label: str) -> int:
    return _MULTI_NAME_ROWS.get(label, 1)


@pytest.mark.parametrize("label", sorted(_DEPRECATIONS))
def test_row_is_filterable_by_its_category(label: str) -> None:
    """One filter on the category silences the row completely."""
    row = _DEPRECATIONS[label]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=deepwave.DeepwaveFutureWarning)
        row.old()
    assert [w for w in record if issubclass(w.category, FutureWarning)] == []


@pytest.mark.parametrize("label", sorted(_DEPRECATIONS))
def test_replacement_is_silent_and_identical(label: str) -> None:
    """The replacement spelling computes the same thing, and warns about nothing."""
    row = _DEPRECATIONS[label]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        replacement = row.new()
    assert [
        w for w in record if issubclass(w.category, deepwave.DeepwaveFutureWarning)
    ] == [], f"{label}: the replacement spelling warns"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deprecated = row.old()
    assert _equal(deprecated, replacement), f"{label}: the two spellings disagree"


def test_removal_version_preserves_two_subsequent_minors() -> None:
    """The 0.1 aliases survive both the 0.2 and 0.3 minor releases."""
    assert deepwave.__version__.startswith("0.1.")
    assert DEPRECATION_REMOVAL_VERSION == "0.4.0"


# ---------------------------------------------------------------------------
# 2. The source cannot drift from the table.
# ---------------------------------------------------------------------------

#: The categories a deprecation may be announced under. Only the first is
#: allowed at a ``warnings.warn`` site; the other two are named so that a
#: reversion to either is caught rather than merely unnoticed.
_DEPRECATION_CATEGORIES = frozenset(
    {"DeepwaveFutureWarning", "FutureWarning", "DeprecationWarning"}
)

#: The one function allowed to emit a deprecation, and the file it lives in.
_EMITTER = "warn_deprecated_name"
_EMITTER_FILE = "_exceptions.py"
#: The one function in the package that calls ``warnings.warn``:
#: ``warn_deprecated_name`` and every other user-facing warning reach it
#: through this, so that the warning is reported at the caller's own line.
_ATTRIBUTION_HELPER = "warn_at_user_call"


def _src_files() -> list[pathlib.Path]:
    """Every Python file in the package, in a stable order."""
    return sorted(_SRC.rglob("*.py"))


def _warnings_aliases(tree: ast.Module) -> set[str]:
    """The names ``import warnings`` is bound to in one module."""
    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "warnings"
    }


def _is_warnings_warn(call: ast.Call, aliases: set[str]) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "warn"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in aliases
    )


def _called_name(call: ast.Call) -> Optional[str]:
    """The bare name of what a call calls, through any attribute path."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _passthrough_warning_helpers(
    paths: list[pathlib.Path],
) -> dict[str, tuple[int, str]]:
    """The helpers that hand their *caller's* category to ``warnings.warn``.

    Found by reading the source rather than listed by hand: any function whose
    ``warnings.warn`` call takes its category from one of that function's own
    parameters leaves the choice of category to whoever calls it, so a
    deprecation routed through it appears at no ``warnings.warn`` site at all
    and would slip past the one-emitter guard.

    The search is transitive, because the chain is: the sole ``warnings.warn``
    call sits in the attribution helper, the once-per-cause emitters in
    ``_common`` and ``_engine`` forward their caller's category to *it*, and a
    deprecation may be handed to any of the three. A search that stopped at
    the functions calling ``warnings.warn`` directly would find only the first
    and let a category routed through the other two past unseen, so the set is
    grown to a fixed point instead.

    Returns:
        Each helper's name, mapped to the position of that parameter in its
        signature and the parameter's own spelling, so that a call to it can be
        read whether the category was passed positionally or by keyword.
    """
    functions: list[tuple[str, list[str], list[ast.Call]]] = []
    direct: dict[str, tuple[int, str]] = {}
    for path in paths:
        tree = ast.parse(path.read_text())
        aliases = _warnings_aliases(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            parameters = [
                argument.arg for argument in (*node.args.posonlyargs, *node.args.args)
            ]
            calls: list[ast.Call] = []
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Call):
                    continue
                if _is_warnings_warn(inner, aliases):
                    forwarded = list(inner.args[1:2])
                    forwarded += [
                        kw.value for kw in inner.keywords if kw.arg == "category"
                    ]
                    for argument in forwarded:
                        if isinstance(argument, ast.Name) and argument.id in parameters:
                            direct[node.name] = (
                                parameters.index(argument.id),
                                argument.id,
                            )
                else:
                    calls.append(inner)
            functions.append((node.name, parameters, calls))

    helpers = dict(direct)
    changed = True
    while changed:
        changed = False
        for name, parameters, calls in functions:
            if name in helpers:
                continue
            for call in calls:
                known = helpers.get(_called_name(call) or "")
                if known is None:
                    continue
                index, keyword = known
                forwarded = list(call.args[index : index + 1])
                forwarded += [kw.value for kw in call.keywords if kw.arg == keyword]
                for argument in forwarded:
                    if isinstance(argument, ast.Name) and argument.id in parameters:
                        helpers[name] = (parameters.index(argument.id), argument.id)
                        changed = True
                        break
                if name in helpers:
                    break
    return helpers


def _deprecation_warn_sites(
    paths: Optional[list[pathlib.Path]] = None,
) -> list[tuple[str, str]]:
    """``(location, category)`` per deprecation-emitting site in ``paths``.

    Both ways a category can reach ``warnings.warn``: named at the call itself,
    and handed to one of the pass-through helpers
    :func:`_passthrough_warning_helpers` finds.
    """
    if paths is None:
        paths = _src_files()
    helpers = _passthrough_warning_helpers(paths)
    sites = []
    for path in paths:
        tree = ast.parse(path.read_text())
        aliases = _warnings_aliases(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _is_warnings_warn(node, aliases):
                args = list(node.args[1:2])
                args += [kw.value for kw in node.keywords if kw.arg == "category"]
            else:
                helper = helpers.get(_called_name(node) or "")
                if helper is None:
                    continue
                index, keyword = helper
                args = list(node.args[index : index + 1])
                args += [kw.value for kw in node.keywords if kw.arg == keyword]
            sites.extend(
                (f"{path.name}:{node.lineno}", arg.id)
                for arg in args
                if isinstance(arg, ast.Name) and arg.id in _DEPRECATION_CATEGORIES
            )
    return sites


def test_one_emitter_for_every_deprecation() -> None:
    """Exactly one site in src announces a deprecation.

    The guard that keeps the table honest: a deprecation added with its own
    ``warnings.warn`` call could pick its own category, forget the removal
    version, or forget to name the way to filter it. Routing all of them
    through one function makes those three things structural rather than
    remembered, and makes this test the only place that has to check them.

    A deprecation handed to a once-per-cause helper counts as its own site: the
    helper passes the category straight through, so the choice is made at the
    call rather than inside ``warn_deprecated_name``.
    """
    sites = _deprecation_warn_sites()
    assert [category for _, category in sites] == ["DeepwaveFutureWarning"], sites
    assert sites[0][0].startswith(_EMITTER_FILE), sites


def test_the_pass_through_helpers_are_found() -> None:
    """The pass-through helpers are located by reading the source, not listed.

    If one of them is renamed, moved or gains a sibling, the search finds it;
    this asserts only that the search finds what is there today, so that a
    search that silently matched nothing could not make the guard above
    vacuous. The three are the attribution helper that holds Deepwave's one
    ``warnings.warn`` call and the two once-per-cause emitters that forward
    their caller's category to it -- which is what makes the search's
    transitivity load-bearing rather than decorative.
    """
    helpers = _passthrough_warning_helpers(_src_files())
    assert set(helpers) == {
        "warn_at_user_call",
        "_warn_once_per_cause",
        "_warn_once",
    }, helpers
    assert helpers["warn_at_user_call"] == (1, "category"), helpers
    assert helpers["_warn_once_per_cause"] == (2, "category"), helpers
    assert helpers["_warn_once"] == (2, "category"), helpers


@pytest.mark.parametrize(
    "call",
    [
        '_warn_once_per_cause("c", "m", DeepwaveFutureWarning, 2)',
        '_warn_once_per_cause("c", "m", category=FutureWarning, stacklevel=2)',
        '_common._warn_once("c", "m", DeprecationWarning, 2)',
    ],
)
def test_the_guard_catches_a_deprecation_routed_through_a_helper(
    call: str, tmp_path: pathlib.Path
) -> None:
    """A deprecation that never names ``warnings.warn`` is still a site.

    The gap this closes: both once-per-cause helpers take their category as an
    argument, so a ``DeepwaveFutureWarning`` passed to one of them emitted a
    deprecation from a second place while
    :func:`_deprecation_warn_sites` saw only ``warn_deprecated_name``. Run
    against a scratch module rather than a real edit to the package, so the
    guard is shown to fail on the code it is meant to reject without the
    package having to contain it.
    """
    scratch = tmp_path / "offender.py"
    scratch.write_text(f"def f() -> None:\n    {call}\n")
    sites = _deprecation_warn_sites([*_src_files(), scratch])
    assert [
        location for location, _ in sites if location.startswith("offender.py")
    ] == ["offender.py:2"], sites
    assert [category for _, category in sites] != ["DeepwaveFutureWarning"], sites


def test_the_emitter_is_the_documented_one() -> None:
    """The deprecation site sits inside ``warn_deprecated_name``.

    And Deepwave's one ``warnings.warn`` call sits inside the attribution
    helper it delegates to, in the same module: every user-facing warning goes
    through that one call so that it is reported at the caller's own line.
    """
    tree = ast.parse((_SRC / _EMITTER_FILE).read_text())
    emitters = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "warn"
            for inner in ast.walk(node)
        )
    ]
    assert emitters == [_ATTRIBUTION_HELPER], emitters
    sites = _deprecation_warn_sites()
    assert [location.split(":")[0] for location, _ in sites] == [_EMITTER_FILE], sites
    (deprecation,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            _called_name(inner) == _ATTRIBUTION_HELPER
            for inner in ast.walk(node)
            if isinstance(inner, ast.Call)
        )
    ]
    assert deprecation.name == _EMITTER, deprecation.name


#: Every file whose text ships to a user: the package, the documentation, the
#: changelog and the readme.
def _shipped_text_files() -> list[pathlib.Path]:
    files = sorted(_SRC.rglob("*.py"))
    docs = _ROOT / "docs"
    files += sorted(
        path
        for pattern in ("*.rst", "*.md")
        for path in docs.rglob(pattern)
        if "_generated" not in path.parts
    )
    files += [_ROOT / "CHANGELOG.md", _ROOT / "README.md"]
    return files


def test_no_shipped_text_promises_0_3_removal() -> None:
    """No public text may shorten the two subsequent minor releases."""
    offenders = [
        str(path.relative_to(_ROOT))
        for path in _shipped_text_files()
        if "removed in 0.3.0" in path.read_text(encoding="utf-8").lower()
    ]
    assert not offenders, offenders


def test_the_deprecation_tables_name_the_earliest_release() -> None:
    """The tables give readers the current earliest removal release."""
    stability = (_ROOT / "docs" / "stability.rst").read_text(encoding="utf-8")
    assert "- Earliest removal\n" in stability
    assert "at least two subsequent minor releases" in stability
    assert "0.4.0" in stability
    upgrading = (_ROOT / "docs" / "upgrading.rst").read_text(encoding="utf-8")
    assert "- Earliest removal\n" in upgrading
    assert "0.4.0" in upgrading


# ---------------------------------------------------------------------------
# 3. Audibility under Python's own default filters.
# ---------------------------------------------------------------------------

_SETUP = """
import torch, deepwave
v = 1500 * torch.ones(8, 8)
kwargs = dict(
    source_amplitudes=torch.ones(1, 1, 4),
    source_locations=torch.tensor([[[4, 4]]]),
    pml_freq=25.0,
)
"""

#: One snippet per path on which no fixed ``stacklevel`` reaches the user, each
#: written the indirect way: a ``DeprecationWarning`` here emits nothing at all
#: under CPython's own filters.
_DEFAULT_FILTER_CASES = {
    "ScalarBorn state_dict 'scatter'": (
        """
        m = deepwave.ScalarBorn(torch.ones(6, 6), 4.0, dv=torch.ones(6, 6))
        m.load_state_dict(
            {"v": torch.full((6, 6), 1.0), "scatter": torch.full((6, 6), 9.0)}
        )
        """,
        "state_dict key 'scatter'",
    ),
    "python_backend= via Module.forward": (
        # forward() called directly: no __call__ frames, so the Module
        # stacklevel runs off the top of the stack and the warning is
        # attributed to <sys>. Nothing Deepwave can do about that -- which
        # is exactly why the category has to be one shown regardless.
        """
        module = deepwave.Scalar(v, 4.0)
        module.forward(0.002, python_backend="eager", **kwargs)
        """,
        "python_backend is deprecated",
    ),
    "source channel 'p' via Module.forward": (
        """
        module = deepwave.Acoustic(v, 1000 * torch.ones(8, 8), 4.0)
        module.forward(
            0.002,
            source_amplitudes_p=kwargs["source_amplitudes"],
            source_locations_p=kwargs["source_locations"],
            pml_freq=25.0,
        )
        """,
        "source_amplitudes_p is deprecated",
    ),
    "scatter= via Module construction": (
        """
        deepwave.ScalarBorn(v, 4.0, scatter=torch.zeros(8, 8))
        """,
        "scatter is deprecated",
    ),
    "deepwave.common shim": (
        """
        deepwave.common.upsample
        """,
        "deepwave.common.upsample is deprecated",
    ),
}


@pytest.mark.parametrize("label", sorted(_DEFAULT_FILTER_CASES))
def test_deprecations_are_visible_under_default_filters(label: str) -> None:
    """Each snippet warns in a plain interpreter, unconfigured.

    ``pytest.warns`` installs ``simplefilter("always")``, so it cannot see
    this class of bug at all. This runs the snippet in a subprocess with no
    ``-W`` flag and no ``PYTHONWARNINGS``, i.e. under CPython's own default
    filter set, and requires the message on stderr.
    """
    body, expected = _DEFAULT_FILTER_CASES[label]
    script = _SETUP + textwrap.dedent(body)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    result = subprocess.run(  # noqa: S603 - fixed argv, our own interpreter
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert expected in result.stderr, (
        f"{label}: nothing on stderr under default filters.\n"
        f"stderr was:\n{result.stderr}"
    )
    assert DEPRECATION_REMOVAL_VERSION in result.stderr


# ---------------------------------------------------------------------------
# 4. The ``shipped`` rows really did ship; the extensions match their parent.
# ---------------------------------------------------------------------------


def _v0027_python() -> str:
    """The ``v0.0.27`` interpreter, or a skip naming how to provide one."""
    path = os.environ.get(_V0027_ENV)
    if not path:
        pytest.skip(
            f"set {_V0027_ENV} to the python of an environment with "
            "deepwave==0.0.27 installed to check that each deprecated name "
            "really shipped in that release"
        )
    return path


_SHIPPED_ROWS = sorted(
    label for label, row in _DEPRECATIONS.items() if row.shipped is not None
)
_EXTENSION_ROWS = sorted(
    label for label, row in _DEPRECATIONS.items() if row.extends is not None
)


def test_every_row_is_either_shipped_or_an_extension() -> None:
    """No row may be neither: both halves of group 4 must cover all of it."""
    assert set(_SHIPPED_ROWS) | set(_EXTENSION_ROWS) == set(_DEPRECATIONS)
    assert not set(_SHIPPED_ROWS) & set(_EXTENSION_ROWS)


@pytest.mark.parametrize("label", _SHIPPED_ROWS)
def test_the_deprecated_name_worked_in_v0027(label: str) -> None:
    """Each deprecated spelling is run under a real ``v0.0.27`` interpreter.

    This is what makes "0.1.0 removes nothing that v0.0.27 shipped" checkable
    rather than asserted: the table claims each of these names was reachable
    in that release, and the claim is settled by running it there.
    """
    interpreter = _v0027_python()
    snippet = _DEPRECATIONS[label].shipped
    assert snippet is not None
    result = subprocess.run(  # noqa: S603 - fixed argv, a user-named interpreter
        [interpreter, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{label}: the snippet failed under {interpreter}:\n{result.stderr}"
    )


@pytest.mark.parametrize("label", _EXTENSION_ROWS)
def test_an_extension_behaves_like_its_released_parent(label: str) -> None:
    """An extension warns exactly as the released row it was extended from.

    The propagators these rows are on never shipped, so there is no ``v0.0.27``
    behaviour to compare against. What is checkable is the thing the extension
    is *for*: one calling convention on all nine propagators, so the
    extension's warning must be the parent's warning with the parent's
    replacement named.
    """
    row = _DEPRECATIONS[label]
    assert row.extends is not None
    parent = _DEPRECATIONS[row.extends]
    assert parent.shipped is not None, "an extension's parent must be a shipped row"
    assert row.replacement == parent.replacement

    def messages(trigger: Callable[[], Any]) -> list[str]:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            trigger()
        return sorted(
            str(w.message)
            for w in record
            if issubclass(w.category, deepwave.DeepwaveFutureWarning)
        )

    child = messages(row.old)
    released = messages(parent.old)
    assert len(child) == len(released), (child, released)

    # The wording is the parent's, differing only in the propagator name each
    # message opens with (``"acoustic: ..."``), so strip that and compare.
    def bodies(texts: list[str]) -> list[str]:
        return sorted(text.split(": ", 1)[-1] for text in texts)

    assert bodies(child) == bodies(released), (child, released)
