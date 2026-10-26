"""Import-surface tests for the namespace-privatisation rename.

Locks the exact public root/``common`` namespace after the module renames
to ``_``-prefixed private names, plus the new curated ``common`` shim, and
confirms the renamed modules are actually private: importable
under their new ``_``-prefixed name, and no longer reachable under the old
public name.
"""

import importlib
import pathlib
import re
import subprocess
import sys
import types
import warnings

import pytest

import deepwave

_EXPECTED_ALL = {
    "Accuracy",
    "Acoustic",
    "AcousticBorn",
    "AcousticVti",
    "AuxiliaryState",
    "Callback",
    "CallbackDirection",
    "CallbackKind",
    "CallbackMode",
    "CallbackRole",
    "CallbackState",
    "DEFAULT_COURANT_NUMBER",
    "DeepwaveError",
    "DeepwaveFloatingPointError",
    "DeepwaveFutureWarning",
    "DeepwaveNotImplementedError",
    "DeepwaveOSError",
    "DeepwaveRuntimeError",
    "DeepwaveTypeError",
    "DeepwaveValueError",
    "DeepwaveWarning",
    "Discretisation",
    "Elastic",
    "ElasticBorn",
    "ElasticVti",
    "MigrationWarning",
    "PmlFreqInferenceWarning",
    "ZeroWidthFaceWarning",
    "Hicks",
    "IGNORE_LOCATION",
    "InitialState",
    "LocationInterp",
    "LocationsLike",
    "MemoryEstimate",
    "Plan",
    "ON_NODE_TOLERANCE",
    "PropagatorOutput",
    "PropagatorSpec",
    "ReceiverSpec",
    "Reduction",
    "RunInfo",
    "Scalar",
    "ScalarBorn",
    "SourceSpec",
    "Storage",
    "StorageLike",
    "StoragePath",
    "StorageMode",
    "Viscoacoustic",
    "WavefieldView",
    "__version__",
    "acoustic",
    "acoustic_born",
    "acoustic_vti",
    "anisotropy",
    "attenuation",
    "boundaries",
    "check_finite",
    "checkpointing",
    "das",
    "diagnostics",
    "discretisation",
    "elastic",
    "elastic_born",
    "elastic_vti",
    "estimate_memory",
    "introspection",
    "lambmubuoyancy_to_vpvsrho",
    "misfits",
    "plan",
    "receiver_names",
    "scalar",
    "scalar_born",
    "spec",
    "suggest_discretisation",
    "time_dispersion",
    "to_cells",
    "transducers",
    "viscoacoustic",
    "vpvsrho_to_lambmubuoyancy",
    "wavefield_names",
    "wavefields",
    "wavelets",
}

# The ten names of Deepwave 0.0.27's ``deepwave.common`` that were part of
# its documented interface. Section 0 of the release policy forbids removing
# anything that shipped, so all ten must stay reachable through the shim for
# the deprecation window.
#
# Verified against a 0.0.27 install: that module had no ``__all__`` and 68
# names not starting with an underscore, so this is a practical subset, not
# the whole of it. The rest were private helpers, imported modules and typing
# imports; they raise, with a message that says so.
_V0027_COMMON_NAMES = [
    "Callback",
    "CallbackState",
    "IGNORE_LOCATION",
    "cfl_condition",
    "cfl_condition_n",
    "cosine_taper_end",
    "downsample",
    "lambmubuoyancy_to_vpvsrho",
    "upsample",
    "vpvsrho_to_lambmubuoyancy",
]

# ``deepwave.common.__all__`` advertises exactly the shimmed names. It is not
# empty: a star-import reads ``__all__`` and then does a normal attribute
# lookup per name, so an empty one would make ``from deepwave.common import *``
# bind nothing and warn about nothing, and the caller's code would fail later
# as an unrelated ``NameError``.
_EXPECTED_COMMON_ALL: set[str] = set(_V0027_COMMON_NAMES)

# Aliases that were added to ``deepwave.common`` during development but never
# shipped from it, in 0.0.27 or earlier, and so were dropped rather than
# frozen as a second permanent spelling. Verified absent from a 0.0.27
# install.
_DROPPED_FROM_COMMON = [
    "critical_courant_number",
    "to_cells",
    "estimate_memory",
    "check_finite",
]


def test_root_all_is_exactly_the_frozen_list() -> None:
    """``deepwave.__all__`` is exactly the frozen post-A6 public surface."""
    assert set(deepwave.__all__) == _EXPECTED_ALL
    # No duplicates.
    assert len(deepwave.__all__) == len(_EXPECTED_ALL)


def test_every_root_all_name_resolves() -> None:
    """Every name in ``deepwave.__all__`` is a real, resolvable attribute."""
    for name in deepwave.__all__:
        getattr(deepwave, name)


def test_common_all_is_exactly_the_curated_shim_list() -> None:
    """``deepwave.common.__all__`` is exactly the curated public shim list."""
    assert set(deepwave.common.__all__) == _EXPECTED_COMMON_ALL
    assert len(deepwave.common.__all__) == len(_EXPECTED_COMMON_ALL)


def test_every_common_all_name_resolves() -> None:
    """Every name in ``deepwave.common.__all__`` is a real, resolvable attribute."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", deepwave.DeepwaveFutureWarning)
        for name in deepwave.common.__all__:
            getattr(deepwave.common, name)


def test_common_all_matches_the_shim_alias_table() -> None:
    """``__all__`` cannot drift from the table ``__getattr__`` actually reads.

    ``__all__`` is written out as a literal so that static tooling can read
    it, which means nothing but this test keeps the two in step. A name in
    ``__all__`` but not in the table would make ``from deepwave.common
    import *`` raise; a name in the table but not in ``__all__`` would be
    silently missing from the star-import.
    """
    aliases = deepwave.common._ALIASES
    assert set(deepwave.common.__all__) == set(aliases)
    assert len(deepwave.common.__all__) == len(aliases)


def test_private_engine_module_is_importable() -> None:
    """The renamed private module ``deepwave._engine`` is importable."""
    import deepwave._engine

    assert hasattr(deepwave._engine, "PropagatorOutput")


@pytest.mark.parametrize(
    "module_name",
    [
        "engine",
        "storage",
    ],
)
def test_old_public_module_names_no_longer_exist(module_name: str) -> None:
    """The private-renamed module names are no longer reachable at all.

    Neither as a ``deepwave`` attribute (``deepwave.engine`` etc.) nor as a
    directly importable submodule (``import deepwave.engine`` etc.): both
    forms must fail now that the module lives only at ``deepwave._engine``
    etc.

    ``backend_utils``, ``regular_grid`` and ``staggered_grid`` were on this
    list and are deliberately not any more: unlike these two, all three *did*
    ship at their public spelling in v0.0.27 (confirmed against a 0.0.27
    install: its top level held fifteen names, five of them modules), so
    under the release policy they are deprecated shims rather than removals.
    See the shim tests below.
    """
    with pytest.raises((AttributeError, ModuleNotFoundError)):
        getattr(deepwave, module_name)
    with pytest.raises((AttributeError, ModuleNotFoundError)):
        __import__(f"deepwave.{module_name}")


def test_boundaries_is_the_helper_function_not_the_old_private_module() -> None:
    """``deepwave.boundaries`` is the helper, and the module is still gone.

    The name was previously the (branch-only, never released) boundary-fill
    module, privatised to ``deepwave._boundaries``. Reusing it for the
    public helper is safe because the module spelling never shipped -- 0.0.27
    had no ``deepwave.boundaries`` -- but the *module* form must still fail,
    so that nothing silently resolves the old meaning.
    """
    assert callable(deepwave.boundaries)
    assert deepwave.boundaries(2, top="free") == (
        ["free", "pml", "pml", "pml"],
        [0, 20, 20, 20],
    )
    with pytest.raises(ModuleNotFoundError):
        __import__("deepwave.boundaries")
    assert importlib.import_module("deepwave._boundaries") is not deepwave.boundaries


# ---------------------------------------------------------------------------
# The compatibility shims (deepwave.common, deepwave.location_interpolation)
# ---------------------------------------------------------------------------


def test_common_shim_warns_on_attribute_access_but_not_on_import() -> None:
    """Importing ``deepwave`` is silent; touching a shim name warns.

    The shim resolves lazily (PEP 562) precisely so that the very common
    ``import deepwave`` does not emit a deprecation warning at every
    interpreter start.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(deepwave.common)
    assert [
        w for w in caught if issubclass(w.category, deepwave.DeepwaveFutureWarning)
    ] == []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert deepwave.common.upsample is not None
    messages = [
        str(w.message)
        for w in caught
        if issubclass(w.category, deepwave.DeepwaveFutureWarning)
    ]
    assert len(messages) == 1
    assert "deepwave.common.upsample is deprecated" in messages[0]
    assert "0.4.0" in messages[0]
    assert "deepwave.discretisation.upsample" in messages[0]


def test_common_shim_warns_on_every_access_not_just_the_first() -> None:
    """The alias is not cached into the module, so each caller is told."""
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert deepwave.common.downsample is not None
        assert [
            w for w in caught if issubclass(w.category, deepwave.DeepwaveFutureWarning)
        ], "a repeat access stopped warning"


@pytest.mark.parametrize("name", _V0027_COMMON_NAMES)
def test_v0027_common_names_are_still_reachable(name: str) -> None:
    """All ten documented ``deepwave.common`` names still resolve, and warn.

    Nothing that shipped in 0.0.27 is removed in this release; they become
    deprecated aliases instead, each with the same warning shape and each
    still offered by ``dir()``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = getattr(deepwave.common, name)
    assert value is not None
    assert name in dir(deepwave.common)
    messages = [
        str(w.message)
        for w in caught
        if issubclass(w.category, deepwave.DeepwaveFutureWarning)
    ]
    assert len(messages) == 1, messages
    assert f"deepwave.common.{name} is deprecated" in messages[0]
    assert "0.4.0" in messages[0]


@pytest.mark.parametrize("name", _DROPPED_FROM_COMMON)
def test_branch_only_common_aliases_are_gone(name: str) -> None:
    """Names that never shipped from ``deepwave.common`` were dropped.

    Keeping them would mint a second permanent public spelling for something
    that only ever had one released home. The error must still route the
    reader to the canonical home rather than merely refusing.
    """
    with pytest.raises(AttributeError, match=name):
        getattr(deepwave.common, name)


def test_missing_name_with_no_public_home_says_so() -> None:
    """The error does not promise a canonical home that does not exist.

    ``setup_propagator`` was one of the many names 0.0.27's ``common`` merely
    happened to expose; it has no public replacement, and the message must
    not route the reader to a top-level or ``discretisation`` spelling that
    is not there.
    """
    with pytest.raises(AttributeError) as excinfo:
        getattr(deepwave.common, "setup_propagator")  # noqa: B009 -- __getattr__
    message = str(excinfo.value)
    assert "no public replacement" in message
    assert "deepwave.discretisation.setup_propagator" not in message
    assert "deepwave.setup_propagator" not in message


def test_star_import_binds_the_shim_names_and_warns_about_each() -> None:
    """``from deepwave.common import *`` is covered by the deprecation.

    A star-import reads ``__all__`` and then looks each name up as an
    attribute, so PEP 562's ``__getattr__`` runs once per name: the names
    bind to the canonical objects *and* every one of them warns. With an
    empty ``__all__`` this route bound nothing and warned about nothing, and
    the caller's next line failed as an unrelated ``NameError``.

    Run in a subprocess so that the star-import happens at module level in a
    clean interpreter, as a user's would.
    """
    expected = sorted(_V0027_COMMON_NAMES)
    source = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    from deepwave.common import *\n"
        "import deepwave\n"
        f"expected = {expected!r}\n"
        "for name in expected:\n"
        "    assert name in dir(), f'{name} was not bound by the star-import'\n"
        "assert upsample is deepwave.discretisation.upsample\n"
        "assert IGNORE_LOCATION == deepwave.IGNORE_LOCATION\n"
        "assert callable(cosine_taper_end)\n"
        "hits = sorted(str(w.message).split()[0] for w in caught\n"
        "              if issubclass(w.category, deepwave.DeepwaveFutureWarning)\n"
        "              and str(w.message).startswith('deepwave.common.'))\n"
        "assert hits == ['deepwave.common.' + n for n in expected], hits\n"
        "assert all(w.filename == '<string>' for w in caught\n"
        "           if str(w.message).startswith('deepwave.common.'))\n"
    )
    subprocess.run(  # noqa: S603 -- the source is this literal, above
        [sys.executable, "-c", source], check=True, capture_output=True, text=True
    )


def test_common_shim_from_import_still_works_and_warns() -> None:
    """``from deepwave.common import X`` goes through the same hook.

    PEP 562's ``__getattr__`` covers ``from``-imports of module attributes,
    and the ``IMPORT_FROM`` bytecode does the lookup in the *importing*
    frame, so ``stacklevel=2`` lands on user code either way.
    """
    source = (
        "import warnings\n"
        "import deepwave\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    from deepwave.common import cosine_taper_end\n"
        "assert cosine_taper_end is not None\n"
        "hits = [w for w in caught\n"
        "        if issubclass(w.category, deepwave.DeepwaveFutureWarning)\n"
        "        and 'deepwave.common.cosine_taper_end' in str(w.message)]\n"
        "assert len(hits) == 1, [str(w.message) for w in caught]\n"
        "assert hits[0].filename == '<string>', hits[0].filename\n"
    )
    subprocess.run(  # noqa: S603 -- the source is this literal, above
        [sys.executable, "-c", source], check=True, capture_output=True, text=True
    )


def test_location_interpolation_shim_warns_and_returns_the_same_class() -> None:
    """The 0.0.27 module spelling is restored, deprecated, and identical.

    Deepwave 0.0.27 did expose ``deepwave.location_interpolation.Hicks``, so
    it cannot simply vanish; it resolves to the same class object as
    ``deepwave.Hicks``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hicks = deepwave.location_interpolation.Hicks
        eps = deepwave.location_interpolation.DEFAULT_EPS
    messages = [
        str(w.message)
        for w in caught
        if issubclass(w.category, deepwave.DeepwaveFutureWarning)
    ]
    assert len(messages) == 2
    assert any("deepwave.Hicks" in m for m in messages)
    assert any("deepwave.ON_NODE_TOLERANCE" in m for m in messages)
    assert all("0.4.0" in m for m in messages)
    assert hicks is deepwave.Hicks
    assert eps == deepwave.ON_NODE_TOLERANCE
    with pytest.raises(AttributeError):
        _ = deepwave.location_interpolation.not_a_real_name


def test_location_interpolation_all_matches_the_shim_alias_table() -> None:
    """``__all__`` cannot drift from the table ``__getattr__`` actually reads.

    The same contract as ``deepwave.common``: ``__all__`` is written out as a
    literal so that static tooling can read it, so only this test keeps the
    two in step.
    """
    shim = deepwave.location_interpolation
    aliases = shim._ALIASES
    assert set(shim.__all__) == set(aliases)
    assert len(shim.__all__) == len(aliases)


def test_location_interpolation_star_import_binds_and_warns() -> None:
    """``from deepwave.location_interpolation import *`` is covered too.

    The sibling of the ``deepwave.common`` star-import test: with an empty
    ``__all__`` this route bound nothing and warned about nothing, and the
    caller's next line failed as an unrelated ``NameError``.

    Run in a subprocess so that the star-import happens at module level in a
    clean interpreter, as a user's would.
    """
    expected = sorted(deepwave.location_interpolation.__all__)
    source = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    from deepwave.location_interpolation import *\n"
        "import deepwave\n"
        f"expected = {expected!r}\n"
        "for name in expected:\n"
        "    assert name in dir(), f'{name} was not bound by the star-import'\n"
        "assert Hicks is deepwave.Hicks\n"
        "assert DEFAULT_EPS == deepwave.ON_NODE_TOLERANCE\n"
        "prefix = 'deepwave.location_interpolation.'\n"
        "hits = sorted(str(w.message).split()[0] for w in caught\n"
        "              if issubclass(w.category, deepwave.DeepwaveFutureWarning)\n"
        "              and str(w.message).startswith(prefix))\n"
        "assert hits == [prefix + n for n in expected], hits\n"
        "assert all(w.filename == '<string>' for w in caught\n"
        "           if str(w.message).startswith(prefix))\n"
    )
    subprocess.run(  # noqa: S603 -- the source is this literal, above
        [sys.executable, "-c", source], check=True, capture_output=True, text=True
    )


# ---------------------------------------------------------------------------
# Every autodoc target is in some __all__
# ---------------------------------------------------------------------------

# Names that a docs page documents but that are deliberately absent from
# their module's ``__all__``. Expected to stay EMPTY: an entry here is a
# visible admission that the documented surface and the advertised surface
# disagree, and must carry a comment saying why.
_AUTODOC_ALLOWED_OUTSIDE_ALL: frozenset[str] = frozenset()

_AUTODOC_DIRECTIVE = re.compile(
    # The target may carry an explicit signature -- ".. autoexception::
    # deepwave.DeepwaveError(*args)" -- which is not part of the name.
    r"^\s*\.\.\s+auto(?:module|class|function|data|method|exception)::"
    r"\s*([^\s(]+)",
    re.M,
)


def _autodoc_targets() -> list[tuple[str, str]]:
    """(target, page) for every autodoc directive under ``docs/``."""
    docs = pathlib.Path(__file__).resolve().parent.parent / "docs"
    out = []
    for path in sorted([*docs.rglob("*.rst"), *docs.rglob("*.md")]):
        page = str(path.relative_to(docs))
        out.extend(
            (target, page) for target in _AUTODOC_DIRECTIVE.findall(path.read_text())
        )
    return out


def test_autodoc_targets_were_found() -> None:
    """The scanner finds targets at all -- otherwise the check is vacuous."""
    targets = _autodoc_targets()
    assert len(targets) > 50, f"only found {len(targets)} autodoc targets"
    assert any(t.startswith("deepwave.") for t, _ in targets)


def test_every_autodoc_target_is_in_some_dunder_all() -> None:
    """Anything the documentation documents is advertised by its module.

    A documented name that is missing from its module's ``__all__`` is
    reachable but unadvertised: ``from deepwave import *`` misses it, tooling
    does not offer it, and the stability policy does not obviously cover it.
    Either add it to the ``__all__`` or point the doc target at the canonical
    home.

    Only the *first* attribute after the module is checked, so a method
    target such as ``deepwave.Hicks.source`` is checked as far as ``Hicks``.
    """
    failures = []
    for target, page in _autodoc_targets():
        if not target.startswith("deepwave"):
            continue
        parts = target.split(".")
        # Longest importable module prefix.
        module = None
        depth = 0
        for i in range(len(parts), 0, -1):
            try:
                module = importlib.import_module(".".join(parts[:i]))
                depth = i
                break
            except ImportError:
                continue
        if module is None:
            failures.append(f"{target} ({page}): no importable module prefix")
            continue
        if depth == len(parts):
            continue  # the target is a module itself
        attribute = parts[depth]
        if attribute in _AUTODOC_ALLOWED_OUTSIDE_ALL:
            continue
        exported = getattr(module, "__all__", None)
        if exported is None:
            failures.append(f"{target} ({page}): {module.__name__} has no __all__")
        elif attribute not in exported:
            failures.append(
                f"{target} ({page}): {attribute!r} is not in {module.__name__}.__all__"
            )
    assert not failures, "documented but not advertised:\n  " + "\n  ".join(failures)


def test_autodoc_allowlist_contains_no_name_that_is_actually_exported() -> None:
    """The exemption list cannot rot into a list of already-fixed names."""
    stale = []
    for name in _AUTODOC_ALLOWED_OUTSIDE_ALL:
        for module_name in (
            "deepwave",
            "deepwave.discretisation",
            "deepwave.anisotropy",
            "deepwave.diagnostics",
        ):
            module = importlib.import_module(module_name)
            if name in getattr(module, "__all__", ()):
                stale.append(f"{name} is in {module_name}.__all__")
    assert not stale, "stale exemptions:\n  " + "\n  ".join(stale)


# ---------------------------------------------------------------------------
# The three module shims restored in 0.1.0 (backend_utils, regular_grid,
# staggered_grid). Deepwave 0.0.27's top level held fifteen names, five of
# them modules; ``common`` and ``location_interpolation`` were restored
# earlier, and these three had simply vanished, which the release policy
# ("nothing that shipped in v0.0.27 is removed in this release") does not
# allow. None of them was ever a designed interface -- they are the compiled
# backend's ctypes plumbing and the eager backend's finite-difference
# stencils -- so each is a deprecated shim rather than a promoted API.
# ---------------------------------------------------------------------------

_RESTORED_SHIMS = ("backend_utils", "regular_grid", "staggered_grid")


@pytest.mark.parametrize("module_name", _RESTORED_SHIMS)
def test_restored_shim_imports_without_warning(module_name: str) -> None:
    """Importing the module is silent; touching a name warns.

    The lazy PEP 562 resolution is the point: ``import deepwave`` imports all
    five shims, and if any of them warned at import time every Deepwave
    session would start with a deprecation notice nobody asked for.
    """
    module = importlib.import_module(f"deepwave.{module_name}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(module)
    assert [
        w for w in caught if issubclass(w.category, deepwave.DeepwaveFutureWarning)
    ] == []


@pytest.mark.parametrize("module_name", _RESTORED_SHIMS)
def test_restored_shim_warns_on_every_attribute_access(module_name: str) -> None:
    """Each forwarded name resolves, warns once, and names 0.4.0."""
    module = importlib.import_module(f"deepwave.{module_name}")
    private = importlib.import_module(f"deepwave._{module_name}")
    for name in module.__all__:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = getattr(module, name)
        messages = [
            str(w.message)
            for w in caught
            if issubclass(w.category, deepwave.DeepwaveFutureWarning)
        ]
        assert len(messages) == 1, (name, messages)
        assert f"deepwave.{module_name}.{name}" in messages[0]
        assert "0.4.0" in messages[0]
        # Forwarded, not copied: the shim reads through to the private module
        # on every access, so a value the engine rebinds is not frozen at
        # first touch.
        assert value is getattr(private, name)


@pytest.mark.parametrize("module_name", _RESTORED_SHIMS)
def test_restored_shim_all_matches_its_forward_table(module_name: str) -> None:
    """``__all__`` cannot drift from the tuple ``__getattr__`` reads."""
    module = importlib.import_module(f"deepwave.{module_name}")
    forwarded = module._FORWARDED
    assert set(module.__all__) == set(forwarded)
    assert len(module.__all__) == len(forwarded)
    assert sorted(module.__dir__()) == sorted(forwarded)


@pytest.mark.parametrize("module_name", _RESTORED_SHIMS)
def test_restored_shim_rejects_an_unknown_name(module_name: str) -> None:
    """An unknown attribute names what the shim does carry."""
    module = importlib.import_module(f"deepwave.{module_name}")
    with pytest.raises(AttributeError) as excinfo:
        _ = module.not_a_real_name
    assert "not_a_real_name" in str(excinfo.value)
    assert module.__all__[0] in str(excinfo.value)


@pytest.mark.parametrize("module_name", _RESTORED_SHIMS)
def test_restored_shim_carries_every_name_v0027_defined(module_name: str) -> None:
    """The shim's surface is the 0.0.27 module's own definitions.

    Frozen here rather than read from a 0.0.27 install, which is not present
    in CI. Each list is the set of names that module defined itself, taken
    from a real 0.0.27 install: its ``dir()`` minus the ``typing`` imports and
    imported modules that its import statements happened to re-export.
    """
    expected = {
        "backend_utils": {
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
        },
        "regular_grid": {
            "diff1",
            "diff2",
            "diffx1",
            "diffx2",
            "diffy1",
            "diffy2",
            "diffz1",
            "diffz2",
            "set_pml_profiles",
        },
        "staggered_grid": {
            "diff1",
            "diff1h",
            "diffx1",
            "diffxh1",
            "diffy1",
            "diffyh1",
            "diffz1",
            "diffzh1",
            "set_pml_profiles",
        },
    }[module_name]
    module = importlib.import_module(f"deepwave.{module_name}")
    assert set(module.__all__) == expected


# ---------------------------------------------------------------------------
# Every public module advertises exactly its own interface
# ---------------------------------------------------------------------------

#: Every module a user is meant to import from, including the ``v0.0.27``
#: shims.
_PUBLIC_MODULES = (
    "anisotropy",
    "backend_utils",
    "checkpointing",
    "common",
    "das",
    "diagnostics",
    "discretisation",
    "introspection",
    "location_interpolation",
    "misfits",
    "regular_grid",
    "staggered_grid",
    "time_dispersion",
    "transducers",
    "types",
    "wavefields",
    "wavelets",
)

#: Names from the standard library that a module imports to write its own
#: annotations. They are visible in the module namespace because Python has
#: no other way to import them, they are not Deepwave's to advertise, and no
#: one could mistake ``deepwave.das.Sequence`` for part of Deepwave. Anything
#: *defined in Deepwave* is not allowlisted, whatever it is called: the check
#: below re-tests that, so this list cannot be used to hide a Deepwave name.
_STDLIB_REEXPORTS = frozenset(
    {
        "Any",
        "Callable",
        "Container",
        "ForwardRef",
        "Iterator",
        "Literal",
        "Mapping",
        "MappingProxyType",
        "NamedTuple",
        "Optional",
        "Protocol",
        "Sequence",
        "TYPE_CHECKING",
        "TypedDict",
        "Union",
        "cast",
        "dataclass",
        "get_args",
        "runtime_checkable",
    }
)

#: The three ``v0.0.27`` shims whose ``__all__`` is fixed by what ``v0.0.27``
#: exported rather than by what this release documents; they are pinned
#: instead by ``test_restored_shim_carries_every_name_v0027_defined``.
_UNDOCUMENTED_SHIMS = frozenset({"backend_utils", "regular_grid", "staggered_grid"})


def _defined_in_deepwave(value: object) -> bool:
    return str(getattr(value, "__module__", "")).startswith("deepwave")


@pytest.mark.parametrize("module_name", _PUBLIC_MODULES)
def test_public_module_namespace_is_exactly_its_dunder_all(module_name: str) -> None:
    """Nothing a module happens to import is advertised as its interface.

    ``__all__`` says what the module offers; everything else in its namespace
    is either an implementation detail (which belongs under an underscore) or
    a standard-library name it imported to write its annotations (which is
    allowlisted above, and only when it is not a Deepwave name).
    """
    module = importlib.import_module(f"deepwave.{module_name}")
    exported = set(module.__all__)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stray = []
        for name in dir(module):
            if name.startswith("_") or name in exported:
                continue
            value = getattr(module, name)
            if isinstance(value, types.ModuleType):
                continue
            if name in _STDLIB_REEXPORTS and not _defined_in_deepwave(value):
                continue
            stray.append(name)
    assert not stray, (
        f"deepwave.{module_name} exposes {stray}, which are not in its __all__; "
        "rename them with a leading underscore (e.g. `import X as _X`) or add "
        "them to __all__ and document them"
    )


def test_stdlib_reexport_allowlist_holds_no_deepwave_name() -> None:
    """The allowlist above cannot be used to hide a Deepwave name."""
    offenders = []
    for module_name in _PUBLIC_MODULES:
        module = importlib.import_module(f"deepwave.{module_name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name in _STDLIB_REEXPORTS:
                value = getattr(module, name, None)
                if value is not None and _defined_in_deepwave(value):
                    offenders.append(f"deepwave.{module_name}.{name}")
    assert not offenders, offenders


@pytest.mark.parametrize("module_name", _PUBLIC_MODULES)
def test_every_dunder_all_name_is_documented(module_name: str) -> None:
    """The converse of ``test_every_autodoc_target_is_in_some_dunder_all``.

    A name a module advertises but that appears nowhere in ``docs/`` is a
    name a user is invited to call and given nothing to read about.
    """
    if module_name in _UNDOCUMENTED_SHIMS:
        pytest.skip(f"deepwave.{module_name} is pinned to the v0.0.27 surface")
    docs = pathlib.Path(__file__).resolve().parent.parent / "docs"
    text = "\n".join(
        path.read_text() for path in [*docs.rglob("*.rst"), *docs.rglob("*.md")]
    )
    module = importlib.import_module(f"deepwave.{module_name}")
    undocumented = [
        name
        for name in module.__all__
        if not re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text)
    ]
    assert not undocumented, (
        f"deepwave.{module_name} advertises {undocumented}, which docs/ never mentions"
    )
