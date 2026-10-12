"""Tests for the stability policy's runtime surface.

Covers the three things `docs/stability.rst` promises that are executable:

1. the version string,
2. the warning *categories*: that `deepwave.DeepwaveWarning` is the base of
   one class per cause, that `deepwave.DeepwaveFutureWarning` is separate
   from it, and that no propagator warns about its own maturity tier,
3. the `deepwave.DeepwaveError` hierarchy's promise that every existing
   `except <builtin>` around a Deepwave call keeps catching what it caught.

The tier *set* itself (which propagator is in which tier) is asserted in
`tests/test_capabilities.py::test_experimental_status_matches_the_documented_set`,
against both the declaration and the generated capability matrix.
"""

import inspect
import os
import pathlib
import pickle
import re
import subprocess
import sys
import warnings
from typing import Any

import pytest
import torch

import deepwave
from deepwave import _capabilities

# The tiers as documented in docs/stability.rst. Written out rather than
# derived from the specs, so that a change of tier has to be a deliberate
# edit here as well as in the code and the docs.
EXPERIMENTAL = (
    "acoustic_born",
    "acoustic_vti",
    "viscoacoustic",
    "elastic_born",
    "elastic_vti",
)
STABLE = ("scalar", "scalar_born", "acoustic", "elastic")


def _clear(*names: str) -> None:
    """Forgets any once-per-process warning ``names`` may already have spent."""
    for name in names:
        for cause in list(deepwave._common._warned_once_causes):
            if cause.endswith(f":{name}") or f":{name}:" in cause:
                deepwave._common._warned_once_causes.discard(cause)


def _run(name: str) -> object:
    """Calls propagator ``name`` on a tiny 2-D problem, from *this* file.

    The call is made here rather than inside a helper in another module so
    that the emitted warning's ``filename`` is this test file: that is what
    makes the ``stacklevel`` assertion below meaningful.
    """
    models = _capabilities._models(name, 2)
    kwargs = _capabilities._base_kwargs(name, 2, None)
    kwargs.update(_capabilities._extra_kwargs(name, 2))
    return getattr(deepwave, name)(*models, **kwargs)


#: Keys of ``_capabilities._extra_kwargs`` that are *model* tensors, which the
#: ``nn.Module`` forms take in their constructor rather than in ``forward``.
_MODEL_KWARGS = frozenset({"dv", "drho", "dlamb", "dmu", "dbuoyancy", "c66", "tilt"})


def _run_module(name: str) -> object:
    """:func:`_run`'s ``nn.Module`` equivalent, called from *this* file too.

    The same problem as :func:`_run`, so the two forms must report the same
    things; the models move from the call to the constructor, as the
    ``nn.Module`` interface requires.
    """
    models = _capabilities._models(name, 2)
    kwargs = _capabilities._base_kwargs(name, 2, None)
    kwargs.update(_capabilities._extra_kwargs(name, 2))
    grid_spacing = kwargs.pop("grid_spacing")
    dt = kwargs.pop("dt")
    constructor = {key: kwargs.pop(key) for key in list(kwargs) if key in _MODEL_KWARGS}
    module_class = getattr(deepwave, "".join(p.title() for p in name.split("_")))
    return module_class(*models, grid_spacing, **constructor)(dt, **kwargs)


def _pml_freq_migration(
    records: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    """The ``pml_freq`` migration warnings in ``records``."""
    return [r for r in records if r.category is deepwave.PmlFreqInferenceWarning]


def _forget_pml_freq_migration() -> None:
    """Forgets every ``pml_freq`` migration warning already spent."""
    for cause in list(deepwave._common._warned_once_causes):
        if cause.startswith("pml_freq_migration:"):
            deepwave._common._warned_once_causes.discard(cause)


def _maturity(records: list[warnings.WarningMessage]) -> list[str]:
    """The messages in ``records`` that talk about a maturity tier.

    There is no category to look for any more -- that is the point -- so
    this matches on the vocabulary a maturity warning would have to use.
    """
    return [
        str(r.message)
        for r in records
        if "experimental" in str(r.message) or "provisional" in str(r.message)
    ]


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def test_version_is_0_1_0() -> None:
    """``deepwave.__version__`` is the version this release ships as."""
    assert deepwave.__version__ == "0.1.0"


# The exact recipe printed in docs/stability.rst, as a runnable program: it
# must work both here and on v0.0.27, where ``deepwave.__version__`` does not
# exist at all.
_RECIPE = """
from importlib.metadata import version as _dist_version

from packaging.version import parse

import deepwave

installed = parse(getattr(deepwave, "__version__", None) or _dist_version("deepwave"))
print(installed, installed >= parse("0.1.0"), hasattr(deepwave, "PropagatorOutput"))
"""


def test_version_supports_the_documented_feature_detection_recipe() -> None:
    """Both halves of docs/stability.rst's dual-version recipe work here."""
    assert hasattr(deepwave, "PropagatorOutput")
    packaging_version = pytest.importorskip("packaging.version")
    assert packaging_version.parse(deepwave.__version__) >= packaging_version.parse(
        "0.1.0"
    )
    # The reason the recipe says to parse rather than compare strings.
    assert not "0.0.9" < "0.0.27"
    assert packaging_version.parse("0.0.9") < packaging_version.parse("0.0.27")
    result = subprocess.run(  # noqa: S603 - fixed argv, this interpreter
        [sys.executable, "-c", _RECIPE],
        capture_output=True,
        text=True,
        check=True,
    )
    version, is_new, has_output = result.stdout.split()
    assert version == deepwave.__version__
    assert (is_new, has_output) == ("True", "True")


def test_feature_detection_recipe_runs_on_v0_0_27() -> None:
    """The recipe works on v0.0.27, which has no ``deepwave.__version__``.

    This is the trap the recipe exists to avoid: reading
    ``deepwave.__version__`` directly raises ``AttributeError`` on the very
    release the code is trying to support, so the recipe falls back to the
    installed distribution's metadata.
    """
    reference = os.environ.get("DEEPWAVE_V0027_PYTHON")
    if not reference or not pathlib.Path(reference).is_file():
        pytest.skip("DEEPWAVE_V0027_PYTHON not set to a reference interpreter")
    absent = subprocess.run(  # noqa: S603 - fixed argv, a known interpreter
        [reference, "-c", "import deepwave; deepwave.__version__"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert absent.returncode != 0
    assert "__version__" in absent.stderr
    result = subprocess.run(  # noqa: S603 - fixed argv, a known interpreter
        [reference, "-c", _RECIPE],
        capture_output=True,
        text=True,
        check=True,
    )
    version, is_new, has_output = result.stdout.split()
    assert version == "0.0.27"
    # v0.0.27 is older than 0.1.0 and returned a plain tuple, so the
    # documented ``hasattr`` branch selects the positional path there.
    assert (is_new, has_output) == ("False", "False")


# ---------------------------------------------------------------------------
# Warning categories
# ---------------------------------------------------------------------------


def test_deepwave_warning_is_a_user_warning() -> None:
    """The base category is public, subclasses ``UserWarning``, is exported."""
    assert issubclass(deepwave.DeepwaveWarning, UserWarning)
    assert deepwave.DeepwaveWarning.__module__ == "deepwave"
    assert "DeepwaveWarning" in deepwave.__all__


@pytest.mark.parametrize(
    "cls",
    [
        deepwave.ZeroWidthFaceWarning,
        deepwave.MigrationWarning,
        deepwave.PmlFreqInferenceWarning,
    ],
)
def test_cause_categories_derive_from_the_base(cls: type[Warning]) -> None:
    """One class per cause, all under the one base a user can filter."""
    assert issubclass(cls, deepwave.DeepwaveWarning)
    assert cls.__module__ == "deepwave"
    assert cls.__name__ in deepwave.__all__


def test_pml_freq_inference_warning_is_a_migration_warning() -> None:
    """The narrower category sits under the cause it is an instance of."""
    assert issubclass(deepwave.PmlFreqInferenceWarning, deepwave.MigrationWarning)


@pytest.mark.parametrize("name", EXPERIMENTAL + STABLE)
@pytest.mark.parametrize("form", ["functional", "module"])
def test_inferred_pml_freq_warns_exactly_once_per_call(name: str, form: str) -> None:
    """All nine warn once per call, in both forms, when they infer pml_freq.

    The survey here is six samples long, so the inferred frequency is not the
    25.0 that Deepwave 0.0.27 used: the numbers differ from that release's and
    the call must say so. Once, not once per propagator-internal resolution
    and not once per model: a warning that repeated would be noise, and one
    that some of the nine omitted would make the migration warning
    unreliable as a check.
    """
    _forget_pml_freq_migration()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _run(name) if form == "functional" else _run_module(name)
    warned = _pml_freq_migration(record)
    assert len(warned) == 1, [str(w.message) for w in warned]
    message = str(warned[0].message)
    assert "pml_freq was not specified" in message
    assert "pml_freq=25.0" in message


def test_inferred_pml_freq_warning_states_the_value_and_how_it_was_inferred() -> None:
    """The message carries everything needed to act on it without the docs."""
    _forget_pml_freq_migration()
    v = torch.full((30, 30), 1500.0, dtype=torch.float64)
    amplitudes = deepwave.wavelets.ricker(15.0, 600, 0.004, 0.06)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        deepwave.scalar(
            v,
            10.0,
            0.004,
            source_amplitudes=amplitudes.to(torch.float64).reshape(1, 1, -1),
            source_locations=torch.full((1, 1, 2), 15, dtype=torch.long),
            receiver_locations=torch.full((1, 1, 2), 10, dtype=torch.long),
            pml_width=5,
            max_vel=1500.0,
        )
    warned = _pml_freq_migration(record)
    assert len(warned) == 1
    message = str(warned[0].message)
    # The resolved value, and how this call reached it.
    assert "resolved to 15 --" in message
    assert "peak of the source amplitudes' energy spectrum" in message
    # Both ways out, and the removal release.
    assert "pml_freq=25.0" in message
    assert "pml_freq=<your dominant frequency>" in message
    assert "0.4.0" in message
    # Its own category, so the filter recipe is in the message being read.
    assert "deepwave.PmlFreqInferenceWarning" in message


def test_pml_freq_of_exactly_25_is_silent() -> None:
    """A 25 Hz Ricker infers the old frequency and needs no migration warning."""
    _forget_pml_freq_migration()
    v = torch.full((30, 30), 1500.0, dtype=torch.float64)
    amplitudes = deepwave.wavelets.ricker(25.0, 600, 0.004, 0.06)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = deepwave.scalar(
            v,
            10.0,
            0.004,
            source_amplitudes=amplitudes.to(torch.float64).reshape(1, 1, -1),
            source_locations=torch.full((1, 1, 2), 15, dtype=torch.long),
            receiver_locations=torch.full((1, 1, 2), 10, dtype=torch.long),
            pml_width=5,
            max_vel=1500.0,
        )
    assert out.info is not None
    assert out.info.pml_freq == 25.0
    assert _pml_freq_migration(record) == []


def test_segmented_run_warns_once_for_the_whole_record() -> None:
    """``run_segments`` resolves pml_freq once, so it warns once, not per segment."""
    _forget_pml_freq_migration()
    v = torch.full((30, 30), 1500.0, dtype=torch.float64)
    amplitudes = (
        deepwave.wavelets.ricker(15.0, 600, 0.004, 0.06)
        .to(torch.float64)
        .reshape(1, 1, -1)
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        deepwave.checkpointing.run_segments(
            deepwave.scalar,
            [v],
            10.0,
            0.004,
            {"source_amplitudes": amplitudes},
            4,
            source_locations=torch.full((1, 1, 2), 15, dtype=torch.long),
            receiver_locations=torch.full((1, 1, 2), 10, dtype=torch.long),
            pml_width=5,
            max_vel=1500.0,
        )
    warned = _pml_freq_migration(record)
    assert len(warned) == 1, [str(w.message) for w in warned]


#: ``(source frequency, expected number of migration warnings)``. 20 Hz and
#: 35 Hz sit inside the band the warning used to keep quiet (a factor of two
#: either side of 25.0, i.e. 12.5-50 Hz); 25.0 is the one value that
#: reproduces Deepwave 0.0.27's fixed default exactly and is therefore silent.
#: The rule is "not exactly 25.0", not "far from 25.0": a 20 Hz wavelet gives
#: a different absorbing boundary, and therefore different numbers, from the
#: release being migrated from, so the call has to say so.
_QUIET_BAND_CASES = ((20.0, 1), (25.0, 0), (35.0, 1))

#: Time samples in the Ricker records the ``pml_freq`` tests below build. Long
#: enough that the inference resolves the wavelet's own centre frequency (to
#: three significant figures, which is what it reports), short enough that all
#: nine propagators run in a moment on a 10-cell grid.
_PML_FREQ_TEST_NT = 600


def _ricker_arguments(
    name: str, freq: float, nt: int = _PML_FREQ_TEST_NT
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    """The tiny 2-D call of :func:`_run`, with a ``freq`` Hz Ricker source.

    ``_capabilities``' own survey is six samples long, whose spectral peak is
    nowhere near 25 Hz; these tests need a source whose inferred frequency is
    a chosen value instead, so every source-amplitude channel is replaced by
    the same Ricker wavelet.
    """
    models = _capabilities._models(name, 2)
    kwargs: dict[str, object] = _capabilities._base_kwargs(name, 2, None)
    kwargs.update(_capabilities._extra_kwargs(name, 2))
    dt = kwargs["dt"]
    assert isinstance(dt, float)
    amplitudes = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq)
    for key, value in list(kwargs.items()):
        if key.startswith("source_amplitudes"):
            assert isinstance(value, torch.Tensor)
            kwargs[key] = (
                amplitudes.to(value.dtype)
                .reshape(1, 1, -1)
                .expand(value.shape[0], value.shape[1], nt)
                .contiguous()
            )
    return models, kwargs


def _run_ricker(name: str, freq: float, form: str) -> Any:
    """Calls propagator ``name`` with a ``freq`` Hz Ricker, from *this* file.

    Called here rather than from a helper module so that the emitted
    warning's ``filename`` is this test file, which is what makes the
    attribution assertions below meaningful.
    """
    models, kwargs = _ricker_arguments(name, freq)
    if form == "functional":
        return getattr(deepwave, name)(*models, **kwargs)
    grid_spacing = kwargs.pop("grid_spacing")
    dt = kwargs.pop("dt")
    constructor = {key: kwargs.pop(key) for key in list(kwargs) if key in _MODEL_KWARGS}
    module_class = getattr(deepwave, "".join(p.title() for p in name.split("_")))
    return module_class(*models, grid_spacing, **constructor)(dt, **kwargs)


@pytest.mark.parametrize("name", EXPERIMENTAL + STABLE)
@pytest.mark.parametrize(("freq", "expected"), _QUIET_BAND_CASES)
@pytest.mark.parametrize("form", ["functional", "module"])
def test_inferred_pml_freq_warns_unless_it_is_exactly_25(
    name: str, freq: float, expected: int, form: str
) -> None:
    """All nine warn for an inferred value of 20 or 35 Hz, and not for 25.

    Both frequencies are inside the quiet band the warning used to have, so
    this is the propagator-level statement that the band is gone: every one
    of the nine, in both call forms, reports an inferred 20 Hz or 35 Hz
    boundary, because it differs from the 25.0 Deepwave 0.0.27 used. A 25 Hz
    Ricker resolves to exactly 25.0, reproduces that release, and stays
    silent -- the warning is about a numerical difference, not about the
    inference having happened.
    """
    _forget_pml_freq_migration()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = _run_ricker(name, freq, form)
    assert out.info.pml_freq == freq
    warned = _pml_freq_migration(record)
    assert len(warned) == expected, [str(w.message) for w in warned]
    for message in (str(w.message) for w in warned):
        assert f"resolved to {freq:g} --" in message
        assert "pml_freq=25.0" in message


@pytest.mark.parametrize("name", ["scalar", "acoustic", "elastic"])
@pytest.mark.parametrize("form", ["functional", "module"])
def test_inferred_pml_freq_warning_names_the_callers_own_line(
    name: str, form: str
) -> None:
    """The warning is attributed to the user's call, not to Deepwave's insides.

    A fixed ``stacklevel`` cannot do this: a propagator function is one frame
    below the caller and an ``nn.Module`` adds ``torch.nn.Module.__call__``'s
    frames on top, so the two forms are different depths. Attribution is by
    the first frame outside Deepwave (and outside PyTorch, whose ``Module``
    machinery is not the caller either), which is this file in both.
    """
    _forget_pml_freq_migration()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _run_ricker(name, 20.0, form)
    warned = _pml_freq_migration(record)
    assert len(warned) == 1, [str(w.message) for w in warned]
    assert warned[0].filename == __file__, warned[0].filename


@pytest.mark.parametrize("route", ["checkpoint_segments", "run_segments"])
def test_segmented_inferred_pml_freq_warning_names_the_callers_own_line(
    route: str,
) -> None:
    """Both ways of segmenting a run attribute the warning to the caller.

    ``checkpoint_segments=`` reaches the segment driver through the engine's
    delegation, and ``run_segments`` is called directly: two different depths
    again, and the delegation's own frame is what a fixed ``stacklevel``
    used to blame for the first of them.
    """
    _forget_pml_freq_migration()
    v = torch.full((30, 30), 1500.0, dtype=torch.float64)
    amplitudes = (
        deepwave.wavelets.ricker(15.0, 600, 0.004, 0.06)
        .to(torch.float64)
        .reshape(1, 1, -1)
    )
    survey = {
        "source_locations": torch.full((1, 1, 2), 15, dtype=torch.long),
        "receiver_locations": torch.full((1, 1, 2), 10, dtype=torch.long),
        "pml_width": 5,
        "max_vel": 1500.0,
    }
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        if route == "checkpoint_segments":
            deepwave.scalar(
                v,
                10.0,
                0.004,
                source_amplitudes=amplitudes,
                checkpoint_segments=4,
                **survey,
            )
        else:
            deepwave.checkpointing.run_segments(
                deepwave.scalar,
                [v],
                10.0,
                0.004,
                {"source_amplitudes": amplitudes},
                4,
                **survey,
            )
    warned = _pml_freq_migration(record)
    assert len(warned) == 1, [str(w.message) for w in warned]
    assert warned[0].filename == __file__, warned[0].filename


def test_inferred_pml_freq_fallback_says_whether_there_were_sources() -> None:
    """The fallback message distinguishes no sources from no usable peak.

    A call with no sources at all has nothing to infer from; a call whose
    sources yielded no peak has amplitudes worth looking at. Reporting the
    second wording for the first case sends the reader to inspect source
    amplitudes that do not exist.
    """
    _forget_pml_freq_migration()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        deepwave._common.set_pml_freq(None, 0.002)
    warned = _pml_freq_migration(record)
    assert len(warned) == 1
    assert "no source amplitudes to infer a dominant frequency from" in str(
        warned[0].message
    )
    _forget_pml_freq_migration()
    constant = torch.ones(1, 1, 40, dtype=torch.float64)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        deepwave._common.set_pml_freq(None, 0.002, [constant])
    warned = _pml_freq_migration(record)
    assert len(warned) == 1
    message = str(warned[0].message)
    assert "no dominant frequency could be inferred from the source" in message
    assert "has no source amplitudes" not in message


def test_future_warning_is_not_a_deepwave_warning() -> None:
    """A deprecated spelling is a different family from a physical surprise.

    Filtering one must not silence the other, which is only true if they do
    not share a base below ``Warning``.
    """
    assert issubclass(deepwave.DeepwaveFutureWarning, FutureWarning)
    assert not issubclass(deepwave.DeepwaveFutureWarning, deepwave.DeepwaveWarning)
    assert not issubclass(deepwave.DeepwaveWarning, FutureWarning)


def test_experimental_warning_is_gone() -> None:
    """The class never shipped and has been removed, not renamed."""
    assert not hasattr(deepwave, "ExperimentalWarning")
    assert "ExperimentalWarning" not in deepwave.__all__


@pytest.mark.parametrize("name", EXPERIMENTAL + STABLE)
def test_no_propagator_warns_about_its_maturity(name: str) -> None:
    """The tier is a documentation label: no call announces it.

    The interface promise is identical on all nine, so an experimental
    propagator must be as quiet as a stable one.
    """
    _clear(name)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _run(name)
    assert _maturity(record) == []


@pytest.mark.parametrize(
    "cls",
    [
        deepwave.DeepwaveWarning,
        deepwave.ZeroWidthFaceWarning,
        deepwave.MigrationWarning,
        deepwave.PmlFreqInferenceWarning,
        deepwave.DeepwaveFutureWarning,
        deepwave.DeepwaveError,
        deepwave.DeepwaveValueError,
        deepwave.DeepwaveTypeError,
        deepwave.DeepwaveRuntimeError,
        deepwave.DeepwaveNotImplementedError,
        deepwave.DeepwaveFloatingPointError,
    ],
)
def test_exception_types_pickle_round_trip(cls: type[BaseException]) -> None:
    """``__module__ = "deepwave"`` must name a real, importable path.

    Pickling an exception (as ``multiprocessing``/``torch.distributed`` do
    when propagating one from a worker) looks the class up at
    ``<__module__>.<__qualname__>``, so a private ``__module__`` -- or a
    public one that does not re-export the name -- would only fail there.
    """
    restored = pickle.loads(  # noqa: S301 - our own bytes, produced above
        pickle.dumps(cls("boom"))
    )
    assert type(restored) is cls
    assert str(restored) == "boom"
    assert getattr(deepwave, cls.__name__) is cls


def test_status_is_declared_not_probed() -> None:
    """``_capabilities`` reads the tier from the spec, so it survives the guard.

    With the once-per-process guard already consumed for every propagator
    (as it is here), the previous warning-sniffing implementation would have
    reported every experimental propagator as stable.
    """
    for name in EXPERIMENTAL + STABLE:
        spec = deepwave.checkpointing._get_spec(getattr(deepwave, name))
        _run(name)  # consume the guard, if it was not already consumed
        assert _capabilities._propagator_status(name) == spec.status
    statuses = {n: _capabilities._propagator_status(n) for n in EXPERIMENTAL + STABLE}
    assert statuses == {
        **dict.fromkeys(EXPERIMENTAL, "experimental"),
        **dict.fromkeys(STABLE, "stable"),
    }


# ---------------------------------------------------------------------------
# Every exported name has a tier
# ---------------------------------------------------------------------------

#: The tier of every name in ``deepwave.__all__``, as documented in
#: ``docs/stability.rst``. This exists so that later work cannot add
#: a public name -- or graduate one -- without saying which tier it is in:
#: the test below fails on any name that is exported but untiered, or tiered
#: but not exported.
_TIER_OF = {
    "experimental": {
        "acoustic_born",
        "AcousticBorn",
        "acoustic_vti",
        "AcousticVti",
        "viscoacoustic",
        "Viscoacoustic",
        "elastic_born",
        "ElasticBorn",
        "elastic_vti",
        "ElasticVti",
    },
    "provisional": {
        "attenuation",
        "checkpointing",
        "wavefields",
        "estimate_memory",
        # The TypedDict ``estimate_memory`` returns: its keys *are* the
        # fields the page calls provisional, so it shares that tier rather
        # than the stable one its function's existence sits in.
        "MemoryEstimate",
        "transducers",
        # ``report`` and the ``Report`` it returns; the module's own
        # docstring declares the tier.
        "diagnostics",
        # The propagator description and the module that publishes it: the
        # commitment is that a propagator *can* be described as data, not
        # the exact field and method set, which is expected to grow.
        "introspection",
        "spec",
        "PropagatorSpec",
        # The pre-flight resolution and what it returns: that a call can be
        # resolved and costed without running it is the commitment, not the
        # field set of the answer.
        "plan",
        "Plan",
        # The grid/time-step suggestion and the NamedTuple it returns. That
        # a discretisation can be suggested from a velocity and a frequency
        # is the commitment; the numbers are not. The cost model that picks
        # the accuracy order is an explicit heuristic, the points-per-
        # wavelength floor it applies is a policy, and the field set of the
        # answer is expected to grow -- the same reasoning as ``plan``/
        # ``Plan`` and ``estimate_memory``/``MemoryEstimate`` above.
        "suggest_discretisation",
        "Discretisation",
        # The combined storage-policy object. The individual storage_mode/
        # storage_path/storage_compression arguments it is an
        # alternative spelling for are stable and not deprecated.
        "Storage",
        # The helper modules and functions that arrived with 0.1.0: a set of
        # misfit functions, of anisotropy conversions, of DAS helpers, a
        # dispersion correction, a boundary-condition builder and a
        # finiteness check. What each computes is settled; which functions
        # there are, and how they spell their arguments, is not.
        "anisotropy",
        "boundaries",
        "check_finite",
        "das",
        "misfits",
        "time_dispersion",
    },
    "stable": {
        "__version__",
        # The returned objects and the state that continues a run: the
        # output layout, the resolved-configuration record and the opaque
        # bundle a continuation hands back.
        "AuxiliaryState",
        "RunInfo",
        "IGNORE_LOCATION",
        "ON_NODE_TOLERANCE",
        "Accuracy",
        "Acoustic",
        "acoustic",
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
        # The CFL, Courant and resampling arithmetic: the same functions
        # that were reachable as ``deepwave.common.*``, at their new home.
        "discretisation",
        "Elastic",
        "elastic",
        "MigrationWarning",
        "PmlFreqInferenceWarning",
        "ZeroWidthFaceWarning",
        "Hicks",
        "InitialState",
        "lambmubuoyancy_to_vpvsrho",
        "LocationInterp",
        "LocationsLike",
        "PropagatorOutput",
        "receiver_names",
        "ReceiverSpec",
        "Reduction",
        "Scalar",
        "scalar",
        "ScalarBorn",
        "scalar_born",
        "SourceSpec",
        "StorageLike",
        "StorageMode",
        "StoragePath",
        "to_cells",
        "vpvsrho_to_lambmubuoyancy",
        "wavefield_names",
        "WavefieldView",
        "wavelets",
    },
}


#: ``_TIER_OF`` inverted: the tier of one name.
_LABEL_OF = {name: tier for tier, names in _TIER_OF.items() for name in names}


def test_every_exported_name_is_in_exactly_one_tier() -> None:
    """``deepwave.__all__`` and the tier lists cover each other exactly."""
    tiered: set[str] = set()
    for names in _TIER_OF.values():
        assert not (tiered & names), sorted(tiered & names)
        tiered |= names
    exported = set(deepwave.__all__)
    assert tiered - exported == set(), sorted(tiered - exported)
    assert exported - tiered == set(), sorted(exported - tiered)


def test_experimental_tier_list_matches_the_propagator_specs() -> None:
    """The experimental tier is exactly the seven propagators and their Modules."""
    functions = {n for n in _TIER_OF["experimental"] if n[0].islower()}
    assert functions == set(EXPERIMENTAL)
    for name in EXPERIMENTAL:
        spec = deepwave.checkpointing._get_spec(getattr(deepwave, name))
        assert spec.status == "experimental", name
    for name in STABLE:
        assert name in _TIER_OF["stable"]


def test_docs_page_names_every_non_stable_name() -> None:
    """Each experimental/provisional name is named in docs/stability.rst.

    Keeps the tier data above and the page a reader actually sees from
    drifting apart.
    """
    page = (
        pathlib.Path(__file__).resolve().parent.parent / "docs" / "stability.rst"
    ).read_text()
    for tier in ("experimental", "provisional"):
        for name in _TIER_OF[tier]:
            if name[0].isupper():
                continue  # the Module forms: the tier-table test below covers them
            assert name in page, f"{name} ({tier}) is not named in stability.rst"


#: The heading of ``docs/stability.rst``'s tier tables, and the heading that
#: follows the last of them.
_TIER_SECTION = ("Every public name, and what may change", "Supported aliases")


def _tier_table_rows() -> list[tuple[str, str]]:
    """``(name cell, label)`` for every labelled row of the page's tier tables.

    The three list-tables between the two headings above are parsed as they
    are written: a row begins with ``   * - `` (its first cell, the names),
    each further cell begins with ``     - ``, and a continuation line is
    indented further. Rows whose second cell is not one of the four labels --
    the header rows -- are dropped.
    """
    page = (
        pathlib.Path(__file__).resolve().parent.parent / "docs" / "stability.rst"
    ).read_text()
    lines = page.splitlines()
    start = lines.index(_TIER_SECTION[0])
    end = lines.index(_TIER_SECTION[1])
    cells: list[list[str]] = []
    rows: list[list[str]] = []
    for line in lines[start:end]:
        if line.startswith("   * - "):
            cells = [line[7:]]
            rows.append(cells)
        elif line.startswith("     - "):
            cells.append(line[7:])
        elif cells and line.startswith("       "):
            cells[-1] += " " + line.strip()
    labels = {"stable", "provisional", "experimental", "deprecated"}
    return [
        (row[0], row[1].strip().strip("*").strip())
        for row in rows
        if len(row) >= 2 and row[1].strip().strip("*").strip() in labels
    ]


def _names_in(cell: str) -> set[str]:
    """The public symbols a tier table's *name* cell names.

    A symbol counts as named when the cell holds it as a whole literal
    (``Hicks``, ``deepwave.__version__``) or as a cross-reference
    (:class:`deepwave.Hicks`). A literal that qualifies a symbol rather than
    being one -- ``Hicks(differentiable=True)``,
    ``elastic(surface_method="pm")`` -- deliberately does not count, because
    the row is about that one keyword rather than about the symbol's own tier.
    """
    found = set(re.findall(r"``([A-Za-z_][A-Za-z_0-9]*)``", cell))
    found |= set(re.findall(r"``deepwave\.([A-Za-z_][A-Za-z_0-9]*)``", cell))
    found |= set(re.findall(r":[a-z]+:`~?deepwave\.([A-Za-z_][A-Za-z_0-9]*)`", cell))
    return found


def test_the_tier_tables_label_every_exported_name() -> None:
    """Every name in ``deepwave.__all__`` has a row, with the tier above.

    ``_TIER_OF`` is the machine-readable copy of the page's tier tables, and
    nothing but a reader checked that the two agreed. The rule: each name in
    ``deepwave.__all__`` is named by at least one row of the three tables, a
    row may name a whole family (the propagators and their ``nn.Module``
    forms, the exception and warning classes), and where more than one row
    names the same symbol every one of them must give it the same tier.
    """
    rows = _tier_table_rows()
    assert len(rows) >= 20, len(rows)
    exported = set(deepwave.__all__)
    labels_of: dict[str, set[str]] = {}
    for cell, label in rows:
        for name in _names_in(cell) & exported:
            labels_of.setdefault(name, set()).add(label)
    missing = exported - set(labels_of)
    assert missing == set(), sorted(missing)
    wrong = {
        name: sorted(labels)
        for name, labels in labels_of.items()
        if labels != {_LABEL_OF[name]}
    }
    assert wrong == {}, wrong


def test_every_propagator_docstring_declares_the_status_the_spec_reports() -> None:
    """Each propagator's ``Stability:`` line is its ``spec(...).status``.

    The page says a name's label is stated in its own documentation and that
    the capability matrix's ``Status`` column is read from the one declaration
    that records it, "so the two cannot disagree". Only the column is
    generated: the docstring line is typed by hand, so this is what makes the
    claim true.
    """
    for name in EXPERIMENTAL + STABLE:
        status = deepwave.spec(getattr(deepwave, name)).status
        expected = f"Stability: {status} (see :doc:`/stability`)."
        module_name = "".join(part.title() for part in name.split("_"))
        for symbol in (name, module_name):
            doc = inspect.getdoc(getattr(deepwave, symbol)) or ""
            stated = [
                line.strip()
                for line in doc.splitlines()
                if line.strip().startswith("Stability:")
            ]
            assert stated == [expected], (symbol, stated)


# ---------------------------------------------------------------------------
# DeepwaveError hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("deepwave_error", "builtin"),
    [
        (deepwave.DeepwaveValueError, ValueError),
        (deepwave.DeepwaveTypeError, TypeError),
        (deepwave.DeepwaveRuntimeError, RuntimeError),
        (deepwave.DeepwaveNotImplementedError, NotImplementedError),
        (deepwave.DeepwaveFloatingPointError, FloatingPointError),
    ],
)
def test_each_error_is_caught_by_its_builtin_base(
    deepwave_error: type[Exception], builtin: type[Exception]
) -> None:
    """An existing ``except <builtin>`` still catches the Deepwave subclass."""
    assert issubclass(deepwave_error, builtin)
    assert issubclass(deepwave_error, deepwave.DeepwaveError)
    assert deepwave_error.__module__ == "deepwave"
    assert deepwave_error.__name__ in deepwave.__all__
    caught: list[Exception] = []
    try:
        raise deepwave_error("boom")
    except builtin as e:  # the pre-0.1.0 spelling
        caught.append(e)
    assert len(caught) == 1
    assert isinstance(caught[0], deepwave.DeepwaveError)


def test_deepwave_error_does_not_catch_unrelated_exceptions() -> None:
    """``except DeepwaveError`` is narrower than ``except Exception``."""
    assert not issubclass(ValueError, deepwave.DeepwaveError)
    escaped = False
    try:
        raise ValueError("not deepwave's")
    except deepwave.DeepwaveError:  # pragma: no cover - must not catch
        pass
    except ValueError:
        escaped = True
    assert escaped


def test_check_finite_raises_a_catchable_floating_point_error() -> None:
    """``check_finite`` still raises ``FloatingPointError``, now a DeepwaveError too."""
    x = torch.tensor([1.0, float("nan")])
    with pytest.raises(FloatingPointError) as excinfo:
        deepwave.check_finite(x, "receiver data")
    assert isinstance(excinfo.value, deepwave.DeepwaveFloatingPointError)
    assert isinstance(excinfo.value, deepwave.DeepwaveError)
    deepwave.check_finite(torch.tensor([1.0, 2.0]))  # finite input still passes


def test_converted_validators_keep_their_builtin_types() -> None:
    """The converted ``_common`` raises are still ``ValueError``/``TypeError``.

    ``set_model_gradient_sampling_interval`` and
    ``check_locations_are_within_model`` are the demonstration conversions;
    their raises must remain catchable exactly as before.
    """
    mgsi = deepwave._common.set_model_gradient_sampling_interval
    with pytest.raises(TypeError) as type_error:
        mgsi(1.5)  # type: ignore[arg-type]
    assert isinstance(type_error.value, deepwave.DeepwaveError)
    with pytest.raises(ValueError, match="must be >= 1"):
        mgsi(0)
    with pytest.raises(ValueError, match="must divide") as value_error:
        mgsi(3, user_nt=10)
    assert isinstance(value_error.value, deepwave.DeepwaveError)

    within = deepwave._common.check_locations_are_within_model
    with pytest.raises(ValueError, match="must not be empty") as empty:
        within([], [None])
    assert isinstance(empty.value, deepwave.DeepwaveError)
    locations = torch.tensor([[[0, 20]]])
    with pytest.raises(RuntimeError, match="out of bounds") as bounds:
        within([10, 10], [locations], ["source_locations"])
    assert isinstance(bounds.value, deepwave.DeepwaveError)


def test_a_propagator_argument_error_is_still_a_value_error() -> None:
    """End to end: a bad ``mgsi`` on a real call still raises ``ValueError``."""
    v = torch.full((12, 12), 1500.0, dtype=torch.float64)
    v.requires_grad_(True)
    with pytest.raises(ValueError, match="must divide") as excinfo:
        deepwave.scalar(
            v,
            4.0,
            0.001,
            source_amplitudes=torch.zeros(1, 1, 10, dtype=torch.float64),
            source_locations=torch.tensor([[[5, 5]]]),
            receiver_locations=torch.tensor([[[6, 6]]]),
            model_gradient_sampling_interval=3,
        )
    assert isinstance(excinfo.value, deepwave.DeepwaveError)
