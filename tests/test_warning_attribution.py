"""Every user-facing warning names the user's own line, in every call form.

A warning whose ``filename``/``lineno`` point inside Deepwave tells the user
nothing: they cannot see which of their own lines caused it, and ``-W
error::UserWarning`` plus a traceback is the only way left to find out. The
paths that reach a warning are different depths below the caller -- a
propagator function is one frame down, an ``nn.Module`` adds
``torch.nn.Module.__call__``'s frames, and ``checkpoint_segments=`` reaches
the segment driver through the engine's delegation -- so no fixed
``stacklevel`` can be right for all of them. Deepwave therefore counts the
frames instead (:func:`deepwave._exceptions.stacklevel_outside_deepwave`),
and every warning it emits goes through the one function that does so.

The tests below assert that, for one representative warning of each
user-facing category and each of the three call forms, the warning is
attributed to this file. ``test_attribution_is_not_free`` shows the assertion
has content: with the frame walk replaced by its fixed fallback, the same
calls blame Deepwave's own modules.
"""

import warnings
from typing import Any

import pytest
import torch

import deepwave
import deepwave._common
import deepwave._engine
import deepwave._exceptions

#: A 20 Hz Ricker: long enough for the ``pml_freq`` inference to resolve its
#: centre frequency, short enough to run in a moment on a 30-cell grid.
_NT = 200
_DT = 0.004

#: The grid spacing that leaves fewer than six cells per wavelength at 20 Hz
#: and 1500 m/s, and the one that leaves plenty.
_COARSE_SPACING = 80.0
_FINE_SPACING = 10.0


def _survey() -> dict[str, Any]:
    """The keyword arguments shared by every call below."""
    return {
        "source_amplitudes": deepwave.wavelets.ricker(
            20.0, _NT, _DT, 1.5 / 20.0
        ).reshape(1, 1, -1),
        "source_locations": torch.full((1, 1, 2), 15, dtype=torch.long),
        "receiver_locations": torch.full((1, 1, 2), 10, dtype=torch.long),
        "pml_width": 5,
        "max_vel": 1500.0,
    }


#: ``category -> (extra keyword arguments, grid spacing, a fragment of the
#: message that identifies it)``. One representative warning per user-facing
#: category: the grid-cells-per-wavelength resolution check, the ``pml_freq``
#: inference report, the Nyquist check, and the deprecated-name announcement
#: (``DeepwaveFutureWarning``, whose single emitter is
#: ``_exceptions.warn_deprecated_name``).
_CATEGORIES = {
    "points_per_wavelength": ({"pml_freq": 20.0}, _COARSE_SPACING, "grid cells per"),
    "pml_freq_inference": ({}, _FINE_SPACING, "pml_freq was not specified"),
    "nyquist": ({"pml_freq": 1000.0}, _FINE_SPACING, "greater than the Nyquist"),
    "deprecated_name": (
        {"pml_freq": 20.0, "python_backend": False},
        _FINE_SPACING,
        "python_backend is deprecated",
    ),
}

_FORMS = ("function", "module", "segments")


def _forget_once_per_process_causes() -> None:
    """Re-arms just the once-per-process causes these tests rely on.

    Discarding the individual causes rather than clearing the registry: a
    wholesale clear would also re-arm warnings another test in this process
    has already spent, which is a different test's business.
    """
    causes = deepwave._common._warned_once_causes
    causes.discard("points_per_wavelength")
    for cause in list(causes):
        if cause.startswith("pml_freq_migration:"):
            causes.discard(cause)


def _call(form: str, extra: dict[str, Any], grid_spacing: float) -> None:
    """Runs ``scalar`` in one of the three call forms, from *this* file.

    Written out here rather than in a helper module because the point of the
    assertions is which file the warning is attributed to: a helper would
    become the caller and the tests would be measuring the helper.
    """
    v = torch.full((30, 30), 1500.0)
    kwargs = {**_survey(), **extra}
    if form == "function":
        deepwave.scalar(v, grid_spacing, _DT, **kwargs)
    elif form == "module":
        deepwave.Scalar(v, grid_spacing)(_DT, **kwargs)
    elif form == "segments":
        deepwave.scalar(v, grid_spacing, _DT, checkpoint_segments=2, **kwargs)
    else:
        raise AssertionError(form)


def _emit(form: str, category: str) -> list[warnings.WarningMessage]:
    """The warnings of ``category`` that one call in ``form`` emits."""
    extra, grid_spacing, fragment = _CATEGORIES[category]
    _forget_once_per_process_causes()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _call(form, extra, grid_spacing)
    return [w for w in record if fragment in str(w.message)]


@pytest.mark.parametrize("category", sorted(_CATEGORIES))
@pytest.mark.parametrize("form", _FORMS)
def test_warning_names_the_callers_own_line(category: str, form: str) -> None:
    """Each category, in each form, is reported at this file's call."""
    warned = _emit(form, category)
    assert warned, f"{category} was not emitted in the {form} form"
    for w in warned:
        assert w.filename == __file__, (category, form, w.filename, w.lineno)


@pytest.mark.parametrize("category", sorted(_CATEGORIES))
def test_attribution_is_not_free(
    category: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the frame walk, the ``Module`` form blames Deepwave itself.

    Replacing the walk by its fixed fallback is exactly what a hard-coded
    ``stacklevel`` does, so this is the behaviour the walk replaced: for the
    call form the fallback was not tuned for, the warning lands inside
    Deepwave. It makes the assertions above non-vacuous -- they are not
    passing merely because a fixed level happens to be right everywhere.
    """
    monkeypatch.setattr(
        deepwave._exceptions, "stacklevel_outside_deepwave", lambda fallback: fallback
    )
    form = "module" if category != "deprecated_name" else "segments"
    warned = _emit(form, category)
    assert warned, f"{category} was not emitted in the {form} form"
    assert all(w.filename != __file__ for w in warned), [w.filename for w in warned]
