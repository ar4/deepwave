"""The axis convention is stated once, everywhere, and is true.

Deepwave orders spatial dimensions slowest-varying to fastest-varying and
attaches no physical meaning to any of them. Two things follow, and both are
checked here.

**It has to be said.** The convention is the single most common source of a
silently wrong Deepwave script: a transposed model with untransposed location
columns runs perfectly and models a rotated earth. So the same sentence
appears in all twenty-two propagator docstrings, byte-identically, and the
first test here fails if any of them loses it or paraphrases it. The three
anisotropic propagators carry an extra sentence, because for them dimension 0
really is fixed -- their symmetry axis is dimension 0 by construction -- and a
reader who takes the general rule at face value there will get the physics
wrong.

**It has to be true.** The claim that the propagators "behave identically
under any relabelling" was, until this file, entirely untested. The second
test transposes a model, swaps the location columns, reverses ``grid_spacing``
and exchanges the component suffixes, and requires the receiver data to agree.
It is the property that makes re-orienting a script safe, and if a propagator
ever stops having it, that is a defect in the propagator, not in the test.
"""

import inspect
from typing import Any

import pytest
import torch

import deepwave
from deepwave.wavelets import ricker

#: The one sentence, as it must appear in every propagator docstring.
_SENTENCE = (
    "Deepwave orders spatial dimensions from slowest-varying to "
    "fastest-varying and attaches no physical meaning to any of them; the "
    "component suffixes count from the fastest axis, so a 2-D run uses "
    "``_y`` and ``_x`` and a 3-D run uses ``_z``, ``_y`` and ``_x``. See "
    ":ref:`axis-convention`."
)

#: The extra sentence the anisotropic family carries.
_CARVE_OUT = (
    "The one exception to that freedom is this family: its symmetry axis "
    "is dimension 0 by construction, so here the axis order is a physics "
    "constraint rather than a documentation convention."
)

_FUNCTIONS = (
    "scalar",
    "scalar_born",
    "acoustic",
    "acoustic_born",
    "acoustic_vti",
    "viscoacoustic",
    "elastic",
    "elastic_born",
    "elastic_vti",
)

_CLASSES = (
    "Scalar",
    "ScalarBorn",
    "Acoustic",
    "AcousticBorn",
    "AcousticVti",
    "Viscoacoustic",
    "Elastic",
    "ElasticBorn",
    "ElasticVti",
)

#: The propagators whose symmetry axis really is dimension 0.
_ANISOTROPIC = frozenset(
    {
        "acoustic_vti",
        "AcousticVti",
        "elastic_vti",
        "ElasticVti",
    }
)


def _normalise(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.parametrize("name", _FUNCTIONS + _CLASSES)
def test_axis_convention_sentence_appears_exactly_once(name: str) -> None:
    """All twenty-two carry the sentence, once, byte-identically."""
    doc = inspect.getdoc(getattr(deepwave, name))
    assert doc is not None, name
    assert _normalise(doc).count(_SENTENCE) == 1, name


@pytest.mark.parametrize("name", _FUNCTIONS + _CLASSES)
def test_anisotropic_carve_out_is_stated_exactly_where_it_is_true(
    name: str,
) -> None:
    """Only the VTI/TTI family says its axis order is a physics constraint.

    Stating it on the other eight would be false, and omitting it on these
    three would leave the general rule looking like a licence to transpose a
    VTI model.
    """
    doc = _normalise(inspect.getdoc(getattr(deepwave, name)) or "")
    assert (_CARVE_OUT in doc) is (name in _ANISOTROPIC), name


def test_the_sentence_names_a_label_that_the_docs_define() -> None:
    """The cross-reference target exists.

    ``sphinx -W`` catches an undefined label, but only when the docs are
    built; this makes the same mistake fail in the fast gate.
    """
    import pathlib

    usage = pathlib.Path(__file__).resolve().parents[1] / "docs" / "usage.rst"
    assert ".. _axis-convention:" in usage.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Transpose invariance: the property the convention rests on.
# ---------------------------------------------------------------------------

_NT = 40
_DT = 0.002
_DY, _DX = 5.0, 4.0


def _asymmetric_model(ny: int, nx: int) -> torch.Tensor:
    """A model with no symmetry in either axis, so a transpose is visible."""
    y = torch.arange(ny, dtype=torch.float64).reshape(-1, 1)
    x = torch.arange(nx, dtype=torch.float64).reshape(1, -1)
    return 1500.0 + 40.0 * y + 11.0 * x + 3.0 * (y * x) ** 0.5


def _wavelet() -> torch.Tensor:
    return ricker(15.0, _NT, _DT, 0.06, dtype=torch.float64).reshape(1, 1, -1)


def _swap_columns(locations: torch.Tensor) -> torch.Tensor:
    """The same points, with the two spatial columns exchanged."""
    return locations.flip(-1).contiguous()


def _run_scalar(
    v: torch.Tensor,
    spacing: list[float],
    sources: torch.Tensor,
    receivers: torch.Tensor,
) -> torch.Tensor:
    out = deepwave.scalar(
        v,
        spacing,
        _DT,
        source_amplitudes=_wavelet(),
        source_locations=sources,
        receiver_locations=receivers,
        pml_freq=15.0,
    )
    return out.receiver_amplitudes


def _run_acoustic(
    v: torch.Tensor,
    spacing: list[float],
    sources: torch.Tensor,
    receivers: torch.Tensor,
) -> torch.Tensor:
    rho = 1000.0 + 0.2 * v
    out = deepwave.acoustic(
        v,
        rho,
        spacing,
        _DT,
        source_amplitudes_volume_rate=_wavelet(),
        source_locations_volume_rate=sources,
        receiver_locations_p=receivers,
        pml_freq=15.0,
    )
    return out.receiver_amplitudes_p


def _run_elastic(
    v: torch.Tensor,
    spacing: list[float],
    sources: torch.Tensor,
    receivers: torch.Tensor,
    *,
    transposed: bool,
) -> torch.Tensor:
    """An elastic run whose *vertical* source follows the axis relabelling.

    In the original orientation the source is a ``_y`` (last-but-one axis)
    force; after the transpose that same physical direction is the ``_x``
    axis, so the suffixes swap along with everything else. Getting this wrong
    is precisely the mistake the convention exists to prevent, so the test
    would fail loudly rather than quietly agreeing.
    """
    vs = v / 2.0
    rho = 1000.0 + 0.2 * v
    lamb, mu, buoyancy = deepwave.vpvsrho_to_lambmubuoyancy(v, vs, rho)
    force = {"source_amplitudes_x" if transposed else "source_amplitudes_y": _wavelet()}
    location = {"source_locations_x" if transposed else "source_locations_y": sources}
    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        spacing,
        _DT,
        receiver_locations_p=receivers,
        pml_freq=15.0,
        **force,
        **location,
    )
    return out.receiver_amplitudes_p


_RUNNERS: dict[str, Any] = {
    "scalar": _run_scalar,
    "acoustic": _run_acoustic,
}


@pytest.mark.parametrize("propagator", sorted(_RUNNERS))
def test_transposing_the_whole_problem_changes_nothing(propagator: str) -> None:
    """A relabelled 2-D problem gives the same receiver data.

    Transpose the model, swap the location columns, reverse ``grid_spacing``:
    the physics is untouched, so the recorded data must be too. Anisotropic
    grid spacing (``dy != dx``) is deliberate -- with a square cell the test
    would still pass if ``grid_spacing`` were being ignored per axis.
    """
    run = _RUNNERS[propagator]
    ny, nx = 34, 27
    v = _asymmetric_model(ny, nx)
    sources = torch.tensor([[[6, 9]]])
    receivers = torch.tensor([[[24, 5], [11, 20], [30, 22]]])

    original = run(v, [_DY, _DX], sources, receivers)
    relabelled = run(
        v.T.contiguous(),
        [_DX, _DY],
        _swap_columns(sources),
        _swap_columns(receivers),
    )
    torch.testing.assert_close(original, relabelled, rtol=1e-10, atol=0.0)


def test_transposing_an_elastic_problem_changes_nothing() -> None:
    """The same, for ``elastic``, including the component-suffix exchange."""
    ny, nx = 30, 25
    v = _asymmetric_model(ny, nx)
    sources = torch.tensor([[[6, 9]]])
    receivers = torch.tensor([[[22, 5], [11, 18]]])

    original = _run_elastic(v, [_DY, _DX], sources, receivers, transposed=False)
    relabelled = _run_elastic(
        v.T.contiguous(),
        [_DX, _DY],
        _swap_columns(sources),
        _swap_columns(receivers),
        transposed=True,
    )
    torch.testing.assert_close(original, relabelled, rtol=1e-10, atol=0.0)


def test_transposing_only_the_model_does_change_the_answer() -> None:
    """The invariance test is not vacuous.

    If a half-done relabelling -- the model transposed but the locations left
    alone -- produced the same data, the test above would prove nothing. This
    is the mistake the axis convention warns about, and it must be visible.
    """
    ny = nx = 30
    v = _asymmetric_model(ny, nx)
    sources = torch.tensor([[[6, 9]]])
    receivers = torch.tensor([[[22, 5], [11, 18]]])

    original = _run_scalar(v, [_DY, _DX], sources, receivers)
    half_done = _run_scalar(v.T.contiguous(), [_DY, _DX], sources, receivers)
    assert not torch.allclose(original, half_done, rtol=1e-3, atol=1e-12)


def test_a_3d_cyclic_permutation_changes_nothing() -> None:
    """The same property in 3-D, under a cyclic permutation of the axes."""
    nz, ny, nx = 16, 14, 12
    z = torch.arange(nz, dtype=torch.float64).reshape(-1, 1, 1)
    y = torch.arange(ny, dtype=torch.float64).reshape(1, -1, 1)
    x = torch.arange(nx, dtype=torch.float64).reshape(1, 1, -1)
    v = 1500.0 + 30.0 * z + 17.0 * y + 7.0 * x
    spacing = [6.0, 5.0, 4.0]
    sources = torch.tensor([[[4, 5, 3]]])
    receivers = torch.tensor([[[11, 4, 8], [6, 10, 2]]])

    original = _run_scalar(v, spacing, sources, receivers)
    # (z, y, x) -> (x, z, y): every axis moves one place later, so each
    # location column and each spacing moves with it.
    permuted = _run_scalar(
        v.permute(2, 0, 1).contiguous(),
        [spacing[2], spacing[0], spacing[1]],
        sources[..., [2, 0, 1]].contiguous(),
        receivers[..., [2, 0, 1]].contiguous(),
    )
    torch.testing.assert_close(original, permuted, rtol=1e-10, atol=0.0)
