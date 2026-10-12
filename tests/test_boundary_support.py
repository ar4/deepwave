"""Which ``boundary`` faces each propagator accepts, and that the rest raise.

A ``"free"``, ``"rigid"`` or ``"periodic"`` face is implemented by refilling
the guard ring outside the domain with an image of the interior, replacing the
interior stencil rows next to that face. Where those replacement rows are not
the transpose of their partners the scheme conserves no discrete energy and a
closed run grows exponentially (``benchmarks/boundary_energy.py`` measures the
property directly on the kernels). Deepwave therefore accepts a face only
where it conserves, and refuses the rest.

This module pins the rule itself, for every propagator, both face sides, both
the function and the ``torch.nn.Module`` form, in 2-D with a 3-D spot check.
The expected verdict below is written out independently of
``deepwave._common``'s own table, so that a change to that table which was not
meant shows up here rather than agreeing with itself. A refused face must
raise :exc:`deepwave.DeepwaveNotImplementedError` while the arguments are
being validated, which is checked by making :func:`deepwave.plan` -- which
stops before the first time-proportional allocation -- raise it too.
"""

import warnings
from typing import Any

import pytest
import torch
from test_adjoint_dot_product import _grid, _media
from test_capabilities import _positional, _survey_and_backend_kwargs

import deepwave
from deepwave._capabilities import PROPAGATOR_NAMES

#: The ``torch.nn.Module`` form of each propagator.
_MODULES = {
    "scalar": deepwave.Scalar,
    "scalar_born": deepwave.ScalarBorn,
    "acoustic": deepwave.Acoustic,
    "acoustic_born": deepwave.AcousticBorn,
    "viscoacoustic": deepwave.Viscoacoustic,
    "acoustic_vti": deepwave.AcousticVti,
    "elastic": deepwave.Elastic,
    "elastic_vti": deepwave.ElasticVti,
    "elastic_born": deepwave.ElasticBorn,
}

_REGULAR_GRID = ("scalar", "scalar_born")
_ISOTROPIC_ACOUSTIC = ("acoustic", "acoustic_born", "viscoacoustic")
_ELASTIC = (
    "elastic",
    "elastic_vti",
    "elastic_born",
)


def _refused(propagator: str, face: str, high: bool, accuracy: int) -> bool:
    """Whether ``propagator`` refuses ``face`` there, written out by hand."""
    if propagator in _REGULAR_GRID or face == "pml":
        return False
    if face == "periodic":
        return True
    if propagator in _ISOTROPIC_ACOUSTIC:
        return face == "free" and high and accuracy >= 4
    if propagator == "acoustic_vti":
        return face == "free"
    assert propagator in _ELASTIC, propagator
    if face == "free":
        return True
    return high and accuracy >= 4


def _boundary(face: str, high: bool, ndim: int) -> list[str]:
    """``face`` on one face of dimension 0, ``"pml"`` on every other face.

    ``"periodic"`` has to be set on both faces of a dimension, so it ignores
    the side; the rule refuses it on both sides alike.
    """
    boundary = ["pml"] * 2 * ndim
    if face == "periodic":
        boundary[0] = boundary[1] = face
    else:
        boundary[1 if high else 0] = face
    return boundary


def _call_arguments(
    propagator: str, face: str, high: bool, accuracy: int, ndim: int
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The positional and keyword arguments of one minimal call."""
    grid = _grid(ndim)
    media = _media(ndim, grid["n"])
    kwargs = _survey_and_backend_kwargs(propagator, ndim, "auto")
    kwargs["accuracy"] = accuracy
    kwargs["pml_width"] = [0] * 2 * ndim
    kwargs["boundary"] = _boundary(face, high, ndim)
    if propagator == "elastic_vti" and ndim == 3:
        # The horizontal-plane shear stiffness has no 2-D counterpart and is
        # required in 3-D; ``c66 = mu`` recovers isotropy, which is all this
        # test needs of the medium.
        kwargs["c66"] = media["mu"]
    return _positional(propagator, media), kwargs


def _check(propagator: str, face: str, high: bool, accuracy: int, ndim: int) -> None:
    """Runs or refuses one configuration, through all three entry points."""
    positional, kwargs = _call_arguments(propagator, face, high, accuracy, ndim)
    function = getattr(deepwave, propagator)
    module_kwargs = dict(kwargs)
    dt = module_kwargs.pop("dt")
    # The Module form holds the models: a perturbation model and the tilt are
    # constructor arguments there, while the function takes them per call.
    module_arguments = {
        name: module_kwargs.pop(name)
        for name in ("tilt", "c66", "dv", "drho", "dlamb", "dmu", "dbuoyancy")
        if name in module_kwargs
    }
    module = _MODULES[propagator](*positional, **module_arguments)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if _refused(propagator, face, high, accuracy):
            with pytest.raises(deepwave.DeepwaveNotImplementedError):
                function(*positional, **kwargs)
            # ``plan`` stops before the first time-proportional allocation,
            # so a refusal it also raises is a refusal made while the
            # arguments were being validated, before anything was computed.
            with pytest.raises(deepwave.DeepwaveNotImplementedError):
                deepwave.plan(propagator, *positional, **kwargs)
            with pytest.raises(deepwave.DeepwaveNotImplementedError):
                module(dt, **module_kwargs)
        else:
            out = function(*positional, **kwargs)
            assert torch.isfinite(out[-1]).all()
            assert torch.isfinite(module(dt, **module_kwargs)[-1]).all()


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
@pytest.mark.parametrize("face", ["pml", "free", "rigid", "periodic"])
@pytest.mark.parametrize("high", [False, True], ids=["low", "high"])
@pytest.mark.parametrize("accuracy", [2, 4])
def test_boundary_face_support_2d(
    name: str, face: str, high: bool, accuracy: int
) -> None:
    """Every face, both sides, accuracy 2 and 4: accepted, or refused."""
    _check(name, face, high, accuracy, 2)


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
@pytest.mark.parametrize(
    ("face", "high"), [("rigid", False), ("rigid", True), ("free", True)]
)
def test_boundary_face_support_3d(name: str, face: str, high: bool) -> None:
    """A 3-D spot check at accuracy 4, where the side matters."""
    if 3 not in deepwave.spec(name).ndims:
        pytest.skip(f"{name} has no 3-D implementation")
    _check(name, face, high, 4, 3)


def test_a_refused_face_names_the_alternative() -> None:
    """The message says what is unavailable, why, and what to do instead."""
    positional, kwargs = _call_arguments("elastic", "free", False, 4, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(
            deepwave.DeepwaveNotImplementedError,
            match="does not conserve a discrete energy",
        ) as excinfo:
            deepwave.elastic(*positional, **kwargs)
    message = str(excinfo.value)
    assert "'free'" in message
    assert "vacuum layer" in message


def test_a_refused_face_is_also_a_not_implemented_error() -> None:
    """``except NotImplementedError`` keeps catching it."""
    positional, kwargs = _call_arguments("elastic", "periodic", False, 2, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotImplementedError, match="scalar"):
            deepwave.elastic(*positional, **kwargs)
