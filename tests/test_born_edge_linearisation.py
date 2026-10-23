"""The Born propagators linearise their background propagator everywhere.

A Born propagator computes the derivative of its background propagator with
respect to the models, applied to the perturbation it is given. That is a
statement about the whole model, including its outermost cells: a
perturbation that lives only on the edge of the model is extended outwards
into the padding (the absorbing layer and the finite-difference halo) exactly
as the background models are, so the perturbed and unperturbed runs differ
inside the padding too, and the linearisation has to account for it.

This file pins that for all three Born propagators, against
:func:`torch.func.jvp` of the corresponding background propagator, which is
the linearisation computed independently. The perturbation is non-zero
**only** in the outermost rows of the model, which is the case a Born
propagator that zeroed its perturbation in the padding would get wrong: the
equivalence then fails by a large fraction of the signal rather than by
rounding error.

Both Python-visible backends are checked, because the padding is applied
before the backend is chosen and neither may drop it.
"""

from typing import Any

import pytest
import torch
from torch.func import jvp

import deepwave

_N = 16
_NT = 24
_DX = 5.0
_DT = 0.0005
_FREQ = 25.0
_DTYPE = torch.double

#: A representative value for each background model, chosen so that every
#: propagator here runs on a physically sensible medium.
_BASE: dict[str, float] = {
    "v": 1500.0,
    "rho": 2200.0,
    "lamb": 2.0e9,
    "mu": 1.0e9,
    "buoyancy": 1.0 / 2200.0,
}

#: Each Born propagator, its background propagator, and the background models
#: whose perturbations it takes (the perturbation of ``m`` is spelled ``dm``).
_CASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "scalar_born": ("scalar", ("v",)),
    "acoustic_born": ("acoustic", ("v", "rho")),
    "elastic_born": ("elastic", ("lamb", "mu", "buoyancy")),
}


def _edge_perturbation(name: str, generator: torch.Generator) -> torch.Tensor:
    """A perturbation of ``name`` that is non-zero only in the outermost rows."""
    perturbation = torch.zeros(_N, _N, dtype=_DTYPE)
    for row in (0, -1):
        perturbation[row] = (
            torch.randn(_N, dtype=_DTYPE, generator=generator) * _BASE[name] * 1e-2
        )
    return perturbation


def _config() -> dict[str, Any]:
    """The resolved-parameter arguments, pinned so the two calls agree."""
    return {
        "accuracy": 2,
        "pml_width": 5,
        "pml_freq": _FREQ,
        "max_vel": 3000.0,
    }


@pytest.mark.parametrize("born", sorted(_CASES))
@pytest.mark.parametrize("backend", ["eager", "compiled"])
def test_born_is_the_linearisation_for_an_edge_perturbation(
    born: str, backend: str
) -> None:
    """Born equals the JVP of its background propagator for an edge perturbation.

    The scattered receiver amplitudes must equal the forward-mode derivative
    of the background propagator's receiver amplitudes, and the background
    receiver amplitudes the undifferentiated value, to rounding error. A Born
    propagator that replaced its perturbation with zeros in the padding fails
    the first of these by a large fraction of the signal.
    """
    base, model_names = _CASES[born]
    base_spec = deepwave.spec(base)
    born_spec = deepwave.spec(born)
    source_channel = base_spec.source_channels(2)[0]
    receiver_channel = base_spec.receiver_channels(2)[0]
    n_receiver_channels = len(base_spec.receiver_channels(2))

    generator = torch.Generator().manual_seed(0)
    models = tuple(
        torch.full((_N, _N), _BASE[name], dtype=_DTYPE) for name in model_names
    )
    perturbations = tuple(_edge_perturbation(name, generator) for name in model_names)

    amplitudes = (
        deepwave.wavelets.ricker(_FREQ, _NT, _DT, 0.05).to(_DTYPE).reshape(1, 1, -1)
    )
    source_locations = torch.tensor([[[_N // 2, _N // 2]]])
    receiver_locations = torch.tensor([[[_N // 2, _N // 2 + 3]]])

    base_kwargs: dict[str, Any] = {
        base_spec.source_amplitudes_kwarg(source_channel): amplitudes,
        base_spec.source_locations_kwarg(source_channel): source_locations,
        base_spec.receiver_locations_kwarg(receiver_channel): receiver_locations,
        **_config(),
    }

    def background(*tensors: torch.Tensor) -> torch.Tensor:
        out = getattr(deepwave, base)(
            *tensors, _DX, _DT, backend="eager", **base_kwargs
        )
        return out[-n_receiver_channels]

    primal, tangent = jvp(background, models, perturbations)

    born_kwargs: dict[str, Any] = {
        born_spec.source_amplitudes_kwarg(source_channel): amplitudes,
        born_spec.source_locations_kwarg(source_channel): source_locations,
        born_spec.receiver_locations_kwarg(receiver_channel): receiver_locations,
        born_spec.bg_receiver_locations_kwarg(receiver_channel): receiver_locations,
        **_config(),
    }
    for name, perturbation in zip(model_names, perturbations):
        born_kwargs["d" + name] = perturbation

    out = getattr(deepwave, born)(*models, _DX, _DT, backend=backend, **born_kwargs)
    scattered = out[-n_receiver_channels]
    background_receivers = out[-2 * n_receiver_channels]

    assert tangent.abs().max() > 0
    # The eager backend reproduces the derivative to the last bit; the
    # compiled C backend is built with fast-math, so it is held to a
    # backend-consistency tolerance rather than to machine precision.
    tolerance = 1e-13 if backend == "eager" else 1e-7
    assert (scattered - tangent).abs().max() <= tolerance * tangent.abs().max()
    assert (background_receivers - primal).abs().max() <= (
        tolerance * primal.abs().max()
    )
