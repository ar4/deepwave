"""``torch.func.vmap`` over each model argument of each propagator.

``deepwave._capabilities``'s Batching column probes ``vmap`` over one model
per propagator -- the first positional one -- which is not enough to support
the claim ``docs/capabilities.rst`` makes about a propagator's *models*. A
model that enters the time loop by a different route can fail where the
first one succeeds: ``elastic_vti``'s ``c55`` reaches only the shear
stresses, so it batched a term that was then accumulated in place into an
unbatched one built from the normal stresses, and raised

    RuntimeError: output with shape [1, 35, 35] doesn't match the
    broadcast shape [3, 1, 35, 35]

while ``c11`` (which does enter the normal stresses) worked.

This module therefore pins the whole matrix: every propagator times every
one of its model arguments, one at a time, on the eager backend. Each case
is checked for more than "it did not raise" -- the batched result must equal
the loop over the same three models element for element, and the three
entries must actually differ, so a propagator that silently ignored the
batching could not pass. The two Born propagators that reject ``vmap`` over
any model are pinned to their ``NotImplementedError`` rather than skipped,
so that a change either way is visible here.
"""

from typing import Any, Callable

import pytest
import torch
from test_adjoint_dot_product import _grid, _media, _survey
from torch.func import vmap

import deepwave
from deepwave._capabilities import PROPAGATOR_NAMES

DTYPE = torch.float64

#: The positional model arguments of each propagator in 2-D, in call order,
#: named by their key in ``test_adjoint_dot_product._media``. Written out
#: here rather than derived from the package, so that adding a model to a
#: propagator has to be a deliberate edit in this file too.
_POSITIONAL_MODELS = {
    "scalar": ("v",),
    "scalar_born": ("v",),
    "acoustic": ("v", "rho"),
    "acoustic_born": ("v", "rho"),
    "viscoacoustic": ("v", "rho", "q"),
    "acoustic_vti": ("c11", "c13", "c33", "buoyancy"),
    "elastic": ("lamb", "mu", "buoyancy"),
    "elastic_born": ("lamb", "mu", "buoyancy"),
    "elastic_vti": ("c11", "c13", "c33", "c55", "buoyancy"),
}

#: The remaining model arguments, which are passed by keyword.
_KEYWORD_MODELS = {
    "scalar_born": ("dv",),
    "acoustic_born": ("dv", "drho"),
    "elastic_born": ("dlamb", "dmu", "dbuoyancy"),
}

#: The propagators that reject ``vmap`` over any of their models outright.
_REJECT_VMAP = ("acoustic_born", "elastic_born")

#: The channel suffix these propagators' survey keywords carry, where it is
#: not derived from the spec. The two scalar propagators have no suffix at
#: all, and ``acoustic_vti``'s scalar channels are synthesised, so its x
#: velocity is used on both sides instead.
_SRC_PREFIX = {"scalar": "", "scalar_born": "", "acoustic_vti": "_x"}


def _model_names(name: str) -> tuple[str, ...]:
    """Every model argument of ``name`` in 2-D, positional ones first."""
    return _POSITIONAL_MODELS[name] + _KEYWORD_MODELS.get(name, ())


def _survey_kwargs(name: str, ndim: int) -> dict[str, Any]:
    """A minimal valid keyword set for ``name`` on the eager backend."""
    g = _grid(ndim)
    n, nt, dt = g["n"], g["nt"], g["dt"]
    src, src_loc, rec_loc = _survey(ndim, n, nt, dt)
    spec = deepwave.spec(name)
    prefix = _SRC_PREFIX.get(name, f"_{spec.source_channels(ndim)[0]}")
    receiver_prefix = _SRC_PREFIX.get(name, f"_{spec.receiver_channels(ndim)[0]}")
    kwargs: dict[str, Any] = {
        "dt": dt,
        "pml_freq": 25.0,
        "accuracy": g["acc"],
        "pml_width": 4,
        # Explicit rather than read from the (batched, and so extremum-less)
        # models -- see deepwave._common.velocity_extremes.
        "max_vel": 2500.0,
        "backend": "eager",
        f"source_amplitudes{prefix}": src,
        f"source_locations{prefix}": src_loc,
    }
    sc_prefix = "sc_" if spec.has_bg_receivers else ""
    kwargs[f"{sc_prefix}receiver_locations{receiver_prefix}"] = rec_loc
    if spec.has_bg_receivers:
        bg = "bg_receiver_locations" + ("" if name == "scalar_born" else "_p")
        kwargs[bg] = rec_loc
    if name in ("viscoacoustic",):
        kwargs["q_freq"] = 25.0
    return kwargs


#: ``test_adjoint_dot_product._media`` builds the stiffnesses from the
#: isotropic Lame parameters, which makes the medium isotropic and therefore
#: rotation-invariant. These factors shift the VTI stiffnesses away from
#: the isotropic point while keeping the medium
#: physical (``c11 c33 > c13**2``, ``c33 > c55``) and well inside the
#: ``max_vel`` this module passes.
_ANISOTROPY = {"c11": 1.3, "c13": 0.9, "c66": 1.15}


def _models(name: str, ndim: int) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    """The positional and keyword model tensors of ``name``."""
    m = _media(ndim, _grid(ndim)["n"])
    positional = [
        m[key].clone() * _ANISOTROPY.get(key, 1.0) for key in _POSITIONAL_MODELS[name]
    ]
    keyword = {}
    for key in _KEYWORD_MODELS.get(name, ()):
        # A non-zero scattering model, so that a Born propagator's scattered
        # receiver data is not identically zero.
        keyword[key] = m["v"].clone() * 1e-3
    return positional, keyword


def _receiver(out: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """The last non-empty output: a receiver channel that was asked for.

    Every propagator returns one entry per receiver channel it defines, and
    the channels not requested come back empty, so the last non-empty entry
    is the single channel this module's survey asks for.
    """
    return next(o for o in reversed(out) if o.numel() > 0)


def _make_call(
    name: str, model: str, ndim: int
) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]:
    """``(f, batch)``: ``f`` runs ``name`` with ``model`` replaced by its argument."""
    positional, keyword = _models(name, ndim)
    kwargs = _survey_kwargs(name, ndim)
    kwargs.update(keyword)
    grid_spacing = 5.0 if ndim == 2 else [5.0] * ndim
    fn = getattr(deepwave, name)
    index = (
        _POSITIONAL_MODELS[name].index(model)
        if model in _POSITIONAL_MODELS[name]
        else None
    )

    def f(x: torch.Tensor) -> torch.Tensor:
        args = list(positional)
        call_kwargs = dict(kwargs)
        if index is None:
            call_kwargs[model] = x
        else:
            args[index] = x
        return _receiver(fn(*args, grid_spacing, **call_kwargs))

    target = positional[index] if index is not None else keyword[model]
    if name == "viscoacoustic" and model == "rho":
        # With a direct pressure-rate source, multiplying the *entire*
        # density field by one constant leaves the pressure equation
        # unchanged: K=rho*v**2 and 1/rho cancel. Probe density's spatial
        # effect instead, so the required nonzero spread has physical meaning.
        depth = torch.linspace(-1, 1, target.shape[0], dtype=target.dtype).reshape(
            -1, *([1] * (target.ndim - 1))
        )
        batch = torch.stack(
            [target, target * (1 + 0.1 * depth), target * (1 + 0.2 * depth)]
        )
    else:
        batch = torch.stack([target, target * 1.001, target * 1.002])
    return f, batch


def _check_vmap_over_one_model(name: str, model: str, ndim: int = 2) -> None:
    """``vmap`` over ``model`` runs and agrees with the equivalent loop."""
    f, batch = _make_call(name, model, ndim)
    if name in _REJECT_VMAP:
        with pytest.raises(NotImplementedError, match="vmap"):
            vmap(f)(batch)
        return
    batched = vmap(f)(batch)
    looped = torch.stack([f(entry) for entry in batch])
    scale = float(looped.abs().max())
    assert scale > 0.0, f"{name}/{model}: the probe recorded nothing"
    spread = float((looped[0] - looped[-1]).abs().max())
    assert spread > 1e-6 * scale, (
        f"{name}/{model}: the three models give the same data, so this case "
        f"would pass even if the batching were ignored (spread {spread}, "
        f"scale {scale})"
    )
    torch.testing.assert_close(batched, looped, rtol=0.0, atol=1e-12 * scale)


@pytest.mark.parametrize(
    ("name", "model"),
    [(name, model) for name in PROPAGATOR_NAMES for model in _model_names(name)],
)
def test_vmap_over_each_model_2d(name: str, model: str) -> None:
    """Each propagator's each model argument, batched on its own, in 2-D."""
    _check_vmap_over_one_model(name, model)


@pytest.mark.parametrize("model", ["c11", "c13", "c33", "c55", "buoyancy", "c66"])
def test_vmap_over_each_elastic_vti_model_3d(model: str) -> None:
    """``elastic_vti`` in 3-D, where ``c66`` is a model argument as well.

    ``c55`` and ``c66`` are the two models that reach only the shear
    stresses, and in 3-D they reach different ones, so both are batched here.
    """
    positional, _ = _models("elastic_vti", 3)
    kwargs = _survey_kwargs("elastic_vti", 3)
    kwargs["c66"] = _media(3, _grid(3)["n"])["c66"].clone() * _ANISOTROPY["c66"]

    def f(x: torch.Tensor) -> torch.Tensor:
        args = list(positional)
        call_kwargs = dict(kwargs)
        if model == "c66":
            call_kwargs["c66"] = x
        else:
            args[_POSITIONAL_MODELS["elastic_vti"].index(model)] = x
        return _receiver(deepwave.elastic_vti(*args, [5.0] * 3, **call_kwargs))

    target = (
        kwargs["c66"]
        if model == "c66"
        else positional[_POSITIONAL_MODELS["elastic_vti"].index(model)]
    )
    batch = torch.stack([target, target * 1.001, target * 1.002])
    batched = vmap(f)(batch)
    looped = torch.stack([f(entry) for entry in batch])
    scale = float(looped.abs().max())
    assert scale > 0.0, model
    torch.testing.assert_close(batched, looped, rtol=0.0, atol=1e-12 * scale)
