"""Tests for propagations that have no sources and no receivers.

A call may supply nothing but models, a grid, a time step and initial
wavefields: the wavefields already in the domain then evolve on their own.
This is what time-chaining and domain decomposition do at every segment
boundary, and what a "let this state continue" call is. The initial
wavefields are then the only thing that can say how many spatial dimensions
the propagation has, and how many shots it has, so every propagator must
read the dimensionality from them -- whether they arrive as explicit
wavefield arguments or inside an ``initial_state`` bundle.
"""

import warnings
from typing import Any, Optional

import pytest
import torch

import deepwave
from deepwave._capabilities import PROPAGATOR_NAMES, _models

DTYPE = torch.float64
N = 10
DX = 5.0
DT = 0.0005
NT = 2
NDIM = 2


def _extras(name: str) -> dict[str, Any]:
    """Arguments a propagator needs beyond its models, grid and time step.

    The Born perturbations (one per pair is enough to make the call valid)
    and the reference frequency the attenuating propagator requires.
    Deliberately no sources and no receivers: that is what these tests are about.
    """
    shape = (N,) * NDIM
    zeros = torch.zeros(shape, dtype=DTYPE)
    if name == "scalar_born":
        return {"dv": zeros}
    if name == "acoustic_born":
        return {"dv": zeros}
    if name == "elastic_born":
        return {"dlamb": zeros}
    if name in ("viscoacoustic",):
        return {"q_freq": 15.0}
    return {}


def _wavefields(name: str) -> dict[str, torch.Tensor]:
    """A zero initial state for ``name`` with one non-zero cell.

    Keyed by the public state names, which are also the names of the
    propagator's initial-wavefield arguments.
    """
    fields = {
        field: torch.zeros(1, *(N,) * NDIM, dtype=DTYPE)
        for field in deepwave.wavefield_names(name, NDIM)
        if field != "auxiliary_state"
    }
    first = next(iter(fields))
    fields[first][0, N // 2, N // 2] = 1.0
    return fields


def _run(name: str, mode: str) -> Any:
    """Propagate ``name`` from initial wavefields alone."""
    wavefields = _wavefields(name)
    extra: dict[str, Any] = dict(_extras(name))
    if mode == "arguments":
        extra.update(wavefields)
    else:
        extra["initial_state"] = wavefields
    with warnings.catch_warnings():
        # The survey extents are ambiguous for a bare wavefield whose shape
        # matches the model; either reading is a valid propagation here.
        warnings.simplefilter("ignore")
        return getattr(deepwave, name)(
            *_models(name, NDIM),
            DX,
            dt=DT,
            nt=NT,
            max_vel=4000.0,
            pml_freq=15.0,
            pml_width=3,
            **extra,
        )


@pytest.mark.parametrize("mode", ["arguments", "initial_state"])
@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_sourceless_call_infers_ndim(name: str, mode: str) -> None:
    """Initial wavefields alone are enough to propagate, on every propagator.

    With no source and no receiver anywhere in the call, the number of
    spatial dimensions comes from the initial wavefields. Both spellings of
    those wavefields -- one keyword argument per field, or the whole state as
    an ``initial_state`` bundle -- must be read, because a propagator that
    reads only one of them rejects a perfectly determined call with
    "The number of spatial dimensions could not be determined".
    """
    out = _run(name, mode)
    assert out.info.ndim == NDIM
    returned: list[torch.Tensor] = []
    for field in deepwave.wavefield_names(name, NDIM):
        if field == "auxiliary_state":
            continue
        value: Optional[torch.Tensor] = getattr(out, field)
        assert value is not None
        assert value.shape[0] == 1
        assert torch.isfinite(value).all()
        returned.append(value)
    assert any(bool(value.abs().max() > 0) for value in returned)


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_sourceless_call_agrees_between_spellings(name: str) -> None:
    """The two spellings of the same initial state propagate identically."""
    via_arguments = _run(name, "arguments")
    via_bundle = _run(name, "initial_state")
    for field in deepwave.wavefield_names(name, NDIM):
        if field == "auxiliary_state":
            continue
        assert torch.equal(getattr(via_arguments, field), getattr(via_bundle, field))
