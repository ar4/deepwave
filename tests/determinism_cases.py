"""Uniform propagator cases for the determinism tests.

One builder drives all nine propagators from their specs, so the
determinism tests cannot quietly stop covering a propagator: the case list
is ``PROPAGATOR_NAMES`` itself.

Every case requests a gradient for every model, every source amplitude and
every initial wavefield, and the loss reads every output tensor, so a
difference anywhere in the forward or the backward shows up.

The builder is kept out of ``test_determinism.py`` so that it can also be
driven directly from a script when investigating a failure.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, NamedTuple

import torch
from helpers import output_tensors

import deepwave

#: The nine propagators.
PROPAGATORS: tuple[str, ...] = deepwave._engine.PROPAGATOR_NAMES

#: The shared grid: 64 cells square, with a 10-cell PML on every face.
N = 64
PML_WIDTH = 10
DX = 5.0
DT = 0.001
NT = 300
FREQ = 25.0
N_SHOTS = 2


def _models(name: str, dtype: torch.dtype, device: torch.device) -> list[torch.Tensor]:
    """A physically sensible model set for ``name``, one tensor per model name.

    The values are smooth rather than random: a determinism test wants the
    same arithmetic every run, and a heterogeneous but well-conditioned
    medium keeps every propagator stable for ``NT`` steps.
    """
    kwargs: dict[str, Any] = {"dtype": dtype, "device": device}
    ramp = torch.linspace(0.0, 1.0, N, **kwargs)
    field = ramp.reshape(-1, 1) + 0.5 * ramp.reshape(1, -1)
    v = 1500.0 + 500.0 * field
    vs = v / 2.0
    rho = 1800.0 + 400.0 * field
    buoyancy = 1.0 / rho
    lamb = rho * (v**2 - 2.0 * vs**2)
    mu = rho * vs**2
    q = 40.0 + 20.0 * field
    c33 = rho * v**2
    c11 = 1.1 * c33
    c55 = rho * vs**2
    c13 = c33 - 2.0 * c55
    tilt = 0.1 + 0.0 * field
    per_name: dict[str, torch.Tensor] = {
        "v": v,
        "rho": rho,
        "q": q,
        "qp": q,
        "qs": q,
        "lamb": lamb,
        "mu": mu,
        "buoyancy": buoyancy,
        "c11": c11,
        "c13": c13,
        "c33": c33,
        "c55": c55,
        "tilt": tilt,
        # Born perturbations: a small, smooth scatterer.
        "dv": 0.01 * v,
        "drho": 0.01 * rho,
        "dlamb": 0.01 * lamb,
        "dmu": 0.01 * mu,
        "dbuoyancy": 0.01 * buoyancy,
    }
    spec = deepwave.spec(name)
    return [per_name[model].clone() for model in spec.model_names(2)]


def _wavefield_scale(field: str) -> float:
    """A scale that keeps every initial field's contribution representable.

    Stresses are moduli-scaled (~1e9 Pa for a strain of 1), velocities are
    not, so one scale for both would lose the smaller field to float32
    rounding and the test would stop exercising its gradient.
    """
    if field.startswith(("sigma", "r_mech", "rp_", "rs_")):
        return 1.0e3
    if field.startswith(("pressure", "wavefield")):
        return 1.0e3
    return 1.0e-3


def build_case(
    name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    *,
    backend: str = "auto",
    nt: int = NT,
) -> tuple[dict[str, Any], list[torch.Tensor], list[str]]:
    """Assemble one propagator call.

    Returns the keyword arguments (including the positional models under
    ``"_models"``), the list of leaf tensors whose gradients the test
    compares, and their names.
    """
    spec = deepwave.spec(name)
    function = getattr(deepwave, name)
    parameters = inspect.signature(function).parameters
    # Everything before ``grid_spacing`` is positional; the Born
    # perturbations are keyword-only.
    positional_names = list(parameters)[: list(parameters).index("grid_spacing")]

    models = _models(name, dtype, device)
    leaves: list[torch.Tensor] = []
    leaf_names: list[str] = []
    positional: list[torch.Tensor] = []
    keyword_models: dict[str, torch.Tensor] = {}
    for model, model_name in zip(models, spec.model_names(2)):
        model.requires_grad_(True)
        leaves.append(model)
        leaf_names.append(f"model:{model_name}")
        if model_name in positional_names:
            positional.append(model)
        else:
            keyword_models[model_name] = model

    generator = torch.Generator(device="cpu").manual_seed(1234)
    kwargs: dict[str, Any] = {
        "pml_width": PML_WIDTH,
        "pml_freq": FREQ,
        "backend": backend,
        "max_vel": 3000.0,
    }
    kwargs.update(keyword_models)
    if name in ("viscoacoustic",):
        kwargs["q_freq"] = FREQ

    wavelet = deepwave.wavelets.ricker(FREQ, nt, DT, 1.5 / FREQ).to(dtype)
    receiver_locations = torch.tensor(
        [[[N // 2 + 6, N // 2 + 2 * i] for i in range(4)]] * N_SHOTS,
        device=device,
    )
    # One cell per channel: two sources of different channels may share a
    # cell, but two of the same channel may not, and giving every channel
    # its own cell keeps one rule for all nine.
    for index, channel in enumerate(spec.source_channels(2)):
        locations = torch.tensor(
            [[[N // 2 + index, N // 2 + 4 * shot]] for shot in range(N_SHOTS)],
            device=device,
        )
        amplitudes = wavelet.reshape(1, 1, -1).repeat(N_SHOTS, 1, 1).to(device)
        amplitudes = amplitudes.clone().detach().requires_grad_(True)
        kwargs[spec.source_amplitudes_kwarg(channel)] = amplitudes
        kwargs[spec.source_locations_kwarg(channel)] = locations
        leaves.append(amplitudes)
        leaf_names.append(f"source:{channel}")
    for channel in spec.receiver_channels(2):
        kwargs[spec.receiver_locations_kwarg(channel)] = receiver_locations
    if spec.has_bg_receivers:
        for channel in spec.bg_receiver_channels(2):
            kwargs[spec.bg_receiver_locations_kwarg(channel)] = receiver_locations

    for field in spec.wavefield_names(2):
        # ``auxiliary_state`` is passed back as a whole through
        # ``initial_state``, and the attenuation memory variables have no
        # initial-value keyword.
        if field == "auxiliary_state" or field not in parameters:
            continue
        values = torch.randn((N_SHOTS, N, N), generator=generator, dtype=dtype)
        values = (values.to(device) * _wavefield_scale(field)).detach()
        values.requires_grad_(True)
        kwargs[field] = values
        leaves.append(values)
        leaf_names.append(f"wavefield:{field}")

    kwargs["_models"] = positional
    return kwargs, leaves, leaf_names


def run_case(
    name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    *,
    backend: str = "auto",
    nt: int = NT,
) -> Result:
    """Run one case forward and backward."""
    kwargs, leaves, leaf_names = build_case(name, device, dtype, backend=backend, nt=nt)
    models = kwargs.pop("_models")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = getattr(deepwave, name)(*models, DX, DT, **kwargs)
    outputs = [tensor.detach().clone() for tensor in output_tensors(out)]
    loss = sum((tensor.to(torch.float64) ** 2).sum() for tensor in output_tensors(out))
    grads = torch.autograd.grad(loss, leaves, allow_unused=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return Result(
        outputs=outputs,
        grads=[torch.zeros(()) if g is None else g.detach().clone() for g in grads],
        names=leaf_names,
        backend=out.info.backend,
    )


class Result(NamedTuple):
    """One run's forward outputs, leaf gradients, leaf names and backend."""

    outputs: list[torch.Tensor]
    grads: list[torch.Tensor]
    names: list[str]
    backend: str


def compare(first: Result, second: Result) -> list[str]:
    """The names of the quantities that are not bitwise equal."""
    differing = [
        f"forward:{index}"
        for index, (a, b) in enumerate(zip(first.outputs, second.outputs))
        if not torch.equal(a, b)
    ]
    differing += [
        name
        for name, a, b in zip(first.names, first.grads, second.grads)
        if not torch.equal(a, b)
    ]
    return differing


def has_compiled_backend(name: str, device: torch.device) -> bool:
    """Whether ``name`` has a compiled backend for this test's case.

    All nine retained propagators have compiled and eager backends.
    """
    try:
        run_case(name, device, backend="compiled", nt=4)
    except deepwave.DeepwaveNotImplementedError:
        return False
    return True
