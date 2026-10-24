"""Defined limiting cases, compared with the differentiable tensor equations."""

from __future__ import annotations

import determinism_cases as dc
import pytest
import torch
from helpers import output_tensors

import deepwave
from deepwave.checkpointing import run_segments


@pytest.fixture(autouse=True)
def small_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the cases small while retaining all sources and receiver channels."""
    monkeypatch.setattr(dc, "N", 16)
    monkeypatch.setattr(dc, "PML_WIDTH", 4)


def _run(
    name: str,
    backend: str,
    speed: int,
    dt: float,
    *,
    device: str,
    loss_kind: str = "squares",
    initial_auxiliary: bool = False,
    checkpoint_mode: bool | None = None,
):
    kwargs, leaves, names = dc.build_case(
        name, torch.device(device), torch.float64, backend=backend, nt=3
    )
    with torch.no_grad():
        for leaf, label in zip(leaves, names):
            if label == "model:v" or (label == "model:dv" and speed < 0):
                leaf.mul_(speed)
    for label in names:
        if label.startswith("wavefield:"):
            field = label.removeprefix("wavefield:")
            kwargs[field] = torch.nn.functional.pad(kwargs[field], (4, 4, 4, 4))
    # Exercise the inferred zero CFL bound as well as the propagator itself.
    if speed == 0:
        kwargs.pop("max_vel")
    kwargs["origin"] = [0, 0]
    models = kwargs.pop("_models")
    if initial_auxiliary:
        # Produce valid, active boundary/attenuation memory independently of
        # the compiled implementation, then differentiate each restart input.
        with torch.no_grad():
            warm = getattr(deepwave, name)(
                *models, dc.DX, dt, **{**kwargs, "backend": "eager"}
            )
        memory = {}
        for field, value in warm.auxiliary_state.items():
            leaf = value.detach().clone().requires_grad_()
            memory[field] = leaf
            leaves.append(leaf)
            names.append(f"auxiliary:{field}")
        kwargs["initial_state"] = {
            "auxiliary_state": deepwave.AuxiliaryState._from_flat(memory)
        }
    if checkpoint_mode is None:
        out = getattr(deepwave, name)(*models, dc.DX, dt, **kwargs)
    else:
        initial_state = kwargs.pop("initial_state", {})
        for label in names:
            if label.startswith("wavefield:"):
                field = label.removeprefix("wavefield:")
                initial_state[field] = kwargs.pop(field)
        sources = {
            key: kwargs.pop(key)
            for key in list(kwargs)
            if key.startswith("source_amplitudes")
        }
        optional_models = {
            key: kwargs.pop(key)
            for key in deepwave.spec(name).model_names(2)
            if key in kwargs
        }
        out = run_segments(
            getattr(deepwave, name),
            models,
            dc.DX,
            dt,
            sources,
            n_segments=3,
            initial_state=initial_state,
            optional_model_args=optional_models,
            checkpoint=True,
            use_reentrant=checkpoint_mode,
            **kwargs,
        )
    tensors = output_tensors(out)
    if loss_kind == "squares":
        loss = sum(t.square().mean() for t in tensors if t.numel())
    else:
        selected = out.auxiliary_state.values() if loss_kind == "auxiliary" else tensors
        generator = torch.Generator(device=device).manual_seed(923)
        loss = sum(
            (
                t
                * torch.randn(
                    t.shape, dtype=t.dtype, device=t.device, generator=generator
                )
            ).mean()
            for t in selected
            if t.numel()
        )
    if checkpoint_mode is None:
        gradients = torch.autograd.grad(loss, leaves, allow_unused=True)
    else:
        # Reentrant torch checkpointing requires backward(), rather than grad().
        loss.backward()
        gradients = tuple(leaf.grad for leaf in leaves)
    gradients = [
        torch.zeros_like(leaf) if grad is None else grad
        for leaf, grad in zip(leaves, gradients)
    ]
    assert all(torch.isfinite(t).all() for t in [*tensors, *gradients])
    return tensors, gradients, names


@pytest.mark.parametrize(
    "name", ["scalar", "scalar_born", "acoustic", "acoustic_born", "viscoacoustic"]
)
@pytest.mark.parametrize("speed", [-1, 0])
def test_signed_and_zero_speed_match_autograd(
    name: str, speed: int, numerics_device: str
) -> None:
    """Every output and input gradient agrees with the tensor implementation."""
    compiled = _run(name, "compiled", speed, dc.DT / 2, device=numerics_device)
    eager = _run(name, "eager", speed, dc.DT / 2, device=numerics_device)
    for actual, expected in zip(compiled[:2], eager[:2]):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    "name", ["scalar", "scalar_born", "acoustic", "acoustic_born", "viscoacoustic"]
)
@pytest.mark.parametrize("backend", ["compiled", "eager"])
def test_speed_sign_symmetry(name: str, backend: str, numerics_device: str) -> None:
    """Changing v and dv signs preserves outputs and reverses their gradients."""
    positive, positive_grad, names = _run(
        name, backend, 1, dc.DT / 2, device=numerics_device
    )
    negative, negative_grad, _ = _run(
        name, backend, -1, dc.DT / 2, device=numerics_device
    )
    torch.testing.assert_close(negative, positive, rtol=0, atol=0)
    for label, actual, expected in zip(names, negative_grad, positive_grad):
        sign = -1 if label in ("model:v", "model:dv") else 1
        torch.testing.assert_close(actual, sign * expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", dc.PROPAGATORS)
def test_zero_time_step_matches_autograd(name: str, numerics_device: str) -> None:
    """The discrete zero-step limit has finite outputs and correct gradients."""
    compiled = _run(name, "compiled", 1, 0.0, device=numerics_device)
    eager = _run(name, "eager", 1, 0.0, device=numerics_device)
    for actual, expected in zip(compiled[:2], eager[:2]):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("backend", ["compiled", "eager"])
def test_scalar_zero_step_obeys_two_level_recurrence(
    backend: str, numerics_device: str
) -> None:
    """At dt=0, u[n+1]=2*u[n]-u[n-1], including its exact adjoint."""
    model = torch.ones(
        8, dtype=torch.float64, device=numerics_device, requires_grad=True
    )
    current = torch.full(
        (1, 12), 3.0, dtype=torch.float64, device=numerics_device, requires_grad=True
    )
    previous = torch.full(
        (1, 12), 2.0, dtype=torch.float64, device=numerics_device, requires_grad=True
    )
    source = torch.ones(
        1, 1, 4, dtype=torch.float64, device=numerics_device, requires_grad=True
    )
    out = deepwave.scalar(
        model,
        1.0,
        0.0,
        source_amplitudes=source,
        source_locations=torch.tensor([[[4]]], device=numerics_device),
        wavefield_0=current,
        wavefield_m1=previous,
        origin=[0],
        pml_width=2,
        backend=backend,
    )
    torch.testing.assert_close(
        out.wavefield_0, torch.full_like(current, 7), rtol=0, atol=0
    )
    torch.testing.assert_close(
        out.wavefield_m1, torch.full_like(previous, 6), rtol=0, atol=0
    )
    gradients = torch.autograd.grad(
        out.wavefield_0.sum(), (current, previous, model, source)
    )
    for actual, expected in zip(gradients, (5.0, -4.0, 0.0, 0.0)):
        torch.testing.assert_close(
            actual, torch.full_like(actual, expected), rtol=0, atol=0
        )


@pytest.mark.parametrize("name", dc.PROPAGATORS)
@pytest.mark.parametrize("loss_kind", ["all", "auxiliary"])
@pytest.mark.parametrize("initial_auxiliary", [False, True])
def test_complete_state_adjoint_matches_autograd(
    name: str, loss_kind: str, initial_auxiliary: bool, numerics_device: str
) -> None:
    """Independent nonzero cotangents exercise every returned state component."""
    compiled = _run(
        name,
        "compiled",
        1,
        dc.DT / 2,
        device=numerics_device,
        loss_kind=loss_kind,
        initial_auxiliary=initial_auxiliary,
    )
    eager = _run(
        name,
        "eager",
        1,
        dc.DT / 2,
        device=numerics_device,
        loss_kind=loss_kind,
        initial_auxiliary=initial_auxiliary,
    )
    for actual, expected in zip(compiled[:2], eager[:2]):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("name", dc.PROPAGATORS)
@pytest.mark.parametrize("use_reentrant", [False, True])
def test_checkpointed_complete_state_adjoint_matches_autograd(
    name: str, use_reentrant: bool, numerics_device: str
) -> None:
    """Checkpoint boundaries preserve state cotangents and every input gradient."""
    compiled = _run(
        name,
        "compiled",
        1,
        dc.DT / 2,
        device=numerics_device,
        loss_kind="all",
        initial_auxiliary=True,
        checkpoint_mode=use_reentrant,
    )
    eager = _run(
        name,
        "eager",
        1,
        dc.DT / 2,
        device=numerics_device,
        loss_kind="all",
        initial_auxiliary=True,
    )
    assert compiled[2] == eager[2]
    for actual, expected in zip(compiled[:2], eager[:2]):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
