"""Native initial-wavefield gradients must transpose boundary fills."""

from dataclasses import replace

import pytest
import torch

import deepwave


@pytest.mark.parametrize(
    "propagator",
    [
        "acoustic",
        "viscoacoustic",
        "acoustic_vti",
        "elastic_vti",
        "acoustic_born",
        "elastic_born",
        "scalar_born",
        "scalar",
    ],
)
@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_compiled_initial_normal_field_gradient_matches_eager(
    propagator: str, accuracy: int
) -> None:
    """Boundary-adjoint contributions reach initial fields on the low face.

    The radius-one stencil is a control; wider stencils exercise the guard
    cells whose adjoints must be folded after the final reverse iteration.
    Eager PyTorch autograd supplies an independent reference for the compiled
    hand-written adjoint.
    """
    dtype = torch.double
    model = torch.ones(24, 24, dtype=dtype)
    initial = torch.zeros(1, 24, 24, dtype=dtype)
    initial[0, 0, 12], initial[0, 1, 12] = 0.3, 0.1
    initial.requires_grad_()
    locations = torch.tensor([[[1, 12], [2, 12], [3, 12]]])
    boundary = ["rigid", "pml", "pml", "pml"]

    def loss(backend: str) -> torch.Tensor:
        common = {
            "grid_spacing": 1.0,
            "dt": 0.05,
            "nt": 4,
            "accuracy": accuracy,
            "pml_width": 0,
            "boundary": boundary,
            "pml_freq": 2.0,
            "origin": [0, 0],
            "backend": backend,
        }
        if propagator == "acoustic":
            out = deepwave.acoustic(
                model,
                model,
                pressure_0=initial,
                receiver_locations_p=locations,
                **common,
            )
            return out.receiver_amplitudes_p.sum()
        if propagator == "viscoacoustic":
            out = deepwave.viscoacoustic(
                model,
                model,
                torch.full_like(model, 100.0),
                pressure_0=initial,
                receiver_locations_p=locations,
                q_freq=25.0,
                **common,
            )
            return out.receiver_amplitudes_p.sum()
        if propagator == "acoustic_vti":
            out = deepwave.acoustic_vti(
                3 * model,
                model,
                3 * model,
                model,
                sigma_v_0=initial,
                receiver_locations_sigma_v=locations,
                **common,
            )
            return out.receiver_amplitudes_sigma_v.sum()
        if propagator == "acoustic_born":
            out = deepwave.acoustic_born(
                model,
                model,
                dv=torch.zeros_like(model),
                pressure_0=initial,
                bg_receiver_locations_p=locations,
                **common,
            )
            return out.bg_receiver_amplitudes_p.sum()
        if propagator == "elastic_born":
            out = deepwave.elastic_born(
                3 * model,
                model,
                model,
                dlamb=torch.zeros_like(model),
                sigmayy_0=initial,
                bg_receiver_locations_y=locations,
                **common,
            )
            return out.bg_receiver_amplitudes_y.sum()
        if propagator == "scalar_born":
            out = deepwave.scalar_born(
                model,
                dv=torch.zeros_like(model),
                wavefield_0=initial,
                bg_receiver_locations=locations,
                **common,
            )
            return out.bg_receiver_amplitudes.sum()
        if propagator == "scalar":
            out = deepwave.scalar(
                model,
                wavefield_0=initial,
                receiver_locations=locations,
                **common,
            )
            return out.receiver_amplitudes.sum()
        out = deepwave.elastic_vti(
            3 * model,
            model,
            3 * model,
            model,
            model,
            sigmayy_0=initial,
            receiver_locations_y=locations,
            **common,
        )
        return out.receiver_amplitudes_y.sum()

    (compiled_grad,) = torch.autograd.grad(loss("compiled"), initial)
    (eager_grad,) = torch.autograd.grad(loss("eager"), initial)
    assert torch.isfinite(compiled_grad).all()
    assert torch.isfinite(eager_grad).all()
    assert eager_grad.abs().max() > 0
    torch.testing.assert_close(compiled_grad, eager_grad, rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize("propagator", ["acoustic_born", "elastic_born"])
@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_compiled_born_scattered_only_initial_gradient_matches_eager(
    propagator: str, accuracy: int
) -> None:
    """Scattered-only backward folds reach initial scattered fields."""
    dtype = torch.double
    model = torch.ones(24, 24, dtype=dtype)
    initial = torch.zeros(1, 24, 24, dtype=dtype)
    initial[0, 0, 12], initial[0, 1, 12] = 0.3, 0.1
    initial.requires_grad_()
    locations = torch.tensor([[[1, 12], [2, 12], [3, 12]]])
    common = {
        "grid_spacing": 1.0,
        "dt": 0.05,
        "nt": 4,
        "accuracy": accuracy,
        "pml_width": 0,
        "boundary": ["rigid", "pml", "pml", "pml"],
        "pml_freq": 2.0,
        "origin": [0, 0],
    }

    def loss(backend: str) -> torch.Tensor:
        if propagator == "acoustic_born":
            out = deepwave.acoustic_born(
                model,
                model,
                dv=torch.zeros_like(model),
                pressure_sc_0=initial,
                sc_receiver_locations_p=locations,
                backend=backend,
                **common,
            )
            return out.sc_receiver_amplitudes_p.sum()
        out = deepwave.elastic_born(
            3 * model,
            model,
            model,
            dlamb=torch.zeros_like(model),
            sigmayy_sc_0=initial,
            sc_receiver_locations_y=locations,
            backend=backend,
            **common,
        )
        return out.sc_receiver_amplitudes_y.sum()

    (compiled_grad,) = torch.autograd.grad(loss("compiled"), initial)
    (eager_grad,) = torch.autograd.grad(loss("eager"), initial)
    assert torch.isfinite(compiled_grad).all()
    assert torch.isfinite(eager_grad).all()
    assert eager_grad.abs().max() > 0
    torch.testing.assert_close(compiled_grad, eager_grad, rtol=1e-7, atol=1e-12)


def test_compiled_gsls_initial_memory_gradients_match_eager() -> None:
    """Every GSLS memory field gets the final low-boundary adjoint fold."""
    dtype = torch.double
    model = torch.ones(24, 24, dtype=dtype)
    q = torch.full_like(model, 100.0)
    locations = torch.tensor([[[1, 12], [2, 12], [3, 12]]])
    boundary = ["rigid", "pml", "pml", "pml"]
    common = {
        "grid_spacing": 1.0,
        "dt": 0.05,
        "nt": 4,
        "accuracy": 8,
        "pml_width": 0,
        "boundary": boundary,
        "pml_freq": 2.0,
        "q_n_mechanisms": 2,
        "q_band": (5.0, 50.0),
        # The API requires a wavefield or source even when the initial
        # auxiliary state contains the quantity under test.
        "pressure_0": torch.zeros((1, 24, 24), dtype=dtype),
        "receiver_locations_p": locations,
    }
    template = deepwave.viscoacoustic(model, model, q, backend="eager", **common)

    def gradients(backend: str) -> tuple[torch.Tensor, ...]:
        memory_inputs: dict[str, torch.Tensor] = {}
        state_tensors = {
            name: value.detach().clone()
            for name, value in template.auxiliary_state.items()
        }
        for name, current in state_tensors.items():
            if name.startswith("r_mech"):
                value = torch.zeros_like(current)
                value[0, 0, 12], value[0, 1, 12] = 0.3, 0.1
                value.requires_grad_()
                memory_inputs[name] = value
                state_tensors[name] = value
        initial_state = {
            "auxiliary_state": replace(template.auxiliary_state, _tensors=state_tensors)
        }
        out = deepwave.viscoacoustic(
            model,
            model,
            q,
            initial_state=initial_state,
            backend=backend,
            **common,
        )
        loss = out.receiver_amplitudes_p.sum()
        return torch.autograd.grad(loss, tuple(memory_inputs.values()))

    compiled_gradients = gradients("compiled")
    eager_gradients = gradients("eager")
    assert len(compiled_gradients) == 2
    for compiled_grad, eager_grad in zip(compiled_gradients, eager_gradients):
        assert torch.isfinite(compiled_grad).all()
        assert torch.isfinite(eager_grad).all()
        assert eager_grad.abs().max() > 0
        torch.testing.assert_close(compiled_grad, eager_grad, rtol=1e-7, atol=1e-12)
