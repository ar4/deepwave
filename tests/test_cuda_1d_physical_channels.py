"""Live compiled 1D CUDA receiver and gradient paths for vector/Born families."""

import pytest
import torch

import deepwave

FAMILIES = ("acoustic_born", "elastic", "elastic_born")
NDIM = 1
DTYPE = torch.float64
GRID_SIZE = 14
NT = 24
DT = 0.0002
DX = 5.0
MAX_VEL = 4000.0
ACCURACY = 8
PML_WIDTH = 3


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("family", FAMILIES)
def test_one_dimensional_cuda_receiver_and_parameter_gradients(family):
    """Exercise nonzero physical signals and derivatives in compiled CUDA 1D."""
    device = torch.device("cuda")
    profile = torch.linspace(-1.0, 1.0, GRID_SIZE, dtype=DTYPE, device=device)

    # SI units: m/s, kg/m^3, Pa, and 1/(kg/m^3), as appropriate. The small
    # spatial variation keeps material and Born-perturbation sensitivities live.
    values = {
        "v": 1500.0 * (1.0 + 0.01 * profile),
        "rho": 2000.0 * (1.0 + 0.01 * profile),
        "lamb": 4.0e9 * (1.0 + 0.01 * profile),
        "mu": 2.0e9 * (1.0 + 0.01 * profile),
        "buoyancy": 5.0e-4 * (1.0 - 0.01 * profile),
    }
    perturbation_values = {
        "dv": 10.0 * (1.0 + 0.05 * profile),
        "drho": 5.0 * (1.0 + 0.05 * profile),
        "dlamb": 1.0e7 * (1.0 + 0.05 * profile),
        "dmu": 5.0e6 * (1.0 + 0.05 * profile),
        "dbuoyancy": 1.0e-5 * (1.0 + 0.05 * profile),
    }
    spec = deepwave.spec(family)
    perturbation_names = set(perturbation_values)
    leaves_by_name = {}
    models = []
    keyword_args = {}
    for name in spec.model_names(NDIM):
        if name in perturbation_names:
            leaf = perturbation_values[name].detach().requires_grad_()
            keyword_args[name] = leaf
        else:
            leaf = values[name].detach().requires_grad_()
            models.append(leaf)
        leaves_by_name[name] = leaf

    # A short, nonzero pulse and adjacent in-domain receiver produce a live
    # signal within 24 samples at the chosen physical dt and grid spacing.
    source_channel = spec.source_channels(NDIM)[0]
    source = torch.zeros((1, 1, NT), dtype=DTYPE, device=device)
    source[0, 0, 1] = 1.0
    source[0, 0, 2] = -0.5
    source = source.requires_grad_()
    source_location = torch.tensor([[[6]]], dtype=torch.long, device=device)
    receiver_location = torch.tensor([[[7]]], dtype=torch.long, device=device)
    keyword_args[spec.source_amplitudes_kwarg(source_channel)] = source
    keyword_args[spec.source_locations_kwarg(source_channel)] = source_location

    for channel in spec.receiver_channels(NDIM):
        keyword_args[spec.receiver_locations_kwarg(channel)] = receiver_location
    if spec.has_bg_receivers:
        for channel in spec.bg_receiver_channels(NDIM):
            keyword_args[spec.bg_receiver_locations_kwarg(channel)] = receiver_location

    propagator = getattr(deepwave, family)
    output = propagator(
        *models,
        DX,
        DT,
        **keyword_args,
        pml_width=PML_WIDTH,
        pml_freq=25.0,
        max_vel=MAX_VEL,
        accuracy=ACCURACY,
        backend="compiled",
        storage_mode="device",
    )
    output_dict = output.as_dict()
    receiver_outputs = {
        name: tensor
        for name, tensor in output_dict.items()
        if "receiver_amplitudes" in name
    }
    assert set(receiver_outputs) == set(spec.receiver_names(NDIM))
    for receiver in receiver_outputs.values():
        assert torch.isfinite(receiver).all()
        assert torch.count_nonzero(receiver).item() > 0

    # Squared receiver energy keeps the objective tied to actual, nonzero
    # forward signals. Every background model, Born perturbation, and source
    # amplitude leaf must have a finite, nonzero derivative.
    loss = sum(receiver.square().sum() for receiver in receiver_outputs.values())
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    gradient_names = [*leaves_by_name, "source_amplitudes"]
    gradient_leaves = [*leaves_by_name.values(), source]
    gradients = torch.autograd.grad(loss, gradient_leaves)
    for name, gradient in zip(gradient_names, gradients):
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient).item() > 0, name
