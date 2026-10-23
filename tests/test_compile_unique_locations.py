"""Runtime location validation across an outer torch.compile graph break."""

from typing import Optional

import pytest
import torch

import deepwave
from deepwave._exceptions import DeepwaveRuntimeError


@pytest.mark.parametrize("solver", ["scalar", "acoustic"])
@pytest.mark.parametrize("compiler_backend", ["eager", None])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_compiled_call_rechecks_sources_and_adjoint_receivers(
    solver: str, compiler_backend: Optional[str], device_name: str
) -> None:
    """A reused compiled callable must inspect changed location values."""
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if device_name == "cpu" and compiler_backend is None:
        pytest.skip("The CUDA case covers the default compiler")
    if solver == "acoustic" and compiler_backend is None:
        pytest.skip("One native CUDA solver covers the default compiler")
    device = torch.device(device_name)
    model = torch.ones(20, device=device)
    amplitudes = torch.ones(1, 3, 2, device=device)
    unique = torch.tensor([[[7], [8], [9]]], device=device)
    repeated = torch.tensor([[[7], [7], [9]]], device=device)
    sentinel = torch.tensor(
        [[[7], [deepwave.IGNORE_LOCATION], [deepwave.IGNORE_LOCATION]]],
        device=device,
    )

    def run(sources: torch.Tensor, receivers: torch.Tensor) -> torch.Tensor:
        shared = {"pml_width": 4, "pml_freq": 1.0, "max_vel": 1.0}
        if solver == "scalar":
            return deepwave.scalar(
                model,
                1.0,
                0.1,
                source_amplitudes=amplitudes,
                source_locations=sources,
                receiver_locations=receivers,
                **shared,
            ).receiver_amplitudes
        return deepwave.acoustic(
            model,
            model,
            1.0,
            0.1,
            source_amplitudes_volume_rate=amplitudes,
            source_locations_volume_rate=sources,
            receiver_locations_p=receivers,
            **shared,
        ).receiver_amplitudes_p

    compiled = torch.compile(run, backend=compiler_backend)
    with pytest.raises(
        DeepwaveRuntimeError, match="locations must be unique within each shot"
    ):
        run(repeated, unique)
    with pytest.raises(
        DeepwaveRuntimeError, match="locations must be unique within each shot"
    ):
        run(unique, repeated)
    compiled(unique, unique)
    compiled(sentinel, sentinel)
    with pytest.raises(
        DeepwaveRuntimeError, match="locations must be unique within each shot"
    ):
        compiled(repeated, unique)
    with pytest.raises(
        DeepwaveRuntimeError, match="locations must be unique within each shot"
    ):
        compiled(unique, repeated)
