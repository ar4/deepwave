"""CUDA native calls preserve the caller's current device."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import deepwave

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("storage_mode", "compressed"),
    [("device", False), ("cpu", True), ("disk", False), ("disk", True)],
)
def test_native_forward_backward_restore_current_device(
    storage_mode: str, compressed: bool, tmp_path: Path
) -> None:
    """Forward, callbacks and adjoint on cuda:1 leave cuda:0 current."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    torch.cuda.set_device(0)
    model = torch.full(
        (24, 24),
        1500.0,
        dtype=torch.float32,
        device="cuda:1",
        requires_grad=True,
    )
    amplitudes = torch.zeros((1, 1, 24), device="cuda:1")
    amplitudes[0, 0, 3] = 1.0
    amplitudes.requires_grad_()
    locations = torch.tensor([[[12, 12]]], device="cuda:1")
    receiver_locations = torch.tensor([[[12, 13]]], device="cuda:1")
    forward_devices: list[int] = []
    backward_devices: list[int] = []

    def forward_callback(_state: object) -> None:
        forward_devices.append(torch.cuda.current_device())

    def backward_callback(_state: object) -> None:
        backward_devices.append(torch.cuda.current_device())

    assert torch.cuda.current_device() == 0
    output = deepwave.scalar(
        model,
        10.0,
        0.001,
        source_amplitudes=amplitudes,
        source_locations=locations,
        receiver_locations=receiver_locations,
        pml_width=4,
        pml_freq=15.0,
        max_vel=1500.0,
        backend="compiled",
        storage_mode=storage_mode,
        storage_path=tmp_path if storage_mode == "disk" else None,
        storage_compression=compressed,
        forward_callback=forward_callback,
        backward_callback=backward_callback,
        callback_frequency=5,
    )
    assert torch.cuda.current_device() == 0
    output.receiver_amplitudes.square().sum().backward()
    assert torch.cuda.current_device() == 0
    assert forward_devices
    assert all(device == 0 for device in forward_devices)
    assert backward_devices
    # Autograd calls backward on the tensor's device and restores cuda:0
    # around the whole backward invocation.
    assert all(device == 1 for device in backward_devices)
    assert model.grad is not None
    assert model.grad.abs().max() > 0
    assert amplitudes.grad is not None
    assert amplitudes.grad.abs().max() > 0
