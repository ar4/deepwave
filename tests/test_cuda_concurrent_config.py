"""Concurrent compiled calls must not exchange CUDA constant-memory settings."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import determinism_cases as dc
import pytest
import torch
from helpers import output_tensors

import deepwave


def _run(
    name: str,
    dt: float,
    pml_freq: float,
    storage_mode: str,
    barrier: Barrier | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
    device = torch.device("cuda:0")
    kwargs, leaves, names = dc.build_case(name, device, backend="compiled", nt=96)
    models = kwargs.pop("_models")
    kwargs.update(pml_freq=pml_freq, storage_mode=storage_mode)
    if storage_mode == "cpu":
        kwargs.update(
            storage_compression=True,
            forward_callback=lambda _: None,
            backward_callback=lambda _: None,
            callback_frequency=13,
        )
    if barrier is not None:
        barrier.wait(timeout=30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out: Any = getattr(deepwave, name)(*models, dc.DX, dt, **kwargs)
    tensors = output_tensors(out)
    loss = sum(t.square().sum() for t in tensors)
    grads = torch.autograd.grad(loss, leaves, allow_unused=True)
    torch.cuda.synchronize(device)
    return (
        [t.detach().clone() for t in tensors],
        [
            torch.zeros((), device=device) if g is None else g.detach().clone()
            for g in grads
        ],
        names,
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("name", "storage_mode"),
    [("scalar", "device"), ("acoustic_born", "cpu"), ("elastic_vti", "device")],
)
def test_same_device_concurrent_configuration(name: str, storage_mode: str) -> None:
    """Different settings agree bitwise with nonzero serial forward/gradients."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.cuda.set_device(0)
    settings = ((0.001, 25.0), (0.0007, 35.0))
    serial = [_run(name, dt, freq, storage_mode) for dt, freq in settings]
    serial_repeat = [_run(name, dt, freq, storage_mode) for dt, freq in settings]
    for first, second in zip(serial, serial_repeat):
        for first_group, second_group in zip(first[:2], second[:2]):
            assert all(torch.equal(a, b) for a, b in zip(first_group, second_group))
    for outputs, grads, names in serial:
        assert any(t.abs().max().item() > 0 for t in outputs)
        for kind in ("model:", "source:"):
            assert any(
                t.abs().max().item() > 0
                for name_, t in zip(names, grads)
                if name_.startswith(kind)
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        for repeat in range(8):
            barrier = Barrier(2)
            futures = [
                pool.submit(_run, name, dt, freq, storage_mode, barrier)
                for dt, freq in settings
            ]
            parallel = [future.result(timeout=60) for future in futures]
            for case, (actual, expected) in enumerate(zip(parallel, serial)):
                for actual_tensors, expected_tensors in zip(actual[:2], expected[:2]):
                    for index, (got, want) in enumerate(
                        zip(actual_tensors, expected_tensors)
                    ):
                        assert torch.equal(got, want), (
                            f"{name} repeat={repeat} case={case} tensor={index} "
                            f"max_abs={(got - want).abs().max().item()}"
                        )
