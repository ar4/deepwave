"""Internal Born model JVPs must be safe across host threads."""

from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch
from torch.utils.checkpoint import checkpoint


@pytest.mark.parametrize("kind", ["acoustic", "elastic"])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("use_reentrant", [False, True])
def test_born_preparation_checkpoint_gradients(
    kind: str, ndim: int, use_reentrant: bool
) -> None:
    """Checkpoint hooks preserve nonzero background and scatter gradients."""
    module = importlib.import_module("deepwave." + kind + "_born")
    function = (
        module.prepare_scatter
        if kind == "acoustic"
        else module.prepare_scatter_parameters
    )
    generator = torch.Generator().manual_seed(41)
    values = tuple(
        (
            1 + torch.rand((1,) + (4,) * ndim, generator=generator, dtype=torch.float64)
        ).requires_grad_()
        for _ in range(4)
    )

    def loss(*models: torch.Tensor) -> torch.Tensor:
        return sum(output.square().sum() for output in function(*models))

    expected_loss = loss(*values)
    expected = tuple(
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(
            values, torch.autograd.grad(expected_loss, values, allow_unused=True)
        )
    )
    actual_loss = checkpoint(loss, *values, use_reentrant=use_reentrant)
    actual_loss.backward()
    actual = tuple(
        torch.zeros_like(value) if value.grad is None else value.grad
        for value in values
    )
    torch.testing.assert_close(actual_loss, expected_loss)
    for got, want in zip(actual, expected):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want)
    assert sum(gradient.abs().sum() for gradient in actual) > 0

    # An enclosing transform must retain its own forward/reverse AD level.
    direction = tuple(torch.ones_like(value) for value in values)
    _, tangent = torch.func.jvp(loss, values, direction)
    torch.testing.assert_close(tangent, sum(gradient.sum() for gradient in expected))
    transformed = torch.func.grad(loss, argnums=(0, 1, 2, 3))(*values)
    for got, want in zip(transformed, expected):
        torch.testing.assert_close(got, want)


def _run(kind: str, barrier: Barrier | None = None) -> list[torch.Tensor]:
    base = torch.ones((1, 12, 12), dtype=torch.float64, requires_grad=True)
    if kind == "acoustic":
        function = importlib.import_module("deepwave.acoustic_born").prepare_scatter
        values = (1500 * base, 2000 * base, 15 * base, 20 * base)
    else:
        function = importlib.import_module(
            "deepwave.elastic_born"
        ).prepare_scatter_parameters
        values = (1e9 * base, 1e-3 * base, 1e7 * base, 1e-5 * base)
    if barrier is not None:
        barrier.wait(timeout=20)
    return [t.detach().clone() for t in function(*values)]


@pytest.mark.parametrize(
    ("first", "second"),
    [("acoustic", "acoustic"), ("elastic", "elastic"), ("acoustic", "elastic")],
)
def test_born_internal_jvp_is_thread_safe(first: str, second: str) -> None:
    """The two Born paths share one forward-AD-level critical section."""
    reference = [_run(first), _run(second)]
    assert all(
        torch.isfinite(t).all() and t.abs().max() > 0
        for case in reference
        for t in case
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        for repeat in range(50):
            barrier = Barrier(2)
            futures = [pool.submit(_run, kind, barrier) for kind in (first, second)]
            actual = [future.result(timeout=20) for future in futures]
            for case, (got, want) in enumerate(zip(actual, reference)):
                assert all(torch.equal(a, b) for a, b in zip(got, want)), (
                    f"repeat={repeat} case={case} kinds=({first},{second})"
                )
