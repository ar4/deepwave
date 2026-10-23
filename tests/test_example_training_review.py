"""Review displayed and executable training loops without importing examples.

Only extracted loss/loop AST nodes run, against small algebraic predictions.
No Deepwave import, wave propagation, compiler, GPU or process group is used.
"""

import ast
import contextlib
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _tree(name, marker=None):
    source = (ROOT / "docs" / name).read_text()
    if name.endswith(".rst"):
        blocks = []
        lines = source.splitlines()
        for i, line in enumerate(lines):
            if not line.endswith("::"):
                continue
            block = []
            for following in lines[i + 1 :]:
                if following and not following.startswith("    "):
                    break
                block.append(following)
            blocks.append(textwrap.dedent("\n".join(block)))
        source = next(block for block in blocks if marker in block)
    return ast.parse(source)


def _execute(node, namespace):
    module = ast.Module(body=[node], type_ignores=[])
    exec(  # noqa: S102 - selected local example AST; no imports or propagation
        compile(ast.fix_missing_locations(module), "<reviewed-example>", "exec"),
        namespace,
    )


@pytest.mark.parametrize("suffix", ["py", "rst"])
@pytest.mark.parametrize("batch_size", [1, 3, 7])
def test_rtm_batch_loss_is_full_survey_mean(suffix, batch_size):
    """Unequal batches must give the same loss and gradient as one batch."""
    tree = _tree(f"example_rtm.{suffix}", "n_batch =")
    loss_node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "loss" for t in n.targets)
        and any(isinstance(c, ast.Name) and c.id == "mask" for c in ast.walk(n))
    )
    x = torch.arange(1, 17, dtype=torch.float64).reshape(8, 2)
    target = x.flip(0) / 3
    parameter = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    expected = (parameter * x - target).square().mean()
    (gradient,) = torch.autograd.grad(expected, parameter)
    total = 0.0
    for start in range(0, 8, batch_size):
        end = min(start + batch_size, 8)
        namespace = {
            "loss_fn": torch.nn.MSELoss(),
            "out": SimpleNamespace(sc_receiver_amplitudes=parameter * x[start:end]),
            "mask": torch.ones_like(x),
            "observed_scatter_masked": target,
            "s": slice(start, end),
            "batch_start": start,
            "batch_end": end,
            "n_shots": 8,
        }
        _execute(loss_node, namespace)
        loss = namespace["loss"]
        total += loss.item()
        loss.backward()
    assert total == pytest.approx(expected.item(), rel=1e-14)
    torch.testing.assert_close(parameter.grad, gradient, rtol=1e-14, atol=0)


def test_ddp_lbfgs_closure_returns_shared_loss():
    """Rank averaging must agree with a full-data objective and derivative."""
    tree = _tree("example_distributed_ddp.py")
    closure = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "closure"
    )
    x = torch.arange(1, 17, dtype=torch.float64).reshape(8, 2)
    target = x.flip(0) / 3
    expected = 1e6 * (0.7 * x - target).square().mean()
    expected_gradient = 2e6 * ((0.7 * x - target) * x).mean()
    contributions, gradients = [], []
    for local_x, local_target in zip(x.tensor_split(3), target.tensor_split(3)):
        parameter = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)

        def all_reduce(value):
            assert not value.requires_grad
            contributions.append(value.item())
            value.copy_(3 * expected)

        namespace = {
            "optimiser": SimpleNamespace(zero_grad=lambda: None),
            "prop": lambda *_: SimpleNamespace(receiver_amplitudes=parameter * local_x),
            "source_amplitudes": local_x,
            "source_locations": None,
            "receiver_locations": None,
            "filt": lambda value, _: value,
            "taper": lambda value: value,
            "sos": None,
            "loss_fn": torch.nn.MSELoss(),
            "observed_data_filt": local_target,
            "world_size": 3,
            "n_shots": 8,
            "dist": SimpleNamespace(all_reduce=all_reduce),
            "loss_history": [],
        }
        _execute(closure, namespace)
        returned = namespace["closure"]()
        torch.testing.assert_close(returned, expected)
        assert namespace["loss_history"] == [expected.item()]
        gradients.append(parameter.grad)
    assert sum(contributions) / 3 == pytest.approx(expected.item())
    torch.testing.assert_close(torch.stack(gradients).mean(), expected_gradient)


@pytest.mark.parametrize("suffix", ["py", "rst"])
@pytest.mark.parametrize("world_size", [3, 5])
def test_flagship_batches_cover_every_shot_and_synchronise_once(suffix, world_size):
    """Uneven rank partitions and partial batches preserve the survey objective."""
    tree = _tree(f"example_flagship_3d.{suffix}", "n_batches =")
    count = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "n_batches" for t in n.targets)
    )
    loop = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id == "batch"
    )
    x = torch.arange(1, 17, dtype=torch.float64).reshape(8, 2)
    target = x.flip(0) / 3
    gradients = []
    for local_x, local_target in zip(
        x.tensor_split(world_size), target.tensor_split(world_size)
    ):
        parameter = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        seen, synced = [], []

        class Prop:
            synchronise = True

            @contextlib.contextmanager
            def no_sync(self):
                self.synchronise = False
                try:
                    yield
                finally:
                    self.synchronise = True

            def __call__(self, amplitudes, *_):
                seen.extend(amplitudes[:, 0].tolist())
                synced.append(self.synchronise)
                return parameter * amplitudes

        namespace = {
            "contextlib": contextlib,
            "prop": Prop(),
            "loss_fn": torch.nn.MSELoss(),
            "source_amplitudes": local_x,
            "source_locations": local_x,
            "receiver_locations": local_x,
            "observed": local_target,
            "n_shots_per_rank": len(local_x),
            "n_shots_total": 8,
            "world_size": world_size,
            "shot_batch": 2,
            "epoch_loss": 0.0,
        }
        _execute(count, namespace)
        _execute(loop, namespace)
        assert seen == local_x[:, 0].tolist()
        assert synced == [False] * (len(synced) - 1) + [True]
        gradients.append(parameter.grad)
    expected_gradient = 2 * ((0.7 * x - target) * x).mean()
    torch.testing.assert_close(torch.stack(gradients).mean(), expected_gradient)
