"""Check the published Born/JVP comparison without importing or running Deepwave."""

import ast
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _displayed_block(marker):
    lines = (ROOT / "docs/example_torch_func.rst").read_text().splitlines()
    for i, line in enumerate(lines):
        if line.endswith("::"):
            block = []
            for following in lines[i + 1 :]:
                if following and not following.startswith("    "):
                    break
                block.append(following)
            source = textwrap.dedent("\n".join(block))
            if marker in source:
                return ast.parse(source)
    raise AssertionError(f"Missing displayed example: {marker}")


@pytest.mark.parametrize("source", ["displayed", "script"])
def test_born_and_jvp_examples_fix_the_same_numerical_setup(source):
    """An equivalence example must compare the same discrete operators.

    Execute only its wrapper and Born call against recorders. Deliberately
    different inference defaults expose omitted settings without simulating
    either wave equation or importing an installed Deepwave package.
    """
    if source == "displayed":
        wrapper_tree = _displayed_block("def forward_fast")
        born_tree = _displayed_block("out = deepwave.scalar_born")
        wrapper_name = "forward_fast"
    else:
        wrapper_tree = born_tree = ast.parse(
            (ROOT / "docs/example_torch_func.py").read_text()
        )
        wrapper_name = "forward_sim"
    wrapper = next(
        node
        for node in wrapper_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == wrapper_name
    )
    born = next(
        node
        for node in ast.walk(born_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "scalar_born"
    )
    calls = []

    def record(*args, **kwargs):
        calls.append({"pml_freq": 19.0, "max_vel": 1800.0, **kwargs})
        return SimpleNamespace(receiver_amplitudes=object())

    namespace = dict.fromkeys(
        [
            "v",
            "v_bg",
            "dv",
            "scatter",
            "source_amplitudes",
            "source_locations",
            "receiver_locations",
        ],
        object(),
    )
    namespace.update(
        deepwave=SimpleNamespace(scalar=record, scalar_born=record),
        dx=4.0,
        dt=0.004,
        freq=25.0,
    )
    exec(  # noqa: S102 - reviewed wrapper only; propagation is replaced by a recorder
        compile(ast.Module(body=[wrapper], type_ignores=[]), "<example>", "exec"),
        namespace,
    )
    namespace[wrapper_name](namespace["v"])
    eval(  # noqa: S307 - reviewed call only; propagation is replaced by a recorder
        compile(ast.Expression(body=born), "<example>", "eval"), namespace
    )
    assert len(calls) == 2
    for key in ("pml_freq", "max_vel"):
        assert calls[0][key] == calls[1][key]
    assert calls[1]["sc_receiver_locations"] is calls[0]["receiver_locations"]
