"""Ensure branch reporting cannot dilute the existing statement coverage gate."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("covered", "statements", "passes"),
    [
        (92, 100, True),
        (100, 100, True),
        (91999, 100000, False),
        (0, 0, False),
        (-1, 100, False),
        (101, 100, False),
        (92.0, 100, False),
    ],
)
def test_statement_gate_is_independent_of_branch_percentage(
    tmp_path,
    covered,
    statements,
    passes,
):
    """Uncovered branches neither hide a statement deficit nor change its floor."""
    path = Path(__file__).resolve().parents[1] / "scripts/check_line_coverage.py"
    spec = importlib.util.spec_from_file_location("check_line_coverage", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = tmp_path / "coverage.json"
    report.write_text(
        json.dumps(
            {
                "totals": {
                    "covered_lines": covered,
                    "num_statements": statements,
                    "covered_branches": 0,
                    "num_branches": 100,
                }
            }
        )
    )
    if passes:
        module.check_report(report)
    else:
        with pytest.raises((ValueError, TypeError), match=r"coverage|Coverage"):
            module.check_report(report)
