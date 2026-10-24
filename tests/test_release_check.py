"""Exercise qualification accounting with tiny pytest children, never Deepwave."""

import sys
from pathlib import Path

import pytest

from scripts import release_check as release


@pytest.mark.parametrize(
    ("body", "collect_only", "valid", "passed", "skipped"),
    [
        ('pytest.skip("unavailable reference")', False, False, 0, 1),
        ("assert True", False, True, 1, 0),
        ("assert False", False, False, 0, 0),
        ('pytest.skip("unavailable reference")', True, True, 0, 0),
    ],
)
def test_stage_requires_executed_tests(
    tmp_path, body, collect_only, valid, passed, skipped
):
    """Collection may plan skipped tests; execution needs a real passing test."""
    child = tmp_path / "test_child.py"
    child.write_text(f"import pytest\n\ndef test_case():\n    {body}\n")
    result = release._pytest(
        sys.executable,
        (str(child), "--confcutdir", str(tmp_path)),
        tmp_path,
        "stage",
        collect_only=collect_only,
        extra_env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )
    assert result["valid"] is valid
    assert result["selected"] == 1
    assert result["passed"] == passed
    assert result["skipped"] == skipped


def test_stage_reports_optional_skip_with_executed_test(tmp_path):
    """An optional reference skip remains visible without invalidating real work."""
    child = tmp_path / "test_mixed.py"
    child.write_text(
        'import pytest\n\ndef test_optional():\n    pytest.skip("optional")\n'
        "\ndef test_executed():\n    assert 2 + 2 == 4\n"
    )
    result = release._pytest(
        sys.executable,
        (str(child), "--confcutdir", str(tmp_path)),
        tmp_path,
        "stage",
        extra_env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )
    assert result["valid"]
    assert result["passed"] == result["skipped"] == 1


def test_stage_cannot_reuse_a_previous_success_report(tmp_path):
    """A successful help-only child supplies no evidence of test execution."""
    child = tmp_path / "test_child.py"
    child.write_text("def test_case():\n    assert True\n")
    args = (str(child), "--confcutdir", str(tmp_path))
    env = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    first = release._pytest(sys.executable, args, tmp_path, "stage", extra_env=env)
    assert first["valid"]
    second = release._pytest(
        sys.executable, (*args, "--help"), tmp_path, "stage", extra_env=env
    )
    assert second["exit_code"] == 0
    assert not second["valid"]
    assert second["selected"] is None


def test_relative_output_is_resolved_before_changing_child_directory(
    tmp_path, monkeypatch
):
    """An invocation outside the checkout keeps its logs and report together."""
    monkeypatch.chdir(tmp_path)
    out = Path("results")
    out.mkdir()
    child = tmp_path / "test_child.py"
    child.write_text("def test_case():\n    assert True\n")
    result = release._pytest(
        sys.executable,
        (str(child), "--confcutdir", str(tmp_path)),
        out,
        "stage",
        extra_env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )
    assert result["valid"]
    assert result["passed"] == 1
    assert (out / "stage.json").is_file()
    assert Path(result["log"]).is_absolute()
