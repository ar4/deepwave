"""The dialect gate covers tracked filenames and native source comments."""

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import check_dialect


def test_american_module_page_and_cpp_comment_fail_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A content-only check must not silently accept renamed public files."""
    package = tmp_path / "src" / "deepwave"
    docs = tmp_path / "docs"
    package.mkdir(parents=True)
    docs.mkdir()
    old_module = package / "discretization.py"
    old_page = docs / "discretization.rst"
    old_image = docs / "example_parameterization.jpg"
    native = package / "solver.cpp"
    old_module.write_text("", encoding="utf-8")
    old_page.write_text("", encoding="utf-8")
    old_image.write_bytes(b"")
    native.write_text("// spatial discretization\n", encoding="utf-8")
    git = shutil.which("git")
    assert git is not None
    subprocess.run([git, "init", "-q"], cwd=tmp_path, check=True)  # noqa: S603
    subprocess.run([git, "add", "-A"], cwd=tmp_path, check=True)  # noqa: S603

    assert check_dialect.main(tmp_path) == 1
    output = capsys.readouterr().err
    assert "src/deepwave/discretization.py: American spelling in filename" in output
    assert "docs/discretization.rst: American spelling in filename" in output
    assert "docs/example_parameterization.jpg: American spelling in filename" in output
    assert "src/deepwave/solver.cpp:1: 'discretization'" in output

    old_module.rename(package / "discretisation.py")
    old_page.rename(docs / "discretisation.rst")
    old_image.rename(docs / "example_parameterisation.jpg")
    native.write_text("// spatial discretisation\n", encoding="utf-8")
    subprocess.run([git, "add", "-A"], cwd=tmp_path, check=True)  # noqa: S603
    assert check_dialect.main(tmp_path) == 0
