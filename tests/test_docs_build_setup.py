"""Check documentation setup diagnoses an incompatible installed package."""

import pathlib
import runpy
import sys
import types
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("matching_version", [True, False])
def test_docs_check_installed_version(
    monkeypatch: pytest.MonkeyPatch, matching_version: bool, tmp_path: pathlib.Path
) -> None:
    """Accept the expected version or explain how to install it."""
    pytest.importorskip("sphinx")
    from sphinx.errors import ConfigError

    conf = runpy.run_path(
        str(pathlib.Path(__file__).resolve().parents[1] / "docs" / "conf.py")
    )
    package = tmp_path / "installed" / "deepwave"
    package.mkdir(parents=True)
    source_package = pathlib.Path(__file__).resolve().parents[1] / "src/deepwave"
    for path in source_package.glob("*.py"):
        (package / path.name).write_bytes(path.read_bytes())
    installed = types.ModuleType("deepwave")
    installed.__file__ = str(package / "__init__.py")
    installed.__version__ = conf["version"] if matching_version else "0.0.0"
    monkeypatch.setitem(sys.modules, "deepwave", installed)
    app = Mock()
    if matching_version:
        conf["setup"](app)
        assert app.connect.call_count == 3
        return
    with pytest.raises(ConfigError, match="python -m pip install") as error:
        conf["setup"](app)
    assert installed.__file__ in str(error.value)
    app.connect.assert_not_called()


@pytest.mark.parametrize("difference", ["changed", "missing", "extra"])
def test_docs_reject_different_sources_at_the_same_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, difference: str
) -> None:
    """A stale same-version installation cannot silently document another API."""
    pytest.importorskip("sphinx")
    from sphinx.errors import ConfigError

    conf = runpy.run_path(
        str(pathlib.Path(__file__).resolve().parents[1] / "docs/conf.py")
    )
    package = tmp_path / "installed" / "deepwave"
    package.mkdir(parents=True)
    source_package = pathlib.Path(__file__).resolve().parents[1] / "src/deepwave"
    for path in source_package.glob("*.py"):
        (package / path.name).write_bytes(path.read_bytes())
    if difference == "changed":
        (package / "scalar.py").write_text("# stale candidate\n")
    elif difference == "missing":
        (package / "scalar.py").unlink()
    else:
        (package / "obsolete.py").write_text("# leftover module\n")
    installed = types.ModuleType("deepwave")
    installed.__file__ = str(package / "__init__.py")
    installed.__version__ = conf["version"]
    monkeypatch.setitem(sys.modules, "deepwave", installed)
    app = Mock()
    with pytest.raises(ConfigError, match="sources that differ") as error:
        conf["setup"](app)
    assert installed.__file__ in str(error.value)
    assert ("obsolete.py" if difference == "extra" else "scalar.py") in str(error.value)
    app.connect.assert_not_called()
