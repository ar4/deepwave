"""A requested example check cannot succeed without running an example."""

import pytest

from scripts import run_examples_smoke as runner


@pytest.fixture
def executed(monkeypatch):
    """Use a tiny manifest and record launches without running simulations."""
    entries = [
        runner.ManifestEntry("docs/example_cpu.py", "cpu", ""),
        runner.ManifestEntry("docs/example_gpu.py", "gpu", "requires CUDA"),
    ]
    monkeypatch.setattr(runner, "_load_manifest", lambda path: entries)
    monkeypatch.setattr(runner, "_check_completeness", lambda entries: [])
    paths = []

    def run(entry, timeout):
        paths.append(entry.path)
        return True, 0.0, ""

    monkeypatch.setattr(runner, "_run_one", run)
    return paths


def test_misspelled_selection_is_an_error(executed, capsys):
    """A typo cannot turn a CI step into a successful zero-test run."""
    assert runner.main(["--only", "docs/example_typo.py"]) == 1
    assert "unknown --only" in capsys.readouterr().err
    assert not executed


def test_inactive_category_is_an_error(executed, capsys):
    """Selecting a GPU example also requires enabling the GPU category."""
    assert runner.main(["--only", "docs/example_gpu.py"]) == 1
    assert "no examples selected" in capsys.readouterr().err
    assert not executed


def test_only_selection_runs_the_requested_example(executed):
    """Selecting one category member does not run unrelated CPU examples."""
    assert runner.main(["--only", "docs/example_gpu.py", "--include", "gpu"]) == 0
    assert executed == ["docs/example_gpu.py"]


def test_default_selection_runs_cpu_examples(executed):
    """The ordinary sweep remains usable on a CPU-only machine."""
    assert runner.main([]) == 0
    assert executed == ["docs/example_cpu.py"]
