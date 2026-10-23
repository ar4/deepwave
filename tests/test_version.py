"""Tests that the version is consistently single-sourced.

The canonical version lives in ``src/deepwave/_version.py``. The build metadata
(``pyproject.toml``) and the documentation (``docs/conf.py``) now derive the
version from that file automatically, but ``CITATION.cff`` is a static file that
external tools (GitHub, Zenodo) read directly, so it cannot be derived at build
time. This test guards against ``CITATION.cff`` drifting from the canonical
version.
"""

import pathlib
import re

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _canonical_version() -> str:
    """The single source of truth: src/deepwave/_version.py."""
    text = (_REPO_ROOT / "src" / "deepwave" / "_version.py").read_text()
    match = re.search(r"""__version__ = ['"]([^'"]+)['"]""", text)
    assert match is not None, "Could not parse __version__ from _version.py"
    return match.group(1)


def test_citation_version_matches_source() -> None:
    """The version in CITATION.cff must match the canonical version."""
    canonical = _canonical_version()
    citation_text = (_REPO_ROOT / "CITATION.cff").read_text()
    match = re.search(r"^version:\s*(\S+)\s*$", citation_text, re.MULTILINE)
    assert match is not None, "Could not find a version field in CITATION.cff"
    citation_version = match.group(1).strip("'\"")
    assert citation_version == canonical, (
        f"CITATION.cff version ({citation_version!r}) does not match the "
        f"canonical src/deepwave/_version.py ({canonical!r}). "
        "Update CITATION.cff to match the canonical version."
    )
