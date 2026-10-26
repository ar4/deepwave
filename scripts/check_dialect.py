#!/usr/bin/env python3
"""British-English dialect checker (extends the codespell CI job).

Deepwave standardised its shipped prose (docstrings, comments, `.rst`/`.md`
docs) and new filenames on British English (see the spelling-normalisation
pass this script was added by). Codespell catches misspellings but is not a
dialect checker
-- "color" and "optimizer" are both correctly spelled English words, just
not the variety this codebase settled on. This script is the narrow,
explicit complement: a fixed list of American forms that this codebase's
own history shows drift towards, grepped over the same prose scope as
codespell.

This is deliberately NOT a general American-vs-British classifier: several
words that differ by dialect are excluded below because they are bound to
something other than free prose in this codebase (a public API identifier,
a third-party name, or a literal quotation) -- see EXCLUDED_ON_PURPOSE.
Flagging those would just be permanent, undismissable noise.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Same prose/docs scope as the codespell job (.github/workflows/lint.yml),
# plus src/deepwave's .c/.cu/.h comments (codespell already covers those
# via `src/`). `docs/_generated/` is excluded: it is gitignored and
# regenerated from source at doc-build/test time (see docs/conf.py), so
# it cannot carry a real regression -- only a stale local build artifact.
SCOPE_ROOTS = ["src/deepwave", "docs", "tests", "scripts", "benchmarks"]
FILENAME_ROOTS = {*SCOPE_ROOTS, ".github"}
SCOPE_FILES = [
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "CMakeLists.txt",
    "pyproject.toml",
]
SCOPE_SUFFIXES = {".py", ".c", ".cpp", ".cu", ".h", ".rst", ".md"}
EXCLUDE_DIR_PARTS = {"_generated"}
EXCLUDE_FILES = {"scripts/check_dialect.py"}  # The checker quotes rejected forms.

# American spelling forms this codebase has standardised away from in prose.
# One regex per family; each covers the common inflections seen in practice.
# Matched case-insensitively, so a capitalised sentence-initial word (e.g.
# "Optimized weights...") is still caught.
AMERICAN_FORMS: list[str] = [
    r"\bbehaviors?\b",
    r"\bbehavioral\b",
    r"\bfavor(?:s|ed|ing|ite|able|ably)?\b",
    r"\bhonor(?:s|ed|ing)?\b",
    r"\blabor(?:s|ed|ing)?\b",
    r"\bneighbor(?:s|ing)?\b",
    r"\brumors?\b",
    r"\bvapors?\b",
    r"\bfibers?\b",
    r"\btheaters?\b",
    r"\bdefenses?\b",
    r"\boffenses?\b",
    r"\bjudgments?\b",
    r"\backnowledgments?\b",
    r"\bfulfill(?:ed|ing|s)?\b",
    r"\bskillful\b",
    r"\benrollments?\b",
    r"\bmodel(?:ed|ing|er|ers)\b",
    r"\btravel(?:ed|ing)\b",
    r"\bcancel(?:ed|ing)\b",
    r"\blabel(?:ed|ing)\b",
    r"\bsignal(?:ed|ing)\b",
    r"\bparallel(?:ize|izes|ized|izing|ization)\b",
    # The "-ize/-ise" family: one explicit stem per row (not a `\w*iz(e|...)`
    # catch-all, which also matches inside third-party identifiers this
    # codebase must keep verbatim, e.g. `cudaStreamSynchronize`/
    # `cudaDeviceSynchronize`, and inside compiler flags like `-fsanitize`).
    r"\bnormaliz(?:e|es|ed|ing|ation|ations)\b",
    r"\borganiz(?:e|es|ed|ing|ation|ations)\b",
    r"\brealiz(?:e|es|ed|ing|ation|ations)\b",
    r"\binitializ(?:e|es|ed|ing|ation|ations|er|ers)\b",
    r"\bparameteriz(?:e|es|ed|ing|ation|ations)\b",
    r"\breparameteriz(?:e|es|ed|ing|ation|ations)\b",
    r"\bminimiz(?:e|es|ed|ing|ation|ations)\b",
    r"\bmaximiz(?:e|es|ed|ing|ation|ations)\b",
    r"\bgeneraliz(?:e|es|ed|ing|ation|ations)\b",
    r"\bdiscretiz(?:e|ed|ing)\b",
    r"\bdiscretization\b",
    r"\bfinaliz(?:e|es|ed|ing)\b",
    r"\bcustomiz(?:e|es|ed|ing|ation|ations)\b",
    r"\bsynchroniz(?:e|es|ed|ing|ation|ations)\b",
    r"\bvisualiz(?:e|es|ed|ing|ation|ations)\b",
    r"\bspecializ(?:e|es|ed|ing|ation|ations)\b",
    r"\brecogniz(?:e|es|ed|ing|ation|ations|able|ably)\b",
    r"\bemphasiz(?:e|es|ed|ing)\b",
    r"\bcategoriz(?:e|es|ed|ing|ation|ations)\b",
    r"\bprioritiz(?:e|es|ed|ing|ation|ations)\b",
    r"\bstandardiz(?:e|es|ed|ing|ation|ations)\b",
    r"\bvectoriz(?:e|es|ed|ing|ation|ations|able)\b",
    r"\bserializ(?:e|es|ed|ing|ation|ations|able)\b",
    r"\bsymmetriz(?:e|es|ed|ing|ation|ations)\b",
    r"\bequaliz(?:e|es|ed|ing)\b",
    r"\bapologiz(?:e|es|ed|ing)\b",
    r"\bmemoriz(?:e|es|ed|ing|able)\b",
    r"\bformaliz(?:e|es|ed|ing|ation|ations)\b",
    r"\blocaliz(?:e|es|ed|ing|ation|ations|able)\b",
    r"\bstabiliz(?:e|es|ed|ing|ation|ations|able)\b",
    r"\bcapitaliz(?:e|es|ed|ing)\b",
    r"\bmobiliz(?:e|es|ed|ing|able)\b",
    r"\bfamiliariz(?:e|es|ed|ing)\b",
    r"\bsummariz(?:e|es|ed|ing|ation|ations|able)\b",
    r"\blineariz(?:e|es|ed|ing|ation|ations|able)\b",
    r"\bquantiz(?:e|es|ed|ing|able)\b",
    r"\bprivatiz(?:e|es|ed|ing|ation|ations|able)\b",
    r"\bunrecognized\b",
    # Added alongside the -able-suffix fix below: apodization/apodisation
    # is genuine free-running prose in deepwave.transducers' docstrings
    # (distinct from the public `apodization`/`apodization_kind` API
    # names, which stay excluded -- see EXCEPTION_CONTEXTS).
    r"\bapodiz(?:e|es|ed|ing|ation|ations|er|ers)\b",
]

# Deliberate exclusions (see the module docstring). Each is a word family
# that a naive American/British grep would flag, kept out of AMERICAN_FORMS
# above (rather than allowlisted per-site) so the check stays a single flat
# list, matching pyproject.toml's [tool.codespell] ignore-words-list style:
#
#   color -- one literal matplotlib ``color="white"`` keyword argument
#       (docs/example_torch_func.py); third-party API spelling, the one
#       permanent exception.
#   parametrize/parametrized/parametrization -- pytest.mark.parametrize's own
#       spelling; this codebase's prose about test parametrisation
#       deliberately mirrors it (see tests/*.py).
#   optimizer/optimizers -- the `torch.optim` example-code variable-naming
#       convention used throughout docs/example_*.py, not Deepwave API.
#   center/centers/center_t -- a local variable name in several test
#       fixtures, for instance in tests/test_das.py and in
#       tests/test_elastic_vti.py; identifiers keep their spelling. The
#       public transducer parameters are spelt `centre` and
#       `centre_of_curvature`.
#   license -- a metadata field name (pyproject.toml's `license`,
#       `license-files`), not prose.
#   artifact/artifacts -- used throughout as the standard cross-dialect
#       build/compiled-output/imaging-artifact term, not a prose spelling
#       choice.
#   analysis, gauge -- not actually American/British spelling pairs (both
#       spelled identically in either dialect); an earlier survey miscounted
#       these.
#   gray -- always a literal matplotlib ``cmap="gray"``/``color="gray"``
#       argument in this codebase's examples, never prose.
#
# ("modeling" is NOT in this list: it is not excluded from AMERICAN_FORMS
# -- it is caught by the `model(?:ed|ing|er|ers)` row above like any other
# prose use of the word. Its handful of legitimate verbatim-quotation sites
# are handled per-site by EXCEPTION_CONTEXTS below instead, since unlike
# the words above it is also perfectly good prose everywhere else in this
# codebase and must still be flagged there.)
#
# (This list is documentation for maintainers reading this file; it is not
# consulted at runtime -- the words above are simply absent from
# AMERICAN_FORMS.)
EXCLUDED_ON_PURPOSE = (
    "color, parametrize/parametrized/parametrization, optimizer/optimizers, "
    "center/centers/center_t, license, artifact/artifacts, analysis, gauge, "
    "gray"
)

# Explicit allowlist of exception CONTEXTS (as opposed to whole words
# above): (file, exact substring) pairs. A hit is exempted when the
# flagged line contains the substring verbatim -- because that line
# quotes someone else's exact words (a paper/section title, a GitHub
# issue title, a third-party API name or tool's own wording), or because
# the American form is a private-scope identifier this pass is not
# allowed to touch.
#
# This replaces an earlier (file, line number) pinning scheme. Pinning by
# line number is brittle: inserting or removing a paragraph anywhere
# above a pinned site silently shifts it off the pin (this happened once,
# when an edit moved a title from line 17 to line 23, and the exception
# stopped applying without any error -- the line simply started failing
# CI again). Matching on the quoted text itself instead of its line
# number is immune to that: the exception keeps working no matter where
# the text ends up, and only breaks (correctly) if the quoted text itself
# is edited.
#
# Each entry's reason is inline; keep this list to genuine, permanent,
# one-off cases -- if a word needs more than a couple of context
# exceptions, it belongs in the excluded-word discussion above instead.
EXCEPTION_CONTEXTS: dict[str, tuple[str, ...]] = {
    "CMakeLists.txt": ('"loop not vectorized"',),  # Verbatim Clang warning.
    "benchmarks/v0027_parity.py": ("torch.cuda.synchronize",),
    "benchmarks/harness.py": ("torch.cuda.synchronize",),
    "benchmarks/checkpoint_reentrancy.py": ("torch.cuda.synchronize",),
    # The regression fixture deliberately contains the American filenames
    # and a C++ comment that the gate must reject in its temporary repository.
    "tests/test_check_dialect.py": (
        "discretization.py",
        "discretization.rst",
        "example_parameterization.jpg",
        "spatial discretization",
        "solver.cpp:1: 'discretization'",
    ),
    # PyTorch API spelling, not prose.
    "src/deepwave/_engine.py": ("torch.cuda.synchronize(",),
    # Quoted paper/section titles (must not be silently reworded):
    "docs/time_dispersion.rst": (
        # Koene et al. (2018) title
        'wave modeling", *Geophysical Journal International*',
    ),
    "docs/dev/boundary_conditions/ROBERTSSON_FS_SUMMARY.md": (
        # Robertsson (1996) title
        "finite-difference modeling in the presence of topography",
    ),
    "docs/dev/boundary_conditions/CAO_CHEN_PM_SUMMARY.md": (
        # Cao & Chen (2018) title
        "implementing surface topography in elastic-wave finite-difference modeling",
        # Cao & Chen (2018), quoted section title
        "generalized adaptive free-surface expression",
        # verbatim quote, p. T326
        "the generalized stress-image method introduces the symmetry",
        # verbatim quote, p. T323
        "this PM method performs the FD modeling with arbitrary irregular topography",
    ),
    # Quoted GitHub issue title (#116):
    "tests/test_elastic_fluid_solid.py": (
        "Modeling a water layer (Vs=0) above a solid",
    ),
    # Third-party API names/wording that must be reproduced verbatim:
    "tests/test_callbacks_scalar.py": (
        "torch.cuda.synchronize",  # PyTorch API name
    ),
    "tests/test_ergonomics.py": (
        "torch.cuda.synchronize",  # PyTorch API name
    ),
    "tests/test_determinism.py": (
        "torch.cuda.synchronize",  # PyTorch API name
    ),
    "tests/determinism_cases.py": (
        "torch.cuda.synchronize",  # PyTorch API name
    ),
    "tests/test_location_interpolation.py": (
        "torch.cuda.synchronize",  # PyTorch API name
        # PyTorch's own sync-debug warning wording, matched verbatim
        "called a synchronizing",
    ),
    "tests/test_cuda_concurrent_config.py": (
        "torch.cuda.synchronize",  # PyTorch API name
    ),
    "tests/test_native_storage_boundaries.py": (
        "torch.cuda.synchronize",  # PyTorch API name
    ),
    # The migration guide's renamed-symbol table: its
    # whole job is to let a reader searching for the spelling they used find
    # the row that tells them what it became, so the old spelling has to
    # appear verbatim.
    "docs/upgrading.rst": ("``transducers``: ``apodization=``, ``apodization_kind=``",),
}

_COMPILED = [re.compile(p, re.IGNORECASE) for p in AMERICAN_FORMS]


def _scope_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for root_name in SCOPE_ROOTS:
        scope_root = root / root_name
        for path in scope_root.rglob("*"):
            if not path.is_file() or path.suffix not in SCOPE_SUFFIXES:
                continue
            if EXCLUDE_DIR_PARTS & set(path.relative_to(root).parts):
                continue
            if str(path.relative_to(root)) in EXCLUDE_FILES:
                continue
            files.append(path)
    for rel in SCOPE_FILES:
        path = root / rel
        if path.is_file():
            files.append(path)
    return files


def _tracked_filename_hits(root: Path) -> list[str]:
    """Check tracked public filenames, including Python module and image names."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to verify public filenames")
    proc = subprocess.run(  # noqa: S603 - fixed git command
        [git, "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError("git ls-files failed; cannot verify public filenames")
    hits: list[str] = []
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        rel = Path(raw.decode())
        if not any(
            rel.parts[: len(scope.parts)] == scope.parts
            for scope in map(Path, FILENAME_ROOTS)
        ):
            continue
        if EXCLUDE_DIR_PARTS & set(rel.parts):
            continue
        hits.extend(
            f"{rel}: American spelling in filename: {token!r}"
            for token in re.split(r"[_\-.]+", rel.name)
            if any(pattern.fullmatch(token) for pattern in _COMPILED)
        )
    return hits


def main(root: Path = ROOT) -> int:
    """Scan prose and tracked filenames; print any hits and return the exit code."""
    hits: list[str] = []
    try:
        hits.extend(_tracked_filename_hits(root))
    except RuntimeError as exc:
        sys.stderr.write(f"check_dialect: {exc}\n")
        return 2
    for path in _scope_files(root):
        rel = str(path.relative_to(root))
        contexts = EXCEPTION_CONTEXTS.get(rel, ())
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(context in line for context in contexts):
                continue
            for pattern in _COMPILED:
                match = pattern.search(line)
                if match:
                    hits.append(f"{rel}:{lineno}: {match.group(0)!r} -- {line.strip()}")
    if hits:
        sys.stderr.write(
            "check_dialect: found American spelling(s); this codebase uses "
            "British English in prose and filenames (see the module "
            "docstring for the deliberately excluded word families):\n"
        )
        for hit in hits:
            sys.stderr.write(f"  {hit}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
