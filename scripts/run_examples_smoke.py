#!/usr/bin/env python3
"""Run docs/example_*.py scripts in smoke mode as an automated check.

Each example honours the ``DEEPWAVE_SMOKE=1`` environment variable by
shrinking its problem size (fewer shots, iterations, grid cells, and so on)
while still exercising its real code path and asserting a sanity condition
at the end. This script discovers every example listed in
``scripts/examples_manifest.json``, checks that the manifest and the set of
``docs/example_*.py`` files on disk agree exactly (so an example can never be
silently left out of continuous integration), and runs each example whose
category has been selected as a subprocess in smoke mode, reporting a
pass/fail/skip line for every example and an overall exit status.

Examples that require a CUDA device, two or more CUDA devices, or a data
file that is not part of this repository are recorded in the manifest with
an explanatory reason and are skipped unless explicitly requested with
``--include``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "scripts" / "examples_manifest.json"
DOCS_DIR = ROOT / "docs"

ALLOWED_CATEGORIES = ("cpu", "gpu", "multi_gpu", "external_data")
SELECTABLE_CATEGORIES = ("gpu", "multi_gpu", "external_data")


@dataclass
class ManifestEntry:
    """One example's entry in ``scripts/examples_manifest.json``."""

    path: str
    category: str
    reason: str


class ManifestError(Exception):
    """The manifest is malformed or disagrees with the files on disk."""


def _load_manifest(path: Path) -> list[ManifestEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for item in raw["examples"]:
        category = item["category"]
        if category not in ALLOWED_CATEGORIES:
            raise ManifestError(
                f"{item['path']}: unknown category {category!r}, must be "
                f"one of {ALLOWED_CATEGORIES}"
            )
        entries.append(
            ManifestEntry(path=item["path"], category=category, reason=item["reason"])
        )
    return entries


def _check_completeness(entries: list[ManifestEntry]) -> list[str]:
    """Return a list of error lines if the manifest and disk disagree."""
    manifest_paths = {entry.path for entry in entries}
    disk_paths = {p.relative_to(ROOT).as_posix() for p in DOCS_DIR.glob("example_*.py")}
    missing_from_manifest = sorted(disk_paths - manifest_paths)
    missing_from_disk = sorted(manifest_paths - disk_paths)
    errors = [
        f"present on disk but missing from manifest: {p}" for p in missing_from_manifest
    ]
    errors.extend(
        f"listed in manifest but missing from disk: {p}" for p in missing_from_disk
    )
    return errors


def _run_one(entry: ManifestEntry, timeout: float) -> tuple[bool, float, str]:
    """Run a single example in smoke mode; return (passed, seconds, output)."""
    abs_path = ROOT / entry.path
    env = dict(os.environ)
    env["DEEPWAVE_SMOKE"] = "1"
    env["MPLBACKEND"] = "Agg"
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv: this interpreter, a manifest-listed script
                [sys.executable, str(abs_path)],
                cwd=tmp_dir,
                env=env,
                timeout=timeout,
                capture_output=True,
                text=True,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - start
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            output = (
                f"TIMEOUT after {timeout:.1f}s\nstdout:\n{stdout}\nstderr:\n{stderr}\n"
            )
            return False, elapsed, output
    elapsed = time.perf_counter() - start
    output = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
    return result.returncode == 0, elapsed, output


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run selected examples, print a report and return status."""
    parser = argparse.ArgumentParser(
        description=(
            "Run docs/example_*.py scripts in DEEPWAVE_SMOKE=1 mode as an "
            "automated check."
        )
    )
    parser.add_argument(
        "--include",
        action="append",
        choices=SELECTABLE_CATEGORIES,
        default=[],
        help=(
            "Additionally run examples of this category (may be repeated). "
            "The 'cpu' category always runs."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-example timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "Restrict the run to this example path, e.g. "
            "docs/example_acoustic_born.py (may be repeated). Examples "
            "excluded by --only are reported as skipped. The example must "
            "still belong to a selected category."
        ),
    )
    args = parser.parse_args(argv)

    try:
        entries = _load_manifest(MANIFEST_PATH)
    except (OSError, json.JSONDecodeError, KeyError, ManifestError) as exc:
        sys.stderr.write(f"run_examples_smoke: cannot load manifest: {exc}\n")
        return 1

    completeness_errors = _check_completeness(entries)
    if completeness_errors:
        sys.stderr.write(
            "run_examples_smoke: manifest and docs/example_*.py disagree "
            f"({len(completeness_errors)} problem(s)):\n"
        )
        for line in completeness_errors:
            sys.stderr.write(f"  {line}\n")
        return 1

    selected_categories = {"cpu", *args.include}
    only_set = set(args.only) if args.only is not None else None
    if only_set is not None:
        unknown = only_set - {entry.path for entry in entries}
        if unknown:
            sys.stderr.write(
                "run_examples_smoke: unknown --only path(s): "
                + ", ".join(sorted(unknown))
                + "\n"
            )
            return 1
    if not any(
        entry.category in selected_categories
        and (only_set is None or entry.path in only_set)
        for entry in entries
    ):
        sys.stderr.write(
            "run_examples_smoke: no examples selected; check --only and use "
            "--include for examples requiring a GPU or external data.\n"
        )
        return 1

    passed = 0
    failed = 0
    skipped = 0
    for entry in sorted(entries, key=lambda e: e.path):
        if entry.category not in selected_categories:
            print(f"SKIP {entry.path} [{entry.category}] {entry.reason}")
            skipped += 1
            continue
        if only_set is not None and entry.path not in only_set:
            print(f"SKIP {entry.path} [{entry.category}] excluded by --only selection")
            skipped += 1
            continue
        ok, elapsed, output = _run_one(entry, args.timeout)
        if ok:
            print(f"PASS {entry.path} ({elapsed:.1f}s)")
            passed += 1
        else:
            print(f"FAIL {entry.path} ({elapsed:.1f}s)")
            print(output)
            failed += 1

    print(
        f"SUMMARY: {passed} passed, {failed} failed, {skipped} skipped "
        f"(of {len(entries)} examples in manifest)"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
