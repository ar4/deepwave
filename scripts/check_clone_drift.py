#!/usr/bin/env python3
"""Clone-drift checker for deliberately duplicated C/CUDA source blocks.

Deepwave's propagator sources duplicate a bounded set of small kernel
functions and host-side launch helpers verbatim across independently edited
files (e.g. ``elastic_vti_kernels.h`` redeclares several of
``elastic_kernels.h``'s per-cell update functions, ported unchanged because
the two propagators share the same staggered-grid arithmetic for those
cells). Each pair is a real, editable copy in two different files -- not one
file included twice, which cannot drift from itself -- so nothing stops one
copy being fixed or tweaked without the other. This script re-extracts each
named function from both files listed in ``clone_manifest.json`` and fails
if the two copies are no longer byte-identical, which is the only signal
that one side missed an edit the other side got.

A few pairs are copies that cannot be byte-identical, because one side
carries preprocessor branches the other side's translation unit is never
built with (``elastic_bc.h`` serves ``elastic.c``, which builds at
``DW_NDIM==1``; ``elastic_vti.c``'s copies of the same functions do not,
because ``elastic_vti`` is only built for ``DW_NDIM`` 2 and 3). Such an
entry carries an ``expected_delta``: the exact unified diff from ``file_a``'s
block to ``file_b``'s, recorded line for line. The pair then has to differ in
precisely that recorded way and in no other -- an edit to either copy that
the other does not get changes the diff and still fails, so these entries are
policed exactly as tightly as the byte-identical ones; only the one recorded,
justified difference is excused. ``expected_delta`` is deliberately a frozen
diff rather than a normalisation rule: a rule that rewrote one side into the
other would also hide any *future* difference of the same shape, whereas a
frozen diff cannot hide anything it does not already spell out. A recorded
diff carries context lines, so an edit made *identically* to both copies
still fails if it lands within three lines of a recorded hunk: that is the
safe direction (a red check, with the new diff printed ready to paste into
the manifest), not a missed drift. The same goes for the (very unlikely)
case of a future ``difflib`` picking a different alignment for the same two
blocks: it re-records, it never silently passes.

Recording a delta is only legitimate when the difference has been shown not
to change the generated code -- for the ``elastic_vti.c`` entries that was a
preprocessed token-stream comparison (``gcc -E``) over every built variant
(``DW_NDIM`` 2/3 x ``DW_ACCURACY`` 2/4/6/8 x float/double x +/-``DW_MPML``);
see those entries' notes. If a pair that records a delta ever becomes
byte-identical, the entry fails too, so a stale recording cannot survive.

Extraction locates a definition by its identifier (e.g. ``forward_vx_cell``,
or ``FUNC(forward_vx_cell)`` where the codebase's macro-based per-variant
name mangling is written out literally in source), then the following ``(``
through its matching ``)`` (the parameter list), then the following ``{``
through its matching ``}`` (the body) -- ignoring parentheses and braces
inside comments or string/character literals. The extracted block runs from
the identifier itself through the closing brace; a preceding doc comment
(which legitimately differs between call sites, e.g. naming the file it was
ported from) is deliberately not part of the compared block.

Several kernels in this codebase share one closing brace across a
``#if DW_NDIM == 3 / #elif DW_NDIM == 2 / #else`` chain, each branch opening
its own ``if (...) {`` before a single common ``#endif``-following ``}``
closes whichever branch was compiled (e.g. ``elastic.cu``'s
``combine_grad``/``forward_kernel_v``). A brace count that does not
understand preprocessor branches sees every branch's opening brace but only
one closing brace, and never rebalances. Locating the true end of such a
function's body therefore first resolves each ``#if``/``#elif``/``#else``
chain to only its first branch (the rest blanked out, same offsets
preserved) before brace/paren matching runs; the extracted text sliced back
out of the original source still includes every branch, unchanged.
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "scripts" / "clone_manifest.json"


@dataclass
class ManifestEntry:
    """One (fileA, fileB, identifier) clone-drift manifest entry."""

    file_a: str
    file_b: str
    identifier: str
    note: str
    # ``expected_delta`` is None for the usual byte-identical pair.
    # Otherwise it is the exact unified diff (as produced by
    # _unified_delta) that the two copies are allowed -- and required
    # -- to differ by.
    expected_delta: tuple[str, ...] | None = None


class ExtractionError(Exception):
    """A named block could not be uniquely located in a source file."""


def _mask_comments_and_strings(text: str) -> str:
    """Blank out comment and string/char-literal interiors, same length.

    Newlines are always preserved (so line numbers computed from the masked
    text still match the original), everything else inside a comment or
    literal becomes a space. This lets identifier search and paren/brace
    balancing run directly on the result without being confused by a
    matching character that is actually inside a comment or a string.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    state = "normal"
    while i < n:
        c = text[i]
        if state == "normal":
            if c == "/" and i + 1 < n and text[i + 1] == "/":
                out.append("  ")
                i += 2
                state = "line_comment"
                continue
            if c == "/" and i + 1 < n and text[i + 1] == "*":
                out.append("  ")
                i += 2
                state = "block_comment"
                continue
            if c == '"':
                out.append(" ")
                i += 1
                state = "string"
                continue
            if c == "'":
                out.append(" ")
                i += 1
                state = "char"
                continue
            out.append(c)
            i += 1
            continue
        if state == "line_comment":
            if c == "\n":
                out.append("\n")
                state = "normal"
            else:
                out.append(" ")
            i += 1
            continue
        if state == "block_comment":
            if c == "*" and i + 1 < n and text[i + 1] == "/":
                out.append("  ")
                i += 2
                state = "normal"
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        if state == "string":
            if c == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if c == '"':
                out.append(" ")
                i += 1
                state = "normal"
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        if state == "char":
            if c == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if c == "'":
                out.append(" ")
                i += 1
                state = "normal"
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
    return "".join(out)


_DIRECTIVE_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b")


def _blank_inactive_branches(text: str) -> str:
    """Blank every non-first branch of each #if/#elif/#else chain, same length.

    Directive lines themselves (``#if``, ``#elif``, ...) are also blanked --
    they are not C code. Only the first branch of any chain is left
    standing, so nested/chained conditionals collapse to one unambiguous
    path and brace/paren matching over the result balances correctly (see
    the module docstring).
    """
    out_parts: list[str] = []
    stack: list[bool] = []
    for line in text.splitlines(keepends=True):
        body = line[:-1] if line.endswith("\n") else line
        match = _DIRECTIVE_RE.match(body)
        if match:
            kind = match.group(1)
            if kind in ("if", "ifdef", "ifndef"):
                stack.append(True)
            elif kind in ("elif", "else") and stack:
                stack[-1] = False
            elif kind == "endif" and stack:
                stack.pop()
            out_parts.append(" " * len(body) + line[len(body) :])
            continue
        if all(stack):
            out_parts.append(line)
        else:
            out_parts.append(" " * len(body) + line[len(body) :])
    return "".join(out_parts)


def _navigation_mask(text: str) -> str:
    """Combine comment/string masking with inactive-branch blanking.

    The result is only used to locate identifiers and balance parens/braces;
    the actual compared block text is always sliced from the untouched
    original source at the offsets this mask helps find.
    """
    comment_masked = _mask_comments_and_strings(text)
    branch_masked = _blank_inactive_branches(text)
    out = []
    for orig, via_comments, via_branches in zip(text, comment_masked, branch_masked):
        if orig == "\n":
            out.append("\n")
        elif via_comments != orig or via_branches != orig:
            out.append(" ")
        else:
            out.append(orig)
    return "".join(out)


def _matching_close(masked: str, open_pos: int, open_ch: str, close_ch: str) -> int:
    """Return the index just past the ``close_ch`` matching ``masked[open_pos]``."""
    depth = 1
    i = open_pos + 1
    n = len(masked)
    while i < n and depth > 0:
        if masked[i] == open_ch:
            depth += 1
        elif masked[i] == close_ch:
            depth -= 1
        i += 1
    if depth != 0:
        raise ExtractionError(
            f"unbalanced {open_ch!r}/{close_ch!r} starting at offset {open_pos}"
        )
    return i


def _find_definitions(text: str, identifier: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of every function definition of ``identifier``.

    ``start`` is the identifier's own start offset; ``end`` is one past the
    matching closing brace of the function body.
    """
    masked = _navigation_mask(text)
    results: list[tuple[int, int]] = []
    search_from = 0
    while True:
        idx = masked.find(identifier, search_from)
        if idx == -1:
            break
        search_from = idx + 1
        after = idx + len(identifier)
        j = after
        while j < len(masked) and masked[j] in " \t\r\n":
            j += 1
        if j >= len(masked) or masked[j] != "(":
            continue
        params_end = _matching_close(masked, j, "(", ")")
        k = params_end
        while k < len(masked) and masked[k] in " \t\r\n":
            k += 1
        if k >= len(masked) or masked[k] != "{":
            continue  # a prototype/declaration, or a call site -- not a definition
        body_end = _matching_close(masked, k, "{", "}")
        results.append((idx, body_end))
    return results


def extract_block(path: Path, identifier: str) -> str:
    """Return the exact source text of ``identifier``'s definition in ``path``."""
    text = path.read_text(encoding="utf-8")
    defs = _find_definitions(text, identifier)
    if not defs:
        raise ExtractionError(f"no definition of {identifier!r} found in {path}")
    if len(defs) > 1:
        lines = ", ".join(str(text.count("\n", 0, start) + 1) for start, _ in defs)
        raise ExtractionError(
            f"{identifier!r} has {len(defs)} definitions in {path} "
            f"(at lines {lines}); manifest identifiers must be unambiguous"
        )
    start, end = defs[0]
    return text[start:end]


def _load_manifest(path: Path) -> list[ManifestEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries: list[ManifestEntry] = []
    for item in raw:
        delta = item.get("expected_delta")
        entries.append(
            ManifestEntry(
                file_a=item["file_a"],
                file_b=item["file_b"],
                identifier=item["identifier"],
                note=item.get("note", ""),
                expected_delta=None if delta is None else tuple(delta),
            )
        )
    return entries


def _unified_delta(entry: ManifestEntry, block_a: str, block_b: str) -> tuple[str, ...]:
    """Return the unified diff from ``block_a`` to ``block_b``, line by line.

    This is the canonical form an entry's ``expected_delta`` records, so it
    is also what a maintainer pastes into the manifest when deliberately
    introducing (or re-recording) a justified difference: the failure report
    below prints exactly this.
    """
    return tuple(
        difflib.unified_diff(
            block_a.splitlines(),
            block_b.splitlines(),
            fromfile=entry.file_a,
            tofile=entry.file_b,
            lineterm="",
        )
    )


def _indent(lines: tuple[str, ...]) -> str:
    return "\n".join(f"      {line}" for line in lines) if lines else "      (none)"


def main() -> int:
    """Check every manifest pair; print a report and return the exit code."""
    entries = _load_manifest(MANIFEST_PATH)
    failures: list[str] = []
    for entry in entries:
        path_a = ROOT / entry.file_a
        path_b = ROOT / entry.file_b
        try:
            block_a = extract_block(path_a, entry.identifier)
        except ExtractionError as exc:
            failures.append(f"{entry.identifier}: {exc}")
            continue
        try:
            block_b = extract_block(path_b, entry.identifier)
        except ExtractionError as exc:
            failures.append(f"{entry.identifier}: {exc}")
            continue
        if entry.expected_delta is None:
            if block_a != block_b:
                failures.append(
                    f"{entry.identifier}: {entry.file_a} and {entry.file_b} have "
                    f"diverged (were expected to be byte-identical -- {entry.note})"
                )
            continue
        # This pair records a justified, frozen difference: require it to be
        # present and to be the *only* difference (see the module docstring).
        if block_a == block_b:
            failures.append(
                f"{entry.identifier}: {entry.file_a} and {entry.file_b} are now "
                f"byte-identical, but the manifest still records an "
                f'"expected_delta" for them; drop that field from the entry'
            )
            continue
        actual = _unified_delta(entry, block_a, block_b)
        if actual != entry.expected_delta:
            failures.append(
                f"{entry.identifier}: {entry.file_a} and {entry.file_b} differ in "
                f"a way the manifest does not record.\n"
                f"    entry note: {entry.note}\n"
                f'    manifest "expected_delta":\n{_indent(entry.expected_delta)}\n'
                f"    actual difference:\n{_indent(actual)}"
            )
    if failures:
        sys.stderr.write(
            "check_clone_drift: the following manifest pairs no longer "
            f"match ({len(failures)} of {len(entries)}):\n"
        )
        for failure in failures:
            sys.stderr.write(f"  {failure}\n")
        sys.stderr.write(
            "\nEither the drift was intentional (update both copies to "
            "match, or edit scripts/clone_manifest.json if the pair is no "
            "longer a true clone) or one copy has an unintended edit the "
            "other is missing.\n"
        )
        return 1
    sys.stdout.write(f"check_clone_drift: {len(entries)} manifest pair(s) match.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
