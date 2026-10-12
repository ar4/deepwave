"""Freezes three properties of Deepwave's public docstrings.

Each of the three was established by hand once, and each is the kind of thing
that regresses silently: a new propagator is added by copying an old one, an
argument entry is reworded, an internal phrase creeps back in during a
refactor. Nothing else in the suite would notice.

1. **No internal jargon.** A frozen deny-list of phrases that once appeared in
   public docstrings and stood in for something the reader could not look up
   ("the A4 convention", "derivation S5-src", a pointer to the commit
   history). They were replaced with the rule, the physics citation or the
   statement they were standing in for; this test stops them coming back, and
   stops new provenance language ("work package", "campaign") arriving with
   them.

2. **Every model argument states its shape.** All nine propagators and their
   ``Module`` classes, every model tensor, one byte-identical sentence, so a
   reader learns from any of them that a leading dimension batches over shots.

3. **The elastic 2-D pressure caveat is stated wherever pressure is offered.**
   In 2-D ``receiver_amplitudes_p`` is the in-plane mean normal stress, which
   under plane strain is not the thermodynamic pressure. It used to be stated
   in exactly one docstring out of ten.

Two mechanics worth knowing before editing this file. Docstrings are
hard-wrapped, so every comparison here is against whitespace-collapsed text --
the sentence breaks across lines differently in each entry and only the
collapsed form is byte-identical. And ``_common.inherit_args`` splices a
functional's ``Args:`` entries onto the matching ``Module.forward`` at import
time, so a class's text is ``getdoc(cls) + getdoc(cls.forward)``; the classes
have no ``Returns:`` section of their own.
"""

import inspect
import re
from typing import Any

import pytest

import deepwave

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat(text: str) -> str:
    """Whitespace-collapsed text, so hard-wrapping cannot affect a match."""
    return " ".join(text.split())


def _raw_docs_of(obj: Any) -> list[str]:
    """Every relevant docstring of ``obj``, dedented but not collapsed."""
    parts = [inspect.getdoc(obj) or ""]
    if inspect.isclass(obj):
        for method in ("__init__", "forward"):
            member = getattr(obj, method, None)
            if member is not None:
                parts.append(inspect.getdoc(member) or "")
    return parts


def _docs_of(obj: Any) -> str:
    """All public docstring text of ``obj``, including an inherited forward."""
    return _flat("\n".join(_raw_docs_of(obj)))


def _public_modules() -> list[tuple[str, Any]]:
    """Every module reachable from ``deepwave.__all__``, plus the package."""
    modules = [("deepwave", deepwave)]
    for name in deepwave.__all__:
        obj = getattr(deepwave, name)
        if inspect.ismodule(obj):
            modules.append((f"deepwave.{name}", obj))
    return modules


def _public_symbols() -> list[tuple[str, Any]]:
    """Every public function/class in ``deepwave`` and its public modules."""
    seen: set[int] = set()
    out: list[tuple[str, Any]] = []
    for module_name, module in _public_modules():
        names = getattr(module, "__all__", None)
        if names is None:
            names = [n for n in dir(module) if not n.startswith("_")]
        for name in names:
            obj = getattr(module, name, None)
            if obj is None or inspect.ismodule(obj):
                continue
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            out.append((f"{module_name}.{name}", obj))
    return out


# ---------------------------------------------------------------------------
# 1. The frozen jargon deny-list
# ---------------------------------------------------------------------------

#: Phrases that must never appear in a public docstring.
#:
#: The first group is internal shorthand that was removed from shipped
#: docstrings: each one named an internal document, derivation or convention
#: that a reader of the published documentation cannot resolve. The second
#: group is development provenance, which belongs in commit messages, not in
#: anything a user reads (a rule that also covers ``docs/`` and the
#: changelog, both swept separately).
#:
#: Add to this list; do not remove from it. A phrase is here because it
#: reached a public docstring once already.
FROZEN_JARGON = (
    # Internal shorthand for documents and conventions that are not published.
    "researcher-lens",
    "through-line",
    "improvement plan",
    "A4 convention",
    "S5-src",
    "M-task",
    "commit history",
    "ROBERTSSON_FS_SUMMARY",
    # Development provenance.
    "work package",
    "campaign",
    "TODO(WP",
)

#: Provenance patterns, as regular expressions, for the forms a plain
#: substring cannot express without false positives.
FROZEN_JARGON_PATTERNS = (
    r"\bWP[0-9]\b",
    r"\bstage [0-9]\b",
)


@pytest.mark.parametrize("phrase", FROZEN_JARGON)
def test_no_frozen_jargon_in_public_module_docstrings(phrase: str) -> None:
    """No public module's ``__doc__`` contains a deny-listed phrase."""
    offenders = [
        name
        for name, module in _public_modules()
        if phrase.lower() in _flat(module.__doc__ or "").lower()
    ]
    assert not offenders, f"{phrase!r} is in the docstring of: {offenders}"


@pytest.mark.parametrize("phrase", FROZEN_JARGON)
def test_no_frozen_jargon_in_public_symbol_docstrings(phrase: str) -> None:
    """No public function or class docstring contains a deny-listed phrase."""
    offenders = [
        name
        for name, obj in _public_symbols()
        if phrase.lower() in _docs_of(obj).lower()
    ]
    assert not offenders, f"{phrase!r} is in the docstring of: {offenders}"


@pytest.mark.parametrize("pattern", FROZEN_JARGON_PATTERNS)
def test_no_provenance_patterns_in_public_docstrings(pattern: str) -> None:
    """No public docstring matches a development-provenance pattern."""
    regex = re.compile(pattern, re.IGNORECASE)
    offenders = [
        name
        for name, obj in [*_public_modules(), *_public_symbols()]
        if regex.search(
            _docs_of(obj) if not inspect.ismodule(obj) else _flat(obj.__doc__ or "")
        )
    ]
    assert not offenders, f"{pattern!r} matches the docstring of: {offenders}"


def test_the_deny_list_is_not_vacuous() -> None:
    """The scan really does read docstrings, so a pass means something."""
    symbols = _public_symbols()
    assert len(symbols) > 50, len(symbols)
    assert any("wave" in _docs_of(obj).lower() for _, obj in symbols)
    # A phrase that IS present must be found, or the matcher is broken.
    assert any("wavespeed" in _docs_of(obj).lower() for _, obj in symbols)


# ---------------------------------------------------------------------------
# 2. Batched-model shapes on every model argument
# ---------------------------------------------------------------------------

#: The sentence every model argument carries, whitespace-collapsed.
#:
#: Anchored on ``of shape`` rather than on ``, of shape``: ``buoyancy``
#: entries lead with the vacuum sentinel and reach the shape via ``It is``
#: instead of a comma.
SHAPE_SENTENCE = (
    "of shape ``[*model_shape]`` or, for a batched model, "
    "``[n_shots, *model_shape]``, where ``model_shape`` has ``ndim`` "
    "entries. See :ref:`batched-models`."
)

#: Every propagator's model arguments. ``scalar_born``'s ``scatter`` is
#: deliberately absent: it is the deprecated alias for ``dv`` and its entry is
#: a pointer, not a model description.
MODEL_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "scalar": ("v",),
    "scalar_born": ("v", "dv"),
    "acoustic": ("v", "rho"),
    "acoustic_born": ("v", "rho", "dv", "drho"),
    "acoustic_vti": ("c11", "c13", "c33", "buoyancy"),
    "viscoacoustic": ("v", "rho", "q"),
    "elastic": ("lamb", "mu", "buoyancy"),
    "elastic_born": ("lamb", "mu", "buoyancy", "dlamb", "dmu", "dbuoyancy"),
    "elastic_vti": ("c11", "c13", "c33", "c55", "c66", "buoyancy"),
}


def _module_class_name(functional: str) -> str:
    """``elastic_vti`` -> ``ElasticVti``."""
    return "".join(word[:1].upper() + word[1:] for word in functional.split("_"))


def _symbol_names() -> list[str]:
    """The 18 public propagator symbols: nine functionals and nine classes."""
    names: list[str] = []
    for functional in MODEL_ARGUMENTS:
        names.append(functional)
        names.append(_module_class_name(functional))
    return names


#: An ``Args:`` entry opens at exactly four spaces of indent (``getdoc``
#: dedents, so the section header sits at column 0 and its entries at four).
_ARG_OPENS = re.compile(r"^ {4}([A-Za-z_][A-Za-z0-9_]*): (.*)$")


def _argument_entries(docstring: str) -> dict[str, str]:
    """Every ``Args:`` entry of one docstring, whitespace-collapsed.

    Parsed from the real indentation rather than from collapsed text. A prose
    colon inside an entry (``... is a vacuum, not an infinitely dense cell:
    it is a *sentinel* ...``) is indistinguishable from the start of a new
    entry once the newlines are gone, so collapsing first silently truncates
    exactly the longest and most important entries.
    """
    entries: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    in_args = False
    for line in docstring.splitlines():
        if line.rstrip() in ("Args:", "Returns:", "Raises:", "Yields:", "Attributes:"):
            if current is not None:
                entries.setdefault(current, _flat(" ".join(lines)))
            current, lines = None, []
            in_args = line.rstrip() == "Args:"
            continue
        if not in_args:
            continue
        opened = _ARG_OPENS.match(line)
        if opened is not None:
            if current is not None:
                entries.setdefault(current, _flat(" ".join(lines)))
            current, lines = opened.group(1), [opened.group(2)]
        elif current is not None:
            lines.append(line.strip())
    if current is not None:
        entries.setdefault(current, _flat(" ".join(lines)))
    return entries


def _argument_entry(obj: Any, argument: str) -> str:
    """The ``Args:`` entry for ``argument``, from any of ``obj``'s docstrings."""
    for docstring in _raw_docs_of(obj):
        entry = _argument_entries(docstring).get(argument)
        if entry:
            return entry
    return ""


#: A ``Returns:`` bullet, ``- name: text``. The indent is captured so that
#: continuation lines (which are indented further) can be told from the next
#: bullet and from the end of the list.
_RETURNS_BULLET = re.compile(r"^(\s*)- ([A-Za-z_][A-Za-z0-9_]*): (.*)$")

#: The section headers that end whatever entry or bullet was being read.
_SECTIONS = (
    "Args:",
    "Returns:",
    "Raises:",
    "Yields:",
    "Attributes:",
    "References:",
    "Examples:",
)


def _returns_bullets(docstring: str) -> dict[str, str]:
    """Every named ``Returns:`` bullet of one docstring, whitespace-collapsed.

    Several propagators describe their output fields as a bullet list under
    ``Returns:`` rather than as ``Args:`` entries, and
    ``vpvsrho_to_lambmubuoyancy`` returns its ``buoyancy`` that way. Read
    from the real indentation, for the same reason ``_argument_entries``
    is: a prose colon inside a bullet is indistinguishable from the start of
    the next one once the newlines are gone.
    """
    bullets: dict[str, str] = {}
    in_returns = False
    current: str | None = None
    lines: list[str] = []
    indent = 0

    def flush() -> None:
        nonlocal current, lines
        if current is not None:
            bullets.setdefault(current, _flat(" ".join(lines)))
        current, lines = None, []

    for line in docstring.splitlines():
        if line.rstrip() in _SECTIONS:
            flush()
            in_returns = line.rstrip() == "Returns:"
            continue
        if not in_returns:
            continue
        opened = _RETURNS_BULLET.match(line)
        if opened is not None:
            flush()
            indent = len(opened.group(1))
            current, lines = opened.group(2), [opened.group(3)]
        elif current is not None:
            if line.strip() and len(line) - len(line.lstrip()) <= indent:
                flush()
            else:
                lines.append(line.strip())
    flush()
    return bullets


def _named_entry(obj: Any, name: str) -> str:
    """The ``Args:`` entry or ``Returns:`` bullet called ``name``."""
    for docstring in _raw_docs_of(obj):
        entry = _argument_entries(docstring).get(name) or _returns_bullets(
            docstring
        ).get(name)
        if entry:
            return entry
    return ""


def _leads_entry(entry: str, lead: str) -> bool:
    """Whether ``lead`` opens ``entry``, after at most one naming sentence.

    Every entry these checks police opens by saying what the argument *is*
    (``The buoyancy (1/density) model.``, ``The background buoyancy
    (1/density) model.``) and then, immediately, the thing the reader must
    not miss. Allowing exactly one leading sentence keeps that shape without
    letting the important sentence drift to the end of a long entry, which
    is the failure this is here to catch.
    """
    if entry.startswith(lead):
        return True
    _, separator, rest = entry.partition(". ")
    return bool(separator) and rest.startswith(lead)


@pytest.mark.parametrize("symbol", _symbol_names())
def test_every_model_argument_states_its_shape(symbol: str) -> None:
    """All 62 model-argument entries carry the batched-shape sentence."""
    functional = symbol if symbol in MODEL_ARGUMENTS else None
    if functional is None:
        functional = next(
            name for name in MODEL_ARGUMENTS if _module_class_name(name) == symbol
        )
    obj = getattr(deepwave, symbol)
    missing = [
        argument
        for argument in MODEL_ARGUMENTS[functional]
        if SHAPE_SENTENCE not in _argument_entry(obj, argument)
    ]
    assert not missing, (
        f"{symbol}: model arguments without the batched-shape sentence: {missing}"
    )


def test_the_shape_sweep_covers_sixty_two_entries() -> None:
    """62 real entries carry the sentence, counted in the docstrings.

    The registry's own arithmetic is a tautology -- it would still say 62
    with every sentence deleted from ``src/``. What is worth asserting is
    that reading the 18 live docstrings finds 62 entries that carry the
    sentence, and that no *other* argument of those symbols has picked it up
    (which would mean the count was made up of the wrong entries).
    """
    assert len(_symbol_names()) == 18
    carrying = 0
    strays: list[str] = []
    for symbol in _symbol_names():
        functional = (
            symbol
            if symbol in MODEL_ARGUMENTS
            else next(
                name for name in MODEL_ARGUMENTS if _module_class_name(name) == symbol
            )
        )
        obj = getattr(deepwave, symbol)
        models = set(MODEL_ARGUMENTS[functional])
        for docstring in _raw_docs_of(obj):
            for argument, entry in _argument_entries(docstring).items():
                if SHAPE_SENTENCE not in entry:
                    continue
                if argument in models:
                    carrying += 1
                else:
                    strays.append(f"{symbol}.{argument}")
    assert not strays, f"the batched-shape sentence is on non-model arguments: {strays}"
    assert carrying == 62, carrying


def test_the_shape_sentence_is_not_vacuous() -> None:
    """A non-model argument does not carry the sentence, so the check bites."""
    assert SHAPE_SENTENCE in _argument_entry(deepwave.scalar, "v")
    assert SHAPE_SENTENCE not in _argument_entry(deepwave.scalar, "dt")
    assert SHAPE_SENTENCE not in _argument_entry(deepwave.scalar, "accuracy")
    # The parser must survive a prose colon inside an entry: elastic's
    # buoyancy entry contains "... dense cell: it is a *sentinel* ...", which
    # a collapsed-text parser mistakes for the start of a new entry.
    buoyancy = _argument_entry(deepwave.elastic, "buoyancy")
    assert "sentinel" in buoyancy
    assert SHAPE_SENTENCE in buoyancy


# ---------------------------------------------------------------------------
# 3. The elastic 2-D pressure caveat
# ---------------------------------------------------------------------------

#: The caveat, whitespace-collapsed, as it appears in the nine short entries.
PRESSURE_CAVEAT = (
    "In 2-D the recorded pressure is the in-plane mean normal stress "
    "``-(sigma_xx + sigma_yy)/2``, which under plane strain is not the "
    "thermodynamic pressure"
)

#: The elastic family: the propagators whose ``receiver_amplitudes_p`` is a
#: mean normal stress rather than a pressure.
ELASTIC_FAMILY = (
    "elastic",
    "elastic_vti",
    "elastic_born",
    "Elastic",
    "ElasticVti",
    "ElasticBorn",
)

#: The acoustic family, which genuinely records a pressure or a stress and is
#: deliberately left alone. Listed so that "we simply forgot these" and "these
#: are excluded on purpose" cannot be confused later.
ACOUSTIC_FAMILY_NOT_CAVEATED = (
    "acoustic",
    "acoustic_born",
    "acoustic_vti",
    "viscoacoustic",
)


#: The entries a reader of a pressure channel actually reads. The caveat has
#: to be in one of these: anywhere else in the docstring it is not where the
#: mistake is made.
PRESSURE_ENTRIES = (
    "receiver_locations_p",
    "sc_receiver_locations_p",
    "bg_receiver_locations_p",
    "receiver_amplitudes_p",
    "sc_receiver_amplitudes_p",
    "bg_receiver_amplitudes_p",
)


def _pressure_entries(obj: Any) -> dict[str, str]:
    """Every pressure-channel entry of ``obj``, by name."""
    found: dict[str, str] = {}
    for name in PRESSURE_ENTRIES:
        entry = _named_entry(obj, name)
        if entry:
            found[name] = entry
    return found


@pytest.mark.parametrize("symbol", ELASTIC_FAMILY)
def test_elastic_family_states_the_2d_pressure_caveat(symbol: str) -> None:
    """The caveat is in the pressure entry itself, not merely somewhere.

    A whole-docstring substring search would pass with the sentence sitting
    under ``accuracy``, or left behind in the module prose after the entry it
    belongs to was reworded. The reader who is about to ask for
    ``receiver_amplitudes_p`` reads that argument's entry, so that is where
    the check has to bite.
    """
    obj = getattr(deepwave, symbol)
    entries = _pressure_entries(obj)
    assert entries, f"{symbol}: offers no pressure entry, so this list is wrong"
    caveated = [name for name, entry in entries.items() if PRESSURE_CAVEAT in entry]
    assert caveated, (
        f"{symbol}: the 2-D pressure caveat is in none of its pressure "
        f"entries ({sorted(entries)}); present elsewhere in the docstring: "
        f"{PRESSURE_CAVEAT in _docs_of(obj)}"
    )


@pytest.mark.parametrize("symbol", ACOUSTIC_FAMILY_NOT_CAVEATED)
def test_acoustic_family_is_deliberately_not_caveated(symbol: str) -> None:
    """The acoustic family records a real pressure and must not claim otherwise."""
    assert PRESSURE_CAVEAT not in _docs_of(getattr(deepwave, symbol))


# ---------------------------------------------------------------------------
# 4. The buoyancy vacuum sentinel leads its entry
# ---------------------------------------------------------------------------

#: The lead the vacuum sentinel must appear in, whitespace-collapsed.
#: ``deepwave.elastic`` keeps a longer canonical entry, so only this opening
#: is asserted rather than the full pointer sentence.
VACUUM_SENTINEL_LEAD = (
    "A cell with ``buoyancy = 0`` is a vacuum, not an infinitely dense cell"
)

#: Every public symbol with a ``buoyancy`` argument or return entry.
BUOYANCY_SYMBOLS = (
    "acoustic_vti",
    "elastic",
    "elastic_born",
    "elastic_vti",
    "AcousticVti",
    "Elastic",
    "ElasticBorn",
    "ElasticVti",
    "vpvsrho_to_lambmubuoyancy",
    "lambmubuoyancy_to_vpvsrho",
)


@pytest.mark.parametrize("symbol", BUOYANCY_SYMBOLS)
def test_buoyancy_entries_lead_with_the_vacuum_sentinel(symbol: str) -> None:
    """A reader of the ``buoyancy`` entry meets the sentinel before anything else.

    ``buoyancy = 0`` is a sentinel, not an arithmetic value: computing
    ``1 / density`` for a vacuum gives an infinity, which Deepwave treats as
    an ordinary very light cell rather than as a free surface. Stating that
    late, or not at all, is what makes the mistake easy.

    The check is on the ``buoyancy`` entry itself and on its opening. A
    whole-docstring search would be satisfied by the sentence appearing under
    a neighbouring argument, or buried at the end of a forty-line entry --
    neither of which is what "leads" means, and both of which are how this
    would regress.
    """
    entry = _named_entry(getattr(deepwave, symbol), "buoyancy")
    assert entry, f"{symbol}: no buoyancy entry in the docstring"
    assert _leads_entry(entry, VACUUM_SENTINEL_LEAD), (
        f"{symbol}: the vacuum sentinel does not lead the buoyancy entry, "
        f"which opens: {entry[:160]!r}"
    )


def test_the_entry_checks_are_not_vacuous() -> None:
    """Moving either sentence out of its entry must fail the check.

    Both checks above once searched the whole docstring, which cannot tell
    "in the right entry" from "somewhere on the page". These are the two
    mutations that proved it, run against synthetic docstrings so that no
    real one has to be edited to keep the guarantee honest.
    """
    moved_buoyancy = (
        "A propagator.\n"
        "\n"
        f"{VACUUM_SENTINEL_LEAD}; stated here in the body instead.\n"
        "\n"
        "Args:\n"
        "    buoyancy: The buoyancy (1/density) model. It is of shape\n"
        "        ``[*model_shape]``.\n"
        "    accuracy: The finite difference order.\n"
    )
    entry = _argument_entries(moved_buoyancy)["buoyancy"]
    assert VACUUM_SENTINEL_LEAD in _flat(moved_buoyancy)
    assert not _leads_entry(entry, VACUUM_SENTINEL_LEAD)

    trailing_buoyancy = (
        "A propagator.\n"
        "\n"
        "Args:\n"
        "    buoyancy: The buoyancy (1/density) model. It is of shape\n"
        "        ``[*model_shape]``, and note also that\n"
        f"        {VACUUM_SENTINEL_LEAD}.\n"
    )
    entry = _argument_entries(trailing_buoyancy)["buoyancy"]
    assert VACUUM_SENTINEL_LEAD in entry
    assert not _leads_entry(entry, VACUUM_SENTINEL_LEAD)

    moved_caveat = (
        "A propagator.\n"
        "\n"
        "Args:\n"
        "    receiver_locations_p: Pressure receiver locations.\n"
        f"    accuracy: The finite difference order. {PRESSURE_CAVEAT}.\n"
    )
    entries = _argument_entries(moved_caveat)
    assert PRESSURE_CAVEAT in _flat(moved_caveat)
    assert PRESSURE_CAVEAT not in entries["receiver_locations_p"]
    assert PRESSURE_CAVEAT in entries["accuracy"]


def test_the_returns_bullet_parser_reads_a_real_docstring() -> None:
    """``Returns:`` bullets are parsed, including one with a prose colon."""
    bullets = _returns_bullets(inspect.getdoc(deepwave.vpvsrho_to_lambmubuoyancy) or "")
    assert set(bullets) >= {"lambda", "mu", "buoyancy"}
    assert bullets["buoyancy"].startswith("The buoyancy (1/density) model.")
    # A bullet stops at the next bullet, not at the end of the section.
    assert "mu" not in bullets["lambda"]
