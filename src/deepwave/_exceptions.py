"""Deepwave's own warning and exception types.

Every name defined here is re-exported at the package root and is documented
under its public spelling (``deepwave.DeepwaveError``, not
``deepwave._exceptions.DeepwaveError``): each class sets
``__module__ = "deepwave"`` so that its ``repr``, its ``pickle`` round trip,
and Sphinx all agree with the one import path users are given. The module
itself is private precisely so that there is only ever one such path.

Three independent things live here:

* :class:`DeepwaveFutureWarning`, the category a deprecated Deepwave
  spelling is announced under. It subclasses ``FutureWarning`` because a
  deprecated *spelling* is a message for the person who wrote the call, and
  Python shows ``FutureWarning`` by default wherever the call was written,
  unlike ``DeprecationWarning``, which the default filters hide outside
  ``__main__``. It is deliberately *not* a :class:`DeepwaveWarning`: a
  deprecated spelling and a physical surprise are different families, and
  filtering one must not silence the other.
  :func:`warn_deprecated_name` is its one emitter, so that every deprecated
  name in Deepwave says the same three things -- the replacement, the
  removal release, and how to filter the category -- in the same words.
* :class:`DeepwaveWarning` and its subclasses, one per *cause*. A warning
  raised under one of these names its own class as the way to filter it, so
  the one-line recipe is always in the message the user is reading.
  ``UserWarning`` is the base, so a filter that already targets
  ``UserWarning`` keeps working -- which is also how the plain
  ``UserWarning`` messages that predate this hierarchy, and are not
  classified under it, are still filtered.
* :class:`DeepwaveError` and its mixed-in subclasses. The mixing is the
  point: :class:`DeepwaveValueError` is *both* a ``DeepwaveError`` and a
  ``ValueError``, so an existing ``except ValueError`` around a Deepwave call
  keeps catching exactly what it caught before, while new code can write
  ``except deepwave.DeepwaveError`` to catch everything Deepwave raises
  deliberately and nothing else. Raising one of these is therefore never a
  breaking change for a caller, which is why modules can convert their
  ``raise`` sites as they are touched rather than all at once.

The builtin bases mirror the exception types Deepwave already raises:
``ValueError`` (invalid argument value), ``TypeError`` (wrong argument type),
``RuntimeError`` (a valid request this build/backend/configuration cannot
serve), ``NotImplementedError`` (a combination that is not implemented yet),
``FloatingPointError`` (a diverged simulation; see
:func:`deepwave.check_finite`), and ``OSError`` (the filesystem refused an
operation Deepwave asked for on the user's behalf).
"""

import pathlib
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    import types

__all__ = [
    "DeepwaveError",
    "DeepwaveFloatingPointError",
    "DeepwaveFutureWarning",
    "DeepwaveNotImplementedError",
    "DeepwaveOSError",
    "DeepwaveRuntimeError",
    "DeepwaveTypeError",
    "DeepwaveValueError",
    "DeepwaveWarning",
    "MigrationWarning",
    "PmlFreqInferenceWarning",
    "ZeroWidthFaceWarning",
]

#: The planned earliest removal release for names deprecated in 0.1.0. They
#: remain available through both subsequent minor releases, 0.2 and 0.3.
#: Single-sourced here, the one module every other module can import without a
#: cycle, so that no deprecation message can quietly name a different version
#: from the rest.
DEPRECATION_REMOVAL_VERSION = "0.4.0"


# The directories whose frames are Deepwave's own plumbing rather than a
# caller: this package, and PyTorch (whose ``nn.Module.__call__`` machinery
# sits between a user's ``Module(...)`` call and the ``forward`` body that
# runs the propagator). A warning attributed to either of them tells the user
# nothing about which of their own lines caused it.
_PLUMBING_DIRECTORIES = (
    pathlib.Path(__file__).resolve().parent,
    pathlib.Path(torch.__file__).resolve().parent,
)

#: Memoises :func:`_is_plumbing` by ``co_filename``. A frame walk asks the
#: same handful of filenames every time, and resolving a path is a syscall,
#: so caching keeps the walk's cost below that of formatting the message it
#: is attributing.
_plumbing_by_filename: dict[str, bool] = {}

# A plan collects only warnings emitted by Deepwave on its own execution
# context. Python's warning filters and showwarning hook stay untouched.
_warning_sink: ContextVar[Optional[list[str]]] = ContextVar(
    "deepwave_warning_sink", default=None
)


@contextmanager
def capture_deepwave_warnings() -> Iterator[list[str]]:
    """Collect Deepwave warning text on the current execution context."""
    caught: list[str] = []
    token = _warning_sink.set(caught)
    try:
        yield caught
    finally:
        _warning_sink.reset(token)


def _is_plumbing(filename: str) -> bool:
    """Whether a frame's source file belongs to Deepwave or PyTorch."""
    cached = _plumbing_by_filename.get(filename)
    if cached is None:
        try:
            path = pathlib.Path(filename).resolve()
            cached = any(
                path.is_relative_to(directory) for directory in _PLUMBING_DIRECTORIES
            )
        except OSError:
            # A frame with no real file behind it (``<stdin>``, ``exec``) is
            # not plumbing: it is where the user typed the call.
            cached = False
        _plumbing_by_filename[filename] = cached
    return cached


def stacklevel_outside_deepwave(fallback: int) -> int:
    """The ``stacklevel`` of the first frame outside Deepwave and PyTorch.

    Counted from the caller of this function, in the units ``warnings.warn``
    takes there: ``1`` is the caller's own frame, ``2`` its caller, and so on.

    A fixed ``stacklevel`` can only be right for one call path, and the paths
    that reach a Deepwave warning differ in depth: a propagator function is
    one frame below the user, an ``nn.Module`` adds
    ``torch.nn.Module.__call__``'s frames, and a ``checkpoint_segments=`` call
    reaches the segment driver through the engine's delegation. Counting the
    frames instead attributes the warning to the user's own line in every one
    of them. ``skip_file_prefixes`` (Python 3.12) would do this inside
    ``warnings.warn``; Deepwave supports Python 3.9, so the walk is here.

    Args:
        fallback: The ``stacklevel`` to use when every frame is Deepwave's or
            PyTorch's own (nothing outside them called this), so that the
            warning is still emitted somewhere sensible.

    Returns:
        The ``stacklevel`` reaching the first frame outside both packages, or
        ``fallback`` if there is none.

    """
    frame: Optional[types.FrameType] = sys._getframe(1)  # noqa: SLF001 - documented
    level = 1
    while frame is not None:
        if not _is_plumbing(frame.f_code.co_filename):
            return level
        frame = frame.f_back
        level += 1
    return fallback


def warn_at_user_call(
    message: str, category: type[Warning], fallback_stacklevel: int
) -> None:
    """Emits ``message``, attributed to the user's own line.

    The single way every user-facing warning in Deepwave reaches
    ``warnings.warn``, so that the filename and line number Python reports are
    the user's call site whichever form they used -- the propagator function,
    the ``Module``, or a ``checkpoint_segments=``/``run_segments`` chain --
    rather than a line of Deepwave's own plumbing. See
    :func:`stacklevel_outside_deepwave` for why a fixed level cannot do this.

    Args:
        message: The formatted warning text.
        category: The warning category.
        fallback_stacklevel: The level to use if every frame above this one
            belongs to Deepwave or PyTorch, counted from the caller of this
            function (``1`` is the caller's own frame).

    """
    sink = _warning_sink.get()
    if sink is not None:
        sink.append(message)
        return
    warnings.warn(
        message,
        category,
        stacklevel=stacklevel_outside_deepwave(fallback_stacklevel + 1),
    )


class DeepwaveFutureWarning(FutureWarning):
    """Warning category for a deprecated Deepwave spelling.

    Emitted once per call per deprecated name, by the name that is going
    away, naming both the replacement and the release that removes it (see
    ``DEPRECATION_REMOVAL_VERSION``). Nothing that emits one changes
    what a call computes: the deprecated spelling keeps doing what it always
    did until it is removed.

    Silence it with one line::

        warnings.filterwarnings(
            "ignore", category=deepwave.DeepwaveFutureWarning
        )

    It subclasses ``FutureWarning``, so any filter that already targets
    ``FutureWarning`` also covers it. ``FutureWarning`` rather than
    ``DeprecationWarning`` because the call that has to change is the user's
    own: Python's default filters show ``FutureWarning`` from any module,
    while ``DeprecationWarning`` is hidden outside ``__main__``, which is
    where a library's deprecation is least likely to be read.
    """

    __module__ = "deepwave"


class DeepwaveWarning(UserWarning):
    """Base class for every warning Deepwave raises about a call's behaviour.

    One subclass per *cause*, never one per propagator or per message: the
    category is what a user filters on, so it has to name the thing they
    would decide about. Every warning raised under one of these categories
    names its class in its own text, so the one-line recipe for silencing
    it is in the message itself. Deepwave also still emits some plain
    ``UserWarning`` messages that predate this hierarchy and are not
    classified under it (the grid-cells-per-wavelength advice, the
    ``pml_freq``-above-Nyquist warning, and the storage/auxiliary-state
    notices); those name no category, and are filtered as ``UserWarning``.

    Filtering the base covers this class and its subclasses::

        warnings.filterwarnings("ignore", category=deepwave.DeepwaveWarning)

    It subclasses ``UserWarning``, so a filter that already targets
    ``UserWarning`` also covers it. :class:`DeepwaveFutureWarning` is
    deliberately outside this hierarchy: a deprecated spelling is a message
    about the code that was written, not about the physics it asks for.
    """

    __module__ = "deepwave"


class ZeroWidthFaceWarning(DeepwaveWarning):
    """A ``"pml"`` face of zero width, which reflects instead of absorbing.

    A ``"pml"`` face whose ``pml_width`` is zero absorbs nothing. It is a
    reflecting boundary whose exact condition depends on the propagator (see
    :doc:`/boundaries`), and it is reached identically whether ``boundary``
    was omitted, given as the single string ``"pml"``, or given per face --
    those three spellings are the same call, so they warn the same way.

    Silence it with one line::

        warnings.filterwarnings(
            "ignore", category=deepwave.ZeroWidthFaceWarning
        )

    The warning is about a silent choice, not an error: the call runs, and
    naming the condition you meant with ``boundary=`` is the fix.
    """

    __module__ = "deepwave"


class MigrationWarning(DeepwaveWarning):
    """A call whose numbers changed from what an older Deepwave computed.

    Nothing about the call is wrong: the same arguments now mean something
    better than they used to. The warning exists so that a difference
    against an older run is explained rather than discovered. Where one line
    restores the previous behaviour exactly, the message names it; where no
    argument can, the message says so and names what the difference is
    confined to instead.

    Silence it with one line::

        warnings.filterwarnings(
            "ignore", category=deepwave.MigrationWarning
        )
    """

    __module__ = "deepwave"


class PmlFreqInferenceWarning(MigrationWarning):
    """``pml_freq`` was inferred, and differs from the old fixed default.

    Deepwave 0.0.27 used a fixed ``pml_freq`` of 25.0 whenever the argument
    was omitted; it is now inferred from the source amplitudes. An inference
    of exactly 25.0 computes what that release computed and says nothing; this
    is the narrower category for every other inferred value, each of which
    gives a different absorbing boundary from that release. It is a migration
    aid, and is removed in Deepwave 0.4.0.

    Silence it with one line::

        warnings.filterwarnings(
            "ignore", category=deepwave.PmlFreqInferenceWarning
        )

    Passing ``pml_freq`` explicitly silences it too, and is the better fix:
    it makes the call say what it means.
    """

    __module__ = "deepwave"


def warn_deprecated_name(
    subject: str,
    replacement: str,
    *,
    stacklevel: int,
    detail: str = "",
) -> None:
    """Announces one deprecated spelling, naming its replacement and 0.4.0.

    The single emitter of :class:`DeepwaveFutureWarning` in Deepwave, so that
    every deprecated name says the same three things in the same words: what
    is going away, what to write instead, and the release that removes it
    (``DEPRECATION_REMOVAL_VERSION``). A deprecation cannot be added that
    forgets one of them, and the one-line recipe for silencing every
    deprecation at once is in the text the user is reading.

    Args:
        subject: The deprecated spelling, as the user wrote it. A propagator
            that names itself in its messages prefixes it (``"scalar_born:
            receiver_locations"``).
        replacement: What to write instead.
        stacklevel: The stack level to fall back on if every frame above this
            one belongs to Deepwave or PyTorch; ``1`` is the caller's own
            frame, ``2`` its caller, and so on. The level actually used is
            :func:`stacklevel_outside_deepwave`'s, so that the deprecation is
            reported at the user's own line whichever form they called
            through.
        detail: An optional sentence inserted before the filtering recipe,
            for a deprecation whose replacement needs more than its name (for
            example that a key was remapped rather than rejected).

    """
    message = (
        f"{subject} is deprecated and will be removed in Deepwave "
        f"{DEPRECATION_REMOVAL_VERSION}; use {replacement}."
    )
    if detail:
        message = f"{message} {detail}"
    warn_at_user_call(
        f"{message} Silence this with warnings.filterwarnings('ignore', "
        "category=deepwave.DeepwaveFutureWarning).",
        DeepwaveFutureWarning,
        # ``stacklevel`` is counted from this function's caller and
        # ``warn_at_user_call``'s fallback from its own, one frame lower.
        stacklevel + 1,
    )


class DeepwaveError(Exception):
    """Base class for every exception Deepwave raises deliberately.

    Catching this catches Deepwave's own validation and capability errors
    without also catching unrelated exceptions from PyTorch or the standard
    library. Deepwave never raises ``DeepwaveError`` itself: every raise site
    uses one of the subclasses below, each of which also inherits the builtin
    exception type that site raised historically, so ``except ValueError``
    (or ``TypeError``, ``RuntimeError``, ``NotImplementedError``,
    ``FloatingPointError``) keeps working unchanged.
    """

    __module__ = "deepwave"


class DeepwaveValueError(DeepwaveError, ValueError):
    """A Deepwave error that is also a ``ValueError``.

    Raised for an argument whose type is right but whose value is not (an
    out-of-range number, an unrecognised option string, a mismatched shape).
    """

    __module__ = "deepwave"


class DeepwaveTypeError(DeepwaveError, TypeError):
    """A Deepwave error that is also a ``TypeError``.

    Raised for an argument of the wrong type (a float where an int is
    required, a non-Tensor model).
    """

    __module__ = "deepwave"


class DeepwaveRuntimeError(DeepwaveError, RuntimeError):
    """A Deepwave error that is also a ``RuntimeError``.

    Raised for a request that is well-formed but that this build, backend, or
    combination of options cannot serve.
    """

    __module__ = "deepwave"


class DeepwaveNotImplementedError(DeepwaveError, NotImplementedError):
    """A Deepwave error that is also a ``NotImplementedError``.

    Raised for a combination of options that is intended to work but has not
    been implemented yet.
    """

    __module__ = "deepwave"


class DeepwaveOSError(DeepwaveError, OSError):
    """A Deepwave error that is also an ``OSError``.

    Raised when the operating system refuses a filesystem operation Deepwave
    performs on the caller's behalf, such as creating the scratch directory
    that ``storage_mode="disk"`` writes its wavefield history into. The
    message names the argument that chose the location (``storage_path``) so
    that the fix is visible from the traceback alone. ``except OSError``
    around a Deepwave call keeps catching it.
    """

    __module__ = "deepwave"


class DeepwaveFloatingPointError(DeepwaveError, FloatingPointError):
    """A Deepwave error that is also a ``FloatingPointError``.

    Raised when a tensor contains non-finite values, which usually means the
    simulation diverged; see :func:`deepwave.check_finite`.
    """

    __module__ = "deepwave"
