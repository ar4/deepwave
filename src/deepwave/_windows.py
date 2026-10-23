"""Shared window/taper alias set and weight formula.

:mod:`deepwave.wavelets` (:func:`~deepwave.wavelets.tone_burst`) and
:mod:`deepwave.transducers` (:func:`~deepwave.transducers.window_weights` and the
element builders) both let the caller choose a named window/taper shape.
This module is the single place that knows the recognised names and computes
the corresponding shape, so both call sites agree on what each name means.

The window shape is parameterised by a symmetric coordinate ``x`` that runs
from ``-1`` at one edge to ``0`` at the centre to ``1`` at the other edge
(callers map their own coordinate, e.g. sample index or normalised time,
onto this range).
"""

import math

import torch

from deepwave._exceptions import DeepwaveValueError

#: Recognised window names, mapped to a canonical name. ``"rect"``,
#: ``"boxcar"``, ``"uniform"`` and ``"none"`` are synonyms for an untapered
#: (all-ones) window; ``"hann"`` and ``"hanning"`` are synonyms for the Hann
#: window.
ALIASES: dict[str, str] = {
    "rect": "rect",
    "boxcar": "rect",
    "uniform": "rect",
    "none": "rect",
    "hann": "hann",
    "hanning": "hann",
    "hamming": "hamming",
    "blackman": "blackman",
}

#: The default window name used where a caller does not specify one.
DEFAULT: str = "hann"


def canonicalize(kind: str) -> str:
    """Validate ``kind`` and return its canonical window name.

    Args:
        kind: A window name or alias (see :data:`ALIASES`).

    Returns:
        The canonical name (one of ``"rect"``, ``"hann"``, ``"hamming"``,
        ``"blackman"``).

    Raises:
        ValueError: If ``kind`` is not a recognised name.

    """
    canonical = ALIASES.get(kind)
    if canonical is None:
        raise DeepwaveValueError(
            f"Unknown window {kind!r}; expected one of {sorted(set(ALIASES))}.",
        )
    return canonical


def weights(x: torch.Tensor, kind: str) -> torch.Tensor:
    """Window weight at symmetric coordinate(s) ``x``.

    Args:
        x: Coordinate(s) in ``[-1, 1]``, ``0`` at the window centre and
            ``+-1`` at its edges.
        kind: A window name or alias (see :data:`ALIASES`); validated via
            :func:`canonicalize`.

    Returns:
        The window weight at each ``x``, the same shape as ``x``.

    """
    canonical = canonicalize(kind)
    if canonical == "rect":
        return torch.ones_like(x)
    if canonical == "hann":
        return 0.5 * (1 + torch.cos(math.pi * x))
    if canonical == "hamming":
        return 0.54 + 0.46 * torch.cos(math.pi * x)
    return 0.42 + 0.5 * torch.cos(math.pi * x) + 0.08 * torch.cos(2 * math.pi * x)
