"""Misfit (loss) functions for full-waveform inversion (FWI).

Stability: provisional (see :doc:`/stability`) -- this module is expected to
gain misfit functions, so its shape may still be corrected.

Provides the conventional squared-error data misfit, :func:`l2`, and two
alternatives that are less prone to cycle-skipping: :func:`envelope` (the
mean squared error between Hilbert envelopes) and
:func:`normalised_cross_correlation` (one minus the per-trace normalised
cross-correlation). All three take ``pred`` and ``target`` tensors of
identical shape ``[..., nt]`` (e.g. ``[shot, receiver, time]``), are fully
differentiable, and share one *shape* contract, so that swapping one for
another is a one-line change. Their *scales* differ, though, so a swap
usually wants the step length adjusted; see below.

That contract is: the time axis is always reduced away first, giving one
value per trace, and ``reduction`` then says what to do with that per-trace
tensor. ``"mean"`` (the default) returns its mean and ``"sum"`` its sum,
both scalars; ``"none"`` returns the per-trace tensor itself, of shape
``pred.shape[:-1]`` (e.g. ``[shot, receiver]``).

Both inputs must contain at least one time sample and at least one trace;
empty time axes and empty leading trace dimensions are rejected consistently
for all three functions.

**Comparison with** :func:`torch.nn.functional.mse_loss`, which is what
:func:`l2` is most often swapped for:

* ``l2(pred, target, reduction="sum")`` is **exactly**
  ``mse_loss(pred, target, reduction="sum")``: both sum the squared
  differences over every element.
* ``l2(pred, target, reduction="mean")`` is **not**
  ``mse_loss(pred, target, reduction="mean")``. It is larger by a factor of
  ``nt``, because :func:`l2` sums over time and then averages over traces,
  whereas ``mse_loss`` averages over every element, time included. Replacing
  ``mse_loss`` with :func:`l2` therefore multiplies the loss *and its
  gradient* by ``nt`` -- often 1e3 to 1e4 -- with nothing raising: divide the
  learning rate by ``nt``, or use ``reduction="sum"`` on both sides, to keep
  the same steps.
* ``reduction="none"`` differs in shape as well: ``mse_loss`` returns an
  element-wise tensor with the time axis still present, whereas these
  functions never leave the time axis, a misfit being inherently a per-trace
  quantity.

**The three misfits do not share a scale either.** :func:`l2` *sums* the
squared differences over time while :func:`envelope` *averages* them, so for
the same data :func:`l2` is of order ``nt`` times the larger, and
:func:`normalised_cross_correlation` is bounded in ``[0, 2]`` per trace
whatever the amplitudes. The one-line swap is a swap of *shapes and
semantics*, not of magnitudes: expect to retune the step length (or
normalise the misfit) after changing one for another.
"""

from typing import get_args

import torch

from deepwave._exceptions import DeepwaveTypeError as _DeepwaveTypeError
from deepwave._exceptions import DeepwaveValueError as _DeepwaveValueError
from deepwave.types import Reduction as _Reduction

__all__ = [
    "envelope",
    "l2",
    "normalised_cross_correlation",
]

# Single-sourced from the ``Reduction`` Literal so the runtime check and the
# static type can never disagree.
_REDUCTIONS: tuple[str, ...] = get_args(_Reduction)


def _validate_pred_target(
    pred: torch.Tensor, target: torch.Tensor, reduction: _Reduction
) -> None:
    """Shared, loud validation for every misfit function."""
    if reduction not in _REDUCTIONS:
        raise _DeepwaveValueError(
            f"reduction must be one of {_REDUCTIONS}, got {reduction!r}."
        )
    if not torch.is_floating_point(pred):
        raise _DeepwaveTypeError(
            f"pred must be a floating-point tensor, got dtype {pred.dtype}."
        )
    if not torch.is_floating_point(target):
        raise _DeepwaveTypeError(
            f"target must be a floating-point tensor, got dtype {target.dtype}."
        )
    if pred.dtype != target.dtype:
        raise _DeepwaveTypeError(
            "pred and target must have the same dtype, got "
            f"{pred.dtype} and {target.dtype}.",
        )
    if pred.shape != target.shape:
        raise _DeepwaveValueError(
            "pred and target must have the same shape, got "
            f"{list(pred.shape)} and {list(target.shape)}.",
        )
    if pred.dim() == 0:
        raise _DeepwaveValueError(
            "pred and target must have at least one dimension (the time "
            "axis, nt), got a 0-dimensional tensor.",
        )
    if pred.shape[-1] == 0:
        raise _DeepwaveValueError(
            "pred and target must contain at least one time sample; got an "
            "empty time axis (nt=0).",
        )
    if any(size == 0 for size in pred.shape[:-1]):
        raise _DeepwaveValueError(
            "pred and target must contain at least one trace; got an empty "
            "leading trace dimension.",
        )


def _reduce_per_trace(per_trace: torch.Tensor, reduction: _Reduction) -> torch.Tensor:
    """Apply ``reduction`` to an already time-reduced, per-trace tensor."""
    if reduction == "none":
        return per_trace
    if reduction == "sum":
        return per_trace.sum()
    return per_trace.mean()


def l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: _Reduction = "mean",
) -> torch.Tensor:
    r"""Squared-error (least-squares) misfit between ``pred`` and ``target``.

    The conventional FWI data misfit. Per trace, with :math:`p[n]` = ``pred``,
    :math:`t[n]` = ``target``, and :math:`n = 0, \dots, n_t - 1`,

    .. math::

        L = \sum_{n} \bigl(p[n] - t[n]\bigr)^2,

    which is then reduced across the leading (non-time) dimensions according
    to ``reduction``.

    The misfit is usually written in the FWI literature with a leading factor
    of :math:`1/2` (which cancels the 2 produced by differentiating the
    square); omitting it, as here, multiplies the misfit and its gradient by
    the same constant 2, moving no minimum and only rescaling the step length
    an optimiser must take.

    This is the misfit that is most prone to cycle-skipping: it compares the
    oscillatory waveforms sample by sample, so a predicted arrival more than
    half a period from the observed one is pulled towards the wrong cycle.
    :func:`envelope` and :func:`normalised_cross_correlation` are drop-in
    alternatives with the same shape contract, though not with the same
    scale: this function sums over time where :func:`envelope` averages, so
    a swap usually wants the step length adjusted. The module docstring gives
    the exact relationship, and the one to
    :func:`torch.nn.functional.mse_loss`, whose ``"mean"`` reduction differs
    from this one by a factor of ``nt``.

    Args:
        pred: Predicted data, a real floating-point tensor of shape
            ``[..., nt]`` (e.g. ``[shot, receiver, time]``).
        target: Observed/target data. Must have the same shape and dtype as
            ``pred``.
        reduction: One of ``"mean"`` (default), ``"sum"``, or ``"none"``. With
            ``"none"``, the time axis is still reduced away (it is not
            meaningful to leave it, since the misfit is inherently a
            per-trace quantity), returning the per-trace sum of squared
            differences with shape ``pred.shape[:-1]``; ``"mean"``/``"sum"``
            further reduce that per-trace tensor to a scalar.

    Returns:
        A scalar tensor (``"mean"``/``"sum"``) or a tensor of shape
        ``pred.shape[:-1]`` (``"none"``).

    Raises:
        TypeError: If ``pred`` or ``target`` is not a floating-point tensor,
            or their dtypes differ.
        ValueError: If ``pred`` and ``target`` have different shapes, either
            has zero dimensions, or ``reduction`` is not recognised.

    """
    _validate_pred_target(pred, target, reduction)
    per_trace = (pred - target).square().sum(dim=-1)
    return _reduce_per_trace(per_trace, reduction)


def envelope(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: _Reduction = "mean",
) -> torch.Tensor:
    r"""Mean squared error between the Hilbert envelopes of ``pred`` and ``target``.

    The Hilbert envelope (analytic-signal magnitude) of a real trace
    :math:`x[n]`, :math:`n = 0, \dots, n_t - 1`, is computed from its discrete
    Fourier transform :math:`X = \mathrm{FFT}(x)` by discarding the
    negative-frequency half of the spectrum and doubling the positive
    frequencies, following the standard FFT-based construction of the discrete
    analytic signal (Marple, 1999):

    .. math::

        A[k] =
        \begin{cases}
        1, & k = 0 \\
        2, & 1 \le k < \lceil n_t / 2 \rceil \\
        1, & n_t \text{ even and } k = n_t / 2 \\
        0, & \text{otherwise,}
        \end{cases}
        \qquad
        z = \mathrm{IFFT}(A \odot X), \qquad e[n] = |z[n]|,

    where :math:`z` is the discrete analytic signal and :math:`e` its
    envelope (the :math:`k = n_t / 2` "Nyquist" case only arises for even
    :math:`n_t`; it is self-conjugate and so kept at weight 1, not doubled).
    Writing :math:`e_{\text{pred}}` and :math:`e_{\text{target}}` for the
    envelopes of ``pred`` and ``target``, the per-trace value is their mean
    squared difference over time,

    .. math::

        L = \frac{1}{n_t} \sum_{n} \bigl(e_{\text{pred}}[n] -
        e_{\text{target}}[n]\bigr)^2,

    which is then reduced across the leading (non-time) dimensions according
    to ``reduction``. Because it responds to envelope (amplitude-of-arrival)
    differences rather than the raw, oscillatory waveform, this misfit is
    markedly less sensitive to the phase/traveltime errors that cause
    cycle-skipping in the raw squared-error misfit, and is a standard FWI
    objective for that reason (Bozdağ, Trampert & Tromp, 2011).

    Args:
        pred: Predicted data, a real floating-point tensor of shape
            ``[..., nt]`` (e.g. ``[shot, receiver, time]``).
        target: Observed/target data. Must have the same shape and dtype as
            ``pred``.
        reduction: One of ``"mean"`` (default), ``"sum"``, or ``"none"``. With
            ``"none"``, the time axis is still reduced away (it is not
            meaningful to leave it, since the misfit is inherently a
            per-trace quantity), returning the per-trace mean squared
            envelope difference with shape ``pred.shape[:-1]``; ``"mean"``/
            ``"sum"`` further reduce that per-trace tensor to a scalar.

    Returns:
        A scalar tensor (``"mean"``/``"sum"``) or a tensor of shape
        ``pred.shape[:-1]`` (``"none"``).

    Raises:
        TypeError: If ``pred`` or ``target`` is not a floating-point tensor,
            or their dtypes differ.
        ValueError: If ``pred`` and ``target`` have different shapes, either
            has zero dimensions, or ``reduction`` is not recognised.

    References:
        S. L. Marple, "Computing the discrete-time analytic signal via FFT,"
        IEEE Transactions on Signal Processing, vol. 47, no. 9, 1999.

        E. Bozdağ, J. Trampert, J. Tromp, "Misfit functions for full waveform
        inversion based on instantaneous phase and envelope measurements,"
        Geophysical Journal International, vol. 185, no. 2, 2011.

    """
    _validate_pred_target(pred, target, reduction)
    nt = pred.shape[-1]

    weight = torch.zeros(nt, dtype=pred.dtype, device=pred.device)
    half = (nt + 1) // 2
    weight[0] = 1
    weight[1:half] = 2
    if nt % 2 == 0:
        weight[nt // 2] = 1

    def envelope_of(x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft(x, dim=-1)
        analytic = torch.fft.ifft(spectrum * weight.to(spectrum.dtype), dim=-1)
        result: torch.Tensor = analytic.abs()
        return result

    per_trace = (envelope_of(pred) - envelope_of(target)).square().mean(dim=-1)
    return _reduce_per_trace(per_trace, reduction)


def normalised_cross_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: _Reduction = "mean",
) -> torch.Tensor:
    r"""One minus the per-trace normalised cross-correlation.

    Per trace, with :math:`p[n]` = ``pred``, :math:`t[n]` = ``target``,
    :math:`n = 0, \dots, n_t - 1`, and :math:`\lVert \cdot \rVert_2` the L2
    norm over the time axis,

    .. math::

        \mathrm{NCC} =
        \begin{cases}
        \dfrac{\sum_n p[n]\, t[n]}{\lVert p \rVert_2 \, \lVert t \rVert_2},
        & \lVert p \rVert_2 \, \lVert t \rVert_2 > 0, \\[6pt]
        0, & \lVert p \rVert_2 \, \lVert t \rVert_2 = 0,
        \end{cases}
        \qquad L = 1 - \mathrm{NCC},

    No mean is subtracted from either trace before normalising, so
    :math:`\mathrm{NCC}` is the cosine similarity of the raw traces, not the
    statistical (mean centred) correlation coefficient: the misfit stays
    sensitive to any DC offset, which is physically meaningful for wave data,
    since a non-zero mean is itself a mismatch worth penalising.

    The :math:`\lVert p \rVert_2 \, \lVert t \rVert_2 = 0` branch applies
    only when at least one of ``pred``/``target`` is an *exactly* all-zero
    trace (its L2 norm is exactly ``0.0``, not merely small): then
    :math:`\mathrm{NCC} = 0` and :math:`L = 1`, a finite, defined result
    (rather than the ``0 / 0`` that an unguarded division would give) that
    reads as "no correlation" for a trace with no signal. It is a branch on
    the value 0.0, not a small-denominator threshold, so it does not fire at
    small-but-nonzero amplitudes: traces many orders of magnitude below unit
    scale are compared exactly like unit-scale data (see
    :doc:`/design/misfits` for the internal rescaling that makes that so).

    :math:`L` is exactly 0 whenever ``pred`` equals ``target`` scaled by any
    positive constant on that trace (a pure amplitude error), and otherwise
    ranges over :math:`[0, 2]`. That amplitude-scale invariance, with
    undiminished sensitivity to phase/traveltime shifts, is why
    normalised-correlation ("global correlation norm") objectives are used as
    an FWI misfit robust to the amplitude mismatches that a squared-error
    misfit penalises but that need not indicate a poor model (Choi &
    Alkhalifah, 2012). The per-trace :math:`L` is then reduced across the
    leading (non-time) dimensions according to ``reduction``.

    Args:
        pred: Predicted data, a real floating-point tensor of shape
            ``[..., nt]`` (e.g. ``[shot, receiver, time]``).
        target: Observed/target data. Must have the same shape and dtype as
            ``pred``.
        reduction: One of ``"mean"`` (default), ``"sum"``, or ``"none"``. With
            ``"none"``, the time axis is still reduced away (it is not
            meaningful to leave it, since the misfit is inherently a
            per-trace quantity), returning the per-trace value with shape
            ``pred.shape[:-1]``; ``"mean"``/``"sum"`` further reduce that
            per-trace tensor to a scalar.

    Returns:
        A scalar tensor (``"mean"``/``"sum"``) or a tensor of shape
        ``pred.shape[:-1]`` (``"none"``).

    Raises:
        TypeError: If ``pred`` or ``target`` is not a floating-point tensor,
            or their dtypes differ.
        ValueError: If ``pred`` and ``target`` have different shapes, either
            has zero dimensions, or ``reduction`` is not recognised.

    References:
        Y. Choi, T. Alkhalifah, "Application of multi-source waveform
        inversion to marine streamer data using the global correlation
        norm," Geophysical Prospecting, vol. 60, no. 4, 2012.

    """
    _validate_pred_target(pred, target, reduction)

    # Rescale each trace, independently, by its own max absolute value before
    # computing the dot product and norms. NCC(pred, target) is exactly
    # scale-invariant in each argument separately (NCC(a*p, b*t) = NCC(p, t)
    # for any a, b > 0), so this changes nothing mathematically -- but it
    # keeps every intermediate value O(1) regardless of the traces' raw
    # amplitude, avoiding the squared-magnitude underflow that computing the
    # L2 norm directly from raw samples would suffer at small-but-nonzero
    # amplitudes (ordinary seismic data, e.g. particle-velocity or
    # strain-rate traces, is routinely far below unit scale). The scale
    # factors are detached: they are purely a numerical-conditioning device,
    # not part of the mathematical NCC/L formula, so no gradient should flow
    # through them.
    pred_scale = pred.detach().abs().amax(dim=-1, keepdim=True)
    target_scale = target.detach().abs().amax(dim=-1, keepdim=True)
    safe = (pred_scale.squeeze(-1) > 0) & (target_scale.squeeze(-1) > 0)
    pred_n = pred / torch.where(pred_scale > 0, pred_scale, torch.ones_like(pred_scale))
    target_n = target / torch.where(
        target_scale > 0, target_scale, torch.ones_like(target_scale)
    )

    dot = (pred_n * target_n).sum(dim=-1)
    denom = pred_n.norm(p=2, dim=-1) * target_n.norm(p=2, dim=-1)
    # A genuine 0.0/nonzero branch, not a small-denominator epsilon floor: it
    # only fires when a trace's max absolute value (equivalently, its L2
    # norm) is *exactly* zero, so it does not bias the result at
    # small-but-nonzero amplitudes (see the docstring). ``torch.where``
    # evaluates both branches, so ``safe_denom`` substitutes a placeholder 1
    # wherever ``denom`` is 0 purely to avoid a 0/0 division; that branch's
    # result is then discarded by the outer ``torch.where``, and its
    # gradient contribution vanishes because ``dot`` is also 0 wherever
    # ``denom`` is 0 (a zero-scale trace is exactly the zero vector, so its
    # dot product with anything is 0).
    safe_denom = torch.where(safe, denom, torch.ones_like(denom))
    ncc = torch.where(safe, dot / safe_denom, torch.zeros_like(denom))
    per_trace = 1 - ncc
    return _reduce_per_trace(per_trace, reduction)
