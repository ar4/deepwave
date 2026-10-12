"""Common source wavelets.

In addition to the seismic :func:`ricker` wavelet, this module provides two
excitations that are standard in ultrasound and non-destructive testing (NDT):
a windowed sinusoid :func:`tone_burst` and a Gaussian-modulated sinusoid
:func:`gaussian_pulse`. As with :func:`ricker`, time is measured in seconds and
frequency in hertz; if you work in another consistent unit system (for example
microseconds and megahertz), simply keep ``dt`` and ``freq`` in those units --
only their product matters. :func:`ormsby` is a fourth, four-corner-frequency
wavelet, also standard in seismic processing.
"""

import math
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch

from deepwave import _windows
from deepwave._exceptions import (
    DeepwaveTypeError as _DeepwaveTypeError,
)
from deepwave._exceptions import (
    DeepwaveValueError as _DeepwaveValueError,
)
from deepwave._exceptions import (
    warn_at_user_call as _warn_at_user_call,
)
from deepwave._exceptions import (
    warn_deprecated_name as _warn_deprecated_name,
)

__all__ = [
    "gaussian_pulse",
    "ormsby",
    "ricker",
    "tone_burst",
]


class _Required:
    """The type of :data:`_REQUIRED`, giving the sentinel a readable repr.

    ``inspect.signature``, ``help()`` and Sphinx all render a parameter's
    default with ``repr``, so a bare ``object()`` would publish
    ``nt: int = <object object at 0x...>`` -- both meaningless to a reader
    and, because the address changes per process, different on every
    regeneration of the extracted API.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Renders as ``<required>`` wherever the default is introspected."""
        return "<required>"


#: Sentinel default for :func:`ricker`'s ``nt`` and ``dt``. They are required
#: arguments, so omitting one is a ``TypeError``; they cannot simply be
#: declared without a default, because ``length`` (a deprecated keyword-only
#: alias for ``nt``) may supply ``nt`` instead. Typed ``Any`` so that ``nt``
#: and ``dt`` still annotate as ``int``/``float`` rather than as optional.
_REQUIRED: Any = _Required()

#: The type of every wavelet builder's keyword-only ``shape`` argument.
_Shape = Optional[Union[int, Sequence[int]]]


def _square_out_of_dtype_range(value: float, dtype: torch.dtype) -> bool:
    """Whether squaring ``value`` is an unsafe intermediate in ``dtype``."""
    try:
        info = torch.finfo(dtype)
    except TypeError:
        return False
    magnitude = abs(value)
    if magnitude == 0:
        return False
    return magnitude > math.sqrt(info.max) or magnitude < math.sqrt(info.tiny)


def _dimensionless_fallback_needed(
    freqs: Sequence[float],
    dt: float,
    peak_time: float,
    peak_was_defaulted: bool,
    nt: int,
    dtype: Optional[torch.dtype],
    *,
    extra_squared_scale: float = 1.0,
    phase_factor: float = 1.0,
) -> bool:
    """Select a scaled calculation when physical-unit intermediates are unsafe.

    Very large finite dimensionless phases still inherit float64 phase-reduction
    limits; this fallback does not promise arbitrary-precision oscillations.
    """
    work_dtype = dtype if dtype is not None else torch.get_default_dtype()
    try:
        info = torch.finfo(work_dtype)
    except TypeError:
        return False

    if peak_was_defaulted:
        time_extent = abs(dt) * max(nt / 2, 0.5)
    else:
        time_extent = abs(dt) * max(nt - 1, 0) + abs(peak_time)
    unsafe = _square_out_of_dtype_range(time_extent, work_dtype)
    unsafe |= any(_square_out_of_dtype_range(f, work_dtype) for f in freqs)
    unsafe |= _square_out_of_dtype_range(extra_squared_scale, work_dtype)
    if phase_factor and any(abs(f) > info.max / phase_factor for f in freqs):
        unsafe = True
    if not unsafe:
        return False

    for freq in freqs:
        q = freq * dt
        if not math.isfinite(q):
            return False
        if peak_was_defaulted:
            largest_phase = abs(q) * max(nt / 2, 0.5)
        else:
            p = freq * peak_time
            if not math.isfinite(p):
                return False
            largest_phase = abs(q) * max(nt - 1, 0) + abs(p)
        if not math.isfinite(largest_phase):
            return False
    return True


def _dimensionless_time(
    freq: float,
    nt: int,
    dt: float,
    peak_time: float,
    peak_was_defaulted: bool,
) -> torch.Tensor:
    """Return ``freq * (sample_time - peak_time)`` in float64 sample units."""
    samples = torch.arange(float(nt), dtype=torch.float64)
    q = freq * dt
    if peak_was_defaulted:
        return (samples - nt / 2) * q
    physical_offset = samples * dt - peak_time
    min_normal = torch.finfo(torch.float64).tiny
    offsets_resolved = abs(dt) >= min_normal and (
        peak_time == 0 or abs(peak_time) >= min_normal
    )
    if offsets_resolved and torch.isfinite(physical_offset).all():
        return physical_offset * freq
    peak_sample = peak_time / dt
    if math.isfinite(peak_sample):
        return (samples - peak_sample) * q
    return samples * q - freq * peak_time


def _cast_wavelet_dtype(y: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    return y.to(dtype if dtype is not None else torch.get_default_dtype())


def _expand_to_shape(y: torch.Tensor, shape: _Shape) -> torch.Tensor:
    """Add leading dimensions to a ``[nt]`` wavelet, without copying.

    Args:
        y: The one-dimensional wavelet, of length ``nt``.
        shape: The leading dimensions to add, as described in the wavelet
            builders' ``shape`` argument. ``None`` returns ``y`` unchanged.

    Returns:
        ``y`` itself if ``shape`` is ``None``, otherwise a
        :meth:`torch.Tensor.expand` view of shape ``[*shape, nt]``.

    Raises:
        DeepwaveTypeError: If ``shape`` is not ``None``, an int, or a
            sequence of ints.
        DeepwaveValueError: If any entry of ``shape`` is not positive.

    """
    if shape is None:
        return y
    if isinstance(shape, bool):
        raise _DeepwaveTypeError(
            "shape must be None, a positive int, or a sequence of positive "
            f"ints, got {shape!r}."
        )
    if isinstance(shape, int):
        dims: tuple[int, ...] = (shape,)
    else:
        try:
            dims = tuple(shape)
        except TypeError:
            raise _DeepwaveTypeError(
                "shape must be None, a positive int, or a sequence of "
                f"positive ints, got {shape!r}."
            ) from None
    for dim in dims:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise _DeepwaveTypeError(
                f"Every entry of shape must be an int, got shape={dims!r}."
            )
        if dim <= 0:
            raise _DeepwaveValueError(
                f"Every entry of shape must be positive, got shape={dims!r}."
            )
    return y.expand(*dims, y.shape[-1])


def ricker(
    freq: float,
    nt: int = _REQUIRED,
    dt: float = _REQUIRED,
    peak_time: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    *,
    length: Optional[int] = None,
    shape: _Shape = None,
) -> torch.Tensor:
    """Return a Ricker wavelet with the specified central frequency.

    Args:
        freq: The central frequency (Hz, if ``dt`` is in seconds). Must be
            positive.
        nt: The number of time samples. Required: omitting it (and its
            deprecated ``length`` alias), or passing ``None``, raises
            ``TypeError``.
        dt: The time sample spacing (seconds). Required: omitting it, or
            passing ``None``, raises ``TypeError``.
        peak_time: The time (in secs) of the peak amplitude. Optional; if not
            provided the wavelet is auto-centred in the record
            (``peak_time = nt * dt / 2``), the same record-centring
            convention as :func:`tone_burst`, :func:`gaussian_pulse` and
            :func:`ormsby` (which also default to
            ``peak_time = nt * dt / 2``), so the (symmetric) wavelet is not
            truncated. Passing an explicit value is unchanged.
        dtype: The PyTorch datatype to use. Optional, defaults to PyTorch's
            default (float32).
        length: Deprecated alias for ``nt``, retained for backward
            compatibility. **Removed in Deepwave 0.4.0.** Providing both
            raises ``ValueError``. Keyword-only, so a positional call such as
            ``ricker(freq, 100, dt)`` is unaffected by this deprecation.
        shape: Keyword-only. The leading dimensions of the returned tensor,
            with ``nt`` always last. ``None`` (the default) returns a
            one-dimensional ``[nt]`` tensor. ``(n_shots, n_sources)`` returns
            ``[n_shots, n_sources, nt]``, ready to pass straight to a
            propagator's ``source_amplitudes``. An ``int`` ``n`` is shorthand
            for ``(n,)`` and returns ``[n, nt]``; an empty sequence ``()``
            returns ``[nt]``, as ``None`` does. Every entry must be a
            positive integer. The leading dimensions are added with
            :meth:`torch.Tensor.expand`, so no data is copied and the result
            is a **view**: every shot and source shares one copy of the
            wavelet, and each added dimension has stride zero. Call
            ``.contiguous()`` or ``.clone()`` on it before modifying it in
            place. Deepwave's propagators do not modify their inputs, so
            passing the view to one directly is safe.

    Returns:
        A PyTorch tensor representing the Ricker wavelet, of shape ``[nt]``,
        or ``[*shape, nt]`` if ``shape`` was given.

    Raises:
        TypeError: If ``nt`` (or its deprecated ``length`` alias) or ``dt``
            is not provided, or is provided as ``None``, or if ``shape`` is
            not ``None``, an int, or a sequence of ints.
        ValueError: If ``freq`` is not positive, if ``dt`` is zero, if
            both ``nt`` and ``length`` are provided, or if any entry of
            ``shape`` is not positive.

    """
    if not math.isfinite(freq) or freq <= 0:
        raise _DeepwaveValueError("freq must be positive and finite.")

    if length is not None:
        _warn_deprecated_name("length", "nt", stacklevel=2)
        if nt is not _REQUIRED and nt is not None:
            raise _DeepwaveValueError(
                "Pass only nt, not both nt and the deprecated length."
            )
        nt = length
    del length
    # ``None`` counts as "not given" as well as the sentinel: ``nt`` and
    # ``dt`` were declared ``Optional`` before they became required, so
    # threading an unset config or argparse default straight through is a
    # real call shape, and it must get the clear missing-argument error
    # rather than a raw ``TypeError`` from the arithmetic below.
    if nt is _REQUIRED or nt is None:
        raise _DeepwaveTypeError("ricker() missing required argument: 'nt'")
    if dt is _REQUIRED or dt is None:
        raise _DeepwaveTypeError("ricker() missing required argument: 'dt'")
    if not math.isfinite(dt) or dt == 0:
        raise _DeepwaveValueError("dt cannot be zero or nonfinite.")

    peak_was_defaulted = peak_time is None
    if peak_time is None:
        peak_time = nt * dt / 2
    if not math.isfinite(peak_time):
        raise _DeepwaveValueError("peak_time must be finite.")

    if _dimensionless_fallback_needed(
        (freq,),
        dt,
        peak_time,
        peak_was_defaulted,
        nt,
        dtype,
        extra_squared_scale=math.sqrt(2) * math.pi * freq,
    ):
        u = _dimensionless_time(freq, nt, dt, peak_time, peak_was_defaulted)
        exponent = (math.pi * u) ** 2
        decay = torch.exp(-exponent)
        ordinary_value = (1 - 2 * exponent) * decay
        # Very remote tails are mathematically zero. Avoid inf * 0 becoming
        # NaN after exponent underflow or a squared dimensionless coordinate.
        y = torch.where((decay == 0) | ~torch.isfinite(exponent), 0, ordinary_value)
        y = _cast_wavelet_dtype(y, dtype)
    else:
        t: torch.Tensor = torch.arange(float(nt), dtype=dtype) * dt - peak_time
        y = (1 - 2 * math.pi**2 * freq**2 * t**2) * torch.exp(
            -(math.pi**2) * freq**2 * t**2,
        )
        if dtype is not None:
            y = y.to(dtype)
    return _expand_to_shape(y, shape)


#: The largest end-of-record amplitude, relative to a tone burst's own unit
#: peak, that :func:`tone_burst` treats as rounding rather than truncation.
#: A burst that fits inside its record is exactly zero at both ends, because
#: the taper is applied with a hard window, so this only has to absorb
#: float32 noise.
_TRUNCATION_TOLERANCE = 0.01


def _warn_if_burst_is_truncated(
    y: torch.Tensor,
    freq: float,
    n_cycles: float,
    nt: int,
    dt: float,
) -> None:
    """Warns when the record is too short to hold the whole tone burst.

    A tone burst is zero outside its window, so a first or last sample that
    is not zero means the record starts or stops part-way through the
    burst's own oscillation. That is a step discontinuity, and its spectrum
    is broad: the caller asked for a narrow-band excitation and would get a
    broadband one, silently. Measuring the end samples rather than comparing
    times keeps the test exact for every taper and both signs of the
    truncation.

    Args:
        y: The finished one-dimensional wavelet, before any expansion.
        freq: The burst's centre frequency, for the message.
        n_cycles: The number of cycles in the burst, for the message.
        nt: The number of time samples.
        dt: The time sample spacing.

    """
    leading = float(y[0].abs())
    trailing = float(y[-1].abs())
    if max(leading, trailing) <= _TRUNCATION_TOLERANCE:
        return
    ends = []
    if leading > _TRUNCATION_TOLERANCE:
        ends.append(f"start (first sample {leading:.3g}, not 0)")
    if trailing > _TRUNCATION_TOLERANCE:
        ends.append(f"end (last sample {trailing:.3g}, not 0)")
    duration = n_cycles / freq
    # Both containment conditions with the default record-centred peak_time:
    # nt * dt >= duration puts the leading edge at or after t = 0, and
    # nt * dt >= duration + 2 * dt keeps the trailing edge on or before the
    # last sample.
    suggested_nt = math.ceil(duration / abs(dt)) + 2
    _warn_at_user_call(
        "tone_burst is truncated at the "
        + " and the ".join(ends)
        + f" of the record. The burst lasts n_cycles / freq = {duration:.6g}, "
        f"but the record is only nt * dt = {nt * dt:.6g} long, so it cannot "
        "hold the whole burst: the wavelet jumps from zero to its truncated "
        "value in one sample, which radiates broadband energy that the "
        "chosen freq and n_cycles were meant to exclude. Either lengthen the "
        f"record (nt >= {suggested_nt} at this dt) or shorten the burst "
        "(smaller n_cycles, or higher freq). If the truncation is "
        "deliberate, silence this with warnings.filterwarnings on the "
        "message.",
        UserWarning,
        3,
    )


def tone_burst(
    freq: float,
    n_cycles: float,
    *,
    nt: int,
    dt: float,
    peak_time: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    window: str = _windows.DEFAULT,
    shape: _Shape = None,
) -> torch.Tensor:
    r"""Return a windowed sinusoid ("tone burst").

    A tone burst is a sinusoid of frequency ``freq`` multiplied by a smooth
    envelope of finite duration. It is the standard excitation in ultrasound
    and non-destructive testing, where the number of cycles ``n_cycles`` sets
    the trade-off between temporal resolution (few cycles, broad band) and
    spectral purity (many cycles, narrow band). The envelope duration is
    ``n_cycles / freq`` seconds, centred on ``peak_time``.

    Concretely, with :math:`T = n_{cycles} / f` the burst duration and
    :math:`\tau = t - t_{peak}`, the wavelet is

    .. math::

        s(t) = w(\tau)\,\sin(2\pi f\,\tau), \qquad |\tau| \le T/2,

    and zero outside that window, where :math:`w` is the chosen taper (a Hann
    window :math:`w(\tau) = \tfrac12\,[1 + \cos(2\pi\tau/T)]` by default).

    **The record must be long enough to hold the burst.** With the default,
    record-centred ``peak_time`` that means ``nt * dt`` of at least
    ``n_cycles / freq + 2 * dt``. A shorter record cuts the burst off
    part-way through its oscillation, so the wavelet steps from zero to a
    non-zero value in a single sample; that step is broadband, which is
    exactly what choosing a ``freq`` and an ``n_cycles`` was meant to avoid,
    and it would otherwise pass unnoticed because nothing about it is an
    error. Deepwave therefore emits a ``UserWarning`` naming the end that
    was cut, the burst duration, the record length and an ``nt`` that would
    fit. Lengthen the record or shorten the burst; if the truncation is
    deliberate, filter the warning.

    Args:
        freq: The centre frequency of the sinusoid (Hz, if ``dt`` is in
            seconds).
        n_cycles: The number of cycles of the sinusoid spanned by the window
            (may be fractional). The window duration is ``n_cycles / freq``,
            in the same time units as ``dt``.
        nt: The number of time samples in the returned wavelet.
        dt: The time sample spacing (seconds).
        peak_time: The time (seconds) of the centre of the window. Optional;
            if not provided the burst is centred in the record
            (``peak_time = nt * dt / 2``), the same record-centring
            convention as :func:`ricker`, :func:`gaussian_pulse` and
            :func:`ormsby`, so that swapping one wavelet for another does not
            move the source in time. Deepwave previously defaulted instead to
            ``peak_time = n_cycles / freq / 2``, which centred the burst
            *window* rather than the record, placing the start of the burst
            at ``t = 0``; pass ``peak_time=n_cycles / freq / 2`` explicitly to
            reproduce that waveform.
        dtype: The PyTorch datatype to use. Optional, defaults to PyTorch's
            default (float32).
        window: Keyword-only. The envelope taper; see
            :func:`deepwave.transducers.window_weights` for the recognised
            names and their aliases (default ``"hann"``).
        shape: Keyword-only. The leading dimensions of the returned tensor,
            with ``nt`` always last. ``None`` (the default) returns a
            one-dimensional ``[nt]`` tensor. ``(n_shots, n_sources)`` returns
            ``[n_shots, n_sources, nt]``, ready to pass straight to a
            propagator's ``source_amplitudes``. An ``int`` ``n`` is shorthand
            for ``(n,)`` and returns ``[n, nt]``; an empty sequence ``()``
            returns ``[nt]``, as ``None`` does. Every entry must be a
            positive integer. The leading dimensions are added with
            :meth:`torch.Tensor.expand`, so no data is copied and the result
            is a **view**: every shot and source shares one copy of the
            wavelet, and each added dimension has stride zero. Call
            ``.contiguous()`` or ``.clone()`` on it before modifying it in
            place. Deepwave's propagators do not modify their inputs, so
            passing the view to one directly is safe.

    Returns:
        A PyTorch tensor of shape ``[nt]`` containing the tone burst, or of
        shape ``[*shape, nt]`` if ``shape`` was given.

    Raises:
        TypeError: If ``shape`` is not ``None``, an int, or a sequence of
            ints.
        ValueError: If ``dt`` is zero, if ``freq`` or ``n_cycles`` is not
            positive, if ``window`` is not a recognised name, or if any entry
            of ``shape`` is not positive.

    """
    if not math.isfinite(dt) or dt == 0:
        raise _DeepwaveValueError("dt cannot be zero or nonfinite.")
    if not math.isfinite(freq) or freq <= 0:
        raise _DeepwaveValueError("freq must be positive and finite.")
    if not math.isfinite(n_cycles) or n_cycles <= 0:
        raise _DeepwaveValueError("n_cycles must be positive and finite.")

    duration = n_cycles / freq
    peak_was_defaulted = peak_time is None
    if peak_time is None:
        peak_time = nt * dt / 2
    if not math.isfinite(peak_time):
        raise _DeepwaveValueError("peak_time must be finite.")

    if _dimensionless_fallback_needed(
        (freq,), dt, peak_time, peak_was_defaulted, nt, dtype, phase_factor=2 * math.pi
    ):
        u = _dimensionless_time(freq, nt, dt, peak_time, peak_was_defaulted)
        tau = u / n_cycles
        in_window = tau.abs() <= 0.5
        envelope = _windows.weights(2 * tau, window)
        carrier = torch.sin(2 * math.pi * u)
        y = torch.where(in_window, envelope * carrier, torch.zeros_like(u))
        y = _cast_wavelet_dtype(y, dtype)
    else:
        t = torch.arange(float(nt), dtype=dtype) * dt - peak_time
        tau = t / duration  # normalised time in [-0.5, 0.5] within the window
        in_window = tau.abs() <= 0.5

        # tau in [-0.5, 0.5] maps onto _windows' symmetric [-1, 1] coordinate.
        envelope = _windows.weights(2 * tau, window)

        carrier = torch.sin(2 * math.pi * freq * t)
        y = torch.where(in_window, envelope * carrier, torch.zeros_like(t))
        if dtype is not None:
            y = y.to(dtype)
    if nt > 0:
        _warn_if_burst_is_truncated(y, freq, n_cycles, nt, dt)
    return _expand_to_shape(y, shape)


def gaussian_pulse(
    freq: float,
    bandwidth_frac: float,
    *,
    nt: int,
    dt: float,
    peak_time: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    shape: _Shape = None,
) -> torch.Tensor:
    r"""Return a Gaussian-modulated sinusoid with a prescribed bandwidth.

    This is a cosine of frequency ``freq`` multiplied by a Gaussian envelope,
    the excitation used to model broadband ultrasound/NDT transducers. The
    envelope width is set by the *fractional -6 dB bandwidth*
    ``bandwidth_frac`` following the convention of ``scipy.signal.gausspulse``
    (F. Harris, "On the use of windows...", Proc. IEEE 66(1), 1978, for the
    windowing background): the magnitude spectrum is Gaussian and falls to
    :math:`-6\,\text{dB}` (a factor :math:`10^{-6/20} = 1/2`) of its peak at
    the frequencies :math:`f_c(1 \pm b/2)`, where :math:`f_c` is ``freq`` and
    :math:`b` is ``bandwidth_frac``. Equivalently the wavelet is

    .. math::

        s(t) = e^{-a\,\tau^2}\,\cos(2\pi f_c\,\tau), \qquad
        a = \frac{(\pi f_c b)^2}{4\ln 2}, \qquad \tau = t - t_{peak},

    since a Gaussian time envelope :math:`e^{-a\tau^2}` has the Gaussian
    spectrum :math:`\propto e^{-\pi^2 f^2 / a}`, whose half-amplitude
    (:math:`-6\,\text{dB}`) half-width is :math:`\Delta f = f_c b / 2`.

    Args:
        freq: The centre frequency of the sinusoid (Hz, if ``dt`` is in
            seconds).
        bandwidth_frac: The fractional -6 dB bandwidth :math:`b = \Delta f_{-6dB}
            / f_c` (e.g. ``0.5`` for a 50%-bandwidth transducer). Must be
            positive.
        nt: The number of time samples in the returned wavelet.
        dt: The time sample spacing (seconds).
        peak_time: The time (seconds) of the envelope peak. Optional; if not
            provided the pulse is centred in the record (``peak_time = nt*dt/2``),
            the same record-centring convention as :func:`ricker`,
            :func:`tone_burst` and :func:`ormsby`.
        dtype: The PyTorch datatype to use. Optional, defaults to PyTorch's
            default (float32).
        shape: Keyword-only. The leading dimensions of the returned tensor,
            with ``nt`` always last. ``None`` (the default) returns a
            one-dimensional ``[nt]`` tensor. ``(n_shots, n_sources)`` returns
            ``[n_shots, n_sources, nt]``, ready to pass straight to a
            propagator's ``source_amplitudes``. An ``int`` ``n`` is shorthand
            for ``(n,)`` and returns ``[n, nt]``; an empty sequence ``()``
            returns ``[nt]``, as ``None`` does. Every entry must be a
            positive integer. The leading dimensions are added with
            :meth:`torch.Tensor.expand`, so no data is copied and the result
            is a **view**: every shot and source shares one copy of the
            wavelet, and each added dimension has stride zero. Call
            ``.contiguous()`` or ``.clone()`` on it before modifying it in
            place. Deepwave's propagators do not modify their inputs, so
            passing the view to one directly is safe.

    Returns:
        A PyTorch tensor of shape ``[nt]`` containing the Gaussian pulse, or
        of shape ``[*shape, nt]`` if ``shape`` was given.

    Raises:
        TypeError: If ``shape`` is not ``None``, an int, or a sequence of
            ints.
        ValueError: If ``dt`` is zero, if ``freq`` or ``bandwidth_frac`` is
            not positive, or if any entry of ``shape`` is not positive.

    """
    if not math.isfinite(dt) or dt == 0:
        raise _DeepwaveValueError("dt cannot be zero or nonfinite.")
    if not math.isfinite(freq) or freq <= 0:
        raise _DeepwaveValueError("freq must be positive and finite.")
    if not math.isfinite(bandwidth_frac) or bandwidth_frac <= 0:
        raise _DeepwaveValueError("bandwidth_frac must be positive and finite.")

    peak_was_defaulted = peak_time is None
    if peak_time is None:
        peak_time = nt * dt / 2
    if not math.isfinite(peak_time):
        raise _DeepwaveValueError("peak_time must be finite.")

    gaussian_scale = math.pi * freq * bandwidth_frac
    if _dimensionless_fallback_needed(
        (freq,),
        dt,
        peak_time,
        peak_was_defaulted,
        nt,
        dtype,
        extra_squared_scale=gaussian_scale,
        phase_factor=2 * math.pi,
    ):
        u = _dimensionless_time(freq, nt, dt, peak_time, peak_was_defaulted)
        exponent = (math.pi**2 / (4 * math.log(2))) * (bandwidth_frac * u) ** 2
        envelope = torch.exp(-exponent)
        carrier = torch.cos(2 * math.pi * u)
        product = envelope * carrier
        y = torch.where((envelope == 0) | ~torch.isfinite(exponent), 0, product)
        y = _cast_wavelet_dtype(y, dtype)
    else:
        a = gaussian_scale**2 / (4 * math.log(2))
        t = torch.arange(float(nt), dtype=dtype) * dt - peak_time
        y = torch.exp(-a * t**2) * torch.cos(2 * math.pi * freq * t)
        if dtype is not None:
            y = y.to(dtype)
    return _expand_to_shape(y, shape)


def ormsby(
    f1: float,
    f2: float,
    f3: float,
    f4: float,
    *,
    nt: int,
    dt: float,
    peak_time: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    shape: _Shape = None,
) -> torch.Tensor:
    r"""Return an Ormsby wavelet with the specified four corner frequencies.

    The Ormsby wavelet is the standard seismic wavelet whose amplitude
    spectrum is an exact trapezoid: zero below ``f1``, ramping linearly up to
    full amplitude at ``f2``, flat (unity) between ``f2`` and ``f3``, then
    ramping linearly back down to zero at ``f4`` and beyond. This gives more
    direct control over the excited passband than :func:`ricker` (a single
    central frequency) or :func:`gaussian_pulse` (a fractional bandwidth
    around one centre frequency).

    With :math:`\tau = t - t_{peak}`, the wavelet is

    .. math::

        w(\tau) \propto
        \frac{\pi f_4^2\,\mathrm{sinc}^2(\pi f_4 \tau)
              - \pi f_3^2\,\mathrm{sinc}^2(\pi f_3 \tau)}{f_4 - f_3}
        -
        \frac{\pi f_2^2\,\mathrm{sinc}^2(\pi f_2 \tau)
              - \pi f_1^2\,\mathrm{sinc}^2(\pi f_1 \tau)}{f_2 - f_1},

    where :math:`\mathrm{sinc}(x) = \sin(x)/x` (the mathematical, *not*
    normalised, sinc), normalised so the returned wavelet's peak absolute
    amplitude is exactly ``1.0``. (Note: ``torch.sinc(x)`` computes the
    *normalised* sinc, :math:`\sin(\pi x)/(\pi x)`; since
    :math:`\mathrm{sinc}(\pi f \tau) = \sin(\pi f \tau) / (\pi f \tau) =
    \mathrm{torch.sinc}(f\tau)`, the mathematical sinc above is computed as
    ``torch.sinc(f * tau)`` directly, with no extra factor of :math:`\pi`.)

    Args:
        f1: The low-cut corner frequency (amplitude zero below this). All four
            corner frequencies are in Hz, if ``dt`` is in seconds, and must
            satisfy ``0 < f1 < f2 < f3 < f4``.
        f2: The low-pass corner frequency (amplitude reaches full scale here).
        f3: The high-pass corner frequency (amplitude starts rolling off here).
        f4: The high-cut corner frequency (amplitude zero above this).
        nt: The number of time samples.
        dt: The time sample spacing (seconds).
        peak_time: The time (in secs) of the peak amplitude. Optional; if not
            provided the wavelet is auto-centred in the record
            (``peak_time = nt * dt / 2``), the same record-centring
            convention as :func:`ricker`, :func:`tone_burst` and
            :func:`gaussian_pulse`, so the (symmetric) wavelet is not
            truncated. Passing an explicit value is unchanged.
        dtype: The PyTorch datatype to use. Optional, defaults to PyTorch's
            default (float32).
        shape: Keyword-only. The leading dimensions of the returned tensor,
            with ``nt`` always last. ``None`` (the default) returns a
            one-dimensional ``[nt]`` tensor. ``(n_shots, n_sources)`` returns
            ``[n_shots, n_sources, nt]``, ready to pass straight to a
            propagator's ``source_amplitudes``. An ``int`` ``n`` is shorthand
            for ``(n,)`` and returns ``[n, nt]``; an empty sequence ``()``
            returns ``[nt]``, as ``None`` does. Every entry must be a
            positive integer. The leading dimensions are added with
            :meth:`torch.Tensor.expand`, so no data is copied and the result
            is a **view**: every shot and source shares one copy of the
            wavelet, and each added dimension has stride zero. Call
            ``.contiguous()`` or ``.clone()`` on it before modifying it in
            place. Deepwave's propagators do not modify their inputs, so
            passing the view to one directly is safe.

    Returns:
        A PyTorch tensor of shape ``[nt]`` containing the Ormsby wavelet, or
        of shape ``[*shape, nt]`` if ``shape`` was given.

    Raises:
        TypeError: If ``shape`` is not ``None``, an int, or a sequence of
            ints.
        ValueError: If ``dt`` is zero, if the corner frequencies do not
            satisfy ``0 < f1 < f2 < f3 < f4``, or if any entry of ``shape``
            is not positive.

    """
    if not math.isfinite(dt) or dt == 0:
        raise _DeepwaveValueError("dt cannot be zero or nonfinite.")
    if not all(math.isfinite(f) for f in (f1, f2, f3, f4)) or not (
        0 < f1 < f2 < f3 < f4
    ):
        raise _DeepwaveValueError(
            "Corner frequencies must satisfy 0 < f1 < f2 < f3 < f4, got "
            f"f1={f1}, f2={f2}, f3={f3}, f4={f4}."
        )

    peak_was_defaulted = peak_time is None
    if peak_time is None:
        peak_time = nt * dt / 2
    if not math.isfinite(peak_time):
        raise _DeepwaveValueError("peak_time must be finite.")

    freqs = (f1, f2, f3, f4)
    if _dimensionless_fallback_needed(
        freqs,
        dt,
        peak_time,
        peak_was_defaulted,
        nt,
        dtype,
        extra_squared_scale=math.sqrt(math.pi) * f4,
    ):
        x1, x2, x3, x4 = (
            _dimensionless_time(f, nt, dt, peak_time, peak_was_defaulted) for f in freqs
        )
        r2, r3, low_ratio = f2 / f4, f3 / f4, f1 / f2
        high = (torch.sinc(x4) ** 2 - r3**2 * torch.sinc(x3) ** 2) / (1 - r3)
        low = r2 * (
            (torch.sinc(x2) ** 2 - low_ratio**2 * torch.sinc(x1) ** 2) / (1 - low_ratio)
        )
        y = high - low
        y = y / y.abs().max()
        y = _cast_wavelet_dtype(y, dtype)
    else:
        t: torch.Tensor = torch.arange(float(nt), dtype=dtype) * dt - peak_time

        def _term(f: float) -> torch.Tensor:
            return math.pi * f**2 * torch.sinc(f * t) ** 2

        y = (_term(f4) - _term(f3)) / (f4 - f3) - (_term(f2) - _term(f1)) / (f2 - f1)
        y = y / y.abs().max()
        if dtype is not None:
            y = y.to(dtype)
    return _expand_to_shape(y, shape)
