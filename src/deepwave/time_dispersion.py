r"""Koene-Robertsson time-dispersion correction.

Stability: provisional (see :doc:`/stability`).

Explicit second-order leapfrog time stepping (the scheme every Deepwave
propagator uses, see :doc:`discretisation`) is only *second*-order accurate in
time: it introduces *temporal dispersion*. Raising the spatial ``accuracy``
order does nothing to it -- it is governed solely by the time step. Concretely,
a mode that oscillates in the simulation at temporal angular frequency
:math:`\omega` carries the physics of a *lower* frequency

.. math::

    \Omega(\omega) = \frac{2}{\Delta t}\,\sin\!\Bigl(\frac{\omega\,\Delta t}{2}\Bigr)
    \;\le\; \omega ,

the exact leapfrog dispersion relation. Equivalently, physics that should
oscillate at :math:`\Omega` appears in the simulation *shifted up* to

.. math::

    \omega(\Omega) = \frac{2}{\Delta t}\,\arcsin\!\Bigl(\frac{\Omega\,\Delta t}{2}\Bigr)
    \;\ge\; \Omega ,
    \qquad \Omega < \frac{2}{\Delta t}.

These two maps are exact inverses on the simulation band
:math:`|\omega| < \pi/\Delta t`, which maps to the physical band
:math:`|\Omega| < 2/\Delta t`.

The scalar leapfrog oscillator identity is that the recursion of a single
oscillator,

.. math::

    u_{n+1} - 2u_n + u_{n-1} = -\Delta t^2\,\Omega_0^2\,u_n,

is solved *exactly* by :math:`u_n = \sin(\omega_\mathrm{num}\,n\,\Delta t)` with
:math:`\omega_\mathrm{num} = (2/\Delta t)\,\arcsin(\Omega_0\,\Delta t/2)`. This
identity describes a single scalar leapfrog oscillator. It does not by itself
establish exact correction of a finite signal or of other propagator families.

Koene, Robertsson, Broggini & Andersson (2018) remove the temporal dispersion
with two frequency-domain remappings, applied *outside* the propagator:

* **Source pre-correction** (:func:`correct_source`) pre-distorts the source
  wavelet so that, after the simulation shifts every frequency up, the physics
  lands where it should. It applies the inverse map: spectral content the user
  places at physical frequency :math:`\Omega` is moved to the simulation
  frequency :math:`\omega(\Omega)`. Concretely the corrected wavelet's spectrum
  at frequency :math:`\omega` is read from the input spectrum at
  :math:`\Omega(\omega) \le \omega`.
* **Receiver post-correction** (:func:`correct_receivers`) undoes the shift in
  the recorded data: content sitting at simulation frequency :math:`\omega` is
  moved back down to the physical frequency :math:`\Omega(\omega)`. Equivalently
  the corrected trace's spectrum at physical frequency :math:`\Omega` is read
  from the recorded spectrum at :math:`\omega(\Omega) \ge \Omega`.

The maps describe the scalar leapfrog frequency relation. Their use on finite
records is approximate: only the retained frequency band is remapped, and
content outside that band is lost. Finite-record spectral leakage, padding and
windowing also affect the result. Exact evaluation of the finite-record DTFT
does not make a remap round trip exact. This scalar relation does not establish
exact correction of attenuating, staggered-grid or PML dynamics, or remove
spatial discretisation error.

Which :math:`\Delta t` governs
------------------------------

The dispersion is set by the leapfrog step actually taken, which is the
propagator's **internal** time step ``internal_dt = dt / step_ratio``, *not* the
user-facing ``dt``. Deepwave subdivides ``dt`` into ``step_ratio`` internal
steps to satisfy the CFL condition (see :func:`deepwave.discretisation.cfl_condition_n`
and :doc:`discretisation`). Both functions here therefore take ``internal_dt``
explicitly -- there is no ``dt``/``step_ratio`` guessing -- and treat the time
axis of the passed tensors as sampled at that same ``internal_dt``.

The simplest recipe is to choose a ``dt`` that is already CFL-stable so
the propagator runs with ``step_ratio == 1`` (then ``internal_dt == dt``):
pre-correct the source at ``dt``, model, post-correct the receiver data at
``dt``. Use :func:`cfl_internal_dt` to obtain the exact ``internal_dt`` the
propagator would use for a given model, grid spacing and ``dt``. When
``step_ratio > 1`` the corrections must be applied at the internal rate; resample
your source/receiver tensors to ``internal_dt`` (with
:func:`deepwave.discretisation.upsample` /
:func:`deepwave.discretisation.downsample`) before calling these functions,
or force ``step_ratio == 1``.

Both transforms operate on the final (time) dimension, are batched over all
leading dimensions, are differentiable (pure tensor operations, so gradients
flow through to the source or data), and preserve the input floating dtype
(``float32`` or ``float64``). They evaluate the finite signal's discrete-time
Fourier transform at remapped frequencies followed by an inverse real FFT.
Finite-record remapping and
reconstruction are approximate and lose content outside the retained band;
padding and windowing affect the error. No machine-precision round-trip bound
is promised.
"""

import math
from collections.abc import Sequence
from typing import Optional, Union

import torch

from deepwave._common import cfl_condition_n as _cfl_condition_n
from deepwave._common import set_grid_spacing as _set_grid_spacing
from deepwave._exceptions import DeepwaveTypeError as _DeepwaveTypeError
from deepwave._exceptions import DeepwaveValueError as _DeepwaveValueError

__all__ = [
    "cfl_internal_dt",
    "correct_receivers",
    "correct_source",
]


def _validate_transform_args(
    signal: torch.Tensor,
    internal_dt: float,
    nt_out: Optional[int],
) -> None:
    """Validate the shared arguments of the two public transforms.

    Args:
        signal: The tensor to be transformed.
        internal_dt: The internal (leapfrog) time step.
        nt_out: The requested output length, or None.

    Raises:
        TypeError: If ``signal`` is not a tensor, ``internal_dt`` is not a
            float, or ``nt_out`` is not an int.
        ValueError: If ``signal`` has no dimensions, is not ``float32`` or
            ``float64``, ``internal_dt`` is not a finite positive number, or
            ``nt_out`` is not positive.
    """
    if not isinstance(signal, torch.Tensor):
        raise _DeepwaveTypeError("signal must be a torch.Tensor.")
    if signal.dim() == 0:
        raise _DeepwaveValueError("signal must have at least one dimension.")
    if signal.dtype not in (torch.float32, torch.float64):
        raise _DeepwaveValueError(
            f"signal must be a float32 or float64 tensor, got {signal.dtype}.",
        )
    try:
        internal_dt = float(internal_dt)
    except (TypeError, ValueError) as e:
        raise _DeepwaveTypeError("internal_dt must be a float.") from e
    if not (math.isfinite(internal_dt) and internal_dt > 0):
        raise _DeepwaveValueError("internal_dt must be a finite positive number.")
    if nt_out is not None:
        if not isinstance(nt_out, int) or isinstance(nt_out, bool):
            raise _DeepwaveTypeError("nt_out must be an int.")
        if nt_out <= 0:
            raise _DeepwaveValueError("nt_out must be positive.")


def _transform(
    signal: torch.Tensor,
    inverse: bool,
    nt_out: Optional[int],
) -> torch.Tensor:
    """Remap a signal's spectrum along the leapfrog dispersion relation.

    The corrected signal ``y`` of length ``nt_out`` (sampled at
    ``internal_dt``) has spectrum ``Y(w) = X(g(w))``, where ``X`` is the exact
    discrete-time Fourier transform of the finite input ``signal`` and ``w``
    ranges over the output rFFT grid ``w_k = 2*pi*k / (nt_out*internal_dt)``.
    The remap ``g`` is the leapfrog dispersion relation:

    * ``inverse=False`` (source pre-correction): ``g(w) = Omega(w) =
      (2/dt) sin(w dt/2)``, so content the caller placed at physical frequency
      ``Omega`` ends up at the simulation frequency ``omega(Omega)``.
    * ``inverse=True`` (receiver post-correction): ``g(w) = omega(w) =
      (2/dt) arcsin(w dt/2)``, moving recorded content at simulation frequency
      ``omega`` back to the physical frequency ``Omega(omega)``. Output
      frequencies above ``2/dt`` (where ``arcsin`` is undefined) are zeroed;
      no physical content can live there.

    Evaluating ``X`` at the (generally off-grid) remapped frequencies is done
    via the DTFT sum ``X(nu) = sum_n signal_n exp(-i nu n dt)`` rather than
    interpolating between FFT bins. This evaluates the finite-record spectrum
    at remapped frequencies; finite-record remapping remains approximate, with
    content outside the retained band lost.

    Args:
        signal: Real tensor of shape ``[..., nt]``; the time axis is the last
            dimension and is treated as sampled at ``internal_dt``.
        inverse: Selects the remap direction (see above).
        nt_out: Length of the output time axis. Defaults to the input length.

    Returns:
        The remapped real tensor of shape ``[..., nt_out]``, same dtype and
        device as ``signal``.
    """
    nt_in = signal.shape[-1]
    nt_out_ = nt_in if nt_out is None else int(nt_out)
    if signal.numel() == 0 or nt_in == 0:
        out_shape = (*signal.shape[:-1], nt_out_)
        return signal.new_zeros(out_shape) + signal.sum() * 0

    device = signal.device
    real_dtype = signal.dtype
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64

    # On the output rFFT grid, w*dt/2 = pi*k/nt_out. Work in these
    # dimensionless units because dt cancels from the DTFT phase; explicitly
    # forming frequencies in rad/s can overflow for finite positive dt.
    n_out_freq = nt_out_ // 2 + 1
    k = torch.arange(n_out_freq, dtype=torch.float64, device=device)
    half = math.pi * k / nt_out_

    # Remapped frequencies nu*dt and the receiver (arcsin) validity mask.
    if inverse:
        valid = half <= 1.0
        nu_dt = 2.0 * torch.arcsin(half.clamp(max=1.0))
    else:
        valid = torch.ones_like(half, dtype=torch.bool)
        nu_dt = 2.0 * torch.sin(half)

    # Exact DTFT of the finite input at the remapped frequencies:
    #   Y_k = sum_n signal_n exp(-i nu_k n dt).
    # Reduce the phase modulo 2*pi in float64 before casting so the complex
    # exponential stays accurate even in the float32 path.
    n = torch.arange(nt_in, dtype=torch.float64, device=device)
    phase = nu_dt.unsqueeze(-1) * n.unsqueeze(0)  # [n_out_freq, nt_in]
    phase = phase - 2.0 * math.pi * torch.round(phase / (2.0 * math.pi))
    phase = phase.to(real_dtype)
    kernel = torch.complex(torch.cos(phase), -torch.sin(phase)).to(complex_dtype)

    spectrum = torch.matmul(signal.to(complex_dtype), kernel.transpose(0, 1))
    if inverse and not bool(valid.all()):
        spectrum = spectrum * valid.to(spectrum.dtype)

    out: torch.Tensor = torch.fft.irfft(spectrum, n=nt_out_, dim=-1)
    return out.to(real_dtype)


def correct_source(
    source_amplitudes: torch.Tensor,
    internal_dt: float,
    nt_out: Optional[int] = None,
) -> torch.Tensor:
    r"""Pre-correct a source wavelet for leapfrog time dispersion.

    Applies the Koene-Robertsson source pre-transform: it pre-distorts the
    wavelet so that, after the second-order leapfrog time stepping shifts every
    temporal frequency up to ``omega(Omega) = (2/dt) arcsin(Omega dt/2)``, the
    modelled physics follows the scalar leapfrog frequency map. Feed the returned
    wavelet to the propagator in place of the original; pair it with
    :func:`correct_receivers` on the recorded data to reduce temporal
    dispersion.

    The transform is the inverse frequency map: the corrected spectrum at
    simulation angular frequency :math:`\omega` equals the input spectrum at
    the lower physical frequency :math:`\Omega(\omega) = (2/dt)\sin(\omega
    dt/2)`. Because the pre-warp pushes energy toward higher frequencies, the
    corrected wavelet is more oscillatory and may need a longer record to avoid
    time wrap-around; extend it with ``nt_out``.

    ``internal_dt`` **must be the propagator's internal (leapfrog) time step**
    ``dt / step_ratio``, and ``source_amplitudes`` are treated as sampled at
    that step. See the module docstring and :func:`cfl_internal_dt` for how to
    obtain it; the simplest usage is a CFL-stable ``dt`` giving
    ``step_ratio == 1`` so ``internal_dt == dt``.

    Args:
        source_amplitudes: Real tensor of shape ``[..., nt]`` (for example the
            usual ``[shot, source, nt]``) holding the source wavelet(s). The
            time axis is the last dimension. Any number of leading batch
            dimensions is supported.
        internal_dt: The internal (leapfrog) time step governing the temporal
            dispersion, in seconds. Must be positive.
        nt_out: Optional length of the returned time axis, in time samples.
            Defaults to the input ``nt``. Use a larger value to give the
            (broader-band) corrected wavelet room and avoid wrap-around.

    Returns:
        The pre-corrected source amplitudes, shape ``[..., nt_out]``, with the
        same dtype and device as ``source_amplitudes``.

    Raises:
        TypeError: If ``source_amplitudes`` is not a tensor, ``internal_dt`` is
            not a float, or ``nt_out`` is not an int.
        ValueError: If ``source_amplitudes`` is not a floating tensor with at
            least one dimension, ``internal_dt`` is not positive, or ``nt_out``
            is not positive.
    """
    _validate_transform_args(source_amplitudes, internal_dt, nt_out)
    return _transform(source_amplitudes, inverse=False, nt_out=nt_out)


def correct_receivers(
    receiver_amplitudes: torch.Tensor,
    internal_dt: float,
    nt_out: Optional[int] = None,
) -> torch.Tensor:
    r"""Post-correct recorded data for leapfrog time dispersion.

    Applies the Koene-Robertsson receiver post-transform: it moves recorded
    spectral content from the simulation frequency :math:`\omega` back down to
    the physical frequency :math:`\Omega(\omega) = (2/dt)\sin(\omega dt/2)`
    at which it should sit, reducing temporal dispersion described by the
    scalar leapfrog frequency map. Apply it to the receiver output of the
    propagator; pair it with :func:`correct_source` on the input wavelet.

    Equivalently, the corrected trace's spectrum at physical angular frequency
    :math:`\Omega` is read from the recorded spectrum at the higher simulation
    frequency :math:`\omega(\Omega) = (2/dt)\arcsin(\Omega dt/2)`. Physical
    frequencies above :math:`2/dt` cannot be represented by the leapfrog
    and are set to zero.

    ``internal_dt`` **must be the propagator's internal (leapfrog) time step**
    ``dt / step_ratio``, and ``receiver_amplitudes`` are treated as sampled at
    that step. See the module docstring and :func:`cfl_internal_dt`; the
    simplest usage is a CFL-stable ``dt`` giving ``step_ratio == 1`` so
    ``internal_dt == dt``.

    Args:
        receiver_amplitudes: Real tensor of shape ``[..., nt]`` (for example the
            usual ``[shot, receiver, nt]``) holding the recorded traces. The
            time axis is the last dimension. Any number of leading batch
            dimensions is supported.
        internal_dt: The internal (leapfrog) time step governing the temporal
            dispersion, in seconds. Must be positive.
        nt_out: Optional length of the returned time axis, in time samples.
            Defaults to the input ``nt``. The corrected traces are narrower
            band (energy moves to lower frequency), so the default length is
            normally sufficient.

    Returns:
        The dispersion-corrected receiver amplitudes, shape ``[..., nt_out]``,
        with the same dtype and device as ``receiver_amplitudes``.

    Raises:
        TypeError: If ``receiver_amplitudes`` is not a tensor, ``internal_dt``
            is not a float, or ``nt_out`` is not an int.
        ValueError: If ``receiver_amplitudes`` is not a floating tensor with at
            least one dimension, ``internal_dt`` is not positive, or ``nt_out``
            is not positive.
    """
    _validate_transform_args(receiver_amplitudes, internal_dt, nt_out)
    return _transform(receiver_amplitudes, inverse=True, nt_out=nt_out)


def cfl_internal_dt(
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    max_vel: float,
    courant_number: Optional[float] = None,
    ndim: Optional[int] = None,
) -> float:
    """Return the internal (leapfrog) time step a propagator would use.

    Convenience wrapper around :func:`deepwave.discretisation.cfl_condition_n` that
    returns just the internal time step ``internal_dt = dt / step_ratio`` -- the
    ``dt`` that governs temporal dispersion and that :func:`correct_source` /
    :func:`correct_receivers` require. Passing the *same* ``grid_spacing``,
    ``dt``, ``max_vel`` (and optional ``courant_number``) you give the
    propagator guarantees you correct with exactly the step the propagator runs.

    The CFL limit depends on the number of spatial dimensions (each dimension
    contributes to the bound), and a propagator applies a single ``grid_spacing``
    value to every dimension of its model. A single value therefore does not on
    its own say what the propagator would do, so ``ndim`` must be given with it;
    a sequence of per-dimension spacings already carries that information and
    needs no ``ndim``.

    Args:
        grid_spacing: The grid cell spacing in each spatial dimension (the same
            value or sequence passed to the propagator). If a single
            value (applied to every dimension, as the propagators do), ``ndim``
            must also be given. Deepwave is unit-agnostic: use whatever
            consistent system of units ``grid_spacing``, ``dt``, and
            ``max_vel`` are expressed in (see :doc:`/units`).
        dt: The user-facing time step, in the time units of that same
            consistent system.
        max_vel: The maximum absolute wavespeed in the model, in
            ``grid_spacing`` units per ``dt`` unit; the propagator derives
            ``step_ratio`` from this via the CFL condition.
        courant_number: Optional Courant number, matching the propagator's
            ``courant_number`` argument. If None (default), the conservative
            value the propagator uses by default (0.6) is applied, reproducing
            the historical internal step exactly.
        ndim: The number of spatial dimensions of the propagation. Required when
            ``grid_spacing`` is a single value; optional (but checked against
            the sequence's length) otherwise.

    Returns:
        The internal time step ``internal_dt``, in the time units of ``dt``.

    Raises:
        TypeError: If ``ndim`` is not an int.
        ValueError: If ``grid_spacing`` is a single value and ``ndim`` was not
            given, or if ``ndim`` is not positive.
        RuntimeError: If ``grid_spacing`` is a sequence whose length is not
            ``ndim``.
    """
    if ndim is not None:
        if not isinstance(ndim, int) or isinstance(ndim, bool):
            raise _DeepwaveTypeError("ndim must be an int.")
        if ndim < 1:
            raise _DeepwaveValueError("ndim must be >= 1.")
        spacing: Union[float, Sequence[float]] = _set_grid_spacing(grid_spacing, ndim)
    else:
        try:
            float(grid_spacing)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            spacing = grid_spacing
        else:
            raise _DeepwaveValueError(
                "ndim must be given when grid_spacing is a single value, as "
                "the number of spatial dimensions cannot be deduced from it "
                "and the CFL limit depends on it. Pass, for example, "
                "cfl_internal_dt(dx, dt, max_vel, ndim=2), or the sequence of "
                "per-dimension spacings, [dx, dx]."
            )
    inner_dt, _step_ratio = _cfl_condition_n(
        spacing,
        dt,
        max_vel,
        courant_number=courant_number,
    )
    return inner_dt
