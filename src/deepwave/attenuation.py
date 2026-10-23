"""Inspect the constitutive response used by :func:`deepwave.viscoacoustic`.

This module is provisional. Frequencies use cycles per unit time and the
positive-loss Fourier convention ``exp(i*omega*t)``.
"""

import math
from typing import Optional

import torch

from deepwave import _gsls
from deepwave._exceptions import DeepwaveTypeError, DeepwaveValueError

__all__ = ["response"]


def response(
    q: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    q_freq: Optional[float] = None,
    q_n_mechanisms: int = 1,
    q_band: Optional[tuple[float, float]] = None,
) -> torch.Tensor:
    """Return the dimensionless complex modulus ``M(omega)/K``.

    The output has shape ``(*q.shape, frequencies.numel())`` and complex64 or
    complex128 dtype matching ``q``. Its real and imaginary parts are storage
    and positive loss. At DC the result is one. For multiple mechanisms,
    ``q`` is the weak-loss band target; ``real/imag`` is the *actual* effective
    Q and generally differs from that target, especially at strong loss.
    Gradients with respect to ``q`` are preserved.
    """
    if not isinstance(q, torch.Tensor) or q.dtype not in (torch.float32, torch.float64):
        raise DeepwaveTypeError(
            "attenuation.response: q must be a float32/float64 tensor."
        )
    if not isinstance(frequencies, torch.Tensor):
        raise DeepwaveTypeError("attenuation.response: frequencies must be a tensor.")
    if frequencies.ndim != 1 or frequencies.is_complex():
        raise DeepwaveValueError(
            "attenuation.response: frequencies must be a one-dimensional real tensor."
        )
    if not bool(((q > 0) & torch.isfinite(q)).all()):
        raise DeepwaveValueError("attenuation.response: q must be positive and finite.")
    if not bool(((frequencies >= 0) & torch.isfinite(frequencies)).all()):
        raise DeepwaveValueError(
            "attenuation.response: frequencies must be finite and nonnegative."
        )
    q_freq, q_n_mechanisms, q_band = _gsls.validate_configuration(
        q_freq, q_n_mechanisms, q_band
    )
    frequency = frequencies.to(device=q.device, dtype=q.dtype)
    if not bool(torch.isfinite(frequency).all()):
        raise DeepwaveValueError(
            "attenuation.response: frequencies are not representable in q's dtype."
        )
    positive = frequency > 0
    log_frequency = torch.where(positive, frequency, 1.0).log()
    if q_n_mechanisms == 1:
        if q_freq is None:
            raise DeepwaveValueError("attenuation.response: q_freq is required.")
        inv_q = 1.0 / q
        root = torch.hypot(torch.ones_like(inv_q), inv_q) + inv_q
        strength = 2.0 * inv_q * root
        log_times = -math.log(2.0 * math.pi) - math.log(q_freq) - root.log()
        times = log_times.exp()
        log_x = log_frequency - math.log(q_freq) - root.log().unsqueeze(-1)
        strength = strength.unsqueeze(-1)
        mechanism_axis = None
    else:
        if q_band is None:
            raise DeepwaveValueError("attenuation.response: q_band is required.")
        times_l, weights_l = _gsls.gsls_fit_weights(
            q_band, q_n_mechanisms, q.dtype, q.device
        )
        log_x = (
            times_l.log().reshape((-1,) + (1,) * q.ndim + (1,))
            + log_frequency
            + math.log(2.0 * math.pi)
        )
        strength = weights_l.reshape((-1,) + (1,) * q.ndim + (1,)) / q.unsqueeze(
            0
        ).unsqueeze(-1)
        times = times_l
        mechanism_axis = 0
    if (
        not bool(torch.isfinite(times).all())
        or bool((times <= 0).any())
        or not bool(torch.isfinite(strength).all())
        or bool((strength < 0).any())
    ):
        raise DeepwaveValueError(
            "attenuation.response: relaxation time or strength is not "
            "representable in q's dtype; use float64 or a less extreme Q/band."
        )
    # Avoid overflowing omega or x, including in an unselected autograd branch.
    # logsumexp also preserves derivatives at the relaxation peak (x == 1).
    dc_frequency = torch.where(positive, 0.0, frequency)
    dc_times = (
        times.unsqueeze(-1)
        if mechanism_axis is None
        else times.reshape((-1,) + (1,) * q.ndim + (1,))
    )
    # Keep the exact rational law at DC, including higher frequency derivatives.
    # Mask before multiplication to keep the unused branch finite.
    dc_x = (dc_times * dc_frequency) * (2.0 * math.pi)
    dc_denominator = 1.0 + dc_x.square()
    storage_basis = torch.where(
        positive, (2.0 * log_x).sigmoid(), dc_x.square() / dc_denominator
    )
    loss_basis = torch.where(
        positive,
        torch.stack((log_x, -log_x)).logsumexp(dim=0).neg().exp(),
        dc_x / dc_denominator,
    )
    storage = strength * storage_basis
    loss = strength * loss_basis
    if mechanism_axis is not None:
        storage = storage.sum(mechanism_axis)
        loss = loss.sum(mechanism_axis)
    modulus = torch.complex(1.0 + storage, loss)
    if not bool(torch.isfinite(modulus).all()):
        raise DeepwaveValueError(
            "attenuation.response: complex modulus is not representable in q's dtype."
        )
    return modulus
