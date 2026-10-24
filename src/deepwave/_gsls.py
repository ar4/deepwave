"""Shared generalised-standard-linear-solid (GSLS) constant-Q machinery.

Used by :mod:`deepwave.viscoacoustic` to attenuate the bulk/pressure modulus.
The technique (Emmerich & Korn 1987; Blanch, Robertsson & Symes
1995; see also Carcione's textbook treatment) is independent of which
physical modulus is being relaxed: a target quality factor ``Q`` at a
reference frequency ``omega0 = 2*pi*q_freq`` (single mechanism), or a flat
effective ``Q`` across a frequency band ``q_band`` (multiple mechanisms), is
mapped to a set of Crank-Nicolson memory-variable coefficients that, applied
to ANY relaxed modulus, produce a first-order (constant-Q) linear
viscoelastic/viscoacoustic response for that modulus.

For a single mechanism (``q_n_mechanisms == 1``), the exact nonlinear
single-SLS relation is used: given the relaxed modulus ``M`` and target
quality factor ``Q`` at ``omega0``, the relaxation times are::

    tausig = (sqrt(1 + 1/Q**2) - 1/Q) / omega0
    taueps = 1 / (omega0**2 * tausig)

so that the SLS quality factor equals ``Q`` exactly at ``omega0``.

For multiple mechanisms (``q_n_mechanisms > 1``, GSLS), ``q_n_mechanisms``
relaxation frequencies are chosen log-spaced across ``q_band`` and their
per-mechanism nonnegative weights are fit ONCE (float64, independent of any model
tensor) by constrained least squares so that the weak-loss ``1/Q(omega)``
approximates a flat target across the band; because the fit is linear in
each mechanism's relaxation-time defect, the weights scale exactly as
``1/Q`` for a spatially-varying target ``Q``, keeping the whole
construction differentiable without needing autograd through
``torch.linalg.lstsq`` itself. This is the standard weak-attenuation
(large-Q) GSLS approximation, distinct from the single-mechanism branch's
exact nonlinear relation; the two coincide to leading order in ``1/Q``.
"""

import math
import operator
from typing import Any, Optional

import torch

from deepwave import _common
from deepwave._exceptions import DeepwaveTypeError, DeepwaveValueError


def validate_configuration(
    q_freq: Optional[float],
    q_n_mechanisms: Any,
    q_band: Optional[tuple[float, float]],
) -> tuple[Optional[float], int, Optional[tuple[float, float]]]:
    """Validate attenuation parameters before direct or segmented preparation.

    Return canonical numeric values so every entry point fits the same
    relaxation law and rejects invalid counts/bands before coefficient work.
    """
    if isinstance(q_n_mechanisms, bool):
        raise DeepwaveTypeError("viscoacoustic: q_n_mechanisms must be an integer.")
    try:
        q_n_mechanisms = operator.index(q_n_mechanisms)
    except TypeError as error:
        raise DeepwaveTypeError(
            "viscoacoustic: q_n_mechanisms must be an integer."
        ) from error
    if q_n_mechanisms < 1:
        raise DeepwaveValueError(
            f"viscoacoustic: q_n_mechanisms must be >= 1, but got {q_n_mechanisms}."
        )
    if q_n_mechanisms == 1:
        if q_band is not None:
            raise DeepwaveValueError(
                "viscoacoustic: q_band must be None when q_n_mechanisms == 1 "
                "(the default single-mechanism SLS is controlled by q_freq "
                "instead)."
            )
        if q_freq is None:
            raise DeepwaveValueError(
                "viscoacoustic: q_freq is required. It is the reference frequency at "
                "which Q is specified, in the same units as 1/dt, and it has "
                "no default: a reference frequency that Deepwave chose would "
                "silently redefine the attenuation model. Pass the frequency "
                "your Q was measured or chosen at, e.g. q_freq=25.0."
            )
        q_freq = _common.check_q_freq(q_freq, "viscoacoustic")
    else:
        if q_band is None:
            raise DeepwaveValueError(
                "viscoacoustic: q_band=(f1, f2) is required when q_n_mechanisms > 1."
            )
        try:
            q_band_f1, q_band_f2 = q_band
        except (TypeError, ValueError) as error:
            raise DeepwaveValueError(
                "viscoacoustic: q_band must contain exactly two frequencies."
            ) from error
        q_band_f1 = _common.check_q_freq(
            q_band_f1, "viscoacoustic", parameter="q_band[0]"
        )
        q_band_f2 = _common.check_q_freq(
            q_band_f2, "viscoacoustic", parameter="q_band[1]"
        )
        if not q_band_f1 < q_band_f2:
            raise DeepwaveValueError(
                "viscoacoustic: q_band=(f1, f2) must contain finite frequencies "
                "and satisfy 0 < f1 < f2, but "
                f"got {(q_band_f1, q_band_f2)}."
            )
        q_band = (q_band_f1, q_band_f2)
        if q_freq is not None:
            raise DeepwaveValueError(
                "viscoacoustic: q_freq must be None when q_n_mechanisms > 1, which "
                "fits a constant Q across a frequency band rather than at a "
                "single reference frequency; pass the band as q_band=(f1, f2)."
            )
    return q_freq, q_n_mechanisms, q_band


def q_to_relaxation_times(
    q: torch.Tensor, omega0: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Maps a quality factor ``Q`` at ``omega0`` to SLS relaxation times.

    Uses the constant-Q-at-omega0 single-SLS relations::

        tausig = (sqrt(1 + 1/Q**2) - 1/Q) / omega0
        taueps = 1 / (omega0**2 * tausig)

    Args:
        q: Quality factor at ``omega0`` (must be positive).
        omega0: Reference angular frequency (rad per unit time).

    Returns:
        Tuple ``(taueps, tausig)``.
    """
    inv_q = 1.0 / q
    # Rationalize sqrt(1 + x**2) - x: the subtraction rounds to zero for
    # small positive Q in float32, although the relaxation time is nonzero.
    tausig = 1.0 / (omega0 * (torch.hypot(torch.ones_like(inv_q), inv_q) + inv_q))
    taueps = 1.0 / (omega0**2 * tausig)
    return taueps, tausig


def gsls_relaxation_frequencies(
    q_band: tuple[float, float], n_mechanisms: int
) -> list[float]:
    """``n_mechanisms`` relaxation frequencies, log-spaced across ``q_band``.

    Standard choice for a generalised-standard-linear-solid (GSLS) constant-Q
    fit (Emmerich & Korn 1987; Blanch, Robertsson & Symes 1995; see also
    Carcione's textbook treatment): the mechanisms' peak-attenuation
    frequencies are spread log-uniformly across the target band so their
    individual ``1/Q_l(omega)`` Debye peaks overlap to approximate a flat
    (frequency-independent) ``1/Q`` across the whole band.
    """
    f1, f2 = q_band
    if n_mechanisms == 1:
        return [f1]
    # Interpolating logs avoids overflowing f2/f1 for representable endpoints.
    log_f1, log_f2 = math.log(f1), math.log(f2)
    return [
        math.exp(log_f1 + (log_f2 - log_f1) * i / (n_mechanisms - 1))
        for i in range(n_mechanisms)
    ]


def _nonnegative_least_squares(design: torch.Tensor) -> torch.Tensor:
    """Deterministic active-set NNLS for a unit target, with a KKT certificate.

    This small CPU float64 problem is solved once per constitutive setup. SVD
    solves on active columns are rank-aware; no per-cell optimizer is involved.
    """
    n_rows, n_cols = design.shape
    target = torch.ones(n_rows, dtype=torch.float64)
    gram = design.T @ design
    rhs = design.T @ target
    weights = torch.zeros(n_cols, dtype=torch.float64)
    active: list[int] = []
    # The certificate is relative to the normal equations. This tolerance is
    # above float64 roundoff accumulated over the fit samples, but small
    # compared with a unit target's normal-equation scale.
    scale = max(1.0, float(rhs.abs().max()))
    tol = 2e-10 * scale
    for _ in range(max(100, 10 * n_cols * n_cols)):
        reduced_gradient = rhs - gram @ weights
        candidates = [i for i in range(n_cols) if i not in active]
        if not candidates or max(float(reduced_gradient[i]) for i in candidates) <= tol:
            break
        entering = max(candidates, key=lambda i: (float(reduced_gradient[i]), -i))
        active.append(entering)
        for _ in range(max(20, 4 * n_cols)):
            solution = torch.linalg.lstsq(
                design[:, active], target[:, None], driver="gelsd"
            ).solution[:, 0]
            if bool((solution > 0).all()):
                weights[active] = solution
                break
            ratios = [
                float(weights[idx] / (weights[idx] - solution[j]))
                for j, idx in enumerate(active)
                if solution[j] <= 0
            ]
            if not ratios:
                raise DeepwaveValueError("GSLS nonnegative fit failed to advance.")
            alpha = min(ratios)
            for j, idx in enumerate(active):
                weights[idx] += alpha * (solution[j] - weights[idx])
            leaving = [idx for idx in active if weights[idx] <= tol / scale]
            for idx in leaving:
                weights[idx] = 0.0
                active.remove(idx)
            if not leaving:
                raise DeepwaveValueError("GSLS nonnegative fit stalled.")
        else:
            raise DeepwaveValueError("GSLS nonnegative fit did not converge.")
    else:
        raise DeepwaveValueError("GSLS nonnegative fit did not converge.")
    residual = gram @ weights - rhs
    positive = weights > 0
    stationarity = (
        float(residual[positive].abs().max()) if bool(positive.any()) else 0.0
    )
    dual = (
        float((-residual[~positive]).clamp_min(0).max())
        if bool((~positive).any())
        else 0.0
    )
    if (
        not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or stationarity > tol
        or dual > tol
    ):
        raise DeepwaveValueError(
            "GSLS nonnegative fit failed its scaled optimality check."
        )
    return weights


def gsls_fit_weights(
    q_band: tuple[float, float],
    n_mechanisms: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constant-Q GSLS fit: per-mechanism relaxation times and weights.

    Chooses ``n_mechanisms`` relaxation times ``tausig_l = 1/(2*pi*f_l)``
    (see :func:`gsls_relaxation_frequencies`) and solves, ONCE, a linear
    least-squares fit (float64, independent of any model tensor) for weights
    ``w_l`` such that ``sum_l w_l * (omega*tausig_l)/(1+(omega*tausig_l)**2)
    ~= 1`` across a fine sampling of angular frequencies ``omega`` spanning
    ``q_band``. Because the GSLS forward relation is linear in the
    per-mechanism relaxation-time defect ``tau_l`` and the target
    ``1/Q(omega) = 1/Q`` is frequency-independent (a flat target), the
    least-squares solution for a general target ``1/Q`` is exactly ``tau_l =
    w_l / Q`` -- i.e. these fixed weights, divided by the (possibly
    spatially-varying, differentiable) target ``Q`` tensor. This keeps the
    only q-dependent step a plain elementwise division, so the fit is
    differentiable w.r.t. ``q`` without needing autograd through
    ``torch.linalg.lstsq`` itself.

    This is the standard weak-attenuation (large-Q) GSLS approximation: each
    mechanism's contribution to ``1/Q`` is linearised in its own relaxation
    defect, exactly as the single-mechanism path's ``tau = taueps/tausig -
    1`` term appears linearly in the unrelaxed modulus. It is deliberately
    NOT the exact nonlinear single-SLS relation :func:`q_to_relaxation_times`
    uses (that relation has no closed-form multi-mechanism least-squares
    generalisation); the two coincide to leading order in ``1/Q``.
    """
    freqs = gsls_relaxation_frequencies(q_band, n_mechanisms)
    tausig64 = torch.tensor(
        [1.0 / (2.0 * math.pi * f) for f in freqs], dtype=torch.float64
    )
    f1, f2 = q_band
    n_probe = max(200, 20 * n_mechanisms)
    log_f = torch.linspace(math.log(f1), math.log(f2), n_probe, dtype=torch.float64)
    log_peaks = torch.tensor([math.log(f) for f in freqs], dtype=torch.float64)
    # x/(1+x²) = exp(-|log(x)|)/(1+exp(-2|log(x)|)). Forming
    # omega*tau first overflows for wide yet representable frequency bands.
    separation = (log_f.unsqueeze(1) - log_peaks.unsqueeze(0)).abs()
    e = torch.exp(-separation)
    design = e / (1.0 + e.square())
    if not bool(torch.isfinite(tausig64).all()) or bool((tausig64 <= 0).any()):
        raise DeepwaveValueError(
            "GSLS relaxation times are not representable in float64."
        )
    weights64 = _nonnegative_least_squares(design)
    times = tausig64.to(dtype=dtype, device=device)
    weights = weights64.to(dtype=dtype, device=device)
    if (
        not bool(torch.isfinite(times).all())
        or bool((times <= 0).any())
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or bool(((weights64 > 0) & (weights == 0).cpu()).any())
    ):
        raise DeepwaveValueError(
            "GSLS fit is not representable in the requested dtype; use float64 "
            "or a narrower frequency band."
        )
    return (
        times,
        weights,
    )


def unrelaxed_factor(
    q: torch.Tensor,
    n_mechanisms: int = 1,
    q_band: Optional[tuple[float, float]] = None,
) -> torch.Tensor:
    """Ratio of the high-frequency modulus to the relaxed modulus.

    This is the same constitutive coefficient used by :func:`sls_coefficients`,
    evaluated before the CFL time step is chosen. Positive infinite Q gives
    the exact lossless limit.
    """
    if n_mechanisms == 1:
        inv_q = 1.0 / q
        return (torch.hypot(torch.ones_like(inv_q), inv_q) + inv_q).square()
    if q_band is None:
        raise DeepwaveValueError("q_band is required when n_mechanisms > 1.")
    _, weights = gsls_fit_weights(q_band, n_mechanisms, q.dtype, q.device)
    return 1.0 + weights.sum() / q


def unrelaxed_speed(
    v: torch.Tensor,
    q: torch.Tensor,
    n_mechanisms: int = 1,
    q_band: Optional[tuple[float, float]] = None,
) -> torch.Tensor:
    """Validate the Q model and derive a representable viscoacoustic CFL speed."""
    if (
        not _common._torch_is_compiling()  # noqa: SLF001
        and _common.values_readable(q)
        and not bool((q > 0).all())
    ):
        raise DeepwaveValueError(
            "viscoacoustic: q must be positive; +inf denotes the lossless limit."
        )
    speed = v.abs() * unrelaxed_factor(q, n_mechanisms, q_band).sqrt()
    if _common.values_readable(speed) and not bool(torch.isfinite(speed).all()):
        raise DeepwaveValueError(
            "viscoacoustic: the unrelaxed wave speed is non-finite. "
            "Check v and q, or use float64 for a larger representable range."
        )
    return speed


def sls_coefficients(
    modulus: torch.Tensor,
    q: torch.Tensor,
    dt: float,
    omega0: float,
    q_n_mechanisms: int = 1,
    q_band: Optional[tuple[float, float]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GSLS Crank-Nicolson memory-variable coefficients for one modulus.

    Given a relaxed modulus ``modulus`` and a target quality factor ``q``
    (at ``omega0`` when ``q_n_mechanisms == 1``, or across ``q_band`` when
    ``q_n_mechanisms > 1``), returns ``(m1, c2, ar)``: ``m1`` is the
    unrelaxed modulus (``modulus * (1 + sum_l tau_l)``), and ``c2``/``ar``
    are the per-mechanism Crank-Nicolson memory-variable update
    coefficients::

        ar_l  = (1 - dt/(2 tausig_l)) / (1 + dt/(2 tausig_l))
        c2_l  = modulus * tau_l * (dt/tausig_l) / (1 + dt/(2 tausig_l))

    A caller advances each mechanism's memory variable ``r_l`` by
    ``r_l_new = ar_l * r_l - c2_l * S`` (``S`` the driving strain-rate term)
    and couples the field being updated to ``m1``/the memory variables with
    whatever sign its own constitutive convention requires (this recursion
    for ``r_l`` itself has a fixed sign, independent of that convention --
    see ``viscoacoustic.update_pressure`` for pressure's
    ``-`` coupling).

    For ``q_n_mechanisms == 1``, ``c2``/``ar`` are plain fields shaped like
    ``modulus``/``q``. For ``q_n_mechanisms > 1``, they gain a new LEADING
    axis of size ``q_n_mechanisms`` (one Crank-Nicolson coefficient pair per
    relaxation mechanism, see :func:`gsls_fit_weights`), and ``m1`` sums the
    per-mechanism relaxation defects.

    Args:
        modulus: The relaxed modulus to be attenuated.
        q: Quality factor at ``omega0`` (``q_n_mechanisms == 1``) or the
            target constant-Q value across ``q_band`` (``q_n_mechanisms >
            1``); must be broadcastable against ``modulus``.
        dt: Internal time step (after any CFL sub-stepping).
        omega0: Reference angular frequency (``q_n_mechanisms == 1`` only).
        q_n_mechanisms: Number of GSLS relaxation mechanisms.
        q_band: ``(f1, f2)`` frequency band for the GSLS constant-Q fit;
            required when ``q_n_mechanisms > 1``.

    Returns:
        ``(m1, c2, ar)``.
    """
    if q_n_mechanisms == 1:
        _, tausig = q_to_relaxation_times(q, omega0)
        inv_q = 1.0 / q
        # Avoid cancellation in taueps/tausig - 1 for weak loss.
        tau = 2.0 * inv_q * (torch.hypot(torch.ones_like(inv_q), inv_q) + inv_q)
        half = dt / (2.0 * tausig)
        ar = (1.0 - half) / (1.0 + half)
        b_r = (dt / tausig) / (1.0 + half)
        m1 = modulus * (1.0 + tau)
        c2 = modulus * tau * b_r
        return m1, c2, ar

    if q_band is None:
        raise DeepwaveValueError("q_band is required when q_n_mechanisms > 1.")
    tausig_l, weights_l = gsls_fit_weights(q_band, q_n_mechanisms, q.dtype, q.device)
    bshape = [q_n_mechanisms] + [1] * q.ndim
    tausig_l = tausig_l.reshape(bshape)
    weights_l = weights_l.reshape(bshape)
    tau = weights_l / q.unsqueeze(0)  # (L, *q.shape)
    half = dt / (2.0 * tausig_l)
    # ar_l has no spatial dependence (tausig_l is a fixed hyperparameter, not
    # derived from q); broadcast it to a full (L, *q.shape) field anyway so
    # the returned coefficient is shape-uniform regardless of L.
    ar = (1.0 - half) / (1.0 + half) + torch.zeros_like(tau)
    b_r = (dt / tausig_l) / (1.0 + half)
    m1 = modulus * (1.0 + tau.sum(0))
    c2 = modulus.unsqueeze(0) * tau * b_r
    return m1, c2, ar
