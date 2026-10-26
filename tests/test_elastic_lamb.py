"""Lamb-wave dispersion validation for the elastic propagator.

A traction-free elastic plate (vacuum above *and* below) is a waveguide whose
guided modes obey the Rayleigh--Lamb dispersion relations. This test excites the
fundamental symmetric mode (S0) at a low frequency--thickness product -- where
only the fundamental modes propagate -- measures its phase velocity from the
finite-difference simulation, and compares it against the S0 root of the
Rayleigh--Lamb equation solved numerically here (by bisection; scipy is not a
Deepwave dependency). This is an analytic-ground-truth test.

Physics / method notes
----------------------
* The plate lies along the fast (``x``) axis; its thickness is along the slow
  (``y``) axis, with a few vacuum cells above and below realising the two free
  surfaces via Deepwave's improved-vacuum method (``vp = vs = rho = 0`` there;
  see ``docs/boundaries.rst``). The ``x`` faces keep a PML so guided energy
  leaves cleanly; the ``y`` faces use ``pml_width = 0`` (the wave never reaches
  the vacuum truncation).
* S0 is excited by a symmetric ``x``-force "piston" spanning the full thickness.
* The S0 phase velocity is measured from the ``x``-velocity (``vx``) recorded at
  two receivers on the plate mid-plane. Measuring ``vx`` *on the mid-plane*
  rejects the antisymmetric A0 mode (whose ``vx`` is odd about the mid-plane and
  therefore zero there), so the phase measurement is a clean S0 reading even if
  a little A0 is present.
* The two receivers are closer than half an S0 wavelength apart, so the phase
  difference is unambiguous (no 2*pi wrap).

The target validated here is the **S0 symmetric Lamb mode** phase velocity. Note
that at this frequency--thickness the numerically solved Rayleigh--Lamb root
(~5328 m/s) is measurably distinct from the thin-plate extensional asymptote
``sqrt(E / (rho (1 - nu^2)))`` (~5414 m/s), so this genuinely tests the full
dispersion relation rather than the asymptote.
"""

import cmath
import math

import torch

from deepwave import elastic, vpvsrho_to_lambmubuoyancy
from deepwave.wavelets import tone_burst


def _rayleigh_lamb_symmetric(
    c: float,
    omega: float,
    c_l: float,
    c_t: float,
    h: float,
) -> float:
    """Symmetric Rayleigh--Lamb characteristic function (real-valued).

    Zero when phase velocity ``c`` is a symmetric-mode solution at angular
    frequency ``omega`` for a plate of half-thickness ``h`` with bulk P- and
    S-wave speeds ``c_l``, ``c_t``. Uses complex ``p``/``q`` so it is valid
    whether they are real or imaginary; the result is real for physical roots.

    tan(qh)/tan(ph) = -4 k^2 p q / (q^2 - k^2)^2, rearranged to avoid poles.
    """
    k = omega / c
    p = cmath.sqrt((omega / c_l) ** 2 - k**2)
    q = cmath.sqrt((omega / c_t) ** 2 - k**2)
    term = (q**2 - k**2) ** 2 * cmath.sin(q * h) * cmath.cos(p * h) + (
        4 * k**2 * p * q * cmath.cos(q * h) * cmath.sin(p * h)
    )
    return term.real


def _solve_symmetric_roots(
    omega: float,
    c_l: float,
    c_t: float,
    h: float,
) -> list:
    """Find symmetric-mode phase-velocity roots in (c_t, c_l) by bisection."""
    lo, hi = c_t * 1.02, c_l * 0.999
    n_scan = 400
    cs = [lo + i * (hi - lo) / n_scan for i in range(n_scan + 1)]
    vals = [_rayleigh_lamb_symmetric(c, omega, c_l, c_t, h) for c in cs]
    roots = []
    for i in range(len(cs) - 1):
        if vals[i] == 0.0 or vals[i] * vals[i + 1] < 0.0:
            a, b, fa = cs[i], cs[i + 1], vals[i]
            for _ in range(80):
                m = 0.5 * (a + b)
                fm = _rayleigh_lamb_symmetric(m, omega, c_l, c_t, h)
                if fa * fm <= 0.0:
                    b = m
                else:
                    a, fa = m, fm
            roots.append(0.5 * (a + b))
    return roots


def test_lamb_s0_phase_velocity_matches_rayleigh_lamb(record_property) -> None:
    """Measured S0 phase velocity matches the Rayleigh--Lamb prediction."""
    dtype = torch.float64
    vp, vs, rho = 6000.0, 3200.0, 2000.0
    dx = 0.5e-3
    vac = 8  # vacuum cells above and below the plate
    thickness = 15  # odd, so the mid-plane is a grid row
    ny = vac + thickness + vac
    nx = 440
    mid = vac + thickness // 2
    f0 = 150e3
    n_cycles = 3
    dt = 3e-8
    nt = 1600

    c_l, c_t = vp, vs
    h = thickness * dx / 2
    omega = 2 * math.pi * f0

    # Only the fundamental modes propagate below fd = c_t / 2.
    assert f0 * thickness * dx < c_t / 2

    roots = _solve_symmetric_roots(omega, c_l, c_t, h)
    assert len(roots) == 1, f"expected a single S0 root, got {roots}"
    c_pred = roots[0]
    # Distinct from the thin-plate asymptote => tests the full relation.
    nu = (c_l**2 - 2 * c_t**2) / (2 * (c_l**2 - c_t**2))
    c_plate = c_t * math.sqrt(2 / (1 - nu))
    assert abs(c_pred - c_plate) / c_plate > 0.01

    # Build the plate model (vacuum above/below).
    vp_t = torch.zeros(ny, nx, dtype=dtype)
    vs_t = torch.zeros(ny, nx, dtype=dtype)
    rho_t = torch.zeros(ny, nx, dtype=dtype)
    vp_t[vac : vac + thickness] = vp
    vs_t[vac : vac + thickness] = vs
    rho_t[vac : vac + thickness] = rho
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp_t, vs_t, rho_t)

    # Symmetric x-force piston across the thickness => S0.
    source_x = 60
    src_rows = list(range(vac, vac + thickness))
    wavelet = tone_burst(
        f0,
        n_cycles,
        nt=nt,
        dt=dt,
        peak_time=n_cycles / f0 / 2,
        dtype=dtype,
    )
    src_amp = wavelet.repeat(1, len(src_rows), 1)
    src_loc = torch.tensor([[[r, source_x] for r in src_rows]])

    rx = (240, 260)  # < half an S0 wavelength apart
    rec_loc = torch.tensor([[[mid, rx[0]], [mid, rx[1]]]])

    out = elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_x=src_amp,
        source_locations_x=src_loc,
        receiver_locations_x=rec_loc,
        pml_width=[0, 0, 20, 20],
        pml_freq=f0,
        accuracy=4,
    )
    rec = out[-1][0]  # receiver_amplitudes_x, shape [2, nt]

    # Single-frequency DFT at omega to read the phase at each receiver.
    n = torch.arange(nt, dtype=dtype)
    phasor = torch.exp(-1j * omega * n * dt)
    spec = (rec * phasor).sum(dim=1)
    # A clean S0 packet arrived at both receivers.
    assert spec.abs().min() > 0.1 * spec.abs().max()

    dphi = (torch.angle(spec[1]) - torch.angle(spec[0])).item()
    dphi = (dphi + math.pi) % (2 * math.pi) - math.pi  # wrap to (-pi, pi]
    # r2 is farther, so it lags r1: dphi in (-pi, 0).
    assert -math.pi < dphi < 0
    phase_lag = -dphi
    sep = (rx[1] - rx[0]) * dx
    c_meas = omega * sep / phase_lag

    rel_err = abs(c_meas - c_pred) / c_pred
    record_property("phase_velocity_relative_error", rel_err)
    record_property("measured_phase_velocity", c_meas)
    record_property("reference_phase_velocity", c_pred)
    assert rel_err < 0.03, (
        f"S0 phase velocity {c_meas:.1f} m/s vs Rayleigh-Lamb {c_pred:.1f} "
        f"m/s (rel err {rel_err:.3f})"
    )
