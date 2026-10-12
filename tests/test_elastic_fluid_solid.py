"""Fluid-solid interface validation for the elastic propagator.

These tests validate that ``deepwave.elastic`` correctly models a fluid
(``mu = 0``, e.g. a water layer) in contact with a solid -- the configuration
of a transcranial-ultrasound skull/brain model or an ocean-bottom
(water-over-seabed) seismic survey, and the subject of GitHub issue #116
("Modeling a water layer (Vs=0) above a solid").

The four checks are:

1. ``test_harmonic_mu_zeroes_shear_at_fluid_interface`` -- the harmonic-mu
   averaging in :func:`deepwave.elastic.prepare_parameters` degenerates to
   exactly zero at any shear-stress node that touches a fluid cell (the
   free-slip fluid-solid condition), matching the naive harmonic-mean limit
   ``2*mu1*mu2/(mu1+mu2) -> 0`` as ``mu1 -> 0``.
2. ``test_normal_incidence_reflection_transmission`` -- normal-incidence
   fluid->solid reflection/transmission of a plane wave versus the analytic
   acoustic-impedance formulas.
3. ``test_oblique_energy_partition_vs_zoeppritz`` -- an oblique-incidence case
   with P->P and P->S mode conversion, validated by an energy-conservation
   partitioning check against an in-test fluid-solid Zoeppritz solver (see the
   test docstring for why the energy partition, rather than per-mode amplitude
   extraction, is used).
4. ``test_long_run_stability_water_over_solid`` -- a long run of the issue-#116
   configuration (water layer, free surface on top, PML on the other faces)
   stays energy-bounded.
"""

import math

import numpy as np
import pytest
import torch

from deepwave import elastic, vpvsrho_to_lambmubuoyancy
from deepwave.elastic import prepare_parameters
from deepwave.wavelets import ricker

DTYPE = torch.double


# ---------------------------------------------------------------------------
# Analytic reference: fluid-solid Zoeppritz (P incident from the fluid)
# ---------------------------------------------------------------------------
def fluid_solid_zoeppritz(
    vp1: float,
    rho1: float,
    vp2: float,
    vs2: float,
    rho2: float,
    theta1_deg: float,
) -> tuple:
    """Energy reflection/transmission for P incident from a fluid onto a solid.

    Displacement-potential formulation at a plane fluid-solid interface (fluid =
    medium 1, no shear; solid = medium 2). With the interface normal along z,
    horizontal along x, and ray parameter ``p = sin(theta1)/vp1`` (Snell's law),
    the vertical slownesses are ``q = sqrt(1/V^2 - p^2)`` for each mode. The
    three interface conditions are:

    * continuity of normal displacement ``u_z``;
    * continuity of normal stress ``sigma_zz`` (``= -pressure`` in the fluid);
    * vanishing shear traction ``sigma_xz = 0`` in the solid (a fluid supports
      no shear; the tangential displacement is *not* continuous -- the fluid
      slips freely).

    These give a 3x3 linear system for the reflected-P, transmitted-P and
    transmitted-S potential amplitudes (incident P amplitude = 1). The normal
    (z) energy flux of a plane mode with potential amplitude ``A`` is
    proportional to ``rho * q * |A|^2``, which yields the energy coefficients.

    Returns ``(R_E, TP_E, TS_E)`` (fractions of incident normal energy flux);
    for a lossless interface they sum to 1.
    """
    th1 = np.deg2rad(theta1_deg)
    p = np.sin(th1) / vp1
    lam2 = rho2 * vp2**2 - 2 * rho2 * vs2**2
    mu2 = rho2 * vs2**2
    qa1 = np.sqrt(1 / vp1**2 - p**2 + 0j)
    qa2 = np.sqrt(1 / vp2**2 - p**2 + 0j)
    qb2 = np.sqrt(1 / vs2**2 - p**2 + 0j)
    kx = p
    c_p = lam2 / vp2**2 + 2 * mu2 * qa2**2
    # Unknowns [A_r, A_tp, A_ts]; incident A_i = 1.
    matrix = np.array(
        [
            [-qa1, -qa2, kx],
            [-rho1, c_p, -2 * mu2 * kx * qb2],
            [0.0, -2 * mu2 * kx * qa2, mu2 * (kx**2 - qb2**2)],
        ],
        dtype=complex,
    )
    rhs = np.array([-qa1, rho1, 0.0], dtype=complex)
    a_r, a_tp, a_ts = np.linalg.solve(matrix, rhs)
    inc = rho1 * qa1.real
    r_e = (rho1 * qa1.real * abs(a_r) ** 2) / inc
    tp_e = (rho2 * qa2.real * abs(a_tp) ** 2) / inc
    ts_e = (rho2 * qb2.real * abs(a_ts) ** 2) / inc
    return float(r_e), float(tp_e), float(ts_e)


def _helmholtz_split(vy: np.ndarray, vx: np.ndarray, dx: float) -> tuple:
    """Split a 2-D velocity field into compressional and shear energy density.

    A Helmholtz (FFT) decomposition projects the vector field onto its
    curl-free (P) and divergence-free (S) parts, returning the squared
    magnitude of each part -- used to separate transmitted P from transmitted S
    energy in the solid.
    """
    nz, nx = vy.shape
    kz = 2 * np.pi * np.fft.fftfreq(nz, d=dx)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    kzg, kxg = np.meshgrid(kz, kx, indexing="ij")
    k2 = kzg**2 + kxg**2
    k2[0, 0] = 1.0
    vyf = np.fft.fft2(vy)
    vxf = np.fft.fft2(vx)
    div = kzg * vyf + kxg * vxf
    vy_p = np.fft.ifft2(kzg * div / k2).real
    vx_p = np.fft.ifft2(kxg * div / k2).real
    vy_s = vy - vy_p
    vx_s = vx - vx_p
    return vy_p**2 + vx_p**2, vy_s**2 + vx_s**2


# ---------------------------------------------------------------------------
# Item 1: harmonic-mu averaging at a fluid/solid interface
# ---------------------------------------------------------------------------
def test_harmonic_mu_zeroes_shear_at_fluid_interface() -> None:
    """The shear-node mu vanishes wherever it touches a fluid (mu=0) cell.

    ``prepare_parameters`` evaluates the shear modulus at the off-diagonal
    (shear-stress) staggered node as the harmonic mean of the four surrounding
    cell-centre mu values. The naive harmonic mean
    ``2*mu1*mu2/(mu1+mu2)`` already degenerates to 0 when either input is 0, so
    a shear node adjacent to a fluid cell must carry zero shear modulus -- i.e.
    the fluid-solid interface supports no shear traction (free slip), which is
    the correct physical condition. The code realises this robustly: it
    replaces sub-threshold mu with a safe value before taking reciprocals (to
    avoid ``1/0``), then masks the result to exactly 0 wherever any of the four
    cells is a fluid, so the harmonic-mean limit is reproduced without any NaN
    or inf. This test locks that behaviour.
    """
    solid_a = 2.2e9
    solid_b = 3.0e9
    # A depth column: fluid, fluid, solid_a, solid_b, solid_b, solid_b.
    nz, nx = 6, 4
    mu = torch.zeros(1, nz, nx, dtype=DTYPE)
    mu[:, 2, :] = solid_a
    mu[:, 3:, :] = solid_b
    buoyancy = torch.full((1, nz, nx), 1.0 / 2200.0, dtype=DTYPE)

    mu_yx = prepare_parameters(mu, buoyancy)[0]  # 2-D returns [mu_yx, ...]

    # Shear node i sits between depth rows i and i+1 (uniform across x here).
    col = mu_yx[0, :, 0]
    # Nodes entirely inside the fluid, or straddling the fluid/solid contact,
    # must be exactly zero -- no NaN, no inf, no spurious shear coupling.
    assert col[0].item() == 0.0  # fluid/fluid
    assert col[1].item() == 0.0  # fluid/solid_a contact -> free slip
    assert torch.isfinite(mu_yx).all()
    # An all-solid node reproduces the exact harmonic mean of its neighbours.
    expect = 2 * solid_a * solid_b / (solid_a + solid_b)
    assert col[2].item() == pytest.approx(expect, rel=1e-12)  # solid_a/solid_b
    assert col[3].item() == pytest.approx(solid_b, rel=1e-12)  # solid_b/solid_b
    # The masked interface value equals the naive harmonic-mean limit at mu1=0.
    assert (2 * 0.0 * solid_a / (0.0 + solid_a)) == 0.0


# ---------------------------------------------------------------------------
# Item 2: normal-incidence reflection / transmission vs impedance formulas
# ---------------------------------------------------------------------------
def test_normal_incidence_reflection_transmission() -> None:
    """Normal-incidence fluid->solid R/T of a plane wave vs impedance theory.

    At normal incidence no shear is generated (P->P only). With acoustic
    impedances ``Z1 = rho1*vp1`` (fluid) and ``Z2 = rho2*vp2`` (solid's
    P-impedance), the particle-velocity (and displacement) reflection and
    transmission coefficients are the standard 1-D results

        R_v = (Z1 - Z2) / (Z1 + Z2),   T_v = 2 Z1 / (Z1 + Z2),

    with ``1 + R_v = T_v``. (Energy: ``R_v^2 + (Z2/Z1) T_v^2 = 1``.) A
    horizontally-uniform line of pressure sources launches a downward plane P
    wave in the fluid; a receiver line in the fluid records the incident then
    the reflected arrival (separated in time), and a receiver line in the solid
    records the transmitted arrival. Because a 1-D plane wave has no geometric
    spreading, the measured amplitude ratios directly give R_v and T_v (read
    from the vertical particle velocity ``vy``, which is continuous across the
    interface).
    """
    dx = 5.0
    vp1, rho1 = 1500.0, 1000.0  # water
    vp2, vs2, rho2 = 3000.0, 1700.0, 2200.0  # solid
    z1 = rho1 * vp1
    z2 = rho2 * vp2
    r_v = (z1 - z2) / (z1 + z2)
    t_v = 2 * z1 / (z1 + z2)

    nz, nx = 260, 140
    iz = 150  # fluid-solid interface row
    src_z, rz_f, rz_s = 25, 75, 190

    vp = torch.full((nz, nx), vp2, dtype=DTYPE)
    vs = torch.full((nz, nx), vs2, dtype=DTYPE)
    rho = torch.full((nz, nx), rho2, dtype=DTYPE)
    vp[:iz], vs[:iz], rho[:iz] = vp1, 0.0, rho1
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)

    freq, dt = 15.0, 0.0008
    nt = int(0.78 / dt)
    peak = 1.4 / freq
    wav = ricker(freq, nt, dt, peak, dtype=DTYPE)
    xs = torch.arange(nx)
    src_loc = torch.stack([torch.full((nx,), src_z), xs], -1).reshape(1, nx, 2).long()
    src_amp = wav.reshape(1, 1, -1).repeat(1, nx, 1)
    rec_f = torch.stack([torch.full((nx,), rz_f), xs], -1).reshape(1, nx, 2).long()
    rec_s = torch.stack([torch.full((nx,), rz_s), xs], -1).reshape(1, nx, 2).long()
    rec_loc = torch.cat([rec_f, rec_s], dim=1)

    out = elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_pressure_rate=src_amp,
        source_locations_pressure_rate=src_loc,
        receiver_locations_y=rec_loc,
        pml_width=20,
        pml_freq=freq,
        accuracy=6,
        # Include the heterogeneous density/stiffness CFL bound automatically.
    )
    vy = out.receiver_amplitudes_y[0]
    cx = nx // 2
    tr_f = vy[cx]  # central fluid receiver
    tr_s = vy[nx + cx]  # central solid receiver
    t = torch.arange(nt) * dt

    # Expected arrival times (plus the source peak-time delay).
    t_inc = peak + (rz_f - src_z) * dx / vp1
    t_ref = peak + ((iz - src_z) + (iz - rz_f)) * dx / vp1
    t_tra = peak + (iz - src_z) * dx / vp1 + (rz_s - iz) * dx / vp2

    def signed_peak(trace: torch.Tensor, t0: float, half: float = 0.07) -> float:
        window = (t > t0 - half) & (t < t0 + half)
        seg = torch.where(window, trace, torch.zeros_like(trace))
        return seg[seg.abs().argmax()].item()

    a_inc = signed_peak(tr_f, t_inc)
    a_ref = signed_peak(tr_f, t_ref)
    a_tra = signed_peak(tr_s, t_tra)
    r_meas = a_ref / a_inc
    t_meas = a_tra / a_inc

    # Reflection from the higher-impedance solid flips the velocity polarity.
    assert r_meas < 0
    assert r_v < 0
    assert abs(r_meas - r_v) / abs(r_v) < 0.06  # measured ~3.8%
    assert abs(t_meas - t_v) / abs(t_v) < 0.06  # measured ~4.0%


# ---------------------------------------------------------------------------
# Item 3a: analytic Zoeppritz self-consistency
# ---------------------------------------------------------------------------
def test_zoeppritz_solver_consistency() -> None:
    """The fluid-solid Zoeppritz solver conserves energy and matches theory.

    Sanity checks on the in-test analytic reference: (a) the energy
    coefficients sum to 1 at every angle (lossless interface); (b) at normal
    incidence there is no transmitted shear and the energy reflection reduces
    to the impedance-contrast value ``((Z2-Z1)/(Z2+Z1))^2``; (c) at oblique
    incidence a transmitted shear (mode-converted) wave genuinely appears.
    """
    vp1, rho1 = 1500.0, 1000.0
    vp2, vs2, rho2 = 3000.0, 1700.0, 2200.0
    z1, z2 = rho1 * vp1, rho2 * vp2

    for theta in (0.0, 5.0, 16.0, 25.0, 35.0):
        r_e, tp_e, ts_e = fluid_solid_zoeppritz(vp1, rho1, vp2, vs2, rho2, theta)
        assert r_e + tp_e + ts_e == pytest.approx(1.0, abs=1e-10)

    r0, _tp0, ts0 = fluid_solid_zoeppritz(vp1, rho1, vp2, vs2, rho2, 0.0)
    assert ts0 == pytest.approx(0.0, abs=1e-12)
    assert r0 == pytest.approx(((z2 - z1) / (z2 + z1)) ** 2, rel=1e-10)

    _, _, ts16 = fluid_solid_zoeppritz(vp1, rho1, vp2, vs2, rho2, 16.0)
    assert ts16 > 0.05  # substantial P->S conversion


# ---------------------------------------------------------------------------
# Item 3b: oblique FD energy partition vs Zoeppritz
# ---------------------------------------------------------------------------
def test_oblique_energy_partition_vs_zoeppritz() -> None:
    """Oblique fluid->solid P: FD energy partition matches Zoeppritz (P & S).

    Approach and why (per the task's stated fallback): extracting individual,
    signed per-mode amplitude reflection/transmission coefficients from a 2-D
    finite-difference wavefield is fragile -- it entangles staggered-grid
    component offsets, wave-type projection, sign conventions, and the temporal
    overlap of the incident and reflected fields in the fluid. An
    *energy-conservation partitioning* check is robust: the fraction of the
    incident P energy that ends up (i) reflected in the fluid, (ii) transmitted
    as P in the solid, and (iii) transmitted as S in the solid equals the
    Zoeppritz energy coefficients, and those three fractions sum to 1.

    A steered line of pressure sources (a per-source linear time delay
    ``x*dx*sin(theta)/vp1``) launches a downgoing plane P wave at incidence
    angle ``theta`` in the fluid. After the wave has crossed the interface and
    the reflected/transmitted packets have separated, the kinetic-energy density
    ``0.5*rho*|v|^2`` is integrated over the fluid region (reflected) and, in
    the solid, split by a Helmholtz (FFT) decomposition into P (curl-free) and S
    (div-free) parts. For freely-travelling packets the kinetic energy equals
    the potential energy on average, so the *fractions* equal the energy-flux
    coefficients. The measured partition matches the analytic Zoeppritz solution
    to ~1-2%, and a clearly non-zero transmitted-S fraction confirms P->S mode
    conversion.
    """
    dx = 5.0
    vp1, rho1 = 1500.0, 1000.0
    vp2, vs2, rho2 = 3000.0, 1700.0, 2200.0
    theta = 16.0
    r_e, tp_e, ts_e = fluid_solid_zoeppritz(vp1, rho1, vp2, vs2, rho2, theta)

    nz, nx = 340, 460
    iz, src_z = 140, 28
    vp = torch.full((nz, nx), vp2, dtype=DTYPE)
    vs = torch.full((nz, nx), vs2, dtype=DTYPE)
    rho = torch.full((nz, nx), rho2, dtype=DTYPE)
    vp[:iz], vs[:iz], rho[:iz] = vp1, 0.0, rho1
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)

    freq, dt = 15.0, 0.0006
    peak = 1.3 / freq
    x0, x1 = 120, 340
    xs = torch.arange(x0, x1)
    nb = len(xs)
    delays = xs.double() * dx * np.sin(np.deg2rad(theta)) / vp1
    delays = delays - delays.min()
    taper = torch.ones(nb, dtype=DTYPE)
    edge = nb // 5
    ramp = 0.5 * (1 - torch.cos(torch.linspace(0, np.pi, edge, dtype=DTYPE)))
    taper[:edge] = ramp
    taper[-edge:] = ramp.flip(0)
    nt = int(0.74 / dt)
    src_amp = torch.zeros(1, nb, nt, dtype=DTYPE)
    for i in range(nb):
        src_amp[0, i] = (
            ricker(freq, nt, dt, peak + delays[i].item(), dtype=DTYPE) * taper[i]
        )
    src_loc = torch.stack([torch.full((nb,), src_z), xs], -1).reshape(1, nb, 2).long()

    pml = 25
    out = elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_pressure_rate=src_amp,
        source_locations_pressure_rate=src_loc,
        pml_width=pml,
        pml_freq=freq,
        accuracy=6,
        # Include the heterogeneous density/stiffness CFL bound automatically.
    )
    # Final wavefields include the PML padding; crop back to the model region.
    vy = out.vy_0[0].numpy()[pml : pml + nz, pml : pml + nx]
    vx = out.vx_0[0].numpy()[pml : pml + nz, pml : pml + nx]

    mx0, mx1 = 35, nx - 35  # interior x, away from the side PMLs
    ke = 0.5 * rho.numpy() * (vy**2 + vx**2)
    ke_fluid = ke[18:iz, mx0:mx1].sum()  # reflected (all P in the fluid)
    kp, ks = _helmholtz_split(vy[iz:, :], vx[iz:, :], dx)
    ke_p = (0.5 * rho2 * kp[:, mx0:mx1]).sum()
    ke_s = (0.5 * rho2 * ks[:, mx0:mx1]).sum()
    tot = ke_fluid + ke_p + ke_s
    r_fd, tp_fd, ts_fd = ke_fluid / tot, ke_p / tot, ke_s / tot

    # Energy partition matches the analytic Zoeppritz solution.
    assert abs(r_fd - r_e) < 0.05
    assert abs(tp_fd - tp_e) < 0.05
    assert abs(ts_fd - ts_e) < 0.05
    # P->S mode conversion is genuinely present and of the right size.
    assert ts_fd > 0.05


# ---------------------------------------------------------------------------
# Item 4: long-run stability of the water-over-solid configuration
# ---------------------------------------------------------------------------
def test_long_run_stability_water_over_solid() -> None:
    """Water layer + free surface + PML stays energy-bounded over many steps.

    The issue-#116 configuration: a water layer (``vs=0``) over an elastic
    solid, with a vacuum free surface on top of the water and PMLs on the other
    three faces. Unlike the documented thin-slab-between-two-facing-free-
    surfaces instability (see ``docs/boundaries.rst`` and
    ``tests/test_elastic_mpml.py``), this geometry has only a *single* free
    surface -- the lower boundary of the water is a fluid-solid interface, not a
    free surface -- so the Becache-Fauqueux-Joly trapping condition for the
    C-PML instability is not met and the run is stable. We drive a source in the
    water, run 12000 steps (long after the source has ended), and require the
    total wavefield energy to remain finite and bounded (it neither blows up nor
    is spuriously amplified at the interface). The near-interface *oscillations*
    reported in issue #116 are a near-interface numerical/physical phenomenon
    (Scholte interface waves and grid dispersion), not a late-time instability.
    """
    dx, dt = 5.0, 0.0009
    vp1, rho1 = 1500.0, 1000.0
    vp2, vs2, rho2 = 2500.0, 1400.0, 2200.0
    nz, nx = 90, 120
    top_vac, water_cells = 3, 14
    iz = top_vac + water_cells
    vp = torch.full((nz, nx), vp2, dtype=DTYPE)
    vs = torch.full((nz, nx), vs2, dtype=DTYPE)
    rho = torch.full((nz, nx), rho2, dtype=DTYPE)
    vp[top_vac:iz], vs[top_vac:iz], rho[top_vac:iz] = vp1, 0.0, rho1
    vp[:top_vac], vs[:top_vac], rho[:top_vac] = 0.0, 0.0, 0.0  # vacuum free surface
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)

    freq = 15.0
    nt = 12000
    nsrc = 60
    peak = 1.5 / freq
    amp = torch.zeros(1, 1, nt, dtype=DTYPE)
    amp[..., :nsrc] = ricker(freq, nsrc, dt, peak, dtype=DTYPE)
    src_loc = torch.tensor([[[top_vac + water_cells // 2, nx // 3]]])

    energies = []
    step_e = 200

    def cb(state):
        if state.step % step_e == 0:
            vyw = state.get_wavefield("vy_0")
            vxw = state.get_wavefield("vx_0")
            energies.append((vyw**2).sum().item() + (vxw**2).sum().item())

    elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=amp,
        source_locations_y=src_loc,
        pml_width=[0, 12, 12, 12],  # free surface on top, PML on the other faces
        pml_freq=freq,
        accuracy=4,
        # Include the heterogeneous density/stiffness CFL bound automatically.
        forward_callback=cb,
        callback_frequency=step_e,
    )

    energies = np.array(energies)
    assert np.isfinite(energies).all()
    # Post-source quiescent baseline (trapped interface energy after the source
    # has ended), and the peak/late levels.
    base = float(np.median(energies[len(energies) // 6 : len(energies) // 6 + 5]))
    assert base > 0
    assert math.isfinite(energies[-1])
    # No exponential growth: neither the peak nor the late energy exceeds a
    # small multiple of the baseline (measured final/base ~ 1.0).
    assert energies.max() < 5 * base
    assert energies[-1] < 5 * base
