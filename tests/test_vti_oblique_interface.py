"""Continuum oblique welded-VTI interface reference, independent of Deepwave.

The mode solver uses the Christoffel determinant in vertical slowness and
matches tangential/normal velocity and traction. It is a reference for later
finite-grid convergence checks, not a test of a Deepwave backend by itself.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

import deepwave


@dataclass(frozen=True)
class Medium:
    """Density and 2-D VTI stiffnesses in one consistent unit system."""

    rho: float
    c11: float
    c13: float
    c33: float
    c55: float
    c66: float = 0.0


@dataclass(frozen=True)
class Mode:
    """One radiating or decaying plane-wave branch at fixed horizontal p."""

    q: complex
    velocity: np.ndarray
    traction: np.ndarray

    @property
    def state(self) -> np.ndarray:
        """Tangential/normal particle velocity followed by traction."""
        return np.r_[self.velocity, self.traction]

    @property
    def downward_flux(self) -> float:
        """Time-average normal energy flux, positive into increasing z."""
        # e^{-i omega t}: traction = -C:(n tensor velocity), so -v*t is
        # positive for a downward travelling wave. The factor 1/2 cancels.
        return float(-np.vdot(self.velocity, self.traction).real / 2)


def vertical_modes(m: Medium, p: float, *, direction: int) -> tuple[Mode, Mode]:
    """Return outgoing qP/qSV modes for horizontal slowness ``p``.

    ``direction=+1`` selects downward radiation/decay and ``-1`` upward.
    For propagating modes the sign of normal energy flux chooses the root;
    for evanescent modes the sign of Im(q) chooses spatial decay.
    """
    c11, c13, c33, c55, rho = m.c11, m.c13, m.c33, m.c55, m.rho
    a = c55 * c33
    b = (c11 * p * p - rho) * c33 + (c55 * p * p - rho) * c55
    b -= (c13 + c55) ** 2 * p * p
    c = (c11 * p * p - rho) * (c55 * p * p - rho)
    q2 = sorted(np.roots([a, b, c]), key=lambda z: z.real)
    modes = []
    for squared in q2:  # qP has the smaller |q| near vertical incidence.
        q = complex(np.lib.scimath.sqrt(squared)) * direction
        g = np.array(
            [
                [c11 * p * p + c55 * q * q - rho, (c13 + c55) * p * q],
                [(c13 + c55) * p * q, c55 * p * p + c33 * q * q - rho],
            ],
            dtype=complex,
        )
        _, _, vh = np.linalg.svd(g)
        velocity = vh[-1].conj()
        traction = -np.array(
            [
                c55 * (q * velocity[0] + p * velocity[1]),
                c13 * p * velocity[0] + c33 * q * velocity[1],
            ]
        )
        mode = Mode(q, velocity, traction)
        if abs(q.imag) < 1e-12:
            assert direction * mode.downward_flux > 0
        else:
            assert direction * q.imag > 0
            assert abs(mode.downward_flux) < 1e-10
        modes.append(mode)
    return modes[0], modes[1]


def welded_coefficients(
    upper: Medium, lower: Medium, p: float, incident: int = 0
) -> tuple[Mode, tuple[Mode, Mode], tuple[Mode, Mode], np.ndarray]:
    """Solve four velocity/traction matching equations at z=0.

    The returned amplitudes are two reflected then two transmitted velocity
    polarization amplitudes. No backend quantities, stencils or impedance
    approximations enter the calculation.
    """
    incoming = vertical_modes(upper, p, direction=1)[incident]
    reflected = vertical_modes(upper, p, direction=-1)
    transmitted = vertical_modes(lower, p, direction=1)
    matrix = np.column_stack(
        [m.state for m in reflected] + [-m.state for m in transmitted]
    )
    coefficients = np.linalg.solve(matrix, -incoming.state)
    return incoming, reflected, transmitted, coefficients


UPPER = Medium(rho=2.0, c11=12.0, c13=3.0, c33=10.0, c55=2.0)
LOWER = Medium(rho=2.5, c11=17.0, c13=4.0, c33=14.0, c55=3.0)


@pytest.mark.parametrize("angle", [0.0, 12.0, 25.0])
def test_oblique_vti_matching_conserves_normal_energy(angle: float) -> None:
    """P incidence generates shear and obeys lossless normal-flux balance."""
    p = np.sin(np.deg2rad(angle)) / np.sqrt(UPPER.c33 / UPPER.rho)
    incoming, reflected, transmitted, amplitudes = welded_coefficients(UPPER, LOWER, p)
    residual = (
        incoming.state
        + sum(
            (a * m.state for a, m in zip(amplitudes[:2], reflected)),
            np.zeros(4, dtype=complex),
        )
        - sum(
            (a * m.state for a, m in zip(amplitudes[2:], transmitted)),
            np.zeros(4, dtype=complex),
        )
    )
    assert np.linalg.norm(residual) < 1e-12
    reflected_energy = sum(
        abs(a) ** 2 * -m.downward_flux for a, m in zip(amplitudes[:2], reflected)
    )
    transmitted_energy = sum(
        abs(a) ** 2 * m.downward_flux for a, m in zip(amplitudes[2:], transmitted)
    )
    assert (reflected_energy + transmitted_energy) / incoming.downward_flux == (
        pytest.approx(1.0, abs=1e-12)
    )
    if angle:
        assert abs(amplitudes[3]) > 1e-3  # converted transmitted qSV


def test_normal_incidence_reduces_to_impedance_formula() -> None:
    """The matching matrix reproduces the signed velocity coefficient."""
    incoming, reflected, transmitted, amplitudes = welded_coefficients(
        UPPER, LOWER, 0.0
    )
    z1 = np.sqrt(UPPER.rho * UPPER.c33)
    z2 = np.sqrt(LOWER.rho * LOWER.c33)
    reflected_vz = amplitudes[0] * reflected[0].velocity[1]
    transmitted_vz = amplitudes[2] * transmitted[0].velocity[1]
    assert reflected_vz / incoming.velocity[1] == pytest.approx(
        (z1 - z2) / (z1 + z2), abs=1e-12
    )
    assert transmitted_vz / incoming.velocity[1] == pytest.approx(
        2 * z1 / (z1 + z2), abs=1e-12
    )
    assert abs(amplitudes[1]) < 1e-12
    assert abs(amplitudes[3]) < 1e-12


def test_supercritical_mode_decays_away_from_interface() -> None:
    """Choose an evanescent P branch by radiation, with zero normal flux."""
    p = 0.52  # lower qP is supercritical, while an upper incident qSV exists
    incoming, _reflected, transmitted, amplitudes = welded_coefficients(
        UPPER, LOWER, p, incident=1
    )
    assert incoming.downward_flux > 0
    assert transmitted[0].q.imag > 0
    assert transmitted[0].downward_flux == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(amplitudes).all()


def _harmonic_field(
    component: str,
    y: np.ndarray,
    x: np.ndarray,
    time: float,
    omega: float,
    p: float,
    interface: float,
    matching: tuple[Mode, tuple[Mode, Mode], tuple[Mode, Mode], np.ndarray],
) -> np.ndarray:
    """Evaluate the piecewise continuum plane-wave solution on staggered nodes."""
    incoming, reflected, transmitted, amplitudes = matching
    value = np.zeros(np.broadcast_shapes(y.shape, x.shape), dtype=complex)
    for medium, modes, strengths, region in (
        (UPPER, (incoming, *reflected), (1, *amplitudes[:2]), y < interface),
        (LOWER, transmitted, amplitudes[2:], y >= interface),
    ):
        for mode, strength in zip(modes, strengths):
            vx, vy = mode.velocity
            if component == "vx":
                factor = vx
            elif component == "vy":
                factor = vy
            elif component == "sigmaxx":
                factor = -(medium.c11 * p * vx + medium.c13 * mode.q * vy)
            elif component == "sigmayy":
                factor = -(medium.c13 * p * vx + medium.c33 * mode.q * vy)
            else:
                factor = -medium.c55 * (mode.q * vx + p * vy)
            phase = omega * (p * x + mode.q * (y - interface) - time)
            value += np.where(region, strength * factor * np.exp(1j * phase), 0)
    return value.real


@pytest.mark.filterwarnings("ignore::deepwave.ZeroWidthFaceWarning")
@pytest.mark.parametrize("backend", ["eager", "compiled"])
@pytest.mark.parametrize("incident", [0, 1], ids=["oblique_qP", "evanescent_qSV"])
def test_oblique_vti_interface_waveform_converges(
    backend: str, incident: int, record_property
) -> None:
    """Propagated oblique and near-critical waves match continuum fields."""
    p = (
        np.sin(np.deg2rad(20.0)) / np.sqrt(UPPER.c33 / UPPER.rho)
        if incident == 0
        else 0.52
    )
    matching = welded_coefficients(UPPER, LOWER, p, incident=incident)
    if incident == 1:
        assert matching[2][0].q.imag > 0  # transmitted qP is evanescent
    omega = 2 * np.pi * np.sqrt(UPPER.c33 / UPPER.rho) / 10
    errors = []
    for n in (64, 128):
        dx = 32 / n
        dt = 0.05 * dx
        nt = round(1.0 / dt)
        iy = (np.arange(n) + 0.5)[:, None] * dx
        ix = (np.arange(n) + 0.5)[None, :] * dx
        lower = iy >= 16.0

        def material(up: float, down: float) -> torch.Tensor:
            return torch.from_numpy(
                np.broadcast_to(np.where(lower, down, up), (n, n)).copy()
            )

        def initial(
            component: str, yoff: float, xoff: float, time: float
        ) -> torch.Tensor:
            return torch.from_numpy(
                _harmonic_field(
                    component,
                    iy + yoff,
                    ix + xoff,
                    time,
                    omega,
                    p,
                    16.0,
                    matching,
                )
            ).unsqueeze(0)

        out = deepwave.elastic_vti(
            material(UPPER.c11, LOWER.c11),
            material(UPPER.c13, LOWER.c13),
            material(UPPER.c33, LOWER.c33),
            material(UPPER.c55, LOWER.c55),
            material(1 / UPPER.rho, 1 / LOWER.rho),
            dx,
            dt,
            nt=nt,
            origin=[0, 0],
            pml_width=0,
            pml_freq=0.1,
            max_vel=4.0,
            accuracy=2,
            backend=backend,
            vy_0=initial("vy", dx / 2, 0, -dt / 2),
            vx_0=initial("vx", 0, dx / 2, -dt / 2),
            sigmayy_0=initial("sigmayy", 0, 0, 0),
            sigmaxx_0=initial("sigmaxx", 0, 0, 0),
            sigmaxy_0=initial("sigmaxy", dx / 2, dx / 2, 0),
        )
        assert out.info.step_ratio == 1
        interior = (0, slice(n // 4, 3 * n // 4), slice(n // 4, 3 * n // 4))
        actual = out.vy_0[interior]
        expected = initial("vy", dx / 2, 0, nt * dt - dt / 2)[interior]
        assert torch.isfinite(actual).all()
        errors.append(
            float(
                torch.linalg.vector_norm(actual - expected)
                / torch.linalg.vector_norm(expected)
            )
        )
    record_property("continuum_relative_errors", errors)
    if incident == 0:
        assert errors[1] < errors[0] / 2.8, errors
        assert errors[1] < 0.005, errors
    else:
        assert errors[1] < errors[0], errors
        assert errors[1] < 0.05, errors


def _normal_modes(
    m: Medium, normal: tuple[float, float], direction: int
) -> tuple[Mode, Mode]:
    """P/SV modes at normal incidence to an arbitrarily dipping contact."""
    ny, nx = normal
    christoffel = np.array(
        [
            [m.c11 * nx * nx + m.c55 * ny * ny, (m.c13 + m.c55) * nx * ny],
            [(m.c13 + m.c55) * nx * ny, m.c55 * nx * nx + m.c33 * ny * ny],
        ]
    )
    values, vectors = np.linalg.eigh(christoffel)
    modes = []
    for index in (1, 0):  # qP then qSV
        q = direction * np.sqrt(m.rho / values[index])
        velocity = vectors[:, index].astype(complex)
        vx, vy = velocity
        kx, ky = q * nx, q * ny
        sxx = -(m.c11 * kx * vx + m.c13 * ky * vy)
        syy = -(m.c13 * kx * vx + m.c33 * ky * vy)
        sxy = -m.c55 * (ky * vx + kx * vy)
        traction = np.array([sxx * nx + sxy * ny, sxy * nx + syy * ny])
        mode = Mode(complex(q), velocity, traction)
        assert direction * mode.downward_flux > 0
        modes.append(mode)
    return modes[0], modes[1]


def _dipping_matching(
    normal: tuple[float, float],
) -> tuple[Mode, tuple[Mode, Mode], tuple[Mode, Mode], np.ndarray]:
    """Match velocity and normal traction for a dipping VTI interface."""
    incoming = _normal_modes(UPPER, normal, 1)[0]
    reflected = _normal_modes(UPPER, normal, -1)
    transmitted = _normal_modes(LOWER, normal, 1)
    matrix = np.column_stack(
        [m.state for m in reflected] + [-m.state for m in transmitted]
    )
    amplitudes = np.linalg.solve(matrix, -incoming.state)
    return incoming, reflected, transmitted, amplitudes


def _dipping_field(
    component: str,
    y: np.ndarray,
    x: np.ndarray,
    time: float,
    omega: float,
    normal: tuple[float, float],
    matching: tuple[Mode, tuple[Mode, Mode], tuple[Mode, Mode], np.ndarray],
) -> np.ndarray:
    """Sample the exact piecewise solution at physical staggered positions."""
    ny, nx = normal
    signed_distance = ny * (y - 16) + nx * (x - 16)
    incoming, reflected, transmitted, amplitudes = matching
    value = np.zeros(np.broadcast_shapes(y.shape, x.shape), dtype=complex)
    for medium, modes, strengths, region in (
        (UPPER, (incoming, *reflected), (1, *amplitudes[:2]), signed_distance < 0),
        (LOWER, transmitted, amplitudes[2:], signed_distance >= 0),
    ):
        for mode, strength in zip(modes, strengths):
            vx, vy = mode.velocity
            kx, ky = mode.q * nx, mode.q * ny
            if component == "vx":
                factor = vx
            elif component == "vy":
                factor = vy
            elif component == "sigmaxx":
                factor = -(medium.c11 * kx * vx + medium.c13 * ky * vy)
            elif component == "sigmayy":
                factor = -(medium.c13 * kx * vx + medium.c33 * ky * vy)
            else:
                factor = -medium.c55 * (ky * vx + kx * vy)
            phase = omega * (mode.q * signed_distance - time)
            value += np.where(region, strength * factor * np.exp(1j * phase), 0)
    return value.real


@pytest.mark.filterwarnings("ignore::deepwave.ZeroWidthFaceWarning")
@pytest.mark.parametrize("backend", ["eager", "compiled"])
def test_dipping_vti_interface_staircase_converges(
    backend: str, record_property
) -> None:
    """A 15° dipping contact converges despite cell-centre staircasing."""
    tilt = np.deg2rad(15.0)
    normal = (float(np.cos(tilt)), float(np.sin(tilt)))
    matching = _dipping_matching(normal)
    omega = 2 * np.pi * np.sqrt(UPPER.c33 / UPPER.rho) / 10
    errors = []
    for n in (32, 64, 128):
        dx = 32 / n
        dt = 0.05 * dx
        nt = round(1.0 / dt)
        iy = (np.arange(n) + 0.5)[:, None] * dx
        ix = (np.arange(n) + 0.5)[None, :] * dx
        signed_distance = normal[0] * (iy - 16) + normal[1] * (ix - 16)
        lower = signed_distance >= 0

        def material(up: float, down: float) -> torch.Tensor:
            return torch.from_numpy(np.where(lower, down, up).copy())

        def initial(
            component: str, yoff: float, xoff: float, time: float
        ) -> torch.Tensor:
            return torch.from_numpy(
                _dipping_field(
                    component,
                    iy + yoff,
                    ix + xoff,
                    time,
                    omega,
                    normal,
                    matching,
                )
            ).unsqueeze(0)

        out = deepwave.elastic_vti(
            material(UPPER.c11, LOWER.c11),
            material(UPPER.c13, LOWER.c13),
            material(UPPER.c33, LOWER.c33),
            material(UPPER.c55, LOWER.c55),
            material(1 / UPPER.rho, 1 / LOWER.rho),
            dx,
            dt,
            nt=nt,
            origin=[0, 0],
            pml_width=0,
            pml_freq=0.1,
            max_vel=4.0,
            accuracy=2,
            backend=backend,
            vy_0=initial("vy", dx / 2, 0, -dt / 2),
            vx_0=initial("vx", 0, dx / 2, -dt / 2),
            sigmayy_0=initial("sigmayy", 0, 0, 0),
            sigmaxx_0=initial("sigmaxx", 0, 0, 0),
            sigmaxy_0=initial("sigmaxy", dx / 2, dx / 2, 0),
        )
        interior = (0, slice(n // 4, 3 * n // 4), slice(n // 4, 3 * n // 4))
        actual = out.vy_0[interior]
        expected = initial("vy", dx / 2, 0, nt * dt - dt / 2)[interior]
        errors.append(
            float(
                torch.linalg.vector_norm(actual - expected)
                / torch.linalg.vector_norm(expected)
            )
        )
    record_property("continuum_relative_errors", errors)
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[2] < errors[1] / 2.0, errors
    assert errors[2] < 0.05, errors


def _stress_3d(
    m: Medium, normal: np.ndarray, q: complex, velocity: np.ndarray
) -> np.ndarray:
    """Return the symmetric stress tensor for a unit velocity plane mode."""
    kz, ky, kx = q * normal
    vz, vy, vx = velocity
    c12 = m.c11 - 2 * m.c66
    szz = -(m.c33 * kz * vz + m.c13 * (ky * vy + kx * vx))
    syy = -(m.c13 * kz * vz + m.c11 * ky * vy + c12 * kx * vx)
    sxx = -(m.c13 * kz * vz + c12 * ky * vy + m.c11 * kx * vx)
    szy = -m.c55 * (kz * vy + ky * vz)
    szx = -m.c55 * (kz * vx + kx * vz)
    syx = -m.c66 * (ky * vx + kx * vy)
    return np.array([[szz, szy, szx], [szy, syy, syx], [szx, syx, sxx]])


def _normal_modes_3d(m: Medium, normal: np.ndarray, direction: int) -> tuple[Mode, ...]:
    """Solve all three 3-D VTI Christoffel modes along a contact normal."""
    nz, ny, nx = normal
    c12 = m.c11 - 2 * m.c66
    gamma = np.array(
        [
            [
                m.c33 * nz * nz + m.c55 * (ny * ny + nx * nx),
                (m.c13 + m.c55) * nz * ny,
                (m.c13 + m.c55) * nz * nx,
            ],
            [
                (m.c13 + m.c55) * nz * ny,
                m.c55 * nz * nz + m.c11 * ny * ny + m.c66 * nx * nx,
                (c12 + m.c66) * ny * nx,
            ],
            [
                (m.c13 + m.c55) * nz * nx,
                (c12 + m.c66) * ny * nx,
                m.c55 * nz * nz + m.c66 * ny * ny + m.c11 * nx * nx,
            ],
        ]
    )
    values, vectors = np.linalg.eigh(gamma)
    modes = []
    for index in (2, 1, 0):
        q = complex(direction * np.sqrt(m.rho / values[index]))
        velocity = vectors[:, index].astype(complex)
        traction = _stress_3d(m, normal, q, velocity) @ normal
        mode = Mode(q, velocity, traction)
        assert direction * mode.downward_flux > 0
        modes.append(mode)
    return tuple(modes)


UPPER_3D = Medium(2.0, 12.0, 3.0, 10.0, 2.0, 5.0)
LOWER_3D = Medium(2.5, 17.0, 4.0, 14.0, 3.0, 6.0)


def _matching_3d(
    normal: np.ndarray, incident_index: int = 0
) -> tuple[Mode, tuple[Mode, ...], tuple[Mode, ...], np.ndarray]:
    """Match all velocity and traction components at a 3-D VTI contact."""
    incoming = _normal_modes_3d(UPPER_3D, normal, 1)[incident_index]
    reflected = _normal_modes_3d(UPPER_3D, normal, -1)
    transmitted = _normal_modes_3d(LOWER_3D, normal, 1)
    matrix = np.column_stack(
        [m.state for m in reflected] + [-m.state for m in transmitted]
    )
    amplitudes = np.linalg.solve(matrix, -incoming.state)
    return incoming, reflected, transmitted, amplitudes


def _field_3d(
    component: str,
    z: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    time: float,
    omega: float,
    normal: np.ndarray,
    matching: tuple[Mode, tuple[Mode, ...], tuple[Mode, ...], np.ndarray],
) -> np.ndarray:
    """Evaluate a 3-D piecewise plane wave at its staggered grid positions."""
    signed_distance = sum(n * (coord - 16) for n, coord in zip(normal, (z, y, x)))
    incoming, reflected, transmitted, amplitudes = matching
    value = np.zeros(np.broadcast_shapes(z.shape, y.shape, x.shape), dtype=complex)
    names = {
        "sigmazz": (0, 0),
        "sigmayy": (1, 1),
        "sigmaxx": (2, 2),
        "sigmayz": (0, 1),
        "sigmaxz": (0, 2),
        "sigmaxy": (1, 2),
    }
    for medium, modes, strengths, region in (
        (UPPER_3D, (incoming, *reflected), (1, *amplitudes[:3]), signed_distance < 0),
        (LOWER_3D, transmitted, amplitudes[3:], signed_distance >= 0),
    ):
        for mode, strength in zip(modes, strengths):
            if component in ("vz", "vy", "vx"):
                factor = mode.velocity["zyx".index(component[-1])]
            else:
                i, j = names[component]
                factor = _stress_3d(medium, normal, mode.q, mode.velocity)[i, j]
            phase = omega * (mode.q * signed_distance - time)
            value += np.where(region, strength * factor * np.exp(1j * phase), 0)
    return value.real


@pytest.mark.parametrize("incident_index", [0, 1, 2], ids=["qP", "qSV", "SH"])
def test_3d_vti_interface_oracle_conserves_all_modal_flux(incident_index: int) -> None:
    """The six 3-D velocity/traction conditions conserve normal energy."""
    tilt, azimuth = np.deg2rad(20.0), np.deg2rad(35.0)
    normal = np.array(
        [np.cos(tilt), np.sin(tilt) * np.sin(azimuth), np.sin(tilt) * np.cos(azimuth)]
    )
    incoming, reflected, transmitted, amplitudes = _matching_3d(normal, incident_index)
    residual = (
        incoming.state
        + sum(
            (a * m.state for a, m in zip(amplitudes[:3], reflected)),
            np.zeros(6, dtype=complex),
        )
        - sum(
            (a * m.state for a, m in zip(amplitudes[3:], transmitted)),
            np.zeros(6, dtype=complex),
        )
    )
    assert np.linalg.norm(residual) < 1e-12
    outgoing_flux = sum(
        -(abs(a) ** 2) * m.downward_flux for a, m in zip(amplitudes[:3], reflected)
    ) + sum(abs(a) ** 2 * m.downward_flux for a, m in zip(amplitudes[3:], transmitted))
    assert outgoing_flux / incoming.downward_flux == pytest.approx(1.0, abs=1e-12)
    assert max(abs(amplitudes[1]), abs(amplitudes[2])) > 1e-3


@pytest.mark.filterwarnings("ignore::deepwave.ZeroWidthFaceWarning")
@pytest.mark.parametrize("backend", ["eager", "compiled"])
@pytest.mark.parametrize(
    ("incident_index", "component", "fine_limit"),
    [(0, "vz", 0.01), (1, "vz", 0.03), (2, "vy", 0.02)],
    ids=["qP", "qSV", "SH"],
)
def test_3d_dipping_vti_interface_waveform_converges(
    backend: str,
    incident_index: int,
    component: str,
    fine_limit: float,
    record_property,
) -> None:
    """All 3-D incident modes converge to the continuum welded solution.

    At fixed frequency, the slower modes have shorter wavelengths and more
    interface staircasing error. Eager errors at dx=0.5 were 0.57%, 2.24%
    and 1.38% for qP, qSV and SH, respectively; each decreases on all three
    grids. The per-mode bounds retain headroom without hiding nonconvergence.
    """
    tilt, azimuth = np.deg2rad(20.0), np.deg2rad(35.0)
    normal = np.array(
        [np.cos(tilt), np.sin(tilt) * np.sin(azimuth), np.sin(tilt) * np.cos(azimuth)]
    )
    matching = _matching_3d(normal, incident_index)
    omega = 2 * np.pi * np.sqrt(UPPER_3D.c33 / UPPER_3D.rho) / 10
    errors = []
    for n in (24, 32, 64):
        dx = 32 / n
        dt = 0.05 * dx
        nt = round(0.8 / dt)
        iz = (np.arange(n) + 0.5)[:, None, None] * dx
        iy = (np.arange(n) + 0.5)[None, :, None] * dx
        ix = (np.arange(n) + 0.5)[None, None, :] * dx
        signed_distance = sum(
            coeff * (coord - 16) for coeff, coord in zip(normal, (iz, iy, ix))
        )
        lower = signed_distance >= 0

        def material(up: float, down: float) -> torch.Tensor:
            return torch.from_numpy(np.where(lower, down, up).copy())

        def initial(
            component: str, zoff: float, yoff: float, xoff: float, time: float
        ) -> torch.Tensor:
            return torch.from_numpy(
                _field_3d(
                    component,
                    iz + zoff,
                    iy + yoff,
                    ix + xoff,
                    time,
                    omega,
                    normal,
                    matching,
                )
            ).unsqueeze(0)

        out = deepwave.elastic_vti(
            material(UPPER_3D.c11, LOWER_3D.c11),
            material(UPPER_3D.c13, LOWER_3D.c13),
            material(UPPER_3D.c33, LOWER_3D.c33),
            material(UPPER_3D.c55, LOWER_3D.c55),
            material(1 / UPPER_3D.rho, 1 / LOWER_3D.rho),
            dx,
            dt,
            c66=material(UPPER_3D.c66, LOWER_3D.c66),
            nt=nt,
            origin=[0, 0, 0],
            pml_width=0,
            pml_freq=0.1,
            max_vel=4.0,
            accuracy=2,
            backend=backend,
            vz_0=initial("vz", dx / 2, 0, 0, -dt / 2),
            vy_0=initial("vy", 0, dx / 2, 0, -dt / 2),
            vx_0=initial("vx", 0, 0, dx / 2, -dt / 2),
            sigmazz_0=initial("sigmazz", 0, 0, 0, 0),
            sigmayy_0=initial("sigmayy", 0, 0, 0, 0),
            sigmaxx_0=initial("sigmaxx", 0, 0, 0, 0),
            sigmayz_0=initial("sigmayz", dx / 2, dx / 2, 0, 0),
            sigmaxz_0=initial("sigmaxz", dx / 2, 0, dx / 2, 0),
            sigmaxy_0=initial("sigmaxy", 0, dx / 2, dx / 2, 0),
        )
        interior = (0, *[slice(n // 4, 3 * n // 4)] * 3)
        offset = (dx / 2, 0, 0) if component == "vz" else (0, dx / 2, 0)
        actual = getattr(out, f"{component}_0")[interior]
        expected = initial(component, *offset, nt * dt - dt / 2)[interior]
        errors.append(
            float(
                torch.linalg.vector_norm(actual - expected)
                / torch.linalg.vector_norm(expected)
            )
        )
    record_property("continuum_relative_errors", errors)
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[2] < errors[1] / 2.0, errors
    assert errors[2] < fine_limit, errors
