"""Independent continuum checks for anisotropic waves and welded interfaces.

These tests use explicit stiffnesses, continuum plane waves and impedance
coefficients, rather than another Deepwave propagator or parameter converter.
Refining both space and time checks convergence to those physical solutions.
The observation regions exclude boundary arrivals; these are not PML tests.
"""

import math

import pytest
import torch
from helpers import c_backend_available

import deepwave

DTYPE = torch.float64
COMPILED = pytest.mark.skipif(
    not c_backend_available("elastic_vti", device=torch.device("cpu")),
    reason="compiled elastic_vti backend unavailable",
)


@pytest.mark.filterwarnings("ignore::deepwave.ZeroWidthFaceWarning")
@pytest.mark.parametrize("backend", ["eager", pytest.param("compiled", marks=COMPILED)])
@pytest.mark.parametrize(
    ("mode", "theta", "azimuth"),
    [("SH", 0, 0), ("SH", 40, 35), ("SH", 90, 0), ("qP", 40, 35), ("qSV", 40, 35)],
)
def test_3d_christoffel_plane_wave_convergence(
    backend, mode, theta, azimuth, record_property
):
    """All three modes converge, including c55 != c66 and oblique azimuth.

    For SH, rho*omega**2 = c55*kz**2 + c66*(ky**2+kx**2).
    qP/qSV are the largest/smallest eigenvalues of the continuum Christoffel
    matrix for this material. Velocity and stress are sampled at their own
    spatial and temporal stagger locations. No discrete stencil is used to
    construct the oracle. Accuracy 2 and fixed dt/dx imply second order.
    """
    theta, azimuth = math.radians(theta), math.radians(azimuth)
    nz, ny, nx = (
        math.cos(theta),
        math.sin(theta) * math.sin(azimuth),
        math.sin(theta) * math.cos(azimuth),
    )
    rho, c11, c13, c33, c55, c66 = 2.0, 12.0, 3.0, 10.0, 2.0, 5.0
    c12 = c11 - 2 * c66
    christoffel = (
        torch.tensor(
            [
                [
                    c33 * nz * nz + c55 * (ny * ny + nx * nx),
                    (c13 + c55) * nz * ny,
                    (c13 + c55) * nz * nx,
                ],
                [
                    (c13 + c55) * nz * ny,
                    c55 * nz * nz + c11 * ny * ny + c66 * nx * nx,
                    (c12 + c66) * ny * nx,
                ],
                [
                    (c13 + c55) * nz * nx,
                    (c12 + c66) * ny * nx,
                    c55 * nz * nz + c66 * ny * ny + c11 * nx * nx,
                ],
            ],
            dtype=DTYPE,
        )
        / rho
    )
    if mode == "SH":
        polarisation = torch.tensor(
            [0, math.cos(azimuth), -math.sin(azimuth)], dtype=DTYPE
        )
        speed = math.sqrt((c55 * nz * nz + c66 * (ny * ny + nx * nx)) / rho)
    else:
        values, vectors = torch.linalg.eigh(christoffel)
        index = -1 if mode == "qP" else 0
        speed = float(values[index].sqrt())
        polarisation = vectors[:, index]
    k = torch.tensor([nz, ny, nx], dtype=DTYPE) * (2 * math.pi / 8)
    omega = speed * 2 * math.pi / 8
    strain = k * polarisation
    normal = (
        -torch.tensor([[c33, c13, c13], [c13, c11, c12], [c13, c12, c11]], dtype=DTYPE)
        @ strain
        / omega
    )
    errors = []
    for n in (32, 64):
        dx, dt, nt = 32 / n, 0.06 * 32 / n, n // 4
        coords = torch.meshgrid(*[torch.arange(n, dtype=DTYPE) * dx] * 3, indexing="ij")
        phase = sum(k[i] * coords[i] for i in range(3))
        one = torch.ones_like(phase)
        initial = {}
        for i, axis in enumerate("zyx"):
            initial[f"v{axis}_0"] = (
                polarisation[i] * torch.cos(phase + k[i] * dx / 2 + omega * dt / 2)
            ).unsqueeze(0)
            initial[f"sigma{axis}{axis}_0"] = (normal[i] * torch.cos(phase)).unsqueeze(
                0
            )
        for i, j, name, shear in (
            (0, 1, "yz", c55),
            (0, 2, "xz", c55),
            (1, 2, "xy", c66),
        ):
            amplitude = (
                -shear * (k[i] * polarisation[j] + k[j] * polarisation[i]) / omega
            )
            initial[f"sigma{name}_0"] = (
                amplitude * torch.cos(phase + (k[i] + k[j]) * dx / 2)
            ).unsqueeze(0)
        out = deepwave.elastic_vti(
            c11 * one,
            c13 * one,
            c33 * one,
            c55 * one,
            one / rho,
            dx,
            dt,
            c66=c66 * one,
            nt=nt,
            origin=[0, 0, 0],
            pml_width=0,
            pml_freq=0.1,
            max_vel=4.0,
            accuracy=2,
            backend=backend,
            **initial,
        )
        assert out.info.step_ratio == 1
        interior = (0, *[slice(n // 3, 2 * n // 3)] * 3)
        actual, expected = [], []
        for i, axis in enumerate("zyx"):
            actual.append(getattr(out, f"v{axis}_0")[interior])
            expected.append(
                (
                    polarisation[i]
                    * torch.cos(phase + k[i] * dx / 2 - omega * (nt * dt - dt / 2))
                ).unsqueeze(0)[interior]
            )
        actual, expected = torch.stack(actual), torch.stack(expected)
        assert torch.isfinite(actual).all()
        errors.append(
            float(
                torch.linalg.vector_norm(actual - expected)
                / torch.linalg.vector_norm(expected)
            )
        )
    record_property("continuum_relative_errors", errors)
    assert errors[1] < 0.01, errors
    assert 2.8 < errors[0] / errors[1] < 5.5, errors


@pytest.mark.filterwarnings("ignore::deepwave.ZeroWidthFaceWarning")
@pytest.mark.parametrize(
    "backend",
    [
        "eager",
        pytest.param(
            "compiled",
            marks=pytest.mark.skipif(
                not c_backend_available("acoustic_vti", device=torch.device("cpu")),
                reason="compiled acoustic_vti backend unavailable",
            ),
        ),
    ],
)
def test_pseudo_acoustic_vti_extra_mode_converges_to_continuum(
    backend, record_property
):
    """The additional VTI mode has the slower continuum Christoffel speed.

    With zero shear stiffness and ``c11*c33 > c13**2``, the 2-D Christoffel
    matrix has two positive eigenvalues at oblique incidence. This constructs
    the slower mode directly from its continuum eigenvector and stress law;
    no Deepwave converter or other propagator supplies the reference.
    """
    theta = math.radians(40)
    normal = torch.tensor([math.cos(theta), math.sin(theta)], dtype=DTYPE)
    rho, c11, c13, c33 = 2.0, 12.0, 3.0, 10.0
    ny, nx = normal.tolist()
    christoffel = (
        torch.tensor(
            [[c33 * ny * ny, c13 * ny * nx], [c13 * ny * nx, c11 * nx * nx]],
            dtype=DTYPE,
        )
        / rho
    )
    speeds_squared, polarisations = torch.linalg.eigh(christoffel)
    assert 0 < speeds_squared[0] < speeds_squared[1]
    speed = float(speeds_squared[0].sqrt())
    polarisation = polarisations[:, 0]
    k = normal * (2 * math.pi / 8)
    omega = speed * 2 * math.pi / 8
    stress_v = -(c33 * k[0] * polarisation[0] + c13 * k[1] * polarisation[1]) / omega
    stress_h = -(c13 * k[0] * polarisation[0] + c11 * k[1] * polarisation[1]) / omega

    errors = []
    for n in (32, 64):
        dx, dt, nt = 32 / n, 0.06 * 32 / n, n // 4
        y, x = torch.meshgrid(
            torch.arange(n, dtype=DTYPE) * dx,
            torch.arange(n, dtype=DTYPE) * dx,
            indexing="ij",
        )
        phase = k[0] * y + k[1] * x
        one = torch.ones_like(phase)
        out = deepwave.acoustic_vti(
            c11 * one,
            c13 * one,
            c33 * one,
            one / rho,
            dx,
            dt,
            nt=nt,
            origin=[0, 0],
            pml_width=0,
            pml_freq=0.1,
            max_vel=4.0,
            accuracy=2,
            backend=backend,
            vy_0=(polarisation[0] * torch.cos(phase + k[0] * dx / 2 + omega * dt / 2))[
                None
            ],
            vx_0=(polarisation[1] * torch.cos(phase + k[1] * dx / 2 + omega * dt / 2))[
                None
            ],
            sigma_v_0=(stress_v * torch.cos(phase))[None],
            sigma_h_0=(stress_h * torch.cos(phase))[None],
        )
        assert out.info.step_ratio == 1
        interior = (0, slice(n // 3, 2 * n // 3), slice(n // 3, 2 * n // 3))
        actual = torch.stack([out.vy_0[interior], out.vx_0[interior]])
        expected = torch.stack(
            [
                (
                    polarisation[i]
                    * torch.cos(phase + k[i] * dx / 2 - omega * (nt * dt - dt / 2))
                )[interior[1:]]
                for i in range(2)
            ]
        )
        assert torch.isfinite(actual).all()
        errors.append(
            float(
                torch.linalg.vector_norm(actual - expected)
                / torch.linalg.vector_norm(expected)
            )
        )
    record_property("continuum_relative_errors", errors)
    assert errors[1] < 0.01, errors
    assert 2.8 < errors[0] / errors[1] < 5.5, errors


@COMPILED
@pytest.mark.filterwarnings("ignore::deepwave.ZeroWidthFaceWarning")
@pytest.mark.parametrize("mode", ["P", "SV"])
def test_vti_welded_interface_reflection_transmission_convergence(
    mode, record_property
):
    """Normal P/SV waveforms converge to continuum welded-interface R/T.

    Continuity of traction and velocity gives R_v=(Z1-Z2)/(Z1+Z2),
    T_v=2*Z1/(Z1+Z2). For propagation along the VTI axis, Z is sqrt(rho*c33)
    for P and sqrt(rho*c55) for SV. Both media are genuinely anisotropic.
    A travelling Gaussian initial state avoids any source-normalisation
    oracle. The complete reflected/transmitted profiles, including signs,
    amplitudes and arrival positions, are compared away from the interface.
    """
    c1, c2 = (2.0, 3.0) if mode == "P" else (1.0, 1.5)
    z1, z2 = 2 * c1, 3 * c2
    errors = []
    for dx in (0.5, 0.25):
        ny, nx, dt = round(60 / dx), round(200 / dx), 0.08 * dx
        nt = round(18 / c1 / dt)
        y = (torch.arange(ny, dtype=DTYPE) + 0.5) * dx
        lower = y[:, None] >= 30

        def material(upper, lower_value):
            return (
                torch.where(lower, lower_value, upper)
                .to(DTYPE)
                .expand(ny, nx)
                .contiguous()
            )

        def pulse(position, time):
            return torch.exp(-0.5 * ((position - 20 - c1 * time) / 1.5) ** 2)

        def plane(profile):
            return profile[:, None].expand(ny, nx).contiguous().unsqueeze(0)

        offset = dx / 2 if mode == "P" else 0
        velocity = plane(pulse(y + offset, -dt / 2))
        if mode == "P":
            stress = plane(-z1 * pulse(y, 0))
            initial = {
                "vy_0": velocity,
                "sigmayy_0": stress,
                "sigmaxx_0": stress * 3 / 8,
            }
        else:
            initial = {"vx_0": velocity, "sigmaxy_0": plane(-z1 * pulse(y + dx / 2, 0))}
        out = deepwave.elastic_vti(
            material(10.0, 35.0),
            material(3.0, 8.0),
            material(8.0, 27.0),
            material(2.0, 6.75),
            material(0.5, 1 / 3),
            dx,
            dt,
            nt=nt,
            origin=[0, 0],
            accuracy=4,
            pml_width=0,
            pml_freq=0.1,
            max_vel=5.0,
            backend="compiled",
            **initial,
        )
        assert out.info.step_ratio == 1
        yy, time = y + offset, nt * dt - dt / 2
        expected = torch.where(
            yy < 30,
            (z1 - z2) / (z1 + z2) * pulse(60 - yy, time),
            2 * z1 / (z1 + z2) * pulse(30 + c1 / c2 * (yy - 30), time),
        )
        actual = (out.vy_0 if mode == "P" else out.vx_0)[0, :, nx // 2]
        # Both pulses fit in this window; boundary contamination is excluded.
        window = (yy > 14) & (yy < 50)
        assert torch.isfinite(actual).all()
        errors.append(
            float(
                torch.linalg.vector_norm((actual - expected)[window])
                / torch.linalg.vector_norm(expected[window])
            )
        )
    record_property("continuum_relative_errors", errors)
    assert errors[1] < 0.005, errors
    assert errors[0] / errors[1] > 2.5, errors
