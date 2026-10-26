"""Anisotropic and attenuating PML against a larger-domain boundary oracle.

The two runs use identical material, wavelet, grid, time step and relative
source/receiver geometry. The larger domain keeps reflected outer-boundary
waves outside the record; the small domain stresses the top/left corner and
a receiver along the top face. A zero-width PML control shows that agreement
is caused by absorption rather than an unexcited boundary.
"""

import pytest
import torch

import deepwave


@pytest.mark.filterwarnings("ignore::deepwave.ZeroWidthFaceWarning")
@pytest.mark.parametrize("backend", ["eager", "compiled"])
@pytest.mark.parametrize("family", ["elastic_vti", "viscoacoustic", "elastic"])
def test_pml_grazing_corner_and_late_record_matches_large_domain(
    family: str, backend: str, record_property
) -> None:
    """Physical 2-D PML error stays small while the no-PML control reflects."""
    dtype = torch.float64
    nt, dt, freq, shift = 300, 0.15, 0.15, 40
    wavelet = deepwave.wavelets.ricker(freq, nt, dt, 10.0).to(dtype)
    amplitude = wavelet.reshape(1, 1, -1)
    source = torch.tensor([[[6, 8]]])
    receivers = torch.tensor([[[6, 40], [22, 40], [24, 24]]])

    def run(n: int, offset: int, width: int) -> torch.Tensor:
        one = torch.ones((n, n), dtype=dtype)
        common = {
            "pml_width": width,
            "pml_freq": freq,
            "max_vel": 4.0,
            "accuracy": 2,
            "backend": backend,
        }
        if family == "elastic":
            output = deepwave.elastic(
                6 * one,
                2 * one,
                0.5 * one,
                1.0,
                dt,
                source_amplitudes_y=amplitude,
                source_locations_y=source + offset,
                receiver_locations_y=receivers + offset,
                **common,
            )
            return output.receiver_amplitudes_y[0]
        if family == "elastic_vti":
            output = deepwave.elastic_vti(
                12 * one,
                3 * one,
                10 * one,
                2 * one,
                0.5 * one,
                1.0,
                dt,
                source_amplitudes_y=amplitude,
                source_locations_y=source + offset,
                receiver_locations_y=receivers + offset,
                **common,
            )
            return output.receiver_amplitudes_y[0]
        output = deepwave.viscoacoustic(
            2 * one,
            2 * one,
            20 * one,
            1.0,
            dt,
            source_amplitudes_pressure_rate=amplitude,
            source_locations_pressure_rate=source + offset,
            receiver_locations_p=receivers + offset,
            q_n_mechanisms=3,
            q_band=(0.1, 0.3),
            **common,
        )
        return output.receiver_amplitudes_p[0]

    with torch.no_grad():
        reference = run(128, shift, 10)
        absorbed = run(48, 0, 10)
        reflecting = run(48, 0, 0)
    assert torch.isfinite(reference).all()
    assert float(reference.abs().max()) > 1e-3
    reference_norm = torch.linalg.vector_norm(reference)
    absorbed_error = float(
        torch.linalg.vector_norm(absorbed - reference) / reference_norm
    )
    reflecting_error = float(
        torch.linalg.vector_norm(reflecting - reference) / reference_norm
    )
    assert absorbed_error < 0.01
    assert reflecting_error > 0.2

    # The larger-domain run has its outer boundary farther from the source
    # and receivers, with its own PML. Compare late samples as well as the
    # full traces so a delayed small-domain reflection cannot hide in the norm.
    late = slice(130, None)
    peak = reference.abs().amax(dim=-1)
    assert bool((peak > 1e-3).all())
    absorbed_late = (absorbed[:, late] - reference[:, late]).abs().amax(dim=-1) / peak
    reflecting_late = (reflecting[:, late] - reference[:, late]).abs().amax(
        dim=-1
    ) / peak
    record_property("pml_relative_error", absorbed_error)
    record_property("no_pml_relative_error", reflecting_error)
    record_property("pml_late_peak_relative_error", float(absorbed_late.max()))
    record_property("no_pml_late_peak_relative_error", float(reflecting_late.max()))
    assert float(absorbed_late.max()) < 0.01
    assert float(reflecting_late.max()) > 0.1


@pytest.mark.filterwarnings("ignore::deepwave.ZeroWidthFaceWarning")
@pytest.mark.parametrize("backend", ["eager", "compiled"])
@pytest.mark.parametrize(
    ("family", "mpml_fraction"),
    [("elastic_vti", 0.0), ("elastic_vti", 0.15), ("viscoacoustic", 0.0)],
)
def test_asymmetric_3d_pml_corner_matches_large_domain(
    family, mpml_fraction, backend, record_property
):
    """Three faces, unequal spacing/widths and M-PML have a physical reference."""
    nt, dt, frequency = 180, 0.15, 0.2
    spacing = [1.0, 1.1, 1.3]
    amplitude = deepwave.wavelets.ricker(
        frequency, nt, dt, 7.5, dtype=torch.float64
    ).reshape(1, 1, -1)
    source = torch.tensor([[[6, 6, 8]]])
    receivers = torch.tensor([[[6, 24, 24], [20, 6, 24], [22, 22, 22]]])

    def run(n, offset, width):
        one = torch.ones((n,) * 3, dtype=torch.float64)
        common = {
            "pml_width": width,
            "pml_freq": frequency,
            "max_vel": 4.0,
            "accuracy": 2,
            "backend": backend,
        }
        if family == "elastic_vti":
            output = deepwave.elastic_vti(
                12 * one,
                3 * one,
                10 * one,
                2 * one,
                0.5 * one,
                spacing,
                dt,
                c66=5 * one,
                mpml_fraction=mpml_fraction,
                source_amplitudes_z=amplitude,
                source_locations_z=source + offset,
                receiver_locations_z=receivers + offset,
                **common,
            )
            return output.receiver_amplitudes_z[0]
        output = deepwave.viscoacoustic(
            2 * one,
            2 * one,
            20 * one,
            spacing,
            dt,
            source_amplitudes_pressure_rate=amplitude,
            source_locations_pressure_rate=source + offset,
            receiver_locations_p=receivers + offset,
            q_n_mechanisms=3,
            q_band=(0.1, 0.4),
            **common,
        )
        return output.receiver_amplitudes_p[0]

    with torch.no_grad():
        reference = run(88, 28, 12)
        absorbed = run(32, 0, [8, 9, 10, 11, 12, 13])
        reflecting = run(32, 0, 0)
    norm = torch.linalg.vector_norm(reference)
    assert float(norm) > 1e-4
    absorbed_error = float(torch.linalg.vector_norm(absorbed - reference) / norm)
    reflecting_error = float(torch.linalg.vector_norm(reflecting - reference) / norm)
    peak = reference.abs().amax(-1)
    assert bool((peak > 1e-5).all())
    late_error = float(
        ((absorbed[:, 110:] - reference[:, 110:]).abs().amax(-1) / peak).max()
    )
    record_property("pml_relative_error", absorbed_error)
    record_property("no_pml_relative_error", reflecting_error)
    record_property("late_peak_relative_error", late_error)
    # Tangential damping makes M-PML imperfectly matched. Qualify its
    # measured tradeoff separately from ordinary PML; the late metric is
    # normalised per receiver, including the weakest corner component.
    norm_limit, late_limit = (0.025, 0.15) if mpml_fraction else (0.01, 0.01)
    assert absorbed_error < norm_limit
    assert late_error < late_limit
    assert reflecting_error > 0.1
