"""The on-boundary node values a ``free``/``rigid`` face must pin to zero.

Mutation testing found that removing the ``* 0`` that holds these two nodes
at zero left the whole test suite green:

* ``py_boundaries_pressure_free_surface_not_zeroed`` --
  ``fill_acoustic_pressure_boundary`` no longer zeroes the on-surface pressure
  node of a ``free`` face;
* ``py_boundaries_rigid_velocity_wall_not_zeroed`` --
  ``fill_acoustic_velocity_boundary`` no longer zeroes the on-wall
  boundary-normal velocity node of a ``rigid`` high face.

Both survived the whole suite. The existing ``free``/``rigid`` tests check
*consequences* -- the analytic reflection coefficient, adjoint parity,
gradcheck -- and the energy leak from an unpinned node is below their
aggregate tolerances. Nothing asserted the node value itself, which is the
direct physical statement of the boundary condition: pressure release means
the pressure on the surface *is* zero, and a rigid wall means the
wall-normal velocity on the wall *is* zero.

These tests assert exactly that, both on the fill functions directly and
end-to-end through ``acoustic()``.

Index layout (see ``_boundaries.py``): the staggered grids carry ``fd_pad``
low guard cells and ``fd_pad - 1`` high ones, so with ``pad = fd_pad`` the
interior spans ``[pad, n - pad + 1)``. The free-surface pressure node of a low
face is therefore at index ``pad``, and the rigid wall's velocity node on a
high face is the last interior node, at index ``n - pad``.
"""

import pytest
import torch

import deepwave
from deepwave import _boundaries
from deepwave._common import BC_FREE, BC_PML, BC_RIGID

DTYPE = torch.float64
FD_PAD = 2  # accuracy 4


def _nonzero_field(n_shots: int, shape: tuple[int, ...]) -> torch.Tensor:
    """A field with no zeros anywhere, so a zero can only come from the fill."""
    generator = torch.Generator().manual_seed(0)
    field = torch.rand(n_shots, *shape, dtype=DTYPE, generator=generator) + 1.0
    assert bool((field != 0).all())
    return field


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_free_face_pins_surface_pressure_node_to_zero(ndim: int) -> None:
    """``free`` holds the on-surface pressure node at exactly zero."""
    shape = tuple(12 + i for i in range(ndim))
    for dim in range(ndim):
        boundary = [BC_PML] * 2 * ndim
        boundary[2 * dim] = BC_FREE
        pressure = _nonzero_field(2, shape)
        filled = _boundaries.fill_acoustic_pressure_boundary(pressure, boundary, FD_PAD)
        surface = filled.narrow(1 + dim, FD_PAD, 1)
        assert bool((surface == 0).all()), (
            f"ndim={ndim}, dim={dim}: the free-surface pressure node is not "
            f"exactly zero (max abs {surface.abs().max().item():.3e})"
        )
        # The neighbouring interior node must NOT have been zeroed too, or
        # the test would also pass for a fill that zeroes everything.
        neighbour = filled.narrow(1 + dim, FD_PAD + 1, 1)
        assert bool((neighbour != 0).all())

    # And the same for a high face.
    for dim in range(ndim):
        boundary = [BC_PML] * 2 * ndim
        boundary[2 * dim + 1] = BC_FREE
        pressure = _nonzero_field(2, shape)
        filled = _boundaries.fill_acoustic_pressure_boundary(pressure, boundary, FD_PAD)
        n = filled.size(1 + dim)
        surface = filled.narrow(1 + dim, n - FD_PAD, 1)
        assert bool((surface == 0).all()), (
            f"ndim={ndim}, dim={dim}: the high-face free-surface pressure "
            f"node is not exactly zero (max abs {surface.abs().max().item():.3e})"
        )


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_rigid_high_face_pins_wall_velocity_node_to_zero(ndim: int) -> None:
    """``rigid`` holds the on-wall boundary-normal velocity node at exactly zero.

    Only ``velocities[dim]`` is imaged along axis ``dim``; the tangential
    velocities are untouched, and this checks that too (zeroing all of them
    would be just as wrong).
    """
    shape = tuple(12 + i for i in range(ndim))
    for dim in range(ndim):
        boundary = [BC_PML] * 2 * ndim
        boundary[2 * dim + 1] = BC_RIGID
        velocities = [_nonzero_field(2, shape) for _ in range(ndim)]
        filled = _boundaries.fill_acoustic_velocity_boundary(
            [v.clone() for v in velocities], boundary, FD_PAD
        )
        n = filled[dim].size(1 + dim)
        wall = filled[dim].narrow(1 + dim, n - FD_PAD, 1)
        assert bool((wall == 0).all()), (
            f"ndim={ndim}, dim={dim}: the rigid-wall normal-velocity node is "
            f"not exactly zero (max abs {wall.abs().max().item():.3e})"
        )
        # The node just inside the wall is untouched.
        inside = filled[dim].narrow(1 + dim, n - FD_PAD - 1, 1)
        assert bool((inside != 0).all())
        # Tangential velocities are not imaged along this axis at all.
        for other in range(ndim):
            if other != dim:
                assert torch.equal(filled[other], velocities[other])


@pytest.mark.parametrize("backend", ["eager", "compiled"])
def test_acoustic_free_surface_pressure_is_zero_end_to_end(backend: str) -> None:
    """A propagated ``free`` face leaves the surface pressure node at zero.

    The direct-fill tests above pin the Python helper; this pins the property
    that actually reaches the user, through a real ``acoustic()`` run on both
    backends. Note that the *returned* wavefields have the ``fd_pad`` guard
    cells stripped, so the surface node is index 0 along that axis, not
    ``fd_pad``.

    ``rigid`` and the default PML on the same face are checked as controls:
    the surface node is emphatically non-zero for both, so a zero there is
    the free-surface condition and not an artefact of an unwritten edge node.
    (The rigid wall's own velocity node is *not* checkable this way -- the
    last ``vy`` row of the returned wavefield is structurally zero even under
    plain PML, being an unused staggered node -- which is why the
    ``rigid`` condition is pinned only by the direct-fill test above.)
    """
    ny, nx = 30, 30
    v = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    rho = torch.full((ny, nx), 2200.0, dtype=DTYPE)
    nt = 80
    wavelet = (
        deepwave.wavelets.ricker(25.0, nt, 0.0005, 1.2 / 25.0)
        .to(DTYPE)
        .reshape(1, 1, -1)
    )
    sources = torch.tensor([[[6, nx // 2]]])

    def run(boundary: "list[str] | None", pml_width: list[int]) -> torch.Tensor:
        out = deepwave.acoustic(
            v,
            rho,
            4.0,
            0.0005,
            source_amplitudes_volume_rate=wavelet,
            source_locations_volume_rate=sources,
            boundary=boundary,
            pml_width=pml_width,
            accuracy=4,
            pml_freq=25.0,
            backend=backend,
        )
        return out.pressure_0

    free_surface = run(["free", "pml", "pml", "pml"], [0, 10, 10, 10])
    assert bool((free_surface[:, 0] == 0).all()), (
        f"backend={backend}: the free-surface pressure node is not exactly "
        f"zero (max abs {free_surface[:, 0].abs().max().item():.3e})"
    )
    # Non-vacuity: the wave really did reach the surface.
    assert free_surface[:, 1].abs().max() > 0, "no energy reached the surface"

    # Controls: the same node is far from zero without a free face, so the
    # assertion above is testing the boundary condition, not an edge artefact.
    rigid_face = run(["rigid", "pml", "pml", "pml"], [0, 10, 10, 10])
    assert rigid_face[:, 0].abs().max() > 0
    absorbing = run(None, [10, 10, 10, 10])
    assert absorbing[:, 0].abs().max() > 0
