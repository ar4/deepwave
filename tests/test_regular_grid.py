"""Physical checks of regular-grid PML profiles."""

import math

import pytest
import torch

from deepwave import _common, _regular_grid


def _profiles(width: list[int], spacing: list[float]) -> list[torch.Tensor]:
    return _regular_grid.set_pml_profiles(
        width,
        4,
        [0, 0, 0, 0],
        0.001,
        spacing,
        1800.0,
        torch.float64,
        torch.device("cpu"),
        20.0,
        (61, 67),
    )


def test_faces_use_their_own_physical_width() -> None:
    """Each face's outer damping has the independent polynomial integral."""
    width = [8, 4, 6, 3]
    spacing = [5.0, 12.0]
    ay, by, dby, ax, bx, dbx = _profiles(width, spacing)
    assert [t.shape for t in (ay, by, dby, ax, bx, dbx)] == [
        (1, 61, 1),
        (1, 61, 1),
        (1, 61, 1),
        (1, 1, 67),
        (1, 1, 67),
        (1, 1, 67),
    ]
    for profile, face_width, dx, i in (
        (ay, 8, 5.0, 0),
        (ay, 4, 5.0, -1),
        (ax, 6, 12.0, 0),
        (ax, 3, 12.0, -1),
    ):
        value = profile.flatten()[i].item()
        sigma0 = -3 * 1800.0 * math.log(0.001) / (2 * face_width * dx)
        assert value == pytest.approx(math.exp(-sigma0 * 0.001), rel=1e-12)
        # A round trip through the continuous polynomial damping layer has
        # attenuation exp(-2/v * integral(sigma dx)) = target R.
        integral = sigma0 * face_width * dx / 3
        assert math.exp(-2 * integral / 1800.0) == pytest.approx(0.001)
    assert abs(by.flatten()[0].item()) > 0
    assert abs(bx.flatten()[0].item()) > 0
    assert bool((dby != 0).any())
    assert bool((dbx != 0).any())


def test_opposite_and_orthogonal_faces_are_independent() -> None:
    """Changing one face cannot renormalise the other three."""
    base = _profiles([8, 4, 6, 3], [5.0, 12.0])
    changed = _profiles([8, 13, 6, 3], [5.0, 12.0])
    for i in (0, 1, 2):
        assert torch.equal(base[i].flatten()[:9], changed[i].flatten()[:9])
    for i in (3, 4, 5):
        assert torch.equal(base[i], changed[i])
    assert not torch.equal(base[0], changed[0])


def test_zero_widths_and_raw_profiles_agree_with_ab() -> None:
    """Absent faces produce zero memory; raw sigma/alpha give the same a/b."""
    assert all(bool((p == 0).all()) for p in _profiles([0, 0, 0, 0], [5.0, 12.0]))
    a, b = _common.setup_pml(
        [8, 0],
        [8.0, 60.0],
        5.0,
        0.001,
        61,
        1800.0,
        torch.float64,
        torch.device("cpu"),
        20.0,
    )
    sigma, alpha = _common.setup_pml_sigma_alpha(
        [8, 0],
        [8.0, 60.0],
        5.0,
        61,
        1800.0,
        torch.float64,
        torch.device("cpu"),
        20.0,
    )
    active = sigma > 0
    expected_a = torch.where(active, torch.exp(-(sigma + alpha) * 0.001), 0)
    expected_b = torch.where(
        active, sigma / (sigma + alpha).clamp(min=1e-9) * (expected_a - 1), 0
    )
    torch.testing.assert_close(a, expected_a, rtol=0, atol=0)
    torch.testing.assert_close(b, expected_b, rtol=0, atol=0)
    assert bool(active.any())
    assert bool((~active).any())
