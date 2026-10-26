"""Physical profile checks and edge/interior pinning for staggered grids."""

import math

import pytest
import torch

from deepwave._staggered_grid import set_pml_profiles


def _profiles(width: list[int], spacing: list[float]) -> list[torch.Tensor]:
    return set_pml_profiles(
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


def test_staggered_faces_have_local_physical_damping() -> None:
    """Full-grid outer coefficients obey each face's own width and cell size."""
    ay, by, ayh, byh, ax, bx, axh, bxh = _profiles([8, 4, 6, 3], [5.0, 12.0])
    assert all(t.shape == (1, 61, 1) for t in (ay, by, ayh, byh))
    assert all(t.shape == (1, 1, 67) for t in (ax, bx, axh, bxh))
    for profile, width, dx, index in (
        (ay, 8, 5.0, 0),
        (ay, 4, 5.0, -1),
        (ax, 6, 12.0, 0),
        (ax, 3, 12.0, -1),
    ):
        sigma0 = -3 * 1800.0 * math.log(0.001) / (2 * width * dx)
        assert profile.flatten()[index].item() == pytest.approx(
            math.exp(-sigma0 * 0.001), rel=1e-12
        )
    assert ayh.flatten()[0] > ay.flatten()[0]
    assert axh.flatten()[0] > ax.flatten()[0]
    assert bool((by != 0).any())
    assert bool((bx != 0).any())
    assert bool((byh != 0).any())
    assert bool((bxh != 0).any())


def test_staggered_face_changes_do_not_renormalise_others() -> None:
    """Changing one face leaves its opposite and orthogonal faces alone."""
    base = _profiles([8, 4, 6, 3], [5.0, 12.0])
    changed = _profiles([8, 13, 6, 3], [5.0, 12.0])
    for i in range(4):
        assert torch.equal(base[i].flatten()[:9], changed[i].flatten()[:9])
    for i in range(4, 8):
        assert torch.equal(base[i], changed[i])
    assert not torch.equal(base[0], changed[0])


def test_staggered_zero_widths_have_no_memory_profile() -> None:
    """A face with no PML contributes no coefficient or memory."""
    assert all(bool((p == 0).all()) for p in _profiles([0, 0, 0, 0], [5.0, 12.0]))
    partial = _profiles([8, 0, 0, 0], [5.0, 12.0])
    assert bool((partial[0] != 0).any())
    assert all(bool((p == 0).all()) for p in partial[4:])


# ---------------------------------------------------------------------------
# Pinning tests for the consolidated ``zero_edges_and_interiors``.
#
# The per-propagator edge/interior zeroing was consolidated into
# ``deepwave._staggered_grid.zero_edges_and_interiors`` (each propagator now
# only supplies its ``half_grid`` / ``pml_edge_idx`` tables). These tests lock
# that consolidation: each ``_ref_*`` function below is a verbatim copy of the
# ORIGINAL per-propagator implementation (calling the real
# ``deepwave._common.zero_interior`` / ``zero_interior_box``), and every test
# asserts the live propagator function is byte-identical to its reference.
# Keep these permanently; they are cheap.
# ---------------------------------------------------------------------------

import importlib  # noqa: E402

import deepwave._common  # noqa: E402

# ``deepwave.acoustic`` (etc.) resolve to the propagator *functions* (the
# ``__init__`` re-export shadows the submodule), so load the modules explicitly.
acoustic = importlib.import_module("deepwave.acoustic")
elastic = importlib.import_module("deepwave.elastic")
viscoacoustic = importlib.import_module("deepwave.viscoacoustic")


def _zero_edge_ref(tensor: torch.Tensor, fd_pad: int, dim: int) -> torch.Tensor:
    tensor[(slice(None),) + (slice(None),) * dim + (-fd_pad,)].fill_(0)
    return tensor


def _ref_acoustic(
    wavefields: list[torch.Tensor],
    ndim: int,
    fd_pad: int,
    fd_pad_list: list[int],
    pml_width: list[int],
    interior: bool = True,
) -> None:
    num_vars = len(wavefields)
    half_grid = [[False] * ndim for _ in range(num_vars)]
    pml_edge_idx: list[list[int]] = [[] * ndim]
    if ndim == 3:
        half_grid[1][0] = True
        half_grid[7][0] = True
        half_grid[2][1] = True
        half_grid[8][1] = True
        half_grid[3][2] = True
        half_grid[9][2] = True
        pml_edge_idx = [[4, 7], [5, 8], [6, 9]]
    elif ndim == 2:
        half_grid[1][0] = True
        half_grid[5][0] = True
        half_grid[2][1] = True
        half_grid[6][1] = True
        pml_edge_idx = [[3, 5], [4, 6]]
    elif ndim == 1:
        half_grid[1][0] = True
        half_grid[3][0] = True
        pml_edge_idx = [[2, 3]]
    for i in range(num_vars):
        for dim in range(ndim):
            if half_grid[i][dim]:
                wavefields[i] = _zero_edge_ref(wavefields[i], fd_pad, dim)
    if interior:
        for dim in range(ndim):
            for i in pml_edge_idx[dim]:
                fd_pad_shifted = list(fd_pad_list)
                if half_grid[i][dim]:
                    fd_pad_shifted[2 * dim + 1] += 1
                wavefields[i] = deepwave._common.zero_interior(
                    wavefields[i], fd_pad_shifted, pml_width, dim
                )


def _ref_viscoacoustic(
    wavefields: list[torch.Tensor],
    ndim: int,
    fd_pad: int,
    fd_pad_list: list[int],
    pml_width: list[int],
    interior: bool = True,
) -> None:
    num_vars = len(wavefields)
    half_grid = [[False] * ndim for _ in range(num_vars)]
    pml_edge_idx: list[list[int]] = [[] * ndim]
    if ndim == 3:
        half_grid[2][0] = True
        half_grid[8][0] = True
        half_grid[3][1] = True
        half_grid[9][1] = True
        half_grid[4][2] = True
        half_grid[10][2] = True
        pml_edge_idx = [[5, 8], [6, 9], [7, 10]]
    elif ndim == 2:
        half_grid[2][0] = True
        half_grid[6][0] = True
        half_grid[3][1] = True
        half_grid[7][1] = True
        pml_edge_idx = [[4, 6], [5, 7]]
    elif ndim == 1:
        half_grid[2][0] = True
        half_grid[4][0] = True
        pml_edge_idx = [[3, 4]]
    for i in range(num_vars):
        for dim in range(ndim):
            if half_grid[i][dim]:
                wavefields[i] = _zero_edge_ref(wavefields[i], fd_pad, dim)
    if interior:
        for dim in range(ndim):
            for i in pml_edge_idx[dim]:
                fd_pad_shifted = list(fd_pad_list)
                if half_grid[i][dim]:
                    fd_pad_shifted[2 * dim + 1] += 1
                wavefields[i] = deepwave._common.zero_interior(
                    wavefields[i], fd_pad_shifted, pml_width, dim
                )


def _ref_elastic(
    wavefields: list[torch.Tensor],
    ndim: int,
    fd_pad: int,
    fd_pad_list: list[int],
    pml_width: list[int],
    interior: bool = True,
    mpml: bool = False,
) -> None:
    if ndim == 3:
        half_grid = [[False, False, False] for _ in range(len(wavefields))]
        for i in [0, 2, 3, 5, 6, 7, 8, 9, 10, 11]:
            half_grid[i][0] = True
            wavefields[i] = _zero_edge_ref(wavefields[i], fd_pad, 0)
        for i in [2, 5, 7, 12, 14, 16, 18, 19, 20, 22]:
            half_grid[i][1] = True
            wavefields[i] = _zero_edge_ref(wavefields[i], fd_pad, 1)
        for i in [3, 6, 8, 13, 16, 18, 19, 21, 23, 26]:
            half_grid[i][2] = True
            wavefields[i] = _zero_edge_ref(wavefields[i], fd_pad, 2)
        pml_edge_idx = [
            [4, 7, 8, 9, 12, 13],
            [5, 10, 17, 19, 20, 21],
            [6, 11, 18, 22, 25, 26],
        ]
    elif ndim == 2:
        half_grid = [[False, False] for _ in range(len(wavefields))]
        for i in [0, 2, 4, 5, 6, 8]:
            half_grid[i][0] = True
            wavefields[i] = _zero_edge_ref(wavefields[i], fd_pad, 0)
        for i in [2, 4, 5, 7, 9, 12]:
            half_grid[i][1] = True
            wavefields[i] = _zero_edge_ref(wavefields[i], fd_pad, 1)
        pml_edge_idx = [[3, 5, 6, 7], [4, 8, 11, 12]]
    elif ndim == 1:
        half_grid = [
            [
                False,
            ]
            for _ in range(len(wavefields))
        ]
        for i in [0, 3]:
            half_grid[i][0] = True
            wavefields[i] = _zero_edge_ref(wavefields[i], fd_pad, 0)
        pml_edge_idx = [[2, 3]]
    else:
        return
    if not interior:
        return
    if mpml:
        for i in sorted({idx for idx_list in pml_edge_idx for idx in idx_list}):
            fd_pad_shifted = list(fd_pad_list)
            for dim in range(ndim):
                if half_grid[i][dim]:
                    fd_pad_shifted[2 * dim + 1] += 1
            wavefields[i] = deepwave._common.zero_interior_box(
                wavefields[i], fd_pad_shifted, pml_width
            )
    else:
        for dim in range(ndim):
            for i in pml_edge_idx[dim]:
                fd_pad_shifted = list(fd_pad_list)
                if half_grid[i][dim]:
                    fd_pad_shifted[2 * dim + 1] += 1
                wavefields[i] = deepwave._common.zero_interior(
                    wavefields[i], fd_pad_shifted, pml_width, dim
                )


# num_vars per propagator per ndim.
_ACOUSTIC_NVARS = {1: 4, 2: 7, 3: 10}
_VISCO_NVARS = {1: 5, 2: 8, 3: 11}
_ELASTIC_NVARS = {1: 4, 2: 13, 3: 27}


def _make_wavefields(num_vars: int, ndim: int, seed: int) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    n = 16 if ndim == 3 else 24
    shape = (2,) + (n,) * ndim
    return [torch.randn(shape, dtype=torch.float64) for _ in range(num_vars)]


def _params(ndim: int) -> tuple:
    fd_pad = 2
    fd_pad_list = [2, 1] * ndim
    pml_width = [3, 4] * ndim
    return fd_pad, fd_pad_list, pml_width


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("interior", [True, False])
def test_zero_pinning_acoustic(ndim: int, interior: bool) -> None:
    """acoustic.zero_edges_and_interiors is byte-identical to the original."""
    fd_pad, fd_pad_list, pml_width = _params(ndim)
    base = _make_wavefields(_ACOUSTIC_NVARS[ndim], ndim, seed=100 + ndim)
    ref = [w.clone() for w in base]
    new = [w.clone() for w in base]
    _ref_acoustic(ref, ndim, fd_pad, fd_pad_list, pml_width, interior=interior)
    acoustic.zero_edges_and_interiors(
        new, ndim, fd_pad, fd_pad_list, pml_width, interior=interior
    )
    for r, x in zip(ref, new):
        assert torch.equal(r, x)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("interior", [True, False])
def test_zero_pinning_viscoacoustic(ndim: int, interior: bool) -> None:
    """viscoacoustic.zero_edges_and_interiors matches the original."""
    fd_pad, fd_pad_list, pml_width = _params(ndim)
    base = _make_wavefields(_VISCO_NVARS[ndim], ndim, seed=200 + ndim)
    ref = [w.clone() for w in base]
    new = [w.clone() for w in base]
    _ref_viscoacoustic(ref, ndim, fd_pad, fd_pad_list, pml_width, interior=interior)
    viscoacoustic.zero_edges_and_interiors(
        new, ndim, fd_pad, fd_pad_list, pml_width, interior=interior
    )
    for r, x in zip(ref, new):
        assert torch.equal(r, x)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("interior", [True, False])
@pytest.mark.parametrize("mpml", [False, True])
def test_zero_pinning_elastic(ndim: int, interior: bool, mpml: bool) -> None:
    """elastic.zero_edges_and_interiors matches the original (incl. mpml)."""
    fd_pad, fd_pad_list, pml_width = _params(ndim)
    base = _make_wavefields(_ELASTIC_NVARS[ndim], ndim, seed=300 + ndim)
    ref = [w.clone() for w in base]
    new = [w.clone() for w in base]
    _ref_elastic(
        ref, ndim, fd_pad, fd_pad_list, pml_width, interior=interior, mpml=mpml
    )
    elastic.zero_edges_and_interiors(
        new, ndim, fd_pad, fd_pad_list, pml_width, interior=interior, mpml=mpml
    )
    for r, x in zip(ref, new):
        assert torch.equal(r, x)
