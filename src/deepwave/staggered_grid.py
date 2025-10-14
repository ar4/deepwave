"""Utility functions for staggered grid PML setup.

This module provides functions to set up Perfectly Matched Layers (PML)
profiles for wave propagation simulations on a staggered grid.
"""

from typing import List, Optional, Tuple

import torch

import deepwave


def set_pml_profiles(
    pml_width: List[int],
    accuracy: int,
    fd_pad: List[int],
    dt: float,
    grid_spacing: List[float],
    max_vel: float,
    dtype: torch.dtype,
    device: torch.device,
    pml_freq: float,
    shape: Tuple[int, ...],
) -> List[torch.Tensor]:
    """Sets up PML profiles for a staggered grid.

    Args:
        pml_width: A list of integers specifying the width of the PML
            on each side (e.g., for 2D: top, bottom, left, right).
        accuracy: The finite-difference accuracy order.
        fd_pad: A list of integers specifying the padding for finite-difference.
        dt: The time step.
        grid_spacing: A list of floats specifying the grid spacing in
            y and x directions.
        max_vel: The maximum velocity in the model.
        dtype: The data type of the tensors (e.g., torch.float32).
        device: The device on which the tensors will be (e.g., 'cuda', 'cpu').
        pml_freq: The PML frequency.
        shape: The number of grid points in each spatial dimension.

    Returns:
        A list of torch.Tensors representing the PML profiles (e.g., in 2D:
        ay, ayh, ax, axh, by, byh, bx, bxh).

    """
    ndim = len(shape)
    pml_start: List[List[int]] = [
        [
            fd_pad[dim * 2] + pml_width[dim * 2],
            shape[dim] - 1 - fd_pad[dim * 2 + 1] - pml_width[dim * 2 + 1],
        ]
        for dim in range(ndim)
    ]
    physical_widths: List[float] = []
    for dim in range(ndim):
        physical_widths.append(pml_width[dim * 2] * grid_spacing[dim])
        physical_widths.append(pml_width[dim * 2 + 1] * grid_spacing[dim])
    max_pml = max(physical_widths)

    pml_profiles: List[torch.Tensor] = []
    for dim in range(ndim):
        a, b = deepwave.common.setup_pml(
            pml_width[2 * dim : 2 * dim + 2],
            [float(v) for v in pml_start[dim]],
            max_pml,
            dt,
            shape[dim],
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.0,
        )
        ah, bh = deepwave.common.setup_pml(
            pml_width[2 * dim : 2 * dim + 2],
            [float(v) for v in pml_start[dim]],
            max_pml,
            dt,
            shape[dim],
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.5,
        )
        s_list: List[Optional[slice]] = [None] * (ndim + 1)
        s_list[1 + dim] = slice(None)
        s: Tuple[Optional[slice], ...] = tuple(s_list)
        pml_profiles.extend([a[tuple(s)], b[tuple(s)], ah[tuple(s)], bh[tuple(s)]])
    return pml_profiles


def diffz1(a: torch.Tensor, accuracy: int, rdz: float) -> torch.Tensor:
    """Calculates the first z derivative at integer grid points."""
    if accuracy == 2:
        return torch.nn.functional.pad(
            (a[..., 1:, :, :] - a[..., :-1, :, :]) * rdz, (0, 0, 0, 0, 1, 0)
        )
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 2:-1, :, :] - a[..., 1:-2, :, :])
                + -1 / 24 * (a[..., 3:, :, :] - a[..., :-3, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 2, 1),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 3:-2, :, :] - a[..., 2:-3, :, :])
                + -25 / 384 * (a[..., 4:-1, :, :] - a[..., 1:-4, :, :])
                + 3 / 640 * (a[..., 5:, :, :] - a[..., :, :-5, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 3, 2),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 4:-3, :, :] - a[..., 3:-4, :, :])
            + -245 / 3072 * (a[..., 5:-2, :, :] - a[..., 2:-5, :, :])
            + 49 / 5120 * (a[..., 6:-1, :, :] - a[..., 1:-6, :, :])
            + -5 / 7168 * (a[..., 7:, :, :] - a[..., :, :-7, :, :])
        )
        * rdz,
        (0, 0, 0, 0, 4, 3),
    )


def diffy1(a: torch.Tensor, accuracy: int, rdy: float) -> torch.Tensor:
    """Calculates the first y derivative at integer grid points."""
    if accuracy == 2:
        return torch.nn.functional.pad(
            (a[..., 1:, :] - a[..., :-1, :]) * rdy, (0, 0, 1, 0)
        )
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 2:-1, :] - a[..., 1:-2, :])
                + -1 / 24 * (a[..., 3:, :] - a[..., :-3, :])
            )
            * rdy,
            (0, 0, 2, 1),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 3:-2, :] - a[..., 2:-3, :])
                + -25 / 384 * (a[..., 4:-1, :] - a[..., 1:-4, :])
                + 3 / 640 * (a[..., 5:, :] - a[..., :-5, :])
            )
            * rdy,
            (0, 0, 3, 2),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 4:-3, :] - a[..., 3:-4, :])
            + -245 / 3072 * (a[..., 5:-2, :] - a[..., 2:-5, :])
            + 49 / 5120 * (a[..., 6:-1, :] - a[..., 1:-6, :])
            + -5 / 7168 * (a[..., 7:, :] - a[..., :-7, :])
        )
        * rdy,
        (0, 0, 4, 3),
    )


def diffx1(a: torch.Tensor, accuracy: int, rdx: float) -> torch.Tensor:
    """Calculates the first x derivative at integer grid points."""
    if accuracy == 2:
        return torch.nn.functional.pad((a[..., 1:] - a[..., :-1]) * rdx, (1, 0))
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 2:-1] - a[..., 1:-2])
                + -1 / 24 * (a[..., 3:] - a[..., :-3])
            )
            * rdx,
            (2, 1),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 3:-2] - a[..., 2:-3])
                + -25 / 384 * (a[..., 4:-1] - a[..., 1:-4])
                + 3 / 640 * (a[..., 5:] - a[..., :-5])
            )
            * rdx,
            (3, 2),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 4:-3] - a[..., 3:-4])
            + -245 / 3072 * (a[..., 5:-2] - a[..., 2:-5])
            + 49 / 5120 * (a[..., 6:-1] - a[..., 1:-6])
            + -5 / 7168 * (a[..., 7:] - a[..., :-7])
        )
        * rdx,
        (4, 3),
    )


def diffzh1(a: torch.Tensor, accuracy: int, rdz: float) -> torch.Tensor:
    """Calculates the first z derivative at half integer grid points."""
    if accuracy == 2:
        return torch.nn.functional.pad(
            (a[..., 2:, :, :] - a[..., 1:-1, :, :]) * rdz, (0, 0, 0, 0, 1, 1)
        )
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 3:-1, :, :] - a[..., 2:-2, :, :])
                + -1 / 24 * (a[..., 4:, :, :] - a[..., 1:-3, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 2, 2),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 4:-2, :, :] - a[..., 3:-3, :, :])
                + -25 / 384 * (a[..., 5:-1, :, :] - a[..., 2:-4, :, :])
                + 3 / 640 * (a[..., 6:, :, :] - a[..., 1:-5, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 3, 3),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 5:-3, :, :] - a[..., 4:-4, :, :])
            + -245 / 3072 * (a[..., 6:-2, :, :] - a[..., 3:-5, :, :])
            + 49 / 5120 * (a[..., 7:-1, :, :] - a[..., 2:-6, :, :])
            + -5 / 7168 * (a[..., 8:, :, :] - a[..., 1:-7, :, :])
        )
        * rdz,
        (0, 0, 0, 0, 4, 4),
    )


def diffyh1(a: torch.Tensor, accuracy: int, rdy: float) -> torch.Tensor:
    """Calculates the first y derivative at half integer grid points."""
    if accuracy == 2:
        return torch.nn.functional.pad(
            (a[..., 2:, :] - a[..., 1:-1, :]) * rdy, (0, 0, 1, 1)
        )
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 3:-1, :] - a[..., 2:-2, :])
                + -1 / 24 * (a[..., 4:, :] - a[..., 1:-3, :])
            )
            * rdy,
            (0, 0, 2, 2),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 4:-2, :] - a[..., 3:-3, :])
                + -25 / 384 * (a[..., 5:-1, :] - a[..., 2:-4, :])
                + 3 / 640 * (a[..., 6:, :] - a[..., 1:-5, :])
            )
            * rdy,
            (0, 0, 3, 3),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 5:-3, :] - a[..., 4:-4, :])
            + -245 / 3072 * (a[..., 6:-2, :] - a[..., 3:-5, :])
            + 49 / 5120 * (a[..., 7:-1, :] - a[..., 2:-6, :])
            + -5 / 7168 * (a[..., 8:, :] - a[..., 1:-7, :])
        )
        * rdy,
        (0, 0, 4, 4),
    )


def diffxh1(a: torch.Tensor, accuracy: int, rdx: float) -> torch.Tensor:
    """Calculates the first x derivative at half integer grid points."""
    if accuracy == 2:
        return torch.nn.functional.pad((a[..., 2:] - a[..., 1:-1]) * rdx, (1, 1))
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 3:-1] - a[..., 2:-2])
                + -1 / 24 * (a[..., 4:] - a[..., 1:-3])
            )
            * rdx,
            (2, 2),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 4:-2] - a[..., 3:-3])
                + -25 / 384 * (a[..., 5:-1] - a[..., 2:-4])
                + 3 / 640 * (a[..., 6:] - a[..., 1:-5])
            )
            * rdx,
            (3, 3),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 5:-3] - a[..., 4:-4])
            + -245 / 3072 * (a[..., 6:-2] - a[..., 3:-5])
            + 49 / 5120 * (a[..., 7:-1] - a[..., 2:-6])
            + -5 / 7168 * (a[..., 8:] - a[..., 1:-7])
        )
        * rdx,
        (4, 4),
    )


def diff1(
    wf: torch.Tensor, dim: int, accuracy: int, rdx: float, ndim: int
) -> torch.Tensor:
    """Perform first-order differentiation on a staggered grid."""
    if ndim - dim == 3:
        return diffz1(wf, accuracy, rdx)
    if ndim - dim == 2:
        return diffy1(wf, accuracy, rdx)
    return diffx1(wf, accuracy, rdx)


def diff1h(
    wf: torch.Tensor, dim: int, accuracy: int, rdx: float, ndim: int
) -> torch.Tensor:
    """Perform first-order differentiation on a staggered grid (half-step)."""
    if ndim - dim == 3:
        return diffzh1(wf, accuracy, rdx)
    if ndim - dim == 2:
        return diffyh1(wf, accuracy, rdx)
    return diffxh1(wf, accuracy, rdx)
