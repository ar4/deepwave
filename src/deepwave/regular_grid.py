"""Utility functions for regular grid PML setup.

This module provides functions to set up Perfectly Matched Layers (PML)
profiles for wave propagation simulations on a regular grid.
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor

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
) -> List[Tensor]:
    """Sets up PML profiles for a regular grid.

    Args:
        pml_width: A list of integers specifying the width of the PML
            on each side (e.g., for 2D: top, bottom, left, right).
        accuracy: The finite-difference accuracy order.
        fd_pad: A list of integers specifying the padding for finite-difference.
        dt: The time step.
        grid_spacing: A list of floats specifying the grid spacing in each
            spatial dimension.
        max_vel: The maximum velocity in the model.
        dtype: The data type of the tensors (e.g., torch.float32).
        device: The device on which the tensors will be (e.g., 'cuda', 'cpu').
        pml_freq: The PML frequency.
        shape: The number of grid points in each spatial dimension.

    Returns:
        A list of torch.Tensors representing the PML profiles (e.g., in 2D:
        ay, by, dbydy, ax, bx, dbxdx).

    """
    ndim = len(shape)
    pml_start: List[List[float]] = [
        [
            float(fd_pad[dim * 2] + pml_width[dim * 2]),
            float(shape[dim] - 1 - fd_pad[dim * 2 + 1] - pml_width[dim * 2 + 1]),
        ]
        for dim in range(ndim)
    ]
    physical_widths: List[float] = []
    for dim in range(ndim):
        physical_widths.append(pml_width[dim * 2] * grid_spacing[dim])
        physical_widths.append(pml_width[dim * 2 + 1] * grid_spacing[dim])
    max_pml = max(physical_widths) if physical_widths else 0.0

    pml_profiles: List[Tensor] = []
    for dim in range(ndim):
        a, b = deepwave.common.setup_pml(
            pml_width[2 * dim : 2 * dim + 2],
            pml_start[dim],
            max_pml,
            dt,
            shape[dim],
            max_vel,
            dtype,
            device,
            pml_freq,
        )
        db = diffx1(b, accuracy, 1 / grid_spacing[dim])
        s: List[Optional[slice]] = [None] * (ndim + 1)
        s[1 + dim] = slice(None)
        pml_profiles.extend([a[tuple(s)], b[tuple(s)], db[tuple(s)]])
    return pml_profiles


def diffz1(a: torch.Tensor, accuracy: int, rdz: float) -> torch.Tensor:
    """Calculates the first derivative in the z-direction."""
    if accuracy == 2:
        return torch.nn.functional.pad(
            (1 / 2 * (a[..., 2:, :, :] - a[..., :-2, :, :])) * rdz, (0, 0, 0, 0, 1, 1)
        )
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                8 / 12 * (a[..., 3:-1, :, :] - a[..., 1:-3, :, :])
                + -1 / 12 * (a[..., 4:, :, :] - a[..., :-4, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 2, 2),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                3 / 4 * (a[..., 4:-2, :, :] - a[..., 2:-4, :, :])
                + -3 / 20 * (a[..., 5:-1, :, :] - a[..., 1:-5, :, :])
                + 1 / 60 * (a[..., 6:, :, :] - a[..., :-6, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 3, 3),
        )
    return torch.nn.functional.pad(
        (
            4 / 5 * (a[..., 5:-3, :, :] - a[..., 3:-5, :, :])
            + -1 / 5 * (a[..., 6:-2, :, :] - a[..., 2:-6, :, :])
            + 4 / 105 * (a[..., 7:-1, :, :] - a[..., 1:-7, :, :])
            + -1 / 280 * (a[..., 8:, :, :] - a[..., :-8, :, :])
        )
        * rdz,
        (0, 0, 0, 0, 4, 4),
    )


def diffy1(a: torch.Tensor, accuracy: int, rdy: float) -> torch.Tensor:
    """Calculates the first derivative in the y-direction."""
    if accuracy == 2:
        return torch.nn.functional.pad(
            (1 / 2 * (a[..., 2:, :] - a[..., :-2, :])) * rdy, (0, 0, 1, 1)
        )
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                8 / 12 * (a[..., 3:-1, :] - a[..., 1:-3, :])
                + -1 / 12 * (a[..., 4:, :] - a[..., :-4, :])
            )
            * rdy,
            (0, 0, 2, 2),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                3 / 4 * (a[..., 4:-2, :] - a[..., 2:-4, :])
                + -3 / 20 * (a[..., 5:-1, :] - a[..., 1:-5, :])
                + 1 / 60 * (a[..., 6:, :] - a[..., :-6, :])
            )
            * rdy,
            (0, 0, 3, 3),
        )
    return torch.nn.functional.pad(
        (
            4 / 5 * (a[..., 5:-3, :] - a[..., 3:-5, :])
            + -1 / 5 * (a[..., 6:-2, :] - a[..., 2:-6, :])
            + 4 / 105 * (a[..., 7:-1, :] - a[..., 1:-7, :])
            + -1 / 280 * (a[..., 8:, :] - a[..., :-8, :])
        )
        * rdy,
        (0, 0, 4, 4),
    )


def diffx1(a: torch.Tensor, accuracy: int, rdx: float) -> torch.Tensor:
    """Calculates the first derivative in the x-direction."""
    if accuracy == 2:
        return torch.nn.functional.pad(
            (1 / 2 * (a[..., 2:] - a[..., :-2])) * rdx, (1, 1)
        )
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                8 / 12 * (a[..., 3:-1] - a[..., 1:-3])
                + -1 / 12 * (a[..., 4:] - a[..., :-4])
            )
            * rdx,
            (2, 2),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                3 / 4 * (a[..., 4:-2] - a[..., 2:-4])
                + -3 / 20 * (a[..., 5:-1] - a[..., 1:-5])
                + 1 / 60 * (a[..., 6:] - a[..., :-6])
            )
            * rdx,
            (3, 3),
        )
    return torch.nn.functional.pad(
        (
            4 / 5 * (a[..., 5:-3] - a[..., 3:-5])
            + -1 / 5 * (a[..., 6:-2] - a[..., 2:-6])
            + 4 / 105 * (a[..., 7:-1] - a[..., 1:-7])
            + -1 / 280 * (a[..., 8:] - a[..., :-8])
        )
        * rdx,
        (4, 4),
    )


def diffz2(a: torch.Tensor, accuracy: int, rdz2: float) -> torch.Tensor:
    """Calculates the second derivative in the z-direction."""
    if accuracy == 2:
        return torch.nn.functional.pad(
            (-2 * a[..., 1:-1, :, :] + 1 * (a[..., 2:, :, :] + a[..., :-2, :, :]))
            * rdz2,
            (0, 0, 0, 0, 1, 1),
        )
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                -5 / 2 * a[..., 2:-2, :, :]
                + 4 / 3 * (a[..., 3:-1, :, :] + a[..., 1:-3, :, :])
                + -1 / 12 * (a[..., 4:, :, :] + a[..., :-4, :, :])
            )
            * rdz2,
            (0, 0, 0, 0, 2, 2),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                -49 / 18 * a[..., 3:-3, :, :]
                + 3 / 2 * (a[..., 4:-2, :, :] + a[..., 2:-4, :, :])
                + -3 / 20 * (a[..., 5:-1, :, :] + a[..., 1:-5, :, :])
                + 1 / 90 * (a[..., 6:, :, :] + a[..., :-6, :, :])
            )
            * rdz2,
            (0, 0, 0, 0, 3, 3),
        )
    return torch.nn.functional.pad(
        (
            -205 / 72 * a[..., 4:-4, :, :]
            + 8 / 5 * (a[..., 5:-3, :, :] + a[..., 3:-5, :, :])
            + -1 / 5 * (a[..., 6:-2, :, :] + a[..., 2:-6, :, :])
            + 8 / 315 * (a[..., 7:-1, :, :] + a[..., 1:-7, :, :])
            + -1 / 560 * (a[..., 8:, :, :] + a[..., :-8, :, :])
        )
        * rdz2,
        (0, 0, 0, 0, 4, 4),
    )


def diffy2(a: torch.Tensor, accuracy: int, rdy2: float) -> torch.Tensor:
    """Calculates the second derivative in the y-direction."""
    if accuracy == 2:
        return torch.nn.functional.pad(
            (-2 * a[..., 1:-1, :] + 1 * (a[..., 2:, :] + a[..., :-2, :])) * rdy2,
            (0, 0, 1, 1),
        )
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                -5 / 2 * a[..., 2:-2, :]
                + 4 / 3 * (a[..., 3:-1, :] + a[..., 1:-3, :])
                + -1 / 12 * (a[..., 4:, :] + a[..., :-4, :])
            )
            * rdy2,
            (0, 0, 2, 2),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                -49 / 18 * a[..., 3:-3, :]
                + 3 / 2 * (a[..., 4:-2, :] + a[..., 2:-4, :])
                + -3 / 20 * (a[..., 5:-1, :] + a[..., 1:-5, :])
                + 1 / 90 * (a[..., 6:, :] + a[..., :-6, :])
            )
            * rdy2,
            (0, 0, 3, 3),
        )
    return torch.nn.functional.pad(
        (
            -205 / 72 * a[..., 4:-4, :]
            + 8 / 5 * (a[..., 5:-3, :] + a[..., 3:-5, :])
            + -1 / 5 * (a[..., 6:-2, :] + a[..., 2:-6, :])
            + 8 / 315 * (a[..., 7:-1, :] + a[..., 1:-7, :])
            + -1 / 560 * (a[..., 8:, :] + a[..., :-8, :])
        )
        * rdy2,
        (0, 0, 4, 4),
    )


def diffx2(a: torch.Tensor, accuracy: int, rdx2: float) -> torch.Tensor:
    """Calculates the second derivative in the x-direction."""
    if accuracy == 2:
        return torch.nn.functional.pad(
            (-2 * a[..., 1:-1] + 1 * (a[..., 2:] + a[..., :-2])) * rdx2, (1, 1)
        )
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                -5 / 2 * a[..., 2:-2]
                + 4 / 3 * (a[..., 3:-1] + a[..., 1:-3])
                + -1 / 12 * (a[..., 4:] + a[..., :-4])
            )
            * rdx2,
            (2, 2),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                -49 / 18 * a[..., 3:-3]
                + 3 / 2 * (a[..., 4:-2] + a[..., 2:-4])
                + -3 / 20 * (a[..., 5:-1] + a[..., 1:-5])
                + 1 / 90 * (a[..., 6:] + a[..., :-6])
            )
            * rdx2,
            (3, 3),
        )
    return torch.nn.functional.pad(
        (
            -205 / 72 * a[..., 4:-4]
            + 8 / 5 * (a[..., 5:-3] + a[..., 3:-5])
            + -1 / 5 * (a[..., 6:-2] + a[..., 2:-6])
            + 8 / 315 * (a[..., 7:-1] + a[..., 1:-7])
            + -1 / 560 * (a[..., 8:] + a[..., :-8])
        )
        * rdx2,
        (4, 4),
    )


def diff1(
    wf: torch.Tensor, dim: int, accuracy: int, rdx: float, ndim: int
) -> torch.Tensor:
    """Perform first-order differentiation."""
    if ndim - dim == 3:
        return diffz1(wf, accuracy, rdx)
    if ndim - dim == 2:
        return diffy1(wf, accuracy, rdx)
    return diffx1(wf, accuracy, rdx)


def diff2(
    wf: torch.Tensor, dim: int, accuracy: int, rdx2: float, ndim: int
) -> torch.Tensor:
    """Perform second-order differentiation."""
    if ndim - dim == 3:
        return diffz2(wf, accuracy, rdx2)
    if ndim - dim == 2:
        return diffy2(wf, accuracy, rdx2)
    return diffx2(wf, accuracy, rdx2)
