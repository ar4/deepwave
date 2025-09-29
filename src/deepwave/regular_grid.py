"""Utility functions for regular grid PML setup.

This module provides functions to set up Perfectly Matched Layers (PML)
profiles for wave propagation simulations on a regular grid.
"""

from typing import List

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
    ny: int,
    nx: int,
) -> List[torch.Tensor]:
    """Sets up PML profiles for a regular grid.

    Args:
        pml_width: A list of integers specifying the width of the PML
            on each side (top, bottom, left, right).
        accuracy: The finite-difference accuracy order.
        fd_pad: A list of integers specifying the padding for finite-difference.
        dt: The time step.
        grid_spacing: A list of floats specifying the grid spacing in
            y and x directions.
        max_vel: The maximum velocity in the model.
        dtype: The data type of the tensors (e.g., torch.float32).
        device: The device on which the tensors will be (e.g., 'cuda', 'cpu').
        pml_freq: The PML frequency.
        ny: The number of grid points in the y direction.
        nx: The number of grid points in the x direction.

    Returns:
        A list of torch.Tensors representing the PML profiles (ay, ax, by, bx,
        dbydy, dbxdx).

    """
    pml_start: List[float] = [
        fd_pad[0] + pml_width[0],
        ny - 1 - fd_pad[1] - pml_width[1],
        fd_pad[2] + pml_width[2],
        nx - 1 - fd_pad[3] - pml_width[3],
    ]
    max_pml = max(
        [
            pml_width[0] * grid_spacing[0],
            pml_width[1] * grid_spacing[0],
            pml_width[2] * grid_spacing[1],
            pml_width[3] * grid_spacing[1],
        ],
    )

    ay, by = deepwave.common.setup_pml(
        pml_width[:2],
        pml_start[:2],
        max_pml,
        dt,
        ny,
        max_vel,
        dtype,
        device,
        pml_freq,
    )
    ax, bx = deepwave.common.setup_pml(
        pml_width[2:],
        pml_start[2:],
        max_pml,
        dt,
        nx,
        max_vel,
        dtype,
        device,
        pml_freq,
    )
    dbydy = diffx1(
        by, accuracy, torch.tensor(1 / grid_spacing[0], dtype=dtype, device=device)
    )
    dbxdx = diffx1(
        bx, accuracy, torch.tensor(1 / grid_spacing[1], dtype=dtype, device=device)
    )
    ay = ay[None, :, None]
    ax = ax[None, None, :]
    by = by[None, :, None]
    bx = bx[None, None, :]
    dbydy = dbydy[None, :, None]
    dbxdx = dbxdx[None, None, :]
    return [ay, ax, by, bx, dbydy, dbxdx]


def diffy1(a: torch.Tensor, accuracy: int, rdy: torch.Tensor) -> torch.Tensor:
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


def diffx1(a: torch.Tensor, accuracy: int, rdx: torch.Tensor) -> torch.Tensor:
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


def diffy2(a: torch.Tensor, accuracy: int, rdy2: torch.Tensor) -> torch.Tensor:
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


def diffx2(a: torch.Tensor, accuracy: int, rdx2: torch.Tensor) -> torch.Tensor:
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
