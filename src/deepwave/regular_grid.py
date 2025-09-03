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
    dbydy = deepwave.common.diff(by, accuracy, grid_spacing[0])
    dbxdx = deepwave.common.diff(bx, accuracy, grid_spacing[1])
    ay = ay[None, :, None]
    ax = ax[None, None, :]
    by = by[None, :, None]
    bx = bx[None, None, :]
    dbydy = dbydy[None, :, None]
    dbxdx = dbxdx[None, None, :]
    return [ay, ax, by, bx, dbydy, dbxdx]
