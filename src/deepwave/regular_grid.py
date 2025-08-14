from typing import List, Tuple
import torch
from torch import Tensor
from deepwave.common import setup_pml, diff

def set_pml_profiles(pml_width: List[int], accuracy: int, fd_pad: List[int], dt: float, grid_spacing: List[float], max_vel: float, dtype: torch.dtype, device: torch.device, pml_freq: float, ny: int, nx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

    pml_start = [fd_pad[0] + pml_width[0],
                  ny - 1 - fd_pad[1] - pml_width[1],
                  fd_pad[2] + pml_width[2],
                  nx - 1 - fd_pad[3] - pml_width[3]]
    max_pml = max([pml_width[0] * grid_spacing[0], pml_width[1] * grid_spacing[0], pml_width[2] * grid_spacing[1], pml_width[3] * grid_spacing[1]])

    ay, by = \
            setup_pml(pml_width[:2], pml_start[:2], max_pml, dt, grid_spacing[0], ny, max_vel,
                  dtype, device, pml_freq)
    ax, bx = \
            setup_pml(pml_width[2:], pml_start[2:], max_pml, dt, grid_spacing[1], nx, max_vel,
                  dtype, device, pml_freq)
    #ay[fd_pad[0] + pml_width[0]:ny-(fd_pad[1] + pml_width[1])] = 0
    #ax[fd_pad[2] + pml_width[2]:nx-(fd_pad[3] + pml_width[3])] = 0
    dbydy = diff(by, accuracy, grid_spacing[0])
    dbxdx = diff(bx, accuracy, grid_spacing[1])
    ay = ay[None, :, None]
    ax = ax[None, None, :]
    by = by[None, :, None]
    bx = bx[None, None, :]
    dbydy = dbydy[None, :, None]
    dbxdx = dbxdx[None, None, :]
    return [ay, ax, by, bx, dbydy, dbxdx]
