from typing import List, Tuple
import torch
from torch import Tensor
from deepwave.common import setup_pml

def set_pml_profiles(pml_width: List[int], accuracy: int, fd_pad: List[int], dt: float, grid_spacing: List[float], max_vel: float, dtype: torch.dtype, device: torch.device, pml_freq: float, ny: int, nx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    pml_start = [pml_width[0],
                  ny - 1 - pml_width[1],
                  pml_width[2],
                  nx - 1 - pml_width[3]]
    max_pml = max([pml_width[0] * grid_spacing[0], pml_width[1] * grid_spacing[0], pml_width[2] * grid_spacing[1], pml_width[3] * grid_spacing[1]])

    ay, by = \
            setup_pml(pml_width[:2], pml_start[:2], max_pml, dt, grid_spacing[0], ny, max_vel,
                  dtype, device, pml_freq, start=-0.5)
    ax, bx = \
            setup_pml(pml_width[2:], pml_start[2:], max_pml, dt, grid_spacing[1], nx, max_vel,
                  dtype, device, pml_freq, start=0.0)

    ayh, byh = \
            setup_pml(pml_width[:2], pml_start[:2], max_pml, dt, grid_spacing[0], ny, max_vel,
                  dtype, device, pml_freq, start=0.0)
    axh, bxh = \
            setup_pml(pml_width[2:], pml_start[2:], max_pml, dt, grid_spacing[1], nx, max_vel,
                  dtype, device, pml_freq, start=0.5)

    return [ay, ayh, ax, axh, by, byh, bx, bxh]
