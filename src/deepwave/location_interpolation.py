"Interpolation of points at arbitrary locations onto a grid."
from typing import List, Optional, Union, Dict, Tuple
import torch
from torch import Tensor
DEFAULT_EPS = 1e-5


def _get_hicks_for_one_location_dim(
        hicks_weight_cache: Dict[Tensor, Tensor],
        location: float, halfwidth: int, beta: Tensor,
        free_surface: List[bool], size: int,
        monopole: bool = True,
        eps: float = DEFAULT_EPS) -> Tuple[Tensor, Tensor]:
    if monopole and abs(location - round(location)) < eps:
        locations = torch.tensor([location]).round().long().to(beta.device)
        weights = torch.ones(1, dtype=beta.dtype, device=beta.device)
    else:
        key = torch.tensor([int((location - int(location)) / eps),
                            halfwidth, int(monopole)])
        x = (torch.arange(-halfwidth + 1, halfwidth + 1, dtype=beta.dtype,
                          device=beta.device) -
             location + int(location))
        locations = (location + x).long()

        if key in hicks_weight_cache:
            weights = hicks_weight_cache[key]
        else:
            if monopole:
                weights = (
                    torch.sinc(x) *
                    torch.i0(beta * (1 - (x / halfwidth)**2).sqrt()) /
                    torch.i0(beta)
                )
            else:
                weights = (
                    (torch.cos(torch.pi * x) - torch.sinc(x)) /
                    (x**2 + eps) * x *
                    torch.i0(beta * (1 - (x / halfwidth)**2).sqrt()) /
                    torch.i0(beta)
                )
            hicks_weight_cache[key] = weights
        if free_surface[0] and locations[0].item() < 0:
            idx0 = int(-locations[0].item())
            locations = locations[idx0:]
            flipped_weights = weights[:idx0-1].flip(0)
            weights[idx0:idx0 + len(flipped_weights)] -= flipped_weights
            weights = weights[idx0:]
        if free_surface[1] and locations[-1].item() >= size:
            idxe = size - int(locations[-1].item()) - 1
            locations = locations[:idxe]
            flipped_weights = weights[idxe+1:].flip(0)
            weights[idxe - len(flipped_weights):idxe] -= flipped_weights
            weights = weights[:idxe]
    return locations, weights


def _get_hicks_locations_and_weights(
        locations: Tensor, halfwidth: int, beta: Tensor,
        free_surfaces: List[bool], model_shape: List[int],
        monopole: Union[Tensor, bool] = True,
        dipole_dim: Union[Tensor, int] = 0,
        eps: float = DEFAULT_EPS
        ) -> Tuple[Tensor, List[List[List[int]]], List[List[List[Tensor]]]]:
    hicks_weight_cache: Dict[Tensor, Tensor] = {}
    n_shots, n_per_shot, _ = locations.shape
    hicks_locations_list: List[List[Tensor]] = []
    hicks_idxs: List[List[List[int]]] = []
    weights: List[List[List[Tensor]]] = []
    n_per_shot_hicks = 0
    for shotidx in range(n_shots):
        shot_location_idxs: List[List[int]] = []
        shot_weights: List[List[Tensor]] = []
        n_hicks_locations = 0
        locations_dict: Dict[Tensor, int] = {}
        for i in range(n_per_shot):
            if isinstance(monopole, Tensor):
                monopole_i = bool(monopole[shotidx, i].item())
            else:
                monopole_i = monopole
            if isinstance(dipole_dim, Tensor):
                dipole_dim_i = int(dipole_dim[shotidx, i].item())
            else:
                dipole_dim_i = dipole_dim
            locations0, weights0 = \
                _get_hicks_for_one_location_dim(
                    hicks_weight_cache,
                    float(locations[shotidx, i, 0].item()), halfwidth, beta,
                    free_surfaces[:2], model_shape[0],
                    monopole_i or (not monopole_i and dipole_dim_i == 1),
                    eps=eps
                )
            locations1, weights1 = \
                _get_hicks_for_one_location_dim(
                    hicks_weight_cache,
                    float(locations[shotidx, i, 1].item()), halfwidth, beta,
                    free_surfaces[2:], model_shape[1],
                    monopole_i or (not monopole_i and dipole_dim_i == 0),
                    eps=eps
                )
            shot_weights.append([weights0, weights1])
            i_idxs: List[int] = []
            for loc in torch.cartesian_prod(locations0, locations1):
                if loc in locations_dict:
                    i_idxs.append(locations_dict[loc])
                else:
                    locations_dict[loc] = n_hicks_locations
                    i_idxs.append(n_hicks_locations)
                    n_hicks_locations += 1
            shot_location_idxs.append(i_idxs)
        hicks_idxs.append(shot_location_idxs)
        weights.append(shot_weights)
        n_per_shot_hicks = max(
            n_per_shot_hicks,
            n_hicks_locations
        )
        hicks_locations_list.append(list(locations_dict.keys()))
    hicks_locations = torch.zeros(
        n_shots, n_per_shot_hicks, 2,
        dtype=torch.long,
        device=locations.device
    )
    for shotidx in range(n_shots):
        for i, loc in enumerate(hicks_locations_list[shotidx]):
            hicks_locations[shotidx, i] = loc

    return hicks_locations, hicks_idxs, weights


def _check_shot_idxs(amplitudes: Tensor,
                     shot_idxs: Optional[Tensor] = None) -> None:
    if (shot_idxs is not None and shot_idxs.shape != (len(amplitudes),)):
        raise RuntimeError("shot_idxs must have the same length "
                           "as amplitudes")


class Hicks:
    """Location interpolation onto grid using method of Hicks

    Hicks (2002, https://doi.org/10.1190/1.1451454) proposed
    using a Kaiser windowed sinc function to interpolate
    source and receiver locations onto a grid when they are
    not centred on a grid point. This class implements this
    approach. It can also be used to create dipole sources
    and receivers.

    Args:
        locations:
            A three dimensional Tensor [shot, per_shot, 2] specifying
            the locations to interpolate, provided as float values in
            units of cells relative to the origin of the grid, so
            [[[0.5, 0.5]]] corresponds to the point half a cell from
            the origin in both dimensions.
        halfwidth:
            An integer specifying the halfwidth of the window that
            will be used to interpolate each point onto the grid. A
            halfwidth of 4 (the default) means that an 8x8 window will
            be used. Possible values are in [1, 10].
        free_surfaces:
            A list of four booleans specifying whether the corresponding
            edge of the grid is a free surface, in the order [beginning
            of first dimension, end of first dimension, beginning of
            second dimension, end of second dimension].
            [True, False, False, True] therefore means that the beginning
            of the first dimension and end of the second dimension are
            free surfaces, while the other two edges are not, for example.
            The default is for no edges to be free surfaces.
        model_shape:
            A list of two integers specifying the size of the grid. This
            is only used when the model contains free surfaces.
        monopole:
            A boolean or Tensor of booleans (of shape [shot, per_shot])
            specifying whether the source/receiver is a monopole. If
            False, the point will be a dipole. Default True.
        dipole_dim:
            An integer or Tensor of integers (of shape [shot, per_shot])
            of value 0 or 1 specifying the dimension in which the
            dipole is oriented. This is only used for points that are
            dipoles (not monopoles). Default 0.
        dtype:
            The datatype to use. Default torch.float.
        eps:
            A small value to prevent division by zero. Points closer to
            a grid cell centre than this will be rounded to the grid cell.
            Default 1e-5.
    """
    def __init__(self, locations: Tensor,
                 halfwidth: int = 4,
                 free_surfaces: Optional[List[bool]] = None,
                 model_shape: Optional[List[int]] = None,
                 monopole: Union[Tensor, bool] = True,
                 dipole_dim: Union[Tensor, int] = 0,
                 dtype: torch.dtype = torch.float,
                 eps: float = DEFAULT_EPS):

        if locations.ndim != 3:
            raise RuntimeError("Locations should have three dimensions")
        if not isinstance(halfwidth, int):
            raise RuntimeError("halfwidth must be an integer")
        if halfwidth < 1 or halfwidth > 10:
            raise RuntimeError("halfwidth must be in [1, 10]")
        if free_surfaces is None:
            free_surfaces = [False, False, False, False]
        if any(free_surfaces) and model_shape is None:
            raise RuntimeError("If there are free surfaces then model_shape "
                               "must be specified")
        if model_shape is None:
            model_shape = [-1, -1]
        if (isinstance(monopole, Tensor) and
            (monopole.shape[0] != locations.shape[0] or
             monopole.shape[1] != locations.shape[1])):
            raise RuntimeError("monopole must have dimensions "
                               "[shot, per_shot]")
        if (isinstance(dipole_dim, Tensor) and
            (dipole_dim.shape[0] != locations.shape[0] or
             dipole_dim.shape[1] != locations.shape[1])):
            raise RuntimeError("dipole_dim must have dimensions "
                               "[shot, per_shot]")
        betas = [0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64]
        beta = (torch.tensor(betas[halfwidth - 1]).to(dtype)
                .to(locations.device))
        self.locations = locations
        self.hicks_locations, self.idxs, self.weights = \
            _get_hicks_locations_and_weights(
                locations, halfwidth, beta, free_surfaces, model_shape,
                monopole, dipole_dim, eps
            )

    def get_locations(self, shot_idxs: Optional[Tensor] = None) -> Tensor:
        """Get the interpolated locations.

        The interpolated locations can be provided to a Deepwave
        propagator as the source or receiver locations.

        Args:
            shot_idxs:
                A 1D Tensor containing the indices of the shots that
                amplitudes are being provided for, within the locations
                provided at initialisation. This is useful if shots are
                being used in batches, so only some of the shots are
                needed now. Default (None) means all shots.

        Returns:
            A Tensor containing the interpolated locations.
        """
        if shot_idxs is not None:
            return self.hicks_locations[shot_idxs]
        return self.hicks_locations

    def source(self, amplitudes: Tensor,
               shot_idxs: Optional[Tensor] = None) -> Tensor:
        """Calculate the amplitudes of the interpolated sources.

        Args:
            amplitudes:
                A Tensor containing the amplitudes of the sources at the
                original locations.
            shot_idxs:
                A 1D Tensor containing the indices of the shots that
                amplitudes are being provided for, within the locations
                provided at initialisation. This is useful if shots are
                being used in batches, so only some of the shots are
                needed now. Default (None) means all shots.

        Returns:
            The amplitudes of the interpolated sources, ready to be provided
            to a Deepwave propagator.
        """

        _check_shot_idxs(amplitudes, shot_idxs)
        n_shots, n_per_shot, nt = amplitudes.shape
        n_per_shot_hicks = self.hicks_locations.shape[1]
        out = torch.zeros(n_shots, n_per_shot_hicks, nt,
                          dtype=amplitudes.dtype,
                          device=amplitudes.device)
        for shotidx in range(n_shots):
            if shot_idxs is not None:
                hicks_shotidx = int(shot_idxs[shotidx])
            else:
                hicks_shotidx = shotidx
            for i in range(n_per_shot):
                out[shotidx, self.idxs[hicks_shotidx][i], :] += (
                    amplitudes[shotidx, i][None] *
                    (self.weights[hicks_shotidx][i][0].reshape(-1, 1) *
                     self.weights[hicks_shotidx][i][1].reshape(1, -1))
                    .reshape(-1)[..., None]
                )
        return out

    def receiver(self, amplitudes: Tensor,
                 shot_idxs: Optional[Tensor] = None) -> Tensor:
        """Convert receiver amplitudes from interpolated to original locations.

        Args:
            amplitudes:
                A Tensor containing the amplitudes recorded at interpolated
                receiver locations.
            shot_idxs:
                A 1D Tensor containing the indices of the shots that
                amplitudes are being provided for, within the locations
                provided at initialisation. This is useful if shots are
                being used in batches, so only some of the shots are
                needed now. Default (None) means all shots.

        Returns:
            The amplitudes of receivers at the original locations.
        """
        _check_shot_idxs(amplitudes, shot_idxs)
        n_shots, _, nt = amplitudes.shape
        n_per_shot = self.locations.shape[1]
        out = torch.zeros(n_shots, n_per_shot, nt,
                          dtype=amplitudes.dtype,
                          device=amplitudes.device)
        for shotidx in range(n_shots):
            if shot_idxs is not None:
                hicks_shotidx = int(shot_idxs[shotidx])
            else:
                hicks_shotidx = shotidx
            for i in range(n_per_shot):
                out[shotidx, i, :] = (
                    amplitudes[shotidx, self.idxs[hicks_shotidx][i]] *
                    (self.weights[hicks_shotidx][i][0].reshape(-1, 1) *
                     self.weights[hicks_shotidx][i][1].reshape(1, -1))
                    .reshape(-1)[..., None]
                ).sum(dim=0)
        return out
