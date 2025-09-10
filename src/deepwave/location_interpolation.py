"""Interpolation of sources and receivers at arbitrary locations onto a grid."""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch

import deepwave

DEFAULT_EPS = 1e-5


def _get_hicks_for_one_location_dim(
    hicks_weight_cache: Dict[Tuple[int, int, int], torch.Tensor],
    location: float,
    halfwidth: int,
    beta: torch.Tensor,
    free_surface: List[bool],
    free_surface_loc: List[float],
    size: int,
    monopole: bool = True,
    eps: float = DEFAULT_EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate Hicks interpolation locations and weights for one dimension.

    Args:
        hicks_weight_cache: Cache for Hicks weights to avoid recomputing.
        location: The floating-point location to interpolate.
        halfwidth: Halfwidth of the interpolation window.
        beta: Kaiser window parameter.
        free_surface: List of two booleans indicating free surface presence at
            boundaries.
        free_surface_loc: List of two floats indicating free surface locations.
        size: Size of the grid in this dimension.
        monopole: If True, calculate for a monopole; otherwise, for a dipole.
        eps: Small value to prevent division by zero.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Locations and corresponding weights.

    """
    if not isinstance(halfwidth, int):
        raise TypeError("halfwidth must be an int.")
    if halfwidth < 1 or halfwidth > 10:
        raise RuntimeError("halfwidth must be in [1, 10].")
    if beta < 0:
        raise RuntimeError("beta must be non-negative.")
    if (
        not isinstance(free_surface_loc, List)
        or len(free_surface_loc) != 2
        or not all(isinstance(f, (int, float)) for f in free_surface_loc)
    ):
        raise RuntimeError("extent must be a list of two floats.")
    if (free_surface[0] or free_surface[1]) and size <= 0:
        raise RuntimeError("n_grid_points must be positive.")
    if monopole and abs(location - round(location)) < eps:
        locations = torch.tensor([location]).round().long().to(beta.device)
        weights = torch.ones(1, dtype=beta.dtype, device=beta.device)
    else:
        key = (int((location - int(location)) / eps), halfwidth, int(monopole))
        x = (
            torch.arange(
                -halfwidth + 1,
                halfwidth + 1,
                dtype=beta.dtype,
                device=beta.device,
            )
            - location
            + int(location)
        )
        locations = (location + x).long()

        if key in hicks_weight_cache:
            weights = hicks_weight_cache[key].clone()
        else:
            if monopole:
                weights = (
                    torch.sinc(x)
                    * torch.i0(beta * (1 - (x / halfwidth) ** 2).sqrt())
                    / torch.i0(beta)
                )
            else:
                weights = (
                    (torch.cos(math.pi * x) - torch.sinc(x))
                    / (x**2 + eps)
                    * x
                    * torch.i0(beta * (1 - (x / halfwidth) ** 2).sqrt())
                    / torch.i0(beta)
                )
            hicks_weight_cache[key] = weights
        if free_surface[0] and locations[0].item() < free_surface_loc[0]:
            # idx0: first index at, or on model side of, free surface
            # idx1: first index on model side of free surface
            idx0 = math.ceil(free_surface_loc[0] - locations[0].item())
            idx1 = idx0 + int(
                math.isclose(round(free_surface_loc[0]), free_surface_loc[0]),
            )
            locations = locations[idx0:]
            flipped_weights = weights[:idx1].flip(0)
            weights[idx0 : idx0 + len(flipped_weights)] -= flipped_weights
            weights = weights[idx0:]
        if free_surface[1] and locations[-1].item() > free_surface_loc[1]:
            idx0 = (
                len(weights) - 1 - math.ceil(locations[-1].item() - free_surface_loc[1])
            )
            idx1 = idx0 - int(
                math.isclose(round(free_surface_loc[1]), free_surface_loc[1]),
            )
            locations = locations[: idx0 + 1]
            flipped_weights = weights[idx1 + 1 :].flip(0)
            weights[idx0 + 1 - len(flipped_weights) : idx0 + 1] -= flipped_weights
            weights = weights[: idx0 + 1]
    return locations, weights


def _get_hicks_locations_and_weights(
    locations: torch.Tensor,
    halfwidth: int,
    beta: torch.Tensor,
    free_surfaces: List[bool],
    free_surface_locs: List[float],
    model_shape: List[int],
    monopole: Union[torch.Tensor, bool] = True,
    dipole_dim: Union[torch.Tensor, int] = 0,
    eps: float = DEFAULT_EPS,
) -> Tuple[torch.Tensor, List[List[List[int]]], List[List[List[torch.Tensor]]]]:
    """Calculate Hicks interpolation locations and weights for all shots/sources.

    Args:
        locations: A torch.Tensor of original source/receiver locations.
        halfwidth: Halfwidth of the interpolation window.
        beta: Kaiser window parameter.
        free_surfaces: List of four booleans indicating free surface presence.
        free_surface_locs: List of four floats indicating free surface locations.
        model_shape: Shape of the model grid.
        monopole: Boolean or torch.Tensor indicating if sources/receivers are monopoles.
        dipole_dim: Integer or torch.Tensor indicating dipole orientation dimension.
        eps: Small value to prevent division by zero.

    Returns:
        Tuple[torch.Tensor, List[List[List[int]]], List[List[List[torch.Tensor]]]]:
            Interpolated locations, indices, and weights.

    """
    hicks_weight_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}
    n_shots, n_per_shot, _ = locations.shape
    hicks_locations_list: List[List[Tuple[int, int]]] = []
    hicks_idxs: List[List[List[int]]] = []
    weights: List[List[List[torch.Tensor]]] = []
    n_per_shot_hicks = 0
    for shotidx in range(n_shots):
        shot_location_idxs: List[List[int]] = []
        shot_weights: List[List[torch.Tensor]] = []
        n_hicks_locations = 0
        locations_dict: Dict[Tuple[int, int], int] = {}
        for i in range(n_per_shot):
            if isinstance(monopole, torch.Tensor):
                monopole_i = bool(monopole[shotidx, i].item())
            else:
                monopole_i = monopole
            if isinstance(dipole_dim, torch.Tensor):
                dipole_dim_i = int(dipole_dim[shotidx, i].item())
            else:
                dipole_dim_i = dipole_dim
            locations0, weights0 = _get_hicks_for_one_location_dim(
                hicks_weight_cache,
                float(locations[shotidx, i, 0].item()),
                halfwidth,
                beta,
                free_surfaces[:2],
                free_surface_locs[:2],
                model_shape[0],
                monopole_i or (not monopole_i and dipole_dim_i == 1),
                eps=eps,
            )
            locations1, weights1 = _get_hicks_for_one_location_dim(
                hicks_weight_cache,
                float(locations[shotidx, i, 1].item()),
                halfwidth,
                beta,
                free_surfaces[2:],
                free_surface_locs[2:],
                model_shape[1],
                monopole_i or (not monopole_i and dipole_dim_i == 0),
                eps=eps,
            )
            shot_weights.append([weights0, weights1])
            i_idxs: List[int] = []
            for loc in torch.cartesian_prod(locations0, locations1):
                loc_tuple = (loc[0].item(), loc[1].item())
                if loc_tuple in locations_dict:
                    i_idxs.append(locations_dict[loc_tuple])
                else:
                    locations_dict[loc_tuple] = n_hicks_locations
                    i_idxs.append(n_hicks_locations)
                    n_hicks_locations += 1
            shot_location_idxs.append(i_idxs)
        hicks_idxs.append(shot_location_idxs)
        weights.append(shot_weights)
        n_per_shot_hicks = max(n_per_shot_hicks, n_hicks_locations)
        hicks_locations_list.append(list(locations_dict.keys()))
    hicks_locations = (
        torch.ones(
            n_shots,
            n_per_shot_hicks,
            2,
            dtype=torch.long,
            device=locations.device,
        )
        * deepwave.common.IGNORE_LOCATION
    )
    for shotidx in range(n_shots):
        for i, loc in enumerate(hicks_locations_list[shotidx]):
            hicks_locations[shotidx, i] = torch.tensor(loc).to(locations.device)

    return hicks_locations, hicks_idxs, weights


def _check_shot_idxs(
    amplitudes: torch.Tensor,
    shot_idxs: Optional[torch.Tensor] = None,
) -> None:
    if shot_idxs is not None and shot_idxs.shape != (len(amplitudes),):
        raise RuntimeError("shot_idxs must have the same length as amplitudes")


class Hicks:
    """Location interpolation onto grid using method of Hicks.

    Hicks (2002, https://doi.org/10.1190/1.1451454) proposed
    using a Kaiser windowed sinc function to interpolate
    source and receiver locations onto a grid when they are
    not centred on a grid point. This class implements this
    approach. It can also be used to create dipole sources
    and receivers.

    Args:
        locations:
            A three dimensional torch.Tensor [shot, per_shot, 2] specifying
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
            edge of the grid is a free surface, in the order:
            [beginning of first dimension, end of first dimension,
            beginning of second dimension, end of second dimension].
            For example, `[True, False, False, True]` means that the beginning
            of the first dimension and end of the second dimension are
            free surfaces, while the other two edges are not.
            Defaults to no edges being free surfaces.
        model_shape:
            A list of two integers specifying the size of the grid. This
            is only used when the model contains free surfaces.
        monopole:
            A boolean or torch.Tensor of booleans (of shape [shot, per_shot])
            specifying whether the source/receiver is a monopole. If
            False, the point will be a dipole. Default True.
        dipole_dim:
            An integer or torch.Tensor of integers (of shape [shot, per_shot])
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

    def __init__(
        self,
        locations: torch.Tensor,
        halfwidth: int = 4,
        free_surfaces: Optional[List[bool]] = None,
        free_surface_locs: Optional[List[float]] = None,
        model_shape: Optional[List[int]] = None,
        monopole: Union[torch.Tensor, bool] = True,
        dipole_dim: Union[torch.Tensor, int] = 0,
        dtype: torch.dtype = torch.float,
        eps: float = DEFAULT_EPS,
    ) -> None:
        """Initializes the Hicks interpolation class.

        Args:
            locations: A three dimensional torch.Tensor [shot, per_shot, 2] specifying
                the locations to interpolate, provided as float values in
                units of cells relative to the origin of the grid.
            halfwidth: An integer specifying the halfwidth of the window that
                will be used to interpolate each point onto the grid.
            free_surfaces: A list of four booleans specifying whether the corresponding
                edge of the grid is a free surface.
            free_surface_locs: A list of four floats indicating free surface locations.
            model_shape: A list of two integers specifying the size of the grid.
            monopole: A boolean or torch.Tensor of booleans specifying whether
                the source/receiver is a monopole.
            dipole_dim: An integer or torch.Tensor of integers specifying the
                dimension in which the dipole is oriented.
            dtype: The datatype to use.
            eps: A small value to prevent division by zero.

        """
        if not isinstance(locations, torch.Tensor):
            raise TypeError("locations must be a torch.Tensor.")
        if locations.ndim != 3:
            raise RuntimeError("locations must have three dimensions.")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype.")
        if not isinstance(halfwidth, int):
            raise RuntimeError("halfwidth must be an integer")
        if halfwidth < 1 or halfwidth > 10:
            raise RuntimeError("halfwidth must be in [1, 10]")
        if free_surfaces is None:
            free_surfaces = [False, False, False, False]
        if free_surfaces is not None and (
            not isinstance(free_surfaces, list)
            or len(free_surfaces) != 4
            or any(not isinstance(f, bool) for f in free_surfaces)
        ):
            raise RuntimeError("free_surface must be a list of four bools")
        if any(free_surfaces) and model_shape is None:
            raise RuntimeError(
                "If there are free surfaces then model_shape must be specified",
            )
        if model_shape is not None:
            if not isinstance(model_shape, list) or len(model_shape) != 2:
                raise RuntimeError("model_shape must be a list with two int entries")
            try:
                model_shape = [int(v) for v in model_shape]
            except (ValueError, TypeError) as exc:
                raise TypeError(
                    "model_shape entries must be convertible to int",
                ) from exc
        if free_surface_locs is not None:
            if not isinstance(free_surface_locs, list) or len(free_surface_locs) != 4:
                raise RuntimeError("free_surface_locs must be a list of four floats")
            try:
                free_surface_locs = [float(v) for v in free_surface_locs]
            except (ValueError, TypeError) as exc:
                raise TypeError(
                    "free_surface_locs entries must be convertible to float",
                ) from exc
        if model_shape is None:
            model_shape = [-1, -1]
        if free_surface_locs is None:
            free_surface_locs = [-0.5, model_shape[0] - 0.5, -0.5, model_shape[1] - 0.5]
        if isinstance(monopole, torch.Tensor) and (
            monopole.shape[0] != locations.shape[0]
            or monopole.shape[1] != locations.shape[1]
        ):
            raise RuntimeError("monopole must have dimensions [shot, per_shot]")
        if isinstance(dipole_dim, torch.Tensor) and (
            dipole_dim.shape[0] != locations.shape[0]
            or dipole_dim.shape[1] != locations.shape[1]
        ):
            raise RuntimeError("dipole_dim must have dimensions [shot, per_shot]")
        betas = [0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64]
        beta = torch.tensor(betas[halfwidth - 1]).to(dtype).to(locations.device)
        self.locations = locations
        self.hicks_locations, self.idxs, self.weights = (
            _get_hicks_locations_and_weights(
                locations,
                halfwidth,
                beta,
                free_surfaces,
                free_surface_locs,
                model_shape,
                monopole,
                dipole_dim,
                eps,
            )
        )

    def get_locations(self, shot_idxs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get the interpolated locations.

        The interpolated locations can be provided to a Deepwave
        propagator as the source or receiver locations.

        Args:
            shot_idxs:
                A 1D torch.Tensor containing the indices of the shots that
                amplitudes are being provided for, within the locations
                provided at initialisation. This is useful if shots are
                being used in batches, so only some of the shots are
                needed now. Default (None) means all shots.

        Returns:
            A torch.Tensor containing the interpolated locations.

        """
        if shot_idxs is not None:
            return self.hicks_locations[shot_idxs]
        return self.hicks_locations

    def source(
        self,
        amplitudes: torch.Tensor,
        shot_idxs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate the amplitudes of the interpolated sources.

        Args:
            amplitudes:
                A torch.Tensor containing the amplitudes of the sources at the
                original locations.
            shot_idxs:
                A 1D torch.Tensor containing the indices of the shots that
                amplitudes are being provided for, within the locations
                provided at initialisation. This is useful if shots are
                being used in batches, so only some of the shots are
                needed now. Default (None) means all shots.

        Returns:
            The amplitudes of the interpolated sources, ready to be provided
            to a Deepwave propagator.

        """
        if not isinstance(amplitudes, torch.Tensor):
            raise TypeError("amplitudes must be a torch.Tensor.")
        if amplitudes.ndim != 3:
            raise RuntimeError("amplitudes must have three dimensions.")

        _check_shot_idxs(amplitudes, shot_idxs)
        n_shots, n_per_shot, nt = amplitudes.shape
        n_per_shot_hicks = self.hicks_locations.shape[1]
        # In the below I put `out` on the CPU because there seems to
        # be a bug (in PyTorch?) that causes the incorrect answer
        # otherwise
        out = torch.zeros(
            n_shots,
            n_per_shot_hicks,
            nt,
            dtype=amplitudes.dtype,
            device=torch.device("cpu"),
        )
        for shotidx in range(n_shots):
            if shot_idxs is not None:
                hicks_shotidx = int(shot_idxs[shotidx])
            else:
                hicks_shotidx = shotidx
            for i in range(n_per_shot):
                out[shotidx, self.idxs[hicks_shotidx][i], :] += (
                    amplitudes[shotidx, i][None]
                    * (
                        self.weights[hicks_shotidx][i][0].reshape(-1, 1)
                        * self.weights[hicks_shotidx][i][1].reshape(1, -1)
                    ).reshape(-1)[..., None]
                ).cpu()
        return out.to(amplitudes.device)

    def receiver(
        self,
        amplitudes: torch.Tensor,
        shot_idxs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert receiver amplitudes from interpolated to original locations.

        Args:
            amplitudes:
                A torch.Tensor containing the amplitudes recorded at interpolated
                receiver locations.
            shot_idxs:
                A 1D torch.Tensor containing the indices of the shots that
                amplitudes are being provided for, within the locations
                provided at initialisation. This is useful if shots are
                being used in batches, so only some of the shots are
                needed now. Default (None) means all shots.

        Returns:
            The amplitudes of receivers at the original locations.

        """
        if not isinstance(amplitudes, torch.Tensor):
            raise TypeError("amplitudes must be a torch.Tensor.")
        if amplitudes.ndim != 3:
            raise RuntimeError("amplitudes must have three dimensions.")

        _check_shot_idxs(amplitudes, shot_idxs)
        n_shots, _, nt = amplitudes.shape
        n_per_shot = self.locations.shape[1]
        out = torch.zeros(
            n_shots,
            n_per_shot,
            nt,
            dtype=amplitudes.dtype,
            device=amplitudes.device,
        )
        for shotidx in range(n_shots):
            if shot_idxs is not None:
                hicks_shotidx = int(shot_idxs[shotidx])
            else:
                hicks_shotidx = shotidx
            for i in range(n_per_shot):
                out[shotidx, i, :] = (
                    amplitudes[shotidx, self.idxs[hicks_shotidx][i]]
                    * (
                        self.weights[hicks_shotidx][i][0]
                        .to(amplitudes.device)
                        .reshape(-1, 1)
                        * self.weights[hicks_shotidx][i][1]
                        .to(amplitudes.device)
                        .reshape(1, -1)
                    ).reshape(-1)[..., None]
                ).sum(dim=0)
        return out
