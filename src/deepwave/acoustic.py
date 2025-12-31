"""Variable-density acoustic wave propagation module for Deepwave.

Implements variable-density acoustic wave equation propagation using a
velocity-pressure formulation on a staggered grid with finite differences
in time (2nd order) and space (user-selectable order: 2, 4, 6, or 8).
Supports CPML boundaries and adjoint mode for gradient computation.

The equations solved are:
    rho * d(v)/dt = -grad(p)
    1/K * d(p)/dt = -div(v) + s

where p is pressure, v is particle velocity, rho is density, K is bulk modulus,
and s is the source term (volume injection rate).

The grid uses a staggered layout:
    - Pressure p and bulk modulus K are at integer grid points (y, x).
    - Velocity v and density rho are at half-integer grid points.
      e.g. in 2D: vy at (y+1/2, x), vx at (y, x+1/2).

    Required inputs: wavespeed model (`v`), density model (`rho`),
grid cell size (`grid_spacing`), time step (`dt`), and either a source term
or number of time steps (`nt`).
Outputs: final wavefields (pressure, velocities, PML memory variables) and
receiver amplitudes (empty if no receivers).
All outputs are differentiable with respect to float torch.Tensor inputs.
"""

from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, cast

import torch

import deepwave.backend_utils
import deepwave.common
import deepwave.staggered_grid


def prepare_models(v: torch.Tensor, rho: torch.Tensor) -> List[torch.Tensor]:
    """Prepares models for the staggered grid acoustic propagator.

    Computes the bulk modulus K at integer nodes and buoyancy (1/rho) at
    half-integer nodes.

    Args:
        v: Wavespeed model.
        rho: Density model.

    Returns:
        List containing [K, buoyancy_z, buoyancy_y, buoyancy_x] (depending on dim).
    """
    ndim = v.ndim - 1
    rfmax = 1 / torch.finfo(v.dtype).max ** (1 / 2)
    models = []

    # Bulk modulus K = rho * v^2
    # Avoid division by zero
    k = rho * v**2
    models.append(k)

    # Buoyancy (inverse of arithmetic mean of density)
    # Arithmetic mean: rho[i+1/2] = (rho[i] + rho[i+1])/2
    # => buoyancy[i+1/2] = 2/(rho[i] + rho[i+1])
    if ndim >= 3:
        rho_z = torch.nn.functional.pad(
            (rho[..., :-1, :, :] + rho[..., 1:, :, :]) / 2, (0, 0, 0, 0, 0, 1)
        )
        mask = rfmax < rho_z.abs()
        rho_z_safe = torch.where(mask, rho_z, torch.ones_like(rho_z))
        buoyancy_z = torch.where(mask, 1 / rho_z_safe, torch.zeros_like(rho))
        models.append(buoyancy_z)
        del rho_z, mask, rho_z_safe, buoyancy_z

    if ndim >= 2:
        rho_y = torch.nn.functional.pad(
            (rho[..., :-1, :] + rho[..., 1:, :]) / 2, (0, 0, 0, 1)
        )
        mask = rfmax < rho_y.abs()
        rho_y_safe = torch.where(mask, rho_y, torch.ones_like(rho_y))
        buoyancy_y = torch.where(mask, 1 / rho_y_safe, torch.zeros_like(rho))
        models.append(buoyancy_y)
        del rho_y, mask, rho_y_safe, buoyancy_y

    rho_x = torch.nn.functional.pad((rho[..., :-1] + rho[..., 1:]) / 2, (0, 1))
    mask = rfmax < rho_x.abs()
    rho_x_safe = torch.where(mask, rho_x, torch.ones_like(rho_x))
    buoyancy_x = torch.where(mask, 1 / rho_x_safe, torch.zeros_like(rho))
    models.append(buoyancy_x)

    return models


class Acoustic(torch.nn.Module):
    """Convenience nn.Module wrapper for acoustic wave propagation.

    Stores `v`, `rho` and `grid_spacing`. Gradients do not propagate to
    the provided wavespeed/density. Use the module's attributes to access them.
    """

    def __init__(
        self,
        v: torch.Tensor,
        rho: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        v_requires_grad: bool = False,
        rho_requires_grad: bool = False,
        storage_mode: Literal["device", "cpu", "disk", "none"] = "device",
        storage_path: str = ".",
        storage_compression: bool = False,
    ) -> None:
        """Initializes the Acoustic propagator module.

        Args:
            v: A torch.Tensor containing the wavespeed model.
            rho: A torch.Tensor containing the density model.
            grid_spacing: The spatial grid cell size. It can be a single number
                that will be used for all dimensions, or a number for each
                dimension.
            v_requires_grad: A bool specifying whether gradients will be
                computed for `v`. Defaults to False.
            rho_requires_grad: A bool specifying whether gradients will be
                computed for `rho`. Defaults to False.
            storage_mode: A string specifying the storage mode for intermediate
                data. One of "device", "cpu", "disk", or "none". Defaults to "device".
            storage_path: A string specifying the path for disk storage.
                Defaults to ".".
            storage_compression: A bool specifying whether to use compression
                for intermediate data. Defaults to False.

        """
        super().__init__()
        if not isinstance(v_requires_grad, bool):
            raise TypeError(
                f"v_requires_grad must be bool, got {type(v_requires_grad).__name__}",
            )
        if not isinstance(v, torch.Tensor):
            raise TypeError("v must be a torch.Tensor.")
        if not isinstance(rho_requires_grad, bool):
            raise TypeError(
                f"rho_requires_grad must be bool, "
                f"got {type(rho_requires_grad).__name__}",
            )
        if not isinstance(rho, torch.Tensor):
            raise TypeError("rho must be a torch.Tensor.")
        self.v = torch.nn.Parameter(v, requires_grad=v_requires_grad)
        self.rho = torch.nn.Parameter(rho, requires_grad=rho_requires_grad)
        self.grid_spacing = grid_spacing
        self.storage_mode = storage_mode
        self.storage_path = storage_path
        self.storage_compression = storage_compression

    def forward(
        self,
        dt: float,
        source_amplitudes_p: Optional[torch.Tensor] = None,
        source_locations_p: Optional[torch.Tensor] = None,
        source_amplitudes_z: Optional[torch.Tensor] = None,
        source_locations_z: Optional[torch.Tensor] = None,
        source_amplitudes_y: Optional[torch.Tensor] = None,
        source_locations_y: Optional[torch.Tensor] = None,
        source_amplitudes_x: Optional[torch.Tensor] = None,
        source_locations_x: Optional[torch.Tensor] = None,
        receiver_locations_p: Optional[torch.Tensor] = None,
        receiver_locations_z: Optional[torch.Tensor] = None,
        receiver_locations_y: Optional[torch.Tensor] = None,
        receiver_locations_x: Optional[torch.Tensor] = None,
        accuracy: int = 4,
        pml_width: Union[int, Sequence[int]] = 20,
        pml_freq: Optional[float] = None,
        max_vel: Optional[float] = None,
        survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
        pressure_0: Optional[torch.Tensor] = None,
        vz_0: Optional[torch.Tensor] = None,
        vy_0: Optional[torch.Tensor] = None,
        vx_0: Optional[torch.Tensor] = None,
        phi_z_0: Optional[torch.Tensor] = None,
        phi_y_0: Optional[torch.Tensor] = None,
        phi_x_0: Optional[torch.Tensor] = None,
        psi_z_0: Optional[torch.Tensor] = None,
        psi_y_0: Optional[torch.Tensor] = None,
        psi_x_0: Optional[torch.Tensor] = None,
        origin: Optional[Sequence[int]] = None,
        nt: Optional[int] = None,
        model_gradient_sampling_interval: int = 1,
        freq_taper_frac: float = 0.0,
        time_pad_frac: float = 0.0,
        time_taper: bool = False,
        forward_callback: Optional[deepwave.common.Callback] = None,
        backward_callback: Optional[deepwave.common.Callback] = None,
        callback_frequency: int = 1,
        python_backend: Union[Literal["eager", "jit", "compile"], bool] = False,
    ) -> List[torch.Tensor]:
        """Performs forward propagation/modelling.

        See :func:`acoustic` for details.
        """
        return acoustic(
            self.v,
            self.rho,
            self.grid_spacing,
            dt,
            source_amplitudes_p=source_amplitudes_p,
            source_locations_p=source_locations_p,
            source_amplitudes_z=source_amplitudes_z,
            source_locations_z=source_locations_z,
            source_amplitudes_y=source_amplitudes_y,
            source_locations_y=source_locations_y,
            source_amplitudes_x=source_amplitudes_x,
            source_locations_x=source_locations_x,
            receiver_locations_p=receiver_locations_p,
            receiver_locations_z=receiver_locations_z,
            receiver_locations_y=receiver_locations_y,
            receiver_locations_x=receiver_locations_x,
            accuracy=accuracy,
            pml_width=pml_width,
            pml_freq=pml_freq,
            max_vel=max_vel,
            survey_pad=survey_pad,
            pressure_0=pressure_0,
            vz_0=vz_0,
            vy_0=vy_0,
            vx_0=vx_0,
            phi_z_0=phi_z_0,
            phi_y_0=phi_y_0,
            phi_x_0=phi_x_0,
            psi_z_0=psi_z_0,
            psi_y_0=psi_y_0,
            psi_x_0=psi_x_0,
            origin=origin,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
            forward_callback=forward_callback,
            backward_callback=backward_callback,
            callback_frequency=callback_frequency,
            python_backend=python_backend,
            storage_mode=self.storage_mode,
            storage_path=self.storage_path,
            storage_compression=self.storage_compression,
        )


def acoustic(
    v: torch.Tensor,
    rho: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitudes_p: Optional[torch.Tensor] = None,
    source_locations_p: Optional[torch.Tensor] = None,
    source_amplitudes_z: Optional[torch.Tensor] = None,
    source_locations_z: Optional[torch.Tensor] = None,
    source_amplitudes_y: Optional[torch.Tensor] = None,
    source_locations_y: Optional[torch.Tensor] = None,
    source_amplitudes_x: Optional[torch.Tensor] = None,
    source_locations_x: Optional[torch.Tensor] = None,
    receiver_locations_p: Optional[torch.Tensor] = None,
    receiver_locations_z: Optional[torch.Tensor] = None,
    receiver_locations_y: Optional[torch.Tensor] = None,
    receiver_locations_x: Optional[torch.Tensor] = None,
    accuracy: int = 4,
    pml_width: Union[int, Sequence[int]] = 20,
    pml_freq: Optional[float] = None,
    max_vel: Optional[float] = None,
    survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
    pressure_0: Optional[torch.Tensor] = None,
    vz_0: Optional[torch.Tensor] = None,
    vy_0: Optional[torch.Tensor] = None,
    vx_0: Optional[torch.Tensor] = None,
    phi_z_0: Optional[torch.Tensor] = None,
    phi_y_0: Optional[torch.Tensor] = None,
    phi_x_0: Optional[torch.Tensor] = None,
    psi_z_0: Optional[torch.Tensor] = None,
    psi_y_0: Optional[torch.Tensor] = None,
    psi_x_0: Optional[torch.Tensor] = None,
    origin: Optional[Sequence[int]] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    forward_callback: Optional[deepwave.common.Callback] = None,
    backward_callback: Optional[deepwave.common.Callback] = None,
    callback_frequency: int = 1,
    python_backend: Union[Literal["eager", "jit", "compile"], bool] = False,
    storage_mode: Literal["device", "cpu", "disk", "none"] = "device",
    storage_path: str = ".",
    storage_compression: bool = False,
) -> List[torch.Tensor]:
    """Acoustic wave propagation (functional interface).

    This function performs forward modelling with the variable-density acoustic
    wave equation on a staggered grid.

    Args:
        v: Wavespeed model.
        rho: Density model.
        grid_spacing: Spatial grid cell size.
        dt: Time step interval.
        source_amplitudes_p: Source amplitudes (volume injection rate).
        source_locations_p: Source locations.
        source_amplitudes_z: Source amplitudes (z-force).
        source_locations_z: Source locations (z-force).
        source_amplitudes_y: Source amplitudes (y-force).
        source_locations_y: Source locations (y-force).
        source_amplitudes_x: Source amplitudes (x-force).
        source_locations_x: Source locations (x-force).
        receiver_locations_p: Pressure receiver locations.
        receiver_locations_z: Z-component velocity receiver locations.
        receiver_locations_y: Y-component velocity receiver locations.
        receiver_locations_x: X-component velocity receiver locations.
        accuracy: Finite difference order (2, 4, 6, 8).
        pml_width: Width of PML.
        pml_freq: Frequency for PML.
        max_vel: Maximum velocity for CFL.
        survey_pad: Padding for survey area.
        pressure_0: Initial pressure (time t=0).
        vz_0: Initial z-velocity (time t=-0.5).
        vy_0: Initial y-velocity (time t=-0.5).
        vx_0: Initial x-velocity (time t=-0.5).
        phi_z_0: Initial PML variable for pressure update (z).
        phi_y_0: Initial PML variable for pressure update (y).
        phi_x_0: Initial PML variable for pressure update (x).
        psi_z_0: Initial PML variable for velocity update (z).
        psi_y_0: Initial PML variable for velocity update (y).
        psi_x_0: Initial PML variable for velocity update (x).
        origin: Origin of initial wavefields.
        nt: Number of time steps.
        model_gradient_sampling_interval: Interval for gradient sampling.
        freq_taper_frac: Frequency taper fraction.
        time_pad_frac: Time padding fraction.
        time_taper: Time taper flag.
        forward_callback: Callback for forward pass.
        backward_callback: Callback for backward pass.
        callback_frequency: Frequency of callback.
        python_backend: Use Python backend.
        storage_mode: Storage mode.
        storage_path: Storage path.
        storage_compression: Storage compression.

    Returns:
        Tuple containing final wavefields:
        (pressure, vz, vy, vx, phi_z, phi_y, phi_x, psi_z, psi_y, psi_x)
        (omitting dimensions that don't exist)
        and receiver_amplitudes (pressure, vz, vy, vx) (omitting dimensions that
        don't exist).
    """
    deepwave.common.check_inputs_not_vmapped(
        pressure_0,
        source_amplitudes_p,
        source_amplitudes_z,
        source_amplitudes_y,
        source_amplitudes_x,
        source_locations_p,
        source_locations_z,
        source_locations_y,
        source_locations_x,
        receiver_locations_p,
        receiver_locations_z,
        receiver_locations_y,
        receiver_locations_x,
        vz_0,
        phi_z_0,
        psi_z_0,
        vy_0,
        phi_y_0,
        psi_y_0,
        vx_0,
        phi_x_0,
        psi_x_0,
    )
    ndim = deepwave.common.get_ndim(
        [v, rho],
        [pressure_0],
        [
            source_locations_p,
            source_locations_z,
            source_locations_y,
            source_locations_x,
            receiver_locations_p,
            receiver_locations_z,
            receiver_locations_y,
            receiver_locations_x,
        ],
        [vz_0, phi_z_0, psi_z_0],
        [vy_0, phi_y_0, psi_y_0],
        [vx_0, phi_x_0, psi_x_0],
    )
    if ndim < 3 and (
        source_locations_z is not None or receiver_locations_z is not None
    ):
        raise ValueError(
            "The propagation was determined to be "
            f"{ndim}d, so locations related to the "
            "z dimension should not be provided."
        )
    if ndim < 2 and (
        source_locations_y is not None or receiver_locations_y is not None
    ):
        raise ValueError(
            "The propagation was determined to be "
            f"{ndim}d, so locations related to the "
            "y dimension should not be provided."
        )

    # Prepare initial wavefields list
    initial_wavefields: List[Optional[torch.Tensor]] = []
    initial_wavefields.append(pressure_0)
    if ndim == 3:
        initial_wavefields.append(vz_0)
    if ndim >= 2:
        initial_wavefields.append(vy_0)
    initial_wavefields.append(vx_0)

    # Phi (PML for pressure)
    if ndim == 3:
        initial_wavefields.append(phi_z_0)
    if ndim >= 2:
        initial_wavefields.append(phi_y_0)
    initial_wavefields.append(phi_x_0)

    # Psi (PML for velocity)
    if ndim == 3:
        initial_wavefields.append(psi_z_0)
    if ndim >= 2:
        initial_wavefields.append(psi_y_0)
    initial_wavefields.append(psi_x_0)

    if deepwave.common.is_inside_vmap(v):
        if max_vel is None:
            raise RuntimeError(
                "If using BatchedTensor inputs, you must specify max_vel"
            )
        max_model_vel = max_vel
        min_nonzero_model_vel = 0.0
    else:
        v_nonzero = v[v != 0]
        if v_nonzero.numel() > 0:
            min_nonzero_model_vel = v_nonzero.abs().min().item()
        else:
            min_nonzero_model_vel = 0.0
        del v_nonzero
        max_model_vel = v.abs().max().item()
    fd_pad = [accuracy // 2, accuracy // 2 - 1] * ndim

    source_amplitudes_list = []
    source_locations_list = []
    if ndim == 3:
        source_amplitudes_list.extend(
            [source_amplitudes_z, source_amplitudes_y, source_amplitudes_x]
        )
        source_locations_list.extend(
            [source_locations_z, source_locations_y, source_locations_x]
        )
    elif ndim == 2:
        source_amplitudes_list.extend([source_amplitudes_y, source_amplitudes_x])
        source_locations_list.extend([source_locations_y, source_locations_x])
    elif ndim == 1:
        source_amplitudes_list.append(source_amplitudes_x)
        source_locations_list.append(source_locations_x)

    receiver_locations_list = []
    if ndim == 3:
        receiver_locations_list.extend(
            [receiver_locations_z, receiver_locations_y, receiver_locations_x]
        )
    elif ndim == 2:
        receiver_locations_list.extend([receiver_locations_y, receiver_locations_x])
    elif ndim == 1:
        receiver_locations_list.append(receiver_locations_x)
    dim_names = ["z", "y", "x"]
    for dim in range(ndim):
        for locations, name in [
            (source_locations_list, "source"),
            (receiver_locations_list, "receiver"),
        ]:
            location = locations[dim]
            if location is None:
                continue
            dim_location = location[..., -ndim + dim]
            dim_location = dim_location[dim_location != deepwave.IGNORE_LOCATION]
            if (
                dim_location.numel() > 0
                and dim_location.max() >= v.shape[-ndim + dim] - 1
            ):
                raise RuntimeError(
                    f"With the provided model, the maximum {dim_names[-ndim + dim]} "
                    f"{name} location in the {dim_names[-ndim + dim]} "
                    f"dimension must be less "
                    f"than {v.shape[-ndim + dim] - 1}.",
                )

    source_amplitudes_list.insert(0, source_amplitudes_p)
    source_locations_list.insert(0, source_locations_p)
    receiver_locations_list.insert(0, receiver_locations_p)

    (
        models,
        source_amplitudes_out,
        wavefields,
        sources_i,
        receivers_i,
        grid_spacing,
        dt,
        nt,
        n_shots,
        step_ratio,
        model_gradient_sampling_interval,
        accuracy,
        pml_width,
        pml_freq,
        max_vel,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        device,
        dtype,
    ) = deepwave.common.setup_propagator(
        [v, rho],
        ["replicate", "replicate"],
        grid_spacing,
        dt,
        source_amplitudes_list,
        source_locations_list,
        receiver_locations_list,
        accuracy,
        fd_pad,
        pml_width,
        pml_freq,
        max_vel,
        min_nonzero_model_vel,
        max_model_vel,
        survey_pad,
        initial_wavefields,
        origin,
        nt,
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        ndim,
    )
    del (
        v,
        rho,
        source_amplitudes_p,
        source_locations_p,
        source_amplitudes_z,
        source_locations_z,
        source_amplitudes_y,
        source_locations_y,
        source_amplitudes_x,
        source_locations_x,
        receiver_locations_p,
        receiver_locations_z,
        receiver_locations_y,
        receiver_locations_x,
    )

    models = prepare_models(models[0], models[1])

    # Scale source amplitudes
    # Pressure update: p(t+1) = p(t) - dt * K * (div(v) - s) -> Add dt * K * s
    # Velocity update: v(t+1/2) = v(t-1/2) - dt * B * (grad(p) - f) -> Add dt * B * f
    # models order: K, B_z (3D), B_y (2D/3D), B_x (1D/2D/3D) matches target indices
    model_shape = models[0].shape[-ndim:]
    flat_model_shape = int(torch.prod(torch.tensor(model_shape)).item())

    for i, (src_amp, src_loc, model) in enumerate(
        zip(source_amplitudes_out, sources_i, models)
    ):
        if src_amp.numel() > 0:
            mask = src_loc == deepwave.common.IGNORE_LOCATION
            sources_i_masked = src_loc.clone()
            sources_i_masked[mask] = 0

            source_amplitudes_out[i] = (
                src_amp
                * (
                    model.view(-1, flat_model_shape)
                    .expand(n_shots, -1)
                    .gather(1, sources_i_masked)
                )
                * dt
            )

    pml_profiles = deepwave.staggered_grid.set_pml_profiles(
        pml_width,
        accuracy,
        fd_pad,
        dt,
        grid_spacing,
        max_vel,
        dtype,
        device,
        pml_freq,
        model_shape,
    )

    if not isinstance(callback_frequency, int):
        raise TypeError("callback_frequency must be an int.")
    if callback_frequency <= 0:
        raise ValueError("callback_frequency must be positive.")

    outputs = list(
        acoustic_func(
            python_backend,
            pml_profiles,
            grid_spacing,
            dt,
            nt,
            step_ratio * model_gradient_sampling_interval,
            accuracy,
            pml_width,
            n_shots,
            forward_callback,
            backward_callback,
            callback_frequency,
            storage_mode,
            storage_path,
            storage_compression,
            *models,
            *source_amplitudes_out,
            *sources_i,
            *receivers_i,
            *wavefields,
        )
    )

    for i in range(len(outputs) - 1 - ndim, len(outputs)):
        outputs[i] = deepwave.common.downsample_and_movedim(
            outputs[i],
            step_ratio,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
        )

    return outputs


def zero_edge(tensor: torch.Tensor, fd_pad: int, dim: int) -> torch.Tensor:
    """Sets values at the end of a dimension of a tensor to zero."""
    tensor[(slice(None),) + (slice(None),) * dim + (-fd_pad,)].fill_(0)
    return tensor


def zero_edges_and_interiors(
    wavefields: List[torch.Tensor],
    ndim: int,
    fd_pad: int,
    fd_pad_list: List[int],
    pml_width: List[int],
    interior: bool = True,
) -> None:
    """Zeros the edges and/or interiors of wavefields."""
    # Based on elastic.py implementation but adapted for acoustic variables
    # Wavefields order: [pressure, vz, vy, vx, phi..., psi...]
    # Pressure (scalar) is at integer grid points.
    # Velocities are at half grid points.

    # Identify which wavefields are "half-grid" in each dimension.
    # > p: (0,0,0)
    # > vz: (1,0,0) (shifted in z) -> ndim=3
    # > vy: (0,1,0) (shifted in y)
    # > vx: (0,0,1) (shifted in x)

    # > phi (PML memory variable for p update): centered at p (0,0,0)
    # > psi (PML memory variable for v update): centered at v.
    # > psi_z matches vz (1,0,0)
    # > psi_y matches vy (0,1,0)
    # > psi_x matches vx (0,0,1)

    # Map wavefield index to half_grid boolean tuple

    num_vars = len(wavefields)
    # Expected num_vars:
    # 3D: p, vz, vy, vx, phiz, phiy, phix, psiz, psiy, psix (10 vars)
    # 2D: p, vy, vx, phiy, phix, psiy, psix (7 vars)
    # 1D: p, vx, phix, psix (4 vars)

    half_grid = [[False] * ndim for _ in range(num_vars)]
    pml_edge_idx: List[List[int]] = [[] * ndim]

    if ndim == 3:
        # p, phis are 0, 4, 5, 6 -> all False (integer grid)
        # vz (1), psiz (7) -> shifted z (dim 0)
        half_grid[1][0] = True
        half_grid[7][0] = True
        # vy (2), psiy (8) -> shifted y (dim 1)
        half_grid[2][1] = True
        half_grid[8][1] = True
        # vx (3), psix (9) -> shifted x (dim 2)
        half_grid[3][2] = True
        half_grid[9][2] = True
        pml_edge_idx = [[4, 7], [5, 8], [6, 9]]
    elif ndim == 2:
        # p (0), phis (3,4) -> False
        # vy (1), psiy (5) -> shifted y (dim 0)
        half_grid[1][0] = True
        half_grid[5][0] = True
        # vx (2), psix (6) -> shifted x (dim 1)
        half_grid[2][1] = True
        half_grid[6][1] = True
        pml_edge_idx = [[3, 5], [4, 6]]
    elif ndim == 1:
        # p (0), phi (2) -> False
        # vx (1), psix (3) -> shifted x (dim 0)
        half_grid[1][0] = True
        half_grid[3][0] = True
        pml_edge_idx = [[2, 3]]

    for i in range(num_vars):
        for dim in range(ndim):
            if half_grid[i][dim]:
                wavefields[i] = zero_edge(wavefields[i], fd_pad, dim)

    # Zero the interiors of the PML memory variables
    if interior:
        for dim in range(ndim):
            for i in pml_edge_idx[dim]:
                fd_pad_shifted = list(fd_pad_list)
                if half_grid[i][dim]:
                    fd_pad_shifted[2 * dim + 1] += 1
                wavefields[i] = deepwave.common.zero_interior(
                    wavefields[i], fd_pad_shifted, pml_width, dim
                )


class AcousticForwardFunc(torch.autograd.Function):
    """Autograd function for the forward pass of acoustic wave propagation."""

    @staticmethod
    def forward(
        ctx: Any,
        pml_profiles: List[torch.Tensor],
        grid_spacing: List[float],
        dt: float,
        nt: int,
        step_ratio: int,
        accuracy: int,
        pml_width: List[int],
        n_shots: int,
        forward_callback: Optional[deepwave.common.Callback],
        backward_callback: Optional[deepwave.common.Callback],
        callback_frequency: int,
        storage_mode_str: str,
        storage_path: str,
        storage_compression: bool,
        *args: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Performs the forward propagation of the acoustic wave equation."""
        ndim = len(grid_spacing)
        args_list = list(args)

        # Parse args based on ndim
        # Structure: models, source_amplitudes, sources_i, receivers_i, wavefields

        # Models: [K, B_z, B_y, B_x] (3D: 4 models),
        #         [K, B_y, B_x] (2D: 3 models),
        #         [K, B_x] (1D: 2 models)
        num_models = ndim + 1

        num_source_amps = ndim + 1
        num_source_locs = ndim + 1
        num_recv_locs = ndim + 1

        # Wavefields:
        # 3D: p, vz, vy, vx, 3*phi, 3*psi -> 10
        # 2D: p, vy, vx, 2*phi, 2*psi -> 7
        # 1D: p, vx, phi, psi -> 4
        num_wavefields = 1 + ndim + ndim + ndim

        wavefields = args_list[-num_wavefields:]
        del args_list[-num_wavefields:]
        receivers_i = args_list[-num_recv_locs:]
        del args_list[-num_recv_locs:]
        sources_i = args_list[-num_source_locs:]
        del args_list[-num_source_locs:]
        source_amplitudes = args_list[-num_source_amps:]
        del args_list[-num_source_amps:]
        models = args_list[-num_models:]
        del args_list[-num_models:]

        if len(args_list) != 0:
            raise AssertionError(
                "Error parsing arguments for AcousticForwardFunc.forward"
            )

        del args
        models = [model.contiguous() for model in models]
        source_amplitudes = [amp.contiguous() for amp in source_amplitudes]
        pml_profiles = [profile.contiguous() for profile in pml_profiles]
        sources_i = [locs.contiguous() for locs in sources_i]
        receivers_i = [locs.contiguous() for locs in receivers_i]

        device = models[0].device
        dtype = models[0].dtype

        # Setup storage
        if str(device) == "cpu" and storage_mode_str == "cpu":
            storage_mode_str = "device"

        if storage_mode_str == "device":
            storage_mode = deepwave.common.StorageMode.DEVICE
        elif storage_mode_str == "cpu":
            storage_mode = deepwave.common.StorageMode.CPU
        elif storage_mode_str == "disk":
            storage_mode = deepwave.common.StorageMode.DISK
        elif storage_mode_str == "none":
            storage_mode = deepwave.common.StorageMode.NONE
        else:
            raise ValueError(f"Invalid storage_mode {storage_mode_str}")

        is_cuda = models[0].is_cuda
        model_shape = models[0].shape[-ndim:]
        n_sources_per_shot_list = [locs.numel() // n_shots for locs in sources_i]
        n_receivers_per_shot_list = [locs.numel() // n_shots for locs in receivers_i]

        storage_manager = deepwave.common.StorageManager(
            model_shape,
            dtype,
            n_shots,
            nt,
            step_ratio,
            storage_mode,
            storage_compression,
            storage_path,
            device,
            is_cuda,
        )

        for model in models:
            storage_manager.allocate(model.requires_grad)

        if (
            any(m.requires_grad for m in models)
            or any(sa.requires_grad for sa in source_amplitudes)
            or any(w.requires_grad for w in wavefields)
        ):
            ctx.save_for_backward(
                *models, *sources_i, *receivers_i, *source_amplitudes, *pml_profiles
            )
            ctx.grid_spacing = grid_spacing
            ctx.dt = dt
            ctx.nt = nt
            ctx.n_shots = n_shots
            ctx.step_ratio = step_ratio
            ctx.accuracy = accuracy
            ctx.pml_width = pml_width
            ctx.source_amplitudes_requires_grad = [
                sa.requires_grad for sa in source_amplitudes
            ]
            ctx.backward_callback = backward_callback
            ctx.callback_frequency = callback_frequency
            ctx.storage_manager = storage_manager

        fd_pad = accuracy // 2
        fd_pad_list = [fd_pad, fd_pad - 1] * ndim
        size_with_batch = (n_shots, *models[0].shape[-ndim:])

        wavefields = [
            deepwave.common.create_or_pad(
                w, fd_pad_list, device, dtype, size_with_batch
            )
            for w in wavefields
        ]

        zero_edges_and_interiors(wavefields, ndim, fd_pad, fd_pad_list, pml_width)

        # Setup PML bounds
        pml_b = [pml_width[2 * i] + fd_pad for i in range(ndim)]
        pml_e = [
            max(pml_b[i], model_shape[i] - pml_width[2 * i + 1] - (fd_pad - 1))
            for i in range(ndim)
        ]

        model_batched = [m.ndim == ndim + 1 and m.shape[0] > 1 for m in models[:2]]

        receiver_amplitudes = [
            torch.empty(0, device=device, dtype=dtype) for _ in receivers_i
        ]
        for i, (locs, n_per_shot) in enumerate(
            zip(receivers_i, n_receivers_per_shot_list)
        ):
            if locs.numel() > 0:
                receiver_amplitudes[i].resize_(nt, n_shots, n_per_shot)
                receiver_amplitudes[i].fill_(0)

        stream: Union[int, torch.Stream] = 0
        if is_cuda:
            aux = models[0].get_device()
            stream = torch.cuda.current_stream(aux)
        elif deepwave.backend_utils.USE_OPENMP:
            aux = min(n_shots, torch.get_num_threads())
        else:
            aux = 1

        forward = deepwave.backend_utils.get_backend_function(
            "acoustic", ndim, "forward", accuracy, dtype, device
        )

        rdx = [1 / dx for dx in grid_spacing]

        if forward_callback is None:
            callback_frequency = nt // step_ratio

        # Wavefield names for callback
        # Same order as wavefields list
        wf_names = ["pressure_0"]
        if ndim == 3:
            wf_names.extend(["vz_0", "vy_0", "vx_0"])
        elif ndim == 2:
            wf_names.extend(["vy_0", "vx_0"])
        elif ndim == 1:
            wf_names.extend(["vx_0"])

        if ndim == 3:
            wf_names.extend(["phi_z_0", "phi_y_0", "phi_x_0"])
        elif ndim == 2:
            wf_names.extend(["phi_y_0", "phi_x_0"])
        elif ndim == 1:
            wf_names.extend(["phi_x_0"])

        if ndim == 3:
            wf_names.extend(["psi_z_0", "psi_y_0", "psi_x_0"])
        elif ndim == 2:
            wf_names.extend(["psi_y_0", "psi_x_0"])
        elif ndim == 1:
            wf_names.extend(["psi_x_0"])

        callback_wavefields = {}
        model_names = ["K"]
        if ndim == 3:
            model_names.extend(["Bz", "By", "Bx"])
        elif ndim == 2:
            model_names.extend(["By", "Bx"])
        elif ndim == 1:
            model_names.extend(["Bx"])

        callback_models = dict(zip(model_names, models))

        # Main Loop
        if wavefields[0].numel() > 0 and nt > 0:
            for step in range(0, nt // step_ratio, callback_frequency):
                if forward_callback is not None:
                    callback_wavefields = dict(zip(wf_names, wavefields))
                    forward_callback(
                        deepwave.common.CallbackState(
                            dt,
                            step,
                            callback_wavefields,
                            callback_models,
                            {},
                            fd_pad_list,
                            pml_width,
                        )
                    )
                step_nt = min(nt // step_ratio - step, callback_frequency)

                if (
                    forward(
                        *[m.data_ptr() for m in models],
                        *[amp.data_ptr() for amp in source_amplitudes],
                        *[w.data_ptr() for w in wavefields],
                        *storage_manager.storage_ptrs,
                        *[amp.data_ptr() for amp in receiver_amplitudes],
                        *[p.data_ptr() for p in pml_profiles],
                        *[loc.data_ptr() for loc in sources_i],
                        *[loc.data_ptr() for loc in receivers_i],
                        *rdx,
                        dt,
                        step_nt * step_ratio,
                        n_shots,
                        *model_shape,
                        *n_sources_per_shot_list,
                        *n_receivers_per_shot_list,
                        step_ratio,
                        storage_mode,
                        storage_manager.shot_bytes_uncomp,
                        storage_manager.shot_bytes_comp,
                        *[
                            m.requires_grad
                            and storage_mode != deepwave.common.StorageMode.NONE
                            for m in models[:2]
                        ],
                        *model_batched,
                        storage_compression,
                        step * step_ratio,
                        *pml_b,
                        *pml_e,
                        aux,
                        stream,
                    )
                    != 0
                ):
                    raise RuntimeError("Compiled backend failed.")

        s = (
            slice(None),
            *(slice(fd_pad, shape - (fd_pad - 1)) for shape in model_shape),
        )
        return (*[w[s] for w in wavefields], *receiver_amplitudes)

    @staticmethod
    def backward(ctx: Any, *args: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Computes gradients during backward pass."""
        # Unpack
        grid_spacing = ctx.grid_spacing
        ndim = len(grid_spacing)

        grad_wavefields = list(args[: -ndim - 1])
        grad_r = list(args[-ndim - 1 :])
        del args

        saved_tensors = list(ctx.saved_tensors)
        num_models = ndim + 1
        num_receiver_types = ndim + 1  # For receivers
        num_source_types = ndim + 1

        models = saved_tensors[:num_models]
        del saved_tensors[:num_models]
        sources_i = saved_tensors[:num_source_types]
        del saved_tensors[:num_source_types]
        receivers_i = saved_tensors[:num_receiver_types]
        del saved_tensors[:num_receiver_types]
        source_amplitudes = saved_tensors[:num_source_types]
        del saved_tensors[:num_source_types]
        pml_profiles = saved_tensors

        grad_r = [g.contiguous() for g in grad_r]
        models = [m.contiguous() for m in models]
        sources_i = [locs.contiguous() for locs in sources_i]
        receivers_i = [locs.contiguous() for locs in receivers_i]
        source_amplitudes = [a.contiguous() for a in source_amplitudes]
        pml_profiles = [p.contiguous() for p in pml_profiles]

        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        pml_width = ctx.pml_width
        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad
        backward_callback = ctx.backward_callback
        callback_frequency = ctx.callback_frequency
        storage_manager = ctx.storage_manager

        device = models[0].device
        dtype = models[0].dtype
        is_cuda = models[0].is_cuda
        model_shape = models[0].shape[-ndim:]
        n_sources_per_shot_list = [locs.numel() // n_shots for locs in sources_i]
        n_sources_per_shot_list_for_grad = [
            count if req else 0
            for count, req in zip(
                n_sources_per_shot_list, source_amplitudes_requires_grad
            )
        ]
        n_receivers_per_shot_list = [locs.numel() // n_shots for locs in receivers_i]

        grad_models = [torch.empty(0, device=device, dtype=dtype) for _ in models]
        grad_models_tmp = [torch.empty(0, device=device, dtype=dtype) for _ in models]
        grad_models_tmp_ptr = [g.data_ptr() for g in grad_models]
        for i, model in enumerate(models):
            if model.requires_grad:
                grad_models[i].resize_(*model.shape)
                grad_models[i].fill_(0)
                grad_models_tmp_ptr[i] = grad_models[i].data_ptr()

        grad_f_list = []
        for i in range(num_source_types):
            grad_f = torch.empty(0, device=device, dtype=dtype)
            if source_amplitudes_requires_grad[i]:
                grad_f.resize_(nt, n_shots, n_sources_per_shot_list[i])
                grad_f.fill_(0)
            grad_f_list.append(grad_f)

        fd_pad = accuracy // 2
        fd_pad_list = [fd_pad, fd_pad - 1] * ndim
        size_with_batch = (n_shots, *model_shape)

        grad_wavefields = [
            deepwave.common.create_or_pad(
                w, fd_pad_list, device, dtype, size_with_batch
            )
            for w in grad_wavefields
        ]
        aux_wavefields = [torch.zeros_like(grad_wavefields[0]) for _ in range(ndim)]

        zero_edges_and_interiors(grad_wavefields, ndim, fd_pad, fd_pad_list, pml_width)

        pml_b = [
            min(pml_width[2 * i] + 2 * fd_pad, model_shape[i] - (fd_pad - 1))
            for i in range(ndim)
        ]
        pml_e = [
            max(pml_b[i], model_shape[i] - pml_width[2 * i + 1] - 2 * fd_pad + 1)
            for i in range(ndim)
        ]

        model_batched = [m.ndim == ndim + 1 and m.shape[0] > 1 for m in models[:2]]

        stream: Union[int, torch.Stream] = 0
        if is_cuda:
            aux = models[0].get_device()
            stream = torch.cuda.current_stream(aux)
            for i, model in enumerate(models):
                if (
                    model.requires_grad
                    and not (model_batched[0] if i == 0 else model_batched[1])
                    and n_shots > 1
                    and storage_manager.storage_mode != deepwave.common.StorageMode.NONE
                ):
                    grad_models_tmp[i].resize_(n_shots, *model_shape)
                    grad_models_tmp[i].fill_(0)
                    grad_models_tmp_ptr[i] = grad_models_tmp[i].data_ptr()
        else:
            if deepwave.backend_utils.USE_OPENMP:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            for i, model in enumerate(models):
                if (
                    model.requires_grad
                    and not (model_batched[0] if i == 0 else model_batched[1])
                    and aux > 1
                    and deepwave.backend_utils.USE_OPENMP
                    and storage_manager.storage_mode != deepwave.common.StorageMode.NONE
                ):
                    grad_models_tmp[i].resize_(aux, *model_shape)
                    grad_models_tmp[i].fill_(0)
                    grad_models_tmp_ptr[i] = grad_models_tmp[i].data_ptr()

        backward = deepwave.backend_utils.get_backend_function(
            "acoustic", ndim, "backward", accuracy, dtype, device
        )
        rdx = [1 / dx for dx in grid_spacing]

        if backward_callback is None:
            callback_frequency = nt // step_ratio

        wf_names = ["pressure_0"]
        if ndim == 3:
            wf_names.extend(["vz_0", "vy_0", "vx_0"])
        elif ndim == 2:
            wf_names.extend(["vy_0", "vx_0"])
        elif ndim == 1:
            wf_names.extend(["vx_0"])

        if ndim == 3:
            wf_names.extend(["phi_z_0", "phi_y_0", "phi_x_0"])
        elif ndim == 2:
            wf_names.extend(["phi_y_0", "phi_x_0"])
        elif ndim == 1:
            wf_names.extend(["phi_x_0"])

        if ndim == 3:
            wf_names.extend(["psi_z_0", "psi_y_0", "psi_x_0"])
        elif ndim == 2:
            wf_names.extend(["psi_y_0", "psi_x_0"])
        elif ndim == 1:
            wf_names.extend(["psi_x_0"])

        callback_wavefields = {}
        model_names = ["K"]
        if ndim == 3:
            model_names.extend(["Bz", "By", "Bx"])
        elif ndim == 2:
            model_names.extend(["By", "Bx"])
        elif ndim == 1:
            model_names.extend(["Bx"])

        callback_models = dict(zip(model_names, models))
        callback_grad_models = dict(zip(model_names, grad_models))

        if grad_wavefields[0].numel() > 0 and nt > 0:
            for step in range(nt // step_ratio, 0, -callback_frequency):
                step_nt = min(step, callback_frequency)
                if (
                    backward(
                        *[m.data_ptr() for m in models],
                        *[g.data_ptr() for g in grad_r],
                        *[field.data_ptr() for field in grad_wavefields],
                        *[field.data_ptr() for field in aux_wavefields],
                        *storage_manager.storage_ptrs,
                        *[g.data_ptr() for g in grad_f_list],
                        *[g.data_ptr() for g in grad_models],
                        *grad_models_tmp_ptr,
                        *[p.data_ptr() for p in pml_profiles],
                        *[loc.data_ptr() for loc in sources_i],
                        *[loc.data_ptr() for loc in receivers_i],
                        *rdx,
                        dt,
                        step_nt * step_ratio,
                        n_shots,
                        *model_shape,
                        *n_sources_per_shot_list_for_grad,
                        *n_receivers_per_shot_list,
                        step_ratio,
                        storage_manager.storage_mode,
                        storage_manager.shot_bytes_uncomp,
                        storage_manager.shot_bytes_comp,
                        *[
                            m.requires_grad
                            and storage_manager.storage_mode
                            != deepwave.common.StorageMode.NONE
                            for m in models[:2]
                        ],
                        *model_batched,
                        storage_manager.storage_compression,
                        step * step_ratio,
                        *pml_b,
                        *pml_e,
                        aux,
                        stream,
                    )
                    != 0
                ):
                    raise RuntimeError("Compiled backend failed.")
                if (step_nt * step_ratio) % 2 != 0:
                    (
                        grad_wavefields[-ndim:],
                        aux_wavefields[:],
                    ) = (
                        aux_wavefields[:],
                        grad_wavefields[-ndim:],
                    )
                if backward_callback is not None:
                    callback_wavefields = dict(zip(wf_names, grad_wavefields))
                    backward_callback(
                        deepwave.common.CallbackState(
                            dt,
                            step - step_nt,
                            callback_wavefields,
                            callback_models,
                            callback_grad_models,
                            fd_pad_list,
                            pml_width,
                        )
                    )

        s = (
            slice(None),
            *(slice(fd_pad, shape - (fd_pad - 1)) for shape in model_shape),
        )
        return tuple(
            [None] * 14
            + grad_models
            + grad_f_list
            + [None] * num_source_types  # sources_i
            + [None] * num_receiver_types  # receivers_i
            + [w[s] for w in grad_wavefields]
        )


def update_velocities(
    models: List[torch.Tensor],
    p: torch.Tensor,
    velocities: List[torch.Tensor],
    psi: List[torch.Tensor],
    pml_profiles: List[torch.Tensor],
    grid_spacing: List[float],
    dt: float,
    accuracy: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Updates the velocity wavefields and PML memory variables."""
    ndim = len(grid_spacing)

    # Extract PML profiles
    ah = pml_profiles[2::4]
    bh = pml_profiles[3::4]

    buoyancy = models[1:]

    for dim in range(ndim):
        # Calculate grad p in dim
        gp = deepwave.staggered_grid.diff1h(
            p, dim, accuracy, 1 / grid_spacing[dim], ndim
        )

        # Update psi
        psi[dim] = ah[dim] * psi[dim] + bh[dim] * gp

        # Update v
        # Note: gp + psi is total effective gradient
        velocities[dim] = velocities[dim] - dt * buoyancy[dim] * (gp + psi[dim])

    return velocities, psi


def update_pressure(
    models: List[torch.Tensor],
    p: torch.Tensor,
    velocities: List[torch.Tensor],
    phi: List[torch.Tensor],
    pml_profiles: List[torch.Tensor],
    grid_spacing: List[float],
    dt: float,
    accuracy: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Updates the pressure wavefield and PML memory variables."""
    ndim = len(grid_spacing)

    # Extract PML profiles
    a = pml_profiles[::4]
    b = pml_profiles[1::4]

    k = models[0]

    div_v_sum = torch.zeros_like(p)
    for dim in range(ndim):
        # Calculate derivative of v in dim
        dv = deepwave.staggered_grid.diff1(
            velocities[dim], dim, accuracy, 1 / grid_spacing[dim], ndim
        )

        # Update phi
        phi[dim] = a[dim] * phi[dim] + b[dim] * dv

        # Accumulate div
        div_v_sum = div_v_sum + dv + phi[dim]

    p = p - dt * k * div_v_sum

    return p, phi


_update_velocities_jit = None
_update_velocities_compile = None
_update_velocities_opt = update_velocities
_update_pressure_jit = None
_update_pressure_compile = None
_update_pressure_opt = update_pressure


def acoustic_python(
    pml_profiles: List[torch.Tensor],
    grid_spacing: List[float],
    dt: float,
    nt: int,
    step_ratio: int,
    accuracy: int,
    pml_width: List[int],
    n_shots: int,
    forward_callback: Optional[deepwave.common.Callback],
    backward_callback: Optional[deepwave.common.Callback],
    callback_frequency: int,
    storage_mode_str: str,
    storage_path: str,
    storage_compression: bool,
    *args: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Python backend for acoustic wave propagation."""
    if backward_callback is not None:
        raise RuntimeError("backward_callback is not supported in the Python backend.")
    if storage_mode_str != "device":
        raise RuntimeError(
            "Specifying storage mode is not supported in Python backend."
        )

    is_batched = any(deepwave.common.is_inside_vmap(x) for x in args)
    ndim = len(grid_spacing)
    args_list = list(args)

    num_models = ndim + 1
    num_source_types = ndim + 1
    num_receiver_types = ndim + 1
    num_wf = 1 + 3 * ndim

    wavefields = args_list[-num_wf:]
    del args_list[-num_wf:]
    receivers_i = args_list[-num_receiver_types:]
    del args_list[-num_receiver_types:]
    sources_i = args_list[-num_source_types:]
    del args_list[-num_source_types:]
    source_amplitudes = args_list[-num_source_types:]
    del args_list[-num_source_types:]
    models = args_list[-num_models:]
    del args_list[-num_models:]

    fd_pad = accuracy // 2
    fd_pad_list = [fd_pad, fd_pad - 1] * ndim
    size_with_batch = (n_shots, *models[0].shape[-ndim:])

    wavefields = [
        deepwave.common.create_or_pad(
            w, fd_pad_list, models[0].device, models[0].dtype, size_with_batch
        )
        for w in wavefields
    ]

    zero_edges_and_interiors(wavefields, ndim, fd_pad, fd_pad_list, pml_width)

    p = wavefields[0]
    velocities = wavefields[1 : 1 + ndim]
    phi = wavefields[1 + ndim : 1 + 2 * ndim]
    psi = wavefields[1 + 2 * ndim :]
    del wavefields

    device = models[0].device
    dtype = models[0].dtype
    shape = models[0].shape[-ndim:]
    flat_shape = int(torch.prod(torch.tensor(shape)).item())
    n_receivers_per_shot_list = [locs.numel() // n_shots for locs in receivers_i]
    receiver_amplitudes = [
        torch.empty(0, device=device, dtype=dtype) for _ in receivers_i
    ]
    receiver_amplitudes_lists: List[List[torch.Tensor]] = [[] for _ in receivers_i]

    for i, (locs, n_per_shot) in enumerate(zip(receivers_i, n_receivers_per_shot_list)):
        if locs.numel() > 0 and not is_batched:
            receiver_amplitudes[i].resize_(nt, n_shots, n_per_shot)
            receiver_amplitudes[i].fill_(0)

    # Prepare sources for injection
    # source_amplitudes here are already scaled by setup in acoustic()
    masked_sources = []
    for i in range(num_source_types):
        src_mask = sources_i[i] != deepwave.common.IGNORE_LOCATION
        sources_i_masked = torch.zeros_like(sources_i[i])
        sources_i_masked[src_mask] = sources_i[i][src_mask]

        src_amp_masked = torch.zeros_like(source_amplitudes[i])
        if source_amplitudes[i].numel() > 0:
            src_amp_masked[:, src_mask] = source_amplitudes[i][:, src_mask]

        masked_sources.append(
            (sources_i_masked, src_amp_masked, i)  # i is the target_idx
        )

    masked_receivers = []
    for i, locs in enumerate(receivers_i):
        recv_mask = locs != deepwave.common.IGNORE_LOCATION
        receivers_i_masked = torch.zeros_like(locs)
        if locs.numel() > 0:
            receivers_i_masked[recv_mask] = locs[recv_mask]
        masked_receivers.append((receivers_i_masked, recv_mask, i))

    # Wavefield names for callback
    # Same order as wavefields list
    wf_names = ["pressure_0"]
    if ndim == 3:
        wf_names.extend(["vz_0", "vy_0", "vx_0"])
    elif ndim == 2:
        wf_names.extend(["vy_0", "vx_0"])
    elif ndim == 1:
        wf_names.extend(["vx_0"])

    if ndim == 3:
        wf_names.extend(["phi_z_0", "phi_y_0", "phi_x_0"])
    elif ndim == 2:
        wf_names.extend(["phi_y_0", "phi_x_0"])
    elif ndim == 1:
        wf_names.extend(["phi_x_0"])

    if ndim == 3:
        wf_names.extend(["psi_z_0", "psi_y_0", "psi_x_0"])
    elif ndim == 2:
        wf_names.extend(["psi_y_0", "psi_x_0"])
    elif ndim == 1:
        wf_names.extend(["psi_x_0"])

    callback_wavefields = {}
    model_names = ["K"]
    if ndim == 3:
        model_names.extend(["Bz", "By", "Bx"])
    elif ndim == 2:
        model_names.extend(["By", "Bx"])
    elif ndim == 1:
        model_names.extend(["Bx"])

    callback_models = dict(zip(model_names, models))

    for step in range(nt // step_ratio):
        if forward_callback is not None and step % callback_frequency == 0:
            current_wavefields = [p, *velocities, *phi, *psi]
            callback_wavefields = dict(zip(wf_names, current_wavefields))
            forward_callback(
                deepwave.common.CallbackState(
                    dt,
                    step,
                    callback_wavefields,
                    callback_models,
                    {},
                    fd_pad_list,
                    pml_width,
                )
            )

        for inner_step in range(step_ratio):
            t = step * step_ratio + inner_step

            # Record receivers
            for i, (receivers_i_masked, _, target_idx) in enumerate(masked_receivers):
                if receivers_i_masked.numel() > 0:
                    val = torch.empty(0)
                    if target_idx == 0:  # Pressure
                        val = p.view(-1, flat_shape).gather(1, receivers_i_masked)
                    elif target_idx > 0 and target_idx <= ndim:  # Velocity
                        v_idx = target_idx - 1
                        val = (
                            velocities[v_idx]
                            .view(-1, flat_shape)
                            .gather(1, receivers_i_masked)
                        )
                    if val.numel() > 0:
                        if is_batched:
                            receiver_amplitudes_lists[i].append(val)
                        else:
                            receiver_amplitudes[i][t] = val

            # Update velocities
            velocities, psi = _update_velocities_opt(
                models, p, velocities, psi, pml_profiles, grid_spacing, dt, accuracy
            )

            # Inject velocity sources
            new_velocities = list(velocities)
            for sources_i_masked, src_amp_masked, target_idx in masked_sources:
                if target_idx > 0 and target_idx <= ndim and src_amp_masked.numel() > 0:
                    v_idx = target_idx - 1
                    new_velocities[v_idx] = (
                        new_velocities[v_idx]
                        .reshape(-1, flat_shape)
                        .scatter_add(1, sources_i_masked, src_amp_masked[t])
                        .view(size_with_batch)
                    )
            velocities = new_velocities

            # Update pressure
            p, phi = _update_pressure_opt(
                models, p, velocities, phi, pml_profiles, grid_spacing, dt, accuracy
            )

            # Inject pressure sources
            for sources_i_masked, src_amp_masked, target_idx in masked_sources:
                if target_idx == 0 and src_amp_masked.numel() > 0:
                    p = (
                        p.reshape(-1, flat_shape)
                        .scatter_add(1, sources_i_masked, src_amp_masked[t])
                        .view(size_with_batch)
                    )

    if is_batched:
        for i in range(len(receiver_amplitudes)):
            if len(receiver_amplitudes_lists[i]) > 0:
                receiver_amplitudes[i] = torch.stack(receiver_amplitudes_lists[i])

    for i, (_, recv_mask, _) in enumerate(masked_receivers):
        if receiver_amplitudes[i].numel() > 0:
            receiver_amplitudes[i].masked_fill_(~recv_mask.unsqueeze(0), 0)

    wavefields = [p, *velocities, *phi, *psi]
    s = (slice(None), *(slice(fd_pad, shape - (fd_pad - 1)) for shape in shape))
    return (*[w[s] for w in wavefields], *receiver_amplitudes)


def acoustic_func(
    python_backend: Union[Literal["eager", "jit", "compile"], bool] = False,
    *args: Any,
) -> Tuple[torch.Tensor, ...]:
    """Helper function to apply the AcousticForwardFunc."""
    global _update_velocities_jit, _update_velocities_compile, _update_velocities_opt
    global _update_pressure_jit, _update_pressure_compile, _update_pressure_opt

    if python_backend:
        if python_backend is True:
            mode = "compile" if deepwave.backend_utils.USE_OPENMP else "jit"
        elif isinstance(python_backend, str):
            mode = python_backend.lower()
        else:
            raise TypeError(
                f"python_backend must be bool or str, but got {type(python_backend)}"
            )

        if mode == "jit":
            if _update_velocities_jit is None:
                _update_velocities_jit = torch.jit.script(update_velocities)
            _update_velocities_opt = _update_velocities_jit
            if _update_pressure_jit is None:
                _update_pressure_jit = torch.jit.script(update_pressure)
            _update_pressure_opt = _update_pressure_jit
        elif mode == "compile":
            if _update_velocities_compile is None:
                _update_velocities_compile = torch.compile(
                    update_velocities, fullgraph=True
                )
            _update_velocities_opt = _update_velocities_compile
            if _update_pressure_compile is None:
                _update_pressure_compile = torch.compile(
                    update_pressure, fullgraph=True
                )
            _update_pressure_opt = _update_pressure_compile
        elif mode == "eager":
            _update_velocities_opt = update_velocities
            _update_pressure_opt = update_pressure
        else:
            raise ValueError(f"Unknown python_backend value {mode!r}.")
        return acoustic_python(*args)

    func = AcousticForwardFunc.apply

    return cast(
        "Tuple[torch.Tensor, ...]",
        func(*args),  # type: ignore[no-untyped-call]
    )
