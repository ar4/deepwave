"""Elastic wave propagation module for Deepwave.

Implements elastic wave equation propagation using a velocity-stress
formulation with finite differences in time (2nd order) and space
(user-selectable order: 2 or 4). Supports Convolutional Perfectly Matched
Layer (C-PML) boundaries for absorbing reflections.

This module provides both a `torch.nn.Module` interface (`Elastic` class)
and a functional interface (`elastic` function) for performing elastic
wave simulations. The outputs are differentiable with respect to the
material parameters (Lam'e parameters and buoyancy) and source amplitudes.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import torch

import deepwave.backend_utils
import deepwave.common
import deepwave.staggered_grid


class Elastic(torch.nn.Module):
    """A PyTorch Module for elastic wave propagation.

    This Module provides a convenient way to perform elastic wave propagation.
    It stores the Lamé parameters (`lamb` and `mu`) and buoyancy (`buoyancy`)
    as `torch.nn.Parameter` objects, allowing gradients to be computed with
    respect to them if desired.

    Args:
        lamb: A torch.Tensor containing the first Lamé parameter.
        mu: A torch.Tensor containing the second Lamé parameter (mu).
        buoyancy: A torch.Tensor containing the buoyancy (1/density).
        grid_spacing: The spatial grid cell size. It can be a single number
            (for isotropic grids) or a sequence of numbers (for anisotropic
            grids).
        lamb_requires_grad: A bool specifying whether gradients should be
            computed for `lamb`. Defaults to False.
        mu_requires_grad: A bool specifying whether gradients should be
            computed for `mu`. Defaults to False.
        buoyancy_requires_grad: A bool specifying whether gradients should be
            computed for `buoyancy`. Defaults to False.

    """

    def __init__(
        self,
        lamb: torch.Tensor,
        mu: torch.Tensor,
        buoyancy: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        lamb_requires_grad: bool = False,
        mu_requires_grad: bool = False,
        buoyancy_requires_grad: bool = False,
    ) -> None:
        """Initializes the Module."""
        super().__init__()
        if not isinstance(lamb_requires_grad, bool):
            raise TypeError(
                f"lamb_requires_grad must be bool, "
                f"got {type(lamb_requires_grad).__name__}",
            )
        if not isinstance(lamb, torch.Tensor):
            raise TypeError("lamb must be a torch.Tensor.")
        if not isinstance(mu_requires_grad, bool):
            raise TypeError(
                f"mu_requires_grad must be bool, got {type(mu_requires_grad).__name__}",
            )
        if not isinstance(mu, torch.Tensor):
            raise TypeError("mu must be a torch.Tensor.")
        if not isinstance(buoyancy_requires_grad, bool):
            raise TypeError(
                f"buoyancy_requires_grad must be bool, "
                f"got {type(buoyancy_requires_grad).__name__}",
            )
        if not isinstance(buoyancy, torch.Tensor):
            raise TypeError("buoyancy must be a torch.Tensor.")
        self.lamb = torch.nn.Parameter(lamb, requires_grad=lamb_requires_grad)
        self.mu = torch.nn.Parameter(mu, requires_grad=mu_requires_grad)
        self.buoyancy = torch.nn.Parameter(
            buoyancy,
            requires_grad=buoyancy_requires_grad,
        )
        self.grid_spacing = grid_spacing

    def forward(
        self,
        dt: float,
        source_amplitudes_y: Optional[torch.Tensor] = None,
        source_amplitudes_x: Optional[torch.Tensor] = None,
        source_locations_y: Optional[torch.Tensor] = None,
        source_locations_x: Optional[torch.Tensor] = None,
        receiver_locations_y: Optional[torch.Tensor] = None,
        receiver_locations_x: Optional[torch.Tensor] = None,
        receiver_locations_p: Optional[torch.Tensor] = None,
        accuracy: int = 4,
        pml_width: Union[int, Sequence[int]] = 20,
        pml_freq: Optional[float] = None,
        max_vel: Optional[float] = None,
        survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
        vy_0: Optional[torch.Tensor] = None,
        vx_0: Optional[torch.Tensor] = None,
        sigmayy_0: Optional[torch.Tensor] = None,
        sigmaxy_0: Optional[torch.Tensor] = None,
        sigmaxx_0: Optional[torch.Tensor] = None,
        m_vyy_0: Optional[torch.Tensor] = None,
        m_vyx_0: Optional[torch.Tensor] = None,
        m_vxy_0: Optional[torch.Tensor] = None,
        m_vxx_0: Optional[torch.Tensor] = None,
        m_sigmayyy_0: Optional[torch.Tensor] = None,
        m_sigmaxyy_0: Optional[torch.Tensor] = None,
        m_sigmaxyx_0: Optional[torch.Tensor] = None,
        m_sigmaxxx_0: Optional[torch.Tensor] = None,
        origin: Optional[Sequence[int]] = None,
        nt: Optional[int] = None,
        model_gradient_sampling_interval: int = 1,
        freq_taper_frac: float = 0.0,
        time_pad_frac: float = 0.0,
        time_taper: bool = False,
        forward_callback: Optional[deepwave.common.Callback] = None,
        backward_callback: Optional[deepwave.common.Callback] = None,
        callback_frequency: int = 1,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Perform forward propagation/modelling.

        The inputs are the same as for :func:`elastic` except that `lamb`,
        `mu`, `buoyancy`, and `grid_spacing` do not need to be provided again.
        See :func:`elastic` for a description of the inputs and outputs.
        """
        return elastic(
            self.lamb,
            self.mu,
            self.buoyancy,
            self.grid_spacing,
            dt,
            source_amplitudes_y=source_amplitudes_y,
            source_amplitudes_x=source_amplitudes_x,
            source_locations_y=source_locations_y,
            source_locations_x=source_locations_x,
            receiver_locations_y=receiver_locations_y,
            receiver_locations_x=receiver_locations_x,
            receiver_locations_p=receiver_locations_p,
            accuracy=accuracy,
            pml_width=pml_width,
            pml_freq=pml_freq,
            max_vel=max_vel,
            survey_pad=survey_pad,
            vy_0=vy_0,
            vx_0=vx_0,
            sigmayy_0=sigmayy_0,
            sigmaxy_0=sigmaxy_0,
            sigmaxx_0=sigmaxx_0,
            m_vyy_0=m_vyy_0,
            m_vyx_0=m_vyx_0,
            m_vxy_0=m_vxy_0,
            m_vxx_0=m_vxx_0,
            m_sigmayyy_0=m_sigmayyy_0,
            m_sigmaxyy_0=m_sigmaxyy_0,
            m_sigmaxyx_0=m_sigmaxyx_0,
            m_sigmaxxx_0=m_sigmaxxx_0,
            origin=origin,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
            forward_callback=forward_callback,
            backward_callback=backward_callback,
            callback_frequency=callback_frequency,
        )


def _average_adjacent(receiver_amplitudes: torch.Tensor) -> torch.Tensor:
    """Averages adjacent elements in a tensor.

    This is used to average velocity receiver samples at t-1/2 and t+1/2
    to approximately get samples at t.

    Args:
        receiver_amplitudes: A torch.Tensor.

    Returns:
        A torch.Tensor with the averaged values.

    """
    if receiver_amplitudes.numel() == 0:
        return receiver_amplitudes
    return (receiver_amplitudes[1:] + receiver_amplitudes[:-1]) / 2


def elastic(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitudes_y: Optional[torch.Tensor] = None,
    source_amplitudes_x: Optional[torch.Tensor] = None,
    source_locations_y: Optional[torch.Tensor] = None,
    source_locations_x: Optional[torch.Tensor] = None,
    receiver_locations_y: Optional[torch.Tensor] = None,
    receiver_locations_x: Optional[torch.Tensor] = None,
    receiver_locations_p: Optional[torch.Tensor] = None,
    accuracy: int = 4,
    pml_width: Union[int, Sequence[int]] = 20,
    pml_freq: Optional[float] = None,
    max_vel: Optional[float] = None,
    survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
    vy_0: Optional[torch.Tensor] = None,
    vx_0: Optional[torch.Tensor] = None,
    sigmayy_0: Optional[torch.Tensor] = None,
    sigmaxy_0: Optional[torch.Tensor] = None,
    sigmaxx_0: Optional[torch.Tensor] = None,
    m_vyy_0: Optional[torch.Tensor] = None,
    m_vyx_0: Optional[torch.Tensor] = None,
    m_vxy_0: Optional[torch.Tensor] = None,
    m_vxx_0: Optional[torch.Tensor] = None,
    m_sigmayyy_0: Optional[torch.Tensor] = None,
    m_sigmaxyy_0: Optional[torch.Tensor] = None,
    m_sigmaxyx_0: Optional[torch.Tensor] = None,
    m_sigmaxxx_0: Optional[torch.Tensor] = None,
    origin: Optional[Sequence[int]] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    forward_callback: Optional[deepwave.common.Callback] = None,
    backward_callback: Optional[deepwave.common.Callback] = None,
    callback_frequency: int = 1,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Elastic wave propagation (functional interface).

    This function performs forward modelling with the elastic wave equation.
    The outputs are differentiable with respect to the input float torch.Tensors.

    For computational performance, multiple shots may be propagated
    simultaneously.

    This propagator uses a staggered grid. See the
    `documentation <https://ausargeo.com/deepwave/elastic.html>`_
    for a description of the grid layout.

    Args:
        lamb: The first Lamé parameter model, lambda.
        mu: The second Lamé parameter model.
        buoyancy: The buoyancy (1/density) model.
        grid_spacing: The spatial grid cell size. It can be a single number
            that will be used for all dimensions, or a number for each
            dimension.
        dt: The time step interval of the input and output (internally a
            smaller interval may be used in propagation to obey the CFL
            condition for stability).
        source_amplitudes_y: A Tensor with dimensions [shot, source, time]
            containing time samples of the source wavelets for sources
            oriented in the first spatial dimension.
        source_amplitudes_x: A Tensor containing source wavelet time samples
            for sources oriented in the second spatial dimension.
        source_locations_y: A Tensor with dimensions [shot, source, 2],
            containing the index in the two spatial dimensions of the cell
            that each source oriented in the first spatial dimension is
            located in, relative to the origin of the model. Setting both
            coordinates to deepwave.IGNORE_LOCATION will result in the
            source being ignored.
        source_locations_x: A Tensor containing the locations of sources
            oriented in the second spatial dimension.
        receiver_locations_y: A Tensor with dimensions [shot, receiver, 2],
            containing the coordinates of the cell containing each receiver
            oriented in the first spatial dimension. Setting both
            coordinates to deepwave.IGNORE_LOCATION will result in the
            receiver being ignored.
        receiver_locations_x: A Tensor containing the coordinates of the
            receivers oriented in the second spatial dimension.
        receiver_locations_p: A Tensor containing the coordinates of the
            pressure receivers.
        accuracy: The finite difference order of accuracy. Possible values are
            2 and 4. Default 4.
        pml_width: A single number, or a sequence of numbers for each side,
            specifying the width (in number of cells) of the PML
            that prevents reflections from the edges of the model.
            If a single value is provided, it will be used for all edges.
            If a sequence is provided, it should contain values for the
            edges in the following order:

            - beginning of first dimension
            - end of first dimension
            - beginning of second dimension
            - end of second dimension

            Larger values result in smaller reflections, with values of 10
            to 20 being typical. For a reflective or "free" surface, set the
            value for that edge to be zero. The wavespeed in the PML region
            is obtained by replicating the values on the edge of the
            model. Optional, default 20.
        pml_freq: The frequency to use when constructing the PML. This is
            usually the dominant frequency of the source wavelet. Optional,
            default 25 Hz (if `dt` is in seconds).
        max_vel: The maximum velocity, used for the CFL condition and PML. If
            not specified, the actual maximum P-wave and S-wave velocities
            in the model will be used. Optional, default None.
        survey_pad: Padding to apply around sources and receivers to restrict
            the simulation domain. See the documentation for the `scalar`
            propagator for a full description.
        vy_0: Initial vy (velocity in the first dimension) wavefield at time
            step -1/2.
        vx_0: Initial vx (velocity in the second dimension) wavefield.
        sigmayy_0: Initial value for the yy component of the stress field at
            time step 0.
        sigmaxy_0: Initial value for the xy component of the stress field at
            time step 0.
        sigmaxx_0: Initial value for the xx component of the stress field at
            time step 0.
        m_vyy_0: Initial value for the "memory variable" for the yy component
            of the velocity, used in the PML.
        m_vyx_0: Initial value for the "memory variable" for the yx component
            of the velocity, used in the PML.
        m_vxy_0: Initial value for the "memory variable" for the xy component
            of the velocity, used in the PML.
        m_vxx_0: Initial value for the "memory variable" for the xx component
            of the velocity, used in the PML.
        m_sigmayyy_0: Initial value for the "memory variable" for the yyy
            component of the stress, used in the PML.
        m_sigmaxyy_0: Initial value for the "memory variable" for the xyy
            component of the stress, used in the PML.
        m_sigmaxyx_0: Initial value for the "memory variable" for the xyx
            component of the stress, used in the PML.
        m_sigmaxxx_0: Initial value for the "memory variable" for the xxx
            component of the stress, used in the PML.
        origin: The origin of the provided initial wavefields relative to the
            origin of the model.
        nt: If the source amplitudes are not provided then you must instead
            specify the number of time steps to run the simulation for.
        model_gradient_sampling_interval: The number of time steps between
            contributions to the model gradient.
        freq_taper_frac: The fraction of the end of the source and
            receiver amplitudes in the frequency domain to cosine taper if
            they are resampled due to the CFL condition.
        time_pad_frac: The amount of zero padding that will be added to the
            source and receiver amplitudes before resampling and removed
            afterwards, if they are resampled due to the CFL condition, as a
            fraction of their length.
        time_taper: Whether to apply a Hann window in time to source and
            receiver amplitudes.
        forward_callback: A function that will be called during the forward
            pass. See :class:`deepwave.common.CallbackState` for the
            state that will be provided to the function.
        backward_callback: A function that will be called during the backward
            pass. See :class:`deepwave.common.CallbackState` for the
            state that will be provided to the function.
        callback_frequency: The number of internal time steps between calls
            to the callback.

    Returns:
        Tuple:

            - vy: Final velocity wavefield in the y-dimension.
            - vx: Final velocity wavefield in the x-dimension.
            - sigmayy: Final stress wavefield (yy component).
            - sigmaxy: Final stress wavefield (xy component).
            - sigmaxx: Final stress wavefield (xx component).
            - m_vyy: Final velocity memory variable for the PML.
            - m_vyx: Final velocity memory variable for the PML.
            - m_vxy: Final velocity memory variable for the PML.
            - m_vxx: Final velocity memory variable for the PML.
            - m_sigmayyy: Final stress memory variable for the PML.
            - m_sigmaxyy: Final stress memory variable for the PML.
            - m_sigmaxyx: Final stress memory variable for the PML.
            - m_sigmaxxx: Final stress memory variable for the PML.
            - receiver_amplitudes_p: Recorded pressure receiver amplitudes.
            - receiver_amplitudes_y: Recorded y-component receiver amplitudes.
            - receiver_amplitudes_x: Recorded x-component receiver amplitudes.

    """
    # Check that sources and receivers are not on the last row or column,
    # as these are not used
    if (
        source_locations_y is not None
        and source_locations_y[..., 1].max() >= lamb.shape[1] - 1
    ):
        raise RuntimeError(
            "With the provided model, the maximum y source "
            f"location in the second dimension must be less than {lamb.shape[1] - 1}.",
        )
    if (
        receiver_locations_y is not None
        and receiver_locations_y[..., 1].max() >= lamb.shape[1] - 1
    ):
        raise RuntimeError(
            "With the provided model, the maximum y "
            "receiver location in the second dimension "
            f"must be less than {lamb.shape[1] - 1}.",
        )
    if source_locations_x is not None:
        dim_location = source_locations_x[..., 0]
        dim_location = dim_location[dim_location != deepwave.common.IGNORE_LOCATION]
        if dim_location.min() <= 0:
            raise RuntimeError(
                "The minimum x source "
                "location in the first dimension must be "
                "greater than 0.",
            )
    if receiver_locations_x is not None:
        dim_location = receiver_locations_x[..., 0]
        dim_location = dim_location[dim_location != deepwave.common.IGNORE_LOCATION]
        if dim_location.min() <= 0:
            raise RuntimeError(
                "The minimum x receiver "
                "location in the first dimension must be "
                "greater than 0.",
            )
    if receiver_locations_p is not None:
        dim_location = receiver_locations_p[..., 0]
        dim_location = dim_location[dim_location != deepwave.common.IGNORE_LOCATION]
        if (
            receiver_locations_p[..., 1].max() >= lamb.shape[1] - 1
            or dim_location.min() <= 0
        ):
            raise RuntimeError(
                "With the provided model, the pressure "
                "receiver locations in the second dimension "
                f"must be less than {lamb.shape[1] - 1} and "
                "in the first dimension must be greater than 0.",
            )
    if accuracy not in [2, 4]:
        raise RuntimeError("The accuracy must be 2 or 4.")
    vp, vs, _ = deepwave.common.lambmubuoyancy_to_vpvsrho(
        lamb.abs(),
        mu.abs(),
        buoyancy.abs(),
    )
    max_model_vel = max(vp.abs().max().item(), vs.abs().max().item())
    vp_nonzero = vp[vp != 0]
    min_nonzero_vp = vp_nonzero.abs().min().item() if vp_nonzero.numel() > 0 else 0.0
    vs_nonzero = vs[vs != 0]
    min_nonzero_vs = vs_nonzero.abs().min().item() if vs_nonzero.numel() > 0 else 0.0
    if min_nonzero_vp == 0 and min_nonzero_vs == 0:
        min_nonzero_model_vel = 0.0
    elif min_nonzero_vp == 0:
        min_nonzero_model_vel = float(min_nonzero_vs)
    elif min_nonzero_vs == 0:
        min_nonzero_model_vel = float(min_nonzero_vp)
    else:
        min_nonzero_model_vel = float(min(min_nonzero_vp, min_nonzero_vs))
    fd_pad = [0] * 4

    (
        models,
        source_amplitudes,
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
        [lamb, mu, buoyancy],
        ["replicate"] * 3,
        grid_spacing,
        dt,
        [source_amplitudes_y, source_amplitudes_x],
        [source_locations_y, source_locations_x],
        [receiver_locations_y, receiver_locations_x, receiver_locations_p],
        accuracy,
        fd_pad,
        pml_width,
        pml_freq,
        max_vel,
        min_nonzero_model_vel,
        max_model_vel,
        survey_pad,
        [
            vy_0,
            vx_0,
            sigmayy_0,
            sigmaxy_0,
            sigmaxx_0,
            m_vyy_0,
            m_vyx_0,
            m_vxy_0,
            m_vxx_0,
            m_sigmayyy_0,
            m_sigmaxyy_0,
            m_sigmaxyx_0,
            m_sigmaxxx_0,
        ],
        origin,
        nt,
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        2,
    )

    if any(s <= (accuracy + 1) * 2 for s in models[0].shape[1:]):
        raise RuntimeError(
            f"The model must have at least {(accuracy + 1) * 2 + 1} "
            "elements in each dimension (including PML).",
        )

    ny, nx = models[0].shape[-2:]
    # source_amplitudes_y
    # Need to interpolate buoyancy to vy
    mask = sources_i[0] == deepwave.common.IGNORE_LOCATION
    sources_i_masked = sources_i[0].clone()
    sources_i_masked[mask] = 0
    if source_amplitudes[0].numel() > 0:
        source_amplitudes[0] = (
            (
                source_amplitudes[0]
                * (
                    models[2]
                    .view(-1, ny * nx)
                    .expand(n_shots, -1)
                    .gather(1, sources_i_masked)
                    + models[2]
                    .view(-1, ny * nx)
                    .expand(n_shots, -1)
                    .gather(1, sources_i_masked + 1)
                    + models[2]
                    .view(-1, ny * nx)
                    .expand(n_shots, -1)
                    .gather(1, sources_i_masked + nx)
                    + models[2]
                    .view(-1, ny * nx)
                    .expand(n_shots, -1)
                    .gather(1, sources_i_masked + nx + 1)
                )
            )
            / 4
            * dt
        )
    # source_amplitudes_x
    mask = sources_i[1] == deepwave.common.IGNORE_LOCATION
    sources_i_masked = sources_i[1].clone()
    sources_i_masked[mask] = 0
    if source_amplitudes[1].numel() > 0:
        source_amplitudes[1] = (
            source_amplitudes[1]
            * (
                models[2]
                .view(-1, ny * nx)
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
        ny,
        nx,
    )

    if not isinstance(callback_frequency, int):
        raise TypeError("callback_frequency must be an int.")
    if callback_frequency <= 0:
        raise ValueError("callback_frequency must be positive.")

    # Run the forward propagator
    (
        vy,
        vx,
        sigmayy,
        sigmaxy,
        sigmaxx,
        m_vyy,
        m_vyx,
        m_vxy,
        m_vxx,
        m_sigmayyy,
        m_sigmaxyy,
        m_sigmaxyx,
        m_sigmaxxx,
        receiver_amplitudes_p,
        receiver_amplitudes_y,
        receiver_amplitudes_x,
    ) = elastic_func(
        *models,
        *source_amplitudes,
        *wavefields,
        *pml_profiles,
        *sources_i,
        *receivers_i,
        *grid_spacing,
        dt,
        nt,
        step_ratio * model_gradient_sampling_interval,
        accuracy,
        pml_width,
        n_shots,
        forward_callback,
        backward_callback,
        callback_frequency,
    )

    receiver_amplitudes_y = _average_adjacent(receiver_amplitudes_y)
    receiver_amplitudes_x = _average_adjacent(receiver_amplitudes_x)
    receiver_amplitudes_y = deepwave.common.downsample_and_movedim(
        receiver_amplitudes_y,
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
    )
    receiver_amplitudes_x = deepwave.common.downsample_and_movedim(
        receiver_amplitudes_x,
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
    )
    receiver_amplitudes_p = deepwave.common.downsample_and_movedim(
        receiver_amplitudes_p,
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
    )

    return (
        vy,
        vx,
        sigmayy,
        sigmaxy,
        sigmaxx,
        m_vyy,
        m_vyx,
        m_vxy,
        m_vxx,
        m_sigmayyy,
        m_sigmaxyy,
        m_sigmaxyx,
        m_sigmaxxx,
        receiver_amplitudes_p,
        receiver_amplitudes_y,
        receiver_amplitudes_x,
    )


def zero_edges(tensor: torch.Tensor, ny: int, nx: int) -> None:
    """Sets the values on all four edges of a 2D tensor to zero in-place.

    Args:
        tensor: The input 2D torch.Tensor to modify, with shape (batch, ny, nx).
        ny: The size of the second dimension (rows).
        nx: The size of the third dimension (columns).

    """
    tensor[:, ny - 1, :] = 0
    tensor[:, :, nx - 1] = 0
    tensor[:, 0, :] = 0
    tensor[:, :, 0] = 0


def zero_edge_top(tensor: torch.Tensor) -> None:
    """Sets the values on the top edge of a 2D tensor to zero in-place.

    Args:
        tensor: The input 2D torch.Tensor to modify, with shape (batch, ny, nx).

    """
    tensor[:, 0, :] = 0


def zero_edge_right(tensor: torch.Tensor, nx: int) -> None:
    """Sets the values on the right edge of a 2D tensor to zero in-place.

    Args:
        tensor: The input 2D torch.Tensor to modify, with shape (batch, ny, nx).
        nx: The size of the third dimension (columns).

    """
    tensor[:, :, nx - 1] = 0


def zero_interior(
    tensor: torch.Tensor,
    ybegin: int,
    yend: int,
    xbegin: int,
    xend: int,
) -> torch.Tensor:
    """Zeros out a specified rectangular interior region of a 2D tensor.

    This function returns a new tensor and does not modify the input tensor.

    Args:
        tensor: The input 2D torch.Tensor.
        ybegin: The starting index of the y-dimension for the region to zero.
        yend: The ending index of the y-dimension for the region to zero.
        xbegin: The starting index of the x-dimension for the region to zero.
        xend: The ending index of the x-dimension for the region to zero.

    Returns:
        A new torch.Tensor with the specified interior region zeroed out.

    """
    tensor = tensor.clone()
    tensor[:, ybegin:yend, xbegin:xend] = 0
    return tensor


class ElasticForwardFunc(torch.autograd.Function):
    """Forward propagation of the elastic wave equation."""

    @staticmethod
    def forward(
        ctx: Any,
        lamb: torch.Tensor,
        mu: torch.Tensor,
        buoyancy: torch.Tensor,
        source_amplitudes_y: torch.Tensor,
        source_amplitudes_x: torch.Tensor,
        vy: torch.Tensor,
        vx: torch.Tensor,
        sigmayy: torch.Tensor,
        sigmaxy: torch.Tensor,
        sigmaxx: torch.Tensor,
        m_vyy: torch.Tensor,
        m_vyx: torch.Tensor,
        m_vxy: torch.Tensor,
        m_vxx: torch.Tensor,
        m_sigmayyy: torch.Tensor,
        m_sigmaxyy: torch.Tensor,
        m_sigmaxyx: torch.Tensor,
        m_sigmaxxx: torch.Tensor,
        ay: torch.Tensor,
        ayh: torch.Tensor,
        ax: torch.Tensor,
        axh: torch.Tensor,
        by: torch.Tensor,
        byh: torch.Tensor,
        bx: torch.Tensor,
        bxh: torch.Tensor,
        sources_y_i: torch.Tensor,
        sources_x_i: torch.Tensor,
        receivers_y_i: torch.Tensor,
        receivers_x_i: torch.Tensor,
        receivers_p_i: torch.Tensor,
        dy: float,
        dx: float,
        dt: float,
        nt: int,
        step_ratio: int,
        accuracy: int,
        pml_width: List[int],
        n_shots: int,
        forward_callback: Optional[deepwave.common.Callback],
        backward_callback: Optional[deepwave.common.Callback],
        callback_frequency: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Performs the forward propagation of the elastic wave equation.

        This method is called by PyTorch during the forward pass. It prepares
        the input tensors, calls the appropriate C/CUDA function for wave
        propagation, and saves necessary tensors for the backward pass.

        Args:
            ctx: A context object for saving information for the backward pass.
            lamb: The first Lam'e parameter (lambda) model tensor.
            mu: The second Lam'e parameter (mu) model tensor.
            buoyancy: The buoyancy (1/density) model tensor.
            source_amplitudes_y: Source amplitudes for y-component sources.
            source_amplitudes_x: Source amplitudes for x-component sources.
            vy: Initial velocity wavefield in y-direction.
            vx: Initial velocity wavefield in x-direction.
            sigmayy: Initial stress wavefield (sigma_yy).
            sigmaxy: Initial stress wavefield (sigma_xy).
            sigmaxx: Initial stress wavefield (sigma_xx).
            m_vyy: Initial memory variable for vy in PML.
            m_vyx: Initial memory variable for vx in PML (y-direction).
            m_vxy: Initial memory variable for vy in PML (x-direction).
            m_vxx: Initial memory variable for vx in PML.
            m_sigmayyy: Initial memory variable for sigmayy in PML.
            m_sigmaxyy: Initial memory variable for sigmaxy in PML (y-direction).
            m_sigmaxyx: Initial memory variable for sigmaxy in PML (x-direction).
            m_sigmaxxx: Initial memory variable for sigmaxx in PML.
            ay: PML absorption profile for y-dimension (a-coefficient).
            ayh: PML absorption profile for y-dimension (a-coefficient, half-step).
            ax: PML absorption profile for x-dimension (a-coefficient).
            axh: PML absorption profile for x-dimension (a-coefficient, half-step).
            by: PML absorption profile for y-dimension (b-coefficient).
            byh: PML absorption profile for y-dimension (b-coefficient, half-step).
            bx: PML absorption profile for x-dimension (b-coefficient).
            bxh: PML absorption profile for x-dimension (b-coefficient, half-step).
            sources_y_i: 1D indices of y-component source locations.
            sources_x_i: 1D indices of x-component source locations.
            receivers_y_i: 1D indices of y-component receiver locations.
            receivers_x_i: 1D indices of x-component receiver locations.
            receivers_p_i: 1D indices of pressure receiver locations.
            dy: Grid spacing in y-dimension.
            dx: Grid spacing in x-dimension.
            dt: Time step interval.
            nt: Total number of time steps.
            step_ratio: Ratio between user dt and internal dt.
            accuracy: Finite difference accuracy order.
            pml_width: List of PML widths for each side.
            n_shots: Number of shots in the batch.
            forward_callback: The forward callback.
            backward_callback: The backward callback.
            callback_frequency: The callback frequency.

        Returns:
            A tuple containing the final wavefields, memory variables, and
            receiver amplitudes.

        """
        lamb = lamb.contiguous()
        mu = mu.contiguous()
        buoyancy = buoyancy.contiguous()
        source_amplitudes_y = source_amplitudes_y.contiguous()
        source_amplitudes_x = source_amplitudes_x.contiguous()
        ay = ay.contiguous()
        ayh = ayh.contiguous()
        ax = ax.contiguous()
        axh = axh.contiguous()
        by = by.contiguous()
        byh = byh.contiguous()
        bx = bx.contiguous()
        bxh = bxh.contiguous()
        sources_y_i = sources_y_i.contiguous()
        sources_x_i = sources_x_i.contiguous()
        receivers_y_i = receivers_y_i.contiguous()
        receivers_x_i = receivers_x_i.contiguous()
        receivers_p_i = receivers_p_i.contiguous()

        device = lamb.device
        dtype = lamb.dtype
        ny = lamb.shape[-2]
        nx = lamb.shape[-1]
        n_sources_y_per_shot = sources_y_i.numel() // n_shots
        n_sources_x_per_shot = sources_x_i.numel() // n_shots
        n_receivers_y_per_shot = receivers_y_i.numel() // n_shots
        n_receivers_x_per_shot = receivers_x_i.numel() // n_shots
        n_receivers_p_per_shot = receivers_p_i.numel() // n_shots
        dvydbuoyancy = torch.empty(0, device=device, dtype=dtype)
        dvxdbuoyancy = torch.empty(0, device=device, dtype=dtype)
        dvydy_store = torch.empty(0, device=device, dtype=dtype)
        dvxdx_store = torch.empty(0, device=device, dtype=dtype)
        dvydxdvxdy_store = torch.empty(0, device=device, dtype=dtype)
        receiver_amplitudes_y = torch.empty(0, device=device, dtype=dtype)
        receiver_amplitudes_x = torch.empty(0, device=device, dtype=dtype)
        receiver_amplitudes_p = torch.empty(0, device=device, dtype=dtype)

        pml_y0 = max(pml_width[0], accuracy // 2)
        pml_y1 = min(ny - pml_width[1], ny - accuracy // 2)
        pml_x0 = max(pml_width[2], accuracy // 2)
        pml_x1 = min(nx - pml_width[3], nx - accuracy // 2)

        size_with_batch = (n_shots, *lamb.shape[-2:])
        vy = deepwave.common.create_or_pad(
            vy,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        vx = deepwave.common.create_or_pad(
            vx,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        sigmayy = deepwave.common.create_or_pad(
            sigmayy,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        sigmaxy = deepwave.common.create_or_pad(
            sigmaxy,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        sigmaxx = deepwave.common.create_or_pad(
            sigmaxx,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vyy = deepwave.common.create_or_pad(
            m_vyy,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vyx = deepwave.common.create_or_pad(
            m_vyx,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vxy = deepwave.common.create_or_pad(
            m_vxy,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vxx = deepwave.common.create_or_pad(
            m_vxx,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmayyy = deepwave.common.create_or_pad(
            m_sigmayyy,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmaxyy = deepwave.common.create_or_pad(
            m_sigmaxyy,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmaxyx = deepwave.common.create_or_pad(
            m_sigmaxyx,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmaxxx = deepwave.common.create_or_pad(
            m_sigmaxxx,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        zero_edges(sigmaxy, ny, nx)
        zero_edges(m_vxy, ny, nx)
        zero_edges(m_vyx, ny, nx)
        zero_edge_right(vy, nx)
        zero_edge_right(m_sigmayyy, nx)
        zero_edge_right(m_sigmaxyx, nx)
        zero_edge_top(vx)
        zero_edge_top(m_sigmaxyy)
        zero_edge_top(m_sigmaxxx)
        zero_edge_top(sigmaxx)
        zero_edge_top(m_vxx)
        zero_edge_top(sigmayy)
        zero_edge_top(m_vyy)
        zero_edge_right(sigmaxx, nx)
        zero_edge_right(m_vxx, nx)
        zero_edge_right(sigmayy, nx)
        zero_edge_right(m_vyy, nx)
        m_sigmayyy = zero_interior(m_sigmayyy, pml_y0, pml_y1, 0, nx)
        m_sigmaxyx = zero_interior(m_sigmaxyx, 0, ny, pml_x0, pml_x1 - 1)
        m_sigmaxyy = zero_interior(m_sigmaxyy, pml_y0 + 1, pml_y1, 0, nx)
        m_sigmaxxx = zero_interior(m_sigmaxxx, 0, ny, pml_x0, pml_x1)
        m_vyy = zero_interior(m_vyy, pml_y0 + 1, pml_y1, 0, nx)
        m_vxx = zero_interior(m_vxx, 0, ny, pml_x0, pml_x1 - 1)
        m_vyx = zero_interior(m_vyx, 0, ny, pml_x0 + 1, pml_x1 - 1)
        m_vxy = zero_interior(m_vxy, pml_y0 + 1, pml_y1 - 1, 0, nx)

        lamb_batched = lamb.ndim == 3 and lamb.shape[0] > 1
        mu_batched = mu.ndim == 3 and mu.shape[0] > 1
        buoyancy_batched = buoyancy.ndim == 3 and buoyancy.shape[0] > 1

        if buoyancy.requires_grad:
            dvydbuoyancy.resize_(nt // step_ratio, n_shots, ny, nx)
            dvxdbuoyancy.resize_(nt // step_ratio, n_shots, ny, nx)
        if lamb.requires_grad or mu.requires_grad:
            dvydy_store.resize_(nt // step_ratio, n_shots, ny, nx)
            dvxdx_store.resize_(nt // step_ratio, n_shots, ny, nx)
        if mu.requires_grad:
            dvydxdvxdy_store.resize_(nt // step_ratio, n_shots, ny, nx)

        if receivers_y_i.numel() > 0:
            receiver_amplitudes_y.resize_(nt + 1, n_shots, n_receivers_y_per_shot)
            receiver_amplitudes_y.fill_(0)
        if receivers_x_i.numel() > 0:
            receiver_amplitudes_x.resize_(nt + 1, n_shots, n_receivers_x_per_shot)
            receiver_amplitudes_x.fill_(0)
        if receivers_p_i.numel() > 0:
            receiver_amplitudes_p.resize_(nt, n_shots, n_receivers_p_per_shot)
            receiver_amplitudes_p.fill_(0)

        if lamb.is_cuda:
            aux = lamb.get_device()
        elif deepwave.backend_utils.USE_OPENMP:
            aux = min(n_shots, torch.get_num_threads())
        else:
            aux = 1
        forward = deepwave.backend_utils.get_backend_function(
            "elastic",
            "forward",
            accuracy,
            dtype,
            lamb.device,
        )

        if forward_callback is None:
            callback_frequency = nt // step_ratio

        if vy.numel() > 0 and nt > 0:
            for step in range(0, nt // step_ratio, callback_frequency):
                if forward_callback is not None:
                    state = deepwave.common.CallbackState(
                        dt,
                        step,
                        {
                            "vy_0": vy,
                            "vx_0": vx,
                            "sigmayy_0": sigmayy,
                            "sigmaxy_0": sigmaxy,
                            "sigmaxx_0": sigmaxx,
                            "m_vyy_0": m_vyy,
                            "m_vyx_0": m_vyx,
                            "m_vxy_0": m_vxy,
                            "m_vxx_0": m_vxx,
                            "m_sigmayyy_0": m_sigmayyy,
                            "m_sigmaxyy_0": m_sigmaxyy,
                            "m_sigmaxyx_0": m_sigmaxyx,
                            "m_sigmaxxx_0": m_sigmaxxx,
                        },
                        {"lamb": lamb, "mu": mu, "buoyancy": buoyancy},
                        {},
                        [0] * 4,
                        pml_width,
                    )
                    forward_callback(state)
                step_nt = min(nt // step_ratio - step, callback_frequency)
                forward(
                    lamb.data_ptr(),
                    mu.data_ptr(),
                    buoyancy.data_ptr(),
                    source_amplitudes_y.data_ptr(),
                    source_amplitudes_x.data_ptr(),
                    vy.data_ptr(),
                    vx.data_ptr(),
                    sigmayy.data_ptr(),
                    sigmaxy.data_ptr(),
                    sigmaxx.data_ptr(),
                    m_vyy.data_ptr(),
                    m_vyx.data_ptr(),
                    m_vxy.data_ptr(),
                    m_vxx.data_ptr(),
                    m_sigmayyy.data_ptr(),
                    m_sigmaxyy.data_ptr(),
                    m_sigmaxyx.data_ptr(),
                    m_sigmaxxx.data_ptr(),
                    dvydbuoyancy.data_ptr(),
                    dvxdbuoyancy.data_ptr(),
                    dvydy_store.data_ptr(),
                    dvxdx_store.data_ptr(),
                    dvydxdvxdy_store.data_ptr(),
                    receiver_amplitudes_y.data_ptr(),
                    receiver_amplitudes_x.data_ptr(),
                    receiver_amplitudes_p.data_ptr(),
                    ay.data_ptr(),
                    ayh.data_ptr(),
                    ax.data_ptr(),
                    axh.data_ptr(),
                    by.data_ptr(),
                    byh.data_ptr(),
                    bx.data_ptr(),
                    bxh.data_ptr(),
                    sources_y_i.data_ptr(),
                    sources_x_i.data_ptr(),
                    receivers_y_i.data_ptr(),
                    receivers_x_i.data_ptr(),
                    receivers_p_i.data_ptr(),
                    dy,
                    dx,
                    dt,
                    step_nt * step_ratio,
                    n_shots,
                    ny,
                    nx,
                    n_sources_y_per_shot,
                    n_sources_x_per_shot,
                    n_receivers_y_per_shot,
                    n_receivers_x_per_shot,
                    n_receivers_p_per_shot,
                    step_ratio,
                    lamb.requires_grad,
                    mu.requires_grad,
                    buoyancy.requires_grad,
                    lamb_batched,
                    mu_batched,
                    buoyancy_batched,
                    step * step_ratio,
                    pml_y0,
                    pml_y1,
                    pml_x0,
                    pml_x1,
                    aux,
                )

        if (
            lamb.requires_grad
            or mu.requires_grad
            or buoyancy.requires_grad
            or source_amplitudes_y.requires_grad
            or source_amplitudes_x.requires_grad
            or vy.requires_grad
            or vx.requires_grad
            or sigmayy.requires_grad
            or sigmaxy.requires_grad
            or sigmaxx.requires_grad
            or m_vyy.requires_grad
            or m_vyx.requires_grad
            or m_vxy.requires_grad
            or m_vxx.requires_grad
            or m_sigmayyy.requires_grad
            or m_sigmaxyy.requires_grad
            or m_sigmaxyx.requires_grad
            or m_sigmaxxx.requires_grad
        ):
            ctx.save_for_backward(
                lamb,
                mu,
                buoyancy,
                ay,
                ayh,
                ax,
                axh,
                by,
                byh,
                bx,
                bxh,
                sources_y_i,
                sources_x_i,
                receivers_y_i,
                receivers_x_i,
                receivers_p_i,
                dvydbuoyancy,
                dvxdbuoyancy,
                dvydy_store,
                dvxdx_store,
                dvydxdvxdy_store,
            )
            ctx.dy = dy
            ctx.dx = dx
            ctx.dt = dt
            ctx.nt = nt
            ctx.n_shots = n_shots
            ctx.step_ratio = step_ratio
            ctx.accuracy = accuracy
            ctx.pml_width = pml_width
            ctx.source_amplitudes_y_requires_grad = source_amplitudes_y.requires_grad
            ctx.source_amplitudes_x_requires_grad = source_amplitudes_x.requires_grad
            ctx.backward_callback = backward_callback
            ctx.callback_frequency = callback_frequency

        return (
            vy,
            vx,
            sigmayy,
            sigmaxy,
            sigmaxx,
            m_vyy,
            m_vyx,
            m_vxy,
            m_vxx,
            m_sigmayyy,
            m_sigmaxyy,
            m_sigmaxyx,
            m_sigmaxxx,
            receiver_amplitudes_p,
            receiver_amplitudes_y,
            receiver_amplitudes_x,
        )

    @staticmethod
    @torch.autograd.function.once_differentiable  # type: ignore[misc]
    def backward(
        ctx: Any,
        vy: torch.Tensor,
        vx: torch.Tensor,
        sigmayy: torch.Tensor,
        sigmaxy: torch.Tensor,
        sigmaxx: torch.Tensor,
        m_vyy: torch.Tensor,
        m_vyx: torch.Tensor,
        m_vxy: torch.Tensor,
        m_vxx: torch.Tensor,
        m_sigmayyy: torch.Tensor,
        m_sigmaxyy: torch.Tensor,
        m_sigmaxyx: torch.Tensor,
        m_sigmaxxx: torch.Tensor,
        grad_r_p: torch.Tensor,
        grad_r_y: torch.Tensor,
        grad_r_x: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Computes the gradients during the backward pass.

        This method is called by PyTorch during the backward pass to compute
        gradients with respect to the inputs of the forward pass.

        Args:
            ctx: A context object with saved information from the forward pass.
            vy: Gradient of the loss wrt the output `vy`.
            vx: Gradient of the loss wrt the output `vx`.
            sigmayy: Gradient of the loss wrt the output `sigmayy`.
            sigmaxy: Gradient of the loss wrt the output `sigmaxy`.
            sigmaxx: Gradient of the loss wrt the output `sigmaxx`.
            m_vyy: Gradient of the loss wrt the output `m_vyy`.
            m_vyx: Gradient of the loss wrt the output `m_vyx`.
            m_vxy: Gradient of the loss wrt the output `m_vxy`.
            m_vxx: Gradient of the loss wrt the output `m_vxx`.
            m_sigmayyy: Gradient of the loss wrt the output `m_sigmayyy`.
            m_sigmaxyy: Gradient of the loss wrt the output `m_sigmaxyy`.
            m_sigmaxyx: Gradient of the loss wrt the output `m_sigmaxyx`.
            m_sigmaxxx: Gradient of the loss wrt the output `m_sigmaxxx`.
            grad_r_p: Gradient of the loss wrt the output `receiver_amplitudes_p`.
            grad_r_y: Gradient of the loss wrt the output `receiver_amplitudes_y`.
            grad_r_x: Gradient of the loss wrt the output `receiver_amplitudes_x`.

        Returns:
            A tuple containing the gradients with respect to the inputs of the
            forward pass.

        """
        (
            lamb,
            mu,
            buoyancy,
            ay,
            ayh,
            ax,
            axh,
            by,
            byh,
            bx,
            bxh,
            sources_y_i,
            sources_x_i,
            receivers_y_i,
            receivers_x_i,
            receivers_p_i,
            dvydbuoyancy,
            dvxdbuoyancy,
            dvydy_store,
            dvxdx_store,
            dvydxdvxdy_store,
        ) = ctx.saved_tensors

        lamb = lamb.contiguous()
        mu = mu.contiguous()
        buoyancy = buoyancy.contiguous()
        grad_r_p = grad_r_p.contiguous()
        grad_r_y = grad_r_y.contiguous()
        grad_r_x = grad_r_x.contiguous()
        ay = ay.contiguous()
        ayh = ayh.contiguous()
        ax = ax.contiguous()
        axh = axh.contiguous()
        by = by.contiguous()
        byh = byh.contiguous()
        bx = bx.contiguous()
        bxh = bxh.contiguous()
        sources_y_i = sources_y_i.contiguous()
        sources_x_i = sources_x_i.contiguous()
        receivers_y_i = receivers_y_i.contiguous()
        receivers_x_i = receivers_x_i.contiguous()
        receivers_p_i = receivers_p_i.contiguous()

        dy = ctx.dy
        dx = ctx.dx
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        pml_width = ctx.pml_width
        source_amplitudes_y_requires_grad = ctx.source_amplitudes_y_requires_grad
        source_amplitudes_x_requires_grad = ctx.source_amplitudes_x_requires_grad
        backward_callback = ctx.backward_callback
        callback_frequency = ctx.callback_frequency
        device = lamb.device
        dtype = lamb.dtype
        ny = lamb.shape[-2]
        nx = lamb.shape[-1]
        n_sources_y_per_shot = sources_y_i.numel() // n_shots
        n_sources_x_per_shot = sources_x_i.numel() // n_shots
        n_receivers_y_per_shot = receivers_y_i.numel() // n_shots
        n_receivers_x_per_shot = receivers_x_i.numel() // n_shots
        n_receivers_p_per_shot = receivers_p_i.numel() // n_shots
        grad_lamb = torch.empty(0, device=device, dtype=dtype)
        grad_lamb_tmp = torch.empty(0, device=device, dtype=dtype)
        grad_lamb_tmp_ptr = grad_lamb.data_ptr()
        if lamb.requires_grad:
            grad_lamb.resize_(*lamb.shape)
            grad_lamb.fill_(0)
            grad_lamb_tmp_ptr = grad_lamb.data_ptr()
        grad_mu = torch.empty(0, device=device, dtype=dtype)
        grad_mu_tmp = torch.empty(0, device=device, dtype=dtype)
        grad_mu_tmp_ptr = grad_mu.data_ptr()
        if mu.requires_grad:
            grad_mu.resize_(*mu.shape)
            grad_mu.fill_(0)
            grad_mu_tmp_ptr = grad_mu.data_ptr()
        grad_buoyancy = torch.empty(0, device=device, dtype=dtype)
        grad_buoyancy_tmp = torch.empty(0, device=device, dtype=dtype)
        grad_buoyancy_tmp_ptr = grad_buoyancy.data_ptr()
        if buoyancy.requires_grad:
            grad_buoyancy.resize_(*buoyancy.shape)
            grad_buoyancy.fill_(0)
            grad_buoyancy_tmp_ptr = grad_buoyancy.data_ptr()
        grad_f_y = torch.empty(0, device=device, dtype=dtype)
        grad_f_x = torch.empty(0, device=device, dtype=dtype)

        pml_y0 = max(pml_width[0], accuracy // 2)
        pml_y1 = min(ny - pml_width[1], ny - accuracy // 2)
        pml_x0 = max(pml_width[2], accuracy // 2)
        pml_x1 = min(nx - pml_width[3], nx - accuracy // 2)
        spml_y0 = max(pml_width[0], accuracy + 1)
        spml_y1 = min(ny - pml_width[1], ny - (accuracy + 1))
        spml_x0 = max(pml_width[2], accuracy + 1)
        spml_x1 = min(nx - pml_width[3], nx - (accuracy + 1))
        vpml_y0 = max(pml_width[0], accuracy)
        vpml_y1 = min(ny - pml_width[1], ny - accuracy)
        vpml_x0 = max(pml_width[2], accuracy)
        vpml_x1 = min(nx - pml_width[3], nx - accuracy)

        lamb_batched = lamb.ndim == 3 and lamb.shape[0] > 1
        mu_batched = mu.ndim == 3 and mu.shape[0] > 1
        buoyancy_batched = buoyancy.ndim == 3 and buoyancy.shape[0] > 1

        size_with_batch = (n_shots, *lamb.shape[-2:])
        vy = deepwave.common.create_or_pad(
            vy,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        vx = deepwave.common.create_or_pad(
            vx,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        sigmayy = deepwave.common.create_or_pad(
            sigmayy,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        sigmaxy = deepwave.common.create_or_pad(
            sigmaxy,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        sigmaxx = deepwave.common.create_or_pad(
            sigmaxx,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vyy = deepwave.common.create_or_pad(
            m_vyy,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vyx = deepwave.common.create_or_pad(
            m_vyx,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vxy = deepwave.common.create_or_pad(
            m_vxy,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vxx = deepwave.common.create_or_pad(
            m_vxx,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmayyy = deepwave.common.create_or_pad(
            m_sigmayyy,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmaxyy = deepwave.common.create_or_pad(
            m_sigmaxyy,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmaxyx = deepwave.common.create_or_pad(
            m_sigmaxyx,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmaxxx = deepwave.common.create_or_pad(
            m_sigmaxxx,
            0,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmayyyn = torch.zeros_like(m_sigmayyy)
        m_sigmaxyyn = torch.zeros_like(m_sigmaxyy)
        m_sigmaxyxn = torch.zeros_like(m_sigmaxyx)
        m_sigmaxxxn = torch.zeros_like(m_sigmaxxx)
        zero_edges(sigmaxy, ny, nx)
        zero_edges(m_vxy, ny, nx)
        zero_edges(m_vyx, ny, nx)
        zero_edge_right(vy, nx)
        zero_edge_right(m_sigmayyy, nx)
        zero_edge_right(m_sigmaxyx, nx)
        zero_edge_top(vx)
        zero_edge_top(m_sigmaxyy)
        zero_edge_top(m_sigmaxxx)
        zero_edge_top(sigmaxx)
        zero_edge_top(m_vxx)
        zero_edge_top(sigmayy)
        zero_edge_top(m_vyy)
        zero_edge_right(sigmaxx, nx)
        zero_edge_right(m_vxx, nx)
        zero_edge_right(sigmayy, nx)
        zero_edge_right(m_vyy, nx)
        m_sigmayyy = zero_interior(m_sigmayyy, pml_y0, pml_y1, 0, nx)
        m_sigmaxyx = zero_interior(m_sigmaxyx, 0, ny, pml_x0, pml_x1 - 1)
        m_sigmaxyy = zero_interior(m_sigmaxyy, pml_y0 + 1, pml_y1, 0, nx)
        m_sigmaxxx = zero_interior(m_sigmaxxx, 0, ny, pml_x0, pml_x1)
        m_vyy = zero_interior(m_vyy, pml_y0 + 1, pml_y1, 0, nx)
        m_vxx = zero_interior(m_vxx, 0, ny, pml_x0, pml_x1 - 1)
        m_vyx = zero_interior(m_vyx, 0, ny, pml_x0 + 1, pml_x1 - 1)
        m_vxy = zero_interior(m_vxy, pml_y0 + 1, pml_y1 - 1, 0, nx)

        if source_amplitudes_y_requires_grad:
            grad_f_y.resize_(nt, n_shots, n_sources_y_per_shot)
            grad_f_y.fill_(0)
        if source_amplitudes_x_requires_grad:
            grad_f_x.resize_(nt, n_shots, n_sources_x_per_shot)
            grad_f_x.fill_(0)

        if lamb.is_cuda:
            aux = lamb.get_device()
            if lamb.requires_grad and not lamb_batched and n_shots > 1:
                grad_lamb_tmp.resize_(n_shots, *lamb.shape[-2:])
                grad_lamb_tmp.fill_(0)
                grad_lamb_tmp_ptr = grad_lamb_tmp.data_ptr()
            if mu.requires_grad and not mu_batched and n_shots > 1:
                grad_mu_tmp.resize_(n_shots, *mu.shape[-2:])
                grad_mu_tmp.fill_(0)
                grad_mu_tmp_ptr = grad_mu_tmp.data_ptr()
            if buoyancy.requires_grad and not buoyancy_batched and n_shots > 1:
                grad_buoyancy_tmp.resize_(n_shots, *buoyancy.shape[-2:])
                grad_buoyancy_tmp.fill_(0)
                grad_buoyancy_tmp_ptr = grad_buoyancy_tmp.data_ptr()
        else:
            if deepwave.backend_utils.USE_OPENMP:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if (
                lamb.requires_grad
                and not lamb_batched
                and aux > 1
                and deepwave.backend_utils.USE_OPENMP
            ):
                grad_lamb_tmp.resize_(aux, *lamb.shape[-2:])
                grad_lamb_tmp.fill_(0)
                grad_lamb_tmp_ptr = grad_lamb_tmp.data_ptr()
            if (
                mu.requires_grad
                and not mu_batched
                and aux > 1
                and deepwave.backend_utils.USE_OPENMP
            ):
                grad_mu_tmp.resize_(n_shots, *mu.shape[-2:])
                grad_mu_tmp.fill_(0)
                grad_mu_tmp_ptr = grad_mu_tmp.data_ptr()
            if (
                buoyancy.requires_grad
                and not buoyancy_batched
                and aux > 1
                and deepwave.backend_utils.USE_OPENMP
            ):
                grad_buoyancy_tmp.resize_(n_shots, *buoyancy.shape[-2:])
                grad_buoyancy_tmp.fill_(0)
                grad_buoyancy_tmp_ptr = grad_buoyancy_tmp.data_ptr()
        backward = deepwave.backend_utils.get_backend_function(
            "elastic",
            "backward",
            accuracy,
            dtype,
            lamb.device,
        )

        if backward_callback is None:
            callback_frequency = nt // step_ratio

        if vy.numel() > 0 and nt > 0:
            for step in range(nt // step_ratio, 0, -callback_frequency):
                step_nt = min(step, callback_frequency)
                backward(
                    lamb.data_ptr(),
                    mu.data_ptr(),
                    buoyancy.data_ptr(),
                    grad_r_y.data_ptr(),
                    grad_r_x.data_ptr(),
                    grad_r_p.data_ptr(),
                    vy.data_ptr(),
                    vx.data_ptr(),
                    sigmayy.data_ptr(),
                    sigmaxy.data_ptr(),
                    sigmaxx.data_ptr(),
                    m_vyy.data_ptr(),
                    m_vyx.data_ptr(),
                    m_vxy.data_ptr(),
                    m_vxx.data_ptr(),
                    m_sigmayyy.data_ptr(),
                    m_sigmaxyy.data_ptr(),
                    m_sigmaxyx.data_ptr(),
                    m_sigmaxxx.data_ptr(),
                    m_sigmayyyn.data_ptr(),
                    m_sigmaxyyn.data_ptr(),
                    m_sigmaxyxn.data_ptr(),
                    m_sigmaxxxn.data_ptr(),
                    dvydbuoyancy.data_ptr(),
                    dvxdbuoyancy.data_ptr(),
                    dvydy_store.data_ptr(),
                    dvxdx_store.data_ptr(),
                    dvydxdvxdy_store.data_ptr(),
                    grad_f_y.data_ptr(),
                    grad_f_x.data_ptr(),
                    grad_lamb.data_ptr(),
                    grad_lamb_tmp_ptr,
                    grad_mu.data_ptr(),
                    grad_mu_tmp_ptr,
                    grad_buoyancy.data_ptr(),
                    grad_buoyancy_tmp_ptr,
                    ay.data_ptr(),
                    ayh.data_ptr(),
                    ax.data_ptr(),
                    axh.data_ptr(),
                    by.data_ptr(),
                    byh.data_ptr(),
                    bx.data_ptr(),
                    bxh.data_ptr(),
                    sources_y_i.data_ptr(),
                    sources_x_i.data_ptr(),
                    receivers_y_i.data_ptr(),
                    receivers_x_i.data_ptr(),
                    receivers_p_i.data_ptr(),
                    dy,
                    dx,
                    dt,
                    step_nt * step_ratio,
                    n_shots,
                    ny,
                    nx,
                    n_sources_y_per_shot * source_amplitudes_y_requires_grad,
                    n_sources_x_per_shot * source_amplitudes_x_requires_grad,
                    n_receivers_y_per_shot,
                    n_receivers_x_per_shot,
                    n_receivers_p_per_shot,
                    step_ratio,
                    lamb.requires_grad,
                    mu.requires_grad,
                    buoyancy.requires_grad,
                    lamb_batched,
                    mu_batched,
                    buoyancy_batched,
                    step * step_ratio,
                    spml_y0,
                    spml_y1,
                    spml_x0,
                    spml_x1,
                    vpml_y0,
                    vpml_y1,
                    vpml_x0,
                    vpml_x1,
                    aux,
                )
                if (step_nt * step_ratio) % 2 != 0:
                    m_sigmayyy, m_sigmaxyx, m_sigmaxyy, m_sigmaxxx = (
                        m_sigmayyyn,
                        m_sigmaxyxn,
                        m_sigmaxyyn,
                        m_sigmaxxxn,
                    )
                if backward_callback is not None:
                    state = deepwave.common.CallbackState(
                        dt,
                        step - 1,
                        {
                            "vy_0": vy,
                            "vx_0": vx,
                            "sigmayy_0": sigmayy,
                            "sigmaxy_0": sigmaxy,
                            "sigmaxx_0": sigmaxx,
                            "m_vyy_0": m_vyy,
                            "m_vyx_0": m_vyx,
                            "m_vxy_0": m_vxy,
                            "m_vxx_0": m_vxx,
                            "m_sigmayyy_0": m_sigmayyy,
                            "m_sigmaxyy_0": m_sigmaxyy,
                            "m_sigmaxyx_0": m_sigmaxyx,
                            "m_sigmaxxx_0": m_sigmaxxx,
                        },
                        {"lamb": lamb, "mu": mu, "buoyancy": buoyancy},
                        {"lamb": grad_lamb, "mu": grad_mu, "buoyancy": grad_buoyancy},
                        [0] * 4,
                        pml_width,
                    )
                    backward_callback(state)

        m_vyy = zero_interior(m_vyy, pml_y0 + 1, pml_y1, 0, nx)
        m_vxx = zero_interior(m_vxx, 0, ny, pml_x0, pml_x1 - 1)
        m_vyx = zero_interior(m_vyx, 0, ny, pml_x0 + 1, pml_x1 - 1)
        m_vxy = zero_interior(m_vxy, pml_y0 + 1, pml_y1 - 1, 0, nx)

        m_sigmayyy = zero_interior(m_sigmayyy, pml_y0, pml_y1, 0, nx)
        m_sigmaxyx = zero_interior(m_sigmaxyx, 0, ny, pml_x0, pml_x1 - 1)
        m_sigmaxyy = zero_interior(m_sigmaxyy, pml_y0 + 1, pml_y1, 0, nx)
        m_sigmaxxx = zero_interior(m_sigmaxxx, 0, ny, pml_x0, pml_x1)
        return (
            grad_lamb,
            grad_mu,
            grad_buoyancy,
            grad_f_y,
            grad_f_x,
            vy,
            vx,
            sigmayy,
            sigmaxy,
            sigmaxx,
            m_vyy,
            m_vyx,
            m_vxy,
            m_vxx,
            m_sigmayyy,
            m_sigmaxyy,
            m_sigmaxyx,
            m_sigmaxxx,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def elastic_func(*args: Any) -> Tuple[torch.Tensor, ...]:
    """A helper function to apply the ElasticForwardFunc.

    This function serves as a convenient wrapper to call the `apply` method
    of `ElasticForwardFunc`, which is the entry point for the autograd graph
    for elastic wave propagation.

    Args:
        *args: Variable length argument list to be passed directly to
            `ElasticForwardFunc.apply`.

    Returns:
        The results of the forward pass from `ElasticForwardFunc.apply`.

    """
    return cast(
        "Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "
        "torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "
        "torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "
        "torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]",
        ElasticForwardFunc.apply(*args),  # type: ignore[no-untyped-call]
    )
