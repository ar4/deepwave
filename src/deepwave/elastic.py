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


def prepare_parameters(
    mu: torch.Tensor, buoyancy: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepares elastic properties for the free surface method.

    This function applies the Zeng et al. (2012) logic for all cell types.

    Args:
        mu (torch.Tensor): Lame parameter mu.
        buoyancy (torch.Tensor): Buoyancy (1/rho).

    Returns:
        tuple: (mu_yx, buoyancy_y, buoyancy_x)
    """
    rfmax = 1 / torch.finfo(mu.dtype).max ** (1 / 2)
    # Buoyancy (inverse of arithmetic mean of density)
    # Arithmetic mean: rho[i+1/2] = (rho[i] + rho[i+1])/2
    # => buoyancy[i+1/2] = 2/(rho[i] + rho[i+1])
    rho = torch.zeros_like(buoyancy)
    mask = rfmax < buoyancy.abs()
    rho[mask] = 1 / buoyancy[mask]
    rho_y = torch.nn.functional.pad(
        (rho[..., :-1, :] + rho[..., 1:, :]) / 2, (0, 0, 0, 1)
    )
    rho_x = torch.nn.functional.pad((rho[..., :-1] + rho[..., 1:]) / 2, (0, 1))
    mask = rfmax < rho_y.abs()
    buoyancy_y = torch.zeros_like(buoyancy)
    buoyancy_y[mask] = 1 / rho_y[mask]
    mask = rfmax < rho_x.abs()
    buoyancy_x = torch.zeros_like(buoyancy)
    buoyancy_x[mask] = 1 / rho_x[mask]

    # Mu (harmonic mean)
    mask = (
        (rfmax < mu[..., 1:, 1:].abs())
        .logical_and(rfmax < mu[..., :-1, :-1].abs())
        .logical_and(rfmax < mu[..., 1:, :-1].abs())
        .logical_and(rfmax < mu[..., :-1, 1:].abs())
    )
    mu_yx = torch.zeros_like(mu[..., :-1, :-1])
    mu_yx[mask] = 4 / (
        1 / mu[..., 1:, 1:][mask]
        + 1 / mu[..., :-1, :-1][mask]
        + 1 / mu[..., 1:, :-1][mask]
        + 1 / mu[..., :-1, 1:][mask]
    )
    mu_yx = torch.nn.functional.pad(mu_yx, (0, 1, 0, 1))

    return mu_yx, buoyancy_y, buoyancy_x


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
        python_backend: Union[bool, str] = False,
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
            python_backend=python_backend,
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
    python_backend: Union[bool, str] = False,
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
        python_backend: Use Python backend rather than compiled C/CUDA.
            Can be a string specifying whether to use PyTorch's JIT ("jit"),
            torch.compile ("compile"), or eager mode ("eager"). Alternatively
            a boolean can be provided, with True using the Python backend
            with torch.compile, while the default, False, instead uses the
            compiled C/CUDA.

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
        and source_locations_y[..., 0].max() >= lamb.shape[-2] - 1
    ):
        raise RuntimeError(
            "With the provided model, the maximum y source "
            f"location in the first dimension must be less than {lamb.shape[-2] - 1}.",
        )
    if (
        receiver_locations_y is not None
        and receiver_locations_y[..., 0].max() >= lamb.shape[-2] - 1
    ):
        raise RuntimeError(
            "With the provided model, the maximum y "
            "receiver location in the first dimension "
            f"must be less than {lamb.shape[-2] - 1}.",
        )
    if (
        source_locations_x is not None
        and source_locations_x[..., 1].max() >= lamb.shape[-1] - 1
    ):
        raise RuntimeError(
            "With the provided model, the maximum x "
            "source location in the second dimension "
            f"must be less than {lamb.shape[-1] - 1}.",
        )
    if (
        receiver_locations_x is not None
        and receiver_locations_x[..., 1].max() >= lamb.shape[-1] - 1
    ):
        raise RuntimeError(
            "With the provided model, the maximum x "
            "receiver location in the second dimension "
            f"must be less than {lamb.shape[-1] - 1}.",
        )
    if receiver_locations_p is not None and (
        receiver_locations_p[..., 0].max() >= lamb.shape[-2] - 1
        or receiver_locations_p[..., 1].max() >= lamb.shape[-1] - 1
    ):
        raise RuntimeError(
            "With the provided model, the maximum p "
            "receiver location in the first dimension "
            f"must be less than {lamb.shape[-2] - 1} and in "
            "the second dimension must be less than "
            f"{lamb.shape[-1] - 1}.",
        )
    vp, vs, _ = deepwave.common.lambmubuoyancy_to_vpvsrho(
        lamb,
        mu,
        buoyancy,
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
    fd_pad = [accuracy // 2, accuracy // 2 - 1, accuracy // 2, accuracy // 2 - 1]

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

    models += prepare_parameters(models[1], models[2])
    del models[2]  # Remove buoyancy as it is no longer needed

    ny, nx = models[0].shape[-2:]
    # source_amplitudes_y
    mask = sources_i[0] == deepwave.common.IGNORE_LOCATION
    sources_i_masked = sources_i[0].clone()
    sources_i_masked[mask] = 0
    if source_amplitudes[0].numel() > 0:
        source_amplitudes[0] = (
            source_amplitudes[0]
            * (
                models[3]
                .view(-1, ny * nx)
                .expand(n_shots, -1)
                .gather(1, sources_i_masked)
            )
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
                models[4]
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
        python_backend,
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


def zero_bottom(tensor: torch.Tensor, fd_pad: int) -> None:
    """Sets values on the edge at the end of the y dimension of a 2D tensor to zero.

    Args:
        tensor: The input 2D torch.Tensor to modify, with shape (batch, ny, nx).
        fd_pad: Half the length of the spatial finite difference stencil.
    """
    tensor[..., -fd_pad, :] = 0


def zero_right(tensor: torch.Tensor, fd_pad: int) -> None:
    """Sets values on the edge at the end of the x dimension of a 2D tensor to zero.

    Args:
        tensor: The input 2D torch.Tensor to modify, with shape (batch, ny, nx).
        fd_pad: Half the length of the spatial finite difference stencil.
    """
    tensor[..., -fd_pad] = 0


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
        mu_yx: torch.Tensor,
        buoyancy_y: torch.Tensor,
        buoyancy_x: torch.Tensor,
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
            mu_yx: The second Lam'e parameter (mu) model tensor at y+1/2, x+1/2.
            buoyancy_y: The buoyancy (1/density) model tensor at y+1/2.
            buoyancy_x: The buoyancy (1/density) model tensor at x+1/2.
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
        mu_yx = mu_yx.contiguous()
        buoyancy_y = buoyancy_y.contiguous()
        buoyancy_x = buoyancy_x.contiguous()
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

        fd_pad = accuracy // 2
        fd_pad_list = [fd_pad, fd_pad - 1, fd_pad, fd_pad - 1]
        size_with_batch = (n_shots, *lamb.shape[-2:])
        vy = deepwave.common.create_or_pad(
            vy,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        vx = deepwave.common.create_or_pad(
            vx,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        sigmayy = deepwave.common.create_or_pad(
            sigmayy,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        sigmaxy = deepwave.common.create_or_pad(
            sigmaxy,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        sigmaxx = deepwave.common.create_or_pad(
            sigmaxx,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vyy = deepwave.common.create_or_pad(
            m_vyy,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vyx = deepwave.common.create_or_pad(
            m_vyx,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vxy = deepwave.common.create_or_pad(
            m_vxy,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vxx = deepwave.common.create_or_pad(
            m_vxx,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmayyy = deepwave.common.create_or_pad(
            m_sigmayyy,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmaxyy = deepwave.common.create_or_pad(
            m_sigmaxyy,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmaxyx = deepwave.common.create_or_pad(
            m_sigmaxyx,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmaxxx = deepwave.common.create_or_pad(
            m_sigmaxxx,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        zero_bottom(sigmaxy, fd_pad)
        zero_right(sigmaxy, fd_pad)
        zero_bottom(m_vxy, fd_pad)
        zero_right(m_vxy, fd_pad)
        zero_bottom(m_vyx, fd_pad)
        zero_right(m_vyx, fd_pad)
        zero_bottom(vy, fd_pad)
        zero_bottom(m_sigmayyy, fd_pad)
        zero_bottom(m_sigmaxyx, fd_pad)
        zero_right(vx, fd_pad)
        zero_right(m_sigmaxyy, fd_pad)
        zero_right(m_sigmaxxx, fd_pad)

        pml_y0 = pml_width[0] + fd_pad
        pml_y1 = max(pml_y0, ny - (fd_pad - 1) - pml_width[1])
        pml_x0 = pml_width[2] + fd_pad
        pml_x1 = max(pml_x0, nx - (fd_pad - 1) - pml_width[3])

        m_sigmayyy = zero_interior(m_sigmayyy, pml_y0, pml_y1 - 1, 0, nx)  # yhx
        m_sigmaxyx = zero_interior(m_sigmaxyx, 0, ny, pml_x0, pml_x1)  # yhx
        m_sigmaxyy = zero_interior(m_sigmaxyy, pml_y0, pml_y1, 0, nx)  # yxh
        m_sigmaxxx = zero_interior(m_sigmaxxx, 0, ny, pml_x0, pml_x1 - 1)  # yxh
        m_vyy = zero_interior(m_vyy, pml_y0, pml_y1, 0, nx)  # yx
        m_vxx = zero_interior(m_vxx, 0, ny, pml_x0, pml_x1)  # yx
        m_vyx = zero_interior(m_vyx, 0, ny, pml_x0, pml_x1 - 1)  # yhxh
        m_vxy = zero_interior(m_vxy, pml_y0, pml_y1 - 1, 0, nx)  # yhxh

        lamb_batched = lamb.ndim == 3 and lamb.shape[0] > 1
        mu_batched = mu.ndim == 3 and mu.shape[0] > 1
        buoyancy_batched = buoyancy_y.ndim == 3 and buoyancy_y.shape[0] > 1

        if lamb.requires_grad or mu.requires_grad:
            dvydy_store.resize_(nt // step_ratio, n_shots, ny, nx)
            dvxdx_store.resize_(nt // step_ratio, n_shots, ny, nx)
        if mu.requires_grad:
            dvydxdvxdy_store.resize_(nt // step_ratio, n_shots, ny, nx)
        if buoyancy_y.requires_grad:
            dvydbuoyancy.resize_(nt // step_ratio, n_shots, ny, nx)
            dvxdbuoyancy.resize_(nt // step_ratio, n_shots, ny, nx)

        if receivers_y_i.numel() > 0:
            receiver_amplitudes_y.resize_(nt + 1, n_shots, n_receivers_y_per_shot)
            receiver_amplitudes_y.fill_(0)
        if receivers_x_i.numel() > 0:
            receiver_amplitudes_x.resize_(nt + 1, n_shots, n_receivers_x_per_shot)
            receiver_amplitudes_x.fill_(0)
        if receivers_p_i.numel() > 0:
            receiver_amplitudes_p.resize_(nt, n_shots, n_receivers_p_per_shot)
            receiver_amplitudes_p.fill_(0)

        if (
            lamb.requires_grad
            or mu.requires_grad
            or buoyancy_y.requires_grad
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
                mu_yx,
                buoyancy_y,
                buoyancy_x,
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
                        {
                            "lamb": lamb,
                            "mu": mu,
                            "mu_yx": mu_yx,
                            "buoyancy_y": buoyancy_y,
                            "buoyancy_x": buoyancy_x,
                        },
                        {},
                        fd_pad_list,
                        pml_width,
                    )
                    forward_callback(state)
                step_nt = min(nt // step_ratio - step, callback_frequency)
                forward(
                    lamb.data_ptr(),
                    mu.data_ptr(),
                    mu_yx.data_ptr(),
                    buoyancy_y.data_ptr(),
                    buoyancy_x.data_ptr(),
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
                    1 / dy,
                    1 / dx,
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
                    buoyancy_y.requires_grad,
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

        s = (
            slice(None),
            slice(fd_pad, ny - (fd_pad - 1)),
            slice(fd_pad, nx - (fd_pad - 1)),
        )
        return (
            vy[s],
            vx[s],
            sigmayy[s],
            sigmaxy[s],
            sigmaxx[s],
            m_vyy[s],
            m_vyx[s],
            m_vxy[s],
            m_vxx[s],
            m_sigmayyy[s],
            m_sigmaxyy[s],
            m_sigmaxyx[s],
            m_sigmaxxx[s],
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
            mu_yx,
            buoyancy_y,
            buoyancy_x,
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
        mu_yx = mu_yx.contiguous()
        buoyancy_y = buoyancy_y.contiguous()
        buoyancy_x = buoyancy_x.contiguous()
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
        grad_mu_yx = torch.empty(0, device=device, dtype=dtype)
        grad_mu_yx_tmp = torch.empty(0, device=device, dtype=dtype)
        grad_mu_yx_tmp_ptr = grad_mu_yx.data_ptr()
        if mu.requires_grad:
            grad_mu.resize_(*mu.shape)
            grad_mu.fill_(0)
            grad_mu_tmp_ptr = grad_mu.data_ptr()
        if mu_yx.requires_grad:
            grad_mu_yx.resize_(*mu_yx.shape)
            grad_mu_yx.fill_(0)
            grad_mu_yx_tmp_ptr = grad_mu_yx.data_ptr()
        grad_buoyancy_y = torch.empty(0, device=device, dtype=dtype)
        grad_buoyancy_y_tmp = torch.empty(0, device=device, dtype=dtype)
        grad_buoyancy_y_tmp_ptr = grad_buoyancy_y.data_ptr()
        grad_buoyancy_x = torch.empty(0, device=device, dtype=dtype)
        grad_buoyancy_x_tmp = torch.empty(0, device=device, dtype=dtype)
        grad_buoyancy_x_tmp_ptr = grad_buoyancy_x.data_ptr()
        if buoyancy_y.requires_grad:
            grad_buoyancy_y.resize_(*buoyancy_y.shape)
            grad_buoyancy_y.fill_(0)
            grad_buoyancy_y_tmp_ptr = grad_buoyancy_y.data_ptr()
        if buoyancy_x.requires_grad:
            grad_buoyancy_x.resize_(*buoyancy_x.shape)
            grad_buoyancy_x.fill_(0)
            grad_buoyancy_x_tmp_ptr = grad_buoyancy_x.data_ptr()
        grad_f_y = torch.empty(0, device=device, dtype=dtype)
        grad_f_x = torch.empty(0, device=device, dtype=dtype)

        lamb_batched = lamb.ndim == 3 and lamb.shape[0] > 1
        mu_batched = mu.ndim == 3 and mu.shape[0] > 1
        buoyancy_batched = buoyancy_y.ndim == 3 and buoyancy_y.shape[0] > 1

        fd_pad = accuracy // 2
        fd_pad_list = [fd_pad, fd_pad - 1, fd_pad, fd_pad - 1]
        size_with_batch = (n_shots, *lamb.shape[-2:])
        vy = deepwave.common.create_or_pad(
            vy,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        vx = deepwave.common.create_or_pad(
            vx,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        sigmayy = deepwave.common.create_or_pad(
            sigmayy,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        sigmaxy = deepwave.common.create_or_pad(
            sigmaxy,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        sigmaxx = deepwave.common.create_or_pad(
            sigmaxx,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vyy = deepwave.common.create_or_pad(
            m_vyy,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vyx = deepwave.common.create_or_pad(
            m_vyx,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vxy = deepwave.common.create_or_pad(
            m_vxy,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_vxx = deepwave.common.create_or_pad(
            m_vxx,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmayyy = deepwave.common.create_or_pad(
            m_sigmayyy,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmaxyy = deepwave.common.create_or_pad(
            m_sigmaxyy,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmaxyx = deepwave.common.create_or_pad(
            m_sigmaxyx,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmaxxx = deepwave.common.create_or_pad(
            m_sigmaxxx,
            fd_pad_list,
            lamb.device,
            lamb.dtype,
            size_with_batch,
        )
        m_sigmayyyn = torch.zeros_like(m_sigmayyy)
        m_sigmaxyyn = torch.zeros_like(m_sigmaxyy)
        m_sigmaxyxn = torch.zeros_like(m_sigmaxyx)
        m_sigmaxxxn = torch.zeros_like(m_sigmaxxx)
        zero_bottom(sigmaxy, fd_pad)
        zero_right(sigmaxy, fd_pad)
        zero_bottom(m_vxy, fd_pad)
        zero_right(m_vxy, fd_pad)
        zero_bottom(m_vyx, fd_pad)
        zero_right(m_vyx, fd_pad)
        zero_bottom(m_vyx, fd_pad)
        zero_right(m_vyx, fd_pad)
        zero_bottom(vy, fd_pad)
        zero_bottom(m_sigmayyy, fd_pad)
        zero_bottom(m_sigmaxyx, fd_pad)
        zero_right(vx, fd_pad)
        zero_right(m_sigmaxyy, fd_pad)
        zero_right(m_sigmaxxx, fd_pad)

        pml_y0 = min(pml_width[0] + 2 * fd_pad, ny - (fd_pad - 1))
        pml_y1 = max(pml_y0, ny - 2 * fd_pad + 1 - pml_width[1])
        pml_x0 = min(pml_width[2] + 2 * fd_pad, nx - (fd_pad - 1))
        pml_x1 = max(pml_x0, nx - 2 * fd_pad + 1 - pml_width[3])

        m_sigmayyy = zero_interior(m_sigmayyy, pml_y0, pml_y1 - 1, 0, nx)  # yhx
        m_sigmaxyx = zero_interior(m_sigmaxyx, 0, ny, pml_x0, pml_x1)  # yhx
        m_sigmaxyy = zero_interior(m_sigmaxyy, pml_y0, pml_y1, 0, nx)  # yxh
        m_sigmaxxx = zero_interior(m_sigmaxxx, 0, ny, pml_x0, pml_x1 - 1)  # yxh
        m_vyy = zero_interior(m_vyy, pml_y0, pml_y1, 0, nx)  # yx
        m_vxx = zero_interior(m_vxx, 0, ny, pml_x0, pml_x1)  # yx
        m_vyx = zero_interior(m_vyx, 0, ny, pml_x0, pml_x1 - 1)  # yhxh
        m_vxy = zero_interior(m_vxy, pml_y0, pml_y1 - 1, 0, nx)  # yhxh

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
            if mu_yx.requires_grad and not mu_batched and n_shots > 1:
                grad_mu_yx_tmp.resize_(n_shots, *mu_yx.shape[-2:])
                grad_mu_yx_tmp.fill_(0)
                grad_mu_yx_tmp_ptr = grad_mu_yx_tmp.data_ptr()
            if buoyancy_y.requires_grad and not buoyancy_batched and n_shots > 1:
                grad_buoyancy_y_tmp.resize_(n_shots, *buoyancy_y.shape[-2:])
                grad_buoyancy_y_tmp.fill_(0)
                grad_buoyancy_y_tmp_ptr = grad_buoyancy_y_tmp.data_ptr()
            if buoyancy_x.requires_grad and not buoyancy_batched and n_shots > 1:
                grad_buoyancy_x_tmp.resize_(n_shots, *buoyancy_x.shape[-2:])
                grad_buoyancy_x_tmp.fill_(0)
                grad_buoyancy_x_tmp_ptr = grad_buoyancy_x_tmp.data_ptr()
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
                mu_yx.requires_grad
                and not mu_batched
                and aux > 1
                and deepwave.backend_utils.USE_OPENMP
            ):
                grad_mu_yx_tmp.resize_(n_shots, *mu_yx.shape[-2:])
                grad_mu_yx_tmp.fill_(0)
                grad_mu_yx_tmp_ptr = grad_mu_yx_tmp.data_ptr()
            if (
                buoyancy_y.requires_grad
                and not buoyancy_batched
                and aux > 1
                and deepwave.backend_utils.USE_OPENMP
            ):
                grad_buoyancy_y_tmp.resize_(n_shots, *buoyancy_y.shape[-2:])
                grad_buoyancy_y_tmp.fill_(0)
                grad_buoyancy_y_tmp_ptr = grad_buoyancy_y_tmp.data_ptr()
            if (
                buoyancy_x.requires_grad
                and not buoyancy_batched
                and aux > 1
                and deepwave.backend_utils.USE_OPENMP
            ):
                grad_buoyancy_x_tmp.resize_(n_shots, *buoyancy_x.shape[-2:])
                grad_buoyancy_x_tmp.fill_(0)
                grad_buoyancy_x_tmp_ptr = grad_buoyancy_x_tmp.data_ptr()
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
                    mu_yx.data_ptr(),
                    buoyancy_y.data_ptr(),
                    buoyancy_x.data_ptr(),
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
                    grad_mu_yx.data_ptr(),
                    grad_mu_yx_tmp_ptr,
                    grad_buoyancy_y.data_ptr(),
                    grad_buoyancy_y_tmp_ptr,
                    grad_buoyancy_x.data_ptr(),
                    grad_buoyancy_x_tmp_ptr,
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
                    1 / dy,
                    1 / dx,
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
                    buoyancy_y.requires_grad,
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
                        {
                            "lamb": lamb,
                            "mu": mu,
                            "mu_yx": mu_yx,
                            "buoyancy_y": buoyancy_y,
                            "buoyancy_x": buoyancy_x,
                        },
                        {
                            "lamb": grad_lamb,
                            "mu": grad_mu,
                            "mu_yx": grad_mu_yx,
                            "buoyancy_y": grad_buoyancy_y,
                            "buoyancy_x": grad_buoyancy_x,
                        },
                        fd_pad_list,
                        pml_width,
                    )
                    backward_callback(state)

        s = (
            slice(None),
            slice(fd_pad, ny - (fd_pad - 1)),
            slice(fd_pad, nx - (fd_pad - 1)),
        )
        return (
            grad_lamb,
            grad_mu,
            grad_mu_yx,
            grad_buoyancy_y,
            grad_buoyancy_x,
            grad_f_y,
            grad_f_x,
            vy[s],
            vx[s],
            sigmayy[s],
            sigmaxy[s],
            sigmaxx[s],
            m_vyy[s],
            m_vyx[s],
            m_vxy[s],
            m_vxx[s],
            m_sigmayyy[s],
            m_sigmaxyy[s],
            m_sigmaxyx[s],
            m_sigmaxxx[s],
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


def update_velocities(
    buoyancy_y: torch.Tensor,
    buoyancy_x: torch.Tensor,
    vy: torch.Tensor,
    vx: torch.Tensor,
    sigmayy: torch.Tensor,
    sigmaxy: torch.Tensor,
    sigmaxx: torch.Tensor,
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
    rdy: torch.Tensor,
    rdx: torch.Tensor,
    dt: torch.Tensor,
    accuracy: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Updates the velocity wavefields and PML memory variables.

    Args:
        buoyancy_y: Buoyancy in y-direction.
        buoyancy_x: Buoyancy in x-direction.
        vy: Velocity in y-direction.
        vx: Velocity in x-direction.
        sigmayy: Stress component sigmayy.
        sigmaxy: Stress component sigmaxy.
        sigmaxx: Stress component sigmaxx.
        m_sigmayyy: PML memory variable for sigmayy.
        m_sigmaxyy: PML memory variable for sigmaxy.
        m_sigmaxyx: PML memory variable for sigmaxy (x-direction).
        m_sigmaxxx: PML memory variable for sigmaxx.
        ay: PML absorption profile in y-direction.
        ayh: Half-step PML absorption profile in y-direction.
        ax: PML absorption profile in x-direction.
        axh: Half-step PML absorption profile in x-direction.
        by: PML damping profile in y-direction.
        byh: Half-step PML damping profile in y-direction.
        bx: PML damping profile in x-direction.
        bxh: Half-step PML damping profile in x-direction.
        rdy: Reciprocal of grid spacing in y-direction.
        rdx: Reciprocal of grid spacing in x-direction.
        dt: Time step interval.
        accuracy: Finite difference accuracy order.

    Returns:
        Tuple of updated velocity wavefields and PML memory variables.
    """
    dsigmayydy = deepwave.staggered_grid.diffyh1(sigmayy, accuracy, rdy)
    dsigmaxydx = deepwave.staggered_grid.diffx1(sigmaxy, accuracy, rdx)

    dsigmaxydy = deepwave.staggered_grid.diffy1(sigmaxy, accuracy, rdy)
    dsigmaxxdx = deepwave.staggered_grid.diffxh1(sigmaxx, accuracy, rdx)

    m_sigmayyy = ayh * m_sigmayyy + byh * dsigmayydy
    dsigmayydy = dsigmayydy + m_sigmayyy
    m_sigmaxyx = ax * m_sigmaxyx + bx * dsigmaxydx
    dsigmaxydx = dsigmaxydx + m_sigmaxyx

    m_sigmaxyy = ay * m_sigmaxyy + by * dsigmaxydy
    dsigmaxydy = dsigmaxydy + m_sigmaxyy
    m_sigmaxxx = axh * m_sigmaxxx + bxh * dsigmaxxdx
    dsigmaxxdx = dsigmaxxdx + m_sigmaxxx

    vy = vy + buoyancy_y * dt * (dsigmayydy + dsigmaxydx)
    vx = vx + buoyancy_x * dt * (dsigmaxydy + dsigmaxxdx)

    return vy, vx, m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx


def update_stresses(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    mu_yx: torch.Tensor,
    vy: torch.Tensor,
    vx: torch.Tensor,
    sigmayy: torch.Tensor,
    sigmaxy: torch.Tensor,
    sigmaxx: torch.Tensor,
    m_vyy: torch.Tensor,
    m_vyx: torch.Tensor,
    m_vxy: torch.Tensor,
    m_vxx: torch.Tensor,
    ay: torch.Tensor,
    ayh: torch.Tensor,
    ax: torch.Tensor,
    axh: torch.Tensor,
    by: torch.Tensor,
    byh: torch.Tensor,
    bx: torch.Tensor,
    bxh: torch.Tensor,
    rdy: torch.Tensor,
    rdx: torch.Tensor,
    dt: torch.Tensor,
    accuracy: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Updates the stress wavefields and PML memory variables.

    Args:
        lamb: First Lame parameter.
        mu: Second Lame parameter.
        mu_yx: Second Lame parameter at y+1/2, x+1/2.
        vy: Velocity in y-direction.
        vx: Velocity in x-direction.
        sigmayy: Stress component sigmayy.
        sigmaxy: Stress component sigmaxy.
        sigmaxx: Stress component sigmaxx.
        m_vyy: PML memory variable for vy.
        m_vyx: PML memory variable for vx in y-direction.
        m_vxy: PML memory variable for vy in x-direction.
        m_vxx: PML memory variable for vx.
        ay: PML absorption profile in y-direction.
        ayh: Half-step PML absorption profile in y-direction.
        ax: PML absorption profile in x-direction.
        axh: Half-step PML absorption profile in x-direction.
        by: PML damping profile in y-direction.
        byh: Half-step PML damping profile in y-direction.
        bx: PML damping profile in x-direction.
        bxh: Half-step PML damping profile in x-direction.
        rdy: Reciprocal of grid spacing in y-direction.
        rdx: Reciprocal of grid spacing in x-direction.
        dt: Time step interval.
        accuracy: Finite difference accuracy order.

    Returns:
        Tuple of updated stress wavefields and PML memory variables.
    """
    dvydy = deepwave.staggered_grid.diffy1(vy, accuracy, rdy)
    dvxdx = deepwave.staggered_grid.diffx1(vx, accuracy, rdx)

    m_vyy = ay * m_vyy + by * dvydy
    dvydy = dvydy + m_vyy
    m_vxx = ax * m_vxx + bx * dvxdx
    dvxdx = dvxdx + m_vxx
    sigmayy = sigmayy + dt * ((lamb + 2 * mu) * dvydy + lamb * dvxdx)
    sigmaxx = sigmaxx + dt * ((lamb + 2 * mu) * dvxdx + lamb * dvydy)

    dvydx = deepwave.staggered_grid.diffxh1(vy, accuracy, rdx)
    dvxdy = deepwave.staggered_grid.diffyh1(vx, accuracy, rdy)

    m_vxy = ayh * m_vxy + byh * dvxdy
    dvxdy = dvxdy + m_vxy
    m_vyx = axh * m_vyx + bxh * dvydx
    dvydx = dvydx + m_vyx
    sigmaxy = sigmaxy + dt * mu_yx * (dvydx + dvxdy)

    return sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx


def elastic_python(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    mu_yx: torch.Tensor,
    buoyancy_y: torch.Tensor,
    buoyancy_x: torch.Tensor,
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
        mu_yx: The second Lam'e parameter (mu) model tensor at y+1/2, x+1/2.
        buoyancy_y: The buoyancy (1/density) model tensor at y+1/2.
        buoyancy_x: The buoyancy (1/density) model tensor at x+1/2.
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
    if backward_callback is not None:
        raise RuntimeError("backward_callback is not supported in the Python backend.")
    lamb = lamb.contiguous()
    mu = mu.contiguous()
    mu_yx = mu_yx.contiguous()
    buoyancy_y = buoyancy_y.contiguous()
    buoyancy_x = buoyancy_x.contiguous()
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
    n_receivers_y_per_shot = receivers_y_i.numel() // n_shots
    n_receivers_x_per_shot = receivers_x_i.numel() // n_shots
    n_receivers_p_per_shot = receivers_p_i.numel() // n_shots
    receiver_amplitudes_y = torch.empty(0, device=device, dtype=dtype)
    receiver_amplitudes_x = torch.empty(0, device=device, dtype=dtype)
    receiver_amplitudes_p = torch.empty(0, device=device, dtype=dtype)

    fd_pad = accuracy // 2
    fd_pad_list = [fd_pad, fd_pad - 1, fd_pad, fd_pad - 1]
    size_with_batch = (n_shots, *lamb.shape[-2:])
    vy = deepwave.common.create_or_pad(
        vy,
        fd_pad_list,
        lamb.device,
        lamb.dtype,
        size_with_batch,
    )
    vx = deepwave.common.create_or_pad(
        vx,
        fd_pad_list,
        lamb.device,
        lamb.dtype,
        size_with_batch,
    )
    sigmayy = deepwave.common.create_or_pad(
        sigmayy,
        fd_pad_list,
        lamb.device,
        lamb.dtype,
        size_with_batch,
    )
    sigmaxy = deepwave.common.create_or_pad(
        sigmaxy,
        fd_pad_list,
        lamb.device,
        lamb.dtype,
        size_with_batch,
    )
    sigmaxx = deepwave.common.create_or_pad(
        sigmaxx,
        fd_pad_list,
        lamb.device,
        lamb.dtype,
        size_with_batch,
    )
    m_vyy = deepwave.common.create_or_pad(
        m_vyy,
        fd_pad_list,
        lamb.device,
        lamb.dtype,
        size_with_batch,
    )
    m_vyx = deepwave.common.create_or_pad(
        m_vyx,
        fd_pad_list,
        lamb.device,
        lamb.dtype,
        size_with_batch,
    )
    m_vxy = deepwave.common.create_or_pad(
        m_vxy,
        fd_pad_list,
        lamb.device,
        lamb.dtype,
        size_with_batch,
    )
    m_vxx = deepwave.common.create_or_pad(
        m_vxx,
        fd_pad_list,
        lamb.device,
        lamb.dtype,
        size_with_batch,
    )
    m_sigmayyy = deepwave.common.create_or_pad(
        m_sigmayyy,
        fd_pad_list,
        lamb.device,
        lamb.dtype,
        size_with_batch,
    )
    m_sigmaxyy = deepwave.common.create_or_pad(
        m_sigmaxyy,
        fd_pad_list,
        lamb.device,
        lamb.dtype,
        size_with_batch,
    )
    m_sigmaxyx = deepwave.common.create_or_pad(
        m_sigmaxyx,
        fd_pad_list,
        lamb.device,
        lamb.dtype,
        size_with_batch,
    )
    m_sigmaxxx = deepwave.common.create_or_pad(
        m_sigmaxxx,
        fd_pad_list,
        lamb.device,
        lamb.dtype,
        size_with_batch,
    )
    zero_bottom(sigmaxy, fd_pad)
    zero_right(sigmaxy, fd_pad)
    zero_bottom(m_vxy, fd_pad)
    zero_right(m_vxy, fd_pad)
    zero_bottom(m_vyx, fd_pad)
    zero_right(m_vyx, fd_pad)
    zero_bottom(vy, fd_pad)
    zero_bottom(m_sigmayyy, fd_pad)
    zero_bottom(m_sigmaxyx, fd_pad)
    zero_right(vx, fd_pad)
    zero_right(m_sigmaxyy, fd_pad)
    zero_right(m_sigmaxxx, fd_pad)

    if receivers_y_i.numel() > 0:
        receiver_amplitudes_y.resize_(nt + 1, n_shots, n_receivers_y_per_shot)
        receiver_amplitudes_y.fill_(0)
    if receivers_x_i.numel() > 0:
        receiver_amplitudes_x.resize_(nt + 1, n_shots, n_receivers_x_per_shot)
        receiver_amplitudes_x.fill_(0)
    if receivers_p_i.numel() > 0:
        receiver_amplitudes_p.resize_(nt, n_shots, n_receivers_p_per_shot)
        receiver_amplitudes_p.fill_(0)

    source_y_mask = sources_y_i != deepwave.common.IGNORE_LOCATION
    sources_y_i_masked = torch.zeros_like(sources_y_i)
    sources_y_i_masked[source_y_mask] = sources_y_i[source_y_mask]
    source_amplitudes_y_masked = torch.zeros_like(source_amplitudes_y)
    source_amplitudes_y_masked[:, source_y_mask] = source_amplitudes_y[:, source_y_mask]

    source_x_mask = sources_x_i != deepwave.common.IGNORE_LOCATION
    sources_x_i_masked = torch.zeros_like(sources_x_i)
    sources_x_i_masked[source_x_mask] = sources_x_i[source_x_mask]
    source_amplitudes_x_masked = torch.zeros_like(source_amplitudes_x)
    source_amplitudes_x_masked[:, source_x_mask] = source_amplitudes_x[:, source_x_mask]

    receiver_y_mask = receivers_y_i != deepwave.common.IGNORE_LOCATION
    receivers_y_i_masked = torch.zeros_like(receivers_y_i)
    receivers_y_i_masked[receiver_y_mask] = receivers_y_i[receiver_y_mask]

    receiver_x_mask = receivers_x_i != deepwave.common.IGNORE_LOCATION
    receivers_x_i_masked = torch.zeros_like(receivers_x_i)
    receivers_x_i_masked[receiver_x_mask] = receivers_x_i[receiver_x_mask]

    receiver_p_mask = receivers_p_i != deepwave.common.IGNORE_LOCATION
    receivers_p_i_masked = torch.zeros_like(receivers_p_i)
    receivers_p_i_masked[receiver_p_mask] = receivers_p_i[receiver_p_mask]

    rdy = torch.tensor(1 / dy, dtype=dtype, device=device)
    rdx = torch.tensor(1 / dx, dtype=dtype, device=device)
    dt_tensor = torch.tensor(dt, dtype=dtype, device=device)

    for step in range(nt // step_ratio):
        if forward_callback is not None and step % callback_frequency == 0:
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
                {
                    "lamb": lamb,
                    "mu": mu,
                    "mu_yx": mu_yx,
                    "buoyancy_y": buoyancy_y,
                    "buoyancy_x": buoyancy_x,
                },
                {},
                fd_pad_list,
                pml_width,
            )
            forward_callback(state)
        for inner_step in range(step_ratio):
            t = step * step_ratio + inner_step

            if receiver_amplitudes_y.numel() > 0:
                receiver_amplitudes_y[t] = vy.view(-1, ny * nx).gather(
                    1, receivers_y_i_masked
                )
            if receiver_amplitudes_x.numel() > 0:
                receiver_amplitudes_x[t] = vx.view(-1, ny * nx).gather(
                    1, receivers_x_i_masked
                )
            if receiver_amplitudes_p.numel() > 0:
                receiver_amplitudes_p[t] = (
                    -(
                        sigmayy.view(-1, ny * nx).gather(1, receivers_p_i_masked)
                        + sigmaxx.view(-1, ny * nx).gather(1, receivers_p_i_masked)
                    )
                    / 2
                )

            vy, vx, m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx = (
                _update_velocities_opt(
                    buoyancy_y,
                    buoyancy_x,
                    vy,
                    vx,
                    sigmayy,
                    sigmaxy,
                    sigmaxx,
                    m_sigmayyy,
                    m_sigmaxyy,
                    m_sigmaxyx,
                    m_sigmaxxx,
                    ay,
                    ayh,
                    ax,
                    axh,
                    by,
                    byh,
                    bx,
                    bxh,
                    rdy,
                    rdx,
                    dt_tensor,
                    accuracy,
                )
            )

            if source_amplitudes_y_masked.numel() > 0:
                vy.view(-1, ny * nx).scatter_add_(
                    1, sources_y_i_masked, source_amplitudes_y_masked[t]
                )
            if source_amplitudes_x_masked.numel() > 0:
                vx.view(-1, ny * nx).scatter_add_(
                    1, sources_x_i_masked, source_amplitudes_x_masked[t]
                )

            sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx = (
                _update_stresses_opt(
                    lamb,
                    mu,
                    mu_yx,
                    vy,
                    vx,
                    sigmayy,
                    sigmaxy,
                    sigmaxx,
                    m_vyy,
                    m_vyx,
                    m_vxy,
                    m_vxx,
                    ay,
                    ayh,
                    ax,
                    axh,
                    by,
                    byh,
                    bx,
                    bxh,
                    rdy,
                    rdx,
                    dt_tensor,
                    accuracy,
                )
            )

            # if source_amplitudes_p is not None:
            #    sigmayy.view(-1, ny * nx).scatter_add_(
            #        1, sources_p_i, source_amplitudes_p[t]
            #    )
            #    sigmaxx.view(-1, ny * nx).scatter_add_(
            #        1, sources_p_i, source_amplitudes_p[t]
            #    )

    if receiver_amplitudes_y.numel() > 0:
        receiver_amplitudes_y[-1] = vy.view(-1, ny * nx).gather(1, receivers_y_i_masked)
    if receiver_amplitudes_x.numel() > 0:
        receiver_amplitudes_x[-1] = vx.view(-1, ny * nx).gather(1, receivers_x_i_masked)

    receiver_amplitudes_y_masked = torch.zeros_like(receiver_amplitudes_y)
    if receiver_amplitudes_y.numel() > 0:
        receiver_amplitudes_y_masked[:, receiver_y_mask] = receiver_amplitudes_y[
            :, receiver_y_mask
        ]
    receiver_amplitudes_x_masked = torch.zeros_like(receiver_amplitudes_x)
    if receiver_amplitudes_x.numel() > 0:
        receiver_amplitudes_x_masked[:, receiver_x_mask] = receiver_amplitudes_x[
            :, receiver_x_mask
        ]
    receiver_amplitudes_p_masked = torch.zeros_like(receiver_amplitudes_p)
    if receiver_amplitudes_p.numel() > 0:
        receiver_amplitudes_p_masked[:, receiver_p_mask] = receiver_amplitudes_p[
            :, receiver_p_mask
        ]

    s = (
        slice(None),
        slice(fd_pad, ny - (fd_pad - 1)),
        slice(fd_pad, nx - (fd_pad - 1)),
    )
    return (
        vy[s],
        vx[s],
        sigmayy[s],
        sigmaxy[s],
        sigmaxx[s],
        m_vyy[s],
        m_vyx[s],
        m_vxy[s],
        m_vxx[s],
        m_sigmayyy[s],
        m_sigmaxyy[s],
        m_sigmaxyx[s],
        m_sigmaxxx[s],
        receiver_amplitudes_p_masked,
        receiver_amplitudes_y_masked,
        receiver_amplitudes_x_masked,
    )


_update_velocities_jit = None
_update_velocities_compile = None
_update_velocities_opt = update_velocities
_update_stresses_jit = None
_update_stresses_compile = None
_update_stresses_opt = update_stresses


def elastic_func(
    python_backend: Union[bool, str], *args: Any
) -> Tuple[torch.Tensor, ...]:
    """A helper function to apply the ElasticForwardFunc.

    This function serves as a convenient wrapper to call the `apply` method
    of `ElasticForwardFunc`, which is the entry point for the autograd graph
    for elastic wave propagation.

    Args:
        python_backend: Bool or string specifying whether to use Python backend.
        *args: Variable length argument list to be passed directly to
            `ElasticForwardFunc.apply`.

    Returns:
        The results of the forward pass from `ElasticForwardFunc.apply`.

    """
    global _update_velocities_jit, _update_velocities_compile, _update_velocities_opt
    global _update_stresses_jit, _update_stresses_compile, _update_stresses_opt

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
            if _update_stresses_jit is None:
                _update_stresses_jit = torch.jit.script(update_stresses)
            _update_stresses_opt = _update_stresses_jit
        elif mode == "compile":
            if _update_velocities_compile is None:
                _update_velocities_compile = torch.compile(
                    update_velocities, fullgraph=True
                )
            _update_velocities_opt = _update_velocities_compile
            if _update_stresses_compile is None:
                _update_stresses_compile = torch.compile(
                    update_stresses, fullgraph=True
                )
            _update_stresses_opt = _update_stresses_compile
        elif mode == "eager":
            _update_velocities_opt = update_velocities
            _update_stresses_opt = update_stresses
        else:
            raise ValueError(f"Unknown python_backend value {mode!r}.")

    func = elastic_python if python_backend else ElasticForwardFunc.apply

    return cast(
        "Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "
        "torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "
        "torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "
        "torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]",
        func(*args),
    )
