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

from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, cast

import torch

import deepwave.backend_utils
import deepwave.common
import deepwave.staggered_grid


def prepare_parameters(mu: torch.Tensor, buoyancy: torch.Tensor) -> List[torch.Tensor]:
    """Prepares elastic properties for the free surface method.

    This function applies the Zeng et al. (2012) logic for all cell types.

    Args:
        mu (torch.Tensor): Lame parameter mu.
        buoyancy (torch.Tensor): Buoyancy (1/rho).

    Returns:
        tuple: (mu_yx, buoyancy_y, buoyancy_x)
    """
    ndim = mu.ndim - 1
    rfmax = 1 / torch.finfo(mu.dtype).max ** (1 / 2)
    parameters = []

    # Mu (harmonic mean)
    mu_safe = torch.where(mu.abs() > rfmax, mu, torch.ones_like(mu))
    if ndim >= 3:
        mask = (
            (rfmax < mu[..., 1:, 1:, :].abs())
            .logical_and(rfmax < mu[..., :-1, :-1, :].abs())
            .logical_and(rfmax < mu[..., 1:, :-1, :].abs())
            .logical_and(rfmax < mu[..., :-1, 1:, :].abs())
        )
        mu_zy_val = 4 / (
            1 / mu_safe[..., 1:, 1:, :]
            + 1 / mu_safe[..., :-1, :-1, :]
            + 1 / mu_safe[..., 1:, :-1, :]
            + 1 / mu_safe[..., :-1, 1:, :]
        )
        mu_zy = torch.where(mask, mu_zy_val, torch.zeros_like(mu_zy_val))
        mu_zy = torch.nn.functional.pad(mu_zy, (0, 0, 0, 1, 0, 1))
        parameters.append(mu_zy)
        del mask, mu_zy_val, mu_zy
        mask = (
            (rfmax < mu[..., 1:, :, 1:].abs())
            .logical_and(rfmax < mu[..., :-1, :, :-1].abs())
            .logical_and(rfmax < mu[..., 1:, :, :-1].abs())
            .logical_and(rfmax < mu[..., :-1, :, 1:].abs())
        )
        mu_zx_val = 4 / (
            1 / mu_safe[..., 1:, :, 1:]
            + 1 / mu_safe[..., :-1, :, :-1]
            + 1 / mu_safe[..., 1:, :, :-1]
            + 1 / mu_safe[..., :-1, :, 1:]
        )
        mu_zx = torch.where(mask, mu_zx_val, torch.zeros_like(mu_zx_val))
        mu_zx = torch.nn.functional.pad(mu_zx, (0, 1, 0, 0, 0, 1))
        parameters.append(mu_zx)
        del mask, mu_zx_val, mu_zx
    if ndim >= 2:
        mask = (
            (rfmax < mu[..., 1:, 1:].abs())
            .logical_and(rfmax < mu[..., :-1, :-1].abs())
            .logical_and(rfmax < mu[..., 1:, :-1].abs())
            .logical_and(rfmax < mu[..., :-1, 1:].abs())
        )
        mu_yx_val = 4 / (
            1 / mu_safe[..., 1:, 1:]
            + 1 / mu_safe[..., :-1, :-1]
            + 1 / mu_safe[..., 1:, :-1]
            + 1 / mu_safe[..., :-1, 1:]
        )
        mu_yx = torch.where(mask, mu_yx_val, torch.zeros_like(mu_yx_val))
        mu_yx = torch.nn.functional.pad(mu_yx, (0, 1, 0, 1))
        parameters.append(mu_yx)
        del mask, mu_yx_val, mu_yx

    # Buoyancy (inverse of arithmetic mean of density)
    # Arithmetic mean: rho[i+1/2] = (rho[i] + rho[i+1])/2
    # => buoyancy[i+1/2] = 2/(rho[i] + rho[i+1])
    mask = rfmax < buoyancy.abs()
    buoyancy_safe = torch.where(mask, buoyancy, torch.ones_like(buoyancy))
    rho = torch.where(mask, 1 / buoyancy_safe, torch.zeros_like(buoyancy))
    del mask, buoyancy_safe
    if ndim >= 3:
        rho_z = torch.nn.functional.pad(
            (rho[..., :-1, :, :] + rho[..., 1:, :, :]) / 2, (0, 0, 0, 0, 0, 1)
        )
        mask = rfmax < rho_z.abs()
        rho_z_safe = torch.where(mask, rho_z, torch.ones_like(rho_z))
        buoyancy_z = torch.where(mask, 1 / rho_z_safe, torch.zeros_like(buoyancy))
        parameters.append(buoyancy_z)
        del rho_z, mask, rho_z_safe, buoyancy_z
    if ndim >= 2:
        rho_y = torch.nn.functional.pad(
            (rho[..., :-1, :] + rho[..., 1:, :]) / 2, (0, 0, 0, 1)
        )
        mask = rfmax < rho_y.abs()
        rho_y_safe = torch.where(mask, rho_y, torch.ones_like(rho_y))
        buoyancy_y = torch.where(mask, 1 / rho_y_safe, torch.zeros_like(buoyancy))
        parameters.append(buoyancy_y)
        del rho_y, mask, rho_y_safe, buoyancy_y
    rho_x = torch.nn.functional.pad((rho[..., :-1] + rho[..., 1:]) / 2, (0, 1))
    mask = rfmax < rho_x.abs()
    rho_x_safe = torch.where(mask, rho_x, torch.ones_like(rho_x))
    buoyancy_x = torch.where(mask, 1 / rho_x_safe, torch.zeros_like(buoyancy))
    parameters.append(buoyancy_x)

    return parameters


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
        storage_mode: A string specifying the storage mode for intermediate
            data. One of "device", "cpu", "disk", or "none". Defaults to "device".
        storage_path: A string specifying the path for disk storage.
            Defaults to ".".
        storage_compression: A bool specifying whether to use compression
            for intermediate data. Defaults to False.

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
        storage_mode: Literal["device", "cpu", "disk", "none"] = "device",
        storage_path: str = ".",
        storage_compression: bool = False,
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
        self.storage_mode = storage_mode
        self.storage_path = storage_path
        self.storage_compression = storage_compression

    def forward(
        self,
        dt: float,
        source_amplitudes_z: Optional[torch.Tensor] = None,
        source_amplitudes_y: Optional[torch.Tensor] = None,
        source_amplitudes_x: Optional[torch.Tensor] = None,
        source_amplitudes_p: Optional[torch.Tensor] = None,
        source_locations_z: Optional[torch.Tensor] = None,
        source_locations_y: Optional[torch.Tensor] = None,
        source_locations_x: Optional[torch.Tensor] = None,
        source_locations_p: Optional[torch.Tensor] = None,
        receiver_locations_z: Optional[torch.Tensor] = None,
        receiver_locations_y: Optional[torch.Tensor] = None,
        receiver_locations_x: Optional[torch.Tensor] = None,
        receiver_locations_p: Optional[torch.Tensor] = None,
        accuracy: int = 4,
        pml_width: Union[int, Sequence[int]] = 20,
        pml_freq: Optional[float] = None,
        max_vel: Optional[float] = None,
        survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
        vz_0: Optional[torch.Tensor] = None,
        vy_0: Optional[torch.Tensor] = None,
        vx_0: Optional[torch.Tensor] = None,
        sigmazz_0: Optional[torch.Tensor] = None,
        sigmayz_0: Optional[torch.Tensor] = None,
        sigmaxz_0: Optional[torch.Tensor] = None,
        sigmayy_0: Optional[torch.Tensor] = None,
        sigmaxy_0: Optional[torch.Tensor] = None,
        sigmaxx_0: Optional[torch.Tensor] = None,
        m_vzz_0: Optional[torch.Tensor] = None,
        m_vzy_0: Optional[torch.Tensor] = None,
        m_vzx_0: Optional[torch.Tensor] = None,
        m_vyz_0: Optional[torch.Tensor] = None,
        m_vxz_0: Optional[torch.Tensor] = None,
        m_vyy_0: Optional[torch.Tensor] = None,
        m_vyx_0: Optional[torch.Tensor] = None,
        m_vxy_0: Optional[torch.Tensor] = None,
        m_vxx_0: Optional[torch.Tensor] = None,
        m_sigmazzz_0: Optional[torch.Tensor] = None,
        m_sigmayzy_0: Optional[torch.Tensor] = None,
        m_sigmaxzx_0: Optional[torch.Tensor] = None,
        m_sigmayzz_0: Optional[torch.Tensor] = None,
        m_sigmaxzz_0: Optional[torch.Tensor] = None,
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
        python_backend: Union[Literal["eager", "jit", "compile"], bool] = False,
    ) -> Tuple[torch.Tensor, ...]:
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
            source_amplitudes_z=source_amplitudes_z,
            source_amplitudes_y=source_amplitudes_y,
            source_amplitudes_x=source_amplitudes_x,
            source_amplitudes_p=source_amplitudes_p,
            source_locations_z=source_locations_z,
            source_locations_y=source_locations_y,
            source_locations_x=source_locations_x,
            source_locations_p=source_locations_p,
            receiver_locations_z=receiver_locations_z,
            receiver_locations_y=receiver_locations_y,
            receiver_locations_x=receiver_locations_x,
            receiver_locations_p=receiver_locations_p,
            accuracy=accuracy,
            pml_width=pml_width,
            pml_freq=pml_freq,
            max_vel=max_vel,
            survey_pad=survey_pad,
            vz_0=vz_0,
            vy_0=vy_0,
            vx_0=vx_0,
            sigmazz_0=sigmazz_0,
            sigmayz_0=sigmayz_0,
            sigmaxz_0=sigmaxz_0,
            sigmayy_0=sigmayy_0,
            sigmaxy_0=sigmaxy_0,
            sigmaxx_0=sigmaxx_0,
            m_vzz_0=m_vzz_0,
            m_vzy_0=m_vzy_0,
            m_vzx_0=m_vzx_0,
            m_vyz_0=m_vyz_0,
            m_vxz_0=m_vxz_0,
            m_vyy_0=m_vyy_0,
            m_vyx_0=m_vyx_0,
            m_vxy_0=m_vxy_0,
            m_vxx_0=m_vxx_0,
            m_sigmazzz_0=m_sigmazzz_0,
            m_sigmayzy_0=m_sigmayzy_0,
            m_sigmaxzx_0=m_sigmaxzx_0,
            m_sigmayzz_0=m_sigmayzz_0,
            m_sigmaxzz_0=m_sigmaxzz_0,
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
            storage_mode=self.storage_mode,
            storage_path=self.storage_path,
            storage_compression=self.storage_compression,
        )


def elastic(
    lamb: torch.Tensor,
    mu: torch.Tensor,
    buoyancy: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitudes_z: Optional[torch.Tensor] = None,
    source_amplitudes_y: Optional[torch.Tensor] = None,
    source_amplitudes_x: Optional[torch.Tensor] = None,
    source_amplitudes_p: Optional[torch.Tensor] = None,
    source_locations_z: Optional[torch.Tensor] = None,
    source_locations_y: Optional[torch.Tensor] = None,
    source_locations_x: Optional[torch.Tensor] = None,
    source_locations_p: Optional[torch.Tensor] = None,
    receiver_locations_z: Optional[torch.Tensor] = None,
    receiver_locations_y: Optional[torch.Tensor] = None,
    receiver_locations_x: Optional[torch.Tensor] = None,
    receiver_locations_p: Optional[torch.Tensor] = None,
    accuracy: int = 4,
    pml_width: Union[int, Sequence[int]] = 20,
    pml_freq: Optional[float] = None,
    max_vel: Optional[float] = None,
    survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
    vz_0: Optional[torch.Tensor] = None,
    vy_0: Optional[torch.Tensor] = None,
    vx_0: Optional[torch.Tensor] = None,
    sigmazz_0: Optional[torch.Tensor] = None,
    sigmayz_0: Optional[torch.Tensor] = None,
    sigmaxz_0: Optional[torch.Tensor] = None,
    sigmayy_0: Optional[torch.Tensor] = None,
    sigmaxy_0: Optional[torch.Tensor] = None,
    sigmaxx_0: Optional[torch.Tensor] = None,
    m_vzz_0: Optional[torch.Tensor] = None,
    m_vzy_0: Optional[torch.Tensor] = None,
    m_vzx_0: Optional[torch.Tensor] = None,
    m_vyz_0: Optional[torch.Tensor] = None,
    m_vxz_0: Optional[torch.Tensor] = None,
    m_vyy_0: Optional[torch.Tensor] = None,
    m_vyx_0: Optional[torch.Tensor] = None,
    m_vxy_0: Optional[torch.Tensor] = None,
    m_vxx_0: Optional[torch.Tensor] = None,
    m_sigmazzz_0: Optional[torch.Tensor] = None,
    m_sigmayzy_0: Optional[torch.Tensor] = None,
    m_sigmaxzx_0: Optional[torch.Tensor] = None,
    m_sigmayzz_0: Optional[torch.Tensor] = None,
    m_sigmaxzz_0: Optional[torch.Tensor] = None,
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
    python_backend: Union[Literal["eager", "jit", "compile"], bool] = False,
    storage_mode: Literal["device", "cpu", "disk", "none"] = "device",
    storage_path: str = ".",
    storage_compression: bool = False,
) -> Tuple[torch.Tensor, ...]:
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
            condition for stability). Due to the staggered time grid,
            stress-related fields (including pressure) are at integer time
            steps `t*dt`, while velocity-related fields are at half time
            steps `(t-0.5)*dt`.
        source_amplitudes_z: A Tensor with dimensions [shot, source, time]
            of force densities in the z direction, with units of
            force/volume. The samples correspond to times `(t-0.5)*dt`.
        source_amplitudes_y: A Tensor with dimensions [shot, source, time]
            of force densities in the y direction, with units of
            force/volume. The samples correspond to times `(t-0.5)*dt`.
        source_amplitudes_x: A Tensor with dimensions [shot, source, time]
            of force densities in the x direction, with units of
            force/volume. The samples correspond to times `(t-0.5)*dt`.
        source_amplitudes_p: A Tensor with dimensions [shot, source, time]
            of pressure rates, with units of pressure/time. The samples
            correspond to times `t*dt`.
        source_locations_z: A Tensor with dimensions [shot, source, ndim],
            containing the index in the ndim spatial dimensions of the cell
            that each source oriented in the z spatial dimension is
            located in, relative to the origin of the model. Setting the
            coordinates to deepwave.IGNORE_LOCATION will result in the
            source being ignored.
        source_locations_y: A Tensor containing the locations of sources
            oriented in the y spatial dimension.
        source_locations_x: A Tensor containing the locations of sources
            oriented in the x spatial dimension.
        source_locations_p: A Tensor containing the locations of pressure
            sources.
        receiver_locations_z: A Tensor with dimensions [shot, receiver, ndim],
            containing the coordinates of the cell containing each receiver
            oriented in the z spatial dimension. Setting the
            coordinates to deepwave.IGNORE_LOCATION will result in the
            receiver being ignored.
        receiver_locations_y: A Tensor containing the coordinates of the
            receivers oriented in the y spatial dimension.
        receiver_locations_x: A Tensor containing the coordinates of the
            receivers oriented in the x spatial dimension.
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
            - ...

            Larger values result in smaller reflections, with values of 10
            to 20 being typical. For a reflective rigid surface, set the
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
        vz_0: Initial vz (velocity in the z dimension) wavefield at time
            step -1/2.
        vy_0: Initial vy (velocity in the y dimension) wavefield at time
            step -1/2.
        vx_0: Initial vx (velocity in the x dimension) wavefield at time
            step -1/2.
        sigmazz_0: Initial value for the yy component of the stress field at
            time step 0.
        sigmayz_0: Initial value for the yz component of the stress field at
            time step 0.
        sigmaxz_0: Initial value for the xz component of the stress field at
            time step 0.
        sigmayy_0: Initial value for the yy component of the stress field at
            time step 0.
        sigmaxy_0: Initial value for the xy component of the stress field at
            time step 0.
        sigmaxx_0: Initial value for the xx component of the stress field at
            time step 0.
        m_vzz_0: Initial value for the "memory variable" for the zz component
            of the velocity, used in the PML.
        m_vzy_0: Initial value for the "memory variable" for the zy component
            of the velocity, used in the PML.
        m_vzx_0: Initial value for the "memory variable" for the zx component
            of the velocity, used in the PML.
        m_vyz_0: Initial value for the "memory variable" for the yz component
            of the velocity, used in the PML.
        m_vxz_0: Initial value for the "memory variable" for the xz component
            of the velocity, used in the PML.
        m_vyy_0: Initial value for the "memory variable" for the yy component
            of the velocity, used in the PML.
        m_vyx_0: Initial value for the "memory variable" for the yx component
            of the velocity, used in the PML.
        m_vxy_0: Initial value for the "memory variable" for the xy component
            of the velocity, used in the PML.
        m_vxx_0: Initial value for the "memory variable" for the xx component
            of the velocity, used in the PML.
        m_sigmazzz_0: Initial value for the "memory variable" for the zzz
            component of the stress, used in the PML.
        m_sigmayzy_0: Initial value for the "memory variable" for the yzy
            component of the stress, used in the PML.
        m_sigmaxzx_0: Initial value for the "memory variable" for the xzx
            component of the stress, used in the PML.
        m_sigmayzz_0: Initial value for the "memory variable" for the yzz
            component of the stress, used in the PML.
        m_sigmaxzz_0: Initial value for the "memory variable" for the xzz
            component of the stress, used in the PML.
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
        storage_mode: A string specifying the storage mode for intermediate
            data. One of "device", "cpu", "disk", or "none". Defaults to "device".
        storage_path: A string specifying the path for disk storage.
            Defaults to ".".
        storage_compression: A bool specifying whether to use compression
            for intermediate data. Defaults to False.

    Returns:
        Tuple:

            - vz: Final velocity wavefield in the z-dimension (if ndim == 3).
            - vy: Final velocity wavefield in the y-dimension (if ndim >= 2).
            - vx: Final velocity wavefield in the x-dimension.
            - sigmazz: Final stress wavefield (zz component) (if ndim == 3).
            - sigmayz: Final stress wavefield (yz component) (if ndim == 3).
            - sigmaxz: Final stress wavefield (xz component) (if ndim == 3).
            - sigmayy: Final stress wavefield (yy component) (if ndim >= 2).
            - sigmaxy: Final stress wavefield (xy component) (if ndim >= 2).
            - sigmaxx: Final stress wavefield (xx component).
            - m_vzz: Final velocity memory variable for the PML (if ndim == 3).
            - m_vzy: Final velocity memory variable for the PML (if ndim == 3).
            - m_vzx: Final velocity memory variable for the PML (if ndim == 3).
            - m_vyz: Final velocity memory variable for the PML (if ndim == 3).
            - m_vxz: Final velocity memory variable for the PML (if ndim == 3).
            - m_vyy: Final velocity memory variable for the PML (if ndim >= 2).
            - m_vyx: Final velocity memory variable for the PML (if ndim >= 2).
            - m_vxy: Final velocity memory variable for the PML (if ndim >= 2).
            - m_vxx: Final velocity memory variable for the PML.
            - m_sigmazzz: Final stress memory variable for the PML (if ndim == 3).
            - m_sigmayzy: Final stress memory variable for the PML (if ndim == 3).
            - m_sigmaxzx: Final stress memory variable for the PML (if ndim == 3).
            - m_sigmayzz: Final stress memory variable for the PML (if ndim == 3).
            - m_sigmaxzz: Final stress memory variable for the PML (if ndim == 3).
            - m_sigmayyy: Final stress memory variable for the PML (if ndim >= 2).
            - m_sigmaxyy: Final stress memory variable for the PML (if ndim >= 2).
            - m_sigmaxyx: Final stress memory variable for the PML (if ndim >= 2).
            - m_sigmaxxx: Final stress memory variable for the PML.
            - receiver_amplitudes_p: Recorded pressure data, with units of
              pressure. The samples correspond to times `t*dt`.
            - receiver_amplitudes_z: Recorded z-component velocity data, with
              units of velocity. The samples correspond to times `(t-0.5)*dt`.
            - receiver_amplitudes_y: Recorded y-component velocity data, with
              units of velocity. The samples correspond to times `(t-0.5)*dt`.
            - receiver_amplitudes_x: Recorded x-component velocity data, with
              units of velocity. The samples correspond to times `(t-0.5)*dt`.

    """
    deepwave.common.check_inputs_not_vmapped(
        source_amplitudes_p,
        source_amplitudes_z,
        source_amplitudes_y,
        source_amplitudes_x,
        source_locations_z,
        source_locations_y,
        source_locations_x,
        source_locations_p,
        receiver_locations_z,
        receiver_locations_y,
        receiver_locations_x,
        receiver_locations_p,
        vz_0,
        sigmazz_0,
        sigmayz_0,
        sigmaxz_0,
        m_vzz_0,
        m_vzy_0,
        m_vzx_0,
        m_vyz_0,
        m_vxz_0,
        m_sigmazzz_0,
        m_sigmayzy_0,
        m_sigmaxzx_0,
        m_sigmayzz_0,
        m_sigmaxzz_0,
        vy_0,
        sigmayy_0,
        sigmaxy_0,
        m_vyy_0,
        m_vyx_0,
        m_vxy_0,
        m_sigmayyy_0,
        m_sigmaxyy_0,
        m_sigmaxyx_0,
        vx_0,
        m_vxx_0,
        m_sigmaxxx_0,
    )
    ndim = deepwave.common.get_ndim(
        [lamb, mu, buoyancy],
        [],
        [
            source_locations_z,
            source_locations_y,
            source_locations_x,
            source_locations_p,
            receiver_locations_z,
            receiver_locations_y,
            receiver_locations_x,
            receiver_locations_p,
        ],
        [
            vz_0,
            sigmazz_0,
            sigmayz_0,
            sigmaxz_0,
            m_vzz_0,
            m_vzy_0,
            m_vzx_0,
            m_vyz_0,
            m_vxz_0,
            m_sigmazzz_0,
            m_sigmayzy_0,
            m_sigmaxzx_0,
            m_sigmayzz_0,
            m_sigmaxzz_0,
        ],
        [
            vy_0,
            sigmayy_0,
            sigmaxy_0,
            m_vyy_0,
            m_vyx_0,
            m_vxy_0,
            m_sigmayyy_0,
            m_sigmaxyy_0,
            m_sigmaxyx_0,
        ],
        [vx_0, m_vxx_0, m_sigmaxxx_0],
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

    # Check that sources and receivers are not on the last row or column,
    # as these are not used
    source_amplitudes: List[Optional[torch.Tensor]] = []
    source_locations: List[Optional[torch.Tensor]] = []
    receiver_locations: List[Optional[torch.Tensor]] = []
    initial_wavefields: List[Optional[torch.Tensor]] = []
    if ndim >= 3:
        source_amplitudes.append(source_amplitudes_z)
        source_locations.append(source_locations_z)
        receiver_locations.append(receiver_locations_z)
        initial_wavefields.extend(
            [
                vz_0,
                sigmazz_0,
                sigmayz_0,
                sigmaxz_0,
                m_vzz_0,
                m_vzy_0,
                m_vzx_0,
                m_vyz_0,
                m_vxz_0,
                m_sigmazzz_0,
                m_sigmayzy_0,
                m_sigmaxzx_0,
                m_sigmayzz_0,
                m_sigmaxzz_0,
            ]
        )
    if ndim >= 2:
        source_amplitudes.append(source_amplitudes_y)
        source_locations.append(source_locations_y)
        receiver_locations.append(receiver_locations_y)
        initial_wavefields.extend(
            [
                vy_0,
                sigmayy_0,
                sigmaxy_0,
                m_vyy_0,
                m_vyx_0,
                m_vxy_0,
                m_sigmayyy_0,
                m_sigmaxyy_0,
                m_sigmaxyx_0,
            ]
        )
    source_amplitudes.append(source_amplitudes_x)
    source_locations.append(source_locations_x)
    receiver_locations.append(receiver_locations_x)
    initial_wavefields.extend(
        [
            vx_0,
            sigmaxx_0,
            m_vxx_0,
            m_sigmaxxx_0,
        ]
    )
    dim_names = ["z", "y", "x"]
    for dim in range(ndim):
        for locations, name in [
            (source_locations, "source"),
            (receiver_locations, "receiver"),
        ]:
            location = locations[dim]
            if location is None:
                continue
            dim_location = location[..., -ndim + dim]
            dim_location = dim_location[dim_location != deepwave.IGNORE_LOCATION]
            if (
                dim_location.numel() > 0
                and dim_location.max() >= lamb.shape[-ndim + dim] - 1
            ):
                raise RuntimeError(
                    f"With the provided model, the maximum {dim_names[-ndim + dim]} "
                    f"{name} location in the {dim_names[-ndim + dim]} "
                    f"dimension must be less "
                    f"than {lamb.shape[-ndim + dim] - 1}.",
                )

    source_amplitudes.append(source_amplitudes_p)
    source_locations.append(source_locations_p)
    receiver_locations.append(receiver_locations_p)
    if any(deepwave.common.is_inside_vmap(prop) for prop in [lamb, mu, buoyancy]):
        if max_vel is None:
            raise RuntimeError(
                "If using BatchedTensor inputs, you must specify max_vel"
            )
        max_model_vel = max_vel
        min_nonzero_model_vel = 0.0
    else:
        vp, vs, _ = deepwave.common.lambmubuoyancy_to_vpvsrho(
            lamb,
            mu,
            buoyancy,
        )
        max_model_vel = max(vp.abs().max().item(), vs.abs().max().item())
        vp_nonzero = vp[vp != 0]
        min_nonzero_vp = (
            vp_nonzero.abs().min().item() if vp_nonzero.numel() > 0 else 0.0
        )
        vs_nonzero = vs[vs != 0]
        min_nonzero_vs = (
            vs_nonzero.abs().min().item() if vs_nonzero.numel() > 0 else 0.0
        )
        if min_nonzero_vp == 0 and min_nonzero_vs == 0:
            min_nonzero_model_vel = 0.0
        elif min_nonzero_vp == 0:
            min_nonzero_model_vel = float(min_nonzero_vs)
        elif min_nonzero_vs == 0:
            min_nonzero_model_vel = float(min_nonzero_vp)
        else:
            min_nonzero_model_vel = float(min(min_nonzero_vp, min_nonzero_vs))
        del vp, vs, vp_nonzero, vs_nonzero
    fd_pad = [accuracy // 2, accuracy // 2 - 1] * ndim

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
        [lamb, mu, buoyancy],
        ["replicate"] * 3,
        grid_spacing,
        dt,
        source_amplitudes,
        source_locations,
        receiver_locations,
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
    del lamb, mu, buoyancy
    if ndim >= 3:
        del (
            source_amplitudes_z,
            source_locations_z,
            receiver_locations_z,
            vz_0,
            sigmazz_0,
            sigmayz_0,
            sigmaxz_0,
            m_vzz_0,
            m_vzy_0,
            m_vzx_0,
            m_vyz_0,
            m_vxz_0,
            m_sigmazzz_0,
            m_sigmayzy_0,
            m_sigmaxzx_0,
            m_sigmayzz_0,
            m_sigmaxzz_0,
        )
    if ndim >= 2:
        del (
            source_amplitudes_y,
            source_locations_y,
            receiver_locations_y,
            vy_0,
            sigmayy_0,
            sigmaxy_0,
            m_vyy_0,
            m_vyx_0,
            m_vxy_0,
            m_sigmayyy_0,
            m_sigmaxyy_0,
            m_sigmaxyx_0,
        )
    del (
        source_amplitudes_x,
        source_locations_x,
        receiver_locations_x,
        vx_0,
        sigmaxx_0,
        m_vxx_0,
        m_sigmaxxx_0,
    )
    del source_amplitudes, source_locations, receiver_locations

    models.extend(prepare_parameters(models[1], models[2]))
    del models[2]  # Remove buoyancy as it is no longer needed

    model_shape = models[0].shape[-ndim:]
    flat_model_shape = int(torch.prod(torch.tensor(model_shape)).item())
    if ndim == 3:
        # source_amplitudes_z
        mask = sources_i[-4] == deepwave.common.IGNORE_LOCATION
        sources_i_masked = sources_i[-4].clone()
        sources_i_masked[mask] = 0
        if source_amplitudes_out[-4].numel() > 0:
            source_amplitudes_out[-4] = (
                source_amplitudes_out[-4]
                * (
                    models[-3]  # buoyancy_z
                    .view(-1, flat_model_shape)
                    .expand(n_shots, -1)
                    .gather(1, sources_i_masked)
                )
                * dt
            )
    if ndim >= 2:
        # source_amplitudes_y
        mask = sources_i[-3] == deepwave.common.IGNORE_LOCATION
        sources_i_masked = sources_i[-3].clone()
        sources_i_masked[mask] = 0
        if source_amplitudes_out[-3].numel() > 0:
            source_amplitudes_out[-3] = (
                source_amplitudes_out[-3]
                * (
                    models[-2]  # buoyancy_y
                    .view(-1, flat_model_shape)
                    .expand(n_shots, -1)
                    .gather(1, sources_i_masked)
                )
                * dt
            )
    # source_amplitudes_x
    mask = sources_i[-2] == deepwave.common.IGNORE_LOCATION
    sources_i_masked = sources_i[-2].clone()
    sources_i_masked[mask] = 0
    if source_amplitudes_out[-2].numel() > 0:
        source_amplitudes_out[-2] = (
            source_amplitudes_out[-2]
            * (
                models[-1]  # buoyancy_x
                .view(-1, flat_model_shape)
                .expand(n_shots, -1)
                .gather(1, sources_i_masked)
            )
            * dt
        )
    # source_amplitudes_p
    if source_amplitudes_out[-1].numel() > 0:
        mask = sources_i[-1] == deepwave.common.IGNORE_LOCATION
        sources_i_masked = sources_i[-1].clone()
        sources_i_masked[mask] = 0
        source_amplitudes_out[-1] = -source_amplitudes_out[-1] * dt

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

    # Run the forward propagator
    outputs = list(
        elastic_func(
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

    # Pressure is calculated as the negative arithmetic mean of normal stresses
    if outputs[-1] is not None:
        outputs[-1] = -outputs[-1] / ndim

    # Reorder the wavefields to that expected by users
    if ndim == 3:
        outputs[: -1 - ndim] = [
            outputs[i]
            for i in [
                0,
                14,
                23,
                1,
                2,
                3,
                15,
                16,
                24,
                4,
                5,
                6,
                7,
                8,
                17,
                18,
                19,
                25,
                9,
                10,
                11,
                12,
                13,
                20,
                21,
                22,
                26,
            ]
        ]
    elif ndim == 2:
        outputs[: -1 - ndim] = [
            outputs[i] for i in [0, 9, 1, 2, 10, 3, 4, 5, 11, 6, 7, 8, 12]
        ]

    return (*outputs[: -1 - ndim], outputs[-1], *outputs[-1 - ndim : -1])


def zero_edge(tensor: torch.Tensor, fd_pad: int, dim: int) -> torch.Tensor:
    """Sets values at the end of a dimension of a tensor to zero.

    This is done because the staggered grid means that elements at
    index -fd_pad in the dimension in which a component is shifted
    by half a grid cell are not considered to be part of the
    computational domain and so should be zeroed.

    Args:
        tensor: The input torch.Tensor to modify.
        fd_pad: Half the length of the spatial finite difference stencil.
        dim: The dimension in which to zero.
    """
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
    """Zeros the edges and/or interiors of wavefields.

    This function modifies the input `wavefields` in-place.

    Args:
        wavefields: A list of wavefield tensors to modify.
        ndim: The number of spatial dimensions.
        fd_pad: The finite difference padding.
        fd_pad_list: A list of finite difference padding for each dimension.
        pml_width: A list of PML widths for each dimension.
        interior: If True, zeros the interior of the wavefields.
    """
    if ndim == 3:
        half_grid = [[False, False, False] for _ in range(len(wavefields))]
        for i in [0, 2, 3, 5, 6, 7, 8, 9, 10, 11]:
            half_grid[i][0] = True
            wavefields[i] = zero_edge(wavefields[i], fd_pad, 0)
        for i in [2, 5, 7, 12, 14, 16, 18, 19, 20, 22]:
            half_grid[i][1] = True
            wavefields[i] = zero_edge(wavefields[i], fd_pad, 1)
        for i in [3, 6, 8, 13, 16, 18, 19, 21, 23, 26]:
            half_grid[i][2] = True
            wavefields[i] = zero_edge(wavefields[i], fd_pad, 2)
        pml_edge_idx = [
            [4, 7, 8, 9, 12, 13],
            [5, 10, 17, 19, 20, 21],
            [6, 11, 18, 22, 25, 26],
        ]
        if interior:
            for dim in range(ndim):
                for i in pml_edge_idx[dim]:
                    fd_pad_shifted = list(fd_pad_list)
                    if half_grid[i][dim]:
                        fd_pad_shifted[2 * dim + 1] += 1
                    wavefields[i] = deepwave.common.zero_interior(
                        wavefields[i], fd_pad_shifted, pml_width, dim
                    )
    elif ndim == 2:
        half_grid = [[False, False] for _ in range(len(wavefields))]
        for i in [0, 2, 4, 5, 6, 8]:
            half_grid[i][0] = True
            wavefields[i] = zero_edge(wavefields[i], fd_pad, 0)
        for i in [2, 4, 5, 7, 9, 12]:
            half_grid[i][1] = True
            wavefields[i] = zero_edge(wavefields[i], fd_pad, 1)
        pml_edge_idx = [[3, 5, 6, 7], [4, 8, 11, 12]]
        if interior:
            for dim in range(ndim):
                for i in pml_edge_idx[dim]:
                    fd_pad_shifted = list(fd_pad_list)
                    if half_grid[i][dim]:
                        fd_pad_shifted[2 * dim + 1] += 1
                    wavefields[i] = deepwave.common.zero_interior(
                        wavefields[i], fd_pad_shifted, pml_width, dim
                    )
    elif ndim == 1:
        half_grid = [
            [
                False,
            ]
            for _ in range(len(wavefields))
        ]
        for i in [0, 3]:
            half_grid[i][0] = True
            wavefields[i] = zero_edge(wavefields[i], fd_pad, 0)
        pml_edge_idx = [[2, 3]]
        if interior:
            for dim in range(ndim):
                for i in pml_edge_idx[dim]:
                    fd_pad_shifted = list(fd_pad_list)
                    if half_grid[i][dim]:
                        fd_pad_shifted[2 * dim + 1] += 1
                    wavefields[i] = deepwave.common.zero_interior(
                        wavefields[i], fd_pad_shifted, pml_width, dim
                    )


class ElasticForwardFunc(torch.autograd.Function):
    """Forward propagation of the elastic wave equation."""

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
        """Performs the forward propagation of the elastic wave equation.

        This method is called by PyTorch during the forward pass. It prepares
        the input tensors, calls the appropriate C/CUDA function for wave
        propagation, and saves necessary tensors for the backward pass.

        Args:
            ctx: A context object for saving information for the backward pass.
            pml_profiles: List of PML profiles.
            grid_spacing: Grid spacing for each spatial dimension.
            dt: Time step interval.
            nt: Total number of time steps.
            step_ratio: Ratio between user dt and internal dt.
            accuracy: Finite difference accuracy order.
            pml_width: List of PML widths for each side.
            n_shots: Number of shots in the batch.
            forward_callback: The forward callback.
            backward_callback: The backward callback.
            callback_frequency: The callback frequency.
            storage_mode_str: Storage mode ("device", "cpu", "disk", "none").
            storage_path: Path for disk storage.
            storage_compression: Whether to use compression.
            args: Property models, source amplitudes, source locations (1D),
                  receiver locations (1D), and initial wavefields.

        Returns:
            A tuple containing the final wavefields, memory variables, and
            receiver amplitudes.

        """
        ndim = len(grid_spacing)
        args_list = list(args)

        # Determine number of models, sources_i, receivers_i, source_amplitudes,
        # wavefields based on ndim
        num_wavefields = 0
        num_receivers_i = 0
        num_sources_i = 0
        num_source_amplitudes = 0
        num_models = 0
        if ndim == 3:
            num_wavefields = 14 + 9 + 4  # vz_0...m_sigmaxxx_0
            num_receivers_i = 4  # z,y,x,p
            num_sources_i = 4  # z,y,x,p
            num_source_amplitudes = 4  # z,y,x,p
            num_models = (
                2 + 3 + 3
            )  # lamb,mu + mu_zy,mu_zx,mu_yx + buoyancy_z,buoyancy_y,buoyancy_x
        elif ndim == 2:
            num_wavefields = 9 + 4
            num_receivers_i = 3
            num_sources_i = 3
            num_source_amplitudes = 3
            num_models = 2 + 1 + 2  # lamb,mu + mu_yx + buoyancy_y,buoyancy_x
        else:  # ndim == 1
            num_wavefields = 4
            num_receivers_i = 2
            num_sources_i = 2
            num_source_amplitudes = 2
            num_models = 2 + 1  # lamb,mu + buoyancy_x

        wavefields = args_list[-num_wavefields:]
        del args_list[-num_wavefields:]
        receivers_i = args_list[-num_receivers_i:]
        del args_list[-num_receivers_i:]
        sources_i = args_list[-num_sources_i:]
        del args_list[-num_sources_i:]
        source_amplitudes = args_list[-num_source_amplitudes:]
        del args_list[-num_source_amplitudes:]
        models = args_list[-num_models:]
        del args_list[-num_models:]

        if len(args_list) != 0:
            raise AssertionError(
                "Error parsing arguments for ElasticForwardFunc.forward"
            )

        del args
        models = [model.contiguous() for model in models]
        source_amplitudes = [
            amplitudes.contiguous() for amplitudes in source_amplitudes
        ]
        pml_profiles = [profile.contiguous() for profile in pml_profiles]
        sources_i = [locs.contiguous() for locs in sources_i]
        receivers_i = [locs.contiguous() for locs in receivers_i]

        device = models[0].device
        dtype = models[0].dtype
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
            raise ValueError(
                "storage_mode must be 'device', 'cpu', 'disk', or 'none', "
                f"but got {storage_mode_str}"
            )

        is_cuda = models[0].is_cuda
        model_shape = models[0].shape[-ndim:]
        n_sources_per_shot = [locs.numel() // n_shots for locs in sources_i]
        n_receivers_per_shot = [locs.numel() // n_shots for locs in receivers_i]

        # Storage allocation for backward_storage variables
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

        lamb_requires_grad = models[0].requires_grad
        mu_requires_grad = models[1].requires_grad
        buoyancy_requires_grad = models[-1].requires_grad

        # Define which storage variables are relevant based on ndim
        storage_vars_info = []
        if ndim >= 3:
            storage_vars_info.append(("dvzdbuoyancy", buoyancy_requires_grad))
            storage_vars_info.append(
                ("dvzdz_store", lamb_requires_grad or mu_requires_grad)
            )
            storage_vars_info.append(("dvzdx_plus_dvxdz_store", mu_requires_grad))
            storage_vars_info.append(("dvzdy_plus_dvydz_store", mu_requires_grad))
        if ndim >= 2:
            storage_vars_info.append(("dvydbuoyancy", buoyancy_requires_grad))
            storage_vars_info.append(
                ("dvydy_store", lamb_requires_grad or mu_requires_grad)
            )
            storage_vars_info.append(("dvydx_plus_dvxdy_store", mu_requires_grad))
        storage_vars_info.append(("dvxdbuoyancy", buoyancy_requires_grad))
        storage_vars_info.append(
            ("dvxdx_store", lamb_requires_grad or mu_requires_grad)
        )

        # Create storage buffers and file arrays
        for _name, requires_grad_cond in storage_vars_info:
            storage_manager.allocate(requires_grad_cond)

        if (
            lamb_requires_grad
            or mu_requires_grad
            or buoyancy_requires_grad
            or any(amp.requires_grad for amp in source_amplitudes)
            or any(wavefield.requires_grad for wavefield in wavefields)
        ):
            ctx.save_for_backward(
                *models,
                *sources_i,
                *receivers_i,
                *source_amplitudes,
                *pml_profiles,
            )
            ctx.grid_spacing = grid_spacing
            ctx.dt = dt
            ctx.nt = nt
            ctx.n_shots = n_shots
            ctx.step_ratio = step_ratio
            ctx.accuracy = accuracy
            ctx.pml_width = pml_width
            ctx.source_amplitudes_requires_grad = [
                amp.requires_grad for amp in source_amplitudes
            ]
            ctx.backward_callback = backward_callback
            ctx.callback_frequency = callback_frequency
            ctx.storage_manager = storage_manager

        fd_pad = accuracy // 2
        fd_pad_list = [fd_pad, fd_pad - 1] * ndim
        size_with_batch = (n_shots, *models[0].shape[-ndim:])
        wavefields = [
            deepwave.common.create_or_pad(
                wavefield,
                fd_pad_list,
                device,
                dtype,
                size_with_batch,
            )
            for wavefield in wavefields
        ]

        zero_edges_and_interiors(wavefields, ndim, fd_pad, fd_pad_list, pml_width)

        pml_b = [pml_width[2 * i] + fd_pad for i in range(ndim)]
        pml_e = [
            max(pml_b[i], model_shape[i] - pml_width[2 * i + 1] - (fd_pad - 1))
            for i in range(ndim)
        ]

        lamb_batched = models[0].ndim == ndim + 1 and models[0].shape[0] > 1
        mu_batched = models[1].ndim == ndim + 1 and models[1].shape[0] > 1
        buoyancy_batched = models[-1].ndim == ndim + 1 and models[-1].shape[0] > 1

        receiver_amplitudes: List[torch.Tensor] = [
            torch.empty(0, device=device, dtype=dtype) for _ in range(ndim + 1)
        ]

        for i, loc in enumerate(receivers_i):
            if loc.numel() > 0:
                receiver_amplitudes[i].resize_(nt, n_shots, n_receivers_per_shot[i])
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
            "elastic",
            ndim,
            "forward",
            accuracy,
            dtype,
            device,
        )

        rdx = [1 / dx for dx in grid_spacing]

        if forward_callback is None:
            callback_frequency = nt // step_ratio

        wavefield_names = []
        model_names = []
        if ndim >= 3:
            wavefield_names.extend(
                [
                    "vz_0",
                    "sigmazz_0",
                    "sigmayz_0",
                    "sigmaxz_0",
                    "m_vzz_0",
                    "m_vzy_0",
                    "m_vzx_0",
                    "m_vyz_0",
                    "m_vxz_0",
                    "m_sigmazzz_0",
                    "m_sigmayzy_0",
                    "m_sigmaxzx_0",
                    "m_sigmayzz_0",
                    "m_sigmaxzz_0",
                ]
            )
            model_names = [
                "lamb",
                "mu",
                "mu_zy",
                "mu_zx",
                "mu_yx",
                "buoyancy_z",
                "buoyancy_y",
                "buoyancy_x",
            ]
        if ndim >= 2:
            wavefield_names.extend(
                [
                    "vy_0",
                    "sigmayy_0",
                    "sigmaxy_0",
                    "m_vyy_0",
                    "m_vyx_0",
                    "m_vxy_0",
                    "m_sigmayyy_0",
                    "m_sigmaxyy_0",
                    "m_sigmaxyx_0",
                ]
            )
            model_names = ["lamb", "mu", "mu_yx", "buoyancy_y", "buoyancy_x"]
        if ndim == 1:
            model_names = ["lamb", "mu", "buoyancy_x"]
        wavefield_names.extend(
            [
                "vx_0",
                "sigmaxx_0",
                "m_vxx_0",
                "m_sigmaxxx_0",
            ]
        )
        callback_models = dict(zip(model_names, models))
        callback_wavefields = {}

        if wavefields[0].numel() > 0 and nt > 0:
            for step in range(0, nt // step_ratio, callback_frequency):
                if forward_callback is not None:
                    callback_wavefields = dict(zip(wavefield_names, wavefields))
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
                        *[model.data_ptr() for model in models],
                        *[amp.data_ptr() for amp in source_amplitudes],
                        *[field.data_ptr() for field in wavefields],
                        *storage_manager.storage_ptrs,
                        *[amp.data_ptr() for amp in receiver_amplitudes],
                        *[profile.data_ptr() for profile in pml_profiles],
                        *[locs.data_ptr() for locs in sources_i],
                        *[locs.data_ptr() for locs in receivers_i],
                        *rdx,
                        dt,
                        step_nt * step_ratio,
                        n_shots,
                        *model_shape,
                        *n_sources_per_shot,
                        *n_receivers_per_shot,
                        step_ratio,
                        storage_manager.storage_mode,
                        storage_manager.shot_bytes_uncomp,
                        storage_manager.shot_bytes_comp,
                        lamb_requires_grad
                        and storage_manager.storage_mode
                        != deepwave.common.StorageMode.NONE,
                        mu_requires_grad
                        and storage_manager.storage_mode
                        != deepwave.common.StorageMode.NONE,
                        buoyancy_requires_grad
                        and storage_manager.storage_mode
                        != deepwave.common.StorageMode.NONE,
                        lamb_batched,
                        mu_batched,
                        buoyancy_batched,
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

        s = (
            slice(None),
            *(slice(fd_pad, shape - (fd_pad - 1)) for shape in model_shape),
        )
        return (
            *[field[s] for field in wavefields],
            *receiver_amplitudes,
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: Any,
        *args: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Computes the gradients during the backward pass.

        This method is called by PyTorch during the backward pass to compute
        gradients with respect to the inputs of the forward pass.

        Args:
            ctx: A context object with saved information from the forward pass.
            args: Gradients of the outputs of the forward pass.

        Returns:
            A tuple containing the gradients with respect to the inputs of the
            forward pass.

        """
        grid_spacing = ctx.grid_spacing
        ndim = len(grid_spacing)
        grad_r = list(args[-ndim - 1 :])
        grad_wavefields = list(args[: -ndim - 1])
        del args

        saved_tensors = list(ctx.saved_tensors)

        # Original saved variables counts:
        # models: 8 (3D), 5 (2D), 3 (1D)
        # sources_i: 4 (3D), 3 (2D), 2 (1D)
        # receivers_i: 4 (3D), 3 (2D), 2 (1D)
        # source_amplitudes: 4 (3D), 3 (2D), 2 (1D)
        # pml_profiles: 12 (3D), 8 (2D), 4 (1D)

        num_models = 0
        num_sources_i = 0
        num_receivers_i = 0
        num_source_amplitudes = 0
        num_pml_profiles = 0

        if ndim == 3:
            num_models = 8
            num_sources_i = 4
            num_receivers_i = 4
            num_source_amplitudes = 4
            num_pml_profiles = 12
        elif ndim == 2:
            num_models = 5
            num_sources_i = 3
            num_receivers_i = 3
            num_source_amplitudes = 3
            num_pml_profiles = 8
        else:
            num_models = 3
            num_sources_i = 2
            num_receivers_i = 2
            num_source_amplitudes = 2
            num_pml_profiles = 4

        models = saved_tensors[:num_models]
        del saved_tensors[:num_models]
        sources_i = saved_tensors[:num_sources_i]
        del saved_tensors[:num_sources_i]
        receivers_i = saved_tensors[:num_receivers_i]
        del saved_tensors[:num_receivers_i]
        source_amplitudes = saved_tensors[:num_source_amplitudes]
        del saved_tensors[:num_source_amplitudes]
        pml_profiles = saved_tensors[:num_pml_profiles]
        del saved_tensors[:num_pml_profiles]

        lamb_requires_grad = models[0].requires_grad
        mu_requires_grad = models[1].requires_grad
        buoyancy_requires_grad = models[-1].requires_grad

        grad_r = [grad.contiguous() for grad in grad_r]
        models = [model.contiguous() for model in models]
        sources_i = [loc.contiguous() for loc in sources_i]
        receivers_i = [loc.contiguous() for loc in receivers_i]
        source_amplitudes = [amp.contiguous() for amp in source_amplitudes]
        pml_profiles = [profile.contiguous() for profile in pml_profiles]

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
        n_sources_per_shot = [loc.numel() // n_shots for loc in sources_i]
        n_receivers_per_shot = [loc.numel() // n_shots for loc in receivers_i]
        grad_models = [torch.empty(0, device=device, dtype=dtype) for _ in models]
        grad_models_tmp = [torch.empty(0, device=device, dtype=dtype) for _ in models]
        grad_models_tmp_ptr = [grad_model.data_ptr() for grad_model in grad_models]
        for i, model in enumerate(models):
            if model.requires_grad:
                grad_models[i].resize_(*model.shape)
                grad_models[i].fill_(0)
                grad_models_tmp_ptr[i] = grad_models[i].data_ptr()
        grad_f = [torch.empty(0, device=device, dtype=dtype) for _ in sources_i]

        lamb_batched = models[0].ndim == ndim + 1 and models[0].shape[0] > 1
        mu_batched = models[1].ndim == ndim + 1 and models[1].shape[0] > 1
        buoyancy_batched = models[-1].ndim == ndim + 1 and models[-1].shape[0] > 1

        lamb_requires_grad = models[0].requires_grad
        mu_requires_grad = models[1].requires_grad
        buoyancy_requires_grad = models[-1].requires_grad

        fd_pad = accuracy // 2
        fd_pad_list = [fd_pad, fd_pad - 1] * ndim
        size_with_batch = (n_shots, *model_shape)
        grad_wavefields = [
            deepwave.common.create_or_pad(
                wavefield,
                fd_pad_list,
                device,
                dtype,
                size_with_batch,
            )
            for wavefield in grad_wavefields
        ]
        aux_wavefields = []
        if ndim >= 3:
            aux_wavefields.extend(
                [torch.zeros_like(grad_wavefields[0]) for _ in range(5)]
            )
        if ndim >= 2:
            aux_wavefields.extend(
                [torch.zeros_like(grad_wavefields[0]) for _ in range(3)]
            )
        aux_wavefields.append(torch.zeros_like(grad_wavefields[0]))

        zero_edges_and_interiors(grad_wavefields, ndim, fd_pad, fd_pad_list, pml_width)

        pml_b = [
            min(pml_width[2 * i] + 2 * fd_pad, model_shape[i] - (fd_pad - 1))
            for i in range(ndim)
        ]
        pml_e = [
            max(pml_b[i], model_shape[i] - pml_width[2 * i + 1] - 2 * fd_pad + 1)
            for i in range(ndim)
        ]

        for i, requires_grad in enumerate(source_amplitudes_requires_grad):
            if requires_grad:
                grad_f[i].resize_(nt, n_shots, n_sources_per_shot[i])
                grad_f[i].fill_(0)

        stream: Union[int, torch.Stream] = 0
        if is_cuda:
            aux = models[0].get_device()
            stream = torch.cuda.current_stream(aux)
            for i, model in enumerate(models):
                batched = model.ndim == ndim + 1 and model.shape[0] > 1
                if (
                    model.requires_grad
                    and not batched
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
                batched = model.ndim == ndim + 1 and model.shape[0] > 1
                if (
                    model.requires_grad
                    and not batched
                    and aux > 1
                    and deepwave.backend_utils.USE_OPENMP
                    and storage_manager.storage_mode != deepwave.common.StorageMode.NONE
                ):
                    grad_models_tmp[i].resize_(n_shots, *model_shape)
                    grad_models_tmp[i].fill_(0)
                    grad_models_tmp_ptr[i] = grad_models_tmp[i].data_ptr()
        backward = deepwave.backend_utils.get_backend_function(
            "elastic",
            ndim,
            "backward",
            accuracy,
            dtype,
            device,
        )

        rdx = [1 / dx for dx in grid_spacing]

        if backward_callback is None:
            callback_frequency = nt // step_ratio

        wavefield_names = []
        model_names = []
        if ndim >= 3:
            wavefield_names.extend(
                [
                    "vz_0",
                    "sigmazz_0",
                    "sigmayz_0",
                    "sigmaxz_0",
                    "m_vzz_0",
                    "m_vzy_0",
                    "m_vzx_0",
                    "m_vyz_0",
                    "m_vxz_0",
                    "m_sigmazzz_0",
                    "m_sigmayzy_0",
                    "m_sigmaxzx_0",
                    "m_sigmayzz_0",
                    "m_sigmaxzz_0",
                ]
            )
            model_names = [
                "lamb",
                "mu",
                "mu_zy",
                "mu_zx",
                "mu_yx",
                "buoyancy_z",
                "buoyancy_y",
                "buoyancy_x",
            ]
        if ndim >= 2:
            wavefield_names.extend(
                [
                    "vy_0",
                    "sigmayy_0",
                    "sigmaxy_0",
                    "m_vyy_0",
                    "m_vyx_0",
                    "m_vxy_0",
                    "m_sigmayyy_0",
                    "m_sigmaxyy_0",
                    "m_sigmaxyx_0",
                ]
            )
            model_names = ["lamb", "mu", "mu_yx", "buoyancy_y", "buoyancy_x"]
        if ndim == 1:
            model_names = ["lamb", "mu", "buoyancy_x"]
        wavefield_names.extend(
            [
                "vx_0",
                "sigmaxx_0",
                "m_vxx_0",
                "m_sigmaxxx_0",
            ]
        )
        callback_models = dict(zip(model_names, models))
        callback_grad_models = dict(zip(model_names, grad_models))

        if grad_wavefields[0].numel() > 0 and nt > 0:
            for step in range(nt // step_ratio, 0, -callback_frequency):
                step_nt = min(step, callback_frequency)
                if (
                    backward(
                        *[model.data_ptr() for model in models],
                        *[amp.data_ptr() for amp in grad_r],
                        *[field.data_ptr() for field in grad_wavefields],
                        *[field.data_ptr() for field in aux_wavefields],
                        *storage_manager.storage_ptrs,
                        *[amp.data_ptr() for amp in grad_f],
                        *[model.data_ptr() for model in grad_models],
                        *grad_models_tmp_ptr,
                        *[profile.data_ptr() for profile in pml_profiles],
                        *[locs.data_ptr() for locs in sources_i],
                        *[locs.data_ptr() for locs in receivers_i],
                        *rdx,
                        dt,
                        step_nt * step_ratio,
                        n_shots,
                        *model_shape,
                        *[
                            n_sources_per_shot[i] * source_amplitudes_requires_grad[i]
                            for i in range(len(n_sources_per_shot))
                        ],
                        *n_receivers_per_shot,
                        step_ratio,
                        storage_manager.storage_mode,
                        storage_manager.shot_bytes_uncomp,
                        storage_manager.shot_bytes_comp,
                        lamb_requires_grad
                        and storage_manager.storage_mode
                        != deepwave.common.StorageMode.NONE,
                        mu_requires_grad
                        and storage_manager.storage_mode
                        != deepwave.common.StorageMode.NONE,
                        buoyancy_requires_grad
                        and storage_manager.storage_mode
                        != deepwave.common.StorageMode.NONE,
                        lamb_batched,
                        mu_batched,
                        buoyancy_batched,
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
                    if ndim == 3:
                        (
                            grad_wavefields[9:14],
                            aux_wavefields[-9:-4],
                        ) = (
                            aux_wavefields[-9:-4],
                            grad_wavefields[9:14],
                        )
                        (
                            grad_wavefields[20:23],
                            aux_wavefields[-4:-1],
                        ) = (
                            aux_wavefields[-4:-1],
                            grad_wavefields[20:23],
                        )
                        grad_wavefields[26], aux_wavefields[-1] = (
                            aux_wavefields[-1],
                            grad_wavefields[26],
                        )
                    elif ndim == 2:
                        (
                            grad_wavefields[6:9],
                            aux_wavefields[-4:-1],
                        ) = (
                            aux_wavefields[-4:-1],
                            grad_wavefields[6:9],
                        )
                        grad_wavefields[12], aux_wavefields[-1] = (
                            aux_wavefields[-1],
                            grad_wavefields[12],
                        )
                    else:
                        grad_wavefields[3], aux_wavefields[-1] = (
                            aux_wavefields[-1],
                            grad_wavefields[3],
                        )

                if backward_callback is not None:
                    callback_wavefields = dict(
                        zip(wavefield_names, grad_wavefields[:-1])
                    )
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
            [
                None,
            ]
            * 14
            + grad_models
            + grad_f
            + [None] * 2 * (ndim + 1)
            + [field[s] for field in grad_wavefields]
        )


def update_velocities(
    models: List[torch.Tensor],
    normal_stresses: List[torch.Tensor],
    shear_stresses: List[List[torch.Tensor]],
    velocities: List[torch.Tensor],
    s_mem_vars_normal: List[torch.Tensor],
    s_mem_vars_shear: List[List[torch.Tensor]],
    pml_profiles: List[torch.Tensor],
    grid_spacing: List[float],
    dt: float,
    accuracy: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]]]:
    """Updates the velocity wavefields and PML memory variables."""
    ndim = len(grid_spacing)

    # Extract PML profiles
    a = pml_profiles[::4]
    b = pml_profiles[1::4]
    ah = pml_profiles[2::4]
    bh = pml_profiles[3::4]
    del pml_profiles

    # Calculate derivatives of normal stresses
    dsigmaxxdx = [
        deepwave.staggered_grid.diff1h(
            normal_stresses[dim],
            dim,
            accuracy,
            1 / grid_spacing[dim],
            ndim,
        )
        for dim in range(ndim)
    ]

    # Update memory variables for siii
    for dim in range(ndim):
        s_mem_vars_normal[dim] = (
            ah[dim] * s_mem_vars_normal[dim] + bh[dim] * dsigmaxxdx[dim]
        )
        dsigmaxxdx[dim] = dsigmaxxdx[dim] + s_mem_vars_normal[dim]

    sigma_sum = list(dsigmaxxdx)

    # Calculate derivatives of shear stresses
    for dim1 in range(ndim - 1):
        for dim2 in range(dim1 + 1, ndim):
            d_shear_d_dim2 = deepwave.staggered_grid.diff1(
                shear_stresses[dim1][dim2],
                dim2,
                accuracy,
                1 / grid_spacing[dim2],
                ndim,
            )
            mem = s_mem_vars_shear[dim1][dim2]
            new_mem = a[dim2] * mem + b[dim2] * d_shear_d_dim2
            s_mem_vars_shear[dim1][dim2] = new_mem
            d_shear_d_dim2 += new_mem
            sigma_sum[dim1] += d_shear_d_dim2

            d_shear_d_dim1 = deepwave.staggered_grid.diff1(
                shear_stresses[dim1][dim2],
                dim1,
                accuracy,
                1 / grid_spacing[dim1],
                ndim,
            )
            mem = s_mem_vars_shear[dim2][dim1]
            new_mem = a[dim1] * mem + b[dim1] * d_shear_d_dim1
            s_mem_vars_shear[dim2][dim1] = new_mem
            d_shear_d_dim1 += new_mem
            sigma_sum[dim2] += d_shear_d_dim1

    buoyancy = models[-ndim:]

    velocities = [
        velocities[dim] + buoyancy[dim] * dt * sigma_sum[dim] for dim in range(ndim)
    ]

    return (
        velocities,
        s_mem_vars_normal,
        s_mem_vars_shear,
    )


def update_stresses(
    models: List[torch.Tensor],
    velocities: List[torch.Tensor],
    normal_stresses: List[torch.Tensor],
    shear_stresses: List[List[torch.Tensor]],
    v_mem_vars_normal: List[torch.Tensor],
    v_mem_vars_shear: List[List[torch.Tensor]],
    pml_profiles: List[torch.Tensor],
    grid_spacing: List[float],
    dt: float,
    accuracy: int,
) -> Tuple[
    List[torch.Tensor],
    List[List[torch.Tensor]],
    List[torch.Tensor],
    List[List[torch.Tensor]],
]:
    """Updates the stress wavefields and PML memory variables."""
    ndim = len(grid_spacing)

    # Calculate derivatives of velocity in dimension i, in dimension i (vi_i)
    dvxdx = [
        deepwave.staggered_grid.diff1(
            velocities[dim], dim, accuracy, 1 / grid_spacing[dim], ndim
        )
        for dim in range(ndim)
    ]

    # Calculate derivatives of velocity in dimension i, in dimension j (vi_j)
    dvxdy: List[List[torch.Tensor]] = [
        [torch.empty(0) for _ in range(ndim)] for _ in range(ndim)
    ]
    for dim1 in range(ndim):
        for dim2 in range(ndim):
            if dim1 == dim2:
                continue
            dvxdy[dim1][dim2] = deepwave.staggered_grid.diff1h(
                velocities[dim1],
                dim2,
                accuracy,
                1 / grid_spacing[dim2],
                ndim,
            )

    # Extract PML profiles
    a = pml_profiles[::4]
    b = pml_profiles[1::4]
    ah = pml_profiles[2::4]
    bh = pml_profiles[3::4]
    del pml_profiles

    # PML correct all derivatives
    for dim in range(ndim):
        v_mem_vars_normal[dim] = a[dim] * v_mem_vars_normal[dim] + b[dim] * dvxdx[dim]
        dvxdx[dim] = dvxdx[dim] + v_mem_vars_normal[dim]

    for dim1 in range(ndim):
        for dim2 in range(ndim):
            if dim1 == dim2:
                continue
            mem = v_mem_vars_shear[dim1][dim2]
            new_mem = ah[dim2] * mem + bh[dim2] * dvxdy[dim1][dim2]
            v_mem_vars_shear[dim1][dim2] = new_mem
            dvxdy[dim1][dim2] = dvxdy[dim1][dim2] + new_mem

    lamb = models[0]
    mu = models[1]

    # Update normal stresses
    v_strain_sum = torch.sum(torch.stack(dvxdx), dim=0)
    normal_stresses = [
        normal_stresses[dim] + dt * (lamb * v_strain_sum + 2 * mu * dvxdx[dim])
        for dim in range(ndim)
    ]

    # Update shear stresses
    if ndim > 1:
        if ndim == 3:
            mu_zy, mu_zx, mu_yx = models[2], models[3], models[4]
            shear_models = [
                [torch.empty(0), mu_zy, mu_zx],
                [mu_zy, torch.empty(0), mu_yx],
                [mu_zx, mu_yx, torch.empty(0)],
            ]
        else:  # ndim == 2
            mu_yx = models[2]
            shear_models = [[torch.empty(0), mu_yx], [mu_yx, torch.empty(0)]]

        for dim1 in range(ndim):
            for dim2 in range(dim1 + 1, ndim):
                shear_stresses[dim1][dim2] = shear_stresses[dim1][
                    dim2
                ] + dt * shear_models[dim1][dim2] * (
                    dvxdy[dim1][dim2] + dvxdy[dim2][dim1]
                )
                shear_stresses[dim2][dim1] = shear_stresses[dim1][dim2]

    return (
        normal_stresses,
        shear_stresses,
        v_mem_vars_normal,
        v_mem_vars_shear,
    )


def elastic_python(
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
    """Performs the forward propagation of the elastic wave equation."""
    is_batched = any(deepwave.common.is_inside_vmap(x) for x in args)
    if backward_callback is not None:
        raise RuntimeError("backward_callback is not supported in the Python backend.")
    if storage_mode_str != "device":
        raise RuntimeError(
            "Specifying the storage mode is not supported in the Python backend."
        )
    if storage_compression:
        raise RuntimeError(
            "Storage compression is not supported in the Python backend."
        )
    ndim = len(grid_spacing)
    args_list = list(args)
    del args
    if ndim == 3:
        wavefields_flat = args_list[-27:]
        del args_list[-27:]
    elif ndim == 2:
        wavefields_flat = args_list[-13:]
        del args_list[-13:]
    else:
        wavefields_flat = args_list[-4:]
        del args_list[-4:]

    device = args_list[0].device
    dtype = args_list[0].dtype
    model_shape = args_list[0].shape[-ndim:]
    flat_model_shape = int(torch.prod(torch.tensor(model_shape)).item())
    fd_pad = accuracy // 2
    fd_pad_list = [fd_pad, fd_pad - 1] * ndim
    size_with_batch = (n_shots, *model_shape)
    wavefields_flat = [
        deepwave.common.create_or_pad(
            wavefield,
            fd_pad_list,
            device,
            dtype,
            size_with_batch,
        )
        for wavefield in wavefields_flat
    ]

    zero_edges_and_interiors(
        wavefields_flat, ndim, fd_pad, fd_pad_list, pml_width, False
    )

    if ndim == 3:
        v_z, v_y, v_x = wavefields_flat[0], wavefields_flat[14], wavefields_flat[23]
        s_zz, s_yy, s_xx = wavefields_flat[1], wavefields_flat[15], wavefields_flat[24]
        s_yz, s_xz, s_xy = wavefields_flat[2], wavefields_flat[3], wavefields_flat[16]
        m_v_zz, m_v_yy, m_v_xx = (
            wavefields_flat[4],
            wavefields_flat[17],
            wavefields_flat[25],
        )
        m_v_zy, m_v_zx = wavefields_flat[5], wavefields_flat[6]
        m_v_yz, m_v_xz = wavefields_flat[7], wavefields_flat[8]
        m_v_yx, m_v_xy = wavefields_flat[18], wavefields_flat[19]
        m_s_zzz, m_s_yyy, m_s_xxx = (
            wavefields_flat[9],
            wavefields_flat[20],
            wavefields_flat[26],
        )
        m_s_yzy, m_s_xzx = wavefields_flat[10], wavefields_flat[11]
        m_s_yzz, m_s_xzz = wavefields_flat[12], wavefields_flat[13]
        m_s_xyy, m_s_xyx = wavefields_flat[21], wavefields_flat[22]

        velocities = [v_z, v_y, v_x]
        normal_stresses = [s_zz, s_yy, s_xx]
        # shear_stresses[i][j] = sigma_ij, where i and j are spatial dimension
        # indices. As sigma_ij = sigma_ji, this is symmetric.
        # For 3D (z=0, y=1, x=2):
        # > shear_stresses[0][1] = shear_stresses[1][0] = s_yz
        # > shear_stresses[0][2] = shear_stresses[2][0] = s_xz
        # > shear_stresses[1][2] = shear_stresses[2][1] = s_xy
        shear_stresses = [
            [torch.empty(0), s_yz, s_xz],
            [s_yz, torch.empty(0), s_xy],
            [s_xz, s_xy, torch.empty(0)],
        ]
        v_mem_vars_normal = [m_v_zz, m_v_yy, m_v_xx]
        v_mem_vars_shear = [
            [torch.empty(0), m_v_zy, m_v_zx],
            [m_v_yz, torch.empty(0), m_v_yx],
            [m_v_xz, m_v_xy, torch.empty(0)],
        ]
        s_mem_vars_normal = [m_s_zzz, m_s_yyy, m_s_xxx]
        # s_mem_vars_shear[i][j] is the memory variable for the shear stress
        # derivative that contributes to the v_i update, where the derivative
        # is with respect to dimension j.
        # e.g. s_mem_vars_shear[0][1] is for d(s_yz)/dy, which contributes to
        # v_z. The corresponding variable is m_s_yzy.
        s_mem_vars_shear = [
            [torch.empty(0), m_s_yzy, m_s_xzx],
            [m_s_yzz, torch.empty(0), m_s_xyx],
            [m_s_xzz, m_s_xyy, torch.empty(0)],
        ]

        del (
            v_z,
            v_y,
            v_x,
            s_zz,
            s_yy,
            s_xx,
            s_yz,
            s_xz,
            s_xy,
            m_v_zz,
            m_v_yy,
            m_v_xx,
            m_v_zy,
            m_v_zx,
            m_v_yz,
            m_v_xz,
            m_v_yx,
            m_v_xy,
            m_s_zzz,
            m_s_yyy,
            m_s_xxx,
            m_s_yzy,
            m_s_xzx,
            m_s_yzz,
            m_s_xzz,
            m_s_xyy,
            m_s_xyx,
        )

        v_names = ["vz_0", "vy_0", "vx_0"]
        normal_s_names = ["sigmazz_0", "sigmayy_0", "sigmaxx_0"]
        shear_s_names = [
            ["sigmayz_0", "sigmaxz_0"],
            [
                "sigmaxy_0",
            ],
        ]
        v_mem_normal_names = ["m_vzz_0", "m_vyy_0", "m_vxx_0"]
        v_mem_shear_names = [
            ["m_vzy_0", "m_vzx_0"],
            ["m_vyz_0", "m_vyx_0"],
            ["m_vxz_0", "m_vxy_0"],
        ]
        s_mem_normal_names = ["m_sigmazzz_0", "m_sigmayyy_0", "m_sigmaxxx_0"]
        s_mem_shear_names = [
            ["m_sigmayzy_0", "m_sigmaxzx_0"],
            ["m_sigmayzz_0", "m_sigmaxyx_0"],
            ["m_sigmaxzz_0", "m_sigmaxyy_0"],
        ]

    elif ndim == 2:
        v_y, v_x = wavefields_flat[0], wavefields_flat[9]
        s_yy, s_xx = wavefields_flat[1], wavefields_flat[10]
        s_xy = wavefields_flat[2]
        m_v_yy, m_v_xx = wavefields_flat[3], wavefields_flat[11]
        m_v_yx, m_v_xy = wavefields_flat[4], wavefields_flat[5]
        m_s_yyy, m_s_xxx = wavefields_flat[6], wavefields_flat[12]
        m_s_xyy, m_s_xyx = wavefields_flat[7], wavefields_flat[8]

        velocities = [v_y, v_x]
        normal_stresses = [s_yy, s_xx]
        # shear_stresses[i][j] = sigma_ij, where i and j are spatial dimension
        # indices (y=0, x=1). As sigma_ij = sigma_ji, this is symmetric.
        shear_stresses = [[torch.empty(0), s_xy], [s_xy, torch.empty(0)]]
        v_mem_vars_normal = [m_v_yy, m_v_xx]
        # v_mem_vars_shear[i][j] is the memory variable for d(v_i)/dj.
        # e.g. v_mem_vars_shear[0][1] = m_v_yx for d(v_y)/dx.
        v_mem_vars_shear = [[torch.empty(0), m_v_yx], [m_v_xy, torch.empty(0)]]
        s_mem_vars_normal = [m_s_yyy, m_s_xxx]
        # s_mem_vars_shear[i][j] is the memory variable for the shear stress
        # derivative that contributes to the v_i update, where the derivative
        # is with respect to dimension j.
        # e.g. s_mem_vars_shear[0][1] is for d(s_xy)/dx, which contributes to
        # v_y. The corresponding variable is m_s_xyx.
        s_mem_vars_shear = [[torch.empty(0), m_s_xyx], [m_s_xyy, torch.empty(0)]]

        del (
            v_y,
            v_x,
            s_yy,
            s_xx,
            s_xy,
            m_v_yy,
            m_v_xx,
            m_v_yx,
            m_v_xy,
            m_s_yyy,
            m_s_xxx,
            m_s_xyy,
            m_s_xyx,
        )

        v_names = ["vy_0", "vx_0"]
        normal_s_names = ["sigmayy_0", "sigmaxx_0"]
        shear_s_names = [
            [
                "sigmaxy_0",
            ],
        ]
        v_mem_normal_names = ["m_vyy_0", "m_vxx_0"]
        v_mem_shear_names = [
            ["m_vyx_0", "m_vxy_0"],
        ]
        s_mem_normal_names = ["m_sigmayyy_0", "m_sigmaxxx_0"]
        s_mem_shear_names = [
            ["m_sigmaxyx_0", "m_sigmaxyy_0"],
        ]

    else:
        v_x = wavefields_flat[0]
        s_xx = wavefields_flat[1]
        m_v_xx = wavefields_flat[2]
        m_s_xxx = wavefields_flat[3]

        velocities = [
            v_x,
        ]
        normal_stresses = [
            s_xx,
        ]
        shear_stresses = [
            [],
        ]
        v_mem_vars_normal = [
            m_v_xx,
        ]
        v_mem_vars_shear = [
            [],
        ]
        s_mem_vars_normal = [
            m_s_xxx,
        ]
        s_mem_vars_shear = [
            [],
        ]
        del v_x, s_xx, m_v_xx, m_s_xxx

        v_names = [
            "vx_0",
        ]
        normal_s_names = [
            "sigmaxx_0",
        ]
        shear_s_names = [
            [],
        ]
        v_mem_normal_names = [
            "m_vxx_0",
        ]
        s_mem_normal_names = [
            "m_sigmaxxx_0",
        ]
        v_mem_shear_names = [
            [],
        ]
        s_mem_shear_names = [
            [],
        ]
    del wavefields_flat

    if ndim == 3:
        receivers_i = args_list[-4:]
        del args_list[-4:]
        sources_i = args_list[-4:]
        del args_list[-4:]
        source_amplitudes = args_list[-4:]
        del args_list[-4:]
        models = args_list[-8:]
        del args_list[-8:]
    elif ndim == 2:
        receivers_i = args_list[-3:]
        del args_list[-3:]
        sources_i = args_list[-3:]
        del args_list[-3:]
        source_amplitudes = args_list[-3:]
        del args_list[-3:]
        models = args_list[-5:]
        del args_list[-5:]
    else:
        receivers_i = args_list[-2:]
        del args_list[-2:]
        sources_i = args_list[-2:]
        del args_list[-2:]
        source_amplitudes = args_list[-2:]
        del args_list[-2:]
        models = args_list[-3:]
        del args_list[-3:]
    del args_list

    n_receivers_per_shot = [locs.numel() // n_shots for locs in receivers_i]

    receiver_amplitudes: List[torch.Tensor] = [
        torch.empty(0, device=device, dtype=dtype) for _ in range(ndim + 1)
    ]
    receiver_amplitudes_lists: List[List[torch.Tensor]] = [[] for _ in range(ndim + 1)]

    for i, loc in enumerate(receivers_i):
        if loc.numel() > 0 and not is_batched:
            receiver_amplitudes[i] = torch.zeros(
                nt, n_shots, n_receivers_per_shot[i], device=device, dtype=dtype
            )

    sources_i_masked = []
    source_amplitudes_masked = []
    for i, loc in enumerate(sources_i):
        source_mask = loc != deepwave.common.IGNORE_LOCATION
        sources_i_masked.append(torch.zeros_like(loc))
        sources_i_masked[-1][source_mask] = loc[source_mask]
        source_amplitudes_masked.append(torch.zeros_like(source_amplitudes[i]))
        if source_amplitudes[i].numel() > 0:
            source_amplitudes_masked[-1][:, source_mask] = source_amplitudes[i][
                :, source_mask
            ]
    del sources_i, source_amplitudes

    receivers_mask = []
    receivers_i_masked = []
    for loc in receivers_i:
        receivers_mask.append(loc != deepwave.common.IGNORE_LOCATION)
        receivers_i_masked.append(torch.zeros_like(loc))
        if loc.numel() > 0:
            receivers_i_masked[-1][receivers_mask[-1]] = loc[receivers_mask[-1]]
    del receivers_i

    model_names = []
    if ndim >= 3:
        model_names.extend(
            [
                "lamb",
                "mu",
                "mu_zy",
                "mu_zx",
                "mu_yx",
                "buoyancy_z",
                "buoyancy_y",
                "buoyancy_x",
            ]
        )
    elif ndim >= 2:
        model_names.extend(["lamb", "mu", "mu_yx", "buoyancy_y", "buoyancy_x"])
    else:
        model_names.extend(["lamb", "mu", "buoyancy_x"])
    callback_models = dict(zip(model_names, models))

    for step in range(nt // step_ratio):
        if forward_callback is not None and step % callback_frequency == 0:
            callback_wavefields = dict(zip(v_names, velocities))
            callback_wavefields.update(dict(zip(normal_s_names, normal_stresses)))
            for i in range(len(shear_s_names)):
                callback_wavefields.update(
                    dict(zip(shear_s_names[i], shear_stresses[i]))
                )
            callback_wavefields.update(dict(zip(v_mem_normal_names, v_mem_vars_normal)))
            for i in range(len(v_mem_shear_names)):
                callback_wavefields.update(
                    dict(zip(v_mem_shear_names[i], v_mem_vars_shear[i]))
                )
            callback_wavefields.update(dict(zip(s_mem_normal_names, s_mem_vars_normal)))
            for i in range(len(s_mem_shear_names)):
                callback_wavefields.update(
                    dict(zip(s_mem_shear_names[i], s_mem_vars_shear[i]))
                )
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

            for ridx in range(ndim):
                if receivers_i_masked[ridx].numel() > 0:
                    val_vel = (
                        velocities[ridx]
                        .reshape(-1, flat_model_shape)
                        .gather(1, receivers_i_masked[ridx])
                    )
                    if is_batched:
                        receiver_amplitudes_lists[ridx].append(val_vel)
                    else:
                        receiver_amplitudes[ridx][t] = val_vel
            if receivers_i_masked[-1].numel() > 0:  # pressure receiver
                val_stress = torch.stack(
                    [
                        normal_stresses[idx]
                        .reshape(-1, flat_model_shape)
                        .gather(1, receivers_i_masked[-1])
                        for idx in range(ndim)
                    ]
                ).sum(dim=0)
                if is_batched:
                    receiver_amplitudes_lists[-1].append(val_stress)
                else:
                    receiver_amplitudes[-1][t] = val_stress

            velocities, s_mem_vars_normal, s_mem_vars_shear = _update_velocities_opt(
                models,
                normal_stresses,
                shear_stresses,
                velocities,
                s_mem_vars_normal,
                s_mem_vars_shear,
                pml_profiles,
                grid_spacing,
                dt,
                accuracy,
            )

            new_velocities_list = list(velocities)
            for sidx in range(ndim):
                if source_amplitudes_masked[sidx].numel() > 0:
                    new_velocities_list[sidx] = (
                        new_velocities_list[sidx]
                        .clone()
                        .reshape(-1, flat_model_shape)
                        .scatter_add_(
                            1,
                            sources_i_masked[sidx],
                            source_amplitudes_masked[sidx][t],
                        )
                        .reshape(size_with_batch)
                    )
            velocities = list(new_velocities_list)
            del new_velocities_list

            (
                normal_stresses,
                shear_stresses,
                v_mem_vars_normal,
                v_mem_vars_shear,
            ) = _update_stresses_opt(
                models,
                velocities,
                normal_stresses,
                shear_stresses,
                v_mem_vars_normal,
                v_mem_vars_shear,
                pml_profiles,
                grid_spacing,
                dt,
                accuracy,
            )

            if source_amplitudes_masked[-1].numel() > 0:
                new_normal_stresses = list(normal_stresses)
                for idx in range(ndim):
                    new_normal_stresses[idx] = (
                        new_normal_stresses[idx]
                        .clone()
                        .reshape(-1, flat_model_shape)
                        .scatter_add_(
                            1,
                            sources_i_masked[-1],
                            source_amplitudes_masked[-1][t],
                        )
                        .reshape(size_with_batch)
                    )
                normal_stresses = list(new_normal_stresses)
                del new_normal_stresses

    if is_batched:
        for i in range(len(receiver_amplitudes)):
            if len(receiver_amplitudes_lists[i]) > 0:
                receiver_amplitudes[i] = torch.stack(receiver_amplitudes_lists[i])

    receiver_amplitudes_masked = []
    for i, amp in enumerate(receiver_amplitudes):
        receiver_amplitudes_masked.append(torch.zeros_like(amp))
        if amp.numel() > 0:
            receiver_amplitudes_masked[-1][:, receivers_mask[i]] = amp[
                :, receivers_mask[i]
            ]

    s = (
        slice(None),
        *(slice(fd_pad, shape - (fd_pad - 1)) for shape in model_shape),
    )
    final_wavefields_list = []
    if ndim == 3:
        final_wavefields_list.extend(
            [
                velocities[0],
                normal_stresses[0],
                shear_stresses[0][1],
                shear_stresses[0][2],
                v_mem_vars_normal[0],
                v_mem_vars_shear[0][1],
                v_mem_vars_shear[0][2],
                v_mem_vars_shear[1][0],
                v_mem_vars_shear[2][0],
                s_mem_vars_normal[0],
                s_mem_vars_shear[0][1],
                s_mem_vars_shear[0][2],
                s_mem_vars_shear[1][0],
                s_mem_vars_shear[2][0],
                velocities[1],
                normal_stresses[1],
                shear_stresses[1][2],
                v_mem_vars_normal[1],
                v_mem_vars_shear[1][2],
                v_mem_vars_shear[2][1],
                s_mem_vars_normal[1],
                s_mem_vars_shear[2][1],
                s_mem_vars_shear[1][2],
                velocities[2],
                normal_stresses[2],
                v_mem_vars_normal[2],
                s_mem_vars_normal[2],
            ]
        )
    elif ndim == 2:
        final_wavefields_list.extend(
            [
                velocities[0],
                normal_stresses[0],
                shear_stresses[0][1],
                v_mem_vars_normal[0],
                v_mem_vars_shear[0][1],
                v_mem_vars_shear[1][0],
                s_mem_vars_normal[0],
                s_mem_vars_shear[1][0],
                s_mem_vars_shear[0][1],
                velocities[1],
                normal_stresses[1],
                v_mem_vars_normal[1],
                s_mem_vars_normal[1],
            ]
        )
    else:
        final_wavefields_list.extend(
            [
                velocities[0],
                normal_stresses[0],
                v_mem_vars_normal[0],
                s_mem_vars_normal[0],
            ]
        )

    return (
        *[field[s] for field in final_wavefields_list],
        *receiver_amplitudes,
    )


_update_velocities_jit = None
_update_velocities_compile = None
_update_velocities_opt = update_velocities
_update_stresses_jit = None
_update_stresses_compile = None
_update_stresses_opt = update_stresses


def elastic_func(
    python_backend: Union[Literal["eager", "jit", "compile"], bool] = False, *args: Any
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
        return elastic_python(*args)

    return cast(
        "Tuple[torch.Tensor, ...]",
        ElasticForwardFunc.apply(*args),  # type: ignore[no-untyped-call]
    )
