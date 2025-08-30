"""Elastic wave propagation

Velocity-stress formulation using C-PML.
"""

from typing import Optional, Union, List, Tuple, Sequence, Any
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
import deepwave
from deepwave.common import (setup_propagator, downsample_and_movedim,
                             PMLConfig, SurveyConfig, _as_list,
                             IGNORE_LOCATION, lambmubuoyancy_to_vpvsrho,
                             create_or_pad)
from deepwave.staggered_grid import set_pml_profiles


class Elastic(torch.nn.Module):
    """A Module wrapper around :func:`elastic`.

    This is a convenience module that allows you to only specify
    `lamb`, `mu`, `buoyancy`, and `grid_spacing` once. They will then be
    added to the list of arguments passed to :func:`elastic` when you call
    the forward method.

    Note that a copy will be made of the provided models. Gradients
    will not backpropagate to the initial guess models that are
    provided. You can use the module's `lamb`, `mu`, and `buoyancy`
    attributes to access the models.

    Args:
        lamb:
            A Tensor containing an initial guess of the first Lamé
            parameter (lambda).
        mu:
            A Tensor containing an initial guess of the second Lamé
            parameter.
        buoyancy:
            A Tensor containing an initial guess of the buoyancy
            (1/density).
        grid_spacing:
            The spatial grid cell size. It can be a single number (int or
            float), a torch.Tensor (scalar or with two elements), or a
            sequence (list or tuple) of two numbers.
        lamb_requires_grad:
            Optional bool specifying how to set the `requires_grad`
            attribute of lamb, and so whether the necessary
            data should be stored to calculate the gradient with respect
            to `lamb` during backpropagation. Default False.
        mu_requires_grad:
            Same as lamb_requires_grad, but for mu.
        buoyancy_requires_grad:
            Same as lamb_requires_grad, but for buoyancy.
    """

    def __init__(
        self,
        lamb: Tensor,
        mu: Tensor,
        buoyancy: Tensor,
        grid_spacing: Union[int, float, torch.Tensor, Sequence[Union[int,
                                                                     float]]],
        lamb_requires_grad: bool = False,
        mu_requires_grad: bool = False,
        buoyancy_requires_grad: bool = False,
    ) -> None:
        super().__init__()
        self.lamb = torch.nn.Parameter(lamb, requires_grad=lamb_requires_grad)
        self.mu = torch.nn.Parameter(mu, requires_grad=mu_requires_grad)
        self.buoyancy = torch.nn.Parameter(
            buoyancy, requires_grad=buoyancy_requires_grad)
        self.grid_spacing = grid_spacing

    def forward(
        self,
        dt: Union[int, float],
        source_amplitudes_y: Optional[Tensor] = None,
        source_amplitudes_x: Optional[Tensor] = None,
        source_locations_y: Optional[Tensor] = None,
        source_locations_x: Optional[Tensor] = None,
        receiver_locations_y: Optional[Tensor] = None,
        receiver_locations_x: Optional[Tensor] = None,
        receiver_locations_p: Optional[Tensor] = None,
        accuracy: int = 4,
        pml_width: Union[int, float, torch.Tensor,
                         Sequence[Union[int, float]]] = 20,
        pml_freq: Optional[Union[int, float]] = None,
        max_vel: Optional[Union[int, float]] = None,
        survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
        vy_0: Optional[Tensor] = None,
        vx_0: Optional[Tensor] = None,
        sigmayy_0: Optional[Tensor] = None,
        sigmaxy_0: Optional[Tensor] = None,
        sigmaxx_0: Optional[Tensor] = None,
        m_vyy_0: Optional[Tensor] = None,
        m_vyx_0: Optional[Tensor] = None,
        m_vxy_0: Optional[Tensor] = None,
        m_vxx_0: Optional[Tensor] = None,
        m_sigmayyy_0: Optional[Tensor] = None,
        m_sigmaxyy_0: Optional[Tensor] = None,
        m_sigmaxyx_0: Optional[Tensor] = None,
        m_sigmaxxx_0: Optional[Tensor] = None,
        origin: Optional[Sequence[int]] = None,
        nt: Optional[int] = None,
        model_gradient_sampling_interval: int = 1
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
               Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Perform forward propagation/modelling.

        The inputs are the same as for :func:`elastic` except that `lamb`,
        `mu`, `buoyancy`, and `grid_spacing` do not need to be provided again.
        See :func:`elastic` for a description of the inputs and outputs.
        """
        pml_config = PMLConfig(_as_list(pml_width, 'pml_width', int), pml_freq)
        survey_config = SurveyConfig(
            source_locations=[source_locations_y, source_locations_x],
            receiver_locations=[
                receiver_locations_y, receiver_locations_x,
                receiver_locations_p
            ],
            source_amplitudes=[source_amplitudes_y, source_amplitudes_x],
            wavefields=[
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
            survey_pad=survey_pad,
            origin=origin,
        )
        return elastic(
            self.lamb,
            self.mu,
            self.buoyancy,
            self.grid_spacing,
            dt,
            accuracy=accuracy,
            pml_config=pml_config,
            max_vel=max_vel,
            survey_config=survey_config,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
        )


def elastic(
    lamb: Tensor,
    mu: Tensor,
    buoyancy: Tensor,
    grid_spacing: Union[int, float, torch.Tensor, Sequence[Union[int, float]]],
    dt: Union[int, float],
    source_amplitudes_y: Optional[Tensor] = None,
    source_amplitudes_x: Optional[Tensor] = None,
    source_locations_y: Optional[Tensor] = None,
    source_locations_x: Optional[Tensor] = None,
    receiver_locations_y: Optional[Tensor] = None,
    receiver_locations_x: Optional[Tensor] = None,
    receiver_locations_p: Optional[Tensor] = None,
    accuracy: int = 4,
    pml_width: Union[int, float, torch.Tensor, Sequence[Union[int,
                                                              float]]] = 20,
    pml_freq: Optional[Union[int, float]] = None,
    max_vel: Optional[Union[int, float]] = None,
    survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
    vy_0: Optional[Tensor] = None,
    vx_0: Optional[Tensor] = None,
    sigmayy_0: Optional[Tensor] = None,
    sigmaxy_0: Optional[Tensor] = None,
    sigmaxx_0: Optional[Tensor] = None,
    m_vyy_0: Optional[Tensor] = None,
    m_vyx_0: Optional[Tensor] = None,
    m_vxy_0: Optional[Tensor] = None,
    m_vxx_0: Optional[Tensor] = None,
    m_sigmayyy_0: Optional[Tensor] = None,
    m_sigmaxyy_0: Optional[Tensor] = None,
    m_sigmaxyx_0: Optional[Tensor] = None,
    m_sigmaxxx_0: Optional[Tensor] = None,
    origin: Optional[Sequence[int]] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    pml_config: Optional[PMLConfig] = None,
    survey_config: Optional[SurveyConfig] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
           Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Elastic wave propagation (functional interface).

    This function performs forward modelling with the elastic wave equation.
    The outputs are differentiable with respect to the input float Tensors.

    For computational performance, multiple shots may be propagated
    simultaneously.

    The elastic wave equation is:
        `rho * d^2u/dt^2 = nabla . (lambda * I * (nabla . u) + mu * (nabla*u + (nabla*u).T)) + f`
    where `u` is the displacement vector, `t` is time, `rho` is density,
    `lambda` and `mu` are the Lame parameters, and `f` is the source vector.

    This propagator uses a staggered grid. See
    https://ausargeo.com/deepwave/elastic for a description.

    Args:
        lamb:
            A Tensor containing the first Lamé parameter model, lambda.
            Unlike the module interface (:class:`Elastic`), in this
            functional interface a copy is not made of the model, so gradients
            will propagate back into the provided Tensor.
        mu:
            A Tensor containing the second Lamé parameter model.
        buoyancy:
            A Tensor containing the buoyancy (1/density) model.
        grid_spacing:
            The spatial grid cell size. It can be a single number (int or
            float), a torch.Tensor (scalar or with two elements), or a
            sequence (list or tuple) of two numbers.
        dt:
            A number (int or float) specifying the time step interval of
            the input and output (internally a smaller interval may be
            used in propagation to obey the CFL condition for stability).
        source_amplitudes_y:
            A Tensor with dimensions [shot, source, time] containing time
            samples of the source wavelets for sources oriented in the
            first spatial dimension. If two shots are being propagated
            simultaneously, each containing three sources, oriented in the
            first spatial dimension, of one hundred time samples, it would
            have shape [2, 3, 100]. The time length will be the number of
            time steps in the simulation. Optional. If provided,
            `source_locations_y` must also be specified. If not provided
            (and `source_amplitudes_x` is also not specified), `nt`
            must be specified.
        source_amplitudes_x:
            A Tensor containing source wavelet time samples for sources
            oriented in the second spatial dimension. If `source_amplitudes_y`
            is also specified, both must have the same length in the
            shot and time dimensions.
        source_locations_y:
            A Tensor with dimensions [shot, source, 2], containing the
            index in the two spatial dimensions of the cell that each
            source oriented in the first spatial dimension is located in,
            relative to the origin of the model. Optional. Must be provided
            if `source_amplitudes_y` is. It should have torch.long (int64)
            datatype. The location of each source oriented in the same
            dimension must be unique within the same shot (you cannot have two
            sources oriented in the first dimension in the same shot that
            both have location [1, 2], for example).
        source_locations_x:
            A Tensor containing the locations of sources oriented in the
            second spatial dimension.
        receiver_locations_y:
            A Tensor with dimensions [shot, receiver, 2], containing
            the coordinates of the cell containing each receiver oriented
            in the first spatial dimension. Optional.
            It should have torch.long (int64) datatype. If not provided,
            the output `receiver_amplitudes_y` Tensor will be empty. If
            backpropagation will be performed, the location of each
            receiver with the same orientation must be unique within the
            same shot.
        receiver_locations_x:
            A Tensor containing the coordinates of the receivers oriented
            in the second spatial dimension.
        receiver_locations_p:
            A Tensor containing the coordinates of the pressure receivers.
        accuracy:
            An int specifying the finite difference order of accuracy. Possible
            values are 2 and 4, with larger numbers resulting in more
            accurate results but greater computational cost. Optional, with
            a default of 4.
        pml_width:
            A number (int or float), a torch.Tensor, or a sequence of
            numbers specifying the width (in number of cells) of the PML
            that prevents reflections from the edges of the model. Floats
            will be truncated to integers. If a single value is provided, 
            it will be used for all edges. If a sequence is provided, it
            should contain the values for the edges in the following order:
            [the beginning of the first dimension,
            the end of the first dimension,
            the beginning of the second dimension,
            the end of the second dimension].
            Larger values result in smaller reflections, with values of 10
            to 20 being typical. For a reflective or "free" surface, set the
            value for that edge to be zero. For example, if your model is
            oriented so that the surface that you want to be reflective is the
            beginning of the second dimension, then you could specify
            `pml_width=[20, 20, 0, 20]`.
            The model values in the PML region
            are obtained by replicating the values on the edge of the
            model. Optional, default 20.
        pml_freq:
            A number (int or float) specifying the frequency that you wish
            to use when constructing the PML. This is usually the dominant
            frequency of the source wavelet. Choosing this value poorly
            will result in the edges becoming more reflective. Optional,
            default 25 Hz (assuming `dt` is in seconds).
        max_vel:
            A number (int or float) specifying the maximum velocity, which
            is used when applying the CFL condition and when constructing
            the PML. If not specified, the actual maximum absolute
            wavespeed in the model (or portion of it that is used) will be
            used. The option to specify this is provided to allow you to
            ensure consistency between runs even if there are changes in
            the wavespeed. Optional, default None.
        survey_pad:
            A single value or list of four values, all of which are either
            an int or None, specifying whether the simulation domain
            should be restricted to a region surrounding the sources
            and receivers. If you have a large model, but the sources
            and receivers of individual shots only cover a small portion
            of it (such as in a towed streamer survey), then it may be
            wasteful to simulate wave propagation over the whole model. This
            parameter allows you to specify what distance around sources
            and receivers you would like to extract from the model to use
            for the simulation. A value of None means that no restriction
            of the model should be performed in that direction, while an
            integer value specifies the minimum number of cells that
            the edge of the simulation domain should be from any source
            or receiver in that direction (if possible). If a single value
            is provided, it applies to all directions, so specifying `None`
            will use the whole model for every simulation, while specifying
            `10` will cause the simulation domain to contain at least 10
            cells of padding in each direction around sources and receivers
            if possible. The padding will end if the edge of the model is
            encountered. Specifying a list, in the following order:
            [towards the beginning of the first dimension,
            towards the end of the first dimension,
            towards the beginning of the second dimension,
            towards the end of the second dimension]
            allows the padding in each direction to be controlled. Ints and
            `None` may be mixed, so a `survey_pad` of [5, None, None, 10]
            means that there should be at least 5 cells of padding towards
            the beginning of the first dimension, no restriction of the
            simulation domain towards the end of the first dimension or
            beginning of the second, and 10 cells of padding towards the
            end of the second dimension. If the simulation contains
            one source at [20, 15], one receiver at [10, 15], and the
            model is of shape [40, 50], then the extracted simulation
            domain with this value of `survey_pad` would cover the region
            [5:40, 0:25]. The same simulation domain will be used for all
            shots propagated simultaneously, so if this option is used then
            it is advisable to propagate shots that cover similar regions
            of the model simultaneously so that it provides a benefit (if
            the shots cover opposite ends of a dimension, then that
            whole dimension will be used regardless of what `survey_pad`
            value is used). This option is disregarded if any initial
            wavefields are provided, as those will instead be used to
            determine the simulation domain. Optional, default None.
        vy_0:
            A Tensor specifying the initial vy (velocity in the first
            dimension) wavefield at time step -1/2 (using Deepwave's internal
            time step interval, which may be smaller than the user provided
            one to obey the CFL condition). It should have three dimensions,
            with the first dimension being shot and the subsequent two
            corresponding to the two spatial dimensions. The spatial shape
            should be equal to the simulation domain, which is the extracted
            model plus the PML. If two shots are being propagated
            simultaneously in a region of size [20, 30] extracted from the
            model for the simulation, and `pml_width=[1, 2, 3, 4]`, then
            `vy_0` should be of shape [2, 23, 37]. Optional, default all zeros.
        vx_0:
            A Tensor specifying the initial vx (velocity in the second
            dimension) wavefield. See the entry for `vy_0` for more details.
        sigmayy_0, sigmaxy_0, sigmaxx_0:
            A Tensor specifying the initial value for the stress field at time
            step 0.
        m_vyy_0, m_vyx_0, m_vxy_0, m_vxx_0:
            A Tensor specifying the initial value for the "memory variable"
            for the velocity, used in the PML.
        m_sigmayyy_0, m_sigmaxyy_0, m_sigmaxyx_0, m_sigmaxxx_0:
            A Tensor specifying the initial value for the "memory variable"
            for the stress, used in the PML.
        origin:
            A list of ints specifying the origin of the provided initial
            wavefields relative to the origin of the model. Only relevant
            if initial wavefields are provided. The origin of a wavefield
            is the cell where the extracted model used for the simulation
            domain began. It does not include the PML. So if a simulation
            is performed using the model region [10:20, 15:30], the origin
            is [10, 15]. Optional, default [0, 0].
        nt:
            If the source amplitudes are not provided then you must instead
            specify the number of time steps to run the simulation for by
            providing an integer for `nt`. You cannot specify both the
            source amplitudes and `nt`.
        model_gradient_sampling_interval:
            An int specifying the number of time steps between
            contributions to the model gradient. The gradient with respect
            to the model is an integral over the backpropagation time steps.
            The time sampling frequency of this integral should be at
            least the Nyquist frequency of the source or data (whichever
            is greater). If this Nyquist frequency is substantially less
            than 1/dt, you may wish to reduce the sampling frequency of
            the integral to reduce computational (especially memory) costs.
            Optional, default 1 (integral is sampled every time step
            interval `dt`).
        freq_taper_frac:
            A float specifying the fraction of the end of the source and
            receiver amplitudes (if present) in the frequency domain to
            cosine taper if they are resampled due to the CFL condition.
            This might be useful to reduce ringing. A value of 0.1 means
            that the top 10% of frequencies will be tapered.
            Default 0.0 (no tapering).
        time_pad_frac:
            A float specifying the amount of zero padding that will
            be added to the source and receiver amplitudes (if present) before
            resampling and removed afterwards, if they are resampled due to
            the CFL condition, as a fraction of their length. This might be
            useful to reduce wraparound artifacts. A value of 0.1 means that
            zero padding of 10% of the number of time samples will be used.
            Default 0.0.
        time_taper:
            A bool specifying whether to apply a Hann window in time to
            source and receiver amplitudes (if present). This is useful
            during correctness tests of the propagators as it ensures that
            signals taper to zero at their edges in time, avoiding the
            possibility of high frequencies being introduced.

    Returns:
        Tuple[Tensor]:

            vy, vx:
                A Tensor containing the final velocity wavefield of the
                simulation.
            sigmayy, sigmaxy, sigmaxx:
                A Tensor containing the final stress field of the simulation.
            m_vyy_0, m_vyx_0, m_vxy_0, m_vxx_0:
                A Tensor containing the final value for the "memory variable"
                for the velocity, used in the PML.
            m_sigmayyy_0, m_sigmaxyy_0, m_sigmaxyx_0, m_sigmaxxx_0:
                A Tensor containing the final value for the "memory variable"
                for the stress, used in the PML.
            receiver_amplitudes_p:
                A Tensor containing the receiver amplitudes for pressure
                receivers.
            receiver_amplitudes_y:
                A Tensor of dimensions [shot, receiver, time] containing
                the receiver amplitudes recorded at the provided receiver
                locations for receivers oriented in the first spatial
                dimension. If no such receiver locations
                were specified then this Tensor will be empty.
            receiver_amplitudes_x:
                A Tensor containing the receiver amplitudes for receivers
                oriented in the second spatial dimension.

    """
    if pml_config is None:
        pml_config = PMLConfig(_as_list(pml_width, 'pml_width', int), pml_freq)
    if survey_config is None:
        survey_config = SurveyConfig(
            source_locations=[source_locations_y, source_locations_x],
            receiver_locations=[
                receiver_locations_y, receiver_locations_x,
                receiver_locations_p
            ],
            source_amplitudes=[source_amplitudes_y, source_amplitudes_x],
            wavefields=[
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
            survey_pad=survey_pad,
            origin=origin,
        )
    # Check that sources and receivers are not on the last row or column,
    # as these are not used
    if (survey_config.source_locations[0] is not None
            and survey_config.source_locations[0][..., 1].max()
            >= lamb.shape[1] - 1):
        raise RuntimeError("With the provided model, the maximum y source "
                           "location in the second dimension must be less "
                           "than " + str(lamb.shape[1] - 1) + ".")
    if (survey_config.receiver_locations[0] is not None
            and survey_config.receiver_locations[0][..., 1].max()
            >= lamb.shape[1] - 1):
        raise RuntimeError("With the provided model, the maximum y "
                           "receiver location in the second dimension "
                           "must be less than " + str(lamb.shape[1] - 1) + ".")
    if (survey_config.source_locations[1] is not None):
        dim_location = survey_config.source_locations[1][..., 0]
        dim_location = dim_location[dim_location != IGNORE_LOCATION]
        if (dim_location.min() <= 0):
            raise RuntimeError("The minimum x source "
                               "location in the first dimension must be "
                               "greater than 0.")
    if (survey_config.receiver_locations[1] is not None):
        dim_location = survey_config.receiver_locations[1][..., 0]
        dim_location = dim_location[dim_location != IGNORE_LOCATION]
        if (dim_location.min() <= 0):
            raise RuntimeError("The minimum x receiver "
                               "location in the first dimension must be "
                               "greater than 0.")
    if (survey_config.receiver_locations[2] is not None):
        dim_location = survey_config.receiver_locations[2][..., 0]
        dim_location = dim_location[dim_location != IGNORE_LOCATION]
        if (survey_config.receiver_locations[2][..., 1].max()
                >= lamb.shape[1] - 1 or dim_location.min() <= 0):
            raise RuntimeError("With the provided model, the pressure "
                               "receiver locations in the second dimension "
                               "must be less than " + str(lamb.shape[1] - 1) +
                               " and "
                               "in the first dimension must be "
                               "greater than 0.")
    if accuracy not in [2, 4]:
        raise RuntimeError("The accuracy must be 2 or 4.")
    vp, vs, _ = lambmubuoyancy_to_vpvsrho(lamb.abs(), mu.abs(), buoyancy.abs())
    max_model_vel = max(vp.abs().max().item(), vs.abs().max().item())
    try:
        min_nonzero_vp = vp[vp.nonzero(as_tuple=True)].abs().min().item()
    except Exception:
        min_nonzero_vp = 0
    try:
        min_nonzero_vs = vs[vs.nonzero(as_tuple=True)].abs().min().item()
    except Exception:
        min_nonzero_vs = 0
    if min_nonzero_vp == 0 and min_nonzero_vs == 0:
        min_nonzero_model_vel = 0.0
    elif min_nonzero_vp == 0:
        min_nonzero_model_vel = float(min_nonzero_vs)
    elif min_nonzero_vs == 0:
        min_nonzero_model_vel = float(min_nonzero_vp)
    else:
        min_nonzero_model_vel = float(min(min_nonzero_vp, min_nonzero_vs))
    fd_pad = [0] * 4

    (models, source_amplitudes_l, wavefields,
     sources_i_l, receivers_i_l,
     grid_spacing, dt, nt, n_shots,
     step_ratio, model_gradient_sampling_interval,
     accuracy, pml_width_l, pml_freq, max_vel, resample_config, device, dtype) = \
        setup_propagator([lamb, mu, buoyancy], ['replicate'] * 3,
                         _as_list(grid_spacing, 'grid_spacing', float), dt,
                         survey_config,
                         accuracy, fd_pad, pml_config,
                         max_vel, min_nonzero_model_vel, max_model_vel,
                         nt, model_gradient_sampling_interval,
                         freq_taper_frac, time_pad_frac, time_taper, 2)

    if any([s <= (accuracy + 1) * 2 for s in models[0].shape[1:]]):
        raise RuntimeError("The model must have at least " +
                           str((accuracy + 1) * 2 + 1) +
                           " elements in each dimension (including PML).")

    ny, nx = models[0].shape[-2:]
    # source_amplitudes_y
    # Need to interpolate buoyancy to vy
    mask = sources_i_l[0] == IGNORE_LOCATION
    sources_i_masked = sources_i_l[0].clone()
    sources_i_masked[mask] = 0
    if source_amplitudes_l[0].numel() > 0:
        source_amplitudes_l[0] = (
            source_amplitudes_l[0] *
            (models[2].view(-1, ny * nx).expand(n_shots, -1).gather(
                1, sources_i_masked) + models[2].view(-1, ny * nx).expand(
                    n_shots, -1).gather(1, sources_i_masked + 1) +
             models[2].view(-1, ny * nx).expand(n_shots, -1).gather(
                 1, sources_i_masked + nx) +
             models[2].view(-1, ny * nx).expand(n_shots, -1).gather(
                 1, sources_i_masked + nx + 1))) / 4 * dt
    # source_amplitudes_x
    mask = sources_i_l[1] == IGNORE_LOCATION
    sources_i_masked = sources_i_l[1].clone()
    sources_i_masked[mask] = 0
    if source_amplitudes_l[1].numel() > 0:
        source_amplitudes_l[1] = (source_amplitudes_l[1] * (models[2].view(
            -1, ny * nx).expand(n_shots, -1).gather(1, sources_i_masked)) * dt)

    pml_profiles = set_pml_profiles(pml_width_l, accuracy, fd_pad, dt,
                                    grid_spacing, max_vel, dtype, device,
                                    pml_freq, ny, nx)

    # Run the forward propagator
    (vy, vx, sigmayy, sigmaxy, sigmaxx,
     m_vyy, m_vyx, m_vxy, m_vxx,
     m_sigmayyy, m_sigmaxyy,
     m_sigmaxyx, m_sigmaxxx, receiver_amplitudes_p, receiver_amplitudes_y,
     receiver_amplitudes_x) = \
        elastic_func(
            *models, *source_amplitudes_l,
            *wavefields, *pml_profiles,
            *sources_i_l, *receivers_i_l,
            *grid_spacing, dt, nt,
            step_ratio * model_gradient_sampling_interval,
            accuracy, pml_width_l, n_shots
        )

    # Average velocity receiver samples at t-1/2 and t+1/2 to approximately
    # get samples at t.
    def average_adjacent(receiver_amplitudes: Tensor) -> Tensor:
        if receiver_amplitudes.numel() == 0:
            return receiver_amplitudes
        return (receiver_amplitudes[1:] + receiver_amplitudes[:-1]) / 2

    receiver_amplitudes_y = average_adjacent(receiver_amplitudes_y)
    receiver_amplitudes_x = average_adjacent(receiver_amplitudes_x)
    receiver_amplitudes_y = downsample_and_movedim(
        receiver_amplitudes_y,
        resample_config.step_ratio,
        resample_config.freq_taper_frac,
        resample_config.time_pad_frac,
        resample_config.time_taper,
    )
    receiver_amplitudes_x = downsample_and_movedim(
        receiver_amplitudes_x,
        resample_config.step_ratio,
        resample_config.freq_taper_frac,
        resample_config.time_pad_frac,
        resample_config.time_taper,
    )
    receiver_amplitudes_p = downsample_and_movedim(
        receiver_amplitudes_p,
        resample_config.step_ratio,
        resample_config.freq_taper_frac,
        resample_config.time_pad_frac,
        resample_config.time_taper,
    )

    return (vy, vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,
            m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx,
            receiver_amplitudes_p, receiver_amplitudes_y,
            receiver_amplitudes_x)


def zero_edges(tensor: Tensor, ny: int, nx: int) -> None:
    tensor[:, ny - 1, :] = 0
    tensor[:, :, nx - 1] = 0
    tensor[:, 0, :] = 0
    tensor[:, :, 0] = 0


def zero_edge_top(tensor: Tensor) -> None:
    tensor[:, 0, :] = 0


def zero_edge_right(tensor: Tensor, nx: int) -> None:
    tensor[:, :, nx - 1] = 0


def zero_interior(tensor: Tensor, ybegin: int, yend: int, xbegin: int,
                  xend: int) -> Tensor:
    tensor = tensor.clone()
    tensor[:, ybegin:yend, xbegin:xend] = 0
    return tensor


class ElasticForwardFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        lamb: Tensor,
        mu: Tensor,
        buoyancy: Tensor,
        source_amplitudes_y: Tensor,
        source_amplitudes_x: Tensor,
        vy: Tensor,
        vx: Tensor,
        sigmayy: Tensor,
        sigmaxy: Tensor,
        sigmaxx: Tensor,
        m_vyy: Tensor,
        m_vyx: Tensor,
        m_vxy: Tensor,
        m_vxx: Tensor,
        m_sigmayyy: Tensor,
        m_sigmaxyy: Tensor,
        m_sigmaxyx: Tensor,
        m_sigmaxxx: Tensor,
        ay: Tensor,
        ayh: Tensor,
        ax: Tensor,
        axh: Tensor,
        by: Tensor,
        byh: Tensor,
        bx: Tensor,
        bxh: Tensor,
        sources_y_i: Tensor,
        sources_x_i: Tensor,
        receivers_y_i: Tensor,
        receivers_x_i: Tensor,
        receivers_p_i: Tensor,
        dy: float,
        dx: float,
        dt: float,
        nt: int,
        step_ratio: int,
        accuracy: int,
        pml_width: List[int],
        n_shots: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
               Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

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

        pml_y0 = int(max(pml_width[0], accuracy // 2))
        pml_y1 = int(min(ny - pml_width[1], ny - accuracy // 2))
        pml_x0 = int(max(pml_width[2], accuracy // 2))
        pml_x1 = int(min(nx - pml_width[3], nx - accuracy // 2))

        size_with_batch = (n_shots, *lamb.shape[-2:])
        vy = create_or_pad(vy, 0, lamb.device, lamb.dtype, size_with_batch)
        vx = create_or_pad(vx, 0, lamb.device, lamb.dtype, size_with_batch)
        sigmayy = create_or_pad(sigmayy, 0, lamb.device, lamb.dtype,
                                size_with_batch)
        sigmaxy = create_or_pad(sigmaxy, 0, lamb.device, lamb.dtype,
                                size_with_batch)
        sigmaxx = create_or_pad(sigmaxx, 0, lamb.device, lamb.dtype,
                                size_with_batch)
        m_vyy = create_or_pad(m_vyy, 0, lamb.device, lamb.dtype,
                              size_with_batch)
        m_vyx = create_or_pad(m_vyx, 0, lamb.device, lamb.dtype,
                              size_with_batch)
        m_vxy = create_or_pad(m_vxy, 0, lamb.device, lamb.dtype,
                              size_with_batch)
        m_vxx = create_or_pad(m_vxx, 0, lamb.device, lamb.dtype,
                              size_with_batch)
        m_sigmayyy = create_or_pad(m_sigmayyy, 0, lamb.device, lamb.dtype,
                                   size_with_batch)
        m_sigmaxyy = create_or_pad(m_sigmaxyy, 0, lamb.device, lamb.dtype,
                                   size_with_batch)
        m_sigmaxyx = create_or_pad(m_sigmaxyx, 0, lamb.device, lamb.dtype,
                                   size_with_batch)
        m_sigmaxxx = create_or_pad(m_sigmaxxx, 0, lamb.device, lamb.dtype,
                                   size_with_batch)
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
            dvydbuoyancy.fill_(0)
            dvxdbuoyancy.resize_(nt // step_ratio, n_shots, ny, nx)
            dvxdbuoyancy.fill_(0)
        if (lamb.requires_grad or mu.requires_grad):
            dvydy_store.resize_(nt // step_ratio, n_shots, ny, nx)
            dvxdx_store.resize_(nt // step_ratio, n_shots, ny, nx)
        if (mu.requires_grad):
            dvydxdvxdy_store.resize_(nt // step_ratio, n_shots, ny, nx)

        if receivers_y_i.numel() > 0:
            receiver_amplitudes_y.resize_(nt + 1, n_shots,
                                          n_receivers_y_per_shot)
            receiver_amplitudes_y.fill_(0)
        if receivers_x_i.numel() > 0:
            receiver_amplitudes_x.resize_(nt + 1, n_shots,
                                          n_receivers_x_per_shot)
            receiver_amplitudes_x.fill_(0)
        if receivers_p_i.numel() > 0:
            receiver_amplitudes_p.resize_(nt, n_shots, n_receivers_p_per_shot)
            receiver_amplitudes_p.fill_(0)

        if lamb.is_cuda:
            aux = lamb.get_device()
            if dtype == torch.float32:
                if accuracy == 2:
                    forward = deepwave.dll.elastic_iso_2_float_forward_cuda
                elif accuracy == 4:
                    forward = deepwave.dll.elastic_iso_4_float_forward_cuda
            else:
                if accuracy == 2:
                    forward = deepwave.dll.elastic_iso_2_double_forward_cuda
                elif accuracy == 4:
                    forward = deepwave.dll.elastic_iso_4_double_forward_cuda
        else:
            if deepwave.USE_OPENMP:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if dtype == torch.float32:
                if accuracy == 2:
                    forward = deepwave.dll.elastic_iso_2_float_forward_cpu
                elif accuracy == 4:
                    forward = deepwave.dll.elastic_iso_4_float_forward_cpu
            else:
                if accuracy == 2:
                    forward = deepwave.dll.elastic_iso_2_double_forward_cpu
                elif accuracy == 4:
                    forward = deepwave.dll.elastic_iso_4_double_forward_cpu

        if vy.numel() > 0 and nt > 0:
            start_t = 0
            forward(lamb.data_ptr(), mu.data_ptr(), buoyancy.data_ptr(),
                    source_amplitudes_y.data_ptr(),
                    source_amplitudes_x.data_ptr(), vy.data_ptr(),
                    vx.data_ptr(), sigmayy.data_ptr(), sigmaxy.data_ptr(),
                    sigmaxx.data_ptr(), m_vyy.data_ptr(), m_vyx.data_ptr(),
                    m_vxy.data_ptr(), m_vxx.data_ptr(), m_sigmayyy.data_ptr(),
                    m_sigmaxyy.data_ptr(), m_sigmaxyx.data_ptr(),
                    m_sigmaxxx.data_ptr(), dvydbuoyancy.data_ptr(),
                    dvxdbuoyancy.data_ptr(), dvydy_store.data_ptr(),
                    dvxdx_store.data_ptr(), dvydxdvxdy_store.data_ptr(),
                    receiver_amplitudes_y.data_ptr(),
                    receiver_amplitudes_x.data_ptr(),
                    receiver_amplitudes_p.data_ptr(), ay.data_ptr(),
                    ayh.data_ptr(), ax.data_ptr(), axh.data_ptr(),
                    by.data_ptr(),
                    byh.data_ptr(), bx.data_ptr(), bxh.data_ptr(),
                    sources_y_i.data_ptr(), sources_x_i.data_ptr(),
                    receivers_y_i.data_ptr(), receivers_x_i.data_ptr(),
                    receivers_p_i.data_ptr(), dy, dx, dt, nt, n_shots, ny, nx,
                    n_sources_y_per_shot, n_sources_x_per_shot,
                    n_receivers_y_per_shot, n_receivers_x_per_shot,
                    n_receivers_p_per_shot, step_ratio, lamb.requires_grad,
                    mu.requires_grad, buoyancy.requires_grad, lamb_batched,
                    mu_batched, buoyancy_batched, start_t, pml_y0, pml_y1,
                    pml_x0, pml_x1, aux)

        if (lamb.requires_grad or mu.requires_grad or buoyancy.requires_grad
                or source_amplitudes_y.requires_grad
                or source_amplitudes_x.requires_grad or vy.requires_grad
                or vx.requires_grad or sigmayy.requires_grad
                or sigmaxy.requires_grad or sigmaxx.requires_grad
                or m_vyy.requires_grad or m_vyx.requires_grad
                or m_vxy.requires_grad or m_vxx.requires_grad
                or m_sigmayyy.requires_grad or m_sigmaxyy.requires_grad
                or m_sigmaxyx.requires_grad or m_sigmaxxx.requires_grad):
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
    @once_differentiable
    def backward(
        ctx: Any,
        vy: Tensor,
        vx: Tensor,
        sigmayy: Tensor,
        sigmaxy: Tensor,
        sigmaxx: Tensor,
        m_vyy: Tensor,
        m_vyx: Tensor,
        m_vxy: Tensor,
        m_vxx: Tensor,
        m_sigmayyy: Tensor,
        m_sigmaxyy: Tensor,
        m_sigmaxyx: Tensor,
        m_sigmaxxx: Tensor,
        grad_r_p: Tensor,
        grad_r_y: Tensor,
        grad_r_x: Tensor,
    ) -> Tuple[Optional[Tensor], ...]:
        (lamb, mu, buoyancy, ay, ayh, ax, axh, by, byh, bx, bxh, sources_y_i,
         sources_x_i, receivers_y_i, receivers_x_i, receivers_p_i,
         dvydbuoyancy, dvxdbuoyancy, dvydy_store, dvxdx_store,
         dvydxdvxdy_store) = ctx.saved_tensors

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

        pml_y0 = int(max(pml_width[0], accuracy // 2))
        pml_y1 = int(min(ny - pml_width[1], ny - accuracy // 2))
        pml_x0 = int(max(pml_width[2], accuracy // 2))
        pml_x1 = int(min(nx - pml_width[3], nx - accuracy // 2))
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
        vy = create_or_pad(vy, 0, lamb.device, lamb.dtype, size_with_batch)
        vx = create_or_pad(vx, 0, lamb.device, lamb.dtype, size_with_batch)
        sigmayy = create_or_pad(sigmayy, 0, lamb.device, lamb.dtype,
                                size_with_batch)
        sigmaxy = create_or_pad(sigmaxy, 0, lamb.device, lamb.dtype,
                                size_with_batch)
        sigmaxx = create_or_pad(sigmaxx, 0, lamb.device, lamb.dtype,
                                size_with_batch)
        m_vyy = create_or_pad(m_vyy, 0, lamb.device, lamb.dtype,
                              size_with_batch)
        m_vyx = create_or_pad(m_vyx, 0, lamb.device, lamb.dtype,
                              size_with_batch)
        m_vxy = create_or_pad(m_vxy, 0, lamb.device, lamb.dtype,
                              size_with_batch)
        m_vxx = create_or_pad(m_vxx, 0, lamb.device, lamb.dtype,
                              size_with_batch)
        m_sigmayyy = create_or_pad(m_sigmayyy, 0, lamb.device, lamb.dtype,
                                   size_with_batch)
        m_sigmaxyy = create_or_pad(m_sigmaxyy, 0, lamb.device, lamb.dtype,
                                   size_with_batch)
        m_sigmaxyx = create_or_pad(m_sigmaxyx, 0, lamb.device, lamb.dtype,
                                   size_with_batch)
        m_sigmaxxx = create_or_pad(m_sigmaxxx, 0, lamb.device, lamb.dtype,
                                   size_with_batch)
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
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = deepwave.dll.elastic_iso_2_float_backward_cuda
                elif accuracy == 4:
                    backward = deepwave.dll.elastic_iso_4_float_backward_cuda
            else:
                if accuracy == 2:
                    backward = deepwave.dll.elastic_iso_2_double_backward_cuda
                elif accuracy == 4:
                    backward = deepwave.dll.elastic_iso_4_double_backward_cuda
        else:
            if deepwave.USE_OPENMP:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if lamb.requires_grad and not lamb_batched and aux > 1 and deepwave.USE_OPENMP:
                grad_lamb_tmp.resize_(n_shots, *lamb.shape[-2:])
                grad_lamb_tmp.fill_(0)
                grad_lamb_tmp_ptr = grad_lamb_tmp.data_ptr()
            if mu.requires_grad and not mu_batched and aux > 1 and deepwave.USE_OPENMP:
                grad_mu_tmp.resize_(n_shots, *mu.shape[-2:])
                grad_mu_tmp.fill_(0)
                grad_mu_tmp_ptr = grad_mu_tmp.data_ptr()
            if buoyancy.requires_grad and not buoyancy_batched and aux > 1 and deepwave.USE_OPENMP:
                grad_buoyancy_tmp.resize_(n_shots, *buoyancy.shape[-2:])
                grad_buoyancy_tmp.fill_(0)
                grad_buoyancy_tmp_ptr = grad_buoyancy_tmp.data_ptr()
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = deepwave.dll.elastic_iso_2_float_backward_cpu
                elif accuracy == 4:
                    backward = deepwave.dll.elastic_iso_4_float_backward_cpu
            else:
                if accuracy == 2:
                    backward = deepwave.dll.elastic_iso_2_double_backward_cpu
                elif accuracy == 4:
                    backward = deepwave.dll.elastic_iso_4_double_backward_cpu

        if vy.numel() > 0 and nt > 0:
            start_t = 0
            backward(
                lamb.data_ptr(), mu.data_ptr(), buoyancy.data_ptr(),
                grad_r_y.data_ptr(), grad_r_x.data_ptr(), grad_r_p.data_ptr(),
                vy.data_ptr(), vx.data_ptr(), sigmayy.data_ptr(),
                sigmaxy.data_ptr(), sigmaxx.data_ptr(), m_vyy.data_ptr(),
                m_vyx.data_ptr(), m_vxy.data_ptr(), m_vxx.data_ptr(),
                m_sigmayyy.data_ptr(), m_sigmaxyy.data_ptr(),
                m_sigmaxyx.data_ptr(), m_sigmaxxx.data_ptr(),
                m_sigmayyyn.data_ptr(), m_sigmaxyyn.data_ptr(),
                m_sigmaxyxn.data_ptr(), m_sigmaxxxn.data_ptr(),
                dvydbuoyancy.data_ptr(), dvxdbuoyancy.data_ptr(),
                dvydy_store.data_ptr(), dvxdx_store.data_ptr(),
                dvydxdvxdy_store.data_ptr(), grad_f_y.data_ptr(),
                grad_f_x.data_ptr(), grad_lamb.data_ptr(), grad_lamb_tmp_ptr,
                grad_mu.data_ptr(), grad_mu_tmp_ptr,
                grad_buoyancy.data_ptr(), grad_buoyancy_tmp_ptr, ay.data_ptr(),
                ayh.data_ptr(), ax.data_ptr(), axh.data_ptr(), by.data_ptr(),
                byh.data_ptr(), bx.data_ptr(), bxh.data_ptr(),
                sources_y_i.data_ptr(), sources_x_i.data_ptr(),
                receivers_y_i.data_ptr(), receivers_x_i.data_ptr(),
                receivers_p_i.data_ptr(), dy, dx, dt, nt, n_shots, ny, nx,
                n_sources_y_per_shot * source_amplitudes_y_requires_grad,
                n_sources_x_per_shot * source_amplitudes_x_requires_grad,
                n_receivers_y_per_shot, n_receivers_x_per_shot,
                n_receivers_p_per_shot, step_ratio, lamb.requires_grad,
                mu.requires_grad, buoyancy.requires_grad, lamb_batched,
                mu_batched, buoyancy_batched, start_t, spml_y0, spml_y1,
                spml_x0, spml_x1, vpml_y0, vpml_y1, vpml_x0, vpml_x1, aux)

        m_vyy = zero_interior(m_vyy, pml_y0 + 1, pml_y1, 0, nx)
        m_vxx = zero_interior(m_vxx, 0, ny, pml_x0, pml_x1 - 1)
        m_vyx = zero_interior(m_vyx, 0, ny, pml_x0 + 1, pml_x1 - 1)
        m_vxy = zero_interior(m_vxy, pml_y0 + 1, pml_y1 - 1, 0, nx)

        if nt % 2 == 0:
            m_sigmayyy = zero_interior(m_sigmayyy, pml_y0, pml_y1, 0, nx)
            m_sigmaxyx = zero_interior(m_sigmaxyx, 0, ny, pml_x0, pml_x1 - 1)
            m_sigmaxyy = zero_interior(m_sigmaxyy, pml_y0 + 1, pml_y1, 0, nx)
            m_sigmaxxx = zero_interior(m_sigmaxxx, 0, ny, pml_x0, pml_x1)
            return (grad_lamb, grad_mu, grad_buoyancy, grad_f_y, grad_f_x, vy,
                    vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,
                    m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx, None, None,
                    None, None, None, None, None, None, None, None, None, None,
                    None, None, None, None, None, None, None, None, None)
        else:
            m_sigmayyyn = zero_interior(m_sigmayyyn, pml_y0, pml_y1, 0, nx)
            m_sigmaxyxn = zero_interior(m_sigmaxyxn, 0, ny, pml_x0, pml_x1 - 1)
            m_sigmaxyyn = zero_interior(m_sigmaxyyn, pml_y0 + 1, pml_y1, 0, nx)
            m_sigmaxxxn = zero_interior(m_sigmaxxxn, 0, ny, pml_x0, pml_x1)
            return (grad_lamb, grad_mu, grad_buoyancy, grad_f_y, grad_f_x, vy,
                    vx, sigmayy, sigmaxy, sigmaxx, m_vyy, m_vyx, m_vxy, m_vxx,
                    m_sigmayyyn, m_sigmaxyyn, m_sigmaxyxn, m_sigmaxxxn, None,
                    None, None, None, None, None, None, None, None, None, None,
                    None, None, None, None, None, None, None, None, None, None)


def elastic_func(*args: Any) -> Tuple[Tensor, ...]:
    return ElasticForwardFunc.apply(*args)
