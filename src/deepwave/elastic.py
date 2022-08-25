"""Elastic wave propagation

Velocity-stress formulation using C-PML.
"""

from typing import Optional, Union, List, Tuple
import torch
from torch import Tensor
from deepwave.common import (setup_propagator,
                             downsample_and_movedim)


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
            A 2D Tensor containing an initial guess of the first Lamé
            parameter (lambda).
        mu:
            A 2D Tensor containing an initial guess of the second Lamé
            parameter.
        buoyancy:
            A 2D Tensor containing an initial guess of the buoyancy
            (1/density).
        grid_spacing:
            The spatial grid cell size, specified with a single real number
            (used for both dimensions) or a List or Tensor of length
            two (the length in each of the two dimensions).
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
    def __init__(self, lamb: Tensor, mu: Tensor, buoyancy: Tensor,
                 grid_spacing: Union[int, float, List[float],
                                     Tensor],
                 lamb_requires_grad: bool = False,
                 mu_requires_grad: bool = False,
                 buoyancy_requires_grad: bool = False) -> None:
        super().__init__()
        if lamb.ndim != 2:
            raise RuntimeError("lamb must have two dimensions")
        if mu.ndim != 2:
            raise RuntimeError("mu must have two dimensions")
        if buoyancy.ndim != 2:
            raise RuntimeError("buoyancy must have two dimensions")
        self.lamb = torch.nn.Parameter(lamb, requires_grad=lamb_requires_grad)
        self.mu = torch.nn.Parameter(mu, requires_grad=mu_requires_grad)
        self.buoyancy = torch.nn.Parameter(
            buoyancy, requires_grad=buoyancy_requires_grad
        )
        self.grid_spacing = grid_spacing

    def forward(self, dt: float,
                source_amplitudes_y: Optional[Tensor] = None,
                source_amplitudes_x: Optional[Tensor] = None,
                source_locations_y: Optional[Tensor] = None,
                source_locations_x: Optional[Tensor] = None,
                receiver_locations_y: Optional[Tensor] = None,
                receiver_locations_x: Optional[Tensor] = None,
                accuracy: int = 4, pml_width: Union[int, List[int]] = 20,
                pml_freq: Optional[float] = None,
                max_vel: Optional[float] = None,
                survey_pad: Optional[Union[int,
                                           List[Optional[int]]]] = None,
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
                origin: Optional[List[int]] = None,
                nt: Optional[int] = None,
                model_gradient_sampling_interval: int = 1) -> Tuple[Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor]:
        """Perform forward propagation/modelling.

        The inputs are the same as for :func:`elastic` except that `lamb`,
        `mu`, `buoyancy`, and `grid_spacing` do not need to be provided again.
        See :func:`elastic` for a description of the inputs and outputs.
        """
        return elastic(self.lamb, self.mu, self.buoyancy, self.grid_spacing,
                       dt,
                       source_amplitudes_y=source_amplitudes_y,
                       source_amplitudes_x=source_amplitudes_x,
                       source_locations_y=source_locations_y,
                       source_locations_x=source_locations_x,
                       receiver_locations_y=receiver_locations_y,
                       receiver_locations_x=receiver_locations_x,
                       accuracy=accuracy, pml_width=pml_width,
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
                       origin=origin, nt=nt,
                       model_gradient_sampling_interval=
                       model_gradient_sampling_interval)


def elastic(lamb: Tensor, mu: Tensor, buoyancy: Tensor,
            grid_spacing: Union[int, float, List[float],
                                Tensor],
            dt: float,
            source_amplitudes_y: Optional[Tensor] = None,
            source_amplitudes_x: Optional[Tensor] = None,
            source_locations_y: Optional[Tensor] = None,
            source_locations_x: Optional[Tensor] = None,
            receiver_locations_y: Optional[Tensor] = None,
            receiver_locations_x: Optional[Tensor] = None,
            accuracy: int = 4, pml_width: Union[int, List[int]] = 20,
            pml_freq: Optional[float] = None,
            max_vel: Optional[float] = None,
            survey_pad: Optional[Union[int, List[Optional[int]]]] = None,
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
            origin: Optional[List[int]] = None, nt: Optional[int] = None,
            model_gradient_sampling_interval: int = 1,
            freq_taper_frac: float = 0.0,
            time_pad_frac: float = 0.0) -> Tuple[Tensor, Tensor,
                                                 Tensor, Tensor,
                                                 Tensor, Tensor,
                                                 Tensor, Tensor,
                                                 Tensor, Tensor,
                                                 Tensor, Tensor,
                                                 Tensor, Tensor,
                                                 Tensor]:
    """Elastic wave propagation (functional interface).

    This function performs forward modelling with the elastic wave equation.
    The outputs are differentiable with respect to the input float Tensors.

    For computational performance, multiple shots may be propagated
    simultaneously.

    Args:
        lamb:
            A 2D Tensor containing the first Lamé parameter model, lambda.
            Unlike the module interface (:class:`Elastic`), in this
            functional interface a copy is not made of the model, so gradients
            will propagate back into the provided Tensor.
        mu:
            A 2D Tensor containing the second Lamé parameter model.
        buoyancy:
            A 2D Tensor containing the buoyancy (1/density) model.
        grid_spacing:
            The spatial grid cell size, specified with a single real number
            (used for both dimensions) or a List or Tensor of length
            two (the length in each of the two dimensions).
        dt:
            A float specifying the time step interval of the input and
            output (internally a smaller interval may be used in
            propagation to obey the CFL condition for stability).
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
        accuracy:
            An int specifying the finite difference order of accuracy. Possible
            values are 2 and 4, with larger numbers resulting in more
            accurate results but greater computational cost. Optional, with
            a default of 4.
        pml_width:
            An int or list of four ints specifying the width (in number of
            cells) of the PML that prevents reflections from the edges of
            the model. If a single integer is provided, that value will be
            used for all edges. If a list is provided, it should contain
            the integer values for the edges in the following order:
            [the beginning of the first dimension,
            the end of the first dimension,
            the beginning of the second dimension,
            the end of the second dimension].
            Larger values result in smaller reflections, with values of 10
            to 20 being typical. For a reflective or "free" surface, set the
            value for that edge to be zero. For example, if your model is
            oriented so that the surface that you want to be reflective is the
            beginning of the second dimension, then you could specify
            `pml_width=[20, 20, 0, 20]`. The model values in the PML region
            are obtained by replicating the values on the edge of the
            model. Optional, default 20.
        pml_freq:
            A float specifying the frequency that you wish to use when
            constructing the PML. This is usually the dominant frequency
            of the source wavelet. Choosing this value poorly will
            result in the edges becoming more reflective. Optional, default
            25 Hz (assuming `dt` is in seconds).
        max_vel:
            A float specifying the maximum velocity, which is used when
            applying the CFL condition and when constructing the PML. If
            not specified, the actual maximum absolute wavespeed in the
            model (or portion of it that is used) will be used. The option
            to specify this is provided to allow you to ensure consistency
            between runs even if there are changes in the wavespeed.
            Optional, default None.
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
    if (source_locations_y is not None and
            source_locations_y[..., 1].max() >= lamb.shape[1] - 1):
        raise RuntimeError("With the provided model, the maximum y source "
                           "location in the second dimension must be less "
                           "than " + str(lamb.shape[1] - 1) + ".")
    if (receiver_locations_y is not None and
            receiver_locations_y[..., 1].max() >= lamb.shape[1] - 1):
        raise RuntimeError("With the provided model, the maximum y "
                           "receiver location in the second dimension "
                           "must be less than " +
                           str(lamb.shape[1] - 1) + ".")
    if (source_locations_x is not None and
            source_locations_x[..., 0].min() <= 0):
        raise RuntimeError("The minimum x source "
                           "location in the first dimension must be "
                           "greater than 0.")
    if (receiver_locations_x is not None and
            receiver_locations_x[..., 0].min() <= 0):
        raise RuntimeError("The minimum x receiver "
                           "location in the first dimension must be "
                           "greater than 0.")
    if accuracy not in [2, 4]:
        raise RuntimeError("The accuracy must be 2 or 4.")

    (models, source_amplitudes, wavefields,
     pml_profiles, sources_i, receivers_i,
     dy, dx, dt, nt, n_batch,
     step_ratio, model_gradient_sampling_interval,
     accuracy, pml_width_list) = \
        setup_propagator([lamb, mu, buoyancy], 'elastic', grid_spacing, dt,
                         [vy_0, vx_0, sigmayy_0, sigmaxy_0, sigmaxx_0,
                          m_vyy_0, m_vyx_0, m_vxy_0, m_vxx_0,
                          m_sigmayyy_0, m_sigmaxyy_0,
                          m_sigmaxyx_0, m_sigmaxxx_0],
                         [source_amplitudes_y, source_amplitudes_x],
                         [source_locations_y, source_locations_x],
                         [receiver_locations_y, receiver_locations_x],
                         accuracy, pml_width, pml_freq, max_vel,
                         survey_pad,
                         origin, nt, model_gradient_sampling_interval,
                         freq_taper_frac, time_pad_frac)
    lamb, mu, buoyancy = models
    source_amplitudes_y, source_amplitudes_x = source_amplitudes
    (vy, vx, sigmayy, sigmaxy, sigmaxx,
     m_vyy, m_vyx, m_vxy, m_vxx,
     m_sigmayyy, m_sigmaxyy,
     m_sigmaxyx, m_sigmaxxx) = wavefields
    ay, ayh, ax, axh, by, byh, bx, bxh = pml_profiles
    sources_y_i, sources_x_i = sources_i
    receivers_y_i, receivers_x_i = receivers_i
    if any([s <= (accuracy + 1) * 2 for s in lamb.shape]):
        raise RuntimeError("The model must have at least " +
                           str((accuracy + 1) * 2 + 1) +
                           " elements in each dimension (including PML).")

    (vy, vx, sigmayy, sigmaxy, sigmaxx,
     m_vyy, m_vyx, m_vxy, m_vxx,
     m_sigmayyy, m_sigmaxyy,
     m_sigmaxyx, m_sigmaxxx, receiver_amplitudes_y,
     receiver_amplitudes_x) = \
        torch.ops.deepwave.elastic(
            lamb, mu, buoyancy, source_amplitudes_y, source_amplitudes_x,
            vy, vx, sigmayy, sigmaxy, sigmaxx,
            m_vyy, m_vyx, m_vxy, m_vxx,
            m_sigmayyy, m_sigmaxyy,
            m_sigmaxyx, m_sigmaxxx,
            ay, ayh, ax, axh, by, byh, bx, bxh,
            sources_y_i, sources_x_i, receivers_y_i, receivers_x_i,
            dy, dx, dt, nt,
            n_batch, step_ratio * model_gradient_sampling_interval,
            accuracy, pml_width_list[0], pml_width_list[1], pml_width_list[2],
            pml_width_list[3]
        )

    receiver_amplitudes_y = downsample_and_movedim(receiver_amplitudes_y,
                                                   step_ratio, freq_taper_frac,
                                                   time_pad_frac)
    receiver_amplitudes_x = downsample_and_movedim(receiver_amplitudes_x,
                                                   step_ratio, freq_taper_frac,
                                                   time_pad_frac)

    return (vy, vx, sigmayy, sigmaxy, sigmaxx,
            m_vyy, m_vyx, m_vxy, m_vxx,
            m_sigmayyy, m_sigmaxyy,
            m_sigmaxyx, m_sigmaxxx, receiver_amplitudes_y,
            receiver_amplitudes_x)
