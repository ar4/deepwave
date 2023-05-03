"""Scalar wave propagation module.

Finite difference in time (2nd order accurate) and space (user specifiable
order of accuracy from 2, 4, 6, or 8) is used to step the 2D scalar wave
equation

    v^2 d^2(w) / dx^2 - d^2(w) / dt^2 = v^2 f

forward in time, where v is the wavespeed, w is the wavefield, x are the
space dimensions, t is the time dimension, and f is a source term. A
Perfectly Matched Layer (PML) based on

    Pasalic, Damir, and Ray McGarry. "Convolutional perfectly matched
    layer for isotropic and anisotropic acoustic wave equations."
    SEG Technical Program Expanded Abstracts 2010.
    Society of Exploration Geophysicists, 2010. 2925-2929.

is used to optionally prevent reflections from one or more of the
boundaries.

The adjoint of this is also provided, so that backpropagation can be used
to calculate the gradient of outputs with respect to the inputs.

The only required inputs are the wavespeed model (`v`), the spatial grid
cell size (`dx`), the time step interval (`dt`), and either the source
term (`source_amplitudes` and `source_locations`) or the number of time
steps (`nt`). You may additionally provide other inputs, such as receiver
locations, and initial wavefields.

The outputs are the wavefields (including those related to the PML) at
the final time step, and the receiver amplitudes (which will be empty
if no receiver locations are provided).

The gradient of all of the outputs with respect to the wavespeed, the
source amplitudes, and the initial wavefields, may be calculated.
"""

import math
from typing import Optional, Union, List, Tuple
import torch
from torch import Tensor
from deepwave.common import (setup_propagator,
                             downsample_and_movedim)


class Scalar(torch.nn.Module):
    """A Module wrapper around :func:`scalar`.

    This is a convenience module that allows you to only specify
    `v` and `grid_spacing` once. They will then be added to the list of
    arguments passed to :func:`scalar` when you call the forward method.

    Note that a copy will be made of the provided wavespeed. Gradients
    will not backpropagate to the initial guess wavespeed that is
    provided. You can use the module's `v` attribute to access the
    wavespeed.

    Args:
        v:
            A 2D Tensor containing an initial guess of the wavespeed.
        grid_spacing:
            The spatial grid cell size, specified with a single real number
            (used for both dimensions) or a List or Tensor of length
            two (the length in each of the two dimensions).
        v_requires_grad:
            Optional bool specifying how to set the `requires_grad`
            attribute of the wavespeed, and so whether the necessary
            data should be stored to calculate the gradient with respect
            to `v` during backpropagation. Default False.
    """
    def __init__(self, v: Tensor,
                 grid_spacing: Union[int, float, List[float],
                                     Tensor],
                 v_requires_grad: bool = False) -> None:
        super().__init__()
        if v.ndim != 2:
            raise RuntimeError("v must have two dimensions")
        self.v = torch.nn.Parameter(v, requires_grad=v_requires_grad)
        self.grid_spacing = grid_spacing

    def forward(self, dt: float, source_amplitudes: Optional[Tensor] = None,
                source_locations: Optional[Tensor] = None,
                receiver_locations: Optional[Tensor] = None,
                accuracy: int = 4, pml_width: Union[int, List[int]] = 20,
                pml_freq: Optional[float] = None,
                max_vel: Optional[float] = None,
                survey_pad: Optional[Union[int,
                                           List[Optional[int]]]] = None,
                wavefield_0: Optional[Tensor] = None,
                wavefield_m1: Optional[Tensor] = None,
                psiy_m1: Optional[Tensor] = None,
                psix_m1: Optional[Tensor] = None,
                zetay_m1: Optional[Tensor] = None,
                zetax_m1: Optional[Tensor] = None,
                origin: Optional[List[int]] = None,
                nt: Optional[int] = None,
                model_gradient_sampling_interval: int = 1) -> Tuple[Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor,
                                                                    Tensor]:
        """Perform forward propagation/modelling.

        The inputs are the same as for :func:`scalar` except that `v` and
        `grid_spacing` do not need to be provided again. See :func:`scalar`
        for a description of the inputs and outputs.
        """
        return scalar(self.v, self.grid_spacing, dt,
                      source_amplitudes=source_amplitudes,
                      source_locations=source_locations,
                      receiver_locations=receiver_locations,
                      accuracy=accuracy, pml_width=pml_width,
                      pml_freq=pml_freq,
                      max_vel=max_vel,
                      survey_pad=survey_pad, wavefield_0=wavefield_0,
                      wavefield_m1=wavefield_m1,
                      psiy_m1=psiy_m1, psix_m1=psix_m1,
                      zetay_m1=zetay_m1, zetax_m1=zetax_m1,
                      origin=origin, nt=nt,
                      model_gradient_sampling_interval=
                      model_gradient_sampling_interval)


def scalar(v: Tensor,
           grid_spacing: Union[int, float, List[float],
                               Tensor],
           dt: float,
           source_amplitudes: Optional[Tensor] = None,
           source_locations: Optional[Tensor] = None,
           receiver_locations: Optional[Tensor] = None,
           accuracy: int = 4, pml_width: Union[int, List[int]] = 20,
           pml_freq: Optional[float] = None,
           max_vel: Optional[float] = None,
           survey_pad: Optional[Union[int, List[Optional[int]]]] = None,
           wavefield_0: Optional[Tensor] = None,
           wavefield_m1: Optional[Tensor] = None,
           psiy_m1: Optional[Tensor] = None,
           psix_m1: Optional[Tensor] = None,
           zetay_m1: Optional[Tensor] = None,
           zetax_m1: Optional[Tensor] = None,
           origin: Optional[List[int]] = None, nt: Optional[int] = None,
           model_gradient_sampling_interval: int = 1,
           freq_taper_frac: float = 0.0,
           time_pad_frac: float = 0.0) -> Tuple[Tensor, Tensor,
                                                Tensor, Tensor,
                                                Tensor, Tensor,
                                                Tensor]:
    """Scalar wave propagation (functional interface).

    This function performs forward modelling with the scalar wave equation.
    The outputs are differentiable with respect to the wavespeed, the
    source amplitudes, and the initial wavefields.

    For computational performance, multiple shots may be propagated
    simultaneously.

    Args:
        v:
            A 2D Tensor containing the wavespeed model. Unlike the module
            interface (:class:`Scalar`), in this functional interface a copy is
            not made of the model, so gradients will propagate back into
            the provided Tensor.
        grid_spacing:
            The spatial grid cell size, specified with a single real number
            (used for both dimensions) or a List or Tensor of length
            two (the length in each of the two dimensions).
        dt:
            A float specifying the time step interval of the input and
            output (internally a smaller interval may be used in
            propagation to obey the CFL condition for stability).
        source_amplitudes:
            A Tensor with dimensions [shot, source, time]. If two shots
            are being propagated simultaneously, each containing three
            sources of one hundred time samples, it would have shape
            [2, 3, 100]. The time length will be the number of time steps
            in the simulation. Optional. If provided, `source_locations` must
            also be specified. If not provided, `nt` must be specified.
        source_locations:
            A Tensor with dimensions [shot, source, 2], containing the
            index in the two spatial dimensions of the cell that each
            source is located in, relative to the origin of the model.
            Optional. Must be provided if `source_amplitudes` is. It should
            have torch.long (int64) datatype. The location of each source
            must be unique within the same shot (you cannot have two
            sources in the same shot that both have location [1, 2], for
            example).
        receiver_locations:
            A Tensor with dimensions [shot, receiver, 2], containing
            the coordinates of the cell containing each receiver. Optional.
            It should have torch.long (int64) datatype. If not provided,
            the output `receiver_amplitudes` Tensor will be empty. If
            backpropagation will be performed, the location of each
            receiver must be unique within the same shot.
        accuracy:
            An int specifying the finite difference order of accuracy. Possible
            values are 2, 4, 6, and 8, with larger numbers resulting in more
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
            `pml_width=[20, 20, 0, 20]`. The wavespeed in the PML region
            is obtained by replicating the values on the edge of the
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
        wavefield_0:
            A Tensor specifying the initial wavefield at time step 0. It
            should have three dimensions, with the first dimension being shot
            and the subsequent two corresponding to the two spatial
            dimensions. The spatial shape should be equal to the simulation
            domain, which is the extracted model plus the PML. If two shots
            are being propagated simultaneously in a region of size [20, 30]
            extracted from the model for the simulation, and
            `pml_width=[1, 2, 3, 4]`, then `wavefield_0` should be of shape
            [2, 23, 37]. Optional, default all zeros.
        wavefield_m1:
            A Tensor specifying the initial wavefield at time step -1 (using
            Deepwave's internal time step interval, which may be smaller than
            the user provided one to obey the CFL condition). See the entry for
            `wavefield_0` for more details.
        psiy_m1, psix_m1, zetay_m1, zetax_m1:
            Tensor specifying the initial value for this PML-related
            wavefield at time step -1. See the entry for `wavefield_0`
            for more details.
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

            wavefield_nt:
                A Tensor containing the wavefield at the final time step.
            wavefield_ntm1:
                A Tensor containing the wavefield at the second-to-last
                time step (using Deepwave's internal time interval, which may
                be smaller than the user provided one to obey the CFL
                condition).
            psiy_ntm1, psix_ntm1, zetay_ntm1, zetax_ntm1:
                Tensor containing the wavefield related to the PML at the
                second-to-last time step.
            receiver_amplitudes:
                A Tensor of dimensions [shot, receiver, time] containing
                the receiver amplitudes recorded at the provided receiver
                locations. If no receiver locations were specified then
                this Tensor will be empty.

    """
    (models, source_amplitudes_l, wavefields,
     pml_profiles, sources_i_l, receivers_i_l,
     dy, dx, dt, nt, n_batch,
     step_ratio, model_gradient_sampling_interval,
     accuracy, pml_width_list) = \
        setup_propagator([v], 'scalar', grid_spacing, dt,
                         [wavefield_0, wavefield_m1, psiy_m1, psix_m1,
                          zetay_m1, zetax_m1],
                         [source_amplitudes],
                         [source_locations], [receiver_locations],
                         accuracy, pml_width, pml_freq, max_vel,
                         survey_pad,
                         origin, nt, model_gradient_sampling_interval,
                         freq_taper_frac, time_pad_frac, jit=True)
    v = models[0]
    wfc, wfp, psiy, psix, zetay, zetax = wavefields
    source_amplitudes = source_amplitudes_l[0]
    sources_i = sources_i_l[0]
    receivers_i = receivers_i_l[0]
    ay, ax, by, bx = pml_profiles

    if sources_i is not None:
        sources_i = (
            sources_i +
            (torch.arange(n_batch, device=v.device) * v.numel())[:, None]
        ).flatten()
    if receivers_i is not None:
        receivers_i = (
            receivers_i +
            (torch.arange(n_batch, device=v.device) * v.numel())[:, None]
        ).flatten()

    wfc = None
    wfp = None
    psiy = None
    psix = None
    zetay = None
    zetax = None

    wfc, wfp, psiy, psix, zetay, zetax, receiver_amplitudes = \
        scalar_prop(
            v, source_amplitudes, wfc, wfp, psiy, psix, zetay, zetax,
            ay, ax, by, bx, sources_i, receivers_i, dy, dx, dt, nt,
            n_batch, step_ratio * model_gradient_sampling_interval,
            accuracy, pml_width_list
        )

    receiver_amplitudes = downsample_and_movedim(receiver_amplitudes,
                                                 step_ratio, freq_taper_frac,
                                                 time_pad_frac, jit=True)

    return (wfc, wfp, psiy, psix, zetay, zetax, receiver_amplitudes)


def scalar_prop(v: Tensor, source_amplitudes: Optional[Tensor],
                wfc0: Optional[Tensor], wfp0: Optional[Tensor],
                psiy0: Optional[Tensor], psix0: Optional[Tensor],
                zetay0: Optional[Tensor], zetax0: Optional[Tensor],
                ay: Tensor, ax: Tensor, by: Tensor, bx: Tensor,
                sources_i: Optional[Tensor],
                receivers_i: Optional[Tensor],
                dy: float, dx: float, dt: float, nt: int,
                n_batch: int, step_ratio: int,
                accuracy: int,
                pml_width: List[int]) -> Tuple[Tensor, Tensor,
                                               Tensor, Tensor,
                                               Tensor, Tensor,
                                               Tensor]:

    device = v.device
    dtype = v.dtype
    ny = v.shape[0]
    nx = v.shape[1]
    size_with_batch = [n_batch, ny, nx]
    fd_pad = accuracy // 2
    pml_regionsy0 = fd_pad
    pml_regionsy1 = min(pml_width[0] + 2 * fd_pad, ny - fd_pad)
    pml_regionsy2 = max(pml_regionsy1, ny - pml_width[1] - 2 * fd_pad)
    pml_regionsy3 = ny - fd_pad
    pml_regionsy = torch.tensor([pml_regionsy0, pml_regionsy1, pml_regionsy2,
                                 pml_regionsy3]).long().to(device)
    pml_regionsx0 = fd_pad
    pml_regionsx1 = min(pml_width[2] + 2 * fd_pad, nx - fd_pad)
    pml_regionsx2 = max(pml_regionsx1, nx - pml_width[3] - 2 * fd_pad)
    pml_regionsx3 = nx - fd_pad
    pml_regionsx = torch.tensor([pml_regionsx0, pml_regionsx1, pml_regionsx2,
                                 pml_regionsx3]).long().to(device)

    wfc = create_or_pad(wfc0, 0, device, dtype, size_with_batch)
    wfp = create_or_pad(wfp0, 0, device, dtype, size_with_batch)
    psiy = create_or_pad(psiy0, 0, device, dtype, size_with_batch)
    psix = create_or_pad(psix0, 0, device, dtype, size_with_batch)
    zetay = create_or_pad(zetay0, 0, device, dtype, size_with_batch)
    zetax = create_or_pad(zetax0, 0, device, dtype, size_with_batch)
    receiver_amplitudes = torch.tensor(0, device=device, dtype=dtype)
    if receivers_i is not None:
        receiver_amplitudes = torch.zeros(nt, n_batch,
                                          receivers_i.numel() // n_batch,
                                          device=device, dtype=dtype)

    zero_interior(psiy, 0, pml_width, True)
    zero_interior(psix, 0, pml_width, False)
    zero_interior(zetay, 0, pml_width, True)
    zero_interior(zetax, 0, pml_width, False)

    fd_coeffs1y, fd_coeffs2y = set_fd_coeffs(accuracy, dy, ny + accuracy, device, dtype)
    fd_coeffs1x, fd_coeffs2x = set_fd_coeffs(accuracy, dx, nx + accuracy, device, dtype)
    daydy = diff(ay, fd_coeffs1y, accuracy)
    daxdx = diff(ax, fd_coeffs1x, accuracy)
    dbydy = diff(by, fd_coeffs1y, accuracy)
    dbxdx = diff(bx, fd_coeffs1x, accuracy)

    dt2 = dt**2
    ay = ay[None, :, None]
    ax = ax[None, None, :]
    by = by[None, :, None]
    bx = bx[None, None, :]
    daydy = daydy[None, :, None]
    daxdx = daxdx[None, None, :]
    dbydy = dbydy[None, :, None]
    dbxdx = dbxdx[None, None, :]

    for t in range(nt):
        if t % step_ratio == 0:
            wfn, psiy, psix, zetay, zetax = \
                forward_kernel(wfc, wfp, psiy, psix,
                               zetay, zetax, v, ay, ax, by, bx,
                               daydy, daxdx, dbydy, dbxdx,
                               fd_coeffs1y, fd_coeffs2y,
                               fd_coeffs1x, fd_coeffs2x,
                               accuracy, dt2, pml_regionsy,
                               pml_regionsx)
            if source_amplitudes is not None and sources_i is not None:
                add_sources(wfn, source_amplitudes[t], sources_i)
            if receivers_i is not None:
                receiver_amplitudes[t] = record_receivers(wfc, receivers_i)
            wfc, wfp = wfn, wfc
        else:
            wfn, psiy, psix, zetay, zetax = \
                forward_kernel(wfc, wfp, psiy, psix,
                               zetay, zetax, v.detach(), ay, ax, by, bx,
                               daydy, daxdx, dbydy, dbxdx,
                               fd_coeffs1y, fd_coeffs2y,
                               fd_coeffs1x, fd_coeffs2x,
                               accuracy, dt2, pml_regionsy,
                               pml_regionsx)
            if source_amplitudes is not None and sources_i is not None:
                add_sources(wfn, source_amplitudes.detach()[t], sources_i)
            if receivers_i is not None:
                receiver_amplitudes[t] = record_receivers(wfc, receivers_i)
            wfc, wfp = wfn, wfc

    return (
        wfc,
        wfp,
        psiy,
        psix,
        zetay,
        zetax,
        receiver_amplitudes
    )


def create_or_pad(tensor: Optional[Tensor], fd_pad: int,
                  device: torch.device, dtype: torch.dtype,
                  size: List[int]) -> Tensor:
    if tensor is None:
        return torch.zeros(size[0], size[1], size[2], device=device,
                           dtype=dtype)
    else:
        return torch.nn.functional.pad(
            tensor, (fd_pad, fd_pad, fd_pad, fd_pad)
        )


def zero_interior(tensor: Tensor, fd_pad: int, pml_width: List[int],
                  y: bool) -> None:
    ny = tensor.shape[1]
    nx = tensor.shape[2]
    if y:
        tensor[:, fd_pad + pml_width[0] : ny - pml_width[1] - fd_pad] = 0
    else:
        tensor[:, :, fd_pad + pml_width[2] : nx - pml_width[3] - fd_pad] = 0


def set_fd_coeffs(accuracy: int, dx: float, n: int, device: torch.device,
                  dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    dx2 = dx * dx
    w = 2 * math.pi * torch.fft.rfftfreq(n, device=device)
    if accuracy == 2:
        fd_coeffs1 = 1 / 2 * 2 * 1j * torch.sin(1 * w) / dx
        fd_coeffs2 = (-2 + 2 * torch.cos(1 * w)) / dx2
    elif accuracy == 4:
        fd_coeffs1 = (
            8 / 12 * 2 * 1j * torch.sin(1 * w)
            -1 / 12 * 2 * 1j * torch.sin(2 * w)
        ) / dx
        fd_coeffs2 = (
            -5 / 2
            +4 / 3 * 2 * torch.cos(1 * w)
            -1 / 12 * 2 * torch.cos(2 * w)
        ) / dx2
#    elif accuracy == 6:
#        fd_coeffs1 = [
#            3 / 4 / dx,
#            -3 / 20 / dx,
#            1 / 60 / dx
#        ]
#        fd_coeffs2 = [
#            -49 / 18 / dx2,
#            3 / 2 / dx2,
#            -3 / 20 / dx2,
#            1 / 90 / dx2
#        ]
#    else:
#        fd_coeffs1 = [
#            4 / 5 / dx,
#            -1 / 5 / dx,
#            4 / 105 / dx,
#            -1 / 280 / dx
#        ]
#        fd_coeffs2 = [
#            -205 / 72 / dx2,
#            8 / 5 / dx2,
#            -1 / 5 / dx2,
#            8 / 315 / dx2,
#            -1 / 560 / dx2
#        ]
    return (
        fd_coeffs1, fd_coeffs2
        #torch.tensor(fd_coeffs1).to(dtype).to(device),
        #torch.tensor(fd_coeffs2).to(dtype).to(device)
    )


def diff(a: Tensor, fd_coeffs: Tensor, accuracy: int) -> Tensor:
    fd_pad = accuracy // 2
    a = torch.nn.functional.pad(a, (fd_pad, fd_pad))
    return torch.fft.irfft(fd_coeffs * torch.fft.rfft(a), n=len(a))[fd_pad:-fd_pad]


def diffy12(a: Tensor, fd_coeffs1: Tensor, fd_coeffs2: Tensor, accuracy: int) -> Tuple[Tensor, Tensor]:
    fd_pad = accuracy // 2
    a = torch.nn.functional.pad(a, (0, 0, fd_pad, fd_pad))
    A = torch.fft.rfft(a, dim=1)
    return (
        torch.fft.irfft(fd_coeffs1[:, None] * A, n=a.shape[1], dim=1)[:, fd_pad:-fd_pad],
        torch.fft.irfft(fd_coeffs2[:, None] * A, n=a.shape[1], dim=1)[:, fd_pad:-fd_pad]
    )


def diffx12(a: Tensor, fd_coeffs1: Tensor, fd_coeffs2: Tensor, accuracy: int) -> Tuple[Tensor, Tensor]:
    fd_pad = accuracy // 2
    a = torch.nn.functional.pad(a, (fd_pad, fd_pad))
    A = torch.fft.rfft(a)
    return (
        torch.fft.irfft(fd_coeffs1 * A, n=a.shape[2])[:, :, fd_pad:-fd_pad],
        torch.fft.irfft(fd_coeffs2 * A, n=a.shape[2])[:, :, fd_pad:-fd_pad]
    )


def diffy1(a: Tensor, fd_coeffs1: Tensor, accuracy: int) -> Tensor:
    fd_pad = accuracy // 2
    a = torch.nn.functional.pad(a, (0, 0, fd_pad, fd_pad))
    A = torch.fft.rfft(a, dim=1)
    return (
        torch.fft.irfft(fd_coeffs1[:, None] * A, n=a.shape[1], dim=1)[:, fd_pad:-fd_pad]
    )


def diffx1(a: Tensor, fd_coeffs1: Tensor, accuracy: int) -> Tensor:
    fd_pad = accuracy // 2
    a = torch.nn.functional.pad(a, (fd_pad, fd_pad))
    A = torch.fft.rfft(a)
    return (
        torch.fft.irfft(fd_coeffs1 * A, n=a.shape[2])[:, :, fd_pad:-fd_pad]
    )


@torch.jit.script
def add_sources(wf: Tensor, f: Tensor, sources_i: Tensor):
    wf.view(-1)[sources_i] += f.view(-1)


@torch.jit.script
def record_receivers(wf: Tensor, receivers_i: Tensor) -> Tensor:
    return wf.view(-1)[receivers_i]


@torch.jit.script
def forward_kernel(wfc: Tensor, wfp: Tensor,
                   psiy: Tensor, psix: Tensor,
                   zetay: Tensor, zetax: Tensor,
                   v: Tensor, ay: Tensor, ax: Tensor,
                   by: Tensor, bx: Tensor,
                   daydy: Tensor, daxdx: Tensor,
                   dbydy: Tensor, dbxdx: Tensor,
                   fd_coeffs1y: Tensor, fd_coeffs2y: Tensor,
                   fd_coeffs1x: Tensor, fd_coeffs2x: Tensor,
                   accuracy: int, dt2: float, pml_regionsy: Tensor,
                   pml_regionsx: Tensor) -> Tuple[Tensor,
                                                  Tensor, Tensor,
                                                  Tensor, Tensor]:

    dwfcdy, d2wfcdy2 = diffy12(wfc, fd_coeffs1y, fd_coeffs2y, accuracy)
    dwfcdx, d2wfcdx2 = diffx12(wfc, fd_coeffs1x, fd_coeffs2x, accuracy)

    tmpy = (
        (1 + by) * d2wfcdy2 +
        dbydy * dwfcdy +
        daydy * psiy +
        ay * diffy1(psiy, fd_coeffs1y, accuracy)
    )
    tmpx = (
        (1 + bx) * d2wfcdx2 +
        dbxdx * dwfcdx +
        daxdx * psix +
        ax * diffx1(psix, fd_coeffs1x, accuracy)
    )
    return (
        v[None]**2 * dt2 * (
            (1 + by) * tmpy + ay * zetay + 
            (1 + bx) * tmpx + ax * zetax
        ) + 2 * wfc - wfp,
        by * dwfcdy + ay * psiy,
        bx * dwfcdx + ax * psix,
        by * tmpy + ay * zetay,
        bx * tmpx + ax * zetax
    )
