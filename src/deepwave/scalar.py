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

from typing import Optional, Union, List, Tuple
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
import deepwave
from deepwave.common import (setup_propagator, downsample_and_movedim,
                             zero_interior, create_or_pad, diff)


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

    def __init__(self,
                 v: Tensor,
                 grid_spacing: Union[int, float, List[float], Tensor],
                 v_requires_grad: bool = False) -> None:
        super().__init__()
        if v.ndim != 2:
            raise RuntimeError("v must have two dimensions")
        self.v = torch.nn.Parameter(v, requires_grad=v_requires_grad)
        self.grid_spacing = grid_spacing

    def forward(
        self,
        dt: float,
        source_amplitudes: Optional[Tensor] = None,
        source_locations: Optional[Tensor] = None,
        receiver_locations: Optional[Tensor] = None,
        accuracy: int = 4,
        pml_width: Union[int, List[int]] = 20,
        pml_freq: Optional[float] = None,
        max_vel: Optional[float] = None,
        survey_pad: Optional[Union[int, List[Optional[int]]]] = None,
        wavefield_0: Optional[Tensor] = None,
        wavefield_m1: Optional[Tensor] = None,
        psiy_m1: Optional[Tensor] = None,
        psix_m1: Optional[Tensor] = None,
        zetay_m1: Optional[Tensor] = None,
        zetax_m1: Optional[Tensor] = None,
        origin: Optional[List[int]] = None,
        nt: Optional[int] = None,
        model_gradient_sampling_interval: int = 1
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Perform forward propagation/modelling.

        The inputs are the same as for :func:`scalar` except that `v` and
        `grid_spacing` do not need to be provided again. See :func:`scalar`
        for a description of the inputs and outputs.
        """
        return scalar(
            self.v,
            self.grid_spacing,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            accuracy=accuracy,
            pml_width=pml_width,
            pml_freq=pml_freq,
            max_vel=max_vel,
            survey_pad=survey_pad,
            wavefield_0=wavefield_0,
            wavefield_m1=wavefield_m1,
            psiy_m1=psiy_m1,
            psix_m1=psix_m1,
            zetay_m1=zetay_m1,
            zetax_m1=zetax_m1,
            origin=origin,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval)


def scalar(
    v: Tensor,
    grid_spacing: Union[int, float, List[float], Tensor],
    dt: float,
    source_amplitudes: Optional[Tensor] = None,
    source_locations: Optional[Tensor] = None,
    receiver_locations: Optional[Tensor] = None,
    accuracy: int = 4,
    pml_width: Union[int, List[int]] = 20,
    pml_freq: Optional[float] = None,
    max_vel: Optional[float] = None,
    survey_pad: Optional[Union[int, List[Optional[int]]]] = None,
    wavefield_0: Optional[Tensor] = None,
    wavefield_m1: Optional[Tensor] = None,
    psiy_m1: Optional[Tensor] = None,
    psix_m1: Optional[Tensor] = None,
    zetay_m1: Optional[Tensor] = None,
    zetax_m1: Optional[Tensor] = None,
    origin: Optional[List[int]] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
            value is used). Optional, default None. Cannot be specified
            if origin is also specified.
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
            is [10, 15]. Optional, default [0, 0]. Cannot be specified if
            survey_pad is also specified.
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
     dy, dx, dt, nt, n_shots,
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
                         freq_taper_frac, time_pad_frac, time_taper)
    v = models[0]
    wfc, wfp, psiy, psix, zetay, zetax = wavefields
    source_amplitudes = source_amplitudes_l[0]
    sources_i = sources_i_l[0]
    receivers_i = receivers_i_l[0]
    ay, ax, by, bx = pml_profiles
    dbydy = diff(by, accuracy, dy)
    dbxdx = diff(bx, accuracy, dx)

    (wfc, wfp, psiy, psix, zetay, zetax, receiver_amplitudes) = \
        scalar_func(
            v, source_amplitudes, wfc, wfp, psiy, psix, zetay, zetax, ay, ax, by, bx, dbydy, dbxdx, sources_i, receivers_i, dy, dx, dt, nt, step_ratio * model_gradient_sampling_interval, accuracy, pml_width_list, n_shots
        )

    receiver_amplitudes = downsample_and_movedim(receiver_amplitudes,
                                                 step_ratio, freq_taper_frac,
                                                 time_pad_frac, time_taper)

    return wfc, wfp, psiy, psix, zetay, zetax, receiver_amplitudes


class ScalarForwardFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v, source_amplitudes, wfc, wfp, psiy, psix, zetay, zetax,
                ay, ax, by, bx, dbydy, dbxdx, sources_i, receivers_i, dy, dx,
                dt, nt, step_ratio, accuracy, pml_width, n_shots):

        if (v.requires_grad or source_amplitudes.requires_grad
                or wfc.requires_grad or wfp.requires_grad or psiy.requires_grad
                or psix.requires_grad or zetay.requires_grad
                or zetax.requires_grad):
            ctx.save_for_backward(v, ay, ax, by, bx, dbydy, dbxdx, sources_i,
                                  receivers_i, source_amplitudes, wfc, wfp,
                                  psiy, psix, zetay, zetax)
            ctx.dy = dy
            ctx.dx = dx
            ctx.dt = dt
            ctx.nt = nt
            ctx.n_shots = n_shots
            ctx.step_ratio = step_ratio
            ctx.accuracy = accuracy
            ctx.pml_width = pml_width
            ctx.source_amplitudes_requires_grad = source_amplitudes.requires_grad

        v = v.contiguous()
        source_amplitudes = source_amplitudes.contiguous()
        ay = ay.contiguous()
        ax = ax.contiguous()
        by = by.contiguous()
        bx = bx.contiguous()
        dbydy = dbydy.contiguous()
        dbxdx = dbxdx.contiguous()
        sources_i = sources_i.contiguous()
        receivers_i = receivers_i.contiguous()

        fd_pad = accuracy // 2
        size_with_batch = (n_shots, *v.shape)
        wfc = create_or_pad(wfc, fd_pad, v.device, v.dtype, size_with_batch)
        wfp = create_or_pad(wfp, fd_pad, v.device, v.dtype, size_with_batch)
        psiy = create_or_pad(psiy, fd_pad, v.device, v.dtype, size_with_batch)
        psix = create_or_pad(psix, fd_pad, v.device, v.dtype, size_with_batch)
        zetay = create_or_pad(zetay, fd_pad, v.device, v.dtype,
                              size_with_batch)
        zetax = create_or_pad(zetax, fd_pad, v.device, v.dtype,
                              size_with_batch)
        zero_interior(psiy, 2 * fd_pad, pml_width, True)
        zero_interior(psix, 2 * fd_pad, pml_width, False)
        zero_interior(zetay, 2 * fd_pad, pml_width, True)
        zero_interior(zetax, 2 * fd_pad, pml_width, False)

        device = v.device
        dtype = v.dtype
        ny = v.shape[0]
        nx = v.shape[1]
        n_sources_per_shot = sources_i.numel() // n_shots
        n_receivers_per_shot = receivers_i.numel() // n_shots
        psiyn = torch.zeros_like(psiy)
        psixn = torch.zeros_like(psix)
        dwdv = torch.empty(0, device=device, dtype=dtype)
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
        pml_y0 = min(pml_width[0] + 2 * fd_pad, ny - fd_pad)
        pml_y1 = max(pml_y0, ny - pml_width[1] - 2 * fd_pad)
        pml_x0 = min(pml_width[2] + 2 * fd_pad, nx - fd_pad)
        pml_x1 = max(pml_x0, nx - pml_width[3] - 2 * fd_pad)

        if v.is_cuda:
            aux = v.get_device()
            if v.requires_grad:
                dwdv.resize_(nt // step_ratio, n_shots, *v.shape)
                dwdv.fill_(0)
            if receivers_i.numel() > 0:
                receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
                receiver_amplitudes.fill_(0)
            if dtype == torch.float32:
                if accuracy == 2:
                    forward = deepwave.dll_cuda.scalar_iso_2_float_forward
                elif accuracy == 4:
                    forward = deepwave.dll_cuda.scalar_iso_4_float_forward
                elif accuracy == 6:
                    forward = deepwave.dll_cuda.scalar_iso_6_float_forward
                else:
                    forward = deepwave.dll_cuda.scalar_iso_8_float_forward
            else:
                if accuracy == 2:
                    forward = deepwave.dll_cuda.scalar_iso_2_double_forward
                elif accuracy == 4:
                    forward = deepwave.dll_cuda.scalar_iso_4_double_forward
                elif accuracy == 6:
                    forward = deepwave.dll_cuda.scalar_iso_6_double_forward
                else:
                    forward = deepwave.dll_cuda.scalar_iso_8_double_forward
        else:
            if deepwave.use_openmp:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if v.requires_grad:
                dwdv.resize_(n_shots, nt // step_ratio, *v.shape)
                dwdv.fill_(0)
            if receivers_i.numel() > 0:
                receiver_amplitudes.resize_(n_shots, nt, n_receivers_per_shot)
                receiver_amplitudes.fill_(0)
            if dtype == torch.float32:
                if accuracy == 2:
                    forward = deepwave.dll_cpu.scalar_iso_2_float_forward
                elif accuracy == 4:
                    forward = deepwave.dll_cpu.scalar_iso_4_float_forward
                elif accuracy == 6:
                    forward = deepwave.dll_cpu.scalar_iso_6_float_forward
                else:
                    forward = deepwave.dll_cpu.scalar_iso_8_float_forward
            else:
                if accuracy == 2:
                    forward = deepwave.dll_cpu.scalar_iso_2_double_forward
                elif accuracy == 4:
                    forward = deepwave.dll_cpu.scalar_iso_4_double_forward
                elif accuracy == 6:
                    forward = deepwave.dll_cpu.scalar_iso_6_double_forward
                else:
                    forward = deepwave.dll_cpu.scalar_iso_8_double_forward

        if wfc.numel() > 0 and nt > 0:
            forward(v.data_ptr(), source_amplitudes.data_ptr(), wfc.data_ptr(),
                    wfp.data_ptr(), psiy.data_ptr(), psix.data_ptr(),
                    psiyn.data_ptr(), psixn.data_ptr(), zetay.data_ptr(),
                    zetax.data_ptr(), dwdv.data_ptr(),
                    receiver_amplitudes.data_ptr(), ay.data_ptr(),
                    ax.data_ptr(), by.data_ptr(), bx.data_ptr(),
                    dbydy.data_ptr(), dbxdx.data_ptr(), sources_i.data_ptr(),
                    receivers_i.data_ptr(), 1 / dy, 1 / dx, 1 / dy**2,
                    1 / dx**2, dt**2, nt, n_shots, ny, nx, n_sources_per_shot,
                    n_receivers_per_shot, step_ratio, v.requires_grad, pml_y0,
                    pml_y1, pml_x0, pml_x1, aux)

        ctx.dwdv = dwdv

        s = (slice(None), slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        if nt % 2 == 0:
            return (wfc[s], wfp[s], psiy[s], psix[s], zetay[s], zetax[s],
                    receiver_amplitudes)
        else:
            return (wfp[s], wfc[s], psiyn[s], psixn[s], zetay[s], zetax[s],
                    receiver_amplitudes)

    @staticmethod
    def backward(ctx, gwfc, gwfp, gpsiy, gpsix, gzetay, gzetax, grad_r):
        (v, ay, ax, by, bx, dbydy, dbxdx, sources_i, receivers_i,
         source_amplitudes, wfc, wfp, psiy, psix, zetay,
         zetax) = ctx.saved_tensors
        dy = ctx.dy
        dx = ctx.dx
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        pml_width = ctx.pml_width
        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad
        dwdv = ctx.dwdv

        return ScalarBackwardFunc.apply(
            gwfc, gwfp, gpsiy, gpsix, gzetay, gzetax, grad_r, v, ay, ax, by,
            bx, dbydy, dbxdx, sources_i, receivers_i, dwdv, source_amplitudes,
            wfc, wfp, psiy, psix, zetay, zetax, dy, dx, dt, nt, n_shots,
            step_ratio, accuracy, pml_width,
            source_amplitudes_requires_grad) + (None, ) * 16


class ScalarBackwardFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, gwfc, gwfp, gpsiy, gpsix, gzetay, gzetax, grad_r, v, ay,
                ax, by, bx, dbydy, dbxdx, sources_i, receivers_i, dwdv,
                source_amplitudes, wfc, wfp, psiy, psix, zetay, zetax, dy, dx,
                dt, nt, n_shots, step_ratio, accuracy, pml_width,
                source_amplitudes_requires_grad):

        ctx.save_for_backward(gwfc, gwfp, gpsiy, gpsix, gzetay, gzetax, grad_r,
                              v, ay, ax, by, bx, dbydy, dbxdx, sources_i,
                              receivers_i, source_amplitudes, wfc, wfp, psiy,
                              psix, zetay, zetax)
        ctx.dy = dy
        ctx.dx = dx
        ctx.dt = dt
        ctx.nt = nt
        ctx.n_shots = n_shots
        ctx.step_ratio = step_ratio
        ctx.accuracy = accuracy
        ctx.pml_width = pml_width
        ctx.source_amplitudes_requires_grad = source_amplitudes.requires_grad

        v = v.contiguous()
        grad_r = grad_r.contiguous()
        ay = ay.contiguous()
        ax = ax.contiguous()
        by = by.contiguous()
        bx = bx.contiguous()
        dbydy = dbydy.contiguous()
        dbxdx = dbxdx.contiguous()
        sources_i = sources_i.contiguous()
        receivers_i = receivers_i.contiguous()
        dwdv = dwdv.contiguous()

        device = v.device
        dtype = v.dtype
        ny = v.shape[0]
        nx = v.shape[1]
        n_sources_per_shot = sources_i.numel() // n_shots
        n_receivers_per_shot = receivers_i.numel() // n_shots
        fd_pad = accuracy // 2

        size_with_batch = (n_shots, *v.shape)
        gwfc = create_or_pad(gwfc, fd_pad, v.device, v.dtype, size_with_batch)
        gwfp = create_or_pad(gwfp, fd_pad, v.device, v.dtype, size_with_batch)
        gpsiy = create_or_pad(gpsiy, fd_pad, v.device, v.dtype,
                              size_with_batch)
        gpsix = create_or_pad(gpsix, fd_pad, v.device, v.dtype,
                              size_with_batch)
        gzetay = create_or_pad(gzetay, fd_pad, v.device, v.dtype,
                               size_with_batch)
        gzetax = create_or_pad(gzetax, fd_pad, v.device, v.dtype,
                               size_with_batch)
        zero_interior(gpsiy, 2 * fd_pad, pml_width, True)
        zero_interior(gpsix, 2 * fd_pad, pml_width, False)
        zero_interior(gzetay, 2 * fd_pad, pml_width, True)
        zero_interior(gzetax, 2 * fd_pad, pml_width, False)

        gpsiyn = torch.zeros_like(gpsiy)
        gpsixn = torch.zeros_like(gpsix)
        gzetayn = torch.zeros_like(gzetay)
        gzetaxn = torch.zeros_like(gzetax)
        grad_v = torch.empty(0, device=device, dtype=dtype)
        grad_v_tmp = torch.empty(0, device=device, dtype=dtype)
        grad_v_tmp_ptr = grad_v.data_ptr()
        if v.requires_grad:
            grad_v.resize_(*v.shape)
            grad_v.fill_(0)
            grad_v_tmp_ptr = grad_v.data_ptr()
        grad_f = torch.empty(0, device=device, dtype=dtype)
        pml_y0 = min(pml_width[0] + 3 * fd_pad, ny - fd_pad)
        pml_y1 = max(pml_y0, ny - pml_width[1] - 3 * fd_pad)
        pml_x0 = min(pml_width[2] + 3 * fd_pad, nx - fd_pad)
        pml_x1 = max(pml_x0, nx - pml_width[3] - 3 * fd_pad)

        if v.is_cuda:
            aux = v.get_device()
            if v.requires_grad and n_shots > 1:
                grad_v_tmp.resize_(n_shots, *v.shape)
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
            if source_amplitudes_requires_grad:
                grad_f.resize_(nt, n_shots, n_sources_per_shot)
                grad_f.fill_(0)
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = deepwave.dll_cuda.scalar_iso_2_float_backward
                elif accuracy == 4:
                    backward = deepwave.dll_cuda.scalar_iso_4_float_backward
                elif accuracy == 6:
                    backward = deepwave.dll_cuda.scalar_iso_6_float_backward
                else:
                    backward = deepwave.dll_cuda.scalar_iso_8_float_backward
            else:
                if accuracy == 2:
                    backward = deepwave.dll_cuda.scalar_iso_2_double_backward
                elif accuracy == 4:
                    backward = deepwave.dll_cuda.scalar_iso_4_double_backward
                elif accuracy == 6:
                    backward = deepwave.dll_cuda.scalar_iso_6_double_backward
                else:
                    backward = deepwave.dll_cuda.scalar_iso_8_double_backward
        else:
            if deepwave.use_openmp:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if v.requires_grad and aux > 1 and deepwave.use_openmp:
                grad_v_tmp.resize_(aux, *v.shape)
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
            if source_amplitudes_requires_grad:
                grad_f.resize_(n_shots, nt, n_sources_per_shot)
                grad_f.fill_(0)
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = deepwave.dll_cpu.scalar_iso_2_float_backward
                elif accuracy == 4:
                    backward = deepwave.dll_cpu.scalar_iso_4_float_backward
                elif accuracy == 6:
                    backward = deepwave.dll_cpu.scalar_iso_6_float_backward
                else:
                    backward = deepwave.dll_cpu.scalar_iso_8_float_backward
            else:
                if accuracy == 2:
                    backward = deepwave.dll_cpu.scalar_iso_2_double_backward
                elif accuracy == 4:
                    backward = deepwave.dll_cpu.scalar_iso_4_double_backward
                elif accuracy == 6:
                    backward = deepwave.dll_cpu.scalar_iso_6_double_backward
                else:
                    backward = deepwave.dll_cpu.scalar_iso_8_double_backward

        v2dt2 = v**2 * dt**2
        gwfp = -gwfp

        if gwfc.numel() > 0 and nt > 0:
            backward(v2dt2.data_ptr(), grad_r.data_ptr(), gwfc.data_ptr(),
                     gwfp.data_ptr(), gpsiy.data_ptr(), gpsix.data_ptr(),
                     gpsiyn.data_ptr(), gpsixn.data_ptr(), gzetay.data_ptr(),
                     gzetax.data_ptr(), gzetayn.data_ptr(), gzetaxn.data_ptr(),
                     dwdv.data_ptr(), grad_f.data_ptr(),
                     grad_v.data_ptr(), grad_v_tmp_ptr, ay.data_ptr(),
                     ax.data_ptr(), by.data_ptr(), bx.data_ptr(),
                     dbydy.data_ptr(), dbxdx.data_ptr(), sources_i.data_ptr(),
                     receivers_i.data_ptr(), 1 / dy, 1 / dx, 1 / dy**2,
                     1 / dx**2, nt, n_shots, ny, nx,
                     n_sources_per_shot * source_amplitudes_requires_grad,
                     n_receivers_per_shot, step_ratio, v.requires_grad, pml_y0,
                     pml_y1, pml_x0, pml_x1, aux)

        s = (slice(None), slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        if nt % 2 == 0:
            return grad_v, grad_f, gwfc[s], -gwfp[s], gpsiy[s], gpsix[
                s], gzetay[s], gzetax[s]
        else:
            return grad_v, grad_f, gwfp[s], -gwfc[s], gpsiyn[s], gpsixn[
                s], gzetayn[s], gzetaxn[s]

    @staticmethod
    @once_differentiable
    def backward(ctx, ggv, ggf, ggwfc, ggwfp, ggpsiy, ggpsix, ggzetay,
                 ggzetax):
        (gwfc, gwfp, gpsiy, gpsix, gzetay, gzetax, grad_r, v, ay, ax, by, bx,
         dbydy, dbxdx, sources_i, receivers_i, source_amplitudes, wfc, wfp,
         psiy, psix, zetay, zetax) = ctx.saved_tensors
        dy = ctx.dy
        dx = ctx.dx
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        pml_width = ctx.pml_width
        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad

        if ggv.numel() == 0:
            ggv = torch.zeros_like(v.detach())
        if ggf.numel() == 0:
            ggf = torch.zeros_like(source_amplitudes.detach())

        v = v.contiguous()
        ggv = ggv.contiguous()
        source_amplitudes = source_amplitudes.contiguous()
        ggsource_amplitudes = ggf.contiguous()
        grad_r = grad_r.contiguous()
        ay = ay.contiguous()
        ax = ax.contiguous()
        by = by.contiguous()
        bx = bx.contiguous()
        dbydy = dbydy.contiguous()
        dbxdx = dbxdx.contiguous()
        fwd_sources_i = sources_i.contiguous()
        fwd_receivers_i = torch.empty(0)
        fwd_ggreceivers_i = receivers_i.contiguous()

        fd_pad = accuracy // 2
        size_with_batch = (n_shots, *v.shape)
        wfc = create_or_pad(wfc, fd_pad, v.device, v.dtype, size_with_batch)
        wfp = create_or_pad(wfp, fd_pad, v.device, v.dtype, size_with_batch)
        psiy = create_or_pad(psiy, fd_pad, v.device, v.dtype, size_with_batch)
        psix = create_or_pad(psix, fd_pad, v.device, v.dtype, size_with_batch)
        zetay = create_or_pad(zetay, fd_pad, v.device, v.dtype,
                              size_with_batch)
        zetax = create_or_pad(zetax, fd_pad, v.device, v.dtype,
                              size_with_batch)
        ggwfc = create_or_pad(ggwfc, fd_pad, v.device, v.dtype,
                              size_with_batch)
        ggwfp = create_or_pad(ggwfp, fd_pad, v.device, v.dtype,
                              size_with_batch)
        ggpsiy = create_or_pad(ggpsiy, fd_pad, v.device, v.dtype,
                               size_with_batch)
        ggpsix = create_or_pad(ggpsix, fd_pad, v.device, v.dtype,
                               size_with_batch)
        ggzetay = create_or_pad(ggzetay, fd_pad, v.device, v.dtype,
                                size_with_batch)
        ggzetax = create_or_pad(ggzetax, fd_pad, v.device, v.dtype,
                                size_with_batch)
        zero_interior(psiy, 2 * fd_pad, pml_width, True)
        zero_interior(psix, 2 * fd_pad, pml_width, False)
        zero_interior(zetay, 2 * fd_pad, pml_width, True)
        zero_interior(zetax, 2 * fd_pad, pml_width, False)
        zero_interior(ggpsiy, 2 * fd_pad, pml_width, True)
        zero_interior(ggpsix, 2 * fd_pad, pml_width, False)
        zero_interior(ggzetay, 2 * fd_pad, pml_width, True)
        zero_interior(ggzetax, 2 * fd_pad, pml_width, False)

        device = v.device
        dtype = v.dtype
        ny = v.shape[0]
        nx = v.shape[1]
        n_sources_per_shot = fwd_sources_i.numel() // n_shots
        n_receivers_per_shot = fwd_receivers_i.numel() // n_shots
        n_ggreceivers_per_shot = fwd_ggreceivers_i.numel() // n_shots
        psiyn = torch.zeros_like(psiy)
        psixn = torch.zeros_like(psix)
        ggpsiyn = torch.zeros_like(ggpsiy)
        ggpsixn = torch.zeros_like(ggpsix)
        w_store = torch.empty(0, device=device, dtype=dtype)
        ggw_store = torch.empty(0, device=device, dtype=dtype)
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
        ggreceiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
        pml_y0 = min(pml_width[0] + 2 * fd_pad, ny - fd_pad)
        pml_y1 = max(pml_y0, ny - pml_width[1] - 2 * fd_pad)
        pml_x0 = min(pml_width[2] + 2 * fd_pad, nx - fd_pad)
        pml_x1 = max(pml_x0, nx - pml_width[3] - 2 * fd_pad)

        if v.is_cuda:
            aux = v.get_device()
            if v.requires_grad:
                w_store.resize_(nt // step_ratio, n_shots, *v.shape)
                w_store.fill_(0)
                ggw_store.resize_(nt // step_ratio, n_shots, *v.shape)
                ggw_store.fill_(0)
            if fwd_receivers_i.numel() > 0:
                receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
                receiver_amplitudes.fill_(0)
            if fwd_ggreceivers_i.numel() > 0:
                ggreceiver_amplitudes.resize_(nt, n_shots,
                                              n_ggreceivers_per_shot)
                ggreceiver_amplitudes.fill_(0)
            if dtype == torch.float32:
                if accuracy == 2:
                    forward = deepwave.dll_cuda.scalar_born_iso_2_float_forward
                elif accuracy == 4:
                    forward = deepwave.dll_cuda.scalar_born_iso_4_float_forward
                elif accuracy == 6:
                    forward = deepwave.dll_cuda.scalar_born_iso_6_float_forward
                else:
                    forward = deepwave.dll_cuda.scalar_born_iso_8_float_forward
            else:
                if accuracy == 2:
                    forward = deepwave.dll_cuda.scalar_born_iso_2_double_forward
                elif accuracy == 4:
                    forward = deepwave.dll_cuda.scalar_born_iso_4_double_forward
                elif accuracy == 6:
                    forward = deepwave.dll_cuda.scalar_born_iso_6_double_forward
                else:
                    forward = deepwave.dll_cuda.scalar_born_iso_8_double_forward
        else:
            if deepwave.use_openmp:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if v.requires_grad:
                w_store.resize_(n_shots, nt // step_ratio, *v.shape)
                w_store.fill_(0)
                ggw_store.resize_(n_shots, nt // step_ratio, *v.shape)
                ggw_store.fill_(0)
            if fwd_receivers_i.numel() > 0:
                receiver_amplitudes.resize_(n_shots, nt, n_receivers_per_shot)
                receiver_amplitudes.fill_(0)
            if fwd_ggreceivers_i.numel() > 0:
                ggreceiver_amplitudes.resize_(n_shots, nt,
                                              n_ggreceivers_per_shot)
                ggreceiver_amplitudes.fill_(0)
            if dtype == torch.float32:
                if accuracy == 2:
                    forward = deepwave.dll_cpu.scalar_born_iso_2_float_forward
                elif accuracy == 4:
                    forward = deepwave.dll_cpu.scalar_born_iso_4_float_forward
                elif accuracy == 6:
                    forward = deepwave.dll_cpu.scalar_born_iso_6_float_forward
                else:
                    forward = deepwave.dll_cpu.scalar_born_iso_8_float_forward
            else:
                if accuracy == 2:
                    forward = deepwave.dll_cpu.scalar_born_iso_2_double_forward
                elif accuracy == 4:
                    forward = deepwave.dll_cpu.scalar_born_iso_4_double_forward
                elif accuracy == 6:
                    forward = deepwave.dll_cpu.scalar_born_iso_6_double_forward
                else:
                    forward = deepwave.dll_cpu.scalar_born_iso_8_double_forward

        if wfc.numel() > 0 and nt > 0:
            forward(v.data_ptr(), ggv.data_ptr(), source_amplitudes.data_ptr(),
                    ggsource_amplitudes.data_ptr(), wfc.data_ptr(),
                    wfp.data_ptr(), psiy.data_ptr(), psix.data_ptr(),
                    psiyn.data_ptr(), psixn.data_ptr(), zetay.data_ptr(),
                    zetax.data_ptr(), ggwfc.data_ptr(), ggwfp.data_ptr(),
                    ggpsiy.data_ptr(), ggpsix.data_ptr(), ggpsiyn.data_ptr(),
                    ggpsixn.data_ptr(), ggzetay.data_ptr(), ggzetax.data_ptr(),
                    w_store.data_ptr(), ggw_store.data_ptr(),
                    receiver_amplitudes.data_ptr(),
                    ggreceiver_amplitudes.data_ptr(), ay.data_ptr(),
                    ax.data_ptr(), by.data_ptr(), bx.data_ptr(),
                    dbydy.data_ptr(), dbxdx.data_ptr(),
                    fwd_sources_i.data_ptr(), fwd_receivers_i.data_ptr(),
                    fwd_ggreceivers_i.data_ptr(), 1 / dy, 1 / dx, 1 / dy**2,
                    1 / dx**2, dt**2, nt, n_shots, ny, nx, n_sources_per_shot,
                    n_receivers_per_shot, n_ggreceivers_per_shot, step_ratio,
                    v.requires_grad, False, pml_y0, pml_y1, pml_x0, pml_x1,
                    aux)

        bwd_sources_i = sources_i.contiguous()
        bwd_receivers_i = torch.empty(0)
        bwd_greceivers_i = receivers_i.contiguous()

        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad
        n_sources_per_shot = bwd_sources_i.numel() // n_shots
        n_receivers_per_shot = bwd_receivers_i.numel() // n_shots
        n_greceivers_per_shot = bwd_greceivers_i.numel() // n_shots

        wfc = create_or_pad(torch.empty(0), fd_pad, v.device, v.dtype,
                            size_with_batch)
        wfp = create_or_pad(torch.empty(0), fd_pad, v.device, v.dtype,
                            size_with_batch)
        psiy = create_or_pad(torch.empty(0), fd_pad, v.device, v.dtype,
                             size_with_batch)
        psix = create_or_pad(torch.empty(0), fd_pad, v.device, v.dtype,
                             size_with_batch)
        zetay = create_or_pad(torch.empty(0), fd_pad, v.device, v.dtype,
                              size_with_batch)
        zetax = create_or_pad(torch.empty(0), fd_pad, v.device, v.dtype,
                              size_with_batch)
        zero_interior(psiy, 2 * fd_pad, pml_width, True)
        zero_interior(psix, 2 * fd_pad, pml_width, False)
        zero_interior(zetay, 2 * fd_pad, pml_width, True)
        zero_interior(zetax, 2 * fd_pad, pml_width, False)
        psiyn = torch.zeros_like(psiy)
        psixn = torch.zeros_like(psix)
        zetayn = torch.zeros_like(zetay)
        zetaxn = torch.zeros_like(zetax)
        gwfc = create_or_pad(gwfc, fd_pad, v.device, v.dtype, size_with_batch)
        gwfp = create_or_pad(gwfp, fd_pad, v.device, v.dtype, size_with_batch)
        gpsiy = create_or_pad(gpsiy, fd_pad, v.device, v.dtype,
                              size_with_batch)
        gpsix = create_or_pad(gpsix, fd_pad, v.device, v.dtype,
                              size_with_batch)
        gzetay = create_or_pad(gzetay, fd_pad, v.device, v.dtype,
                               size_with_batch)
        gzetax = create_or_pad(gzetax, fd_pad, v.device, v.dtype,
                               size_with_batch)
        zero_interior(gpsiy, 2 * fd_pad, pml_width, True)
        zero_interior(gpsix, 2 * fd_pad, pml_width, False)
        zero_interior(gzetay, 2 * fd_pad, pml_width, True)
        zero_interior(gzetax, 2 * fd_pad, pml_width, False)
        gpsiyn = torch.zeros_like(gpsiy)
        gpsixn = torch.zeros_like(gpsix)
        gzetayn = torch.zeros_like(gzetay)
        gzetaxn = torch.zeros_like(gzetax)
        grad_v = torch.empty(0, device=device, dtype=dtype)
        grad_v_tmp = torch.empty(0, device=device, dtype=dtype)
        grad_v_tmp_ptr = grad_v.data_ptr()
        grad_scatter = torch.empty(0, device=device, dtype=dtype)
        grad_scatter_tmp = torch.empty(0, device=device, dtype=dtype)
        grad_scatter_tmp_ptr = grad_scatter.data_ptr()
        if v.requires_grad:
            grad_v.resize_(*v.shape)
            grad_v.fill_(0)
            grad_v_tmp_ptr = grad_v.data_ptr()
        grad_f = torch.empty(0, device=device, dtype=dtype)
        grad_gf = torch.empty(0, device=device, dtype=dtype)
        pml_y0 = min(pml_width[0] + 3 * fd_pad, ny - fd_pad)
        pml_y1 = max(pml_y0, ny - pml_width[1] - 3 * fd_pad)
        pml_x0 = min(pml_width[2] + 3 * fd_pad, nx - fd_pad)
        pml_x1 = max(pml_x0, nx - pml_width[3] - 3 * fd_pad)

        if v.is_cuda:
            aux = v.get_device()
            if v.requires_grad and n_shots > 1:
                grad_v_tmp.resize_(n_shots, *v.shape)
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
            if source_amplitudes_requires_grad:
                grad_f.resize_(nt, n_shots, n_sources_per_shot)
                grad_f.fill_(0)
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = deepwave.dll_cuda.scalar_born_iso_2_float_backward
                elif accuracy == 4:
                    backward = deepwave.dll_cuda.scalar_born_iso_4_float_backward
                elif accuracy == 6:
                    backward = deepwave.dll_cuda.scalar_born_iso_6_float_backward
                else:
                    backward = deepwave.dll_cuda.scalar_born_iso_8_float_backward
            else:
                if accuracy == 2:
                    backward = deepwave.dll_cuda.scalar_born_iso_2_double_backward
                elif accuracy == 4:
                    backward = deepwave.dll_cuda.scalar_born_iso_4_double_backward
                elif accuracy == 6:
                    backward = deepwave.dll_cuda.scalar_born_iso_6_double_backward
                else:
                    backward = deepwave.dll_cuda.scalar_born_iso_8_double_backward
        else:
            if deepwave.use_openmp:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if v.requires_grad and aux > 1 and deepwave.use_openmp:
                grad_v_tmp.resize_(aux, *v.shape)
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
            if source_amplitudes_requires_grad:
                grad_f.resize_(n_shots, nt, n_sources_per_shot)
                grad_f.fill_(0)
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = deepwave.dll_cpu.scalar_born_iso_2_float_backward
                elif accuracy == 4:
                    backward = deepwave.dll_cpu.scalar_born_iso_4_float_backward
                elif accuracy == 6:
                    backward = deepwave.dll_cpu.scalar_born_iso_6_float_backward
                else:
                    backward = deepwave.dll_cpu.scalar_born_iso_8_float_backward
            else:
                if accuracy == 2:
                    backward = deepwave.dll_cpu.scalar_born_iso_2_double_backward
                elif accuracy == 4:
                    backward = deepwave.dll_cpu.scalar_born_iso_4_double_backward
                elif accuracy == 6:
                    backward = deepwave.dll_cpu.scalar_born_iso_6_double_backward
                else:
                    backward = deepwave.dll_cpu.scalar_born_iso_8_double_backward

        gwfp = -gwfp

        if wfc.numel() > 0 and nt > 0 and v.requires_grad:
            backward(
                v.data_ptr(), ggv.data_ptr(),
                torch.empty(0).data_ptr(), grad_r.data_ptr(), wfc.data_ptr(),
                wfp.data_ptr(), psiy.data_ptr(), psix.data_ptr(),
                psiyn.data_ptr(), psixn.data_ptr(), zetay.data_ptr(),
                zetax.data_ptr(), zetayn.data_ptr(), zetaxn.data_ptr(),
                gwfc.data_ptr(), gwfp.data_ptr(), gpsiy.data_ptr(),
                gpsix.data_ptr(), gpsiyn.data_ptr(), gpsixn.data_ptr(),
                gzetay.data_ptr(), gzetax.data_ptr(), gzetayn.data_ptr(),
                gzetaxn.data_ptr(), w_store.data_ptr(), ggw_store.data_ptr(),
                grad_f.data_ptr(), grad_gf.data_ptr(), grad_v.data_ptr(),
                torch.empty(0).data_ptr(), grad_v_tmp_ptr,
                torch.empty(0).data_ptr(), ay.data_ptr(), ax.data_ptr(),
                by.data_ptr(),
                bx.data_ptr(), dbydy.data_ptr(), dbxdx.data_ptr(),
                bwd_sources_i.data_ptr(), bwd_receivers_i.data_ptr(),
                bwd_greceivers_i.data_ptr(), 1 / dy, 1 / dx, 1 / dy**2,
                1 / dx**2, dt**2, nt, n_shots, ny, nx,
                n_sources_per_shot * source_amplitudes_requires_grad, 0,
                n_receivers_per_shot, n_greceivers_per_shot, step_ratio,
                v.requires_grad, False, pml_y0, pml_y1, pml_x0, pml_x1, aux)

        s = (slice(None), slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        if nt % 2 == 0:
            return ggwfc[s], ggwfp[s], ggpsiy[s], ggpsix[s], ggzetay[s], ggzetax[
                s], ggreceiver_amplitudes, grad_v, None, None, None, None, None, None, None, None, None, grad_f, wfc[
                    s], -wfp[s], psiy[s], psix[s], zetay[s], zetax[
                        s], None, None, None, None, None, None, None, None, None
        else:
            return ggwfp[s], ggwfc[s], ggpsiyn[s], ggpsixn[s], ggzetay[s], ggzetax[
                s], ggreceiver_amplitudes, grad_v, None, None, None, None, None, None, None, None, None, grad_f, wfp[
                    s], -wfc[s], psiyn[s], psixn[s], zetayn[s], zetaxn[
                        s], None, None, None, None, None, None, None, None, None


def scalar_func(*args):
    return ScalarForwardFunc.apply(*args)
