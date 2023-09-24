"""Scalar Born wave propagation module.

Implements Born forward modelling (and its adjoint for backpropagation)
of the scalar wave equation. The implementation is similar to that
described in the scalar module, with the addition of a scattered
wavefield that uses 2 / v * scatter * dt^2 * wavefield as the source term.
"""

from typing import Optional, Union, Tuple, List
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
import deepwave
from deepwave.common import (setup_propagator, downsample_and_movedim,
                             zero_interior, create_or_pad, diff)


class ScalarBorn(torch.nn.Module):
    """A Module wrapper around :func:`scalar_born`.

    This is a convenience module that allows you to only specify
    `v`, `scatter`, and `grid_spacing` once. They will then be added to the
    list of arguments passed to :func:`scalar_born` when you call the
    forward method.

    Note that a copy will be made of the provided `v` and `scatter`. Gradients
    will not backpropagate to the initial guess wavespeed and scattering
    potential that are provided. You can use the module's `v` and `scatter`
    attributes to access them.

    Args:
        v:
            A 2D Tensor containing an initial guess of the wavespeed.
        scatter:
            A 2D Tensor containing an initial guess of the scattering
            potential.
        grid_spacing:
            The spatial grid cell size, specified with a single real number
            (used for both dimensions) or a List or Tensor of length
            two (the length in each of the two dimensions).
        v_requires_grad:
            Optional bool specifying how to set the `requires_grad`
            attribute of the wavespeed, and so whether the necessary
            data should be stored to calculate the gradient with respect
            to `v` during backpropagation. Default False.
        scatter_requires_grad:
            Similar, for the scattering potential.
    """

    def __init__(self,
                 v: Tensor,
                 scatter: Tensor,
                 grid_spacing: Union[int, float, List[float], Tensor],
                 v_requires_grad: bool = False,
                 scatter_requires_grad: bool = False) -> None:
        super().__init__()
        if v.ndim != 2:
            raise RuntimeError("v must have two dimensions")
        if scatter.ndim != 2:
            raise RuntimeError("scatter must have two dimensions")
        if v.device != scatter.device:
            raise RuntimeError("v and scatter must be on the same device")
        if v.dtype != scatter.dtype:
            raise RuntimeError("v and scatter must have the same dtype")
        self.v = torch.nn.Parameter(v, requires_grad=v_requires_grad)
        self.scatter = torch.nn.Parameter(scatter,
                                          requires_grad=scatter_requires_grad)
        self.grid_spacing = grid_spacing

    def forward(
        self,
        dt: float,
        source_amplitudes: Optional[Tensor] = None,
        source_locations: Optional[Tensor] = None,
        receiver_locations: Optional[Tensor] = None,
        bg_receiver_locations: Optional[Tensor] = None,
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
        wavefield_sc_0: Optional[Tensor] = None,
        wavefield_sc_m1: Optional[Tensor] = None,
        psiy_sc_m1: Optional[Tensor] = None,
        psix_sc_m1: Optional[Tensor] = None,
        zetay_sc_m1: Optional[Tensor] = None,
        zetax_sc_m1: Optional[Tensor] = None,
        origin: Optional[List[int]] = None,
        nt: Optional[int] = None,
        model_gradient_sampling_interval: int = 1
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
               Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Perform forward propagation/modelling.

        The inputs are the same as for :func:`scalar_born` except that `v`,
        `scatter`, and `grid_spacing` do not need to be provided again. See
        :func:`scalar_born` for a description of the inputs and outputs.
        """

        return scalar_born(
            self.v,
            self.scatter,
            self.grid_spacing,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            bg_receiver_locations=bg_receiver_locations,
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
            wavefield_sc_0=wavefield_sc_0,
            wavefield_sc_m1=wavefield_sc_m1,
            psiy_sc_m1=psiy_sc_m1,
            psix_sc_m1=psix_sc_m1,
            zetay_sc_m1=zetay_sc_m1,
            zetax_sc_m1=zetax_sc_m1,
            origin=origin,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval)


def scalar_born(
    v: Tensor,
    scatter: Tensor,
    grid_spacing: Union[int, float, List[float], Tensor],
    dt: float,
    source_amplitudes: Optional[Tensor] = None,
    source_locations: Optional[Tensor] = None,
    receiver_locations: Optional[Tensor] = None,
    bg_receiver_locations: Optional[Tensor] = None,
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
    wavefield_sc_0: Optional[Tensor] = None,
    wavefield_sc_m1: Optional[Tensor] = None,
    psiy_sc_m1: Optional[Tensor] = None,
    psix_sc_m1: Optional[Tensor] = None,
    zetay_sc_m1: Optional[Tensor] = None,
    zetax_sc_m1: Optional[Tensor] = None,
    origin: Optional[List[int]] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
           Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Scalar Born wave propagation (functional interface).

    This function performs Born forward modelling with the scalar wave
    equation. The outputs are differentiable with respect to the wavespeed,
    the scattering potential, the source amplitudes, and the initial
    wavefields.

    For computational performance, multiple shots may be propagated
    simultaneously.

    Most arguments and returns are the same as those of :func:`scalar`, so
    only those that are different will be described here.

    Args:
        scatter:
            A 2D Tensor containing an initial guess of the scattering
            potential. Unlike the module interface (:class:`ScalarBorn`),
            in this functional interface a copy is not made of the model,
            so gradients will propagate back into the provided Tensor.
        bg_receiver_locations:
            A Tensor with dimensions [shot, receiver, 2], containing
            the coordinates of the cell containing each receiver of the
            background wavefield. Optional.
            It should have torch.long (int64) datatype. If not provided,
            the output `bg_receiver_amplitudes` Tensor will be empty. If
            backpropagation will be performed, the location of each
            background receiver must be unique within the same shot.
        wavefield_sc_0, wavefield_sc_m1:
            The equivalent of `wavefield_0`, etc., for the scattered
            wavefield.
        psiy_sc_m1, psix_sc_m1, zetay_sc_m1, zetax_sc_m1:
            The equivalent of `psiy_m1`, etc., for the scattered
            wavefield.

    Returns:
        Tuple[Tensor]:

            wavefield_nt:
                A Tensor containing the non-scattered wavefield at the final
                time step.
            wavefield_ntm1:
                A Tensor containing the non-scattered wavefield at the
                second-to-last time step.
            psiy_ntm1, psix_ntm1, zetay_ntm1, zetax_ntm1:
                Tensor containing the wavefield related to the PML at the
                second-to-last time step for the non-scattered wavefield.
            wavefield_sc_nt, wavefield_sc_ntm1:
                Tensor containing the scattered wavefield.
            psiy_sc_ntm1, psix_sc_ntm1, zetay_sc_ntm1, zetax_sc_ntm1:
                Tensor containing the wavefield related to the scattered
                wavefield PML.
            bg_receiver_amplitudes:
                A Tensor of dimensions [shot, receiver, time] containing
                the receiver amplitudes recorded at the provided receiver
                locations, extracted from the background wavefield. If no
                receiver locations were specified then this Tensor will be
                empty.
            receiver_amplitudes:
                A Tensor of dimensions [shot, receiver, time] containing
                the receiver amplitudes recorded at the provided receiver
                locations, extracted from the scattered wavefield. If no
                receiver locations were specified then this Tensor will be
                empty.

    """

    (models, source_amplitudes_l, wavefields,
     pml_profiles, sources_i_l, receivers_i_l,
     dy, dx, dt, nt, n_shots,
     step_ratio, model_gradient_sampling_interval,
     accuracy, pml_width_list) = \
        setup_propagator([v, scatter], 'scalar_born', grid_spacing, dt,
                         [wavefield_0, wavefield_m1, psiy_m1, psix_m1,
                          zetay_m1, zetax_m1, wavefield_sc_0, wavefield_sc_m1,
                          psiy_sc_m1, psix_sc_m1, zetay_sc_m1, zetax_sc_m1],
                         [source_amplitudes, source_amplitudes],
                         [source_locations, source_locations],
                         [receiver_locations, bg_receiver_locations],
                         accuracy, pml_width, pml_freq, max_vel,
                         survey_pad,
                         origin, nt, model_gradient_sampling_interval,
                         freq_taper_frac, time_pad_frac, time_taper)
    v, scatter = models
    (wfc, wfp, psiy, psix, zetay, zetax, wfcsc, wfpsc, psiysc, psixsc, zetaysc,
     zetaxsc) = wavefields
    source_amplitudes = source_amplitudes_l[0]
    source_amplitudessc = source_amplitudes_l[1]
    sources_i = sources_i_l[0]
    receivers_i = receivers_i_l[0]
    bg_receivers_i = receivers_i_l[1]
    ay, ax, by, bx = pml_profiles
    dbydy = diff(by, accuracy, dy)
    dbxdx = diff(bx, accuracy, dx)

    (wfc, wfp, psiy, psix, zetay, zetax, wfcsc, wfpsc, psiysc, psixsc, zetaysc, zetaxsc, receiver_amplitudes, receiver_amplitudessc) = \
        scalar_born_func(
            v, scatter, source_amplitudes, source_amplitudessc, wfc, wfp, psiy, psix, zetay, zetax, wfcsc, wfpsc, psiysc, psixsc, zetaysc, zetaxsc, ay, ax, by, bx, dbydy, dbxdx, sources_i, bg_receivers_i, receivers_i, dy, dx, dt, nt, step_ratio * model_gradient_sampling_interval, accuracy, pml_width_list, n_shots
        )

    receiver_amplitudes = downsample_and_movedim(receiver_amplitudes,
                                                 step_ratio, freq_taper_frac,
                                                 time_pad_frac, time_taper)
    receiver_amplitudessc = downsample_and_movedim(receiver_amplitudessc,
                                                   step_ratio, freq_taper_frac,
                                                   time_pad_frac, time_taper)

    return (wfc, wfp, psiy, psix, zetay, zetax, wfcsc, wfpsc, psiysc, psixsc,
            zetaysc, zetaxsc, receiver_amplitudes, receiver_amplitudessc)


class ScalarBornForwardFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v, scatter, source_amplitudes, source_amplitudessc, wfc, wfp,
                psiy, psix, zetay, zetax, wfcsc, wfpsc, psiysc, psixsc,
                zetaysc, zetaxsc, ay, ax, by, bx, dbydy, dbxdx, sources_i,
                receivers_i, receiverssc_i, dy, dx, dt, nt, step_ratio,
                accuracy, pml_width, n_shots):

        v = v.contiguous()
        scatter = scatter.contiguous()
        source_amplitudes = source_amplitudes.contiguous()
        source_amplitudessc = source_amplitudessc.contiguous()
        ay = ay.contiguous()
        ax = ax.contiguous()
        by = by.contiguous()
        bx = bx.contiguous()
        dbydy = dbydy.contiguous()
        dbxdx = dbxdx.contiguous()
        sources_i = sources_i.contiguous()
        receivers_i = receivers_i.contiguous()
        receiverssc_i = receiverssc_i.contiguous()

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
        wfcsc = create_or_pad(wfcsc, fd_pad, v.device, v.dtype,
                              size_with_batch)
        wfpsc = create_or_pad(wfpsc, fd_pad, v.device, v.dtype,
                              size_with_batch)
        psiysc = create_or_pad(psiysc, fd_pad, v.device, v.dtype,
                               size_with_batch)
        psixsc = create_or_pad(psixsc, fd_pad, v.device, v.dtype,
                               size_with_batch)
        zetaysc = create_or_pad(zetaysc, fd_pad, v.device, v.dtype,
                                size_with_batch)
        zetaxsc = create_or_pad(zetaxsc, fd_pad, v.device, v.dtype,
                                size_with_batch)
        zero_interior(psiy, 2 * fd_pad, pml_width, True)
        zero_interior(psix, 2 * fd_pad, pml_width, False)
        zero_interior(zetay, 2 * fd_pad, pml_width, True)
        zero_interior(zetax, 2 * fd_pad, pml_width, False)
        zero_interior(psiysc, 2 * fd_pad, pml_width, True)
        zero_interior(psixsc, 2 * fd_pad, pml_width, False)
        zero_interior(zetaysc, 2 * fd_pad, pml_width, True)
        zero_interior(zetaxsc, 2 * fd_pad, pml_width, False)

        device = v.device
        dtype = v.dtype
        ny = v.shape[0]
        nx = v.shape[1]
        n_sources_per_shot = sources_i.numel() // n_shots
        n_receivers_per_shot = receivers_i.numel() // n_shots
        n_receiverssc_per_shot = receiverssc_i.numel() // n_shots
        psiyn = torch.zeros_like(psiy)
        psixn = torch.zeros_like(psix)
        psiynsc = torch.zeros_like(psiysc)
        psixnsc = torch.zeros_like(psixsc)
        w_store = torch.empty(0, device=device, dtype=dtype)
        wsc_store = torch.empty(0, device=device, dtype=dtype)
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
        receiver_amplitudessc = torch.empty(0, device=device, dtype=dtype)
        pml_y0 = min(pml_width[0] + 2 * fd_pad, ny - fd_pad)
        pml_y1 = max(pml_y0, ny - pml_width[1] - 2 * fd_pad)
        pml_x0 = min(pml_width[2] + 2 * fd_pad, nx - fd_pad)
        pml_x1 = max(pml_x0, nx - pml_width[3] - 2 * fd_pad)

        if v.is_cuda:
            aux = v.get_device()
            if v.requires_grad or scatter.requires_grad:
                w_store.resize_(nt // step_ratio, n_shots, *v.shape)
                w_store.fill_(0)
            if v.requires_grad:
                wsc_store.resize_(nt // step_ratio, n_shots, *v.shape)
                wsc_store.fill_(0)
            if receivers_i.numel() > 0:
                receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
                receiver_amplitudes.fill_(0)
            if receiverssc_i.numel() > 0:
                receiver_amplitudessc.resize_(nt, n_shots,
                                              n_receiverssc_per_shot)
                receiver_amplitudessc.fill_(0)
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
            if v.requires_grad or scatter.requires_grad:
                w_store.resize_(n_shots, nt // step_ratio, *v.shape)
                w_store.fill_(0)
            if v.requires_grad:
                wsc_store.resize_(n_shots, nt // step_ratio, *v.shape)
                wsc_store.fill_(0)
            if receivers_i.numel() > 0:
                receiver_amplitudes.resize_(n_shots, nt, n_receivers_per_shot)
                receiver_amplitudes.fill_(0)
            if receiverssc_i.numel() > 0:
                receiver_amplitudessc.resize_(n_shots, nt,
                                              n_receiverssc_per_shot)
                receiver_amplitudessc.fill_(0)
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
            forward(v.data_ptr(), scatter.data_ptr(),
                    source_amplitudes.data_ptr(),
                    source_amplitudessc.data_ptr(), wfc.data_ptr(),
                    wfp.data_ptr(), psiy.data_ptr(), psix.data_ptr(),
                    psiyn.data_ptr(), psixn.data_ptr(), zetay.data_ptr(),
                    zetax.data_ptr(), wfcsc.data_ptr(), wfpsc.data_ptr(),
                    psiysc.data_ptr(), psixsc.data_ptr(), psiynsc.data_ptr(),
                    psixnsc.data_ptr(), zetaysc.data_ptr(), zetaxsc.data_ptr(),
                    w_store.data_ptr(), wsc_store.data_ptr(),
                    receiver_amplitudes.data_ptr(),
                    receiver_amplitudessc.data_ptr(), ay.data_ptr(),
                    ax.data_ptr(), by.data_ptr(), bx.data_ptr(),
                    dbydy.data_ptr(), dbxdx.data_ptr(), sources_i.data_ptr(),
                    receivers_i.data_ptr(), receiverssc_i.data_ptr(), 1 / dy,
                    1 / dx, 1 / dy**2, 1 / dx**2, dt**2, nt, n_shots, ny, nx,
                    n_sources_per_shot, n_receivers_per_shot,
                    n_receiverssc_per_shot, step_ratio, v.requires_grad,
                    scatter.requires_grad, pml_y0, pml_y1, pml_x0, pml_x1, aux)

        if (v.requires_grad or scatter.requires_grad
                or source_amplitudes.requires_grad
                or source_amplitudessc.requires_grad or wfc.requires_grad
                or wfp.requires_grad or psiy.requires_grad
                or psix.requires_grad or zetay.requires_grad
                or zetax.requires_grad or wfcsc.requires_grad
                or wfpsc.requires_grad or psiysc.requires_grad
                or psixsc.requires_grad or zetaysc.requires_grad
                or zetaxsc.requires_grad):
            ctx.save_for_backward(v, scatter, ay, ax, by, bx, dbydy, dbxdx,
                                  sources_i, receivers_i, receiverssc_i,
                                  w_store, wsc_store)
            ctx.dy = dy
            ctx.dx = dx
            ctx.dt = dt
            ctx.nt = nt
            ctx.n_shots = n_shots
            ctx.step_ratio = step_ratio
            ctx.accuracy = accuracy
            ctx.pml_width = pml_width
            ctx.source_amplitudes_requires_grad = source_amplitudes.requires_grad
            ctx.source_amplitudessc_requires_grad = source_amplitudessc.requires_grad
            ctx.non_sc = (v.requires_grad or source_amplitudes.requires_grad
                          or wfc.requires_grad or wfp.requires_grad
                          or psiy.requires_grad or psix.requires_grad
                          or zetay.requires_grad or zetax.requires_grad)

        s = (slice(None), slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        if nt % 2 == 0:
            return (wfc[s], wfp[s], psiy[s], psix[s], zetay[s], zetax[s],
                    wfcsc[s], wfpsc[s], psiysc[s], psixsc[s], zetaysc[s],
                    zetaxsc[s], receiver_amplitudes, receiver_amplitudessc)
        else:
            return (wfp[s], wfc[s], psiyn[s], psixn[s], zetay[s], zetax[s],
                    wfpsc[s], wfcsc[s], psiynsc[s], psixnsc[s], zetaysc[s],
                    zetaxsc[s], receiver_amplitudes, receiver_amplitudessc)

    @staticmethod
    @once_differentiable
    def backward(ctx, wfc, wfp, psiy, psix, zetay, zetax, wfcsc, wfpsc, psiysc,
                 psixsc, zetaysc, zetaxsc, grad_r, grad_rsc):
        (v, scatter, ay, ax, by, bx, dbydy, dbxdx, sources_i, receivers_i,
         receiverssc_i, w_store, wsc_store) = ctx.saved_tensors

        v = v.contiguous()
        scatter = scatter.contiguous()
        grad_r = grad_r.contiguous()
        grad_rsc = grad_rsc.contiguous()
        ay = ay.contiguous()
        ax = ax.contiguous()
        by = by.contiguous()
        bx = bx.contiguous()
        dbydy = dbydy.contiguous()
        dbxdx = dbxdx.contiguous()
        sources_i = sources_i.contiguous()
        receivers_i = receivers_i.contiguous()
        receiverssc_i = receiverssc_i.contiguous()

        dy = ctx.dy
        dx = ctx.dx
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        pml_width = ctx.pml_width
        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad
        source_amplitudessc_requires_grad = ctx.source_amplitudessc_requires_grad
        non_sc = ctx.non_sc
        device = v.device
        dtype = v.dtype
        ny = v.shape[0]
        nx = v.shape[1]
        n_sources_per_shot = sources_i.numel() // n_shots
        n_receivers_per_shot = receivers_i.numel() // n_shots
        n_receiverssc_per_shot = receiverssc_i.numel() // n_shots
        fd_pad = accuracy // 2

        size_with_batch = (n_shots, *v.shape)
        if non_sc:
            wfc = create_or_pad(wfc, fd_pad, v.device, v.dtype,
                                size_with_batch)
            wfp = create_or_pad(wfp, fd_pad, v.device, v.dtype,
                                size_with_batch)
            psiy = create_or_pad(psiy, fd_pad, v.device, v.dtype,
                                 size_with_batch)
            psix = create_or_pad(psix, fd_pad, v.device, v.dtype,
                                 size_with_batch)
            zetay = create_or_pad(zetay, fd_pad, v.device, v.dtype,
                                  size_with_batch)
            zetax = create_or_pad(zetax, fd_pad, v.device, v.dtype,
                                  size_with_batch)
            zero_interior(psiy, 2 * fd_pad, pml_width, True)
            zero_interior(psix, 2 * fd_pad, pml_width, False)
            zero_interior(zetay, 2 * fd_pad, pml_width, True)
            zero_interior(zetax, 2 * fd_pad, pml_width, False)
            psiyn = torch.zeros_like(psiy)
            psixn = torch.zeros_like(psix)
            zetayn = torch.zeros_like(zetay)
            zetaxn = torch.zeros_like(zetax)
        wfcsc = create_or_pad(wfcsc, fd_pad, v.device, v.dtype,
                              size_with_batch)
        wfpsc = create_or_pad(wfpsc, fd_pad, v.device, v.dtype,
                              size_with_batch)
        psiysc = create_or_pad(psiysc, fd_pad, v.device, v.dtype,
                               size_with_batch)
        psixsc = create_or_pad(psixsc, fd_pad, v.device, v.dtype,
                               size_with_batch)
        zetaysc = create_or_pad(zetaysc, fd_pad, v.device, v.dtype,
                                size_with_batch)
        zetaxsc = create_or_pad(zetaxsc, fd_pad, v.device, v.dtype,
                                size_with_batch)
        zero_interior(psiysc, 2 * fd_pad, pml_width, True)
        zero_interior(psixsc, 2 * fd_pad, pml_width, False)
        zero_interior(zetaysc, 2 * fd_pad, pml_width, True)
        zero_interior(zetaxsc, 2 * fd_pad, pml_width, False)
        psiynsc = torch.zeros_like(psiysc)
        psixnsc = torch.zeros_like(psixsc)
        zetaynsc = torch.zeros_like(zetaysc)
        zetaxnsc = torch.zeros_like(zetaxsc)
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
        if scatter.requires_grad:
            grad_scatter.resize_(*scatter.shape)
            grad_scatter.fill_(0)
            grad_scatter_tmp_ptr = grad_scatter.data_ptr()
        grad_f = torch.empty(0, device=device, dtype=dtype)
        grad_fsc = torch.empty(0, device=device, dtype=dtype)
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
            if scatter.requires_grad and n_shots > 1:
                grad_scatter_tmp.resize_(n_shots, *scatter.shape)
                grad_scatter_tmp.fill_(0)
                grad_scatter_tmp_ptr = grad_scatter_tmp.data_ptr()
            if source_amplitudes_requires_grad:
                grad_f.resize_(nt, n_shots, n_sources_per_shot)
                grad_f.fill_(0)
            if source_amplitudessc_requires_grad:
                grad_fsc.resize_(nt, n_shots, n_sources_per_shot)
                grad_fsc.fill_(0)
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = deepwave.dll_cuda.scalar_born_iso_2_float_backward
                    backward_sc = deepwave.dll_cuda.scalar_born_iso_2_float_backward_sc
                elif accuracy == 4:
                    backward = deepwave.dll_cuda.scalar_born_iso_4_float_backward
                    backward_sc = deepwave.dll_cuda.scalar_born_iso_4_float_backward_sc
                elif accuracy == 6:
                    backward = deepwave.dll_cuda.scalar_born_iso_6_float_backward
                    backward_sc = deepwave.dll_cuda.scalar_born_iso_6_float_backward_sc
                else:
                    backward = deepwave.dll_cuda.scalar_born_iso_8_float_backward
                    backward_sc = deepwave.dll_cuda.scalar_born_iso_8_float_backward_sc
            else:
                if accuracy == 2:
                    backward = deepwave.dll_cuda.scalar_born_iso_2_double_backward
                    backward_sc = deepwave.dll_cuda.scalar_born_iso_2_double_backward_sc
                elif accuracy == 4:
                    backward = deepwave.dll_cuda.scalar_born_iso_4_double_backward
                    backward_sc = deepwave.dll_cuda.scalar_born_iso_4_double_backward_sc
                elif accuracy == 6:
                    backward = deepwave.dll_cuda.scalar_born_iso_6_double_backward
                    backward_sc = deepwave.dll_cuda.scalar_born_iso_6_double_backward_sc
                else:
                    backward = deepwave.dll_cuda.scalar_born_iso_8_double_backward
                    backward_sc = deepwave.dll_cuda.scalar_born_iso_8_double_backward_sc
        else:
            if deepwave.use_openmp:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if v.requires_grad and aux > 1 and deepwave.use_openmp:
                grad_v_tmp.resize_(aux, *v.shape)
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
            if scatter.requires_grad and aux > 1 and deepwave.use_openmp:
                grad_scatter_tmp.resize_(aux, *scatter.shape)
                grad_scatter_tmp.fill_(0)
                grad_scatter_tmp_ptr = grad_scatter_tmp.data_ptr()
            if source_amplitudes_requires_grad:
                grad_f.resize_(n_shots, nt, n_sources_per_shot)
                grad_f.fill_(0)
            if source_amplitudessc_requires_grad:
                grad_fsc.resize_(n_shots, nt, n_sources_per_shot)
                grad_fsc.fill_(0)
            if dtype == torch.float32:
                if accuracy == 2:
                    backward = deepwave.dll_cpu.scalar_born_iso_2_float_backward
                    backward_sc = deepwave.dll_cpu.scalar_born_iso_2_float_backward_sc
                elif accuracy == 4:
                    backward = deepwave.dll_cpu.scalar_born_iso_4_float_backward
                    backward_sc = deepwave.dll_cpu.scalar_born_iso_4_float_backward_sc
                elif accuracy == 6:
                    backward = deepwave.dll_cpu.scalar_born_iso_6_float_backward
                    backward_sc = deepwave.dll_cpu.scalar_born_iso_6_float_backward_sc
                else:
                    backward = deepwave.dll_cpu.scalar_born_iso_8_float_backward
                    backward_sc = deepwave.dll_cpu.scalar_born_iso_8_float_backward_sc
            else:
                if accuracy == 2:
                    backward = deepwave.dll_cpu.scalar_born_iso_2_double_backward
                    backward_sc = deepwave.dll_cpu.scalar_born_iso_2_double_backward_sc
                elif accuracy == 4:
                    backward = deepwave.dll_cpu.scalar_born_iso_4_double_backward
                    backward_sc = deepwave.dll_cpu.scalar_born_iso_4_double_backward_sc
                elif accuracy == 6:
                    backward = deepwave.dll_cpu.scalar_born_iso_6_double_backward
                    backward_sc = deepwave.dll_cpu.scalar_born_iso_6_double_backward_sc
                else:
                    backward = deepwave.dll_cpu.scalar_born_iso_8_double_backward
                    backward_sc = deepwave.dll_cpu.scalar_born_iso_8_double_backward_sc

        wfp = -wfp
        wfpsc = -wfpsc

        if wfc.numel() > 0 and nt > 0:
            if non_sc:
                backward(
                    v.data_ptr(), scatter.data_ptr(), grad_r.data_ptr(),
                    grad_rsc.data_ptr(), wfc.data_ptr(), wfp.data_ptr(),
                    psiy.data_ptr(), psix.data_ptr(), psiyn.data_ptr(),
                    psixn.data_ptr(), zetay.data_ptr(), zetax.data_ptr(),
                    zetayn.data_ptr(), zetaxn.data_ptr(), wfcsc.data_ptr(),
                    wfpsc.data_ptr(), psiysc.data_ptr(), psixsc.data_ptr(),
                    psiynsc.data_ptr(), psixnsc.data_ptr(), zetaysc.data_ptr(),
                    zetaxsc.data_ptr(), zetaynsc.data_ptr(),
                    zetaxnsc.data_ptr(), w_store.data_ptr(),
                    wsc_store.data_ptr(), grad_f.data_ptr(),
                    grad_fsc.data_ptr(), grad_v.data_ptr(),
                    grad_scatter.data_ptr(), grad_v_tmp_ptr,
                    grad_scatter_tmp_ptr, ay.data_ptr(), ax.data_ptr(),
                    by.data_ptr(), bx.data_ptr(), dbydy.data_ptr(),
                    dbxdx.data_ptr(), sources_i.data_ptr(),
                    receivers_i.data_ptr(), receiverssc_i.data_ptr(), 1 / dy,
                    1 / dx, 1 / dy**2, 1 / dx**2, dt**2, nt, n_shots, ny, nx,
                    n_sources_per_shot * source_amplitudes_requires_grad,
                    n_sources_per_shot * source_amplitudessc_requires_grad,
                    n_receivers_per_shot, n_receiverssc_per_shot, step_ratio,
                    v.requires_grad, scatter.requires_grad, pml_y0, pml_y1,
                    pml_x0, pml_x1, aux)
            else:
                backward_sc(
                    v.data_ptr(), grad_rsc.data_ptr(), wfcsc.data_ptr(),
                    wfpsc.data_ptr(), psiysc.data_ptr(), psixsc.data_ptr(),
                    psiynsc.data_ptr(), psixnsc.data_ptr(), zetaysc.data_ptr(),
                    zetaxsc.data_ptr(), zetaynsc.data_ptr(),
                    zetaxnsc.data_ptr(),
                    w_store.data_ptr(), grad_fsc.data_ptr(),
                    grad_scatter.data_ptr(), grad_scatter_tmp_ptr,
                    ay.data_ptr(), ax.data_ptr(), by.data_ptr(), bx.data_ptr(),
                    dbydy.data_ptr(), dbxdx.data_ptr(), sources_i.data_ptr(),
                    receiverssc_i.data_ptr(), 1 / dy, 1 / dx, 1 / dy**2,
                    1 / dx**2, dt**2, nt, n_shots, ny, nx,
                    n_sources_per_shot * source_amplitudessc_requires_grad,
                    n_receiverssc_per_shot, step_ratio, scatter.requires_grad,
                    pml_y0, pml_y1, pml_x0, pml_x1, aux)

        s = (slice(None), slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        if non_sc:
            if nt % 2 == 0:
                return grad_v, grad_scatter, grad_f, grad_fsc, wfc[s], -wfp[
                    s], psiy[s], psix[s], zetay[s], zetax[s], wfcsc[s], -wfpsc[
                        s], psiysc[s], psixsc[s], zetaysc[s], zetaxsc[
                            s], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
            else:
                return grad_v, grad_scatter, grad_f, grad_fsc, wfp[s], -wfc[
                    s], psiyn[s], psixn[s], zetayn[s], zetaxn[s], wfpsc[s], -wfcsc[
                        s], psiynsc[s], psixnsc[s], zetaynsc[s], zetaxnsc[
                            s], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        else:
            if nt % 2 == 0:
                return None, grad_scatter, None, grad_fsc, None, None, None, None, None, None, wfcsc[
                    s], -wfpsc[s], psiysc[s], psixsc[s], zetaysc[s], zetaxsc[
                        s], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
            else:
                return None, grad_scatter, None, grad_fsc, None, None, None, None, None, None, wfpsc[
                    s], -wfcsc[s], psiynsc[s], psixnsc[s], zetaynsc[s], zetaxnsc[
                        s], None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def scalar_born_func(*args):
    return ScalarBornForwardFunc.apply(*args)
