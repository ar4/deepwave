"""Scalar Born wave propagation module.

Implements Born forward modelling (and its adjoint for backpropagation)
of the scalar wave equation. The implementation is similar to that
described in the scalar module, with the addition of a scattered
wavefield that uses 2 / v * scatter * dt^2 * wavefield as the source term.
"""

from typing import Optional, Union, List
import torch
from torch import Tensor
from deepwave.common import (set_dx, check_inputs, pad_model, pad_locations,
                             location_to_index,
                             extract_survey,
                             get_n_batch, upsample, downsample,
                             convert_to_contiguous, setup_pml,
                             cfl_condition)


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
    def __init__(self, v, scatter, grid_spacing, v_requires_grad=False,
                 scatter_requires_grad=False):
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

    def forward(self, dt: float, source_amplitudes: Optional[Tensor] = None,
                source_locations: Optional[Tensor] = None,
                receiver_locations: Optional[Tensor] = None,
                accuracy: int = 4, pml_width: Union[int, List[int]] = 20,
                pml_freq: Optional[float] = None,
                max_vel: Optional[float] = None,
                survey_pad: Optional[Union[int, List[Optional[int]]]] = None,
                wavefield_0: Optional[Tensor] = None,
                wavefield_m1: Optional[Tensor] = None,
                psix_m1: Optional[Tensor] = None,
                psiy_m1: Optional[Tensor] = None,
                zetax_m1: Optional[Tensor] = None,
                zetay_m1: Optional[Tensor] = None,
                wavefield_sc_0: Optional[Tensor] = None,
                wavefield_sc_m1: Optional[Tensor] = None,
                psix_sc_m1: Optional[Tensor] = None,
                psiy_sc_m1: Optional[Tensor] = None,
                zetax_sc_m1: Optional[Tensor] = None,
                zetay_sc_m1: Optional[Tensor] = None,
                origin: Optional[List[int]] = None, nt: Optional[int] = None,
                model_gradient_sampling_interval: int = 1):
        """Perform forward propagation/modelling.

        The inputs are the same as for :func:`scalar_born` except that `v`,
        `scatter`, and `grid_spacing` do not need to be provided again. See
        :func:`scalar_born` for a description of the inputs and outputs.
        """

        return scalar_born(self.v, self.scatter, self.grid_spacing, dt,
                           source_amplitudes=source_amplitudes,
                           source_locations=source_locations,
                           receiver_locations=receiver_locations,
                           accuracy=accuracy, pml_width=pml_width,
                           pml_freq=pml_freq, max_vel=max_vel,
                           survey_pad=survey_pad,
                           wavefield_0=wavefield_0,
                           wavefield_m1=wavefield_m1,
                           psix_m1=psix_m1, psiy_m1=psiy_m1,
                           zetax_m1=zetax_m1, zetay_m1=zetay_m1,
                           wavefield_sc_0=wavefield_sc_0,
                           wavefield_sc_m1=wavefield_sc_m1,
                           psix_sc_m1=psix_sc_m1, psiy_sc_m1=psiy_sc_m1,
                           zetax_sc_m1=zetax_sc_m1,
                           zetay_sc_m1=zetay_sc_m1,
                           origin=origin, nt=nt,
                           model_gradient_sampling_interval=
                           model_gradient_sampling_interval)


def scalar_born(v: Tensor, scatter: Tensor,
                grid_spacing: Union[int, float, List[Union[int, float]],
                                    Tensor],
                dt: float, source_amplitudes: Optional[Tensor] = None,
                source_locations: Optional[Tensor] = None,
                receiver_locations: Optional[Tensor] = None,
                accuracy: int = 4, pml_width: Union[int, List[int]] = 20,
                pml_freq: Optional[float] = None,
                max_vel: Optional[float] = None,
                survey_pad: Optional[Union[int, List[Optional[int]]]] = None,
                wavefield_0: Optional[Tensor] = None,
                wavefield_m1: Optional[Tensor] = None,
                psix_m1: Optional[Tensor] = None,
                psiy_m1: Optional[Tensor] = None,
                zetax_m1: Optional[Tensor] = None,
                zetay_m1: Optional[Tensor] = None,
                wavefield_sc_0: Optional[Tensor] = None,
                wavefield_sc_m1: Optional[Tensor] = None,
                psix_sc_m1: Optional[Tensor] = None,
                psiy_sc_m1: Optional[Tensor] = None,
                zetax_sc_m1: Optional[Tensor] = None,
                zetay_sc_m1: Optional[Tensor] = None,
                origin: Optional[List[int]] = None,
                nt: Optional[int] = None,
                model_gradient_sampling_interval: int = 1):
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
        wavefield_sc_0, wavefield_sc_m1:
            The equivalent of `wavefield_0`, etc., for the scattered
            wavefield.
        psix_sc_m1, psiy_sc_m1, zetax_sc_m1, zetay_sc_m1:
            The equivalent of `psix_m1`, etc., for the scattered
            wavefield.

    Returns:
        Tuple[Tensor]:

            wavefield_nt:
                A Tensor containing the non-scattered wavefield at timestep
                `nt`.
            wavefield_ntm1:
                A Tensor containing the non-scattered wavefield at timestep
                `nt-1`.
            psix_ntm1, psiy_ntm1, zetax_ntm1, zetay_ntm1:
                Tensor containing the wavefield related to the PML at timestep
                `nt-1` for the non-scattered wavefield.
            wavefield_sc_nt, wavefield_sc_ntm1:
                Tensor containing the scattered wavefield.
            psix_sc_ntm1, psiy_sc_ntm1, zetax_sc_ntm1, zetay_sc_ntm1:
                Tensor containing the wavefield related to the scattered
                wavefield PML.
            receiver_amplitudes:
                A Tensor of dimensions [shot, receiver, time] containing
                the receiver amplitudes recorded at the provided receiver
                locations, extracted from the scattered wavefield. If no
                receiver locations were specified then this Tensor will be
                empty.

    """

    # Check inputs
    check_inputs(source_amplitudes, source_locations, receiver_locations,
                 [wavefield_0, wavefield_m1, psix_m1, psiy_m1,
                  zetax_m1, zetay_m1, wavefield_sc_0, wavefield_sc_m1,
                  psix_sc_m1, psiy_sc_m1, zetax_sc_m1, zetay_sc_m1],
                 accuracy, nt, v)

    if nt is None:
        nt = 0
        if source_amplitudes is not None:
            nt = source_amplitudes.shape[-1]
    device = v.device
    dtype = v.dtype
    dx, dy = set_dx(grid_spacing)
    if isinstance(pml_width, int):
        pml_width = [pml_width for _ in range(4)]
    fd_pad = accuracy // 2
    pad = [fd_pad + width for width in pml_width]
    models, locations = extract_survey(
        [v, scatter], [source_locations, receiver_locations], survey_pad,
        [wavefield_0, wavefield_m1, psix_m1, psiy_m1, zetax_m1, zetay_m1,
         wavefield_sc_0, wavefield_sc_m1, psix_sc_m1, psiy_sc_m1,
         zetax_sc_m1, zetay_sc_m1],
        origin, pml_width
    )
    v, scatter = models
    source_locations, receiver_locations = locations
    v_pad = pad_model(v, pad)
    scatter_pad = pad_model(scatter, pad, mode="constant")
    if max_vel is None:
        max_vel = v.abs().max().item()
    max_vel = abs(max_vel)
    dt, step_ratio = cfl_condition(dx, dy, dt, max_vel)
    if source_amplitudes is not None and source_locations is not None:
        source_locations = pad_locations(source_locations, pad)
        sources_i = location_to_index(source_locations, v_pad.shape[1])
        source_amplitudes = upsample(source_amplitudes, step_ratio)
    else:
        sources_i = None
    if receiver_locations is not None:
        receiver_locations = pad_locations(receiver_locations, pad)
        receivers_i = location_to_index(receiver_locations, v_pad.shape[1])
    else:
        receivers_i = None
    n_batch = get_n_batch([source_locations, receiver_locations,
                           wavefield_0, wavefield_m1, psix_m1, psiy_m1,
                           zetax_m1, zetay_m1, wavefield_0, wavefield_m1,
                           psix_m1, psiy_m1, zetax_m1, zetay_m1])
    ax, ay, bx, by = \
        setup_pml(pml_width, fd_pad, dt, v_pad, max_vel, pml_freq)
    nt *= step_ratio

    if source_amplitudes is not None:
        if source_amplitudes.device == torch.device('cpu'):
            source_amplitudes = torch.movedim(source_amplitudes, -1, 1)
        else:
            source_amplitudes = torch.movedim(source_amplitudes, -1, 0)

    v_pad = convert_to_contiguous(v_pad)
    scatter_pad = convert_to_contiguous(scatter_pad)
    source_amplitudes = (convert_to_contiguous(source_amplitudes)
                         .to(dtype).to(device))
    wavefield_0 = convert_to_contiguous(wavefield_0)
    wavefield_m1 = convert_to_contiguous(wavefield_m1)
    psix_m1 = convert_to_contiguous(psix_m1)
    psiy_m1 = convert_to_contiguous(psiy_m1)
    zetax_m1 = convert_to_contiguous(zetax_m1)
    zetay_m1 = convert_to_contiguous(zetay_m1)
    wavefield_sc_0 = convert_to_contiguous(wavefield_sc_0)
    wavefield_sc_m1 = convert_to_contiguous(wavefield_sc_m1)
    psix_sc_m1 = convert_to_contiguous(psix_sc_m1)
    psiy_sc_m1 = convert_to_contiguous(psiy_sc_m1)
    zetax_sc_m1 = convert_to_contiguous(zetax_sc_m1)
    zetay_sc_m1 = convert_to_contiguous(zetay_sc_m1)
    sources_i = convert_to_contiguous(sources_i)
    receivers_i = convert_to_contiguous(receivers_i)
    ax = convert_to_contiguous(ax)
    ay = convert_to_contiguous(ay)
    bx = convert_to_contiguous(bx)
    by = convert_to_contiguous(by)

    (wfc, wfp, psix, psiy, zetax, zetay, wfcsc, wfpsc, psixsc, psiysc,
     zetaxsc, zetaysc, receiver_amplitudes) = \
        torch.ops.deepwave.scalar_born(v_pad, scatter_pad,
                                       source_amplitudes, wavefield_0,
                                       wavefield_m1, psix_m1, psiy_m1,
                                       zetax_m1, zetay_m1, wavefield_sc_0,
                                       wavefield_sc_m1, psix_sc_m1,
                                       psiy_sc_m1, zetax_sc_m1,
                                       zetay_sc_m1, ax, ay, bx, by,
                                       sources_i.long(),
                                       receivers_i.long(),
                                       dx, dy, dt, nt, n_batch,
                                       step_ratio *
                                       model_gradient_sampling_interval,
                                       accuracy, pml_width[0],
                                       pml_width[1], pml_width[2],
                                       pml_width[3])

    if receiver_amplitudes.numel() > 0:
        if receiver_amplitudes.device == torch.device('cpu'):
            receiver_amplitudes = torch.movedim(receiver_amplitudes, 1, -1)
        else:
            receiver_amplitudes = torch.movedim(receiver_amplitudes, 0, -1)
        receiver_amplitudes = downsample(receiver_amplitudes, step_ratio)

    return (wfc, wfp, psix, psiy, zetax, zetay, wfcsc, wfpsc, psixsc,
            psiysc, zetaxsc, zetaysc, receiver_amplitudes)
