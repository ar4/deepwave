"""Scalar Born wave propagation module.

Implements Born forward modelling (and its adjoint for backpropagation)
of the scalar wave equation. The implementation is similar to that
described in the scalar module, with the addition of a scattered
wavefield that uses 2 / v * scatter * dt^2 * wavefield as the source term.
"""

from typing import Optional, Union, Tuple, List
import torch
from torch import Tensor
from deepwave.common import (setup_propagator,
                             downsample_and_movedim)


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
    def __init__(self, v: Tensor, scatter: Tensor,
                 grid_spacing: Union[int, float, List[float],
                                     Tensor],
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
                wavefield_sc_0: Optional[Tensor] = None,
                wavefield_sc_m1: Optional[Tensor] = None,
                psiy_sc_m1: Optional[Tensor] = None,
                psix_sc_m1: Optional[Tensor] = None,
                zetay_sc_m1: Optional[Tensor] = None,
                zetax_sc_m1: Optional[Tensor] = None,
                origin: Optional[List[int]] = None,
                nt: Optional[int] = None,
                model_gradient_sampling_interval: int = 1) -> Tuple[
                    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                    Tensor]:
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
                           psiy_m1=psiy_m1, psix_m1=psix_m1,
                           zetay_m1=zetay_m1, zetax_m1=zetax_m1,
                           wavefield_sc_0=wavefield_sc_0,
                           wavefield_sc_m1=wavefield_sc_m1,
                           psiy_sc_m1=psiy_sc_m1, psix_sc_m1=psix_sc_m1,
                           zetay_sc_m1=zetay_sc_m1,
                           zetax_sc_m1=zetax_sc_m1,
                           origin=origin, nt=nt,
                           model_gradient_sampling_interval=
                           model_gradient_sampling_interval)


def scalar_born(v: Tensor, scatter: Tensor,
                grid_spacing: Union[int, float, List[float],
                                    Tensor],
                dt: float, source_amplitudes: Optional[Tensor] = None,
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
                time_pad_frac: float = 0.0) -> Tuple[
                    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                    Tensor]:
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
            receiver_amplitudes:
                A Tensor of dimensions [shot, receiver, time] containing
                the receiver amplitudes recorded at the provided receiver
                locations, extracted from the scattered wavefield. If no
                receiver locations were specified then this Tensor will be
                empty.

    """

    (models, source_amplitudes_l, wavefields,
     pml_profiles, sources_i_l, receivers_i_l,
     dy, dx, dt, nt, n_batch,
     step_ratio, model_gradient_sampling_interval,
     accuracy, pml_width_list) = \
        setup_propagator([v, scatter], 'scalar_born', grid_spacing, dt,
                         [wavefield_0, wavefield_m1, psiy_m1, psix_m1,
                          zetay_m1, zetax_m1, wavefield_sc_0, wavefield_sc_m1,
                          psiy_sc_m1, psix_sc_m1, zetay_sc_m1, zetax_sc_m1],
                         [source_amplitudes],
                         [source_locations], [receiver_locations],
                         accuracy, pml_width, pml_freq, max_vel,
                         survey_pad,
                         origin, nt, model_gradient_sampling_interval,
                         freq_taper_frac, time_pad_frac)
    v, scatter = models
    (wfc, wfp, psiy, psix, zetay, zetax, wfcsc, wfpsc, psiysc, psixsc,
     zetaysc, zetaxsc) = wavefields
    source_amplitudes = source_amplitudes_l[0]
    sources_i = sources_i_l[0]
    receivers_i = receivers_i_l[0]
    ay, ax, by, bx = pml_profiles

    (wfc, wfp, psiy, psix, zetay, zetax, wfcsc, wfpsc, psiysc, psixsc,
     zetaysc, zetaxsc, receiver_amplitudes) = \
        torch.ops.deepwave.scalar_born(v, scatter,
                                       source_amplitudes, wfc, wfp,
                                       psiy, psix, zetay, zetax,
                                       wfcsc, wfpsc, psiysc, psixsc,
                                       zetaysc, zetaxsc, ay, ax, by, bx,
                                       sources_i,
                                       receivers_i,
                                       dy, dx, dt, nt, n_batch,
                                       step_ratio *
                                       model_gradient_sampling_interval,
                                       accuracy, pml_width_list[0],
                                       pml_width_list[1], pml_width_list[2],
                                       pml_width_list[3])

    receiver_amplitudes = downsample_and_movedim(receiver_amplitudes,
                                                 step_ratio, freq_taper_frac,
                                                 time_pad_frac)

    return (wfc, wfp, psiy, psix, zetay, zetax, wfcsc, wfpsc, psiysc,
            psixsc, zetaysc, zetaxsc, receiver_amplitudes)
