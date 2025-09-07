"""Scalar Born wave propagation module.

Implements Born forward modelling (and its adjoint for backpropagation)
of the scalar wave equation. The implementation is similar to that
described in the scalar module, with the addition of a scattered
wavefield that uses 2 / v * scatter * dt^2 * wavefield as the source term.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import torch

import deepwave.backend_utils
import deepwave.common
import deepwave.regular_grid


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
        v: A torch.Tensor containing an initial guess of the wavespeed.
        scatter: A torch.Tensor containing an initial guess of the scattering
            potential.
        grid_spacing: The spatial grid cell size. It can be a single number
            that will be used for all dimensions, or a number for each
            dimension.
        v_requires_grad: A bool specifying whether the `requires_grad`
            attribute of the wavespeed should be set, and thus whether the
            necessary data should be stored to calculate the gradient with
            respect to `v` during backpropagation. Defaults to False.
        scatter_requires_grad: A bool specifying whether the `requires_grad`
            attribute of the scattering potential should be set, and thus
            whether the necessary data should be stored to calculate the
            gradient with respect to `scatter` during backpropagation.
            Defaults to False.

    """

    def __init__(
        self,
        v: torch.Tensor,
        scatter: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        v_requires_grad: bool = False,
        scatter_requires_grad: bool = False,
    ) -> None:
        """Initializes the ScalarBorn propagator module.

        Args:
            v: A torch.Tensor containing an initial guess of the wavespeed.
            scatter: A torch.Tensor containing an initial guess of the scattering
                potential.
            grid_spacing: The spatial grid cell size. It can be a single number
                that will be used for all dimensions, or a number for each
                dimension.
            v_requires_grad: A bool specifying whether the `requires_grad`
                attribute of the wavespeed should be set.
            scatter_requires_grad: A bool specifying whether the `requires_grad`
                attribute of the scattering potential should be set.

        """
        super().__init__()
        if not isinstance(v_requires_grad, bool):
            raise TypeError(
                f"v_requires_grad must be bool, got {type(v_requires_grad).__name__}",
            )
        if not isinstance(v, torch.Tensor):
            raise TypeError("v must be a torch.Tensor.")
        if not isinstance(scatter_requires_grad, bool):
            raise TypeError(
                "scatter_requires_grad must be bool, "
                f"got {type(scatter_requires_grad).__name__}",
            )
        if not isinstance(scatter, torch.Tensor):
            raise TypeError("scatter must be a torch.Tensor.")
        self.v = torch.nn.Parameter(v, requires_grad=v_requires_grad)
        self.scatter = torch.nn.Parameter(scatter, requires_grad=scatter_requires_grad)
        self.grid_spacing = grid_spacing

    def forward(
        self,
        dt: float,
        source_amplitudes: Optional[torch.Tensor] = None,
        source_locations: Optional[torch.Tensor] = None,
        receiver_locations: Optional[torch.Tensor] = None,
        bg_receiver_locations: Optional[torch.Tensor] = None,
        accuracy: int = 4,
        pml_width: Union[int, Sequence[int]] = 20,
        pml_freq: Optional[float] = None,
        max_vel: Optional[float] = None,
        survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
        wavefield_0: Optional[torch.Tensor] = None,
        wavefield_m1: Optional[torch.Tensor] = None,
        psiy_m1: Optional[torch.Tensor] = None,
        psix_m1: Optional[torch.Tensor] = None,
        zetay_m1: Optional[torch.Tensor] = None,
        zetax_m1: Optional[torch.Tensor] = None,
        wavefield_sc_0: Optional[torch.Tensor] = None,
        wavefield_sc_m1: Optional[torch.Tensor] = None,
        psiy_sc_m1: Optional[torch.Tensor] = None,
        psix_sc_m1: Optional[torch.Tensor] = None,
        zetay_sc_m1: Optional[torch.Tensor] = None,
        zetax_sc_m1: Optional[torch.Tensor] = None,
        origin: Optional[Sequence[int]] = None,
        nt: Optional[int] = None,
        model_gradient_sampling_interval: int = 1,
        freq_taper_frac: float = 0.0,
        time_pad_frac: float = 0.0,
        time_taper: bool = False,
        forward_callback: Optional[deepwave.common.Callback] = None,
        backward_callback: Optional[deepwave.common.Callback] = None,
        callback_frequency: int = 1,
    ) -> Tuple[torch.Tensor, ...]:
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
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
            forward_callback=forward_callback,
            backward_callback=backward_callback,
            callback_frequency=callback_frequency,
        )


def scalar_born(
    v: torch.Tensor,
    scatter: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitudes: Optional[torch.Tensor] = None,
    source_locations: Optional[torch.Tensor] = None,
    receiver_locations: Optional[torch.Tensor] = None,
    bg_receiver_locations: Optional[torch.Tensor] = None,
    accuracy: int = 4,
    pml_width: Union[int, Sequence[int]] = 20,
    pml_freq: Optional[float] = None,
    max_vel: Optional[float] = None,
    survey_pad: Optional[Union[int, Sequence[Optional[int]]]] = None,
    wavefield_0: Optional[torch.Tensor] = None,
    wavefield_m1: Optional[torch.Tensor] = None,
    psiy_m1: Optional[torch.Tensor] = None,
    psix_m1: Optional[torch.Tensor] = None,
    zetay_m1: Optional[torch.Tensor] = None,
    zetax_m1: Optional[torch.Tensor] = None,
    wavefield_sc_0: Optional[torch.Tensor] = None,
    wavefield_sc_m1: Optional[torch.Tensor] = None,
    psiy_sc_m1: Optional[torch.Tensor] = None,
    psix_sc_m1: Optional[torch.Tensor] = None,
    zetay_sc_m1: Optional[torch.Tensor] = None,
    zetax_sc_m1: Optional[torch.Tensor] = None,
    origin: Optional[Sequence[int]] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    forward_callback: Optional[deepwave.common.Callback] = None,
    backward_callback: Optional[deepwave.common.Callback] = None,
    callback_frequency: int = 1,
) -> Tuple[torch.Tensor, ...]:
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
        v: A torch.Tensor containing the wavespeed.
        scatter: A torch.Tensor containing the scattering potential.
        grid_spacing: The spatial grid cell size.
        dt: The temporal grid cell size.
        source_amplitudes: A torch.Tensor containing the source amplitudes.
        source_locations: A torch.Tensor containing the source locations.
        receiver_locations: A torch.Tensor containing the receiver locations.
        bg_receiver_locations: A torch.Tensor with dimensions
            [shot, receiver, 2], containing the coordinates of the cell
            containing each receiver of the background wavefield. Optional.
            It should have torch.long (int64) datatype. If not provided,
            the output `bg_receiver_amplitudes` torch.Tensor will be empty.
            If backpropagation will be performed, the location of each
            background receiver must be unique within the same shot. Setting
            both coordinates to deepwave.IGNORE_LOCATION will result in the
            receiver being ignored.
        accuracy: The spatial accuracy of the finite difference stencil.
        pml_width: The width of the PML in grid cells.
        pml_freq: The frequency of the PML.
        max_vel: The maximum velocity of the model.
        survey_pad: The padding to apply to the survey area.
        wavefield_0: The wavefield at time step 0.
        wavefield_m1: The wavefield at time step -1.
        psiy_m1: The PML wavefield in the y direction at time step -1.
        psix_m1: The PML wavefield in the x direction at time step -1.
        zetay_m1: The PML wavefield in the y direction at time step -1.
        zetax_m1: The PML wavefield in the x direction at time step -1.
        wavefield_sc_0: The scattered wavefield at time step 0.
        wavefield_sc_m1: The scattered wavefield at time step -1.
        psiy_sc_m1: The scattered PML wavefield in the y direction at time
            step -1.
        psix_sc_m1: The scattered PML wavefield in the x direction at time
            step -1.
        zetay_sc_m1: The scattered PML wavefield in the y direction at time
            step -1.
        zetax_sc_m1: The scattered PML wavefield in the x direction at time
            step -1.
        origin: The origin of the grid.
        nt: The number of time steps.
        model_gradient_sampling_interval: The sampling interval for the
            model gradient.
        freq_taper_frac: The fraction of the frequency to taper.
        time_pad_frac: The fraction of the time to pad.
        time_taper: Whether to taper the time.
        forward_callback: Forward pass callback function.
        backward_callback: Backward pass callback function.
        callback_frequency: Time steps between callbacks.

    Returns:
        Tuple:

            - wavefield_nt: The non-scattered wavefield at the final time step.
            - wavefield_ntm1: The non-scattered wavefield at the
              second-to-last time step.
            - psiy_ntm1: The wavefield related to the PML at the
              second-to-last time step for the non-scattered wavefield.
            - psix_ntm1: The wavefield related to the PML at the
              second-to-last time step for the non-scattered wavefield.
            - zetay_ntm1: The wavefield related to the PML at the
              second-to-last time step for the non-scattered wavefield.
            - zetax_ntm1: The wavefield related to the PML at the
              second-to-last time step for the non-scattered wavefield.
            - wavefield_sc_nt: The scattered wavefield.
            - wavefield_sc_ntm1: The scattered wavefield.
            - psiy_sc_ntm1: The wavefield related to the scattered
              wavefield PML.
            - psix_sc_ntm1: The wavefield related to the scattered
              wavefield PML.
            - zetay_sc_ntm1: The wavefield related to the scattered
              wavefield PML.
            - zetax_sc_ntm1: The wavefield related to the scattered
              wavefield PML.
            - bg_receiver_amplitudes: The receiver amplitudes recorded at the
              provided receiver locations, extracted from the background
              wavefield.
            - receiver_amplitudes: The receiver amplitudes recorded at the
              provided receiver locations, extracted from the scattered
              wavefield.

    """
    v_nonzero = v[v != 0]
    if v_nonzero.numel() > 0:
        min_nonzero_model_vel = v_nonzero.abs().min().item()
    else:
        min_nonzero_model_vel = 0.0
    max_model_vel = v.abs().max().item()
    fd_pad = [accuracy // 2] * 4
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
        [v, scatter],
        ["replicate", "constant"],
        grid_spacing,
        dt,
        [source_amplitudes, source_amplitudes],
        [source_locations, source_locations],
        [bg_receiver_locations, receiver_locations],
        accuracy,
        fd_pad,
        pml_width,
        pml_freq,
        max_vel,
        min_nonzero_model_vel,
        max_model_vel,
        survey_pad,
        [
            wavefield_0,
            wavefield_m1,
            psiy_m1,
            psix_m1,
            zetay_m1,
            zetax_m1,
            wavefield_sc_0,
            wavefield_sc_m1,
            psiy_sc_m1,
            psix_sc_m1,
            zetay_sc_m1,
            zetax_sc_m1,
        ],
        origin,
        nt,
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        2,
    )

    ny, nx = models[0].shape[-2:]
    # Background (multiply source amplitudes by -v^2*dt^2)
    mask = sources_i[0] == deepwave.common.IGNORE_LOCATION
    sources_i_masked = sources_i[0].clone()
    sources_i_masked[mask] = 0
    source_amplitudes_out[0] = (
        -source_amplitudes_out[0]
        * (models[0].view(-1, ny * nx).expand(n_shots, -1).gather(1, sources_i_masked))
        ** 2
        * dt**2
    )
    # Scattered (multiply source amplitudes by -2*v*scatter*dt^2)
    mask = sources_i[1] == deepwave.common.IGNORE_LOCATION
    sources_i_masked = sources_i[1].clone()
    sources_i_masked[mask] = 0
    source_amplitudes_out[1] = (
        -2
        * source_amplitudes_out[1]
        * (models[0].view(-1, ny * nx).expand(n_shots, -1).gather(1, sources_i_masked))
        * (models[1].view(-1, ny * nx).expand(n_shots, -1).gather(1, sources_i_masked))
        * dt**2
    )

    pml_profiles = deepwave.regular_grid.set_pml_profiles(
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

    (
        wfc,
        wfp,
        psiy,
        psix,
        zetay,
        zetax,
        wfcsc,
        wfpsc,
        psiysc,
        psixsc,
        zetaysc,
        zetaxsc,
        receiver_amplitudes,
        receiver_amplitudessc,
    ) = scalar_born_func(
        *models,
        *source_amplitudes_out,
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

    receiver_amplitudes = deepwave.common.downsample_and_movedim(
        receiver_amplitudes,
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
    )
    receiver_amplitudessc = deepwave.common.downsample_and_movedim(
        receiver_amplitudessc,
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
    )

    return (
        wfc,
        wfp,
        psiy,
        psix,
        zetay,
        zetax,
        wfcsc,
        wfpsc,
        psiysc,
        psixsc,
        zetaysc,
        zetaxsc,
        receiver_amplitudes,
        receiver_amplitudessc,
    )


class ScalarBornForwardFunc(torch.autograd.Function):
    """Forward propagation function for scalar Born modeling."""

    @staticmethod
    def forward(
        ctx: Any,
        v: torch.Tensor,
        scatter: torch.Tensor,
        source_amplitudes: torch.Tensor,
        source_amplitudessc: torch.Tensor,
        wfc: torch.Tensor,
        wfp: torch.Tensor,
        psiy: torch.Tensor,
        psix: torch.Tensor,
        zetay: torch.Tensor,
        zetax: torch.Tensor,
        wfcsc: torch.Tensor,
        wfpsc: torch.Tensor,
        psiysc: torch.Tensor,
        psixsc: torch.Tensor,
        zetaysc: torch.Tensor,
        zetaxsc: torch.Tensor,
        ay: torch.Tensor,
        ax: torch.Tensor,
        by: torch.Tensor,
        bx: torch.Tensor,
        dbydy: torch.Tensor,
        dbxdx: torch.Tensor,
        sources_i: torch.Tensor,
        unused_tensor: torch.Tensor,
        receivers_i: torch.Tensor,
        receiverssc_i: torch.Tensor,
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
    ) -> Tuple[torch.Tensor, ...]:
        """Forward propagation of the scalar Born wave equation.

        Args:
            ctx: Context object for backpropagation.
            v: Wavespeed model.
            scatter: Scattering potential model.
            source_amplitudes: Source amplitudes for the background wavefield.
            source_amplitudessc: Source amplitudes for the scattered wavefield.
            wfc: Current wavefield.
            wfp: Previous wavefield.
            psiy: PML variable for the y-direction.
            psix: PML variable for the x-direction.
            zetay: PML variable for the y-direction.
            zetax: PML variable for the x-direction.
            wfcsc: Current scattered wavefield.
            wfpsc: Previous scattered wavefield.
            psiysc: PML variable for the y-direction of the scattered wavefield.
            psixsc: PML variable for the x-direction of the scattered wavefield.
            zetaysc: PML variable for the y-direction of the scattered wavefield.
            zetaxsc: PML variable for the x-direction of the scattered wavefield.
            ay: PML coefficient.
            ax: PML coefficient.
            by: PML coefficient.
            bx: PML coefficient.
            dbydy: Derivative of PML coefficient.
            dbxdx: Derivative of PML coefficient.
            sources_i: Source locations.
            unused_tensor: Unused tensor.
            receivers_i: Receiver locations.
            receiverssc_i: Scattered wavefield receiver locations.
            dy: Grid spacing in the y-direction.
            dx: Grid spacing in the x-direction.
            dt: Time step size.
            nt: Number of time steps.
            step_ratio: Step ratio for storing wavefields.
            accuracy: Accuracy of the finite-difference scheme.
            pml_width: Width of the PML.
            n_shots: Number of shots.
            forward_callback: The forward callback.
            backward_callback: The backward callback.
            callback_frequency: The callback frequency.

        Returns:
            A tuple containing the final wavefields and receiver data.

        """
        del unused_tensor  # Unused.
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
        size_with_batch = (n_shots, *v.shape[-2:])
        wfc = deepwave.common.create_or_pad(
            wfc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        wfp = deepwave.common.create_or_pad(
            wfp,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psiy = deepwave.common.create_or_pad(
            psiy,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psix = deepwave.common.create_or_pad(
            psix,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        zetay = deepwave.common.create_or_pad(
            zetay,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        zetax = deepwave.common.create_or_pad(
            zetax,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        wfcsc = deepwave.common.create_or_pad(
            wfcsc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        wfpsc = deepwave.common.create_or_pad(
            wfpsc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psiysc = deepwave.common.create_or_pad(
            psiysc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psixsc = deepwave.common.create_or_pad(
            psixsc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        zetaysc = deepwave.common.create_or_pad(
            zetaysc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        zetaxsc = deepwave.common.create_or_pad(
            zetaxsc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psiy = deepwave.common.zero_interior(psiy, fd_pad, pml_width, True)
        psix = deepwave.common.zero_interior(psix, fd_pad, pml_width, False)
        zetay = deepwave.common.zero_interior(zetay, fd_pad, pml_width, True)
        zetax = deepwave.common.zero_interior(zetax, fd_pad, pml_width, False)
        psiysc = deepwave.common.zero_interior(psiysc, fd_pad, pml_width, True)
        psixsc = deepwave.common.zero_interior(psixsc, fd_pad, pml_width, False)
        zetaysc = deepwave.common.zero_interior(zetaysc, fd_pad, pml_width, True)
        zetaxsc = deepwave.common.zero_interior(zetaxsc, fd_pad, pml_width, False)

        device = v.device
        dtype = v.dtype
        ny = v.shape[-2]
        nx = v.shape[-1]
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

        v_batched = v.ndim == 3 and v.shape[0] > 1
        scatter_batched = scatter.ndim == 3 and scatter.shape[0] > 1

        if v.requires_grad or scatter.requires_grad:
            w_store.resize_(nt // step_ratio, *wfc.shape)
            w_store.fill_(0)
        if v.requires_grad:
            wsc_store.resize_(nt // step_ratio, *wfc.shape)
            wsc_store.fill_(0)
        if receivers_i.numel() > 0:
            receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
            receiver_amplitudes.fill_(0)
        if receiverssc_i.numel() > 0:
            receiver_amplitudessc.resize_(nt, n_shots, n_receiverssc_per_shot)
            receiver_amplitudessc.fill_(0)

        if v.is_cuda:
            aux = v.get_device()
        elif deepwave.backend_utils.USE_OPENMP:
            aux = min(n_shots, torch.get_num_threads())
        else:
            aux = 1
        forward = deepwave.backend_utils.get_backend_function(
            "scalar_born",
            "forward",
            accuracy,
            dtype,
            v.device,
        )

        if forward_callback is None:
            callback_frequency = nt // step_ratio

        if wfc.numel() > 0 and nt > 0:
            for step in range(0, nt // step_ratio, callback_frequency):
                state = deepwave.common.CallbackState(
                    dt,
                    step,
                    {
                        "wavefield_0": wfc,
                        "wavefield_m1": wfp,
                        "psiy_m1": psiy,
                        "psix_m1": psix,
                        "zetay_m1": zetay,
                        "zetax_m1": zetax,
                        "wavefield_sc_0": wfcsc,
                        "wavefield_sc_m1": wfpsc,
                        "psiy_sc_m1": psiysc,
                        "psix_sc_m1": psixsc,
                        "zetay_sc_m1": zetaysc,
                        "zetax_sc_m1": zetaxsc,
                    },
                    {"v": v, "scatter": scatter},
                    {},
                    [fd_pad] * 4,
                    pml_width,
                )
                if forward_callback is not None:
                    forward_callback(state)
                step_nt = min(nt // step_ratio - step, callback_frequency)
                forward(
                    v.data_ptr(),
                    scatter.data_ptr(),
                    source_amplitudes.data_ptr(),
                    source_amplitudessc.data_ptr(),
                    wfc.data_ptr(),
                    wfp.data_ptr(),
                    psiy.data_ptr(),
                    psix.data_ptr(),
                    psiyn.data_ptr(),
                    psixn.data_ptr(),
                    zetay.data_ptr(),
                    zetax.data_ptr(),
                    wfcsc.data_ptr(),
                    wfpsc.data_ptr(),
                    psiysc.data_ptr(),
                    psixsc.data_ptr(),
                    psiynsc.data_ptr(),
                    psixnsc.data_ptr(),
                    zetaysc.data_ptr(),
                    zetaxsc.data_ptr(),
                    w_store.data_ptr(),
                    wsc_store.data_ptr(),
                    receiver_amplitudes.data_ptr(),
                    receiver_amplitudessc.data_ptr(),
                    ay.data_ptr(),
                    ax.data_ptr(),
                    by.data_ptr(),
                    bx.data_ptr(),
                    dbydy.data_ptr(),
                    dbxdx.data_ptr(),
                    sources_i.data_ptr(),
                    receivers_i.data_ptr(),
                    receiverssc_i.data_ptr(),
                    1 / dy,
                    1 / dx,
                    1 / dy**2,
                    1 / dx**2,
                    dt**2,
                    step_nt * step_ratio,
                    n_shots,
                    ny,
                    nx,
                    n_sources_per_shot,
                    n_receivers_per_shot,
                    n_receiverssc_per_shot,
                    step_ratio,
                    v.requires_grad,
                    scatter.requires_grad,
                    v_batched,
                    scatter_batched,
                    step * step_ratio,
                    pml_y0,
                    pml_y1,
                    pml_x0,
                    pml_x1,
                    aux,
                )
                if (step_nt * step_ratio) % 2 != 0:
                    wfc, wfp, psiy, psix, psiyn, psixn = (
                        wfp,
                        wfc,
                        psiyn,
                        psixn,
                        psiy,
                        psix,
                    )
                    wfcsc, wfpsc, psiysc, psixsc, psiynsc, psixnsc = (
                        wfpsc,
                        wfcsc,
                        psiynsc,
                        psixnsc,
                        psiysc,
                        psixsc,
                    )

        if (
            v.requires_grad
            or scatter.requires_grad
            or source_amplitudes.requires_grad
            or source_amplitudessc.requires_grad
            or wfc.requires_grad
            or wfp.requires_grad
            or psiy.requires_grad
            or psix.requires_grad
            or zetay.requires_grad
            or zetax.requires_grad
            or wfcsc.requires_grad
            or wfpsc.requires_grad
            or psiysc.requires_grad
            or psixsc.requires_grad
            or zetaysc.requires_grad
            or zetaxsc.requires_grad
        ):
            ctx.save_for_backward(
                v,
                scatter,
                ay,
                ax,
                by,
                bx,
                dbydy,
                dbxdx,
                sources_i,
                receivers_i,
                receiverssc_i,
                w_store,
                wsc_store,
            )
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
            ctx.backward_callback = backward_callback
            ctx.callback_frequency = callback_frequency
            ctx.non_sc = (
                v.requires_grad
                or source_amplitudes.requires_grad
                or wfc.requires_grad
                or wfp.requires_grad
                or psiy.requires_grad
                or psix.requires_grad
                or zetay.requires_grad
                or zetax.requires_grad
            )

        s = (slice(None), slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        return (
            wfc[s],
            wfp[s],
            psiy[s],
            psix[s],
            zetay[s],
            zetax[s],
            wfcsc[s],
            wfpsc[s],
            psiysc[s],
            psixsc[s],
            zetaysc[s],
            zetaxsc[s],
            receiver_amplitudes,
            receiver_amplitudessc,
        )

    @staticmethod
    @torch.autograd.function.once_differentiable  # type: ignore[misc]
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward propagation of the scalar Born wave equation.

        Args:
            ctx: Context object from the forward pass.
            grad_outputs: Gradients of the outputs of the forward pass.

        Returns:
            A tuple containing the gradients with respect to the inputs of the
            forward pass.

        """
        (
            wfc,
            wfp,
            psiy,
            psix,
            zetay,
            zetax,
            wfcsc,
            wfpsc,
            psiysc,
            psixsc,
            zetaysc,
            zetaxsc,
            grad_r,
            grad_rsc,
        ) = grad_outputs
        (
            v,
            scatter,
            ay,
            ax,
            by,
            bx,
            dbydy,
            dbxdx,
            sources_i,
            receivers_i,
            receiverssc_i,
            w_store,
            wsc_store,
        ) = ctx.saved_tensors

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
        backward_callback = ctx.backward_callback
        callback_frequency = ctx.callback_frequency
        non_sc = ctx.non_sc
        device = v.device
        dtype = v.dtype
        ny = v.shape[-2]
        nx = v.shape[-1]
        n_sources_per_shot = sources_i.numel() // n_shots
        n_receivers_per_shot = receivers_i.numel() // n_shots
        n_receiverssc_per_shot = receiverssc_i.numel() // n_shots
        fd_pad = accuracy // 2

        size_with_batch = (n_shots, *v.shape[-2:])
        if non_sc:
            wfc = deepwave.common.create_or_pad(
                wfc,
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            wfp = deepwave.common.create_or_pad(
                wfp,
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            psiy = deepwave.common.create_or_pad(
                psiy,
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            psix = deepwave.common.create_or_pad(
                psix,
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            zetay = deepwave.common.create_or_pad(
                zetay,
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            zetax = deepwave.common.create_or_pad(
                zetax,
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            psiy = deepwave.common.zero_interior(psiy, fd_pad, pml_width, True)
            psix = deepwave.common.zero_interior(psix, fd_pad, pml_width, False)
            zetay = deepwave.common.zero_interior(zetay, fd_pad, pml_width, True)
            zetax = deepwave.common.zero_interior(zetax, fd_pad, pml_width, False)
            psiyn = torch.zeros_like(psiy)
            psixn = torch.zeros_like(psix)
            zetayn = torch.zeros_like(zetay)
            zetaxn = torch.zeros_like(zetax)
        wfcsc = deepwave.common.create_or_pad(
            wfcsc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        wfpsc = deepwave.common.create_or_pad(
            wfpsc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psiysc = deepwave.common.create_or_pad(
            psiysc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psixsc = deepwave.common.create_or_pad(
            psixsc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        zetaysc = deepwave.common.create_or_pad(
            zetaysc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        zetaxsc = deepwave.common.create_or_pad(
            zetaxsc,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psiysc = deepwave.common.zero_interior(psiysc, fd_pad, pml_width, True)
        psixsc = deepwave.common.zero_interior(psixsc, fd_pad, pml_width, False)
        zetaysc = deepwave.common.zero_interior(zetaysc, fd_pad, pml_width, True)
        zetaxsc = deepwave.common.zero_interior(zetaxsc, fd_pad, pml_width, False)
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

        v_batched = v.ndim == 3 and v.shape[0] > 1
        scatter_batched = scatter.ndim == 3 and scatter.shape[0] > 1

        if source_amplitudes_requires_grad:
            grad_f.resize_(nt, n_shots, n_sources_per_shot)
            grad_f.fill_(0)
        if source_amplitudessc_requires_grad:
            grad_fsc.resize_(nt, n_shots, n_sources_per_shot)
            grad_fsc.fill_(0)

        if v.is_cuda:
            aux = v.get_device()
            if v.requires_grad and not v_batched and n_shots > 1:
                grad_v_tmp.resize_(n_shots, *v.shape[-2:])
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
            if scatter.requires_grad and not scatter_batched and n_shots > 1:
                grad_scatter_tmp.resize_(n_shots, *scatter.shape[-2:])
                grad_scatter_tmp.fill_(0)
                grad_scatter_tmp_ptr = grad_scatter_tmp.data_ptr()
        else:
            if deepwave.backend_utils.USE_OPENMP:
                aux = min(n_shots, torch.get_num_threads())
            else:
                aux = 1
            if (
                v.requires_grad
                and not v_batched
                and aux > 1
                and deepwave.backend_utils.USE_OPENMP
            ):
                grad_v_tmp.resize_(aux, *v.shape[-2:])
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
            if (
                scatter.requires_grad
                and not scatter_batched
                and aux > 1
                and deepwave.backend_utils.USE_OPENMP
            ):
                grad_scatter_tmp.resize_(aux, *scatter.shape[-2:])
                grad_scatter_tmp.fill_(0)
                grad_scatter_tmp_ptr = grad_scatter_tmp.data_ptr()
        backward = deepwave.backend_utils.get_backend_function(
            "scalar_born",
            "backward",
            accuracy,
            dtype,
            v.device,
        )
        backward_sc = deepwave.backend_utils.get_backend_function(
            "scalar_born",
            "backward",
            accuracy,
            dtype,
            v.device,
            extra="_sc",
        )

        wfp = -wfp
        wfpsc = -wfpsc

        if backward_callback is None:
            callback_frequency = nt // step_ratio

        if wfc.numel() > 0 and nt > 0:
            if non_sc:
                for step in range(nt // step_ratio, 0, -callback_frequency):
                    step_nt = min(step, callback_frequency)
                    backward(
                        v.data_ptr(),
                        scatter.data_ptr(),
                        grad_r.data_ptr(),
                        grad_rsc.data_ptr(),
                        wfc.data_ptr(),
                        wfp.data_ptr(),
                        psiy.data_ptr(),
                        psix.data_ptr(),
                        psiyn.data_ptr(),
                        psixn.data_ptr(),
                        zetay.data_ptr(),
                        zetax.data_ptr(),
                        zetayn.data_ptr(),
                        zetaxn.data_ptr(),
                        wfcsc.data_ptr(),
                        wfpsc.data_ptr(),
                        psiysc.data_ptr(),
                        psixsc.data_ptr(),
                        psiynsc.data_ptr(),
                        psixnsc.data_ptr(),
                        zetaysc.data_ptr(),
                        zetaxsc.data_ptr(),
                        zetaynsc.data_ptr(),
                        zetaxnsc.data_ptr(),
                        w_store.data_ptr(),
                        wsc_store.data_ptr(),
                        grad_f.data_ptr(),
                        grad_fsc.data_ptr(),
                        grad_v.data_ptr(),
                        grad_scatter.data_ptr(),
                        grad_v_tmp_ptr,
                        grad_scatter_tmp_ptr,
                        ay.data_ptr(),
                        ax.data_ptr(),
                        by.data_ptr(),
                        bx.data_ptr(),
                        dbydy.data_ptr(),
                        dbxdx.data_ptr(),
                        sources_i.data_ptr(),
                        receivers_i.data_ptr(),
                        receiverssc_i.data_ptr(),
                        1 / dy,
                        1 / dx,
                        1 / dy**2,
                        1 / dx**2,
                        dt**2,
                        step_nt * step_ratio,
                        n_shots,
                        ny,
                        nx,
                        n_sources_per_shot * source_amplitudes_requires_grad,
                        n_sources_per_shot * source_amplitudessc_requires_grad,
                        n_receivers_per_shot,
                        n_receiverssc_per_shot,
                        step_ratio,
                        v.requires_grad,
                        scatter.requires_grad,
                        v_batched,
                        scatter_batched,
                        step * step_ratio,
                        pml_y0,
                        pml_y1,
                        pml_x0,
                        pml_x1,
                        aux,
                    )
                    if (step_nt * step_ratio) % 2 != 0:
                        (
                            wfc,
                            wfp,
                            psiy,
                            psix,
                            psiyn,
                            psixn,
                            zetay,
                            zetax,
                            zetayn,
                            zetaxn,
                        ) = (
                            wfp,
                            wfc,
                            psiyn,
                            psixn,
                            psiy,
                            psix,
                            zetayn,
                            zetaxn,
                            zetay,
                            zetax,
                        )
                        (
                            wfcsc,
                            wfpsc,
                            psiysc,
                            psixsc,
                            psiynsc,
                            psixnsc,
                            zetaysc,
                            zetaxsc,
                            zetaynsc,
                            zetaxnsc,
                        ) = (
                            wfpsc,
                            wfcsc,
                            psiynsc,
                            psixnsc,
                            psiysc,
                            psixsc,
                            zetaynsc,
                            zetaxnsc,
                            zetaysc,
                            zetaxsc,
                        )
                    if backward_callback is not None:
                        state = deepwave.common.CallbackState(
                            dt,
                            step - 1,
                            {
                                "wavefield_0": wfc,
                                "wavefield_m1": wfp,
                                "psiy_m1": psiy,
                                "psix_m1": psix,
                                "zetay_m1": zetay,
                                "zetax_m1": zetax,
                                "wavefield_sc_0": wfcsc,
                                "wavefield_sc_m1": wfpsc,
                                "psiy_sc_m1": psiysc,
                                "psix_sc_m1": psixsc,
                                "zetay_sc_m1": zetaysc,
                                "zetax_sc_m1": zetaxsc,
                            },
                            {"v": v, "scatter": scatter},
                            {"v": grad_v, "scatter": grad_scatter},
                            [fd_pad] * 4,
                            pml_width,
                        )
                        backward_callback(state)
            else:
                for step in range(nt // step_ratio, 0, -callback_frequency):
                    step_nt = min(step, callback_frequency)
                    backward_sc(
                        v.data_ptr(),
                        grad_rsc.data_ptr(),
                        wfcsc.data_ptr(),
                        wfpsc.data_ptr(),
                        psiysc.data_ptr(),
                        psixsc.data_ptr(),
                        psiynsc.data_ptr(),
                        psixnsc.data_ptr(),
                        zetaysc.data_ptr(),
                        zetaxsc.data_ptr(),
                        zetaynsc.data_ptr(),
                        zetaxnsc.data_ptr(),
                        w_store.data_ptr(),
                        grad_fsc.data_ptr(),
                        grad_scatter.data_ptr(),
                        grad_scatter_tmp_ptr,
                        ay.data_ptr(),
                        ax.data_ptr(),
                        by.data_ptr(),
                        bx.data_ptr(),
                        dbydy.data_ptr(),
                        dbxdx.data_ptr(),
                        sources_i.data_ptr(),
                        receiverssc_i.data_ptr(),
                        1 / dy,
                        1 / dx,
                        1 / dy**2,
                        1 / dx**2,
                        dt**2,
                        step_nt * step_ratio,
                        n_shots,
                        ny,
                        nx,
                        n_sources_per_shot * source_amplitudessc_requires_grad,
                        n_receiverssc_per_shot,
                        step_ratio,
                        scatter.requires_grad,
                        v_batched,
                        scatter_batched,
                        step * step_ratio,
                        pml_y0,
                        pml_y1,
                        pml_x0,
                        pml_x1,
                        aux,
                    )
                    if (step_nt * step_ratio) % 2 != 0:
                        (
                            wfcsc,
                            wfpsc,
                            psiysc,
                            psixsc,
                            psiynsc,
                            psixnsc,
                            zetaysc,
                            zetaxsc,
                            zetaynsc,
                            zetaxnsc,
                        ) = (
                            wfpsc,
                            wfcsc,
                            psiynsc,
                            psixnsc,
                            psiysc,
                            psixsc,
                            zetaynsc,
                            zetaxnsc,
                            zetaysc,
                            zetaxsc,
                        )
                    if backward_callback is not None:
                        state = deepwave.common.CallbackState(
                            dt,
                            step - 1,
                            {
                                "wavefield_sc_0": wfcsc,
                                "wavefield_sc_m1": wfpsc,
                                "psiy_sc_m1": psiysc,
                                "psix_sc_m1": psixsc,
                                "zetay_sc_m1": zetaysc,
                                "zetax_sc_m1": zetaxsc,
                            },
                            {"v": v, "scatter": scatter},
                            {"scatter": grad_scatter},
                            [fd_pad] * 4,
                            pml_width,
                        )
                        backward_callback(state)

        s = (slice(None), slice(fd_pad, -fd_pad), slice(fd_pad, -fd_pad))
        if non_sc:
            return (
                grad_v,
                grad_scatter,
                grad_f,
                grad_fsc,
                wfc[s],
                -wfp[s],
                psiy[s],
                psix[s],
                zetay[s],
                zetax[s],
                wfcsc[s],
                -wfpsc[s],
                psiysc[s],
                psixsc[s],
                zetaysc[s],
                zetaxsc[s],
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
        return (
            None,
            grad_scatter,
            None,
            grad_fsc,
            None,
            None,
            None,
            None,
            None,
            None,
            wfcsc[s],
            -wfpsc[s],
            psiysc[s],
            psixsc[s],
            zetaysc[s],
            zetaxsc[s],
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


def scalar_born_func(
    *args: Any,
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
]:
    """Helper function to apply the ScalarBornForwardFunc.

    This function serves as a convenient wrapper to call the `apply` method
    of `ScalarBornForwardFunc`, which is the entry point for the autograd graph
    for scalar Born wave propagation.

    Args:
        *args: Variable length argument list to be passed directly to
            `ScalarBornForwardFunc.apply`.

    Returns:
        The results of the forward pass from `ScalarBornForwardFunc.apply`.

    """
    return cast(
        "Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "
        "torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "
        "torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "
        "torch.Tensor, torch.Tensor]",
        ScalarBornForwardFunc.apply(*args),  # type: ignore[no-untyped-call]
    )
