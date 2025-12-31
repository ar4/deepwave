"""Scalar Born wave propagation module.

Implements Born forward modelling (and its adjoint for backpropagation)
of the scalar wave equation. The implementation is similar to that
described in the scalar module, with the addition of a scattered
wavefield that uses 2 / v * scatter * dt^2 * wavefield as the source term.
"""

from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, cast

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
        storage_mode: A string specifying the storage mode for intermediate
            data. One of "device", "cpu", "disk", or "none". Defaults to "device".
        storage_path: A string specifying the path for disk storage.
            Defaults to ".".
        storage_compression: A bool specifying whether to use compression
            for intermediate data. Defaults to False.

    """

    def __init__(
        self,
        v: torch.Tensor,
        scatter: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        v_requires_grad: bool = False,
        scatter_requires_grad: bool = False,
        storage_mode: Literal["device", "cpu", "disk", "none"] = "device",
        storage_path: str = ".",
        storage_compression: bool = False,
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
            storage_mode: A string specifying the storage mode.
            storage_path: A string specifying the storage path.
            storage_compression: A bool specifying whether to use compression.

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
        self.storage_mode = storage_mode
        self.storage_path = storage_path
        self.storage_compression = storage_compression

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
        psiz_m1: Optional[torch.Tensor] = None,
        psiy_m1: Optional[torch.Tensor] = None,
        psix_m1: Optional[torch.Tensor] = None,
        zetaz_m1: Optional[torch.Tensor] = None,
        zetay_m1: Optional[torch.Tensor] = None,
        zetax_m1: Optional[torch.Tensor] = None,
        wavefield_sc_0: Optional[torch.Tensor] = None,
        wavefield_sc_m1: Optional[torch.Tensor] = None,
        psiz_sc_m1: Optional[torch.Tensor] = None,
        psiy_sc_m1: Optional[torch.Tensor] = None,
        psix_sc_m1: Optional[torch.Tensor] = None,
        zetaz_sc_m1: Optional[torch.Tensor] = None,
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
        python_backend: Union[Literal["eager", "jit", "compile"], bool] = False,
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
            psiz_m1=psiz_m1,
            psiy_m1=psiy_m1,
            psix_m1=psix_m1,
            zetaz_m1=zetaz_m1,
            zetay_m1=zetay_m1,
            zetax_m1=zetax_m1,
            wavefield_sc_0=wavefield_sc_0,
            wavefield_sc_m1=wavefield_sc_m1,
            psiz_sc_m1=psiz_sc_m1,
            psiy_sc_m1=psiy_sc_m1,
            psix_sc_m1=psix_sc_m1,
            zetaz_sc_m1=zetaz_sc_m1,
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
            python_backend=python_backend,
            storage_mode=self.storage_mode,
            storage_path=self.storage_path,
            storage_compression=self.storage_compression,
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
    psiz_m1: Optional[torch.Tensor] = None,
    psiy_m1: Optional[torch.Tensor] = None,
    psix_m1: Optional[torch.Tensor] = None,
    zetaz_m1: Optional[torch.Tensor] = None,
    zetay_m1: Optional[torch.Tensor] = None,
    zetax_m1: Optional[torch.Tensor] = None,
    wavefield_sc_0: Optional[torch.Tensor] = None,
    wavefield_sc_m1: Optional[torch.Tensor] = None,
    psiz_sc_m1: Optional[torch.Tensor] = None,
    psiy_sc_m1: Optional[torch.Tensor] = None,
    psix_sc_m1: Optional[torch.Tensor] = None,
    zetaz_sc_m1: Optional[torch.Tensor] = None,
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
    python_backend: Union[Literal["eager", "jit", "compile"], bool] = False,
    storage_mode: Literal["device", "cpu", "disk", "none"] = "device",
    storage_path: str = ".",
    storage_compression: bool = False,
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
            [shot, receiver, ndim], containing the coordinates of the cell
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
        psiz_m1: The PML wavefield in the z direction at time step -1.
        psiy_m1: The PML wavefield in the y direction at time step -1.
        psix_m1: The PML wavefield in the x direction at time step -1.
        zetaz_m1: The PML wavefield in the z direction at time step -1.
        zetay_m1: The PML wavefield in the y direction at time step -1.
        zetax_m1: The PML wavefield in the x direction at time step -1.
        wavefield_sc_0: The scattered wavefield at time step 0.
        wavefield_sc_m1: The scattered wavefield at time step -1.
        psiz_sc_m1: The scattered PML wavefield in the z direction at time
            step -1.
        psiy_sc_m1: The scattered PML wavefield in the y direction at time
            step -1.
        psix_sc_m1: The scattered PML wavefield in the x direction at time
            step -1.
        zetaz_sc_m1: The scattered PML wavefield in the z direction at time
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
        python_backend: Use Python backend rather than compiled C/CUDA.
            Can be a string specifying whether to use PyTorch's JIT ("jit"),
            torch.compile ("compile"), or eager mode ("eager"). Alternatively
            a boolean can be provided, with True using the Python backend
            with torch.compile, while the default, False, instead uses the
            compiled C/CUDA.
        storage_mode: A string specifying the storage mode.
        storage_path: A string specifying the storage path.
        storage_compression: A bool specifying whether to use compression.

    Returns:
        Tuple:

            - wavefield_nt: The non-scattered wavefield at the final time step.
            - wavefield_ntm1: The non-scattered wavefield at the
              second-to-last time step.
            - psiz_ntm1: The wavefield related to the PML at the
              second-to-last time step for the non-scattered wavefield (ndim == 3).
            - psiy_ntm1: The wavefield related to the PML at the
              second-to-last time step for the non-scattered wavefield (ndim >= 2).
            - psix_ntm1: The wavefield related to the PML at the
              second-to-last time step for the non-scattered wavefield.
            - zetaz_ntm1: The wavefield related to the PML at the
              second-to-last time step for the non-scattered wavefield (ndim == 3).
            - zetay_ntm1: The wavefield related to the PML at the
              second-to-last time step for the non-scattered wavefield (ndim >= 2).
            - zetax_ntm1: The wavefield related to the PML at the
              second-to-last time step for the non-scattered wavefield.
            - wavefield_sc_nt: The scattered wavefield.
            - wavefield_sc_ntm1: The scattered wavefield.
            - psiz_sc_ntm1: The wavefield related to the scattered
              wavefield PML (ndim == 3).
            - psiy_sc_ntm1: The wavefield related to the scattered
              wavefield PML (ndim >= 2).
            - psix_sc_ntm1: The wavefield related to the scattered
              wavefield PML.
            - zetaz_sc_ntm1: The wavefield related to the scattered
              wavefield PML (ndim == 3).
            - zetay_sc_ntm1: The wavefield related to the scattered
              wavefield PML (ndim >= 2).
            - zetax_sc_ntm1: The wavefield related to the scattered
              wavefield PML.
            - bg_receiver_amplitudes: The receiver amplitudes recorded at the
              provided receiver locations, extracted from the background
              wavefield.
            - receiver_amplitudes: The receiver amplitudes recorded at the
              provided receiver locations, extracted from the scattered
              wavefield.

    """
    deepwave.common.check_inputs_not_vmapped(
        wavefield_0,
        wavefield_m1,
        wavefield_sc_0,
        wavefield_sc_m1,
        source_amplitudes,
        source_locations,
        receiver_locations,
        bg_receiver_locations,
        psiz_m1,
        zetaz_m1,
        psiz_sc_m1,
        zetaz_sc_m1,
        psiy_m1,
        zetay_m1,
        psiy_sc_m1,
        zetay_sc_m1,
        psix_m1,
        zetax_m1,
        psix_sc_m1,
        zetax_sc_m1,
    )
    ndim = deepwave.common.get_ndim(
        [v, scatter],
        [wavefield_0, wavefield_m1, wavefield_sc_0, wavefield_sc_m1],
        [source_locations, receiver_locations, bg_receiver_locations],
        [
            psiz_m1,
            zetaz_m1,
            psiz_sc_m1,
            zetaz_sc_m1,
        ],
        [psiy_m1, zetay_m1, psiy_sc_m1, zetay_sc_m1],
        [psix_m1, zetax_m1, psix_sc_m1, zetax_sc_m1],
    )
    psi: List[Optional[torch.Tensor]] = []
    zeta: List[Optional[torch.Tensor]] = []
    psi_sc: List[Optional[torch.Tensor]] = []
    zeta_sc: List[Optional[torch.Tensor]] = []
    if ndim == 3:
        psi.append(psiz_m1)
        zeta.append(zetaz_m1)
        psi_sc.append(psiz_sc_m1)
        zeta_sc.append(zetaz_sc_m1)
    if ndim >= 2:
        psi.append(psiy_m1)
        zeta.append(zetay_m1)
        psi_sc.append(psiy_sc_m1)
        zeta_sc.append(zetay_sc_m1)
    if ndim >= 1:
        psi.append(psix_m1)
        zeta.append(zetax_m1)
        psi_sc.append(psix_sc_m1)
        zeta_sc.append(zetax_sc_m1)
    initial_wavefields: List[Optional[torch.Tensor]] = [
        wavefield_0,
        wavefield_m1,
        *psi,
        *zeta,
        wavefield_sc_0,
        wavefield_sc_m1,
        *psi_sc,
        *zeta_sc,
    ]

    if deepwave.common.is_inside_vmap(v):
        if max_vel is None:
            raise RuntimeError(
                "If using BatchedTensor inputs, you must specify max_vel"
            )
        max_model_vel = max_vel
        min_nonzero_model_vel = 0.0
    else:
        v_nonzero = v[v != 0]
        if v_nonzero.numel() > 0:
            min_nonzero_model_vel = v_nonzero.abs().min().item()
        else:
            min_nonzero_model_vel = 0.0
        del v_nonzero
        max_model_vel = v.abs().max().item()
    fd_pad = [accuracy // 2] * 2 * ndim
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
        initial_wavefields,
        origin,
        nt,
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        ndim,
    )
    del (
        v,
        scatter,
        source_amplitudes,
        source_locations,
        bg_receiver_locations,
        receiver_locations,
    )

    model_shape = models[0].shape[-ndim:]
    flat_model_shape = int(torch.prod(torch.tensor(model_shape)).item())
    # Background (multiply source amplitudes by -v^2*dt^2)
    mask = sources_i[0] == deepwave.common.IGNORE_LOCATION
    sources_i_masked = sources_i[0].clone()
    sources_i_masked[mask] = 0
    source_amplitudes_out[0] = (
        -source_amplitudes_out[0]
        * (
            models[0]
            .view(-1, flat_model_shape)
            .expand(n_shots, -1)
            .gather(1, sources_i_masked)
        )
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
        * (
            models[0]
            .view(-1, flat_model_shape)
            .expand(n_shots, -1)
            .gather(1, sources_i_masked)
        )
        * (
            models[1]
            .view(-1, flat_model_shape)
            .expand(n_shots, -1)
            .gather(1, sources_i_masked)
        )
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
        model_shape,
    )

    if not isinstance(callback_frequency, int):
        raise TypeError("callback_frequency must be an int.")
    if callback_frequency <= 0:
        raise ValueError("callback_frequency must be positive.")

    outputs = list(
        scalar_born_func(
            python_backend,
            *models,
            *source_amplitudes_out,
            pml_profiles,
            *sources_i,
            *receivers_i,
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
            *wavefields,
        )
    )

    outputs[-2] = deepwave.common.downsample_and_movedim(
        outputs[-2],
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
    )
    outputs[-1] = deepwave.common.downsample_and_movedim(
        outputs[-1],
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
    )

    return tuple(outputs)


class ScalarBornForwardFunc(torch.autograd.Function):
    """Forward propagation function for scalar Born modeling."""

    @staticmethod
    def forward(
        ctx: Any,
        v: torch.Tensor,
        scatter: torch.Tensor,
        source_amplitudes: torch.Tensor,
        source_amplitudessc: torch.Tensor,
        pml_profiles: List[torch.Tensor],
        sources_i: torch.Tensor,
        unused_tensor: torch.Tensor,
        receivers_i: torch.Tensor,
        receiverssc_i: torch.Tensor,
        grid_spacing: Sequence[float],
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
        *wavefields: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward propagation of the scalar Born wave equation.

        Args:
            ctx: Context object for backpropagation.
            v: Wavespeed model.
            scatter: Scattering potential model.
            source_amplitudes: Source amplitudes for the background wavefield.
            source_amplitudessc: Source amplitudes for the scattered wavefield.
            pml_profiles: List of PML profiles.
            sources_i: Source locations.
            unused_tensor: Unused tensor.
            receivers_i: Receiver locations.
            receiverssc_i: Scattered wavefield receiver locations.
            grid_spacing: Grid spacing for each spatial dimension.
            dt: Time step size.
            nt: Number of time steps.
            step_ratio: Step ratio for storing wavefields.
            accuracy: Accuracy of the finite-difference scheme.
            pml_width: Width of the PML.
            n_shots: Number of shots.
            forward_callback: The forward callback.
            backward_callback: The backward callback.
            callback_frequency: The callback frequency.
            storage_mode_str: Storage mode ("device", "cpu", "disk", "none").
            storage_path: Path for disk storage.
            storage_compression: Whether to use compression.
            wavefields: List of wavefields.

        Returns:
            A tuple containing the final wavefields and receiver data.

        """
        del unused_tensor  # Unused.
        v = v.contiguous()
        scatter = scatter.contiguous()
        source_amplitudes = source_amplitudes.contiguous()
        source_amplitudessc = source_amplitudessc.contiguous()
        pml_profiles = [profile.contiguous() for profile in pml_profiles]
        sources_i = sources_i.contiguous()
        receivers_i = receivers_i.contiguous()
        receiverssc_i = receiverssc_i.contiguous()

        device = v.device
        dtype = v.dtype
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

        ndim = len(grid_spacing)
        is_cuda = v.is_cuda
        model_shape = v.shape[-ndim:]
        n_sources_per_shot = sources_i.numel() // n_shots
        n_receivers_per_shot = receivers_i.numel() // n_shots
        n_receiverssc_per_shot = receiverssc_i.numel() // n_shots

        # Storage allocation
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

        for requires_grad in [
            v.requires_grad or scatter.requires_grad,
            v.requires_grad,
        ]:  # w_store and wsc_store
            storage_manager.allocate(requires_grad)

        bg_wavefield_requires_grad = any(
            wavefield.requires_grad for wavefield in wavefields[: 2 + 2 * ndim]
        )
        wavefield_requires_grad = any(
            wavefield.requires_grad for wavefield in wavefields
        )
        if (
            v.requires_grad
            or scatter.requires_grad
            or source_amplitudes.requires_grad
            or source_amplitudessc.requires_grad
            or wavefield_requires_grad
        ):
            ctx.save_for_backward(
                v,
                scatter,
                sources_i,
                receivers_i,
                receiverssc_i,
                *pml_profiles,
            )
            ctx.grid_spacing = grid_spacing
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
                or bg_wavefield_requires_grad
            )
            ctx.storage_manager = storage_manager

        fd_pad = accuracy // 2
        size_with_batch = (n_shots, *v.shape[-ndim:])
        wavefields = tuple(
            deepwave.common.create_or_pad(
                wavefield,
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            for wavefield in wavefields
        )

        wfc, wfp, *pml_wavefields = wavefields[: 2 + 2 * ndim]
        wfcsc, wfpsc, *pml_wavefields_sc = wavefields[2 + 2 * ndim :]
        del wavefields
        # The PML wavefields are ordered [psi, zeta] for each dimension
        psi = list(pml_wavefields[:ndim])
        zeta = list(pml_wavefields[ndim:])
        psisc = list(pml_wavefields_sc[:ndim])
        zetasc = list(pml_wavefields_sc[ndim:])
        del pml_wavefields, pml_wavefields_sc

        for i in range(ndim):
            psi[i] = deepwave.common.zero_interior(psi[i], fd_pad, pml_width, i)
            zeta[i] = deepwave.common.zero_interior(zeta[i], fd_pad, pml_width, i)
            psisc[i] = deepwave.common.zero_interior(psisc[i], fd_pad, pml_width, i)
            zetasc[i] = deepwave.common.zero_interior(zetasc[i], fd_pad, pml_width, i)

        psin = [torch.zeros_like(wf) for wf in psi]
        psinsc = [torch.zeros_like(wf) for wf in psisc]
        pml_b = [
            min(pml_width[2 * i] + 2 * fd_pad, model_shape[i] - fd_pad)
            for i in range(ndim)
        ]
        pml_e = [
            max(pml_b[i], model_shape[i] - pml_width[2 * i + 1] - 2 * fd_pad)
            for i in range(ndim)
        ]

        v_batched = v.ndim == ndim + 1 and v.shape[0] > 1
        scatter_batched = scatter.ndim == ndim + 1 and scatter.shape[0] > 1

        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
        receiver_amplitudessc = torch.empty(0, device=device, dtype=dtype)
        if receivers_i.numel() > 0:
            receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
            receiver_amplitudes.fill_(0)
        if receiverssc_i.numel() > 0:
            receiver_amplitudessc.resize_(nt, n_shots, n_receiverssc_per_shot)
            receiver_amplitudessc.fill_(0)

        stream: Union[int, torch.Stream] = 0
        if is_cuda:
            aux = v.get_device()
            stream = torch.cuda.current_stream(aux)
        elif deepwave.backend_utils.USE_OPENMP:
            aux = min(n_shots, torch.get_num_threads())
        else:
            aux = 1
        forward = deepwave.backend_utils.get_backend_function(
            "scalar_born",
            ndim,
            "forward",
            accuracy,
            dtype,
            v.device,
        )

        rdx = [1 / dx for dx in grid_spacing]
        rdx2 = [1 / dx**2 for dx in grid_spacing]

        if forward_callback is None:
            callback_frequency = nt // step_ratio

        dim_names = ["z", "y", "x"]
        callback_wavefields = {}

        if wfc.numel() > 0 and nt > 0:
            for step in range(0, nt // step_ratio, callback_frequency):
                if forward_callback is not None:
                    callback_wavefields = {
                        "wavefield_0": wfc,
                        "wavefield_m1": wfp,
                        "wavefield_sc_0": wfcsc,
                        "wavefield_sc_m1": wfpsc,
                    }
                    for i in range(ndim):
                        callback_wavefields[f"psi{dim_names[-ndim + i]}_m1"] = psi[i]
                        callback_wavefields[f"zeta{dim_names[-ndim + i]}_m1"] = zeta[i]
                        callback_wavefields[f"psi{dim_names[-ndim + i]}_sc_m1"] = psisc[
                            i
                        ]
                        callback_wavefields[f"zeta{dim_names[-ndim + i]}_sc_m1"] = (
                            zetasc[i]
                        )
                    forward_callback(
                        deepwave.common.CallbackState(
                            dt,
                            step,
                            callback_wavefields,
                            {"v": v, "scatter": scatter},
                            {},
                            [fd_pad] * 2 * ndim,
                            pml_width,
                        )
                    )
                step_nt = min(nt // step_ratio - step, callback_frequency)
                if (
                    forward(
                        v.data_ptr(),
                        scatter.data_ptr(),
                        source_amplitudes.data_ptr(),
                        source_amplitudessc.data_ptr(),
                        wfc.data_ptr(),
                        wfp.data_ptr(),
                        *[field.data_ptr() for field in psi],
                        *[field.data_ptr() for field in psin],
                        *[field.data_ptr() for field in zeta],
                        wfcsc.data_ptr(),
                        wfpsc.data_ptr(),
                        *[field.data_ptr() for field in psisc],
                        *[field.data_ptr() for field in psinsc],
                        *[field.data_ptr() for field in zetasc],
                        *storage_manager.storage_ptrs,
                        receiver_amplitudes.data_ptr(),
                        receiver_amplitudessc.data_ptr(),
                        *[profile.data_ptr() for profile in pml_profiles],
                        sources_i.data_ptr(),
                        receivers_i.data_ptr(),
                        receiverssc_i.data_ptr(),
                        *rdx,
                        *rdx2,
                        dt**2,
                        step_nt * step_ratio,
                        n_shots,
                        *model_shape,
                        n_sources_per_shot,
                        n_receivers_per_shot,
                        n_receiverssc_per_shot,
                        step_ratio,
                        storage_mode,
                        storage_manager.shot_bytes_uncomp,
                        storage_manager.shot_bytes_comp,
                        v.requires_grad
                        and storage_mode != deepwave.common.StorageMode.NONE,
                        scatter.requires_grad
                        and storage_mode != deepwave.common.StorageMode.NONE,
                        v_batched,
                        scatter_batched,
                        storage_compression,
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
                    wfc, wfp, psi, psin = (
                        wfp,
                        wfc,
                        psin,
                        psi,
                    )
                    wfcsc, wfpsc, psisc, psinsc = (
                        wfpsc,
                        wfcsc,
                        psinsc,
                        psisc,
                    )

        s = (slice(None),) + (slice(fd_pad, -fd_pad),) * ndim
        return (
            wfc[s],
            wfp[s],
            *[field[s] for field in psi],
            *[field[s] for field in zeta],
            wfcsc[s],
            wfpsc[s],
            *[field[s] for field in psisc],
            *[field[s] for field in zetasc],
            receiver_amplitudes,
            receiver_amplitudessc,
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: Any,
        *args: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward propagation of the scalar Born wave equation.

        Args:
            ctx: Context object from the forward pass.
            args: Gradients of the outputs of the forward pass.

        Returns:
            A tuple containing the gradients with respect to the inputs of the
            forward pass.

        """
        grad_r = args[-2]
        grad_rsc = args[-1]
        grad_wavefields = list(args[:-2])
        del args
        (
            v,
            scatter,
            sources_i,
            receivers_i,
            receiverssc_i,
            *pml_profiles,
        ) = ctx.saved_tensors

        v = v.contiguous()
        scatter = scatter.contiguous()
        grad_r = grad_r.contiguous()
        grad_rsc = grad_rsc.contiguous()
        grad_wavefields = [p.contiguous() for p in grad_wavefields]
        pml_profiles = [p.contiguous() for p in pml_profiles]
        sources_i = sources_i.contiguous()
        receivers_i = receivers_i.contiguous()
        receiverssc_i = receiverssc_i.contiguous()

        grid_spacing = ctx.grid_spacing
        ndim = len(grid_spacing)
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
        is_cuda = v.is_cuda
        model_shape = v.shape[-ndim:]
        n_sources_per_shot = sources_i.numel() // n_shots
        n_receivers_per_shot = receivers_i.numel() // n_shots
        n_receiverssc_per_shot = receiverssc_i.numel() // n_shots
        fd_pad = accuracy // 2
        storage_manager = ctx.storage_manager

        size_with_batch = (n_shots, *model_shape)
        grad_wavefields[2 + 2 * ndim :] = [
            deepwave.common.create_or_pad(
                wavefield,
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            for wavefield in grad_wavefields[2 + 2 * ndim :]
        ]
        grad_wfcsc, grad_wfpsc, *grad_pml_wavefieldssc = grad_wavefields[2 + 2 * ndim :]
        del grad_wavefields[2 + 2 * ndim :]
        grad_psisc = list(grad_pml_wavefieldssc[:ndim])
        grad_zetasc = list(grad_pml_wavefieldssc[ndim:])
        del grad_pml_wavefieldssc
        for i in range(ndim):
            grad_psisc[i] = deepwave.common.zero_interior(
                grad_psisc[i], fd_pad, pml_width, i
            )
            grad_zetasc[i] = deepwave.common.zero_interior(
                grad_zetasc[i], fd_pad, pml_width, i
            )
        grad_psinsc = [torch.zeros_like(gwf) for gwf in grad_psisc]
        grad_zetansc = [torch.zeros_like(gwf) for gwf in grad_zetasc]

        if non_sc:
            grad_wavefields[: 2 + 2 * ndim] = [
                deepwave.common.create_or_pad(
                    wavefield,
                    fd_pad,
                    v.device,
                    v.dtype,
                    size_with_batch,
                )
                for wavefield in grad_wavefields[: 2 + 2 * ndim]
            ]
            grad_wfc, grad_wfp, *grad_pml_wavefields = grad_wavefields[: 2 + 2 * ndim]
            del grad_wavefields[: 2 + 2 * ndim]
            grad_psi = list(grad_pml_wavefields[:ndim])
            grad_zeta = list(grad_pml_wavefields[ndim:])
            del grad_pml_wavefields
            for i in range(ndim):
                grad_psi[i] = deepwave.common.zero_interior(
                    grad_psi[i], fd_pad, pml_width, i
                )
                grad_zeta[i] = deepwave.common.zero_interior(
                    grad_zeta[i], fd_pad, pml_width, i
                )
            grad_psin = [torch.zeros_like(gwf) for gwf in grad_psi]
            grad_zetan = [torch.zeros_like(gwf) for gwf in grad_zeta]

        del grad_wavefields

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
        pml_b = [
            min(pml_width[2 * i] + 3 * fd_pad, model_shape[i] - fd_pad)
            for i in range(ndim)
        ]
        pml_e = [
            max(pml_b[i], model_shape[i] - pml_width[2 * i + 1] - 3 * fd_pad)
            for i in range(ndim)
        ]

        v_batched = v.ndim == ndim + 1 and v.shape[0] > 1
        scatter_batched = scatter.ndim == ndim + 1 and scatter.shape[0] > 1

        if source_amplitudes_requires_grad:
            grad_f.resize_(nt, n_shots, n_sources_per_shot)
            grad_f.fill_(0)
        if source_amplitudessc_requires_grad:
            grad_fsc.resize_(nt, n_shots, n_sources_per_shot)
            grad_fsc.fill_(0)

        stream: Union[int, torch.Stream] = 0
        if is_cuda:
            aux = v.get_device()
            stream = torch.cuda.current_stream(aux)
            if (
                v.requires_grad
                and not v_batched
                and n_shots > 1
                and storage_manager.storage_mode != deepwave.common.StorageMode.NONE
            ):
                grad_v_tmp.resize_(n_shots, *model_shape)
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
            if (
                scatter.requires_grad
                and not scatter_batched
                and n_shots > 1
                and storage_manager.storage_mode != deepwave.common.StorageMode.NONE
            ):
                grad_scatter_tmp.resize_(n_shots, *model_shape)
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
                and storage_manager.storage_mode != deepwave.common.StorageMode.NONE
            ):
                grad_v_tmp.resize_(aux, *model_shape)
                grad_v_tmp.fill_(0)
                grad_v_tmp_ptr = grad_v_tmp.data_ptr()
            if (
                scatter.requires_grad
                and not scatter_batched
                and aux > 1
                and deepwave.backend_utils.USE_OPENMP
                and storage_manager.storage_mode != deepwave.common.StorageMode.NONE
            ):
                grad_scatter_tmp.resize_(aux, *model_shape)
                grad_scatter_tmp.fill_(0)
                grad_scatter_tmp_ptr = grad_scatter_tmp.data_ptr()
        backward = deepwave.backend_utils.get_backend_function(
            "scalar_born",
            ndim,
            "backward",
            accuracy,
            dtype,
            v.device,
        )
        backward_sc = deepwave.backend_utils.get_backend_function(
            "scalar_born",
            ndim,
            "backward",
            accuracy,
            dtype,
            v.device,
            extra="_sc",
        )

        rdx = [1 / dx for dx in grid_spacing]
        rdx2 = [1 / dx**2 for dx in grid_spacing]

        if non_sc:
            grad_wfp = -grad_wfp
        grad_wfpsc = -grad_wfpsc

        if backward_callback is None:
            callback_frequency = nt // step_ratio

        dim_names = ["z", "y", "x"]

        if grad_wfcsc.numel() > 0 and nt > 0:
            if non_sc:
                for step in range(nt // step_ratio, 0, -callback_frequency):
                    step_nt = min(step, callback_frequency)
                    if (
                        backward(
                            v.data_ptr(),
                            scatter.data_ptr(),
                            grad_r.data_ptr(),
                            grad_rsc.data_ptr(),
                            grad_wfc.data_ptr(),
                            grad_wfp.data_ptr(),
                            *[field.data_ptr() for field in grad_psi],
                            *[field.data_ptr() for field in grad_psin],
                            *[field.data_ptr() for field in grad_zeta],
                            *[field.data_ptr() for field in grad_zetan],
                            grad_wfcsc.data_ptr(),
                            grad_wfpsc.data_ptr(),
                            *[field.data_ptr() for field in grad_psisc],
                            *[field.data_ptr() for field in grad_psinsc],
                            *[field.data_ptr() for field in grad_zetasc],
                            *[field.data_ptr() for field in grad_zetansc],
                            *storage_manager.storage_ptrs,
                            grad_f.data_ptr(),
                            grad_fsc.data_ptr(),
                            grad_v.data_ptr(),
                            grad_scatter.data_ptr(),
                            grad_v_tmp_ptr,
                            grad_scatter_tmp_ptr,
                            *[profile.data_ptr() for profile in pml_profiles],
                            sources_i.data_ptr(),
                            receivers_i.data_ptr(),
                            receiverssc_i.data_ptr(),
                            *rdx,
                            *rdx2,
                            dt**2,
                            step_nt * step_ratio,
                            n_shots,
                            *model_shape,
                            n_sources_per_shot * source_amplitudes_requires_grad,
                            n_sources_per_shot * source_amplitudessc_requires_grad,
                            n_receivers_per_shot,
                            n_receiverssc_per_shot,
                            step_ratio,
                            storage_manager.storage_mode,
                            storage_manager.shot_bytes_uncomp,
                            storage_manager.shot_bytes_comp,
                            v.requires_grad
                            and storage_manager.storage_mode
                            != deepwave.common.StorageMode.NONE,
                            scatter.requires_grad
                            and storage_manager.storage_mode
                            != deepwave.common.StorageMode.NONE,
                            v_batched,
                            scatter_batched,
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
                        (
                            grad_wfc,
                            grad_wfp,
                            grad_psi,
                            grad_psin,
                            grad_zeta,
                            grad_zetan,
                        ) = (
                            grad_wfp,
                            grad_wfc,
                            grad_psin,
                            grad_psi,
                            grad_zetan,
                            grad_zeta,
                        )
                        (
                            grad_wfcsc,
                            grad_wfpsc,
                            grad_psisc,
                            grad_psinsc,
                            grad_zetasc,
                            grad_zetansc,
                        ) = (
                            grad_wfpsc,
                            grad_wfcsc,
                            grad_psinsc,
                            grad_psisc,
                            grad_zetansc,
                            grad_zetasc,
                        )
                    if backward_callback is not None:
                        callback_wavefields = {
                            "wavefield_0": grad_wfc,
                            "wavefield_m1": grad_wfp,
                            "wavefield_sc_0": grad_wfcsc,
                            "wavefield_sc_m1": grad_wfpsc,
                        }
                        for i in range(ndim):
                            callback_wavefields[f"psi{dim_names[-ndim + i]}_m1"] = (
                                grad_psi[i]
                            )
                            callback_wavefields[f"zeta{dim_names[-ndim + i]}_m1"] = (
                                grad_zeta[i]
                            )
                            callback_wavefields[f"psi{dim_names[-ndim + i]}_sc_m1"] = (
                                grad_psisc[i]
                            )
                            callback_wavefields[f"zeta{dim_names[-ndim + i]}_sc_m1"] = (
                                grad_zetasc[i]
                            )
                        backward_callback(
                            deepwave.common.CallbackState(
                                dt,
                                step - step_nt,
                                callback_wavefields,
                                {"v": v, "scatter": scatter},
                                {"v": grad_v, "scatter": grad_scatter},
                                [fd_pad] * 2 * ndim,
                                pml_width,
                            )
                        )
            else:
                for step in range(nt // step_ratio, 0, -callback_frequency):
                    step_nt = min(step, callback_frequency)
                    if (
                        backward_sc(
                            v.data_ptr(),
                            grad_rsc.data_ptr(),
                            grad_wfcsc.data_ptr(),
                            grad_wfpsc.data_ptr(),
                            *[field.data_ptr() for field in grad_psisc],
                            *[field.data_ptr() for field in grad_psinsc],
                            *[field.data_ptr() for field in grad_zetasc],
                            *[field.data_ptr() for field in grad_zetansc],
                            *storage_manager.storage_ptrs[:5],  # Not scattered
                            grad_fsc.data_ptr(),
                            grad_scatter.data_ptr(),
                            grad_scatter_tmp_ptr,
                            *[profile.data_ptr() for profile in pml_profiles],
                            sources_i.data_ptr(),
                            receiverssc_i.data_ptr(),
                            *rdx,
                            *rdx2,
                            dt**2,
                            step_nt * step_ratio,
                            n_shots,
                            *model_shape,
                            n_sources_per_shot * source_amplitudessc_requires_grad,
                            n_receiverssc_per_shot,
                            step_ratio,
                            storage_manager.storage_mode,
                            storage_manager.shot_bytes_uncomp,
                            storage_manager.shot_bytes_comp,
                            scatter.requires_grad
                            and storage_manager.storage_mode
                            != deepwave.common.StorageMode.NONE,
                            v_batched,
                            scatter_batched,
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
                        (
                            grad_wfcsc,
                            grad_wfpsc,
                            grad_psisc,
                            grad_psinsc,
                            grad_zetasc,
                            grad_zetansc,
                        ) = (
                            grad_wfpsc,
                            grad_wfcsc,
                            grad_psinsc,
                            grad_psisc,
                            grad_zetansc,
                            grad_zetasc,
                        )
                    if backward_callback is not None:
                        callback_wavefields = {
                            "wavefield_sc_0": grad_wfcsc,
                            "wavefield_sc_m1": grad_wfpsc,
                        }
                        for i in range(ndim):
                            callback_wavefields[f"psi{dim_names[-ndim + i]}_sc_m1"] = (
                                grad_psisc[i]
                            )
                            callback_wavefields[f"zeta{dim_names[-ndim + i]}_sc_m1"] = (
                                grad_zetasc[i]
                            )
                        backward_callback(
                            deepwave.common.CallbackState(
                                dt,
                                step - step_nt,
                                callback_wavefields,
                                {"v": v, "scatter": scatter},
                                {"scatter": grad_scatter},
                                [fd_pad] * 2 * ndim,
                                pml_width,
                            )
                        )

        s = (
            slice(None),
            *(slice(fd_pad, dim - fd_pad) for dim in model_shape),
        )
        grad_wfpsc = -grad_wfpsc
        if non_sc:
            grad_wfp = -grad_wfp
            return (
                (
                    grad_v,
                    grad_scatter,
                    grad_f,
                    grad_fsc,
                )
                + (None,) * 18
                + (
                    grad_wfc[s],
                    grad_wfp[s],
                    *[gwf[s] for gwf in grad_psi],
                    *[gwf[s] for gwf in grad_zeta],
                    grad_wfcsc[s],
                    grad_wfpsc[s],
                    *[gwf[s] for gwf in grad_psisc],
                    *[gwf[s] for gwf in grad_zetasc],
                )
            )
        return (
            (
                None,
                grad_scatter,
                None,
                grad_fsc,
            )
            + (None,) * (20 + 2 * ndim)
            + (
                grad_wfcsc[s],
                grad_wfpsc[s],
                *[gwf[s] for gwf in grad_psisc],
                *[gwf[s] for gwf in grad_zetasc],
            )
        )


def _forward_step(
    v: torch.Tensor,
    scatter: torch.Tensor,
    grid_spacing: List[float],
    dt: float,
    accuracy: int,
    pml_profiles: List[torch.Tensor],
    wavefields: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Performs a single time step of scalar Born wave propagation."""
    ndim = len(grid_spacing)

    wfc, wfp = wavefields[:2]
    psi = list(wavefields[2 : 2 + ndim])
    zeta = list(wavefields[2 + ndim : 2 + 2 * ndim])
    wfcsc, wfpsc = wavefields[2 + 2 * ndim : 4 + 2 * ndim]
    psisc = list(wavefields[4 + 2 * ndim : 4 + 3 * ndim])
    zetasc = list(wavefields[4 + 3 * ndim : 4 + 4 * ndim])
    del wavefields
    a = pml_profiles[::3]
    b = pml_profiles[1::3]
    dbdx = pml_profiles[2::3]
    del pml_profiles
    dudx = [
        deepwave.regular_grid.diff1(wfc, dim, accuracy, 1 / grid_spacing[dim], ndim)
        for dim in range(ndim)
    ]
    duscdx = [
        deepwave.regular_grid.diff1(wfcsc, dim, accuracy, 1 / grid_spacing[dim], ndim)
        for dim in range(ndim)
    ]

    tmp = [
        (1 + b[dim])
        * deepwave.regular_grid.diff2(
            wfc, dim, accuracy, 1 / grid_spacing[dim] ** 2, ndim
        )
        + dbdx[dim] * dudx[dim]
        + deepwave.regular_grid.diff1(
            a[dim] * psi[dim], dim, accuracy, 1 / grid_spacing[dim], ndim
        )
        for dim in range(ndim)
    ]
    tmpsc = [
        (1 + b[dim])
        * deepwave.regular_grid.diff2(
            wfcsc, dim, accuracy, 1 / grid_spacing[dim] ** 2, ndim
        )
        + dbdx[dim] * duscdx[dim]
        + deepwave.regular_grid.diff1(
            a[dim] * psisc[dim], dim, accuracy, 1 / grid_spacing[dim], ndim
        )
        for dim in range(ndim)
    ]

    w_sum = torch.zeros_like(tmp[0])
    wsc_sum = torch.zeros_like(tmpsc[0])
    for dim in range(ndim):
        w_sum = w_sum + (1 + b[dim]) * tmp[dim] + a[dim] * zeta[dim]
        wsc_sum = wsc_sum + (1 + b[dim]) * tmpsc[dim] + a[dim] * zetasc[dim]

    return list(
        [
            v**2 * dt * dt * w_sum + 2 * wfc - wfp,
            wfc,
        ]
        + [b[dim] * dudx[dim] + a[dim] * psi[dim] for dim in range(ndim)]
        + [b[dim] * tmp[dim] + a[dim] * zeta[dim] for dim in range(ndim)]
        + [
            v**2 * dt * dt * wsc_sum
            + 2 * wfcsc
            - wfpsc
            + 2 * v * scatter * dt * dt * w_sum,
            wfcsc,
        ]
        + [b[dim] * duscdx[dim] + a[dim] * psisc[dim] for dim in range(ndim)]
        + [b[dim] * tmpsc[dim] + a[dim] * zetasc[dim] for dim in range(ndim)]
    )


_forward_step_jit = None
_forward_step_compile = None
_forward_step_opt = _forward_step


def scalar_born_python(
    v: torch.Tensor,
    scatter: torch.Tensor,
    source_amplitudes: torch.Tensor,
    source_amplitudessc: torch.Tensor,
    pml_profiles: List[torch.Tensor],
    sources_i: torch.Tensor,
    unused_tensor: torch.Tensor,
    receivers_i: torch.Tensor,
    receiverssc_i: torch.Tensor,
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
    *wavefields_tuple: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Forward propagation of the scalar Born wave equation.

    Args:
        v: Wavespeed model.
        scatter: Scattering potential model.
        source_amplitudes: Source amplitudes for the background wavefield.
        source_amplitudessc: Source amplitudes for the scattered wavefield.
        pml_profiles: List of PML profiles.
        sources_i: Source locations.
        unused_tensor: Unused tensor.
        receivers_i: Receiver locations.
        receiverssc_i: Scattered wavefield receiver locations.
        grid_spacing: List of grid spacing in each spatial dimension.
        dt: Time step size.
        nt: Number of time steps.
        step_ratio: Step ratio for storing wavefields.
        accuracy: Accuracy of the finite-difference scheme.
        pml_width: Width of the PML.
        n_shots: Number of shots.
        forward_callback: The forward callback.
        backward_callback: The backward callback.
        callback_frequency: The callback frequency.
        storage_mode_str: The storage mode (Unused).
        storage_path: The path to disk storage (Unused).
        storage_compression: Whether to apply compression to storage (Unused).
        wavefields_tuple: Tuple of wavefields.

    Returns:
        A tuple containing the final wavefields and receiver data.

    """
    is_batched = any(
        deepwave.common.is_inside_vmap(x)
        for x in [
            v,
            scatter,
            source_amplitudes,
            source_amplitudessc,
            sources_i,
            receivers_i,
            receiverssc_i,
            *wavefields_tuple,
        ]
    )
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
    del unused_tensor  # Unused.
    fd_pad = accuracy // 2
    size_with_batch = (n_shots, *v.shape[1:])
    wavefields = list(wavefields_tuple)
    del wavefields_tuple
    wavefields = [
        deepwave.common.create_or_pad(
            wavefield,
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        for wavefield in wavefields
    ]

    device = v.device
    dtype = v.dtype
    shape = v.shape[1:]
    flat_shape = int(torch.prod(torch.tensor(shape)).item())
    ndim = len(shape)
    n_receivers_per_shot = receivers_i.numel() // n_shots
    n_receiverssc_per_shot = receiverssc_i.numel() // n_shots
    receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
    receiver_amplitudessc = torch.empty(0, device=device, dtype=dtype)
    receiver_amplitudes_list: List[torch.Tensor] = []
    receiver_amplitudessc_list: List[torch.Tensor] = []

    if receivers_i.numel() > 0 and not is_batched:
        receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
        receiver_amplitudes.fill_(0)
    if receiverssc_i.numel() > 0 and not is_batched:
        receiver_amplitudessc.resize_(nt, n_shots, n_receiverssc_per_shot)
        receiver_amplitudessc.fill_(0)

    source_mask = sources_i != deepwave.common.IGNORE_LOCATION
    sources_i_masked = torch.zeros_like(sources_i)
    sources_i_masked[source_mask] = sources_i[source_mask]
    source_amplitudes_masked = torch.zeros_like(source_amplitudes)
    if source_amplitudes.numel() > 0:
        source_amplitudes_masked[:, source_mask] = source_amplitudes[:, source_mask]
        del source_amplitudes
    source_amplitudessc_masked = torch.zeros_like(source_amplitudessc)
    if source_amplitudessc.numel() > 0:
        source_amplitudessc_masked[:, source_mask] = source_amplitudessc[:, source_mask]
        del source_amplitudessc

    receiver_mask = receivers_i != deepwave.common.IGNORE_LOCATION
    receivers_i_masked = torch.zeros_like(receivers_i)
    if receivers_i.numel() > 0:
        receivers_i_masked[receiver_mask] = receivers_i[receiver_mask]
        del receivers_i
    receiversc_mask = receiverssc_i != deepwave.common.IGNORE_LOCATION
    receiverssc_i_masked = torch.zeros_like(receiverssc_i)
    if receiverssc_i.numel() > 0:
        receiverssc_i_masked[receiversc_mask] = receiverssc_i[receiversc_mask]
        del receiverssc_i

    dim_names = ["z", "y", "x"]

    for step in range(nt // step_ratio):
        if forward_callback is not None and step % callback_frequency == 0:
            callback_wavefields = {
                "wavefield_0": wavefields[0],
                "wavefield_m1": wavefields[1],
                "wavefield_sc_0": wavefields[2 + 2 * ndim],
                "wavefield_sc_m1": wavefields[3 + 2 * ndim],
            }
            for i in range(ndim):
                callback_wavefields[f"psi{dim_names[-ndim + i]}_m1"] = wavefields[2 + i]
                callback_wavefields[f"zeta{dim_names[-ndim + i]}_m1"] = wavefields[
                    2 + ndim + i
                ]
                callback_wavefields[f"psi{dim_names[-ndim + i]}_sc_m1"] = wavefields[
                    4 + 2 * ndim + i
                ]
                callback_wavefields[f"zeta{dim_names[-ndim + i]}_sc_m1"] = wavefields[
                    4 + 3 * ndim + i
                ]

            forward_callback(
                deepwave.common.CallbackState(
                    dt,
                    step,
                    callback_wavefields,
                    {"v": v, "scatter": scatter},
                    {},
                    [fd_pad] * 2 * ndim,
                    pml_width,
                )
            )
        for inner_step in range(step_ratio):
            t = step * step_ratio + inner_step
            if receivers_i_masked.numel() > 0:
                val = wavefields[0].view(-1, flat_shape).gather(1, receivers_i_masked)
                if is_batched:
                    receiver_amplitudes_list.append(val)
                else:
                    receiver_amplitudes[t] = val
            if receiverssc_i_masked.numel() > 0:
                val = (
                    wavefields[2 + 2 * ndim]
                    .view(-1, flat_shape)
                    .gather(1, receiverssc_i_masked)
                )
                if is_batched:
                    receiver_amplitudessc_list.append(val)
                else:
                    receiver_amplitudessc[t] = val
            wavefields = _forward_step_opt(
                v,
                scatter,
                grid_spacing,
                dt,
                accuracy,
                pml_profiles,
                wavefields,
            )
            if source_amplitudes_masked.numel() > 0:
                wavefields[0].view(-1, flat_shape).scatter_add_(
                    1, sources_i_masked, source_amplitudes_masked[t]
                )
                wavefields[2 + 2 * ndim].view(-1, flat_shape).scatter_add_(
                    1, sources_i_masked, source_amplitudessc_masked[t]
                )

    if is_batched and len(receiver_amplitudes_list) > 0:
        receiver_amplitudes = torch.stack(receiver_amplitudes_list)
    if is_batched and len(receiver_amplitudessc_list) > 0:
        receiver_amplitudessc = torch.stack(receiver_amplitudessc_list)

    receiver_amplitudes_masked = torch.zeros_like(receiver_amplitudes)
    if receiver_amplitudes.numel() > 0:
        receiver_amplitudes_masked[:, receiver_mask] = receiver_amplitudes[
            :, receiver_mask
        ]
        del receiver_amplitudes
    receiver_amplitudessc_masked = torch.zeros_like(receiver_amplitudessc)
    if receiver_amplitudessc.numel() > 0:
        receiver_amplitudessc_masked[:, receiversc_mask] = receiver_amplitudessc[
            :, receiversc_mask
        ]
        del receiver_amplitudessc
    s = (
        slice(None),
        *(slice(fd_pad, -fd_pad) for _ in range(ndim)),
    )
    return tuple(
        [wavefield[s] for wavefield in wavefields]
        + [receiver_amplitudes_masked, receiver_amplitudessc_masked]
    )


def scalar_born_func(
    python_backend: Union[Literal["eager", "jit", "compile"], bool] = False,
    *args: Any,
) -> Tuple[torch.Tensor, ...]:
    """Helper function to apply the ScalarBornForwardFunc.

    This function serves as a convenient wrapper to call the `apply` method
    of `ScalarBornForwardFunc`, which is the entry point for the autograd graph
    for scalar Born wave propagation.

    Args:
        python_backend: Bool or string specifying whether to use Python backend.
        *args: Variable length argument list to be passed directly to
            `ScalarBornForwardFunc.apply`.

    Returns:
        The results of the forward pass from `ScalarBornForwardFunc.apply`.

    """
    global _forward_step_jit, _forward_step_compile, _forward_step_opt

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
            if _forward_step_jit is None:
                _forward_step_jit = torch.jit.script(_forward_step)
            _forward_step_opt = _forward_step_jit
        elif mode == "compile":
            if _forward_step_compile is None:
                _forward_step_compile = torch.compile(_forward_step, fullgraph=True)
            _forward_step_opt = _forward_step_compile
        elif mode == "eager":
            _forward_step_opt = _forward_step
        else:
            raise ValueError(f"Unknown python_backend value {mode!r}.")
        return scalar_born_python(*args)

    func = ScalarBornForwardFunc.apply

    return cast(
        "Tuple[torch.Tensor, ...]",
        func(*args),  # type: ignore[no-untyped-call]
    )
