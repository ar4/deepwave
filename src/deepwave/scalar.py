"""Scalar wave propagation module for Deepwave.

Implements scalar wave equation propagation using finite differences in time
(2nd order) and space (user-selectable order: 2, 4, 6, or 8). Supports PML
boundaries and adjoint mode for gradient computation. For PML details, see

    Pasalic, Damir, and Ray McGarry. "Convolutional perfectly matched
        layer for isotropic and anisotropic acoustic wave equations."
        SEG Technical Program Expanded Abstracts 2010.
        Society of Exploration Geophysicists, 2010. 2925-2929.


    Required inputs: wavespeed model (`v`), grid cell size (`dx`), time step (`dt`),
and either source term (`source_amplitudes` and `source_locations`) or number
of time steps (`nt`).
Outputs: final wavefields (including PML) and receiver amplitudes (empty if no
receivers).
All outputs are differentiable with respect to float torch.Tensor inputs.
"""

from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, cast

import torch

import deepwave.backend_utils
import deepwave.common
import deepwave.regular_grid


class Scalar(torch.nn.Module):
    """Convenience nn.Module wrapper for scalar wave propagation.

    Stores `v` and `grid_spacing`. Gradients do not propagate to the
    provided wavespeed. Use the module's `v` attribute to access the wavespeed.
    """

    def __init__(
        self,
        v: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        v_requires_grad: bool = False,
        storage_mode: Literal["device", "cpu", "disk", "none"] = "device",
        storage_path: str = ".",
        storage_compression: bool = False,
    ) -> None:
        """Initializes the Scalar propagator module.

        Args:
            v: A torch.Tensor containing the wavespeed model.
            grid_spacing: The spatial grid cell size. It can be a single number
                that will be used for all dimensions, or a number for each
                dimension.
            v_requires_grad: A bool specifying whether gradients will be
                computed for `v`. Defaults to False.
            storage_mode: A string specifying the storage mode for intermediate
                data. One of "device", "cpu", "disk", or "none". Defaults to "device".
            storage_path: A string specifying the path for disk storage.
                Defaults to ".".
            storage_compression: A bool specifying whether to use compression
                for intermediate data. Defaults to False.

        """
        super().__init__()
        if not isinstance(v_requires_grad, bool):
            raise TypeError(
                f"v_requires_grad must be bool, got {type(v_requires_grad).__name__}",
            )
        if not isinstance(v, torch.Tensor):
            raise TypeError("v must be a torch.Tensor.")
        self.v = torch.nn.Parameter(v, requires_grad=v_requires_grad)
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
    ) -> List[torch.Tensor]:
        """Performs forward propagation/modelling.

        See :func:`scalar` for details.
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
            psiz_m1=psiz_m1,
            psiy_m1=psiy_m1,
            psix_m1=psix_m1,
            zetaz_m1=zetaz_m1,
            zetay_m1=zetay_m1,
            zetax_m1=zetax_m1,
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


def scalar(
    v: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitudes: Optional[torch.Tensor] = None,
    source_locations: Optional[torch.Tensor] = None,
    receiver_locations: Optional[torch.Tensor] = None,
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
) -> List[torch.Tensor]:
    """Scalar wave propagation (functional interface).

    This function performs forward modelling with the scalar wave equation.
    The outputs are differentiable with respect to the wavespeed, the
    source amplitudes, and the initial wavefields.

    For computational performance, multiple shots may be propagated
    simultaneously.

    The function returns the final wavefields and the data recorded at
    the specified receiver locations. All returned torch.Tensors are
    differentiable with respect to the float torch.Tensors in the input.

    Args:
        v: A torch.Tensor containing the wavespeed model.
        grid_spacing: The spatial grid cell size. It can be a single number
            that will be used for all dimensions, or a number for each
            dimension.
        dt: A float specifying the time step interval of the input and
            output (internally a smaller interval may be used in
            propagation to obey the CFL condition for stability).
        source_amplitudes: A torch.Tensor with dimensions [shot, source, time].
            For example, if two shots are propagated simultaneously, each
            containing three sources of one hundred time samples, the shape
            would be [2, 3, 100]. The time dimension length determines the
            number of time steps in the simulation. Optional. If
            provided, `source_locations` must also be specified. If not
            provided, `nt` must be specified.
        source_locations: A torch.Tensor with dimensions [shot, source, ndim],
            containing the index in the two spatial dimensions of the cell
            that each source is located in, relative to the origin of the
            model. Optional. Must be provided if `source_amplitudes` is. It
            should have torch.long (int64) datatype. The location of each
            source must be unique within the same shot (you cannot have two
            sources in the same shot that both have location [1, 2], for
            example). Setting both coordinates to
            deepwave.IGNORE_LOCATION will result in the source being
            ignored.
        receiver_locations: A torch.Tensor with dimensions
            [shot, receiver, ndim], containing the coordinates of the cell
            containing each receiver. Optional. It should have torch.long
            (int64) datatype. If not provided, the output
            `receiver_amplitudes` torch.Tensor will be empty. If
            backpropagation will be performed, the location of each
            receiver must be unique within the same shot.
            Setting both coordinates to deepwave.IGNORE_LOCATION will
            result in the receiver being ignored.
        accuracy: An int specifying the finite difference order of accuracy.
            Possible values are 2, 4, 6, and 8, with larger numbers resulting
            in more accurate results but greater computational cost. Optional,
            with a default of 4.
        pml_width: A single number, or two numbers for each dimension,
            specifying the width (in number of cells) of the PML
            that prevents reflections from the edges of the model.
            If a single value is provided, it will be used for all edges.
            If a sequence is provided, it should contain values for the
            edges in the following order:

            - beginning of first (slowest) dimension
            - end of first dimension
            - beginning of second dimension
            - end of second dimension
            - ...

            Larger values result in smaller reflections, with values of 10
            to 20 being typical. For a reflective rigid surface, set the
            value for that edge to be zero. For example, if your 2D model is
            oriented so that the surface that you want to be reflective is the
            beginning of the second dimension, then you could specify
            `pml_width=[20, 20, 0, 20]`. The wavespeed in the PML region
            is obtained by replicating the values on the edge of the
            model. Optional, default 20.
        pml_freq: A float specifying the frequency that you wish to use when
            constructing the PML. This is usually the dominant frequency
            of the source wavelet. Choosing this value poorly will
            result in the edges becoming more reflective. Optional, default
            25 Hz (assuming `dt` is in seconds).
        max_vel: A float specifying the maximum velocity, which is used when
            applying the CFL condition and when constructing the PML. If
            not specified, the actual maximum absolute wavespeed in the
            model (or portion of it that is used) will be used. The option
            to specify this is provided to allow you to ensure consistency
            between runs even if there are changes in the wavespeed.
            Optional, default None.
        survey_pad: A single value or list of two values for each dimension,
            all of which are either an int or None, specifying whether the
            simulation domain should be restricted to a region surrounding
            the sources and receivers. If you have a large model, but the
            sources and receivers of individual shots only cover a small
            portion of it (such as in a towed streamer survey), then it may
            be wasteful to simulate wave propagation over the whole model.
            This parameter allows you to specify what distance around sources
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
            encountered. Specifying a list, in the following order, allows
            the padding in each direction to be controlled:

            - beginning of first (slowest) dimension
            - end of first dimension
            - beginning of second dimension
            - end of second dimension
            - ...

            Ints and `None` may be mixed, so a `survey_pad` of
            [5, None, None, 10] means that there should be at least 5 cells
            of padding towards the beginning of the first dimension, no
            restriction of the simulation domain towards the end of the
            first dimension or beginning of the second, and 10 cells of
            padding towards the end of the second dimension. If the
            simulation contains one source at [20, 15], one receiver at
            [10, 15], and the model is of shape [40, 50], then the
            extracted simulation domain with this value of `survey_pad`
            would cover the region [5:40, 0:25]. The same simulation
            domain will be used for all shots propagated simultaneously, so
            if this option is used then it is advisable to propagate shots
            that cover similar regions of the model simultaneously so that
            it provides a benefit (if the shots cover opposite ends of a
            dimension, then that whole dimension will be used regardless of
            what `survey_pad` value is used). Optional, default None.
            Cannot be specified if origin is also specified.

        wavefield_0: A torch.Tensor specifying the initial wavefield at time
            step 0. It should have (ndim + 1) dimensions, with the first
            dimension being shot and the subsequent ones corresponding to the
            spatial dimensions. The spatial shape should be equal to the
            simulation domain, which is the extracted model plus the PML. If
            two shots are being propagated simultaneously in a region of size
            [20, 30] extracted from the model for the simulation, and
            `pml_width=[1, 2, 3, 4]`, then `wavefield_0` should be of shape
            [2, 23, 37]. Optional, default all zeros.
        wavefield_m1: A torch.Tensor specifying the initial wavefield at time
            step -1 (using Deepwave's internal time step interval, which may
            be smaller than the user provided one to obey the CFL condition).
            See the entry for `wavefield_0` for more details.
        psiz_m1: PML-related wavefield at time step -1.
        psiy_m1: PML-related wavefield at time step -1.
        psix_m1: PML-related wavefield at time step -1.
        zetaz_m1: PML-related wavefield at time step -1.
        zetay_m1: PML-related wavefield at time step -1.
        zetax_m1: PML-related wavefield at time step -1.
        origin: A list of ints specifying the origin of the provided initial
            wavefields relative to the origin of the model. Only relevant
            if initial wavefields are provided. The origin of a wavefield
            is the cell where the extracted model used for the simulation
            domain began. It does not include the PML. So if a simulation
            is performed using the model region [10:20, 15:30], the origin
            is [10, 15]. Optional, default all zero. Cannot be specified if
            survey_pad is also specified.
        nt: If the source amplitudes are not provided then you must instead
            specify the number of time steps to run the simulation for by
            providing an integer for `nt`. You cannot specify both the
            source amplitudes and `nt`.
        model_gradient_sampling_interval: An int specifying the number of time
            steps between contributions to the model gradient. The gradient
            with respect to the model is an integral over the backpropagation
            time steps. The time sampling frequency of this integral should be
            at least the Nyquist frequency of the source or data (whichever
            is greater). If this Nyquist frequency is substantially less
            than 1/dt, you may wish to reduce the sampling frequency of
            the integral to reduce computational (especially memory) costs.
            Optional, default 1 (integral is sampled every time step
            interval `dt`).
        freq_taper_frac: A float specifying the fraction of the end of the
            source and receiver amplitudes (if present) in the frequency
            domain to cosine taper if they are resampled due to the CFL
            condition. This might be useful to reduce ringing. A value of 0.1
            means that the top 10% of frequencies will be tapered.
            Default 0.0 (no tapering).
        time_pad_frac: A float specifying the amount of zero padding that will
            be added to the source and receiver amplitudes (if present) before
            resampling and removed afterwards, if they are resampled due to
            the CFL condition, as a fraction of their length. This might be
            useful to reduce wraparound artifacts. A value of 0.1 means that
            zero padding of 10% of the number of time samples will be used.
            Default 0.0.
        time_taper: A bool specifying whether to apply a Hann window in time to
            source and receiver amplitudes (if present). This is useful
            during correctness tests of the propagators as it ensures that
            signals taper to zero at their edges in time, avoiding the
            possibility of high frequencies being introduced.
        forward_callback: A function that will be called during the forward
            pass. See :class:`deepwave.common.CallbackState` for the
            state that will be provided to the function.
        backward_callback: A function that will be called during the backward
            pass. See :class:`deepwave.common.CallbackState` for the
            state that will be provided to the function.
        callback_frequency: The number of time steps between calls to the
            callback.
        storage_mode: A string specifying the storage mode for intermediate
            data. One of "device", "cpu", "disk", or "none". Defaults to "device".
        storage_path: A string specifying the path for disk storage.
            Defaults to ".".
        storage_compression: A bool specifying whether to use compression
            for intermediate data. Defaults to False.
        python_backend: Use Python backend rather than compiled C/CUDA.
            Can be a string specifying whether to use PyTorch's JIT ("jit"),
            torch.compile ("compile"), or eager mode ("eager"). Alternatively
            a boolean can be provided, with True using the Python backend
            with torch.compile, while the default, False, instead uses the
            compiled C/CUDA.

    Returns:
        List[torch.Tensor]:

            - wavefield_nt: The wavefield at the final time step.
            - wavefield_ntm1: The wavefield at the second-to-last time step.
            - psiz_ntm1: PML-related wavefield (if ndim == 3).
            - psiy_ntm1: PML-related wavefield (if ndim >= 2).
            - psix_ntm1: PML-related wavefield.
            - zetaz_ntm1: PML-related wavefield (if ndim == 3).
            - zetay_ntm1: PML-related wavefield (if ndim >= 2).
            - zetax_ntm1: PML-related wavefield.
            - receiver_amplitudes: The receiver amplitudes. Empty if no
              receivers were specified.

    """
    deepwave.common.check_inputs_not_vmapped(
        wavefield_0,
        wavefield_m1,
        source_amplitudes,
        source_locations,
        receiver_locations,
        psiz_m1,
        zetaz_m1,
        psiy_m1,
        zetay_m1,
        psix_m1,
        zetax_m1,
    )
    ndim = deepwave.common.get_ndim(
        [v],
        [wavefield_0, wavefield_m1],
        [source_locations, receiver_locations],
        [psiz_m1, zetaz_m1],
        [psiy_m1, zetay_m1],
        [psix_m1, zetax_m1],
    )
    psi: List[Optional[torch.Tensor]] = []
    zeta: List[Optional[torch.Tensor]] = []
    if ndim == 3:
        psi.append(psiz_m1)
        zeta.append(zetaz_m1)
    if ndim >= 2:
        psi.append(psiy_m1)
        zeta.append(zetay_m1)
    if ndim >= 1:
        psi.append(psix_m1)
        zeta.append(zetax_m1)
    initial_wavefields: List[Optional[torch.Tensor]] = [
        wavefield_0,
        wavefield_m1,
        *psi,
        *zeta,
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
        [v],
        ["replicate"],
        grid_spacing,
        dt,
        [source_amplitudes],
        [source_locations],
        [receiver_locations],
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
    del v, source_amplitudes, source_locations, receiver_locations

    # In the finite difference implementation, the source amplitudes we
    # add to the wavefield each time step are multiplied by
    # -v^2 * dt^2 (where v is the velocity model, stored in models[0]).
    # For simplicity, we multiply the amplitudes by that here.
    # We use 0 as the location of the sources that are to be ignored to
    # avoid out-of-bounds accesses to the model. Since these sources will not be
    # used, their amplitudes are not important.
    model_shape = models[0].shape[-ndim:]
    flat_model_shape = int(torch.prod(torch.tensor(model_shape)).item())
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
        scalar_func(
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

    outputs[-1] = deepwave.common.downsample_and_movedim(
        outputs[-1],
        step_ratio,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
    )

    return outputs


class ScalarForwardFunc(torch.autograd.Function):
    """Autograd function for the forward pass of scalar wave propagation.

    This class defines the forward and backward passes for the scalar wave
    equation, allowing PyTorch to compute gradients through the wave propagation
    operation. It interfaces directly with the C/CUDA backend.
    """

    @staticmethod
    def forward(
        ctx: Any,
        v: torch.Tensor,
        source_amplitudes: torch.Tensor,
        pml_profiles: List[torch.Tensor],
        sources_i: torch.Tensor,
        receivers_i: torch.Tensor,
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
        *wavefields_tuple: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Performs the forward propagation of the scalar wave equation.

        This method is called by PyTorch during the forward pass. It prepares
        the input tensors, calls the appropriate C/CUDA function for wave
        propagation, and saves necessary tensors for the backward pass.

        Args:
            ctx: A context object that can be used to save information for
                the backward pass.
            v: The wavespeed model tensor.
            source_amplitudes: The source amplitudes tensor.
            pml_profiles: List of PML profiles.
            sources_i: 1D indices of source locations.
            receivers_i: 1D indices of receiver locations.
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
            storage_path: Path to the directory for disk storage.
            storage_compression: Whether to use compression.
            wavefields_tuple: List of wavefields (current, previous, PML).

        Returns:
            A tuple containing the output wavefields and receiver amplitudes.

        """
        # Ensure all input tensors are contiguous in memory. This is a
        # requirement for the C/CUDA backend, as it expects pointers to
        # contiguous blocks of data.
        v = v.contiguous()
        source_amplitudes = source_amplitudes.contiguous()
        pml_profiles = [profile.contiguous() for profile in pml_profiles]
        sources_i = sources_i.contiguous()
        receivers_i = receivers_i.contiguous()
        wavefields = list(wavefields_tuple)
        del wavefields_tuple

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

        # Get model properties and dimensions.
        ndim = len(grid_spacing)
        is_cuda = v.is_cuda
        model_shape = v.shape[-ndim:]
        n_sources_per_shot = sources_i.numel() // n_shots
        n_receivers_per_shot = receivers_i.numel() // n_shots

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
        storage_manager.allocate(v.requires_grad)

        # If any of the input Tensors require gradients, a backward pass (and
        # possibly a double backward pass) may be performed. Save the Tensors
        # and parameters that will be needed. Some of these, such as the
        # source amplitudes and initial wavefields, are only required for the
        # double backward pass.
        if (
            v.requires_grad
            or source_amplitudes.requires_grad
            or any(wavefield.requires_grad for wavefield in wavefields)
        ):
            ctx.save_for_backward(
                v,
                sources_i,
                receivers_i,
                source_amplitudes,
                *wavefields,
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
            ctx.backward_callback = backward_callback
            ctx.callback_frequency = callback_frequency
            ctx.storage_manager = storage_manager

        # The finite difference (FD) stencil has a radius of `accuracy // 2`.
        # To avoid special boundary handling, the wavefields are padded with
        # this many zeros on each side.
        fd_pad = accuracy // 2
        size_with_batch = (n_shots, *v.shape[-ndim:])
        # The `create_or_pad` function will either create a new zero tensor of
        # the correct size (if the initial wavefield is not provided) or pad
        # the provided initial wavefield.
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

        wfc, wfp, *pml_wavefields = wavefields
        del wavefields
        # The PML wavefields are ordered [psi, zeta] for each dimension
        psi: List[torch.Tensor] = pml_wavefields[:ndim]
        zeta: List[torch.Tensor] = pml_wavefields[ndim:]
        del pml_wavefields

        # The PML auxiliary variables (psi, zeta) are only non-zero in the PML
        # regions. To avoid unnecessary computation in the interior of the
        # domain, their values in the interior are set to zero here. The
        # C/CUDA code will then skip PML calculations for these grid points.
        for i in range(ndim):
            psi[i] = deepwave.common.zero_interior(psi[i], fd_pad, pml_width, i)
            zeta[i] = deepwave.common.zero_interior(zeta[i], fd_pad, pml_width, i)

        # `psiyn` and `psixn` are temporary tensors for updating the PML
        # auxiliary variables `psiy` and `psix`. A temporary variable is
        # needed because the update of other wavefields at a grid point
        # depends on the values of `psiy` and `psix` at neighboring grid
        # points, so they cannot be updated in-place.
        psin = [torch.zeros_like(wf) for wf in psi]

        # The computational domain is divided into regions for performance,
        # and the boundaries of these regions are calculated here. The PML
        # update involves spatial derivatives of PML variables and profiles,
        # which have a stencil radius of `fd_pad`. This causes the PML's
        # effect to extend `fd_pad` cells into the non-PML region. An
        # additional `fd_pad` is therefore added to the PML region width.
        # The `min` and `max` calls ensure the boundaries are valid for
        # small models.
        pml_b = [
            min(pml_width[2 * i] + 2 * fd_pad, model_shape[i] - fd_pad)
            for i in range(ndim)
        ]
        pml_e = [
            max(pml_b[i], model_shape[i] - pml_width[2 * i + 1] - 2 * fd_pad)
            for i in range(ndim)
        ]

        # Check if a different velocity model is provided for each shot (i.e.,
        # if the batch dimension of `v` is greater than 1).
        v_batched = v.ndim == ndim + 1 and v.shape[0] > 1

        # `receiver_amplitudes` will store the recorded data at receiver
        # locations.
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        # If `v` requires gradients, allocate memory for `dwdv`. This tensor
        # stores a partial result for use in the backward pass. The gradient
        # contributions must be sampled at at least the Nyquist frequency of
        # the source. The maximum possible Nyquist frequency corresponds to
        # sampling every `step_ratio` internal time steps, so a snapshot is
        # stored at this rate.
        # If receivers are present, allocate memory for the receiver amplitudes.
        if receivers_i.numel() > 0:
            receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
            receiver_amplitudes.fill_(0)

        # The `aux` variable has different meanings depending on the backend.
        # On GPU, it is the device ID. On CPU, it is the number of threads to
        # use.
        stream: Union[int, torch.Stream] = 0
        if is_cuda:
            aux = v.get_device()
            stream = torch.cuda.current_stream(aux)
        elif deepwave.backend_utils.USE_OPENMP:
            aux = min(n_shots, torch.get_num_threads())
        else:
            aux = 1

        # Get the appropriate C/CUDA backend function to call. This depends on
        # the desired FD accuracy, the data type, and the device.
        forward = deepwave.backend_utils.get_backend_function(
            "scalar",
            ndim,
            "forward",
            accuracy,
            dtype,
            v.device,
        )

        rdx = [1 / dx for dx in grid_spacing]
        rdx2 = [1 / dx**2 for dx in grid_spacing]

        # If no callback is provided, the entire propagation can be done in a
        # single call to the backend function. We achieve this by setting the
        # callback frequency to be the total number of steps.
        if forward_callback is None:
            callback_frequency = nt // step_ratio

        dim_names = ["z", "y", "x"]

        # The main time-stepping loop. The propagation is chunked into segments
        # of `callback_frequency` steps to allow for callbacks.
        if wfc.numel() > 0 and nt > 0:
            for step in range(0, nt // step_ratio, callback_frequency):
                # If a `forward_callback` is provided, call it with the current
                # state.
                if forward_callback is not None:
                    callback_wavefields = {
                        "wavefield_0": wfc,
                        "wavefield_m1": wfp,
                    }
                    for i in range(ndim):
                        callback_wavefields[f"psi{dim_names[-ndim + i]}_m1"] = psi[i]
                        callback_wavefields[f"zeta{dim_names[-ndim + i]}_m1"] = zeta[i]
                    forward_callback(
                        deepwave.common.CallbackState(
                            dt,
                            step,
                            callback_wavefields,
                            {"v": v},
                            {},
                            [fd_pad] * 2 * ndim,
                            pml_width,
                        )
                    )
                # `step_nt` is the number of steps to propagate in this chunk.
                step_nt = min(nt // step_ratio - step, callback_frequency)
                # Call the C/CUDA backend function to perform the wave
                # propagation.
                if (
                    forward(
                        v.data_ptr(),
                        source_amplitudes.data_ptr(),
                        wfc.data_ptr(),
                        wfp.data_ptr(),
                        *[field.data_ptr() for field in psi],
                        *[field.data_ptr() for field in psin],
                        *[field.data_ptr() for field in zeta],
                        *storage_manager.storage_ptrs,
                        receiver_amplitudes.data_ptr(),
                        *[profile.data_ptr() for profile in pml_profiles],
                        sources_i.data_ptr(),
                        receivers_i.data_ptr(),
                        *rdx,
                        *rdx2,
                        dt**2,
                        step_nt * step_ratio,
                        n_shots,
                        *model_shape,
                        n_sources_per_shot,
                        n_receivers_per_shot,
                        step_ratio,
                        storage_mode,
                        storage_manager.shot_bytes_uncomp,
                        storage_manager.shot_bytes_comp,
                        v.requires_grad and storage_mode != 3,
                        v_batched,
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
                # The backend uses ping-pong buffering for the wavefields and
                # PML variables. If an odd number of time steps were performed
                # in the chunk, the pointers to the current and previous time
                # step buffers need to be swapped for the next iteration.
                if (step_nt * step_ratio) % 2 != 0:
                    wfc, wfp = wfp, wfc
                    psi, psin = psin, psi

        # Before returning the final wavefields, remove the padding that was
        # added for the finite difference stencil.
        s = (slice(None),) + (slice(fd_pad, -fd_pad),) * ndim
        # The tuple that is returned contains the Tensors that will be passed
        # to backward.
        return (
            wfc[s],
            wfp[s],
            *[field[s] for field in psi],
            *[field[s] for field in zeta],
            receiver_amplitudes,
        )

    @staticmethod
    def backward(ctx: Any, *args: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Computes the gradients during the backward pass.

        This method is called by PyTorch during the backward pass to compute
        gradients with respect to the inputs of the forward pass.

        Args:
            ctx: A context object that contains information saved during the
                forward pass.
            args: Gradients of the loss with respect to the outputs of forward.

        Returns:
            Gradients with respect to the inputs of the forward pass.

        """
        ndim = len(ctx.grid_spacing)
        grad_r = args[-1]
        grad_wavefields = list(args[:-1])
        del args
        v, sources_i, receivers_i, source_amplitudes = ctx.saved_tensors[:4]
        wavefields = ctx.saved_tensors[4 : 6 + 2 * ndim]
        pml_profiles = ctx.saved_tensors[6 + 2 * ndim : 6 + 5 * ndim]

        grid_spacing = ctx.grid_spacing
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        pml_width = ctx.pml_width
        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad

        result = ScalarBackwardFunc.apply(  # type: ignore[no-untyped-call]
            grad_r,
            v,
            pml_profiles,
            sources_i,
            receivers_i,
            ctx.storage_manager,
            source_amplitudes,
            grid_spacing,
            dt,
            nt,
            n_shots,
            step_ratio,
            accuracy,
            pml_width,
            source_amplitudes_requires_grad,
            ctx.backward_callback,
            ctx.callback_frequency,
            *grad_wavefields,
            *wavefields,
        )
        return (
            result[0],  # grad_v
            result[1],  # grad_source_amplitudes
            None,  # pml_profiles
            None,  # sources_i
            None,  # receivers_i
            None,  # grid_spacing
            None,  # dt
            None,  # nt
            None,  # step_ratio
            None,  # accuracy
            None,  # pml_width
            None,  # n_shots
            None,  # forward_callback
            None,  # backward_callback
            None,  # callback_frequency
            None,  # storage_mode
            None,  # storage_path
            None,  # storage_compression
            *result[2:],  # grad_wavefields
        )


class ScalarBackwardFunc(torch.autograd.Function):
    """Autograd function for the backward pass of scalar wave propagation.

    This class defines the forward and backward passes for computing gradients
    of the scalar wave equation, interfacing directly with the C/CUDA backend.
    It is typically called by `ScalarForwardFunc.backward`.
    """

    @staticmethod
    def forward(
        ctx: Any,
        grad_r: torch.Tensor,
        v: torch.Tensor,
        pml_profiles: List[torch.Tensor],
        sources_i: torch.Tensor,
        receivers_i: torch.Tensor,
        storage_manager: deepwave.common.StorageManager,
        source_amplitudes: torch.Tensor,
        grid_spacing: Sequence[float],
        dt: float,
        nt: int,
        n_shots: int,
        step_ratio: int,
        accuracy: int,
        pml_width: List[int],
        source_amplitudes_requires_grad: bool,
        backward_callback: Optional[deepwave.common.Callback],
        callback_frequency: int,
        *wavefields_tuple: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Performs the backward propagation of the scalar wave equation.

        This method is called by PyTorch during the backward pass to compute
        gradients with respect to the inputs of the forward pass.

        Args:
            ctx: A context object for saving information for the backward pass.
            grad_r: Gradient of the loss with respect to `receiver_amplitudes`.
            v: The wavespeed model tensor.
            pml_profiles: List of PML profiles.
            sources_i: 1D indices of source locations.
            receivers_i: 1D indices of receiver locations.
            storage_manager: The storage manager.
            source_amplitudes: The source amplitudes tensor.
            grid_spacing: Grid spacing for each spatial dimension.
            dt: Time step interval.
            nt: Total number of time steps.
            n_shots: Number of shots in the batch.
            step_ratio: Ratio between user dt and internal dt.
            accuracy: Finite difference accuracy order.
            pml_width: List of PML widths for each side.
            source_amplitudes_requires_grad: If source amplitudes need grads.
            backward_callback: The backward callback.
            callback_frequency: The callback frequency.
            wavefields_tuple: Gradients of the loss with respect to the
                output wavefields from the forward pass, followed by
                initial wavefields (current, previous, PML) from forward pass.

        Returns:
            A tuple containing the gradients of the inputs.

        """
        ndim = len(grid_spacing)
        grad_wavefields = list(wavefields_tuple[: 2 + 2 * ndim])
        wavefields = list(wavefields_tuple[2 + 2 * ndim :])
        del wavefields_tuple
        ctx.save_for_backward(
            v,
            sources_i,
            receivers_i,
            source_amplitudes,
            grad_r,
            *wavefields,
            *grad_wavefields,
            *pml_profiles,
        )
        ctx.grid_spacing = grid_spacing
        ctx.dt = dt
        ctx.nt = nt
        ctx.n_shots = n_shots
        ctx.step_ratio = step_ratio
        ctx.accuracy = accuracy
        ctx.pml_width = pml_width
        ctx.source_amplitudes_requires_grad = source_amplitudes_requires_grad
        ctx.storage_mode = storage_manager.storage_mode
        ctx.storage_path = storage_manager.storage_path
        ctx.storage_compression = storage_manager.storage_compression

        v = v.contiguous()
        sources_i = sources_i.contiguous()
        receivers_i = receivers_i.contiguous()
        source_amplitudes = source_amplitudes.contiguous()
        wavefields = [p.contiguous() for p in wavefields]
        grad_wavefields = [p.contiguous() for p in grad_wavefields]
        pml_profiles = [p.contiguous() for p in pml_profiles]
        grad_r = grad_r.contiguous()

        device = v.device
        dtype = v.dtype
        is_cuda = v.is_cuda
        model_shape = v.shape[-ndim:]
        n_sources_per_shot = sources_i.numel() // n_shots
        n_receivers_per_shot = receivers_i.numel() // n_shots
        fd_pad = accuracy // 2

        size_with_batch = (n_shots, *model_shape)
        grad_wavefields = [
            deepwave.common.create_or_pad(
                wavefield,
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            for wavefield in grad_wavefields
        ]
        grad_wfc, grad_wfp, *grad_pml_wavefields = grad_wavefields
        del grad_wavefields
        # The PML wavefields are ordered [psi, zeta] for each dimension
        grad_psi: List[torch.Tensor] = grad_pml_wavefields[:ndim]
        grad_zeta: List[torch.Tensor] = grad_pml_wavefields[ndim:]
        del grad_pml_wavefields
        for i in range(ndim):
            grad_psi[i] = deepwave.common.zero_interior(
                grad_psi[i], fd_pad, pml_width, i
            )
            grad_zeta[i] = deepwave.common.zero_interior(
                grad_zeta[i], fd_pad, pml_width, i
            )

        # In the backward pass, temporary tensors are needed for the gradients
        # with respect to both `psi` and `zeta`, as their current values at
        # neighboring grid points are required for the update of other
        # gradient wavefields, preventing in-place updates.
        grad_psin = [torch.zeros_like(gwf) for gwf in grad_psi]
        grad_zetan = [torch.zeros_like(gwf) for gwf in grad_zeta]
        grad_v = torch.empty(0, device=device, dtype=dtype)
        grad_v_tmp = torch.empty(0, device=device, dtype=dtype)
        grad_v_tmp_ptr = grad_v.data_ptr()
        if v.requires_grad:
            grad_v.resize_(*v.shape)
            grad_v.fill_(0)
            grad_v_tmp_ptr = grad_v.data_ptr()
        grad_f = torch.empty(0, device=device, dtype=dtype)

        # In the backward pass, the PML region is extended by an additional
        # `fd_pad` compared to the forward pass (for a total of `3 * fd_pad`).
        # This is because the backward calculations involve spatial
        # derivatives of terms that are themselves spatial derivatives,
        # widening the stencil and the extent of the PML's influence into
        # the non-PML region.
        pml_b = [
            min(pml_width[2 * i] + 3 * fd_pad, model_shape[i] - fd_pad)
            for i in range(ndim)
        ]
        pml_e = [
            max(pml_b[i], model_shape[i] - pml_width[2 * i + 1] - 3 * fd_pad)
            for i in range(ndim)
        ]

        v_batched = v.ndim == ndim + 1 and v.shape[0] > 1

        if source_amplitudes_requires_grad:
            grad_f.resize_(nt, n_shots, n_sources_per_shot)
            grad_f.fill_(0)

        # To avoid race conditions when multiple threads/shots contribute to the
        # model gradient `grad_v` simultaneously, a temporary buffer
        # `grad_v_tmp` is used. Each thread writes to its own slice of this
        # buffer, and the results are summed after the loop.
        # `grad_v_tmp_ptr` points to `grad_v_tmp` if it's needed (e.g. multiple
        # threads on a non-batched model), otherwise it points directly to
        # `grad_v` (e.g., for single-threaded execution or when each shot has
        # its own model and memory space).
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
        backward = deepwave.backend_utils.get_backend_function(
            "scalar",
            ndim,
            "backward",
            accuracy,
            dtype,
            v.device,
        )

        # The backward propagation only requires the `v**2 * dt**2` term, so
        # pre-calculating it saves computation inside the time-stepping loop.
        v2dt2 = v**2 * dt**2

        rdx = [1 / dx for dx in grid_spacing]
        rdx2 = [1 / dx**2 for dx in grid_spacing]

        # A mathematical trick is used to implement the adjoint propagation.
        # The standard adjoint update would require negating a wavefield at
        # each time step. By negating one of the gradient wavefields at the
        # beginning and end of the propagation, a modified update rule can be
        # used that allows for a simple pointer swap between the two gradient
        # wavefields each time step, avoiding the negation within the loop.
        # This initial negation is the first part of that trick.
        grad_wfp = -grad_wfp

        if backward_callback is None:
            callback_frequency = nt // step_ratio

        dim_names = ["z", "y", "x"]

        if grad_wfc.numel() > 0 and nt > 0:
            for step in range(nt // step_ratio, 0, -callback_frequency):
                step_nt = min(step, callback_frequency)
                if (
                    backward(
                        v2dt2.data_ptr(),
                        grad_r.data_ptr(),
                        grad_wfc.data_ptr(),
                        grad_wfp.data_ptr(),
                        *[field.data_ptr() for field in grad_psi],
                        *[field.data_ptr() for field in grad_psin],
                        *[field.data_ptr() for field in grad_zeta],
                        *[field.data_ptr() for field in grad_zetan],
                        *storage_manager.storage_ptrs,
                        grad_f.data_ptr(),
                        grad_v.data_ptr(),
                        grad_v_tmp_ptr,
                        *[profile.data_ptr() for profile in pml_profiles],
                        sources_i.data_ptr(),
                        receivers_i.data_ptr(),
                        *rdx,
                        *rdx2,
                        step_nt * step_ratio,
                        n_shots,
                        *model_shape,
                        n_sources_per_shot * source_amplitudes_requires_grad,
                        n_receivers_per_shot,
                        step_ratio,
                        storage_manager.storage_mode,
                        storage_manager.shot_bytes_uncomp,
                        storage_manager.shot_bytes_comp,
                        v.requires_grad
                        and storage_manager.storage_mode
                        != deepwave.common.StorageMode.NONE,
                        v_batched,
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
                    grad_wfc, grad_wfp = grad_wfp, grad_wfc
                    grad_psi, grad_psin = grad_psin, grad_psi
                    grad_zeta, grad_zetan = grad_zetan, grad_zeta

                if backward_callback is not None:
                    # The time step index is `step - step_nt` because the callback is
                    # executed *after* the calculations for the current
                    # backward step are complete, so the state reflects that
                    # of time step index `step - step_nt`.
                    callback_wavefields = {
                        "wavefield_0": grad_wfc,
                        "wavefield_m1": grad_wfp,
                    }
                    for i in range(ndim):
                        callback_wavefields[f"psi{dim_names[-ndim + i]}_m1"] = grad_psi[
                            i
                        ]
                        callback_wavefields[f"zeta{dim_names[-ndim + i]}_m1"] = (
                            grad_zeta[i]
                        )
                    backward_callback(
                        deepwave.common.CallbackState(
                            dt,
                            step - step_nt,
                            callback_wavefields,
                            {"v": v},
                            {"v": grad_v},
                            [fd_pad] * 2 * ndim,
                            pml_width,
                        )
                    )

        s = (
            slice(None),
            *(slice(fd_pad, dim - fd_pad) for dim in model_shape),
        )
        # The second part of the mathematical trick, which involves negating
        # the other gradient wavefield to transform the propagated state back
        # to the true adjoint state.
        grad_wfp = -grad_wfp
        return (
            grad_v,
            grad_f,
            grad_wfc[s],
            grad_wfp[s],
            *[gwf[s] for gwf in grad_psi],
            *[gwf[s] for gwf in grad_zeta],
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, *args: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Performs double backpropagations.

        Args:
            *args: The gradients with respect to the outputs.
        This method implements the backward pass for `ScalarBackwardFunc`, which
        is itself the backward pass of the forward propagation. This
        "double backward" operation is used to compute second-order
        derivatives of the original objective function with respect to the
        model parameters.

        The primary goal of this function is to compute the Hessian-vector
        product (HVP), `H*h`, where `H` is the Hessian of the original
        scalar loss function `J`, and `h` is a perturbation vector.

        This is achieved using the scalar Born propagator, which enables
        calculation of the effect a perturbation of the inputs will have
        on the gradients.

        The background wavefield of the forward pass will be regular forward
        propagation (repeating what was already done during the original
        forward propagation), while the scattered field will calculate the
        effect of the perturbation on the forward wavefield.

        The scattered field of the backward pass will be regular backpropagation
        (repeating what was already done during the original backpropagation),
        while the background wavefield will represent the effect of the
        velocity perturbation on the backward wavefield.

        Args:
            ctx: A context object that contains information saved during the
                forward pass of the backward function.
            ggv: Gradient of the loss with respect to the gradient of `v`.
            ggf: Gradient of the loss with respect to the gradient of
                `source_amplitudes`.
            ggwfc: Gradient of the loss with respect to the gradient of `wfc`.
            ggwfp: Gradient of the loss with respect to the gradient of `wfp`.
            ggpsiy: Gradient of the loss with respect to the gradient of `psiy`.
            ggpsix: Gradient of the loss with respect to the gradient of `psix`.
            ggzetay: Gradient of the loss with respect to the gradient of
                `zetay`.
            ggzetax: Gradient of the loss with respect to the gradient of
                `zetax`.

        Returns:
            Second-order gradients with respect to the inputs of the
            `ScalarBackwardFunc.forward` method.

        """
        # Retrieve all the tensors saved from the forward and first backward passes.
        # This includes the original forward wavefields (`wfc`, `wfp`), the
        # first-order adjoint sources (`grad_r`), the final state of the first-
        # order adjoint wavefields (`gwfc`, `gwfp`), and model parameters.
        ndim = len(ctx.grid_spacing)
        ggv = args[0]
        ggf = args[1]
        gradgrad_wavefields = list(args[2:])
        del args
        (
            v,
            sources_i,
            receivers_i,
            source_amplitudes,
            grad_r,
        ) = ctx.saved_tensors[:5]
        wavefields = ctx.saved_tensors[5 : 7 + 2 * ndim]
        grad_wavefields = ctx.saved_tensors[7 + 2 * ndim : 9 + 4 * ndim]
        pml_profiles = ctx.saved_tensors[9 + 4 * ndim : 9 + 7 * ndim]
        # Retrieve scalar variables from the context.
        grid_spacing = ctx.grid_spacing
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        pml_width = ctx.pml_width
        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad
        storage_mode = ctx.storage_mode
        storage_path = ctx.storage_path
        storage_compression = ctx.storage_compression

        # If no gradient is provided for a given input (e.g. `ggv`), it means
        # we are not interested in the second derivative with respect to that
        # term. We create a zero tensor as a placeholder for the calculation.
        if ggv.numel() == 0:
            ggv = torch.zeros_like(v.detach())
        if ggf.numel() == 0:
            ggf = torch.zeros_like(source_amplitudes.detach())

        v = v.contiguous()
        ggv = ggv.contiguous()
        source_amplitudes = source_amplitudes.contiguous()
        ggsource_amplitudes = ggf.contiguous()
        grad_r = grad_r.contiguous()
        pml_profiles = [p.contiguous() for p in pml_profiles]
        fwd_sources_i = sources_i.contiguous()

        # Step 1: Forward pass

        # The forward Born propagation will re-calculate the original forward
        # wavefield `u` and the perturbed forward wavefield `delta_u`.
        # We need snapshots of these fields over the whole grid, but we do not
        # need to record `u` at any specific receiver locations for the double
        # backward pass itself, so `fwd_receivers_i` is empty.
        fwd_receivers_i = torch.empty(0, dtype=torch.long, device=v.device)
        # We do, however, need to record `delta_u` at the original receiver
        # locations if we want to compute the gradient with respect to the
        # adjoint source `grad_r`. So we provide the original receiver
        # locations for the scattered field receivers.
        fwd_ggreceivers_i = receivers_i.contiguous()

        fd_pad = accuracy // 2
        model_shape = v.shape[-ndim:]
        size_with_batch = (n_shots, *model_shape)
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
        gradgrad_wavefields = [
            deepwave.common.create_or_pad(
                wavefield,
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            for wavefield in gradgrad_wavefields
        ]
        wfc, wfp, *pml_wavefields = wavefields
        del wavefields
        gradgrad_wfc, gradgrad_wfp, *gradgrad_pml_wavefields = gradgrad_wavefields
        del gradgrad_wavefields
        psi: List[torch.Tensor] = pml_wavefields[:ndim]
        zeta: List[torch.Tensor] = pml_wavefields[ndim:]
        del pml_wavefields
        gradgrad_psi: List[torch.Tensor] = gradgrad_pml_wavefields[:ndim]
        gradgrad_zeta: List[torch.Tensor] = gradgrad_pml_wavefields[ndim:]
        del gradgrad_pml_wavefields
        for i in range(ndim):
            psi[i] = deepwave.common.zero_interior(psi[i], fd_pad, pml_width, i)
            zeta[i] = deepwave.common.zero_interior(zeta[i], fd_pad, pml_width, i)
            gradgrad_psi[i] = deepwave.common.zero_interior(
                gradgrad_psi[i], fd_pad, pml_width, i
            )
            gradgrad_zeta[i] = deepwave.common.zero_interior(
                gradgrad_zeta[i], fd_pad, pml_width, i
            )

        device = v.device
        dtype = v.dtype
        is_cuda = v.is_cuda
        n_sources_per_shot = fwd_sources_i.numel() // n_shots
        n_receivers_per_shot = fwd_receivers_i.numel() // n_shots
        n_ggreceivers_per_shot = fwd_ggreceivers_i.numel() // n_shots
        psin = [torch.zeros_like(wf) for wf in psi]
        gradgrad_psin = [torch.zeros_like(wf) for wf in gradgrad_psi]
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
        ggreceiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
        pml_b = [
            min(pml_width[2 * i] + 2 * fd_pad, model_shape[i] - fd_pad)
            for i in range(ndim)
        ]
        pml_e = [
            max(pml_b[i], model_shape[i] - pml_width[2 * i + 1] - 2 * fd_pad)
            for i in range(ndim)
        ]

        v_batched = v.ndim == ndim + 1 and v.shape[0] > 1
        ggv_batched = ggv.ndim == ndim + 1 and ggv.shape[0] > 1

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
        for requires_grad in [v.requires_grad, v.requires_grad]:
            storage_manager.allocate(requires_grad)

        if fwd_receivers_i.numel() > 0:
            receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
            receiver_amplitudes.fill_(0)
        if fwd_ggreceivers_i.numel() > 0:
            ggreceiver_amplitudes.resize_(nt, n_shots, n_ggreceivers_per_shot)
            ggreceiver_amplitudes.fill_(0)

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

        if (
            wfc.numel() > 0
            and nt > 0
            and forward(
                v.data_ptr(),
                ggv.data_ptr(),
                source_amplitudes.data_ptr(),
                ggsource_amplitudes.data_ptr(),
                wfc.data_ptr(),
                wfp.data_ptr(),
                *[field.data_ptr() for field in psi],
                *[field.data_ptr() for field in psin],
                *[field.data_ptr() for field in zeta],
                gradgrad_wfc.data_ptr(),
                gradgrad_wfp.data_ptr(),
                *[field.data_ptr() for field in gradgrad_psi],
                *[field.data_ptr() for field in gradgrad_psin],
                *[field.data_ptr() for field in gradgrad_zeta],
                *storage_manager.storage_ptrs,
                receiver_amplitudes.data_ptr(),
                ggreceiver_amplitudes.data_ptr(),
                *[profile.data_ptr() for profile in pml_profiles],
                fwd_sources_i.data_ptr(),
                fwd_receivers_i.data_ptr(),
                fwd_ggreceivers_i.data_ptr(),
                *rdx,
                *rdx2,
                dt**2,
                nt,
                n_shots,
                *model_shape,
                n_sources_per_shot,
                n_receivers_per_shot,
                n_ggreceivers_per_shot,
                step_ratio,
                storage_manager.storage_mode,
                storage_manager.shot_bytes_uncomp,
                storage_manager.shot_bytes_comp,
                v.requires_grad
                and storage_manager.storage_mode != deepwave.common.StorageMode.NONE,
                False,
                v_batched,
                ggv_batched,
                storage_manager.storage_compression,
                0,
                *pml_b,
                *pml_e,
                aux,
                stream,
            )
            != 0
        ):
            raise RuntimeError("Compiled backend failed.")

        # Step 2: Backward pass

        bwd_sources_i = sources_i.contiguous()
        bwd_receivers_i = torch.empty(0, dtype=torch.long, device=v.device)
        bwd_greceivers_i = receivers_i.contiguous()

        source_amplitudes_requires_grad = ctx.source_amplitudes_requires_grad
        n_sources_per_shot = bwd_sources_i.numel() // n_shots
        n_receivers_per_shot = bwd_receivers_i.numel() // n_shots
        n_greceivers_per_shot = bwd_greceivers_i.numel() // n_shots

        # Set the initial wavefields for the background propagation to zero.
        wfc = deepwave.common.create_or_pad(
            torch.empty(0),
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        wfp = deepwave.common.create_or_pad(
            torch.empty(0),
            fd_pad,
            v.device,
            v.dtype,
            size_with_batch,
        )
        psi = [
            deepwave.common.create_or_pad(
                torch.empty(0),
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            for _ in range(ndim)
        ]
        zeta = [
            deepwave.common.create_or_pad(
                torch.empty(0),
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            for _ in range(ndim)
        ]
        psin = [torch.zeros_like(wf) for wf in psi]
        zetan = [torch.zeros_like(wf) for wf in zeta]
        grad_wavefields = [
            deepwave.common.create_or_pad(
                wavefield,
                fd_pad,
                v.device,
                v.dtype,
                size_with_batch,
            )
            for wavefield in grad_wavefields
        ]
        grad_wfc, grad_wfp, *grad_pml_wavefields = grad_wavefields
        del grad_wavefields
        grad_psi = grad_pml_wavefields[:ndim]
        grad_zeta = grad_pml_wavefields[ndim:]
        del grad_pml_wavefields
        for i in range(ndim):
            grad_psi[i] = deepwave.common.zero_interior(
                grad_psi[i], fd_pad, pml_width, i
            )
            grad_zeta[i] = deepwave.common.zero_interior(
                grad_zeta[i], fd_pad, pml_width, i
            )
        grad_psin = [torch.zeros_like(wf) for wf in grad_psi]
        grad_zetan = [torch.zeros_like(wf) for wf in grad_zeta]
        grad_v = torch.empty(0, device=device, dtype=dtype)
        grad_v_tmp = torch.empty(0, device=device, dtype=dtype)
        grad_v_tmp_ptr = grad_v.data_ptr()
        if v.requires_grad:
            grad_v.resize_(*v.shape)
            grad_v.fill_(0)
            grad_v_tmp_ptr = grad_v.data_ptr()
        grad_f = torch.empty(0, device=device, dtype=dtype)
        grad_gf = torch.empty(0, device=device, dtype=dtype)
        pml_b = [
            min(pml_width[2 * i] + 3 * fd_pad, model_shape[i] - fd_pad)
            for i in range(ndim)
        ]
        pml_e = [
            max(pml_b[i], model_shape[i] - pml_width[2 * i + 1] - 3 * fd_pad)
            for i in range(ndim)
        ]

        if source_amplitudes_requires_grad:
            grad_f.resize_(nt, n_shots, n_sources_per_shot)
            grad_f.fill_(0)

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

        backward = deepwave.backend_utils.get_backend_function(
            "scalar_born",
            ndim,
            "backward",
            accuracy,
            dtype,
            v.device,
        )

        rdx = [1 / dx for dx in grid_spacing]
        rdx2 = [1 / dx**2 for dx in grid_spacing]

        grad_wfp = -grad_wfp

        if (
            wfc.numel() > 0
            and nt > 0
            and v.requires_grad
            and backward(
                v.data_ptr(),
                ggv.data_ptr(),
                torch.empty(0).data_ptr(),
                grad_r.data_ptr(),
                wfc.data_ptr(),
                wfp.data_ptr(),
                *[field.data_ptr() for field in psi],
                *[field.data_ptr() for field in psin],
                *[field.data_ptr() for field in zeta],
                *[field.data_ptr() for field in zetan],
                grad_wfc.data_ptr(),
                grad_wfp.data_ptr(),
                *[field.data_ptr() for field in grad_psi],
                *[field.data_ptr() for field in grad_psin],
                *[field.data_ptr() for field in grad_zeta],
                *[field.data_ptr() for field in grad_zetan],
                *storage_manager.storage_ptrs,
                grad_f.data_ptr(),
                grad_gf.data_ptr(),
                grad_v.data_ptr(),
                torch.empty(0).data_ptr(),
                grad_v_tmp_ptr,
                torch.empty(0).data_ptr(),
                *[profile.data_ptr() for profile in pml_profiles],
                bwd_sources_i.data_ptr(),
                bwd_receivers_i.data_ptr(),
                bwd_greceivers_i.data_ptr(),
                *rdx,
                *rdx2,
                dt**2,
                nt,
                n_shots,
                *model_shape,
                n_sources_per_shot * source_amplitudes_requires_grad,
                0,
                n_receivers_per_shot,
                n_greceivers_per_shot,
                step_ratio,
                storage_manager.storage_mode,
                storage_manager.shot_bytes_uncomp,
                storage_manager.shot_bytes_comp,
                v.requires_grad
                and storage_manager.storage_mode != deepwave.common.StorageMode.NONE,
                False,
                v_batched,
                ggv_batched,
                storage_manager.storage_compression,
                nt,
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
            *(slice(fd_pad, dim - fd_pad) for dim in model_shape),
        )
        if nt % 2 == 0:
            return (
                ggreceiver_amplitudes,
                grad_v,
                None,
                None,
                None,
                None,
                grad_f,
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
                gradgrad_wfc[s],
                gradgrad_wfp[s],
                *[wf[s] for wf in gradgrad_psi],
                *[wf[s] for wf in gradgrad_zeta],
                wfc[s],
                -wfp[s],
                *[wf[s] for wf in psi],
                *[wf[s] for wf in zeta],
            )
        return (
            ggreceiver_amplitudes,
            grad_v,
            None,
            None,
            None,
            None,
            grad_f,
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
            gradgrad_wfp[s],
            gradgrad_wfc[s],
            *[wf[s] for wf in gradgrad_psin],
            *[wf[s] for wf in gradgrad_zeta],
            wfp[s],
            -wfc[s],
            *[wf[s] for wf in psin],
            *[wf[s] for wf in zetan],
        )


def _forward_step(
    v: torch.Tensor,
    grid_spacing: List[float],
    dt: float,
    accuracy: int,
    pml_profiles: List[torch.Tensor],
    wavefields: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Performs a single time step of scalar wave propagation."""
    ndim = len(grid_spacing)

    wfc, wfp = wavefields[:2]
    psi: List[torch.Tensor] = wavefields[2 : 2 + ndim]
    zeta: List[torch.Tensor] = wavefields[2 + ndim :]
    del wavefields
    a = pml_profiles[::3]
    b = pml_profiles[1::3]
    dbdx = pml_profiles[2::3]
    del pml_profiles
    dudx = [
        deepwave.regular_grid.diff1(wfc, dim, accuracy, 1 / grid_spacing[dim], ndim)
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

    w_sum = torch.zeros_like(tmp[0])
    for dim in range(ndim):
        w_sum = w_sum + (1 + b[dim]) * tmp[dim] + a[dim] * zeta[dim]

    return (
        [
            v**2 * dt * dt * w_sum + 2 * wfc - wfp,
            wfc,
        ]
        + [b[dim] * dudx[dim] + a[dim] * psi[dim] for dim in range(ndim)]
        + [b[dim] * tmp[dim] + a[dim] * zeta[dim] for dim in range(ndim)]
    )


_forward_step_jit = None
_forward_step_compile = None
_forward_step_opt = _forward_step


def scalar_python(
    v: torch.Tensor,
    source_amplitudes: torch.Tensor,
    pml_profiles: List[torch.Tensor],
    sources_i: torch.Tensor,
    receivers_i: torch.Tensor,
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
    """Performs the forward propagation of the scalar wave equation.

    This method is written purely in PyTorch, rather than using the
    compiled C/CUDA code.

    Args:
        v: The wavespeed model tensor.
        source_amplitudes: The source amplitudes tensor.
        pml_profiles: List of PML profiles.
        sources_i: 1D indices of source locations.
        receivers_i: 1D indices of receiver locations.
        grid_spacing: List of grid spacing in each spatial dimension.
        dt: Time step interval.
        nt: Total number of time steps.
        step_ratio: Ratio between user dt and internal dt.
        accuracy: Finite difference accuracy order.
        pml_width: List of PML widths for each side.
        n_shots: Number of shots in the batch.
        forward_callback: The forward callback.
        backward_callback: The backward callback.
        callback_frequency: The callback frequency.
        storage_mode_str: The storage mode (Unused).
        storage_path: The path to disk storage (Unused).
        storage_compression: Whether to apply compression to storage (Unused).
        wavefields_tuple: List of wavefields (current, previous, PML).

    Returns:
        A tuple containing the output wavefields and receiver amplitudes.

    """
    is_batched = any(
        deepwave.common.is_inside_vmap(x)
        for x in [v, source_amplitudes, sources_i, receivers_i, *wavefields_tuple]
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
    receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
    receiver_amplitudes_list: List[torch.Tensor] = []

    if receivers_i.numel() > 0 and not is_batched:
        receiver_amplitudes.resize_(nt, n_shots, n_receivers_per_shot)
        receiver_amplitudes.fill_(0)

    source_mask = sources_i != deepwave.common.IGNORE_LOCATION
    sources_i_masked = torch.zeros_like(sources_i)
    sources_i_masked[source_mask] = sources_i[source_mask]
    source_amplitudes_masked = torch.zeros_like(source_amplitudes)
    if source_amplitudes.numel() > 0:
        source_amplitudes_masked[:, source_mask] = source_amplitudes[:, source_mask]
        del source_amplitudes

    receiver_mask = receivers_i != deepwave.common.IGNORE_LOCATION
    receivers_i_masked = torch.zeros_like(receivers_i)
    if receivers_i.numel() > 0:
        receivers_i_masked[receiver_mask] = receivers_i[receiver_mask]
        del receivers_i

    dim_names = ["z", "y", "x"]

    for step in range(nt // step_ratio):
        if forward_callback is not None and step % callback_frequency == 0:
            wavefield_dict = {
                "wavefield_0": wavefields[0],
                "wavefield_m1": wavefields[1],
            }
            for i in range(ndim):
                wavefield_dict[f"psi{dim_names[-ndim + i]}_m1"] = wavefields[2 + i]
                wavefield_dict[f"zeta{dim_names[-ndim + i]}_m1"] = wavefields[
                    2 + ndim + i
                ]
            forward_callback(
                deepwave.common.CallbackState(
                    dt,
                    step,
                    wavefield_dict,
                    {"v": v},
                    {},
                    [fd_pad] * ndim * 2,
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
            wavefields = _forward_step_opt(
                v, grid_spacing, dt, accuracy, pml_profiles, wavefields
            )
            if source_amplitudes_masked.numel() > 0:
                wavefields[0].view(-1, flat_shape).scatter_add_(
                    1, sources_i_masked, source_amplitudes_masked[t]
                )

    if is_batched and len(receiver_amplitudes_list) > 0:
        receiver_amplitudes = torch.stack(receiver_amplitudes_list)

    receiver_amplitudes_masked = torch.zeros_like(receiver_amplitudes)
    if receiver_amplitudes.numel() > 0:
        receiver_amplitudes_masked[:, receiver_mask] = receiver_amplitudes[
            :, receiver_mask
        ]
        del receiver_amplitudes
    s = (
        slice(None),
        *(slice(fd_pad, -fd_pad) for _ in range(ndim)),
    )
    return tuple(
        [wavefield[s] for wavefield in wavefields] + [receiver_amplitudes_masked]
    )


def scalar_func(
    python_backend: Union[Literal["eager", "jit", "compile"], bool], *args: Any
) -> Tuple[torch.Tensor, ...]:
    """Helper function to apply the ScalarForwardFunc.

    This function serves as a convenient wrapper to call the `apply` method
    of `ScalarForwardFunc`, which is the entry point for the autograd graph
    for scalar wave propagation.

    Args:
        python_backend: Bool or string specifying whether to use Python backend.
        *args: Variable length argument list to be passed directly to
            `ScalarForwardFunc.apply`.

    Returns:
        The results of the forward pass from `ScalarForwardFunc.apply`.

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
        return scalar_python(*args)

    func = ScalarForwardFunc.apply

    return cast(
        "Tuple[torch.Tensor, ...]",
        func(*args),  # type: ignore[no-untyped-call]
    )
