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
from deepwave.generic_forward_func import GenericForwardFunc
from deepwave.scalar_equation import ScalarEquation


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
    """Helper function to apply the GenericForwardFunc with ScalarEquation.

    Args:
        python_backend: Bool or string specifying whether to use Python backend.
        *args: Variable length argument list.

    Returns:
        The results of the forward pass.

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

    v = args[0]
    source_amplitudes = args[1]
    pml_profiles = args[2]
    sources_i = args[3]
    receivers_i = args[4]
    grid_spacing = args[5]
    dt = args[6]
    nt = args[7]
    step_ratio = args[8]
    accuracy = args[9]
    pml_width = args[10]
    n_shots = args[11]
    forward_callback = args[12]
    backward_callback = args[13]
    callback_frequency = args[14]
    storage_mode = args[15]
    storage_path = args[16]
    storage_compression = args[17]
    wavefields = args[18:]

    equation = ScalarEquation()
    packed_args = equation.pack_args(
        [v],
        [source_amplitudes],
        pml_profiles,
        [sources_i],
        [receivers_i],
        list(wavefields),
    )

    return GenericForwardFunc.apply(
        equation,
        grid_spacing,
        dt,
        nt,
        step_ratio,
        accuracy,
        pml_width,
        n_shots,
        forward_callback,
        backward_callback,
        callback_frequency,
        storage_mode,
        storage_path,
        storage_compression,
        *packed_args,
    )
