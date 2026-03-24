"""Scalar wave propagation module for Deepwave.

Implements scalar wave equation propagation using finite differences in time
(2nd order) and space (user-selectable order: 2, 4, 6, or 8). Supports PML
boundaries and adjoint modelling.
"""

from typing import Any, List, Literal, Optional, Tuple, Union

import torch

import deepwave.backend_utils
import deepwave.common
import deepwave.regular_grid
from deepwave.generic_forward_func import GenericForwardFunc
from deepwave.scalar_equation import ScalarEquation


def scalar(
    v: torch.Tensor,
    grid_spacing: Union[float, List[float]],
    dt: float,
    source_amplitudes: Optional[torch.Tensor] = None,
    source_locations: Optional[torch.Tensor] = None,
    receiver_locations: Optional[torch.Tensor] = None,
    accuracy: int = 4,
    pml_width: Union[int, List[int]] = 20,
    pml_freq: Optional[float] = None,
    max_vel: Optional[float] = None,
    survey_pad: Optional[Union[int, List[Optional[int]]]] = None,
    wavefield_0: Optional[torch.Tensor] = None,
    wavefield_m1: Optional[torch.Tensor] = None,
    psiz_m1: Optional[torch.Tensor] = None,
    zetaz_m1: Optional[torch.Tensor] = None,
    psiy_m1: Optional[torch.Tensor] = None,
    zetay_m1: Optional[torch.Tensor] = None,
    psix_m1: Optional[torch.Tensor] = None,
    zetax_m1: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    n_checkpoints: Optional[int] = None,
    checkpoint_mode: Union[int, str] = "forward",
    python_backend: Union[bool, str] = False,
    **kwargs: Any,
) -> Tuple[torch.Tensor, ...]:
    """Scalar wave propagation.

    Args:
        v:
            A 2D or 3D Tensor containing the wave speed model.
        grid_spacing:
            The spacing between grid points.
        dt:
            The time step interval.
        source_amplitudes:
            A Tensor containing the source amplitudes.
        source_locations:
            A Tensor containing the source locations.
        receiver_locations:
            A Tensor containing the receiver locations.
        accuracy:
            The accuracy order of the finite difference scheme.
        pml_width:
            The width of the PML boundary layer.
        pml_freq:
            The frequency for the PML profile.
        max_vel:
            The maximum velocity in the model.
        survey_pad:
            Padding for the survey area.
        wavefield_0:
            The wavefield at the current time step.
        wavefield_m1:
            The wavefield at the previous time step.
        psiz_m1, zetaz_m1, psiy_m1, zetay_m1, psix_m1, zetax_m1:
            PML memory variables.
        origin:
            The origin coordinate of the model.
        nt:
            The number of time steps.
        model_gradient_sampling_interval:
            Interval for sampling the model gradient.
        n_checkpoints:
            Number of checkpoints for memory optimization.
        checkpoint_mode:
            Strategy for checkpointing.
        python_backend:
            Whether to use the Python backend.
        **kwargs:
            Additional arguments.

    Returns:
        A tuple containing the output wavefields and receiver amplitudes.

    """
    if kwargs.get("propagator_name") is not None:
        raise RuntimeError("propagator_name is deprecated.")

    # Determine ndim from grid_spacing or other inputs
    if not isinstance(grid_spacing, (int, float)):
        ndim = len(grid_spacing)
    else:
        try:
            ndim = deepwave.common.get_ndim(
                [v],
                [wavefield_0, wavefield_m1],
                [source_locations, receiver_locations],
                [psiz_m1, zetaz_m1],
                [psiy_m1, zetay_m1],
                [psix_m1, zetax_m1],
            )
        except RuntimeError:
            # If we couldn't determine ndim from other inputs, use v.ndim
            # and assume it doesn't have a batch dimension (it will be added
            # later if needed).
            ndim = v.ndim

    # We need to construct wavefields list
    equation = ScalarEquation()
    wavefields_dict = {
        "wavefield_0": wavefield_0,
        "wavefield_m1": wavefield_m1,
        "psiz_m1": psiz_m1,
        "zetaz_m1": zetaz_m1,
        "psiy_m1": psiy_m1,
        "zetay_m1": zetay_m1,
        "psix_m1": psix_m1,
        "zetax_m1": zetax_m1,
    }
    wavefields_list = equation.build_initial_wavefields(wavefields_dict, ndim)

    # Calculate max/min vel
    if max_vel is None:
        max_model_vel, min_nonzero_model_vel = equation.get_max_vel([v])
    else:
        max_model_vel = max_vel
        min_nonzero_model_vel = 0.0

    fd_pad_list = equation.get_fd_pad(accuracy, ndim)

    (
        models_out,
        source_amplitudes_out,
        wavefields_out,
        source_locations_out,
        receiver_locations_out,
        grid_spacing_out,
        dt_out,
        nt_out,
        n_shots,
        step_ratio,
        _,
        accuracy_out,
        pml_width_out,
        pml_freq_out,
        max_vel_out,
        freq_taper_frac_out,
        time_pad_frac_out,
        time_taper_out,
        device,
        dtype,
    ) = deepwave.common.setup_propagator(
        [v],
        equation.model_pad_modes,
        grid_spacing,
        dt,
        [source_amplitudes],
        [source_locations],
        [receiver_locations],
        accuracy,
        fd_pad_list,
        pml_width,
        pml_freq,
        max_vel,
        min_nonzero_model_vel,
        max_model_vel,
        survey_pad,
        wavefields_list,
        origin,
        nt,
        model_gradient_sampling_interval,
        kwargs.get("freq_taper_frac", 0.0),
        kwargs.get("time_pad_frac", 0.0),
        kwargs.get("time_taper", False),
        ndim,
    )

    v = models_out[0]
    model_shape = v.shape[1:]

    # PML Profiles
    fd_pad = fd_pad_list

    # We need to scale source amplitudes for the C backend?
    # ScalarEquation.scale_source_amplitudes does exactly this.
    #
    # Wait, `scalar_func` calls `GenericForwardFunc`.
    # `GenericForwardFunc` does NOT call `scale_source_amplitudes`.
    # `ScalarEquation.prepare_models` is identity.
    #
    # So we MUST perform scaling HERE if `scalar_func` doesn't do it.

    # We need sources_i for scaling.
    source_amplitudes_list = source_amplitudes_out
    sources_i_list = source_locations_out

    # Check if we have sources
    if len(source_amplitudes_list) > 0 and len(sources_i_list) > 0:
        source_amplitudes = source_amplitudes_list[0]
        sources_i = sources_i_list[0]

        flat_model_shape = int(torch.prod(torch.tensor(model_shape)).item())
        mask = sources_i == deepwave.common.IGNORE_LOCATION
        sources_i_masked = sources_i.clone()
        sources_i_masked[mask] = 0

        # We need to scale source amplitudes.
        # Note: we must handle IGNORE_LOCATION to avoid gathering v[0] dependency if not needed.
        v_at_sources = (
            v.view(-1, flat_model_shape)
            .expand(n_shots, -1)
            .gather(1, sources_i_masked.long())
        )
        v_at_sources = v_at_sources.clone()
        v_at_sources[mask] = 0

        # Apply scaling
        source_amplitudes = (
            -source_amplitudes
            * v_at_sources ** 2
            * dt_out**2
        )
        source_amplitudes_list = [source_amplitudes]
    else:
        source_amplitudes_list = []
        sources_i_list = []

    pml_profiles = deepwave.regular_grid.set_pml_profiles(
        pml_width_out,
        accuracy_out,
        fd_pad,
        dt_out,
        grid_spacing_out,
        max_vel_out,
        dtype,
        device,
        pml_freq_out,
        model_shape,
    )

    callback_frequency = kwargs.get("callback_frequency", nt_out // step_ratio)
    if not isinstance(callback_frequency, int):
        raise TypeError("callback_frequency must be an int.")
    if callback_frequency <= 0:
        raise ValueError("callback_frequency must be positive.")

    # Call scalar_func
    # scalar_func will dispatch to python or generic.
    # Generic uses ScalarEquation.
    # ScalarEquation.pack_args packs the sources.
    # ScalarEquation.scale_source_amplitudes is present but UNUSED by GenericForwardFunc?
    # If I scale here, I should probably make ScalarEquation.scale_source_amplitudes IDENTITY?
    # OR verify GenericForwardFunc doesn't call it.
    #
    # I verified GenericForwardFunc doesn't call it.
    # So it's safe to scale here.

    outputs = list(
        scalar_func(
            python_backend,
            v,
            source_amplitudes_list[0] if len(source_amplitudes_list) > 0 else torch.empty(0, device=device, dtype=dtype),
            pml_profiles,
            sources_i_list[0] if len(sources_i_list) > 0 else torch.empty(0, dtype=torch.long, device=device),
            receiver_locations_out[0] if len(receiver_locations_out) > 0 else torch.empty(0, dtype=torch.long, device=device),
            grid_spacing_out,
            dt_out,
            nt_out,
            step_ratio,
            accuracy_out,
            pml_width_out,
            n_shots,
            kwargs.get("forward_callback"),
            kwargs.get("backward_callback"),
            callback_frequency,
            kwargs.get("storage_mode", "device"),
            kwargs.get("storage_path", ""),
            kwargs.get("storage_compression", False),
            *wavefields_out,
        )
    )

    outputs[-1] = deepwave.common.downsample_and_movedim(
        outputs[-1],
        step_ratio,
        freq_taper_frac_out,
        time_pad_frac_out,
        time_taper_out,
    )

    return tuple(outputs)


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
                    1,
                    sources_i_masked.expand(source_amplitudes_masked.shape[1], -1),
                    source_amplitudes_masked[t],
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


class Scalar(torch.nn.Module):
    """Scalar wave propagation module.

    See :func:`scalar` for arguments.
    """

    def __init__(
        self,
        v: torch.Tensor,
        grid_spacing: Union[float, List[float]],
        v_requires_grad: bool = False,
        storage_mode: Literal["device", "cpu", "disk", "none"] = "device",
        storage_path: str = ".",
        storage_compression: bool = False,
    ) -> None:
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
        pml_width: Union[int, List[int]] = 20,
        pml_freq: Optional[float] = None,
        max_vel: Optional[float] = None,
        survey_pad: Optional[Union[int, List[Optional[int]]]] = None,
        wavefield_0: Optional[torch.Tensor] = None,
        wavefield_m1: Optional[torch.Tensor] = None,
        psiz_m1: Optional[torch.Tensor] = None,
        zetaz_m1: Optional[torch.Tensor] = None,
        psiy_m1: Optional[torch.Tensor] = None,
        zetay_m1: Optional[torch.Tensor] = None,
        psix_m1: Optional[torch.Tensor] = None,
        zetax_m1: Optional[torch.Tensor] = None,
        origin: Optional[List[int]] = None,
        nt: Optional[int] = None,
        model_gradient_sampling_interval: int = 1,
        n_checkpoints: Optional[int] = None,
        checkpoint_mode: Union[int, str] = "forward",
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, ...]:
        """Perform forward propagation."""
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
            zetaz_m1=zetaz_m1,
            psiy_m1=psiy_m1,
            zetay_m1=zetay_m1,
            psix_m1=psix_m1,
            zetax_m1=zetax_m1,
            origin=origin,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            n_checkpoints=n_checkpoints,
            checkpoint_mode=checkpoint_mode,
            storage_mode=self.storage_mode,
            storage_path=self.storage_path,
            storage_compression=self.storage_compression,
            **kwargs,
        )
