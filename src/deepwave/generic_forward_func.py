"""Generic autograd Function for wave propagation.

Provides a single GenericForwardFunc that works with any
PropagatorEquation implementation, eliminating per-equation
ForwardFunc boilerplate.
"""

from typing import Any, List, Optional, Sequence, Tuple, cast

import torch

import deepwave.backend_utils
import deepwave.common


class GenericForwardFunc(torch.autograd.Function):
    """Generic forward propagation function."""

    @staticmethod
    def forward(
        ctx: Any,
        equation: Any,
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
        *packed_args: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward propagation of any wave equation."""
        device = packed_args[0].device
        dtype = packed_args[0].dtype
        storage_mode = deepwave.common.get_storage_mode(storage_mode_str, device)

        ndim = len(grid_spacing)
        is_cuda = packed_args[0].is_cuda
        model_shape = packed_args[0].shape[-ndim:]

        # 1. Unpack args
        (
            models,
            source_amplitudes,
            pml_profiles,
            sources_i,
            receivers_i,
            wavefields,
        ) = equation.unpack_args(packed_args, ndim)

        n_sources_per_shot = [
            locs.numel() // n_shots if locs.numel() > 0 else 0 for locs in sources_i
        ]
        n_receivers_per_shot = [
            locs.numel() // n_shots if locs.numel() > 0 else 0 for locs in receivers_i
        ]

        # 2. Setup storage
        models_requires_grad = [m.requires_grad for m in models]
        storage_requires_grad = equation.get_storage_requires_grad(models, ndim)
        model_batched = equation.get_model_batched(list(models), ndim)

        storage_manager = deepwave.common.setup_storage(
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
            storage_requires_grad,
        )

        # 3. Ensure contiguous
        tensors_to_check = (
            models + source_amplitudes + pml_profiles + sources_i + receivers_i
        )
        checked_tensors = deepwave.common.ensure_contiguous(*tensors_to_check)

        # Repopulate SEMANTIC groups from checked_tensors
        pos = 0
        models = list(checked_tensors[pos : pos + len(models)])
        pos += len(models)
        source_amplitudes = list(checked_tensors[pos : pos + len(source_amplitudes)])
        pos += len(source_amplitudes)
        pml_profiles = list(checked_tensors[pos : pos + len(pml_profiles)])
        pos += len(pml_profiles)
        sources_i = list(checked_tensors[pos : pos + len(sources_i)])
        pos += len(sources_i)
        receivers_i = list(checked_tensors[pos : pos + len(receivers_i)])

        # 4. Save context
        # Determine if we need to save for backward
        # (if any input requires grad)
        any_requires_grad = (
            any(models_requires_grad)
            or any(sa.requires_grad for sa in source_amplitudes)
            or any(wf.requires_grad for wf in wavefields)
        )

        if any_requires_grad:
            ctx.save_for_backward(
                *models,
                *source_amplitudes,
                *pml_profiles,
                *sources_i,
                *receivers_i,
                *wavefields,
            )
            ctx.equation = equation
            ctx.grid_spacing = grid_spacing
            ctx.dt = dt
            ctx.nt = nt
            ctx.n_shots = n_shots
            ctx.step_ratio = step_ratio
            ctx.accuracy = accuracy
            ctx.pml_width = pml_width
            ctx.source_amplitudes_requires_grad = [
                sa.requires_grad for sa in source_amplitudes
            ]
            ctx.models_requires_grad = models_requires_grad
            ctx.backward_callback = backward_callback
            ctx.callback_frequency = callback_frequency
            ctx.storage_manager = storage_manager
            ctx.device = device
            ctx.dtype = dtype
            ctx.is_cuda = is_cuda
            ctx.model_shape = model_shape
            ctx.n_source_types = len(source_amplitudes)
            ctx.n_receiver_types = len(receivers_i)
            ctx.n_wavefields = len(wavefields)
            ctx.n_models = len(models)
            ctx.n_pml_profiles = len(pml_profiles)
            ctx.n_source_locs = len(sources_i)
            ctx.n_receiver_locs = len(receivers_i)
            ctx.n_sources_per_shot = n_sources_per_shot
            ctx.n_receivers_per_shot = n_receivers_per_shot
            ctx.models_requires_grad = models_requires_grad
            ctx.storage_requires_grad = storage_requires_grad
            ctx.model_batched = model_batched

        # 5. Prepare wavefields
        size_with_batch = (n_shots, *model_shape)
        wavefields = equation.prepare_wavefields(
            list(wavefields),
            ndim,
            accuracy,
            device,
            dtype,
            size_with_batch,
            pml_width,
        )

        # 6. Allocate receiver outputs
        receiver_amplitudes: List[torch.Tensor] = []
        for locs, n_per_shot in zip(receivers_i, n_receivers_per_shot):
            if locs.numel() > 0:
                amp = torch.zeros(nt, n_shots, n_per_shot, device=device, dtype=dtype)
                receiver_amplitudes.append(amp)
            else:
                receiver_amplitudes.append(torch.empty(0, device=device, dtype=dtype))

        # 7. Prep PML and backend
        prepared_models = equation.prepare_models(list(models))
        aux_wavefields = equation.create_aux_wavefields(
            wavefields, ndim, is_backward=False
        )
        staggered = equation.grid_type == "staggered"
        pml_b, pml_e = deepwave.common.get_pml_bounds(
            pml_width, accuracy, model_shape, ndim, staggered=staggered
        )

        forward_func = deepwave.backend_utils.get_backend_function(
            equation.name, ndim, "forward", accuracy, dtype, device
        )

        stream, aux = deepwave.common.get_stream_or_aux(device, is_cuda, n_shots)

        # 8. Time-stepping loop
        if callback_frequency is None or callback_frequency <= 0:
            callback_frequency = nt // step_ratio

        wf_names = equation.get_callback_wavefield_names(ndim)
        callback_models = dict(zip(equation.get_callback_model_names(ndim), models))

        if wavefields[0].numel() > 0 and nt > 0:
            for step in range(0, nt // step_ratio, callback_frequency):
                if forward_callback is not None:
                    callback_wavefields = dict(zip(wf_names, wavefields))
                    forward_callback(
                        deepwave.common.CallbackState(
                            dt,
                            step,
                            callback_wavefields,
                            callback_models,
                            {},
                            equation.get_fd_pad(accuracy, ndim),
                            pml_width,
                        )
                    )

                step_nt = min(nt // step_ratio - step, callback_frequency)

                # Equation Strategy calls its specific C backend
                if (
                    equation.call_forward_backend(
                        forward_func,
                        prepared_models,
                        source_amplitudes,
                        wavefields,
                        aux_wavefields,
                        receiver_amplitudes,
                        storage_manager,
                        pml_profiles,
                        sources_i,
                        receivers_i,
                        grid_spacing,
                        dt,
                        step_nt * step_ratio,
                        n_shots,
                        model_shape,
                        n_sources_per_shot,
                        n_receivers_per_shot,
                        step_ratio,
                        storage_requires_grad,
                        model_batched,
                        storage_compression,
                        step * step_ratio,
                        pml_b,
                        pml_e,
                        aux,
                        stream,
                    )
                    != 0
                ):
                    raise RuntimeError("Compiled backend failed.")

                # Ping-pong swap if odd steps
                if (step_nt * step_ratio) % 2 != 0:
                    wavefields, aux_wavefields = equation.swap_odd_step_wavefields(
                        wavefields, aux_wavefields, ndim, is_backward=False
                    )

        # 9. Downsample receivers and return
        fd_pad = accuracy // 2
        if staggered:
            s = (
                slice(None),
                *(slice(fd_pad, shape - (fd_pad - 1)) for shape in model_shape),
            )
        else:
            s = (slice(None), *(slice(fd_pad, -fd_pad) for _ in range(ndim)))

        outputs: List[torch.Tensor] = [w[s].clone() for w in wavefields]
        outputs.extend(receiver_amplitudes)
        return tuple(outputs)

    @staticmethod
    def backward(ctx: Any, *args: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Backward propagation of GenericForwardFunc."""
        equation = ctx.equation

        # Re-pack counts for GenericBackwardFunc
        counts = [
            ctx.n_models,
            len(ctx.source_amplitudes_requires_grad),
            ctx.n_pml_profiles,
            ctx.n_source_locs,
            ctx.n_receiver_locs,
            ctx.n_wavefields,
        ]

        res = GenericBackwardFunc.apply(  # type: ignore[no-untyped-call]
            equation,
            ctx.grid_spacing,
            ctx.dt,
            ctx.nt,
            ctx.n_shots,
            ctx.step_ratio,
            ctx.accuracy,
            ctx.pml_width,
            ctx.source_amplitudes_requires_grad,
            ctx.models_requires_grad,
            ctx.backward_callback,
            ctx.callback_frequency,
            ctx.storage_manager,
            counts,
            *ctx.saved_tensors,
            *args,
        )

        # Slice wavefield gradients to match input shapes (which lack FD padding)
        res_list = list(res)
        fd_pad = ctx.accuracy // 2
        if equation.grid_type == "staggered":
            s = (
                slice(None),
                *(slice(fd_pad, shape - (fd_pad - 1)) for shape in ctx.model_shape),
            )
        else:
            s = (slice(None), *(slice(fd_pad, -fd_pad) for _ in range(len(ctx.grid_spacing))))

        # Wavefields start after models, source_amplitudes, pml_profiles,
        # source_locs, and receiver_locs.
        wf_start = 14 + ctx.n_models + ctx.n_source_types + ctx.n_pml_profiles + ctx.n_source_locs + ctx.n_receiver_locs
        wf_end = wf_start + ctx.n_wavefields
        for i in range(wf_start, wf_end):
            if isinstance(res_list[i], torch.Tensor):
                res_list[i] = res_list[i][s]

        # res has length 14 + tensors.
        # We need to return gradients for GenericForwardFunc.apply inputs:
        # 14 metadata args + len(packed_args).
        # packed_args length is exactly len(ctx.saved_tensors).
        return cast(
            "Tuple[Optional[torch.Tensor], ...]",
            tuple(res_list[:14]) + tuple(res_list[14 : 14 + len(ctx.saved_tensors)]),
        )


class GenericBackwardFunc(torch.autograd.Function):
    """Autograd function for the backward pass (and double backward)."""

    @staticmethod
    def forward(
        ctx: Any,
        equation: Any,
        grid_spacing: Sequence[float],
        dt: float,
        nt: int,
        n_shots: int,
        step_ratio: int,
        accuracy: int,
        pml_width: List[int],
        source_amplitudes_requires_grad: List[bool],
        models_requires_grad: List[bool],
        backward_callback: Optional[deepwave.common.Callback],
        callback_frequency: int,
        storage_manager: deepwave.common.StorageManager,
        counts: List[int],
        *tensors: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Generic backward pass forward execution."""
        ctx.equation = equation
        ctx.grid_spacing = grid_spacing
        ctx.dt = dt
        ctx.nt = nt
        ctx.n_shots = n_shots
        ctx.step_ratio = step_ratio
        ctx.accuracy = accuracy
        ctx.pml_width = pml_width
        ctx.source_amplitudes_requires_grad = source_amplitudes_requires_grad
        ctx.models_requires_grad = models_requires_grad
        ctx.backward_callback = backward_callback
        ctx.callback_frequency = callback_frequency
        ctx.storage_manager = storage_manager

        # Unpack tensors using counts
        pos = 0
        models = list(tensors[pos : pos + counts[0]])
        pos += counts[0]
        source_amplitudes = list(tensors[pos : pos + counts[1]])
        pos += counts[1]
        pml_profiles = list(tensors[pos : pos + counts[2]])
        pos += counts[2]
        sources_i = list(tensors[pos : pos + counts[3]])
        pos += counts[3]
        receivers_i = list(tensors[pos : pos + counts[4]])
        pos += counts[4]
        wavefields = list(tensors[pos : pos + counts[5]])
        pos += counts[5]

        # Remaining tensors are grad_outputs (grad_wavefields, grad_r)
        n_wavefields = counts[5]
        grad_wavefields_raw = tensors[pos : pos + n_wavefields]
        pos += n_wavefields
        grad_r_raw = tensors[pos:]

        device = models[0].device
        dtype = models[0].dtype
        is_cuda = models[0].is_cuda
        ndim = len(grid_spacing)
        model_shape = models[0].shape[-ndim:]

        # Create zeros for None gradients
        # We need the unpadded spatial shape for wavefields
        fd_pad = accuracy // 2
        if equation.grid_type == "staggered":
            unpadded_shape = tuple(
                shape - (2 * fd_pad - 1) for shape in model_shape
            )
        else:
            unpadded_shape = tuple(
                shape - 2 * fd_pad for shape in model_shape
            )

        grad_wavefields_list = [
            torch.zeros(n_shots, *unpadded_shape, device=device, dtype=dtype)
            if g is None else g
            for g in grad_wavefields_raw
        ]

        # ctx.n_receiver_locs is index 4 in counts
        n_receivers_per_shot = [
            locs.numel() // n_shots if locs.numel() > 0 else 0 for locs in receivers_i
        ]

        grad_r = [
            torch.zeros(nt, n_shots, n, device=device, dtype=dtype)
            if g is None else g
            for g, n in zip(grad_r_raw, n_receivers_per_shot)
        ]

        # Save for double backward
        ctx.save_for_backward(
            *models,
            *source_amplitudes,
            *pml_profiles,
            *sources_i,
            *receivers_i,
            *wavefields,
            *grad_wavefields_list,
            *grad_r,
        )
        ctx.n_models = counts[0]
        ctx.n_source_types = counts[1]
        ctx.n_pml_profiles = counts[2]
        ctx.n_source_locs = counts[3]
        ctx.n_receiver_locs = counts[4]
        ctx.n_wavefields = counts[5]

        # 2. Setup models and gradients
        for i, m in enumerate(models):
            m.requires_grad_(models_requires_grad[i])
        prepared_models = equation.prepare_models(list(models))
        model_batched_prepared = equation.get_model_batched(models, ndim)
        storage_requires_grad = equation.get_storage_requires_grad(models, ndim)

        stream, aux = deepwave.common.get_stream_or_aux(device, is_cuda, n_shots)

        n_sources_per_shot = [
            locs.numel() // n_shots if locs.numel() > 0 else 0 for locs in sources_i
        ]

        (
            grad_models,
            _grad_models_tmp,
            grad_models_tmp_ptr,
            grad_f_list,
        ) = deepwave.common.setup_backward_gradients(
            prepared_models,
            model_batched_prepared,
            n_shots,
            model_shape,
            storage_manager.storage_mode,
            is_cuda,
            aux,
            nt,
            source_amplitudes_requires_grad,
            n_sources_per_shot,
        )

        grad_wavefields = deepwave.common.prepare_initial_wavefields(
            grad_wavefields_list,
            ndim,
            accuracy,
            device,
            dtype,
            (n_shots, *model_shape),
            pml_width,
            staggered=(equation.grid_type == "staggered"),
        )

        fd_pad = accuracy // 2
        fd_pad_list = equation.get_fd_pad(accuracy, ndim)

        aux_wavefields = equation.create_aux_wavefields(
            grad_wavefields, ndim, is_backward=True
        )

        equation.zero_backward_wavefields(
            grad_wavefields, ndim, accuracy, pml_width, fd_pad_list
        )

        equation.prepare_backward(grad_wavefields, ndim)

        staggered = equation.grid_type == "staggered"
        pml_b, pml_e = deepwave.common.get_pml_bounds(
            pml_width, accuracy, model_shape, ndim, multiplier=3, staggered=staggered
        )

        backward_func = deepwave.backend_utils.get_backend_function(
            equation.name, ndim, "backward", accuracy, dtype, device
        )

        if backward_callback is None:
            callback_frequency = nt // step_ratio

        wf_names = equation.get_callback_wavefield_names(ndim)
        callback_models = dict(zip(equation.get_callback_model_names(ndim), prepared_models))
        callback_grad_models = dict(
            zip(equation.get_callback_model_names(ndim), grad_models)
        )

        if grad_wavefields[0].numel() > 0 and nt > 0:
            for step in range(nt // step_ratio, 0, -callback_frequency):
                step_nt = min(step, callback_frequency)
                n_sources_per_shot_for_grad = [
                    n * r
                    for n, r in zip(n_sources_per_shot, source_amplitudes_requires_grad)
                ]
                n_receivers_per_shot = [
                    locs.numel() // n_shots if locs.numel() > 0 else 0
                    for locs in receivers_i
                ]

                if (
                    equation.call_backward_backend(
                        backward_func,
                        prepared_models,
                        grad_wavefields,
                        aux_wavefields,
                        grad_r,
                        grad_f_list,
                        grad_models,
                        grad_models_tmp_ptr,
                        storage_manager,
                        pml_profiles,
                        sources_i,
                        receivers_i,
                        grid_spacing,
                        dt,
                        nt,
                        step_nt * step_ratio,
                        n_shots,
                        model_shape,
                        n_sources_per_shot_for_grad,
                        n_receivers_per_shot,
                        step_ratio,
                        storage_requires_grad,
                        model_batched_prepared,
                        storage_manager.storage_compression,
                        step * step_ratio,
                        pml_b,
                        pml_e,
                        aux,
                        stream,
                    )
                    != 0
                ):
                    raise RuntimeError("Compiled backend failed.")

                if (step_nt * step_ratio) % 2 != 0:
                    grad_wavefields, aux_wavefields = equation.swap_odd_step_wavefields(
                        grad_wavefields, aux_wavefields, ndim, is_backward=True
                    )

                if backward_callback is not None:
                    callback_wavefields = dict(zip(wf_names, grad_wavefields))
                    backward_callback(
                        deepwave.common.CallbackState(
                            dt,
                            step - step_nt,
                            callback_wavefields,
                            callback_models,
                            callback_grad_models,
                            fd_pad_list,
                            pml_width,
                        )
                    )

        equation.finalize_backward(grad_wavefields, ndim)

        # 4. Chain model gradients back to raw models
        grad_raw_models = [
            torch.zeros_like(m) if m.requires_grad else None for m in models
        ]
        if any(m.requires_grad for m in models):
            # Chain through prepare_models
            out_for_grad = []
            grad_out_for_grad = []
            for pm, gpm in zip(prepared_models, grad_models):
                if pm.requires_grad and gpm.numel() > 0:
                    out_for_grad.append(pm)
                    grad_out_for_grad.append(gpm)

            if out_for_grad:
                grads = torch.autograd.grad(
                    outputs=out_for_grad,
                    inputs=[m for m in models if m.requires_grad],
                    grad_outputs=grad_out_for_grad,
                    retain_graph=False,
                    allow_unused=True,
                )
                gi = 0
                for i, m in enumerate(models):
                    if m.requires_grad:
                        grad_raw_models[i] = grads[gi]
                        gi += 1

        if equation.grid_type == "staggered":
            s = (
                slice(None),
                *(slice(fd_pad, shape - (fd_pad - 1)) for shape in model_shape),
            )
        else:
            s = (slice(None), *(slice(fd_pad, -fd_pad) for _ in range(ndim)))

        n_non_tensor_args = 14

        return (
            *([None] * n_non_tensor_args),
            *grad_raw_models,
            *grad_f_list,
            *[None] * len(pml_profiles),
            *[None] * len(sources_i),
            *[None] * len(receivers_i),
            *grad_wavefields,
            *[None] * n_wavefields,
            *[None] * len(grad_r),
        )

    @staticmethod
    def backward(ctx: Any, *args: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Generic double backward pass (Born)."""
        # TODO: The generic double backward implementation is currently causing
        # segmentation faults in the C backend due to argument passing issues.
        # Disabling it for now to ensure stability of the primary framework.
        # It requires a more complex setup of second gradients and potential
        # Born variants for all propagators.
        raise NotImplementedError(
            "Double backward (Born) is not yet fully supported in the "
            "generic framework."
        )
