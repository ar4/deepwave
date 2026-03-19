"""Generic autograd Function for wave propagation.

Provides a single GenericForwardFunc that works with any
PropagatorEquation implementation, eliminating per-equation
ForwardFunc boilerplate.
"""

from typing import Any, List, Optional, Sequence, Tuple

import torch

import deepwave.backend_utils
import deepwave.common


class GenericForwardFunc(torch.autograd.Function):
    """Autograd function that delegates equation-specific logic.

    Works with any equation class that provides pack_args / unpack_args
    and call_forward_backend / call_backward_backend methods.
    """

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
        *args: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Generic forward propagation.

        Args are equation-specific tensors packed by equation.pack_args().
        """
        ndim = len(grid_spacing)
        (
            models,
            source_amplitudes,
            pml_profiles,
            sources_i,
            receivers_i,
            wavefields,
        ) = equation.unpack_args(args, ndim)
        del args

        device = models[0].device
        dtype = models[0].dtype
        is_cuda = models[0].is_cuda
        model_shape = models[0].shape[-ndim:]
        n_sources_per_shot = [
            locs.numel() // n_shots if locs.numel() > 0 else 0 for locs in sources_i
        ]
        n_receivers_per_shot = [
            locs.numel() // n_shots if locs.numel() > 0 else 0 for locs in receivers_i
        ]
        storage_mode = deepwave.common.get_storage_mode(storage_mode_str, device)

        models_requires_grad = [m.requires_grad for m in models]
        storage_requires_grad = equation.get_storage_requires_grad(models, ndim)

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

        GenericForwardFunc._save_ctx(
            ctx,
            equation,
            models,
            sources_i,
            receivers_i,
            source_amplitudes,
            pml_profiles,
            wavefields,
            grid_spacing,
            dt,
            nt,
            n_shots,
            step_ratio,
            accuracy,
            pml_width,
            backward_callback,
            callback_frequency,
            storage_manager,
            source_amplitudes_requires_grad=[
                sa.requires_grad for sa in source_amplitudes
            ],
            models_requires_grad=models_requires_grad,
        )

        size_with_batch = (n_shots, *model_shape)
        wavefields = equation.prepare_wavefields(
            wavefields, ndim, accuracy, device, dtype, size_with_batch, pml_width
        )

        (
            models,
            source_amplitudes,
            pml_profiles,
            sources_i,
            receivers_i,
            wavefields,
        ) = deepwave.common.ensure_contiguous(
            models,
            source_amplitudes,
            pml_profiles,
            sources_i,
            receivers_i,
            list(wavefields),
        )

        pml_b, pml_e = deepwave.common.get_pml_bounds(
            pml_width,
            accuracy,
            model_shape,
            ndim,
            staggered=equation.grid_type == "staggered",
        )

        model_batched = [m.ndim == ndim + 1 and m.shape[0] > 1 for m in models]

        receiver_amplitudes = [
            torch.empty(0, device=device, dtype=dtype) for _ in receivers_i
        ]
        for i, (locs, n_per_shot) in enumerate(zip(receivers_i, n_receivers_per_shot)):
            if locs.numel() > 0:
                receiver_amplitudes[i].resize_(nt, n_shots, n_per_shot)
                receiver_amplitudes[i].fill_(0)

        aux_wavefields = equation.create_aux_wavefields(
            wavefields, ndim, is_backward=False
        )

        stream, aux = deepwave.common.get_stream_or_aux(device, is_cuda, n_shots)

        forward_func = deepwave.backend_utils.get_backend_function(
            equation.name,
            ndim,
            "forward",
            accuracy,
            dtype,
            device,
        )

        if forward_callback is None:
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

                if (
                    equation.call_forward_backend(
                        forward_func,
                        models,
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
                        models_requires_grad,
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

                if (step_nt * step_ratio) % 2 != 0:
                    wavefields, aux_wavefields = equation.swap_odd_step_wavefields(
                        wavefields, aux_wavefields, ndim, is_backward=False
                    )

        # Slice off fd_pad from wavefields
        fd_pad = accuracy // 2
        if equation.grid_type == "staggered":
            s = (
                slice(None),
                *(slice(fd_pad, shape - (fd_pad - 1)) for shape in model_shape),
            )
        else:
            s = (
                slice(None),
                *(slice(fd_pad, -fd_pad) for _ in range(ndim)),
            )

        # Downsample receiver amplitudes
        outputs: List[torch.Tensor] = [w[s] for w in wavefields]
        outputs.extend(receiver_amplitudes)
        return tuple(outputs)

    @staticmethod
    def _save_ctx(
        ctx: Any,
        equation: Any,
        models: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        receivers_i: List[torch.Tensor],
        source_amplitudes: List[torch.Tensor],
        pml_profiles: List[torch.Tensor],
        wavefields: List[torch.Tensor],
        grid_spacing: Sequence[float],
        dt: float,
        nt: int,
        n_shots: int,
        step_ratio: int,
        accuracy: int,
        pml_width: List[int],
        backward_callback: Optional[deepwave.common.Callback],
        callback_frequency: int,
        storage_manager: deepwave.common.StorageManager,
        source_amplitudes_requires_grad: List[bool],
        models_requires_grad: List[bool],
    ) -> None:
        """Save tensors and metadata to ctx for backward pass."""
        if (
            any(m.requires_grad for m in models)
            or any(sa.requires_grad for sa in source_amplitudes)
            or any(w.requires_grad for w in wavefields)
        ):
            ctx.save_for_backward(
                *models,
                *source_amplitudes,
                *pml_profiles,
                *sources_i,
                *receivers_i,
                *wavefields,
            )
            ctx.equation = equation
            ctx.n_models = len(models)
            ctx.n_source_types = len(source_amplitudes)
            ctx.n_pml_profiles = len(pml_profiles)
            ctx.n_source_locs = len(sources_i)
            ctx.n_receiver_locs = len(receivers_i)
            ctx.n_wavefields = len(wavefields)
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

    @staticmethod
    def backward(ctx: Any, *args: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Generic backward pass."""
        equation = ctx.equation
        counts = [
            ctx.n_models,
            ctx.n_source_types,
            ctx.n_pml_profiles,
            ctx.n_source_locs,
            ctx.n_receiver_locs,
            ctx.n_wavefields,
        ]
        
        return GenericBackwardFunc.apply(
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
        ctx.counts = counts

        pos = 0
        n_models, n_source_types, n_pml_profiles, n_source_locs, n_receiver_locs, n_wavefields = counts
        models = list(tensors[pos : pos + n_models])
        pos += n_models
        source_amplitudes = list(tensors[pos : pos + n_source_types])
        pos += n_source_types
        pml_profiles = list(tensors[pos : pos + n_pml_profiles])
        pos += n_pml_profiles
        sources_i = list(tensors[pos : pos + n_source_locs])
        pos += n_source_locs
        receivers_i = list(tensors[pos : pos + n_receiver_locs])
        pos += n_receiver_locs
        wavefields = list(tensors[pos : pos + n_wavefields])
        pos += n_wavefields
        # grad_wavefields has same length as wavefields
        grad_wavefields = list(tensors[pos : pos + n_wavefields])
        pos += n_wavefields
        grad_r = list(tensors[pos:])

        if any(t.requires_grad for t in tensors):
            ctx.save_for_backward(*tensors)

        (
            grad_r,
            models,
            source_amplitudes,
            pml_profiles,
            sources_i,
            receivers_i,
            wavefields,
            grad_wavefields
        ) = deepwave.common.ensure_contiguous(
            grad_r,
            models,
            source_amplitudes,
            pml_profiles,
            sources_i,
            receivers_i,
            wavefields,
            grad_wavefields
        )

        device = models[0].device
        dtype = models[0].dtype
        is_cuda = models[0].is_cuda
        ndim = len(grid_spacing)
        model_shape = models[0].shape[-ndim:]
        n_sources_per_shot = [
            locs.numel() // n_shots if locs.numel() > 0 else 0 for locs in sources_i
        ]

        model_batched = [m.ndim == ndim + 1 and m.shape[0] > 1 for m in models]
        stream, aux = deepwave.common.get_stream_or_aux(device, is_cuda, n_shots)

        (
            grad_models,
            _grad_models_tmp,
            grad_models_tmp_ptr,
            grad_f_list,
        ) = deepwave.common.setup_backward_gradients(
            models,
            model_batched,
            n_shots,
            model_shape,
            storage_manager.storage_mode,
            is_cuda,
            aux,
            nt,
            source_amplitudes_requires_grad,
            n_sources_per_shot,
        )

        size_with_batch = (n_shots, *model_shape)
        fd_pad = accuracy // 2
        fd_pad_list = equation.get_fd_pad(accuracy, ndim)

        grad_wavefields = [
            deepwave.common.create_or_pad(
                w,
                fd_pad_list if equation.grid_type == "staggered" else fd_pad,
                device,
                dtype,
                size_with_batch,
            )
            for w in grad_wavefields
        ]

        aux_wavefields = equation.create_aux_wavefields(
            grad_wavefields, ndim, is_backward=True
        )

        equation.zero_backward_wavefields(
            grad_wavefields, ndim, accuracy, pml_width, fd_pad_list
        )

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
        callback_models = dict(zip(equation.get_callback_model_names(ndim), models))
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
                    locs.numel() // n_shots if locs.numel() > 0 else 0 for locs in receivers_i
                ]

                if equation.call_backward_backend(
                    backward_func,
                    models,
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
                    step_nt * step_ratio,
                    n_shots,
                    model_shape,
                    n_sources_per_shot_for_grad,
                    n_receivers_per_shot,
                    step_ratio,
                    models_requires_grad,
                    model_batched,
                    storage_manager.storage_compression,
                    step * step_ratio,
                    pml_b,
                    pml_e,
                    aux,
                    stream,
                ) != 0:
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

        if equation.grid_type == "staggered":
             s = (slice(None), *(slice(fd_pad, shape - (fd_pad - 1)) for shape in model_shape))
        else:
             s = (slice(None), *(slice(fd_pad, -fd_pad) for _ in range(ndim)))

        n_non_tensor_args = 14
        
        return (
            *([None] * n_non_tensor_args),
            *grad_models,
            *grad_f_list,
            *[None] * len(pml_profiles),
            *[None] * len(sources_i),
            *[None] * len(receivers_i),
            *[w[s] for w in grad_wavefields],
            *[None] * n_wavefields,
            *[None] * len(grad_r)
        )
    
    @staticmethod
    def backward(ctx: Any, *args: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Generic double backward pass (Born)."""
        # TODO: The generic double backward implementation is currently causing
        # segmentation faults in the C backend due to argument passing issues.
        # Disabling it for now to ensure stability of the primary framework.
        raise NotImplementedError("Double backward (Born) is not yet fully supported in the generic framework.")
