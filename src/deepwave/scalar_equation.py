"""Scalar equation strategy for the propagator framework.

Implements the PropagatorEquation interface for the constant-density
scalar wave equation (regular grid).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import deepwave.backend_utils
import deepwave.common
import deepwave.regular_grid


class ScalarEquation:
    """Equation class for scalar wave propagation on a regular grid.

    Encapsulates all equation-specific logic needed by the generic
    propagator framework: model preparation, source scaling, PML
    profiles, wavefield layout, and callback naming.
    """

    @property
    def name(self) -> str:
        """C function prefix."""
        return "scalar"

    @property
    def grid_type(self) -> str:
        """Grid type."""
        return "regular"

    @property
    def n_models(self) -> int:
        """Number of input model tensors."""
        return 1

    @property
    def model_pad_modes(self) -> List[str]:
        """Padding mode per model tensor."""
        return ["replicate"]

    def get_ndim(
        self,
        models: List[torch.Tensor],
        initial_wavefields: List[Optional[torch.Tensor]],
        location_tensors: List[Optional[torch.Tensor]],
    ) -> int:
        """Determine spatial dimensions from inputs.

        For scalar, we have one model (v), two main wavefields (wfc, wfp),
        and PML variables grouped per dimension.
        """
        v = models[0]
        wavefield_0 = initial_wavefields[0]
        wavefield_m1 = initial_wavefields[1]
        source_locations = location_tensors[0]
        receiver_locations = location_tensors[1]
        # PML wavefields are packed as: [psi_z, psi_y, psi_x, zeta_z, ...]
        # after wfc, wfp in the flat list.
        psiz_m1 = initial_wavefields[2] if len(initial_wavefields) > 2 else None
        zetaz_m1 = initial_wavefields[3] if len(initial_wavefields) > 3 else None
        psiy_m1 = initial_wavefields[4] if len(initial_wavefields) > 4 else None
        zetay_m1 = initial_wavefields[5] if len(initial_wavefields) > 5 else None
        psix_m1 = initial_wavefields[6] if len(initial_wavefields) > 6 else None
        zetax_m1 = initial_wavefields[7] if len(initial_wavefields) > 7 else None
        return deepwave.common.get_ndim(
            [v],
            [wavefield_0, wavefield_m1],
            [source_locations, receiver_locations],
            [psiz_m1, zetaz_m1],
            [psiy_m1, zetay_m1],
            [psix_m1, zetax_m1],
        )

    def build_initial_wavefields(
        self,
        user_inputs: Dict[str, Optional[torch.Tensor]],
        ndim: int,
    ) -> List[Optional[torch.Tensor]]:
        """Convert named user wavefields to flat list.

        The returned list is: [wavefield_0, wavefield_m1, psi..., zeta...]
        where psi/zeta are ordered: z (if 3D), y (if >= 2D), x.
        """
        wavefield_0 = user_inputs.get("wavefield_0")
        wavefield_m1 = user_inputs.get("wavefield_m1")
        psi: List[Optional[torch.Tensor]] = []
        zeta: List[Optional[torch.Tensor]] = []
        if ndim == 3:
            psi.append(user_inputs.get("psiz_m1"))
            zeta.append(user_inputs.get("zetaz_m1"))
        if ndim >= 2:
            psi.append(user_inputs.get("psiy_m1"))
            zeta.append(user_inputs.get("zetay_m1"))
        if ndim >= 1:
            psi.append(user_inputs.get("psix_m1"))
            zeta.append(user_inputs.get("zetax_m1"))
        return [wavefield_0, wavefield_m1, *psi, *zeta]

    def get_fd_pad(self, accuracy: int, ndim: int) -> List[int]:
        """Finite difference padding for regular grid."""
        return [accuracy // 2] * 2 * ndim

    def prepare_models(self, raw_models: List[torch.Tensor]) -> List[torch.Tensor]:
        """Identity transform: [v] -> [v]."""
        return list(raw_models)

    def scale_source_amplitudes(
        self,
        source_amplitudes: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        prepared_models: List[torch.Tensor],
        dt: float,
        n_shots: int,
    ) -> List[torch.Tensor]:
        """Scale source amplitudes: -source * v^2 * dt^2 at source locations."""
        v = prepared_models[0]
        model_shape = v.shape[1:]
        flat_model_shape = int(torch.prod(torch.tensor(model_shape)).item())
        mask = sources_i[0] == deepwave.common.IGNORE_LOCATION
        sources_i_masked = sources_i[0].clone()
        sources_i_masked[mask] = 0
        scaled = (
            -source_amplitudes[0]
            * (
                v.view(-1, flat_model_shape)
                .expand(n_shots, -1)
                .gather(1, sources_i_masked)
            )
            ** 2
            * dt**2
        )
        return [scaled]

    def set_pml_profiles(
        self,
        pml_width: List[int],
        accuracy: int,
        fd_pad: List[int],
        dt: float,
        grid_spacing: List[float],
        max_vel: float,
        dtype: torch.dtype,
        device: torch.device,
        pml_freq: float,
        model_shape: torch.Size,
    ) -> List[torch.Tensor]:
        """Set PML profiles using regular_grid module."""
        return deepwave.regular_grid.set_pml_profiles(
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

    def get_max_vel(self, raw_models: List[torch.Tensor]) -> Tuple[float, float]:
        """Compute max and min nonzero velocity."""
        v = raw_models[0]
        if deepwave.common.is_inside_vmap(v):
            raise RuntimeError(
                "If using BatchedTensor inputs, you must specify max_vel"
            )
        v_nonzero = v[v != 0]
        if v_nonzero.numel() > 0:
            min_nonzero_model_vel = v_nonzero.abs().min().item()
        else:
            min_nonzero_model_vel = 0.0
        del v_nonzero
        max_model_vel = v.abs().max().item()
        return max_model_vel, min_nonzero_model_vel

    def n_wavefields(self, ndim: int) -> int:
        """Number of wavefield tensors: wfc + wfp + psi[ndim] + zeta[ndim]."""
        return 2 + 2 * ndim

    def n_source_types(self, ndim: int) -> int:
        """Number of source types."""
        return 1

    def n_receiver_types(self, ndim: int) -> int:
        """Number of receiver types."""
        return 1

    def get_storage_requires_grad(
        self, raw_models: List[torch.Tensor], ndim: int
    ) -> List[bool]:
        """Which model params require gradient storage."""
        return [raw_models[0].requires_grad]

    def prepare_wavefields(
        self,
        wavefields: List[torch.Tensor],
        ndim: int,
        accuracy: int,
        device: torch.device,
        dtype: torch.dtype,
        size_with_batch: Tuple[int, ...],
        pml_width: List[int],
    ) -> List[torch.Tensor]:
        """Prepare wavefields for the C kernel.

        Delegates to common.prepare_initial_wavefields and zeros PML interior.
        """
        wavefields = deepwave.common.prepare_initial_wavefields(
            wavefields,
            ndim,
            accuracy,
            device,
            dtype,
            size_with_batch,
            pml_width,
        )
        # Zero interiors of PML wavefields
        # Scalar layout: [wfc, wfp, psi..., zeta...]
        fd_pad = accuracy // 2
        for i in range(ndim):
            wavefields[2 + i] = deepwave.common.zero_interior(
                wavefields[2 + i], fd_pad, pml_width, i
            )
            wavefields[2 + ndim + i] = deepwave.common.zero_interior(
                wavefields[2 + ndim + i], fd_pad, pml_width, i
            )
        return wavefields

    def get_callback_wavefield_names(self, ndim: int) -> List[str]:
        """Names for the callback wavefield dict."""
        dim_names = ["z", "y", "x"]
        names = ["wavefield_0", "wavefield_m1"]
        for i in range(ndim):
            names.append(f"psi{dim_names[-ndim + i]}_m1")
            names.append(f"zeta{dim_names[-ndim + i]}_m1")
        return names

    def get_callback_model_names(self, ndim: int) -> List[str]:
        """Names for the callback model dict."""
        return ["v"]

    def pack_args(
        self,
        models: List[torch.Tensor],
        source_amplitudes: List[torch.Tensor],
        pml_profiles: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        receivers_i: List[torch.Tensor],
        wavefields: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        """Pack tensors in scalar ForwardFunc order.

        Scalar native: v, source_amplitudes, pml_profiles (as list),
        sources_i, receivers_i, wavefields...
        """
        return (
            *models,
            *source_amplitudes,
            *pml_profiles,
            *sources_i,
            *receivers_i,
            *wavefields,
        )

    def unpack_args(
        self, args: Tuple[torch.Tensor, ...], ndim: int
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """Unpack flat args into semantic groups.

        Layout: v, sa, pml_profiles[3*ndim], sources_i, receivers_i,
                wavefields[2+2*ndim].
        """
        pos = 0
        models = list(args[pos : pos + self.n_models])
        pos += self.n_models
        source_amplitudes = list(args[pos : pos + self.n_source_types(ndim)])
        pos += self.n_source_types(ndim)
        pml_profiles = list(args[pos : pos + 3 * ndim])
        pos += 3 * ndim
        sources_i = list(args[pos : pos + self.n_source_types(ndim)])
        pos += self.n_source_types(ndim)
        receivers_i = list(args[pos : pos + self.n_receiver_types(ndim)])
        pos += self.n_receiver_types(ndim)
        wavefields = list(args[pos : pos + self.n_wavefields(ndim)])
        return (
            models,
            source_amplitudes,
            pml_profiles,
            sources_i,
            receivers_i,
            wavefields,
        )

    def call_forward_backend(
        self,
        backend_func: Any,
        models: List[torch.Tensor],
        source_amplitudes: List[torch.Tensor],
        wavefields: List[torch.Tensor],
        aux_wavefields: List[torch.Tensor],
        receiver_amplitudes: List[torch.Tensor],
        storage_manager: Any,
        pml_profiles: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        receivers_i: List[torch.Tensor],
        grid_spacing: Sequence[float],
        dt: float,
        step_nt: int,
        n_shots: int,
        model_shape: torch.Size,
        n_sources_per_shot: List[int],
        n_receivers_per_shot: List[int],
        step_ratio: int,
        models_requires_grad: List[bool],
        model_batched: List[bool],
        storage_compression: bool,
        step: int,
        pml_b: List[int],
        pml_e: List[int],
        aux: int,
        stream: Any,
    ) -> int:
        """Call scalar C backend.

        Scalar wavefield structure: [wfc, wfp, psi..., zeta...].
        psin is taken from aux_wavefields.
        """
        v = models[0]
        wfc = wavefields[0]
        wfp = wavefields[1]
        ndim = len(grid_spacing)
        psi = wavefields[2 : 2 + ndim]
        zeta = wavefields[2 + ndim :]
        psin = aux_wavefields[:ndim]

        rdx = [1 / dx for dx in grid_spacing]
        rdx2 = [1 / dx**2 for dx in grid_spacing]

        v_requires_grad = models_requires_grad[0]
        storage_mode = storage_manager.storage_mode

        return backend_func(  # type: ignore[no-any-return]
            v.data_ptr(),
            source_amplitudes[0].data_ptr(),
            wfc.data_ptr(),
            wfp.data_ptr(),
            *[field.data_ptr() for field in psi],
            *[field.data_ptr() for field in psin],
            *[field.data_ptr() for field in zeta],
            *storage_manager.storage_ptrs,
            receiver_amplitudes[0].data_ptr(),
            *[profile.data_ptr() for profile in pml_profiles],
            sources_i[0].data_ptr(),
            receivers_i[0].data_ptr(),
            *rdx,
            *rdx2,
            dt**2,
            step_nt,
            n_shots,
            *model_shape,

            n_sources_per_shot[0],
            n_receivers_per_shot[0],
            step_ratio,
            storage_mode,
            storage_manager.shot_bytes_uncomp,
            storage_manager.shot_bytes_comp,
            v_requires_grad and storage_mode != deepwave.common.StorageMode.NONE,
            model_batched[0],
            storage_compression,
            step,
            *pml_b,
            *pml_e,
            aux,
            stream,
        )

    def call_born_backend(
        self,
        backend_func: Any,
        models: List[torch.Tensor],
        grad_models: List[torch.Tensor],
        source_amplitudes: List[torch.Tensor],
        grad_source_amplitudes: List[torch.Tensor],
        wavefields: List[torch.Tensor],
        grad_wavefields: List[torch.Tensor],
        aux_wavefields: List[torch.Tensor],
        grad_aux_wavefields: List[torch.Tensor],
        receiver_amplitudes: List[torch.Tensor],
        grad_receiver_amplitudes: List[torch.Tensor],
        storage_manager: Any,
        pml_profiles: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        receivers_i: List[torch.Tensor],
        grad_receivers_i: List[torch.Tensor],
        grid_spacing: Sequence[float],
        dt: float,
        step_nt: int,
        n_shots: int,
        model_shape: torch.Size,
        n_sources_per_shot: List[int],
        n_receivers_per_shot: List[int],
        n_grad_receivers_per_shot: List[int],
        step_ratio: int,
        models_requires_grad: List[bool],
        model_batched: List[bool],
        grad_model_batched: List[bool],
        storage_compression: bool,
        step: int,
        pml_b: List[int],
        pml_e: List[int],
        aux: int,
        stream: Any,
    ) -> int:
        """Call scalar Born backend."""
        v = models[0]
        ggv = grad_models[0]
        wfc = wavefields[0]
        wfp = wavefields[1]
        gwfc = grad_wavefields[0]
        gwfp = grad_wavefields[1]
        ndim = len(grid_spacing)
        psi = wavefields[2 : 2 + ndim]
        zeta = wavefields[2 + ndim :]
        gpsi = grad_wavefields[2 : 2 + ndim]
        gzeta = grad_wavefields[2 + ndim :]
        psin = aux_wavefields[:ndim]
        gpsin = grad_aux_wavefields[:ndim]

        rdx = [1 / dx for dx in grid_spacing]
        rdx2 = [1 / dx**2 for dx in grid_spacing]

        v_requires_grad = models_requires_grad[0]
        storage_mode = storage_manager.storage_mode

        return backend_func(  # type: ignore[no-any-return]
            v.data_ptr(),
            ggv.data_ptr(),
            source_amplitudes[0].data_ptr(),
            grad_source_amplitudes[0].data_ptr(),
            wfc.data_ptr(),
            wfp.data_ptr(),
            *[field.data_ptr() for field in psi],
            *[field.data_ptr() for field in psin],
            *[field.data_ptr() for field in zeta],
            gwfc.data_ptr(),
            gwfp.data_ptr(),
            *[field.data_ptr() for field in gpsi],
            *[field.data_ptr() for field in gpsin],
            *[field.data_ptr() for field in gzeta],
            *storage_manager.storage_ptrs,
            receiver_amplitudes[0].data_ptr(),
            grad_receiver_amplitudes[0].data_ptr(),
            *[profile.data_ptr() for profile in pml_profiles],
            sources_i[0].data_ptr(),
            receivers_i[0].data_ptr(),
            grad_receivers_i[0].data_ptr(),
            *rdx,
            *rdx2,
            dt**2,
            step_nt,
            n_shots,
            *model_shape,
            n_sources_per_shot[0],
            n_receivers_per_shot[0],
            n_grad_receivers_per_shot[0],
            step_ratio,
            storage_mode,
            storage_manager.shot_bytes_uncomp,
            storage_manager.shot_bytes_comp,
            v_requires_grad and storage_mode != deepwave.common.StorageMode.NONE,
            False,
            model_batched[0],
            grad_model_batched[0],
            storage_compression,
            step,
            *pml_b,
            *pml_e,
            aux,
            stream,
        )

    def create_aux_wavefields(
        self,
        wavefields: List[torch.Tensor],
        ndim: int,
        is_backward: bool = False,
    ) -> List[torch.Tensor]:
        """Create scalar aux vars.

        Forward: psin.
        Backward: grad_psin + grad_zetan.
        """
        psi = wavefields[2 : 2 + ndim]
        if is_backward:
            zeta = wavefields[2 + ndim :]
            return [torch.zeros_like(wf) for wf in psi] + [
                torch.zeros_like(wf) for wf in zeta
            ]
        return [torch.zeros_like(wf) for wf in psi]

    def swap_odd_step_wavefields(
        self,
        wavefields: List[torch.Tensor],
        aux_wavefields: List[torch.Tensor],
        ndim: int,
        is_backward: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Swap scalar wavefields.

        Forward: wfc/wfp and psi/psin.
        Backward: wfc/wfp, psi/psin, zeta/zetan.
        """
        wavefields[0], wavefields[1] = (
            wavefields[1],
            wavefields[0],
        )
        if is_backward:
            psin = aux_wavefields[:ndim]
            zetan = aux_wavefields[ndim:]
            for i in range(ndim):
                # Swap psi/psin
                wavefields[2 + i], psin[i] = psin[i], wavefields[2 + i]
                # Swap zeta/zetan
                wavefields[2 + ndim + i], zetan[i] = zetan[i], wavefields[2 + ndim + i]
        else:
            psin = aux_wavefields
            for i in range(ndim):
                # Swap psi/psin
                wavefields[2 + i], psin[i] = psin[i], wavefields[2 + i]

        return wavefields, aux_wavefields

    def zero_backward_wavefields(
        self,
        grad_wavefields: List[torch.Tensor],
        ndim: int,
        accuracy: int,
        pml_width: List[int],
        fd_pad_list: List[int],
    ) -> None:
        """Zero PML gradient wavefields per dimension."""
        fd_pad = accuracy // 2
        grad_psi = grad_wavefields[2 : 2 + ndim]
        grad_zeta = grad_wavefields[2 + ndim :]
        for i in range(ndim):
            deepwave.common.zero_interior(grad_psi[i], fd_pad, pml_width, i)
            deepwave.common.zero_interior(grad_zeta[i], fd_pad, pml_width, i)
