"""Scalar Born equation strategy for the propagator framework.

Implements the PropagatorEquation interface for the constant-density
scalar Born wave equation (regular grid).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import deepwave.backend_utils
import deepwave.common
import deepwave.regular_grid
from deepwave.propagator_equation import PropagatorEquation


class ScalarBornEquation(PropagatorEquation):
    """Equation class for scalar Born wave propagation on a regular grid."""

    @property
    def name(self) -> str:
        """C function prefix."""
        return "scalar_born"

    @property
    def grid_type(self) -> str:
        """Grid type."""
        return "regular"

    @property
    def n_models(self) -> int:
        """Number of input model tensors."""
        return 2  # v, scatter

    @property
    def model_pad_modes(self) -> List[str]:
        """Padding mode per model tensor."""
        return ["replicate", "constant"]

    def get_ndim(
        self,
        models: List[torch.Tensor],
        initial_wavefields: List[Optional[torch.Tensor]],
        location_tensors: List[Optional[torch.Tensor]],
    ) -> int:
        """Determine spatial dimensions from inputs."""
        # We rely on common.get_ndim which looks at everything.
        # We pass representative tensors.
        return deepwave.common.get_ndim(
            models,
            initial_wavefields[:2],  # wfc, wfp
            location_tensors,
            [],
            [],
            [],  # We can skip PML vars for ndim check usually
        )

    def build_initial_wavefields(
        self,
        user_inputs: Dict[str, Optional[torch.Tensor]],
        ndim: int,
    ) -> List[Optional[torch.Tensor]]:
        """Convert named user wavefields to flat list."""
        wavefield_0 = user_inputs.get("wavefield_0")
        wavefield_m1 = user_inputs.get("wavefield_m1")
        psi: List[Optional[torch.Tensor]] = []
        zeta: List[Optional[torch.Tensor]] = []

        wavefield_sc_0 = user_inputs.get("wavefield_sc_0")
        wavefield_sc_m1 = user_inputs.get("wavefield_sc_m1")
        psi_sc: List[Optional[torch.Tensor]] = []
        zeta_sc: List[Optional[torch.Tensor]] = []

        if ndim == 3:
            psi.append(user_inputs.get("psiz_m1"))
            zeta.append(user_inputs.get("zetaz_m1"))
            psi_sc.append(user_inputs.get("psiz_sc_m1"))
            zeta_sc.append(user_inputs.get("zetaz_sc_m1"))
        if ndim >= 2:
            psi.append(user_inputs.get("psiy_m1"))
            zeta.append(user_inputs.get("zetay_m1"))
            psi_sc.append(user_inputs.get("psiy_sc_m1"))
            zeta_sc.append(user_inputs.get("zetay_sc_m1"))
        if ndim >= 1:
            psi.append(user_inputs.get("psix_m1"))
            zeta.append(user_inputs.get("zetax_m1"))
            psi_sc.append(user_inputs.get("psix_sc_m1"))
            zeta_sc.append(user_inputs.get("zetax_sc_m1"))

        return [
            wavefield_0,
            wavefield_m1,
            *psi,
            *zeta,
            wavefield_sc_0,
            wavefield_sc_m1,
            *psi_sc,
            *zeta_sc,
        ]

    def get_fd_pad(self, accuracy: int, ndim: int) -> List[int]:
        """Finite difference padding for regular grid."""
        return [accuracy // 2] * 2 * ndim

    def prepare_models(self, raw_models: List[torch.Tensor]) -> List[torch.Tensor]:
        """Identity transform: [v, scatter] -> [v, scatter]."""
        return list(raw_models)

    def scale_source_amplitudes(
        self,
        source_amplitudes: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        prepared_models: List[torch.Tensor],
        dt: float,
        n_shots: int,
    ) -> List[torch.Tensor]:
        """Scale source amplitudes."""
        v = prepared_models[0]
        scatter = prepared_models[1]
        model_shape = v.shape[1:]
        flat_model_shape = int(torch.prod(torch.tensor(model_shape)).item())

        # Background: scale by -source * v^2 * dt^2
        mask = sources_i[0] == deepwave.common.IGNORE_LOCATION
        sources_i_masked = sources_i[0].clone()
        sources_i_masked[mask] = 0
        scaled_bg = (
            -source_amplitudes[0]
            * (
                v.view(-1, flat_model_shape)
                .expand(n_shots, -1)
                .gather(1, sources_i_masked)
            )
            ** 2
            * dt**2
        )

        # Scattered: scale by -2 * source * v * scatter * dt^2
        # The second source amplitude entry is the scattered one
        mask_sc = sources_i[1] == deepwave.common.IGNORE_LOCATION
        sources_i_masked_sc = sources_i[1].clone()
        sources_i_masked_sc[mask_sc] = 0
        scaled_sc = (
            -2
            * source_amplitudes[1]
            * (
                v.view(-1, flat_model_shape)
                .expand(n_shots, -1)
                .gather(1, sources_i_masked_sc)
            )
            * (
                scatter.view(-1, flat_model_shape)
                .expand(n_shots, -1)
                .gather(1, sources_i_masked_sc)
            )
            * dt**2
        )

        return [scaled_bg, scaled_sc]

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
        """Compute max and min nonzero velocity (from v, ignoring scatter)."""
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
        """Number of wavefield tensors: 2 * (2 + 2 * ndim)."""
        return 4 + 4 * ndim

    def n_source_types(self, ndim: int) -> int:
        """Number of source types (background + scattered)."""
        return 2

    def n_receiver_types(self, ndim: int) -> int:
        """Number of receiver types (background + scattered)."""
        return 2

    def get_storage_requires_grad(
        self, raw_models: List[torch.Tensor], ndim: int
    ) -> List[bool]:
        """Which model params require gradient storage."""
        # Flags for v_requires_grad or scatter_requires_grad, and v_requires_grad
        v_rg = raw_models[0].requires_grad
        scatter_rg = raw_models[1].requires_grad
        return [v_rg or scatter_rg, v_rg]

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
        """Prepare wavefields for the C kernel."""
        wavefields = deepwave.common.prepare_initial_wavefields(
            wavefields,
            ndim,
            accuracy,
            device,
            dtype,
            size_with_batch,
            pml_width,
        )

        # Scalar Born layout:
        # [wfc, wfp, psi..., zeta...] (Background)
        # [wfcsc, wfpsc, psisc..., zetasc...] (Scattered)

        n_bg = 2 + 2 * ndim

        # Zero interiors of PML wavefields
        fd_pad = accuracy // 2

        # Background PML
        for i in range(ndim):
            wavefields[2 + i] = deepwave.common.zero_interior(
                wavefields[2 + i], fd_pad, pml_width, i
            )
            wavefields[2 + ndim + i] = deepwave.common.zero_interior(
                wavefields[2 + ndim + i], fd_pad, pml_width, i
            )

        # Scattered PML
        for i in range(ndim):
            wavefields[n_bg + 2 + i] = deepwave.common.zero_interior(
                wavefields[n_bg + 2 + i], fd_pad, pml_width, i
            )
            wavefields[n_bg + 2 + ndim + i] = deepwave.common.zero_interior(
                wavefields[n_bg + 2 + ndim + i], fd_pad, pml_width, i
            )

        return wavefields

    def get_callback_wavefield_names(self, ndim: int) -> List[str]:
        """Names for the callback wavefield dict."""
        dim_names = ["z", "y", "x"]
        names = ["wavefield_0", "wavefield_m1"]
        for i in range(ndim):
            names.append(f"psi{dim_names[-ndim + i]}_m1")
            names.append(f"zeta{dim_names[-ndim + i]}_m1")

        names.append("wavefield_sc_0")
        names.append("wavefield_sc_m1")
        for i in range(ndim):
            names.append(f"psi{dim_names[-ndim + i]}_sc_m1")
            names.append(f"zeta{dim_names[-ndim + i]}_sc_m1")

        return names

    def get_callback_model_names(self, ndim: int) -> List[str]:
        """Names for the callback model dict."""
        return ["v", "scatter"]

    def pack_args(
        self,
        models: List[torch.Tensor],
        source_amplitudes: List[torch.Tensor],
        pml_profiles: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        receivers_i: List[torch.Tensor],
        wavefields: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        """Pack tensors."""
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
        """Unpack flat args into semantic groups."""
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
        """Call scalar born C backend."""
        v = models[0]
        scatter = models[1]

        ndim = len(grid_spacing)
        n_bg = 2 + 2 * ndim

        # Background wavefields
        wfc = wavefields[0]
        wfp = wavefields[1]
        psi = wavefields[2 : 2 + ndim]
        zeta = wavefields[2 + ndim : n_bg]

        # Scattered wavefields
        wfcsc = wavefields[n_bg]
        wfpsc = wavefields[n_bg + 1]
        psisc = wavefields[n_bg + 2 : n_bg + 2 + ndim]
        zetasc = wavefields[n_bg + 2 + ndim :]

        # Aux wavefields (psin, psinsc)
        # Note: GenericForwardFunc allocates aux_wavefields.
        # ScalarBornEquation.create_aux_wavefields should allocate both.
        psin = aux_wavefields[:ndim]
        psinsc = aux_wavefields[ndim:]

        rdx = [1 / dx for dx in grid_spacing]
        rdx2 = [1 / dx**2 for dx in grid_spacing]

        v_requires_grad = models_requires_grad[0]
        scatter_requires_grad = models_requires_grad[1]
        storage_mode = storage_manager.storage_mode

        # NOTE: ScalarBorn backend expects n_receiverssc_per_shot as the last count arg
        n_receivers = n_receivers_per_shot[0]
        n_receiverssc = n_receivers_per_shot[1]

        return backend_func(  # type: ignore[no-any-return]
            v.data_ptr(),
            scatter.data_ptr(),
            source_amplitudes[0].data_ptr(),
            source_amplitudes[1].data_ptr(),
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
            receiver_amplitudes[0].data_ptr(),
            receiver_amplitudes[1].data_ptr(),
            *[profile.data_ptr() for profile in pml_profiles],
            sources_i[0].data_ptr(),
            receivers_i[0].data_ptr(),
            receivers_i[1].data_ptr(),
            *rdx,
            *rdx2,
            dt**2,
            step_nt,
            n_shots,
            *model_shape,
            n_sources_per_shot[0],
            n_receivers,
            n_receiverssc,
            step_ratio,
            storage_mode,
            storage_manager.shot_bytes_uncomp,
            storage_manager.shot_bytes_comp,
            v_requires_grad and storage_mode != deepwave.common.StorageMode.NONE,
            scatter_requires_grad and storage_mode != deepwave.common.StorageMode.NONE,
            model_batched[0],
            model_batched[1],
            storage_compression,
            step,
            *pml_b,
            *pml_e,
            aux,
            stream,
        )

    def call_backward_backend(
        self,
        backend_func: Any,
        models: List[torch.Tensor],
        grad_wavefields: List[torch.Tensor],
        aux_wavefields: List[torch.Tensor],
        grad_r: List[torch.Tensor],
        grad_f_list: List[torch.Tensor],
        grad_models: List[torch.Tensor],
        grad_models_tmp_ptr: List[int],
        storage_manager: Any,
        pml_profiles: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        receivers_i: List[torch.Tensor],
        grid_spacing: Sequence[float],
        dt: float,
        nt: int,
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
        """Call scalar born backward C backend."""
        v = models[0]
        scatter = models[1]

        ndim = len(grid_spacing)
        n_bg = 2 + 2 * ndim

        grad_wfc = grad_wavefields[0]
        grad_wfp = grad_wavefields[1]
        grad_psi = grad_wavefields[2 : 2 + ndim]
        grad_zeta = grad_wavefields[2 + ndim : n_bg]

        grad_wfcsc = grad_wavefields[n_bg]
        grad_wfpsc = grad_wavefields[n_bg + 1]
        grad_psisc = grad_wavefields[n_bg + 2 : n_bg + 2 + ndim]
        grad_zetasc = grad_wavefields[n_bg + 2 + ndim :]

        # Aux: grad_psin, grad_zetan, grad_psinsc, grad_zetansc
        grad_psin = aux_wavefields[:ndim]
        grad_zetan = aux_wavefields[ndim : 2 * ndim]
        grad_psinsc = aux_wavefields[2 * ndim : 3 * ndim]
        grad_zetansc = aux_wavefields[3 * ndim :]

        rdx = [1 / dx for dx in grid_spacing]
        rdx2 = [1 / dx**2 for dx in grid_spacing]

        v_requires_grad = models_requires_grad[0]
        scatter_requires_grad = models_requires_grad[1]
        storage_mode = storage_manager.storage_mode

        n_sources = n_sources_per_shot[0]
        n_sources_sc = n_sources_per_shot[1]

        sa_req_grad = grad_f_list[0].numel() > 0
        sasc_req_grad = grad_f_list[1].numel() > 0

        return backend_func(  # type: ignore[no-any-return]
            v.data_ptr(),
            scatter.data_ptr(),
            grad_r[0].data_ptr(),
            grad_r[1].data_ptr(),
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
            grad_f_list[0].data_ptr(),
            grad_f_list[1].data_ptr(),
            grad_models[0].data_ptr(),
            grad_models[1].data_ptr(),
            grad_models_tmp_ptr[0],
            grad_models_tmp_ptr[1],
            *[profile.data_ptr() for profile in pml_profiles],
            sources_i[0].data_ptr(),
            receivers_i[0].data_ptr(),
            receivers_i[1].data_ptr(),
            *rdx,
            *rdx2,
            dt**2,
            step_nt,
            n_shots,
            *model_shape,
            n_sources * int(sa_req_grad),
            n_sources_sc * int(sasc_req_grad),
            n_receivers_per_shot[0],
            n_receivers_per_shot[1],
            step_ratio,
            storage_mode,
            storage_manager.shot_bytes_uncomp,
            storage_manager.shot_bytes_comp,
            v_requires_grad and storage_mode != deepwave.common.StorageMode.NONE,
            scatter_requires_grad and storage_mode != deepwave.common.StorageMode.NONE,
            model_batched[0],
            model_batched[1],
            storage_compression,
            step,
            *pml_b,
            *pml_e,
            aux,
            stream,
        )

    def prepare_backward(
        self,
        grad_wavefields: List[torch.Tensor],
        ndim: int,
    ) -> None:
        """Prepare scalar born backward by negating grad_wfp and grad_wfpsc."""
        n_bg = 2 + 2 * ndim
        grad_wavefields[1].neg_()
        grad_wavefields[n_bg + 1].neg_()

    def finalize_backward(
        self,
        grad_wavefields: List[torch.Tensor],
        ndim: int,
    ) -> None:
        """Finalize scalar born backward by negating grad_wfp and grad_wfpsc."""
        n_bg = 2 + 2 * ndim
        grad_wavefields[1].neg_()
        grad_wavefields[n_bg + 1].neg_()

    def create_aux_wavefields(
        self,
        wavefields: List[torch.Tensor],
        ndim: int,
        is_backward: bool = False,
    ) -> List[torch.Tensor]:
        """Create scalar born aux vars."""
        n_bg = 2 + 2 * ndim
        psi = wavefields[2 : 2 + ndim]
        psi_sc = wavefields[n_bg + 2 : n_bg + 2 + ndim]

        if is_backward:
            zeta = wavefields[2 + ndim : n_bg]
            zeta_sc = wavefields[n_bg + 2 + ndim :]

            # Layout: [psin..., zetan..., psinsc..., zetansc...]
            return (
                [torch.zeros_like(wf) for wf in psi]
                + [torch.zeros_like(wf) for wf in zeta]
                + [torch.zeros_like(wf) for wf in psi_sc]
                + [torch.zeros_like(wf) for wf in zeta_sc]
            )

        # Forward: [psin..., psinsc...]
        return [torch.zeros_like(wf) for wf in psi] + [
            torch.zeros_like(wf) for wf in psi_sc
        ]

    def swap_odd_step_wavefields(
        self,
        wavefields: List[torch.Tensor],
        aux_wavefields: List[torch.Tensor],
        ndim: int,
        is_backward: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Swap scalar born wavefields."""
        n_bg = 2 + 2 * ndim

        # Swap background wfc/wfp
        wavefields[0], wavefields[1] = wavefields[1], wavefields[0]
        # Swap scattered wfc/wfp
        wavefields[n_bg], wavefields[n_bg + 1] = wavefields[n_bg + 1], wavefields[n_bg]

        if is_backward:
            # Aux layout: [psin..., zetan..., psinsc..., zetansc...]
            psin = aux_wavefields[:ndim]
            zetan = aux_wavefields[ndim : 2 * ndim]
            psinsc = aux_wavefields[2 * ndim : 3 * ndim]
            zetansc = aux_wavefields[3 * ndim :]

            # Background
            for i in range(ndim):
                # psi
                wavefields[2 + i], psin[i] = psin[i], wavefields[2 + i]
                # zeta
                wavefields[2 + ndim + i], zetan[i] = zetan[i], wavefields[2 + ndim + i]

            # Scattered
            for i in range(ndim):
                # psi
                wavefields[n_bg + 2 + i], psinsc[i] = (
                    psinsc[i],
                    wavefields[n_bg + 2 + i],
                )
                # zeta
                wavefields[n_bg + 2 + ndim + i], zetansc[i] = (
                    zetansc[i],
                    wavefields[n_bg + 2 + ndim + i],
                )

        else:
            # Aux layout: [psin..., psinsc...]
            psin = aux_wavefields[:ndim]
            psinsc = aux_wavefields[ndim:]

            # Background
            for i in range(ndim):
                wavefields[2 + i], psin[i] = psin[i], wavefields[2 + i]

            # Scattered
            for i in range(ndim):
                wavefields[n_bg + 2 + i], psinsc[i] = (
                    psinsc[i],
                    wavefields[n_bg + 2 + i],
                )

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
        n_bg = 2 + 2 * ndim

        grad_psi = grad_wavefields[2 : 2 + ndim]
        grad_zeta = grad_wavefields[2 + ndim : n_bg]

        grad_psi_sc = grad_wavefields[n_bg + 2 : n_bg + 2 + ndim]
        grad_zeta_sc = grad_wavefields[n_bg + 2 + ndim :]

        for i in range(ndim):
            deepwave.common.zero_interior(grad_psi[i], fd_pad, pml_width, i)
            deepwave.common.zero_interior(grad_zeta[i], fd_pad, pml_width, i)
            deepwave.common.zero_interior(grad_psi_sc[i], fd_pad, pml_width, i)
            deepwave.common.zero_interior(grad_zeta_sc[i], fd_pad, pml_width, i)
