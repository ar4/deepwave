"""Elastic equation strategy for the propagator framework.

Implements the PropagatorEquation interface for the elastic
wave equation (staggered grid).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import deepwave.backend_utils
import deepwave.common
import deepwave.staggered_grid
from deepwave.elastic import zero_edges_and_interiors as _elastic_zero_edges


class ElasticEquation:
    """Equation class for elastic wave propagation on a staggered grid.

    Encapsulates all equation-specific logic needed by the generic
    propagator framework: model preparation, source scaling, PML
    profiles, wavefield layout, and callback naming.
    """

    @property
    def name(self) -> str:
        """C function prefix."""
        return "elastic"

    @property
    def grid_type(self) -> str:
        """Grid type."""
        return "staggered"

    @property
    def n_models(self) -> int:
        """Number of input model tensors: lamb, mu, buoyancy."""
        return 3

    @property
    def model_pad_modes(self) -> List[str]:
        """Padding mode per model tensor."""
        return ["replicate", "replicate", "replicate"]

    def get_ndim(
        self,
        models: List[torch.Tensor],
        initial_wavefields: List[Optional[torch.Tensor]],
        location_tensors: List[Optional[torch.Tensor]],
    ) -> int:
        """Determine spatial dimensions from inputs."""
        lamb = models[0]
        mu = models[1]
        buoyancy = models[2]

        # Sources and receivers: z, y, x, p
        source_locations_z = location_tensors[0] if len(location_tensors) > 0 else None
        source_locations_y = location_tensors[1] if len(location_tensors) > 1 else None
        source_locations_x = location_tensors[2] if len(location_tensors) > 2 else None
        source_locations_p = location_tensors[3] if len(location_tensors) > 3 else None
        receiver_locations_z = (
            location_tensors[4] if len(location_tensors) > 4 else None
        )
        receiver_locations_y = (
            location_tensors[5] if len(location_tensors) > 5 else None
        )
        receiver_locations_x = (
            location_tensors[6] if len(location_tensors) > 6 else None
        )
        receiver_locations_p = (
            location_tensors[7] if len(location_tensors) > 7 else None
        )

        # Velocity wavefields (extract from initial_wavefields)
        vz_0 = initial_wavefields[0] if len(initial_wavefields) > 0 else None
        # In 3D: indices 0-13 are z-fields, 14-22 are y-fields, 23-26 are x-fields
        # In 2D: indices 0-8 are y-fields, 9-12 are x-fields
        # In 1D: indices 0-3 are x-fields
        # For get_ndim we need to split correctly; use dimension heuristic
        vy_0 = None
        vx_0 = None
        if len(initial_wavefields) > 14:
            vy_0 = initial_wavefields[14]
            vx_0 = initial_wavefields[23] if len(initial_wavefields) > 23 else None
        elif len(initial_wavefields) > 9:
            vy_0 = initial_wavefields[0]
            vx_0 = initial_wavefields[9]
        elif len(initial_wavefields) > 0:
            vx_0 = initial_wavefields[0]

        return deepwave.common.get_ndim(
            [lamb, mu, buoyancy],
            [],
            [
                source_locations_z,
                source_locations_y,
                source_locations_x,
                source_locations_p,
                receiver_locations_z,
                receiver_locations_y,
                receiver_locations_x,
                receiver_locations_p,
            ],
            [vz_0],
            [vy_0],
            [vx_0],
        )

    def build_initial_wavefields(
        self,
        user_inputs: Dict[str, Optional[torch.Tensor]],
        ndim: int,
    ) -> List[Optional[torch.Tensor]]:
        """Convert named user wavefields to flat list.

        Order is dimension-dependent.
        """
        result: List[Optional[torch.Tensor]] = []
        if ndim >= 3:
            result.extend(
                [
                    user_inputs.get("vz_0"),
                    user_inputs.get("sigmazz_0"),
                    user_inputs.get("sigmayz_0"),
                    user_inputs.get("sigmaxz_0"),
                    user_inputs.get("m_vzz_0"),
                    user_inputs.get("m_vzy_0"),
                    user_inputs.get("m_vzx_0"),
                    user_inputs.get("m_vyz_0"),
                    user_inputs.get("m_vxz_0"),
                    user_inputs.get("m_sigmazzz_0"),
                    user_inputs.get("m_sigmayzy_0"),
                    user_inputs.get("m_sigmaxzx_0"),
                    user_inputs.get("m_sigmayzz_0"),
                    user_inputs.get("m_sigmaxzz_0"),
                ]
            )
        if ndim >= 2:
            result.extend(
                [
                    user_inputs.get("vy_0"),
                    user_inputs.get("sigmayy_0"),
                    user_inputs.get("sigmaxy_0"),
                    user_inputs.get("m_vyy_0"),
                    user_inputs.get("m_vyx_0"),
                    user_inputs.get("m_vxy_0"),
                    user_inputs.get("m_sigmayyy_0"),
                    user_inputs.get("m_sigmaxyy_0"),
                    user_inputs.get("m_sigmaxyx_0"),
                ]
            )
        result.extend(
            [
                user_inputs.get("vx_0"),
                user_inputs.get("sigmaxx_0"),
                user_inputs.get("m_vxx_0"),
                user_inputs.get("m_sigmaxxx_0"),
            ]
        )
        return result

    def get_fd_pad(self, accuracy: int, ndim: int) -> List[int]:
        """Finite difference padding for staggered grid."""
        return [accuracy // 2, accuracy // 2 - 1] * ndim

    def prepare_models(self, raw_models: List[torch.Tensor]) -> List[torch.Tensor]:
        """Transform [lamb, mu, buoyancy] -> prepared models.

        Adds harmonic-averaged mu and arithmetic-averaged buoyancy
        at staggered locations.
        """
        lamb = raw_models[0]
        mu = raw_models[1]
        buoyancy = raw_models[2]
        ndim = mu.ndim - 1
        rfmax = 1 / torch.finfo(mu.dtype).max ** (1 / 2)
        parameters: List[torch.Tensor] = []

        # Mu (harmonic mean)
        mu_safe = torch.where(mu.abs() > rfmax, mu, torch.ones_like(mu))
        if ndim >= 3:
            mask = (
                (rfmax < mu[..., 1:, 1:, :].abs())
                .logical_and(rfmax < mu[..., :-1, :-1, :].abs())
                .logical_and(rfmax < mu[..., 1:, :-1, :].abs())
                .logical_and(rfmax < mu[..., :-1, 1:, :].abs())
            )
            mu_zy_val = 4 / (
                1 / mu_safe[..., 1:, 1:, :]
                + 1 / mu_safe[..., :-1, :-1, :]
                + 1 / mu_safe[..., 1:, :-1, :]
                + 1 / mu_safe[..., :-1, 1:, :]
            )
            mu_zy = torch.where(mask, mu_zy_val, torch.zeros_like(mu_zy_val))
            mu_zy = torch.nn.functional.pad(mu_zy, (0, 0, 0, 1, 0, 1))
            parameters.append(mu_zy)
            del mask, mu_zy_val, mu_zy
            mask = (
                (rfmax < mu[..., 1:, :, 1:].abs())
                .logical_and(rfmax < mu[..., :-1, :, :-1].abs())
                .logical_and(rfmax < mu[..., 1:, :, :-1].abs())
                .logical_and(rfmax < mu[..., :-1, :, 1:].abs())
            )
            mu_zx_val = 4 / (
                1 / mu_safe[..., 1:, :, 1:]
                + 1 / mu_safe[..., :-1, :, :-1]
                + 1 / mu_safe[..., 1:, :, :-1]
                + 1 / mu_safe[..., :-1, :, 1:]
            )
            mu_zx = torch.where(mask, mu_zx_val, torch.zeros_like(mu_zx_val))
            mu_zx = torch.nn.functional.pad(mu_zx, (0, 1, 0, 0, 0, 1))
            parameters.append(mu_zx)
            del mask, mu_zx_val, mu_zx
        if ndim >= 2:
            mask = (
                (rfmax < mu[..., 1:, 1:].abs())
                .logical_and(rfmax < mu[..., :-1, :-1].abs())
                .logical_and(rfmax < mu[..., 1:, :-1].abs())
                .logical_and(rfmax < mu[..., :-1, 1:].abs())
            )
            mu_yx_val = 4 / (
                1 / mu_safe[..., 1:, 1:]
                + 1 / mu_safe[..., :-1, :-1]
                + 1 / mu_safe[..., 1:, :-1]
                + 1 / mu_safe[..., :-1, 1:]
            )
            mu_yx = torch.where(mask, mu_yx_val, torch.zeros_like(mu_yx_val))
            mu_yx = torch.nn.functional.pad(mu_yx, (0, 1, 0, 1))
            parameters.append(mu_yx)
            del mask, mu_yx_val, mu_yx

        # Buoyancy (inverse of arithmetic mean of density)
        mask = rfmax < buoyancy.abs()
        buoyancy_safe = torch.where(mask, buoyancy, torch.ones_like(buoyancy))
        rho = torch.where(mask, 1 / buoyancy_safe, torch.zeros_like(buoyancy))
        del mask, buoyancy_safe
        if ndim >= 3:
            rho_z = torch.nn.functional.pad(
                (rho[..., :-1, :, :] + rho[..., 1:, :, :]) / 2,
                (0, 0, 0, 0, 0, 1),
            )
            mask = rfmax < rho_z.abs()
            rho_z_safe = torch.where(mask, rho_z, torch.ones_like(rho_z))
            buoyancy_z = torch.where(mask, 1 / rho_z_safe, torch.zeros_like(buoyancy))
            parameters.append(buoyancy_z)
            del rho_z, mask, rho_z_safe, buoyancy_z
        if ndim >= 2:
            rho_y = torch.nn.functional.pad(
                (rho[..., :-1, :] + rho[..., 1:, :]) / 2, (0, 0, 0, 1)
            )
            mask = rfmax < rho_y.abs()
            rho_y_safe = torch.where(mask, rho_y, torch.ones_like(rho_y))
            buoyancy_y = torch.where(mask, 1 / rho_y_safe, torch.zeros_like(buoyancy))
            parameters.append(buoyancy_y)
            del rho_y, mask, rho_y_safe, buoyancy_y
        rho_x = torch.nn.functional.pad((rho[..., :-1] + rho[..., 1:]) / 2, (0, 1))
        mask = rfmax < rho_x.abs()
        rho_x_safe = torch.where(mask, rho_x, torch.ones_like(rho_x))
        buoyancy_x = torch.where(mask, 1 / rho_x_safe, torch.zeros_like(buoyancy))
        parameters.append(buoyancy_x)

        return [lamb, mu, *parameters]

    def scale_source_amplitudes(
        self,
        source_amplitudes: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        prepared_models: List[torch.Tensor],
        dt: float,
        n_shots: int,
    ) -> List[torch.Tensor]:
        """Scale source amplitudes.

        Force sources are multiplied by buoyancy at source location * dt.
        Pressure source is multiplied by -dt.
        Source order: [z, y, x, p] (force components first, then pressure).
        """
        ndim = (len(prepared_models) - 2) // 2
        # Models after prepare_models: [lamb, mu, mu_zy(3D), mu_zx(3D),
        #   mu_yx(2D/3D), buoyancy_z(3D), buoyancy_y(2D/3D), buoyancy_x]
        # Buoyancy models are the last ndim entries
        buoyancy_models = prepared_models[-ndim:]

        model_shape = prepared_models[0].shape[1:]
        flat_model_shape = int(torch.prod(torch.tensor(model_shape)).item())

        scaled = []
        # Force sources (indices 0 to ndim-1): multiply by buoyancy * dt
        for i in range(ndim):
            if source_amplitudes[i].numel() > 0:
                mask = sources_i[i] == deepwave.common.IGNORE_LOCATION
                sources_i_masked = sources_i[i].clone()
                sources_i_masked[mask] = 0
                scaled.append(
                    source_amplitudes[i]
                    * (
                        buoyancy_models[i]
                        .view(-1, flat_model_shape)
                        .expand(n_shots, -1)
                        .gather(1, sources_i_masked)
                    )
                    * dt
                )
            else:
                scaled.append(source_amplitudes[i])
        # Pressure source (last index): multiply by -dt
        if source_amplitudes[-1].numel() > 0:
            scaled.append(-source_amplitudes[-1] * dt)
        else:
            scaled.append(source_amplitudes[-1])
        return scaled

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
        """Set PML profiles using staggered_grid module."""
        return deepwave.staggered_grid.set_pml_profiles(
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
        """Compute max and min nonzero velocity from vp and vs."""
        lamb = raw_models[0]
        mu = raw_models[1]
        buoyancy = raw_models[2]
        if deepwave.common.is_inside_vmap(lamb):
            raise RuntimeError(
                "If using BatchedTensor inputs, you must specify max_vel"
            )
        vp, vs, _ = deepwave.common.lambmubuoyancy_to_vpvsrho(lamb, mu, buoyancy)
        max_model_vel = max(vp.abs().max().item(), vs.abs().max().item())
        vp_nonzero = vp[vp != 0]
        min_nonzero_vp = (
            vp_nonzero.abs().min().item() if vp_nonzero.numel() > 0 else 0.0
        )
        vs_nonzero = vs[vs != 0]
        min_nonzero_vs = (
            vs_nonzero.abs().min().item() if vs_nonzero.numel() > 0 else 0.0
        )
        if min_nonzero_vp == 0 and min_nonzero_vs == 0:
            min_nonzero_model_vel = 0.0
        elif min_nonzero_vp == 0:
            min_nonzero_model_vel = float(min_nonzero_vs)
        elif min_nonzero_vs == 0:
            min_nonzero_model_vel = float(min_nonzero_vp)
        else:
            min_nonzero_model_vel = float(min(min_nonzero_vp, min_nonzero_vs))
        del vp, vs, vp_nonzero, vs_nonzero
        return max_model_vel, min_nonzero_model_vel

    def n_wavefields(self, ndim: int) -> int:
        """Number of wavefield tensors (dimension-dependent)."""
        if ndim == 3:
            return 27  # 14 (z) + 9 (y) + 4 (x)
        if ndim == 2:
            return 13  # 9 (y) + 4 (x)
        return 4  # 4 (x)

    def n_source_types(self, ndim: int) -> int:
        """Number of source types: force components + pressure."""
        return ndim + 1

    def n_receiver_types(self, ndim: int) -> int:
        """Number of receiver types: velocity components + pressure."""
        return ndim + 1

    def get_storage_requires_grad(
        self, raw_models: List[torch.Tensor], ndim: int
    ) -> List[bool]:
        """Which model params require gradient storage.

        Elastic storage slots are dimension-dependent, combining
        lamb, mu, and buoyancy gradient requirements.
        """
        lamb_requires_grad = raw_models[0].requires_grad
        mu_requires_grad = raw_models[1].requires_grad
        buoyancy_requires_grad = raw_models[2].requires_grad
        if ndim == 3:
            return [
                buoyancy_requires_grad,
                lamb_requires_grad or mu_requires_grad,
                mu_requires_grad,
                mu_requires_grad,
                buoyancy_requires_grad,
                lamb_requires_grad or mu_requires_grad,
                mu_requires_grad,
                buoyancy_requires_grad,
                lamb_requires_grad or mu_requires_grad,
            ]
        if ndim == 2:
            return [
                buoyancy_requires_grad,
                lamb_requires_grad or mu_requires_grad,
                mu_requires_grad,
                buoyancy_requires_grad,
                lamb_requires_grad or mu_requires_grad,
            ]
        return [
            buoyancy_requires_grad,
            lamb_requires_grad or mu_requires_grad,
        ]

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
        """Prepare wavefields for the C kernel (staggered grid)."""
        return deepwave.common.prepare_initial_wavefields(
            wavefields,
            ndim,
            accuracy,
            device,
            dtype,
            size_with_batch,
            pml_width,
            staggered=True,
        )

    def get_callback_wavefield_names(self, ndim: int) -> List[str]:
        """Names for the callback wavefield dict."""
        names: List[str] = []
        if ndim >= 3:
            names.extend(
                [
                    "vz_0",
                    "sigmazz_0",
                    "sigmayz_0",
                    "sigmaxz_0",
                    "m_vzz_0",
                    "m_vzy_0",
                    "m_vzx_0",
                    "m_vyz_0",
                    "m_vxz_0",
                    "m_sigmazzz_0",
                    "m_sigmayzy_0",
                    "m_sigmaxzx_0",
                    "m_sigmayzz_0",
                    "m_sigmaxzz_0",
                ]
            )
        if ndim >= 2:
            names.extend(
                [
                    "vy_0",
                    "sigmayy_0",
                    "sigmaxy_0",
                    "m_vyy_0",
                    "m_vyx_0",
                    "m_vxy_0",
                    "m_sigmayyy_0",
                    "m_sigmaxyy_0",
                    "m_sigmaxyx_0",
                ]
            )
        names.extend(
            [
                "vx_0",
                "sigmaxx_0",
                "m_vxx_0",
                "m_sigmaxxx_0",
            ]
        )
        return names

    def get_callback_model_names(self, ndim: int) -> List[str]:
        """Names for the callback model dict."""
        if ndim >= 3:
            return [
                "lamb",
                "mu",
                "mu_zy",
                "mu_zx",
                "mu_yx",
                "buoyancy_z",
                "buoyancy_y",
                "buoyancy_x",
            ]
        if ndim >= 2:
            return ["lamb", "mu", "mu_yx", "buoyancy_y", "buoyancy_x"]
        return ["lamb", "mu", "buoyancy_x"]

    def pack_args(
        self,
        models: List[torch.Tensor],
        source_amplitudes: List[torch.Tensor],
        pml_profiles: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        receivers_i: List[torch.Tensor],
        wavefields: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        """Pack tensors in elastic ForwardFunc order.

        Elastic ForwardFunc has pml_profiles in metadata, not in *args.
        We pack them into *args for the generic framework, with unpack_args
        extracting them.
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

        Layout: models, source_amplitudes, pml_profiles[4*ndim],
        sources_i, receivers_i, wavefields.
        """
        pos = 0
        n_mod = self.n_models + (ndim - 1) if ndim >= 2 else self.n_models
        # After prepare_models: 1D=3, 2D=5, 3D=8
        if ndim == 3:
            n_mod = 8
        elif ndim == 2:
            n_mod = 5
        else:
            n_mod = 3
        n_src = ndim + 1
        n_pml = 4 * ndim
        n_wf = self.n_wavefields(ndim)

        models = list(args[pos : pos + n_mod])
        pos += n_mod
        source_amplitudes = list(args[pos : pos + n_src])
        pos += n_src
        pml_profiles = list(args[pos : pos + n_pml])
        pos += n_pml
        sources_i = list(args[pos : pos + n_src])
        pos += n_src
        receivers_i = list(args[pos : pos + n_src])
        pos += n_src
        wavefields = list(args[pos : pos + n_wf])
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
        """Call elastic C backend."""
        rdx = [1 / dx for dx in grid_spacing]
        storage_mode = storage_manager.storage_mode

        return backend_func(  # type: ignore[no-any-return]
            *[model.data_ptr() for model in models],
            *[amp.data_ptr() for amp in source_amplitudes],
            *[field.data_ptr() for field in wavefields],
            *storage_manager.storage_ptrs,
            *[amp.data_ptr() for amp in receiver_amplitudes],
            *[profile.data_ptr() for profile in pml_profiles],
            *[locs.data_ptr() for locs in sources_i],
            *[locs.data_ptr() for locs in receivers_i],
            *rdx,
            dt,
            step_nt,
            n_shots,
            *model_shape,
            *n_sources_per_shot,
            *n_receivers_per_shot,
            step_ratio,
            storage_mode,
            storage_manager.shot_bytes_uncomp,
            storage_manager.shot_bytes_comp,
            *[
                models_requires_grad[i]
                and storage_mode != deepwave.common.StorageMode.NONE
                for i in range(min(len(models_requires_grad), 3))
            ],
            *model_batched,
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
        """Call elastic backward C backend."""
        rdx = [1 / dx for dx in grid_spacing]
        storage_mode = storage_manager.storage_mode

        return backend_func(  # type: ignore[no-any-return]
            *[model.data_ptr() for model in models],
            *[amp.data_ptr() for amp in grad_r],
            *[field.data_ptr() for field in grad_wavefields],
            *[field.data_ptr() for field in aux_wavefields],
            *storage_manager.storage_ptrs,
            *[amp.data_ptr() for amp in grad_f_list],
            *[model.data_ptr() for model in grad_models],
            *grad_models_tmp_ptr,
            *[profile.data_ptr() for profile in pml_profiles],
            *[locs.data_ptr() for locs in sources_i],
            *[locs.data_ptr() for locs in receivers_i],
            *rdx,
            dt,
            step_nt,
            n_shots,
            *model_shape,
            *[
                n_sources_per_shot[i] * models_requires_grad[i]
                for i in range(len(n_sources_per_shot))
            ],
            *n_receivers_per_shot,
            step_ratio,
            storage_mode,
            storage_manager.shot_bytes_uncomp,
            storage_manager.shot_bytes_comp,
            *[
                models_requires_grad[i]
                and storage_mode != deepwave.common.StorageMode.NONE
                for i in range(min(len(models_requires_grad), 3))
            ],
            *model_batched,
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
        """Call elastic Born backend (Not Implemented)."""
        raise NotImplementedError("Elastic Born not implemented yet.")

    def create_aux_wavefields(
        self,
        wavefields: List[torch.Tensor],
        ndim: int,
        is_backward: bool = False,
    ) -> List[torch.Tensor]:
        """Create elastic aux wavefields.

        Forward: [].
        Backward: dimension-dependent temp vars.
        """
        if not is_backward:
            return []

        aux: List[torch.Tensor] = []
        if ndim >= 3:
            aux.extend([torch.zeros_like(wavefields[0]) for _ in range(5)])
        if ndim >= 2:
            aux.extend([torch.zeros_like(wavefields[0]) for _ in range(3)])
        aux.append(torch.zeros_like(wavefields[0]))
        return aux

    def swap_odd_step_wavefields(
        self,
        wavefields: List[torch.Tensor],
        aux_wavefields: List[torch.Tensor],
        ndim: int,
        is_backward: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Swap wavefields after odd steps.

        Forward: no swap.
        Backward: Elastic swaps specific stress wavefield indices based on ndim.
        """
        if not is_backward:
            return wavefields, aux_wavefields

        if ndim == 3:
            wavefields[9:14], aux_wavefields[-9:-4] = (
                aux_wavefields[-9:-4],
                wavefields[9:14],
            )
            wavefields[20:23], aux_wavefields[-4:-1] = (
                aux_wavefields[-4:-1],
                wavefields[20:23],
            )
            wavefields[26], aux_wavefields[-1] = (
                aux_wavefields[-1],
                wavefields[26],
            )
        elif ndim == 2:
            wavefields[6:9], aux_wavefields[-4:-1] = (
                aux_wavefields[-4:-1],
                wavefields[6:9],
            )
            wavefields[12], aux_wavefields[-1] = (
                aux_wavefields[-1],
                wavefields[12],
            )
        else:
            wavefields[3], aux_wavefields[-1] = (
                aux_wavefields[-1],
                wavefields[3],
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
        """Zero elastic backward wavefields using staggered half-grid masks."""
        fd_pad = accuracy // 2
        _elastic_zero_edges(grad_wavefields, ndim, fd_pad, fd_pad_list, pml_width)
