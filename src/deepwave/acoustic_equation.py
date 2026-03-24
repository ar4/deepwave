"""Acoustic equation strategy for the propagator framework.

Implements the PropagatorEquation interface for the variable-density
acoustic wave equation (staggered grid).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import deepwave.backend_utils
import deepwave.common
import deepwave.staggered_grid
from deepwave.propagator_equation import PropagatorEquation


def zero_edges_and_interiors(
    wavefields: List[torch.Tensor],
    ndim: int,
    fd_pad: int,
    fd_pad_list: List[int],
    pml_width: List[int],
    interior: bool = True,
) -> None:
    """Zeros the edges and/or interiors of wavefields."""
    num_vars = len(wavefields)
    half_grid_mask: List[List[int]] = [[] for _ in range(num_vars)]
    interior_mask: List[Tuple[int, int]] = []

    if ndim == 3:
        # vz (1), psiz (7) -> shifted z (dim 0)
        half_grid_mask[1] = [0]
        half_grid_mask[7] = [0]
        # vy (2), psiy (8) -> shifted y (dim 1)
        half_grid_mask[2] = [1]
        half_grid_mask[8] = [1]
        # vx (3), psix (9) -> shifted x (dim 2)
        half_grid_mask[3] = [2]
        half_grid_mask[9] = [2]
        if interior:
            interior_mask = [(4, 0), (7, 0), (5, 1), (8, 1), (6, 2), (9, 2)]
    elif ndim == 2:
        # vy (1), psiy (5) -> shifted y (dim 0)
        half_grid_mask[1] = [0]
        half_grid_mask[5] = [0]
        # vx (2), psix (6) -> shifted x (dim 1)
        half_grid_mask[2] = [1]
        half_grid_mask[6] = [1]
        if interior:
            interior_mask = [(3, 0), (5, 0), (4, 1), (6, 1)]
    elif ndim == 1:
        # vx (1), psix (3) -> shifted x (dim 0)
        half_grid_mask[1] = [0]
        half_grid_mask[3] = [0]
        if interior:
            interior_mask = [(2, 0), (3, 0)]

    deepwave.common.zero_edges_and_interiors(
        wavefields,
        ndim,
        fd_pad,
        fd_pad_list,
        pml_width,
        half_grid_mask,
        interior_mask,
    )


class AcousticEquation(PropagatorEquation):
    """Equation class for acoustic wave propagation on a staggered grid.

    Encapsulates all equation-specific logic needed by the generic
    propagator framework: model preparation, source scaling, PML
    profiles, wavefield layout, and callback naming.
    """

    @property
    def name(self) -> str:
        """C function prefix."""
        return "acoustic"

    @property
    def grid_type(self) -> str:
        """Grid type."""
        return "staggered"

    @property
    def n_models(self) -> int:
        """Number of input model tensors: v, rho."""
        return 2

    @property
    def model_pad_modes(self) -> List[str]:
        """Padding mode per model tensor."""
        return ["replicate", "replicate"]

    def get_ndim(
        self,
        models: List[torch.Tensor],
        initial_wavefields: List[Optional[torch.Tensor]],
        location_tensors: List[Optional[torch.Tensor]],
    ) -> int:
        """Determine spatial dimensions from inputs."""
        v = models[0]
        rho = models[1]
        pressure_0 = initial_wavefields[0]
        # velocity wavefields: vz, vy, vx (dimension dependent)
        vz_0 = initial_wavefields[1] if len(initial_wavefields) > 1 else None
        vy_0 = initial_wavefields[2] if len(initial_wavefields) > 2 else None
        vx_0 = initial_wavefields[3] if len(initial_wavefields) > 3 else None
        # phi (PML for pressure): phi_z, phi_y, phi_x
        phi_z_0 = initial_wavefields[4] if len(initial_wavefields) > 4 else None
        phi_y_0 = initial_wavefields[5] if len(initial_wavefields) > 5 else None
        phi_x_0 = initial_wavefields[6] if len(initial_wavefields) > 6 else None
        # psi (PML for velocity): psi_z, psi_y, psi_x
        psi_z_0 = initial_wavefields[7] if len(initial_wavefields) > 7 else None
        psi_y_0 = initial_wavefields[8] if len(initial_wavefields) > 8 else None
        psi_x_0 = initial_wavefields[9] if len(initial_wavefields) > 9 else None

        # Sources and receivers: p, z, y, x
        source_locations_p = location_tensors[0]
        source_locations_z = location_tensors[1] if len(location_tensors) > 1 else None
        source_locations_y = location_tensors[2] if len(location_tensors) > 2 else None
        source_locations_x = location_tensors[3] if len(location_tensors) > 3 else None
        receiver_locations_p = (
            location_tensors[4] if len(location_tensors) > 4 else None
        )
        receiver_locations_z = (
            location_tensors[5] if len(location_tensors) > 5 else None
        )
        receiver_locations_y = (
            location_tensors[6] if len(location_tensors) > 6 else None
        )
        receiver_locations_x = (
            location_tensors[7] if len(location_tensors) > 7 else None
        )

        return deepwave.common.get_ndim(
            [v, rho],
            [pressure_0],
            [
                source_locations_p,
                source_locations_z,
                source_locations_y,
                source_locations_x,
                receiver_locations_p,
                receiver_locations_z,
                receiver_locations_y,
                receiver_locations_x,
            ],
            [vz_0, phi_z_0, psi_z_0],
            [vy_0, phi_y_0, psi_y_0],
            [vx_0, phi_x_0, psi_x_0],
        )

    def build_initial_wavefields(
        self,
        user_inputs: Dict[str, Optional[torch.Tensor]],
        ndim: int,
    ) -> List[Optional[torch.Tensor]]:
        """Convert named user wavefields to flat list.

        Order: pressure, velocity(z,y,x), phi(z,y,x), psi(z,y,x).
        """
        result: List[Optional[torch.Tensor]] = []
        # Pressure
        result.append(user_inputs.get("pressure_0"))
        # Velocity
        if ndim == 3:
            result.append(user_inputs.get("vz_0"))
        if ndim >= 2:
            result.append(user_inputs.get("vy_0"))
        result.append(user_inputs.get("vx_0"))
        # Phi (PML for pressure)
        if ndim == 3:
            result.append(user_inputs.get("phi_z_0"))
        if ndim >= 2:
            result.append(user_inputs.get("phi_y_0"))
        result.append(user_inputs.get("phi_x_0"))
        # Psi (PML for velocity)
        if ndim == 3:
            result.append(user_inputs.get("psi_z_0"))
        if ndim >= 2:
            result.append(user_inputs.get("psi_y_0"))
        result.append(user_inputs.get("psi_x_0"))
        return result

    def get_fd_pad(self, accuracy: int, ndim: int) -> List[int]:
        """Finite difference padding for staggered grid."""
        return [accuracy // 2, accuracy // 2 - 1] * ndim

    def prepare_models(self, raw_models: List[torch.Tensor]) -> List[torch.Tensor]:
        """Transform [v, rho] -> [K, buoyancy_z, buoyancy_y, buoyancy_x]."""
        v = raw_models[0]
        rho = raw_models[1]
        ndim = v.ndim - 1
        rfmax = 1 / torch.finfo(v.dtype).max ** (1 / 2)
        models = []

        # Bulk modulus K = rho * v^2
        k = rho * v**2
        models.append(k)

        # Buoyancy (inverse of arithmetic mean of density)
        if ndim >= 3:
            rho_z = torch.nn.functional.pad(
                (rho[..., :-1, :, :] + rho[..., 1:, :, :]) / 2,
                (0, 0, 0, 0, 0, 1),
            )
            mask = rfmax < rho_z.abs()
            rho_z_safe = torch.where(mask, rho_z, torch.ones_like(rho_z))
            buoyancy_z = torch.where(mask, 1 / rho_z_safe, torch.zeros_like(rho))
            models.append(buoyancy_z)
            del rho_z, mask, rho_z_safe, buoyancy_z

        if ndim >= 2:
            rho_y = torch.nn.functional.pad(
                (rho[..., :-1, :] + rho[..., 1:, :]) / 2, (0, 0, 0, 1)
            )
            mask = rfmax < rho_y.abs()
            rho_y_safe = torch.where(mask, rho_y, torch.ones_like(rho_y))
            buoyancy_y = torch.where(mask, 1 / rho_y_safe, torch.zeros_like(rho))
            models.append(buoyancy_y)
            del rho_y, mask, rho_y_safe, buoyancy_y

        rho_x = torch.nn.functional.pad((rho[..., :-1] + rho[..., 1:]) / 2, (0, 1))
        mask = rfmax < rho_x.abs()
        rho_x_safe = torch.where(mask, rho_x, torch.ones_like(rho_x))
        buoyancy_x = torch.where(mask, 1 / rho_x_safe, torch.zeros_like(rho))
        models.append(buoyancy_x)

        return models

    def scale_source_amplitudes(
        self,
        source_amplitudes: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        prepared_models: List[torch.Tensor],
        dt: float,
        n_shots: int,
    ) -> List[torch.Tensor]:
        """Scale source amplitudes by model value at source location * dt."""
        scaled = []
        model_shape = prepared_models[0].shape[1:]
        flat_model_shape = int(torch.prod(torch.tensor(model_shape)).item())
        for src_amp, src_loc, model in zip(
            source_amplitudes, sources_i, prepared_models
        ):
            if src_amp.numel() > 0:
                mask = src_loc == deepwave.common.IGNORE_LOCATION
                sources_i_masked = src_loc.clone()
                sources_i_masked[mask] = 0
                scaled.append(
                    src_amp
                    * (
                        model.view(-1, flat_model_shape)
                        .expand(n_shots, -1)
                        .gather(1, sources_i_masked)
                    )
                    * dt
                )
            else:
                scaled.append(src_amp)
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
        """Number of wavefield tensors: 1 (pressure) + 3*ndim (vel+phi+psi)."""
        return 1 + 3 * ndim

    def n_source_types(self, ndim: int) -> int:
        """Number of source types: pressure + velocity components."""
        return 1 + ndim

    def n_receiver_types(self, ndim: int) -> int:
        """Number of receiver types: pressure + velocity components."""
        return 1 + ndim

    def get_storage_requires_grad(
        self, raw_models: List[torch.Tensor], ndim: int
    ) -> List[bool]:
        """Which model params require gradient storage.

        Acoustic has K (from v) and buoyancy components (from rho).
        The pattern is: [K_requires_grad] + [rho_requires_grad] * ndim.
        """
        k_requires_grad = raw_models[0].requires_grad or raw_models[1].requires_grad
        rho_requires_grad = raw_models[1].requires_grad
        return [k_requires_grad] + [rho_requires_grad] * ndim

    def get_model_batched(
        self, raw_models: List[torch.Tensor], ndim: int
    ) -> List[bool]:
        """Which prepared model params are batched."""
        v_batched = (
            raw_models[0].ndim == ndim + 1 and raw_models[0].shape[0] > 1
        )
        rho_batched = (
            raw_models[1].ndim == ndim + 1 and raw_models[1].shape[0] > 1
        )
        return [v_batched or rho_batched] + [rho_batched] * ndim

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
        wavefields = deepwave.common.prepare_initial_wavefields(
            wavefields,
            ndim,
            accuracy,
            device,
            dtype,
            size_with_batch,
            pml_width,
            staggered=True,
        )
        fd_pad = accuracy // 2
        fd_pad_list = [fd_pad, fd_pad - 1] * ndim
        zero_edges_and_interiors(wavefields, ndim, fd_pad, fd_pad_list, pml_width)
        return wavefields

    def get_callback_wavefield_names(self, ndim: int) -> List[str]:
        """Names for the callback wavefield dict."""
        names = ["pressure_0"]
        if ndim == 3:
            names.extend(["vz_0", "vy_0", "vx_0"])
        elif ndim == 2:
            names.extend(["vy_0", "vx_0"])
        elif ndim == 1:
            names.extend(["vx_0"])
        if ndim == 3:
            names.extend(["phi_z_0", "phi_y_0", "phi_x_0"])
        elif ndim == 2:
            names.extend(["phi_y_0", "phi_x_0"])
        elif ndim == 1:
            names.extend(["phi_x_0"])
        if ndim == 3:
            names.extend(["psi_z_0", "psi_y_0", "psi_x_0"])
        elif ndim == 2:
            names.extend(["psi_y_0", "psi_x_0"])
        elif ndim == 1:
            names.extend(["psi_x_0"])
        return names

    def get_callback_model_names(self, ndim: int) -> List[str]:
        """Names for the callback model dict."""
        names = ["K"]
        if ndim == 3:
            names.extend(["Bz", "By", "Bx"])
        elif ndim == 2:
            names.extend(["By", "Bx"])
        elif ndim == 1:
            names.extend(["Bx"])
        return names

    def pack_args(
        self,
        models: List[torch.Tensor],
        source_amplitudes: List[torch.Tensor],
        pml_profiles: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        receivers_i: List[torch.Tensor],
        wavefields: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        """Pack tensors in acoustic ForwardFunc order."""
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

        Layout: models[ndim+1], source_amplitudes[ndim+1],
        pml_profiles[4*ndim], sources_i[ndim+1], receivers_i[ndim+1],
        wavefields[1+3*ndim].
        """
        pos = 0
        n_mod = self.n_models
        n_src = ndim + 1
        n_pml = 4 * ndim
        n_wf = 1 + 3 * ndim
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
        """Call acoustic C backend."""
        rdx = [1 / dx for dx in grid_spacing]
        storage_mode = storage_manager.storage_mode

        return backend_func(  # type: ignore[no-any-return]
            *[m.data_ptr() for m in models],
            *[amp.data_ptr() for amp in source_amplitudes],
            *[w.data_ptr() for w in wavefields],
            *storage_manager.storage_ptrs,
            *[amp.data_ptr() for amp in receiver_amplitudes],
            *[p.data_ptr() for p in pml_profiles],
            *[loc.data_ptr() for loc in sources_i],
            *[loc.data_ptr() for loc in receivers_i],
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
            models_requires_grad[0]
            and storage_mode != deepwave.common.StorageMode.NONE,
            models_requires_grad[1]
            and storage_mode != deepwave.common.StorageMode.NONE,
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
        """Call acoustic backward C backend."""
        rdx = [1 / dx for dx in grid_spacing]
        storage_mode = storage_manager.storage_mode

        models_ptr = [m.data_ptr() for m in models]
        grad_r_ptr = [g.data_ptr() for g in grad_r]
        grad_wavefields_ptr = [field.data_ptr() for field in grad_wavefields]
        aux_wavefields_ptr = [field.data_ptr() for field in aux_wavefields]
        grad_f_list_ptr = [g.data_ptr() for g in grad_f_list]
        grad_models_ptr = [g.data_ptr() for g in grad_models]
        pml_profiles_ptr = [p.data_ptr() for p in pml_profiles]
        sources_i_ptr = [loc.data_ptr() for loc in sources_i]
        receivers_i_ptr = [loc.data_ptr() for loc in receivers_i]
        n_sources = [n * (g.numel() > 0) for n, g in zip(n_sources_per_shot, grad_f_list)]

        print(f"DEBUG: ndim={len(grid_spacing)}")
        print(f"DEBUG: len(models_ptr)={len(models_ptr)}")
        print(f"DEBUG: len(grad_r_ptr)={len(grad_r_ptr)}")
        print(f"DEBUG: len(grad_wavefields_ptr)={len(grad_wavefields_ptr)}")
        print(f"DEBUG: len(aux_wavefields_ptr)={len(aux_wavefields_ptr)}")
        print(f"DEBUG: len(storage_manager.storage_ptrs)={len(storage_manager.storage_ptrs)}")
        print(f"DEBUG: len(grad_f_list_ptr)={len(grad_f_list_ptr)}")
        print(f"DEBUG: len(grad_models_ptr)={len(grad_models_ptr)}")
        print(f"DEBUG: len(grad_models_tmp_ptr)={len(grad_models_tmp_ptr)}")
        print(f"DEBUG: len(pml_profiles_ptr)={len(pml_profiles_ptr)}")
        print(f"DEBUG: len(sources_i_ptr)={len(sources_i_ptr)}")
        print(f"DEBUG: len(receivers_i_ptr)={len(receivers_i_ptr)}")
        print(f"DEBUG: len(rdx)={len(rdx)}")
        print(f"DEBUG: len(model_shape)={len(model_shape)}")
        print(f"DEBUG: len(n_sources)={len(n_sources)}")
        print(f"DEBUG: len(n_receivers_per_shot)={len(n_receivers_per_shot)}")
        print(f"DEBUG: len(pml_b)={len(pml_b)}")
        print(f"DEBUG: len(pml_e)={len(pml_e)}")

        final_args = []
        final_args.extend(models_ptr)
        final_args.extend(grad_r_ptr)
        final_args.extend(grad_wavefields_ptr)
        final_args.extend(aux_wavefields_ptr)
        final_args.extend(storage_manager.storage_ptrs)
        final_args.extend(grad_f_list_ptr)
        final_args.extend(grad_models_ptr)
        final_args.extend(grad_models_tmp_ptr)
        final_args.extend(pml_profiles_ptr)
        final_args.extend(sources_i_ptr)
        final_args.extend(receivers_i_ptr)
        final_args.extend(rdx)
        final_args.append(dt)
        final_args.append(step_nt)
        final_args.append(n_shots)
        final_args.extend(model_shape)
        final_args.extend(n_sources)
        final_args.extend(n_receivers_per_shot)
        final_args.append(step_ratio)
        final_args.append(storage_mode)
        final_args.append(storage_manager.shot_bytes_uncomp)
        final_args.append(storage_manager.shot_bytes_comp)
        final_args.append(
            models_requires_grad[0]
            and storage_mode != deepwave.common.StorageMode.NONE
        )
        final_args.append(
            models_requires_grad[1]
            and storage_mode != deepwave.common.StorageMode.NONE
        )
        final_args.append(model_batched[0])
        final_args.append(model_batched[1])
        final_args.append(storage_compression)
        final_args.append(step)
        final_args.extend(pml_b)
        final_args.extend(pml_e)
        final_args.append(aux)
        final_args.append(stream)

        print(f"DEBUG: final_args has {len(final_args)} elements")
        return backend_func(*final_args)

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
        nt: int,
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
        """Call acoustic Born backend (Not Implemented)."""
        raise NotImplementedError("Acoustic Born not implemented yet.")

    def create_aux_wavefields(
        self,
        wavefields: List[torch.Tensor],
        ndim: int,
        is_backward: bool = False,
    ) -> List[torch.Tensor]:
        """Create acoustic aux wavefields.

        Forward: [].
        Backward: ndim zero tensors.
        """
        if is_backward:
            return [torch.zeros_like(wavefields[0]) for _ in range(ndim)]
        return []

    def swap_odd_step_wavefields(
        self,
        wavefields: List[torch.Tensor],
        aux_wavefields: List[torch.Tensor],
        ndim: int,
        is_backward: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Swap acoustic wavefields.

        Forward: no swap.
        Backward: swap last ndim wavefields with aux.
        """
        if is_backward:
            wavefields[-ndim:], aux_wavefields[:] = (
                aux_wavefields[:],
                wavefields[-ndim:],
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
        """Zero acoustic backward wavefields using staggered half-grid masks."""
        fd_pad = accuracy // 2
        zero_edges_and_interiors(grad_wavefields, ndim, fd_pad, fd_pad_list, pml_width)
