"""Abstract base class for propagator equation strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


class PropagatorEquation(ABC):
    """Abstract base class encapsulating equation-specific propagator logic."""

    @property
    @abstractmethod
    def name(self) -> str:
        """C function prefix: 'scalar', 'acoustic', 'elastic'."""
        ...

    @property
    @abstractmethod
    def grid_type(self) -> str:
        """'regular' or 'staggered'."""
        ...

    @property
    @abstractmethod
    def n_models(self) -> int:
        """Number of input model tensors (before prepare_models)."""
        ...

    @property
    def model_pad_modes(self) -> List[str]:
        """Padding mode per model tensor. Default: replicate for all."""
        return ["replicate"] * self.n_models

    @abstractmethod
    def get_ndim(
        self,
        models: List[torch.Tensor],
        initial_wavefields: List[Optional[torch.Tensor]],
        location_tensors: List[Optional[torch.Tensor]],
    ) -> int:
        """Determine spatial dimensions from inputs."""
        ...

    @abstractmethod
    def build_initial_wavefields(
        self, user_inputs: Dict[str, Optional[torch.Tensor]], ndim: int
    ) -> List[Optional[torch.Tensor]]:
        """Convert user-provided named wavefields to flat list for setup_propagator."""
        ...

    @abstractmethod
    def get_fd_pad(self, accuracy: int, ndim: int) -> List[int]:
        """Finite difference padding.

        Regular: [acc/2]*2*ndim. Staggered: [acc/2, acc/2-1]*ndim.
        """
        ...

    @abstractmethod
    def prepare_models(self, raw_models: List[torch.Tensor]) -> List[torch.Tensor]:
        """Transform input models for the C kernel.

        Scalar: identity [v] -> [v].
        Acoustic: [v, rho] -> [K, buoyancy...].
        Elastic: [lamb, mu, buoyancy] -> [lamb, mu, mu_..., buoyancy...].
        """
        ...

    @abstractmethod
    def scale_source_amplitudes(
        self,
        source_amplitudes: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        prepared_models: List[torch.Tensor],
        dt: float,
        n_shots: int,
    ) -> List[torch.Tensor]:
        """Scale source amplitudes before calling the C kernel.

        Scalar: -source * v^2 * dt^2.
        Acoustic: multiply by model buoyancy at source locations * dt.
        Elastic: buoyancy-weighted for force sources, -dt for pressure source.
        """
        ...

    @abstractmethod
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
        """Set PML profiles.

        Regular grid -> regular_grid module. Staggered -> staggered_grid.
        """
        ...

    @abstractmethod
    def get_max_vel(self, raw_models: List[torch.Tensor]) -> Tuple[float, float]:
        """Compute (max_vel, min_nonzero_vel) from raw models."""
        ...

    @abstractmethod
    def n_wavefields(self, ndim: int) -> int:
        """Number of wavefield tensors at given ndim."""
        ...

    @abstractmethod
    def n_source_types(self, ndim: int) -> int:
        """Number of source type tensors at given ndim."""
        ...

    @abstractmethod
    def n_receiver_types(self, ndim: int) -> int:
        """Number of receiver type tensors at given ndim."""
        ...

    @abstractmethod
    def get_storage_requires_grad(
        self, raw_models: List[torch.Tensor], ndim: int
    ) -> List[bool]:
        """Which model params require gradient storage allocation.

        Uses prepared models and ndim to compute the flags
        needed by setup_storage.
        """
        ...

    @abstractmethod
    def get_model_batched(
        self, raw_models: List[torch.Tensor], ndim: int
    ) -> List[bool]:
        """Which prepared model params are batched."""
        ...

    @abstractmethod
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
        """Prepare wavefields (create_or_pad + zero_interior). Grid-specific."""
        ...

    @abstractmethod
    def get_callback_wavefield_names(self, ndim: int) -> List[str]:
        """Names for the callback wavefield dict."""
        ...

    @abstractmethod
    def get_callback_model_names(self, ndim: int) -> List[str]:
        """Names for the callback model dict."""
        ...

    @abstractmethod
    def pack_args(
        self,
        models: List[torch.Tensor],
        source_amplitudes: List[torch.Tensor],
        pml_profiles: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        receivers_i: List[torch.Tensor],
        wavefields: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        """Pack tensor arguments in canonical order for GenericForwardFunc.

        Returns a tuple whose layout is equation-specific. The generic
        forward function passes these as *args, and unpack_args recovers
        the semantic groups.
        """
        ...

    @abstractmethod
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
        """Unpack the flat *args tuple from GenericForwardFunc.forward.

        Returns (models, source_amplitudes, pml_profiles, sources_i,
                 receivers_i, wavefields).
        """
        ...

    @abstractmethod
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
        """Call the C/CUDA forward backend function.

        Each equation implements this with its specific C function
        argument order. All arguments are passed through from the
        generic framework.

        Returns 0 on success.
        """
        ...

    @abstractmethod
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
        """Call the C/CUDA backward backend function.

        Each equation implements this with its specific C function
        argument order.

        Returns 0 on success.
        """
        ...

    @abstractmethod
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
        """Call the C/CUDA Born backend function.

        Used for double backward (Hessian-vector product).
        Returns 0 on success.
        """
        ...

    def prepare_backward(
        self,
        grad_wavefields: List[torch.Tensor],
        ndim: int,
    ) -> None:
        """Prepare gradient wavefields for backprop (e.g. negation)."""
        _ = (grad_wavefields, ndim)

    def finalize_backward(
        self,
        grad_wavefields: List[torch.Tensor],
        ndim: int,
    ) -> None:
        """Finalize gradient wavefields after backprop."""
        _ = (grad_wavefields, ndim)

    @abstractmethod
    def create_aux_wavefields(
        self,
        wavefields: List[torch.Tensor],
        ndim: int,
        is_backward: bool = False,
    ) -> List[torch.Tensor]:
        """Create auxiliary wavefields for odd-step swap.

        Scalar: creates psin (+ zetan if backward).
        Acoustic: creates ndim zero tensors (backward only).
        Elastic: dimension-dependent.
        """
        ...

    @abstractmethod
    def swap_odd_step_wavefields(
        self,
        wavefields: List[torch.Tensor],
        aux_wavefields: List[torch.Tensor],
        ndim: int,
        is_backward: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Swap wavefields after odd number of steps.

        Returns updated (wavefields, aux_wavefields).
        """
        ...

    @abstractmethod
    def zero_backward_wavefields(
        self,
        grad_wavefields: List[torch.Tensor],
        ndim: int,
        accuracy: int,
        pml_width: List[int],
        fd_pad_list: List[int],
    ) -> None:
        """Zero edges and interiors of backward gradient wavefields.

        Equation-specific: scalar zeros PML vars per-dim, acoustic/elastic
        use zero_edges_and_interiors with staggered half-grid masks.
        """
        ...
