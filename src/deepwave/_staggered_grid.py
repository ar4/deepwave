"""Utility functions for staggered grid PML setup.

This module provides functions to set up Perfectly Matched Layers (PML)
profiles for wave propagation simulations on a staggered grid.

Internal module -- no stability guarantee.
"""

import math
from collections.abc import Sequence
from typing import Optional

import torch

from deepwave import _common
from deepwave._common import _DEFAULT_COURANT_NUMBER, _critical_courant_number
from deepwave.types import Accuracy


def elastic_stability_speed(
    stiffness_bound: torch.Tensor,
    buoyancy: torch.Tensor,
    accuracy: Accuracy,
    maximum_speed: float,
    courant_number: Optional[float] = None,
    material_models: Optional[Sequence[torch.Tensor]] = None,
) -> float:
    """Bound the velocity/strain operator including heterogeneous density.

    ``stiffness_bound`` bounds the constitutive matrix in Frobenius strain
    coordinates (the shear eigenvalues are twice the physical shear moduli).
    Arithmetic density staggering and harmonic shear averaging cannot exceed
    the global positive buoyancy/stiffness maxima. Symmetric strain is an
    orthogonal projection of the displacement gradient, so its norm is at
    most the gradient norm. This supplies a conservative operator CFL bound,
    including fluid and zero-stiffness/zero-buoyancy vacuum cells, without
    dividing by the vacuum sentinel.
    """
    courant = _DEFAULT_COURANT_NUMBER if courant_number is None else courant_number
    critical = _critical_courant_number(accuracy, True)
    product = float(stiffness_bound.detach().max()) * float(buoyancy.detach().max())
    operator_speed = courant / critical * math.sqrt(product)
    if operator_speed <= maximum_speed:
        return maximum_speed
    # For constant coefficients the Fourier symbol is the Christoffel
    # matrix at the vector of staggered derivative symbols. The physical
    # directional speed bound already bounds it. The constitutive norm
    # above deliberately discards that directional structure and is only
    # needed when heterogeneous coefficients prevent this reduction.
    if material_models is not None and all(
        torch.equal(model.detach(), model.detach().flatten()[0].expand_as(model))
        for model in material_models
    ):
        return maximum_speed
    return operator_speed


def pressure_stability_speed(
    speed: torch.Tensor,
    density: torch.Tensor,
    accuracy: Accuracy,
    grid_spacing: list[float],
    courant_number: Optional[float] = None,
) -> float:
    """Bound the heterogeneous pressure/velocity operator in speed units.

    ``speed`` is the unrelaxed speed for attenuation. A long derivative
    couples a stiff pressure cell to velocity nodes beyond its immediate
    density neighbours, so the pointwise speed alone need not bound CFL.
    For each signed stencil shift, bound ``sqrt(K / rho_edge)``. The triangle
    inequality bounds each weighted derivative, and the sum of their squared
    norms bounds the multidimensional divergence. Replicated and periodic
    material extensions are both included. No dense operator is allocated.

    The result uses the caller's Courant convention: choosing dt with this
    speed ensures ``dt * ||sqrt(K) D sqrt(b)|| <= 2`` for the interior
    negative-transpose pair. PML stability is qualified separately. A cheap
    global density bound avoids stencil scans when the ordinary CFL margin
    already covers the contrast. This numerical setup is not differentiated.
    """
    speed, density = speed.detach(), density.detach()
    maximum = float(speed.abs().max())
    courant = _DEFAULT_COURANT_NUMBER if courant_number is None else courant_number
    coefficients = {
        2: (1.0,),
        4: (9 / 8, -1 / 24),
        6: (75 / 64, -25 / 384, 3 / 640),
        8: (1225 / 1024, -245 / 3072, 49 / 5120, -5 / 7168),
    }[accuracy]
    minimum_density = float(density.min())
    contrast = (
        float(density.max()) / minimum_density if minimum_density > 0 else float("inf")
    )
    if courant * sum(abs(c) for c in coefficients) * math.sqrt(contrast) <= 1:
        return maximum

    bulk = density * speed.square()
    axis_bounds = []
    for dim, dx in enumerate(grid_spacing):
        axis = density.ndim - len(grid_spacing) + dim
        index = torch.arange(density.shape[axis], device=density.device)
        bound = 0.0
        for k, coefficient in enumerate(coefficients):
            for offset in (k, -k - 1):
                peak = 0.0
                for periodic in (False, True):
                    left, right = index + offset, index + offset + 1
                    if periodic:
                        left, right = left % index.numel(), right % index.numel()
                    else:
                        left = left.clamp(0, index.numel() - 1)
                        right = right.clamp(0, index.numel() - 1)
                    edge_density = (
                        density.index_select(axis, left) * 0.5
                        + density.index_select(axis, right) * 0.5
                    )
                    active = edge_density > torch.finfo(density.dtype).max ** -0.5
                    safe_density = torch.where(active, edge_density, 1.0)
                    ratio = torch.where(active, bulk / safe_density, 0.0)
                    peak = max(peak, float(ratio.max()))
                bound += abs(coefficient) * math.sqrt(peak)
        axis_bounds.append(bound / dx)
    inverse_spacing_norm = math.sqrt(sum(dx**-2 for dx in grid_spacing))
    weighted_bound = math.sqrt(sum(bound**2 for bound in axis_bounds))
    return max(maximum, courant * weighted_bound / (2 * inverse_spacing_norm))


def set_pml_profiles(
    pml_width: list[int],
    accuracy: int,
    fd_pad: list[int],
    dt: float,
    grid_spacing: list[float],
    max_vel: float,
    dtype: torch.dtype,
    device: torch.device,
    pml_freq: float,
    shape: tuple[int, ...],
    mpml_fraction: float = 0.0,
) -> list[torch.Tensor]:
    """Sets up PML profiles for a staggered grid.

    Args:
        pml_width: A list of integers specifying the width of the PML
            on each side (e.g., for 2D: top, bottom, left, right).
        accuracy: The finite-difference accuracy order.
        fd_pad: A list of integers specifying the padding for finite-difference.
        dt: The time step.
        grid_spacing: A list of floats specifying the grid spacing in
            y and x directions.
        max_vel: The maximum velocity in the model.
        dtype: The data type of the tensors (e.g., torch.float32).
        device: The device on which the tensors will be (e.g., 'cuda', 'cpu').
        pml_freq: The PML frequency.
        shape: The number of grid points in each spatial dimension.
        mpml_fraction: The multiaxial (M-PML) cross-damping fraction. When 0 (the
            default) the standard CFS-C-PML per-axis 1-D profiles are returned
            (bit-identical to the historical behaviour). When positive, each
            axis's damping is augmented by ``mpml_fraction`` times the other axes'
            damping (Meza-Fajardo & Papageorgiou, 2008), making the returned
            profiles full-grid (they vary on every spatial axis). Only used by
            the elastic propagator.

    Returns:
        A list of torch.Tensors representing the PML profiles (e.g., in 2D:
        ay, by, ayh, byh, ax, bx, axh, bxh). Per dimension four tensors are
        appended in the order ``[a, b, ah, bh]``. When ``mpml_fraction > 0`` each
        tensor spans the full grid instead of a single axis.

    """
    if mpml_fraction > 0:
        return _set_pml_profiles_mpml(
            pml_width,
            fd_pad,
            dt,
            grid_spacing,
            max_vel,
            dtype,
            device,
            pml_freq,
            shape,
            mpml_fraction,
        )
    ndim = len(shape)
    pml_start: list[list[int]] = [
        [
            fd_pad[dim * 2] + pml_width[dim * 2],
            shape[dim] - 1 - fd_pad[dim * 2 + 1] - pml_width[dim * 2 + 1],
        ]
        for dim in range(ndim)
    ]

    pml_profiles: list[torch.Tensor] = []
    for dim in range(ndim):
        a, b = _common.setup_pml(
            pml_width[2 * dim : 2 * dim + 2],
            [float(v) for v in pml_start[dim]],
            grid_spacing[dim],
            dt,
            shape[dim],
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.0,
        )
        ah, bh = _common.setup_pml(
            pml_width[2 * dim : 2 * dim + 2],
            [float(v) for v in pml_start[dim]],
            grid_spacing[dim],
            dt,
            shape[dim],
            max_vel,
            dtype,
            device,
            pml_freq,
            start=0.5,
        )
        s_list: list[Optional[slice]] = [None] * (ndim + 1)
        s_list[1 + dim] = slice(None)
        s: tuple[Optional[slice], ...] = tuple(s_list)
        pml_profiles.extend([a[tuple(s)], b[tuple(s)], ah[tuple(s)], bh[tuple(s)]])
    return pml_profiles


def _mpml_reshape_axis(vec: torch.Tensor, axis: int, ndim: int) -> torch.Tensor:
    """Reshape a 1-D per-axis profile so it broadcasts on spatial ``axis``.

    The returned tensor has ``ndim + 1`` dims (a leading shot-broadcast axis of
    size 1 followed by the spatial axes), matching the shape convention used by
    the standard (non-M-PML) profiles and by the eager/C backends.
    """
    profile_shape = [1] * (ndim + 1)
    profile_shape[1 + axis] = vec.numel()
    return vec.reshape(profile_shape)


def _mpml_combine(
    sigma_self: torch.Tensor,
    sigma_others_scaled: list[torch.Tensor],
    alpha_self: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine cross-axis damping and apply the C-PML ``a``/``b`` transform.

    ``sigma_self`` is the axis's own (broadcast) raw damping; each entry of
    ``sigma_others_scaled`` is ``mpml_fraction`` times another axis's (broadcast) raw
    damping. ``alpha_self`` is the axis's own CFS frequency-shift term. The
    result is materialised to the full grid (all spatial axes) so the backends
    receive genuine full-grid arrays.
    """
    sigma_eff = sigma_self
    for other in sigma_others_scaled:
        sigma_eff = sigma_eff + other
    sigmaalpha = sigma_eff + alpha_self
    a = torch.exp(-sigmaalpha * abs(dt))
    safe = sigmaalpha.clamp(min=1e-30)
    b = sigma_eff / safe * (a - 1)
    # Match setup_pml's ``a[pml_frac == 0] = 0``: where there is no damping at
    # all the memory variable stays zero, so this is behaviourally inert but
    # keeps the p -> 0 limit consistent with the standard profiles.
    a = torch.where(sigma_eff == 0, torch.zeros_like(a), a)
    # Broadcast both to the common full grid so they materialise as real
    # full-grid tensors (the kernels index them on every axis).
    a = a + torch.zeros_like(b)
    b = b + torch.zeros_like(a)
    return a, b


def _set_pml_profiles_mpml(
    pml_width: list[int],
    fd_pad: list[int],
    dt: float,
    grid_spacing: list[float],
    max_vel: float,
    dtype: torch.dtype,
    device: torch.device,
    pml_freq: float,
    shape: tuple[int, ...],
    mpml_fraction: float,
) -> list[torch.Tensor]:
    """Build multiaxial (M-PML) full-grid profiles (cross-damping ``mpml_fraction``).

    Reconstructs each axis's raw ``sigma``/``alpha`` (exactly as
    :func:`deepwave._common.setup_pml` does internally), augments each axis's
    damping with ``mpml_fraction`` times the other axes' damping, and recomputes the
    ``a``/``b`` C-PML coefficients from the combined damping. Because the cross
    term makes each axis's coefficients depend on every coordinate, the returned
    tensors span the full grid. At ``mpml_fraction == 0`` this reduces exactly to the
    standard profiles (which is why the caller takes the untouched fast path
    instead of calling this function).
    """
    ndim = len(shape)
    pml_start: list[list[float]] = [
        [
            float(fd_pad[dim * 2] + pml_width[dim * 2]),
            float(shape[dim] - 1 - fd_pad[dim * 2 + 1] - pml_width[dim * 2 + 1]),
        ]
        for dim in range(ndim)
    ]

    # Reconstruct the raw 1-D sigma/alpha per axis for the integer (start=0.0)
    # and half (start=0.5) staggered grids.
    sigma: dict[float, list[torch.Tensor]] = {0.0: [], 0.5: []}
    alpha: dict[float, list[torch.Tensor]] = {0.0: [], 0.5: []}
    for start in (0.0, 0.5):
        for dim in range(ndim):
            s, al = _common.setup_pml_sigma_alpha(
                pml_width[2 * dim : 2 * dim + 2],
                pml_start[dim],
                grid_spacing[dim],
                shape[dim],
                max_vel,
                dtype,
                device,
                pml_freq,
                start=start,
            )
            sigma[start].append(s)
            alpha[start].append(al)

    # The cross-axis terms sample the OTHER axes' damping at integer positions
    # for both the integer and half-grid profiles. A profile is staggered only
    # along its own axis, and integer sampling keeps every profile's nonzero
    # support within the other axes' integer-grid PML strips, which is what the
    # compiled kernels' union-of-strips fast-path region test assumes (a
    # half-sampled cross term would extend one cell further into the interior
    # on the end side). The half-cell difference in the auxiliary
    # cross-damping strength has no physical significance (M-PML is an
    # empirically tuned stabiliser).
    others_scaled = [
        mpml_fraction * _mpml_reshape_axis(sigma[0.0][dim], dim, ndim)
        for dim in range(ndim)
    ]
    pml_profiles: list[torch.Tensor] = []
    for dim in range(ndim):
        others = [others_scaled[k] for k in range(ndim) if k != dim]

        # Integer grid (a, b)
        sig_self = _mpml_reshape_axis(sigma[0.0][dim], dim, ndim)
        al_self = _mpml_reshape_axis(alpha[0.0][dim], dim, ndim)
        a, b = _mpml_combine(sig_self, others, al_self, dt)

        # Half grid (ah, bh)
        sig_self_h = _mpml_reshape_axis(sigma[0.5][dim], dim, ndim)
        al_self_h = _mpml_reshape_axis(alpha[0.5][dim], dim, ndim)
        ah, bh = _mpml_combine(sig_self_h, others, al_self_h, dt)

        pml_profiles.extend([a, b, ah, bh])
    return pml_profiles


def diffz1(a: torch.Tensor, accuracy: int, rdz: float) -> torch.Tensor:
    """Calculates the first z derivative at integer grid points."""
    if accuracy == 2:
        return torch.nn.functional.pad(
            (a[..., 1:, :, :] - a[..., :-1, :, :]) * rdz, (0, 0, 0, 0, 1, 0)
        )
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 2:-1, :, :] - a[..., 1:-2, :, :])
                + -1 / 24 * (a[..., 3:, :, :] - a[..., :-3, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 2, 1),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 3:-2, :, :] - a[..., 2:-3, :, :])
                + -25 / 384 * (a[..., 4:-1, :, :] - a[..., 1:-4, :, :])
                + 3 / 640 * (a[..., 5:, :, :] - a[..., :-5, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 3, 2),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 4:-3, :, :] - a[..., 3:-4, :, :])
            + -245 / 3072 * (a[..., 5:-2, :, :] - a[..., 2:-5, :, :])
            + 49 / 5120 * (a[..., 6:-1, :, :] - a[..., 1:-6, :, :])
            + -5 / 7168 * (a[..., 7:, :, :] - a[..., :-7, :, :])
        )
        * rdz,
        (0, 0, 0, 0, 4, 3),
    )


def diffy1(a: torch.Tensor, accuracy: int, rdy: float) -> torch.Tensor:
    """Calculates the first y derivative at integer grid points."""
    if accuracy == 2:
        return torch.nn.functional.pad(
            (a[..., 1:, :] - a[..., :-1, :]) * rdy, (0, 0, 1, 0)
        )
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 2:-1, :] - a[..., 1:-2, :])
                + -1 / 24 * (a[..., 3:, :] - a[..., :-3, :])
            )
            * rdy,
            (0, 0, 2, 1),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 3:-2, :] - a[..., 2:-3, :])
                + -25 / 384 * (a[..., 4:-1, :] - a[..., 1:-4, :])
                + 3 / 640 * (a[..., 5:, :] - a[..., :-5, :])
            )
            * rdy,
            (0, 0, 3, 2),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 4:-3, :] - a[..., 3:-4, :])
            + -245 / 3072 * (a[..., 5:-2, :] - a[..., 2:-5, :])
            + 49 / 5120 * (a[..., 6:-1, :] - a[..., 1:-6, :])
            + -5 / 7168 * (a[..., 7:, :] - a[..., :-7, :])
        )
        * rdy,
        (0, 0, 4, 3),
    )


def diffx1(a: torch.Tensor, accuracy: int, rdx: float) -> torch.Tensor:
    """Calculates the first x derivative at integer grid points."""
    if accuracy == 2:
        return torch.nn.functional.pad((a[..., 1:] - a[..., :-1]) * rdx, (1, 0))
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 2:-1] - a[..., 1:-2])
                + -1 / 24 * (a[..., 3:] - a[..., :-3])
            )
            * rdx,
            (2, 1),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 3:-2] - a[..., 2:-3])
                + -25 / 384 * (a[..., 4:-1] - a[..., 1:-4])
                + 3 / 640 * (a[..., 5:] - a[..., :-5])
            )
            * rdx,
            (3, 2),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 4:-3] - a[..., 3:-4])
            + -245 / 3072 * (a[..., 5:-2] - a[..., 2:-5])
            + 49 / 5120 * (a[..., 6:-1] - a[..., 1:-6])
            + -5 / 7168 * (a[..., 7:] - a[..., :-7])
        )
        * rdx,
        (4, 3),
    )


def diffzh1(a: torch.Tensor, accuracy: int, rdz: float) -> torch.Tensor:
    """Calculates the first z derivative at half integer grid points."""
    if accuracy == 2:
        return torch.nn.functional.pad(
            (a[..., 2:, :, :] - a[..., 1:-1, :, :]) * rdz, (0, 0, 0, 0, 1, 1)
        )
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 3:-1, :, :] - a[..., 2:-2, :, :])
                + -1 / 24 * (a[..., 4:, :, :] - a[..., 1:-3, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 2, 2),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 4:-2, :, :] - a[..., 3:-3, :, :])
                + -25 / 384 * (a[..., 5:-1, :, :] - a[..., 2:-4, :, :])
                + 3 / 640 * (a[..., 6:, :, :] - a[..., 1:-5, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 3, 3),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 5:-3, :, :] - a[..., 4:-4, :, :])
            + -245 / 3072 * (a[..., 6:-2, :, :] - a[..., 3:-5, :, :])
            + 49 / 5120 * (a[..., 7:-1, :, :] - a[..., 2:-6, :, :])
            + -5 / 7168 * (a[..., 8:, :, :] - a[..., 1:-7, :, :])
        )
        * rdz,
        (0, 0, 0, 0, 4, 4),
    )


def diffyh1(a: torch.Tensor, accuracy: int, rdy: float) -> torch.Tensor:
    """Calculates the first y derivative at half integer grid points."""
    if accuracy == 2:
        return torch.nn.functional.pad(
            (a[..., 2:, :] - a[..., 1:-1, :]) * rdy, (0, 0, 1, 1)
        )
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 3:-1, :] - a[..., 2:-2, :])
                + -1 / 24 * (a[..., 4:, :] - a[..., 1:-3, :])
            )
            * rdy,
            (0, 0, 2, 2),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 4:-2, :] - a[..., 3:-3, :])
                + -25 / 384 * (a[..., 5:-1, :] - a[..., 2:-4, :])
                + 3 / 640 * (a[..., 6:, :] - a[..., 1:-5, :])
            )
            * rdy,
            (0, 0, 3, 3),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 5:-3, :] - a[..., 4:-4, :])
            + -245 / 3072 * (a[..., 6:-2, :] - a[..., 3:-5, :])
            + 49 / 5120 * (a[..., 7:-1, :] - a[..., 2:-6, :])
            + -5 / 7168 * (a[..., 8:, :] - a[..., 1:-7, :])
        )
        * rdy,
        (0, 0, 4, 4),
    )


def diffxh1(a: torch.Tensor, accuracy: int, rdx: float) -> torch.Tensor:
    """Calculates the first x derivative at half integer grid points."""
    if accuracy == 2:
        return torch.nn.functional.pad((a[..., 2:] - a[..., 1:-1]) * rdx, (1, 1))
    if accuracy == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 3:-1] - a[..., 2:-2])
                + -1 / 24 * (a[..., 4:] - a[..., 1:-3])
            )
            * rdx,
            (2, 2),
        )
    if accuracy == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 4:-2] - a[..., 3:-3])
                + -25 / 384 * (a[..., 5:-1] - a[..., 2:-4])
                + 3 / 640 * (a[..., 6:] - a[..., 1:-5])
            )
            * rdx,
            (3, 3),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 5:-3] - a[..., 4:-4])
            + -245 / 3072 * (a[..., 6:-2] - a[..., 3:-5])
            + 49 / 5120 * (a[..., 7:-1] - a[..., 2:-6])
            + -5 / 7168 * (a[..., 8:] - a[..., 1:-7])
        )
        * rdx,
        (4, 4),
    )


def diff1(
    wf: torch.Tensor,
    dim: int,
    accuracy: int,
    rdx: float,
    ndim: int,
) -> torch.Tensor:
    """Perform first-order differentiation on a staggered grid."""
    if ndim - dim == 3:
        return diffz1(wf, accuracy, rdx)
    if ndim - dim == 2:
        return diffy1(wf, accuracy, rdx)
    return diffx1(wf, accuracy, rdx)


def diff1h(
    wf: torch.Tensor,
    dim: int,
    accuracy: int,
    rdx: float,
    ndim: int,
) -> torch.Tensor:
    """Perform first-order differentiation on a staggered grid (half-step)."""
    if ndim - dim == 3:
        return diffzh1(wf, accuracy, rdx)
    if ndim - dim == 2:
        return diffyh1(wf, accuracy, rdx)
    return diffxh1(wf, accuracy, rdx)


def zero_edge(tensor: torch.Tensor, fd_pad: int, dim: int) -> torch.Tensor:
    """Sets values at the end of a spatial dimension of a tensor to zero.

    This is done because the staggered grid means that elements at index
    ``-fd_pad`` in the dimension in which a component is shifted by half a grid
    cell are not considered to be part of the computational domain and so
    should be zeroed.

    Args:
        tensor: The input torch.Tensor to modify.
        fd_pad: Half the length of the spatial finite difference stencil.
        dim: The spatial dimension in which to zero.

    Returns:
        The modified tensor.
    """
    tensor[(slice(None),) + (slice(None),) * dim + (-fd_pad,)].fill_(0)
    return tensor


def zero_edges_and_interiors(
    wavefields: list[torch.Tensor],
    ndim: int,
    fd_pad: int,
    fd_pad_list: list[int],
    pml_width: list[int],
    half_grid: list[list[bool]],
    pml_edge_idx: list[list[int]],
    interior: bool = True,
    mpml: bool = False,
) -> None:
    """Zero the staggered-grid edges and/or PML memory-variable interiors.

    This is the shared implementation used by all the staggered-grid
    propagators (acoustic, viscoacoustic, elastic, elastic VTI). Each
    propagator supplies its own two per-wavefield tables and this function
    performs the identical zeroing pass, modifying ``wavefields`` in place.

    Args:
        wavefields: The list of wavefield tensors to modify in place.
        ndim: The number of spatial dimensions.
        fd_pad: The (scalar) finite-difference padding.
        fd_pad_list: The per-side finite-difference padding
            (length ``2 * ndim``).
        pml_width: The per-side PML widths (length ``2 * ndim``).
        half_grid: ``half_grid[i][dim]`` is ``True`` when wavefield ``i`` is
            shifted by half a grid cell along dimension ``dim`` (its outermost
            edge along that dimension must be zeroed).
        pml_edge_idx: ``pml_edge_idx[dim]`` lists the wavefield indices of the
            PML memory variables whose interior must be zeroed along dimension
            ``dim``.
        interior: If ``True``, also zero the interiors of the PML memory
            variables.
        mpml: If ``True`` (multiaxial PML), a PML memory variable is
            legitimately nonzero anywhere *any* axis is inside its PML (the
            cross-axis damping), so only the pure-interior box (interior along
            every axis) is zeroed, rather than the standard per-axis interior
            slabs.
    """
    num_vars = len(wavefields)
    for i in range(num_vars):
        for dim in range(ndim):
            if half_grid[i][dim]:
                wavefields[i] = zero_edge(wavefields[i], fd_pad, dim)

    if not interior:
        return

    if mpml:
        # Zero only the box that is interior along every axis; the per-axis
        # slabs would destroy legitimate cross-damping memory-variable state
        # inside the other axes' PML strips (e.g. on initial_state
        # continuation runs).
        for i in sorted({idx for idx_list in pml_edge_idx for idx in idx_list}):
            fd_pad_shifted = list(fd_pad_list)
            for dim in range(ndim):
                if half_grid[i][dim]:
                    fd_pad_shifted[2 * dim + 1] += 1
            wavefields[i] = _common.zero_interior_box(
                wavefields[i], fd_pad_shifted, pml_width
            )
    else:
        for dim in range(ndim):
            for i in pml_edge_idx[dim]:
                fd_pad_shifted = list(fd_pad_list)
                if half_grid[i][dim]:
                    fd_pad_shifted[2 * dim + 1] += 1
                wavefields[i] = _common.zero_interior(
                    wavefields[i], fd_pad_shifted, pml_width, dim
                )


def stagger_density_arithmetic_mean(
    density: torch.Tensor, ndim: int, *, invert_first: bool = False
) -> list[torch.Tensor]:
    """Arithmetic-mean density staggering to half-integer (velocity) nodes.

    Arithmetic mean: ``rho[i+1/2] = (rho[i] + rho[i+1]) / 2``, so
    ``buoyancy[i+1/2] = 2 / (rho[i] + rho[i+1])``. Division by (near-)zero
    density is guarded so vacuum cells map to exactly zero buoyancy rather
    than inf/nan. Shared by acoustic/viscoacoustic (whose public API takes
    ``rho`` directly) and elastic (whose public API takes ``buoyancy``
    already, hence ``invert_first``).

    Args:
        density: Density model. If ``invert_first`` is ``False`` this is
            ``rho``; if ``True`` this is ``buoyancy`` (``1 / rho``) and is
            inverted back to ``rho`` first, with the same zero-guarded
            division as the staggering itself.
        ndim: Number of spatial dimensions (1, 2, or 3).
        invert_first: Whether ``density`` is a buoyancy needing inversion
            (elastic) rather than a density (acoustic/viscoacoustic).

    Returns:
        ``[buoyancy_z?, buoyancy_y?, buoyancy_x]`` (``buoyancy_z``/
        ``buoyancy_y`` present only for ``ndim >= 3``/``ndim >= 2``
        respectively), at half-integer nodes along each axis.
    """
    rfmax = 1 / torch.finfo(density.dtype).max ** (1 / 2)
    if invert_first:
        buoyancy = density
        mask = rfmax < buoyancy.abs()
        buoyancy_safe = torch.where(mask, buoyancy, torch.ones_like(buoyancy))
        rho = torch.where(mask, 1 / buoyancy_safe, torch.zeros_like(buoyancy))
        del mask, buoyancy_safe
    else:
        rho = density

    result = []
    # Halve before adding: finite densities can have an unrepresentable sum
    # even though their arithmetic mean and its reciprocal are representable.
    if ndim >= 3:
        rho_z = torch.nn.functional.pad(
            rho[..., :-1, :, :] * 0.5 + rho[..., 1:, :, :] * 0.5,
            (0, 0, 0, 0, 0, 1),
        )
        mask = rfmax < rho_z.abs()
        rho_z_safe = torch.where(mask, rho_z, torch.ones_like(rho_z))
        buoyancy_z = torch.where(mask, 1 / rho_z_safe, torch.zeros_like(rho))
        result.append(buoyancy_z)
        del rho_z, mask, rho_z_safe, buoyancy_z

    if ndim >= 2:
        rho_y = torch.nn.functional.pad(
            rho[..., :-1, :] * 0.5 + rho[..., 1:, :] * 0.5, (0, 0, 0, 1)
        )
        mask = rfmax < rho_y.abs()
        rho_y_safe = torch.where(mask, rho_y, torch.ones_like(rho_y))
        buoyancy_y = torch.where(mask, 1 / rho_y_safe, torch.zeros_like(rho))
        result.append(buoyancy_y)
        del rho_y, mask, rho_y_safe, buoyancy_y

    rho_x = torch.nn.functional.pad(rho[..., :-1] * 0.5 + rho[..., 1:] * 0.5, (0, 1))
    mask = rfmax < rho_x.abs()
    rho_x_safe = torch.where(mask, rho_x, torch.ones_like(rho_x))
    buoyancy_x = torch.where(mask, 1 / rho_x_safe, torch.zeros_like(rho))
    result.append(buoyancy_x)

    return result
