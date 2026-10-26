"""Per-propagator boundary-face fill machinery for Deepwave propagators.

This module holds the guard-ring fill functions that implement the
``boundary=`` free/rigid/periodic faces for the regular-grid (scalar,
scalar_born) and staggered-grid (acoustic) propagators, plus the model-
coefficient mirror-pad helpers the compiled adjoints use
(``mirror_pad_acoustic_models``, ``mirror_pad_scalar_born_models``). The
integer boundary codes (``BC_*``) and the ``boundary`` validation live in
:mod:`deepwave._common`; these functions consume those codes to build the
ghost slabs each backend reads.
"""

import torch

from deepwave._common import BC_FREE, BC_PERIODIC, BC_PML, BC_RIGID


def _fill_boundary_face(
    interior: torch.Tensor, dim: int, pad: int, code: int, is_low: bool
) -> torch.Tensor:
    """Build a ``pad``-wide ghost slab for one nodal (scalar) face.

    ``interior`` is the guard-free interior slab along ``dim`` (length ``m``).
    The returned slab is the image of the interior for face type ``code``
    (see ``BC_*``); the mirror-plane conventions are documented in
    ``docs/boundaries.rst`` (free: odd about the outer node; rigid: even about
    the half-cell plane; periodic: wrap with period ``m``).
    """
    m = interior.size(dim)
    if code == 3:  # periodic
        if is_low:
            return interior.narrow(dim, m - pad, pad)
        return interior.narrow(dim, 0, pad)
    if code == 2:  # rigid: even about the half-cell plane
        if is_low:
            return torch.flip(interior.narrow(dim, 0, pad), [dim])
        return torch.flip(interior.narrow(dim, m - pad, pad), [dim])
    # free: odd about the outer node (that node sits on the plane -> zero)
    zero = interior.narrow(dim, 0, 1) * 0
    if pad == 1:
        return zero
    if is_low:
        return torch.cat(
            [-torch.flip(interior.narrow(dim, 0, pad - 1), [dim]), zero], dim
        )
    return torch.cat(
        [zero, -torch.flip(interior.narrow(dim, m - pad + 1, pad - 1), [dim])], dim
    )


def fill_scalar_boundary(
    wf: torch.Tensor, boundary: list[int], fd_pad: int
) -> torch.Tensor:
    """Fill the guard ring of a scalar (nodal) wavefield per-face.

    ``wf`` has shape ``(n_shots, *spatial)`` with ``fd_pad`` guard cells on each
    side of every spatial axis. Faces whose code is ``BC_PML`` (0) are left
    untouched (so a ``None``/all-PML boundary is a no-op, preserving the exact
    previous behaviour). Returns a new tensor; the interior is unchanged.
    """
    ndim = wf.dim() - 1
    p = fd_pad
    for d in range(ndim):
        lo = boundary[2 * d]
        hi = boundary[2 * d + 1]
        if lo == 0 and hi == 0:
            continue
        axis = 1 + d
        n = wf.size(axis)
        m = n - 2 * p
        interior = wf.narrow(axis, p, m)
        low = (
            wf.narrow(axis, 0, p)
            if lo == 0
            else _fill_boundary_face(interior, axis, p, lo, True)
        )
        high = (
            wf.narrow(axis, n - p, p)
            if hi == 0
            else _fill_boundary_face(interior, axis, p, hi, False)
        )
        wf = torch.cat([low, interior, high], axis)
    return wf


def _acoustic_p_face(
    interior: torch.Tensor, ax: int, pad: int, code: int, is_low: bool
) -> torch.Tensor:
    """Build the ghost slab for the pressure field on one staggered face.

    ``interior`` is the guard-free pressure slab along ``ax`` (the ``m`` nodes
    ``[P, n-P]`` of the padded grid, ``m = n - 2P + 1``). Returns the image slab
    for face type ``code`` (see ``BC_*``): width ``pad`` on the low face, ``pad -
    1`` on the high face (the staggered low/high pad asymmetry). Conventions
    (docs/boundaries.rst): ``free`` = pressure odd about the surface pressure
    node; ``rigid`` = pressure even about the half-cell wall plane; ``periodic``
    = wrap with period ``m``.
    """
    m = interior.size(ax)
    if is_low:
        if code == 3:  # periodic: wrap with period m-1 (staggered-cell count)
            return interior.narrow(ax, m - 1 - pad, pad)
        if code == 2:  # rigid: even about the half-cell plane P-1/2
            return torch.flip(interior.narrow(ax, 0, pad), [ax])
        # free: odd about the surface node P (guard mirrors nodes P+1..2P)
        return -torch.flip(interior.narrow(ax, 1, pad), [ax])
    w = pad - 1
    if code == 3:  # periodic
        return interior.narrow(ax, 1, w)
    if code == 2:  # rigid: even about n-P+1/2
        return torch.flip(interior.narrow(ax, m - w, w), [ax])
    # free: odd about the surface node n-P
    return -torch.flip(interior.narrow(ax, m - pad, w), [ax])


def _acoustic_v_face(
    interior: torch.Tensor, ax: int, pad: int, code: int, is_low: bool
) -> torch.Tensor:
    """Build the ghost slab for a boundary-normal velocity on one staggered face.

    ``interior`` is the velocity slab along ``ax`` (nodes ``[P, n-P]``). Returns
    the image slab (width ``pad`` low, ``pad - 1`` high). Conventions: ``free`` =
    normal velocity even about the surface pressure node; ``rigid`` = normal
    velocity odd about the wall (its on-plane node is zero); ``periodic`` = wrap.
    The rigid low face's on-plane node is the outermost guard cell (zeroed here);
    the rigid high face's on-plane node is an interior node zeroed by the caller.
    """
    m = interior.size(ax)
    if is_low:
        if code == 3:  # periodic: wrap with period m-1
            return interior.narrow(ax, m - 1 - pad, pad)
        if code == 1:  # free: even about the surface node P
            return torch.flip(interior.narrow(ax, 0, pad), [ax])
        # rigid: odd about P-1/2; the on-plane guard node (outermost) is zero
        body = -torch.flip(interior.narrow(ax, 0, pad - 1), [ax])
        zero = interior.narrow(ax, 0, 1) * 0
        return torch.cat([body, zero], ax)
    w = pad - 1
    if code == 3:  # periodic
        return interior.narrow(ax, 1, w)
    if code == 1:  # free: even about the surface node n-P
        return torch.flip(interior.narrow(ax, m - pad - 1, w), [ax])
    # rigid: odd about the wall node n-P (that interior node is zeroed by caller)
    return -torch.flip(interior.narrow(ax, m - pad, w), [ax])


def fill_acoustic_pressure_boundary(
    p: torch.Tensor, boundary: list[int], fd_pad: int
) -> torch.Tensor:
    """Fill the pressure guard ring of the staggered acoustic grid per-face.

    ``p`` has shape ``(n_shots, *spatial)`` with ``fd_pad`` low / ``fd_pad - 1``
    high guard cells per axis. ``BC_PML`` faces are untouched (so an all-PML
    ``boundary`` is a no-op, preserving bit-identical behaviour). On a ``free``
    face the surface pressure node itself is held to zero. Returns a new tensor.
    """
    ndim = p.dim() - 1
    pad = fd_pad
    for d in range(ndim):
        lo = boundary[2 * d]
        hi = boundary[2 * d + 1]
        if lo == 0 and hi == 0:
            continue
        ax = 1 + d
        n = p.size(ax)
        m = n - 2 * pad + 1
        interior = p.narrow(ax, pad, m)
        # A free surface holds the on-surface pressure node itself to zero.
        if lo == 1:
            interior = torch.cat(
                [interior.narrow(ax, 0, 1) * 0, interior.narrow(ax, 1, m - 1)], ax
            )
        if hi == 1:
            interior = torch.cat(
                [interior.narrow(ax, 0, m - 1), interior.narrow(ax, m - 1, 1) * 0], ax
            )
        low = (
            p.narrow(ax, 0, pad)
            if lo == 0
            else _acoustic_p_face(interior, ax, pad, lo, True)
        )
        high = (
            p.narrow(ax, n - (pad - 1), pad - 1)
            if hi == 0
            else _acoustic_p_face(interior, ax, pad, hi, False)
        )
        p = torch.cat([low, interior, high], ax)
    return p


def fill_acoustic_velocity_boundary(
    velocities: list[torch.Tensor], boundary: list[int], fd_pad: int
) -> list[torch.Tensor]:
    """Fill the boundary-normal velocity guard ring per-face (staggered grid).

    Only ``velocities[d]`` (the dimension-``d`` velocity) is imaged along axis
    ``d``; the tangential velocities and corner guard cells are never read by an
    interior update (see docs/boundaries.rst). On a ``rigid`` high face the
    on-wall velocity node is an *interior* node (the staggered low/high pad
    asymmetry) and is held to zero here. On a ``periodic`` face the extra high
    interior velocity node is slaved to its wrap image. Returns the list.
    """
    pad = fd_pad
    ndim = len(velocities)
    for d in range(ndim):
        lo = boundary[2 * d]
        hi = boundary[2 * d + 1]
        if lo == 0 and hi == 0:
            continue
        ax = 1 + d
        v = velocities[d]
        n = v.size(ax)
        m = n - 2 * pad + 1
        interior = v.narrow(ax, pad, m)
        # The rigid high-face wall node is the last interior velocity node.
        if hi == 2:
            interior = torch.cat(
                [interior.narrow(ax, 0, m - 1), interior.narrow(ax, m - 1, 1) * 0], ax
            )
        # Periodic: the extra high interior velocity node wraps to node P.
        elif hi == 3:
            interior = torch.cat(
                [interior.narrow(ax, 0, m - 1), interior.narrow(ax, 0, 1)], ax
            )
        low = (
            v.narrow(ax, 0, pad)
            if lo == 0
            else _acoustic_v_face(interior, ax, pad, lo, True)
        )
        high = (
            v.narrow(ax, n - (pad - 1), pad - 1)
            if hi == 0
            else _acoustic_v_face(interior, ax, pad, hi, False)
        )
        velocities[d] = torch.cat([low, interior, high], ax)
    return velocities


def _model_face_image(
    interior: torch.Tensor, ax: int, pad: int, code: int, is_low: bool, plane: str
) -> torch.Tensor:
    """Even/wrap guard image of a *model* coefficient on one staggered face.

    The compiled acoustic adjoint reads the model coefficients at neighbour
    offsets near a reflecting face, so their guard ring must hold the model's
    reflected image. Unlike the *field* fill this is a pure even (or periodic
    wrap) extension -- no sign flip and no on-plane zeroing -- about the same
    plane the field reflects about. ``plane`` is ``"node"`` (reflection about
    the on-grid surface node) or ``"half"`` (about the half-cell wall), matching
    the field's convention for this grid/face. ``interior`` is the guard-free
    slab (``m = n - 2*pad + 1`` nodes). Returns the guard slab: width ``pad``
    (low) or ``pad - 1`` (high). The on-plane guard cell of the ``"node"`` case
    is left as the interior edge value (there is no interior model there).
    """
    m = interior.size(ax)
    w = pad if is_low else pad - 1
    if code == BC_PERIODIC:
        if is_low:
            return interior.narrow(ax, m - 1 - pad, pad)
        return interior.narrow(ax, 1, w)
    if plane == "half":
        # Even about the half-cell wall: same source indices as the field's
        # rigid (pressure) / free (velocity) image, without the sign flip.
        if is_low:
            return torch.flip(interior.narrow(ax, 0, pad), [ax])
        return torch.flip(interior.narrow(ax, m - w, w), [ax])
    # Even about the on-grid surface node (drop the field's odd sign flip). The
    # on-plane guard cell has no interior source, so it keeps its edge value.
    if is_low:
        return torch.flip(interior.narrow(ax, 1, pad), [ax])
    return torch.flip(interior.narrow(ax, m - pad, w), [ax])


def _mirror_pad_model_axis(
    model: torch.Tensor, ax: int, pad: int, lo: int, hi: int, grid: str
) -> torch.Tensor:
    """Even/wrap mirror-pad one model's guard ring along a single axis.

    ``grid`` is ``"pressure"`` (K) or ``"velocity"`` (a buoyancy), selecting the
    reflection plane per face: pressure reflects free about the surface node and
    rigid about the half-cell; velocity is the opposite (free about the
    half-cell, rigid about the surface node) -- matching the field fills.
    """
    if lo == BC_PML and hi == BC_PML:
        return model
    n = model.size(ax)
    m = n - 2 * pad + 1
    interior = model.narrow(ax, pad, m)
    # Reflection plane per face type and grid.
    if grid == "pressure":
        plane = {BC_FREE: "node", BC_RIGID: "half", BC_PERIODIC: "wrap"}
    else:
        plane = {BC_FREE: "half", BC_RIGID: "node", BC_PERIODIC: "wrap"}
    low = (
        model.narrow(ax, 0, pad)
        if lo == BC_PML
        else _model_face_image(interior, ax, pad, lo, True, plane.get(lo, "half"))
    )
    high = (
        model.narrow(ax, n - (pad - 1), pad - 1)
        if hi == BC_PML
        else _model_face_image(interior, ax, pad, hi, False, plane.get(hi, "half"))
    )
    return torch.cat([low, interior, high], ax)


def mirror_pad_acoustic_models(
    models: list[torch.Tensor], boundary: list[int], fd_pad: int
) -> list[torch.Tensor]:
    """Return fresh model clones with reflecting-face guard rings mirror-padded.

    ``models`` is ``[K, buoyancy_0, ..., buoyancy_{ndim-1}]`` (staggered acoustic
    grid). The compiled adjoint reads ``K`` (pressure grid) at offsets along
    *every* axis and each ``buoyancy_d`` (its velocity grid) at offsets along
    *its own* axis ``d`` only, so those guard rings must hold the model's even /
    periodic-wrap image. An all-``BC_PML`` boundary is a no-op (the inputs are
    returned unchanged). Fresh clones are returned so the (possibly aliased)
    saved user model tensors are never written in place.
    """
    if all(code == BC_PML for code in boundary):
        return models
    ndim = len(models) - 1
    out = list(models)
    # K lives on the pressure grid and is imaged along every reflecting axis.
    k = out[0].clone()
    for d in range(ndim):
        lo, hi = boundary[2 * d], boundary[2 * d + 1]
        k = _mirror_pad_model_axis(k, 1 + d, fd_pad, lo, hi, "pressure")
    out[0] = k
    # Each buoyancy lives on its own velocity grid, imaged along its own axis.
    for d in range(ndim):
        lo, hi = boundary[2 * d], boundary[2 * d + 1]
        if lo == BC_PML and hi == BC_PML:
            continue
        out[1 + d] = _mirror_pad_model_axis(
            out[1 + d].clone(), 1 + d, fd_pad, lo, hi, "velocity"
        )
    # This runs inside the adjoint ``Function.forward`` (grad tracking disabled),
    # where ``clone``/``cat`` drop ``requires_grad``; the compiled backend sizes
    # each model's gradient buffer from that flag, so restore it to match the
    # originals (a fresh, non-aliasing tensor is still returned).
    for i, m in enumerate(models):
        if out[i] is not m and m.requires_grad and not out[i].requires_grad:
            out[i] = out[i].detach().requires_grad_(True)
    return out


def mirror_pad_viscoacoustic_models(
    models: list[torch.Tensor], boundary: list[int], fd_pad: int
) -> list[torch.Tensor]:
    """Return fresh viscoacoustic model clones with guard rings mirror-padded.

    ``models`` is ``[m1, c2, ar, buoyancy_0, ..., buoyancy_{ndim-1}]``. The
    compiled adjoint's ``B_EFF`` combination (see viscoacoustic_kernels.h)
    reads ``m1``/``c2`` at neighbour offsets along *every* axis -- exactly like
    acoustic's ``K`` -- because the discrete adjoint of the coupled
    pressure/memory-variable update distributes both the pressure-adjoint and
    the memory-variable-adjoint contributions through the same stencil that
    reads the pressure-grid models; each ``buoyancy_d`` is read at neighbour
    offsets along its own axis only, exactly like acoustic's ``buoyancy_d``.
    ``ar`` is read only at the centre point (never at a neighbour offset --
    ``AR(...)`` was dropped from viscoacoustic_kernels.h as dead code), so it
    needs no mirror-pad. Mirrors :func:`mirror_pad_acoustic_models` exactly,
    generalised to viscoacoustic's three pressure-grid models (``m1``, ``c2``
    share ``K``'s role; ``ar`` has none). An all-``BC_PML`` boundary is a no-op
    (the inputs are returned unchanged). Fresh clones are returned so the
    (possibly aliased) saved user model tensors are never written in place.
    """
    if all(code == BC_PML for code in boundary):
        return models
    ndim = len(models) - 3
    out = list(models)
    # m1, c2 live on the pressure grid and are imaged along every reflecting
    # axis (like acoustic's K). The spatial axes are addressed from the END
    # of the shape (m.ndim - ndim + d) rather than as the fixed 1 + d,
    # because c2 may carry a leading per-mechanism axis in addition to the
    # batch axis (q_n_mechanisms > 1: shape (L, batch, *spatial)); for the
    # plain (batch, *spatial) case the two indexings are identical.
    for idx in (0, 1):
        m = out[idx].clone()
        for d in range(ndim):
            lo, hi = boundary[2 * d], boundary[2 * d + 1]
            m = _mirror_pad_model_axis(m, m.ndim - ndim + d, fd_pad, lo, hi, "pressure")
        out[idx] = m
    # ar (index 2) is read only at the centre point -- no mirror-pad needed.
    # Each buoyancy lives on its own velocity grid, imaged along its own axis.
    for d in range(ndim):
        lo, hi = boundary[2 * d], boundary[2 * d + 1]
        if lo == BC_PML and hi == BC_PML:
            continue
        out[3 + d] = _mirror_pad_model_axis(
            out[3 + d].clone(), 1 + d, fd_pad, lo, hi, "velocity"
        )
    # This runs inside the adjoint pass (grad tracking disabled), where
    # ``clone``/``cat`` drop ``requires_grad``; the compiled backend sizes each
    # model's gradient buffer from that flag, so restore it to match the
    # originals (a fresh, non-aliasing tensor is still returned).
    for i, m in enumerate(models):
        if out[i] is not m and m.requires_grad and not out[i].requires_grad:
            out[i] = out[i].detach().requires_grad_(True)
    return out


def mirror_pad_acoustic_vti_models(
    models: list[torch.Tensor], boundary: list[int], fd_pad: int
) -> list[torch.Tensor]:
    """Return fresh acoustic_vti model clones with guard rings mirror-padded.

    ``models`` is ``[c11, c13, c33, buoyancy_0, ..., buoyancy_{ndim-1}]``
    (``acoustic_vti_model_names`` order; dimension 0 is the symmetry axis --
    ``z`` in 3-D, ``y`` in 2-D -- and the remaining dimensions are transverse).

    acoustic_vti's compiled adjoint velocity update (``backward_vz_cell``/
    ``backward_vy_cell``/``backward_vx_cell`` in acoustic_vti_kernels.h, via
    the ``VZ_Z``/``VY_Y``/``VX_X`` macros) reads ``c13``/``c33`` (axis) or
    ``c11``/``c13`` (transverse) at neighbour offsets along the differentiated
    axis, combined pointwise with ``sigma_h``/``sigma_v`` before
    differencing -- structurally the transpose of ``forward_sigma_cell``'s
    ``dvzdz``/``dvydy``/``dvxdx`` term feeding *both* stresses (see
    acoustic_vti.c's header, "CONTINUOUS SYSTEM": ``sigma_v`` is fed by the
    axis derivative weighted by ``c33`` and by the transverse derivatives
    weighted by ``c13``; ``sigma_h`` is fed by the transverse derivatives
    weighted by ``c11`` and the axis derivative weighted by ``c13``).

    Because only ``sigma_v`` is guard-filled on the axis face and only
    ``sigma_h`` is guard-filled on transverse faces (see
    ``acoustic_vti.py``'s ``_acoustic_vti_axis_boundary``/
    ``_acoustic_vti_transverse_boundary`` -- directly mirroring
    :func:`fill_acoustic_pressure_boundary`'s per-axis selectivity, since
    forward's ``sigma_h``/``sigma_v`` guard rings are provably unread outside
    that selective face: acoustic_vti.c's forward kernel never reads
    ``sigma_h`` at a z-offset or ``sigma_v`` at a transverse offset), the
    *other* stress's guard ring is exactly zero on any face where a boundary
    is set (freshly-allocated wavefields, never written there). Every term
    weighted by ``c13`` therefore multiplies a stress that is zero in its own
    ghost region on that face, so ``c13``'s own ghost value is irrelevant
    there and needs no mirror-pad at all. Only ``c33`` (pairs with
    ``sigma_v``, axis-only) and ``c11`` (pairs with ``sigma_h``,
    transverse-only) need the even/periodic-wrap "pressure" mirror-pad (same
    convention as acoustic's ``K`` / viscoacoustic's ``m1``/``c2``), each
    restricted to the face where its paired stress is actually guard-filled.
    This is the acoustic image rule applied independently to two decoupled
    "acoustic-like" subsystems (``sigma_v``/``c33``/``vz``-axis and
    ``sigma_h``/``c11``/``v``-transverse); ``c13`` is unmodified (not even
    cloned) because it never needs one. Each ``buoyancy_d`` is read at
    neighbour offsets along its own axis ``d`` only (own-axis coupling is
    unchanged from acoustic/elastic), exactly like
    :func:`mirror_pad_acoustic_models`. An all-``BC_PML`` boundary is a no-op
    (the inputs are returned unchanged). Fresh clones are returned so the
    (possibly aliased) saved user model tensors are never written in place.
    """
    if all(code == BC_PML for code in boundary):
        return models
    ndim = len(models) - 3
    out = list(models)
    # c33 (index 2) pairs with sigma_v: mirror-padded on the axis face (dim 0)
    # only.
    axis_lo, axis_hi = boundary[0], boundary[1]
    if not (axis_lo == BC_PML and axis_hi == BC_PML):
        out[2] = _mirror_pad_model_axis(
            out[2].clone(), 1, fd_pad, axis_lo, axis_hi, "pressure"
        )
    # c11 (index 0) pairs with sigma_h: mirror-padded on every transverse face
    # (dims 1..ndim-1) only.
    c11_needed = any(
        not (boundary[2 * d] == BC_PML and boundary[2 * d + 1] == BC_PML)
        for d in range(1, ndim)
    )
    if c11_needed:
        c11 = out[0].clone()
        for d in range(1, ndim):
            lo, hi = boundary[2 * d], boundary[2 * d + 1]
            if lo == BC_PML and hi == BC_PML:
                continue
            c11 = _mirror_pad_model_axis(c11, 1 + d, fd_pad, lo, hi, "pressure")
        out[0] = c11
    # c13 (index 1) is never read at a neighbour offset paired with a
    # nonzero-in-its-ghost-region stress (see the docstring) -- no mirror-pad.
    # Each buoyancy lives on its own velocity grid, imaged along its own axis.
    for d in range(ndim):
        lo, hi = boundary[2 * d], boundary[2 * d + 1]
        if lo == BC_PML and hi == BC_PML:
            continue
        out[3 + d] = _mirror_pad_model_axis(
            out[3 + d].clone(), 1 + d, fd_pad, lo, hi, "velocity"
        )
    # This runs inside the adjoint pass (grad tracking disabled), where
    # ``clone``/``cat`` drop ``requires_grad``; the compiled backend sizes each
    # model's gradient buffer from that flag, so restore it to match the
    # originals (a fresh, non-aliasing tensor is still returned).
    for i, m in enumerate(models):
        if out[i] is not m and m.requires_grad and not out[i].requires_grad:
            out[i] = out[i].detach().requires_grad_(True)
    return out


def _mirror_pad_nodal_model_axis(
    model: torch.Tensor, dim: int, pad: int, lo: int, hi: int
) -> torch.Tensor:
    """Even/wrap mirror-pad a nodal model coefficient's guard ring on one axis.

    Used by scalar_born, whose model coefficients (v, scatter) live on the same
    regular nodal grid as the field. Unlike the field fill there is no sign
    flip; for a ``free`` face the single on-plane guard cell (the cell with no
    interior mirror partner) is left at its pre-existing value rather than
    forced to zero, exactly matching scalar_born.c's
    ``fill_bc_line(..., is_model=1)`` convention (it skips that cell rather
    than overwriting it -- there is no interior model value at the reflection
    plane to derive it from).
    """
    if lo == BC_PML and hi == BC_PML:
        return model
    n = model.size(dim)
    m = n - 2 * pad
    interior = model.narrow(dim, pad, m)
    if lo == BC_PML:
        low = model.narrow(dim, 0, pad)
    elif lo == BC_PERIODIC:
        low = interior.narrow(dim, m - pad, pad)
    elif lo == BC_RIGID:
        low = torch.flip(interior.narrow(dim, 0, pad), [dim])
    elif pad == 1:  # BC_FREE, no cells to mirror besides the on-plane one
        low = model.narrow(dim, 0, pad)
    else:  # BC_FREE: even about the outer node; on-plane cell kept as-is
        mirrored = torch.flip(interior.narrow(dim, 0, pad - 1), [dim])
        on_plane = model.narrow(dim, pad - 1, 1)
        low = torch.cat([mirrored, on_plane], dim)
    if hi == BC_PML:
        high = model.narrow(dim, n - pad, pad)
    elif hi == BC_PERIODIC:
        high = interior.narrow(dim, 0, pad)
    elif hi == BC_RIGID:
        high = torch.flip(interior.narrow(dim, m - pad, pad), [dim])
    elif pad == 1:  # BC_FREE
        high = model.narrow(dim, n - pad, pad)
    else:  # BC_FREE
        on_plane = model.narrow(dim, n - pad, 1)
        mirrored = torch.flip(interior.narrow(dim, m - (pad - 1), pad - 1), [dim])
        high = torch.cat([on_plane, mirrored], dim)
    return torch.cat([low, interior, high], dim)


def mirror_pad_scalar_born_models(
    models: list[torch.Tensor], boundary: list[int], fd_pad: int
) -> list[torch.Tensor]:
    """Return fresh scalar_born model clones with guard rings mirror-padded.

    ``models`` is ``[v, scatter]`` (scalar_born's regular-grid nodal model
    coefficients). Unlike the forward, which reads ``v``/``scatter`` only at
    the current grid point, the compiled adjoint (backward/backward_sc) reads
    them at neighbour offsets within the (necessarily pml_width=0) region next
    to a reflecting face, so their guard rings must hold the even/
    periodic-wrap image there for those near-boundary coefficient reads to be
    correct. An all-``BC_PML`` boundary is a no-op (the inputs are returned
    unchanged). Fresh clones are returned so the (possibly aliased) saved user
    model tensors are never written in place.
    """
    if all(code == BC_PML for code in boundary):
        return models
    ndim = len(boundary) // 2
    out = list(models)
    for i, model in enumerate(models):
        new_model = model.clone()
        for d in range(ndim):
            lo, hi = boundary[2 * d], boundary[2 * d + 1]
            if lo == BC_PML and hi == BC_PML:
                continue
            new_model = _mirror_pad_nodal_model_axis(new_model, 1 + d, fd_pad, lo, hi)
        out[i] = new_model
    # This runs inside the adjoint ``Function.backward``/``run_backward_core``
    # (grad tracking disabled there), where ``clone``/``cat`` drop
    # ``requires_grad``; the compiled backend sizes each model's gradient
    # buffer from that flag, so restore it to match the originals (a fresh,
    # non-aliasing tensor is still returned).
    for i, m in enumerate(models):
        if out[i] is not m and m.requires_grad and not out[i].requires_grad:
            out[i] = out[i].detach().requires_grad_(True)
    return out


# --- Elastic (full stress-tensor) boundary fill -----------------------------
#
# Physics source: docs/dev/boundary_conditions/ROBERTSSON_FS_SUMMARY.md
# Sec 1.2-1.3 (node geometry + the flat-face image method), cited point by
# point below. docs/boundaries.rst's Policy section records the resulting
# per-field parity table and the free/rigid duality judgement call.
#
# Per axis A, exactly six of the (2*ndim+1) wavefields are ever read across
# that face by any velocity/stress update (verified by inspecting every term
# of the continuous elastic system that differentiates in the A direction):
#   sigma_AA          (normal stress on axis A -- lattice-coincident with the
#                       face, "on-row": Sec 1.2, "the free surface plane is
#                       placed exactly through the row of sigma_xx, sigma_zz
#                       ... nodes")
#   sigma_AB, sigma_AC (the two shear stresses sharing axis A -- staggered
#                       half a cell from the face, "off-row": Sec 1.2,
#                       "the shear stress sigma_xz [sits] in the lower-right
#                       corner")
#   v_A, v_B, v_C      (every velocity component -- Sec 1.3 point 3,
#                       "Particle velocities above the surface are set to
#                       ZERO", stated generically, not just the axis-normal
#                       component)
# The other three fields (sigma_BB, sigma_CC, sigma_BC) never differentiate
# in the A direction anywhere in the elastic system, so they need no guard
# fill on face A at all (their own values are still correct because they
# read v_A's -- already filled -- ghost region for their own divergence
# term).
#
# Free-face treatment (Sec 1.1/1.3):
#   sigma_AA: the traction-free condition itself (Sec 1.1, "sigma_zz = 0");
#     the on-row node is held to zero and its ghost is imaged ODD (point 1),
#     exactly :func:`fill_acoustic_pressure_boundary`'s ``p`` treatment (p is
#     the same kind of lattice-coincident, single-PDE Dirichlet field).
#   sigma_AB/AC: "the components that need to be defined above the surface
#     ... are sigma_zz and sigma_xz ... imaged as ODD" (point 1) -- odd
#     about the half-cell plane, but (unlike acoustic's velocity) with no
#     on-plane node to hold to zero (off-row, no coincident lattice point).
#   v_A/v_B/v_C: "Particle velocities above the surface are set to ZERO"
#     (point 3) -- ghost region literally zeroed (not mirrored); this is also
#     explicitly the flat-surface limit of the improved-vacuum method (Sec
#     2.1: buoyancy=0 makes the velocity update a no-op above the surface),
#     which is the property this file's IVM-agreement gate checks. This
#     zero-fill is a ghost-region/stencil-construction device (the ghost
#     points sit outside the physical domain); it does not claim real
#     surface velocity is zero -- the correctly-imaged (odd) stresses, read
#     by the interior stencil at/near the real surface row, are what
#     reproduce the true nonzero surface velocity (e.g. Rayleigh-wave
#     motion) at actual grid points. This is a different code path from
#     (and not constrained by) the Hicks source/receiver-folding parity note
#     elsewhere in docs/boundaries.rst, which concerns the *continuous*
#     half-space Green's function's analytic symmetry for stencil-tap
#     folding near the (IVM) surface, not this discrete guard-ring's own
#     ghost-fill convention.
#
# Rigid-face treatment is NOT covered by Robertsson (a free-surface paper) --
# it is derived by the same stress<->velocity role-swap duality the
# scalar/acoustic/viscoacoustic/acoustic_vti precedents already use for
# their own free/rigid pairs (docs/boundaries.rst records this as the
# explicit judgement call): a rigid (clamped) wall fixes every velocity
# component to zero (the standard elastodynamics Dirichlet condition, v=0
# vector) and leaves every stress component unconstrained (even/Neumann-like
# image, no zero). Concretely this swaps which of {stress, velocity} gets
# the "clamp" (odd + on-plane zero) treatment on a given lattice, reusing
# the same acoustic index arithmetic (see the docstrings below) with the
# free/rigid roles exchanged.
#
# Periodic wraps whichever of the six fields is touched on that face, using
# the same wrap arithmetic as the acoustic precedent (period ``m - 1``, the
# staggered-cell count) -- BC_PERIODIC on both faces of an axis is enforced
# by ``set_boundary``.


def _onrow_odd_zero(
    interior: torch.Tensor, ax: int, pad: int, is_low: bool
) -> torch.Tensor:
    """Odd image about a lattice-coincident node forced to zero by the caller.

    Same arithmetic as :func:`_acoustic_p_face`'s ``free`` branch; reused for
    any elastic field whose own physical boundary condition clamps it to
    zero on a node exactly on the face plane (``sigma_AA`` under ``free``;
    the tangential velocities under ``rigid``).
    """
    return _acoustic_p_face(interior, ax, pad, BC_FREE, is_low)


def _onrow_even(
    interior: torch.Tensor, ax: int, pad: int, is_low: bool
) -> torch.Tensor:
    """Even image about a lattice-coincident node, no zero.

    Same arithmetic as :func:`_acoustic_p_face`'s ``rigid`` branch; reused
    for any elastic field that is unconstrained on a node exactly on the
    face plane (``sigma_AA`` under ``rigid``).
    """
    return _acoustic_p_face(interior, ax, pad, BC_RIGID, is_low)


def _onrow_wrap(
    interior: torch.Tensor, ax: int, pad: int, is_low: bool
) -> torch.Tensor:
    """Periodic wrap for a lattice-coincident field (period ``m - 1``)."""
    return _acoustic_p_face(interior, ax, pad, BC_PERIODIC, is_low)


def _offrow_clamp_zero(
    interior: torch.Tensor, ax: int, pad: int, is_low: bool
) -> torch.Tensor:
    """Odd image about the half-cell plane; the coincident ghost cell is zero.

    (Low face only -- the high face's coincident node is *interior* and is
    zeroed by the caller, the staggered low/high pad asymmetry.)

    Same arithmetic as :func:`_acoustic_v_face`'s ``rigid`` branch; reused
    for the axis-normal velocity ``v_A`` under ``rigid`` (the standard
    elastodynamics clamped-wall condition applied to the one velocity
    component that is naturally staggered onto this axis's half-cell plane).
    """
    return _acoustic_v_face(interior, ax, pad, BC_RIGID, is_low)


def _offrow_wrap(
    interior: torch.Tensor, ax: int, pad: int, is_low: bool
) -> torch.Tensor:
    """Periodic wrap for a half-cell-staggered field (period ``m - 1``)."""
    return _acoustic_v_face(interior, ax, pad, BC_PERIODIC, is_low)


def _shear_face(
    interior: torch.Tensor, ax: int, pad: int, code: int, is_low: bool
) -> torch.Tensor:
    """Ghost image for a shared-axis shear stress (``sigma_AB``/``sigma_AC``).

    Off-row (staggered half a cell from the face, like a velocity) but with
    no on-plane node ever forced to zero (no shear-stress lattice point
    coincides with either the ``free`` or the ``rigid`` wall plane -- Sec 1.2
    places it "in the lower-right corner", half a cell from *both* the
    normal-stress row and the tangential-velocity node). ``free``: ODD about
    the half-cell plane (Sec 1.3 point 1, "sigma_xz ... imaged as ODD"; the
    on-plane-zero half of that point does not apply since no node sits
    there -- only ``sigma_AA`` gets the on-plane zero). ``rigid``: EVEN about
    the half-cell plane (the role-swapped dual: shear stress is
    unconstrained at a clamped-velocity wall, so it gets the same
    "unconstrained" parity ``sigma_AA`` gets under ``rigid``, just applied on
    the half-cell lattice).

    Derived by taking :func:`_acoustic_v_face`'s (already-validated) index
    selection for each side and sign-flipping/dropping its on-plane zero to
    match the ODD-free/EVEN-rigid/no-zero-ever parity above, rather than
    re-deriving new index arithmetic:
      * ``free``:  odd, no zero  = ``-1 *`` (acoustic's ``free`` branch, which
        is even, no zero).
      * ``rigid`` low:  even, no zero = acoustic's ``free``-branch low (that
        branch is already even/no-zero on the low side).
      * ``rigid`` high: even, no zero = ``-1 *`` (acoustic's ``rigid``-branch
        high, which is odd but already has no zero on the high side -- only
        its low side has the on-plane zero, handled by the caller for
        ``v_A``, never invoked here).
    """
    if code == BC_PERIODIC:
        return _acoustic_v_face(interior, ax, pad, BC_PERIODIC, is_low)
    if code == BC_FREE:
        return -_acoustic_v_face(interior, ax, pad, BC_FREE, is_low)
    # BC_RIGID
    if is_low:
        return _acoustic_v_face(interior, ax, pad, BC_FREE, True)
    return -_acoustic_v_face(interior, ax, pad, BC_RIGID, False)


def fill_elastic_stress_boundary(
    normal_stresses: list[torch.Tensor],
    shear_stresses: list[list[torch.Tensor]],
    boundary: list[int],
    fd_pad: int,
) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
    """Fill the stress guard ring of the elastic staggered grid per-face.

    ``normal_stresses[d]`` is ``sigma_dd``; ``shear_stresses[d1][d2] ==
    shear_stresses[d2][d1]`` is ``sigma_{d1 d2}`` (``d1 != d2``). ``BC_PML``
    faces are untouched (an all-PML ``boundary`` is a no-op). On a ``free``
    face the on-surface normal-stress node itself is held to zero. Axes are
    processed in order so multi-axis (corner) shear stresses see each face's
    update in turn, mirroring :func:`fill_scalar_boundary`'s sequential
    per-axis pass. Returns new lists (fresh tensors on touched faces; the
    same objects on untouched ones).
    """
    ndim = len(normal_stresses)
    pad = fd_pad
    normal_stresses = list(normal_stresses)
    shear_stresses = [list(row) for row in shear_stresses]
    for a in range(ndim):
        lo = boundary[2 * a]
        hi = boundary[2 * a + 1]
        if lo == BC_PML and hi == BC_PML:
            continue
        ax = 1 + a

        # sigma_AA: on-row (lattice-coincident with the face).
        s = normal_stresses[a]
        n = s.size(ax)
        m = n - 2 * pad + 1
        interior = s.narrow(ax, pad, m)
        if lo == BC_FREE:
            interior = torch.cat(
                [interior.narrow(ax, 0, 1) * 0, interior.narrow(ax, 1, m - 1)], ax
            )
        if hi == BC_FREE:
            interior = torch.cat(
                [interior.narrow(ax, 0, m - 1), interior.narrow(ax, m - 1, 1) * 0], ax
            )
        if lo == BC_PML:
            low = s.narrow(ax, 0, pad)
        elif lo == BC_FREE:
            low = _onrow_odd_zero(interior, ax, pad, True)
        elif lo == BC_RIGID:
            low = _onrow_even(interior, ax, pad, True)
        else:
            low = _onrow_wrap(interior, ax, pad, True)
        if hi == BC_PML:
            high = s.narrow(ax, n - (pad - 1), pad - 1)
        elif hi == BC_FREE:
            high = _onrow_odd_zero(interior, ax, pad, False)
        elif hi == BC_RIGID:
            high = _onrow_even(interior, ax, pad, False)
        else:
            high = _onrow_wrap(interior, ax, pad, False)
        normal_stresses[a] = torch.cat([low, interior, high], ax)

        # sigma_AB / sigma_AC (b != a): off-row, shared axis A, never zeroed.
        for b in range(ndim):
            if b == a:
                continue
            sh = shear_stresses[a][b]
            n = sh.size(ax)
            m = n - 2 * pad + 1
            interior = sh.narrow(ax, pad, m)
            # Off-row fields (like a velocity) have one redundant high-side
            # interior node under a periodic domain of period m - 1; slave
            # it to node 0 (same arithmetic as
            # fill_elastic_velocity_boundary's own off-row/periodic case,
            # itself matching fill_acoustic_velocity_boundary's precedent) --
            # otherwise that node evolves as an independent (un-wrapped)
            # extra degree of freedom and periodic translation invariance
            # drifts once the shear-stress field becomes dynamically active.
            if hi == BC_PERIODIC:
                interior = torch.cat(
                    [interior.narrow(ax, 0, m - 1), interior.narrow(ax, 0, 1)], ax
                )
            low = (
                sh.narrow(ax, 0, pad)
                if lo == BC_PML
                else _shear_face(interior, ax, pad, lo, True)
            )
            high = (
                sh.narrow(ax, n - (pad - 1), pad - 1)
                if hi == BC_PML
                else _shear_face(interior, ax, pad, hi, False)
            )
            new_sh = torch.cat([low, interior, high], ax)
            shear_stresses[a][b] = new_sh
            shear_stresses[b][a] = new_sh
    return normal_stresses, shear_stresses


def fill_elastic_velocity_boundary(
    velocities: list[torch.Tensor], boundary: list[int], fd_pad: int
) -> list[torch.Tensor]:
    """Fill the velocity guard ring of the elastic staggered grid per-face.

    Unlike acoustic (where only the boundary-normal velocity is imaged),
    *every* velocity component is touched on each reflecting face: each
    normal stress's own divergence term reads every velocity's ghost region
    along that axis (see the module-level derivation above), and (for
    ``free``) Robertsson's point 3 zeroes "particle velocities" generically,
    not just the normal component.

    ``free``: every velocity's ghost region is set to zero (not mirrored --
    Sec 1.3 point 3; this is also IVM's own flat-surface behaviour, Sec 2.1).
    ``rigid``: every velocity component is clamped to zero at the wall (the
    standard elastodynamics fixed-boundary condition) -- ``v_A`` (off-row,
    staggered onto this axis) via :func:`_offrow_clamp_zero` (acoustic's own
    rigid-velocity arithmetic, unchanged); ``v_B``/``v_C`` (on-row, sharing
    the normal-stress lattice) via :func:`_onrow_odd_zero` (the same
    arithmetic ``sigma_AA`` uses under ``free``, since both are a
    lattice-coincident field being clamped to zero).
    ``periodic``: wrap (offrow/onrow arithmetic per component, matching
    :func:`fill_acoustic_velocity_boundary`'s own extra-high-node slaving for
    the axis-normal component).

    Returns a new list (fresh tensors on touched faces; the same objects on
    untouched ones).
    """
    ndim = len(velocities)
    pad = fd_pad
    velocities = list(velocities)
    for a in range(ndim):
        lo = boundary[2 * a]
        hi = boundary[2 * a + 1]
        if lo == BC_PML and hi == BC_PML:
            continue
        ax = 1 + a
        for k in range(ndim):
            v = velocities[k]
            n = v.size(ax)
            m = n - 2 * pad + 1
            interior = v.narrow(ax, pad, m)
            if k == a:
                # Off-row: v_A is naturally staggered onto axis A's own
                # half-cell lattice (acoustic's own velocity convention).
                if hi == BC_RIGID:
                    interior = torch.cat(
                        [
                            interior.narrow(ax, 0, m - 1),
                            interior.narrow(ax, m - 1, 1) * 0,
                        ],
                        ax,
                    )
                elif hi == BC_PERIODIC:
                    interior = torch.cat(
                        [interior.narrow(ax, 0, m - 1), interior.narrow(ax, 0, 1)], ax
                    )
                if lo == BC_PML:
                    low = v.narrow(ax, 0, pad)
                elif lo == BC_FREE:
                    low = v.narrow(ax, 0, pad) * 0
                elif lo == BC_RIGID:
                    low = _offrow_clamp_zero(interior, ax, pad, True)
                else:
                    low = _offrow_wrap(interior, ax, pad, True)
                if hi == BC_PML:
                    high = v.narrow(ax, n - (pad - 1), pad - 1)
                elif hi == BC_FREE:
                    high = v.narrow(ax, n - (pad - 1), pad - 1) * 0
                elif hi == BC_RIGID:
                    high = _offrow_clamp_zero(interior, ax, pad, False)
                else:
                    high = _offrow_wrap(interior, ax, pad, False)
            else:
                # On-row: v_B/v_C share the normal-stress lattice along A.
                if lo == BC_RIGID:
                    interior = torch.cat(
                        [interior.narrow(ax, 0, 1) * 0, interior.narrow(ax, 1, m - 1)],
                        ax,
                    )
                if hi == BC_RIGID:
                    interior = torch.cat(
                        [
                            interior.narrow(ax, 0, m - 1),
                            interior.narrow(ax, m - 1, 1) * 0,
                        ],
                        ax,
                    )
                if lo == BC_PML:
                    low = v.narrow(ax, 0, pad)
                elif lo == BC_FREE:
                    low = v.narrow(ax, 0, pad) * 0
                elif lo == BC_RIGID:
                    low = _onrow_odd_zero(interior, ax, pad, True)
                else:
                    low = _onrow_wrap(interior, ax, pad, True)
                if hi == BC_PML:
                    high = v.narrow(ax, n - (pad - 1), pad - 1)
                elif hi == BC_FREE:
                    high = v.narrow(ax, n - (pad - 1), pad - 1) * 0
                elif hi == BC_RIGID:
                    high = _onrow_odd_zero(interior, ax, pad, False)
                else:
                    high = _onrow_wrap(interior, ax, pad, False)
            velocities[k] = torch.cat([low, interior, high], ax)
    return velocities


def mirror_pad_elastic_models(
    models: list[torch.Tensor], boundary: list[int], fd_pad: int
) -> list[torch.Tensor]:
    """Return fresh elastic model clones with mirror-padded guard rings.

    Padded on reflecting faces, for the compiled adjoint's near-boundary
    neighbour reads.

    ``models`` is ``elastic_model_names`` order: ``[lamb, mu, (mu_zy, mu_zx,)
    mu_yx?, buoyancy_z?, buoyancy_y?, buoyancy_x]`` (the shear-stiffness and
    ``z``/``y`` buoyancy entries present only for ndim >= 2/3 respectively;
    see ``elastic.elastic_model_names``).

    Per ``elastic_kernels.h``'s own documented addressing convention (the
    comment above ``FUNC(backward_vz_cell)``/``FUNC(backward_sigmaii_cell)``):
    ``FUNC(backward_v*_cell)`` (the adjoint velocity update, structurally the
    transpose of the forward stress update) differentiates ``lamb``/``mu``/
    every ``mu_*`` shear stiffness *with a neighbour offset* -- exactly like
    acoustic's own ``K`` -- so all of those need the even/periodic-wrap
    "pressure" mirror-pad on *every* reflecting axis. ``FUNC(backward_sigma*
    _cell)`` (the adjoint stress update, transpose of the forward velocity
    update) differentiates each ``buoyancy_d`` with a neighbour offset *on
    its own axis* -- exactly like acoustic's own per-axis ``buoyancy_d`` --
    so each buoyancy is mirror-padded only on its own axis, using the
    "velocity" plane convention. Mirrors :func:`mirror_pad_acoustic_models`
    exactly, generalised from acoustic's single ``K`` to elastic's five
    stress-role models (``lamb``, ``mu``, and every ``mu_*`` shear
    stiffness). An all-``BC_PML`` boundary is a no-op (the inputs are
    returned unchanged). Fresh clones are returned so the (possibly aliased)
    saved user model tensors are never written in place.
    """
    if all(code == BC_PML for code in boundary):
        return models
    ndim = len(boundary) // 2
    # elastic_model_names(ndim) order: lamb, mu, [mu_zy, mu_zx,] [mu_yx,]
    # [buoyancy_z,] [buoyancy_y,] buoyancy_x. The shear-stiffness block has
    # 3 entries for ndim==3 (mu_zy, mu_zx, mu_yx), 1 for ndim==2 (mu_yx), 0
    # for ndim==1; the buoyancy block always has ndim entries.
    n_shear = {3: 3, 2: 1, 1: 0}[ndim]
    stress_role_idx = list(range(2 + n_shear))  # lamb, mu, mu_*
    out = list(models)
    for idx in stress_role_idx:
        m = out[idx].clone()
        for d in range(ndim):
            lo, hi = boundary[2 * d], boundary[2 * d + 1]
            m = _mirror_pad_model_axis(m, 1 + d, fd_pad, lo, hi, "pressure")
        out[idx] = m
    # Each buoyancy lives on its own velocity grid, imaged along its own
    # axis only (buoyancy_z is models[2+n_shear], ..., buoyancy_x is the
    # last model, matching elastic_model_names' trailing ndim entries).
    buoyancy_start = 2 + n_shear
    for d in range(ndim):
        lo, hi = boundary[2 * d], boundary[2 * d + 1]
        if lo == BC_PML and hi == BC_PML:
            continue
        out[buoyancy_start + d] = _mirror_pad_model_axis(
            out[buoyancy_start + d].clone(), 1 + d, fd_pad, lo, hi, "velocity"
        )
    # This runs inside the adjoint pass (grad tracking disabled), where
    # ``clone``/``cat`` drop ``requires_grad``; the compiled backend sizes
    # each model's gradient buffer from that flag, so restore it to match
    # the originals (a fresh, non-aliasing tensor is still returned).
    for i, m in enumerate(models):
        if out[i] is not m and m.requires_grad and not out[i].requires_grad:
            out[i] = out[i].detach().requires_grad_(True)
    return out


def mirror_pad_elastic_vti_models(
    models: list[torch.Tensor], boundary: list[int], fd_pad: int
) -> list[torch.Tensor]:
    """Return fresh elastic_vti model clones with mirror-padded guard rings.

    ``models`` is ``elastic_vti_model_names`` order: 3-D ``[c11, c12, c13,
    c33, c55_zy, c55_zx, c66_yx, buoyancy_z, buoyancy_y, buoyancy_x]``; 2-D
    ``[c11, c13, c33, c55_yx, buoyancy_y, buoyancy_x]`` (see
    ``elastic_vti.elastic_vti_model_names``).

    elastic_vti's compiled adjoint velocity update (``backward_vz_cell``/
    ``backward_vy_cell``/``backward_vx_cell`` in ``elastic_vti_kernels.h``)
    reads a subset of the stress-role stiffnesses (``c11``/``c12``/``c13``/
    ``c33``/``c55_*``/``c66_yx``, whichever couples into that axis's normal
    or shared-axis shear terms -- see that header's own documented
    addressing convention, e.g. ``backward_vz_cell``'s comment: "c13_shot/
    c33_shot/c55_zy_shot/c55_zx_shot are POINTERS ... since the VZ_Z(_PML)/
    VZ_Y(_PML)/VZ_X(_PML) stencils touch them at every offset the
    finite-difference stencil visits") *with a neighbour offset*, exactly
    like isotropic elastic's ``lamb``/``mu``/``mu_*``. Rather than work out
    the precise per-model/per-axis subset (VTI's multi-coefficient coupling
    makes that a more intricate case analysis than isotropic elastic's
    single ``lamb``/``mu`` pair), this follows
    :func:`mirror_pad_elastic_models`'s own safe/conservative choice
    verbatim: every stress-role model is mirror-padded on *every* reflecting
    axis (a strict superset of what any single kernel actually reads at a
    neighbour offset, so never wrong -- only possibly a few redundant
    clone/cat calls on model tensors, which is irrelevant to the cost of a
    guard-ring computation). Each ``buoyancy_d`` is read at neighbour
    offsets along its own axis ``d`` only (the velocity update, PML,
    staggering are identical to isotropic elastic per this module's own
    header -- see ``elastic_vti.c``'s "ADJOINT / GRADIENTS" section), mirrored
    with the "velocity" plane convention, exactly like
    :func:`mirror_pad_elastic_models`. An all-``BC_PML`` boundary is a no-op
    (the inputs are returned unchanged). Fresh clones are returned so the
    (possibly aliased) saved user model tensors are never written in place.
    """
    if all(code == BC_PML for code in boundary):
        return models
    ndim = len(boundary) // 2
    # elastic_vti_model_names(ndim) order: stress-role models first (c11,
    # [c12,] c13, c33, c55_*, [c66_yx,]), then ndim buoyancies. 3-D has 7
    # stress-role entries (c11, c12, c13, c33, c55_zy, c55_zx, c66_yx); 2-D
    # has 4 (c11, c13, c33, c55_yx).
    n_stress_role = {3: 7, 2: 4}[ndim]
    out = list(models)
    for idx in range(n_stress_role):
        m = out[idx].clone()
        for d in range(ndim):
            lo, hi = boundary[2 * d], boundary[2 * d + 1]
            m = _mirror_pad_model_axis(m, 1 + d, fd_pad, lo, hi, "pressure")
        out[idx] = m
    # Each buoyancy lives on its own velocity grid, imaged along its own
    # axis only (buoyancy_z is models[n_stress_role], ..., buoyancy_x is the
    # last model, matching elastic_vti_model_names' trailing ndim entries).
    for d in range(ndim):
        lo, hi = boundary[2 * d], boundary[2 * d + 1]
        if lo == BC_PML and hi == BC_PML:
            continue
        out[n_stress_role + d] = _mirror_pad_model_axis(
            out[n_stress_role + d].clone(), 1 + d, fd_pad, lo, hi, "velocity"
        )
    # This runs inside the adjoint pass (grad tracking disabled), where
    # ``clone``/``cat`` drop ``requires_grad``; the compiled backend sizes
    # each model's gradient buffer from that flag, so restore it to match
    # the originals (a fresh, non-aliasing tensor is still returned).
    for i, m in enumerate(models):
        if out[i] is not m and m.requires_grad and not out[i].requires_grad:
            out[i] = out[i].detach().requires_grad_(True)
    return out
