"""Spatial domain decomposition for the scalar propagator (EXPLORATORY PROTOTYPE).

This is a self-contained exploration kept OUTSIDE the installed ``deepwave``
package (see the directory ``README.md`` for the lessons learned and the open
questions). It is not imported by deepwave and ships in no wheel; it is preserved
in git so its findings inform future decisions. The maintainer has deferred
integrating domain decomposition into the package pending those decisions.

This module splits the model into a stack of *slab* subdomains along one chosen
axis, propagates each slab with the ordinary :func:`deepwave.scalar` propagator,
and exchanges a thin halo of cells between neighbouring slabs at every internal
time step. The result is identical (to machine precision) to a single monolithic
:func:`scalar` call — both the forward field/receivers and the gradient.

Scope of this MVP — read before relying on it for scale. This is the
**algorithmic and adjoint-correctness foundation** of Workstream E, *not* a
finished scale-out solution. As written it runs all subdomains **sequentially in
one process on a single device** and **does not yet reduce peak memory**: every
slab's state is held simultaneously, and the differentiable path retains the
whole autograd graph across all steps (so gradients use *more* memory than the
monolithic call, not less). The "no single propagation holds the whole model"
payoff requires two further pieces, both future work: (a) placing slabs on
**separate devices** with cross-device halo exchange (``torch.distributed`` / peer
copy at the PyTorch level — the C/CUDA backend stays untouched), and (b)
**per-subdomain checkpointing** so each device stores only its slab's snapshots.
What is proven today is that the overlapping-Schwarz halo exchange reproduces the
monolithic run exactly, forward and adjoint — the necessary correctness base for
those next steps.

Performance: the halo exchange itself is cheap (a few ``fd_pad`` planes copied in
host PyTorch). The cost is that the time loop is driven **one internal step at a
time from Python**, so each step re-runs the full per-call front end (survey
extraction, PML-profile setup, allocation, the autograd ``Function``) once per
slab — wall-clock is substantially slower than a monolithic ``scalar`` call,
which amortises that setup over the whole time loop inside one C call. A faster
callback-threaded path is future work. This module is purely additive: it does
not touch ``scalar``, the engine, or the C/CUDA backend, so runs that do not use
it are completely unaffected.

Why slabs (and not a full block decomposition) for the MVP: a slab split along a
single axis gives every subdomain exactly two neighbours and full-face halos, so
there are no diagonal/corner halo exchanges to get wrong. This is the bounded,
fully CPU-testable first step of Workstream E; block decomposition (corners) is a
later generalisation.

The mechanism (overlapping Schwarz). The propagator's own finite-difference ghost
ring (``fd_pad = accuracy // 2`` cells) is *internal*: it is stripped from the
returned wavefield and re-created as zeros on input, so it behaves as a rigid
``u = 0`` wall and cannot be used to inject neighbour values. Each subdomain
therefore carries an explicit halo of ``fd_pad`` *real* interior cells beyond the
cells it owns, with ``pml_width = 0`` on its interior faces. After every internal
step the halo cells (which the subdomain updated incorrectly against its own zero
ghost) are overwritten with the neighbour's freshly-computed owned values. Because
an owned cell's stencil reaches at most ``fd_pad`` cells — i.e. no further than the
last halo cell — the owned cells are always updated correctly, so the stitched
result matches the monolithic run exactly.

Limitations (MVP): scalar propagator only; a single split axis; a constant
(non-batched) velocity model; one shared ``max_vel``/``pml_freq`` across all
subdomains (required for a common internal time step); single process, single
device (see "Scope" above). Sources/receivers may be placed anywhere.

Stability: with three or more slabs an interior slab has zero PML on both split
faces, and if the PML on the *split* dimension is wider than a perpendicular
PML, a spurious mode grows at the seam/perpendicular-PML corner. This regime is
rejected (see the guard below); it does not affect the common symmetric-PML case
or two-slab/1-D decompositions. A seam interface condition that removes the
restriction is future work.
"""

from typing import List, Optional, Sequence, Tuple, Union

import torch

import deepwave
from deepwave._common import downsample_and_movedim, set_grid_spacing
from deepwave.common import (
    IGNORE_LOCATION,
    cfl_condition_n,
    upsample,
)
from deepwave.scalar import scalar_wavefield_names


def _partition(n: int, k: int) -> List[Tuple[int, int]]:
    """Split ``[0, n)`` into ``k`` contiguous, near-equal owned ranges."""
    base, rem = divmod(n, k)
    ranges = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def scalar_decomposed(
    v: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitudes: Optional[torch.Tensor] = None,
    source_locations: Optional[torch.Tensor] = None,
    receiver_locations: Optional[torch.Tensor] = None,
    n_subdomains: int = 2,
    split_dim: int = 0,
    accuracy: int = 4,
    pml_width: Union[int, Sequence[int]] = 20,
    pml_freq: Optional[float] = None,
    max_vel: Optional[float] = None,
    nt: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Domain-decomposed scalar propagation over a slab partition.

    The arguments mirror :func:`deepwave.scalar`. The model is split into
    ``n_subdomains`` slabs along ``split_dim``; the returned final wavefield and
    receiver amplitudes match a monolithic :func:`scalar` call to machine
    precision.

    Args:
        v: Velocity model of shape ``model_shape`` (no batch dimension).
        grid_spacing: Grid cell size (scalar or per-dimension).
        dt: Time step interval.
        source_amplitudes: ``(n_shots, n_sources, nt)`` or ``None``.
        source_locations: ``(n_shots, n_sources, ndim)`` integer cell indices, or
            ``None``. Coordinates are global (into the full model).
        receiver_locations: ``(n_shots, n_receivers, ndim)`` or ``None``.
        n_subdomains: Number of slabs to split the model into.
        split_dim: The model dimension (``0..ndim-1``) to split along.
        accuracy: Spatial finite-difference accuracy order.
        pml_width: PML width (scalar or per-face ``2*ndim`` list). Interior slab
            faces are forced to width 0; exterior faces keep this width.
        pml_freq: Dominant frequency for the PML profile (shared by all slabs).
        max_vel: Maximum velocity for the CFL/time-step calculation. Defaults to
            ``v.abs().max()``; pinned across all slabs.
        nt: Number of time steps (inferred from ``source_amplitudes`` if ``None``).

    Returns:
        ``(wavefield, receiver_amplitudes)`` where ``wavefield`` has shape
        ``(n_shots, *model_shape)`` (the final field over the model interior, no
        PML padding) and ``receiver_amplitudes`` has shape
        ``(n_shots, n_receivers, nt)``.
    """
    ndim = v.ndim
    if not 0 <= split_dim < ndim:
        raise ValueError(f"split_dim must be in [0, {ndim}); got {split_dim}.")
    if n_subdomains < 1:
        raise ValueError("n_subdomains must be >= 1.")
    if (source_amplitudes is None) != (source_locations is None):
        raise ValueError(
            "source_amplitudes and source_locations must both be given or both None."
        )
    if (
        source_amplitudes is not None
        and nt is not None
        and nt != source_amplitudes.shape[-1]
    ):
        raise ValueError("Only one of nt or source_amplitudes should be specified.")
    grid_spacing = set_grid_spacing(grid_spacing, ndim)
    model_shape = list(v.shape)
    n_split = model_shape[split_dim]

    if max_vel is None:
        max_vel = v.detach().abs().max().item()
    inner_dt, step_ratio = cfl_condition_n(grid_spacing, dt, max_vel)

    # Normalise pml_width to a per-face list [d0_begin, d0_end, d1_begin, ...].
    if isinstance(pml_width, int):
        pml = [pml_width] * (2 * ndim)
    else:
        pml = list(pml_width)
        if len(pml) != 2 * ndim:
            raise ValueError(f"pml_width must have length {2 * ndim}.")

    # Stability guard for the PML "corner" instability. With >=3 slabs an
    # interior slab has zero PML on both split faces; if the (outer-slab) PML on
    # the split dimension is wider than a perpendicular-dimension PML, a spurious
    # mode at the seam/perpendicular-PML corner grows exponentially (a silent,
    # catastrophic wrong answer). Empirically the scheme is stable exactly when
    # the split-dimension PML is no wider than every perpendicular PML. (Two
    # slabs and 1-D are always stable.) Reject the unstable regime explicitly.
    if n_subdomains >= 3 and ndim >= 2:
        split_pml = max(pml[2 * split_dim], pml[2 * split_dim + 1])
        perp_pml = min(
            min(pml[2 * d], pml[2 * d + 1]) for d in range(ndim) if d != split_dim
        )
        if split_pml > perp_pml:
            raise ValueError(
                "Unstable configuration: with n_subdomains >= 3 the PML width on "
                f"the split dimension ({split_pml}) must not exceed the smallest "
                f"perpendicular PML width ({perp_pml}); otherwise a PML-corner "
                "mode grows without bound. Use n_subdomains <= 2, split along a "
                "different axis, widen the perpendicular PML, or narrow the "
                "split-dimension PML."
            )

    fd_pad = accuracy // 2
    halo = fd_pad  # one internal step per call (step_ratio == 1 at inner_dt)
    wf_axis = split_dim + 1  # the split axis within the (batched) wavefield

    ranges = _partition(n_split, n_subdomains)
    for (s, e) in ranges:
        if e - s < halo:
            raise ValueError(
                f"Each subdomain must own at least {halo} cells along the split "
                f"axis (accuracy//2); {n_split} cells / {n_subdomains} subdomains "
                "is too fine."
            )

    # Determine each shot's source/receiver count and time steps.
    n_shots = 0
    for loc in (source_locations, receiver_locations):
        if loc is not None:
            n_shots = loc.shape[0]
    if source_amplitudes is not None:
        nt_user = source_amplitudes.shape[-1] if nt is None else nt
        # Global, one-shot-up-front resampling to the internal time step.
        source_inner = upsample(source_amplitudes, step_ratio)
    else:
        if nt is None:
            raise ValueError("nt must be given when source_amplitudes is None.")
        nt_user = nt
        source_inner = None
    nt_inner = nt_user * step_ratio

    # Build per-subdomain models, per-face PML, and global<->local geometry.
    subs = []
    for k, (start, end) in enumerate(ranges):
        left_halo = halo if k > 0 else 0
        right_halo = halo if k < n_subdomains - 1 else 0
        lo = start - left_halo  # model start along split axis (global)
        hi = end + right_halo  # model end (exclusive)
        sub_v = v.narrow(split_dim, lo, hi - lo)
        sub_pml = list(pml)
        # Interior split faces are absorbed by halo exchange, not PML.
        if k > 0:
            sub_pml[2 * split_dim] = 0
        if k < n_subdomains - 1:
            sub_pml[2 * split_dim + 1] = 0
        pad_b = sub_pml[2 * split_dim]  # wavefield padding before model (split)
        # Padded wavefield shape (model + per-face PML); seed zero leapfrog state
        # so ndim is inferable even for a source/receiver-free slab on step 0.
        wf_shape = [
            ((hi - lo) if d == split_dim else model_shape[d])
            + sub_pml[2 * d]
            + sub_pml[2 * d + 1]
            for d in range(ndim)
        ]
        state = {name: None for name in scalar_wavefield_names(ndim)}
        state["wavefield_0"] = torch.zeros((max(n_shots, 1), *wf_shape), dtype=v.dtype)
        state["wavefield_m1"] = torch.zeros((max(n_shots, 1), *wf_shape), dtype=v.dtype)
        subs.append(
            {
                "v": sub_v,
                "pml": sub_pml,
                "start": start,
                "end": end,
                "lo": lo,
                "pad_b": pad_b,
                "state": state,
            }
        )

    def route_locations(locations: torch.Tensor, k: int) -> torch.Tensor:
        """Global locations -> this slab's local indices; foreign pts ignored."""
        start, end, lo = subs[k]["start"], subs[k]["end"], subs[k]["lo"]
        loc = locations.clone()
        split_coord = locations[..., split_dim]
        owned = (split_coord >= start) & (split_coord < end)
        owned &= split_coord != IGNORE_LOCATION
        loc[..., split_dim] = torch.where(
            owned, split_coord - lo, torch.full_like(split_coord, IGNORE_LOCATION)
        )
        # Mark every coordinate of a foreign point as ignored.
        for d in range(ndim):
            if d != split_dim:
                loc[..., d] = torch.where(
                    owned, loc[..., d], torch.full_like(loc[..., d], IGNORE_LOCATION)
                )
        return loc

    def route(locations: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:
        """Per-subdomain routed locations; ``None`` where a slab owns no point."""
        if locations is None:
            return [None] * n_subdomains
        routed = []
        for k in range(n_subdomains):
            loc = route_locations(locations, k)
            # A slab owning no active point globally would crash the in-extents
            # check (empty min) -> pass None and skip it.
            owns_any = bool((loc[..., split_dim] != IGNORE_LOCATION).any())
            routed.append(loc if owns_any else None)
        return routed

    sub_src_loc = route(source_locations)
    sub_rec_loc = route(receiver_locations)

    state_names = scalar_wavefield_names(ndim)
    rec_per_step: List[torch.Tensor] = []

    for t in range(nt_inner):
        new_states = []
        outputs = []
        for k in range(n_subdomains):
            sub = subs[k]
            kw = dict(sub["state"])  # wavefield_0/m1/psi*/zeta* by kwarg name
            if source_inner is not None and sub_src_loc[k] is not None:
                kw["source_amplitudes"] = source_inner[..., t : t + 1]
                kw["source_locations"] = sub_src_loc[k]
            if sub_rec_loc[k] is not None:
                kw["receiver_locations"] = sub_rec_loc[k]
            out = deepwave.scalar(
                sub["v"],
                grid_spacing,
                inner_dt,
                accuracy=accuracy,
                pml_width=sub["pml"],
                pml_freq=pml_freq,
                max_vel=max_vel,
                nt=1,
                origin=[0] * ndim,  # locations are slab-local; silences extent warning
                return_dataclass=True,
                **kw,
            )
            new_states.append({name: getattr(out, name) for name in state_names})
            outputs.append(out)
        # Halo exchange: overwrite the wavefield_0/wavefield_m1 halos (the only
        # fields read across the seam) with the neighbour's owned cells.
        for k in range(n_subdomains - 1):
            left = new_states[k]
            right = new_states[k + 1]
            for f in ("wavefield_0", "wavefield_m1"):
                lc = left[f].clone()
                rc = right[f].clone()
                wl = lc.shape[wf_axis]
                # left's right halo  <- right's first owned cells [halo, 2*halo)
                lc.narrow(wf_axis, wl - halo, halo).copy_(
                    right[f].narrow(wf_axis, halo, halo)
                )
                # right's left halo   <- left's last owned cells [wl-2h, wl-h)
                rc.narrow(wf_axis, 0, halo).copy_(
                    left[f].narrow(wf_axis, wl - 2 * halo, halo)
                )
                left[f] = lc
                right[f] = rc
        for k in range(n_subdomains):
            subs[k]["state"] = new_states[k]
        if receiver_locations is not None:
            # Each slab returns the full global receiver set with foreign points
            # zeroed; summing over the slabs that own receivers reassembles it.
            step_rec = sum(
                outputs[k].receiver_amplitudes
                for k in range(n_subdomains)
                if sub_rec_loc[k] is not None
            )
            rec_per_step.append(step_rec[..., 0])  # (n_shots, n_receivers)

    # Reconstruct the final field over the model interior (strip all PML padding).
    wavefield = torch.zeros((max(n_shots, 1), *model_shape), dtype=v.dtype)
    other_starts = [pml[2 * d] for d in range(ndim)]  # PML pad before model, per dim
    for k in range(n_subdomains):
        sub = subs[k]
        field = sub["state"]["wavefield_0"]  # (n_shots, *padded_sub_shape)
        owned_len = sub["end"] - sub["start"]
        # Crop split axis to the owned cells.
        crop = field.narrow(wf_axis, sub["pad_b"] + (sub["start"] - sub["lo"]), owned_len)
        # Crop the other (non-split) axes to the model interior.
        for d in range(ndim):
            if d != split_dim:
                crop = crop.narrow(d + 1, other_starts[d], model_shape[d])
        wavefield.narrow(split_dim + 1, sub["start"], owned_len).copy_(crop)

    if receiver_locations is not None:
        # (nt_inner, n_shots, n_receivers) -> downsample to user dt.
        rec = torch.stack(rec_per_step, dim=0)
        receiver_amplitudes = downsample_and_movedim(rec, step_ratio)
    else:
        receiver_amplitudes = torch.empty(0)

    return wavefield, receiver_amplitudes
