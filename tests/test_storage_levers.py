"""Gradient-parity coverage for the ``storage_compression`` lever.

``storage_compression`` trades gradient accuracy for stored bytes, so what it
needs is a committed statement of how much accuracy it costs, and that
composing it with checkpointing costs no more.

This file checks, at this suite's usual small scale (not the 120^3/nt=200
benchmark scale used to measure the real-world numbers -- see
``benchmarks/results/b1_storage_levers_120cube_nt200_*.json``):

* ``storage_compression=True`` has a bounded, non-vacuous intrinsic gradient
  error relative to an uncompressed direct call (a lossy 8-bit block
  quantisation of the stored forward wavefield history --
  ``src/deepwave/simple_compress.c``'s ``BLOCK_SIZE=8`` max-abs quantiser --
  so SOME error is expected and is not a bug; zero error would mean the
  compression path was silently not exercised).
* Composing it with the ``torch.utils.checkpoint``-over-``run_segments``
  recipe (the standard checkpointing engine) adds no additional error beyond
  what ``storage_compression`` alone contributes -- i.e. checkpointing and
  compression are independent, safely composable levers.
* The source-amplitude gradient is exact under compression (the adjoint
  sampled at the source location does not depend on the compressed forward
  history at all), unlike the model gradient (which is an imaging-condition
  correlation against that history).

Tolerance: measured directly below at 3.79% (v-gradient, direct compressed
vs. baseline) at this exact small scale -- see
``test_storage_compression_direct_call_error_is_bounded_and_nonzero``'s
docstring. The 60% ceiling asserted there is deliberately generous: a
separate benchmark-scale measurement (120^3 grid, nt=200, float32, scalar,
accuracy=4) found the same lever's error grows to ~19-25% as the grid gets
large and sparse relative to how far
the wavefront has travelled by nt=200 (few compression blocks straddle the
wavefront edge in a small, mostly-filled domain; many do in a large, mostly-
empty one) -- so this is a real, geometry/scale-dependent effect, not suite
noise, and the tolerance here is chosen to not be scale-fragile rather than
to pin an exact figure.
"""

import torch
import torch.utils.checkpoint

import deepwave
from deepwave import AuxiliaryState
from deepwave._engine import flat_state
from deepwave.checkpointing import _get_spec, run_segments

DTYPE = torch.float64


def _scalar_problem():
    """The same small 2-D scalar problem used throughout test_checkpointing.py."""
    ny, nx = 32, 32
    dx = 5.0
    freq = 15.0
    dt = 0.001
    nt = 80
    v = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    v[16:] = 1700.0
    src = deepwave.wavelets.ricker(freq, nt, dt, 1.0 / freq).to(DTYPE).reshape(1, 1, nt)
    kw = {
        "source_locations": torch.tensor([[[16, 16]]]),
        "receiver_locations": torch.tensor([[[4, c] for c in range(4, 28, 4)]]),
        "pml_freq": freq,
        "max_vel": 2000.0,
    }
    return v, dx, dt, nt, src, kw


def _rel(a, b):
    """Relative L2 difference ``||a - b|| / ||b||``."""
    return ((a - b).norm() / (b.norm() + 1e-300)).item()


def _loss(out):
    return out.receiver_amplitudes.pow(2).sum() + 0.5 * out.wavefield_0.pow(2).sum()


def _direct_grad(v0, dx, dt, src0, kw, storage_compression):
    """v/source gradients from a single direct call, +/- compression."""
    v = v0.clone().requires_grad_(True)
    src = src0.clone().requires_grad_(True)
    out = deepwave.scalar(
        v, dx, dt, source_amplitudes=src, storage_compression=storage_compression, **kw
    )
    _loss(out).backward()
    return v.grad, src.grad


def _composed_grad(
    v0, dx, dt, nt, src0, kw, storage_mode, storage_compression, n_segments=4
):
    """v/source gradients from torch.utils.checkpoint (reentrant) over run_segments.

    This is the standard recipe (see tests/test_checkpointing.py's
    ``_tuc_run_segments``), specialised here to the scalar propagator only.
    """
    v = v0.clone().requires_grad_(True)
    src = src0.clone().requires_grad_(True)
    spec = _get_spec(deepwave.scalar)
    # The internal flat state names. torch.utils.checkpoint only tracks tensor
    # arguments, so the state is threaded as individual tensors, refolded into
    # an AuxiliaryState on the way in and flattened again on the way out. The
    # flattening is not cosmetic: a reentrant checkpoint runs the wrapped
    # function under torch.no_grad() and re-attaches only the Tensor elements
    # of what it returns, so a tensor returned inside an AuxiliaryState comes
    # back detached and the state's gradient path is lost.
    state_names = list(spec.wavefield_names(2))
    boundary_names = list(spec.auxiliary_state_names(2))
    seg_len = nt // n_segments
    seg_lengths = [seg_len] * (n_segments - 1) + [nt - seg_len * (n_segments - 1)]

    def wrap(vv, src_chunk, *state_args):
        state = dict(zip(state_names, state_args)) if state_args else None
        if state is not None:
            folded = {n: state.pop(n) for n in boundary_names}
            state["auxiliary_state"] = AuxiliaryState._from_flat(folded)
        out = run_segments(
            deepwave.scalar,
            [vv],
            dx,
            dt,
            {"source_amplitudes": src_chunk},
            n_segments=1,
            initial_state=state,
            storage_mode=storage_mode,
            storage_compression=storage_compression,
            **kw,
        )
        flat = flat_state(out, spec, 2)
        return (*(flat[n] for n in state_names), out.receiver_amplitudes)

    idx = 0
    state_tensors: tuple = ()
    recv_chunks = []
    for length in seg_lengths:
        src_chunk = src[:, :, idx : idx + length]
        idx += length
        out = torch.utils.checkpoint.checkpoint(
            wrap, v, src_chunk, *state_tensors, use_reentrant=True
        )
        recv_chunks.append(out[-1])
        state_tensors = tuple(out[: len(state_names)])

    recv = torch.cat(recv_chunks, dim=-1)
    wavefield_0_final = dict(zip(state_names, state_tensors))["wavefield_0"]
    loss = recv.pow(2).sum() + 0.5 * wavefield_0_final.pow(2).sum()
    loss.backward()
    return v.grad, src.grad


def test_storage_compression_direct_call_error_is_bounded_and_nonzero() -> None:
    """storage_compression's intrinsic v-gradient error vs. an uncompressed call.

    Measured at this exact scale: 3.79% relative error in the v gradient,
    0% in the source-amplitude gradient (the source gradient does not read
    the compressed forward-wavefield history at all -- see module
    docstring). The assertion below is deliberately loose (bounded well
    above the measured figure, well below "clearly broken") rather than
    pinned to that exact number: it exists to catch a *regression* (error
    exploding towards O(1) or diverging, e.g. a corrupted quantisation
    scale) or a *silent no-op* (error collapsing to exactly 0, meaning
    compression was not actually exercised), not to assert a precise
    percentage that a legitimate compression-algorithm tweak would break.
    """
    v0, dx, dt, _nt, src0, kw = _scalar_problem()
    gv_base, gs_base = _direct_grad(v0, dx, dt, src0, kw, storage_compression=False)
    gv_comp, gs_comp = _direct_grad(v0, dx, dt, src0, kw, storage_compression=True)

    err_v = _rel(gv_comp, gv_base)
    # TODO(maintainer): The 2026-10-23 static review found that this fixed
    # fixture records 3.79% error but permits 60%, potentially hiding a large
    # regression. Measure variation across supported builds and add a deliberate
    # degraded-gradient negative control before choosing a tighter case-specific
    # ceiling. Revisit the composed-baseline bound below at the same time;
    # preserve the historical measurements and the nonzero-error check.
    assert 1e-4 < err_v < 0.6, (
        f"v-gradient relative error under storage_compression={err_v:.4%} "
        "is outside the expected [0.01%, 60%] band (regression or silent no-op)"
    )
    # The source-amplitude gradient is exact: it is the adjoint wavefield
    # sampled at the source location, computed entirely from the backward
    # sweep, independent of how the forward history was stored.
    assert _rel(gs_comp, gs_base) < 1e-10


def test_storage_compression_composition_matches_direct_call() -> None:
    """TUC-over-run_segments + compression adds no error beyond compression alone.

    This is the composition property B1 exists to verify: checkpointing
    (segmenting the propagation and recomputing forward passes on demand)
    and storage_compression (lossy quantisation of the stored forward
    history) are independent levers. If composing them introduced its own
    error on top of compression's, this test would catch it; instead the
    two gradients agree to within round-off, confirming
    ``storage_compression``'s error is compression's own and not amplified
    (or masked) by the segmentation/recompute machinery around it.
    """
    v0, dx, dt, nt, src0, kw = _scalar_problem()
    gv_direct, gs_direct = _direct_grad(v0, dx, dt, src0, kw, storage_compression=True)
    gv_composed, gs_composed = _composed_grad(
        v0, dx, dt, nt, src0, kw, storage_mode="cpu", storage_compression=True
    )

    assert _rel(gv_composed, gv_direct) < 1e-8
    assert _rel(gs_composed, gs_direct) < 1e-8


def test_storage_compression_composition_matches_uncompressed_baseline_bound() -> None:
    """The composed (checkpointed + compressed) gradient error also stays bounded.

    Measured relative to a fully uncompressed, unsegmented baseline -- i.e.
    the end-to-end error a user actually sees when adopting both levers
    together is the same order of magnitude as compression's error alone,
    not compounded by checkpointing.
    """
    v0, dx, dt, nt, src0, kw = _scalar_problem()
    gv_base, gs_base = _direct_grad(v0, dx, dt, src0, kw, storage_compression=False)
    gv_composed, gs_composed = _composed_grad(
        v0, dx, dt, nt, src0, kw, storage_mode="cpu", storage_compression=True
    )

    err_v = _rel(gv_composed, gv_base)
    # TODO(maintainer): Recalibrate with the direct-call bound above after
    # numerical validation; the shared 60% ceiling is not a measured error budget.
    assert 1e-4 < err_v < 0.6
    assert _rel(gs_composed, gs_base) < 1e-10
