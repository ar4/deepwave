"""Tests for deepwave.checkpointing.

Covers, in order:

* The exactness conditions the whole design rests on: a naive segmented
  propagation threaded through ``initial_state`` is bit-exact iff
  ``step_ratio == 1``; the dynamics are exact across a boundary at any
  ``step_ratio`` (only the source resampling breaks it).
* Gradient equality: driving a propagator through
  ``run_segments(checkpoint=True)`` (each segment but the last
  wrapped in ``torch.utils.checkpoint.checkpoint``) gives the same gradient as
  an unsegmented direct call -- for scalar (2-D and 3-D), acoustic, elastic,
  scalar_born, viscoacoustic, elastic_vti, and all three Born propagators
  (scalar_born, acoustic_born, elastic_born), each exercising the full
  two-field adjoint (background model gradients plus scattering-model
  gradients) plus the source gradient. This is the standard, recommended way
  to bound Deepwave's backward-pass memory for a long propagation; see
  ``docs/example_checkpointing.rst``.
* Automatic whole-trace resampling: forward and gradient parity with an
  unsegmented call at ``step_ratio > 1``, the ``max_vel`` requirement it
  brings, and the manual outer-loop pattern that remains the caller's job when
  the caller owns the segment loop.
* ``run_segments`` forward equivalence and its per-boundary callback.
* Input validation shared by ``run_segments``/``snapshots`` (kwarg rejection,
  the checkpointing-parameter guards, ...).
* Differentiable intermediate snapshots (``snapshots``).
"""

import dataclasses
import inspect
import warnings

import pytest
import torch
import torch.utils.checkpoint
from helpers import elements_equal

import deepwave
from deepwave import (
    AuxiliaryState,
    acoustic,
    acoustic_born,
    elastic,
    elastic_born,
    elastic_vti,
    scalar,
    scalar_born,
    viscoacoustic,
)
from deepwave._engine import _grid_padding, _pml_region, flat_state
from deepwave._exceptions import DeepwaveTypeError, DeepwaveValueError
from deepwave.checkpointing import (
    _SEGMENT_DRIVER_RESERVED,
    SnapshotResult,
    _any_requires_grad,
    _build_slab_plan,
    _compress_state,
    _decompress_state,
    _get_spec,
    _ndim_of,
    _reject_storage_mode_kwarg,
    _SegmentResampling,
    run_segments,
    snapshots,
)
from deepwave.discretisation import cfl_condition_n
from deepwave.elastic import elastic_pml_slab_axes

DTYPE = torch.float64

# The gradient-parity ("TUC over run_segments") tests below run on whichever
# device is visible: CPU by default, or CUDA if available -- e.g. under
# CUDA_VISIBLE_DEVICES=0 / =1, to exercise each card's compiled kernels
# individually. Problem-construction helpers (_scalar_problem and friends)
# deliberately build plain CPU tensors, as elsewhere in the suite; _grad_device
# / _grad_tuc / _tuc_run_segments move them to _DEVICE (models, sources, and
# any tensor-valued kwargs such as location tensors) before running.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_device(x):
    """Moves a tensor to the module's compute device; passes non-tensors through."""
    return x.to(_DEVICE) if isinstance(x, torch.Tensor) else x


def _kw_to_device(kw):
    """Moves every tensor value in a propagator-kwargs dict to the compute device."""
    return {k: _to_device(v) for k, v in kw.items()}


def _ricker(freq, nt, dt):
    """A batched Ricker wavelet tensor of shape ``[1, 1, nt]``."""
    return (
        deepwave.wavelets.ricker(freq, nt, dt, 1.0 / freq).to(DTYPE).reshape(1, 1, nt)
    )


def _scalar_problem(nt=80, dt=0.001):
    """A small 2-D scalar test problem (model, dx, dt, source, kwargs)."""
    ny, nx = 32, 32
    dx = 5.0
    freq = 15.0
    v = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    v[16:] = 1700.0
    src = _ricker(freq, nt, dt)
    kw = {
        "source_locations": torch.tensor([[[16, 16]]]),
        "receiver_locations": torch.tensor([[[4, c] for c in range(4, 28, 4)]]),
        "pml_freq": freq,
        "max_vel": 2000.0,
    }
    return v, dx, dt, src, kw


def _rel(a, b):
    """Relative L2 difference ``||a - b|| / ||b||``."""
    return ((a - b).norm() / (b.norm() + 1e-300)).item()


# ---------------------------------------------------------------------------
# MILESTONE 1: exactness conditions
# ---------------------------------------------------------------------------


def test_segmentation_bit_exact_at_step_ratio_1():
    """Two segments threading full state reproduce a single run bitwise."""
    v, dx, dt, src, kw = _scalar_problem(nt=100, dt=0.001)
    full = scalar(v, dx, dt, source_amplitudes=src, **kw)
    nt1 = 43  # deliberately not a "round" boundary
    seg1 = scalar(v, dx, dt, source_amplitudes=src[..., :nt1], **kw)
    seg2 = scalar(
        v,
        dx,
        dt,
        source_amplitudes=src[..., nt1:],
        initial_state=seg1,
        **kw,
    )
    assert torch.equal(full.wavefield_0, seg2.wavefield_0)
    assert torch.equal(full.wavefield_m1, seg2.wavefield_m1)
    rec = torch.cat([seg1.receiver_amplitudes, seg2.receiver_amplitudes], dim=-1)
    assert torch.equal(full.receiver_amplitudes, rec)


def test_segmentation_not_exact_at_step_ratio_gt_1():
    """At step_ratio > 1 the source resampling breaks boundary exactness."""
    ny, nx = 32, 32
    dx = 5.0
    freq = 15.0
    dt = 0.004  # large dt -> internal sub-stepping (step_ratio > 1)
    nt = 60
    v = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    src = _ricker(freq, nt, dt)
    kw = {
        "source_locations": torch.tensor([[[16, 16]]]),
        "receiver_locations": torch.tensor([[[4, 4]]]),
        "pml_freq": freq,
        "max_vel": 2000.0,
    }
    full = scalar(v, dx, dt, source_amplitudes=src, **kw)
    nt1 = 30
    seg1 = scalar(v, dx, dt, source_amplitudes=src[..., :nt1], **kw)
    seg2 = scalar(
        v,
        dx,
        dt,
        source_amplitudes=src[..., nt1:],
        initial_state=seg1,
        **kw,
    )
    diff = (full.wavefield_0 - seg2.wavefield_0).abs().max().item()
    assert diff > 1e-6  # materially different, so checkpoint mode must reject it


def test_source_free_continuation_exact_at_any_step_ratio():
    """Without a source, the dynamics thread exactly even at step_ratio > 1."""
    ny, nx = 32, 32
    dx = 5.0
    freq = 15.0
    dt = 0.004
    v = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    kw = {
        "source_locations": torch.tensor([[[16, 16]]]),
        "receiver_locations": torch.tensor([[[4, 4]]]),
        "pml_freq": freq,
        "max_vel": 2000.0,
    }
    warm = scalar(v, dx, dt, source_amplitudes=_ricker(freq, 40, dt), **kw)
    nt = 60
    zeros = torch.zeros(1, 1, nt, dtype=DTYPE)
    full = scalar(
        v,
        dx,
        dt,
        source_amplitudes=zeros,
        initial_state=warm,
        **kw,
    )
    for nt1 in (30, 37):  # boundary need not align with anything
        s1 = scalar(
            v,
            dx,
            dt,
            source_amplitudes=zeros[..., :nt1],
            initial_state=warm,
            **kw,
        )
        s2 = scalar(
            v,
            dx,
            dt,
            source_amplitudes=zeros[..., nt1:],
            initial_state=s1,
            **kw,
        )
        assert torch.equal(full.wavefield_0, s2.wavefield_0)


# ---------------------------------------------------------------------------
# Gradient equality: torch.utils.checkpoint over run_segments vs a direct call
# ---------------------------------------------------------------------------


def _grad_device(
    propagator, model_args, dx, dt, sources, kw, lossfn, optional_model_args=None
):
    """Model and source gradients from a single storage_mode='device' run.

    ``optional_model_args`` (e.g. ``{"dv": dv, "drho": drho}``) are the A4
    keyword-only optional model tensors (positioned after grid_spacing/dt),
    unlike ``model_args`` (required, positional). Both get a fresh
    ``requires_grad_`` leaf clone, moved to ``_DEVICE`` (as are ``sources``
    and any tensor-valued entries of ``kw``, e.g. location tensors).
    """
    kw = _kw_to_device(kw)
    ms = [m.clone().to(_DEVICE).requires_grad_(True) for m in model_args]
    oms = (
        {}
        if optional_model_args is None
        else {
            k: v.clone().to(_DEVICE).requires_grad_(True)
            for k, v in optional_model_args.items()
        }
    )
    ss = {k: v.clone().to(_DEVICE).requires_grad_(True) for k, v in sources.items()}
    out = propagator(*ms, dx, dt, **oms, **ss, storage_mode="device", **kw)
    lossfn(out).backward()
    return (
        [m.grad for m in ms],
        {k: v.grad for k, v in ss.items()},
        {k: v.grad for k, v in oms.items()},
    )


def _tuc_run_segments(
    propagator,
    model_args,
    dx,
    dt,
    source_amplitudes,
    kw,
    n_segments,
    optional_model_args=None,
    initial_state=None,
    checkpoint_device=None,
    use_reentrant=True,
    slab_pml=True,
):
    """Runs a propagator through TUC-checkpointed segments of ``run_segments``.

    This is the standard recipe for bounding Deepwave's backward-pass memory:
    split the propagation into ``n_segments`` contiguous segments, thread each
    segment's output state into the next, and wrap every segment except the
    last in ``torch.utils.checkpoint.checkpoint``. The recipe is implemented
    inside the library, by ``run_segments(checkpoint=True)``, so this
    helper is now a thin adapter that lets the whole parity matrix below
    exercise the shipped implementation with the argument shape the tests
    already use.

    ``model_args``/``source_amplitudes``/``optional_model_args``/
    ``initial_state`` must already be on the intended compute device (the
    two callers below, ``_grad_tuc`` and the standalone tests that call this
    directly, are responsible for that); tensor-valued entries of ``kw``
    (e.g. location tensors) are moved to ``_DEVICE`` here.

    Returns the combined :class:`~deepwave.PropagatorOutput` for the whole
    run (final state, time-concatenated receiver amplitudes), exactly like a
    direct call.
    """
    kw = _kw_to_device(kw)
    return run_segments(
        propagator,
        list(model_args),
        dx,
        dt,
        source_amplitudes,
        n_segments=n_segments,
        initial_state=initial_state,
        optional_model_args=optional_model_args,
        checkpoint=True,
        use_reentrant=use_reentrant,
        checkpoint_device=checkpoint_device,
        _slab_pml=slab_pml,
        storage_mode="device",
        **kw,
    )


def _grad_tuc(
    propagator,
    model_args,
    dx,
    dt,
    sources,
    kw,
    lossfn,
    n_segments,
    optional_model_args=None,
    initial_state=None,
    checkpoint_device=None,
    use_reentrant=True,
    slab_pml=True,
):
    """Model and source gradients from a TUC-checkpointed segmented run.

    See :func:`_grad_device` for ``optional_model_args`` and the ``_DEVICE``
    move; see :func:`_tuc_run_segments` for the segmenting/checkpointing
    recipe. ``initial_state``, if given, must already be on ``_DEVICE`` (the
    caller builds it, typically with a direct call on device-resident
    tensors).
    """
    ms = [m.clone().to(_DEVICE).requires_grad_(True) for m in model_args]
    oms = (
        {}
        if optional_model_args is None
        else {
            k: v.clone().to(_DEVICE).requires_grad_(True)
            for k, v in optional_model_args.items()
        }
    )
    ss = {k: v.clone().to(_DEVICE).requires_grad_(True) for k, v in sources.items()}
    out = _tuc_run_segments(
        propagator,
        ms,
        dx,
        dt,
        ss,
        kw,
        n_segments,
        optional_model_args=(oms or None),
        initial_state=initial_state,
        checkpoint_device=checkpoint_device,
        use_reentrant=use_reentrant,
        slab_pml=slab_pml,
    )
    lossfn(out).backward()
    return (
        [m.grad for m in ms],
        {k: v.grad for k, v in ss.items()},
        {k: v.grad for k, v in oms.items()},
    )


@pytest.mark.parametrize("n_segments", [1, 2, 4, 8, 16, 10])
def test_scalar_2d_grad_matches_device(n_segments):
    """Scalar 2-D model and source gradients match device mode to round-off."""
    v, dx, dt, src, kw = _scalar_problem(nt=96, dt=0.001)
    sources = {"source_amplitudes": src}

    def loss(o):
        return o.receiver_amplitudes.pow(2).sum() + 0.5 * o.wavefield_0.pow(2).sum()

    gm_d, gs_d, _ = _grad_device(scalar, [v], dx, dt, sources, kw, loss)
    gm_c, gs_c, _ = _grad_tuc(scalar, [v], dx, dt, sources, kw, loss, n_segments)
    assert _rel(gm_c[0], gm_d[0]) < 1e-12
    assert _rel(gs_c["source_amplitudes"], gs_d["source_amplitudes"]) < 1e-12


def test_scalar_3d_grad_matches_device():
    """Scalar 3-D model and source gradients match device mode to round-off."""
    nz = ny = nx = 14
    dx = 5.0
    freq = 15.0
    dt = 0.0005  # step_ratio == 1 in 3-D
    nt = 50
    v = torch.full((nz, ny, nx), 1500.0, dtype=DTYPE)
    src = _ricker(freq, nt, dt)
    kw = {
        "source_locations": torch.tensor([[[7, 7, 7]]]),
        "receiver_locations": torch.tensor([[[3, 3, 3], [10, 10, 10]]]),
        "pml_freq": freq,
        "max_vel": 2000.0,
    }
    sources = {"source_amplitudes": src}

    def loss(o):
        return o.receiver_amplitudes.pow(2).sum() + 0.4 * o.wavefield_0.pow(2).sum()

    gm_d, gs_d, _ = _grad_device(scalar, [v], dx, dt, sources, kw, loss)
    gm_c, gs_c, _ = _grad_tuc(scalar, [v], dx, dt, sources, kw, loss, 6)
    assert _rel(gm_c[0], gm_d[0]) < 1e-12
    assert _rel(gs_c["source_amplitudes"], gs_d["source_amplitudes"]) < 1e-12


def test_checkpoint_device_cpu_matches_device():
    """Holding the between-segment state on the CPU does not change gradients.

    Generic equivalent of the deleted custom ``checkpoint()``'s
    ``checkpoint_device`` parameter: :func:`_tuc_run_segments`'s
    ``checkpoint_device`` moves the state threaded between segments onto the
    given device (here ``"cpu"``, freeing GPU/compute-device memory between
    checkpointed calls) and moves it back before each segment runs. Since
    device transfers are themselves differentiable, this is a strict subset
    of ordinary autograd and does not change the gradient.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=80, dt=0.001)
    sources = {"source_amplitudes": src}

    def loss(o):
        return o.receiver_amplitudes.pow(2).sum()

    gm_d, gs_d, _ = _grad_device(scalar, [v], dx, dt, sources, kw, loss)
    gm_c, gs_c, _ = _grad_tuc(
        scalar, [v], dx, dt, sources, kw, loss, 8, checkpoint_device="cpu"
    )
    assert _rel(gm_c[0], gm_d[0]) < 1e-12
    assert _rel(gs_c["source_amplitudes"], gs_d["source_amplitudes"]) < 1e-12


def test_initial_state_continuation_grad_matches_device():
    """Continuing a segmented, checkpointed propagation matches device mode.

    ``warm`` (a detached, no-grad run) is threaded in as the auxiliary state
    for the very first checkpointed segment, exactly as
    ``test_initial_state_continuation_with_survey_pad_recovers_origin`` and
    friends thread it into a plain (unsegmented) continuation.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=80, dt=0.001)
    v = v.to(_DEVICE)
    src = src.to(_DEVICE)
    kw = _kw_to_device(kw)
    with torch.no_grad():
        warm = scalar(
            v,
            dx,
            dt,
            source_amplitudes=_ricker(15.0, 40, dt).to(_DEVICE),
            **kw,
        )

    def loss(o):
        return o.receiver_amplitudes.pow(2).sum()

    ms = [v.clone().requires_grad_(True)]
    ss = {"source_amplitudes": src.clone().requires_grad_(True)}
    out = scalar(
        *ms,
        dx,
        dt,
        **ss,
        initial_state=warm,
        storage_mode="device",
        **kw,
    )
    loss(out).backward()
    gm_d = [m.grad for m in ms]
    gs_d = {k: t.grad for k, t in ss.items()}
    gm_c, gs_c, _ = _grad_tuc(
        scalar, [v], dx, dt, {"source_amplitudes": src}, kw, loss, 8, initial_state=warm
    )
    assert _rel(gm_c[0], gm_d[0]) < 1e-12
    assert _rel(gs_c["source_amplitudes"], gs_d["source_amplitudes"]) < 1e-12


def _scalar_survey_pad_problem(nt, dt=0.001):
    """A 2-D scalar problem with sources/receivers far from the model origin.

    A ``survey_pad`` extraction is then centred well away from [0, 0], so a
    continuation that silently assumed origin zero would extract the wrong
    region.
    """
    ny, nx = 80, 80
    dx = 5.0
    freq = 15.0
    v = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    v[40:] = 1700.0
    src = _ricker(freq, nt, dt)
    kw = {
        "source_locations": torch.tensor([[[55, 55]]]),
        "receiver_locations": torch.tensor([[[50, c] for c in range(50, 70, 4)]]),
        "pml_freq": freq,
        "max_vel": 2000.0,
        "survey_pad": 6,
    }
    return v, dx, dt, src, kw


def test_initial_state_continuation_with_survey_pad_recovers_origin():
    """A continuation seeded from a ``survey_pad``-extracted run works.

    It recovers the same survey origin from the ``RunInfo`` carried by
    ``initial_state``, even though neither ``survey_pad`` nor ``origin`` is
    passed on the second call, reproducing a single unsegmented run over the
    combined time range.
    """
    v, dx, dt, src, kw = _scalar_survey_pad_problem(nt=100)
    full = scalar(v, dx, dt, source_amplitudes=src, **kw)
    nt1 = 43  # deliberately not a "round" boundary
    seg1 = scalar(v, dx, dt, source_amplitudes=src[..., :nt1], **kw)
    # Second call deliberately omits survey_pad and origin: correctness here
    # depends entirely on initial_state's RunInfo carrying the origin that
    # survey_pad resolved to on the first call.
    kw_no_pad = {k: v for k, v in kw.items() if k != "survey_pad"}
    seg2 = scalar(
        v,
        dx,
        dt,
        source_amplitudes=src[..., nt1:],
        initial_state=seg1,
        **kw_no_pad,
    )
    assert torch.equal(full.wavefield_0, seg2.wavefield_0)
    assert torch.equal(full.wavefield_m1, seg2.wavefield_m1)
    rec = torch.cat([seg1.receiver_amplitudes, seg2.receiver_amplitudes], dim=-1)
    assert torch.equal(full.receiver_amplitudes, rec)


def test_initial_state_continuation_with_survey_pad_reapplied_still_works():
    """Re-passing the same ``survey_pad`` on the continuation call must work.

    This guards against the origin recovered from ``RunInfo`` ever being
    combined with an explicitly re-passed ``survey_pad``, which would
    otherwise make ``extract_survey`` see both ``origin`` and ``survey_pad``
    set and raise "survey_pad and origin cannot both be specified."
    """
    v, dx, dt, src, kw = _scalar_survey_pad_problem(nt=100)
    full = scalar(v, dx, dt, source_amplitudes=src, **kw)
    nt1 = 43
    seg1 = scalar(v, dx, dt, source_amplitudes=src[..., :nt1], **kw)
    seg2 = scalar(
        v,
        dx,
        dt,
        source_amplitudes=src[..., nt1:],
        initial_state=seg1,
        **kw,  # survey_pad passed again explicitly
    )
    assert torch.equal(full.wavefield_0, seg2.wavefield_0)
    rec = torch.cat([seg1.receiver_amplitudes, seg2.receiver_amplitudes], dim=-1)
    assert torch.equal(full.receiver_amplitudes, rec)


def test_run_segments_tuc_allows_differentiable_initial_state():
    """A ``requires_grad`` ``initial_state`` composes fine with this recipe.

    The deleted custom ``checkpoint()`` hard-rejected a ``requires_grad``
    ``initial_state`` (its Revolve backward re-ran segments from detached
    checkpoints, so no gradient could flow back into it). ``run_segments``
    (and hence this ``torch.utils.checkpoint``-based recipe) has no such
    restriction: it simply passes ``initial_state`` straight through to the
    propagator, so plain autograd threads a gradient back into it exactly as
    it would for any other differentiable input -- a genuine capability
    improvement over the deleted module, not merely a smaller feature set.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=60, dt=0.001)
    kw = _kw_to_device(kw)
    warm_v = v.to(_DEVICE).clone().requires_grad_(True)
    warm = scalar(
        warm_v,
        dx,
        dt,
        source_amplitudes=_ricker(15.0, 20, dt).to(_DEVICE),
        **kw,
    )

    def loss(o):
        return o.receiver_amplitudes.pow(2).sum()

    ss = {"source_amplitudes": src.to(_DEVICE).clone().requires_grad_(True)}
    out = _tuc_run_segments(scalar, [warm_v], dx, dt, ss, kw, 4, initial_state=warm)
    loss(out).backward()
    # warm_v received a gradient from two paths: directly (as the model of
    # every segment) and indirectly (through warm, the boundary condition of
    # the first segment) -- so this would fail loudly (None, or a much
    # smaller/wrong value) if the initial_state path were silently detached.
    assert warm_v.grad is not None
    assert torch.isfinite(warm_v.grad).all()
    assert warm_v.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Propagator-agnostic: acoustic, elastic, scalar_born, viscoacoustic, ...
# ---------------------------------------------------------------------------


def test_acoustic_grad_matches_device():
    """Acoustic (two models, pressure source) gradients match device mode."""
    ny, nx = 30, 30
    dx = 5.0
    freq = 15.0
    dt = 0.0005
    nt = 70
    v = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    rho = torch.full((ny, nx), 2000.0, dtype=DTYPE)
    src = _ricker(freq, nt, dt)
    kw = {
        "source_locations_volume_rate": torch.tensor([[[15, 15]]]),
        "receiver_locations_p": torch.tensor([[[4, 4], [24, 24]]]),
        "pml_freq": freq,
        "max_vel": 2000.0,
    }
    sources = {"source_amplitudes_volume_rate": src}

    def loss(o):
        return o.receiver_amplitudes_p.pow(2).sum() + 0.3 * o.pressure_0.pow(2).sum()

    gm_d, gs_d, _ = _grad_device(acoustic, [v, rho], dx, dt, sources, kw, loss)
    gm_c, gs_c, _ = _grad_tuc(acoustic, [v, rho], dx, dt, sources, kw, loss, 7)
    assert _rel(gm_c[0], gm_d[0]) < 1e-12
    assert _rel(gm_c[1], gm_d[1]) < 1e-12
    assert (
        _rel(
            gs_c["source_amplitudes_volume_rate"], gs_d["source_amplitudes_volume_rate"]
        )
        < 1e-12
    )


def test_elastic_grad_matches_device():
    """Elastic (three Lame models, y source) gradients match device mode."""
    ny, nx = 30, 30
    dx = 5.0
    freq = 15.0
    dt = 0.0005
    nt = 70
    vp = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    vs = torch.full((ny, nx), 800.0, dtype=DTYPE)
    rho = torch.full((ny, nx), 2000.0, dtype=DTYPE)
    lamb = (vp**2 - 2 * vs**2) * rho
    mu = vs**2 * rho
    buoyancy = 1.0 / rho
    src = _ricker(freq, nt, dt)
    kw = {
        "source_locations_y": torch.tensor([[[15, 15]]]),
        "receiver_locations_y": torch.tensor([[[4, 4], [24, 24]]]),
        "pml_freq": freq,
        "max_vel": 1600.0,
    }
    sources = {"source_amplitudes_y": src}

    def loss(o):
        return o.receiver_amplitudes_y.pow(2).sum()

    gm_d, gs_d, _ = _grad_device(
        elastic, [lamb, mu, buoyancy], dx, dt, sources, kw, loss
    )
    gm_c, gs_c, _ = _grad_tuc(
        elastic, [lamb, mu, buoyancy], dx, dt, sources, kw, loss, 7
    )
    for a, b in zip(gm_c, gm_d):
        assert _rel(a, b) < 1e-12
    assert _rel(gs_c["source_amplitudes_y"], gs_d["source_amplitudes_y"]) < 1e-12


def test_scalar_born_grad_matches_device():
    """scalar_born (v + scatter, two receiver sets) gradients match device."""
    ny, nx = 30, 30
    dx = 5.0
    freq = 15.0
    dt = 0.001
    nt = 70
    v = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    scatter = torch.zeros((ny, nx), dtype=DTYPE)
    scatter[15, 15] = 100.0
    src = _ricker(freq, nt, dt)
    kw = {
        "source_locations": torch.tensor([[[15, 15]]]),
        "sc_receiver_locations": torch.tensor([[[4, 4], [24, 24]]]),
        "pml_freq": freq,
        "max_vel": 2000.0,
    }
    sources = {"source_amplitudes": src}

    def loss(o):
        return (
            o.sc_receiver_amplitudes.pow(2).sum()
            + o.bg_receiver_amplitudes.pow(2).sum()
        )

    gm_d, gs_d, gom_d = _grad_device(
        scalar_born, [v], dx, dt, sources, kw, loss, optional_model_args={"dv": scatter}
    )
    gm_c, gs_c, gom_c = _grad_tuc(
        scalar_born,
        [v],
        dx,
        dt,
        sources,
        kw,
        loss,
        7,
        optional_model_args={"dv": scatter},
    )
    for a, b in zip(gm_c, gm_d):
        assert _rel(a, b) < 1e-12
    assert _rel(gs_c["source_amplitudes"], gs_d["source_amplitudes"]) < 1e-12
    assert _rel(gom_c["dv"], gom_d["dv"]) < 1e-12


def test_viscoacoustic_grad_matches_device():
    """Viscoacoustic (v, rho, Q models, pressure source) gradients match device."""
    ny, nx = 30, 30
    dx = 5.0
    freq = 15.0
    dt = 0.0005
    nt = 70
    v = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    rho = torch.full((ny, nx), 2000.0, dtype=DTYPE)
    q = torch.full((ny, nx), 30.0, dtype=DTYPE)
    src = _ricker(freq, nt, dt)
    kw = {
        "source_locations_pressure_rate": torch.tensor([[[15, 15]]]),
        "receiver_locations_p": torch.tensor([[[4, 4], [24, 24]]]),
        "pml_freq": freq,
        "q_freq": freq,
        "max_vel": 2000.0,
    }
    sources = {"source_amplitudes_pressure_rate": src}

    def loss(o):
        return o.receiver_amplitudes_p.pow(2).sum()

    gm_d, gs_d, _ = _grad_device(viscoacoustic, [v, rho, q], dx, dt, sources, kw, loss)
    gm_c, gs_c, _ = _grad_tuc(viscoacoustic, [v, rho, q], dx, dt, sources, kw, loss, 7)
    for a, b in zip(gm_c, gm_d):
        assert _rel(a, b) < 1e-12
    assert (
        _rel(
            gs_c["source_amplitudes_pressure_rate"],
            gs_d["source_amplitudes_pressure_rate"],
        )
        < 1e-12
    )


def test_elastic_vti_grad_matches_device():
    """elastic_vti (3-D VTI stiffnesses, x source) gradients match device mode.

    Uses 3-D so ``c66`` is a real (required) tensor; ``c66`` is passed via
    ``optional_model_args`` (the A4 keyword-only optional model tensor
    convention), not ``model_args``.
    """
    n = 8
    dx = 5.0
    freq = 15.0
    dt = 0.0005
    nt = 40
    torch.manual_seed(3)
    vp = torch.full((n, n, n), 1500.0, dtype=DTYPE)
    vs = torch.full((n, n, n), 800.0, dtype=DTYPE)
    rho = torch.full((n, n, n), 2000.0, dtype=DTYPE)
    lamb, mu, buoyancy = deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    c11 = lamb + 2 * mu
    c13 = lamb.clone()
    c33 = lamb + 2 * mu
    c55 = mu.clone()
    c66 = mu.clone()
    src = _ricker(freq, nt, dt)
    kw = {
        "source_locations_x": torch.tensor([[[4, 4, 4]]]),
        "receiver_locations_x": torch.tensor([[[2, 2, 2], [6, 6, 6]]]),
        "pml_freq": freq,
        "max_vel": 2200.0,
        "pml_width": 3,
    }
    sources = {"source_amplitudes_x": src}

    def loss(o):
        return o.receiver_amplitudes_x.pow(2).sum()

    model_args = [c11, c13, c33, c55, buoyancy]
    opt = {"c66": c66}
    gm_d, gs_d, gom_d = _grad_device(
        elastic_vti, model_args, dx, dt, sources, kw, loss, optional_model_args=opt
    )
    gm_c, gs_c, gom_c = _grad_tuc(
        elastic_vti, model_args, dx, dt, sources, kw, loss, 5, optional_model_args=opt
    )
    for a, b in zip(gm_c, gm_d):
        assert _rel(a, b) < 1e-10
    assert _rel(gs_c["source_amplitudes_x"], gs_d["source_amplitudes_x"]) < 1e-10
    assert _rel(gom_c["c66"], gom_d["c66"]) < 1e-10


def test_acoustic_born_forward_matches_device():
    """acoustic_born segmented forward (run_segments) matches a direct call.

    Forward equivalence, complementing the gradient parity checked in
    ``test_acoustic_born_grad_matches_device``. This exercises
    ``run_segments`` directly (not checkpointed): checkpointing only affects
    what is stored for the backward pass, not the forward values.
    """
    ny, nx = 30, 30
    dx = 5.0
    freq = 15.0
    dt = 0.0005
    nt = 70
    v = torch.full((ny, nx), 1500.0, dtype=DTYPE, device=_DEVICE)
    rho = torch.full((ny, nx), 2000.0, dtype=DTYPE, device=_DEVICE)
    dv = torch.zeros((ny, nx), dtype=DTYPE, device=_DEVICE)
    drho = torch.zeros((ny, nx), dtype=DTYPE, device=_DEVICE)
    dv[15, 15] = 50.0
    drho[15, 15] = 80.0
    src = _ricker(freq, nt, dt).to(_DEVICE)
    kw = _kw_to_device(
        {
            "source_locations_volume_rate": torch.tensor([[[15, 15]]]),
            "sc_receiver_locations_p": torch.tensor([[[4, 4], [24, 24]]]),
            "pml_freq": freq,
            "max_vel": 2000.0,
        }
    )
    out_d = acoustic_born(
        v,
        rho,
        dx,
        dt,
        dv=dv,
        drho=drho,
        source_amplitudes_volume_rate=src,
        storage_mode="device",
        **kw,
    )
    out_c = run_segments(
        acoustic_born,
        [v, rho],
        dx,
        dt,
        {"source_amplitudes_volume_rate": src},
        n_segments=7,
        dv=dv,
        drho=drho,
        **kw,
    )
    assert _rel(out_c.sc_receiver_amplitudes_p, out_d.sc_receiver_amplitudes_p) < 1e-10
    assert _rel(out_c.bg_receiver_amplitudes_p, out_d.bg_receiver_amplitudes_p) < 1e-10


@pytest.mark.parametrize("n_segments", [1, 3])
def test_acoustic_born_grad_matches_device(n_segments):
    """acoustic_born (full two-field adjoint) gradients match device mode.

    All four model tensors require grad -- the background ``v``/``rho`` and the
    scattering ``dv``/``drho`` -- exercising acoustic_born's full two-field
    (background + scattered) adjoint, together with the shared pressure-source
    gradient. Both the scattered (``receiver_amplitudes_p``) and background
    (``bg_receiver_amplitudes_p``) receiver sets are recorded, and ``nt`` is
    long enough for the scattered field to reach the receivers, so every
    gradient is well above the float64 noise floor. ``n_segments=1`` reduces
    to a single ordinary backward; ``n_segments=3`` threads the two coupled
    wavefields (and their PML auxiliaries) across segment boundaries.

    Regression guard for the fixed contiguity root cause: the compiled
    acoustic_born backward reads the incoming receiver-amplitude gradients
    through a raw ``data_ptr()`` assuming a C-contiguous ``[nt, n_shots,
    n_receivers]`` layout, and segmenting the propagation and concatenating
    each segment's receiver amplitudes (as ``run_segments`` does) can hand it
    a non-contiguous gradient; before the backend forced them contiguous,
    this injected the wrong adjoint source and produced ~100% wrong model
    gradients even at ``n_segments=1``.
    """
    ny, nx = 30, 30
    dx = 5.0
    freq = 15.0
    dt = 0.0005
    nt = 70
    v = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    rho = torch.full((ny, nx), 2000.0, dtype=DTYPE)
    torch.manual_seed(1)
    dv = torch.randn((ny, nx), dtype=DTYPE) * 20.0
    drho = torch.randn((ny, nx), dtype=DTYPE) * 30.0
    src = _ricker(freq, nt, dt)
    kw = {
        "source_locations_volume_rate": torch.tensor([[[15, 15]]]),
        "sc_receiver_locations_p": torch.tensor([[[4, 4], [24, 24], [4, 24]]]),
        "bg_receiver_locations_p": torch.tensor([[[4, 4], [24, 24]]]),
        "pml_freq": freq,
        "max_vel": 2000.0,
    }
    sources = {"source_amplitudes_volume_rate": src}

    def loss(o):
        return (
            o.sc_receiver_amplitudes_p.pow(2).sum()
            + 0.5 * o.bg_receiver_amplitudes_p.pow(2).sum()
        )

    opt = {"dv": dv, "drho": drho}
    gm_d, gs_d, gom_d = _grad_device(
        acoustic_born, [v, rho], dx, dt, sources, kw, loss, optional_model_args=opt
    )
    gm_c, gs_c, gom_c = _grad_tuc(
        acoustic_born,
        [v, rho],
        dx,
        dt,
        sources,
        kw,
        loss,
        n_segments,
        optional_model_args=opt,
    )
    for a, b in zip(gm_c, gm_d):
        assert _rel(a, b) < 1e-10
    assert (
        _rel(
            gs_c["source_amplitudes_volume_rate"], gs_d["source_amplitudes_volume_rate"]
        )
        < 1e-10
    )
    assert _rel(gom_c["dv"], gom_d["dv"]) < 1e-10
    assert _rel(gom_c["drho"], gom_d["drho"]) < 1e-10


def test_acoustic_born_dv_only_grad_matches_device():
    """acoustic_born TUC-segmented gradients match device with only ``dv`` given.

    A4.4: Born perturbations are independently optional; this exercises
    ``optional_model_args`` with a single entry (``drho`` omitted, defaulting
    to zero inside ``acoustic_born``) rather than always all of them together.
    """
    ny, nx = 30, 30
    dx = 5.0
    freq = 15.0
    dt = 0.0005
    nt = 70
    v = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    rho = torch.full((ny, nx), 2000.0, dtype=DTYPE)
    torch.manual_seed(2)
    dv = torch.randn((ny, nx), dtype=DTYPE) * 20.0
    src = _ricker(freq, nt, dt)
    kw = {
        "source_locations_volume_rate": torch.tensor([[[15, 15]]]),
        "sc_receiver_locations_p": torch.tensor([[[4, 4], [24, 24], [4, 24]]]),
        "bg_receiver_locations_p": torch.tensor([[[4, 4], [24, 24]]]),
        "pml_freq": freq,
        "max_vel": 2000.0,
    }
    sources = {"source_amplitudes_volume_rate": src}

    def loss(o):
        return (
            o.sc_receiver_amplitudes_p.pow(2).sum()
            + 0.5 * o.bg_receiver_amplitudes_p.pow(2).sum()
        )

    opt = {"dv": dv}
    gm_d, gs_d, gom_d = _grad_device(
        acoustic_born, [v, rho], dx, dt, sources, kw, loss, optional_model_args=opt
    )
    gm_c, gs_c, gom_c = _grad_tuc(
        acoustic_born,
        [v, rho],
        dx,
        dt,
        sources,
        kw,
        loss,
        3,
        optional_model_args=opt,
    )
    for a, b in zip(gm_c, gm_d):
        assert _rel(a, b) < 1e-10
    assert (
        _rel(
            gs_c["source_amplitudes_volume_rate"], gs_d["source_amplitudes_volume_rate"]
        )
        < 1e-10
    )
    assert _rel(gom_c["dv"], gom_d["dv"]) < 1e-10


def test_elastic_born_grad_matches_device():
    """elastic_born (three Lame scattering models, y source) gradients match.

    Regression guard for A4.6: ``elastic_born`` was previously excluded from
    the checkpointed recipe because its scattering models
    (``dlamb``/``dmu``/``dbuoyancy``) are keyword-only, positioned after
    ``grid_spacing``/``dt`` -- the same A4.4 convention ``scalar_born``'s
    ``dv`` and ``acoustic_born``'s ``dv``/``drho`` follow. ``run_segments``
    forwards them like any other keyword argument (unlike the deleted custom
    ``checkpoint()``, it does not cross a ``torch.autograd.Function``
    boundary of its own, so it needs no special-cased ``optional_model_args``
    threading to keep them differentiable).
    """
    ny, nx = 30, 30
    dx = 5.0
    freq = 15.0
    dt = 0.0005
    nt = 70
    vp = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    vs = torch.full((ny, nx), 800.0, dtype=DTYPE)
    rho = torch.full((ny, nx), 2000.0, dtype=DTYPE)
    lamb = (vp**2 - 2 * vs**2) * rho
    mu = vs**2 * rho
    buoyancy = 1.0 / rho
    torch.manual_seed(4)
    dlamb = torch.randn((ny, nx), dtype=DTYPE) * lamb.mean().item() * 0.01
    dmu = torch.randn((ny, nx), dtype=DTYPE) * mu.mean().item() * 0.01
    dbuoyancy = torch.randn((ny, nx), dtype=DTYPE) * buoyancy.mean().item() * 0.01
    src = _ricker(freq, nt, dt)
    kw = {
        "source_locations_y": torch.tensor([[[15, 15]]]),
        "sc_receiver_locations_y": torch.tensor([[[4, 4], [24, 24]]]),
        "bg_receiver_locations_y": torch.tensor([[[4, 4], [24, 24]]]),
        "pml_freq": freq,
        "max_vel": 1600.0,
    }
    sources = {"source_amplitudes_y": src}

    def loss(o):
        return (
            o.sc_receiver_amplitudes_y.pow(2).sum()
            + 0.5 * o.bg_receiver_amplitudes_y.pow(2).sum()
        )

    opt = {"dlamb": dlamb, "dmu": dmu, "dbuoyancy": dbuoyancy}
    gm_d, gs_d, gom_d = _grad_device(
        elastic_born,
        [lamb, mu, buoyancy],
        dx,
        dt,
        sources,
        kw,
        loss,
        optional_model_args=opt,
    )
    gm_c, gs_c, gom_c = _grad_tuc(
        elastic_born,
        [lamb, mu, buoyancy],
        dx,
        dt,
        sources,
        kw,
        loss,
        3,
        optional_model_args=opt,
    )
    for a, b in zip(gm_c, gm_d):
        assert _rel(a, b) < 1e-10
    assert _rel(gs_c["source_amplitudes_y"], gs_d["source_amplitudes_y"]) < 1e-10
    for name in ("dlamb", "dmu", "dbuoyancy"):
        assert _rel(gom_c[name], gom_d[name]) < 1e-10


# ---------------------------------------------------------------------------
# run_segments forward driver
# ---------------------------------------------------------------------------


def test_run_segments_matches_single_run_and_callback():
    """run_segments reproduces a single run and fires the boundary callback."""
    v, dx, dt, src, kw = _scalar_problem(nt=90, dt=0.001)
    full = scalar(v, dx, dt, source_amplitudes=src, **kw)
    boundaries = []
    combined = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=9,
        segment_callback=lambda cumulative_step, out: boundaries.append(
            cumulative_step
        ),
        **kw,
    )
    assert torch.equal(full.wavefield_0, combined.wavefield_0)
    assert torch.equal(full.receiver_amplitudes, combined.receiver_amplitudes)
    # nt=90 in 9 equal segments -> the callback fires with the running outer-step
    # count 10, 20, ..., 90.
    assert boundaries == list(range(10, 91, 10))


_RESERVED_STORAGE_NAMES = ("storage_mode", "storage_path", "storage_compression")


@pytest.mark.parametrize("name", _RESERVED_STORAGE_NAMES)
def test_reject_storage_mode_kwarg_covers_all_three_reserved_names(name):
    """The shared kwargs-rejection helper covers everything run_segments promises.

    run_segments's own docstring promises ``storage_mode``, ``storage_path``,
    and ``storage_compression`` are all rejected up front if smuggled into its
    forwarded ``**kwargs`` -- it exposes all three as explicit parameters, so
    a well-formed call can never actually land one of these in its internal
    ``kwargs`` (Python resolves them onto the named parameter first), but a
    caller like ``snapshots()``/``measure()`` -- which do not reserve
    ``storage_path``/``storage_compression`` themselves -- could forward such
    a kwargs dict on to it. Only checking ``storage_mode`` would leave that
    path to a raw, unfriendly ``TypeError``.
    """
    kwargs = {"pml_freq": 15.0, name: object()}
    with pytest.raises(ValueError, match=f"controls {name!r}"):
        _reject_storage_mode_kwarg(kwargs, "run_segments", _RESERVED_STORAGE_NAMES)


@pytest.mark.parametrize("name", _SEGMENT_DRIVER_RESERVED)
def test_snapshots_rejects_segment_driver_kwargs(name):
    """snapshots() reserves run_segments' checkpointing parameters.

    They are named parameters of ``run_segments``, which ``snapshots()`` calls
    itself, so smuggling one through ``snapshots()``'s forwarded ``**kwargs``
    would either silently change how the segments run or hit a raw
    ``TypeError`` from a duplicate keyword argument.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    with pytest.raises(ValueError, match=f"controls {name!r}"):
        snapshots(
            scalar, [v], dx, dt, {"source_amplitudes": src}, [10], **{name: True}, **kw
        )


@pytest.mark.parametrize("name", _SEGMENT_DRIVER_RESERVED)
def test_measure_rejects_segment_driver_kwargs(name):
    """measure() reserves the same run_segments checkpointing parameters."""
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    with pytest.raises(ValueError, match=f"controls {name!r}"):
        deepwave.wavefields.measure(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            [deepwave.wavefields.RunningDFT(frequencies=[15.0], on="u")],
            **{name: True},
            **kw,
        )


def _batched_scalar_problem(nt=60, dt=0.001):
    """A 2-D scalar problem with a per-shot batched model (leading shot dim)."""
    ny, nx = 32, 32
    dx = 5.0
    freq = 15.0
    v = torch.stack(
        [
            torch.full((ny, nx), 1500.0, dtype=DTYPE),
            torch.full((ny, nx), 1700.0, dtype=DTYPE),
        ],
    )
    src = _ricker(freq, nt, dt).repeat(2, 1, 1)
    kw = {
        "source_locations": torch.tensor([[[16, 16]], [[16, 16]]]),
        "receiver_locations": torch.tensor([[[4, c] for c in range(4, 28, 4)]] * 2),
        "pml_freq": freq,
        "max_vel": 2000.0,
    }
    return v, dx, dt, src, kw


def test_run_segments_batched_model_matches_single_run():
    """A per-shot batched model segments exactly, like an unbatched one.

    A batched 2-D model has three dimensions, exactly as an unbatched 3-D model
    does, so the number of spatial dimensions is taken from the location tensors
    (as the propagators themselves take it) rather than from the model.
    """
    v, dx, dt, src, kw = _batched_scalar_problem(nt=60, dt=0.001)
    assert v.dim() == 3
    full = scalar(v, dx, dt, source_amplitudes=src, **kw)
    combined = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=5,
        **kw,
    )
    assert torch.equal(full.wavefield_0, combined.wavefield_0)
    assert torch.equal(full.wavefield_m1, combined.wavefield_m1)
    assert torch.equal(full.receiver_amplitudes, combined.receiver_amplitudes)
    # The two shots see different models, so they are genuinely batched.
    assert not torch.allclose(
        combined.receiver_amplitudes[0],
        combined.receiver_amplitudes[1],
    )


def test_ndim_of_uses_locations_not_model_shape():
    """The spatial-dimension count comes from the locations, as in the propagators."""
    from deepwave.checkpointing import _ndim_of

    batched_2d = torch.zeros(2, 8, 8)
    unbatched_3d = torch.zeros(8, 8, 8)
    loc_2d = {"source_locations": torch.zeros(2, 1, 2, dtype=torch.long)}
    loc_3d = {"receiver_locations": torch.zeros(1, 1, 3, dtype=torch.long)}
    assert _ndim_of([batched_2d], loc_2d) == 2
    assert _ndim_of([unbatched_3d], loc_3d) == 3
    # A Hicks interpolator carries the same information.
    hicks = deepwave.Hicks(torch.tensor([[[4.5, 4.5]]]))
    assert _ndim_of([batched_2d], {"source_locations": hicks}) == 2
    # Non-location keyword arguments are ignored, and with no location argument
    # at all the (then necessarily unbatched) model is used.
    assert _ndim_of([unbatched_3d], {"location_interpolation": "hicks"}) == 3
    assert _ndim_of([unbatched_3d]) == 3


def test_batched_model_grad_matches_device():
    """Gradients of a per-shot batched model match device mode to round-off."""
    v, dx, dt, src, kw = _batched_scalar_problem(nt=60, dt=0.001)
    sources = {"source_amplitudes": src}

    def loss(o):
        return o.receiver_amplitudes.pow(2).sum() + 0.5 * o.wavefield_0.pow(2).sum()

    gm_d, gs_d, _ = _grad_device(scalar, [v], dx, dt, sources, kw, loss)
    gm_c, gs_c, _ = _grad_tuc(scalar, [v], dx, dt, sources, kw, loss, n_segments=4)
    assert gm_c[0].shape == v.shape
    assert _rel(gm_c[0], gm_d[0]) < 1e-12
    assert _rel(gs_c["source_amplitudes"], gs_d["source_amplitudes"]) < 1e-12


# ---------------------------------------------------------------------------
# Automatic whole-trace resampling at step_ratio > 1
# ---------------------------------------------------------------------------


def _substepping_scalar_problem(nt=60, dt=0.004):
    """A 2-D scalar problem whose dt needs internal sub-stepping.

    Same shape as :func:`_scalar_problem`, but with a ``dt`` large enough that
    the propagator runs internally at a smaller time step (``step_ratio > 1``),
    which is the case ``run_segments`` handles by resampling the whole trace.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=nt, dt=dt)
    _, step_ratio = cfl_condition_n([dx, dx], dt, kw["max_vel"])
    assert step_ratio > 1, step_ratio
    return v, dx, dt, src, kw, step_ratio


def test_run_segments_step_ratio_gt_1_matches_direct():
    """A multi-segment run at step_ratio > 1 reproduces an unsegmented call.

    The final wavefield state must match bitwise. The receiver amplitudes are
    compared to round-off rather than bitwise because the two paths hand the
    final downsampling FFT differently-laid-out inputs: a direct call
    downsamples a permuted view of its internal-rate trace, while the segmented
    path downsamples the contiguous concatenation of the segments' traces,
    which can differ in the last ulp.
    """
    v, dx, dt, src, kw, _ = _substepping_scalar_problem()
    full = scalar(v, dx, dt, source_amplitudes=src, **kw)
    combined = run_segments(
        scalar, [v], dx, dt, {"source_amplitudes": src}, n_segments=4, **kw
    )
    assert torch.equal(full.wavefield_0, combined.wavefield_0)
    assert torch.equal(full.wavefield_m1, combined.wavefield_m1)
    assert combined.receiver_amplitudes.shape == full.receiver_amplitudes.shape
    assert _rel(combined.receiver_amplitudes, full.receiver_amplitudes) < 1e-13


def test_run_segments_single_segment_step_ratio_gt_1_is_direct():
    """A single segment at step_ratio > 1 is just a direct call (no resampling).

    One segment has no boundary to protect, so the automation must stay inert
    and the propagator's own internal resampling must be the only one applied.
    """
    v, dx, dt, src, kw, _ = _substepping_scalar_problem()
    full = scalar(v, dx, dt, source_amplitudes=src, **kw)
    combined = run_segments(
        scalar, [v], dx, dt, {"source_amplitudes": src}, n_segments=1, **kw
    )
    assert torch.equal(full.wavefield_0, combined.wavefield_0)
    assert torch.equal(full.receiver_amplitudes, combined.receiver_amplitudes)


@pytest.mark.parametrize("n_segments", [2, 5])
def test_scalar_grad_step_ratio_gt_1_matches_device(n_segments):
    """Checkpointed segments at step_ratio > 1 match an unsegmented gradient."""
    v, dx, dt, src, kw, _ = _substepping_scalar_problem(nt=60)
    sources = {"source_amplitudes": src}

    def loss(o):
        return o.receiver_amplitudes.pow(2).sum() + 0.5 * o.wavefield_0.pow(2).sum()

    gm_d, gs_d, _ = _grad_device(scalar, [v], dx, dt, sources, kw, loss)
    gm_c, gs_c, _ = _grad_tuc(scalar, [v], dx, dt, sources, kw, loss, n_segments)
    assert _rel(gm_c[0], gm_d[0]) < 1e-12
    assert _rel(gs_c["source_amplitudes"], gs_d["source_amplitudes"]) < 1e-12


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_scalar_grad_step_ratio_gt_1_both_reentrant_modes(use_reentrant):
    """Both torch.utils.checkpoint implementations give the same gradient.

    The non-reentrant implementation retains more native wavefield history,
    but the public choice must remain numerically equivalent.
    """
    v, dx, dt, src, kw, _ = _substepping_scalar_problem(nt=48)
    sources = {"source_amplitudes": src}

    def loss(o):
        return o.receiver_amplitudes.pow(2).sum()

    gm_d, gs_d, _ = _grad_device(scalar, [v], dx, dt, sources, kw, loss)
    gm_c, gs_c, _ = _grad_tuc(
        scalar, [v], dx, dt, sources, kw, loss, 4, use_reentrant=use_reentrant
    )
    assert _rel(gm_c[0], gm_d[0]) < 1e-12
    assert _rel(gs_c["source_amplitudes"], gs_d["source_amplitudes"]) < 1e-12


def test_elastic_grad_step_ratio_gt_1_matches_device():
    """A multi-source-type family (elastic) at step_ratio > 1 matches too.

    ``max_vel`` is essential here: the elastic model tensors are Lame
    parameters, so the maximum-absolute-model-value proxy is meaningless as a
    wave speed.
    """
    ny, nx = 30, 30
    dx = 5.0
    freq = 15.0
    dt = 0.002  # step_ratio > 1 for max_vel = 1600
    nt = 40
    vp = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    vs = torch.full((ny, nx), 800.0, dtype=DTYPE)
    rho = torch.full((ny, nx), 2000.0, dtype=DTYPE)
    lamb = (vp**2 - 2 * vs**2) * rho
    mu = vs**2 * rho
    buoyancy = 1.0 / rho
    src = _ricker(freq, nt, dt)
    kw = {
        "source_locations_y": torch.tensor([[[15, 15]]]),
        "receiver_locations_y": torch.tensor([[[4, 4], [24, 24]]]),
        "pml_freq": freq,
        "max_vel": 1600.0,
    }
    _, step_ratio = cfl_condition_n([dx, dx], dt, kw["max_vel"])
    assert step_ratio > 1, step_ratio
    sources = {"source_amplitudes_y": src}

    def loss(o):
        return o.receiver_amplitudes_y.pow(2).sum()

    gm_d, gs_d, _ = _grad_device(
        elastic, [lamb, mu, buoyancy], dx, dt, sources, kw, loss
    )
    gm_c, gs_c, _ = _grad_tuc(
        elastic, [lamb, mu, buoyancy], dx, dt, sources, kw, loss, 4
    )
    for a, b in zip(gm_c, gm_d):
        assert _rel(a, b) < 1e-12
    assert _rel(gs_c["source_amplitudes_y"], gs_d["source_amplitudes_y"]) < 1e-12


def test_scalar_born_grad_step_ratio_gt_1_matches_device():
    """Optional model tensors thread through checkpointing with resampling.

    ``dv`` is passed via ``optional_model_args``, so it crosses each checkpoint
    boundary positionally; this checks that composes with the whole-trace
    resampling.
    """
    v, dx, dt, src, kw, _ = _substepping_scalar_problem(nt=60)
    scatter = torch.zeros_like(v)
    scatter[15, 15] = 100.0
    sources = {"source_amplitudes": src}

    def loss(o):
        return (
            o.sc_receiver_amplitudes.pow(2).sum()
            + o.bg_receiver_amplitudes.pow(2).sum()
        )

    opt = {"dv": scatter}
    gm_d, gs_d, gom_d = _grad_device(
        scalar_born, [v], dx, dt, sources, kw, loss, optional_model_args=opt
    )
    gm_c, gs_c, gom_c = _grad_tuc(
        scalar_born, [v], dx, dt, sources, kw, loss, 4, optional_model_args=opt
    )
    assert _rel(gm_c[0], gm_d[0]) < 1e-12
    assert _rel(gs_c["source_amplitudes"], gs_d["source_amplitudes"]) < 1e-12
    assert _rel(gom_c["dv"], gom_d["dv"]) < 1e-12


def test_mgsi_forwarding_at_step_ratio_gt_1():
    """model_gradient_sampling_interval is scaled to the internal rate.

    A direct call samples the model-gradient integral every ``step_ratio *
    model_gradient_sampling_interval`` internal steps, so the segments (which
    run at the internal step) must be asked for that same stride. ``nt=96`` in
    four segments gives 24 user steps per segment, divisible by the requested
    interval of 2, so the sampled steps line up with the direct call's.
    """
    v, dx, dt, src, kw, _ = _substepping_scalar_problem(nt=96)
    kw = {**kw, "model_gradient_sampling_interval": 2}
    sources = {"source_amplitudes": src}

    def loss(o):
        return o.receiver_amplitudes.pow(2).sum()

    gm_d, gs_d, _ = _grad_device(scalar, [v], dx, dt, sources, kw, loss)
    gm_c, gs_c, _ = _grad_tuc(scalar, [v], dx, dt, sources, kw, loss, 4)
    assert _rel(gm_c[0], gm_d[0]) < 1e-12
    assert _rel(gs_c["source_amplitudes"], gs_d["source_amplitudes"]) < 1e-12


def test_run_segments_step_ratio_gt_1_resolves_max_vel():
    """Automatic speed resolution uses the whole-call CFL setup."""
    v, dx, dt, src, kw, _ = _substepping_scalar_problem()
    kw = {k: value for k, value in kw.items() if k != "max_vel"}
    full = scalar(v, dx, dt, source_amplitudes=src, **kw)
    combined = run_segments(
        scalar, [v], dx, dt, {"source_amplitudes": src}, n_segments=4, **kw
    )
    assert full.info.step_ratio > 1
    assert combined.info.step_ratio == full.info.step_ratio
    assert torch.allclose(
        full.wavefield_0, combined.wavefield_0, rtol=1e-12, atol=1e-12
    )
    assert torch.allclose(
        full.receiver_amplitudes, combined.receiver_amplitudes, rtol=1e-12, atol=1e-12
    )


def test_run_segments_no_max_vel_ok_at_step_ratio_1():
    """Without max_vel but with no sub-stepping, the run proceeds as before.

    The real preparation resolves the same step_ratio of 1 for both calls;
    no whole-record resampling is needed.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=60, dt=0.001)
    kw = {k: value for k, value in kw.items() if k != "max_vel"}
    full = scalar(v, dx, dt, source_amplitudes=src, **kw)
    combined = run_segments(
        scalar, [v], dx, dt, {"source_amplitudes": src}, n_segments=4, **kw
    )
    assert torch.equal(full.wavefield_0, combined.wavefield_0)
    assert torch.equal(full.receiver_amplitudes, combined.receiver_amplitudes)


def test_segment_resampling_upsample_empty_source():
    """An empty source tensor keeps its (scaled) time length when upsampled.

    ``upsample`` returns empty tensors unchanged, but the time axis must still
    grow by ``step_ratio`` so that the per-segment split of an empty source
    stays consistent with the non-empty ones.
    """
    resampling = _SegmentResampling(
        dt=0.001, step_ratio=4, freq_taper_frac=0.0, time_pad_frac=0.0, time_taper=False
    )
    empty = torch.zeros(1, 0, 40, dtype=DTYPE)
    out = resampling.upsample(empty)
    assert out.shape == (1, 0, 160)
    assert out.dtype == empty.dtype
    assert out.device == empty.device
    assert torch.equal(resampling.downsample(empty), empty)


def test_manual_outer_tuc_loop_composition():
    """The manual pre-upsampling outer loop still works and stays exact.

    A caller that drives the segments itself (rather than letting
    ``run_segments`` do it) must still upsample the whole source before
    chunking and downsample the concatenated receiver output at the end, since
    no per-segment call can see the rest of the trace. This pins that pattern
    down, and with it that the automation stays inert when each call is a
    single segment already at the internal time step.
    """
    v, dx, dt, src, kw, _ = _substepping_scalar_problem(nt=60)
    kw_dev = _kw_to_device(kw)
    inner_dt, r = cfl_condition_n([dx, dx], dt, kw["max_vel"])
    n_segments = 3

    def loss(o):
        return o.receiver_amplitudes.pow(2).sum()

    gm_d, gs_d, _ = _grad_device(
        scalar, [v], dx, dt, {"source_amplitudes": src}, kw, loss
    )

    model = v.clone().to(_DEVICE).requires_grad_(True)
    source = src.clone().to(_DEVICE).requires_grad_(True)
    up = deepwave.discretisation.upsample(source, r)
    spec = _get_spec(scalar)
    state_names = list(spec.wavefield_names(_ndim_of([model], kw_dev)))

    boundary_names = list(spec.auxiliary_state_names(_ndim_of([model], kw_dev)))

    def wrap(model_tensor, chunk, *state_tensors):
        state = dict(zip(state_names, state_tensors)) if state_tensors else None
        if state is not None:
            folded = {n: state.pop(n) for n in boundary_names}
            state["auxiliary_state"] = AuxiliaryState._from_flat(folded)
        out = run_segments(
            scalar,
            [model_tensor],
            dx,
            inner_dt,
            {"source_amplitudes": chunk},
            n_segments=1,
            initial_state=state,
            model_gradient_sampling_interval=r,
            **kw_dev,
        )
        # A flat tuple of tensors: a reentrant checkpoint re-attaches only the
        # top-level Tensor elements of what the wrapped function returns, so
        # tensors returned inside an AuxiliaryState come back detached and the
        # state's gradient path across the segment boundary is lost.
        flat = flat_state(out, spec, 2)
        return (*(flat[n] for n in state_names), out.receiver_amplitudes)

    state_tensors = ()
    chunks = []
    pieces = torch.chunk(up, n_segments, dim=-1)
    for i, chunk in enumerate(pieces):
        if i == len(pieces) - 1:
            out = wrap(model, chunk, *state_tensors)
        else:
            out = torch.utils.checkpoint.checkpoint(
                wrap, model, chunk, *state_tensors, use_reentrant=True
            )
        chunks.append(out[-1])
        state_tensors = tuple(out[: len(state_names)])
    receiver_amplitudes = deepwave.discretisation.downsample(
        torch.cat(chunks, dim=-1), r
    )
    receiver_amplitudes.pow(2).sum().backward()
    assert _rel(model.grad, gm_d[0]) < 1e-12
    assert _rel(source.grad, gs_d["source_amplitudes"]) < 1e-12


# ---------------------------------------------------------------------------
# The checkpoint wrapper is skipped when there is nothing to differentiate
# ---------------------------------------------------------------------------


def _spy_on_checkpoint(monkeypatch):
    """Counts ``torch.utils.checkpoint.checkpoint`` calls, still doing them.

    Returns a one-element list holding the running call count.
    """
    calls = [0]
    real = torch.utils.checkpoint.checkpoint

    def spy(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", spy)
    return calls


def _tensor_fields(out):
    """Flattens a PropagatorOutput to name -> tensor, auxiliary state included."""
    flat = {}
    for name, value in out.as_dict().items():
        if isinstance(value, torch.Tensor):
            flat[name] = value
        elif isinstance(value, AuxiliaryState):
            for inner, tensor in value.as_dict().items():
                flat[f"{name}.{inner}"] = tensor
    return flat


def test_no_grad_checkpointed_run_emits_no_warnings(monkeypatch):
    """A forward-only checkpoint=True run neither warns nor wraps a segment.

    torch warns "None of the inputs have requires_grad=True" once for every
    segment it wraps, so the count of warnings is also a count of wrapped
    segments here.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=80, dt=0.001)
    calls = _spy_on_checkpoint(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = run_segments(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            n_segments=4,
            checkpoint=True,
            **kw,
        )
    assert [str(w.message) for w in caught] == []
    assert calls[0] == 0
    assert out.receiver_amplitudes.shape[-1] == 80


def test_checkpoint_skip_is_bit_exact_with_the_wrapper(monkeypatch):
    """Skipping the wrapper is a performance change and not a numeric one.

    The two paths are compared on the same inputs: the skip, then the wrapper
    forced back on by making the "does anything require grad?" predicate say
    yes. Every returned tensor -- receiver amplitudes, wavefields and the
    auxiliary state -- must be bitwise identical, since
    ``torch.utils.checkpoint`` runs the very same segment function, only under
    ``torch.no_grad()``.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=80, dt=0.001)

    def go():
        return run_segments(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            n_segments=4,
            checkpoint=True,
            **kw,
        )

    calls = _spy_on_checkpoint(monkeypatch)
    skipped = go()
    assert calls[0] == 0

    monkeypatch.setattr("deepwave.checkpointing._any_requires_grad", lambda obj: True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapped = go()
    # The comparison is only meaningful if the second run really took the
    # wrapped path: three of the four segments wrapped (the last never is),
    # each one warning that it had nothing to differentiate.
    assert calls[0] == 3
    assert sum("requires_grad=True" in str(w.message) for w in caught) == 3

    left = _tensor_fields(skipped)
    right = _tensor_fields(wrapped)
    assert sorted(left) == sorted(right)
    assert "receiver_amplitudes" in left
    assert "wavefield_0" in left
    for name, value in left.items():
        assert torch.equal(value, right[name]), name


def test_no_grad_checkpointed_run_matches_a_direct_call():
    """The skipped-wrapper path still reproduces an unsegmented call bitwise."""
    v, dx, dt, src, kw = _scalar_problem(nt=80, dt=0.001)
    full = scalar(v, dx, dt, source_amplitudes=src, **kw)
    segmented = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=4,
        checkpoint=True,
        **kw,
    )
    assert torch.equal(full.receiver_amplitudes, segmented.receiver_amplitudes)
    assert torch.equal(full.wavefield_0, segmented.wavefield_0)


def test_grad_enabled_run_still_checkpoints_and_matches_device(monkeypatch):
    """With a differentiable input the wrapper is used and gradients are right."""
    v, dx, dt, src, kw = _scalar_problem(nt=80, dt=0.001)
    sources = {"source_amplitudes": src}

    def loss(o):
        return o.receiver_amplitudes.pow(2).sum() + 0.5 * o.wavefield_0.pow(2).sum()

    gm_d, gs_d, _ = _grad_device(scalar, [v], dx, dt, sources, kw, loss)
    calls = _spy_on_checkpoint(monkeypatch)
    gm_c, gs_c, _ = _grad_tuc(scalar, [v], dx, dt, sources, kw, loss, 4)
    # Every segment but the last.
    assert calls[0] == 3
    assert _rel(gm_c[0], gm_d[0]) < 1e-12
    assert _rel(gs_c["source_amplitudes"], gs_d["source_amplitudes"]) < 1e-12


def test_checkpoint_wrapper_used_for_a_differentiable_source_alone(monkeypatch):
    """The source amplitudes are in the scanned set, models aside."""
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    calls = _spy_on_checkpoint(monkeypatch)
    out = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src.clone().requires_grad_(True)},
        n_segments=4,
        checkpoint=True,
        **kw,
    )
    assert calls[0] == 3
    assert out.receiver_amplitudes.requires_grad


def test_checkpoint_wrapper_used_for_a_differentiable_initial_state(monkeypatch):
    """The caller's initial state is in the scanned set, models aside."""
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    first = scalar(v, dx, dt, source_amplitudes=src, **kw)
    start = first.wavefield_0.detach().clone().requires_grad_(True)
    calls = _spy_on_checkpoint(monkeypatch)
    out = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=4,
        checkpoint=True,
        initial_state={"wavefield_0": start},
        **kw,
    )
    assert calls[0] == 3
    assert out.receiver_amplitudes.requires_grad


def test_checkpoint_wrapper_used_for_a_differentiable_optional_model(monkeypatch):
    """``optional_model_args`` is in the scanned set, models aside."""
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    scatter = torch.zeros_like(v).requires_grad_(True)
    calls = _spy_on_checkpoint(monkeypatch)
    out = run_segments(
        scalar_born,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=4,
        checkpoint=True,
        optional_model_args={"dv": scatter},
        **kw,
    )
    assert calls[0] == 3
    assert out.sc_receiver_amplitudes.requires_grad


def test_checkpoint_wrapper_skipped_under_no_grad(monkeypatch):
    """``torch.no_grad()`` makes requires_grad inputs undifferentiable anyway."""
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    model = v.clone().requires_grad_(True)
    calls = _spy_on_checkpoint(monkeypatch)
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = run_segments(
            scalar,
            [model],
            dx,
            dt,
            {"source_amplitudes": src.clone().requires_grad_(True)},
            n_segments=4,
            checkpoint=True,
            **kw,
        )
    assert calls[0] == 0
    assert [str(w.message) for w in caught] == []
    assert not out.receiver_amplitudes.requires_grad


def test_any_requires_grad_sees_nested_tensors():
    """The predicate looks inside the containers the input set can use."""
    plain = torch.zeros(2)
    live = torch.zeros(2, requires_grad=True)
    assert not _any_requires_grad(plain)
    assert _any_requires_grad(live)
    assert not _any_requires_grad({"a": [plain, (plain,)]})
    assert _any_requires_grad({"a": [plain, (live,)]})
    assert not _any_requires_grad("wavefield_0")
    assert not _any_requires_grad(None)


# ---------------------------------------------------------------------------
# Input validation / guards
# ---------------------------------------------------------------------------


def test_checkpoint_device_requires_checkpoint():
    """checkpoint_device is meaningless without checkpointed segments."""
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    with pytest.raises(ValueError, match="checkpoint=True"):
        run_segments(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            n_segments=2,
            checkpoint_device="cpu",
            **kw,
        )


def test_checkpoint_rejects_differentiable_kwargs_tensor():
    """A requires_grad tensor in kwargs would silently get no gradient."""
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    scatter = torch.zeros_like(v).requires_grad_(True)
    with pytest.raises(ValueError, match="optional_model_args"):
        run_segments(
            scalar_born,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            n_segments=2,
            checkpoint=True,
            dv=scatter,
            **kw,
        )


def _viscous_problem(name, nt=24, dt=0.0002, n=24):
    """A tiny attenuating problem, as ``(model_args, dx, dt, sources, kwargs)``.

    Viscoacoustic state field count follows ``q_n_mechanisms``.
    """
    ones = torch.ones((n, n), dtype=DTYPE)
    dx = 5.0
    # The scalar source channel is named for the quantity it injects, which
    # differs between the two families; the receiver channel is ``p`` on both.
    rate = deepwave.spec(name).source_channels(2)[0]
    kw = {
        f"source_locations_{rate}": torch.tensor([[[n // 2, n // 2]]]),
        "receiver_locations_p": torch.tensor([[[4, 4]]]),
        "pml_freq": 15.0,
        "q_freq": 15.0,
        "pml_width": 3,
        "max_vel": 2000.0,
    }
    sources = {f"source_amplitudes_{rate}": _ricker(15.0, nt, dt)}
    if name == "viscoacoustic":
        models = [1500.0 * ones, 2200.0 * ones, 20.0 * ones]
    else:
        models = [
            2.25e9 * ones,
            1.0e9 * ones,
            (1 / 2200) * ones,
            20.0 * ones,
            20.0 * ones,
        ]
    return models, dx, dt, sources, kw


@pytest.mark.parametrize("n_mechanisms", [2, 3])
@pytest.mark.parametrize("use_reentrant", [True, False])
def test_checkpointed_multi_mechanism_matches_direct(n_mechanisms, use_reentrant):
    """All GSLS memories survive segment flattening and recomputation."""
    models, dx, dt, sources, kw = _viscous_problem("viscoacoustic", nt=48)
    kw = {key: value for key, value in kw.items() if key != "q_freq"}
    kw["receiver_locations_p"] = kw["source_locations_pressure_rate"]
    extra = {"q_n_mechanisms": n_mechanisms, "q_band": (5.0, 60.0)}

    def calculate(segmented):
        local_models = [m.detach().clone().requires_grad_() for m in models]
        local_sources = {
            k: s.detach().clone().requires_grad_() for k, s in sources.items()
        }
        if segmented:
            out = run_segments(
                deepwave.viscoacoustic,
                local_models,
                dx,
                dt,
                local_sources,
                n_segments=3,
                checkpoint=True,
                use_reentrant=use_reentrant,
                **kw,
                **extra,
            )
        else:
            out = deepwave.viscoacoustic(
                *local_models, dx, dt, **local_sources, **kw, **extra
            )
        loss = out.receiver_amplitudes_p.square().sum()
        loss.backward()
        return (
            out,
            [m.grad for m in local_models],
            next(iter(local_sources.values())).grad,
        )

    direct, direct_models, direct_source = calculate(False)
    split, split_models, split_source = calculate(True)
    assert len(split.auxiliary_state) == len(direct.auxiliary_state)
    assert _rel(split.receiver_amplitudes_p, direct.receiver_amplitudes_p) < 1e-10
    for actual, expected in zip(split_models, direct_models):
        assert actual is not None
        assert expected is not None
        assert _rel(actual, expected) < 1e-7
    assert split_source is not None
    assert direct_source is not None
    assert _rel(split_source, direct_source) < 1e-7


@pytest.mark.parametrize(
    "name",
    [
        "viscoacoustic",
    ],
)
@pytest.mark.parametrize("n_mechanisms", [2, 3])
def test_uncheckpointed_multi_mechanism_segments_still_match(name, n_mechanisms):
    """The guard is on checkpointing alone; ``checkpoint=False`` is untouched.

    That branch hands each segment the previous segment's whole
    ``PropagatorOutput``, whose auxiliary state is opaque and therefore
    complete however many mechanisms it holds -- so it still reproduces the
    single combined call, and must keep doing so.

    To round-off, not bit for bit: the multi-mechanism update's segmented and
    unsegmented forms were measured agreeing exactly within any one process
    but not across processes, a reproducibility limit that predates this guard
    and has nothing to do with segmentation (see the same note in
    ``tests/test_auxiliary_state.py``). At one mechanism it is exact, which
    the test below pins.
    """
    propagator = getattr(deepwave, name)
    models, dx, dt, sources, kw = _viscous_problem(name)
    # Above one mechanism the constant-Q fit spans q_band, and the single
    # reference frequency ``_viscous_problem`` pins is refused.
    kw = {key: value for key, value in kw.items() if key != "q_freq"}
    extra = {"q_n_mechanisms": n_mechanisms, "q_band": (5.0, 60.0)}
    with torch.no_grad():
        direct = propagator(*models, dx, dt, **sources, **kw, **extra)
        segmented = run_segments(
            propagator,
            models,
            dx,
            dt,
            sources,
            n_segments=3,
            checkpoint=False,
            **kw,
            **extra,
        )
    assert _rel(segmented.receiver_amplitudes_p, direct.receiver_amplitudes_p) < 1e-12


@pytest.mark.parametrize(
    "name",
    [
        "viscoacoustic",
    ],
)
def test_checkpointed_single_mechanism_is_unaffected(name):
    """One mechanism is what the spec's fixed names describe, so it still runs."""
    propagator = getattr(deepwave, name)
    models, dx, dt, sources, kw = _viscous_problem(name)
    with torch.no_grad():
        direct = propagator(*models, dx, dt, **sources, **kw)
        segmented = run_segments(
            propagator, models, dx, dt, sources, n_segments=3, checkpoint=True, **kw
        )
    assert torch.equal(direct.receiver_amplitudes_p, segmented.receiver_amplitudes_p)


def test_checkpointed_format_version_mismatch_raises():
    """A stale ``format_version`` is refused under checkpointing too.

    The per-segment bundle is rebuilt from the threaded tensors by copying the
    caller's fingerprint field by field. ``format_version`` has to be one of
    the fields copied: without it the rebuilt bundle carries the class default,
    the compatibility check compares this release's version against itself, and
    a state written to a layout this release cannot read is silently used.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=24, dt=0.001)
    with torch.no_grad():
        out = scalar(v, dx, dt, source_amplitudes=src, **kw)
    stale = dataclasses.replace(
        out.auxiliary_state,
        format_version=out.auxiliary_state.format_version + 1,
    )
    state = {
        name: tensor
        for name, tensor in out.final_state.items()
        if name != "auxiliary_state"
    }
    state["auxiliary_state"] = stale
    for checkpoint in (False, True):
        with pytest.raises(DeepwaveValueError) as excinfo:
            run_segments(
                scalar,
                [v],
                dx,
                dt,
                {"source_amplitudes": src},
                n_segments=3,
                checkpoint=checkpoint,
                initial_state=state,
                **kw,
            )
        message = str(excinfo.value)
        assert "format_version" in message
        assert str(stale.format_version) in message
        assert str(out.auxiliary_state.format_version) in message


def test_auxiliary_state_format_version_must_be_an_int():
    """``1.0 == 1``, so a float would pass a layout check it does not name."""
    with pytest.raises(deepwave.DeepwaveTypeError, match="must be an int"):
        AuxiliaryState({}, format_version=1.0)
    with pytest.raises(deepwave.DeepwaveTypeError, match="must be an int"):
        AuxiliaryState({}, format_version=True)


def test_optional_model_args_rejects_kwargs_collision():
    """The same optional model tensor must not be given twice."""
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    scatter = torch.zeros_like(v)
    with pytest.raises(ValueError, match="both optional_model_args and kwargs"):
        run_segments(
            scalar_born,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            n_segments=2,
            optional_model_args={"dv": scatter},
            dv=scatter,
            **kw,
        )


def test_partial_initial_state_checkpoint_mode():
    """A partial initial-state mapping threads through checkpointed segments.

    Only ``wavefield_0`` is given; the remaining state entries (including the
    PML auxiliaries) must default to zero exactly as they do without
    checkpointing, so the two modes agree bitwise.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=60, dt=0.001)
    kw = _kw_to_device(kw)
    v = v.to(_DEVICE).clone().requires_grad_(True)
    src = src.to(_DEVICE)
    with torch.no_grad():
        warm = scalar(
            v.detach(),
            dx,
            dt,
            source_amplitudes=_ricker(15.0, 30, dt).to(_DEVICE),
            **kw,
        )
    partial = {"wavefield_0": warm.wavefield_0}
    plain = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=3,
        initial_state=partial,
        **kw,
    )
    checkpointed = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=3,
        initial_state=partial,
        checkpoint=True,
        **kw,
    )
    assert torch.equal(plain.wavefield_0, checkpointed.wavefield_0)
    assert torch.equal(plain.receiver_amplitudes, checkpointed.receiver_amplitudes)


def test_checkpoint_mode_continuation_with_survey_pad_recovers_origin():
    """Checkpointed segments recover the survey origin from initial_state.

    The state is threaded between checkpointed segments as plain tensors, which
    loses the ``RunInfo`` a whole ``PropagatorOutput`` carries, so the origin of
    the ``survey_pad``-extracted run that produced ``initial_state`` is read off
    it once and passed to every segment. Without that, the segments would
    extract a region centred on the model origin instead.
    """
    v, dx, dt, src, kw = _scalar_survey_pad_problem(nt=60)
    kw = _kw_to_device(kw)
    src = src.to(_DEVICE)
    with torch.no_grad():
        warm = scalar(
            v.to(_DEVICE),
            dx,
            dt,
            source_amplitudes=_ricker(15.0, 30, dt).to(_DEVICE),
            **kw,
        )
    # Requiring grad keeps torch.utils.checkpoint from warning that none of its
    # inputs need one; only the forward values are compared here.
    v = v.to(_DEVICE).clone().requires_grad_(True)
    kw_no_pad = {k: value for k, value in kw.items() if k != "survey_pad"}
    plain = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=3,
        initial_state=warm,
        **kw_no_pad,
    )
    checkpointed = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=3,
        initial_state=warm,
        checkpoint=True,
        **kw_no_pad,
    )
    assert torch.equal(plain.wavefield_0, checkpointed.wavefield_0)
    assert torch.equal(plain.receiver_amplitudes, checkpointed.receiver_amplitudes)


def test_checkpoint_mode_origin_recovery_fires_when_origin_explicitly_none():
    """Origin recovery must key on the resolved value, not key presence.

    ``origin=None`` is the documented default for every propagator -- a
    caller (or a wrapper) that passes it explicitly, rather than omitting it,
    is expressing exactly the same "let it be inferred" intent as omitting it
    altogether. The origin-recovery guard must treat the two identically; if
    it instead checked ``"origin" not in seg_kwargs`` (key presence), passing
    ``origin=None`` explicitly would silently skip recovery in checkpoint
    mode only, extracting each segment around the model origin instead of
    the survey_pad-resolved one -- while the non-checkpoint path (which
    keeps the whole PropagatorOutput/RunInfo and never needs this recovery)
    would be unaffected, so the two modes would silently disagree.
    """
    v, dx, dt, src, kw = _scalar_survey_pad_problem(nt=60)
    kw = _kw_to_device(kw)
    src = src.to(_DEVICE)
    with torch.no_grad():
        warm = scalar(
            v.to(_DEVICE),
            dx,
            dt,
            source_amplitudes=_ricker(15.0, 30, dt).to(_DEVICE),
            **kw,
        )
    v = v.to(_DEVICE).clone().requires_grad_(True)
    kw_no_pad = {k: value for k, value in kw.items() if k != "survey_pad"}
    plain = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=3,
        initial_state=warm,
        **kw_no_pad,
    )
    checkpointed = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=3,
        initial_state=warm,
        checkpoint=True,
        origin=None,
        survey_pad=None,
        **kw_no_pad,
    )
    assert torch.equal(plain.wavefield_0, checkpointed.wavefield_0)
    assert torch.equal(plain.receiver_amplitudes, checkpointed.receiver_amplitudes)


def test_run_segments_rejects_too_many_segments():
    """run_segments rejects n_segments exceeding nt, via the shared _segment_lengths."""
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    sources = {"source_amplitudes": src}
    with pytest.raises(ValueError, match="cannot exceed"):
        run_segments(scalar, [v], dx, dt, sources, n_segments=1000, **kw)


def test_get_spec_rejects_non_propagator_without_a_hand_maintained_list():
    """``_get_spec``'s error describes the SPEC rule, not a fixed propagator list.

    Regression test for a stale hard-coded list that once named only 7 of
    the (now 11) propagator modules exposing a ``PropagatorSpec``: the
    message must describe the general rule instead, so it cannot go stale
    again as new propagators are added.
    """

    def not_a_propagator() -> None:
        return None

    with pytest.raises(TypeError, match="PropagatorSpec named SPEC") as exc_info:
        _get_spec(not_a_propagator)
    message = str(exc_info.value)
    # Not an exhaustive, hand-maintained enumeration.
    assert "elastic_born" not in message
    assert "elastic_tti" not in message


# ---------------------------------------------------------------------------
# Differentiable intermediate snapshots (snapshots)
# ---------------------------------------------------------------------------


def test_snapshots_equal_forward_callback_bitwise():
    """Snapshots equal the wavefields a forward_callback observes, bitwise."""
    v, dx, dt, src, kw = _scalar_problem(nt=70, dt=0.001)
    steps = [12, 33, 55, 70]
    recorded = {}

    def fcb(state):
        if state.step in steps:
            recorded[state.step] = state.get_wavefield(
                "wavefield_0", view="pml"
            ).clone()

    ref = scalar(
        v,
        dx,
        dt,
        source_amplitudes=src,
        **kw,
        forward_callback=fcb,
        callback_frequency=1,
    )
    final, snaps = snapshots(
        scalar, [v], dx, dt, {"source_amplitudes": src}, steps, **kw
    )
    # Final output equals a single plain run bitwise.
    assert torch.equal(final.receiver_amplitudes, ref.receiver_amplitudes)
    assert torch.equal(final.wavefield_0, ref.wavefield_0)
    # Every intermediate snapshot matches the forward_callback record bitwise.
    # (The forward callback fires *before* the chunk at ``step``, so step == nt
    # is never observed by it; the final state is checked above instead.)
    for s in steps:
        if s == 70:
            continue
        assert torch.equal(snaps[s]["wavefield_0"], recorded[s]), s


def test_snapshots_gradcheck_intermediate():
    """A loss on an intermediate snapshot is differentiable (float64 FD)."""
    ny, nx = 12, 12
    dx, dt, freq, nt = 5.0, 0.001, 15.0, 16
    v0 = torch.full((ny, nx), 1500.0, dtype=DTYPE)
    v0[6:] = 1600.0
    src = _ricker(freq, nt, dt)
    kw = {
        "source_locations": torch.tensor([[[6, 6]]]),
        "pml_freq": freq,
        "max_vel": 2000.0,
        "pml_width": 3,
    }
    steps = [7, nt]

    def loss_fn(v, s):
        _, snaps = snapshots(scalar, [v], dx, dt, {"source_amplitudes": s}, steps, **kw)
        return (snaps[7]["wavefield_0"] ** 2).sum()

    v = v0.clone().requires_grad_(True)
    s = src.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(loss_fn, (v, s), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_snapshots_step_ratio_gt_1():
    """snapshots() works at step_ratio > 1, matching a direct call's wavefields.

    The whole-trace resampling makes segmentation exact, so the snapshot after
    ``s`` user steps is still exactly the wavefield a ``forward_callback``
    observes at ``step == s`` in a single unsegmented run (which internally
    takes ``s * step_ratio`` steps to get there).
    """
    v, dx, dt, src, kw, _ = _substepping_scalar_problem(nt=60)
    steps = [12, 33, 60]
    recorded = {}

    def fcb(state):
        if state.step in steps:
            recorded[state.step] = state.get_wavefield(
                "wavefield_0", view="pml"
            ).clone()

    ref = scalar(
        v,
        dx,
        dt,
        source_amplitudes=src,
        **kw,
        forward_callback=fcb,
        callback_frequency=1,
    )
    final, snaps = snapshots(
        scalar, [v], dx, dt, {"source_amplitudes": src}, steps, **kw
    )
    for s in steps:
        if s == 60:  # the forward callback fires before a chunk, never at nt
            continue
        assert torch.equal(snaps[s]["wavefield_0"], recorded[s]), s
    assert torch.equal(final.wavefield_0, ref.wavefield_0)
    assert _rel(final.receiver_amplitudes, ref.receiver_amplitudes) < 1e-13


def test_snapshots_step_ratio_gt_1_resolves_max_vel():
    """Automatic speed resolution preserves sampled snapshots and receivers."""
    v, dx, dt, src, kw, _ = _substepping_scalar_problem(nt=40)
    kw = {k: value for k, value in kw.items() if k != "max_vel"}
    steps = [10, 20]
    automatic, snaps = snapshots(
        scalar, [v], dx, dt, {"source_amplitudes": src}, steps, **kw
    )
    direct = scalar(v, dx, dt, source_amplitudes=src, **kw)
    pinned, pinned_snaps = snapshots(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        steps,
        max_vel=direct.info.max_vel,
        **kw,
    )
    assert direct.info.step_ratio > 1
    assert _rel(automatic.receiver_amplitudes, direct.receiver_amplitudes) < 1e-13
    assert torch.equal(automatic.wavefield_0, pinned.wavefield_0)
    for step in steps:
        assert torch.equal(
            snaps[step]["wavefield_0"], pinned_snaps[step]["wavefield_0"]
        )


def test_snapshots_input_validation():
    """snapshots() validates the step range and the storage_mode kwarg."""
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    sources = {"source_amplitudes": src}
    with pytest.raises(ValueError, match="1 <= step <= nt"):
        snapshots(scalar, [v], dx, dt, sources, [0, 20], **kw)
    with pytest.raises(ValueError, match="1 <= step <= nt"):
        snapshots(scalar, [v], dx, dt, sources, [10, 41], **kw)
    with pytest.raises(ValueError, match="at least one step"):
        snapshots(scalar, [v], dx, dt, sources, [], **kw)
    with pytest.raises(ValueError, match="storage_mode"):
        snapshots(scalar, [v], dx, dt, sources, [10], storage_mode="device", **kw)


@pytest.mark.parametrize("bad_steps", [[1.5], [True], ["2"], [2.0]])
def test_snapshots_require_integer_indices(bad_steps):
    """A fractional or boolean step cannot silently select another step."""
    v, dx, dt, src, kw = _scalar_problem(nt=8, dt=0.001)
    with pytest.raises(DeepwaveTypeError, match="integer step indices"):
        snapshots(scalar, [v], dx, dt, {"source_amplitudes": src}, bad_steps, **kw)


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("use_reentrant", [False, True])
@pytest.mark.parametrize(
    "bad_state", [[], (), {"auxiliary_state": None}, {"auxiliary_state": 7}]
)
def test_run_segments_rejects_bad_continuation_state(
    checkpoint, use_reentrant, bad_state
):
    """Checkpointing applies the direct engine's state type rules."""
    v, dx, dt, src, kw = _scalar_problem(nt=8, dt=0.001)
    with pytest.raises(DeepwaveTypeError, match="initial_state"):
        run_segments(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            2,
            initial_state=bad_state,
            checkpoint=checkpoint,
            use_reentrant=use_reentrant,
            **kw,
        )


def test_snapshots_last_step_only_runs_full_propagation():
    """A single snapshot before nt still runs and returns the full final output."""
    v, dx, dt, src, kw = _scalar_problem(nt=50, dt=0.001)
    ref = scalar(v, dx, dt, source_amplitudes=src, **kw)
    final, snaps = snapshots(
        scalar, [v], dx, dt, {"source_amplitudes": src}, [20], **kw
    )
    assert set(snaps) == {20}
    assert torch.equal(final.receiver_amplitudes, ref.receiver_amplitudes)


# ---------------------------------------------------------------------------
# Slab storage of the PML memory variables in the threaded auxiliary state
# ---------------------------------------------------------------------------
#
# ``run_segments(checkpoint=True)`` with slab-stored PML memory variables (the
# default) threads only the two per-axis end slabs of each PML memory-variable
# state field across a checkpoint boundary, rebuilding the zero interior inside
# the segment. Everything below is therefore a bitwise (``torch.equal``) parity
# assertion against ``_slab_pml=False``, plus a direct check that the
# interior really is exactly zero on real propagated data and that the kept
# margin encloses what ``_pml_region`` lets the kernels write.


def _slab_scalar_2d(nt=120, dt=0.001, pml_width=6, n=32):
    """A 2-D scalar problem whose PML is thin enough to leave a real interior.

    Two shots, so the tests also cover a batched state (the slabs are cut along
    a spatial axis only, leaving the leading shot dimension untouched).
    """
    v = torch.full((n, n), 1500.0, dtype=DTYPE)
    v[n // 2 :] = 1700.0
    kw = {
        "source_locations": torch.tensor([[[n // 2, n // 2]], [[n // 3, n // 3]]]),
        "receiver_locations": torch.tensor(
            [[[4, 4], [n - 5, n - 5]], [[5, 5], [n - 6, n - 6]]]
        ),
        "pml_freq": 15.0,
        "max_vel": 2000.0,
        "pml_width": pml_width,
        "accuracy": 4,
    }
    return {
        "propagator": scalar,
        "model_args": [v],
        "dx": 5.0,
        "dt": dt,
        "sources": {"source_amplitudes": _ricker(15.0, nt, dt).repeat(2, 1, 1)},
        "kw": kw,
        "opt": None,
    }


def _slab_scalar_3d(nt=70, dt=0.001, pml_width=6, n=16):
    """A 3-D scalar problem (the flagship shape) with a thin PML."""
    v = torch.full((n, n, n), 1500.0, dtype=DTYPE)
    kw = {
        "source_locations": torch.tensor([[[n // 2, n // 2, n // 2]]]),
        "receiver_locations": torch.tensor([[[3, 3, 3], [n - 4, n - 4, n - 4]]]),
        "pml_freq": 15.0,
        "max_vel": 2000.0,
        "pml_width": pml_width,
        "accuracy": 4,
    }
    return {
        "propagator": scalar,
        "model_args": [v],
        "dx": 5.0,
        "dt": dt,
        "sources": {"source_amplitudes": _ricker(15.0, nt, dt)},
        "kw": kw,
        "opt": None,
    }


def _slab_acoustic_2d(nt=180, dt=0.0005, pml_width=6, n=32):
    """A 2-D acoustic (staggered-grid) problem with a thin PML."""
    v = torch.full((n, n), 1500.0, dtype=DTYPE)
    rho = torch.full((n, n), 2000.0, dtype=DTYPE)
    kw = {
        "source_locations_volume_rate": torch.tensor([[[n // 2, n // 2]]]),
        "receiver_locations_p": torch.tensor([[[4, 4], [n - 5, n - 5]]]),
        "pml_freq": 15.0,
        "max_vel": 2000.0,
        "pml_width": pml_width,
        "accuracy": 4,
    }
    return {
        "propagator": acoustic,
        "model_args": [v, rho],
        "dx": 5.0,
        "dt": dt,
        "sources": {"source_amplitudes_volume_rate": _ricker(15.0, nt, dt)},
        "kw": kw,
        "opt": None,
    }


def _slab_elastic_2d(nt=180, dt=0.0005, pml_width=6, n=32, mpml_fraction=None):
    """A 2-D elastic problem with a thin PML (optionally M-PML)."""
    vp = torch.full((n, n), 1500.0, dtype=DTYPE)
    vs = torch.full((n, n), 800.0, dtype=DTYPE)
    rho = torch.full((n, n), 2000.0, dtype=DTYPE)
    lamb, mu, buoyancy = deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    kw = {
        "source_locations_y": torch.tensor([[[n // 2, n // 2]]]),
        "receiver_locations_y": torch.tensor([[[4, 4], [n - 5, n - 5]]]),
        "pml_freq": 15.0,
        "max_vel": 1600.0,
        "pml_width": pml_width,
        "accuracy": 4,
    }
    if mpml_fraction is not None:
        kw["mpml_fraction"] = mpml_fraction
    return {
        "propagator": elastic,
        "model_args": [lamb, mu, buoyancy],
        "dx": 5.0,
        "dt": dt,
        "sources": {"source_amplitudes_y": _ricker(15.0, nt, dt)},
        "kw": kw,
        "opt": None,
    }


def _slab_elastic_3d(nt=120, dt=0.0005, pml_width=5, n=16):
    """A 3-D elastic problem: all 18 ``m_*`` memory variables at once."""
    vp = torch.full((n, n, n), 1500.0, dtype=DTYPE)
    vs = torch.full((n, n, n), 800.0, dtype=DTYPE)
    rho = torch.full((n, n, n), 2000.0, dtype=DTYPE)
    lamb, mu, buoyancy = deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    kw = {
        "source_locations_z": torch.tensor([[[n // 2, n // 2, n // 2]]]),
        "receiver_locations_z": torch.tensor([[[3, 3, 3]]]),
        "pml_freq": 15.0,
        "max_vel": 1600.0,
        "pml_width": pml_width,
        "accuracy": 4,
    }
    return {
        "propagator": elastic,
        "model_args": [lamb, mu, buoyancy],
        "dx": 5.0,
        "dt": dt,
        "sources": {"source_amplitudes_z": _ricker(15.0, nt, dt)},
        "kw": kw,
        "opt": None,
    }


def _slab_scalar_born_2d(nt=120, dt=0.001, pml_width=6, n=32):
    """A 2-D scalar_born problem (two full field sets) with a thin PML."""
    problem = _slab_scalar_2d(nt=nt, dt=dt, pml_width=pml_width, n=n)
    scatter = torch.zeros((n, n), dtype=DTYPE)
    scatter[n // 2, n // 2] = 100.0
    problem["propagator"] = scalar_born
    problem["opt"] = {"dv": scatter}
    return problem


def _slab_viscoacoustic_2d(nt=180, dt=0.0005, pml_width=6, n=32):
    """A 2-D viscoacoustic problem; its ``r_*`` fields must never compress."""
    problem = _slab_acoustic_2d(nt=nt, dt=dt, pml_width=pml_width, n=n)
    problem["propagator"] = viscoacoustic
    problem["kw"]["source_locations_pressure_rate"] = problem["kw"].pop(
        "source_locations_volume_rate"
    )
    problem["sources"]["source_amplitudes_pressure_rate"] = problem["sources"].pop(
        "source_amplitudes_volume_rate"
    )
    problem["model_args"] = [
        problem["model_args"][0],
        problem["model_args"][1],
        torch.full(problem["model_args"][0].shape, 30.0, dtype=DTYPE),
    ]
    # The frequency Q is specified at; the attenuating propagators require it.
    problem["kw"] = {**problem["kw"], "q_freq": 15.0}
    return problem


_SLAB_PROBLEMS = {
    "scalar_2d": _slab_scalar_2d,
    "scalar_3d": _slab_scalar_3d,
    "acoustic_2d": _slab_acoustic_2d,
    "elastic_2d": _slab_elastic_2d,
    "elastic_3d": _slab_elastic_3d,
    "scalar_born_2d": _slab_scalar_born_2d,
    "viscoacoustic_2d": _slab_viscoacoustic_2d,
}


def _slab_direct(problem):
    """Runs one direct (unsegmented, no-grad) propagator call on ``_DEVICE``."""
    kw = _kw_to_device(problem["kw"])
    models = [m.to(_DEVICE) for m in problem["model_args"]]
    sources = {k: v.to(_DEVICE) for k, v in problem["sources"].items()}
    opt = {k: v.to(_DEVICE) for k, v in (problem["opt"] or {}).items()}
    with torch.no_grad():
        return problem["propagator"](
            *models, problem["dx"], problem["dt"], **opt, **sources, **kw
        )


def _slab_plan_for(problem, out):
    """Builds the slab plan for ``out``'s state, as ``run_segments`` would."""
    propagator = problem["propagator"]
    spec = _get_spec(propagator)
    ndim = problem["model_args"][0].ndim
    state_names = spec.wavefield_names(ndim)
    # Through the flat-state accessor, as run_segments does: the individual
    # absorbing-boundary field names are internal, and reading them as output
    # attributes is the deprecated path.
    state = flat_state(out, spec, ndim)
    return _build_slab_plan(
        spec, propagator, ndim, problem["kw"], state_names, state
    ), state


def _slab_roundtrip(name, tensor, plan):
    """Compresses then reconstructs one state field through the slab helpers."""
    compressed = _compress_state((tensor,), [name], plan)[0]
    return _decompress_state({name: compressed}, plan)[name]


def _assert_matches_within_backend_noise(on, off_a, off_b, label):
    """Requires ``on`` to match ``off_a`` as exactly as ``off_a`` matches ``off_b``.

    Slab storage is bitwise-exact by construction, but the thing it must be
    bitwise-exact *against* is not always bitwise-reproducible itself: the
    compiled CUDA model-gradient kernels accumulate with atomics, so two
    identical runs can produce model gradients differing in the last bits (a
    known property of the backend, unrelated to segmentation). Source
    gradients, and every forward output, are reproducible. Comparing the
    slab-on result against a *repeat* of the slab-off baseline therefore holds
    slab storage to the strongest standard the backend actually achieves:
    strictly bitwise wherever the backend is bitwise (always, on CPU, and for
    every source gradient), and otherwise within the backend's own run-to-run
    spread, floored at a few units in the last place so the comparison does not
    itself depend on which way a race happened to fall. Truncating real state
    would show up as a relative error many orders of magnitude larger.
    """
    if torch.equal(off_a, off_b):
        assert torch.equal(on, off_a), label
        return
    ulp = 16 * torch.finfo(off_a.dtype).eps * off_a.abs().max()
    tol = torch.maximum((off_a - off_b).abs().max(), ulp)
    assert (on - off_a).abs().max() <= tol, label


@pytest.mark.parametrize("problem_name", sorted(_SLAB_PROBLEMS))
def test_slab_state_support_and_roundtrip(problem_name):
    """Every declared PML field is exactly zero outside its slabs, on real data.

    This is the property the whole optimisation rests on, checked directly
    rather than inferred: propagate long enough that the wavefront is inside
    the PML (asserted, so the test cannot pass on an all-zero state), then
    compress and reconstruct each declared field and require bitwise equality
    with the original. Reconstruction writes exact zeros into the interior, so
    equality holds only if the interior really was exactly zero.
    """
    problem = _SLAB_PROBLEMS[problem_name]()
    out = _slab_direct(problem)
    plan, state = _slab_plan_for(problem, out)
    assert plan is not None, "the plan must not be empty for these geometries"
    assert any(
        state[name].shape[slab.axis + 1] > slab.lo + slab.hi
        for name, slab in plan.items()
    )
    assert any(state[name].abs().sum() > 0 for name in plan), (
        "no PML memory variable is populated; propagate for longer so the "
        "test is not vacuous"
    )
    for name in plan:
        tensor = state[name]
        assert torch.equal(_slab_roundtrip(name, tensor, plan), tensor), name


@pytest.mark.parametrize("problem_name", ["scalar_3d", "acoustic_2d"])
def test_slab_bitwise_forward(problem_name):
    """Slab-stored auxiliary states give bitwise-identical forward outputs."""
    problem = _SLAB_PROBLEMS[problem_name]()
    propagator = problem["propagator"]
    kw = _kw_to_device(problem["kw"])
    models = [m.to(_DEVICE) for m in problem["model_args"]]
    sources = {k: v.to(_DEVICE) for k, v in problem["sources"].items()}

    def run(slab_pml):
        with torch.no_grad():
            return run_segments(
                propagator,
                models,
                problem["dx"],
                problem["dt"],
                sources,
                n_segments=4,
                checkpoint=True,
                _slab_pml=slab_pml,
                storage_mode="device",
                **kw,
            )

    off = run(False)
    on = run(True)
    for name in _get_spec(propagator).output_names(models[0].ndim):
        assert elements_equal(getattr(on, name), getattr(off, name)), name


@pytest.mark.parametrize("use_reentrant", [True, False])
@pytest.mark.parametrize("problem_name", ["scalar_2d", "elastic_2d"])
def test_slab_bitwise_gradients(problem_name, use_reentrant):
    """Slab-stored auxiliary states give bitwise-identical gradients.

    The loss deliberately includes a declared PML memory variable of the final
    state as well as the receiver data, so the reconstruction node's backward
    is on the path to every leaf. Agreement is asserted through
    :func:`_assert_matches_within_backend_noise`, which is a strict
    ``torch.equal`` except where the backend is not itself reproducible between
    two identical runs (CUDA model gradients).
    """
    problem = _SLAB_PROBLEMS[problem_name]()
    propagator = problem["propagator"]
    spec = _get_spec(propagator)
    ndim = problem["model_args"][0].ndim
    pml_field = sorted(spec.pml_slab_axes(ndim))[0]
    recv_names = [
        n
        for n in spec.output_names(ndim)
        if n.startswith("receiver_amplitudes") and n not in spec.wavefield_names(ndim)
    ]

    def loss(out):
        # ``pml_field`` is an absorbing-boundary memory variable, so it is
        # reached through ``auxiliary_state``; reading it as an output
        # attribute is the deprecated path.
        total = out.auxiliary_state[pml_field].pow(2).sum()
        for name in recv_names:
            total = total + getattr(out, name).pow(2).sum()
        return total

    kwargs = {
        "model_args": problem["model_args"],
        "dx": problem["dx"],
        "dt": problem["dt"],
        "sources": problem["sources"],
        "kw": problem["kw"],
        "lossfn": loss,
        "n_segments": 4,
        "optional_model_args": problem["opt"],
        "use_reentrant": use_reentrant,
    }
    gm_off, gs_off, _ = _grad_tuc(propagator, slab_pml=False, **kwargs)
    gm_off2, gs_off2, _ = _grad_tuc(propagator, slab_pml=False, **kwargs)
    gm_on, gs_on, _ = _grad_tuc(propagator, slab_pml=True, **kwargs)
    for i, (on, off_a, off_b) in enumerate(zip(gm_on, gm_off, gm_off2)):
        _assert_matches_within_backend_noise(on, off_a, off_b, f"model {i}")
    for name in gs_off:
        _assert_matches_within_backend_noise(
            gs_on[name], gs_off[name], gs_off2[name], name
        )


def test_slab_gates():
    """The plan is refused wherever the zero-outside-the-slabs proof does not hold."""
    problem = _slab_elastic_2d()
    out = _slab_direct(problem)
    spec = _get_spec(elastic)
    state_names = spec.wavefield_names(2)
    # Through the flat-state accessor, as run_segments does: the individual
    # absorbing-boundary field names are internal, and reading them as output
    # attributes is the deprecated path.
    state = flat_state(out, spec, 2)

    def plan_with(**overrides):
        return _build_slab_plan(
            spec, elastic, 2, {**problem["kw"], **overrides}, state_names, state
        )

    assert plan_with() is not None
    # Both spellings of "use the compiled kernel" keep the optimisation. The
    # default is ``"auto"``, so a test that only covered the absent-key case
    # would not notice this check being written against a stale literal --
    # which would silently disable slab compression on every checkpointed
    # run rather than raising.
    assert plan_with(backend="auto") is not None
    assert plan_with(backend="compiled") is not None
    # M-PML memory variables are legitimately nonzero wherever any axis is in
    # its PML, so the per-axis slabs would discard real state.
    assert plan_with(mpml_fraction=0.1) is None
    # Non-default boundary conditions were not audited against the supports.
    assert plan_with(boundary=[0, 0, 0, 0]) is None
    # The invariant is a property of the compiled kernels.
    assert plan_with(backend="eager") is None
    assert plan_with(python_backend="eager") is None
    # A propagator that does not declare pml_slab_axes gets no compression.
    vti_spec = _get_spec(elastic_vti)
    assert vti_spec.pml_slab_axes is None
    assert (
        _build_slab_plan(vti_spec, elastic_vti, 2, problem["kw"], state_names, state)
        is None
    )


def test_slab_mpml_falls_back_bitwise():
    """With ``mpml_fraction > 0`` the gate falls back to full-size states."""
    problem = _slab_elastic_2d(nt=90, mpml_fraction=0.1)
    propagator = problem["propagator"]
    kw = _kw_to_device(problem["kw"])
    models = [m.to(_DEVICE) for m in problem["model_args"]]
    sources = {k: v.to(_DEVICE) for k, v in problem["sources"].items()}

    def run(slab_pml):
        with torch.no_grad():
            return run_segments(
                propagator,
                models,
                problem["dx"],
                problem["dt"],
                sources,
                n_segments=3,
                checkpoint=True,
                _slab_pml=slab_pml,
                storage_mode="device",
                **kw,
            )

    off = run(False)
    on = run(True)
    for name in _get_spec(propagator).output_names(2):
        assert elements_equal(getattr(on, name), getattr(off, name)), name


def test_slab_zero_width_faces():
    """A zero-width PML face degenerates to an ``accuracy``-thick slab, still exact."""
    problem = _slab_scalar_2d(nt=100)
    problem["kw"]["pml_width"] = [0, 0, 10, 10]
    kw = _kw_to_device(problem["kw"])
    v = problem["model_args"][0].to(_DEVICE)
    src = problem["sources"]["source_amplitudes"].to(_DEVICE)

    def run(slab_pml):
        vv = v.clone().requires_grad_(True)
        ss = src.clone().requires_grad_(True)
        out = run_segments(
            scalar,
            [vv],
            problem["dx"],
            problem["dt"],
            {"source_amplitudes": ss},
            n_segments=4,
            checkpoint=True,
            _slab_pml=slab_pml,
            storage_mode="device",
            **kw,
        )
        (
            out.receiver_amplitudes.pow(2).sum()
            + out.auxiliary_state["psiy_m1"].pow(2).sum()
        ).backward()
        return out, vv.grad, ss.grad

    off, gv_off, gs_off = run(False)
    _, gv_off2, gs_off2 = run(False)
    on, gv_on, gs_on = run(True)
    for name in _get_spec(scalar).output_names(2):
        assert elements_equal(getattr(on, name), getattr(off, name)), name
    _assert_matches_within_backend_noise(gv_on, gv_off, gv_off2, "v")
    _assert_matches_within_backend_noise(gs_on, gs_off, gs_off2, "source")


def test_slab_asymmetric_pml_width():
    """Slab storage is exact when the two PML faces of an axis differ in width.

    A mutation-testing gap (``py_checkpoint_slab_lo_hi_swap``): swapping
    ``_build_slab_plan``'s low and high slab thicknesses --

        lo = pml[2 * axis] + accuracy      ->  lo = pml[2 * axis + 1] + accuracy
        hi = pml[2 * axis + 1] + accuracy  ->  hi = pml[2 * axis] + accuracy

    -- survived the entire suite, because every existing slab test uses a
    single scalar ``pml_width`` (or an all-equal list), for which ``lo == hi``
    and the swap is a no-op. With different widths per face the swap cuts the
    slabs in the wrong places: the thinner slab no longer covers its PML
    region, so nonzero memory-variable values fall in the "known zero" gap and
    are silently lost across a checkpoint boundary.

    The widths below are chosen so that all four faces differ, and so that no
    two of them can be exchanged without changing the plan.
    """
    problem = _slab_scalar_2d(nt=100)
    problem["kw"]["pml_width"] = [3, 11, 5, 9]
    kw = _kw_to_device(problem["kw"])
    v = problem["model_args"][0].to(_DEVICE)
    src = problem["sources"]["source_amplitudes"].to(_DEVICE)

    def run(slab_pml):
        vv = v.clone().requires_grad_(True)
        ss = src.clone().requires_grad_(True)
        out = run_segments(
            scalar,
            [vv],
            problem["dx"],
            problem["dt"],
            {"source_amplitudes": ss},
            n_segments=4,
            checkpoint=True,
            _slab_pml=slab_pml,
            storage_mode="device",
            **kw,
        )
        # Include the PML memory variables of both axes in the loss: they are
        # exactly the fields the slab plan compresses, so a mis-cut slab shows
        # up here even when the receiver amplitudes barely move.
        (
            out.receiver_amplitudes.pow(2).sum()
            + out.auxiliary_state["psiy_m1"].pow(2).sum()
            + out.auxiliary_state["psix_m1"].pow(2).sum()
            + out.auxiliary_state["zetay_m1"].pow(2).sum()
            + out.auxiliary_state["zetax_m1"].pow(2).sum()
        ).backward()
        return out, vv.grad, ss.grad

    off, gv_off, gs_off = run(False)
    _, gv_off2, gs_off2 = run(False)
    on, gv_on, gs_on = run(True)
    for name in _get_spec(scalar).output_names(2):
        assert elements_equal(getattr(on, name), getattr(off, name)), name
    _assert_matches_within_backend_noise(gv_on, gv_off, gv_off2, "v")
    _assert_matches_within_backend_noise(gs_on, gs_off, gs_off2, "source")
    # Non-vacuity: the PML memory variables really are nonzero, so an
    # incorrectly-placed slab boundary would have something to lose.
    assert off.auxiliary_state["psiy_m1"].abs().max() > 0
    assert off.auxiliary_state["psix_m1"].abs().max() > 0


def test_slab_with_checkpoint_device():
    """Slab storage composes with ``checkpoint_device`` (it just moves fewer bytes)."""
    problem = _slab_scalar_2d(nt=100)
    kw = _kw_to_device(problem["kw"])
    v = problem["model_args"][0].to(_DEVICE)
    src = problem["sources"]["source_amplitudes"].to(_DEVICE)

    def run(slab_pml):
        vv = v.clone().requires_grad_(True)
        ss = src.clone().requires_grad_(True)
        out = run_segments(
            scalar,
            [vv],
            problem["dx"],
            problem["dt"],
            {"source_amplitudes": ss},
            n_segments=4,
            checkpoint=True,
            checkpoint_device="cpu",
            _slab_pml=slab_pml,
            storage_mode="device",
            **kw,
        )
        (
            out.receiver_amplitudes.pow(2).sum()
            + out.auxiliary_state["zetax_m1"].pow(2).sum()
        ).backward()
        return out, vv.grad, ss.grad

    off, gv_off, gs_off = run(False)
    _, gv_off2, gs_off2 = run(False)
    on, gv_on, gs_on = run(True)
    for name in _get_spec(scalar).output_names(2):
        assert elements_equal(getattr(on, name), getattr(off, name)), name
    _assert_matches_within_backend_noise(gv_on, gv_off, gv_off2, "v")
    _assert_matches_within_backend_noise(gs_on, gs_off, gs_off2, "source")


def test_slab_degenerate_geometry():
    """When the two slabs would span the axis, the field is threaded full-size.

    A PML wider than the interior it surrounds is exactly the case where
    ``_pml_region``'s clamping lets the kernels write anywhere along the axis,
    so no field may be compressed -- and the run must still be exact.
    """
    problem = _slab_scalar_2d(nt=60, pml_width=10, n=6)
    out = _slab_direct(problem)
    plan, _ = _slab_plan_for(problem, out)
    assert plan is None
    kw = _kw_to_device(problem["kw"])
    v = problem["model_args"][0].to(_DEVICE)
    src = problem["sources"]["source_amplitudes"].to(_DEVICE)

    def run(slab_pml):
        with torch.no_grad():
            return run_segments(
                scalar,
                [v],
                problem["dx"],
                problem["dt"],
                {"source_amplitudes": src},
                n_segments=3,
                checkpoint=True,
                _slab_pml=slab_pml,
                storage_mode="device",
                **kw,
            )

    off = run(False)
    on = run(True)
    for name in _get_spec(scalar).output_names(2):
        assert elements_equal(getattr(on, name), getattr(off, name)), name


@pytest.mark.parametrize("staggered", [False, True])
@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_slab_margin_envelope_pins_pml_region(staggered, direction):
    """``pml_width + accuracy`` per side encloses every region the kernels write.

    ``_pml_region`` gives padded-coordinate bounds that depend on the grid type
    and the propagation direction; the kept margin is the envelope over all
    four combinations, expressed in pml-view coordinates. If a future change to
    ``_pml_region`` widened a support past that envelope, slab storage would
    silently truncate real state, so the containment is pinned here.
    """
    ndim = 3
    for accuracy in (2, 4, 6, 8):
        fd_pad = accuracy // 2
        fd_pad_list = _grid_padding(staggered, fd_pad, ndim)
        for pml_width in ([20] * 6, [0, 0, 7, 3, 12, 12], [3] * 6):
            view = [60, 55, 48]
            model_shape = [
                view[i] + fd_pad_list[2 * i] + fd_pad_list[2 * i + 1]
                for i in range(ndim)
            ]
            pml_b, pml_e = _pml_region(
                staggered, direction, model_shape, pml_width, fd_pad, ndim
            )
            for i in range(ndim):
                lo = pml_width[2 * i] + accuracy
                hi = pml_width[2 * i + 1] + accuracy
                if lo + hi >= view[i]:
                    continue  # not compressed; the plan skips these
                # Written region in view coordinates: [0, b_view) and
                # [e_view, view[i]).
                b_view = pml_b[i] - fd_pad_list[2 * i]
                e_view = pml_e[i] - fd_pad_list[2 * i]
                assert b_view <= lo
                assert e_view >= view[i] - hi


def test_slab_plan_arithmetic_and_elastic_axis_table():
    """Pins the slab thickness arithmetic and the elastic name-to-axis mapping."""
    spec = _get_spec(scalar)
    names = spec.wavefield_names(3)
    state = {n: torch.zeros(1, 160, 160, 160) for n in names}
    plan = _build_slab_plan(
        spec, scalar, 3, {"pml_width": 20, "accuracy": 4}, names, state
    )
    assert plan is not None
    assert set(plan) == {
        "psiz_m1",
        "psiy_m1",
        "psix_m1",
        "zetaz_m1",
        "zetay_m1",
        "zetax_m1",
    }
    for name, slab in plan.items():
        assert (slab.lo, slab.hi, slab.full) == (24, 24, 160), name
    assert plan["psiz_m1"].axis == 0
    assert plan["psiy_m1"].axis == 1
    assert plan["psix_m1"].axis == 2

    # The 3-D elastic table, verbatim from elastic.zero_edges_and_interiors'
    # audited ``pml_edge_idx`` (axis -> memory-variable names).
    expected_axis_names = {
        0: {
            "m_vzz_0",
            "m_vyz_0",
            "m_vxz_0",
            "m_sigmazzz_0",
            "m_sigmayzz_0",
            "m_sigmaxzz_0",
        },
        1: {
            "m_vzy_0",
            "m_sigmayzy_0",
            "m_vyy_0",
            "m_vxy_0",
            "m_sigmayyy_0",
            "m_sigmaxyy_0",
        },
        2: {
            "m_vzx_0",
            "m_sigmaxzx_0",
            "m_vyx_0",
            "m_sigmaxyx_0",
            "m_vxx_0",
            "m_sigmaxxx_0",
        },
    }
    axes = elastic_pml_slab_axes(3)
    assert len(axes) == 18
    for axis, expected in expected_axis_names.items():
        assert {n for n, a in axes.items() if a == axis} == expected


def test_slab_partial_initial_state():
    """A partial user initial_state is never compressed and stays bit-exact.

    Segment 0 consumes the caller's own ``initial_state`` mapping (which may
    omit fields entirely), so it is threaded untouched; only the propagator's
    own outputs are ever slab-stored.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=60, dt=0.001)
    kw = _kw_to_device(kw)
    v = v.to(_DEVICE).clone().requires_grad_(True)
    src = src.to(_DEVICE)
    with torch.no_grad():
        warm = scalar(
            v.detach(),
            dx,
            dt,
            source_amplitudes=_ricker(15.0, 30, dt).to(_DEVICE),
            **kw,
        )
    partial = {"wavefield_0": warm.wavefield_0}
    plain = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=3,
        initial_state=partial,
        **kw,
    )
    checkpointed = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=3,
        initial_state=partial,
        checkpoint=True,
        _slab_pml=True,
        **kw,
    )
    assert torch.equal(plain.wavefield_0, checkpointed.wavefield_0)
    assert torch.equal(plain.receiver_amplitudes, checkpointed.receiver_amplitudes)


def test_slab_at_step_ratio_gt_1():
    """Slab storage composes with the driver's whole-trace resampling."""
    problem = _slab_scalar_2d(nt=60, dt=0.004)  # dt forces step_ratio > 1
    kw = _kw_to_device(problem["kw"])
    v = problem["model_args"][0].to(_DEVICE)
    src = problem["sources"]["source_amplitudes"].to(_DEVICE)

    def run(slab_pml):
        vv = v.clone().requires_grad_(True)
        ss = src.clone().requires_grad_(True)
        out = run_segments(
            scalar,
            [vv],
            problem["dx"],
            problem["dt"],
            {"source_amplitudes": ss},
            n_segments=4,
            checkpoint=True,
            _slab_pml=slab_pml,
            storage_mode="device",
            **kw,
        )
        (
            out.receiver_amplitudes.pow(2).sum()
            + out.auxiliary_state["psiy_m1"].pow(2).sum()
        ).backward()
        return out, vv.grad, ss.grad

    off, gv_off, gs_off = run(False)
    _, gv_off2, gs_off2 = run(False)
    on, gv_on, gs_on = run(True)
    for name in _get_spec(scalar).output_names(2):
        assert elements_equal(getattr(on, name), getattr(off, name)), name
    _assert_matches_within_backend_noise(gv_on, gv_off, gv_off2, "v")
    _assert_matches_within_backend_noise(gs_on, gs_off, gs_off2, "source")


def test_run_segments_no_slab_pml_parameter():
    """Slab storage is unconditional, so there is no public parameter for it.

    The optimisation is exact and self-disabling, so exposing an on/off switch
    only offered a way to make a run slower for no benefit. The underscored
    escape hatch remains, for the tests above that must exercise both paths.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    with pytest.raises(TypeError, match="checkpoint_slab_pml"):
        run_segments(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            n_segments=2,
            checkpoint=True,
            checkpoint_slab_pml=True,
            **kw,
        )
    assert "checkpoint_slab_pml" not in _SEGMENT_DRIVER_RESERVED
    assert inspect.signature(run_segments).parameters["_slab_pml"].default is True


def test_run_segments_public_reentrant_parameter():
    """The public mode defaults to the memory-saving reentrant choice."""
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    with pytest.raises(TypeError, match="checkpoint_use_reentrant"):
        run_segments(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            n_segments=2,
            checkpoint=True,
            checkpoint_use_reentrant=False,
            **kw,
        )
    assert "checkpoint_use_reentrant" not in _SEGMENT_DRIVER_RESERVED
    assert inspect.signature(run_segments).parameters["use_reentrant"].default is True
    assert "_use_reentrant" not in inspect.signature(run_segments).parameters


def test_run_segments_nonreentrant_supports_autograd_grad():
    """The public mode permits gradients without calling backward()."""
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    model = v.detach().clone().requires_grad_()
    source = src.detach().clone().requires_grad_()
    output = run_segments(
        scalar,
        [model],
        dx,
        dt,
        {"source_amplitudes": source},
        n_segments=2,
        checkpoint=True,
        use_reentrant=False,
        **kw,
    )
    model_grad, source_grad = torch.autograd.grad(
        output.receiver_amplitudes.square().sum(), (model, source)
    )
    assert model_grad.isfinite().all()
    assert model_grad.abs().max() > 0
    assert source_grad.isfinite().all()
    assert source_grad.abs().max() > 0


def test_segment_driver_reserved_is_exactly_the_public_checkpoint_parameters():
    """The reserved set contains the public checkpointing parameters.

    Underscored escape hatches are deliberately not reserved: nothing outside
    Deepwave should be passing them, and a caller who does gets the behaviour
    they asked for rather than a message about a parameter that does not
    render in the documentation.
    """
    assert _SEGMENT_DRIVER_RESERVED == (
        "checkpoint",
        "checkpoint_device",
        "use_reentrant",
    )


# ---------------------------------------------------------------------------
# pml_freq is resolved once for the whole record
# ---------------------------------------------------------------------------


def _collect_pml_freq(into):
    """A run_segments callback that records each segment's resolved pml_freq.

    ``run_segments``' combined output does not carry a ``RunInfo``, but the
    per-segment outputs handed to the callback do.
    """

    def callback(_cumulative_step, out):
        assert out.info is not None
        into.append(out.info.pml_freq)

    return callback


def test_run_segments_without_pml_freq_matches_the_single_call_scalar():
    """With pml_freq omitted, a segmented run equals the single call bitwise.

    Each segment sees only its own slice of the source record, so inferring
    pml_freq per segment would give the segments different PML profiles while
    one set of boundary memory variables is threaded through them all --
    silently breaking the ``run_segments == single call`` guarantee. It must
    therefore be resolved once, from the whole record, and pinned.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=100, dt=0.001)
    kw = {k: value for k, value in kw.items() if k != "pml_freq"}
    full = scalar(v, dx, dt, source_amplitudes=src, **kw)
    seen = []
    segmented = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=4,
        segment_callback=_collect_pml_freq(seen),
        **kw,
    )
    assert full.info is not None
    assert seen == [full.info.pml_freq] * 4
    assert torch.equal(full.receiver_amplitudes, segmented.receiver_amplitudes)
    assert torch.equal(full.wavefield_0, segmented.wavefield_0)
    assert full.receiver_amplitudes.abs().max() > 0


def test_run_segments_without_pml_freq_matches_the_single_call_elastic():
    """The same guarantee for a staggered-grid, multi-channel propagator."""
    ny, nx = 32, 32
    dx, dt, nt, freq = 5.0, 0.0002, 80, 15.0
    vp, vs, rho = 2000.0, 1150.0, 2200.0
    lamb = torch.full((ny, nx), rho * (vp**2 - 2 * vs**2), dtype=DTYPE)
    mu = torch.full((ny, nx), rho * vs**2, dtype=DTYPE)
    buoyancy = torch.full((ny, nx), 1 / rho, dtype=DTYPE)
    src = _ricker(freq, nt, dt)
    kw = {
        "source_locations_y": torch.tensor([[[16, 16]]]),
        "receiver_locations_y": torch.tensor([[[4, c] for c in range(4, 28, 4)]]),
        "max_vel": 2500.0,
    }
    full = elastic(lamb, mu, buoyancy, dx, dt, source_amplitudes_y=src, **kw)
    seen = []
    segmented = run_segments(
        elastic,
        [lamb, mu, buoyancy],
        dx,
        dt,
        {"source_amplitudes_y": src},
        n_segments=4,
        segment_callback=_collect_pml_freq(seen),
        **kw,
    )
    assert full.info is not None
    assert seen == [full.info.pml_freq] * 4
    assert torch.equal(full.receiver_amplitudes_y, segmented.receiver_amplitudes_y)
    assert torch.equal(full.vy_0, segmented.vy_0)
    assert full.receiver_amplitudes_y.abs().max() > 0


def test_run_segments_resolves_pml_freq_from_the_whole_record():
    """A per-segment inference would be visibly different; pin the whole record.

    The first quarter of this record holds only the low-frequency part of a
    two-tone source, so a per-segment inference would give segment 1 a very
    different pml_freq from the whole record's.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=200, dt=0.001)
    kw = {k: value for k, value in kw.items() if k != "pml_freq"}
    low = _ricker(8.0, 200, dt)
    high = _ricker(40.0, 200, dt)
    src = low + torch.roll(high, 120, dims=-1)
    whole = scalar(v, dx, dt, source_amplitudes=src, **kw)
    first_quarter = scalar(v, dx, dt, source_amplitudes=src[..., :50], **kw)
    assert whole.info is not None
    assert first_quarter.info is not None
    # Guard against a vacuous test: the two really do infer different values.
    assert whole.info.pml_freq != first_quarter.info.pml_freq
    seen = []
    segmented = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=4,
        segment_callback=_collect_pml_freq(seen),
        **kw,
    )
    assert seen == [whole.info.pml_freq] * 4
    assert torch.equal(whole.receiver_amplitudes, segmented.receiver_amplitudes)


def test_run_segments_keeps_an_explicit_pml_freq():
    """An explicit pml_freq is passed through untouched."""
    v, dx, dt, src, kw = _scalar_problem(nt=60, dt=0.001)
    kw["pml_freq"] = 33.0
    seen = []
    run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=3,
        segment_callback=_collect_pml_freq(seen),
        **kw,
    )
    assert seen == [33.0] * 3


def test_run_segments_segments_still_report_pml_freq_as_inferred():
    """The segments' resolution warning must not blame the user for the value.

    ``run_segments`` resolves ``pml_freq`` once and then passes it to each
    segment *explicitly*, which would otherwise make each segment's
    grid-cells-per-wavelength warning read as though the user had chosen the
    frequency.
    """
    deepwave._common._warned_once_causes.discard("points_per_wavelength")
    # A grid deliberately too coarse for the 40 Hz wavelet, so the warning fires.
    dt, nt, freq = 0.001, 100, 40.0
    v = torch.full((32, 32), 1500.0, dtype=DTYPE)
    src = _ricker(freq, nt, dt)
    with pytest.warns(UserWarning, match="inferred from the source amplitudes"):
        run_segments(
            scalar,
            [v],
            60.0,
            dt,
            {"source_amplitudes": src},
            n_segments=2,
            source_locations=torch.tensor([[[16, 16]]]),
            receiver_locations=torch.tensor([[[4, 4]]]),
            max_vel=2000.0,
        )


# ---------------------------------------------------------------------------
# The n_segments x model_gradient_sampling_interval hazard raises up front
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("checkpoint", [False, True])
def test_segment_length_not_multiple_of_mgsi_raises(checkpoint):
    """A misaligned split raises in the forward, in both modes.

    nt=90 in 4 near-equal segments is 23, 23, 22, 22, none of which is a
    multiple of ``model_gradient_sampling_interval=2``. Each segment restarts
    the sampling phase at 0, so the segments would sample the gradient
    integral at different steps from the single call they replace.

    Without the up-front check the non-checkpointed path raised from inside
    the first segment, naming 23 -- a number the caller never typed -- and
    saying nothing about segments; the checkpointed path did not raise at all
    until ``.backward()``, because the reentrant checkpoint runs the segment
    under ``no_grad`` and the interval is then silently reset to 1. Both now
    raise before any segment runs, which is what ``no backward()`` in this
    test pins.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=90, dt=0.001)
    vv = v.clone().requires_grad_(True)
    with pytest.raises(DeepwaveValueError) as excinfo:
        run_segments(
            scalar,
            [vv],
            dx,
            dt,
            {"source_amplitudes": src},
            n_segments=4,
            checkpoint=checkpoint,
            model_gradient_sampling_interval=2,
            **kw,
        )
    message = str(excinfo.value)
    assert "Segment 1 of 4 is 23 time steps long" in message
    assert "model_gradient_sampling_interval=2" in message
    assert "Segment lengths are 23, 23, 22, 22" in message
    assert "n_segments=3 (lengths 30, 30, 30)" in message
    assert "seg_lengths= with every length a multiple of 2" in message


def test_segment_mgsi_suggestion_actually_works():
    """The n_segments the message names is one that runs, and matches.

    A suggestion that does not work would be worse than no suggestion.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=90, dt=0.001)
    vv = v.clone().requires_grad_(True)

    def loss(out):
        return out.receiver_amplitudes.pow(2).sum()

    direct = scalar(
        vv, dx, dt, source_amplitudes=src, model_gradient_sampling_interval=2, **kw
    )
    loss(direct).backward()
    direct_grad = vv.grad.clone()
    vv.grad = None
    segmented = run_segments(
        scalar,
        [vv],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=3,
        model_gradient_sampling_interval=2,
        **kw,
    )
    loss(segmented).backward()
    assert _rel(vv.grad, direct_grad) < 1e-12


def test_segment_mgsi_check_uses_seg_lengths_when_given():
    """An explicit misaligned ``seg_lengths`` is caught the same way."""
    v, dx, dt, src, kw = _scalar_problem(nt=90, dt=0.001)
    vv = v.clone().requires_grad_(True)
    with pytest.raises(DeepwaveValueError, match="Segment 2 of 3 is 31 time steps"):
        run_segments(
            scalar,
            [vv],
            dx,
            dt,
            {"source_amplitudes": src},
            seg_lengths=[30, 31, 29],
            model_gradient_sampling_interval=2,
            **kw,
        )


def test_segment_mgsi_message_names_the_internal_interval_when_substepping():
    """At step_ratio > 1 the message says what the segments actually run at.

    The lengths and the interval it quotes stay in *user* time steps, because
    those are the numbers the caller typed and ``seg_lengths=`` is in them
    too; the internal pair is named separately so the two cannot be confused.
    Scaling multiplies both sides by ``step_ratio``, so it never changes
    whether the split is aligned -- checking the user-level pair is enough.
    """
    v, dx, dt, src, kw, step_ratio = _substepping_scalar_problem(nt=90)
    vv = v.clone().requires_grad_(True)
    with pytest.raises(DeepwaveValueError) as excinfo:
        run_segments(
            scalar,
            [vv],
            dx,
            dt,
            {"source_amplitudes": src},
            n_segments=4,
            model_gradient_sampling_interval=2,
            **kw,
        )
    message = str(excinfo.value)
    assert "Segment 1 of 4 is 23 time steps long" in message
    assert (
        f"the internal sampling interval is {2 * step_ratio} after scaling by "
        f"step_ratio={step_ratio}"
    ) in message
    assert "Segment lengths are 23, 23, 22, 22" in message


def test_segment_mgsi_check_skipped_without_model_grad():
    """A forward-only misaligned run still works.

    ``set_model_gradient_sampling_interval`` resets the interval to 1 when no
    model gradient will be computed, so refusing the run would refuse it for a
    gradient it never computes. The condition must match: no model requiring
    grad, or ``torch.no_grad()``.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=90, dt=0.001)
    out = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=4,
        model_gradient_sampling_interval=2,
        **kw,
    )
    assert out.receiver_amplitudes.shape[-1] == 90
    vv = v.clone().requires_grad_(True)
    with torch.no_grad():
        out = run_segments(
            scalar,
            [vv],
            dx,
            dt,
            {"source_amplitudes": src},
            n_segments=4,
            model_gradient_sampling_interval=2,
            **kw,
        )
    assert out.receiver_amplitudes.shape[-1] == 90


def test_segment_mgsi_check_allows_an_aligned_split():
    """Non-vacuity: the same call with an aligned split is not refused."""
    v, dx, dt, src, kw = _scalar_problem(nt=90, dt=0.001)
    vv = v.clone().requires_grad_(True)
    out = run_segments(
        scalar,
        [vv],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=3,
        model_gradient_sampling_interval=2,
        **kw,
    )
    assert out.receiver_amplitudes.shape[-1] == 90


def test_segment_mgsi_check_sees_a_born_perturbation_model():
    """The grad condition covers optional models, however they were passed.

    ``scalar_born``'s ``dv`` is an optional model tensor, which may travel in
    ``optional_model_args`` or (without checkpointing) in ``kwargs``. Either
    way it is a model whose gradient the interval samples, so it must switch
    the check on.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=90, dt=0.001)
    dv = torch.zeros_like(v).requires_grad_(True)
    for placement in ("optional_model_args", "kwargs"):
        extra = (
            {"optional_model_args": {"dv": dv}}
            if placement == "optional_model_args"
            else {"dv": dv}
        )
        with pytest.raises(DeepwaveValueError, match="Segment 1 of 4"):
            run_segments(
                scalar_born,
                [v],
                dx,
                dt,
                {"source_amplitudes": src},
                n_segments=4,
                model_gradient_sampling_interval=2,
                **extra,
                **kw,
            )


# ---------------------------------------------------------------------------
# origin is pinned from segment 0's own RunInfo in the checkpointed branch
# ---------------------------------------------------------------------------


def _origin_warnings(record):
    """The "origin was not provided" warnings in a warning record."""
    return [
        w
        for w in record
        if issubclass(w.category, UserWarning)
        and "because origin was not provided" in str(w.message)
    ]


def test_checkpointed_segments_emit_no_origin_warning():
    """A checkpointed run with no origin and no initial_state warns zero times.

    Segment 0 starts from rest, so it has no wavefield to infer a survey
    origin from and never warns. Every later segment starts from a threaded
    wavefield, which on its own says nothing about which region of the model
    it covers, so without being told each one would infer the origin from the
    shape and warn -- in the forward and again in every backward recompute.
    Two things stop that: the threaded ``AuxiliaryState`` records segment 0's
    ``origin``, and the driver additionally pins it into the keywords the
    later segments run with. This test pins the property, not either
    mechanism; ``test_origin_is_carried_by_the_threaded_auxiliary_state``
    below pins which one is doing the work today.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=100, dt=0.001)
    vv = v.clone().requires_grad_(True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = run_segments(
            scalar,
            [vv],
            dx,
            dt,
            {"source_amplitudes": src},
            n_segments=5,
            checkpoint=True,
            **kw,
        )
        out.receiver_amplitudes.pow(2).sum().backward()
    assert _origin_warnings(caught) == []


def test_uncheckpointed_segments_also_emit_no_origin_warning():
    """The control: the non-checkpointed path was already silent, and stays so."""
    v, dx, dt, src, kw = _scalar_problem(nt=100, dt=0.001)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_segments(
            scalar, [v], dx, dt, {"source_amplitudes": src}, n_segments=5, **kw
        )
    assert _origin_warnings(caught) == []


def _future_warnings(record):
    """The Deepwave deprecation warnings in a warning record."""
    return [w for w in record if issubclass(w.category, deepwave.DeepwaveFutureWarning)]


@pytest.mark.parametrize("n_segments", [1, 4, 8])
def test_deprecated_keyword_warns_once_per_call_not_once_per_segment(n_segments):
    """A deprecated spelling warns once for the whole segmented call.

    ``checkpoint_segments=`` delegates to the segment driver, which calls the
    propagator once per segment. A deprecated keyword left in the delegated
    arguments would therefore be re-announced by every segment -- and, worse,
    attributed to Deepwave's own re-entry point rather than to the caller,
    because the default ``FutureWarning`` filter deduplicates by location.
    The propagator resolves the deprecated spellings before delegating, so
    the segments only ever see the canonical keywords.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=80, dt=0.001)
    rho = torch.full(v.shape, 2000.0, dtype=DTYPE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deprecated = acoustic(
            v,
            rho,
            dx,
            dt,
            source_amplitudes_p=src,
            source_locations_p=kw["source_locations"],
            receiver_locations_p=kw["receiver_locations"],
            pml_freq=kw["pml_freq"],
            max_vel=kw["max_vel"],
            checkpoint_segments=n_segments,
        )
    warned = _future_warnings(caught)
    assert len(warned) == 2, [str(w.message) for w in warned]
    assert any("source_amplitudes_p" in str(w.message) for w in warned)
    assert any("source_locations_p" in str(w.message) for w in warned)
    # Attributed to this call site, not to Deepwave's segment re-entry.
    assert all(w.filename == __file__ for w in warned), [w.filename for w in warned]
    canonical = acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=src,
        source_locations_volume_rate=kw["source_locations"],
        receiver_locations_p=kw["receiver_locations"],
        pml_freq=kw["pml_freq"],
        max_vel=kw["max_vel"],
        checkpoint_segments=n_segments,
    )
    for name in _get_spec(acoustic).output_names(v.ndim):
        assert elements_equal(getattr(deprecated, name), getattr(canonical, name)), name


def test_scalar_born_bare_receiver_locations_warns_once_per_segmented_call():
    """The same, for ``scalar_born``'s deprecated bare ``receiver_locations``."""
    v, dx, dt, src, kw = _scalar_problem(nt=80, dt=0.001)
    dv = torch.zeros_like(v)
    dv[16:18, 16:18] = 15.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deprecated = scalar_born(
            v,
            dx,
            dt,
            dv=dv,
            source_amplitudes=src,
            source_locations=kw["source_locations"],
            receiver_locations=kw["receiver_locations"],
            pml_freq=kw["pml_freq"],
            max_vel=kw["max_vel"],
            checkpoint_segments=4,
        )
    warned = _future_warnings(caught)
    assert len(warned) == 1, [str(w.message) for w in warned]
    assert "receiver_locations" in str(warned[0].message)
    assert warned[0].filename == __file__
    canonical = scalar_born(
        v,
        dx,
        dt,
        dv=dv,
        source_amplitudes=src,
        source_locations=kw["source_locations"],
        sc_receiver_locations=kw["receiver_locations"],
        pml_freq=kw["pml_freq"],
        max_vel=kw["max_vel"],
        checkpoint_segments=4,
    )
    for name in _get_spec(scalar_born).output_names(v.ndim):
        assert elements_equal(getattr(deprecated, name), getattr(canonical, name)), name


def test_origin_is_carried_by_the_threaded_auxiliary_state():
    """The auxiliary state a segment threads records the origin to restart at.

    This is the primary mechanism behind the silence above, and it is why the
    driver's own origin pin is a backstop rather than the fix: every
    propagator carries an auxiliary state, and ``resolve_initial_state``
    recovers ``origin`` from it whenever the caller gave neither ``origin``
    nor ``survey_pad``. A propagator that stopped carrying one, or a
    auxiliary state that stopped recording ``origin``, would fall back on the
    pin -- and this test would fail first, saying which changed.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=100, dt=0.001)
    out = scalar(v, dx, dt, source_amplitudes=src[..., :50], **kw)
    assert _get_spec(scalar).auxiliary_state_names(2)
    assert out.auxiliary_state.origin is not None
    assert list(out.auxiliary_state.origin) == list(out.info.origin)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scalar(
            v,
            dx,
            dt,
            source_amplitudes=src[..., 50:],
            initial_state={
                "wavefield_0": out.wavefield_0,
                "auxiliary_state": out.auxiliary_state,
            },
            **kw,
        )
    assert _origin_warnings(caught) == []


def test_origin_warning_test_is_not_vacuous():
    """The warning really does fire for a bare restart with no origin.

    Otherwise the two tests above would pass with the pin removed and the
    warning simply never emitted by this problem at all.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=100, dt=0.001)
    first = scalar(v, dx, dt, source_amplitudes=src[..., :50], **kw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scalar(
            v,
            dx,
            dt,
            source_amplitudes=src[..., 50:],
            initial_state={"wavefield_0": first.wavefield_0},
            **kw,
        )
    assert len(_origin_warnings(caught)) == 1


def test_origin_pin_changes_no_number():
    """The pin is a diagnostic fix: the forward and every gradient are bitwise.

    Compared against the non-checkpointed path, which never had the warning
    and never had a pin, so it is an independent witness to the values.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=100, dt=0.001)

    def run(checkpoint):
        vv = v.clone().requires_grad_(True)
        ss = src.clone().requires_grad_(True)
        out = run_segments(
            scalar,
            [vv],
            dx,
            dt,
            {"source_amplitudes": ss},
            n_segments=5,
            checkpoint=checkpoint,
            storage_mode="device",
            **kw,
        )
        (
            out.receiver_amplitudes.pow(2).sum()
            + out.auxiliary_state["psiy_m1"].pow(2).sum()
        ).backward()
        return out, vv.grad, ss.grad

    plain, gv_plain, gs_plain = run(False)
    ckpt, gv_ckpt, gs_ckpt = run(True)
    for name in _get_spec(scalar).output_names(2):
        assert elements_equal(getattr(plain, name), getattr(ckpt, name)), name
    assert elements_equal(gv_plain, gv_ckpt)
    assert elements_equal(gs_plain, gs_ckpt)


def test_origin_pin_respects_an_explicit_origin_and_survey_pad():
    """The pin only fires when the caller gave neither origin nor survey_pad."""
    v, dx, dt, src, kw = _scalar_problem(nt=60, dt=0.001)
    seen = []

    def note(_step, out):
        seen.append(tuple(out.info.origin))

    run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=3,
        checkpoint=True,
        origin=[0, 0],
        segment_callback=note,
        **kw,
    )
    assert seen == [(0, 0)] * 3


# ---------------------------------------------------------------------------
# combine=False cannot deliver user-rate receivers
# ---------------------------------------------------------------------------


def test_combine_false_with_resampling_raises():
    """combine=False plus active resampling is refused, not silently wrong.

    The downsample is a global FFT over the whole time axis, so it cannot be
    applied per segment; retaining every segment's internal-rate chunk until
    the end is exactly the O(nt) storage combine=False exists to avoid.
    """
    v, dx, dt, src, kw, step_ratio = _substepping_scalar_problem(nt=48)
    with pytest.raises(DeepwaveValueError) as excinfo:
        run_segments(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            n_segments=4,
            combine=False,
            **kw,
        )
    message = str(excinfo.value)
    assert f"step_ratio={step_ratio}" in message
    assert "Use combine=True" in message
    assert "pass the internal time step dt=" in message


def test_combine_false_is_allowed_without_resampling():
    """Non-vacuity: the same streaming call at step_ratio == 1 is fine."""
    v, dx, dt, src, kw = _scalar_problem(nt=48, dt=0.001)
    out = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=4,
        combine=False,
        **kw,
    )
    assert out.receiver_amplitudes.shape[-1] == 12


def test_combine_false_is_allowed_with_one_segment_at_step_ratio_gt_1():
    """A single segment does no segment resampling, so there is nothing to refuse.

    This is the path ``deepwave.wavefields.measure`` uses for a
    receiver-only measurement at ``n_segments=1``.
    """
    v, dx, dt, src, kw, _ = _substepping_scalar_problem(nt=48)
    out = run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=1,
        combine=False,
        **kw,
    )
    assert out.receiver_amplitudes.shape[-1] == 48


# ---------------------------------------------------------------------------
# segment_callback, and the named results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("callback", "segment_callback"),
        ("checkpoint_use_reentrant", "use_reentrant"),
        ("slab_pml", "_slab_pml"),
        ("checkpoint_slab_pml", "_slab_pml"),
    ],
)
def test_retired_run_segments_knobs_name_run_segments_and_the_replacement(old, new):
    """Each retired keyword is diagnosed here, not by the propagator.

    All of these fall through ``**kwargs`` if nothing catches them, and the
    propagator has none of them either, so the caller was told that the
    propagator rejected an argument they typed on ``run_segments`` -- with no
    mention of what to type instead.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    with pytest.raises(TypeError) as caught:
        run_segments(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            n_segments=2,
            **{old: True},
            **kw,
        )
    message = str(caught.value)
    assert "run_segments" in message
    assert old in message
    assert new in message
    assert "scalar()" not in message


def test_run_segments_refuses_checkpoint_segments_rather_than_nesting():
    """``n_segments=2`` with ``checkpoint_segments=3`` silently ran six.

    The propagator does have ``checkpoint_segments=``, so forwarding it made
    each of the driver's two segments segment itself into three. The numbers
    were right; the cost was not, and nothing said so.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    with pytest.raises(TypeError) as caught:
        run_segments(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            n_segments=2,
            checkpoint_segments=3,
            **kw,
        )
    message = str(caught.value)
    assert "run_segments" in message
    assert "n_segments" in message


@pytest.mark.parametrize("nt", [20, 40, 60])
@pytest.mark.parametrize("dt", [0.0005, 0.004])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_run_segments_nt_applies_to_whole_record(nt, dt, checkpoint):
    """Explicit duration must not pad/truncate each segment independently."""
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=dt)

    def evaluate(segmented):
        model = v.clone().requires_grad_()
        source = src.clone().requires_grad_()
        if segmented:
            out = run_segments(
                scalar,
                [model],
                dx,
                dt,
                {"source_amplitudes": source},
                n_segments=2,
                nt=nt,
                checkpoint=checkpoint,
                **kw,
            )
        else:
            out = scalar(model, dx, dt, source_amplitudes=source, nt=nt, **kw)
        assert out.info.nt == out.receiver_amplitudes.shape[-1] == nt
        assert out.receiver_amplitudes.norm() > 0
        out.receiver_amplitudes.square().sum().backward()
        assert model.grad is not None
        assert model.grad.norm() > 0
        assert source.grad is not None
        assert source.grad.norm() > 0
        if nt < src.shape[-1]:
            assert torch.count_nonzero(source.grad[..., nt:]) == 0
        return out.receiver_amplitudes, out.wavefield_0, model.grad, source.grad

    for actual, expected in zip(evaluate(True), evaluate(False)):
        assert expected.norm() > 0
        assert _rel(actual, expected) < 1e-8


def test_segment_callback_name():
    """``callback=`` is gone; ``forward_callback=`` still reaches the propagator.

    The old name collided conceptually with the propagators' own
    ``forward_callback``, which ``**kwargs`` forwards through the very same
    call.
    """
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    with pytest.raises(TypeError, match="callback") as caught:
        run_segments(
            scalar,
            [v],
            dx,
            dt,
            {"source_amplitudes": src},
            n_segments=2,
            callback=lambda step, out: None,
            **kw,
        )
    # It is not enough that *something* raised: forwarded through **kwargs
    # this reached the propagator, which never had a ``callback=`` either, so
    # the message blamed the propagator and never named the replacement.
    message = str(caught.value)
    assert "run_segments" in message
    assert "segment_callback" in message
    assert "scalar()" not in message
    boundaries = []
    run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=2,
        segment_callback=lambda step, out: boundaries.append(step),
        **kw,
    )
    assert boundaries == [20, 40]
    steps = []
    run_segments(
        scalar,
        [v],
        dx,
        dt,
        {"source_amplitudes": src},
        n_segments=2,
        forward_callback=lambda state: steps.append(state.step),
        callback_frequency=10,
        **kw,
    )
    assert steps


def test_snapshots_returns_a_named_tuple():
    """``snapshots`` returns a SnapshotResult; positional unpacking is unchanged."""
    v, dx, dt, src, kw = _scalar_problem(nt=40, dt=0.001)
    result = snapshots(scalar, [v], dx, dt, {"source_amplitudes": src}, [10, 20], **kw)
    assert isinstance(result, SnapshotResult)
    assert result._fields == ("output", "snapshots")
    combined, snaps = result
    assert combined is result.output
    assert snaps is result.snapshots
    assert sorted(snaps) == [10, 20]
    assert combined.receiver_amplitudes.shape[-1] == 40


# ---------------------------------------------------------------------------
# A segmented run accepts every call form the propagator accepts
# ---------------------------------------------------------------------------


def _segment_scalar_survey():
    """A tiny scalar problem, plus its survey in both spellings."""
    v = torch.full((30, 30), 1500.0, dtype=torch.float64)
    amplitudes = (
        deepwave.wavelets.ricker(15, 40, 0.004, 0.06).double().reshape(1, 1, -1)
    )
    source_locations = torch.tensor([[[5, 5]]])
    receiver_locations = torch.tensor([[[20, 20], [22, 10]]])
    flat = {
        "source_amplitudes": amplitudes,
        "source_locations": source_locations,
        "receiver_locations": receiver_locations,
    }
    mapping = {
        "sources": {"u": (amplitudes, source_locations)},
        "receivers": {"u": receiver_locations},
    }
    return v, flat, mapping


def test_checkpoint_segments_accepts_the_mapping_call_form() -> None:
    """``sources=``/``receivers=`` under ``checkpoint_segments=`` are accepted.

    The mapping form is the propagator's own call form; a segmented run is
    the same call, so it must accept it -- and produce bit-identical
    numbers, since the mapping is resolved into exactly the flat keywords
    the driver already reads.
    """
    v, flat, mapping = _segment_scalar_survey()
    common = {"pml_width": 5, "max_vel": 1500.0, "checkpoint_segments": 3}
    from_flat = deepwave.scalar(v, 4.0, 0.004, **flat, **common)
    from_mapping = deepwave.scalar(v, 4.0, 0.004, **mapping, **common)
    assert torch.equal(from_flat.receiver_amplitudes, from_mapping.receiver_amplitudes)


def test_checkpoint_segments_mapping_form_gradients_match() -> None:
    """The resolved mapping reaches the caller's own tensors, so grads match."""
    v, flat, mapping = _segment_scalar_survey()
    common = {"pml_width": 5, "max_vel": 1500.0, "checkpoint_segments": 3}
    grads = []
    for survey in (flat, mapping):
        model = v.clone().requires_grad_()
        out = deepwave.scalar(model, 4.0, 0.004, **survey, **common)
        out.receiver_amplitudes.sum().backward()
        grads.append(model.grad)
    assert torch.equal(grads[0], grads[1])


def test_run_segments_accepts_the_mapping_call_form() -> None:
    """``run_segments`` takes the mapping form too, not only its own mapping."""
    v, flat, mapping = _segment_scalar_survey()
    common = {"pml_width": 5, "max_vel": 1500.0}
    expected = deepwave.scalar(v, 4.0, 0.004, **flat, **common, checkpoint_segments=3)
    got = run_segments(
        deepwave.scalar,
        [v],
        4.0,
        0.004,
        {},
        3,
        checkpoint=True,
        **mapping,
        **common,
    )
    assert torch.equal(got.receiver_amplitudes, expected.receiver_amplitudes)


def _segment_elastic_survey():
    """A tiny elastic problem, plus its survey in both spellings."""
    n = 30
    models = (
        torch.full((n, n), 2.25e9, dtype=torch.float64),
        torch.full((n, n), 1.0e9, dtype=torch.float64),
        torch.full((n, n), 1.0 / 2200.0, dtype=torch.float64),
    )
    amplitudes = (
        deepwave.wavelets.ricker(15, 60, 0.0005, 0.06).double().reshape(1, 1, -1) * 1e6
    )
    source_locations = torch.tensor([[[8, 8]]])
    receiver_locations = torch.tensor([[[20, 20]]])
    flat = {
        "source_amplitudes_y": amplitudes,
        "source_locations_y": source_locations,
        "receiver_locations_y": receiver_locations,
    }
    mapping = {
        "sources": {"y": (amplitudes, source_locations)},
        "receivers": {"y": receiver_locations},
    }
    return models, flat, mapping


def test_run_segments_accepts_flat_source_amplitude_keywords() -> None:
    """A flat ``source_amplitudes*`` keyword is lifted into the sliced mapping.

    It used to be refused outright, which made the driver accept fewer call
    forms than the propagator it drives. (The scalar family's flat spelling
    is ``source_amplitudes``, which is already the driver's own parameter,
    so this is only observable on a channel-suffixed propagator.)
    """
    models, flat, _mapping = _segment_elastic_survey()
    common = {"pml_width": 5, "max_vel": 2000.0, "pml_freq": 15.0}
    expected = deepwave.elastic(
        *models, 4.0, 0.0005, **flat, **common, checkpoint_segments=2
    )
    got = run_segments(
        deepwave.elastic,
        list(models),
        4.0,
        0.0005,
        {},
        2,
        checkpoint=True,
        **flat,
        **common,
    )
    assert torch.equal(got.receiver_amplitudes_y, expected.receiver_amplitudes_y)


def test_run_segments_rejects_a_doubly_given_source_amplitude() -> None:
    """One spelling per channel: giving both is an error, not a silent pick."""
    models, flat, _mapping = _segment_elastic_survey()
    with pytest.raises(DeepwaveValueError, match="Give it once"):
        run_segments(
            deepwave.elastic,
            list(models),
            4.0,
            0.0005,
            {"source_amplitudes_y": flat["source_amplitudes_y"]},
            2,
            checkpoint=True,
            **flat,
            pml_width=5,
            max_vel=2000.0,
            pml_freq=15.0,
        )


def test_checkpoint_segments_mapping_form_on_a_multichannel_propagator() -> None:
    """The channel-suffixed propagators agree too, not just the scalar family."""
    n = 30
    lamb = torch.full((n, n), 2.25e9, dtype=torch.float64)
    mu = torch.full((n, n), 1.0e9, dtype=torch.float64)
    buoyancy = torch.full((n, n), 1.0 / 2200.0, dtype=torch.float64)
    amplitudes = (
        deepwave.wavelets.ricker(15, 60, 0.0005, 0.06).double().reshape(1, 1, -1) * 1e6
    )
    source_locations = torch.tensor([[[8, 8]]])
    receiver_locations = torch.tensor([[[20, 20]]])
    common = {
        "pml_width": 5,
        "max_vel": 2000.0,
        "pml_freq": 15.0,
        "checkpoint_segments": 2,
    }
    from_flat = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        4.0,
        0.0005,
        source_amplitudes_y=amplitudes,
        source_locations_y=source_locations,
        receiver_locations_y=receiver_locations,
        **common,
    )
    from_mapping = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        4.0,
        0.0005,
        sources={"y": (amplitudes, source_locations)},
        receivers={"y": receiver_locations},
        **common,
    )
    assert torch.equal(
        from_flat.receiver_amplitudes_y, from_mapping.receiver_amplitudes_y
    )
