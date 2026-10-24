"""Tests for the ergonomics/safety helpers: check_finite and estimate_memory.

These cover a cheap opt-in NaN/Inf check on tensors, and a pre-flight memory
estimator that mirrors StorageManager's and allocate_model_gradients' real
sizing logic.

Also covers PropagatorOutput ergonomics now that ``return_dataclass`` has been
removed (every propagator always returns a ``PropagatorOutput``): pickle
round-tripping, composing with ``torch.utils.checkpoint``, and agreement
between positional indexing/unpacking and named attribute access.
"""

import pickle
import re
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from helpers import elements_equal, run_checkpoint_equivalence

import deepwave
from deepwave import check_finite, estimate_memory

# The backend's private ``IntEnum`` of storage codes, aliased exactly as
# ``src/`` aliases it: the public ``deepwave.StorageMode`` is the *string*
# alias that appears in signatures, and this file passes those strings too.
from deepwave._storage import StorageMode as _StorageModeEnum

# ---------------------------------------------------------------------------
# check_finite
# ---------------------------------------------------------------------------


def test_check_finite_passes_through_finite_tensor() -> None:
    """A fully finite tensor is returned unchanged (same object)."""
    x = torch.tensor([1.0, -2.0, 0.0])
    out = check_finite(x)
    assert out is x


def test_check_finite_raises_on_nan() -> None:
    """A NaN entry raises FloatingPointError, reporting the NaN count."""
    x = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(FloatingPointError) as excinfo:
        check_finite(x, "receiver data")
    message = str(excinfo.value)
    assert "receiver data" in message
    assert "1 NaN" in message
    assert "0 Inf" in message
    assert "diverged" in message


def test_check_finite_raises_on_inf() -> None:
    """An Inf entry raises FloatingPointError, reporting the Inf count."""
    x = torch.tensor([1.0, float("inf"), -float("inf")])
    with pytest.raises(FloatingPointError) as excinfo:
        check_finite(x, "v.grad")
    message = str(excinfo.value)
    assert "v.grad" in message
    assert "0 NaN" in message
    assert "2 Inf" in message


def test_check_finite_default_name() -> None:
    """The default label is used when `name` is not given."""
    x = torch.tensor([float("nan")])
    with pytest.raises(FloatingPointError, match=re.escape("tensor contains")):
        check_finite(x)


def test_check_finite_top_level_export() -> None:
    """check_finite is importable as deepwave.check_finite."""
    assert deepwave.check_finite is check_finite
    assert deepwave.check_finite(torch.zeros(3)) is not None


def test_check_finite_on_real_propagator_output() -> None:
    """End-to-end: check_finite accepts finite receiver data from scalar()."""
    v = 1500 * torch.ones(10, 10)
    out = deepwave.scalar(
        v,
        4.0,
        0.001,
        source_amplitudes=torch.zeros(1, 1, 5),
        source_locations=torch.tensor([[[0, 0]]]),
        receiver_locations=torch.tensor([[[0, 5]]]),
        pml_freq=25.0,
    )
    receiver_data = deepwave.check_finite(out[-1], "receiver data")
    assert torch.equal(receiver_data, out[-1])


# ---------------------------------------------------------------------------
# check_output_finite
# ---------------------------------------------------------------------------


def _diverging_scalar(**extra: Any) -> deepwave.PropagatorOutput:
    """Finite forcing overflows float32 even with a valid CFL time step.

    Understating max_vel is rejected before propagation. Instead, use a
    representable source whose accumulated response exceeds the dtype range
    to exercise the output guard independently of input validation.
    """
    return deepwave.scalar(
        1500 * torch.ones(20, 20),
        4.0,
        0.001,
        max_vel=1500.0,
        source_amplitudes=torch.full((1, 1, 60), 1e38),
        source_locations=torch.tensor([[[10, 10]]]),
        receiver_locations=torch.tensor([[[10, 15]]]),
        pml_freq=25.0,
        **extra,
    )


def _tiny_call(name: str, ndim: int, **extra: Any) -> deepwave.PropagatorOutput:
    """One tiny, healthy propagation of ``name`` in ``ndim`` dimensions."""
    spec = deepwave.spec(name)
    shape = (12,) * ndim
    models = {
        "scalar": lambda: [torch.full(shape, 1500.0)],
        "acoustic": lambda: [torch.full(shape, 1500.0), torch.full(shape, 2000.0)],
        "elastic": lambda: [
            torch.full(shape, 2.0e9),
            torch.full(shape, 1.0e9),
            torch.full(shape, 1.0 / 2000.0),
        ],
    }[name]()
    channel = spec.source_channels(ndim)[0]
    amplitudes = deepwave.wavelets.ricker(15.0, 12, 0.0005, 0.006).reshape(1, 1, -1)
    locations = torch.tensor([[[6] * ndim]])
    kwargs: dict[str, Any] = {
        spec.source_amplitudes_kwarg(channel): amplitudes,
        spec.source_locations_kwarg(channel): locations,
        spec.receiver_locations_kwarg(spec.receiver_channels(ndim)[0]): locations,
        "pml_freq": 15.0,
        "pml_width": 2,
        "accuracy": 2,
        **extra,
    }
    return getattr(deepwave, name)(*models, 5.0, 0.0005, **kwargs)


def test_check_output_finite_raises_on_a_diverged_run() -> None:
    """A diverged run raises, naming the field, the count and the way out."""
    with pytest.raises(deepwave.DeepwaveFloatingPointError) as excinfo:
        _diverging_scalar()
    message = str(excinfo.value)
    assert "deepwave.scalar produced non-finite output" in message
    assert "'wavefield_0'" in message
    assert "non-finite values" in message
    assert "nt=60" in message
    assert "diverged" in message
    assert "check_output_finite=False" in message


def test_check_output_finite_false_returns_the_non_finite_output() -> None:
    """With the check off, the same run returns its NaNs instead of raising."""
    out = _diverging_scalar(check_output_finite=False)
    assert not torch.isfinite(out.wavefield_0).all()


def test_check_output_finite_error_is_catchable_three_ways() -> None:
    """The raise is a DeepwaveError, a FloatingPointError and an Exception."""
    for exception_type in (
        deepwave.DeepwaveError,
        FloatingPointError,
        Exception,
    ):
        with pytest.raises(exception_type):
            _diverging_scalar()


@pytest.mark.parametrize("name", ["scalar", "acoustic", "elastic"])
@pytest.mark.parametrize("ndim", [2, 3])
def test_check_output_finite_does_not_change_a_healthy_result(
    name: str, ndim: int
) -> None:
    """A healthy run returns bit-identical outputs with the check on and off.

    And says nothing extra while doing it: whatever a call warns about, it
    warns about identically either way, so the check itself is silent.
    """
    with warnings.catch_warnings(record=True) as checked_warnings:
        warnings.simplefilter("always")
        checked = _tiny_call(name, ndim)
    with warnings.catch_warnings(record=True) as unchecked_warnings:
        warnings.simplefilter("always")
        unchecked = _tiny_call(name, ndim, check_output_finite=False)
    assert [str(w.message) for w in checked_warnings] == [
        str(w.message) for w in unchecked_warnings
    ]
    assert len(checked) == len(unchecked)
    assert checked._names == unchecked._names
    for name_, a, b in zip(checked._names, checked, unchecked):
        assert elements_equal(a, b), f"{name} ndim={ndim}: {name_} differs"


def test_check_output_finite_is_forwarded_by_run_segments() -> None:
    """``run_segments`` passes the flag through to each segment it runs."""
    kwargs: dict[str, Any] = {
        "max_vel": 1500.0,
        "source_locations": torch.tensor([[[10, 10]]]),
        "receiver_locations": torch.tensor([[[10, 15]]]),
        "pml_freq": 25.0,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(deepwave.DeepwaveFloatingPointError):
            deepwave.checkpointing.run_segments(
                deepwave.scalar,
                [1500 * torch.ones(20, 20)],
                4.0,
                0.001,
                {"source_amplitudes": torch.full((1, 1, 60), 1e38)},
                2,
                **kwargs,
            )
        out = deepwave.checkpointing.run_segments(
            deepwave.scalar,
            [1500 * torch.ones(20, 20)],
            4.0,
            0.001,
            {"source_amplitudes": torch.full((1, 1, 60), 1e38)},
            2,
            check_output_finite=False,
            **kwargs,
        )
    assert not torch.isfinite(out.wavefield_0).all()


# ---------------------------------------------------------------------------
# estimate_memory
# ---------------------------------------------------------------------------


def test_estimate_memory_top_level_export() -> None:
    """estimate_memory is importable as deepwave.estimate_memory."""
    assert deepwave.estimate_memory is estimate_memory


def test_estimate_memory_padded_shape_matches_fd_and_pml_padding() -> None:
    """padded_model_shape adds 2*(accuracy//2) and the PML width per side."""
    est = estimate_memory(
        "scalar",
        model_shape=(100, 200),
        nt=50,
        n_shots=2,
        accuracy=4,
        pml_width=20,
    )
    # fd_pad = 4 // 2 = 2 per side; pml_width = 20 per side.
    assert est["padded_model_shape"] == (100 + 2 * 2 + 2 * 20, 200 + 2 * 2 + 2 * 20)
    assert est["padded_model_elements"] == (100 + 44) * (200 + 44)


def test_estimate_memory_per_side_pml_width() -> None:
    """A per-face pml_width list is honoured, like set_pml_width."""
    est = estimate_memory(
        "scalar",
        model_shape=(100, 200),
        nt=50,
        n_shots=2,
        accuracy=2,
        pml_width=[1, 2, 3, 4],
    )
    # fd_pad = 2 // 2 = 1 per side.
    assert est["padded_model_shape"] == (100 + 2 * 1 + 1 + 2, 200 + 2 * 1 + 3 + 4)


def test_estimate_memory_backward_storage_formula_device_mode() -> None:
    """backward_storage_bytes mirrors StorageManager's uncompressed device sizing.

    nt // model_gradient_sampling_interval sampled steps, times n_shots,
    times the padded model's element count, times dtype size (4 for
    float32), times the number of models requiring a gradient.
    """
    est = estimate_memory(
        "scalar",
        model_shape=(10, 10),
        nt=100,
        n_shots=3,
        dtype=torch.float32,
        accuracy=2,
        pml_width=0,
        model_gradient_sampling_interval=2,
        storage_mode="device",
        n_models_requiring_grad=1,
    )
    padded_elements = 12 * 12  # fd_pad=1 each side, no PML
    expected_steps = 100 // 2
    expected_bytes = expected_steps * 3 * padded_elements * 4
    assert est["padded_model_elements"] == padded_elements
    assert est["backward_storage_bytes"] == expected_bytes


def test_estimate_memory_step_ratio_does_not_change_backward_storage() -> None:
    """step_ratio cancels out of the backward-storage formula (see docstring).

    The internal nt actually simulated is nt * step_ratio, and the number of
    stored steps is (nt * step_ratio) // (step_ratio *
    model_gradient_sampling_interval), which equals nt //
    model_gradient_sampling_interval for any step_ratio >= 1.
    """
    kwargs = {
        "model_shape": (10, 10),
        "nt": 100,
        "n_shots": 3,
        "accuracy": 2,
        "pml_width": 0,
        "model_gradient_sampling_interval": 2,
        "storage_mode": "device",
    }
    with pytest.raises(TypeError, match="step_ratio"):
        estimate_memory("scalar", step_ratio=1, **kwargs)  # type: ignore[call-arg]
    # It cancelled out exactly, which is why removing it changes no answer.
    assert estimate_memory("scalar", **kwargs)["backward_storage_bytes"] > 0


def test_estimate_memory_storage_mode_none_has_no_backward_cost() -> None:
    """storage_mode='none' matches StorageManager.allocate: nothing stored."""
    est = estimate_memory(
        "scalar",
        model_shape=(10, 10),
        nt=100,
        n_shots=3,
        storage_mode="none",
    )
    assert est["backward_storage_bytes"] == 0
    assert est["gradient_scratch_bytes"] == 0
    # Forward wavefield buffers are still needed regardless of storage mode.
    assert est["forward_wavefields_bytes"] > 0


def test_estimate_memory_gradient_scratch_needs_more_than_one_thread_or_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No per-thread/per-shot scratch buffer when only one worker is active.

    Mirrors allocate_model_gradients: on CPU, a temporary buffer is only
    allocated when min(n_shots, n_threads) > 1. ``n_threads`` is not a
    parameter of ``estimate_memory`` (it reflects the real
    ``torch.get_num_threads()``), so the thread count is pinned here via
    monkeypatch to make the test deterministic.
    """
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    est = estimate_memory(
        "scalar",
        model_shape=(10, 10),
        nt=10,
        n_shots=1,
        storage_mode="device",
    )
    assert est["gradient_scratch_bytes"] == 0

    est_multi = estimate_memory(
        "scalar",
        model_shape=(10, 10),
        nt=10,
        n_shots=4,
        storage_mode="device",
    )
    assert est_multi["gradient_scratch_bytes"] > 0


def test_estimate_memory_gradient_scratch_cuda_uses_n_shots() -> None:
    """On CUDA, allocate_model_gradients sizes the scratch buffer by n_shots.

    ``n_threads`` is forced to 1 on CUDA regardless of the host's thread
    count, so no monkeypatching is needed here.
    """
    est = estimate_memory(
        "scalar",
        model_shape=(10, 10),
        nt=10,
        n_shots=4,
        storage_mode="device",
        device="cuda",
    )
    padded_elements = est["padded_model_elements"]
    assert est["gradient_scratch_bytes"] == 4 * padded_elements * 4


def test_estimate_memory_n_models_requiring_grad_scales_storage() -> None:
    """Doubling the stored-history slot count doubles backward_storage_bytes.

    Named models rather than a bare count, and one propagator rather than
    two, so that the comparison is genuinely one slot against two:
    ``n_models_requiring_grad`` is a worst case over that many models, and on
    most propagators one model already implies several slots.
    """
    base = {"model_shape": (10, 10), "nt": 10, "n_shots": 2}
    est1 = estimate_memory("scalar_born", **base, models_requiring_grad=["dv"])
    est2 = estimate_memory("scalar_born", **base, models_requiring_grad=["v", "dv"])
    assert est2["backward_storage_bytes"] == 2 * est1["backward_storage_bytes"]


def test_estimate_memory_total_is_sum_of_components() -> None:
    """total_bytes is exactly the sum of the three reported components."""
    est = estimate_memory("scalar", model_shape=(20, 30), nt=40, n_shots=2)
    assert est["total_bytes_lower_bound"] == (
        est["forward_wavefields_bytes"]
        + est["backward_storage_bytes"]
        + est["gradient_scratch_bytes"]
    )


def test_estimate_memory_invalid_storage_mode_raises() -> None:
    """An unrecognised storage_mode raises ValueError, like resolve_storage_mode."""
    with pytest.raises(ValueError, match="storage_mode must be"):
        estimate_memory(
            "scalar", model_shape=(10, 10), nt=10, n_shots=1, storage_mode="bogus"
        )


def test_estimate_memory_empty_model_shape_raises() -> None:
    """An empty model_shape raises ValueError."""
    with pytest.raises(ValueError, match="model_shape must not be empty"):
        estimate_memory("scalar", model_shape=(), nt=10, n_shots=1)


def test_estimate_memory_zero_model_gradient_sampling_interval_raises() -> None:
    """Zero is rejected, as it is by the propagators (it is a divisor)."""
    with pytest.raises(
        ValueError, match="model_gradient_sampling_interval must be >= 1"
    ):
        estimate_memory(
            "scalar",
            model_shape=(10, 10),
            nt=10,
            n_shots=1,
            model_gradient_sampling_interval=0,
        )


def test_estimate_memory_returns_storage_mode_enum_consistent_values() -> None:
    """Sanity check that the internal StorageMode mapping used is the real enum."""
    assert _StorageModeEnum.NONE.value == 3


# ---------------------------------------------------------------------------
# estimate_memory: storage_compression=
# ---------------------------------------------------------------------------


def _real_storage_bytes(
    padded_shape: tuple[int, ...],
    dtype: torch.dtype,
    n_shots: int,
    nt: int,
    storage_mode: _StorageModeEnum,
    storage_compression: bool,
    n_models_requiring_grad: int,
) -> int:
    """Sums the actual bytes StorageManager.allocate allocates for its buffers.

    Ground truth for the ``storage_compression`` arithmetic below: one
    StorageManager per gradient-requiring model, exactly like the real
    propagators construct one per model that requires a gradient.
    """
    from deepwave._storage import StorageManager

    total = 0
    for _ in range(n_models_requiring_grad):
        manager = StorageManager(
            padded_shape,
            dtype,
            n_shots,
            nt,
            1,
            storage_mode,
            storage_compression,
            "",
            torch.device("cpu"),
            False,
        )
        manager.allocate(True)
        storage = manager.storages[-1]
        for tensor in (
            storage.store_1a,
            storage.store_1b,
            storage.store_2,
            storage.store_3,
        ):
            total += tensor.numel() * tensor.element_size()
    return total


@pytest.mark.parametrize(
    ("padded_shape", "dtype", "n_shots", "nt", "n_models_requiring_grad"),
    [
        ((20, 30), torch.float32, 3, 40, 1),
        ((20, 30), torch.float32, 3, 40, 2),
        ((17, 23, 11), torch.float64, 5, 77, 2),
        ((64, 64), torch.float32, 1, 200, 1),
        ((9, 9), torch.float32, 2, 13, 1),  # smaller than one 8-per-side block
    ],
)
@pytest.mark.parametrize(
    "storage_mode_str",
    [
        "device",
        pytest.param(
            "cpu",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is required for the pinned offload allocation",
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize("storage_compression", [False, True])
def test_estimate_memory_storage_compression_matches_storage_manager(
    padded_shape: tuple[int, ...],
    dtype: torch.dtype,
    n_shots: int,
    nt: int,
    n_models_requiring_grad: int,
    storage_mode_str: str,
    storage_compression: bool,
) -> None:
    """backward_storage_bytes matches StorageManager's real allocations exactly.

    Uses ``accuracy=2, pml_width=0`` (1-element finite-difference padding per
    side) and an ``unpadded`` ``model_shape`` chosen so ``padded_model_shape``
    comes out to ``padded_shape`` exactly, so the two byte counts are directly
    comparable.
    """
    fd_pad = 1
    unpadded = tuple(s - 2 * fd_pad for s in padded_shape)
    storage_mode_enum = {
        "device": _StorageModeEnum.DEVICE,
        "cpu": _StorageModeEnum.CPU,
    }[storage_mode_str]
    real = _real_storage_bytes(
        padded_shape,
        dtype,
        n_shots,
        nt,
        storage_mode_enum,
        storage_compression,
        n_models_requiring_grad,
    )
    # ``_real_storage_bytes`` builds exactly ``n_models_requiring_grad``
    # StorageManagers, so the estimate has to charge exactly that many slots
    # -- not a worst case over that many models. ``scalar_born`` is named
    # with the models that produce precisely that count: ``dv`` alone drives
    # one slot, ``v`` and ``dv`` together drive two.
    est = estimate_memory(
        "scalar_born",
        model_shape=unpadded,
        nt=nt,
        n_shots=n_shots,
        dtype=dtype,
        accuracy=2,
        pml_width=0,
        storage_mode=storage_mode_str,
        models_requiring_grad=(["dv"] if n_models_requiring_grad == 1 else ["v", "dv"]),
        storage_compression=storage_compression,
    )
    assert est["padded_model_shape"] == padded_shape
    assert est["backward_storage_bytes"] == real


def test_estimate_memory_storage_compression_disk_costed_same_as_device() -> None:
    """Disk stays costed identically to device with compression on or off.

    ``storage_mode="disk"``'s real footprint is much smaller (it streams
    per-step to files rather than keeping the whole history resident), but
    the documented simplification treats it as an upper bound equal to
    device mode; that equivalence must hold with compression too.
    """
    for storage_compression in (False, True):
        kwargs: dict[str, Any] = {
            "model_shape": (18, 28),
            "nt": 40,
            "n_shots": 3,
            "accuracy": 2,
            "pml_width": 0,
            "n_models_requiring_grad": 2,
            "storage_compression": storage_compression,
        }
        est_disk = estimate_memory("acoustic", storage_mode="disk", **kwargs)
        est_device = estimate_memory("acoustic", storage_mode="device", **kwargs)
        assert (
            est_disk["backward_storage_bytes"] == est_device["backward_storage_bytes"]
        )


def test_estimate_memory_storage_compression_ignored_when_storage_mode_none() -> None:
    """storage_compression is a no-op under storage_mode='none' (per StorageManager)."""
    est_off = estimate_memory(
        "scalar",
        model_shape=(10, 10),
        nt=100,
        n_shots=3,
        storage_mode="none",
        storage_compression=False,
    )
    est_on = estimate_memory(
        "scalar",
        model_shape=(10, 10),
        nt=100,
        n_shots=3,
        storage_mode="none",
        storage_compression=True,
    )
    assert est_off == est_on
    assert est_on["backward_storage_bytes"] == 0


@pytest.mark.parametrize("storage_mode_str", ["device", "cpu", "disk"])
def test_estimate_memory_storage_compression_reduces_backward_storage(
    storage_mode_str: str,
) -> None:
    """Compression shrinks backward_storage_bytes for a grid large enough to matter.

    At tiny grids the two full-size single-shot scratch buffers device/disk
    mode keeps alongside the compressed history (or, for CPU, the extra
    compressed shot buffer) can outweigh the quantisation saving at low
    ``nt``; this is checked at a scale large enough (``nt=500``) that the
    history term dominates, matching the intended use case (large 3-D runs).
    """
    kwargs: dict[str, Any] = {
        "model_shape": (200, 200),
        "nt": 500,
        "n_shots": 4,
        "accuracy": 4,
        "pml_width": 20,
        "storage_mode": storage_mode_str,
    }
    est_off = estimate_memory("acoustic", storage_compression=False, **kwargs)
    est_on = estimate_memory("acoustic", storage_compression=True, **kwargs)
    assert est_on["backward_storage_bytes"] < est_off["backward_storage_bytes"]


# ---------------------------------------------------------------------------
# estimate_memory: propagator=
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("propagator", "ndim", "expected"),
    [
        ("scalar", 2, 6),  # 2 + 2*2
        ("scalar", 3, 8),  # 2 + 2*3
        ("scalar_born", 2, 12),  # 2 * (2 + 2*2)
        ("scalar_born", 3, 16),  # 2 * (2 + 2*3)
        ("acoustic", 2, 7),  # 1 + 3*2
        ("acoustic", 3, 10),  # 1 + 3*3
        ("elastic", 2, 13),
        ("elastic", 3, 27),
    ],
)
def test_estimate_memory_propagator_fills_n_forward_wavefields(
    propagator, ndim, expected
) -> None:
    """propagator= fills the forward-wavefield count from the docstring's table."""
    model_shape = (10,) * ndim
    est_propagator = estimate_memory(
        model_shape=model_shape, nt=10, n_shots=2, propagator=propagator
    )
    # dtype defaults to float32 (4 bytes/element); padded_model_elements is
    # 10 (+ 2*2 fd_pad + 2*20 pml_width by default) per dimension.
    padded_elements = est_propagator["padded_model_elements"]
    assert (
        est_propagator["forward_wavefields_bytes"] == expected * 2 * padded_elements * 4
    )


def test_estimate_memory_unrecognized_propagator_raises() -> None:
    """An unrecognised propagator name raises, naming every valid one.

    ``estimate_memory`` used to know only the four propagators in a
    hand-maintained table; it now resolves the propagator through
    :func:`deepwave.spec` and covers all nine, so the message lists all
    nine too.
    """
    with pytest.raises(ValueError, match="not the name of a Deepwave") as excinfo:
        estimate_memory(model_shape=(10, 10), nt=10, n_shots=2, propagator="bogus")
    message = str(excinfo.value)
    assert "'bogus'" in message
    for name in deepwave._engine.PROPAGATOR_NAMES:
        assert name in message


def test_estimate_memory_accepts_the_three_propagator_spellings() -> None:
    """The callable, the Module class and the name all name one propagator."""
    base = {"model_shape": (10, 10), "nt": 10, "n_shots": 2}
    by_name = estimate_memory(**base, propagator="elastic")
    for form in (deepwave.elastic, deepwave.Elastic):
        assert estimate_memory(**base, propagator=form) == by_name


def test_estimate_memory_requires_propagator() -> None:
    """``propagator`` is required, so no answer is ever a placeholder.

    It used to be optional, and omitting it substituted a hard-coded guess of
    three forward wavefields -- a documented lie that no propagator's spec
    agreed with. Every propagator now answers exactly, so the guess is gone
    and its absence is an error rather than a wrong number.
    """
    with pytest.raises(TypeError, match="propagator"):
        estimate_memory(model_shape=(10, 10), nt=10, n_shots=2)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# estimate_memory: models_requiring_grad= (exact backward-storage slot count)
# ---------------------------------------------------------------------------

# The user-facing model arguments of each propagator that estimate_memory
# knows about, in the order its module's flag functions expect them.
_USER_MODELS = {
    "scalar": ("v",),
    "scalar_born": ("v", "dv"),
    "acoustic": ("v", "rho"),
    "elastic": ("lamb", "mu", "buoyancy"),
    "acoustic_born": ("v", "rho", "dv", "drho"),
    "acoustic_vti": ("c11", "c13", "c33", "buoyancy"),
    "viscoacoustic": ("v", "rho", "q"),
    "elastic_born": ("lamb", "mu", "buoyancy", "dlamb", "dmu", "dbuoyancy"),
    "elastic_vti": ("c11", "c13", "c33", "c55", "buoyancy"),
}

# The dimensionalities each retained propagator accepts.
_TABLE_NDIMS = {
    "scalar": (1, 2, 3),
    "scalar_born": (1, 2, 3),
    "acoustic": (1, 2, 3),
    "elastic": (1, 2, 3),
    "acoustic_born": (1, 2, 3),
    "acoustic_vti": (2, 3),
    "viscoacoustic": (1, 2, 3),
    "elastic_born": (1, 2, 3),
    "elastic_vti": (2, 3),
}


def _propagate_tiny(propagator: str, ndim: int, requires_grad: dict[str, bool]) -> None:
    """Runs a two-step propagation of ``propagator`` on a tiny grid."""
    n = 12
    shape = (n,) * ndim
    nt = 2
    dt = 0.0005
    dx = 5.0
    amplitudes = torch.zeros(1, 1, nt)
    locations = torch.tensor([[[n // 2] * ndim]])
    receivers = torch.tensor([[[n // 2 - 2] + [n // 2] * (ndim - 1)]])
    common: dict[str, Any] = {"pml_width": 3, "pml_freq": 25.0, "accuracy": 2}
    # Channel names, and the keyword prefixes that carry them, differ between
    # propagators (acoustic_vti has no pressure *receiver*; the Born families
    # spell theirs "sc_receiver_..."), so use the channel-mapping form, which
    # is the same on all nine.
    spec = deepwave.spec(propagator)
    src_channels = spec.source_channels(ndim)
    recv_channels = spec.receiver_channels(ndim)
    channels: dict[str, Any] = {}
    if src_channels:
        channels["sources"] = {src_channels[0]: (amplitudes, locations)}
    else:
        channels["source_amplitudes"] = amplitudes
        channels["source_locations"] = locations
    if recv_channels:
        channels["receivers"] = {recv_channels[0]: receivers}
    else:
        channels["receiver_locations"] = receivers

    if propagator in ("scalar", "scalar_born"):
        v = (1500 * torch.ones(shape)).requires_grad_(requires_grad["v"])
        if propagator == "scalar":
            deepwave.scalar(
                v,
                dx,
                dt,
                **channels,
                **common,
            )
        else:
            dv = torch.zeros(shape).requires_grad_(requires_grad["dv"])
            deepwave.scalar_born(
                v,
                dx,
                dt,
                dv=dv,
                **channels,
                **common,
            )
    elif propagator == "acoustic":
        v = (1500 * torch.ones(shape)).requires_grad_(requires_grad["v"])
        rho = (1000 * torch.ones(shape)).requires_grad_(requires_grad["rho"])
        deepwave.acoustic(
            v,
            rho,
            dx,
            dt,
            **channels,
            **common,
        )
    elif propagator == "elastic":
        lamb = (3.3e9 * torch.ones(shape)).requires_grad_(requires_grad["lamb"])
        mu = (1.5e9 * torch.ones(shape)).requires_grad_(requires_grad["mu"])
        buoyancy = (torch.ones(shape) / 2000).requires_grad_(requires_grad["buoyancy"])
        deepwave.elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            **channels,
            **common,
        )
    elif propagator == "acoustic_born":
        v = (1500 * torch.ones(shape)).requires_grad_(requires_grad["v"])
        rho = (1000 * torch.ones(shape)).requires_grad_(requires_grad["rho"])
        dv = torch.zeros(shape).requires_grad_(requires_grad["dv"])
        drho = torch.zeros(shape).requires_grad_(requires_grad["drho"])
        deepwave.acoustic_born(
            v,
            rho,
            dx,
            dt,
            dv=dv,
            drho=drho,
            **channels,
            **common,
        )
    elif propagator == "acoustic_vti":
        c11 = (1e10 * torch.ones(shape)).requires_grad_(requires_grad["c11"])
        c13 = (5e9 * torch.ones(shape)).requires_grad_(requires_grad["c13"])
        c33 = (9e9 * torch.ones(shape)).requires_grad_(requires_grad["c33"])
        buoyancy = (torch.ones(shape) / 2000).requires_grad_(requires_grad["buoyancy"])
        deepwave.acoustic_vti(
            c11,
            c13,
            c33,
            buoyancy,
            dx,
            dt,
            **channels,
            **common,
        )
    elif propagator == "viscoacoustic":
        v = (1500 * torch.ones(shape)).requires_grad_(requires_grad["v"])
        rho = (1000 * torch.ones(shape)).requires_grad_(requires_grad["rho"])
        q = (50 * torch.ones(shape)).requires_grad_(requires_grad["q"])
        deepwave.viscoacoustic(
            v,
            rho,
            q,
            dx,
            dt,
            **channels,
            q_freq=25.0,
            **common,
        )
    elif propagator == "elastic_born":
        lamb = (3.3e9 * torch.ones(shape)).requires_grad_(requires_grad["lamb"])
        mu = (1.5e9 * torch.ones(shape)).requires_grad_(requires_grad["mu"])
        buoyancy = (torch.ones(shape) / 2000).requires_grad_(requires_grad["buoyancy"])
        dlamb = torch.zeros(shape).requires_grad_(requires_grad["dlamb"])
        dmu = torch.zeros(shape).requires_grad_(requires_grad["dmu"])
        dbuoyancy = torch.zeros(shape).requires_grad_(requires_grad["dbuoyancy"])
        deepwave.elastic_born(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            dlamb=dlamb,
            dmu=dmu,
            dbuoyancy=dbuoyancy,
            **channels,
            **common,
        )
    elif propagator == "elastic_vti":
        c11 = (1e10 * torch.ones(shape)).requires_grad_(requires_grad["c11"])
        c13 = (5e9 * torch.ones(shape)).requires_grad_(requires_grad["c13"])
        c33 = (9e9 * torch.ones(shape)).requires_grad_(requires_grad["c33"])
        c55 = (2e9 * torch.ones(shape)).requires_grad_(requires_grad["c55"])
        buoyancy = (torch.ones(shape) / 2000).requires_grad_(requires_grad["buoyancy"])
        extra: dict[str, Any] = {}
        if ndim == 3:
            extra["c66"] = (2.5e9 * torch.ones(shape)).requires_grad_(
                requires_grad["c66"]
            )
        deepwave.elastic_vti(
            c11,
            c13,
            c33,
            c55,
            buoyancy,
            dx,
            dt,
            source_amplitudes_pressure_rate=amplitudes,
            source_locations_pressure_rate=locations,
            receiver_locations_p=receivers,
            **extra,
            **common,
        )
    else:
        raise AssertionError(f"no tiny-propagation recipe for {propagator}")


def _real_storage_slots(
    propagator: str,
    ndim: int,
    requires_grad: dict[str, bool],
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    """The number of populated backward-storage slots a real forward pass takes.

    Counts the ``StorageManager.allocate(requires_grad=True)`` calls the engine
    makes, which is the ground truth ``estimate_memory``'s stored-history term
    is supposed to mirror.
    """
    from deepwave import _storage

    allocated: list[bool] = []
    real_allocate = _storage.StorageManager.allocate

    def counting_allocate(
        self: Any, requires_grad_flag: bool, *args: Any, **kwargs: Any
    ) -> Any:
        allocated.append(bool(requires_grad_flag))
        return real_allocate(self, requires_grad_flag, *args, **kwargs)

    monkeypatch.setattr(_storage.StorageManager, "allocate", counting_allocate)
    _propagate_tiny(propagator, ndim, requires_grad)
    return sum(allocated)


def _estimated_storage_slots(
    propagator: str, ndim: int, models_requiring_grad: list[str]
) -> int:
    """The slot count implied by estimate_memory's backward_storage_bytes."""
    nt = 7
    est = estimate_memory(
        model_shape=(10,) * ndim,
        nt=nt,
        n_shots=1,
        propagator=propagator,
        models_requiring_grad=models_requiring_grad,
    )
    per_slot = nt * est["padded_model_elements"] * 4
    assert est["backward_storage_bytes"] % per_slot == 0
    return est["backward_storage_bytes"] // per_slot


@pytest.mark.parametrize("propagator", sorted(_USER_MODELS))
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_estimate_memory_storage_slots_match_the_engine(
    propagator: str, ndim: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The anti-drift test: the estimate's slot count is the engine's slot count.

    For every propagator, dimensionality, and combination of models requiring
    a gradient, the number of stored wavefield histories
    ``estimate_memory(models_requiring_grad=...)`` charges for must equal the
    number of populated ``StorageManager`` slots a real forward pass of that
    configuration allocates. This is what stops the single-sourced
    ``storage_slot_flags`` functions and the estimator from diverging.
    """
    if ndim not in _TABLE_NDIMS[propagator]:
        pytest.skip(f"{propagator} does not support ndim={ndim}")
    names = _USER_MODELS[propagator]
    if propagator == "elastic_vti" and ndim == 3:
        names = (*names, "c66")
    for combination in range(2 ** len(names)):
        requires_grad = {
            name: bool(combination & (1 << i)) for i, name in enumerate(names)
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            real = _real_storage_slots(propagator, ndim, requires_grad, monkeypatch)
        estimated = _estimated_storage_slots(
            propagator, ndim, [name for name in names if requires_grad[name]]
        )
        assert estimated == real, (
            f"{propagator} {ndim}-D with {requires_grad}: estimate_memory "
            f"charges for {estimated} stored histories, engine allocates {real}"
        )


@pytest.mark.parametrize(
    ("propagator", "ndim", "expected"),
    [
        # acoustic: rho alone already stores K's history plus one buoyancy
        # history per axis, so one model is worth 1 + ndim slots.
        ("acoustic", 2, 3),
        ("acoustic", 3, 4),
        # elastic: mu alone is the worst single model (every slot but the
        # buoyancy ones), lamb or buoyancy alone are cheaper.
        ("elastic", 2, 3),
        ("elastic", 3, 6),
        ("scalar", 3, 1),
    ],
)
def test_estimate_memory_n_models_requiring_grad_is_a_worst_case(
    propagator: str, ndim: int, expected: int
) -> None:
    """With propagator but no names, the slot count is a worst-case upper bound."""
    nt = 7
    est = estimate_memory(
        model_shape=(10,) * ndim,
        nt=nt,
        n_shots=1,
        propagator=propagator,
        n_models_requiring_grad=1,
    )
    per_slot = nt * est["padded_model_elements"] * 4
    assert est["backward_storage_bytes"] == expected * per_slot
    # It really is an upper bound over every single-model choice.
    for name in _USER_MODELS[propagator]:
        assert _estimated_storage_slots(propagator, ndim, [name]) <= expected


def test_estimate_memory_unrecognised_model_name_raises() -> None:
    """An unknown model name names the propagator's real models."""
    with pytest.raises(
        ValueError,
        match=r"Unrecognised model name\(s\) \['bogus'\].*'acoustic'.*\['v', 'rho'\]",
    ):
        estimate_memory(
            "acoustic",
            model_shape=(10, 10),
            nt=10,
            n_shots=1,
            models_requiring_grad=["v", "bogus"],
        )


def test_estimate_memory_duplicate_model_name_raises() -> None:
    """A repeated model name is a mistake, not two slots."""
    with pytest.raises(ValueError, match="must not contain duplicates"):
        estimate_memory(
            "acoustic",
            model_shape=(10, 10),
            nt=10,
            n_shots=1,
            models_requiring_grad=["v", "v"],
        )


def test_estimate_memory_inconsistent_model_count_raises() -> None:
    """Giving both a count and names that disagree is an error, not a silent choice."""
    with pytest.raises(ValueError, match="inconsistent with models_requiring_grad"):
        estimate_memory(
            "acoustic",
            model_shape=(10, 10),
            nt=10,
            n_shots=1,
            models_requiring_grad=["v", "rho"],
            n_models_requiring_grad=1,
        )


def test_estimate_memory_too_many_models_raises() -> None:
    """A count larger than the propagator has models is an error."""
    with pytest.raises(ValueError, match="must be between 0 and 1 for propagator"):
        estimate_memory(
            "scalar",
            model_shape=(10, 10),
            nt=10,
            n_shots=1,
            n_models_requiring_grad=2,
        )


def test_estimate_memory_negative_model_count_raises() -> None:
    """A negative count is rejected before anything else looks at it."""
    with pytest.raises(
        ValueError, match="n_models_requiring_grad must be non-negative"
    ):
        estimate_memory(
            "scalar", model_shape=(10, 10), nt=10, n_shots=1, n_models_requiring_grad=-1
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_estimate_memory_close_to_measured_peak_acoustic_2d() -> None:
    """A real acoustic v+rho run's peak is within a quarter of the estimate.

    The regression this guards is the one this API exists for: before the
    stored-history term counted storage slots rather than models, the same
    measurement came out at 1.5-2x the estimate for acoustic with a density
    gradient. Kept small and short (< 1 s of propagation) so it can run in
    the ordinary GPU selection.
    """
    device = torch.device("cuda:0")
    shape = (60, 120)
    nt = 400
    dt = 0.0005
    dx = 5.0
    pml_width = 20
    v = (1500 * torch.ones(*shape, device=device)).requires_grad_(True)
    rho = (1000 * torch.ones(*shape, device=device)).requires_grad_(True)
    amplitudes = deepwave.wavelets.ricker(25.0, nt, dt, 0.05).reshape(1, 1, nt)
    est = estimate_memory(
        "acoustic",
        shape,
        nt=nt,
        n_shots=1,
        accuracy=4,
        pml_width=pml_width,
        models_requiring_grad=["v", "rho"],
        device=device,
    )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=amplitudes.to(device),
        source_locations_volume_rate=torch.tensor([[[30, 60]]], device=device),
        receiver_locations_p=torch.tensor([[[20, 60]]], device=device),
        accuracy=4,
        pml_width=pml_width,
        pml_freq=25.0,
    )
    (out[-1] ** 2).sum().backward()
    torch.cuda.synchronize(device)
    measured = torch.cuda.max_memory_allocated(device) - base
    # Measured on Box C (RTX A4000): 1.02x. The bound is deliberately loose
    # (the unmodelled fixed per-call overhead is real), but far below the
    # 1.5-2x this configuration produced before the slot-count fix.
    assert measured < 1.25 * est["total_bytes_lower_bound"], (
        f"measured {measured} bytes vs estimate {est['total_bytes_lower_bound']} bytes"
    )


# ---------------------------------------------------------------------------
# CUDA OOM interception (run_forward_engine's ``_forward_oom_message`` and its
# ``except torch.cuda.OutOfMemoryError`` wiring).
#
# A real device OOM is deliberately not exercised here: forcing an actual
# allocator failure on a shared box is fast but can fragment the caching
# allocator for later tests sharing the same GPU (this box is currently
# running a gradcheck tranche). ``torch.cuda.OutOfMemoryError`` can be raised
# in pure Python without a GPU, so the wiring test below monkeypatches an
# allocation deep inside ``run_forward_engine``'s try block to raise it,
# which exercises the real interception path (catch, augment, chain) on CPU.
# ---------------------------------------------------------------------------


def test_forward_oom_message_mentions_estimate_and_levers() -> None:
    """The message-formatting helper reports the estimate figure and levers."""
    from deepwave import _engine

    spec = deepwave.scalar.__globals__["SPEC"]
    message = _engine._forward_oom_message(
        spec,
        (140, 140, 140),
        200,
        4,
        torch.float32,
        torch.device("cpu"),
        4,
        [20, 20, 20, 20, 20, 20],
        1,
        "device",
        False,
    )
    assert "scalar" in message
    assert "nt=200" in message
    assert "n_shots=4" in message
    assert "torch.float32" in message
    assert "estimate_memory predicts at least" in message
    assert "GB" in message
    assert "n_shots" in message
    assert "batch" in message
    assert 'storage_mode="cpu" or "disk"' in message
    assert "storage_compression=True" in message
    assert "model_gradient_sampling_interval" in message
    assert "deepwave.checkpointing.run_segments" in message


def test_forward_oom_estimate_accounts_for_sampled_history() -> None:
    """Sampling four times less often removes three quarters of stored history."""
    from deepwave import _engine

    spec = deepwave.scalar.__globals__["SPEC"]
    estimates = []
    for interval in (1, 4):
        message = _engine._forward_oom_estimate_line(
            spec,
            (140, 140),
            20000,
            1,
            torch.float32,
            torch.device("cpu"),
            4,
            [20] * 4,
            interval,
            "device",
            False,
        )
        assert message is not None
        estimates.append(float(message.split("at least ")[1].split(" GB")[0]))
    # Scalar stores one float32 grid at each sampled step. The two reported
    # estimates are rounded independently to 0.01 GB; other buffers cancel.
    removed_gb = 140 * 140 * (20000 - 5000) * 4 / 1e9
    assert abs((estimates[0] - estimates[1]) - removed_gb) <= 0.01


def test_forward_oom_message_estimate_uses_the_real_models_requiring_grad() -> None:
    """Given the models, the quoted estimate is exact, not the worst case.

    Elastic in 3-D stores 3 wavefield histories for a ``lamb``-only gradient
    but up to 6 for the worst single model (``mu``), so passing the models
    must lower the figure the message quotes.
    """
    from importlib import import_module

    from deepwave import _engine

    spec = import_module("deepwave.elastic").SPEC
    # Internal models (lamb, mu, mu_zy, mu_zx, mu_yx, buoyancy_[zyx]); only
    # lamb requires a gradient.
    models = [torch.zeros(1, requires_grad=(i == 0)) for i in range(8)]
    args = (
        spec,
        (140, 140, 140),
        40,
        1,
        torch.float32,
        torch.device("cpu"),
        4,
        [20, 20, 20, 20, 20, 20],
        1,
        "device",
        False,
    )
    exact = _engine._forward_oom_message(*args, models)
    worst_case = _engine._forward_oom_message(*args)
    pattern = r"predicts at least ([0-9.]+) GB"
    exact_gb = float(re.search(pattern, exact).group(1))  # type: ignore[union-attr]
    worst_case_gb = float(re.search(pattern, worst_case).group(1))  # type: ignore[union-attr]
    # 0.28 GiB of forward wavefields plus 3 (not 6) stored histories.
    assert exact_gb == pytest.approx(1.61, abs=0.01)
    assert worst_case_gb > exact_gb


def test_forward_oom_message_omits_storage_compression_lever_when_already_on() -> None:
    """The lever list does not suggest storage_compression=True if it's already set.

    The estimate figure itself must still reflect the active compression
    (checked against a separately computed estimate, not duplicated here).
    """
    from deepwave import _engine

    spec = deepwave.scalar.__globals__["SPEC"]
    message = _engine._forward_oom_message(
        spec,
        (140, 140, 140),
        200,
        4,
        torch.float32,
        torch.device("cpu"),
        4,
        [20, 20, 20, 20, 20, 20],
        1,
        "device",
        True,
    )
    assert "storage_compression=True" not in message
    assert 'storage_mode="cpu" or "disk"' in message


def test_forward_oom_message_omits_estimate_when_unavailable() -> None:
    """An unpaddable (negative) shape drops the estimate line, not the rest."""
    from deepwave import _engine

    spec = deepwave.scalar.__globals__["SPEC"]
    # pml_width alone (40 per side) exceeds the padded model shape (10), so
    # the derived "unpadded" shape used for estimate_memory would be
    # negative; the helper must skip the estimate rather than raise.
    message = _engine._forward_oom_message(
        spec,
        (10, 10),
        5,
        1,
        torch.float32,
        torch.device("cpu"),
        4,
        [40, 40, 40, 40],
        1,
        "device",
        False,
    )
    assert "estimate_memory predicts" not in message
    assert "deepwave.checkpointing.run_segments" in message


def test_run_forward_engine_augments_oom_and_chains_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An OOM raised deep inside the forward engine is caught and augmented.

    It is also chained onto the original error via ``raise ... from e``.
    """
    from deepwave import _engine

    real_storage_manager = _engine.StorageManager

    def raise_oom(*args: object, **kwargs: object) -> None:
        raise torch.cuda.OutOfMemoryError("simulated allocator failure")

    # ``_engine.StorageManager`` (the name imported into _engine's namespace)
    # is only used inside run_forward_engine's try block (for the propagator's
    # *backward*-storage manager); scalar.py's own second, raw-forward
    # storage manager referenced via ``_storage.StorageManager`` is a
    # separate name and is unaffected, so this patch lands exactly at the
    # engine's chokepoint rather than upstream of it.
    monkeypatch.setattr(_engine, "StorageManager", raise_oom)

    v = 1500 * torch.ones(10, 10)
    with pytest.raises(torch.cuda.OutOfMemoryError) as excinfo:
        deepwave.scalar(
            v,
            4.0,
            0.001,
            source_amplitudes=torch.zeros(1, 1, 5),
            source_locations=torch.tensor([[[0, 0]]]),
            receiver_locations=torch.tensor([[[0, 5]]]),
            pml_freq=25.0,
        )

    message = str(excinfo.value)
    assert "scalar propagation ran out of GPU memory" in message
    assert "estimate_memory predicts at least" in message
    assert "storage_compression=True" in message
    original = excinfo.value.__cause__
    assert isinstance(original, torch.cuda.OutOfMemoryError)
    assert "simulated allocator failure" in str(original)

    # Sanity: the monkeypatch is what triggered this, not the real class.
    assert _engine.StorageManager is not real_storage_manager


def test_run_forward_engine_does_not_catch_non_oom_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-OOM error from the same call site is left completely alone."""
    from deepwave import _engine

    def raise_value_error(*args: object, **kwargs: object) -> None:
        raise ValueError("unrelated failure")

    monkeypatch.setattr(_engine, "StorageManager", raise_value_error)

    v = 1500 * torch.ones(10, 10)
    with pytest.raises(ValueError, match="unrelated failure"):
        deepwave.scalar(
            v,
            4.0,
            0.001,
            source_amplitudes=torch.zeros(1, 1, 5),
            source_locations=torch.tensor([[[0, 0]]]),
            receiver_locations=torch.tensor([[[0, 5]]]),
            pml_freq=25.0,
        )


# ---------------------------------------------------------------------------
# PropagatorOutput ergonomics (every
# propagator always returns a PropagatorOutput).
# ---------------------------------------------------------------------------


def _small_named_output() -> deepwave.PropagatorOutput:
    """A tiny scalar() PropagatorOutput, for the tests below."""
    v = 1500 * torch.ones(10, 10)
    return deepwave.scalar(
        v,
        4.0,
        0.001,
        source_amplitudes=torch.zeros(1, 1, 5),
        source_locations=torch.tensor([[[0, 0]]]),
        receiver_locations=torch.tensor([[[0, 5]]]),
        pml_freq=25.0,
    )


def test_propagator_output_pickle_roundtrip_preserves_values_and_names() -> None:
    """Pickling and unpickling a PropagatorOutput preserves values, _names, and info.

    ``PropagatorOutput.__getnewargs_ex__`` supplies both positional constructor
    arguments (names and values) plus ``info`` as a keyword argument (``info``
    is keyword-only on ``__new__``), so a ``tuple`` subclass round-trips
    through pickle (and, by extension, ``copy``/``deepcopy``) instead of
    losing its field names or its run metadata.
    """
    out = _small_named_output()
    assert out.info is not None  # sanity: info is populated by default
    restored = pickle.loads(pickle.dumps(out))  # noqa: S301 - trusted, in-process data
    assert isinstance(restored, deepwave.PropagatorOutput)
    assert restored._names == out._names
    assert len(restored) == len(out)
    for got, want in zip(restored, out):
        assert elements_equal(got, want)
    assert restored.info == out.info


def test_propagator_output_checkpoint_preserves_requires_grad() -> None:
    """``torch.utils.checkpoint`` does not detach a PropagatorOutput (issue #120).

    Mirrors ``test_scalar.py``'s ``test_checkpoint_gradient_named_return``:
    since ``return_dataclass`` was removed and every propagator call now
    returns a ``PropagatorOutput`` unconditionally, this is simply the default
    behaviour, but the regression it guards is the same one: ``tuple``
    subclassing is what lets ``torch.utils.checkpoint``'s wrapping
    ``autograd.Function`` propagate ``requires_grad`` to the output at all.
    """
    ny, nx = 16, 16
    v = 1500.0 * torch.ones(ny, nx)
    v[ny // 2 :] = 1600.0
    source_amplitudes = deepwave.wavelets.ricker(25.0, 8, 0.001, 1.5 / 25.0).reshape(
        1, 1, -1
    )

    def call(v_in: torch.Tensor) -> deepwave.PropagatorOutput:
        return deepwave.scalar(
            v_in,
            5.0,
            0.001,
            source_amplitudes=source_amplitudes,
            source_locations=torch.tensor([[[8, 8]]]),
            receiver_locations=torch.tensor([[[9, 9]]]),
            pml_freq=25.0,
            max_vel=1600.0,
        )

    run_checkpoint_equivalence(v, call, use_reentrant=True)
    run_checkpoint_equivalence(v, call, use_reentrant=False)


def test_propagator_output_indexing_and_attribute_access_agree() -> None:
    """``out[-1]``, ``*ws, r = out``, and ``out.receiver_amplitudes`` all agree."""
    out = _small_named_output()
    *wavefields, receivers = out
    assert torch.equal(out[-1], receivers)
    assert torch.equal(out.receiver_amplitudes, receivers)
    assert torch.equal(out.receiver_amplitudes, out[-1])
    assert len(wavefields) == len(out) - 1


# ---------------------------------------------------------------------------
# torch.nn.DataParallel cannot gather a full PropagatorOutput.
# ``DataParallel.gather`` reconstructs a module's
# return value via ``type(out)(map(gather_map, zip(*outputs, strict=True)))``
# -- the same convention it uses for a real ``namedtuple`` -- but
# ``PropagatorOutput``'s field names are per-*instance* (they depend on
# ``ndim``/``n_mechanisms``), not fixed by the class the way a namedtuple's
# are, so they cannot be recovered from ``type(out)`` alone at reconstruction
# time; see ``PropagatorOutput``'s docstring in ``src/deepwave/_engine.py``
# for the full investigation of, and reason for rejecting, an
# iterable-accepting ``__new__``. This is therefore documented (
# ``docs/example_distributed.rst``) rather than fixed: these tests lock in
# the current (crashing) behaviour so the documentation is caught out of date
# if it is ever fixed, and confirm the documented single-tensor workaround
# actually works, including a real 2-GPU smoke test on this box's two GPUs.
# ---------------------------------------------------------------------------


def test_propagator_output_not_reconstructible_from_single_iterable() -> None:
    """Reproduces DataParallel.gather's exact ``type(out)(iterable)`` call.

    No GPU needed: this exercises the same Python-level construction
    ``torch.nn.parallel.scatter_gather.gather``'s ``gather_map`` performs for
    a non-namedtuple ``tuple`` subclass (``_is_namedtuple(out)`` is ``False``
    for ``PropagatorOutput`` -- it has ``_asdict`` but not the ``_fields``
    class attribute a real namedtuple has), independent of CUDA or an actual
    multi-GPU replica/gather round trip.
    """
    out = _small_named_output()
    assert not hasattr(out, "_fields")  # confirms _is_namedtuple(out) is False
    # ``gather_map`` writes ``zip(*outputs, strict=True)``; ``strict`` is
    # Python 3.10+ and Deepwave supports 3.9, so a plain ``zip`` is used
    # here -- the two stand-in replicas have equal length, which is the only
    # thing ``strict`` would add.
    with pytest.raises(TypeError, match="cannot be reconstructed from a single"):
        type(out)(v for v in zip(*[out, out]))


def test_propagator_output_single_tensor_survives_gather_style_call() -> None:
    """The documented workaround (return one field) sidesteps the crash.

    Simulates ``DataParallel.gather`` combining two single-GPU replicas'
    ``receiver_amplitudes`` (a plain ``torch.Tensor``, not the full
    ``PropagatorOutput``) with ``torch.cat`` (``gather``'s ``Gather.apply``
    reduces to a concatenation along the batch dimension for CPU tensors).
    """
    out = _small_named_output()
    receiver_amplitudes = out.receiver_amplitudes
    gathered = torch.cat([receiver_amplitudes, receiver_amplitudes], dim=1)
    assert gathered.shape[1] == 2 * receiver_amplitudes.shape[1]
    assert torch.equal(gathered[:, : receiver_amplitudes.shape[1]], receiver_amplitudes)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2+ GPUs")
def test_propagator_output_dataparallel_two_gpu_smoke() -> None:
    """Real 2-GPU ``DataParallel`` smoke test.

    Returning the full ``PropagatorOutput`` from ``forward()`` raises the
    documented ``TypeError`` once ``DataParallel`` actually gathers across 2
    devices (with a single device it takes a shortcut and never calls
    ``gather`` at all, so this needs real hardware, not a mock, to be
    meaningful). Returning just ``out.receiver_amplitudes`` -- the documented
    workaround -- works and produces the expected gathered shape.
    """

    class Prop(torch.nn.Module):
        def __init__(self, v: torch.Tensor, dx: float, full_output: bool) -> None:
            super().__init__()
            self.v = torch.nn.Parameter(v)
            self.dx = dx
            self.full_output = full_output

        def forward(
            self,
            dt: float,
            source_amplitudes: torch.Tensor,
            source_locations: torch.Tensor,
            receiver_locations: torch.Tensor,
        ) -> Any:
            out = deepwave.scalar(
                self.v,
                self.dx,
                dt,
                source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                pml_freq=15.0,
            )
            return out if self.full_output else out.receiver_amplitudes

    n_shots = 4
    n = 16
    dt = 0.001
    nt = 8
    v = 1500.0 * torch.ones(n, n)
    source_amplitudes = (
        deepwave.wavelets.ricker(15.0, nt, dt, 0.05)
        .reshape(1, 1, -1)
        .repeat(n_shots, 1, 1)
        .to("cuda:0")
    )
    locations = torch.full((n_shots, 1, 2), n // 2, dtype=torch.long).to("cuda:0")

    def run(full_output: bool) -> Any:
        model = Prop(v.clone(), 4.0, full_output).to("cuda:0")
        parallel = torch.nn.DataParallel(model, device_ids=[0, 1])
        return parallel(dt, source_amplitudes, locations, locations)

    with pytest.raises(TypeError, match="cannot be reconstructed from a single"):
        run(full_output=True)

    receiver_amplitudes = run(full_output=False)
    assert receiver_amplitudes.shape[:2] == (n_shots, 1)


# ---------------------------------------------------------------------------
# out.info: RunInfo run metadata.
# ---------------------------------------------------------------------------


def test_run_info_step_ratio_matches_cfl_condition_n_when_one() -> None:
    """out.info.step_ratio matches cfl_condition_n's value when step_ratio == 1."""
    out = _small_named_output()  # v=1500, grid_spacing=4.0, dt=0.001
    _, expected_step_ratio = deepwave.discretisation.cfl_condition_n(
        [4.0, 4.0], 0.001, 1500.0
    )
    assert expected_step_ratio == 1
    assert out.info is not None
    assert out.info.step_ratio == expected_step_ratio


def test_run_info_step_ratio_matches_cfl_condition_n_when_three() -> None:
    """out.info.step_ratio matches cfl_condition_n for an engineered case.

    A high max_vel forces 3 internal steps per user time sample.
    """
    grid_spacing = [4.0, 4.0]
    dt = 0.001
    max_vel = 4000.0
    v = max_vel * torch.ones(10, 10)
    out = deepwave.scalar(
        v,
        grid_spacing,
        dt,
        source_amplitudes=torch.zeros(1, 1, 5),
        source_locations=torch.tensor([[[0, 0]]]),
        receiver_locations=torch.tensor([[[0, 5]]]),
        pml_freq=25.0,
    )
    _, expected_step_ratio = deepwave.discretisation.cfl_condition_n(
        grid_spacing, dt, max_vel
    )
    assert expected_step_ratio == 3
    assert out.info is not None
    assert out.info.step_ratio == expected_step_ratio
    # nt is the number of *user-level* time samples (5), not the internal
    # step count (5 * step_ratio) that setup_propagator works with internally.
    assert out.info.nt == 5


def test_run_info_internal_dt_is_dt_over_step_ratio_exactly() -> None:
    """internal_dt == dt / step_ratio exactly (not merely approximately)."""
    dt = 0.001
    out = _small_named_output()
    assert out.info is not None
    assert out.info.internal_dt == dt / out.info.step_ratio


def test_run_info_dt_internal_is_not_an_attribute() -> None:
    """``dt_internal`` never shipped, so it raises, naming ``internal_dt``.

    The message is the point: the two spellings are the same words the other
    way round, so a bare ``AttributeError`` would leave a reader guessing.
    """
    out = _small_named_output()
    with pytest.raises(AttributeError, match="internal_dt"):
        _ = out.info.dt_internal


def test_run_info_fields_for_default_survey() -> None:
    """pml_width/origin/extent/courant_number are the resolved run values."""
    out = _small_named_output()  # 10x10 model, default pml_width=20, no survey_pad
    assert out.info is not None
    assert out.info.pml_width == (20, 20, 20, 20)
    assert out.info.origin == (0, 0)
    assert out.info.extent == (10, 10)
    assert out.info.courant_number == 0.6  # the conservative default
    assert out.info.nt == 5


def test_run_info_does_not_affect_len_or_iteration() -> None:
    """``info`` is metadata, not a tuple element.

    len()/iteration/unpacking of the output tuple are unaffected by its
    presence.
    """
    out = _small_named_output()
    assert out.info is not None
    assert len(out) == len(out._names)
    values_from_iteration = list(out)
    assert len(values_from_iteration) == len(out)
    *wavefields, receivers = out
    assert len(wavefields) == len(out) - 1
    assert torch.equal(receivers, out.receiver_amplitudes)


# ---------------------------------------------------------------------------
# README Quick Example: must run without triggering any warning.
# ---------------------------------------------------------------------------


def _extract_readme_quick_example() -> str:
    """Extracts the fenced Python code block under README.md's Quick Example.

    Reading the code from README.md at test time (rather than duplicating
    it) means this test cannot silently drift out of sync with the actual
    example a user copies and pastes.
    """
    readme_path = Path(__file__).resolve().parent.parent / "README.md"
    text = readme_path.read_text()
    match = re.search(r"## Quick Example\n.*?```python\n(.*?)```", text, re.DOTALL)
    assert match is not None, "README.md's Quick Example code block was not found."
    code = match.group(1)
    # The plotting section is unrelated to the warnings under test here and
    # would need a display backend; everything up to and including the
    # gradient computation is kept.
    assert "_, ax = plt.subplots" in code, (
        "README.md's Quick Example no longer has the expected plotting "
        "section; update this test's extraction."
    )
    return code.split("\n_, ax = plt.subplots")[0]


def test_readme_quick_example_is_warning_free() -> None:
    """The README's Quick Example triggers zero warnings when run.

    The example is synthetic and self-contained (a `torch.ones`-built
    velocity model, no external data files), so it can be executed
    directly. `deepwave` is already imported at module scope above, so the
    example's own `import deepwave` is a cheap no-op here, not a fresh
    module import.
    """
    code = _extract_readme_quick_example()
    namespace: dict[str, Any] = {}
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        exec(compile(code, "readme_quick_example", "exec"), namespace)  # noqa: S102
    assert record == []
