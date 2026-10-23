"""Tests for ``deepwave.plan`` and ``checkpoint_segments=``.

The load-bearing test here is ``test_plan_info_equals_run_info``: it is the
anti-drift lock for the whole pre-flight feature. ``plan()`` is only worth
having if its answer cannot differ from the run's, and the way that is enforced
is by comparing the two ``RunInfo`` objects field for field, for every
propagator and every dimensionality. If a propagator gains a resolution step
that ``plan()`` does not see, this test fails, and that is correct.
"""

import dataclasses
import warnings

import pytest
import torch

import deepwave
from deepwave._exceptions import (
    DeepwaveTypeError,
    DeepwaveValueError,
)


def _wavelet(nt=60, dt=0.004, dtype=torch.float32):
    return deepwave.wavelets.ricker(15, nt, dt, 0.06).reshape(1, 1, -1).to(dtype)


def _case(name, ndim, dtype=torch.float32, n=None):
    """Positional arguments and keywords for one propagator at one ndim.

    Returns None when the combination is not supported, so a parametrised
    test can skip it.
    """
    n = n if n is not None else (24 if ndim == 3 else 40)
    shape = (n,) * ndim
    ones = torch.ones(*shape, dtype=dtype)
    v = ones * 1500.0
    rho = ones * 2200.0
    q = ones * 50.0
    lamb, mu, buoyancy = ones * 2.25e9, ones * 1e9, ones / 2200.0
    stiffness = ones * 1e10
    sl = torch.tensor([[[5] * ndim]])
    rl = torch.tensor([[[10] * ndim]])
    src = _wavelet(dtype=dtype)
    common = {"grid_spacing": 4.0, "dt": 0.004, "pml_width": 5, "max_vel": 3000.0}
    flat = {
        "source_amplitudes": src,
        "source_locations": sl,
        "receiver_locations": rl,
    }
    # The scalar source channel is named for the quantity it injects: a
    # volume rate on the acoustic family, a pressure rate on the elastic
    # family and acoustic_vti. The receiver channel is ``p`` on all of them.
    channel_volume_rate = {
        "source_amplitudes_volume_rate": src,
        "source_locations_volume_rate": sl,
        "receiver_locations_p": rl,
    }
    channel_pressure_rate = {
        "source_amplitudes_pressure_rate": src,
        "source_locations_pressure_rate": sl,
        "receiver_locations_p": rl,
    }
    channel_y = {
        "source_amplitudes_y": src,
        "source_locations_y": sl,
        "receiver_locations_y": rl,
    }
    born_y = {
        "source_amplitudes_y": src,
        "source_locations_y": sl,
        "bg_receiver_locations_y": rl,
    }
    born_volume_rate = {
        "source_amplitudes_volume_rate": src,
        "source_locations_volume_rate": sl,
        "bg_receiver_locations_p": rl,
    }
    vti_p = {
        "source_amplitudes_pressure_rate": src,
        "source_locations_pressure_rate": sl,
        "receiver_locations_y": rl,
    }
    cases = {
        "scalar": ([v], flat),
        "scalar_born": ([v], {"dv": torch.zeros_like(v), **flat}),
        "acoustic": ([v, rho], channel_volume_rate),
        "acoustic_born": ([v, rho], {"dv": torch.zeros_like(v), **born_volume_rate}),
        "acoustic_vti": (
            [stiffness, stiffness * 0.5, stiffness, buoyancy],
            vti_p,
        ),
        "elastic": ([lamb, mu, buoyancy], channel_y),
        "elastic_born": (
            [lamb, mu, buoyancy],
            {"dlamb": torch.zeros_like(v), **born_y},
        ),
        "elastic_vti": (
            [stiffness, stiffness * 0.3, stiffness, stiffness * 0.2, buoyancy],
            ({**channel_y, "c66": stiffness * 0.25} if ndim == 3 else dict(channel_y)),
        ),
        "viscoacoustic": ([v, rho, q], channel_pressure_rate),
    }
    args, kwargs = cases[name]
    if name == "viscoacoustic":
        # The frequency Q is specified at; required on these two, refused on
        # the other nine.
        kwargs = {**kwargs, "q_freq": 25.0}
    return args, {**common, **kwargs}


PROPAGATORS = [
    "scalar",
    "scalar_born",
    "acoustic",
    "acoustic_born",
    "acoustic_vti",
    "elastic",
    "elastic_born",
    "elastic_vti",
    "viscoacoustic",
]


def _call(name, args, kwargs, **extra):
    fn = getattr(deepwave, name)
    grid_spacing = kwargs["grid_spacing"]
    dt = kwargs["dt"]
    rest = {k: v for k, v in kwargs.items() if k not in ("grid_spacing", "dt")}
    return fn(*args, grid_spacing, dt, **rest, **extra)


def _plan(name, args, kwargs, **extra):
    grid_spacing = kwargs["grid_spacing"]
    dt = kwargs["dt"]
    rest = {k: v for k, v in kwargs.items() if k not in ("grid_spacing", "dt")}
    return deepwave.plan(
        getattr(deepwave, name), *args, grid_spacing, dt, **rest, **extra
    )


# ---------------------------------------------------------------------------
# plan(): the anti-drift lock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPAGATORS)
@pytest.mark.parametrize("ndim", [2, 3])
def test_plan_info_equals_run_info(name, ndim):
    """plan() and the run must resolve to the same RunInfo, field for field.

    This is the whole promise of the pre-flight: it is the run's own
    resolution, stopped early, not a second implementation of it.
    """
    case = _case(name, ndim)
    if case is None:
        pytest.skip(f"{name} does not support ndim={ndim}")
    args, kwargs = case
    planned = _plan(name, args, kwargs)
    ran = _call(name, args, kwargs)
    assert planned.info == ran.info


def test_plan_accepts_a_name_as_well_as_the_function():
    """The propagator's name and the function itself plan identically."""
    args, kwargs = _case("scalar", 2)
    by_function = _plan("scalar", args, kwargs)
    grid_spacing = kwargs["grid_spacing"]
    dt = kwargs["dt"]
    rest = {k: v for k, v in kwargs.items() if k not in ("grid_spacing", "dt")}
    by_name = deepwave.plan("scalar", *args, grid_spacing, dt, **rest)
    assert by_name.info == by_function.info
    assert by_name.memory == by_function.memory


def test_plan_accepts_a_module_instance_and_refuses_the_class():
    """A ``Module`` is planned as built; the class is not planned at all.

    The class form used to be accepted and quietly planned the *functional*.
    That is not the same call: the functional honours ``v.requires_grad``,
    while the constructor stores the model without a gradient unless its own
    ``requires_grad=`` arguments say otherwise, so the class form reported the
    gradient machinery of a run the ``Module`` would not perform -- an
    over-estimate. The old test compared only ``info``, which is equal either way,
    so it could not see this.
    """
    args, kwargs = _case("scalar", 2)
    rest = {k: v for k, v in kwargs.items() if k not in ("grid_spacing", "dt")}
    with pytest.raises(deepwave.DeepwaveTypeError, match="class itself") as caught:
        deepwave.plan(
            deepwave.Scalar, *args, kwargs["grid_spacing"], kwargs["dt"], **rest
        )
    # The message must point at all three forms that do work.
    message = str(caught.value)
    assert "deepwave.scalar" in message
    assert "'scalar'" in message
    assert "Scalar(...)" in message

    args[0].requires_grad_(True)
    module = deepwave.Scalar(args[0], kwargs["grid_spacing"])
    from_instance = deepwave.plan(module, kwargs["dt"], **rest)
    from_functional = _plan("scalar", args, kwargs)
    # Same resolution, different cost: that difference is why the class form
    # could not stand in for either of them.
    assert from_instance.info.model_gradient_stride is None
    assert from_functional.info.model_gradient_stride is not None
    assert (
        dataclasses.replace(
            from_instance.info,
            model_gradient_stride=from_functional.info.model_gradient_stride,
        )
        == from_functional.info
    )
    assert from_instance.memory["backward_storage_bytes"] == 0
    assert from_functional.memory["backward_storage_bytes"] > 0
    # ...and the constructor can be told to keep the gradient, which brings
    # the two back into agreement.
    with_grad = deepwave.Scalar(args[0], kwargs["grid_spacing"], v_requires_grad=True)
    assert (
        deepwave.plan(with_grad, kwargs["dt"], **rest).memory == from_functional.memory
    )


def test_plan_captures_warnings_as_text_rather_than_raising_them():
    """The warnings a call would emit are reported, not emitted."""
    # "grid cells per wavelength" is emitted at most once per process, so an
    # earlier test in this process may already have spent its budget. Clearing
    # the cause makes this test independent of the order it runs in.
    deepwave._common._warned_once_causes.discard("points_per_wavelength")
    slow = torch.ones(40, 40) * 100.0
    _, kwargs = _case("scalar", 2)
    kwargs = dict(kwargs, grid_spacing=40.0, max_vel=None)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        planned = _plan("scalar", [slow], kwargs)
    assert record == []
    assert any("grid cells per wavelength" in text for text in planned.warnings)


def test_plan_does_not_change_process_warning_filters():
    """A plan collects its own messages while unrelated warnings keep their policy."""
    deepwave._common._warned_once_causes.discard("points_per_wavelength")
    slow = torch.ones(40, 40) * 100.0
    _, kwargs = _case("scalar", 2)
    kwargs = dict(kwargs, grid_spacing=40.0, max_vel=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        filters = list(warnings.filters)
        planned = _plan("scalar", [slow], kwargs)
        assert warnings.filters == filters
        with pytest.raises(UserWarning, match="unrelated"):
            warnings.warn("unrelated", UserWarning, stacklevel=1)
    assert any("grid cells per wavelength" in message for message in planned.warnings)


def test_plan_cleans_up_after_a_call_that_raises():
    """A failed plan must leave no probe behind for the next one."""
    from deepwave._plan import _PROBE

    args, kwargs = _case("scalar", 2)
    with pytest.raises(DeepwaveValueError):
        _plan("scalar", args, kwargs, storage_mode="bogus")
    assert _PROBE.get() is None
    # And a later plan still works.
    assert _plan("scalar", args, kwargs).info.nt > 0


def test_plan_allocates_no_wavefield_history(monkeypatch):
    """plan() must never reach the allocation that dominates a real run."""
    calls = []
    original = deepwave._storage.StorageManager.allocate

    def spy(self, requires_grad):
        calls.append(requires_grad)
        return original(self, requires_grad)

    monkeypatch.setattr(deepwave._storage.StorageManager, "allocate", spy)
    args, kwargs = _case("scalar", 2)
    args[0].requires_grad_(True)
    _plan("scalar", args, kwargs)
    assert calls == []
    # The same call, actually run, does allocate -- so the assertion above is
    # not vacuously true of this configuration.
    _call("scalar", args, kwargs)
    assert calls


def test_plan_reports_resampling():
    """``resampling`` is true exactly when the call would sub-step."""
    args, kwargs = _case("scalar", 2)
    fine = dict(kwargs, dt=0.0005)
    assert _plan("scalar", args, fine).resampling is False
    assert _plan("scalar", args, fine).info.step_ratio == 1
    coarse = dict(kwargs, dt=0.008)
    planned = _plan("scalar", args, coarse)
    assert planned.resampling is (planned.info.step_ratio > 1)
    assert planned.resampling is True


def test_plan_summary_and_repr_report_the_resolved_call():
    """``summary()`` is a readable block and ``repr`` is one line."""
    args, kwargs = _case("scalar", 2)
    planned = _plan("scalar", args, kwargs)
    text = planned.summary()
    assert "\n" in text
    for fragment in ("padded to", "backend", "storage", "memory at least"):
        assert fragment in text
    line = repr(planned)
    assert "\n" not in line
    assert line.startswith("Plan(")
    assert "x".join(str(s) for s in planned.memory["padded_model_shape"]) in line


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"storage_mode": "bogus"}, "but got bogus"),
        ({"courant_number": 99.0}, "exceeds the critical Courant number"),
        ({"model_gradient_sampling_interval": 7}, "must divide"),
        ({"storage_mode": "none"}, "retains no forward"),
        # The backend-dependent rejections. Without varying ``backend=`` this
        # test could not see them at all: the Python backends refuse every
        # storage mode but "device", and all compression, and they do it from
        # below the point a plan stops at, so a plan that did not run the
        # same check would accept what the call refuses.
        (
            {"backend": "eager", "storage_mode": "disk"},
            'storage_mode="disk" is not supported',
        ),
        (
            {"backend": "eager", "storage_mode": "cpu"},
            'storage_mode="cpu" is not supported',
        ),
        (
            {"backend": "eager", "storage_compression": True},
            "Storage compression is not supported",
        ),
        # Both the backend rejection and the model-gradient rejection apply
        # here; the run hits the backend one first, so the plan must too.
        (
            {"backend": "eager", "storage_mode": "none"},
            'storage_mode="none" is not supported',
        ),
        (
            {"backend": "torch_compile", "storage_mode": "disk"},
            'storage_mode="disk" is not supported',
        ),
        (
            {"backend": "torch_compile", "storage_compression": True},
            "Storage compression is not supported",
        ),
    ],
)
def test_plan_raises_the_same_error_the_call_would(extra, match):
    """Every argument-validity error must surface from plan(), identically."""
    args, kwargs = _case("scalar", 2)
    args[0].requires_grad_(True)
    with pytest.raises(Exception, match=match) as planned_error:
        _plan("scalar", args, kwargs, **extra)
    with pytest.raises(Exception, match=match) as run_error:
        _call("scalar", args, kwargs, **extra)
    assert type(planned_error.value) is type(run_error.value)
    assert str(planned_error.value) == str(run_error.value)


@pytest.mark.parametrize(
    "propagator",
    [
        "scalar",
        "scalar_born",
        "acoustic",
        "acoustic_born",
        "acoustic_vti",
        "elastic",
        "elastic_born",
        "elastic_vti",
        "viscoacoustic",
    ],
)
@pytest.mark.parametrize(
    "extra",
    [
        {"backend": "eager", "storage_mode": "disk"},
        {"backend": "eager", "storage_mode": "cpu"},
        {"backend": "eager", "storage_compression": True},
        {"backend": "eager", "storage_mode": "none"},
    ],
)
def test_plan_matches_the_backend_rejection_on_every_propagator(propagator, extra):
    """All nine retained propagators must match the run's rejection."""
    args, kwargs = _case(propagator, 2)
    args[0].requires_grad_(True)
    with pytest.raises(RuntimeError, match="not supported") as planned_error:
        _plan(propagator, args, kwargs, **extra)
    with pytest.raises(RuntimeError, match="not supported") as run_error:
        _call(propagator, args, kwargs, **extra)
    assert type(planned_error.value) is type(run_error.value)
    assert str(planned_error.value) == str(run_error.value)


def test_plan_does_not_consume_once_per_process_warnings(tmp_path):
    """A plan must not spend the warning budget the real call needs."""
    from deepwave._common import _warned_once_causes

    _warned_once_causes.discard("storage_default_disk_path")
    args, kwargs = _case("scalar", 2)
    monkey = dict(kwargs, storage_mode="disk")
    _plan("scalar", args, monkey)
    assert "storage_default_disk_path" not in _warned_once_causes
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _call("scalar", args, monkey)
    assert any("default directory" in str(w.message) for w in record)


def test_plan_memory_has_every_documented_key():
    """The plan's estimate has the same keys as the parametric one, and they sum."""
    args, kwargs = _case("scalar", 2)
    memory = _plan("scalar", args, kwargs).memory
    parametric = deepwave.estimate_memory(
        "scalar", model_shape=(40, 40), nt=60, n_shots=1, pml_width=5
    )
    assert set(memory) == set(parametric)
    assert memory["total_bytes_lower_bound"] == (
        memory["forward_wavefields_bytes"]
        + memory["backward_storage_bytes"]
        + memory["gradient_scratch_bytes"]
        + memory["auxiliary_state_bytes"]
    )


def test_plan_memory_uses_the_runs_own_padded_shape():
    """The run-shaped estimate is exact where the parametric one approximates.

    ``acoustic`` is staggered, so it really pads one cell less on the far side
    of each axis; the parametric estimate documents that it does not model
    that, and the plan, reading the padded model itself, does.
    """
    args, kwargs = _case("acoustic", 2)
    planned = _plan("acoustic", args, kwargs)
    parametric = deepwave.estimate_memory(
        "acoustic", model_shape=(40, 40), nt=60, n_shots=1, pml_width=5
    )
    assert planned.memory["padded_model_shape"] != parametric["padded_model_shape"]
    assert all(
        a <= b
        for a, b in zip(
            planned.memory["padded_model_shape"], parametric["padded_model_shape"]
        )
    )


# ---------------------------------------------------------------------------
# RunInfo's storage fields
# ---------------------------------------------------------------------------


def test_run_info_records_the_resolved_storage_policy(tmp_path):
    """``RunInfo.storage`` reports what was used, not what was asked for."""
    args, kwargs = _case("scalar", 2)
    out = _call("scalar", args, kwargs, storage_mode="cpu")
    assert out.info.storage is not None
    # On a CPU device "cpu" and "device" are the same memory, and the engine
    # downgrades; RunInfo must report what was used, not what was asked for.
    assert out.info.storage.mode == "device"
    assert out.info.storage.path is None
    out = _call("scalar", args, kwargs, storage_mode="disk", storage_path=str(tmp_path))
    assert out.info.storage is not None
    assert out.info.storage.mode == "disk"
    assert out.info.storage.path == str(tmp_path)


def test_run_info_lossy_compression_used_is_resolved_not_requested():
    """Compression that no mode applied is reported as unused."""
    args, kwargs = _case("scalar", 2)
    assert _call("scalar", args, kwargs).info.lossy_compression_used is False
    used = _call("scalar", args, kwargs, storage_compression=True)
    assert used.info.lossy_compression_used is True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unused = _call(
            "scalar",
            args,
            kwargs,
            storage=deepwave.Storage(mode="none", lossy_compression=True),
        )
    assert unused.info.lossy_compression_used is False


# ---------------------------------------------------------------------------
# estimate_memory's segment term
# ---------------------------------------------------------------------------


def test_estimate_memory_n_segments_reduces_the_history():
    """Segmenting divides the history term and adds a boundary term."""
    base = {
        "model_shape": (200, 200),
        "nt": 1000,
        "n_shots": 4,
        "pml_width": 20,
        "propagator": "scalar",
    }
    one = deepwave.estimate_memory(**base)
    assert deepwave.estimate_memory(**base, n_segments=1) == one
    split = deepwave.estimate_memory(**base, n_segments=5)
    assert split["backward_storage_bytes"] == one["backward_storage_bytes"] // 5
    assert split["auxiliary_state_bytes"] > 0
    assert one["auxiliary_state_bytes"] == 0
    assert split["total_bytes_lower_bound"] < one["total_bytes_lower_bound"]


def test_estimate_memory_boundary_term_is_zero_without_a_gradient():
    """No gradient means no checkpointing, so one state is held, not ``k``.

    A forward-only segmented run threads a single state forward and drops the
    previous one, so the boundary term must not scale with the segment count.
    It used to, which made the "lower bound" more than fourteen times the
    measured peak at sixty segments.
    """
    base = {
        "model_shape": (200, 200),
        "nt": 1000,
        "n_shots": 4,
        "pml_width": 20,
        "propagator": "scalar",
        "models_requiring_grad": [],
    }
    few = deepwave.estimate_memory(**base, n_segments=5)
    many = deepwave.estimate_memory(**base, n_segments=60)
    assert few["backward_storage_bytes"] == 0
    assert few["auxiliary_state_bytes"] > 0
    assert many["auxiliary_state_bytes"] == few["auxiliary_state_bytes"]
    assert many["total_bytes_lower_bound"] == few["total_bytes_lower_bound"]
    # With a gradient the states really are held one per segment, so there
    # the term does scale -- the two cases must not be conflated.
    with_grad = {**base, "models_requiring_grad": ["v"]}
    grad_few = deepwave.estimate_memory(**with_grad, n_segments=5)
    grad_many = deepwave.estimate_memory(**with_grad, n_segments=60)
    assert grad_many["auxiliary_state_bytes"] == 12 * grad_few["auxiliary_state_bytes"]


def test_estimate_memory_boundary_term_stays_below_the_real_state():
    """The boundary term is a floor on the states, not a full-grid ceiling.

    It is what stopped the total being a lower bound at larger segment counts:
    a term that over-estimates its own contribution, multiplied by ``k``,
    eventually dominates. Each state is priced without the finite-difference
    halo and with the PML memory variables as boundary slabs, so it must come
    out below the naive count of whole padded grids.
    """
    base = {
        "model_shape": (100, 100, 100),
        "nt": 600,
        "n_shots": 2,
        "pml_width": 20,
        "accuracy": 4,
        "propagator": "scalar",
        "models_requiring_grad": ["v"],
    }
    k = 30
    est = deepwave.estimate_memory(**base, n_segments=k)
    padded = est["padded_model_elements"]
    n_fields = est["forward_wavefields_bytes"] // (2 * padded * 4)
    naive = k * n_fields * 2 * padded * 4
    assert 0 < est["auxiliary_state_bytes"] < naive
    # Sized field by field: two whole physical wavefields plus six slabs at
    # 48/140 of the pml view, against eight whole padded grids.
    assert est["auxiliary_state_bytes"] < 0.5 * naive


@pytest.mark.parametrize(
    ("bad", "exc", "match"),
    [
        (True, deepwave.DeepwaveTypeError, "not a bool"),
        (1.5, deepwave.DeepwaveTypeError, "must be an int"),
        ("2", deepwave.DeepwaveTypeError, "must be an int"),
        (0, deepwave.DeepwaveValueError, "positive int"),
        (-1, deepwave.DeepwaveValueError, "positive int"),
        (101, deepwave.DeepwaveValueError, "cannot exceed the number of time steps"),
    ],
)
def test_estimate_memory_validates_n_segments(bad, exc, match):
    """The same standard the propagator's own segment count is held to.

    ``1.5`` in particular used to be accepted and returned float "bytes" from
    a TypedDict whose fields are declared ``int``; ``True`` was accepted as
    one segment; and a count above ``nt`` was accepted and over-estimated by
    several times, though the propagator rejects it.
    """
    with pytest.raises(exc, match=match):
        deepwave.estimate_memory(
            "scalar", (50, 50, 50), 100, 1, models_requiring_grad=["v"], n_segments=bad
        )


def test_plan_costs_checkpoint_segments():
    """A plan costs the segmentation while still resolving the whole run."""
    args, kwargs = _case("scalar", 2)
    args[0].requires_grad_(True)
    whole = _plan("scalar", args, kwargs)
    split = _plan("scalar", args, kwargs, checkpoint_segments=4)
    assert split.checkpoint_segments == 4
    assert whole.checkpoint_segments is None
    # The resolution describes the whole run either way; only the cost moves.
    assert split.info == whole.info
    assert (
        split.memory["backward_storage_bytes"]
        == whole.memory["backward_storage_bytes"] // 4
    )


# ---------------------------------------------------------------------------
# checkpoint_segments= on the propagators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_segments", [1, 2, 5])
def test_checkpoint_segments_matches_run_segments(n_segments):
    """The keyword is exactly the run_segments call it saves you writing."""
    args, kwargs = _case("scalar", 2, dtype=torch.float64)
    v = args[0].clone().requires_grad_(True)

    direct = deepwave.checkpointing.run_segments(
        deepwave.scalar,
        [v],
        kwargs["grid_spacing"],
        kwargs["dt"],
        {"source_amplitudes": kwargs["source_amplitudes"]},
        n_segments,
        checkpoint=True,
        source_locations=kwargs["source_locations"],
        receiver_locations=kwargs["receiver_locations"],
        pml_width=kwargs["pml_width"],
        max_vel=kwargs["max_vel"],
    )
    direct.receiver_amplitudes.pow(2).sum().backward()
    direct_grad = v.grad.clone()
    v.grad = None

    through = _call("scalar", [v], kwargs, checkpoint_segments=n_segments)
    through.receiver_amplitudes.pow(2).sum().backward()

    assert torch.equal(direct.receiver_amplitudes, through.receiver_amplitudes)
    assert torch.equal(direct_grad, v.grad)
    assert direct.names == through.names


def test_checkpoint_segments_output_is_indistinguishable():
    """A segmented call returns the same shape of answer as a direct one."""
    args, kwargs = _case("scalar", 2, dtype=torch.float64)
    whole = _call("scalar", args, kwargs)
    split = _call("scalar", args, kwargs, checkpoint_segments=3)
    assert split.names == whole.names
    assert split.info is not None
    assert split.info.nt == whole.info.nt
    assert torch.equal(split.receiver_amplitudes, whole.receiver_amplitudes)
    assert split.auxiliary_state is not None


def test_checkpoint_segments_gradient_matches_an_unsegmented_run():
    """Segmenting costs summation order, not accuracy."""
    args, kwargs = _case("scalar", 2, dtype=torch.float64)
    v = args[0].clone().requires_grad_(True)
    _call("scalar", [v], kwargs).receiver_amplitudes.pow(2).sum().backward()
    whole = v.grad.clone()
    v.grad = None
    _call("scalar", [v], kwargs, checkpoint_segments=5).receiver_amplitudes.pow(
        2
    ).sum().backward()
    # Segmenting reorders the gradient's time integral, so this is a
    # summation-order difference and nothing more.
    assert (v.grad - whole).norm() / whole.norm() < 1e-12


@pytest.mark.parametrize("extra", [{"forward_callback": lambda state: None}])
def test_plan_raises_a_segmented_calls_argument_errors_too(extra):
    """A pre-flight must not accept a combination the run refuses."""
    args, kwargs = _case("scalar", 2)
    with pytest.raises(DeepwaveValueError) as planned_error:
        _plan("scalar", args, kwargs, checkpoint_segments=2, **extra)
    with pytest.raises(DeepwaveValueError) as run_error:
        _call("scalar", args, kwargs, checkpoint_segments=2, **extra)
    assert str(planned_error.value) == str(run_error.value)


@pytest.mark.parametrize(
    ("segments", "change"),
    [
        (100, {}),
        (4, {"model_gradient_sampling_interval": 4}),
    ],
)
def test_plan_uses_segment_driver_guards(segments, change):
    """Planning must reject the same driver-only invalid call as execution."""
    args, kwargs = _case("scalar", 2)
    args[0].requires_grad_(True)
    kwargs = {**kwargs, **change}
    with pytest.raises((DeepwaveValueError, DeepwaveTypeError)) as planned:
        _plan("scalar", args, kwargs, checkpoint_segments=segments)
    with pytest.raises((DeepwaveValueError, DeepwaveTypeError)) as executed:
        _call("scalar", args, kwargs, checkpoint_segments=segments)
    assert str(planned.value) == str(executed.value)


def test_segmented_plan_resolves_automatic_cfl_like_execution():
    """Automatic CFL is supported and uses the same whole-record preparation."""
    args, kwargs = _case("scalar", 2)
    args[0].requires_grad_(True)
    del kwargs["max_vel"]
    planned = _plan("scalar", args, kwargs, checkpoint_segments=2)
    output = _call("scalar", args, kwargs, checkpoint_segments=2)
    assert planned.info == output.info
    assert planned.info.step_ratio > 1


def test_segmented_plan_applies_nt_once_and_matches_module():
    """Functional and Module plans see the same shortened whole record."""
    args, kwargs = _case("scalar", 2)
    nt = 40
    functional = _plan("scalar", args, kwargs, checkpoint_segments=4, nt=nt)
    module = deepwave.Scalar(args[0], kwargs["grid_spacing"])
    rest = {k: v for k, v in kwargs.items() if k not in ("grid_spacing", "dt")}
    modular = deepwave.plan(module, kwargs["dt"], checkpoint_segments=4, nt=nt, **rest)
    assert functional.info == modular.info
    assert functional.info.nt == nt
    assert functional.memory == modular.memory


def test_segmented_plan_accepts_mapped_survey_with_nt():
    """The real delegate expands mapped channels before the segment guard."""
    args, kwargs = _case("scalar", 2)
    mapped = {
        k: v
        for k, v in kwargs.items()
        if k not in ("source_amplitudes", "source_locations", "receiver_locations")
    }
    mapped["sources"] = {"u": (kwargs["source_amplitudes"], kwargs["source_locations"])}
    mapped["receivers"] = {"u": kwargs["receiver_locations"]}
    planned = _plan("scalar", args, mapped, checkpoint_segments=2, nt=12)
    executed = _call("scalar", args, mapped, checkpoint_segments=2, nt=12)
    whole = _call("scalar", args, mapped, nt=12)
    assert planned.info == whole.info
    assert executed.info.nt == planned.info.nt


def test_checkpoint_segments_accepts_nt_with_source_amplitudes():
    """``nt`` fits the record once, before it is divided into segments."""
    args, kwargs = _case("scalar", 2)
    nt = kwargs["source_amplitudes"].shape[-1] - 4
    segmented = _call("scalar", args, kwargs, checkpoint_segments=2, nt=nt)
    trimmed = dict(kwargs)
    trimmed["source_amplitudes"] = kwargs["source_amplitudes"][..., :nt]
    direct = _call("scalar", args, trimmed, checkpoint_segments=2)
    assert segmented.info.nt == nt
    assert torch.equal(segmented.receiver_amplitudes, direct.receiver_amplitudes)


def test_checkpoint_segments_still_refuses_nt_without_source_amplitudes():
    """With nothing to divide, ``nt`` has no meaning for the segment driver."""
    args, kwargs = _case("scalar", 2)
    sourceless = {
        k: v for k, v in kwargs.items() if not k.startswith("source_amplitudes")
    }
    sourceless = {
        k: v for k, v in sourceless.items() if not k.startswith("source_locations")
    }
    sourceless["wavefield_0"] = torch.zeros(1, 40, 40)
    with pytest.raises(DeepwaveValueError, match="nt="):
        _call("scalar", args, sourceless, checkpoint_segments=2, nt=10)


def test_plan_rejects_a_bool_segment_count():
    """The bool guard applies to a plan as it does to a call."""
    args, kwargs = _case("scalar", 2)
    with pytest.raises(DeepwaveTypeError, match="not a bool"):
        _plan("scalar", args, kwargs, checkpoint_segments=True)


def test_checkpoint_segments_true_is_a_type_error():
    """``True`` must not silently mean 'one segment'."""
    args, kwargs = _case("scalar", 2)
    with pytest.raises(DeepwaveTypeError, match="not a bool"):
        _call("scalar", args, kwargs, checkpoint_segments=True)


@pytest.mark.parametrize("bad", [0, -3])
def test_checkpoint_segments_must_be_positive(bad):
    """A segment count below one is refused."""
    args, kwargs = _case("scalar", 2)
    with pytest.raises(DeepwaveValueError, match="positive int"):
        _call("scalar", args, kwargs, checkpoint_segments=bad)


@pytest.mark.parametrize(
    ("extra", "named", "alternative"),
    [
        (
            {"wavefield_0": torch.zeros(1, 40, 40)},
            "wavefield_0=",
            "initial_state=",
        ),
        (
            {"forward_callback": lambda state: None},
            "forward_callback=",
            "run_segments",
        ),
        (
            {"backward_callback": lambda state: None},
            "backward_callback=",
            "run_segments",
        ),
    ],
)
def test_checkpoint_segments_rejects_what_it_cannot_express(extra, named, alternative):
    """Each unsupported argument raises, naming itself and the way round it."""
    args, kwargs = _case("scalar", 2)
    with pytest.raises(DeepwaveValueError) as error:
        _call("scalar", args, kwargs, checkpoint_segments=2, **extra)
    assert named in str(error.value)
    assert alternative in str(error.value)


def test_checkpoint_segments_supports_the_storage_arguments(tmp_path):
    """The storage policy reaches every segment, and changes no number."""
    args, kwargs = _case("scalar", 2, dtype=torch.float64)
    v = args[0].clone().requires_grad_(True)
    device_run = _call("scalar", [v], kwargs, checkpoint_segments=3)
    device_run.receiver_amplitudes.pow(2).sum().backward()
    device_grad = v.grad.clone()
    v.grad = None
    disk_run = _call(
        "scalar",
        [v],
        kwargs,
        checkpoint_segments=3,
        storage=deepwave.Storage(mode="disk", path=str(tmp_path)),
    )
    disk_run.receiver_amplitudes.pow(2).sum().backward()
    assert torch.equal(device_run.receiver_amplitudes, disk_run.receiver_amplitudes)
    assert torch.equal(device_grad, v.grad)


def test_checkpoint_segments_on_a_module_matches_the_functional():
    """The Module forwards the keyword to its functional unchanged."""
    args, kwargs = _case("scalar", 2, dtype=torch.float64)
    module = deepwave.Scalar(args[0], kwargs["grid_spacing"])
    rest = {
        k: v for k, v in kwargs.items() if k not in ("grid_spacing", "dt", "max_vel")
    }
    from_module = module(
        kwargs["dt"], **rest, max_vel=kwargs["max_vel"], checkpoint_segments=3
    )
    functional = _call("scalar", args, kwargs, checkpoint_segments=3)
    assert torch.equal(from_module.receiver_amplitudes, functional.receiver_amplitudes)
