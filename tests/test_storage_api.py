"""The ``storage=``/``Storage`` policy argument and the lossy-compression rename.

Covers the one-argument storage policy, its equivalence to the individual
``storage_mode``/``storage_path``/``storage_compression`` arguments that remain
permanently supported, the conflict rules between the two spellings, the
call-time resolution of ``storage_path``, and the documented properties of
lossy compression.

The equivalence tests are bit-exactness tests: a second spelling of the same
policy that produced *nearly* the same gradient would be a defect, so they
assert ``torch.equal`` rather than a tolerance.
"""

import dataclasses
import pathlib
import warnings

import pytest
import torch

import deepwave
from deepwave import _storage
from deepwave._storage import Storage

ALL_PROPAGATORS = (
    "scalar",
    "scalar_born",
    "acoustic",
    "acoustic_born",
    "acoustic_vti",
    "viscoacoustic",
    "elastic",
    "elastic_born",
    "elastic_vti",
)

MODES = ("device", "cpu", "disk", "none")


def _scalar_problem(dtype: torch.dtype = torch.float64) -> dict[str, object]:
    """A tiny differentiable ``scalar`` call, small enough to run many times."""
    nt = 40
    dt = 0.004
    return {
        "grid_spacing": 4.0,
        "dt": dt,
        "source_amplitudes": deepwave.wavelets.ricker(15.0, nt, dt, 0.05).to(dtype)[
            None, None
        ],
        "source_locations": torch.tensor([[[5, 5]]]),
        "receiver_locations": torch.tensor([[[20, 20]]]),
        "pml_freq": 15.0,
        "max_vel": 1500.0,
    }


def _run_scalar(**storage_kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the tiny problem, returning (receivers, model gradient)."""
    problem = _scalar_problem()
    v = torch.full((30, 30), 1500.0, dtype=torch.float64, requires_grad=True)
    out = deepwave.scalar(
        v,
        problem.pop("grid_spacing"),
        problem.pop("dt"),
        **problem,  # type: ignore[arg-type]
        **storage_kwargs,  # type: ignore[arg-type]
    )
    out.receiver_amplitudes.pow(2).sum().backward()
    assert v.grad is not None
    assert v.grad.abs().max() > 0, "vacuous: the gradient is identically zero"
    return out.receiver_amplitudes.detach().clone(), v.grad.clone()


# ---------------------------------------------------------------------------
# The Storage object itself
# ---------------------------------------------------------------------------


def test_storage_frozen() -> None:
    """``Storage`` is immutable, so one can be shared across a sweep."""
    storage = Storage(mode="cpu")
    with pytest.raises(dataclasses.FrozenInstanceError):
        storage.mode = "disk"  # type: ignore[misc]


def test_storage_replace_is_the_documented_derivation() -> None:
    """``dataclasses.replace`` gives a variant without mutating the original."""
    base = Storage(mode="cpu", lossy_compression=True)
    derived = dataclasses.replace(base, mode="disk")
    assert derived.mode == "disk"
    assert derived.lossy_compression is True
    assert base.mode == "cpu"


def test_storage_rejects_an_unknown_mode() -> None:
    """An unrecognised mode raises, with the one shared wording."""
    with pytest.raises(deepwave.DeepwaveValueError, match="storage_mode must be"):
        Storage(mode="bogus")  # type: ignore[arg-type]


def test_storage_mode_message_is_single_sourced() -> None:
    """The three paths that reject a mode all use identical text.

    They had drifted before this was single-sourced: one copy interpolated
    ``repr(value)`` and ended in a full stop while the other did neither.
    """
    from deepwave._storage import storage_mode_error_message

    expected = storage_mode_error_message("bogus")
    with pytest.raises(deepwave.DeepwaveValueError) as storage_exc:
        Storage(mode="bogus")  # type: ignore[arg-type]
    assert str(storage_exc.value) == expected

    with pytest.raises(ValueError, match="storage_mode must be") as estimate_exc:
        deepwave.estimate_memory(
            model_shape=(10, 10),
            nt=5,
            n_shots=1,
            propagator="scalar",
            storage_mode="bogus",  # type: ignore[arg-type]
        )
    assert str(estimate_exc.value) == expected


def test_storage_repr_is_one_line_and_has_no_tensors() -> None:
    """The repr is one short line, and is honest about an unresolved path."""
    text = repr(Storage(mode="disk", path="/scratch/me", lossy_compression=True))
    assert "\n" not in text
    assert text == ("Storage(mode='disk', path='/scratch/me', lossy_compression=True)")
    assert str(Storage(mode="disk", path="/x")) == repr(Storage(mode="disk", path="/x"))


def test_storage_repr_does_not_invent_a_path() -> None:
    """``path=None`` says it is resolved later, rather than naming a directory."""
    assert "None (resolved at call time)" in repr(Storage())
    assert "resolved at call time" in repr(Storage())
    assert repr(Storage()).count("path=") == 1


def test_storage_path_ignored_by_other_modes_warns() -> None:
    """A path no mode will read is a mistake worth naming -- but not fatal."""
    _storage_warn_reset()
    with pytest.warns(UserWarning, match="is ignored when mode='cpu'"):
        Storage(mode="cpu", path="/scratch")


def test_storage_compression_with_mode_none_warns() -> None:
    """Compression of a history that is never stored is likewise inert."""
    _storage_warn_reset()
    with pytest.warns(UserWarning, match="ignored when mode='none'"):
        Storage(mode="none", lossy_compression=True)


def _storage_warn_reset() -> None:
    """Clear the once-per-process warning registry these tests share."""
    from deepwave import _common

    _common._warned_once_causes.discard("storage_path_ignored")
    _common._warned_once_causes.discard("storage_compression_ignored")
    _common._warned_once_causes.discard("storage_default_disk_path")


# ---------------------------------------------------------------------------
# The resolution funnel
# ---------------------------------------------------------------------------


def test_resolve_storage_returns_none_when_nothing_is_given() -> None:
    """A missing policy is distinguishable from an explicit default one.

    That distinction is what lets ``Module.forward`` tell an override from a
    default and fall back to the constructor's value.
    """
    assert _storage.resolve_storage() is None


def test_storage_default_none_is_not_a_conflict() -> None:
    """An omitted policy leaves the individual arguments available."""
    resolved = _storage.resolve_storage(storage_mode="cpu")
    assert resolved is not None
    assert resolved.mode == "cpu"
    # And through a real call.
    _run_scalar(storage_mode="cpu")


def test_explicit_storage_device_is_still_a_conflict() -> None:
    """``storage="device"`` written out is a policy, not an absence of one."""
    with pytest.raises(deepwave.DeepwaveValueError, match="not both"):
        _storage.resolve_storage("device", storage_path="/scratch")


@pytest.mark.parametrize(
    ("kwarg", "value"),
    [
        ("storage_mode", "cpu"),
        ("storage_path", "/scratch"),
        ("storage_compression", True),
    ],
)
def test_storage_conflicts_raise(kwarg: str, value: object) -> None:
    """``storage=`` with any individual argument raises, naming both."""
    with pytest.raises(deepwave.DeepwaveValueError) as exc:
        _storage.resolve_storage("disk", **{kwarg: value})  # type: ignore[arg-type]
    message = str(exc.value)
    assert "storage='disk'" in message
    assert kwarg in message
    assert "deepwave.Storage" in message


def test_storage_conflict_raises_through_a_propagator() -> None:
    """The funnel is reached from a real call, not only in isolation."""
    with pytest.raises(deepwave.DeepwaveValueError, match="not both"):
        _run_scalar(storage="disk", storage_mode="cpu")


def test_storage_lossy_compression_is_not_a_keyword() -> None:
    """The long spelling never shipped and is not accepted anywhere."""
    with pytest.raises(TypeError, match="storage_lossy_compression"):
        _run_scalar(storage_lossy_compression=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="storage_lossy_compression"):
        _storage.resolve_storage(storage_lossy_compression=True)  # type: ignore[call-arg]


def test_storage_rejects_a_wrong_type() -> None:
    """``storage=`` takes a mode string or a Storage, and says so."""
    with pytest.raises(deepwave.DeepwaveValueError, match="must be a mode string"):
        _storage.resolve_storage(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Equivalence: every spelling of one policy produces identical numbers
# ---------------------------------------------------------------------------


def test_storage_string_equals_storage_mode() -> None:
    """``storage=m`` and ``storage_mode=m`` are the same run, bit for bit."""
    for mode in ("device", "cpu"):
        by_mode = _run_scalar(storage_mode=mode)
        by_storage = _run_scalar(storage=mode)
        assert torch.equal(by_mode[0], by_storage[0])
        assert torch.equal(by_mode[1], by_storage[1])


def test_storage_object_equals_triple(tmp_path: pathlib.Path) -> None:
    """A ``Storage`` equals the individual arguments carrying the same policy."""
    triple = _run_scalar(
        storage_mode="disk", storage_path=str(tmp_path), storage_compression=True
    )
    obj = _run_scalar(
        storage=Storage(mode="disk", path=str(tmp_path), lossy_compression=True)
    )
    assert torch.equal(triple[0], obj[0])
    assert torch.equal(triple[1], obj[1])


def test_storage_default_equals_passing_nothing() -> None:
    """Passing the default explicitly changes nothing."""
    nothing = _run_scalar()
    explicit = _run_scalar(storage=Storage())
    assert torch.equal(nothing[0], explicit[0])
    assert torch.equal(nothing[1], explicit[1])


# ---------------------------------------------------------------------------
# Call-time path resolution
# ---------------------------------------------------------------------------


def test_storage_path_resolved_at_call_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """``TMPDIR`` set *after* importing deepwave is honoured.

    This is the point of the change: the default used to be a
    default-argument expression, evaluated once at import, so this test
    would fail on the released behaviour.
    """
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    resolved = _storage.materialise_storage(None)
    assert resolved.path == str(tmp_path)
    assert resolved.used_default_path is True


def test_storage_path_is_not_frozen_in_any_signature() -> None:
    """No entry point carries a temporary directory as a default argument.

    The failure this prevents is subtle and permanent once shipped: a
    ``tempfile.gettempdir()`` default-argument expression bakes one machine's
    directory into the rendered signature and ignores ``TMPDIR`` for the life
    of the process.
    """
    import inspect

    for name in ALL_PROPAGATORS:
        signature = inspect.signature(getattr(deepwave, name))
        assert signature.parameters["storage_path"].default is None, name
    run_segments_sig = inspect.signature(deepwave.checkpointing.run_segments)
    assert run_segments_sig.parameters["storage_path"].default is None


def test_storage_path_accepts_pathlib(tmp_path: pathlib.Path) -> None:
    """A ``pathlib.Path`` works, and gives the same numbers as its string."""
    as_path = _run_scalar(storage_mode="disk", storage_path=tmp_path)
    as_str = _run_scalar(storage_mode="disk", storage_path=str(tmp_path))
    assert torch.equal(as_path[0], as_str[0])
    assert torch.equal(as_path[1], as_str[1])


def test_disk_default_path_warning_names_the_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The tmpfs warning fires once, names the directory, and is avoidable."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    resolved = _storage.materialise_storage(None)
    resolved = dataclasses.replace(resolved, mode="disk")

    _storage_warn_reset()
    with pytest.warns(UserWarning, match="default directory") as record:
        _storage.warn_if_default_disk_path(resolved, 41_300_000_000, stacklevel=2)
    message = str(record[0].message)
    assert str(tmp_path) in message
    assert "41.3 GB" in message
    assert "storage_path=" in message
    assert "TMPDIR" in message

    # Once per process.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _storage.warn_if_default_disk_path(resolved, 41_300_000_000, stacklevel=2)


def test_disk_default_path_warns_from_a_real_run() -> None:
    """The warning reaches a real ``storage_mode="disk"`` call, not just the helper.

    Every propagator's ``storage_path`` documentation promises this, so a
    helper with a test but no caller would make that promise false. Checked
    with the default warning filters left alone except for the ``record``
    capture, and asserted to point at the caller's own line rather than
    somewhere inside Deepwave.
    """
    _storage_warn_reset()
    with pytest.warns(UserWarning, match="default directory") as record:
        _run_scalar(storage_mode="disk")
    assert len(record) >= 1
    message = str(record[0].message)
    assert "tempfile.gettempdir()" in message
    assert "storage_path=" in message
    # The warning must blame the user's call, not deepwave's internals.
    assert record[0].filename == __file__

    # Once per process.
    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        _run_scalar(storage_mode="disk")
    assert not [w for w in second if "default directory" in str(w.message)]


def test_disk_explicit_path_never_warns(tmp_path: pathlib.Path) -> None:
    """Choosing the directory yourself is not something to be warned about."""
    _storage_warn_reset()
    resolved = _storage.materialise_storage(Storage(mode="disk", path=str(tmp_path)))
    assert resolved.used_default_path is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _storage.warn_if_default_disk_path(resolved, 41_300_000_000, stacklevel=2)


def test_unwritable_storage_path_error_names_the_path(
    tmp_path: pathlib.Path,
) -> None:
    """An unwritable directory raises DeepwaveOSError naming ``storage_path``."""
    import os

    if os.geteuid() == 0:
        pytest.skip("root can write to a 0o500 directory")
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    try:
        with pytest.raises(deepwave.DeepwaveOSError) as exc:
            _storage.TemporaryStorage(str(locked), 1, history_bytes=41_300_000_000)
        message = str(exc.value)
        assert str(locked) in message
        assert "storage_path=" in message
        assert "41.3 GB" in message
        assert isinstance(exc.value, OSError)
    finally:
        locked.chmod(0o700)


# ---------------------------------------------------------------------------
# Lossy compression
# ---------------------------------------------------------------------------


def test_lossy_compression_actually_changes_the_model_gradient() -> None:
    """A vacuity guard: if this passed with compression off it proves nothing."""
    plain = _run_scalar()
    compressed = _run_scalar(storage_compression=True)
    assert not torch.equal(plain[1], compressed[1])
    relative = (compressed[1] - plain[1]).norm() / plain[1].norm()
    assert 1e-4 < relative < 0.6, relative


def test_forward_output_is_bit_identical_under_compression() -> None:
    """The docstring's "unaffected" claims are asserted, not just stated."""
    plain = _run_scalar()
    compressed = _run_scalar(storage_compression=True)
    assert torch.equal(plain[0], compressed[0])


def test_source_and_initial_wavefield_gradients_unaffected_by_compression() -> None:
    """Compression touches only the history that forms the *model* gradient."""
    problem = _scalar_problem()
    grid_spacing = problem.pop("grid_spacing")
    dt = problem.pop("dt")

    def run(compress: bool) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        v = torch.full((30, 30), 1500.0, dtype=torch.float64)
        amplitudes = problem["source_amplitudes"].clone().requires_grad_(True)  # type: ignore[union-attr]
        wavefield_0 = torch.randn(1, 30, 30, dtype=torch.float64) * 1e-3
        wavefield_0.requires_grad_(True)
        kwargs = dict(problem)
        kwargs["source_amplitudes"] = amplitudes
        out = deepwave.scalar(
            v,
            grid_spacing,  # type: ignore[arg-type]
            dt,  # type: ignore[arg-type]
            wavefield_0=wavefield_0,
            storage_compression=compress,
            **kwargs,  # type: ignore[arg-type]
        )
        out.receiver_amplitudes.pow(2).sum().backward()
        assert amplitudes.grad is not None
        assert wavefield_0.grad is not None
        return amplitudes.grad.clone(), wavefield_0.grad.clone()

    plain_source, plain_field = run(False)
    comp_source, comp_field = run(True)
    assert plain_source.abs().max() > 0
    assert plain_field.abs().max() > 0
    assert torch.equal(plain_source, comp_source)
    assert torch.equal(plain_field, comp_field)


def test_every_entry_point_documents_lossiness() -> None:
    """No rendered page can show a compression argument without "Lossy".

    The "See :func:`deepwave.elastic`" pattern used by nine of the nine
    propagators would otherwise hide the warning behind a cross-reference.
    """
    import inspect

    checked = 0
    for name in ALL_PROPAGATORS:
        functional = getattr(deepwave, name)
        class_name = "".join(part.title() for part in name.split("_"))
        module_class = getattr(deepwave, class_name)
        for obj in (functional, module_class):
            doc = inspect.getdoc(obj) or ""
            assert "storage_compression" in doc, (name, obj)
            index = doc.index("storage_compression:")
            entry = doc[index : index + 400]
            assert "Lossy" in entry or "lossy" in entry, (name, obj)
            checked += 1
    assert checked == 2 * len(ALL_PROPAGATORS)


# ---------------------------------------------------------------------------
# Module.forward
# ---------------------------------------------------------------------------


def test_forward_storage_overrides_constructor() -> None:
    """``forward``'s storage arguments beat the constructor's, bit for bit."""
    problem = _scalar_problem()
    grid_spacing = problem.pop("grid_spacing")
    dt = problem.pop("dt")

    def run(ctor: dict[str, object], fwd: dict[str, object]) -> torch.Tensor:
        v = torch.full((30, 30), 1500.0, dtype=torch.float64)
        module = deepwave.Scalar(v, grid_spacing, requires_grad=True, **ctor)  # type: ignore[arg-type]
        out = module(dt, **problem, **fwd)  # type: ignore[arg-type]
        out.receiver_amplitudes.pow(2).sum().backward()
        assert module.v.grad is not None
        return module.v.grad.clone()

    reference = run({"storage_mode": "cpu"}, {})
    overridden = run({"storage_mode": "device"}, {"storage_mode": "cpu"})
    assert torch.equal(reference, overridden)


def test_forward_storage_none_uses_the_constructor_value() -> None:
    """Omitting them on ``forward`` reproduces the constructor's policy."""
    problem = _scalar_problem()
    grid_spacing = problem.pop("grid_spacing")
    dt = problem.pop("dt")
    v = torch.full((30, 30), 1500.0, dtype=torch.float64)
    module = deepwave.Scalar(
        v,
        grid_spacing,
        requires_grad=True,  # type: ignore[arg-type]
        storage_mode="cpu",
    )
    module(dt, **problem)  # type: ignore[arg-type]
    assert module.storage_mode == "cpu"


def test_constructor_attributes_are_back_filled() -> None:
    """``module.storage_mode`` keeps working, and reflects ``storage=``."""
    v = torch.full((30, 30), 1500.0, dtype=torch.float64)
    plain = deepwave.Scalar(v, 4.0)
    assert plain.storage_mode == "device"
    assert plain.storage_path is None
    assert plain.storage_compression is False

    combined = deepwave.Scalar(
        v, 4.0, storage=Storage(mode="disk", path="/scratch", lossy_compression=True)
    )
    assert combined.storage_mode == "disk"
    assert combined.storage_path == "/scratch"
    assert combined.storage_compression is True


def test_module_default_carries_no_policy() -> None:
    """A Module given no storage arguments holds ``None``, not a default policy.

    That is what lets ``forward``'s ``None`` mean "use the constructor's
    value" without the constructor's own default masking a ``forward``
    override.
    """
    v = torch.full((30, 30), 1500.0, dtype=torch.float64)
    assert deepwave.Scalar(v, 4.0).storage is None
    assert deepwave.Scalar(v, 4.0, storage_mode="cpu").storage == Storage(mode="cpu")


@pytest.mark.parametrize("name", ALL_PROPAGATORS)
def test_every_module_forward_accepts_all_storage_arguments(name: str) -> None:
    """All nine, not just the one the behavioural tests use."""
    import inspect

    class_name = "".join(part.title() for part in name.split("_"))
    parameters = inspect.signature(getattr(deepwave, class_name).forward).parameters
    for argument in (
        "storage",
        "storage_mode",
        "storage_path",
        "storage_compression",
    ):
        assert argument in parameters, (name, argument)
        if argument == "storage":
            assert parameters[argument].default is None
        else:
            assert parameters[argument].default is None


@pytest.mark.parametrize("name", ALL_PROPAGATORS)
def test_every_functional_accepts_all_storage_arguments(name: str) -> None:
    """The same, on the functional interface."""
    import inspect

    parameters = inspect.signature(getattr(deepwave, name)).parameters
    for argument in (
        "storage",
        "storage_mode",
        "storage_path",
        "storage_compression",
    ):
        assert argument in parameters, (name, argument)


def test_backend_failure_message_names_the_disk_path_and_its_free_space(
    tmp_path,
):
    """A compiled failure under disk storage says what most likely went wrong.

    The first sentence must stay byte-identical whatever the storage mode --
    it is the part callers and other tests match on -- so the storage clause
    is appended rather than woven in.
    """
    from deepwave._engine import backend_failure_message

    plain = backend_failure_message("scalar", "forward", 1)
    assert plain.startswith(
        "Compiled backend failed: the 'scalar' propagator's compiled forward "
        "pass returned non-zero status code 1."
    )
    assert "storage_path" not in plain

    enriched = backend_failure_message("scalar", "forward", 1, "disk", str(tmp_path))
    assert enriched.startswith(
        "Compiled backend failed: the 'scalar' propagator's compiled forward "
        "pass returned non-zero status code 1."
    )
    assert f"storage_path={str(tmp_path)!r}" in enriched
    assert "free" in enriched
    assert 'storage_mode="cpu"' in enriched
    assert "docs/troubleshooting.rst" in enriched
    # A mode that writes nothing to disk gets no disk clause.
    assert (
        backend_failure_message("scalar", "forward", 1, "cpu", str(tmp_path)) == plain
    )
