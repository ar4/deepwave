"""Tests for the deprecated ``python_backend`` alias of ``backend``.

Every propagator's public parameter is ``backend: str = "auto"``.
``python_backend`` is still accepted as a deprecated alias -- on **all
nine** propagators, not only the four ``v0.0.27`` shipped, because a
4-of-9 alias breaks every ``**kwargs`` wrapper; see
``tests/test_signature_consistency.py`` for the guard on where it exists.

It is handled by :func:`deepwave._engine._coerce_backend`: passing any
non-``None`` value emits a ``FutureWarning`` naming ``backend`` as the
replacement and the removal version, and the value is mapped **as
``v0.0.27`` mapped it**:

* ``True`` -> ``"torch_compile"`` when the installed compiled library was
  built with OpenMP, ``"eager"`` otherwise. ``v0.0.27`` chose between
  ``torch.compile`` and ``torch.jit.script`` on exactly that condition
  (``deepwave.backend_utils.USE_OPENMP``) and never ran eager, so an
  unconditional ``True -> "eager"`` mapping would be a silent change of
  meaning for released code. Deepwave no longer calls ``torch.jit`` at all,
  so the branch that chose it runs the eager step functions instead -- the
  same arithmetic, differing only in rounding. This file pins both branches.
* ``False`` -> ``"auto"`` (``v0.0.27`` used the compiled C backend).
* ``"jit"`` -> ``"eager"``, for the same reason.
* ``"eager"``/``"compile"`` -> their canonical spellings.

Passing both a non-default ``backend`` and ``python_backend`` raises
``ValueError``.

``python_backend=None`` is read as "the alias was not given", so it neither
warns nor overrides ``backend``. In ``v0.0.27`` ``None`` was outside the
parameter's declared type and behaviourally identical to omitting it (both
selected the compiled backend), so the only thing this loses is a warning on
a usage that was never declared -- in exchange for the alias being typed
honestly rather than defaulting to an ``Any``-typed sentinel.

This file covers the coercion helper directly (message/category/mapping
precision) and end-to-end propagator calls (function and Module) for the
deprecated alias path. It is the one file in the test suite that exercises
``python_backend=`` -- every other test file uses ``backend=``.
"""

import warnings

import pytest
import torch

import deepwave
import deepwave._backend_utils
import deepwave._engine

# ---------------------------------------------------------------------------
# _coerce_backend unit tests
# ---------------------------------------------------------------------------


def test_coerce_backend_only_no_warning() -> None:
    """Passing only ``backend`` (no ``python_backend``) never warns.

    Also pins the internal mode each canonical value resolves to: the
    vocabulary every helper downstream of this function has always spoken.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert deepwave._engine._coerce_backend("auto").mode is None
        assert deepwave._engine._coerce_backend("compiled").mode is None
        assert deepwave._engine._coerce_backend("eager").mode == "eager"
        assert deepwave._engine._coerce_backend("torch_compile").mode == "compile"


@pytest.mark.parametrize("spelling", ["jit", "torch_jit"])
def test_coerce_backend_rejects_the_jit_spellings(spelling: str) -> None:
    """``backend`` has no ``torch.jit`` mode, under either spelling."""
    with pytest.raises(deepwave.DeepwaveValueError, match=r"backend must be one of"):
        deepwave._engine._coerce_backend(spelling)


def test_coerce_backend_rejects_compile_as_a_backend_value() -> None:
    """``"compile"`` is a ``python_backend`` value only, never a ``backend`` one."""
    with pytest.raises(deepwave.DeepwaveValueError, match=r"got 'compile'\."):
        deepwave._engine._coerce_backend("compile")
    # Under the deprecated argument that shipped it, it still maps.
    with pytest.warns(FutureWarning):
        assert (
            deepwave._engine._coerce_backend("auto", python_backend="compile").canonical
            == "torch_compile"
        )


def test_coerce_backend_only_compiled_is_strict() -> None:
    """Only an explicit ``"compiled"`` demands the compiled kernel."""
    assert deepwave._engine._coerce_backend("compiled").strict
    assert not deepwave._engine._coerce_backend("auto").strict
    assert not deepwave._engine._coerce_backend("eager").strict


def test_coerce_backend_invalid_string_raises_value_error() -> None:
    """An unrecognised ``backend`` string raises ``ValueError`` listing the values."""
    with pytest.raises(ValueError, match=r"got 'bogus'\.") as excinfo:
        deepwave._engine._coerce_backend("bogus")
    message = str(excinfo.value)
    for value in deepwave._engine.BACKEND_VALUES:
        assert repr(value) in message, value


@pytest.mark.parametrize(
    ("python_backend", "expected"),
    [
        (False, "auto"),
        ("eager", "eager"),
        ("jit", "eager"),
        ("compile", "torch_compile"),
    ],
)
def test_coerce_python_backend_alias_mapping(
    python_backend: object, expected: str
) -> None:
    """Every deprecated-alias value warns and maps to its canonical equivalent."""
    with pytest.warns(
        FutureWarning,
        match="python_backend is deprecated and will be removed in Deepwave 0.4.0",
    ):
        result = deepwave._engine._coerce_backend("auto", python_backend=python_backend)
    assert result.canonical == expected


def test_python_backend_true_maps_to_openmp_dependent_mode() -> None:
    """``True`` reproduces ``v0.0.27``: torch.compile on OpenMP, eager else.

    ``v0.0.27``'s ``python_backend=True`` chose
    ``mode = "compile" if backend_utils.USE_OPENMP else "jit"``, so it never
    meant eager on an OpenMP build; mapping it to ``"eager"`` unconditionally
    would silently change what released code did, which is the regression
    this pins. The non-OpenMP branch is ``"eager"`` because Deepwave no
    longer calls ``torch.jit``: the same step functions, differing only in
    rounding. Both branches are exercised, the second by monkeypatching
    ``USE_OPENMP`` (this box's build has it enabled).
    """
    for use_openmp, expected in ((True, "torch_compile"), (False, "eager")):
        saved = deepwave._backend_utils.USE_OPENMP
        deepwave._backend_utils.USE_OPENMP = use_openmp
        try:
            with pytest.warns(FutureWarning, match="python_backend is deprecated"):
                result = deepwave._engine._coerce_backend("auto", python_backend=True)
        finally:
            deepwave._backend_utils.USE_OPENMP = saved
        assert result.canonical == expected, use_openmp


@pytest.mark.parametrize("backend", ["auto", "compiled", "eager", "torch_compile"])
def test_coerce_python_backend_none_means_not_given(backend: str) -> None:
    """``python_backend=None`` neither warns nor overrides ``backend``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert deepwave._engine._coerce_backend(
            backend, python_backend=None
        ) == deepwave._engine._coerce_backend(backend)


def test_coerce_python_backend_invalid_value_raises_value_error() -> None:
    """An unrecognised deprecated-alias value still warns, then raises."""
    with (
        pytest.warns(FutureWarning, match="python_backend is deprecated"),
        pytest.raises(
            ValueError,
            match=(
                r"python_backend must be None, a bool, 'eager', 'jit', or "
                r"'compile', got 'bogus'\."
            ),
        ),
    ):
        deepwave._engine._coerce_backend("auto", python_backend="bogus")


def test_coerce_both_backend_and_python_backend_raises_value_error() -> None:
    """Passing a non-default ``backend`` together with ``python_backend`` raises.

    The warning is still emitted (the deprecated alias was used), even
    though the call goes on to raise for the conflicting pair.
    """
    with (
        pytest.warns(FutureWarning, match="python_backend is deprecated"),
        pytest.raises(
            ValueError,
            match=(
                r"Pass only backend, not both backend and the deprecated "
                r"python_backend\."
            ),
        ),
    ):
        deepwave._engine._coerce_backend("eager", python_backend="jit")


def test_coerce_backend_default_and_python_backend_default_does_not_raise() -> None:
    """Both left at their defaults: no warning, resolves to the compiled backend."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        request = deepwave._engine._coerce_backend("auto")
    assert request.canonical == "auto"
    assert request.mode is None


# ---------------------------------------------------------------------------
# End-to-end: deprecated python_backend path on a real propagator (scalar)
# ---------------------------------------------------------------------------


def _tiny_scalar_kwargs() -> dict:
    v = 1500 * torch.ones(10, 10)
    return {
        "v": v,
        "grid_spacing": 4.0,
        "dt": 0.001,
        "source_amplitudes": torch.randn(1, 1, 5),
        "source_locations": torch.tensor([[[0, 0]]]),
        "receiver_locations": torch.tensor([[[0, 5]]]),
        "pml_freq": 25.0,
    }


def test_scalar_python_backend_true_deprecated_runs_a_python_backend() -> None:
    """``python_backend=True`` warns and runs the mode ``v0.0.27`` would have.

    On this box both Python modes are bit-identical to each other for
    ``scalar``, which is why the numbers may be compared against ``eager``;
    the *reported* backend is the one ``v0.0.27`` would have selected, and
    that is what is pinned.
    """
    kwargs = _tiny_scalar_kwargs()
    v = kwargs.pop("v")

    with pytest.warns(FutureWarning, match="python_backend is deprecated"):
        out_true = deepwave.scalar(v, python_backend=True, **kwargs)
    expected = "torch_compile" if deepwave._backend_utils.USE_OPENMP else "eager"
    assert out_true.info.backend == expected
    out_eager = deepwave.scalar(v, backend="eager", **kwargs)
    assert torch.equal(out_true[-1], out_eager[-1])


def test_python_backend_jit_runs_the_eager_backend() -> None:
    """``python_backend="jit"`` runs the eager backend, and reports it.

    ``v0.0.27``'s TorchScript mode no longer exists, so the value that named
    it runs the eager step functions -- the same arithmetic, differing only
    in rounding -- rather than raising on code that still passes it.
    """
    kwargs = _tiny_scalar_kwargs()
    v = kwargs.pop("v")
    with pytest.warns(FutureWarning, match="python_backend is deprecated"):
        out_jit = deepwave.scalar(v, python_backend="jit", **kwargs)
    assert out_jit.info.backend == "eager"
    out_eager = deepwave.scalar(v, backend="eager", **kwargs)
    assert torch.equal(out_jit[-1], out_eager[-1])


def test_scalar_python_backend_false_deprecated_matches_compiled_default() -> None:
    """``python_backend=False`` warns; behaves identically to the default."""
    kwargs = _tiny_scalar_kwargs()
    v = kwargs.pop("v")

    with pytest.warns(FutureWarning, match="python_backend is deprecated"):
        out_false = deepwave.scalar(v, python_backend=False, **kwargs)
    out_default = deepwave.scalar(v, **kwargs)

    assert torch.equal(out_false[-1], out_default[-1])


def test_scalar_python_backend_none_is_silent_and_matches_default() -> None:
    """``python_backend=None`` means "not given": no warning, default backend."""
    kwargs = _tiny_scalar_kwargs()
    v = kwargs.pop("v")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out_none = deepwave.scalar(v, python_backend=None, **kwargs)
    out_default = deepwave.scalar(v, **kwargs)

    assert torch.equal(out_none[-1], out_default[-1])


def test_scalar_python_backend_none_does_not_override_backend() -> None:
    """``backend="eager"`` with ``python_backend=None`` is not a conflicting pair."""
    kwargs = _tiny_scalar_kwargs()
    v = kwargs.pop("v")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out_none = deepwave.scalar(v, backend="eager", python_backend=None, **kwargs)
    out_eager = deepwave.scalar(v, backend="eager", **kwargs)

    assert torch.equal(out_none[-1], out_eager[-1])


def test_python_backend_only_on_released_signatures() -> None:
    """The alias remains only on the four propagators shipped in v0.0.27."""
    import inspect

    import deepwave._capabilities

    released = {"scalar", "scalar_born", "acoustic", "elastic"}
    for name in deepwave._capabilities.PROPAGATOR_NAMES:
        params = inspect.signature(getattr(deepwave, name)).parameters
        assert ("python_backend" in params) == (name in released)


def test_python_backend_rejected_on_a_new_propagator() -> None:
    """New solvers expose only the canonical backend spelling."""
    c11 = 2.25e9 * torch.ones(20, 20, dtype=torch.float64)
    buoyancy = torch.full((20, 20), 1 / 2200.0, dtype=torch.float64)
    with pytest.raises(TypeError, match="python_backend"):
        deepwave.acoustic_vti(
            c11, c11, c11, buoyancy, 4.0, 0.001, python_backend="eager"
        )


def test_python_backend_warning_names_removal_version() -> None:
    """The deprecation warning states the release that removes the alias."""
    with pytest.warns(FutureWarning, match="python_backend is deprecated") as record:
        deepwave._engine._coerce_backend("auto", python_backend=False)
    assert "0.4.0" in str(record[0].message)


def test_python_backend_warning_points_at_user_line() -> None:
    """The warning blames the caller's own line, on both entry paths.

    ``stacklevel`` differs between them -- a ``Module.forward`` adds
    ``torch.nn.Module.__call__``'s frames -- so a single shared constant was
    wrong for one of the two. Both are pinned here.
    """
    kwargs = _tiny_scalar_kwargs()
    v = kwargs.pop("v")
    dt = kwargs.pop("dt")
    grid_spacing = kwargs.pop("grid_spacing")

    with pytest.warns(FutureWarning, match="python_backend is deprecated") as record:
        deepwave.scalar(v, grid_spacing, dt, python_backend=False, **kwargs)
    assert record[0].filename == __file__, "function path"

    module = deepwave.Scalar(v, grid_spacing)
    with pytest.warns(FutureWarning, match="python_backend is deprecated") as record:
        module(dt, python_backend=False, **kwargs)
    assert record[0].filename == __file__, "Module.forward path"
    # Exactly one: forward canonicalises the alias and hands the functional a
    # plain ``backend``, so the functional's own coercion must not warn again.
    alias_warnings = [
        w for w in record if "python_backend is deprecated" in str(w.message)
    ]
    assert len(alias_warnings) == 1, alias_warnings


def test_scalar_python_backend_invalid_value_raises_value_error() -> None:
    """An invalid ``python_backend`` string raises ``ValueError`` (exact message)."""
    kwargs = _tiny_scalar_kwargs()
    v = kwargs.pop("v")
    with pytest.raises(
        ValueError,
        match=(
            r"python_backend must be None, a bool, 'eager', 'jit', or "
            r"'compile', got 'bogus'\."
        ),
    ):
        deepwave.scalar(v, python_backend="bogus", **kwargs)


def test_scalar_both_backend_and_python_backend_raises_value_error() -> None:
    """Passing both ``backend`` and ``python_backend`` to a real propagator raises."""
    kwargs = _tiny_scalar_kwargs()
    v = kwargs.pop("v")
    with pytest.raises(
        ValueError,
        match=(
            r"Pass only backend, not both backend and the deprecated "
            r"python_backend\."
        ),
    ):
        deepwave.scalar(v, backend="torch_compile", python_backend="eager", **kwargs)


def test_scalar_module_forward_python_backend_true_deprecated() -> None:
    """The ``Scalar`` module's ``forward`` also coerces (and warns on) the alias."""
    kwargs = _tiny_scalar_kwargs()
    v = kwargs.pop("v")
    dt = kwargs.pop("dt")
    grid_spacing = kwargs.pop("grid_spacing")
    module = deepwave.Scalar(v, grid_spacing)

    with pytest.warns(FutureWarning, match="python_backend is deprecated"):
        out_true = module(dt, python_backend=True, **kwargs)
    expected = "torch_compile" if deepwave._backend_utils.USE_OPENMP else "eager"
    assert out_true.info.backend == expected
    out_eager = module(dt, backend="eager", **kwargs)
    assert torch.equal(out_true[-1], out_eager[-1])


# ---------------------------------------------------------------------------
# StepCache uses PyTorch's cache and leaves global compiler settings alone.
# ---------------------------------------------------------------------------


def test_step_cache_uses_one_wrapper_without_changing_torch_config(monkeypatch) -> None:
    """Configuration variants are PyTorch's concern, including its limits."""
    from deepwave._engine import StepCache

    created = []

    def fake_compile(fn):
        created.append(fn)
        return lambda x: fn(x)

    monkeypatch.setattr(torch, "compile", fake_compile)
    before = torch._dynamo.config.cache_size_limit
    cache = StepCache(lambda x: x + 1)
    first = cache.resolve("compile", (1, 2, torch.float32))
    second = cache.resolve("compile", (3, 8, torch.float64))
    assert first is second
    assert first(3) == 4
    assert len(created) == 1
    assert created[0] is cache.base
    assert torch._dynamo.config.cache_size_limit == before


def test_compile_backend_runs_distinct_shapes_with_default_cache() -> None:
    """Real compiled calls work without a library override of PyTorch limits."""
    if not deepwave._backend_utils.USE_OPENMP:
        pytest.skip("compile backend requires the OpenMP-enabled build")
    limit = torch._dynamo.config.cache_size_limit
    torch._dynamo.reset()
    try:
        for ndim, accuracy, dtype in (
            (1, 2, torch.float32),
            (2, 4, torch.float64),
        ):
            model = torch.full((20,) * ndim, 1500.0, dtype=dtype)
            source = torch.zeros(1, 1, 8, dtype=dtype)
            source[0, 0, 2] = 1.0
            locations = torch.full((1, 1, ndim), 10, dtype=torch.long)
            out = deepwave.scalar(
                model,
                5.0,
                0.0002,
                source_amplitudes=source,
                source_locations=locations,
                receiver_locations=locations,
                pml_freq=25.0,
                pml_width=3,
                accuracy=accuracy,
                backend="torch_compile",
            )
            assert torch.isfinite(out.receiver_amplitudes).all()
            assert out.info.backend == "torch_compile"
        assert torch._dynamo.config.cache_size_limit == limit
    finally:
        torch._dynamo.reset()


@pytest.mark.slow
def test_compile_backend_sweep_respects_pytorch_recompile_limit() -> None:
    """A long configuration sweep follows PyTorch defaults without failing."""
    if not deepwave._backend_utils.USE_OPENMP:
        pytest.skip("compile backend requires the OpenMP-enabled build")
    limit = torch._dynamo.config.cache_size_limit
    torch._dynamo.reset()
    cases = [
        (ndim, accuracy, torch.float32 if (ndim + accuracy) % 2 else torch.float64)
        for ndim in (1, 2, 3)
        for accuracy in (2, 4, 6, 8)
    ]
    try:
        for ndim, accuracy, dtype in cases:
            model = torch.full((16,) * ndim, 1500.0, dtype=dtype)
            source = torch.zeros(1, 1, 4, dtype=dtype)
            source[0, 0, 1] = 1.0
            locations = torch.full((1, 1, ndim), 8, dtype=torch.long)
            kwargs = {
                "source_amplitudes": source,
                "source_locations": locations,
                "receiver_locations": locations,
                "pml_freq": 25.0,
                "pml_width": 3,
                "accuracy": accuracy,
            }
            compiled = deepwave.scalar(
                model, 5.0, 0.0002, backend="torch_compile", **kwargs
            )
            eager = deepwave.scalar(model, 5.0, 0.0002, backend="eager", **kwargs)
            torch.testing.assert_close(
                compiled.receiver_amplitudes,
                eager.receiver_amplitudes,
                rtol=1e-5,
                atol=1e-7,
            )
            assert torch._dynamo.config.cache_size_limit == limit
    finally:
        torch._dynamo.reset()


# ---------------------------------------------------------------------------
# CUDA stream safety: the compiled backend's kernel configuration lives in
# ``__constant__`` device globals rewritten by a synchronous
# cudaMemcpyToSymbol before each call, which is not stream-safe. Running the
# compiled backend outside the default CUDA stream must raise rather than
# risk a race with a concurrent call on another stream silently corrupting
# the configuration.
# ---------------------------------------------------------------------------


def _tiny_scalar_cuda_kwargs(device: torch.device) -> dict:
    v = 1500 * torch.ones(10, 10, device=device)
    return {
        "v": v,
        "grid_spacing": 4.0,
        "dt": 0.001,
        "source_amplitudes": torch.randn(1, 1, 5, device=device),
        "source_locations": torch.tensor([[[0, 0]]]),
        "receiver_locations": torch.tensor([[[0, 5]]]),
        "pml_freq": 25.0,
    }


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_backend_raises_on_non_default_cuda_stream() -> None:
    """The compiled CUDA backend refuses to run outside the default stream."""
    device = torch.device("cuda:0")
    kwargs = _tiny_scalar_cuda_kwargs(device)
    v = kwargs.pop("v")
    stream = torch.cuda.Stream(device)
    with (
        torch.cuda.stream(stream),
        pytest.raises(
            RuntimeError,
            match=r"Deepwave's compiled CUDA backend requires the default CUDA stream",
        ),
    ):
        deepwave.scalar(v, **kwargs)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_backend_on_default_cuda_stream_still_works() -> None:
    """On the default CUDA stream, the compiled backend runs as before."""
    device = torch.device("cuda:0")
    kwargs = _tiny_scalar_cuda_kwargs(device)
    v = kwargs.pop("v")
    out = deepwave.scalar(v, **kwargs)
    assert torch.isfinite(out[-1]).all()
