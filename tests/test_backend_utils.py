"""Tests for deepwave._backend_utils."""

import ctypes
import importlib
import inspect
import platform
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from deepwave import _backend_utils as backend_utils
from deepwave import acoustic, acoustic_vti, scalar
from deepwave.wavelets import ricker


# Tests for _get_argtypes
def test_get_argtypes_scalar_forward_float():
    """Test _get_argtypes for scalar forward propagation with float dtype."""
    template_name = "scalar_forward"
    float_type = ctypes.c_float
    argtypes = backend_utils._get_argtypes(template_name, 2, float_type)

    # Check that FLOAT_TYPE placeholders are replaced with c_float
    assert all(
        t == ctypes.c_float
        or t == ctypes.c_void_p
        or t == ctypes.c_int64
        or t == ctypes.c_bool
        or t == ctypes.c_size_t
        for t in argtypes
    )
    assert argtypes.count(ctypes.c_float) == 5  # Based on scalar_forward_template
    # 30, not an earlier count of 25: the compiled scalar forward gained a
    # second, opt-in raw-forward checkpoint buffer for CallbackState
    # role="forward" support (four pointers -- raw_store_1a/1b/2/3 -- plus
    # one raw_filenames_ptr). Cross-checked against the authoritative
    # tests/abi_reference.json entry for scalar|forward|2|float, which
    # test_abi_stability.py/test_abi_drift.py verify continuously.
    assert argtypes.count(ctypes.c_void_p) == 30
    # 19, not an earlier count of 18: the raw-forward checkpoint buffer above
    # gained its own storage_mode int64 (independent of the primary
    # checkpoint's), threaded so the compiled backend can honour
    # storage_mode="cpu"/"disk" for it rather than always using a
    # DEVICE-resident buffer. Cross-checked against the authoritative
    # tests/abi_reference.json entry for scalar|forward|2|float, which
    # test_abi_stability.py/test_abi_drift.py verify continuously.
    assert argtypes.count(ctypes.c_int64) == 19
    # 5, not an earlier count of 4: the raw-forward checkpoint's own
    # storage_compression flag, accompanying its storage_mode above.
    assert argtypes.count(ctypes.c_bool) == 5
    # 4, not an earlier count of 2: the raw-forward checkpoint's own
    # shot_bytes_uncomp/shot_bytes_comp byte counts, accompanying its
    # storage_mode above.
    assert argtypes.count(ctypes.c_size_t) == 4


def test_get_argtypes_elastic_backward_double():
    """Test _get_argtypes for elastic backward propagation with double dtype."""
    template_name = "elastic_backward"
    float_type = ctypes.c_double
    argtypes = backend_utils._get_argtypes(template_name, 2, float_type)

    # Check that FLOAT_TYPE placeholders are replaced with c_double
    assert all(
        t == ctypes.c_double
        or t == ctypes.c_void_p
        or t == ctypes.c_int64
        or t == ctypes.c_bool
        or t == ctypes.c_size_t
        for t in argtypes
    )
    assert argtypes.count(ctypes.c_double) == 3  # Based on elastic_backward_template
    assert argtypes.count(ctypes.c_void_p) == 78
    # 22, not the ABI's earlier count of 18: additive int64 args (the
    # per-face `bc` boundary codes from the elastic first-class boundary
    # faces, commit 8d4e292) were threaded through the elastic
    # backward template since this literal was last updated. Cross-checked
    # against the authoritative tests/abi_reference.json entry for
    # elastic|backward|2|double, which test_abi_stability.py verifies
    # continuously.
    assert argtypes.count(ctypes.c_int64) == 22
    assert argtypes.count(ctypes.c_bool) == 7
    assert argtypes.count(ctypes.c_size_t) == 2


# Tests for get_backend_function
def test_get_backend_function_valid_call():
    """Test get_backend_function with a valid call."""
    with patch("deepwave._backend_utils.dll") as mock_dll:
        # Configure mock_dll to return a mock function for the expected call
        mock_func = MagicMock()
        mock_dll.scalar_iso_2d_4_float_forward_cpu = mock_func

        func = backend_utils.get_backend_function(
            propagator="scalar",
            ndim=2,
            pass_name="forward",  # noqa: S106
            accuracy=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert func == mock_func


def test_get_backend_function_unsupported_dtype():
    """Test get_backend_function with an unsupported dtype."""
    with pytest.raises(TypeError, match="Unsupported dtype"):
        backend_utils.get_backend_function(
            propagator="scalar",
            ndim=2,
            pass_name="forward",  # noqa: S106
            accuracy=4,
            dtype=torch.int32,  # Unsupported dtype
            device=torch.device("cpu"),
        )


def test_get_backend_function_not_found():
    """Test get_backend_function when the backend function is not found."""
    with (
        patch("deepwave._backend_utils.dll", spec=ctypes.CDLL),
        pytest.raises(AttributeError, match=r"Backend function .* not found."),
    ):
        # Now, the mock dll should behave like a ctypes.CDLL instance.
        # Accessing a non-existent attribute on it should raise AttributeError.
        backend_utils.get_backend_function(
            propagator="non_existent",
            ndim=2,
            pass_name="forward",  # noqa: S106
            accuracy=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


# Tests for the CPU capability probe (x86 SIGILL guard + diagnostics)
def test_cpu_features_returns_expected_keys():
    """cpu_features() reports the known feature names plus metadata keys."""
    feats = backend_utils.cpu_features()
    for name in backend_utils._CPU_FEATURE_NAMES:
        assert name in feats
        assert feats[name] in (True, False, None)
    assert "required_baseline" in feats
    assert feats["required_baseline"] is None or isinstance(
        feats["required_baseline"], str
    )
    assert "machine" in feats


@pytest.mark.skipif(sys.platform != "win32", reason="MSVC CPU probe is Windows-only")
def test_windows_x86_cpu_probe_reports_avx2_support():
    """The wheel can decide whether AVX2 kernels are safe before calling them."""
    if platform.machine().lower() not in ("amd64", "x86_64", "x86"):
        pytest.skip("x86 CPU probe is not applicable on this machine")
    assert backend_utils._cpu_supports("avx2") in (0, 1)


def test_check_cpu_baseline_does_not_raise_on_this_machine():
    """The running interpreter must be able to load and run its own backend."""
    # If this raised, the library we just imported could not execute here.
    backend_utils._check_cpu_baseline()


def test_cpu_supports_missing_symbol_returns_none():
    """When the library predates the probe helper, the query degrades to None."""
    mock_dll = MagicMock(spec=ctypes.CDLL)  # spec => missing attrs raise
    with patch("deepwave._backend_utils.dll", new=mock_dll):
        assert backend_utils._cpu_supports("avx2") is None
        assert backend_utils._required_baseline() is None
        # A library without the helpers must not raise (no false SIGILL guard).
        backend_utils._check_cpu_baseline()


def test_check_cpu_baseline_raises_when_cpu_lacks_required_isa():
    """A fixed-ISA binary on an unsupported CPU raises a clear, searchable error."""
    mock_dll = MagicMock()
    baseline_fn = MagicMock()
    baseline_fn.return_value = b"avx2"
    supports_fn = MagicMock()
    supports_fn.return_value = 0  # CPU does NOT support the required ISA
    mock_dll.deepwave_cpu_required_baseline = baseline_fn
    mock_dll.deepwave_cpu_supports = supports_fn

    with (
        patch("deepwave._backend_utils.dll", new=mock_dll),
        pytest.raises(RuntimeError, match="Illegal instruction") as excinfo,
    ):
        backend_utils._check_cpu_baseline()
    # The escape hatch must be discoverable from the message.
    assert "--no-binary deepwave" in str(excinfo.value)


def test_check_cpu_baseline_ok_when_cpu_supports_required_isa():
    """A fixed-ISA binary on a supporting CPU passes the guard silently."""
    mock_dll = MagicMock()
    mock_dll.deepwave_cpu_required_baseline = MagicMock(return_value=b"avx2")
    mock_dll.deepwave_cpu_supports = MagicMock(return_value=1)
    with patch("deepwave._backend_utils.dll", new=mock_dll):
        backend_utils._check_cpu_baseline()


def test_check_cpu_baseline_noop_when_no_fixed_baseline():
    """The runtime-dispatch build reports no fixed baseline and never raises."""
    mock_dll = MagicMock()
    mock_dll.deepwave_cpu_required_baseline = MagicMock(return_value=b"")
    # _cpu_supports must not even be consulted; make it explode if it is.
    mock_dll.deepwave_cpu_supports = MagicMock(side_effect=AssertionError)
    with patch("deepwave._backend_utils.dll", new=mock_dll):
        backend_utils._check_cpu_baseline()


# Tests for use_openmp
def test_use_openmp_true():
    """Test that USE_OPENMP is True when omp_get_num_threads is available."""
    with patch("deepwave._backend_utils.dll") as mock_dll:
        # Explicitly set omp_get_num_threads to exist
        mock_dll.omp_get_num_threads = MagicMock()

        # Reload backend_utils to re-evaluate USE_OPENMP
        importlib.reload(backend_utils)
        assert backend_utils.USE_OPENMP is True


# Tests for _assign_argtypes
def test_initial_argtype_assignment() -> None:
    """Test that argtypes are correctly assigned during initialisation."""
    mock_dll = MagicMock()
    # Simulate the functions that _assign_argtypes would try to set argtypes on
    mock_func_scalar_forward_cpu = MagicMock()
    mock_func_scalar_backward_cuda = MagicMock()
    mock_func_elastic_forward_cpu = MagicMock()
    mock_func_scalar_born_backward_sc_cuda = MagicMock()

    # Configure mock_dll to return these specific mock functions
    mock_dll.scalar_iso_2d_2_float_forward_cpu = mock_func_scalar_forward_cpu
    mock_dll.scalar_iso_3d_8_double_backward_cuda = mock_func_scalar_backward_cuda
    mock_dll.elastic_iso_2d_4_float_forward_cpu = mock_func_elastic_forward_cpu
    mock_dll.scalar_born_iso_1d_6_double_backward_sc_cuda = (
        mock_func_scalar_born_backward_sc_cuda
    )

    # Temporarily replace backend_utils.dll with our mock for the duration of this test
    with patch("deepwave._backend_utils.dll", new=mock_dll):
        # Call _assign_argtypes directly for a few permutations
        # Note: _assign_argtypes is called internally during backend_utils import.
        # Here, we are calling it directly for unit testing purposes.
        backend_utils._assign_argtypes("scalar", 2, 2, "float", "forward")
        backend_utils._assign_argtypes("scalar", 3, 8, "double", "backward")
        backend_utils._assign_argtypes("elastic", 2, 4, "float", "forward")
        backend_utils._assign_argtypes(
            "scalar_born",
            1,
            6,
            "double",
            "backward",
            extra="_sc",
        )

    # Assert that argtypes were set correctly on the mock functions
    assert mock_func_scalar_forward_cpu.argtypes is not None
    assert isinstance(mock_func_scalar_forward_cpu.argtypes, list)
    assert len(mock_func_scalar_forward_cpu.argtypes) > 0

    assert mock_func_scalar_backward_cuda.argtypes is not None
    assert isinstance(mock_func_scalar_backward_cuda.argtypes, list)
    assert len(mock_func_scalar_backward_cuda.argtypes) > 0

    assert mock_func_elastic_forward_cpu.argtypes is not None
    assert isinstance(mock_func_elastic_forward_cpu.argtypes, list)
    assert len(mock_func_elastic_forward_cpu.argtypes) > 0

    assert mock_func_scalar_born_backward_sc_cuda.argtypes is not None
    assert isinstance(mock_func_scalar_born_backward_sc_cuda.argtypes, list)
    assert len(mock_func_scalar_born_backward_sc_cuda.argtypes) > 0


# --- Tests for PropagatorSpec.leading_args / leading_arg_offset -------------
#
# Every propagator with a compiled backend used to have its ``*_func``
# dispatcher reshuffle a flat positional ``args`` tuple by hardcoded integer
# literal (e.g. ``args[7]`` for "boundary", ``args[15]`` for the first model
# tensor). Those literals are now derived at import time from
# ``PROPAGATOR_REGISTRY[propagator].leading_args`` via ``leading_arg_offset``/
# ``len(leading_args)`` (see e.g. ``scalar.py``'s ``_BOUNDARY_IDX``/
# ``_N_LEADING_ARGS``). The tests below do not just re-assert what the
# dispatcher itself computes from the registry (that would be tautological);
# they cross-check ``leading_args`` against two things the registry does not
# feed: the compiled ``ForwardFunc.forward`` classes' own signatures (an
# independent, hand-written source of truth), and, for a representative
# sample of propagators, the actual runtime behaviour of a real compiled-vs-
# eager forward call with a distinctive non-PML face.

# Maps a propagator name to the module/class holding its compiled forward
# autograd Function (``import_module`` -- not the ``from deepwave import
# scalar`` form used elsewhere in this file, which resolves to the public
# *function* re-exported under the same name -- is needed to reach the
# *module* and its ``XxxForwardFunc`` class).
_FORWARD_FUNC_CLASSES = {
    "scalar": ("deepwave.scalar", "ScalarForwardFunc"),
    "scalar_born": ("deepwave.scalar_born", "ScalarBornForwardFunc"),
    "acoustic": ("deepwave.acoustic", "AcousticForwardFunc"),
    "acoustic_vti": ("deepwave.acoustic_vti", "AcousticVtiForwardFunc"),
    "elastic": ("deepwave.elastic", "ElasticForwardFunc"),
    "elastic_vti": ("deepwave.elastic_vti", "ElasticVtiForwardFunc"),
    "viscoacoustic": ("deepwave.viscoacoustic", "ViscoacousticForwardFunc"),
    "acoustic_born": ("deepwave.acoustic_born", "AcousticBornForwardFunc"),
    "elastic_born": ("deepwave.elastic_born", "ElasticBornForwardFunc"),
}

# ``ForwardFunc.forward`` spells the storage-mode leading arg
# "storage_mode_str"; ``leading_args`` spells it "storage_mode" (matching the
# name at each propagator's own public call site). Purely a naming
# difference, normalised away below.
_FORWARD_FUNC_PARAM_ALIASES = {"storage_mode_str": "storage_mode"}


@pytest.mark.parametrize("propagator", sorted(_FORWARD_FUNC_CLASSES))
def test_leading_args_matches_forward_func_signature(propagator):
    """``leading_args`` must match on ``XxxForwardFunc.forward``'s real signature.

    Every propagator's ``*_func`` dispatcher moves ``boundary`` to the end of
    the leading-arg block (a no-op for the propagators whose ``leading_args``
    already end in ``boundary``) before calling the compiled ``ForwardFunc``;
    this reconstructs the expected post-reshuffle order purely from
    ``leading_args`` and checks it against the real class signature.

    This catches a leading argument being added, removed, or renamed in
    ``PROPAGATOR_REGISTRY`` without a matching change to the propagator's
    real call site/``ForwardFunc.forward`` signature -- the exact "someone
    edited the leading-args order but forgot to update the registry" failure
    mode this declarative scheme exists to prevent.
    """
    spec = backend_utils.PROPAGATOR_REGISTRY[propagator]
    leading = spec.leading_args
    assert leading, f"{propagator} has no leading_args"
    assert len(set(leading)) == len(leading), (
        f"{propagator}'s leading_args has duplicate names: {leading}"
    )

    module_name, class_name = _FORWARD_FUNC_CLASSES[propagator]
    forward_func_cls = getattr(importlib.import_module(module_name), class_name)
    real_params = [
        _FORWARD_FUNC_PARAM_ALIASES.get(name, name)
        for name in inspect.signature(forward_func_cls.forward).parameters
    ]

    expected = [name for name in leading if name != "boundary"] + ["boundary"]

    # real_params has trailing dispatcher-specific extras not tracked by
    # leading_args (elastic's "pm_c_batched", viscoacoustic's
    # "n_mechanisms") and the final tensor catch-all ("args"); keep only the
    # names `expected` names, preserving the real signature's order.
    real_leading = [name for name in real_params if name in expected]
    assert real_leading == expected, (
        f"{propagator}: PROPAGATOR_REGISTRY['{propagator}'].leading_args "
        f"does not match {class_name}.forward's real leading parameters "
        f"(after accounting for the boundary reshuffle) -- "
        f"got {real_leading}, expected {expected}"
    )


def test_scalar_leading_args_boundary_offset_matches_real_face():
    """Value-based check that scalar's registry-derived boundary offset is real.

    ``scalar.py``'s ``_BOUNDARY_IDX``/``_N_LEADING_ARGS`` (derived from
    ``PROPAGATOR_REGISTRY["scalar"].leading_args``) drive the compiled-backend
    reshuffle in ``scalar_func``. This does not just check structure: it runs
    a real compiled forward call with a distinctive per-face boundary
    (``["pml", "free", "rigid", "pml"]``, mirroring
    ``test_scalar_c_matches_eager_grad`` in ``test_boundary.py``)
    and checks it against the eager reference. If ``_BOUNDARY_IDX`` picked
    the wrong tensor out of ``args`` (e.g. because ``leading_args`` silently
    drifted out of sync with scalar()'s real call site), the compiled call
    would either raise (wrong argument type at that position) or produce
    physics that disagree with the eager backend -- this test would fail
    either way, not just a structural comparison.
    """
    torch.manual_seed(0)
    shape = (18, 18)
    dx, dt, nt = 5.0, 0.0007, 60
    freq = 15.0
    sa = ricker(freq, nt, dt, 1.4 / freq).reshape(1, 1, nt)
    v = torch.ones(shape) * 1500.0
    source_locations = torch.tensor([[[13, 4]]])
    receiver_locations = torch.tensor([[[15, 3]]])

    def run(backend):
        return scalar(
            v,
            dx,
            dt,
            source_amplitudes=sa,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_width=[6, 0, 0, 6],
            boundary=["pml", "free", "rigid", "pml"],
            accuracy=4,
            pml_freq=freq,
            backend=backend,
        )[-1]

    out_c = run("compiled")
    out_e = run("eager")
    assert torch.allclose(out_c, out_e, atol=1e-4, rtol=1e-3)


def test_acoustic_leading_args_boundary_offset_matches_real_face():
    """Acoustic sibling of the scalar boundary-offset smoke check above.

    Exercises the same boundary-move reshuffle as the scalar check, with a
    real compiled-vs-eager forward call for a staggered-grid propagator.
    """
    torch.manual_seed(0)
    shape = (14, 14)
    dx, dt, nt = 5.0, 0.0007, 60
    freq = 15.0
    sa = ricker(freq, nt, dt, 1.4 / freq).reshape(1, 1, nt)
    v = torch.ones(shape) * 1500.0
    rho = torch.ones(shape) * 1000.0
    source_locations = torch.tensor([[[11, 4]]])
    receiver_locations = torch.tensor([[[13, 3]]])

    def run(backend):
        return acoustic(
            v,
            rho,
            dx,
            dt,
            source_amplitudes_volume_rate=sa,
            source_locations_volume_rate=source_locations,
            receiver_locations_p=receiver_locations,
            pml_width=[6, 0, 0, 6],
            boundary=["pml", "rigid", "free", "pml"],
            accuracy=4,
            pml_freq=freq,
            backend=backend,
        )[-2]

    out_c = run("compiled")
    out_e = run("eager")
    assert torch.allclose(out_c, out_e, atol=1e-4, rtol=1e-3)


def test_acoustic_vti_leading_args_model_offset_matches_real_face():
    """acoustic_vti sibling of the boundary-offset smoke checks above.

    Unlike scalar/acoustic, acoustic_vti's call site already places
    ``boundary`` where the compiled backend wants it (see the
    ``PROPAGATOR_REGISTRY["acoustic_vti"]`` comment in ``_backend_utils.py``),
    so ``acoustic_vti_func`` has only a registry-derived model-tensor offset
    (``_N_LEADING_ARGS``, no ``_BOUNDARY_IDX``) to check -- this exercises
    that offset with a real compiled-vs-eager forward call.
    """
    torch.manual_seed(0)
    shape = (14, 14)
    dx, dt, nt = 5.0, 0.0001, 30
    freq = 25.0
    c = 1500.0
    rho = 1000.0
    sa = ricker(freq, nt, dt, 0.02).reshape(1, 1, nt)
    c33 = torch.ones(shape) * c**2 * rho
    c11 = c33.clone()
    c13 = c33.clone()
    buoyancy = torch.ones(shape) / rho
    source_locations = torch.tensor([[[11, 4]]])
    receiver_locations = torch.tensor([[[13, 3]]])

    def run(backend):
        out = acoustic_vti(
            c11,
            c13,
            c33,
            buoyancy,
            dx,
            dt,
            source_amplitudes_sigma_v=sa,
            source_amplitudes_sigma_h=sa,
            source_locations_sigma_v=source_locations,
            source_locations_sigma_h=source_locations,
            receiver_locations_sigma_v=receiver_locations,
            receiver_locations_sigma_h=receiver_locations,
            pml_width=[6, 0, 0, 6],
            boundary=["pml", "rigid", "rigid", "pml"],
            accuracy=4,
            pml_freq=freq,
            backend=backend,
        )
        return out[-2] + out[-1]

    out_c = run("compiled")
    out_e = run("eager")
    assert torch.allclose(out_c, out_e, atol=1e-4, rtol=1e-3)
