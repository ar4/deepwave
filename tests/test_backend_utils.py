import builtins
import ctypes
import importlib
from unittest.mock import MagicMock, patch

import pytest
import torch

# Import backend_utils first, so we can patch its 'dll' attribute
# This assumes that the initial import of backend_utils will succeed
# without a mocked ctypes.CDLL, which might be an issue if libdeepwave_C.so
# is not found. We'll address this if it becomes a problem.
from deepwave import backend_utils


def test_get_argtypes_scalar_forward_float():
    template_name = "scalar_forward"
    float_type = ctypes.c_float
    argtypes = backend_utils._get_argtypes(template_name, float_type)

    # Check that FLOAT_TYPE placeholders are replaced with c_float
    assert all(
        t == ctypes.c_float
        or t == ctypes.c_void_p
        or t == ctypes.c_int64
        or t == ctypes.c_bool
        for t in argtypes
    )
    assert argtypes.count(ctypes.c_float) == 5  # Based on scalar_forward_template
    assert argtypes.count(ctypes.c_void_p) == 20
    assert argtypes.count(ctypes.c_int64) == 13
    assert argtypes.count(ctypes.c_bool) == 2


def test_get_argtypes_elastic_backward_double():
    template_name = "elastic_backward"
    float_type = ctypes.c_double
    argtypes = backend_utils._get_argtypes(template_name, float_type)

    # Check that FLOAT_TYPE placeholders are replaced with c_double
    assert all(
        t == ctypes.c_double
        or t == ctypes.c_void_p
        or t == ctypes.c_int64
        or t == ctypes.c_bool
        for t in argtypes
    )
    assert argtypes.count(ctypes.c_double) == 3  # Based on elastic_backward_template
    assert argtypes.count(ctypes.c_void_p) == 49
    assert argtypes.count(ctypes.c_int64) == 20
    assert argtypes.count(ctypes.c_bool) == 6


def test_get_backend_function_valid_call():
    with patch("deepwave.backend_utils.dll") as mock_dll:
        # Configure mock_dll to return a mock function for the expected call
        mock_func = MagicMock()
        mock_dll.attach_mock(mock_func, "scalar_iso_4_float_forward_cpu")

        func = backend_utils.get_backend_function(
            propagator="scalar",
            pass_name="forward",
            accuracy=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert func == mock_func


def test_get_backend_function_unsupported_dtype():
    with pytest.raises(TypeError, match="Unsupported dtype"):
        backend_utils.get_backend_function(
            propagator="scalar",
            pass_name="forward",
            accuracy=4,
            dtype=torch.int32,  # Unsupported dtype
            device=torch.device("cpu"),
        )


def test_get_backend_function_not_found():
    with patch("deepwave.backend_utils.dll", spec=ctypes.CDLL):
        # Now, the mock dll should behave like a ctypes.CDLL instance.
        # Accessing a non-existent attribute on it should raise AttributeError.
        with pytest.raises(AttributeError, match="Backend function .* not found."):
            backend_utils.get_backend_function(
                propagator="non_existent",
                pass_name="forward",
                accuracy=4,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )


def test_use_openmp_true():
    with patch("deepwave.backend_utils.dll") as mock_dll:
        # Explicitly set omp_get_num_threads to exist
        mock_dll.omp_get_num_threads = MagicMock()

        # Reload backend_utils to re-evaluate USE_OPENMP
        importlib.reload(backend_utils)
        assert backend_utils.USE_OPENMP is True


def test_use_openmp_false():
    original_hasattr = builtins.hasattr  # Store the original hasattr
    with patch("deepwave.backend_utils.dll"):
        with patch(
            "builtins.hasattr",
            side_effect=lambda obj, name: False
            if name == "omp_get_num_threads"
            else original_hasattr(obj, name),
        ):
            # Reload backend_utils to re-evaluate USE_OPENMP
            importlib.reload(backend_utils)
            assert backend_utils.USE_OPENMP is False


def test_initial_argtype_assignment():
    mock_dll = MagicMock()
    # Simulate the functions that _assign_argtypes would try to set argtypes on
    mock_func_scalar_forward_cpu = MagicMock()
    mock_func_scalar_backward_cuda = MagicMock()
    mock_func_elastic_forward_cpu = MagicMock()
    mock_func_scalar_born_backward_sc_cuda = MagicMock()

    # Configure mock_dll to return these specific mock functions
    mock_dll.scalar_iso_2_float_forward_cpu = mock_func_scalar_forward_cpu
    mock_dll.scalar_iso_8_double_backward_cuda = mock_func_scalar_backward_cuda
    mock_dll.elastic_iso_4_float_forward_cpu = mock_func_elastic_forward_cpu
    mock_dll.scalar_born_iso_6_double_backward_sc_cuda = (
        mock_func_scalar_born_backward_sc_cuda
    )

    # Temporarily replace backend_utils.dll with our mock for the duration of this test
    with patch("deepwave.backend_utils.dll", new=mock_dll):
        # Call _assign_argtypes directly for a few permutations
        # Note: _assign_argtypes is called internally during backend_utils import.
        # Here, we are calling it directly for unit testing purposes.
        backend_utils._assign_argtypes("scalar", 2, "float", "forward")
        backend_utils._assign_argtypes("scalar", 8, "double", "backward")
        backend_utils._assign_argtypes("elastic", 4, "float", "forward")
        backend_utils._assign_argtypes(
            "scalar_born",
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
