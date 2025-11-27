"""Tests for deepwave.backend_utils."""

import ctypes
import importlib
from unittest.mock import MagicMock, patch

import pytest
import torch

from deepwave import backend_utils


# Tests for _get_argtypes
def test_get_argtypes_scalar_forward_float():
    """Test _get_argtypes for scalar forward propagation with float dtype."""
    template_name = "scalar_forward"
    float_type = ctypes.c_float
    argtypes = backend_utils._get_argtypes(template_name, 2, float_type)  # noqa: SLF001

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
    assert argtypes.count(ctypes.c_void_p) == 23
    assert argtypes.count(ctypes.c_int64) == 14
    assert argtypes.count(ctypes.c_bool) == 3
    assert argtypes.count(ctypes.c_size_t) == 2


def test_get_argtypes_elastic_backward_double():
    """Test _get_argtypes for elastic backward propagation with double dtype."""
    template_name = "elastic_backward"
    float_type = ctypes.c_double
    argtypes = backend_utils._get_argtypes(template_name, 2, float_type)  # noqa: SLF001

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
    assert argtypes.count(ctypes.c_void_p) == 72
    assert argtypes.count(ctypes.c_int64) == 18
    assert argtypes.count(ctypes.c_bool) == 7
    assert argtypes.count(ctypes.c_size_t) == 2


# Tests for get_backend_function
def test_get_backend_function_valid_call():
    """Test get_backend_function with a valid call."""
    with patch("deepwave.backend_utils.dll") as mock_dll:
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
    with patch("deepwave.backend_utils.dll", spec=ctypes.CDLL), pytest.raises(
        AttributeError, match=r"Backend function .* not found."
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


# Tests for use_openmp
def test_use_openmp_true():
    """Test that USE_OPENMP is True when omp_get_num_threads is available."""
    with patch("deepwave.backend_utils.dll") as mock_dll:
        # Explicitly set omp_get_num_threads to exist
        mock_dll.omp_get_num_threads = MagicMock()

        # Reload backend_utils to re-evaluate USE_OPENMP
        importlib.reload(backend_utils)
        assert backend_utils.USE_OPENMP is True


# Tests for _assign_argtypes
def test_initial_argtype_assignment() -> None:
    """Test that argtypes are correctly assigned during initialization."""
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
    with patch("deepwave.backend_utils.dll", new=mock_dll):
        # Call _assign_argtypes directly for a few permutations
        # Note: _assign_argtypes is called internally during backend_utils import.
        # Here, we are calling it directly for unit testing purposes.
        backend_utils._assign_argtypes(  # noqa: SLF001
            "scalar", 2, 2, "float", "forward"
        )
        backend_utils._assign_argtypes(  # noqa: SLF001
            "scalar", 3, 8, "double", "backward"
        )
        backend_utils._assign_argtypes(  # noqa: SLF001
            "elastic", 2, 4, "float", "forward"
        )
        backend_utils._assign_argtypes(  # noqa: SLF001
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
