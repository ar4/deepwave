"""Backend utilities for Deepwave's C/CUDA interface.

This module handles the loading of the compiled C/CUDA shared library
and the dynamic assignment of argument types (argtypes) to the C functions
using ctypes. It ensures proper data marshalling between Python (PyTorch Tensors)
and the underlying C/CUDA implementations.
"""

import ctypes
import pathlib
import platform
from ctypes import c_bool, c_double, c_float, c_int64, c_void_p
from typing import Any, List, TypeAlias

import torch

CFunctionPointer: TypeAlias = Any

# Platform-specific shared library extension
SO_EXT = {"Linux": "so", "Darwin": "dylib", "Windows": "dll"}.get(platform.system())
if SO_EXT is None:
    raise RuntimeError("Unsupported OS or platform type")

dll = ctypes.CDLL(
    str(pathlib.Path(__file__).resolve().parent / f"libdeepwave_C.{SO_EXT}"),
)

# Check if was compiled with OpenMP support
USE_OPENMP = hasattr(dll, "omp_get_num_threads")

# Define ctypes argument type templates to reduce repetition while preserving order.
# A placeholder will be replaced by the appropriate float type (c_float or
# c_double).
FLOAT_TYPE: type = c_float

# Templates for argtype lists
scalar_forward_template: List[type] = (
    [c_void_p] * 20 + [FLOAT_TYPE] * 5 + [c_int64] * 7 + [c_bool] * 2 + [c_int64] * 6
)

scalar_backward_template: List[type] = (
    [c_void_p] * 24 + [FLOAT_TYPE] * 4 + [c_int64] * 7 + [c_bool] * 2 + [c_int64] * 6
)

scalar_born_forward_template: List[type] = (
    [c_void_p] * 33 + [FLOAT_TYPE] * 5 + [c_int64] * 8 + [c_bool] * 4 + [c_int64] * 6
)

scalar_born_backward_template: List[type] = (
    [c_void_p] * 41 + [FLOAT_TYPE] * 5 + [c_int64] * 9 + [c_bool] * 4 + [c_int64] * 6
)

scalar_born_backward_sc_template: List[type] = (
    [c_void_p] * 24 + [FLOAT_TYPE] * 5 + [c_int64] * 7 + [c_bool] * 3 + [c_int64] * 6
)

elastic_forward_template: List[type] = (
    [c_void_p] * 39 + [FLOAT_TYPE] * 3 + [c_int64] * 10 + [c_bool] * 6 + [c_int64] * 6
)

elastic_backward_template: List[type] = (
    [c_void_p] * 49 + [FLOAT_TYPE] * 3 + [c_int64] * 10 + [c_bool] * 6 + [c_int64] * 10
)

# A dictionary to hold all the templates
templates = {
    "scalar_forward": scalar_forward_template,
    "scalar_backward": scalar_backward_template,
    "scalar_born_forward": scalar_born_forward_template,
    "scalar_born_backward": scalar_born_backward_template,
    "scalar_born_backward_sc": scalar_born_backward_sc_template,
    "elastic_forward": elastic_forward_template,
    "elastic_backward": elastic_backward_template,
}


def _get_argtypes(template_name: str, float_type: type) -> List[type]:
    """Generates a concrete argtype list from a template and a float type.

    This function takes a template name (e.g., "scalar_forward") and a
    specific float type (e.g., `c_float` or `c_double`) and replaces
    the `FLOAT_TYPE` placeholders in the template with the provided
    `float_type`.

    Args:
        template_name: The name of the argtype template to use.
        float_type: The `ctypes` float type (`c_float` or `c_double`)
            to substitute into the template.

    Returns:
        List[type]: A list of `ctypes` types representing the argument
            signature for a C function.

    """
    return [float_type if t is FLOAT_TYPE else t for t in templates[template_name]]


def _assign_argtypes(
    propagator: str,
    accuracy: int,
    dtype: str,
    direction: str,
    extra: str = "",
) -> None:
    """Dynamically assigns ctypes argtypes to a given C function.

    This function constructs the full C function name based on the provided
    parameters (propagator, accuracy, dtype, direction, and extra suffix)
    and then assigns the corresponding `ctypes` argument types to it.
    It handles both CPU and CUDA versions of the functions.

    Args:
        propagator: The name of the propagator (e.g., "scalar", "elastic").
        accuracy: The finite-difference accuracy order (e.g., 2, 4, 6, 8).
        dtype: The data type as a string (e.g., "float", "double").
        direction: The direction of propagation (e.g., "forward", "backward").
        extra: An optional extra suffix for the function name (e.g., "_sc").

    Raises:
        AttributeError: If a function with the constructed name is not found
            in the loaded shared library (this is caught internally and skipped).

    """
    argtypes_name = f"{propagator}_{direction}{extra}_{dtype}_argtypes"
    argtypes = globals()[argtypes_name]

    for device in ["cpu", "cuda"]:
        func_name = f"{propagator}_iso_{accuracy}_{dtype}_{direction}{extra}_{device}"
        try:
            func = getattr(dll, func_name)
            func.argtypes = argtypes
            func.restype = None  # All C functions return void
        except AttributeError:
            continue


def get_backend_function(
    propagator: str,
    pass_name: str,
    accuracy: int,
    dtype: torch.dtype,
    device: torch.device,
    extra: str = "",
) -> CFunctionPointer:
    """Selects and returns the appropriate backend C/CUDA function.

    Args:
        propagator: The name of the propagator (e.g., "scalar", "elastic").
        pass_name: The name of the pass (e.g., "forward", "backward").
        accuracy: The finite-difference accuracy order.
        dtype: The torch.dtype of the tensors.
        device: The torch.device the tensors are on.
        extra: An optional extra suffix for the function name.

    Returns:
        The backend function pointer.

    Raises:
        AttributeError: If the function is not found in the shared library.
        TypeError: If the dtype is not torch.float32 or torch.float64.

    """
    if dtype == torch.float32:
        dtype_str = "float"
    elif dtype == torch.float64:
        dtype_str = "double"
    else:
        raise TypeError(f"Unsupported dtype {dtype}")

    device_str = device.type

    func_name = (
        f"{propagator}_iso_{accuracy}_{dtype_str}_{pass_name}{extra}_{device_str}"
    )

    try:
        return getattr(dll, func_name)
    except AttributeError as e:
        raise AttributeError(f"Backend function {func_name} not found.") from e


# Loop through all permutations and assign argtypes
# First, create specific argtype lists for each combination of template and dtype.
# This pre-generates the ctypes argument signatures for all functions,
# allowing _assign_argtypes to simply retrieve them by name.
for dtype_str, dtype_c in [("float", c_float), ("double", c_double)]:
    for key in templates:
        globals()[f"{key}_{dtype_str}_argtypes"] = _get_argtypes(key, dtype_c)

# Now, iterate through all defined accuracies and data types to assign the
# correct argtypes to the dynamically loaded C/CUDA functions.
# The function names are constructed based on the naming convention
# used in the CMake build system (e.g., scalar_iso_4_float_forward_cpu).
for current_accuracy in [2, 4, 6, 8]:
    for current_dtype in ["float", "double"]:
        _assign_argtypes("scalar", current_accuracy, current_dtype, "forward")
        _assign_argtypes("scalar", current_accuracy, current_dtype, "backward")
        _assign_argtypes("scalar_born", current_accuracy, current_dtype, "forward")
        _assign_argtypes("scalar_born", current_accuracy, current_dtype, "backward")
        _assign_argtypes(
            "scalar_born",
            current_accuracy,
            current_dtype,
            "backward",
            extra="_sc",
        )

# Elastic propagators currently only support 2nd and 4th order accuracy.
for current_accuracy in [2, 4]:
    for current_dtype in ["float", "double"]:
        _assign_argtypes("elastic", current_accuracy, current_dtype, "forward")
        _assign_argtypes("elastic", current_accuracy, current_dtype, "backward")
