"""Wave propagation modules for PyTorch.

Provides forward modelling and backpropagation of the scalar wave
equation (regular and Born), and elastic wave equation.
It can thus be used to generate synthetic data, and to perform
FWI and LSRTM.

Module and functional interfaces are provided, and allow the propagators
to be easily integrated into a larger computational graph, if desired.

For better computational performance, the code is written in C++ and
CUDA.
"""

import platform
import ctypes
from ctypes import c_void_p, c_int64, c_float, c_double, c_bool
import pathlib
from typing import List, Type, Any

# These imports are for exposing the public API
from deepwave.scalar import Scalar, scalar
from deepwave.scalar_born import ScalarBorn, scalar_born
from deepwave.elastic import Elastic, elastic
import deepwave.wavelets
import deepwave.location_interpolation
from ._version import __version__

# Platform-specific shared library extension
SO_EXT = {"Linux": "so", "Darwin": "dylib", "Windows": "dll"}.get(
    platform.system()
)
if SO_EXT is None:
    raise RuntimeError("Unsupported OS or platform type")

dll = ctypes.CDLL(
    str(pathlib.Path(__file__).resolve().parent / f"libdeepwave_C.{SO_EXT}")
)

# Check if was compiled with OpenMP support
try:
    dll.omp_get_num_threads
    use_openmp = True
except AttributeError:
    use_openmp = False

# Define ctypes argument type templates to reduce repetition while preserving order.
# A placeholder will be replaced by the appropriate float type (c_float or c_double).
FLOAT_TYPE = Any

# Templates for argtype lists
scalar_forward_template: List[Type] = (
    [c_void_p] * 20
    + [FLOAT_TYPE] * 5
    + [c_int64] * 7
    + [c_bool] * 2
    + [c_int64] * 6
)

scalar_backward_template: List[Type] = (
    [c_void_p] * 24
    + [FLOAT_TYPE] * 4
    + [c_int64] * 7
    + [c_bool] * 2
    + [c_int64] * 6
)

scalar_born_forward_template: List[Type] = (
    [c_void_p] * 33
    + [FLOAT_TYPE] * 5
    + [c_int64] * 8
    + [c_bool] * 4
    + [c_int64] * 6
)

scalar_born_backward_template: List[Type] = (
    [c_void_p] * 41
    + [FLOAT_TYPE] * 5
    + [c_int64] * 9
    + [c_bool] * 4
    + [c_int64] * 6
)

scalar_born_backward_sc_template: List[Type] = (
    [c_void_p] * 24
    + [FLOAT_TYPE] * 5
    + [c_int64] * 7
    + [c_bool] * 3
    + [c_int64] * 6
)

elastic_forward_template: List[Type] = (
    [c_void_p] * 39
    + [FLOAT_TYPE] * 3
    + [c_int64] * 10
    + [c_bool] * 6
    + [c_int64] * 6
)

elastic_backward_template: List[Type] = (
    [c_void_p] * 49
    + [FLOAT_TYPE] * 3
    + [c_int64] * 10
    + [c_bool] * 6
    + [c_int64] * 10
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


def _get_argtypes(template_name: str, float_type: Type) -> List[Type]:
    """Generate a concrete argtype list from a template and float type."""
    return [
        float_type if t is FLOAT_TYPE else t for t in templates[template_name]
    ]


# Generate all the specific argtype lists
for dtype_str, dtype_c in [("float", c_float), ("double", c_double)]:
    for key in templates:
        globals()[f"{key}_{dtype_str}_argtypes"] = _get_argtypes(key, dtype_c)


def _assign_argtypes(
    propagator: str, accuracy: int, dtype: str, direction: str, extra: str = ""
) -> None:
    """Dynamically assign ctypes argtypes to a given function."""
    argtypes_name = f"{propagator}_{direction}{extra}_{dtype}_argtypes"
    argtypes = globals()[argtypes_name]

    for device in ["cpu", "cuda"]:
        func_name = (
            f"{propagator}_iso_{accuracy}_{dtype}_{direction}{extra}_{device}"
        )
        try:
            func = getattr(dll, func_name)
            func.argtypes = argtypes
            func.restype = None  # All C functions return void
        except AttributeError:
            continue


# Loop through all permutations and assign argtypes
for accuracy in [2, 4, 6, 8]:
    for dtype in ["float", "double"]:
        _assign_argtypes("scalar", accuracy, dtype, "forward")
        _assign_argtypes("scalar", accuracy, dtype, "backward")
        _assign_argtypes("scalar_born", accuracy, dtype, "forward")
        _assign_argtypes("scalar_born", accuracy, dtype, "backward")
        _assign_argtypes("scalar_born", accuracy, dtype, "backward", extra="_sc")

for accuracy in [2, 4]:
    for dtype in ["float", "double"]:
        _assign_argtypes("elastic", accuracy, dtype, "forward")
        _assign_argtypes("elastic", accuracy, dtype, "backward")
