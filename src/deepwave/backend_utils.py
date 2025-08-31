import platform
import ctypes
from ctypes import c_void_p, c_int64, c_float, c_double, c_bool
import pathlib
from typing import List

# Platform-specific shared library extension
SO_EXT = {
    "Linux": "so",
    "Darwin": "dylib",
    "Windows": "dll"
}.get(platform.system())
if SO_EXT is None:
    raise RuntimeError("Unsupported OS or platform type")

dll = ctypes.CDLL(
    str(pathlib.Path(__file__).resolve().parent / f"libdeepwave_C.{SO_EXT}"))

# Check if was compiled with OpenMP support
USE_OPENMP = hasattr(dll, "omp_get_num_threads")

# Define ctypes argument type templates to reduce repetition while preserving order.
# A placeholder will be replaced by the appropriate float type (c_float or
# c_double).
FLOAT_TYPE: type = c_float

# Templates for argtype lists
scalar_forward_template: List[type] = ([c_void_p] * 20 + [FLOAT_TYPE] * 5 +
                                       [c_int64] * 7 + [c_bool] * 2 +
                                       [c_int64] * 6)

scalar_backward_template: List[type] = ([c_void_p] * 24 + [FLOAT_TYPE] * 4 +
                                        [c_int64] * 7 + [c_bool] * 2 +
                                        [c_int64] * 6)

scalar_born_forward_template: List[type] = ([c_void_p] * 33 +
                                            [FLOAT_TYPE] * 5 + [c_int64] * 8 +
                                            [c_bool] * 4 + [c_int64] * 6)

scalar_born_backward_template: List[type] = ([c_void_p] * 41 +
                                             [FLOAT_TYPE] * 5 + [c_int64] * 9 +
                                             [c_bool] * 4 + [c_int64] * 6)

scalar_born_backward_sc_template: List[type] = ([c_void_p] * 24 +
                                                [FLOAT_TYPE] * 5 +
                                                [c_int64] * 7 + [c_bool] * 3 +
                                                [c_int64] * 6)

elastic_forward_template: List[type] = ([c_void_p] * 39 + [FLOAT_TYPE] * 3 +
                                        [c_int64] * 10 + [c_bool] * 6 +
                                        [c_int64] * 6)

elastic_backward_template: List[type] = ([c_void_p] * 49 + [FLOAT_TYPE] * 3 +
                                         [c_int64] * 10 + [c_bool] * 6 +
                                         [c_int64] * 10)

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
    """Generate a concrete argtype list from a template and float type."""
    return [
        float_type if t is FLOAT_TYPE else t for t in templates[template_name]
    ]


def _assign_argtypes(propagator: str,
                     accuracy: int,
                     dtype: str,
                     direction: str,
                     extra: str = "") -> None:
    """Dynamically assign ctypes argtypes to a given function."""
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


# Loop through all permutations and assign argtypes
for dtype_str, dtype_c in [("float", c_float), ("double", c_double)]:
    for key in templates:
        globals()[f"{key}_{dtype_str}_argtypes"] = _get_argtypes(key, dtype_c)

for current_accuracy in [2, 4, 6, 8]:
    for current_dtype in ["float", "double"]:
        _assign_argtypes("scalar", current_accuracy, current_dtype, "forward")
        _assign_argtypes("scalar", current_accuracy, current_dtype, "backward")
        _assign_argtypes("scalar_born", current_accuracy, current_dtype,
                         "forward")
        _assign_argtypes("scalar_born", current_accuracy, current_dtype,
                         "backward")
        _assign_argtypes("scalar_born",
                         current_accuracy,
                         current_dtype,
                         "backward",
                         extra="_sc")

for current_accuracy in [2, 4]:
    for current_dtype in ["float", "double"]:
        _assign_argtypes("elastic", current_accuracy, current_dtype, "forward")
        _assign_argtypes("elastic", current_accuracy, current_dtype,
                         "backward")
