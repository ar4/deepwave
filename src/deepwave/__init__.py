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
from deepwave.scalar import Scalar, scalar
from deepwave.scalar_born import ScalarBorn, scalar_born
from deepwave.elastic import Elastic, elastic
import deepwave.wavelets
import deepwave.location_interpolation
from ._version import __version__

if platform.system() == 'Linux':
    so_ext = "so"
elif platform.system() == 'Darwin':
    so_ext = "dylib"
elif platform.system() == 'Windows':
    so_ext = "dll"
else:
    raise RuntimeError("Unsupported OS or platform type")

dll = ctypes.CDLL(
    str(pathlib.Path(__file__).resolve().parent / ("libdeepwave_C." + so_ext)))

# Check if was compiled with OpenMP support
try:
    dll.omp_get_num_threads
    use_openmp = True
except AttributeError:
    use_openmp = False

scalar_iso_float_forward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_float,
    c_float, c_float, c_float, c_float, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_bool, c_bool, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64
]
scalar_iso_double_forward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_double,
    c_double, c_double, c_double, c_double, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_bool, c_bool, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64
]
scalar_iso_float_backward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_float, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_bool, c_bool,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_int64
]
scalar_iso_double_backward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_double, c_double, c_double, c_double,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_bool,
    c_bool, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64
]
scalar_born_iso_float_forward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_float,
    c_float, c_float, c_float, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_int64
]
scalar_born_iso_double_forward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_double, c_double,
    c_double, c_double, c_double, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_int64
]
scalar_born_iso_float_backward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_float,
    c_float, c_float, c_float, c_float, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool,
    c_bool, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64
]
scalar_born_iso_double_backward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_double,
    c_double, c_double, c_double, c_double, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool,
    c_bool, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64
]
scalar_born_iso_double_backward_sc_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_double, c_double, c_double, c_double,
    c_double, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_bool, c_bool, c_bool, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_int64
]
scalar_born_iso_float_backward_sc_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_float, c_float,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_bool,
    c_bool, c_bool, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64
]
elastic_iso_float_forward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_bool, c_bool, c_bool, c_bool, c_bool, c_bool, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_int64
]
elastic_iso_double_forward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_double, c_double, c_double,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_bool, c_bool, c_bool, c_bool, c_bool, c_bool, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_int64
]
elastic_iso_float_backward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_float, c_float, c_float, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool,
    c_bool, c_bool, c_bool, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_int64
]
elastic_iso_double_backward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_double, c_double, c_double, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool,
    c_bool, c_bool, c_bool, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_int64
]

for accuracy in [2, 4, 6, 8]:
    for dtype in ["float", "double"]:
        getattr(dll, f"scalar_iso_{accuracy}_{dtype}_forward_cpu").argtypes = \
            globals()[f"scalar_iso_{dtype}_forward_argtypes"]
        getattr(dll, f"scalar_iso_{accuracy}_{dtype}_backward_cpu").argtypes = \
            globals()[f"scalar_iso_{dtype}_backward_argtypes"]
        getattr(dll, f"scalar_born_iso_{accuracy}_{dtype}_forward_cpu").argtypes = \
            globals()[f"scalar_born_iso_{dtype}_forward_argtypes"]
        getattr(dll, f"scalar_born_iso_{accuracy}_{dtype}_backward_cpu").argtypes = \
            globals()[f"scalar_born_iso_{dtype}_backward_argtypes"]
        getattr(dll, f"scalar_born_iso_{accuracy}_{dtype}_backward_sc_cpu").argtypes = \
            globals()[f"scalar_born_iso_{dtype}_backward_sc_argtypes"]
        try:
            getattr(dll, f"scalar_iso_{accuracy}_{dtype}_forward_cuda").argtypes = \
                globals()[f"scalar_iso_{dtype}_forward_argtypes"]
            getattr(dll, f"scalar_iso_{accuracy}_{dtype}_backward_cuda").argtypes = \
                globals()[f"scalar_iso_{dtype}_backward_argtypes"]
            getattr(dll, f"scalar_born_iso_{accuracy}_{dtype}_forward_cuda").argtypes = \
                globals()[f"scalar_born_iso_{dtype}_forward_argtypes"]
            getattr(dll, f"scalar_born_iso_{accuracy}_{dtype}_backward_cuda").argtypes = \
                globals()[f"scalar_born_iso_{dtype}_backward_argtypes"]
            getattr(dll, f"scalar_born_iso_{accuracy}_{dtype}_backward_sc_cuda").argtypes = \
                globals()[f"scalar_born_iso_{dtype}_backward_sc_argtypes"]
        except AttributeError:
            pass

for accuracy in [2, 4]:
    for dtype in ["float", "double"]:
        getattr(dll, f"elastic_iso_{accuracy}_{dtype}_forward_cpu").argtypes = \
            globals()[f"elastic_iso_{dtype}_forward_argtypes"]
        getattr(dll, f"elastic_iso_{accuracy}_{dtype}_backward_cpu").argtypes = \
            globals()[f"elastic_iso_{dtype}_backward_argtypes"]
        try:
            getattr(dll, f"elastic_iso_{accuracy}_{dtype}_forward_cuda").argtypes = \
                globals()[f"elastic_iso_{dtype}_forward_argtypes"]
            getattr(dll, f"elastic_iso_{accuracy}_{dtype}_backward_cuda").argtypes = \
                globals()[f"elastic_iso_{dtype}_backward_argtypes"]
        except AttributeError:
            pass
