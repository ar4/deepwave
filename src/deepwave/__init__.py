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
if platform.system() == 'Linux':
    dll_cpu = ctypes.CDLL(
        str(
            pathlib.Path(__file__).resolve().parent /
            "libdeepwave_cpu_linux_x86_64.so"))
    dll_cuda = ctypes.CDLL(
        str(
            pathlib.Path(__file__).resolve().parent /
            "libdeepwave_cuda_linux_x86_64.so"))
elif platform.system() == 'Darwin':
    if platform.machine() == 'arm64':
        dll_cpu = ctypes.CDLL(
            str(
                pathlib.Path(__file__).resolve().parent /
                "libdeepwave_cpu_macos_arm64.dylib"))
    else:
        dll_cpu = ctypes.CDLL(
            str(
                pathlib.Path(__file__).resolve().parent /
                "libdeepwave_cpu_macos_x86_64.dylib"))
elif platform.system() == 'Windows':
    dll_cpu = ctypes.CDLL(
        str(
            pathlib.Path(__file__).resolve().parent /
            "libdeepwave_cpu_windows_x86_64.dll"))
    dll_cuda = ctypes.CDLL(
        str(
            pathlib.Path(__file__).resolve().parent /
            "libdeepwave_cuda_windows_x86_64.dll"))
else:
    raise RuntimeError("Unsupported OS or platform type")
# Check if was compiled with OpenMP support
try:
    dll_cpu.omp_get_num_threads
    use_openmp = True
except AttributeError:
    use_openmp = False
dll_cpu.scalar_iso_2_float_forward.restype = None
dll_cpu.scalar_iso_4_float_forward.restype = None
dll_cpu.scalar_iso_6_float_forward.restype = None
dll_cpu.scalar_iso_8_float_forward.restype = None
dll_cpu.scalar_iso_2_double_forward.restype = None
dll_cpu.scalar_iso_4_double_forward.restype = None
dll_cpu.scalar_iso_6_double_forward.restype = None
dll_cpu.scalar_iso_8_double_forward.restype = None
dll_cpu.scalar_iso_2_float_backward.restype = None
dll_cpu.scalar_iso_4_float_backward.restype = None
dll_cpu.scalar_iso_6_float_backward.restype = None
dll_cpu.scalar_iso_8_float_backward.restype = None
dll_cpu.scalar_iso_2_double_backward.restype = None
dll_cpu.scalar_iso_4_double_backward.restype = None
dll_cpu.scalar_iso_6_double_backward.restype = None
dll_cpu.scalar_iso_8_double_backward.restype = None
dll_cpu.scalar_born_iso_2_float_forward.restype = None
dll_cpu.scalar_born_iso_4_float_forward.restype = None
dll_cpu.scalar_born_iso_6_float_forward.restype = None
dll_cpu.scalar_born_iso_8_float_forward.restype = None
dll_cpu.scalar_born_iso_2_double_forward.restype = None
dll_cpu.scalar_born_iso_4_double_forward.restype = None
dll_cpu.scalar_born_iso_6_double_forward.restype = None
dll_cpu.scalar_born_iso_8_double_forward.restype = None
dll_cpu.scalar_born_iso_2_float_backward.restype = None
dll_cpu.scalar_born_iso_4_float_backward.restype = None
dll_cpu.scalar_born_iso_6_float_backward.restype = None
dll_cpu.scalar_born_iso_8_float_backward.restype = None
dll_cpu.scalar_born_iso_2_double_backward.restype = None
dll_cpu.scalar_born_iso_4_double_backward.restype = None
dll_cpu.scalar_born_iso_6_double_backward.restype = None
dll_cpu.scalar_born_iso_8_double_backward.restype = None
dll_cpu.scalar_born_iso_2_float_backward_sc.restype = None
dll_cpu.scalar_born_iso_4_float_backward_sc.restype = None
dll_cpu.scalar_born_iso_6_float_backward_sc.restype = None
dll_cpu.scalar_born_iso_8_float_backward_sc.restype = None
dll_cpu.scalar_born_iso_2_double_backward_sc.restype = None
dll_cpu.scalar_born_iso_4_double_backward_sc.restype = None
dll_cpu.scalar_born_iso_6_double_backward_sc.restype = None
dll_cpu.scalar_born_iso_8_double_backward_sc.restype = None
dll_cpu.elastic_iso_2_float_forward.restype = None
dll_cpu.elastic_iso_4_float_forward.restype = None
dll_cpu.elastic_iso_2_double_forward.restype = None
dll_cpu.elastic_iso_4_double_forward.restype = None
dll_cpu.elastic_iso_2_float_backward.restype = None
dll_cpu.elastic_iso_4_float_backward.restype = None
dll_cpu.elastic_iso_2_double_backward.restype = None
dll_cpu.elastic_iso_4_double_backward.restype = None
if platform.system() != 'Darwin':
    dll_cuda.scalar_iso_2_float_forward.restype = None
    dll_cuda.scalar_iso_4_float_forward.restype = None
    dll_cuda.scalar_iso_6_float_forward.restype = None
    dll_cuda.scalar_iso_8_float_forward.restype = None
    dll_cuda.scalar_iso_2_double_forward.restype = None
    dll_cuda.scalar_iso_4_double_forward.restype = None
    dll_cuda.scalar_iso_6_double_forward.restype = None
    dll_cuda.scalar_iso_8_double_forward.restype = None
    dll_cuda.scalar_iso_2_float_backward.restype = None
    dll_cuda.scalar_iso_4_float_backward.restype = None
    dll_cuda.scalar_iso_6_float_backward.restype = None
    dll_cuda.scalar_iso_8_float_backward.restype = None
    dll_cuda.scalar_iso_2_double_backward.restype = None
    dll_cuda.scalar_iso_4_double_backward.restype = None
    dll_cuda.scalar_iso_6_double_backward.restype = None
    dll_cuda.scalar_iso_8_double_backward.restype = None
    dll_cuda.scalar_born_iso_2_float_forward.restype = None
    dll_cuda.scalar_born_iso_4_float_forward.restype = None
    dll_cuda.scalar_born_iso_6_float_forward.restype = None
    dll_cuda.scalar_born_iso_8_float_forward.restype = None
    dll_cuda.scalar_born_iso_2_double_forward.restype = None
    dll_cuda.scalar_born_iso_4_double_forward.restype = None
    dll_cuda.scalar_born_iso_6_double_forward.restype = None
    dll_cuda.scalar_born_iso_8_double_forward.restype = None
    dll_cuda.scalar_born_iso_2_float_backward.restype = None
    dll_cuda.scalar_born_iso_4_float_backward.restype = None
    dll_cuda.scalar_born_iso_6_float_backward.restype = None
    dll_cuda.scalar_born_iso_8_float_backward.restype = None
    dll_cuda.scalar_born_iso_2_double_backward.restype = None
    dll_cuda.scalar_born_iso_4_double_backward.restype = None
    dll_cuda.scalar_born_iso_6_double_backward.restype = None
    dll_cuda.scalar_born_iso_8_double_backward.restype = None
    dll_cuda.scalar_born_iso_2_float_backward_sc.restype = None
    dll_cuda.scalar_born_iso_4_float_backward_sc.restype = None
    dll_cuda.scalar_born_iso_6_float_backward_sc.restype = None
    dll_cuda.scalar_born_iso_8_float_backward_sc.restype = None
    dll_cuda.scalar_born_iso_2_double_backward_sc.restype = None
    dll_cuda.scalar_born_iso_4_double_backward_sc.restype = None
    dll_cuda.scalar_born_iso_6_double_backward_sc.restype = None
    dll_cuda.scalar_born_iso_8_double_backward_sc.restype = None
    dll_cuda.elastic_iso_2_float_forward.restype = None
    dll_cuda.elastic_iso_4_float_forward.restype = None
    dll_cuda.elastic_iso_2_double_forward.restype = None
    dll_cuda.elastic_iso_4_double_forward.restype = None
    dll_cuda.elastic_iso_2_float_backward.restype = None
    dll_cuda.elastic_iso_4_float_backward.restype = None
    dll_cuda.elastic_iso_2_double_backward.restype = None
    dll_cuda.elastic_iso_4_double_backward.restype = None
scalar_iso_float_forward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_float,
    c_float, c_float, c_float, c_float, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_bool, c_int64, c_int64, c_int64, c_int64,
    c_int64
]
scalar_iso_double_forward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_double,
    c_double, c_double, c_double, c_double, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_bool, c_int64, c_int64, c_int64, c_int64,
    c_int64
]
scalar_iso_float_backward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_float, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_bool, c_int64,
    c_int64, c_int64, c_int64, c_int64
]
scalar_iso_double_backward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_double, c_double, c_double, c_double,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_bool,
    c_int64, c_int64, c_int64, c_int64, c_int64
]
scalar_born_iso_float_forward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_float,
    c_float, c_float, c_float, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_bool, c_bool, c_int64, c_int64, c_int64,
    c_int64, c_int64
]
scalar_born_iso_double_forward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_double, c_double,
    c_double, c_double, c_double, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_bool, c_bool, c_int64, c_int64, c_int64,
    c_int64, c_int64
]
scalar_born_iso_float_backward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_float,
    c_float, c_float, c_float, c_float, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_int64,
    c_int64, c_int64, c_int64, c_int64
]
scalar_born_iso_double_backward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_double,
    c_double, c_double, c_double, c_double, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_int64,
    c_int64, c_int64, c_int64, c_int64
]
scalar_born_iso_double_backward_sc_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_double, c_double, c_double, c_double,
    c_double, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_bool, c_int64, c_int64, c_int64, c_int64, c_int64
]
scalar_born_iso_float_backward_sc_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_float, c_float,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_bool,
    c_int64, c_int64, c_int64, c_int64, c_int64
]
elastic_iso_float_forward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_bool, c_bool, c_bool, c_int64, c_int64, c_int64, c_int64,
    c_int64
]
elastic_iso_double_forward_argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_void_p, c_double, c_double, c_double,
    c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_int64, c_int64, c_bool, c_bool, c_bool, c_int64, c_int64, c_int64,
    c_int64, c_int64
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
    c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_int64
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
    c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64, c_int64,
    c_int64
]
dll_cpu.scalar_iso_2_float_forward.argtypes = scalar_iso_float_forward_argtypes
dll_cpu.scalar_iso_4_float_forward.argtypes = scalar_iso_float_forward_argtypes
dll_cpu.scalar_iso_6_float_forward.argtypes = scalar_iso_float_forward_argtypes
dll_cpu.scalar_iso_8_float_forward.argtypes = scalar_iso_float_forward_argtypes
dll_cpu.scalar_iso_2_double_forward.argtypes = scalar_iso_double_forward_argtypes
dll_cpu.scalar_iso_4_double_forward.argtypes = scalar_iso_double_forward_argtypes
dll_cpu.scalar_iso_6_double_forward.argtypes = scalar_iso_double_forward_argtypes
dll_cpu.scalar_iso_8_double_forward.argtypes = scalar_iso_double_forward_argtypes
dll_cpu.scalar_iso_2_float_backward.argtypes = scalar_iso_float_backward_argtypes
dll_cpu.scalar_iso_4_float_backward.argtypes = scalar_iso_float_backward_argtypes
dll_cpu.scalar_iso_6_float_backward.argtypes = scalar_iso_float_backward_argtypes
dll_cpu.scalar_iso_8_float_backward.argtypes = scalar_iso_float_backward_argtypes
dll_cpu.scalar_iso_2_double_backward.argtypes = scalar_iso_double_backward_argtypes
dll_cpu.scalar_iso_4_double_backward.argtypes = scalar_iso_double_backward_argtypes
dll_cpu.scalar_iso_6_double_backward.argtypes = scalar_iso_double_backward_argtypes
dll_cpu.scalar_iso_8_double_backward.argtypes = scalar_iso_double_backward_argtypes
dll_cpu.scalar_born_iso_2_float_forward.argtypes = scalar_born_iso_float_forward_argtypes
dll_cpu.scalar_born_iso_4_float_forward.argtypes = scalar_born_iso_float_forward_argtypes
dll_cpu.scalar_born_iso_6_float_forward.argtypes = scalar_born_iso_float_forward_argtypes
dll_cpu.scalar_born_iso_8_float_forward.argtypes = scalar_born_iso_float_forward_argtypes
dll_cpu.scalar_born_iso_2_double_forward.argtypes = scalar_born_iso_double_forward_argtypes
dll_cpu.scalar_born_iso_4_double_forward.argtypes = scalar_born_iso_double_forward_argtypes
dll_cpu.scalar_born_iso_6_double_forward.argtypes = scalar_born_iso_double_forward_argtypes
dll_cpu.scalar_born_iso_8_double_forward.argtypes = scalar_born_iso_double_forward_argtypes
dll_cpu.scalar_born_iso_2_float_backward.argtypes = scalar_born_iso_float_backward_argtypes
dll_cpu.scalar_born_iso_4_float_backward.argtypes = scalar_born_iso_float_backward_argtypes
dll_cpu.scalar_born_iso_6_float_backward.argtypes = scalar_born_iso_float_backward_argtypes
dll_cpu.scalar_born_iso_8_float_backward.argtypes = scalar_born_iso_float_backward_argtypes
dll_cpu.scalar_born_iso_2_double_backward.argtypes = scalar_born_iso_double_backward_argtypes
dll_cpu.scalar_born_iso_4_double_backward.argtypes = scalar_born_iso_double_backward_argtypes
dll_cpu.scalar_born_iso_6_double_backward.argtypes = scalar_born_iso_double_backward_argtypes
dll_cpu.scalar_born_iso_8_double_backward.argtypes = scalar_born_iso_double_backward_argtypes
dll_cpu.scalar_born_iso_2_float_backward_sc.argtypes = scalar_born_iso_float_backward_sc_argtypes
dll_cpu.scalar_born_iso_4_float_backward_sc.argtypes = scalar_born_iso_float_backward_sc_argtypes
dll_cpu.scalar_born_iso_6_float_backward_sc.argtypes = scalar_born_iso_float_backward_sc_argtypes
dll_cpu.scalar_born_iso_8_float_backward_sc.argtypes = scalar_born_iso_float_backward_sc_argtypes
dll_cpu.scalar_born_iso_2_double_backward_sc.argtypes = scalar_born_iso_double_backward_sc_argtypes
dll_cpu.scalar_born_iso_4_double_backward_sc.argtypes = scalar_born_iso_double_backward_sc_argtypes
dll_cpu.scalar_born_iso_6_double_backward_sc.argtypes = scalar_born_iso_double_backward_sc_argtypes
dll_cpu.scalar_born_iso_8_double_backward_sc.argtypes = scalar_born_iso_double_backward_sc_argtypes
dll_cpu.elastic_iso_2_float_forward.argtypes = elastic_iso_float_forward_argtypes
dll_cpu.elastic_iso_4_float_forward.argtypes = elastic_iso_float_forward_argtypes
dll_cpu.elastic_iso_2_double_forward.argtypes = elastic_iso_double_forward_argtypes
dll_cpu.elastic_iso_4_double_forward.argtypes = elastic_iso_double_forward_argtypes
dll_cpu.elastic_iso_2_float_backward.argtypes = elastic_iso_float_backward_argtypes
dll_cpu.elastic_iso_4_float_backward.argtypes = elastic_iso_float_backward_argtypes
dll_cpu.elastic_iso_2_double_backward.argtypes = elastic_iso_double_backward_argtypes
dll_cpu.elastic_iso_4_double_backward.argtypes = elastic_iso_double_backward_argtypes
if platform.system() != 'Darwin':
    dll_cuda.scalar_iso_2_float_forward.argtypes = scalar_iso_float_forward_argtypes
    dll_cuda.scalar_iso_4_float_forward.argtypes = scalar_iso_float_forward_argtypes
    dll_cuda.scalar_iso_6_float_forward.argtypes = scalar_iso_float_forward_argtypes
    dll_cuda.scalar_iso_8_float_forward.argtypes = scalar_iso_float_forward_argtypes
    dll_cuda.scalar_iso_2_double_forward.argtypes = scalar_iso_double_forward_argtypes
    dll_cuda.scalar_iso_4_double_forward.argtypes = scalar_iso_double_forward_argtypes
    dll_cuda.scalar_iso_6_double_forward.argtypes = scalar_iso_double_forward_argtypes
    dll_cuda.scalar_iso_8_double_forward.argtypes = scalar_iso_double_forward_argtypes
    dll_cuda.scalar_iso_2_float_backward.argtypes = scalar_iso_float_backward_argtypes
    dll_cuda.scalar_iso_4_float_backward.argtypes = scalar_iso_float_backward_argtypes
    dll_cuda.scalar_iso_6_float_backward.argtypes = scalar_iso_float_backward_argtypes
    dll_cuda.scalar_iso_8_float_backward.argtypes = scalar_iso_float_backward_argtypes
    dll_cuda.scalar_iso_2_double_backward.argtypes = scalar_iso_double_backward_argtypes
    dll_cuda.scalar_iso_4_double_backward.argtypes = scalar_iso_double_backward_argtypes
    dll_cuda.scalar_iso_6_double_backward.argtypes = scalar_iso_double_backward_argtypes
    dll_cuda.scalar_iso_8_double_backward.argtypes = scalar_iso_double_backward_argtypes
    dll_cuda.scalar_born_iso_2_float_forward.argtypes = scalar_born_iso_float_forward_argtypes
    dll_cuda.scalar_born_iso_4_float_forward.argtypes = scalar_born_iso_float_forward_argtypes
    dll_cuda.scalar_born_iso_6_float_forward.argtypes = scalar_born_iso_float_forward_argtypes
    dll_cuda.scalar_born_iso_8_float_forward.argtypes = scalar_born_iso_float_forward_argtypes
    dll_cuda.scalar_born_iso_2_double_forward.argtypes = scalar_born_iso_double_forward_argtypes
    dll_cuda.scalar_born_iso_4_double_forward.argtypes = scalar_born_iso_double_forward_argtypes
    dll_cuda.scalar_born_iso_6_double_forward.argtypes = scalar_born_iso_double_forward_argtypes
    dll_cuda.scalar_born_iso_8_double_forward.argtypes = scalar_born_iso_double_forward_argtypes
    dll_cuda.scalar_born_iso_2_float_backward.argtypes = scalar_born_iso_float_backward_argtypes
    dll_cuda.scalar_born_iso_4_float_backward.argtypes = scalar_born_iso_float_backward_argtypes
    dll_cuda.scalar_born_iso_6_float_backward.argtypes = scalar_born_iso_float_backward_argtypes
    dll_cuda.scalar_born_iso_8_float_backward.argtypes = scalar_born_iso_float_backward_argtypes
    dll_cuda.scalar_born_iso_2_double_backward.argtypes = scalar_born_iso_double_backward_argtypes
    dll_cuda.scalar_born_iso_4_double_backward.argtypes = scalar_born_iso_double_backward_argtypes
    dll_cuda.scalar_born_iso_6_double_backward.argtypes = scalar_born_iso_double_backward_argtypes
    dll_cuda.scalar_born_iso_8_double_backward.argtypes = scalar_born_iso_double_backward_argtypes
    dll_cuda.scalar_born_iso_2_float_backward_sc.argtypes = scalar_born_iso_float_backward_sc_argtypes
    dll_cuda.scalar_born_iso_4_float_backward_sc.argtypes = scalar_born_iso_float_backward_sc_argtypes
    dll_cuda.scalar_born_iso_6_float_backward_sc.argtypes = scalar_born_iso_float_backward_sc_argtypes
    dll_cuda.scalar_born_iso_8_float_backward_sc.argtypes = scalar_born_iso_float_backward_sc_argtypes
    dll_cuda.scalar_born_iso_2_double_backward_sc.argtypes = scalar_born_iso_double_backward_sc_argtypes
    dll_cuda.scalar_born_iso_4_double_backward_sc.argtypes = scalar_born_iso_double_backward_sc_argtypes
    dll_cuda.scalar_born_iso_6_double_backward_sc.argtypes = scalar_born_iso_double_backward_sc_argtypes
    dll_cuda.scalar_born_iso_8_double_backward_sc.argtypes = scalar_born_iso_double_backward_sc_argtypes
    dll_cuda.elastic_iso_2_float_forward.argtypes = elastic_iso_float_forward_argtypes
    dll_cuda.elastic_iso_4_float_forward.argtypes = elastic_iso_float_forward_argtypes
    dll_cuda.elastic_iso_2_double_forward.argtypes = elastic_iso_double_forward_argtypes
    dll_cuda.elastic_iso_4_double_forward.argtypes = elastic_iso_double_forward_argtypes
    dll_cuda.elastic_iso_2_float_backward.argtypes = elastic_iso_float_backward_argtypes
    dll_cuda.elastic_iso_4_float_backward.argtypes = elastic_iso_float_backward_argtypes
    dll_cuda.elastic_iso_2_double_backward.argtypes = elastic_iso_double_backward_argtypes
    dll_cuda.elastic_iso_4_double_backward.argtypes = elastic_iso_double_backward_argtypes
