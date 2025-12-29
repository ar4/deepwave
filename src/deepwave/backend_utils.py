"""Backend utilities for Deepwave's C/CUDA interface.

This module handles the loading of the compiled C/CUDA shared library
and the dynamic assignment of argument types (argtypes) to the C functions
using ctypes. It ensures proper data marshalling between Python (PyTorch Tensors)
and the underlying C/CUDA implementations.
"""

import ctypes
import pathlib
import platform
from ctypes import c_bool, c_double, c_float, c_int64, c_size_t, c_void_p
from typing import Any, Callable, List, TypeAlias

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


def get_scalar_forward_template(ndim: int) -> List[Any]:
    """Returns the argtype template for the scalar forward propagator."""
    if not 1 <= ndim <= 3:
        raise ValueError("ndim must be 1, 2, or 3")
    # Based on src/deepwave/scalar.c
    args: List[Any] = []
    args += [c_void_p] * 4  # v, f, wfc, wfp
    args += [c_void_p] * ndim  # psi
    args += [c_void_p] * ndim  # psin
    args += [c_void_p] * ndim  # zeta
    args += [c_void_p] * 4  # w_store_1a, w_store_1b, w_store_2, w_store_3
    args += [c_void_p]  # w_filenames_ptr
    args += [c_void_p]  # r
    args += [c_void_p] * (3 * ndim)  # a, b, dbdx
    args += [c_void_p] * 2  # sources_i, receivers_i
    args += [FLOAT_TYPE] * ndim  # rd
    args += [FLOAT_TYPE] * ndim  # rd2
    args += [FLOAT_TYPE]  # dt2
    args += [c_int64] * 2  # nt, n_shots
    args += [c_int64] * ndim  # n
    args += [c_int64] * 3  # n_sources_per_shot, n_receivers_per_shot, step_ratio
    args += [c_int64]  # storage_mode
    args += [c_size_t] * 2  # shot_bytes_uncomp, shot_bytes_comp
    args += [c_bool] * 3  # v_requires_grad, v_batched, storage_compression
    args += [c_int64]  # start_t
    args += [c_int64] * (2 * ndim)  # pml
    args += [c_int64]  # n_threads
    args += [c_void_p]  # stream
    return args


def get_scalar_backward_template(ndim: int) -> List[Any]:
    """Returns the argtype template for the scalar backward propagator."""
    if not 1 <= ndim <= 3:
        raise ValueError("ndim must be 1, 2, or 3")
    # Based on src/deepwave/scalar.c
    args: List[Any] = []
    args += [c_void_p] * 4  # v2dt2, grad_r, wfc, wfp
    args += [c_void_p] * ndim  # psi
    args += [c_void_p] * ndim  # psin
    args += [c_void_p] * ndim  # zeta
    args += [c_void_p] * ndim  # zetan
    args += [c_void_p] * 4  # w_store_1a, w_store_1b, w_store_2, w_store_3
    args += [c_void_p]  # w_filenames_ptr
    args += [c_void_p] * 3  # grad_f, grad_v, grad_v_thread
    args += [c_void_p] * (3 * ndim)  # a, b, dbdx
    args += [c_void_p] * 2  # sources_i, receivers_i
    args += [FLOAT_TYPE] * ndim  # rd
    args += [FLOAT_TYPE] * ndim  # rd2
    args += [c_int64] * 2  # nt, n_shots
    args += [c_int64] * ndim  # n
    args += [c_int64] * 3  # n_sources_per_shot, n_receivers_per_shot, step_ratio
    args += [c_int64]  # storage_mode
    args += [c_size_t] * 2  # shot_bytes_uncomp, shot_bytes_comp
    args += [c_bool] * 3  # v_requires_grad, v_batched, storage_compression
    args += [c_int64]  # start_t
    args += [c_int64] * (2 * ndim)  # pml
    args += [c_int64]  # n_threads
    args += [c_void_p]  # stream
    return args


def get_scalar_born_forward_template(ndim: int) -> List[Any]:
    """Returns the argtype template for the scalar Born forward propagator."""
    if not 1 <= ndim <= 3:
        raise ValueError("ndim must be 1, 2, or 3")
    # Based on src/deepwave/scalar_born.c
    args: List[Any] = []
    args += [c_void_p] * 4  # v, scatter, f, fsc
    args += [c_void_p] * 2  # wfc, wfp
    args += [c_void_p] * ndim  # psiy, psix...
    args += [c_void_p] * ndim  # psiyn, psixn...
    args += [c_void_p] * ndim  # zetay, zetax...
    args += [c_void_p] * 2  # wfcsc, wfpsc
    args += [c_void_p] * ndim  # psiysc, psixsc...
    args += [c_void_p] * ndim  # psiynsc, psixnsc...
    args += [c_void_p] * ndim  # zetaysc, zetaxsc...
    args += [c_void_p] * 4  # w_store_1a, w_store_1b, w_store_2, w_store_3
    args += [c_void_p]  # w_filenames
    args += [c_void_p] * 4  # wsc_store_1a, w_store_1b, wsc_store_2, wsc_store_3
    args += [c_void_p]  # sc w_filenames
    args += [c_void_p] * 2  # r, rsc
    args += [c_void_p] * (3 * ndim)  # a, b, dbdx
    args += [c_void_p] * 3  # sources_i, receivers_i, receiverssc_i
    args += [FLOAT_TYPE] * ndim  # rdy, rdx...
    args += [FLOAT_TYPE] * ndim  # rdy2, rdx2...
    args += [FLOAT_TYPE]  # dt2
    args += [c_int64] * 2  # nt, n_shots
    args += [c_int64] * ndim  # ny, nx...
    args += [
        c_int64
    ] * 3  # n_sources_per_shot, n_receivers_per_shot, n_receiverssc_per_shot
    args += [c_int64]  # step_ratio
    args += [c_int64]  # storage_mode
    args += [c_size_t] * 2  # shot_bytes_uncomp, shot_bytes_comp
    args += [
        c_bool
    ] * 5  # v_requires_grad, scatter_requires_grad, v_batched, scatter_batched,
    # storage_compression
    args += [c_int64]  # start_t
    args += [c_int64] * (2 * ndim)  # pml
    args += [c_int64]  # n_threads
    args += [c_void_p]  # stream
    return args


def get_scalar_born_backward_template(ndim: int) -> List[Any]:
    """Returns the argtype template for the scalar Born backward propagator."""
    if not 1 <= ndim <= 3:
        raise ValueError("ndim must be 1, 2, or 3")
    # Based on src/deepwave/scalar_born.c
    args: List[Any] = []
    args += [c_void_p] * 4  # v, scatter, grad_r, grad_rsc
    args += [c_void_p] * 2  # wfc, wfp
    args += [c_void_p] * ndim  # psi
    args += [c_void_p] * ndim  # psin
    args += [c_void_p] * ndim  # zeta
    args += [c_void_p] * ndim  # zetan
    args += [c_void_p] * 2  # wfcsc, wfpsc
    args += [c_void_p] * ndim  # psisc
    args += [c_void_p] * ndim  # psinsc
    args += [c_void_p] * ndim  # zetasc
    args += [c_void_p] * ndim  # zetansc
    args += [c_void_p] * 4  # w_store_1a, w_store_1b, w_store_2, w_store_3
    args += [c_void_p]  # w_filenames
    args += [c_void_p] * 4  # wsc_store_1a, w_store_1b, wsc_store_2, wsc_store_3
    args += [c_void_p]  # wsc_filenames
    args += [
        c_void_p
    ] * 6  # grad_f, grad_fsc, grad_v, grad_scatter, grad_v_thread, grad_scatter_thread
    args += [c_void_p] * (3 * ndim)  # a, b, dbdx
    args += [c_void_p] * 3  # sources_i, receivers_i, receiverssc_i
    args += [FLOAT_TYPE] * ndim  # rd
    args += [FLOAT_TYPE] * ndim  # rd2
    args += [FLOAT_TYPE]  # dt2
    args += [c_int64] * 2  # nt, n_shots
    args += [c_int64] * ndim  # n
    args += [c_int64] * 4  # n_sources_per_shot, n_sourcessc_per_shot,
    # n_receivers_per_shot, n_receiverssc_per_shot
    args += [c_int64]  # step_ratio
    args += [c_int64]  # storage_mode
    args += [c_size_t] * 2  # shot_bytes_uncomp, shot_bytes_comp
    args += [
        c_bool
    ] * 5  # v_requires_grad, scatter_requires_grad, v_batched, scatter_batched,
    # storage_compression
    args += [c_int64]  # start_t
    args += [c_int64] * (2 * ndim)  # pml
    args += [c_int64]  # n_threads
    args += [c_void_p]  # stream
    return args


def get_scalar_born_backward_sc_template(ndim: int) -> List[Any]:
    """Returns the argtype template for the scalar Born backward_sc propagator."""
    if not 1 <= ndim <= 3:
        raise ValueError("ndim must be 1, 2, or 3")
    # Based on src/deepwave/scalar_born.c
    args: List[Any] = []
    args += [c_void_p] * 2  # v, grad_rsc
    args += [c_void_p] * 2  # wfcsc, wfpsc
    args += [c_void_p] * ndim  # psisc
    args += [c_void_p] * ndim  # psinsc
    args += [c_void_p] * ndim  # zetasc
    args += [c_void_p] * ndim  # zetansc
    args += [c_void_p] * 4  # w_store_1a, w_store_1b, w_store_2, w_store_3
    args += [c_void_p] * 1  # w_filenames
    args += [c_void_p] * 3  # grad_fsc, grad_scatter, grad_scatter_thread
    args += [c_void_p] * (3 * ndim)  # a, b, dbdx
    args += [c_void_p] * 2  # sources_i, receiverssc_i
    args += [FLOAT_TYPE] * ndim  # rd
    args += [FLOAT_TYPE] * ndim  # rd2
    args += [FLOAT_TYPE]  # dt2
    args += [c_int64] * 2  # nt, n_shots
    args += [c_int64] * ndim  # n
    args += [c_int64] * 2  # n_sourcessc_per_shot, n_receiverssc_per_shot
    args += [c_int64]  # step_ratio
    args += [c_int64]  # storage_mode
    args += [c_size_t] * 2  # shot_bytes_uncomp, shot_bytes_comp
    args += [
        c_bool
    ] * 4  # scatter_requires_grad, v_batched, scatter_batched, storage_compression
    args += [c_int64]  # start_t
    args += [c_int64] * (2 * ndim)  # pml
    args += [c_int64]  # n_threads
    args += [c_void_p]  # stream
    return args


def get_elastic_forward_template(ndim: int) -> List[Any]:
    """Returns the argtype template for the elastic forward propagator."""
    if not 1 <= ndim <= 3:
        raise ValueError("ndim must be 1, 2, or 3")
    # Based on src/deepwave/elastic.c
    args: List[Any] = []
    # Model
    args += [c_void_p] * 2  # lamb, mu
    if ndim == 3:
        args += [c_void_p] * 2  # mu_zy, mu_zx
    if ndim >= 2:
        args += [c_void_p]  # mu_yx
    if ndim == 3:
        args += [c_void_p]  # buoyancy_z
    if ndim >= 2:
        args += [c_void_p]  # buoyancy_y
    args += [c_void_p]  # buoyancy_x

    # source
    args += [c_void_p] * (ndim + 1)  # f

    # wavefields
    if ndim == 3:
        args += [c_void_p] * 14
    if ndim >= 2:
        args += [c_void_p] * 9
    args += [c_void_p] * 4

    # backward storage
    if ndim == 3:
        args += [c_void_p] * 4 * 5
    if ndim >= 2:
        args += [c_void_p] * 3 * 5
    args += [c_void_p] * 2 * 5

    # recorded data
    if ndim == 3:
        args += [c_void_p]
    if ndim >= 2:
        args += [c_void_p]
    args += [c_void_p] * 2

    # PML profile coefficients
    if ndim == 3:
        args += [c_void_p] * 4
    if ndim >= 2:
        args += [c_void_p] * 4
    args += [c_void_p] * 4

    # source locations
    if ndim == 3:
        args += [c_void_p]
    if ndim >= 2:
        args += [c_void_p]
    args += [c_void_p] * 2

    # receiver locations
    if ndim == 3:
        args += [c_void_p]
    if ndim >= 2:
        args += [c_void_p]
    args += [c_void_p] * 2

    # spatial and temporal discretization
    args += [FLOAT_TYPE] * ndim  # rdx
    args += [FLOAT_TYPE]  # dt

    # sizes
    args += [c_int64] * 2  # nt, n_shots
    args += [c_int64] * ndim  # model_shape
    args += [c_int64] * (ndim + 1)  # n_sources_per_shot
    args += [c_int64] * (ndim + 1)  # n_receivers_per_shot
    args += [c_int64]  # step_ratio

    # options
    args += [c_int64]  # storage_mode
    args += [c_size_t] * 2  # shot_bytes_uncomp, shot_bytes_comp
    args += [c_bool] * 6  # requires_grad, batched
    args += [c_bool]  # storage_compression

    # start_t
    args += [c_int64]

    # pml range indices
    args += [c_int64] * (2 * ndim)

    # aux
    args += [c_int64]
    args += [c_void_p]  # stream

    return args


def get_elastic_backward_template(ndim: int) -> List[Any]:
    """Returns the argtype template for the elastic backward propagator."""
    if not 1 <= ndim <= 3:
        raise ValueError("ndim must be 1, 2, or 3")
    # Based on src/deepwave/elastic.c
    args: List[Any] = []
    # Model
    args += [c_void_p] * 2  # lamb, mu
    if ndim == 3:
        args += [c_void_p] * 2  # mu_zy, mu_zx
    if ndim >= 2:
        args += [c_void_p]  # mu_yx
    if ndim == 3:
        args += [c_void_p]  # buoyancy_z
    if ndim >= 2:
        args += [c_void_p]  # buoyancy_y
    args += [c_void_p]  # buoyancy_x

    # source
    args += [c_void_p] * (ndim + 1)  # grad_r

    # wavefields
    if ndim == 3:
        args += [c_void_p] * 14
    if ndim >= 2:
        args += [c_void_p] * 9
    args += [c_void_p] * 4

    # m_sigma_n
    if ndim == 3:
        args += [c_void_p] * 5
    if ndim >= 2:
        args += [c_void_p] * 3
    args += [c_void_p]

    # backward storage
    if ndim == 3:
        args += [c_void_p] * 4 * 5
    if ndim >= 2:
        args += [c_void_p] * 3 * 5
    args += [c_void_p] * 2 * 5

    # grad_f
    if ndim == 3:
        args += [c_void_p]
    if ndim >= 2:
        args += [c_void_p]
    args += [c_void_p] * 2

    # Model gradients
    args += [c_void_p] * 2  # lamb, mu
    if ndim == 3:
        args += [c_void_p] * 2  # mu_zy, mu_zx
    if ndim >= 2:
        args += [c_void_p]  # mu_yx
    if ndim == 3:
        args += [c_void_p]  # buoyancy_z
    if ndim >= 2:
        args += [c_void_p]  # buoyancy_y
    args += [c_void_p]  # buoyancy_x

    # Model gradients (thread)
    args += [c_void_p] * 2  # lamb, mu
    if ndim == 3:
        args += [c_void_p] * 2  # mu_zy, mu_zx
    if ndim >= 2:
        args += [c_void_p]  # mu_yx
    if ndim == 3:
        args += [c_void_p]  # buoyancy_z
    if ndim >= 2:
        args += [c_void_p]  # buoyancy_y
    args += [c_void_p]  # buoyancy_x

    # PML profile coefficients
    if ndim == 3:
        args += [c_void_p] * 4
    if ndim >= 2:
        args += [c_void_p] * 4
    args += [c_void_p] * 4

    # source locations
    if ndim == 3:
        args += [c_void_p]
    if ndim >= 2:
        args += [c_void_p]
    args += [c_void_p] * 2

    # receiver locations
    if ndim == 3:
        args += [c_void_p]
    if ndim >= 2:
        args += [c_void_p]
    args += [c_void_p] * 2

    # spatial and temporal discretization
    args += [FLOAT_TYPE] * ndim  # rdx
    args += [FLOAT_TYPE]  # dt

    # sizes
    args += [c_int64] * 2  # nt
    args += [c_int64] * ndim  # model_shape
    args += [c_int64] * (ndim + 1)  # n_sources_per_shot
    args += [c_int64] * (ndim + 1)  # n_receivers_per_shot
    args += [c_int64]  # step_ratio

    # options
    args += [c_int64]  # storage_mode
    args += [c_size_t] * 2  # shot_bytes_uncomp, shot_bytes_comp
    args += [c_bool] * 6  # requires_grad, batched
    args += [c_bool]  # storage_compression

    # start_t
    args += [c_int64]

    # pml range indices
    args += [c_int64] * (2 * ndim)

    # aux
    args += [c_int64]
    args += [c_void_p]  # stream

    return args


def get_acoustic_forward_template(ndim: int) -> List[Any]:
    """Returns the argtype template for the acoustic forward propagator."""
    if not 1 <= ndim <= 3:
        raise ValueError("ndim must be 1, 2, or 3")
    # Based on src/deepwave/acoustic.c
    args: List[Any] = []
    args += [c_void_p] * (1 + ndim)  # k, buoyancy
    args += [c_void_p] * (1 + ndim)  # f
    args += [c_void_p] * (1 + 3 * ndim)  # p, v, phi, psi
    args += [c_void_p] * 5  # k_store_1a, k_store_1b, k_store_2, k_store_3, k_filenames
    args += [c_void_p] * (5 * ndim)  # b_store...
    args += [c_void_p] * (1 + ndim)  # receiver_amplitudes
    args += [c_void_p] * (4 * ndim)  # a, b, ah, bh
    args += [c_void_p] * (1 + ndim)  # sources_i
    args += [c_void_p] * (1 + ndim)  # receivers_i
    args += [FLOAT_TYPE] * (ndim + 1)  # rdx, dt
    args += [c_int64] * 2  # nt, n_shots
    args += [c_int64] * ndim  # n
    args += [c_int64] * (1 + ndim)  # n_sources_per_shot
    args += [c_int64] * (1 + ndim)  # n_receivers_per_shot
    args += [c_int64] * 2  # step_ratio, storage_mode
    args += [c_size_t] * 2  # shot_bytes_uncomp, shot_bytes_comp
    args += [c_bool] * 5  # k_req_grad, b_req_grad, k_batched, b_batched, compress
    args += [c_int64]  # start_t
    args += [c_int64] * (2 * ndim)  # pml
    args += [c_int64]  # n_threads
    args += [c_void_p]  # stream
    return args


def get_acoustic_backward_template(ndim: int) -> List[Any]:
    """Returns the argtype template for the acoustic backward propagator."""
    if not 1 <= ndim <= 3:
        raise ValueError("ndim must be 1, 2, or 3")
    # Based on src/deepwave/acoustic.c
    args: List[Any] = []
    args += [c_void_p] * (1 + ndim)  # k, buoyancy
    args += [c_void_p] * (1 + ndim)  # grad_r
    args += [c_void_p] * (1 + 3 * ndim)  # p, v, phi, psi
    args += [c_void_p] * (ndim)  # psin
    args += [c_void_p] * 5  # k_store
    args += [c_void_p] * (5 * ndim)  # b_store
    args += [c_void_p] * (1 + ndim)  # grad_f
    args += [c_void_p] * (2 + 2 * ndim)  # grad_k, grad_b, grad_k_thread, grad_b_thread
    args += [c_void_p] * (4 * ndim)  # a, b, ah, bh
    args += [c_void_p] * (1 + ndim)  # sources_i
    args += [c_void_p] * (1 + ndim)  # receivers_i
    args += [FLOAT_TYPE] * (ndim + 1)  # rdx, dt
    args += [c_int64] * 2  # nt, n_shots
    args += [c_int64] * ndim  # n
    args += [c_int64] * (1 + ndim)  # n_sources_per_shot
    args += [c_int64] * (1 + ndim)  # n_receivers_per_shot
    args += [c_int64] * 2  # step_ratio, storage_mode
    args += [c_size_t] * 2  # shot_bytes_uncomp, shot_bytes_comp
    args += [c_bool] * 5  # k_req_grad, b_req_grad, k_batched, b_batched, compress
    args += [c_int64]  # start_t
    args += [c_int64] * (2 * ndim)  # pml
    args += [c_int64]  # n_threads
    args += [c_void_p]  # stream
    return args


# A dictionary to hold all the template generator functions
templates: dict[str, Callable[[int], List[Any]]] = {
    "scalar_forward": get_scalar_forward_template,
    "scalar_backward": get_scalar_backward_template,
    "scalar_born_forward": get_scalar_born_forward_template,
    "scalar_born_backward": get_scalar_born_backward_template,
    "scalar_born_backward_sc": get_scalar_born_backward_sc_template,
    "elastic_forward": get_elastic_forward_template,
    "elastic_backward": get_elastic_backward_template,
    "acoustic_forward": get_acoustic_forward_template,
    "acoustic_backward": get_acoustic_backward_template,
}


def _get_argtypes(template_name: str, ndim: int, float_type: type) -> List[Any]:
    """Generates a concrete argtype list from a template and a float type.

    This function takes a template name (e.g., "scalar_forward"), the number
    of dimensions, and a specific float type (e.g., `c_float` or `c_double`)
    and produces the list of ctypes for the C function.

    Args:
        template_name: The name of the argtype template to use.
        ndim: The number of spatial dimensions (1, 2, or 3).
        float_type: The `ctypes` float type (`c_float` or `c_double`)
            to substitute into the template.

    Returns:
        List[Any]: A list of `ctypes` types representing the argument
            signature for a C function.

    """
    template = templates[template_name](ndim)
    return [float_type if t is FLOAT_TYPE else t for t in template]


def _assign_argtypes(
    propagator: str,
    ndim: int,
    accuracy: int,
    dtype: str,
    direction: str,
    extra: str = "",
) -> None:
    """Dynamically assigns ctypes argtypes to a given C function.

    This function constructs the full C function name based on the provided
    parameters (propagator, ndim, accuracy, dtype, direction, and extra suffix)
    and then assigns the corresponding `ctypes` argument types to it.
    It handles both CPU and CUDA versions of the functions.

    Args:
        propagator: The name of the propagator (e.g., "scalar", "elastic").
        ndim: The number of spatial dimensions.
        accuracy: The finite-difference accuracy order (e.g., 2, 4, 6, 8).
        dtype: The data type as a string (e.g., "float", "double").
        direction: The direction of propagation (e.g., "forward", "backward").
        extra: An optional extra suffix for the function name (e.g., "_sc").

    Raises:
        AttributeError: If a function with the constructed name is not found
            in the loaded shared library (this is caught internally and skipped).

    """
    template_name = f"{propagator}_{direction}{extra}"
    float_type = c_float if dtype == "float" else c_double
    argtypes = _get_argtypes(template_name, ndim, float_type)

    for device in ["cpu", "cuda"]:
        func_name = (
            f"{propagator}_iso_{ndim}d_{accuracy}_{dtype}_{direction}{extra}_{device}"
        )
        try:
            func = getattr(dll, func_name)
            func.argtypes = argtypes
            func.restype = ctypes.c_int
        except AttributeError:
            continue


def get_backend_function(
    propagator: str,
    ndim: int,
    pass_name: str,
    accuracy: int,
    dtype: torch.dtype,
    device: torch.device,
    extra: str = "",
) -> CFunctionPointer:
    """Selects and returns the appropriate backend C/CUDA function.

    Args:
        propagator: The name of the propagator (e.g., "scalar", "elastic").
        ndim: The number of spatial dimensions.
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
        f"{propagator}_iso_{ndim}d_{accuracy}_{dtype_str}_"
        f"{pass_name}{extra}_{device_str}"
    )

    try:
        return getattr(dll, func_name)
    except AttributeError as e:
        raise AttributeError(f"Backend function {func_name} not found.") from e


# Now, iterate through all defined dimensions, accuracies, and data types to
# assign the correct argtypes to the dynamically loaded C/CUDA functions.
# The function names are constructed based on the naming convention
# used in the CMake build system (e.g., scalar_iso_2d_4_float_forward_cpu).
for current_ndim in [1, 2, 3]:
    for current_accuracy in [2, 4, 6, 8]:
        for current_dtype in ["float", "double"]:
            _assign_argtypes(
                "scalar", current_ndim, current_accuracy, current_dtype, "forward"
            )
            _assign_argtypes(
                "scalar", current_ndim, current_accuracy, current_dtype, "backward"
            )
            _assign_argtypes(
                "scalar_born", current_ndim, current_accuracy, current_dtype, "forward"
            )
            _assign_argtypes(
                "scalar_born", current_ndim, current_accuracy, current_dtype, "backward"
            )
            _assign_argtypes(
                "scalar_born",
                current_ndim,
                current_accuracy,
                current_dtype,
                "backward",
                extra="_sc",
            )
            _assign_argtypes(
                "elastic", current_ndim, current_accuracy, current_dtype, "forward"
            )
            _assign_argtypes(
                "elastic", current_ndim, current_accuracy, current_dtype, "backward"
            )
            _assign_argtypes(
                "acoustic", current_ndim, current_accuracy, current_dtype, "forward"
            )
            _assign_argtypes(
                "acoustic", current_ndim, current_accuracy, current_dtype, "backward"
            )
