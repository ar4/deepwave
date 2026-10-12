"""Backend utilities for Deepwave's C/CUDA interface.

This module handles the loading of the compiled C/CUDA shared library
and the dynamic assignment of argument types (argtypes) to the C functions
using ctypes. It ensures proper data marshalling between Python (PyTorch Tensors)
and the underlying C/CUDA implementations.

Internal module -- no stability guarantee.
"""

import ctypes
import importlib.util
import pathlib
import platform
from ctypes import c_bool, c_double, c_float, c_int64, c_size_t, c_void_p
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from deepwave._exceptions import (
    DeepwaveRuntimeError,
    DeepwaveTypeError,
    DeepwaveValueError,
)

# An implicit alias rather than an explicit ``x: TypeAlias = ...`` one:
# ``typing.TypeAlias`` is Python 3.10+, and this module is imported
# unconditionally by ``import deepwave``, so naming it here would make the
# whole package unimportable on Python 3.9 -- pyproject.toml's
# ``requires-python`` floor. Type checkers read the assignment below as an
# alias for ``Any`` either way.
CFunctionPointer = Any

# Platform-specific shared library extension
SO_EXT = {"Linux": "so", "Darwin": "dylib", "Windows": "dll"}.get(platform.system())
if SO_EXT is None:
    raise DeepwaveRuntimeError("Unsupported OS or platform type")


def _find_compiled_library() -> str:
    """Locates the compiled Deepwave shared library.

    The build system (scikit-build-core) records the library as the
    ``deepwave.libdeepwave_C`` module, so importlib can tell us where it lives:
    for a regular wheel install the bare ``.so`` is discovered by importlib's
    extension-module loader, and for an editable install scikit-build-core's
    import finder maps the name to the built artifact (which lives in the
    installed package directory rather than next to these Python sources). We
    therefore ask importlib rather than guessing the path from ``__file__``,
    falling back to a sibling file only if the module is not registered.
    """
    spec = importlib.util.find_spec("deepwave.libdeepwave_C")
    if spec is not None and spec.origin is not None:
        return spec.origin
    fallback = pathlib.Path(__file__).resolve().parent / f"libdeepwave_C.{SO_EXT}"
    if fallback.exists():
        return str(fallback)
    raise DeepwaveRuntimeError(
        "Could not locate the compiled Deepwave library "
        f"(libdeepwave_C.{SO_EXT}). Is Deepwave installed correctly?"
    )


dll = ctypes.CDLL(_find_compiled_library())

# Check if was compiled with OpenMP support
USE_OPENMP = hasattr(dll, "omp_get_num_threads")


# --- CPU capability probe (x86 SIGILL guard + diagnostics) ------------------
#
# The compiled backend may require a particular instruction set. Historically
# the wheels baked in AVX2, so importing Deepwave and then running on a pre-AVX2
# x86 CPU aborted the process with a bare ``Illegal instruction (core dumped)``
# and no Python traceback. The library now exposes two helpers compiled for the
# baseline ISA only (see ``cpu_features.c``) that let us detect this *before* the
# first kernel call and raise a clear, actionable error instead -- and expose the
# running CPU's SIMD capabilities for diagnostics. Both helpers are optional: an
# older ``.so`` built before they were added simply lacks the symbols, in which
# case the probe degrades to a no-op.

_CPU_FEATURE_NAMES = ("sse2", "avx", "avx2", "avx512f")


def _cpu_supports(feature: str) -> Optional[int]:
    """Query the running CPU for an x86 SIMD ``feature``.

    Returns 1 if supported, 0 if not, -1 if it cannot be determined (e.g. on a
    non-x86 build), or ``None`` if the loaded library predates the probe helper.
    """
    fn = getattr(dll, "deepwave_cpu_supports", None)
    if fn is None:
        return None
    fn.argtypes = [ctypes.c_char_p]
    fn.restype = ctypes.c_int
    return int(fn(feature.encode("ascii")))


def _required_baseline() -> Optional[str]:
    """Return the ISA the compiled kernels require as an absolute baseline.

    An empty string means "no fixed requirement" (runtime dispatch, or an
    unforced baseline); ``None`` means the loaded library predates the helper.
    """
    fn = getattr(dll, "deepwave_cpu_required_baseline", None)
    if fn is None:
        return None
    fn.argtypes = []
    fn.restype = ctypes.c_char_p
    val = fn()
    return val.decode("ascii") if val else ""


def cpu_features() -> dict[str, Any]:
    """Report the running CPU's SIMD support and the backend's ISA baseline.

    Intended for diagnostics (e.g. when filing a bug report). On non-x86
    architectures the per-feature values are ``None`` (not applicable). Keys are
    the x86 feature names in ``_CPU_FEATURE_NAMES`` (each ``True``/``False``/
    ``None``), ``"required_baseline"`` (``str`` or ``None``), and ``"machine"``.
    """
    feats: dict[str, Any] = {}
    for name in _CPU_FEATURE_NAMES:
        res = _cpu_supports(name)
        feats[name] = None if res is None or res < 0 else bool(res)
    feats["required_baseline"] = _required_baseline()
    feats["machine"] = platform.machine()
    return feats


def _check_cpu_baseline() -> None:
    """Turn a would-be SIGILL into a clear error before the first kernel call.

    Raises ``RuntimeError`` if the compiled kernels require an instruction set
    the running CPU lacks. With ``target_clones`` runtime dispatch the kernels'
    baseline clone runs on any x86-64 CPU, so this never fires; it guards the
    fallback case where a wheel was built with a fixed ISA baseline (e.g.
    ``-mavx2``) and is then run on an older CPU.
    """
    required = _required_baseline()
    if not required:
        return
    if _cpu_supports(required) == 0:
        raise DeepwaveRuntimeError(
            f"This Deepwave binary was compiled requiring the {required!r} "
            "instruction set, but the current CPU does not support it. Without "
            "this check the process would abort with 'Illegal instruction "
            "(core dumped)' on the first wave propagation. Build a version "
            "with the portable CPU baseline instead (the source-build "
            "default; do not enable DW_FIXED_AVX2_OPT_IN):\n\n"
            "    python -m pip install --force-reinstall --no-deps "
            "--no-binary deepwave deepwave\n\n"
            "(see the 'Troubleshooting' section of the Deepwave documentation)."
        )


_check_cpu_baseline()

# --- ctypes argtype templates ----------------------------------------------
#
# Each backend entry point has a flat C signature whose arguments follow the
# canonical grouping
#   [models][sources][wavefields(+temps)][storage][receivers][model coeffs]
#   [locations]  then the scalar tail [spacings][sizes][flags][pml][stream];
# only the *contents* differ between propagators. Rather than spell out every
# argument, each template below returns a compact list of ``(ctype, count)``
# segments which :func:`_seg` expands into the flat argtype list. ``FLOAT_TYPE``
# marks the float/double arguments and is substituted with the concrete type by
# :func:`_get_argtypes`. The exact expanded result for every signature is frozen
# by ``tests/abi_reference.json`` (see ``tests/test_abi_stability.py``), and
# ``tests/test_abi_drift.py`` checks that the matching ``state.backend(...)``
# call sites emit arguments consistent with these types.

# Placeholder replaced by the appropriate float type (c_float or c_double).
FLOAT_TYPE: type = c_float

# Short aliases used by the segment tables below.
_P = c_void_p  # pointer (tensor data, filename array, stream)
_I = c_int64  # 64-bit integer (sizes, counts, indices)
_Z = c_size_t  # byte counts
_B = c_bool  # boolean flag
_F = FLOAT_TYPE  # float or double (resolved in _get_argtypes)

_Segments = list[Any]


def _seg(segments: _Segments) -> list[Any]:
    """Expand ``(ctype, count)`` segments into a flat argtype list."""
    out: list[Any] = []
    for ctype, count in segments:
        out += [ctype] * count
    return out


def _check_ndim(ndim: int) -> None:
    if not 1 <= ndim <= 3:
        raise DeepwaveValueError("ndim must be 1, 2, or 3")


def _tri(ndim: int, three: int, two: int, base: int) -> _Segments:
    """Pointer segments that grow with ndim (3D and 2D add extra dimensions).

    Used by the staggered elastic signatures, whose per-group pointer counts
    accumulate ``three`` extra in 3D and ``two`` extra in 2D on top of ``base``.
    """
    segs: _Segments = []
    if ndim == 3:
        segs.append((_P, three))
    if ndim >= 2:
        segs.append((_P, two))
    segs.append((_P, base))
    return segs


def _elastic_model(ndim: int) -> _Segments:
    """The elastic model (or model-gradient) pointer block for ``ndim``."""
    segs: _Segments = [(_P, 2)]  # lamb, mu
    if ndim == 3:
        segs.append((_P, 2))  # mu_zy, mu_zx
    if ndim >= 2:
        segs.append((_P, 1))  # mu_yx
    if ndim == 3:
        segs.append((_P, 1))  # buoyancy_z
    if ndim >= 2:
        segs.append((_P, 1))  # buoyancy_y
    segs.append((_P, 1))  # buoyancy_x
    return segs


def _elastic_vti_model(ndim: int) -> _Segments:
    """The elastic VTI model (or model-gradient) pointer block for ``ndim``.

    2-D: c11, c13, c33, c55_yx, buoyancy_y, buoyancy_x.
    3-D: c11, c12, c13, c33, c55_zy, c55_zx, c66_yx, buoyancy_z, buoyancy_y,
    buoyancy_x. (VTI has no 1-D form; anisotropy about the vertical axis is
    meaningless with a single spatial dimension.)
    """
    segs: _Segments = [(_P, 1)]  # c11
    if ndim == 3:
        segs.append((_P, 1))  # c12
    segs.append((_P, 2))  # c13, c33
    if ndim == 3:
        segs.append((_P, 2))  # c55_zy, c55_zx
    segs.append((_P, 1))  # c66_yx (3-D) / c55_yx (2-D)
    if ndim == 3:
        segs.append((_P, 1))  # buoyancy_z
    segs.append((_P, 2))  # buoyancy_y, buoyancy_x
    return segs


def _n_elastic_vti_flag_models(ndim: int) -> int:
    """Number of independent VTI models with requires-grad/batched flags.

    2-D: c11, c13, c33, c55, buoyancy (5).
    3-D: c11, c12, c13, c33, c55, c66, buoyancy (7).
    """
    return 7 if ndim == 3 else 5


def _acoustic_vti_model(ndim: int) -> _Segments:
    """The acoustic VTI model (or model-gradient) pointer block for ``ndim``.

    2-D: c11, c13, c33, buoyancy_y, buoyancy_x.
    3-D: c11, c13, c33, buoyancy_z, buoyancy_y, buoyancy_x. (No c12/c55/c66:
    the ``V_S = 0`` reduction has no shear stiffness. No 1-D form.)
    """
    segs: _Segments = [(_P, 1)]  # c11
    segs.append((_P, 2))  # c13, c33
    if ndim == 3:
        segs.append((_P, 1))  # buoyancy_z
    segs.append((_P, 2))  # buoyancy_y, buoyancy_x
    return segs


def _n_acoustic_vti_flag_models() -> int:
    """Number of independent acoustic VTI models with requires-grad/batched flags.

    c11, c13, c33, buoyancy (4, the same set for 2-D and 3-D -- unlike
    elastic_vti there is no ndim-dependent shear stiffness).
    """
    return 4


def _acoustic_vti_wavefields(ndim: int) -> _Segments:
    """The acoustic VTI wavefield pointer block for ``ndim``.

    Matches the order of
    ``acoustic_vti.acoustic_vti_wavefield_names``: velocities,
    the two normal stresses, the velocity-derivative C-PML memories (stress
    update), then the stress-derivative C-PML memories (velocity update).
    Each block grows by one pointer per dimension (no shear cross terms
    survive the ``V_S = 0`` reduction, so unlike elastic_vti's ``_tri``
    counts there is exactly one memory per dimension per block).
    """
    segs: _Segments = []
    segs += _tri(ndim, 1, 1, 1)  # velocities: vz(3-D), vy, vx
    segs.append((_P, 2))  # sigma_v, sigma_h
    segs += _tri(ndim, 1, 1, 1)  # m_vzz(3-D), m_vyy, m_vxx
    segs += _tri(ndim, 1, 1, 1)  # m_sigma_v_z(3-D)/m_sigma_h_y, m_sigma_h_x
    return segs


def get_acoustic_vti_forward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the acoustic VTI forward propagator.

    Structurally the same canonical staggered-grid layout as
    :func:`get_elastic_vti_forward_template`, reduced to the pseudo-acoustic
    field set: ``ndim + 2`` source/receiver channels (the velocities plus the
    two normal stresses ``sigma_h``, ``sigma_v``) instead of elastic_vti's
    ``ndim + 1`` (a single combined pressure channel), and 4 independent
    models instead of elastic_vti's 5/7 (no shear). VTI supports only 2-D
    and 3-D.
    """
    if ndim not in (2, 3):
        raise DeepwaveValueError("acoustic_vti supports only ndim 2 and 3")
    n = ndim
    nf = _n_acoustic_vti_flag_models()
    segments: _Segments = []
    segments += _acoustic_vti_model(n)  # c11, c13, c33, buoyancy_*
    segments += [(_P, n + 2)]  # source amplitudes (v's, sigma_h, sigma_v)
    segments += _acoustic_vti_wavefields(n)  # wavefields
    segments += _tri(n, 10, 10, 10)  # backward storage (2 slots x 5 ptrs/block)
    segments += [(_P, n + 2)]  # recorded data (receiver amplitudes)
    segments += _tri(n, 4, 4, 4)  # PML profile coefficients
    segments += [(_P, n + 2)]  # source locations
    segments += [(_P, n + 2)]  # receiver locations
    segments += [(_F, n), (_F, 1)]  # rdx, dt
    segments += [(_I, 2), (_I, n), (_I, n + 2), (_I, n + 2), (_I, 1)]  # sizes
    segments += [(_I, 1), (_Z, 2), (_B, 2 * nf), (_B, 1)]  # storage, bytes, flags
    segments += [(_I, 1), (_I, 2 * n), (_I, 2 * n), (_I, 1), (_P, 1)]
    # start_t, pml, bc, aux, stream (bc mirrors acoustic's bc-threading)
    return _seg(segments)


def get_acoustic_vti_backward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the acoustic VTI backward propagator.

    Adds, relative to the forward template: the ping-ponged "n"-suffixed aux
    copies of the stress-derivative C-PML memories (one per dimension --
    m_sigma_v_z(3-D)/m_sigma_h_y(3-D)/m_sigma_v_y(2-D)/m_sigma_h_x, mirroring
    elastic_vti's ``m_sigma_n`` block) between the wavefields and the
    (read-mode) backward storage, and the model-gradient blocks (twice: the
    direct output and the per-thread scratch).
    """
    if ndim not in (2, 3):
        raise DeepwaveValueError("acoustic_vti supports only ndim 2 and 3")
    n = ndim
    nf = _n_acoustic_vti_flag_models()
    segments: _Segments = []
    segments += _acoustic_vti_model(n)  # model
    segments += [(_P, n + 2)]  # source grad_r
    segments += _acoustic_vti_wavefields(n)  # wavefields
    segments += _tri(n, 1, 1, 1)  # aux "n" ping-pong (stress-deriv memories)
    segments += _tri(n, 10, 10, 10)  # backward storage (read)
    segments += [(_P, n + 2)]  # grad_f
    segments += _acoustic_vti_model(n)  # model gradients
    segments += _acoustic_vti_model(n)  # model gradients (per thread)
    segments += _tri(n, 4, 4, 4)  # PML profile coefficients
    segments += [(_P, n + 2)]  # source locations
    segments += [(_P, n + 2)]  # receiver locations
    segments += [(_F, n), (_F, 1)]  # rdx, dt
    segments += [(_I, 2), (_I, n), (_I, n + 2), (_I, n + 2), (_I, 1)]  # sizes
    segments += [(_I, 1), (_Z, 2), (_B, 2 * nf), (_B, 1)]  # storage, bytes, flags
    segments += [(_I, 1), (_I, 2 * n), (_I, 2 * n), (_I, 1), (_P, 1)]
    # start_t, pml, bc, aux, stream (bc mirrors acoustic's bc-threading)
    return _seg(segments)


def get_scalar_forward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the scalar forward propagator."""
    _check_ndim(ndim)
    n = ndim
    return _seg(
        [
            (_P, 4),
            (_P, n),
            (_P, n),
            (_P, n),  # v, f, wfc, wfp, psi, psin, zeta
            (_P, 4),
            (_P, 1),
            (_P, 1),  # w_store_1a/1b/2/3, w_filenames, r
            (_P, 3 * n),
            (_P, 2),  # a/b/dbdx, sources_i, receivers_i
            (_F, n),
            (_F, n),
            (_F, 1),  # rd, rd2, dt2
            (_I, 2),
            (_I, n),
            (_I, 3),  # nt/n_shots, n, n_src/n_rec/step_ratio
            (_I, 1),
            (_Z, 2),
            (_B, 3),  # storage_mode, byte counts, flags
            (_I, 1),
            (_I, 2 * n),
            (_I, 2 * n),  # start_t, pml, bc
            (_P, 4),
            (_P, 1),
            (_B, 1),
            # raw_store_1a/1b/2/3, raw_filenames, store_raw_forward (see
            # CallbackState role="forward": the compiled scalar backend's opt-in
            # second checkpoint buffer for the backward_callback's reconstructed
            # forward wavefield)
            (_I, 1),
            (_Z, 2),
            (_B, 1),
            # raw storage_mode, raw shot_bytes_uncomp/comp, raw storage_compression
            # (the raw-forward checkpoint's own storage configuration, threaded
            # independently of the primary checkpoint's -- see scalar.py)
            (_I, 1),  # n_threads
            (_P, 1),  # stream
        ]
    )


def get_scalar_backward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the scalar backward propagator."""
    _check_ndim(ndim)
    n = ndim
    return _seg(
        [
            (_P, 4),
            (_P, n),
            (_P, n),
            (_P, n),
            (_P, n),  # fields + psi/psin/zeta/zetan
            (_P, 4),
            (_P, 1),
            (_P, 3),  # w_store, w_filenames, grad_f/v/v_thread
            (_P, 3 * n),
            (_P, 2),  # a/b/dbdx, sources_i, receivers_i
            (_F, n),
            (_F, n),  # rd, rd2
            (_I, 2),
            (_I, n),
            (_I, 3),  # nt/n_shots, n, n_src/n_rec/step_ratio
            (_I, 1),
            (_Z, 2),
            (_B, 3),  # storage_mode, byte counts, flags
            (_I, 1),
            (_I, 2 * n),
            (_I, 2 * n),  # start_t, pml, bc
            (_P, 4),
            (_P, 1),
            (_B, 1),
            # raw_store_1a/1b/2/3, raw_filenames, store_raw_forward (see
            # get_scalar_forward_template; the compiled backward pass now
            # reads this buffer back for CPU/disk-mode raw-forward storage,
            # loading the step the backward_callback is about to fire for --
            # see FUNC(backward) in scalar.c/scalar.cu)
            (_I, 1),
            (_Z, 2),
            (_B, 1),
            # raw storage_mode, raw shot_bytes_uncomp/comp, raw storage_compression
            (_I, 1),  # n_threads
            (_P, 1),  # stream
        ]
    )


def get_scalar_born_forward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the scalar Born forward propagator."""
    _check_ndim(ndim)
    n = ndim
    return _seg(
        [
            (_P, 4),
            (_P, 2),  # v/scatter/f/fsc, wfc/wfp
            (_P, n),
            (_P, n),
            (_P, n),  # psi, psin, zeta
            (_P, 2),
            (_P, n),
            (_P, n),
            (_P, n),  # wfcsc/wfpsc, psisc/psinsc/zetasc
            (_P, 4),
            (_P, 1),
            (_P, 4),
            (_P, 1),  # w_store/filenames, wsc_store/filenames
            (_P, 2),
            (_P, 3 * n),
            (_P, 3),  # r/rsc, a/b/dbdx, source/receiver locations
            (_F, n),
            (_F, n),
            (_F, 1),  # rd, rd2, dt2
            (_I, 2),
            (_I, n),
            (_I, 3),
            (_I, 1),  # nt/n_shots, n, src/rec counts, step_ratio
            (_I, 1),
            (_Z, 2),
            (_B, 5),  # storage_mode, byte counts, flags
            (_I, 1),
            (_I, 2 * n),
            (_I, 2 * n),
            (_I, 1),  # start_t, pml, bc, n_threads
            (_P, 1),  # stream
        ]
    )


def get_scalar_born_backward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the scalar Born backward propagator."""
    _check_ndim(ndim)
    n = ndim
    return _seg(
        [
            (_P, 4),
            (_P, 2),  # v/scatter/grad_r/grad_rsc, wfc/wfp
            (_P, n),
            (_P, n),
            (_P, n),
            (_P, n),  # psi, psin, zeta, zetan
            (_P, 2),
            (_P, n),
            (_P, n),
            (_P, n),
            (_P, n),  # wfcsc/wfpsc + 4 sc aux fields
            (_P, 4),
            (_P, 1),
            (_P, 4),
            (_P, 1),  # w_store/filenames, wsc_store/filenames
            (_P, 6),  # grad_f/fsc/v/scatter + v_thread/scatter_thread
            (_P, 3 * n),
            (_P, 3),  # a/b/dbdx, source/receiver locations
            (_F, n),
            (_F, n),
            (_F, 1),  # rd, rd2, dt2
            (_I, 2),
            (_I, n),
            (_I, 4),
            (_I, 1),  # nt/n_shots, n, src/rec counts, step_ratio
            (_I, 1),
            (_Z, 2),
            (_B, 5),  # storage_mode, byte counts, flags
            (_I, 1),
            (_I, 2 * n),
            (_I, 2 * n),
            (_I, 1),  # start_t, pml, bc, n_threads
            (_P, 1),  # stream
        ]
    )


def get_scalar_born_backward_sc_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the scalar Born backward_sc propagator."""
    _check_ndim(ndim)
    n = ndim
    return _seg(
        [
            (_P, 2),
            (_P, 2),  # v/grad_rsc, wfcsc/wfpsc
            (_P, n),
            (_P, n),
            (_P, n),
            (_P, n),  # psisc, psinsc, zetasc, zetansc
            (_P, 4),
            (_P, 1),
            (_P, 3),  # w_store/filenames, grad_fsc/scatter/scatter_thread
            (_P, 3 * n),
            (_P, 2),  # a/b/dbdx, sources_i, receiverssc_i
            (_F, n),
            (_F, n),
            (_F, 1),  # rd, rd2, dt2
            (_I, 2),
            (_I, n),
            (_I, 2),
            (_I, 1),  # nt/n_shots, n, srcsc/recsc, step_ratio
            (_I, 1),
            (_Z, 2),
            (_B, 4),  # storage_mode, byte counts, flags
            (_I, 1),
            (_I, 2 * n),
            (_I, 2 * n),
            (_I, 1),  # start_t, pml, bc, n_threads
            (_P, 1),  # stream
        ]
    )


def get_elastic_forward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the elastic forward propagator."""
    _check_ndim(ndim)
    n = ndim
    segments: _Segments = []
    segments += _elastic_model(n)  # lamb, mu, (mu_*), buoyancy_*
    segments += [(_P, n + 1)]  # source f
    segments += _tri(n, 14, 9, 4)  # wavefields
    segments += _tri(n, 20, 15, 10)  # backward storage (5 arrays per dimension)
    segments += _tri(n, 1, 1, 2)  # recorded data
    segments += _tri(n, 4, 4, 4)  # PML profile coefficients
    segments += _tri(n, 1, 1, 2)  # source locations
    segments += _tri(n, 1, 1, 2)  # receiver locations
    segments += [(_F, n), (_F, 1)]  # rdx, dt
    segments += [(_I, 2), (_I, n), (_I, n + 1), (_I, n + 1), (_I, 1)]  # sizes
    segments += [(_I, 1), (_Z, 2), (_B, 6), (_B, 1)]  # storage_mode, bytes, flags
    segments += [
        (_I, 1),
        (_I, 2 * n),
        (_I, 2 * n),
        (_I, 1),
        (_P, 1),
    ]  # start_t, pml, bc, aux, stream
    return _seg(segments)


def get_elastic_backward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the elastic backward propagator."""
    _check_ndim(ndim)
    n = ndim
    segments: _Segments = []
    segments += _elastic_model(n)  # model
    segments += [(_P, n + 1)]  # source grad_r
    segments += _tri(n, 14, 9, 4)  # wavefields
    segments += _tri(n, 5, 3, 1)  # m_sigma_n
    segments += _tri(n, 20, 15, 10)  # backward storage
    segments += _tri(n, 1, 1, 2)  # grad_f
    segments += _elastic_model(n)  # model gradients
    segments += _elastic_model(n)  # model gradients (per thread)
    segments += _tri(n, 4, 4, 4)  # PML profile coefficients
    segments += _tri(n, 1, 1, 2)  # source locations
    segments += _tri(n, 1, 1, 2)  # receiver locations
    segments += [(_F, n), (_F, 1)]  # rdx, dt
    segments += [(_I, 2), (_I, n), (_I, n + 1), (_I, n + 1), (_I, 1)]  # sizes
    segments += [(_I, 1), (_Z, 2), (_B, 6), (_B, 1)]  # storage_mode, bytes, flags
    segments += [
        (_I, 1),
        (_I, 2 * n),
        (_I, 2 * n),
        (_I, 1),
        (_P, 1),
    ]  # start_t, pml, bc, aux, stream
    return _seg(segments)


def get_elastic_forward_pm_template(ndim: int) -> list[Any]:
    """Argtype template for elastic's ``DW_PM`` (parameter-modified) forward.

    Identical to :func:`get_elastic_forward_template` but with a trailing
    ``c_batched`` bool and the four per-component surface-stiffness array
    pointers (``c00``, ``c01``, ``c10``, ``c11``) the parameter-modified free
    surface (Cao and Chen, 2018) needs at the surface cells. Appended after
    every other parameter (including ``stream``), so the default symbol's
    signature is unaffected. 2-D only. See ``elastic.c``/``elastic.cu``'s
    ``FUNC(forward)`` under ``DW_PM``.
    """
    return [*get_elastic_forward_template(ndim), _B, _P, _P, _P, _P]


def get_elastic_backward_pm_template(ndim: int) -> list[Any]:
    """Argtype template for elastic's ``DW_PM`` (parameter-modified) backward.

    Trailing block: ``c_batched`` bool, the four surface-stiffness input
    pointers (``c00``..``c11``), their four gradient outputs
    (``grad_c00``..``grad_c11``), and four per-shot gradient temporaries
    (used by the CUDA reduction across shots; the CPU symbol accepts them for
    ABI parity and accumulates directly instead). See ``elastic.c``/
    ``elastic.cu``'s ``FUNC(backward)`` under ``DW_PM``.
    """
    return [*get_elastic_backward_template(ndim), _B, *([_P] * 12)]


def get_elastic_vti_forward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the elastic VTI forward propagator.

    Identical to the isotropic elastic template except for the model pointer
    block (VTI stiffnesses instead of Lame parameters) and the number of
    requires-grad/batched boolean flags (one pair per independent model).
    VTI supports only 2-D and 3-D.
    """
    if ndim not in (2, 3):
        raise DeepwaveValueError("elastic_vti supports only ndim 2 and 3")
    n = ndim
    nf = _n_elastic_vti_flag_models(n)
    segments: _Segments = []
    segments += _elastic_vti_model(n)  # c11, (c12), c13, c33, c*_*, buoyancy_*
    segments += [(_P, n + 1)]  # source f
    segments += _tri(n, 14, 9, 4)  # wavefields
    segments += _tri(n, 20, 15, 10)  # backward storage (5 arrays per dimension)
    segments += _tri(n, 1, 1, 2)  # recorded data
    segments += _tri(n, 4, 4, 4)  # PML profile coefficients
    segments += _tri(n, 1, 1, 2)  # source locations
    segments += _tri(n, 1, 1, 2)  # receiver locations
    segments += [(_F, n), (_F, 1)]  # rdx, dt
    segments += [(_I, 2), (_I, n), (_I, n + 1), (_I, n + 1), (_I, 1)]  # sizes
    segments += [(_I, 1), (_Z, 2), (_B, 2 * nf), (_B, 1)]  # storage, bytes, flags
    segments += [
        (_I, 1),
        (_I, 2 * n),
        (_I, 2 * n),
        (_I, 1),
        (_P, 1),
    ]  # start_t, pml, bc, aux, stream (bc mirrors elastic's bc-threading)
    return _seg(segments)


def get_elastic_vti_backward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the elastic VTI backward propagator."""
    if ndim not in (2, 3):
        raise DeepwaveValueError("elastic_vti supports only ndim 2 and 3")
    n = ndim
    nf = _n_elastic_vti_flag_models(n)
    segments: _Segments = []
    segments += _elastic_vti_model(n)  # model
    segments += [(_P, n + 1)]  # source grad_r
    segments += _tri(n, 14, 9, 4)  # wavefields
    segments += _tri(n, 5, 3, 1)  # m_sigma_n
    segments += _tri(n, 20, 15, 10)  # backward storage
    segments += _tri(n, 1, 1, 2)  # grad_f
    segments += _elastic_vti_model(n)  # model gradients
    segments += _elastic_vti_model(n)  # model gradients (per thread)
    segments += _tri(n, 4, 4, 4)  # PML profile coefficients
    segments += _tri(n, 1, 1, 2)  # source locations
    segments += _tri(n, 1, 1, 2)  # receiver locations
    segments += [(_F, n), (_F, 1)]  # rdx, dt
    segments += [(_I, 2), (_I, n), (_I, n + 1), (_I, n + 1), (_I, 1)]  # sizes
    segments += [(_I, 1), (_Z, 2), (_B, 2 * nf), (_B, 1)]  # storage, bytes, flags
    segments += [
        (_I, 1),
        (_I, 2 * n),
        (_I, 2 * n),
        (_I, 1),
        (_P, 1),
    ]  # start_t, pml, bc, aux, stream (bc mirrors elastic's bc-threading)
    return _seg(segments)


def get_acoustic_forward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the acoustic forward propagator."""
    _check_ndim(ndim)
    n = ndim
    return _seg(
        [
            (_P, 1 + n),
            (_P, 1 + n),  # k/buoyancy, f
            (_P, 1 + 3 * n),  # p, v, phi, psi
            (_P, 5),
            (_P, 5 * n),  # k_store(+filenames), b_store
            (_P, 1 + n),  # receiver_amplitudes
            (_P, 4 * n),  # a, b, ah, bh
            (_P, 1 + n),
            (_P, 1 + n),  # sources_i, receivers_i
            (_F, n + 1),  # rdx, dt
            (_I, 2),
            (_I, n),
            (_I, 1 + n),
            (_I, 1 + n),  # nt/n_shots, n, n_src, n_rec
            (_I, 2),
            (_Z, 2),
            (_B, 5),  # step_ratio/storage_mode, bytes, flags
            (_I, 1),
            (_I, 2 * n),
            (_I, 2 * n),
            (_I, 1),  # start_t, pml, bc, n_threads
            (_P, 1),  # stream
        ]
    )


def get_acoustic_backward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the acoustic backward propagator."""
    _check_ndim(ndim)
    n = ndim
    return _seg(
        [
            (_P, 1 + n),
            (_P, 1 + n),  # k/buoyancy, grad_r
            (_P, 1 + 3 * n),
            (_P, n),  # p/v/phi/psi, psin
            (_P, 5),
            (_P, 5 * n),  # k_store, b_store
            (_P, 1 + n),  # grad_f
            (_P, 2 + 2 * n),  # grad_k, grad_b, grad_k_thread, grad_b_thread
            (_P, 4 * n),  # a, b, ah, bh
            (_P, 1 + n),
            (_P, 1 + n),  # sources_i, receivers_i
            (_F, n + 1),  # rdx, dt
            (_I, 2),
            (_I, n),
            (_I, 1 + n),
            (_I, 1 + n),  # nt/n_shots, n, n_src, n_rec
            (_I, 2),
            (_Z, 2),
            (_B, 5),  # step_ratio/storage_mode, bytes, flags
            (_I, 1),
            (_I, 2 * n),
            (_I, 2 * n),
            (_I, 1),  # start_t, pml, bc, n_threads
            (_P, 1),  # stream
        ]
    )


def get_acoustic_born_forward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the acoustic Born forward propagator.

    Mirrors :func:`get_acoustic_forward_template`, doubled for the
    background/scattered model, source, wavefield, and receiver groups (the
    ``forward`` kernel checkpoints only the background terms, consumed by
    ``backward_sc``).
    """
    _check_ndim(ndim)
    n = ndim
    return _seg(
        [
            (_P, 1 + n),  # k, buoyancy_*
            (_P, 1 + n),  # dk, dbuoyancy_*
            (_P, 1 + n),  # f_p, f_v*
            (_P, 1 + n),  # f_p_sc, f_v*_sc
            (_P, 1 + 3 * n),  # p, v*, phi_*, psi_*
            (_P, 1 + 3 * n),  # psc, v*sc, phi_*_sc, psi_*_sc
            (_P, 5),
            (_P, 5 * n),  # k_grad_store, b*_grad_store (background terms)
            (_P, 5),
            (_P, 5 * n),  # k_grad_sc_store, b*_grad_sc_store (scattered terms)
            (_P, 1 + n),  # r_p, r_v*
            (_P, 1 + n),  # r_p_sc, r_v*_sc
            (_P, 4 * n),  # a, b, ah, bh
            (_P, 1 + n),  # sources_i
            (_P, 1 + n),  # sources_i_sc
            (_P, 1 + n),  # receivers_i
            (_P, 1 + n),  # receivers_i_sc
            (_F, n),
            (_F, 1),  # rdx, dt
            (_I, 2),
            (_I, n),
            (_I, 1 + n),
            (_I, 1 + n),  # nt/n_shots, n, n_src, n_src_sc
            (_I, 1 + n),
            (_I, 1 + n),  # n_rec, n_rec_sc
            (_I, 2),
            (_Z, 2),
            (_B, 7),  # step_ratio/storage_mode, bytes, flags (k/b/dk/db grad,
            # k/b batched, compression)
            (_I, 1),
            (_I, 2 * n),
            (_I, 2 * n),
            (_I, 1),  # start_t, pml, bc, n_threads
            (_P, 1),  # unused
        ]
    )


def get_acoustic_born_backward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the acoustic Born full ``backward`` pass.

    Full two-field adjoint: propagates the background AND scattered adjoint
    fields (with the Born coupling in the background field), consuming both the
    background-term and scattered-term stores, and accumulating background and
    scattered model gradients plus background and scattered source gradients.
    """
    _check_ndim(ndim)
    n = ndim
    return _seg(
        [
            (_P, 1 + n),  # k, buoyancy_*
            (_P, 1 + n),  # dk, dbuoyancy_*
            (_P, 1 + n),  # grad_r_p, grad_r_v* (background residuals)
            (_P, 1 + n),  # grad_r_p_sc, grad_r_v*_sc (scattered residuals)
            (_P, 1 + 3 * n),  # p, v*, phi_*, psi_* (background)
            (_P, n),  # psi_*n (background)
            (_P, 1 + 3 * n),  # psc, v*sc, phi_*_sc, psi_*_sc
            (_P, n),  # psi_*n_sc
            (_P, 5),
            (_P, 5 * n),  # k_grad_store, b*_grad_store (background terms)
            (_P, 5),
            (_P, 5 * n),  # k_grad_sc_store, b*_grad_sc_store (scattered terms)
            (_P, 1 + n),  # grad_f_p, grad_f_v* (background source grads)
            (_P, 1 + n),  # grad_f_p_sc, grad_f_v*_sc (scattered source grads)
            (_P, 1 + n),  # grad_k, grad_b* (background model grads)
            (_P, 1 + n),  # grad_dk, grad_db* (scattered model grads)
            (_P, 1 + n),  # grad_k_thread, grad_b*_thread
            (_P, 1 + n),  # grad_dk_thread, grad_db*_thread
            (_P, 4 * n),  # a, b, ah, bh
            (_P, 1 + n),  # sources_i
            (_P, 1 + n),  # sources_i_sc
            (_P, 1 + n),  # receivers_i
            (_P, 1 + n),  # receivers_i_sc
            (_F, n),
            (_F, 1),  # rdx, dt
            (_I, 2),
            (_I, n),
            (_I, 1 + n),
            (_I, 1 + n),  # nt/n_shots, n, n_src, n_src_sc
            (_I, 1 + n),
            (_I, 1 + n),  # n_rec, n_rec_sc
            (_I, 2),
            (_Z, 2),
            (_B, 9),  # step_ratio/storage_mode, bytes, flags (k/b/dk/db grad,
            # k/b/dk/db batched, compression)
            (_I, 1),
            (_I, 2 * n),
            (_I, 2 * n),
            (_I, 1),  # start_t, pml, bc, n_threads
            (_P, 1),  # unused
        ]
    )


def get_acoustic_born_backward_sc_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the acoustic Born ``backward_sc`` pass.

    Scattered-only adjoint (fixed background K, B_d): propagates only the
    scattered adjoint field and accumulates only ``grad_dK``/``grad_dB_d``.
    """
    _check_ndim(ndim)
    n = ndim
    return _seg(
        [
            (_P, 1 + n),  # k, buoyancy_*
            (_P, 1 + n),  # grad_r_p_sc, grad_r_v*_sc
            (_P, 1 + 3 * n),  # psc, v*sc, phi_*_sc, psi_*_sc
            (_P, n),  # psi_*n_sc
            (_P, 5),
            (_P, 5 * n),  # k_grad_store, b*_grad_store (read-only)
            (_P, 1 + n),  # grad_f_p_sc, grad_f_v*_sc
            (_P, 1 + n),  # grad_dk, grad_db*
            (_P, 1 + n),  # grad_dk_thread, grad_db*_thread
            (_P, 4 * n),  # a, b, ah, bh
            (_P, 1 + n),  # sources_i_sc
            (_P, 1 + n),  # receivers_i_sc
            (_F, n),
            (_F, 1),  # rdx, dt
            (_I, 2),
            (_I, n),
            (_I, 1 + n),
            (_I, 1 + n),  # nt/n_shots, n, n_src_sc, n_rec_sc
            (_I, 2),
            (_Z, 2),
            (_B, 7),  # step_ratio/storage_mode, bytes, flags (incl. dk/db_batched)
            (_I, 1),
            (_I, 2 * n),
            (_I, 2 * n),
            (_I, 1),  # start_t, pml, bc, n_threads
            (_P, 1),  # unused
        ]
    )


def get_elastic_born_forward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the elastic Born forward propagator.

    Coupled background + scattered elastic Born forward pass. Mirrors the
    elastic forward pointer blocks, doubled for the background/scattered model,
    source, wavefield, and receiver groups; source *locations* are shared (a
    single block). Gradient-store checkpointing is engine-style: each of the
    N_BG_STORE (9/5/2 for 3/2/1-D) background finite-difference terms gets its
    own 5-param slot group (``store_1``, ``store_1b``, ``store_2``, ``store_3``,
    ``filenames_ptr``), exactly mirroring :func:`get_elastic_forward_template`'s
    storage block, doubled for the scattered-term analogues; each group is
    written only when ``store_bg``/``store_sc`` (respectively) is true AND the
    chunk-relative ``t % step_ratio == 0``. The forward now accepts and uses
    the same ``step_ratio``/``storage_mode``/``shot_bytes_*``/
    ``storage_compression``/``start_t`` chunk-and-storage-mode machinery as
    every other engine-backed forward (unlike the pre-M1 bespoke full-history
    packed-buffer scheme). See ``elastic_born.c`` and
    ``docs/elastic_born_derivation.md``.
    """
    _check_ndim(ndim)
    n = ndim
    segments: _Segments = []
    segments += _elastic_model(n)  # background models
    segments += _elastic_model(n)  # scattering models
    segments += [(_P, n + 1)]  # background sources f
    segments += [(_P, n + 1)]  # scattered sources f_sc
    segments += _tri(n, 14, 9, 4)  # background wavefields
    segments += _tri(n, 14, 9, 4)  # scattered wavefields
    segments += [(_P, n + 1)]  # background receivers
    segments += [(_P, n + 1)]  # scattered receivers
    segments += _tri(n, 20, 15, 10)  # bg gradient stores (5 params x 9/5/2 slots)
    segments += _tri(n, 20, 15, 10)  # sc gradient stores (5 params x 9/5/2 slots)
    segments += _tri(n, 4, 4, 4)  # PML profile coefficients
    segments += [(_P, n + 1)]  # shared source locations
    segments += [(_P, n + 1)]  # background receiver locations
    segments += [(_P, n + 1)]  # scattered receiver locations
    segments += [(_F, n), (_F, 1)]  # rdx, dt
    segments += [(_I, 2), (_I, n)]  # nt/n_shots, model shape
    segments += [(_I, n + 1), (_I, n + 1)]  # n_sources bg, n_sources sc
    segments += [(_I, n + 1), (_I, n + 1)]  # n_receivers bg, n_receivers sc
    segments += [(_I, 1)]  # step_ratio
    segments += [(_I, 1)]  # storage_mode
    segments += [(_Z, 2)]  # shot_bytes_uncomp, shot_bytes_comp
    segments += [(_B, 6)]  # lamb/mu/buoyancy + dlamb/dmu/dbuoyancy batched
    segments += [(_B, 1)]  # store_bg (checkpoint background gradient stores)
    segments += [(_B, 1)]  # store_sc (checkpoint scattered gradient stores)
    segments += [(_B, 1)]  # storage_compression
    segments += [(_I, 1)]  # start_t
    segments += [(_I, 2 * n)]  # pml (begins then ends)
    segments += [(_I, 2 * n), (_I, 1)]  # bc (begins then ends), n_threads/device
    segments += [(_P, 1)]  # unused (CPU) / CUDA stream
    return _seg(segments)


def get_elastic_born_backward_sc_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the elastic Born ``backward_sc`` pass.

    Scattered-only adjoint (fixed background models): propagates a single
    scattered adjoint field with the plain elastic adjoint operator, seeded by
    the scattered data residuals, and correlates it with the background
    engine-storage-slot stores (loaded per chunk via SETUP_STORE_LOAD, M2) to
    accumulate only the scattering-model gradients (``grad_dlamb``/
    ``grad_dmu``/``grad_dmu_ij``/``grad_dbuoyancy_i``). The ``m_sigma*``
    memory-variable aux ping-pong buffers are now caller-passed (Python swaps
    them across chunks) rather than calloc'd/freed internally per call.
    Mirrors ``elastic_born.c`` FUNC(backward_sc); see
    ``docs/elastic_born_derivation.md`` SS4.1 for the imaging-condition
    derivation. Parameter order for the ``m_sigma*n`` aux ping-pong buffers
    must match ``elastic._elastic_make_aux_wavefields``'s output
    order exactly, since a name-order mismatch is invisible to a type-only
    ABI check and only caught by a multi-chunk correctness test (see
    ``elastic_born.c``'s FUNC(backward) parameter comment).
    """
    _check_ndim(ndim)
    n = ndim
    segments: _Segments = []
    segments += _elastic_model(n)  # background models (read-only)
    segments += [(_P, n + 1)]  # scattered data residuals (adjoint sources)
    segments += _tri(n, 14, 9, 4)  # scattered adjoint wavefields
    segments += _tri(n, 5, 3, 1)  # m_sigma*n aux ping-pong buffers
    segments += _tri(n, 20, 15, 10)  # bg gradient stores (5 params x 9/5/2 slots)
    segments += _elastic_model(n)  # grad scattering models
    segments += _elastic_model(n)  # grad scattering models (thread buffers)
    segments += [(_P, n + 1)]  # grad scattered source amplitudes
    segments += _tri(n, 4, 4, 4)  # PML profile coefficients
    segments += [(_P, n + 1)]  # shared source locations
    segments += [(_P, n + 1)]  # scattered receiver locations
    segments += [(_F, n), (_F, 1)]  # rdx, dt
    segments += [(_I, 2), (_I, n)]  # nt/n_shots, model shape
    segments += [(_I, n + 1), (_I, n + 1)]  # n_sources_sc, n_receivers_sc
    segments += [(_I, 1)]  # step_ratio
    segments += [(_I, 1)]  # storage_mode
    segments += [(_Z, 2)]  # shot_bytes_uncomp, shot_bytes_comp
    # lamb/mu/buoyancy batched + dlamb/dmu/dbuoyancy requires_grad + batched
    segments += [(_B, 9)]
    segments += [(_B, 1)]  # storage_compression
    segments += [(_I, 1)]  # start_t
    segments += [(_I, 2 * n)]  # pml (begins then ends)
    segments += [(_I, 2 * n), (_I, 1)]  # bc (begins then ends), n_threads/device
    segments += [(_P, 1)]  # unused (CPU) / CUDA stream
    return _seg(segments)


def get_elastic_born_backward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the elastic Born full ``backward`` pass.

    Two-field adjoint (background AND scattered). Propagates the scattered
    adjoint field with the plain elastic adjoint (as ``backward_sc``) and the
    background adjoint field with the SS4.2 coupled adjoint (every ``model*Q``
    also picks up ``+ Dmodel*Q_sc``). Accumulates the scattering-model gradients
    (scattered adjoint x ``bg_store``) AND the background-model gradients
    (background adjoint x ``bg_store`` PLUS scattered adjoint x ``sc_store``; two
    terms each, SS4.3), plus the background/scattered source-amplitude and
    initial-wavefield gradients. Engine-style (M3): per-quantity 5-param
    storage slot groups for BOTH ``bg_store`` and ``sc_store`` (doubled
    ``_tri(n, 20, 15, 10)``, mirroring the forward's storage parameters and
    ``backward_sc``'s bg group exactly), caller-passed persistent ``m_sigma*``
    aux ping-pong buffers doubled for the background AND scattered adjoint
    fields independently (each ordered per
    ``deepwave.elastic._elastic_make_aux_wavefields``'s convention -- 3D-only
    block, then 2D+ block, then the always-present element LAST -- checked
    independently for both halves per the M2 aux-ordering lesson), and a real
    ``start_t`` chunk parameter. Mirrors ``elastic_born.c`` FUNC(backward).
    """
    _check_ndim(ndim)
    n = ndim
    segments: _Segments = []
    segments += _elastic_model(n)  # background models (read-only)
    segments += _elastic_model(n)  # scattering models (read-only; SS4.2 coupling)
    segments += [(_P, n + 1)]  # background data residuals (adjoint sources)
    segments += [(_P, n + 1)]  # scattered data residuals (adjoint sources)
    segments += _tri(n, 14, 9, 4)  # background adjoint wavefields
    segments += _tri(n, 14, 9, 4)  # scattered adjoint wavefields
    segments += _tri(n, 5, 3, 1)  # background-field m_sigma*n aux buffers
    segments += _tri(n, 5, 3, 1)  # scattered-field m_sigma*n aux buffers
    segments += _tri(n, 20, 15, 10)  # bg gradient stores (5 params x 9/5/2 slots)
    segments += _tri(n, 20, 15, 10)  # sc gradient stores (5 params x 9/5/2 slots)
    segments += _elastic_model(n)  # grad background models
    segments += _elastic_model(n)  # grad scattering models
    segments += _elastic_model(n)  # grad background models (thread buffers)
    segments += _elastic_model(n)  # grad scattering models (thread buffers)
    segments += [(_P, n + 1)]  # grad background source amplitudes
    segments += [(_P, n + 1)]  # grad scattered source amplitudes
    segments += _tri(n, 4, 4, 4)  # PML profile coefficients
    segments += [(_P, n + 1)]  # shared source locations
    segments += [(_P, n + 1)]  # background receiver locations
    segments += [(_P, n + 1)]  # scattered receiver locations
    segments += [(_F, n), (_F, 1)]  # rdx, dt
    segments += [(_I, 2), (_I, n)]  # nt/n_shots, model shape
    segments += [(_I, n + 1), (_I, n + 1)]  # n_sources bg, n_sources sc
    segments += [(_I, n + 1), (_I, n + 1)]  # n_receivers bg, n_receivers sc
    segments += [(_I, 1)]  # step_ratio
    segments += [(_I, 1)]  # storage_mode
    segments += [(_Z, 2)]  # shot_bytes_uncomp, shot_bytes_comp
    # lamb/mu/buoyancy batched + lamb/mu/buoyancy requires_grad
    # + dlamb/dmu/dbuoyancy requires_grad + dlamb/dmu/dbuoyancy batched
    segments += [(_B, 12)]
    segments += [(_B, 1)]  # storage_compression
    segments += [(_I, 1)]  # start_t
    segments += [(_I, 2 * n)]  # pml (begins then ends)
    segments += [(_I, 2 * n), (_I, 1)]  # bc (begins then ends), n_threads/device
    segments += [(_P, 1)]  # unused (CPU) / CUDA stream
    return _seg(segments)


def get_viscoacoustic_forward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the viscoacoustic forward propagator.

    Like acoustic but with three pressure-node model fields (``m1``, ``c2``,
    ``ar``) instead of the single ``K``, the extra memory-variable wavefield
    ``rmem`` (so ``2 + 3*ndim`` wavefields), and one requires-grad/batched flag
    per independent model group (``kmod`` covering ``m1``+``c2``, ``ar``,
    buoyancy => 3 groups, ``2*3 + 1 = 7`` booleans).
    """
    _check_ndim(ndim)
    n = ndim
    return _seg(
        [
            (_P, 3 + n),  # m1, c2, ar, buoyancy
            (_P, 1 + n),  # f
            (_P, 2 + 3 * n),  # p, rmem, v, phi, psi
            (_P, 5 * (3 + n)),  # m1/c2/ar/buoyancy stores (+filenames)
            (_P, 1 + n),  # receiver_amplitudes
            (_P, 4 * n),  # a, b, ah, bh
            (_P, 1 + n),
            (_P, 1 + n),  # sources_i, receivers_i
            (_F, n + 1),  # rdx, dt
            (_I, 2),
            (_I, n),
            (_I, 1 + n),
            (_I, 1 + n),  # nt/n_shots, n, n_src, n_rec
            (_I, 2),
            (_Z, 2),
            (_B, 7),  # step_ratio/storage_mode, bytes, flags
            (_I, 1),
            (_I, 2 * n),
            (_I, 2 * n),
            (_I, 1),  # start_t, pml, bc, n_threads
            (_P, 1),  # stream
        ]
    )


def get_viscoacoustic_backward_template(ndim: int) -> list[Any]:
    """Returns the argtype template for the viscoacoustic backward propagator."""
    _check_ndim(ndim)
    n = ndim
    return _seg(
        [
            (_P, 3 + n),  # m1, c2, ar, buoyancy
            (_P, 1 + n),  # grad_r (adjoint sources at receivers)
            (_P, 2 + 3 * n),  # p, rmem, v, phi, psi
            (_P, n),  # psin
            (_P, 5 * (3 + n)),  # stores
            (_P, 1 + n),  # grad_f
            (_P, 2 * (3 + n)),  # grad_models, grad_models_thread
            (_P, 4 * n),  # a, b, ah, bh
            (_P, 1 + n),
            (_P, 1 + n),  # sources_i, receivers_i
            (_F, n + 1),  # rdx, dt
            (_I, 2),
            (_I, n),
            (_I, 1 + n),
            (_I, 1 + n),  # nt/n_shots, n, n_src, n_rec
            (_I, 2),
            (_Z, 2),
            (_B, 7),  # step_ratio/storage_mode, bytes, flags
            (_I, 1),
            (_I, 2 * n),
            (_I, 2 * n),
            (_I, 1),  # start_t, pml, bc, n_threads
            (_P, 1),  # stream
        ]
    )


def _symbol_family(propagator: str) -> str:
    """The symbol-name family for a propagator.

    The historical propagators carry an ``_iso`` (isotropic) infix in their
    exported symbol names (e.g. ``elastic_iso_2d_4_float_forward_cpu``).
    Anisotropic propagators name their symmetry class directly in the family
    name instead (e.g. ``elastic_vti_2d_4_float_forward_cpu``).
    """
    if propagator in ("elastic_vti", "acoustic_vti"):
        return propagator
    return f"{propagator}_iso"


def _get_argtypes(template_name: str, ndim: int, float_type: type) -> list[Any]:
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

    family = _symbol_family(propagator)
    for device in ["cpu", "cuda"]:
        func_name = f"{family}_{ndim}d_{accuracy}_{dtype}_{direction}{extra}_{device}"
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
        raise DeepwaveTypeError(f"Unsupported dtype {dtype}")

    device_str = device.type

    func_name = (
        f"{_symbol_family(propagator)}_{ndim}d_{accuracy}_{dtype_str}_"
        f"{pass_name}{extra}_{device_str}"
    )

    try:
        return getattr(dll, func_name)
    except AttributeError as e:
        raise AttributeError(f"Backend function {func_name} not found.") from e


# --- Propagator registry ----------------------------------------------------
#
# Everything above this point is per-propagator, ABI-critical segment layout
# (the hand-written ``get_*_template`` functions) that genuinely differs
# between propagators and cannot be mechanically derived. Everything below
# is bookkeeping: which (direction, variant-suffix) combinations exist for
# each propagator, at which ndim, and how their argtype lists are built from
# the templates above (directly, by aliasing another variant unchanged, or by
# appending one mechanical trailing argument). A single ``PROPAGATOR_REGISTRY``
# is the one place this bookkeeping is expressed; the ``templates`` dict used
# by :func:`_get_argtypes` and the registration loop that assigns argtypes to
# the loaded ``dll`` symbols are both derived from it.


@dataclass(frozen=True)
class Variant:
    """One buildable (direction, variant-suffix) combination for a propagator.

    ``suffix`` is ``""`` for a propagator's base direction, else e.g.
    ``"_mpml"``, ``"_pm"``, ``"_gsls"``. Exactly one of ``template``,
    ``alias_of``, or ``appended_arg`` must be set:

    * ``template`` -- a hand-written, ABI-critical segment layout: either a
      base (``suffix == ""``) direction, or a real non-mechanical variant
      (currently only elastic's ``_pm``, which appends more than one
      argument and so cannot be expressed via ``appended_arg``).
    * ``alias_of`` -- reuse another variant's template completely unchanged
      (the M-PML case: same ctypes pointers, different array contents/extent
      at runtime). Names the ``f"{direction}{suffix}"`` key of another
      ``Variant`` already in the same :class:`PropagatorSpec` (in every
      current case, that variant's own suffix-less base direction).
    * ``appended_arg`` -- the mechanical case (viscoacoustic's ``_gsls``
      appending the trailing ``n_mechanisms`` int64): the resolved template
      is ``[*base(ndim), appended_arg]``, where ``base`` is this variant's
      own suffix-stripped sibling (same ``direction``, ``suffix=""``) in the
      same :class:`PropagatorSpec`.
    """

    direction: str
    suffix: str = ""
    ndim_range: frozenset[int] = frozenset({1, 2, 3})
    template: Optional[Callable[[int], list[Any]]] = None
    alias_of: Optional[str] = None
    appended_arg: Optional[Any] = None

    def __post_init__(self) -> None:
        n_set = sum(
            x is not None for x in (self.template, self.alias_of, self.appended_arg)
        )
        if n_set != 1:
            raise DeepwaveValueError(
                f"Variant(direction={self.direction!r}, suffix={self.suffix!r}) "
                "must set exactly one of template/alias_of/appended_arg, got "
                f"{n_set}"
            )


@dataclass(frozen=True)
class PropagatorSpec:
    """All the buildable (ndim, direction, variant) combinations for a propagator.

    ``leading_args`` additionally names this propagator's "leading" (i.e.
    pre-tensor-block) positional arguments, in the order its ``*_func``
    dispatcher (e.g. ``scalar_func``) receives them from its public call
    site -- the same order the compiled-backend reshuffle in that dispatcher
    reads named values (``boundary``, the first model tensor, ...) out of by
    position. See :func:`leading_arg_offset`.
    """

    name: str
    symbol_family: str
    ndim_range: frozenset[int]
    directions: tuple[str, ...]
    variants: tuple[Variant, ...]
    leading_args: tuple[str, ...] = ()


def _variant_key(variant: Variant) -> str:
    """The ``f"{direction}{suffix}"`` key identifying a variant within its spec."""
    return f"{variant.direction}{variant.suffix}"


def _find_variant(spec: PropagatorSpec, key: str) -> Variant:
    for variant in spec.variants:
        if _variant_key(variant) == key:
            return variant
    raise KeyError(f"{spec.name}: no variant with key {key!r}")


def _resolve_variant_template(
    spec: PropagatorSpec, variant: Variant
) -> Callable[[int], list[Any]]:
    """Resolve a :class:`Variant` to a concrete ``(ndim) -> argtypes`` callable."""
    if variant.template is not None:
        return variant.template
    if variant.alias_of is not None:
        return _resolve_variant_template(spec, _find_variant(spec, variant.alias_of))
    if variant.appended_arg is not None:
        # This variant's own suffix-less base (same direction) supplies the
        # segment layout that the mechanical trailing argument is appended to.
        base_template = _resolve_variant_template(
            spec, _find_variant(spec, variant.direction)
        )
        appended_arg = variant.appended_arg

        def _mechanical_template(ndim: int) -> list[Any]:
            return [*base_template(ndim), appended_arg]

        return _mechanical_template
    raise AssertionError(  # pragma: no cover -- Variant.__post_init__ enforces this
        "unreachable: Variant must set template, alias_of, or appended_arg"
    )


# Shared ``leading_args`` shapes, reused verbatim by several propagators
# below (see ``PropagatorSpec.leading_args``/``leading_arg_offset``).
#
# ``_LEADING_ARGS_15``: scalar, scalar_born, and acoustic all thread
# ``boundary`` straight through as the 8th leading argument (index 7, right
# after ``pml_width``), with no M-PML slot.
_LEADING_ARGS_15: tuple[str, ...] = (
    "pml_profiles",
    "grid_spacing",
    "dt",
    "nt",
    "step_ratio",
    "accuracy",
    "pml_width",
    "boundary",
    "n_shots",
    "forward_callback",
    "backward_callback",
    "callback_frequency",
    "storage_mode",
    "storage_path",
    "storage_compression",
)

# ``_LEADING_ARGS_16_MPML``: elastic and elastic_vti add a trailing M-PML
# flag (index 15) after ``storage_compression``, but otherwise match
# ``_LEADING_ARGS_15`` (``boundary`` still at index 7).
_LEADING_ARGS_16_MPML: tuple[str, ...] = (*_LEADING_ARGS_15, "mpml")

# ``_LEADING_ARGS_16_RAW_FORWARD``: scalar alone adds a trailing flag (index
# 15) after ``storage_compression``, opting the second (raw forward)
# wavefield history in. It is scalar's alone because scalar is the only
# propagator that can serve ``CallbackState.get_wavefield(role="forward")``.
_LEADING_ARGS_16_RAW_FORWARD: tuple[str, ...] = (
    *_LEADING_ARGS_15,
    "backward_callback_needs_forward_wavefield",
)


PROPAGATOR_REGISTRY: dict[str, PropagatorSpec] = {
    "scalar": PropagatorSpec(
        name="scalar",
        symbol_family=_symbol_family("scalar"),
        ndim_range=frozenset({1, 2, 3}),
        directions=("forward", "backward"),
        leading_args=_LEADING_ARGS_16_RAW_FORWARD,
        variants=(
            Variant(direction="forward", template=get_scalar_forward_template),
            Variant(direction="backward", template=get_scalar_backward_template),
        ),
    ),
    "scalar_born": PropagatorSpec(
        name="scalar_born",
        symbol_family=_symbol_family("scalar_born"),
        ndim_range=frozenset({1, 2, 3}),
        directions=("forward", "backward", "backward_sc"),
        leading_args=_LEADING_ARGS_15,
        variants=(
            Variant(direction="forward", template=get_scalar_born_forward_template),
            Variant(direction="backward", template=get_scalar_born_backward_template),
            # "backward_sc" is scalar_born's third compiled entry point (the
            # scattered-only adjoint), treated as its own direction: its
            # symbol name and template-dict key are exactly
            # "backward"+"_sc" concatenated, which is indistinguishable from
            # a direction literally spelled "backward_sc".
            Variant(
                direction="backward_sc",
                template=get_scalar_born_backward_sc_template,
            ),
        ),
    ),
    "elastic": PropagatorSpec(
        name="elastic",
        symbol_family=_symbol_family("elastic"),
        ndim_range=frozenset({1, 2, 3}),
        directions=("forward", "backward"),
        leading_args=_LEADING_ARGS_16_MPML,
        variants=(
            Variant(direction="forward", template=get_elastic_forward_template),
            # Multiaxial-PML variant: shares the elastic signature exactly
            # (the PML profile arrays are the same ctypes pointers; only
            # their contents/extent differ), so it aliases the base
            # template. Built only for ndim >= 2.
            Variant(
                direction="forward",
                suffix="_mpml",
                alias_of="forward",
                ndim_range=frozenset({2, 3}),
            ),
            # Parameter-modified free-surface (DW_PM) variant: adds trailing
            # per-component surface-stiffness arrays, so it needs its own
            # template (not mechanical -- more than one appended argument).
            # 2-D only (matches the eager scope and the CMake NDIM == 2
            # build guard).
            Variant(
                direction="forward",
                suffix="_pm",
                template=get_elastic_forward_pm_template,
                ndim_range=frozenset({2}),
            ),
            Variant(direction="backward", template=get_elastic_backward_template),
            Variant(
                direction="backward",
                suffix="_mpml",
                alias_of="backward",
                ndim_range=frozenset({2, 3}),
            ),
            Variant(
                direction="backward",
                suffix="_pm",
                template=get_elastic_backward_pm_template,
                ndim_range=frozenset({2}),
            ),
        ),
    ),
    "elastic_vti": PropagatorSpec(
        name="elastic_vti",
        symbol_family=_symbol_family("elastic_vti"),
        # VTI has no 1-D form (anisotropy about the vertical axis is
        # meaningless with a single spatial dimension).
        ndim_range=frozenset({2, 3}),
        directions=("forward", "backward"),
        leading_args=_LEADING_ARGS_16_MPML,
        variants=(
            Variant(direction="forward", template=get_elastic_vti_forward_template),
            Variant(direction="forward", suffix="_mpml", alias_of="forward"),
            Variant(direction="backward", template=get_elastic_vti_backward_template),
            Variant(direction="backward", suffix="_mpml", alias_of="backward"),
        ),
    ),
    "acoustic_vti": PropagatorSpec(
        name="acoustic_vti",
        symbol_family=_symbol_family("acoustic_vti"),
        ndim_range=frozenset({2, 3}),
        directions=("forward", "backward"),
        # Unlike elastic/elastic_vti/viscoacoustic, acoustic_vti's call site
        # places ``n_shots`` right after ``pml_width`` (index 7) and
        # ``boundary`` last (index 15, after ``mpml``) -- already in the
        # position the compiled backend wants, so acoustic_vti_func needs no
        # leading-argument reshuffle at all.
        leading_args=(
            "pml_profiles",
            "grid_spacing",
            "dt",
            "nt",
            "step_ratio",
            "accuracy",
            "pml_width",
            "n_shots",
            "forward_callback",
            "backward_callback",
            "callback_frequency",
            "storage_mode",
            "storage_path",
            "storage_compression",
            "mpml",
            "boundary",
        ),
        variants=(
            Variant(direction="forward", template=get_acoustic_vti_forward_template),
            Variant(direction="forward", suffix="_mpml", alias_of="forward"),
            Variant(direction="backward", template=get_acoustic_vti_backward_template),
            Variant(direction="backward", suffix="_mpml", alias_of="backward"),
        ),
    ),
    "acoustic": PropagatorSpec(
        name="acoustic",
        symbol_family=_symbol_family("acoustic"),
        ndim_range=frozenset({1, 2, 3}),
        directions=("forward", "backward"),
        leading_args=_LEADING_ARGS_15,
        variants=(
            Variant(direction="forward", template=get_acoustic_forward_template),
            Variant(direction="backward", template=get_acoustic_backward_template),
        ),
    ),
    "acoustic_born": PropagatorSpec(
        name="acoustic_born",
        symbol_family=_symbol_family("acoustic_born"),
        ndim_range=frozenset({1, 2, 3}),
        directions=("forward", "backward", "backward_sc"),
        # Unlike the other 8 propagators, ``acoustic_born_func``/
        # ``AcousticBornForwardFunc.forward`` receive these already in their
        # final order (built directly by named keyword at the
        # ``acoustic_born()`` call site) -- there is no positional reshuffle
        # to do.
        leading_args=(
            "pml_profiles",
            "grid_spacing",
            "dt",
            "nt",
            "step_ratio",
            "accuracy",
            "pml_width",
            "n_shots",
            "forward_callback",
            "backward_callback",
            "callback_frequency",
            "storage_mode",
            "storage_path",
            "storage_compression",
            "boundary",
        ),
        variants=(
            Variant(direction="forward", template=get_acoustic_born_forward_template),
            Variant(direction="backward", template=get_acoustic_born_backward_template),
            Variant(
                direction="backward_sc",
                template=get_acoustic_born_backward_sc_template,
            ),
        ),
    ),
    "elastic_born": PropagatorSpec(
        name="elastic_born",
        symbol_family=_symbol_family("elastic_born"),
        ndim_range=frozenset({1, 2, 3}),
        directions=("forward", "backward", "backward_sc"),
        # See acoustic_born's leading_args comment above -- elastic_born_func/
        # ElasticBornForwardFunc.forward likewise receive these already in
        # final order, with no positional reshuffle, and ``mpml`` as a real
        # trailing leading argument.
        leading_args=(
            "pml_profiles",
            "grid_spacing",
            "dt",
            "nt",
            "step_ratio",
            "accuracy",
            "pml_width",
            "n_shots",
            "forward_callback",
            "backward_callback",
            "callback_frequency",
            "storage_mode",
            "storage_path",
            "storage_compression",
            "mpml",
            "boundary",
        ),
        variants=(
            Variant(direction="forward", template=get_elastic_born_forward_template),
            Variant(direction="backward", template=get_elastic_born_backward_template),
            Variant(
                direction="backward_sc",
                template=get_elastic_born_backward_sc_template,
            ),
        ),
    ),
    "viscoacoustic": PropagatorSpec(
        name="viscoacoustic",
        symbol_family=_symbol_family("viscoacoustic"),
        ndim_range=frozenset({1, 2, 3}),
        directions=("forward", "backward"),
        # ``viscoacoustic_func`` also has a ``q_n_mechanisms`` parameter, but
        # (like ``backend``) it is its own named parameter, not part
        # of the flat leading-args block below.
        leading_args=_LEADING_ARGS_15,
        variants=(
            Variant(direction="forward", template=get_viscoacoustic_forward_template),
            # Multi-mechanism GSLS (DW_GSLS) variant: adds one trailing
            # int64 (n_mechanisms). Its own ndim_range is unrestricted (the
            # default), matching viscoacoustic's own ndim support (1-3).
            Variant(direction="forward", suffix="_gsls", appended_arg=_I),
            Variant(direction="backward", template=get_viscoacoustic_backward_template),
            Variant(direction="backward", suffix="_gsls", appended_arg=_I),
        ),
    ),
}


def leading_arg_offset(propagator: str, name: str) -> int:
    """Index of a named leading argument in a propagator's flat args tuple.

    Looks up ``name`` in ``PROPAGATOR_REGISTRY[propagator].leading_args``
    -- the ordered names of the leading (pre-tensor-block) positional
    arguments a propagator's ``*_func`` dispatcher receives from its public
    call site. Replaces hardcoded integer literals (e.g. ``args[7]``) in the
    compiled-backend reshuffle each dispatcher does, with a self-documenting
    lookup, so a future reordering of a propagator's leading arguments
    cannot silently desynchronise from the hardcoded indices that used to
    read them.

    Args:
        propagator: A key of :data:`PROPAGATOR_REGISTRY`.
        name: One of that propagator's ``leading_args`` names.

    Returns:
        The 0-based index of ``name`` within the flat leading-args block.

    Raises:
        KeyError: If ``propagator`` is not in :data:`PROPAGATOR_REGISTRY`.
        ValueError: If ``name`` is not one of its ``leading_args``.
    """
    return PROPAGATOR_REGISTRY[propagator].leading_args.index(name)


def _build_templates(
    registry: dict[str, PropagatorSpec],
) -> dict[str, Callable[[int], list[Any]]]:
    """Resolve every ``Variant`` in ``registry`` into the ``templates`` dict shape.

    Keyed exactly as the old hand-maintained dict was --
    ``f"{propagator}_{direction}{suffix}"`` -- so :func:`_get_argtypes` and
    :func:`_assign_argtypes` need no changes to their call shape.
    """
    result: dict[str, Callable[[int], list[Any]]] = {}
    for spec in registry.values():
        for variant in spec.variants:
            key = f"{spec.name}_{_variant_key(variant)}"
            result[key] = _resolve_variant_template(spec, variant)
    return result


# A dictionary to hold all the template generator functions, derived from
# PROPAGATOR_REGISTRY (see _build_templates).
templates: dict[str, Callable[[int], list[Any]]] = _build_templates(PROPAGATOR_REGISTRY)


# Now, iterate through every propagator, direction, and variant in the
# registry (restricted to each variant's supported ndim), and every
# accuracy/dtype, to assign the correct argtypes to the dynamically loaded
# C/CUDA functions. The function names are constructed based on the naming
# convention used in the CMake build system (e.g.,
# scalar_iso_2d_4_float_forward_cpu).
for _spec in PROPAGATOR_REGISTRY.values():
    for _variant in _spec.variants:
        for current_ndim in sorted(_spec.ndim_range & _variant.ndim_range):
            for current_accuracy in [2, 4, 6, 8]:
                for current_dtype in ["float", "double"]:
                    _assign_argtypes(
                        _spec.name,
                        current_ndim,
                        current_accuracy,
                        current_dtype,
                        _variant.direction,
                        extra=_variant.suffix,
                    )
