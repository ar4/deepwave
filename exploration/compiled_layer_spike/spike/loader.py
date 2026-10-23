"""ctypes loading of the generated kernel library.

The compiled library has a flat ``extern "C"`` ABI and links nothing but libc,
libm and libpthread (plus libcuda, via dlopen, for the GPU variants), so no
part of it depends on the torch ABI, on ATen or on ``Python.h``.
"""

from __future__ import annotations

import ctypes
import functools
import pathlib

from dwgen import abi, build as buildmod, model as modelmod
from dwgen.spec import PhysicsSpec

_CT = {
    "ptr": ctypes.c_void_p,
    "cptr": ctypes.c_void_p,
    "iptr": ctypes.c_void_p,
    "vptr": ctypes.c_void_p,
    "i64": ctypes.c_int64,
}


class Kernel:
    def __init__(self, fn, args: list[abi.Arg], dtype: str):
        self.fn = fn
        self.args = args
        self.dtype = dtype

    def __call__(self, values: list) -> int:
        return self.fn(*values)


class Library:
    def __init__(
        self,
        path: pathlib.Path,
        variants: list[buildmod.Variant],
        devices=("cpu",),
        fatbins=(),
    ):
        self.path = path
        self.dll = ctypes.CDLL(str(path))
        self.kernels: dict[tuple, Kernel] = {}
        self.models: dict[tuple, modelmod.Model] = {}
        if fatbins:
            init = self.dll.dw_cuda_init
            init.argtypes = [ctypes.c_char_p]
            init.restype = ctypes.c_int
            for fb in fatbins:
                rc = init(str(fb).encode())
                if rc != 0:
                    raise RuntimeError(f"dw_cuda_init({fb}) -> {rc}")
        for v in variants:
            m = modelmod.build(v.spec, v.accuracy)
            self.models[(v.spec.name, v.spec.ndim, v.accuracy, v.dtype)] = m
            for direction in ("fwd", "bwd"):
                for device in devices:
                    name = abi.kernel_name(
                        v.spec, v.accuracy, v.dtype, direction, device
                    )
                    fn = getattr(self.dll, name)
                    args = abi.signature(v.spec, m, direction, device)
                    ftype = ctypes.c_float if v.dtype == "f32" else ctypes.c_double
                    fn.argtypes = [
                        ftype if a.kind == "T" else _CT[a.kind] for a in args
                    ]
                    fn.restype = ctypes.c_int
                    self.kernels[
                        (
                            v.spec.name,
                            v.spec.ndim,
                            v.accuracy,
                            v.dtype,
                            direction,
                            device,
                        )
                    ] = Kernel(fn, args, v.dtype)


_LIB: Library | None = None


def get_library(
    variants: list[buildmod.Variant], verify: bool = True, cuda: bool = False
) -> Library:
    global _LIB
    if _LIB is None:
        res = buildmod.build(variants, verify=verify, cuda=cuda)
        devices = ("cpu", "cuda") if cuda else ("cpu",)
        _LIB = Library(res.lib, variants, devices, res.fatbins)
        _LIB.build_result = res  # type: ignore[attr-defined]
    return _LIB
