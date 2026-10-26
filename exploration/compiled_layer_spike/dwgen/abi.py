"""The flat, extern "C" ABI shared by the C/CUDA emitters and the ctypes loader.

Nothing here (or in the emitted source) references libtorch, ATen, Python.h or
libcudart: every buffer arrives as a raw pointer and every shape as an
``int64_t``.  The signature is derived from the physics spec, so the emitter
and the Python binding cannot drift apart.
"""

from __future__ import annotations

import dataclasses

from .spec import PhysicsSpec


@dataclasses.dataclass(frozen=True)
class Arg:
    kind: str  # 'ptr', 'cptr', 'iptr', 'T', 'i64'
    name: str


def kernel_name(
    spec: PhysicsSpec, accuracy: int, dtype: str, direction: str, device: str
) -> str:
    return f"dw_{spec.name}_{spec.ndim}d_a{accuracy}_{dtype}_{direction}_{device}"


def signature(spec: PhysicsSpec, m, direction: str, device: str = "cpu") -> list[Arg]:
    a: list[Arg] = []
    for f in spec.state:
        a.append(Arg("ptr", f"{f}_cur"))
        a.append(Arg("ptr", f"{f}_nxt"))
    for p in spec.params:
        a.append(Arg("cptr", f"par_{p}"))
    for pa in spec.profile_arrays():
        a.append(Arg("cptr", f"prof_{pa}"))
    for s in m.stores:
        a.append(Arg("ptr", f"st_{s}"))
    if direction == "fwd":
        for k in range(spec.n_sources):
            a.append(Arg("cptr", f"src{k}_amp"))
            a.append(Arg("iptr", f"src{k}_i"))
        for k in range(spec.n_receivers):
            a.append(Arg("ptr", f"rec{k}_out"))
            a.append(Arg("iptr", f"rec{k}_i"))
    else:
        for k in range(spec.n_receivers):
            a.append(Arg("cptr", f"rec{k}_grad"))
            a.append(Arg("iptr", f"rec{k}_i"))
        for k in range(spec.n_sources):
            a.append(Arg("ptr", f"src{k}_grad"))
            a.append(Arg("iptr", f"src{k}_i"))
        for p in spec.params:
            a.append(Arg("ptr", f"grad_{p}"))
    for s in spec.scalars:
        a.append(Arg("T", s))
    a.append(Arg("i64", "nt"))
    a.append(Arg("i64", "n_shots"))
    for ax in spec.axis_names:
        a.append(Arg("i64", f"n{ax}"))
    a.append(Arg("i64", "par_batched"))
    a.append(Arg("i64", "step_ratio"))
    a.append(Arg("i64", "store_all_steps"))
    for k in range(spec.n_sources):
        a.append(Arg("i64", f"n_src{k}"))
    for k in range(spec.n_receivers):
        a.append(Arg("i64", f"n_rec{k}"))
    for ax in spec.axis_names:
        a.append(Arg("i64", f"pml_b_{ax}"))
    for ax in spec.axis_names:
        a.append(Arg("i64", f"pml_e_{ax}"))
    a.append(Arg("i64", "need_store"))
    a.append(Arg("i64", "n_threads"))
    if device == "cuda":
        a.append(Arg("vptr", "stream_ptr"))
    return a


C_TYPE = {
    "ptr": "{T} *__restrict const",
    "cptr": "{T} const *__restrict const",
    "iptr": "int64_t const *__restrict const",
    "T": "{T} const",
    "i64": "int64_t const",
    "vptr": "void *const",
}


def c_params(args: list[Arg], ctype: str) -> str:
    return ",\n    ".join(C_TYPE[a.kind].format(T=ctype) + " " + a.name for a in args)
