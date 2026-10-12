"""Generate, symbolically verify, and compile kernels for a set of variants."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
import subprocess
import time

from . import abi, emit_c, emit_cuda, model, symcheck
from .spec import PhysicsSpec

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "runtime"
OUT = pathlib.Path(os.environ.get("DWSPIKE_BUILD", "/tmp/dwspike-build"))

CFLAGS = [
    "-O3",
    "-ffast-math",
    "-fno-math-errno",
    "-march=native",
    "-fPIC",
    "-std=c11",
    "-Wall",
    "-pthread",
    # The generated kernels load many __restrict pointers out of a context
    # struct; GCC honours those only via runtime alias versioning, and its
    # default cap of 10 checks silently disables vectorisation of the
    # interior stencil.
    "--param=vect-max-version-for-alias-checks=200",
]


@dataclasses.dataclass
class Variant:
    spec: PhysicsSpec
    accuracy: int
    dtype: str  # 'f32' | 'f64'

    @property
    def tag(self) -> str:
        return f"{self.spec.name}_{self.spec.ndim}d_a{self.accuracy}_{self.dtype}"


NVCC = "/usr/local/cuda/bin/nvcc"
NVFLAGS = [
    "-O3",
    "--use_fast_math",
    "-fatbin",
    "-gencode",
    "arch=compute_86,code=sm_86",
    "-gencode",
    "arch=compute_89,code=sm_89",
    "-gencode",
    "arch=compute_89,code=compute_89",
]


@dataclasses.dataclass
class BuildResult:
    lib: pathlib.Path
    sources: list[pathlib.Path]
    gen_seconds: float
    check_seconds: float
    compile_seconds: float
    generated_lines: int
    lib_bytes: int
    fatbins: list[pathlib.Path] = dataclasses.field(default_factory=list)
    fatbin_bytes: int = 0
    cuda_compile_seconds: float = 0.0


def generate(
    variants: list[Variant],
    verify: bool = True,
    active_size: int = 4,
    cuda: bool = False,
):
    OUT.mkdir(parents=True, exist_ok=True)
    srcs: list[pathlib.Path] = []
    cusrcs: list[pathlib.Path] = []
    lines = 0
    t_gen = t_chk = 0.0
    checked: set[tuple[str, int, int]] = set()
    for v in variants:
        t0 = time.perf_counter()
        m = model.build(v.spec, v.accuracy)
        src = emit_c.emit(m, v.dtype)
        t_gen += time.perf_counter() - t0
        key = (v.spec.name, v.spec.ndim, v.accuracy)
        if verify and key not in checked:
            checked.add(key)
            t0 = time.perf_counter()
            errs = symcheck.run(m, active_size=active_size)
            t_chk += time.perf_counter() - t0
            if errs:
                raise AssertionError(
                    f"doubly-derived adjoint check FAILED for {v.tag} "
                    f"({len(errs)} mismatches)\n" + "\n".join(errs[:10])
                )
        p = OUT / f"{v.tag}.c"
        p.write_text(src)
        lines += len(src.splitlines())
        srcs.append(p)
        if cuda:
            t0 = time.perf_counter()
            hdr, cu, host = emit_cuda.emit(m, v.dtype, v.tag)
            t_gen += time.perf_counter() - t0
            (OUT / f"{v.tag}_p.h").write_text(hdr)
            (OUT / f"{v.tag}.cu").write_text(cu)
            (OUT / f"{v.tag}_host.c").write_text(host)
            lines += (
                len(cu.splitlines()) + len(host.splitlines()) + len(hdr.splitlines())
            )
            cusrcs.append(OUT / f"{v.tag}.cu")
            srcs.append(OUT / f"{v.tag}_host.c")
    return srcs, cusrcs, lines, t_gen, t_chk


def compile_lib(
    srcs: list[pathlib.Path], name: str = "libdwspike.so", jobs: int = 4
) -> tuple[pathlib.Path, float]:
    lib = OUT / name
    objs = []
    t0 = time.perf_counter()
    procs = []
    for s in srcs:
        o = OUT / (s.stem + ".o")
        objs.append(o)
        cmd = ["gcc", *CFLAGS, f"-I{RUNTIME}", f"-I{OUT}", "-c", str(s), "-o", str(o)]
        procs.append(subprocess.Popen(cmd))
        while len([p for p in procs if p.poll() is None]) >= jobs:
            time.sleep(0.02)
    for p in procs:
        if p.wait() != 0:
            raise RuntimeError("compilation failed")
    cmd = [
        "gcc",
        "-shared",
        "-o",
        str(lib),
        *[str(o) for o in objs],
        "-pthread",
        "-lm",
        "-ldl",
    ]
    if subprocess.run(cmd).returncode != 0:
        raise RuntimeError("link failed")
    return lib, time.perf_counter() - t0


def compile_fatbins(cusrcs: list[pathlib.Path], jobs: int = 2):
    t0 = time.perf_counter()
    outs = []
    procs = []
    for s in cusrcs:
        o = s.with_suffix(".fatbin")
        outs.append(o)
        procs.append(subprocess.Popen([NVCC, *NVFLAGS, "-o", str(o), str(s)]))
        while len([p for p in procs if p.poll() is None]) >= jobs:
            time.sleep(0.05)
    for p in procs:
        if p.wait() != 0:
            raise RuntimeError("nvcc failed")
    return outs, time.perf_counter() - t0


def build(
    variants: list[Variant], verify: bool = True, jobs: int = 4, cuda: bool = False
) -> BuildResult:
    srcs, cusrcs, lines, t_gen, t_chk = generate(variants, verify=verify, cuda=cuda)
    fatbins, t_cu = ([], 0.0)
    if cuda:
        fatbins, t_cu = compile_fatbins(cusrcs, jobs=2)
        srcs = srcs + [RUNTIME / "dw_cuda.c"]
    lib, t_c = compile_lib(srcs, jobs=jobs)
    return BuildResult(
        lib=lib,
        sources=srcs,
        gen_seconds=t_gen,
        check_seconds=t_chk,
        compile_seconds=t_c,
        generated_lines=lines,
        lib_bytes=lib.stat().st_size,
        fatbins=fatbins,
        fatbin_bytes=sum(f.stat().st_size for f in fatbins),
        cuda_compile_seconds=t_cu,
    )
