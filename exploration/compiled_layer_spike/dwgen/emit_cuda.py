"""Emit CUDA device code plus a plain-C driver-API host driver.

Three artefacts per variant:

* ``<tag>_p.h``       the kernel-parameter struct, shared verbatim by the .cu
                      and the host .c so their ABI cannot drift.
* ``<tag>.cu``        ``extern "C" __global__`` kernels only -- compiled to a
                      fatbin, never linked against anything.
* ``<tag>_host.c``    the time loop, reaching CUDA through libcuda.so.1 with
                      the driver API (``cuModuleLoadData``/``cuLaunchKernel``);
                      no libcudart, no libtorch.

The interior kernel and the PML kernel are separate ``__global__`` functions:
the interior one has no PML pointer in scope at all.
"""

from __future__ import annotations

from . import abi, emit_c
from .emit_c import CTYPE, expand, full_region, group_body, grad_body, pr, regions
from .model import Model
from .spec import Inject, Record, Stage

BLOCK = {1: (128, 1, 1), 2: (32, 8, 1), 3: (32, 4, 2)}


def params_struct(m: Model, tag: str) -> str:
    f = []
    for s in m.spec.state:
        f.append(f"  dw_t *{s}_c; dw_t *{s}_n;")
    for p in m.spec.params:
        f.append(f"  dw_t const *par_{p};")
        f.append(f"  dw_t *grad_{p};")
    for pa in m.spec.profile_arrays():
        f.append(f"  dw_t const *prof_{pa};")
    for s in m.stores:
        f.append(f"  dw_t *st_{s};")
    for s in m.spec.scalars:
        f.append(f"  dw_t {s};")
    for a in m.spec.axis_names:
        f.append(f"  int64_t n{a}; int64_t pmlb_{a}; int64_t pmle_{a};")
    f.append("  int64_t step_ratio;")
    for k in range(m.spec.ndim):
        f.append(f"  int64_t lo{k}; int64_t hi{k};")
    f.append("  dw_t *io_arr; int64_t const *io_idx;")
    f.append("  int64_t io_n; int64_t io_arr_off; int64_t io_idx_off;")
    return (
        f"#ifndef {tag.upper()}_P_H\n#define {tag.upper()}_P_H\n"
        "#include <stdint.h>\n"
        "typedef struct {\n" + "\n".join(f) + f"\n}} {tag}_kp;\n#endif\n"
    )


def _kernel_locals(m: Model, use_pml: bool) -> list[str]:
    out = []
    for s in m.spec.state:
        out.append(f"  dw_t *__restrict const {s}_c = pr.{s}_c;")
        out.append(f"  dw_t *__restrict const {s}_n = pr.{s}_n; (void){s}_n;")
    for p in m.spec.params:
        out.append(
            f"  dw_t const *__restrict const par_{p} = pr.par_{p}; (void)par_{p};"
        )
        out.append(f"  dw_t *__restrict const grad_{p} = pr.grad_{p}; (void)grad_{p};")
    if use_pml:
        for pa in m.spec.profile_arrays():
            out.append(f"  dw_t const *__restrict const prof_{pa} = pr.prof_{pa};")
    for s in m.stores:
        out.append(f"  dw_t *__restrict const st_{s} = pr.st_{s}; (void)st_{s};")
    for s in m.spec.scalars:
        out.append(f"  dw_t const {s} = pr.{s}; (void){s};")
    for a in m.spec.axis_names:
        out.append(f"  int64_t const n{a} = pr.n{a};")
        out.append(f"  int64_t const pmlb_{a} = pr.pmlb_{a}; (void)pmlb_{a};")
        out.append(f"  int64_t const pmle_{a} = pr.pmle_{a}; (void)pmle_{a};")
    out.append("  int64_t const step_ratio = pr.step_ratio;")
    return out


def _thread_map(m: Model) -> list[str]:
    ax = m.spec.axis_names
    nd = m.spec.ndim
    lanes = ["x", "y", "z"]
    out = []
    for k in range(nd):
        lane = lanes[nd - 1 - k]
        out.append(
            f"  int64_t const {ax[k]} = pr.lo{k} + (int64_t)(blockIdx.{lane}"
            f" * blockDim.{lane} + threadIdx.{lane});"
        )
    cond = " || ".join(f"{ax[k]} >= pr.hi{k}" for k in range(nd))
    out.append(f"  if ({cond}) return;")
    strides = [
        f"{ax[k]}" + "".join(f" * n{ax[j]}" for j in range(k + 1, nd))
        for k in range(nd)
    ]
    out.append(f"  int64_t const i = {' + '.join(strides)};")
    return out


def _interior_guard(m: Model, stag) -> str:
    """PML kernel early-out for cells inside the interior box."""
    conds = []
    for k in range(m.spec.ndim):
        lo, hi = emit_c.axis_range(m, stag, k, "interior")
        a = m.spec.axis_names[k]
        conds.append(f"({a} >= {lo} && {a} < {hi})")
    return "  if (" + " && ".join(conds) + ") return;"


def kernels(m: Model, tag: str) -> tuple[str, list[str]]:
    out: list[str] = []
    names: list[str] = []
    for direction in ("fwd", "bwd"):
        plans = m.fwd_plans if direction == "fwd" else m.bwd_plans
        for si, plan in enumerate(plans):
            for do_store in (True, False):
                st = "s" if do_store else "n"
                if direction == "bwd" and do_store:
                    fs = len(m.fwd_plans) - 1 - si
                    by_stag: dict = {}
                    for g in m.grads:
                        if g.stage == fs:
                            by_stag.setdefault(m.spec.stag(g.adj_field), []).append(g)
                    for gi, (stag, contribs) in enumerate(by_stag.items()):
                        n = f"{tag}_k_{direction}{si}_grad{gi}"
                        names.append(n)
                        L = [f'extern "C" __global__ void {n}({tag}_kp pr) {{']
                        L += _kernel_locals(m, False)
                        L += _thread_map(m)
                        L.append(grad_body(m, contribs, "  "))
                        L.append("}")
                        out.append("\n".join(L))
                for gi, gr in enumerate(plan.groups_full):
                    gint = next((g for g in plan.groups_int if g.stag == gr.stag), None)
                    if gint is not None and gint.fields:
                        n = f"{tag}_k_{direction}{si}_g{gi}_{st}_int"
                        names.append(n)
                        L = [f'extern "C" __global__ void {n}({tag}_kp pr) {{']
                        L += _kernel_locals(m, False)
                        L += _thread_map(m)
                        L.append(
                            group_body(
                                m,
                                plan,
                                gint,
                                True,
                                do_store and direction == "fwd",
                                "  ",
                            )
                        )
                        L.append("}")
                        out.append("\n".join(L))
                    n = f"{tag}_k_{direction}{si}_g{gi}_{st}_pml"
                    names.append(n)
                    L = [f'extern "C" __global__ void {n}({tag}_kp pr) {{']
                    L += _kernel_locals(m, True)
                    L += _thread_map(m)
                    L.append(_interior_guard(m, gr.stag))
                    L.append(
                        group_body(
                            m, plan, gr, False, do_store and direction == "fwd", "  "
                        )
                    )
                    L.append("}")
                    out.append("\n".join(L))
    # source/receiver kernels
    seen = set()
    for act in m.spec.actions:
        if isinstance(act, Stage):
            continue
        nf = len(act.fields)
        for kind in ("gather", "scatter"):
            n = f"{tag}_io_{kind}{nf}"
            if n in seen:
                continue
            seen.add(n)
            names.append(n)
            args = ", ".join(f"dw_t *f{j}" for j in range(nf))
            L = [f'extern "C" __global__ void {n}({tag}_kp pr, {args}) {{']
            L.append(
                "  int64_t const q = (int64_t)(blockIdx.x * blockDim.x + threadIdx.x);"
            )
            L.append("  if (q >= pr.io_n) return;")
            L.append("  int64_t const loc = pr.io_idx[pr.io_idx_off + q];")
            L.append("  if (loc < 0) return;")
            if kind == "gather":
                terms = " + ".join(f"f{j}[loc]" for j in range(nf))
                L.append(f"  pr.io_arr[pr.io_arr_off + q] = {terms};")
            else:
                for j in range(nf):
                    L.append(f"  f{j}[loc] += pr.io_arr[pr.io_arr_off + q];")
            L.append("}")
            out.append("\n".join(L))
    return "\n\n".join(out), names


def host_driver(m: Model, tag: str, dtype: str, direction: str) -> str:
    spec = m.spec
    args = abi.signature(spec, m, direction, "cuda")
    name = abi.kernel_name(spec, m.accuracy, dtype, direction, "cuda")
    ax = spec.axis_names
    nd = spec.ndim
    ngrid = " * ".join(f"n{a}" for a in ax)
    actions = m.fwd if direction == "fwd" else m.bwd
    plans = m.fwd_plans if direction == "fwd" else m.bwd_plans
    bl = BLOCK[nd]
    L = [f"int {name}(\n    " + abi.c_params(args, "dw_t") + ") {"]
    L.append(f"  int64_t const ngrid = {ngrid};")
    L.append(f"  {tag}_kp pr;")
    L.append("  dw_custream const stream = (dw_custream)stream_ptr;")
    L.append("  memset(&pr, 0, sizeof(pr));")
    for a in ax:
        L.append(f"  int64_t const pmlb_{a} = pml_b_{a}; (void)pmlb_{a};")
        L.append(f"  int64_t const pmle_{a} = pml_e_{a}; (void)pmle_{a};")
    for a in ax:
        L.append(f"  pr.n{a} = n{a}; pr.pmlb_{a} = pml_b_{a}; pr.pmle_{a} = pml_e_{a};")
    for s in spec.scalars:
        L.append(f"  pr.{s} = {s};")
    L.append("  pr.step_ratio = step_ratio;")
    for pa in spec.profile_arrays():
        L.append(f"  pr.prof_{pa} = prof_{pa};")
    L.append("  for (int64_t shot = 0; shot < n_shots; ++shot) {")
    L.append("    int64_t const sbase = shot * ngrid;")
    L.append("    int64_t par = 0;")
    for p in spec.params:
        L.append(f"    pr.par_{p} = par_{p} + (par_batched ? sbase : 0);")
        if direction == "bwd":
            L.append(f"    pr.grad_{p} = grad_{p} + sbase;")
    L.append("    for (int64_t k = 0; k < nt; ++k) {")
    L.append(
        "      int64_t const t = " + ("k;" if direction == "fwd" else "nt - 1 - k;")
    )
    L.append("      int64_t const do_store = need_store && ((t % step_ratio) == 0);")
    for s in m.stores:
        L.append(
            f"      pr.st_{s} = st_{s} + sbase + (store_all_steps ?"
            f" (t / step_ratio) * n_shots * ngrid : 0);"
        )
    si = 0
    for act in actions:
        if isinstance(act, Stage):
            plan = plans[si]
            for f in spec.state:
                if plan.mode == "pingpong":
                    L.append(
                        f"      pr.{f}_c = (par ? {f}_nxt : {f}_cur) + sbase;"
                        f" pr.{f}_n = (par ? {f}_cur : {f}_nxt) + sbase;"
                    )
                else:
                    L.append(
                        f"      pr.{f}_c = (par ? {f}_nxt : {f}_cur) + sbase;"
                        f" pr.{f}_n = pr.{f}_c;"
                    )
            L.append("      {")
            if direction == "bwd":
                fs = len(m.fwd_plans) - 1 - si
                by_stag: dict = {}
                for g in m.grads:
                    if g.stage == fs:
                        by_stag.setdefault(m.spec.stag(g.adj_field), []).append(g)
                for gi, (stag, _c) in enumerate(by_stag.items()):
                    L += _launch(
                        m,
                        f"{tag}_k_bwd{si}_grad{gi}",
                        full_region(m, stag),
                        bl,
                        guard="do_store",
                    )
            for gi, gr in enumerate(plan.groups_full):
                regs = regions(m, gr.stag)
                gint = next((g for g in plan.groups_int if g.stag == gr.stag), None)
                for stflag, sttag in ((1, "s"), (0, "n")):
                    cond = "do_store" if stflag else "!do_store"
                    if gint is not None and gint.fields:
                        L += _launch(
                            m,
                            f"{tag}_k_{direction}{si}_g{gi}_{sttag}_int",
                            regs[0][1],
                            bl,
                            guard=cond,
                        )
                    L += _launch(
                        m,
                        f"{tag}_k_{direction}{si}_g{gi}_{sttag}_pml",
                        full_region(m, gr.stag),
                        bl,
                        guard=cond,
                    )
            L.append("      }")
            if plan.mode == "pingpong":
                L.append("      par ^= 1;")
            si += 1
        else:
            L += _io_launch(m, tag, act, direction)
    L.append("    }")
    if any(p.mode == "pingpong" for p in plans):
        L.append("    if (par) {")
        for f in spec.state:
            L.append(
                f"      dw_cuda_copy({f}_cur + sbase, {f}_nxt + sbase, ngrid, stream);"
            )
        L.append("    }")
    L.append("  }")
    L.append("  return 0;")
    L.append("}")
    return "\n".join(L)


def _launch(m: Model, kname: str, rng, bl, guard: str) -> list[str]:
    nd = m.spec.ndim
    L = [f"      if ({guard}) {{"]
    for k in range(nd):
        lo, hi = rng[k]
        L.append(f"        pr.lo{k} = {lo}; pr.hi{k} = {hi};")
    ext = [f"(pr.hi{k} - pr.lo{k})" for k in range(nd)]
    gx = f"dw_ceil_div({ext[nd - 1]}, {bl[0]})"
    gy = f"dw_ceil_div({ext[nd - 2]}, {bl[1]})" if nd >= 2 else "1"
    gz = f"dw_ceil_div({ext[nd - 3]}, {bl[2]})" if nd >= 3 else "1"
    L.append(
        f'        dw_cuda_launch("{kname}", {gx}, {gy}, {gz},'
        f" {bl[0]}, {bl[1]}, {bl[2]}, stream, &pr);"
    )
    L.append("      }")
    return L


def _io_launch(m: Model, tag: str, act, direction: str) -> list[str]:
    spec = m.spec
    fwd = direction == "fwd"
    if isinstance(act, Record):
        tg, arr = ("rec", "out") if fwd else ("src", "grad")
        kind = "gather"
    else:
        tg, arr = ("src", "amp") if fwd else ("rec", "grad")
        kind = "scatter"
    k = act.idx
    n = f"n_{tg}{k}"
    nf = len(act.fields)
    L = ["      {"]
    L.append(f"        pr.io_arr = (dw_t *){tg}{k}_{arr};")
    L.append(f"        pr.io_idx = {tg}{k}_i;")
    L.append(f"        pr.io_n = {n};")
    L.append(f"        pr.io_arr_off = t * n_shots * {n} + shot * {n};")
    L.append(f"        pr.io_idx_off = shot * {n};")
    ptrs = ", ".join(f"(par ? {f}_nxt : {f}_cur) + sbase" for f in act.fields)
    L.append(
        f'        dw_io_launch_{nf}("{tag}_io_{kind}{nf}", {n}, stream, &pr, {ptrs});'
    )
    L.append("      }")
    return L


def emit(m: Model, dtype: str, tag: str) -> tuple[str, str, str]:
    kern, names = kernels(m, tag)
    hdr = params_struct(m, tag)
    cu = "\n".join(
        [
            "/* Generated by dwgen -- do not edit. */",
            "#include <stdint.h>",
            f"typedef {CTYPE[dtype]} dw_t;",
            f'#include "{tag}_p.h"',
            "",
            kern,
            "",
        ]
    )
    nfs = sorted({len(a.fields) for a in m.spec.actions if not isinstance(a, Stage)})
    io_helpers = []
    for nf in nfs:
        args = ", ".join(f"dw_t *f{j}" for j in range(nf))
        pass_ = "".join(f"    void *a{j} = &f{j};\n" for j in range(nf))
        arr = ", ".join(f"a{j}" for j in range(nf))
        io_helpers.append(
            f"static int dw_io_launch_{nf}(const char *nm, int64_t n,"
            f" dw_custream s, void *p, {args}) {{\n"
            f"  if (n <= 0) return 0;\n"
            f"  dw_cufunction fn = dw_cuda_fn(nm);\n"
            f"  if (!fn) return -1;\n"
            f"{pass_}"
            f"  void *args[] = {{p, {arr}}};\n"
            f"  return dw_cuLaunchKernel(fn, dw_ceil_div(n, 64), 1, 1, 64, 1, 1,"
            f" 0, s, args, NULL);\n}}\n"
        )
    host = "\n".join(
        [
            "/* Generated by dwgen -- do not edit. */",
            "#include <stdint.h>",
            "#include <string.h>",
            f"typedef {CTYPE[dtype]} dw_t;",
            '#include "dw_cuda.h"',
            f'#include "{tag}_p.h"',
            "",
            *io_helpers,
            "static void dw_cuda_copy(dw_t *dst, dw_t const *src, int64_t n,"
            " dw_custream s) {",
            "  dw_cuda_memcpy_dtod(dst, src, (size_t)n * sizeof(dw_t), s);",
            "}",
            "",
            host_driver(m, tag, dtype, "fwd"),
            "",
            host_driver(m, tag, dtype, "bwd"),
            "",
        ]
    )
    return hdr, cu, host
