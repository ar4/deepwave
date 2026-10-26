"""Emit plain C (no OpenMP, no libtorch, no Python.h, no CUDA) for a Model."""

from __future__ import annotations

import itertools
from fractions import Fraction

from . import abi, ir
from .ir import Add, Diff, Expr, Mul, Num, Par, Prof, Ref, Sca
from .model import Group, Model, StagePlan
from .spec import Inject, Record, Stage

CTYPE = {"f32": "float", "f64": "double"}


def index_expr(off: tuple[int, ...], ax: tuple[str, ...], base: str = "i") -> str:
    nd = len(ax)
    parts: list[str] = []
    for k, o in enumerate(off):
        if o == 0:
            continue
        stride = "".join(f" * n{ax[j]}" for j in range(k + 1, nd))
        parts.append(f"({o}){stride}")
    if not parts:
        return base
    return base + " + " + " + ".join(parts)


def num_str(v: Fraction) -> str:
    if v.denominator == 1:
        return f"(dw_t)({v.numerator})"
    return f"(dw_t)({v.numerator}.0 / {v.denominator}.0)"


def pr(e: Expr, ax: tuple[str, ...], suffix: str) -> str:
    if isinstance(e, Num):
        return num_str(e.v)
    if isinstance(e, Sca):
        return e.name
    if isinstance(e, Ref):
        return f"{e.name}{suffix}[{index_expr(e.off, ax)}]"
    if isinstance(e, Par):
        return f"par_{e.name}[{index_expr(e.off, ax)}]"
    if isinstance(e, Prof):
        c = ax[e.axis]
        o = f" + ({e.off})" if e.off else ""
        return f"prof_{e.name}{c}[{c}{o}]"
    if isinstance(e, Add):
        return "(" + " + ".join(pr(a, ax, suffix) for a in e.args) + ")"
    if isinstance(e, Mul):
        return "(" + " * ".join(pr(a, ax, suffix) for a in e.args) + ")"
    raise TypeError(f"unexpanded or unknown node {type(e)}")


def expand(e: Expr, m: Model) -> Expr:
    """Expand every ``Diff`` in place, leaving the surrounding Add/Mul nesting.

    Flattening the whole tree into a sum of scalar products (which the adjoint
    derivation does internally) is correct but emits ``v*v*dt2`` once per
    stencil tap and defeats GCC's loop vectoriser; keeping the spec's own
    factoring emits the same shape a hand-written kernel would.
    """
    nd, ax = m.spec.ndim, m.spec.axis_names
    if isinstance(e, Diff):
        scale = ir.diff_scale(e.kind, e.axis, ax)
        arg = expand(e.arg, m)
        parts = []
        for sh, c in ir.taps(e.kind, m.accuracy):
            sv = [0] * nd
            sv[e.axis] = sh
            parts.append(ir.mul(Num(c), ir.shift(arg, tuple(sv))))
        return ir.mul(scale, ir.add(*parts))
    if isinstance(e, Add):
        return ir.add(*[expand(a, m) for a in e.args])
    if isinstance(e, Mul):
        return ir.mul(*[expand(a, m) for a in e.args])
    return e


# ---------------------------------------------------------------------------
# loop nests
# ---------------------------------------------------------------------------


def axis_range(m: Model, stag, k: int, which: str) -> tuple[str, str]:
    """(lo, hi) for axis ``k``: 'full', 'interior', 'lo' PML or 'hi' PML."""
    a = m.spec.axis_names[k]
    lo_pad, hi_pad = m.spec.pad(m.accuracy)
    h = 1 if stag[k] else 0
    first, last = f"{lo_pad}", f"(n{a} - {hi_pad + h})"
    b = f"pmlb_{a}"
    e = f"pmle_{a}" if not h else f"(pmlb_{a} > pmle_{a} - 1 ? pmlb_{a} : pmle_{a} - 1)"
    return {
        "full": (first, last),
        "interior": (b, e),
        "lo": (first, b),
        "hi": (e, last),
    }[which]


def regions(m: Model, stag) -> list[tuple[str, list[tuple[str, str]]]]:
    """Interior box plus the 2*ndim slabs that tile its complement."""
    nd = m.spec.ndim
    out = [("interior", [axis_range(m, stag, k, "interior") for k in range(nd)])]
    for k in range(nd):
        for side in ("lo", "hi"):
            rng = []
            for j in range(nd):
                if j == k:
                    rng.append(axis_range(m, stag, j, side))
                elif j < k:
                    rng.append(axis_range(m, stag, j, "interior"))
                else:
                    rng.append(axis_range(m, stag, j, "full"))
            out.append((f"pml{k}{side}", rng))
    return out


def full_region(m: Model, stag) -> list[tuple[str, str]]:
    return [axis_range(m, stag, k, "full") for k in range(m.spec.ndim)]


def loops(m: Model, rng: list[tuple[str, str]], body: str, ind: str) -> str:
    ax = m.spec.axis_names
    nd = m.spec.ndim
    out = []
    for k in range(nd):
        a = ax[k]
        lo, hi = rng[k]
        if k == 0:
            out.append(f"{ind}{{")
            out.append(f"{ind}int64_t const tlo = dw_band_lo({lo}, {hi}, tid, nth);")
            out.append(f"{ind}int64_t const thi = dw_band_hi({lo}, {hi}, tid, nth);")
            out.append(f"{ind}for (int64_t {a} = tlo; {a} < thi; ++{a})")
        else:
            out.append(f"{ind}for (int64_t {a} = {lo}; {a} < {hi}; ++{a})")
        ind += "  "
    out.append(f"{ind}{{")
    strides = [
        f"{ax[k]}" + "".join(f" * n{ax[j]}" for j in range(k + 1, nd))
        for k in range(nd)
    ]
    out.append(f"{ind}  int64_t const i = {' + '.join(strides)};")
    out.append(body)
    out.append(f"{ind}}}")
    out.append("}")
    return "\n".join(out)


def group_body(
    m: Model,
    plan: StagePlan,
    gr: Group,
    interior: bool,
    do_store: bool,
    ind: str,
) -> str:
    ax = m.spec.axis_names
    st = plan.stage_int if interior else plan.stage_full
    stores = m.stores_int if interior else m.stores_full
    rd = "_c"
    wr = "_n" if plan.mode == "pingpong" else "_c"
    lines = []
    tmp = []
    for f in gr.fields:
        if f not in st.updates:
            continue
        tmp.append((f, pr(expand(st.updates[f], m), ax, rd)))
    for j, (f, e) in enumerate(tmp):
        lines.append(f"{ind}dw_t const tmp{j} = {e};")
    if do_store:
        for s, owner in m.store_owner.items():
            if owner in gr.fields and s in stores:
                lines.append(f"{ind}st_{s}[i] = {pr(expand(stores[s], m), ax, rd)};")
    for j, (f, _) in enumerate(tmp):
        lines.append(f"{ind}{f}{wr}[i] = tmp{j};")
    return "\n".join(lines)


def grad_body(m: Model, contribs: list, ind: str) -> str:
    ax = m.spec.axis_names
    by_param: dict[str, list[str]] = {}
    for g in contribs:
        idx = index_expr(g.off, ax)
        by_param.setdefault(g.param, []).append(
            f"st_{g.store}[{idx}] * {g.adj_field}_c[{idx}]"
        )
    return "\n".join(
        f"{ind}grad_{p}[i] += ({' + '.join(t)}) * (dw_t)step_ratio;"
        for p, t in by_param.items()
    )


# ---------------------------------------------------------------------------
# context struct / task functions / driver
# ---------------------------------------------------------------------------


def ctx_struct(m: Model, direction: str) -> str:
    f = []
    for s in m.spec.state:
        f.append(f"  dw_t *{s}_c; dw_t *{s}_n;")
    for p in m.spec.params:
        f.append(f"  dw_t const *par_{p};")
    for pa in m.spec.profile_arrays():
        f.append(f"  dw_t const *prof_{pa};")
    for s in m.stores:
        f.append(f"  dw_t *st_{s};")
    if direction == "bwd":
        for p in m.spec.params:
            f.append(f"  dw_t *grad_{p};")
    for s in m.spec.scalars:
        f.append(f"  dw_t {s};")
    for a in m.spec.axis_names:
        f.append(f"  int64_t n{a}; int64_t pmlb_{a}; int64_t pmle_{a};")
    f.append("  int64_t step_ratio;")
    return "typedef struct {\n" + "\n".join(f) + f"\n}} ctx_{direction};"


def _locals(m: Model, direction: str, pml: bool = True) -> list[str]:
    out = [f"  ctx_{direction} *const c = (ctx_{direction} *)vc;"]
    for s in m.spec.state:
        out.append(f"  dw_t *__restrict const {s}_c = c->{s}_c;")
        out.append(f"  dw_t *__restrict const {s}_n = c->{s}_n;")
        out.append(f"  (void){s}_c; (void){s}_n;")
    for p in m.spec.params:
        out.append(
            f"  dw_t const *__restrict const par_{p} = c->par_{p}; (void)par_{p};"
        )
    if pml:
        for pa in m.spec.profile_arrays():
            out.append(
                f"  dw_t const *__restrict const prof_{pa} = c->prof_{pa};"
                f" (void)prof_{pa};"
            )
    for s in m.stores:
        out.append(f"  dw_t *__restrict const st_{s} = c->st_{s}; (void)st_{s};")
    if direction == "bwd":
        for p in m.spec.params:
            out.append(
                f"  dw_t *__restrict const grad_{p} = c->grad_{p}; (void)grad_{p};"
            )
    for s in m.spec.scalars:
        out.append(f"  dw_t const {s} = c->{s}; (void){s};")
    for a in m.spec.axis_names:
        out.append(f"  int64_t const n{a} = c->n{a}; (void)n{a};")
        out.append(f"  int64_t const pmlb_{a} = c->pmlb_{a}; (void)pmlb_{a};")
        out.append(f"  int64_t const pmle_{a} = c->pmle_{a}; (void)pmle_{a};")
    out.append("  int64_t const step_ratio = c->step_ratio; (void)step_ratio;")
    return out


def task_fns(
    m: Model, direction: str, si: int, do_store: bool
) -> tuple[list[str], list[str]]:
    """Emit one task function per group (and per gradient sweep).

    Each becomes its own ``dw_pool_run``, so the pool dispatch doubles as the
    barrier that in-place group ordering needs.
    """
    plans = m.fwd_plans if direction == "fwd" else m.bwd_plans
    plan = plans[si]
    nd = m.spec.ndim
    centre = (1,) * nd
    tag = "s" if do_store else "n"
    fns: list[str] = []
    names: list[str] = []

    if direction == "bwd" and do_store:
        fwd_stage = len(m.fwd_plans) - 1 - si
        by_stag: dict[tuple, list] = {}
        for g in m.grads:
            if g.stage != fwd_stage:
                continue
            by_stag.setdefault(m.spec.stag(g.adj_field), []).append(g)
        for gi, (stag, contribs) in enumerate(by_stag.items()):
            name = f"task_{direction}{si}_grad{gi}"
            L = [f"static void {name}(void *vc, int64_t tid, int64_t nth) {{"]
            L += _locals(m, direction, pml=False)
            L.append("  /* deterministic model-gradient accumulation: one thread")
            L.append("     owns each cell, per-shot private buffers, no atomics */")
            L.append(
                loops(m, full_region(m, stag), grad_body(m, contribs, "      "), "  ")
            )
            L.append("}")
            fns.append("\n".join(L))
            names.append(name)

    ws = do_store and direction == "fwd"
    for gi, gr in enumerate(plan.groups_full):
        gint = next((g for g in plan.groups_int if g.stag == gr.stag), None)
        regs = regions(m, gr.stag)
        # The interior sweep and the PML sweeps are separate functions, not
        # just separate loops: the interior one has no PML pointer in scope,
        # and keeping it small is what lets the vectoriser see it.
        if gint is not None and gint.fields:
            name = f"task_{direction}{si}_g{gi}_{tag}_int"
            L = [f"static void {name}(void *vc, int64_t tid, int64_t nth) {{"]
            L += _locals(m, direction, pml=False)
            L.append(f"  /* interior fast path, fields: {', '.join(gint.fields)} */")
            L.append(
                loops(
                    m, regs[0][1], group_body(m, plan, gint, True, ws, "      "), "  "
                )
            )
            L.append("}")
            fns.append("\n".join(L))
            names.append(name)
        name = f"task_{direction}{si}_g{gi}_{tag}_pml"
        L = [f"static void {name}(void *vc, int64_t tid, int64_t nth) {{"]
        L += _locals(m, direction, pml=True)
        L.append(f"  /* PML faces, fields: {', '.join(gr.fields)} */")
        body_pml = group_body(m, plan, gr, False, ws, "      ")
        for _, rng in regs[1:]:
            L.append(loops(m, rng, body_pml, "  "))
        L.append("}")
        fns.append("\n".join(L))
        names.append(name)
    return fns, names


def driver(m: Model, direction: str, dtype: str) -> str:
    spec = m.spec
    args = abi.signature(spec, m, direction)
    name = abi.kernel_name(spec, m.accuracy, dtype, direction, "cpu")
    ax = spec.axis_names
    ngrid = " * ".join(f"n{a}" for a in ax)
    actions = m.fwd if direction == "fwd" else m.bwd
    plans = m.fwd_plans if direction == "fwd" else m.bwd_plans
    npp = sum(1 for p in plans if p.mode == "pingpong")
    L = [f"int {name}(\n    " + abi.c_params(args, "dw_t") + ") {"]
    L.append(f"  int64_t const ngrid = {ngrid};")
    L.append(f"  ctx_{direction} c;")
    L.append("  dw_pool *const pool = dw_pool_get(n_threads);")
    for a in ax:
        L.append(f"  c.n{a} = n{a}; c.pmlb_{a} = pml_b_{a}; c.pmle_{a} = pml_e_{a};")
    for s in spec.scalars:
        L.append(f"  c.{s} = {s};")
    L.append("  c.step_ratio = step_ratio;")
    for pa in spec.profile_arrays():
        L.append(f"  c.prof_{pa} = prof_{pa};")
    L.append("  for (int64_t shot = 0; shot < n_shots; ++shot) {")
    L.append("    int64_t const sbase = shot * ngrid;")
    L.append("    int64_t par = 0;")
    for p in spec.params:
        L.append(f"    c.par_{p} = par_{p} + (par_batched ? sbase : 0);")
    if direction == "bwd":
        for p in spec.params:
            L.append(f"    c.grad_{p} = grad_{p} + sbase;")
    L.append("    for (int64_t k = 0; k < nt; ++k) {")
    L.append(
        "      int64_t const t = " + ("k;" if direction == "fwd" else "nt - 1 - k;")
    )
    L.append("      int64_t const do_store = need_store && ((t % step_ratio) == 0);")
    for s in m.stores:
        L.append(
            f"      c.st_{s} = st_{s} + sbase + (store_all_steps ?"
            f" (t / step_ratio) * n_shots * ngrid : 0);"
        )
    si = 0
    for act in actions:
        if isinstance(act, Stage):
            plan = plans[si]
            for f in spec.state:
                if plan.mode == "pingpong":
                    L.append(
                        f"      c.{f}_c = (par ? {f}_nxt : {f}_cur) + sbase;"
                        f" c.{f}_n = (par ? {f}_cur : {f}_nxt) + sbase;"
                    )
                else:
                    L.append(
                        f"      c.{f}_c = (par ? {f}_nxt : {f}_cur) + sbase;"
                        f" c.{f}_n = c.{f}_c;"
                    )
            _, ns = task_fns(m, direction, si, True)
            _, nn = task_fns(m, direction, si, False)
            L.append("      if (do_store) {")
            for nm in ns:
                L.append(f"        dw_pool_run(pool, {nm}, &c);")
            L.append("      } else {")
            for nm in nn:
                L.append(f"        dw_pool_run(pool, {nm}, &c);")
            L.append("      }")
            if plan.mode == "pingpong":
                L.append("      par ^= 1;")
            si += 1
        elif isinstance(act, Record):
            L.append(_io(m, act, direction, "record"))
        elif isinstance(act, Inject):
            L.append(_io(m, act, direction, "inject"))
    L.append("    }")
    if npp:
        L.append("    if (par) {")
        for f in spec.state:
            L.append(
                f"      for (int64_t q = 0; q < ngrid; ++q)"
                f" {f}_cur[sbase + q] = {f}_nxt[sbase + q];"
            )
        L.append("    }")
    L.append("  }")
    L.append("  return 0;")
    L.append("}")
    return "\n".join(L)


def _io(m: Model, act, direction: str, op: str) -> str:
    """Emit a receiver record or a source injection (or their adjoints)."""
    spec = m.spec
    fwd = direction == "fwd"
    if isinstance(act, Record):
        tag, arr = ("rec", "out") if fwd else ("src", "grad")
    else:
        tag, arr = ("src", "amp") if fwd else ("rec", "grad")
    k = act.idx
    n = f"n_{tag}{k}"
    slot = f"{tag}{k}_{arr}[t * n_shots * {n} + shot * {n} + q]"
    ptr = f"(par ? %s_nxt : %s_cur) + sbase"
    L = [f"      for (int64_t q = 0; q < {n}; ++q) {{"]
    L.append(f"        int64_t const loc = {tag}{k}_i[shot * {n} + q];")
    L.append("        if (loc < 0) continue;")
    if op == "record":
        terms = " + ".join(f"({ptr % (f, f)})[loc]" for f in act.fields)
        L.append(f"        {slot} = {terms};")
    else:
        for f in act.fields:
            L.append(f"        ({ptr % (f, f)})[loc] += {slot};")
    L.append("      }")
    return "\n".join(L)


def emit(m: Model, dtype: str) -> str:
    out = [
        "/* Generated by dwgen -- do not edit. */",
        "#include <stdint.h>",
        "#include <stdbool.h>",
        "#include <stdlib.h>",
        '#include "dw_pool.h"',
        f"typedef {CTYPE[dtype]} dw_t;",
        "",
    ]
    for direction, plans in (("fwd", m.fwd_plans), ("bwd", m.bwd_plans)):
        out.append(ctx_struct(m, direction))
        for si in range(len(plans)):
            for ds in (True, False):
                fns, _ = task_fns(m, direction, si, ds)
                out.extend(fns)
        out.append(driver(m, direction, dtype))
        out.append("")
    return "\n".join(out)
