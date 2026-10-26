"""Doubly-derived adjoint check.

The backward update the generator emits is derived by *structural
transposition* of the forward update (``adjoint.py``): no differentiation is
involved, only index algebra on linear terms.

This module derives the same thing a second time, by a genuinely different
mechanism: it instantiates the forward update on a small symbolic grid and
differentiates it with SymPy, giving the Jacobian of the forward time-step
map explicitly.  The emitted adjoint must be that Jacobian's transpose, entry
by entry, and the emitted model-gradient accumulation must equal
``sum_i adj_g[i] * d F_g[i] / d p[j]``.

Everything is exact (rational arithmetic over symbols), so a wrong field
pairing, a missing scale factor, a sign error or a mis-shifted coefficient
shows up as a non-zero residual rather than as a small numerical discrepancy.
Run at generation time for every (family, ndim, accuracy) that is emitted.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import sympy

from . import ir
from .ir import Add, Diff, Expr, Mul, Num, Par, Prof, Ref, Sca
from .model import Model
from .spec import Stage


class Ctx:
    def __init__(self, m: Model, shape: tuple[int, ...], active: list[tuple[int, ...]]):
        self.m = m
        self.shape = shape
        self.active = set(active)
        self.sca: dict[str, sympy.Symbol] = {}
        self.par: dict[tuple[str, tuple[int, ...]], sympy.Symbol] = {}
        self.prof: dict[tuple[str, int, int], sympy.Symbol] = {}
        self.state: dict[tuple[str, tuple[int, ...]], sympy.Symbol] = {}
        for f in m.spec.state:
            for c in active:
                self.state[(f, c)] = sympy.Symbol(f"S_{f}_" + "_".join(map(str, c)))

    def s_sca(self, n: str) -> sympy.Symbol:
        return self.sca.setdefault(n, sympy.Symbol(f"K_{n}"))

    def s_par(self, n: str, c: tuple[int, ...]) -> sympy.Symbol:
        return self.par.setdefault(
            (n, c), sympy.Symbol(f"P_{n}_" + "_".join(map(str, c)))
        )

    def s_prof(self, n: str, ax: int, i: int) -> sympy.Symbol:
        return self.prof.setdefault((n, ax, i), sympy.Symbol(f"F_{n}_{ax}_{i}"))

    def inside(self, c: tuple[int, ...]) -> bool:
        return all(0 <= v < n for v, n in zip(c, self.shape))


def ev(e: Expr, cell: tuple[int, ...], ctx: Ctx, accuracy: int):
    if isinstance(e, Num):
        return sympy.Rational(e.v.numerator, e.v.denominator)
    if isinstance(e, Sca):
        return ctx.s_sca(e.name)
    if isinstance(e, Ref):
        c = tuple(a + b for a, b in zip(cell, e.off))
        return ctx.state.get((e.name, c), sympy.Integer(0))
    if isinstance(e, Par):
        c = tuple(a + b for a, b in zip(cell, e.off))
        return ctx.s_par(e.name, c) if ctx.inside(c) else sympy.Integer(0)
    if isinstance(e, Prof):
        i = cell[e.axis] + e.off
        return (
            ctx.s_prof(e.name, e.axis, i)
            if 0 <= i < ctx.shape[e.axis]
            else sympy.Integer(0)
        )
    if isinstance(e, Add):
        return sum((ev(a, cell, ctx, accuracy) for a in e.args), sympy.Integer(0))
    if isinstance(e, Mul):
        out = sympy.Integer(1)
        for a in e.args:
            out = out * ev(a, cell, ctx, accuracy)
        return out
    if isinstance(e, Diff):
        scale = ev(
            ir.diff_scale(e.kind, e.axis, ctx.m.spec.axis_names), cell, ctx, accuracy
        )
        acc = sympy.Integer(0)
        for s, cf in ir.taps(e.kind, accuracy):
            sh = [0] * len(cell)
            sh[e.axis] = s
            c2 = tuple(a + b for a, b in zip(cell, sh))
            acc += sympy.Rational(cf.numerator, cf.denominator) * ev(
                e.arg, c2, ctx, accuracy
            )
        return acc * scale
    raise TypeError(type(e))


def _jacobian(updates: dict[str, Expr], ctx: Ctx, accuracy: int):
    """J[(g, cell_out)][(f, cell_in)] = d out / d in, by SymPy differentiation."""
    J: dict[tuple[str, tuple[int, ...]], dict] = {}
    inv = {v: k for k, v in ctx.state.items()}
    for g, e in updates.items():
        for c in sorted(ctx.active):
            expr = sympy.expand(ev(e, c, ctx, accuracy))
            row = {}
            for sym in expr.free_symbols:
                k = inv.get(sym)
                if k is None:
                    continue
                d = sympy.expand(sympy.diff(expr, sym))
                if d != 0:
                    row[k] = d
            J[(g, c)] = row
    return J


def check_stage(
    m: Model,
    fwd: Stage,
    adj: Stage,
    shape: tuple[int, ...],
    active: list[tuple[int, ...]],
    state: list[str],
) -> list[str]:
    ctx = Ctx(m, shape, active)
    ctx.state = {k: v for k, v in ctx.state.items() if k[0] in state}
    J = _jacobian(fwd.updates, ctx, m.accuracy)
    K = _jacobian(adj.updates, ctx, m.accuracy)
    # Fields a stage does not write are the identity; neither the forward
    # spec nor the transposer spells that out, so reinstate those rows on
    # both sides before comparing.
    for f in state:
        if f not in fwd.updates:
            for c in active:
                J[(f, c)] = {(f, c): sympy.Integer(1)}
        if f not in adj.updates:
            for c in active:
                K[(f, c)] = {(f, c): sympy.Integer(1)}
    errs: list[str] = []
    keys = set()
    for (g, c1), row in J.items():
        for (f, c2), val in row.items():
            keys.add(((g, c1), (f, c2)))
    for (f, c2), row in K.items():
        for (g, c1), val in row.items():
            keys.add(((g, c1), (f, c2)))
    for (g, c1), (f, c2) in sorted(keys, key=str):
        a = J.get((g, c1), {}).get((f, c2), sympy.Integer(0))
        b = K.get((f, c2), {}).get((g, c1), sympy.Integer(0))
        if sympy.expand(a - b) != 0:
            errs.append(
                f"adjoint mismatch: d({g}@{c1})/d({f}@{c2}) = {a}, "
                f"transposed kernel has {b}"
            )
    return errs


def check_gradients(
    m: Model,
    stage_index: int,
    fwd: Stage,
    shape: tuple[int, ...],
    active: list[tuple[int, ...]],
) -> list[str]:
    ctx = Ctx(m, shape, active)
    errs: list[str] = []
    adj = {
        (f, c): sympy.Symbol("A_" + f + "_" + "_".join(map(str, c)))
        for f in m.spec.state
        for c in active
    }
    contribs = [g for g in m.grads if g.stage == stage_index]
    if not contribs:
        return errs
    fwd_expr = {
        (g, c): sympy.expand(ev(e, c, ctx, m.accuracy))
        for g, e in fwd.updates.items()
        for c in active
    }
    store_expr = {
        (s, c): sympy.expand(ev(m.stores_full[s], c, ctx, m.accuracy))
        for s in m.stores_full
        for c in active
        if m.store_stage[s] == stage_index
    }
    for p in m.spec.params:
        ps = [g for g in contribs if g.param == p]
        if not ps:
            continue
        for j in sorted(active):
            want = sympy.Integer(0)
            pj = ctx.s_par(p, j)
            for (g, c), e in fwd_expr.items():
                if pj not in e.free_symbols:
                    continue
                want += adj[(g, c)] * sympy.diff(e, pj)
            got = sympy.Integer(0)
            for gc in ps:
                c = tuple(a + b for a, b in zip(j, gc.off))
                if c not in ctx.active:
                    continue
                got += store_expr[(gc.store, c)] * adj[(gc.adj_field, c)]
            if sympy.expand(want - got) != 0:
                errs.append(
                    f"model-gradient mismatch for {p} at {j}: "
                    f"symbolic {sympy.expand(want)} vs generated {sympy.expand(got)}"
                )
    return errs


def run(m: Model, active_size: int = 5) -> list[str]:
    """Check every stage of ``m``; returns a list of error strings (empty = pass)."""
    nd = m.spec.ndim
    lo, hi = m.spec.pad(m.accuracy)
    shape = tuple(active_size + lo + hi for _ in range(nd))
    active = [
        tuple(c)
        for c in itertools.product(*[range(lo, lo + active_size) for _ in range(nd)])
    ]
    errs: list[str] = []
    nst = len(m.fwd_plans)
    for si in range(nst):
        fp = m.fwd_plans[si]
        bp = m.bwd_plans[nst - 1 - si]
        int_state = [f for f in m.spec.state if f not in m.spec.pml_state]
        errs += [
            f"[stage {si} full] {e}"
            for e in check_stage(
                m, fp.stage_full, bp.stage_full, shape, active, m.spec.state
            )
        ]
        errs += [
            f"[stage {si} interior] {e}"
            for e in check_stage(
                m, fp.stage_int, bp.stage_int, shape, active, int_state
            )
        ]
        errs += [
            f"[stage {si} grads] {e}"
            for e in check_gradients(m, si, fp.stage_full, shape, active)
        ]
    return errs
