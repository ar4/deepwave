"""Derivation of the adjoint (backward) time step by structural transposition.

Each ``Stage`` is a linear map ``s' = T s`` on the state.  Writing an output as
a sum of terms ``c(i) * f(i + off)``, the matrix entry is
``T[(g,i), (f,i+off)] = c(i)``, so the transpose is ``Tt[(f,j), (g,j-off)] =
c(j-off)`` -- realised by translating the whole term by ``-off``.  No
differentiation of any kind is involved, which is what makes this an
*independent* derivation from the SymPy check in ``symcheck.py``.

The whole time step is a composition of actions, so the adjoint step is the
reversed list of transposed actions.

The model-parameter gradient uses the same algebra: a coefficient ``c(i)``
depending on ``p(i+o)`` contributes ``grad_p[j] += adj_g[j-o] * W[j-o]`` with
``W[i] = dc(i)/dp(i+o) * f(i+off)``, a forward-pass quantity stored during the
forward sweep.
"""

from __future__ import annotations

import dataclasses

from . import ir
from .ir import Expr, Num, Ref, Term
from .spec import Action, Inject, PhysicsSpec, Record, Stage


@dataclasses.dataclass(frozen=True)
class GradContribution:
    stage: int
    param: str
    store: str
    adj_field: str
    off: tuple[int, ...]  # read adj/store at index i + off


def transpose_stage(
    st: Stage,
    state: list[str],
    accuracy: int,
    ndim: int,
    axis_names: tuple[str, ...],
) -> Stage:
    """Structurally transpose one stage (unmentioned fields are the identity)."""
    zero = (0,) * ndim
    contributions: dict[str, list[Term]] = {f: [] for f in state}
    for g in state:
        e = st.updates.get(g, Ref(g, zero))
        for t in ir.linearize(e, accuracy, ndim, axis_names):
            if t.field is None:
                continue
            if t.field not in contributions:
                raise ValueError(f"unknown state field {t.field!r}")
            back = tuple(-o for o in t.off)
            contributions[t.field].append(Term(ir.shift(t.coef, back), g, back))
    out: dict[str, Expr] = {}
    for f, ts in contributions.items():
        e = ir.terms_to_expr(ts)
        if e == Ref(f, zero):  # identity: nothing to write
            continue
        out[f] = e
    return Stage(out)


def transpose_actions(
    actions: list[Action],
    state: list[str],
    accuracy: int,
    ndim: int,
    axis_names: tuple[str, ...],
) -> list[Action]:
    out: list[Action] = []
    for a in reversed(actions):
        if isinstance(a, Stage):
            out.append(transpose_stage(a, state, accuracy, ndim, axis_names))
        elif isinstance(a, Record):
            # r[k] = sum(fields)  ->  each field's adjoint += grad_r[k]
            out.append(Inject(a.fields, a.idx))
        elif isinstance(a, Inject):
            # field += s[k]  ->  grad_s[k] = sum(field adjoints)
            out.append(Record(a.fields, a.idx))
        else:
            raise TypeError(type(a))
    return out


def gradients(
    stages: list[Stage],
    state: list[str],
    params: list[str],
    accuracy: int,
    ndim: int,
    axis_names: tuple[str, ...],
) -> tuple[dict[str, Expr], list[GradContribution]]:
    """Derive the stored forward quantities and the gradient accumulations."""
    stores: dict[str, Expr] = {}
    store_by_key: dict[str, str] = {}
    grads: list[GradContribution] = []
    for si, st in enumerate(stages):
        for p in params:
            for g, e in st.updates.items():
                terms = ir.linearize(e, accuracy, ndim, axis_names)
                offsets: set[tuple[int, ...]] = set()
                for t in terms:
                    for name, off in ir.collect_pars(t.coef):
                        if name == p:
                            offsets.add(off)
                for o in sorted(offsets):
                    parts: list[Term] = []
                    for t in terms:
                        if t.field is None:
                            continue
                        d = ir.deriv_par(t.coef, p, o)
                        if isinstance(d, Num) and d.v == 0:
                            continue
                        parts.append(Term(d, t.field, t.off))
                    if not parts:
                        continue
                    expr = ir.terms_to_expr(parts)
                    k = ir.key(expr)
                    if k in store_by_key:
                        sname = store_by_key[k]
                    else:
                        sname = f"w{si}_{p}_{g}"
                        if any(o):
                            sname += "_" + "_".join(str(v).replace("-", "m") for v in o)
                        stores[sname] = expr
                        store_by_key[k] = sname
                    grads.append(
                        GradContribution(
                            stage=si,
                            param=p,
                            store=sname,
                            adj_field=g,
                            off=tuple(-v for v in o),
                        )
                    )
    return stores, grads
