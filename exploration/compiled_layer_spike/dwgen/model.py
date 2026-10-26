"""Assemble the full generated model for one (spec, accuracy) pair.

Also decides, per stage, whether the state can be updated in place (cheap) or
needs a ping-pong buffer, and in which order the per-staggering field groups
must be swept so that every read sees the pre-stage value.
"""

from __future__ import annotations

import dataclasses

from . import adjoint, ir, spec as spec_mod
from .ir import Expr, Ref
from .spec import Action, Inject, PhysicsSpec, Record, Stage


@dataclasses.dataclass
class Group:
    """Fields swept together because they share a staggering signature."""

    stag: tuple[bool, ...]
    fields: list[str]


@dataclasses.dataclass
class StagePlan:
    stage_full: Stage
    stage_int: Stage
    mode: str  # 'inplace' or 'pingpong'
    groups_full: list[Group]
    groups_int: list[Group]


@dataclasses.dataclass
class Model:
    spec: PhysicsSpec
    accuracy: int
    fwd: list[Action]  # Stage entries carry an index into fwd_plans
    bwd: list[Action]
    fwd_plans: list[StagePlan]
    bwd_plans: list[StagePlan]
    stores_full: dict[str, Expr]
    stores_int: dict[str, Expr]
    grads: list[adjoint.GradContribution]
    store_stage: dict[str, int]
    store_owner: dict[str, str]

    @property
    def stores(self) -> list[str]:
        return list(self.stores_full)


def _nonzero_offset_reads(
    e: Expr, accuracy: int, ndim: int, ax: tuple[str, ...]
) -> set[str]:
    out: set[str] = set()
    for t in ir.linearize(e, accuracy, ndim, ax):
        if t.field is not None and any(t.off):
            out.add(t.field)
    return out


def _all_reads(e: Expr, accuracy: int, ndim: int, ax: tuple[str, ...]) -> set[str]:
    return {t.field for t in ir.linearize(e, accuracy, ndim, ax) if t.field}


def _group(spec: PhysicsSpec, st: Stage) -> list[Group]:
    by: dict[tuple[bool, ...], list[str]] = {}
    for f in st.updates:
        by.setdefault(spec.stag(f), []).append(f)
    return [Group(k, v) for k, v in by.items()]


def _plan_stage(spec: PhysicsSpec, st: Stage, accuracy: int) -> StagePlan:
    nd, ax = spec.ndim, spec.axis_names
    groups = _group(spec, st)
    written = set(st.updates)
    mode = "inplace"
    # A group may not read, at a non-zero offset, a field it writes itself.
    for gr in groups:
        gw = set(gr.fields)
        for f in gr.fields:
            if _nonzero_offset_reads(st.updates[f], accuracy, nd, ax) & gw:
                mode = "pingpong"
    if mode == "inplace":
        # Reader-before-writer ordering across groups.
        idx = {f: i for i, gr in enumerate(groups) for f in gr.fields}
        edges: dict[int, set[int]] = {i: set() for i in range(len(groups))}
        for i, gr in enumerate(groups):
            for f in gr.fields:
                for r in _all_reads(st.updates[f], accuracy, nd, ax):
                    j = idx.get(r)
                    if j is not None and j != i:
                        # group i reads group j's output, so i must be swept
                        # *before* j overwrites it in place
                        edges[i].add(j)
        order: list[int] = []
        remaining = set(range(len(groups)))
        while remaining:
            ready = [
                i
                for i in sorted(remaining)
                if not any(i in edges[k] for k in remaining if k != i)
            ]
            if not ready:
                mode = "pingpong"
                break
            for i in ready:
                order.append(i)
                remaining.discard(i)
        if mode == "inplace":
            groups = [groups[i] for i in order]
    if mode == "pingpong":
        # Every state field must be written so the target buffer is complete.
        zero = (0,) * nd
        upd = dict(st.updates)
        for f in spec.state:
            upd.setdefault(f, Ref(f, zero))
        st = Stage(upd)
        groups = _group(spec, st)
    int_stage = spec_mod.interior_stage(st, spec.pml_state)
    int_groups = [
        Group(gr.stag, [f for f in gr.fields if f in int_stage.updates])
        for gr in groups
    ]
    int_groups = [gr for gr in int_groups if gr.fields]
    return StagePlan(st, int_stage, mode, groups, int_groups)


def build(spec: PhysicsSpec, accuracy: int) -> Model:
    nd, ax = spec.ndim, spec.axis_names
    fwd_stages = [a for a in spec.actions if isinstance(a, Stage)]
    bwd_actions = adjoint.transpose_actions(spec.actions, spec.state, accuracy, nd, ax)
    bwd_stages = [a for a in bwd_actions if isinstance(a, Stage)]

    fwd_plans = [_plan_stage(spec, s, accuracy) for s in fwd_stages]
    bwd_plans = [_plan_stage(spec, s, accuracy) for s in bwd_stages]

    stores_full, grads = adjoint.gradients(
        fwd_stages, spec.state, spec.params, accuracy, nd, ax
    )
    stores_int = {
        k: ir.simplify(spec_mod.zero_pml(v, spec.pml_state))
        for k, v in stores_full.items()
    }
    store_stage: dict[str, int] = {}
    store_owner: dict[str, str] = {}
    for g in grads:
        store_stage.setdefault(g.store, g.stage)
        store_owner.setdefault(g.store, g.adj_field)
    return Model(
        spec=spec,
        accuracy=accuracy,
        fwd=list(spec.actions),
        bwd=bwd_actions,
        fwd_plans=fwd_plans,
        bwd_plans=bwd_plans,
        stores_full=stores_full,
        stores_int=stores_int,
        grads=grads,
        store_stage=store_stage,
        store_owner=store_owner,
    )
