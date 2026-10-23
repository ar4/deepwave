"""Constant-density acoustic (scalar) wave equation with C-PML.

    d^2 u / dt^2 = v^2 laplacian(u) - v^2 f

second order in time, ``accuracy``-order in space, C-PML absorbing faces.
Source amplitudes arrive already scaled by ``-v^2 dt^2`` (applied in the
front-end, where autograd sees it), so the kernel just adds them to the new
wavefield.

State: the wavefield ``u``, its previous value ``p``, and per axis the two
C-PML memory variables ``psi<axis>`` and ``zeta<axis>``.
"""

from __future__ import annotations

from ..ir import Diff, Num, Par, Prof, Ref, Sca, add, mul
from ..spec import Inject, PhysicsSpec, Record, Stage


def make(ndim: int) -> PhysicsSpec:
    axis_names = ("z", "y", "x")[3 - ndim :]
    zero = (0,) * ndim
    u = Ref("u", zero)
    p = Ref("p", zero)
    v = Par("v", zero)
    dt2 = Sca("dt2")

    pml_state: list[str] = []
    for a in axis_names:
        pml_state += [f"psi{a}", f"zeta{a}"]
    state = ["u", "p"] + pml_state

    updates = {}
    w_sum_parts = []
    for k, a in enumerate(axis_names):
        psi = Ref(f"psi{a}", zero)
        zeta = Ref(f"zeta{a}", zero)
        b = Prof("b", k, 0)
        aa = Prof("a", k, 0)
        db = Prof("db", k, 0)
        du = Diff(k, "c1", u)
        a_psi = mul(Prof("a", k, 0), Ref(f"psi{a}", zero))
        tmp = add(
            mul(add(Num(1), b), Diff(k, "c2", u)),
            mul(db, du),
            Diff(k, "c1", a_psi),
        )
        w_sum_parts.append(add(mul(add(Num(1), b), tmp), mul(aa, zeta)))
        updates[f"psi{a}"] = add(mul(b, du), mul(aa, psi))
        updates[f"zeta{a}"] = add(mul(b, tmp), mul(aa, zeta))
    w_sum = add(*w_sum_parts)
    updates["u"] = add(mul(v, v, dt2, w_sum), mul(Num(2), u), mul(Num(-1), p))
    updates["p"] = u

    scalars = ["dt2"]
    scalars += [f"rd{a}" for a in axis_names]
    scalars += [f"rd{a}2" for a in axis_names]

    return PhysicsSpec(
        name="scalar",
        ndim=ndim,
        axis_names=axis_names,
        state=state,
        pml_state=pml_state,
        params=["v"],
        profiles=["a", "b", "db"],
        scalars=scalars,
        actions=[
            Record(("u",), 0),
            Stage(updates),
            Inject(("u",), 0),
        ],
        n_sources=1,
        n_receivers=1,
        staggered_grid=False,
    )
