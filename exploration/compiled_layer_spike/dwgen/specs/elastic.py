"""Isotropic elastic wave equation, velocity-stress on a staggered grid, C-PML.

Two stages per time step: velocities from stresses, then stresses from the
(source-injected) velocities.  Body-force sources arrive already scaled by
``buoyancy * dt`` and the pressure source by ``-dt`` (done in the front-end,
where autograd sees it).

Staggering (2-D, axes ``y`` then ``x``): normal stresses and the ``m_v?y``/
``m_v?x`` memory variables sit on integer nodes, ``vy`` at ``y+1/2``, ``vx``
at ``x+1/2``, ``sigmaxy`` at both.  A half-node result pairs with the ``ah``/
``bh`` profile of its axis, an integer-node result with ``a``/``b``.
"""

from __future__ import annotations

from ..ir import Diff, Num, Par, Prof, Ref, Sca, add, mul
from ..spec import Inject, PhysicsSpec, Record, Stage

T, F = True, False


def make(ndim: int) -> PhysicsSpec:
    if ndim != 2:
        raise NotImplementedError("the spike implements 2-D elastic only")
    axis_names = ("y", "x")
    Y, X = 0, 1
    z2 = (0, 0)

    def R(n):
        return Ref(n, z2)

    def P(n):
        return Par(n, z2)

    dt = Sca("dt")

    stagger = {
        "vy": (T, F),
        "vx": (F, T),
        "sigmayy": (F, F),
        "sigmaxx": (F, F),
        "sigmaxy": (T, T),
        "m_vyy": (F, F),
        "m_vxx": (F, F),
        "m_vxy": (T, T),
        "m_vyx": (T, T),
        "m_sigmayyy": (T, F),
        "m_sigmaxyx": (T, F),
        "m_sigmaxyy": (F, T),
        "m_sigmaxxx": (F, T),
    }
    pml_state = [
        "m_vyy",
        "m_vxx",
        "m_vxy",
        "m_vyx",
        "m_sigmayyy",
        "m_sigmaxyx",
        "m_sigmaxyy",
        "m_sigmaxxx",
    ]
    state = ["vy", "vx", "sigmayy", "sigmaxx", "sigmaxy"] + pml_state

    def pml_corrected(raw, mem, ax, half):
        """d + m', where m' = a*m + b*d: the C-PML-corrected derivative."""
        pa = Prof("ah" if half else "a", ax, 0)
        pb = Prof("bh" if half else "b", ax, 0)
        new_mem = add(mul(pa, R(mem)), mul(pb, raw))
        return add(raw, new_mem), new_mem

    # ---- stage 1: velocities from stresses -------------------------------
    dyh_syy = Diff(Y, "f1", R("sigmayy"))
    ty, m_syyy = pml_corrected(dyh_syy, "m_sigmayyy", Y, half=True)
    dx_sxy = Diff(X, "b1", R("sigmaxy"))
    tx, m_sxyx = pml_corrected(dx_sxy, "m_sigmaxyx", X, half=False)

    dy_sxy = Diff(Y, "b1", R("sigmaxy"))
    uy, m_sxyy = pml_corrected(dy_sxy, "m_sigmaxyy", Y, half=False)
    dxh_sxx = Diff(X, "f1", R("sigmaxx"))
    ux, m_sxxx = pml_corrected(dxh_sxx, "m_sigmaxxx", X, half=True)

    vel = {
        "vy": add(R("vy"), mul(P("buoyancy_y"), dt, add(ty, tx))),
        "m_sigmayyy": m_syyy,
        "m_sigmaxyx": m_sxyx,
        "vx": add(R("vx"), mul(P("buoyancy_x"), dt, add(uy, ux))),
        "m_sigmaxyy": m_sxyy,
        "m_sigmaxxx": m_sxxx,
    }

    # ---- stage 2: stresses from velocities --------------------------------
    dvydy_raw = Diff(Y, "b1", R("vy"))
    dvydy, m_vyy = pml_corrected(dvydy_raw, "m_vyy", Y, half=False)
    dvxdx_raw = Diff(X, "b1", R("vx"))
    dvxdx, m_vxx = pml_corrected(dvxdx_raw, "m_vxx", X, half=False)
    w = mul(P("lamb"), add(dvydy, dvxdx))

    dvxdy_raw = Diff(Y, "f1", R("vx"))
    dvxdy, m_vxy = pml_corrected(dvxdy_raw, "m_vxy", Y, half=True)
    dvydx_raw = Diff(X, "f1", R("vy"))
    dvydx, m_vyx = pml_corrected(dvydx_raw, "m_vyx", X, half=True)

    stress = {
        "sigmayy": add(R("sigmayy"), mul(dt, add(w, mul(Num(2), P("mu"), dvydy)))),
        "sigmaxx": add(R("sigmaxx"), mul(dt, add(w, mul(Num(2), P("mu"), dvxdx)))),
        "m_vyy": m_vyy,
        "m_vxx": m_vxx,
        "sigmaxy": add(R("sigmaxy"), mul(dt, P("mu_yx"), add(dvxdy, dvydx))),
        "m_vxy": m_vxy,
        "m_vyx": m_vyx,
    }

    return PhysicsSpec(
        name="elastic",
        ndim=2,
        axis_names=axis_names,
        state=state,
        pml_state=pml_state,
        params=["lamb", "mu", "mu_yx", "buoyancy_y", "buoyancy_x"],
        profiles=["a", "b", "ah", "bh"],
        scalars=["dt", "rdy", "rdx"],
        actions=[
            Record(("vy",), 0),
            Record(("vx",), 1),
            Record(("sigmayy", "sigmaxx"), 2),
            Stage(vel),
            Inject(("vy",), 0),
            Inject(("vx",), 1),
            Stage(stress),
            Inject(("sigmayy", "sigmaxx"), 2),
        ],
        n_sources=3,
        n_receivers=3,
        stagger=stagger,
        staggered_grid=True,
    )
