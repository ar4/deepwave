"""Does the doubly-derived adjoint check actually catch anything?

Four bugs are injected into the *derived* adjoint / gradient model -- each one
a mistake a human hand-writing an adjoint kernel plausibly makes -- and the
SymPy check is run against the (unmodified) forward update.  For each bug the
script reports whether the check flags it.

The injection points are the outputs of ``adjoint.transpose_stage`` and
``adjoint.gradients``, i.e. exactly the objects the C/CUDA emitters consume,
so a bug that survives here would be emitted into the shipped kernel.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from fractions import Fraction

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dwgen import adjoint, ir, model, symcheck
from dwgen.ir import Add, Mul, Num, Prof, Sca, Term
from dwgen.spec import Stage
from dwgen.specs import scalar as scalar_spec


def _map_expr(e, fn):
    e2 = fn(e)
    if e2 is not e:
        return e2
    if isinstance(e, Add):
        return ir.add(*[_map_expr(a, fn) for a in e.args])
    if isinstance(e, Mul):
        return ir.mul(*[_map_expr(a, fn) for a in e.args])
    return e


def bug_sign_flip(m: model.Model) -> str:
    """Flip the sign of one off-centre term in the transposed wavefield update.

    The classic hand-adjoint mistake: the transpose of a first derivative is
    *minus* the transposed stencil, and it is easy to transpose the stencil
    without flipping the sign (or vice versa).
    """
    plan = m.bwd_plans[0]
    upd = dict(plan.stage_full.updates)
    terms = ir.linearize(upd["u"], m.accuracy, m.spec.ndim, m.spec.axis_names)
    off = (0, 1)
    upd["u"] = ir.terms_to_expr(
        [
            Term(ir.mul(Num(Fraction(-1)), t.coef), t.field, t.off)
            if t.off == off
            else t
            for t in terms
        ]
    )
    m.bwd_plans[0] = dataclasses.replace(plan, stage_full=Stage(upd))
    return "adjoint of u: sign flipped on the +1x neighbour"


def bug_missing_scale(m: model.Model) -> str:
    """Drop the dt^2 factor from the stored model-gradient quantity.

    A missing constant scale is invisible in a forward-only test and produces
    a gradient that points the right way but is wrong by a constant factor.
    """
    name = next(iter(m.stores_full))

    def drop(x):
        if isinstance(x, Sca) and x.name == "dt2":
            return Num(Fraction(1))
        return x

    m.stores_full[name] = _map_expr(m.stores_full[name], drop)
    m.stores_int[name] = _map_expr(m.stores_int[name], drop)
    return "model-gradient store: dt^2 factor dropped"


def bug_wrong_field_pairing(m: model.Model) -> str:
    """Correlate the stored forward field with the wrong adjoint wavefield.

    ``u`` and ``p`` are both plausible-looking wavefields at the same grid
    points; using the previous wavefield instead of the current one in the
    imaging condition is a real, hard-to-eyeball error.
    """
    m.grads = [dataclasses.replace(g, adj_field="p") for g in m.grads]
    return "model gradient: correlated against adjoint p instead of adjoint u"


def bug_profile_offset(m: model.Model) -> str:
    """Sample a PML profile at the cell instead of at the shifted stencil tap.

    The transpose must translate the position-dependent *coefficients* as well
    as the field access; forgetting to do so is the easiest thing to get wrong
    when transposing by hand, and it only shows up inside the absorbing layer.
    """
    plan = m.bwd_plans[0]
    upd = dict(plan.stage_full.updates)

    def shift_prof(x):
        if isinstance(x, Prof) and x.name == "b" and x.off != 0:
            return Prof(x.name, x.axis, 0)
        return x

    for k in list(upd):
        upd[k] = _map_expr(upd[k], shift_prof)
    m.bwd_plans[0] = dataclasses.replace(plan, stage_full=Stage(upd))
    return "adjoint PML: b-profile read at the cell, not at the shifted tap"


BUGS = [bug_sign_flip, bug_missing_scale, bug_wrong_field_pairing, bug_profile_offset]


def main() -> None:
    spec = scalar_spec.make(2)
    m = model.build(spec, 4)
    errs = symcheck.run(m, active_size=3)
    print("baseline (no bug injected)")
    print(
        f"  doubly-derived adjoint check: {len(errs)} mismatches -> "
        f"{'PASS' if not errs else 'FAIL'}\n"
    )
    for bug in BUGS:
        m = model.build(spec, 4)
        desc = bug(m)
        errs = symcheck.run(m, active_size=3)
        print(bug.__name__)
        print(f"  injected : {desc}")
        print(
            f"  result   : {'CAUGHT' if errs else 'MISSED'} "
            f"({len(errs)} symbolic mismatches)"
        )
        if errs:
            print(f"  example  : {errs[0][:180]}")
        print()


if __name__ == "__main__":
    main()
