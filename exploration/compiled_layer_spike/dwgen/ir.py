"""Symbolic intermediate representation for finite-difference stencil updates.

An update is an expression tree over

* ``Ref``   -- a sample of a *state* field (wavefield or PML memory variable)
* ``Par``   -- a sample of a full-grid *model parameter* array
* ``Prof``  -- a sample of a per-axis 1-D profile array (PML a/b/db)
* ``Sca``   -- a runtime scalar (dt, 1/dx, ...)
* ``Num``   -- an exact rational constant
* ``Add`` / ``Mul`` / ``Diff``

``Diff`` is a finite-difference operator along one axis.  Four kinds are
supported::

    'c1'  centred first derivative      (taps at +-1 .. +-a/2)
    'c2'  centred second derivative     (taps at 0, +-1 .. +-a/2)
    'f1'  staggered first derivative, forward  (taps at -a/2+1 .. a/2)
    'b1'  staggered first derivative, backward (taps at -a/2 .. a/2-1)

The whole update must be *linear in the state fields*: every product may
contain at most one ``Ref``.  That is what makes the adjoint derivable by
structural transposition (see ``adjoint.py``).
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from typing import Union

Number = Union[int, Fraction]


class Expr:
    """Base class for expression nodes."""

    def __add__(self, other: "Expr | Number") -> "Expr":
        return add(self, lift(other))

    def __radd__(self, other: "Expr | Number") -> "Expr":
        return add(lift(other), self)

    def __sub__(self, other: "Expr | Number") -> "Expr":
        return add(self, neg(lift(other)))

    def __rsub__(self, other: "Expr | Number") -> "Expr":
        return add(lift(other), neg(self))

    def __mul__(self, other: "Expr | Number") -> "Expr":
        return mul(self, lift(other))

    def __rmul__(self, other: "Expr | Number") -> "Expr":
        return mul(lift(other), self)

    def __neg__(self) -> "Expr":
        return neg(self)


@dataclasses.dataclass(frozen=True)
class Num(Expr):
    v: Fraction


@dataclasses.dataclass(frozen=True)
class Sca(Expr):
    name: str


@dataclasses.dataclass(frozen=True)
class Ref(Expr):
    name: str
    off: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Par(Expr):
    name: str
    off: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Prof(Expr):
    name: str
    axis: int
    off: int


@dataclasses.dataclass(frozen=True)
class Add(Expr):
    args: tuple[Expr, ...]


@dataclasses.dataclass(frozen=True)
class Mul(Expr):
    args: tuple[Expr, ...]


@dataclasses.dataclass(frozen=True)
class Diff(Expr):
    axis: int
    kind: str
    arg: Expr


def lift(x: "Expr | Number") -> Expr:
    if isinstance(x, Expr):
        return x
    return Num(Fraction(x))


def add(*args: Expr) -> Expr:
    flat: list[Expr] = []
    for a in args:
        if isinstance(a, Add):
            flat.extend(a.args)
        elif isinstance(a, Num) and a.v == 0:
            continue
        else:
            flat.append(a)
    if not flat:
        return Num(Fraction(0))
    if len(flat) == 1:
        return flat[0]
    return Add(tuple(flat))


def mul(*args: Expr) -> Expr:
    flat: list[Expr] = []
    const = Fraction(1)
    for a in args:
        if isinstance(a, Mul):
            for b in a.args:
                if isinstance(b, Num):
                    const *= b.v
                else:
                    flat.append(b)
        elif isinstance(a, Num):
            const *= a.v
        else:
            flat.append(a)
    if const == 0:
        return Num(Fraction(0))
    out: list[Expr] = []
    if const != 1 or not flat:
        out.append(Num(const))
    out.extend(flat)
    if len(out) == 1:
        return out[0]
    return Mul(tuple(out))


def neg(a: Expr) -> Expr:
    return mul(Num(Fraction(-1)), a)


# ---------------------------------------------------------------------------
# finite-difference tap tables
# ---------------------------------------------------------------------------

_C1: dict[int, list[Fraction]] = {
    2: [Fraction(1, 2)],
    4: [Fraction(8, 12), Fraction(-1, 12)],
    6: [Fraction(3, 4), Fraction(-3, 20), Fraction(1, 60)],
    8: [Fraction(4, 5), Fraction(-1, 5), Fraction(4, 105), Fraction(-1, 280)],
}

_C2: dict[int, tuple[Fraction, list[Fraction]]] = {
    2: (Fraction(-2), [Fraction(1)]),
    4: (Fraction(-5, 2), [Fraction(4, 3), Fraction(-1, 12)]),
    6: (Fraction(-49, 18), [Fraction(3, 2), Fraction(-3, 20), Fraction(1, 90)]),
    8: (
        Fraction(-205, 72),
        [Fraction(8, 5), Fraction(-1, 5), Fraction(8, 315), Fraction(-1, 560)],
    ),
}

# Staggered (half-cell) first derivative coefficients: value at cell centre is
# built from taps at +-1/2, +-3/2, ...  Expressed on the integer index grid,
# the 'forward' variant reads offsets 1-k .. k and the 'backward' variant
# reads offsets -k .. k-1.
_S1: dict[int, list[Fraction]] = {
    2: [Fraction(1)],
    4: [Fraction(9, 8), Fraction(-1, 24)],
    6: [Fraction(75, 64), Fraction(-25, 384), Fraction(3, 640)],
    8: [
        Fraction(1225, 1024),
        Fraction(-245, 3072),
        Fraction(49, 5120),
        Fraction(-5, 7168),
    ],
}


def taps(kind: str, accuracy: int) -> list[tuple[int, Fraction]]:
    """Return ``[(shift, coefficient), ...]`` for a difference operator kind."""
    if kind == "c1":
        out = []
        for k, c in enumerate(_C1[accuracy], start=1):
            out.append((k, c))
            out.append((-k, -c))
        return out
    if kind == "c2":
        c0, cs = _C2[accuracy]
        out = [(0, c0)]
        for k, c in enumerate(cs, start=1):
            out.append((k, c))
            out.append((-k, c))
        return out
    if kind in ("f1", "b1"):
        out = []
        for k, c in enumerate(_S1[accuracy], start=1):
            if kind == "f1":
                out.append((k, c))
                out.append((1 - k, -c))
            else:
                out.append((k - 1, c))
                out.append((-k, -c))
        return out
    raise ValueError(f"unknown difference kind {kind!r}")


def diff_scale(kind: str, axis: int, axis_names: tuple[str, ...]) -> Expr:
    """The runtime scale factor (1/dx or 1/dx^2) that a difference carries."""
    a = axis_names[axis]
    if kind == "c2":
        return Sca(f"rd{a}2")
    return Sca(f"rd{a}")


def fd_pad(accuracy: int) -> int:
    return accuracy // 2


# ---------------------------------------------------------------------------
# structural helpers
# ---------------------------------------------------------------------------


def shift(e: Expr, s: tuple[int, ...]) -> Expr:
    """Translate every position-dependent access in ``e`` by ``s``."""
    if isinstance(e, Ref):
        return Ref(e.name, tuple(a + b for a, b in zip(e.off, s)))
    if isinstance(e, Par):
        return Par(e.name, tuple(a + b for a, b in zip(e.off, s)))
    if isinstance(e, Prof):
        return Prof(e.name, e.axis, e.off + s[e.axis])
    if isinstance(e, (Num, Sca)):
        return e
    if isinstance(e, Add):
        return Add(tuple(shift(a, s) for a in e.args))
    if isinstance(e, Mul):
        return Mul(tuple(shift(a, s) for a in e.args))
    if isinstance(e, Diff):
        return Diff(e.axis, e.kind, shift(e.arg, s))
    raise TypeError(type(e))


@dataclasses.dataclass(frozen=True)
class Term:
    """A single linear term: ``coef`` (state-free) times ``field`` at ``off``.

    ``field`` is ``None`` for a term that does not touch the state at all.
    """

    coef: Expr
    field: str | None
    off: tuple[int, ...]


def linearize(
    e: Expr, accuracy: int, ndim: int, axis_names: tuple[str, ...]
) -> list[Term]:
    """Flatten ``e`` into a sum of terms, expanding every ``Diff``.

    Raises ``ValueError`` if the expression is not linear in the state.
    """
    zero = (0,) * ndim
    if isinstance(e, Ref):
        return [Term(Num(Fraction(1)), e.name, e.off)]
    if isinstance(e, (Num, Sca, Par, Prof)):
        return [Term(e, None, zero)]
    if isinstance(e, Add):
        out: list[Term] = []
        for a in e.args:
            out.extend(linearize(a, accuracy, ndim, axis_names))
        return out
    if isinstance(e, Mul):
        acc: list[Term] = [Term(Num(Fraction(1)), None, zero)]
        for a in e.args:
            sub = linearize(a, accuracy, ndim, axis_names)
            new: list[Term] = []
            for t1 in acc:
                for t2 in sub:
                    if t1.field is not None and t2.field is not None:
                        raise ValueError("update is not linear in the state fields")
                    field = t1.field if t1.field is not None else t2.field
                    off = t1.off if t1.field is not None else t2.off
                    new.append(Term(mul(t1.coef, t2.coef), field, off))
            acc = new
        return acc
    if isinstance(e, Diff):
        scale = diff_scale(e.kind, e.axis, axis_names)
        sub = linearize(e.arg, accuracy, ndim, axis_names)
        out = []
        for s, c in taps(e.kind, accuracy):
            sv = [0] * ndim
            sv[e.axis] = s
            svt = tuple(sv)
            for t in sub:
                out.append(
                    Term(
                        mul(Num(c), scale, shift(t.coef, svt)),
                        t.field,
                        tuple(a + b for a, b in zip(t.off, svt)) if t.field else t.off,
                    )
                )
        return out
    raise TypeError(type(e))


def terms_to_expr(terms: list[Term]) -> Expr:
    """Rebuild an expression from terms, grouping by field access offset."""
    groups: dict[tuple[str | None, tuple[int, ...]], list[Expr]] = {}
    order: list[tuple[str | None, tuple[int, ...]]] = []
    for t in terms:
        key = (t.field, t.off)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(t.coef)
    parts: list[Expr] = []
    for key in order:
        field, off = key
        coef = simplify(add(*groups[key]))
        if isinstance(coef, Num) and coef.v == 0:
            continue
        if field is None:
            parts.append(coef)
        else:
            parts.append(mul(coef, Ref(field, off)))
    return simplify(add(*parts))


def simplify(e: Expr) -> Expr:
    """Constant-fold and collapse trivial Add/Mul nodes."""
    if isinstance(e, Add):
        args = [simplify(a) for a in e.args]
        const = Fraction(0)
        rest: list[Expr] = []
        for a in args:
            if isinstance(a, Num):
                const += a.v
            else:
                rest.append(a)
        if const != 0:
            rest.append(Num(const))
        return add(*rest)
    if isinstance(e, Mul):
        return mul(*[simplify(a) for a in e.args])
    if isinstance(e, Diff):
        return Diff(e.axis, e.kind, simplify(e.arg))
    return e


def collect_refs(e: Expr) -> set[str]:
    if isinstance(e, Ref):
        return {e.name}
    if isinstance(e, (Num, Sca, Par, Prof)):
        return set()
    if isinstance(e, (Add, Mul)):
        out: set[str] = set()
        for a in e.args:
            out |= collect_refs(a)
        return out
    if isinstance(e, Diff):
        return collect_refs(e.arg)
    raise TypeError(type(e))


def collect_pars(e: Expr) -> set[tuple[str, tuple[int, ...]]]:
    if isinstance(e, Par):
        return {(e.name, e.off)}
    if isinstance(e, (Num, Sca, Ref, Prof)):
        return set()
    if isinstance(e, (Add, Mul)):
        out: set[tuple[str, tuple[int, ...]]] = set()
        for a in e.args:
            out |= collect_pars(a)
        return out
    if isinstance(e, Diff):
        return collect_pars(e.arg)
    raise TypeError(type(e))


def deriv_par(e: Expr, name: str, off: tuple[int, ...]) -> Expr:
    """Symbolic derivative of a *state-free* expression w.r.t. ``Par(name, off)``."""
    if isinstance(e, Par):
        return (
            Num(Fraction(1)) if (e.name == name and e.off == off) else Num(Fraction(0))
        )
    if isinstance(e, (Num, Sca, Prof, Ref)):
        return Num(Fraction(0))
    if isinstance(e, Add):
        return add(*[deriv_par(a, name, off) for a in e.args])
    if isinstance(e, Mul):
        parts: list[Expr] = []
        for i, a in enumerate(e.args):
            d = deriv_par(a, name, off)
            if isinstance(d, Num) and d.v == 0:
                continue
            others = list(e.args[:i]) + list(e.args[i + 1 :])
            parts.append(mul(d, *others))
        return add(*parts)
    raise TypeError(type(e))


def key(e: Expr) -> str:
    """A canonical string key, used to deduplicate structurally equal trees."""
    if isinstance(e, Num):
        return f"N{e.v}"
    if isinstance(e, Sca):
        return f"S{e.name}"
    if isinstance(e, Ref):
        return f"R{e.name}{e.off}"
    if isinstance(e, Par):
        return f"P{e.name}{e.off}"
    if isinstance(e, Prof):
        return f"F{e.name}{e.axis},{e.off}"
    if isinstance(e, Add):
        return "A(" + "+".join(sorted(key(a) for a in e.args)) + ")"
    if isinstance(e, Mul):
        return "M(" + "*".join(sorted(key(a) for a in e.args)) + ")"
    if isinstance(e, Diff):
        return f"D{e.kind}{e.axis}(" + key(e.arg) + ")"
    raise TypeError(type(e))
