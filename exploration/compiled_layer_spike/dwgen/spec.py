"""Physics specification: the small per-family symbolic description.

A time step is an ordered list of *actions*:

``Stage``   a linear update of some state fields (fields not mentioned are
            unchanged).  Every expression must be linear in the state.
``Record``  ``r[k] = sum(fields)`` at the receiver locations.
``Inject``  ``field += s[k]`` at the source locations, for each field listed.

The adjoint time step is the reversed list with every action transposed, which
is derived mechanically in ``adjoint.py``.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

from . import ir
from .ir import Add, Diff, Expr, Mul, Num, Par, Prof, Ref, Sca


@dataclasses.dataclass(frozen=True)
class Stage:
    updates: dict[str, Expr]


@dataclasses.dataclass(frozen=True)
class Record:
    fields: tuple[str, ...]
    idx: int


@dataclasses.dataclass(frozen=True)
class Inject:
    fields: tuple[str, ...]
    idx: int


Action = Stage | Record | Inject


@dataclasses.dataclass
class PhysicsSpec:
    name: str
    ndim: int
    axis_names: tuple[str, ...]
    state: list[str]
    pml_state: list[str]
    params: list[str]
    profiles: list[str]  # per-axis 1-D arrays, name -> f"{name}{axis}"
    scalars: list[str]
    actions: list[Action]
    n_sources: int
    n_receivers: int
    # half-cell staggering per field; () means all-integer
    stagger: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=dict)
    staggered_grid: bool = False

    def pad(self, accuracy: int) -> tuple[int, int]:
        """(low, high) guard-cell counts of the padded grid."""
        p = accuracy // 2
        return (p, p - 1) if self.staggered_grid else (p, p)

    def stag(self, field: str) -> tuple[bool, ...]:
        return self.stagger.get(field, (False,) * self.ndim)

    def profile_arrays(self) -> list[str]:
        return [f"{p}{a}" for a in self.axis_names for p in self.profiles]

    @property
    def stages(self) -> list[Stage]:
        return [a for a in self.actions if isinstance(a, Stage)]


def zero_pml(e: Expr, pml_state: list[str]) -> Expr:
    """Substitute every PML profile and memory variable with zero."""
    if isinstance(e, Prof):
        return Num(Fraction(0))
    if isinstance(e, Ref):
        return Num(Fraction(0)) if e.name in pml_state else e
    if isinstance(e, (Num, Sca, Par)):
        return e
    if isinstance(e, Add):
        return ir.add(*[zero_pml(a, pml_state) for a in e.args])
    if isinstance(e, Mul):
        return ir.mul(*[zero_pml(a, pml_state) for a in e.args])
    if isinstance(e, Diff):
        arg = zero_pml(e.arg, pml_state)
        if isinstance(arg, Num) and arg.v == 0:
            return Num(Fraction(0))
        return Diff(e.axis, e.kind, arg)
    raise TypeError(type(e))


def interior_stage(st: Stage, pml_state: list[str]) -> Stage:
    """The interior (PML-free) variant: PML state is structurally absent."""
    return Stage(
        {
            k: ir.simplify(zero_pml(v, pml_state))
            for k, v in st.updates.items()
            if k not in pml_state
        }
    )
