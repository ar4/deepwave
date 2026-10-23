# Source/receiver location coordinate semantics — decision record

*Status: the **unit** of float coordinates is decided (grid cells). A separate
question — a physical-unit `PointSource`-style specification object — is
**deferred** for later consideration (see "Open question").*

This note records a public-API design discussion so it can be resumed without
re-deriving it. It concerns how source/receiver locations are interpreted,
especially for off-grid placement and for the staggered (acoustic/elastic)
propagators.

## Background

- Integer (`torch.long`) source/receiver locations are **array indices** into
  each field component's grid. The exact-injection promise (integer cells do
  exactly what is asked; byte-identical across versions) is a stated core
  value, enforced by the integer-locations byte-identical tests in
  `tests/test_location_interpolation.py`.
- Off-grid placement uses `deepwave.Hicks`
  (Kaiser-windowed sinc), which works in **grid-cell** units and has a
  per-component half-cell `offset` to handle staggering.
- Staggered grids (`docs/acoustic.rst`, `docs/elastic.rst`): pressure / normal
  stresses and the model parameters sit on integer nodes `(y,x)`; velocity
  components are half a cell off — `vy` at `(y+½, x)`, `vx` at `(y, x+½)`; shear
  stress `σxy` at `(y+½, x+½)`. So integer index `i` of a velocity component is
  physically at `i + 0.5` cells.

## Decision (locked): float coordinates are in **grid cells**

- **Integer/`long` location** → array index, exact, byte-identical to before.
- **Float location** → continuous position in grid cells; interpolated via Hicks
  with the field component's half-cell `offset` removed
  (`round(location − offset)` for the on-node/exact case).
- **Key invariant:** integer `N` and float `N.0` denote the **same physical
  location**. For a staggered component, integer `N` equals float `N + offset`
  (e.g. a `vy` source at integer `3` equals float `3.5`). The half-cell shift is
  the natural "rounding" relationship between the discrete index and the
  continuous coordinate — one unit, two readings that coincide on the overlap.

### Rejected alternatives

- **Float = component array index (no offset).** Self-consistent but forces the
  user to know the stagger to place a physical point; defeats the feature.
- **All locations physical (break compat).** Silently shifts *and* interpolates
  every existing integer velocity source and destroys exact injection — an
  invisible, correctness-degrading break. Disqualified.
- **Float = physical length (same units as `grid_spacing`, e.g. metres).**
  Tempting (matches survey geometry, makes the magnitudes of cells vs physical
  differ). Rejected: it **overloads one argument with two units by dtype** —
  `tensor([745])` = cell 745 but `tensor([745.0])` = 745 m = a different cell. A
  stray `.float()` or trailing dot would silently move a source ~`dx`-fold and
  **poison an FWI gradient with no error**. It does **not** cure the half-cell
  stagger surprise (intrinsic to staggered grids, not a unit issue); it is
  *worst* at `grid_spacing ≈ 1` (most tests/tutorials, where metres ≡ cells); and
  it breaks the integer `IGNORE_LOCATION` sentinel under the divide. Two
  independent adversarial reviews (API-architect lens and heavy-user lens)
  concurred. The invariant "integer `N` and float `N.0` are the same location" is
  the most valuable safety property, and only cell-units preserves it.

### Conveniences to add (consistent with the decision; not yet implemented)

- An explicit metres→cells path — a small `to_cells(positions, grid_spacing)`
  helper and/or documented `positions / grid_spacing` — never by overloading the
  float dtype.
- Improve the out-of-bounds error (`common.check_locations_are_within_model`,
  `common.py:1687`) to hint: "value looks like physical metres? locations are in
  grid cells — divide by `grid_spacing`."

## Open question (deferred — revisit when ready)

A dedicated source/receiver **specification object** (working name
`PointSource`) could carry locations in **physical units even when given as
integers**. Because a distinct *object* signals the unit unambiguously, there is
**no int-vs-float dtype overload** — which is exactly the danger that sank the
"float = metres on a raw tensor" option. So this is a clean way to offer
physical-unit geometry while raw Tensors stay in cells.

Sketch (to be designed): `PointSource(locations, monopole=True, dipole_dim=0,
interpolation=None)` accepted anywhere a location tensor is accepted. It would:

- carry the user's intent (physical-or-cell locations + monopole/dipole
  directivity), but **not** `offset`, `model_shape`, or `free_surfaces` — the
  **propagator** owns those, so it attaches each component's correct half-cell
  offset and the user cannot pick a wrong one;
- give a single, self-documenting idiom for dipoles / moment-tensor sources on
  acoustic/elastic (today moment tensors are built by hand-splitting forces
  across cells; for scalar, dipoles currently go through a `Hicks` object).

Trade-off to weigh: it introduces a second idiom for "how do I place a source"
(raw Tensor vs object), a teachability/maintenance cost. Decide this **together
with the acoustic/elastic float-coordinate wiring**, since that is when the
staggered offset and the physical-unit question first bite a real user.

## Implementation status (as of this note)

- `Hicks`: generalised to 1/2/3-D with a per-component `offset`; cell units;
  `IGNORE_LOCATION`-aware; `should_interpolate`/`grid_locations` are offset-aware.
  (An ignored point's safe placeholder coordinate is now the component's own node
  (`offset`) rather than a bare `0`, so a staggered component's disabled point no
  longer produces an out-of-model negative stencil cell.)
- `scalar`: float coordinates and the `Hicks`-object (dipole) route are wired
  (offset 0, regular grid); integer inputs byte-identical.
- `acoustic` / `elastic` / `scalar_born`: **now wired** — each per-component
  location argument gets its own offset-aware interpolator, source-amplitude
  spread, and receiver gather-back, via the shared helpers
  `deepwave.common.resolve_source_interpolators` /
  `resolve_receiver_interpolators` / `apply_receiver_interpolators` (driven by
  `velocity_component_offsets`); `scalar` uses the same helpers as the
  zero-offset case. Integer inputs remain byte-identical. The new
  `location_interpolation` / `location_interpolation_halfwidth` kwargs mirror
  `scalar`.
- `PointSource`: still **not** added — per the maintainer's decision (see
  "Rejected alternatives" and "Open question" above) it remains rejected; raw
  Tensors (cells) and `Hicks` objects are the two supported idioms.
- Per-component offsets (authoritative, from the docs) as wired:
  pressure / normal stress `[0,0]`; `vy` `[0.5,0]`; `vx` `[0,0.5]`;
  `σxy` `[0.5,0.5]` (staggered but has no public source/receiver argument, so it
  is not exposed). (Component letter → spatial dim: `x` is the last dim, `y`
  second-last, `z` third-last; in 3-D `vz` is `[0.5,0,0]` and `vy` is
  `[0,0.5,0]`.) The on-node consistency test (`float(N+offset)` ≡ integer `N`) is
  covered per component in `tests/test_location_interpolation.py`.
