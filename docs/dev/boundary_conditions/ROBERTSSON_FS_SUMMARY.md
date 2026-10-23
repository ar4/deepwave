# Robertsson (1996) free-surface method — digest and Deepwave cost assessment

Source: `robertsson_elastic_free_surface_geophysics.pdf` —
Johan O. A. Robertsson, "A numerical free-surface condition for elastic/viscoelastic
finite-difference modeling in the presence of topography", *Geophysics* 61(6),
1921–1934 (1996). 2-D presentation; 3-D stated to be "relatively straightforward
and obvious" from the 2-D description (not actually carried out in the paper).
Built on the Virieux (1986) / Levander (1988) staggered velocity–stress scheme,
O(2,4) accuracy, and the Robertsson–Blanch–Symes (1994) viscoelastic memory-variable
formulation.

This is a self-contained digest so the PDF need not be re-read.

---

## PART 1 — METHOD SUMMARY

### 1.1 The free-surface condition being approximated
The physical free surface requires the traction normal to the surface to vanish.
For a flat horizontal surface in 2-D (z = depth), this is
`sigma_zz = 0` and `sigma_xz = 0` (and the corresponding memory variables
`r_zz = 0`, `r_xz = 0`). Equivalently the surface is enforced via the velocity
gradients (paper eqs. 4–5): `dvx/dz = -dvz/dx` and
`dvz/dz = -(1 - 2 mu/(lambda+2mu)) dvx/dx`. Note the high-Poisson factor
`(1 - 2 tau^s mu / (tau^p pi))` in the viscoelastic case reduces to the elastic
`(1 - 2 mu/(lambda+2mu))`.

### 1.2 Staggered-grid node geometry (critical to the method)
Within each staggered cell (paper Fig. 3): the normal stresses and their memory
variables `sigma_xx, sigma_zz, r_xx, r_zz` sit in the upper-left corner (the
"integer" node); `vx` is immediately to the right; `vz` immediately below; and the
shear stress `sigma_xz, r_xz` in the lower-right corner. **The free surface plane is
placed exactly through the row of `sigma_xx, sigma_zz, r_xx, r_zz, and vx` nodes
(z = 0).** That is, the surface runs through the normal-stress / horizontal-velocity
row. `vz` and `sigma_xz` live half a cell off the surface.

### 1.3 What is done each timestep — the image (mirroring) method (flat surface)
Robertsson favours "**image method 3**" (his own variant of Levander 1988):

1. **Zero the surface-normal/shear stresses on the surface row.** Set `sigma_zz = 0`
   and the components that must vanish there. Because of the staggering, the
   components that need to be defined *above* the surface for the FD stencil are
   `sigma_zz` and `sigma_xz` (and `r_zz`, `r_xz`). These are **imaged as ODD
   (anti-symmetric) functions about the surface** — i.e. the value at the mirror
   node one (or more) rows above equals the negative of the value the corresponding
   number of rows below. This guarantees they pass through zero at the surface.
2. **`sigma_xx`/`sigma_zz` updates near the surface use ONLY horizontal velocity
   derivatives** via eq. (5); the vertical velocity derivative `dvz/dz` is *not*
   taken across the surface. So `sigma_xx`/`sigma_zz` on the surface row are updated
   with horizontal central differences of `vx` only (`sigma_xx`,`r_xx` updated with
   4th-order horizontal stencil along the surface). This is what enforces the
   free-surface condition exactly in every grid cell along the boundary.
3. **Particle velocities above the surface are set to ZERO** (image method 3's
   defining choice). Robertsson found the velocity values above the surface are not
   critical for accuracy or stability, and zeroing them is "physically reasonable";
   it is also what falls out of setting the reciprocal density (buoyancy) to zero in
   the air region. (Levander/image-1 instead extrapolates them with eq. 5;
   Crase/image-2 mirrors displacement evenly. All three give similar accuracy.)
4. **Memory variables need NO imaging.** Because no spatial derivatives of the memory
   variables appear in the velocity/stress/memory update equations (paper eqs. 1–3),
   the memory variables above the surface are simply treated like the stress
   components they accompany and never require their own stencil across the surface.

Net effect for the flat case: a cheap mirror copy of `sigma_zz`/`sigma_xz`
(+`r_zz`/`r_xz`) into the rows above the surface, plus a one-sided (horizontal-only)
update rule for the normal stresses on the surface row, plus zeroing velocities above.

### 1.4 Topography generalisation and the NODE CATEGORIES
For staircase (grid-aligned) topography the surface is constrained to run *exactly*
along staggered-cell boundaries (the thick black line in Fig. 3). The grid points on
the discretised boundary are classified — **before time-stepping begins** — into
exactly **SEVEN categories** (memory note "~7" is confirmed as exactly 7):

1. **H**  — horizontal boundary point (air directly above; flat segment). Imaging
   only in the vertical direction; `sigma_zz`/`r_zz` imaged vertically; `sigma_xx`/`r_xx`
   updated with horizontal 4th-order central differences (this is identical to the
   flat image-method-3 rule).
2. **VL** — vertical boundary, air to the left. Imaging only in the horizontal
   direction; `sigma_xx`/`r_xx` imaged horizontally; `sigma_zz`/`r_zz` updated like the
   normal stresses of an H-point but in the vertical.
3. **IL** — inner corner, air above-left. `sigma_xx, sigma_zz, r_xx, r_zz` lie on the
   boundary and are set to zero; no imaging.
4. **OL** — outer corner, air to the left. Surface passes through `vx, vz, sigma_xz,
   r_xz`; normal stresses live in the air and are zeroed; `sigma_xz, r_xz` imaged both
   horizontally (about the right wall) and vertically (about the lower wall);
   `sigma_xx`/`sigma_zz` imaged appropriately. If adjacent to an IL-point the in-between
   velocity must be zeroed for stability.
5. **VR** — vertical boundary, air to the right (mirror of VL).
6. **IR** — inner corner, air above-right (mirror of IL). Only `sigma_xz` imaged.
7. **OR** — outer corner, air to the right (mirror of OL).

The update is sequenced as: (a) do all **vertical** imaging of stress components
(H, OR, OL, IR) then add vertical derivative contributions to the velocity update;
(b) then do all **horizontal** imaging (VL, VR, OR, OL) and add horizontal derivative
contributions; (c) zero `vx` at VR/OR points (reciprocal velocity nonzero there).
Splitting the velocity update into independent vertical-derivative and
horizontal-derivative passes is what lets multi-directional corner imaging be done
consistently and is the key trick that keeps it stable.

Because the boundary is forced to lie exactly on cell boundaries, "viewed from the
perspective of the individual staggered components the free surface is completely
symmetric" and the imaging reduces to a "simple and computationally inexpensive
mirroring scheme" aligned with the x/z axes. Topography limits: a crest must be
≥ ~4 grid points wide (one OL, two H, one OR), a trough ≥ 3 (one IR, three H, one IL);
overhangs are explicitly excluded.

### 1.5 Accuracy / stability / sampling characteristics
- Benchmarked against Sherwood (1958) analytic semi-infinite-medium solution.
  Image method 3 matches the analytic solution well and is **markedly more accurate
  than the "direct" (apply eq. 5 pointwise) method**, which is only 1st/2nd-order at
  the surface and mis-models the direct wave.
- **Stable and accurate for Poisson's ratios from 0 to 0.5** (incl. the extreme
  Poisson-solid case, vp/vs→∞), which is the high-Poisson robustness the direct
  method lacks.
- **Stability criterion is NOT affected by topography** (a major selling point vs.
  grid-deformation methods).
- Sampling: ~5 ppw (shortest wavelength) is adequate for the O(2,4) interior on a
  flat surface; for surfaces *with topography* ≥ ~15 grid points per shortest
  wavelength are recommended, because the staircase representation of a smooth slope
  scatters Rayleigh waves (the staircase, not the FS condition itself, is the
  accuracy limiter; ±45° staircases are actually more accurate than shallow slopes
  because the steps fall on individual cells).
- Designed for and demonstrated at O(2,4); the surface-row normal-stress update is
  done with a 4th-order *horizontal* stencil. Higher orders need wider one-sided /
  imaged stencils near the surface.

### 1.6 Extra material arrays? — NO (this is the headline contrast with PM)
The method uses **standard memory only — no extra per-grid material/stiffness
arrays**. Its data needs are (i) a per-boundary-point integer **category label**
(one of the 7), computed once before time-stepping, and (ii) the existing
velocity/stress/memory arrays. The mirroring and one-sided updates are pure index
arithmetic on the existing fields. Robertsson explicitly contrasts this with the
Muir et al. (1992) effective-medium / parameter-averaging approach, which "requires
finite material properties on both sides of the interface" — i.e. the property-based
philosophy that PM follows. Robertsson is a **kernel-logic** method, not a
**property-array** method.

---

## PART 2 — DEEPWAVE IMPLEMENTATION-COST ASSESSMENT

### 2.1 How Deepwave realises the free surface TODAY (no in-kernel imaging)
Deepwave currently uses the **Improved Vacuum Method (IVM, Zeng et al. 2012)** and
does the free surface **entirely as a material-property preprocessing step** — there
is zero free-surface logic in any time-stepping kernel:
- `elastic.py:prepare_parameters` (lines ~147–247) builds the staggered material
  arrays. Vacuum cells carry `lamb = mu = 0` and `buoyancy = 0` (a vacuum
  *sentinel*, not literal 1/rho — see the docstring at lines ~529–535). Interface
  coefficients are formed by **harmonic mean of mu** (so a vacuum neighbour zeroes the
  edge `mu_yx`/`mu_zx`/`mu_zy`) and **arithmetic-mean density → buoyancy** (so the
  air-side buoyancy is zero). Setting buoyancy = 0 above the surface makes the
  velocity update there a no-op, automatically reproducing image-method-3's
  "velocities are zero above the surface" — but as a *property* effect, not a branch.
- The kernels (`elastic.c`, `elastic.cu`, and `elastic_python`) are **completely
  free-surface-agnostic**: they just march the velocity–stress–memory equations over
  the staggered grid; the zeros in the property arrays do all the work.
- Source/receiver Hicks imaging already interacts with surfaces via a separate
  `rigid_surfaces` mechanism (per the boundary-interpolation memory note); it is
  orthogonal to how the propagator enforces the surface and would be unaffected by a
  switch of FS method.

### 2.2 Kernel structure that Robertsson would have to invade
- **C (`elastic.c`, 2254 lines)** and **CUDA (`elastic.cu`, 3058 lines)**: the body
  is generated by recompiling the *same* source with macros `DW_NDIM` (1/2/3),
  `DW_ACCURACY` (2/4/6/8), `DW_DTYPE` (float/double), `DW_DEVICE`. Each wavefield
  component (`vz, vy, vx, sigmazz, sigmayy, sigmaxx, sigmaxz, sigmayz, sigmaxy`) is
  updated in its **own** `#pragma omp simd` triple-nested loop, and each loop is run
  over a **3×3×3 (3×3 in 2-D) PML region decomposition** (`pml_bounds_*` arrays,
  `for (pml_z…) for (pml_y…) for (pml_x…)`, with `pml==1` = interior, `pml==0/2` =
  PML sides; see lines ~614–626 and the per-component loops ~849–1191 forward,
  ~1771–2123 backward). Robertsson surface imaging would need either new surface
  "regions" carved out of these bounds or extra `if (z == surface_row …)` branches in
  the hottest inner loops, **for both the forward and the adjoint/backward kernel**.
- **Eager (`elastic.py`)**: `update_velocities` (~1524–1606) and `update_stresses`
  (~1609–1711) are written with whole-array `diff1`/`diff1h` staggered-derivative ops
  (vectorised, no explicit grid loops). Robertsson here means inserting
  surface-row slicing + sign-flipped mirror writes + a separate horizontal-only
  normal-stress update on the surface row — and a category mask if topography is to be
  honoured. It does not compose naturally with the clean "one whole-array stencil"
  style; it would add a special-cased boundary block (and its adjoint, since autograd
  runs through these ops, so the manual `swap`/`backward` paths must stay consistent).

### 2.3 (a) How many backends/paths and how invasive
Three backends × two directions (forward + hand-written adjoint) × 2-D and 3-D.
- The 3-D node-category set is *not in the paper* — it must be derived. 2-D has 7
  categories; 3-D grows combinatorially (faces, 12 edges, 8 corner types → on the
  order of ~20+ categories), each with its own multi-directional imaging + which
  velocity components to zero. This derivation has to be done once mathematically,
  then **re-expressed in three different kernel languages** (C SIMD loops, CUDA
  threads, vectorised PyTorch) and **mirrored in each adjoint kernel**.
- In C/CUDA the imaging touches the innermost SIMD/threaded loops (branch divergence,
  loss of `#pragma omp simd`/coalescing on surface rows; ghost rows above the surface
  must be allocated/managed and kept consistent with `fd_pad`). Practically this is a
  large, error-prone addition to the two most complex files in the package.

### 2.4 (b) Composition with PML region decomposition and the FS∩PML corner
Poor. The kernels are already partitioned into 27 (or 9) PML regions per component.
A free surface that can lie at an arbitrary depth row — and topography that wanders in
z — cuts *across* those PML bounds, so the surface rows intersect PML side/corner
regions. The free-surface∩PML corner (a known stability hot-spot — see the
boundary+PML-stability memory note and the M-PML work) would now also carry imaging
logic, multiplying the special cases (surface × {interior, PML-side, PML-corner}).
The current IVM property approach sidesteps this entirely: vacuum cells are just data,
so the existing PML regions need no awareness of the surface.

### 2.5 (c) Conflict with the "ONE mechanism across all 4 (+future) propagators / 3 backends" goal
This is the decisive point. Robertsson imaging is **inherently per-equation, per-grid,
per-backend kernel logic**. It must be **re-derived and re-implemented for every
propagator**: scalar (acoustic pressure FS is the trivial pressure-release sub-case,
fine), but acoustic (velocity–pressure), elastic, scalar_born, and any future
anisotropic / attenuation propagator each have *different field sets and staggering*,
so each needs its own imaging rules, in each of the three backends, in both
directions. A property-array method (IVM today, or Cao–Chen PM) is the opposite: the
free surface is encoded once in the *parameter preparation* (pure PyTorch,
backend-independent), and **every propagator and backend consumes the same already-zeroed
/ already-modified arrays with no kernel changes**. That is exactly the "one
consistent mechanism" the project wants. Robertsson cannot provide it: it has to be
re-derived N_propagators × N_backends × {fwd,adj} × {2-D,3-D} times.

### 2.6 (d) Does staircase topography keep it grid-aligned, or force extra cases?
Staircase keeps the *imaging* grid-aligned (only x/z mirroring, no interpolation) —
which is the good news the paper emphasises — **but staircase is exactly what forces
the 7 (2-D) / ~20+ (3-D) node categories** (the H/VL/IL/OL/VR/IR/OR taxonomy exists
*because* of staircase corners). So "staircase is sufficient" does not shrink
Robertsson's cost; it is the source of its bookkeeping. A pure vacuum/property method
also realises staircase topography — but with zero categories and zero kernel cases,
just zeros placed in the property arrays.

---

## VERDICT / RECOMMENDATION

**What Robertsson buys:** the classic, expert-expected, well-benchmarked staggered-grid
free surface; demonstrably accurate against Sherwood's analytic solution; robust for
Poisson ratios 0–0.5 incl. the Poisson-solid extreme; topography-independent stability
criterion; and — notably — **no extra per-grid material arrays** (its only auxiliary
data is a precomputed integer category label per boundary point).

**What it costs in Deepwave:** it is fundamentally a *kernel-logic* boundary condition.
It must be hand-derived and hand-written as branchy, ghost-row-managing, SIMD/coalescing-
breaking surface code in **all three backends, in both the forward and adjoint kernels,
for 2-D and separately for 3-D (whose category set the paper does not even give), and
separately for each propagator** (acoustic, elastic, scalar_born, future
anisotropic/attenuation). It composes badly with the existing 27-region PML
decomposition and reintroduces logic into the already-delicate free-surface∩PML corner.

**Recommendation: do NOT adopt Robertsson as the carried free-surface mechanism.**
Its central virtue (no extra material arrays) does not outweigh the fact that it
violates the project's hard constraint of one mechanism reused across all
propagators and all three backends — it would have to be re-derived and re-coded a
dozen-plus times and maintained in lock-step. A **property/parameter-array method (the
current IVM, or Cao–Chen PM) is strictly better aligned with the architecture**: the
free surface is expressed once in backend-independent `prepare_parameters`-style
PyTorch, and every propagator/backend/direction consumes the modified arrays with no
kernel changes and no PML-corner special-casing. Given "staircase topography is
sufficient" (which Robertsson does not exploit to reduce its own cost), the decision
should be made **between the two property-based methods (IVM vs PM)** on accuracy/
high-Poisson grounds — keep Robertsson only as a reference/benchmark, not as the
implemented mechanism. If maximum near-surface accuracy ever becomes the dominant
requirement for the elastic propagator *specifically*, Robertsson is the gold standard
to target *there alone* — but it should not be the package-wide standard.

---

Summary file path: `/root/deepwave/ROBERTSSON_FS_SUMMARY.md`
