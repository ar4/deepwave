# Cao & Chen (2018) — Parameter-Modified (PM) Free-Surface Method

**Full citation:** Jian Cao and Jing-Bo Chen, 2018, "A parameter-modified method for
implementing surface topography in elastic-wave finite-difference modeling,"
*Geophysics*, **83**(6), T313–T332, doi:10.1190/geo2018-0098.1.

**Purpose of this document:** self-contained, implementation-ready description of the PM
free-surface/topography method for a standard velocity–stress staggered-grid (SSG)
elastic FD code, so it can be implemented without re-reading the PDF. Equations are
transcribed in plain ASCII/LaTeX-ish notation. Page/eq/table/figure references point
into the paper.

---

## 0. One-paragraph headline

The PM method implements an arbitrary (staircase-discretised) free surface **purely by
redefining the medium-parameter arrays once, before time stepping** — the standard
interior SSG velocity and stress update kernels are used unchanged at every step. It
generalises the "adaptive free-surface expression" of Cao et al. (2018) (a Moczo-style
average-medium / improved-vacuum idea) so the stress-free condition is satisfied
adaptively for **any Poisson's ratio**. Above topography the reciprocal density
(buoyancy) and Lamé parameters are set to zero (vacuum); at the discrete free-surface
points the **constitutive coefficients of the normal-stress update are modified
per-component** (this is the crux — see §2); shear moduli and densities at the surface
are halved/averaged. Reported accuracy at 15 grid points per minimum (shear) wavelength
is ~99% (2D) / ~98% (3D) vs SPECFEM, dramatically better than the improved-vacuum and
stress-image methods, especially at large Poisson ratio.

---

## 1. Grid convention assumed (§ "The generalized adaptive free-surface expression", p. T316; Figure 1)

Standard first-order velocity–stress staggered grid (the **second** of Kristek et al.
(2002)'s two configurations — normal-stress nodes lie ON the free surface; shear-stress
nodes are embedded in the medium). Spatial grid spacing `h`.

Node placement (3D, integer index `(i,j,k)`):
- **Integer grid points `(i,j,k)`**: normal stresses `τxx, τyy, τzz` AND medium
  parameters `λ, μ, ρ`.
- **Half-grid points**: shear stresses `τxy, τyz, τzx` and particle velocities
  `vx, vy, vz` (each on its appropriate half-offset node).

Free-surface plane placement: the free surface is chosen to pass **through the
normal-stress (integer) nodes**, with shear-stress components embedded in the medium.
Consequence (quoted, p. T316): "this grid configuration … the half-grid deviation from
the original free-surface position will not occur in the actual computation." (Contrast:
the improved-vacuum configuration shifts the free surface half a grid above its true
position — see §5.)

Staggered receiver/source note (p. T326): because of staggering, recorded `vx` and `vz`
are shifted half a grid in x and z respectively; the paper does NOT correct this position
inconsistency for PM (unlike IV, which averages velocity across the fictitious
free-surface layer). For receivers exactly at the surface they shift the receiver one
grid toward the solid to avoid sitting in the vacuum.

### Strain–stress / constitutive (continuous reference), eqs (3),(2)

Strains: `εxx=∂ux/∂x`, `εyy=∂uy/∂y`, `εzz=∂uz/∂z`,
`εxy=½(∂ux/∂y+∂uy/∂x)`, `εyz=½(∂uy/∂z+∂uz/∂y)`, `εzx=½(∂uz/∂x+∂ux/∂z)`.

Interior constitutive: `τxx=(λ+2μ)εxx+λεyy+λεzz`, etc.; `τxy=2μεxy`, etc.

---

## 2. THE PM SCHEME — exact discrete update and the per-component modified parameters

### 2.1 Discrete SSG update (the only kernel; eqs 20, 21)

Velocity update (`D-t` = time derivative; `D+x` etc. forward/backward spatial diff
operators):
```
D-t vx = b̃x ( D+x τxx + D+y τxy + D+z τxz )
D-t vy = b̃y ( D+x τxy + D+y τyy + D+z τyz )
D-t vz = b̃z ( D+x τzx + D+y τzy + D+z τzz )
```

Stress update (eq 21):
```
D+t τxx = η1^xx · D-x vx + η2^xx · D-y vy + η3^xx · D-z vz
D+t τyy = η1^yy · D-x vx + η2^yy · D-y vy + η3^yy · D-z vz
D+t τzz = η1^zz · D-x vx + η2^zz · D-y vy + η3^zz · D-z vz
D+t τxy = ⟨μ̃⟩xy ( D+y vx + D+x vy )
D+t τyz = ⟨μ̃⟩yz ( D+z vy + D+y vz )
D+t τzx = ⟨μ̃⟩zx ( D+z vx + D+x vz )
```

The **redefined medium parameters** are:
- reciprocal densities (buoyancies): `b̃x, b̃y, b̃z`
- **nine normal-stress stiffness coefficients**:
  `η1^xx, η2^xx, η3^xx` (for the τxx update),
  `η1^yy, η2^yy, η3^yy` (τyy),
  `η1^zz, η2^zz, η3^zz` (τzz)
- three shear moduli: `⟨μ̃⟩xy, ⟨μ̃⟩yz, ⟨μ̃⟩zx`

In 2D (x–z) this collapses to: `τxx` and `τzz` updates with coefficients
`η1^xx, η3^xx, η1^zz, η3^zz`, shear `⟨μ̃⟩zx`, buoyancies `b̃x, b̃z`.
```
D+t τxx = η1^xx · D-x vx + η3^xx · D-z vz
D+t τzz = η1^zz · D-x vx + η3^zz · D-z vz
D+t τxz = ⟨μ̃⟩zx ( D+z vx + D+x vz )
```

### 2.2 Base parameter averaging (off the free surface; eqs 4, 5, 22)

Densities arithmetic-averaged to velocity nodes (eq 4):
```
ρx=(ρ_{i,j,k}+ρ_{i+1,j,k})/2 ,  ρy=(ρ_{i,j,k}+ρ_{i,j+1,k})/2 ,  ρz=(ρ_{i,j,k}+ρ_{i,j,k+1})/2
```
Implemented equivalently as harmonic averaging of buoyancy `b=1/ρ` (eq 22):
```
bx = 2 b_{i,j,k} b_{i+1,j,k} / (b_{i,j,k}+b_{i+1,j,k})   (and y, z analogues)
```
μ harmonic-averaged to shear nodes (eq 5):
```
⟨μ⟩xy = ( ¼ ( 1/μ_{i,j,k}+1/μ_{i+1,j,k}+1/μ_{i,j+1,k}+1/μ_{i+1,j+1,k} ) )^-1
⟨μ⟩yz = ( ¼ ( 1/μ_{i,j,k}+1/μ_{i,j+1,k}+1/μ_{i,j,k+1}+1/μ_{i,j+1,k+1} ) )^-1
⟨μ⟩zx = ( ¼ ( 1/μ_{i,j,k}+1/μ_{i+1,j,k}+1/μ_{i,j,k+1}+1/μ_{i+1,j,k+1} ) )^-1
```

### 2.3 Parameter settings by location (eqs 23, 24, 25) — the actual PM rule

Define the two scalar combinations used throughout the free-surface expressions:
```
α = 2μ(λ+μ)/(λ+2μ)     (the "modified P-modulus")
β = μλ/(λ+2μ)          (the modified cross / off-diagonal coupling)
```
(Note `α + β ≠ λ+2μ`. Derivation: imposing the stress-free condition on the interior
constitutive relation reduces the diagonal normal stiffness to
`(λ+2μ) − λ²/(λ+2μ) = 4μ(λ+μ)/(λ+2μ) = 2α` and the off-diagonal one to
`λ − λ²/(λ+2μ) = 2λμ/(λ+2μ) = 2β`; the surface node then takes the arithmetic
average of this BC-reduced solid-side stiffness with the vacuum's zero, giving the
factor ½ that produces `α` and `β`. The same ½ vacuum average is what halves the
surface densities and the surface-plane shear moduli. Symbolically re-derived and
verified entry-by-entry in `verify_topo_tables.py`.)

**(a) Interior points (eq 23) — equivalent to the standard scheme:**
```
b̃x=bx , b̃y=by , b̃z=bz
η1^xx = η2^yy = η3^zz = λ+2μ          (the three "diagonal" coefficients)
η2^xx = η3^xx = η1^yy = η3^yy = η1^zz = η2^zz = λ   (the six "off-diagonal" coefficients)
⟨μ̃⟩xy=⟨μ⟩xy , ⟨μ̃⟩yz=⟨μ⟩yz , ⟨μ̃⟩zx=⟨μ⟩zx
```
So in the interior the nine `η` reduce to just `λ+2μ` (diagonal) and `λ`
(off-diagonal) — i.e. recoverable from a single `(λ, μ)` pair.

**(b) Above the topography (vacuum):** set the reciprocal density (buoyancy) AND the Lamé
parameters to zero everywhere above the surface. Then `b̃ = 0` and all `η = 0`, so
velocities and stresses there go to zero every step automatically (Robertsson 1996;
Hayashi et al. 2001 idea). The paper sets the *reciprocal* density to zero rather than
the density itself.

**(c) At the discrete free-surface points — per boundary type.** The modification
realises Table 2 (3D) / Table 1 (2D). Worked example given in the paper for the **VR
boundary** (air to the right; eq 24):
```
b̃x=bx , b̃y=2by , b̃z=2bz
η2^yy = η3^zz = α = 2μ(λ+μ)/(λ+2μ)
η3^yy = η2^zz = β = μλ/(λ+2μ)
η1^xx = η2^xx = η3^xx = η1^yy = η1^zz = 0
⟨μ̃⟩xy=⟨μ⟩xy , ⟨μ̃⟩yz=⟨μ⟩yz/2 , ⟨μ̃⟩zx=⟨μ⟩zx
```
(Comment from the paper, eq 24: you need NOT explicitly zero `b̃x, τxx, τzz` because for
cells with air to the right/back the harmonic averaging of `τxz` and `ρx` in eqs 22/5
zeroes them automatically — "it is unnecessary to distinguish the direction of the air
relative to the free surface" for those.)

**(d) Transition / outer-corner points (eq 25):**
```
b̃x=2bx , b̃y=2by , b̃z=2bz
η1^xx=η2^xx=η3^xx = 0
η1^yy=η2^yy=η3^yy = 0
η1^zz=η2^zz=η3^zz = 0
⟨μ̃⟩xy=⟨μ⟩xy/2 , ⟨μ̃⟩yz=⟨μ⟩yz/2 , ⟨μ̃⟩zx=⟨μ⟩zx/2
```
(Derivation status of the all-zero normal stiffness — see
`verify_topo_tables.py`: in 2D, imposing the traction-free
conditions of both intersecting surface planes forces `εxx = εzz = εzx = 0`, so the
all-zero rule is **exact**; likewise at a 3D corner where three faces meet. Along a 3D
**edge** (two faces), the edge-parallel normal stress physically retains a uniaxial
term `τyy = E εyy` with `E = μ(3λ+2μ)/(λ+μ)` (Young's modulus) that eq 25 also zeroes
— an approximation in the paper's rule, confined to the 1-node-wide edge lines.)

### 2.4 The underlying modified constitutive matrices (where α, β come from)

The per-type modified constitutive relations that the `η` arrays encode (3D, eqs 16–19;
2D eqs 6, 12; collected in Tables 1–2). Here normal direction `n` points from solid into
air.

**2D (x–z), Table 1** (with `2μ(λ+μ)/(λ+2μ)=α`):
| Type | τxx | τzz | τxz | ρx | ρz |
|---|---|---|---|---|---|
| H (horizontal free surface, n=(0,−1)) | `α·εxx` | `0` | `2μ εxz` | `0.5 ρx` | `ρz` |
| VL (air to left, n=(−1,0)) | `0` | `α·εzz` | `2μ εxz` | `ρx` | `0.5 ρz` |
| VR (air to right, n=(1,0)) | `0` | `α·εzz` | `0` | `0` | `0.5 ρz` |

The H row (eq 6) in matrix form:
```
[τxx]   [α 0 0][εxx]
[τzz] = [0 0 0][εzz]
[τzx]   [0 0 0][εzx]   →  τxx = α εxx , τzz = 0 , τzx = 0
```
The VL row (eq 12, derived by a 90° coordinate rotation of the H expression):
```
[τzz ]   [α 0 0][ εzz]
[τxx ] = [0 0 0][ εxx]   →  τzz = α εzz , τxx = 0 , τzx = 0
[−τzx]   [0 0 0][−εzx]
```

**3D, Table 2** (with `α=2μ(λ+μ)/(λ+2μ)`, `β=μλ/(λ+2μ)`):
| Type | τxx | τyy | τzz | τxy | τyz | τzx | ρx | ρy | ρz |
|---|---|---|---|---|---|---|---|---|---|
| H  | `α εxx + β εyy` | `β εxx + α εyy` | `0` | `μ εxy` | `2μ εyz` | `2μ εzx` | `0.5ρx` | `0.5ρy` | `ρz` |
| VL | `0` | `β εzz + α εyy` | `α εzz + β εyy` | `2μ εxy` | `μ εyz` | `2μ εzx` | `ρx` | `0.5ρy` | `0.5ρz` |
| VR | `0` | `β εzz + α εyy` | `α εzz + β εyy` | `0` | `μ εyz` | `0` | `0` | `0.5ρy` | `0.5ρz` |
| VF | `α εxx + β εzz` | `0` | `β εxx + α εzz` | `2μ εxy` | `2μ εyz` | `μ εzx` | `0.5ρx` | `ρy` | `0.5ρz` |
| VB | `α εxx + β εzz` | `0` | `β εxx + α εzz` | `0` | `μ εyz` | `μ εzx` | `0.5ρx` | `0` | `0.5ρz` |

3D VL boundary in matrix form (eq 16):
```
[τzz]   [α  β  0][εzz]
[τyy] = [β  α  0][εyy]
[τyz]   [0  0  μ][εyz]
```
(VF/VB are the y-axis-rotated analogues, eqs 18–19.) The free-surface expressions are
derived by writing the flat-surface modified relation (eqs 2, 6) in a rotated coordinate
system aligned with each vertical/oblique boundary (rotation matrices eqs 7–10 in 2D, eqs
13–14 in 3D); **the rotations are an offline derivation device only** — no rotation
appears in the running kernel.

Reading Table 2 (established by the symbolic re-derivation in
`verify_topo_tables.py`, which re-derives every cell of Tables 1–2
and eqs 23–25 from the traction-free conditions + the ½ vacuum average + the staggered
node placement):
- Every shear-stress node lying **in** the discrete surface plane carries the halved
  modulus (`⟨μ̃⟩ = ⟨μ⟩/2`, i.e. `μ ε..` in table notation): `τxy` for H, `τyz` for
  VL/VR, `τzx` for VF/VB. This is required by the rotation consistency with eq 24
  (`⟨μ̃⟩yz = ⟨μ⟩yz/2` for VR) and by momentum-balance consistency with the halved
  surface normal stresses and densities. (An earlier transcription of this table had
  `2μ εxy` in the H row — wrong; the halved `μ εxy` is the derivable value.)
- Shear nodes offset **toward the vacuum** are effectively zero because the base
  harmonic averaging (eq 5) with a zero-μ vacuum neighbour vanishes — the table writes
  some of these as `0` (VR `τxy`/`τzx`, VB `τxy`) and one in formal halved notation
  (VB `τyz = μ εyz`, whose base harmonic average is 0, so it is effectively 0 too).
- Shear nodes offset **into the solid** keep the full interior `2μ ε..`.
- The same trichotomy gives the density column: in-plane node → `0.5ρ`, vacuum-side
  node → `0` (harmonic buoyancy average), solid-side node → `ρ`.

Three positional categories the implementation must classify (p. T318, list 1–3):
1. **Above** the free surface → vacuum (stresses, density set to give zero).
2. **At** the free surface → use the modified constitutive + density (Table 1/2).
3. **Below** the free surface → interior points, original relation unchanged.

---

## 2′. ANSWER TO THE CRITICAL COST QUESTION — does PM need a separate per-dimension stiffness model?

**YES.** PM requires the normal-stress update to carry **independent per-component
stiffness coefficients** that cannot all be reconstructed from a single shared `(λ, μ)`
at/near the free surface.

Evidence (eqs 21, 23, 24):
- In the interior (eq 23) the nine coefficients collapse to two values:
  diagonal `η1^xx=η2^yy=η3^zz=λ+2μ`, off-diagonal everything-else`=λ`. A single `(λ, μ)`
  serves the τxx, τyy, τzz updates — this is exactly what a conventional code stores.
- At a free-surface point (e.g. VR, eq 24) the diagonal coefficients become
  `η2^yy=η3^zz=α=2μ(λ+μ)/(λ+2μ)` (NOT `λ+2μ`), the off-diagonals become
  `η3^yy=η2^zz=β=μλ/(λ+2μ)` (NOT `λ`), and several others are forced to **0**. Crucially
  the value multiplying `D-x vx` in the τxx update (`η1^xx`) is set to 0 while the value
  multiplying `D-y vy` in the τyy update (`η2^yy`) is `α` — i.e. the coefficient acting on
  the *same* spatial derivative now differs between normal-stress components, and within
  one component the diagonal/off-diagonal pair is no longer `(λ+2μ, λ)`.

Therefore the constitutive coefficients used in the τxx vs τyy vs τzz updates are
**genuinely distinct fields**. To implement PM in a code that currently stores one
`(λ, μ)` (or `λ+2μ` and `λ`) shared across all normal-stress updates, you must add
**separate coefficient arrays**:
- **3D:** 9 normal-stiffness arrays `η{1,2,3}^{xx,yy,zz}` (in practice 3 per normal-stress
  component) + 3 shear arrays `⟨μ̃⟩{xy,yz,zx}` + 3 buoyancy arrays `b̃{x,y,z}`.
- **2D:** 4 normal-stiffness arrays `η1^xx, η3^xx, η1^zz, η3^zz` + 1 shear `⟨μ̃⟩zx` + 2
  buoyancy `b̃x, b̃z`.

(Caveat for cost estimation: away from the surface the arrays are redundant — they equal
`λ+2μ`/`λ`. A memory-lean implementation could store full per-component arrays only in a
thin band near topography and fall back to scalar `(λ, μ)` in the interior. But the
*update kernel* itself, to be uniform, multiplies separate coefficients per component.)

---

## 3. Difference from the Improved-Vacuum method (Zeng et al. 2012) and extra arrays

| Aspect | Improved-Vacuum (Zeng 2012, IV) | PM (Cao–Chen 2018) |
|---|---|---|
| What is modified | Reciprocal density (buoyancy) only — zeroed above surface; density halving | Buoyancy zeroed/halved **AND** per-component constitutive (`η`) coefficients modified at the surface (`α, β`) |
| Constitutive relation at surface | Unchanged interior `(λ, μ)` | Replaced by adaptive average-medium relation (Table 1/2) |
| Free-surface position | Shifted **half a grid above** true position (grid config 1) | Stays exactly at the discrete topography (grid config 2) — no shift |
| Poisson-ratio robustness | Degrades badly at high σ | Adaptive for any σ |
| Extra arrays needed vs single (λ,μ) | none (just a buoyancy/vacuum mask) | **per-component stiffness arrays** `η`, plus modified shear `⟨μ̃⟩` and buoyancy `b̃` at the surface |
| Receiver position fix | averages velocity across fictitious free-surface layer | none applied (no fictitious layer exists) |

Quote (p. T326): "The improved vacuum formulation puts the free surface half a grid above
its original position, and the generalized stress-image method introduces the symmetry
planes at the transition points for stress imaging." PM avoids both.

---

## 4. Topography handling — staircase via material arrays only

**Confirmed: PM handles arbitrary topography with a staircase discretisation using ONLY a
one-time modification of the medium-parameter arrays — no coordinate rotation in the
kernel, no per-timestep wavefield imaging.**

Key quotes:
- (p. T323) "this PM method performs the FD modeling with arbitrary irregular topography
  only through a modification of redefined medium parameters before the wave simulation,
  avoiding the handling of wavefield variables on the boundary in every time step."
- (Abstract) "the stress-free condition is implemented on the rugged surface by modifying
  the redefined medium parameters at the discrete topography boundary points. This
  numerical treatment for topography needs to be performed only once before the wave
  simulation."
- 3-step integrated algorithm (p. T323): (1) set buoyancy `b` and Lamé `(λ, μ)` to zero
  everywhere above the topography; (2) classify the grid points at the surface
  (H / VL / VR / VF / VB / transition / outer-corner) and compute their redefined medium
  parameters from the generalised free-surface expressions; (3) run the standard SSG FD
  (eqs 20–21) with these arrays — interior kernel essentially unchanged.
- Transition cells (inner-corner / outer-corner) connect adjacent H and vertical
  boundaries to keep the discrete surface continuous (Figure 4c, eqs 25). Unlike the
  generalised stress-image method, PM does **not** move the free surface inside OL/OR
  cells, so "the position of the discrete free surface is unchanged during the
  computation" — fewer artificial staircases/diffractions.

Coordinate rotations (eqs 7–14) are used **only to derive** the vertical/oblique boundary
expressions offline; they are not applied during time stepping.

---

## 5. Accuracy results (Table 4 L2 misfit norms; Figs 11–12)

Benchmark: SPECFEM2D / SPECFEM3D spectral-element reference. Two 2D homogeneous half-space
models (Fig 5): trough-like with **σ = 0.25** (Vp=4500, Vs=2598 m/s) and crest-like with
**σ = 0.45** (Vp=4500, Vs=1357 m/s); ρ=1500. 15 Hz Ricker, fmax 40 Hz. Methods compared:
IV = improved vacuum, Image = generalised stress-image (Robertsson 1996), PM2D.

**Headline accuracy:** PM achieves **~99% average accuracy in 2D and ~98% in 3D** at the
fourth-order scheme with `λmin/h = 15`.

Selected L2 misfit-norm numbers from **Table 4** (percent; lower = better):

Trough-like model, σ = 0.25:
- R2, PPW 30, 4th order, vx: **IV 17.0 | Image 10.2 | PM 0.4**;  vz: IV 17.9 | Image 10.2 | PM 0.1
- R2, PPW 30, 2nd order, vx: IV 17.1 | Image 20.1 | PM 0.4
- R2, PPW 15, 4th order, vz: **IV 59.5 | Image 49.1 | PM 0.5**;  vx: IV 49.9 | Image 41.8 | PM 0.4
- R2, PPW 15, 2nd order, vx: IV 55.4 | Image 60.9 | PM 0.4
- R3, PPW 15, 2nd order, vx: **IV 123.3 | Image 129.4 | PM 7.9**;  vz: IV 159.2 | Image 171.7 | PM 2.4
- R1, PPW 30, 4th order, vx: IV 0.7 | Image 1.0 | PM 1.4

Crest-like model, σ = 0.45:
- R2, PPW 30, 4th order, vx: IV 15.7 | Image 9.3 | PM 0.2;  vz: IV 5.6 | Image 2.2 | PM 0.04
- R2, PPW 15, 4th order, vx: IV 55.0 | Image 57.7 | PM 0.3;  vz: IV 22.1 | Image 17.3 | PM 0.2
- R3, PPW 15, 2nd order, vx: IV 43.7 | Image 53.0 | PM 0.7;  vz: IV 49.2 | Image 94.5 | PM 1.8

Interpretation matching the cost-analysis brief: the IV method's misfit grows large and
strongly Poisson-dependent (tens of percent at PPW30, exceeding 50–130% at PPW15), while
PM stays at the sub-percent-to-few-percent level across the whole σ range and both orders.
(The brief's remembered "~17% at σ=0.25 rising to ~59% at σ=0.45 for IV" corresponds to
the R2 trough PPW30-4th IV vx = 17.0% and the R2 trough PPW15-4th IV vz = 59.5% entries;
the precise table values are above.)

3D Gaussian-hill homogeneous tests (Fig 12, σ=0.25 and σ=0.4): PM L2 misfits ~0.22–3.27%,
averaged accuracy ~98% vs SPECFEM3D.

Complex-model demonstrations: 2D Foothill model (Fig 13–14) and 3D heterogeneous basin
model (Fig 15–17) — feasible with strong topography and Poisson-ratio discontinuity.

---

## 6. Order of accuracy, PPW requirement, stability, limitations

**Accuracy order validated:** 2nd- and 4th-order spatial SSG (time integration is 2nd
order in all tests). PM is "not restricted by the order of the FD operators." 4th order
suppresses Rayleigh-wave numerical dispersion but does **not** reduce staircase
diffractions (those are controlled by sampling, not order).

**Points-per-wavelength requirement:** recommended sampling rule **`λmin/h = 15`** (15
grid points per **minimum shear** wavelength). They tested `λmin/h = 5, 10, 15, 30`:
λmin/h=5 shows heavy Rayleigh-wave staircase diffraction; at 15 "there are no significant
artificial diffractions," and 30 gives only marginal improvement — so 15 is chosen for
cost/accuracy balance. Body-wave propagation in the interior still only needs the usual
5 PPW (4th order) / more (2nd order); the 15-PPW demand is specifically to control
staircase diffraction at the topography.

**Stability:** ordinary Courant criterion suffices (eq 28):
```
Δt < Δt_lim = 1 / ( Vmax · sqrt( Σ_i |β_i| · (1/Δx² + 1/Δy² + 1/Δz²) ) )
```
(`β_i` = staggered-grid FD coefficients; `Vmax` = max P velocity.) Time step set as
`Δt = γ Δt_lim`, γ ∈ (0,1). Key claim (p. T329): "this PM method does not need a smaller
Courant number, and a time step satisfying the Courant criterion will ensure stability."
Long-run energy tests (Fig 18) remain stable over **20,000 steps even at γ = 99%**, for
both 2D and 3D and both σ values. Tests in the paper use γ = 95%. Rationale: the modified
parameters produce slower near-surface waves (Rayleigh/Love/converted), all below Vmax, so
they cannot violate the Courant bound.

**Stated limitations / caveats:**
- Requires the relatively dense `λmin/h ≈ 15` sampling → costly for large uniform-grid
  3D problems. Mitigation: a **discontinuous-grid strategy** (oversample only near
  topography, coarsen below; eqs 26–27, grid ratio 3, Fourier interpolation for stable
  coarse→fine upsampling, Table 5) — ~70% point-count reduction (2D Foothill), ~47% (3D
  basin).
- Staircase-shaped discretisation inherently produces artificial diffractions at corners;
  controlled (not eliminated) by sampling. Effect is noted to be **more severe in 3D**
  than 2D at the same λmin/h.
- Receiver/source half-grid position inconsistency is **not corrected** for PM (see §1);
  they shift surface receivers one grid into the solid. A wavenumber-domain (Fourier/phase
  shift) correction is possible but not implemented.

---

## 7. PML / absorbing-boundary interaction

- PML applied on all boundaries **except** the top free surface/topography; works with the
  standard SSG PML unchanged.
- **Topography must be extended into the PML region.** Quote (p. T327, ref Fig 13e):
  "extensions of the topography on the left and right edges are performed to include the
  PML." Practically: continue the free-surface profile (and hence the vacuum / surface
  parameter classification) horizontally across the PML padding so the PML sees a
  consistent surface.
- No special PM↔PML coupling formula is required beyond extending the topography arrays;
  the PM treatment is just part of the medium-parameter setup, which the PML region also
  uses.
- 3D basin example used 21 PML layers (p. T328).

---

## 8. Implementation checklist (distilled)

1. Choose grid: normal-stress + `(λ,μ,ρ)` on integer nodes; shear-stress + velocities on
   half nodes; free surface through integer normal-stress nodes.
2. Sample topography to staircase on integer nodes; extend across PML.
3. Precompute base averaged arrays: buoyancies `bx,by,bz` (harmonic, eq 22) and shear
   `⟨μ⟩xy,⟨μ⟩yz,⟨μ⟩zx` (harmonic, eq 5).
4. Allocate per-component coefficient arrays (3D: `η{1,2,3}^{xx,yy,zz}`, `⟨μ̃⟩{xy,yz,zx}`,
   `b̃{x,y,z}`; 2D: `η1^xx,η3^xx,η1^zz,η3^zz,⟨μ̃⟩zx,b̃x,b̃z`).
5. Fill: interior → eq 23; above surface → 0; at-surface points → classify
   H/VL/VR/VF/VB and use Table 1/2 expressions (α, β, halved ρ/μ); transition/outer-corner
   → eq 25. Do this **once**, before time stepping.
6. Time-step with the standard SSG kernel (eqs 20, 21) using these arrays. No per-step
   boundary special-casing.
7. Use `λmin/h ≈ 15` near topography (optionally discontinuous grid), 4th order spatial,
   Δt at ~95% Courant.

---

## 9. Verification record

- **Symbolic (all table entries):** `verify_topo_tables.py` re-derives every cell of
  Table 1 (2D), Table 2 (3D) and eqs 23–25 with sympy from only the stated conditions
  (isotropic constitutive law; traction-free elimination; ½ vacuum arithmetic average
  for nodes in a surface plane; staggered node placement with harmonic auto-zeroing on
  the vacuum side), and cross-checks the vertical-boundary rows against the rotated
  flat-H relation (the paper's own derivation device). All checks pass. The corrections
  it forced on this document's earlier text: the Table 2 H-row `τxy` cell (`μ εxy`, not
  `2μ εxy`), the reading of formally-halved-but-effectively-zero vacuum-side cells
  (e.g. VB `τyz`), the `2α` identity in §2.3, and the 3D-edge approximation note on
  eq 25 (§2.3d).
- **Numeric, flat (2D eager prototype):** `verify_topo.py` — the mask-classifier PM
  kernels (`pm_topo.py`) reproduce the stock scheme bit-for-bit with no free surface
  (rel ~6e-16) and reduce exactly (rel 0.0) to the separately validated flat-H PM
  kernels (`pm_kernels.py`) on a flat surface.
- **Numeric, tilted (2D eager prototype):** `verify_topo_tilted.py` — a
  staircase-discretised trough (25% grades, H/VL/VR/corner cells all exercised,
  σ=0.25, 15 Hz Ricker, receivers on both flanks) self-converges under dx=4→2→1 m
  refinement (17/33/67 points per minimum shear wavelength) at order ≥ the flat
  control's on identical machinery, with comparable fine-grid misfit — the staircase
  does not degrade convergence relative to a flat surface. Both cases share a
  convergence floor from the surface scheme's Rayleigh-wave dispersion; the script's
  `--floor-probe` mode shows the misfits are identical to 5 decimals under doubled PML
  thickness and 160 m extra solid depth, so the floor is not an absorbing-boundary or
  truncation artifact. This matches the paper's ≥15 points-per-minimum-shear-wavelength
  sampling guidance for surface waves.
- The prototype kernels (`pm_kernels.py`, `pm_topo.py`) and these three drivers track
  the current in-repo eager elastic backend signature (trailing optional `fd_coeffs`;
  `PropagatorOutput` return). The older stage/bench/convergence drivers in this
  directory predate that signature and would need the same two mechanical updates to
  rerun.
