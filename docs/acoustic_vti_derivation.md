# Derivation: pseudo-acoustic VTI (`acoustic_vti`) propagator

*Developer/reference documentation: the specification and discrete derivation
behind `deepwave/acoustic_vti` and its adjoint. See {doc}`acoustic_vti` for
the user-facing propagator page. Any implementation that disagrees with this
derivation should be treated as a discrepancy to resolve deliberately, not
silently patched in either direction.*

**Status: implemented.** Both the eager backend and the compiled C/CUDA
kernels (with hand-written adjoints) are landed. The implementation has been
audited against this document line by line, and the compiled gradients match
compiled `elastic_vti(c55=c66=0)` to ~1e-15 at
`model_gradient_sampling_interval` in {1, 2, 3}, 2-D and 3-D.

## 1. Continuous system

Duveneck et al. (2008), SEG expanded abstracts (10.1190/1.3059320):
first-order velocity–stress pseudo-acoustic VTI ("acoustic VTI" =
Alkhalifah's V_S = 0 limit of VTI elasticity), variable density. Fields:
particle velocities and TWO stress-like scalars,

- `sigma_h` — the horizontal normal stress (equal in all directions
  perpendicular to the symmetry axis),
- `sigma_v` — the normal stress along the symmetry axis.

With b = buoyancy = 1/rho, and the symmetry axis along the SLOWEST spatial
dimension (Deepwave convention, matching `elastic_vti`: axis = y in 2-D
(y, x) ordering, axis = z in 3-D (z, y, x) ordering — throughout this
document "z" denotes the axis dimension and "x, y" the transverse ones;
in 2-D drop y):

    d(sigma_h)/dt = c11 (dvx/dx + dvy/dy) + c13 dvz/dz + s_h(t)
    d(sigma_v)/dt = c13 (dvx/dx + dvy/dy) + c33 dvz/dz + s_v(t)
    d(vx)/dt      = b d(sigma_h)/dx
    d(vy)/dt      = b d(sigma_h)/dy
    d(vz)/dt      = b d(sigma_v)/dz

Parameters (Cij are density-normalised-free, i.e. true stiffnesses, exactly
as `elastic_vti` defines them):

    c33 = rho vp^2          (vp = P velocity along the axis)
    c11 = c33 (1 + 2 eps)
    c13 = c33 sqrt(1 + 2 delta)

`thomsen_to_cij(vp, vs=0, rho, eps, delta[, gamma])` already produces
exactly these: its general `c13 = sqrt(2 delta c33 (c33 - c55) +
(c33 - c55)^2) - c55` reduces at `c55 = 0` to `c33 sqrt(1 + 2 delta)`, and
`c55 = c66 = 0` follow from `vs = 0`. So the Thomsen path is available,
differentiably, with no new helper.

**Stability (Grechka/Duveneck condition).** The system is well-posed only
where `eps >= delta`, equivalently

    c11 c33 >= c13^2       (elementwise, AFTER any model-preparation
                            averaging onto grid nodes)

`acoustic_vti` must validate this and raise `ValueError` naming the
condition and its physics. Note
`eps = delta` (elliptical anisotropy) is allowed and is a useful test point
(no pseudo-shear artifact).

**Pseudo-shear source artifact.** A point source in a region where
`eps != delta` emits a diamond/rhombic pseudo-shear wave (the V_S = 0
kinematic remnant). This is intrinsic to the formulation, not a bug.
Mitigation (optional helper, task G): blend `c13 -> sqrt((c11)(c33))`
(i.e. delta -> eps, elliptical) inside a small radius of each source
location with a raised-cosine taper (Duveneck's mitigation; SWEEP
implements the same idea). Document that the taper perturbs the model near
the source.

## 2. Exact reduction map from `elastic_vti` (the oracle)

`acoustic_vti` IS `elastic_vti` with `c55 = c66 = 0`, with the two (3-D:
three) normal stresses collapsing onto (sigma_h, sigma_v). This section is
the load-bearing part of the document: it defines the correspondence used
by the implementation, the tests, and the adjoint.

In `elastic_vti` with `c66 = 0` we get `c12 = c11 - 2 c66 = c11`, so the
3-D normal-stress updates become

    sigmaxx += dt (c11 (dvxdx + dvydy) + c13 dvzdz)
    sigmayy += dt (c11 (dvxdx + dvydy) + c13 dvzdz)     [identical]
    sigmazz += dt (c13 (dvxdx + dvydy) + c33 dvzdz)

and all shear stresses, if initially zero and unsourced, remain identically
zero. Hence with equal initial data and matched sources,

    sigma_h(t) == sigmaxx(t) == sigmayy(t),   sigma_v(t) == sigmazz(t),
    vx, vy, vz identical in both systems.

(2-D, axis = y: sigma_h == sigmaxx, sigma_v == sigmayy, sigmaxy == 0.)

**The duplication map and its transpose.** Write the reduced state
`u = (v, sigma_h, sigma_v)` and the full state
`U = (v, sigmaxx, sigmayy, sigmazz, shear=0)`. The embedding is
`U = D u` where D duplicates sigma_h into (sigmaxx, sigmayy). Then:

- Forward states map by duplication: `sigma_h -> (sigmaxx, sigmayy)`.
- A SOURCE on sigma_h with amplitude a corresponds to sources on BOTH
  sigmaxx AND sigmayy, each with amplitude a (sources transform like the
  state they are added to).
- ADJOINT states map by the TRANSPOSE (summation):
  `s_h == s_xx + s_yy`, `s_v == s_zz` — the adjoint of the reduced sigma_h
  accumulates what the full adjoint spreads over the two horizontal
  stresses. **This is the factor-of-two hazard class for 3-D:** any
  place the implementation or a test equates `s_h` with `s_xx` alone (or
  averages instead of sums) will be silently wrong by a factor that the
  2-D case and the isotropic limit may not expose.
- A RECEIVER on sigma_h reads the same numbers as a full-system receiver
  on sigmaxx (or sigmayy — they are equal), BUT the gradient pullback of a
  sigma_h receiver injects its residual into s_h, i.e. into BOTH s_xx and
  s_yy of the full system. A full-system test that records only sigmaxx
  therefore has a DIFFERENT adjoint injection and will NOT reproduce
  reduced-system stress-receiver gradients. Consequence for the oracle
  gradient tests: use VELOCITY receivers (identical states, identical
  pullbacks) for strict parity; verify stress-receiver conventions
  separately against this map (full-system receivers on sigmaxx and
  sigmayy, residuals injected into both, summed loss).

**Parameter-gradient correspondence** (needed to compare gradients in the
oracle test; recall `elastic_vti`'s Python derives `c12 = c11 - 2 c66`, so
autograd folds grad_c12 into c11 and c66):

    grad_c11[acoustic_vti] == grad_c11[elastic_vti] + grad_c12[elastic_vti]
                              (autograd's folded c11 gradient at c66=0
                               already equals exactly this sum)
    grad_c13[acoustic_vti] == grad_c13[elastic_vti]
    grad_c33[acoustic_vti] == grad_c33[elastic_vti]
    grad_b  [acoustic_vti] == grad_b  [elastic_vti]
    grad_c55, grad_c66 in the full system: not required to vanish (the
    forward is stationary at c55=0 only through the shear stencil terms;
    verify what autograd returns and document) — the oracle asserts
    equality of the SHARED parameters only.

## 3. Discrete forward scheme

Inherit `elastic_vti`'s discretisation wholesale (Virieux staggering,
leapfrog, C-PML of Komatitsch & Martin 2007, IVM free surface of Zeng et
al. 2012, `accuracy` in {2, 4, 6, 8}, model preparation averaging):

- Staggering: velocities at half-integer nodes in their own dimension;
  sigma_h and sigma_v both live at the INTEGER (normal-stress) nodes; c11,
  c13, c33 at integer nodes (no shear stiffness exists, so the entire
  shear-node averaging path of elastic_vti is absent); buoyancy averaged
  onto each velocity's staggered node exactly as in `elastic`
  (arithmetic two-point IVM averaging — reuse the existing helper).
- Update order per time step exactly as elastic/elastic_vti (velocities
  from stresses, then stresses from velocities), with sources injected as
  elastic does (additive after the corresponding update; sigma_h/sigma_v
  and per-velocity-component source channels; an "explosive/pressure"
  source is a sigma_h AND sigma_v injection — provide it as documented
  convention, not magic: users pass the same wavelet to both channels).
- C-PML memory fields: only the 2*ndim surviving derivative terms need
  memories — ndim velocity-derivative memories (one per dvi/dxi feeding
  the stress updates) and ndim stress-derivative memories (d sigma_h/dx,
  (d sigma_h/dy,) d sigma_v/dz feeding the velocity updates). All shear
  memories are dropped. Profiles/half-node conventions identical to
  elastic.
- Free surface: inherit the IVM machinery unchanged. Physically the
  traction-free condition at the axis-normal surface is sigma_v = 0 (all
  shear already zero). Acceptance is empirical via the oracle: `elastic_vti
  (c55=c66=0)` with a free face must match. If IVM at c55=0 proves unstable
  or wrong in that comparison, RESTRICT (raise for free faces) and record —
  do not improvise a new surface scheme mid-task.
- CFL / step_ratio: the qP phase velocity is bounded by
  `vmax = sqrt(max(c11, c33) * max(b_unaveraged))` (transverse and axial
  extremes; the qP velocity surface for V_S=0 attains its maximum on an
  axis). Use that vmax in the same CFL machinery elastic uses. Note
  `sqrt(c11 b)` uses colocated values before staggering-averaging;
  a conservative elementwise max over the raw inputs is fine.

## 4. Discrete adjoint

The skeleton (time reversal, PML transpose, storage/decimation) is
inherited from elastic_vti verbatim; only the constitutive couplings
reduce. Deriving directly in the reduced variables (equivalently: apply
the §2 transpose map to elastic_vti.c's documented adjoint):

Forward constitutive step (interior, PML-corrected derivatives d_i):

    sigma_h += dt (c11 (dvx/dx + dvy/dy) + c13 dvz/dz)
    sigma_v += dt (c13 (dvx/dx + dvy/dy) + c33 dvz/dz)

Transpose — the adjoint velocities pick up (s_h, s_v = adjoint stresses;
these lines are the acoustic_vti analogue of elastic_vti.c's VX_X/VY_Y/VZ_Z
access-term macros):

    to dvx/dx and dvy/dy:   c11 s_h + c13 s_v
    to dvz/dz:              c13 s_h + c33 s_v

(2-D: to dvx/dx: c11 s_h + c13 s_v; to dvy/dy [axis]: c13 s_h + c33 s_v.)

The adjoint stress update transposes the velocity update (`v += dt b
d sigma/d`), giving the usual adjoint-stress-from-adjoint-velocity terms
with the SAME staggered stencils transposed (identical structure to
elastic's; nothing VTI-specific survives here because buoyancy is
isotropic).

Sanity identity: at c11 = c33 = c13 (elliptical, isotropic limit) the
constitutive adjoint couplings collapse to the isotropic ones with
lamb = c13, mu = 0.

## 5. Model gradients

With `S_a := dt * (PML-corrected forward dv_a/da at the stress node,
as stored by the forward for the backward pass)` — the same stored
quantities elastic_vti uses — and s_h, s_v the adjoint stresses AFTER the
time-t adjoint update (matching elastic_vti.c's convention for which
time level correlates):

    grad_c11 = sum_t  s_h * (S_x + S_y)          [2-D: s_h * S_x]
    grad_c13 = sum_t  (s_v * (S_x + S_y) + s_h * S_z)
                                                 [2-D: s_v S_x + s_h S_y]
    grad_c33 = sum_t  s_v * S_z                  [2-D: s_v * S_y]
    grad_b   : identical in structure to elastic's buoyancy gradient
               (adjoint velocity times stored stress-derivative), with the
               averaging adjoint SCATTERING each staggered-node
               contribution back to the two neighbouring integer nodes
               (transpose of the two-point average) — reuse elastic's path.

Consistency check against elastic_vti.c's documented full-system formulas
under the §2 transpose map (s_h = s_xx + s_yy, S_xx/S_yy stored
identically):

    grad_c11^full + grad_c12^full
      = sum (s_xx S_x + s_yy S_y) + sum (s_xx S_y + s_yy S_x)
      = sum (s_xx + s_yy)(S_x + S_y) = sum s_h (S_x + S_y)   ✓
    grad_c13^full = sum (s_zz (S_x + S_y) + (s_xx + s_yy) S_z)
                  = sum (s_v (S_x + S_y) + s_h S_z)          ✓
    grad_c33^full = sum s_zz S_z = sum s_v S_z               ✓

**step_ratio (model_gradient_sampling_interval).** Every gradient
accumulation above is gated on `(t % step_ratio) == 0` AND multiplied by
`(DW_DTYPE)step_ratio` — exactly the pattern elastic.c implements. This
scaling is silently a NO-OP at step_ratio == 1 (an easy omission to miss);
verify explicitly at step_ratio in {1, 2, 3}, since a parity gate run only
at 1 proves nothing about it.

## 6. Oracle test protocol

Strict parity vs `elastic_vti(c11, c13, c33, c55=0[, c66=0], buoyancy)`,
eager-vs-eager and compiled-vs-compiled:

1. Same grid, dt, pml_width, accuracy, dtype=float64; models with genuine
   heterogeneity and eps > delta somewhere; sources: one run with velocity-
   component sources, one with the sigma_h <-> (sigmaxx AND sigmayy, same
   amplitude each) mapping of §2; nt odd AND even (an HVP bug in this class
   can be odd-nt-only, so both parities must be checked).
2. Forward: sigma_h == sigmaxx == sigmayy, sigma_v == sigmazz, velocities
   equal — near machine precision (both systems compute the same floating-
   point quantities up to summation-order differences; pin the achieved
   level, expect <= ~1e-12 rel in float64).
3. Gradients: VELOCITY receivers only (per §2); identical random residual
   projection as loss; compare grad_c11 (vs the folded full-system c11
   gradient), grad_c13, grad_c33, grad_buoyancy, grad_source. 2-D and 3-D.
   3-D is MANDATORY — the s_h = s_xx + s_yy summation hazard is invisible
   in 2-D.
4. Free surface: repeat (2) with one free face; if mismatch/instability,
   restrict per §3 and record.
5. If `elastic_vti` rejects exactly-zero c55/c66 anywhere in validation or
   averaging: escalate immediately (do NOT substitute 1e-30 silently —
   that invalidates the oracle's precision claim).

Beyond the oracle (independent physics checks): Christoffel
phase-velocity vs angle for the V_S=0 qP surface (derive in-test from the
Cij Christoffel matrix, pattern in test_elastic_vti.py); isotropic-limit
comparison against `acoustic` (pin the achieved agreement level — the
formulations differ by the two-stress split, so document rather than
assume machine equality); reciprocity; PML absorption bound; absolute FD
gradcheck of every input; eps < delta raises; artifact-taper helper
closed-form check.

## 7. Implementation notes (eager / compiled kernels)

- Eager: dedicated core in the elastic eager-core style with the REDUCED
  field set (2-D: vx, vy, sigma_h, sigma_v + 4 PML memories; 3-D: vx, vy,
  vz, sigma_h, sigma_v + 6 PML memories). Do not run elastic_vti's core
  with zero shear models — halving the field count and dropping the shear
  stencils IS the feature (cost). Keep loop structure parallel to
  elastic_vti's eager core so the compiled kernels can template from it
  directly.
- Kernels: template from elastic_vti.c/.cu by deletion + renaming (symbol
  family `acoustic_vti_*`), keeping the region-macro skeleton, storage,
  IVM, and C-PML machinery. Put this document's §4/§5 formulas in the .c
  header comment (house style) and cite this file.
- Hazard checklist for the reviewer: (a) 3-D s_h summation factor (§2);
  (b) step_ratio gating AND scaling on every accumulation site (§5) —
  grep-count `(DW_DTYPE)step_ratio` sites explicitly rather than assuming
  it is applied everywhere it is needed; (c) ping-pong/aux block swaps in
  any chunked backward (verify chunked == single-chunk bit-identical);
  (d) store-write bools gated on storage_mode; (e) .c/.cu lockstep —
  parameter-position extraction via gcc -E / nvcc -E; (f) eps >= delta
  validated AFTER model preparation/averaging, not only on raw inputs.
