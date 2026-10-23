# Derivation: 2-D elastic TTI (`elastic_tti`) propagator

*Developer/reference documentation: the specification and discrete derivation
behind `deepwave.elastic_tti` (2-D, no compiled C/CUDA backend). See
{doc}`usage` (the "ElasticTti" section) for the user-facing propagator entry.
Any implementation that disagrees with this derivation should be treated as a
discrepancy to resolve deliberately, not silently patched in either
direction.*

**Status: implemented (2-D, both Python modes -- eager and torch_compile --
no compiled C/CUDA backend).** The compiled C/CUDA backend and 3-D (including
the `azimuth` degree of freedom and `c66`) are future work.

## 1. Continuous system

A tilted transversely-isotropic (TTI) medium is a VTI medium whose symmetry
axis has been rotated away from the vertical by a dip angle `tilt`. The VTI
stiffness (Voigt notation, symmetry axis = "3") is parameterised by
`c11, c13, c33, c55` (and `c66`; in the 2-D P-SV plane `c66` does not appear,
see §2). The rotated stiffness is obtained by the **Bond transformation**
(Bond, 1943, *Bell Syst. Tech. J.* **22**, 1), which is the Voigt form of the
fourth-rank tensor rotation

    C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs.

For the 2-D P-SV problem (motion in the vertical `(y, x)` plane, `y` = slow =
axis dimension; propagation invariant along the out-of-plane direction), the
relevant rotation is by `tilt` about the out-of-plane axis, taking the
symmetry axis from vertical toward the fast (`x`) dimension.

## 2. Symmetry class after rotation (the nonzero-`Cij` classification)

Rotating a VTI tensor by a general in-plane `tilt` produces a **monoclinic**
in-plane stiffness. Working in the reduced 2-D Voigt basis `(xx, yy, xy)`
(here `yy` denotes the axis-normal-stress component and `xy` the in-plane
shear), the in-plane P-SV subsystem is exactly the 3×3 block

    C' = [[C11', C13', C15'],
          [C13', C33', C35'],
          [C15', C35', C55']]

with **six** independent constants. Two facts were verified symbolically
(sympy) before any kernel was written, by two independent routes — the 2-D
Bond matrix `C' = M Cred Mᵀ` and the full fourth-rank tensor rotation — which
agree identically:

1. **Reduction.** At `tilt = 0`, `C15' = C35' = 0` and
   `(C11', C13', C33', C55') = (c11, c13, c33, c55)` exactly, so the system
   reduces to `elastic_vti`.
2. **Decoupling.** The in-plane P-SV subsystem `(xx, yy, xy)` decouples from
   the out-of-plane (SH) motion for any `tilt`; the couplings to the
   out-of-plane strain `e_zz` (the only place `c66` appears) are irrelevant in
   2-D. **Hence `c66` does not enter the 2-D P-SV equations and is required to
   be `None`** (as in `elastic_vti`).

With `M = [[c², s², 2cs], [s², c², −2cs], [−cs, cs, c²−s²]]`
(`c = cos tilt`, `s = sin tilt`) and
`Cred = [[c11, c13, 0], [c13, c33, 0], [0, 0, c55]]`, the six constants are the
entries of `M Cred Mᵀ` (implemented in `_bond_rotate_2d`). The Bond rotation is
applied once, differentiably, at model-preparation time, so autograd
propagates gradients back to `c11, c13, c33, c55` and `tilt`.

The continuous 2-D stress-rate relation (axis dimension `y` slowest, `x`
fastest; `e_yy = dvy/dy`, `e_xx = dvx/dx`, `gamma = dvx/dy + dvy/dx`):

    d(sigma_yy)/dt = C33' e_yy + C13' e_xx + C35' gamma
    d(sigma_xx)/dt = C13' e_yy + C11' e_xx + C15' gamma
    d(sigma_xy)/dt = C35' e_yy + C15' e_xx + C55' gamma

The `C15'`/`C35'` terms are the **normal/shear coupling** absent in VTI; they
are what makes the medium monoclinic.

## 3. Discrete scheme

The staggered grid, velocity update (`v += b·dt·div σ`, physics-independent),
C-PML/M-PML memory variables, IVM free surface, source/receiver handling, and
model-preparation averaging are inherited unchanged from
`elastic`/`elastic_vti`; only the constitutive stress update differs. The
velocity update is unchanged because buoyancy is isotropic.

**Co-location of the cross-terms.** On the standard (Virieux) staggered grid,
the normal strains `e_yy, e_xx` live at the integer normal-stress node, while
the shear strain `gamma` lives at the staggered shear-stress node. The
`C15'`/`C35'` couplings therefore combine quantities defined at different
node locations and require interpolation. Let `A` be the four-point average
from the integer nodes onto the shear node (the same four cells and trailing
zero-pad used for the shear-modulus averaging, so interpolated strains are
co-located with `c55_rot_yx` on the shear node), and `Aᵀ` its exact transpose.
The discrete update, with the integer-node rotated stiffnesses `c11_rot`,
`c13_rot`, `c33_rot` and the rotated shear-node stiffnesses `c15_rot_yx`,
`c35_rot_yx`, `c55_rot_yx` (the last harmonically averaged like the isotropic
shear modulus, so `tilt = 0` reproduces `elastic_vti` exactly; the
sign-changing `C15'`/`C35'` arithmetically averaged), is

    sigma_yy += dt (c33_rot e_yy + c13_rot e_xx + Aᵀ(c35_rot_yx · gamma))
    sigma_xx += dt (c13_rot e_yy + c11_rot e_xx + Aᵀ(c15_rot_yx · gamma))
    sigma_xy += dt (c35_rot_yx · A(e_yy) + c15_rot_yx · A(e_xx)
                    + c55_rot_yx · gamma).

Every rotated (Bond-transformed) quantity carries `_rot` in its name, and
these are the names the callback and gradient interfaces use. The unsuffixed
`c11`, `c13`, `c33`, `c55` always denote the *public* untilted VTI
stiffnesses the user passes to `deepwave.elastic_tti`, which are a different
quantity wherever `tilt` is nonzero.

**Symmetry (reciprocity).** With the cross-coupling assembled as `c·A` in the
shear-stress update and `Aᵀ·c` in the normal-stress update, the discrete
stiffness operator `K` (mapping the strain triple to the stress triple) is
exactly symmetric — `(cA)ᵀ = Aᵀc` for the diagonal (pointwise) `c` — so the
scheme is energy-conserving and reciprocal. This is verified in the tests both
directly (the `A`/`Aᵀ` transpose identity) and end-to-end (elastodynamic
reciprocity). Because the eager kernels are pure differentiable PyTorch ops,
the adjoint is native autograd and is exact.

**Accuracy note.** The `A`/`Aᵀ` interpolation makes the `C15'`/`C35'`
cross-terms second-order accurate in space regardless of the `accuracy`
setting (which still controls the underlying first-derivative stencils). This
is a standard limitation of TTI on the non-rotated staggered grid; the
phase-velocity oracle (§4) confirms it is accurate to well within a couple of
percent across tilt and angle.

**Maximum phase velocity / CFL.** The maximum phase velocity of a homogeneous
anisotropic medium is rotation-invariant (rotating the medium rotates the
slowness surface without changing its extremal radius). The `elastic_vti`
trace-of-Christoffel bound `sqrt((max(c11, c33) + c55)·buoyancy)` computed from
the untilted stiffnesses is therefore a valid upper bound for any `tilt`, and
is reused directly for the CFL condition and PML profiles.

## 4. Christoffel phase-velocity oracle

The 2-D monoclinic Christoffel matrix for a unit slowness/propagation
direction `n = (n_x, n_z)` (motion `u_x, u_z`) is `Gamma_ik = C'_ijkl n_j n_l`,
reduced to

    G11 = C11' n_x² + C55' n_z² + 2 C15' n_x n_z
    G33 = C55' n_x² + C33' n_z² + 2 C35' n_x n_z
    G13 = C15' n_x² + C35' n_z² + (C13' + C55') n_x n_z,

with `rho v² ` the eigenvalues of `[[G11, G13], [G13, G33]]` (qP the larger,
qSV the smaller). It was verified numerically that this **equals the VTI
Christoffel solution evaluated at the tilt-shifted angle** `phi − tilt`
to ~1e-13 across a sweep of tilt and propagation angles — the analytic
statement that tilting rotates the phase-velocity surface by exactly `tilt`.
The test suite measures plane-wave slownesses in a homogeneous tilted medium
and compares them to this oracle (with a wrong-sign-tilt negative control that
must fail).

## 5. Stability and `mpml_fraction`

Tilted anisotropy is a well-known trigger of classical C-PML instability: the
off-axis group/phase velocity mismatch in the boundary layer can excite
growing modes. The multiaxial PML (M-PML) of Meza-Fajardo & Papageorgiou
(2008, *BSSA* **98**, 1811) cures this by adding a small fraction `mpml_fraction` of
cross-axis damping, at the cost of slightly larger boundary reflections and a
small (deliberate) break of perfect matching and exact reciprocity.
`elastic_tti` defaults `mpml_fraction` to 0.0, matching every other propagator;
users who see PML instability with a tilted medium may want to pass
`mpml_fraction` around 0.15. The test suite documents a validated operating
envelope at `mpml_fraction = 0.15` by sweeping the corners of (max tilt) ×
(anisotropy strength) — including 90° tilt and strong/negative `delta` —
and running each corner for four domain-crossing times, asserting the
wavefield energy decays (does not blow up).

## 6. Scope and follow-ups

Implemented: 2-D, both Python modes (eager/torch_compile), gradients
(including `tilt`) via autograd.
Deferred to a later release: the compiled C/CUDA backend
(a general-`Cij` monoclinic stencil consuming the precomputed rotated fields,
templated from `elastic_vti.c/.cu`), and full 3-D TTI (Bond rotation with
`tilt` and `azimuth`, the resulting orthorhombic-in-rotated-frame / up-to-21
constant stencil, and `c66`).
