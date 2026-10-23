#ifndef DW_ACOUSTIC_VTI_KERNELS_H
#define DW_ACOUSTIC_VTI_KERNELS_H

/*
 * acoustic_vti_kernels.h
 *
 * Single-source per-grid-point stencil math for the pseudo-acoustic VTI
 * (vertically transverse isotropic, V_S = 0) wave equation propagator (see
 * acoustic_vti.c for the physics/PML/free-surface description and the
 * derivation citations, docs/acoustic_vti_derivation.md). This header is
 * included by BOTH acoustic_vti.c (compiled by gcc) and acoustic_vti.cu
 * (compiled by nvcc), after each has defined the FUNC()/CAT() name-mangling
 * macros (and the DW_MPML-dependent ACOUSTIC_VTI_VARIANT()) and before either
 * uses the ND_INDEX/P_A(*)/P_B(*)/VZ/VY/VX/SIGMA_V/SIGMA_H/... family below.
 *
 * acoustic_vti has TWO velocity-fed stress components (sigma_v, sigma_h)
 * updated TOGETHER as one combined equation (they share the same dv/dn
 * terms, weighted by c11/c13/c33 -- the c66 = 0 reduction of elastic_vti's
 * normal-stress update), and THREE velocity components, each fed by exactly
 * ONE stress derivative (no shear cross-terms survive the V_S = 0
 * reduction: vz's axis derivative always couples to sigma_v; vx's transverse
 * derivative always couples to sigma_h; vy is axis in 2-D [couples to
 * sigma_v] but transverse in 3-D [couples to sigma_h]). This mirrors
 * acoustic_kernels.h's shape exactly (three independent per-axis velocity
 * cells + one combined stress cell, forward and backward), not elastic's
 * (which has independent shear-stress cells too, since elastic_vti's shear
 * stresses are identically zero here and were already dropped by the
 * acoustic_vti.c/.cu reduction). It contains:
 *
 *  * The per-axis addressing macros (ND_INDEX, the M-PML-or-1-D PML profile
 *    accessors P_AZ/P_BZ/P_AZH/P_BZH/.../P_AXH/P_BXH, the field-access macros
 *    VZ/VY/VX/SIGMA_V/SIGMA_H, the model-access macros C11/C13/C33/
 *    BUOYANCY_Z/Y/X, the M_VZZ/M_VYY/M_VXX/M_SIGMA_V_Z/M_SIGMA_H_Y/
 *    M_SIGMA_V_Y/M_SIGMA_H_X PML-memory families, and the adjoint-term
 *    macros VZ_Z(+_PML), VY_Y(+_PML), VX_X(+_PML), SIGMA_V_Z(+_PML),
 *    SIGMA_H_Y(+_PML), SIGMA_V_Y(+_PML), SIGMA_H_X(+_PML)), lifted verbatim
 *    from acoustic_vti.c's/acoustic_vti.cu's copies of these macros (the two
 *    files' copies were already textually identical modulo the `i`-vs-`j`
 *    addressing difference resolved below -- see the addressing-convention
 *    note). The DW_MPML-dependent PML-region-skip decision (DW_PML_SKIP on
 *    the CPU driver, DW_PML_TEST on the CUDA driver) is NOT part of this
 *    header: it is driver-side scaffolding (it depends on the outer
 *    region-loop indices on the CPU driver and on the raw z/y/x position
 *    against the __constant__ pml_*0/pml_*1 bounds on the CUDA driver, two
 *    genuinely different parallelisation strategies), so both drivers keep
 *    computing their own `pml_z`/`pml_y`/`pml_x` bool and simply pass the
 *    resulting bool into the functions below (matching
 *    acoustic_kernels.h's/elastic_kernels.h's `pml_z`/`pml_y`/`pml_x`
 *    parameters exactly).
 *  * FUNC(forward_vz_cell) [DW_NDIM >= 3] / FUNC(forward_vy_cell) /
 *    FUNC(forward_vx_cell): the per-grid-point half-grid velocity update for
 *    each axis (single stress-derivative term plus that axis's C-PML memory
 *    recursion), matching acoustic_vti.c's/acoustic_vti.cu's three
 *    independent velocity loop/kernel bodies. FUNC(forward_sigma_cell): the
 *    combined sigma_v/sigma_h update (all present axes' dv/dn terms, each
 *    weighted by c11/c13/c33 per the header derivation, plus each axis's
 *    velocity-derivative C-PML memory recursion and grad-store writes),
 *    matching acoustic_vti.c's/acoustic_vti.cu's single combined stress
 *    loop/kernel body -- the direct analogue of acoustic_kernels.h's
 *    FUNC(forward_p_cell).
 *  * FUNC(backward_vz_cell) [DW_NDIM >= 3] / FUNC(backward_vy_cell) /
 *    FUNC(backward_vx_cell): the adjoint velocity update per axis --
 *    structurally the transpose of FUNC(forward_sigma_cell) (it
 *    differentiates the sigma_v/sigma_h adjoint fields, weighted by
 *    c11/c13/c33, to update the velocity adjoint), plus that axis's
 *    M_SIGMA_V_Z/M_SIGMA_H_Y/M_SIGMA_V_Y/M_SIGMA_H_X ping-ponged PML-memory
 *    recursion and the buoyancy gradient accumulation.
 *    FUNC(backward_sigma_cell): the adjoint sigma_v/sigma_h update --
 *    structurally the transpose of FUNC(forward_vz_cell)/vy_cell/vx_cell (it
 *    differentiates the velocity adjoint fields, weighted by buoyancy, to
 *    update the stress adjoint), the (grad_c11, grad_c13, grad_c33)
 *    accumulation, and the M_VZZ/M_VYY/M_VXX PML-memory update (read by
 *    FUNC(backward_vz_cell)/vy_cell/vx_cell at the *next* [i.e. previously
 *    processed, since time runs backward] time step).
 *
 * PML memory-variable buffering: the M_VZZ/M_VYY/M_VXX family is a SINGLE
 * buffer per array, read (with a finite-difference neighbour offset, via the
 * VZ_Z_PML/VY_Y_PML/VX_X_PML macros) by FUNC(backward_vz_cell)/vy_cell/
 * vx_cell and written in place (centre point only) by
 * FUNC(backward_sigma_cell) LATER in the same time step -- since the backward
 * driver always runs the velocity-adjoint region before the sigma-adjoint
 * region, this is safe without ping-pong (the velocity-adjoint functions
 * always see the value the sigma-adjoint function wrote at the *previous*
 * [chronologically later] time step). The M_SIGMA_V_Z/M_SIGMA_H_Y/
 * M_SIGMA_V_Y/M_SIGMA_H_X family is the reverse dependency (written by
 * FUNC(backward_vz_cell)/vy_cell/vx_cell, read with a neighbour offset by
 * FUNC(backward_sigma_cell) via the SIGMA_V_Z_PML/SIGMA_H_Y_PML/
 * SIGMA_V_Y_PML/SIGMA_H_X_PML macros) but in the OPPOSITE order within a time
 * step (the velocity-adjoint region runs first and would otherwise read the
 * value it is about to overwrite this same step), so it DOES need the
 * old/new ping-pong pair the driver already maintains (m_sigma_v_z vs
 * m_sigma_v_zn, etc.); FUNC(backward_vz_cell)/vy_cell/vx_cell simply take
 * "old" (read, and also read at the centre point to seed the recursion) and
 * "new" (write, centre point only) pointers, matching the existing
 * ping-pong parameter names.
 *
 * Everything about HOW the loop over grid points is launched and threaded --
 * the OpenMP parallel/simd/collapse loop nest and PML-region triple loop in
 * acoustic_vti.c, the CUDA kernel launch configuration and per-thread
 * grid-point assignment (including the two-kernel forward_kernel_v/
 * forward_kernel_sigma and backward_kernel_sigma/backward_kernel_v split) in
 * acoustic_vti.cu -- stays in acoustic_vti.c/acoustic_vti.cu exactly as
 * before; only the math performed once the caller has identified "this" grid
 * point (and, for the backward functions, decided whether this cell is
 * within the relevant axis's PML region) lives here.
 *
 * Addressing convention (matches scalar_kernels.h/acoustic_kernels.h/
 * elastic_kernels.h exactly): vz/vy/vx/sigma_v/sigma_h/m_vzz/m_vyy/m_vxx/
 * m_sigma_v_z/m_sigma_h_y/m_sigma_v_y/m_sigma_h_x/dv(*)d(*)_store_1/
 * grad_buoyancy_(*)/... are addressed by `i` (the CPU driver's per-shot-local
 * flat index, which is also valid as the CUDA driver's global shot-spanning
 * index since both drivers pass already-appropriately-offset pointers);
 * c11_shot/c13_shot/c33_shot/buoyancy_*_shot are addressed by `j` (the
 * per-shot-local index) wherever a POINTER is taken (used only by the
 * backward functions, via the C11/C13/C33/BUOYANCY_Z/Y/X macros), because the
 * CUDA driver's model pointers are shot-local (offset by shot_idx*shot_numel,
 * not `i`) while the wavefield arrays span all shots. On the CPU driver,
 * wavefields/model arrays are ALL already offset to the current shot, so `i`
 * and `j` are the same value there and the CPU driver simply passes `i` for
 * both. The forward functions need no `j`: c11/c13/c33/buoyancy_* are only
 * ever read at the *centre* point (no neighbour offset -- confirmed against
 * both acoustic_vti.c's and acoustic_vti.cu's forward loops/kernels), so the
 * caller passes that value directly, by value, exactly as
 * acoustic_kernels.h's forward_v*_cell/forward_p_cell do for buoyancy_*_shot/
 * k_shot.
 *
 * DW_HOST_DEVICE marks these functions callable from either gcc (a plain
 * forced-inline static function) or nvcc (additionally callable from device
 * code).
 */

#ifdef __CUDACC__
#define DW_HOST_DEVICE static __host__ __device__ __forceinline__
#elif defined(_MSC_VER)
#define DW_HOST_DEVICE static __forceinline
#else
#define DW_HOST_DEVICE static inline __attribute__((always_inline))
#endif

#if DW_NDIM == 3
#define ND_INDEX(i, dz, dy, dx) (i + (dz) * ny * nx + (dy) * nx + (dx))
#elif DW_NDIM == 2
#define ND_INDEX(i, dz, dy, dx) (i + (dy) * nx + (dx))
#else /* DW_NDIM == 1 */
#define ND_INDEX(i, dz, dy, dx) (i + (dx))
#endif

/* PML profile access. The standard (non-M-PML) profiles are 1-D per axis, so a
 * given axis's profile is indexed by that axis's own coordinate only
 * (e.g. by[y + dy]). The multiaxial (M-PML) profiles vary on every axis, so
 * they are indexed at the full-grid spatial cell. PML_IDX is the spatial-only
 * flattened index (no shot offset -- the profiles have no shot dimension) of
 * the cell offset (dz, dy, dx) from the current (z, y, x). Each P_* accessor
 * takes the single offset along that profile's own axis; under DW_MPML it
 * expands to a full-grid load, otherwise to the historical 1-D load (which is
 * textually equivalent to the previous code, so the default kernel is
 * unchanged). */
#if DW_NDIM == 3
#define PML_IDX(dz, dy, dx) \
  ((z + (dz)) * ny * nx + (y + (dy)) * nx + (x + (dx)))
#elif DW_NDIM == 2
#define PML_IDX(dz, dy, dx) ((y + (dy)) * nx + (x + (dx)))
#else
#define PML_IDX(dz, dy, dx) (x + (dx))
#endif

#ifdef DW_MPML
#if DW_NDIM >= 3
#define P_AZ(d) az[PML_IDX(d, 0, 0)]
#define P_BZ(d) bz[PML_IDX(d, 0, 0)]
#define P_AZH(d) azh[PML_IDX(d, 0, 0)]
#define P_BZH(d) bzh[PML_IDX(d, 0, 0)]
#endif
#if DW_NDIM >= 2
#define P_AY(d) ay[PML_IDX(0, d, 0)]
#define P_BY(d) by[PML_IDX(0, d, 0)]
#define P_AYH(d) ayh[PML_IDX(0, d, 0)]
#define P_BYH(d) byh[PML_IDX(0, d, 0)]
#endif
#define P_AX(d) ax[PML_IDX(0, 0, d)]
#define P_BX(d) bx[PML_IDX(0, 0, d)]
#define P_AXH(d) axh[PML_IDX(0, 0, d)]
#define P_BXH(d) bxh[PML_IDX(0, 0, d)]
#else /* standard 1-D per-axis profiles */
#if DW_NDIM >= 3
#define P_AZ(d) az[z + (d)]
#define P_BZ(d) bz[z + (d)]
#define P_AZH(d) azh[z + (d)]
#define P_BZH(d) bzh[z + (d)]
#endif
#if DW_NDIM >= 2
#define P_AY(d) ay[y + (d)]
#define P_BY(d) by[y + (d)]
#define P_AYH(d) ayh[y + (d)]
#define P_BYH(d) byh[y + (d)]
#endif
#define P_AX(d) ax[x + (d)]
#define P_BX(d) bx[x + (d)]
#define P_AXH(d) axh[x + (d)]
#define P_BXH(d) bxh[x + (d)]
#endif /* DW_MPML */

// Wavefields, addressed by i (see the addressing-convention note above).
#if DW_NDIM >= 3
#define VZ(dz, dy, dx) vz[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define VY(dz, dy, dx) vy[ND_INDEX(i, dz, dy, dx)]
#endif
#define VX(dz, dy, dx) vx[ND_INDEX(i, dz, dy, dx)]
#define SIGMA_V(dz, dy, dx) sigma_v[ND_INDEX(i, dz, dy, dx)]
#define SIGMA_H(dz, dy, dx) sigma_h[ND_INDEX(i, dz, dy, dx)]

// Models, addressed by the shot-local index j (see the addressing-convention
// note above) -- used only by the backward (adjoint) functions, via the
// VZ_Z/VY_Y/VX_X(+_PML)/SIGMA_V_Z/SIGMA_H_Y/SIGMA_V_Y/SIGMA_H_X(+_PML) macros
// below.
#define C11(dz, dy, dx) c11_shot[ND_INDEX(j, dz, dy, dx)]
#define C13(dz, dy, dx) c13_shot[ND_INDEX(j, dz, dy, dx)]
#define C33(dz, dy, dx) c33_shot[ND_INDEX(j, dz, dy, dx)]
#if DW_NDIM >= 3
#define BUOYANCY_Z(dz, dy, dx) buoyancy_z_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define BUOYANCY_Y(dz, dy, dx) buoyancy_y_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#define BUOYANCY_X(dz, dy, dx) buoyancy_x_shot[ND_INDEX(j, dz, dy, dx)]

// C-PML memory variables for the velocity-derivative terms feeding the
// stress update (M_VZZ/M_VYY/M_VXX -- see the PML memory-variable buffering
// note above) and for the stress-derivative terms feeding the velocity
// update (M_SIGMA_V_Z/M_SIGMA_H_Y/M_SIGMA_V_Y/M_SIGMA_H_X), addressed by i.
// Each resolves to whatever array a given calling function names its own
// parameter (mirroring elastic_kernels.h/acoustic_vti.c/acoustic_vti.cu).
#if DW_NDIM >= 3
#define M_VZZ(dz, dy, dx) m_vzz[ND_INDEX(i, dz, dy, dx)]
#endif
#define M_VYY(dz, dy, dx) m_vyy[ND_INDEX(i, dz, dy, dx)]
#define M_VXX(dz, dy, dx) m_vxx[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM == 3
#define M_SIGMA_V_Z(dz, dy, dx) m_sigma_v_z[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMA_H_Y(dz, dy, dx) m_sigma_h_y[ND_INDEX(i, dz, dy, dx)]
#else
#define M_SIGMA_V_Y(dz, dy, dx) m_sigma_v_y[ND_INDEX(i, dz, dy, dx)]
#endif
#define M_SIGMA_H_X(dz, dy, dx) m_sigma_h_x[ND_INDEX(i, dz, dy, dx)]

// Adjoint access terms (backward velocity update): the stiffness matrix is
// symmetric, so the transpose reuses the same Cij; each velocity is fed by
// exactly one axis-coupling term (dvz/dz [3-D] / dvy/dy [2-D, axis]: c13 s_h
// + c33 s_v) or transverse-coupling term (dvx/dx always, dvy/dy [3-D,
// transverse]: c11 s_h + c13 s_v) -- see acoustic_vti.c Sec. 4 for the
// derivation. s_h/s_v (sigma_h/sigma_v) ARE the reduced adjoint stresses
// already (not "duplicate-then-sum", unlike elastic_vti), so these
// coefficients are used directly, once each.
#if DW_NDIM >= 3
#define VZ_Z(dz, dy, dx)                         \
  (dt * (C13(dz, dy, dx) * SIGMA_H(dz, dy, dx) + \
         C33(dz, dy, dx) * SIGMA_V(dz, dy, dx)))
#define VZ_Z_PML(dz, dy, dx)                     \
  (dt * (1 + P_BZ(dz)) *                         \
       (C13(dz, dy, dx) * SIGMA_H(dz, dy, dx) +  \
        C33(dz, dy, dx) * SIGMA_V(dz, dy, dx)) + \
   P_BZ(dz) * M_VZZ(dz, dy, dx))
#endif

#if DW_NDIM == 2
// 2-D: axis = y. dvydy couples to sigma_h via c13, to sigma_v via c33.
#define VY_Y(dz, dy, dx)                         \
  (dt * (C13(dz, dy, dx) * SIGMA_H(dz, dy, dx) + \
         C33(dz, dy, dx) * SIGMA_V(dz, dy, dx)))
#define VY_Y_PML(dz, dy, dx)                     \
  (dt * (1 + P_BY(dy)) *                         \
       (C13(dz, dy, dx) * SIGMA_H(dz, dy, dx) +  \
        C33(dz, dy, dx) * SIGMA_V(dz, dy, dx)) + \
   P_BY(dy) * M_VYY(dz, dy, dx))
#else
// 3-D: dvydy (transverse) couples to sigma_h via c11, to sigma_v via c13.
#define VY_Y(dz, dy, dx)                         \
  (dt * (C11(dz, dy, dx) * SIGMA_H(dz, dy, dx) + \
         C13(dz, dy, dx) * SIGMA_V(dz, dy, dx)))
#define VY_Y_PML(dz, dy, dx)                     \
  (dt * (1 + P_BY(dy)) *                         \
       (C11(dz, dy, dx) * SIGMA_H(dz, dy, dx) +  \
        C13(dz, dy, dx) * SIGMA_V(dz, dy, dx)) + \
   P_BY(dy) * M_VYY(dz, dy, dx))
#endif

// dvxdx (always transverse) couples to sigma_h via c11, to sigma_v via c13.
#define VX_X(dz, dy, dx)                         \
  (dt * (C11(dz, dy, dx) * SIGMA_H(dz, dy, dx) + \
         C13(dz, dy, dx) * SIGMA_V(dz, dy, dx)))
#define VX_X_PML(dz, dy, dx)                     \
  (dt * (1 + P_BX(dx)) *                         \
       (C11(dz, dy, dx) * SIGMA_H(dz, dy, dx) +  \
        C13(dz, dy, dx) * SIGMA_V(dz, dy, dx)) + \
   P_BX(dx) * M_VXX(dz, dy, dx))

// Adjoint access terms (backward stress update): the transpose of the
// forward velocity update's dependence on the stress it reads (via
// buoyancy). sigma_v is fed by exactly one velocity (the axis one); sigma_h
// is fed by vx always AND (3-D only) by vy -- TWO feed terms, because in the
// full elastic_vti system BOTH the forward vx and vy velocity updates read
// sigmaxx/sigmayy respectively, and sigma_h is their reduction.
#if DW_NDIM >= 3
#define SIGMA_V_Z_PML(dz, dy, dx)                                   \
  ((1 + P_BZH(dz)) * dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx) + \
   P_BZH(dz) * M_SIGMA_V_Z(dz, dy, dx))
#define SIGMA_V_Z(dz, dy, dx) (dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))
#define SIGMA_H_Y_PML(dz, dy, dx)                                   \
  ((1 + P_BYH(dy)) * dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx) + \
   P_BYH(dy) * M_SIGMA_H_Y(dz, dy, dx))
#define SIGMA_H_Y(dz, dy, dx) (dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#else
#define SIGMA_V_Y_PML(dz, dy, dx)                                   \
  ((1 + P_BYH(dy)) * dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx) + \
   P_BYH(dy) * M_SIGMA_V_Y(dz, dy, dx))
#define SIGMA_V_Y(dz, dy, dx) (dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#endif
#define SIGMA_H_X_PML(dz, dy, dx)                                   \
  ((1 + P_BXH(dx)) * dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx) + \
   P_BXH(dx) * M_SIGMA_H_X(dz, dy, dx))
#define SIGMA_H_X(dz, dy, dx) (dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))

/* ---- Forward: per-axis half-grid velocity updates ---- */

#if DW_NDIM >= 3
/* vz += buoyancy_z * dt * d(sigma_v)/dz (the axis derivative -- vz is fed by
 * exactly this one stress-derivative term; no cross terms). `i` addresses
 * sigma_v/vz/m_sigma_v_z/dvzdbuoyancy_store_1 (per-shot-local on the CPU
 * driver, global on the CUDA driver); `buoyancy_z_shot` is the (already
 * shot-selected) buoyancy VALUE at this point (read once by the caller --
 * this update never needs a neighbour offset of it); `pml_z` is true when
 * this point is inside the z-axis PML region (false in the interior). `y`/
 * `x` are needed only for the M-PML full-grid PML_IDX lookup (P_AZH/P_BZH),
 * not for the DIFFZH1 stencil itself.
 */
DW_HOST_DEVICE void FUNC(forward_vz_cell)(
    DW_DTYPE const *__restrict const sigma_v, DW_DTYPE *__restrict const vz,
    DW_DTYPE *__restrict const m_sigma_v_z,
    DW_DTYPE *__restrict const dvzdbuoyancy_store_1,
    DW_DTYPE const buoyancy_z_shot, DW_DTYPE const *__restrict const azh,
    DW_DTYPE const *__restrict const bzh, DW_DTYPE const dt,
    bool const buoyancy_requires_grad, int64_t const i, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, DW_DTYPE const rdz) {
  DW_DTYPE w_sum;
  if (!pml_z) {
    w_sum = DIFFZH1(SIGMA_V);
  } else {
    DW_DTYPE d3 = DIFFZH1(SIGMA_V);
    m_sigma_v_z[i] = P_AZH(0) * m_sigma_v_z[i] + P_BZH(0) * d3;
    d3 += m_sigma_v_z[i];
    w_sum = d3;
  }
  if (buoyancy_requires_grad) {
    dvzdbuoyancy_store_1[i] = dt * w_sum;
  }
  vz[i] += buoyancy_z_shot * dt * w_sum;
}
#endif

/* vy += buoyancy_y * dt * d(sigma_h)/dy [3-D, transverse] or
 * buoyancy_y * dt * d(sigma_v)/dy [2-D, axis] (see FUNC(forward_vz_cell)). */
DW_HOST_DEVICE void FUNC(forward_vy_cell)(
#if DW_NDIM == 3
    DW_DTYPE const *__restrict const sigma_h,
    DW_DTYPE *__restrict const m_sigma_h_y,
#else
    DW_DTYPE const *__restrict const sigma_v,
    DW_DTYPE *__restrict const m_sigma_v_y,
#endif
    DW_DTYPE *__restrict const vy,
    DW_DTYPE *__restrict const dvydbuoyancy_store_1,
    DW_DTYPE const buoyancy_y_shot, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const dt,
    bool const buoyancy_requires_grad, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z,
#endif
    int64_t const y, int64_t const x, bool const pml_y, DW_DTYPE const rdy) {
  DW_DTYPE w_sum;
#if DW_NDIM == 3
  if (!pml_y) {
    w_sum = DIFFYH1(SIGMA_H);
  } else {
    DW_DTYPE d2 = DIFFYH1(SIGMA_H);
    m_sigma_h_y[i] = P_AYH(0) * m_sigma_h_y[i] + P_BYH(0) * d2;
    d2 += m_sigma_h_y[i];
    w_sum = d2;
  }
#else
  if (!pml_y) {
    w_sum = DIFFYH1(SIGMA_V);
  } else {
    DW_DTYPE d2 = DIFFYH1(SIGMA_V);
    m_sigma_v_y[i] = P_AYH(0) * m_sigma_v_y[i] + P_BYH(0) * d2;
    d2 += m_sigma_v_y[i];
    w_sum = d2;
  }
#endif
  if (buoyancy_requires_grad) {
    dvydbuoyancy_store_1[i] = dt * w_sum;
  }
  vy[i] += buoyancy_y_shot * dt * w_sum;
}

/* vx += buoyancy_x * dt * d(sigma_h)/dx (always transverse; see
 * FUNC(forward_vz_cell)). */
DW_HOST_DEVICE void FUNC(forward_vx_cell)(
    DW_DTYPE const *__restrict const sigma_h, DW_DTYPE *__restrict const vx,
    DW_DTYPE *__restrict const m_sigma_h_x,
    DW_DTYPE *__restrict const dvxdbuoyancy_store_1,
    DW_DTYPE const buoyancy_x_shot, DW_DTYPE const *__restrict const axh,
    DW_DTYPE const *__restrict const bxh, DW_DTYPE const dt,
    bool const buoyancy_requires_grad, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z,
#endif
    int64_t const y, int64_t const x, bool const pml_x, DW_DTYPE const rdx) {
  DW_DTYPE w_sum;
  if (!pml_x) {
    w_sum = DIFFXH1(SIGMA_H);
  } else {
    DW_DTYPE d1 = DIFFXH1(SIGMA_H);
    m_sigma_h_x[i] = P_AXH(0) * m_sigma_h_x[i] + P_BXH(0) * d1;
    d1 += m_sigma_h_x[i];
    w_sum = d1;
  }
  if (buoyancy_requires_grad) {
    dvxdbuoyancy_store_1[i] = dt * w_sum;
  }
  vx[i] += buoyancy_x_shot * dt * w_sum;
}

/* ---- Forward: combined sigma_v/sigma_h update ---- */

/* sigma_v += dt*(c33*dvzdz + c13*(dvxdx+dvydy)) [3-D] / dt*(c33*dvydy +
 * c13*dvxdx) [2-D]; sigma_h += dt*(c11*(dvxdx+dvydy) + c13*dvzdz) [3-D] /
 * dt*(c11*dvxdx + c13*dvydy) [2-D] -- the c66 = 0 reduction of elastic_vti's
 * normal-stress update (see acoustic_vti.c Sec. 3), combining every present
 * axis (like scalar_kernels.h's forward_cell combines its axes, and matching
 * acoustic_kernels.h's FUNC(forward_p_cell)). `i` addresses
 * vz/vy/vx/sigma_v/sigma_h/m_vzz/m_vyy/m_vxx/dv?d?_store_1 (per-shot-local on
 * the CPU driver, global on the CUDA driver); `c11_shot`/`c13_shot`/
 * `c33_shot` are the (already shot-selected) stiffness VALUES at this point
 * (read once by the caller -- never need a neighbour offset here); `pml_z`/
 * `pml_y`/`pml_x` are true when this point is inside that axis's PML region.
 */
DW_HOST_DEVICE void FUNC(forward_sigma_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
#endif
    DW_DTYPE const *__restrict const vy, DW_DTYPE const *__restrict const vx,
    DW_DTYPE *__restrict const sigma_v, DW_DTYPE *__restrict const sigma_h,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const m_vzz,
#endif
    DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const dvzdz_store_1,
#endif
    DW_DTYPE *__restrict const dvydy_store_1,
    DW_DTYPE *__restrict const dvxdx_store_1, DW_DTYPE const c11_shot,
    DW_DTYPE const c13_shot, DW_DTYPE const c33_shot, DW_DTYPE const dt,
    bool const cnormal_requires_grad, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const *__restrict const az,
    DW_DTYPE const *__restrict const bz, DW_DTYPE const rdz,
#endif
    int64_t const y, bool const pml_y, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const by, DW_DTYPE const rdy, int64_t const x,
    bool const pml_x, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const bx, DW_DTYPE const rdx) {
#if DW_NDIM >= 3
  DW_DTYPE dvzdz;
  if (pml_z) {
    dvzdz = DIFFZ1(VZ);
    m_vzz[i] = P_AZ(0) * m_vzz[i] + P_BZ(0) * dvzdz;
    dvzdz += m_vzz[i];
  } else {
    dvzdz = DIFFZ1(VZ);
  }
  if (cnormal_requires_grad) {
    dvzdz_store_1[i] = dt * dvzdz;
  }
#endif

  DW_DTYPE dvydy;
  if (pml_y) {
    dvydy = DIFFY1(VY);
    m_vyy[i] = P_AY(0) * m_vyy[i] + P_BY(0) * dvydy;
    dvydy += m_vyy[i];
  } else {
    dvydy = DIFFY1(VY);
  }
  if (cnormal_requires_grad) {
    dvydy_store_1[i] = dt * dvydy;
  }

  DW_DTYPE dvxdx;
  if (pml_x) {
    dvxdx = DIFFX1(VX);
    m_vxx[i] = P_AX(0) * m_vxx[i] + P_BX(0) * dvxdx;
    dvxdx += m_vxx[i];
  } else {
    dvxdx = DIFFX1(VX);
  }
  if (cnormal_requires_grad) {
    dvxdx_store_1[i] = dt * dvxdx;
  }

#if DW_NDIM == 3
  sigma_v[i] += dt * (c33_shot * dvzdz + c13_shot * (dvxdx + dvydy));
  sigma_h[i] += dt * (c11_shot * (dvxdx + dvydy) + c13_shot * dvzdz);
#else
  // 2-D: axis = y. sigma_v = sigmayy-analogue, sigma_h = sigmaxx-analogue.
  sigma_v[i] += dt * (c33_shot * dvydy + c13_shot * dvxdx);
  sigma_h[i] += dt * (c11_shot * dvxdx + c13_shot * dvydy);
#endif
}

/* ---- Backward: per-axis half-grid adjoint velocity updates ---- */

#if DW_NDIM >= 3
/* vz += -d/dz(c13*sigma_h + c33*sigma_v) (transpose of FUNC(forward_sigma_
 * cell)'s dvzdz term); m_sigma_v_zn = ping-ponged PML memory (read by
 * FUNC(backward_sigma_cell) at the *next* [chronologically earlier] reversed
 * time step); grad_buoyancy_z accumulation. `i` addresses sigma_h/sigma_v/vz/
 * m_sigma_v_z/m_sigma_v_zn/dvzdbuoyancy_store_1/grad_buoyancy_z
 * (per-shot-local on the CPU driver, global on the CUDA driver); `j`
 * addresses the already-shot-sliced c13_shot/c33_shot (equal to `i` on the
 * CPU driver; the local per-shot grid index on the CUDA driver, distinct
 * from the global `i`) -- see the addressing-convention note at the top of
 * this file. `m_vzz` is the OLD (this-time-step, single-buffer) velocity-
 * derivative C-PML memory written by FUNC(backward_sigma_cell) at the
 * *previous* [chronologically later] reversed time step (read only here);
 * `m_sigma_v_z` is the OLD (pre-this-step) stress-derivative buffer (read
 * only); `m_sigma_v_zn` is the NEW buffer (written); `pml_z` is true inside
 * the z-axis PML region.
 */
DW_HOST_DEVICE void FUNC(backward_vz_cell)(
    DW_DTYPE const *__restrict const c13_shot,
    DW_DTYPE const *__restrict const c33_shot,
    DW_DTYPE const *__restrict const sigma_h,
    DW_DTYPE const *__restrict const sigma_v, DW_DTYPE *__restrict const vz,
    DW_DTYPE const *__restrict const m_vzz, DW_DTYPE const *__restrict const bz,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, DW_DTYPE const rdz) {
  DW_DTYPE const w_sum = pml_z ? -DIFFZH1(VZ_Z_PML) : -DIFFZH1(VZ_Z);
  vz[i] += w_sum;
}

/* M_SIGMA_V_Z PML memory-variable update for FUNC(backward_vz_cell) --
 * split out and moved to run in a separate pass AFTER FUNC(fill_v_bc_
 * adjoint) (see the driver-side comment at its new call site). vz[i] here
 * must be read post-fold: for a cell the fold accumulates into or clamps to
 * zero (e.g. a rigid on-row velocity node), the value FUNC(backward_vz_cell)
 * computed pre-fold is stale -- the fold changes it immediately afterward --
 * so writing this memory variable, or accumulating the buoyancy gradient
 * (done by the driver right after calling this, using the same post-fold
 * vz[i]), from the pre-fold value would be wrong for exactly those cells.
 * For every other cell (the fold never touches it) vz[i] is identical pre-
 * and post-fold, so this split changes nothing there. */
DW_HOST_DEVICE void FUNC(backward_vz_cell_pml_mem)(
    DW_DTYPE const *__restrict const vz,
    DW_DTYPE const *__restrict const m_sigma_v_z,
    DW_DTYPE *__restrict const m_sigma_v_zn, DW_DTYPE const buoyancy_z_shot,
    DW_DTYPE const *__restrict const azh, DW_DTYPE const dt, int64_t const i,
    int64_t const ny, int64_t const nx, int64_t const z, int64_t const y,
    int64_t const x, bool const pml_z) {
  if (pml_z) {
    m_sigma_v_zn[i] =
        buoyancy_z_shot * dt * P_AZH(0) * vz[i] + P_AZH(0) * m_sigma_v_z[i];
  }
}

#endif

/* Adjoint velocity update, y axis: 3-D transverse (fed by sigma_h, via c11/
 * c13) or 2-D axis (fed by sigma_v, via c13/c33) -- see
 * FUNC(backward_vz_cell). */
DW_HOST_DEVICE void FUNC(backward_vy_cell)(
#if DW_NDIM == 3
    DW_DTYPE const *__restrict const c11_shot,
    DW_DTYPE const *__restrict const c13_shot,
#else
    DW_DTYPE const *__restrict const c13_shot,
    DW_DTYPE const *__restrict const c33_shot,
#endif
    DW_DTYPE const *__restrict const sigma_h,
    DW_DTYPE const *__restrict const sigma_v, DW_DTYPE *__restrict const vy,
    DW_DTYPE const *__restrict const m_vyy, DW_DTYPE const *__restrict const by,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z,
#endif
    int64_t const y, int64_t const x, bool const pml_y, DW_DTYPE const rdy) {
  DW_DTYPE const w_sum = pml_y ? -DIFFYH1(VY_Y_PML) : -DIFFYH1(VY_Y);
  vy[i] += w_sum;
}

/* M_SIGMA_H_Y/M_SIGMA_V_Y PML memory-variable update for FUNC(backward_
 * vy_cell) -- see FUNC(backward_vz_cell_pml_mem)'s comment (same split,
 * same reason: vy[i] must be read post-fold). */
DW_HOST_DEVICE void FUNC(backward_vy_cell_pml_mem)(
    DW_DTYPE const *__restrict const vy,
#if DW_NDIM == 3
    DW_DTYPE const *__restrict const m_sigma_h_y,
    DW_DTYPE *__restrict const m_sigma_h_yn,
#else
    DW_DTYPE const *__restrict const m_sigma_v_y,
    DW_DTYPE *__restrict const m_sigma_v_yn,
#endif
    DW_DTYPE const buoyancy_y_shot, DW_DTYPE const *__restrict const ayh,
    DW_DTYPE const dt, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny, int64_t const z,
#endif
    int64_t const nx, int64_t const y, int64_t const x, bool const pml_y) {
  if (pml_y) {
#if DW_NDIM == 3
    m_sigma_h_yn[i] =
        buoyancy_y_shot * dt * P_AYH(0) * vy[i] + P_AYH(0) * m_sigma_h_y[i];
#else
    m_sigma_v_yn[i] =
        buoyancy_y_shot * dt * P_AYH(0) * vy[i] + P_AYH(0) * m_sigma_v_y[i];
#endif
  }
}

/* Adjoint velocity update, x axis: always transverse (fed by sigma_h, via
 * c11/c13) -- see FUNC(backward_vz_cell). */
DW_HOST_DEVICE void FUNC(backward_vx_cell)(
    DW_DTYPE const *__restrict const c11_shot,
    DW_DTYPE const *__restrict const c13_shot,
    DW_DTYPE const *__restrict const sigma_h,
    DW_DTYPE const *__restrict const sigma_v, DW_DTYPE *__restrict const vx,
    DW_DTYPE const *__restrict const m_vxx, DW_DTYPE const *__restrict const bx,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z,
#endif
    int64_t const y, int64_t const x, bool const pml_x, DW_DTYPE const rdx) {
  DW_DTYPE const w_sum = pml_x ? -DIFFXH1(VX_X_PML) : -DIFFXH1(VX_X);
  vx[i] += w_sum;
}

/* M_SIGMA_H_X PML memory-variable update for FUNC(backward_vx_cell) -- see
 * FUNC(backward_vz_cell_pml_mem)'s comment (same split, same reason: vx[i]
 * must be read post-fold). */
DW_HOST_DEVICE void FUNC(backward_vx_cell_pml_mem)(
    DW_DTYPE const *__restrict const vx,
    DW_DTYPE const *__restrict const m_sigma_h_x,
    DW_DTYPE *__restrict const m_sigma_h_xn, DW_DTYPE const buoyancy_x_shot,
    DW_DTYPE const *__restrict const axh, DW_DTYPE const dt, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny, int64_t const z,
#endif
    int64_t const nx, int64_t const y, int64_t const x, bool const pml_x) {
  if (pml_x) {
    m_sigma_h_xn[i] =
        buoyancy_x_shot * dt * P_AXH(0) * vx[i] + P_AXH(0) * m_sigma_h_x[i];
  }
}

/* ---- Backward: combined sigma_v/sigma_h adjoint update ---- */

/* grad_c11/grad_c13/grad_c33 accumulation (gated on step_ratio); m_vzz/m_vyy/
 * m_vxx PML-memory update (single buffer -- see the PML memory-variable
 * buffering note above); sigma_v += -div(buoyancy_z*vz) [axis],
 * sigma_h += -div(buoyancy_y*vy [3-D transverse] + buoyancy_x*vx) (transpose
 * of FUNC(forward_vz_cell)/vy_cell/vx_cell, see FUNC(forward_sigma_cell)).
 * `i`/`j` addressing matches FUNC(backward_vz_cell) above. `sigma_h`/
 * `sigma_v` here are read (for the grad accumulation, at the centre point
 * only) BEFORE being updated (matching the source order in acoustic_vti.c/
 * acoustic_vti.cu exactly: grad accumulation, then the PML-memory update,
 * then the sigma_v/sigma_h adjoint update itself). `vz`/`vy`/`vx` are
 * POINTERS here (addressed by `i`) because the SIGMA_V_Z(_PML)/SIGMA_H_Y
 * (_PML)/SIGMA_H_X(_PML) macros read them with a finite-difference neighbour
 * offset; `buoyancy_z_shot`/`buoyancy_y_shot`/`buoyancy_x_shot` are POINTERS
 * here (addressed by `j`) for the same reason. `m_sigma_v_z`/`m_sigma_h_y`/
 * `m_sigma_v_y`/`m_sigma_h_x` are the OLD (pre-this-step) stress-derivative
 * C-PML memories -- the SAME buffer FUNC(backward_vz_cell)/vy_cell/vx_cell
 * read (via the M_SIGMA_V_Z/M_SIGMA_H_Y/M_SIGMA_V_Y/M_SIGMA_H_X macros inside
 * SIGMA_V_Z_PML/SIGMA_H_Y_PML/SIGMA_V_Y_PML/SIGMA_H_X_PML), read only here
 * (never written -- the "n" buffer those functions write is not touched by
 * this function).
 */
DW_HOST_DEVICE void FUNC(backward_sigma_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
    DW_DTYPE const *__restrict const buoyancy_z_shot,
#endif
    DW_DTYPE const *__restrict const vy,
    DW_DTYPE const *__restrict const buoyancy_y_shot,
    DW_DTYPE const *__restrict const vx,
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    DW_DTYPE *__restrict const sigma_v, DW_DTYPE *__restrict const sigma_h,
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const m_vzz,
#endif
    DW_DTYPE *__restrict const m_vyy, DW_DTYPE *__restrict const m_vxx,
#if DW_NDIM == 3
    DW_DTYPE const *__restrict const m_sigma_v_z,
    DW_DTYPE const *__restrict const m_sigma_h_y,
#else
    DW_DTYPE const *__restrict const m_sigma_v_y,
#endif
    DW_DTYPE const *__restrict const m_sigma_h_x,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const dvzdz_store_1,
#endif
    DW_DTYPE const *__restrict const dvydy_store_1,
    DW_DTYPE const *__restrict const dvxdx_store_1, DW_DTYPE const c11_shot,
    DW_DTYPE const c13_shot, DW_DTYPE const c33_shot,
    DW_DTYPE *__restrict const grad_c11, DW_DTYPE *__restrict const grad_c13,
    DW_DTYPE *__restrict const grad_c33, DW_DTYPE const dt,
    int64_t const step_ratio, bool const c11_requires_grad,
    bool const c13_requires_grad, bool const c33_requires_grad, int64_t const i,
    int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const *__restrict const az,
    DW_DTYPE const *__restrict const bzh, DW_DTYPE const rdz,
#endif
    int64_t const y, bool const pml_y, DW_DTYPE const *__restrict const ay,
    DW_DTYPE const *__restrict const byh, DW_DTYPE const rdy, int64_t const x,
    bool const pml_x, DW_DTYPE const *__restrict const ax,
    DW_DTYPE const *__restrict const bxh, DW_DTYPE const rdx) {
#if DW_NDIM == 3
  if (c11_requires_grad) {
    grad_c11[i] += sigma_h[i] * (dvxdx_store_1[i] + dvydy_store_1[i]) *
                   (DW_DTYPE)step_ratio;
  }
  if (c13_requires_grad) {
    grad_c13[i] += (sigma_v[i] * (dvxdx_store_1[i] + dvydy_store_1[i]) +
                    sigma_h[i] * dvzdz_store_1[i]) *
                   (DW_DTYPE)step_ratio;
  }
  if (c33_requires_grad) {
    grad_c33[i] += sigma_v[i] * dvzdz_store_1[i] * (DW_DTYPE)step_ratio;
  }
#else
  // 2-D: axis = y. grad_c11 = s_h S_x; grad_c13 = s_v S_x + s_h S_y;
  // grad_c33 = s_v S_y.
  if (c11_requires_grad) {
    grad_c11[i] += sigma_h[i] * dvxdx_store_1[i] * (DW_DTYPE)step_ratio;
  }
  if (c13_requires_grad) {
    grad_c13[i] +=
        (sigma_v[i] * dvxdx_store_1[i] + sigma_h[i] * dvydy_store_1[i]) *
        (DW_DTYPE)step_ratio;
  }
  if (c33_requires_grad) {
    grad_c33[i] += sigma_v[i] * dvydy_store_1[i] * (DW_DTYPE)step_ratio;
  }
#endif

#if DW_NDIM >= 3
  if (pml_z) {
    m_vzz[i] = dt * P_AZ(0) * (c13_shot * sigma_h[i] + c33_shot * sigma_v[i]) +
               P_AZ(0) * m_vzz[i];
  }
#endif
  if (pml_y) {
    m_vyy[i] = dt * P_AY(0) *
#if DW_NDIM == 3
                   (c11_shot * sigma_h[i] + c13_shot * sigma_v[i]) +
#else
                   (c13_shot * sigma_h[i] + c33_shot * sigma_v[i]) +
#endif
               P_AY(0) * m_vyy[i];
  }
  if (pml_x) {
    m_vxx[i] = dt * P_AX(0) * (c11_shot * sigma_h[i] + c13_shot * sigma_v[i]) +
               P_AX(0) * m_vxx[i];
  }

#if DW_NDIM >= 3
  sigma_v[i] += pml_z ? -DIFFZ1(SIGMA_V_Z_PML) : -DIFFZ1(SIGMA_V_Z);
#else
  sigma_v[i] += pml_y ? -DIFFY1(SIGMA_V_Y_PML) : -DIFFY1(SIGMA_V_Y);
#endif
#if DW_NDIM >= 3
  sigma_h[i] += pml_y ? -DIFFY1(SIGMA_H_Y_PML) : -DIFFY1(SIGMA_H_Y);
#endif
  sigma_h[i] += pml_x ? -DIFFX1(SIGMA_H_X_PML) : -DIFFX1(SIGMA_H_X);
}

/* ========================= Guard-band gathers ========================= *
 *
 * The backward gather kernels above (FUNC(backward_v*_cell)/FUNC(backward_
 * sigma_cell)) only ever run over the grid's strict interior/PML region
 * (acoustic_vti.c's/acoustic_vti.cu's pml_bounds_* / pml_bounds_*h arrays
 * both start at FD_PAD and stop FD_PAD or FD_PAD-1 short of the far edge),
 * so they never deposit an adjoint contribution into the guard ring next to
 * a reflecting (non-PML) face. FUNC(fill_p_bc_adjoint)/FUNC(fill_v_bc_
 * adjoint) (acoustic_vti.c) then fold that guard ring into the interior --
 * but with nothing genuine ever written there, that fold silently discards
 * real gradient contributions. The functions below are guard-band-only
 * companions to the interior gathers: same physics (transpose of the
 * corresponding FUNC(forward_*_cell)), but evaluated with bounds-safe
 * stencil taps (DW_DIFF*_SAFE, which treat any tap landing outside [0,
 * nz)/[0, ny)/[0, nx) as exactly zero -- ported unchanged from elastic_vti_
 * kernels.h's identical mechanism, itself ported from elastic_kernels.h) so
 * they may be called at guard-ring cells, where a stencil can reach up to
 * FD_PAD cells past the physical array. They omit the grad_c11/grad_c13/
 * grad_c33 accumulation the interior functions perform (guard cells are not
 * independent model-gradient locations -- the model itself is only
 * mirror-padded, never an independent DOF there) and, for the velocity
 * gathers, the buoyancy-gradient accumulation and M_SIGMA_V_Z/M_SIGMA_H_Y/
 * M_SIGMA_V_Y/M_SIGMA_H_X PML memory-variable write the interior functions
 * used to perform too (moved to a separate post-fold pass -- see
 * acoustic_vti.c's/acoustic_vti.cu's driver-side comment at its new call
 * site -- since a pre-fold read of a near-face interior cell is stale once
 * the fold changes it; guard cells never need this write at all, since they
 * are never inside a PML region -- see FUNC(dw_is_pml_region) below).
 *
 * Unlike elastic(_vti), each acoustic_vti velocity component is fed by
 * exactly one stress-derivative term along its OWN axis only (no shear
 * cross-terms survive the V_S = 0 reduction -- see this header's own
 * module comment and acoustic_vti.c's header), so FUNC(backward_vz_cell)/
 * vy_cell/vx_cell each only need a guard-band extension on their own axis,
 * not on transverse axes too. sigma_v is imaged only on the axis face and
 * sigma_h only on the transverse faces (see acoustic_vti.c's module comment
 * above FUNC(fill_p_bc_line)), so their guard-band companions are likewise
 * split per own-axis (sigma_v) or per-transverse-axis (sigma_h, one
 * function per transverse axis, mirroring FUNC(backward_sigmazz_z_cell_
 * guard)-style per-field-per-axis splitting in elastic_vti_kernels.h).
 *
 * acoustic_vti.c's/acoustic_vti.cu's new guard-band driver loops call these
 * over exactly the *new* cells a field's boundary fill touches beyond the
 * existing interior/PML loop -- see FUNC(dw_compute_ext_range) below -- so
 * double-accumulation against the interior loop's own cells never occurs.
 */

/* Guard-extension bookkeeping for one field's use of one axis -- see
 * elastic_vti_kernels.h's identical FUNC(dw_ext_range)/FUNC(dw_compute_ext_
 * range) for the full derivation (this is a verbatim, FUNC-namespaced
 * port: the bookkeeping is pure grid-indexing arithmetic, independent of
 * which propagator's fields it is applied to). `n` is that axis's full
 * length (nz/ny/nx); `bc_lo`/`bc_hi` are that axis's two face codes (0 =
 * pml, 1 = free, 2 = rigid, 3 = periodic); `is_half` selects the half-grid
 * convention (velocity's own axis) vs full-grid (sigma_v/sigma_h). */
typedef struct {
  int64_t target_lo, target_hi;
  int64_t std_lo, std_hi;
  bool has_exclude;
  int64_t exclude_idx;
} FUNC(dw_ext_range);

DW_HOST_DEVICE FUNC(dw_ext_range)
    FUNC(dw_compute_ext_range)(int64_t const n, int64_t const bc_lo,
                               int64_t const bc_hi, bool const is_half) {
  FUNC(dw_ext_range) r;
  r.std_lo = FD_PAD;
  r.std_hi = is_half ? (n - FD_PAD) : (n - FD_PAD + 1);
  r.target_lo = bc_lo != 0 ? 0 : r.std_lo;
  r.target_hi = bc_hi != 0 ? n : r.std_hi;
  r.exclude_idx = n - FD_PAD;
  r.has_exclude = is_half && (bc_hi == 1 || bc_hi == 2);
  return r;
}

/* True iff coordinate `c` is in this field-axis's boundary-fill-aware
 * target set (see FUNC(dw_compute_ext_range)'s comment). */
DW_HOST_DEVICE bool FUNC(dw_in_target)(FUNC(dw_ext_range) const r,
                                       int64_t const c) {
  if (c < r.target_lo || c >= r.target_hi) return false;
  if (r.has_exclude && c == r.exclude_idx) return false;
  return true;
}

/* True iff coordinate `c` is in this field-axis's existing (unextended)
 * interior/PML loop range -- i.e. already handled by the interior gather. */
DW_HOST_DEVICE bool FUNC(dw_in_std)(FUNC(dw_ext_range) const r,
                                    int64_t const c) {
  return c >= r.std_lo && c < r.std_hi;
}

/* Whether spatial coordinate `c` on one axis is inside that axis's PML
 * sub-region -- see elastic_vti_kernels.h's identical FUNC(dw_is_pml_
 * region) for the full derivation. */
DW_HOST_DEVICE bool FUNC(dw_is_pml_region)(int64_t const c,
                                           int64_t const pml_lo,
                                           int64_t const pml_hi,
                                           int64_t const std_hi) {
  return (c >= FD_PAD && c < pml_lo) || (c >= pml_hi && c < std_hi);
}

#if DW_ACCURACY == 2
#define DW_FD_WEIGHT0 ((DW_DTYPE)1.0)
#elif DW_ACCURACY == 4
#define DW_FD_WEIGHT0 ((DW_DTYPE)(9.0 / 8.0))
#define DW_FD_WEIGHT1 ((DW_DTYPE)(-1.0 / 24.0))
#elif DW_ACCURACY == 6
#define DW_FD_WEIGHT0 ((DW_DTYPE)(75.0 / 64.0))
#define DW_FD_WEIGHT1 ((DW_DTYPE)(-25.0 / 384.0))
#define DW_FD_WEIGHT2 ((DW_DTYPE)(3.0 / 640.0))
#elif DW_ACCURACY == 8
#define DW_FD_WEIGHT0 ((DW_DTYPE)(1225.0 / 1024.0))
#define DW_FD_WEIGHT1 ((DW_DTYPE)(-245.0 / 3072.0))
#define DW_FD_WEIGHT2 ((DW_DTYPE)(49.0 / 5120.0))
#define DW_FD_WEIGHT3 ((DW_DTYPE)(-5.0 / 7168.0))
#endif

/* Bounds-safe single-tap read of compound field-access macro `a` (e.g.
 * VZ_Z, SIGMA_H_X): `a` itself (which may read model/wavefield arrays at
 * that offset, all at once, via the ND_INDEX-based macros above) is only
 * evaluated -- so only ever dereferenced -- when the offset cell is inside
 * the array; a tap landing outside contributes exactly zero. */
#if DW_NDIM >= 3
#define DW_SAFE_TAP_Z(a, d) \
  ((z + (d) >= 0 && z + (d) < nz) ? a(d, 0, 0) : (DW_DTYPE)0)
#endif
#if DW_NDIM >= 2
#define DW_SAFE_TAP_Y(a, d) \
  ((y + (d) >= 0 && y + (d) < ny) ? a(0, d, 0) : (DW_DTYPE)0)
#endif
#define DW_SAFE_TAP_X(a, d) \
  ((x + (d) >= 0 && x + (d) < nx) ? a(0, 0, d) : (DW_DTYPE)0)

/* Bounds-safe weighted sum reproducing DIFF*1/DIFF*H1's own structure
 * (staggered_grid.h) -- see elastic_vti_kernels.h's identical mechanism for
 * the full derivation. acoustic_vti_kernels.h may not modify
 * staggered_grid.h (out of scope), hence the duplication here. */
#if FD_PAD == 1
#define DW_SAFE_SUM(TAP, a, s) \
  (DW_FD_WEIGHT0 * (TAP(a, (0 + (s))) - TAP(a, (-1 + (s)))))
#elif FD_PAD == 2
#define DW_SAFE_SUM(TAP, a, s)                                \
  (DW_FD_WEIGHT0 * (TAP(a, (0 + (s))) - TAP(a, (-1 + (s)))) + \
   DW_FD_WEIGHT1 * (TAP(a, (1 + (s))) - TAP(a, (-2 + (s)))))
#elif FD_PAD == 3
#define DW_SAFE_SUM(TAP, a, s)                                \
  (DW_FD_WEIGHT0 * (TAP(a, (0 + (s))) - TAP(a, (-1 + (s)))) + \
   DW_FD_WEIGHT1 * (TAP(a, (1 + (s))) - TAP(a, (-2 + (s)))) + \
   DW_FD_WEIGHT2 * (TAP(a, (2 + (s))) - TAP(a, (-3 + (s)))))
#elif FD_PAD == 4
#define DW_SAFE_SUM(TAP, a, s)                                \
  (DW_FD_WEIGHT0 * (TAP(a, (0 + (s))) - TAP(a, (-1 + (s)))) + \
   DW_FD_WEIGHT1 * (TAP(a, (1 + (s))) - TAP(a, (-2 + (s)))) + \
   DW_FD_WEIGHT2 * (TAP(a, (2 + (s))) - TAP(a, (-3 + (s)))) + \
   DW_FD_WEIGHT3 * (TAP(a, (3 + (s))) - TAP(a, (-4 + (s)))))
#endif

#if DW_NDIM >= 3
#define DW_DIFFZ1_SAFE(a) (DW_SAFE_SUM(DW_SAFE_TAP_Z, a, 0) * rdz)
#define DW_DIFFZH1_SAFE(a) (DW_SAFE_SUM(DW_SAFE_TAP_Z, a, 1) * rdz)
#endif
#if DW_NDIM >= 2
#define DW_DIFFY1_SAFE(a) (DW_SAFE_SUM(DW_SAFE_TAP_Y, a, 0) * rdy)
#define DW_DIFFYH1_SAFE(a) (DW_SAFE_SUM(DW_SAFE_TAP_Y, a, 1) * rdy)
#endif
#define DW_DIFFX1_SAFE(a) (DW_SAFE_SUM(DW_SAFE_TAP_X, a, 0) * rdx)
#define DW_DIFFXH1_SAFE(a) (DW_SAFE_SUM(DW_SAFE_TAP_X, a, 1) * rdx)

#if DW_NDIM >= 3
/* Guard-band companion to FUNC(backward_vz_cell)'s gather (no PML branch,
 * no M_SIGMA_V_Z write, no buoyancy-grad accumulation -- see the module
 * comment above). `nz` is required here (unlike FUNC(backward_vz_cell),
 * which never needs it for flat indexing) purely for DW_DIFFZH1_SAFE's
 * bounds check. */
DW_HOST_DEVICE void FUNC(backward_vz_cell_guard)(
    DW_DTYPE const *__restrict const c13_shot,
    DW_DTYPE const *__restrict const c33_shot,
    DW_DTYPE const *__restrict const sigma_h,
    DW_DTYPE const *__restrict const sigma_v, DW_DTYPE *__restrict const vz,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const nz,
    int64_t const ny, int64_t const nx, int64_t const z, int64_t const y,
    int64_t const x, DW_DTYPE const rdz) {
  vz[i] += -DW_DIFFZH1_SAFE(VZ_Z);
}
#endif

/* Guard-band companion to FUNC(backward_vy_cell)'s gather (see FUNC(
 * backward_vz_cell_guard)). `ny` is required unconditionally here (unlike
 * FUNC(backward_vy_cell), which only needs it for ND_INDEX when DW_NDIM==3)
 * purely for DW_DIFFYH1_SAFE's bounds check. */
DW_HOST_DEVICE void FUNC(backward_vy_cell_guard)(
#if DW_NDIM == 3
    DW_DTYPE const *__restrict const c11_shot,
    DW_DTYPE const *__restrict const c13_shot,
#else
    DW_DTYPE const *__restrict const c13_shot,
    DW_DTYPE const *__restrict const c33_shot,
#endif
    DW_DTYPE const *__restrict const sigma_h,
    DW_DTYPE const *__restrict const sigma_v, DW_DTYPE *__restrict const vy,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const nz,
#endif
    int64_t const ny, int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z,
#endif
    int64_t const y, int64_t const x, DW_DTYPE const rdy) {
  vy[i] += -DW_DIFFYH1_SAFE(VY_Y);
}

/* Guard-band companion to FUNC(backward_vx_cell)'s gather (see FUNC(
 * backward_vz_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_vx_cell_guard)(
    DW_DTYPE const *__restrict const c11_shot,
    DW_DTYPE const *__restrict const c13_shot,
    DW_DTYPE const *__restrict const sigma_h,
    DW_DTYPE const *__restrict const sigma_v, DW_DTYPE *__restrict const vx,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const nz,
#endif
    int64_t const ny, int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z,
#endif
    int64_t const y, int64_t const x, DW_DTYPE const rdx) {
  vx[i] += -DW_DIFFXH1_SAFE(VX_X);
}

#if DW_NDIM >= 3
/* Guard-band companion to the sigma_v sub-update inside FUNC(backward_
 * sigma_cell): sigma_v's only term is its own (z) axis (SIGMA_V_Z) -- a
 * guard cell here (which by construction only exists on an axis with a
 * non-PML face) is never inside a PML region on that same axis, so the
 * M_VZZ PML-memory write the interior function performs is unreachable and
 * intentionally omitted, and grad_c13/grad_c33 (also computed there, using
 * sigma_v's value at the *old* cell, not the new guard cell) is left
 * entirely to the interior function -- see the module comment above. */
DW_HOST_DEVICE void FUNC(backward_sigma_v_z_cell_guard)(
    DW_DTYPE const *__restrict const buoyancy_z_shot,
    DW_DTYPE const *__restrict const vz, DW_DTYPE *__restrict const sigma_v,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const nz,
    int64_t const ny, int64_t const nx, int64_t const z, DW_DTYPE const rdz) {
  sigma_v[i] += -DW_DIFFZ1_SAFE(SIGMA_V_Z);
}

/* Guard-band companion to the sigma_h sub-update inside FUNC(backward_
 * sigma_cell) for its y (3-D transverse) term -- see FUNC(backward_sigma_
 * v_z_cell_guard). */
DW_HOST_DEVICE void FUNC(backward_sigma_h_y_cell_guard)(
    DW_DTYPE const *__restrict const buoyancy_y_shot,
    DW_DTYPE const *__restrict const vy, DW_DTYPE *__restrict const sigma_h,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const ny,
    int64_t const nx, int64_t const y, DW_DTYPE const rdy) {
  sigma_h[i] += -DW_DIFFY1_SAFE(SIGMA_H_Y);
}
#else
/* Guard-band companion to the sigma_v sub-update inside FUNC(backward_
 * sigma_cell) (2-D: axis = y) -- see FUNC(backward_sigma_v_z_cell_guard). */
DW_HOST_DEVICE void FUNC(backward_sigma_v_y_cell_guard)(
    DW_DTYPE const *__restrict const buoyancy_y_shot,
    DW_DTYPE const *__restrict const vy, DW_DTYPE *__restrict const sigma_v,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const ny,
    int64_t const nx, int64_t const y, DW_DTYPE const rdy) {
  sigma_v[i] += -DW_DIFFY1_SAFE(SIGMA_V_Y);
}
#endif

/* Guard-band companion to the sigma_h sub-update inside FUNC(backward_
 * sigma_cell) for its x (always transverse) term -- see FUNC(backward_
 * sigma_v_z_cell_guard). */
DW_HOST_DEVICE void FUNC(backward_sigma_h_x_cell_guard)(
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    DW_DTYPE const *__restrict const vx, DW_DTYPE *__restrict const sigma_h,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx, int64_t const x, DW_DTYPE const rdx) {
  sigma_h[i] += -DW_DIFFX1_SAFE(SIGMA_H_X);
}

#endif /* DW_ACOUSTIC_VTI_KERNELS_H */
