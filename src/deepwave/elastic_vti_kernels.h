#ifndef DW_ELASTIC_VTI_KERNELS_H
#define DW_ELASTIC_VTI_KERNELS_H

/*
 * elastic_vti_kernels.h
 *
 * Single-source per-grid-point stencil math for the elastic VTI (vertically
 * transverse isotropic, variable density, staggered grid, velocity-stress)
 * wave equation propagator (see elastic_vti.c for the physics/PML/staggered-
 * grid-layout description, and elastic_kernels.h for the isotropic-elastic
 * precedent this header follows). This header is included by BOTH
 * elastic_vti.c (compiled by gcc) and elastic_vti.cu (compiled by nvcc),
 * after each has defined the FUNC()/CAT() name-mangling macros (and, for
 * elastic_vti, the DW_MPML-dependent ELASTIC_VARIANT()) and before either
 * uses the ND_INDEX/P_A(*)/P_B(*)/VZ/VY/VX/SIGMA(*)/C11/C12/C13/C33/C_XY/
 * C55_ZX/C55_ZY/BUOYANCY_(*)/M_(*)/... family below.
 *
 * The velocity update is IDENTICAL to the isotropic elastic propagator's --
 * VTI anisotropy only changes the stress-strain (constitutive) relation, not
 * how velocity is updated from stress divergence -- so FUNC(forward_vz_cell)/
 * forward_vy_cell/forward_vx_cell and FUNC(backward_sigmaii_cell)'s
 * sigma-update tail (SIGMAZZ_Z/SIGMAYY_Y/SIGMAXX_X) are structurally the same
 * shape as elastic_kernels.h's; only the model parameters they close over
 * differ (buoyancy, unchanged).
 *
 * The VTI constitutive relation does NOT reduce to a single lambda*div(v)
 * term shared by all three normal stresses (unlike isotropic elastic), so
 * FUNC(forward_sigmaii_cell) computes dvzdz/dvydy/dvxdx (with the same PML
 * recursion machinery as elastic's) and then combines them into
 * sigmazz/sigmayy/sigmaxx with THREE DIFFERENT per-component weightings
 * (c11/c12/c13/c33, symmetric stiffness matrix; see elastic_vti.c's top
 * comment for the full relation), rather than a single shared w_sum. Its
 * adjoint, FUNC(backward_sigmaii_cell), likewise accumulates FOUR separate
 * model gradients (grad_c11, grad_c12 [3-D only], grad_c13, grad_c33) instead
 * of elastic's two (grad_lamb, grad_mu). The shear stress/velocity updates
 * (sigmaxy/sigmaxz/sigmayz and their adjoints) are structurally identical to
 * elastic's, one independent shear stiffness (c_xy = c66_yx in 3-D / c55_yx
 * in 2-D, c55_zx, c55_zy) replacing elastic's mu_yx/mu_zx/mu_zy.
 *
 * This gives the same 7 forward + 7 backward = 14 per-grid-point functions
 * as elastic_kernels.h: FUNC(forward_vz_cell)/forward_vy_cell/
 * forward_vx_cell/forward_sigmaii_cell/forward_sigmaxy_cell/
 * forward_sigmaxz_cell/forward_sigmayz_cell, and the corresponding seven
 * FUNC(backward_*_cell) adjoints.
 *
 * Addressing convention (matches elastic_kernels.h exactly): vz/vy/vx/
 * sigma(*)/m_v(**)/m_sigma(**)/dv(*)d(*)_store_1/grad_buoyancy_(*)/... are
 * addressed by `i` (the CPU driver's per-shot-local flat index, which is also
 * valid as the CUDA driver's global shot-spanning index since both drivers
 * pass already-appropriately-offset pointers); c11_shot/c12_shot/c13_shot/
 * c33_shot/c_xy_shot/c55_zx_shot/c55_zy_shot/buoyancy_*_shot are addressed by
 * `j` (the per-shot-local index) wherever a POINTER is taken (used only by
 * the backward functions, via the C11/C12/C13/C33/C_XY/C55_ZX/C55_ZY/
 * BUOYANCY_Z/Y/X macros), because the CUDA driver's model pointers are
 * shot-local (offset by shot_idx*shot_numel, not `i`) while the wavefield
 * arrays span all shots. On the CPU driver, wavefields/model arrays are ALL
 * already offset to the current shot, so `i` and `j` are the same value there
 * and the CPU driver simply passes `i` for both. The forward functions need
 * no `j`: they only ever read the *centre* value of the relevant model field
 * (no neighbour offset), so the caller passes that value directly, by value.
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
#define ND_INDEX(i, dz, dy, dx) ((i) + (dz) * ny * nx + (dy) * nx + (dx))
#elif DW_NDIM == 2
#define ND_INDEX(i, dz, dy, dx) ((i) + (dy) * nx + (dx))
#else /* DW_NDIM == 1 */
#define ND_INDEX(i, dz, dy, dx) (i + (dx))
#endif

/* PML profile access. The standard (non-M-PML) profiles are 1-D per axis, so
 * a given axis's profile is indexed by that axis's own coordinate only
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
#if DW_NDIM >= 3
#define SIGMAZZ(dz, dy, dx) sigmazz[ND_INDEX(i, dz, dy, dx)]
#elif DW_NDIM == 2
#define SIGMAZZ(dz, dy, dx) 0
#endif
#if DW_NDIM >= 2
#define SIGMAYY(dz, dy, dx) sigmayy[ND_INDEX(i, dz, dy, dx)]
#endif
#define SIGMAXX(dz, dy, dx) sigmaxx[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 2
#define SIGMAXY(dz, dy, dx) sigmaxy[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define SIGMAXZ(dz, dy, dx) sigmaxz[ND_INDEX(i, dz, dy, dx)]
#define SIGMAYZ(dz, dy, dx) sigmayz[ND_INDEX(i, dz, dy, dx)]
#endif

// Models, addressed by the shot-local index j (see the addressing-convention
// note above) -- used only by the backward (adjoint) functions, via the
// C11/C12/C13/C33/C_XY/C55_ZX/C55_ZY/BUOYANCY_Z/Y/X macros below.
#define C11(dz, dy, dx) c11_shot[ND_INDEX(j, dz, dy, dx)]
#define C13(dz, dy, dx) c13_shot[ND_INDEX(j, dz, dy, dx)]
#define C33(dz, dy, dx) c33_shot[ND_INDEX(j, dz, dy, dx)]
#if DW_NDIM >= 3
#define C12(dz, dy, dx) c12_shot[ND_INDEX(j, dz, dy, dx)]
#endif
/* Shear stiffness at the sigma_xy node: c66 in 3-D, c55 in 2-D. */
#if DW_NDIM >= 2
#define C_XY(dz, dy, dx) c_xy_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define C55_ZX(dz, dy, dx) c55_zx_shot[ND_INDEX(j, dz, dy, dx)]
#define C55_ZY(dz, dy, dx) c55_zy_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define BUOYANCY_Z(dz, dy, dx) buoyancy_z_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#if DW_NDIM >= 2
#define BUOYANCY_Y(dz, dy, dx) buoyancy_y_shot[ND_INDEX(j, dz, dy, dx)]
#endif
#define BUOYANCY_X(dz, dy, dx) buoyancy_x_shot[ND_INDEX(j, dz, dy, dx)]

// PML memory variables for the velocity equations (the "M_SIGMA**" family --
// see elastic_kernels.h's ping-pong-buffering note, which applies unchanged
// here): written (centre point only) by FUNC(backward_v*_cell), read (with a
// neighbour offset, via these macros) by FUNC(backward_sigma*_cell).
#if DW_NDIM >= 2
#define M_SIGMAXYY(dz, dy, dx) m_sigmaxyy[ND_INDEX(i, dz, dy, dx)]
#endif
#define M_SIGMAXXX(dz, dy, dx) m_sigmaxxx[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 3
#define M_SIGMAXZZ(dz, dy, dx) m_sigmaxzz[ND_INDEX(i, dz, dy, dx)]
#endif

#if DW_NDIM >= 2
#define M_SIGMAXYX(dz, dy, dx) m_sigmaxyx[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMAYYY(dz, dy, dx) m_sigmayyy[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define M_SIGMAYZZ(dz, dy, dx) m_sigmayzz[ND_INDEX(i, dz, dy, dx)]
#endif

#if DW_NDIM >= 3
#define M_SIGMAXZX(dz, dy, dx) m_sigmaxzx[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMAYZY(dz, dy, dx) m_sigmayzy[ND_INDEX(i, dz, dy, dx)]
#define M_SIGMAZZZ(dz, dy, dx) m_sigmazzz[ND_INDEX(i, dz, dy, dx)]
#endif

// PML memory variables for the stress equations (the "M_V**" family -- see
// elastic_kernels.h's ping-pong-buffering note): read (with a neighbour
// offset, via these macros) by FUNC(backward_v*_cell), written (centre point
// only) by FUNC(backward_sigma*_cell).
#if DW_NDIM >= 2
#define M_VXY(dz, dy, dx) m_vxy[ND_INDEX(i, dz, dy, dx)]
#define M_VYX(dz, dy, dx) m_vyx[ND_INDEX(i, dz, dy, dx)]
#endif
#define M_VXX(dz, dy, dx) m_vxx[ND_INDEX(i, dz, dy, dx)]
#if DW_NDIM >= 2
#define M_VYY(dz, dy, dx) m_vyy[ND_INDEX(i, dz, dy, dx)]
#endif
#if DW_NDIM >= 3
#define M_VZZ(dz, dy, dx) m_vzz[ND_INDEX(i, dz, dy, dx)]
#define M_VXZ(dz, dy, dx) m_vxz[ND_INDEX(i, dz, dy, dx)]
#define M_VZX(dz, dy, dx) m_vzx[ND_INDEX(i, dz, dy, dx)]
#define M_VYZ(dz, dy, dx) m_vyz[ND_INDEX(i, dz, dy, dx)]
#define M_VZY(dz, dy, dx) m_vzy[ND_INDEX(i, dz, dy, dx)]
#endif

// Adjoint-term macros used by the backward (adjoint) velocity update
// (FUNC(backward_v*_cell)): the transpose of the forward VTI stress
// equations. The stiffness matrix is symmetric, so the transpose reuses the
// same Cij; the coefficient of SIGMA?? is the (??, this-derivative) entry of
// the VTI matrix (see elastic_vti.c's top comment for the full relation).
#if DW_NDIM >= 3
// dvzdz (Voigt "3") couples to sigmaxx/sigmayy via c13 and to sigmazz via c33.
#define VZ_Z(dz, dy, dx)                         \
  (dt * (C13(dz, dy, dx) * SIGMAXX(dz, dy, dx) + \
         C13(dz, dy, dx) * SIGMAYY(dz, dy, dx) + \
         C33(dz, dy, dx) * SIGMAZZ(dz, dy, dx)))
#define VZ_Y(dz, dy, dx) (dt * C55_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx))
#define VZ_X(dz, dy, dx) (dt * C55_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx))
#define VZ_Z_PML(dz, dy, dx)                     \
  (dt * (1 + P_BZ(dz)) *                         \
       (C13(dz, dy, dx) * SIGMAXX(dz, dy, dx) +  \
        C13(dz, dy, dx) * SIGMAYY(dz, dy, dx) +  \
        C33(dz, dy, dx) * SIGMAZZ(dz, dy, dx)) + \
   P_BZ(dz) * M_VZZ(dz, dy, dx))
#define VZ_Y_PML(dz, dy, dx)                                         \
  (dt * (1 + P_BYH(dy)) * C55_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx) + \
   P_BYH(dy) * M_VZY(dz, dy, dx))
#define VZ_X_PML(dz, dy, dx)                                         \
  (dt * (1 + P_BXH(dx)) * C55_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx) + \
   P_BXH(dx) * M_VZX(dz, dy, dx))
#endif

#if DW_NDIM >= 2
#if DW_NDIM >= 3
#define VY_Z(dz, dy, dx) (dt * C55_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx))
#endif
#if DW_NDIM == 2
// 2-D: axis = y ("3"). dvydy couples to sigmaxx via c13, sigmayy via c33.
#define VY_Y(dz, dy, dx)                         \
  (dt * (C13(dz, dy, dx) * SIGMAXX(dz, dy, dx) + \
         C33(dz, dy, dx) * SIGMAYY(dz, dy, dx)))
#else
// 3-D: dvydy ("2") couples to sigmaxx via c12, sigmayy via c11, sigmazz c13.
#define VY_Y(dz, dy, dx)                         \
  (dt * (C12(dz, dy, dx) * SIGMAXX(dz, dy, dx) + \
         C11(dz, dy, dx) * SIGMAYY(dz, dy, dx) + \
         C13(dz, dy, dx) * SIGMAZZ(dz, dy, dx)))
#endif
#define VY_X(dz, dy, dx) (dt * C_XY(dz, dy, dx) * SIGMAXY(dz, dy, dx))
#if DW_NDIM >= 3
#define VY_Z_PML(dz, dy, dx)                                         \
  (dt * (1 + P_BZH(dz)) * C55_ZY(dz, dy, dx) * SIGMAYZ(dz, dy, dx) + \
   P_BZH(dz) * M_VYZ(dz, dy, dx))
#endif
#if DW_NDIM == 2
#define VY_Y_PML(dz, dy, dx)                     \
  (dt * (1 + P_BY(dy)) *                         \
       (C13(dz, dy, dx) * SIGMAXX(dz, dy, dx) +  \
        C33(dz, dy, dx) * SIGMAYY(dz, dy, dx)) + \
   P_BY(dy) * M_VYY(dz, dy, dx))
#else
#define VY_Y_PML(dz, dy, dx)                     \
  (dt * (1 + P_BY(dy)) *                         \
       (C12(dz, dy, dx) * SIGMAXX(dz, dy, dx) +  \
        C11(dz, dy, dx) * SIGMAYY(dz, dy, dx) +  \
        C13(dz, dy, dx) * SIGMAZZ(dz, dy, dx)) + \
   P_BY(dy) * M_VYY(dz, dy, dx))
#endif
#define VY_X_PML(dz, dy, dx)                                       \
  (dt * (1 + P_BXH(dx)) * C_XY(dz, dy, dx) * SIGMAXY(dz, dy, dx) + \
   P_BXH(dx) * M_VYX(dz, dy, dx))
#endif

#if DW_NDIM >= 3
#define VX_Z(dz, dy, dx) (dt * C55_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define VX_Y(dz, dy, dx) (dt * C_XY(dz, dy, dx) * SIGMAXY(dz, dy, dx))
#endif
#if DW_NDIM == 1
#define VX_X(dz, dy, dx) (dt * C11(0, 0, dx) * SIGMAXX(0, 0, dx))
#elif DW_NDIM == 2
// 2-D: dvxdx ("1") couples to sigmaxx via c11, sigmayy via c13.
#define VX_X(dz, dy, dx)                         \
  (dt * (C11(dz, dy, dx) * SIGMAXX(dz, dy, dx) + \
         C13(dz, dy, dx) * SIGMAYY(dz, dy, dx)))
#else
// 3-D: dvxdx ("1") couples to sigmaxx via c11, sigmayy via c12, sigmazz c13.
#define VX_X(dz, dy, dx)                         \
  (dt * (C11(dz, dy, dx) * SIGMAXX(dz, dy, dx) + \
         C12(dz, dy, dx) * SIGMAYY(dz, dy, dx) + \
         C13(dz, dy, dx) * SIGMAZZ(dz, dy, dx)))
#endif
#if DW_NDIM >= 3
#define VX_Z_PML(dz, dy, dx)                                         \
  (dt * (1 + P_BZH(dz)) * C55_ZX(dz, dy, dx) * SIGMAXZ(dz, dy, dx) + \
   P_BZH(dz) * M_VXZ(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define VX_Y_PML(dz, dy, dx)                                       \
  (dt * (1 + P_BYH(dy)) * C_XY(dz, dy, dx) * SIGMAXY(dz, dy, dx) + \
   P_BYH(dy) * M_VXY(dz, dy, dx))
#endif
#if DW_NDIM == 1
#define VX_X_PML(dz, dy, dx)                                   \
  (dt * (1 + P_BX(dx)) * (C11(0, 0, dx) * SIGMAXX(0, 0, dx)) + \
   P_BX(dx) * M_VXX(0, 0, dx))
#elif DW_NDIM == 2
#define VX_X_PML(dz, dy, dx)                     \
  (dt * (1 + P_BX(dx)) *                         \
       (C11(dz, dy, dx) * SIGMAXX(dz, dy, dx) +  \
        C13(dz, dy, dx) * SIGMAYY(dz, dy, dx)) + \
   P_BX(dx) * M_VXX(dz, dy, dx))
#else
#define VX_X_PML(dz, dy, dx)                     \
  (dt * (1 + P_BX(dx)) *                         \
       (C11(dz, dy, dx) * SIGMAXX(dz, dy, dx) +  \
        C12(dz, dy, dx) * SIGMAYY(dz, dy, dx) +  \
        C13(dz, dy, dx) * SIGMAZZ(dz, dy, dx)) + \
   P_BX(dx) * M_VXX(dz, dy, dx))
#endif

// Adjoint-term macros used by the backward (adjoint) stress update
// (FUNC(backward_sigma*_cell)): the transpose of the forward velocity
// equations. Identical in shape to elastic_kernels.h's (no coefficient
// dependency -- buoyancy * velocity only).
#if DW_NDIM >= 3
#define SIGMAZZ_Z_PML(dz, dy, dx)                                   \
  ((1 + P_BZH(dz)) * dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx) + \
   P_BZH(dz) * M_SIGMAZZZ(dz, dy, dx))
#define SIGMAZZ_Z(dz, dy, dx) (dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))
#endif
#if DW_NDIM >= 2
#define SIGMAYY_Y_PML(dz, dy, dx)                                   \
  ((1 + P_BYH(dy)) * dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx) + \
   P_BYH(dy) * M_SIGMAYYY(dz, dy, dx))
#define SIGMAYY_Y(dz, dy, dx) (dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#endif
#define SIGMAXX_X_PML(dz, dy, dx)                                   \
  ((1 + P_BXH(dx)) * dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx) + \
   P_BXH(dx) * M_SIGMAXXX(dz, dy, dx))
#define SIGMAXX_X(dz, dy, dx) (dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))

#if DW_NDIM >= 2
#define SIGMAXY_Y_PML(dz, dy, dx)                                  \
  ((1 + P_BY(dy)) * dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx) + \
   P_BY(dy) * M_SIGMAXYY(dz, dy, dx))
#define SIGMAXY_Y(dz, dy, dx) (dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))
#define SIGMAXY_X_PML(dz, dy, dx)                                  \
  ((1 + P_BX(dx)) * dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx) + \
   P_BX(dx) * M_SIGMAXYX(dz, dy, dx))
#define SIGMAXY_X(dz, dy, dx) (dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#endif

#if DW_NDIM >= 3
#define SIGMAXZ_Z_PML(dz, dy, dx)                                  \
  ((1 + P_BZ(dz)) * dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx) + \
   P_BZ(dz) * M_SIGMAXZZ(dz, dy, dx))
#define SIGMAXZ_Z(dz, dy, dx) (dt * BUOYANCY_X(dz, dy, dx) * VX(dz, dy, dx))
#define SIGMAXZ_X_PML(dz, dy, dx)                                  \
  ((1 + P_BX(dx)) * dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx) + \
   P_BX(dx) * M_SIGMAXZX(dz, dy, dx))
#define SIGMAXZ_X(dz, dy, dx) (dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))

#define SIGMAYZ_Z_PML(dz, dy, dx)                                  \
  ((1 + P_BZ(dz)) * dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx) + \
   P_BZ(dz) * M_SIGMAYZZ(dz, dy, dx))
#define SIGMAYZ_Z(dz, dy, dx) (dt * BUOYANCY_Y(dz, dy, dx) * VY(dz, dy, dx))
#define SIGMAYZ_Y_PML(dz, dy, dx)                                  \
  ((1 + P_BY(dy)) * dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx) + \
   P_BY(dy) * M_SIGMAYZY(dz, dy, dx))
#define SIGMAYZ_Y(dz, dy, dx) (dt * BUOYANCY_Z(dz, dy, dx) * VZ(dz, dy, dx))
#endif

/* ============================= Forward ============================= */

#if DW_NDIM >= 3
/* vz += buoyancy_z * dt * (d(sigmazz)/dz + d(sigmayz)/dy + d(sigmaxz)/dx),
 * each term accumulating its axis's PML memory variable when inside that
 * axis's PML region. Identical to elastic_kernels.h's FUNC(forward_vz_cell)
 * (the velocity update does not depend on the (an)isotropy of the
 * constitutive relation). `i` addresses sigmazz/sigmayz/sigmaxz/vz/
 * m_sigmazzz/m_sigmayzy/m_sigmaxzx/dvzdbuoyancy_store_1 (per-shot-local on
 * the CPU driver, global on the CUDA driver); `buoyancy_z_shot` is the
 * (already shot-selected) buoyancy VALUE at this point; `pml_z`/`pml_y`/
 * `pml_x` are true when this point is inside that axis's PML region.
 */
DW_HOST_DEVICE void FUNC(forward_vz_cell)(
    DW_DTYPE const* __restrict const sigmazz,
    DW_DTYPE const* __restrict const sigmayz,
    DW_DTYPE const* __restrict const sigmaxz, DW_DTYPE* __restrict const vz,
    DW_DTYPE* __restrict const m_sigmazzz,
    DW_DTYPE* __restrict const m_sigmayzy,
    DW_DTYPE* __restrict const m_sigmaxzx,
    DW_DTYPE* __restrict const dvzdbuoyancy_store_1,
    DW_DTYPE const buoyancy_z_shot, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bzh, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const dt,
    bool const buoyancy_requires_grad, int64_t const i, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, bool const pml_y, bool const pml_x, DW_DTYPE const rdz,
    DW_DTYPE const rdy, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;
  if (!pml_z) {
    w_sum += DIFFZH1(SIGMAZZ);
  } else {
    DW_DTYPE d3 = DIFFZH1(SIGMAZZ);
    m_sigmazzz[i] = P_AZH(0) * m_sigmazzz[i] + P_BZH(0) * d3;
    d3 += m_sigmazzz[i];
    w_sum += d3;
  }
  if (!pml_y) {
    w_sum += DIFFY1(SIGMAYZ);
  } else {
    DW_DTYPE d2 = DIFFY1(SIGMAYZ);
    m_sigmayzy[i] = P_AY(0) * m_sigmayzy[i] + P_BY(0) * d2;
    d2 += m_sigmayzy[i];
    w_sum += d2;
  }
  if (!pml_x) {
    w_sum += DIFFX1(SIGMAXZ);
  } else {
    DW_DTYPE d1 = DIFFX1(SIGMAXZ);
    m_sigmaxzx[i] = P_AX(0) * m_sigmaxzx[i] + P_BX(0) * d1;
    d1 += m_sigmaxzx[i];
    w_sum += d1;
  }

  if (buoyancy_requires_grad) {
    dvzdbuoyancy_store_1[i] = dt * w_sum;
  }
  vz[i] += buoyancy_z_shot * dt * w_sum;
}
#endif

#if DW_NDIM >= 2
/* vy += buoyancy_y * dt * ((d(sigmayz)/dz, if DW_NDIM>=3) + d(sigmayy)/dy +
 * d(sigmaxy)/dx) (see FUNC(forward_vz_cell)). */
DW_HOST_DEVICE void FUNC(forward_vy_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmayz,
#endif
    DW_DTYPE const* __restrict const sigmayy,
    DW_DTYPE const* __restrict const sigmaxy, DW_DTYPE* __restrict const vy,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_sigmayzz,
#endif
    DW_DTYPE* __restrict const m_sigmayyy,
    DW_DTYPE* __restrict const m_sigmaxyx,
    DW_DTYPE* __restrict const dvydbuoyancy_store_1,
    DW_DTYPE const buoyancy_y_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
#endif
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const dt, bool const buoyancy_requires_grad, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const rdz,
#endif
    int64_t const y, int64_t const x, bool const pml_y, bool const pml_x,
    DW_DTYPE const rdy, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;

#if DW_NDIM >= 3
  if (!pml_z) {
    w_sum += DIFFZ1(SIGMAYZ);
  } else {
    DW_DTYPE d3 = DIFFZ1(SIGMAYZ);
    m_sigmayzz[i] = P_AZ(0) * m_sigmayzz[i] + P_BZ(0) * d3;
    d3 += m_sigmayzz[i];
    w_sum += d3;
  }
#endif
  if (!pml_y) {
    w_sum += DIFFYH1(SIGMAYY);
  } else {
    DW_DTYPE d2 = DIFFYH1(SIGMAYY);
    m_sigmayyy[i] = P_AYH(0) * m_sigmayyy[i] + P_BYH(0) * d2;
    d2 += m_sigmayyy[i];
    w_sum += d2;
  }
  if (!pml_x) {
    w_sum += DIFFX1(SIGMAXY);
  } else {
    DW_DTYPE d1 = DIFFX1(SIGMAXY);
    m_sigmaxyx[i] = P_AX(0) * m_sigmaxyx[i] + P_BX(0) * d1;
    d1 += m_sigmaxyx[i];
    w_sum += d1;
  }

  if (buoyancy_requires_grad) {
    dvydbuoyancy_store_1[i] = dt * w_sum;
  }
  vy[i] += buoyancy_y_shot * dt * w_sum;
}
#endif

/* vx += buoyancy_x * dt * ((d(sigmaxz)/dz, if DW_NDIM>=3) +
 * (d(sigmaxy)/dy, if DW_NDIM>=2) + d(sigmaxx)/dx) (see
 * FUNC(forward_vz_cell)). */
DW_HOST_DEVICE void FUNC(forward_vx_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmaxy,
#endif
    DW_DTYPE const* __restrict const sigmaxx, DW_DTYPE* __restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const m_sigmaxzz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const m_sigmaxyy,
#endif
    DW_DTYPE* __restrict const m_sigmaxxx,
    DW_DTYPE* __restrict const dvxdbuoyancy_store_1,
    DW_DTYPE const buoyancy_x_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const bz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const by,
#endif
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, bool const buoyancy_requires_grad, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;

#if DW_NDIM >= 3
  if (!pml_z) {
    w_sum += DIFFZ1(SIGMAXZ);
  } else {
    DW_DTYPE d3 = DIFFZ1(SIGMAXZ);
    m_sigmaxzz[i] = P_AZ(0) * m_sigmaxzz[i] + P_BZ(0) * d3;
    d3 += m_sigmaxzz[i];
    w_sum += d3;
  }
#endif
#if DW_NDIM >= 2
  if (!pml_y) {
    w_sum += DIFFY1(SIGMAXY);
  } else {
    DW_DTYPE d2 = DIFFY1(SIGMAXY);
    m_sigmaxyy[i] = P_AY(0) * m_sigmaxyy[i] + P_BY(0) * d2;
    d2 += m_sigmaxyy[i];
    w_sum += d2;
  }
#endif
  if (!pml_x) {
    w_sum += DIFFXH1(SIGMAXX);
  } else {
    DW_DTYPE d1 = DIFFXH1(SIGMAXX);
    m_sigmaxxx[i] = P_AXH(0) * m_sigmaxxx[i] + P_BXH(0) * d1;
    d1 += m_sigmaxxx[i];
    w_sum += d1;
  }

  if (buoyancy_requires_grad) {
    dvxdbuoyancy_store_1[i] = dt * w_sum;
  }
  vx[i] += buoyancy_x_shot * dt * w_sum;
}

/* sigmazz/sigmayy/sigmaxx += dt * (per-component weighted sum of
 * dvzdz/dvydy/dvxdx by c11/c12/c13/c33 -- see elastic_vti.c's top comment for
 * the full VTI constitutive relation), the combined normal-stress update.
 * Unlike elastic_kernels.h's FUNC(forward_sigmaii_cell) (isotropic: a single
 * lambda*div(v) term shared by all three normal stresses), VTI's three
 * normal stresses each mix c11/c12/c13/c33 differently, so this function
 * computes dvzdz/dvydy/dvxdx (with the same PML recursion machinery as
 * elastic's -- each axis's own PML memory-variable recursion, and (if
 * gradients are needed) storing dt*(dv_axis/d(axis)) for the backward pass)
 * and then combines them into THREE separately-weighted sigma updates rather
 * than one shared w_sum. `i` addresses vz/vy/vx/sigmazz/sigmayy/sigmaxx/
 * m_vzz/m_vyy/m_vxx/dv*d*_store_1 (per-shot-local on the CPU driver, global
 * on the CUDA driver); `c11_shot`/`c12_shot`/`c13_shot`/`c33_shot` are the
 * (already shot-selected) model VALUEs at this point (read once by the
 * caller -- never need a neighbour offset here); `pml_z`/`pml_y`/`pml_x` are
 * true when this point is inside that axis's PML region.
 */
DW_HOST_DEVICE void FUNC(forward_sigmaii_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vz, DW_DTYPE* __restrict const sigmazz,
    DW_DTYPE* __restrict const m_vzz, DW_DTYPE* __restrict const dvzdz_store_1,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vy, DW_DTYPE* __restrict const sigmayy,
    DW_DTYPE* __restrict const m_vyy, DW_DTYPE* __restrict const dvydy_store_1,
#endif
    DW_DTYPE const* __restrict const vx, DW_DTYPE* __restrict const sigmaxx,
    DW_DTYPE* __restrict const m_vxx, DW_DTYPE* __restrict const dvxdx_store_1,
    DW_DTYPE const c11_shot,
#if DW_NDIM >= 3
    DW_DTYPE const c12_shot,
#endif
    DW_DTYPE const c13_shot,
#if DW_NDIM >= 2
    DW_DTYPE const c33_shot,
#endif
    DW_DTYPE const dt, bool const cnormal_requires_grad, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const* __restrict const az,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const by, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const rdx) {
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

#if DW_NDIM >= 2
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
#endif

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

  // VTI constitutive relation (symmetric stiffness matrix). Symmetry axis is
  // the slowest dimension: z ("3") in 3-D, y ("3") in 2-D.
#if DW_NDIM == 3
  sigmazz[i] += dt * (c13_shot * dvxdx + c13_shot * dvydy + c33_shot * dvzdz);
  sigmayy[i] += dt * (c12_shot * dvxdx + c11_shot * dvydy + c13_shot * dvzdz);
  sigmaxx[i] += dt * (c11_shot * dvxdx + c12_shot * dvydy + c13_shot * dvzdz);
#else
  // 2-D P-SV: axis = y ("3"); x is Voigt "1".
  sigmayy[i] += dt * (c13_shot * dvxdx + c33_shot * dvydy);
  sigmaxx[i] += dt * (c11_shot * dvxdx + c13_shot * dvydy);
#endif
}

#if DW_NDIM >= 2
/* sigmaxy += dt * c_xy * (d(vx)/dy + d(vy)/dx) (see
 * FUNC(forward_sigmaii_cell)). c_xy is the shear stiffness at the sigma_xy
 * node: c66_yx in 3-D, c55_yx in 2-D. */
DW_HOST_DEVICE void FUNC(forward_sigmaxy_cell)(
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vy,
    DW_DTYPE* __restrict const sigmaxy, DW_DTYPE* __restrict const m_vxy,
    DW_DTYPE* __restrict const m_vyx,
    DW_DTYPE* __restrict const dvydx_plus_dvxdy_store_1,
    DW_DTYPE const c_xy_shot, DW_DTYPE const dt, bool const c_xy_requires_grad,
    int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny, int64_t const z,
#endif
    int64_t const nx, int64_t const y, int64_t const x, bool const pml_y,
    bool const pml_x, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const byh, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bxh, DW_DTYPE const rdy,
    DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;
  DW_DTYPE dvxdy, dvydx;

  if (pml_y) {
    dvxdy = DIFFYH1(VX);
    m_vxy[i] = P_AYH(0) * m_vxy[i] + P_BYH(0) * dvxdy;
    dvxdy += m_vxy[i];
  } else {
    dvxdy = DIFFYH1(VX);
  }
  w_sum += dvxdy;

  if (pml_x) {
    dvydx = DIFFXH1(VY);
    m_vyx[i] = P_AXH(0) * m_vyx[i] + P_BXH(0) * dvydx;
    dvydx += m_vyx[i];
  } else {
    dvydx = DIFFXH1(VY);
  }
  w_sum += dvydx;

  if (c_xy_requires_grad) {
    dvydx_plus_dvxdy_store_1[i] = dt * w_sum;
  }
  sigmaxy[i] += dt * c_xy_shot * w_sum;
}
#endif

#if DW_NDIM >= 3
/* sigmaxz += dt * c55_zx * (d(vx)/dz + d(vz)/dx) (see
 * FUNC(forward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(forward_sigmaxz_cell)(
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vz,
    DW_DTYPE* __restrict const sigmaxz, DW_DTYPE* __restrict const m_vxz,
    DW_DTYPE* __restrict const m_vzx,
    DW_DTYPE* __restrict const dvzdx_plus_dvxdz_store_1,
    DW_DTYPE const c55_zx_shot, DW_DTYPE const dt, bool const c55_requires_grad,
    int64_t const i, int64_t const ny, int64_t const nx, int64_t const z,
    int64_t const y, int64_t const x, bool const pml_z, bool const pml_x,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bzh,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const rdz, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;
  DW_DTYPE dvxdz, dvzdx;

  if (pml_z) {
    dvxdz = DIFFZH1(VX);
    m_vxz[i] = P_AZH(0) * m_vxz[i] + P_BZH(0) * dvxdz;
    dvxdz += m_vxz[i];
  } else {
    dvxdz = DIFFZH1(VX);
  }
  w_sum += dvxdz;

  if (pml_x) {
    dvzdx = DIFFXH1(VZ);
    m_vzx[i] = P_AXH(0) * m_vzx[i] + P_BXH(0) * dvzdx;
    dvzdx += m_vzx[i];
  } else {
    dvzdx = DIFFXH1(VZ);
  }
  w_sum += dvzdx;

  if (c55_requires_grad) {
    dvzdx_plus_dvxdz_store_1[i] = dt * w_sum;
  }
  sigmaxz[i] += dt * c55_zx_shot * w_sum;
}

/* sigmayz += dt * c55_zy * (d(vy)/dz + d(vz)/dy) (see
 * FUNC(forward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(forward_sigmayz_cell)(
    DW_DTYPE const* __restrict const vy, DW_DTYPE const* __restrict const vz,
    DW_DTYPE* __restrict const sigmayz, DW_DTYPE* __restrict const m_vyz,
    DW_DTYPE* __restrict const m_vzy,
    DW_DTYPE* __restrict const dvzdy_plus_dvydz_store_1,
    DW_DTYPE const c55_zy_shot, DW_DTYPE const dt, bool const c55_requires_grad,
    int64_t const i, int64_t const ny, int64_t const nx, int64_t const z,
    int64_t const y, int64_t const x, bool const pml_z, bool const pml_y,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bzh,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const rdz, DW_DTYPE const rdy) {
  DW_DTYPE w_sum = 0;
  DW_DTYPE dvydz, dvzdy;

  if (pml_z) {
    dvydz = DIFFZH1(VY);
    m_vyz[i] = P_AZH(0) * m_vyz[i] + P_BZH(0) * dvydz;
    dvydz += m_vyz[i];
  } else {
    dvydz = DIFFZH1(VY);
  }
  w_sum += dvydz;

  if (pml_y) {
    dvzdy = DIFFYH1(VZ);
    m_vzy[i] = P_AYH(0) * m_vzy[i] + P_BYH(0) * dvzdy;
    dvzdy += m_vzy[i];
  } else {
    dvzdy = DIFFYH1(VZ);
  }
  w_sum += dvzdy;

  if (c55_requires_grad) {
    dvzdy_plus_dvydz_store_1[i] = dt * w_sum;
  }
  sigmayz[i] += dt * c55_zy_shot * w_sum;
}
#endif

/* ============================= Backward ============================= */

#if DW_NDIM >= 3
/* vz += -(d(VZ_Z)/dz + d(VZ_Y)/dy + d(VZ_X)/dx) (the transpose of
 * FUNC(forward_sigmaii_cell)/sigmaxy_cell/sigmaxz_cell's use of vz). The
 * M_SIGMA** PML memory-variable recursion and the buoyancy-gradient
 * accumulation are split out into FUNC(backward_vz_cell_pml_mem) below and
 * run in a separate pass after the velocity guard-ring fold (see that
 * function's comment). Only defined for DW_NDIM>=3 (VZ_Z/VZ_Y/VZ_X only
 * exist there). `i`/`j` addressing matches elastic_kernels.h's
 * FUNC(backward_vz_cell); `c13_shot`/`c33_shot`/`c55_zy_shot`/`c55_zx_shot`
 * are POINTERS (dereferenced through the C13(0,0,0)-style macros, since the
 * VZ_Z(_PML)/VZ_Y(_PML)/VZ_X(_PML) stencils touch them at every offset the
 * finite-difference stencil visits, not just the centre point). */
DW_HOST_DEVICE void FUNC(backward_vz_cell)(
    DW_DTYPE const* __restrict const c13_shot,
    DW_DTYPE const* __restrict const c33_shot,
    DW_DTYPE const* __restrict const c55_zy_shot,
    DW_DTYPE const* __restrict const c55_zx_shot,
    DW_DTYPE const* __restrict const sigmazz,
    DW_DTYPE const* __restrict const sigmayy,
    DW_DTYPE const* __restrict const sigmaxx,
    DW_DTYPE const* __restrict const sigmayz,
    DW_DTYPE const* __restrict const sigmaxz, DW_DTYPE* __restrict const vz,
    DW_DTYPE const* __restrict const m_vzz,
    DW_DTYPE const* __restrict const m_vzy,
    DW_DTYPE const* __restrict const m_vzx,
    DW_DTYPE const* __restrict const m_sigmazzz_old,
    DW_DTYPE* __restrict const m_sigmazzz_new,
    DW_DTYPE const* __restrict const m_sigmayzy_old,
    DW_DTYPE* __restrict const m_sigmayzy_new,
    DW_DTYPE const* __restrict const m_sigmaxzx_old,
    DW_DTYPE* __restrict const m_sigmaxzx_new,
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, bool const pml_y, bool const pml_x, DW_DTYPE const rdz,
    DW_DTYPE const rdy, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;
  w_sum += pml_z ? -DIFFZH1(VZ_Z_PML) : -DIFFZH1(VZ_Z);
  w_sum += pml_y ? -DIFFY1(VZ_Y_PML) : -DIFFY1(VZ_Y);
  w_sum += pml_x ? -DIFFX1(VZ_X_PML) : -DIFFX1(VZ_X);

  vz[i] += w_sum;
}

/* M_SIGMA** PML memory-variable update for FUNC(backward_vz_cell) -- split
 * out and moved to run in a separate pass AFTER FUNC(fill_elastic_velocity_
 * bc_adjoint) (see the driver-side comment at its new call site). vz[i]
 * here must be read post-fold: for a cell the fold accumulates into or
 * clamps to zero (e.g. a rigid on-row velocity node), the value FUNC(
 * backward_vz_cell) computed pre-fold is stale -- the fold changes it
 * immediately afterward -- so writing this memory variable from the
 * pre-fold value would seed the next time step's recursion with the wrong
 * number for exactly those cells. For every other cell (the fold never
 * touches it) vz[i] is identical pre- and post-fold, so this split changes
 * nothing there. */
DW_HOST_DEVICE void FUNC(backward_vz_cell_pml_mem)(
    DW_DTYPE const* __restrict const vz,
    DW_DTYPE const* __restrict const m_sigmazzz_old,
    DW_DTYPE* __restrict const m_sigmazzz_new,
    DW_DTYPE const* __restrict const m_sigmayzy_old,
    DW_DTYPE* __restrict const m_sigmayzy_new,
    DW_DTYPE const* __restrict const m_sigmaxzx_old,
    DW_DTYPE* __restrict const m_sigmaxzx_new, DW_DTYPE const buoyancy_z_shot,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const ax, DW_DTYPE const dt, int64_t const i,
    int64_t const ny, int64_t const nx, int64_t const z, int64_t const y,
    int64_t const x, bool const pml_z, bool const pml_y, bool const pml_x) {
  if (pml_z) {
    m_sigmazzz_new[i] =
        buoyancy_z_shot * dt * P_AZH(0) * vz[i] + P_AZH(0) * m_sigmazzz_old[i];
  }
  if (pml_y) {
    m_sigmayzy_new[i] =
        buoyancy_z_shot * dt * P_AY(0) * vz[i] + P_AY(0) * m_sigmayzy_old[i];
  }
  if (pml_x) {
    m_sigmaxzx_new[i] =
        buoyancy_z_shot * dt * P_AX(0) * vz[i] + P_AX(0) * m_sigmaxzx_old[i];
  }
}
#endif

#if DW_NDIM >= 2
/* Adjoint velocity update, y axis (see FUNC(backward_vz_cell)). The
 * coefficient pointers needed depend on DW_NDIM: 2-D's VY_Y couples through
 * c13/c33 (axis = y is Voigt "3" there); 3-D's VY_Y couples through
 * c12/c11/c13 (axis = y is Voigt "2" there), and VY_Z needs c55_zy. */
DW_HOST_DEVICE void FUNC(backward_vy_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const c55_zy_shot,
    DW_DTYPE const* __restrict const c11_shot,
    DW_DTYPE const* __restrict const c12_shot,
    DW_DTYPE const* __restrict const c13_shot,
#else /* DW_NDIM == 2 */
    DW_DTYPE const* __restrict const c13_shot,
    DW_DTYPE const* __restrict const c33_shot,
#endif
    DW_DTYPE const* __restrict const c_xy_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmazz,
#endif
    DW_DTYPE const* __restrict const sigmayy,
    DW_DTYPE const* __restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmayz,
#endif
    DW_DTYPE const* __restrict const sigmaxy, DW_DTYPE* __restrict const vy,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_vyz,
#endif
    DW_DTYPE const* __restrict const m_vyy,
    DW_DTYPE const* __restrict const m_vyx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmayzz_old,
    DW_DTYPE* __restrict const m_sigmayzz_new,
#endif
    DW_DTYPE const* __restrict const m_sigmayyy_old,
    DW_DTYPE* __restrict const m_sigmayyy_new,
    DW_DTYPE const* __restrict const m_sigmaxyx_old,
    DW_DTYPE* __restrict const m_sigmaxyx_new,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const rdz,
#endif
    int64_t const y, int64_t const x, bool const pml_y, bool const pml_x,
    DW_DTYPE const rdy, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;

#if DW_NDIM >= 3
  w_sum += pml_z ? -DIFFZ1(VY_Z_PML) : -DIFFZ1(VY_Z);
#endif
  w_sum += pml_y ? -DIFFYH1(VY_Y_PML) : -DIFFYH1(VY_Y);
  w_sum += pml_x ? -DIFFX1(VY_X_PML) : -DIFFX1(VY_X);

  vy[i] += w_sum;
}

/* M_SIGMA** PML memory-variable update for FUNC(backward_vy_cell) -- see
 * FUNC(backward_vz_cell_pml_mem)'s comment (same split, same reason: vy[i]
 * must be read post-fold). */
DW_HOST_DEVICE void FUNC(backward_vy_cell_pml_mem)(
    DW_DTYPE const* __restrict const vy,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmayzz_old,
    DW_DTYPE* __restrict const m_sigmayzz_new,
#endif
    DW_DTYPE const* __restrict const m_sigmayyy_old,
    DW_DTYPE* __restrict const m_sigmayyy_new,
    DW_DTYPE const* __restrict const m_sigmaxyx_old,
    DW_DTYPE* __restrict const m_sigmaxyx_new, DW_DTYPE const buoyancy_y_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az,
#endif
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const dt, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny, int64_t const z, bool const pml_z,
#endif
    int64_t const nx, int64_t const y, int64_t const x, bool const pml_y,
    bool const pml_x) {
#if DW_NDIM >= 3
  if (pml_z) {
    m_sigmayzz_new[i] =
        buoyancy_y_shot * dt * P_AZ(0) * vy[i] + P_AZ(0) * m_sigmayzz_old[i];
  }
#endif
  if (pml_y) {
    m_sigmayyy_new[i] =
        buoyancy_y_shot * dt * P_AYH(0) * vy[i] + P_AYH(0) * m_sigmayyy_old[i];
  }
  if (pml_x) {
    m_sigmaxyx_new[i] =
        buoyancy_y_shot * dt * P_AX(0) * vy[i] + P_AX(0) * m_sigmaxyx_old[i];
  }
}
#endif

/* Adjoint velocity update, x axis (see FUNC(backward_vz_cell)). The
 * coefficient pointers needed depend on DW_NDIM (see VX_X/VX_Y/VX_Z above):
 * 1-D needs only c11; 2-D needs c11/c13/c_xy; 3-D needs c11/c12/c13/c_xy/
 * c55_zx. */
DW_HOST_DEVICE void FUNC(backward_vx_cell)(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const c55_zx_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const c_xy_shot,
#endif
    DW_DTYPE const* __restrict const c11_shot,
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const c13_shot,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const c12_shot,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmayy,
#endif
    DW_DTYPE const* __restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmaxy,
#endif
    DW_DTYPE* __restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_vxy,
#endif
    DW_DTYPE const* __restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmaxzz_old,
    DW_DTYPE* __restrict const m_sigmaxzz_new,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmaxyy_old,
    DW_DTYPE* __restrict const m_sigmaxyy_new,
#endif
    DW_DTYPE const* __restrict const m_sigmaxxx_old,
    DW_DTYPE* __restrict const m_sigmaxxx_new,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;

#if DW_NDIM >= 3
  w_sum += pml_z ? -DIFFZ1(VX_Z_PML) : -DIFFZ1(VX_Z);
#endif
#if DW_NDIM >= 2
  w_sum += pml_y ? -DIFFY1(VX_Y_PML) : -DIFFY1(VX_Y);
#endif
  w_sum += pml_x ? -DIFFXH1(VX_X_PML) : -DIFFXH1(VX_X);

  vx[i] += w_sum;
}

/* M_SIGMA** PML memory-variable update for FUNC(backward_vx_cell) -- see
 * FUNC(backward_vz_cell_pml_mem)'s comment (same split, same reason: vx[i]
 * must be read post-fold). */
DW_HOST_DEVICE void FUNC(backward_vx_cell_pml_mem)(
    DW_DTYPE const* __restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmaxzz_old,
    DW_DTYPE* __restrict const m_sigmaxzz_new,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmaxyy_old,
    DW_DTYPE* __restrict const m_sigmaxyy_new,
#endif
    DW_DTYPE const* __restrict const m_sigmaxxx_old,
    DW_DTYPE* __restrict const m_sigmaxxx_new, DW_DTYPE const buoyancy_x_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay,
#endif
    DW_DTYPE const* __restrict const axh, DW_DTYPE const dt, int64_t const i,
#if DW_NDIM >= 3
    int64_t const ny, int64_t const z, bool const pml_z,
#endif
#if DW_NDIM >= 2
    int64_t const nx, int64_t const y, bool const pml_y,
#endif
    int64_t const x, bool const pml_x) {
#if DW_NDIM >= 3
  if (pml_z) {
    m_sigmaxzz_new[i] =
        buoyancy_x_shot * dt * P_AZ(0) * vx[i] + P_AZ(0) * m_sigmaxzz_old[i];
  }
#endif
#if DW_NDIM >= 2
  if (pml_y) {
    m_sigmaxyy_new[i] =
        buoyancy_x_shot * dt * P_AY(0) * vx[i] + P_AY(0) * m_sigmaxyy_old[i];
  }
#endif
  if (pml_x) {
    m_sigmaxxx_new[i] =
        buoyancy_x_shot * dt * P_AXH(0) * vx[i] + P_AXH(0) * m_sigmaxxx_old[i];
  }
}

/* grad_c11/grad_c12/grad_c13/grad_c33 accumulation (cross-correlations of the
 * adjoint stresses with the stored forward velocity derivatives -- see
 * elastic_vti.c's top comment), the M_V** PML memory-variable update (read by
 * FUNC(backward_v*_cell) at the *next* time step), and sigmazz/sigmayy/
 * sigmaxx += -(d(SIGMAZZ_Z)/dz + d(SIGMAYY_Y)/dy + d(SIGMAXX_X)/dx) (the
 * transpose of FUNC(forward_vz_cell)/vy_cell/vx_cell's use of sigma* --
 * identical in shape to elastic_kernels.h's FUNC(backward_sigmaii_cell), no
 * coefficient dependency). `i`/`j` addressing matches FUNC(backward_vz_cell)
 * above. `c11_shot`/`c12_shot`/`c13_shot`/`c33_shot` are read only at the
 * centre point here (no finite-difference neighbour offset touches them in
 * this function), but are passed as POINTERS (dereferenced through the
 * C11(0,0,0)-style macro) to match elastic_vti.cu's existing convention.
 * `buoyancy_z_shot`/`buoyancy_y_shot`/`buoyancy_x_shot` ARE read with a
 * neighbour offset here (via SIGMAZZ_Z(_PML) etc.), so are genuinely
 * POINTERS. */
DW_HOST_DEVICE void FUNC(backward_sigmaii_cell)(
    DW_DTYPE const* __restrict const c11_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const c12_shot,
#endif
    DW_DTYPE const* __restrict const c13_shot,
    DW_DTYPE const* __restrict const c33_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const buoyancy_z_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const buoyancy_y_shot,
#endif
    DW_DTYPE const* __restrict const buoyancy_x_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const vz, DW_DTYPE* __restrict const sigmazz,
    DW_DTYPE* __restrict const m_vzz,
    DW_DTYPE const* __restrict const m_sigmazzz,
    DW_DTYPE const* __restrict const dvzdz_store,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const vy, DW_DTYPE* __restrict const sigmayy,
    DW_DTYPE* __restrict const m_vyy,
    DW_DTYPE const* __restrict const m_sigmayyy,
    DW_DTYPE const* __restrict const dvydy_store,
#endif
    DW_DTYPE const* __restrict const vx, DW_DTYPE* __restrict const sigmaxx,
    DW_DTYPE* __restrict const m_vxx,
    DW_DTYPE const* __restrict const m_sigmaxxx,
    DW_DTYPE const* __restrict const dvxdx_store,
    DW_DTYPE* __restrict const grad_c11,
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const grad_c12,
#endif
    DW_DTYPE* __restrict const grad_c13, DW_DTYPE* __restrict const grad_c33,
    DW_DTYPE const dt, int64_t const step_ratio, bool const c11_requires_grad,
#if DW_NDIM >= 3
    bool const c12_requires_grad,
#endif
    bool const c13_requires_grad, bool const c33_requires_grad, int64_t const i,
    int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
#if DW_NDIM >= 2
    int64_t const nx,
#endif
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const* __restrict const az,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const bzh, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const byh, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const bxh, DW_DTYPE const rdx) {
  if (c11_requires_grad) {
#if DW_NDIM == 3
    grad_c11[i] += (sigmaxx[i] * dvxdx_store[i] + sigmayy[i] * dvydy_store[i]) *
                   (DW_DTYPE)step_ratio;
#else
    grad_c11[i] += sigmaxx[i] * dvxdx_store[i] * (DW_DTYPE)step_ratio;
#endif
  }
#if DW_NDIM >= 3
  if (c12_requires_grad) {
    grad_c12[i] += (sigmaxx[i] * dvydy_store[i] + sigmayy[i] * dvxdx_store[i]) *
                   (DW_DTYPE)step_ratio;
  }
#endif
  if (c13_requires_grad) {
#if DW_NDIM == 3
    grad_c13[i] += (sigmazz[i] * (dvxdx_store[i] + dvydy_store[i]) +
                    (sigmaxx[i] + sigmayy[i]) * dvzdz_store[i]) *
                   (DW_DTYPE)step_ratio;
#elif DW_NDIM == 2
    grad_c13[i] += (sigmaxx[i] * dvydy_store[i] + sigmayy[i] * dvxdx_store[i]) *
                   (DW_DTYPE)step_ratio;
#endif
  }
  if (c33_requires_grad) {
#if DW_NDIM == 3
    grad_c33[i] += sigmazz[i] * dvzdz_store[i] * (DW_DTYPE)step_ratio;
#elif DW_NDIM == 2
    grad_c33[i] += sigmayy[i] * dvydy_store[i] * (DW_DTYPE)step_ratio;
#endif
  }

  // Adjoint PML memory (transpose of the constitutive coupling of each
  // normal-velocity derivative into the stresses).
#if DW_NDIM >= 3
  if (pml_z) {
    m_vzz[i] = dt * P_AZ(0) *
                   (C13(0, 0, 0) * sigmaxx[i] + C13(0, 0, 0) * sigmayy[i] +
                    C33(0, 0, 0) * sigmazz[i]) +
               P_AZ(0) * m_vzz[i];
  }
#endif
#if DW_NDIM >= 2
  if (pml_y) {
    m_vyy[i] = dt * P_AY(0) *
#if DW_NDIM == 3
                   (C12(0, 0, 0) * sigmaxx[i] + C11(0, 0, 0) * sigmayy[i] +
                    C13(0, 0, 0) * sigmazz[i]) +
#else
                   (C13(0, 0, 0) * sigmaxx[i] + C33(0, 0, 0) * sigmayy[i]) +
#endif
               P_AY(0) * m_vyy[i];
  }
#endif
  if (pml_x) {
    m_vxx[i] = dt * P_AX(0) *
#if DW_NDIM == 3
                   (C11(0, 0, 0) * sigmaxx[i] + C12(0, 0, 0) * sigmayy[i] +
                    C13(0, 0, 0) * sigmazz[i]) +
#elif DW_NDIM == 2
                   (C11(0, 0, 0) * sigmaxx[i] + C13(0, 0, 0) * sigmayy[i]) +
#else
                   (C11(0, 0, 0) * sigmaxx[i]) +
#endif
               P_AX(0) * m_vxx[i];
  }

#if DW_NDIM >= 3
  sigmazz[i] += pml_z ? -DIFFZ1(SIGMAZZ_Z_PML) : -DIFFZ1(SIGMAZZ_Z);
#endif
#if DW_NDIM >= 2
  sigmayy[i] += pml_y ? -DIFFY1(SIGMAYY_Y_PML) : -DIFFY1(SIGMAYY_Y);
#endif
  sigmaxx[i] += pml_x ? -DIFFX1(SIGMAXX_X_PML) : -DIFFX1(SIGMAXX_X);
}

#if DW_NDIM >= 2
/* grad_c_xy accumulation, M_VXY/M_VYX PML memory update, and sigmaxy +=
 * -(d(SIGMAXY_Y)/dy + d(SIGMAXY_X)/dx) (see
 * FUNC(backward_sigmaii_cell)). c_xy is the shear stiffness at the sigma_xy
 * node: c66_yx in 3-D, c55_yx in 2-D. */
DW_HOST_DEVICE void FUNC(backward_sigmaxy_cell)(
    DW_DTYPE const* __restrict const c_xy_shot,
    DW_DTYPE const* __restrict const buoyancy_x_shot,
    DW_DTYPE const* __restrict const buoyancy_y_shot,
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vy,
    DW_DTYPE* __restrict const sigmaxy, DW_DTYPE* __restrict const m_vxy,
    DW_DTYPE* __restrict const m_vyx,
    DW_DTYPE const* __restrict const m_sigmaxyy,
    DW_DTYPE const* __restrict const m_sigmaxyx,
    DW_DTYPE const* __restrict const dvydx_plus_dvxdy_store,
    DW_DTYPE* __restrict const grad_c_xy, DW_DTYPE const dt,
    int64_t const step_ratio, bool const c_xy_requires_grad, int64_t const i,
    int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny, int64_t const z,
#endif
    int64_t const nx, int64_t const y, int64_t const x, bool const pml_y,
    bool const pml_x, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const byh, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const bxh, DW_DTYPE const rdy,
    DW_DTYPE const rdx) {
  if (c_xy_requires_grad) {
    grad_c_xy[i] +=
        sigmaxy[i] * dvydx_plus_dvxdy_store[i] * (DW_DTYPE)step_ratio;
  }

  if (pml_y) {
    m_vxy[i] = C_XY(0, 0, 0) * dt * P_AYH(0) * sigmaxy[i] + P_AYH(0) * m_vxy[i];
  }
  if (pml_x) {
    m_vyx[i] = C_XY(0, 0, 0) * dt * P_AXH(0) * sigmaxy[i] + P_AXH(0) * m_vyx[i];
  }

  sigmaxy[i] += (pml_y ? -DIFFYH1(SIGMAXY_Y_PML) : -DIFFYH1(SIGMAXY_Y)) +
                (pml_x ? -DIFFXH1(SIGMAXY_X_PML) : -DIFFXH1(SIGMAXY_X));
}
#endif

#if DW_NDIM >= 3
/* grad_c55_zx accumulation, M_VXZ/M_VZX PML memory update, and sigmaxz +=
 * -(d(SIGMAXZ_Z)/dz + d(SIGMAXZ_X)/dx) (see
 * FUNC(backward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(backward_sigmaxz_cell)(
    DW_DTYPE const* __restrict const c55_zx_shot,
    DW_DTYPE const* __restrict const buoyancy_x_shot,
    DW_DTYPE const* __restrict const buoyancy_z_shot,
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vz,
    DW_DTYPE* __restrict const sigmaxz, DW_DTYPE* __restrict const m_vxz,
    DW_DTYPE* __restrict const m_vzx,
    DW_DTYPE const* __restrict const m_sigmaxzz,
    DW_DTYPE const* __restrict const m_sigmaxzx,
    DW_DTYPE const* __restrict const dvzdx_plus_dvxdz_store,
    DW_DTYPE* __restrict const grad_c55_zx, DW_DTYPE const dt,
    int64_t const step_ratio, bool const c55_requires_grad, int64_t const i,
    int64_t const j, int64_t const ny, int64_t const nx, int64_t const z,
    int64_t const y, int64_t const x, bool const pml_z, bool const pml_x,
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const rdz, DW_DTYPE const rdx) {
  if (c55_requires_grad) {
    grad_c55_zx[i] +=
        sigmaxz[i] * dvzdx_plus_dvxdz_store[i] * (DW_DTYPE)step_ratio;
  }

  if (pml_z) {
    m_vxz[i] =
        C55_ZX(0, 0, 0) * dt * P_AZH(0) * sigmaxz[i] + P_AZH(0) * m_vxz[i];
  }
  if (pml_x) {
    m_vzx[i] =
        C55_ZX(0, 0, 0) * dt * P_AXH(0) * sigmaxz[i] + P_AXH(0) * m_vzx[i];
  }

  sigmaxz[i] += (pml_z ? -DIFFZH1(SIGMAXZ_Z_PML) : -DIFFZH1(SIGMAXZ_Z)) +
                (pml_x ? -DIFFXH1(SIGMAXZ_X_PML) : -DIFFXH1(SIGMAXZ_X));
}

/* grad_c55_zy accumulation, M_VYZ/M_VZY PML memory update, and sigmayz +=
 * -(d(SIGMAYZ_Z)/dz + d(SIGMAYZ_Y)/dy) (see
 * FUNC(backward_sigmaii_cell)). */
DW_HOST_DEVICE void FUNC(backward_sigmayz_cell)(
    DW_DTYPE const* __restrict const c55_zy_shot,
    DW_DTYPE const* __restrict const buoyancy_y_shot,
    DW_DTYPE const* __restrict const buoyancy_z_shot,
    DW_DTYPE const* __restrict const vy, DW_DTYPE const* __restrict const vz,
    DW_DTYPE* __restrict const sigmayz, DW_DTYPE* __restrict const m_vyz,
    DW_DTYPE* __restrict const m_vzy,
    DW_DTYPE const* __restrict const m_sigmayzz,
    DW_DTYPE const* __restrict const m_sigmayzy,
    DW_DTYPE const* __restrict const dvzdy_plus_dvydz_store,
    DW_DTYPE* __restrict const grad_c55_zy, DW_DTYPE const dt,
    int64_t const step_ratio, bool const c55_requires_grad, int64_t const i,
    int64_t const j, int64_t const ny, int64_t const nx, int64_t const z,
    int64_t const y, int64_t const x, bool const pml_z, bool const pml_y,
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const rdz, DW_DTYPE const rdy) {
  if (c55_requires_grad) {
    grad_c55_zy[i] +=
        sigmayz[i] * dvzdy_plus_dvydz_store[i] * (DW_DTYPE)step_ratio;
  }

  if (pml_z) {
    m_vyz[i] =
        C55_ZY(0, 0, 0) * dt * P_AZH(0) * sigmayz[i] + P_AZH(0) * m_vyz[i];
  }
  if (pml_y) {
    m_vzy[i] =
        C55_ZY(0, 0, 0) * dt * P_AYH(0) * sigmayz[i] + P_AYH(0) * m_vzy[i];
  }

  sigmayz[i] += (pml_z ? -DIFFZH1(SIGMAYZ_Z_PML) : -DIFFZH1(SIGMAYZ_Z)) +
                (pml_y ? -DIFFYH1(SIGMAYZ_Y_PML) : -DIFFYH1(SIGMAYZ_Y));
}
#endif

/* ========================= Guard-band gathers ========================= *
 *
 * The backward gather kernels above (FUNC(backward_v*_cell)/FUNC(backward_
 * sigma*_cell)) only ever run over the grid's strict interior/PML region
 * (elastic_vti.c's/elastic_vti.cu's pml_bounds_* / pml_bounds_*h arrays both
 * start at FD_PAD and stop FD_PAD or FD_PAD-1 short of the far edge), so
 * they never deposit an adjoint contribution into the guard ring next to a
 * reflecting (non-PML) face. FUNC(fill_elastic_stress_bc_adjoint)/FUNC(
 * fill_elastic_velocity_bc_adjoint) (see the module comment above those,
 * elastic_vti.c) then fold that guard ring into the interior -- but with
 * nothing genuine ever written there, that fold silently discards real
 * gradient contributions. The functions below are guard-band-only
 * companions to the interior gathers: same physics (transpose of the
 * corresponding FUNC(forward_*_cell)), but evaluated with bounds-safe
 * stencil taps (DW_DIFF*_SAFE, which treat any tap landing outside [0,
 * nz)/[0, ny)/[0, nx) as exactly zero -- see elastic_kernels.h's identical
 * mechanism, ported unchanged here) so they may be called at guard-ring
 * cells, where a stencil can reach up to FD_PAD cells past the physical
 * array. They omit the grad_c11/grad_c12/grad_c13/grad_c33/grad_c_xy/
 * grad_c55_zx/grad_c55_zy accumulation the interior functions perform
 * (guard cells are not independent model-gradient locations -- the model
 * itself is only mirror-padded, never an independent DOF there) and, for
 * the velocity gathers, the buoyancy-gradient accumulation the interior
 * functions used to perform too (moved to a separate post-fold pass -- see
 * elastic_vti.c's/elastic_vti.cu's driver-side comment at its new call
 * site -- since a pre-fold read of a near-face interior cell is stale once
 * the fold changes it).
 *
 * elastic_vti.c's/elastic_vti.cu's new guard-band driver loops call these
 * over exactly the *new* cells a field's boundary fill touches beyond the
 * existing interior/PML loop -- see FUNC(dw_compute_ext_range) below -- so
 * double-accumulation against the interior loop's own cells never occurs.
 * Every axis argument here (including ones that happen to be this call's
 * "currently extended" axis, which is always non-PML by construction) is
 * treated fully generally: a transverse axis can legitimately be PML-active
 * even while another axis of the same cell is in that axis's guard ring
 * (e.g. boundary=["free","rigid","pml","pml"]: Y is guard-extended, X has a
 * real PML region the Y-guard-band loop still sweeps through), so
 * is_pml_z/is_pml_y/is_pml_x are ordinary parameters here exactly as in the
 * interior functions, and the PML memory-variable read/write these
 * functions perform is left completely intact (a sibling field's OWN
 * guard-band gather -- e.g. sigmaxy's, which also extends into the same Y
 * guard ring -- may need to read the M_SIGMA** / M_V** value a
 * velocity/stress guard gather wrote there this same time step).
 */

/* Guard-extension bookkeeping for one field's use of one axis. `n` is that
 * axis's full length (nz/ny/nx); `bc_lo`/`bc_hi` are that axis's two face
 * codes (0 = pml, 1 = free, 2 = rigid, 3 = periodic); `is_half` selects the
 * half-grid convention (velocity's own axis; a shear stress's two own axes)
 * vs full-grid (everything else). For an axis this field does not touch at
 * all, callers pass `bc_lo=bc_hi=0` regardless of that axis's real face
 * codes, so `target` degenerates to `std` and the new-cell test below is
 * trivially false everywhere on that axis.
 *
 * `target_lo/target_hi` is the field's full boundary-fill-aware target
 * range on this axis; `std_lo/std_hi` is the existing interior/PML loop's
 * own (unextended) range on this axis (matches pml_bounds_z/pml_bounds_zh
 * etc.'s own [FD_PAD, n-FD_PAD+1) / [FD_PAD, n-FD_PAD) span) -- cells with
 * EVERY touched axis inside `std` are already handled by the interior loop
 * and must be skipped -- callers do this with FUNC(dw_in_target) AND
 * NOT-ALL-axes-FUNC(dw_in_std) below. `has_exclude`/`exclude_idx`
 * implements the half-lattice "extra node" (`n - FD_PAD`) rule: under a
 * half-grid axis's free or rigid high face that node is excluded from the
 * target set (a live untouched DOF under free; a clamped, fold-zeroed DOF
 * under rigid -- either way not part of the new-cell set), included under
 * periodic (the slaved node the fold-fill's `bc_hi==3` step folds and
 * zeros).
 */
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
 * sub-region, replicating elastic_vti.c's/elastic_vti.cu's own region split
 * (pml_bounds_* / pml_bounds_*h: region 0 = [FD_PAD, pml_lo), region 2 =
 * [pml_hi, std_hi)) so a guard-band gather agrees with the interior
 * gather's own treatment of any cell both loops cover -- in particular the
 * single boundary-adjacent cell that a genuinely reflecting face's
 * `pml_hi == std_hi - 1` (a reflecting face always has zero PML width, so
 * `pml_hi` sits exactly one cell inside the array edge) still labels as
 * region 2/is_pml, relying on the PML profile arrays being trivial (a=1,
 * b=0) there. Cells strictly inside the guard ring (c < FD_PAD or c >=
 * std_hi) always evaluate false here, since `pml_lo`/`pml_hi` never reach
 * past FD_PAD/std_hi on a genuinely reflecting face -- so this never reads
 * a PML profile array out of bounds either. `pml_hi` must be pre-adjusted
 * by the caller for a half-grid axis (`MAX(pml_lo, pml_hi_raw - 1)`,
 * matching pml_bounds_*h's own construction) before being passed in here;
 * `std_hi` should be the same-role FUNC(dw_ext_range)'s `std_hi` (so the
 * half/full adjustment stays consistent between the two). */
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
 * VZ_Z, SIGMAXY_X_PML) at this axis's offset `d` from the current cell: `a`
 * itself (which may read model/wavefield/PML-memory arrays at that offset,
 * all at once, via the ND_INDEX-based macros above) is only evaluated -- so
 * only ever dereferenced -- when the offset cell is inside the array; a
 * tap landing outside contributes exactly zero. */
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
 * (staggered_grid.h) via the general identity DIFF1(a) = rd * sum_k
 * w[k]*(a(k) - a(-k-1)), DIFFH1(a) = rd * sum_k w[k]*(a(k+1) - a(-k)), k =
 * 0..FD_PAD-1 (`s` is 0 for the *1 family, 1 for the *H1 family) -- the
 * Taylor coefficients above satisfy this identity by construction, so
 * DW_DIFF*_SAFE below agrees bit-for-bit with DIFF*1/DIFF*H1 whenever every
 * tap is in bounds. elastic_vti_kernels.h may not modify staggered_grid.h
 * (out of scope), hence the duplication here. */
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
/* Guard-band companion to FUNC(backward_vz_cell)'s gather (no buoyancy-grad
 * accumulation -- see the module comment above). `nz` is required here
 * (unlike FUNC(backward_vz_cell), which never needs it for flat indexing)
 * purely for DW_SAFE_TAP_Z's bounds check. */
DW_HOST_DEVICE void FUNC(backward_vz_cell_guard)(
    DW_DTYPE const* __restrict const c13_shot,
    DW_DTYPE const* __restrict const c33_shot,
    DW_DTYPE const* __restrict const c55_zy_shot,
    DW_DTYPE const* __restrict const c55_zx_shot,
    DW_DTYPE const* __restrict const sigmazz,
    DW_DTYPE const* __restrict const sigmayy,
    DW_DTYPE const* __restrict const sigmaxx,
    DW_DTYPE const* __restrict const sigmayz,
    DW_DTYPE const* __restrict const sigmaxz, DW_DTYPE* __restrict const vz,
    DW_DTYPE const* __restrict const m_vzz,
    DW_DTYPE const* __restrict const m_vzy,
    DW_DTYPE const* __restrict const m_vzx,
    DW_DTYPE const* __restrict const m_sigmazzz_old,
    DW_DTYPE* __restrict const m_sigmazzz_new,
    DW_DTYPE const* __restrict const m_sigmayzy_old,
    DW_DTYPE* __restrict const m_sigmayzy_new,
    DW_DTYPE const* __restrict const m_sigmaxzx_old,
    DW_DTYPE* __restrict const m_sigmaxzx_new, DW_DTYPE const buoyancy_z_shot,
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const nz,
    int64_t const ny, int64_t const nx, int64_t const z, int64_t const y,
    int64_t const x, bool const pml_z, bool const pml_y, bool const pml_x,
    DW_DTYPE const rdz, DW_DTYPE const rdy, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;
  w_sum += pml_z ? -DW_DIFFZH1_SAFE(VZ_Z_PML) : -DW_DIFFZH1_SAFE(VZ_Z);
  w_sum += pml_y ? -DW_DIFFY1_SAFE(VZ_Y_PML) : -DW_DIFFY1_SAFE(VZ_Y);
  w_sum += pml_x ? -DW_DIFFX1_SAFE(VZ_X_PML) : -DW_DIFFX1_SAFE(VZ_X);

  vz[i] += w_sum;

  if (pml_z) {
    m_sigmazzz_new[i] =
        buoyancy_z_shot * dt * P_AZH(0) * vz[i] + P_AZH(0) * m_sigmazzz_old[i];
  }
  if (pml_y) {
    m_sigmayzy_new[i] =
        buoyancy_z_shot * dt * P_AY(0) * vz[i] + P_AY(0) * m_sigmayzy_old[i];
  }
  if (pml_x) {
    m_sigmaxzx_new[i] =
        buoyancy_z_shot * dt * P_AX(0) * vz[i] + P_AX(0) * m_sigmaxzx_old[i];
  }
}
#endif

#if DW_NDIM >= 2
/* Guard-band companion to FUNC(backward_vy_cell)'s gather (see
 * FUNC(backward_vz_cell_guard)). `ny` is required unconditionally here
 * (unlike FUNC(backward_vy_cell), which only needs it for ND_INDEX when
 * DW_NDIM==3) purely for DW_SAFE_TAP_Y's bounds check. */
DW_HOST_DEVICE void FUNC(backward_vy_cell_guard)(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const c55_zy_shot,
    DW_DTYPE const* __restrict const c11_shot,
    DW_DTYPE const* __restrict const c12_shot,
    DW_DTYPE const* __restrict const c13_shot,
#else /* DW_NDIM == 2 */
    DW_DTYPE const* __restrict const c13_shot,
    DW_DTYPE const* __restrict const c33_shot,
#endif
    DW_DTYPE const* __restrict const c_xy_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmazz,
#endif
    DW_DTYPE const* __restrict const sigmayy,
    DW_DTYPE const* __restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmayz,
#endif
    DW_DTYPE const* __restrict const sigmaxy, DW_DTYPE* __restrict const vy,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_vyz,
#endif
    DW_DTYPE const* __restrict const m_vyy,
    DW_DTYPE const* __restrict const m_vyx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmayzz_old,
    DW_DTYPE* __restrict const m_sigmayzz_new,
#endif
    DW_DTYPE const* __restrict const m_sigmayyy_old,
    DW_DTYPE* __restrict const m_sigmayyy_new,
    DW_DTYPE const* __restrict const m_sigmaxyx_old,
    DW_DTYPE* __restrict const m_sigmaxyx_new, DW_DTYPE const buoyancy_y_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const nz,
#endif
    int64_t const ny, int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const rdz,
#endif
    int64_t const y, int64_t const x, bool const pml_y, bool const pml_x,
    DW_DTYPE const rdy, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;

#if DW_NDIM >= 3
  w_sum += pml_z ? -DW_DIFFZ1_SAFE(VY_Z_PML) : -DW_DIFFZ1_SAFE(VY_Z);
#endif
  w_sum += pml_y ? -DW_DIFFYH1_SAFE(VY_Y_PML) : -DW_DIFFYH1_SAFE(VY_Y);
  w_sum += pml_x ? -DW_DIFFX1_SAFE(VY_X_PML) : -DW_DIFFX1_SAFE(VY_X);

  vy[i] += w_sum;

#if DW_NDIM >= 3
  if (pml_z) {
    m_sigmayzz_new[i] =
        buoyancy_y_shot * dt * P_AZ(0) * vy[i] + P_AZ(0) * m_sigmayzz_old[i];
  }
#endif
  if (pml_y) {
    m_sigmayyy_new[i] =
        buoyancy_y_shot * dt * P_AYH(0) * vy[i] + P_AYH(0) * m_sigmayyy_old[i];
  }
  if (pml_x) {
    m_sigmaxyx_new[i] =
        buoyancy_y_shot * dt * P_AX(0) * vy[i] + P_AX(0) * m_sigmaxyx_old[i];
  }
}
#endif

/* Guard-band companion to FUNC(backward_vx_cell)'s gather (see
 * FUNC(backward_vz_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_vx_cell_guard)(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const c55_zx_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const c_xy_shot,
#endif
    DW_DTYPE const* __restrict const c11_shot,
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const c13_shot,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const c12_shot,
#endif
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmayy,
#endif
    DW_DTYPE const* __restrict const sigmaxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmaxy,
#endif
    DW_DTYPE* __restrict const vx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_vxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_vxy,
#endif
    DW_DTYPE const* __restrict const m_vxx,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const m_sigmaxzz_old,
    DW_DTYPE* __restrict const m_sigmaxzz_new,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const m_sigmaxyy_old,
    DW_DTYPE* __restrict const m_sigmaxyy_new,
#endif
    DW_DTYPE const* __restrict const m_sigmaxxx_old,
    DW_DTYPE* __restrict const m_sigmaxxx_new, DW_DTYPE const buoyancy_x_shot,
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const az, DW_DTYPE const* __restrict const azh,
    DW_DTYPE const* __restrict const bz, DW_DTYPE const* __restrict const bzh,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const ay, DW_DTYPE const* __restrict const ayh,
    DW_DTYPE const* __restrict const by, DW_DTYPE const* __restrict const byh,
#endif
    DW_DTYPE const* __restrict const ax, DW_DTYPE const* __restrict const axh,
    DW_DTYPE const* __restrict const bx, DW_DTYPE const* __restrict const bxh,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const nz,
#endif
#if DW_NDIM >= 2
    int64_t const ny,
#endif
    int64_t const nx,
#if DW_NDIM >= 3
    int64_t const z, bool const pml_z, DW_DTYPE const rdz,
#endif
#if DW_NDIM >= 2
    int64_t const y, bool const pml_y, DW_DTYPE const rdy,
#endif
    int64_t const x, bool const pml_x, DW_DTYPE const rdx) {
  DW_DTYPE w_sum = 0;

#if DW_NDIM >= 3
  w_sum += pml_z ? -DW_DIFFZ1_SAFE(VX_Z_PML) : -DW_DIFFZ1_SAFE(VX_Z);
#endif
#if DW_NDIM >= 2
  w_sum += pml_y ? -DW_DIFFY1_SAFE(VX_Y_PML) : -DW_DIFFY1_SAFE(VX_Y);
#endif
  w_sum += pml_x ? -DW_DIFFXH1_SAFE(VX_X_PML) : -DW_DIFFXH1_SAFE(VX_X);

  vx[i] += w_sum;

#if DW_NDIM >= 3
  if (pml_z) {
    m_sigmaxzz_new[i] =
        buoyancy_x_shot * dt * P_AZ(0) * vx[i] + P_AZ(0) * m_sigmaxzz_old[i];
  }
#endif
#if DW_NDIM >= 2
  if (pml_y) {
    m_sigmaxyy_new[i] =
        buoyancy_x_shot * dt * P_AY(0) * vx[i] + P_AY(0) * m_sigmaxyy_old[i];
  }
#endif
  if (pml_x) {
    m_sigmaxxx_new[i] =
        buoyancy_x_shot * dt * P_AXH(0) * vx[i] + P_AXH(0) * m_sigmaxxx_old[i];
  }
}

#if DW_NDIM >= 3
/* Guard-band companion to the sigmazz sub-update inside FUNC(backward_
 * sigmaii_cell): sigmazz's only term is its own (z) axis (SIGMAZZ_Z/
 * SIGMAZZ_Z_PML), so a guard cell here -- which by construction only
 * exists on an axis with a non-PML face -- is never inside a PML region on
 * that same axis; the M_VZZ PML-memory write the interior function
 * performs (gated by the identical, always-false-here condition) is
 * therefore unreachable and intentionally omitted, and grad_c33/grad_c13
 * (also computed there, using sigmazz's value at the *old* cell, not the
 * new guard cell) is left entirely to the interior function -- see the
 * module comment above. */
DW_HOST_DEVICE void FUNC(backward_sigmazz_z_cell_guard)(
    DW_DTYPE const* __restrict const buoyancy_z_shot,
    DW_DTYPE const* __restrict const vz, DW_DTYPE* __restrict const sigmazz,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const nz,
    int64_t const ny, int64_t const nx, int64_t const z, DW_DTYPE const rdz) {
  sigmazz[i] += -DW_DIFFZ1_SAFE(SIGMAZZ_Z);
}
#endif

#if DW_NDIM >= 2
/* Guard-band companion to the sigmayy sub-update inside FUNC(backward_
 * sigmaii_cell) (see FUNC(backward_sigmazz_z_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sigmayy_y_cell_guard)(
    DW_DTYPE const* __restrict const buoyancy_y_shot,
    DW_DTYPE const* __restrict const vy, DW_DTYPE* __restrict const sigmayy,
    DW_DTYPE const dt, int64_t const i, int64_t const j, int64_t const ny,
    int64_t const nx, int64_t const y, DW_DTYPE const rdy) {
  sigmayy[i] += -DW_DIFFY1_SAFE(SIGMAYY_Y);
}
#endif

/* Guard-band companion to the sigmaxx sub-update inside FUNC(backward_
 * sigmaii_cell) (see FUNC(backward_sigmazz_z_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sigmaxx_x_cell_guard)(
    DW_DTYPE const* __restrict const buoyancy_x_shot,
    DW_DTYPE const* __restrict const vx, DW_DTYPE* __restrict const sigmaxx,
    DW_DTYPE const dt, int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const ny,
#endif
    int64_t const nx, int64_t const x, DW_DTYPE const rdx) {
  sigmaxx[i] += -DW_DIFFX1_SAFE(SIGMAXX_X);
}

#if DW_NDIM >= 2
/* Guard-band companion to FUNC(backward_sigmaxy_cell)'s gather (see
 * FUNC(backward_vz_cell_guard) -- like the velocity guards, this keeps the
 * PML memory-variable read/write intact since sigmaxy's guard ring may be
 * genuinely PML-active on its *other* axis). `ny` is required
 * unconditionally here (unlike FUNC(backward_sigmaxy_cell), which only
 * needs it for ND_INDEX when DW_NDIM==3) purely for DW_SAFE_TAP_Y's bounds
 * check. */
DW_HOST_DEVICE void FUNC(backward_sigmaxy_cell_guard)(
    DW_DTYPE const* __restrict const c_xy_shot,
    DW_DTYPE const* __restrict const buoyancy_x_shot,
    DW_DTYPE const* __restrict const buoyancy_y_shot,
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vy,
    DW_DTYPE* __restrict const sigmaxy, DW_DTYPE* __restrict const m_vxy,
    DW_DTYPE* __restrict const m_vyx,
    DW_DTYPE const* __restrict const m_sigmaxyy,
    DW_DTYPE const* __restrict const m_sigmaxyx, DW_DTYPE const dt,
    int64_t const i, int64_t const j,
#if DW_NDIM >= 3
    int64_t const nz, int64_t const z,
#endif
    int64_t const ny, int64_t const nx, int64_t const y, int64_t const x,
    bool const pml_y, bool const pml_x, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const byh, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const bxh, DW_DTYPE const rdy,
    DW_DTYPE const rdx) {
  if (pml_y) {
    m_vxy[i] = C_XY(0, 0, 0) * dt * P_AYH(0) * sigmaxy[i] + P_AYH(0) * m_vxy[i];
  }
  if (pml_x) {
    m_vyx[i] = C_XY(0, 0, 0) * dt * P_AXH(0) * sigmaxy[i] + P_AXH(0) * m_vyx[i];
  }

  sigmaxy[i] +=
      (pml_y ? -DW_DIFFYH1_SAFE(SIGMAXY_Y_PML) : -DW_DIFFYH1_SAFE(SIGMAXY_Y)) +
      (pml_x ? -DW_DIFFXH1_SAFE(SIGMAXY_X_PML) : -DW_DIFFXH1_SAFE(SIGMAXY_X));
}
#endif

#if DW_NDIM >= 3
/* Guard-band companion to FUNC(backward_sigmaxz_cell)'s gather (see
 * FUNC(backward_sigmaxy_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sigmaxz_cell_guard)(
    DW_DTYPE const* __restrict const c55_zx_shot,
    DW_DTYPE const* __restrict const buoyancy_x_shot,
    DW_DTYPE const* __restrict const buoyancy_z_shot,
    DW_DTYPE const* __restrict const vx, DW_DTYPE const* __restrict const vz,
    DW_DTYPE* __restrict const sigmaxz, DW_DTYPE* __restrict const m_vxz,
    DW_DTYPE* __restrict const m_vzx,
    DW_DTYPE const* __restrict const m_sigmaxzz,
    DW_DTYPE const* __restrict const m_sigmaxzx, DW_DTYPE const dt,
    int64_t const i, int64_t const j, int64_t const nz, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, bool const pml_x, DW_DTYPE const* __restrict const az,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const bzh, DW_DTYPE const* __restrict const ax,
    DW_DTYPE const* __restrict const axh, DW_DTYPE const* __restrict const bx,
    DW_DTYPE const* __restrict const bxh, DW_DTYPE const rdz,
    DW_DTYPE const rdx) {
  if (pml_z) {
    m_vxz[i] =
        C55_ZX(0, 0, 0) * dt * P_AZH(0) * sigmaxz[i] + P_AZH(0) * m_vxz[i];
  }
  if (pml_x) {
    m_vzx[i] =
        C55_ZX(0, 0, 0) * dt * P_AXH(0) * sigmaxz[i] + P_AXH(0) * m_vzx[i];
  }

  sigmaxz[i] +=
      (pml_z ? -DW_DIFFZH1_SAFE(SIGMAXZ_Z_PML) : -DW_DIFFZH1_SAFE(SIGMAXZ_Z)) +
      (pml_x ? -DW_DIFFXH1_SAFE(SIGMAXZ_X_PML) : -DW_DIFFXH1_SAFE(SIGMAXZ_X));
}

/* Guard-band companion to FUNC(backward_sigmayz_cell)'s gather (see
 * FUNC(backward_sigmaxy_cell_guard)). */
DW_HOST_DEVICE void FUNC(backward_sigmayz_cell_guard)(
    DW_DTYPE const* __restrict const c55_zy_shot,
    DW_DTYPE const* __restrict const buoyancy_y_shot,
    DW_DTYPE const* __restrict const buoyancy_z_shot,
    DW_DTYPE const* __restrict const vy, DW_DTYPE const* __restrict const vz,
    DW_DTYPE* __restrict const sigmayz, DW_DTYPE* __restrict const m_vyz,
    DW_DTYPE* __restrict const m_vzy,
    DW_DTYPE const* __restrict const m_sigmayzz,
    DW_DTYPE const* __restrict const m_sigmayzy, DW_DTYPE const dt,
    int64_t const i, int64_t const j, int64_t const nz, int64_t const ny,
    int64_t const nx, int64_t const z, int64_t const y, int64_t const x,
    bool const pml_z, bool const pml_y, DW_DTYPE const* __restrict const az,
    DW_DTYPE const* __restrict const azh, DW_DTYPE const* __restrict const bz,
    DW_DTYPE const* __restrict const bzh, DW_DTYPE const* __restrict const ay,
    DW_DTYPE const* __restrict const ayh, DW_DTYPE const* __restrict const by,
    DW_DTYPE const* __restrict const byh, DW_DTYPE const rdz,
    DW_DTYPE const rdy) {
  if (pml_z) {
    m_vyz[i] =
        C55_ZY(0, 0, 0) * dt * P_AZH(0) * sigmayz[i] + P_AZH(0) * m_vyz[i];
  }
  if (pml_y) {
    m_vzy[i] =
        C55_ZY(0, 0, 0) * dt * P_AYH(0) * sigmayz[i] + P_AYH(0) * m_vzy[i];
  }

  sigmayz[i] +=
      (pml_z ? -DW_DIFFZH1_SAFE(SIGMAYZ_Z_PML) : -DW_DIFFZH1_SAFE(SIGMAYZ_Z)) +
      (pml_y ? -DW_DIFFYH1_SAFE(SIGMAYZ_Y_PML) : -DW_DIFFYH1_SAFE(SIGMAYZ_Y));
}
#endif

#endif /* DW_ELASTIC_VTI_KERNELS_H */
