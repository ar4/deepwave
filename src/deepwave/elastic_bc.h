#ifndef ELASTIC_BC_H
#define ELASTIC_BC_H

/* elastic_bc.h
 *
 * Whole-face guard-ring boundary appliers, and the pressure-trace helpers
 * that sit beside them, for the propagators whose fields are elastic's full
 * staggered-grid stress tensor + velocity set: the FUNC(fill_elastic_
 * stress_bc)/FUNC(fill_elastic_velocity_bc) forward fills, their exact
 * adjoint (transpose) counterparts, and add_pressure/record_pressure (not
 * name-mangled by FUNC() -- each is `static inline` and every DW_NDIM/
 * DW_ACCURACY/DW_DTYPE build variant of the including file is its own
 * translation unit, so no cross-variant symbol collision is possible).
 * Included by elastic.c, elastic_born.c, and elastic_vti.c,
 * after each of those files has defined FUNC()/DW_DTYPE/FD_PAD and included
 * line_bc.h (for FUNC(fill_p_bc_line)/FUNC(fill_p_bc_line_adjoint), which
 * handle sigma_AA -- the on-row, lattice-coincident-with-the-face
 * component) and elastic_line_bc.h (for FUNC(fill_elastic_onrow_velocity_
 * line)/FUNC(fill_elastic_offrow_velocity_line)/FUNC(fill_elastic_shear_
 * line) and their _adjoint transposes, which handle the off-row/shear
 * components) -- exactly the include contract those two headers already
 * use. This header knows WHICH lines to apply those per-line kernels to
 * (the enumeration over faces and the axis strides); the driver .c file
 * knows WHEN, in the time-step order, each has to happen.
 *
 * Ports the flat-face free/rigid/periodic guard-ring image method (see
 * line_bc.h's FUNC(fill_p_bc_line)/FUNC(fill_v_bc_line)) to elastic's full
 * stress tensor + velocity field set. Physics citations and the derivation
 * of which field gets which treatment live in docs/dev/boundary_conditions/
 * ROBERTSSON_FS_SUMMARY.md and docs/boundaries.rst's Policy section; this is
 * a line-for-line port of the already-validated Python arithmetic in
 * _boundaries.py's fill_elastic_stress_boundary/fill_elastic_velocity_
 * boundary (see that module's docstrings for the physics "why" of each
 * branch below).
 *
 * Per axis A, exactly six wavefields are read across that face: sigma_AA
 * (on-row/lattice-coincident with the face), the two shear stresses sharing
 * axis A (off-row), and every velocity component (v_A off-row on its own
 * axis; the other velocities on-row, sharing sigma_AA's lattice). The other
 * three fields (the other two normal stresses, and the shear stress not
 * involving A) never differentiate in the A direction anywhere in the
 * elastic system, so they need no guard fill on face A at all. This applies
 * identically to elastic_born's background and scattered fields (both obey
 * the same isotropic elastic system on the same staggered grid) and to
 * elastic_vti (the guard-fill index arithmetic is part of the UNCHANGED
 * staggered-grid/PML/free-surface skeleton -- see that file's own module
 * docstring -- only the constitutive stress-strain relation differs
 * elsewhere); elastic_vti has no DW_NDIM==1 build (CMakeLists.txt only
 * builds it for DW_NDIM 2 and 3), so the DW_NDIM==1 branches below are
 * compiled but never instantiated for it.
 *
 * --- Adjoint (transpose) of the boundary-face guard-ring fill ------------
 *
 * The compiled backward pass needs the discrete TRANSPOSE of each forward
 * fill above, not the fill itself. Unlike acoustic's single-field pressure/
 * velocity fill (a projection that happens to be self-adjoint, see
 * staggered_bc.h's own backward FILL_P_BC/FILL_V_BC reuse), elastic's
 * multi-field fills combine an on-surface zero (sigma_AA under free, v_A
 * under rigid), a sign-flipped mirror image, and (under periodic) a slaved
 * high-side interior node -- a combination whose adjoint is NOT the same
 * operation, so reapplying a forward fill verbatim to the adjoint field is
 * wrong.
 *
 * Mechanical derivation (matches the autograd semantics of
 * _boundaries.py's fill_elastic_stress_boundary/fill_elastic_velocity_
 * boundary): every forward assignment `dest = sign * src` has adjoint
 * `adj[src] += sign * adj[dest]`; every forward assignment `dest = 0` (a
 * hard zero -- no source) has no accumulation step. After all accumulations
 * for one line's fill, every cell the forward fill wrote (ghost cells,
 * on-plane zeros, slaved nodes) is zeroed -- those cells are not
 * independent degrees of freedom, so no adjoint value may remain on them
 * (the ghost <-> interior image map is injective -- see docs/boundaries.rst's
 * mirror-plane conventions -- so this scatter-add never has two ghost cells
 * racing to accumulate into the same interior cell within one line's fill).
 *
 * Two ordering requirements fall out of doing this faithfully (both mirror
 * how PyTorch's autograd would walk the forward graph backward through the
 * same sequence of tensor writes):
 *   - Within one line's fill, a later forward assignment may read a cell an
 *     earlier one wrote (the periodic high-side slaved node, written before
 *     the ghost wraps -- see fill_elastic_offrow_velocity_line's and
 *     fill_elastic_shear_line's own `bc_hi == 3` slave step); the adjoint
 *     below undoes assignments in the reverse of the forward order
 *     (high-face ghost wrap, then low-face ghost wrap, then the
 *     slaved-node/on-plane-zero step) so any such read is folded before its
 *     source is touched again.
 *   - Each stress/shear or velocity component is refilled once per
 *     reflecting axis it touches (e.g. sigma_xy on both an x-face and a
 *     y-face); the later face's forward fill overwrites the earlier face's
 *     corner ghost cells, so only the later face's fill is part of the true
 *     map at that corner. FUNC(fill_elastic_stress_bc_adjoint)/FUNC(fill_
 *     elastic_velocity_bc_adjoint) below process faces in the exact reverse
 *     of FUNC(fill_elastic_stress_bc)/FUNC(fill_elastic_velocity_bc)'s own
 *     face order, so the later (in forward time) face's adjoint runs first
 *     and zeros the corner; the earlier face's adjoint then finds that cell
 *     already zero and correctly contributes nothing there, without any
 *     explicit corner special-casing.
 *
 * --- Pressure-trace helpers -----------------------------------------------
 *
 * add_pressure/record_pressure inject/read a pressure source/receiver as
 * the trace of the normal-stress tensor (sigmaxx + sigmayy + sigmazz,
 * negated sign convention matching the acoustic pressure field); their
 * arithmetic does not depend on the boundary condition, but
 * they are shared verbatim across the same three files as the appliers
 * above, so they live in this header rather than a fourth copy.
 */

/* The four guard-ring appliers below are wrapped in this opt-out so that a
 * file whose local copies are functionally but not textually identical
 * (their own DW_NDIM==1-only-defensive (void) casts differ, since that
 * file never builds at DW_NDIM==1) can still share add_pressure/
 * record_pressure below without also taking on a byte-for-byte-different
 * copy of these four -- define DW_ELASTIC_BC_NO_FILLS before including
 * this header to keep your own local versions (see elastic_vti.c, and
 * clone_manifest.json for the pairing this is policed by). */
#ifndef DW_ELASTIC_BC_NO_FILLS

/* Apply the stress guard fill on every reflecting axis: sigma_AA (reusing
 * FUNC(fill_p_bc_line) verbatim) plus the two shear stresses sharing axis A
 * (FUNC(fill_elastic_shear_line)), for each axis with a non-PML face. */
static void FUNC(fill_elastic_stress_bc)(
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const sigmazz, DW_DTYPE* __restrict const sigmayz,
    DW_DTYPE* __restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const sigmayy, DW_DTYPE* __restrict const sigmaxy,
#endif
    DW_DTYPE* __restrict const sigmaxx, int64_t const* __restrict const bc,
    int64_t const nz_, int64_t const ny_, int64_t const nx_) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[4] != 0 || bc[5] != 0) /* x-face */
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy) {
          int64_t const base = (zz * ny_ + yy) * nx_;
          FUNC(fill_p_bc_line)(sigmaxx, base, nx_, 1, bc[4], bc[5]);
          FUNC(fill_elastic_shear_line)(sigmaxz, base, nx_, 1, bc[4], bc[5]);
          FUNC(fill_elastic_shear_line)(sigmaxy, base, nx_, 1, bc[4], bc[5]);
        }
    if (bc[2] != 0 || bc[3] != 0) /* y-face */
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = zz * ny_ * nx_ + xx;
          FUNC(fill_p_bc_line)(sigmayy, base, ny_, nx_, bc[2], bc[3]);
          FUNC(fill_elastic_shear_line)(sigmayz, base, ny_, nx_, bc[2], bc[3]);
          FUNC(fill_elastic_shear_line)(sigmaxy, base, ny_, nx_, bc[2], bc[3]);
        }
    if (bc[0] != 0 || bc[1] != 0) /* z-face */
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = yy * nx_ + xx;
          FUNC(fill_p_bc_line)(sigmazz, base, nz_, ny_ * nx_, bc[0], bc[1]);
          FUNC(fill_elastic_shear_line)(sigmayz, base, nz_, ny_ * nx_, bc[0],
                                        bc[1]);
          FUNC(fill_elastic_shear_line)(sigmaxz, base, nz_, ny_ * nx_, bc[0],
                                        bc[1]);
        }
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bc[4] != 0 || bc[5] != 0) /* x-face */
      for (yy = 0; yy < ny_; ++yy) {
        int64_t const base = yy * nx_;
        FUNC(fill_p_bc_line)(sigmaxx, base, nx_, 1, bc[4], bc[5]);
        FUNC(fill_elastic_shear_line)(sigmaxy, base, nx_, 1, bc[4], bc[5]);
      }
    if (bc[2] != 0 || bc[3] != 0) /* y-face */
      for (xx = 0; xx < nx_; ++xx) {
        int64_t const base = xx;
        FUNC(fill_p_bc_line)(sigmayy, base, ny_, nx_, bc[2], bc[3]);
        FUNC(fill_elastic_shear_line)(sigmaxy, base, ny_, nx_, bc[2], bc[3]);
      }
  }
#else
  FUNC(fill_p_bc_line)(sigmaxx, 0, nx_, 1, bc[4], bc[5]);
#endif
}

/* Apply the velocity guard fill on every reflecting axis: unlike acoustic
 * (where only the boundary-normal velocity is imaged), *every* velocity
 * component is touched on each reflecting face here -- each normal
 * stress's own divergence term reads every velocity's ghost region along
 * that axis (see the module comment above), and (for free) Robertsson's
 * point 3 zeroes "particle velocities" generically, not just the normal
 * component. */
static void FUNC(fill_elastic_velocity_bc)(
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy,
#endif
    DW_DTYPE* __restrict const vx, int64_t const* __restrict const bc,
    int64_t const nz_, int64_t const ny_, int64_t const nx_) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[4] != 0 || bc[5] != 0) /* x-face: vx off-row, vz/vy on-row */
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy) {
          int64_t const base = (zz * ny_ + yy) * nx_;
          FUNC(fill_elastic_offrow_velocity_line)(vx, base, nx_, 1, bc[4],
                                                  bc[5]);
          FUNC(fill_elastic_onrow_velocity_line)(vz, base, nx_, 1, bc[4],
                                                 bc[5]);
          FUNC(fill_elastic_onrow_velocity_line)(vy, base, nx_, 1, bc[4],
                                                 bc[5]);
        }
    if (bc[2] != 0 || bc[3] != 0) /* y-face: vy off-row, vz/vx on-row */
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = zz * ny_ * nx_ + xx;
          FUNC(fill_elastic_offrow_velocity_line)(vy, base, ny_, nx_, bc[2],
                                                  bc[3]);
          FUNC(fill_elastic_onrow_velocity_line)(vz, base, ny_, nx_, bc[2],
                                                 bc[3]);
          FUNC(fill_elastic_onrow_velocity_line)(vx, base, ny_, nx_, bc[2],
                                                 bc[3]);
        }
    if (bc[0] != 0 || bc[1] != 0) /* z-face: vz off-row, vy/vx on-row */
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = yy * nx_ + xx;
          FUNC(fill_elastic_offrow_velocity_line)(vz, base, nz_, ny_ * nx_,
                                                  bc[0], bc[1]);
          FUNC(fill_elastic_onrow_velocity_line)(vy, base, nz_, ny_ * nx_,
                                                 bc[0], bc[1]);
          FUNC(fill_elastic_onrow_velocity_line)(vx, base, nz_, ny_ * nx_,
                                                 bc[0], bc[1]);
        }
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bc[4] != 0 || bc[5] != 0) /* x-face: vx off-row, vy on-row */
      for (yy = 0; yy < ny_; ++yy) {
        int64_t const base = yy * nx_;
        FUNC(fill_elastic_offrow_velocity_line)(vx, base, nx_, 1, bc[4], bc[5]);
        FUNC(fill_elastic_onrow_velocity_line)(vy, base, nx_, 1, bc[4], bc[5]);
      }
    if (bc[2] != 0 || bc[3] != 0) /* y-face: vy off-row, vx on-row */
      for (xx = 0; xx < nx_; ++xx) {
        int64_t const base = xx;
        FUNC(fill_elastic_offrow_velocity_line)(vy, base, ny_, nx_, bc[2],
                                                bc[3]);
        FUNC(fill_elastic_onrow_velocity_line)(vx, base, ny_, nx_, bc[2],
                                               bc[3]);
      }
  }
#else
  FUNC(fill_elastic_offrow_velocity_line)(vx, 0, nx_, 1, bc[4], bc[5]);
#endif
}

/* Transpose of FUNC(fill_elastic_stress_bc): faces undone in the reverse of
 * that function's own face order (see the module comment above). */
static void FUNC(fill_elastic_stress_bc_adjoint)(
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const sigmazz, DW_DTYPE* __restrict const sigmayz,
    DW_DTYPE* __restrict const sigmaxz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const sigmayy, DW_DTYPE* __restrict const sigmaxy,
#endif
    DW_DTYPE* __restrict const sigmaxx, int64_t const* __restrict const bc,
    int64_t const nz_, int64_t const ny_, int64_t const nx_) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[0] != 0 || bc[1] != 0) /* z-face */
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = yy * nx_ + xx;
          FUNC(fill_p_bc_line_adjoint)(sigmazz, base, nz_, ny_ * nx_, bc[0],
                                       bc[1]);
          FUNC(fill_elastic_shear_line_adjoint)(sigmayz, base, nz_, ny_ * nx_,
                                                bc[0], bc[1]);
          FUNC(fill_elastic_shear_line_adjoint)(sigmaxz, base, nz_, ny_ * nx_,
                                                bc[0], bc[1]);
        }
    if (bc[2] != 0 || bc[3] != 0) /* y-face */
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = zz * ny_ * nx_ + xx;
          FUNC(fill_p_bc_line_adjoint)(sigmayy, base, ny_, nx_, bc[2], bc[3]);
          FUNC(fill_elastic_shear_line_adjoint)(sigmayz, base, ny_, nx_, bc[2],
                                                bc[3]);
          FUNC(fill_elastic_shear_line_adjoint)(sigmaxy, base, ny_, nx_, bc[2],
                                                bc[3]);
        }
    if (bc[4] != 0 || bc[5] != 0) /* x-face */
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy) {
          int64_t const base = (zz * ny_ + yy) * nx_;
          FUNC(fill_p_bc_line_adjoint)(sigmaxx, base, nx_, 1, bc[4], bc[5]);
          FUNC(fill_elastic_shear_line_adjoint)(sigmaxz, base, nx_, 1, bc[4],
                                                bc[5]);
          FUNC(fill_elastic_shear_line_adjoint)(sigmaxy, base, nx_, 1, bc[4],
                                                bc[5]);
        }
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bc[2] != 0 || bc[3] != 0) /* y-face */
      for (xx = 0; xx < nx_; ++xx) {
        int64_t const base = xx;
        FUNC(fill_p_bc_line_adjoint)(sigmayy, base, ny_, nx_, bc[2], bc[3]);
        FUNC(fill_elastic_shear_line_adjoint)(sigmaxy, base, ny_, nx_, bc[2],
                                              bc[3]);
      }
    if (bc[4] != 0 || bc[5] != 0) /* x-face */
      for (yy = 0; yy < ny_; ++yy) {
        int64_t const base = yy * nx_;
        FUNC(fill_p_bc_line_adjoint)(sigmaxx, base, nx_, 1, bc[4], bc[5]);
        FUNC(fill_elastic_shear_line_adjoint)(sigmaxy, base, nx_, 1, bc[4],
                                              bc[5]);
      }
  }
#else
  FUNC(fill_p_bc_line_adjoint)(sigmaxx, 0, nx_, 1, bc[4], bc[5]);
#endif
}

/* Transpose of FUNC(fill_elastic_velocity_bc): faces undone in the reverse
 * of that function's own face order (see the module comment above). */
static void FUNC(fill_elastic_velocity_bc_adjoint)(
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const vy,
#endif
    DW_DTYPE* __restrict const vx, int64_t const* __restrict const bc,
    int64_t const nz_, int64_t const ny_, int64_t const nx_) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[0] != 0 || bc[1] != 0) /* z-face */
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = yy * nx_ + xx;
          FUNC(fill_elastic_offrow_velocity_line_adjoint)(
              vz, base, nz_, ny_ * nx_, bc[0], bc[1]);
          FUNC(fill_elastic_onrow_velocity_line_adjoint)(
              vy, base, nz_, ny_ * nx_, bc[0], bc[1]);
          FUNC(fill_elastic_onrow_velocity_line_adjoint)(
              vx, base, nz_, ny_ * nx_, bc[0], bc[1]);
        }
    if (bc[2] != 0 || bc[3] != 0) /* y-face */
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx) {
          int64_t const base = zz * ny_ * nx_ + xx;
          FUNC(fill_elastic_offrow_velocity_line_adjoint)(vy, base, ny_, nx_,
                                                          bc[2], bc[3]);
          FUNC(fill_elastic_onrow_velocity_line_adjoint)(vz, base, ny_, nx_,
                                                         bc[2], bc[3]);
          FUNC(fill_elastic_onrow_velocity_line_adjoint)(vx, base, ny_, nx_,
                                                         bc[2], bc[3]);
        }
    if (bc[4] != 0 || bc[5] != 0) /* x-face */
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy) {
          int64_t const base = (zz * ny_ + yy) * nx_;
          FUNC(fill_elastic_offrow_velocity_line_adjoint)(vx, base, nx_, 1,
                                                          bc[4], bc[5]);
          FUNC(fill_elastic_onrow_velocity_line_adjoint)(vz, base, nx_, 1,
                                                         bc[4], bc[5]);
          FUNC(fill_elastic_onrow_velocity_line_adjoint)(vy, base, nx_, 1,
                                                         bc[4], bc[5]);
        }
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bc[2] != 0 || bc[3] != 0) /* y-face */
      for (xx = 0; xx < nx_; ++xx) {
        int64_t const base = xx;
        FUNC(fill_elastic_offrow_velocity_line_adjoint)(vy, base, ny_, nx_,
                                                        bc[2], bc[3]);
        FUNC(fill_elastic_onrow_velocity_line_adjoint)(vx, base, ny_, nx_,
                                                       bc[2], bc[3]);
      }
    if (bc[4] != 0 || bc[5] != 0) /* x-face */
      for (yy = 0; yy < ny_; ++yy) {
        int64_t const base = yy * nx_;
        FUNC(fill_elastic_offrow_velocity_line_adjoint)(vx, base, nx_, 1, bc[4],
                                                        bc[5]);
        FUNC(fill_elastic_onrow_velocity_line_adjoint)(vy, base, nx_, 1, bc[4],
                                                       bc[5]);
      }
  }
#else
  FUNC(fill_elastic_offrow_velocity_line_adjoint)(vx, 0, nx_, 1, bc[4], bc[5]);
#endif
}

#endif /* DW_ELASTIC_BC_NO_FILLS */

static inline void add_pressure(
#if DW_NDIM >= 3
    DW_DTYPE* __restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE* __restrict const sigmayy,
#endif
    DW_DTYPE* __restrict const sigmaxx,
    int64_t const* __restrict const sources_i,
    DW_DTYPE const* __restrict const f, int64_t const n_sources_per_shot) {
  int64_t source_idx;
#pragma omp simd
  for (source_idx = 0; source_idx < n_sources_per_shot; ++source_idx) {
    if (sources_i[source_idx] < 0) continue;
#if DW_NDIM >= 3
    sigmazz[sources_i[source_idx]] += f[source_idx];
#endif
#if DW_NDIM >= 2
    sigmayy[sources_i[source_idx]] += f[source_idx];
#endif
    sigmaxx[sources_i[source_idx]] += f[source_idx];
  }
}

static inline void record_pressure(
#if DW_NDIM >= 3
    DW_DTYPE const* __restrict const sigmazz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const* __restrict const sigmayy,
#endif
    DW_DTYPE const* __restrict const sigmaxx,
    int64_t const* __restrict const locations,
    DW_DTYPE* __restrict const amplitudes, int64_t n) {
  int64_t i;
#pragma omp simd
  for (i = 0; i < n; ++i) {
    if (0 <= locations[i])
#if DW_NDIM == 3
      amplitudes[i] = (sigmazz[locations[i]] + sigmayy[locations[i]] +
                       sigmaxx[locations[i]]);
#elif DW_NDIM == 2
      amplitudes[i] = (sigmayy[locations[i]] + sigmaxx[locations[i]]);
#else
      amplitudes[i] = sigmaxx[locations[i]];
#endif
  }
}

#endif /* ELASTIC_BC_H */
