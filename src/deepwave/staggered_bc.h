#ifndef DW_STAGGERED_BC_H
#define DW_STAGGERED_BC_H

/* staggered_bc.h
 *
 * Whole-face boundary-condition machinery for the propagators whose pressure
 * and velocity fields live on the staggered acoustic grid: the boundary-code
 * vocabulary, and the face APPLIERS that drive line_bc.h's and adj_line_bc.h's
 * per-line kernels over every face of the grid.
 *
 * The split is: line_bc.h/adj_line_bc.h know what to do to ONE line (the image
 * rule and its exact transpose); this header knows WHICH lines to do it to (the
 * enumeration over faces, the axis strides, and the interior node ranges that
 * receive adjoint-stencil contributions); the driver .c file knows WHEN, in the
 * time-step order, each has to happen. Only the last of those three is
 * genuinely propagator-specific, which is why the first two are shared files
 * rather than copies. The enumeration is the part that is easiest to get subtly
 * wrong and hardest to notice: folds run over EVERY line, gathers only over the
 * other axes' interior node ranges [FD_PAD, n - FD_PAD + 1).
 *
 * Included by acoustic.c and viscoacoustic.c, after that file has defined
 * FUNC()/DW_DTYPE/FD_PAD, has included line_bc.h and its propagator's
 * *_kernels.h (which supplies adj_line_bc.h and the
 * FUNC(adj_gather_v_guard_line) / FUNC(adj_gather_v_guard_line_fr) per-line
 * kernels), and has defined the five DW_VGUARD_ macros below -- exactly the
 * include contract line_bc.h and the *_kernels.h headers already use. The
 * DW_BC_PML/FREE/RIGID/PERIODIC face codes used throughout come from
 * line_bc.h, alongside the per-line fills that implement them.
 *
 * PAYLOAD CONTRACT. Every applier here is propagator-independent except the two
 * adjoint-VELOCITY guard gathers, whose per-line kernel needs the propagator's
 * effective pressure-adjoint field: dt*K*Pbar for acoustic, but the SLS-coupled
 * B_EFF = dt*(m1 - c2/2)*Pbar + c2*Rbar for viscoacoustic (see
 * viscoacoustic.c's header comment). The including file therefore names that
 * payload:
 *
 *   DW_VGUARD_MODEL_PARAMS     the parameters carrying it, as declared here
 *   DW_VGUARD_MODEL_ARGS       those parameters' names, forwarded to the
 *                              per-line kernel
 *   DW_VGUARD_MODEL_SHOT_ARGS  the DRIVER's shot-local names for the same
 *                              fields, used by the ADJ_GATHER_V_GUARDS and
 *                              ADJ_GATHER_V_GUARDS_FRX dispatch macros below
 *   DW_VGUARD_EXTRA_PARAMS     trailing propagator extras, each written with a
 *   DW_VGUARD_EXTRA_ARGS       leading comma, or empty (viscoacoustic's
 *                              multi-mechanism GSLS build passes its mechanism
 *                              geometry here; acoustic defines both empty)
 *
 * PARAMETER NAMING. The appliers take their axis lengths as nz_/ny_/nx_ and
 * their reciprocal spacings and time step as rdz_/rdy_/rdx_/dt_, with a
 * trailing underscore, because in fewer than three dimensions the dispatch
 * macros pass a literal 1 for the absent axes rather than a real grid length:
 * the underscore marks them as possibly-synthetic stand-ins, distinct from the
 * driver's nz/ny/nx. The (void) casts at the top of each applier absorb the
 * ones a given DW_NDIM does not read.
 *
 * DISPATCH MACROS. The FILL_ and ADJ_ macros at the end of this header wrap
 * each applier with the driver's dimension-appropriate argument list, so a
 * driver writes "if (has_bc) FILL_P_BC(p_shot);" once instead of three
 * conditionally-compiled calls. They expand inside FUNC(forward)/FUNC(backward)
 * and read these names from the caller's scope: p_shot, vz_shot/vy_shot/
 * vx_shot, buoyancy_z_shot/buoyancy_y_shot/buoyancy_x_shot, the
 * DW_VGUARD_MODEL_SHOT_ARGS fields, bc, bc_fr, bc_frx, nz/ny/nx, rdz/rdy/rdx,
 * dt, and stag_w. That capture is deliberate -- it is what keeps the call sites
 * readable in a function whose every argument list is already shredded by
 * dimension conditionals -- and the list above is its contract.
 *
 * A driver that carries MORE THAN ONE field set on the grid cannot use those
 * macros, because each one names a single set's arrays: acoustic_born, for
 * instance, has both a background and a scattered (p, v_d) field and must
 * drive the appliers over each in turn, composing a Born-coupled payload as
 * two single-payload gather calls into the same target. Such a driver defines
 * DW_STAGGERED_BC_CUSTOM_DISPATCH before including this header and supplies
 * its own set-parameterised macros; it still shares every applier above,
 * which is where the face enumeration -- the part worth not duplicating --
 * actually lives.
 */

/* Apply the pressure guard fill on every reflecting axis. Viscoacoustic
 * applies this to its SLS memory variable too -- the memory variable
 * accompanies the pressure and takes the pressure's image rule, it has no
 * stencil of its own (see the note above viscoacoustic.c's include of this
 * header). */
static void FUNC(fill_p_bc)(DW_DTYPE *__restrict const p,
                            int64_t const *__restrict const bc,
                            int64_t const nz_, int64_t const ny_,
                            int64_t const nx_) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[4] != DW_BC_PML || bc[5] != DW_BC_PML)
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy)
          FUNC(fill_p_bc_line)(p, (zz * ny_ + yy) * nx_, nx_, 1, bc[4], bc[5]);
    if (bc[2] != DW_BC_PML || bc[3] != DW_BC_PML)
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(fill_p_bc_line)(p, zz * ny_ * nx_ + xx, ny_, nx_, bc[2], bc[3]);
    if (bc[0] != DW_BC_PML || bc[1] != DW_BC_PML)
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(fill_p_bc_line)(p, yy * nx_ + xx, nz_, ny_ * nx_, bc[0], bc[1]);
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bc[4] != DW_BC_PML || bc[5] != DW_BC_PML)
      for (yy = 0; yy < ny_; ++yy)
        FUNC(fill_p_bc_line)(p, yy * nx_, nx_, 1, bc[4], bc[5]);
    if (bc[2] != DW_BC_PML || bc[3] != DW_BC_PML)
      for (xx = 0; xx < nx_; ++xx)
        FUNC(fill_p_bc_line)(p, xx, ny_, nx_, bc[2], bc[3]);
  }
#else
  FUNC(fill_p_bc_line)(p, 0, nx_, 1, bc[4], bc[5]);
#endif
}

/* Apply the boundary-normal velocity guard fill: only ``v[d]`` is imaged along
 * axis ``d`` (tangential velocities and corner cells are never read). */
static void FUNC(fill_v_bc)(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx, int64_t const *__restrict const bc,
    int64_t const nz_, int64_t const ny_, int64_t const nx_) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[4] != DW_BC_PML || bc[5] != DW_BC_PML)
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy)
          FUNC(fill_v_bc_line)(vx, (zz * ny_ + yy) * nx_, nx_, 1, bc[4], bc[5]);
    if (bc[2] != DW_BC_PML || bc[3] != DW_BC_PML)
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(fill_v_bc_line)(vy, zz * ny_ * nx_ + xx, ny_, nx_, bc[2], bc[3]);
    if (bc[0] != DW_BC_PML || bc[1] != DW_BC_PML)
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(fill_v_bc_line)(vz, yy * nx_ + xx, nz_, ny_ * nx_, bc[0], bc[1]);
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bc[4] != DW_BC_PML || bc[5] != DW_BC_PML)
      for (yy = 0; yy < ny_; ++yy)
        FUNC(fill_v_bc_line)(vx, yy * nx_, nx_, 1, bc[4], bc[5]);
    if (bc[2] != DW_BC_PML || bc[3] != DW_BC_PML)
      for (xx = 0; xx < nx_; ++xx)
        FUNC(fill_v_bc_line)(vy, xx, ny_, nx_, bc[2], bc[3]);
  }
#else
  FUNC(fill_v_bc_line)(vx, 0, nx_, 1, bc[4], bc[5]);
#endif
}

/* Periodic-face adjoint guard maintenance (backward only): apply the
 * per-line fold/gather functions from adj_line_bc.h (pulled in by the
 * propagator's *_kernels.h) on every periodic axis. The folds run over every
 * line (like the fills); the gathers run only over the other axes' interior
 * node ranges [FD_PAD, n - FD_PAD + 1), the only lines whose guard cells
 * receive adjoint-stencil contributions. */
static void FUNC(adj_fold_p_bc)(DW_DTYPE *__restrict const p,
                                int64_t const *__restrict const bc,
                                int64_t const nz_, int64_t const ny_,
                                int64_t const nx_) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[4] == DW_BC_PERIODIC)
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy)
          FUNC(adj_fold_p_line)(p, (zz * ny_ + yy) * nx_, nx_, 1);
    if (bc[2] == DW_BC_PERIODIC)
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(adj_fold_p_line)(p, zz * ny_ * nx_ + xx, ny_, nx_);
    if (bc[0] == DW_BC_PERIODIC)
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(adj_fold_p_line)(p, yy * nx_ + xx, nz_, ny_ * nx_);
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bc[4] == DW_BC_PERIODIC)
      for (yy = 0; yy < ny_; ++yy) FUNC(adj_fold_p_line)(p, yy * nx_, nx_, 1);
    if (bc[2] == DW_BC_PERIODIC)
      for (xx = 0; xx < nx_; ++xx) FUNC(adj_fold_p_line)(p, xx, ny_, nx_);
  }
#else
  if (bc[4] == DW_BC_PERIODIC) FUNC(adj_fold_p_line)(p, 0, nx_, 1);
#endif
}

static void FUNC(adj_fold_v_bc)(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx, int64_t const *__restrict const bc,
    int64_t const nz_, int64_t const ny_, int64_t const nx_) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[4] == DW_BC_PERIODIC)
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy)
          FUNC(adj_fold_v_line)(vx, (zz * ny_ + yy) * nx_, nx_, 1);
    if (bc[2] == DW_BC_PERIODIC)
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(adj_fold_v_line)(vy, zz * ny_ * nx_ + xx, ny_, nx_);
    if (bc[0] == DW_BC_PERIODIC)
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(adj_fold_v_line)(vz, yy * nx_ + xx, nz_, ny_ * nx_);
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bc[4] == DW_BC_PERIODIC)
      for (yy = 0; yy < ny_; ++yy) FUNC(adj_fold_v_line)(vx, yy * nx_, nx_, 1);
    if (bc[2] == DW_BC_PERIODIC)
      for (xx = 0; xx < nx_; ++xx) FUNC(adj_fold_v_line)(vy, xx, ny_, nx_);
  }
#else
  if (bc[4] == DW_BC_PERIODIC) FUNC(adj_fold_v_line)(vx, 0, nx_, 1);
#endif
}

static void FUNC(adj_gather_v_guards)(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx, DW_DTYPE const *__restrict const p,
    DW_VGUARD_MODEL_PARAMS, int64_t const *__restrict const bc,
    int64_t const nz_, int64_t const ny_, int64_t const nx_,
#if DW_NDIM >= 3
    DW_DTYPE const rdz_,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy_,
#endif
    DW_DTYPE const rdx_, DW_DTYPE const dt_,
    DW_DTYPE const *__restrict const sw DW_VGUARD_EXTRA_PARAMS) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[4] == DW_BC_PERIODIC)
      for (zz = FD_PAD; zz < nz_ - FD_PAD + 1; ++zz)
        for (yy = FD_PAD; yy < ny_ - FD_PAD + 1; ++yy) {
          int64_t const b_ = (zz * ny_ + yy) * nx_;
          FUNC(adj_gather_v_guard_line)(vx, p, DW_VGUARD_MODEL_ARGS, b_, b_,
                                        nx_, 1, dt_, rdx_,
                                        sw DW_VGUARD_EXTRA_ARGS);
        }
    if (bc[2] == DW_BC_PERIODIC)
      for (zz = FD_PAD; zz < nz_ - FD_PAD + 1; ++zz)
        for (xx = FD_PAD; xx < nx_ - FD_PAD + 1; ++xx) {
          int64_t const b_ = zz * ny_ * nx_ + xx;
          FUNC(adj_gather_v_guard_line)(vy, p, DW_VGUARD_MODEL_ARGS, b_, b_,
                                        ny_, nx_, dt_, rdy_,
                                        sw DW_VGUARD_EXTRA_ARGS);
        }
    if (bc[0] == DW_BC_PERIODIC)
      for (yy = FD_PAD; yy < ny_ - FD_PAD + 1; ++yy)
        for (xx = FD_PAD; xx < nx_ - FD_PAD + 1; ++xx) {
          int64_t const b_ = yy * nx_ + xx;
          FUNC(adj_gather_v_guard_line)(vz, p, DW_VGUARD_MODEL_ARGS, b_, b_,
                                        nz_, ny_ * nx_, dt_, rdz_,
                                        sw DW_VGUARD_EXTRA_ARGS);
        }
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bc[4] == DW_BC_PERIODIC)
      for (yy = FD_PAD; yy < ny_ - FD_PAD + 1; ++yy)
        FUNC(adj_gather_v_guard_line)(vx, p, DW_VGUARD_MODEL_ARGS, yy * nx_,
                                      yy * nx_, nx_, 1, dt_, rdx_,
                                      sw DW_VGUARD_EXTRA_ARGS);
    if (bc[2] == DW_BC_PERIODIC)
      for (xx = FD_PAD; xx < nx_ - FD_PAD + 1; ++xx)
        FUNC(adj_gather_v_guard_line)(vy, p, DW_VGUARD_MODEL_ARGS, xx, xx, ny_,
                                      nx_, dt_, rdy_, sw DW_VGUARD_EXTRA_ARGS);
  }
#else
  if (bc[4] == DW_BC_PERIODIC)
    FUNC(adj_gather_v_guard_line)(vx, p, DW_VGUARD_MODEL_ARGS, 0, 0, nx_, 1,
                                  dt_, rdx_, sw DW_VGUARD_EXTRA_ARGS);
#endif
}

static void FUNC(adj_gather_p_guards)(
    DW_DTYPE *__restrict const p,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
    DW_DTYPE const *__restrict const buoyancy_z_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy,
    DW_DTYPE const *__restrict const buoyancy_y_shot,
#endif
    DW_DTYPE const *__restrict const vx,
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    int64_t const *__restrict const bc, int64_t const nz_, int64_t const ny_,
    int64_t const nx_,
#if DW_NDIM >= 3
    DW_DTYPE const rdz_,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy_,
#endif
    DW_DTYPE const rdx_, DW_DTYPE const dt_,
    DW_DTYPE const *__restrict const sw) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bc[4] == DW_BC_PERIODIC)
      for (zz = FD_PAD; zz < nz_ - FD_PAD + 1; ++zz)
        for (yy = FD_PAD; yy < ny_ - FD_PAD + 1; ++yy) {
          int64_t const b_ = (zz * ny_ + yy) * nx_;
          FUNC(adj_gather_p_guard_line)(p, vx, buoyancy_x_shot, b_, b_, nx_, 1,
                                        dt_, rdx_, sw);
        }
    if (bc[2] == DW_BC_PERIODIC)
      for (zz = FD_PAD; zz < nz_ - FD_PAD + 1; ++zz)
        for (xx = FD_PAD; xx < nx_ - FD_PAD + 1; ++xx) {
          int64_t const b_ = zz * ny_ * nx_ + xx;
          FUNC(adj_gather_p_guard_line)(p, vy, buoyancy_y_shot, b_, b_, ny_,
                                        nx_, dt_, rdy_, sw);
        }
    if (bc[0] == DW_BC_PERIODIC)
      for (yy = FD_PAD; yy < ny_ - FD_PAD + 1; ++yy)
        for (xx = FD_PAD; xx < nx_ - FD_PAD + 1; ++xx) {
          int64_t const b_ = yy * nx_ + xx;
          FUNC(adj_gather_p_guard_line)(p, vz, buoyancy_z_shot, b_, b_, nz_,
                                        ny_ * nx_, dt_, rdz_, sw);
        }
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bc[4] == DW_BC_PERIODIC)
      for (yy = FD_PAD; yy < ny_ - FD_PAD + 1; ++yy)
        FUNC(adj_gather_p_guard_line)(p, vx, buoyancy_x_shot, yy * nx_,
                                      yy * nx_, nx_, 1, dt_, rdx_, sw);
    if (bc[2] == DW_BC_PERIODIC)
      for (xx = FD_PAD; xx < nx_ - FD_PAD + 1; ++xx)
        FUNC(adj_gather_p_guard_line)(p, vy, buoyancy_y_shot, xx, xx, ny_, nx_,
                                      dt_, rdy_, sw);
  }
#else
  if (bc[4] == DW_BC_PERIODIC)
    FUNC(adj_gather_p_guard_line)(p, vx, buoyancy_x_shot, 0, 0, nx_, 1, dt_,
                                  rdx_, sw);
#endif
}

/* Free/rigid-face exact-adjoint guard maintenance (backward only): apply the
 * per-line fr fold/gather functions from adj_line_bc.h on every axis
 * whose faces need the exact transpose (bcx holds the codes of those axes'
 * faces, zero elsewhere). Line enumeration matches the periodic appliers:
 * folds over every line, gathers over the other axes' interior node ranges
 * [FD_PAD, n - FD_PAD + 1). */
static void FUNC(adj_fold_p_bc_frx)(DW_DTYPE *__restrict const p,
                                    int64_t const *__restrict const bcx,
                                    int64_t const nz_, int64_t const ny_,
                                    int64_t const nx_) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bcx[4] != DW_BC_PML || bcx[5] != DW_BC_PML)
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy)
          FUNC(adj_fold_p_line_fr)(p, (zz * ny_ + yy) * nx_, nx_, 1, bcx[4],
                                   bcx[5]);
    if (bcx[2] != DW_BC_PML || bcx[3] != DW_BC_PML)
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(adj_fold_p_line_fr)(p, zz * ny_ * nx_ + xx, ny_, nx_, bcx[2],
                                   bcx[3]);
    if (bcx[0] != DW_BC_PML || bcx[1] != DW_BC_PML)
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(adj_fold_p_line_fr)(p, yy * nx_ + xx, nz_, ny_ * nx_, bcx[0],
                                   bcx[1]);
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bcx[4] != DW_BC_PML || bcx[5] != DW_BC_PML)
      for (yy = 0; yy < ny_; ++yy)
        FUNC(adj_fold_p_line_fr)(p, yy * nx_, nx_, 1, bcx[4], bcx[5]);
    if (bcx[2] != DW_BC_PML || bcx[3] != DW_BC_PML)
      for (xx = 0; xx < nx_; ++xx)
        FUNC(adj_fold_p_line_fr)(p, xx, ny_, nx_, bcx[2], bcx[3]);
  }
#else
  if (bcx[4] != DW_BC_PML || bcx[5] != DW_BC_PML)
    FUNC(adj_fold_p_line_fr)(p, 0, nx_, 1, bcx[4], bcx[5]);
#endif
}

static void FUNC(adj_fold_v_bc_frx)(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx, int64_t const *__restrict const bcx,
    int64_t const nz_, int64_t const ny_, int64_t const nx_) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bcx[4] != DW_BC_PML || bcx[5] != DW_BC_PML)
      for (zz = 0; zz < nz_; ++zz)
        for (yy = 0; yy < ny_; ++yy)
          FUNC(adj_fold_v_line_fr)(vx, (zz * ny_ + yy) * nx_, nx_, 1, bcx[4],
                                   bcx[5]);
    if (bcx[2] != DW_BC_PML || bcx[3] != DW_BC_PML)
      for (zz = 0; zz < nz_; ++zz)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(adj_fold_v_line_fr)(vy, zz * ny_ * nx_ + xx, ny_, nx_, bcx[2],
                                   bcx[3]);
    if (bcx[0] != DW_BC_PML || bcx[1] != DW_BC_PML)
      for (yy = 0; yy < ny_; ++yy)
        for (xx = 0; xx < nx_; ++xx)
          FUNC(adj_fold_v_line_fr)(vz, yy * nx_ + xx, nz_, ny_ * nx_, bcx[0],
                                   bcx[1]);
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bcx[4] != DW_BC_PML || bcx[5] != DW_BC_PML)
      for (yy = 0; yy < ny_; ++yy)
        FUNC(adj_fold_v_line_fr)(vx, yy * nx_, nx_, 1, bcx[4], bcx[5]);
    if (bcx[2] != DW_BC_PML || bcx[3] != DW_BC_PML)
      for (xx = 0; xx < nx_; ++xx)
        FUNC(adj_fold_v_line_fr)(vy, xx, ny_, nx_, bcx[2], bcx[3]);
  }
#else
  if (bcx[4] != DW_BC_PML || bcx[5] != DW_BC_PML)
    FUNC(adj_fold_v_line_fr)(vx, 0, nx_, 1, bcx[4], bcx[5]);
#endif
}

static void FUNC(adj_gather_v_guards_frx)(
#if DW_NDIM >= 3
    DW_DTYPE *__restrict const vz,
#endif
#if DW_NDIM >= 2
    DW_DTYPE *__restrict const vy,
#endif
    DW_DTYPE *__restrict const vx, DW_DTYPE const *__restrict const p,
    DW_VGUARD_MODEL_PARAMS, int64_t const *__restrict const bcx,
    int64_t const nz_, int64_t const ny_, int64_t const nx_,
#if DW_NDIM >= 3
    DW_DTYPE const rdz_,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy_,
#endif
    DW_DTYPE const rdx_, DW_DTYPE const dt_,
    DW_DTYPE const *__restrict const sw DW_VGUARD_EXTRA_PARAMS) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bcx[4] != DW_BC_PML || bcx[5] != DW_BC_PML)
      for (zz = FD_PAD; zz < nz_ - FD_PAD + 1; ++zz)
        for (yy = FD_PAD; yy < ny_ - FD_PAD + 1; ++yy) {
          int64_t const b_ = (zz * ny_ + yy) * nx_;
          FUNC(adj_gather_v_guard_line_fr)(vx, p, DW_VGUARD_MODEL_ARGS, b_, b_,
                                           nx_, 1, dt_, rdx_, sw, bcx[4],
                                           bcx[5] DW_VGUARD_EXTRA_ARGS);
        }
    if (bcx[2] != DW_BC_PML || bcx[3] != DW_BC_PML)
      for (zz = FD_PAD; zz < nz_ - FD_PAD + 1; ++zz)
        for (xx = FD_PAD; xx < nx_ - FD_PAD + 1; ++xx) {
          int64_t const b_ = zz * ny_ * nx_ + xx;
          FUNC(adj_gather_v_guard_line_fr)(vy, p, DW_VGUARD_MODEL_ARGS, b_, b_,
                                           ny_, nx_, dt_, rdy_, sw, bcx[2],
                                           bcx[3] DW_VGUARD_EXTRA_ARGS);
        }
    if (bcx[0] != DW_BC_PML || bcx[1] != DW_BC_PML)
      for (yy = FD_PAD; yy < ny_ - FD_PAD + 1; ++yy)
        for (xx = FD_PAD; xx < nx_ - FD_PAD + 1; ++xx) {
          int64_t const b_ = yy * nx_ + xx;
          FUNC(adj_gather_v_guard_line_fr)(vz, p, DW_VGUARD_MODEL_ARGS, b_, b_,
                                           nz_, ny_ * nx_, dt_, rdz_, sw,
                                           bcx[0], bcx[1] DW_VGUARD_EXTRA_ARGS);
        }
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bcx[4] != DW_BC_PML || bcx[5] != DW_BC_PML)
      for (yy = FD_PAD; yy < ny_ - FD_PAD + 1; ++yy)
        FUNC(adj_gather_v_guard_line_fr)(vx, p, DW_VGUARD_MODEL_ARGS, yy * nx_,
                                         yy * nx_, nx_, 1, dt_, rdx_, sw,
                                         bcx[4], bcx[5] DW_VGUARD_EXTRA_ARGS);
    if (bcx[2] != DW_BC_PML || bcx[3] != DW_BC_PML)
      for (xx = FD_PAD; xx < nx_ - FD_PAD + 1; ++xx)
        FUNC(adj_gather_v_guard_line_fr)(vy, p, DW_VGUARD_MODEL_ARGS, xx, xx,
                                         ny_, nx_, dt_, rdy_, sw, bcx[2],
                                         bcx[3] DW_VGUARD_EXTRA_ARGS);
  }
#else
  if (bcx[4] != DW_BC_PML || bcx[5] != DW_BC_PML)
    FUNC(adj_gather_v_guard_line_fr)(vx, p, DW_VGUARD_MODEL_ARGS, 0, 0, nx_, 1,
                                     dt_, rdx_, sw, bcx[4],
                                     bcx[5] DW_VGUARD_EXTRA_ARGS);
#endif
}

static void FUNC(adj_gather_p_guards_frx)(
    DW_DTYPE *__restrict const p,
#if DW_NDIM >= 3
    DW_DTYPE const *__restrict const vz,
    DW_DTYPE const *__restrict const buoyancy_z_shot,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const *__restrict const vy,
    DW_DTYPE const *__restrict const buoyancy_y_shot,
#endif
    DW_DTYPE const *__restrict const vx,
    DW_DTYPE const *__restrict const buoyancy_x_shot,
    int64_t const *__restrict const bcx, int64_t const nz_, int64_t const ny_,
    int64_t const nx_,
#if DW_NDIM >= 3
    DW_DTYPE const rdz_,
#endif
#if DW_NDIM >= 2
    DW_DTYPE const rdy_,
#endif
    DW_DTYPE const rdx_, DW_DTYPE const dt_,
    DW_DTYPE const *__restrict const sw) {
  (void)nz_;
  (void)ny_;
#if DW_NDIM == 3
  {
    int64_t zz, yy, xx;
    if (bcx[4] != DW_BC_PML || bcx[5] != DW_BC_PML)
      for (zz = FD_PAD; zz < nz_ - FD_PAD + 1; ++zz)
        for (yy = FD_PAD; yy < ny_ - FD_PAD + 1; ++yy) {
          int64_t const b_ = (zz * ny_ + yy) * nx_;
          FUNC(adj_gather_p_guard_line_fr)(p, vx, buoyancy_x_shot, b_, b_, nx_,
                                           1, dt_, rdx_, sw, bcx[4], bcx[5]);
        }
    if (bcx[2] != DW_BC_PML || bcx[3] != DW_BC_PML)
      for (zz = FD_PAD; zz < nz_ - FD_PAD + 1; ++zz)
        for (xx = FD_PAD; xx < nx_ - FD_PAD + 1; ++xx) {
          int64_t const b_ = zz * ny_ * nx_ + xx;
          FUNC(adj_gather_p_guard_line_fr)(p, vy, buoyancy_y_shot, b_, b_, ny_,
                                           nx_, dt_, rdy_, sw, bcx[2], bcx[3]);
        }
    if (bcx[0] != DW_BC_PML || bcx[1] != DW_BC_PML)
      for (yy = FD_PAD; yy < ny_ - FD_PAD + 1; ++yy)
        for (xx = FD_PAD; xx < nx_ - FD_PAD + 1; ++xx) {
          int64_t const b_ = yy * nx_ + xx;
          FUNC(adj_gather_p_guard_line_fr)(p, vz, buoyancy_z_shot, b_, b_, nz_,
                                           ny_ * nx_, dt_, rdz_, sw, bcx[0],
                                           bcx[1]);
        }
  }
#elif DW_NDIM == 2
  {
    int64_t yy, xx;
    if (bcx[4] != DW_BC_PML || bcx[5] != DW_BC_PML)
      for (yy = FD_PAD; yy < ny_ - FD_PAD + 1; ++yy)
        FUNC(adj_gather_p_guard_line_fr)(p, vx, buoyancy_x_shot, yy * nx_,
                                         yy * nx_, nx_, 1, dt_, rdx_, sw,
                                         bcx[4], bcx[5]);
    if (bcx[2] != DW_BC_PML || bcx[3] != DW_BC_PML)
      for (xx = FD_PAD; xx < nx_ - FD_PAD + 1; ++xx)
        FUNC(adj_gather_p_guard_line_fr)(p, vy, buoyancy_y_shot, xx, xx, ny_,
                                         nx_, dt_, rdy_, sw, bcx[2], bcx[3]);
  }
#else
  if (bcx[4] != DW_BC_PML || bcx[5] != DW_BC_PML)
    FUNC(adj_gather_p_guard_line_fr)(p, vx, buoyancy_x_shot, 0, 0, nx_, 1, dt_,
                                     rdx_, sw, bcx[4], bcx[5]);
#endif
}

#ifndef DW_STAGGERED_BC_CUSTOM_DISPATCH
#if DW_NDIM == 3
#define FILL_P_BC(p) FUNC(fill_p_bc)(p, bc, nz, ny, nx)
#define FILL_V_BC() FUNC(fill_v_bc)(vz_shot, vy_shot, vx_shot, bc, nz, ny, nx)
#define FILL_P_BC_FR(p) FUNC(fill_p_bc)(p, bc_fr, nz, ny, nx)
#define FILL_V_BC_FR() \
  FUNC(fill_v_bc)(vz_shot, vy_shot, vx_shot, bc_fr, nz, ny, nx)
#define ADJ_FOLD_P_BC(p) FUNC(adj_fold_p_bc)(p, bc, nz, ny, nx)
#define ADJ_FOLD_V_BC() \
  FUNC(adj_fold_v_bc)(vz_shot, vy_shot, vx_shot, bc, nz, ny, nx)
#define ADJ_GATHER_V_GUARDS()                                               \
  FUNC(adj_gather_v_guards)(vz_shot, vy_shot, vx_shot, p_shot,              \
                            DW_VGUARD_MODEL_SHOT_ARGS, bc, nz, ny, nx, rdz, \
                            rdy, rdx, dt, stag_w DW_VGUARD_EXTRA_ARGS)
#define ADJ_GATHER_P_GUARDS()                                                  \
  FUNC(adj_gather_p_guards)(p_shot, vz_shot, buoyancy_z_shot, vy_shot,         \
                            buoyancy_y_shot, vx_shot, buoyancy_x_shot, bc, nz, \
                            ny, nx, rdz, rdy, rdx, dt, stag_w)
#define ADJ_FOLD_P_BC_FRX(p) FUNC(adj_fold_p_bc_frx)(p, bc_frx, nz, ny, nx)
#define ADJ_FOLD_V_BC_FRX() \
  FUNC(adj_fold_v_bc_frx)(vz_shot, vy_shot, vx_shot, bc_frx, nz, ny, nx)
#define ADJ_GATHER_V_GUARDS_FRX()                                           \
  FUNC(adj_gather_v_guards_frx)(                                            \
      vz_shot, vy_shot, vx_shot, p_shot, DW_VGUARD_MODEL_SHOT_ARGS, bc_frx, \
      nz, ny, nx, rdz, rdy, rdx, dt, stag_w DW_VGUARD_EXTRA_ARGS)
#define ADJ_GATHER_P_GUARDS_FRX()                                          \
  FUNC(adj_gather_p_guards_frx)(p_shot, vz_shot, buoyancy_z_shot, vy_shot, \
                                buoyancy_y_shot, vx_shot, buoyancy_x_shot, \
                                bc_frx, nz, ny, nx, rdz, rdy, rdx, dt, stag_w)
#elif DW_NDIM == 2
#define FILL_P_BC(p) FUNC(fill_p_bc)(p, bc, 1, ny, nx)
#define FILL_V_BC() FUNC(fill_v_bc)(vy_shot, vx_shot, bc, 1, ny, nx)
#define FILL_P_BC_FR(p) FUNC(fill_p_bc)(p, bc_fr, 1, ny, nx)
#define FILL_V_BC_FR() FUNC(fill_v_bc)(vy_shot, vx_shot, bc_fr, 1, ny, nx)
#define ADJ_FOLD_P_BC(p) FUNC(adj_fold_p_bc)(p, bc, 1, ny, nx)
#define ADJ_FOLD_V_BC() FUNC(adj_fold_v_bc)(vy_shot, vx_shot, bc, 1, ny, nx)
#define ADJ_GATHER_V_GUARDS()                                              \
  FUNC(adj_gather_v_guards)(vy_shot, vx_shot, p_shot,                      \
                            DW_VGUARD_MODEL_SHOT_ARGS, bc, 1, ny, nx, rdy, \
                            rdx, dt, stag_w DW_VGUARD_EXTRA_ARGS)
#define ADJ_GATHER_P_GUARDS()                                             \
  FUNC(adj_gather_p_guards)(p_shot, vy_shot, buoyancy_y_shot, vx_shot,    \
                            buoyancy_x_shot, bc, 1, ny, nx, rdy, rdx, dt, \
                            stag_w)
#define ADJ_FOLD_P_BC_FRX(p) FUNC(adj_fold_p_bc_frx)(p, bc_frx, 1, ny, nx)
#define ADJ_FOLD_V_BC_FRX() \
  FUNC(adj_fold_v_bc_frx)(vy_shot, vx_shot, bc_frx, 1, ny, nx)
#define ADJ_GATHER_V_GUARDS_FRX()                                             \
  FUNC(adj_gather_v_guards_frx)(vy_shot, vx_shot, p_shot,                     \
                                DW_VGUARD_MODEL_SHOT_ARGS, bc_frx, 1, ny, nx, \
                                rdy, rdx, dt, stag_w DW_VGUARD_EXTRA_ARGS)
#define ADJ_GATHER_P_GUARDS_FRX()                                             \
  FUNC(adj_gather_p_guards_frx)(p_shot, vy_shot, buoyancy_y_shot, vx_shot,    \
                                buoyancy_x_shot, bc_frx, 1, ny, nx, rdy, rdx, \
                                dt, stag_w)
#else
#define FILL_P_BC(p) FUNC(fill_p_bc)(p, bc, 1, 1, nx)
#define FILL_V_BC() FUNC(fill_v_bc)(vx_shot, bc, 1, 1, nx)
#define FILL_P_BC_FR(p) FUNC(fill_p_bc)(p, bc_fr, 1, 1, nx)
#define FILL_V_BC_FR() FUNC(fill_v_bc)(vx_shot, bc_fr, 1, 1, nx)
#define ADJ_FOLD_P_BC(p) FUNC(adj_fold_p_bc)(p, bc, 1, 1, nx)
#define ADJ_FOLD_V_BC() FUNC(adj_fold_v_bc)(vx_shot, bc, 1, 1, nx)
#define ADJ_GATHER_V_GUARDS()                                                  \
  FUNC(adj_gather_v_guards)(vx_shot, p_shot, DW_VGUARD_MODEL_SHOT_ARGS, bc, 1, \
                            1, nx, rdx, dt, stag_w DW_VGUARD_EXTRA_ARGS)
#define ADJ_GATHER_P_GUARDS()                                               \
  FUNC(adj_gather_p_guards)(p_shot, vx_shot, buoyancy_x_shot, bc, 1, 1, nx, \
                            rdx, dt, stag_w)
#define ADJ_FOLD_P_BC_FRX(p) FUNC(adj_fold_p_bc_frx)(p, bc_frx, 1, 1, nx)
#define ADJ_FOLD_V_BC_FRX() FUNC(adj_fold_v_bc_frx)(vx_shot, bc_frx, 1, 1, nx)
#define ADJ_GATHER_V_GUARDS_FRX()                                           \
  FUNC(adj_gather_v_guards_frx)(vx_shot, p_shot, DW_VGUARD_MODEL_SHOT_ARGS, \
                                bc_frx, 1, 1, nx, rdx, dt,                  \
                                stag_w DW_VGUARD_EXTRA_ARGS)
#define ADJ_GATHER_P_GUARDS_FRX()                                            \
  FUNC(adj_gather_p_guards_frx)(p_shot, vx_shot, buoyancy_x_shot, bc_frx, 1, \
                                1, nx, rdx, dt, stag_w)
#endif
#endif /* DW_STAGGERED_BC_CUSTOM_DISPATCH */

#endif /* DW_STAGGERED_BC_H */
