#ifndef LINE_BC_H
#define LINE_BC_H

/* First-class reflecting/periodic boundary faces on the staggered regular
 * grid (see docs/boundaries.rst and the validated eager reference
 * implementations ``fill_acoustic_pressure_boundary`` /
 * ``fill_acoustic_velocity_boundary`` in common.py, which these mirror
 * exactly). ``bc`` codes: 0=pml (untouched), 1=free (pressure release),
 * 2=rigid (hard wall), 3=periodic. Each axis has ``P = FD_PAD`` guard cells
 * on the low side and ``P - 1`` on the high side; the pressure/velocity
 * interior along an axis of length ``nax`` is the ``m = nax - 2P + 1`` nodes
 * ``[P, nax-P]``. These fills touch only guard cells (and, for the
 * documented on-surface/on-wall cases, specific interior nodes); the model
 * mirror-pad the adjoint needs is applied separately (in Python, on fresh
 * model clones) because the models are not guaranteed fresh temporaries.
 *
 * Byte-identical (md5-confirmed) across every propagator whose pressure/
 * velocity fields live on this same staggered layout -- acoustic,
 * acoustic_born, acoustic_vti, viscoacoustic, elastic, elastic_born,
 * elastic_vti (fill_p_bc_line; fill_v_bc_line is used by the non-elastic
 * four, since the elastic family's velocity fill is split across
 * fill_elastic_onrow_velocity_line/fill_elastic_offrow_velocity_line/
 * fill_elastic_shear_line instead). Included after each file's own
 * FUNC()/DW_DTYPE/FD_PAD definitions are in scope, exactly like the
 * *_kernels.h headers this follows the pattern of. */

/* Names for the ``bc`` codes described above, as passed down from Python in
 * the bc_z0 ... bc_x1 arguments and stored in the drivers' face-order bc[6]
 * arrays [z0, z1, y0, y1, x0, x1]. They live here, with the per-line fills
 * that implement the rules, so that every propagator on this layout shares one
 * spelling of the vocabulary. */
#define DW_BC_PML 0      /* absorbing: guard cells untouched by these fills */
#define DW_BC_FREE 1     /* pressure release: p odd about the surface node */
#define DW_BC_RIGID 2    /* hard wall: v odd, p even about the wall */
#define DW_BC_PERIODIC 3 /* wrap-around */

/* Refill the pressure guard ring of one line (free also zeros the on-surface
 * pressure node, an interior node). */
static inline void FUNC(fill_p_bc_line)(DW_DTYPE* __restrict const p,
                                        int64_t const base, int64_t const nax,
                                        int64_t const stride,
                                        int64_t const bc_lo,
                                        int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1; /* pressure interior node count */
  /* Free surfaces hold the on-surface pressure node itself to zero. */
  if (bc_lo == 1) p[base + P * stride] = 0;         /* node P */
  if (bc_hi == 1) p[base + (nax - P) * stride] = 0; /* node n-P (= P+m-1) */
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) {
      int64_t src;
      DW_DTYPE sign = 1;
      if (bc_lo == 3) {
        src = m - 1 + g; /* periodic wrap */
      } else if (bc_lo == 2) {
        src = 2 * P - 1 - g; /* rigid: even about the half-cell P-1/2 */
      } else {               /* free: odd about the surface node P */
        src = 2 * P - g;
        sign = -1;
      }
      p[base + g * stride] = sign * p[base + src * stride];
    }
  }
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j; /* high guard cell */
      int64_t src;
      DW_DTYPE sign = 1;
      if (bc_hi == 3) {
        src = P + 1 + j; /* periodic wrap */
      } else if (bc_hi == 2) {
        src = P + m - 1 - j; /* rigid: even about (n-P)+1/2 */
      } else {               /* free: odd about the surface node n-P */
        src = P + m - 2 - j;
        sign = -1;
      }
      p[base + xh * stride] = sign * p[base + src * stride];
    }
  }
}

/* Refill the boundary-normal velocity guard ring of one line. The rigid high
 * wall node and the periodic extra high node are interior nodes fixed first. */
static inline void FUNC(fill_v_bc_line)(DW_DTYPE* __restrict const v,
                                        int64_t const base, int64_t const nax,
                                        int64_t const stride,
                                        int64_t const bc_lo,
                                        int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1; /* velocity interior node count */
  /* The rigid high-face wall velocity node (interior) is held to zero; the
   * periodic extra high interior node wraps to node P. */
  if (bc_hi == 2) {
    v[base + (nax - P) * stride] = 0; /* node P+m-1 = n-P */
  } else if (bc_hi == 3) {
    v[base + (nax - P) * stride] = v[base + P * stride];
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) {
      int64_t src;
      DW_DTYPE sign = 1;
      int zero = 0;
      if (bc_lo == 3) {
        src = m - 1 + g; /* periodic wrap */
      } else if (bc_lo == 1) {
        src = 2 * P - 1 - g; /* free: even about the surface node P */
      } else {               /* rigid: odd about P-1/2; on-plane guard is 0 */
        if (g == P - 1) {
          zero = 1;
          src = 0;
        } else {
          src = 2 * P - 2 - g;
          sign = -1;
        }
      }
      v[base + g * stride] = zero ? (DW_DTYPE)0 : sign * v[base + src * stride];
    }
  }
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j; /* high guard cell */
      int64_t src;
      DW_DTYPE sign = 1;
      if (bc_hi == 3) {
        src = P + 1 + j; /* periodic wrap */
      } else if (bc_hi == 1) {
        src = P + m - 3 - j; /* free: even about the surface node n-P */
      } else {               /* rigid: odd about the wall node n-P */
        src = P + m - 2 - j;
        sign = -1;
      }
      v[base + xh * stride] = sign * v[base + src * stride];
    }
  }
}

#endif /* LINE_BC_H */
