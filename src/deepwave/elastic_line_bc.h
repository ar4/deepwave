#ifndef ELASTIC_LINE_BC_H
#define ELASTIC_LINE_BC_H

/* --- boundary=["free"/"rigid"/"periodic",...] guard-ring image fill,
 * elastic-family portion -------------------------------------------------
 *
 * Ports the flat-face free/rigid/periodic guard-ring image method (see
 * line_bc.h's FUNC(fill_p_bc_line)/FUNC(fill_v_bc_line)) to the elastic
 * staggered-grid field set (full stress tensor + velocity), shared verbatim
 * (md5-confirmed) across elastic.c, elastic_born.c, elastic_vti.c.
 * FUNC(fill_p_bc_line_adjoint) here is also shared with acoustic_vti.c
 * (whose sigma_v/sigma_h fill is byte-identical to elastic's sigma_AA
 * fill). Physics citations and the derivation of which field gets which
 * treatment live in docs/dev/boundary_conditions/
 * ROBERTSSON_FS_SUMMARY.md and docs/boundaries.rst's Policy section.
 *
 * Per axis A, exactly six wavefields are read across that face: sigma_AA
 * (on-row/lattice-coincident with the face, handled by line_bc.h's
 * FUNC(fill_p_bc_line)), the two shear stresses sharing axis A (off-row),
 * and every velocity component (v_A off-row on its own axis; the other
 * velocities on-row, sharing sigma_AA's lattice). The other three fields
 * (the other two normal stresses, and the shear stress not involving A)
 * never differentiate in the A direction anywhere in the elastic system, so
 * they need no guard fill on face A at all.
 */

/* On-row velocity (the two velocity components sharing axis A's lattice,
 * e.g. vy/vx on a z-face): free zeroes the ghost (not mirrored --
 * ROBERTSSON_FS_SUMMARY.md Sec 1.3 point 3, "particle velocities above the
 * surface are set to ZERO"); rigid clamps the on-row node to zero (the
 * standard elastodynamics fixed-wall condition) + odd images about it
 * (same source-index arithmetic as FUNC(fill_p_bc_line)'s own free branch,
 * role-swapped onto rigid here); periodic wraps (same arithmetic as
 * FUNC(fill_p_bc_line)'s periodic branch). */
static inline void FUNC(fill_elastic_onrow_velocity_line)(
    DW_DTYPE *__restrict const v, int64_t const base, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  if (bc_lo == 2) v[base + P * stride] = 0;         /* rigid clamp, node P */
  if (bc_hi == 2) v[base + (nax - P) * stride] = 0; /* rigid clamp, node n-P */
  if (bc_lo != 0) {
    int64_t g;
    for (g = 0; g < P; ++g) {
      if (bc_lo == 1) { /* free: ghost zeroed, not mirrored */
        v[base + g * stride] = 0;
      } else if (bc_lo == 3) { /* periodic wrap */
        v[base + g * stride] = v[base + (m - 1 + g) * stride];
      } else { /* rigid: odd about the (clamped) node P */
        v[base + g * stride] = -v[base + (2 * P - g) * stride];
      }
    }
  }
  if (bc_hi != 0) {
    int64_t j;
    for (j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      if (bc_hi == 1) {
        v[base + xh * stride] = 0;
      } else if (bc_hi == 3) {
        v[base + xh * stride] = v[base + (P + 1 + j) * stride];
      } else { /* rigid: odd about the (clamped) node n-P */
        v[base + xh * stride] = -v[base + (P + m - 2 - j) * stride];
      }
    }
  }
}

/* Off-row velocity (v_A itself, naturally staggered onto axis A's own
 * half-cell lattice, exactly like acoustic's own velocity convention):
 * free zeroes the ghost (Sec 1.3 point 3, same as the on-row case above --
 * this is IVM's own flat-surface behaviour too, Sec 2.1); rigid and
 * periodic are IDENTICAL to FUNC(fill_v_bc_line)'s own rigid/periodic
 * branches (the standard elastodynamics clamped-wall condition applied to
 * the one velocity component that is naturally staggered onto this axis's
 * half-cell plane -- no elastic-specific re-derivation needed there). */
static inline void FUNC(fill_elastic_offrow_velocity_line)(
    DW_DTYPE *__restrict const v, int64_t const base, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  if (bc_hi == 2) {
    v[base + (nax - P) * stride] = 0; /* rigid: wall node n-P is interior */
  } else if (bc_hi == 3) {
    v[base + (nax - P) * stride] = v[base + P * stride]; /* periodic slave */
  }
  if (bc_lo != 0) {
    int64_t g;
    for (g = 0; g < P; ++g) {
      if (bc_lo == 1) { /* free: ghost zeroed, not mirrored */
        v[base + g * stride] = 0;
      } else if (bc_lo == 3) { /* periodic wrap */
        v[base + g * stride] = v[base + (m - 1 + g) * stride];
      } else { /* rigid: odd about P-1/2; on-plane guard (g==P-1) is 0 */
        if (g == P - 1) {
          v[base + g * stride] = 0;
        } else {
          v[base + g * stride] = -v[base + (2 * P - 2 - g) * stride];
        }
      }
    }
  }
  if (bc_hi != 0) {
    int64_t j;
    for (j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      if (bc_hi == 1) {
        v[base + xh * stride] = 0;
      } else if (bc_hi == 3) {
        v[base + xh * stride] = v[base + (P + 1 + j) * stride];
      } else { /* rigid: odd about the wall node n-P */
        v[base + xh * stride] = -v[base + (P + m - 2 - j) * stride];
      }
    }
  }
}

/* Shared-axis shear stress (sigma_AB/sigma_AC, e.g. sigma_yz/sigma_xz on a
 * z-face): off-row like a velocity, but no on-plane node is ever forced to
 * zero (no shear-stress lattice point coincides with either wall plane).
 * free: ODD about the half-cell plane (ROBERTSSON_FS_SUMMARY.md Sec 1.3
 * point 1, "sigma_xz ... imaged as ODD"); rigid: EVEN about the half-cell
 * plane (the role-swapped dual: shear stress is unconstrained at a
 * clamped-velocity wall, the same "unconstrained" parity sigma_AA gets
 * under rigid, just on the half-cell lattice); periodic: wrap. Low-side
 * source index is the SAME for free/rigid (only the sign differs, matching
 * FUNC(fill_v_bc_line)'s own free-branch source -- reused for both here);
 * the high-side source index DIFFERS between free (matches
 * FUNC(fill_v_bc_line)'s free-branch high source) and rigid (matches its
 * rigid-branch high source, sign-flipped back to positive) -- this
 * asymmetry is inherited directly from FUNC(fill_v_bc_line)'s own
 * low/high staggered-pad asymmetry, not invented here. */
static inline void FUNC(fill_elastic_shear_line)(
    DW_DTYPE *__restrict const s, int64_t const base, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  /* Off-row fields (like a velocity, see FUNC(fill_elastic_offrow_velocity_
   * line)'s matching step) have one redundant high-side interior node under
   * a periodic domain of period m - 1; slave it to node P (the first
   * interior node) before it is read as a wrap source below -- otherwise
   * that node evolves as an independent (un-wrapped) extra degree of
   * freedom and periodic translation invariance drifts once the shear
   * stress becomes dynamically active (mirrors _boundaries.py's
   * fill_elastic_stress_boundary's own ``if hi == BC_PERIODIC`` slaving
   * step for shear_stresses). */
  if (bc_hi == 3) {
    s[base + (nax - P) * stride] = s[base + P * stride];
  }
  if (bc_lo != 0) {
    int64_t g;
    for (g = 0; g < P; ++g) {
      if (bc_lo == 3) {
        s[base + g * stride] = s[base + (m - 1 + g) * stride];
      } else if (bc_lo == 1) { /* free: odd about the half-cell P-1/2 */
        s[base + g * stride] = -s[base + (2 * P - 1 - g) * stride];
      } else { /* rigid: even about the half-cell P-1/2 */
        s[base + g * stride] = s[base + (2 * P - 1 - g) * stride];
      }
    }
  }
  if (bc_hi != 0) {
    int64_t j;
    for (j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      if (bc_hi == 3) {
        s[base + xh * stride] = s[base + (P + 1 + j) * stride];
      } else if (bc_hi == 1) { /* free: odd (same source as fill_v_bc_line's
                                * own free-branch high) */
        s[base + xh * stride] = -s[base + (P + m - 3 - j) * stride];
      } else { /* rigid: even (same source as fill_v_bc_line's own
                * rigid-branch high, sign flipped back to positive) */
        s[base + xh * stride] = s[base + (P + m - 2 - j) * stride];
      }
    }
  }
}

/* Transpose of FUNC(fill_p_bc_line) (shared with acoustic_vti.c, whose
 * sigma_v/sigma_h fill is byte-identical to elastic's sigma_AA fill). */
static inline void FUNC(fill_p_bc_line_adjoint)(
    DW_DTYPE *__restrict const p, int64_t const base, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      int64_t src;
      DW_DTYPE sign = 1;
      if (bc_hi == 3) {
        src = P + 1 + j;
      } else if (bc_hi == 2) {
        src = P + m - 1 - j;
      } else {
        src = P + m - 2 - j;
        sign = -1;
      }
      p[base + src * stride] += sign * p[base + xh * stride];
    }
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) {
      int64_t src;
      DW_DTYPE sign = 1;
      if (bc_lo == 3) {
        src = m - 1 + g;
      } else if (bc_lo == 2) {
        src = 2 * P - 1 - g;
      } else {
        src = 2 * P - g;
        sign = -1;
      }
      p[base + src * stride] += sign * p[base + g * stride];
    }
  }
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      p[base + (nax - P + 1 + j) * stride] = 0;
    }
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) p[base + g * stride] = 0;
  }
  if (bc_hi == 1) p[base + (nax - P) * stride] = 0;
  if (bc_lo == 1) p[base + P * stride] = 0;
}

/* Transpose of FUNC(fill_elastic_onrow_velocity_line). */
static inline void FUNC(fill_elastic_onrow_velocity_line_adjoint)(
    DW_DTYPE *__restrict const v, int64_t const base, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      if (bc_hi == 3) {
        v[base + (P + 1 + j) * stride] += v[base + xh * stride];
      } else if (bc_hi == 2) {
        v[base + (P + m - 2 - j) * stride] += -v[base + xh * stride];
      }
      /* bc_hi == 1 (free): ghost was a hard zero -- no source to fold. */
    }
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) {
      if (bc_lo == 3) {
        v[base + (m - 1 + g) * stride] += v[base + g * stride];
      } else if (bc_lo == 2) {
        v[base + (2 * P - g) * stride] += -v[base + g * stride];
      }
    }
  }
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      v[base + (nax - P + 1 + j) * stride] = 0;
    }
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) v[base + g * stride] = 0;
  }
  if (bc_hi == 2) v[base + (nax - P) * stride] = 0;
  if (bc_lo == 2) v[base + P * stride] = 0;
}

/* Transpose of FUNC(fill_elastic_offrow_velocity_line). */
static inline void FUNC(fill_elastic_offrow_velocity_line_adjoint)(
    DW_DTYPE *__restrict const v, int64_t const base, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      if (bc_hi == 3) {
        v[base + (P + 1 + j) * stride] += v[base + xh * stride];
      } else if (bc_hi == 2) {
        v[base + (P + m - 2 - j) * stride] += -v[base + xh * stride];
      }
    }
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) {
      if (bc_lo == 3) {
        v[base + (m - 1 + g) * stride] += v[base + g * stride];
      } else if (bc_lo == 2 && g != P - 1) {
        v[base + (2 * P - 2 - g) * stride] += -v[base + g * stride];
      }
      /* bc_lo == 1 (free), or the bc_lo == 2 on-plane guard cell
       * (g == P - 1): the forward ghost was a hard zero. */
    }
  }
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      v[base + (nax - P + 1 + j) * stride] = 0;
    }
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) v[base + g * stride] = 0;
  }
  if (bc_hi == 2) {
    v[base + (nax - P) * stride] = 0;
  } else if (bc_hi == 3) {
    v[base + P * stride] += v[base + (nax - P) * stride];
    v[base + (nax - P) * stride] = 0;
  }
}

/* Transpose of FUNC(fill_elastic_shear_line). */
static inline void FUNC(fill_elastic_shear_line_adjoint)(
    DW_DTYPE *__restrict const s, int64_t const base, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      if (bc_hi == 3) {
        s[base + (P + 1 + j) * stride] += s[base + xh * stride];
      } else if (bc_hi == 1) {
        s[base + (P + m - 3 - j) * stride] += -s[base + xh * stride];
      } else { /* rigid */
        s[base + (P + m - 2 - j) * stride] += s[base + xh * stride];
      }
    }
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) {
      if (bc_lo == 3) {
        s[base + (m - 1 + g) * stride] += s[base + g * stride];
      } else if (bc_lo == 1) {
        s[base + (2 * P - 1 - g) * stride] += -s[base + g * stride];
      } else { /* rigid */
        s[base + (2 * P - 1 - g) * stride] += s[base + g * stride];
      }
    }
  }
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      s[base + (nax - P + 1 + j) * stride] = 0;
    }
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) s[base + g * stride] = 0;
  }
  if (bc_hi == 3) {
    s[base + P * stride] += s[base + (nax - P) * stride];
    s[base + (nax - P) * stride] = 0;
  }
}

#endif /* ELASTIC_LINE_BC_H */
