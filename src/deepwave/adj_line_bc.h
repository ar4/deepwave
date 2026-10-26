#ifndef ADJ_LINE_BC_H
#define ADJ_LINE_BC_H

/* ---- Backward: periodic-face adjoint guard maintenance ---- */
/*
 * On a periodic axis the forward guard fill copies interior wrap images into
 * the guard cells (and slaves the extra high interior velocity node to node
 * FD_PAD), so its exact transpose is a FOLD: each guard cell's adjoint value
 * is added to the interior node the forward copied it from, then the guard
 * cell is zeroed. Unlike the free/rigid faces (where re-applying the forward
 * fill to the adjoint field realises the transpose through the mirror-padded
 * models), the fold writes into interior seam nodes, so the backward drivers
 * must (a) apply the pressure fold before the source-gradient read, (b)
 * accumulate the buoyancy gradient only after the velocity fold (the fused
 * accumulation in FUNC(backward_v*_cell) is suppressed on periodic axes),
 * and (c) first accumulate, via the *_guard_line gather functions, the
 * adjoint-stencil contributions that fall on the guard cells themselves
 * (out-of-range taps are zero: the adjoint fields are zero outside their
 * interior nodes once folded). The line functions below are shared by the
 * CPU and CUDA drivers of every propagator on this staggered layout, which
 * supply their own line enumeration, exactly like the *_cell functions in
 * each propagator's own *_kernels.h.
 *
 * Byte-identical (md5-confirmed) across acoustic_kernels.h/
 * acoustic_born_kernels.h/viscoacoustic_kernels.h -- the only one of this
 * family that is NOT shared here is the pressure-side counterpart's velocity
 * sibling, FUNC(adj_gather_v_guard_line)[_fr]: viscoacoustic's version
 * couples in the SLS memory-variable/relaxation terms (attenuation is a
 * velocity-equation term only), so it stays local to
 * viscoacoustic_kernels.h rather than being parameterised here.
 */

/* Transpose of the periodic branch of fill_p_bc_line for one line. */
DW_HOST_DEVICE void FUNC(adj_fold_p_line)(DW_DTYPE* __restrict const p,
                                          int64_t const base, int64_t const nax,
                                          int64_t const stride) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  int64_t g, j;
  for (g = 0; g < P; ++g) {
    p[base + (m - 1 + g) * stride] += p[base + g * stride];
    p[base + g * stride] = 0;
  }
  for (j = 0; j < P - 1; ++j) {
    int64_t const xh = nax - P + 1 + j;
    p[base + (P + 1 + j) * stride] += p[base + xh * stride];
    p[base + xh * stride] = 0;
  }
}

/* Transpose of the periodic branch of fill_v_bc_line for one line (guard
 * cells plus the extra high interior node the forward slaves to node
 * FD_PAD). */
DW_HOST_DEVICE void FUNC(adj_fold_v_line)(DW_DTYPE* __restrict const v,
                                          int64_t const base, int64_t const nax,
                                          int64_t const stride) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  int64_t g, j;
  for (g = 0; g < P; ++g) {
    v[base + (m - 1 + g) * stride] += v[base + g * stride];
    v[base + g * stride] = 0;
  }
  for (j = 0; j < P - 1; ++j) {
    int64_t const xh = nax - P + 1 + j;
    v[base + (P + 1 + j) * stride] += v[base + xh * stride];
    v[base + xh * stride] = 0;
  }
  v[base + P * stride] += v[base + (nax - P) * stride];
  v[base + (nax - P) * stride] = 0;
}

/* Adjoint-pressure guard gather for one periodic-axis line: the own-axis
 * -DIFF*1(-dt*BUOYANCY*V) term of FUNC(backward_p_cell) evaluated at the
 * guard cells (the other axes' terms are zero there because the adjoint
 * velocities are zero on this axis's guard cells). */
DW_HOST_DEVICE void FUNC(adj_gather_p_guard_line)(
    DW_DTYPE* __restrict const p, DW_DTYPE const* __restrict const v,
    DW_DTYPE const* __restrict const b_shot, int64_t const base,
    int64_t const jbase, int64_t const nax, int64_t const stride,
    DW_DTYPE const dt, DW_DTYPE const rd, DW_DTYPE const* __restrict const sw) {
  int64_t const P = FD_PAD;
  int64_t c, q;
  for (c = 0; c < 2 * P - 1; ++c) {
    int64_t const y = (c < P) ? c : nax - 2 * P + 1 + c;
    DW_DTYPE acc = 0;
    for (q = 0; q < P; ++q) {
      int64_t const th = y + q;
      int64_t const tl = y - 1 - q;
      if (th < nax)
        acc += sw[q] * b_shot[jbase + th * stride] * v[base + th * stride];
      if (tl >= 0)
        acc -= sw[q] * b_shot[jbase + tl * stride] * v[base + tl * stride];
    }
    p[base + y * stride] += dt * rd * acc;
  }
}

/* ---- Backward: free/rigid-face adjoint guard maintenance ---- */
/*
 * Re-applying the forward guard fill to the adjoint fields (with
 * mirror-padded models) realises the transpose of the free/rigid fills ONLY
 * for the per-axis (free-low, rigid-high) pairing; every other free/rigid
 * combination needs the same exact fold/gather treatment as the periodic
 * faces above, with the mirror/sign-flip transposes of fill_p_bc_line /
 * fill_v_bc_line instead of the periodic wrap fold. The fold targets are the
 * forward fill's source indices with the forward's signs, applied in the
 * reverse of the forward's op order (high guards, then low guards, then the
 * transposes of the forward's leading on-surface/on-wall interior-node
 * writes). The gathers cover only the faces that are free/rigid; unlike the
 * periodic case the velocity's extra high interior node (n-P) needs no
 * gather: rigid-high zeroes its adjoint in the fold, and for a free/pml high
 * face it is dead (never updated by the forward, so nothing propagates
 * through it). The same ordering rules as periodic apply (pressure fold
 * before the source-gradient read; buoyancy gradient from the post-fold
 * adjoint velocity) because these folds also write interior nodes. */

/* Fold-then-zero transpose of the free/rigid branches of fill_p_bc_line
 * (bc codes 1=free, 2=rigid, 0=leave face alone). */
DW_HOST_DEVICE void FUNC(adj_fold_p_line_fr)(
    DW_DTYPE* __restrict const p, int64_t const base, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  int64_t g, j;
  if (bc_hi != 0) {
    for (j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      if (bc_hi == 2) { /* rigid: even about (n-P)+1/2 */
        p[base + (P + m - 1 - j) * stride] += p[base + xh * stride];
      } else { /* free: odd about the surface node n-P */
        p[base + (P + m - 2 - j) * stride] -= p[base + xh * stride];
      }
      p[base + xh * stride] = 0;
    }
  }
  if (bc_lo != 0) {
    for (g = 0; g < P; ++g) {
      if (bc_lo == 2) { /* rigid: even about the half-cell P-1/2 */
        p[base + (2 * P - 1 - g) * stride] += p[base + g * stride];
      } else { /* free: odd about the surface node P */
        p[base + (2 * P - g) * stride] -= p[base + g * stride];
      }
      p[base + g * stride] = 0;
    }
  }
  /* Transpose of the forward's on-surface pressure-node zeroing. */
  if (bc_hi == 1) p[base + (nax - P) * stride] = 0;
  if (bc_lo == 1) p[base + P * stride] = 0;
}

/* Fold-then-zero transpose of the free/rigid branches of fill_v_bc_line. */
DW_HOST_DEVICE void FUNC(adj_fold_v_line_fr)(
    DW_DTYPE* __restrict const v, int64_t const base, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  int64_t g, j;
  if (bc_hi != 0) {
    for (j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      if (bc_hi == 1) { /* free: even about the surface node n-P */
        v[base + (P + m - 3 - j) * stride] += v[base + xh * stride];
      } else { /* rigid: odd about the wall node n-P */
        v[base + (P + m - 2 - j) * stride] -= v[base + xh * stride];
      }
      v[base + xh * stride] = 0;
    }
  }
  if (bc_lo != 0) {
    for (g = 0; g < P; ++g) {
      if (bc_lo == 1) { /* free: even about the surface node P */
        v[base + (2 * P - 1 - g) * stride] += v[base + g * stride];
      } else if (g != P - 1) { /* rigid: odd about P-1/2 (on-plane guard 0) */
        v[base + (2 * P - 2 - g) * stride] -= v[base + g * stride];
      }
      v[base + g * stride] = 0;
    }
  }
  /* Transpose of the forward's rigid high-wall velocity-node zeroing. */
  if (bc_hi == 2) v[base + (nax - P) * stride] = 0;
}

/* Adjoint-pressure guard gather for one free/rigid-axis line (see
 * FUNC(adj_gather_p_guard_line) for the payload, which is identical). */
DW_HOST_DEVICE void FUNC(adj_gather_p_guard_line_fr)(
    DW_DTYPE* __restrict const p, DW_DTYPE const* __restrict const v,
    DW_DTYPE const* __restrict const b_shot, int64_t const base,
    int64_t const jbase, int64_t const nax, int64_t const stride,
    DW_DTYPE const dt, DW_DTYPE const rd, DW_DTYPE const* __restrict const sw,
    int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t c, q;
  for (c = 0; c < 2 * P - 1; ++c) {
    int64_t y;
    if (c < P) {
      if (bc_lo == 0) continue;
      y = c;
    } else {
      if (bc_hi == 0) continue;
      y = nax - 2 * P + 1 + c;
    }
    {
      DW_DTYPE acc = 0;
      for (q = 0; q < P; ++q) {
        int64_t const th = y + q;
        int64_t const tl = y - 1 - q;
        if (th < nax)
          acc += sw[q] * b_shot[jbase + th * stride] * v[base + th * stride];
        if (tl >= 0)
          acc -= sw[q] * b_shot[jbase + tl * stride] * v[base + tl * stride];
      }
      p[base + y * stride] += dt * rd * acc;
    }
  }
}

#endif /* ADJ_LINE_BC_H */
