#ifndef ELASTIC_LINE_BC_DEV_H
#define ELASTIC_LINE_BC_DEV_H

/* CUDA device-function counterparts of elastic_line_bc.h (see that header).
 * fill_p_bc_line_adjoint_dev is byte-identical across acoustic_vti.cu,
 * elastic.cu, elastic_born.cu, elastic_vti.cu; the rest are byte-identical
 * (modulo comments) across elastic.cu, elastic_born.cu, elastic_vti.cu
 * only (acoustic_vti has no onrow/offrow/shear split). Single-threaded per
 * line (each runs on one CUDA thread), so the write-then-read ordering
 * within each function is race-free. */

__device__ inline void fill_p_bc_line_adjoint_dev(
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

__device__ inline void fill_elastic_onrow_velocity_line_dev(
    DW_DTYPE *__restrict const v, int64_t const base, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  if (bc_lo == 2) v[base + P * stride] = 0;
  if (bc_hi == 2) v[base + (nax - P) * stride] = 0;
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) {
      if (bc_lo == 1) {
        v[base + g * stride] = 0;
      } else if (bc_lo == 3) {
        v[base + g * stride] = v[base + (m - 1 + g) * stride];
      } else {
        v[base + g * stride] = -v[base + (2 * P - g) * stride];
      }
    }
  }
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      if (bc_hi == 1) {
        v[base + xh * stride] = 0;
      } else if (bc_hi == 3) {
        v[base + xh * stride] = v[base + (P + 1 + j) * stride];
      } else {
        v[base + xh * stride] = -v[base + (P + m - 2 - j) * stride];
      }
    }
  }
}

__device__ inline void fill_elastic_offrow_velocity_line_dev(
    DW_DTYPE *__restrict const v, int64_t const base, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  if (bc_hi == 2) {
    v[base + (nax - P) * stride] = 0;
  } else if (bc_hi == 3) {
    v[base + (nax - P) * stride] = v[base + P * stride];
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) {
      if (bc_lo == 1) {
        v[base + g * stride] = 0;
      } else if (bc_lo == 3) {
        v[base + g * stride] = v[base + (m - 1 + g) * stride];
      } else {
        if (g == P - 1) {
          v[base + g * stride] = 0;
        } else {
          v[base + g * stride] = -v[base + (2 * P - 2 - g) * stride];
        }
      }
    }
  }
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      if (bc_hi == 1) {
        v[base + xh * stride] = 0;
      } else if (bc_hi == 3) {
        v[base + xh * stride] = v[base + (P + 1 + j) * stride];
      } else {
        v[base + xh * stride] = -v[base + (P + m - 2 - j) * stride];
      }
    }
  }
}

__device__ inline void fill_elastic_shear_line_dev(
    DW_DTYPE *__restrict const s, int64_t const base, int64_t const nax,
    int64_t const stride, int64_t const bc_lo, int64_t const bc_hi) {
  int64_t const P = FD_PAD;
  int64_t const m = nax - 2 * P + 1;
  /* Off-row fields (like a velocity, see fill_elastic_offrow_velocity_line_
   * dev's matching step) have one redundant high-side interior node under a
   * periodic domain of period m - 1; slave it to node P before it is read
   * as a wrap source below (mirrors elastic.c's/elastic_vti.c's own
   * FUNC(fill_elastic_shear_line) fix -- see that file's comment for the
   * full derivation/citation). Single-threaded per line (this whole
   * function runs on one CUDA thread), so the write-then-read ordering
   * within this function is race-free. */
  if (bc_hi == 3) {
    s[base + (nax - P) * stride] = s[base + P * stride];
  }
  if (bc_lo != 0) {
    for (int64_t g = 0; g < P; ++g) {
      if (bc_lo == 3) {
        s[base + g * stride] = s[base + (m - 1 + g) * stride];
      } else if (bc_lo == 1) {
        s[base + g * stride] = -s[base + (2 * P - 1 - g) * stride];
      } else {
        s[base + g * stride] = s[base + (2 * P - 1 - g) * stride];
      }
    }
  }
  if (bc_hi != 0) {
    for (int64_t j = 0; j < P - 1; ++j) {
      int64_t const xh = nax - P + 1 + j;
      if (bc_hi == 3) {
        s[base + xh * stride] = s[base + (P + 1 + j) * stride];
      } else if (bc_hi == 1) {
        s[base + xh * stride] = -s[base + (P + m - 3 - j) * stride];
      } else {
        s[base + xh * stride] = s[base + (P + m - 2 - j) * stride];
      }
    }
  }
}

__device__ inline void fill_elastic_onrow_velocity_line_adjoint_dev(
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

__device__ inline void fill_elastic_offrow_velocity_line_adjoint_dev(
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

__device__ inline void fill_elastic_shear_line_adjoint_dev(
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
      } else {
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
      } else {
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

#endif /* ELASTIC_LINE_BC_DEV_H */
