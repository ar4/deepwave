#ifndef DW_STAGGERED_GRID_H
#define DW_STAGGERED_GRID_H

/* staggered_grid.h
 * Small collection of finite-difference helper macros used by CPU and GPU
 * implementations of staggered grid propagators. DIFF* macros implement spatial
 * FD operators; FD_PAD is the stencil 'radius' (number of guard cells required
 * to evaluate stencils).
 */

/* Each axis is swept as three separately-bounded regions -- low PML, interior,
 * high PML -- so that the PML memory-variable recursions are compiled away
 * entirely in the interior (the overwhelmingly common case) instead of being
 * branched on per grid point. The drivers nest one such loop per dimension and
 * pass "region != DW_PML_INTERIOR_REGION" to the per-cell kernels as that
 * axis's "in the PML" flag.
 *
 * Note that "#pragma GCC unroll" does not macro-expand its operand, so the
 * unroll counts next to these loops necessarily repeat the literal 3. */
#define DW_PML_REGIONS 3
#define DW_PML_INTERIOR_REGION 1

#if DW_ACCURACY == 2

#define DIFFZ1(a) ((a(0, 0, 0) - a(-1, 0, 0)) * rdz)
#define DIFFY1(a) ((a(0, 0, 0) - a(0, -1, 0)) * rdy)
#define DIFFX1(a) ((a(0, 0, 0) - a(0, 0, -1)) * rdx)
#define DIFFZH1(a) ((a(1, 0, 0) - a(0, 0, 0)) * rdz)
#define DIFFYH1(a) ((a(0, 1, 0) - a(0, 0, 0)) * rdy)
#define DIFFXH1(a) ((a(0, 0, 1) - a(0, 0, 0)) * rdx)
#define FD_PAD 1

#elif DW_ACCURACY == 4

#define DIFFZ1(a)                                           \
  (((DW_DTYPE)(9.0 / 8.0) * (a(0, 0, 0) - a(-1, 0, 0)) +    \
    (DW_DTYPE)(-1.0 / 24.0) * (a(1, 0, 0) - a(-2, 0, 0))) * \
   rdz)
#define DIFFY1(a)                                           \
  (((DW_DTYPE)(9.0 / 8.0) * (a(0, 0, 0) - a(0, -1, 0)) +    \
    (DW_DTYPE)(-1.0 / 24.0) * (a(0, 1, 0) - a(0, -2, 0))) * \
   rdy)
#define DIFFX1(a)                                           \
  (((DW_DTYPE)(9.0 / 8.0) * (a(0, 0, 0) - a(0, 0, -1)) +    \
    (DW_DTYPE)(-1.0 / 24.0) * (a(0, 0, 1) - a(0, 0, -2))) * \
   rdx)
#define DIFFZH1(a)                                          \
  (((DW_DTYPE)(9.0 / 8.0) * (a(1, 0, 0) - a(0, 0, 0)) +     \
    (DW_DTYPE)(-1.0 / 24.0) * (a(2, 0, 0) - a(-1, 0, 0))) * \
   rdz)
#define DIFFYH1(a)                                          \
  (((DW_DTYPE)(9.0 / 8.0) * (a(0, 1, 0) - a(0, 0, 0)) +     \
    (DW_DTYPE)(-1.0 / 24.0) * (a(0, 2, 0) - a(0, -1, 0))) * \
   rdy)
#define DIFFXH1(a)                                          \
  (((DW_DTYPE)(9.0 / 8.0) * (a(0, 0, 1) - a(0, 0, 0)) +     \
    (DW_DTYPE)(-1.0 / 24.0) * (a(0, 0, 2) - a(0, 0, -1))) * \
   rdx)

#define FD_PAD 2

#elif DW_ACCURACY == 6

#define DIFFZ1(a)                                            \
  (((DW_DTYPE)(75.0 / 64.0) * (a(0, 0, 0) - a(-1, 0, 0)) +   \
    (DW_DTYPE)(-25.0 / 384.0) * (a(1, 0, 0) - a(-2, 0, 0)) + \
    (DW_DTYPE)(3.0 / 640.0) * (a(2, 0, 0) - a(-3, 0, 0))) *  \
   rdz)
#define DIFFY1(a)                                            \
  (((DW_DTYPE)(75.0 / 64.0) * (a(0, 0, 0) - a(0, -1, 0)) +   \
    (DW_DTYPE)(-25.0 / 384.0) * (a(0, 1, 0) - a(0, -2, 0)) + \
    (DW_DTYPE)(3.0 / 640.0) * (a(0, 2, 0) - a(0, -3, 0))) *  \
   rdy)
#define DIFFX1(a)                                            \
  (((DW_DTYPE)(75.0 / 64.0) * (a(0, 0, 0) - a(0, 0, -1)) +   \
    (DW_DTYPE)(-25.0 / 384.0) * (a(0, 0, 1) - a(0, 0, -2)) + \
    (DW_DTYPE)(3.0 / 640.0) * (a(0, 0, 2) - a(0, 0, -3))) *  \
   rdx)
#define DIFFZH1(a)                                           \
  (((DW_DTYPE)(75.0 / 64.0) * (a(1, 0, 0) - a(0, 0, 0)) +    \
    (DW_DTYPE)(-25.0 / 384.0) * (a(2, 0, 0) - a(-1, 0, 0)) + \
    (DW_DTYPE)(3.0 / 640.0) * (a(3, 0, 0) - a(-2, 0, 0))) *  \
   rdz)
#define DIFFYH1(a)                                           \
  (((DW_DTYPE)(75.0 / 64.0) * (a(0, 1, 0) - a(0, 0, 0)) +    \
    (DW_DTYPE)(-25.0 / 384.0) * (a(0, 2, 0) - a(0, -1, 0)) + \
    (DW_DTYPE)(3.0 / 640.0) * (a(0, 3, 0) - a(0, -2, 0))) *  \
   rdy)
#define DIFFXH1(a)                                           \
  (((DW_DTYPE)(75.0 / 64.0) * (a(0, 0, 1) - a(0, 0, 0)) +    \
    (DW_DTYPE)(-25.0 / 384.0) * (a(0, 0, 2) - a(0, 0, -1)) + \
    (DW_DTYPE)(3.0 / 640.0) * (a(0, 0, 3) - a(0, 0, -2))) *  \
   rdx)

#define FD_PAD 3

#elif DW_ACCURACY == 8

#define DIFFZ1(a)                                              \
  (((DW_DTYPE)(1225.0 / 1024.0) * (a(0, 0, 0) - a(-1, 0, 0)) + \
    (DW_DTYPE)(-245.0 / 3072.0) * (a(1, 0, 0) - a(-2, 0, 0)) + \
    (DW_DTYPE)(49.0 / 5120.0) * (a(2, 0, 0) - a(-3, 0, 0)) +   \
    (DW_DTYPE)(-5.0 / 7168.0) * (a(3, 0, 0) - a(-4, 0, 0))) *  \
   rdz)
#define DIFFY1(a)                                              \
  (((DW_DTYPE)(1225.0 / 1024.0) * (a(0, 0, 0) - a(0, -1, 0)) + \
    (DW_DTYPE)(-245.0 / 3072.0) * (a(0, 1, 0) - a(0, -2, 0)) + \
    (DW_DTYPE)(49.0 / 5120.0) * (a(0, 2, 0) - a(0, -3, 0)) +   \
    (DW_DTYPE)(-5.0 / 7168.0) * (a(0, 3, 0) - a(0, -4, 0))) *  \
   rdy)
#define DIFFX1(a)                                              \
  (((DW_DTYPE)(1225.0 / 1024.0) * (a(0, 0, 0) - a(0, 0, -1)) + \
    (DW_DTYPE)(-245.0 / 3072.0) * (a(0, 0, 1) - a(0, 0, -2)) + \
    (DW_DTYPE)(49.0 / 5120.0) * (a(0, 0, 2) - a(0, 0, -3)) +   \
    (DW_DTYPE)(-5.0 / 7168.0) * (a(0, 0, 3) - a(0, 0, -4))) *  \
   rdx)
#define DIFFZH1(a)                                             \
  (((DW_DTYPE)(1225.0 / 1024.0) * (a(1, 0, 0) - a(0, 0, 0)) +  \
    (DW_DTYPE)(-245.0 / 3072.0) * (a(2, 0, 0) - a(-1, 0, 0)) + \
    (DW_DTYPE)(49.0 / 5120.0) * (a(3, 0, 0) - a(-2, 0, 0)) +   \
    (DW_DTYPE)(-5.0 / 7168.0) * (a(4, 0, 0) - a(-3, 0, 0))) *  \
   rdz)
#define DIFFYH1(a)                                             \
  (((DW_DTYPE)(1225.0 / 1024.0) * (a(0, 1, 0) - a(0, 0, 0)) +  \
    (DW_DTYPE)(-245.0 / 3072.0) * (a(0, 2, 0) - a(0, -1, 0)) + \
    (DW_DTYPE)(49.0 / 5120.0) * (a(0, 3, 0) - a(0, -2, 0)) +   \
    (DW_DTYPE)(-5.0 / 7168.0) * (a(0, 4, 0) - a(0, -3, 0))) *  \
   rdy)
#define DIFFXH1(a)                                             \
  (((DW_DTYPE)(1225.0 / 1024.0) * (a(0, 0, 1) - a(0, 0, 0)) +  \
    (DW_DTYPE)(-245.0 / 3072.0) * (a(0, 0, 2) - a(0, 0, -1)) + \
    (DW_DTYPE)(49.0 / 5120.0) * (a(0, 0, 3) - a(0, 0, -2)) +   \
    (DW_DTYPE)(-5.0 / 7168.0) * (a(0, 0, 4) - a(0, 0, -3))) *  \
   rdx)

#define FD_PAD 4  // Number of grid points padded for finite difference stencils

#else

#error DW_ACCURACY must be specified

#endif /* DW_ACCURACY */

/* The Taylor staggered first-derivative stencil weights of the DIFF* macros
 * above, as an FD_PAD-element array initialiser (weight q multiplies the
 * (q, -1-q) tap pair of DIFF*1 and the (1+q, -q) pair of DIFF*H1). For code
 * that needs the weights at runtime with bounds-checked taps (e.g. the
 * periodic-face adjoint guard gathers), where the fixed-footprint DIFF*
 * macros cannot be used. */
#if DW_ACCURACY == 2
#define DW_STAG_TAYLOR_W {(DW_DTYPE)1.0}
#elif DW_ACCURACY == 4
#define DW_STAG_TAYLOR_W {(DW_DTYPE)(9.0 / 8.0), (DW_DTYPE)(-1.0 / 24.0)}
#elif DW_ACCURACY == 6
#define DW_STAG_TAYLOR_W \
  {(DW_DTYPE)(75.0 / 64.0), (DW_DTYPE)(-25.0 / 384.0), (DW_DTYPE)(3.0 / 640.0)}
#else
#define DW_STAG_TAYLOR_W                                     \
  {(DW_DTYPE)(1225.0 / 1024.0), (DW_DTYPE)(-245.0 / 3072.0), \
   (DW_DTYPE)(49.0 / 5120.0), (DW_DTYPE)(-5.0 / 7168.0)}
#endif

#endif /* DW_STAGGERED_GRID_H */
