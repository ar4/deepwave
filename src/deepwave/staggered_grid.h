#ifndef DW_STAGGERED_GRID_H
#define DW_STAGGERED_GRID_H

/* staggered_grid.h
 * Small collection of finite-difference helper macros used by CPU and GPU
 * implementations of staggered grid propagators. DIFF* macros implement spatial
 * FD operators; FD_PAD is the stencil 'radius' (number of guard cells required
 * to evaluate stencils).
 */

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

#endif /* DW_STAGGERED_GRID_H */
