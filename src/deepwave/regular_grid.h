#ifndef DW_REGULAR_GRID_H
#define DW_REGULAR_GRID_H

/* regular_grid.h
 * Small collection of finite-difference helper macros used by CPU and GPU
 * implementations of regular grid propagators. DIFF* macros implement spatial
 * FD operators; FD_PAD is the stencil 'radius' (number of guard cells required
 * to evaluate stencils).
 */

#if DW_ACCURACY == 2

#define DIFFZ1(a) (((DW_DTYPE)(1.0 / 2.0) * (a(1, 0, 0) - a(-1, 0, 0))) * rdz)
#define DIFFY1(a) (((DW_DTYPE)(1.0 / 2.0) * (a(0, 1, 0) - a(0, -1, 0))) * rdy)
#define DIFFX1(a) (((DW_DTYPE)(1.0 / 2.0) * (a(0, 0, 1) - a(0, 0, -1))) * rdx)
#define DIFFZ2(a) ((-2 * a(0, 0, 0) + (a(1, 0, 0) + a(-1, 0, 0))) * rdz2)
#define DIFFY2(a) ((-2 * a(0, 0, 0) + (a(0, 1, 0) + a(0, -1, 0))) * rdy2)
#define DIFFX2(a) ((-2 * a(0, 0, 0) + (a(0, 0, 1) + a(0, 0, -1))) * rdx2)
#define FD_PAD 1

#elif DW_ACCURACY == 4

#define DIFFZ1(a)                                           \
  (((DW_DTYPE)(8.0 / 12.0) * (a(1, 0, 0) - a(-1, 0, 0)) +   \
    (DW_DTYPE)(-1.0 / 12.0) * (a(2, 0, 0) - a(-2, 0, 0))) * \
   rdz)
#define DIFFY1(a)                                           \
  (((DW_DTYPE)(8.0 / 12.0) * (a(0, 1, 0) - a(0, -1, 0)) +   \
    (DW_DTYPE)(-1.0 / 12.0) * (a(0, 2, 0) - a(0, -2, 0))) * \
   rdy)
#define DIFFX1(a)                                           \
  (((DW_DTYPE)(8.0 / 12.0) * (a(0, 0, 1) - a(0, 0, -1)) +   \
    (DW_DTYPE)(-1.0 / 12.0) * (a(0, 0, 2) - a(0, 0, -2))) * \
   rdx)
#define DIFFZ2(a)                                           \
  (((DW_DTYPE)(-5.0 / 2.0) * a(0, 0, 0) +                   \
    (DW_DTYPE)(4.0 / 3.0) * (a(1, 0, 0) + a(-1, 0, 0)) +    \
    (DW_DTYPE)(-1.0 / 12.0) * (a(2, 0, 0) + a(-2, 0, 0))) * \
   rdz2)
#define DIFFY2(a)                                           \
  (((DW_DTYPE)(-5.0 / 2.0) * a(0, 0, 0) +                   \
    (DW_DTYPE)(4.0 / 3.0) * (a(0, 1, 0) + a(0, -1, 0)) +    \
    (DW_DTYPE)(-1.0 / 12.0) * (a(0, 2, 0) + a(0, -2, 0))) * \
   rdy2)
#define DIFFX2(a)                                           \
  (((DW_DTYPE)(-5.0 / 2.0) * a(0, 0, 0) +                   \
    (DW_DTYPE)(4.0 / 3.0) * (a(0, 0, 1) + a(0, 0, -1)) +    \
    (DW_DTYPE)(-1.0 / 12.0) * (a(0, 0, 2) + a(0, 0, -2))) * \
   rdx2)
#define FD_PAD 2

#elif DW_ACCURACY == 6

#define DIFFZ1(a)                                          \
  (((DW_DTYPE)(3.0 / 4.0) * (a(1, 0, 0) - a(-1, 0, 0)) +   \
    (DW_DTYPE)(-3.0 / 20.0) * (a(2, 0, 0) - a(-2, 0, 0)) + \
    (DW_DTYPE)(1.0 / 60.0) * (a(3, 0, 0) - a(-3, 0, 0))) * \
   rdz)
#define DIFFY1(a)                                          \
  (((DW_DTYPE)(3.0 / 4.0) * (a(0, 1, 0) - a(0, -1, 0)) +   \
    (DW_DTYPE)(-3.0 / 20.0) * (a(0, 2, 0) - a(0, -2, 0)) + \
    (DW_DTYPE)(1.0 / 60.0) * (a(0, 3, 0) - a(0, -3, 0))) * \
   rdy)
#define DIFFX1(a)                                          \
  (((DW_DTYPE)(3.0 / 4.0) * (a(0, 0, 1) - a(0, 0, -1)) +   \
    (DW_DTYPE)(-3.0 / 20.0) * (a(0, 0, 2) - a(0, 0, -2)) + \
    (DW_DTYPE)(1.0 / 60.0) * (a(0, 0, 3) - a(0, 0, -3))) * \
   rdx)
#define DIFFZ2(a)                                          \
  (((DW_DTYPE)(-49.0 / 18.0) * a(0, 0, 0) +                \
    (DW_DTYPE)(3.0 / 2.0) * (a(1, 0, 0) + a(-1, 0, 0)) +   \
    (DW_DTYPE)(-3.0 / 20.0) * (a(2, 0, 0) + a(-2, 0, 0)) + \
    (DW_DTYPE)(1.0 / 90.0) * (a(3, 0, 0) + a(-3, 0, 0))) * \
   rdz2)
#define DIFFY2(a)                                          \
  (((DW_DTYPE)(-49.0 / 18.0) * a(0, 0, 0) +                \
    (DW_DTYPE)(3.0 / 2.0) * (a(0, 1, 0) + a(0, -1, 0)) +   \
    (DW_DTYPE)(-3.0 / 20.0) * (a(0, 2, 0) + a(0, -2, 0)) + \
    (DW_DTYPE)(1.0 / 90.0) * (a(0, 3, 0) + a(0, -3, 0))) * \
   rdy2)
#define DIFFX2(a)                                          \
  (((DW_DTYPE)(-49.0 / 18.0) * a(0, 0, 0) +                \
    (DW_DTYPE)(3.0 / 2.0) * (a(0, 0, 1) + a(0, 0, -1)) +   \
    (DW_DTYPE)(-3.0 / 20.0) * (a(0, 0, 2) + a(0, 0, -2)) + \
    (DW_DTYPE)(1.0 / 90.0) * (a(0, 0, 3) + a(0, 0, -3))) * \
   rdx2)
#define FD_PAD 3

#elif DW_ACCURACY == 8

#define DIFFZ1(a)                                            \
  (((DW_DTYPE)(4.0 / 5.0) * (a(1, 0, 0) - a(-1, 0, 0)) +     \
    (DW_DTYPE)(-1.0 / 5.0) * (a(2, 0, 0) - a(-2, 0, 0)) +    \
    (DW_DTYPE)(4.0 / 105.0) * (a(3, 0, 0) - a(-3, 0, 0)) +   \
    (DW_DTYPE)(-1.0 / 280.0) * (a(4, 0, 0) - a(-4, 0, 0))) * \
   rdz)
#define DIFFY1(a)                                            \
  (((DW_DTYPE)(4.0 / 5.0) * (a(0, 1, 0) - a(0, -1, 0)) +     \
    (DW_DTYPE)(-1.0 / 5.0) * (a(0, 2, 0) - a(0, -2, 0)) +    \
    (DW_DTYPE)(4.0 / 105.0) * (a(0, 3, 0) - a(0, -3, 0)) +   \
    (DW_DTYPE)(-1.0 / 280.0) * (a(0, 4, 0) - a(0, -4, 0))) * \
   rdy)
#define DIFFX1(a)                                            \
  (((DW_DTYPE)(4.0 / 5.0) * (a(0, 0, 1) - a(0, 0, -1)) +     \
    (DW_DTYPE)(-1.0 / 5.0) * (a(0, 0, 2) - a(0, 0, -2)) +    \
    (DW_DTYPE)(4.0 / 105.0) * (a(0, 0, 3) - a(0, 0, -3)) +   \
    (DW_DTYPE)(-1.0 / 280.0) * (a(0, 0, 4) - a(0, 0, -4))) * \
   rdx)
#define DIFFZ2(a)                                            \
  (((DW_DTYPE)(-205.0 / 72.0) * a(0, 0, 0) +                 \
    (DW_DTYPE)(8.0 / 5.0) * (a(1, 0, 0) + a(-1, 0, 0)) +     \
    (DW_DTYPE)(-1.0 / 5.0) * (a(2, 0, 0) + a(-2, 0, 0)) +    \
    (DW_DTYPE)(8.0 / 315.0) * (a(3, 0, 0) + a(-3, 0, 0)) +   \
    (DW_DTYPE)(-1.0 / 560.0) * (a(4, 0, 0) + a(-4, 0, 0))) * \
   rdz2)
#define DIFFY2(a)                                            \
  (((DW_DTYPE)(-205.0 / 72.0) * a(0, 0, 0) +                 \
    (DW_DTYPE)(8.0 / 5.0) * (a(0, 1, 0) + a(0, -1, 0)) +     \
    (DW_DTYPE)(-1.0 / 5.0) * (a(0, 2, 0) + a(0, -2, 0)) +    \
    (DW_DTYPE)(8.0 / 315.0) * (a(0, 3, 0) + a(0, -3, 0)) +   \
    (DW_DTYPE)(-1.0 / 560.0) * (a(0, 4, 0) + a(0, -4, 0))) * \
   rdy2)
#define DIFFX2(a)                                            \
  (((DW_DTYPE)(-205.0 / 72.0) * a(0, 0, 0) +                 \
    (DW_DTYPE)(8.0 / 5.0) * (a(0, 0, 1) + a(0, 0, -1)) +     \
    (DW_DTYPE)(-1.0 / 5.0) * (a(0, 0, 2) + a(0, 0, -2)) +    \
    (DW_DTYPE)(8.0 / 315.0) * (a(0, 0, 3) + a(0, 0, -3)) +   \
    (DW_DTYPE)(-1.0 / 560.0) * (a(0, 0, 4) + a(0, 0, -4))) * \
   rdx2)
#define FD_PAD 4  // Number of grid points padded for finite difference stencils

#else

#error DW_ACCURACY must be specified

#endif /* DW_ACCURACY */

#endif /* DW_REGULAR_GRID_H */
