Accuracy and grid design
========================

Two choices dominate the accuracy and stability of a finite-difference wave
simulation: the spatial grid cell spacing (``grid_spacing``) and the time step
interval (``dt``). This page gives the rules of thumb Deepwave uses internally,
so you can choose them well.

Points per wavelength (spatial sampling)
----------------------------------------

To represent a wave on a grid without excessive numerical dispersion (waves of
different frequencies travelling at slightly wrong speeds), there must be
enough grid cells across each wavelength. The shortest wavelength in the
simulation is

.. math::

    \lambda_\mathrm{min} = \frac{v_\mathrm{min}}{f_\mathrm{max}}

where :math:`v_\mathrm{min}` is the smallest (non-zero) wavespeed in the model
and :math:`f_\mathrm{max}` is the highest significant frequency in the source
wavelet. The number of cells per wavelength is then
:math:`\lambda_\mathrm{min} / \Delta x`.

Deepwave recommends **at least six grid cells per shortest wavelength** and
will emit a warning if your grid is coarser than this (the check uses
``pml_freq`` as a proxy for the dominant frequency -- either the value you
gave, or, if you gave none, the peak of your source amplitudes' spectrum,
which the warning then says it inferred). Higher finite-difference orders of ``accuracy``
(see below) tolerate coarser grids; lower orders need finer grids. In
practice:

* Increase ``accuracy`` (to 4, 6, or 8) to allow a coarser grid for the same
  dispersion error, at the cost of more computation per cell.
* Decrease ``grid_spacing`` (a finer grid) to reduce dispersion, at the cost of
  more cells (and, through the CFL condition below, more time steps).
* Limit the high-frequency content of your source wavelet to what the grid can
  represent.

Letting Deepwave suggest a grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`deepwave.suggest_discretisation` does the arithmetic on this page for
you. Give it :math:`v_\mathrm{min}`, :math:`f_\mathrm{max}` and the number of
dimensions and it returns a ``grid_spacing``, a ``dt`` at or below the CFL
limit below, the ``points_per_wavelength`` they achieve, and a ``note``
explaining the choice::

    >>> import deepwave
    >>> suggestion = deepwave.suggest_discretisation(1500, 25, 2, max_vel=4500)
    >>> suggestion.grid_spacing
    (10.0, 10.0)

Pass ``max_vel`` whenever your model has faster material in it than
``min_vel``: the CFL condition is set by the *fastest* wavespeed, and without
it the suggestion assumes a homogeneous model and returns a ``dt`` that is
too large. With ``accuracy=None`` (the default) it also picks the accuracy
order estimated to be cheapest for the resolution you asked for. It respects
both the per-order dispersion criterion and the six-cells-per-wavelength
warning threshold below, taking whichever is stricter, so what it suggests
never trips that warning; :doc:`discretisation` gives the table it reads and
explains why the two disagree.

The CFL condition (time sampling and stability)
-----------------------------------------------

An explicit time-stepping scheme is only stable if the time step is small
enough that information does not cross more than a fraction of a grid cell per
step. Deepwave enforces the Courant--Friedrichs--Lewy (CFL) condition

.. math::

    \Delta t \le \frac{C}{v_\mathrm{max} \sqrt{\sum_i 1 / \Delta x_i^2}}

with a Courant number :math:`C = 0.6`, where :math:`v_\mathrm{max}` is the
maximum wavespeed. For an isotropic grid (equal spacing :math:`\Delta x` in all
:math:`n` dimensions) this simplifies to

.. math::

    \Delta t \le \frac{0.6\, \Delta x}{v_\mathrm{max} \sqrt{n}}.

For the **elastic** propagator the controlling velocity is the faster P-wave
velocity :math:`v_p = \sqrt{(\lambda + 2\mu)/\rho}`, not the S-wave velocity.

The constant :math:`C = 0.6` is deliberately conservative: it is the same for
every accuracy order and every number of dimensions (the dimensionality enters
only through the :math:`\sqrt{\sum_i 1/\Delta x_i^2}` term). The *true* critical
Courant number, derived exactly from each stencil, ranges from about 0.78 to
1.0 depending on order and scheme (see :doc:`discretisation` for the table and
its derivation), so 0.6 leaves real headroom.

.. _opt-in-courant:

Opt-in: choosing the Courant number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to trade some of that safety margin for speed, every propagator
accepts an optional ``courant_number`` argument. It is the *actual*
dimensionless Courant number the internal time step will target (there is no
hidden multiplier). Leaving it ``None`` reproduces the default 0.6 behaviour
bit-for-bit, while a larger value permits a larger internal step and hence fewer
internal steps. Because the step count scales as :math:`1/C`, moving a 2nd-order
run from 0.6 to 0.9 uses :math:`0.6/0.9 \approx 0.67\times` as many steps (a
~1.5x speed-up). The value is validated against the critical Courant number for
your chosen ``accuracy`` order and grid scheme, and Deepwave raises an error if
you exceed that stencil bound::

    # 2nd-order interior stencil: 0.9 is below its critical value of 1.0.
    out = scalar(v, dx, dt, accuracy=2, courant_number=0.9, ...)

Note that raising the Courant number (a larger internal ``dt``) increases
*temporal* dispersion, which is independent of the spatial ``accuracy`` order;
see :doc:`discretisation`.

The stencil bound assumes the corresponding lossless homogeneous interior.
Passing it does not prove stability for arbitrary heterogeneous materials,
anisotropy, attenuation, PML or surface closures. Check material admissibility
(:doc:`materials`) and convergence for the configuration you use. For
viscoacoustic propagation, ``max_vel`` must bound the unrelaxed speed;
the supplied ``v`` is the slower relaxed speed.

You do not have to satisfy this yourself. If the ``dt`` you provide is too
large, Deepwave automatically runs the internal propagation at a smaller time
step (an integer fraction of your ``dt``) that obeys the CFL condition, and
resamples the sources and receivers accordingly. Providing a ``dt`` that is
already at or below the CFL limit simply avoids that internal upsampling. The
``max_vel`` argument lets you fix :math:`v_\mathrm{max}` explicitly (useful for
consistency across runs, or when using batched/``vmap`` models where the
maximum cannot be read from the model).

Putting it together
-------------------

A typical workflow is:

1. Choose the maximum frequency you need and the minimum velocity in your
   model; pick ``grid_spacing`` to give at least six cells per shortest
   wavelength (more for low ``accuracy``).
2. Pick ``dt`` at or below the CFL limit above using the maximum velocity. If
   you choose a larger ``dt`` for convenience, Deepwave will subsample
   internally for stability, but the recorded output is returned at your
   requested ``dt``.
3. If you see the points-per-wavelength warning, either refine the grid, raise
   ``accuracy``, or reduce the source's high-frequency content.

The helper functions :func:`deepwave.discretisation.cfl_condition` and
:func:`deepwave.discretisation.cfl_condition_n` implement the CFL calculation and can
be called directly if you wish to compute a suitable time step.

Compiler floating-point flags
------------------------------

The compiled CPU/CUDA kernels build by default with ``-Ofast`` (CPU;
Clang uses the equivalent ``-O3 -ffast-math -fstrict-aliasing``) /
``--use_fast_math`` (CUDA), the same flags released versions of Deepwave
have always shipped. Fast-math flags relax IEEE 754 semantics (NaN/Inf
handling, denormal flush-to-zero) in exchange for extra speed. On the CPU
one relaxation is deliberately left out: the build adds
``-fno-associative-math``, because reassociation would let the compiler
evaluate a cell's stencil in an order that depends on how the grid was split
between threads, costing the thread-count invariance described in
:doc:`what_deepwave_calculates`. ``--use_fast_math`` has no equivalent
control, so the CUDA build does not disable reassociation. What the CUDA
side is tested for instead is that repeated runs of one call, on one device
and build, give bitwise identical gradients, again as described in
:doc:`what_deepwave_calculates`.

This default was decided by measurement, not assumption. The deciding
comparison was a same-commit A/B: the same source tree built twice, once
with ``-Ofast``/``--use_fast_math`` and once with precise ``-O3``
(``-fno-math-errno`` on CPU), run back-to-back on the same hardware with
nothing else varying. Three regimes were measured:

* **Many-shots/small-model CPU sweep** (enough shots to occupy every host
  thread, a small ~200x500 grid -- the shape most likely to expose
  per-step spatial-domain OpenMP parallelism losing to fast-math's
  per-thread numerical shortcuts): precise ``-O3`` cost a median 25.6%
  more wall-clock time than fast-math, worst case +36.2%.
* **Meso-tier CPU** (all compiled propagators): precise ``-O3`` cost a
  median 2.5% more wall-clock time, with wide case-by-case dispersion
  (roughly a third of cases exceeded a 5% regression, worst case +86.3%;
  a smaller number of cases were faster under ``-O3``, best case -15.2%).
* **Meso-tier GPU** (all compiled propagators): precise ``-O3`` cost a
  median 7.7% more wall-clock time, worst case +44.4%; no case was more
  than 5% faster under ``-O3``.

Peak resident memory (CPU) and peak allocator usage (GPU) were not
meaningfully different between the two builds in this comparison (median
deltas at or near zero in all three regimes).

The many-shots/small-model regime alone exceeds the 5% median-cost
threshold this decision was gated on, and the meso-tier GPU regime
independently exceeds it too; only meso-tier CPU stays under the threshold
on its own. Because at least one regime breaches the gate, fast-math stays
the default. The accuracy-sensitive test suite (the Taylor/linearisation
convergence-rate tests and a representative gradcheck subset spanning the
acoustic/elastic VTI, Born, scalar, and viscoacoustic families) passes
identically under both flag choices at their documented tolerances, so
there is no accuracy finding forcing precise math -- the trade is purely a
speed-vs-IEEE-semantics one.

On GCC, an ``-Ofast``-linked *executable* pulls in ``crtfastmath.o``, whose
constructor (visible as the ``set_fast_math`` symbol) sets the FTZ/DAZ
floating-point control bits process-wide when it runs. On GCC 13 and later,
however, this constructor is linked into executables only, not shared
libraries. Deepwave's compiled kernels are a shared library
(``libdeepwave_C.so``), and on a GCC 14 build that library carries no
``set_fast_math`` symbol at all -- confirmed both by symbol inspection and
by a runtime denormal-handling test that showed no change in the process's
floating-point environment before and after importing the fast-math build.
Importing Deepwave built on a modern GCC therefore does not alter the
floating-point behaviour of the caller's other code. Wheels built with an
older GCC toolchain may not have this property -- see "Denormal handling
and older wheels" below.

If you have a workload where strict IEEE semantics or bit-for-bit gradient
determinism matters more than the extra fast-math speed, build from source
with the CMake option ``-DDW_FAST_MATH=OFF`` to build IEEE-precise ``-O3``
kernels instead.

Denormal handling and older wheels
------------------------------------

Because a modern-GCC build of ``libdeepwave_C.so`` gets no FTZ/DAZ from the
linker (see above), the CPU kernels' own per-thread state is what decides
whether decaying wave amplitudes that underflow into the float32 subnormal
range are flushed to zero or computed at full (slow) denormal-arithmetic
cost. Measurement (a same-commit A/B using a process-wide FTZ/DAZ proxy,
meso-tier and many-shots CPU sweeps, reps>=5 for statistical reliability)
found this is a real, material effect under the default ``-Ofast`` build:
decaying PML wave tails do reach the subnormal range in practice, and
avoiding the resulting slow arithmetic sped up many meso-tier cases by
20-70% and every many-shots-sweep case (forward and ``forward_backward``
alike) by 6-68%, with zero regressions found. Deepwave's compiled CPU
kernels therefore set flush-to-zero and denormals-are-zero
(``_MM_SET_FLUSH_ZERO_MODE`` / ``_MM_SET_DENORMALS_ZERO_MODE``) for each
OpenMP worker thread at the start of every parallel region and restore the
thread's prior state at the end of it. This is scoped per-thread-per-region,
not the process-wide linker-constructor behaviour described above -- it
cannot silently change the numerics of any other code that later runs on
the same OS thread outside Deepwave's own kernels (e.g. other torch/numpy
operations in the same process). GPU kernels are unaffected: CUDA's
``--use_fast_math`` already flushes denormals per instruction in the
generated device code, with no equivalent runtime mode to toggle.

A consequence of this default is that subnormal operands and results may be
replaced by zero inside the compiled propagation kernels. This improves
performance when long, decaying wave tails would otherwise cause slow
subnormal arithmetic. It is a numerical approximation: small local changes
can accumulate over many time steps, so the small magnitude of the flushed
values alone does not establish an error bound on the final outputs or
gradients. Numerical comparisons with the tensor implementation and analytic
solutions remain necessary. The regular-grid second derivative uses weighted
neighbour-minus-centre differences to reduce cancellation from separately
rounded stencil coefficients.


Released wheels built with ``-Ofast`` on an older-GCC manylinux image
could, in principle, still carry the process-wide ``crtfastmath.o``
constructor described above, meaning importing such a wheel would set
FTZ/DAZ for the *whole process*, not just Deepwave's own kernel threads --
silently changing the numerics of the importing program's other
floating-point code. **Checked directly against the actual published
wheel** (``deepwave==0.0.27`` from PyPI, a ``manylinux2014``/
``manylinux_2_17``/``manylinux_2_28`` build): its ``libdeepwave_C.so``
carries no ``set_fast_math`` symbol (``nm -D``/``nm``, both empty), and a
runtime denormal-preservation test shows no change in the importing
process's floating-point environment before and after import. **This
concern does not apply to the current release** -- its build toolchain
already behaves like the modern-GCC case described above, not the
older-GCC one. If a *future* release is ever built on an older manylinux
image, or the toolchain otherwise changes, re-check with
``nm -D <path-to-libdeepwave_C.so> | grep set_fast_math`` or the runtime
test above before assuming this still holds.
