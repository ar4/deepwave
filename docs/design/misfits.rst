Normalised cross-correlation: numerical conditioning
====================================================

:func:`deepwave.misfits.normalised_cross_correlation` documents what it
computes and what its zero-norm branch returns. This page records the
numerical-conditioning device that sits underneath that definition, and why
the definition is written the way it is.

Internal max-abs rescaling
--------------------------

Before the dot product and the norms are computed, each trace is divided,
independently, by its own maximum absolute value. That scale factor is
*detached*: it is purely a numerical-conditioning device, not part of the
mathematical NCC formula, so no gradient flows through it.

The rescaling leaves the result mathematically unchanged, because NCC is
exactly scale-invariant in each argument separately
(:math:`\mathrm{NCC}(ap, bt) = \mathrm{NCC}(p, t)` for any :math:`a, b > 0`).
What it buys is that every intermediate quantity is :math:`O(1)` regardless
of the traces' raw amplitude, so the sum of squares behind the L2 norm cannot
underflow. Computing the norm directly from raw samples would suffer exactly
that underflow at small-but-nonzero amplitudes, and ordinary seismic data is
routinely far below unit scale -- particle-velocity or strain-rate traces of
:math:`10^{-8}` or smaller are unremarkable. With the rescaling, such data is
normalised and compared exactly like unit-scale data, all the way down to the
point where a trace's own peak amplitude underflows to exactly 0 in the
working dtype.

Why the zero branch is a branch, not an epsilon
-----------------------------------------------

The guard on a zero denominator is a genuine branch on the value ``0.0``, not
a rounded-down small-denominator threshold, so it does not fire for
small-but-nonzero amplitudes. It applies only when at least one of
``pred``/``target`` is an *exactly* all-zero trace, whose L2 norm is exactly
``0.0`` rather than merely small; the rescaling above is what makes that a
safe test to write, since it removes the underflow that would otherwise turn
a small-but-real trace into a zero one.

In the implementation, ``torch.where`` evaluates both branches, so a
placeholder denominator of 1 is substituted wherever the true denominator is
0, purely to avoid a ``0 / 0`` division. That branch's result is then
discarded by the outer ``torch.where``, and its gradient contribution
vanishes because the dot product is also 0 wherever the denominator is 0 (a
zero-scale trace is exactly the zero vector, so its dot product with anything
is 0).

Why no mean is subtracted
-------------------------

Not subtracting a mean is a deliberate choice, and the reason is stated in
the function's own docstring: it keeps the misfit sensitive to any DC offset,
which is physically meaningful for wave data, since a non-zero mean is itself
a mismatch worth penalising.

Mean-centring would not, in fact, cost the exact per-trace amplitude-scale
invariance that makes this misfit useful for FWI -- that property survives
either way. The argument for omitting it is the narrower one that it is an
unneeded operation in the formula, and the formula is easier to reason about,
and cheaper, without it.
