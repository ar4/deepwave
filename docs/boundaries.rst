Boundary conditions
===================

Every finite simulation has edges, and what happens at those edges matters: you
usually want most edges to absorb outgoing waves (as if the model continued
forever), but sometimes you want an edge to reflect (for example, a free
surface at the top of an Earth model). This page explains how Deepwave treats
the edges of the model and how to control them. Getting this wrong is a common,
silent source of confusion (unexpected reflections, or a missing surface
multiple), so it is worth understanding.

Absorbing edges: the PML
------------------------

By default each edge of the model is surrounded by a Perfectly Matched Layer
(PML), an absorbing region that strongly attenuates waves entering it so they do
not reflect back into the simulation. Every propagator uses the same family of
absorbing boundary: a convolutional, complex-frequency-shifted PML (CFS-C-PML).
The complex-frequency-shift ("alpha") term is what makes this a *CFS* PML rather
than a plain PML. It is designed to give good absorption of
near-grazing-incidence waves and of evanescent waves, and to avoid the long-time,
low-frequency drift that a plain (non-shifted) PML can suffer. Concretely, the
scalar propagator uses the CFS-C-PML of `Pasalic and McGarry (2010)
<https://doi.org/10.1190/1.3513453>`_; the acoustic and elastic propagators use
the CFS-C-PML of `Komatitsch and Martin (2007)
<https://doi.org/10.1190/1.2757586>`_.

The width of the PML on each edge is controlled by ``pml_width``. You can give a
single number (used for every edge) or a list with one value per face, in the
order

* beginning of the first (slowest) dimension,
* end of the first dimension,
* beginning of the second dimension,
* end of the second dimension,
* ... and so on for higher dimensions.

Typical widths are 10 to 20 cells; wider PMLs absorb better (and cost more).
The wavespeed used inside the PML is obtained by replicating the values on the
edge of the model.

.. note::

   **Each absorbing face uses its own physical width.** For polynomial order
   :math:`N`, the damping amplitude is
   :math:`\sigma_0=-(N+1)v_{\max}\log(R)/(2L)`, where :math:`L` is that
   face's cell width times its axis's grid spacing and :math:`R=0.001` is the
   continuous normal-incidence design target. Widening another face does not
   weaken this face's damping. This follows the local-width normalisation in
   the authors' `SEISMIC_CPML implementation
   <https://github.com/geodynamics/seismic_cpml/blob/master/seismic_CPML_2D_pressure_second_order.f90>`_.

   The design target is **not a bound on measured discrete reflection**.
   Frequency, angle, material contrasts and grid resolution matter; a very
   thin layer can reflect strongly despite the same continuous target.
   Check absorption for your configuration. Versions through v0.0.27 used
   the globally longest physical width, so unequal face widths or unequal
   grid spacing can give different results after this correction.

.. _pml-freq:

Choosing ``pml_freq``
---------------------

``pml_freq`` sets the CFS (complex frequency shift) of the PML: the frequency
shift is :math:`\alpha = \pi\,f_\text{pml}`, tapering from that value at the
inner edge of the layer to zero at the outer edge. The layer absorbs best
around this frequency, so it should be the dominant frequency of your source.

**You do not have to pass it.** If you leave ``pml_freq`` unset, Deepwave
infers it from the peak of the source amplitudes' energy spectrum -- for a
Ricker wavelet, the wavelet's own centre frequency. Spectra from all shots and
all sources are combined as a sum of squared magnitudes, so the answer does not
depend on the sources' relative delays or polarities, and zero-padding a
wavelet in time does not change it. Channels in different units (a pressure and
a body force, say) are each normalised before being combined, so no channel
dominates merely by having larger numbers. The result is rounded to three
significant figures, so trimming or extending a record does not quietly move
it. A simulation with **no sources at all** (an initial-wavefield-only run) has
nothing to infer from and uses :math:`0.5/dt/5`, a fifth of the Nyquist
frequency. Either way, the value actually used is reported as
``out.info.pml_freq``.

.. warning::

   **Pin** ``pml_freq`` **explicitly for anything you publish, and for any
   inversion.** Inference makes the absorbing boundary a function of the
   wavelet, so changing the wavelet -- a different source estimate, a
   band-limited FWI stage, a filtered field record -- also changes the
   boundary, and with it the last few digits of every trace. Passing
   ``pml_freq=<your dominant frequency>`` (or feeding back the
   ``out.info.pml_freq`` of a reference run) makes the boundary a fixed
   property of the experiment. It is also required when pairing
   :func:`deepwave.wavefields.record_boundary` with
   :func:`deepwave.wavefields.reinject_boundary`; this keeps the absorbing
   boundary configuration consistent across the recording and replay runs.

.. note::

   Deepwave 0.0.27 and earlier used a fixed ``pml_freq`` of ``25.0`` whenever
   the argument was omitted -- a seismic value, and wrong by five orders of
   magnitude for, say, ultrasound. The inference reproduces it exactly for the
   classic 25 Hz seismic call, which therefore says nothing. Wherever the
   inferred value is anything other than exactly ``25.0`` Deepwave warns once,
   naming the resolved value and how it was inferred;
   ``pml_freq=25.0`` restores the old behaviour exactly. That warning is a
   migration aid and is removed in 0.4.0.

**Is a lower** ``pml_freq`` **better?** Sometimes, and not reliably. A smaller
:math:`\alpha` is closer to a classic (unshifted) PML; the frequency shift is
what a CFS-PML adds to handle evanescent and grazing energy and to keep the
layer well behaved at late times, so driving it towards zero trades one kind
of error for another. Measurements across scalar and elastic propagators, 2-D
and 3-D, PML widths 10/20/40, normal and grazing incidence found
``pml_freq`` = a quarter to a half of the peak frequency better than the peak
itself for 2-D scalar (by up to 2x in reflection error), essentially neutral
in 3-D, and *worse* for some elastic cases -- while ``pml_freq = 0`` (no
frequency shift at all) left 2-10x more energy in a scalar model at late
times than any small non-zero choice. There is no fraction that wins
everywhere, so Deepwave infers the peak itself, which is also the value
``check_points_per_wavelength`` wants. (It is *not* used for
viscoacoustic ``q_freq``, which you must give: the frequency at
which you specified ``Q`` is a property of your medium, not of the boundary.)
If boundary reflections dominate your error budget in a 2-D scalar
simulation, it is worth trying
``pml_freq`` at half your dominant frequency -- and measuring.

.. _bare-zero-width-face:

Reflecting edges
----------------

To make an edge reflective instead of absorbing, set its ``pml_width`` to zero.
For example, in 2D, to make the beginning of the second dimension (the top, in a
typical depth-oriented model) reflective while keeping the other three edges
absorbing::

    pml_width=[20, 20, 0, 20]

What that edge does mechanically is determined entirely by the discrete update,
and is most precisely stated in terms of *which quantity is held to zero just
outside the model* and *how the evolved field reflects*. The two propagators
behave differently because they discretise the wavefield differently.

**Scalar.** The scalar propagator evolves a single field :math:`u` that obeys
:math:`\nabla^2 u - c^{-2}\ddot u = f`; :math:`u` is whatever quantity you choose
to model with that equation (often, but not necessarily, pressure). At a
zero-PML edge the exterior values of :math:`u` itself are held to zero, which
imposes :math:`u = 0` on the boundary — a Dirichlet (nodal) condition. The
evolved field therefore reflects with **inverted** polarity, reflection
coefficient :math:`R_u = -1`, and is *odd* (antisymmetric) about a mirror plane
**one cell** outside the model edge (exactly, for ``accuracy=2``).

Whether you call this a "free" or a "rigid" surface is *not* an intrinsic
property of the propagator — it depends on what :math:`u` represents, because
"free" and "rigid" are statements about a mechanical surface, while :math:`u=0`
is a statement about the field:

* If :math:`u` is **pressure**, then :math:`u=0` is the pressure-release
  condition — a **free surface** (e.g. the sea surface). Pressure has a node
  there.
* If :math:`u` is a **displacement** (or particle velocity), then :math:`u=0` is
  the clamped condition — a **rigid** surface. Displacement has a node there.

Both are the *same* numerics (:math:`u=0`, :math:`R_u=-1`); the label flips
because pressure is proportional to the spatial gradient of displacement, so
where one has a node the other has an antinode. The intrinsic, unambiguous facts
are: :math:`u=0` on the boundary, :math:`R_u=-1`, odd image one cell out. The
"free surface" name used elsewhere in Deepwave assumes the pressure
interpretation.

**Acoustic.** The acoustic propagator evolves pressure :math:`p` *and* the
particle-velocity components on a staggered grid, so the outermost grid point
just outside the model edge is a normal-*velocity* node (half a cell beyond the
last pressure node). A zero-PML edge holds that velocity to zero. Zero normal
velocity is an unambiguous **rigid** wall (no flow through it) — this does *not*
depend on interpretation, because particle velocity is a named physical
quantity. Consequently pressure reflects with **preserved** polarity,
:math:`R_p = +1`, and is *even* (symmetric) about a mirror plane **half a cell**
outside the edge; the normal velocity is odd (:math:`R_v=-1`) with a node on that
plane.

So the scalar and acoustic zero-PML edges differ in both the *sign* of the
reflection and the *position* of the effective mirror plane (one cell out vs.
half a cell out). This matters when interpolating sources and receivers near such
an edge (see below).

Per-propagator reflection coefficients
--------------------------------------

The condition a bare zero-``pml_width`` face produces, measured on every
propagator. **Prefer naming the condition** with ``boundary`` (or
:func:`deepwave.boundaries`) when a supported named face represents it: the
call then states the physics and uses its exact image. The elastic family's
bare frictionless wall has no named equivalent. Deepwave emits a one-time
:class:`deepwave.ZeroWidthFaceWarning` for a bare face.

.. _zero-width-pml-face-spellings:

**A zero-width** ``"pml"`` **face is the same call however it is spelled.**
``"pml"`` is the spelled-out default, so these three are one call, and all
three warn::

    pml_width=[0, 20, 20, 20]                                  # boundary omitted
    pml_width=[0, 20, 20, 20], boundary="pml"                   # one string, every face
    pml_width=[0, 20, 20, 20], boundary=["pml"] * 4             # per face

Naming a face ``"pml"`` says it absorbs, and a zero-width face does not, so
saying it explicitly cannot make the warning go away. It never raises: a
zero-width ``"pml"`` face is Deepwave 0.0.27's ``pml_width=0`` idiom and
stays fully supported on all nine propagators. What the face *is*:
**exact Dirichlet** (the field held to zero on the boundary) on the scalar
family, and a **frictionless (roller) wall** on the elastic family -- clamped
normal velocity, free tangential velocity, which is *not* the same condition
as ``boundary="rigid"`` (see the warning below).

To silence the warning, either name the condition you meant -- the point of
the warning, and the nearest named spelling is given in the table's last
column, exactly (scalar family) or with the caveat below (elastic family) --
or filter the category::

    warnings.filterwarnings(
        "ignore", category=deepwave.ZeroWidthFaceWarning
    )

.. list-table::
   :header-rows: 1
   :widths: 18 10 16 20 22 14

   * - Propagator
     - Grid
     - Channel measured
     - Condition
     - R (accuracy 2 / 4 / 6 / 8)
     - Surface
   * - ``scalar``
     - regular
     - wavefield
     - free surface
     - −1.0043 / −1.0001 / −1.0001 / −1.0001
     - 1 cell out
   * - ``scalar_born``
     - regular
     - wavefield (bg, sc)
     - free surface
     - bg −1.0043 / −1.0001 / −1.0001 / −1.0001;
       sc −0.9998 / −1.0000 / −1.0000 / −1.0000
     - 1 cell out
   * - ``acoustic``
     - staggered
     - pressure ``p``
     - rigid for pressure
     - +0.9999 / +1.0000 / +1.0000 / +1.0000
     - ½ cell out
   * - ``acoustic_born``
     - staggered
     - pressure ``p`` (bg, sc)
     - rigid for pressure
     - bg +0.9999 / +1.0000 / +1.0000 / +1.0000;
       sc +1.0058 / +1.0002 / +1.0002 / +1.0002
     - ½ cell out
   * - ``acoustic_vti``
     - staggered
     - ``sigma_v``, ``sigma_h``
     - rigid for pressure
     - +0.9999 / +1.0000 / +1.0000 / +1.0000
     - ½ cell out
   * - ``viscoacoustic``
     - staggered
     - pressure ``p``
     - rigid for pressure
     - +0.9984 / +0.9985 / +0.9985 / +0.9985
     - ½ cell out
   * - ``elastic``
     - staggered
     - ``sigma_AA``
     - frictionless rigid
     - +0.9999 / +1.0001 / +1.0001 / +1.0001
     - ½ cell out
   * - ``elastic_born``
     - staggered
     - ``sigma_AA`` (bg, sc)
     - frictionless rigid
     - bg +1.0000 / +1.0000 / +1.0000 / +1.0000;
       sc +1.0026 / +1.0001 / +1.0001 / +1.0001
     - ½ cell out
   * - ``elastic_vti``
     - staggered
     - ``sigma_AA``
     - frictionless rigid
     - +0.9999 / +1.0001 / +1.0001 / +1.0001
     - ½ cell out

Every number is measured, at normal incidence on a flat face in a homogeneous
medium, in a model whose only reflector is the face under test, with the
incident and reflected pulses separated in time at a receiver placed so that
no other arrival overlaps them; ``tests/test_boundary.py``'s
reflection-coefficient tests measure the same quantity the same way. The
viscoacoustic family's ~0.1% deficit is Q attenuation over the reflected
path, not boundary error. The Born propagators are measured on both of their
channels, and the ~0.3–0.6% excursions on the scattered channel at accuracy 2
are grid dispersion at the lowest order, not boundary error: an explicit
``free``/``rigid`` face on the same geometry drifts by the same amount.

The warning Deepwave emits for a bare zero-width face quotes these same
measurements, rounded to three decimal places, as a range per propagator.
The two are **not** generated from one another, so they must be kept in sync
by hand: if you change a row here, re-measure and change
``_BARE_ZERO_WIDTH_CONDITION`` in ``deepwave/_common.py`` as well.
``tests/test_boundary.py::test_docs_reflection_table_matches_the_warning_table``
compares them, so the two cannot drift silently.

**Why the table splits the way it does.** There is no distinct boundary code
for a bare zero-width face: it is internally a ``pml`` face, and a ``pml``
face is never visited by any guard-ring fill, so its guard cells keep their
initial value of **zero** for the whole run. Whether that reads as "free" or
"rigid" depends only on which field the guard ring holds. Zeroing the
wavefield (the regular-grid, scalar family) is a Dirichlet/free surface.
Zeroing the *velocity* outside the pressure nodes (the staggered families) is
a Neumann/rigid surface for pressure. That one sentence makes the whole table
predictable instead of a list of coincidences.

.. warning::

   **The elastic families' bare face is frictionless, and is not the same
   thing as** ``boundary="rigid"``. Outside the last normal-stress row sit
   *both* the normal velocity and the shear stress, and both stay zero
   forever. That is simultaneously a clamped normal velocity and a
   traction-free tangential direction — a **roller wall**. Measured on the
   velocity components: R = −1 on the normal velocity (clamped, as for
   ``rigid``) but R = **+1** on the tangential velocity (free, the *opposite*
   sign to ``rigid``, which is genuinely no-slip). If you want a clamped
   wall, ask for ``boundary="rigid"``; there is no named condition for the
   frictionless one.

**A bare zero-width face is only an approximate reflector above accuracy 2.**
The guard ring is never refilled with the exact odd/even image, so at accuracy
2 — where the stencil reaches only the nearest exterior cell — it *is* the
exact image and the two paths are bit-identical, while at higher orders the
deeper exterior cells the stencil reaches are also left at zero. The resulting
spurious reflection is a few percent of the reflected amplitude and does not
decrease with accuracy order (up to ~5% / ~7% / ~7% at accuracy 4 / 6 / 8 on
the geometry of
``test_free_removes_high_order_spurious_reflection``; smaller on a
finer grid). **The peak amplitude is nonetheless ≈ ∓1 at every order** — the
table above — so the error is in waveform *shape*, not in the polarity or the
magnitude of the reflection. Naming the face with ``boundary`` removes it.

First-class boundary faces
--------------------------

In addition to the implicit zero-``pml_width`` reflector above, the scalar
propagator accepts an explicit ``boundary`` argument that selects a boundary
*type* per face. It fills the exterior with the **exact** image of the interior
at every accuracy order (not just accuracy 2). ``boundary`` is either ``None``
(the default: every face is absorbing/PML, bit-identical to previous behaviour)
or a single string applying to every face, or a list of ``2 * ndim``
strings, in the same face order as ``pml_width``. A single string is
broadcast, not iterated: ``boundary="free"`` means ``"free"`` on every face,
not one character per face. The entries are:

* ``"pml"`` — absorbing, using ``pml_width`` for that face as usual;
* ``"free"`` — a pressure-release (free) surface: the field is held to zero on
  the boundary and reflects with **inverted** polarity, :math:`R=-1`;
* ``"rigid"`` — a rigid wall: the field reflects with **preserved** polarity,
  :math:`R=+1`;
* ``"periodic"`` — the face wraps around to the opposite face (no reflection;
  energy that leaves one side re-enters the other travelling in the same
  direction).

A ``"free"``, ``"rigid"``, or ``"periodic"`` face requires ``pml_width`` to be
zero on that face (validated, with a clear error otherwise), and ``"periodic"``
must be set on **both** faces of a dimension. Example (2-D, free surface on the
beginning of the second dimension, absorbing elsewhere)::

    out = deepwave.scalar(
        v, grid_spacing, dt,
        source_amplitudes=source, source_locations=source_locations,
        pml_width=[20, 20, 0, 20],
        boundary=["pml", "pml", "free", "pml"],
        backend="eager",
    )

**Why this is more accurate than a bare zero-**\ ``pml_width``\ ** face.** A
bare zero-``pml_width`` face is internally still a ``pml`` face, and no
guard-ring fill ever runs on it, so *every* exterior cell it reaches keeps
its initial value of zero for the whole simulation. Holding the exterior at
zero happens to be the exact image at ``accuracy=2``, where the stencil
reaches only the nearest exterior cell; at higher orders the stencil also
reaches deeper exterior cells, which the exact image would not have set to
zero, so the reflecting edge carries a spurious reflection of several
percent that does not decrease with accuracy order. A
``"free"``/``"rigid"`` face instead refills the whole guard ring with the exact
image every time step, so the analytic reflection coefficient
(:math:`R=\mp 1`) is met to better than 1% at **all** accuracy orders. Measured
spurious reflection removed (fraction of the main reflection, scalar free
surface): 0% at accuracy 2 (the two coincide — the default path stays
bit-identical), ~5% at accuracy 4, ~7% at accuracy 6, and ~7% at accuracy 8.

.. note::

   **Accuracy of the reflecting edge.** The reflection coefficients and
   mirror-plane positions above are *exact* for a 2nd-order spatial stencil
   (``accuracy=2``): there the stencil reaches only the nearest exterior
   cell, and leaving that at zero is precisely the image condition. At higher
   accuracy the stencil also reaches deeper exterior cells, which are also
   left at zero rather than filled with the exact (odd/even) image, so the
   reflecting edge is only an approximate image.
   It carries a spurious reflection error of order several percent (up to ~10% of
   the reflected amplitude in some configurations) that does **not** decrease
   with interior accuracy order. The PML-absorbing edges are unaffected; this
   caveat applies only to zero-``pml_width`` reflecting edges.

The flat-face ``boundary=`` argument is available on every propagator; see
`boundary= across propagators`_ below for the per-propagator backend and
test status. Not every propagator accepts every face type, and for one of
them it depends on which face of the axis and on ``accuracy``:
:ref:`boundary-faces-accepted` is the full list, and a face that is not
accepted raises rather than running. See the appendix
(:ref:`boundaries-appendix`) for the exact per-field image formulas, the
adjoint implementation, and the full list of checks.

Placing sources and receivers near a reflecting edge
----------------------------------------------------

A source or receiver given at a non-integer (floating point) location is spread
onto / gathered from the grid with a windowed-sinc (Hicks) interpolation
stencil. That stencil is ``2 * location_interpolation_halfwidth`` cells wide
(eight by default), so a point fewer than
``location_interpolation_halfwidth - 1`` cells from an edge has part of it
outside the model. Near a reflecting edge the exterior part has to be folded
back into the model by the method of images:

* a **free surface** (the scalar reflecting edge) folds with a sign flip;
* a **rigid surface** (the acoustic pressure reflecting edge) folds without a
  sign flip.

``scalar`` and ``scalar_born`` do this for you. Their source and receiver
points live on the regular grid, and the mirror plane of their ``"free"`` and
``"rigid"`` faces is exactly the one :class:`deepwave.Hicks` images about, so
a plain float location next to such a face is folded automatically::

    out = deepwave.scalar(
        v, dx, dt,
        source_amplitudes=source,
        source_locations=torch.tensor([[[1.5, 30.0]]]),   # 1.5 cells down
        boundary=["free", "pml", "pml", "pml"],
        pml_width=[0, 20, 20, 20],
    )

which gives exactly (bit for bit) what building the interpolator yourself
gives::

    from deepwave import Hicks
    hicks = Hicks(locations, free_surfaces=[True, False, ...],
                  model_shape=model.shape, dtype=model.dtype)
    # feed hicks.get_locations() / hicks.source(amplitudes) to the propagator,
    # with pml_width=0 on that same face.

Everywhere else the fold is **not** applied, and a float location whose
stencil would cross the face raises
:exc:`deepwave.DeepwaveNotImplementedError` naming what to do instead, rather
than quietly interpolating against the wrong image. That covers the nine
staggered-grid propagators, whose source and receiver components sit on
staggered nodes where the image is component dependent (Robertsson, 1996);
the ``"periodic"`` faces, which have no image at all; and any call that
propagates an extracted sub-region (``survey_pad``, ``origin``, or an initial
wavefield that sets the extents), where the face is not at the edge of the
model the coordinates are measured against. The two ways round it are to give
such a point an integer (on-node) coordinate in that dimension, which takes
the exact path and needs no stencil, or to keep off-node coordinates at least
``location_interpolation_halfwidth - 1`` cells from the face.

Building the interpolator yourself with the right reflection type and location
reproduces the field of a true point source in the half space to machine
precision (validated against the propagators in the test suite). Using the
wrong type or location injects a slightly mispositioned, wrong-polarity image
and degrades near-surface amplitudes.

An **absorbing** (``"pml"``) face has no image either: the model is padded
with the absorbing layer, but interpolation happens in the coordinates of the
unpadded model, so a stencil that leaves it is out of bounds. The message
reports the coordinate you passed, its grid cell, and the cell the stencil
reaches.

Free surface in the elastic propagator
--------------------------------------

This is the way to put a free surface on an elastic model:
``boundary="free"`` is not accepted by the elastic family or by
``acoustic_vti`` (see :ref:`boundary-faces-accepted`).

The elastic propagator models a true free surface using the "improved vacuum"
method of Zeng et al. (2012). You represent the vacuum above the surface
directly in the model, by setting the stiffness (``mu``, and ``lamb``) of the
cells above the surface to zero. Deepwave's elastic parameter preparation
harmonically averages ``mu`` so that these zero-stiffness cells impose the free
condition at the surface. Combine this with ``pml_width=0`` on that face so the
surface is not absorbing. The ``example_elastic_complex_freesurface`` example
demonstrates this, including a non-flat (topographic) free surface.

The vacuum free-surface method is accurate for typical configurations but
becomes less accurate at steep angles of incidence and where there is a strong
material contrast immediately below the surface. Keep this in mind when
interpreting surface waves and near-surface multiples.

.. note::

   **A bare zero-``pml_width`` elastic edge is not a free surface.** If you
   set ``pml_width=0`` on an elastic face *without* a vacuum layer, the
   truncation leaves both the exterior normal velocity and the exterior shear
   stress at zero, which clamps the normal velocity (it reflects with
   inverted polarity, and the normal stress with :math:`R=+1`) while leaving
   the tangential direction traction-free. That is a **frictionless** rigid
   wall, not a clamped one, and not a stress-free surface either — see
   :ref:`bare-zero-width-face` for the measurements, including the tangential
   velocity, which distinguishes it from ``boundary="rigid"``. A genuine
   elastic free surface requires the vacuum layer described above. (The
   scalar propagator's bare zero-PML edge, by contrast, *is* a free surface.)

.. _vacuum-sentinel:

.. warning::

   **Do not compute** ``buoyancy`` **as** ``1/density`` **for a vacuum.** A
   vacuum cell is one whose ``buoyancy`` is **zero**. That is a *sentinel*, not
   an arithmetic value: literally ``1/density`` would be infinite for a vacuum,
   and a literal infinity is treated as an ordinary, very light cell rather
   than as vacuum. Dividing by a zero density therefore produces a model that
   looks plausible and is not a free surface.

.. note::

   **Vacuum convention, and how to build one.** A vacuum cell is one whose
   ``buoyancy`` is zero; it is treated internally as having zero density and
   zero stiffness.

   The safe way to build one is to work in
   :math:`v_p`/:math:`v_s`/:math:`\rho` and let Deepwave do the conversion.
   :func:`deepwave.vpvsrho_to_lambmubuoyancy` maps ``rho = 0`` to
   ``buoyancy = 0`` rather than to infinity, and zero velocities give
   ``lamb = mu = 0``, so one mask expression produces all three models::

       vacuum = torch.arange(ny).reshape(-1, 1) < surface.reshape(1, -1)
       vp[vacuum] = 0.0
       vs[vacuum] = 0.0
       rho[vacuum] = 0.0
       lamb, mu, buoyancy = deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho)

   :doc:`example_topography` does exactly this for an irregular surface.

   **What the sentinel does at the interface.** Deepwave's elastic scheme is
   staggered, and the models are averaged onto the staggered nodes before the
   kernels see them, so a zeroed cell is not simply a hole. ``buoyancy`` is
   averaged as a density (``buoyancy[i+1/2] = 2 / (rho[i] + rho[i+1])``), so a
   node between a vacuum cell and a solid cell gets **twice** the solid's
   buoyancy -- half the density -- rather than zero; only a node with vacuum on
   both sides is itself zero. That doubling is the improved-vacuum free surface
   working as intended, and is the same halving of the density described for
   the velocity nodes in :doc:`elastic`. ``mu`` is averaged harmonically with a
   hard vacuum mask, so one zeroed ``mu`` cell zeroes all four adjacent shear
   nodes, while ``lamb`` is used unaveraged. In short: set the whole cell to
   vacuum and let the averaging build the surface; do not try to place the
   surface by zeroing individual staggered components.

**Placing sources and receivers near an elastic free surface.** The elastic
free surface is realised by the material properties of real grid cells (the
vacuum cells are part of the domain), so it is *not* a truncated domain
boundary. The image-folding used for the scalar/acoustic reflecting edges
does **not** apply here. Use ordinary interior interpolation
(:class:`~deepwave.Hicks` with ``free_surfaces`` and
``rigid_surfaces`` left at their defaults, i.e. all ``False``) and place sources
and receivers in the solid (the first solid row and below; a receiver placed on a
vacuum cell records zero). The free-surface image symmetry in the elastic case is
*component dependent* (`Robertsson (1996)
<https://doi.org/10.1190/1.1444107>`_): e.g. :math:`\sigma_{yy}` and
:math:`\sigma_{xy}` are odd, :math:`v_y` even, :math:`v_x` odd about the
surface. A single scalar sign-flip fold therefore cannot be correct for all
staggered components simultaneously — the vacuum method encodes each component's parity
implicitly through the moduli/density averaging, so no folding is needed.

``boundary=`` across propagators
---------------------------------

All nine of Deepwave's supported propagators accept the flat-face ``boundary=``
argument. The available face types differ: in particular, the elastic
family rejects ``free`` and ``periodic`` guard-ring faces because their
discrete updates do not conserve energy. ``scalar``, ``scalar_born``,
``acoustic``, ``acoustic_born``, ``viscoacoustic``, ``acoustic_vti``,
``elastic``, ``elastic_vti``, and ``elastic_born`` implement their available
faces on both eager and compiled backends.

The improved-vacuum method described above remains the only way to model an
irregular (non-flat) free surface. ``boundary=`` is a separate, faster
shortcut for a flat, grid-aligned face. It is implemented independently, by
refilling the guard ring outside the domain edge with the exact mirror image
of the interior field, rather than by representing a vacuum region in the
model; the two are checked to agree wherever both apply. See the
appendix (:ref:`boundaries-appendix`) for the full per-propagator
implementation notes, citations, and gate evidence.

.. _boundary-faces-accepted:

Which faces each propagator accepts
------------------------------------

The ``boundary=`` argument is the same on all nine propagators, but the
faces it accepts are not. A face is accepted only where its guard-ring fill
**conserves a discrete energy**, so that a run cannot grow without bound; a
face that does not conserve raises :class:`deepwave.DeepwaveNotImplementedError`
while the arguments are validated, before anything is computed, on every
backend and through the Module form alike.

.. list-table:: Faces accepted, by propagator
   :header-rows: 1
   :widths: 30 14 20 20 12

   * - Propagator
     - ``"pml"``
     - ``"free"``
     - ``"rigid"``
     - ``"periodic"``
   * - ``scalar``, ``scalar_born``
     - every face
     - every face
     - every face
     - yes
   * - ``acoustic``, ``acoustic_born``, ``viscoacoustic``
     - every face
     - beginning of an axis at every accuracy; either face at ``accuracy=2``
     - every face
     - no
   * - ``acoustic_vti`` (pseudo-acoustic VTI)
     - every face
     - no
     - every face
     - no
   * - ``elastic``, ``elastic_vti``, ``elastic_born``
     - every face
     - no
     - beginning of an axis at every accuracy; either face at ``accuracy=2``
     - no

"Beginning of an axis" is the low face, the one at index 0 along that
dimension -- entries 0, 2 and 4 of ``boundary``. The high face of the same
axis is accepted at ``accuracy=2``, where the guard ring is a single cell and
the fill is the exact image, and not at ``accuracy`` 4 or more, where it
reaches deeper and the replacement rows are no longer the transpose of their
partners. A zero-width ``"pml"`` face is always accepted: it is reflecting
(see :ref:`bare-zero-width-face`) but it runs no fill at all, so it conserves.

**What to do instead.**

* *A free surface on an elastic or acoustic_vti model.* Use the
  vacuum layer described in `Free surface in the elastic propagator`_: set
  ``vp``, ``vs`` and ``rho`` (or, for ``acoustic_vti``, the stiffnesses and
  ``buoyancy``) to zero in the cells above the surface, with ``pml_width=0``
  on that face. That is also the only way to model an irregular surface, and
  ``elastic`` adds ``surface_method="pm"`` for a more accurate flat or
  topographic one.
* *A reflecting face at the end of an axis, at ``accuracy`` 4 or more.* Put it
  at the beginning of the axis instead -- flip the model along that axis, and
  the sources, receivers and any initial wavefield with it -- or use
  ``accuracy=2``.
* *A periodic axis.* ``scalar`` and ``scalar_born`` accept ``"periodic"``; no
  other propagator does.

Faces may be accepted in a later release, as their fills are derived so that
they conserve; that is an addition, and nothing accepted here is withdrawn.

Known limitations
------------------

**Elastic long-run PML instability with thin free-surface waveguides.** Elastic
models that combine (a) a thin propagating region sandwiched between two facing
free surfaces (vacuum layers) — for example a thin, high-Poisson's-ratio
near-surface layer — with (b) a PML on a face perpendicular to those free
surfaces, can exhibit slow exponential growth after many thousands of
timesteps, eventually swamping the simulation. This is not a Deepwave-specific
bug: it is a known weakness of ordinary C-PML (including the CFS variant
Deepwave uses) for elastic media, rooted in the geometric PML stability
condition of `Bécache, Fauqueux and Joly (2003)
<https://doi.org/10.1016/S0021-9991(03)00184-0>`_. A mode is unstable whenever
its phase and group velocity have components of opposite sign along the PML
normal, which can happen for evanescent/surface-wave energy trapped between two
nearby free surfaces at high Poisson's ratio.

.. note::

   **A fluid layer over a solid (water-over-seabed / transcranial) is *not*
   subject to this instability.** A common fluid-solid configuration -- a water
   layer (``vs = 0``) over an elastic solid, with a vacuum free surface on top of
   the water -- has only a *single* free surface. The water's lower boundary is a
   fluid-solid interface, which partially transmits energy into the solid rather
   than reflecting it like a second free surface. The Bécache-Fauqueux-Joly
   trapping condition is therefore not met, and such models are long-run stable.
   Direct testing used a water layer over a solid, free surface on top, PMLs
   on the other faces, driven and then run many thousands of steps past the
   end of the source. In both thin and thick water layers, at ``mpml_fraction = 0``
   (no M-PML needed), the total wavefield energy stays bounded -- it neither
   grows nor is spuriously amplified at the interface. This is the
   configuration of Deepwave issue #116: the *oscillations near the
   interface* reported there are a near-interface numerical/physical effect
   (Scholte interface waves and grid dispersion), not a late-time
   instability. See :doc:`ultrasound` for the
   fluid-solid modelling recipe and :doc:`troubleshooting` for reducing the
   oscillations. The genuinely-unstable case remains the thin high-Poisson slab
   between two *facing* free surfaces described above.

Multiaxial PML (``mpml_fraction``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The elastic propagator provides an opt-in multiaxial PML (M-PML), following
`Meza-Fajardo and Papageorgiou (2008)
<https://doi.org/10.1785/0120070223>`_, that addresses this failure mode
directly. It is controlled by the ``mpml_fraction`` argument (the cross-damping
fraction):

* ``mpml_fraction=0`` (the default) is the standard CFS-C-PML; behaviour and
  performance are byte-identical to not having the feature.
* ``mpml_fraction>0`` adds, inside each PML region, a fraction ``mpml_fraction`` of the
  boundary-normal damping to the tangential directions as well (a CFS-M-PML
  hybrid — the CFS frequency shift is retained). This tangential damping is
  what cures the late-time growth.

The trade-off is that M-PML is no longer *perfectly* matched: the reflection
from the layer grows roughly linearly with ``mpml_fraction``, so use the smallest
value that stabilises your model. A value in the range **0.15–0.2** is a good
starting point (it stabilises the reproduction above while keeping the extra
reflection small); increase it only if growth persists. There is no universal
value — the required fraction is medium-dependent — so treat ``mpml_fraction`` as a
stability knob to be tuned per problem.

For a measured 3D elastic VTI example with unequal grid spacing, asymmetric
8--13-cell layers, grazing/corner receivers and a late record, ordinary PML
has 0.040% relative trace error against a larger-domain reference, compared
with 96.1% without absorption. At ``mpml_fraction=0.15``, the trace error is
1.90%; the largest late residual is 11.7% of that receiver's direct peak
(0.29% with ordinary PML). These are scoped measurements, not universal
reflection bounds. Cross damping can be useful for stability while materially
changing a weak receiver component; compare both stability and reflected
amplitudes when choosing the fraction.

M-PML requires at least two spatial
dimensions (the cross-axis damping has no meaning in 1D). It is offered only
on the propagators with the instability-prone physics it targets: the elastic
family (``elastic``, ``elastic_vti``, ``elastic_born``) and
``acoustic_vti`` (the pseudo-acoustic reduction of
``elastic_vti``, which inherits the same anisotropic corner behaviour). It is
not offered on the isotropic scalar/acoustic propagators (``scalar``,
``scalar_born``, ``acoustic``, ``acoustic_born``, ``viscoacoustic``), whose
isotropic PML is already stable. There are no shear waves and no
free-surface waveguide mode to trap, and ``viscoacoustic``'s SLS attenuation
is itself dissipative.

Example::

    out = deepwave.elastic(
        lamb, mu, buoyancy, grid_spacing, dt,
        source_amplitudes_y=source, source_locations_y=source_locations,
        pml_freq=freq,
        mpml_fraction=0.15,   # opt in to multiaxial PML
    )

**A slab bounded by any two reflectors, not only by two vacuum free
surfaces.** The trapping condition needs a slab, not a particular way of
closing it, so the same late growth appears when a reflecting *boundary face*
plays the part of the second surface. Measured: a 2-D ``elastic`` waveguide --
a low-velocity surface layer (:math:`v_p` 1200, :math:`v_s` 400 m/s) over
2000/1000 m/s, 60x300 cells of 5 m, ``dt`` 0.5 ms, float64, with
``boundary=["rigid", "pml", "pml", "pml"]`` and C-PML on the other faces --
grew from about 24,000 time steps onwards at ``mpml_fraction=0``, and stayed bounded
for 60,000 time steps at ``mpml_fraction=0.2``. If a long run in a layered model with
a reflecting top face starts to grow late, ``mpml_fraction`` is the first thing to
try.

If you would rather not enable M-PML, the following workarounds also reduce the
instability without any change to ``mpml_fraction``:

* widen the PML on the affected face(s) (``pml_width``);
* reduce the Poisson's-ratio contrast in the material immediately below the
  free surface, if physically acceptable;
* shorten the simulation, or taper/limit the source, so that less late-time
  energy is present to be amplified;
* move the model boundary (and hence the PML) farther from the free surfaces,
  giving the trapped mode more room to decay before it reaches the PML.

**Elastic free-surface (IVM) accuracy with rugged topography.** The improved
vacuum method (Zeng et al., 2012, used for the elastic free surface as
described above) is accurate for flat surfaces. Per the analysis of
vacuum-family methods in `Cao and Chen (2018)
<https://doi.org/10.1190/geo2018-0098.1>`_, however, its Rayleigh-wave accuracy
degrades with staircase (rugged) topography, and that degradation grows with
Poisson's ratio. If you are modelling rugged topography, particularly in
high-Poisson's-ratio near-surface material, increase the number of points per
wavelength near the surface to compensate.

Practical notes
---------------

* A zero-``pml_width`` edge is reflecting; a non-zero one is absorbing. There is
  no separate flag to turn an edge "off" — absorbing is the default and
  reflecting is requested by zero width.
* Because the reflecting edge in the scalar propagator inverts polarity (it is a
  free surface), a surface reflection (or surface multiple) recorded just below
  such an edge will have opposite sign to the direct arrival. The acoustic
  reflecting edge is a rigid surface and preserves polarity, so its surface
  reflection has the same sign as the direct arrival — a useful way to tell which
  edge condition you are actually using.
* When you crop the model for efficiency with ``survey_pad`` or ``origin``, the
  PML is still added around the cropped simulation domain, so interior crop
  faces remain absorbing.

.. _boundaries-appendix:

Appendix: image-rule derivations and implementation policy
=============================================================

This appendix is verification and implementation detail: the exact per-field
image formulas the guard-fill uses (odd/even parity, adjoint mechanics), and
the policy and evidence trail for how the flat-face ``boundary=`` argument is
implemented and validated for each propagator. It is written for a maintainer
extending ``boundary=`` support to a new propagator, or a reader who wants to
check the mathematics or the test evidence directly. None of it is required
to use Deepwave's boundary conditions -- see the sections above for that.

.. _ghost-plane-conventions:

Ghost-plane conventions (the image rules)
--------------------------------------------

Let :math:`P` be the stencil half-width (``accuracy // 2``), so the model
interior occupies indices :math:`[P,\,n-P)` of the padded grid and the guard
ring is the outer :math:`P` cells on each side. For the beginning face of a
dimension (the end face is the mirror image):

**Scalar (nodal grid, field** :math:`u` **).**

* ``"free"`` (pressure release, :math:`u=0`): :math:`u` is **odd** about the
  node plane one cell outside the model edge (grid node :math:`P-1`). The image
  fill is :math:`u_{\text{guard}}[g] = -u[\,2P-2-g\,]` for
  :math:`g=0,\dots,P-1`; the node on the plane (:math:`g=P-1`) is zero. This is
  the exact continuation of the current scalar zero-PML edge (identical at
  accuracy 2), so :math:`R=-1` at every order.
* ``"rigid"`` (:math:`\partial_n u = 0`): :math:`u` is **even** about the
  half-cell plane :math:`P-\tfrac12` (no node lies on the plane, so there is no
  under-determined boundary value). The fill is
  :math:`u_{\text{guard}}[g] = +u[\,2P-1-g\,]`, giving :math:`R=+1`.
* ``"periodic"``: :math:`u_{\text{guard}}[g] = u[\,g+m\,]` with period
  :math:`m=n-2P` (the interior/model length); exactly translation-invariant.

**Acoustic (staggered grid: pressure** :math:`p` **at integer nodes, velocity
components at the half-nodes; low pad** :math:`P` **, high pad** :math:`P-1`
**).** The staggered image rules follow from the first-order system
:math:`\partial_t p = -K\,\nabla\!\cdot v`,
:math:`\partial_t v = -b\,\nabla p`, which fix each field's parity from the
imposed condition:

* ``"rigid"`` (:math:`v_n=0`, wall at the half-cell plane, the outer
  velocity node): **pressure even**, **normal velocity odd** (its node sits on
  the wall and is held to zero), **tangential velocity even**. This coincides
  with the current acoustic zero-PML edge at accuracy 2 (:math:`R_p=+1`).
* ``"free"`` (:math:`p=0`, surface at the first interior pressure node):
  **pressure odd** (the surface pressure node is held to zero), **normal
  velocity even**, **tangential velocity odd** (its node on the surface is held
  to zero). Gives :math:`R_p=-1` — a genuine pressure-release surface for the
  acoustic propagator, previously only available (as a true free surface) in the
  elastic propagator via the vacuum method.
* ``"periodic"``: index wrap with period equal to the model length, applied to
  every component.

Because only the boundary-normal derivative of each field is taken across a
face (the divergence/gradient stencils are dimension-separable), a face in
dimension :math:`d` only needs the pressure and the dimension-:math:`d` velocity
guard cells refilled along axis :math:`d`; corner guard cells and the
tangential-velocity guard cells along that axis are never read by an interior
update.

**Adjoint.** The guard fill is a linear operator, and the pure-PyTorch
backend's autograd supplies its exact adjoint automatically (verified by
gradcheck for every face type). The compiled backend hand-implements the
transpose; see the note below for how it does so per face combination.

.. note::

   **Current availability.** The first-class ``boundary`` argument is
   implemented for the **scalar, acoustic, scalar_born, and acoustic_born**
   propagators in **both** backends: the pure-PyTorch backend
   (``backend="eager"``) and the compiled C/CUDA backend. Each is
   validated against the analytic reflection coefficients at every accuracy
   order 2-8 (``free`` :math:`\to R=-1`, ``rigid`` :math:`\to R=+1`, to
   better than 1% at high points-per-wavelength), the periodic energy
   budget (exact translation-invariance and wrap-around), and gradchecked
   for every face type in 1-D/2-D/3-D -- with the compiled and eager
   backends agreeing to machine precision (forward fields and gradients).
   The acoustic image rules above (pressure even/odd, normal velocity
   odd/even with the on-plane node held to zero) are what the
   staggered-grid guard-fill applies; the compiled backend threads them
   through a per-face ``bc`` code and applies the same guard-ring fill
   every step. For the compiled adjoint, re-applying the forward fill to
   the adjoint fields (with the model coefficients mirror-padded into the
   guard ring) realises the transpose only for an axis whose faces are
   (``free``-low, ``rigid``-high), which keeps that re-fill path; every
   other combination of ``free``/``rigid``/``periodic`` faces instead gets
   the exact per-operation transpose of its fill — the adjoint-stencil
   contributions that fall on the guard cells are gathered, then each
   guard cell is folded back into the source node the forward fill copied
   it from (with the fill's sign) and zeroed, with the transposes of the
   fill's on-surface/on-wall interior-node writes applied last — and the
   source-gradient read and the buoyancy-gradient accumulation are ordered
   after the fold (see the ``adj_fold_*``/``adj_gather_*`` functions in
   ``acoustic_kernels.h``; the periodic case additionally needs this
   because its seam nodes carry independent model parameters, so the
   operator is not circulant and no re-fill can be its transpose).
   Since ``scalar_born``/``acoustic_born`` now both
   thread ``bc`` fully, ``scalar``'s and ``acoustic``'s forward-mode AD
   (``torch.func.jvp``) and second-order derivatives (Hessian-vector
   products) on the compiled backend -- both of which route through their
   respective Born backend -- support non-PML faces too, validated against
   the eager backend and gradcheck. The zero-``pml_width`` reflecting edge
   (below) also remains available in every backend and every propagator.

Policy: flat-face ``boundary=`` vs. the improved-vacuum method
----------------------------------------------------------------

This section states the policy governing how the first-class ``boundary=``
argument (flat, grid-aligned free/rigid/periodic faces) is implemented, and
how it relates to the improved-vacuum method (IVM) used for irregular free
surfaces. Read it before extending ``boundary=`` support to a new propagator.

**Support matrix.** This subsection is about *backends*: which backend of each
propagator carries the guard-fill implementation. Which face *types* a
propagator offers is a separate and narrower question, answered by
:ref:`boundary-faces-accepted` -- a face whose fill does not conserve a
discrete energy is refused even where the implementation below exists.

All nine of Deepwave's propagators support the flat-face ``boundary=``
argument: ``scalar``, ``scalar_born``, ``acoustic``,
``acoustic_born``, ``viscoacoustic``, ``acoustic_vti``, ``elastic``,
``elastic_vti``, and ``elastic_born`` -- support it on the eager
(``backend="eager"``) and default compiled C/CUDA backends.
:func:`deepwave.acoustic`'s and :func:`deepwave.elastic`'s forward-mode AD
(``torch.func.jvp``) and second-order derivatives (Hessian-vector products) on
the compiled backend -- which route through the ``acoustic_born``/
``elastic_born`` compiled kernels as an implementation detail -- also support
non-PML faces, since those Born-family compiled backends carry the same
guard-fill support. ``acoustic_born``'s and ``elastic_born``'s own ``jvp``/
second-order rules remain unconditionally unsupported regardless of
``boundary`` (see each propagator's own docstring). The zero-``pml_width``
reflecting edge described above also remains available in every backend and
every propagator.

* **The improved-vacuum method (IVM) remains the canonical way to represent an
  irregular or material elastic free surface** (arbitrary topography, or any
  free surface that does not coincide with a domain edge). IVM is the
  "Free surface in the elastic propagator" mechanism described above: the
  surface is encoded once, in backend-independent parameter preparation, as a
  zero-modulus/zero-buoyancy vacuum region in the model that every propagator
  and backend already supports generically, with zero free-surface logic in
  any time-stepping kernel. The flat-face ``boundary=`` argument does not
  replace this: IVM is still the only way to model a non-flat free surface.

* **``boundary=["free", ...]``-style flags are a separate, flat-face idiom**:
  a fast, first-class shortcut for the common case of a flat (grid-aligned,
  lying exactly on a domain edge) free or rigid face. It is implemented by the
  stress/velocity(-and-memory-variable) image method — refilling the guard
  ring outside the domain edge with the exact mirror image of the interior
  field every step — **not** by IVM, because a flat face has no vacuum region
  to encode and the image method is markedly cheaper (no extra model arrays,
  no extra grid cells) for this restricted case. The two mechanisms solve
  different problems (arbitrary topography vs. a fast flat-edge shortcut) and
  are not interchangeable in general, but they must agree where their domains
  of applicability overlap (see the next point).

* **On the elastic family** (any propagator with a genuine stress tensor --
  ``elastic``, ``elastic_vti`` and their Born counterpart ``elastic_born``,
  whose stress tensor and velocity
  fields are elastic's own, unmodified): ``boundary="free"`` is **not
  accepted** (see :ref:`boundary-faces-accepted`). A per-field stress-image
  guard fill for it, imaging each stress and memory-variable component that
  must vanish at the surface per Robertsson's (1996) image-method-3
  (component-dependent: e.g. :math:`\sigma_{yy}` and :math:`\sigma_{xy}` odd,
  :math:`v_y` even, :math:`v_x` odd about the surface), does not conserve a
  discrete energy at the replacement rows, so the face raises
  :class:`deepwave.DeepwaveNotImplementedError` while the arguments are
  validated rather than running a scheme that can grow without bound. The
  vacuum-cell IVM construction above is the free surface these propagators
  offer, and it is not restricted to a flat face; ``elastic`` additionally
  offers ``surface_method="pm"``. ``"rigid"`` *is* accepted on them, at the
  beginning of an axis at every accuracy and at either face at
  ``accuracy=2``.

* **viscoacoustic (and any other pressure-only propagator) has no IVM
  analogue.** IVM represents a free surface via zero shear/normal moduli in a
  vacuum region — a construction specific to media with a shear/normal stress
  tensor. A pressure-only propagator has no such tensor (only a scalar
  pressure, velocities, and — for viscoacoustic — a scalar SLS memory
  variable), so there is no vacuum-cell IVM construction to fall back on for
  an irregular pressure-only free surface, and no flat-vs-IVM parity gate to
  satisfy. For viscoacoustic, ``"free"``/``"rigid"`` are directly analogous to
  the existing acoustic staggered-grid image rule (pressure odd/even about the
  surface node, as documented above), extended only by the treatment of the
  SLS memory variable ``r``: because no spatial derivative of the memory
  variable appears in the velocity/stress/memory update equations (Robertsson
  1996), the memory variable above the surface needs no independent imaging
  of its own — it is simply treated like the pressure component it
  accompanies.

* **acoustic_vti has two independent "pressure-like" fields, each governed by
  the acoustic per-axis image rule applied on its own restricted face set.**
  ``acoustic_vti`` (the pseudo-acoustic VTI propagator, :math:`V_S = 0`) is
  structurally part of the elastic/IVM family — it implements a true free
  surface via the improved-vacuum method (vacuum cells, zero buoyancy) exactly
  like ``elastic``/``elastic_vti`` — but it has no shear stress at all (the
  :math:`V_S = 0` reduction removes it entirely), so the elastic family's
  shear-component parity above does not apply, and the flat-``"free"``-vs-IVM
  numerical-agreement gate above is scoped to ``elastic``/``elastic_vti``
  specifically, not to ``acoustic_vti``. Its two normal stresses — ``sigma_v``
  (axial, dimension 0: ``z`` in 3-D, ``y`` in 2-D) and ``sigma_h``
  (transverse) — are each fed by, and feed back into, a disjoint set of
  velocity components (see the propagator's own module docstring, "CONTINUOUS
  SYSTEM"): ``sigma_v`` couples only to the axis velocity, ``sigma_h`` only
  to the transverse velocities. Applying the acoustic per-axis image rule
  (this page's "Acoustic" ghost-plane convention) independently to each
  field, restricted to the face set it actually couples to, is therefore a
  direct, citation-free-of-invention consequence of the propagator's own
  already-documented constitutive equations, not a new physics guess:
  ``sigma_v`` is guard-filled (odd/even/periodic, per face type) only on the
  axis face, ``sigma_h`` only on the transverse faces; each velocity is
  imaged on its own axis exactly as in ``acoustic``. The compiled adjoint's
  coupling term (the off-diagonal stiffness ``c13``, which appears in every
  velocity's adjoint update) needs no boundary treatment of its own: it
  always multiplies whichever stress is *not* guard-filled on that face,
  which is therefore exactly zero in its own ghost region there (nothing ever
  reads it, forward or backward, on that gridwork) — see
  ``_boundaries.mirror_pad_acoustic_vti_models`` for the full derivation.
  Validated (``tests/test_boundary.py``'s ``test_acoustic_vti_*``): analytic
  reflection coefficients ``R = -1``/``+1`` to <1% at accuracy 2–8 on both the
  axis and a transverse face (an isotropic medium plus a periodic-transverse/
  uniform-line-source construction collapses the 2-D domain to an exact 1-D
  plane wave, avoiding a 2-D point-source Green's-function correction),
  periodic translation invariance, ``boundary=None`` bit-identity, and
  compiled-vs-eager FD gradcheck across every face type including mixed
  PML/reflecting-axis cases.

* **No invented parities.** Every per-field image parity decision (which
  fields get odd/even mirroring, which stay one-sided, which need no imaging
  at all) is justified either by Robertsson's (1996) image-method-3 (for the
  elastic family) or, for the pressure-only case, by direct analogy to the
  existing scalar/acoustic image rule documented above — never guessed. If a
  specific point is ambiguous in the source, that must be treated as an open
  question rather than resolved by assumption.

* **elastic: the flat-face parity table.** Per axis :math:`A`, exactly six of
  the :math:`2n_{\text{dim}}+1` wavefields are ever read across that face
  (:math:`\sigma_{AA}`, the two shears sharing axis :math:`A`, and *every*
  velocity component, including the tangential ones — verified by inspecting
  every term of the continuous system that differentiates in the :math:`A`
  direction; the other three normal/shear stresses need no guard fill at
  all). Per Robertsson (1996):

  * :math:`\sigma_{AA}`: odd about the on-surface node, with a hard zero on
    the surface node itself (exactly
    ``_boundaries.fill_acoustic_pressure_boundary``'s ``p`` treatment) —
    the physical free-surface condition requires :math:`\sigma_{AA}=0` at the
    surface, imaged as odd.
  * The shared-axis shears: odd about the half-cell plane, with no on-plane
    zero.
  * *Every* velocity component's ghost region: literally zeroed, not
    mirrored — Robertsson states this generically for particle velocities
    above the surface (not just the axis-normal component), and it is also
    the discrete counterpart of IVM's own flat-surface behaviour: setting
    ``buoyancy=0`` above the surface makes the velocity update there a
    no-op, i.e. the same "velocities are zero above the surface" outcome
    falls out of a material property rather than an explicit imaging step.
  * Rigid (not covered by Robertsson, a free-surface paper) is derived by the
    same stress<->velocity role-swap duality the acoustic/viscoacoustic/
    acoustic_vti precedents use for their own free/rigid pairs.

  This per-field, discrete guard-ring zero-velocity fill is not in tension
  with the "Free surface in the elastic propagator" note above (on the Hicks
  source/receiver-folding parity used near an IVM surface), which states
  that the axis velocity is even and the tangential velocity is odd about
  the surface for the *same two fields* — i.e. mirror images, not the
  zero-fill used here. These describe two different problems: that note is
  explicitly scoped to Hicks stencil-tap folding, which approximates the
  *continuous* half-space Green's function for taps that reach past a
  source/receiver location (and, as already stated above, does not apply to
  the elastic/IVM case at all), whereas this section's guard-ring fill is a
  discrete image-method device for cells outside the physical domain that no
  interior stencil ever reads as a physical field value. A free surface does
  not physically pin velocity to zero (that would suppress Rayleigh-wave
  motion and behave like a rigid wall); the zero-ghost fill works numerically
  because the free-surface condition is fundamentally a stress condition
  (:math:`\sigma=0`), and the correctly-imaged (odd) stress, read by the
  interior stencil at/near the real surface row, is what reproduces the true
  nonzero surface velocity at actual grid points.

  **Gates.** ``"free"`` is not an accepted face here (see
  :ref:`boundary-faces-accepted`), so what is gated is ``"rigid"``, the
  reflecting face this propagator does accept. It is adjudicated using the
  same methodology as ``_acoustic_vti_reflection_coefficient`` (a
  periodic-line source/receiver collapses the domain to an exact 1-D plane
  wave; an incident-only run is subtracted to isolate the reflected
  arrival), applied through the *same* public :func:`deepwave.elastic` API on
  both sides of the comparison:

  * ``tests/test_boundary.py``'s
    ``test_elastic_rigid_reflection_coefficient``: analytic ``R = +1`` on
    ``sigma_AA`` to <1%, at accuracy 2-8.
  * The free surface these propagators do offer -- the vacuum-cell IVM
    construction, and ``elastic``'s ``surface_method="pm"`` -- is gated
    separately, by the Rayleigh-Lamb oracles in
    ``tests/test_elastic_lamb.py`` and the topography benchmarks in
    ``tests/test_elastic_pm.py``, neither of which uses
    ``boundary="free"``.

  **Implementation.** Both the eager backend (``backend="eager"``) and
  the default compiled C/CUDA backend support ``boundary``; :func:`deepwave.elastic`
  and :class:`deepwave.Elastic` accept the ``boundary`` keyword at the same
  position/with the same semantics as :func:`deepwave.acoustic`'s. Eager is
  wired through ``_elastic_eager_core`` via
  ``_boundaries.fill_elastic_stress_boundary``/
  ``_boundaries.fill_elastic_velocity_boundary``. The compiled backend
  threads a ``bc[2*ndim]`` int64 parameter (``2*ndim`` individual scalar
  arguments in begins-then-ends order, matching acoustic's/viscoacoustic's/
  acoustic_vti's own ABI convention — see ``_backend_utils.py``) through
  ``elastic.c``/``elastic.cu``'s ``FUNC(forward)``/``FUNC(backward)`` for
  both the base variant (ndim 1/2/3) and the M-PML variant (ndim 2/3, per
  ``CMakeLists.txt``'s own gating); the fill logic
  (``FUNC(fill_elastic_stress_bc)``/``FUNC(fill_elastic_velocity_bc)``) is a
  direct, per-line port of the parity table above, and the compiled
  backward's near-boundary model reads are covered by
  ``_boundaries.mirror_pad_elastic_models``. ``boundary=None`` (the
  default) is unaffected and bit-identical to before on every backend, and
  compiled/eager agree on both outputs and gradients (FD gradcheck plus
  direct compiled-vs-eager comparison, ``tests/test_boundary.py``'s
  ``test_elastic_c_gradcheck_and_matches_eager``). jvp (forward-mode AD) and
  the compiled second-order (HVP) path go through the separate
  ``elastic_born`` compiled backend (see its own entry below), mirroring the
  ``scalar``/``acoustic`` jvp/HVP routing.

* **elastic_vti: the same flat-face parity table as isotropic elastic,
  unchanged by VTI's stiffness.** ``elastic_vti`` is a full elastic-family
  propagator, not a pseudo-acoustic reduction like ``acoustic_vti`` -- it has
  genuine shear stress and uses IVM for its own irregular free surface (the
  same "Free surface in the elastic propagator" mechanism described above,
  since ``elastic_vti`` shares ``elastic``'s staggered-grid layout and
  reuses its parameter-preparation machinery). It is therefore treated as
  part of "the elastic family" for both elastic-specific gates
  (flat-``"free"``-vs-IVM agreement, and the Rayleigh-wave-speed test),
  unlike ``acoustic_vti`` (whose :math:`V_S = 0` reduction has no shear
  stress at all, and is explicitly carved out of both gates above).

  Robertsson's per-field image parity (``sigma_AA`` odd + on-surface zero;
  the two shears sharing axis ``A`` odd, no zero; every velocity component's
  ghost zeroed) is a statement about which fields are stresses vs.
  velocities and which axis a field differentiates across -- both
  geometric/topological facts about the staggered-grid layout, independent
  of the numeric stiffness coefficients used to compute the stress-rate
  *updates* from those fields. ``elastic_vti``'s own module docstring
  ("CONTINUOUS SYSTEM") confirms the coupling *shape* needed for this
  argument to transfer unchanged: every normal stress ``sigma_AA`` is still
  driven only by axis-divergence terms (``d[0]``, ``d[1]``, ``d[2]`` --
  just with independent VTI coefficients ``c11``/``c12``/``c13``/``c33``
  instead of the two-coefficient isotropic ``lamb``/``lamb + 2 mu``
  combination), and each shear stress is still driven only by its own two
  cross-derivative terms (``c55``/``c66`` instead of ``mu``). Since the set
  of *differentiated fields* per axis is unchanged from isotropic elastic,
  Robertsson's parity table -- and therefore
  ``_boundaries.fill_elastic_stress_boundary``/
  ``_boundaries.fill_elastic_velocity_boundary`` -- apply to
  ``elastic_vti`` **verbatim, with no VTI-specific re-derivation**: the
  eager backend reuses those two functions completely unchanged, and the
  compiled C/CUDA guard-fill functions
  (``FUNC(fill_elastic_stress_bc)``/``FUNC(fill_elastic_velocity_bc)`` in
  ``elastic_vti.c``/``elastic_vti.cu``) are a direct, unchanged-arithmetic
  copy of elastic's own (necessarily a separate copy, not a shared
  translation unit, mirroring how elastic's own fill functions were
  themselves a fresh copy of acoustic.c's ``FUNC(fill_p_bc_line)``/
  ``FUNC(fill_v_bc_line)`` rather than a cross-file reuse).

  ``elastic_vti`` has no 1-D form (``CMakeLists.txt`` builds its base and
  M-PML variants only for ``DW_NDIM`` 2 and 3, unlike isotropic elastic's
  base variant, which additionally supports ``DW_NDIM=1``), so the
  compiled guard-fill functions need no
  ``DW_NDIM==1`` branch at all -- simpler than elastic's own version of the
  same functions.

  The compiled backward's near-boundary model reads are covered by
  ``_boundaries.mirror_pad_elastic_vti_models``, which follows
  ``_boundaries.mirror_pad_elastic_models``'s own safe/conservative
  choice verbatim (every stress-role stiffness -- ``c11``/``c12``/``c13``/
  ``c33``/``c55_*``/``c66_yx`` -- mirror-padded on *every* reflecting axis,
  a strict superset of what any single compiled adjoint kernel actually
  reads at a neighbour offset per ``elastic_vti_kernels.h``'s own documented
  addressing convention, so never wrong; each buoyancy mirror-padded on its
  own axis only) rather than acoustic_vti's more surgical, field-specific
  optimisation (which required proving a specific coefficient's paired
  stress is always zero in its own ghost region -- a more intricate case
  analysis for VTI's multi-coefficient coupling than is needed here).

  Both gates are satisfied using an **isotropic-reduction VTI medium**
  (``c11 = c33 = lamb + 2 mu``, ``c13 = lamb``, ``c55 = c66 = mu``) run
  through the *same* public :func:`deepwave.elastic_vti` API on every side of every
  comparison, rather than a genuinely anisotropic oracle: a P-wave
  travelling along the symmetry axis (or, in this isotropic-reduction case,
  any axis) is physically identical to isotropic elastic's own
  already-validated case, so the identical analytic ``R = -1``/``+1`` reflection
  coefficients and Rayleigh-Lamb S0 root apply unchanged, while still
  exercising ``elastic_vti``'s own model set, guard-fill wiring, and
  compiled kernels end to end (rather than isotropic elastic's). This
  reduction is a deliberate scope-limiting choice, not a physics gap: a
  genuinely anisotropic flat-face oracle would need new analytic
  reflection-coefficient/surface-wave derivations (anisotropic Rayleigh-wave speed has
  no simple closed form in general) that are out of scope here.

  * ``"free"`` is not an accepted face on ``elastic_vti`` (see
    :ref:`boundary-faces-accepted`), so ``"rigid"`` is what is gated:
    ``tests/test_boundary.py``'s
    ``test_elastic_vti_rigid_reflection_coefficient`` (eager) and its
    ``_c_`` compiled-backend counterpart give the analytic ``R = +1`` on
    ``sigma_AA`` to <1%, at accuracy 2-8. The vacuum-cell IVM free surface,
    which ``elastic_vti`` does offer, is gated by the Rayleigh-Lamb oracles
    in ``tests/test_elastic_lamb.py``, which use no ``boundary`` argument.
  * Periodic translation invariance
    (``test_elastic_vti_periodic_translation_invariance``/
    ``test_elastic_vti_c_periodic_translation_invariance``,
    <1e-9), with ``c55 > 0`` (shear dynamically active) so the check is
    non-trivial: this specifically exercises, for ``elastic_vti``'s own
    ``FUNC(fill_elastic_shear_line)`` copies in both ``elastic_vti.c`` and
    ``elastic_vti.cu``, the "extra high-side interior node slaved to wrap"
    step -- a periodic-boundary detail that is easy to omit and silently
    breaks translation invariance if it is (verified explicitly here, not
    assumed).
  * ``boundary=None`` bit-identity (eager and compiled), compiled-vs-eager
    FD gradcheck across every face type including mixed PML/reflecting-axis
    cases
    (``test_elastic_vti_boundary_gradcheck``/``test_elastic_vti_c_gradcheck_and_matches_eager``).

  **Implementation.** Both the eager backend (``backend="eager"``)
  and the default compiled C/CUDA backend support ``boundary``.
  :func:`deepwave.elastic_vti` and :class:`deepwave.ElasticVti` accept the ``boundary``
  keyword at the same position/with the same semantics as :func:`deepwave.elastic`'s.
  elastic_vti's autograd architecture is simpler than isotropic elastic's
  (a single ``ElasticVtiForwardFunc``, no separate Born-routed jvp/HVP path
  to guard -- ``jvp``/second-order are unconditionally unsupported
  regardless of ``boundary``), so the wiring follows ``acoustic_vti``'s own
  ``boundary=`` pattern (a ``boundary`` parameter threaded into
  ``_engine.run_forward_engine``/``_engine.run_backward``, the
  latter via a ``model_transform`` closure calling
  ``_boundaries.mirror_pad_elastic_vti_models``) rather than isotropic
  elastic's more complex two-``Function`` structure. ``boundary=None`` (the
  default) is unaffected and bit-identical to before on every backend.

* **acoustic_born: the background and scattered fields are each an ordinary
  acoustic staggered-grid field, so the acoustic parity table transfers
  unchanged to both, independently.** ``acoustic_born``'s background field
  ``(p, v_d)`` solves exactly :func:`deepwave.acoustic`'s own system; its
  scattered field ``(dp, dv_d)`` solves the *same* acoustic system on the
  *same* grid, driven only by a virtual source built from the scattering
  perturbations ``dK``/``dB_d`` (see the module docstring). Both fields
  therefore obey the identical acoustic per-axis image rule documented above
  (pressure odd/even about the surface node, normal velocity even/odd with
  the on-plane node held to zero, tangential velocity untouched) -- this is
  a direct analogy, not a new physics derivation (``acoustic_born``/
  ``elastic_born`` are the coupled-field siblings of the already-covered
  single-field propagators, so no citation beyond what
  :func:`deepwave.acoustic`'s own entry above already establishes is
  needed).

  **Eager backend**: ``acoustic_born_python`` calls
  ``_boundaries.fill_acoustic_pressure_boundary``/
  ``fill_acoustic_velocity_boundary`` -- the exact same functions
  :func:`deepwave.acoustic`'s own eager core uses -- twice per step, once
  on the background field arrays and once on the scattered field arrays,
  at the same points in the coupled velocity/pressure update loop that
  :func:`deepwave.acoustic`'s single-field loop uses.

  **Compiled backend**: ``acoustic_born.c``/``acoustic_born.cu`` use a
  direct, unchanged-arithmetic copy of ``acoustic.c``'s
  ``FUNC(fill_p_bc_line)``/``FUNC(fill_v_bc_line)`` guard-fill functions
  (a fresh copy, not shared across translation units, mirroring every
  other propagator's own guard-fill precedent), called on the background
  field pointers then the scattered field pointers at the same points in
  the coupled time loop as the eager driver, for all three compiled entry
  points (the coupled ``forward``, the scattered-only ``backward_sc``, and
  the full two-field ``backward``). The adjoint's near-boundary model reads
  are covered by ``_boundaries.mirror_pad_acoustic_models`` -- the
  same function :func:`deepwave.acoustic`'s own adjoint uses -- applied
  independently to the background model group ``[K, B_*]`` and the
  scattering model group ``[dK, dB_*]`` (each plays the identical
  "pressure"/"velocity" structural role on the same staggered grid, so
  the same function applies to both without modification). ABI: a
  ``bc[2*ndim]`` int64 parameter (``2*ndim`` individual scalar arguments,
  begins-then-ends order) threaded through all three compiled entry
  points, matching every other propagator's own ABI convention.

  This also lifts the non-PML-face restriction on :func:`deepwave.acoustic`'s
  own compiled forward-mode AD (``torch.func.jvp``/
  ``torch.autograd.forward_ad``) and second-order differentiation
  (double-backward / Hessian-vector products via ``create_graph=True``):
  both route through this module's compiled backend (jvp through the
  forward pass only; the HVP through one forward *and* one full two-field
  backward), and both now thread the same per-face ``bc`` codes as the
  primal/first-order adjoint through to the Born calls, instead of
  refusing with ``NotImplementedError`` for any non-PML face.

  .. note::

     **Compiled-vs-eager gradient checks are sensitive to numerical
     conditioning.** A discrepancy that grows with accuracy order and with
     the number of ``"rigid"``/``"periodic"`` faces can look like a
     correctness bug specific to a boundary implementation, but before
     suspecting new code, check whether the same discrepancy reproduces on
     an already-shipped, unrelated propagator using the same ``dt`` and
     perturbation magnitude. An under-conditioned choice (e.g. a very small
     ``dt`` together with an unbounded Gaussian velocity perturbation) can
     produce a relative discrepancy of ``1e-5`` to ``1e-3`` even with no
     boundary-related bug present, purely from numerical conditioning,
     whereas a well-conditioned choice (a larger, bounded perturbation --
     matching e.g. ``test_acoustic_c_gradcheck_and_matches_eager``'s own
     ``dt``/perturbation choice) agrees to machine precision (``~1e-13``
     relative) on every propagator and every face type, including
     ``"rigid"``/``"periodic"`` corners. A larger-than-expected
     compiled-vs-eager gradient discrepancy is, before anything else, worth
     checking against the *same* configuration run through an
     already-shipped propagator (or a smaller, better-conditioned ``dt``/
     perturbation choice) before concluding new code is at fault.

  Validated (``tests/test_boundary.py``):
  ``test_acoustic_born_free_reflection_coefficient``/
  ``test_acoustic_born_rigid_reflection_coefficient``
  (eager) and their ``_c_`` compiled counterparts: analytic
  ``R = -1``/``+1`` to <1% at accuracy 2-8, on *both* the background and
  the scattered pressure receiver (a single point scatterer past the
  receiver; the background field's reflection is measured exactly as in
  :func:`deepwave.acoustic`'s own reflection test, and the scattered
  field's reflection is isolated the same way :func:`deepwave.scalar_born`'s
  own ``_born_reflection_coefficient`` isolates it -- an
  incident-vs-reflecting-face subtraction applied to the scattered receiver directly,
  which is possible because the scattered field obeys the same boundary
  condition as an ordinary field on the same grid, not because any new
  parity was invented for it). Periodic translation invariance (background
  and scattered together, source/receiver/scatterer shifted as one unit),
  eager and compiled. ``boundary=None`` bit-identity (eager, and
  eager-vs-compiled, both backends). Absolute-FD gradcheck plus compiled-vs-eager
  gradient agreement w.r.t. ``v``, ``rho``, ``dv``, and the shared source
  amplitude together, across every face type including mixed PML/
  reflecting corners, 1-D through 3-D
  (``test_acoustic_born_c_gradcheck_and_matches_eager``).
  ``test_acoustic_c_jvp_supports_non_pml``/
  ``test_acoustic_c_hvp_supports_non_pml`` cover the lifted forward-mode/
  second-order restriction on :func:`deepwave.acoustic` itself. Both
  standalone ``nvcc -c`` compile-cleanliness and live execution of the
  compiled CUDA kernels on GPU hardware (comparing against the CPU/eager
  backend) were verified for this propagator.

* **elastic_born: the background and scattered fields are each an ordinary
  elastic staggered-grid field, so the isotropic elastic parity table
  transfers unchanged to both, independently.** ``elastic_born``'s
  background field ``(v, sigma)`` solves exactly :func:`deepwave.elastic`'s
  own system; its scattered field ``(dv, dsigma)`` solves the *same*
  isotropic elastic system on the *same* grid, driven only by a virtual
  source built from the background field's own strain-rate/stress-divergence
  terms times the scattering models (``dlamb``/``dmu``/``dbuoyancy_i``, see
  the module docstring and :doc:`elastic_born_derivation`). Both fields
  therefore obey the identical per-field image parity documented above on
  ``elastic`` (``sigma_AA`` odd + on-surface zero; the two shears sharing
  axis ``A`` odd, no zero; every velocity component's ghost zeroed) -- this
  is a direct analogy, not a new physics derivation, generalising the same
  reasoning ``acoustic_born``'s own entry above uses from the acoustic
  staggered pair to elastic's full stress tensor.

  **Eager backend**: ``elastic_born_eager_core`` calls
  ``_boundaries.fill_elastic_stress_boundary``/
  ``fill_elastic_velocity_boundary`` -- the exact same functions
  :func:`deepwave.elastic`'s own eager core (``_elastic_eager_core``) uses
  -- twice per step, once on the background field arrays and once on the
  scattered field arrays, at the same points in the coupled velocity/stress
  update loop that :func:`deepwave.elastic`'s single-field loop uses (stress
  guard fill before the receiver record and the velocity update; velocity
  guard fill after velocity-source injection and before the stress update).

  **Compiled backend**: ``elastic_born.c``/``elastic_born.cu`` use a
  direct, unchanged-arithmetic copy of ``elastic.c``'s
  ``FUNC(fill_p_bc_line)``/``FUNC(fill_elastic_onrow_velocity_line)``/
  ``FUNC(fill_elastic_offrow_velocity_line)``/``FUNC(fill_elastic_shear_line)``/
  ``FUNC(fill_elastic_stress_bc)``/``FUNC(fill_elastic_velocity_bc)``
  guard-fill functions (a fresh copy, not shared across translation
  units), called on the background field pointers then the scattered field
  pointers at the same points in the coupled time loop as the eager driver,
  for all three compiled entry points (the coupled ``forward``, the
  scattered-only ``backward_sc``, and the full two-field ``backward``) in
  both ``.c`` and ``.cu`` (elastic_born's CUDA adjoint kernels are real --
  see the ``backward_sc``/``backward`` ``extern "C"`` entry points in
  ``elastic_born.cu``). The adjoint's near-boundary model reads use
  ``_boundaries.mirror_pad_elastic_models`` -- the same function
  :func:`deepwave.elastic`'s own adjoint uses -- applied independently to
  the background model group (``[lamb, mu, mu_*, buoyancy_*]``) and the
  scattering model group (``[dlamb, dmu, dmu_*, dbuoyancy_*]``) (each plays
  the identical "elastic model set" role on the same staggered grid, so the
  same function applies to both without modification), in both the
  production dispatch (``ElasticBornForwardFunc._backward_sc_dispatch``/
  ``_backward_full_dispatch``) and the standalone test/diagnostic
  bridges (``_elastic_born_backward_sc_compiled``/
  ``elastic_born_backward_compiled``, the latter also called directly
  by :func:`deepwave.elastic`'s own HVP -- see below). ABI: a
  ``bc[2*ndim]`` int64 parameter (``2*ndim`` individual scalar arguments,
  begins-then-ends order) threaded through all three compiled entry points,
  matching every other propagator's own ABI convention.

  This also lifts the non-PML-face restriction on :func:`deepwave.elastic`'s
  own compiled forward-mode AD (``torch.func.jvp``/
  ``torch.autograd.forward_ad``) and second-order differentiation
  (double-backward / Hessian-vector products via ``create_graph=True``):
  both route through this module's compiled backend (jvp through
  ``elastic_born_compiled_core``'s forward only; the HVP through one
  forward *and* one full two-field backward via
  ``elastic_born_backward_compiled``), and both now thread the same
  per-face ``bc`` codes as the primal/first-order adjoint through to the
  Born calls, instead of refusing with ``NotImplementedError`` for any
  non-PML face (the guard was in
  ``deepwave.elastic._elastic_forward_jvp``/``ElasticBackwardFunc.backward``;
  see ``test_elastic_c_jvp_supports_non_pml``/
  ``test_elastic_c_hvp_supports_non_pml`` in ``tests/test_boundary.py``). ``elastic_born``'s own
  ``jvp``/second-order rules remain unconditionally unsupported regardless
  of ``boundary`` (see the propagator's own docstring) -- only
  :func:`deepwave.elastic`'s rules, which route through elastic_born's
  compiled kernels as an implementation detail, gain non-PML support here.

  .. note::

     **A short simulated time gives a vacuous forward-mode/second-order
     agreement check.** If the simulated time is too short for the wave to
     reach a reflecting face, both the eager and compiled outputs are
     statistically indistinguishable from the numerical noise floor (an
     absolute agreement at the level of ``~1e-35`` to ``~1e-44`` is a
     symptom of this, not evidence of correctness) -- the guard-fill/
     mirror-pad path is never actually exercised, so this kind of check can
     let a real compiled-backward adjoint bug at reflecting faces go
     undetected. The real correctness evidence for the compiled backward at
     reflecting faces is a well-conditioned, long-``nt`` compiled-vs-eager
     adjoint-parity check with the wave reflecting many times off every
     face type, using relative rather than absolute tolerances, and ideally
     a negative control that reproduces the failure signature when a known
     fix is reverted.

  .. note::

     **Isolating a single-scatter arrival needs enough source-receiver
     separation.** When measuring the scattered-channel reflection
     coefficient via an incident-vs-reflecting-face subtraction, a
     higher-order arrival (e.g. the background wave reflecting off the
     tested face *before* reaching the scatterer, then scattering directly
     to the receiver) can land inside the same measurement window as the
     intended single-scatter/single-bounce arrival and bias the measured
     ``R`` by a consistent fraction across every accuracy order. A bias
     whose size does not shrink with accuracy order is the signature of
     this kind of test-geometry defect (an uncontrolled second physical
     arrival in the window) rather than a real implementation bug; the fix
     is to widen the source-receiver separation so the two arrivals' path
     lengths differ enough to fall outside both the measurement window and,
     comfortably, the simulated time window entirely (see
     ``_elastic_born_reflection_coefficient``'s own comment).

  Validated (``tests/test_boundary.py``): ``"free"`` is not an accepted face
  on ``elastic_born`` (see :ref:`boundary-faces-accepted`), so the gate is
  ``test_elastic_born_rigid_reflection_coefficient``
  (eager) and its ``_c_`` compiled counterpart: analytic
  ``R = +1`` to <1% at accuracy 2-8, on *both* the background
  (``bg_receiver_amplitudes_p``) and the scattered (``receiver_amplitudes_p``)
  pressure-type channel. Unlike ``acoustic_born``'s single-point scatterer
  (its domain is genuinely 1-D), this reuses :func:`deepwave.elastic`'s own
  periodic-line-source 1-D-equivalent construction (a full line source/
  receiver on the tangential axis collapses the 2-D domain to an exact 1-D
  plane wave), so the scattering perturbation (``dlamb``) is a "sheet" held
  uniform along the periodic axis at one position of the tested axis rather
  than a single point, which would break the tangential translation
  invariance the construction relies on; the background field's reflection
  is measured exactly as in :func:`deepwave.elastic`'s own reflection test,
  and the scattered field's reflection is isolated the same
  incident-vs-reflecting-face subtraction technique :func:`deepwave.acoustic_born`'s own
  entry above and :func:`deepwave.scalar_born`'s
  ``_born_reflection_coefficient`` use. Periodic translation invariance (background and
  scattered together, source/receiver/scatterer shifted as one unit), eager
  and compiled. ``boundary=None`` bit-identity (eager, and eager-vs-compiled).
  Absolute-FD gradcheck plus compiled-vs-eager gradient agreement w.r.t.
  ``lamb``, ``mu``, ``buoyancy``, ``dlamb``, ``dmu``, ``dbuoyancy``, and the
  shared source amplitude together, across every face type including mixed
  PML/reflecting corners in 2-D
  (``test_elastic_born_c_gradcheck_and_matches_eager``), plus a dedicated 3-D case
  (``test_elastic_born_c_gradcheck_3d``). The free surface
  ``elastic_born`` offers is the vacuum-cell IVM construction, which needs no
  ``boundary`` argument and is gated on the background channel by the same
  Rayleigh-Lamb oracles as :func:`deepwave.elastic` itself -- elastic_born's
  background field IS elastic's own system with a zero scattering
  perturbation, so no new physics is involved. Both standalone
  ``nvcc -c`` compile-cleanliness and live execution of the compiled CUDA
  kernels on GPU hardware (comparing against the CPU/eager backend,
  including gradients) were verified for this propagator.
