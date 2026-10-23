Time-dispersion correction
==========================

.. py:module:: deepwave.time_dispersion

This page explains temporal dispersion -- the error a leapfrog time step
introduces independently of the spatial grid -- and how to remove it after the
fact approximately with the optional :mod:`deepwave.time_dispersion` module.
It is for anyone pushing ``dt`` close to the stability limit to save compute,
and wanting the accuracy of a smaller ``dt`` without paying its cost.

Every Deepwave propagator advances the wavefield with an explicit,
second-order-accurate leapfrog step (see :doc:`discretisation`). That scheme is
only *second* order in time, so it introduces **temporal dispersion**: a
frequency-dependent error in the wave speed that, unlike spatial dispersion, is
**not** reduced by raising the ``accuracy`` order. It is controlled solely by
the time step. Pushing ``dt`` toward the :term:`CFL` stability limit to save
compute therefore trades speed for temporal-dispersion error — unless you
correct for it.

The :mod:`deepwave.time_dispersion` module implements the correction of Koene,
Robertsson, Broggini & Andersson ("Eliminating time dispersion from seismic
wave modeling", *Geophysical Journal International* **213**, 2018). It is
opt-in, pure PyTorch, differentiable, and model-independent.

The current transforms evaluate finite-record spectra with dense frequency by
time matrices. Their temporary memory grows quadratically with record length:
a 10,000-sample record uses roughly 50 million complex entries per matrix.
Use them for moderate records, and budget memory accordingly.

The dispersion relation
-----------------------

Substituting a single Fourier mode :math:`u_n = e^{i\omega n\Delta t}` into the
leapfrog second difference shows that a mode oscillating in the simulation at
temporal angular frequency :math:`\omega` carries the physics of the *lower*
frequency

.. math::

    \Omega(\omega) = \frac{2}{\Delta t}\,\sin\!\Bigl(\frac{\omega\,\Delta t}{2}\Bigr)
    \;\le\; \omega .

Equivalently, physics that should oscillate at :math:`\Omega` is *shifted up* in
the simulation to

.. math::

    \omega(\Omega) = \frac{2}{\Delta t}\,\arcsin\!\Bigl(\frac{\Omega\,\Delta t}{2}\Bigr)
    \;\ge\; \Omega ,
    \qquad \Omega < \frac{2}{\Delta t}.

These two maps are exact inverses on the simulation band
:math:`|\omega| < \pi/\Delta t`, which maps to the physical band
:math:`|\Omega| < 2/\Delta t`.

The scalar identity for a *single* oscillator is that the leapfrog recursion,

.. math::

    u_{n+1} - 2u_n + u_{n-1} = -\Delta t^2\,\Omega_0^2\,u_n ,

is solved exactly by :math:`u_n = \sin(\omega_{\mathrm{num}}\,n\,\Delta t)` with
:math:`\omega_{\mathrm{num}} = (2/\Delta t)\,\arcsin(\Omega_0\,\Delta t/2)`.
This identity describes the scalar oscillator only. It does not establish exact
correction of finite signals or of other propagator families.

The two transforms
------------------

The correction is a pair of frequency-domain remappings applied *outside* the
propagator:

* :func:`~deepwave.time_dispersion.correct_source` pre-distorts the source
  wavelet, applying the inverse map so that content the caller places at
  physical frequency :math:`\Omega` is injected at the simulation frequency
  :math:`\omega(\Omega)`. After the simulation shifts it back down, it lands
  where intended.
* :func:`~deepwave.time_dispersion.correct_receivers` post-processes the
  recorded data, moving content from the simulation frequency :math:`\omega`
  back down to the physical frequency :math:`\Omega(\omega)`.

The maps describe the scalar leapfrog frequency relation. Their use on finite
records is approximate: only the retained frequency band is remapped, and
content outside that band is lost. Finite-record spectral leakage, padding and
windowing also affect the result. Exact evaluation of the finite-record DTFT
does not make a remap round trip exact. This scalar relation does not establish
exact correction of attenuating, staggered-grid or PML dynamics, or remove
spatial discretisation error.

Which time step to pass
-----------------------

The dispersion is governed by the leapfrog step the propagator *actually* takes
— its **internal** time step ``internal_dt = dt / step_ratio`` — not by the
user-facing ``dt``. Deepwave subdivides ``dt`` into ``step_ratio`` internal
steps to satisfy the CFL condition. Both transforms therefore take
``internal_dt`` explicitly, and treat the passed tensors' time axis as sampled
at that step.

The simplest recipe is to pick a ``dt`` that is already CFL-stable, so
the propagator runs with ``step_ratio == 1`` and ``internal_dt == dt``; then
correct the source and the receivers at ``dt``.
:func:`~deepwave.time_dispersion.cfl_internal_dt` returns the exact
``internal_dt`` the propagator would use for a given grid spacing, ``dt``,
maximum velocity and ``courant_number``. Pass the grid spacing as the sequence
of per-dimension spacings you give the propagator (``[dx, dx]``), or as a
single value together with ``ndim`` (``cfl_internal_dt(dx, dt, max_vel,
ndim=2)``), since the CFL limit depends on the number of dimensions. When
``step_ratio > 1``, resample your source/receiver tensors to ``internal_dt``
(with :func:`deepwave.discretisation.upsample` / :func:`deepwave.discretisation.downsample`)
before correcting, or force ``step_ratio == 1``.

See :doc:`example_time_dispersion` for a runnable end-to-end example.

API
---

correct_source Function
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: deepwave.time_dispersion.correct_source

correct_receivers Function
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: deepwave.time_dispersion.correct_receivers

cfl_internal_dt Function
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: deepwave.time_dispersion.cfl_internal_dt
