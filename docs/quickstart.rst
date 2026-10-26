Quick Start
===========

The example below is a complete, self-contained Deepwave program. It builds a
two-layer velocity model, propagates a single shot through it with the scalar
wave equation, and backpropagates to obtain the gradient of a simple objective
function with respect to the velocity model. It demonstrates the two things
that make Deepwave useful: forward wave propagation, and the fact that the
output is differentiable with respect to the inputs (here the model), so it
plugs directly into PyTorch's automatic differentiation and optimisers. If you
are new to seismic/wave-equation terminology, keep the :doc:`glossary` open
alongside this page -- it defines terms (such as PML, CFL, and FWI) the first
time each is used below.

Install PyTorch for your CPU or CUDA device first (see :doc:`index`), then
install Deepwave and the plotting dependency used below::

    python -m pip install deepwave matplotlib

.. code-block:: python

    import torch
    import deepwave
    import matplotlib.pyplot as plt

    v = 1500 * torch.ones(100, 100)
    v[50:] = 2000
    v.requires_grad_()

    out = deepwave.scalar(
        v, grid_spacing=4, dt=0.004,
        source_amplitudes=deepwave.wavelets.ricker(25, 200, 0.004, 0.06, shape=(1, 1)),
        source_locations=torch.tensor([[[0, 0]]]),
        receiver_locations=torch.tensor([[[0, 99]]]),
        pml_freq=25,
        max_vel=2000,
    )

    (out.receiver_amplitudes**2).sum().backward()

    _, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(v.detach())
    ax[0].set_title("Velocity model")
    ax[1].plot(out.receiver_amplitudes.detach().flatten())
    ax[1].set_title("Receiver data")
    ax[2].imshow(v.grad.detach(), vmin=-1e-5, vmax=1e-5)
    ax[2].set_title("Gradient")
    plt.show()

A few things to note:

* ``plt.show()`` only opens a window if a display is available. Running this
  on a headless machine (e.g. over SSH, or in a plain terminal with no
  X server), it does nothing and the script exits without complaint; replace
  it with ``plt.savefig("quickstart.png")`` to see the result in that case.
* Deepwave does not attach units to anything: it only requires that yours are
  consistent. These docs use SI throughout, which is what the numbers above
  are: ``v`` is a wavespeed in metres per second, ``grid_spacing`` is a cell
  size in metres, ``dt`` is a time step in seconds, and ``pml_freq`` (below)
  is in hertz. The propagators that take a density take it in kilograms per
  cubic metre, and the elastic ones take :math:`\lambda` and :math:`\mu` in
  pascals and buoyancy (the reciprocal of density) in cubic metres per
  kilogram. Source and receiver amplitudes are a separate question -- what
  physical quantity each propagator injects and records is set out in
  :doc:`units`.
* ``source_amplitudes`` has shape ``[shot, source, time]`` and
  ``source_locations`` / ``receiver_locations`` have shape
  ``[shot, source_or_receiver, dimension]``. Here there is one shot, one
  source, and one receiver.
* The locations are integer grid cell indices (``torch.long``). You may also
  provide floating-point coordinates to place sources and receivers off-grid;
  see :doc:`usage`.
* Deepwave orders spatial dimensions from slowest-varying to fastest-varying,
  and attaches no physical meaning to any of them: dimension 0 is depth only
  if you decide it is. These docs and examples put depth in dimension 0
  throughout, which is why the location columns above read ``[depth, x]``.
  See :ref:`axis-convention` for the full rule, including what the ``_y``/
  ``_z``/``_x`` component suffixes mean and what changes when a 2-D script is
  promoted to 3-D.
* ``out`` is a :class:`~deepwave.PropagatorOutput`, so ``out.receiver_amplitudes``
  is the recorded receiver data, of shape ``[shot, receiver, time]``. ``out``
  is still a plain tuple at its core, so ``out[-1]`` and unpacking (e.g.
  ``*wavefields, receivers = out``) keep working the same way. The earlier
  elements are the final wavefields, plus one ``auxiliary_state`` entry (an
  opaque :class:`~deepwave.AuxiliaryState`, not a tensor) carrying the
  absorbing boundary's memory variables -- for ``scalar`` this is
  ``wavefield_0, wavefield_m1, auxiliary_state``. Pass the whole output back in
  as ``initial_state=out`` to continue the propagation or to use it as an
  initial condition; see :doc:`usage`.
* Because ``v.requires_grad_()`` was called, the receiver data is
  differentiable with respect to ``v``, so ``backward()`` populates
  ``v.grad``. This is the foundation of :term:`full-waveform inversion <FWI>`
  and related methods.
  The objective here is receiver energy, not a data misfit: an inversion
  compares predictions with observations. ``max_vel=2000`` fixes the speed
  bound used for the time step and PML. At this ``dt``, CFL substepping
  makes the default model gradient a sampled approximation. Exact discrete
  model derivatives require additional conditions, including sampling every
  internal step and lossless storage; see :ref:`gradient-guarantees` before
  comparing gradients with finite differences.
* ``pml_freq`` is a frequency, in hertz (assuming ``dt`` is in seconds) --
  normally the dominant frequency of your source wavelet, 25 Hz here to match
  the Ricker wavelet above. Choose it poorly and the absorbing boundary works
  less well, reflecting energy back into the model instead of absorbing it.
  It is optional: leave it out and Deepwave infers it from the peak of your
  source amplitudes' spectrum (25 Hz for the wavelet above), reporting the
  value it used as ``out.info.pml_freq``. Give it explicitly, as here, for
  anything you publish -- an inferred value ties the absorbing boundary to
  the wavelet, so changing the wavelet changes the boundary
  (see :ref:`pml-freq`).
* Unsure what ``grid_spacing`` and ``dt`` to use? Ask::

      deepwave.suggest_discretisation(min_vel=1500, max_freq=25, ndim=2, max_vel=2000)

  For this model that suggests ``grid_spacing=10.0`` and ``dt=0.00212`` at
  ``accuracy=4`` (the example above uses a finer 4 m grid, which is simply
  more accurate and more expensive), along with a ``note`` saying which
  constraint was binding. See :doc:`accuracy`.
* The source and receiver above (at grid cell 0 and cell 99) are placed right
  at the edge of the model. The default 20-cell :term:`PML` is *padding
  outside* the model, so cell 0 is not inside it -- it is at its inner edge,
  the first cell of the model proper. That is fine for a first run, but it is
  worth knowing: the absorbing layer assigns a disproportionate share of the
  gradient to the model cells next to it (see
  :doc:`example_large_edge_gradient`). See :doc:`boundaries` for what the PML
  does and how to keep your survey away from it (or make an edge reflective
  instead) if that matters for your use case.
* This example never moves ``v`` or the other tensors to a GPU, so it runs on
  the CPU even if one is available; move them there yourself with
  ``.to(device)`` (see :doc:`example_fwi` for the idiom) once you are ready
  to use one.

Elastic in ~10 lines
--------------------

The elastic propagator has three model parameters (:math:`\lambda`,
:math:`\mu`, and buoyancy, the reciprocal of density) instead of one, and
separates its wavefield into velocity/stress components, but a minimal run
looks much the same. This example converts from the more familiar
:math:`v_p`/:math:`v_s`/:math:`\rho`
parameterisation using :func:`deepwave.vpvsrho_to_lambmubuoyancy`, runs
one shot, and looks at the vertical particle velocity received at one point:

.. code-block:: python

    import torch
    import deepwave

    vp = 1500 * torch.ones(50, 50)
    vs = 1000 * torch.ones(50, 50)
    rho = 2200 * torch.ones(50, 50)
    lamb, mu, buoyancy = deepwave.vpvsrho_to_lambmubuoyancy(vp, vs, rho)

    out = deepwave.elastic(
        lamb, mu, buoyancy, grid_spacing=4, dt=0.004,
        source_amplitudes_y=deepwave.wavelets.ricker(25, 200, 0.004, 0.06, shape=(1, 1)),
        source_locations_y=torch.tensor([[[0, 25]]]),
        receiver_locations_y=torch.tensor([[[0, 49]]]),
        pml_freq=25,
    )

    vy_receiver_data = out.receiver_amplitudes_y

Elastic has more outputs than scalar (one wavefield/receiver pair per velocity
component, plus pressure), so ``out.receiver_amplitudes_y`` is used here to
select the vertical-component receiver data by name rather than track its
position in a plain tuple. For 2-D propagation the ``_y`` suffix denotes the in-plane
vertical component (``_x`` is the in-plane horizontal component; ``_z`` is
only used in 3-D); see :doc:`example_elastic` for a full walkthrough and
:doc:`usage` for the complete set of components and the
:math:`v_p`/:math:`v_s`/:math:`\rho` :math:`\leftrightarrow`
:math:`\lambda`/:math:`\mu`/buoyancy conversion functions.

Once you have run this, :doc:`example_fwi_generated` is the next step: a
complete full-waveform inversion in one file, with nothing to download.
:doc:`example_rtm` does the same for :term:`reverse-time migration <RTM>`,
and the :doc:`examples` show progressively more realistic
workflows (forward modelling on the Marmousi model, full-waveform inversion,
migration, and more), and :doc:`usage` documents
the propagators in detail. If you are new to PyTorch, the :doc:`pytorch` page
covers the concepts that Deepwave relies on. For choosing the grid spacing and
time step, see :doc:`accuracy`; for the treatment of model edges, see
:doc:`boundaries`; for source/receiver amplitude conventions, see
:doc:`units`; and if something goes wrong, see :doc:`troubleshooting`.
