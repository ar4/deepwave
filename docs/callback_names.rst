Internal callback names
=======================

Every lookup a callback makes is by *name*.
:meth:`deepwave.CallbackState.get_model`,
:meth:`deepwave.CallbackState.get_wavefield` and
:meth:`deepwave.CallbackState.get_gradient` each take a string, and the
strings they accept are Deepwave's **internal** ones: the kernel's own
spelling of the models it reads and of the wavefields it steps, not the
arguments you passed the propagator. ``deepwave.acoustic`` is *called* with
``v`` and ``rho``, for example, but *steps* the bulk modulus ``K`` and the
staggered buoyancies ``By``/``Bx``, and it is those three that its callbacks
see. They are unguessable, so this page lists them, per propagator and per
dimensionality.

The table is also the answer to "which gradient can I reach from a
``backward_callback``?". The model column is the gradient column: the
gradient accumulator for an internal model is keyed by that model's own
name, so ``state.get_gradient("K")`` is the live accumulator behind
``acoustic``'s ``K``, and mutating it in place changes the gradient the call
ultimately returns. That modified result is a user-defined backward value,
not necessarily the derivative of the unchanged forward loss. This is an
intermediate accumulator, even at the last
callback. Use a PyTorch tensor gradient hook or ``model.grad`` after
``.backward()`` for complete-gradient transformations, as in
:doc:`example_custom_imaging_condition`.
Only the compiled backend populates gradients at all; ask
:meth:`deepwave.CallbackState.writable` rather than assuming.

Internal diagnostics and stable physical fields
------------------------------------------------

Public physical wavefield names follow the stable interface policy. Internal
PML/attenuation keys and prepared-model names are diagnostic interfaces with
provisional names: a change to the numerical implementation may change them,
with release notes. Keeping them visible for inspection does not freeze an
absorbing-boundary representation or a particular material preprocessing step.
Pass ``out.auxiliary_state`` back whole when continuing a run.

Callback accessors expose live tensors. Differentiable forward callbacks may
observe models and wavefields but must not mutate them; use
:meth:`deepwave.CallbackState.writable` to check the current context. The
stable accessors remain available even if a provisional diagnostic name changes.

Do not confuse that with the *public* side of a call, which does not take
these spellings at all. ``v0.0.27`` returned the absorbing boundary's memory
variables as individual tuple elements and took them back in as individual
keyword arguments (``psiy_m1=``, ``zetax_m1=``, ``phi_y_0=``, ...). The
public interface now carries the whole set opaquely in one
``auxiliary_state`` field, so those keyword arguments no longer exist on any
propagator (passing one raises ``TypeError``), asking an output for one by
name raises ``AttributeError``, and naming one as an ``initial_state`` key
raises :exc:`deepwave.DeepwaveValueError` pointing at ``initial_state=out``
(see :ref:`upgrade-auxiliary-state`). Neither the attribute nor the
``initial_state`` spelling ever shipped, which is why they were removed in
Deepwave 0.1.0 rather than deprecated. The identical name passed to
``state.get_wavefield(...)`` inside a callback, or read from an
``auxiliary_state``, is accepted on every supported propagator.

One name on the callback side *is* deprecated, and it is not in the table:
``scatter``, ``v0.0.27``'s spelling of ``scalar_born``'s perturbation model,
which :meth:`~deepwave.CallbackState.get_model` and
:meth:`~deepwave.CallbackState.get_gradient` still resolve to ``dv`` with a
:class:`deepwave.DeepwaveFutureWarning` (see :ref:`upgrade-scatter-dv`). It is
removed in Deepwave 0.4.0. The table lists the current spelling only.

How to read the table
---------------------

Models (``get_model``, ``get_gradient``)
    The models the kernel actually reads, in the order it holds them. These
    are the gradient accumulator names as well. Several are *staggered*
    per-axis variants of one user argument (``elastic`` is given one ``mu``
    and holds ``mu``, ``mu_zy``, ``mu_zx``, ``mu_yx``), and several are
    derived quantities with no user argument of their own (``acoustic``'s
    ``K``, ``viscoacoustic``'s GSLS coefficients ``m1``/``c2``/``ar``). The
    Born propagators hold the background models followed by the
    perturbations, spelled with a leading ``d`` (``dv``, ``dK``, ``dlamb``,
    ``dmu_yx``, ``dbuoyancy_y``).

Wavefields (``get_wavefield``)
    Every field the propagator steps, including the auxiliary ones. A
    trailing ``_0`` or ``_m1`` is the time level (the current step, and the
    previous step for the second-order scalar family), not an index into
    anything. The Born propagators' scattered fields carry ``_sc`` before
    that time level: ``wavefield_sc_0``, ``vy_sc_0``, ``m_sigmaxyy_sc_0``.

Of which absorbing-boundary/attenuation auxiliary
    The subset of the previous column that the public interface folds into
    the single opaque ``auxiliary_state`` field: the PML memory variables
    (``psiy_m1``/``zetax_m1`` for the scalar family, ``phi_y_0``/``psi_x_0``
    for the acoustic family, ``m_vyy_0``/``m_sigmaxyy_0`` for the elastic
    family) and ``viscoacoustic``'s constant-Q relaxation memory
    (``r_0``; ``rp_0``, ``rs_y_0``, ... ). Folded on the public side, plain
    named wavefields to a callback.

``q_n_mechanisms``
    ``viscoacoustic`` carries one set of constant-Q memory fields per
    relaxation mechanism, so its row is true for a particular
    ``q_n_mechanisms``, which their Dimensionality cell names. It is the
    argument's default, 1. With ``q_n_mechanisms=N`` the memory names run
    from ``_0`` to ``_{N-1}`` instead (``viscoacoustic`` gains ``r_1``);
    every other name in the row is unaffected.

A name the accessors do not recognise raises a ``KeyError`` that lists the
names that *are* present, so a typo in a callback tells you the right
spelling rather than leaving you to guess it.

**How this page stays accurate.** Not one name below is typed into this
page. The table is generated from each propagator's own
``PropagatorSpec`` -- its ``model_names``, ``wavefield_names`` and
``auxiliary_state_names`` -- which are the exact callables
``deepwave._engine`` and the eager backends zip the live tensors against
when they build the :class:`deepwave.CallbackState` a callback receives. It
is regenerated from the installed ``deepwave`` package every time this
documentation is built (a ``conf.py`` hook on Sphinx's ``builder-inited``
event writes the included fragment below, exactly as for the
:doc:`capability matrix <capabilities>`), and
``tests/test_callback_names.py`` re-derives it independently by running a
real callback on every propagator and asking it which names it was actually
handed. A rename in a kernel therefore shows up here automatically instead
of leaving a stale list.

.. include:: _generated/callback_names_table.rst
