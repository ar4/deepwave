Acoustic propagator reference
=============================

The variable-density acoustic propagators -- ``acoustic``,
``acoustic_born``, ``viscoacoustic`` and ``acoustic_vti`` -- in both
their Module class and functional forms. :doc:`usage` covers the
conventions they share with every other propagator, and
:doc:`acoustic`, :doc:`acoustic_born`, :doc:`viscoacoustic` and
:doc:`acoustic_vti` cover the physics.

.. contents:: :local:

Acoustic Class
--------------

.. autoclass:: deepwave.Acoustic
    :members:

Acoustic Function
-----------------

.. autofunction:: deepwave.acoustic

AcousticBorn Class
------------------

Born (linearised) modelling for the variable-density acoustic wave equation.
It propagates a background field through ``v``/``rho`` and, alongside it, the
*scattered* field: the exact first-order linearisation of the response to the
model perturbations ``dv``/``drho``. This is the basis of :term:`LSRTM` and
the linear map differentiated in :term:`FWI`. The scattered receiver data is
linear in that ``(dv, drho)`` scattering. See :doc:`acoustic_born` for the physics
(and :doc:`scalar_born` for the background theory of Born modelling in
general) and :doc:`acoustic_born_derivation` for this propagator's discrete
derivation and its adjoint. The compiled backend -- what ``backend="auto"``
(the default) resolves to here -- supports
``forward_callback``/``backward_callback``/``callback_frequency`` and the
``storage_mode`` options; ``backend="eager"`` selects a pure-PyTorch
coupled Born driver that runs under native autograd and composes with
``torch.func`` transforms (see the ``backend`` argument below).

.. autoclass:: deepwave.AcousticBorn
    :members:

AcousticBorn Function
---------------------

.. autofunction:: deepwave.acoustic_born

Viscoacoustic Class
-------------------

.. autoclass:: deepwave.Viscoacoustic
    :members:

Viscoacoustic Function
----------------------

.. autofunction:: deepwave.viscoacoustic

Acoustic VTI
------------

.. autoclass:: deepwave.AcousticVti
    :members:

Acoustic VTI Function
---------------------

.. autofunction:: deepwave.acoustic_vti
