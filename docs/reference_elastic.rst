Elastic propagator reference
============================

The elastic propagators -- ``elastic``, ``elastic_born`` and
``elastic_vti`` -- in both their
Module class and functional forms. :doc:`usage` covers the conventions
they share with every other propagator, and :doc:`elastic`,
:doc:`elastic_born` and :doc:`elastic_vti` cover the physics.

.. contents:: :local:

Elastic Class
-------------

.. autoclass:: deepwave.Elastic
    :members:

Elastic Function
----------------

.. autofunction:: deepwave.elastic

ElasticBorn Class
-----------------

Born (linearised) modelling for the isotropic elastic wave equation. The
background field solves the ordinary elastic system for
``lamb``/``mu``/``buoyancy``, while the scattered field is the exact
first-order linearisation for the perturbations ``dlamb``/``dmu``/``dbuoyancy``.
This is the elastic analogue of LSRTM and the linear map differentiated in
elastic FWI. The scattered receiver data is linear in that
``(dlamb, dmu, dbuoyancy)`` scattering. See :doc:`elastic_born` for the physics (and :doc:`scalar_born`
for the background theory of Born modelling in general) and
:doc:`elastic_born_derivation` for this propagator's discrete derivation and
its adjoint.

.. autoclass:: deepwave.ElasticBorn
    :members:

ElasticBorn Function
--------------------

.. autofunction:: deepwave.elastic_born

Elastic VTI
-----------

.. autoclass:: deepwave.ElasticVti
    :members:

Elastic VTI Function
--------------------

.. autofunction:: deepwave.elastic_vti
