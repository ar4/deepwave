Design notes
============

These pages record *why* parts of Deepwave are built the way they are: the
constraint that forced a decision, the alternatives that were considered and
rejected, and the derivation behind a guarantee that the API states in one
line. They are developer reference, not user documentation -- nothing here is
needed to use Deepwave, and none of it describes API you call. If you are
looking for how to use a feature, see the :doc:`user guide </usage>` and the
examples instead.

The audience is anyone maintaining or extending Deepwave, including anyone
tempted to change one of these decisions: each page is written so that the
cost of the rejected alternative is visible before it is re-attempted.

.. toctree::
   :maxdepth: 1

   propagator_output
   checkpointing
   misfits
