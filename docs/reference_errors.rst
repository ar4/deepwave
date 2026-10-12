Errors and warnings reference
=============================

The exception and warning classes Deepwave raises. :doc:`usage`
covers the conventions of the interface that raises them.

Exceptions and warnings
-----------------------

Deepwave raises its own exception types. Each one also inherits the builtin
exception type that the same failure raised historically, so an existing
``except ValueError`` (or ``TypeError``, ``RuntimeError``,
``NotImplementedError``, ``OSError``, ``FloatingPointError``) keeps catching
exactly what it caught before -- while ``except deepwave.DeepwaveError``
catches everything Deepwave raises deliberately, and nothing else. See
:doc:`stability` for what may change about them.

What "deliberately" covers
--------------------------

Every rejection Deepwave decides on -- an argument it validates, a
combination it cannot serve, a build or backend that is unavailable, a
diverged wavefield, a scratch directory the filesystem refused -- raises one
of the classes below, so::

    try:
        out = deepwave.scalar(v, dx, dt, source_amplitudes=a,
                              source_locations=x_s)
    except deepwave.DeepwaveError as error:
        ...

catches all of them. Two kinds of failure are deliberately outside it, and
neither can be brought inside:

* A ``TypeError`` raised by Python's own argument binding, before any
  Deepwave code runs: a missing required argument, a keyword spelled wrongly
  enough that no propagator accepts it, or an argument given twice. Python
  raises it while matching the call to the signature, so Deepwave has no
  opportunity to classify it. (A keyword a propagator *recognises* but
  cannot act on is a different thing: Deepwave checks that itself and raises
  :exc:`deepwave.DeepwaveValueError`.)
* Any exception from PyTorch or from the standard library -- a shape
  mismatch inside an operation, an out-of-memory error, a CUDA failure. It
  stays exactly the exception those libraries raised; Deepwave does not wrap
  it, so a traceback still points at the operation that failed.

Deepwave also raises a few builtin exceptions where Python's own protocols
require them, rather than as a message to the caller: ``AttributeError``
from a ``__getattr__`` for an unknown name, ``KeyError`` from a mapping
lookup for an absent key, and ``AssertionError`` for an internal invariant
that no call can violate (reaching one is a bug in Deepwave; please report
it). These are protocol obligations and bug reports, not rejected requests,
so they are not :exc:`deepwave.DeepwaveError` subclasses either. Everything
else Deepwave raises on purpose is: ``tests/test_error_classes.py`` walks the
source of every Deepwave module and fails if a ``raise`` of any other builtin
exception type appears.

The classes
-----------

.. Every one of these carries an explicit ``(*args)`` signature, which is
   the ordinary exception one. Without it autodoc tries runtime
   introspection, and that is unreliable here for two compounding reasons:
   these classes have no Python-level ``__init__`` of their own (the builtin
   they mix in supplies a C-level one that
   ``sphinx.util.inspect.signature`` cannot read), and each sets
   ``__module__ = "deepwave"`` so that it documents under its public name.
   The second makes ``inspect.getsource`` look for the class in
   ``deepwave/__init__.py`` at the line number it occupies in
   ``deepwave/_exceptions.py``, so what autodoc parses depends on how long
   ``__init__.py`` happens to be -- adding a single line to ``__all__`` can
   turn a clean build into an ``IndexError``. Stating the signature stops
   autodoc introspecting at all, which is what makes the build reproducible.
.. autoexception:: deepwave.DeepwaveError(*args)

.. autoexception:: deepwave.DeepwaveValueError(*args)

.. autoexception:: deepwave.DeepwaveTypeError(*args)

.. autoexception:: deepwave.DeepwaveRuntimeError(*args)

.. autoexception:: deepwave.DeepwaveNotImplementedError(*args)

.. autoexception:: deepwave.DeepwaveOSError(*args)

.. autoexception:: deepwave.DeepwaveFloatingPointError(*args)

.. autoexception:: deepwave.DeepwaveFutureWarning(*args)

.. autoexception:: deepwave.DeepwaveWarning(*args)

.. autoexception:: deepwave.ZeroWidthFaceWarning(*args)

.. autoexception:: deepwave.MigrationWarning(*args)

.. autoexception:: deepwave.PmlFreqInferenceWarning(*args)
