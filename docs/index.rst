Deepwave
========

Deepwave provides PyTorch-integrated wave propagators for forward modelling
and backpropagation, for gradient calculation in inversion and optimisation
tasks. It supports standard and :term:`Born` modelling of the scalar,
variable-density acoustic, and elastic wave equations in 1D, 2D, and 3D,
with viscoacoustic attenuation in 1D–3D and VTI anisotropy in 2D and 3D.
The compiled backends support CPUs and CUDA GPUs. See the :doc:`capability matrix
<capabilities>` for exactly which options each propagator supports. New to
seismic/wave-equation terminology? The :doc:`glossary` defines the terms
(PML, CFL, Born, FWI, and more) used throughout these docs.

Installation
------------

Deepwave requires Python 3.9 or later and PyTorch 2.4 or later. Install
PyTorch for your CPU or CUDA device by following the instructions on the
`PyTorch website <https://pytorch.org>`_, then install Deepwave:

.. code-block:: bash

    python -m pip install deepwave

This method should work for most users. However, if you have an unusual system configuration, you might need to build Deepwave from source. Instructions for building from source are provided below.

macOS wheels support Apple Silicon. Intel Mac wheels are no longer built,
because PyTorch 2.4 and newer do not provide Intel macOS wheels.

PyTorch is Deepwave's only dependency. Nothing else is needed to use it. The
example scripts under ``docs/`` do import a few other packages --
``matplotlib`` to draw their figures, and, in some of them, ``scipy``
(smoothing and filter design) and ``torchaudio`` (applying those filters) --
but those are needed only to run the examples, never by Deepwave itself.
See :ref:`running-the-examples`.

Deepwave is built on PyTorch, so understanding PyTorch well makes it easier to use. The `PyTorch website <https://pytorch.org>`_ offers many helpful tutorials, and the :doc:`pytorch` page here covers the specific parts that matter most for Deepwave.

Building from source
--------------------

Deepwave uses `cibuildwheel` and `scikit-build-core` for its build process. When you run `pip install deepwave`, pip will first attempt to download a precompiled wheel if one is available for your system. If a suitable precompiled wheel is not found, pip will automatically download the source distribution and attempt to compile it on your system. You will need to have a C compiler and CMake available, and a CUDA compiler if you wish to have CUDA support.

If you wish to force building from source rather than using a wheel, you can use the following command:

.. code-block:: bash

    python -m pip install --force-reinstall --no-deps --no-binary deepwave deepwave

This replaces an existing installation with a source build, leaving your
chosen PyTorch installation in place. It is the recommended fix if a
precompiled wheel does not run on your CPU (see Installation troubleshooting
below).

Installation troubleshooting
----------------------------

Prebuilt x86 wheels require AVX2. On a CPU without it, current wheels report
the missing instruction set; older wheels may crash with
``Illegal instruction (core dumped)``. A source build uses a portable CPU
baseline by default. See
:doc:`troubleshooting` for this and other installation and runtime problems.

Citing
------

Deepwave is archived on Zenodo with the DOI `10.5281/zenodo.3829886 <https://doi.org/10.5281/zenodo.3829886>`_ (this DOI always resolves to the latest release). If you would like to cite Deepwave, I suggest:

.. parsed-literal::

    @software{richardson_alan_2026,
      author       = {Richardson, Alan},
      title        = {Deepwave},
      year         = 2026,
      publisher    = {Zenodo},
      version      = {|version|},
      doi          = {10.5281/zenodo.3829886},
      url          = {https://doi.org/10.5281/zenodo.3829886}
    }

Support
-------

If you encounter any issues (e.g., typos, broken links, unexpected behaviour, examples that don't work), or have suggestions for improvement, please `file an issue on GitHub <https://github.com/ar4/deepwave/issues>`_ or `send an email <mailto:alan@ausargeo.com>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   quickstart
   glossary
   pytorch
   examples
   upgrading

.. toctree::
   :maxdepth: 2
   :caption: User guide

   stability
   usage
   wavefields
   backends
   capabilities
   callback_names
   accuracy
   boundaries
   units
   materials
   time_dispersion
   troubleshooting
   scaling

.. toctree::
   :maxdepth: 1
   :caption: API reference

   reference_scalar
   reference_acoustic
   reference_elastic
   reference_utilities
   reference_errors

.. toctree::
   :maxdepth: 1
   :caption: Propagator reference

   scalar
   scalar_born
   acoustic
   acoustic_born
   viscoacoustic
   elastic
   elastic_born
   elastic_vti
   acoustic_vti

.. toctree::
   :maxdepth: 2
   :caption: Theory and guarantees

   what_deepwave_calculates
   discretisation
   operator_contract

.. toctree::
   :maxdepth: 1
   :caption: Derivations (developer reference)

   acoustic_born_derivation
   acoustic_vti_derivation
   elastic_born_derivation

.. toctree::
   :maxdepth: 1
   :caption: Design notes (developer reference)

   design/index

.. toctree::
   :maxdepth: 2
   :caption: Applications and alternatives

   ultrasound
   das
   comparison

.. toctree::
   :maxdepth: 1
   :caption: Release reference (maintainer)

   dev/release_validation
   dev/releasing
