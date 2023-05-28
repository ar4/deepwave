What Deepwave calculates
========================

Deepwave implements forward modelling and backpropagation of wave equations, using the finite difference method, in PyTorch.

Source terms are scalar functions of time. They are located at grid cell centres in the scalar propagators, and at the positions of :math:`v_y` and :math:`v_x` components in the elastic propagator. Achieving a different desired effective source, such as one with an arbitrary direction, or one at a fractional grid cell location, may require splitting the source into an array of multiple sources over neighbouring cells (the :class:`Hicks <deepwave.location_interpolation.Hicks>` method may help).

Receiver data is produced by recording the relevant wavefield at the specified receiver locations. The receiver data covers the same time range as the source data, and has the same time sampling. As with sources, approximating directed receivers or ones at fractional grid cell locations, can be achieved by recording from multiple neighbouring grid cells and then combining the resulting data. The :class:`Hicks <deepwave.location_interpolation.Hicks>` method may again help.

In order to obey the `CFL condition <https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition>`_, Deepwave may internally use a smaller time step interval than that used for the input source and output receiver data. The interval that is used will be a divisor of the source/receiver time sampling interval (and is calculated using :func:`cfl_condition <deepwave.common.cfl_condition>`). The source amplitudes will internally be upsampled to this new sampling rate, using zero-padding in the Fourier domain, and the receiver data will be downsampled, using truncation in the Fourier domain, before being returned to the user (using :func:`upsample <deepwave.common.upsample>` and :func:`downsample <deepwave.common.downsample>`). As it can be shown that the time sampling in the integrals over time to produce the gradients with respect to the input models does not need to be more frequent than the time sampling of the input source, Deepwave only adds to these gradients at the input/output time sampling rate. This avoid unnecessary calculations and memory consumption without affecting results.

The propagators return the final wavefields, and the recorded receiver data. The wavefields are the full wavefields used by Deepwave, including padding for the `Perfectly Matched Layer (PML) <https://en.wikipedia.org/wiki/Perfectly_matched_layer>`_. This means that they can be passed in as the initial wavefields to another instance of the propagator to continue propagation.

More detailed information specific to each propagator is presented on the following pages.

.. toctree::
    scalar
    scalar_born
    elastic
