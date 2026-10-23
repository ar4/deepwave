Least-Squares Reverse-Time Migration (LSRTM)
============================================

This example demonstrates inversion for the scattering potential with Born modelling, commonly known as imaging or migration.

The code is very similar to that used in :doc:`the FWI example <example_fwi>`.
We load the Marmousi 1 velocity model and smooth it to use as the migration
velocity model, load the data generated in :doc:`the forward modelling
example <example_forward_model>`, and then perform iterative optimisation,
repeatedly running forward modelling and backpropagation to calculate the
gradients and update the model. The main differences are that we now use
the Born propagator and invert for the scattering model rather than the
velocity; the rest of this page focuses on those changes.

Born modelling propagates a wavefield in a "background" velocity model.
That wavefield interacts with a provided scattering potential at each time
step to produce a source term for a second, scattered wavefield, which is
also propagated using the background velocity model and is what receivers
record. The recorded data therefore only contains singly-scattered waves,
whereas the observed dataset also contains strong direct arrivals; trying
to get Born modelling to match those direct arrivals would cause strong
artefacts. There are multiple ways to handle this. Here, we model the
direct arrivals by forward modelling through the smooth migration velocity
model with the regular (non-Born) propagator, then subtract them from the
observed dataset before using it in inversion::

    out = scalar(
        v_mig, dx, dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_freq=freq,
        accuracy=8,
        max_vel=v_true.max(),
    )
    observed_scattered_data = observed_data - out.receiver_amplitudes

Note that we specified the `max_vel` parameter to be the maximum velocity in the true velocity model. This will cause Deepwave's internal time step size to be the same as when `observed_data` was created, to get a better match between the two datasets (and so fuller cancellation of the direct arrivals). From the results we see that subtracting this estimate of the direct arrivals successfully attenuates them relative to the scattered arrivals.

.. image:: example_lsrtm_scattered.jpg

We now run the optimisation to invert for the scattering potential. To do this we create an initial scattering model (which is all zeros), specify that gradients will need to be calculated with respect to it, and then iteratively optimise it so that the output of Born propagation matches the observed scattered data::

    # Create scattering amplitude that we will invert for
    scatter = torch.zeros_like(v_mig)
    scatter.requires_grad_()

    # Setup optimiser to perform inversion
    optimiser = torch.optim.LBFGS([scatter])
    loss_fn = torch.nn.MSELoss()

    # Run optimisation/inversion
    n_epochs = 3

    for epoch in range(n_epochs):
        def closure():
            optimiser.zero_grad()
            out = scalar_born(
                v_mig, dx, dt,
                dv=scatter,
                source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                sc_receiver_locations=receiver_locations,
                pml_freq=freq,
            )
            loss = 1e6 * loss_fn(
                out.sc_receiver_amplitudes, observed_scattered_data
            )
            loss.backward()
            return loss.item()

        optimiser.step(closure)

This produces the following image (note again that the result is worse on the right because there are no sources there).

.. image:: example_lsrtm.jpg

`Full example code <https://github.com/ar4/deepwave/blob/master/docs/example_lsrtm.py>`_
