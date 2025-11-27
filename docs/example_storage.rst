3D FWI with storage options to reduce memory usage
==================================================

Deepwave supports 3D wave propagation, which naturally requires significantly more memory than 2D propagation. For large 3D models, or even large 2D models, the memory required to store the intermediate wavefields needed for gradient calculation (backpropagation) can exceed the available memory on the compute device (e.g., GPU).

To address this, Deepwave allows you to offload the storage of these intermediate wavefields to CPU memory or to disk, and optionally compress them. This can drastically reduce the peak memory usage on the compute device, enabling the processing of much larger models.

The storage behavior is controlled by three arguments to the propagator:
    - ``storage_mode``: Where to store the intermediate data. Options are ``"device"`` (default), ``"cpu"``, ``"disk"``, or ``"none"``.
    - ``storage_path``: The directory to use for disk storage (default is current directory).
    - ``storage_compression``: Whether to apply a lossy compression to the data (default ``False``).

Note that if none of the input property models (such as ``v``) require gradients (``requires_grad=True``), Deepwave will not store any intermediate data regardless of the ``storage_mode`` setting. Thus, for pure forward modelling with a fixed model, the storage settings will not affect memory usage.

In this example, we demonstrate 3D FWI on a small model. We use ``storage_mode="cpu"`` and ``storage_compression=True`` to minimize the memory footprint on the device. While this small example would fit in GPU memory without these options, the same approach applies to large-scale 3D problems. Combining these storage options with the batched shots approach described in :doc:`the previous example <example_rtm>` is a good idea, but it is not used in this example for simplicity.

We define a simple 3D velocity model with a gradient background and a block anomaly::

    # Define the 3D model
    nx, ny, nz = 30, 30, 10
    # ... (setup code) ...
    v_background = torch.full((nz, ny, nx), 1500.0, device=device)
    v_true = v_background.clone()
    # Add anomaly
    v_true[4:8, 13:18, 13:18] = 1600

The interface for 3D propagation is consistent with 2D, but input tensors and coordinates now have an additional dimension. `pml_width` can be specified as a list of 6 integers (start/end for each of the 3 dimensions)::

    # Inversion loop
    optimizer = torch.optim.SGD([v], lr=1e8)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1):
        optimizer.zero_grad()
        
        # We use storage_mode='cpu' and storage_compression=True
        out = scalar(
            v,
            [dz, dy, dx],
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_width=[0, 20, 20, 20, 20, 20],
            storage_mode="cpu",
            storage_compression=True,
        )
        
        loss = loss_fn(out[-1], observed_data)
        loss.backward()
        optimizer.step()

Using ``storage_mode="disk"`` works similarly, but writes data to temporary files in ``storage_path``. This is the most memory-efficient option but may be slower due to disk I/O.

Dividing the data into :doc:`batches containing several shots <example_rtm>`, combined with compression and possibly transferring to CPU or disk for storage, is usually a good approach.

`Full example code <https://github.com/ar4/deepwave/blob/master/docs/example_storage.py>`_
