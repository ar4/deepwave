# Historical research

These unsupported research snapshots are excluded from wheels, source
distributions and ordinary pytest collection. Old imports and APIs may require
adaptation. Preserved tests and measurements describe historical experiments,
not qualification of the supported release.

- [Deferred propagators](deferred_propagators/README.md): elastic TTI needs
  independent heterogeneous interface and vacuum-gradient physics; viscoelastic
  needs an explicit constitutive and Q contract. The linked notes preserve the
  specific reasons for deferral.
- [Domain decomposition](domain_decomposition/README.md): scalar slab sources,
  tests and lessons, including a PML-corner instability. Sequential slabs do not
  reduce memory or provide multiple-device execution.
- [Compiled layer prototype](compiled_layer_spike/REPORT.md): generator, runtime,
  fault injection and measured results. This is an experimental architecture,
  not the package backend or a promise of present API compatibility.
- [Torch compile reproducer](torch_compile_dynamic_int/repro.py): a standalone
  historical reproducer for dynamic integer behaviour; compiler versions matter.

Physics and prototype limitations have not been repaired as part of release
preparation. Reintroduction requires independent validation.
