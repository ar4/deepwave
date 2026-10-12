# Deferred propagator research

These files preserve unreleased elastic TTI and viscoelastic work. They are not part of the Deepwave package, distribution, supported API, or release test suite. They are historical research snapshots; imports and shared interfaces may require adaptation before use.

Elastic TTI needs an established heterogeneous discretization with independently checked interface physics and differentiable vacuum handling. The previous cross-stiffness limiter produced NaN model gradients at a supported vacuum free surface. Viscoelastic needs a deliberate physical constitutive contract: the previous two-dimensional bulk/deviatoric split differed from three-dimensional plane strain, and `qp` denoted bulk-modulus Q rather than longitudinal-wave Q. Reintroduction requires independent evidence for those choices, not simply passing the preserved tests.
