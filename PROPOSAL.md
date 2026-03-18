# Proposal: Deepwave Code Quality Improvement and Feature Enhancement

## Executive Summary

This proposal outlines a systematic approach to dramatically reduce code duplication in the Deepwave propagators while adding Born propagators (JVP support) for acoustic and elastic wave equations, and enabling functorch integration (vmap, jvp) for all propagators.

The key insight is that all propagators share a common structure despite differing in:
- Property models (velocity only vs. velocity+density vs. Lamé parameters)
- Source/receiver types (scalar vs. vector quantities)
- Grid type (regular vs. staggered)
- Wavefield update equations (in the C/CUDA kernels)

The Python wrapper code is currently ~70% duplicated across the four propagators. This can be reduced to a single generic propagator framework with equation-specific modules that only define their unique aspects.

---

## 1. Current State Analysis

### 1.1 File Sizes and Duplication

| File | Lines | Description |
|------|-------|-------------|
| `scalar.py` | ~2100 | Scalar propagator |
| `scalar_born.py` | ~1600 | Born scalar propagator |
| `acoustic.py` | ~1500 | Variable-density acoustic propagator |
| `elastic.py` | ~2100 | Elastic propagator |
| `common.py` | ~3200 | Shared utilities |

**Estimated duplication**: ~4000 lines of near-identical Python code across propagators.

### 1.2 Duplicated Patterns

Each propagator follows the same structure:

1. **Module class** (`Scalar`, `Acoustic`, `Elastic`, `ScalarBorn`): ~100 lines each
2. **Functional interface** (`scalar()`, `acoustic()`, `elastic()`, `scalar_born()`): ~200 lines each
3. **ForwardFunc** class with:
   - `_parse_args()`: ~60 lines each
   - `_save_ctx()`: ~50 lines each
   - `_prepare_wavefields()`: ~50 lines each
   - `forward()`: ~250 lines each
4. **BackwardFunc** class with:
   - `_parse_args()`: ~60 lines each
   - `_save_ctx()`: ~50 lines each
   - `forward()`: ~300 lines each
   - `backward()` (for double backward): ~200 lines each (only scalar)

### 1.3 Key Differences Between Propagators

| Aspect | Scalar | Scalar Born | Acoustic | Elastic |
|--------|--------|-------------|----------|---------|
| Models | `v` | `v`, `scatter` | `v`, `rho` | `lambda`, `mu`, `buoyancy` |
| Grid | Regular | Regular | Staggered | Staggered |
| Wavefields | `wfc`, `wfp`, `psi[ndim]`, `zeta[ndim]` | Same + scattered versions | `p`, `v[ndim]`, `phi[ndim]`, `psi[ndim]` | `v[ndim]`, `sigma[ndim*(ndim+1)/2]`, many PML vars |
| Sources | Scalar injection | Scalar injection + scatter | Pressure + velocity injection | Stress injection |
| Receivers | Scalar | Scalar + scattered | Pressure + velocity | Velocity components |
| Backend | `scalar` | `scalar_born` | `acoustic` | `elastic` |
| Born support | Via scalar_born | Built-in | No | No |
| Double backward | Yes | N/A (used for it) | No | No |

### 1.4 vmap Support (Current)

Currently, vmap has partial support:
- Property models (velocity) can be vmapped
- Sources, receivers, and wavefields cannot be vmapped
- `is_inside_vmap()` checks in `common.py` detect vmapped tensors
- Each propagator has custom vmap-handling code

---

## 2. Proposed Architecture

### 2.1 Core Design: Generic Propagator Framework

Create a `propagator_framework.py` module that provides:

```python
class PropagatorConfig:
    """Configuration for a specific propagator type."""
    name: str                    # e.g., "scalar", "acoustic", "elastic"
    model_names: List[str]       # e.g., ["v"] or ["v", "rho"]
    model_pad_modes: Dict        # padding mode per model
    wavefield_names: List[str]   # names of wavefields
    wavefield_pml_names: List[str]  # PML auxiliary wavefields
    source_types: List[str]      # e.g., ["scalar"] or ["pressure", "vz", "vy", "vx"]
    receiver_types: List[str]    # similar to source_types
    uses_staggered_grid: bool
    has_born_variant: bool

class PropagatorFunc(torch.autograd.Function):
    """Generic autograd function for wave propagation."""
    # Implements forward/backward using config

def make_propagator(config: PropagatorConfig) -> Callable:
    """Factory function that creates a propagator function."""
    # Returns a function with the appropriate signature
```

### 2.2 Equation-Specific Modules (Simplified)

Each propagator becomes a thin wrapper that:
1. Defines a `PropagatorConfig`
2. Provides model preparation (e.g., compute bulk modulus from velocity)
3. Calls the generic framework

Example for `scalar.py`:
```python
from deepwave.propagator_framework import PropagatorConfig, make_propagator

SCALAR_CONFIG = PropagatorConfig(
    name="scalar",
    model_names=["v"],
    model_pad_modes={"v": "replicate"},
    wavefield_names=["wfc", "wfp"],
    wavefield_pml_names=["psi", "zeta"],
    source_types=["scalar"],
    receiver_types=["scalar"],
    uses_staggered_grid=False,
    has_born_variant=True,
)

_scalar_impl = make_propagator(SCALAR_CONFIG)

def scalar(v, grid_spacing, dt, ...):
    """Scalar wave propagation."""
    return _scalar_impl(
        models={"v": v},
        grid_spacing=grid_spacing,
        dt=dt,
        ...
    )

class Scalar(torch.nn.Module):
    # Thin wrapper storing v and grid_spacing
    def forward(self, dt, ...):
        return scalar(self.v, self.grid_spacing, dt, ...)
```

### 2.3 Born Propagator Framework

Extend the framework to support Born propagation:

```python
class BornPropagatorConfig(PropagatorConfig):
    """Configuration for Born propagators."""
    scatter_names: List[str]     # e.g., ["scatter"] for scalar
    scatter_source_term: str     # How scatter couples to wavefield
    # For scalar: "2 * v * scatter * dt^2 * wavefield"

def make_born_propagator(config: BornPropagatorConfig) -> Callable:
    """Factory function for Born propagators."""
```

### 2.4 Functorch Integration Layer

Create a `functorch_compat.py` module:

```python
def enable_jvp_for_propagator(propagator_func):
    """Wrap a propagator to support torch.func.jvp."""
    # Implements JVP by running forward + perturbed forward
    # using the Born propagator internally

def enable_vmap_for_propagator(propagator_func):
    """Wrap a propagator to support torch.func.vmap."""
    # Handles batching over shots, models, etc.

# Decorator-style API
@supports_jvp
@supports_vmap
def scalar(v, grid_spacing, dt, ...):
    ...
```

---

## 3. Implementation Plan

### Phase 1: Extract Common Python Infrastructure (Low Risk)

**Goal**: Create shared utilities without changing any public API.

**Steps**:
1. Create `propagator_framework.py` with:
   - `PropagatorConfig` dataclass
   - Common forward/backward argument handling
   - Common wavefield preparation utilities
   - Common callback handling

2. Extract duplicated code patterns:
   - `_parse_args()` logic → generic function
   - `_save_ctx()` logic → generic function
   - Wavefield setup (create_or_pad, zero_interior) → already in common.py
   - Storage setup → already in common.py
   - C function invocation pattern → generic wrapper

3. Add tests to verify extracted code matches original behavior.

**Expected outcome**: ~1000 lines of new shared code, each propagator reduced by ~200-300 lines.

### Phase 2: Refactor Scalar Propagator as Template (Medium Risk)

**Goal**: Prove the framework works by converting one propagator.

**Steps**:
1. Rewrite `scalar.py` using the framework
2. Ensure all existing tests pass
3. Benchmark to verify no performance regression
4. Document the pattern for other propagators

**Expected outcome**: `scalar.py` reduced from ~2100 to ~800 lines.

### Phase 3: Refactor Remaining Propagators (Medium Risk)

**Goal**: Convert acoustic, elastic, and scalar_born.

**Steps**:
1. Convert `acoustic.py` using the proven pattern
2. Convert `elastic.py` (most complex due to many wavefields)
3. Merge `scalar_born.py` into `scalar.py` (it's a variant)

**Expected outcome**: Each propagator reduced to 400-600 lines.

### Phase 4: Add Born Propagators for Acoustic and Elastic (Higher Risk)

**Goal**: Implement JVP support for all propagators.

**Steps**:
1. **Acoustic Born**:
   - Create `acoustic_born.c`/`.cu` (C/CUDA kernel)
   - Add scattered wavefield for pressure and velocity
   - Scatter source: `scatter * background_wavefield`
   - Update `backend_utils.py` with new function templates

2. **Elastic Born**:
   - Create `elastic_born.c`/`.cu` (C/CUDA kernel)
   - Add scattered wavefields for velocity and stress
   - Scatter source from Lamé parameter perturbations
   - More complex due to multiple model parameters

3. **Python wrappers**:
   - Add `AcousticBorn` class and `acoustic_born()` function
   - Add `ElasticBorn` class and `elastic_born()` function
   - Use the Born propagator framework from Phase 1

4. **Double backward**:
   - Add `BackwardFunc.backward()` for acoustic and elastic
   - Uses the Born propagator for Hessian-vector products

**Expected outcome**: All propagators support JVP and double backward.

### Phase 5: Functorch Integration (Medium Risk)

**Goal**: Enable `torch.func.jvp`, `torch.func.vmap` with all propagators.

**Steps**:

1. **JVP Support**:
   ```python
   # For each propagator, register a custom JVP rule
   @torch.library.custom_op("deepwave::scalar_jvp", mutates_args=())
   def scalar_jvp(v, v_tangent, ...):
       # Run background propagation
       # Run Born propagation with v_tangent as scatter
       # Return both outputs
   ```

2. **vmap Enhancement**:
   - Current vmap only works for property models
   - Extend to support vmapping over:
     - Multiple shots (batch dimension)
     - Multiple source configurations
     - Multiple model perturbations (for optimization)

3. **Integration with PyTorch's autograd.Function**:
   - Ensure compatibility with `torch.compile`
   - Register proper JVP rules using `torch.autograd.Function.setup_context`

**Expected outcome**:
```python
from torch.func import jvp, vmap

# JVP: effect of velocity perturbation on output
output, output_tangent = jvp(
    lambda v: deepwave.scalar(v, ...),
    (v,),
    (v_tangent,)
)

# vmap: batch over multiple shots
batched_scalar = vmap(lambda v, src: deepwave.scalar(v, sources=src, ...))
outputs = batched_scalar(velocities, sources)
```

---

## 4. Functorch Integration Details

### 4.1 JVP Implementation

For the scalar propagator with JVP:

```python
def scalar_with_jvp(v, v_tangent, source_amplitudes, ...):
    """Scalar propagation with Jacobian-vector product."""
    # Forward pass with background velocity
    outputs = scalar(v, source_amplitudes, ...)

    # For JVP, we need the Born propagator
    # The scattered field is sourced by: 2 * v * v_tangent * dt^2 * wavefield
    scatter = v_tangent / v  # Normalize to get scattering potential
    outputs_born = scalar_born(
        v, scatter, source_amplitudes, ...
    )

    return outputs, outputs_born[-1]  # Main output and tangent
```

### 4.2 Double Backward (Hessian-Vector Products)

The scalar propagator supports double backward for computing Hessian-vector products (HVP). This works by:

1. **First backward** (`ScalarBackwardFunc.forward()`): Runs adjoint propagation to compute gradients
2. **Second backward** (`ScalarBackwardFunc.backward()`): Uses the Born propagator to compute how perturbations affect the gradients

The second backward works as follows:
- Background wavefield: Regular forward propagation of original sources
- Scattered wavefield: Born propagation with perturbation as scatter
- Adjoint background: Regular backward propagation of gradient sources
- Adjoint scattered: Born backward propagation to compute HVP

Key implementation detail: The Born propagator is called twice:
1. Forward Born: Computes perturbed wavefield from velocity/source perturbations
2. Backward Born: Computes how these perturbations affect the gradients

Extending this to acoustic and elastic requires:
1. Born propagators for each equation (Phase 4)
2. `BackwardFunc.backward()` implementations that call the Born propagator
3. Proper handling of multiple model parameters (elastic has 3: lambda, mu, buoyancy)

### 4.3 Custom Op Registration

```python
@torch.library.custom_op("deepwave::scalar_forward", mutates_args=())
def scalar_forward(
    v: torch.Tensor,
    source_amplitudes: torch.Tensor,
    ...
) -> tuple[torch.Tensor, ...]:
    # Implementation
    ...

@scalar_forward.register_fake
def scalar_forward_fake(v, source_amplitudes, ...):
    # Return fake tensors with correct shapes
    ...

# Register JVP
@torch.library.register_jvp("deepwave::scalar_forward", "scalar_forward_jvp")
def scalar_forward_jvp(primals, tangents):
    v, source_amplitudes, ... = primals
    v_t, source_amplitudes_t, ... = tangents

    # Run forward with primals
    outputs = scalar_forward(v, source_amplitudes, ...)

    # Run Born with tangents
    # ... compute JVP ...

    return outputs, output_tangents
```

### 4.3 vmap Implementation

```python
@torch.library.register_vmap("deepwave::scalar_forward", "scalar_forward_vmap")
def scalar_forward_vmap(info, batch_dims, v, source_amplitudes, ...):
    """vmap support for scalar propagator."""
    # batch_dims tells us which inputs have batch dimensions
    # We need to handle:
    # 1. Batching over shots (already supported natively)
    # 2. Batching over models (run multiple propagations)
    # 3. Batching over sources (multiple source configurations)

    if all(bd is None for bd in batch_dims):
        # No batched inputs, just run normally
        return scalar_forward(v, source_amplitudes, ...), batch_dims

    # Delegate to specialized batching logic
    return _batched_scalar_forward(
        info, batch_dims, v, source_amplitudes, ...
    ), batch_dims
```

---

## 5. Performance Considerations

### 5.1 C/CUDA Kernels (No Changes to Hot Path)

The refactoring only changes Python wrapper code. The actual computation:
- C/CUDA kernels remain unchanged
- Function signatures remain identical
- No additional memory allocations in the forward/backward pass

### 5.2 Python Overhead

- Generic framework adds minimal overhead (~microseconds per call)
- All tensor operations remain in C/CUDA
- Autograd graph structure unchanged

### 5.3 Benchmarking Plan

Before/after benchmarks for:
1. Forward pass time (should be identical)
2. Backward pass time (should be identical)
3. Memory usage (should be identical)
4. Python call overhead (should be <1% different)

---

## 6. Risk Mitigation

### 6.1 Backward Compatibility

- Public API remains identical: `scalar()`, `acoustic()`, `elastic()`, `scalar_born()`
- All existing tests must pass
- Add deprecation warnings for any changed internal APIs

### 6.2 Incremental Rollout

- Phase 1-2: Internal refactoring, fully tested
- Phase 3: Expand to other propagators
- Phase 4-6: New features, opt-in initially

### 6.3 Testing Strategy

1. **Unit tests**: Each framework component
2. **Integration tests**: Each propagator against reference
3. **Performance tests**: Benchmark suite
4. **Born propagator tests**:
   - JVP vs finite differences
   - Born forward vs perturbed regular forward
   - Double backward for Hessian-vector products

---

## 7. Summary of Expected Outcomes

| Metric | Current | After Refactoring |
|--------|---------|-------------------|
| Total Python lines | ~9300 | ~4500 |
| Propagator implementations | 4 separate files | 1 framework + 4 thin wrappers |
| Born propagators | 1 (scalar only) | 4 (all propagators) |
| Double backward support | 1 (scalar only) | 4 (all propagators) |
| functorch JVP support | Partial/manual | Native |
| functorch vmap support | Partial | Full |
| Public API changes | N/A | None |
| Performance impact | N/A | Negligible |

---

## 8. Recommended Next Steps

1. **Review and approve this proposal**
2. **Start with Phase 1**: Extract common infrastructure
3. **Prototype Phase 2**: Refactor scalar as proof of concept
4. **Validate**: Run full test suite, benchmark
5. **Iterate**: Apply pattern to remaining propagators
6. **Extend**: Add Born propagators and functorch support

---

## 9. Elastic Propagator Special Considerations

The elastic propagator is the most complex due to:

### 9.1 Multiple Wavefield Components

In 3D, the elastic propagator has:
- 3 velocity components (vx, vy, vz)
- 6 stress components (sigmaxx, sigmayy, sigmazz, sigmaxy, sigmaxz, sigmayz)
- 14 PML memory variables for velocity (m_vxx, m_vxy, m_vxz, m_vyx, m_vyy, m_vyz, m_vzx, m_vzy, m_vzz, plus 5 more for stress)
- Total: 23+ wavefields in 3D

### 9.2 Multiple Source/Receiver Types

- Pressure sources/receivers (hydrophone)
- Velocity component sources/receivers (geophones) - one per dimension
- Each with separate locations and amplitudes

### 9.3 Free Surface Boundary

The elastic propagator uses the "improved vacuum method" (Zeng et al., 2012) which requires:
- Harmonic averaging of mu at staggered locations
- Special buoyancy handling at free surfaces
- Modified material parameters in `prepare_parameters()`

### 9.4 Framework Adaptation

The generic framework needs special handling for elastic:
- `wavefield_names` becomes a dict keyed by dimension: `{3: [...], 2: [...], 1: [...]}`
- Source/receiver type lists are dimension-dependent
- Model preparation is more complex (mu averaging for free surface)

### 9.5 Born Elastic Design Recommendation

For elastic Born, I recommend:
1. **Separate scattering potentials for each material parameter**: `scatter_lambda`, `scatter_mu`, `scatter_buoyancy`
2. **Born propagator computes**: Effect of perturbations in each parameter on the wavefield
3. **Usage**: `elastic_born(lamb, mu, buoyancy, scatter_lambda, scatter_mu, scatter_buoyancy, ...)`
4. **Gradient computation**: Each scatter parameter has its own gradient

---
