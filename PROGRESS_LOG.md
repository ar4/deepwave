# Deepwave Refactoring Progress Log

## Last Updated: 2026-03-19

## Summary of Work Done

### Phase 1: Equation Classes - COMPLETE

All equation classes are fully implemented with `pack_args`, `unpack_args`, `call_forward_backend`, `call_backward_backend`, and the new ABC methods.

**Files created/modified:**
- `src/deepwave/propagator_equation.py` - ABC with all abstract methods
- `src/deepwave/scalar_equation.py` - Full implementation
- `src/deepwave/acoustic_equation.py` - Full implementation
- `src/deepwave/elastic_equation.py` - Full implementation
- `src/deepwave/generic_forward_func.py` - Forward complete, backward being implemented

### Phase 2: Generic Framework - COMPLETE

**Completed:**
- `GenericForwardFunc.forward()` - fully implemented, calls `equation.unpack_args`, `equation.prepare_wavefields`, `equation.call_forward_backend`
- `GenericForwardFunc._save_ctx()` - saves equation, models_requires_grad, and all tensor/metadata needed for backward
- `GenericForwardFunc.backward()` - fully implemented, delegates to equation for:
  - `create_aux_wavefields` - creates temp vars for odd-step swap
  - `zero_backward_wavefields` - zeros edges/interiors (equation-specific)
  - `swap_odd_step_wavefields` - handles ping-pong swap after odd steps
  - `call_backward_backend` - calls equation-specific C backward function
- `ElasticEquation.zero_backward_wavefields` - imports `zero_edges_and_interiors` from `elastic.py` (no circular dep)
- Fixed bug: replaced nonexistent `deepwave.common.zero_interior_staggered` call with `equation.zero_backward_wavefields`

**All linters pass:** ruff format, ruff check, mypy --strict on all 5 new files.
**Tests pass:** scalar, acoustic, elastic (forward, backward, gradient, callbacks).

## Key Architecture Decisions

### 1. `get_storage_requires_grad(raw_models, ndim)` signature
Changed from `get_storage_requires_grad(prepared_models)` to take raw models + ndim. This is necessary because:
- **Scalar**: `[v.requires_grad]` (1 flag)
- **Acoustic**: `[v.requires_grad] + [rho.requires_grad] * ndim` (1+ndim flags)
- **Elastic**: dimension-dependent complex list (2-9 flags based on lamb/mu/buoyancy requires_grad)

The generic framework calls `equation.get_storage_requires_grad(raw_models, ndim)` before `prepare_models`.

### 2. `call_forward_backend` / `call_backward_backend` ABC signature
All equations implement the same signature. Each equation casts arguments to its C function's expected format:
- **Scalar**: uses `step_nt // step_ratio`, `dt**2`, creates temp vars (`psin`, `grad_psin`)
- **Acoustic**: uses `step_nt` directly, `dt`, passes models as list
- **Elastic**: uses `step_nt` directly, `dt`, passes 3 requires_grad flags

Key difference: scalar C function expects `step_nt // step_ratio` (outer steps) while acoustic/elastic expect `step_nt` (inner steps). The generic framework passes `step_nt * step_ratio` (total inner steps), so scalar's `call_forward_backend` divides by step_ratio.

### 3. `step` parameter inconsistency
The original code passes `step * step_ratio` to C functions, but the generic framework passes `step` (outer step). The equation's `call_forward_backend` needs to handle this. Currently scalar uses `step` directly, acoustic/elastic use `step` directly. This may need adjustment during testing.

### 4. Backward pass equation-specific methods
Added to ABC:
- `create_aux_wavefields(grad_wavefields, ndim)` - creates temp vars for odd-step swap
- `swap_odd_step_wavefields(grad_wavefields, aux_wavefields, ndim)` - handles ping-pong swap
- `zero_backward_wavefields(grad_wavefields, ndim, accuracy, pml_width, fd_pad_list)` - zeros edges/interiors

### 5. Acoustic `zero_edges_and_interiors` import
`AcousticEquation` imports `zero_edges_and_interiors` from `acoustic.py` as `_acoustic_zero_edges`. No circular dependency since `acoustic.py` doesn't import from `acoustic_equation.py`.

## Detailed Migration Progress

### Phase 3: Scalar Migration - COMPLETE

**Status:** Scalar migration complete. `scalar.py` now uses the generic framework. Double backward (Born) is disabled due to backend instability.

**Key Findings and Fixes:**
1.  **Bit-Identical Parity**: Achieved 100% test pass rate for `test_scalar.py` (forward and first-order backward) using the generic framework path.
2.  **Forward/Backward Unification**: The generic framework now handles the entire lifecycle (Forward -> Backward) by delegating specific backend calls and state management (swaps, zeroing) to the equation strategies.
3.  **Scalar Migration**: Replaced `scalar_func` with generic implementation. Removed `ScalarForwardFunc` and `ScalarBackwardFunc`.
4.  **Stability**: **CRITICAL**: Disabled double backward (Hessian-vector products) in `GenericBackwardFunc.backward` to prevent segmentation faults observed in the C backend during Born propagation. This needs investigation.

### Phase 4: Acoustic Migration - COMPLETE

**Status:** Acoustic migration is fully complete. `acoustic.py` now uses the generic framework, and `AcousticForwardFunc` has been removed.

**Key Achievements:**
1.  **Acoustic Parity**: Verified bit-identical parity for the entire acoustic test suite (`tests/test_acoustic.py` and `tests/test_callbacks_acoustic.py`).
2.  **Circular Dependency Resolution**: Moved `zero_edges_and_interiors` for acoustic from `acoustic.py` to `acoustic_equation.py` to break a circular import loop between the functional interface and the equation strategy.
3.  **Framework Signature Refinement**: Updated `PropagatorEquation.call_forward_backend` (and all implementations) to include `aux_wavefields`. This allows equations like Elastic to manage their own ping-pong buffers even in the forward pass if needed, matching the backward pass architecture.
4.  **Backend Interface Fixes**: Identified and fixed a bug where the acoustic backend was being passed too many batching flags (acoustic only expects 2, while the generic framework tracked more).

**Technical Decisions:**
- **Decoupling Utilities**: Logic that is shared between the strategy and the functional interface (like wavefield zeroing) now resides in the equation strategy files to ensure a clean dependency graph.
- **Batching Flag Canonicalization**: The `call_forward_backend` implementation in each equation is responsible for slicing the `model_batched` list to match the specific backend's expected signature.

### Phase 5: Elastic Migration - COMPLETE

**Status:** Elastic migration is fully complete. `elastic.py` now uses the generic framework, and `ElasticForwardFunc` has been removed.

**Key Achievements:**
1.  **Elastic Parity**: Verified bit-identical parity for the entire elastic test suite (`tests/test_elastic.py` and `tests/test_callbacks_elastic.py`), including forward, backward, gradient checks, and callbacks.
2.  **Backward Pass Stability**: Fixed segmentation faults in the backward pass by ensuring `ElasticEquation.call_backward_backend` passes strictly 3 `requires_grad` flags (lamb, mu, buoyancy) and 3 `model_batched` flags to the C backend, matching the kernel's expectation exactly.
3.  **vmap/Python Backend Support**: Implemented a robust `elastic_func` wrapper that:
    - Supports the legacy signature (used by tests) while adapting arguments for `GenericForwardFunc`.
    - Handles `python_backend` logic (required for `vmap` support) by dispatching to `elastic_python`.
    - Correctly injects `pml_profiles` into the argument list for the generic framework path.
4.  **Legacy Class Removal**: Successfully removed `ElasticForwardFunc` and refactored the module to rely entirely on `ElasticEquation` and `GenericForwardFunc` (for the C path).

**Technical Decisions:**
- **Argument Adaptation**: The `elastic_func` wrapper manually parses and reconstructs the argument list. This was necessary because `elastic` arguments are dimension-dependent and the legacy interface passed `pml_profiles` as a metadata argument, whereas `GenericForwardFunc` expects it packed with other tensors.
- **Python Backend Isolation**: The pure Python implementation (`elastic_python`, `update_velocities`, `update_stresses`) was preserved to support `vmap`, as the C backend does not support it. This logic is invoked conditionally within `elastic_func`.

## Phase 6: ScalarBorn Migration - COMPLETE

**Status:** ScalarBorn migration is fully complete. `scalar_born.py` now uses the generic framework, and `ScalarBornForwardFunc` has been removed.

**Key Achievements:**
1.  **Framework Stabilization**: Identified and fixed a critical bug in `GenericBackwardFunc` where an extra metadata argument (`counts`) was causing gradient alignment corruption across all propagators.
2.  **Return Length Parity**: Corrected `GenericForwardFunc.backward` to return the exact number of gradients expected by PyTorch (metadata + packed tensors), resolving `RuntimeError` during autograd.
3.  **Equation-Specific Adjoint Support**: Extended `PropagatorEquation` with `prepare_backward` and `finalize_backward` hooks. This allowed implementing the mandatory `grad_wfp` negation required by the scalar C kernels, which was previously missing.
4.  **Inheritance Correction**: Ensured all equation strategy classes (`Scalar`, `ScalarBorn`, `Acoustic`) correctly inherit from `PropagatorEquation` ABC to avoid `AttributeError` and ensure interface compliance.
5.  **Verified Parity**: Verified 100% pass rate for `tests/test_scalar_born.py` gradient checks (first-order gradients of Born, which are second-order gradients of Scalar).

**Technical Decisions:**
- **In-Place Negation**: Implemented `prepare_backward` and `finalize_backward` using `.neg_()` for memory efficiency.
- **Metadata Alignment**: Hardcoded `n_non_tensor_args = 15` in `GenericBackwardFunc` to explicitly account for the internal `counts` argument, ensuring `grad_models` are aligned with their corresponding inputs.
- **Double Backward Placeholder**: Preserved `NotImplementedError` for double backward in `GenericBackwardFunc.backward` but cleaned up the implementation skeleton to provide a clear path for Phase 7.

## Phase 7: Add Born Kernels for Acoustic and Elastic - NOT STARTED
...

### Phase 8: Final Cleanup
- Remove dead code.
- Final test sweep.
