# Deepwave Refactoring Progress Log

## Last Updated: 2026-03-18

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

## What Still Needs To Be Done

### Phase 3: Migrate Scalar
- Step 3.1: Create `scalar_func_generic` using `GenericForwardFunc` + `ScalarEquation`
- Step 3.2: Verify bit-identical outputs
- Step 3.3-3.6: Replace and remove old `ScalarForwardFunc`

### Phase 4: Migrate Acoustic
- Same pattern as Phase 3

### Phase 5: Migrate Elastic
- Same pattern, but more complex due to dimension-dependent structure

### Phase 6: Migrate ScalarBorn
- Create `ScalarBornEquation`

### Phase 7: Add Born Kernels for Acoustic and Elastic
- Requires modifying C code (backend_utils.py, CMakeLists.txt)

### Phase 8: Final Cleanup

## Important Code Locations

### Existing ForwardFunc/BackwardFunc classes (to be replaced):
- `scalar.py:586` - `ScalarForwardFunc`
- `scalar.py:1141` - `ScalarBackwardFunc`
- `acoustic.py:693` - `AcousticForwardFunc`
- `elastic.py:1163` - `ElasticForwardFunc`

### Key differences between propagators:

**Forward C call pattern:**
- Scalar: `backend_func(v.data_ptr(), sa.data_ptr(), wfc, wfp, *psi, *psin, *zeta, *storage, recv, *pml, src_i, recv_i, *rdx, *rdx2, dt**2, step_nt//step_ratio, ...)`
- Acoustic: `backend_func(*models, *sa, *wavefields, *storage, *recv, *pml, *src_i, *recv_i, *rdx, dt, step_nt, ...)`
- Elastic: same as acoustic but with 3 requires_grad flags

**Backward C call pattern:**
- Scalar: `backend_func(v2dt2, grad_r, grad_wfc, grad_wfp, *grad_psi, *grad_psin, *grad_zeta, *grad_zetan, *storage, grad_f, grad_v, grad_v_tmp, *pml, ...)`
- Acoustic: `backend_func(*models, *grad_r, *grad_wf, *aux_wf, *storage, *grad_f, *grad_models, *grad_tmp, *pml, ...)`
- Elastic: same pattern as acoustic

### Files NOT to modify (per instructions):
- `backend_utils.py`
- `common.py`
- `regular_grid.py`, `staggered_grid.py`
- `CMakeLists.txt`, `storage_utils.c/cu`
- `wavelets.py`, `location_interpolation.py`

## Running the Tests

Since the full test suite takes >1 hour, run targeted tests:
```bash
# Test that new code doesn't break anything
pytest tests/test_scalar.py -x -v

# After migration, compare old vs new
pytest tests/test_acoustic.py tests/test_callbacks_acoustic.py -x -v
pytest tests/test_elastic.py tests/test_callbacks_elastic.py -x -v
```

## Linter Commands (run after EVERY change)
```bash
ruff format src/deepwave/
ruff check --fix src/deepwave/
mypy --strict src/
```
