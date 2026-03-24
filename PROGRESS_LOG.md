# Deepwave Refactoring Progress Log

## Last Updated: 2026-03-22

## Summary of Work Done (March 22, 2026)

### Phase 1-6: Migration - COMPLETE
All propagators migrated to `GenericForwardFunc` and `PropagatorEquation`. 

### Major Fixes Applied
1. **API/Typing**: Reverted `Scalar` API to original signature (fixing `tests/test_propagator.py`) and fixed `fd_pad` type casting.
2. **Batching/Vmap**: Updated `ndim` detection in `scalar.py` and `GenericForwardFunc` to be robust against `vmap` (using `get_ndim` and `equation.get_model_batched`).
3. **Step Logic**: Corrected argument passing for `start_t` (outer steps for `Scalar`, inner steps for `Acoustic/Elastic`).
4. **Model Preparation**: Fixed double-preparation bug. `GenericForwardFunc` now prepares models before the C backend call. Updated `unpack_args` to correctly expect raw inputs.

### Known Regression / Open Issues
1. **Storage Mode (Disk/Compressed)**: 3D multi-shot gradients mismatch when using disk/compressed storage. Suspected interaction between the loop structure in `GenericForwardFunc` and `StorageManager` IO.
2. **Double Backward (Born)**: Currently disabled via `NotImplementedError` in `GenericBackwardFunc`. To be implemented in Phase 7.
3. **Acoustic/Elastic Numerical Divergence**: `gradcheck` failures exist for Acoustic/Elastic in 3D, potentially linked to the same storage/memory corruption issues as the disk mode regression.
