# Deepwave Refactoring Instructions

## Objective

Refactor the Deepwave library to dramatically reduce code duplication across propagators (`scalar`, `scalar_born`, `acoustic`, `elastic`) by extracting their shared Python wrapper structure into a generic framework, enabling Born variants (and thus double backpropagation) for all propagators.

**Highest priority:** Code quality. The public Python API must remain **unchanged**.

## Verified Architecture

This section contains facts verified by reading the source. Trust these and do not re-investigate.

### File Sizes and Key Classes

| File | Lines | Key Classes/Functions |
|------|-------|-----------------------|
| `scalar.py` | 2356 | `Scalar` (Module), `scalar()`, `ScalarForwardFunc`, `ScalarBackwardFunc` |
| `scalar_born.py` | 1714 | `ScalarBorn` (Module), `scalar_born()`, `ScalarBornForwardFunc` |
| `acoustic.py` | 1715 | `Acoustic` (Module), `acoustic()`, `AcousticForwardFunc` |
| `elastic.py` | 2738 | `Elastic` (Module), `elastic()`, `ElasticForwardFunc` |
| `common.py` | 3208 | Shared utilities (already extracted) |

### C/CUDA Backend (Do Not Modify)

The compiled C/CUDA kernels are loaded via ctypes in `backend_utils.py`. Kernels have completely different signatures per propagator and cannot be unified without performance loss. The C/CUDA code, `backend_utils.py`, `common.py`, `regular_grid.py`, `staggered_grid.py`, `CMakeLists.txt`, `storage_utils.c/cu`, `wavelets.py`, and `location_interpolation.py` must not be modified.

### What Each Propagator Differs In

**Scalar** (regular grid, single wavefield pair + 2*ndim PML vars):
- Models: `[v]` (1 tensor)
- Source types: 1 (scalar injection)
- Receiver types: 1
- Wavefields: `[wfc, wfp, psi[ndim], zeta[ndim]]` = 2 + 2*ndim
- fd_pad: `[accuracy//2]*2*ndim`
- Uses `regular_grid.set_pml_profiles`
- Source amplitude scaling: `-source * v^2 * dt^2` (velocity-squared at source locations)
- `ScalarForwardFunc` has `_parse_args`, `_save_ctx`, `_prepare_wavefields`
- `ScalarBackwardFunc` has `backward()` that calls `scalar_born` for double backward

**ScalarBorn** (regular grid, background + scattered wavefields):
- Models: `[v, scatter]` (2 tensors)
- Source types: 2 (scalar source + scatter source)
- Receiver types: 2 (scalar + scattered scalar)
- Wavefields: doubles (background + scattered, each with PML vars)
- Used internally by `ScalarBackwardFunc.backward()` for Hessian-vector products

**Acoustic** (staggered grid, pressure + velocity + PML vars):
- Models: `[v, rho]` → prepared to `[K, buoyancy_z, buoyancy_y, buoyancy_x]`
- Source types: 1+ndim (pressure + velocity components)
- Receiver types: 1+ndim
- Wavefields: `[p, v_z, v_y, v_x, phi_z, phi_y, phi_x, psi_z, psi_y, psi_x]` = 1+3*ndim
- fd_pad: `[accuracy//2, accuracy//2-1]*ndim`
- Uses `staggered_grid.set_pml_profiles`
- Source amplitude scaling: multiply by model buoyancy at source location * dt (velocity sources), -dt (pressure source)
- `prepare_models()` computes K and buoyancy from v, rho
- `AcousticForwardFunc` has `_parse_args`, `_save_ctx`, `_parse_backward_args`, `_setup_callback_args`
- Has `zero_edges_and_interiors()` with staggered-grid half-grid masks
- No double backward currently

**Elastic** (staggered grid, velocity + stress + many PML vars):
- Models: `[lamb, mu, buoyancy]` → prepared to `[lamb, mu, mu_zy, mu_zx, mu_yx, buoyancy_z, buoyancy_y, buoyancy_x]`
- Source types: ndim+1 (force components + pressure)
- Receiver types: ndim+1
- Wavefields: dimension-dependent (1D: 4, 2D: 9, 3D: 23+)
- fd_pad: `[accuracy//2, accuracy//2-1]*ndim`
- Uses `staggered_grid.set_pml_profiles`
- `prepare_parameters()` does harmonic averaging of mu and buoyancy from buoyancy
- Source amplitude scaling: buoyancy-weighted for force sources, -dt for pressure source
- `ElasticForwardFunc` has **no** `_parse_args` — all parsing is inline in `forward()`
- Has `zero_edges_and_interiors()` with elastic-specific staggered half-grid masks
- `forward()` returns `(vz, vy, vx, stress..., m_vel..., m_stress..., recv_p, recv_z, recv_y, recv_x)` — dimension-dependent
- No separate `BackwardFunc` class — backward is a static method on `ForwardFunc`
- No double backward currently

### Argument Packing Differences

Each propagator's functional interface packs arguments differently before passing to `*Func.apply()`:
- **Scalar:** packs `*models, *source_amplitudes, pml_profiles, *sources_i, *receivers_i, grid_spacing, dt, nt, ...`, then `*wavefields`
- **Acoustic:** packs `grid_spacing, dt, nt, ...`, then `*models, *source_amplitudes, *pml_profiles, *sources_i, *receivers_i, *wavefields`
- **Elastic:** similar to acoustic

The generic framework must handle this, either by canonicalizing the argument order or by having each equation provide a `pack_args()` method.

---

## Design: PropagatorEquation (Strategy Pattern)

Use a concrete abstract class (NOT a dataclass) because:
- Elastic has dimension-dependent wavefield counts that vary by ndim
- Model preparation is non-trivial (harmonic averaging, bulk modulus computation)
- Wavefield preparation differs between regular and staggered grids
- Source amplitude scaling is propagator-specific

### Abstract Base Class

```python
# src/deepwave/propagator_equation.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import torch
import deepwave.common


class PropagatorEquation(ABC):
    """Abstract base class encapsulating equation-specific propagator logic."""

    @property
    @abstractmethod
    def name(self) -> str:
        """C function prefix: 'scalar', 'acoustic', 'elastic'."""
        ...

    @property
    @abstractmethod
    def grid_type(self) -> str:
        """'regular' or 'staggered'."""
        ...

    @property
    @abstractmethod
    def n_models(self) -> int:
        """Number of input model tensors (before prepare_models)."""
        ...

    @property
    def model_pad_modes(self) -> List[str]:
        """Padding mode per model tensor. Default: replicate for all."""
        return ["replicate"] * self.n_models

    @abstractmethod
    def get_ndim(
        self,
        models: List[torch.Tensor],
        initial_wavefields: List[Optional[torch.Tensor]],
        location_tensors: List[Optional[torch.Tensor]],
    ) -> int:
        """Determine spatial dimensions from inputs."""
        ...

    @abstractmethod
    def build_initial_wavefields(
        self, user_inputs: Dict[str, Optional[torch.Tensor]], ndim: int
    ) -> List[Optional[torch.Tensor]]:
        """Convert user-provided named wavefields to flat list for setup_propagator."""
        ...

    @abstractmethod
    def get_fd_pad(self, accuracy: int, ndim: int) -> List[int]:
        """Finite difference padding. Regular: [acc/2]*2*ndim. Staggered: [acc/2, acc/2-1]*ndim."""
        ...

    @abstractmethod
    def prepare_models(self, raw_models: List[torch.Tensor]) -> List[torch.Tensor]:
        """Transform input models for the C kernel.

        Scalar: identity [v] -> [v].
        Acoustic: [v, rho] -> [K, buoyancy_z, buoyancy_y, buoyancy_x].
        Elastic: [lamb, mu, buoyancy] -> [lamb, mu, mu_zy, mu_zx, mu_yx, buoyancy_z, buoyancy_y, buoyancy_x].
        """
        ...

    @abstractmethod
    def scale_source_amplitudes(
        self,
        source_amplitudes: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        prepared_models: List[torch.Tensor],
        dt: float,
        n_shots: int,
    ) -> List[torch.Tensor]:
        """Scale source amplitudes before calling the C kernel.

        Scalar: -source * v^2 * dt^2.
        Acoustic: multiply by model buoyancy at source locations * dt.
        Elastic: buoyancy-weighted for force sources, -dt for pressure source.
        """
        ...

    @abstractmethod
    def set_pml_profiles(
        self,
        pml_width: List[int],
        accuracy: int,
        fd_pad: List[int],
        dt: float,
        grid_spacing: List[float],
        max_vel: float,
        dtype: torch.dtype,
        device: torch.device,
        pml_freq: float,
        model_shape: torch.Size,
    ) -> List[torch.Tensor]:
        """Set PML profiles. Regular grid -> regular_grid module. Staggered -> staggered_grid module."""
        ...

    @abstractmethod
    def get_max_vel(
        self, raw_models: List[torch.Tensor]
    ) -> Tuple[float, float]:
        """Compute (max_vel, min_nonzero_vel) from raw models."""
        ...

    @abstractmethod
    def n_wavefields(self, ndim: int) -> int:
        """Number of wavefield tensors at given ndim."""
        ...

    @abstractmethod
    def n_source_types(self, ndim: int) -> int:
        """Number of source type tensors at given ndim."""
        ...

    @abstractmethod
    def n_receiver_types(self, ndim: int) -> int:
        """Number of receiver type tensors at given ndim."""
        ...

    @abstractmethod
    def get_storage_requires_grad(
        self, prepared_models: List[torch.Tensor]
    ) -> List[bool]:
        """Which model params require gradient storage allocation."""
        ...

    @abstractmethod
    def prepare_wavefields(
        self,
        wavefields: List[torch.Tensor],
        ndim: int,
        accuracy: int,
        device: torch.device,
        dtype: torch.dtype,
        size_with_batch: Tuple[int, ...],
        pml_width: List[int],
    ) -> List[torch.Tensor]:
        """Prepare wavefields (create_or_pad + zero_interior). Grid-specific."""
        ...

    @abstractmethod
    def get_callback_wavefield_names(self, ndim: int) -> List[str]:
        """Names for the callback wavefield dict."""
        ...

    @abstractmethod
    def get_callback_model_names(self, ndim: int) -> List[str]:
        """Names for the callback model dict."""
        ...
```

---

## Implementation Steps

The task is divided into very small steps. After EVERY step:
1. Run `ruff format src/deepwave/`
2. Run `ruff check --fix src/deepwave/`
3. Run `mypy --strict src/`
4. Run `pytest tests/ -x` (stops at first failure)

If ANY step fails, STOP. Fix the issue before proceeding. Do not accumulate broken states.

---

### Phase 1: Create Equation Classes (Pure New Code, No Risk)

#### Step 1.1: Create the abstract base class

Create `src/deepwave/propagator_equation.py` containing the `PropagatorEquation` ABC exactly as designed above. No other files are modified. Run `ruff format`, `ruff check --fix`, `mypy --strict src/`. You do no need to run the tests during this phase since no existing code is changed.

#### Step 1.2: Create ScalarEquation with only name, grid_type, n_models

Create `src/deepwave/scalar_equation.py` with a `ScalarEquation` class that implements only `name`, `grid_type`, and `n_models` properties. All other abstract methods can have placeholder `raise NotImplementedError`. Run linters.

#### Step 1.3: Add get_ndim and get_fd_pad to ScalarEquation

- `get_ndim`: delegates to `deepwave.common.get_ndim([v], [wavefield_0, wavefield_m1], [source_locations, receiver_locations], [psiz_m1, zetaz_m1], [psiy_m1, zetay_m1], [psix_m1, zetax_m1])`
- `get_fd_pad`: returns `[accuracy // 2] * 2 * ndim`

These are extracted from `scalar.py` function `scalar()` lines ~418-459. Verify by comparing outputs for the same inputs. Run linters.

#### Step 1.4: Add build_initial_wavefields to ScalarEquation

Extract the logic from `scalar.py` function `scalar()` lines ~426-442 that builds the `initial_wavefields` list from the named user inputs (wavefield_0, wavefield_m1, psiz_m1, etc.). Run linters.

#### Step 1.5: Add prepare_models to ScalarEquation

For scalar this is identity: `[v] -> [v]`. Implement as `return list(raw_models)`. Run linters.

#### Step 1.6: Add scale_source_amplitudes to ScalarEquation

Extract from `scalar.py` function `scalar()` lines ~515-530:
```python
# -source * v^2 * dt^2 at source locations
```
Use the exact same logic, including the IGNORE_LOCATION masking. Run linters.

#### Step 1.7: Add set_pml_profiles to ScalarEquation

Delegates to `deepwave.regular_grid.set_pml_profiles(...)` — extracted from `scalar.py` lines ~532-543. Run linters.

#### Step 1.8: Add get_max_vel to ScalarEquation

Extract from `scalar.py` lines ~444-458 (the max_vel computation). Returns `(max_vel, min_nonzero_vel)`. Run linters.

#### Step 1.9: Add remaining ScalarEquation methods

Add `n_wavefields`, `n_source_types`, `n_receiver_types`, `get_storage_requires_grad`, `prepare_wavefields`, `get_callback_wavefield_names`, `get_callback_model_names`. Extract their logic from `scalar.py` and `ScalarForwardFunc`. Run linters.

#### Step 1.10: Remove all placeholder NotImplementedError from ScalarEquation

Ensure every abstract method has a real implementation. Run linters. **This is the completion checkpoint for ScalarEquation.**

#### Step 1.11: Create AcousticEquation (name, grid_type, n_models, get_ndim, get_fd_pad)

Create `src/deepwave/acoustic_equation.py`. Start with the simple properties and `get_ndim` (from `acoustic.py` lines ~371-387) and `get_fd_pad` (returns `[accuracy//2, accuracy//2-1]*ndim`). Run linters.

#### Step 1.12: Add build_initial_wavefields to AcousticEquation

Extract from `acoustic.py` lines ~406-426 (building the named wavefield list). Run linters.

#### Step 1.13: Add prepare_models to AcousticEquation

Move `acoustic.py:prepare_models()` (lines 37-88) into `AcousticEquation.prepare_models()`. The original function computes K = rho*v^2 and buoyancy = 1/arithmetic_mean(rho) at staggered locations. Run linters.

#### Step 1.14: Add scale_source_amplitudes to AcousticEquation

Extract from `acoustic.py` lines ~566-584 (the source scaling loop). Run linters.

#### Step 1.15: Add set_pml_profiles and get_max_vel to AcousticEquation

- `set_pml_profiles`: delegates to `deepwave.staggered_grid.set_pml_profiles(...)`
- `get_max_vel`: extracted from `acoustic.py` lines ~428-442

Run linters.

#### Step 1.16: Add remaining AcousticEquation methods

Add `n_wavefields(returns 1+3*ndim)`, `n_source_types(returns 1+ndim)`, `n_receiver_types(returns 1+ndim)`, `get_storage_requires_grad`, `prepare_wavefields` (delegates to `deepwave.common.prepare_initial_wavefields` then applies acoustic-specific `zero_interior`), `get_callback_wavefield_names`, `get_callback_model_names`. Run linters. **Completion checkpoint for AcousticEquation.**

#### Step 1.17: Create ElasticEquation (name, grid_type, n_models, get_ndim, get_fd_pad)

Create `src/deepwave/elastic_equation.py`. Start with properties and `get_ndim` (from `elastic.py` lines ~655-696) and `get_fd_pad`. Run linters.

#### Step 1.18: Add build_initial_wavefields to ElasticEquation

Extract from `elastic.py` lines ~716-769 (building the named wavefield list). This is the most complex — dimension-dependent wavefield counts. Run linters.

#### Step 1.19: Add prepare_models to ElasticEquation

Move `elastic.py:prepare_parameters()` (lines 23-123) into `ElasticEquation.prepare_models()`. This does harmonic averaging of mu and arithmetic averaging of buoyancy at staggered locations. Run linters.

#### Step 1.20: Add scale_source_amplitudes to ElasticEquation

Extract from `elastic.py` lines ~926-978 (the dimension-dependent source scaling). Elastic has buoyancy-weighted force sources and -dt for pressure source. Run linters.

#### Step 1.21: Add set_pml_profiles and get_max_vel to ElasticEquation

- `set_pml_profiles`: delegates to `deepwave.staggered_grid.set_pml_profiles(...)`
- `get_max_vel`: extracted from `elastic.py` lines ~795-825 (computes vp, vs from lamb, mu, buoyancy)

Run linters.

#### Step 1.22: Add remaining ElasticEquation methods

Add `n_wavefields` (dimension-dependent dict), `n_source_types(returns ndim+1)`, `n_receiver_types(returns ndim+1)`, `get_storage_requires_grad`, `prepare_wavefields` (elastic-specific zero_interior with staggered half-grid masks from `elastic.py:zero_edges_and_interiors()`), `get_callback_wavefield_names`, `get_callback_model_names`. Run linters. **Completion checkpoint for ElasticEquation.**

#### Step 1.23: Run full test suite

`pytest tests/ -v`. All tests must pass (no existing code was modified yet, so this should be trivial). This verifies the equation classes are importable and don't break anything.

---

### Phase 2: Create Generic Framework (Still Pure New Code)

#### Step 2.1: Create generic_forward_func.py with ForwardFunc skeleton

Create `src/deepwave/generic_forward_func.py` with a `GenericForwardFunc(torch.autograd.Function)` class. For now, implement only:
- `forward(ctx, equation, ...)` that takes an equation instance and prints/log a message
- The class should NOT be used by any propagator yet

Run linters and tests.

#### Step 2.2: Implement _parse_args in GenericForwardFunc

The generic `_parse_args` must handle the argument packing differences. Design it to work with a canonical argument order defined by the equation:
- `n_models` model tensors
- `n_source_types` source amplitude tensors
- PML profile lists
- `n_source_types` source location tensors
- `n_receiver_types` receiver location tensors
- Metadata (grid_spacing, dt, nt, ...)
- `n_wavefields` wavefield tensors

Write this carefully. The scalar packing has `pml_profiles` as a single list, while acoustic has individual pml tensors. Test with synthetic data. Run linters and tests.

#### Step 2.3: Implement _save_ctx in GenericForwardFunc

Generic version of context saving. Extracted from the common pattern across all ForwardFuncs. Run linters and tests.

#### Step 2.4: Implement forward() body for GenericForwardFunc

Wire together:
1. Parse args via equation
2. Ensure contiguous
3. Setup storage
4. Save context
5. Prepare wavefields via equation
6. Get PML bounds
7. Get backend function
8. Time-stepping loop with callbacks
9. Downsample receiver outputs

This is the core of the framework. The key insight is that the C kernel call itself varies per propagator (different argument counts/types), so the equation must provide a method to call the kernel, OR the generic forward must build the argument list dynamically from the equation's counts.

**Design decision:** Have the equation provide a `call_backend()` method that takes the common infrastructure (backend function, wavefields, models, sources, receivers, etc.) and calls the C function with the correct argument order for that propagator. This avoids the generic framework needing to know the C function signature.

```python
@abstractmethod
def call_backend(
    self,
    backend_func,
    prepared_models: List[torch.Tensor],
    source_amplitudes: List[torch.Tensor],
    pml_profiles: List[torch.Tensor],
    sources_i: List[torch.Tensor],
    receivers_i: List[torch.Tensor],
    wavefields: List[torch.Tensor],
    # ... metadata ...
) -> List[torch.Tensor]:
    """Call the C backend function with the correct argument order."""
    ...
```

Run linters and tests.

#### Step 2.5: Implement backward() body for GenericForwardFunc

The backward pass. Similar structure to forward but calling the backward C function. Run linters and tests.

#### Step 2.6: Create GenericBackwardFunc

If the backward pass needs to support double backward (like `ScalarBackwardFunc`), create a separate `GenericBackwardFunc` class. For now, model it on `ScalarBackwardFunc` but make the Born invocation equation-specific. Run linters and tests.

#### Step 2.7: Full framework test

Run `pytest tests/ -v`. All tests pass (framework is not yet used by any propagator).

---

### Phase 3: Migrate Scalar (First Propagator — Prove the Framework)

#### Step 3.1: Create scalar_func_generic as a parallel copy

In `scalar.py`, create a second function `scalar_func_generic` that uses `GenericForwardFunc` with `ScalarEquation`. Keep the original `scalar_func` alongside it. Wire `scalar()` to call both (original as primary, generic as verification — compare outputs). Run linters and tests. Compare outputs numerically.

#### Step 3.2: Verify scalar_func_generic matches scalar_func exactly

Write a test that runs both with identical inputs and asserts outputs are bit-identical. If they differ, debug the generic framework against the equation class.

#### Step 3.3: Replace scalar_func with generic version

In `scalar.py`, replace `scalar_func` calls with the generic version. Remove the old `ScalarForwardFunc` class. Keep `ScalarBackwardFunc` for now (the double backward path).

#### Step 3.4: Remove ScalarForwardFunc

Delete the class. Run linters and tests. `pytest tests/test_scalar.py -v`.

#### Step 3.5: Migrate ScalarBackwardFunc

Replace `ScalarBackwardFunc` with generic version if possible. The `backward()` method of `ScalarBackwardFunc` calls `scalar_born_func` — ensure this still works. Run `pytest tests/test_scalar.py tests/test_scalar_born.py -v`.

#### Step 3.6: Verify scalar.py is shorter

Check line count. Target: under 1200 lines (down from 2356). Run full test suite.

---

### Phase 4: Migrate Acoustic

#### Step 4.1: Create acoustic_func_generic as parallel copy

In `acoustic.py`, create a version using `GenericForwardFunc` + `AcousticEquation`. Keep original. Compare outputs. Run linters and tests.

#### Step 4.2: Verify exact match

Run both versions with identical inputs. Assert bit-identical outputs.

#### Step 4.3: Replace acoustic_func with generic version

Swap in the generic version. Run `pytest tests/test_acoustic.py tests/test_callbacks_acoustic.py -v`.

#### Step 4.4: Remove AcousticForwardFunc

Delete the old class. Run linters and tests. Verify `acoustic.py` is shorter.

---

### Phase 5: Migrate Elastic

#### Step 5.1: Create elastic_func_generic as parallel copy

In `elastic.py`, create a version using `GenericForwardFunc` + `ElasticEquation`. Keep original. Compare outputs. Run linters and tests.

#### Step 5.2: Verify exact match

Run both versions with identical inputs. Assert bit-identical outputs. **Elastic is the hardest** — pay extra attention to dimension-dependent behavior (test 1D, 2D, 3D).

#### Step 5.3: Replace elastic_func with generic version

Swap in the generic version. Run `pytest tests/test_elastic.py tests/test_callbacks_elastic.py -v`.

#### Step 5.4: Remove ElasticForwardFunc

Delete the old class. Run linters and tests. Verify `elastic.py` is shorter.

---

### Phase 6: Migrate ScalarBorn

#### Step 6.1: Create ScalarBornEquation

Implement `ScalarBornEquation` as a subclass of `ScalarEquation` (or directly of `PropagatorEquation`). ScalarBorn has doubled wavefields (background + scattered), two source types, two receiver types. Run linters and tests.

#### Step 6.2: Create scalar_born_func_generic as parallel copy

In `scalar_born.py`, create a version using `GenericForwardFunc` + `ScalarBornEquation`. Keep original. Compare outputs. Run linters and tests.

#### Step 6.3: Replace scalar_born_func with generic version

Swap in the generic version. Run `pytest tests/test_scalar_born.py -v`. Verify exact match.

#### Step 6.4: Remove ScalarBornForwardFunc

Delete the old class. Run linters and tests. Verify `scalar_born.py` is shorter.

---

### Phase 7: Add Born Kernels for Acoustic and Elastic

#### Step 7.1: Design acoustic Born C kernel

Study `scalar_born.c` and `acoustic.c` to understand the pattern. The acoustic Born kernel needs:
- Background wavefields: pressure + velocity + PML vars (from regular acoustic)
- Scattered wavefields: same layout
- Scatter source: computed from background wavefield and scatter perturbation
- Two sets of receivers: background + scattered

Write the C kernel design document (even if just as code comments). This is a research task.

#### Step 7.2: Implement acoustic_born.c/cu

Write the C and CUDA kernels. Update `backend_utils.py` with new function templates. Update `CMakeLists.txt`.

#### Step 7.3: Create AcousticBornEquation

Implement using the framework. Run linters and tests.

#### Step 7.4: Create acoustic_born() function and AcousticBorn module

Follow the scalar_born pattern. Run linters and tests.

#### Step 7.5: Design and implement elastic Born C kernel

Similar to 7.1-7.4 but for elastic. More complex due to multiple material parameters (lambda, mu, buoyancy each produce scatter).

#### Step 7.6: Add double backward to acoustic and elastic

Use the new Born propagators to add `BackwardFunc.backward()` to acoustic and elastic, enabling double backpropagation.

---

### Phase 8: Final Cleanup

#### Step 8.1: Remove dead code

Any leftover functions, imports, or comments from the migration.

#### Step 8.2: Final full test suite

`pytest tests/ -v`. Every test must pass.

#### Step 8.3: Final linting

`ruff format src/`, `ruff check --fix src/`, `mypy --strict src/`.

#### Step 8.4: Verify examples

Run `python examples/example_fwi.py` (if it exists) and verify it works.

---

## Critical Rules

### After EVERY file change:
```bash
ruff format src/deepwave/
ruff check --fix src/deepwave/
mypy --strict src/
pytest tests/ -x
```

### Never add conditional branches to the generic framework

Do NOT write `if equation.name == "elastic": ...` in the generic framework. If the framework needs to know something equation-specific, add an abstract method to `PropagatorEquation`.

### Never modify files outside the scope

The following files must NOT be modified:
- `backend_utils.py`
- `common.py`
- `regular_grid.py`, `staggered_grid.py`
- `CMakeLists.txt`, `storage_utils.c/cu`
- `wavelets.py`, `location_interpolation.py`

(Phase 6 is the exception: `backend_utils.py` and `CMakeLists.txt` may be modified to add new Born kernel templates.)

### Verify numerical parity

When migrating a propagator, always create a parallel copy first, compare outputs bit-identically, and only then remove the original. Never trust that "it should be the same" — verify it.

### Handle ScalarBorn carefully

`scalar_born.py` is used internally by `ScalarBackwardFunc.backward()` for double backward. When modifying the scalar path, ensure the Born invocation still works. Do not break the backward chain.

### Elastic dimension-dependent structure

Elastic's ForwardFunc has no `_parse_args` helper — everything is inlined in `forward()`. When creating `ElasticEquation`, carefully extract each piece. Test 1D, 2D, and 3D cases separately.
