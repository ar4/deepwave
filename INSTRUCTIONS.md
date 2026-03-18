# Deepwave Refactoring Instructions

## Objective

Refactor the Deepwave library to dramatically reduce code duplication across propagators (`scalar`, `scalar_born`, `acoustic`, `elastic`) by extracting their shared Python wrapper structure into a generic framework. The primary goal is **code quality improvement**. Functorch/vmap/JVP integration is explicitly deferred and should not be attempted.

**Constraint:** The public Python API must remain **unchanged**. Every test in `tests/` must pass after each phase.

---

## Verified Architecture (Read This Before Starting)

This section contains findings you should trust and use. Do not re-investigate these facts.

### File Sizes and Key Classes

| File | Lines | Key Classes/Functions |
|------|-------|-----------------------|
| `scalar.py` | 2356 | `Scalar` (Module), `scalar()` (functional), `ScalarForwardFunc`, `ScalarBackwardFunc` |
| `scalar_born.py` | 1714 | `ScalarBorn` (Module), `scalar_born()` (functional), `ScalarBornForwardFunc` |
| `acoustic.py` | 1715 | `Acoustic` (Module), `acoustic()` (functional), `AcousticForwardFunc` |
| `elastic.py` | 2738 | `Elastic` (Module), `elastic()` (functional), `ElasticForwardFunc` |
| `common.py` | 3208 | Shared utilities (already extracted) |

### C/CUDA Backend (No Changes Needed)

The compiled C/CUDA kernels are loaded via ctypes (`backend_utils.py`). The kernels have completely different signatures per propagator and cannot be unified without performance loss. **Do not attempt to modify the C/CUDA code or `backend_utils.py`.** The refactoring is Python-only.

Key backend facts:
- C functions are named like `scalar_iso_2d_4_float_forward_cpu`
- Backend functions are retrieved via `deepwave.backend_utils.get_backend_function(name, ndim, "forward", accuracy, dtype, device)`
- The function receives raw pointers via `data_ptr()` and returns `c_int`

### Why Python-Level Born Composition Is Not Feasible

The acoustic and elastic C kernels **hardcode their source types**. The acoustic kernel accepts only pressure and velocity-component sources in fixed slots. The elastic kernel accepts only stress and force-component sources. There is no way to inject an arbitrary "scatter source" into the background wavefield without writing new C/CUDA Born kernels. This was verified by examining `backend_utils.py` function templates.

### Why Functorch Is Not Feasible

The ctypes-based architecture is incompatible with `torch.library.custom_op`. The kernels operate on raw memory pointers, not PyTorch tensors with metadata. Integrating with `torch.func.vmap`/`torch.func.jvp` would require rewriting the build system (CMake + scikit-build-core) and the entire C interface. This is a separate future project and should not be attempted here.

### What vmap Support Currently Exists

`common.py` imports `torch._C._functorch` and has:
- `is_inside_vmap(tensor)` - checks if a tensor is inside a vmap
- `check_inputs_not_vmapped(*args)` - raises if most inputs are vmapped
- Property models (velocity) can be vmapped; sources, receivers, and wavefields cannot

This is partial and fragile. Leave it as-is.

---

## Shared Patterns (What to Extract)

Each propagator's `ForwardFunc` follows this identical structure. The differences are in **counts** (how many models, how many wavefields, how many source/receiver types) and **model preparation** (what happens between `setup_propagator` and calling the C kernel).

### Pattern in the Functional Interface (`scalar()`, `acoustic()`, `elastic()`)

1. Call `check_inputs_not_vmapped()` on all tensor inputs
2. Call `get_ndim()` to determine spatial dimensions
3. Build `initial_wavefields` list (dimension-dependent for acoustic/elastic)
4. Handle `max_vel` / vmap detection
5. Set `fd_pad` (different: scalar uses `[accuracy//2]*2*ndim`, acoustic/elastic use `[accuracy//2, accuracy//2-1]*ndim`)
6. Call `deepwave.common.setup_propagator()` with models, sources, receivers
7. Scale source amplitudes (propagator-specific logic)
8. Set PML profiles (regular_grid vs staggered_grid)
9. Call the `*Func` class via `scalar_func()` / `acoustic_func()` / `elastic_func()`
10. Downsample receiver outputs

### Pattern in ForwardFunc.forward()

1. Parse args into models, source_amplitudes, pml_profiles, sources_i, receivers_i, wavefields
2. Ensure all tensors are contiguous
3. Get device, dtype, ndim
4. Call `setup_storage()`
5. Save context (for backward)
6. Prepare initial wavefields (create_or_pad, zero_interior)
7. Get backend function pointer
8. Compute PML bounds
9. Enter time-stepping loop with callback support
10. Call the C kernel function with proper arguments
11. Return output wavefields and receiver_amplitudes

### Pattern in ForwardFunc.backward()

1. Parse saved tensors from ctx
2. Ensure contiguity of gradients
3. Get backend function pointer (for backward direction)
4. Set up storage
5. Prepare adjoint wavefields
6. Call backward C kernel
7. Return gradients for models, sources, wavefields

### Differences That Must Be Preserved

**Scalar:**
- Models: `[v]` (1 tensor)
- Source/receiver types: 1 each (scalar)
- Wavefields: `[wfc, wfp, psi[ndim], zeta[ndim]]` = 2 + 2*ndim
- fd_pad: `[accuracy//2]*2*ndim`
- Uses `regular_grid.set_pml_profiles`
- Source amplitude scaling: `-source * v^2 * dt^2`

**ScalarBorn:**
- Models: `[v, scatter]` (2 tensors)
- Source types: 2 (scalar source + scatter source)
- Receiver types: 2 (scalar + scattered scalar)
- Wavefields: doubles (background + scattered)
- Has its own ForwardFunc with no `_parse_args` helper

**Acoustic:**
- Models: `[v, rho]` → prepared to `[K, buoyancy_z, buoyancy_y, buoyancy_x]`
- Source types: 1 + ndim (pressure + velocity components)
- Receiver types: 1 + ndim
- Wavefields: `[p, v_z, v_y, v_x, phi_z, phi_y, phi_x, psi_z, psi_y, psi_x]` = 1 + 3*ndim
- fd_pad: `[accuracy//2, accuracy//2-1]*ndim`
- Uses `staggered_grid.set_pml_profiles`
- Source amplitude scaling: multiply by model value at source location * dt (velocity sources), -dt (pressure source)
- `prepare_models()` computes bulk modulus K and buoyancy from v, rho

**Elastic:**
- Models: `[lamb, mu, buoyancy]` → prepared to `[lamb, mu, mu_zy, mu_zx, mu_yx, buoyancy_z, buoyancy_y, buoyancy_x]`
- Source types: ndim + 1 (force components + pressure)
- Receiver types: ndim + 1
- Wavefields: dimension-dependent with many PML memory variables (3D: 3 velocity + 6 stress + 9 velocity PML + 5 stress PML = 23)
- fd_pad: `[accuracy//2, accuracy//2-1]*ndim`
- Uses `staggered_grid.set_pml_profiles`
- `prepare_parameters()` does harmonic averaging of mu and buoyancy from buoyancy
- Elastic's ForwardFunc has **no `_parse_args`** - everything is in `forward()`
- Elastic's `forward()` has no separate `_save_ctx` - context saving is inline
- Elastic does not have a separate BackwardFunc class (backward is a static method on ForwardFunc)
- Elastic has `zero_edges_and_interiors()` with staggered-grid-specific half-grid masks

---

## Design: The Propagator Equation (Strategy Pattern)

Create a concrete abstract class that each propagator implements. This is NOT a dataclass of lists - it is a strategy with methods.

```python
# src/deepwave/propagator_equation.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import deepwave.common


class PropagatorEquation(ABC):
    """Abstract base for equation-specific logic.

    Subclasses encapsulate the differences between propagators:
    - How many models and what they are
    - How to prepare models (e.g., compute K from v, rho)
    - How many wavefields and their layout
    - How to set PML profiles
    - How to scale source amplitudes
    - How to call the backend function
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Propagator name matching C function prefix (e.g., 'scalar', 'acoustic')."""
        ...

    @property
    @abstractmethod
    def grid_type(self) -> str:
        """'regular' or 'staggered'."""
        ...

    @property
    @abstractmethod
    def n_models(self) -> int:
        """Number of model tensors expected (e.g., 1 for scalar, 3 for elastic)."""
        ...

    @property
    def model_pad_modes(self) -> List[str]:
        """Padding mode for each model. Default: replicate for all."""
        return ["replicate"] * self.n_models

    @abstractmethod
    def get_ndim(self, models: List[torch.Tensor],
                 initial_wavefields: List[Optional[torch.Tensor]],
                 sources_i: List[Optional[torch.Tensor]],
                 receivers_i: List[Optional[torch.Tensor]]) -> int:
        """Determine ndim from inputs. Uses common.get_ndim."""
        ...

    @abstractmethod
    def build_initial_wavefields(
        self, user_wavefields: Dict[str, Optional[torch.Tensor]], ndim: int
    ) -> List[Optional[torch.Tensor]]:
        """Convert user-provided named wavefields to the flat list for setup_propagator."""
        ...

    @abstractmethod
    def get_fd_pad(self, accuracy: int, ndim: int) -> List[int]:
        """Compute fd_pad. Differs between regular and staggered grids."""
        ...

    @abstractmethod
    def prepare_models(self, models: List[torch.Tensor]) -> List[torch.Tensor]:
        """Prepare models for the C kernel.

        For scalar: return [v] unchanged.
        For acoustic: compute K and buoyancy from [v, rho].
        For elastic: compute staggered mu and buoyancy from [lamb, mu, buoyancy].
        """
        ...

    @abstractmethod
    def scale_source_amplitudes(
        self,
        source_amplitudes: List[torch.Tensor],
        sources_i: List[torch.Tensor],
        models: List[torch.Tensor],
        dt: float,
        n_shots: int,
    ) -> List[torch.Tensor]:
        """Scale source amplitudes before calling the C kernel.

        Each propagator scales differently (e.g., scalar uses -v^2*dt^2).
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
        """Set PML profiles using the appropriate grid module."""
        ...

    @abstractmethod
    def get_max_vel(
        self, models: List[torch.Tensor]
    ) -> Tuple[float, float]:
        """Compute max velocity and min nonzero velocity from models.

        Returns (max_vel, min_nonzero_vel).
        """
        ...

    @abstractmethod
    def n_wavefields(self, ndim: int) -> int:
        """Number of wavefield tensors for this equation at given ndim."""
        ...

    @abstractmethod
    def n_source_types(self, ndim: int) -> int:
        """Number of source type tensors (e.g., 1 for scalar, 1+ndim for acoustic)."""
        ...

    @abstractmethod
    def n_receiver_types(self, ndim: int) -> int:
        """Number of receiver type tensors."""
        ...

    @abstractmethod
    def get_storage_requires_grad(
        self, models: List[torch.Tensor]
    ) -> List[bool]:
        """Which model parameters contribute to gradient storage allocation."""
        ...

    @abstractmethod
    def prepare_initial_wavefields(
        self,
        wavefields: List[torch.Tensor],
        ndim: int,
        accuracy: int,
        device: torch.device,
        dtype: torch.dtype,
        size_with_batch: Tuple[int, ...],
        pml_width: List[int],
    ) -> List[torch.Tensor]:
        """Prepare wavefields (create_or_pad + zero_interior).

        This differs between regular and staggered grids.
        """
        ...

    @abstractmethod
    def get_callback_wavefield_names(self, ndim: int) -> List[str]:
        """Get names for callback wavefield dict."""
        ...

    @abstractmethod
    def get_callback_model_names(self, ndim: int) -> List[str]:
        """Get names for callback model dict."""
        ...
```

### Why This Design (Not a Dataclass of Lists)

A static `PropagatorConfig` dataclass cannot handle:
- Elastic's dimension-dependent argument counts (23+ wavefields in 3D)
- Model preparation logic (harmonic averaging, bulk modulus computation)
- Different source amplitude scaling per propagator
- Different wavefield preparation (regular vs staggered grid zero_interior)

A strategy pattern with abstract methods handles all of these cleanly.

---

## Implementation Phases

### Phase 1: Create the Equation Classes (No Propagator Changes)

**Goal:** Create the abstract base class and concrete implementations. This is pure new code with no risk to existing functionality.

1. Create `src/deepwave/propagator_equation.py` with `PropagatorEquation` ABC
2. Create `src/deepwave/scalar_equation.py` implementing `ScalarEquation`:
   - Extract `n_models=1`, `name="scalar"`, `grid_type="regular"`
   - Extract fd_pad computation: `[accuracy//2]*2*ndim`
   - Extract source scaling: `-source * v^2 * dt^2`
   - Extract `prepare_models` (identity for scalar)
   - Extract PML profile setup (delegates to `regular_grid.set_pml_profiles`)
   - Extract wavefield preparation (uses `common.prepare_initial_wavefields` with regular grid zero_interior)
3. Create `src/deepwave/acoustic_equation.py` implementing `AcousticEquation`:
   - Extract `prepare_models` (move from `acoustic.py:prepare_models`)
   - Extract source scaling logic
   - Extract fd_pad: `[accuracy//2, accuracy//2-1]*ndim`
   - Extract PML profile setup (delegates to `staggered_grid.set_pml_profiles`)
   - Extract wavefield preparation (staggered grid zero_interior with half-grid masks)
4. Create `src/deepwave/elastic_equation.py` implementing `ElasticEquation`:
   - Extract `prepare_parameters` (move from `elastic.py:prepare_parameters`)
   - Extract source scaling logic (buoyancy-weighted sources, dimension-dependent indices)
   - Extract fd_pad, PML profiles, wavefield preparation
   - Handle dimension-dependent wavefield counts
5. Write tests in `tests/test_propagator_equation.py` that verify each equation class produces the same models, fd_pad, and PML profiles as the original code

**Acceptance criteria:**
- All existing tests pass (no existing files modified yet)
- New equation classes produce identical outputs to the original inline code
- `ruff check src/` passes
- `mypy src/deepwave/propagator_equation.py` passes (if type checking is configured)

### Phase 2: Create the Generic Propagator Framework

**Goal:** Create the generic `ForwardFunc` and `BackwardFunc` that use `PropagatorEquation`.

1. Create `src/deepwave/propagator_framework.py` with:
   - `GenericForwardFunc(torch.autograd.Function)` that accepts a `PropagatorEquation` instance
   - `GenericBackwardFunc(torch.autograd.Function)` that accepts a `PropagatorEquation` instance
   - A factory function or mechanism to create equation-specific `*Func` classes

   The generic forward must handle:
   - `_parse_args`: Equations declare their counts (`n_models`, `n_wavefields`, etc.)
   - `_save_ctx`: Same structure for all, save models, sources, receivers, wavefields, pml_profiles
   - `_prepare_wavefields`: Delegates to equation's `prepare_initial_wavefields`
   - `forward`: The full propagation pipeline

   **Important design consideration:** The generic forward function receives all arguments via `*args` and must parse them according to the equation's configuration. The argument packing order in each propagator's `*_func()` function determines how arguments arrive. The generic `_parse_args` must unpack in the same order.

   Look carefully at how each propagator packs arguments before passing to `*Func.apply()`:
   - `scalar`: packs `*models, *source_amplitudes, pml_profiles, *sources_i, *receivers_i, grid_spacing, dt, ...`, then `*wavefields`
   - `acoustic`: packs `grid_spacing, dt, nt, ..., *models, *source_amplitudes, *pml_profiles, *sources_i, *receivers_i, *wavefields`
   - `elastic`: packs similar to acoustic

   The generic framework must support **both** packing styles, OR each equation must provide a consistent packing function. Recommendation: have each equation provide a `pack_args()` method that reorders arguments into a canonical order, and the generic `_parse_args` unpacks from that canonical order.

2. Write tests that verify the generic forward produces the same outputs as the original `ScalarForwardFunc`

**Acceptance criteria:**
- All existing tests pass
- Generic forward produces bit-identical outputs to original scalar forward
- The framework correctly handles the argument packing differences

### Phase 3: Migrate Scalar to the Framework

**Goal:** Replace `ScalarForwardFunc` and `ScalarBackwardFunc` with the generic framework.

1. Update `src/deepwave/scalar.py`:
   - Keep the `scalar()` function signature unchanged
   - Replace `ScalarForwardFunc` with `GenericForwardFunc` + `ScalarEquation`
   - Replace `ScalarBackwardFunc` with `GenericBackwardFunc` + `ScalarEquation`
   - Keep `scalar_python()` and `_forward_step()` for the Python backend (these are used for `python_backend=True`)
   - **Keep `ScalarBackwardFunc.backward()` (the double backward)** - this is the Born-based Hessian-vector product. It may need to stay as a custom backward for now, or be integrated into the generic framework if the framework design supports it

2. Update `src/deepwave/scalar_born.py` if needed (scalar_born is used by ScalarBackwardFunc.backward for double backward)

3. Run all tests:
   - `pytest tests/test_scalar.py`
   - `pytest tests/test_scalar_born.py`
   - `pytest tests/test_propagator.py`
   - `pytest tests/test_callbacks_scalar.py`

**Acceptance criteria:**
- All tests pass with no output differences
- `scalar.py` is significantly shorter (target: ~800-1000 lines, down from 2356)
- The public API of `scalar()` and `Scalar` is unchanged
- Performance is identical (verify with a simple timing test)

### Phase 4: Migrate Acoustic to the Framework

**Goal:** Replace `AcousticForwardFunc` with the generic framework.

1. Update `src/deepwave/acoustic.py`:
   - Keep the `acoustic()` function signature unchanged
   - Replace `AcousticForwardFunc` with `GenericForwardFunc` + `AcousticEquation`
   - Keep `acoustic_python()` and the Python update functions
   - Keep `zero_edges_and_interiors()` (acoustic-specific, though simpler than elastic's)

2. Run all tests:
   - `pytest tests/test_acoustic.py`
   - `pytest tests/test_callbacks_acoustic.py`

**Acceptance criteria:**
- All tests pass
- `acoustic.py` is significantly shorter
- The public API is unchanged

### Phase 5: Migrate Elastic to the Framework (Most Complex)

**Goal:** Replace `ElasticForwardFunc` with the generic framework.

1. Update `src/deepwave/elastic.py`:
   - Keep the `elastic()` function signature unchanged
   - Replace `ElasticForwardFunc` with `GenericForwardFunc` + `ElasticEquation`
   - Keep `elastic_python()` and the Python update functions
   - Keep `zero_edges_and_interiors()` (elastic-specific with staggered-grid half-grid masks)

2. Run all tests:
   - `pytest tests/test_elastic.py`
   - `pytest tests/test_callbacks_elastic.py`

**Acceptance criteria:**
- All tests pass
- `elastic.py` is significantly shorter
- The public API is unchanged

### Phase 6: Add Born Kernels for Acoustic and Elastic (C/CUDA + Python)

**Goal:** Add Born propagators for acoustic and elastic, enabling double backward for all propagators.

**This phase requires writing new C/CUDA code.** The Born propagators for acoustic and elastic need new kernels that handle background + scattered wavefields, just as `scalar_born.c` extends `scalar.c`.

1. **Acoustic Born C kernel** (`src/deepwave/acoustic_born.c` and `.cu`):
   - Based on `acoustic.c` but with doubled wavefields (background + scattered)
   - Scatter source term couples background wavefield to scattered wavefield
   - For variable-density acoustic: scatter from perturbations in v and rho
   - Update `backend_utils.py` with new function templates
   - Update `CMakeLists.txt` to compile the new kernels

2. **Elastic Born C kernel** (`src/deepwave/elastic_born.c` and `.cu`):
   - Based on `elastic.c` but with doubled wavefields
   - Scatter from perturbations in lambda, mu, buoyancy
   - More complex due to stress symmetry and dimension-dependent structure
   - Update `backend_utils.py` and `CMakeLists.txt`

3. **Python wrappers:**
   - Create `AcousticBornEquation` (subclass of `AcousticEquation` or `PropagatorEquation`)
   - Create `ElasticBornEquation`
   - Add `acoustic_born()` function and `AcousticBorn` module (following scalar_born's pattern)
   - Add `elastic_born()` function and `ElasticBorn` module

4. **Double backward:**
   - Add `AcousticForwardFunc.backward()` (or use generic framework) that calls `acoustic_born`
   - Add `ElasticForwardFunc.backward()` that calls `elastic_born`

**Acceptance criteria:**
- New Born propagators produce correct linearized forward (verified with finite-difference JVP tests)
- Double backward works for acoustic and elastic (verified with finite-difference HVP tests)
- All existing tests still pass
- New tests in `tests/test_acoustic_born.py`, `tests/test_elastic_born.py`

### Phase 7: Final Cleanup

1. Remove any dead code left from the migration
2. Update docstrings if needed
3. Run the full test suite one final time
4. Verify examples still work: `python examples/example_fwi.py`

---

## Critical Guidance

### Incremental Validation

After each phase, run ALL tests. Do not proceed to the next phase until:
```
pytest tests/ -v
```
passes with 0 failures. If a test fails, stop and fix it before proceeding.

### What to Keep Unchanged

- All public Python API function signatures
- All C/CUDA kernel code
- `backend_utils.py`
- `common.py` (you may add to it, but do not change existing functions)
- `regular_grid.py`
- `staggered_grid.py`
- `wavelets.py`
- `location_interpolation.py`
- `storage_utils.c`/`.cu`
- `CMakeLists.txt`

### What to Move (Not Delete)

When extracting code into equation classes, move the logic but verify the original location still works. For example:
- `prepare_models` from `acoustic.py` → `AcousticEquation.prepare_models()`, and update `acoustic.py` to call it
- `prepare_parameters` from `elastic.py` → `ElasticEquation.prepare_parameters()`, and update `elastic.py` to call it

### Handling ScalarBorn

`scalar_born.py` is used internally by `ScalarBackwardFunc.backward()` for computing Hessian-vector products (double backward). It is NOT used by users directly for JVP/functorch. The refactoring should:

1. Keep `scalar_born.py` as-is initially (it is used by the double backward path)
2. When migrating `scalar.py` to the framework, ensure the `ScalarBackwardFunc.backward()` still calls `scalar_born_func()` correctly
3. Eventually, `scalar_born.py` could also be migrated to use the framework, but this is lower priority

### Handling Elastic's Complexity

Elastic is the hardest because:
- Dimension-dependent wavefield counts (1D: 4, 2D: 9, 3D: 14 + PML variables)
- Dimension-dependent source/receiver types
- `prepare_parameters()` does harmonic averaging of mu
- Source amplitude scaling uses dimension-dependent buoyancy indices

The `ElasticEquation` should have methods like:
- `n_wavefields(self, ndim: int) -> int` returning `{1: 4, 2: 9, 3: 14}[ndim]`
- `n_source_types(self, ndim: int) -> int` returning `ndim + 1`

### Error Handling

If you encounter a situation where the generic framework cannot cleanly handle an edge case:
1. Stop and analyze whether the abstraction needs to be extended
2. Do NOT add `if propagator_name == "elastic": ...` conditionals to the generic framework
3. Instead, add an abstract method to `PropagatorEquation` that the equation can override
4. The generic code should only call equation methods, never check which equation it is

### Ruff and MyPy

Run `ruff check src/` after each phase. The code must pass linting. If you need to add type annotations to the equation classes, follow the existing patterns in `common.py`.

---

## Summary

The core insight is correct: all propagators share the same structure, differing only in counts, model preparation, and grid handling. The strategy pattern (`PropagatorEquation`) captures these differences cleanly. The generic framework handles the shared structure. Each propagator becomes a thin wrapper that:
1. Defines its equation (the strategy)
2. Calls the generic framework
3. Preserves its exact public API

This reduces duplication from ~4000 lines to ~500 lines of equation-specific code, while keeping all existing behavior identical.
