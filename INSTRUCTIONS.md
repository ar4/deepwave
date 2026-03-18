# Deepwave Refactoring Instructions

## Context & Objective
**Objective:** Refactor the Deepwave library to massively reduce code duplication by implementing a generic propagator framework. The goal is to unify the structure of `scalar`, `acoustic`, and `elastic` propagators. Currently, only `scalar` has a Born variant (used for double backpropagation). The refactoring should enable Born variants (and thus JVP/double-backprop) for `acoustic` and `elastic` as well.

**Constraint:** The public Python API must remain **unchanged**.

**Strategy:**
1.  **Analyze & Prototype:** Understand the C/CUDA interface and feasibility of "Born-aware" generic design.
2.  **Build Framework:** Create a `PropagatorEquation` strategy pattern (to handle model/wavefield complexity) and a `GenericPropagator` (autograd function).
3.  **Refactor Scalar:** Prove the framework with the simplest case (Scalar), integrating its existing Born capabilities.
4.  **Refactor Acoustic/Elastic:** Port the remaining propagators to the framework.
5.  **Add Born Support:** Implement Born kernels/logic for Acoustic and Elastic to complete the unification.

---

## Phase 0: Investigation & Technical Strategy (CRITICAL)
*Before writing core code, you must verify assumptions to prevent architectural dead-ends.*

1.  **Analyze C/CUDA Signatures:**
    *   Examine `src/deepwave/scalar.c`, `acoustic.c`, `elastic.c` and `backend_utils.py`.
    *   **Task:** Map out exactly how arguments differ (order, types, count). The generic framework must handle `scalar` (velocity only) vs `elastic` (20+ fields, dimension-dependent inputs).
    *   **Decision:** Confirm that a "list of names" config is insufficient and a **Strategy Pattern** (e.g., `PropagatorEquation` class) is required to handle argument construction and model preparation.

2.  **Born Implementation Strategy:**
    *   **Task:** Check if existing `acoustic`/`elastic` C-kernels can accept arbitrary source terms for all wavefields.
    *   **Hypothesis:** If they can, "Born" might be implemented via Python-level source injection (calling forward twice: once for background, once for scatter source).
    *   **Reality Check:** If kernels hardcode source types (e.g., only pressure sources, not stress sources), you will likely need to write new `*_born.c` kernels. *Assume new kernels are needed unless proven otherwise.*

3.  **Functorch/vmap Feasibility:**
    *   Deepwave uses `ctypes` to call compiled code. This is notoriously hard to integrate with `torch.func.vmap` / `torch.compile`.
    *   **Directive:** Do **not** attempt to rewrite the build system to `torch.library` or C++ extensions in this refactor. Focus on **Python-level composition** for JVP. If `vmap` is too complex given the C-pointer logic, de-prioritize it. **Code quality and unification are the primary goals.**

---

## Phase 1: The Generic Framework (Design)
*Goal: Create the abstract machinery without modifying existing propagators yet.*

1.  **Create `src/deepwave/prop_framework.py`:**
    *   Define an abstract base class `PropagatorEquation`.
        *   Methods needed: `prepare_models()`, `prepare_wavefields()`, `get_c_kernel_args()`, `get_compute_capability_args()`.
    *   Define `GenericPropagator(torch.autograd.Function)`.
        *   It should accept a `PropagatorEquation` instance.
        *   **Crucial:** Design `forward` and `backward` signatures to accept **optional scattered wavefields/sources** from Day 1. Even if Acoustic/Elastic don't use them yet, the slots must exist to avoid rewriting the framework later.

2.  **Handle "Batch Flattening" (If attempting vmap support):**
    *   The C-kernels expect `[shot, ...]` dimensions. `vmap` adds arbitrary batch dimensions.
    *   The framework must detect extra batch dims, flatten them into the "shot" dimension, ensure contiguous memory, call the kernel, and unflatten.

---

## Phase 2: Refactor Scalar Propagator (The Pilot)
*Goal: Prove the framework works by migrating the simplest propagator.*

1.  **Implement `ScalarEquation` (subclass of `PropagatorEquation`):**
    *   Move logic from `scalar.py`'s `_prepare_models`, `_prepare_wavefields` into this class.
    *   Implement the C-argument generation specific to scalar (velocity, grid, etc.).

2.  **Update `src/deepwave/scalar.py`:**
    *   Replace `ScalarForwardFunc` and `ScalarBackwardFunc` with usage of `GenericPropagator` instantiated with `ScalarEquation`.
    *   **Verify:** Existing tests in `tests/test_scalar.py` **must pass**.
    *   **Performance:** Check that overhead is negligible.

3.  **Integrate Scalar Born:**
    *   The current `scalar.py` handles Born for double-backprop. Ensure `ScalarEquation` can handle the "Born mode" (scattering).
    *   *Self-Correction:* If `ScalarBorn` logic is too different, subclass `ScalarEquation` as `ScalarBornEquation`.

---

## Phase 3: Refactor Acoustic & Elastic (The Heavy Lifting)
*Goal: Port the complex propagators to the framework. This provides the massive code reduction.*

1.  **Refactor Acoustic:**
    *   Implement `AcousticEquation`. Logic from `acoustic.py` (bulk modulus conversion, etc.) moves here.
    *   Update `acoustic.py` to use the framework.
    *   Verify with `tests/test_acoustic.py`.

2.  **Refactor Elastic (Complex):**
    *   Implement `ElasticEquation`.
    *   **Challenge:** Elastic has dimension-dependent logic (2D vs 3D wavefields) and complex model prep (harmonic averaging of `mu`, buoyancy limits).
    *   **Guidance:** Do not simplify the math. Copy the exact logic into the `Equation` class. The "Generic Framework" simply calls `equation.prepare_models()`.
    *   Update `elastic.py`.
    *   Verify with `tests/test_elastic.py`.

---

## Phase 4: Add Born Support to Acoustic & Elastic
*Goal: Achieve feature parity and unification.*

1.  **Develop Kernels (if needed):**
    *   If Phase 0 determined new kernels are needed, create `src/deepwave/acoustic_born.c/cu` and `elastic_born.c/cu`.
    *   These kernels must compute the scattered wavefield `u_sc` driven by the interaction of the background field `u_0` and model perturbation `dm`.
    *   *Warning:* Elastic Born source terms are complex. Ensure physical correctness.

2.  **Update Equations:**
    *   Update `AcousticEquation` and `ElasticEquation` to handle the "Born" pass (loading the Born kernel, passing scattered wavefields).

3.  **Enable JVP:**
    *   Expose `acoustic_born()` and `elastic_born()` Python functions (internal or public).
    *   Implement the JVP logic (Forward Mode AD) for these propagators.

---

## Phase 5: Cleanup & Validation
1.  **Remove Duplication:**
    *   Delete the old `_parse_args`, `_save_ctx` implementations from individual files if they are now fully handled by `prop_framework.py`.
2.  **Final Testing:**
    *   Run the full test suite.
    *   Verify that `example_*.py` scripts still run correctly (e.g., `example_fwi.py`).

## Guidelines for AI Execution
*   **Incremental Validation:** Do not write all code at once. Write `prop_framework.py`, then test it. Write `ScalarEquation`, then test `scalar.py`.
*   **Context Efficiency:** You do not need to read all C files to refactor the Python layer, but you *do* need to read `backend_utils.py` to understand the C interface.
*   **Safety:** Do not delete old code until the new implementation is verified. Comment it out or rename it during transition.
*   **Communication:** If you hit a blocker (e.g., "Elastic structure doesn't fit the framework"), stop and propose a revision to the `PropagatorEquation` design rather than hacking a workaround.
