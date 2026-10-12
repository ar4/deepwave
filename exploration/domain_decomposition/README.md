# Domain decomposition — exploration & lessons (Workstream E)

**Status: EXPLORATORY. Not integrated into the `deepwave` package; deferred by the
maintainer pending a decision.** This directory is preserved in git so the lessons
below inform all future decisions about domain decomposition and the propagator
code. The validating prototype (`scalar_decomposed.py`) and its test
(`test_scalar_decomposed.py`) live here, *outside* `src/deepwave/`, so they ship in
no wheel and affect nothing in the package.

This is a historical snapshot. API statements and the measurements below refer
to the prototype at the time of its evaluation. The present state API may need
adaptation before these tests can run. It is unsupported and excluded from
ordinary test collection and distributions.

Historical validation command from the repo root:

```
pytest exploration/domain_decomposition/test_scalar_decomposed.py
```

---

## What was explored

A spatial **slab** domain decomposition for the `scalar` propagator: split the
model into contiguous slabs along one axis, propagate each slab with the ordinary
`deepwave.scalar` propagator, and exchange a thin halo between neighbours every
internal time step, so that the stitched result equals a single monolithic
`scalar()` call. The goal of Workstream E is to break the **single-device
model-size ceiling** (one model domain must currently fit on one device), which is
what caps large 3-D — not shot throughput.

The prototype was built deliberately as the *correctness foundation only*, after an
adversarial design review chose domain decomposition over the other big bets (the
functorch/`setup_context` epic and viscoacoustic-Q) because it is pure-Python,
fully CPU-testable, additive, and touches no C/CUDA.

## The algorithm (overlapping Schwarz)

The single most important mechanical fact, which any future design must respect:

> **The propagator's own finite-difference ghost ring (`fd_pad = accuracy//2`
> cells) is internal and NOT injectable.** It is stripped from the returned
> wavefield and re-created as zeros on input, so it behaves as a rigid `u = 0`
> wall (this is exactly why a `pml_width=0` face reflects). You cannot exchange
> the propagator's own ghost.

So the scheme must be **overlapping**: each slab carries a halo of `fd_pad` *real*
interior cells beyond the cells it owns, with `pml_width=0` on its interior faces.
After each internal step the halo cells (which the slab updated incorrectly against
its own zero ghost) are overwritten with the neighbour's freshly-computed owned
values. Because an owned cell's stencil reaches at most `fd_pad` cells — no further
than the last halo cell — the owned cells are always correct, so the stitch is
exact. `halo = accuracy//2` is exactly sufficient (verified for accuracy 2/4/6/8).

Implementation facts worth reusing:

- **Only `wavefield_0`/`wavefield_m1` halos need exchanging.** The PML memory
  (`psi`/`zeta`) at *owned* cells stays correct because it is updated pointwise
  from the wavefield gradient, which only needs a correct `wavefield_0` halo. (We
  tested also exchanging `psi`/`zeta`; it changes nothing.)
- **State continuation is already a public capability.** `scalar()` accepts
  `psi*_m1`/`zeta*_m1` as inputs and returns them as outputs, and the output
  dataclass field names equal the input kwarg names — so a slab can be stepped one
  step at a time losslessly, carrying a dict of state keyed by
  `scalar_wavefield_names(ndim)`. (The suite's `scalarpropchained` already
  gradchecks time-chaining a single domain; this prototype adds the *spatial seam*,
  which chaining does not cover.)
- **Source/receiver routing** is by `IGNORE_LOCATION` masking: a point is set to
  the sentinel in every slab that does not own it (mask it out), receiver outputs
  are summed across slabs (each receiver active in exactly one slab). A slab that
  owns no active point globally must be passed `None`, or the in-extents check
  crashes on an empty `min()`.
- **CFL / `step_ratio`** must be handled globally: pin one `max_vel` (and
  `pml_freq`, `dt`) across all slabs, upsample the source to the internal `dt`
  **once up front** (frequency-domain resampling is global-in-time — never per
  step), step at the internal `dt` (`step_ratio==1` per call), and downsample
  receivers at the end (`shift=0` matches `scalar` exactly).
- Seed zero `wavefield_0/m1` so `ndim` is inferable even for a source/receiver-free
  slab on step 0; pass `origin=[0]*ndim` to silence the survey-extent warning.

## What was proven (empirically, vs monolithic `scalar()`)

To **machine precision (~1e-15)**, both the forward field + receivers and the
**gradient w.r.t. `v`** (the transpose-halo adjoint — autograd does the
overlap-region gradient summation for free when slabs index slices of one shared
`v` leaf), across: 1-D / 2-D / 3-D; 2 and 3 subdomains; both split axes;
`step_ratio` 1 and >1 (the resampling path); zero and real PML; and a heterogeneous
random model (exercising the global-`max_vel` pinning). See
`test_scalar_decomposed.py`.

## The big lesson: a silent PML-corner instability

An adversarial review found — and the prototype's own tests had **missed** — a
genuine instability:

> **With `n_subdomains >= 3` AND `ndim >= 2` AND the split-dimension PML wider than
> a perpendicular-dimension PML, a spurious mode at the seam/perpendicular-PML
> corner grows ~exponentially (~1.4×/step) from ~1e-12 to >100%.** It is silent
> (invisible at small `nt`) and catastrophic.

- Stable **exactly** when the split-dim PML ≤ every perpendicular PML. Symmetric
  PML, two-slab, and 1-D are always stable.
- **Neither a wider halo NOR exchanging `psi`/`zeta` fixes it** (both tested). It is
  a real overlapping-Schwarz **PML-corner interface/stability** problem, not a
  missing-field bug. A transparent seam interface condition is needed — this is
  research-level (DD + PML corner stability is a known hard problem).
- The prototype **guards** it: `scalar_decomposed` raises `ValueError("Unstable
  configuration ...")` for that regime.
- **Process lesson:** the forward tests were all symmetric-PML + short `nt`, the one
  configuration that never triggers it. Future correctness tests for any
  stencil/boundary scheme should include **asymmetric** parameters and **long**
  runs, where growing modes surface.

## Performance & honest scope (do not overstate)

This prototype is the **algorithm + adjoint correctness foundation, not a scale
solution.** Concretely, as written it:

- runs all slabs **sequentially in one process on one device** (the seeds don't
  even carry `v.device`) — there are no multiple "processors" yet;
- **does not reduce peak memory** — it holds every slab's state simultaneously, and
  the differentiable path retains the full autograd graph over all steps (so
  gradients use *more* memory than the monolithic call, not less);
- is **slower** than monolithic in wall-clock — the dominant cost is the **per-step
  Python front-end re-entry** (one `scalar()` call per slab per internal step
  re-runs survey extraction, PML-profile setup, allocation, the autograd
  `Function`), not the halo copies (a few `fd_pad` planes — negligible);
- **communicates entirely in host-side PyTorch** (`clone()` + `narrow().copy_()`).
  There is **no** C/CUDA, `torch.distributed`, NCCL/MPI, or peer copy.

Realising the actual scale benefit needs further work (see below).

## Implications for future decisions about the code

1. **Domain decomposition can be a pure-Python, additive layer with ZERO backend
   or ABI changes and ZERO effect on non-decomposed runs.** The longevity
   constraint (no torch ops / pybind / struct-ABI; see the backend-design analysis)
   is fully respected — even a future multi-device version communicates at the
   PyTorch level, not in kernels.
2. **The public `scalar()` API is already DD-ready** in the one way that matters:
   lossless single-step state continuation including PML auxiliaries. Keep this
   property (input == output state field names) intact in any refactor.
3. **The blockers to a *shippable* DD feature**, in priority order:
   - **(a) The seam interface condition** that removes the PML-corner restriction.
     This is the headline open problem and is research-level.
   - **(b) Multi-device placement + cross-device halo exchange**
     (`torch.distributed` / `.to()`), plus **per-subdomain checkpointing**, to
     actually drop peak memory — without these there is no scale win.
   - **(c) The per-step Python overhead.** A callback-driven path (one long
     `scalar()` per slab with `callback_frequency=1`, slabs in threads, exchanging
     via the callback's read/writable wavefield views) avoids re-entering the front
     end every step, at the cost of thread/barrier complexity.
4. **Invariants any DD design must hold** (each a silent-wrong-answer trap):
   global `max_vel`/`pml_freq`/`dt` pinning; one up-front source resample; slabs
   avoid diagonal/corner halos but a true **block** decomposition (multiple split
   axes) does not, and re-opens the corner-coupling question the PML instability is
   a first taste of.

## If/when this is revisited

Natural next steps, smallest-first: (1) attempt the seam interface condition on a
2-slab 1-D case with asymmetric PML; (2) generalise to a 2-D/3-D block split on
scalar (corners); (3) the multi-device + checkpointing path for the real memory
win; (4) a callback-threaded driver for performance; (5) extension to
acoustic/elastic (staggered-grid halos). The prototype here is the validated
starting point for all of these.
