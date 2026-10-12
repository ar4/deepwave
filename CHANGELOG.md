# Changelog

All notable changes to Deepwave are documented in this file.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Versioning policy

Deepwave 0.1.0 labels its public interfaces stable, experimental, provisional
or deprecated. Patch releases preserve stable calls apart from documented
correctness or security fixes. Stable deprecations require a replacement,
warning, migration example and at least two subsequent minor releases of
overlap; after 1.0, removal also requires a major release. Provisional
interfaces may evolve in a minor release with explicit notes and a deprecation
path where feasible. Scientific meaning must not change silently. The
[stability policy](docs/stability.rst) names each tier and its detailed rules.

Aliases deprecated in 0.1.0 remain available through 0.2 and 0.3, with
removal no earlier than 0.4. For a fixed propagator and dimensionality, the
current tuple layout follows the stable compatibility policy; read receivers
by name and continue with `initial_state=out`. If you pin Deepwave in a
downstream project, consult this changelog before upgrading, especially for
changes to gradients or boundary conditions.

## [0.1.0]

The first release since `v0.0.27`, which shipped four propagators (`scalar`,
`scalar_born`, `acoustic`, `elastic`). This one ships nine, on one shared
calling convention and an explicit compatibility policy.

**Read `docs/upgrading.rst` before upgrading.** It lists documented behaviour
differences from `v0.0.27`, whether each raises, warns or changes silently,
and migration examples for common calls. Some old positive output indices can
name a different field without a warning.

### Highlights

- **Heterogeneous CFL safety.** The velocity/stress families include density
  and constitutive weighting in timestep bounds. Higher-order stencils at
  sharp density interfaces can require more substeps than pointwise
  wavespeeds suggest; an explicit `max_vel` must cover the conservative bound.
- **Automatic segmented CFL resolution.** Checkpointed and ordinary segments
  resolve an omitted speed from the whole propagator setup before resampling,
  preserving receivers and gradients at sharp heterogeneous interfaces.
- **Five new propagators**, each validated against an analytic solution and
  against the propagator it reduces to or linearises.
- **Explicit compatibility policy.** Stable interfaces follow documented
  deprecation rules; provisional interfaces can evolve in minor releases with
  migration notes. See [stability](docs/stability.rst) for the tiers and
  numerical validation criteria.
- **Scoped numerical compatibility checks.** The v0.0.27 harness compares
  selected calls and each version's separate float64 reference, accounting for
  documented intentional changes. These comparisons do not promise bitwise
  equality or validate every physical configuration.
- **One convention everywhere:** every shared option is accepted on every
  propagator at its documented default, every receiver output has the same
  rank, and Deepwave validation failures use the `deepwave.DeepwaveError`
  hierarchy. The documentation covers stability, upgrading, capabilities,
  accuracy, boundaries, units and discretisation.

### Release scope

The unreleased `elastic_tti` and `viscoelastic` prototypes are deferred and are
not part of the installed package. Their implementations and experiments are
preserved under `exploration/deferred_propagators/`. Tilted elasticity needs an
established, independently validated heterogeneous discretisation; attenuating
elasticity needs conventional plane-strain/3-D and quality-factor contracts.

### Scope and architecture review

- Keep the sampled model-gradient default, including CFL substepping; its
  wavefield-history savings and band-limited correlation rationale remain
  intentional. Sampling and compression still have documented accuracy limits.
- Remove the unreleased `Storage.budget` field and its recursive segmented
  preflight. Use `plan` or `estimate_memory` to inspect memory estimates.
  Omitted `storage` now uses ordinary `None`; explicit modes still work.
- Keep auxiliary-state continuation stable while treating internal PML keys
  and prepared-model callback names as provisional diagnostics, so they do not
  freeze a numerical implementation.
- Apply illumination preconditioning to the completed PyTorch gradient in the
  example. Internal backward callbacks expose partial accumulators; forward
  callback `writable` now reflects the prohibition on differentiable mutations.
- Correct PML damping to use each face's physical width. An unrelated wider
  face no longer weakens a narrower face. Equal physical widths retain their
  profiles; asymmetric widths and anisotropic grid spacing can change results.
  The continuous reflection target is not a discrete reflection guarantee.

### Behaviour changes

- Resampling always removes an even-length Nyquist bin, including when a
  positive frequency taper is too small to cover any bins. `downsample` now
  rejects time lengths not divisible by `step_ratio`; pad or trim explicitly.
- Viscoacoustic's scalar source is `pressure_rate` and
  `source_amplitudes_pressure_rate`, injected directly as pressure per time.
  It no longer uses the misleading volume-rate name and has no volume-rate
  alias.

`docs/upgrading.rst` has every entry below with its exact message and one-line
fix. Each raises unless marked otherwise.

- Every argument after `dt` is keyword-only, and after `grid_spacing` on the
  `Module` constructors; `Hicks` after `halfwidth`.
- The returned tuple is shorter: the absorbing boundary's auxiliary fields are
  one `auxiliary_state` element, not one each. Their 42 *input* keyword
  arguments are gone with no alias; pass `initial_state=out`. They are the one
  thing `v0.0.27` shipped that `0.1.0` removes outright.
- `scalar_born`'s second positional argument is `grid_spacing`, and its
  scattering model the keyword `dv`.
- `ricker`'s `nt` and `dt` are required again; `cfl_condition_n`'s
  `max_abs_vel`/`c_max` are `max_vel`/`courant_number`; `CallbackState`
  requires `direction` and `n_steps`.
- A non-finite, empty, non-`Tensor` or unsupported-dtype model is rejected, and
  `model_gradient_sampling_interval` must divide `nt`.
- `storage_mode="none"` with a model requiring a gradient raises, as does a
  shared option at a non-default value the propagator cannot act on, and a
  diverged run (`check_output_finite=False` opts out).
- A compiled CUDA call must run on the default stream, and `max_vel` is
  required when `vmap` batches a model the maximum wave speed is read from.
- Some failures raise a `ValueError` subclass where `v0.0.27` raised
  `RuntimeError`; catch `deepwave.DeepwaveError`, or the builtin base.
- **Warns:** `pml_freq` is inferred from the source spectrum when omitted
  instead of defaulting to `25.0`; `pml_freq=25.0` restores it and
  `out.info.pml_freq` reports what ran. A Ricker wavelet infers its own centre
  frequency, so a 25 Hz Ricker restores the old frequency setting; other
  numerical changes can still affect its result;
  away from that scale the old default was wrong, an ultrasound call at 3 MHz
  getting a PML five orders of magnitude off.
- **Warns:** `scalar_born` extends `dv` into the absorbing layer and halo, as
  every other model is extended, aligning its edge perturbations with the
  linearisation of `scalar`. Only a `dv` non-zero in the outermost cells of the
  simulated region changes, by up to 85% relative on the scattered receivers.
- **Warns:** an off-node coordinate is interpolated, not truncated to the node
  below it; `locations.floor().long()` reproduces the old placement.
- **Warns:** a `Hicks` window's cells are computed exactly, fixing a `float32`
  truncation that put one stencil point a whole cell from where it belonged for
  **6.2% of off-node coordinates** in `[1, 50]`. The weights are unchanged bit
  for bit, and now follow the default dtype rather than a hard-coded `float32`.
  See tee2s’s report and proposed fix in [PR #122](https://github.com/ar4/deepwave/pull/122).
- **Warns:** `storage_path` defaults to the system temporary directory, saying
  once how much is about to be written there; the `step_ratio` resolution
  warning fires earlier.
- **Silent:** `PropagatorOutput` is a `tuple` subclass, not a plain `tuple`; an
  unrequested receiver channel is `[n_shots, 0, nt]`, not a one-dimensional
  `(0,)` Tensor, every receiver output being three-dimensional; and the
  compiled CUDA kernels ship a curated architecture list, older GPUs using PTX.

### Deprecated

Every name deprecated in `0.1.0` stays available through `0.3.0`, warns with
`deepwave.DeepwaveFutureWarning` naming its replacement, and is scheduled for
removal no earlier than `0.4.0`. Later stable deprecations require a
replacement, warning, migration example and two subsequent minor releases
of overlap. These are the current families:

- `python_backend=` → `backend=`, taking `"auto"`, `"compiled"`, `"eager"` or
  `"torch_compile"`. Deepwave no longer calls `torch.jit`, so
  `python_backend="jit"` runs the eager backend, moving results only by
  rounding.
- `scalar_born`'s `scatter=`, `.scatter`, `scatter_requires_grad=` and its
  `scatter` state-dict and callback names → `dv`; the bare
  `receiver_locations=` → `sc_receiver_locations=`, with the scattered data at
  `out.sc_receiver_amplitudes` (`out.receiver_amplitudes` is a permanent
  alias). `wavelets.ricker(length=)` → `nt=`.
- `"p"` as a *source* channel name, on the nine propagators with a scalar
  source channel: it named two quantities differing by roughly the bulk
  modulus, a volume-injection rate on the acoustic family and a pressure rate
  elsewhere. Use `"volume_rate"` or `"pressure_rate"`; `"p"` is unchanged as a
  *receiver* name.
- The five module spellings `v0.0.27` exposed — `deepwave.common`,
  `location_interpolation`, `backend_utils`, `regular_grid`, `staggered_grid` —
  are private, each kept as a shim that warns per attribute access. The grid
  and resampling helpers live in `deepwave.discretisation`; `deepwave.Hicks` is
  the canonical interpolator. The released `storage_compression` option is
  retained and is lossy when enabled.

### Added

**Propagators**, each with an `nn.Module` form under the CamelCase spelling of
its name: `viscoacoustic` (variable-density acoustic, standard-linear-solid `Q`
at `q_freq`, reducing to `acoustic` as `Q → ∞`); `elastic_vti` (`c11`, `c13`, `c33`,
`c55`, and `c66` in 3-D, reducing exactly to `elastic` when isotropic);
`acoustic_vti`, the Duveneck et
al. (2008) pseudo-acoustic VTI equation; and `acoustic_born` and
`elastic_born`, through which `acoustic` and `elastic` gain **forward-mode and
second-order AD** on both backends. Compiled double backward, including mixed source/model derivatives, yields a
true Hessian-vector product when the effective gradient sampling stride is one;
larger strides are rejected because the sampled first-gradient map can have a
nonsymmetric Jacobian.

**Surveys and returns.** `sources=`, `receivers=` and `bg_receivers=` take
mappings keyed by the quantity injected or recorded (`"u"`, `"p"`, `"z"`,
`"y"`, `"x"`, `"volume_rate"`, `"pressure_rate"`, `"sigma_v"`, `"sigma_h"`),
read back through `out.receivers`, plus boolean `source_mask`/`receiver_mask`;
the flat keywords remain supported. `PropagatorOutput` is attribute-accessible,
with `names`, `as_dict()`, `final_state` and `detach()`, and the
absorbing-boundary and attenuation memory variables are carried in one opaque
`auxiliary_state` that records the run producing it and refuses an incompatible
reuse. `initial_state=out` carries the physical and PML state into the next
compatible segment, inheriting the `pml_freq` and `max_vel` the first call
resolved. Both are pytree nodes. `nt`
and source amplitudes may be given together.

**Boundaries.** `boundary=` takes `"pml"`, `"free"`, `"rigid"` or `"periodic"`
per face, refilling the guard ring with the appropriate odd/even image every
step. Supported configurations have analytic and conservation checks; their
limits and known growth cases are documented in `docs/boundaries.rst`. An
unsupported combination raises
`DeepwaveNotImplementedError` during argument validation. `scalar` and
`scalar_born` take all four codes on every face and `"periodic"` is theirs
alone; `docs/boundaries.rst` tabulates what the other seven accept, and
accepting more later is additive. An elastic free surface is the vacuum layer,
which also handles topography; `elastic` adds the Cao and Chen (2018)
`surface_method="pm"` and `mpml_fraction` multiaxial PML damping.

**Storage and segmentation.** `checkpoint_segments=k` runs the time axis as `k`
checkpointed segments. `deepwave.Storage` carries the storage policy as `storage=`;
`plan` and `estimate_memory` provide estimates rather than enforce a memory budget.
`deepwave.checkpointing` adds `run_segments` and `snapshots`. Peak memory
depends on state size, storage policy and the number of segments.

**Helper modules (provisional).** `deepwave.wavefields` (volumetric sources,
running-DFT reductions, boundary record/reinject), `misfits`, `das`,
`transducers`, `time_dispersion` (the Koene-Robertsson correction),
`anisotropy`, `discretisation` and `diagnostics`. `tone_burst`,
`gaussian_pulse` and `ormsby` join `wavelets.ricker`; `deepwave.Hicks` covers
1-D/2-D/3-D, is staggered-grid aware, and can differentiate positions.

**Introspection.** `deepwave.plan(...)` validates and resolves a propagator
configuration before time-proportional allocation, and returns its `RunInfo`
and memory estimate. `RunInfo` (also
`out.info`) reports the whole numerical configuration that ran, from the
resolved backend and `pml_freq` to the storage policy and field names.
`deepwave.spec`, `introspection`, `estimate_memory`, `boundaries()` and the
generated `docs/capabilities.rst` matrix cover the rest.

**Errors and warnings.** `DeepwaveError` and its
`Value`/`Type`/`Runtime`/`NotImplemented`/`FloatingPoint`/`OS` subclasses each
mix in the builtin type the failure already raised, so existing `except`
clauses keep working for the corresponding errors; `DeepwaveWarning`,
`DeepwaveFutureWarning`,
`MigrationWarning`, `PmlFreqInferenceWarning` and `ZeroWidthFaceWarning` have
distinct filter categories. Every public module defines `__all__`. Repeated
run reproducibility depends on a fixed build, device, configuration and
thread count; see `docs/what_deepwave_calculates.rst`.

### Fixed

Thanks to barkure and tee2s for reporting problems and providing clear
reproducers and proposed fixes. Their reports are acknowledged here; the PR
implementations were not merged.

These were defects in `v0.0.27`. barkure’s report in
[PR #121](https://github.com/ar4/deepwave/pull/121) covers callback mapping and
input validation (it excludes #119). Fixes were implemented against the new
engine: `backward_callback`'s wavefield dict silently dropped the last
wavefield name; eager `elastic`'s `forward_callback` dict misnamed several
shear fields, so a callback could receive a different field's data;
`cfl_condition_n` let `dt=0` reach a division; an all-sentinel location tensor
crashed validation.

- **Silent wrong gradient (CUDA).** With a `backward_callback` and
  `callback_frequency < nt`, a model shared across shots and `n_shots > 1`, the
  per-shot gradient scratch was zeroed only once at allocation, so every later
  chunk re-added each earlier chunk's partial sum and model gradients came back
  inflated by 21x to 580x. Fixed on every compiled propagator.
- **Crash at accuracy 6/8.** A typo in the eager staggered-grid `diffz1`
  literals crashed 3-D elastic runs at those orders.
- **`NaN` output.** `pml_freq=0` made the PML `b` profile `0/0` across the
  interior, so every output came back `NaN`. It now raises. A full disk could
  likewise truncate a `storage_mode="disk"` snapshot, segfault or leak file
  handles; it now raises, and the temporary directory's lifetime is documented.
- **Backend disagreement.** An empty receiver channel and `backward_callback`'s
  schedule behaved differently on the two backends; all backends now agree.
- **Uninformative diagnostics.** The resolution warnings blamed a line inside
  Deepwave rather than your own call, which also made
  `warnings.filterwarnings(..., module=...)` useless; and a CUDA out-of-memory
  error names the levers that reduce the requirement.

Corrections in the new helpers and checkpoint paths include:

- Segmented helpers apply an explicit `nt` to the whole source record before
  splitting it, rather than padding or truncating every segment separately.
- CPU history decompression preserves finite values near the floating-point
  maximum instead of overflowing through a rounded quantization scale.
- Elastic VTI CPU backward calls safely handle absent shear-modulus gradients,
  including batched models and multiaxial PML.
- Born model-preparation gradients survive PyTorch checkpoint recomputation,
  including on the PyTorch 2.4 dependency floor.
- Extreme finite frequency and time units no longer cause spurious overflow or
  underflow in resolved wavelet samples. Empty time-dispersion inputs retain
  their autograd connection, and finite time-unit changes preserve the sampled
  time-dispersion map.

### Performance

- **Denormal arithmetic.** The compiled CPU kernels set flush-to-zero and
  denormals-are-zero per worker thread inside a propagator call and restore the
  previous mode afterwards, so other code on the same threads is unaffected.
  Decaying PML tails reach that range, and avoiding the slow arithmetic is
  worth 20-70% on many mid-sized cases (`docs/accuracy.rst`).
- **Intra-shot parallelism.** `scalar`, `scalar_born`, `acoustic` and `elastic`
  now spread a *single* shot's interior stencil across all CPU threads;
  previously only the shot loop was threaded, so a one-shot job used one core.
- **CUDA occupancy.** Several 3-D stencil launchers use a 32x4x4 thread block
  at accuracy 4 and above instead of 32x4x1, lifting kernels pinned at the
  255-register cap: `scalar` 1.15-1.32x faster forward and 1.04-1.71x backward,
  `scalar_born` 4-27%, `elastic`/`elastic_vti` forward 8-20%. Launchers it
  measured slower keep the narrow block.
- **Memory and compilation.** The PML memory variables threaded between
  checkpointed segments are stored as per-axis boundary slabs, bitwise exactly;
  an unbuilt compiled combination fails before the wavefield history is
  allocated; and `backend="torch_compile"` uses PyTorch's default compilation cache
  and limits without changing process-wide compiler settings. In local comparisons with v0.0.27, a 64-shot
  scalar forward run on 100-by-100 CPU grids was 16% slower at 24 threads;
  forward plus backward was 18% faster in the same setting. Float64 elastic
  GPU forward runs were 6-9% slower, and a small float32 elastic GPU backward
  case was 11% slower. A 3-D acoustic GPU forward case used 17 MB more memory.
  These are measured trade-offs for particular workloads, not universal speed
  or memory guarantees.

### Packaging

- Clang sanitizer builds use `-O0` for 3-D CPU objects to bound compilation
  memory, retaining ASan, UBSan and debug information; other objects use `-O1`.
- Sanitizer CI exercises bounded native paths and ABI regressions; normal and
  physics CI retain scientific checks. Tests reset Dynamo between cases to
  prevent earlier shape variants from forcing compiled references into eager
  fallback, while preserving PyTorch's configured compiler limits.
- Requires `torch>=2.4`, the oldest locally verified version for compiled
  `torch.func.jvp`, and Python 3.9 or later. PyTorch 2.0/2.1 lack the
  functorch stack-clear helper needed by that path; the declared floor was
  raised from 2.0 after installed-wheel checks on Python 3.10.
  Wheel workflows target Linux, Windows and macOS. A `deepwave[examples]` extra
  names the third-party packages the runnable documentation examples import
  (`matplotlib`, `numpy`, `scipy`, `torchaudio`, `torchvision`, `geomloss`);
  Deepwave itself still depends only on PyTorch.
- `deepwave.__version__` is `0.1.0`, single-sourced from
  `src/deepwave/_version.py`, which `pyproject.toml` and the documentation both
  read. `v0.0.27` had no such top-level attribute, so code supporting both
  should use `getattr(deepwave, "__version__", None)` with
  `importlib.metadata.version("deepwave")` as the fallback.
- Curating the embedded CUDA architecture list and removing an unused kernel
  axis reduced the historical wheel size. `DW_FAST_MATH` (default `ON`, as `v0.0.27` effectively
  was) can be set `OFF` for a precise `-O3` build, at a measured median cost of
  25.6% on a CPU sweep and 7.7% on the GPU benchmark.
- The source distribution is built from an explicit allowlist: the package
  sources, the build files and the project metadata, and nothing else.

## Earlier releases

Versions v0.0.26 and earlier predate this changelog. See the
[GitHub releases](https://github.com/ar4/deepwave/releases) and commit
history for their history.
