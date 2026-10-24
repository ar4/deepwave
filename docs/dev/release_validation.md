# Deepwave 0.1.0 validation scope

The evidence below describes an earlier candidate, not qualification of the
current source tree. Runtime Python and native sources have since changed;
the affected correctness, performance and platform checks must be repeated
before release. CI receipts must identify the actual tree being promoted.

That earlier local qualification used revision `35dcf3d5fb2a503f5cddccb8a39c53f5d9ee30c1`
with a verified 453-file source manifest SHA-256 of
`ba768051649e206a6107eab65900c03514b3f61543d2a4ca47df64d48514766a`.
Its release preparation changed packaging metadata, CI, credit, research navigation
and documentation; runtime Python and native sources were unchanged at that point.
The repaired sdist had a new archive identity; original distributions and toolchain,
source-member and acceptance receipts are retained privately.

## Retained local evidence

Linux installed-wheel qualification covers all nine supported propagators,
independent analytic, adjoint, Taylor and conservation tests, checkpoint
and imaging gradients, ABI/native storage checks, examples and strict docs,
lint, types and format checks. Recorded fast installed scope was 6,487 passes
and 45 conditional skips, with separately recorded heavy and finite-difference
anchor selections. Python 3.9/PyTorch 2.4 and Python 3.11/PyTorch 2.6 CPU
representatives each passed 1,781 cases; this is CPU compatibility evidence.
GCC/Clang sanitizer evidence has bounded native and ABI scopes, with earlier
unchanged batches explicitly retained. Partial or failed batches are preserved
and are not counted as successful completion.

The qualified Linux wheel used GCC 13.3.1, CUDA compiler 13.3.73 and glibc
2.28; its native and device-image identities are retained with the private
qualification receipts.

CUDA qualification used Python 3.13.5/PyTorch 2.13.0+cu130, RTX A4000 (sm86)
and RTX 4000 Ada (sm89). Final installed GPU integration passed 137 cases with
four conditional skips on each card and four multiple-GPU cases. Memcheck and
Initcheck each covered 138 effective cases per card; forced PTX covered 50
distinct cases per card, and compression Racecheck/Synccheck covered 36 cases
per tool per card, with zero errors in ten verified tool receipts. Device image
binding and source/toolchain identities are retained. This does not qualify the
CUDA dependency floor, every GPU architecture, or execution of the rebuilt
Actions wheels. The maintainer accepts rebuilding identical native inputs
without repeating the GPU campaign.

## Compatibility and performance qualifications

The legacy comparison accounts for 232 cases: 98 affected/new results and 134
retained unchanged results, with unchanged tolerances and accuracy 6 included.
The original automatic-default comparison retained 27 elastic differences.
Conservative inferred `max_vel` can change PML parameters even when the timestep
stays fixed; one air-layer example changes the internal step ratio from 1 to 17.
Using the same explicit conservative bound passes matched-setting parity. Each
version's automatic call agrees bit for bit with its own resolved explicit call.
See [migration notes](../upgrading.rst) for behavioural changes and warnings.

A 24-case portable compiler screen showed no confirmed slowdown; a broader
72-case screen measured ratios 0.337–1.198. A CPU VTI microcase repeated at
1.157×, while 18 representative CPU cases measured 0.930–1.021×. Against
0.0.27, small GPU scalar/acoustic forward calls reproduced 1.394×/1.460× costs,
approximately 1.86/2.72 ms extra. Non-reentrant checkpointing measured about 13%
extra CPU time and 3.42× incremental memory in specific cases. These are measured
qualifications, not universal performance bounds.

## Release repair checks

The repair work adds independent high-precision equation checks for standard
and multiaxial PML, generated CFL/unit-scaling checks, and complete-state adjoint
comparisons. The focused numerical selection passed 1,163 cases on each of
PyTorch 2.4.0 and 2.9.1, with 141 transform cases on each. Both interpreters here
use Python 3.11; the Python 3.9 and 3.13 combinations still require Actions.
The M-PML tests bound exceptional float64 temporaries to 65,536 elements when
preparing float32 profiles. Ordinary profiles stay in the requested dtype.
Measured M-PML setup overhead was 0.6–2.3 ms on the small CPU cases; it does not
add work to the propagation loop.

The regular-grid stencil retains neighbour-minus-centre differences. Across
four small single-threaded scalar cases, with portable GCC and AVX2 builds,
its median runtime ratios against the former stencil were 1.020–1.118 for
forward calls and 1.033–1.137 for forward-plus-gradient calls. Both builds kept
native subnormal flushing enabled. These are measured correctness costs, not a
speed improvement or a GPU performance result. An alternative grouped adjoint
was rejected: Clang fast-math removed its overflow fallback. The retained
constant, quadratic and near-overflow native regressions pass all 64 cases
across GCC 14 and Clang 19, both dtypes and all four stencil orders.

The suppression review corrected the OOM estimate's ignored gradient-sampling
argument, rejected fractional boundary widths, removed obsolete internal
arguments and dead expected-failure machinery, and restored an excluded VTI
model-batching case. Unexpected functorch unwrap failures now propagate.
Unused-argument exemptions are scoped to existing callback/protocol files;
clang-tidy's suspicious-argument check is enabled. Remaining exceptions cover
third-party typing gaps, package-internal access, typing-only coverage and
explicit style/protocol choices. Neither those exceptions nor a clean analyser
run establishes semantic correctness.

The CUDA integer review found products evaluated in 32 bits before assignment
to a 64-bit tensor index, and tensor sizes narrowed before launch-grid division.
These now preserve the full tensor range. CPU thread counts and CUDA device
ordinals retain their existing 64-bit ABI but are validated before conversion
to the APIs' `int` arguments. C99 and C++11 boundary probes pass under GCC and
Clang with undefined-behaviour sanitisation, without allocating large arrays.
All 60 ABI tests pass with the 21 updated CPU entry points compiled separately;
this is not a complete rebuild of the optimised CPU library. The ABI recorder
now also observes scalar Born backward calls, which its previous hook missed.

NVCC 13.3 fully compiled 22 selected CUDA variants, covering every feature
family, all compression dimensions/dtypes and a scalar 3D case, with warnings
treated as errors. Compiled compression entry points reject oversized launches
before accessing data or invoking the GPU runtime. Matched SM75 compilation
of 2D/3D scalar and 2D elastic kernels shows register-count changes from -2 to
+8, with no stack or register spills across the 34 compared kernels. This
does not measure GPU runtime; the integer correction needs GPU timing as well
as numerical qualification. Build and analysis artefacts reside on the internal
drive, and compilation is bounded by memory watchdogs.

The full NVCC front-end sweep passed all 292 configured CUDA variants; the
54 compression/M-PML variants were rechecked after the final cleanup. Clang 23
analysis passed all 16 representative feature families, with an additional
device-mode structural check. It exposed an inherited `false && condition`
workaround that hid unused ordinary-PML bounds in M-PML builds. Those bounds
are now declared only where used. All eight sampled GPU device images are
byte-identical before and after that cleanup. Clang's `-Ofast` deprecation
suppression was also removed in favour of its documented explicit flags;
all 64 native stencil probes pass with compiler warnings treated as errors.

The next GPU qualification will install the actual accepted Linux Actions
wheel in Colab, using tests from the matching source tree. Preserve its checksum,
loaded library identity, driver, GPU and PyTorch versions, and pass/fail/skip
reports. Single-device results and tests requiring two visible GPUs must be
reported separately. Historical GPU receipts above do not qualify these changes.

## External release boundary

Windows 2022 and Apple Silicon macOS build/test evidence remains
pending. The repaired workflow rebuilds the sole sdist and validates all three
platform wheel sets with CPU tests, native smoke and platform storage checks.
Required tests, docs, lint and sanitizer workflows must qualify the same actual
Git tree before promotion. Local collection is not platform execution, and test
plan coverage is not a count of fresh passes.

Future CI links belong in the PR/release evidence and permanent release assets.
Do not modify a tested tree simply to add result URLs before promotion. Follow
[the release procedure](releasing.md) to promote the exact accepted artifacts.
