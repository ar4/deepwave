# Deepwave 0.1.0 validation scope

Local qualification used revision `35dcf3d5fb2a503f5cddccb8a39c53f5d9ee30c1`
with a verified 453-file source manifest SHA-256 of
`ba768051649e206a6107eab65900c03514b3f61543d2a4ca47df64d48514766a`.
Release preparation changes packaging metadata, CI, credit, research navigation
and documentation. Runtime Python and native sources are unchanged. The repaired
sdist has a new archive identity; original distributions and detailed toolchain,
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

## External release boundary

Windows 2022, Intel macOS and Apple Silicon macOS build/test evidence remains
pending. The repaired workflow rebuilds the sole sdist and validates all four
platform wheel sets with CPU tests, native smoke and platform storage checks.
Required tests, docs, lint and sanitizer workflows must qualify the same actual
Git tree before promotion. Local collection is not platform execution, and test
plan coverage is not a count of fresh passes.

Future CI links belong in the PR/release evidence and permanent release assets.
Do not modify a tested tree simply to add result URLs before promotion. Follow
[the release procedure](releasing.md) to promote the exact accepted artifacts.
