## What

<!-- What does this PR change? -->

## Why

<!-- What problem does this solve, or what does it enable? Link an issue if there is one. -->

## Tests run

<!-- What did you run locally to validate this, and what were the results?
     e.g. `pytest -m "not slow"`, or the full suite including gradcheck. -->

- [ ] `pytest tests/` (or the relevant subset) passes locally
- [ ] `ruff format src/deepwave/ && ruff check --fix src/deepwave/ && mypy --strict src/` is clean

## Gradient-path changes

<!-- If this PR touches anything on the path from inputs to gradients (kernel
     math, an autograd Function, checkpointing/storage logic, boundary
     handling), confirm the full gradcheck family was run, not just the fast
     `-m "not slow"` subset. Delete this section if not applicable. -->

- [ ] Not applicable — no gradient-path code touched
- [ ] Applicable — ran the full suite including the `slow` (gradcheck) tests

## `.c`/`.cu` lockstep

<!-- If this PR changes a compiled kernel, confirm the CPU (.c) and CUDA (.cu)
     files were updated together, and briefly say how you reasoned about CUDA
     correctness given CI here cannot execute CUDA. Delete this section if not
     applicable. -->

- [ ] Not applicable — no kernel (`.c`/`.cu`) files touched
- [ ] Applicable — `.c` and `.cu` changed together; ABI reference
      (`tests/abi_reference.json`) updated if the signature changed
