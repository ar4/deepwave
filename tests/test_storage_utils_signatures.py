"""Signature-consistency guard for storage_utils.h's snapshot save/load pairs.

``save_snapshot_cpu``/``save_snapshot_gpu`` and ``load_snapshot_cpu``/
``load_snapshot_gpu`` form two symmetric save/load pairs used by every
propagator's checkpointing path. Each declares a ``step_idx`` parameter that
is multiplied by a byte count to compute a file/device offset -- this must be
a 64-bit type everywhere, since a 32-bit ``step_idx`` silently narrows the
caller's ``int64_t`` step index and can wrap into a wrong seek offset for a
long enough run (corrupting checkpointed snapshots without any other
symptom). Directly exercising that overflow would require an on-the-order-of
2**31-timestep simulation, which is not practical to run as a test, so this
instead pins the source-level invariant that broke last time: all four
functions must declare ``step_idx`` as ``int64_t``.
"""

import re
from pathlib import Path

_HEADER = (
    Path(__file__).resolve().parent.parent / "src" / "deepwave" / "storage_utils.h"
)

_SNAPSHOT_FUNCS = [
    "save_snapshot_cpu",
    "save_snapshot_gpu",
    "load_snapshot_cpu",
    "load_snapshot_gpu",
]


def test_step_idx_is_int64_t_in_every_snapshot_function():
    """Regression guard: all four snapshot functions must declare int64_t."""
    text = _HEADER.read_text()
    for name in _SNAPSHOT_FUNCS:
        # Match the FUNCTION DECLARATION specifically (a return-type keyword
        # immediately before STORAGE_FUNC(name) at the start of a line), not
        # any macro-body call site that also happens to reference
        # STORAGE_FUNC(name) -- e.g. SETUP_STORE_LOAD's own body calls
        # STORAGE_FUNC(load_snapshot_cpu)(...) as a plain invocation (no
        # return type, no declared step_idx parameter type) earlier in the
        # file than that function's real declaration.
        m = re.search(
            rf"^\s*(?:void|int)\s+STORAGE_FUNC\({name}\)\(.*?\bstep_idx\b",
            text,
            re.DOTALL | re.MULTILINE,
        )
        assert m is not None, f"could not locate step_idx parameter for {name}"
        # The declared type is whatever precedes step_idx after the last comma.
        param_text = m.group(0).rsplit(",", 1)[-1]
        assert "int64_t" in param_text, (
            f"{name}'s step_idx parameter is not declared int64_t "
            f"(got {param_text.strip()!r}); this silently narrows the "
            f"caller's int64_t step index and can produce a wrong file/"
            f"device seek offset for long-running checkpointed simulations."
        )
