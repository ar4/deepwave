"""Negative controls for the public v0.0.27 comparison harness."""

import copy
import math
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import benchmarks.v0027_parity as parity
import deepwave


def _case(*, change: bool = False) -> parity.Case:
    return parity.Case(
        name="synthetic_parity_control",
        block="kernel",
        prop="scalar",
        ndim=2,
        accuracy=4,
        dtype="float64",
        device="cpu",
        locations="off_node" if change else "integer",
    )


def _record(case: parity.Case) -> dict[str, object]:
    return {
        "quantities": {
            key: {
                "diff": 0.0,
                "current_rounding": 0.0,
                "v027_rounding": 0.0,
                "reference_maxabs": 1.0,
                "v027_spread": 0.0,
            }
            for key in parity._expected_quantity_keys(case)
        },
        "warnings_current": ["off-node interpolation window changed"],
    }


def test_genuine_algorithmic_discrepancy_fails() -> None:
    """An unexplained receiver change fails the rounding-level clause."""
    case = _case()
    record = _record(case)
    record["quantities"]["recv:receiver_amplitudes"]["diff"] = 1e-5
    assert parity.classify(case, record)["status"] == "unexplained"


def test_nonfinite_comparison_fails() -> None:
    """A NaN cannot silently pass a comparison with a numeric tolerance."""
    case = _case()
    record = _record(case)
    record["quantities"]["state:wavefield_0"]["diff"] = math.nan
    assert parity.classify(case, record)["status"] == "non-finite"


def test_missing_quantity_fails_even_if_both_versions_omit_it() -> None:
    """Every expected output and gradient must be present."""
    case = _case()
    record = _record(case)
    del record["quantities"]["grad:source:u"]
    verdict = parity.classify(case, record)
    assert verdict["status"] == "error"
    assert "grad:source:u" in verdict["detail"]


def test_wrong_gradient_is_not_excused_by_an_intentional_change() -> None:
    """The intentional-change band cannot hide a large gradient error."""
    case = _case(change=True)
    record = _record(case)
    record["quantities"]["recv:receiver_amplitudes"]["diff"] = 1e-3
    record["quantities"]["grad:model:v"]["diff"] = 4.0
    verdict = parity.classify(case, record)
    assert verdict["status"] == "out-of-band"
    assert "grad:model:v" in verdict["detail"]


def test_born_edge_change_cannot_excuse_background_data() -> None:
    """A dv-padding change cannot alter the Born background forward field."""
    case = parity.Case(
        name="synthetic_born_edge_control",
        block="kernel",
        prop="scalar_born",
        ndim=2,
        accuracy=4,
        dtype="float64",
        device="cpu",
    )
    record = _record(case)
    record["warnings_current"] = ["dv outermost cells changed"]
    record["quantities"]["recv:sc_receiver_amplitudes"]["diff"] = 1e-3
    record["quantities"]["recv:bg_receiver_amplitudes"]["diff"] = 1e-3
    verdict = parity.classify(case, record)
    assert verdict["status"] == "out-of-band"
    assert "recv:bg_receiver_amplitudes" in verdict["unrelated_quantities"]


def test_missing_measurement_field_fails_as_malformed() -> None:
    """A malformed record must fail before any numeric comparison."""
    case = _case()
    record = copy.deepcopy(_record(case))
    del record["quantities"]["grad:model:v"]["diff"]
    assert parity.classify(case, record)["status"] == "error"


def test_kernel_design_includes_accuracy_six() -> None:
    """Every legacy-supported kernel point includes its accuracy-6 variant."""
    kernels = [case for case in parity.build_design() if case.block == "kernel"]
    points = {(c.prop, c.ndim, c.dtype, c.device) for c in kernels}
    assert len(kernels) == 4 * len(points)
    for point in points:
        assert {
            c.accuracy for c in kernels if (c.prop, c.ndim, c.dtype, c.device) == point
        } == {2, 4, 6, 8}


def test_elastic_bound_is_shared_across_encoded_precisions(monkeypatch, tmp_path):
    """Workers and own-float64 references receive the parent's single bound."""
    seen = []

    def fake_plan(fn, lamb, mu, buoyancy, *args, **kwargs):
        seen.append(lamb.dtype)
        bound = 3000.0 if lamb.dtype == torch.float32 else 3001.0
        return SimpleNamespace(info=SimpleNamespace(max_vel=bound))

    monkeypatch.setattr(deepwave, "plan", fake_plan)
    case = replace(_case(), prop="elastic", dtype="float32")
    resolved = parity._resolve_elastic_bounds([case], tmp_path)[0]
    assert seen == [torch.float32, torch.float64]
    assert resolved.max_vel == 3001.0 * 1.000001
    jobs = parity._jobs_for([resolved])
    assert {job["case"]["max_vel"] for role in jobs.values() for job in role} == {
        resolved.max_vel
    }
    for run_dtype in (torch.float32, torch.float64):
        call = parity._build_call(
            resolved, torch.float32, "cpu", tmp_path, run_dtype=run_dtype
        )
        assert call.kwargs["max_vel"] == resolved.max_vel
    assert parity._resolve_elastic_bounds([resolved, _case()], tmp_path) == [
        resolved,
        _case(),
    ]


def test_capture_exception_preserves_evidence(monkeypatch, tmp_path):
    """An interrupted worker must leave its diagnostic tensors available."""
    monkeypatch.setattr(
        parity,
        "_interpreter_facts",
        lambda _: {"deepwave": "0.0.27", "cuda": False, "file": "reference"},
    )

    def interrupted_dispatch(python, jobs, scratch, device, role):
        (scratch / "diagnostic.pt").write_bytes(b"partial diagnostic")
        raise RuntimeError("injected worker interruption")

    monkeypatch.setattr(parity, "_dispatch", interrupted_dispatch)
    scratch = tmp_path / "captures"
    args = SimpleNamespace(
        reference_python=sys.executable,
        repeats=1,
        quick=True,
        only="^kernel_scalar_2d_a4_f64_cpu$",
        no_cuda=True,
        scratch=str(scratch),
        cuda_device="cuda:0",
        out=str(tmp_path / "result.json"),
        keep_tensors=False,
    )
    with pytest.raises(RuntimeError, match="injected worker interruption"):
        parity.cmd_run(args)
    assert (
        scratch / "group-000" / "diagnostic.pt"
    ).read_bytes() == b"partial diagnostic"
