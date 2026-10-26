"""``elastic``'s ``lamb``/``mu`` gradients when ``step_ratio > 1``.

**This file does not kill the ``ek_grad_c00_step_ratio`` mutant**, and
is not the test that closes that gap. That mutant drops the ``step_ratio``
factor from ``grad_c00`` in ``elastic_kernels.h``::

    grad_c00[i] += sigmayy[i] * dvydy_store[i] * (DW_DTYPE)step_ratio;
    ->
    grad_c00[i] += sigmayy[i] * dvydy_store[i];

but that statement lives inside ``#ifdef DW_PM``, so it is reachable only via
``surface_method="pm"``. The test that kills it is
``tests/test_elastic_pm.py::test_pm_compiled_gradients_match_eager_at_step_ratio_gt_1``;
building the mutant and running the ordinary elastic path against it is how
that was established.

What this file *is*: the same class of check on the ordinary (non-PM)
``grad_lamb``/``grad_mu`` terms, which carry the same ``step_ratio`` factor
and had no ``step_ratio > 1`` coverage either -- no elastic test combined
``step_ratio > 1`` with ``lamb``/``mu`` requiring a gradient, though the
equivalent scalar, scalar_born and engine ``step_ratio`` mutants were all
killed. It guards the non-PM half of the same hazard.

**What this pins.** The compiled backend's model gradient is a strided
rectangle-rule quadrature of the exact adjoint time integral (it samples the
integrand once per ``step_ratio`` block and rescales by ``step_ratio``); the
eager backend steps every internal step under autograd and is the exact
discrete adjoint. See ``docs/what_deepwave_calculates.rst``. So:

* at ``step_ratio == 1`` the two must agree to machine precision -- both
  formulas reduce to the same sum;
* at ``step_ratio == 2`` they differ only by that documented O(dt) quadrature
  error, which on this problem is about **2%** relative.

Dropping a ``step_ratio`` factor from one of four accumulated terms is not an
O(dt) effect: it changes that term's contribution by a factor of
``step_ratio``, which is a large, structural error in ``grad_lamb``/``grad_mu``
and far outside the band the ``step_ratio == 2`` assertion below allows.

If this test starts failing, that is a kernel-behaviour change worth
investigating, not a reason to loosen the tolerance -- the same standing
instruction ``test_acoustic_vti.py``'s
``test_step_ratio_gt_1_model_gradient_is_the_documented_quadrature`` carries.
"""

import pytest
import torch

import deepwave
from deepwave import vpvsrho_to_lambmubuoyancy

DTYPE = torch.float64
N = 14
NT = 24
DX = 5.0
ACCURACY = 4
PML_WIDTH = 3
PML_FREQ = 15.0
MAX_VEL = 2000.0

# dt=0.001 is at/below the CFL limit here (step_ratio 1); 0.002 forces 2.
DT_BY_STEP_RATIO = {1: 0.001, 2: 0.002}


def _models() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vp = torch.full((N, N), 1500.0, dtype=DTYPE)
    vs = torch.full((N, N), 800.0, dtype=DTYPE)
    rho = torch.full((N, N), 1000.0, dtype=DTYPE)
    return vpvsrho_to_lambmubuoyancy(vp, vs, rho)


def _gradients(dt: float, backend: str) -> tuple[list[torch.Tensor], int]:
    """``(grad_lamb, grad_mu, grad_buoyancy)`` and the resolved ``step_ratio``."""
    lamb, mu, buoyancy = (model.clone().requires_grad_() for model in _models())
    torch.manual_seed(0)
    amplitudes = torch.randn(1, 1, NT, dtype=DTYPE)
    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        DX,
        dt,
        source_amplitudes_y=amplitudes,
        source_locations_y=torch.tensor([[[N // 2, N // 2]]]),
        receiver_locations_y=torch.tensor([[[4, 4]]]),
        pml_width=PML_WIDTH,
        accuracy=ACCURACY,
        pml_freq=PML_FREQ,
        max_vel=MAX_VEL,
        backend=backend,
    )
    out.receiver_amplitudes_y.pow(2).sum().backward()
    gradients = []
    for model in (lamb, mu, buoyancy):
        assert model.grad is not None
        gradients.append(model.grad.detach().clone())
    return gradients, int(out.info.step_ratio)


def test_step_ratio_1_gradients_agree_to_machine_precision() -> None:
    """Control: at ``step_ratio == 1`` compiled and eager are the same sum."""
    compiled, step_ratio = _gradients(DT_BY_STEP_RATIO[1], "compiled")
    eager, _ = _gradients(DT_BY_STEP_RATIO[1], "eager")
    assert step_ratio == 1, f"expected step_ratio 1, got {step_ratio}"
    for name, actual, expected in zip(("lamb", "mu", "buoyancy"), compiled, eager):
        assert expected.norm() > 0, f"{name}: eager gradient is zero; test is vacuous"
        relative = ((actual - expected).norm() / expected.norm()).item()
        assert relative < 1e-12, (
            f"step_ratio=1, {name}: compiled and eager gradients differ by a "
            f"relative {relative:.3e}, but at step_ratio=1 the quadrature is "
            "exact and they must agree to machine precision"
        )


@pytest.mark.parametrize("name_index", [(0, "lamb"), (1, "mu"), (2, "buoyancy")])
def test_step_ratio_2_gradient_is_within_the_documented_quadrature_error(
    name_index: tuple[int, str],
) -> None:
    """``step_ratio == 2`` differs from eager only by the O(dt) quadrature error.

    Measured on this problem: about 2.2e-2 relative for ``lamb`` and ``mu``,
    5.1e-3 for ``buoyancy``. The bound below leaves generous headroom over
    those while staying far under what dropping a ``step_ratio`` factor from
    one of the four accumulated ``lamb``/``mu`` terms would produce.
    """
    index, name = name_index
    compiled, step_ratio = _gradients(DT_BY_STEP_RATIO[2], "compiled")
    eager, _ = _gradients(DT_BY_STEP_RATIO[2], "eager")
    assert step_ratio == 2, f"expected step_ratio 2, got {step_ratio}"
    expected = eager[index]
    assert expected.norm() > 0, f"{name}: eager gradient is zero; test is vacuous"
    relative = ((compiled[index] - expected).norm() / expected.norm()).item()
    assert relative < 6e-2, (
        f"step_ratio=2, {name}: compiled and eager model gradients differ by "
        f"a relative {relative:.3e}, well beyond the documented O(dt) strided-"
        "quadrature error (about 2.2e-2 here). Check that every gradient term "
        "in elastic_kernels.h still carries its step_ratio factor."
    )
