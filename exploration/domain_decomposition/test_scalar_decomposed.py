"""Tests for the exploratory slab domain-decomposition prototype (Workstream E).

This is an EXPLORATION, not part of the installed deepwave package (see the
directory README.md). The slab domain decomposition must reproduce a monolithic
:func:`scalar` call to machine precision — both the final wavefield and the
receiver amplitudes — and its gradient (the transpose-halo adjoint) must match
too. These are external-reference checks against the trusted single-domain
propagator across dimensions, subdomain counts, split axes, time-step ratios, PML
widths, and heterogeneous models.

Run from the repo root with::

    pytest exploration/domain_decomposition/test_scalar_decomposed.py
"""

import os
import sys
from typing import List

import pytest
import torch

from deepwave import scalar
from deepwave.wavelets import ricker

# The prototype lives alongside this test, outside the installed package.
sys.path.insert(0, os.path.dirname(__file__))
from scalar_decomposed import scalar_decomposed  # noqa: E402

C = 1500.0
FREQ = 25.0
DX = 5.0


def _crop_interior(wavefield: torch.Tensor, pml: List[int], ndim: int) -> torch.Tensor:
    """Strip the per-face PML padding from a monolithic wavefield."""
    out = wavefield
    for dim in range(ndim):
        out = out.narrow(
            dim + 1, pml[2 * dim], out.shape[dim + 1] - pml[2 * dim] - pml[2 * dim + 1]
        )
    return out


def _setup(ndim, n_split, split_dim, courant, pml_w, dtype=torch.double):
    """Build a small homogeneous problem with a source and two receivers."""
    dt = courant * DX / C
    shape = [16] * ndim
    shape[split_dim] = n_split
    v = torch.full(shape, C, dtype=dtype)
    nt = 70
    amp = ricker(FREQ, nt, dt, 1.5 / FREQ, dtype=dtype).reshape(1, 1, -1)
    src = [s // 2 for s in shape]
    src[split_dim] = n_split // 3
    rec1 = [s // 2 for s in shape]
    rec1[split_dim] = n_split // 3 + 1
    rec2 = [s // 2 for s in shape]
    rec2[split_dim] = int(n_split * 0.7)  # likely in a different subdomain
    src_loc = torch.tensor([[src]])
    rec_loc = torch.tensor([[rec1, rec2]])
    pml = [pml_w] * (2 * ndim)
    return v, dt, amp, src_loc, rec_loc, pml, nt


# (ndim, n_split, n_subdomains, split_dim, courant, pml_width)
FORWARD_CASES = [
    (1, 60, 2, 0, 0.4, 0),  # reflecting box, step_ratio == 1
    (1, 60, 3, 0, 0.4, 10),  # three slabs, real PML
    (1, 60, 2, 0, 0.9, 10),  # high Courant -> step_ratio > 1 (resampling path)
    (2, 40, 2, 0, 0.4, 8),  # 2-D split along the slow axis
    (2, 40, 3, 1, 0.4, 8),  # 2-D split along the fast axis, three slabs
    (2, 40, 2, 0, 0.9, 8),  # 2-D, step_ratio > 1
    (3, 28, 3, 2, 0.4, 6),  # 3-D, separately compiled kernels
]


@pytest.mark.parametrize("ndim,n_split,n_sub,split_dim,courant,pml_w", FORWARD_CASES)
def test_decomposed_matches_monolithic(ndim, n_split, n_sub, split_dim, courant, pml_w):
    """Stitched forward wavefield and receivers equal a single scalar() call."""
    v, dt, amp, src_loc, rec_loc, pml, nt = _setup(
        ndim, n_split, split_dim, courant, pml_w
    )
    mono = scalar(
        v, DX, dt, source_amplitudes=amp, source_locations=src_loc,
        receiver_locations=rec_loc, accuracy=4, pml_width=pml, max_vel=C,
        pml_freq=FREQ, nt=nt, return_dataclass=True,
    )
    mono_wf = _crop_interior(mono.wavefield_0, pml, ndim)

    dd_wf, dd_rec = scalar_decomposed(
        v, DX, dt, source_amplitudes=amp, source_locations=src_loc,
        receiver_locations=rec_loc, n_subdomains=n_sub, split_dim=split_dim,
        accuracy=4, pml_width=pml, max_vel=C, pml_freq=FREQ, nt=nt,
    )

    wf_err = (dd_wf - mono_wf).norm() / mono_wf.norm()
    rec_err = (
        (dd_rec - mono.receiver_amplitudes).norm() / mono.receiver_amplitudes.norm()
    )
    assert wf_err < 1e-9, f"wavefield mismatch {wf_err:.2e}"
    assert rec_err < 1e-9, f"receiver mismatch {rec_err:.2e}"


@pytest.mark.parametrize(
    "ndim,n_split,n_sub,split_dim,courant,pml_w",
    [
        (1, 50, 2, 0, 0.4, 10),
        (1, 50, 3, 0, 0.9, 10),  # gradient through the resampling path
        (2, 30, 2, 0, 0.4, 6),
        (2, 30, 3, 1, 0.4, 6),
    ],
)
def test_decomposed_gradient_matches_monolithic(
    ndim, n_split, n_sub, split_dim, courant, pml_w
):
    """The transpose-halo adjoint reproduces the monolithic gradient w.r.t. v."""
    v0, dt, amp, src_loc, rec_loc, pml, nt = _setup(
        ndim, n_split, split_dim, courant, pml_w
    )

    vm = v0.clone().requires_grad_(True)
    mono = scalar(
        vm, DX, dt, source_amplitudes=amp, source_locations=src_loc,
        receiver_locations=rec_loc, accuracy=4, pml_width=pml, max_vel=C,
        pml_freq=FREQ, nt=nt, return_dataclass=True,
    )
    (mono.receiver_amplitudes ** 2).sum().backward()
    grad_mono = vm.grad.clone()

    vd = v0.clone().requires_grad_(True)
    _, dd_rec = scalar_decomposed(
        vd, DX, dt, source_amplitudes=amp, source_locations=src_loc,
        receiver_locations=rec_loc, n_subdomains=n_sub, split_dim=split_dim,
        accuracy=4, pml_width=pml, max_vel=C, pml_freq=FREQ, nt=nt,
    )
    (dd_rec ** 2).sum().backward()
    grad_dd = vd.grad.clone()

    rel = (grad_dd - grad_mono).norm() / grad_mono.norm()
    assert rel < 1e-6, f"gradient mismatch {rel:.2e}"


def test_decomposed_heterogeneous_model():
    """A heterogeneous model (varying local max velocity) still matches exactly.

    This exercises the global ``max_vel`` pinning: every slab must use the same
    internal time step, derived from the whole-model maximum velocity, or the
    slabs desync. ``max_vel`` is left to default (the global maximum).
    """
    dtype = torch.double
    torch.manual_seed(1)
    shape = [34, 40]
    v = 1500.0 + 1500.0 * torch.rand(shape, dtype=dtype)
    dt = 0.4 * DX / float(v.max())
    nt = 70
    amp = ricker(FREQ, nt, dt, 1.5 / FREQ, dtype=dtype).reshape(1, 1, -1)
    src_loc = torch.tensor([[[17, 12]]])
    rec_loc = torch.tensor([[[17, 13], [17, 30]]])
    pml = [8, 8, 8, 8]

    mono = scalar(
        v, DX, dt, source_amplitudes=amp, source_locations=src_loc,
        receiver_locations=rec_loc, accuracy=6, pml_width=pml, pml_freq=FREQ,
        nt=nt, return_dataclass=True,
    )
    mono_wf = _crop_interior(mono.wavefield_0, pml, 2)
    dd_wf, dd_rec = scalar_decomposed(
        v, DX, dt, source_amplitudes=amp, source_locations=src_loc,
        receiver_locations=rec_loc, n_subdomains=3, split_dim=0, accuracy=6,
        pml_width=pml, pml_freq=FREQ, nt=nt,
    )
    assert (dd_wf - mono_wf).norm() / mono_wf.norm() < 1e-9
    assert (
        (dd_rec - mono.receiver_amplitudes).norm() / mono.receiver_amplitudes.norm()
    ) < 1e-9


def test_long_run_stable_three_slabs():
    """Three slabs with symmetric PML stay exact over a long run.

    The PML-corner instability (rejected for split-PML > perpendicular-PML) must
    not appear in the supported symmetric-PML regime even after many steps, where
    an exponentially-growing seam mode would have surfaced. ``nt`` is long enough
    that the wave crosses both seams and reaches the PML corners repeatedly.
    """
    dtype = torch.double
    dt = 0.4 * DX / C
    shape = (30, 40)
    v = torch.full(shape, C, dtype=dtype)
    nt = 220
    amp = ricker(FREQ, nt, dt, 1.5 / FREQ, dtype=dtype).reshape(1, 1, -1)
    src_loc = torch.tensor([[[15, 8]]])
    rec_loc = torch.tensor([[[15, 20]]])
    pml = [8, 8, 8, 8]
    mono = scalar(
        v, DX, dt, source_amplitudes=amp, source_locations=src_loc,
        receiver_locations=rec_loc, accuracy=4, pml_width=pml, max_vel=C,
        pml_freq=FREQ, nt=nt, return_dataclass=True,
    )
    mono_wf = _crop_interior(mono.wavefield_0, pml, 2)
    dd_wf, dd_rec = scalar_decomposed(
        v, DX, dt, source_amplitudes=amp, source_locations=src_loc,
        receiver_locations=rec_loc, n_subdomains=3, split_dim=1, accuracy=4,
        pml_width=pml, max_vel=C, pml_freq=FREQ,
    )
    assert (dd_wf - mono_wf).norm() / mono_wf.norm() < 1e-9
    assert (
        (dd_rec - mono.receiver_amplitudes).norm() / mono.receiver_amplitudes.norm()
    ) < 1e-9


def test_wide_split_pml_three_slabs_rejected():
    """The unstable PML-corner regime (split-PML > perpendicular-PML) is rejected."""
    v = torch.full((30, 40), C, dtype=torch.double)
    amp = ricker(FREQ, 40, 0.001, 1.5 / FREQ, dtype=torch.double).reshape(1, 1, -1)
    src_loc = torch.tensor([[[15, 8]]])
    with pytest.raises(ValueError, match="Unstable configuration"):
        scalar_decomposed(
            v, DX, 0.001, source_amplitudes=amp, source_locations=src_loc,
            n_subdomains=3, split_dim=1, accuracy=4,
            pml_width=[6, 6, 12, 12], max_vel=C, pml_freq=FREQ,
        )
    # The same wide PML on the split axis is fine with only two slabs.
    scalar_decomposed(
        v, DX, 0.001, source_amplitudes=amp, source_locations=src_loc,
        n_subdomains=2, split_dim=1, accuracy=4,
        pml_width=[6, 6, 12, 12], max_vel=C, pml_freq=FREQ,
    )


def test_partition_too_fine_raises():
    """Splitting below the halo width (accuracy//2 cells per slab) is rejected."""
    v = torch.full((8,), C, dtype=torch.double)
    amp = ricker(FREQ, 20, 0.001, 1.5 / FREQ, dtype=torch.double).reshape(1, 1, -1)
    with pytest.raises(ValueError, match="at least"):
        scalar_decomposed(
            v, DX, 0.001, source_amplitudes=amp,
            source_locations=torch.tensor([[[4]]]), n_subdomains=5, accuracy=4,
            pml_width=0, max_vel=C, pml_freq=FREQ,
        )
