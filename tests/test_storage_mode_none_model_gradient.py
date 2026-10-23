"""``storage_mode="none"`` must refuse a model gradient, not fabricate one.

``storage_mode="none"`` keeps no forward wavefield history, so the imaging
correlation between the forward and adjoint fields -- which *is* the model
gradient -- cannot be formed. The engine correctly leaves that term out.

What it used to do next was the problem: it neither raised nor returned zero.
The propagators' Python front ends reparameterise their inputs in terms of the
models (``scalar``, for instance, premultiplies the source amplitudes by
``-v**2 * dt**2`` gathered at the source cells), and that path is
differentiable, so ``v.grad`` came back as a plausible-looking spike at the
source locations -- non-zero, wrong by ~1300% on a small case, and exactly the
kind of thing an FWI loop will happily descend on. ``acoustic_born`` and
``elastic_born`` already raised for this combination; the other nine
propagators did not.

The guard lives in the shared engine (``_engine.run_forward_engine``) so it
cannot drift per propagator, and it is deliberately **model-specific**:
gradients with respect to the source amplitudes and the initial wavefields do
not need the forward history and are genuinely correct in this mode, so they
must keep working.

``acoustic_born`` and ``elastic_born`` are deliberately **not** in ``CASES``
below. Their own pre-existing guards are broader: they reject *every*
gradient under ``storage_mode="none"``, source amplitudes and initial
wavefields included, so ``"none"`` is forward-only without exception for those
two. That difference is left alone here (changing it is an interface decision,
not a bug fix) but it does mean the "source gradients still work" contract the
tests below assert is propagator-dependent -- see ``docs/example_storage.rst``.
"""

import pytest
import torch
from helpers import output_tensors

import deepwave

DTYPE = torch.float64
NY, NX, NT = 24, 24, 30
DX, DT = 5.0, 0.0005
PML_FREQ = 25.0

# The propagator's model values and its 2-D source/receiver component
# suffixes, as in test_callbacks_chunked_gradient.py.
CASES = {
    "scalar": ([1500.0], "", ""),
    "acoustic": ([1500.0, 2200.0], "_p", "_p"),
    "viscoacoustic": ([1500.0, 2200.0, 40.0], "_pressure_rate", "_p"),
    "elastic": ([1.2e10, 0.8e10, 1 / 2200.0], "_p", "_p"),
    "acoustic_vti": ([1.2e10, 0.4e10, 1.0e10, 1 / 2200.0], "_sigma_v", "_y"),
    "elastic_vti": ([1.2e10, 0.4e10, 1.0e10, 0.5e10, 1 / 2200.0], "_y", "_y"),
}


def _call(
    name: str,
    *,
    model_requires_grad: bool,
    source_requires_grad: bool,
    storage_mode: str,
) -> tuple[list[torch.Tensor], torch.Tensor, tuple[torch.Tensor, ...]]:
    values, source_suffix, receiver_suffix = CASES[name]
    models = [
        torch.full(
            (NY, NX),
            value,
            dtype=DTYPE,
            requires_grad=model_requires_grad,
        )
        for value in values
    ]
    sources = torch.full((1, 1, 2), 8, dtype=torch.long)
    receivers = torch.full((1, 1, 2), 14, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)
    amplitudes = torch.randn(1, 1, NT, dtype=DTYPE, generator=generator)
    amplitudes.requires_grad_(source_requires_grad)
    # ``q_freq`` is required on the attenuating propagators, and refused on
    # the rest, so it is added only where it belongs.
    q_kwargs = {"q_freq": PML_FREQ} if name == "viscoacoustic" else {}
    forcing = (
        amplitudes * (2200.0 * 1500.0**2) if name == "viscoacoustic" else amplitudes
    )
    out = getattr(deepwave, name)(
        *models,
        DX,
        DT,
        pml_freq=PML_FREQ,
        storage_mode=storage_mode,
        **q_kwargs,
        **{
            f"source_amplitudes{source_suffix}": forcing,
            f"source_locations{source_suffix}": sources,
            f"receiver_locations{receiver_suffix}": receivers,
        },
    )
    return models, amplitudes, tuple(out)


@pytest.mark.parametrize("name", sorted(CASES))
def test_storage_mode_none_with_model_requiring_grad_raises(name: str) -> None:
    """A model requiring grad under ``"none"`` is an error, not a wrong number."""
    with pytest.raises(RuntimeError, match='storage_mode="none"'):
        _call(
            name,
            model_requires_grad=True,
            source_requires_grad=False,
            storage_mode="none",
        )


@pytest.mark.parametrize("name", sorted(CASES))
def test_storage_mode_none_source_gradient_still_works(name: str) -> None:
    """Source-amplitude gradients need no forward history, so they stay allowed.

    They are also *correct*: this checks the ``"none"`` gradient against the
    same call under ``"device"``, which keeps the full history.
    """
    reference = None
    for storage_mode in ("device", "none"):
        _, amplitudes, out = _call(
            name,
            model_requires_grad=False,
            source_requires_grad=True,
            storage_mode=storage_mode,
        )
        sum(
            tensor.sum() for tensor in output_tensors(out) if tensor.numel() > 0
        ).backward()
        assert amplitudes.grad is not None
        gradient = amplitudes.grad.detach().clone()
        assert gradient.norm() > 0, (
            f"{name}: the source-amplitude gradient under "
            f'storage_mode="{storage_mode}" is zero, so this test is vacuous'
        )
        if reference is None:
            reference = gradient
        else:
            relative = ((gradient - reference).norm() / reference.norm()).item()
            assert relative < 1e-12, (
                f"{name}: the source-amplitude gradient under "
                f'storage_mode="none" differs from the storage_mode="device" '
                f"one by a relative {relative:.3e}"
            )


@pytest.mark.parametrize("name", sorted(CASES))
def test_storage_mode_none_forward_only_still_works(name: str) -> None:
    """Forward-only ``"none"`` is a supported low-memory mode; it must not raise.

    It must also return the same numbers as a run that does keep the history:
    ``"none"`` changes only what is stored for the backward pass.
    """
    reference = None
    for storage_mode in ("device", "none"):
        _, _, out = _call(
            name,
            model_requires_grad=False,
            source_requires_grad=False,
            storage_mode=storage_mode,
        )
        outputs = [tensor for tensor in output_tensors(out) if tensor.numel() > 0]
        assert outputs, f"{name}: no non-empty outputs, so this test is vacuous"
        if reference is None:
            reference = outputs
        else:
            for index, (actual, expected) in enumerate(zip(outputs, reference)):
                assert torch.equal(actual, expected), (
                    f"{name}: output {index} differs between "
                    'storage_mode="none" and storage_mode="device"'
                )
