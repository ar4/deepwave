"""Tests for deepwave.callbacks."""

import os
import uuid
from typing import Any

import pytest
import torch
from helpers import output_tensors

import deepwave
from deepwave.scalar import scalar_model_names, scalar_wavefield_names


@pytest.mark.parametrize(
    "backend",
    [
        "eager",
        "compiled",
    ],
)
def test_scalar_callback_call_count(backend) -> None:
    """Check that the callbacks are called the correct number of times."""
    v = torch.ones(10, 10) * 1500
    v.requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes = torch.zeros(1, 1, nt)
    source_amplitudes[0, 0, 5] = 1
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations[0, 0, 0] = 5
    source_locations[0, 0, 1] = 5
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations[0, 0, 0] = 5
    receiver_locations[0, 0, 1] = 5

    class Counter:
        """A simple counter class for callbacks."""

        def __init__(self) -> None:
            self.count = 0

        def __call__(self, state: deepwave.CallbackState) -> None:
            """Increments the counter."""
            self.count += 1

    # Test with a frequency that divides nt evenly
    forward_counter = Counter()
    backward_counter = Counter()
    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        forward_callback=forward_counter,
        backward_callback=backward_counter,
        callback_frequency=2,
        backend=backend,
    )
    assert forward_counter.count == nt / 2

    # The backward callback must fire once per forward callback step on
    # every backend, including the step whose forward wavefield is the
    # (non-requiring-grad) initial state.
    out[-1].sum().backward()
    assert backward_counter.count == nt / 2
    v.grad.zero_()

    # Test with a frequency that does not divide nt evenly
    forward_counter = Counter()
    backward_counter = Counter()
    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        forward_callback=forward_counter,
        backward_callback=backward_counter,
        callback_frequency=3,
        backend=backend,
    )
    assert forward_counter.count == (nt + 2) // 3

    out[-1].sum().backward()
    assert backward_counter.count == (nt + 2) // 3


def test_scalar_storage_mode() -> None:
    """Check that the storage mode does not affect the gradient."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = torch.ones(10, 10, device=device) * 1500
    v.requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes = torch.zeros(1, 1, nt, device=device)
    source_amplitudes[0, 0, 5] = 1
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    source_locations[0, 0, 0] = 5
    source_locations[0, 0, 1] = 5
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    receiver_locations[0, 0, 0] = 5
    receiver_locations[0, 0, 1] = 5

    def noop(state: deepwave.CallbackState) -> None:
        pass

    out1 = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        forward_callback=noop,
        backward_callback=noop,
    )
    out1[-1].sum().backward()
    grad1 = v.grad.clone()
    v.grad.zero_()

    out2 = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        storage_mode="disk",
        forward_callback=noop,
        backward_callback=noop,
    )
    out2[-1].sum().backward()
    grad2 = v.grad.clone()

    assert torch.allclose(grad1, grad2)


def _make_unwritable_disk_storage_dir(tmp_path, monkeypatch) -> None:
    """Pin the disk-storage subdirectory name and make it unwritable.

    ``deepwave._storage.TemporaryStorage`` creates its own randomly-named
    subdirectory under ``storage_path`` via
    ``Path.mkdir(parents=True, exist_ok=True)``, which succeeds silently if
    the subdirectory already exists, regardless of that subdirectory's
    permissions. Pinning ``uuid4`` and pre-creating the subdirectory
    ourselves (before removing its write permission) means Python's own
    directory-creation step does not fail first -- the compiled backend's own
    ``fopen`` call inside that subdirectory does, exercising the error
    handling under test.
    """
    fixed_uuid = uuid.UUID(int=0)
    monkeypatch.setattr("deepwave._storage.uuid4", lambda: fixed_uuid)
    storage_dir = tmp_path / f"deepwave_tmp_{os.getpid()}_{fixed_uuid.hex}"
    storage_dir.mkdir(parents=True)
    storage_dir.chmod(0o555)


def test_scalar_storage_mode_disk_unwritable_path_cpu(tmp_path, monkeypatch) -> None:
    """Check that disk storage to an unwritable path raises cleanly on CPU.

    Regression test: the compiled CPU backend used to call ``fopen`` for its
    disk-storage snapshot files without checking for failure, so an
    unwritable ``storage_path`` led straight to a null-pointer ``fwrite``/
    ``fseek`` (undefined behaviour) inside ``storage_utils.c`` rather than a
    clean Python exception.
    """
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("running as root bypasses directory permission checks")
    _make_unwritable_disk_storage_dir(tmp_path, monkeypatch)

    v = torch.ones(10, 10) * 1500
    v.requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes = torch.zeros(1, 1, nt)
    source_amplitudes[0, 0, 5] = 1
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations[0, 0, 0] = 5
    source_locations[0, 0, 1] = 5
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations[0, 0, 0] = 5
    receiver_locations[0, 0, 1] = 5

    with pytest.raises(RuntimeError):
        deepwave.scalar(
            v,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            storage_mode="disk",
            storage_path=str(tmp_path),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_scalar_storage_mode_disk_unwritable_path_cuda(tmp_path, monkeypatch) -> None:
    """Check that disk storage to an unwritable path raises cleanly on CUDA.

    Regression test: the compiled CUDA backend's ``ScopedFile::open`` used to
    call ``fopen`` for its disk-storage snapshot files without checking for
    failure, so an unwritable ``storage_path`` led straight to unconditional
    ``fseek``/``fwrite``/``fread`` calls on a null ``FILE*`` (undefined
    behaviour, reliably a segfault) inside ``storage_utils.cu`` rather than a
    clean Python exception.
    """
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("running as root bypasses directory permission checks")
    _make_unwritable_disk_storage_dir(tmp_path, monkeypatch)

    device = torch.device("cuda")
    v = torch.ones(10, 10, device=device) * 1500
    v.requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes = torch.zeros(1, 1, nt, device=device)
    source_amplitudes[0, 0, 5] = 1
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    source_locations[0, 0, 0] = 5
    source_locations[0, 0, 1] = 5
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    receiver_locations[0, 0, 0] = 5
    receiver_locations[0, 0, 1] = 5

    with pytest.raises(RuntimeError):
        deepwave.scalar(
            v,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            storage_mode="disk",
            storage_path=str(tmp_path),
        )


@pytest.mark.parametrize(
    "backend",
    [
        "eager",
        "compiled",
    ],
)
def test_scalar_callback_wavefield_shape(backend) -> None:
    """Check that the wavefield has the correct shape for each view."""
    v = torch.ones(10, 10) * 1500
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes = torch.zeros(1, 1, nt)
    source_amplitudes[0, 0, 5] = 1
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations[0, 0, 0] = 5
    source_locations[0, 0, 1] = 5
    pml_width = 5

    class Checker:
        """A checker class for wavefield shapes."""

        def __call__(self, state: deepwave.CallbackState) -> None:
            """Checks the shape of the wavefield for different views."""
            w_inner = state.get_wavefield("wavefield_0", "inner")
            assert w_inner.shape == (1, 10, 10)
            w_pml = state.get_wavefield("wavefield_0", "pml")
            assert w_pml.shape == (1, 10 + 2 * pml_width, 10 + 2 * pml_width)
            w_full = state.get_wavefield("wavefield_0", "full")
            assert w_full.shape == (
                1,
                10 + 2 * pml_width + 2 * 4,
                10 + 2 * pml_width + 2 * 4,
            )

    deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        pml_width=pml_width,
        accuracy=8,
        forward_callback=Checker(),
        callback_frequency=20,
        backend=backend,
    )


@pytest.mark.parametrize(
    "backend",
    [
        "eager",
        "compiled",
    ],
)
def test_scalar_callback_wavefield_modification(backend) -> None:
    """Check that the wavefield can be modified in the forward callback."""
    v = torch.ones(10, 10) * 1500
    dx = 5.0
    dt = 0.0001
    nt = 20
    source_amplitudes = torch.zeros(1, 1, nt)
    source_amplitudes[0, 0, 5] = 1
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations[0, 0, 0] = 5
    source_locations[0, 0, 1] = 5
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations[0, 0, 0] = 5
    receiver_locations[0, 0, 1] = 5

    class Setter:
        """A setter class for wavefield modification."""

        def __init__(self) -> None:
            self.expected = torch.zeros(nt)

        def __call__(self, state: deepwave.CallbackState) -> None:
            """Modifies the wavefield at a specific location."""
            val = torch.randn(1)
            self.expected[state.step] = val
            w_pml = state.get_wavefield("wavefield_0", "pml")
            # The coordinates are relative to the padded model, so we need to
            # add the PML width to get the correct index.
            w_pml[0, 5 + state._pml_width[0], 5 + state._pml_width[2]] = val

    setter = Setter()
    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        forward_callback=setter,
        callback_frequency=1,
        pml_width=5,
        backend=backend,
    )
    assert torch.allclose(out[-1].flatten(), setter.expected)


def test_scalar_callback_gradient_modification() -> None:
    """Check that the gradient can be modified in the backward callback."""
    v = torch.ones(10, 10) * 1500
    v.requires_grad_()
    dx = 5.0
    dt = 0.004
    source_amplitudes = torch.zeros(1, 1, 20)
    source_amplitudes[0, 0, 5] = 1
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations[0, 0, 0] = 5
    source_locations[0, 0, 1] = 5
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations[0, 0, 0] = 5
    receiver_locations[0, 0, 1] = 5

    def modifier(state: deepwave.CallbackState) -> None:
        """Modifies the gradient by multiplying it by 2."""
        grad_v = state.get_gradient("v", "pml")
        grad_v *= 2

    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
    )
    out[-1].sum().backward()
    grad1 = v.grad.clone()
    v.grad.zero_()
    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        backward_callback=modifier,
        callback_frequency=20,
    )
    out[-1].sum().backward()
    grad2 = v.grad.clone()
    # Zero source location to ignore source contribution
    grad1[source_locations[0, 0, 0], source_locations[0, 0, 1]] = 0
    grad2[source_locations[0, 0, 0], source_locations[0, 0, 1]] = 0
    assert torch.allclose(grad1 * 2, grad2)


def test_scalar_callback_gradient_modification_inner() -> None:
    """Check that modifying an inner view of the gradient works."""
    v = torch.ones(10, 10) * 1500
    v.requires_grad_()
    dx = 5.0
    dt = 0.004
    source_amplitudes = torch.zeros(1, 1, 20)
    source_amplitudes[0, 0, 5] = 1
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations[0, 0, 0] = 5
    source_locations[0, 0, 1] = 5
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations[0, 0, 0] = 5
    receiver_locations[0, 0, 1] = 5
    pml_width = 20

    def modifier(state: deepwave.CallbackState) -> None:
        """Modifies the inner gradient by multiplying it by 2."""
        grad_v = state.get_gradient("v", "inner")
        grad_v[:, 1:-1, 1:-1] *= 2

    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_width=pml_width,
    )
    out[-1].sum().backward()
    grad1 = v.grad.clone()
    v.grad.zero_()
    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_width=pml_width,
        backward_callback=modifier,
        callback_frequency=20,
    )
    out[-1].sum().backward()
    grad2 = v.grad.clone()
    # The gradient should be doubled, except for the edges (which
    # will be influenced by the PML regions that were not scaled)
    # and the source location
    grad1[source_locations[0, 0, 0], source_locations[0, 0, 1]] = 0
    grad2[source_locations[0, 0, 0], source_locations[0, 0, 1]] = 0
    assert torch.allclose(grad1[1:-1, 1:-1] * 2, grad2[1:-1, 1:-1])
    grad1[1:-1, 1:-1] = 0
    grad2[1:-1, 1:-1] = 0
    assert torch.allclose(grad1, grad2)


@pytest.mark.parametrize(
    "backend",
    [
        "compiled",
        "eager",
    ],
)
def test_scalar_callback_live_view_mutation_contract(backend) -> None:
    """Pin the contract that forward_callback wavefields are live views.

    ``CallbackState.get_wavefield`` returns a view into Deepwave's actual
    internal wavefield tensor, not a copy. In-place mutation of the returned
    tensor is a supported way to alter the running simulation, not merely to
    inspect it. This test locks that contract on both the default compiled
    backend and the eager backend: it compares a mutated run against an
    unmutated control run and requires them to differ, so a future change
    that defensively ``.clone()``s the returned tensor (silently breaking
    every user callback that relies on mutation) would be caught here.
    """
    v = torch.ones(10, 10) * 1500
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes = torch.zeros(1, 1, nt)
    source_amplitudes[0, 0, 5] = 1
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations[0, 0, 0] = 5
    source_locations[0, 0, 1] = 5
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations[0, 0, 0] = 5
    receiver_locations[0, 0, 1] = 5

    def make_forward_mutator():
        def mutate(state: deepwave.CallbackState) -> None:
            if state.step == 10:
                state.get_wavefield("wavefield_0").add_(1.0)

        return mutate

    def no_op(state: deepwave.CallbackState) -> None:
        pass

    def run(forward_callback):
        return deepwave.scalar(
            v,
            dx,
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            forward_callback=forward_callback,
            callback_frequency=1,
            backend=backend,
        )

    control = run(no_op)
    mutated = run(make_forward_mutator())
    assert not torch.allclose(control.receiver_amplitudes, mutated.receiver_amplitudes)
    assert not torch.allclose(control[-1], mutated[-1])


def test_scalar_callback_live_view_mutation_contract_gradient() -> None:
    """Pin the contract that backward_callback gradients are live views.

    ``CallbackState.get_gradient`` returns a view into the actual gradient
    tensor Deepwave is accumulating, not a copy: in-place mutation of the
    returned tensor changes the gradient ultimately assigned to ``v.grad``
    (this is exactly how the illumination compensation in
    ``docs/example_custom_imaging_condition.py`` works). Checked on the
    default compiled backend, which is where ``get_gradient`` is populated
    inside a ``backward_callback`` (the eager backend's backward callback
    exposes the adjoint wavefield via ``get_wavefield``, not a per-step
    ``get_gradient``; see ``CallbackState.__init__``'s ``gradients`` param).
    """
    v = torch.ones(10, 10) * 1500
    v.requires_grad_()
    dx = 5.0
    dt = 0.004
    source_amplitudes = torch.zeros(1, 1, 20)
    source_amplitudes[0, 0, 5] = 1
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations[0, 0, 0] = 5
    source_locations[0, 0, 1] = 5
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations[0, 0, 0] = 5
    receiver_locations[0, 0, 1] = 5

    def gradient_mutator(state: deepwave.CallbackState) -> None:
        state.get_gradient("v").add_(1.0)

    out_control = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
    )
    out_control[-1].sum().backward()
    grad_control = v.grad.detach().clone()
    v.grad.zero_()

    out_mutated = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        backward_callback=gradient_mutator,
        callback_frequency=1,
    )
    out_mutated[-1].sum().backward()
    grad_mutated = v.grad.detach().clone()
    v.grad.zero_()

    assert not torch.allclose(grad_control, grad_mutated)


def test_scalar_callback_equivalence() -> None:
    """Check that a do-nothing callback does not change the output."""
    v = torch.ones(10, 10) * 1500
    v.requires_grad_()
    dx = 5.0
    dt = 0.004
    source_amplitudes = torch.zeros(1, 1, 20)
    source_amplitudes[0, 0, 5] = 1
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations[0, 0, 0] = 5
    source_locations[0, 0, 1] = 5
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations[0, 0, 0] = 5
    receiver_locations[0, 0, 1] = 5

    def do_nothing(state: deepwave.CallbackState) -> None:
        """A do-nothing callback function."""

    out1 = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
    )
    out1[-1].sum().backward()
    grad1 = v.grad.detach().clone()
    v.grad.zero_()

    out2 = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        forward_callback=do_nothing,
        backward_callback=do_nothing,
        callback_frequency=1,
    )
    out2[-1].sum().backward()
    grad2 = v.grad.detach().clone()
    # The auxiliary state is one output element holding several tensors.
    for x, y in zip(output_tensors(out1), output_tensors(out2)):
        assert torch.allclose(x, y)
    assert torch.allclose(grad1, grad2)
    v.grad.zero_()

    # Test with a frequency that does not divide nt evenly
    out3 = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        forward_callback=do_nothing,
        backward_callback=do_nothing,
        callback_frequency=3,
    )
    out3[-1].sum().backward()
    grad3 = v.grad.detach().clone()
    # The auxiliary state is one output element holding several tensors.
    for x, y in zip(output_tensors(out1), output_tensors(out3)):
        assert torch.allclose(x, y)
    assert torch.allclose(grad1, grad3)


def test_scalar_state_preservation() -> None:
    """Check that the wavefield state is preserved between callbacks."""
    v = torch.ones(10, 10) * 1500
    dx = 5.0
    dt = 0.0001
    source_amplitudes = torch.zeros(1, 1, 20)
    source_amplitudes[0, 0, 5] = 1
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations[0, 0, 0] = 5
    source_locations[0, 0, 1] = 5

    class StateChecker:
        """A checker class for wavefield state preservation."""

        def __init__(self) -> None:
            self.wfc = None

        def __call__(self, state: deepwave.CallbackState) -> None:
            """Checks if the wavefield state is preserved."""
            if self.wfc is not None:
                assert torch.allclose(
                    self.wfc, state.get_wavefield("wavefield_m1", "full")
                )
            self.wfc = state.get_wavefield("wavefield_0", "full").clone()

    deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        forward_callback=StateChecker(),
        callback_frequency=1,
    )


def test_scalar_multishot_equivalence() -> None:
    """Check that a do-nothing callback does not change the output.

    Checks it for multiple shots.
    """
    v = torch.ones(10, 10) * 1500
    dx = 5.0
    dt = 0.004
    source_amplitudes = torch.zeros(2, 1, 20)
    source_amplitudes[0, 0, 5] = 1
    source_amplitudes[1, 0, 8] = 1
    source_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    source_locations[0, 0, 0] = 5
    source_locations[0, 0, 1] = 5
    source_locations[1, 0, 0] = 3
    source_locations[1, 0, 1] = 3
    receiver_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    receiver_locations[0, 0, 0] = 5
    receiver_locations[0, 0, 1] = 5
    receiver_locations[1, 0, 0] = 3
    receiver_locations[1, 0, 1] = 3

    def do_nothing(state: deepwave.CallbackState) -> None:
        """A do-nothing callback function."""

    out1 = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
    )
    out2 = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        forward_callback=do_nothing,
        callback_frequency=1,
    )
    # The auxiliary state is one output element holding several tensors.
    for x, y in zip(output_tensors(out1), output_tensors(out2)):
        assert torch.allclose(x, y)


def test_scalar_backward_callback_only_call_count() -> None:
    """Check that the backward callback is called the correct number of times.

    Check it when no forward callback is provided.
    """
    v = torch.ones(10, 10) * 1500
    v.requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes = torch.zeros(1, 1, nt)
    source_amplitudes[0, 0, 5] = 1
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations[0, 0, 0] = 5
    source_locations[0, 0, 1] = 5
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations[0, 0, 0] = 5
    receiver_locations[0, 0, 1] = 5

    class Counter:
        """A simple counter class for callbacks."""

        def __init__(self) -> None:
            self.count = 0

        def __call__(self, state: deepwave.CallbackState) -> None:
            """Increments the counter."""
            self.count += 1

    # Test with a frequency that divides nt evenly
    backward_counter = Counter()
    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        backward_callback=backward_counter,
        callback_frequency=2,
    )
    out[-1].sum().backward()
    assert backward_counter.count == nt / 2

    # Test with a frequency that does not divide nt evenly
    v.grad.zero_()
    backward_counter = Counter()
    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        backward_callback=backward_counter,
        callback_frequency=3,
    )
    out[-1].sum().backward()
    assert backward_counter.count == (nt + 2) // 3


@pytest.mark.parametrize("ndim", [2, 3])
def test_scalar_backward_callback_dict_keys(ndim) -> None:
    """Compiled backward-callback wavefield/gradient dict key sets are exact.

    ``state._wavefields`` must equal ``scalar_wavefield_names(ndim)`` exactly:
    a prior bug (fixed for elastic/elastic_vti/acoustic_vti in commit
    837b3cbf, via ``_engine.run_backward_core``'s removed
    ``callback_drop_last`` parameter) silently dropped the last wavefield name
    from this dict. scalar's compiled backend shares that same engine code
    path, so it is equally exposed and gets the same regression test here.
    ``state._gradients`` must equal ``scalar_model_names(ndim)`` exactly (the
    compiled backend's backward callback also exposes per-model gradient
    tensors as they accumulate).
    """
    n = 10
    shape = (n,) * ndim
    v = (torch.ones(*shape) * 1500).requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 8
    source_amplitudes = torch.zeros(1, 1, nt)
    source_amplitudes[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded: dict[str, dict[str, torch.Tensor]] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded["wavefields"] = dict(state._wavefields)
        recorded["gradients"] = dict(state._gradients)

    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=loc,
        receiver_locations=loc,
        pml_width=3,
        backward_callback=recorder,
        callback_frequency=1,
    )
    out.receiver_amplitudes.sum().backward()

    assert set(recorded["wavefields"]) == set(scalar_wavefield_names(ndim))
    assert all(t.numel() > 0 for t in recorded["wavefields"].values())
    assert set(recorded["gradients"]) == set(scalar_model_names(ndim))
    assert all(t.numel() > 0 for t in recorded["gradients"].values())


@pytest.mark.parametrize("ndim", [2, 3])
def test_scalar_backward_callback_eager_dict_is_primary_wavefield_only(ndim) -> None:
    """The eager backward-callback wavefield dict has a deliberately narrow scope.

    Unlike the compiled backend (see
    ``test_scalar_backward_callback_dict_keys``), the eager backend fires
    ``backward_callback`` from an autograd hook registered on the primary
    wavefield tensor (``scalar.py``'s ``_register_backward_callback``): it
    supplies only ``{"wavefield_0": grad}``, never the psi/zeta auxiliaries,
    and its gradients dict is always empty (model gradients are ordinary
    autograd leaves in the eager backend, not backend-accumulated buffers).
    This is asserted explicitly, as documented behaviour, so a future change
    that widens or accidentally narrows this contract is caught either way.
    """
    n = 10
    shape = (n,) * ndim
    v = (torch.ones(*shape) * 1500).requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 8
    source_amplitudes = torch.zeros(1, 1, nt)
    source_amplitudes[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded: dict[str, dict[str, torch.Tensor]] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded["wavefields"] = dict(state._wavefields)
        recorded["gradients"] = dict(state._gradients)

    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=loc,
        receiver_locations=loc,
        pml_width=3,
        backward_callback=recorder,
        callback_frequency=1,
        backend="eager",
    )
    out.receiver_amplitudes.sum().backward()

    assert set(recorded["wavefields"]) == {"wavefield_0"}
    assert recorded["wavefields"]["wavefield_0"].numel() > 0
    assert recorded["gradients"] == {}


@pytest.mark.parametrize("ndim", [2, 3])
def test_scalar_forward_callback_dict_keys_and_cross_backend_agreement(ndim) -> None:
    """Forward-callback wavefield/model dict keys match spec names on both backends.

    Checks the dict key sets against ``scalar_wavefield_names``/
    ``scalar_model_names`` (the source of truth also used by
    ``_engine.named_output``/``PropagatorSpec``) for both the eager and the
    compiled backend, that every value has nonzero ``numel()``, and that the
    eager and compiled wavefields agree numerically at the final callback
    call. No test in this file previously compared ``backend="eager"``
    output against the compiled backend numerically; the tolerance used here
    (``atol=5e-4 * <value>.abs().max()``) is the formula this file already
    uses for its established python-backend-vs-compiled-backend comparisons
    (``test_direct``, ``test_v_batched`` in ``test_scalar.py``), which compare
    a ``torch.compile``d Python backend against the compiled C backend -- the
    closest existing convention, since no eager-specific numeric comparison
    exists to copy directly.
    """
    n = 10
    shape = (n,) * ndim
    v = torch.full(shape, 1500.0)
    dx = 5.0
    dt = 0.0005
    nt = 12
    source_amplitudes = torch.zeros(1, 1, nt)
    source_amplitudes[0, 0, 2] = 1.0
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded: dict[str, dict[str, torch.Tensor]] = {}
    recorded_models: dict[str, dict[str, torch.Tensor]] = {}

    def make_recorder(key: str):
        def recorder(state: deepwave.CallbackState) -> None:
            recorded[key] = {
                wf_name: t.clone() for wf_name, t in state._wavefields.items()
            }
            recorded_models[key] = {
                m_name: t.clone() for m_name, t in state._models.items()
            }

        return recorder

    common_kwargs: dict[str, Any] = {
        "source_amplitudes": source_amplitudes,
        "source_locations": loc,
        "pml_width": 4,
        "callback_frequency": 1,
    }
    deepwave.scalar(
        v,
        dx,
        dt,
        backend="eager",
        forward_callback=make_recorder("eager"),
        **common_kwargs,
    )
    deepwave.scalar(
        v,
        dx,
        dt,
        backend="compiled",
        forward_callback=make_recorder("compiled"),
        **common_kwargs,
    )

    for key in ("eager", "compiled"):
        assert set(recorded[key]) == set(scalar_wavefield_names(ndim))
        assert all(t.numel() > 0 for t in recorded[key].values())
        assert set(recorded_models[key]) == set(scalar_model_names(ndim))
        assert all(t.numel() > 0 for t in recorded_models[key].values())

    eager_dict = recorded["eager"]
    compiled_dict = recorded["compiled"]
    for name, eager_val in eager_dict.items():
        compiled_val = compiled_dict[name]
        assert eager_val.shape == compiled_val.shape, (
            f"{name}: shape mismatch {eager_val.shape} vs {compiled_val.shape}"
        )
        atol = 5e-4 * eager_val.abs().max().item()
        assert torch.allclose(eager_val, compiled_val, atol=atol), (
            f"{name}: eager/compiled mismatch"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_scalar_backward_callback_storage_mode_cpu_bounds_device_memory() -> None:
    """storage_mode="cpu" + backward_callback must not add a full device history.

    Regression test: ``ScalarForwardFunc.forward`` used to always allocate its
    opt-in raw-forward checkpoint buffer (paid whenever a ``backward_callback``
    is supplied) as a full DEVICE-resident, uncompressed wavefield history,
    ignoring the user's ``storage_mode``. With ``storage_mode="cpu"`` neither
    the primary checkpoint nor this second one should keep a full multi-step
    history on the device, so peak device memory stays well under the
    two-full-histories bound the old, broken behaviour (one full history for
    each) would approach.
    """
    device = torch.device("cuda")
    ny, nx = 48, 48
    nt = 80
    dx = 5.0
    dt = 0.001
    v = torch.ones(ny, nx, device=device) * 1500
    v.requires_grad_()
    source_amplitudes = torch.zeros(1, 1, nt, device=device)
    source_amplitudes[0, 0, 5] = 1
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    source_locations[0, 0, :] = ny // 2
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    receiver_locations[0, 0, :] = ny // 2

    def noop(state: deepwave.CallbackState) -> None:
        pass

    est = deepwave.estimate_memory(
        "scalar",
        model_shape=(ny, nx),
        nt=nt,
        n_shots=1,
        storage_mode="device",
        n_models_requiring_grad=0,
    )
    single_history_bytes = nt * est["padded_model_elements"] * 4  # float32

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        storage_mode="cpu",
        backward_callback=noop,
        callback_frequency=4,
    )
    out[-1].sum().backward()
    peak = torch.cuda.max_memory_allocated(device)
    assert peak < 1.5 * single_history_bytes, (peak, single_history_bytes)


@pytest.mark.parametrize("storage_mode", ["cpu", "disk"])
def test_scalar_backward_callback_raw_forward_matches_across_storage_modes(
    storage_mode, tmp_path
) -> None:
    """The raw-forward field surfaced to backward_callback is storage_mode-independent.

    Regression test for the bug where the raw-forward checkpoint buffer was
    always allocated DEVICE-resident/uncompressed, ignoring storage_mode.
    Runs the same small problem with storage_mode="device" (the known-good
    baseline) and again with storage_mode="cpu"/"disk", and asserts the
    values ``backward_callback`` receives via ``get_wavefield(...,
    role="forward")`` are bitwise identical at every step -- the check that
    would catch a subtly wrong CPU/disk-mode load (e.g. the wrong step, or
    garbage/zero data from a buffer that was never actually persisted).
    ``storage_mode="cpu"`` is skipped on a CPU-only box, where it is
    transparently downgraded to "device" (see
    ``deepwave._engine.resolve_storage_mode``) and so exercises no different
    code path than the baseline; "disk" is a genuinely different code path on
    both CPU and CUDA, so it alone already covers the CPU-only case.
    """
    if storage_mode == "cpu" and not torch.cuda.is_available():
        pytest.skip("storage_mode='cpu' is downgraded to 'device' on CPU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ny, nx, nt, dt, freq = 16, 16, 25, 0.001, 15.0
    dx = 5.0
    src = (
        deepwave.wavelets.ricker(freq, nt, dt, 1.0 / freq)
        .to(torch.float64)
        .reshape(1, 1, nt)
        .to(device)
    )
    source_locations = torch.tensor([[[2, nx // 2]]])
    receiver_locations = torch.tensor([[[2, c] for c in range(2, nx - 2, 4)]])

    def run(mode: str) -> dict:
        v = torch.full(
            (ny, nx), 1500.0, dtype=torch.float64, device=device, requires_grad=True
        )
        surfaced = {}

        def bcb(state: deepwave.CallbackState) -> None:
            surfaced[state.step] = (
                state.get_wavefield("wavefield_0", view="pml", role="forward")
                .clone()
                .cpu()
            )

        out = deepwave.scalar(
            v,
            dx,
            dt,
            source_amplitudes=src,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_freq=freq,
            max_vel=1800.0,
            storage_mode=mode,
            storage_path=str(tmp_path),
            backward_callback=bcb,
            backward_callback_needs_forward_wavefield=True,
            callback_frequency=1,
        )
        out.receiver_amplitudes.pow(2).sum().backward()
        return surfaced

    baseline = run("device")
    other = run(storage_mode)

    common_steps = set(baseline) & set(other)
    assert len(common_steps) >= nt - 2
    for step in common_steps:
        assert torch.equal(baseline[step], other[step]), step


def _storage_mode_none_problem():
    """A small scalar() problem driving a real backward pass under storage_mode="none".

    ``v`` deliberately does NOT require grad: ``storage_mode="none"`` keeps
    no forward wavefield history, so no model gradient can be formed from it
    and the shared engine now rejects that combination outright (see
    ``test_storage_mode_none_model_gradient.py``). Requiring grad on
    ``source_amplitudes`` instead still drives a full, real backward pass --
    its gradient comes from the adjoint field
    directly, not from any stored/reconstructed forward snapshot, which is
    why it remains available in this mode -- so ``backward_callback`` fires
    normally.
    """
    v = torch.ones(10, 10) * 1500
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes = torch.zeros(1, 1, nt, requires_grad=True)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations[0, 0, :] = 5
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations[0, 0, :] = 5
    return v, dx, dt, source_amplitudes, source_locations, receiver_locations


def test_scalar_backward_callback_storage_mode_none_does_not_crash() -> None:
    """storage_mode="none" + a backward_callback ignoring role="forward" still works.

    Regression test: routing the raw-forward checkpoint through the user's
    real storage_mode (see the storage_mode="cpu"/"disk" tests above) means
    storage_mode="none" resolves to a raw-forward buffer that
    StorageManager.allocate deliberately leaves zero-sized (matching what it
    already does for the primary checkpoint in that mode -- see
    test_estimate_memory_storage_mode_none_has_no_backward_cost in
    test_ergonomics.py). The compiled backend and
    forward_wavefields_for_step must both recognise this and skip the
    buffer entirely, rather than write past its end (undefined behaviour)
    or read stale/empty data, whenever the callback does not itself ask for
    role="forward".
    """
    v, dx, dt, source_amplitudes, source_locations, receiver_locations = (
        _storage_mode_none_problem()
    )
    calls = []

    def bcb(state: deepwave.CallbackState) -> None:
        calls.append(state.step)

    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        storage_mode="none",
        backward_callback=bcb,
        callback_frequency=3,
    )
    out[-1].sum().backward()
    assert len(calls) > 0
    assert source_amplitudes.grad is not None
    assert source_amplitudes.grad.abs().sum() > 0


def test_scalar_backward_callback_storage_mode_none_role_forward_raises() -> None:
    """storage_mode="none" + role="forward" raises a clear error instead of crashing.

    Companion to the "does not crash" test above: a callback that DOES ask
    for the reconstructed forward wavefield in storage_mode="none" (where
    nothing was ever persisted) must get the same clear
    ``get_wavefield(role="forward")`` error as a backend that never supplies
    it at all, not an out-of-bounds read or a bare ``IndexError``.
    """
    v, dx, dt, source_amplitudes, source_locations, receiver_locations = (
        _storage_mode_none_problem()
    )
    captured = {}

    def bcb(state: deepwave.CallbackState) -> None:
        if "err" not in captured:
            try:
                state.get_wavefield("wavefield_0", role="forward")
            except NotImplementedError as e:
                captured["err"] = str(e)

    out = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        storage_mode="none",
        backward_callback=bcb,
        backward_callback_needs_forward_wavefield=True,
        callback_frequency=3,
    )
    out[-1].sum().backward()
    assert "err" in captured
    assert "forward" in captured["err"]
