"""Tests for deepwave.callbacks_acoustic."""

from typing import Any

import pytest
import torch
from helpers import output_tensors

import deepwave
from deepwave.acoustic import acoustic_model_names, acoustic_wavefield_names


@pytest.mark.parametrize(
    "backend",
    [
        "eager",
        "compiled",
    ],
)
def test_acoustic_callback_call_count(backend) -> None:
    """Check that the callbacks are called the correct number of times."""
    v = torch.ones(10, 10) * 1500
    rho = torch.ones(10, 10) * 2200
    v.requires_grad_()
    rho.requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes_volume_rate = torch.zeros(1, 1, nt)
    source_amplitudes_volume_rate[0, 0, 5] = 1
    source_locations_volume_rate = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_volume_rate[0, 0, 0] = 5
    source_locations_volume_rate[0, 0, 1] = 5
    receiver_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations_p[0, 0, 0] = 5
    receiver_locations_p[0, 0, 1] = 5

    class Counter:
        """A simple counter class for callbacks."""

        def __init__(self) -> None:
            self.count = 0

        def __call__(self, state: deepwave.CallbackState) -> None:
            """Increments the counter."""
            self.count += 1

    # Test with a frequency that divides nt evenly
    forward_counter = Counter()
    backward_counter = None if backend == "eager" else Counter()
    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
        receiver_locations_p=receiver_locations_p,
        forward_callback=forward_counter,
        backward_callback=backward_counter,
        callback_frequency=2,
        backend=backend,
    )
    assert forward_counter.count == nt / 2

    if backend == "compiled":
        out[-3].sum().backward()
        assert backward_counter.count == nt / 2
        v.grad.zero_()
        rho.grad.zero_()

    # Test with a frequency that does not divide nt evenly
    forward_counter = Counter()
    backward_counter = None if backend == "eager" else Counter()
    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
        receiver_locations_p=receiver_locations_p,
        forward_callback=forward_counter,
        backward_callback=backward_counter,
        callback_frequency=3,
        backend=backend,
    )
    assert forward_counter.count == (nt + 2) // 3

    if backend == "compiled":
        out[-3].sum().backward()
        assert backward_counter.count == (nt + 2) // 3


def test_acoustic_storage_mode() -> None:
    """Check that the storage mode does not affect the gradient."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = torch.ones(10, 10, device=device) * 1500
    rho = torch.ones(10, 10, device=device) * 2200
    v.requires_grad_()
    rho.requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes_volume_rate = torch.zeros(1, 1, nt, device=device)
    source_amplitudes_volume_rate[0, 0, 5] = 1
    source_locations_volume_rate = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    source_locations_volume_rate[0, 0, 0] = 5
    source_locations_volume_rate[0, 0, 1] = 5
    receiver_locations_p = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    receiver_locations_p[0, 0, 0] = 5
    receiver_locations_p[0, 0, 1] = 5

    def noop(state: deepwave.CallbackState) -> None:
        pass

    out1 = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
        receiver_locations_p=receiver_locations_p,
        forward_callback=noop,
        backward_callback=noop,
    )
    out1[-3].sum().backward()
    grad1_v = v.grad.clone()
    grad1_rho = rho.grad.clone()
    v.grad.zero_()
    rho.grad.zero_()

    out2 = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
        receiver_locations_p=receiver_locations_p,
        storage_mode="disk",
        forward_callback=noop,
        backward_callback=noop,
    )
    out2[-3].sum().backward()
    grad2_v = v.grad.clone()
    grad2_rho = rho.grad.clone()

    assert torch.allclose(grad1_v, grad2_v)
    assert torch.allclose(grad1_rho, grad2_rho)


@pytest.mark.parametrize(
    "backend",
    [
        "eager",
        "compiled",
    ],
)
def test_acoustic_callback_wavefield_shape(backend) -> None:
    """Check that the wavefield has the correct shape for each view."""
    v = torch.ones(10, 10) * 1500
    rho = torch.ones(10, 10) * 2200
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes_volume_rate = torch.zeros(1, 1, nt)
    source_amplitudes_volume_rate[0, 0, 5] = 1
    source_locations_volume_rate = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_volume_rate[0, 0, 0] = 5
    source_locations_volume_rate[0, 0, 1] = 5
    pml_width = 5

    class Checker:
        """A checker class for wavefield shapes."""

        def __call__(self, state: deepwave.CallbackState) -> None:
            """Checks the shape of the wavefield for different views."""
            w_inner = state.get_wavefield("pressure_0", "inner")
            assert w_inner.shape == (1, 10, 10)
            w_pml = state.get_wavefield("pressure_0", "pml")
            assert w_pml.shape == (1, 10 + 2 * pml_width, 10 + 2 * pml_width)
            w_full = state.get_wavefield("pressure_0", "full")
            assert w_full.shape == (1, 10 + 2 * pml_width + 3, 10 + 2 * pml_width + 3)

    deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
        pml_width=pml_width,
        accuracy=4,
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
def test_acoustic_callback_wavefield_modification(backend) -> None:
    """Check that the wavefield can be modified in the forward callback."""
    v = torch.ones(10, 10) * 1500
    rho = torch.ones(10, 10) * 2200
    dx = 5.0
    dt = 0.0001
    nt = 20
    source_amplitudes_volume_rate = torch.zeros(1, 1, nt)
    source_amplitudes_volume_rate[0, 0, 5] = 1
    source_locations_volume_rate = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_volume_rate[0, 0, 0] = 5
    source_locations_volume_rate[0, 0, 1] = 5
    receiver_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations_p[0, 0, 0] = 5
    receiver_locations_p[0, 0, 1] = 5

    class Setter:
        """A setter class for wavefield modification."""

        def __init__(self) -> None:
            self.expected = torch.zeros(nt)

        def __call__(self, state: deepwave.CallbackState) -> None:
            """Modifies the wavefield at a specific location."""
            val = torch.randn(1)
            self.expected[state.step] = val
            p_pml = state.get_wavefield("pressure_0", "pml")
            # The coordinates are relative to the padded model, so we need to
            # add the PML width to get the correct index.
            p_pml[0, 5 + state._pml_width[0], 5 + state._pml_width[1]] = val

    setter = Setter()
    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
        receiver_locations_p=receiver_locations_p,
        forward_callback=setter,
        callback_frequency=1,
        pml_width=5,
        backend=backend,
    )
    assert torch.allclose(out[-3].flatten(), setter.expected)


@pytest.mark.parametrize(
    "backend",
    [
        "eager",
        "compiled",
    ],
)
def test_acoustic_multishot_equivalence(backend) -> None:
    """Check that a do-nothing callback does not change the output.

    Checks it for multiple shots.
    """
    v = torch.ones(10, 10) * 1500
    rho = torch.ones(10, 10) * 2200
    dx = 5.0
    dt = 0.004
    source_amplitudes_volume_rate = torch.zeros(2, 1, 20)
    source_amplitudes_volume_rate[0, 0, 5] = 1
    source_amplitudes_volume_rate[1, 0, 8] = 1
    source_locations_volume_rate = torch.zeros(2, 1, 2, dtype=torch.long)
    source_locations_volume_rate[0, 0, 0] = 5
    source_locations_volume_rate[0, 0, 1] = 5
    source_locations_volume_rate[1, 0, 0] = 3
    source_locations_volume_rate[1, 0, 1] = 3
    receiver_locations_p = torch.zeros(2, 1, 2, dtype=torch.long)
    receiver_locations_p[0, 0, 0] = 5
    receiver_locations_p[0, 0, 1] = 5
    receiver_locations_p[1, 0, 0] = 3
    receiver_locations_p[1, 0, 1] = 3

    def do_nothing(state: deepwave.CallbackState) -> None:
        """A do-nothing callback function."""

    out1 = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
        receiver_locations_p=receiver_locations_p,
        backend=backend,
    )
    out2 = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
        receiver_locations_p=receiver_locations_p,
        forward_callback=do_nothing,
        callback_frequency=1,
        backend=backend,
    )
    # The auxiliary state is one output element holding several tensors.
    for a, b in zip(output_tensors(out1), output_tensors(out2)):
        assert torch.allclose(a, b)


def test_acoustic_callback_gradient_modification() -> None:
    """Check that the gradient can be modified in the backward callback."""
    v = torch.ones(10, 10) * 1500
    rho = torch.ones(10, 10) * 2200
    v.requires_grad_()
    rho.requires_grad_()
    dx = 5.0
    dt = 0.004
    source_amplitudes_volume_rate = torch.zeros(1, 1, 20)
    source_amplitudes_volume_rate[0, 0, 5] = 1
    source_locations_volume_rate = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_volume_rate[0, 0, 0] = 5
    source_locations_volume_rate[0, 0, 1] = 5
    receiver_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations_p[0, 0, 0] = 5
    receiver_locations_p[0, 0, 1] = 5

    def modifier(state: deepwave.CallbackState) -> None:
        # Let's modify gradient of K.
        grad_k = state.get_gradient("K", "pml")
        grad_k *= 2

    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
        receiver_locations_p=receiver_locations_p,
    )
    out[-3].sum().backward()
    grad1 = v.grad.clone()
    v.grad.zero_()
    rho.grad.zero_()
    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
        receiver_locations_p=receiver_locations_p,
        backward_callback=modifier,
        callback_frequency=20,
    )
    out[-3].sum().backward()
    grad2 = v.grad.clone()

    # Zero source location to ignore source contribution
    grad1[
        source_locations_volume_rate[0, 0, 0], source_locations_volume_rate[0, 0, 1]
    ] = 0
    grad2[
        source_locations_volume_rate[0, 0, 0], source_locations_volume_rate[0, 0, 1]
    ] = 0

    # Since K = rho * v^2, dL/dv = dL/dK * dK/dv = dL/dK * 2*rho*v.
    # If we double dL/dK, dL/dv should double.
    assert torch.allclose(grad1 * 2, grad2)


def test_acoustic_callback_equivalence() -> None:
    """Check that a do-nothing callback does not change the output."""
    v = torch.ones(10, 10) * 1500
    rho = torch.ones(10, 10) * 2200
    v.requires_grad_()
    rho.requires_grad_()
    dx = 5.0
    dt = 0.004
    source_amplitudes_volume_rate = torch.zeros(1, 1, 20)
    source_amplitudes_volume_rate[0, 0, 5] = 1
    source_locations_volume_rate = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_volume_rate[0, 0, 0] = 5
    source_locations_volume_rate[0, 0, 1] = 5
    receiver_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations_p[0, 0, 0] = 5
    receiver_locations_p[0, 0, 1] = 5

    def do_nothing(state: deepwave.CallbackState) -> None:
        pass

    out1 = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
        receiver_locations_p=receiver_locations_p,
    )
    out1[-1].sum().backward()
    grad1_v = v.grad.clone()
    grad1_rho = rho.grad.clone()
    v.grad.zero_()
    rho.grad.zero_()

    out2 = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
        receiver_locations_p=receiver_locations_p,
        forward_callback=do_nothing,
        backward_callback=do_nothing,
        callback_frequency=1,
    )
    out2[-1].sum().backward()
    grad2_v = v.grad.clone()
    grad2_rho = rho.grad.clone()
    # The auxiliary state is one output element holding several tensors.
    for a, b in zip(output_tensors(out1), output_tensors(out2)):
        assert torch.allclose(a, b)
    assert torch.allclose(grad1_v, grad2_v)
    assert torch.allclose(grad1_rho, grad2_rho)
    v.grad.zero_()
    rho.grad.zero_()

    # Test with a frequency that does not divide nt evenly
    out3 = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
        receiver_locations_p=receiver_locations_p,
        forward_callback=do_nothing,
        backward_callback=do_nothing,
        callback_frequency=3,
    )
    out3[-1].sum().backward()
    grad3_v = v.grad.clone()
    grad3_rho = rho.grad.clone()
    # The auxiliary state is one output element holding several tensors.
    for x, y in zip(output_tensors(out1), output_tensors(out3)):
        assert torch.allclose(x, y)
    assert torch.allclose(grad1_v, grad3_v)
    assert torch.allclose(grad1_rho, grad3_rho)


def test_acoustic_backward_callback_only_call_count() -> None:
    """Check that the backward callback is called the correct number of times.

    Check it when no forward callback is provided.
    """
    v = torch.ones(10, 10) * 1500
    rho = torch.ones(10, 10) * 2200
    v.requires_grad_()
    rho.requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes_volume_rate = torch.zeros(1, 1, nt)
    source_amplitudes_volume_rate[0, 0, 5] = 1
    source_locations_volume_rate = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_volume_rate[0, 0, 0] = 5
    source_locations_volume_rate[0, 0, 1] = 5
    receiver_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations_p[0, 0, 0] = 5
    receiver_locations_p[0, 0, 1] = 5

    class Counter:
        """A simple counter class for callbacks."""

        def __init__(self) -> None:
            self.count = 0

        def __call__(self, state: deepwave.CallbackState) -> None:
            """Increments the counter."""
            self.count += 1

    # Test with a frequency that divides nt evenly
    backward_counter = Counter()
    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
        receiver_locations_p=receiver_locations_p,
        backward_callback=backward_counter,
        callback_frequency=2,
    )
    out[-3].sum().backward()
    assert backward_counter.count == nt / 2

    # Test with a frequency that does not divide nt evenly
    v.grad.zero_()
    rho.grad.zero_()
    backward_counter = Counter()
    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=source_locations_volume_rate,
        receiver_locations_p=receiver_locations_p,
        backward_callback=backward_counter,
        callback_frequency=3,
    )
    out[-3].sum().backward()
    assert backward_counter.count == (nt + 2) // 3


@pytest.mark.parametrize("ndim", [2, 3])
def test_acoustic_backward_callback_dict_keys(ndim) -> None:
    """Compiled backward-callback wavefield/gradient dict key sets are exact.

    ``state._wavefields`` must equal ``acoustic_wavefield_names(ndim)``
    exactly and ``state._gradients`` must equal ``acoustic_model_names(ndim)``
    exactly (the same bug class fixed for acoustic_vti in commit 837b3cbf,
    which shares ``_engine.run_backward_core`` with acoustic). Only the
    compiled backend is exercised here: ``backend="eager"`` raises
    ``RuntimeError`` for ``backward_callback`` (see ``acoustic_python``), so
    it has no comparable contract to check.
    """
    n = 10
    shape = (n,) * ndim
    v = (torch.ones(*shape) * 1500).requires_grad_()
    rho = (torch.ones(*shape) * 2200).requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 8
    source_amplitudes_volume_rate = torch.zeros(1, 1, nt)
    source_amplitudes_volume_rate[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded: dict[str, dict[str, torch.Tensor]] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded["wavefields"] = dict(state._wavefields)
        recorded["gradients"] = dict(state._gradients)

    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_volume_rate=source_amplitudes_volume_rate,
        source_locations_volume_rate=loc,
        receiver_locations_p=loc,
        pml_width=3,
        backward_callback=recorder,
        callback_frequency=1,
    )
    out[-1].sum().backward()

    assert set(recorded["wavefields"]) == set(acoustic_wavefield_names(ndim))
    assert all(t.numel() > 0 for t in recorded["wavefields"].values())
    assert set(recorded["gradients"]) == set(acoustic_model_names(ndim))
    assert all(t.numel() > 0 for t in recorded["gradients"].values())


@pytest.mark.parametrize("ndim", [2, 3])
def test_acoustic_forward_callback_dict_keys_and_cross_backend_agreement(ndim) -> None:
    """Forward-callback wavefield/model dict keys match spec names on both backends.

    Checks the dict key sets against ``acoustic_wavefield_names``/
    ``acoustic_model_names``, that every value has nonzero ``numel()``, and
    that eager and compiled agree numerically at the final callback call. The
    tolerance (``atol=5e-4 * <value>.abs().max()``) is copied from this file's
    established python-backend-vs-compiled-backend comparisons in
    ``test_direct``/``test_v_batched``/``test_rho_batched``
    (``test_acoustic.py``), the closest existing convention since no
    eager-specific numeric comparison exists there to copy directly (those
    tests compare a ``torch.compile``d Python backend against the compiled
    C backend, not eager).
    """
    n = 10
    shape = (n,) * ndim
    v = torch.full(shape, 1500.0)
    rho = torch.full(shape, 2200.0)
    dx = 5.0
    dt = 0.0005
    nt = 12
    source_amplitudes_volume_rate = torch.zeros(1, 1, nt)
    source_amplitudes_volume_rate[0, 0, 2] = 1.0
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
        "source_amplitudes_volume_rate": source_amplitudes_volume_rate,
        "source_locations_volume_rate": loc,
        "pml_width": 4,
        "callback_frequency": 1,
    }
    deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        backend="eager",
        forward_callback=make_recorder("eager"),
        **common_kwargs,
    )
    deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        backend="compiled",
        forward_callback=make_recorder("compiled"),
        **common_kwargs,
    )

    for key in ("eager", "compiled"):
        assert set(recorded[key]) == set(acoustic_wavefield_names(ndim))
        assert all(t.numel() > 0 for t in recorded[key].values())
        assert set(recorded_models[key]) == set(acoustic_model_names(ndim))
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
