"""Tests for deepwave.callbacks_born."""

from typing import Any

import pytest
import torch
from helpers import output_tensors

import deepwave
from deepwave.scalar_born import scalar_born_model_names, scalar_born_wavefield_names


@pytest.mark.parametrize(
    "backend",
    [
        "eager",
        "compiled",
    ],
)
def test_scalar_born_callback_call_count(backend) -> None:
    """Check that the callbacks are called the correct number of times."""
    v = torch.ones(10, 10) * 1500
    scatter = torch.ones(10, 10)
    v.requires_grad_()
    scatter.requires_grad_()
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
    backward_counter = None if backend == "eager" else Counter()
    out = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
        forward_callback=forward_counter,
        backward_callback=backward_counter,
        callback_frequency=2,
        backend=backend,
    )
    assert forward_counter.count == nt / 2

    if backend == "compiled":
        out[-1].sum().backward()
        assert backward_counter.count == nt / 2
        v.grad.zero_()
        scatter.grad.zero_()

    # Test with a frequency that does not divide nt evenly
    forward_counter = Counter()
    backward_counter = None if backend == "eager" else Counter()
    out = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
        forward_callback=forward_counter,
        backward_callback=backward_counter,
        callback_frequency=3,
    )
    assert forward_counter.count == (nt + 2) // 3

    if backend == "compiled":
        out[-1].sum().backward()
        assert backward_counter.count == (nt + 2) // 3


@pytest.mark.parametrize(
    "backend",
    [
        "eager",
        "compiled",
    ],
)
def test_scalar_born_callback_wavefield_shape(backend) -> None:
    """Check that the wavefield has the correct shape for each view."""
    v = torch.ones(10, 10) * 1500
    scatter = torch.ones(10, 10)
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
            w_inner = state.get_wavefield("wavefield_sc_0", "inner")
            assert w_inner.shape == (1, 10, 10)
            w_pml = state.get_wavefield("wavefield_sc_0", "pml")
            assert w_pml.shape == (1, 10 + 2 * pml_width, 10 + 2 * pml_width)
            w_full = state.get_wavefield("wavefield_sc_0", "full")
            assert w_full.shape == (
                1,
                10 + 2 * pml_width + 2 * 4,
                10 + 2 * pml_width + 2 * 4,
            )

    deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
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
def test_scalar_born_callback_wavefield_modification(backend) -> None:
    """Check that the wavefield can be modified in the forward callback."""
    v = torch.ones(10, 10) * 1500
    scatter = torch.ones(10, 10)
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
            w_pml = state.get_wavefield("wavefield_sc_0", "pml")
            # The coordinates are relative to the padded model, so we need to
            # add the PML width to get the correct index.
            w_pml[0, 5 + state._pml_width[0], 5 + state._pml_width[2]] = val

    setter = Setter()
    out = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
        forward_callback=setter,
        callback_frequency=1,
        pml_width=5,
        backend=backend,
    )
    assert torch.allclose(out[-1].flatten(), setter.expected)


@pytest.mark.parametrize(
    "backend",
    [
        "eager",
        "compiled",
    ],
)
def test_scalar_born_state_preservation(backend) -> None:
    """Check that the wavefield state is preserved between callbacks."""
    v = torch.ones(10, 10) * 1500
    scatter = torch.ones(10, 10)
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

    deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        forward_callback=StateChecker(),
        callback_frequency=1,
        backend=backend,
    )


def test_scalar_born_callback_gradient_modification() -> None:
    """Check that the gradient can be modified in the backward callback."""
    v = torch.ones(10, 10) * 1500
    scatter = torch.ones(10, 10)
    v.requires_grad_()
    scatter.requires_grad_()
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
        grad_scatter = state.get_gradient("dv", "pml")
        grad_scatter *= 2

    out = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
    )
    out[-1].sum().backward()
    grad1 = scatter.grad.clone()
    v.grad.zero_()
    scatter.grad.zero_()
    out = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
        backward_callback=modifier,
        callback_frequency=20,
    )
    out[-1].sum().backward()
    grad2 = scatter.grad.clone()
    # Zero source location to ignore source contribution
    grad1[source_locations[0, 0, 0], source_locations[0, 0, 1]] = 0
    grad2[source_locations[0, 0, 0], source_locations[0, 0, 1]] = 0
    assert torch.allclose(grad1 * 2, grad2)


def test_scalar_born_callback_equivalence() -> None:
    """Check that a do-nothing callback does not change the output."""
    v = torch.ones(10, 10) * 1500
    scatter = torch.ones(10, 10)
    v.requires_grad_()
    scatter.requires_grad_()
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
        pass

    out1 = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
    )
    out1[-1].sum().backward()
    grad1_v = v.grad.clone()
    grad1_scatter = scatter.grad.clone()
    v.grad.zero_()
    scatter.grad.zero_()

    out2 = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
        forward_callback=do_nothing,
        backward_callback=do_nothing,
        callback_frequency=1,
    )
    out2[-1].sum().backward()
    grad2_v = v.grad.clone()
    grad2_scatter = scatter.grad.clone()
    # The auxiliary state is one output element holding several tensors.
    for x, y in zip(output_tensors(out1), output_tensors(out2)):
        assert torch.allclose(x, y)
    assert torch.allclose(grad1_v, grad2_v)
    assert torch.allclose(grad1_scatter, grad2_scatter)


def test_scalar_born_storage_mode() -> None:
    """Check that the storage mode does not affect the gradient."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = torch.ones(10, 10, device=device) * 1500
    scatter = torch.ones(10, 10, device=device)
    v.requires_grad_()
    scatter.requires_grad_()
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

    out1 = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
        forward_callback=noop,
        backward_callback=noop,
    )
    out1[-1].sum().backward()
    grad1_v = v.grad.clone()
    grad1_scatter = scatter.grad.clone()
    v.grad.zero_()
    scatter.grad.zero_()

    out2 = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
        storage_mode="disk",
        forward_callback=noop,
        backward_callback=noop,
    )
    out2[-1].sum().backward()
    grad2_v = v.grad.clone()
    grad2_scatter = scatter.grad.clone()

    assert torch.allclose(grad1_v, grad2_v)
    assert torch.allclose(grad1_scatter, grad2_scatter)


def test_scalar_born_backward_callback_only_call_count() -> None:
    """Check that the backward callback is called the correct number of times.

    Check it when no forward callback is provided.
    """
    v = torch.ones(10, 10) * 1500
    scatter = torch.ones(10, 10)
    v.requires_grad_()
    scatter.requires_grad_()
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
    out = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
        backward_callback=backward_counter,
        callback_frequency=2,
    )
    out[-1].sum().backward()
    grad1_scatter = scatter.grad.detach()
    assert backward_counter.count == nt / 2

    # Test with a frequency that does not divide nt evenly
    v.grad.zero_()
    scatter.grad.zero_()
    backward_counter = Counter()
    out = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
        backward_callback=backward_counter,
        callback_frequency=3,
    )
    out[-1].sum().backward()
    grad2_scatter = scatter.grad.detach()
    assert backward_counter.count == (nt + 2) // 3
    assert torch.allclose(grad1_scatter, grad2_scatter)


def test_scalar_born_multishot_equivalence() -> None:
    """Check that a do-nothing callback does not change the output.

    Checks it for multiple shots.
    """
    v = torch.ones(10, 10) * 1500
    scatter = torch.ones(10, 10)
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
        pass

    out1 = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
    )
    out2 = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        sc_receiver_locations=receiver_locations,
        forward_callback=do_nothing,
        callback_frequency=1,
    )
    # The auxiliary state is one output element holding several tensors.
    for x, y in zip(output_tensors(out1), output_tensors(out2)):
        assert torch.allclose(x, y)


@pytest.mark.parametrize("ndim", [2, 3])
def test_scalar_born_backward_callback_dict_keys(ndim) -> None:
    """Compiled backward-callback wavefield/gradient dict key sets are exact.

    ``state._wavefields`` must equal ``scalar_born_wavefield_names(ndim)``
    exactly and ``state._gradients`` must equal ``scalar_born_model_names(ndim)``
    exactly (the same bug class fixed for elastic/elastic_vti/acoustic_vti in
    commit 837b3cbf, which shares ``_engine.run_backward_core`` with
    scalar_born). Only the compiled backend is exercised here:
    ``backend="eager"`` raises ``RuntimeError`` for
    ``backward_callback`` (see ``scalar_born_python``), so it has no
    comparable contract to check.

    Both ``v`` and ``scatter`` are set to require grad so that
    ``ScalarBornBackwardFunc.setup_context``'s ``non_sc`` optimisation takes
    the full background-and-scattered adjoint path (see ``scalar_born.py``):
    when only ``scatter`` requires grad, the backward-callback dict is a
    deliberately narrower, correct subset (only the scattered-field names,
    ``scalar_born_wavefield_names(ndim)[2 + 2 * ndim:]``), not a bug.
    """
    n = 10
    shape = (n,) * ndim
    v = (torch.ones(*shape) * 1500).requires_grad_()
    scatter = torch.ones(*shape).requires_grad_()
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

    out = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        source_amplitudes=source_amplitudes,
        source_locations=loc,
        sc_receiver_locations=loc,
        pml_width=3,
        backward_callback=recorder,
        callback_frequency=1,
    )
    out[-1].sum().backward()

    assert set(recorded["wavefields"]) == set(scalar_born_wavefield_names(ndim))
    assert all(t.numel() > 0 for t in recorded["wavefields"].values())
    assert set(recorded["gradients"]) == set(scalar_born_model_names(ndim))
    assert all(t.numel() > 0 for t in recorded["gradients"].values())


@pytest.mark.parametrize("ndim", [2, 3])
def test_scalar_born_forward_callback_dict_keys_and_cross_backend_agreement(
    ndim,
) -> None:
    """Forward-callback wavefield/model dict keys match spec names on both backends.

    Checks the dict key sets against ``scalar_born_wavefield_names``/
    ``scalar_born_model_names``, that every value has nonzero ``numel()``, and
    that eager and compiled agree numerically at the final callback call. The
    tolerance (``atol=1e-3``, unscaled) is copied verbatim from this file's
    own established python-backend-vs-compiled-backend comparison in
    ``test_born_scatter`` (``test_scalar_born.py``), the closest existing
    convention since no eager-specific numeric comparison exists there to
    copy directly (that comparison is between a ``torch.compile``d Python
    backend and the compiled C backend, not eager). The source amplitude
    is scaled up so the recorded wavefields reach an order of magnitude
    comparable to that established test's own receiver-amplitude signal
    (roughly 0.1-1): at the unscaled amplitude used by the rest of this file
    (order 1), the wavefields a short 12-step run reaches are themselves only
    order 1e-3, making an unscaled ``atol=1e-3`` almost as large as the
    signal -- large enough to silently pass a genuine field-swap bug (verified
    directly: this test's negative control needed this rescaling to actually
    go red).
    """
    n = 10
    shape = (n,) * ndim
    v = torch.full(shape, 1500.0)
    scatter = torch.full(shape, 1.0)
    dx = 5.0
    dt = 0.0005
    nt = 12
    source_amplitudes = torch.zeros(1, 1, nt)
    source_amplitudes[0, 0, 2] = 1000.0
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
        "dv": scatter,
        "source_amplitudes": source_amplitudes,
        "source_locations": loc,
        "pml_width": 4,
        "callback_frequency": 1,
    }
    deepwave.scalar_born(
        v,
        dx,
        dt,
        backend="eager",
        forward_callback=make_recorder("eager"),
        **common_kwargs,
    )
    deepwave.scalar_born(
        v,
        dx,
        dt,
        backend="compiled",
        forward_callback=make_recorder("compiled"),
        **common_kwargs,
    )

    for key in ("eager", "compiled"):
        assert set(recorded[key]) == set(scalar_born_wavefield_names(ndim))
        assert all(t.numel() > 0 for t in recorded[key].values())
        assert set(recorded_models[key]) == set(scalar_born_model_names(ndim))
        assert all(t.numel() > 0 for t in recorded_models[key].values())

    eager_dict = recorded["eager"]
    compiled_dict = recorded["compiled"]
    for name, eager_val in eager_dict.items():
        compiled_val = compiled_dict[name]
        assert eager_val.shape == compiled_val.shape, (
            f"{name}: shape mismatch {eager_val.shape} vs {compiled_val.shape}"
        )
        assert torch.allclose(eager_val, compiled_val, atol=1e-3), (
            f"{name}: eager/compiled mismatch"
        )
