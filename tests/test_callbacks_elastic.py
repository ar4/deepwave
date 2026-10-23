"""Tests for deepwave.callbacks_elastic."""

import pytest
import torch
from helpers import output_tensors

import deepwave
from deepwave.elastic import (
    elastic_model_names,
    elastic_output_names,
    elastic_wavefield_names,
)


@pytest.mark.parametrize(
    "backend",
    [
        "eager",
        "compiled",
    ],
)
def test_elastic_callback_call_count(backend) -> None:
    """Check that the callbacks are called the correct number of times."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lamb = torch.ones(10, 10, device=device) * 2200
    mu = torch.ones(10, 10, device=device) * 1000
    buoyancy = torch.ones(10, 10, device=device) * 1 / 2200
    lamb.requires_grad_()
    mu.requires_grad_()
    buoyancy.requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes_y = torch.zeros(1, 1, nt, device=device)
    source_amplitudes_y[0, 0, 5] = 1
    source_locations_y = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    source_locations_y[0, 0, 0] = 5
    source_locations_y[0, 0, 1] = 5
    receiver_locations_y = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    receiver_locations_y[0, 0, 0] = 5
    receiver_locations_y[0, 0, 1] = 5

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
    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        receiver_locations_y=receiver_locations_y,
        forward_callback=forward_counter,
        backward_callback=backward_counter,
        callback_frequency=2,
        backend=backend,
    )
    assert forward_counter.count == nt / 2

    if backend == "compiled":
        out[-2].sum().backward()
        assert backward_counter.count == nt / 2
        lamb.grad.zero_()
        mu.grad.zero_()
        buoyancy.grad.zero_()

    # Test with a frequency that does not divide nt evenly
    forward_counter = Counter()
    backward_counter = None if backend == "eager" else Counter()
    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        receiver_locations_y=receiver_locations_y,
        forward_callback=forward_counter,
        backward_callback=backward_counter,
        callback_frequency=3,
    )
    assert forward_counter.count == (nt + 2) // 3

    if backend == "compiled":
        out[-2].sum().backward()
        assert backward_counter.count == (nt + 2) // 3


def test_elastic_storage_mode() -> None:
    """Check that the storage mode does not affect the gradient."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lamb = torch.ones(10, 10, device=device) * 2200
    mu = torch.ones(10, 10, device=device) * 1000
    buoyancy = torch.ones(10, 10, device=device) * 1 / 2200
    lamb.requires_grad_()
    mu.requires_grad_()
    buoyancy.requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes_y = torch.zeros(1, 1, nt, device=device)
    source_amplitudes_y[0, 0, 5] = 1
    source_locations_y = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    source_locations_y[0, 0, 0] = 5
    source_locations_y[0, 0, 1] = 5
    receiver_locations_y = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    receiver_locations_y[0, 0, 0] = 5
    receiver_locations_y[0, 0, 1] = 5

    def noop(state: deepwave.CallbackState) -> None:
        pass

    out1 = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        receiver_locations_y=receiver_locations_y,
        forward_callback=noop,
        backward_callback=noop,
    )
    out1[-2].sum().backward()
    grad1_lamb = lamb.grad.clone()
    grad1_mu = mu.grad.clone()
    grad1_buoyancy = buoyancy.grad.clone()
    lamb.grad.zero_()
    mu.grad.zero_()
    buoyancy.grad.zero_()

    out2 = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        receiver_locations_y=receiver_locations_y,
        storage_mode="disk",
        forward_callback=noop,
        backward_callback=noop,
    )
    out2[-2].sum().backward()
    grad2_lamb = lamb.grad.clone()
    grad2_mu = mu.grad.clone()
    grad2_buoyancy = buoyancy.grad.clone()

    assert torch.allclose(grad1_lamb, grad2_lamb)
    assert torch.allclose(grad1_mu, grad2_mu)
    assert torch.allclose(grad1_buoyancy, grad2_buoyancy)


@pytest.mark.parametrize(
    "backend",
    [
        "eager",
        "compiled",
    ],
)
def test_elastic_callback_wavefield_shape(backend) -> None:
    """Check that the wavefield has the correct shape for each view."""
    lamb = torch.ones(10, 10) * 2200
    mu = torch.ones(10, 10) * 1000
    buoyancy = torch.ones(10, 10) * 1 / 2200
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes_y = torch.zeros(1, 1, nt)
    source_amplitudes_y[0, 0, 5] = 1
    source_locations_y = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_y[0, 0, 0] = 5
    source_locations_y[0, 0, 1] = 5
    pml_width = 5

    class Checker:
        """A checker class for wavefield shapes."""

        def __call__(self, state: deepwave.CallbackState) -> None:
            """Checks the shape of the wavefield for different views."""
            w_inner = state.get_wavefield("vy_0", "inner")
            assert w_inner.shape == (1, 10, 10)
            w_pml = state.get_wavefield("vy_0", "pml")
            assert w_pml.shape == (1, 10 + 2 * pml_width, 10 + 2 * pml_width)
            w_full = state.get_wavefield("vy_0", "full")
            assert w_full.shape == (1, 10 + 2 * pml_width + 3, 10 + 2 * pml_width + 3)

    deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
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
def test_elastic_callback_wavefield_modification(backend) -> None:
    """Check that the wavefield can be modified in the forward callback."""
    lamb = torch.ones(10, 10) * 2200
    mu = torch.ones(10, 10) * 1000
    buoyancy = torch.ones(10, 10) * 1 / 2200
    dx = 5.0
    dt = 0.0001
    nt = 20
    source_amplitudes_y = torch.zeros(1, 1, nt)
    source_amplitudes_y[0, 0, 5] = 1
    source_locations_y = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_y[0, 0, 0] = 5
    source_locations_y[0, 0, 1] = 5
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
            wy_pml = state.get_wavefield("sigmayy_0", "pml")
            wx_pml = state.get_wavefield("sigmaxx_0", "pml")
            # The coordinates are relative to the padded model, so we need to
            # add the PML width to get the correct index.
            wy_pml[0, 5 + state._pml_width[0], 5 + state._pml_width[2]] = -val
            wx_pml[0, 5 + state._pml_width[0], 5 + state._pml_width[2]] = -val

    setter = Setter()
    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
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
def test_elastic_multishot_equivalence(backend) -> None:
    """Check that a do-nothing callback does not change the output.

    Checks it for multiple shots.
    """
    lamb = torch.ones(10, 10) * 2200
    mu = torch.ones(10, 10) * 1000
    buoyancy = torch.ones(10, 10) * 1 / 2200
    dx = 5.0
    dt = 0.004
    source_amplitudes_y = torch.zeros(2, 1, 20)
    source_amplitudes_y[0, 0, 5] = 1
    source_amplitudes_y[1, 0, 8] = 1
    source_locations_y = torch.zeros(2, 1, 2, dtype=torch.long)
    source_locations_y[0, 0, 0] = 5
    source_locations_y[0, 0, 1] = 5
    source_locations_y[1, 0, 0] = 3
    source_locations_y[1, 0, 1] = 3
    receiver_locations_y = torch.zeros(2, 1, 2, dtype=torch.long)
    receiver_locations_y[0, 0, 0] = 5
    receiver_locations_y[0, 0, 1] = 5
    receiver_locations_y[1, 0, 0] = 3
    receiver_locations_y[1, 0, 1] = 3

    def do_nothing(state: deepwave.CallbackState) -> None:
        """A do-nothing callback function."""

    out1 = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        receiver_locations_y=receiver_locations_y,
        backend=backend,
    )
    out2 = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        receiver_locations_y=receiver_locations_y,
        forward_callback=do_nothing,
        callback_frequency=1,
        backend=backend,
    )
    # The auxiliary state is one output element holding several tensors.
    for x, y in zip(output_tensors(out1), output_tensors(out2)):
        assert torch.allclose(x, y)
    assert torch.allclose(out1[-1], out2[-1])


def test_elastic_callback_gradient_modification() -> None:
    """Check that the gradient can be modified in the backward callback."""
    lamb = torch.ones(10, 10) * 2200
    mu = torch.ones(10, 10) * 1000
    buoyancy = torch.ones(10, 10) * 1 / 2200
    lamb.requires_grad_()
    mu.requires_grad_()
    buoyancy.requires_grad_()
    dx = 5.0
    dt = 0.004
    source_amplitudes_y = torch.zeros(1, 1, 20)
    source_amplitudes_y[0, 0, 5] = 1
    source_locations_y = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_y[0, 0, 0] = 5
    source_locations_y[0, 0, 1] = 5
    receiver_locations_y = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations_y[0, 0, 0] = 5
    receiver_locations_y[0, 0, 1] = 5

    def modifier(state: deepwave.CallbackState) -> None:
        grad_mu = state.get_gradient("mu", "pml")
        grad_mu *= 2

    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        receiver_locations_y=receiver_locations_y,
    )
    out[-2].sum().backward()
    grad1 = mu.grad.clone()
    lamb.grad.zero_()
    mu.grad.zero_()
    buoyancy.grad.zero_()
    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        receiver_locations_y=receiver_locations_y,
        backward_callback=modifier,
        callback_frequency=20,
    )
    out[-2].sum().backward()
    grad2 = mu.grad.clone()
    assert torch.allclose(grad1 * 2, grad2)


def test_elastic_callback_equivalence() -> None:
    """Check that a do-nothing callback does not change the output."""
    lamb = torch.ones(10, 10) * 2200
    mu = torch.ones(10, 10) * 1000
    buoyancy = torch.ones(10, 10) * 1 / 2200
    lamb.requires_grad_()
    mu.requires_grad_()
    buoyancy.requires_grad_()
    dx = 5.0
    dt = 0.004
    source_amplitudes_y = torch.zeros(1, 1, 20)
    source_amplitudes_y[0, 0, 5] = 1
    source_locations_y = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_y[0, 0, 0] = 5
    source_locations_y[0, 0, 1] = 5
    receiver_locations_y = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations_y[0, 0, 0] = 5
    receiver_locations_y[0, 0, 1] = 5

    def do_nothing(state: deepwave.CallbackState) -> None:
        pass

    out1 = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        receiver_locations_y=receiver_locations_y,
    )
    out1[-1].sum().backward()
    grad1_lamb = lamb.grad.clone()
    grad1_mu = mu.grad.clone()
    grad1_buoyancy = buoyancy.grad.clone()
    lamb.grad.zero_()
    mu.grad.zero_()
    buoyancy.grad.zero_()

    out2 = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        receiver_locations_y=receiver_locations_y,
        forward_callback=do_nothing,
        backward_callback=do_nothing,
        callback_frequency=1,
    )
    out2[-1].sum().backward()
    grad2_lamb = lamb.grad.clone()
    grad2_mu = mu.grad.clone()
    grad2_buoyancy = buoyancy.grad.clone()
    # The auxiliary state is one output element holding several tensors.
    for x, y in zip(output_tensors(out1), output_tensors(out2)):
        assert torch.allclose(x, y)
    assert torch.allclose(grad1_lamb, grad2_lamb)
    assert torch.allclose(grad1_mu, grad2_mu)
    assert torch.allclose(grad1_buoyancy, grad2_buoyancy)
    lamb.grad.zero_()
    mu.grad.zero_()
    buoyancy.grad.zero_()

    # Test with a frequency that does not divide nt evenly
    out3 = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        receiver_locations_y=receiver_locations_y,
        forward_callback=do_nothing,
        backward_callback=do_nothing,
        callback_frequency=3,
    )
    out3[-1].sum().backward()
    grad3_lamb = lamb.grad.clone()
    grad3_mu = mu.grad.clone()
    grad3_buoyancy = buoyancy.grad.clone()
    # The auxiliary state is one output element holding several tensors.
    for x, y in zip(output_tensors(out1), output_tensors(out3)):
        assert torch.allclose(x, y)
    assert torch.allclose(grad1_lamb, grad3_lamb)
    assert torch.allclose(grad1_mu, grad3_mu)
    assert torch.allclose(grad1_buoyancy, grad3_buoyancy)


def test_elastic_backward_callback_role_forward_raises() -> None:
    """Elastic's backward_callback cannot request the reconstructed forward field.

    Unlike scalar, elastic's backends do not retain a reconstructed forward
    wavefield for a backward callback to read, so
    ``get_wavefield(role="forward")`` inside an elastic ``backward_callback``
    raises ``NotImplementedError`` naming which propagator does support it.
    """
    lamb = torch.ones(10, 10) * 2200
    mu = torch.ones(10, 10) * 1000
    buoyancy = torch.ones(10, 10) * 1 / 2200
    lamb.requires_grad_()
    mu.requires_grad_()
    buoyancy.requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes_y = torch.zeros(1, 1, nt)
    source_amplitudes_y[0, 0, 5] = 1
    source_locations_y = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_y[0, 0, 0] = 5
    source_locations_y[0, 0, 1] = 5
    receiver_locations_y = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations_y[0, 0, 0] = 5
    receiver_locations_y[0, 0, 1] = 5

    captured = {}

    def bcb(state: deepwave.CallbackState) -> None:
        if "err" not in captured:
            try:
                state.get_wavefield("vy_0", role="forward")
            except NotImplementedError as e:
                captured["err"] = str(e)

    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        receiver_locations_y=receiver_locations_y,
        backward_callback=bcb,
        callback_frequency=2,
    )
    out.receiver_amplitudes_y.sum().backward()
    assert "err" in captured
    assert "scalar" in captured["err"]


def test_elastic_backward_callback_only_call_count() -> None:
    """Check that the backward callback is called the correct number of times.

    Check it when no forward callback is provided.
    """
    lamb = torch.ones(10, 10) * 2200
    mu = torch.ones(10, 10) * 1000
    buoyancy = torch.ones(10, 10) * 1 / 2200
    lamb.requires_grad_()
    mu.requires_grad_()
    buoyancy.requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes_y = torch.zeros(1, 1, nt)
    source_amplitudes_y[0, 0, 5] = 1
    source_locations_y = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_y[0, 0, 0] = 5
    source_locations_y[0, 0, 1] = 5
    receiver_locations_y = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations_y[0, 0, 0] = 5
    receiver_locations_y[0, 0, 1] = 5

    class Counter:
        """A simple counter class for callbacks."""

        def __init__(self) -> None:
            self.count = 0

        def __call__(self, state: deepwave.CallbackState) -> None:
            """Increments the counter."""
            self.count += 1

    # Test with a frequency that divides nt evenly
    backward_counter = Counter()
    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        receiver_locations_y=receiver_locations_y,
        backward_callback=backward_counter,
        callback_frequency=2,
    )
    out[-2].sum().backward()
    assert backward_counter.count == nt / 2

    # Test with a frequency that does not divide nt evenly
    lamb.grad.zero_()
    mu.grad.zero_()
    buoyancy.grad.zero_()
    backward_counter = Counter()
    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        receiver_locations_y=receiver_locations_y,
        backward_callback=backward_counter,
        callback_frequency=3,
    )
    out[-2].sum().backward()
    assert backward_counter.count == (nt + 2) // 3


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_backward_callback_dict_keys(ndim) -> None:
    """The backward-callback dict's key set matches elastic_wavefield_names exactly.

    Previously ``callback_drop_last=1`` dropped the last wavefield name from
    the dict a user's ``backward_callback`` receives (see PR #121).
    """
    n = 10
    shape = (n,) * ndim
    lamb = (torch.ones(*shape) * 2200).requires_grad_()
    mu = (torch.ones(*shape) * 1000).requires_grad_()
    buoyancy = (torch.ones(*shape) * 1 / 2200).requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 8
    source_amplitudes_x = torch.zeros(1, 1, nt)
    source_amplitudes_x[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded_keys: set[str] = set()

    def recorder(state: deepwave.CallbackState) -> None:
        recorded_keys.update(state._wavefields.keys())

    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_x=source_amplitudes_x,
        source_locations_x=loc,
        receiver_locations_x=loc,
        pml_width=3,
        backward_callback=recorder,
        callback_frequency=1,
    )
    out.receiver_amplitudes_x.sum().backward()
    assert recorded_keys == set(elastic_wavefield_names(ndim))


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_backward_callback_model_and_gradient_dict_keys(ndim) -> None:
    """The backward-callback model/gradient dicts' key sets match elastic_model_names.

    ``CallbackState._models``/``_gradients`` are built by
    ``deepwave._engine.run_backward_core`` from ``SPEC.model_names(ndim)``,
    the same mechanism 837b3cb fixed for the wavefield dict (see
    ``test_elastic_backward_callback_dict_keys`` above); the model/gradient
    dicts' key sets were never separately asserted.
    """
    n = 10
    shape = (n,) * ndim
    lamb = (torch.ones(*shape) * 2200).requires_grad_()
    mu = (torch.ones(*shape) * 1000).requires_grad_()
    buoyancy = (torch.ones(*shape) * 1 / 2200).requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 8
    source_amplitudes_x = torch.zeros(1, 1, nt)
    source_amplitudes_x[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded_models: dict[str, torch.Tensor] = {}
    recorded_grads: dict[str, torch.Tensor] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded_models.update(state._models)
        recorded_grads.update(state._gradients)

    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_x=source_amplitudes_x,
        source_locations_x=loc,
        receiver_locations_x=loc,
        pml_width=3,
        backward_callback=recorder,
        callback_frequency=1,
    )
    out.receiver_amplitudes_x.sum().backward()
    expected = set(elastic_model_names(ndim))
    assert set(recorded_models) == expected
    assert set(recorded_grads) == expected
    for name, t in recorded_models.items():
        assert t.numel() > 0, f"{name}: zero numel in model dict"
    for name, t in recorded_grads.items():
        assert t.numel() > 0, f"{name}: zero numel in gradient dict"


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_forward_callback_model_dict_keys(ndim) -> None:
    """The forward-callback model dict's key set matches elastic_model_names.

    Complements ``test_elastic_forward_callback_shear_fields_match_compiled``
    (``tests/test_elastic.py``, added by 837b3cb), which checks the
    forward-callback wavefield dict but not the model dict. The
    forward-callback's gradient dict is always ``{}`` by design (no adjoint
    exists yet during the forward pass), so only the model dict is checked
    here.
    """
    n = 10
    shape = (n,) * ndim
    lamb = torch.ones(*shape) * 2200
    mu = torch.ones(*shape) * 1000
    buoyancy = torch.ones(*shape) * 1 / 2200
    dx = 5.0
    dt = 0.004
    nt = 8
    source_amplitudes_x = torch.zeros(1, 1, nt)
    source_amplitudes_x[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    recorded_models: dict[str, torch.Tensor] = {}

    def recorder(state: deepwave.CallbackState) -> None:
        recorded_models.update(state._models)
        assert state._gradients == {}

    deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_x=source_amplitudes_x,
        source_locations_x=loc,
        receiver_locations_x=loc,
        pml_width=3,
        forward_callback=recorder,
        callback_frequency=1,
    )
    assert set(recorded_models) == set(elastic_model_names(ndim))
    for name, t in recorded_models.items():
        assert t.numel() > 0, f"{name}: zero numel in model dict"


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_output_names(ndim) -> None:
    """PropagatorOutput field names match elastic_output_names exactly."""
    n = 10
    shape = (n,) * ndim
    lamb = torch.ones(*shape) * 2200
    mu = torch.ones(*shape) * 1000
    buoyancy = torch.ones(*shape) * 1 / 2200
    dx = 5.0
    dt = 0.004
    nt = 8
    source_amplitudes_x = torch.zeros(1, 1, nt)
    source_amplitudes_x[0, 0, 3] = 1
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    loc[0, 0, :] = n // 2

    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_x=source_amplitudes_x,
        source_locations_x=loc,
        receiver_locations_x=loc,
        pml_width=3,
    )
    assert out._names == tuple(elastic_output_names(ndim))
