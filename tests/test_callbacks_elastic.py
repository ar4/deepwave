"""Tests for deepwave.callbacks_elastic."""

import torch

import deepwave


def test_elastic_callback_call_count() -> None:
    """Check that the callbacks are called the correct number of times."""
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

        def __call__(self, state: deepwave.common.CallbackState) -> None:
            """Increments the counter."""
            self.count += 1

    # Test with a frequency that divides nt evenly
    forward_counter = Counter()
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
        forward_callback=forward_counter,
        backward_callback=backward_counter,
        callback_frequency=2,
    )
    out[-1].sum().backward()
    assert forward_counter.count == nt / 2
    assert backward_counter.count == nt / 2

    # Test with a frequency that does not divide nt evenly
    lamb.grad.zero_()
    mu.grad.zero_()
    buoyancy.grad.zero_()
    forward_counter = Counter()
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
        forward_callback=forward_counter,
        backward_callback=backward_counter,
        callback_frequency=3,
    )
    out[-1].sum().backward()
    assert forward_counter.count == (nt + 2) // 3
    assert backward_counter.count == (nt + 2) // 3


def test_elastic_callback_wavefield_shape() -> None:
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

        def __call__(self, state: deepwave.common.CallbackState) -> None:
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
    )


def test_elastic_callback_wavefield_modification() -> None:
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

        def __call__(self, state: deepwave.common.CallbackState) -> None:
            """Modifies the wavefield at a specific location."""
            val = torch.randn(1)
            self.expected[state.step] = val
            wy_pml = state.get_wavefield("sigmayy_0", "pml")
            wx_pml = state.get_wavefield("sigmaxx_0", "pml")
            # The coordinates are relative to the padded model, so we need to
            # add the PML width to get the correct index.
            wy_pml[0, 5 + state._pml_width[0], 5 + state._pml_width[2]] = -val  # noqa: SLF001
            wx_pml[0, 5 + state._pml_width[0], 5 + state._pml_width[2]] = -val  # noqa: SLF001

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
    )
    assert torch.allclose(out[-3].flatten(), setter.expected)


def test_elastic_multishot_equivalence() -> None:
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

    def do_nothing(state: deepwave.common.CallbackState) -> None:
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
    )
    for i in range(len(out1)):
        assert torch.allclose(out1[i], out2[i])
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

    def modifier(state: deepwave.common.CallbackState) -> None:
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
    out[-1].sum().backward()
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
    out[-1].sum().backward()
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

    def do_nothing(state: deepwave.common.CallbackState) -> None:
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
    for i in range(len(out1)):
        assert torch.allclose(out1[i], out2[i])
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
    for i in range(len(out1)):
        assert torch.allclose(out1[i], out3[i])
    assert torch.allclose(grad1_lamb, grad3_lamb)
    assert torch.allclose(grad1_mu, grad3_mu)
    assert torch.allclose(grad1_buoyancy, grad3_buoyancy)


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

        def __call__(self, state: deepwave.common.CallbackState) -> None:
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
    out[-1].sum().backward()
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
    out[-1].sum().backward()
    assert backward_counter.count == (nt + 2) // 3
