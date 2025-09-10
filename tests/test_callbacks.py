"""Tests for deepwave.callbacks."""

import torch

import deepwave


def test_scalar_callback_call_count() -> None:
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

        def __call__(self, state: deepwave.common.CallbackState) -> None:
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
    )
    out[-1].sum().backward()
    assert forward_counter.count == nt / 2
    assert backward_counter.count == nt / 2

    # Test with a frequency that does not divide nt evenly
    v.grad.zero_()
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
    )
    out[-1].sum().backward()
    assert forward_counter.count == (nt + 2) // 3
    assert backward_counter.count == (nt + 2) // 3


def test_scalar_callback_wavefield_shape() -> None:
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

        def __call__(self, state: deepwave.common.CallbackState) -> None:
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
    )


def test_scalar_callback_wavefield_modification() -> None:
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

        def __call__(self, state: deepwave.common.CallbackState) -> None:
            """Modifies the wavefield at a specific location."""
            val = torch.randn(1)
            self.expected[state.step] = val
            w_pml = state.get_wavefield("wavefield_0", "pml")
            # The coordinates are relative to the padded model, so we need to
            # add the PML width to get the correct index.
            w_pml[0, 5 + state._pml_width[0], 5 + state._pml_width[2]] = val  # noqa: SLF001

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

    def modifier(state: deepwave.common.CallbackState) -> None:
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

    def modifier(state: deepwave.common.CallbackState) -> None:
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

    def do_nothing(state: deepwave.common.CallbackState) -> None:
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
    for i in range(len(out1)):
        assert torch.allclose(out1[i], out2[i])
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
    for i in range(len(out1)):
        assert torch.allclose(out1[i], out3[i])
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

        def __call__(self, state: deepwave.common.CallbackState) -> None:
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

    def do_nothing(state: deepwave.common.CallbackState) -> None:
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
    for i in range(len(out1)):
        assert torch.allclose(out1[i], out2[i])
