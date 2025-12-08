"""Tests for deepwave.callbacks_acoustic."""

import pytest
import torch

import deepwave


@pytest.mark.parametrize(
    "python_backend",
    [
        True,
        False,
    ],
)
def test_acoustic_callback_call_count(python_backend) -> None:
    """Check that the callbacks are called the correct number of times."""
    v = torch.ones(10, 10) * 1500
    rho = torch.ones(10, 10) * 2200
    v.requires_grad_()
    rho.requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes_p = torch.zeros(1, 1, nt)
    source_amplitudes_p[0, 0, 5] = 1
    source_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_p[0, 0, 0] = 5
    source_locations_p[0, 0, 1] = 5
    receiver_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations_p[0, 0, 0] = 5
    receiver_locations_p[0, 0, 1] = 5

    class Counter:
        """A simple counter class for callbacks."""

        def __init__(self) -> None:
            self.count = 0

        def __call__(self, state: deepwave.common.CallbackState) -> None:
            """Increments the counter."""
            self.count += 1

    # Test with a frequency that divides nt evenly
    forward_counter = Counter()
    backward_counter = None if python_backend else Counter()
    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
        receiver_locations_p=receiver_locations_p,
        forward_callback=forward_counter,
        backward_callback=backward_counter,
        callback_frequency=2,
        python_backend=python_backend,
    )
    assert forward_counter.count == nt / 2

    if not python_backend:
        out[-3].sum().backward()
        assert backward_counter.count == nt / 2
        v.grad.zero_()
        rho.grad.zero_()

    # Test with a frequency that does not divide nt evenly
    forward_counter = Counter()
    backward_counter = None if python_backend else Counter()
    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
        receiver_locations_p=receiver_locations_p,
        forward_callback=forward_counter,
        backward_callback=backward_counter,
        callback_frequency=3,
        python_backend=python_backend,
    )
    assert forward_counter.count == (nt + 2) // 3

    if not python_backend:
        out[-3].sum().backward()
        assert backward_counter.count == (nt + 2) // 3


def test_acoustic_storage_mode() -> None:
    """Check that the storage mode does not affect the gradient."""
    v = torch.ones(10, 10) * 1500
    rho = torch.ones(10, 10) * 2200
    v.requires_grad_()
    rho.requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes_p = torch.zeros(1, 1, nt)
    source_amplitudes_p[0, 0, 5] = 1
    source_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_p[0, 0, 0] = 5
    source_locations_p[0, 0, 1] = 5
    receiver_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations_p[0, 0, 0] = 5
    receiver_locations_p[0, 0, 1] = 5

    def noop(state: deepwave.common.CallbackState) -> None:
        pass

    out1 = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
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
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
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
    "python_backend",
    [
        True,
        False,
    ],
)
def test_acoustic_callback_wavefield_shape(python_backend) -> None:
    """Check that the wavefield has the correct shape for each view."""
    v = torch.ones(10, 10) * 1500
    rho = torch.ones(10, 10) * 2200
    dx = 5.0
    dt = 0.004
    nt = 20
    source_amplitudes_p = torch.zeros(1, 1, nt)
    source_amplitudes_p[0, 0, 5] = 1
    source_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_p[0, 0, 0] = 5
    source_locations_p[0, 0, 1] = 5
    pml_width = 5

    class Checker:
        """A checker class for wavefield shapes."""

        def __call__(self, state: deepwave.common.CallbackState) -> None:
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
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
        pml_width=pml_width,
        accuracy=4,
        forward_callback=Checker(),
        callback_frequency=20,
        python_backend=python_backend,
    )


@pytest.mark.parametrize(
    "python_backend",
    [
        True,
        False,
    ],
)
def test_acoustic_callback_wavefield_modification(python_backend) -> None:
    """Check that the wavefield can be modified in the forward callback."""
    v = torch.ones(10, 10) * 1500
    rho = torch.ones(10, 10) * 2200
    dx = 5.0
    dt = 0.0001
    nt = 20
    source_amplitudes_p = torch.zeros(1, 1, nt)
    source_amplitudes_p[0, 0, 5] = 1
    source_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_p[0, 0, 0] = 5
    source_locations_p[0, 0, 1] = 5
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
            p_pml = state.get_wavefield("pressure_0", "pml")
            # The coordinates are relative to the padded model, so we need to
            # add the PML width to get the correct index.
            p_pml[0, 5 + state._pml_width[0], 5 + state._pml_width[1]] = val  # noqa: SLF001

    setter = Setter()
    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
        receiver_locations_p=receiver_locations_p,
        forward_callback=setter,
        callback_frequency=1,
        pml_width=5,
        python_backend=python_backend,
    )
    assert torch.allclose(out[-3].flatten(), setter.expected)


@pytest.mark.parametrize(
    "python_backend",
    [
        True,
        False,
    ],
)
def test_acoustic_multishot_equivalence(python_backend) -> None:
    """Check that a do-nothing callback does not change the output.

    Checks it for multiple shots.
    """
    v = torch.ones(10, 10) * 1500
    rho = torch.ones(10, 10) * 2200
    dx = 5.0
    dt = 0.004
    source_amplitudes_p = torch.zeros(2, 1, 20)
    source_amplitudes_p[0, 0, 5] = 1
    source_amplitudes_p[1, 0, 8] = 1
    source_locations_p = torch.zeros(2, 1, 2, dtype=torch.long)
    source_locations_p[0, 0, 0] = 5
    source_locations_p[0, 0, 1] = 5
    source_locations_p[1, 0, 0] = 3
    source_locations_p[1, 0, 1] = 3
    receiver_locations_p = torch.zeros(2, 1, 2, dtype=torch.long)
    receiver_locations_p[0, 0, 0] = 5
    receiver_locations_p[0, 0, 1] = 5
    receiver_locations_p[1, 0, 0] = 3
    receiver_locations_p[1, 0, 1] = 3

    def do_nothing(state: deepwave.common.CallbackState) -> None:
        """A do-nothing callback function."""

    out1 = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
        receiver_locations_p=receiver_locations_p,
        python_backend=python_backend,
    )
    out2 = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
        receiver_locations_p=receiver_locations_p,
        forward_callback=do_nothing,
        callback_frequency=1,
        python_backend=python_backend,
    )
    for i in range(len(out1)):
        assert torch.allclose(out1[i], out2[i])


def test_acoustic_callback_gradient_modification() -> None:
    """Check that the gradient can be modified in the backward callback."""
    v = torch.ones(10, 10) * 1500
    rho = torch.ones(10, 10) * 2200
    v.requires_grad_()
    rho.requires_grad_()
    dx = 5.0
    dt = 0.004
    source_amplitudes_p = torch.zeros(1, 1, 20)
    source_amplitudes_p[0, 0, 5] = 1
    source_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_p[0, 0, 0] = 5
    source_locations_p[0, 0, 1] = 5
    receiver_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations_p[0, 0, 0] = 5
    receiver_locations_p[0, 0, 1] = 5

    def modifier(state: deepwave.common.CallbackState) -> None:
        # Let's modify gradient of K.
        grad_k = state.get_gradient("K", "pml")
        grad_k *= 2

    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
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
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
        receiver_locations_p=receiver_locations_p,
        backward_callback=modifier,
        callback_frequency=20,
    )
    out[-3].sum().backward()
    grad2 = v.grad.clone()

    # Zero source location to ignore source contribution
    grad1[source_locations_p[0, 0, 0], source_locations_p[0, 0, 1]] = 0
    grad2[source_locations_p[0, 0, 0], source_locations_p[0, 0, 1]] = 0

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
    source_amplitudes_p = torch.zeros(1, 1, 20)
    source_amplitudes_p[0, 0, 5] = 1
    source_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_p[0, 0, 0] = 5
    source_locations_p[0, 0, 1] = 5
    receiver_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations_p[0, 0, 0] = 5
    receiver_locations_p[0, 0, 1] = 5

    def do_nothing(state: deepwave.common.CallbackState) -> None:
        pass

    out1 = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
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
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
        receiver_locations_p=receiver_locations_p,
        forward_callback=do_nothing,
        backward_callback=do_nothing,
        callback_frequency=1,
    )
    out2[-1].sum().backward()
    grad2_v = v.grad.clone()
    grad2_rho = rho.grad.clone()
    for i in range(len(out1)):
        assert torch.allclose(out1[i], out2[i])
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
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
        receiver_locations_p=receiver_locations_p,
        forward_callback=do_nothing,
        backward_callback=do_nothing,
        callback_frequency=3,
    )
    out3[-1].sum().backward()
    grad3_v = v.grad.clone()
    grad3_rho = rho.grad.clone()
    for i in range(len(out1)):
        assert torch.allclose(out1[i], out3[i])
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
    source_amplitudes_p = torch.zeros(1, 1, nt)
    source_amplitudes_p[0, 0, 5] = 1
    source_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations_p[0, 0, 0] = 5
    source_locations_p[0, 0, 1] = 5
    receiver_locations_p = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations_p[0, 0, 0] = 5
    receiver_locations_p[0, 0, 1] = 5

    class Counter:
        """A simple counter class for callbacks."""

        def __init__(self) -> None:
            self.count = 0

        def __call__(self, state: deepwave.common.CallbackState) -> None:
            """Increments the counter."""
            self.count += 1

    # Test with a frequency that divides nt evenly
    backward_counter = Counter()
    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
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
        source_amplitudes_p=source_amplitudes_p,
        source_locations_p=source_locations_p,
        receiver_locations_p=receiver_locations_p,
        backward_callback=backward_counter,
        callback_frequency=3,
    )
    out[-3].sum().backward()
    assert backward_counter.count == (nt + 2) // 3
