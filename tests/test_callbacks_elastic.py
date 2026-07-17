"""Tests for deepwave.callbacks_elastic."""

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
def test_elastic_callback_call_count(python_backend) -> None:
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

        def __call__(self, state: deepwave.common.CallbackState) -> None:
            """Increments the counter."""
            self.count += 1

    # Test with a frequency that divides nt evenly
    forward_counter = Counter()
    backward_counter = None if python_backend else Counter()
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
        python_backend=python_backend,
    )
    assert forward_counter.count == nt / 2

    if not python_backend:
        out[-2].sum().backward()
        assert backward_counter.count == nt / 2
        lamb.grad.zero_()
        mu.grad.zero_()
        buoyancy.grad.zero_()

    # Test with a frequency that does not divide nt evenly
    forward_counter = Counter()
    backward_counter = None if python_backend else Counter()
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

    if not python_backend:
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

    def noop(state: deepwave.common.CallbackState) -> None:
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
    "python_backend",
    [
        True,
        False,
    ],
)
def test_elastic_callback_wavefield_shape(python_backend) -> None:
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
        python_backend=python_backend,
    )


@pytest.mark.parametrize(
    "python_backend",
    [
        True,
        False,
    ],
)
def test_elastic_callback_wavefield_modification(python_backend) -> None:
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
def test_elastic_multishot_equivalence(python_backend) -> None:
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
        python_backend=python_backend,
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
        python_backend=python_backend,
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


EXPECTED_WAVEFIELD_NAMES_2D = {
    "vy_0",
    "sigmayy_0",
    "sigmaxy_0",
    "m_vyy_0",
    "m_vyx_0",
    "m_vxy_0",
    "m_sigmayyy_0",
    "m_sigmaxyy_0",
    "m_sigmaxyx_0",
    "vx_0",
    "sigmaxx_0",
    "m_vxx_0",
    "m_sigmaxxx_0",
}

EXPECTED_WAVEFIELD_NAMES_3D = {
    "vz_0",
    "sigmazz_0",
    "sigmayz_0",
    "sigmaxz_0",
    "m_vzz_0",
    "m_vzy_0",
    "m_vzx_0",
    "m_vyz_0",
    "m_vxz_0",
    "m_sigmazzz_0",
    "m_sigmayzy_0",
    "m_sigmaxzx_0",
    "m_sigmayzz_0",
    "m_sigmaxzz_0",
    *EXPECTED_WAVEFIELD_NAMES_2D,
}


def _elastic_models(ndim, device=None):
    """Create constant elastic models for the given number of dimensions."""
    shape = (10,) * ndim
    lamb = torch.ones(*shape, device=device) * 2200
    mu = torch.ones(*shape, device=device) * 1000
    buoyancy = torch.ones(*shape, device=device) * 1 / 2200
    return lamb, mu, buoyancy


def _elastic_survey(ndim, nt, device=None):
    """Create a central y-directed point source for the given ndim."""
    source_amplitudes_y = torch.zeros(1, 1, nt, device=device)
    source_amplitudes_y[0, 0, 5] = 1
    source_locations_y = torch.zeros(1, 1, ndim, dtype=torch.long, device=device)
    source_locations_y[0, 0, :] = 5
    return source_amplitudes_y, source_locations_y


@pytest.mark.parametrize("python_backend", [False, "eager"])
@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_callback_wavefield_names(python_backend, ndim) -> None:
    """Check that callbacks expose the full set of non-empty wavefields."""
    lamb, mu, buoyancy = _elastic_models(ndim)
    lamb.requires_grad_()
    mu.requires_grad_()
    buoyancy.requires_grad_()
    dx = 5.0
    dt = 0.004
    nt = 10
    source_amplitudes_y, source_locations_y = _elastic_survey(ndim, nt)
    expected = EXPECTED_WAVEFIELD_NAMES_3D if ndim == 3 else EXPECTED_WAVEFIELD_NAMES_2D

    class Checker:
        """Checks the names and contents of the callback wavefields."""

        def __call__(self, state: deepwave.common.CallbackState) -> None:
            """Check that all expected wavefields are present and non-empty."""
            assert set(state._wavefields.keys()) == expected  # noqa: SLF001
            for name in expected:
                assert state.get_wavefield(name).numel() > 0

    # Use "eager" rather than True/"compile" so name checks are not coupled
    # to process-global torch.compile state.
    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        source_amplitudes_y=source_amplitudes_y,
        source_locations_y=source_locations_y,
        forward_callback=Checker(),
        backward_callback=None if python_backend else Checker(),
        callback_frequency=1,
        python_backend=python_backend,
    )
    if not python_backend:
        out[-1].sum().backward()


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_callback_wavefield_backend_consistency(ndim) -> None:
    """Check that both backends expose the same wavefields in callbacks."""
    lamb, mu, buoyancy = _elastic_models(ndim)
    dx = 5.0
    dt = 0.004
    nt = 10
    source_amplitudes_y, source_locations_y = _elastic_survey(ndim, nt)
    expected = EXPECTED_WAVEFIELD_NAMES_3D if ndim == 3 else EXPECTED_WAVEFIELD_NAMES_2D

    class Recorder:
        """Records the inner view of every wavefield at each callback."""

        def __init__(self) -> None:
            self.records = []

        def __call__(self, state: deepwave.common.CallbackState) -> None:
            """Record a clone of every wavefield."""
            self.records.append(
                {
                    name: state.get_wavefield(name).clone()
                    for name in state._wavefields  # noqa: SLF001
                }
            )

    # Compare the C/CUDA backend against the eager Python reference, not
    # python_backend=True (torch.compile), which can be order-dependent.
    recorders = {}
    for python_backend in [False, "eager"]:
        recorder = Recorder()
        deepwave.elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_y=source_amplitudes_y,
            source_locations_y=source_locations_y,
            forward_callback=recorder,
            callback_frequency=3,
            python_backend=python_backend,
        )
        recorders[python_backend] = recorder

    compiled_records = recorders[False].records
    python_records = recorders["eager"].records
    assert len(compiled_records) == len(python_records)
    assert len(compiled_records) > 1
    for compiled_step, python_step in zip(compiled_records, python_records):
        assert set(compiled_step.keys()) == expected
        assert set(python_step.keys()) == expected
        for name in expected:
            assert torch.allclose(compiled_step[name], python_step[name]), name
    # The simulation should have propagated non-zero shear wavefields, so
    # the comparison above is not trivially comparing zeros.
    shear_name = "sigmaxz_0" if ndim == 3 else "sigmaxy_0"
    assert any(
        step[shear_name].abs().max() > 0 for step in python_records[1:]
    )
