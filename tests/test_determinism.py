"""The same call must give the same numbers twice.

Three ways a propagator could fail that, all of which it once did:

* on CUDA, a gradient that differed between runs of the same call;
* on CPU, forward outputs and gradients that differed between thread counts;
* on both, multi-mechanism attenuation whose results depended on what the
  process had allocated and freed earlier, because a buffer somewhere below
  was read before it was written and so picked up whatever the allocator had
  last left there.

All are bitwise assertions. Deepwave does not promise bitwise agreement
between *different* builds, devices, backends or versions -- only that one
build, on one device, gives one answer.

``determinism_cases.py`` builds the calls, one per propagator, from each
propagator's own spec, so a new propagator is covered as soon as it is
registered. ``deepwave.das.gauge_average`` is covered separately: no
propagator calls it, but it is public, differentiable, and pads its input
like the propagators do.
"""

from collections.abc import Iterator

import determinism_cases as dc
import pytest
import torch
from helpers import output_tensors

import deepwave
from deepwave import _gsls

#: How many times the CUDA case is run. The first is the reference.
REPEATS = 10

#: The thread counts the CPU results must agree across. 1, 2 and 3 alone are
#: not enough: a thread count decides where a vectorised loop body ends and
#: its scalar remainder begins, and the small counts tend to agree with each
#: other while a larger one diverges, so the range is sampled further up too.
THREAD_COUNTS = (1, 2, 3, 5, 8, 16)

#: The ``gauge_average`` case, sized to expose a nondeterministic pad: the
#: gauge must cover enough channels for the edge channels' gradients to be
#: summed over many terms. 512 channels averaged over a 64-channel gauge
#: does; 64 channels over a 32-channel gauge does not.
DAS_N_CHANNELS = 512
DAS_NT = 2000
DAS_GAUGE_LENGTH = 64.0


@pytest.fixture
def restore_num_threads() -> Iterator[None]:
    """Puts back the thread count the rest of the suite expects."""
    previous = torch.get_num_threads()
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.gpu
@pytest.mark.parametrize("name", dc.PROPAGATORS)
def test_cuda_gradients_are_reproducible(name: str) -> None:
    """Ten identical backward passes give ten identical sets of gradients."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    device = torch.device("cuda")
    reference = dc.run_case(name, device)
    for repeat in range(1, REPEATS):
        differing = dc.compare(reference, dc.run_case(name, device))
        assert not differing, (
            f"{name}: repeat {repeat} of {REPEATS} differs from the first in "
            f"{differing}"
        )


@pytest.mark.gpu
def test_cuda_gauge_average_gradients_are_reproducible() -> None:
    """Ten identical backward passes through ``gauge_average`` agree bitwise.

    ``gauge_average`` is a public differentiable function that no propagator
    calls, so the cases above do not reach it. It pads the channel axis
    before averaging, and a replicate pad's CUDA backward scatters with an
    ``atomicAdd``.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    device = torch.device("cuda")
    data = torch.randn(1, DAS_N_CHANNELS, DAS_NT, device=device)
    outputs = []
    grads = []
    for _ in range(REPEATS):
        leaf = data.clone().requires_grad_()
        out = deepwave.das.gauge_average(leaf, DAS_GAUGE_LENGTH, 1.0, dim=1)
        loss = (out.to(torch.float64) ** 2).sum()
        (grad,) = torch.autograd.grad(loss, leaf)
        torch.cuda.synchronize()
        outputs.append(out.detach().clone())
        grads.append(grad.detach().clone())
    for repeat in range(1, REPEATS):
        assert torch.equal(outputs[0], outputs[repeat]), (
            f"repeat {repeat} of {REPEATS} differs from the first in the forward output"
        )
        assert torch.equal(grads[0], grads[repeat]), (
            f"repeat {repeat} of {REPEATS} differs from the first in the gradient"
        )


@pytest.mark.parametrize("backend", ["compiled", "eager"])
@pytest.mark.parametrize("name", dc.PROPAGATORS)
def test_cpu_results_do_not_depend_on_thread_count(
    name: str,
    backend: str,
    restore_num_threads: None,
) -> None:
    """Every thread count gives the same forward outputs and gradients."""
    device = torch.device("cpu")
    if backend == "compiled" and not dc.has_compiled_backend(name, device):
        pytest.skip(f"{name} has no compiled CPU backend")
    runs = {}
    for count in THREAD_COUNTS:
        torch.set_num_threads(count)
        runs[count] = dc.run_case(name, device, backend=backend, nt=150)
    assert runs[THREAD_COUNTS[0]].backend == backend
    for count in THREAD_COUNTS[1:]:
        differing = dc.compare(runs[THREAD_COUNTS[0]], runs[count])
        assert not differing, (
            f"{name}/{backend}: {count} threads differ from "
            f"{THREAD_COUNTS[0]} in {differing}"
        )


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scalar_compiled_threads_across_kernel_variants(
    ndim: int,
    accuracy: int,
    dtype: torch.dtype,
    restore_num_threads: None,
) -> None:
    """Irregular domains retain identical scalar adjoints across SIMD chunks."""
    shape = {1: (37,), 2: (23, 31), 3: (17, 19, 23)}[ndim]
    generator = torch.Generator().manual_seed(714)
    speed = 1500 + 250 * torch.rand(shape, generator=generator, dtype=dtype)
    source = torch.randn(2, 1, 24, generator=generator, dtype=dtype)
    # Scalar continuation fields include the six-cell PML on both faces.
    initial = [
        torch.randn(2, *(n + 12 for n in shape), generator=generator, dtype=dtype)
        for _ in range(2)
    ]
    locations = torch.tensor([[[n // 2 for n in shape]]] * 2)
    reference = None
    for count in THREAD_COUNTS:
        torch.set_num_threads(count)
        leaves = [value.clone().requires_grad_() for value in [speed, source, *initial]]
        out = deepwave.scalar(
            leaves[0],
            5.0,
            0.0004,
            source_amplitudes=leaves[1],
            source_locations=locations,
            receiver_locations=locations,
            wavefield_0=leaves[2],
            wavefield_m1=leaves[3],
            origin=[0] * ndim,
            accuracy=accuracy,
            pml_width=6,
            pml_freq=25.0,
            max_vel=2000.0,
            backend="compiled",
        )
        tensors = output_tensors(out)
        loss = sum(tensor.to(torch.float64).square().sum() for tensor in tensors)
        grads = torch.autograd.grad(loss, leaves)
        assert all(torch.isfinite(grad).all() and grad.norm() > 0 for grad in grads)
        current = [tensor.detach().clone() for tensor in [*tensors, *grads]]
        if reference is None:
            reference = current
        else:
            assert all(torch.equal(a, b) for a, b in zip(reference, current)), (
                f"{ndim}D/order {accuracy}/{dtype}: "
                f"{count} threads differ from {THREAD_COUNTS[0]}"
            )


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n_mechanisms", [1, 3])
def test_viscoacoustic_compiled_threads_across_kernel_variants(
    ndim: int,
    accuracy: int,
    dtype: torch.dtype,
    n_mechanisms: int,
    restore_num_threads: None,
) -> None:
    """Irregular SIMD remainders preserve every output and requested adjoint."""
    shape = {1: (37,), 2: (23, 31), 3: (17, 19, 23)}[ndim]
    generator = torch.Generator().manual_seed(714)
    field = torch.rand(shape, generator=generator, dtype=dtype)
    models = [1500 + 250 * field, 1800 + 400 * field, 20 + 30 * field]
    amplitudes = 1e6 * torch.randn(2, 1, 24, generator=generator, dtype=dtype)
    initial_pressure = 1e3 * torch.randn(2, *shape, generator=generator, dtype=dtype)
    locations = torch.tensor([[[n // 2 for n in shape]]] * 2)
    configuration = {"q_freq": 25.0} if n_mechanisms == 1 else {"q_band": (5.0, 60.0)}
    reference = None
    for count in THREAD_COUNTS:
        torch.set_num_threads(count)
        leaves = [
            value.clone().requires_grad_()
            for value in [*models, amplitudes, initial_pressure]
        ]
        out = deepwave.viscoacoustic(
            *leaves[:3],
            5.0,
            0.0004,
            source_amplitudes_pressure_rate=leaves[3],
            source_locations_pressure_rate=locations,
            receiver_locations_p=locations,
            pressure_0=leaves[4],
            origin=[0] * ndim,
            q_n_mechanisms=n_mechanisms,
            accuracy=accuracy,
            pml_width=6,
            pml_freq=25.0,
            max_vel=4000.0,
            backend="compiled",
            **configuration,
        )
        tensors = output_tensors(out)
        loss = sum(tensor.to(torch.float64).square().sum() for tensor in tensors)
        grads = torch.autograd.grad(loss, leaves)
        assert all(torch.isfinite(grad).all() and grad.norm() > 0 for grad in grads)
        current = [tensor.detach().clone() for tensor in [*tensors, *grads]]
        if reference is None:
            reference = current
        else:
            assert all(torch.equal(a, b) for a, b in zip(reference, current)), (
                f"{ndim}D/order {accuracy}/{dtype}/{n_mechanisms} mechanisms: "
                f"{count} threads differ from {THREAD_COUNTS[0]}"
            )


#: The constant-Q band the multi-mechanism cases below fit across, in Hz.
POISON_BAND = (5.0, 60.0)

#: The mechanism counts the fit is checked at. One mechanism takes the exact
#: single-relaxation branch instead, which has no fit to be disturbed.
POISON_MECHANISMS = (2, 3, 4, 6)

#: How many poison-and-repeat rounds each comparison below runs. A stale
#: buffer does not change every result, so one round is not enough to call
#: the behaviour absent.
POISON_ROUNDS = 6

#: What the poisoned blocks are filled with. Large and non-finite values are
#: chosen so that a buffer that is used unwritten shows up as a wildly wrong
#: number rather than a plausible one; the denormal is there because a stale
#: tiny value is the form this failure took when it was real.
_POISON_FILLS = (float("nan"), 1e30, -1e30, 1e-300)

#: The block lengths poisoned, in elements. The spread covers the allocator's
#: small-block free lists as well as whole wavefield-sized buffers.
_POISON_SIZES = (7, 64, 577, 4096, 65536)


def poison_the_allocator() -> None:
    """Fills blocks of many sizes with junk and frees them again.

    A buffer that is read before it is written takes whatever the allocator
    last left there, so anything whose result changes across a call to this
    is reading uninitialised memory. Calling it leaves no lasting state: the
    blocks are freed immediately, and only their contents linger, in memory
    nothing is entitled to read.
    """
    blocks = [
        torch.full((size,), fill, dtype=dtype)
        for size in _POISON_SIZES
        for fill in _POISON_FILLS
        for dtype in (torch.float64, torch.float32)
    ]
    blocks += [
        torch.full((size,), -1, dtype=dtype)
        for size in _POISON_SIZES
        for dtype in (torch.int64, torch.int32)
    ]
    del blocks


@pytest.mark.parametrize("n_mechanisms", POISON_MECHANISMS)
def test_constant_q_fit_does_not_depend_on_earlier_allocations(
    n_mechanisms: int,
) -> None:
    """The multi-mechanism fit gives one answer, whatever ran before it.

    The fit's weights multiply every mechanism's memory-variable
    coefficients, so a last-bit change in them moves the wavefields, the
    receiver amplitudes and the gradients of any attenuating call. It is
    solved from constants alone, so it must give the same weights every
    time; that it once did not was a least-squares driver taking its initial
    column permutation from an array it was handed uninitialised, which made
    the pivot order, and so the rounding, a function of the process's
    allocation history.
    """
    device = torch.device("cpu")
    reference = _gsls.gsls_fit_weights(POISON_BAND, n_mechanisms, torch.float64, device)
    for round_number in range(POISON_ROUNDS):
        poison_the_allocator()
        repeat = _gsls.gsls_fit_weights(
            POISON_BAND, n_mechanisms, torch.float64, device
        )
        for quantity, first, again in zip(
            ("relaxation times", "weights"), reference, repeat
        ):
            assert torch.equal(first, again), (
                f"{n_mechanisms} mechanisms: the fitted {quantity} changed in "
                f"round {round_number} of {POISON_ROUNDS} after the allocator "
                "was poisoned"
            )


@pytest.mark.parametrize(
    "name",
    [
        "viscoacoustic",
    ],
)
def test_attenuating_results_do_not_depend_on_earlier_allocations(
    name: str,
) -> None:
    """Identical multi-mechanism calls agree bitwise across poisoned rounds.

    The propagator-level statement of the test above: the same call, made
    again after the allocator has been filled with junk and emptied, must
    give the same receiver amplitudes and the same gradients.
    """
    device = torch.device("cpu")
    n_mechanisms = 3
    shape = (24, 24)
    ones = torch.ones(shape, dtype=torch.float64, device=device)
    velocity = torch.full(shape, 1500.0, dtype=torch.float64, device=device)
    velocity[12:] = 1700.0
    if name == "viscoacoustic":
        models = [velocity, 2200.0 * ones, 40.0 * ones]
        kwargs = {
            "source_locations_pressure_rate": torch.tensor([[[12, 12]]]),
            "receiver_locations_p": torch.tensor([[[4, 4], [4, 18]]]),
        }
        source_kwarg = "source_amplitudes_pressure_rate"
    else:
        models = [
            2.25e9 * ones,
            1.0e9 * ones,
            (1 / 2200) * ones,
            40.0 * ones,
            30.0 * ones,
        ]
        kwargs = {
            "source_locations_y": torch.tensor([[[12, 12]]]),
            "receiver_locations_y": torch.tensor([[[4, 4], [4, 18]]]),
        }
        source_kwarg = "source_amplitudes_y"
    wavelet = (
        deepwave.wavelets.ricker(15.0, 24, 0.001, 1.0 / 15.0)
        .to(torch.float64)
        .reshape(1, 1, -1)
    )
    # Preserve the old experiment with fixed reference bulk scaling.
    wavelet = wavelet * (2200.0 * 1500.0**2)
    reference = None
    for round_number in range(POISON_ROUNDS):
        poison_the_allocator()
        leaves = [model.clone().requires_grad_() for model in models]
        amplitudes = wavelet.clone().requires_grad_()
        out = getattr(deepwave, name)(
            *leaves,
            5.0,
            0.001,
            **{source_kwarg: amplitudes},
            **kwargs,
            q_n_mechanisms=n_mechanisms,
            q_band=POISON_BAND,
            pml_freq=15.0,
            max_vel=2000.0,
            backend="eager",
        )
        outputs = [
            tensor
            for tensor in out.as_dict().values()
            if torch.is_tensor(tensor) and tensor.numel() > 0
        ]
        loss = sum((tensor**2).sum() for tensor in outputs)
        grads = torch.autograd.grad(loss, [*leaves, amplitudes], allow_unused=True)
        current = [tensor.detach().clone() for tensor in outputs]
        current += [
            torch.zeros(()) if grad is None else grad.detach().clone() for grad in grads
        ]
        if reference is None:
            reference = current
            continue
        differing = [
            index
            for index, (first, again) in enumerate(zip(reference, current))
            if not torch.equal(first, again)
        ]
        assert not differing, (
            f"{name}: round {round_number} of {POISON_ROUNDS} differs from the "
            f"first after the allocator was poisoned, in {differing}"
        )
