"""How exact the model gradients are, and the conditions for it.

Deepwave's reverse-mode gradients are the exact adjoint of the **discrete**
scheme it ran, for fixed numerical parameters. Three of those parameters are
themselves derived from the inputs and are then held fixed:

* the maximum wave speed, which sets the absorbing boundary's profiles and,
  through the CFL condition, the internal time step. When ``max_vel`` is
  given it is a constant of the call and the gradient is exact everywhere.
  When it is omitted it is read from the model, so the model value at the
  cell the maximum comes from also acts through the profiles and the time
  step -- a path the gradient does not follow. The gradient at that one cell
  is therefore incomplete; everywhere else it is unaffected;
* ``out.info.step_ratio``, an integer, so it is locally constant in the model
  and contributes nothing except at the isolated models where it steps;
* ``pml_freq``, when omitted, inferred from the source amplitudes' spectrum,
  so gradients with respect to the source amplitudes miss that path too.

The tests below pin all of that:

* :func:`test_taylor_remainder_is_second_order` -- a directional Taylor test
  along a direction that is zero at the cell the maximum comes from, with
  ``max_vel`` omitted. The second-order remainder must quarter at each
  halving of the step, which it can only do if the gradient is the exact
  derivative of what was run. The resolved maximum and step ratio are
  asserted identical at every step, so the test cannot pass by comparing
  different discretisations;
* :func:`test_cell_local_gradient_matches_finite_differences` -- with
  ``max_vel`` given, the gradient at the cell the maximum comes from, at a
  corner, at an edge, at a source cell and at a receiver cell agrees with a
  central finite difference of the same call to 1e-6 relative;
* :func:`test_omitting_max_vel_leaves_every_other_cell_exact` and
  :func:`test_inferring_max_vel_costs_the_gradient_at_the_cell_it_is_read_from`
  -- with ``max_vel`` omitted, that agreement still holds at every cell but
  the one the maximum is read from, and against an objective the absorbing
  profiles dominate it fails there, which is what makes the condition above
  non-vacuous;
* :func:`test_one_sided_difference_at_a_zero_shear_modulus` -- the same check
  where ``mu = 0`` (a fluid), where only a one-sided difference is available;
* the ``vmap`` tests -- a batched model the maximum is read from requires
  ``max_vel``, and with it given the batched run is the per-sample loop
  bitwise;
* the ``jvp`` tests -- the forward-mode PML caveat warning fires only when
  the profiles really do depend on the model, and the forward-mode tangent
  inside the absorbing band is the derivative of the call when they do not.

Every case is float64 on the CPU, on a small grid: these are exactness
questions, not physics ones.
"""

import contextlib
import inspect
import math
import warnings
from collections.abc import Iterator
from typing import Any, Callable, Optional

import pytest
import torch

import deepwave

DTYPE = torch.float64
DEVICE = torch.device("cpu")

#: The nine propagators.
PROPAGATORS: tuple[str, ...] = deepwave._engine.PROPAGATOR_NAMES

N = 16
DX = 5.0
DT = 0.001
NT = 30
FREQ = 25.0
PML_WIDTH = 3
ACCURACY = 4

#: The cell whose material is the fastest, by a clear margin, so that it is
#: the cell the maximum wave speed is read from whichever models a
#: propagator reads it from.
FASTEST_CELL = (9, 6)

#: How much faster that cell is than every other, in m/s. Comfortably more
#: than the 1 m/s the Taylor direction's step size is bounded against.
FASTEST_GAP = 20.0

#: Relative agreement required between the adjoint gradient and a finite
#: difference of the same call. The measured agreement is ~1e-9.
RTOL = 1e-6


def _base_velocity() -> torch.Tensor:
    """A smooth P velocity field with a unique, clearly fastest cell."""
    ramp = torch.linspace(0.0, 1.0, N, dtype=DTYPE)
    field = ramp.reshape(-1, 1) + 0.5 * ramp.reshape(1, -1)
    v = 1500.0 + 40.0 * field / 1.5
    v[FASTEST_CELL] = float(v.max()) + FASTEST_GAP
    return v


#: The density, which does not vary with the velocity field below, so that
#: perturbing that field is a pure wave-speed perturbation.
def _density() -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, N, dtype=DTYPE)
    field = ramp.reshape(1, -1) + 0.5 * ramp.reshape(-1, 1)
    return 1800.0 + 200.0 * field / 1.5


def _models(name: str, v: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every model ``name`` takes, derived from one P velocity field.

    Deriving them all from ``v`` means a perturbation of ``v`` at a cell is a
    perturbation of that cell's wave speed whichever models the propagator
    takes, so one direction and one finite-difference step serve all nine
    -- and the gradient the test checks is the one an inversion for velocity
    would use, assembled from every model's gradient by the chain rule.
    """
    rho = _density()
    vs = v / 2.0
    buoyancy = 1.0 / rho
    mu = rho * vs**2
    lamb = rho * (v**2 - 2.0 * vs**2)
    c33 = rho * v**2
    c11 = 1.1 * c33
    c55 = rho * vs**2
    c13 = c33 - 2.0 * c55
    constant = torch.full((N, N), 1.0, dtype=DTYPE)
    per_name: dict[str, torch.Tensor] = {
        "v": v,
        "rho": rho,
        "q": 40.0 * constant,
        "qp": 40.0 * constant,
        "qs": 30.0 * constant,
        "lamb": lamb,
        "mu": mu,
        "buoyancy": buoyancy,
        "c11": c11,
        "c13": c13,
        "c33": c33,
        "c55": c55,
        "tilt": 0.1 * constant,
    }
    per_name.update(_scatterers())
    return {model: per_name[model] for model in deepwave.spec(name).model_names(2)}


def _scatterers() -> dict[str, torch.Tensor]:
    """The Born propagators' perturbations: a fixed, separate input.

    Built from the unperturbed velocity field rather than from the field
    passed to :func:`_models`, so that perturbing that field is a
    perturbation of the background alone. A scatterer that moved with it
    would make a finite difference of the call differ from the gradient with
    respect to the background for a reason that has nothing to do with the
    propagator.
    """
    v = _base_velocity()
    rho = _density()
    vs = v / 2.0
    return {
        "dv": 0.01 * v,
        "drho": 0.01 * rho,
        "dlamb": 0.01 * rho * (v**2 - 2.0 * vs**2),
        "dmu": 0.01 * rho * vs**2,
        "dbuoyancy": 0.01 / rho,
    }


def _wavelet(nt: int) -> torch.Tensor:
    return (
        deepwave.wavelets.ricker(FREQ, nt, DT, 1.5 / FREQ).to(DTYPE).reshape(1, 1, -1)
    )


def _call(
    name: str,
    v: torch.Tensor,
    receiver_cells: list[tuple[int, int]],
    source_cell: tuple[int, int],
    *,
    nt: int = NT,
    models: Optional[dict[str, torch.Tensor]] = None,
    **overrides: Any,
) -> Any:
    """Run ``name`` on the shared small grid, returning its output tuple.

    The models come from :func:`_models` (or are supplied whole), the
    keyword names all come from the propagator's spec, and ``overrides``
    layers anything else (``max_vel``, ``backend``) on top.
    """
    spec = deepwave.spec(name)
    function = getattr(deepwave, name)
    model_values = _models(name, v) if models is None else models
    parameters = list(inspect.signature(function).parameters)
    positional_names = parameters[: parameters.index("grid_spacing")]
    positional = [
        model_values[model] for model in positional_names if model in model_values
    ]
    kwargs: dict[str, Any] = {
        model: value
        for model, value in model_values.items()
        if model not in positional_names
    }
    kwargs.update(
        {
            "pml_width": PML_WIDTH,
            "accuracy": ACCURACY,
            "pml_freq": FREQ,
        }
    )
    amplitudes = _wavelet(nt)
    for index, channel in enumerate(spec.source_channels(2)):
        # Two sources of the same channel may not share a cell; giving every
        # channel its own cell keeps one rule for all nine.
        cell = (source_cell[0], source_cell[1] + index)
        kwargs[spec.source_amplitudes_kwarg(channel)] = amplitudes
        kwargs[spec.source_locations_kwarg(channel)] = torch.tensor([[list(cell)]])
    locations = torch.tensor([[list(cell) for cell in receiver_cells]])
    channel = spec.receiver_channels(2)[0]
    kwargs[spec.receiver_locations_kwarg(channel)] = locations
    if spec.has_bg_receivers:
        bg_channel = spec.bg_receiver_channels(2)[0]
        kwargs[spec.bg_receiver_locations_kwarg(bg_channel)] = locations
    if name in ("viscoacoustic",):
        kwargs["q_freq"] = FREQ
    kwargs.update(overrides)
    return function(*positional, DX, DT, **kwargs)


def _receivers(name: str, out: Any) -> torch.Tensor:
    """The receiver amplitudes :func:`_call` asked for."""
    spec = deepwave.spec(name)
    channel = spec.receiver_channels(2)[0]
    amplitudes: torch.Tensor = getattr(out, spec.receiver_output_name(channel))
    return amplitudes


def _weights(n_receivers: int, nt: int) -> torch.Tensor:
    """Fixed random weights, so the losses below are not sums of the traces."""
    generator = torch.Generator().manual_seed(20240)
    return torch.randn((n_receivers, nt), generator=generator, dtype=DTYPE)


def _source_cell(name: str) -> tuple[int, int]:
    """A source cell in the interior, away from the fastest cell."""
    del name
    return (N // 2, N // 2 - 2)


def _probe_cells(name: str) -> dict[str, tuple[int, int]]:
    """The cells the cell-local checks probe, each with a receiver on it."""
    return {
        "fastest": FASTEST_CELL,
        "corner": (0, 0),
        "edge": (0, N // 2),
        "source": _source_cell(name),
        "receiver": (N - 4, N - 5),
    }


def _cell_losses(
    name: str, v: torch.Tensor, **overrides: Any
) -> tuple[list[torch.Tensor], dict[str, int]]:
    """One localised loss per probe cell, plus the cell-name to index map.

    Each probe cell carries a receiver of its own and its loss reads only
    that receiver, so the loss is of the same order as its sensitivity to
    that cell's material. A single loss over every receiver would be
    dominated by the receiver nearest the source, and a finite difference of
    it at a cell inside the absorbing boundary -- whose influence on that
    receiver is ~1e-7 of the interior's -- would be pure round-off.
    """
    cells = _probe_cells(name)
    order = list(cells)
    out = _call(name, v, [cells[key] for key in order], _source_cell(name), **overrides)
    amplitudes = _receivers(name, out)
    weights = _weights(len(order), amplitudes.shape[-1])
    losses = [
        (amplitudes[0, index] * weights[index]).sum() for index in range(len(order))
    ]
    return losses, {key: index for index, key in enumerate(order)}


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Suppress the resolution warnings these tests are not about."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------------------
# (a) The Taylor test
# ---------------------------------------------------------------------------


def _taylor_direction() -> torch.Tensor:
    """A random direction that is exactly zero at the fastest cell.

    Zero there because that is the one cell whose value the resolved maximum
    wave speed depends on: a direction that does not move it leaves the
    absorbing profiles and the internal time step alone, so the gradient
    along it is exact even with ``max_vel`` omitted -- which is what the
    remainder ratio then measures.
    """
    generator = torch.Generator().manual_seed(7)
    direction = torch.randn((N, N), generator=generator, dtype=DTYPE)
    direction /= direction.abs().max()
    direction[FASTEST_CELL] = 0.0
    return direction


@pytest.mark.parametrize("name", PROPAGATORS)
def test_taylor_remainder_is_second_order(name: str) -> None:
    """``|J(m+hd) - J(m) - h<g,d>|`` quarters at each halving of ``h``.

    A first-order remainder (ratio 2) is what a wrong or incomplete gradient
    gives; a second-order one can only come from the exact derivative of the
    function that was evaluated. ``max_vel`` is omitted, so this is also the
    statement that inferring it costs nothing away from the one cell it is
    read from.
    """
    v = _base_velocity()
    direction = _taylor_direction()
    # The fastest cell leads the rest by FASTEST_GAP; keeping the step below
    # half of that keeps it the fastest cell at every step, so every
    # evaluation uses one discretisation.
    assert direction.abs().max().item() * 2.0 < FASTEST_GAP / 2.0
    cells = list(_probe_cells(name).values())
    with _quiet():
        leaf = v.clone().requires_grad_(True)
        losses, _ = _cell_losses(name, leaf)
        base_loss = torch.stack(losses).sum()
        base_loss.backward()
        assert leaf.grad is not None
        slope = (leaf.grad * direction).sum().item()
        assert abs(slope) > 0, "vacuous: the directional derivative is zero"
        base = base_loss.item()

        resolved = set()
        remainders = []
        for step in [2.0 / 2**power for power in range(7)]:
            out = _call(name, v + step * direction, cells, _source_cell(name))
            amplitudes = _receivers(name, out)
            weights = _weights(len(cells), amplitudes.shape[-1])
            shifted = (amplitudes[0] * weights).sum().item()
            resolved.add((out.info.max_vel, out.info.step_ratio))
            remainders.append(abs(shifted - base - step * slope))

    assert len(resolved) == 1, f"{name}: the discretisation moved: {resolved}"
    ratios = [
        remainders[index] / remainders[index + 1]
        for index in range(len(remainders) - 1)
        if remainders[index + 1] > 0
    ]
    runs = [
        index
        for index in range(len(ratios) - 2)
        if all(3.5 <= ratios[index + offset] <= 4.5 for offset in range(3))
    ]
    assert runs, f"{name}: no three consecutive second-order ratios: {ratios}"


# ---------------------------------------------------------------------------
# (b) Cell-local finite differences
# ---------------------------------------------------------------------------


def _central_difference(
    loss: Callable[[torch.Tensor], float],
    v: torch.Tensor,
    cell: tuple[int, int],
    step: float,
) -> float:
    """``(J(v + h) - J(v - h)) / 2h`` at one cell."""
    plus = v.clone()
    plus[cell] += step
    minus = v.clone()
    minus[cell] -= step
    return (loss(plus) - loss(minus)) / (2.0 * step)


def _one_sided_difference(
    loss: Callable[[torch.Tensor], float],
    v: torch.Tensor,
    cell: tuple[int, int],
    step: float,
) -> float:
    """``(-3 J(0) + 4 J(h) - J(2h)) / 2h``: second order, one side only.

    For a model whose value is at the bottom of its valid range -- a zero
    shear modulus, where the medium is a fluid and a negative value is not a
    medium at all -- the central difference has no left-hand point.
    """
    one = v.clone()
    one[cell] += step
    two = v.clone()
    two[cell] += 2.0 * step
    return (-3.0 * loss(v) + 4.0 * loss(one) - loss(two)) / (2.0 * step)


def _cell_local_errors(
    name: str, v: torch.Tensor, step: float, **overrides: Any
) -> dict[str, float]:
    """The relative adjoint-vs-finite-difference error at each probe cell."""
    cells = _probe_cells(name)
    leaf = v.clone().requires_grad_(True)
    losses, index_of = _cell_losses(name, leaf, **overrides)
    errors: dict[str, float] = {}
    for key, cell in cells.items():
        index = index_of[key]
        (gradient,) = torch.autograd.grad(losses[index], leaf, retain_graph=True)

        def loss_at(field: torch.Tensor, index: int = index) -> float:
            values, _ = _cell_losses(name, field, **overrides)
            return float(values[index].item())

        adjoint = gradient[cell].item()
        difference = _central_difference(loss_at, v, cell, step)
        errors[key] = abs(adjoint - difference) / max(abs(difference), 1e-300)
    return errors


@pytest.mark.parametrize("name", PROPAGATORS)
def test_cell_local_gradient_matches_finite_differences(name: str) -> None:
    """With ``max_vel`` given, every probed cell's gradient is the derivative.

    The probed cells are the one the maximum wave speed is read from, a
    corner and an edge (both inside the absorbing boundary), a source cell
    and a receiver cell -- the places where a missing term in an adjoint
    kernel (a boundary face, a guard ring, a source or receiver stencil)
    would hide.
    """
    v = _base_velocity()
    with _quiet():
        out = _call(name, v, [FASTEST_CELL], _source_cell(name))
        # Keep setup fixed while also bounding the positively perturbed
        # model, including viscoacoustic's unrelaxed speed.
        max_vel = 1.01 * out.info.max_vel
        errors = _cell_local_errors(name, v, 0.05, max_vel=max_vel)
    failed = {key: error for key, error in errors.items() if not error < RTOL}
    assert not failed, f"{name}: {failed} (all: {errors})"


@pytest.mark.parametrize("name", PROPAGATORS)
def test_omitting_max_vel_leaves_every_other_cell_exact(name: str) -> None:
    """Inferring the maximum costs nothing at any cell but the one it reads.

    The same comparison with ``max_vel`` omitted. The cell the maximum is
    read from is excluded -- it is the subject of the test below -- and every
    other probed cell still meets the same tolerance.
    """
    v = _base_velocity()
    with _quiet():
        errors = _cell_local_errors(name, v, 0.05)
    others = {key: error for key, error in errors.items() if key != "fastest"}
    failed = {key: error for key, error in others.items() if not error < RTOL}
    assert not failed, f"{name}: {failed} (all: {errors})"


def _pml_sensitive_loss(name: str, v: torch.Tensor, **overrides: Any) -> torch.Tensor:
    """A loss that reads a receiver inside the absorbing boundary.

    The maximum wave speed acts on the run through the absorbing profiles and
    the internal time step. A receiver in the interior is barely sensitive to
    the profiles, so a loss reading one would hide the missing term below in
    its own much larger direct sensitivity; a receiver inside the absorbing
    band is sensitive to little else.
    """
    cells = [FASTEST_CELL, (0, 0)]
    out = _call(name, v, cells, _source_cell(name), **overrides)
    amplitudes = _receivers(name, out)
    weights = _weights(len(cells), amplitudes.shape[-1])
    return (amplitudes[0, 1] * weights[1]).sum()


@pytest.mark.parametrize("name", PROPAGATORS)
def test_inferring_max_vel_costs_the_gradient_at_the_cell_it_is_read_from(
    name: str,
) -> None:
    """The documented exception, measured.

    With ``max_vel`` omitted the fastest cell's own value also sets the
    absorbing profiles and the internal time step, and the gradient does not
    follow that path: against a finite difference of the same call it is
    wrong by far more than the tolerance. Passing ``max_vel`` removes the
    path and the same gradient becomes exact. A neighbouring cell, which the
    maximum is not read from, is unaffected either way -- so this is one
    cell's term, not a general inexactness.
    """
    v = _base_velocity()
    neighbour = (FASTEST_CELL[0], FASTEST_CELL[1] + 1)

    def errors(**overrides: Any) -> dict[tuple[int, int], float]:
        leaf = v.clone().requires_grad_(True)
        _pml_sensitive_loss(name, leaf, **overrides).backward()
        assert leaf.grad is not None
        measured = {}
        for cell in (FASTEST_CELL, neighbour):
            adjoint = leaf.grad[cell].item()

            def loss_at(field: torch.Tensor) -> float:
                return float(_pml_sensitive_loss(name, field, **overrides).item())

            difference = _central_difference(loss_at, v, cell, 0.05)
            measured[cell] = abs(adjoint - difference) / max(abs(difference), 1e-300)
        return measured

    with _quiet():
        max_vel = 1.01 * _call(name, v, [FASTEST_CELL], _source_cell(name)).info.max_vel
        inferred = errors()
        given = errors(max_vel=max_vel)
    assert inferred[FASTEST_CELL] > 1e-3, (name, inferred)
    assert given[FASTEST_CELL] < RTOL, (name, given)
    assert inferred[neighbour] < RTOL, (name, inferred)
    assert given[neighbour] < RTOL, (name, given)


def test_one_sided_difference_at_a_zero_shear_modulus() -> None:
    """Where ``mu = 0`` the gradient still matches a one-sided difference.

    A fluid layer is the one place a central difference is unavailable: a
    negative shear modulus is not a medium. The second-order one-sided
    difference is what replaces it, and the elastic adjoint has to meet it
    at a cell where the forward's shear terms all vanish.
    """
    v = _base_velocity()
    models = _models("elastic", v)
    fluid = 4
    mu = models["mu"].clone()
    mu[:fluid] = 0.0
    models = {**models, "mu": mu.detach()}
    cell = (fluid - 2, N // 2)
    receiver_cells = [cell, (N - 4, N - 5)]
    source_cell = _source_cell("elastic")

    def loss_of(shear: torch.Tensor) -> torch.Tensor:
        out = _call(
            "elastic",
            v,
            receiver_cells,
            source_cell,
            models={**models, "mu": shear},
            max_vel=max_vel,
        )
        amplitudes = _receivers("elastic", out)
        weights = _weights(len(receiver_cells), amplitudes.shape[-1])
        return (amplitudes[0, 0] * weights[0]).sum()

    with _quiet():
        max_vel = float(v.max())
        leaf = mu.clone().requires_grad_(True)
        loss_of(leaf).backward()
        assert leaf.grad is not None
        adjoint = leaf.grad[cell].item()
        assert adjoint != 0.0, "vacuous: the shear-modulus gradient is zero"

        def scalar_loss(shear: torch.Tensor) -> float:
            return float(loss_of(shear).item())

        # A 0.05 m/s shear velocity in this fluid's density is ~1e4 Pa of
        # shear modulus at the linearisation point mu = 0; the difference is
        # taken over a range small enough to stay in the fluid's regime.
        step = 1.0e5
        difference = _one_sided_difference(scalar_loss, mu, cell, step)
    error = abs(adjoint - difference) / abs(difference)
    assert error < RTOL, (adjoint, difference, error)


# ---------------------------------------------------------------------------
# (c) A batched model the maximum wave speed is read from
# ---------------------------------------------------------------------------

#: The propagators whose models ``vmap`` can be mapped over at all: the Born
#: pairs reject a vmapped model outright, before ``max_vel`` is reached (see
#: ``tests/test_torch_func_max_vel.py``).
VMAP_PROPAGATORS: tuple[str, ...] = tuple(
    name for name in PROPAGATORS if name not in ("acoustic_born", "elastic_born")
)


@pytest.mark.parametrize("name", VMAP_PROPAGATORS)
def test_a_batched_maximum_model_with_max_vel_is_the_per_sample_loop(
    name: str,
) -> None:
    """Given ``max_vel``, the batched run equals the loop, bit for bit.

    ``max_vel`` has to be given when a model the maximum wave speed is read
    from is batched by ``vmap``, because the batched values cannot be read.
    Given it, nothing about the run is batch-dependent any more, and that is
    what this pins: the two samples come back exactly as they come back one
    at a time, so passing the argument is a complete substitute for the read
    rather than an approximation of it.
    """
    first = _base_velocity()
    second = first.clone()
    second[FASTEST_CELL] -= 5.0
    batched_velocity = torch.stack([first, second])
    cells = [FASTEST_CELL, (N - 4, N - 5)]
    source_cell = _source_cell(name)
    max_vel = 3000.0

    def run(v: torch.Tensor) -> torch.Tensor:
        return _receivers(
            name,
            _call(
                name,
                v,
                cells,
                source_cell,
                max_vel=max_vel,
                backend="eager",
            ),
        )

    with _quiet():
        batched = torch.func.vmap(run)(batched_velocity)
        loop = torch.stack([run(first), run(second)])
    assert torch.equal(batched, loop), (batched - loop).abs().max().item()


@pytest.mark.parametrize("name", VMAP_PROPAGATORS)
def test_a_batched_maximum_model_without_max_vel_raises(name: str) -> None:
    """Omitting it is the error, named for the argument that fixes it."""
    batched_velocity = torch.stack([_base_velocity()] * 2)
    cells = [FASTEST_CELL]
    source_cell = _source_cell(name)

    def run(v: torch.Tensor) -> torch.Tensor:
        return _receivers(name, _call(name, v, cells, source_cell, backend="eager"))

    with _quiet(), pytest.raises(deepwave.DeepwaveRuntimeError, match="max_vel="):
        torch.func.vmap(run)(batched_velocity)


# ---------------------------------------------------------------------------
# The forward-mode PML caveat
# ---------------------------------------------------------------------------

#: The propagators with a Born-based ``torch.func.jvp`` rule that warns about
#: the absorbing profiles.
JVP_PROPAGATORS: tuple[str, ...] = ("scalar", "acoustic", "elastic")


def _tangent(where: str) -> torch.Tensor:
    """A model tangent confined to the absorbing band or to the interior.

    The ``"pml"`` tangent covers the absorbing band on the first face and
    nothing else, so the whole of it is the case the forward-mode caveat is
    about. It does not reach the cell the maximum wave speed is read from, so
    the maximum, the absorbing profiles and the internal time step are the
    same for every call it is used in.
    """
    generator = torch.Generator().manual_seed(11)
    values = torch.randn((N, N), generator=generator, dtype=DTYPE)
    tangent = torch.zeros((N, N), dtype=DTYPE)
    band = PML_WIDTH + ACCURACY // 2
    if where == "pml":
        tangent[:band] = values[:band]
    else:
        interior = slice(band, N - band)
        tangent[interior, interior] = values[interior, interior]
    return tangent


def _jvp_of(
    name: str, tangent_cells: str, **overrides: Any
) -> tuple[torch.Tensor, torch.Tensor, list[warnings.WarningMessage]]:
    """A ``jvp`` whose model tangent is inside or outside the PML band."""
    v = _base_velocity()
    tangent = _tangent(tangent_cells)
    cells = [FASTEST_CELL, (N - 4, N - 5)]
    source_cell = _source_cell(name)

    def forward(field: torch.Tensor) -> torch.Tensor:
        return _receivers(name, _call(name, field, cells, source_cell, **overrides))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        primal, derivative = torch.func.jvp(forward, (v,), (tangent,))
    return (
        primal,
        derivative,
        [record for record in caught if "PML" in str(record.message)],
    )


@pytest.mark.parametrize("name", JVP_PROPAGATORS)
def test_the_jvp_pml_caveat_is_silent_when_max_vel_is_given(name: str) -> None:
    """Given ``max_vel``, the profiles are constants, so there is no caveat.

    The warning is about the forward-mode rule not linearising the absorbing
    profiles' dependence on the model. That dependence is entirely through
    the maximum wave speed, so when the caller supplies it the profiles do
    not depend on the model at all and there is nothing left to be inexact
    about -- a tangent inside the PML band included.
    """
    _, _, caught = _jvp_of(name, "pml", max_vel=3000.0)
    assert not caught, [str(record.message) for record in caught]


@pytest.mark.parametrize("name", JVP_PROPAGATORS)
def test_the_jvp_pml_caveat_fires_when_the_maximum_is_inferred(name: str) -> None:
    """With ``max_vel`` omitted the profiles do depend on the model."""
    _, _, caught = _jvp_of(name, "pml")
    assert len(caught) == 1, [str(record.message) for record in caught]
    message = str(caught[0].message)
    assert "max_vel" in message, message


@pytest.mark.parametrize("name", JVP_PROPAGATORS)
def test_the_jvp_is_exact_when_max_vel_is_given(name: str) -> None:
    """And the silence is earned: it is the derivative of the call.

    The tangent is confined to the absorbing band -- exactly the case the
    warning fires on when the maximum is inferred. The reference is the same
    call on the eager backend, where the tangent is PyTorch's own
    forward-mode derivative of the reference implementation rather than the
    compiled backend's Born-based rule. With ``max_vel`` given, every
    numerical parameter is a constant of the call, so that reference *is* the
    exact derivative, and the two agree to rounding. A finite difference is
    not used here: the absorbing band's contribution to an interior receiver
    is ~1e-8 of the receiver's own amplitude, so a difference of the forward
    would be measuring round-off.
    """
    _, derivative, _ = _jvp_of(name, "pml", max_vel=3000.0)
    _, reference, _ = _jvp_of(name, "pml", max_vel=3000.0, backend="eager")
    scale = reference.abs().max().item()
    assert scale > 0, "vacuous: the tangent is identically zero"
    assert (derivative - reference).abs().max().item() / scale < 1e-10


def test_pml_freq_inference_is_the_documented_source_gradient_condition() -> None:
    """Inferring ``pml_freq`` changes the source-amplitude gradient.

    When ``pml_freq`` is omitted it is inferred from the source amplitudes'
    spectrum, so the amplitudes act on the absorbing profiles as well as on
    the wavefield -- a path the gradient does not follow. Pinning it removes
    the path, which is what the documentation tells an inversion to do; this
    shows the two are not the same calculation, so the advice is not empty.
    """
    v = _base_velocity()
    cells = [FASTEST_CELL, (N - 4, N - 5)]
    source_cell = _source_cell("scalar")
    amplitudes = _wavelet(NT)

    def gradient(**overrides: Any) -> torch.Tensor:
        leaf = amplitudes.clone().requires_grad_(True)
        out = _call(
            "scalar",
            v,
            cells,
            source_cell,
            max_vel=3000.0,
            source_amplitudes=leaf,
            **overrides,
        )
        _receivers("scalar", out).square().sum().backward()
        assert leaf.grad is not None
        return leaf.grad

    with _quiet():
        pinned = gradient(pml_freq=FREQ)
        inferred_run = _call(
            "scalar", v, cells, source_cell, max_vel=3000.0, pml_freq=None
        )
        inferred = gradient(pml_freq=None)
    assert inferred_run.info.pml_freq is not None
    assert not math.isclose(inferred_run.info.pml_freq, FREQ, rel_tol=1e-12)
    # The two gradients are of different calls (different profiles), so they
    # differ; neither follows the amplitudes' path into the profiles.
    assert not torch.equal(pinned, inferred)
