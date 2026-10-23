"""``PropagatorOutput`` and ``AuxiliaryState`` as PyTorch pytree nodes.

Both are registered as pytree nodes, which is what lets ``torch.func`` and
``torch.compile`` see the tensors inside them instead of treating the whole
object as one opaque leaf. Before the registration, ``torch.utils._pytree``
reported a single leaf of type ``PropagatorOutput`` for an output, and a
``torch.func.jacrev(..., has_aux=True)`` returning one raised -- with
``max_vel`` supplied, so that the call got as far as building an output --
``RuntimeError: Expected tensors, got unsupported type
<class 'deepwave.AuxiliaryState'>``.

What the registration has to preserve is checked here as well as the fact
that it works: the flattened order is the documented positional return order,
the field names and ``info`` come back, and an auxiliary state's recorded run
parameters -- ``format_version`` above all -- survive a round trip, so a state
that has been through a pytree is still a state a continuation can check
itself against.

Two limits of the registration are pinned here too, because both are choices
rather than accidents: a ``treespec`` containing either type is not
serialisable to JSON (neither type invents a serialisation format for its
context), and repeated registration preserves the original flattening contract. The
required keyed pytree API is available in the declared PyTorch 2.4 minimum.
"""

import io
import pickle
import subprocess
import sys
import warnings

import pytest
import torch
import torch.utils._pytree as pytree

import deepwave
from deepwave import AuxiliaryState, PropagatorOutput

DTYPE = torch.float64
NY, NX = 30, 30
DX = 4.0
DT = 0.001
NT = 12
FREQ = 25.0


def _velocity(scale: float = 1.0) -> torch.Tensor:
    """A homogeneous velocity model."""
    return torch.full((NY, NX), 1500.0 * scale, dtype=DTYPE)


def _source() -> torch.Tensor:
    """A short Ricker wavelet for one shot with one source."""
    return deepwave.wavelets.ricker(FREQ, NT, DT, 1.5 / FREQ).to(DTYPE)[None, None]


def _run(v: torch.Tensor, **kwargs: object) -> PropagatorOutput:
    """A tiny ``scalar`` call whose output carries an auxiliary state."""
    return deepwave.scalar(
        v,
        DX,
        DT,
        source_amplitudes=_source(),
        source_locations=torch.tensor([[[15, 15]]]),
        receiver_locations=torch.tensor([[[20, 20]]]),
        pml_width=3,
        pml_freq=FREQ,
        **kwargs,  # type: ignore[arg-type]
    )


def _tensor_count(out: PropagatorOutput) -> int:
    """The number of tensors in an output, with the auxiliary state expanded."""
    return sum(len(value) if isinstance(value, AuxiliaryState) else 1 for value in out)


# ---------------------------------------------------------------------------
# PropagatorOutput
# ---------------------------------------------------------------------------


def test_propagator_output_flattens_to_exactly_its_tensors() -> None:
    """The leaves are tensors only: the auxiliary state is not a leaf itself.

    Registered as a leaf (which is what an unregistered type is), the same
    call gave one leaf of type ``PropagatorOutput``.
    """
    out = _run(_velocity())
    leaves = pytree.tree_leaves(out)
    assert len(leaves) == _tensor_count(out) > 1
    assert all(isinstance(leaf, torch.Tensor) for leaf in leaves)


def test_flattened_order_is_the_positional_return_order() -> None:
    """The children are the tuple elements, in the order the propagator returns.

    The auxiliary state expands in place, so the leading physical fields keep
    their positions and the trailing receiver fields keep theirs.
    """
    out = _run(_velocity())
    leaves = pytree.tree_leaves(out)
    expected: list[torch.Tensor] = []
    for value in out:
        if isinstance(value, AuxiliaryState):
            expected.extend(value.values())
        else:
            expected.append(value)
    assert len(leaves) == len(expected)
    assert all(a is b for a, b in zip(leaves, expected))


def test_round_trip_restores_the_names_info_and_auxiliary_state() -> None:
    """Flatten then unflatten gives back an equivalent output.

    The field names vary per instance, so they travel in the context together
    with ``info``; without them the object could not be rebuilt at all.
    """
    out = _run(_velocity())
    leaves, spec = pytree.tree_flatten(out)
    rebuilt = pytree.tree_unflatten(leaves, spec)
    assert isinstance(rebuilt, PropagatorOutput)
    assert rebuilt.names == out.names
    assert rebuilt.info == out.info
    assert all(a is b for a, b in zip(rebuilt, out) if isinstance(a, torch.Tensor))
    state = rebuilt.auxiliary_state
    assert isinstance(state, AuxiliaryState)
    assert state.names == out.auxiliary_state.names
    assert state.format_version == out.auxiliary_state.format_version
    assert state.propagator == "scalar"
    assert state.internal_dt == out.auxiliary_state.internal_dt


def test_two_outputs_of_the_same_call_share_a_treespec() -> None:
    """The context compares equal, which is what lets a transform reuse a spec."""
    first = pytree.tree_structure(_run(_velocity()))
    second = pytree.tree_structure(_run(_velocity()))
    assert first == second
    # PyTorch 2.4 TreeSpec itself is unhashable (its children are a list).
    # Equality is the public composition contract shared by supported versions.


def test_keyed_flatten_names_every_leaf() -> None:
    """Each leaf's key path is its field name (and, in the state, its tensor)."""
    out = _run(_velocity())
    keyed, _spec = pytree.tree_flatten_with_path(out)
    paths = [pytree.keystr(path) for path, _leaf in keyed]
    assert paths[0] == ".wavefield_0"
    assert paths[-1] == ".receiver_amplitudes"
    assert any(path.startswith(".auxiliary_state[") for path in paths)


def test_continuation_after_a_tree_map_round_trip_is_unchanged() -> None:
    """``initial_state=`` accepts an output that has been through ``tree_map``.

    A transform rebuilds the output through the registered unflatten, so a
    state that has been round-tripped must still continue the simulation
    exactly: the recorded run parameters the continuation checks itself
    against have to survive with it.
    """
    first = _run(_velocity())
    direct = _run(_velocity(), initial_state=first)
    mapped = pytree.tree_map(lambda t: t.clone(), first)
    assert isinstance(mapped, PropagatorOutput)
    via_tree = _run(_velocity(), initial_state=mapped)
    assert torch.equal(direct.receiver_amplitudes, via_tree.receiver_amplitudes)
    assert torch.equal(direct.wavefield_0, via_tree.wavefield_0)


def test_pickling_still_round_trips() -> None:
    """The registration does not disturb the existing pickle path."""
    out = _run(_velocity())
    revived = pickle.loads(pickle.dumps(out))  # noqa: S301 -- our own object
    assert isinstance(revived, PropagatorOutput)
    assert revived.names == out.names
    assert revived.info == out.info
    assert torch.equal(revived.receiver_amplitudes, out.receiver_amplitudes)
    assert revived.auxiliary_state.format_version == out.auxiliary_state.format_version


# ---------------------------------------------------------------------------
# torch.func and torch.compile
# ---------------------------------------------------------------------------


def test_vmap_over_a_model_returns_a_batched_output() -> None:
    """``vmap`` can add its batch dimension to every field of the output.

    ``vmap`` cannot map over the source amplitudes (see
    ``test_vmap_over_source_amplitudes_is_refused_for_its_own_reason``), so
    the model is what is mapped here.
    """
    models = torch.stack([_velocity(), _velocity(1.01)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = torch.func.vmap(lambda v: _run(v, backend="eager", max_vel=2000.0))(
            models
        )
    assert isinstance(out, PropagatorOutput)
    assert out.receiver_amplitudes.shape[0] == 2
    assert out.wavefield_0.shape[0] == 2
    for tensor in out.auxiliary_state.values():
        assert tensor.shape[0] == 2
    assert not torch.equal(out.receiver_amplitudes[0], out.receiver_amplitudes[1]), (
        "vacuous: the two models gave the same data"
    )


def test_vmap_over_source_amplitudes_is_refused_for_its_own_reason() -> None:
    """Mapping a non-model input is a separate, pre-existing limitation.

    It is refused by the propagator before the output is built, so it says
    nothing about whether the output is a pytree.
    """
    amplitudes = torch.stack([_source(), 2 * _source()])
    with pytest.raises(NotImplementedError, match="Only property models"):
        torch.func.vmap(
            lambda a: (
                deepwave.scalar(
                    _velocity(),
                    DX,
                    DT,
                    source_amplitudes=a,
                    source_locations=torch.tensor([[[15, 15]]]),
                    receiver_locations=torch.tensor([[[20, 20]]]),
                    pml_width=3,
                    pml_freq=FREQ,
                    backend="eager",
                    max_vel=2000.0,
                ).receiver_amplitudes
            )
        )(amplitudes)


def test_jacrev_can_carry_the_whole_output_as_its_auxiliary_value() -> None:
    """``jacrev(has_aux=True)`` returning the output used to raise.

    On the unregistered types, a call that got as far as building an output
    raised ``RuntimeError: Expected tensors, got unsupported type <class
    'deepwave.AuxiliaryState'>``, because functorch walks a returned ``tuple``
    but had no way through the state inside it. (Written as below, with
    ``max_vel`` omitted, the old code raised its ``max_vel`` error first: a
    ``jacrev`` wrapper was treated as hiding the model's values. Both are
    fixed, and the ``max_vel`` half is pinned in
    ``tests/test_torch_func_max_vel.py``.)
    """
    v = _velocity()

    def loss_and_output(
        model: torch.Tensor,
    ) -> tuple[torch.Tensor, PropagatorOutput]:
        out = _run(model, backend="eager")
        return out.receiver_amplitudes.pow(2).sum(), out

    jacobian, aux = torch.func.jacrev(loss_and_output, has_aux=True)(v)
    assert jacobian.shape == v.shape
    assert jacobian.abs().max() > 0, "vacuous: the Jacobian is identically zero"
    assert isinstance(aux, PropagatorOutput)
    assert aux.names == _run(v).names
    assert isinstance(aux.auxiliary_state, AuxiliaryState)
    assert aux.auxiliary_state.format_version == (
        deepwave._engine.AUXILIARY_STATE_FORMAT_VERSION
    )


def test_torch_compile_returns_a_propagator_output() -> None:
    """A compiled function may return the output itself.

    The call graph-breaks (the propagators make data-dependent reads and enter
    a C/CUDA backend), so this checks the return value survives the breaks,
    not that the propagator compiles into one graph.
    """
    compiled = torch.compile(lambda v: _run(v), backend="eager")
    out = compiled(_velocity())
    reference = _run(_velocity())
    assert isinstance(out, PropagatorOutput)
    assert out.names == reference.names
    assert torch.equal(out.receiver_amplitudes, reference.receiver_amplitudes)
    assert isinstance(out.auxiliary_state, AuxiliaryState)


# ---------------------------------------------------------------------------
# AuxiliaryState
# ---------------------------------------------------------------------------


def test_auxiliary_state_children_are_its_tensors() -> None:
    """The leaves are the tensors, in the order the state holds them."""
    state = _run(_velocity()).auxiliary_state
    leaves = pytree.tree_leaves(state)
    assert len(leaves) == len(state) > 0
    assert all(a is b for a, b in zip(leaves, state.values()))


def test_auxiliary_state_round_trip_preserves_every_recorded_parameter() -> None:
    """Names and all recorded run parameters travel as context.

    They are what a continuation checks itself against, so a state that lost
    them would be accepted by a call it does not belong to.
    """
    state = _run(_velocity()).auxiliary_state
    leaves, spec = pytree.tree_flatten(state)
    rebuilt = pytree.tree_unflatten(leaves, spec)
    assert isinstance(rebuilt, AuxiliaryState)
    assert rebuilt.names == state.names
    for name in deepwave._engine.AUXILIARY_STATE_PARAMETER_NAMES:
        assert getattr(rebuilt, name) == getattr(state, name), name
    assert rebuilt.compatible_with(state)


def test_auxiliary_state_round_trip_preserves_an_unusual_format_version() -> None:
    """``format_version`` is carried, not defaulted.

    A state written by another layout must come back still claiming that
    layout, so that handing it to a propagator raises rather than having its
    tensors read as this release's layout.
    """
    tensors = {"psiy_m1": torch.zeros(1, 4, 4, dtype=DTYPE)}
    state = AuxiliaryState(tensors, propagator="scalar", format_version=99)
    leaves, spec = pytree.tree_flatten(state)
    rebuilt = pytree.tree_unflatten(leaves, spec)
    assert rebuilt.format_version == 99
    assert not rebuilt.compatible_with(AuxiliaryState(tensors, propagator="scalar"))


def test_auxiliary_state_torch_save_round_trip_still_works() -> None:
    """``torch.save``/``torch.load`` of a state is unaffected."""
    state = _run(_velocity()).auxiliary_state
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    assert isinstance(loaded, AuxiliaryState)
    assert loaded.names == state.names
    assert loaded.format_version == state.format_version
    assert loaded.compatible_with(state)
    assert all(torch.equal(a, b) for a, b in zip(loaded.values(), state.values()))


def test_auxiliary_state_unflatten_rejects_a_wrong_number_of_tensors() -> None:
    """A truncated child list is an error, not a state with fields missing."""
    state = _run(_velocity()).auxiliary_state
    children, context = deepwave._engine._flatten_auxiliary_state(state)
    with pytest.raises(deepwave.DeepwaveValueError, match="cannot be rebuilt from"):
        deepwave._engine._unflatten_auxiliary_state(children[:-1], context)


# ---------------------------------------------------------------------------
# What the registration deliberately does not do
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("registered", [PropagatorOutput, AuxiliaryState])
def test_a_treespec_holding_one_is_not_serialisable(registered: type) -> None:
    """``treespec_dumps`` refuses, saying no serialised name is registered.

    The context of either type holds objects that are not JSON (a ``RunInfo``;
    an auxiliary state's recorded run parameters), and neither type invents an
    encoding for them, so no ``serialized_type_name`` is registered and a
    ``treespec`` containing one has no JSON form. Flattening and every
    transform are unaffected -- only ``treespec`` *serialisation*, which
    ``torch.export`` uses.
    """
    out = _run(_velocity())
    tree = out if registered is PropagatorOutput else out.auxiliary_state
    # PyTorch's own message spells the word the American way; it is matched
    # around rather than quoted, so this file keeps the British spelling.
    with pytest.raises(NotImplementedError, match=r"No registered serial\w+ name"):
        pytree.treespec_dumps(pytree.tree_structure(tree))


def test_importing_deepwave_emits_no_warning() -> None:
    """A fresh interpreter imports the package silently.

    The supported PyTorch registration API should not warn during import.
    """
    result = subprocess.run(  # noqa: S603 - current interpreter, fixed code
        [
            sys.executable,
            "-c",
            "import warnings\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            "    import deepwave\n"
            "print([str(w.message) for w in caught])\n",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("[]"), result.stdout


def test_a_second_registration_leaves_the_first_alone() -> None:
    """Registering again (another library, or a reload) is not an error.

    PyTorch refuses a second registration of the same type; the wrapper treats
    that as success and leaves the first one in place, because objects already
    flattened under it are still being held -- a replacement would leave their
    ``treespec`` objects unflattening through different functions.
    """
    before = pytree.SUPPORTED_NODES[PropagatorOutput]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert deepwave._pytree.register(
            PropagatorOutput,
            lambda output: (list(output), None),
            lambda children, _context: tuple(children),
        )
    assert pytree.SUPPORTED_NODES[PropagatorOutput] is before
    # And the real registration is still the one in effect: the replacement
    # offered above would have rebuilt a plain tuple with no names.
    leaves, spec = pytree.tree_flatten(_run(_velocity()))
    assert isinstance(pytree.tree_unflatten(leaves, spec), PropagatorOutput)
