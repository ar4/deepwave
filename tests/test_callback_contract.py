"""The ``CallbackState`` contract, the stepping attributes, and continuation.

Pins what a callback may rely on: which accessors support in-place mutation
and which do not, what ``mode=`` returns, what ``step``/``direction``/
``n_steps`` mean and in what unit, when ``role="forward"`` is available, and
the ``survey_pad``/``origin`` continuation rule.
"""

import dataclasses
import warnings
from typing import Any

import pytest
import torch

import deepwave
import deepwave.checkpointing
from deepwave._capabilities import PROPAGATOR_NAMES

DTYPE = torch.float64
FREQ = 25.0


def _problem(
    nt: int = 12,
    dt: float = 0.001,
    n: int = 40,
    n_shots: int = 1,
    max_vel: float = 2500.0,
) -> dict:
    """A small 2-D scalar problem, as plain keyword arguments."""
    amp = (
        deepwave.wavelets.ricker(FREQ, nt, dt, 1.5 / FREQ)
        .to(DTYPE)
        .reshape(1, 1, -1)
        .repeat(n_shots, 1, 1)
    )
    return {
        "v": torch.full((n, n), 1500.0, dtype=DTYPE),
        "dx": 5.0,
        "dt": dt,
        "kwargs": {
            "source_amplitudes": amp,
            "source_locations": torch.tensor([[[n // 2, n // 2]]]).repeat(
                n_shots, 1, 1
            ),
            "receiver_locations": torch.tensor([[[n // 2 + 2, n // 2 + 2]]]).repeat(
                n_shots, 1, 1
            ),
            "max_vel": max_vel,
            "pml_freq": FREQ,
        },
    }


def _run(callback_kind: str = "forward", **overrides: Any) -> Any:
    """Run ``scalar`` with a callback, returning the output."""
    p = _problem(**{k: overrides.pop(k) for k in ("nt", "dt", "n") if k in overrides})
    kwargs = {**p["kwargs"], **overrides}
    key = f"{callback_kind}_callback"
    if key not in kwargs:
        raise AssertionError("caller must supply the callback")
    return deepwave.scalar(p["v"], p["dx"], p["dt"], **kwargs)


# --------------------------------------------------------------------------
# Mutation and the accessors
# --------------------------------------------------------------------------


def test_get_gradient_mutation_still_changes_v_grad() -> None:
    """In-place mutation of ``get_gradient`` is supported, and still bites.

    The one mutation promise the narrowed contract keeps (it is what
    docs/example_custom_imaging_condition.py relies on). Compiled backend,
    which is the only one that populates the gradient dictionary.
    """
    p = _problem(n=60)
    factor = 4.0

    def control_cb(state: deepwave.CallbackState) -> None:
        pass

    def scaling_cb(state: deepwave.CallbackState) -> None:
        if state.step == 0:
            state.get_gradient("v", view="full").div_(factor)

    grads = []
    for cb in (control_cb, scaling_cb):
        v = p["v"].clone().requires_grad_(True)
        out = deepwave.scalar(
            v,
            p["dx"],
            p["dt"],
            **p["kwargs"],
            pml_width=10,
            backward_callback=cb,
            callback_frequency=1,
        )
        out.receiver_amplitudes.pow(2).sum().backward()
        assert v.grad is not None
        grads.append(v.grad.clone())

    control, scaled = grads
    assert not torch.equal(control, scaled)
    # The division is applied at the last backward callback (step 0) and
    # scales every cell of the returned gradient exactly -- except the source
    # cell, which receives a further contribution after the last callback
    # fires (the callback sees the accumulator, not the finished gradient;
    # compare the "neither callback sees the final state" note on
    # CallbackState.step). That cell is excluded from the comparison, and
    # then asserted against positively: masking alone would leave the test
    # indifferent to what that cell does, so it is pinned as a known gap that
    # a future fix has to come here and delete, rather than one that could be
    # closed without anyone noticing.
    source = (60 // 2, 60 // 2)
    mask = torch.ones_like(control, dtype=torch.bool)
    mask[source] = False
    assert torch.allclose(scaled[mask], control[mask] / factor, rtol=1e-12, atol=0.0)
    assert mask.sum() > 0
    assert not torch.isclose(
        scaled[source], control[source] / factor, rtol=1e-6, atol=0.0
    ), (
        "the source cell now scales with the rest: the last-backward-callback "
        "mutation reaches it, so delete this exclusion and the caveat on "
        "CallbackState.step that documents it"
    )


def test_set_wavefield_round_trip() -> None:
    """``set_wavefield`` writes exactly what ``get_wavefield`` + mutation did.

    The supported replacement for mutating a ``get_wavefield`` result.
    """
    p = _problem()

    def mutating_cb(state: deepwave.CallbackState) -> None:
        if state.step == 3:
            state.get_wavefield("wavefield_0").add_(1.0)

    def setting_cb(state: deepwave.CallbackState) -> None:
        if state.step == 3:
            state.set_wavefield("wavefield_0", state.get_wavefield("wavefield_0") + 1.0)

    def noop_cb(state: deepwave.CallbackState) -> None:
        pass

    results = {}
    for name, cb in (
        ("mutate", mutating_cb),
        ("set", setting_cb),
        ("noop", noop_cb),
    ):
        out = deepwave.scalar(
            p["v"],
            p["dx"],
            p["dt"],
            **p["kwargs"],
            forward_callback=cb,
            callback_frequency=1,
        )
        results[name] = out.receiver_amplitudes.clone()

    assert torch.equal(results["mutate"], results["set"])
    assert not torch.equal(results["noop"], results["set"])


@pytest.mark.parametrize("backend", ["compiled", "eager"])
def test_set_wavefield_round_trip_both_backends(backend: str) -> None:
    """``set_wavefield`` behaves identically on both backends."""
    p = _problem()

    def setting_cb(state: deepwave.CallbackState) -> None:
        if state.step == 3:
            state.set_wavefield("wavefield_0", state.get_wavefield("wavefield_0") + 1.0)

    def noop_cb(state: deepwave.CallbackState) -> None:
        pass

    outs = [
        deepwave.scalar(
            p["v"],
            p["dx"],
            p["dt"],
            **p["kwargs"],
            forward_callback=cb,
            callback_frequency=1,
            backend=backend,
        ).receiver_amplitudes
        for cb in (setting_cb, noop_cb)
    ]
    assert not torch.equal(outs[0], outs[1])


def test_set_wavefield_rejects_wrong_shape_dtype_device() -> None:
    """A mismatched tensor raises, naming both sides, rather than converting.

    A silent cast loses precision where it is least visible and a silent
    host-to-device copy inside a time loop is a performance trap.
    """
    p = _problem()
    errors: dict[str, str] = {}

    def cb(state: deepwave.CallbackState) -> None:
        if state.step != 0:
            return
        good = state.get_wavefield("wavefield_0")
        try:
            state.set_wavefield(
                "wavefield_0", torch.zeros(good.shape[0], 3, 3, dtype=good.dtype)
            )
        except ValueError as e:
            errors["shape"] = str(e)
        try:
            state.set_wavefield("wavefield_0", good.to(torch.float32))
        except TypeError as e:
            errors["dtype"] = str(e)

    deepwave.scalar(
        p["v"],
        p["dx"],
        p["dt"],
        **p["kwargs"],
        forward_callback=cb,
        callback_frequency=1,
    )
    assert "shape" in errors, "no shape error raised"
    assert "[1, 3, 3]" in errors["shape"]
    assert "view=" in errors["shape"]
    assert "dtype" in errors, "no dtype error raised"
    assert "float32" in errors["dtype"]
    assert "float64" in errors["dtype"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_set_wavefield_rejects_wrong_device() -> None:
    """A tensor on the wrong device raises, naming both devices."""
    p = _problem()
    errors: dict[str, str] = {}

    def cb(state: deepwave.CallbackState) -> None:
        if state.step == 0:
            good = state.get_wavefield("wavefield_0")
            try:
                state.set_wavefield("wavefield_0", good.cpu())
            except ValueError as e:
                errors["device"] = str(e)

    deepwave.scalar(
        p["v"].cuda(),
        p["dx"],
        p["dt"],
        **{k: (v.cuda() if torch.is_tensor(v) else v) for k, v in p["kwargs"].items()},
        forward_callback=cb,
        callback_frequency=1,
    )
    assert "device" in errors
    assert "cpu" in errors["device"]
    assert "cuda" in errors["device"]


def test_set_wavefield_view_default_matches_get_wavefield() -> None:
    """``set_wavefield``'s default ``view`` mirrors ``get_wavefield``'s.

    So ``set_wavefield(n, get_wavefield(n))`` is the identity, with no
    ``view=`` on either side.
    """
    p = _problem()
    checked: list[bool] = []

    def cb(state: deepwave.CallbackState) -> None:
        before = state.get_wavefield("wavefield_0", view="full").clone()
        state.set_wavefield("wavefield_0", state.get_wavefield("wavefield_0"))
        after = state.get_wavefield("wavefield_0", view="full")
        checked.append(torch.equal(before, after))

    deepwave.scalar(
        p["v"],
        p["dx"],
        p["dt"],
        **p["kwargs"],
        forward_callback=cb,
        callback_frequency=1,
    )
    assert checked
    assert all(checked)


def test_set_wavefield_unknown_name_lists_what_exists() -> None:
    """A name that is not a wavefield raises, listing the names that are."""
    p = _problem()
    errors: list[str] = []

    def cb(state: deepwave.CallbackState) -> None:
        if state.step == 0:
            try:
                state.set_wavefield("v", state.get_wavefield("wavefield_0"))
            except KeyError as e:
                errors.append(str(e))

    deepwave.scalar(
        p["v"],
        p["dx"],
        p["dt"],
        **p["kwargs"],
        forward_callback=cb,
        callback_frequency=1,
    )
    assert errors
    assert "wavefield_0" in errors[0]


def test_mode_read_returns_detached_view() -> None:
    """``mode="read"`` detaches but does not copy.

    Both halves matter: the autograd edge is gone, and the buffer is still
    live, so the docstring's "``.clone()`` it if you keep it" is honest. If
    someone later makes ``"read"`` return a copy, this test must be updated
    deliberately, not silently.
    """
    p = _problem()
    seen: dict[str, Any] = {}

    def cb(state: deepwave.CallbackState) -> None:
        if state.step == 0:
            r = state.get_wavefield("wavefield_0", view="full", mode="read")
            w = state.get_wavefield("wavefield_0", view="full", mode="write")
            seen["requires_grad"] = r.requires_grad
            seen["grad_fn"] = r.grad_fn
            seen["equal"] = torch.equal(r, w)
            seen["ptr"] = r.data_ptr() == w.data_ptr()
            seen["kept"] = r
            seen["snapshot"] = r.clone()
        if state.step == 5 and "kept" in seen:
            # The retained "read" tensor tracks the live buffer: it is a
            # view, not a snapshot.
            seen["changed"] = not torch.equal(seen["kept"], seen["snapshot"])

    deepwave.scalar(
        p["v"],
        p["dx"],
        p["dt"],
        **p["kwargs"],
        forward_callback=cb,
        callback_frequency=1,
    )
    assert seen["requires_grad"] is False
    assert seen["grad_fn"] is None
    assert seen["equal"] is True
    assert seen["ptr"] is True, "read mode must not copy"
    assert seen["changed"] is True, "read mode must be a live view, not a snapshot"


def test_mode_write_is_the_default_and_unchanged() -> None:
    """The default is ``"write"`` and returns the very tensor it always did."""
    p = _problem()
    same: list[bool] = []

    def cb(state: deepwave.CallbackState) -> None:
        a = state.get_wavefield("wavefield_0")
        b = state.get_wavefield("wavefield_0", mode="write")
        same.append(a.data_ptr() == b.data_ptr())
        m1 = state.get_model("v")
        m2 = state.get_model("v", mode="write")
        same.append(m1.data_ptr() == m2.data_ptr())

    deepwave.scalar(
        p["v"],
        p["dx"],
        p["dt"],
        **p["kwargs"],
        forward_callback=cb,
        callback_frequency=1,
    )
    assert same
    assert all(same)


def test_get_model_mode_read_detaches() -> None:
    """``get_model(mode="read")`` detaches, and its default stays ``"write"``."""
    p = _problem()
    seen: dict[str, Any] = {}
    v = p["v"].clone().requires_grad_(True)

    def cb(state: deepwave.CallbackState) -> None:
        if state.step == 0:
            seen["read"] = state.get_model("v", mode="read").requires_grad
            seen["write"] = state.get_model("v").requires_grad

    deepwave.scalar(
        v,
        p["dx"],
        p["dt"],
        **p["kwargs"],
        forward_callback=cb,
        callback_frequency=1,
    )
    assert seen["read"] is False
    assert seen["write"] is True


def test_writable_reports_gradient_only_where_populated() -> None:
    """``writable`` answers per name space, and only where a write is supported."""
    p = _problem()
    fwd: dict[str, Any] = {}
    bwd_compiled: dict[str, Any] = {}
    bwd_eager: dict[str, Any] = {}

    def record(into: dict) -> Any:
        def cb(state: deepwave.CallbackState) -> None:
            into["grad_v"] = state.writable("v", "gradient")
            into["wf"] = state.writable("wavefield_0")
            into["model"] = state.writable("v", "model")
            into["unknown"] = state.writable("nonesuch", "gradient")
            into["unknown_wf"] = state.writable("nonesuch")

        return cb

    v = p["v"].clone().requires_grad_(True)
    out = deepwave.scalar(
        v,
        p["dx"],
        p["dt"],
        **p["kwargs"],
        forward_callback=record(fwd),
        backward_callback=record(bwd_compiled),
        callback_frequency=1,
    )
    out.receiver_amplitudes.pow(2).sum().backward()

    v2 = p["v"].clone().requires_grad_(True)
    out2 = deepwave.scalar(
        v2,
        p["dx"],
        p["dt"],
        **p["kwargs"],
        backend="eager",
        backward_callback=record(bwd_eager),
        callback_frequency=1,
    )
    out2.receiver_amplitudes.pow(2).sum().backward()

    # Only the compiled backward pass populates gradients.
    assert bwd_compiled["grad_v"] is True
    assert fwd["grad_v"] is False
    assert bwd_eager["grad_v"] is False
    # Differentiable forward callbacks are observers.
    assert fwd["wf"] is False
    assert fwd["model"] is False
    assert bwd_compiled["wf"] is True
    for d in (bwd_compiled, bwd_eager):
        assert d["model"] is True
    # Unknown names are a query, not an error.
    for d in (fwd, bwd_compiled, bwd_eager):
        assert d["unknown"] is False
        assert d["unknown_wf"] is False


def test_writable_agrees_with_the_accessor_on_a_deprecated_alias() -> None:
    """``writable`` must predict :meth:`get_gradient`, aliases included.

    Its whole purpose is "branch on this instead of a ``try``/``except``", so
    a name the accessor still resolves (``"scatter"`` for ``"dv"``) and
    ``writable`` denied would send a callback down the wrong branch. Asking
    is a query, so it must not itself emit the deprecation warning.
    """
    p = _problem(n=30)
    seen: dict[str, Any] = {}

    def cb(state: deepwave.CallbackState) -> None:
        if state.step != 0:
            return
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            seen["writable_alias"] = state.writable("scatter", "gradient")
        seen["query_warnings"] = [w for w in record if "scatter" in str(w.message)]
        seen["writable_current"] = state.writable("dv", "gradient")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state.get_gradient("scatter")
            seen["accessor_alias"] = True

    v = p["v"].clone().requires_grad_(True)
    dv = torch.zeros_like(v).requires_grad_(True)
    out = deepwave.scalar_born(
        v,
        p["dx"],
        p["dt"],
        dv=dv,
        **p["kwargs"],
        backward_callback=cb,
        callback_frequency=1,
    )
    out.sc_receiver_amplitudes.pow(2).sum().backward()
    assert seen["writable_current"] is True
    assert seen["accessor_alias"] is True
    assert seen["writable_alias"] is True
    assert seen["query_warnings"] == []


def test_writable_rejects_an_unknown_kind() -> None:
    """An unknown ``kind`` is a mistake, not a query, so it raises."""
    p = _problem()
    errors: list[str] = []

    def cb(state: deepwave.CallbackState) -> None:
        if state.step == 0:
            try:
                state.writable("v", "not_a_kind")  # type: ignore[arg-type]
            except ValueError as e:
                errors.append(str(e))

    deepwave.scalar(
        p["v"],
        p["dx"],
        p["dt"],
        **p["kwargs"],
        forward_callback=cb,
        callback_frequency=1,
    )
    assert errors
    assert "kind" in errors[0]


# --------------------------------------------------------------------------
# Stepping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("dt", "expected_step_ratio"), [(0.0008, 1), (0.0024, 3)])
@pytest.mark.parametrize("mgsi", [1, 2])
@pytest.mark.parametrize("backend", ["compiled", "eager"])
@pytest.mark.parametrize("cf", [1, 3])
def test_step_direction_n_steps_forward_and_backward(
    dt: float, expected_step_ratio: int, mgsi: int, backend: str, cf: int
) -> None:
    """The exact ``(step, direction, n_steps)`` sequences of both passes.

    ``nt`` is divisible by every ``mgsi`` used, because a remainder raises.
    """
    user_nt = 12
    p = _problem(nt=user_nt, dt=dt, max_vel=2500.0)
    v = p["v"].clone().requires_grad_(True)
    fwd: list[tuple[int, str, int]] = []
    bwd: list[tuple[int, str, int]] = []

    def fcb(state: deepwave.CallbackState) -> None:
        fwd.append((state.step, state.direction, state.n_steps))

    def bcb(state: deepwave.CallbackState) -> None:
        bwd.append((state.step, state.direction, state.n_steps))

    kwargs = dict(p["kwargs"])
    out = deepwave.scalar(
        v,
        p["dx"],
        p["dt"],
        **kwargs,
        backend=backend,
        model_gradient_sampling_interval=mgsi,
        callback_frequency=cf,
        forward_callback=fcb,
        backward_callback=bcb if backend == "compiled" else None,
    )
    # Do not assume the step_ratio: assert the one that was actually resolved.
    assert out.info.step_ratio == expected_step_ratio
    out.receiver_amplitudes.pow(2).sum().backward()

    assert [s for s, _, _ in fwd] == list(range(0, user_nt, cf))
    assert {d for _, d, _ in fwd} == {"forward"}
    assert {n for _, _, n in fwd} == {user_nt}

    if backend == "compiled":
        expected_bwd = list(reversed(range(0, user_nt, cf)))
        assert [s for s, _, _ in bwd] == expected_bwd
        assert bwd[-1][0] == 0, "the backward pass always ends at step 0"
        assert {d for _, d, _ in bwd} == {"backward"}
        assert {n for _, _, n in bwd} == {user_nt}


def test_step_is_outer_not_internal() -> None:
    """``step`` counts outer steps, not internal ones.

    The regression test for the docstring that said ``callback_frequency``
    counted internal time steps: it was wrong by a factor of ``step_ratio``.
    At ``step_ratio=4`` the forward sequence is 0..11, not 0..47.
    """
    user_nt = 12
    p = _problem(nt=user_nt, dt=0.0024)
    steps: list[int] = []

    out = deepwave.scalar(
        p["v"],
        p["dx"],
        p["dt"],
        **p["kwargs"],
        callback_frequency=1,
        forward_callback=lambda s: steps.append(s.step),
    )
    assert out.info.step_ratio == 3
    assert steps == list(range(user_nt))
    assert steps[-1] == user_nt - 1


def test_state_dt_is_user_dt() -> None:
    """Callback time metadata uses user units and names the internal step."""
    p = _problem(nt=12, dt=0.0024)
    seen: list[tuple[float, float, float]] = []

    out = deepwave.scalar(
        p["v"],
        p["dx"],
        p["dt"],
        **p["kwargs"],
        callback_frequency=1,
        forward_callback=lambda s: seen.append((s.dt, s.internal_dt, s.time)),
    )
    assert out.info.step_ratio == 3
    assert seen[0][0] == pytest.approx(p["dt"])
    assert seen[0][1] == pytest.approx(out.info.internal_dt)
    assert [entry[2] for entry in seen] == pytest.approx(
        [i * p["dt"] for i in range(12)]
    )


def test_forward_only_mgsi_reset_cadence() -> None:
    """Model-gradient activation does not change callback observations."""
    user_nt = 12
    p = _problem(nt=user_nt)
    steps_no_grad: list[int] = []
    steps_grad: list[int] = []

    out_no_grad = deepwave.scalar(
        p["v"],
        p["dx"],
        p["dt"],
        **p["kwargs"],
        model_gradient_sampling_interval=2,
        callback_frequency=1,
        forward_callback=lambda s: steps_no_grad.append(s.step),
    )
    v = p["v"].clone().requires_grad_(True)
    out_grad = deepwave.scalar(
        v,
        p["dx"],
        p["dt"],
        **p["kwargs"],
        model_gradient_sampling_interval=2,
        callback_frequency=1,
        forward_callback=lambda s: steps_grad.append(s.step),
    )
    assert steps_no_grad == list(range(user_nt))
    assert steps_grad == list(range(user_nt))
    assert torch.equal(
        out_no_grad.receiver_amplitudes, out_grad.receiver_amplitudes.detach()
    )


@pytest.mark.parametrize("backend", ["compiled", "eager"])
@pytest.mark.parametrize(("mgsi", "frequency"), [(2, 5), (3, 4)])
def test_callbacks_do_not_change_sampled_gradient_or_forward_fields(
    backend: str, mgsi: int, frequency: int
) -> None:
    """User boundaries can split native chunks inside a history interval."""
    p = _problem(nt=24, dt=0.0024, max_vel=2500.0)

    def run(with_callback: bool):
        v = p["v"].clone().requires_grad_(True)
        observed: list[tuple[int, torch.Tensor]] = []
        backward_steps: list[int] = []

        def forward(state: deepwave.CallbackState) -> None:
            observed.append(
                (
                    state.step,
                    state.get_wavefield("wavefield_0", view="pml").detach().clone(),
                )
            )
            assert state.time == pytest.approx(state.step * p["dt"])
            assert state.dt == pytest.approx(p["dt"])

        def backward(state: deepwave.CallbackState) -> None:
            backward_steps.append(state.step)

        out = deepwave.scalar(
            v,
            p["dx"],
            p["dt"],
            **p["kwargs"],
            backend=backend,
            model_gradient_sampling_interval=mgsi,
            callback_frequency=frequency,
            forward_callback=forward if with_callback else None,
            backward_callback=backward
            if with_callback and backend == "compiled"
            else None,
        )
        out.receiver_amplitudes.square().sum().backward()
        assert v.grad is not None
        return out, v.grad, observed, backward_steps

    plain, plain_grad, _, _ = run(False)
    chunked, chunked_grad, observations, backward_steps = run(True)
    expected = list(range(0, 24, frequency))
    assert [step for step, _ in observations] == expected
    if backend == "compiled":
        assert backward_steps == expected[::-1]
    assert torch.allclose(chunked.receiver_amplitudes, plain.receiver_amplitudes)
    assert torch.allclose(chunked_grad, plain_grad, rtol=1e-10, atol=1e-12)

    no_grad_snapshots: list[torch.Tensor] = []
    with torch.no_grad():
        forward_only = deepwave.scalar(
            p["v"],
            p["dx"],
            p["dt"],
            **p["kwargs"],
            backend=backend,
            model_gradient_sampling_interval=mgsi,
            callback_frequency=frequency,
            forward_callback=lambda state: no_grad_snapshots.append(
                state.get_wavefield("wavefield_0", view="pml").clone()
            ),
        )
    assert torch.allclose(forward_only.receiver_amplitudes, chunked.receiver_amplitudes)
    assert len(no_grad_snapshots) == len(observations)
    for no_grad, (_, with_grad) in zip(no_grad_snapshots, observations):
        assert torch.allclose(no_grad, with_grad)


def test_backward_forward_snapshot_requires_aligned_history():
    """Opt-in backward forward fields reject an unsaved user boundary."""
    p = _problem(nt=24, dt=0.0024, max_vel=2500.0)
    v = p["v"].clone().requires_grad_(True)
    fired: list[int] = []
    with pytest.raises(deepwave.DeepwaveValueError, match="divisible"):
        deepwave.scalar(
            v,
            p["dx"],
            p["dt"],
            **p["kwargs"],
            model_gradient_sampling_interval=2,
            callback_frequency=3,
            backward_callback=lambda state: fired.append(state.step),
            backward_callback_needs_forward_wavefield=True,
        )
    assert not fired

    forward: dict[int, torch.Tensor] = {}
    backward: dict[int, torch.Tensor] = {}
    out = deepwave.scalar(
        v,
        p["dx"],
        p["dt"],
        **p["kwargs"],
        model_gradient_sampling_interval=2,
        callback_frequency=4,
        forward_callback=lambda state: forward.update(
            {
                state.step: state.get_wavefield("wavefield_0", view="pml")
                .detach()
                .clone()
            }
        ),
        backward_callback=lambda state: backward.update(
            {
                state.step: state.get_wavefield(
                    "wavefield_0", role="forward", view="pml"
                )
                .detach()
                .clone()
            }
        ),
        backward_callback_needs_forward_wavefield=True,
    )
    out.receiver_amplitudes.square().sum().backward()
    assert tuple(forward) == (0, 4, 8, 12, 16, 20)
    assert tuple(backward) == (20, 16, 12, 8, 4, 0)
    for step, value in forward.items():
        assert torch.allclose(backward[step], value)


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
@pytest.mark.parametrize("mgsi", [1, 2])
def test_callback_cadence_unchanged_all_propagators(name: str, mgsi: int) -> None:
    """Every propagator's forward ``step`` sequence, hard-coded.

    The permanent form of the before/after cadence guard: a change of unit
    anywhere in the engine moves one of these. Each propagator is its own
    case so one experimental propagator failing does not hide the rest.
    """
    from deepwave import _capabilities as cap

    user_nt = 12
    steps: list[int] = []
    models = list(cap._models(name, 2))
    models[0] = models[0].clone().requires_grad_(True)
    spec = deepwave.spec(name)
    channel = "x" if name == "acoustic_vti" else spec.source_channels(2)[0]
    amp = (
        deepwave.wavelets.ricker(cap._FREQ, user_nt, cap._DT, 1.5 / cap._FREQ)
        .to(cap._DTYPE)
        .reshape(1, 1, -1)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cap._call(
            name,
            2,
            None,
            **{spec.source_amplitudes_kwarg(channel): amp},
            model_gradient_sampling_interval=mgsi,
            callback_frequency=1,
            forward_callback=lambda s: steps.append(s.step),
            _models_override=models,
        )
    assert steps == list(range(user_nt)), name


# --------------------------------------------------------------------------
# Roles and the opt-in forward history
# --------------------------------------------------------------------------


def test_has_forward_wavefield_and_available_roles() -> None:
    """The branch-on-able form of the ``role="forward"`` capability."""
    p = _problem()
    with_flag: dict[str, Any] = {}
    without_flag: dict[str, Any] = {}

    def record(into: dict) -> Any:
        def cb(state: deepwave.CallbackState) -> None:
            into["has"] = state.has_forward_wavefield
            into["roles"] = state.available_roles

        return cb

    for into, flag in ((with_flag, True), (without_flag, False)):
        v = p["v"].clone().requires_grad_(True)
        out = deepwave.scalar(
            v,
            p["dx"],
            p["dt"],
            **p["kwargs"],
            backward_callback=record(into),
            backward_callback_needs_forward_wavefield=flag,
            callback_frequency=1,
        )
        out.receiver_amplitudes.pow(2).sum().backward()

    assert with_flag["has"] is True
    assert with_flag["roles"] == ("adjoint", "forward")
    assert without_flag["has"] is False
    assert without_flag["roles"] == ("adjoint",)


def test_has_forward_wavefield_false_on_another_propagator() -> None:
    """A propagator that cannot supply the forward field says so."""
    from deepwave import _capabilities as cap

    into: dict[str, Any] = {}

    def cb(state: deepwave.CallbackState) -> None:
        into["has"] = state.has_forward_wavefield
        into["roles"] = state.available_roles

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        primary = cap._models("acoustic", 2)[0].clone().requires_grad_(True)
        out = cap._receiver("acoustic", 2, None, primary, backward_callback=cb)
        out.sum().backward()

    assert into["has"] is False
    assert into["roles"] == ("adjoint",)


def test_role_backward_raises_inside_a_real_callback() -> None:
    """``role="backward"`` is not a value, and saying so does not stop the run."""
    p = _problem()
    seen: dict[str, Any] = {}

    def cb(state: deepwave.CallbackState) -> None:
        if "adjoint" in seen:
            return
        seen["adjoint"] = state.get_wavefield("wavefield_0", role="adjoint").clone()
        with pytest.raises(deepwave.DeepwaveValueError, match='role="adjoint"'):
            state.get_wavefield("wavefield_0", role="backward")  # type: ignore[arg-type]

    v = p["v"].clone().requires_grad_(True)
    out = deepwave.scalar(
        v,
        p["dx"],
        p["dt"],
        **p["kwargs"],
        backward_callback=cb,
        callback_frequency=1,
    )
    out.receiver_amplitudes.pow(2).sum().backward()
    assert "adjoint" in seen


def test_role_backward_error_names_adjoint() -> None:
    """The error says what to write instead.

    ``role="backward"`` was a development-branch synonym for ``"adjoint"``, so
    no release ever accepted it; it is named in the message rather than
    deprecated, because the two words are easy to confuse.
    """
    state = deepwave.CallbackState(
        0.001,
        0,
        {"wavefield_0": torch.zeros(1, 4, 4)},
        {"v": torch.zeros(1, 4, 4)},
        {},
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        direction="backward",
        n_steps=1,
    )
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        state.get_wavefield("wavefield_0", role="backward")  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert 'role="adjoint"' in message
    assert '"backward"' in message


def test_role_forward_without_flag_raises_naming_the_flag() -> None:
    """Asking for the forward field without opting in says exactly what to pass."""
    p = _problem()
    errors: list[str] = []

    def cb(state: deepwave.CallbackState) -> None:
        if not errors:
            try:
                state.get_wavefield("wavefield_0", role="forward")
            except NotImplementedError as e:
                errors.append(str(e))

    v = p["v"].clone().requires_grad_(True)
    out = deepwave.scalar(
        v,
        p["dx"],
        p["dt"],
        **p["kwargs"],
        backward_callback=cb,
        callback_frequency=1,
    )
    out.receiver_amplitudes.pow(2).sum().backward()
    assert errors
    assert "backward_callback_needs_forward_wavefield" in errors[0]
    assert "deepwave.scalar" in errors[0]


@pytest.mark.parametrize("backend", ["compiled", "eager"])
def test_opt_in_history_does_not_change_numbers(backend: str) -> None:
    """The opt-in flag changes what is retained, never what is computed."""
    p = _problem(n_shots=3)
    results = []
    for flag in (False, True):
        v = p["v"].clone().requires_grad_(True)
        out = deepwave.scalar(
            v,
            p["dx"],
            p["dt"],
            **p["kwargs"],
            backend=backend,
            backward_callback=lambda s: None,
            backward_callback_needs_forward_wavefield=flag,
            callback_frequency=1,
        )
        out.receiver_amplitudes.pow(2).sum().backward()
        assert v.grad is not None
        results.append(
            (
                out.receiver_amplitudes.detach().clone(),
                out.wavefield_0.detach().clone(),
                v.grad.clone(),
            )
        )
    for a, b in zip(results[0], results[1]):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_opt_in_history_is_not_allocated_by_default() -> None:
    """The default really does not allocate the second history.

    The memory claim, not just the flag: peak device memory with the flag off
    must be materially below the same run with it on.
    """
    p = _problem(nt=64, n=60)
    peaks = {}
    for flag in (False, True):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        v = p["v"].cuda().requires_grad_(True)
        kwargs = {
            k: (t.cuda() if torch.is_tensor(t) else t) for k, t in p["kwargs"].items()
        }
        out = deepwave.scalar(
            v,
            p["dx"],
            p["dt"],
            **kwargs,
            storage_mode="device",
            backward_callback=lambda s: None,
            backward_callback_needs_forward_wavefield=flag,
            callback_frequency=1,
        )
        out.receiver_amplitudes.pow(2).sum().backward()
        peaks[flag] = torch.cuda.max_memory_allocated()
        del out, v
    assert peaks[False] < peaks[True], peaks
    # The saved buffer is a full sampled wavefield history, so the saving is
    # a large fraction of the peak, not a rounding error.
    assert peaks[True] - peaks[False] > 0.1 * peaks[True], peaks


# --------------------------------------------------------------------------
# Continuation: the survey_pad/origin rule
# --------------------------------------------------------------------------


def _continuation_problem(nt: int = 6) -> dict:
    n = 60
    amp = (
        deepwave.wavelets.ricker(FREQ, nt, 0.001, 1.5 / FREQ)
        .to(DTYPE)
        .reshape(1, 1, -1)
    )
    return {
        "v": torch.full((n, n), 1500.0, dtype=DTYPE),
        "dx": 5.0,
        "dt": 0.001,
        "kwargs": {
            "source_amplitudes": amp,
            "source_locations": torch.tensor([[[n // 2, n // 2]]]),
            "receiver_locations": torch.tensor([[[n // 2 + 2, n // 2 + 2]]]),
            "max_vel": 2500.0,
            "pml_freq": FREQ,
        },
    }


def test_continuation_reuses_stored_region() -> None:
    """Passing only ``initial_state`` continues in the region run 1 used."""
    p = _continuation_problem(nt=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out1 = deepwave.scalar(p["v"], p["dx"], p["dt"], **p["kwargs"], survey_pad=20)
        out2 = deepwave.scalar(
            p["v"], p["dx"], p["dt"], **p["kwargs"], initial_state=out1
        )
    assert out1.info.origin != (0, 0), "survey_pad must actually window the model"
    assert out2.info.origin == out1.info.origin
    assert out2.info.extent == out1.info.extent


def test_continuation_explicit_origin_mismatch_raises() -> None:
    """An explicit ``origin`` that disagrees with the stored one raises.

    Naming both origins: continuing over a different part of the model
    displaces every field, and used to do so silently.
    """
    p = _continuation_problem(nt=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out1 = deepwave.scalar(p["v"], p["dx"], p["dt"], **p["kwargs"], survey_pad=20)
    stored = list(out1.info.origin)
    with pytest.raises(ValueError, match="disagrees with the region") as excinfo:
        deepwave.scalar(
            p["v"],
            p["dx"],
            p["dt"],
            **p["kwargs"],
            initial_state=out1,
            origin=[0, 0],
        )
    message = str(excinfo.value)
    assert "[0, 0]" in message
    assert str(stored) in message


def test_continuation_matching_explicit_origin_is_accepted() -> None:
    """An explicit ``origin`` equal to the stored one is fine.

    ``deepwave.checkpointing.run_segments`` deliberately re-passes the stored
    origin alongside the same state, so the check must key on disagreement,
    not on the presence of both.
    """
    p = _continuation_problem(nt=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out1 = deepwave.scalar(p["v"], p["dx"], p["dt"], **p["kwargs"], survey_pad=20)
        out2 = deepwave.scalar(
            p["v"],
            p["dx"],
            p["dt"],
            **p["kwargs"],
            initial_state=out1,
            origin=list(out1.info.origin),
        )
    assert out2.info.origin == out1.info.origin


def _moved_survey_kwargs(p: dict, row: int) -> dict:
    """``p``'s kwargs with the source and receiver moved to ``row``."""
    kwargs = dict(p["kwargs"])
    kwargs["source_locations"] = torch.tensor([[[row, 10]]])
    kwargs["receiver_locations"] = torch.tensor([[[row, 20]]])
    return kwargs


@pytest.mark.parametrize("form", ["output", "mapping"])
def test_continuation_survey_pad_moved_region_raises(form: str) -> None:
    """The same ``survey_pad`` around a moved survey is a region disagreement.

    The region ``survey_pad`` selects has the same *shape* as the stored one,
    so nothing about the wavefields is ambiguous and no shape check can see
    it: without this the stored wavefields are propagated through a different
    part of the model, silently and with no warning.
    """
    p = _continuation_problem(nt=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out1 = deepwave.scalar(
            p["v"], p["dx"], p["dt"], **_moved_survey_kwargs(p, 10), survey_pad=5
        )
    state = out1 if form == "output" else out1.final_state
    with pytest.raises(ValueError, match="disagrees with the region") as excinfo:
        deepwave.scalar(
            p["v"],
            p["dx"],
            p["dt"],
            **_moved_survey_kwargs(p, 40),
            initial_state=state,
            survey_pad=5,
        )
    message = str(excinfo.value)
    assert "survey_pad" in message
    assert "[35, 5]" in message, message
    assert str(list(out1.info.origin)) in message, message


def test_continuation_survey_pad_same_region_is_accepted() -> None:
    """Re-passing ``survey_pad`` for the same survey continues as before.

    The check is keyed on the region actually selected, not on the presence
    of ``survey_pad`` alongside a state, so the ordinary case of continuing a
    windowed simulation must be untouched.
    """
    p = _continuation_problem(nt=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out1 = deepwave.scalar(p["v"], p["dx"], p["dt"], **p["kwargs"], survey_pad=20)
        out2 = deepwave.scalar(
            p["v"],
            p["dx"],
            p["dt"],
            **p["kwargs"],
            initial_state=out1,
            survey_pad=20,
        )
    assert out1.info.origin != (0, 0), "survey_pad must actually window the model"
    assert out2.info.origin == out1.info.origin


def test_continuation_region_check_reads_the_field_the_recovery_reads() -> None:
    """The check and the origin recovery must key on the same record.

    A ``PropagatorOutput`` carries the region twice: in its ``RunInfo`` and in
    its ``AuxiliaryState``. The recovery (what runs when no ``origin`` is
    given) prefers the ``RunInfo``, so a check that preferred the other one
    could reject the very origin the recovery would have chosen.
    """
    p = _continuation_problem(nt=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out1 = deepwave.scalar(p["v"], p["dx"], p["dt"], **p["kwargs"], survey_pad=20)
    values = list(out1)
    values[list(out1._names).index("auxiliary_state")] = dataclasses.replace(
        out1.auxiliary_state,
        origin=tuple(o + 1 for o in out1.info.origin),
    )
    disagreeing = type(out1)(out1._names, values, info=out1.info)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recovered = deepwave.scalar(
            p["v"], p["dx"], p["dt"], **p["kwargs"], initial_state=disagreeing
        )
        explicit = deepwave.scalar(
            p["v"],
            p["dx"],
            p["dt"],
            **p["kwargs"],
            initial_state=disagreeing,
            origin=list(out1.info.origin),
        )
    assert recovered.info.origin == out1.info.origin
    assert explicit.info.origin == recovered.info.origin


def test_run_segments_origin_error_is_stated_in_its_own_terms() -> None:
    """``run_segments`` manufactures the state, so it must own the message.

    Advice to "pass initial wavefields that belong to the region" is
    unactionable for a caller who never built an ``initial_state``: every
    segment after the first is started from the previous segment's output by
    ``run_segments`` itself.
    """
    p = _continuation_problem(nt=6)
    v = p["v"].clone()
    with pytest.raises(ValueError, match="disagrees with the region") as excinfo:
        deepwave.checkpointing.run_segments(
            deepwave.scalar,
            [v],
            p["dx"],
            p["dt"],
            {"source_amplitudes": p["kwargs"]["source_amplitudes"]},
            2,
            source_locations=p["kwargs"]["source_locations"],
            receiver_locations=p["kwargs"]["receiver_locations"],
            max_vel=p["kwargs"]["max_vel"],
            pml_freq=FREQ,
            origin=[5, 5],
        )
    message = str(excinfo.value)
    assert "run_segments" in message, message
    assert "omit origin=" in message, message
    assert "initial wavefields that belong" not in message, message


def test_continuation_survey_pad_shape_mismatch_names_both_regions() -> None:
    """A wavefield that does not fit the region ``survey_pad`` selects raises.

    The message names the region this call propagates, not only the shapes,
    so the caller can see which part of the model they are about to run.
    """
    p = _continuation_problem(nt=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out1 = deepwave.scalar(p["v"], p["dx"], p["dt"], **p["kwargs"], survey_pad=5)
    with pytest.raises(ValueError, match="spatial shape") as excinfo:
        deepwave.scalar(
            p["v"],
            p["dx"],
            p["dt"],
            **p["kwargs"],
            initial_state=out1.final_state,
            survey_pad=20,
        )
    message = str(excinfo.value)
    assert "spatial shape" in message
    assert "origin=" in message
    assert "extent=" in message


def test_continuation_location_outside_stored_region_raises() -> None:
    """A source outside the stored region raises rather than being clipped."""
    p = _continuation_problem(nt=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out1 = deepwave.scalar(p["v"], p["dx"], p["dt"], **p["kwargs"], survey_pad=5)
    kwargs = dict(p["kwargs"])
    kwargs["source_locations"] = torch.tensor([[[1, 1]]])
    with pytest.raises(RuntimeError, match="survey extents"):
        deepwave.scalar(p["v"], p["dx"], p["dt"], **kwargs, initial_state=out1)


def _clear_warning_registry() -> None:
    from deepwave import _common, _engine

    _engine._warned_once_causes.clear()
    _common._warned_once_causes.clear()


def test_mapping_initial_state_without_origin_warns() -> None:
    """A bare mapping records no region, so assuming origin zero is announced.

    The warning is :func:`get_survey_extents_from_wavefields`'s, which reports
    the extents it actually inferred; this pins that it still names the
    remedy for a continuation, so nobody trims that sentence away.
    """
    p = _continuation_problem(nt=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out1 = deepwave.scalar(p["v"], p["dx"], p["dt"], **p["kwargs"], survey_pad=20)
    mapping = {
        "wavefield_0": out1.wavefield_0,
        "wavefield_m1": out1.wavefield_m1,
    }
    _clear_warning_registry()
    with pytest.warns(UserWarning, match="origin"):
        deepwave.scalar(p["v"], p["dx"], p["dt"], **p["kwargs"], initial_state=mapping)


@pytest.mark.parametrize("case", ["propagator_output", "with_origin", "with_pad"])
def test_mapping_origin_warning_does_not_fire_when_region_is_known(
    case: str,
) -> None:
    """The warning is about a missing region, so it must not fire otherwise."""
    p = _continuation_problem(nt=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out1 = deepwave.scalar(p["v"], p["dx"], p["dt"], **p["kwargs"], survey_pad=20)
    mapping = {
        "wavefield_0": out1.wavefield_0,
        "wavefield_m1": out1.wavefield_m1,
    }
    extra: dict[str, Any]
    if case == "propagator_output":
        extra = {"initial_state": out1}
    elif case == "with_origin":
        extra = {"initial_state": mapping, "origin": list(out1.info.origin)}
    else:
        extra = {"initial_state": mapping, "survey_pad": 20}

    _clear_warning_registry()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deepwave.scalar(p["v"], p["dx"], p["dt"], **p["kwargs"], **extra)
    offenders = [
        str(w.message) for w in caught if "no record of the region" in str(w.message)
    ]
    assert not offenders, offenders


def test_final_state_mapping_recovers_origin_from_auxiliary_state() -> None:
    """``out.final_state`` carries an ``AuxiliaryState``, so it knows its region."""
    p = _continuation_problem(nt=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out1 = deepwave.scalar(p["v"], p["dx"], p["dt"], **p["kwargs"], survey_pad=20)
        out2 = deepwave.scalar(
            p["v"],
            p["dx"],
            p["dt"],
            **p["kwargs"],
            initial_state=out1.final_state,
        )
    assert out2.info.origin == out1.info.origin


# --------------------------------------------------------------------------
# The default Courant number is single-sourced
# --------------------------------------------------------------------------


def test_default_courant_number_is_single_sourced() -> None:
    """The exported default is the very constant the propagators use."""
    from deepwave._common import _DEFAULT_COURANT_NUMBER

    assert deepwave.DEFAULT_COURANT_NUMBER == 0.6
    assert deepwave.DEFAULT_COURANT_NUMBER is _DEFAULT_COURANT_NUMBER
    assert (
        deepwave.discretisation.DEFAULT_COURANT_NUMBER
        is deepwave.DEFAULT_COURANT_NUMBER
    )
    assert "DEFAULT_COURANT_NUMBER" in deepwave.__all__


def test_default_courant_number_reproduces_the_default_run() -> None:
    """Passing the constant explicitly is byte-identical to leaving it None."""
    p = _problem()
    a = deepwave.scalar(p["v"], p["dx"], p["dt"], **p["kwargs"])
    b = deepwave.scalar(
        p["v"],
        p["dx"],
        p["dt"],
        **p["kwargs"],
        courant_number=deepwave.DEFAULT_COURANT_NUMBER,
    )
    assert torch.equal(a.receiver_amplitudes, b.receiver_amplitudes)
    assert a.info.courant_number == b.info.courant_number


def test_n_shots_from_initial_wavefields_when_no_sources() -> None:
    """With ``nt`` and no sources, the initial wavefields set the shot count."""
    n, n_shots = 40, 3
    v = torch.full((n, n), 1500.0, dtype=DTYPE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seeded = deepwave.scalar(
            v,
            5.0,
            0.001,
            nt=4,
            max_vel=2500.0,
            pml_freq=FREQ,
            source_amplitudes=torch.zeros(n_shots, 1, 4, dtype=DTYPE),
            source_locations=torch.tensor([[[n // 2, n // 2]]]).repeat(n_shots, 1, 1),
        )
        out = deepwave.scalar(
            v,
            5.0,
            0.001,
            nt=4,
            max_vel=2500.0,
            pml_freq=FREQ,
            initial_state=seeded,
        )
    assert out.wavefield_0.shape[0] == n_shots


def test_no_sources_and_no_wavefields_raises() -> None:
    """With neither sources nor initial wavefields, the shot count is unknowable."""
    v = torch.full((40, 40), 1500.0, dtype=DTYPE)
    with pytest.raises(RuntimeError, match="at least one source or initial"):
        deepwave.scalar(v, 5.0, 0.001, nt=4, max_vel=2500.0, pml_freq=FREQ)


# ---------------------------------------------------------------------------
# One name, one quantity
#
# A callback's names are a single vocabulary spanning nine propagators and
# two spaces (the public model arguments and the internal state). A name that
# appears twice in it must mean the same thing both times, or a callback
# written against one propagator silently computes something else on the
# next. Keep this invariant explicit even though every retained propagator
# currently uses one quantity per callback name.
# ---------------------------------------------------------------------------

_NAME_N = 10


def _one_medium() -> dict[str, torch.Tensor]:
    """One physical medium, expressed in every propagator's own parameters.

    Spatially varying, and varying *differently* along each axis, so that two
    quantities that merely happen to have the same units are not equal by
    accident: a constant medium would make ``c11`` and ``c33`` equal and the
    comparisons below vacuous. The isotropic and anisotropic parameter sets
    describe different media (the stiffnesses are deliberately anisotropic),
    which is harmless because only *shared names* are compared and the two
    families share only the buoyancies -- and those come from the one ``rho``.
    """
    n = _NAME_N
    y = torch.arange(n, dtype=DTYPE).reshape(n, 1)
    x = torch.arange(n, dtype=DTYPE).reshape(1, n)
    rho = 1800.0 + 40.0 * y + 7.0 * x
    vp = 1800.0 + 30.0 * y + 11.0 * x
    vs = 900.0 + 12.0 * y + 5.0 * x
    mu = rho * vs**2
    return {
        "rho": rho,
        "v": vp,
        "vs": vs,
        "mu": mu,
        "lamb": rho * vp**2 - 2.0 * mu,
        "buoyancy": 1.0 / rho,
        "c33": rho * vp**2,
        "c11": 1.15 * rho * vp**2,
        "c13": rho * vp**2 - 2.0 * mu,
        "c55": mu,
        "q": 30.0 + y + x,
        "qp": 30.0 + y + x,
        "qs": 20.0 + 2.0 * y + x,
        "zero": torch.zeros(n, n, dtype=DTYPE),
    }


def _public_models(name: str, medium: dict[str, torch.Tensor]) -> dict[str, Any]:
    """``name``'s public model arguments, all drawn from the one medium."""
    m = medium
    positional = {
        "scalar": ("v",),
        "scalar_born": ("v",),
        "acoustic": ("v", "rho"),
        "acoustic_born": ("v", "rho"),
        "acoustic_vti": ("c11", "c13", "c33", "buoyancy"),
        "viscoacoustic": ("v", "rho", "q"),
        "elastic": ("lamb", "mu", "buoyancy"),
        "elastic_born": ("lamb", "mu", "buoyancy"),
        "elastic_vti": ("c11", "c13", "c33", "c55", "buoyancy"),
    }[name]
    keyword: dict[str, Any] = {}
    if name == "scalar_born":
        keyword["dv"] = m["zero"]
    elif name == "acoustic_born":
        keyword["dv"] = m["zero"]
        keyword["drho"] = m["zero"]
    elif name == "elastic_born":
        keyword["dlamb"] = m["zero"]
        keyword["dmu"] = m["zero"]
        keyword["dbuoyancy"] = m["zero"]
    return {"positional": [m[p] for p in positional], "keyword": keyword}


def _callback_models(name: str) -> dict[str, torch.Tensor]:
    """Every internal model a forward callback of ``name`` can read, by name."""
    from deepwave import _capabilities as cap

    medium = _one_medium()
    built = _public_models(name, medium)
    seen: dict[str, torch.Tensor] = {}
    internal = deepwave._engine.propagator_spec(name).model_names(2)

    def record(state: deepwave.CallbackState) -> None:
        if seen:
            return
        for model in internal:
            assert state.writable(model, "model"), f"{name}: {model} not writable"
            seen[model] = state.get_model(model, view="inner").detach().clone()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cap._call(
            name,
            2,
            None,
            forward_callback=record,
            callback_frequency=1,
            _models_override=built["positional"],
            **built["keyword"],
        )
    return seen


@pytest.fixture(scope="module")
def callback_models() -> dict[str, dict[str, torch.Tensor]]:
    """Every propagator's internal models, all built from the one medium."""
    return {name: _callback_models(name) for name in PROPAGATOR_NAMES}


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_a_name_valid_publicly_and_internally_is_one_quantity(
    name: str, callback_models: dict[str, dict[str, torch.Tensor]]
) -> None:
    """A public model name reused internally names the same tensor.

    The propagator is given a medium whose parameters all differ from each
    other, so a name that is both a public argument and an internal model
    passes this only if the internal tensor really is the argument.
    """
    medium = _one_medium()
    public = dict(
        zip(
            deepwave.spec(name).model_names(2),
            _public_models(name, medium)["positional"],
        )
    )
    shared = sorted(set(public) & set(callback_models[name]))
    for model in shared:
        got = callback_models[name][model]
        want = public[model].expand_as(got)
        assert torch.equal(got, want), (
            f"{name}: the internal model {model!r} is not the public "
            f"argument of the same name; one of the two needs a name of "
            f"its own."
        )


def test_a_name_valid_on_two_propagators_is_one_quantity(
    callback_models: dict[str, dict[str, torch.Tensor]],
) -> None:
    """An internal name shared by two propagators names the same tensor.

    Every propagator is given the same medium, so any shared internal name
    that denotes the same physical quantity, prepared the same way, holds
    identical numbers. A name that two propagators use for two different
    quantities fails here, and the fix is a name of its own rather than a
    footnote.
    """
    holders: dict[str, list[str]] = {}
    for name, models in callback_models.items():
        for model in models:
            holders.setdefault(model, []).append(name)
    shared = {m: names for m, names in holders.items() if len(names) > 1}
    assert shared, "no internal model name is shared; the test has stopped testing"
    for model, names in sorted(shared.items()):
        first = names[0]
        for other in names[1:]:
            assert torch.equal(
                callback_models[first][model], callback_models[other][model]
            ), (
                f"{model!r} is a different quantity on {first} and {other}, "
                f"although they spell it the same."
            )


def test_in_place_model_write_from_a_callback_changes_the_simulation() -> None:
    """Writing into ``get_model``'s result in place is supported, and bites.

    The promise :meth:`deepwave.CallbackState.writable` makes for
    ``kind="model"``, and what ``v0.0.27`` already did. There is no
    ``set_model``: writing in place *is* the route, so it has to work rather
    than merely happen to.
    """
    p = _problem()
    results = {}
    for label, factor in (("noop", 1.0), ("scaled", 0.81)):

        def cb(state: deepwave.CallbackState, factor: float = factor) -> None:
            if state.step == 0:
                assert state.writable("v", "model")
                state.get_model("v", view="full").mul_(factor)

        results[label] = deepwave.scalar(
            p["v"],
            p["dx"],
            p["dt"],
            **p["kwargs"],
            forward_callback=cb,
            callback_frequency=1,
        ).receiver_amplitudes.clone()

    assert not torch.equal(results["noop"], results["scaled"])
