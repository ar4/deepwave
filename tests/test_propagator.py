"""Tests for deepwave.propagator."""

import re
from typing import Any

import pytest
import torch
from helpers import elements_equal

import deepwave
from deepwave._engine import receiver_output_names
from deepwave.checkpointing import _get_spec

NZ = 5
NY = 3
DX = 5.0


@pytest.fixture
def prop() -> deepwave.Scalar:
    """Return a propagator."""
    model = torch.ones(NZ, NY) * 1500
    dx = DX
    return deepwave.Scalar(model, dx)


def test_passes(prop: deepwave.Scalar) -> None:
    """Check that the test passes when everything is correct."""
    source_amplitudes = torch.zeros(1, 1, 2)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    dt = 0.004
    prop(
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
    )


@pytest.mark.parametrize(
    ("arg_name", "arg_value", "expected_error_match", "expected_error_type"),
    [
        ("accuracy", 3, "accuracy must be 2, 4, 6, or 8", ValueError),
        ("accuracy", 4.0, "accuracy must be an int.", TypeError),
        ("pml_width", -1, "pml_width must be non-negative.", ValueError),
        (
            "pml_width",
            [1, 2, 3],
            "Expected pml_width to be of length 1 or 4, got 3.",
            RuntimeError,
        ),
        ("pml_freq", -1.0, "pml_freq must be non-negative.", ValueError),
        (
            "model_gradient_sampling_interval",
            -1,
            "model_gradient_sampling_interval must be >= 1",
            ValueError,
        ),
        (
            "model_gradient_sampling_interval",
            0,
            "model_gradient_sampling_interval must be >= 1",
            ValueError,
        ),
        ("freq_taper_frac", -0.1, r"freq_taper_frac must be in \[0, 1\]", ValueError),
        ("time_pad_frac", 1.1, r"time_pad_frac must be in \[0, 1\]", ValueError),
        ("time_taper", "invalid", "time_taper must be a bool.", TypeError),
    ],
)
def test_scalar_forward_invalid_optional_args(
    prop: deepwave.Scalar,
    arg_name: str,
    arg_value: Any,
    expected_error_match: str,
    expected_error_type: Any,
) -> None:
    """Test scalar raises for invalid optional arguments."""
    source_amplitudes = torch.zeros(1, 1, 2)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    dt = 0.004
    kwargs = {arg_name: arg_value}
    if expected_error_type is UserWarning:
        with pytest.warns(expected_error_type, match=expected_error_match):
            prop(
                dt,
                source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                **kwargs,
            )
    else:
        with pytest.raises(expected_error_type, match=expected_error_match):
            prop(
                dt,
                source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                **kwargs,
            )


@pytest.mark.parametrize(
    (
        "source_amplitudes_shape",
        "source_locations_shape",
        "receiver_locations_shape",
        "exception_type",
        "expected_error_match",
    ),
    [
        (
            (1, 1),
            (1, 1, 2),
            (1, 1, 2),
            RuntimeError,
            "source amplitudes Tensors should have 3 dimensions",
        ),  # wrong num dims
        (
            (1, 1, 2),
            (2, 1, 2),
            (1, 1, 2),
            RuntimeError,
            "source_amplitudes has shape (1, 1, 2), but Deepwave expects "
            "[shot, source, time] = [2, *, 2]",
        ),  # shape mismatch 1
        (
            (1, 1, 2),
            (1, 2, 2),
            (1, 1, 2),
            RuntimeError,
            "Expected source amplitudes and locations to be the same size in the "
            "n_sources_per_shot dimension",
        ),  # shape mismatch 2
        (
            (1, 1, 2),
            (1, 1, 3),
            (1, 1, 2),
            ValueError,
            "The number of spatial coordinates in a location "
            "Tensor does not match expectations: "
            "Expected 3, but found 2.",
        ),  # shape mismatch 3
        (
            (2, 1, 1),
            (1, 1, 2),
            (1, 1, 2),
            RuntimeError,
            "source_amplitudes has shape (2, 1, 1), but Deepwave expects "
            "[shot, source, time] = [1, *, 1]",
        ),  # shape mismatch 4
    ],
)
def test_input_shape_mismatch(
    prop: deepwave.Scalar,
    source_amplitudes_shape: tuple[int, ...],
    source_locations_shape: tuple[int, ...],
    receiver_locations_shape: tuple[int, ...],
    exception_type: Exception,
    expected_error_match: str,
) -> None:
    """Test that scalar forward propagation raises errors for input shape mismatches."""
    source_amplitudes = torch.zeros(source_amplitudes_shape)
    source_locations = torch.zeros(source_locations_shape, dtype=torch.long)
    receiver_locations = torch.zeros(receiver_locations_shape, dtype=torch.long)
    dt = 0.004
    with pytest.raises(exception_type, match=re.escape(expected_error_match)):
        prop(
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
        )


def test_source_outside_model(prop: deepwave.Scalar) -> None:
    """Check error when source location not inside model."""
    source_amplitudes = torch.zeros(1, 1, 2)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    source_locations[0, 0, 0] = -1.0
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    dt = 0.004
    with pytest.raises(RuntimeError, match=r"source_locations is out of bounds"):
        prop(
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
        )


def test_receiver_outside_model(prop: deepwave.Scalar) -> None:
    """Check error when receiver location not inside model."""
    source_amplitudes = torch.zeros(1, 1, 2)
    source_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(1, 1, 2, dtype=torch.long)
    receiver_locations[0, 0, 1] = NY
    dt = 0.004
    with pytest.raises(RuntimeError, match=r"receiver_locations is out of bounds"):
        prop(
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
        )


@pytest.mark.parametrize(
    ("wavefield_shape", "expected_error_match"),
    [
        (
            (3, NZ + 2 * 20, NY + 2 * 20),
            "Inconsistent batch size",
        ),  # batch size too big
        (
            (1, NZ + 2 * 20, NY + 2 * 20),
            "Inconsistent batch size",
        ),  # batch size too small
        (
            (2, NZ + 2 * 20 + 1, NY + 2 * 20),
            "Survey extents are larger than the model.",
        ),  # too big in Z
        (
            (2, NZ + 2 * 20, NY + 2 * 20 + 1),
            "Survey extents are larger than the model.",
        ),  # too big in Y
    ],
)
def test_wavefield_shape_mismatch(
    prop: deepwave.Scalar, wavefield_shape: tuple[int, ...], expected_error_match: str
) -> None:
    """Test scalar raises for wavefield shape mismatches."""
    source_amplitudes = torch.zeros(2, 1, 2)
    source_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    dt = 0.004
    wavefield_0 = torch.zeros(wavefield_shape)
    with pytest.raises(RuntimeError, match=re.escape(expected_error_match)):
        prop(
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            wavefield_0=wavefield_0,
        )


def test_pml_width_list_too_long(prop: deepwave.Scalar) -> None:
    """Check error when pml_width list is too long."""
    source_amplitudes = torch.zeros(2, 1, 2)
    source_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    receiver_locations = torch.zeros(2, 1, 2, dtype=torch.long)
    dt = 0.004
    with pytest.raises(
        RuntimeError,
        match=r"Expected pml_width to be of length 1 or 4, got 5.",
    ):
        prop(
            dt,
            source_amplitudes=source_amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_width=[1, 20, 20, 20, 20],
        )


# --- B5: optional named/dataclass returns -----------------------------------


def _named_return_propagators(ndim: int) -> dict:
    """Runs every propagator (ndim spatial dims) and returns name -> output."""
    torch.manual_seed(ndim)
    shape = (10, 11, 12)[:ndim]
    dx = 5.0
    dt = 0.0005
    nt = 6
    src_amp = torch.randn(1, 1, nt)
    centre = [s // 2 for s in shape]
    src_loc = torch.tensor([[centre]])
    rec_loc = torch.tensor([[centre, [c + 1 for c in centre]]])
    v = 1500 * torch.ones(*shape)
    rho = 2200 * torch.ones(*shape)
    lamb = 2200 * 1500.0**2 * torch.ones(*shape)
    mu = 2200 * 900.0**2 * torch.ones(*shape)
    buoyancy = (1 / 2200) * torch.ones(*shape)
    scatter = torch.randn(*shape)
    return {
        "scalar": deepwave.scalar(
            v,
            dx,
            dt,
            source_amplitudes=src_amp,
            source_locations=src_loc,
            receiver_locations=rec_loc,
        ),
        "scalar_born": deepwave.scalar_born(
            v,
            dx,
            dt,
            dv=scatter,
            source_amplitudes=src_amp,
            source_locations=src_loc,
            sc_receiver_locations=rec_loc,
            bg_receiver_locations=rec_loc,
        ),
        "acoustic": deepwave.acoustic(
            v,
            rho,
            dx,
            dt,
            source_amplitudes_volume_rate=src_amp,
            source_locations_volume_rate=src_loc,
            receiver_locations_p=rec_loc,
        ),
        "elastic": deepwave.elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            dt,
            source_amplitudes_y=src_amp,
            source_locations_y=src_loc,
            receiver_locations_p=rec_loc,
        ),
    }


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("name", ["scalar", "scalar_born", "acoustic", "elastic"])
def test_run_info_state_names_are_in_return_order(name: str, ndim: int) -> None:
    """``info.wavefield_names``/``final_state`` follow the return order.

    The elastic family's public field order is a permutation of the backend's,
    so recording the backend order here would contradict what both are
    documented to be, and would make ``final_state`` iterate in an order the
    return never uses.
    """
    out = _named_return_propagators(ndim)[name]
    receivers = receiver_output_names(_get_spec(getattr(deepwave, name)))
    state = [field for field in out.names if field not in receivers]
    assert list(out.info.wavefield_names) == state
    assert list(out.info.receiver_names) == [
        field for field in out.names if field in receivers
    ]
    assert list(out.final_state) == state
    assert out.info.ndim == ndim


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("name", ["scalar", "scalar_born", "acoustic", "elastic"])
def test_named_return_matches_positional(name: str, ndim: int) -> None:
    """The default return is a PropagatorOutput whose tuple view matches it."""
    out = _named_return_propagators(ndim)[name]
    assert isinstance(out, deepwave.PropagatorOutput)
    # PropagatorOutput subclasses tuple, so converting to a plain tuple must
    # match it element-wise.
    positional = tuple(out)
    assert isinstance(positional, tuple)
    assert not isinstance(positional, deepwave.PropagatorOutput)
    assert len(positional) == len(out)
    for got, want in zip(positional, out):
        assert elements_equal(got, want)
    # Indexing, slicing and unpacking all continue to work.
    assert torch.equal(out[-1], positional[-1])
    assert all(elements_equal(a, b) for a, b in zip(out[:2], positional[:2]))
    *head, tail = out
    assert torch.equal(tail, positional[-1])
    assert len(head) == len(positional) - 1


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("name", ["scalar", "scalar_born", "acoustic", "elastic"])
def test_named_return_field_names(name: str, ndim: int) -> None:
    """Field names cover the outputs and give ndim-independent receiver access."""
    named = _named_return_propagators(ndim)[name]
    fields = list(named._asdict().keys())
    assert len(fields) == len(named)
    assert len(set(fields)) == len(fields)  # unique
    # Every propagator exposes its receiver data by a stable name regardless of
    # ndim, which is the whole point of the feature. The expected positional
    # index of that named field is what would otherwise shift with ndim. The
    # scalar propagators have a single (scattered) receiver at the very end;
    # acoustic/elastic have ``ndim + 1`` receivers with pressure first.
    receiver_field, receiver_index = {
        "scalar": ("receiver_amplitudes", len(named) - 1),
        "scalar_born": ("sc_receiver_amplitudes", len(named) - 1),
        "acoustic": ("receiver_amplitudes_p", len(named) - 1 - ndim),
        "elastic": ("receiver_amplitudes_p", len(named) - 1 - ndim),
    }[name]
    assert torch.equal(getattr(named, receiver_field), named[receiver_index])
    # scalar_born exposes both background and scattered receivers by name.
    if name == "scalar_born":
        assert torch.equal(named.bg_receiver_amplitudes, named[-2])
        assert torch.equal(named.sc_receiver_amplitudes, named[-1])


def test_named_return_unknown_field_raises() -> None:
    """Accessing a field that does not exist gives a helpful AttributeError."""
    named = _named_return_propagators(2)["scalar"]
    with pytest.raises(AttributeError, match="has no field 'not_a_field'"):
        _ = named.not_a_field
    # A 3-D-only field is absent in a 2-D run.
    with pytest.raises(AttributeError, match="has no field 'psiz_m1'"):
        _ = named.psiz_m1


def test_named_return_rejects_colliding_field_name() -> None:
    """A field name that would shadow a class attribute is rejected loudly."""
    t = torch.zeros(1)
    info = _small_run_info()
    # ``index`` and ``count`` are abc.Sequence methods; the rest are ours.
    # ``_final_state`` is the instance-``__dict__`` cache ``final_state``
    # writes: it is not a class attribute, so only the reserved-name list
    # catches it.
    for bad in (
        "index",
        "count",
        "_asdict",
        "names",
        "as_dict",
        "final_state",
        "info",
        "_final_state",
    ):
        with pytest.raises(AssertionError, match="collides"):
            deepwave.PropagatorOutput([bad], [t], info=info)
    # A normal field name is fine and round-trips.
    out = deepwave.PropagatorOutput(["receiver_amplitudes"], [t], info=info)
    assert out.receiver_amplitudes is t
    # ``auxiliary_state`` shadows a property written to return that very field,
    # so it is the one name allowed through.
    state = deepwave.scalar(
        1500 * torch.ones(12, 12),
        5.0,
        0.001,
        source_amplitudes=torch.zeros(1, 1, 3),
        source_locations=torch.tensor([[[6, 6]]]),
        receiver_locations=torch.tensor([[[7, 7]]]),
        pml_freq=25.0,
    ).auxiliary_state
    folded = deepwave.PropagatorOutput(["auxiliary_state"], [state], info=info)
    assert folded.auxiliary_state is state
    assert folded[0] is state


def _small_run_info() -> deepwave.RunInfo:
    """A minimal RunInfo, for hand-built PropagatorOutputs."""
    return deepwave.RunInfo(
        step_ratio=1,
        internal_dt=0.001,
        courant_number=0.6,
        nt=1,
        pml_width=(0, 0, 0, 0),
        pml_freq=25.0,
        origin=(0, 0),
        extent=(1, 1),
    )


def test_named_return_preserves_gradients() -> None:
    """Tensors in the named return are differentiable just like positional."""
    shape = (12, 13)
    dx, dt, nt = 5.0, 0.0005, 6
    v = (1500 * torch.ones(*shape)).requires_grad_(True)
    src_amp = torch.randn(1, 1, nt)
    src_loc = torch.tensor([[[6, 6]]])
    rec_loc = torch.tensor([[[7, 7]]])
    named = deepwave.scalar(
        v,
        dx,
        dt,
        source_amplitudes=src_amp,
        source_locations=src_loc,
        receiver_locations=rec_loc,
    )
    named.receiver_amplitudes.sum().backward()
    assert v.grad is not None
    assert torch.any(v.grad != 0)


# ---------------------------------------------------------------------------
# The optional initial_state bundle (the input-bundle counterpart of the
# output bundle).
#
# Because each propagator names its output wavefields identically to its
# initial-wavefield arguments, feeding a previous PropagatorOutput back in as
# ``initial_state`` must continue the simulation exactly (the basis of
# time-chaining and domain decomposition). These tests pin that round trip plus
# the equivalence, error, and gradient behaviour of the bundle.
# ---------------------------------------------------------------------------


def _initial_state_case(name: str, ndim: int) -> dict:
    """Builds a single-source/single-receiver run closure for one propagator.

    Returns a dict with ``run`` (a callable taking a source-amplitude slice and
    an optional ``initial_state`` and returning the PropagatorOutput),
    ``full_amp`` (the whole source wavelet), ``split`` (where to cut it), and
    ``rec_field`` (the primary receiver field name to concatenate/compare).
    """
    torch.manual_seed(100 + ndim)
    shape = (12, 13, 14)[:ndim]
    dx, dt, nt = 5.0, 0.0005, 12
    full_amp = torch.randn(1, 1, nt, dtype=torch.float64)
    centre = [s // 2 for s in shape]
    src_loc = torch.tensor([[centre]])
    rec_loc = torch.tensor([[centre, [c + 1 for c in centre]]])
    v = 1500 * torch.ones(*shape, dtype=torch.float64)
    rho = 2200 * torch.ones(*shape, dtype=torch.float64)
    lamb = 2200 * 1500.0**2 * torch.ones(*shape, dtype=torch.float64)
    mu = 2200 * 900.0**2 * torch.ones(*shape, dtype=torch.float64)
    buoyancy = (1 / 2200) * torch.ones(*shape, dtype=torch.float64)
    scatter = torch.randn(*shape, dtype=torch.float64)
    # Pin max_vel/pml_freq and pass an explicit origin so the chained segments
    # use an identical internal time step and stop warning about extents.
    common = {
        "dt": dt,
        "max_vel": 2000.0,
        "pml_freq": 15.0,
        "pml_width": 8,
        "origin": [0] * ndim,
    }

    def run_scalar(amp: torch.Tensor, **extra: Any) -> Any:
        return deepwave.scalar(
            v,
            dx,
            source_amplitudes=amp,
            source_locations=src_loc,
            receiver_locations=rec_loc,
            **common,
            **extra,
        )

    def run_born(amp: torch.Tensor, **extra: Any) -> Any:
        return deepwave.scalar_born(
            v,
            dx,
            source_amplitudes=amp,
            source_locations=src_loc,
            sc_receiver_locations=rec_loc,
            bg_receiver_locations=rec_loc,
            dv=scatter,
            **common,
            **extra,
        )

    def run_acoustic(amp: torch.Tensor, **extra: Any) -> Any:
        return deepwave.acoustic(
            v,
            rho,
            dx,
            source_amplitudes_volume_rate=amp,
            source_locations_volume_rate=src_loc,
            receiver_locations_p=rec_loc,
            **common,
            **extra,
        )

    def run_elastic(amp: torch.Tensor, **extra: Any) -> Any:
        # Use the x-component force/receiver: x is the only dimension present in
        # every ndim (1/2/3-D), unlike y/z.
        return deepwave.elastic(
            lamb,
            mu,
            buoyancy,
            dx,
            source_amplitudes_x=amp * 1e6,
            source_locations_x=src_loc,
            receiver_locations_x=rec_loc,
            **common,
            **extra,
        )

    run, rec_field = {
        "scalar": (run_scalar, "receiver_amplitudes"),
        "scalar_born": (run_born, "sc_receiver_amplitudes"),
        "acoustic": (run_acoustic, "receiver_amplitudes_p"),
        "elastic": (run_elastic, "receiver_amplitudes_x"),
    }[name]
    return {
        "run": run,
        "full_amp": full_amp,
        "split": 7,
        "rec_field": rec_field,
    }


_PROP_NAMES = ["scalar", "scalar_born", "acoustic", "elastic"]


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("name", _PROP_NAMES)
def test_initial_state_continues_simulation(name: str, ndim: int) -> None:
    """Feeding a previous output back as initial_state continues exactly.

    Running the whole wavelet in one call must equal running it in two
    segments, where the second segment is started from the first segment's
    output via ``initial_state``: identical final wavefields and identical
    (concatenated) receiver data, to bit precision.
    """
    case = _initial_state_case(name, ndim)
    run, amp, split, rec = (
        case["run"],
        case["full_amp"],
        case["split"],
        case["rec_field"],
    )
    # Bit-exact continuation across a segment boundary requires the same
    # internal time step throughout, i.e. no CFL sub-stepping (step_ratio == 1).
    # The case parameters are chosen to satisfy this; assert it so the premise
    # cannot silently drift (with step_ratio > 1 the boundary introduces a small
    # discretisation difference and the equalities below would not hold).
    _, step_ratio = deepwave.discretisation.cfl_condition_n(
        [5.0] * ndim, 0.0005, 2000.0
    )
    assert step_ratio == 1, step_ratio
    full = run(amp)
    seg1 = run(amp[..., :split])
    seg2 = run(amp[..., split:], initial_state=seg1)

    # Every final wavefield matches the monolithic run bit-for-bit. The
    # ``auxiliary_state`` field is compared tensor by tensor.
    for field, value in full.as_dict().items():
        if "receiver" in field:
            continue
        assert elements_equal(value, seg2.as_dict()[field]), field

    # The receiver data, concatenated across the two segments, matches.
    chained = torch.cat([getattr(seg1, rec), getattr(seg2, rec)], dim=-1)
    assert torch.equal(getattr(full, rec), chained)


def _auxiliary_state_names(name: str, ndim: int) -> set[str]:
    """The wavefield names ``name`` carries in its auxiliary state, if any."""
    spec = _get_spec(getattr(deepwave, name))
    return set(spec.auxiliary_state_names(ndim))


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("name", _PROP_NAMES)
def test_initial_state_matches_explicit_kwargs(name: str, ndim: int) -> None:
    """A bundled initial_state equals passing the same wavefields as kwargs.

    The split between "explicit keyword argument" and "only reachable through
    ``initial_state``" is now the fold itself: every public state field is a
    keyword argument, and the one field that is not -- ``auxiliary_state``,
    which holds exactly ``spec.auxiliary_state_names(ndim)`` -- travels in a
    (nested) ``initial_state`` bundle. Nothing here consults the old
    ``psi*``/``zeta*``/``phi_*``/``m_*`` name taxonomy that the fold replaced.
    """
    case = _initial_state_case(name, ndim)
    run, amp, split = case["run"], case["full_amp"], case["split"]
    seg1 = run(amp[..., :split])
    via_bundle = run(amp[..., split:], initial_state=seg1)
    kwargs = {}
    extra_state = {}
    for field, value in seg1.as_dict().items():
        if "receiver" in field:
            continue
        if field == "auxiliary_state":
            extra_state[field] = value
        else:
            kwargs[field] = value
    # The bundled field really is the whole absorbing-boundary block, so the
    # kwargs above are the entire remainder of the state -- otherwise this
    # comparison could pass while quietly dropping a field on both sides.
    assert set(extra_state["auxiliary_state"].names) == _auxiliary_state_names(
        name, ndim
    )
    kwargs["initial_state"] = extra_state
    via_kwargs = run(amp[..., split:], **kwargs)
    for field, value in via_bundle.as_dict().items():
        assert elements_equal(value, via_kwargs.as_dict()[field]), field


def test_initial_state_accepts_plain_mapping() -> None:
    """initial_state also accepts a plain dict of field-name -> value.

    The public keys are the propagator's output field names, so the absorbing
    boundary's memory variables travel as the single ``auxiliary_state`` entry
    rather than one key each.
    """
    case = _initial_state_case("scalar", 2)
    run, amp, split = case["run"], case["full_amp"], case["split"]
    seg1 = run(amp[..., :split])
    mapping = {
        "wavefield_0": seg1.wavefield_0,
        "wavefield_m1": seg1.wavefield_m1,
        "auxiliary_state": seg1.auxiliary_state,
    }
    via_dict = run(amp[..., split:], initial_state=mapping)
    via_output = run(amp[..., split:], initial_state=seg1)
    assert torch.equal(via_dict.receiver_amplitudes, via_output.receiver_amplitudes)


def test_initial_state_rejects_the_per_field_mapping() -> None:
    """The pre-fold one-key-per-memory-variable mapping is not accepted.

    ``v0.0.27`` had no ``initial_state`` at all -- it took each memory
    variable as its own keyword argument -- so this spelling was never a
    released one. It raises, naming the propagator, every offending key and
    the bundle that replaces them. The twin above is the supported spelling.
    """
    case = _initial_state_case("scalar", 2)
    run, amp, split = case["run"], case["full_amp"], case["split"]
    seg1 = run(amp[..., :split])
    mapping = {
        "wavefield_0": seg1.wavefield_0,
        "wavefield_m1": seg1.wavefield_m1,
        "psiy_m1": seg1.auxiliary_state["psiy_m1"],
        "psix_m1": seg1.auxiliary_state["psix_m1"],
        "zetay_m1": seg1.auxiliary_state["zetay_m1"],
        "zetax_m1": seg1.auxiliary_state["zetax_m1"],
    }
    with pytest.raises(deepwave.DeepwaveValueError) as excinfo:
        run(amp[..., split:], initial_state=mapping)
    message = str(excinfo.value)
    assert message.startswith("scalar: ")
    assert "initial_state=out" in message
    for key in ("psiy_m1", "psix_m1", "zetay_m1", "zetax_m1"):
        assert repr(key) in message


def test_initial_state_conflict_raises() -> None:
    """Giving a wavefield both in initial_state and explicitly is an error."""
    case = _initial_state_case("scalar", 2)
    run, amp, split = case["run"], case["full_amp"], case["split"]
    seg1 = run(amp[..., :split])
    with pytest.raises(ValueError, match="both in initial_state"):
        run(amp[..., split:], initial_state=seg1, wavefield_0=seg1.wavefield_0)


def test_initial_state_unknown_key_raises() -> None:
    """An unrecognised key in initial_state is rejected (catches typos)."""
    case = _initial_state_case("scalar", 2)
    run, amp, split = case["run"], case["full_amp"], case["split"]
    seg1 = run(amp[..., :split])
    with pytest.raises(ValueError, match="unexpected key 'wavefield_typo'"):
        run(amp[..., split:], initial_state={"wavefield_typo": seg1.wavefield_0})


def test_initial_state_none_entry_is_not_a_conflict() -> None:
    """A None value in the bundle never conflicts with an explicit argument."""
    case = _initial_state_case("scalar", 2)
    run, amp, split = case["run"], case["full_amp"], case["split"]
    seg1 = run(amp[..., :split])
    # wavefield_0 None in the bundle, given explicitly: no conflict, and the
    # explicit value is used (equivalent to a full-bundle continuation).
    mixed = {
        "wavefield_0": None,
        "wavefield_m1": seg1.wavefield_m1,
        "auxiliary_state": seg1.auxiliary_state,
    }
    via_mixed = run(amp[..., split:], initial_state=mixed, wavefield_0=seg1.wavefield_0)
    via_output = run(amp[..., split:], initial_state=seg1)
    assert torch.equal(via_mixed.receiver_amplitudes, via_output.receiver_amplitudes)


def test_initial_state_invalid_type_raises() -> None:
    """initial_state must be a PropagatorOutput or a mapping."""
    case = _initial_state_case("scalar", 2)
    run, amp, split = case["run"], case["full_amp"], case["split"]
    with pytest.raises(TypeError, match="PropagatorOutput or a mapping"):
        run(amp[..., split:], initial_state=[1, 2, 3])


@pytest.mark.parametrize("name", _PROP_NAMES)
def test_initial_state_module_forms_pass_through(name: str) -> None:
    """The nn.Module forms forward initial_state to the functional form.

    Each Module.forward passes ``initial_state`` straight through, so a chained
    two-segment run through the Module must reproduce a monolithic Module run
    (at step_ratio == 1), exercising the passthrough wiring in all four modules.
    """
    torch.manual_seed(7)
    shape = (16, 17)
    dx, dt, nt, split = 5.0, 0.0005, 12, 7
    amp = torch.randn(1, 1, nt, dtype=torch.float64)
    centre = [s // 2 for s in shape]
    src_loc = torch.tensor([[centre]])
    rec_loc = torch.tensor([[centre, [c + 1 for c in centre]]])
    v = 1500 * torch.ones(*shape, dtype=torch.float64)
    common = {
        "max_vel": 2000.0,
        "pml_freq": 15.0,
        "origin": [0, 0],
    }

    if name == "scalar":
        module = deepwave.Scalar(v, dx)
        src_kw = {"source_locations": src_loc, "receiver_locations": rec_loc}
        rec = "receiver_amplitudes"
    elif name == "scalar_born":
        module = deepwave.ScalarBorn(v, dx, dv=torch.randn(*shape, dtype=torch.float64))
        src_kw = {"source_locations": src_loc, "sc_receiver_locations": rec_loc}
        rec = "sc_receiver_amplitudes"
    elif name == "acoustic":
        rho = 2200 * torch.ones(*shape, dtype=torch.float64)
        module = deepwave.Acoustic(v, rho, dx)
        src_kw = {
            "source_locations_volume_rate": src_loc,
            "receiver_locations_p": rec_loc,
        }
        rec = "receiver_amplitudes_p"
    else:
        lamb = 2200 * 1500.0**2 * torch.ones(*shape, dtype=torch.float64)
        mu = 2200 * 900.0**2 * torch.ones(*shape, dtype=torch.float64)
        buoyancy = (1 / 2200) * torch.ones(*shape, dtype=torch.float64)
        module = deepwave.Elastic(lamb, mu, buoyancy, dx)
        amp = amp * 1e6
        src_kw = {"source_locations_x": src_loc, "receiver_locations_x": rec_loc}
        rec = "receiver_amplitudes_x"

    amp_key = (
        "source_amplitudes_volume_rate"
        if name == "acoustic"
        else ("source_amplitudes_x" if name == "elastic" else "source_amplitudes")
    )
    full = module(dt, **{amp_key: amp}, **src_kw, **common)
    seg1 = module(dt, **{amp_key: amp[..., :split]}, **src_kw, **common)
    seg2 = module(
        dt, **{amp_key: amp[..., split:]}, **src_kw, initial_state=seg1, **common
    )
    chained = torch.cat([getattr(seg1, rec), getattr(seg2, rec)], dim=-1)
    assert torch.equal(getattr(full, rec), chained)


def test_initial_state_preserves_gradients() -> None:
    """Autograd flows through initial_state, linking chained segments."""
    case = _initial_state_case("scalar", 2)
    run, amp, split = case["run"], case["full_amp"], case["split"]
    amp1 = amp[..., :split].clone().requires_grad_(True)
    seg1 = run(amp1)
    seg2 = run(amp[..., split:], initial_state=seg1)
    # The second segment's receivers depend on the first segment's source
    # through the carried-over wavefields, so a gradient must reach amp1.
    seg2.receiver_amplitudes.sum().backward()
    assert amp1.grad is not None
    assert torch.any(amp1.grad != 0)
