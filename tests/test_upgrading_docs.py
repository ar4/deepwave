"""``docs/upgrading.rst``'s quoted messages are the messages Deepwave emits.

The migration guide is only useful if a reader who pastes an error into a
search box lands on the entry that explains it. That is a property of two
things staying in step -- a page and a hundred message strings scattered
through ``src/`` -- and nothing but a test keeps them together. The
"Migrating from v0.0.27" section of the changelog drifted from the code
twice before this file existed (it described a ``ValueError`` the default
path does not raise, and quoted a tolerance from a helper the propagators
never reach); both were found by reading, not by CI.

How it works. Each entry in the page carries a machine-readable marker::

    .. upgrade-break: keyword_only_arguments

    .. code-block:: text

       TypeError: scalar() takes 3 positional arguments but 6 were given

The marker is an RST *comment*, so it renders as nothing and cannot affect
the page. This module holds a trigger function per marker id. The trigger
reproduces the condition on a tiny grid and the test asserts that the quoted
block appears in what came out, formatted as ``ExceptionName: message`` (or
``WarningCategory: message``).

Two deliberate loosenings, both so that the page can stay readable:

* **Whitespace is normalised** on both sides, so the ``.rst`` may wrap a long
  message across lines.
* **The comparison is a substring test**, so an entry may quote the
  distinctive opening of a message that runs to eight lines, or one that ends
  in a machine-specific path or byte count, rather than the whole of it.
  Every quote is still verbatim as far as it goes.

The set-equality assertions are the part that stops the file rotting: a
marker with no trigger, or a trigger with no marker, fails here.
"""

import pathlib
import re
import warnings
from collections.abc import Callable, Iterator
from typing import Any, Optional

import pytest
import torch
from helpers import c_backend_available

import deepwave
import deepwave._common
import deepwave._engine
from deepwave.wavelets import ricker

_UPGRADING = pathlib.Path(__file__).resolve().parents[1] / "docs" / "upgrading.rst"

#: ``.. upgrade-break: <id>``, an RST comment (no ``::``, so not a directive).
_MARKER = re.compile(r"^\.\.\s+upgrade-break:\s+(\w+)\s*$")

#: The opening of a literal block holding a quoted message.
_CODE_BLOCK = re.compile(r"^(\s*)\.\.\s+code-block::\s+text\s*$")


def _normalise(text: str) -> str:
    """Normalise wrapping and Python's constructor-name qualification.

    Python 3.9 reports ``__init__()`` where newer interpreters report
    ``ClassName.__init__()``. Keep the exception type, parameter names and
    argument counts intact: those are the useful migration instructions.
    """
    text = re.sub(r"\b[A-Za-z_]\w*\.__init__\b", "__init__", text)
    return " ".join(text.split())


def _parse_page() -> dict[str, str]:
    """Maps each marker id to the ``text`` block that follows it.

    Strict by design: a marker whose block cannot be found is an error, not
    a skip. A silently skipped row is exactly the failure mode this file
    exists to prevent.
    """
    lines = _UPGRADING.read_text(encoding="utf-8").splitlines()
    quoted: dict[str, str] = {}
    for index, line in enumerate(lines):
        match = _MARKER.match(line)
        if match is None:
            continue
        name = match.group(1)
        assert name not in quoted, f"duplicate upgrade-break marker {name!r}"
        quoted[name] = _read_block_after(lines, index, name)
    return quoted


def _read_block_after(lines: list[str], index: int, name: str) -> str:
    """The body of the first ``code-block:: text`` after ``index``."""
    for cursor in range(index + 1, len(lines)):
        opening = _CODE_BLOCK.match(lines[cursor])
        if opening is None:
            if _MARKER.match(lines[cursor]):
                raise AssertionError(
                    f"upgrade-break {name!r} has no 'code-block:: text' before "
                    "the next marker"
                )
            continue
        return _block_body(lines, cursor, len(opening.group(1)))
    raise AssertionError(f"upgrade-break {name!r} has no 'code-block:: text'")


def _block_body(lines: list[str], start: int, base_indent: int) -> str:
    """The indented body of the literal block opened at ``start``."""
    body: list[str] = []
    for line in lines[start + 1 :]:
        if not line.strip():
            if body:
                body.append("")
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base_indent:
            break
        body.append(line.strip())
    text = " ".join(part for part in body if part)
    assert text, "empty code-block after an upgrade-break marker"
    return text


# --------------------------------------------------------------------------
# A tiny, fast fixture set. Ten cells, eight steps, float64 CPU -- enough to
# reach every validation path in the propagator front end without the cost of
# a simulation, so this file stays in the default (non-slow) gate.
# --------------------------------------------------------------------------

_NY = _NX = 12
_NT = 8
_DT = 0.004
_DX = 4.0


def _v() -> torch.Tensor:
    return torch.full((_NY, _NX), 1500.0, dtype=torch.float64)


def _source() -> torch.Tensor:
    return ricker(25.0, _NT, _DT, 0.02, dtype=torch.float64).reshape(1, 1, -1)


def _source_locations() -> torch.Tensor:
    return torch.tensor([[[5, 5]]])


def _receiver_locations() -> torch.Tensor:
    return torch.tensor([[[7, 7]]])


def _scalar(**kwargs: Any) -> Any:
    """A minimal, valid ``scalar`` call, with ``kwargs`` layered on top."""
    call: dict[str, Any] = {
        "source_amplitudes": _source(),
        "source_locations": _source_locations(),
        "receiver_locations": _receiver_locations(),
        "pml_freq": 25.0,
    }
    call.update(kwargs)
    return deepwave.scalar(_v(), _DX, _DT, **call)


# --------------------------------------------------------------------------
# Triggers. One per marker id in docs/upgrading.rst.
# --------------------------------------------------------------------------


def _trigger_keyword_only_arguments() -> None:
    deepwave.scalar(  # type: ignore[misc]
        _v(), _DX, _DT, _source(), _source_locations(), _receiver_locations()
    )


def _trigger_module_constructor_keyword_only() -> None:
    deepwave.Scalar(_v(), _DX, True)  # type: ignore[misc]


def _trigger_tuple_index_out_of_range() -> None:
    assert _scalar()[4] is not None


def _trigger_not_enough_values_to_unpack() -> None:
    # The v0.0.27 2-D ``scalar`` unpacking, which now finds four values.
    _, _, _, _, _, _, _ = _scalar()


def _trigger_auxiliary_state_attribute() -> None:
    assert _scalar().psiy_m1 is not None


def _trigger_auxiliary_state_input_keyword() -> None:
    # The ``v0.0.27`` continuation form: the auxiliary fields handed back in
    # as keyword arguments. The parameters no longer exist.
    _scalar(psiy_m1=torch.zeros(1, _NY, _NX, dtype=torch.float64))


def _trigger_auxiliary_state_initial_state_key() -> None:
    out = _scalar()
    _scalar(initial_state={"psiy_m1": out.auxiliary_state["psiy_m1"]})


def _trigger_model_as_grid_spacing() -> None:
    deepwave.scalar_born(
        _v(),
        torch.zeros(_NY, _NX, dtype=torch.float64),  # type: ignore[arg-type]
        dt=_DT,
        source_amplitudes=_source(),
        source_locations=_source_locations(),
        sc_receiver_locations=_receiver_locations(),
        pml_freq=25.0,
    )


def _trigger_pml_freq_inference() -> None:
    deepwave.scalar(
        _v(),
        _DX,
        _DT,
        source_amplitudes=ricker(2.0, 400, _DT, 0.5, dtype=torch.float64).reshape(
            1, 1, -1
        ),
        source_locations=_source_locations(),
        receiver_locations=_receiver_locations(),
    )


def _trigger_born_edge_scatter() -> None:
    dv = torch.zeros(_NY, _NX, dtype=torch.float64)
    dv[0] = 1.0
    deepwave.scalar_born(
        _v(),
        _DX,
        _DT,
        dv=dv,
        source_amplitudes=_source(),
        source_locations=_source_locations(),
        sc_receiver_locations=_receiver_locations(),
        pml_freq=25.0,
    )


def _trigger_python_backend() -> None:
    _scalar(python_backend=False)


def _trigger_scatter_keyword() -> None:
    deepwave.scalar_born(
        _v(),
        _DX,
        _DT,
        scatter=torch.zeros(_NY, _NX, dtype=torch.float64),
        source_amplitudes=_source(),
        source_locations=_source_locations(),
        sc_receiver_locations=_receiver_locations(),
        pml_freq=25.0,
    )


def _trigger_scatter_state_dict() -> None:
    module = deepwave.ScalarBorn(
        _v(), _DX, dv=torch.zeros(_NY, _NX, dtype=torch.float64)
    )
    module.load_state_dict(
        {"v": _v(), "scatter": torch.zeros(_NY, _NX, dtype=torch.float64)}
    )


def _trigger_storage_path_default() -> None:
    deepwave.scalar(
        _v().requires_grad_(True),
        _DX,
        _DT,
        source_amplitudes=_source(),
        source_locations=_source_locations(),
        receiver_locations=_receiver_locations(),
        pml_freq=25.0,
        storage_mode="disk",
    )


def _trigger_storage_mode_none_gradient() -> None:
    deepwave.scalar(
        _v().requires_grad_(True),
        _DX,
        _DT,
        source_amplitudes=_source(),
        source_locations=_source_locations(),
        receiver_locations=_receiver_locations(),
        pml_freq=25.0,
        storage_mode="none",
    )


def _trigger_mgsi_divides_nt() -> None:
    deepwave.scalar(
        _v().requires_grad_(True),
        _DX,
        _DT,
        source_amplitudes=_source(),
        source_locations=_source_locations(),
        receiver_locations=_receiver_locations(),
        pml_freq=25.0,
        model_gradient_sampling_interval=3,
    )


def _trigger_non_finite_model() -> None:
    deepwave.scalar(
        torch.full((_NY, _NX), float("nan"), dtype=torch.float64),
        _DX,
        _DT,
        source_amplitudes=_source(),
        source_locations=_source_locations(),
        receiver_locations=_receiver_locations(),
        pml_freq=25.0,
    )


def _trigger_non_finite_output() -> None:
    """A valid material and time step whose huge finite source overflows float32."""
    deepwave.scalar(
        torch.full((_NY, _NX), 1500.0),
        _DX,
        0.001,
        source_amplitudes=torch.full((1, 1, 60), 1e36),
        source_locations=_source_locations(),
        receiver_locations=_receiver_locations(),
        pml_freq=25.0,
    )


def _trigger_list_model() -> None:
    deepwave.scalar(
        [[1500.0] * _NX] * _NY,  # type: ignore[arg-type]
        _DX,
        _DT,
        source_amplitudes=_source(),
        source_locations=_source_locations(),
        receiver_locations=_receiver_locations(),
        pml_freq=25.0,
    )


def _trigger_empty_model() -> None:
    deepwave.scalar(
        torch.zeros(0, 0, dtype=torch.float64), _DX, _DT, nt=3, pml_freq=25.0
    )


def _trigger_model_dtype() -> None:
    deepwave.scalar(
        torch.full((_NY, _NX), 1500, dtype=torch.int64),
        _DX,
        _DT,
        source_amplitudes=_source(),
        source_locations=_source_locations(),
        receiver_locations=_receiver_locations(),
        pml_freq=25.0,
    )


def _trigger_meaningless_option() -> None:
    deepwave.viscoacoustic(
        _v(),
        torch.full((_NY, _NX), 1000.0, dtype=torch.float64),
        torch.full((_NY, _NX), 50.0, dtype=torch.float64),
        _DX,
        _DT,
        source_amplitudes_pressure_rate=_source(),
        source_locations_pressure_rate=_source_locations(),
        receiver_locations_p=_receiver_locations(),
        pml_freq=25.0,
        mpml_fraction=0.1,
    )


def _trigger_location_interpolation_shim() -> None:
    assert deepwave.location_interpolation.Hicks is not None


def _trigger_callback_state_required_arguments() -> None:
    deepwave.CallbackState(  # type: ignore[call-arg]
        0.004,
        0,
        {},
        {},
        {},
        [0, 0],
        [0, 0, 0, 0],
    )


def _trigger_cfl_condition_n_signature() -> None:
    deepwave.discretisation.cfl_condition_n(
        [_DX, _DX],
        _DT,
        max_abs_vel=1500.0,  # type: ignore[call-arg]
    )


def _trigger_common_shim() -> None:
    assert deepwave.common.cfl_condition is not None


def _trigger_backend_utils_shim() -> None:
    assert deepwave.backend_utils.dll is not None


def _trigger_regular_grid_shim() -> None:
    assert deepwave.regular_grid.set_pml_profiles is not None


def _trigger_staggered_grid_shim() -> None:
    assert deepwave.staggered_grid.diff1 is not None


def _trigger_hicks_keyword_only() -> None:
    deepwave.Hicks(_source_locations()[0], 4, True)  # type: ignore[misc]


def _trigger_ricker_required_arguments() -> None:
    ricker(25.0)  # type: ignore[call-arg]


def _trigger_ricker_length() -> None:
    ricker(25.0, dt=_DT, length=_NT)  # type: ignore[call-arg]


def _trigger_off_node_none() -> None:
    deepwave.scalar(
        _v(),
        _DX,
        _DT,
        source_amplitudes=_source(),
        source_locations=torch.tensor([[[5.05, 5.0]]], dtype=torch.float64),
        receiver_locations=_receiver_locations(),
        pml_freq=25.0,
        location_interpolation="none",
    )


def _trigger_truncated_coordinate() -> None:
    """A coordinate ``v0.0.27`` accepted and truncated to the node below."""
    _scalar(source_locations=torch.tensor([[[5.05, 5.0]]], dtype=torch.float64))


def _trigger_hicks_stencil() -> None:
    """A coordinate whose window cells differ from ``v0.0.27``'s arithmetic."""
    deepwave.Hicks(torch.tensor([[[5.6, 7.0]]]))


def _trigger_hicks_default_dtype() -> None:
    """``Hicks`` with no ``dtype=`` under a non-``float32`` default dtype."""
    torch.set_default_dtype(torch.float64)
    try:
        deepwave.Hicks(torch.tensor([[[5.5, 7.0]]], dtype=torch.float64))
    finally:
        torch.set_default_dtype(torch.float32)


def _trigger_step_ratio_warning() -> None:
    deepwave.scalar(
        _v(),
        _DX,
        0.02,
        source_amplitudes=ricker(25.0, _NT, 0.02, 0.04, dtype=torch.float64).reshape(
            1, 1, -1
        ),
        source_locations=_source_locations(),
        receiver_locations=_receiver_locations(),
        pml_freq=25.0,
    )


def _trigger_cuda_non_default_stream() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if not c_backend_available(
        "scalar", accuracy=4, dtype=torch.float32, device=torch.device("cuda")
    ):
        pytest.skip("installed Deepwave library has no scalar CUDA kernel")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        deepwave.scalar(
            _v().float().cuda(),
            _DX,
            _DT,
            source_amplitudes=_source().float().cuda(),
            source_locations=_source_locations().cuda(),
            receiver_locations=_receiver_locations().cuda(),
            pml_freq=25.0,
        )


def _trigger_max_vel_batched_model() -> None:
    torch.func.vmap(
        lambda model: (
            deepwave.scalar(
                model,
                _DX,
                _DT,
                source_amplitudes=_source(),
                source_locations=_source_locations(),
                receiver_locations=_receiver_locations(),
                pml_freq=25.0,
                backend="eager",
            ).receiver_amplitudes
        )
    )(_v().unsqueeze(0))


#: Marker id -> the callable that reproduces it. Kept in set-equality with
#: the page's markers by ``test_every_marker_has_a_trigger``.
_TRIGGERS: dict[str, Callable[[], None]] = {
    "backend_utils_shim": _trigger_backend_utils_shim,
    "auxiliary_state_attribute": _trigger_auxiliary_state_attribute,
    "auxiliary_state_initial_state_key": _trigger_auxiliary_state_initial_state_key,
    "auxiliary_state_input_keyword": _trigger_auxiliary_state_input_keyword,
    "callback_state_required_arguments": _trigger_callback_state_required_arguments,
    "cfl_condition_n_signature": _trigger_cfl_condition_n_signature,
    "common_shim": _trigger_common_shim,
    "cuda_non_default_stream": _trigger_cuda_non_default_stream,
    "empty_model": _trigger_empty_model,
    "hicks_default_dtype": _trigger_hicks_default_dtype,
    "hicks_keyword_only": _trigger_hicks_keyword_only,
    "hicks_stencil": _trigger_hicks_stencil,
    "keyword_only_arguments": _trigger_keyword_only_arguments,
    "list_model": _trigger_list_model,
    "location_interpolation_shim": _trigger_location_interpolation_shim,
    "max_vel_batched_model": _trigger_max_vel_batched_model,
    "meaningless_option": _trigger_meaningless_option,
    "mgsi_divides_nt": _trigger_mgsi_divides_nt,
    "model_as_grid_spacing": _trigger_model_as_grid_spacing,
    "model_dtype": _trigger_model_dtype,
    "module_constructor_keyword_only": _trigger_module_constructor_keyword_only,
    "non_finite_model": _trigger_non_finite_model,
    "non_finite_output": _trigger_non_finite_output,
    "not_enough_values_to_unpack": _trigger_not_enough_values_to_unpack,
    "off_node_none": _trigger_off_node_none,
    "born_edge_scatter": _trigger_born_edge_scatter,
    "pml_freq_inference": _trigger_pml_freq_inference,
    "python_backend": _trigger_python_backend,
    "regular_grid_shim": _trigger_regular_grid_shim,
    "ricker_length": _trigger_ricker_length,
    "ricker_required_arguments": _trigger_ricker_required_arguments,
    "scatter_keyword": _trigger_scatter_keyword,
    "scatter_state_dict": _trigger_scatter_state_dict,
    "staggered_grid_shim": _trigger_staggered_grid_shim,
    "step_ratio_warning": _trigger_step_ratio_warning,
    "storage_mode_none_gradient": _trigger_storage_mode_none_gradient,
    "storage_path_default": _trigger_storage_path_default,
    "truncated_coordinate": _trigger_truncated_coordinate,
    "tuple_index_out_of_range": _trigger_tuple_index_out_of_range,
}


def _messages(trigger: Callable[..., None], *args: Any) -> Iterator[str]:
    """Every message a trigger produces, as ``TypeName: text``.

    Both raises and warnings, in one vocabulary, so that the page quotes one
    kind of thing.

    Several of the messages quoted on the page are emitted at most once per
    process, keyed on a cause string. Whether one of them has already been
    spent depends on which other tests ran first in this worker, so the
    registries are cleared here rather than left to chance -- otherwise this
    file would pass alone and fail in a full run, which is the worst of both.
    They are **restored** afterwards rather than left cleared: leaving them
    empty would re-arm every once-per-process warning for whatever runs next
    in this worker, which could make a later test see a warning it asserts is
    absent. Clearing without restoring turns one flaky file into two.
    """
    saved_common = set(deepwave._common._warned_once_causes)
    saved_engine = set(deepwave._engine._warned_once_causes)
    deepwave._common._warned_once_causes.clear()
    deepwave._engine._warned_once_causes.clear()
    try:
        yield from _collect(trigger, *args)
    finally:
        deepwave._common._warned_once_causes.clear()
        deepwave._common._warned_once_causes.update(saved_common)
        deepwave._engine._warned_once_causes.clear()
        deepwave._engine._warned_once_causes.update(saved_engine)


def _collect(trigger: Callable[..., None], *args: Any) -> Iterator[str]:
    """Runs ``trigger`` and yields every warning and raise it produced."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raised: Optional[BaseException] = None
        try:
            trigger(*args)
        except pytest.skip.Exception:
            raise
        except BaseException as exc:  # noqa: BLE001
            raised = exc
    for entry in caught:
        yield f"{entry.category.__name__}: {entry.message}"
    if raised is not None:
        yield f"{type(raised).__name__}: {raised}"


def test_the_page_parses() -> None:
    """The parser finds markers and a quoted block for each."""
    quoted = _parse_page()
    assert quoted, "no upgrade-break markers found in docs/upgrading.rst"
    for name, text in quoted.items():
        assert ":" in text, f"{name}: quoted block is not 'TypeName: message'"


def test_every_marker_has_a_trigger() -> None:
    """Set equality both ways: no orphan marker, no orphan trigger."""
    quoted = set(_parse_page())
    registered = set(_TRIGGERS)
    assert quoted - registered == set(), (
        f"docs/upgrading.rst markers with no trigger: {sorted(quoted - registered)}"
    )
    assert registered - quoted == set(), (
        f"triggers with no marker in docs/upgrading.rst: {sorted(registered - quoted)}"
    )


@pytest.mark.parametrize("name", sorted(_TRIGGERS))
def test_quoted_message_is_the_real_message(
    name: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each quoted block really is (part of) what Deepwave emits."""
    if name == "storage_path_default":
        # The message names the resolved directory and its free space, so
        # the run is pointed at a directory this test owns. Only the fixed
        # prefix is quoted on the page.
        monkeypatch.setenv("TMPDIR", str(tmp_path))
    expected = _normalise(_parse_page()[name])
    produced = [_normalise(message) for message in _messages(_TRIGGERS[name])]
    assert any(expected in message for message in produced), (
        f"{name}: docs/upgrading.rst quotes\n  {expected}\nbut the code "
        f"produced\n  " + "\n  ".join(produced or ["(nothing)"])
    )


def test_no_provenance_language_on_the_page() -> None:
    """The page is user-facing text, not a record of how it was written."""
    text = _UPGRADING.read_text(encoding="utf-8").lower()
    for word in (
        "work package",
        "campaign",
        "todo(",
    ):
        assert word not in text, f"{word!r} appears in docs/upgrading.rst"


def test_every_deprecated_name_in_the_table_still_resolves() -> None:
    """The renamed/deprecated table's shim modules all still work.

    The table promises that nothing ``v0.0.27`` shipped has been removed,
    save the auxiliary-field input keywords its last row records (covered by
    ``auxiliary_state_input_keyword`` above). These are the three
    whole-module promises; the per-argument rows are covered by
    ``tests/test_deprecation_removal_versions.py``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", deepwave.DeepwaveFutureWarning)
        assert deepwave.location_interpolation.Hicks is deepwave.Hicks
        assert deepwave.location_interpolation.DEFAULT_EPS == deepwave.ON_NODE_TOLERANCE
        assert deepwave.common.cfl_condition is deepwave.discretisation.cfl_condition
        assert deepwave.backend_utils.SO_EXT is not None
        assert deepwave.regular_grid.diff1 is not None
        assert deepwave.staggered_grid.diff1 is not None
