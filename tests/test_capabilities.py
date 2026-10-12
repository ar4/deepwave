"""Conformance checks for ``docs/capabilities.rst``'s declared table.

``deepwave._capabilities`` publishes declared support and checks it with
runtime probes. The tests below fall into two groups:

* Structural sanity: declarations cover all nine propagators and the
  rendered table names every one of them. The probes must match the promises.
* Independent re-derivation of the trickiest, most rot-prone facts (the
  best-available-backend fallback, the ``role="forward"`` nuance, and the
  ``mpml_fraction`` and ``surface_method`` placements): each of these
  tests builds its own minimal problem (reusing the vetted grid/model/survey
  fixtures from ``test_adjoint_dot_product.py`` rather than
  ``deepwave._capabilities``'s own helpers) and calls the propagators
  directly, then cross-checks the result against
  the declarations' corresponding field. A bug that made
  ``deepwave._capabilities``'s own probing silently wrong (not just a stale
  hand-written claim) would still be caught here, because these tests never
  call into ``deepwave._capabilities``'s internal ``_call``/``_receiver``/
  ``_models`` machinery -- only its public declaration.
"""

import pathlib
import re
import tempfile
import warnings

import pytest
import torch
from test_adjoint_dot_product import _grid, _media, _survey
from torch.func import jvp, vmap

import deepwave
from deepwave._capabilities import (
    PROPAGATOR_NAMES,
    _render_storage_modes,
    capabilities_rst,
    declared_capabilities,
    probe_capabilities,
)
from deepwave._engine import BACKEND_VALUES

DTYPE = torch.float64

# Independent per-propagator (primary model, remaining positional args,
# source/receiver keyword prefix) mapping -- deliberately re-derived here
# rather than imported from deepwave._capabilities, so a bug in that
# module's own ``_models``/``_base_kwargs`` cannot hide from these tests.
#: The channel suffix these propagators' survey keywords carry, where it is
#: not derived from the spec below. The two scalar propagators have no
#: suffix at all, and ``acoustic_vti``'s scalar channels are synthesised, so
#: its x velocity is used on both sides instead.
_SRC_PREFIX = {
    "scalar": "",
    "scalar_born": "",
    "acoustic_vti": "_x",
}


def _positional(prop: str, m: dict) -> tuple:
    """The full positional-argument tuple (models..., grid_spacing) for ``prop``."""
    if prop in ("scalar", "scalar_born"):
        return (m["v"], 5.0)
    if prop in ("acoustic", "acoustic_born"):
        return (m["v"], m["rho"], 5.0)
    if prop == "viscoacoustic":
        return (m["v"], m["rho"], m["q"], 5.0)
    if prop in ("elastic", "elastic_born"):
        return (m["lamb"], m["mu"], m["buoyancy"], 5.0)
    if prop == "acoustic_vti":
        return (m["c11"], m["c13"], m["c33"], m["buoyancy"], 5.0)
    if prop in ("elastic_vti",):
        return (m["c11"], m["c13"], m["c33"], m["c55"], m["buoyancy"], 5.0)
    raise AssertionError(prop)


def _survey_and_backend_kwargs(prop: str, ndim: int, backend) -> dict:
    """A minimal valid, propagator-specific ``kwargs`` dict (own fixture)."""
    g = _grid(ndim)
    n, nt, dt = g["n"], g["nt"], g["dt"]
    src, src_loc, rec_loc = _survey(ndim, n, nt, dt)
    # A source is named for the quantity it injects and a receiver for the
    # field it records, so the two sides do not share a suffix.
    spec = deepwave.spec(prop)
    prefix = _SRC_PREFIX.get(prop, f"_{spec.source_channels(ndim)[0]}")
    receiver_prefix = _SRC_PREFIX.get(prop, f"_{spec.receiver_channels(ndim)[0]}")
    kwargs: dict = {
        "dt": dt,
        "pml_freq": 25.0,
        "accuracy": g["acc"],
        "pml_width": 4,
        # Explicit (rather than data-dependent) so vmap/jvp-batched model
        # tensors are usable -- see deepwave._common.velocity_extremes.
        "max_vel": 4000.0,
        "backend": backend,
        f"source_amplitudes{prefix}": src,
        f"source_locations{prefix}": src_loc,
    }
    # The Born propagators spell their scattered receivers ``sc_``, so
    # the keyword is resolved from the spec rather than assumed.
    sc_prefix = "sc_" if spec.has_bg_receivers else ""
    kwargs[f"{sc_prefix}receiver_locations{receiver_prefix}"] = rec_loc
    if prop in ("viscoacoustic",):
        kwargs["q_freq"] = 25.0
    zeros = torch.zeros((n,) * ndim, dtype=DTYPE)
    if prop == "scalar_born":
        kwargs["dv"] = zeros
        kwargs["bg_receiver_locations"] = rec_loc
    elif prop == "acoustic_born":
        kwargs["dv"] = zeros
        kwargs["drho"] = zeros
        kwargs["bg_receiver_locations_p"] = rec_loc
    elif prop == "elastic_born":
        kwargs["dlamb"] = zeros
        kwargs["dmu"] = zeros
        kwargs["dbuoyancy"] = zeros
        kwargs["bg_receiver_locations_p"] = rec_loc
    return kwargs


def _minimal_call(prop: str, ndim: int = 2, backend="auto", **overrides):
    """Calls ``prop`` on an independently-built minimal problem."""
    g = _grid(ndim)
    m = _media(ndim, g["n"])
    fn = getattr(deepwave, prop)
    kwargs = _survey_and_backend_kwargs(prop, ndim, backend)
    kwargs.update(overrides)
    return fn(*_positional(prop, m), **kwargs)


_ROWS = {row.name: row for row in declared_capabilities()}


def test_runtime_probes_match_declared_capabilities() -> None:
    """A runtime failure cannot silently shrink the published support table."""
    observed = {row.name: row for row in probe_capabilities()}
    assert observed == _ROWS


def test_promised_runtime_failure_does_not_change_documented_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken kernel remains a failed promise rather than a smaller table."""
    import deepwave._capabilities as capabilities

    def broken_receiver(*args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("injected backend failure")

    monkeypatch.setattr(capabilities, "_receiver", broken_receiver)
    assert capabilities._probe_dims("scalar") == ()
    assert declared_capabilities(("scalar",))[0].dims == (1, 2, 3)
    assert "1-D, 2-D, 3-D" in capabilities_rst(declared_capabilities(("scalar",)))


def test_nonfinite_promised_result_fails_probe_without_shrinking_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NaN-returning kernel is a failed conformance case, not support data."""
    import deepwave._capabilities as capabilities

    def broken_receiver(*args: object, **kwargs: object) -> torch.Tensor:
        return torch.full((1,), float("nan"))

    monkeypatch.setattr(capabilities, "_receiver", broken_receiver)
    assert capabilities._probe_dims("scalar") == ()
    assert declared_capabilities(("scalar",))[0].dims == (1, 2, 3)


def test_probe_capabilities_covers_every_propagator() -> None:
    """Every one of the nine retained propagators gets a row."""
    assert tuple(_ROWS) == PROPAGATOR_NAMES
    assert len(PROPAGATOR_NAMES) == 9


def test_capabilities_rst_mentions_every_propagator_and_table() -> None:
    """The rendered RST names every propagator and all three tables."""
    rst = capabilities_rst(list(_ROWS.values()))
    for name in PROPAGATOR_NAMES:
        assert f"``{name}``" in rst
    for title in (
        "Scope and backends",
        "Physics options and differentiation",
        "Callbacks",
    ):
        assert title in rst
    assert rst.count(".. list-table::") == 3


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_rendered_storage_modes_name_only_real_backends(name: str) -> None:
    """Every backend spelling in a Storage-modes cell is a valid ``backend=``.

    The cell is written as ``backend spellings: modes`` segments separated by
    semicolons, so the double-backtick literals to the left of each colon are
    the ``backend=`` values the reader is invited to pass. A stale spelling
    (for example the removed ``compile``) must not be offered.
    """
    cell = _render_storage_modes(_ROWS[name])
    spellings = []
    for segment in cell.split(";"):
        head = segment.split(":", 1)[0]
        spellings.extend(re.findall(r"``(\w+)``", head))
    assert spellings, cell
    assert "eager" in spellings, cell
    for spelling in spellings:
        assert spelling in BACKEND_VALUES, (spelling, cell)
    # The same text must reach the generated table.
    assert cell in capabilities_rst([_ROWS[name]])


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_mpml_p_placement_matches_runtime(name: str) -> None:
    """A propagator accepts ``mpml_fraction`` as a keyword iff its row says "yes".

    Every propagator whose row is "no" must reject ``mpml_fraction`` with a
    ``TypeError`` (unexpected keyword argument) on every backend -- not
    silently ignore it, which would be a worse
    bug than simply lacking the option. Every propagator whose row starts
    with "yes" must accept the keyword (no ``TypeError``) on at least one
    backend; ``acoustic_born``/``elastic_born`` accept it everywhere but
    only actually apply it (no exception at all) on the backend(s) named --
    a real backend-availability gap this test discovered independently of
    ``deepwave._capabilities``'s own probing (see that module's
    ``_probe_mpml_fraction`` docstring).
    """
    claim = _ROWS[name].mpml_fraction
    if claim == "no":
        for backend in ("auto", "eager"):
            with pytest.raises(TypeError, match="mpml_fraction"):
                _minimal_call(name, backend=backend, mpml_fraction=0.1)
        return
    assert claim.startswith("yes"), (
        f"unexpected mpml_fraction claim for {name}: {claim!r}"
    )
    for backend in ("auto", "eager"):
        try:
            _minimal_call(name, backend=backend, mpml_fraction=0.1)
            worked = True
        except TypeError as exc:
            pytest.fail(
                f"{name} claims mpml_fraction support ({claim}) but backend="
                f"{backend!r} rejected the keyword entirely: {exc}"
            )
        except Exception:  # noqa: BLE001 - see the per-backend check below
            worked = False
        backend_label = "default" if backend == "auto" else "eager"
        if claim == "yes (all backends)" or f"{backend_label} only" in claim:
            assert worked, f"{name}: {backend_label} should accept mpml_fraction > 0"
        else:
            assert not worked, (
                f"{name}: {backend_label} should NOT accept "
                f"mpml_fraction > 0 (claim: {claim})"
            )


#: The shared options, with a default that must be accepted everywhere and a
#: non-default that must raise wherever the propagator cannot use it.
_SHARED_OPTIONS = (
    ("mpml_fraction", 0.0, 0.15),
    ("surface_method", "ivm", "pm"),
    ("q_freq", None, 25.0),
    ("q_n_mechanisms", 1, 2),
    ("q_band", None, (5.0, 50.0)),
)


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_physics_options_match_the_spec(name: str) -> None:
    """Only implemented physics settings are declared in the signature."""
    import inspect

    supported = deepwave.spec(name).supported_options
    params = inspect.signature(getattr(deepwave, name)).parameters
    for option, default, _ in _SHARED_OPTIONS:
        assert (option in params) == (option in supported)
        if option in supported:
            assert params[option].default == default
            if option != "q_freq":
                _minimal_call(name, **{option: default})
        else:
            with pytest.raises(TypeError, match=option):
                _minimal_call(name, **{option: default})


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_surface_method_placement_matches_runtime(name: str) -> None:
    """``surface_method``'s row is "no" iff a non-default value raises."""
    claim = _ROWS[name].surface_method
    supported = "surface_method" in deepwave.spec(name).supported_options
    assert (claim != "no") == supported, (name, claim)
    if not supported:
        with pytest.raises(TypeError, match="surface_method"):
            _minimal_call(name, surface_method="pm")


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_q_options_placement_matches_runtime(name: str) -> None:
    """The constant-Q row is "no" iff a non-default ``q_freq`` raises."""
    claim = _ROWS[name].q_options
    supported = "q_freq" in deepwave.spec(name).supported_options
    assert (claim != "no") == supported, (name, claim)
    if supported:
        # The two attenuating propagators require ``q_freq``: it is the
        # frequency their Q was specified at, and has no default.
        _minimal_call(name, q_freq=25.0, q_n_mechanisms=1, q_band=None)
        with pytest.raises(deepwave.DeepwaveValueError, match="q_freq is required"):
            _minimal_call(name, q_freq=None)
    else:
        with pytest.raises(TypeError, match="q_freq"):
            _minimal_call(name, q_freq=25.0)


def test_role_forward_is_scalar_only_independently() -> None:
    """Only ``scalar`` supports ``get_wavefield(role="forward")``.

    Runs a real ``backward_callback`` on ``scalar``, ``acoustic``, and
    ``elastic`` (its own minimal, independently-built problem) and checks
    that only ``scalar``'s callback can call ``role="forward"`` without a
    ``NotImplementedError``, cross-checked against ``probe_capabilities()``'s
    ``role_forward`` field for all nine propagators (not just these three).
    """
    results = {}
    for name in ("scalar", "acoustic", "elastic"):
        outcome = {}

        def cb(state, outcome=outcome) -> None:
            try:
                state.get_wavefield("__test_probe__", role="forward")
                outcome["role_forward"] = True
            except KeyError:
                outcome["role_forward"] = True
            except NotImplementedError:
                outcome["role_forward"] = False

        m = _media(2, _grid(2)["n"])
        primary = _positional(name, m)[0].clone().requires_grad_(True)
        rest = _positional(name, m)[1:]
        kwargs = _survey_and_backend_kwargs(name, 2, "compiled")
        kwargs["backward_callback"] = cb
        # Retaining the forward wavefield is opt-in and the argument that
        # asks for it is scalar-only, so ask only where it exists: sending it
        # to acoustic/elastic would be a TypeError, not the
        # NotImplementedError this test is about.
        if name == "scalar":
            kwargs["backward_callback_needs_forward_wavefield"] = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn = getattr(deepwave, name)
            out = fn(primary, *rest, **kwargs)
            # scalar's/acoustic's/elastic's primary receiver output is the
            # last element of the returned tuple for these three propagators.
            out[-1].sum().backward()
        results[name] = outcome["role_forward"]

    assert results == {"scalar": True, "acoustic": False, "elastic": False}
    for name, role_forward in results.items():
        assert _ROWS[name].role_forward == role_forward, (
            f"{name}: independent role='forward' probe ({role_forward}) "
            f"disagrees with probe_capabilities() ({_ROWS[name].role_forward})"
        )


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_backward_callback_support_matches_runtime(name: str) -> None:
    """``backward_callback`` works on every compiled backend.

    Independently confirms, per propagator, whether a ``backward_callback``
    fires during ``.backward()`` on the default backend and on the eager
    backend, and cross-checks both against ``probe_capabilities()``.
    """
    fired_backends = []
    for backend, label in (("auto", "default"), ("eager", "eager")):
        fired = []

        def cb(state, fired=fired) -> None:
            fired.append(state.step)

        m = _media(2, _grid(2)["n"])
        primary = _positional(name, m)[0].clone().requires_grad_(True)
        rest = _positional(name, m)[1:]
        kwargs = _survey_and_backend_kwargs(name, 2, backend)
        kwargs["backward_callback"] = cb
        fn = getattr(deepwave, name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = fn(primary, *rest, **kwargs)
                out[-1].sum().backward()
        except Exception:  # noqa: BLE001, S112 - that backend rejects it
            continue
        if fired:
            fired_backends.append(label)
    expected = _ROWS[name].backward_callback
    actual = " + ".join(fired_backends) if fired_backends else "not supported"
    assert actual == expected, (
        f"{name}: independent backward_callback probe ({actual!r}) disagrees "
        f"with probe_capabilities() ({expected!r})"
    )


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_jvp_support_matches_runtime(name: str) -> None:
    """Independently re-derives which backends support ``torch.func.jvp``."""
    supported = []
    for backend, label in (("auto", "default"), ("eager", "eager")):
        m = _media(2, _grid(2)["n"])
        primary = _positional(name, m)[0]
        rest = _positional(name, m)[1:]
        tangent = torch.zeros_like(primary)

        def f(v, backend=backend, rest=rest):
            kwargs = _survey_and_backend_kwargs(name, 2, backend)
            fn = getattr(deepwave, name)
            return fn(v, *rest, **kwargs)[-1]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jvp(f, (primary,), (tangent,))
        except Exception:  # noqa: BLE001, S112 - that backend does not support jvp
            continue
        supported.append(label)
    expected = _ROWS[name].jvp
    actual = " + ".join(supported) if supported else "not supported"
    assert actual == expected, (
        f"{name}: independent jvp probe ({actual!r}) disagrees with "
        f"probe_capabilities() ({expected!r})"
    )


# ---------------------------------------------------------------------------
# Fail-loud audit: every propagator x kwarg/value combination that
# probe_capabilities() records as unsupported must RAISE, never silently
# no-op. A silent no-op is worse than simply lacking the option: it is
# indistinguishable from a correct run and would produce gradients computed
# as if the option were off with no indication anything was ignored. These
# tests assert the negative side of exactly the facts probe_capabilities()
# already establishes (reusing its rows, not a second hand-maintained
# table), independently re-calling the propagators via this file's own
# ``_media``/``_positional``/``_survey_and_backend_kwargs`` fixtures rather
# than deepwave._capabilities's internal ``_call``/``_receiver``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_second_order_unsupported_backend_raises(name: str) -> None:
    """An unsupported backend or effective stride actually raises.

    ``probe_capabilities()``'s own probe only requires that a second
    ``torch.autograd.grad`` call (with ``create_graph=True``) either succeeds
    or raises; it does not positively assert a raise for the unsupported
    case. Silently returning a wrong (or zero) second-order gradient instead
    of raising would be a strictly worse failure mode than the documented
    ``DeepwaveRuntimeError``.
    """
    claim = _ROWS[name].second_order
    supported = set(claim.split(" + ")) if claim != "not supported" else set()
    for backend, label in (("auto", "default"), ("eager", "eager")):
        if label in supported:
            continue
        m = _media(2, _grid(2)["n"])
        primary = _positional(name, m)[0].clone().requires_grad_(True)
        rest = _positional(name, m)[1:]
        kwargs = _survey_and_backend_kwargs(name, 2, backend)
        # Keep the declared conservative speed bound while making this
        # particular second-order call an effective-stride-one experiment.
        kwargs["dt"] = 0.0001
        fn = getattr(deepwave, name)

        def run(*, sampling_interval: int = 1) -> None:
            out = fn(
                primary,
                *rest,
                **kwargs,
                model_gradient_sampling_interval=sampling_interval,
            )
            (grad,) = torch.autograd.grad(
                (out[-1] ** 2).sum(), primary, create_graph=True
            )
            torch.autograd.grad((grad * torch.ones_like(grad)).sum(), primary)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if f"{label} (effective stride 1)" in supported:
                # This call's dt is CFL-stable at its explicit speed bound.
                # The ordinary call must work;
                # sampling every other model-gradient step must refuse it.
                run()
                with pytest.raises(
                    deepwave.DeepwaveRuntimeError, match=r"sampling|sampled"
                ):
                    run(sampling_interval=2)
            else:
                with pytest.raises(
                    deepwave.DeepwaveRuntimeError, match=r"Second-order"
                ):
                    run()


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_batching_unsupported_raises(name: str) -> None:
    """Where ``batching`` is "not supported", ``vmap`` over the model raises."""
    if _ROWS[name].batching != "not supported":
        pytest.skip(f"{name} supports batching per probe_capabilities()")
    m = _media(2, _grid(2)["n"])
    primary = _positional(name, m)[0]
    batch = torch.stack([primary, primary * 1.001, primary * 1.002])
    rest = _positional(name, m)[1:]
    kwargs = _survey_and_backend_kwargs(name, 2, "auto")
    fn = getattr(deepwave, name)

    def f(v: torch.Tensor) -> torch.Tensor:
        return fn(v, *rest, **kwargs)[-1]

    def run() -> None:
        vmap(f)(batch)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotImplementedError, match="vmap"):
            run()


#: The ``boundary``/``pml_width`` pair behind each face label the "Boundary
#: faces accepted" column can name -- written out again here rather than
#: imported from ``deepwave._capabilities._BOUNDARY_PROBES``, so that a bug in
#: that module's own probe configuration cannot hide from this test. A
#: non-absorbing face is probed with ``pml_width=0`` everywhere; ``pml`` keeps
#: a real width, since a zero-width PML absorbs nothing.
_BOUNDARY_CONFIGS: dict[str, tuple[list[str], object]] = {
    "free on every face": (["free"] * 4, [0, 0, 0, 0]),
    "free on the low faces": (["free", "pml", "free", "pml"], [0, 0, 0, 0]),
    "rigid on every face": (["rigid"] * 4, [0, 0, 0, 0]),
    "rigid on the low faces": (["rigid", "pml", "rigid", "pml"], [0, 0, 0, 0]),
    "periodic": (["periodic"] * 4, [0, 0, 0, 0]),
    "pml": (["pml"] * 4, 4),
    "mixed per-face combination": (
        ["free", "pml", "rigid", "pml"],
        [0, 0, 0, 0],
    ),
}


def _claimed_boundary_backends(cell: str) -> dict[str, set[str]]:
    """Parses a "Boundary faces accepted" cell into face -> backends claimed.

    The cell groups face types that behave identically, e.g.
    ``"free, rigid (default + eager); periodic (not accepted)"``.
    """
    claimed: dict[str, set[str]] = {}
    for part in cell.split("; "):
        match = re.fullmatch(r"(.+) \((.+)\)", part)
        assert match is not None, f"unparsable boundary-faces cell: {cell!r}"
        faces, labels = match.group(1), match.group(2)
        backends = set() if labels == "not accepted" else set(labels.split(" + "))
        for face in faces.split(", "):
            claimed[face] = backends
    return claimed


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_boundary_faces_accepted_matches_runtime(name: str) -> None:
    """Every face type is accepted on exactly the backends its row names.

    The positive half (a claimed face type really does run) and the negative
    half (an unclaimed one really does raise, rather than silently running
    with some other boundary condition) are both checked, per backend, from
    this file's own independently-built minimal problem.
    """
    claimed = _claimed_boundary_backends(_ROWS[name].boundary_faces)
    assert set(claimed) == set(_BOUNDARY_CONFIGS), (name, sorted(claimed))
    for face, (boundary, pml_width) in _BOUNDARY_CONFIGS.items():
        for backend, label in (("auto", "default"), ("eager", "eager")):

            def run(
                backend: str = backend,
                boundary: list[str] = boundary,
                pml_width: object = pml_width,
            ) -> None:
                _minimal_call(
                    name,
                    backend=backend,
                    boundary=boundary,
                    pml_width=pml_width,
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if label in claimed[face]:
                    run()
                else:
                    with pytest.raises(Exception, match=r".+"):
                        run()


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_storage_modes_unsupported_raise(name: str) -> None:
    """A ``storage_mode`` absent from the propagator's row actually raises."""
    unsupported = {"device", "cpu", "disk", "none"} - set(_ROWS[name].storage_modes)
    for mode in unsupported:
        overrides: dict = {"storage_mode": mode}
        with tempfile.TemporaryDirectory() as storage_path:
            if mode == "disk":
                overrides["storage_path"] = storage_path
            with pytest.raises(Exception, match=r".+"):
                _minimal_call(name, **overrides)


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_python_backend_storage_modes_match_runtime(name: str) -> None:
    """``python_backend_storage_modes`` agrees with real backend behaviour.

    For each of ``backend="eager"``/``"torch_compile"`` and each of the
    four ``storage_mode`` values, a real call must raise exactly when the
    propagator's declared ``python_backend_storage_modes`` (surfaced here as
    ``probe_capabilities()``'s field of the same name, read directly from
    ``PropagatorSpec`` rather than probed) excludes that mode, and must
    succeed when it includes it. The raise is pinned down to
    ``deepwave._engine.reject_unsupported_python_backend_options``'s actual
    ``RuntimeError`` and wording (naming both the rejected ``storage_mode``
    and the backend actually requested), not just "some exception" -- a
    message that named a different backend than the one actually requested
    must fail this test.

    The backend named in that message is not always the spelling this test
    passes in: for a propagator with a compiled C/CUDA backend, reaching
    this rejection at all means the caller explicitly chose one of the two
    Python modes, so the message uses the resolved internal mode name
    (``"compile"``); for the two propagators with no compiled
    backend at all, ``backend`` is reported as the caller's own
    pre-resolution (canonical) choice instead -- see
    ``reject_unsupported_python_backend_options``'s own docstring.
    """
    supported = set(_ROWS[name].python_backend_storage_modes)
    has_compiled_backend = deepwave.checkpointing._get_spec(
        getattr(deepwave, name)
    ).has_compiled_backend
    for backend, internal_mode in (
        ("eager", "eager"),
        ("torch_compile", "compile"),
    ):
        reported_backend = internal_mode if has_compiled_backend else backend
        for mode in ("device", "cpu", "disk", "none"):
            overrides: dict = {"storage_mode": mode}
            with tempfile.TemporaryDirectory() as storage_path:
                if mode == "disk":
                    overrides["storage_path"] = storage_path
                if mode in supported:
                    _minimal_call(name, backend=backend, **overrides)
                else:
                    expected = (
                        re.escape(f'storage_mode="{mode}"')
                        + r".*"
                        + re.escape(f'backend="{reported_backend}"')
                    )
                    with pytest.raises(RuntimeError, match=expected):
                        _minimal_call(name, backend=backend, **overrides)


@pytest.mark.parametrize("name", PROPAGATOR_NAMES)
def test_backward_callback_unsupported_backend_raises(name: str) -> None:
    """Where ``backward_callback`` is not counted as supported, the call raises.

    ``test_backward_callback_support_matches_runtime`` above only requires
    that an unsupported backend's callback did not *fire*; that would also
    be satisfied by a silent no-op (the call succeeding without ever
    invoking the callback, with no indication anything was skipped). This
    closes that gap by requiring the unsupported case to be a genuine raise.
    """
    claim = _ROWS[name].backward_callback
    supported = set(claim.split(" + ")) if claim != "not supported" else set()
    for backend, label in (("auto", "default"), ("eager", "eager")):
        if label in supported:
            continue
        m = _media(2, _grid(2)["n"])

        def cb(state) -> None:
            raise AssertionError("backward_callback fired despite being unsupported")

        primary = _positional(name, m)[0].clone().requires_grad_(True)
        rest = _positional(name, m)[1:]
        kwargs = _survey_and_backend_kwargs(name, 2, backend)
        kwargs["backward_callback"] = cb
        fn = getattr(deepwave, name)

        def run() -> None:
            out = fn(primary, *rest, **kwargs)
            out[-1].sum().backward()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(Exception, match=r".+"):
                run()


def test_experimental_status_matches_the_documented_set() -> None:
    """The seven never-released propagators are experimental, the other four stable.

    Two independent derivations of the same fact must agree: the tier each
    propagator declares on its own ``PropagatorSpec``, and the tier the
    capability matrix reports. The set itself is the one written down in
    ``docs/stability.rst``, spelled out here
    rather than derived, so that adding or graduating a propagator has to be
    a deliberate edit in both places.

    Re-checking this by *calling* each propagator and looking for the warning
    (what this test used to do) is no longer a valid independent derivation:
    the warning fires once per process per propagator, so any earlier test in
    the same process consumes it. ``tests/test_stability.py`` covers the
    warning-emission behaviour directly, clearing the guard first.
    """
    expected_experimental = {
        "acoustic_born",
        "acoustic_vti",
        "viscoacoustic",
        "elastic_born",
        "elastic_vti",
    }
    assert expected_experimental <= set(PROPAGATOR_NAMES)
    for name in PROPAGATOR_NAMES:
        expected_status = "experimental" if name in expected_experimental else "stable"
        spec = deepwave.checkpointing._get_spec(getattr(deepwave, name))
        assert spec.status == expected_status, name
        assert _ROWS[name].status == expected_status, name


@pytest.mark.parametrize("name", ["scalar", "acoustic"])
def test_unsupported_storage_mode_typo_raises_standard_value_error(name: str) -> None:
    """A ``storage_mode`` typo is the same ``ValueError`` on every propagator.

    Before this fix, the opt-in pure-Python backend reported a
    ``storage_mode`` typo as a backend-capability limit (a ``RuntimeError``
    claiming, incorrectly, that e.g. "cpu/disk/none" storage would have
    worked). It must instead be the same ``ValueError`` every propagator's
    default (compiled) backend already raises for a typo.
    """
    m = _media(2, _grid(2)["n"])
    fn = getattr(deepwave, name)
    # The eager path has its own option validation; the default compiled
    # backend's separate resolve_storage_mode check already raised correctly.
    kwargs = _survey_and_backend_kwargs(name, 2, "eager")
    kwargs["storage_mode"] = "bogus"
    with pytest.raises(
        ValueError,
        match=r"storage_mode must be 'device', 'cpu', 'disk', or 'none', but got bogus",
    ):
        fn(*_positional(name, m), **kwargs)


#: The statement ``docs/capabilities.rst`` makes about wrapping a propagator
#: in an *outer* ``torch.compile`` (as opposed to the supported
#: ``backend="torch_compile"`` option, which compiles Deepwave's own eager
#: reference implementation). Kept as fragments rather than one block of
#: prose so that rewrapping the page does not break the test.
_OUTER_COMPILE_DOC_FRAGMENTS = (
    "Deepwave's compiled backend enters C/CUDA through ``ctypes``",
    "will graph-break",
    "and is untested",
    "cannot serialise it",
    'The ``backend="torch_compile"``',
    "**option** is different and is supported",
)


def test_outer_torch_compile_is_documented_as_unsupported() -> None:
    """Documents the outer-``torch.compile`` graph break; claims no support.

    Deepwave's compiled backend reaches C/CUDA through ``ctypes``, which
    Dynamo cannot trace, so ``torch.compile`` around a propagator call can
    only graph-break at that call. Whether Dynamo then falls back and runs
    the call eagerly (what it does today) or gives up and raises is an
    implementation detail of whichever PyTorch version is installed, and
    Deepwave supports neither outcome: this test therefore accepts both and
    asserts only what the documentation actually claims, so that a future
    PyTorch release changing the outcome does not read as a capability
    Deepwave has measured and can promise.
    """
    m = _media(2, _grid(2)["n"])
    kwargs = _survey_and_backend_kwargs("scalar", 2, "auto")

    def run() -> torch.Tensor:
        return deepwave.scalar(*_positional("scalar", m), **kwargs)[-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = torch.compile(run)()
        except Exception:  # noqa: BLE001, S110 - either outcome is "not supported"
            pass
        else:
            # If it did run, it must at least have produced the same answer
            # the uncompiled call does -- a wrong answer would be worse than
            # the documented graph break.
            assert torch.allclose(out, run())

    page = pathlib.Path(__file__).resolve().parents[1] / "docs" / "capabilities.rst"
    if not page.is_file():  # pragma: no cover - source checkout only
        pytest.skip("docs/capabilities.rst is not present in this checkout")
    text = page.read_text()
    for fragment in _OUTER_COMPILE_DOC_FRAGMENTS:
        assert fragment in text, fragment
