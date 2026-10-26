"""Cross-check that each propagator's ctypes call matches its argtypes.

The argtype templates in ``backend_utils`` and the positional ``state.backend(...)``
calls in ``common``/``scalar``/``scalar_born`` are two hand-written encodings of
the same C signature. Nothing else verifies they agree -- a missing or extra
argument, or a wrong type class (e.g. a float where the C function expects a
``bool``), would silently corrupt the call and only surface in a gradient check.

This test wraps the real backend functions with a recorder, runs a tiny forward
and backward of every propagator, and asserts that for every call the emitted
argument list has the same length and compatible per-position type class as the
function's assigned ``argtypes``.
"""

import ctypes

import determinism_cases as dc
import pytest
import torch

import deepwave
from deepwave import (
    acoustic,
    acoustic_born,
    acoustic_vti,
    elastic,
    elastic_born,
    elastic_vti,
    scalar,
    scalar_born,
    viscoacoustic,
    vpvsrho_to_lambmubuoyancy,
)
from deepwave.wavelets import ricker

# Common base class of all ctypes objects (not exposed publicly as ctypes._CData).
_CDATA_BASE = ctypes.c_void_p.__mro__[-2]


def _type_ok(argtype, value):
    """Whether ``value`` is consistent with the ctypes ``argtype`` class."""
    if argtype is ctypes.c_bool:
        return isinstance(value, bool)
    if argtype in (ctypes.c_float, ctypes.c_double):
        return isinstance(value, float) and not isinstance(value, bool)
    if argtype in (ctypes.c_int64, ctypes.c_size_t):
        # bool is an int subclass; the backend never wants a bool here.
        return isinstance(value, int) and not isinstance(value, bool)
    if argtype is ctypes.c_void_p:
        # A pointer: an address (int), NULL (0/None), an object exposing the
        # ctypes _as_parameter_ protocol (e.g. a CUDA stream), or any ctypes
        # object such as the c_char_p filename array (passed by reference).
        return (
            value is None
            or isinstance(value, int)
            or hasattr(value, "_as_parameter_")
            or isinstance(value, _CDATA_BASE)
        )
    return True


def _run_with_recording(thunk, *, check_thread_range=False):
    """Run ``thunk`` with backend lookups wrapped to record call arguments."""
    from deepwave import _backend_utils as backend_utils

    real_lookup = backend_utils.get_backend_function
    records = []

    def wrapped_lookup(*a, **k):
        func = real_lookup(*a, **k)

        def recorder(*args):
            if check_thread_range:
                assert func.argtypes[-2] is ctypes.c_int64
                for count in (0, -1, 2**31, 2**63 - 1):
                    invalid = (*args[:-2], count, args[-1])
                    assert func(*invalid) == 1, (func.__name__, count)
            records.append((func, args))
            return func(*args)

        return recorder

    backend_utils.get_backend_function = wrapped_lookup
    try:
        thunk()
    finally:
        backend_utils.get_backend_function = real_lookup
    return records


def _assert_records(records):
    assert records, "no backend calls were recorded"
    for func, args in records:
        argtypes = func.argtypes
        assert argtypes is not None
        assert len(args) == len(argtypes), (
            f"{func.__name__ if hasattr(func, '__name__') else func}: "
            f"emitted {len(args)} args but argtypes has {len(argtypes)}"
        )
        for i, (t, v) in enumerate(zip(argtypes, args)):
            assert _type_ok(t, v), f"arg {i}: value {v!r} not compatible with {t}"


def _grid(ndim, dtype):
    n = (12,) * ndim
    nt = 8
    dx = 5.0
    dt = 0.0004
    src = ricker(25.0, nt, dt, 0.06).to(dtype).reshape(1, 1, nt)
    src_loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    src_loc[..., :] = 6
    rec_loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    rec_loc[..., :] = 4
    return n, dx, dt, src, src_loc, rec_loc


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scalar_call_matches_argtypes(ndim, dtype):
    """The scalar call emits arguments consistent with its argtypes."""
    n, dx, dt, src, src_loc, rec_loc = _grid(ndim, dtype)
    v = torch.full(n, 1500.0, dtype=dtype, requires_grad=True)

    def thunk():
        out = scalar(
            v,
            dx,
            dt,
            source_amplitudes=src,
            source_locations=src_loc,
            receiver_locations=rec_loc,
            pml_width=2,
            accuracy=2,
        )
        out[-1].sum().backward()

    _assert_records(_run_with_recording(thunk))


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scalar_born_call_matches_argtypes(ndim, dtype):
    """The scalar Born call emits arguments consistent with its argtypes."""
    n, dx, dt, src, src_loc, rec_loc = _grid(ndim, dtype)
    v = torch.full(n, 1500.0, dtype=dtype)
    scat = torch.zeros(n, dtype=dtype, requires_grad=True)

    def thunk():
        out = scalar_born(
            v,
            dx,
            dt,
            dv=scat,
            source_amplitudes=src,
            source_locations=src_loc,
            sc_receiver_locations=rec_loc,
            bg_receiver_locations=rec_loc,
            pml_width=2,
            accuracy=2,
        )
        out[-1].sum().backward()

    _assert_records(_run_with_recording(thunk))


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_acoustic_call_matches_argtypes(ndim, dtype):
    """The acoustic call emits arguments consistent with its argtypes."""
    n, dx, dt, src, src_loc, rec_loc = _grid(ndim, dtype)
    v = torch.full(n, 1500.0, dtype=dtype, requires_grad=True)
    rho = torch.full(n, 2200.0, dtype=dtype)

    def thunk():
        out = acoustic(
            v,
            rho,
            dx,
            dt,
            source_amplitudes_volume_rate=src,
            source_locations_volume_rate=src_loc,
            receiver_locations_p=rec_loc,
            pml_width=2,
            accuracy=2,
        )
        out[-1].sum().backward()

    _assert_records(_run_with_recording(thunk))


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_acoustic_born_call_matches_argtypes(ndim, dtype):
    """The acoustic Born call emits arguments consistent with its argtypes."""
    n, dx, dt, src, src_loc, rec_loc = _grid(ndim, dtype)
    v = torch.full(n, 1500.0, dtype=dtype)
    rho = torch.full(n, 2200.0, dtype=dtype)
    dv = torch.zeros(n, dtype=dtype, requires_grad=True)

    def thunk():
        out = acoustic_born(
            v,
            rho,
            dx,
            dt,
            dv=dv,
            source_amplitudes_volume_rate=src,
            source_locations_volume_rate=src_loc,
            sc_receiver_locations_p=rec_loc,
            bg_receiver_locations_p=rec_loc,
            pml_width=2,
            accuracy=2,
        )
        out[-1].sum().backward()

    _assert_records(_run_with_recording(thunk))


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_acoustic_vti_call_matches_argtypes(ndim, dtype):
    """The acoustic_vti call emits arguments consistent with its argtypes."""
    n, dx, dt, src, src_loc, rec_loc = _grid(ndim, dtype)
    c33 = torch.full(n, 2200.0 * 1500.0**2, dtype=dtype, requires_grad=True)
    c11 = c33 * 1.4
    c13 = c33 * 1.1
    buoyancy = torch.full(n, 1 / 2200.0, dtype=dtype)

    def thunk():
        out = acoustic_vti(
            c11,
            c13,
            c33,
            buoyancy,
            dx,
            dt,
            source_amplitudes_y=src,
            source_locations_y=src_loc,
            receiver_locations_y=rec_loc,
            pml_width=2,
            accuracy=2,
        )
        out[-1].sum().backward()

    _assert_records(_run_with_recording(thunk))


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_elastic_call_matches_argtypes(ndim, dtype):
    """The elastic call emits arguments consistent with its argtypes."""
    n, dx, dt, src, src_loc, rec_loc = _grid(ndim, dtype)
    vp = torch.full(n, 1500.0, dtype=dtype)
    vs = torch.full(n, 1100.0, dtype=dtype)
    rho = torch.full(n, 2200.0, dtype=dtype)
    lamb, mu, buoy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    lamb = lamb.requires_grad_(True)

    def thunk():
        out = elastic(
            lamb,
            mu,
            buoy,
            dx,
            dt,
            source_amplitudes_y=src,
            source_locations_y=src_loc,
            receiver_locations_y=rec_loc,
            pml_width=2,
            accuracy=2,
        )
        out[-1].sum().backward()

    _assert_records(_run_with_recording(thunk))


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_elastic_born_call_matches_argtypes(ndim, dtype):
    """elastic_born's forward + FULL backward calls match their argtypes.

    elastic_born's backend calls come from two independently hand-written
    encodings (``backend_utils`` templates and the ``_make_elastic_born_*_call``
    closures) that this test guards against silent drift. Requiring grad on a
    BACKGROUND model (``lamb``) and a SCATTERING model (``dlamb``) exercises the
    forward kernel and the full two-field ``backward`` kernel (SS4.2/SS4.3).
    """
    n, dx, dt, src, src_loc, rec_loc = _grid(ndim, dtype)
    vp = torch.full(n, 1500.0, dtype=dtype)
    vs = torch.full(n, 1100.0, dtype=dtype)
    rho = torch.full(n, 2200.0, dtype=dtype)
    lamb, mu, buoy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    lamb = lamb.requires_grad_(True)
    dlamb = torch.zeros(n, dtype=dtype).requires_grad_(True)

    def thunk():
        out = elastic_born(
            lamb,
            mu,
            buoy,
            dx,
            dt,
            dlamb=dlamb,
            source_amplitudes_y=src,
            source_locations_y=src_loc,
            sc_receiver_locations_y=rec_loc,
            pml_width=2,
            accuracy=2,
        )
        # Seed both a background and a scattered receiver output so both adjoint
        # fields are driven.
        (out[-1].sum() + out[-(ndim + 2)].sum()).backward()

    _assert_records(_run_with_recording(thunk))


@pytest.mark.parametrize("ndim", [2, 3])
def test_elastic_born_backward_sc_call_matches_argtypes(ndim):
    """elastic_born's forward + scattered-only ``backward_sc`` calls match argtypes.

    Requiring grad on ONLY a scattering model (``dlamb``) routes the backward
    through the compiled scattered-only adjoint (``backward_sc``, SS4.1), the
    third elastic_born kernel (the other two are covered by
    ``test_elastic_born_call_matches_argtypes``).
    """
    dtype = torch.float64
    n, dx, dt, src, src_loc, rec_loc = _grid(ndim, dtype)
    vp = torch.full(n, 1500.0, dtype=dtype)
    vs = torch.full(n, 1100.0, dtype=dtype)
    rho = torch.full(n, 2200.0, dtype=dtype)
    lamb, mu, buoy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    dlamb = torch.zeros(n, dtype=dtype).requires_grad_(True)

    def thunk():
        out = elastic_born(
            lamb,
            mu,
            buoy,
            dx,
            dt,
            dlamb=dlamb,
            source_amplitudes_y=src,
            source_locations_y=src_loc,
            sc_receiver_locations_y=rec_loc,
            pml_width=2,
            accuracy=2,
        )
        out[-1].sum().backward()

    _assert_records(_run_with_recording(thunk))


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_elastic_vti_call_matches_argtypes(ndim, dtype):
    """The elastic_vti call emits arguments consistent with its argtypes."""
    n, dx, dt, src, src_loc, rec_loc = _grid(ndim, dtype)
    vp = torch.full(n, 1500.0, dtype=dtype)
    vs = torch.full(n, 1100.0, dtype=dtype)
    rho = torch.full(n, 2200.0, dtype=dtype)
    lamb, mu, buoy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    lamb = lamb.requires_grad_(True)
    c11 = lamb + 2 * mu
    c13 = lamb
    c33 = c11
    c55 = mu
    c66 = mu if ndim == 3 else None

    def thunk():
        out = elastic_vti(
            c11,
            c13,
            c33,
            c55,
            buoy,
            dx,
            dt,
            c66=c66,
            source_amplitudes_y=src,
            source_locations_y=src_loc,
            receiver_locations_y=rec_loc,
            pml_width=2,
            accuracy=2,
        )
        out[-1].sum().backward()

    _assert_records(_run_with_recording(thunk))


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_viscoacoustic_call_matches_argtypes(ndim, dtype):
    """The viscoacoustic call emits arguments consistent with its argtypes."""
    n, dx, dt, src, src_loc, rec_loc = _grid(ndim, dtype)
    v = torch.full(n, 1500.0, dtype=dtype, requires_grad=True)
    rho = torch.full(n, 2200.0, dtype=dtype)
    q = torch.full(n, 20.0, dtype=dtype)

    def thunk():
        out = viscoacoustic(
            v,
            rho,
            q,
            dx,
            dt,
            source_amplitudes_pressure_rate=src,
            source_locations_pressure_rate=src_loc,
            receiver_locations_p=rec_loc,
            pml_width=2,
            accuracy=2,
            q_freq=25.0,
        )
        out[-1].sum().backward()

    _assert_records(_run_with_recording(thunk))


@pytest.mark.parametrize(
    ("name", "scatter_only"),
    [(name, False) for name in dc.PROPAGATORS]
    + [(name, True) for name in ("scalar_born", "acoustic_born", "elastic_born")],
)
def test_native_thread_count_rejects_truncation(name, scatter_only, monkeypatch):
    """All CPU entry paths reject invalid int controls before using any buffers."""
    monkeypatch.setattr(dc, "N", 32)
    monkeypatch.setattr(dc, "PML_WIDTH", 4)

    def run():
        kwargs, leaves, names = dc.build_case(
            name, torch.device("cpu"), torch.float64, backend="compiled", nt=3
        )
        models = kwargs.pop("_models")
        if scatter_only:
            perturbations = {
                "model:dv",
                "model:drho",
                "model:dlamb",
                "model:dmu",
                "model:dbuoyancy",
            }
            for leaf, label in zip(leaves, names):
                leaf.requires_grad_(label in perturbations)
        kwargs["origin"] = [0, 0]
        for label in names:
            if label.startswith("wavefield:"):
                field = label.removeprefix("wavefield:")
                kwargs[field] = torch.nn.functional.pad(kwargs[field], (4, 4, 4, 4))
        with torch.no_grad():
            for leaf, label in zip(leaves, names):
                if label == "model:dv":
                    leaf[0, :] = leaf[-1, :] = 0
                    leaf[:, 0] = leaf[:, -1] = 0
        output = getattr(deepwave, name)(*models, dc.DX, dc.DT / 2, **kwargs)
        from helpers import output_tensors

        loss = sum(t.square().mean() for t in output_tensors(output) if t.numel())
        torch.autograd.grad(
            loss, [t for t in leaves if t.requires_grad], allow_unused=True
        )

    records = _run_with_recording(run, check_thread_range=True)
    _assert_records(records)
    directions = {func.__name__.split("_double_")[1] for func, _ in records}
    assert "forward_cpu" in directions
    assert ("backward_sc_cpu" if scatter_only else "backward_cpu") in directions
