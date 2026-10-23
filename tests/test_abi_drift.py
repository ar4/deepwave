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

import pytest
import torch

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


def _run_with_recording(thunk):
    """Run ``thunk`` with backend lookups wrapped to record call arguments."""
    import deepwave._common as common

    real_lookup = common._backend_function
    records = []

    def wrapped_lookup(*a, **k):
        func = real_lookup(*a, **k)

        def recorder(*args):
            records.append((func, args))
            return func(*args)

        return recorder

    common._backend_function = wrapped_lookup
    try:
        thunk()
    finally:
        common._backend_function = real_lookup
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
