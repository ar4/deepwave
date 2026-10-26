"""Tests for deepwave._location_interpolation."""

import warnings
from collections.abc import Callable, Iterator

import pytest
import torch
from helpers import direct_2d_approx

import deepwave
import deepwave._common
from deepwave._location_interpolation import (
    _STENCIL_CAUSE,
    _TRUNCATED_COORDINATE_CAUSE,
    Hicks,
    _combine_weights,
    _get_hicks_for_one_location_dim,
    _grid_locations,
    _has_off_node_coord,
    _resolve_interpolator,
    _should_interpolate,
)


def test_hicks_init_invalid_locations_type() -> None:
    """Test that Hicks.__init__ raises TypeError if locations is not a tensor."""
    with pytest.raises(TypeError, match=r"locations must be a torch.Tensor."):
        Hicks([1, 2, 3])


def test_hicks_init_invalid_locations_ndim() -> None:
    """Test raises if locations does not have three dimensions."""
    locations = torch.randn(10, 2)  # Should be 3D
    with pytest.raises(RuntimeError, match=r"locations must have three dimensions."):
        Hicks(locations)


def test_hicks_init_zero_batch_size() -> None:
    """Test that Hicks.__init__ handles zero batch size."""
    locations = torch.randn(0, 10, 2)
    hicks = Hicks(locations)
    assert hicks.locations.shape[0] == 0


def test_hicks_init_zero_sources_per_shot() -> None:
    """Test that Hicks.__init__ handles zero sources per shot."""
    locations = torch.randn(10, 0, 2)
    hicks = Hicks(locations)
    assert hicks.locations.shape[1] == 0


def test_hicks_init_invalid_dtype() -> None:
    """Test that Hicks.__init__ raises TypeError if dtype is not a torch.dtype."""
    locations = torch.randn(10, 10, 2)
    with pytest.raises(TypeError, match=r"dtype must be a torch.dtype."):
        Hicks(locations, dtype="float")


@pytest.mark.parametrize("side", ["low", "high"])
@pytest.mark.parametrize("surface", ["free", "rigid"])
def test_hicks_cached_boundary_stencil_is_point_independent(side, surface) -> None:
    """Folding one point must not change a later point's cached stencil."""
    near, interior = (0.25, 12.25) if side == "low" else (39.25, 20.25)
    face = [side == "low", side == "high"]
    options = {
        "model_shape": [40],
        "dtype": torch.double,
        "free_surfaces": face if surface == "free" else None,
        "rigid_surfaces": face if surface == "rigid" else None,
    }

    def spread(coords: list[float], amplitudes: list[float]) -> dict[int, float]:
        hicks = Hicks(
            torch.tensor([[[coord] for coord in coords]], dtype=torch.double), **options
        )
        values = hicks.source(
            torch.tensor([[[amp] for amp in amplitudes]], dtype=torch.double)
        )
        locations = hicks.get_locations()[0, :, 0].tolist()
        return dict(zip(locations, values[0, :, 0].tolist()))

    separate = [spread([coord], [amp]) for coord, amp in ((near, 2.0), (interior, 3.0))]
    expected: dict[int, float] = {}
    for point in separate:
        for cell, value in point.items():
            expected[cell] = expected.get(cell, 0.0) + value
    for coords, amps in (
        ([near, interior], [2.0, 3.0]),
        ([interior, near], [3.0, 2.0]),
    ):
        actual = spread(coords, amps)
        assert actual.keys() == expected.keys()
        for cell, value in expected.items():
            assert actual[cell] == pytest.approx(value, abs=1e-12)

        hicks = Hicks(
            torch.tensor([[[coord] for coord in coords]], dtype=torch.double),
            **options,
        )
        cells = hicks.get_locations()[0, :, 0].to(torch.double)
        measured = hicks.receiver((cells.square() + 1).reshape(1, -1, 1))
        for index, coord in enumerate(coords):
            alone = Hicks(torch.tensor([[[coord]]], dtype=torch.double), **options)
            alone_cells = alone.get_locations()[0, :, 0].to(torch.double)
            reference = alone.receiver((alone_cells.square() + 1).reshape(1, -1, 1))
            assert torch.allclose(measured[0, index, 0], reference[0, 0, 0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hicks_cpu_geometry_cuda_source_keeps_gradients() -> None:
    """Source weights transfer to CUDA with their position gradient intact."""
    coords = torch.tensor(
        [[[8.25], [deepwave.IGNORE_LOCATION]]], dtype=torch.double, requires_grad=True
    )
    hicks = Hicks(coords, dtype=torch.double, model_shape=[32], differentiable=True)
    amplitudes = torch.tensor(
        [[[2.0], [5.0]]], device="cuda", dtype=torch.double, requires_grad=True
    )
    out = hicks.source(amplitudes)
    assert out.device.type == "cuda"
    assert torch.count_nonzero(out).item() > 0
    out.square().sum().backward()
    assert coords.grad is not None
    assert coords.grad[0, 0].abs().sum() > 0
    assert amplitudes.grad is not None
    assert amplitudes.grad[0, 0].abs().sum() > 0
    assert amplitudes.grad[0, 1].abs().sum() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_propagation_with_cpu_hicks_geometry_has_position_gradient() -> None:
    """Explicit and automatic Hicks placement retain a useful CUDA gradient."""
    model = torch.full((32,), 1500.0, device="cuda", dtype=torch.double)
    receiver = torch.tensor([[[9]]], device="cuda")

    def propagate(manual: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coords = torch.tensor([[[8.25]]], dtype=torch.double, requires_grad=True)
        amplitudes = torch.zeros((1, 1, 32), device="cuda", dtype=torch.double)
        amplitudes[0, 0, 3] = 1.0
        amplitudes.requires_grad_()
        if manual:
            hicks = Hicks(
                coords, dtype=torch.double, model_shape=[32], differentiable=True
            )
            spread = hicks.source(amplitudes)
            locations = hicks.get_locations()
        else:
            spread = amplitudes
            locations = coords
        out = deepwave.scalar(
            model,
            10.0,
            0.001,
            source_amplitudes=spread,
            source_locations=locations,
            receiver_locations=receiver,
            pml_width=4,
            pml_freq=25.0,
            max_vel=1500.0,
        ).receiver_amplitudes
        out.square().sum().backward()
        assert coords.grad is not None
        assert amplitudes.grad is not None
        return out.detach(), coords.grad.clone(), amplitudes.grad.clone()

    manual_out, manual_position_grad, manual_source_grad = propagate(True)
    auto_out, auto_position_grad, auto_source_grad = propagate(False)
    assert manual_out.abs().max() > 0
    assert manual_position_grad.abs().max() > 0
    assert manual_source_grad.abs().max() > 0
    assert torch.allclose(auto_out, manual_out)
    assert torch.allclose(auto_position_grad, manual_position_grad)
    assert torch.allclose(auto_source_grad, manual_source_grad)


def test_monopole(
    c=1500,
    freq=25,
    dx=(5, 5),
    dt=0.001,
    nx=(50, 50),
    nt=200,
    device=None,
    dtype=torch.double,
):
    """Test monopole source and receiver with Hicks interpolation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peak_time = 1.5 / freq
    model = torch.ones(*nx, dtype=dtype, device=device) * c
    dx = torch.tensor(dx)
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time, dtype=dtype)
        .reshape(1, 1, -1)
        .to(device)
        .repeat(20, 1, 1)
    )
    source_locations = torch.zeros(20, 1, 2, device=device)
    source_locations[..., 0] = (
        torch.linspace(20.0, 20.9, 10, device=device).reshape(10, 1).repeat(2, 1)
    )
    source_locations[:10, :, 1] = 25.0
    source_locations[10:, :, 1] = 25.25
    receiver_locations = torch.zeros(20, 20, 2, device=device)
    receiver_locations[..., 0] = (
        torch.linspace(30.0, 30.9, 10, device=device).reshape(1, -1).repeat(20, 2)
    )
    receiver_locations[:10, :, 1] = 25.0
    receiver_locations[10:, :, 1] = 25.25
    hicks_source = Hicks(source_locations, dtype=dtype, model_shape=list(nx))
    hicks_source_locations = hicks_source.hicks_locations
    hicks_amplitudes = hicks_source.source(source_amplitudes)
    hicks_receiver = Hicks(receiver_locations, dtype=dtype, model_shape=list(nx))
    hicks_receiver_locations = hicks_receiver.hicks_locations
    o = deepwave.scalar(
        model,
        dx.tolist(),
        dt,
        source_amplitudes=hicks_amplitudes,
        source_locations=hicks_source_locations,
        receiver_locations=hicks_receiver_locations,
    )
    receiver_amplitudes = hicks_receiver.receiver(o[-1])
    for i, source_location in enumerate(source_locations):
        for j, receiver_location in enumerate(receiver_locations[i]):
            e = direct_2d_approx(
                receiver_location.float().cpu(),
                source_location[0].float().cpu(),
                dx,
                dt,
                c,
                -source_amplitudes[i].flatten().cpu(),
            )
            assert torch.allclose(receiver_amplitudes[i, j].cpu(), e, atol=0.025)


@pytest.mark.parametrize(
    ("nx", "dx", "src", "rec"),
    [
        ((80,), (5.0,), (20.3,), (55.7,)),
        ((40, 40), (5.0, 5.0), (20.3, 18.6), (12.4, 30.7)),
        ((16, 16, 16), (5.0, 5.0, 5.0), (5.3, 7.6, 8.2), (10.4, 4.7, 9.3)),
    ],
)
def test_hicks_reciprocity(
    nx,
    dx,
    src,
    rec,
    c=1500,
    freq=25,
    dt=0.001,
    nt=120,
    dtype=torch.double,
):
    """Off-grid Hicks source/receiver must satisfy reciprocity in 1-D/2-D/3-D.

    Swapping an (interpolated) source and receiver leaves the recorded trace
    unchanged, because the discrete Green's function is symmetric and Hicks
    injects and records with the same weights. This holds to numerical
    precision regardless of dispersion or boundary reflections, so it is a
    strong, analytic-free check of the N-D interpolation plumbing.
    """
    ndim = len(nx)
    model = torch.ones(*nx, dtype=dtype) * c
    wavelet = deepwave.wavelets.ricker(freq, nt, dt, 1.5 / freq, dtype=dtype).reshape(
        1,
        1,
        -1,
    )

    def trace(source_loc, receiver_loc):
        hicks_s = Hicks(
            torch.tensor([[source_loc]], dtype=dtype),
            dtype=dtype,
            model_shape=list(nx),
        )
        hicks_r = Hicks(
            torch.tensor([[receiver_loc]], dtype=dtype),
            dtype=dtype,
            model_shape=list(nx),
        )
        out = deepwave.scalar(
            model,
            list(dx),
            dt,
            source_amplitudes=hicks_s.source(wavelet),
            source_locations=hicks_s.get_locations(),
            receiver_locations=hicks_r.get_locations(),
        )
        return hicks_r.receiver(out[-1])[0, 0]

    forward = trace(src, rec)
    reverse = trace(rec, src)
    assert ndim  # parametrization sanity
    # Exact in real arithmetic; differs only by floating-point summation order.
    assert torch.allclose(forward, reverse, atol=1e-8, rtol=1e-4)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_on_node_reduces_to_integer(ndim, dtype=torch.double):
    """A point on a grid node must interpolate to exactly that single cell."""
    n_shots, n_per_shot, nt = 3, 4, 7
    model_shape = [40] * ndim
    # Distinct, well-separated integer cells (no dedup collisions) so the
    # interpolated locations line up one-to-one with the input points.
    base = torch.arange(n_per_shot).reshape(1, n_per_shot, 1) * 4 + 2
    shot_offset = torch.arange(n_shots).reshape(n_shots, 1, 1)
    locations = (base + shot_offset).expand(n_shots, n_per_shot, ndim).to(dtype)
    hicks = Hicks(locations, dtype=dtype, model_shape=model_shape)
    hicks_locations = hicks.get_locations()
    # Each point maps to exactly one grid cell, equal to its integer location.
    assert hicks_locations.shape == (n_shots, n_per_shot, ndim)
    assert torch.equal(hicks_locations, locations.long())
    amplitudes = torch.randn(n_shots, n_per_shot, nt, dtype=dtype)
    spread = hicks.source(amplitudes)
    # On-node spreading has unit weight, so amplitudes are unchanged.
    assert torch.allclose(spread, amplitudes)
    # Receiver is the exact inverse on-node.
    gathered = hicks.receiver(spread)
    assert torch.allclose(gathered, amplitudes)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_offset_shifts_into_component_index_space(ndim, dtype=torch.double):
    """An offset must shift interpolation into the component's index space.

    Interpolating location ``L`` with component ``offset`` must give the same
    grid cells (shifted by the integer part of the offset) and identical weights
    as interpolating ``L - offset`` with no offset.
    """
    torch.manual_seed(1)
    n_shots, n_per_shot = 2, 3
    offset = [0.5] * ndim
    locations = torch.rand(n_shots, n_per_shot, ndim, dtype=dtype) * 10 + 5
    hicks_off = Hicks(locations, dtype=dtype, offset=offset)
    hicks_ref = Hicks(locations - torch.tensor(offset), dtype=dtype)
    assert torch.equal(hicks_off.get_locations(), hicks_ref.get_locations())
    amplitudes = torch.randn(n_shots, n_per_shot, 5, dtype=dtype)
    assert torch.allclose(hicks_off.source(amplitudes), hicks_ref.source(amplitudes))


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_offset_on_node_snaps(ndim, dtype=torch.double):
    """A point at node+offset with that offset must snap to a single cell."""
    offset = [0.5] * ndim
    node = [7] * ndim
    locations = torch.tensor([[[float(n) + 0.5 for n in node]]], dtype=dtype)
    hicks = Hicks(locations, dtype=dtype, offset=offset)
    loc = hicks.get_locations()
    assert loc.shape == (1, 1, ndim)
    assert torch.equal(loc[0, 0], torch.tensor(node, dtype=torch.long))


def test_free_surface_through_class(dtype=torch.double):
    """A point near a top free surface gives finite, in-model interpolation.

    The free-surface imaging in _get_hicks_for_one_location_dim folds the part
    of the window above the surface back into the model (with a sign flip). This
    checks that behaviour end-to-end through the Hicks class in 2-D: no NaNs,
    and every interpolated cell lies on the model side of the surface.
    """
    ny, nx = 30, 30
    # Source 1.5 cells below a free surface at the start of dim 0 (default
    # free-surface plane is one cell outside the model edge, loc -1).
    locations = torch.tensor([[[1.5, 15.0]]], dtype=dtype)
    hicks = Hicks(
        locations,
        dtype=dtype,
        halfwidth=4,
        free_surfaces=[True, False, False, False],
        model_shape=[ny, nx],
    )
    spread = hicks.source(torch.ones(1, 1, 1, dtype=dtype))
    assert torch.isfinite(spread).all()
    # Every active (non-ignored) interpolated cell is within the model in dim 0.
    locs = hicks.get_locations()[0]
    valid = locs[:, 0] != deepwave.IGNORE_LOCATION
    assert (locs[valid, 0] >= 0).all()
    assert (locs[valid, 0] < ny).all()


def test_free_surface_with_offset_stays_in_model(dtype=torch.double):
    """A staggered component offset must never produce an out-of-model index.

    Combining a non-zero ``offset`` with a free surface shifts the surface plane
    in the component's index space, which can place it on or beyond the model
    edge. The interpolator must still only emit valid in-model cells (formerly a
    bug that retained a negative index). Physical exactness of the imaging for a
    staggered component is a separate, documented limitation (the symmetry is
    component dependent); this only checks the safety/index property.
    """
    ny, nx = 30, 30
    for offset0 in (0.5, 0.25):
        locations = torch.tensor([[[2.3, 15.0]]], dtype=dtype)
        hicks = Hicks(
            locations,
            dtype=dtype,
            halfwidth=4,
            offset=[offset0, 0.0],
            free_surfaces=[True, False, False, False],
            model_shape=[ny, nx],
        )
        spread = hicks.source(torch.ones(1, 1, 1, dtype=dtype))
        assert torch.isfinite(spread).all()
        locs = hicks.get_locations()[0]
        valid = locs[:, 0] != deepwave.IGNORE_LOCATION
        assert (locs[valid, 0] >= 0).all()
        assert (locs[valid, 0] < ny).all()


@pytest.mark.parametrize("offset", [0.5, 0.25])
def test_offset_near_origin_no_nan(offset, dtype=torch.double):
    """A near-origin point whose component coord is negative must not be NaN.

    With a staggered component offset, a physical point within ``offset`` of the
    origin maps to a negative coordinate in the component's index space. The
    window centring must use floor() so the Kaiser argument stays in range
    (otherwise negative coordinates produce NaN weights).
    """
    locations = torch.tensor([[[0.3]]], dtype=dtype)  # comp coord 0.3 - offset < 0
    hicks = Hicks(locations, dtype=dtype, offset=[offset], halfwidth=8)
    spread = hicks.source(torch.ones(1, 1, 1, dtype=dtype))
    assert torch.isfinite(spread).all()
    # Weights still form an (approximate) partition of unity, no NaNs.
    assert spread.sum().item() == pytest.approx(1.0, abs=5e-3)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_partition_of_unity(ndim, dtype=torch.double):
    """Spreading an off-grid monopole must conserve total amplitude.

    The Kaiser-windowed sinc weights form an (approximate) partition of unity,
    so the sum of the spread amplitudes equals the original amplitude.
    """
    torch.manual_seed(2)
    locations = torch.rand(1, 1, ndim, dtype=dtype) * 6 + 10
    hicks = Hicks(locations, dtype=dtype, halfwidth=8)
    amplitudes = torch.ones(1, 1, 1, dtype=dtype)
    spread = hicks.source(amplitudes)
    assert spread.sum().item() == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Physical boundary-imaging tests (method of images vs the real propagators).
#
# An off-grid source/receiver placed near a reflecting surface must, after
# interpolation, reproduce the field of a true point source in the half space.
# By the method of images that equals a full-space field with an added image
# point (sign -1 for a free surface, +1 for a rigid surface) at the mirror
# position. These tests confirm that the boundary-aware Hicks placement matches
# each Deepwave propagator's actual reflecting boundary:
#
#   - scalar  pml_width=0 face : FREE  surface, odd mirror at index -1 / n
#   - acoustic pml_width=0 face: RIGID surface, even mirror at -0.5 / n-0.5
#
# (both being the per-face Hicks defaults). The "zero-PML ghost" closure is an
# exact image only for a 2nd-order stencil, so accuracy=2 is used to obtain a
# machine-precision (rather than ~1%) match. Reciprocity is structural and is
# checked at the default accuracy.
# ---------------------------------------------------------------------------

_DX, _DT, _FREQ = 5.0, 0.0005, 25.0


def _ricker(nt, dtype=torch.double):
    return deepwave.wavelets.ricker(_FREQ, nt, _DT, 1.5 / _FREQ, dtype=dtype).reshape(
        1,
        1,
        -1,
    )


def _scalar_rec(nx, src_locs, src_amps, recv_locs, pml, acc):
    out = deepwave.scalar(
        torch.full((nx,), 1500.0, dtype=torch.double),
        _DX,
        _DT,
        source_amplitudes=src_amps,
        source_locations=src_locs,
        receiver_locations=recv_locs,
        pml_width=pml,
        pml_freq=_FREQ,
        accuracy=acc,
    )
    return out[-1]


def _acoustic_rec(nx, src_locs, src_amps, recv_locs, pml, acc):
    out = deepwave.acoustic(
        torch.full((nx,), 1500.0, dtype=torch.double),
        torch.full((nx,), 1000.0, dtype=torch.double),
        _DX,
        _DT,
        source_amplitudes_volume_rate=src_amps,
        source_locations_volume_rate=src_locs,
        receiver_locations_p=recv_locs,
        pml_width=pml,
        pml_freq=_FREQ,
        accuracy=acc,
    )
    return out.receiver_amplitudes_p


@pytest.mark.parametrize("side", ["begin", "end"])
@pytest.mark.parametrize(
    "kind",
    ["scalar_free", "acoustic_rigid"],
)
def test_source_imaging_matches_propagator_boundary(kind, side, nx=70, nt=220):
    """An off-grid source imaged near a surface equals an explicit image source.

    The half space is the propagator with a ``pml_width=0`` (reflecting) face and
    the source placed by a boundary-aware ``Hicks``. The reference is the same
    source plus its method-of-images twin in a fully absorbing domain. They must
    agree to machine precision, which simultaneously validates the reflection
    sign, the (default) mirror-plane location, and the folding arithmetic.
    """
    acc = 2
    run = _scalar_rec if kind == "scalar_free" else _acoustic_rec
    free = kind == "scalar_free"
    # mirror plane (regular-grid default) and image sign for this propagator
    if side == "begin":
        plane = -1.0 if free else -0.5
        x = 2.3
        pml = [0, 40]
    else:
        plane = float(nx) if free else nx - 0.5
        x = nx - 1 - 2.3
        pml = [40, 0]
    sign = -1.0 if free else 1.0

    surf = [False, False]
    surf[0 if side == "begin" else 1] = True
    hs = Hicks(
        torch.tensor([[[x]]], dtype=torch.double),
        model_shape=[nx],
        free_surfaces=surf if free else None,
        rigid_surfaces=None if free else surf,
        dtype=torch.double,
    )
    amp = _ricker(nt)
    half = run(
        nx,
        hs.get_locations(),
        hs.source(amp),
        torch.arange(nx).reshape(1, nx, 1),
        pml,
        acc,
    )[0]

    # Full-space reference: real source at x and image at 2*plane-x.
    pad = 60
    nxf = nx + pad + 40
    hf = Hicks(
        torch.tensor([[[x + pad], [2 * plane - x + pad]]], dtype=torch.double),
        dtype=torch.double,
    )
    amps = torch.cat([amp, sign * amp], dim=1)
    full = run(
        nxf,
        hf.get_locations(),
        hf.source(amps),
        torch.arange(nxf).reshape(1, nxf, 1),
        [40, 40],
        acc,
    )[0]

    # Compare an interior window adjacent to the reflecting face.
    if side == "begin":
        a, b = half[:40], full[pad : pad + 40]
    else:
        a, b = half[nx - 40 :], full[pad + nx - 40 : pad + nx]
    assert (a - b).norm() / a.norm() < 1e-10


@pytest.mark.parametrize(
    "kind",
    ["scalar_free", "acoustic_rigid"],
)
def test_surface_imaging_reciprocity(kind, nx=70, nt=300, x=2.3, on_grid=40):
    """Boundary-aware off-grid placement must preserve source-receiver reciprocity.

    Swapping an on-grid interior point and an off-grid point imaged against a
    reflecting surface leaves the recorded trace unchanged. Combined with the
    source-imaging test (which pins the source spread to the true half-space
    field), this confirms the receiver gather is correct, since the gather uses
    the same weights as the spread.
    """
    acc = 4
    run = _scalar_rec if kind == "scalar_free" else _acoustic_rec
    free = kind == "scalar_free"
    surf = [True, False]
    hx = Hicks(
        torch.tensor([[[x]]], dtype=torch.double),
        model_shape=[nx],
        free_surfaces=surf if free else None,
        rigid_surfaces=None if free else surf,
        dtype=torch.double,
    )
    amp = _ricker(nt)
    on_grid_loc = torch.tensor([[[on_grid]]])
    # source on-grid, receiver off-grid (imaged)
    t1 = hx.receiver(
        run(nx, on_grid_loc, amp, hx.get_locations(), [0, 40], acc),
    )[0, 0]
    # source off-grid (imaged), receiver on-grid
    t2 = run(nx, hx.get_locations(), hx.source(amp), on_grid_loc, [0, 40], acc)[0, 0]
    assert (t1 - t2).norm() / t1.norm() < 1e-9


def test_reflecting_edge_is_exact_image_only_at_low_order(nx=70, nt=220, x=2.3):
    """The propagator's reflecting edge is an exact image only at ``accuracy=2``.

    The Hicks folding uses the *exact* method-of-images reflection, but the
    propagator realises its reflecting edge by holding the exterior ghost cells to
    zero. At ``accuracy=2`` only the single nearest ghost is read, and zeroing it
    *is* the exact image (free for scalar, rigid for acoustic) — so the half-space
    run matches the source+image construction to machine precision. At higher
    accuracy the stencil also reads deeper ghosts, which are zeroed rather than
    image-extended, so the edge is only an approximate image and carries a
    spurious reflection error of order several percent that does **not** decrease
    with interior accuracy order. This test documents and guards that behaviour
    (it is a property of the propagator boundary, not of the Hicks imaging).
    """
    hs = Hicks(
        torch.tensor([[[x]]], dtype=torch.double),
        model_shape=[nx],
        free_surfaces=[True, False],
        dtype=torch.double,
    )
    amp = _ricker(nt)
    pad, plane, sign = 60, -1.0, -1.0
    nxf = nx + pad + 40
    hf = Hicks(
        torch.tensor([[[x + pad], [2 * plane - x + pad]]], dtype=torch.double),
        dtype=torch.double,
    )
    amps = torch.cat([amp, sign * amp], dim=1)

    def residual(acc):
        half = _scalar_rec(
            nx,
            hs.get_locations(),
            hs.source(amp),
            torch.arange(nx).reshape(1, nx, 1),
            [0, 40],
            acc,
        )[0]
        full = _scalar_rec(
            nxf,
            hf.get_locations(),
            hf.source(amps),
            torch.arange(nxf).reshape(1, nxf, 1),
            [40, 40],
            acc,
        )[0]
        a, b = half[:40], full[pad : pad + 40]
        return ((a - b).norm() / a.norm()).item()

    assert residual(2) < 1e-10  # exact image at 2nd order
    assert residual(8) > 1e-2  # not an exact image at higher order


def _scalar_setup(
    nx=(30, 30),
    dx=(5.0, 5.0),
    c=1500,
    freq=25,
    nt=80,
    dtype=torch.double,
):
    """Build a constant model and a Ricker wavelet for float-coord tests."""
    model = torch.ones(*nx, dtype=dtype) * c
    wavelet = deepwave.wavelets.ricker(freq, nt, 0.001, 1.5 / freq, dtype=dtype)
    return model, list(dx), 0.001, wavelet


def test_float_coords_match_manual_hicks():
    """scalar() with float coords must match the manual Hicks workflow exactly."""
    model, dx, dt, wavelet = _scalar_setup()
    src = torch.tensor([[[10.3, 12.7], [18.6, 9.2]]], dtype=torch.double)
    rec = torch.tensor([[[20.4, 11.1], [22.8, 19.5], [8.7, 14.2]]], dtype=torch.double)
    amp = wavelet.reshape(1, 1, -1).repeat(1, src.shape[1], 1)

    hicks_s = Hicks(src, dtype=torch.double)
    hicks_r = Hicks(rec, dtype=torch.double)
    manual = deepwave.scalar(
        model,
        dx,
        dt,
        source_amplitudes=hicks_s.source(amp),
        source_locations=hicks_s.get_locations(),
        receiver_locations=hicks_r.get_locations(),
    )
    manual_rec = hicks_r.receiver(manual[-1])

    auto = deepwave.scalar(
        model,
        dx,
        dt,
        source_amplitudes=amp,
        source_locations=src,
        receiver_locations=rec,
    )
    assert torch.allclose(auto[-1], manual_rec)


def test_should_interpolate_offset_aware():
    """On-node detection must be made in the component's index space.

    For a staggered velocity component (offset 0.5) the nodes are at half-integer
    physical positions, so a whole-number coordinate is off-node (interpolate),
    while a half-integer coordinate is on-node (exact).
    """
    half = torch.tensor([[[7.5, 3.0]]], dtype=torch.double)  # on vy node in dim 0
    whole = torch.tensor([[[7.0, 3.0]]], dtype=torch.double)  # off vy node in dim 0
    # Regular grid (no offset): whole numbers are on-node, half-integers are not.
    assert _should_interpolate(half, "auto") is True
    assert _should_interpolate(whole, "auto") is False
    # vy component (offset 0.5 in dim 0): the roles flip in dim 0.
    assert _should_interpolate(half, "auto", offset=[0.5, 0.0]) is False
    assert _should_interpolate(whole, "auto", offset=[0.5, 0.0]) is True
    # An all-ignored point must never force interpolation, even with an offset.
    ignore = float(deepwave.IGNORE_LOCATION)
    ignored = torch.tensor([[[ignore, ignore]]], dtype=torch.double)
    assert _should_interpolate(ignored, "auto", offset=[0.5, 0.0]) is False


def test_grid_locations_offset_snap():
    """A non-interpolated float must snap to the component's array index.

    A float on the vy node (physical N+0.5 in dim 0) must snap to array index N,
    not to round(physical) = N (regular grid) shifted by the stagger.
    """
    locs = torch.tensor([[[7.5, 3.0]]], dtype=torch.double)  # on vy node in dim 0
    # offset 0.5 in dim 0: array index is round(7.5 - 0.5) = 7 (not 8).
    snapped = _grid_locations(locs, None, offset=[0.5, 0.0])
    assert torch.equal(snapped[0, 0], torch.tensor([7, 3]))
    # Regular grid (offset None): plain round.
    assert torch.equal(
        _grid_locations(locs, None)[0, 0],
        torch.tensor([8, 3]),
    )
    # Integer/long tensors pass through unchanged.
    long_locs = torch.tensor([[[7, 3]]])
    assert _grid_locations(long_locs, None, offset=[0.5, 0.0]) is long_locs


def test_resolve_interpolator_none_off_node_raises():
    """mode='none' raises for an off-node float instead of silently rounding.

    Decision: ``location_interpolation="none"`` is the exact integer path only;
    an off-node float coordinate that used to be silently rounded to the nearest
    cell is now a ValueError. Integer/on-node inputs are unaffected.
    """
    off = torch.tensor([[[5.3, 7.0]]], dtype=torch.double)
    with pytest.raises(ValueError, match="location_interpolation='none'"):
        _resolve_interpolator(off, mode="none")
    # A staggered component (offset 0.5 in dim 0): a whole number is off its node.
    with pytest.raises(ValueError, match="location_interpolation='none'"):
        _resolve_interpolator(off, mode="none", offset=[0.5, 0.0])
    # Other modes are unaffected: 'auto' interpolates the off-node float.
    assert _resolve_interpolator(off, mode="auto") is not None


def test_resolve_interpolator_none_on_node_ok_and_bit_identical():
    """mode='none' accepts an exactly-on-node float; result matches int input."""
    on_node = torch.tensor([[[5.0, 7.0]]], dtype=torch.double)
    interp = _resolve_interpolator(on_node, mode="none")
    # No interpolation for an on-node point in 'none' mode.
    assert interp is None
    gl_float = _grid_locations(on_node, interp)
    gl_int = _grid_locations(torch.tensor([[[5, 7]]]), None)
    assert torch.equal(gl_float, gl_int)
    assert gl_float.dtype == torch.long
    # Integer input in 'none' mode is passed straight through (no error).
    int_locs = torch.tensor([[[5, 7]]])
    assert _resolve_interpolator(int_locs, mode="none") is None
    # An all-ignored point is never off-node, so 'none' accepts it.
    ignore = float(deepwave.IGNORE_LOCATION)
    ignored = torch.tensor([[[ignore, ignore]]], dtype=torch.double)
    assert _resolve_interpolator(ignored, mode="none") is None


def test_propagator_accepts_hicks_object_dipole():
    """Passing a Hicks object to the propagator enables dipoles and matches manual.

    A Hicks instance given as a location argument is used directly by the
    propagator (which performs the spread/gather), so it must equal the manual
    workflow, and a dipole specification must differ from a monopole.
    """
    model, dx, dt, wavelet = _scalar_setup()
    src = torch.tensor([[[10.3, 12.7]]], dtype=torch.double)
    rec = torch.tensor([[[20.4, 11.1]]], dtype=torch.double)
    amp = wavelet.reshape(1, 1, -1)
    hicks_s = Hicks(src, dtype=torch.double, monopole=False, dipole_dim=0)
    hicks_r = Hicks(rec, dtype=torch.double, monopole=False, dipole_dim=0)

    manual = deepwave.scalar(
        model,
        dx,
        dt,
        source_amplitudes=hicks_s.source(amp),
        source_locations=hicks_s.get_locations(),
        receiver_locations=hicks_r.get_locations(),
    )
    manual_rec = hicks_r.receiver(manual[-1])

    auto = deepwave.scalar(
        model,
        dx,
        dt,
        source_amplitudes=amp,
        source_locations=hicks_s,
        receiver_locations=hicks_r,
    )
    assert torch.allclose(auto[-1], manual_rec)

    # A dipole source/receiver differs from the monopole result.
    mono = deepwave.scalar(
        model,
        dx,
        dt,
        source_amplitudes=amp,
        source_locations=src,
        receiver_locations=rec,
    )
    assert not torch.allclose(auto[-1], mono[-1])


def test_float_coords_on_grid_snap_to_integer():
    """Integer-valued float coords must reproduce the integer-location result."""
    model, dx, dt, wavelet = _scalar_setup()
    src_long = torch.tensor([[[10, 12]]])
    rec_long = torch.tensor([[[20, 11], [8, 14]]])
    amp = wavelet.reshape(1, 1, -1)

    baseline = deepwave.scalar(
        model,
        dx,
        dt,
        source_amplitudes=amp,
        source_locations=src_long,
        receiver_locations=rec_long,
    )
    snapped = deepwave.scalar(
        model,
        dx,
        dt,
        source_amplitudes=amp,
        source_locations=src_long.double(),
        receiver_locations=rec_long.double(),
    )
    # On-grid float coords take the exact path: bit-for-bit identical.
    assert torch.equal(baseline[-1], snapped[-1])


def test_float_coords_gradients_flow():
    """Gradients must flow through interpolated source amplitudes and the model."""
    model, dx, dt, wavelet = _scalar_setup(nt=60)
    model = model.requires_grad_(True)
    src = torch.tensor([[[10.3, 12.7]]], dtype=torch.double)
    rec = torch.tensor([[[20.4, 11.1]]], dtype=torch.double)
    amp = wavelet.reshape(1, 1, -1).requires_grad_(True)
    out = deepwave.scalar(
        model,
        dx,
        dt,
        source_amplitudes=amp,
        source_locations=src,
        receiver_locations=rec,
    )
    out[-1].abs().sum().backward()
    assert model.grad is not None
    assert model.grad.abs().sum() > 0
    assert amp.grad is not None
    assert amp.grad.abs().sum() > 0


def test_float_coords_ignore_location():
    """An interpolated point flagged IGNORE_LOCATION must be disabled."""
    model, dx, dt, wavelet = _scalar_setup()
    ignore = float(deepwave.IGNORE_LOCATION)
    src = torch.tensor([[[10.3, 12.7]]], dtype=torch.double)
    rec_with = torch.tensor(
        [[[20.4, 11.1], [22.8, 19.5]]],
        dtype=torch.double,
    )
    rec_ignored = torch.tensor(
        [[[20.4, 11.1], [ignore, ignore]]],
        dtype=torch.double,
    )
    amp = wavelet.reshape(1, 1, -1)
    out_with = deepwave.scalar(
        model,
        dx,
        dt,
        source_amplitudes=amp,
        source_locations=src,
        receiver_locations=rec_with,
    )
    out_ignored = deepwave.scalar(
        model,
        dx,
        dt,
        source_amplitudes=amp,
        source_locations=src,
        receiver_locations=rec_ignored,
    )
    # First receiver unchanged; the ignored one records exactly zero.
    assert torch.allclose(out_with[-1][:, 0], out_ignored[-1][:, 0])
    assert torch.count_nonzero(out_ignored[-1][:, 1]) == 0


def test_location_interpolation_modes():
    """mode='none' rejects off-grid coords; mode='hicks' forces interpolation."""
    model, dx, dt, wavelet = _scalar_setup()
    src = torch.tensor([[[10.3, 12.7]]], dtype=torch.double)
    rec = torch.tensor([[[20.4, 11.1]]], dtype=torch.double)
    amp = wavelet.reshape(1, 1, -1)
    # mode='none' now raises for an off-node float coordinate (decision: the
    # mode means the exact integer path only; it no longer silently rounds).
    with pytest.raises(ValueError, match="location_interpolation='none'"):
        deepwave.scalar(
            model,
            dx,
            dt,
            source_amplitudes=amp,
            source_locations=src,
            receiver_locations=rec,
            location_interpolation="none",
        )
    # An exactly-on-node float in mode='none' is accepted and is bit-identical
    # to passing the integer cells directly.
    src_on = torch.tensor([[[10.0, 12.0]]], dtype=torch.double)
    rec_on = torch.tensor([[[20.0, 11.0]]], dtype=torch.double)
    none_on = deepwave.scalar(
        model,
        dx,
        dt,
        source_amplitudes=amp,
        source_locations=src_on,
        receiver_locations=rec_on,
        location_interpolation="none",
    )
    int_out = deepwave.scalar(
        model,
        dx,
        dt,
        source_amplitudes=amp,
        source_locations=src_on.long(),
        receiver_locations=rec_on.long(),
    )
    assert torch.equal(none_on[-1], int_out[-1])
    # mode='hicks' on integer coords reduces to the exact result (unit weights).
    hicks_out = deepwave.scalar(
        model,
        dx,
        dt,
        source_amplitudes=amp,
        source_locations=torch.tensor([[[10, 12]]]).double(),
        receiver_locations=torch.tensor([[[20, 11]]]).double(),
        location_interpolation="hicks",
    )
    exact = deepwave.scalar(
        model,
        dx,
        dt,
        source_amplitudes=amp,
        source_locations=torch.tensor([[[10, 12]]]),
        receiver_locations=torch.tensor([[[20, 11]]]),
    )
    assert torch.allclose(hicks_out[-1], exact[-1])


def test_shot_idxs(n_shots=10, n_per_shot=3, nt=5, device=None, dtype=torch.double):
    """Test that Hicks handles shot_idxs correctly."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    locations = torch.rand(n_shots, n_per_shot, 2, dtype=dtype, device=device) * 10
    amplitudes = torch.randn(n_shots, n_per_shot, nt, dtype=dtype, device=device)
    hicks = Hicks(locations, dtype=dtype, model_shape=[100, 100])
    hicks_amplitudes_fwd = hicks.source(amplitudes)
    shot_idxs = torch.tensor([n_shots // 2])
    hicks_amplitudes_half = hicks.source(amplitudes[shot_idxs], shot_idxs)
    assert torch.allclose(hicks_amplitudes_fwd[shot_idxs], hicks_amplitudes_half)
    shot_idxs = torch.arange(n_shots).flip(0)
    hicks_amplitudes_bwd = hicks.source(amplitudes.flip(0), shot_idxs)
    assert torch.allclose(hicks_amplitudes_fwd.flip(0), hicks_amplitudes_bwd)


def test_hicks_source_invalid_amplitudes_type() -> None:
    """Test that Hicks.source raises TypeError if amplitudes is not a tensor."""
    locations = torch.randn(1, 1, 2)
    hicks = Hicks(locations, model_shape=[100, 100])
    with pytest.raises(TypeError, match=r"amplitudes must be a torch.Tensor."):
        hicks.source([1, 2, 3])


def test_hicks_source_invalid_amplitudes_ndim() -> None:
    """Test Hicks.source raises if amplitudes.ndim != 3."""
    locations = torch.randn(1, 1, 2)
    hicks = Hicks(locations, model_shape=[100, 100])
    amplitudes = torch.randn(10)  # Should be 3D
    with pytest.raises(RuntimeError, match=r"amplitudes must have three dimensions."):
        hicks.source(amplitudes)


def test_hicks_receiver_invalid_amplitudes_type() -> None:
    """Test that Hicks.receiver raises TypeError if amplitudes is not a tensor."""
    locations = torch.randn(1, 1, 2)
    hicks = Hicks(locations, model_shape=[100, 100])
    with pytest.raises(TypeError, match=r"amplitudes must be a torch.Tensor."):
        hicks.receiver([1, 2, 3])


def test_hicks_receiver_invalid_amplitudes_ndim() -> None:
    """Test Hicks.receiver raises if amplitudes.ndim != 3."""
    locations = torch.randn(1, 1, 2)
    hicks = Hicks(locations, model_shape=[100, 100])
    amplitudes = torch.randn(10)  # Should be 3D
    with pytest.raises(RuntimeError, match=r"amplitudes must have three dimensions."):
        hicks.receiver(amplitudes)


def test_free_surface_left(halfwidth=8):
    """Test _get_hicks_for_one_location_dim with free surface on the left."""
    betas = [0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64]
    beta = torch.tensor(betas[halfwidth - 1])
    l0, w0 = _get_hicks_for_one_location_dim(
        {},
        0.5,
        halfwidth,
        beta,
        [False, False],
        [-0.5, 19.5],
        20,
        True,
    )
    l1, w1 = _get_hicks_for_one_location_dim(
        {},
        0.5,
        halfwidth,
        beta,
        [True, False],
        [-0.5, 19.5],
        20,
        True,
    )
    assert len(l0) == 2 * halfwidth
    assert l0[0] == -halfwidth + 1
    assert len(l1) == halfwidth + 1
    assert l1[0] == 0
    assert w0[-1] == w1[-1]
    assert w0[-2] == w1[-2]
    for i in range(halfwidth - 1):
        assert w1[i] == pytest.approx(w0[halfwidth - 1 + i] - w0[halfwidth - 2 - i])


def test_free_surface_right(halfwidth=8):
    """Test _get_hicks_for_one_location_dim with free surface on the right."""
    betas = [0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64]
    beta = torch.tensor(betas[halfwidth - 1])
    l0, w0 = _get_hicks_for_one_location_dim(
        {},
        18.5,
        halfwidth,
        beta,
        [False, False],
        [-0.5, 19.5],
        20,
        True,
    )
    l1, w1 = _get_hicks_for_one_location_dim(
        {},
        18.5,
        halfwidth,
        beta,
        [False, True],
        [-0.5, 19.5],
        20,
        True,
    )
    assert len(l0) == 2 * halfwidth
    assert l0[-1] == 19 + halfwidth - 1
    assert len(l1) == halfwidth + 1
    assert l1[-1] == 19
    assert w0[0] == w1[0]
    assert w0[1] == w1[1]
    for i in range(halfwidth - 1):
        assert w1[len(w1) - 1 - i] == pytest.approx(
            w0[len(w0) - 1 - (halfwidth - 1 + i)]
            - w0[len(w0) - 1 - (halfwidth - 2 - i)],
        )


def _images_reference(l0, w0, plane, sign, size, begin):
    """Independent method-of-images reference for the surface folding.

    Given the no-surface stencil (``l0``, ``w0``), fold every cell on the
    exterior side of ``plane`` back into the interior: an interior cell keeps
    its weight; an exterior cell ``c`` contributes ``sign * w`` to its mirror
    ``2 * plane - c``; a cell exactly on ``plane`` contributes ``(1 + sign)``
    times its weight to itself (zero for a free surface, double for a rigid
    one). For a ``begin`` surface the interior is the side with larger indices;
    for an end surface it is the smaller-index side. Cells outside ``[0, size)``
    are dropped. Returns ascending (locations, weights), matching the function
    under test.
    """
    acc = {}
    for c, w in zip(l0.tolist(), w0.tolist()):
        interior = c > plane if begin else c < plane
        exterior = c < plane if begin else c > plane
        if interior:
            acc[c] = acc.get(c, 0.0) + w
        elif exterior:
            mirror = round(2 * plane - c)
            acc[mirror] = acc.get(mirror, 0.0) + sign * w
        elif 1.0 + sign != 0.0:  # on the plane: rigid doubles, free zeros (drop)
            acc[c] = acc.get(c, 0.0) + (1.0 + sign) * w
    cells = sorted(c for c in acc if 0 <= c < size)
    return cells, [acc[c] for c in cells]


@pytest.mark.parametrize("side", ["begin", "end"])
@pytest.mark.parametrize("surface", ["free", "rigid"])
@pytest.mark.parametrize("plane_on_node", [True, False])
def test_surface_folding_matches_method_of_images(
    side,
    surface,
    plane_on_node,
    halfwidth=8,
    size=20,
):
    """The free/rigid folding must equal an explicit method-of-images sum.

    This validates the arithmetic of both reflection types (odd/free and
    even/rigid), for a surface lying exactly on a grid node and between nodes,
    at the beginning and end of the dimension, including that an on-node free
    surface drops the (zeroed) surface point while an on-node rigid surface
    keeps (and doubles) it.
    """
    betas = [0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64]
    beta = torch.tensor(betas[halfwidth - 1])
    # The location is deliberately off-node (fractional) so a real
    # interpolation window is built and pokes past the nearby surface.
    if side == "begin":
        plane = 1.0 if plane_on_node else 0.5
        location = plane + 0.7
        free = [surface == "free", False]
        rigid = [surface == "rigid", False]
    else:
        plane = float(size - 2) if plane_on_node else size - 2.5
        location = plane - 0.7
        free = [False, surface == "free"]
        rigid = [False, surface == "rigid"]
    locs = [-0.5, size - 0.5]
    locs[0 if side == "begin" else 1] = plane
    l0, w0 = _get_hicks_for_one_location_dim(
        {},
        location,
        halfwidth,
        beta,
        [False, False],
        locs,
        size,
        True,
    )
    l1, w1 = _get_hicks_for_one_location_dim(
        {},
        location,
        halfwidth,
        beta,
        free,
        locs,
        size,
        True,
        rigid_surface=rigid,
    )
    sign = -1.0 if surface == "free" else 1.0
    exp_cells, exp_weights = _images_reference(
        l0,
        w0,
        plane,
        sign,
        size,
        side == "begin",
    )
    assert l1.tolist() == exp_cells
    assert w1.tolist() == pytest.approx(exp_weights)
    # Every emitted cell is a valid in-model index.
    assert all(0 <= c < size for c in l1.tolist())


@pytest.mark.parametrize("side", ["begin", "end"])
@pytest.mark.parametrize("surface", ["free", "rigid"])
def test_on_node_monopole_on_surface_plane(side, surface, halfwidth=4, size=40):
    """A monopole exactly on an integer surface-plane node must image, not pass.

    Regression: the on-node fast path used to short-circuit to a single
    unit-weight cell, bypassing the surface fold. A free surface must zero (drop)
    a point sitting on it; a rigid surface must double it.
    """
    betas = [0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64]
    beta = torch.tensor(betas[halfwidth - 1])
    plane = 10.0  # an interior integer node, on the surface
    free = [surface == "free", False] if side == "begin" else [False, surface == "free"]
    rigid = (
        [surface == "rigid", False] if side == "begin" else [False, surface == "rigid"]
    )
    fsl = [-0.5, size - 0.5]
    fsl[0 if side == "begin" else 1] = plane
    locs, weights = _get_hicks_for_one_location_dim(
        {},
        plane,
        halfwidth,
        beta,
        free,
        fsl,
        size,
        True,
        rigid_surface=rigid,
    )
    on_plane = (locs == int(plane)).nonzero().flatten()
    if surface == "free":
        assert on_plane.numel() == 0  # zeroed and dropped
    else:
        assert on_plane.numel() == 1  # doubled (real point coincides with image)
        assert weights[on_plane].item() == pytest.approx(2.0)


@pytest.mark.parametrize("side", ["begin", "end"])
@pytest.mark.parametrize("surface", ["free", "rigid"])
def test_window_edge_on_surface_plane_is_folded(side, surface, halfwidth=3, size=20):
    """A window EDGE landing exactly on an interior surface plane must fold.

    Distinct from test_on_node_monopole_on_surface_plane above: here the
    *location* is off-plane (4.2, not on an integer node), but with this
    halfwidth its stencil window's extreme cell lands exactly on the
    integer surface plane (window [2..7] against a plane at 2, for the
    "begin" side). Regression: the fold guard used to be a strict `<`/`>`
    comparison, which skipped folding entirely whenever the window edge
    was exactly ON the plane (as opposed to strictly past it), silently
    leaving that cell's full, un-imaged weight in place instead of
    zeroing (free) or doubling (rigid) it.
    """
    betas = [0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64]
    beta = torch.tensor(betas[halfwidth - 1])
    # Window is [floor(location) - halfwidth + 1, ..., floor(location) + halfwidth].
    # Pick location so that extreme cell lands exactly on `plane` (matching the
    # reported repro: plane=2, halfwidth=3 -> location=4.2, window=[2..7]).
    plane = 2.0
    location = plane + halfwidth - 1 + 0.2  # locations[0] == plane
    if side == "end":
        plane = size - 1 - 2.0
        location = plane - halfwidth + 0.2  # locations[-1] == plane
    free = [surface == "free", False] if side == "begin" else [False, surface == "free"]
    rigid = (
        [surface == "rigid", False] if side == "begin" else [False, surface == "rigid"]
    )
    fsl = [-0.5, size - 0.5]
    fsl[0 if side == "begin" else 1] = plane

    locs, weights = _get_hicks_for_one_location_dim(
        {},
        location,
        halfwidth,
        beta,
        free,
        fsl,
        size,
        True,
        rigid_surface=rigid,
    )
    # Baseline (no surface) weight at the on-plane cell, for comparison.
    locs_none, weights_none = _get_hicks_for_one_location_dim(
        {},
        location,
        halfwidth,
        beta,
        [False, False],
        [-0.5, size - 0.5],
        size,
        True,
        rigid_surface=[False, False],
    )
    on_plane_none = (locs_none == int(plane)).nonzero().flatten()
    assert on_plane_none.numel() == 1
    baseline_weight = weights_none[on_plane_none].item()

    on_plane = (locs == int(plane)).nonzero().flatten()
    if surface == "free":
        assert on_plane.numel() == 0  # zeroed and dropped
    else:
        assert on_plane.numel() == 1  # doubled
        assert weights[on_plane].item() == pytest.approx(2 * baseline_weight)


@pytest.mark.parametrize("side", ["begin", "end"])
@pytest.mark.parametrize("surface", ["free", "rigid"])
@pytest.mark.parametrize("offset", [0.0, 0.25, 0.5])
def test_surface_interior_plane_is_index_safe(side, surface, offset, halfwidth=4):
    """A surface plane several cells inside the model must not crash or escape.

    Regression: when the mirror plane sits far enough inside the window that some
    images reflect past its far end (reachable via a staggered ``offset`` shift,
    or a custom interior surface location), the in-place fold used to raise a
    tensor-size mismatch. It must instead drop the unrepresentable images and
    emit only valid in-model cells.
    """
    ny, nx = 40, 40
    # Source within a window of the (offset-shifted) edge surface.
    if side == "begin":
        loc = 0.7
        free = [surface == "free", False, False, False]
        rigid = [surface == "rigid", False, False, False]
    else:
        loc = ny - 1 - 0.3
        free = [False, surface == "free", False, False]
        rigid = [False, surface == "rigid", False, False]
    hicks = Hicks(
        torch.tensor([[[loc, 15.0]]], dtype=torch.double),
        dtype=torch.double,
        halfwidth=halfwidth,
        offset=[offset, 0.0],
        free_surfaces=free,
        rigid_surfaces=rigid,
        model_shape=[ny, nx],
    )
    spread = hicks.source(torch.ones(1, 1, 1, dtype=torch.double))
    assert torch.isfinite(spread).all()
    loc0 = hicks.get_locations()[0]
    valid = loc0[:, 0] != deepwave.IGNORE_LOCATION
    assert (loc0[valid, 0] >= 0).all()
    assert (loc0[valid, 0] < ny).all()


def test_get_hicks_for_one_location_dim_invalid_halfwidth() -> None:
    """Test _get_hicks_for_one_location_dim with invalid halfwidth."""
    with pytest.raises(RuntimeError, match=r"halfwidth must be in \[1, 10\]"):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            0,
            torch.tensor(1.0),
            [False, False],
            [-0.5, 19.5],
            20,
        )
    with pytest.raises(RuntimeError, match=r"halfwidth must be in \[1, 10\]"):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            11,
            torch.tensor(1.0),
            [False, False],
            [-0.5, 19.5],
            20,
        )


def test_get_hicks_for_one_location_dim_invalid_beta() -> None:
    """Test _get_hicks_for_one_location_dim with invalid beta."""
    with pytest.raises(RuntimeError, match=r"beta must be non-negative."):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            4,
            torch.tensor(-1.0),
            [False, False],
            [-0.5, 19.5],
            20,
        )


def test_get_hicks_for_one_location_dim_invalid_extent() -> None:
    """Test _get_hicks_for_one_location_dim with invalid extent."""
    with pytest.raises(RuntimeError, match=r"extent must be a list of two floats."):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            4,
            torch.tensor(1.0),
            [False, False],
            [-0.5],
            20,
        )

    with pytest.raises(RuntimeError, match=r"extent must be a list of two floats."):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            4,
            torch.tensor(1.0),
            [False, False],
            [-0.5, 19.5, 20.0],
            20,
        )
    with pytest.raises(RuntimeError, match=r"extent must be a list of two floats."):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            4,
            torch.tensor(1.0),
            [False, False],
            ["a", 19.5],
            20,
        )


def test_get_hicks_for_one_location_dim_invalid_n_grid_points() -> None:
    """Test _get_hicks_for_one_location_dim with invalid n_grid_points."""
    with pytest.raises(RuntimeError, match=r"n_grid_points must be positive."):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            4,
            torch.tensor(1.0),
            [True, False],
            [-0.5, 19.5],
            0,
        )
    with pytest.raises(RuntimeError, match=r"n_grid_points must be positive."):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            4,
            torch.tensor(1.0),
            [True, False],
            [-0.5, 19.5],
            -1,
        )


# ---------------------------------------------------------------------------
# Staggered-grid (acoustic / elastic) and scalar_born float-coordinate wiring.
#
# These exercise the per-component half-cell offset table: pressure / normal
# stresses sit on the integer grid, while a velocity component staggered in
# spatial dimension d has offset 0.5 in d. The binding invariant (see
# docs/dev/location_coordinate_semantics.md) is that integer index N and float
# N+offset denote the same physical point, so a velocity source at integer N
# must equal the same source at float N+0.5.
# ---------------------------------------------------------------------------

from deepwave import vpvsrho_to_lambmubuoyancy  # noqa: E402


def _acoustic_setup(ny=26, nx=26, c=1500.0, rho=1000.0, nt=70, dtype=torch.double):
    v = torch.full((ny, nx), c, dtype=dtype)
    rho_t = torch.full((ny, nx), rho, dtype=dtype)
    wavelet = deepwave.wavelets.ricker(_FREQ, nt, _DT, 1.5 / _FREQ, dtype=dtype)
    return v, rho_t, [_DX, _DX], _DT, wavelet.reshape(1, 1, -1)


def _elastic_setup(ny=26, nx=26, nt=70, dtype=torch.double):
    vp = torch.full((ny, nx), 1500.0, dtype=dtype)
    vs = torch.full((ny, nx), 900.0, dtype=dtype)
    rho = torch.full((ny, nx), 1000.0, dtype=dtype)
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    wavelet = deepwave.wavelets.ricker(_FREQ, nt, _DT, 1.5 / _FREQ, dtype=dtype)
    return lamb, mu, buoyancy, [_DX, _DX], _DT, wavelet.reshape(1, 1, -1)


# The half-cell offset for each velocity component in a 2-D (y, x) grid.
_ACOUSTIC_VEL_2D = {"y": [0.5, 0.0], "x": [0.0, 0.5]}


def test_acoustic_integer_locations_byte_identical():
    """Integer (long) acoustic locations must be bit-for-bit unchanged.

    Guards the promise that wiring in float-coordinate support does not perturb
    the exact-injection path for any component (pressure or velocity).
    """
    v, rho, dx, dt, amp = _acoustic_setup()
    kwargs = dict(  # noqa: C408
        source_amplitudes_volume_rate=amp,
        source_locations_volume_rate=torch.tensor([[[8, 9]]]),
        source_amplitudes_y=amp,
        source_locations_y=torch.tensor([[[10, 7]]]),
        source_amplitudes_x=amp,
        source_locations_x=torch.tensor([[[6, 12]]]),
        receiver_locations_p=torch.tensor([[[14, 15], [11, 9]]]),
        receiver_locations_y=torch.tensor([[[13, 8]]]),
        receiver_locations_x=torch.tensor([[[7, 16]]]),
        pml_width=8,
        pml_freq=_FREQ,
    )
    a = deepwave.acoustic(v, rho, dx, dt, **kwargs)
    b = deepwave.acoustic(v, rho, dx, dt, **kwargs)
    for name in (
        "receiver_amplitudes_p",
        "receiver_amplitudes_y",
        "receiver_amplitudes_x",
    ):
        assert torch.equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize("comp", ["p", "y", "x"])
def test_acoustic_on_node_float_equals_integer(comp):
    """A float coord on a component's node equals the integer-index result.

    Pressure nodes are the integer grid (float N.0 == integer N); a velocity
    component staggered in dimension d has its node at N+0.5 there, so float
    N+0.5 must equal integer N. This is the staggered on-node invariant.
    """
    v, rho, dx, dt, amp = _acoustic_setup()
    src_int = torch.tensor([[[10, 11]]])
    rec_int = torch.tensor([[[13, 9], [8, 14]]])
    offset = {"p": [0.0, 0.0], **_ACOUSTIC_VEL_2D}[comp]
    src_float = src_int.double() + torch.tensor(offset)
    rec_float = rec_int.double() + torch.tensor(offset)

    src_key = f"source_amplitudes_{comp}"
    sloc_key = f"source_locations_{comp}"
    rloc_key = f"receiver_locations_{comp}"
    rname = f"receiver_amplitudes_{comp}"

    base = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        pml_width=8,
        pml_freq=_FREQ,
        **{src_key: amp, sloc_key: src_int, rloc_key: rec_int},
    )
    snapped = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        pml_width=8,
        pml_freq=_FREQ,
        **{src_key: amp, sloc_key: src_float, rloc_key: rec_float},
    )
    assert torch.equal(getattr(base, rname), getattr(snapped, rname))


def test_acoustic_float_matches_manual_hicks_velocity():
    """Off-grid velocity source/receiver must match the manual Hicks workflow.

    Confirms the propagator builds the interpolator with the component's
    half-cell offset and performs the same spread/gather as the documented
    manual route.
    """
    v, rho, dx, dt, amp = _acoustic_setup()
    src = torch.tensor([[[10.3, 12.7]]], dtype=torch.double)
    rec = torch.tensor([[[18.4, 9.1], [7.6, 15.2]]], dtype=torch.double)
    offset = _ACOUSTIC_VEL_2D["x"]
    hs = Hicks(src, offset=offset, dtype=torch.double)
    hr = Hicks(rec, offset=offset, dtype=torch.double)

    manual = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        pml_width=8,
        pml_freq=_FREQ,
        source_amplitudes_x=hs.source(amp),
        source_locations_x=hs.get_locations(),
        receiver_locations_x=hr.get_locations(),
    )
    manual_rec = hr.receiver(manual.receiver_amplitudes_x)
    auto = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        pml_width=8,
        pml_freq=_FREQ,
        source_amplitudes_x=amp,
        source_locations_x=src,
        receiver_locations_x=rec,
    )
    assert torch.allclose(auto.receiver_amplitudes_x, manual_rec)


def test_acoustic_offgrid_source_receiver_reciprocity():
    """Genuinely off-grid source and receiver preserve source-receiver reciprocity.

    A pressure source at a fractional position recorded by a pressure receiver
    at another fractional position must equal the swapped configuration, within
    tolerance. This checks the spread and the gather are mutually consistent for
    real (non-snapped) off-grid placement.
    """
    v, rho, dx, dt, amp = _acoustic_setup(ny=40, nx=40, nt=180)
    a = torch.tensor([[[12.3, 15.7]]], dtype=torch.double)
    b = torch.tensor([[[27.6, 22.4]]], dtype=torch.double)
    # source at a, receiver at b
    t1 = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        pml_width=15,
        pml_freq=_FREQ,
        source_amplitudes_volume_rate=amp,
        source_locations_volume_rate=a,
        receiver_locations_p=b,
    ).receiver_amplitudes_p
    # source at b, receiver at a
    t2 = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        pml_width=15,
        pml_freq=_FREQ,
        source_amplitudes_volume_rate=amp,
        source_locations_volume_rate=b,
        receiver_locations_p=a,
    ).receiver_amplitudes_p
    assert (t1 - t2).norm() / t1.norm() < 1e-6


def test_acoustic_float_coords_gradients_flow():
    """Gradients must flow through the spread/gather to source amplitudes and model."""
    v, rho, dx, dt, amp = _acoustic_setup(nt=60)
    v = v.requires_grad_(True)
    amp = amp.clone().requires_grad_(True)
    src = torch.tensor([[[10.3, 12.7]]], dtype=torch.double)
    rec = torch.tensor([[[18.4, 9.1]]], dtype=torch.double)
    out = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        pml_width=8,
        pml_freq=_FREQ,
        source_amplitudes_x=amp,
        source_locations_x=src,
        receiver_locations_x=rec,
    )
    out.receiver_amplitudes_x.abs().sum().backward()
    assert v.grad is not None
    assert v.grad.abs().sum() > 0
    assert amp.grad is not None
    assert amp.grad.abs().sum() > 0


def test_acoustic_float_coords_ignore_location():
    """IGNORE_LOCATION must still disable a point when float coords are used."""
    v, rho, dx, dt, amp = _acoustic_setup()
    ignore = float(deepwave.IGNORE_LOCATION)
    src = torch.tensor([[[10.3, 12.7]]], dtype=torch.double)
    rec_with = torch.tensor([[[18.4, 9.1], [7.6, 15.2]]], dtype=torch.double)
    rec_ignored = torch.tensor([[[18.4, 9.1], [ignore, ignore]]], dtype=torch.double)
    common = dict(pml_width=8, pml_freq=_FREQ)  # noqa: C408
    out_with = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_x=amp,
        source_locations_x=src,
        receiver_locations_x=rec_with,
        **common,
    ).receiver_amplitudes_x
    out_ign = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        source_amplitudes_x=amp,
        source_locations_x=src,
        receiver_locations_x=rec_ignored,
        **common,
    ).receiver_amplitudes_x
    # The ignored receiver records only zeros; the active one is unchanged.
    assert torch.equal(out_ign[:, 1], torch.zeros_like(out_ign[:, 1]))
    assert torch.allclose(out_ign[:, 0], out_with[:, 0])


def test_acoustic_float_coords_out_of_bounds_raises():
    """An off-grid point whose Hicks stencil leaves the model must raise."""
    v, rho, dx, dt, amp = _acoustic_setup(ny=26, nx=26)
    # x within half a window of the far edge: the expanded stencil exceeds nx.
    src = torch.tensor([[[13.0, 24.6]]], dtype=torch.double)
    with pytest.raises(RuntimeError):
        deepwave.acoustic(
            v,
            rho,
            dx,
            dt,
            pml_width=8,
            pml_freq=_FREQ,
            source_amplitudes_volume_rate=amp,
            source_locations_volume_rate=src,
            receiver_locations_p=torch.tensor([[[10, 10]]]),
        )


def test_acoustic_accepts_hicks_object():
    """A user-provided Hicks object is used directly for a velocity component.

    The propagator does not check the offset of a user-provided Hicks (the user
    may deliberately override it), so a Hicks built with an explicit velocity
    offset must reproduce the manual spread/gather.
    """
    v, rho, dx, dt, amp = _acoustic_setup()
    src = torch.tensor([[[10.3, 12.7]]], dtype=torch.double)
    rec = torch.tensor([[[18.4, 9.1]]], dtype=torch.double)
    hs = Hicks(src, offset=_ACOUSTIC_VEL_2D["y"], dtype=torch.double)
    hr = Hicks(rec, offset=_ACOUSTIC_VEL_2D["y"], dtype=torch.double)
    manual = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        pml_width=8,
        pml_freq=_FREQ,
        source_amplitudes_y=hs.source(amp),
        source_locations_y=hs.get_locations(),
        receiver_locations_y=hr.get_locations(),
    )
    manual_rec = hr.receiver(manual.receiver_amplitudes_y)
    auto = deepwave.acoustic(
        v,
        rho,
        dx,
        dt,
        pml_width=8,
        pml_freq=_FREQ,
        source_amplitudes_y=amp,
        source_locations_y=hs,
        receiver_locations_y=hr,
    )
    assert torch.allclose(auto.receiver_amplitudes_y, manual_rec)


def test_elastic_integer_locations_byte_identical():
    """Integer (long) elastic locations must be bit-for-bit unchanged."""
    lamb, mu, buoyancy, dx, dt, amp = _elastic_setup()
    kwargs = dict(  # noqa: C408
        source_amplitudes_pressure_rate=amp,
        source_locations_pressure_rate=torch.tensor([[[8, 9]]]),
        source_amplitudes_y=amp,
        source_locations_y=torch.tensor([[[10, 7]]]),
        source_amplitudes_x=amp,
        source_locations_x=torch.tensor([[[6, 12]]]),
        receiver_locations_p=torch.tensor([[[14, 15]]]),
        receiver_locations_y=torch.tensor([[[13, 8]]]),
        receiver_locations_x=torch.tensor([[[7, 16]]]),
        pml_width=8,
        pml_freq=_FREQ,
    )
    a = deepwave.elastic(lamb, mu, buoyancy, dx, dt, **kwargs)
    b = deepwave.elastic(lamb, mu, buoyancy, dx, dt, **kwargs)
    for name in (
        "receiver_amplitudes_p",
        "receiver_amplitudes_y",
        "receiver_amplitudes_x",
    ):
        assert torch.equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize("comp", ["p", "y", "x"])
def test_elastic_on_node_float_equals_integer(comp):
    """Elastic: float on a component's node equals the integer-index result."""
    lamb, mu, buoyancy, dx, dt, amp = _elastic_setup()
    src_int = torch.tensor([[[10, 11]]])
    rec_int = torch.tensor([[[13, 9]]])
    offset = {"p": [0.0, 0.0], **_ACOUSTIC_VEL_2D}[comp]
    src_float = src_int.double() + torch.tensor(offset)
    rec_float = rec_int.double() + torch.tensor(offset)
    src_key, sloc_key = f"source_amplitudes_{comp}", f"source_locations_{comp}"
    rloc_key, rname = f"receiver_locations_{comp}", f"receiver_amplitudes_{comp}"
    base = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        pml_width=8,
        pml_freq=_FREQ,
        **{src_key: amp, sloc_key: src_int, rloc_key: rec_int},
    )
    snapped = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        pml_width=8,
        pml_freq=_FREQ,
        **{src_key: amp, sloc_key: src_float, rloc_key: rec_float},
    )
    assert torch.equal(getattr(base, rname), getattr(snapped, rname))


def test_elastic_float_matches_manual_hicks_velocity():
    """Elastic off-grid velocity source/receiver matches the manual Hicks route."""
    lamb, mu, buoyancy, dx, dt, amp = _elastic_setup()
    src = torch.tensor([[[10.3, 12.7]]], dtype=torch.double)
    rec = torch.tensor([[[18.4, 9.1]]], dtype=torch.double)
    offset = _ACOUSTIC_VEL_2D["y"]
    hs = Hicks(src, offset=offset, dtype=torch.double)
    hr = Hicks(rec, offset=offset, dtype=torch.double)
    manual = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        pml_width=8,
        pml_freq=_FREQ,
        source_amplitudes_y=hs.source(amp),
        source_locations_y=hs.get_locations(),
        receiver_locations_y=hr.get_locations(),
    )
    manual_rec = hr.receiver(manual.receiver_amplitudes_y)
    auto = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        dx,
        dt,
        pml_width=8,
        pml_freq=_FREQ,
        source_amplitudes_y=amp,
        source_locations_y=src,
        receiver_locations_y=rec,
    )
    assert torch.allclose(auto.receiver_amplitudes_y, manual_rec)


def _scalar_born_setup(ny=26, nx=26, nt=70, dtype=torch.double):
    v = torch.full((ny, nx), 1500.0, dtype=dtype)
    scatter = torch.zeros(ny, nx, dtype=dtype)
    scatter[ny // 2, nx // 2] = 100.0
    wavelet = deepwave.wavelets.ricker(_FREQ, nt, _DT, 1.5 / _FREQ, dtype=dtype)
    return v, scatter, [_DX, _DX], _DT, wavelet.reshape(1, 1, -1)


def test_scalar_born_integer_locations_byte_identical():
    """Integer (long) scalar_born locations must be bit-for-bit unchanged."""
    v, scatter, dx, dt, amp = _scalar_born_setup()
    kwargs = dict(  # noqa: C408
        source_amplitudes=amp,
        source_locations=torch.tensor([[[8, 9]]]),
        bg_receiver_locations=torch.tensor([[[14, 15], [11, 9]]]),
        sc_receiver_locations=torch.tensor([[[13, 8], [7, 16]]]),
        pml_width=8,
        pml_freq=_FREQ,
    )
    a = deepwave.scalar_born(v, dx, dt, dv=scatter, **kwargs)
    b = deepwave.scalar_born(v, dx, dt, dv=scatter, **kwargs)
    assert torch.equal(a.bg_receiver_amplitudes, b.bg_receiver_amplitudes)
    assert torch.equal(a.sc_receiver_amplitudes, b.sc_receiver_amplitudes)


def test_scalar_born_float_on_grid_snap_to_integer():
    """scalar_born integer-valued float coords reproduce the integer-location result."""
    v, scatter, dx, dt, amp = _scalar_born_setup()
    s_long = torch.tensor([[[8, 9]]])
    bg_long = torch.tensor([[[14, 15]]])
    r_long = torch.tensor([[[13, 8]]])
    base = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        pml_width=8,
        pml_freq=_FREQ,
        source_amplitudes=amp,
        source_locations=s_long,
        bg_receiver_locations=bg_long,
        sc_receiver_locations=r_long,
    )
    snapped = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        pml_width=8,
        pml_freq=_FREQ,
        source_amplitudes=amp,
        source_locations=s_long.double(),
        bg_receiver_locations=bg_long.double(),
        sc_receiver_locations=r_long.double(),
    )
    assert torch.equal(base.bg_receiver_amplitudes, snapped.bg_receiver_amplitudes)
    assert torch.equal(base.sc_receiver_amplitudes, snapped.sc_receiver_amplitudes)


def test_scalar_born_float_matches_manual_hicks():
    """scalar_born off-grid coords match the manual Hicks workflow.

    The single source spread is shared by the background and scattered inputs,
    and each receiver set is gathered with its own interpolator.
    """
    v, scatter, dx, dt, amp = _scalar_born_setup()
    src = torch.tensor([[[10.3, 12.7]]], dtype=torch.double)
    bg = torch.tensor([[[18.4, 9.1]]], dtype=torch.double)
    rec = torch.tensor([[[7.6, 15.2]]], dtype=torch.double)
    hs = Hicks(src, dtype=torch.double)
    hbg = Hicks(bg, dtype=torch.double)
    hr = Hicks(rec, dtype=torch.double)
    manual = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        pml_width=8,
        pml_freq=_FREQ,
        source_amplitudes=hs.source(amp),
        source_locations=hs.get_locations(),
        bg_receiver_locations=hbg.get_locations(),
        sc_receiver_locations=hr.get_locations(),
    )
    manual_bg = hbg.receiver(manual.bg_receiver_amplitudes)
    manual_rec = hr.receiver(manual.sc_receiver_amplitudes)
    auto = deepwave.scalar_born(
        v,
        dx,
        dt,
        dv=scatter,
        pml_width=8,
        pml_freq=_FREQ,
        source_amplitudes=amp,
        source_locations=src,
        bg_receiver_locations=bg,
        sc_receiver_locations=rec,
    )
    assert torch.allclose(auto.bg_receiver_amplitudes, manual_bg)
    assert torch.allclose(auto.sc_receiver_amplitudes, manual_rec)


# ---------------------------------------------------------------------------
# Differentiable positions (Hicks(differentiable=True) and the auto path)
# ---------------------------------------------------------------------------


def _combine(hicks, shotidx, i):
    """Flat outer-product weights for one point (test helper)."""
    return _combine_weights(hicks.weights[shotidx][i])


def test_differentiable_default_path_bit_identical():
    """differentiable=False must produce bit-identical weights and locations.

    Guards the default path against the shared-helper refactor: the weights are
    compared with the historical inline Kaiser-windowed-sinc expression, written
    out verbatim, and the spread/gather outputs of differentiable=True are also
    checked to match (to floating-point equality of the same expression graph).
    """
    torch.manual_seed(0)
    src = torch.tensor(
        [[[10.3, 12.7], [18.6, 9.2], [5.0, 7.25]]],
        dtype=torch.double,
    )
    amp = torch.randn(1, 3, 5, dtype=torch.double)
    halfwidth = 4
    h0 = Hicks(src, halfwidth=halfwidth, dtype=torch.double)
    # Historical (pre-refactor) weight expression, evaluated directly.
    import math

    betas = [0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64]
    # Replicate the historical beta construction exactly (a default-dtype
    # tensor upcast to the target dtype), as Hicks has always done.
    beta = torch.tensor(betas[halfwidth - 1]).to(torch.double)
    for i in range(src.shape[1]):
        for d in range(2):
            loc = float(src[0, i, d])
            if abs(loc - round(loc)) < 1e-5:
                expected = torch.ones(1, dtype=torch.double)
            else:
                floor_location = math.floor(loc)
                x = (
                    torch.arange(
                        -halfwidth + 1,
                        halfwidth + 1,
                        dtype=torch.double,
                    )
                    - loc
                    + floor_location
                )
                expected = (
                    torch.sinc(x)
                    * torch.i0(beta * (1 - (x / halfwidth) ** 2).sqrt())
                    / torch.i0(beta)
                )
            assert torch.equal(h0.weights[0][i][d], expected)
    # The differentiable path spreads to the same values at the same cells up
    # to round-off: it never takes the on-node unit-spike shortcut, and at an
    # on-node coordinate its window's off-cells carry sinc(n*pi) residuals of
    # ~1e-16 rather than exact zeros.
    hd = Hicks(
        src.clone().requires_grad_(True),
        halfwidth=halfwidth,
        dtype=torch.double,
        differentiable=True,
    )
    s0 = h0.source(amp)
    sd = hd.source(amp)
    # Scatter both to a common grid to compare independently of cell ordering.
    grid0 = torch.zeros(30, 30, 5, dtype=torch.double)
    gridd = torch.zeros(30, 30, 5, dtype=torch.double)
    l0 = h0.get_locations()[0]
    ld = hd.get_locations()[0]
    for j in range(l0.shape[0]):
        if l0[j, 0] != deepwave.IGNORE_LOCATION:
            grid0[l0[j, 0], l0[j, 1]] += s0[0, j]
    for j in range(ld.shape[0]):
        if ld[j, 0] != deepwave.IGNORE_LOCATION:
            gridd[ld[j, 0], ld[j, 1]] += sd[0, j].detach()
    assert (grid0 - gridd).abs().max() < 1e-14


def test_differentiable_gradcheck_source_positions():
    """Gradcheck (float64) of spread amplitudes w.r.t. source positions."""
    torch.manual_seed(0)
    amp = torch.randn(1, 2, 3, dtype=torch.double)
    loc = torch.tensor(
        [[[10.3, 12.7], [18.6, 9.2]]],
        dtype=torch.double,
        requires_grad=True,
    )

    def spread(location):
        return Hicks(location, dtype=torch.double, differentiable=True).source(
            amp,
        )

    assert torch.autograd.gradcheck(spread, (loc,))


def test_differentiable_gradcheck_dipole_source_positions():
    """Gradcheck of a dipole source's spread amplitudes w.r.t. positions."""
    torch.manual_seed(0)
    amp = torch.randn(1, 1, 3, dtype=torch.double)
    loc = torch.tensor(
        [[[10.3, 12.7]]],
        dtype=torch.double,
        requires_grad=True,
    )

    def spread(location):
        return Hicks(
            location,
            dtype=torch.double,
            differentiable=True,
            monopole=False,
            dipole_dim=1,
        ).source(amp)

    assert torch.autograd.gradcheck(spread, (loc,))


def test_differentiable_gradcheck_receiver_positions():
    """Gradcheck (float64) of gathered receiver traces w.r.t. positions."""
    torch.manual_seed(0)
    # A fixed "wavefield sampled on the grid": gather amplitudes for the
    # interpolated cells from it, then Hicks.receiver maps them back to the
    # original (off-grid) receiver locations.
    grid = torch.randn(30, 30, 4, dtype=torch.double)
    loc = torch.tensor(
        [[[20.4, 11.1], [8.7, 14.2]]],
        dtype=torch.double,
        requires_grad=True,
    )

    def gather(location):
        h = Hicks(location, dtype=torch.double, differentiable=True)
        cells = h.get_locations()[0]
        amps = grid[cells[:, 0], cells[:, 1]].unsqueeze(0)
        return h.receiver(amps)

    assert torch.autograd.gradcheck(gather, (loc,))


def test_differentiable_on_node_gradient_finite():
    """The position gradient must be finite at exactly on-node positions.

    The window edge point has ``x = +-halfwidth`` there, where the naive
    autograd path through ``sqrt(1 - (x/halfwidth)**2)`` gives ``0 * inf =
    NaN``; the grad-safe Kaiser expression must give the finite analytic value.
    """
    torch.manual_seed(0)
    amp = torch.randn(1, 1, 3, dtype=torch.double)
    loc = torch.tensor(
        [[[17.0, 23.0]]],
        dtype=torch.double,
        requires_grad=True,
    )
    h = Hicks(loc, dtype=torch.double, differentiable=True)
    h.source(amp).sum().backward()
    assert loc.grad is not None
    assert bool(torch.isfinite(loc.grad).all())
    # The on-node window must still act as a unit spike: spreading a source at
    # an integer position gives the same values as the exact integer path.
    cells = h.get_locations()[0]
    spread = h.source(amp)[0]
    on_node = (cells[:, 0] == 17) & (cells[:, 1] == 23)
    assert torch.allclose(spread[on_node][0], amp[0, 0])
    assert torch.allclose(
        spread[~on_node],
        torch.zeros_like(spread[~on_node]),
    )


def test_differentiable_weights_continuous_across_cell_crossing():
    """Weights and gradients are continuous as a point crosses an integer.

    When the position crosses an integer the support set shifts by one cell,
    but the entering/leaving cell's weight tends to zero at the transition, so
    the spread (aligned by absolute cell index) and the position-gradient are
    continuous there.
    """

    def spread_dict(y):
        """Spread weights keyed by absolute cell index."""
        h = Hicks(
            torch.tensor([[[y, 12.25]]], dtype=torch.double),
            dtype=torch.double,
            differentiable=True,
        )
        cells = h.get_locations()[0]
        w = _combine_weights(h.weights[0][0]).detach()
        return {
            tuple(int(v) for v in c): float(wt)
            for c, wt in zip(cells.tolist(), w.tolist())
        }

    step = 1e-6
    below = spread_dict(12 - step)
    above = spread_dict(12 + step)
    # The support set changes by exactly one column of cells...
    assert set(below) != set(above)
    # ...but the weights, aligned by absolute cell, are continuous: the jump
    # across the crossing is no larger than a same-sized step elsewhere.
    all_cells = set(below) | set(above)
    jump_at = max(abs(above.get(c, 0.0) - below.get(c, 0.0)) for c in all_cells)
    generic_a = spread_dict(11.63 - step)
    generic_b = spread_dict(11.63 + step)
    jump_generic = max(
        abs(generic_b.get(c, 0.0) - generic_a.get(c, 0.0))
        for c in set(generic_a) | set(generic_b)
    )
    assert jump_at <= jump_generic
    # The entering/leaving cells have (numerically) zero weight at the
    # transition itself.
    changed = set(below) ^ set(above)
    at = spread_dict(12.0)
    for c in changed:
        assert abs(at.get(c, 0.0)) < 1e-12

    # Gradient continuity: the analytic d(sum of squared weights)/dy approaches
    # the same magnitude from both sides (it is odd around the node).
    def grad_at(y):
        loc = torch.tensor(
            [[[y, 12.25]]],
            dtype=torch.double,
            requires_grad=True,
        )
        h = Hicks(loc, dtype=torch.double, differentiable=True)
        w = _combine_weights(h.weights[0][0])
        (w * w).sum().backward()
        return float(loc.grad[0, 0, 0])

    g_below = grad_at(12 - 1e-7)
    g_above = grad_at(12 + 1e-7)
    assert abs(g_below + g_above) < 1e-4 * max(abs(g_below), 1e-12) + 1e-9


def test_differentiable_ignore_location_zero_gradient():
    """IGNORE_LOCATION sentinel points get exactly zero position gradient."""
    torch.manual_seed(0)
    ignore = float(deepwave.IGNORE_LOCATION)
    loc = torch.tensor(
        [[[10.3, 12.7], [ignore, ignore]]],
        dtype=torch.double,
        requires_grad=True,
    )
    amp = torch.randn(1, 2, 4, dtype=torch.double)
    h = Hicks(loc, dtype=torch.double, differentiable=True)
    h.source(amp).sum().backward()
    assert loc.grad is not None
    assert bool(torch.isfinite(loc.grad).all())
    assert loc.grad[0, 0].abs().sum() > 0
    assert torch.equal(loc.grad[0, 1], torch.zeros(2, dtype=torch.double))


def test_differentiable_surfaces_raise():
    """Free/rigid surfaces are not supported in differentiable mode."""
    loc = torch.tensor([[[1.3, 12.7]]], dtype=torch.double)
    with pytest.raises(NotImplementedError):
        Hicks(
            loc,
            dtype=torch.double,
            differentiable=True,
            free_surfaces=[True, False, False, False],
            model_shape=[30, 30],
        )
    with pytest.raises(NotImplementedError):
        Hicks(
            loc,
            dtype=torch.double,
            differentiable=True,
            rigid_surfaces=[True, False, False, False],
            model_shape=[30, 30],
        )


def test_differentiable_requires_float_locations():
    """differentiable=True requires floating-point locations."""
    loc = torch.tensor([[[1, 12]]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="floating-point"):
        Hicks(loc, dtype=torch.double, differentiable=True)


def test_should_interpolate_requires_grad_forces_hicks():
    """An on-node float tensor with requires_grad must still interpolate.

    Only the interpolated (differentiable) path can carry position gradients;
    snapping to the exact path would silently sever them.
    """
    on_node = torch.tensor([[[7.0, 3.0]]], dtype=torch.double)
    assert _should_interpolate(on_node, "auto") is False
    assert _should_interpolate(on_node.clone().requires_grad_(True), "auto") is True


def test_differentiable_position_gradient_through_propagator():
    """dLoss/dposition through a full scalar propagation matches central FD.

    Uses the auto path (float positions with requires_grad on a propagator
    call). Central finite difference over a 1e-3-cell position perturbation in
    float64; tolerance 1e-4 relative.
    """
    dt = 0.001
    model = torch.ones(40, 40, dtype=torch.double) * 1500.0
    wav = deepwave.wavelets.ricker(
        15,
        70,
        dt,
        1.5 / 15,
        dtype=torch.double,
    ).reshape(1, 1, -1)
    rec = torch.tensor(
        [[[25.0, 20.0], [10.0, 30.0], [30.0, 30.0]]],
        dtype=torch.double,
    )
    src0 = torch.tensor([[[15.3, 20.4]]], dtype=torch.double)

    def loss_at(src_tensor):
        out = deepwave.scalar(
            model,
            5.0,
            dt,
            source_amplitudes=wav,
            source_locations=src_tensor,
            receiver_locations=rec,
            pml_freq=15,
        )
        return out[-1].pow(2).sum()

    src = src0.clone().requires_grad_(True)
    loss_at(src).backward()
    assert src.grad is not None
    h = 1e-3
    for d in range(2):
        sp = src0.clone()
        sp[0, 0, d] += h
        sm = src0.clone()
        sm[0, 0, d] -= h
        fd = float((loss_at(sp) - loss_at(sm)) / (2 * h))
        auto = float(src.grad[0, 0, d])
        assert abs(auto - fd) < 1e-4 * abs(fd)


def test_differentiable_receiver_position_gradient_through_propagator():
    """dLoss/d(receiver position) through a full scalar propagation vs FD."""
    dt = 0.001
    model = torch.ones(40, 40, dtype=torch.double) * 1500.0
    wav = deepwave.wavelets.ricker(
        15,
        70,
        dt,
        1.5 / 15,
        dtype=torch.double,
    ).reshape(1, 1, -1)
    src = torch.tensor([[[15.3, 20.4]]], dtype=torch.double)
    rec0 = torch.tensor([[[25.6, 19.2]]], dtype=torch.double)

    def loss_at(rec_tensor):
        out = deepwave.scalar(
            model,
            5.0,
            dt,
            source_amplitudes=wav,
            source_locations=src,
            receiver_locations=rec_tensor,
            pml_freq=15,
        )
        return out[-1].pow(2).sum()

    rec = rec0.clone().requires_grad_(True)
    loss_at(rec).backward()
    assert rec.grad is not None
    h = 1e-3
    for d in range(2):
        rp = rec0.clone()
        rp[0, 0, d] += h
        rm = rec0.clone()
        rm[0, 0, d] -= h
        fd = float((loss_at(rp) - loss_at(rm)) / (2 * h))
        auto = float(rec.grad[0, 0, d])
        assert abs(auto - fd) < 1e-4 * abs(fd)


def test_differentiable_source_localization_descent():
    """Gradient descent moves a mislocated source toward its true position."""
    dt = 0.001
    model = torch.ones(40, 40, dtype=torch.double) * 1500.0
    wav = deepwave.wavelets.ricker(
        15,
        70,
        dt,
        1.5 / 15,
        dtype=torch.double,
    ).reshape(1, 1, -1)
    rec = torch.tensor(
        [[[25.0, 20.0], [10.0, 30.0], [30.0, 30.0]]],
        dtype=torch.double,
    )
    true_src = torch.tensor([[[20.0, 20.0]]], dtype=torch.double)
    observed = deepwave.scalar(
        model,
        5.0,
        dt,
        source_amplitudes=wav,
        source_locations=true_src,
        receiver_locations=rec,
        pml_freq=15,
    )[-1].detach()
    guess = torch.tensor([[[17.0, 23.0]]], dtype=torch.double)

    def position_error(g):
        return float(((g - true_src) ** 2).sum().sqrt())

    initial_error = position_error(guess)
    losses = []
    for _ in range(2):
        p = guess.clone().requires_grad_(True)
        out = deepwave.scalar(
            model,
            5.0,
            dt,
            source_amplitudes=wav,
            source_locations=p,
            receiver_locations=rec,
            pml_freq=15,
        )[-1]
        loss = (out - observed).pow(2).sum()
        loss.backward()
        losses.append(float(loss))
        assert p.grad is not None
        # The gradient must point the right way: the true source is at larger
        # dim-0 and smaller dim-1 than the guess, so descent (-grad) must move
        # dim 0 up (grad < 0) and dim 1 down (grad > 0).
        assert float(p.grad[0, 0, 0]) < 0
        assert float(p.grad[0, 0, 1]) > 0
        guess = (p - p.grad / p.grad.abs().max()).detach()
    assert position_error(guess) < initial_error
    assert losses == sorted(losses, reverse=True)


# ---------------------------------------------------------------------------
# Coverage gap-list follow-up tests: Hicks.__init__'s public-constructor
# validation (previously only exercised one layer down, on the private
# _get_hicks_for_one_location_dim helper), _should_interpolate's mode guard,
# shot_idxs on get_locations/receiver, and the Tensor-valued
# monopole/dipole_dim per-point paths.
# ---------------------------------------------------------------------------


def test_should_interpolate_invalid_mode() -> None:
    """_should_interpolate rejects a mode outside auto/hicks/none."""
    with pytest.raises(ValueError, match="location_interpolation must be one of"):
        _should_interpolate(torch.rand(1, 1, 2), mode="bogus")


def test_get_hicks_for_one_location_dim_halfwidth_type() -> None:
    """The private helper's halfwidth TypeError differs from its range check."""
    with pytest.raises(TypeError, match=r"halfwidth must be an int\."):
        _get_hicks_for_one_location_dim(
            {}, 0.5, 4.5, torch.tensor(1.0), [False, False], [-0.5, 19.5], 20
        )


def test_get_hicks_for_one_location_dim_both_free_and_rigid() -> None:
    """A face cannot be both free and rigid (checked again at this layer)."""
    with pytest.raises(RuntimeError, match="cannot be both a free and a rigid"):
        _get_hicks_for_one_location_dim(
            {},
            0.5,
            4,
            torch.tensor(1.0),
            [True, False],
            [-0.5, 19.5],
            20,
            rigid_surface=[True, False],
        )


def test_hicks_init_differentiable_type() -> None:
    """Differentiable must be a bool."""
    with pytest.raises(TypeError, match="differentiable must be a bool"):
        Hicks(torch.randn(2, 3, 2), differentiable="yes")


def test_hicks_init_ndim_out_of_range() -> None:
    """Locations' final dimension must be 1, 2, or 3."""
    with pytest.raises(RuntimeError, match="final dimension of size 1, 2, or 3"):
        Hicks(torch.randn(2, 3, 4))


def test_hicks_init_halfwidth_validation() -> None:
    """Hicks.__init__ validates halfwidth's type and range itself."""
    locations = torch.randn(2, 3, 2)
    with pytest.raises(TypeError, match="halfwidth must be an integer"):
        Hicks(locations, halfwidth=4.5)
    with pytest.raises(RuntimeError, match=r"halfwidth must be in \[1, 10\]"):
        Hicks(locations, halfwidth=0)
    with pytest.raises(RuntimeError, match=r"halfwidth must be in \[1, 10\]"):
        Hicks(locations, halfwidth=11)


def test_hicks_init_surfaces_validation() -> None:
    """free_surfaces/rigid_surfaces validation on the public Hicks() constructor."""
    locations = torch.randn(2, 3, 2)
    with pytest.raises(
        RuntimeError, match="free_surfaces must be a sequence of 4 bools"
    ):
        Hicks(locations, free_surfaces=[True, False])  # wrong length
    with pytest.raises(
        RuntimeError, match="free_surfaces must be a sequence of 4 bools"
    ):
        Hicks(locations, free_surfaces=[1, 0, 0, 0])  # non-bool entries
    with pytest.raises(
        RuntimeError, match="rigid_surfaces must be a sequence of 4 bools"
    ):
        Hicks(locations, rigid_surfaces=[True, False])
    with pytest.raises(RuntimeError, match="cannot be both a free and a rigid"):
        Hicks(
            locations,
            free_surfaces=[True, False, False, False],
            rigid_surfaces=[True, False, False, False],
            model_shape=[20, 20],
        )
    with pytest.raises(RuntimeError, match="model_shape must be specified"):
        Hicks(locations, free_surfaces=[True, False, False, False])


def test_hicks_init_model_shape_validation() -> None:
    """model_shape's type/length and int-convertibility checks."""
    locations = torch.randn(2, 3, 2)
    with pytest.raises(
        RuntimeError, match=r"model_shape must be a sequence with 2 int"
    ):
        Hicks(locations, model_shape=[20])
    with pytest.raises(TypeError, match="model_shape entries must be convertible"):
        Hicks(locations, model_shape=[20, "a"])


def test_hicks_init_free_surface_locs_validation() -> None:
    """free_surface_locs' type/length and float-convertibility checks."""
    locations = torch.randn(2, 3, 2)
    with pytest.raises(
        RuntimeError, match=r"free_surface_locs must be a sequence of 4 floats"
    ):
        Hicks(locations, free_surface_locs=[-0.5, 19.5])
    with pytest.raises(TypeError, match="free_surface_locs entries"):
        Hicks(locations, free_surface_locs=[-0.5, 19.5, "a", 19.5])


def test_hicks_init_offset_validation() -> None:
    """Offset's type/length and float-convertibility checks."""
    locations = torch.randn(2, 3, 2)
    with pytest.raises(RuntimeError, match=r"offset must be a sequence of 2 floats"):
        Hicks(locations, offset=[0.5])
    with pytest.raises(TypeError, match="offset entries must be convertible"):
        Hicks(locations, offset=[0.5, "a"])


def test_hicks_init_monopole_dipole_tensor_validation() -> None:
    """Tensor-valued monopole/dipole_dim shape and range checks."""
    locations = torch.randn(2, 3, 2)
    with pytest.raises(RuntimeError, match=r"monopole must have dimensions"):
        Hicks(locations, monopole=torch.ones(2, 2, dtype=torch.bool))
    with pytest.raises(RuntimeError, match=r"dipole_dim must have dimensions"):
        Hicks(locations, dipole_dim=torch.zeros(2, 2, dtype=torch.long))
    with pytest.raises(RuntimeError, match=r"dipole_dim entries must be in"):
        Hicks(locations, dipole_dim=torch.full((2, 3), 5, dtype=torch.long))
    with pytest.raises(RuntimeError, match=r"dipole_dim must be in"):
        Hicks(locations, dipole_dim=5)


def test_hicks_tensor_valued_monopole_and_dipole_dim() -> None:
    """Per-point Tensor monopole/dipole_dim give different weights per point.

    Two points at the same location: one a monopole, one a dipole oriented
    along dim 0 vs dim 1 -- their Hicks weight profiles must all differ from
    each other, confirming the per-point (not per-call) resolution actually
    takes effect (source_field/dipole_dim as scalars is already covered
    elsewhere; this is the array-valued [shot, per_shot] path).
    """
    loc = torch.tensor([[[5.3, 6.7], [5.3, 6.7], [5.3, 6.7]]])
    monopole = torch.tensor([[True, False, False]])
    dipole_dim = torch.tensor([[0, 0, 1]])
    hicks = Hicks(loc, monopole=monopole, dipole_dim=dipole_dim, model_shape=[20, 20])
    w0 = torch.cat([w.reshape(-1) for w in hicks.weights[0][0]])
    w1 = torch.cat([w.reshape(-1) for w in hicks.weights[0][1]])
    w2 = torch.cat([w.reshape(-1) for w in hicks.weights[0][2]])
    assert not torch.allclose(w0, w1)
    assert not torch.allclose(w1, w2)
    assert not torch.allclose(w0, w2)


def test_hicks_get_locations_shot_idxs() -> None:
    """get_locations(shot_idxs) indexes the same way as slicing the full result."""
    torch.manual_seed(7)
    locations = torch.rand(6, 2, 2) * 10
    hicks = Hicks(locations, model_shape=[20, 20])
    shot_idxs = torch.tensor([4, 1, 0])
    full = hicks.get_locations()
    subset = hicks.get_locations(shot_idxs)
    assert torch.equal(subset, full[shot_idxs])


def test_hicks_receiver_shot_idxs() -> None:
    """receiver(amplitudes, shot_idxs) matches gathering the full-batch result."""
    torch.manual_seed(8)
    n_shots, n_per_shot, nt = 6, 3, 5
    locations = torch.rand(n_shots, n_per_shot, 2, dtype=torch.double) * 10
    hicks = Hicks(locations, dtype=torch.double, model_shape=[100, 100])
    hicks_amplitudes = torch.randn(
        n_shots, len(hicks.hicks_locations[0]), nt, dtype=torch.double
    )
    full = hicks.receiver(hicks_amplitudes)
    shot_idxs = torch.tensor([n_shots // 2])
    half = hicks.receiver(hicks_amplitudes[shot_idxs], shot_idxs)
    assert torch.allclose(full[shot_idxs], half)


def test_check_shot_idxs_length_mismatch() -> None:
    """A shot_idxs whose length doesn't match amplitudes raises."""
    locations = torch.rand(4, 2, 2) * 10
    hicks = Hicks(locations, model_shape=[20, 20])
    amplitudes = torch.randn(4, 2, 5)
    with pytest.raises(RuntimeError, match="shot_idxs must have the same length"):
        hicks.source(amplitudes, shot_idxs=torch.tensor([0, 1]))


# ---------------------------------------------------------------------------
# Hicks dtype=None, the offset-mismatch warning, and "auto" routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_hicks_dtype_none_uses_default_dtype(default_dtype: torch.dtype) -> None:
    """``dtype=None`` means ``torch.get_default_dtype()``.

    This matches every other ``dtype`` parameter in the package; the old
    default was a hard-coded ``torch.float``.
    """
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        locations = torch.tensor([[[3.3, 4.7]]], dtype=default_dtype)
        interpolator = Hicks(locations)
        weights = interpolator.source(torch.ones(1, 1, 5, dtype=default_dtype))
        assert weights.dtype == default_dtype
    finally:
        torch.set_default_dtype(previous)


def test_hicks_dtype_none_is_bit_identical_under_a_float32_default() -> None:
    """Under the default (float32) default dtype, nothing moves at all.

    ``dtype=None`` resolves to ``torch.float32``, which is exactly the old
    hard-coded ``torch.float``, so every existing call is bit-identical.
    Under a *float64* default dtype the weights are now computed in float64
    rather than float32-then-promoted, which is a small, documented
    accuracy improvement rather than a regression -- see the CHANGELOG.
    """
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        locations = torch.tensor([[[3.3, 4.7]]])
        amplitudes = torch.ones(1, 1, 5)
        explicit = Hicks(locations, dtype=torch.float).source(amplitudes)
        implicit = Hicks(locations, dtype=None).source(amplitudes)
        assert torch.equal(explicit, implicit)
    finally:
        torch.set_default_dtype(previous)


def test_hicks_records_its_offset() -> None:
    """The offset is retrievable, which is what makes a mismatch detectable."""
    locations = torch.tensor([[[3.3, 4.7]]])
    assert Hicks(locations).offset == [0.0, 0.0]
    assert Hicks(locations, offset=[0.5, 0.0]).offset == [0.5, 0.0]


def _clear_offset_mismatch_warnings() -> None:
    """Drops the once-per-process guard so the warning can fire again."""
    from deepwave import _common

    for cause in list(_common._warned_once_causes):
        if cause.startswith("hicks-offset-mismatch:"):
            _common._warned_once_causes.discard(cause)


def test_hicks_offset_mismatch_warns_once() -> None:
    """A Hicks built for the wrong node is no longer silently accepted.

    ``_resolve_interpolator`` returns a user-supplied ``Hicks`` as-is,
    ignoring the component's own half-cell offset. That is correct -- it is
    how a custom offset is requested -- but it was completely silent, so a
    regular-grid ``Hicks`` used for a staggered velocity component
    interpolated half a cell out with no indication.
    """
    _clear_offset_mismatch_warnings()
    locations = torch.tensor([[[3.3, 4.7]]])
    regular = Hicks(locations)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _resolve_interpolator(regular, offset=[0.5, 0.0], kind="source")
    messages = [
        str(w.message) for w in caught if "Hicks object passed" in str(w.message)
    ]
    assert len(messages) == 1
    # Names both offsets and the fix.
    assert "offset=[0.0, 0.0]" in messages[0]
    assert "offset=[0.5, 0.0]" in messages[0]
    assert "Hicks(..., offset=[0.5, 0.0])" in messages[0]
    assert "source" in messages[0]

    # Once per process per (kind, offsets).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _resolve_interpolator(regular, offset=[0.5, 0.0], kind="source")
    assert not [w for w in caught if "Hicks object passed" in str(w.message)]


def test_hicks_matching_offset_does_not_warn() -> None:
    """The correct offset is the silent case."""
    _clear_offset_mismatch_warnings()
    locations = torch.tensor([[[3.3, 4.7]]])
    staggered = Hicks(locations, offset=[0.5, 0.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _resolve_interpolator(staggered, offset=[0.5, 0.0], kind="source")
        _resolve_interpolator(Hicks(locations), offset=None, kind="receiver")
    assert not [w for w in caught if "Hicks object passed" in str(w.message)]


def test_hicks_offset_mismatch_warns_through_a_propagator() -> None:
    """The warning reaches a real elastic call, not just the helper."""
    _clear_offset_mismatch_warnings()
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        ny = nx = 40
        lamb = torch.full((ny, nx), 2.25e9, dtype=torch.float64)
        mu = torch.full((ny, nx), 1.0e9, dtype=torch.float64)
        buoyancy = torch.full((ny, nx), 1 / 2000.0, dtype=torch.float64)
        amplitudes = torch.zeros(1, 1, 30, dtype=torch.float64)
        amplitudes[0, 0, 5] = 1.0
        regular = Hicks(torch.tensor([[[20.5, 20.0]]], dtype=torch.float64))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deepwave.elastic(
                lamb,
                mu,
                buoyancy,
                4.0,
                0.0005,
                source_amplitudes_y=amplitudes,
                source_locations_y=regular,
                pml_freq=25.0,
                backend="eager",
            )
        assert [w for w in caught if "Hicks object passed" in str(w.message)]
    finally:
        torch.set_default_dtype(previous)


def test_auto_interpolates_on_node_requires_grad_locations() -> None:
    """``"auto"`` always interpolates a ``requires_grad`` locations Tensor.

    Even when every coordinate is exactly on-node: only the interpolated
    path carries gradients back to the positions. That changes the forward
    values relative to the same positions without ``requires_grad``, which
    is why the routing is now documented and recorded.
    """
    ny = nx = 40
    v = torch.full((ny, nx), 1500.0)
    amplitudes = torch.zeros(1, 1, 40)
    amplitudes[0, 0, 5] = 1.0
    receivers = torch.tensor([[[10, 10]]])

    def run(locations: torch.Tensor) -> deepwave.PropagatorOutput:
        return deepwave.scalar(
            v,
            4.0,
            0.001,
            source_amplitudes=amplitudes,
            source_locations=locations,
            receiver_locations=receivers,
            pml_freq=25.0,
        )

    on_node = torch.tensor([[[20.0, 20.0]]])
    plain = run(on_node)
    differentiable = run(on_node.clone().requires_grad_(True))

    # Integer receivers take the exact path, so only the source moved. That
    # is an ordinary arrangement, not per-component routing, so it is
    # reported as "hicks": sources are compared with sources and receivers
    # with receivers, never with each other.
    assert plain.info.location_interpolation_used == "none"
    assert differentiable.info.location_interpolation_used == "hicks"
    # The routing really does change the forward values.
    assert not torch.equal(
        plain.receiver_amplitudes, differentiable.receiver_amplitudes.detach()
    )


def test_location_interpolation_used_reports_mixed() -> None:
    """Some components of one kind interpolated and some not is "mixed".

    Reporting that as plain ``"hicks"`` would tell a reader that every
    coordinate was band-limited when half of them were exact. It takes a
    propagator with several channels of one kind to produce it: this is
    per-component routing, which is the only thing "mixed" reports.
    """
    ny = nx = 40
    lamb = torch.full((ny, nx), 2.25e9)
    mu = torch.full((ny, nx), 1.0e9)
    buoyancy = torch.full((ny, nx), 1 / 2000.0)
    amplitudes = torch.zeros(1, 1, 40)
    amplitudes[0, 0, 5] = 1.0
    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        4.0,
        0.0005,
        source_amplitudes_y=amplitudes,
        source_locations_y=torch.tensor([[[20.3, 20.7]]]),  # off-node -> hicks
        source_amplitudes_x=amplitudes,
        source_locations_x=torch.tensor([[[20, 20]]]),  # integer -> exact
        receiver_locations_y=torch.tensor([[[10, 10]]]),
        pml_freq=25.0,
        backend="eager",
    )
    assert out.info.location_interpolation_used == "mixed"


def test_location_interpolation_used_is_hicks_for_off_node_source_only() -> None:
    """An off-node source with integer receivers is "hicks", not "mixed".

    Sources are compared with sources and receivers with receivers, never
    with each other, so this everyday single-channel arrangement is not
    per-component routing and must not be reported as if it were.
    """
    ny = nx = 40
    v = torch.full((ny, nx), 1500.0)
    amplitudes = torch.zeros(1, 1, 40)
    amplitudes[0, 0, 5] = 1.0
    out = deepwave.scalar(
        v,
        4.0,
        0.001,
        source_amplitudes=amplitudes,
        source_locations=torch.tensor([[[20.3, 20.7]]]),  # off-node -> hicks
        receiver_locations=torch.tensor([[[10, 10]]]),  # integer -> exact
        pml_freq=25.0,
    )
    assert out.info.location_interpolation_used == "hicks"


@pytest.mark.parametrize("empty", ["none", "tensor", "sentinel"])
def test_location_interpolation_used_ignores_channels_with_no_locations(
    empty: str,
) -> None:
    """A channel with nothing in it has taken neither path.

    ``None``, an empty Tensor and a Tensor of nothing but
    ``IGNORE_LOCATION`` are three spellings of "nothing here", so all three
    must be ignored, not counted as an on-node channel that disagrees with
    the interpolated one.
    """
    ny = nx = 40
    lamb = torch.full((ny, nx), 2.25e9)
    mu = torch.full((ny, nx), 1.0e9)
    buoyancy = torch.full((ny, nx), 1 / 2000.0)
    amplitudes = torch.zeros(1, 1, 40)
    amplitudes[0, 0, 5] = 1.0
    receivers: dict[str, object] = {
        "none": None,
        "tensor": torch.zeros(1, 0, 2, dtype=torch.long),
        "sentinel": torch.full((1, 1, 2), deepwave.IGNORE_LOCATION, dtype=torch.long),
    }
    out = deepwave.elastic(
        lamb,
        mu,
        buoyancy,
        4.0,
        0.0005,
        source_amplitudes_y=amplitudes,
        source_locations_y=torch.tensor([[[20.3, 20.7]]]),
        receiver_locations_y=receivers[empty],
        pml_freq=25.0,
        backend="eager",
    )
    assert out.info.location_interpolation_used == "hicks"


def test_location_interpolation_used_reports_hicks_when_every_group_moves() -> None:
    """Every location group interpolated is reported as ``"hicks"``."""
    ny = nx = 40
    v = torch.full((ny, nx), 1500.0)
    amplitudes = torch.zeros(1, 1, 40)
    amplitudes[0, 0, 5] = 1.0
    out = deepwave.scalar(
        v,
        4.0,
        0.001,
        source_amplitudes=amplitudes,
        source_locations=torch.tensor([[[20.3, 20.7]]]),
        receiver_locations=torch.tensor([[[10.4, 10.6]]]),
        pml_freq=25.0,
    )
    assert out.info.location_interpolation_used == "hicks"


# ---------------------------------------------------------------------------
# Migration warnings: the two changes of meaning a v0.0.27 call can meet here
# ---------------------------------------------------------------------------


def _clear_migration_causes() -> None:
    """Re-arms every once-per-process migration warning.

    By prefix rather than by name: the ``Hicks`` dtype cause carries the dtype
    it resolved to, so there is one key per dtype rather than one key. The two
    named constants are included so that a rename of either fails here.
    """
    assert _TRUNCATED_COORDINATE_CAUSE.startswith("migration:")
    assert _STENCIL_CAUSE.startswith("migration:")
    causes = deepwave._common._warned_once_causes
    for cause in [c for c in causes if c.startswith("migration:")]:
        causes.discard(cause)


@pytest.fixture
def rearmed_migration_warnings() -> Iterator[None]:
    """Runs the test with the migration causes unspent, and restores them.

    Restored rather than left cleared: an emptied registry would re-arm every
    once-per-process warning for whatever runs next in this worker, which could
    make a later test see a warning it asserts is absent.
    """
    saved = set(deepwave._common._warned_once_causes)
    _clear_migration_causes()
    try:
        yield
    finally:
        deepwave._common._warned_once_causes.clear()
        deepwave._common._warned_once_causes.update(saved)


def _scalar_with_source_at(y: float, x: float = 20.0) -> None:
    """A one-shot ``scalar`` call with one float source coordinate."""
    deepwave.scalar(
        torch.full((40, 40), 1500.0),
        4.0,
        0.001,
        source_amplitudes=torch.ones(1, 1, 4),
        source_locations=torch.tensor([[[y, x]]]),
        pml_freq=25.0,
    )


def _migrations(fn: Callable[[], None]) -> list[str]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fn()
    return [
        str(w.message)
        for w in record
        if issubclass(w.category, deepwave.MigrationWarning)
    ]


# The upper edge of the band, ``0.1``, is deliberately not tested: a
# coordinate written as ``20.1`` has a float32 fractional part of
# ``0.10000038``, so which side of the limit it falls on is a property of the
# literal rather than of the rule.
@pytest.mark.parametrize("fraction", [0.001, 0.05, 0.09])
@pytest.mark.usefixtures("rearmed_migration_warnings")
def test_a_coordinate_v0027_truncated_warns(fraction: float) -> None:
    """A coordinate inside ``v0.0.27``'s acceptance band says it moved.

    That release took anything within ``0.1`` cells of the node below and
    truncated it there; the same call now interpolates, which is a different
    number for the same arguments.
    """
    messages = _migrations(lambda: _scalar_with_source_at(20 + fraction))
    truncation = [m for m in messages if "truncated it to that node" in m]
    assert len(truncation) == 1, messages
    assert "locations.floor().long()" in truncation[0]
    assert "0.4.0" in truncation[0]
    assert "deepwave.MigrationWarning" in truncation[0]


@pytest.mark.parametrize("y", [20.0, 20.000001, 20.5, 20.9])
@pytest.mark.usefixtures("rearmed_migration_warnings")
def test_a_coordinate_v0027_rejected_or_snapped_says_nothing(y: float) -> None:
    """Outside the band there is nothing to migrate.

    On the node (or within the on-node tolerance) the call is what ``v0.0.27``
    computed; further than ``0.1`` cells off-node that release refused the call
    outright, so no result of its can differ from this one.
    """
    messages = _migrations(lambda: _scalar_with_source_at(y))
    assert [m for m in messages if "truncated it to that node" in m] == [], messages


@pytest.mark.usefixtures("rearmed_migration_warnings")
def test_the_truncation_warning_is_emitted_once_per_process() -> None:
    """An inversion loop hears it once, not once per iteration."""
    messages = _migrations(
        lambda: [_scalar_with_source_at(20.05) for _ in range(3)] and None
    )
    assert len([m for m in messages if "truncated it to that node" in m]) == 1


@pytest.mark.usefixtures("rearmed_migration_warnings")
def test_the_truncation_warning_is_filterable_by_its_category() -> None:
    """One line on the category silences it."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=deepwave.MigrationWarning)
        _scalar_with_source_at(20.05)
    assert [
        w for w in record if issubclass(w.category, deepwave.MigrationWarning)
    ] == []


@pytest.mark.usefixtures("rearmed_migration_warnings")
def test_integer_locations_never_warn() -> None:
    """The exact path is untouched: an integer survey says nothing at all."""
    messages = _migrations(
        lambda: deepwave.scalar(
            torch.full((40, 40), 1500.0),
            4.0,
            0.001,
            source_amplitudes=torch.ones(1, 1, 4),
            source_locations=torch.tensor([[[20, 20]]]),
            receiver_locations=torch.tensor([[[10, 10]]]),
            pml_freq=25.0,
        )
    )
    assert messages == []


@pytest.mark.usefixtures("rearmed_migration_warnings")
def test_hicks_default_dtype_warns_only_off_float32() -> None:
    """``Hicks()`` with no ``dtype=`` follows the default dtype, and says so."""
    locations = torch.tensor([[[20.5, 20.5]]], dtype=torch.float64)
    torch.set_default_dtype(torch.float64)
    try:
        messages = _migrations(lambda: Hicks(locations))
    finally:
        torch.set_default_dtype(torch.float32)
    dtype_messages = [m for m in messages if "built with no dtype=" in m]
    assert len(dtype_messages) == 1, messages
    assert "dtype=torch.float32" in dtype_messages[0]
    assert "0.4.0" in dtype_messages[0]
    assert "deepwave.MigrationWarning" in dtype_messages[0]


@pytest.mark.usefixtures("rearmed_migration_warnings")
def test_hicks_default_dtype_is_silent_under_the_stock_default() -> None:
    """Under the ``float32`` default it is the same call ``v0.0.27`` made."""
    assert torch.get_default_dtype() is torch.float32
    messages = _migrations(lambda: Hicks(torch.tensor([[[20.5, 20.5]]])))
    assert [m for m in messages if "built with no dtype=" in m] == [], messages


@pytest.mark.usefixtures("rearmed_migration_warnings")
def test_hicks_explicit_dtype_is_silent_under_any_default() -> None:
    """Naming the dtype says what you meant, and silences the warning."""
    locations = torch.tensor([[[20.5, 20.5]]], dtype=torch.float64)
    torch.set_default_dtype(torch.float64)
    try:
        messages = _migrations(lambda: Hicks(locations, dtype=torch.float64))
    finally:
        torch.set_default_dtype(torch.float32)
    assert [m for m in messages if "built with no dtype=" in m] == [], messages


def test_hicks_float32_dtype_reproduces_the_v0027_weights() -> None:
    """The fix the warning names really does give the old weights.

    ``dtype=torch.float32`` under a ``float64`` default must build exactly the
    weights the ``float32`` default builds, because that is what ``v0.0.27``
    did whatever the default was.
    """
    locations = torch.tensor([[[20.5, 20.5]]], dtype=torch.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        under_stock = Hicks(locations.float(), dtype=torch.float32)
        torch.set_default_dtype(torch.float64)
        try:
            pinned = Hicks(locations, dtype=torch.float32)
        finally:
            torch.set_default_dtype(torch.float32)
    assert len(under_stock.weights[0][0]) == len(pinned.weights[0][0])
    for a, b in zip(under_stock.weights[0][0], pinned.weights[0][0]):
        assert torch.equal(a, b)


@pytest.mark.usefixtures("rearmed_migration_warnings")
def test_the_truncation_question_adds_no_device_synchronisation() -> None:
    """Asking it costs no extra read of the coordinates.

    A migration warning that cost a host synchronisation per call would be a
    performance regression for every CUDA user, paid to say something only a
    migrating caller needs to hear. The question is therefore answered inside
    the reduction the interpolation decision already performs, and both
    answers come back in one transfer. Measured rather than asserted, by
    counting PyTorch's own synchronisation warnings with and without the
    question being asked.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    locations = torch.tensor([[[20.05, 20.0], [25.5, 25.0]]], device="cuda")

    def syncs(*, warn_truncated: bool) -> int:
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                _has_off_node_coord(
                    locations, None, 1e-5, warn_truncated=warn_truncated
                )
        finally:
            torch.cuda.set_sync_debug_mode("default")
        return len([w for w in record if "called a synchronizing" in str(w.message)])

    without = syncs(warn_truncated=False)
    _clear_migration_causes()
    with_question = syncs(warn_truncated=True)
    assert without > 0, "the reduction is expected to read back to the host"
    assert with_question == without, (with_question, without)


# ---------------------------------------------------------------------------
# Reflecting faces on the automatic (float location Tensor) path
# ---------------------------------------------------------------------------
# A float source or receiver coordinate is interpolated over a window of
# ``2 * halfwidth`` cells, so a point within ``halfwidth - 1`` cells of a face
# has part of that window outside the model. Where the propagator's own
# boundary condition is one whose image ``Hicks`` reproduces, the automatic
# path now folds the window back exactly as a hand-built ``Hicks`` carrying
# that surface does; everywhere else the call is refused with a message that
# names the coordinate the caller wrote.


def _scalar_run(
    locations: torch.Tensor,
    amplitudes: torch.Tensor,
    boundary: list[str],
    **kwargs: object,
) -> torch.Tensor:
    """One short scalar run, returning the recorded trace."""
    v = torch.full((40, 40), 1500.0)
    return deepwave.scalar(
        v,
        4.0,
        0.004,
        source_amplitudes=amplitudes,
        source_locations=locations,
        receiver_locations=torch.tensor([[[30, 30]]]),
        boundary=boundary,
        pml_width=[0, 20, 20, 20],
        pml_freq=25.0,
        **kwargs,
    ).receiver_amplitudes


def _wavelet() -> torch.Tensor:
    return deepwave.wavelets.ricker(25.0, 20, 0.004, 0.06).reshape(1, 1, -1)


@pytest.mark.parametrize("condition", ["free", "rigid"])
def test_auto_fold_matches_hand_built_hicks(condition: str) -> None:
    """The automatic fold is bitwise the hand-built ``Hicks`` with that face.

    Non-vacuous: without the fold the same window reaches cell -2, which is
    what the automatic path used to hand to the bounds check (and be rejected
    for), so the two runs cannot agree by both doing nothing.
    """
    locations = torch.tensor([[[1.5, 20.0]]])
    amplitudes = _wavelet()
    boundary = [condition, "pml", "pml", "pml"]
    unfolded = Hicks(locations)
    assert int(unfolded.get_locations().min()) < 0

    hand_built = Hicks(
        locations,
        free_surfaces=[condition == "free", False, False, False],
        rigid_surfaces=[condition == "rigid", False, False, False],
        model_shape=[40, 40],
    )
    expected = _scalar_run(
        hand_built.get_locations(), hand_built.source(amplitudes), boundary
    )
    assert torch.equal(_scalar_run(locations, amplitudes, boundary), expected)


def test_scalar_born_auto_fold_matches_hand_built_hicks() -> None:
    """``scalar_born`` folds the same way ``scalar`` does."""
    locations = torch.tensor([[[1.5, 20.0]]])
    amplitudes = _wavelet()
    v = torch.full((40, 40), 1500.0)
    scatter = torch.zeros(40, 40)
    scatter[20, 20] = 100.0
    hand_built = Hicks(
        locations, free_surfaces=[True, False, False, False], model_shape=[40, 40]
    )

    def run(locs: torch.Tensor, amps: torch.Tensor) -> torch.Tensor:
        return deepwave.scalar_born(
            v,
            4.0,
            0.004,
            dv=scatter,
            source_amplitudes=amps,
            source_locations=locs,
            sc_receiver_locations=torch.tensor([[[30, 30]]]),
            boundary=["free", "pml", "pml", "pml"],
            pml_width=[0, 20, 20, 20],
            pml_freq=25.0,
        ).sc_receiver_amplitudes

    assert torch.equal(
        run(locations, amplitudes),
        run(hand_built.get_locations(), hand_built.source(amplitudes)),
    )


def test_auto_fold_leaves_distant_points_alone() -> None:
    """A point whose window stays inside the model is interpolated as before."""
    locations = torch.tensor([[[20.5, 20.0]]])
    amplitudes = _wavelet()
    with_face = _scalar_run(locations, amplitudes, ["free", "pml", "pml", "pml"])
    hand_built = Hicks(locations)
    assert torch.equal(
        with_face,
        _scalar_run(
            hand_built.get_locations(),
            hand_built.source(amplitudes),
            ["free", "pml", "pml", "pml"],
        ),
    )


def test_periodic_face_is_refused_rather_than_truncated() -> None:
    """``Hicks`` has no periodic wrap, so the window cannot be folded."""
    with pytest.raises(
        deepwave.DeepwaveNotImplementedError, match="no periodic wrap"
    ) as excinfo:
        _scalar_run(
            torch.tensor([[[1.5, 20.0]]]),
            _wavelet(),
            ["periodic", "periodic", "pml", "pml"],
        )
    assert isinstance(excinfo.value, NotImplementedError)
    assert "integer (on-node) coordinate" in str(excinfo.value)


def test_extracted_sub_region_is_refused() -> None:
    """``survey_pad`` moves the face somewhere these coordinates cannot name."""
    with pytest.raises(
        deepwave.DeepwaveNotImplementedError, match="sub-region of the model"
    ):
        _scalar_run(
            torch.tensor([[[1.5, 20.0]]]),
            _wavelet(),
            ["free", "pml", "pml", "pml"],
            survey_pad=5,
        )


def test_differentiable_positions_near_a_reflecting_face_are_refused() -> None:
    """The fold is not differentiable, so it is refused rather than skipped."""
    locations = torch.tensor([[[1.5, 20.0]]], requires_grad=True)
    with pytest.raises(
        deepwave.DeepwaveNotImplementedError, match="requires_grad=True"
    ):
        _scalar_run(locations, _wavelet(), ["free", "pml", "pml", "pml"])


def test_differentiable_positions_away_from_the_face_still_work() -> None:
    """Only a window that actually crosses the face is refused."""
    locations = torch.tensor([[[20.5, 20.0]]], requires_grad=True)
    _scalar_run(locations, _wavelet(), ["free", "pml", "pml", "pml"]).sum().backward()
    assert locations.grad is not None
    assert bool(locations.grad.abs().sum() > 0)


def test_a_staggered_propagator_refuses_a_reflecting_face() -> None:
    """``acoustic``'s components are staggered, so their image is not Hicks'."""
    v = torch.full((40, 40), 1500.0)
    rho = torch.full((40, 40), 1000.0)
    with pytest.raises(
        deepwave.DeepwaveNotImplementedError, match="is not implemented"
    ) as excinfo:
        deepwave.acoustic(
            v,
            rho,
            4.0,
            0.004,
            source_amplitudes_volume_rate=_wavelet(),
            source_locations_volume_rate=torch.tensor([[[1.5, 20.0]]]),
            boundary=["rigid", "pml", "pml", "pml"],
            pml_width=[0, 20, 20, 20],
            pml_freq=25.0,
        )
    assert "boundary='rigid'" in str(excinfo.value)


def test_out_of_bounds_reports_the_coordinate_the_caller_wrote() -> None:
    """Not the expanded window index that first left the model.

    Non-vacuous: the index the un-folded window reaches first (55) is the
    number the message used to carry, and it is no longer in it.
    """
    v = torch.full((40, 40), 1500.0)
    with pytest.raises(RuntimeError) as excinfo:
        deepwave.scalar(
            v,
            4.0,
            0.004,
            source_amplitudes=_wavelet(),
            source_locations=torch.tensor([[[20.0, 20.0]]]),
            receiver_locations=torch.tensor([[[58.5, 30.3]]]),
            pml_freq=25.0,
        )
    message = str(excinfo.value)
    assert "the coordinate is 58.5" in message
    assert "grid cell 58" in message
    assert "only 40 cells" in message
    assert "55" not in message


def test_out_of_bounds_reports_the_stencil_reach() -> None:
    """A point inside the model whose window is not: say which, and why."""
    v = torch.full((40, 40), 1500.0)
    with pytest.raises(RuntimeError) as excinfo:
        deepwave.scalar(
            v,
            4.0,
            0.004,
            source_amplitudes=_wavelet(),
            source_locations=torch.tensor([[[1.5, 20.0]]]),
            pml_freq=25.0,
        )
    message = str(excinfo.value)
    assert "the coordinate is 1.5" in message
    assert "which is inside the model" in message
    assert "halfwidth 4" in message
    assert "reaches grid cell -2" in message
    assert "the beginning of dimension 0" in message


def test_integer_locations_near_a_face_are_untouched() -> None:
    """The exact path has no window, so none of this applies to it."""
    amplitudes = _wavelet()
    integer = _scalar_run(
        torch.tensor([[[1, 20]]]), amplitudes, ["free", "pml", "pml", "pml"]
    )
    on_node_float = _scalar_run(
        torch.tensor([[[1.0, 20.0]]]), amplitudes, ["free", "pml", "pml", "pml"]
    )
    assert torch.equal(integer, on_node_float)
