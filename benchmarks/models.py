"""Turn a `Case` into ready-to-call deepwave propagator inputs.

Each `build_inputs` implementation returns a `BuiltCase`: the propagator
callable, its kwargs, the leaf tensors requiring grad (for `mode ==
"forward_backward"`), and sizing info used to compute Mcells/s.

Models are homogeneous (constant wavespeed/density) -- the FD kernels' cost
depends on grid shape, accuracy order, and dtype, not on the field values, so
a constant model is representative for timing while being cheap and free of
allocation-shape surprises. Most cases use a single source and a handful of
receivers with `n_shots=1`; the CPU kernels parallelise each time step
across the spatial grid, so a single shot already saturates all threads on
large grids. Multi-shot cases (`Case.n_shots > 1`, one source per shot
spread along the last dimension) exist to measure the many-shots/
small-model regime, where per-step spatial parallelism over a small grid
competes with the batch dimension as the parallelism source.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch

import deepwave
from deepwave.wavelets import ricker

try:
    from deepwave.discretisation import cfl_condition_n
except ModuleNotFoundError:
    # The released comparator predates the public discretisation module.
    from deepwave.common import cfl_condition_n, vpvsrho_to_lambmubuoyancy
else:
    from deepwave import vpvsrho_to_lambmubuoyancy

if TYPE_CHECKING:
    from .cases import Case

GRID_SPACING = 5.0
BACKGROUND_V = 1500.0
BACKGROUND_VS = 800.0
BACKGROUND_RHO = 2200.0
SOURCE_FREQ = 15.0


@dataclass
class BuiltCase:
    """A `Case` realised as a ready-to-call propagator function + kwargs."""

    fn: Callable[..., Any]
    kwargs: dict[str, Any]
    grad_tensors: list[torch.Tensor]
    interior_cells: int
    nt: int
    step_ratio: int
    dtype: torch.dtype

    @property
    def nt_effective(self) -> int:
        """Internal time steps actually executed (`nt * step_ratio`)."""
        return self.nt * self.step_ratio


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _pml_width(size: int) -> int:
    return min(10, max(2, size // 8))


def _shape(case: Case) -> list[int]:
    return list(case.grid_shape)


def _grid_spacing(case: Case) -> list[float]:
    return [GRID_SPACING] * case.ndim


def _dt(case: Case, max_vel: float) -> float:
    """A dt that satisfies the CFL condition with step_ratio == 1.

    Picking dt this way (rather than letting the propagator internally
    subdivide) keeps "nt time steps" and "nt internal steps" equal for
    most cases, which makes the reported Mcells/s comparable across
    accuracy orders. `step_ratio` is still measured and reported (see
    `nt_effective`) in case a case ever needs internal subsampling.
    """
    dt_guess = 0.4 * GRID_SPACING / max_vel
    inner_dt, _ = _cfl(_grid_spacing(case), dt_guess, max_vel, 0.5)
    return inner_dt


def _cfl(
    spacing: list[float], dt: float, max_vel: float, courant: float
) -> tuple[float, int]:
    """Use the same Courant target with both versions' public spellings."""
    key = (
        "courant_number"
        if "courant_number" in inspect.signature(cfl_condition_n).parameters
        else "c_max"
    )
    return cfl_condition_n(spacing, dt, max_vel, **{key: courant})


def _center_point(shape: list[int], pml: int) -> list[int]:
    return [max(pml + 1, s // 2) for s in shape]


def _receiver_points(shape: list[int], pml: int, n: int = 4) -> list[list[int]]:
    interior_lo = pml + 1
    points = []
    for i in range(n):
        pt = []
        for d, s in enumerate(shape):
            lo = interior_lo
            hi = max(lo + 1, s - pml - 1)
            offset = (i + d + 1) * (hi - lo) // (n + 1)
            pt.append(min(hi, lo + offset))
        points.append(pt)
    return points


def _source_amplitudes(case: Case, dt: float) -> torch.Tensor:
    wavelet = ricker(
        SOURCE_FREQ,
        case.nt,
        dt,
        min(1.5 / SOURCE_FREQ, case.nt * dt / 2),
        dtype=_dtype(case.dtype),
    )
    return wavelet.reshape(1, 1, -1).repeat(case.n_shots, 1, 1).to(case.device)


def _source_points(shape: list[int], pml: int, n_shots: int) -> list[list[int]]:
    """One source point per shot.

    A single shot uses the grid centre; multiple shots are spread evenly
    along the last (fastest) dimension's interior at the other dims'
    centres, like a surface acquisition line.
    """
    if n_shots == 1:
        return [_center_point(shape, pml)]
    fixed = _center_point(shape, pml)[:-1]
    lo = pml + 1
    hi = max(lo + 1, shape[-1] - pml - 2)
    return [[*fixed, lo + (i + 1) * (hi - lo) // (n_shots + 1)] for i in range(n_shots)]


def _locations(
    points: list[list[int]], device: str = "cpu", n_shots: int = 1
) -> torch.Tensor:
    """[n_points, ndim] -> [n_shots, n_points, ndim] (same points every shot)."""
    return (
        torch.tensor(points, dtype=torch.long, device=device)
        .unsqueeze(0)
        .repeat(n_shots, 1, 1)
    )


def _shot_locations(points: list[list[int]], device: str = "cpu") -> torch.Tensor:
    """[n_shots, ndim] (one point per shot) -> [n_shots, 1, ndim]."""
    return torch.tensor(points, dtype=torch.long, device=device).unsqueeze(1)


def _interior_cells(shape: list[int], pml: int) -> int:
    cells = 1
    for s in shape:
        cells *= max(1, s - 2 * pml)
    return cells


def _build_scalar(case: Case, born: bool) -> BuiltCase:
    dtype = _dtype(case.dtype)
    shape = _shape(case)
    pml = _pml_width(case.size_per_dim)
    max_vel = BACKGROUND_V * 1.05
    dt = _dt(case, max_vel)

    v = torch.full(shape, BACKGROUND_V, dtype=dtype, device=case.device)
    grad_tensors = []
    if case.mode == "forward_backward":
        v.requires_grad_(True)
        grad_tensors.append(v)

    src_pts = _source_points(shape, pml, case.n_shots)
    rec_pts = _receiver_points(shape, pml)
    source_locations = _shot_locations(src_pts, case.device)
    receiver_locations = _locations(rec_pts, case.device, case.n_shots)
    source_amplitudes = _source_amplitudes(case, dt)

    kwargs: dict[str, Any] = {
        "v": v,
        "grid_spacing": _grid_spacing(case),
        "dt": dt,
        "source_amplitudes": source_amplitudes,
        "source_locations": source_locations,
        "receiver_locations": receiver_locations,
        "accuracy": case.accuracy,
        "pml_width": pml,
        "max_vel": max_vel,
        "pml_freq": SOURCE_FREQ,
        "storage_mode": case.storage_mode,
        "storage_compression": case.storage_compression,
        "backend": case.backend,
    }
    fn: Callable[..., Any] = deepwave.scalar

    if born:
        scatter = torch.zeros(shape, dtype=dtype, device=case.device)
        scatter[tuple(size // 2 for size in shape)] = 0.01 * BACKGROUND_V
        if case.mode == "forward_backward":
            scatter.requires_grad_(True)
            grad_tensors.append(scatter)
        perturbation_name = (
            "dv"
            if "dv" in inspect.signature(deepwave.scalar_born).parameters
            else "scatter"
        )
        kwargs[perturbation_name] = scatter
        fn = deepwave.scalar_born
        kwargs["bg_receiver_locations"] = receiver_locations
        if "sc_receiver_locations" in inspect.signature(fn).parameters:
            kwargs["sc_receiver_locations"] = kwargs.pop("receiver_locations")

    _, step_ratio = _cfl(_grid_spacing(case), dt, max_vel, 0.6)
    return BuiltCase(
        fn=fn,
        kwargs=kwargs,
        grad_tensors=grad_tensors,
        interior_cells=_interior_cells(shape, pml) * case.n_shots,
        nt=case.nt,
        step_ratio=step_ratio,
        dtype=dtype,
    )


def _build_acoustic(case: Case) -> BuiltCase:
    dtype = _dtype(case.dtype)
    shape = _shape(case)
    pml = _pml_width(case.size_per_dim)
    max_vel = BACKGROUND_V * 1.05
    dt = _dt(case, max_vel)

    v = torch.full(shape, BACKGROUND_V, dtype=dtype, device=case.device)
    rho = torch.full(shape, BACKGROUND_RHO, dtype=dtype, device=case.device)
    grad_tensors = []
    if case.mode == "forward_backward":
        v.requires_grad_(True)
        rho.requires_grad_(True)
        grad_tensors.extend([v, rho])

    src_pts = _source_points(shape, pml, case.n_shots)
    rec_pts = _receiver_points(shape, pml)
    source_amplitudes_volume_rate = _source_amplitudes(case, dt)

    kwargs: dict[str, Any] = {
        "v": v,
        "rho": rho,
        "grid_spacing": _grid_spacing(case),
        "dt": dt,
        "source_amplitudes_volume_rate": source_amplitudes_volume_rate,
        "source_locations_volume_rate": _shot_locations(src_pts, case.device),
        "receiver_locations_p": _locations(rec_pts, case.device, case.n_shots),
        "accuracy": case.accuracy,
        "pml_width": pml,
        "max_vel": max_vel,
        "pml_freq": SOURCE_FREQ,
        "storage_mode": case.storage_mode,
        "storage_compression": case.storage_compression,
        "backend": case.backend,
    }
    _, step_ratio = _cfl(_grid_spacing(case), dt, max_vel, 0.6)
    return BuiltCase(
        fn=deepwave.acoustic,
        kwargs=kwargs,
        grad_tensors=grad_tensors,
        interior_cells=_interior_cells(shape, pml) * case.n_shots,
        nt=case.nt,
        step_ratio=step_ratio,
        dtype=dtype,
    )


def _build_elastic(case: Case) -> BuiltCase:
    dtype = _dtype(case.dtype)
    shape = _shape(case)
    pml = _pml_width(case.size_per_dim)
    max_vel = BACKGROUND_V * 1.1
    dt = _dt(case, max_vel)

    vp = torch.full(shape, BACKGROUND_V, dtype=dtype, device=case.device)
    vs = torch.full(shape, BACKGROUND_VS, dtype=dtype, device=case.device)
    rho = torch.full(shape, BACKGROUND_RHO, dtype=dtype, device=case.device)
    lamb, mu, buoyancy = vpvsrho_to_lambmubuoyancy(vp, vs, rho)
    lamb, mu, buoyancy = lamb.contiguous(), mu.contiguous(), buoyancy.contiguous()

    grad_tensors = []
    if case.mode == "forward_backward":
        lamb.requires_grad_(True)
        mu.requires_grad_(True)
        buoyancy.requires_grad_(True)
        grad_tensors.extend([lamb, mu, buoyancy])

    src_pts = _source_points(shape, pml, case.n_shots)
    rec_pts = _receiver_points(shape, pml)
    source_amplitudes_x = _source_amplitudes(case, dt)

    # The "x" component (the fastest/last spatial dimension) exists for
    # every ndim in {1, 2, 3}; "z" only exists once ndim == 3 (see
    # elastic._dim_names: ["z", "y", "x"][-ndim:]).
    kwargs: dict[str, Any] = {
        "lamb": lamb,
        "mu": mu,
        "buoyancy": buoyancy,
        "grid_spacing": _grid_spacing(case),
        "dt": dt,
        "source_amplitudes_x": source_amplitudes_x,
        "source_locations_x": _shot_locations(src_pts, case.device),
        "receiver_locations_x": _locations(rec_pts, case.device, case.n_shots),
        "accuracy": case.accuracy,
        "pml_width": pml,
        "max_vel": max_vel,
        "pml_freq": SOURCE_FREQ,
        "storage_mode": case.storage_mode,
        "storage_compression": case.storage_compression,
        "backend": case.backend,
    }
    _, step_ratio = _cfl(_grid_spacing(case), dt, max_vel, 0.6)
    return BuiltCase(
        fn=deepwave.elastic,
        kwargs=kwargs,
        grad_tensors=grad_tensors,
        interior_cells=_interior_cells(shape, pml) * case.n_shots,
        nt=case.nt,
        step_ratio=step_ratio,
        dtype=dtype,
    )


def _build_inputs(case: Case) -> BuiltCase:
    """Dispatch to the propagator-specific builder for `case`."""
    if case.propagator == "scalar":
        return _build_scalar(case, born=False)
    if case.propagator == "scalar_born":
        return _build_scalar(case, born=True)
    if case.propagator == "acoustic":
        return _build_acoustic(case)
    if case.propagator == "elastic":
        return _build_elastic(case)
    if case.propagator in ("acoustic_born", "viscoacoustic"):
        built = _build_acoustic(case)
        kwargs = built.kwargs
        built.fn = getattr(deepwave, case.propagator)
        if case.propagator == "acoustic_born":
            for name, background in (("dv", "v"), ("drho", "rho")):
                perturbation = (0.01 * kwargs[background].detach()).requires_grad_(
                    case.mode == "forward_backward"
                )
                kwargs[name] = perturbation
                if perturbation.requires_grad:
                    built.grad_tensors.append(perturbation)
            locations = kwargs.pop("receiver_locations_p")
            kwargs.update(
                bg_receiver_locations_p=locations, sc_receiver_locations_p=locations
            )
        else:
            q = torch.full_like(
                kwargs["v"], 60.0, requires_grad=case.mode == "forward_backward"
            )
            kwargs.update(
                q=q, q_n_mechanisms=3, q_band=(SOURCE_FREQ / 2, 2 * SOURCE_FREQ)
            )
            if q.requires_grad:
                built.grad_tensors.append(q)
            kwargs["source_amplitudes_pressure_rate"] = kwargs.pop(
                "source_amplitudes_volume_rate"
            ) * (BACKGROUND_RHO * BACKGROUND_V**2)
            kwargs["source_locations_pressure_rate"] = kwargs.pop(
                "source_locations_volume_rate"
            )
        return built
    if case.propagator == "elastic_born":
        built = _build_elastic(case)
        built.fn = deepwave.elastic_born
        for name in ("lamb", "mu", "buoyancy"):
            perturbation = (0.01 * built.kwargs[name].detach()).requires_grad_(
                case.mode == "forward_backward"
            )
            built.kwargs[f"d{name}"] = perturbation
            if perturbation.requires_grad:
                built.grad_tensors.append(perturbation)
        locations = built.kwargs.pop("receiver_locations_x")
        built.kwargs.update(
            bg_receiver_locations_x=locations, sc_receiver_locations_x=locations
        )
        return built
    if case.propagator in ("elastic_vti", "acoustic_vti"):
        built = _build_elastic(case)
        kwargs = built.kwargs
        lamb, mu = kwargs.pop("lamb"), kwargs.pop("mu")
        c33 = lamb.detach() + 2 * mu.detach()
        material = {"c11": 1.2 * c33, "c13": 0.7 * c33, "c33": c33}
        if case.propagator == "elastic_vti":
            material["c55"] = mu.detach().clone()
            if case.ndim == 3:
                material["c66"] = 1.1 * mu.detach()
        built.grad_tensors = []
        for name, model in {**material, "buoyancy": kwargs["buoyancy"]}.items():
            model.requires_grad_(case.mode == "forward_backward")
            kwargs[name] = model
            if model.requires_grad:
                built.grad_tensors.append(model)
        kwargs["max_vel"] = BACKGROUND_V * 1.5
        kwargs["dt"] = _dt(case, kwargs["max_vel"])
        kwargs["source_amplitudes_x"] = _source_amplitudes(case, kwargs["dt"])
        built.fn = getattr(deepwave, case.propagator)
        return built
    raise ValueError(f"Unknown propagator {case.propagator!r}")


def build_inputs(case: Case) -> BuiltCase:
    """Build identical physical inputs under the candidate and released APIs."""
    built = _build_inputs(case)
    parameters = inspect.signature(built.fn).parameters
    if "backend" not in parameters:
        backend = built.kwargs.pop("backend")
        built.kwargs["python_backend"] = {
            "compiled": False,
            "eager": "eager",
            "torch_compile": "compile",
        }[backend]
    if (
        "source_amplitudes_volume_rate" in built.kwargs
        and "source_amplitudes_volume_rate" not in parameters
    ):
        built.kwargs["source_amplitudes_p"] = built.kwargs.pop(
            "source_amplitudes_volume_rate"
        )
        built.kwargs["source_locations_p"] = built.kwargs.pop(
            "source_locations_volume_rate"
        )
    return built
