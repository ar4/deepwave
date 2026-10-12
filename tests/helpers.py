"""Shared test helpers, imported by name -- NOT a pytest test module.

This file deliberately does not match pytest's ``test_*.py`` collection
pattern, so it is never collected as a suite of tests itself; it exists so
several test files can share one implementation of small, general-purpose
helpers (analytic Green's-function references, source/coordinate builders,
and cross-backend equivalence checks) instead of each keeping its own copy.

``tests/test_utils.py`` (which unit-tests these functions) imports them from
here, as do ``test_acoustic.py``, ``test_elastic.py``, ``test_scalar.py``,
``test_scalar_born.py``, and ``test_location_interpolation.py``.
"""

import math
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import scipy.special
import torch

import deepwave
import deepwave._backend_utils
from deepwave.wavelets import ricker


def element_tensors(value: Any) -> list[torch.Tensor]:
    """The tensors held by one element of a propagator's output tuple.

    Every element is a ``torch.Tensor`` except ``auxiliary_state``, which is a
    ``deepwave.AuxiliaryState`` holding several. Returning a list either way
    lets a caller treat the output uniformly.
    """
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, deepwave.AuxiliaryState):
        return list(value.values())
    raise TypeError(f"Unexpected propagator output element {type(value).__name__}.")


def output_tensors(out: Sequence[Any]) -> list[torch.Tensor]:
    """Every tensor of a propagator output, with the auxiliary state expanded."""
    tensors: list[torch.Tensor] = []
    for value in out:
        tensors.extend(element_tensors(value))
    return tensors


def auxiliary_state_arg(
    **tensors: Optional[torch.Tensor],
) -> Optional[dict[str, Any]]:
    """An ``initial_state`` mapping carrying the given auxiliary tensors.

    ``None`` when none of them was supplied, so the call starts that part of
    the state from rest. The fingerprint is left unrecorded: a hand-built
    auxiliary state has no run behind it, so there is nothing to check it
    against, and the existing device/dtype/shape checks still fire.

    The keys are the internal flat names, which is what
    ``AuxiliaryState._from_flat`` takes; passing those same names as top-level
    ``initial_state`` keys is the deprecated per-field path instead.
    """
    supplied = {name: t for name, t in tensors.items() if t is not None}
    if not supplied:
        return None
    return {"auxiliary_state": deepwave.AuxiliaryState._from_flat(supplied)}


def elements_equal(a: Any, b: Any) -> bool:
    """``torch.equal`` extended to an ``auxiliary_state`` element."""
    tensors_a = element_tensors(a)
    tensors_b = element_tensors(b)
    if isinstance(a, deepwave.AuxiliaryState) != isinstance(b, deepwave.AuxiliaryState):
        return False
    if isinstance(a, deepwave.AuxiliaryState) and a.names != b.names:
        return False
    return len(tensors_a) == len(tensors_b) and all(
        torch.equal(x, y) for x, y in zip(tensors_a, tensors_b)
    )


def c_backend_available(
    propagator: str,
    ndim: int = 2,
    direction: str = "forward",
    accuracy: int = 2,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> bool:
    """Whether the compiled ``propagator`` symbols exist in the loaded DLL.

    A single parametrised implementation of the ``_c_backend_available()``
    helper that ``test_elastic_vti.py`` and ``test_viscoacoustic.py`` each used
    to define separately (identical except for the propagator name).
    """
    if device is None:
        device = torch.device("cpu")
    try:
        deepwave._backend_utils.get_backend_function(
            propagator, ndim, direction, accuracy, dtype, device
        )
    except AttributeError:
        return False
    return True


def _set_sources(
    x_s: torch.Tensor,
    freq: Union[int, float],
    dt: Union[int, float],
    nt: int,
    dtype: Optional[torch.dtype] = None,
    dpeak_time: float = 0.3,
) -> dict[str, torch.Tensor]:
    """Create sources with amplitudes that have randomly shifted start times."""
    if not isinstance(x_s, torch.Tensor):
        raise TypeError("x_s must be a torch.Tensor.")
    if x_s.ndim != 3:
        raise RuntimeError("x_s must have three dimensions.")
    if not isinstance(freq, (int, float)):
        raise TypeError("freq must be a number.")
    if freq <= 0:
        raise ValueError("freq must be positive.")
    if not isinstance(dt, (int, float)):
        raise TypeError("dt must be a number.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if not isinstance(nt, int):
        raise TypeError("nt must be an int.")
    if nt <= 0:
        raise ValueError("nt must be positive.")
    if dtype is not None and not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype.")
    if dpeak_time < 0:
        raise ValueError("dpeak_time must be non-negative.")

    num_shots, num_sources_per_shot = x_s.shape[:2]
    sources = {}
    sources["amplitude"] = torch.zeros(num_shots, num_sources_per_shot, nt, dtype=dtype)

    sources["locations"] = x_s

    for shot in range(num_shots):
        for source in range(num_sources_per_shot):
            peak_time = 0.05 + torch.rand(1).item() * dpeak_time
            sources["amplitude"][shot, source, :] = ricker(
                freq,
                nt,
                dt,
                peak_time,
                dtype=dtype,
            )
    return sources


def _set_coords(
    num_shots: int,
    num_per_shot: int,
    nx: tuple[int, ...],
    location: str = "top",
) -> torch.Tensor:
    """Create an array of coordinates at the specified location."""
    if not isinstance(num_shots, int):
        raise TypeError("num_shots must be an int.")
    if num_shots <= 0:
        raise ValueError("num_shots must be positive.")
    if not isinstance(num_per_shot, int):
        raise TypeError("num_per_shot must be an int.")
    if num_per_shot <= 0:
        raise ValueError("num_per_shot must be positive.")
    if not isinstance(nx, (list, tuple)):
        raise TypeError("nx must be a list or tuple.")
    if not nx:
        raise ValueError("nx must not be empty.")
    for dim_size in nx:
        if not isinstance(dim_size, int):
            raise TypeError("nx elements must be integers.")
        if dim_size <= 0:
            raise ValueError("nx elements must be positive.")

    ndim = len(nx)
    coords = torch.zeros(num_shots, num_per_shot, ndim)
    coords[..., 0] = torch.arange(num_shots * num_per_shot).reshape(
        num_shots,
        num_per_shot,
    )
    if location == "top":
        pass
    elif location == "bottom":
        coords[..., 0] = float(nx[0] - 2) - coords[..., 0]
    elif location == "middle":
        coords[..., 0] += int(float(nx[0]) / 2)
    else:
        raise ValueError("unsupported location")

    for dim in range(1, ndim):
        coords[..., dim] = torch.round(torch.tensor(float(nx[dim]) / 2))

    return coords.long()


def _shift(f: torch.Tensor, t_shift: float) -> torch.Tensor:
    """Shift a 1D tensor by a fractional amount in time."""
    nt = len(f)
    w = torch.fft.rfftfreq(nt, 1.0)
    fw = torch.fft.rfft(f)
    fw_shifted = fw * torch.exp(-1j * 2 * torch.pi * w * t_shift)
    return torch.fft.irfft(fw_shifted, nt)


def direct_1d(
    x: torch.Tensor,
    x_s: torch.Tensor,
    dx: torch.Tensor,
    dt: Union[int, float],
    c: Union[int, float],
    f: torch.Tensor,
) -> torch.Tensor:
    """Calculate receiver data using the 1D Green's function."""
    r = torch.abs(x - x_s).item() * dx.item()
    t_shift = r / c / dt
    # The analytical solution involves a scaled, time-shifted,
    # cumulative sum of the source.
    f_cumsum = torch.cumsum(f, dim=0)
    f_shifted_cumsum = _shift(f_cumsum, t_shift)
    return f_shifted_cumsum * dx.item() * dt * c / 2


def direct_2d_approx(
    x: torch.Tensor,
    x_s: torch.Tensor,
    dx: torch.Tensor,
    dt: Union[int, float],
    c: Union[int, float],
    f: torch.Tensor,
) -> torch.Tensor:
    """Calculates the 2D Green's function approximation for wavefield at a location.

    This function uses an approximation of the 2D Green's function to calculate the
    wavefield at a given location due to the given source.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor.")
    if x.ndim != 1 or x.shape[0] != 2:
        raise RuntimeError("x must be a 1D Tensor of length 2.")
    if not isinstance(x_s, torch.Tensor):
        raise TypeError("x_s must be a torch.Tensor.")
    if x_s.ndim != 1 or x_s.shape[0] != 2:
        raise RuntimeError("x_s must be a 1D Tensor of length 2.")
    if not isinstance(dx, torch.Tensor):
        raise TypeError("dx must be a torch.Tensor.")
    if dx.ndim != 1 or dx.shape[0] != 2:
        raise RuntimeError("dx must be a 1D Tensor of length 2.")
    if not isinstance(dt, (int, float)):
        raise TypeError("dt must be a number.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if not isinstance(c, (int, float)):
        raise TypeError("c must be a number.")
    if c <= 0:
        raise ValueError("c must be positive.")
    if not isinstance(f, torch.Tensor):
        raise TypeError("f must be a torch.Tensor.")
    if f.ndim != 1:
        raise RuntimeError("f must be a 1D Tensor.")

    r = torch.norm(x.double() * dx - x_s.double() * dx).item()
    nt = len(f)
    w = torch.fft.rfftfreq(nt, dt)
    fw = torch.fft.rfft(f)
    h = scipy.special.hankel1(0, -2 * math.pi * w.numpy() * r / c)
    g = 1j / 4 * torch.tensor(h)
    g[0] = 0
    s = g * fw * torch.prod(dx).item()
    return torch.fft.irfft(s, nt)


def direct_3d(
    x: torch.Tensor,
    x_s: torch.Tensor,
    dx: torch.Tensor,
    dt: float,
    c: float,
    f: torch.Tensor,
) -> torch.Tensor:
    """Calculate receiver data using the 3D Green's function."""
    r = torch.norm(x.double() * dx - x_s.double() * dx).item()
    t_shift = (r / c) / dt
    return torch.prod(dx) / 4 / torch.pi / r * _shift(f, t_shift)


def scattered(
    x: torch.Tensor,
    x_s: torch.Tensor,
    x_p: torch.Tensor,
    dx: torch.Tensor,
    dt: Union[int, float],
    c: Union[int, float],
    dc: Union[int, float],
    f: torch.Tensor,
    ndim: int,
) -> torch.Tensor:
    """Calculate the scattered wavefield at a given location."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor.")
    if x.ndim != 1 or x.shape[0] != ndim:
        raise RuntimeError("x must be a 1D Tensor of length ndim.")
    if not isinstance(x_s, torch.Tensor):
        raise TypeError("x_s must be a torch.Tensor.")
    if x_s.ndim != 1 or x_s.shape[0] != ndim:
        raise RuntimeError("x_s must be a 1D Tensor of length ndim.")
    if not isinstance(x_p, torch.Tensor):
        raise TypeError("x_p must be a torch.Tensor.")
    if x_p.ndim != 1 or x_p.shape[0] != ndim:
        raise RuntimeError("x_p must be a 1D Tensor of length ndim.")
    if not isinstance(dx, torch.Tensor):
        raise TypeError("dx must be a torch.Tensor.")
    if dx.ndim != 1 or dx.shape[0] != ndim:
        raise RuntimeError("dx must be a 1D Tensor of length ndim.")
    if not isinstance(dt, (int, float)):
        raise TypeError("dt must be a number.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if not isinstance(c, (int, float)):
        raise TypeError("c must be a number.")
    if c <= 0:
        raise ValueError("c must be positive.")
    if not isinstance(dc, (int, float)):
        raise TypeError("dc must be a number.")
    if not isinstance(f, torch.Tensor):
        raise TypeError("f must be a torch.Tensor.")
    if f.ndim != 1:
        raise RuntimeError("f must be a 1D Tensor.")
    if not isinstance(ndim, int):
        raise TypeError("ndim must be an integer.")
    if ndim < 1 or ndim > 3:
        raise ValueError("ndim must be 1, 2, or 3.")

    if ndim == 1:
        direct = direct_1d
    elif ndim == 2:
        direct = direct_2d_approx
    else:
        direct = direct_3d

    u_p = direct(x_p, x_s, dx, dt, c, f)
    du_pdt2 = _second_deriv(u_p, dt)
    return 2 * dc / c**3 * direct(x, x_p, dx, dt, c, du_pdt2)


def _second_deriv(arr: torch.Tensor, dt: Union[int, float]) -> torch.Tensor:
    """Calculate the second derivative."""
    if not isinstance(arr, torch.Tensor):
        raise TypeError("arr must be a torch.Tensor.")
    if arr.ndim != 1:
        raise RuntimeError("arr must be a 1D Tensor.")
    if arr.numel() < 3:
        raise ValueError(
            "arr must have at least 3 elements for second derivative calculation.",
        )
    if not isinstance(dt, (int, float)):
        raise TypeError("dt must be a number.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    d2dt2 = torch.zeros_like(arr)
    d2dt2[1:-1] = (arr[2:] - 2 * arr[1:-1] + arr[:-2]) / dt**2
    d2dt2[0] = d2dt2[1]
    d2dt2[-1] = d2dt2[-2]
    return d2dt2


def compute_backend_parity_nt(nx, dx, min_speed, freq, dt, nt_add=0):
    """Number of time steps for a backend-parity fixture.

    Long enough for two round trips across the model at ``min_speed`` plus
    a peak-time/frequency margin; falls back to 1500 m/s if ``min_speed``
    is exactly zero (e.g. an all-zero velocity perturbation).
    """
    speed = min_speed if min_speed != 0 else 1500
    nt = int((2 * torch.norm(nx.float() * dx) / speed + 0.1 + 2 / freq) / dt)
    return nt + nt_add


def _normalize_pml_width(pml_width: Union[int, list[int]], ndim: int) -> list[int]:
    """Expand a scalar PML width into the ``2 * ndim`` per-side list expected."""
    if isinstance(pml_width, int):
        return [pml_width for _ in range(2 * ndim)]
    return pml_width


def _default_wavefield_size(
    nx: torch.Tensor, pml_width: list[int], ndim: int
) -> tuple[int, ...]:
    """Default wavefield shape (excluding the shot dimension): model size plus PML."""
    return tuple(nx[i] + pml_width[2 * i] + pml_width[2 * i + 1] for i in range(ndim))


def _requires_grad_if_present(
    tensors: Sequence[Optional[torch.Tensor]], requires_grad: bool
) -> None:
    """Set ``requires_grad`` in place on every non-``None`` tensor in ``tensors``."""
    for t in tensors:
        if t is not None:
            t.requires_grad_(requires_grad)


def _zero_scalar_style_wavefields(
    provide_wavefields: bool,
    wavefield_size: Optional[tuple[int, ...]],
    num_shots: int,
    ndim: int,
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
    wavefield_0_requires_grad: bool,
    wavefield_m1_requires_grad: bool,
    psi_m1_requires_grad: bool,
    zeta_m1_requires_grad: bool,
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    list[Optional[torch.Tensor]],
    list[Optional[torch.Tensor]],
]:
    """Build the zero wavefield_0/wavefield_m1/psi_m1/zeta_m1 set.

    Shared by the scalar and scalar_born propagators' backend-parity tests.
    scalar_born calls this twice per test (once for the background field, once
    for the scattered field, with independent ``requires_grad`` flags and
    variable names at the call site). ``psi_m1``/``zeta_m1`` are always
    returned left-padded with ``None`` to length 3, matching the propagators'
    fixed 3-slot (z, y, x) PML-memory-variable argument convention regardless
    of ``ndim``.
    """
    if provide_wavefields:
        assert wavefield_size is not None
        wavefield_0 = torch.zeros(
            num_shots, *wavefield_size, dtype=dtype, device=device
        )
        wavefield_m1 = torch.zeros_like(wavefield_0)
        psi_m1: list[Optional[torch.Tensor]] = [
            torch.zeros_like(wavefield_0) for _ in range(ndim)
        ]
        zeta_m1: list[Optional[torch.Tensor]] = [
            torch.zeros_like(wavefield_0) for _ in range(ndim)
        ]
        wavefield_0.requires_grad_(wavefield_0_requires_grad)
        wavefield_m1.requires_grad_(wavefield_m1_requires_grad)
        for i in range(ndim):
            psi_m1[i].requires_grad_(psi_m1_requires_grad)  # type: ignore[union-attr]
            zeta_m1[i].requires_grad_(zeta_m1_requires_grad)  # type: ignore[union-attr]
    else:
        wavefield_0 = None
        wavefield_m1 = None
        psi_m1 = []
        zeta_m1 = []

    psi_m1 = [None] * (3 - len(psi_m1)) + psi_m1
    zeta_m1 = [None] * (3 - len(zeta_m1)) + zeta_m1
    return wavefield_0, wavefield_m1, psi_m1, zeta_m1


def _build_multicomponent_sources(
    key_order: list[str],
    x_s: torch.Tensor,
    freq: Union[int, float],
    dt: Union[int, float],
    nt: int,
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
    ndim: int,
    source_requires_grad: bool,
) -> tuple[dict[str, Optional[torch.Tensor]], dict[str, Optional[torch.Tensor]]]:
    """Build the per-component source amplitude/location dicts.

    Shared by the acoustic and elastic velocity-stress propagators'
    backend-parity tests. ``key_order`` fixes the returned dicts' key order --
    which must match each propagator's keyword-argument order -- and
    pre-populates every key with ``None``. A source is always created for the
    "x" and "p" components; "y" is added when ``ndim >= 2`` and "z" when
    ``ndim >= 3``.
    """
    source_amplitudes: dict[str, Optional[torch.Tensor]] = {
        f"source_amplitudes_{k}": None for k in key_order
    }
    source_locations: dict[str, Optional[torch.Tensor]] = {
        f"source_locations_{k}": None for k in key_order
    }

    def _add(component: str) -> None:
        s = _set_sources(x_s, freq, dt, nt, dtype, dpeak_time=0.05)
        s["amplitude"] = s["amplitude"].to(device)
        s["amplitude"].requires_grad_(source_requires_grad)
        source_amplitudes[f"source_amplitudes_{component}"] = s["amplitude"]
        source_locations[f"source_locations_{component}"] = s["locations"]

    _add("x")
    _add("p")
    if ndim >= 2:
        _add("y")
    if ndim >= 3:
        _add("z")
    return source_amplitudes, source_locations


def _build_multicomponent_receivers(
    key_order: list[str], x_r: Optional[torch.Tensor], ndim: int
) -> dict[str, Optional[torch.Tensor]]:
    """Build the per-component receiver-location dict.

    Shared by the acoustic and elastic velocity-stress propagators'
    backend-parity tests. Companion to :func:`_build_multicomponent_sources`:
    same "x"/"p" always, "y" if ``ndim >= 2``, "z" if ``ndim >= 3``
    convention, all pointed at the same ``x_r`` coordinates.
    """
    receiver_locations: dict[str, Optional[torch.Tensor]] = {
        f"receiver_locations_{k}": None for k in key_order
    }
    receiver_locations["receiver_locations_x"] = x_r
    receiver_locations["receiver_locations_p"] = x_r
    if ndim >= 2:
        receiver_locations["receiver_locations_y"] = x_r
    if ndim >= 3:
        receiver_locations["receiver_locations_z"] = x_r
    return receiver_locations


def _select_requiring_grad(
    seq: Sequence[torch.Tensor], idxs: list[int]
) -> list[torch.Tensor]:
    """Cache (in ``idxs``) and select the elements of ``seq`` with ``requires_grad``.

    ``idxs`` is a mutable list owned by the caller: populated the first time this
    is called, left untouched afterwards, so repeated calls (e.g. once per backend
    in a Python-vs-compiled comparison) select the same indices.
    """
    if not idxs:
        idxs.extend(i for i, x in enumerate(seq) if x.requires_grad)
    return [seq[i] for i in idxs]


def compute_backend_parity_grad(
    out: Sequence[torch.Tensor],
    inputs: Sequence[Any],
    out_idxs: list[int],
) -> tuple[torch.Tensor, ...]:
    """Select requires-grad outputs and return them plus their gradient.

    Shared body of the acoustic, elastic, and scalar_born backend-parity
    ``wrap()`` closures (none of which do a second-order gradgrad pass, unlike
    scalar's -- see :func:`compute_backend_parity_grad_and_gradgrad`). Draws
    random cotangents for ``out`` (after caching/reusing which indices of
    ``out`` require grad in ``out_idxs``) and differentiates w.r.t. every
    ``requires_grad`` tensor in ``inputs``.
    """
    out_sel = _select_requiring_grad(out, out_idxs)
    v = [torch.randn_like(o.contiguous()) for o in out_sel]
    leaves = [p for p in inputs if isinstance(p, torch.Tensor) and p.requires_grad]
    grads = torch.autograd.grad(out_sel, leaves, grad_outputs=v)
    return tuple(out_sel) + grads


def compute_backend_parity_grad_and_gradgrad(
    out: Sequence[torch.Tensor],
    inputs: Sequence[Any],
    out_idxs: list[int],
    gradgrad: bool,
    grad_idxs: list[int],
) -> tuple[torch.Tensor, ...]:
    """Scalar-family variant with an optional second-order (gradgrad) pass.

    Companion to :func:`compute_backend_parity_grad`. When ``gradgrad`` is
    False, returns *only* the first-order gradient tuple (deliberately not
    prefixed with the selected outputs): some backend-parity tests need this
    to get one real comparison in configurations where the second-order
    gradient would be identically zero -- see
    ``test_backend_parity_only_source_2d`` in test_scalar.py for the full
    rationale. When ``gradgrad`` is True, also differentiates that
    first-order gradient a second time and returns ``tuple(selected outputs)
    + first-order grads + second-order grads``.
    """
    out_sel = _select_requiring_grad(out, out_idxs)
    v = [torch.randn_like(o.contiguous()) for o in out_sel]
    leaves = [p for p in inputs if isinstance(p, torch.Tensor) and p.requires_grad]
    grads = torch.autograd.grad(out_sel, leaves, grad_outputs=v, create_graph=gradgrad)
    if not gradgrad:
        return grads

    grads_sel = _select_requiring_grad(grads, grad_idxs)
    if len(grads_sel) == 0:
        # A leaf whose only requiring-grad path through the propagator is
        # linear (e.g. source_amplitudes or wavefield_0 alone, with every
        # other leaf fixed) has an identically-zero second derivative, so
        # grad_idxs is empty here. Returning grads (empty) at this point
        # would make the caller's comparison loop silently compare zero
        # tensor pairs, passing regardless of whether the primal outputs
        # or the first-order gradient actually agree between backends.
        # Fail loudly instead: this configuration must pass
        # gradgrad=False to get a real comparison; see
        # test_backend_parity_only_source_2d.
        raise AssertionError(
            "run_backend_parity: gradgrad=True but every first-order "
            "gradient is non-requiring-grad (identically linear in the "
            "leaves that require grad), so there is nothing left to "
            "compare for the second-order pass. Pass gradgrad=False for "
            "this configuration instead."
        )
    w = [torch.randn_like(g.contiguous()) for g in grads_sel]
    grad_grads = torch.autograd.grad(grads_sel, leaves, grad_outputs=w)
    return tuple(out_sel) + tuple(grads_sel) + grad_grads


def run_python_vs_compiled_backend_parity(wrap, python_flag, atol=1e-8, rtol=1e-5):
    """Run ``wrap`` under the Python and compiled backends and compare.

    ``wrap`` must accept a single ``backend`` argument (the value to set
    ``backend`` to, e.g. ``"eager"``) and return a tuple of tensors to compare
    elementwise. Seeds are reset before each call so both backends draw the
    same ``torch.randn_like`` cotangents.
    """
    torch.manual_seed(1)
    out_python = wrap(python_flag)
    torch.manual_seed(1)
    out_compiled = wrap("compiled")
    for oc, op in zip(out_compiled, out_python):
        assert torch.allclose(oc, op, atol=atol, rtol=rtol)


def assert_compiled_matches_eager_grad(f, args, compiled_backend="compiled", tol=1e-6):
    """Compare compiled- and eager-backend gradients of ``f`` at ``args``.

    ``f`` must accept ``(*args, backend)`` and return a tensor to sum
    before differentiating. ``compiled_backend`` is the value passed for the
    non-eager side (usually the default ``"compiled"``; some call sites pass
    an explicit non-default mode instead, so this is a parameter, not
    hardcoded).
    """
    g_c = torch.autograd.grad(f(*args, compiled_backend).sum(), args)
    args2 = tuple(a.detach().requires_grad_(True) for a in args)
    g_e = torch.autograd.grad(f(*args2, "eager").sum(), args2)
    for a, b in zip(g_c, g_e):
        assert (a - b).abs().max() < tol


def assert_compiled_matches_eager_forward_and_grad(
    f, args, compiled_backend="compiled", tol=1e-6
):
    """Compare compiled- vs eager-backend forward output and full gradient tuple.

    Unlike :func:`assert_compiled_matches_eager_grad`, this also checks the
    forward output (relative to the eager output's max magnitude, matching
    the gradient check's own relative-tolerance formula) and returns the
    compiled-side gradient tuple ``g_c`` so callers can reuse it for a
    downstream finite-difference arbitration. ``f`` must accept ``(*args,
    backend)`` and return a tensor to sum before differentiating.
    ``compiled_backend`` is the value passed for the non-eager side (usually
    the default ``"compiled"``; some call sites pass an explicit non-default
    mode instead, so this is a parameter, not hardcoded).
    """
    out_c = f(*args, compiled_backend)
    g_c = torch.autograd.grad(out_c.sum(), args)
    args2 = tuple(a.detach().requires_grad_(True) for a in args)
    out_e = f(*args2, "eager")
    g_e = torch.autograd.grad(out_e.sum(), args2)
    assert (out_c - out_e).abs().max() <= tol * out_e.abs().max()
    for a, b in zip(g_c, g_e):
        assert (a - b).abs().max() <= tol * b.abs().max().clamp_min(1e-300)
    return g_c


def assert_compiled_matches_eager_forward_and_grad_on_device(
    f, args, device, compiled_backend="compiled", tol=1e-6
):
    """Device sibling of ``assert_compiled_matches_eager_forward_and_grad``.

    ``f`` must accept ``(*args, backend, dev="cpu")``: the compiled
    side runs on ``device``; the eager side always runs on CPU, and
    compiled-side outputs/gradients are moved to CPU before comparison.
    Returns the (still ``device``-resident) compiled-side gradient tuple
    ``g_c`` so callers can reuse it for a downstream finite-difference
    arbitration.
    """
    out_c = f(*args, compiled_backend, device)
    g_c = torch.autograd.grad(out_c.sum(), args)
    args2 = tuple(a.detach().requires_grad_(True) for a in args)
    out_e = f(*args2, "eager")
    g_e = torch.autograd.grad(out_e.sum(), args2)
    assert (out_c.cpu() - out_e).abs().max() <= tol * out_e.abs().max()
    for a, b in zip(g_c, g_e):
        assert (a.cpu() - b).abs().max() <= tol * b.abs().max().clamp_min(1e-300)
    return g_c


def run_reciprocity_check(
    models,
    dx,
    dt,
    source_amplitudes,
    src_comp,
    rec_comp,
    src_idx,
    rec_idx,
    loc_a,
    loc_b,
    prop_kwargs,
    propagator,
    receiver_kwarg="receiver_locations",
    atol=5e-3,
    source_comps=None,
):
    """Helper for test_reciprocity.

    ``receiver_kwarg`` is the receiver-locations keyword prefix (default
    ``"receiver_locations"``; the Born propagators' scattered-field
    reciprocity check instead reads the background receivers, passed as
    ``"bg_receiver_locations"``).

    ``source_comps`` is the ``(src_comp, rec_comp)`` pair as the *source*
    keywords spell them, where that differs from the receiver spelling. It
    does on the scalar channel of the nine multi-channel propagators, whose
    source is named for the quantity it injects (``_volume_rate`` or
    ``_pressure_rate``) while the receiver is named for the field it records
    (``_p``). ``None`` (the default) means the two sides spell the channel
    the same way, which is true of every velocity channel.
    """
    src_src_comp, src_rec_comp = source_comps or (src_comp, rec_comp)
    # Run 1: src_comp at A, rec_comp at B
    kwargs1 = {
        f"source_amplitudes{src_src_comp}": source_amplitudes,
        f"source_locations{src_src_comp}": loc_a,
        f"{receiver_kwarg}{rec_comp}": loc_b,
    }

    outputs1 = propagator(*models, list(dx), dt, **kwargs1, prop_kwargs=prop_kwargs)

    rec1 = outputs1[rec_idx]

    # Run 2: rec_comp at B, src_comp at A
    kwargs2 = {
        f"source_amplitudes{src_rec_comp}": source_amplitudes,
        f"source_locations{src_rec_comp}": loc_b,
        f"{receiver_kwarg}{src_comp}": loc_a,
    }

    outputs2 = propagator(*models, list(dx), dt, **kwargs2, prop_kwargs=prop_kwargs)

    rec2 = outputs2[src_idx]

    assert torch.allclose(rec1, rec2, atol=atol)


def run_checkpoint_equivalence(
    model: torch.Tensor,
    call: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
    use_reentrant: bool,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Assert that ``torch.utils.checkpoint`` does not change a gradient.

    This is a regression guard for issue #120, where wrapping a propagator in
    ``torch.utils.checkpoint`` (the pattern in ``docs/example_checkpointing.py``)
    silently produced a wrong gradient. ``call(m)`` runs a propagator and
    returns its tuple of output Tensors; the gradient of the sum of squares of
    those outputs w.r.t. ``model`` must be identical whether or not the call is
    wrapped in a checkpoint, because checkpointing only trades compute for
    memory.

    It catches the two ways of breaking checkpointing:

    * A propagator returning a ``list`` rather than a ``tuple``. Reentrant
      checkpoint wraps the call in an autograd ``Function`` whose outputs only
      get ``requires_grad`` propagated when forward returns a Tensor or a tuple
      of Tensors, so a list detaches every output and ``backward`` raises here.
    * A ``backward`` that reads ``ctx.saved_tensors`` more than once, which
      makes the non-reentrant checkpoint unpack hook raise.

    The loss covers the tensors held in ``auxiliary_state`` only under
    ``use_reentrant=False``. Under ``use_reentrant=True`` the reentrant
    implementation re-attaches only the top-level ``Tensor`` elements of the
    return, so those tensors come back detached and would contribute a
    constant to the checkpointed loss and a gradient to the reference one --
    the documented behaviour, pinned by
    ``tests/test_auxiliary_state.py``'s reentrant tests, not a regression for
    this helper to catch.
    """
    import torch.utils.checkpoint

    def grad(checkpointed: bool) -> torch.Tensor:
        leaf = model.clone().requires_grad_(True)
        if checkpointed:
            outputs = torch.utils.checkpoint.checkpoint(
                call, leaf, use_reentrant=use_reentrant
            )
        else:
            outputs = call(leaf)
        tensors = (
            [o for o in outputs if isinstance(o, torch.Tensor)]
            if use_reentrant
            else output_tensors(outputs)
        )
        loss = sum((tensor**2).sum() for tensor in tensors)
        loss.backward()
        assert leaf.grad is not None
        return leaf.grad.clone()

    reference = grad(checkpointed=False)
    checkpointed = grad(checkpointed=True)
    # A non-trivial reference gradient ensures the comparison is meaningful.
    assert reference.norm() > 0
    assert torch.allclose(reference, checkpointed, atol=atol, rtol=rtol)
