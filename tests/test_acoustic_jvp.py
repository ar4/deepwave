"""Forward-mode (jvp) tests for the acoustic propagator.

The scattered field of :func:`deepwave.acoustic_born.acoustic_born` is the
directional derivative (JVP) of :func:`deepwave.acoustic` (see
``docs/acoustic_born_derivation.md`` section 6). These tests verify that
``torch.func.jvp`` (and the legacy ``torch.autograd.forward_ad``) route through
the fast compiled Born backend and produce the Born scattered output, agree with
finite differences of ``acoustic``, warn on a PML-region model tangent, and
refuse nested transforms.
"""

import importlib
import warnings

import pytest
import torch
import torch.autograd.forward_ad as forward_ad

import deepwave

# ``deepwave.acoustic_born`` resolves to the re-exported *function* on the
# package (see ``deepwave/__init__.py``); fetch the actual submodule via
# ``importlib`` so ``ab`` is the module.
ab = importlib.import_module("deepwave.acoustic_born")


def _problem(
    ndim: int, dtype: torch.dtype = torch.double
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    shape = {1: (18,), 2: (16, 16), 3: (14, 13, 13)}[ndim]
    v = torch.full(shape, 1500.0, dtype=dtype)
    rho = torch.full(shape, 2200.0, dtype=dtype)
    loc = torch.zeros(1, 1, ndim, dtype=torch.long)
    for d in range(ndim):
        loc[..., d] = shape[d] // 2
    rloc = loc.clone()
    rloc[..., -1] = shape[-1] // 2 - 2
    nt = 20
    torch.manual_seed(0)
    samp = torch.randn(1, 1, nt, dtype=dtype)
    kwargs = {
        "source_amplitudes_volume_rate": samp,
        "source_locations_volume_rate": loc,
        "receiver_locations_p": rloc,
        "pml_width": 4,
        "max_vel": 2000.0,
        "nt": nt,
        "pml_freq": 25.0,
        "accuracy": 4,
    }
    return v, rho, kwargs


def _interior_perturbation(
    shape: tuple[int, ...], pml_width: int, accuracy: int = 4
) -> torch.Tensor:
    """A random perturbation confined to the strict interior.

    Zero everywhere in the finite-difference halo (``accuracy // 2`` cells) and
    the ``pml_width`` absorbing band, so the Born-based jvp is exact there (no
    PML-linearisation caveat) and no warning is emitted.
    """
    fd_pad = accuracy // 2
    torch.manual_seed(1)
    p = torch.zeros(shape, dtype=torch.double)
    interior = tuple(slice(fd_pad + pml_width, s - fd_pad - pml_width) for s in shape)
    p[interior] = torch.randn_like(p[interior]) * 5.0
    assert p.abs().max() > 0
    return p


def _rp(ndim: int, out: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """The pressure receiver amplitudes from an acoustic output tuple."""
    return out[-(ndim + 1)]


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_jvp_receiver_equals_born(ndim: int) -> None:
    """Jvp tangent equals the acoustic_born scattered output (C backend)."""
    v, rho, kwargs = _problem(ndim)
    dv = _interior_perturbation(v.shape, kwargs["pml_width"])
    drho = _interior_perturbation(v.shape, kwargs["pml_width"]) * 4.0

    def f(v_: torch.Tensor, rho_: torch.Tensor) -> torch.Tensor:
        return _rp(ndim, deepwave.acoustic(v_, rho_, 6.0, 4e-4, **kwargs))

    primal, tangent = torch.func.jvp(f, (v, rho), (dv, drho))

    out = ab.acoustic_born(
        v,
        rho,
        6.0,
        4e-4,
        dv=dv,
        drho=drho,
        bg_receiver_locations_p=kwargs["receiver_locations_p"],
        # acoustic_born spells its scattered receivers ``sc_``, so the
        # shared kwargs dict's bare key is renamed on the way in.
        **{
            (f"sc_{k}" if k.startswith("receiver_locations") else k): v
            for k, v in kwargs.items()
        },
    )
    names = ab.acoustic_born_output_names(ndim)
    d = dict(zip(names, out))
    assert torch.allclose(primal, d["bg_receiver_amplitudes_p"], atol=1e-12)
    assert torch.allclose(tangent, d["sc_receiver_amplitudes_p"], atol=1e-12)


@pytest.mark.parametrize("ndim", [1, 2])
def test_jvp_v_rho_tangent_vs_finite_difference(ndim: int) -> None:
    """Jvp tangent matches a directional finite difference of acoustic."""
    v, rho, kwargs = _problem(ndim)
    dv = _interior_perturbation(v.shape, kwargs["pml_width"])
    drho = _interior_perturbation(v.shape, kwargs["pml_width"]) * 4.0

    def f(v_: torch.Tensor, rho_: torch.Tensor) -> torch.Tensor:
        return _rp(ndim, deepwave.acoustic(v_, rho_, 6.0, 4e-4, **kwargs))

    _, tangent = torch.func.jvp(f, (v, rho), (dv, drho))
    eps = 1e-3
    fd = (f(v + eps * dv, rho + eps * drho) - f(v - eps * dv, rho - eps * drho)) / (
        2 * eps
    )
    assert (tangent - fd).abs().max() / fd.abs().max() < 1e-6


def test_jvp_source_tangent_vs_finite_difference() -> None:
    """A source-amplitude tangent matches a finite difference of acoustic."""
    ndim = 2
    v, rho, kwargs = _problem(ndim)
    samp = kwargs["source_amplitudes_volume_rate"]
    torch.manual_seed(2)
    dsamp = torch.randn_like(samp)

    def f(s_: torch.Tensor) -> torch.Tensor:
        kw = dict(kwargs)
        kw["source_amplitudes_volume_rate"] = s_
        return _rp(ndim, deepwave.acoustic(v, rho, 6.0, 4e-4, **kw))

    _, tangent = torch.func.jvp(f, (samp,), (dsamp,))
    eps = 1e-3
    fd = (f(samp + eps * dsamp) - f(samp - eps * dsamp)) / (2 * eps)
    assert (tangent - fd).abs().max() / fd.abs().max() < 1e-6


def test_jvp_initial_wavefield_tangent_vs_finite_difference() -> None:
    """An initial-wavefield tangent matches a finite difference of acoustic."""
    ndim = 2
    v, rho, kwargs = _problem(ndim)
    pml_width = kwargs["pml_width"]
    padded = tuple(s + 2 * pml_width for s in v.shape)
    torch.manual_seed(3)
    p0 = torch.zeros(1, *padded, dtype=v.dtype)
    dp0 = torch.zeros_like(p0)
    interior = (
        slice(None),
        *(slice(pml_width + 2, s - pml_width - 2) for s in padded),
    )
    dp0[interior] = torch.randn_like(dp0[interior])

    def f(p0_: torch.Tensor) -> torch.Tensor:
        return _rp(
            ndim,
            deepwave.acoustic(
                v, rho, 6.0, 4e-4, initial_state={"pressure_0": p0_}, **kwargs
            ),
        )

    _, tangent = torch.func.jvp(f, (p0,), (dp0,))
    eps = 1e-2
    fd = (f(p0 + eps * dp0) - f(p0 - eps * dp0)) / (2 * eps)
    assert (tangent - fd).abs().max() / fd.abs().max() < 1e-6


def test_jvp_v_tangent_in_pml_warns() -> None:
    """A model tangent inside the PML warns, unless ``max_vel`` is given.

    The PML profile's amplitude is built from the maximum wave speed, so it
    depends on the model only while that maximum is inferred from it.
    Supplying ``max_vel`` makes the profile a constant of the call, so the
    caveat does not apply and is not raised.
    """
    ndim = 2
    v, rho, kwargs = _problem(ndim)
    pinned = dict(kwargs)
    inferred = {name: value for name, value in kwargs.items() if name != "max_vel"}
    assert "max_vel" in pinned
    dv = torch.zeros_like(v)
    torch.manual_seed(4)
    dv[: kwargs["pml_width"]] = torch.randn_like(dv[: kwargs["pml_width"]]) * 20.0

    def f(v_: torch.Tensor, kw: dict) -> torch.Tensor:
        return _rp(ndim, deepwave.acoustic(v_, rho, 6.0, 4e-4, **kw))

    with pytest.warns(UserWarning, match="PML"):
        torch.func.jvp(lambda v_: f(v_, inferred), (v,), (dv,))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.func.jvp(lambda v_: f(v_, pinned), (v,), (dv,))
    assert not [record for record in caught if "PML" in str(record.message)], [
        str(record.message) for record in caught
    ]


def test_jvp_matches_legacy_forward_ad() -> None:
    """The torch.func.jvp and torch.autograd.forward_ad paths agree."""
    ndim = 2
    v, rho, kwargs = _problem(ndim)
    dv = _interior_perturbation(v.shape, kwargs["pml_width"])

    def f(v_: torch.Tensor) -> torch.Tensor:
        return _rp(ndim, deepwave.acoustic(v_, rho, 6.0, 4e-4, **kwargs))

    _, tangent_func = torch.func.jvp(f, (v,), (dv,))
    with forward_ad.dual_level():
        vd = forward_ad.make_dual(v, dv)
        out = f(vd)
        _, tangent_legacy = forward_ad.unpack_dual(out)
    assert torch.allclose(tangent_func, tangent_legacy, atol=1e-12)


def test_jvp_nested_transforms_raise() -> None:
    """Nested torch.func transforms over the fast jvp rule raise clearly."""
    ndim = 2
    v, rho, kwargs = _problem(ndim)
    dv = _interior_perturbation(v.shape, kwargs["pml_width"])

    def f(v_: torch.Tensor) -> torch.Tensor:
        return _rp(ndim, deepwave.acoustic(v_, rho, 6.0, 4e-4, **kwargs)).sum()

    with pytest.raises(NotImplementedError, match=r"nested|single"):
        torch.func.jvp(lambda x: torch.func.jvp(f, (x,), (dv,))[1], (v,), (dv,))
