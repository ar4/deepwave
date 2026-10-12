"""Tests for ``ScalarBorn``'s ``v0.0.27`` state-dict compatibility.

The scattering model was called ``scatter`` in ``v0.0.27`` and is called
``dv`` now. An attribute alias cannot cover a checkpoint, because
``load_state_dict`` matches on the *key*, so a ``v0.0.27`` checkpoint used to
fail outright with::

    RuntimeError: Error(s) in loading state_dict for ScalarBorn: Missing
    key(s) in state_dict: "dv". Unexpected key(s) in state_dict: "scatter".

``ScalarBorn._load_from_state_dict`` now remaps the key, with a
``FutureWarning`` naming the release that removes the remap.
"""

import warnings

import pytest
import torch

import deepwave


def _module() -> deepwave.ScalarBorn:
    v = 1500 * torch.ones(8, 8)
    dv = torch.zeros(8, 8)
    return deepwave.ScalarBorn(v, 4.0, dv=dv)


def _v0027_state_dict() -> dict:
    """A state dict shaped the way ``v0.0.27``'s ``ScalarBorn`` saved one."""
    return {
        "v": 1500 * torch.ones(8, 8),
        "scatter": torch.full((8, 8), 3.0),
    }


def test_v0027_scatter_checkpoint_loads_with_warning() -> None:
    """The old key loads, warns, names 0.4.0, and lands on ``dv``."""
    module = _module()
    saved = _v0027_state_dict()
    with pytest.warns(FutureWarning, match="state_dict key 'scatter'") as record:
        module.load_state_dict(saved)
    assert "0.4.0" in str(record[0].message)
    assert torch.equal(module.dv, torch.full((8, 8), 3.0))
    assert torch.equal(module.v, 1500 * torch.ones(8, 8))


def test_state_dict_roundtrip_is_silent() -> None:
    """A freshly saved dict loads with no warning at all."""
    module = _module()
    saved = module.state_dict()
    other = _module()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        other.load_state_dict(saved)
    assert torch.equal(other.dv, module.dv)


def test_nested_module_prefix_is_respected() -> None:
    """A ``ScalarBorn`` inside a container is remapped under its own prefix."""

    class Container(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.born = _module()

    container = Container()
    saved = {
        "born.v": 1500 * torch.ones(8, 8),
        "born.scatter": torch.full((8, 8), 7.0),
    }
    with pytest.warns(FutureWarning, match="state_dict key 'scatter'"):
        container.load_state_dict(saved)
    assert torch.equal(container.born.dv, torch.full((8, 8), 7.0))


def test_both_keys_present_is_not_remapped() -> None:
    """A checkpoint carrying both keys is a user error, reported as such.

    ``dv`` wins (it is the real key) and ``scatter`` is reported by
    ``load_state_dict``'s own ``unexpected_keys``, rather than being silently
    remapped over the top of the value the user actually saved.
    """
    module = _module()
    saved = {
        "v": 1500 * torch.ones(8, 8),
        "dv": torch.full((8, 8), 1.0),
        "scatter": torch.full((8, 8), 2.0),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = module.load_state_dict(saved, strict=False)
    assert "scatter" in result.unexpected_keys
    assert torch.equal(module.dv, torch.full((8, 8), 1.0))


def test_strict_true_still_reports_a_genuinely_missing_key() -> None:
    """The remap does not weaken ``strict=True`` for a real mismatch."""
    module = _module()
    with pytest.raises(RuntimeError, match="Missing key"):
        module.load_state_dict({"v": 1500 * torch.ones(8, 8)})


def test_strict_false_still_works() -> None:
    """``strict=False`` loads what it can and reports the rest."""
    module = _module()
    with pytest.warns(FutureWarning, match="state_dict key 'scatter'"):
        result = module.load_state_dict(
            {"scatter": torch.full((8, 8), 5.0)}, strict=False
        )
    assert "v" in result.missing_keys
    assert not result.unexpected_keys
    assert torch.equal(module.dv, torch.full((8, 8), 5.0))


def test_saved_dict_uses_the_new_key_only() -> None:
    """Saving never writes the deprecated key, so a re-save silences the warning."""
    keys = set(_module().state_dict())
    assert keys == {"v", "dv"}
