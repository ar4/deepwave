import torch
import pytest
import deepwave.base.model


def test_init_scalar():
    """Init model with scalars"""
    properties = {'a': torch.ones(3, 4),
                  'b': torch.zeros(3, 4)}
    dx = 5.0
    model = deepwave.base.model.Model(properties, dx, pad_width=1, origin=2.0)
    assert model.properties == properties
    assert model.device == properties['a'].device
    assert model.ndim == 2
    assert (model.shape == torch.Tensor([3, 4, 1]).long()).all()
    assert (model.dx == dx * torch.ones(2)).all()
    assert (model.pad_width == torch.Tensor([1, 1, 1, 1, 0, 0]).long()).all()
    assert (model.origin == torch.Tensor([2.0, 2.0])).all()
    assert model.interior == [slice(1, 2), slice(1, 3)]


def test_init_list():
    """Init model with lists"""
    properties = {'a': torch.ones(3, 4),
                  'b': torch.zeros(3, 4)}
    dx = [5.0, 5.0]
    pad_width = [1, 1, 1, 1, 0, 0]
    origin = [2.0, 2.0]
    model = deepwave.base.model.Model(properties, dx, pad_width=pad_width,
                                      origin=origin)
    assert model.properties == properties
    assert model.device == properties['a'].device
    assert model.ndim == 2
    assert (model.shape == torch.Tensor([3, 4, 1]).long()).all()
    assert (model.dx == torch.Tensor(dx)).all()
    assert (model.pad_width == torch.Tensor([1, 1, 1, 1, 0, 0]).long()).all()
    assert (model.origin == torch.Tensor([2.0, 2.0])).all()
    assert model.interior == [slice(1, 2), slice(1, 3)]


def test_not_tensor():
    """One of the properties is not a Tensor"""
    properties = {'a': torch.ones(3, 4),
                  'b': [0, 1]}
    with pytest.raises(TypeError):
        deepwave.base.model.Model(properties, 5.0, pad_width=1,
                                  origin=2.0)


def test_different_types():
    """Properties have different types"""
    properties = {'a': torch.ones(3, 4),
                  'b': torch.zeros(3, 4, dtype=torch.double)}
    with pytest.raises(RuntimeError):
        deepwave.base.model.Model(properties, 5.0, pad_width=1,
                                  origin=2.0)


def test_different_sizes1():
    """Properties have different sizes (same ndim)"""
    properties = {'a': torch.ones(3, 4),
                  'b': torch.zeros(3, 5)}
    with pytest.raises(RuntimeError):
        deepwave.base.model.Model(properties, 5.0, pad_width=1,
                                  origin=2.0)


def test_different_sizes2():
    """Properties have different sizes (different ndim)"""
    properties = {'a': torch.ones(3, 4),
                  'b': torch.zeros(3, 4, 1)}
    with pytest.raises(RuntimeError):
        deepwave.base.model.Model(properties, 5.0, pad_width=1,
                                  origin=2.0)


def test_nonpositive_dx1():
    """Nonpositive dx (scalar)"""
    properties = {'a': torch.ones(3, 4),
                  'b': torch.zeros(3, 4)}
    with pytest.raises(RuntimeError):
        deepwave.base.model.Model(properties, -5.0, pad_width=1,
                                  origin=2.0)


def test_nonpositive_dx2():
    """Nonpositive dx (list)"""
    properties = {'a': torch.ones(3, 4),
                  'b': torch.zeros(3, 4)}
    dx = [5.0, 0.0]
    with pytest.raises(RuntimeError):
        deepwave.base.model.Model(properties, dx, pad_width=1,
                                  origin=2.0)


def test_negative_pad1():
    """Negative pad (scalar)"""
    properties = {'a': torch.ones(3, 4),
                  'b': torch.zeros(3, 4)}
    with pytest.raises(RuntimeError):
        deepwave.base.model.Model(properties, 5.0, pad_width=-1,
                                  origin=2.0)


def test_negative_pad2():
    """Negative pad (list)"""
    properties = {'a': torch.ones(3, 4),
                  'b': torch.zeros(3, 4)}
    pad_width = [1, 1, -1, 1, 0, 0]
    with pytest.raises(RuntimeError):
        deepwave.base.model.Model(properties, 5.0, pad_width=pad_width,
                                  origin=2.0)


def test_integer_origin():
    """Origin is int instead of float"""
    properties = {'a': torch.ones(3, 4),
                  'b': torch.zeros(3, 4)}
    with pytest.raises(TypeError):
        deepwave.base.model.Model(properties, 5.0, pad_width=1,
                                  origin=2)


def test_extract():
    """Extract portion of model"""
    properties = {'a': torch.ones(3, 4),
                  'b': torch.zeros(3, 4)}
    model = deepwave.base.model.Model(properties, 5.0, pad_width=1, origin=2.0)
    model_extract = model[:, 1:2]
    assert (model_extract.shape == torch.Tensor([3, 3, 1]).long()).all()
    assert model_extract.properties['a'].shape == torch.Size([3, 3])
    assert model_extract.properties['b'].shape == torch.Size([3, 3])
    assert model_extract.ndim == 2
    assert (model_extract.pad_width ==
            torch.Tensor([1, 1, 1, 1, 0, 0]).long()).all()
    assert (model_extract.origin == torch.Tensor([2.0, 7.0])).all()
    assert model_extract.interior == [slice(1, 2), slice(1, 2)]


def test_pad1():
    """Change pad_width from 1 to 2"""
    properties = {'a': torch.ones(3, 4),
                  'b': torch.zeros(3, 4)}
    model = deepwave.base.model.Model(properties, 5.0, pad_width=1, origin=2.0)
    model_pad = model.pad(2)
    assert (model_pad.shape == torch.Tensor([5, 6, 1]).long()).all()
    assert model_pad.properties['a'].shape == torch.Size([5, 6])
    assert model_pad.properties['b'].shape == torch.Size([5, 6])
    assert model_pad.ndim == 2
    assert (model_pad.pad_width ==
            torch.Tensor([2, 2, 2, 2, 0, 0]).long()).all()
    assert (model_pad.origin == torch.Tensor([2.0, 2.0])).all()
    assert model_pad.interior == [slice(2, 3), slice(2, 4)]


def test_pad2():
    """Add two pad_widths"""
    properties = {'a': torch.ones(3, 4),
                  'b': torch.zeros(3, 4)}
    model = deepwave.base.model.Model(properties, 5.0, pad_width=1, origin=2.0)
    model_pad = model.pad(1, 1)
    assert (model_pad.shape == torch.Tensor([5, 6, 1]).long()).all()
    assert model_pad.properties['a'].shape == torch.Size([5, 6])
    assert model_pad.properties['b'].shape == torch.Size([5, 6])
    assert model_pad.ndim == 2
    assert (model_pad.pad_width ==
            torch.Tensor([2, 2, 2, 2, 0, 0]).long()).all()
    assert (model_pad.origin == torch.Tensor([2.0, 2.0])).all()
    assert model_pad.interior == [slice(2, 3), slice(2, 4)]


def test_pad3():
    """Verify that padded model has correct values"""
    properties = {'a': torch.arange(6).float().reshape(2, 3)}
    model = deepwave.base.model.Model(properties, 5.0)
    model_pad = model.pad([1,0,0,0,0,0])
    assert (model_pad.properties['a'] == torch.tensor([[0.0, 1.0, 2.0],
                                                       [0.0, 1.0, 2.0],
                                                       [3.0, 4.0, 5.0]])).all()
