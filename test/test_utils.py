"""Tests for deepwave.utils"""
import numpy as np
import torch
import deepwave


def test_tail_forward1():
    """Test chopping works."""
    shape_pred = (5, 2)
    shape_true = (3, 2)
    y_pred, y_pred_chop, y_true, y_true_chop = \
        run_tail_forward(shape_pred, shape_true)
    assert y_pred.shape == shape_pred
    assert y_pred_chop.shape == shape_true
    assert y_true.shape == shape_true
    assert y_true_chop.shape == shape_true
    assert np.allclose(y_pred_chop.numpy(), y_pred[2:].numpy())
    assert np.allclose(y_true_chop.numpy(), y_true.numpy())


def test_tail_forward2():
    """Test correct behaviour when y_pred shorter than y_true"""
    shape_pred = (3, 2)
    shape_true = (5, 2)
    y_pred, y_pred_chop, y_true, y_true_chop = \
        run_tail_forward(shape_pred, shape_true)
    assert y_pred.shape == shape_pred
    assert y_pred_chop.shape == shape_pred
    assert y_true.shape == shape_true
    assert y_true_chop.shape == shape_true
    assert np.allclose(y_pred_chop.numpy(), y_pred.numpy())
    assert np.allclose(y_true_chop.numpy(), y_true.numpy())


def test_tail_forward3():
    """Test correct behaviour when y_pred same length as y_true"""
    shape_pred = (3, 2)
    shape_true = (3, 2)
    y_pred, y_pred_chop, y_true, y_true_chop = \
        run_tail_forward(shape_pred, shape_true)
    assert y_pred.shape == shape_pred
    assert y_pred_chop.shape == shape_pred
    assert y_true.shape == shape_true
    assert y_true_chop.shape == shape_true
    assert np.allclose(y_pred_chop.numpy(), y_pred.numpy())
    assert np.allclose(y_true_chop.numpy(), y_true.numpy())


def run_tail_forward(shape_pred, shape_true, requires_grad=False):
    """Run the Tail module forward.

    Create y_pred and y_true of the specified shapes and then apply
    the Tail module. The two inputs are filled with random values.

    Args:
        shape_pred: A tuple specifying the shape of y_pred
        shape_true: A tuple specifying the shape of y_true
        requires_grad: Bool specifying y_pred.requires_grad

    Returns:
        y_pred: y_pred Tensor before chopping
        y_pred_chop: y_pred after chopping
        y_true: y_true Tensor before chopping
        y_true_chop: y_true after chopping (should be same)
    """
    y_pred = torch.rand(shape_pred)
    y_pred.requires_grad = requires_grad
    y_true = torch.rand(shape_true)
    tail = deepwave.utils.Tail()
    y_pred_chop, y_true_chop = tail(y_pred, y_true)

    return y_pred, y_pred_chop, y_true, y_true_chop


def test_tail_backward1():
    """Test correct gradient after chopping"""
    shape_pred = (5, 2)
    shape_true = (3, 2)
    grad = run_tail_backward(shape_pred, shape_true)
    expected_grad = np.zeros(shape_pred)
    expected_grad[2:] = 1
    assert grad[0].shape == shape_pred
    assert np.allclose(grad[0].numpy(), expected_grad)


def test_tail_backward2():
    """Test correct gradient when y_pred shorter than y_true"""
    shape_pred = (3, 2)
    shape_true = (5, 2)
    grad = run_tail_backward(shape_pred, shape_true)
    expected_grad = np.ones(shape_pred)
    assert grad[0].shape == shape_pred
    assert np.allclose(grad[0].numpy(), expected_grad)


def run_tail_backward(shape_pred, shape_true):
    """Run the Tail module backward and return the gradient.

    The gradient should be zero in the chopped region and one elsewhere.

    Args:
        shape_pred: A tuple specifying the shape of y_pred
        shape_true: A tuple specifying the shape of y_true

    Returns:
        The gradient of y_pred_chop w.r.t. y_pred
    """
    y_pred, y_pred_chop, _, _ = run_tail_forward(shape_pred, shape_true,
                                                 requires_grad=True)
    grad = torch.ones_like(y_pred_chop)
    grad = torch.autograd.grad(y_pred_chop, y_pred, grad_outputs=grad)
    return grad


def test_tail_loss():
    """Verify correct loss function value after running tail."""
    shape_pred = (5, 2)
    shape_true = (3, 2)
    loss = run_tail_loss(shape_pred, shape_true)
    assert np.isclose(loss, 1 / 6)


def run_tail_loss(shape_pred, shape_true):
    """Test the Tail module with a loss function.

    y_pred is filled with random values in the area that will be chopped,
    and in the non-chopped area only differs from y_true by one in one cell.
    The expected loss is therefore 1/shape_true.

    Args:
        shape_pred: A tuple specifying the shape of y_pred
        shape_true: A tuple specifying the shape of y_true
        The first dimension of shape_pred must be >= that of shape_true

    Returns:
        The MSE loss
    """
    y_pred = torch.rand(shape_pred)
    y_true = y_pred[-shape_true[0]:].clone()
    y_true[(0, ) * y_true.dim()] += 1
    tail = deepwave.utils.Tail()
    criterion = torch.nn.MSELoss()
    loss = criterion(*tail(y_pred, y_true))

    return loss
