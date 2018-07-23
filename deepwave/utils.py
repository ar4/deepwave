"""Utilities to make Deepwave easier to use"""
import torch


class Tail(torch.nn.Module):
    """PyTorch Module to chop the predicted output to the true output length.
    """

    def forward(self, y_pred, y_true):
        """Chop off the beginning of the predicted output if necessary.

        If y_pred is shorter than y_true then no changes are made.

        Args:
            y_pred: The predicted output Tensor of shape [num_steps, ...]
            y_true: The true output Tensor of shape [num_steps', ...]

        Returns:
            y_pred with the beginning chopped off to make it the same length
                as y_true
            y_true, unmodified
        """
        num_steps = len(y_true)
        if len(y_pred) < num_steps:
            return y_pred, y_true
        y_pred = y_pred[-num_steps:]
        return y_pred, y_true
