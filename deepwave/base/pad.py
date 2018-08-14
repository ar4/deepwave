"""Pad a model."""
import torch


class Pad(torch.nn.Module):
    """Pad the model.

    Args:
        pad_widths: Ints or Nones or lists of such of length
            6, specifying padding at beginning and end of each dimension.
            When multiple pad_widths are specified, they will be added.
    """

    def __init__(self, *pad_widths):
        super(Pad, self).__init__()
        self.pad_widths = pad_widths

    def forward(self, model):
        """Perform extraction.

        Args:
            model: A Model object

        Returns:
            Padded model as a Model object
        """
        return model.pad(*self.pad_widths)
