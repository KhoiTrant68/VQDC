import numpy as np
import torch
import torch.nn as nn


def convert_to_coord_format(
    batch_size, height, width, device="cpu", integer_values=False
):
    """
    Generates a 2-channel tensor representing x and y coordinates in the range [-1, 1] or as integers.

    Args:
        batch_size (int): Batch size for the coordinate tensor.
        height (int): Height of the coordinate map.
        width (int): Width of the coordinate map.
        device (str, optional): Device on which to create the tensor (default: 'cpu').
        integer_values (bool, optional): If True, generates integer coordinates instead of normalized values.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 2, height, width) representing x, y coordinates.
    """
    if integer_values:
        x_channel = (
            torch.arange(width, dtype=torch.float, device=device)
            .view(1, 1, 1, -1)
            .repeat(batch_size, 1, height, 1)
        )
        y_channel = (
            torch.arange(height, dtype=torch.float, device=device)
            .view(1, 1, -1, 1)
            .repeat(batch_size, 1, 1, width)
        )
    else:
        x_channel = (
            torch.linspace(-1, 1, width, device=device)
            .view(1, 1, 1, -1)
            .repeat(batch_size, 1, height, 1)
        )
        y_channel = (
            torch.linspace(-1, 1, height, device=device)
            .view(1, 1, -1, 1)
            .repeat(batch_size, 1, 1, width)
        )
    return torch.cat((x_channel, y_channel), dim=1)


class ConLinear(nn.Module):
    """
    A 1x1 Convolution layer, often used as a linear transformation for feature maps.

    Args:
        ch_in (int): Number of input channels.
        ch_out (int): Number of output channels.
        is_first (bool, optional): Flag indicating whether this is the first layer,
                                    affecting weight initialization (default: False).
        bias (bool, optional): Whether to include a bias term (default: True).
    """

    def __init__(self, ch_in, ch_out, is_first=False, bias=True):
        super(ConLinear, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=bias)
        if is_first:
            nn.init.uniform_(self.conv.weight, -np.sqrt(9 / ch_in), np.sqrt(9 / ch_in))
        else:
            nn.init.uniform_(self.conv.weight, -np.sqrt(3 / ch_in), np.sqrt(3 / ch_in))

    def forward(self, x):
        return self.conv(x)


class LFF(nn.Module):
    """
    Learned Frequency Filter (LFF) module.

    This module processes coordinate inputs with a 1x1 convolution and applies a sine activation.

    Args:
        hidden_size (int): Number of output channels/hidden units.
    """

    def __init__(
        self,
        hidden_size,
    ):
        super(LFF, self).__init__()
        self.ffm = ConLinear(
            2, hidden_size, is_first=True
        )  # 2 channels for x, y coords
        self.activation = torch.sin  # Use the efficient PyTorch sin function

    def forward(self, x):
        x = self.ffm(x)
        x = self.activation(x)
        return x


class FourierPositionEmbedding(nn.Module):
    """
    Fourier Feature Positional Embedding module.

    This module generates Fourier features from coordinates and adds them to an input tensor.

    Args:
        coord_size (int): Size of the coordinate map (assumed square).
        hidden_size (int): Number of hidden units for the LFF module.
        integer_values (bool, optional): If True, use integer coordinate values (default: False).
    """

    def __init__(self, coord_size, hidden_size, integer_values=False):
        super().__init__()
        # Pre-compute coordinates for efficiency - device doesn't matter here
        self.coord = convert_to_coord_format(
            1, coord_size, coord_size, integer_values=integer_values
        )
        self.lff = LFF(hidden_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Input tensor with added Fourier features.
        """
        batch_size = x.shape[0]
        # Expand coordinates to match the batch size
        coord = self.coord.repeat(batch_size, 1, 1, 1).to(x.device)
        fourier_features = self.lff(coord)
        x = x + fourier_features  # Add Fourier features to the input
        return x


# Example usage
# if __name__ == "__main__":
#     x = torch.randn(10, 64, 32, 32)
#     module = FourierPositionEmbedding(coord_size=32, hidden_size=64)
#     output = module(x)
#     print(output.shape)  # Output shape should be the same as input shape
