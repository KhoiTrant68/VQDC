import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class Entropy(nn.Module):
    """
    Calculates the entropy of image patches.

    This module first divides the input image into non-overlapping patches and then calculates the entropy
    of each patch using a marginal probability distribution function estimated with a Gaussian kernel.

    Args:
        patch_size (int): Size of the square patches.
        image_width (int): Width of the input image.
        image_height (int): Height of the input image.
    """

    def __init__(self, patch_size: int, image_width: int, image_height: int):
        super().__init__()
        self.width = image_width
        self.height = image_height
        self.patch_size = patch_size
        self.patch_num = (self.width * self.height) // (self.patch_size**2)
        self.hw = self.width // self.patch_size
        self.unfold = nn.Unfold(
            kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size
        )
        self.bins = torch.linspace(-1, 1, 32)
        self.sigma = torch.tensor(0.01)

    def entropy(self, values: torch.Tensor) -> torch.Tensor:
        """
        Calculates the entropy using the marginal probability distribution.

        Args:
            values: shape [B*N, 1, P*P], patch values.

        Returns:
            torch.Tensor: Entropy for each patch, shape [B, H, W].
        """
        epsilon = 1e-40
        # Correcting the broadcasting for bins:
        residuals = values - self.bins.unsqueeze(1).to(values.device)
        kernel_values = torch.exp(
            -0.5 * (residuals / self.sigma.to(values.device)).pow(2)
        )

        pdf = torch.mean(
            kernel_values, dim=2
        )  # Calculate mean along the pixel dimension
        normalization = torch.sum(pdf, dim=1, keepdim=True) + epsilon
        pdf = pdf / normalization + epsilon
        entropy = -torch.sum(pdf * torch.log(pdf), dim=1)
        return rearrange(entropy, "(B H W) -> B H W", H=self.hw, W=self.hw)

    def forward(self, inputs: Tensor) -> torch.Tensor:
        """
        Forward pass of the Entropy module.

        Args:
            inputs: shape [B, C, H, W], Input image tensor.

        Returns:
            torch.Tensor: Entropy for each patch, shape [B, H/patch_size, W/patch_size].
        """
        gray_images = (
            0.2989 * inputs[:, 0:1, :, :]
            + 0.5870 * inputs[:, 1:2, :, :]
            + 0.1140 * inputs[:, 2:, :, :]
        )
        unfolded_images = self.unfold(gray_images)
        # Correcting the rearrange operation:
        unfolded_images = rearrange(
            unfolded_images,
            "B (P1 P2) N -> (B N) 1 (P1 P2)",
            P1=self.patch_size,
            P2=self.patch_size,
        )
        entropy = self.entropy(unfolded_images)
        return entropy
