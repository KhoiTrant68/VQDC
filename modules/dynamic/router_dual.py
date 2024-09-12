import json

import numpy as np
import torch
from torch import nn


class DualGrainFeatureRouter(nn.Module):
    """
    Routes features based on a learnable gate mechanism.

    Args:
        num_channels (int): Number of input channels.
        normalization_type (str, optional): Type of normalization.
            Defaults to "none". Options: "none", "group-{num_groups}".
        gate_type (str, optional): Type of gate function.
            Defaults to "1layer-fc". Options: "1layer-fc", "2layer-fc-SiLu".
    """

    def __init__(self, num_channels, normalization_type="none", gate_type="1layer-fc"):
        super().__init__()
        self.gate_pool = nn.AvgPool2d(2, 2)
        self.gate_type = gate_type

        if gate_type == "1layer-fc":
            self.gate = nn.Linear(num_channels * 2, 2)
        elif gate_type == "2layer-fc-SiLu":
            self.gate = nn.Sequential(
                nn.Linear(num_channels * 2, num_channels * 2),
                nn.SiLU(inplace=True),
                nn.Linear(num_channels * 2, 2),
            )
        else:
            raise ValueError(f"Unsupported gate type: {gate_type}")

        self.num_splits = 2
        self.normalization_type = normalization_type

        if self.normalization_type == "none":
            self.feature_norm_fine = nn.Identity()
            self.feature_norm_coarse = nn.Identity()
        elif self.normalization_type.startswith("group-"):
            num_groups = int(self.normalization_type.split("-")[-1])
            self.feature_norm_fine = nn.GroupNorm(
                num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True
            )
            self.feature_norm_coarse = nn.GroupNorm(
                num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True
            )
        else:
            raise ValueError(f"Unsupported normalization type: {normalization_type}")

    def forward(self, h_fine, h_coarse, entropy=None):
        """
        Forward pass of the DualGrainFeatureRouter.

        Args:
            h_fine (torch.Tensor): Fine-grained features.
            h_coarse (torch.Tensor): Coarse-grained features.
            entropy (torch.Tensor, optional): Entropy values. Not used in this router.

        Returns:
            torch.Tensor: Routing gate.
        """
        h_fine = self.feature_norm_fine(h_fine)
        h_coarse = self.feature_norm_coarse(h_coarse)

        avg_h_fine = self.gate_pool(h_fine)
        h_logistic = torch.cat([h_coarse, avg_h_fine], dim=1).permute(0, 2, 3, 1)

        gate = self.gate(h_logistic)
        return gate


class DualGrainEntropyRouter(nn.Module):
    """
    Base class for entropy-based dual-grain routers.
    """

    def __init__(self, json_path):
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            self.entropy_thresholds = json.load(f)

    def _get_gate_from_threshold(self, entropy, threshold):
        """
        Computes the routing gate based on entropy and threshold.

        Args:
            entropy (torch.Tensor): Entropy values.
            threshold (float): Entropy threshold.

        Returns:
            torch.Tensor: Routing gate.
        """
        gate_fine = (entropy > threshold).bool().long().unsqueeze(-1)
        gate_coarse = (entropy <= threshold).bool().long().unsqueeze(-1)
        gate = torch.cat([gate_coarse, gate_fine], dim=-1)
        return gate


class DualGrainFixedEntropyRouter(DualGrainEntropyRouter):
    """
    Routes features based on a fixed entropy threshold.

    Args:
        json_path (str): Path to JSON file with entropy thresholds.
        fine_grain_ratio (float): Ratio of fine-grained tokens.
    """

    def __init__(self, json_path, fine_grain_ratio):
        super().__init__(json_path)
        self.fine_grain_threshold = self.entropy_thresholds[
            f"{int(100 - fine_grain_ratio * 100)}"
        ]

    def forward(self, h_fine=None, h_coarse=None, entropy=None):
        """
        Forward pass of the DualGrainFixedEntropyRouter.

        Args:
            h_fine (torch.Tensor, optional): Fine-grained features. Not used in this router.
            h_coarse (torch.Tensor, optional): Coarse-grained features. Not used in this router.
            entropy (torch.Tensor): Entropy values.

        Returns:
            torch.Tensor: Routing gate.
        """
        gate = self._get_gate_from_threshold(entropy, self.fine_grain_threshold)
        return gate


class DualGrainDynamicEntropyRouter(DualGrainEntropyRouter):
    """
    Routes features based on a dynamically sampled entropy threshold.

    Args:
        json_path (str): Path to JSON file with entropy thresholds.
        fine_grain_ratio_min (float): Minimum ratio of fine-grained tokens.
        fine_grain_ratio_max (float): Maximum ratio of fine-grained tokens.
    """

    def __init__(self, json_path, fine_grain_ratio_min=0.01, fine_grain_ratio_max=0.99):
        super().__init__(json_path)
        self.fine_grain_ratio_min = int(fine_grain_ratio_min * 100)
        self.fine_grain_ratio_max = int(fine_grain_ratio_max * 100) + 1

    def forward(self, h_fine=None, h_coarse=None, entropy=None):
        """
        Forward pass of the DualGrainDynamicEntropyRouter.

        Args:
            h_fine (torch.Tensor, optional): Fine-grained features. Not used in this router.
            h_coarse (torch.Tensor, optional): Coarse-grained features. Not used in this router.
            entropy (torch.Tensor): Entropy values.

        Returns:
            torch.Tensor: Routing gate.
        """
        fine_grain_ratio = np.random.randint(
            self.fine_grain_ratio_min, self.fine_grain_ratio_max
        )
        fine_grain_threshold = self.entropy_thresholds[str(fine_grain_ratio)]
        gate = self._get_gate_from_threshold(entropy, fine_grain_threshold)
        return gate


if __name__ == "__main__":
    model = DualGrainFixedEntropyRouter(
        json_path="D:\AwesomeCV\VQDC_testing\entropy_thresholds_imagenet_train_patch-16.json",
        fine_grain_ratio=0.5,
    )
