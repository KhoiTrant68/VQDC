import torch
import torch.nn as nn

class TripleGrainFeatureRouter(nn.Module):
    """
    Routes features in a triple-grain encoder based on learned features.

    Args:
        num_channels (int): Number of input channels.
        normalization_type (str, optional): Type of normalization to apply. 
            Defaults to "none". Options: "none", "group-{num_groups}".
        gate_type (str, optional): Type of gate function to use. 
            Defaults to "1layer-fc". Options: "1layer-fc", "2layer-fc-SiLu", 
            "2layer-fc-ReLU".
    """
    def __init__(self, num_channels, normalization_type="none", gate_type="1layer-fc"):
        super().__init__()

        # Define pooling layers for fine and median grains
        self.gate_median_pool = nn.AvgPool2d(2, 2)
        self.gate_fine_pool = nn.AvgPool2d(4, 4)
        self.num_splits = 3  # Number of grain levels

        # Define the gate function based on gate_type
        self.gate_type = gate_type
        if gate_type == "1layer-fc":
            self.gate = nn.Linear(num_channels * self.num_splits, self.num_splits)
        elif gate_type == "2layer-fc-SiLu":
            self.gate = nn.Sequential(
                nn.Linear(
                    num_channels * self.num_splits, num_channels * self.num_splits
                ),
                nn.SiLU(inplace=True),
                nn.Linear(num_channels * self.num_splits, self.num_splits),
            )
        elif gate_type == "2layer-fc-ReLU":
            self.gate = nn.Sequential(
                nn.Linear(
                    num_channels * self.num_splits, num_channels * self.num_splits
                ),
                nn.ReLU(inplace=True),
                nn.Linear(num_channels * self.num_splits, self.num_splits),
            )
        else:
            raise ValueError(f"Unsupported gate type: {gate_type}")

        # Define normalization layers based on normalization_type

        self.normalization_type = normalization_type
        if self.normalization_type == "none":
            self.feature_norm_fine = nn.Identity()
            self.feature_norm_median = nn.Identity()
            self.feature_norm_coarse = nn.Identity()
        elif self.normalization_type.startswith("group-"):
            num_groups = int(self.normalization_type.split("-")[-1])
            self.feature_norm_fine = nn.GroupNorm(
                num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True
            )
            self.feature_norm_median = nn.GroupNorm(
                num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True
            )
            self.feature_norm_coarse = nn.GroupNorm(
                num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True
            )
        else:
            raise ValueError(f"Unsupported normalization type: {normalization_type}")

    def forward(self, h_fine, h_median, h_coarse, entropy=None):
        """
        Forward pass of the TripleGrainFeatureRouter.

        Args:
            h_fine (torch.Tensor): Fine-grained feature map.
            h_median (torch.Tensor): Median-grained feature map.
            h_coarse (torch.Tensor): Coarse-grained feature map.
            entropy (torch.Tensor, optional): Entropy tensor. Not used in this router. 
                Defaults to None.

        Returns:
            torch.Tensor: Routing gate tensor.
        """
        # Normalize the input feature maps
        h_fine = self.feature_norm_fine(h_fine)
        h_median = self.feature_norm_median(h_median)
        h_coarse = self.feature_norm_coarse(h_coarse)

        # Downsample fine and median features to match coarse resolution
        avg_h_fine = self.gate_fine_pool(h_fine)
        avg_h_median = self.gate_median_pool(h_median)

        # Concatenate features and apply the gate function

        h_logistic = torch.cat(
            [h_coarse, avg_h_median, avg_h_fine], dim=1
        ).permute(0, 2, 3, 1)
        gate = self.gate(h_logistic)
        return gate