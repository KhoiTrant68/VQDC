import torch
import torch.nn as nn
import numpy as np
import json

class DualGrainFeatureRouter(nn.Module):
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
            raise NotImplementedError()

        self.num_splits = 2
        self.normalization_type = normalization_type
        if self.normalization_type == "none":
            self.feature_norm_fine = nn.Identity()
            self.feature_norm_coarse = nn.Identity()
        elif "group" in self.normalization_type:  # like "group-32"
            num_groups = int(self.normalization_type.split("-")[-1])
            self.feature_norm_fine = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
            self.feature_norm_coarse = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
        else:
            raise NotImplementedError()


    def forward(self, h_fine, h_coarse, entropy=None):
        h_fine = self.feature_norm_fine(h_fine)
        h_coarse = self.feature_norm_coarse(h_coarse)

        avg_h_fine = self.gate_pool(h_fine)
        h_logistic = torch.cat([h_coarse, avg_h_fine], dim=1).permute(0,2,3,1)
        
        gate = self.gate(h_logistic) # torch.Size([30, 16, 16, 2])
        return gate


class DualGrainFixedEntropyRouter(nn.Module):
    def __init__(self, json_path, fine_grain_ratito,):
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        self.fine_grain_threshold = content["{}".format(str(int(100 - fine_grain_ratito * 100)))]
    
    def forward(self, h_fine=None, h_coarse=None, entropy=None):
        gate_fine = (entropy > self.fine_grain_threshold).bool().long().unsqueeze(-1)
        gate_coarse = (entropy <= self.fine_grain_threshold).bool().long().unsqueeze(-1)
        gate = torch.cat([gate_coarse, gate_fine], dim=-1)
        return gate