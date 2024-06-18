import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.diffusion.model import (
    AttnBlock,
    Downsample,
    Normalize,
    ResnetBlock,
    nonlinearity,
)
from modules.dynamic.dynamic_utils import instantiate_from_config

sys.path.append(os.getcwd())


class MiddleBlock(nn.Module):
    def __init__(self, block_in, dropout, temb_ch):
        super(MiddleBlock, self).__init__()
        self.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=temb_ch,
            dropout=dropout,
        )
        self.attn_1 = AttnBlock(block_in)
        self.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=temb_ch,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.block_1(x, None)
        x = self.attn_1(x)
        x = self.block_2(x, None)
        return x


class DualGrainEncoder(nn.Module):
    def __init__(
        self,
        ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=256,
        router_config=None,
        update_router=True,
        **ignore_kwargs,
    ):
        super().__init__()

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.update_router = update_router

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # Downsampling layers
        self.down, block_in = self._make_downsampling_layers(
            ch, ch_mult, attn_resolutions, dropout, resamp_with_conv
        )

        # Middle blocks
        self.mid_coarse = MiddleBlock(block_in, dropout, self.temb_ch)
        self.norm_out_coarse = Normalize(block_in)
        self.conv_out_coarse = nn.Conv2d(
            block_in, z_channels, kernel_size=3, stride=1, padding=1
        )

        block_in_finegrain = block_in // (ch_mult[-1] // ch_mult[-2])
        
        self.mid_fine = MiddleBlock(block_in_finegrain, dropout, self.temb_ch)
        self.norm_out_fine = Normalize(block_in_finegrain)
        self.conv_out_fine = nn.Conv2d(
            block_in_finegrain, z_channels, kernel_size=3, stride=1, padding=1
        )

        # Router
        self.router = instantiate_from_config(router_config)

    def _make_downsampling_layers(
        self, ch, ch_mult, attn_resolutions, dropout, resamp_with_conv
    ):
        layers = nn.ModuleList()
        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * in_ch_mult[0]

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res //= 2
            layers.append(down)
        return layers, block_in

    def forward(self, x, x_entropy):
        assert (
            x.shape[2] == x.shape[3] == self.resolution
        ), f"{x.shape[2]}, {x.shape[3]}, {self.resolution}"

        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            h = hs[-1]
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, None)
                if len(self.down[i_level].attn) > i_block:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
            if i_level == self.num_resolutions - 2:
                h_fine = h

        # Middle for h_coarse
        h_coarse = hs[-1]
        h_coarse = self.mid_coarse(h_coarse)
        h_coarse = self.norm_out_coarse(h_coarse)
        h_coarse = nonlinearity(h_coarse)
        h_coarse = self.conv_out_coarse(h_coarse)

        # Middle for h_fine
        h_fine = self.mid_fine(h_fine)
        h_fine = self.norm_out_fine(h_fine)
        h_fine = nonlinearity(h_fine)
        h_fine = self.conv_out_fine(h_fine)

        # Dynamic routing
        gate = self.router(h_fine=h_fine, h_coarse=h_coarse, entropy=x_entropy)
        if self.update_router and self.training:
            gate = F.gumbel_softmax(gate, dim=-1, hard=True)
        gate = gate.permute(0, 3, 1, 2)
        indices = gate.argmax(dim=1)

        h_coarse = h_coarse.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
        indices_repeat = (
            indices.repeat_interleave(2, dim=-1)
            .repeat_interleave(2, dim=-2)
            .unsqueeze(1)
        )
        h_dual = torch.where(indices_repeat == 0, h_coarse, h_fine)

        if self.update_router and self.training:
            gate_grad = gate.max(dim=1, keepdim=True)[0]
            gate_grad = gate_grad.repeat_interleave(2, dim=-1).repeat_interleave(
                2, dim=-2
            )
            h_dual = h_dual * gate_grad

        coarse_mask = 0.25 * torch.ones_like(indices_repeat).to(h_dual.device)
        fine_mask = 1.0 * torch.ones_like(indices_repeat).to(h_dual.device)
        codebook_mask = torch.where(indices_repeat == 0, coarse_mask, fine_mask)

        return {
            "h_dual": h_dual,
            "indices": indices,
            "codebook_mask": codebook_mask,
            "gate": gate,
        }
