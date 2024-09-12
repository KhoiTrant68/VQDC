import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from modules.dynamic.fourier_embedding import FourierPositionEmbedding
from modules.taming.diffusion_modules import (AttnBlock, Normalize,
                                              ResnetBlock, Upsample)


class PositionEmbedding2DLearned(nn.Module):
    """
    Learned 2D position embedding module.

    This module learns separate embeddings for row and column positions
    and combines them to create a 2D position embedding.

    Args:
        n_row (int): Number of rows.
        feats_dim (int): Dimensionality of the embedding features.
        n_col (int, optional): Number of columns (defaults to n_row).
    """

    def __init__(self, n_row, feats_dim, n_col=None):
        super().__init__()
        n_col = n_col if n_col is not None else n_row
        self.row_embed = nn.Embedding(n_row, feats_dim)
        self.col_embed = nn.Embedding(n_col, feats_dim)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the embedding weights using truncated normal distribution.
        """
        torch.nn.init.trunc_normal_(self.row_embed.weight)
        torch.nn.init.trunc_normal_(self.col_embed.weight)

    def forward(self, x):
        """
        Forward pass of the position embedding module.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W) or (B, C, T, H, W).

        Returns:
            torch.Tensor: Input tensor with added position embeddings.
        """
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i).unsqueeze(0).repeat(h, 1, 1)
        y_emb = self.row_embed(j).unsqueeze(1).repeat(1, w, 1)
        pos = (x_emb + y_emb).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        # Handle relative position embedding (5D input)
        if x.dim() == 5:
            pos = pos.unsqueeze(-3)

        x = x + pos
        return x


class Decoder(nn.Module):
    """
    Decoder module for a generative model.

    This decoder upsamples a latent representation through a series of
    residual blocks, attention blocks, and upsampling layers. It supports
    different position embedding techniques for enhanced spatial awareness.

    Args:
        ch (int): Base channel size.
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        ch_mult (tuple of int): Channel multipliers for each resolution level.
        num_res_blocks (int): Number of residual blocks per resolution level.
        resolution (int): Output resolution.
        attn_resolutions (list of int): Resolutions to apply attention blocks.
        dropout (float, optional): Dropout probability (default: 0.0).
        resamp_with_conv (bool, optional): Use convolution for upsampling (default: True).
        give_pre_end (bool, optional): Return output before final layers (default: False).
        latent_size (int, optional): Size of the latent representation (default: 32).
        window_size (int, optional): Window size for relative position embedding (default: 2).
        position_type (str, optional): Type of position embedding
                                       ("relative", "learned", "fourier", "fourier+learned", default: "relative").
    """

    def __init__(
        self,
        ch,
        in_ch,
        out_ch,
        ch_mult,
        num_res_blocks,
        resolution,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        give_pre_end=False,
        latent_size=32,
        window_size=2,
        position_type="relative",
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        self.temb_ch = 0  # Placeholder for time embedding channels (not used)
        self.ch = ch
        self.give_pre_end = give_pre_end

        # Calculate input block channels and initial resolution
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, in_ch, curr_res, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # Input convolution layer
        self.conv_in = torch.nn.Conv2d(
            in_ch, block_in, kernel_size=3, stride=1, padding=1
        )

        # Middle layers (residual blocks and attention)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # Upsampling layers
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
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
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # Prepend to maintain order

        # Output layers (normalization and convolution)
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

        # Position embedding setup
        self.position_type = position_type
        if self.position_type == "learned":
            self.position_bias = PositionEmbedding2DLearned(
                n_row=latent_size, feats_dim=in_ch
            )
        elif self.position_type == "learned-relative":
            self.position_bias = PositionEmbedding2DLearned(
                n_row=window_size, feats_dim=in_ch
            )
            self.window_size = window_size
            self.window_num = latent_size // window_size
        elif self.position_type == "fourier":
            self.position_bias = FourierPositionEmbedding(
                coord_size=latent_size, hidden_size=in_ch
            )
        elif self.position_type == "fourier+learned":
            self.position_bias_fourier = FourierPositionEmbedding(
                coord_size=latent_size, hidden_size=in_ch
            )
            self.position_bias_learned = PositionEmbedding2DLearned(
                n_row=latent_size, feats_dim=in_ch
            )
        else:
            raise NotImplementedError(
                f"Position embedding type '{self.position_type}' not implemented."
            )

    def forward(self, h, grain_indices):
        """
        Forward pass of the decoder module.

        Args:
            h (torch.Tensor): Latent representation tensor.
            grain_indices (torch.Tensor, optional): Grain indices (not used in this implementation).

        Returns:
            torch.Tensor: Upsampled output tensor.
        """
        # Apply position embedding
        if self.position_type == "learned":
            h = self.position_bias(h)
        elif self.position_type == "learned-relative":
            h = rearrange(
                h,
                "B C (n1 nH) (n2 nW) -> B C (n1 n2) nH nW",
                n1=self.window_num,
                nH=self.window_size,
                n2=self.window_num,
                nW=self.window_size,
            )
            h = self.position_bias(h)
            h = rearrange(
                h,
                "B C (n1 n2) nH nW -> B C (n1 nH) (n2 nW)",
                n1=self.window_num,
                nH=self.window_size,
            )
        elif self.position_type == "fourier":
            h = self.position_bias(h)
        elif self.position_type == "fourier+learned":
            h = self.position_bias_fourier(h)
            h = self.position_bias_learned(h)

        # Initial convolution
        temb = None  # Placeholder for time embedding
        h = self.conv_in(h)

        # Middle layers
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Upsampling layers
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Return intermediate output if specified
        if self.give_pre_end:
            return h

        # Output layers
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h
