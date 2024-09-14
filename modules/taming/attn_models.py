import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.append(os.getcwd())

from modules.taming.diffusion_modules import (AttnBlock, Normalize,
                                              ResnetBlock, Upsample)


class ResnetBlock_kernel_1(nn.Module):
    """
    A ResNet block with customizable kernel size (1 or 3) and optional time embedding.

    This block performs the following operations:
    1. Normalization and activation of input
    2. Convolution
    3. Optional time embedding projection
    4. Second normalization and activation
    5. Dropout
    6. Second convolution
    7. Residual connection

    Args:
        in_channels (int): Number of input channels.
        out_channels (int, optional): Number of output channels. If None, same as in_channels.
        conv_shortcut (bool): Whether to use convolution for shortcut connection.
        dropout (float): Dropout rate.
        temb_channels (int): Number of time embedding channels. Set to 0 to disable time embedding.
        kernel_size (int): Size of the convolutional kernel (1 or 3).

    """

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        kernel_size=1
    ):
        super().__init__()
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb=None, **ignore_kwargs):
        """
        Forward pass of the ResNet block.

        Args:
            x (torch.Tensor): Input tensor.
            temb (torch.Tensor, optional): Time embedding tensor.
            **ignore_kwargs: Additional keyword arguments (ignored).

        Returns:
            torch.Tensor: Output tensor after passing through the ResNet block.
        """
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class SelfAttnBlock(nn.Module):
    """
    Self-attention block for image-like inputs.

    This module applies self-attention to the input tensor, allowing each position
    to attend to all other positions in the input.

    Args:
        in_channels (int): Number of input channels.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, **ignore_kwargs):
        """
        Forward pass of the self-attention block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            **ignore_kwargs: Additional keyword arguments (ignored).

        Returns:
            torch.Tensor: Output tensor after self-attention, same shape as input.
        """
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))  # Scale dot products
        w_ = torch.nn.functional.softmax(
            w_, dim=2
        )  # Apply softmax to get attention weights

        # Apply attention to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_  # Residual connection


class BiasedSelfAttnBlock(nn.Module):
    """
    Biased self-attention block for image-like inputs with optional mask and reweighting.

    This module applies self-attention to the input tensor, allowing each position
    to attend to all other positions in the input, with the ability to apply a mask
    and optionally reweight the attention.

    Args:
        in_channels (int): Number of input channels.
        reweight (bool): Whether to apply reweighting to the attention weights.
    """

    def __init__(self, in_channels, reweight=False):
        super().__init__()
        self.in_channels = in_channels
        self.apply_reweight = reweight

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, mask, **ignore_kwargs):
        """
        Forward pass of the biased self-attention block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            mask (torch.Tensor): Mask tensor to apply to attention weights.
            **ignore_kwargs: Additional keyword arguments (ignored).

        Returns:
            torch.Tensor: Output tensor after biased self-attention, same shape as input.
        """
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))  # Scale dot products
        w_ = torch.nn.functional.softmax(
            w_, dim=2
        )  # Apply softmax to get attention weights

        if mask is not None:
            unsqueezed_mask = mask.unsqueeze(-2)
            w_ = w_ * unsqueezed_mask  # Apply mask to attention weights

            if self.apply_reweight:
                w_sum = torch.sum(w_, dim=-1, keepdim=True)
                w_ = w_ / w_sum  # Reweight attention weights

        # Apply attention to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_  # Residual connection


class TokenReconstruction(nn.Module):
    """
    A module for token reconstruction using a combination of ResNet blocks and attention mechanisms.

    This module applies a series of ResNet blocks and attention layers to reconstruct tokens,
    with optional mask updating between attention layers.

    Args:
        n_layer (int): Number of attention layers.
        input_dim (int): Input dimension.
        dropout (float): Dropout rate.
        attn_type (str): Type of attention mechanism ('self-attn' or 'bias-self-attn').
        resnet_kernel_size (int): Kernel size for ResNet blocks.
        mask_update_mode (str): Mode for updating the mask ('square', 'cube', 'linear', or 'const').
        reweight (bool): Whether to reweight attention in BiasedSelfAttnBlock.
        fix_bug (bool): Whether to use the fixed version of ResNet block initialization.

    Attributes:
        n_layer (int): Number of attention layers.
        mask_update_mode (str): Mode for updating the mask.
        middle (nn.ModuleList): List of ResNet and attention blocks.
    """

    def __init__(
        self,
        n_layer,
        input_dim,
        dropout,
        attn_type="self-attn",
        resnet_kernel_size=1,
        mask_update_mode="square",
        reweight=False,
        fix_bug=False,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.mask_update_mode = mask_update_mode
        self.middle = nn.ModuleList()

        # Initial ResNet block
        self.middle.append(
            ResnetBlock_kernel_1(
                in_channels=input_dim, dropout=dropout, kernel_size=resnet_kernel_size
            )
        )

        # Alternating attention and ResNet blocks
        for i in range(self.n_layer):
            # Add attention block
            if attn_type == "self-attn":
                self.middle.append(SelfAttnBlock(input_dim))
            elif attn_type == "bias-self-attn":
                self.middle.append(BiasedSelfAttnBlock(input_dim, reweight))
            else:
                raise ValueError("Invalid attention type")

            # Add ResNet block
            if not fix_bug:
                self.middle.append(
                    ResnetBlock_kernel_1(in_channels=input_dim, dropout=dropout),
                )
            else:
                self.middle.append(
                    ResnetBlock_kernel_1(
                        in_channels=input_dim,
                        dropout=dropout,
                        kernel_size=resnet_kernel_size,
                    ),
                )

    def forward(self, x, mask):
        """
        Forward pass of the TokenReconstruction module.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Reconstructed token tensor.
        """
        # Initialize mask update parameters
        if self.mask_update_mode in ["square", "cube"]:
            mask = mask + 0.02 * (1 - mask)
        elif self.mask_update_mode == "linear":
            gain = 1 / (self.n_layer - 1)
            original_mask = mask

        # Process through the middle layers
        for i, module in enumerate(self.middle):
            x = module(x=x, mask=mask)

            # Update mask after each attention layer
            if i % 2 == 1:
                if self.mask_update_mode == "const":
                    pass  # mask remains unchanged
                elif self.mask_update_mode == "square":
                    mask = torch.sqrt(mask)
                elif self.mask_update_mode == "cube":
                    mask = torch.pow(mask, (1 / 3))
                elif self.mask_update_mode == "linear":
                    mask = original_mask + (i // 2 + 1) * gain * (1 - original_mask)
                else:
                    raise ValueError("Invalid mask update mode")

        return x


class AttnDecoder(nn.Module):
    """
    Attention-based decoder for image generation tasks.

    This decoder uses a combination of residual blocks, attention mechanisms,
    and upsampling to generate high-resolution images from low-dimensional latent representations.

    Args:
        ch (int): Base number of channels.
        out_ch (int): Number of output channels (typically 3 for RGB images).
        ch_mult (tuple): Channel multipliers for each resolution level.
        num_res_blocks (int): Number of residual blocks per resolution level.
        attn_resolutions (list): Resolutions at which to apply attention.
        dropout (float): Dropout rate.
        resamp_with_conv (bool): Whether to use convolution for upsampling.
        in_channels (int): Number of input channels.
        resolution (int): Input resolution.
        z_channels (int): Number of channels in the latent space.
        give_pre_end (bool): If True, return the output before the final normalization and convolution.
        token_n_layer (int): Number of layers in the TokenReconstruction module.
        token_attn_type (str): Type of attention to use in the TokenReconstruction module.
        resnet_kernel_size (int): Kernel size for ResNet blocks in TokenReconstruction.
        mask_update_mode (str): Mode for updating the mask in TokenReconstruction.
        reweight (bool): Whether to use reweighting in TokenReconstruction.
        fix_bug (bool): Whether to apply a bug fix in TokenReconstruction.
        **ignorekwargs: Additional keyword arguments (ignored).
    """

    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        token_n_layer=6,
        token_attn_type="self-attn",
        resnet_kernel_size=1,
        mask_update_mode="square",
        reweight=False,
        fix_bug=False,
        **ignorekwargs
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # Compute in_ch_mult, block_in and curr_res at lowest resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # Initial convolution: z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # Middle: TokenReconstruction module
        self.mid = TokenReconstruction(
            n_layer=token_n_layer,
            input_dim=block_in,
            dropout=dropout,
            attn_type=token_attn_type,
            resnet_kernel_size=resnet_kernel_size,
            mask_update_mode=mask_update_mode,
            reweight=reweight,
            fix_bug=fix_bug,
        )

        # Upsampling
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
            self.up.insert(0, up)  # prepend to get consistent order

        # Final normalization and convolution
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x, mask=None):
        """
        Forward pass of the AttnDecoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, z_channels, height, width).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, height * width).
                If None, a default mask of ones will be used.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_ch, output_height, output_width).
        """
        if mask is None:
            mask = torch.ones(x.size(0), x.size(2) * x.size(3)).to(x.device)
        self.last_z_shape = x.shape

        # No timestep embedding in this implementation
        temb = None

        # Initial convolution
        h = self.conv_in(x)

        # Middle: TokenReconstruction
        h = self.mid(h, mask)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Final processing
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


if __name__ == "__main__":
    # Test code for TokenReconstruction
    x = torch.randn(10, 256, 16, 16)
    mask = torch.randint(0, 2, (10, 256))

    model = TokenReconstruction(
        n_layer=6,
        input_dim=256,
        dropout=0.0,
        attn_type="bias-self-attn",
        mask_update_mode="cube",
        reweight=True,
    )

    y = model(x=x, mask=mask)
    print(y.size())

    # Test code for AttnDecoder
    model = AttnDecoder(
        ch=32,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=1,
        attn_resolutions=[],
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=256,
        resolution=256,
        z_channels=256,
        give_pre_end=False,
        token_n_layer=6,
        token_attn_type="bias-self-attn",
        mask_update_mode="square",
    )

    print(model)
    y = model(x=x, mask=mask)
    print(y.size())

    # # Prepare dummy input tensors
    dummy_input = torch.randn(
        1, 256, 16, 16
    )  # Adjust size based on your input requirements
    dummy_mask = torch.randint(0, 2, (10, 256))  # Adjust size based on your mask requirements

    # Export the model
    torch.onnx.export(
        model,  # model being run
        (dummy_input, dummy_mask),  # model input (or a tuple for multiple inputs)
        "attn_decoder.onnx",  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input", "mask"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
