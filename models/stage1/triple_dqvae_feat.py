from typing import List, Tuple

import torch

from models.stage1.base_stage1 import Stage1Model
from utils.utils_modules import instantiate_from_config


class TripleGrainVQModel(Stage1Model):
    """
    Triple Grain VQ Model for image generation.

    This module implements a triple grain vector quantization model. It consists of an encoder,
    a decoder, a vector quantization layer, and a discriminator for adversarial training.

    Args:
        encoderconfig: Configuration for the encoder.
        decoderconfig: Configuration for the decoder.
        lossconfig: Configuration for the loss function.
        vqconfig: Configuration for the vector quantization layer.
        quant_before_dim: Number of channels before quantization.
        quant_after_dim: Number of channels after quantization.
        quant_sample_temperature: Temperature for sampling from the quantized embeddings.
        ckpt_path: Path to a checkpoint file to load weights from.
        ignore_keys: List of keys to ignore when loading weights from a checkpoint.
        image_key: Key for the input image in the input batch dictionary.
    """

    def __init__(
        self,
        encoderconfig,
        decoderconfig,
        lossconfig,
        vqconfig,
        quant_before_dim: int,
        quant_after_dim: int,
        quant_sample_temperature: float = 0.0,
        ckpt_path: str = None,
        ignore_keys: List[str] = [],
        image_key: str = "image",
    ):
        super().__init__()

        self.image_key = image_key
        self.encoder = instantiate_from_config(encoderconfig)
        self.decoder = instantiate_from_config(decoderconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = instantiate_from_config(vqconfig)

        self.quant_conv = torch.nn.Conv2d(quant_before_dim, quant_after_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(quant_after_dim, quant_before_dim, 1)
        self.quant_sample_temperature = quant_sample_temperature

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict, torch.Tensor, torch.Tensor]:
        """
        Encodes an input image into quantized embeddings.

        Args:
            x: Input image tensor.

        Returns:
            A tuple containing:
                - Quantized embeddings.
                - Embedding loss.
                - Information dictionary from the quantization layer.
                - Grain indices.
                - Gate values.
        """
        h_dict = self.encoder(x, None)
        h = h_dict["h_triple"]
        grain_indices = h_dict["indices"]
        codebook_mask = h_dict["codebook_mask"]
        gate = h_dict["gate"]

        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(
            x=h, temp=self.quant_sample_temperature, codebook_mask=codebook_mask
        )
        return quant, emb_loss, info, grain_indices, gate

    def decode(
        self, quant: torch.Tensor, grain_indices: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Decodes quantized embeddings into an image.

        Args:
            quant: Quantized embeddings.
            grain_indices: Grain indices.

        Returns:
            Decoded image tensor.
        """
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, grain_indices=None)
        return dec

    def decode_code(self, code_b):
        """
        Decodes quantized codes into an image.

        Args:
            code_b: Quantized codes.

        Returns:
            Decoded image tensor.
        """
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input: Input image tensor.

        Returns:
            A tuple containing:
                - Decoded image tensor.
                - Quantization loss.
                - Grain indices.
                - Gate values.
        """
        quant, diff, _, grain_indices, gate = self.encode(input)
        dec = self.decode(quant, grain_indices)
        return dec, diff, grain_indices, gate
