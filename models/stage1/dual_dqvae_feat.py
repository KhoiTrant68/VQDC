from typing import Dict, List, Tuple

import torch
from torch import nn

from utils.utils_modules import instantiate_from_config
from modules.dynamic.dynamic_utils import draw_dual_grain_256res_color


class DualGrainVQModel(nn.Module):
    """
    Dual Grain VQ Model for image generation.

    This module implements a dual grain vector quantization model. It consists of an encoder,
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

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = []):
        """
        Initializes the model weights from a checkpoint file.

        Args:
            path: Path to the checkpoint file.
            ignore_keys: List of keys to ignore when loading weights.
        """
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

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
        h = h_dict["h_dual"]
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
        dec = self.decoder(quant, grain_indices)
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

    def get_input(self, batch: dict, k: str) -> torch.Tensor:
        """
        Extracts and preprocesses the input image from a batch dictionary.

        Args:
            batch: Batch dictionary.
            k: Key for the input image.

        Returns:
            Preprocessed input image tensor.
        """
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.size(1) != 3:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def calculate_loss(
        self,
        x: torch.Tensor,
        xrec: torch.Tensor,
        qloss: torch.Tensor,
        step: int,
        optimizer_idx: int,
        gate: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculates the loss for either autoencoder or discriminator.

        Args:
            x: Input image tensor.
            xrec: Reconstructed image tensor.
            qloss: Quantization loss.
            step: Current training step (epoch or global step).
            optimizer_idx: Index of the optimizer being used (0 for AE, 1 for Disc).
            gate: Gate values.

        Returns:
            A tuple containing:
                - Calculated loss.
                - Log dictionary.
        """
        if optimizer_idx == 0:
            loss, log_dict = self.loss(
                qloss,
                x,
                xrec,
                optimizer_idx,
                step,
                last_layer=self.get_last_layer(),
                split="train",
                gate=gate,
            )
        else:
            loss, log_dict = self.loss(
                qloss,
                x,
                xrec,
                optimizer_idx,
                step,
                last_layer=self.get_last_layer(),
                split="train",
            )
        return loss, log_dict

    def get_last_layer(self) -> torch.Tensor:
        """
        Returns the weights of the last layer of the decoder.

        Returns:
            Weights of the last decoder layer.
        """
        try:
            return self.decoder.conv_out.weight
        except AttributeError:
            return self.decoder.last_layer

    def log_images(self, batch: dict, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Logs input images, reconstructions, grain maps, and entropy maps.

        Args:
            batch: Batch dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary containing logged image tensors.
        """
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _, grain_indices, gate = self(x)
        log["inputs"] = x
        log["reconstructions"] = xrec
        log["grain_color"] = draw_dual_grain_256res_color(
            images=x.clone(), indices=grain_indices, scaler=0.7
        )
        return log

    def get_code_emb_with_depth(self, code):
        return self.quantize.embed_code_with_depth(code)
