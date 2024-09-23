import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torchvision import transforms

# Define color dictionary outside functions for reusability
COLOR_DICT = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "blue": (0, 0, 255),
}

# Use torchvision.transforms.ToPILImage() directly instead of defining a variable
transform_to_pil = transforms.ToPILImage()


def image_normalize(tensor, value_range=None, scale_each=False):
    """
    Normalizes a tensor to a given value range.

    Args:
        tensor (torch.Tensor): The input tensor.
        value_range (tuple, optional): The desired value range (low, high).
            If None, uses the minimum and maximum of the tensor. Defaults to None.
        scale_each (bool, optional): Whether to scale each element in the tensor
            independently. Defaults to False.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    tensor = tensor.clone()

    def _norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    if scale_each:
        for t in tensor:
            low, high = value_range or (float(t.min()), float(t.max()))
            _norm_ip(t, low, high)
    else:
        low, high = value_range or (float(tensor.min()), float(tensor.max()))
        _norm_ip(tensor, low, high)

    return tensor


def _draw_grain_base(indices, low_color, high_color, scaler=0.1):
    """
    Base function to draw grain images based on indices.

    Args:
        indices (torch.Tensor): Indices tensor with shape (B, H, W).
        low_color (str): Color for low values.
        high_color (str): Color for high values.
        scaler (float, optional): Blending factor for the grain. Defaults to 0.1.

    Returns:
        torch.Tensor: Blended image tensor with shape (B, 3, H, W).
    """
    indices = indices.unsqueeze(1).float()
    size = 256 // indices.size(-1)
    indices = indices.repeat_interleave(size, dim=-1).repeat_interleave(size, dim=-2)

    low_color_rgb = COLOR_DICT[low_color]
    high_color_rgb = COLOR_DICT[high_color]

    # Use broadcasting for more efficient color array creation
    low = np.array(low_color_rgb, dtype=np.uint8)[None, None, :]
    high = np.array(high_color_rgb, dtype=np.uint8)[None, None, :]

    # Vectorized operations for performance improvement
    score_map_np = rearrange(indices, "B C H W -> B H W C").cpu().numpy()
    score_map_np = np.clip(score_map_np, 0, 1)  # Clip values to 0-1 range

    score_map_blend = (high * score_map_np + low * (1 - score_map_np)).astype(np.uint8)
    score_map_blend = np.transpose(score_map_blend, (0, 3, 1, 2))

    return score_map_blend


def draw_dual_grain_256res_color(
    images=None, indices=None, low_color="blue", high_color="red", scaler=0.1
):
    """
    Draw dual-grain images based on indices.

    Args:
        images (torch.Tensor, optional): Base images tensor. Defaults to None.
        indices (torch.Tensor): Indices tensor with shape (B, H, W).
        low_color (str): Color for low values.
        high_color (str): Color for high values.
        scaler (float, optional): Blending factor for the grain. Defaults to 0.1.

    Returns:
        torch.Tensor: Blended image tensor with shape (B, 3, H, W).
    """
    if images is None:
        images = torch.ones(indices.size(0), 3, 256, 256)

    score_map_blend = _draw_grain_base(indices, low_color, high_color, scaler)

    # Apply blending with base images
    blended_images = []
    for i in range(images.size(0)):
        image_i_pil = transform_to_pil(image_normalize(images[i]))
        # score_map_i_blend = Image.fromarray(score_map_blend[i])
        score_map_i_blend = Image.fromarray(score_map_blend[i].astype(np.uint8))
        image_i_blend = Image.blend(
            image_i_pil.convert("RGB"), score_map_i_blend, scaler
        )
        blended_images.append(
            torchvision.transforms.functional.to_tensor(image_i_blend)
        )

    return torch.stack(blended_images, dim=0)


def draw_triple_grain_256res_color(
    images=None,
    indices=None,
    low_color="blue",
    mid_color="yellow",
    high_color="red",
    scaler=0.1,
):
    """
    Draw triple-grain images based on indices.

    Args:
        images (torch.Tensor, optional): Base images tensor. Defaults to None.
        indices (torch.Tensor): Indices tensor with shape (B, H, W).
        low_color (str): Color for low values.
        mid_color (str): Color for middle values.
        high_color (str): Color for high values.
        scaler (float, optional): Blending factor for the grain. Defaults to 0.1.

    Returns:
        torch.Tensor: Blended image tensor with shape (B, 3, H, W).
    """
    if images is None:
        images = torch.ones(indices.size(0), 3, 256, 256)

    indices = indices.unsqueeze(1).float()
    size = 256 // indices.size(-1)
    indices = indices.repeat_interleave(size, dim=-1).repeat_interleave(size, dim=-2)

    low_color_rgb = COLOR_DICT[low_color]
    mid_color_rgb = COLOR_DICT[mid_color]
    high_color_rgb = COLOR_DICT[high_color]

    # Use broadcasting for more efficient color array creation
    low = np.array(low_color_rgb, dtype=np.uint8)[None, None, :]
    mid = np.array(mid_color_rgb, dtype=np.uint8)[None, None, :]
    high = np.array(high_color_rgb, dtype=np.uint8)[None, None, :]

    # Vectorized operations for performance improvement
    score_map_np = rearrange(indices, "B C H W -> B H W C").cpu().numpy()
    score_map_np = np.clip(score_map_np, 0, 2)  # Clip values to 0-2 range

    low_mask = (score_map_np < 1).astype(np.float32)
    mid_mask = ((score_map_np >= 1) & (score_map_np < 2)).astype(np.float32)
    high_mask = (score_map_np >= 2).astype(np.float32)

    score_map_blend = (low * low_mask + mid * mid_mask + high * high_mask).astype(
        np.uint8
    )
    score_map_blend = np.transpose(score_map_blend, (0, 3, 1, 2))

    # Apply blending with base images
    blended_images = []
    for i in range(images.size(0)):
        image_i_pil = transform_to_pil(image_normalize(images[i]))
        score_map_i_blend = Image.fromarray(score_map_blend[i])
        image_i_blend = Image.blend(
            image_i_pil.convert("RGB"), score_map_i_blend, scaler
        )
        blended_images.append(
            torchvision.transforms.functional.to_tensor(image_i_blend)
        )

    return torch.stack(blended_images, dim=0)


if __name__ == "__main__":
    test_image_path = "D:\\AwesomeCV\\VQDC\\test.jpg"
    images = Image.open(test_image_path)
    indices_dual = torch.randint(0, 2, (4, 8, 8))
    indices_triple = torch.randint(0, 3, (4, 8, 8))
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    image_tensor = transform(images)

    images_dual = draw_dual_grain_256res_color(
        images=image_tensor.unsqueeze(0), indices=indices_dual, scaler=0.1
    )
    torchvision.utils.save_image(images_dual, "test_draw_dual_grain.png")

    images_triple = draw_triple_grain_256res_color(
        images=image_tensor.unsqueeze(0), indices=indices_triple, scaler=0.1
    )
    torchvision.utils.save_image(images_triple, "test_draw_triple_grain.png")
