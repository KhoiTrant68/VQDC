import numpy as np

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as F

from PIL import Image
from einops import rearrange

color_dict = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "blue": (5, 39, 175),
}

transform_PIL = transforms.Compose([transforms.ToPILImage()])


def image_normalize(tensor, value_range=None, scale_each=False):
    tensor = tensor.clone()
    
    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))
    
    if scale_each:
        for t in tensor:  # loop over mini-batch dimension
            low, high = (value_range if value_range is not None 
                         else (float(t.min()), float(t.max())))
            norm_ip(t, low, high)
    else:
        low, high = (value_range if value_range is not None 
                     else (float(tensor.min()), float(tensor.max())))
        norm_ip(tensor, low, high)
    
    return tensor

def draw_dual_grain_256res_color_old(
    images=None, indices=None, low_color="blue", high_color="red", scaler=0.9
):
    # indices: [batch_size, height, weight]
    # 0 for coarse-grained and 1 for fine-grained
    if images is None:
        images = torch.ones(indices.size(0), 3, 256, 256)
    indices = indices.unsqueeze(1)
    size = 256 // indices.size(-1)
    indices = indices.repeat_interleave(size, dim=-1).repeat_interleave(size, dim=-2)

    bs = images.size(0)

    low = Image.new("RGB", (images.size(-2), images.size(-1)), color_dict[low_color])
    high = Image.new("RGB", (images.size(-2), images.size(-1)), color_dict[high_color])

    for i in range(bs):
        image_i_pil = transform_PIL(image_normalize(images[i]))
        score_map_i_np = rearrange(indices[i], "C H W -> H W C").cpu().detach().numpy()
        score_map_i_blend = Image.fromarray(
            np.uint8(high * score_map_i_np + low * (1 - score_map_i_np))
        )

        image_i_blend = Image.blend(image_i_pil, score_map_i_blend, scaler)

        if i == 0:
            blended_images = torchvision.transforms.functional.to_tensor(
                image_i_blend
            ).unsqueeze(0)
        else:
            blended_images = torch.cat(
                [
                    blended_images,
                    torchvision.transforms.functional.to_tensor(
                        image_i_blend
                    ).unsqueeze(0),
                ],
                dim=0,
            )
    return blended_images

def draw_dual_grain_256res_color(images=None, indices=None, low_color="blue", high_color="red", scaler=0.9):
    """
    Draw dual-grain images based on indices.
    """
    if images is None:
        images = torch.ones(indices.size(0), 3, 256, 256)
    indices = indices.unsqueeze(1)
    size = 256 // indices.size(-1)
    indices = indices.repeat_interleave(size, dim=-1).repeat_interleave(size, dim=-2)

    bs = images.size(0)

    low_color_rgb = color_dict[low_color]
    high_color_rgb = color_dict[high_color]

    blended_images = []
    
    for i in range(bs):
        image_i_pil = transform_PIL(image_normalize(images[i]))

        score_map_i_np = rearrange(indices[i], "C H W -> H W C").cpu().detach().numpy()
        score_map_i_np = np.clip(score_map_i_np, 0, 1)  # Ensure values are in [0, 1]

        low = Image.new("RGB", (images.size(-1), images.size(-2)), low_color_rgb)
        high = Image.new("RGB", (images.size(-1), images.size(-2)), high_color_rgb)

        score_map_i_blend = Image.fromarray(np.uint8(high * score_map_i_np + low * (1 - score_map_i_np)))
        image_i_blend = Image.blend(image_i_pil, score_map_i_blend, scaler)

        blended_images.append(torchvision.transforms.functional.to_tensor(image_i_blend))

    return torch.stack(blended_images, dim=0)


# Example usage:
images = torch.randn(2, 3, 256, 256)  
indices = torch.randint(0, 2, (2, 32, 32))  
blended_images_old = draw_dual_grain_256res_color_old(images, indices)
blended_images_new = draw_dual_grain_256res_color(images, indices)

assert torch.allclose(blended_images_old, blended_images_new, atol=1e-3), "The new function does not match the old function's output."

