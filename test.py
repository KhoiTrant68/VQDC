import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from einops import rearrange

# Corrected color dictionary
color_dict = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "blue": (0, 0, 255),
}

transform_PIL = transforms.ToPILImage()

def image_normalize(tensor, value_range=None, scale_each=False):
    tensor = tensor.clone()

    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    if scale_each:
        for t in tensor:  # loop over mini-batch dimension
            low, high = (
                value_range
                if value_range is not None
                else (float(t.min()), float(t.max()))
            )
            norm_ip(t, low, high)
    else:
        low, high = (
            value_range
            if value_range is not None
            else (float(tensor.min()), float(tensor.max()))
        )
        norm_ip(tensor, low, high)

    return tensor

def draw_triple_grain_256res_color(
    images=None, indices=None, low_color="blue", mid_color="yellow", high_color="red", scaler=0.1
):
    """
    Draw triple-grain images based on indices.
    """

    if images is None:
        images = torch.ones(indices.size(0), 3, 256, 256)
    indices = indices.unsqueeze(1)
    size = 256 // indices.size(-1)
    indices = indices.repeat_interleave(size, dim=-1).repeat_interleave(size, dim=-2)
    indices = indices.float()
    print("triple", indices.shape)

    bs = images.size(0)
    low_color_rgb = color_dict[low_color]
    mid_color_rgb = color_dict[mid_color]
    high_color_rgb = color_dict[high_color]

    blended_images = []

    for i in range(bs):
        image_i_pil = transform_PIL(image_normalize(images[i]))

        score_map_i_np = rearrange(indices[i], "C H W -> H W C").cpu().detach().numpy()
        score_map_i_np = np.clip(score_map_i_np, 0, 2)  # Ensure values are in [0, 2]

        low = np.array(Image.new("RGB", (images.size(-1), images.size(-2)), low_color_rgb))
        mid = np.array(Image.new("RGB", (images.size(-1), images.size(-2)), mid_color_rgb))
        high = np.array(Image.new("RGB", (images.size(-1), images.size(-2)), high_color_rgb))

        low_mask = (score_map_i_np < 1).astype(np.float32)
        mid_mask = ((score_map_i_np >= 1) & (score_map_i_np < 2)).astype(np.float32)
        high_mask = (score_map_i_np >= 2).astype(np.float32)

        score_map_i_blend = (
            low * low_mask + mid * mid_mask + high * high_mask
        ).astype(np.uint8)
        score_map_i_blend = Image.fromarray(score_map_i_blend)

        image_i_blend = Image.blend(
            image_i_pil.convert("RGB"), score_map_i_blend, scaler
        )

        blended_images.append(
            torchvision.transforms.functional.to_tensor(image_i_blend)
        )

    return torch.stack(blended_images, dim=0)

def draw_dual_grain_256res_color(
    images=None, indices=None, low_color="blue", high_color="red", scaler=0.1
):
    """
    Draw dual-grain images based on indices.
    """

    if images is None:
        images = torch.ones(indices.size(0), 3, 256, 256)
    indices = indices.unsqueeze(1)
    size = 256 // indices.size(-1)
    indices = indices.repeat_interleave(size, dim=-1).repeat_interleave(size, dim=-2)
    indices = indices.float()
    print("dual", indices.shape)

    bs = images.size(0)
    low_color_rgb = color_dict[low_color]
    high_color_rgb = color_dict[high_color]

    blended_images = []

    for i in range(bs):
        image_i_pil = transform_PIL(image_normalize(images[i]))

        score_map_i_np = rearrange(indices[i], "C H W -> H W C").cpu().detach().numpy()
        score_map_i_np = np.clip(score_map_i_np, 0, 1)  # Ensure values are in [0, 1]

        low = np.array(Image.new("RGB", (images.size(-1), images.size(-2)), low_color_rgb))
        high = np.array(Image.new("RGB", (images.size(-1), images.size(-2)), high_color_rgb))

        score_map_i_blend = (high * score_map_i_np + low * (1 - score_map_i_np)).astype(np.uint8)
        score_map_i_blend = Image.fromarray(score_map_i_blend)

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
    indices_dual = torch.randint(0, 2, (4, 8, 8))  # Adjusted to 1 for single image
    indices_triple = torch.randint(0, 3, (4, 8, 8))  # Adjusted to 1 for single image
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    image_tensor = transform(images)

    images_dual = draw_dual_grain_256res_color(
        images=image_tensor.unsqueeze(0), indices=indices_dual, scaler=0.5
    )
    torchvision.utils.save_image(images_dual, "test_draw_dual_grain.png")

    images_triple = draw_triple_grain_256res_color(
        images=image_tensor.unsqueeze(0), indices=indices_triple, scaler=0.5
    )
    torchvision.utils.save_image(images_triple, "test_draw_triple_grain.png")
