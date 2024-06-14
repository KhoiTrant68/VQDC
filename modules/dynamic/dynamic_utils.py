import importlib
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


def draw_dual_grain_256res_color(
    images=None, indices=None, low_color="blue", high_color="red", scaler=0.9
):
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

        score_map_i_blend = Image.fromarray(
            np.uint8(high * score_map_i_np + low * (1 - score_map_i_np))
        )
        image_i_blend = Image.blend(image_i_pil, score_map_i_blend, scaler)

        blended_images.append(
            torchvision.transforms.functional.to_tensor(image_i_blend)
        )

    return torch.stack(blended_images, dim=0)


def draw_triple_grain_256res_color(
    images=None, indices=None, low_color="blue", high_color="red", scaler=0.25
):
    """
    Draw triple-grain images based on indices.
    """
    if images is None:
        images = torch.ones(indices.size(0), 3, 256, 256)

    indices = indices.unsqueeze(1)
    size = 256 // indices.size(-1)
    indices = indices.repeat_interleave(size, dim=-1).repeat_interleave(size, dim=-2)
    indices = indices.float() / 2.0  # Normalize indices to [0, 1]

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

        score_map_i_blend = Image.fromarray(
            np.uint8(high * score_map_i_np + low * (1 - score_map_i_np))
        )


        image_i_blend = Image.blend(image_i_pil.convert("RGB") , score_map_i_blend, scaler)

        blended_images.append(
            torchvision.transforms.functional.to_tensor(image_i_blend)
        )

    return torch.stack(blended_images, dim=0)


def instantiate_from_config(config):
    def get_obj_from_str(string, reload=False):
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), cls)

    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """

    def gumbel_softmax_sample(logits, temperature=1, eps=1e-20):
        U = torch.rand(logits.shape).to(logits.device)
        sampled_gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        y = logits + sampled_gumbel_noise
        return F.softmax(y / temperature, dim=-1)

    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


if __name__ == "__main__":
    test_image_path = "D:\\AwesomeCV\\VQDC\\test.jpg"
    images = Image.open(test_image_path)
    indices = torch.randint(0, 3, (4, 8, 8))
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  
            transforms.ToTensor(),  
        ]
    )
    image_tensor = transform(images)
    # images = draw_dual_grain_256res_color(indices=indices)
    images = draw_triple_grain_256res_color(images=image_tensor, indices=indices)
    torchvision.utils.save_image(images, "test_draw_triple_grain.png")
