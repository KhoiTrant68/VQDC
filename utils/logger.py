import os

import numpy as np
import torch
import torchvision
import wandb
from omegaconf import OmegaConf
from PIL import Image


class SetupCallback:
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, argv_content=None):
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.argv_content = argv_content

    def on_training_start(self):
        # Create necessary directories
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)
        os.makedirs(self.cfgdir, exist_ok=True)

        # Print and save project configuration
        print("Project config")
        print(OmegaConf.to_yaml(self.config))
        OmegaConf.save(
            self.config, os.path.join(self.cfgdir, f"{self.now}-project.yaml")
        )

        # Save command-line arguments to a text file
        if self.argv_content:
            with open(os.path.join(self.logdir, "argv_content.txt"), "w") as f:
                f.write(str(self.argv_content))


class CaptionImageLogger:
    def __init__(self, batch_frequency, max_images, clamp=True, type="wandb"):
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.clamp = clamp
        self.type = type

    def log_img(
        self,
        model,
        batch,
        batch_idx,
        split="train",
        mode=None,
        epoch=None,
        step=None,
        accelerator=None,
    ):
        # Check if it's time to log images
        if (
            (batch_idx % self.batch_freq == 0)
            and hasattr(model, "log_images")
            and callable(model.log_images)
            and (self.max_images > 0)
        ):
            is_train = model.training
            if is_train:
                model.eval()

            with torch.no_grad():
                images = model.log_images(
                    batch=batch, device=accelerator.device, mode=mode, split=split
                )

            # Remove text captions from images
            self.remove_text_captions(images)

            # Prepare images for logging
            self.prepare_images(images)

            # Gather images from all processes if using distributed training
            if accelerator is not None:
                images = accelerator.gather(images)

            # Log images to WandB and save locally
            if accelerator is not None and accelerator.is_main_process:
                self.log_to_wandb(images, split, step)
                self.log_local(
                    accelerator.project_dir, split, images, step, epoch, batch_idx
                )

            if is_train:
                model.train()

    def remove_text_captions(self, images):
        # Remove text captions from images
        if "groundtruth_captions" in images:
            del images["groundtruth_captions"]
        if "dest_captions" in images:
            del images["dest_captions"]
        if "sample_captions" in images:
            del images["sample_captions"]

    def prepare_images(self, images):
        # Prepare images for logging
        for k in images:
            N = min(images[k].shape[0], self.max_images)
            images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

    def log_to_wandb(self, images, split, step):
        # Log images to WandB
        if self.type == "wandb":
            grids = {}
            for k in images:
                grid = torchvision.utils.make_grid(images[k], normalize=True)
                grids[f"{split}/{k}"] = wandb.Image(grid)
            wandb.log(grids, step=step)

    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        # Save images locally
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4, normalize=True)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = f"Step_{global_step:06}-Epoch_{current_epoch:03}-Batch_{batch_idx:06}-{k}.png"
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)
