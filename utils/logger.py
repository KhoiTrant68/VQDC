import os
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image
import wandb

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
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)
        os.makedirs(self.cfgdir, exist_ok=True)
        print("Project config")
        print(OmegaConf.to_yaml(self.config))
        OmegaConf.save(self.config, os.path.join(self.cfgdir, f"{self.now}-project.yaml"))
        if self.argv_content:
            with open(os.path.join(self.logdir, "argv_content.txt"), "w") as f:
                f.write(str(self.argv_content))

class CaptionImageLogger:
    def __init__(self, batch_frequency, max_images, clamp=True):
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.clamp = clamp

    def log_img(self, model, batch, batch_idx, split="train", mode=None, epoch=None, step=None, accelerator=None):
        if batch_idx % self.batch_freq == 0:
            is_train = model.training
            if is_train:
                model.eval()
            with torch.no_grad():
                images = model.module.log_images(batch=batch, device=accelerator.device, mode=mode, split=split)
            self.remove_text_captions(images)
            self.prepare_images(images)
            for k in images:
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].to(accelerator.device)
            if accelerator is not None:
                images = accelerator.gather(images)
            if accelerator is not None and accelerator.is_main_process:
                self.log_to_wandb(images, split, step)
                self.log_local(accelerator.project_dir, split, images, step, epoch, batch_idx)
            if is_train:
                model.train()
        else:
            print("not logging")

    def remove_text_captions(self, images):
        if "groundtruth_captions" in images:
            del images["groundtruth_captions"]
        if "dest_captions" in images:
            del images["dest_captions"]
        if "sample_captions" in images:
            del images["sample_captions"]

    def prepare_images(self, images):
        for k in images:
            N = min(images[k].shape[0], self.max_images)
            images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

    def log_to_wandb(self, images, split, step):
        grids = {}
        for k in images:
            grid = torchvision.utils.make_grid(images[k], normalize=True)
            grids[f"{split}/{k}"] = wandb.Image(grid)
        wandb.log(grids, step=step)

    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4, normalize=True)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.detach().cpu().numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = f"Step_{global_step:06}-Epoch_{current_epoch:03}-Batch_{batch_idx:06}-{k}.png"
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)