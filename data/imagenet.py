import os
from glob import glob
from typing import Any

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from data.default import DefaultDataPath
from data.imagenet_base import BaseDataset
from data.utils_data import retrieve


class ImageNetBase(Dataset):
    """Base class for ImageNet datasets."""

    def __init__(self, split: str = None, config: str = None):
        self.split = split
        self.config = config or {}  # Use an empty dict if config is None
        if not isinstance(self.config, dict):
            self.config = OmegaConf.to_container(self.config)

        self.prepare()
        self.load()

    def prepare(self):
        """Prepare the dataset (implemented in subclasses)."""
        raise NotImplementedError()

    def load(self):
        """Load image paths from the prepared filelist."""
        with open(self.txt_file, "r") as f:
            self.relpaths = f.read().splitlines()

        self.abspaths = [os.path.join(self.data_dir, p) for p in self.relpaths][:200]
        self.data = BaseDataset(
            split=self.split,
            paths=self.abspaths,
            size=self.config.get("size", 0),  # Use .get for optional keys
            random_crop=self.random_crop,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:
        return self.data[index]


class ImageNetDataset(ImageNetBase):
    """ImageNet dataset implementation."""

    def __init__(self, split: str = None, config: str = None):
        # Initialize self.split and self.config here
        self.split = split
        self.config = config or {}
        if not isinstance(self.config, dict):
            self.config = OmegaConf.to_container(self.config)

        super().__init__(self.split, self.config)  # Pass split and config to superclass

    def prepare(self):
        self.random_crop = self.config.get("ImageNetTrain/random_crop", True)
        self.root = DefaultDataPath.ImageNet.root

        self.write_root = getattr(DefaultDataPath.ImageNet, f"{self.split}_write_root")
        self.data_dir = os.path.join(self.root, self.split)
        # self.txt_file = os.path.join(self.write_root, "filelist.txt")
        self.txt_file = os.path.join("/kaggle/working", "filelist.txt")

        with open(self.txt_file, "w") as f:
            for p in glob(os.path.join(self.data_dir, "**", "*.JPEG")):
                f.write(os.path.relpath(p, self.data_dir) + "\n")


# Example usage (uncommented)
if __name__ == "__main__":
    config = {"size": 512}
    dset_train = ImageNetDataset(split="train", config=config)
    dset_val = ImageNetDataset(split="val", config=config)

    dloader = DataLoader(dset_train, batch_size=16, shuffle=True)
    dloader_val = DataLoader(dset_val, batch_size=16, shuffle=True)

    print(len(dloader))
    print(len(dloader_val))
