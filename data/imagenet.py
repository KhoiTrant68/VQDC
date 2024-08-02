import os
import glob
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils_data import retrieve, is_prepared, mark_prepared
from imagenet_base import BaseDataset
from default import DefaultDataPath

def str_to_indices(str):
    ranges = str.split(",")
    indices = []
    for r in ranges:
        if "-" in r:
            s, e = r.split("-")
            indices.extend(list(range(int(s), int(e) + 1)))
        else:
            indices.append(int(r))
    return indices


def give_synsets_from_indices(indices, path_to_yaml="data/imagenet_idx_to_synset.yml"):
    import yaml
    with open(path_to_yaml, "r") as f:
        idx2syn = yaml.load(f)
    return [idx2syn[i] for i in indices]

class ImageNetBase(BaseDataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not isinstance(self.config, dict):
            self.config = OmegaConf.to_container(self.config)
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        raise NotImplementedError()

    def _filter_relpaths(self, relpaths):
        ignore = {"n06596364_9591.JPEG"}
        relpaths = [rpath for rpath in relpaths if rpath.split("/")[-1] not in ignore]
        if "sub_indices" in self.config:
            indices = str_to_indices(self.config["sub_indices"])
            synsets = give_synsets_from_indices(indices, path_to_yaml=self.idx2syn)
            return [rpath for rpath in relpaths if rpath.split("/")[0] in synsets]
        return relpaths

    def _load(self):
        with open(self.txt_filelist, "r") as f:
            self.relpaths = self._filter_relpaths(f.read().splitlines())

        self.synsets = [p.split("/")[0] for p in self.relpaths]
        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

        unique_synsets, self.class_labels = np.unique(self.synsets, return_inverse=True)

        self.data = BaseDataset(
            split=self.split,
            paths=self.abspaths,
            labels={"class_label": self.class_labels},
            size=retrieve(self.config, "size", default=0),
            random_crop=self.random_crop,
        )


class ImageNetTrain(ImageNetBase):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not isinstance(self.config, dict):
            self.config = OmegaConf.to_container(self.config)
        self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop", default=True)
        self.split = "train"
        self.root = DefaultDataPath.ImageNet.root
        self.write_root = DefaultDataPath.ImageNet.train_write_root
        self.datadir = os.path.join(self.root, "train")
        self.txt_filelist = os.path.join(self.write_root, "filelist.txt")
        self.idx2syn = os.path.join(self.write_root, "imagenet_idx_to_synset.yml")
        print(self.idx2syn)

        super().__init__(config=self.config)

        # Assuming data already exists, no preparation needed

class ImageNetValidation(ImageNetBase):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not isinstance(self.config, dict):
            self.config = OmegaConf.to_container(self.config)
        self.random_crop = retrieve(self.config, "ImageNetValidation/random_crop", default=False)
        self.split = "val"
        self.root = DefaultDataPath.ImageNet.root
        self.write_root = DefaultDataPath.ImageNet.val_write_root
        self.datadir = os.path.join(self.root, "val")
        self.txt_filelist = os.path.join(self.write_root, "filelist.txt")
        self.idx2syn = os.path.join(self.write_root, "imagenet_idx_to_synset.yml")

        super().__init__(config=self.config)

        # Assuming data already exists, no preparation needed


if __name__ == "__main__":
    config = {"is_eval": False, "size": 512}
    dset = ImageNetTrain(config)
    dset_val = ImageNetValidation(config)

    # Use Accelerate for DataLoader
    from accelerate import Accelerator
    accelerator = Accelerator()
    dloader = DataLoader(dset, batch_size=4, num_workers=0, shuffle=True)
    dloader_val = DataLoader(dset_val, batch_size=4, num_workers=0, shuffle=True)

    # Prepare dataloaders for distributed training
    dloader = accelerator.prepare(dloader)
    dloader_val = accelerator.prepare(dloader_val)

    print(len(dloader))
    print(len(dloader_val))

