import os
import tqdm
import glob
import shutil
import tarfile

import numpy as np
from omegaconf import OmegaConf

from torch.utils.data import Dataset

from data.utils_data import (
    retrieve,
    download,
    mark_prepared,
    is_prepared,
    str_to_indices,
    give_synsets_from_indices,
)
from data.imagenet_base import BaseDataset
from data.default import DefaultDataPath


class ImageNetBase(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not isinstance(self.config, dict):
            self.config = OmegaConf.to_container(self.config)
        self._prepare()
        # self._prepare_synset_to_human()
        # self._prepare_idx_to_synset()
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
            files = []
            for rpath in relpaths:
                syn = rpath.split("/")[0]
                if syn in synsets:
                    files.append(rpath)
            return files
        else:
            return relpaths

    # def _prepare_synset_to_human(self):
    #     SIZE = 2655750
    #     self.human_dict = os.path.join(self.write_root, "synset_human.txt")
    #     assert os.path.exists(self.human_dict)
    #     assert os.path.getsize(self.human_dict) == SIZE

    # def _prepare_idx_to_synset(self):
    #     self.idx2syn = os.path.join(self.write_root, "imagenet_idx_to_synset.yml")
    #     assert os.path.exists(self.idx2syn)

    def _load(self):
        with open(self.txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines()
            l1 = len(self.relpaths)
            self.relpaths = self._filter_relpaths(self.relpaths)
            print(
                f"Removed {l1 - len(self.relpaths)} files from filelist during filtering."
            )

        self.synsets = [p.split("/")[0] for p in self.relpaths]
        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

        unique_synsets = np.unique(self.synsets)
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        self.class_labels = [class_dict[s] for s in self.synsets]

        # with open(self.human_dict, "r") as f:
        #     human_dict = f.read().splitlines()
        #     human_dict = dict(line.split(maxsplit=1) for line in human_dict)

        # self.human_labels = [human_dict[s] for s in self.synsets]

        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            # "human_label": np.array(self.human_labels),
        }
        self.data = BaseDataset(
            split=self.split,
            paths=self.abspaths,
            labels=labels,
            size=retrieve(self.config, "size", default=0),
            random_crop=self.random_crop,
        )


class ImageNetTrain(ImageNetBase):
    NAME = "ILSVRC2012_train"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "a306397ccf9c2ead27155983c254227c0fd938e2"
    FILES = [
        "ILSVRC2012_img_train.tar",
    ]
    SIZES = [
        147897477120,
    ]

    def _prepare(self):
        self.random_crop = retrieve(
            self.config, "ImageNetTrain/random_crop", default=True
        )
        self.root = DefaultDataPath.ImageNet.root
        self.write_root = DefaultDataPath.ImageNet.train_write_root

        self.split = "train"
        self.datadir = os.path.join(self.root, "train")
        self.txt_filelist = os.path.join(self.write_root, "filelist.txt")
        self.expected_length = 1281167
        if not is_prepared(self.write_root):
            print(f"Preparing dataset {self.NAME} in {self.root}")
            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                print('path--------------',path)
                assert os.path.exists(path)
                assert os.path.getsize(path) == self.SIZES[0]

                print(f"Extracting {path} to {datadir}")
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                print("Extracting sub-tars.")
                subpaths = sorted(glob.glob(os.path.join(datadir, "*.tar")))
                for subpath in tqdm.tqdm(subpaths):
                    subdir = subpath[: -len(".tar")]
                    os.makedirs(subdir, exist_ok=True)
                    with tarfile.open(subpath, "r:") as tar:
                        tar.extractall(path=subdir)

            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist) + "\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            mark_prepared(self.write_root)


class ImageNetValidation(ImageNetBase):
    NAME = "ILSVRC2012_validation"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5"
    VS_URL = "https://heibox.uni-heidelberg.de/f/3e0f6e9c624e45f2bd73/?dl=1"
    FILES = [
        "ILSVRC2012_img_val.tar",
        "validation_synset.txt",
    ]
    SIZES = [
        6744924160,
        1950000,
    ]

    def _prepare(self):
        self.random_crop = retrieve(
            self.config, "ImageNetValidation/random_crop", default=False
        )
        self.root = DefaultDataPath.ImageNet.root
        self.write_root = DefaultDataPath.ImageNet.val_write_root

        self.split = "val"
        self.datadir = os.path.join(self.root, "val")
        self.txt_filelist = os.path.join(self.write_root, "filelist.txt")

        self.expected_length = 50000
        if not is_prepared(self.write_root):
            print(f"Preparing dataset {self.NAME} in {self.root}")
            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or os.path.getsize(path) != self.SIZES[0]:
                    download(self.AT_HASH, datastore=self.root)

                print(f"Extracting {path} to {datadir}")
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                vspath = os.path.join(self.root, self.FILES[1])
                if (
                    not os.path.exists(vspath)
                    or os.path.getsize(vspath) != self.SIZES[1]
                ):
                    download(self.VS_URL, vspath)

                with open(vspath, "r") as f:
                    synset_dict = f.read().splitlines()
                    synset_dict = dict(line.split() for line in synset_dict)

                print("Reorganizing into synset folders")
                synsets = np.unique(list(synset_dict.values()))
                for s in synsets:
                    os.makedirs(os.path.join(datadir, s), exist_ok=True)
                for k, v in synset_dict.items():
                    src = os.path.join(datadir, k)
                    dst = os.path.join(datadir, v)
                    shutil.move(src, dst)

            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist) + "\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            mark_prepared(self.write_root)
