from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset(Dataset):
    def __init__(self, split, paths, size=None, random_crop=False, labels=None):
        self.paths = sorted(paths)
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths

        if split == "train":
            self.transforms = transforms.Compose(
                [
                    (
                        transforms.RandomResizedCrop(256)
                        if self.random_crop
                        else transforms.Resize((256, 256))
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

    def __len__(self):
        return len(self.paths)

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        return image

    def __getitem__(self, i):
        return {
            "input": self.preprocess_image(self.labels["file_path_"][i]),
            **{k: v[i] for k, v in self.labels.items()},
        }
