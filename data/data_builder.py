from torch.utils.data import DataLoader, Dataset

from utils.utils_modules import \
    instantiate_from_config  # Assuming this function doesn't depend on PyTorch Lightning


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a PyTorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig:
    def __init__(
        self,
        batch_size,
        train=None,
        val=None,
        test=None,
        wrap=False,
        num_workers=None,
        train_val=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_val = train_val
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if train is not None:
            self.dataset_configs["train"] = train
        if val is not None:
            self.dataset_configs["val"] = val
        if test is not None:
            self.dataset_configs["test"] = test
        self.wrap = wrap

        self.datasets = self.setup_datasets()

    def setup_datasets(self):
        datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )
        if self.wrap:
            for k in datasets:
                datasets[k] = WrappedDataset(datasets[k])

        if self.train_val:
            if "train" in datasets and "val" in datasets:
                datasets["train"] = datasets["train"] + datasets["val"]
        return datasets

    def get_dataloader(self, split):
        dataset = self.datasets[split]
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=split == "train",
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("val")

    def test_dataloader(self):
        return self.get_dataloader("test")
