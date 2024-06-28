from torch.utils.data import DataLoader, Dataset
from utils.utils_modules import instantiate_from_config

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a PyTorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataModuleFromConfig:
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None, train_val=False):
        self.batch_size = batch_size
        self.train_val = train_val
        self.num_workers = num_workers if num_workers is not None else batch_size * 2

        self.datasets = self._instantiate_datasets(train, validation, test, wrap)

        if self.train_val:
            if "train" in self.datasets and "validation" in self.datasets:
                self.datasets["train"] += self.datasets["validation"]

        for k, dataset in self.datasets.items():
            print(f"Dataset: {k}, Length: {len(dataset)}")
    
    def _instantiate_datasets(self, train, validation, test, wrap):
        """Instantiates datasets from configurations."""
        datasets = {}
        for split, config in zip(["train", "validation", "test"], [train, validation, test]):
            if config is not None:
                dataset = instantiate_from_config(config)
                if wrap:
                    dataset = WrappedDataset(dataset)
                datasets[split] = dataset
        return datasets

    def get_dataloader(self, split):
        """Returns the DataLoader for the specified split."""
        if split not in self.datasets:
            raise ValueError(f"Invalid split: {split}")

        dataset = self.datasets[split]
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=split == "train",  # Shuffle only training data
            collate_fn=collate_fn,
        )




