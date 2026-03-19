from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from loss_grid.config import DataConfig


class Cifar10Dataset(Dataset):
    def __init__(self, config: DataConfig):
        root = Path(config.root)
        if not root.exists():
            raise FileNotFoundError(f"CIFAR-10 root does not exist: {root}")

        split = config.split.lower()
        if split == "train":
            batch_paths = [root / f"data_batch_{index}" for index in range(1, 6)]
        elif split == "test":
            batch_paths = [root / "test_batch"]
        else:
            raise ValueError(f"Unsupported CIFAR-10 split: {config.split}")

        features = []
        labels = []
        for batch_path in batch_paths:
            if not batch_path.exists():
                raise FileNotFoundError(f"Missing CIFAR-10 batch file: {batch_path}")
            with batch_path.open("rb") as handle:
                payload = pickle.load(handle, encoding="bytes")
            batch_features = torch.from_numpy(payload[b"data"]).to(torch.float32).reshape(-1, 3, 32, 32)
            batch_features = batch_features / 255.0
            batch_labels = torch.tensor(payload[b"labels"], dtype=torch.long)
            features.append(batch_features)
            labels.append(batch_labels)

        all_features = torch.cat(features, dim=0)
        all_labels = torch.cat(labels, dim=0)

        if config.subset_size > 0:
            limit = min(int(config.subset_size), int(all_features.shape[0]))
            all_features = all_features[:limit]
            all_labels = all_labels[:limit]

        mean = torch.tensor(config.normalize_mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(config.normalize_std, dtype=torch.float32).view(3, 1, 1)
        self.features = (all_features - mean) / std
        self.labels = all_labels

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def build_dataloader(
    config: DataConfig, batch_size_override: int | None = None
) -> DataLoader:
    if config.name != "cifar10":
        raise ValueError("Unsupported dataset. Expected 'cifar10'.")

    dataset = Cifar10Dataset(config)
    return DataLoader(
        dataset,
        batch_size=batch_size_override or config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
