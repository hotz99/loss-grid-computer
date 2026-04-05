from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.config import DataConfig

CIFAR10_NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_NORMALIZE_STD = (0.2470, 0.2435, 0.2616)


class Cifar10Dataset(Dataset):
    def __init__(self, config: DataConfig):
        root = Path(config.root)
        if not root.exists():
            raise FileNotFoundError(f"CIFAR-10 root does not exist: {root}")

        batch_paths = [root / "test_batch"]

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

        mean = torch.tensor(CIFAR10_NORMALIZE_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(CIFAR10_NORMALIZE_STD, dtype=torch.float32).view(3, 1, 1)
        self.features = (all_features - mean) / std
        self.labels = all_labels

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def build_dataloader(
    config: DataConfig, batch_size_override: int | None = None
) -> DataLoader[Cifar10Dataset]:
    dataset = Cifar10Dataset(config)
    return DataLoader(
        dataset,
        batch_size=batch_size_override or config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
