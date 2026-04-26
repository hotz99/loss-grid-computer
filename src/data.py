from __future__ import annotations

from pathlib import Path
from typing import Tuple
import warnings

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


CIFAR10_NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_NORMALIZE_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_ROOT = Path("assets")


class Cifar10Dataset(Dataset):
    def __init__(self, subset_size: int):
        cifar_batches = CIFAR10_ROOT / "cifar-10-batches-py"
        if not cifar_batches.exists():
            raise FileNotFoundError(f"CIFAR-10 root does not exist: {cifar_batches}")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
                category=Warning,
            )
            self.dataset = CIFAR10(
                root=str(CIFAR10_ROOT),
                train=False,
                download=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR10_NORMALIZE_MEAN, CIFAR10_NORMALIZE_STD),
                    ]
                ),
            )
        self.limit = (
            min(int(subset_size), len(self.dataset))
            if subset_size > 0
            else len(self.dataset)
        )

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features, label = self.dataset[index]
        return features, torch.tensor(label, dtype=torch.long)
