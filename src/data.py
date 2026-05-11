from __future__ import annotations

from pathlib import Path
from typing import Tuple
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from sklearn.datasets import fetch_california_housing

CIFAR10_NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_NORMALIZE_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_DIRNAME = "cifar-10-batches-py"
MNIST_NORMALIZE_MEAN = (0.1307,)
MNIST_NORMALIZE_STD = (0.3081,)
MNIST_DIRNAME = "MNIST"


class Cifar10Dataset(Dataset):
    def __init__(self, root: Path, subset_size: int):
        cifar_batches = (
            root
            if root.name == CIFAR10_DIRNAME and root.exists()
            else root / CIFAR10_DIRNAME
        )
        if not cifar_batches.exists():
            raise FileNotFoundError(f"CIFAR-10 root does not exist: {cifar_batches}")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
                category=Warning,
            )
            self.dataset = CIFAR10(
                root=str(cifar_batches.parent),
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


class MnistDataset(Dataset):
    def __init__(self, root: Path, subset_size: int):
        mnist_root = (
            root.parent
            if root.name == MNIST_DIRNAME and root.exists()
            else root
        )
        mnist_dir = mnist_root / MNIST_DIRNAME
        if not mnist_dir.exists():
            raise FileNotFoundError(f"MNIST root does not exist: {mnist_dir}")

        self.dataset = MNIST(
            root=str(mnist_root),
            train=False,
            download=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(MNIST_NORMALIZE_MEAN, MNIST_NORMALIZE_STD),
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


class CaliforniaHousingDataset(Dataset):
    def __init__(self, root: Path, subset_size: int, seed: int):
        dataset = fetch_california_housing(
            data_home=str(root / "sklearn"),
            download_if_missing=True,
            return_X_y=True,
            as_frame=False,
        )
        features, targets = dataset

        generator = np.random.default_rng(seed)
        permutation = generator.permutation(features.shape[0])
        eval_count = max(1, int(0.2 * features.shape[0]))
        eval_indices = permutation[-eval_count:]
        train_indices = permutation[:-eval_count]

        train_features = features[train_indices].astype(np.float32)
        eval_features = features[eval_indices].astype(np.float32)
        eval_targets = targets[eval_indices].astype(np.float32)

        feature_mean = train_features.mean(axis=0, keepdims=True)
        feature_std = train_features.std(axis=0, keepdims=True)
        feature_std = np.where(feature_std == 0.0, 1.0, feature_std)

        normalized_features = (eval_features - feature_mean) / feature_std
        limit = (
            min(int(subset_size), normalized_features.shape[0])
            if subset_size > 0
            else normalized_features.shape[0]
        )

        self.features = torch.from_numpy(normalized_features[:limit]).to(torch.float32)
        self.targets = torch.from_numpy(eval_targets[:limit]).to(torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]
