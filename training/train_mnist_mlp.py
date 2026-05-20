#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.data import MNIST_NORMALIZE_MEAN, MNIST_NORMALIZE_STD  # noqa: E402
except ImportError:
    MNIST_NORMALIZE_MEAN = (0.1307,)
    MNIST_NORMALIZE_STD = (0.3081,)
from src.schemas import DatasetSpec, MLTaskSpec  # noqa: E402

WORKLOAD_NAME = "mnist_mlp_classification"
MODEL_NAME = "mnist_mlp"
DEFAULT_CHECKPOINT_PATH = Path("assets/mnist-mlp-0.pkl")


class FallbackMnistMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    spec, build_model, model_source = resolve_model_builder()
    workload_checkpoint = (
        Path(spec.checkpoint_path)
        if spec.checkpoint_path is not None
        else DEFAULT_CHECKPOINT_PATH
    )
    data_root = args.data_root if args.data_root is not None else Path(spec.dataset.path)
    output = args.output if args.output is not None else workload_checkpoint
    train_dataset, val_dataset, test_dataset = build_datasets(
        data_root,
        args.val_size,
        args.seed,
        args.max_train_samples,
        args.download,
    )
    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        True,
        device,
        args.num_workers,
        args.seed,
    )
    val_loader = build_loader(
        val_dataset,
        args.batch_size,
        False,
        device,
        args.num_workers,
        args.seed,
    )
    test_loader = build_loader(
        test_dataset,
        args.batch_size,
        False,
        device,
        args.num_workers,
        args.seed,
    )

    model = build_model(replace(spec, checkpoint_path=None)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    print(
        json.dumps(
            {
                "event": "start",
                "workload": WORKLOAD_NAME,
                "model_source": model_source,
                "seed": args.seed,
                "device": str(device),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset),
                "data_root": str(data_root),
                "output": str(output),
                "registered_checkpoint_path": str(workload_checkpoint),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    best_val_accuracy = -1.0
    best_state: OrderedDict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}",
            flush=True,
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = OrderedDict(
                (key, value.detach().cpu())
                for key, value in model.state_dict().items()
            )

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state")

    model.load_state_dict(best_state, strict=True)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    smoke_loss, smoke_accuracy = evaluate(
        model,
        build_loader(
            Subset(test_dataset, range(min(args.smoke_samples, len(test_dataset)))),
            args.batch_size,
            False,
            device,
            args.num_workers,
            args.seed,
        ),
        criterion,
        device,
    )
    if not np.isfinite(smoke_loss):
        raise RuntimeError(f"Smoke evaluation produced non-finite loss: {smoke_loss}")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output)
    verify_checkpoint_loads(output, spec, build_model)
    print(
        json.dumps(
            {
                "event": "saved",
                "output": str(output),
                "registered_checkpoint_path": str(workload_checkpoint),
                "matches_registered_checkpoint": output == workload_checkpoint,
                "best_val_accuracy": best_val_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "smoke_samples": min(args.smoke_samples, len(test_dataset)),
                "smoke_loss": smoke_loss,
                "smoke_accuracy": smoke_accuracy,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MNIST MLP checkpoint for loss-grid workloads.",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=int, default=10_000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--smoke-samples", type=int, default=512)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_model_builder() -> tuple[
    MLTaskSpec,
    Callable[[MLTaskSpec], nn.Module],
    str,
]:
    try:
        from src.workloads import WORKLOADS

        definition = WORKLOADS.get(WORKLOAD_NAME)
        if definition is not None:
            return definition.spec, definition.build_model, "src.workloads"
    except (ImportError, AttributeError):
        pass

    spec = MLTaskSpec(
        WORKLOAD_NAME,
        DatasetSpec(
            "mnist",
            "assets/mnist",
            (1, 28, 28),
            1024,
        ),
        MODEL_NAME,
        "image_classification",
        "cross_entropy",
        str(DEFAULT_CHECKPOINT_PATH),
    )
    try:
        from src.models.mnist_mlp import build_model

        return spec, build_model, "src.models.mnist_mlp"
    except ImportError:
        return spec, lambda _spec: FallbackMnistMLP(), "fallback"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def build_datasets(
    data_root: Path,
    val_size: int,
    seed: int,
    max_train_samples: int,
    download: bool,
) -> tuple[Subset, Subset, MNIST]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_NORMALIZE_MEAN, MNIST_NORMALIZE_STD),
        ]
    )
    train_full = MNIST(
        root=str(data_root),
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = MNIST(
        root=str(data_root),
        train=False,
        download=download,
        transform=transform,
    )
    if val_size <= 0 or val_size >= len(train_full):
        raise ValueError(f"val_size must be in [1, {len(train_full) - 1}]")

    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        train_full,
        [train_size, val_size],
        generator=generator,
    )
    if max_train_samples > 0:
        train_dataset = Subset(
            train_dataset,
            range(min(max_train_samples, len(train_dataset))),
        )
    return train_dataset, val_dataset, test_dataset


def build_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().detach().cpu())
        total += batch_size
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().detach().cpu())
        total += batch_size
    return total_loss / max(1, total), correct / max(1, total)


def verify_checkpoint_loads(
    checkpoint_path: Path,
    spec: MLTaskSpec,
    build_model: Callable[[MLTaskSpec], nn.Module],
) -> None:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")
    cleaned = OrderedDict(
        (
            key.removeprefix("module."),
            value.to(torch.float32) if torch.is_floating_point(value) else value,
        )
        for key, value in state_dict.items()
    )
    model = build_model(replace(spec, checkpoint_path=None))
    model.load_state_dict(cleaned, strict=True)


if __name__ == "__main__":
    main()
