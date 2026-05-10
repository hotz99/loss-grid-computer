#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import CIFAR10_NORMALIZE_MEAN, CIFAR10_NORMALIZE_STD
from src.models.row_gru import RowGRUClassifier


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_dataset, val_dataset, test_dataset = build_datasets(
        args.data_root,
        args.val_size,
        args.seed,
        args.max_train_samples,
    )
    train_loader = build_loader(train_dataset, args.batch_size, True, device, args.num_workers)
    val_loader = build_loader(val_dataset, args.batch_size, False, device, args.num_workers)
    test_loader = build_loader(test_dataset, args.batch_size, False, device, args.num_workers)

    model = RowGRUClassifier().to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)

    best_val_accuracy = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.amp,
        )
        scheduler.step()
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6g}",
            flush=True,
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {
                key: value.detach().cpu()
                for key, value in unwrap_model(model).state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state")

    unwrap_model(model).load_state_dict(best_state, strict=True)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output)
    print(
        f"saved={args.output} "
        f"best_val_acc={best_val_accuracy:.4f} "
        f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export the Task B CIFAR-10 row-GRU checkpoint.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("assets"))
    parser.add_argument("--output", type=Path, default=Path("assets/cifar10-row-gru-0.pkl"))
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_NORMALIZE_MEAN, CIFAR10_NORMALIZE_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_NORMALIZE_MEAN, CIFAR10_NORMALIZE_STD),
        ]
    )
    train_full = CIFAR10(
        root=str(data_root),
        train=True,
        download=False,
        transform=train_transform,
    )
    val_full = CIFAR10(
        root=str(data_root),
        train=True,
        download=False,
        transform=eval_transform,
    )
    test_dataset = CIFAR10(
        root=str(data_root),
        train=False,
        download=False,
        transform=eval_transform,
    )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(train_full), generator=generator).tolist()
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    if max_train_samples > 0:
        train_indices = train_indices[:max_train_samples]
    return (
        Subset(train_full, train_indices),
        Subset(val_full, val_indices),
        test_dataset,
    )


def build_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and amp):
            logits = model(inputs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

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


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


if __name__ == "__main__":
    main()
