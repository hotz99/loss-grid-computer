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
from sklearn.datasets import fetch_california_housing
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.schemas import DatasetSpec, MLTaskSpec  # noqa: E402

WORKLOAD_NAME = "california_mlp_regression"
MODEL_NAME = "mlp_regressor"
DEFAULT_CHECKPOINT_PATH = Path("assets/california-mlp-0.pkl")


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
        args.seed,
        args.test_fraction,
        args.val_fraction,
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
    criterion = nn.MSELoss()

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

    best_val_mse = float("inf")
    best_state: OrderedDict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        train_mse, train_mae = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_mse, val_mae = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch={epoch:03d} "
            f"train_mse={train_mse:.6f} train_mae={train_mae:.6f} "
            f"val_mse={val_mse:.6f} val_mae={val_mae:.6f}",
            flush=True,
        )
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = OrderedDict(
                (key, value.detach().cpu())
                for key, value in model.state_dict().items()
            )

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state")

    model.load_state_dict(best_state, strict=True)
    test_mse, test_mae = evaluate(model, test_loader, criterion, device)
    smoke_count = min(args.smoke_samples, len(test_dataset))
    smoke_loader = build_loader(
        TensorDataset(
            test_dataset.tensors[0][:smoke_count],
            test_dataset.tensors[1][:smoke_count],
        ),
        args.batch_size,
        False,
        device,
        args.num_workers,
        args.seed,
    )
    smoke_mse, smoke_mae = evaluate(model, smoke_loader, criterion, device)
    if not np.isfinite(smoke_mse):
        raise RuntimeError(f"Smoke evaluation produced non-finite MSE: {smoke_mse}")

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
                "best_val_mse": best_val_mse,
                "test_mse": test_mse,
                "test_mae": test_mae,
                "smoke_samples": smoke_count,
                "smoke_mse": smoke_mse,
                "smoke_mae": smoke_mae,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the California MLP checkpoint for loss-grid workloads.",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
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
            "california_housing",
            "assets",
            (8,),
            1024,
        ),
        MODEL_NAME,
        "tabular_regression",
        "mse",
        str(DEFAULT_CHECKPOINT_PATH),
    )
    from src.models.mlp_regressor import build_model

    return spec, build_model, "src.models.mlp_regressor"


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
    seed: int,
    test_fraction: float,
    val_fraction: float,
    max_train_samples: int,
    download: bool,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be in (0, 1)")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")

    features, targets = fetch_california_housing(
        data_home=str(data_root / "sklearn"),
        download_if_missing=download,
        return_X_y=True,
        as_frame=False,
    )
    features = features.astype(np.float32)
    targets = targets.astype(np.float32)

    generator = np.random.default_rng(seed)
    permutation = generator.permutation(features.shape[0])
    test_count = max(1, int(test_fraction * features.shape[0]))
    test_indices = permutation[-test_count:]
    train_pool_indices = permutation[:-test_count]
    val_count = max(1, int(val_fraction * train_pool_indices.shape[0]))
    val_indices = train_pool_indices[-val_count:]
    train_indices = train_pool_indices[:-val_count]

    if train_indices.size == 0:
        raise ValueError("No training samples remain after split; reduce val/test fractions")

    if max_train_samples > 0:
        train_indices = train_indices[: min(max_train_samples, train_indices.size)]

    train_features = features[train_indices]
    val_features = features[val_indices]
    test_features = features[test_indices]
    train_targets = targets[train_indices]
    val_targets = targets[val_indices]
    test_targets = targets[test_indices]

    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_std = train_features.std(axis=0, keepdims=True)
    feature_std = np.where(feature_std == 0.0, 1.0, feature_std)

    train_norm = (train_features - feature_mean) / feature_std
    val_norm = (val_features - feature_mean) / feature_std
    test_norm = (test_features - feature_mean) / feature_std

    train_dataset = TensorDataset(
        torch.from_numpy(train_norm).to(torch.float32),
        torch.from_numpy(train_targets).to(torch.float32),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_norm).to(torch.float32),
        torch.from_numpy(val_targets).to(torch.float32),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_norm).to(torch.float32),
        torch.from_numpy(test_targets).to(torch.float32),
    )
    return train_dataset, val_dataset, test_dataset


def build_loader(
    dataset: TensorDataset,
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
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs).squeeze(-1)
        mse = criterion(predictions, targets)
        mse.backward()
        optimizer.step()

        batch_count = int(targets.shape[0])
        mae = torch.mean(torch.abs(predictions - targets))
        total_mse += float(mse.detach().cpu()) * batch_count
        total_mae += float(mae.detach().cpu()) * batch_count
        total_count += batch_count

    if total_count == 0:
        raise RuntimeError("Training loader produced zero samples")
    return total_mse / total_count, total_mae / total_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device, dtype=torch.float32, non_blocking=True)
        predictions = model(inputs).squeeze(-1)
        mse = criterion(predictions, targets)
        mae = torch.mean(torch.abs(predictions - targets))
        batch_count = int(targets.shape[0])
        total_mse += float(mse.detach().cpu()) * batch_count
        total_mae += float(mae.detach().cpu()) * batch_count
        total_count += batch_count

    if total_count == 0:
        raise RuntimeError("Evaluation loader produced zero samples")
    return total_mse / total_count, total_mae / total_count


def verify_checkpoint_loads(
    checkpoint_path: Path,
    spec: MLTaskSpec,
    build_model: Callable[[MLTaskSpec], nn.Module],
) -> None:
    loaded = build_model(replace(spec, checkpoint_path=str(checkpoint_path)))
    state_dict = loaded.state_dict()
    if not state_dict:
        raise RuntimeError(f"Loaded checkpoint has empty state dict: {checkpoint_path}")


if __name__ == "__main__":
    main()
