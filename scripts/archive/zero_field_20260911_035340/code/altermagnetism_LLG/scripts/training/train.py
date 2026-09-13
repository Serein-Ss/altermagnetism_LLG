"""Train stochastic flow matching or its deterministic same-backbone baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from altermagnetism_LLG.scripts.model.baseline import DeterministicPathModel  # noqa: E402
from altermagnetism_LLG.scripts.datasets.path_data import (  # noqa: E402
    ScalablePathDataset,
    SizeBucketBatchSampler,
    scalar_condition_statistics,
)
from altermagnetism_LLG.scripts.model.network import PeriodicEquivariantFlowNet  # noqa: E402
from altermagnetism_LLG.scripts.model.sphere import (  # noqa: E402
    geodesic_interpolate,
    sample_reference_path,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-type", choices=("flow", "deterministic"), default="flow")
    p.add_argument("--input", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--crop-sizes", type=int, nargs="+", default=None,
        help="per-input crop; use 0 for the complete lattice",
    )
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument(
        "--architecture-version", type=int, choices=(1, 2), default=2
    )
    p.add_argument("--crop-size", type=int, default=None)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true")
    return p


def make_generator(device: str, seed: int) -> torch.Generator:
    generator_device = device if device.startswith("cuda") else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def make_loader(
    dataset: ScalablePathDataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, SizeBucketBatchSampler]:
    sampler = SizeBucketBatchSampler(
        dataset, batch_size, shuffle=shuffle, seed=seed
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def epoch_loss(
    model,
    model_type: str,
    loader: DataLoader,
    device: str,
    *,
    optimizer=None,
    generator: torch.Generator | None = None,
    gradient_accumulation: int = 1,
) -> float:
    training = optimizer is not None
    model.train(training)
    weighted_total = 0.0
    weight_total = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    if training:
        optimizer.zero_grad(set_to_none=True)
    with context:
        for batch_index, batch in enumerate(loader):
            target = batch["spins"].to(device)
            initial = batch["initial"].to(device)
            physical_time = batch["physical_time"].to(device)
            scalar = batch["scalar_condition"].to(device)
            vector = batch["vector_condition"].to(device)
            weights = batch["sample_weight"].to(device)
            if model_type == "flow":
                tau = torch.rand(
                    target.shape[0],
                    device=device,
                    dtype=target.dtype,
                    generator=generator,
                )
                reference = sample_reference_path(
                    initial, target.shape[1], generator=generator
                )
                state, velocity_target = geodesic_interpolate(
                    reference, target, tau
                )
                prediction = model(
                    state, tau, initial, scalar, vector, physical_time
                )
                error = (prediction - velocity_target).square().sum(dim=-1)
            else:
                prediction = model(
                    initial,
                    target.shape[1],
                    scalar,
                    vector,
                    physical_time,
                )
                error = (prediction - target).square().sum(dim=-1)
            per_sample = error.mean(dim=(1, 2, 3, 4))
            loss = (per_sample * weights).sum() / weights.sum()
            if training:
                group_start = (
                    batch_index // gradient_accumulation
                ) * gradient_accumulation
                group_size = min(
                    gradient_accumulation, len(loader) - group_start
                )
                (loss / group_size).backward()
                update = (
                    (batch_index + 1) % gradient_accumulation == 0
                    or batch_index + 1 == len(loader)
                )
                if update:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            weighted_total += float((per_sample.detach() * weights).sum())
            weight_total += float(weights.sum())
    return weighted_total / weight_total


def build_model(model_type: str, config: dict):
    if model_type == "flow":
        return PeriodicEquivariantFlowNet(**config)
    return DeterministicPathModel(**config)


def main() -> None:
    args = parser().parse_args()
    if args.epochs < 1 or args.patience < 1 or args.num_workers < 0:
        raise ValueError("epochs and patience must be positive; num-workers nonnegative")
    if args.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation must be positive")
    if args.crop_size is not None and args.crop_sizes is not None:
        raise ValueError("use crop-size or crop-sizes, not both")
    if (
        args.crop_sizes is not None
        and len(args.crop_sizes) != len(args.input)
    ):
        raise ValueError("crop-sizes must match the number of inputs")
    if args.crop_sizes is not None and min(args.crop_sizes) < 0:
        raise ValueError("crop-sizes entries must be nonnegative")
    crop_sizes = (
        [None if size == 0 else size for size in args.crop_sizes]
        if args.crop_sizes is not None else None
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; submit through a GPU allocation")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    output = args.output or ROOT / "output" / "production" / args.model_type
    checkpoint_path = output / "best_checkpoint.pt"
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite {checkpoint_path}; pass --overwrite"
        )

    train_data = ScalablePathDataset(
        args.input,
        "train",
        args.crop_size,
        random_crop=True,
        crop_sizes=crop_sizes,
    )
    validation_data = ScalablePathDataset(
        args.input,
        "validation",
        args.crop_size,
        random_crop=False,
        crop_sizes=crop_sizes,
    )
    test_data = ScalablePathDataset(
        args.input,
        "test",
        args.crop_size,
        random_crop=False,
        crop_sizes=crop_sizes,
    )
    if not len(train_data) or not len(validation_data) or not len(test_data):
        raise ValueError("train, validation and test splits must all be non-empty")

    train_loader, train_sampler = make_loader(
        train_data,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    validation_loader, _ = make_loader(
        validation_data,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    test_loader, _ = make_loader(
        test_data,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    condition_mean, condition_std = scalar_condition_statistics(
        args.input, "train"
    )
    model_config = {
        "hidden": args.hidden,
        "blocks": args.blocks,
        "scalar_condition_dim": 10,
        "vector_condition_count": 5,
        "architecture_version": args.architecture_version,
        "condition_mean": condition_mean.tolist(),
        "condition_std": condition_std.tolist(),
    }
    model = build_model(args.model_type, model_config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    output.mkdir(parents=True, exist_ok=True)
    training_generator = make_generator(args.device, args.seed + 1)
    history = []
    best = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_loss = epoch_loss(
            model,
            args.model_type,
            train_loader,
            args.device,
            optimizer=optimizer,
            generator=training_generator,
            gradient_accumulation=args.gradient_accumulation,
        )
        validation_generator = make_generator(args.device, args.seed + 2)
        validation_loss = epoch_loss(
            model,
            args.model_type,
            validation_loader,
            args.device,
            generator=validation_generator,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"epoch={epoch} train={train_loss:.6g} "
            f"validation={validation_loss:.6g}",
            flush=True,
        )
        if validation_loss < best:
            best = validation_loss
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_type": args.model_type,
                    "model_config": model_config,
                    "training_config": {
                        "input": [str(path) for path in args.input],
                        "output": str(output),
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "crop_size": args.crop_size,
                        "crop_sizes": args.crop_sizes,
                        "gradient_accumulation": args.gradient_accumulation,
                        "architecture_version": args.architecture_version,
                        "hidden": args.hidden,
                        "blocks": args.blocks,
                        "lr": args.lr,
                        "patience": args.patience,
                        "num_workers": args.num_workers,
                        "seed": args.seed,
                        "device": args.device,
                    },
                    "epoch": epoch,
                    "validation_loss": best,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stop epoch={epoch}", flush=True)
                break

    checkpoint = torch.load(
        checkpoint_path, map_location=args.device, weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"])
    test_generator = make_generator(args.device, args.seed + 3)
    test_loss = epoch_loss(
        model,
        args.model_type,
        test_loader,
        args.device,
        generator=test_generator,
    )
    metrics = {
        "status": "trained_and_internal_tested",
        "model_type": args.model_type,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_trajectories": len(train_data),
        "validation_trajectories": len(validation_data),
        "test_trajectories": len(test_data),
        "best_epoch": checkpoint["epoch"],
        "best_validation_loss": best,
        "test_loss": test_loss,
        "history": history,
        "claim_boundary": (
            "training/test loss alone does not establish path-distribution "
            "or large-size physical generalization"
        ),
    }
    (output / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    train_data.close()
    validation_data.close()
    test_data.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
