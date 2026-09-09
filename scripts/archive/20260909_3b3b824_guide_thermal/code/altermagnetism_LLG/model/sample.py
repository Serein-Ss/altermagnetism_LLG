"""Sample a complete spin path from a trained flow or deterministic baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from altermagnetism_LLG.model.baseline import DeterministicPathModel  # noqa: E402
from altermagnetism_LLG.model.network import PeriodicEquivariantFlowNet  # noqa: E402
from altermagnetism_LLG.model.sphere import sample_reference_path, sphere_exp  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--condition-source", type=Path, required=True)
    p.add_argument("--system", default="d_wave_altermagnet")
    p.add_argument("--trajectory", type=int, default=0)
    p.add_argument("--size", type=int, nargs=2, default=None)
    p.add_argument("--integration-steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "flow_model" / "sample.npz",
    )
    return p


def tile_initial(initial: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    nx, ny = initial.shape[-3:-1]
    repeat_x = (size[0] + nx - 1) // nx
    repeat_y = (size[1] + ny - 1) // ny
    return initial.repeat(1, 1, repeat_x, repeat_y, 1)[
        ..., : size[0], : size[1], :
    ]


def build_model(checkpoint: dict):
    model_type = checkpoint["model_type"]
    if model_type == "flow":
        model = PeriodicEquivariantFlowNet(**checkpoint["model_config"])
    elif model_type == "deterministic":
        model = DeterministicPathModel(**checkpoint["model_config"])
    else:
        raise ValueError(f"unsupported checkpoint model_type: {model_type}")
    return model_type, model


@torch.no_grad()
def main() -> None:
    args = parser().parse_args()
    if args.integration_steps < 1:
        raise ValueError("integration-steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; submit through a GPU allocation")
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=True
    )
    model_type, model = build_model(checkpoint)
    model = model.to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with h5py.File(args.condition_source, "r") as h5:
        group = h5[args.system]
        initial = torch.from_numpy(
            group["initial_spins"][args.trajectory]
        ).unsqueeze(0).to(args.device)
        scalar = torch.from_numpy(
            group["model_condition"][args.trajectory]
        ).unsqueeze(0).to(args.device)
        vector = torch.from_numpy(
            group["vector_condition"][args.trajectory]
        ).unsqueeze(0).to(args.device)
        frames = group["spins"].shape[1]
        time = h5["time"][:].astype(np.float32)
    if args.size is not None:
        initial = tile_initial(initial, tuple(args.size))
    physical_time = torch.from_numpy(time).unsqueeze(0).to(args.device)

    if model_type == "flow":
        generator_device = args.device if args.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(args.seed)
        state = sample_reference_path(initial, frames, generator=generator)
        step = 1.0 / args.integration_steps
        for index in range(args.integration_steps):
            tau = torch.full(
                (1,), index * step, device=args.device, dtype=state.dtype
            )
            velocity = model(
                state, tau, initial, scalar, vector, physical_time
            )
            state = sphere_exp(state, step * velocity)
            state[:, 0] = initial
        prediction = state
    else:
        prediction = model(
            initial, frames, scalar, vector, physical_time
        )

    sample = prediction[0].cpu().numpy().astype(np.float32)
    norm_error = float(np.abs(np.linalg.norm(sample, axis=-1) - 1.0).max())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        spins=sample,
        time=time,
        scalar_condition=scalar.cpu().numpy(),
        vector_condition=vector.cpu().numpy(),
    )
    report = {
        "status": "generated_sample_awaiting_distributional_evaluation",
        "model_type": model_type,
        "shape": list(sample.shape),
        "max_spin_norm_error": norm_error,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "condition_source": str(args.condition_source.resolve()),
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
