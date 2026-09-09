#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=altermagnetism_LLG/logs/slurm/%x-%j-smoke-v2-3ps-memory.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME

bash run_zrs_mag.sh - <<'PY'
import gc
import torch

from altermagnetism_LLG.scripts.model.baseline import DeterministicPathModel
from altermagnetism_LLG.scripts.model.network import PeriodicEquivariantFlowNet
from altermagnetism_LLG.scripts.model.sphere import geodesic_interpolate, sample_reference_path

device = "cuda"
assert torch.cuda.is_available()
shape = (1, 301, 2, 48, 48, 3)
target = torch.randn(shape, device=device)
target = torch.nn.functional.normalize(target, dim=-1)
initial = target[:, 0].clone()
scalar = torch.zeros((1, 10), device=device)
scalar[:, 6] = 0.5 / 3.0
vector = torch.randn((1, 5, 3), device=device)
vector = torch.nn.functional.normalize(vector, dim=-1)
time = torch.linspace(0.0, 3.0e-12, 301, device=device)[None]

for kind, cls in (("flow", PeriodicEquivariantFlowNet), ("deterministic", DeterministicPathModel)):
    model = cls(hidden=32, blocks=4, architecture_version=2).to(device)
    if kind == "flow":
        tau = torch.rand(1, device=device)
        reference = sample_reference_path(initial, 301)
        state, velocity = geodesic_interpolate(reference, target, tau)
        prediction = model(state, tau, initial, scalar, vector, time)
        loss = (prediction - velocity).square().mean()
    else:
        prediction = model(initial, 301, scalar, vector, time)
        loss = (prediction - target).square().mean()
    loss.backward()
    print(kind, float(loss), torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
    del model, prediction, loss
    if kind == "flow":
        del reference, state, velocity
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
PY
