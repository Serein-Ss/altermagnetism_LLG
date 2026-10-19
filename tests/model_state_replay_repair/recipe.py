"""Only replay-state distribution differs between the paired training arms."""
import numpy as np
from tests.training_restart_from_scratch.recipe import rate as full_rate

def replay_roles(step,arm):
    if step<1 or arm not in ('real_control','model_replay'):raise ValueError('invalid step/arm')
    roles=np.zeros(3,dtype=np.int64) if arm=='real_control' else (step+np.arange(3))%2
    return np.repeat(roles,16)

def rate(step,total):
    # Paired conservative restart: 200 warmup updates, peak3e-4, end9e-6.
    return .3*full_rate(step,total)
