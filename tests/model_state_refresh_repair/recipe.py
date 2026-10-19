"""Balanced group-level replacement; preserve the preceding training recipe."""
import numpy as np
from tests.model_state_replay_repair.recipe import rate

def replay_roles(step,arm):
    if step<1 or arm not in ('stale_replay','refreshed_replay'):raise ValueError('invalid step/arm')
    roles=(step+np.arange(3))%2
    if arm=='refreshed_replay':roles=roles*2
    return np.repeat(roles,16)
