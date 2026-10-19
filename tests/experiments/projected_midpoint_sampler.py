"""Explicit midpoint with sphere projection; flow time, NOT a stochastic LLG solver."""
import torch
from tests.experiments.local_equivariant_transition import source


def midpoint_step(field, x, time, dt):
    middle = torch.nn.functional.normalize(x+(dt/2)*field(x, time), dim=-1)
    return torch.nn.functional.normalize(x+dt*field(middle, time+dt/2), dim=-1)


@torch.no_grad()
def sample(model, initial, theta, lag, axis, generator, scale, steps):
    if steps < 1:
        raise ValueError('positive step count required')
    x = source(initial, generator, scale)
    def field(state, time):
        tau = initial.new_full((len(initial),), time)
        return model(state, tau, initial, theta, lag, axis)
    for step in range(steps):
        x = midpoint_step(field, x, step/steps, 1/steps)
    return x
