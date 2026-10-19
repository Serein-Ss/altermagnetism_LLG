"""Experimental differentiable sampler and independent-replica mean objective."""
import torch
from scripts.model.sphere import sphere_exp
from tests.bauer_two_stage_20260919.models import base_path


def differentiable_sample(net, initial, times, seed, steps, temperature):
    if steps < 1 or len(times) != 2:
        raise ValueError('Positive integration steps and a single transition required')
    rng = torch.Generator(device=initial.device).manual_seed(seed)
    x = base_path(initial, 2, rng, 'anchored', .7, .8)
    # No in-place updates of states needed by autograd.
    x = torch.cat((initial[:, None], x[:, 1:]), 1)
    for k in range(steps):
        velocity = net(x, x.new_full((len(x),), k/steps), initial, times, temperature)
        future = sphere_exp(x, velocity/steps)[:, 1:]
        x = torch.cat((initial[:, None], future), 1)
    return x


def replica_mean_loss(generated_a, generated_b, reference_a, reference_b):
    """Unbiased mean-square error when all four branch groups are independent.

    Arrays are [parent, branch, site, component]. Each parent shares one
    initial state. Do not clamp negative estimates: that changes the objective.
    Pairing distinct initial states is invalid. Caller enforces data provenance.
    This constrains only the mean; retain flow matching to learn the distribution.
    """
    arrays = [generated_a, generated_b, reference_a, reference_b]
    if any(x.ndim != 4 or x.shape[1] < 1 for x in arrays):
        raise ValueError('Expected parent,branch,site,component tensors')
    expected = generated_a.shape[0], generated_a.shape[2:]
    if any((x.shape[0], x.shape[2:]) != expected for x in arrays):
        raise ValueError('Parent/site/component shapes must agree')
    residual_a = generated_a.mean(1)-reference_a.detach().mean(1)
    residual_b = generated_b.mean(1)-reference_b.detach().mean(1)
    return (residual_a*residual_b).sum(-1).mean()
