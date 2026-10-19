"""Training-only chain-index reversal for homogeneous symmetric open chains."""
import torch
from tests.experiments.director_budget import train_step


def reflect_pairs(initial, target, mask):
    """Reverse sites together, preserving time direction and spin components."""
    select = mask[:, None, None]
    return (torch.where(select, initial.flip(1), initial),
            torch.where(select, target.flip(1), target))


def augment_pairs(initial, target, generator):
    """Use a dedicated augmentation RNG, separate from flow and sample RNGs."""
    mask = torch.rand(len(initial), device=initial.device, generator=generator) < .5
    a, b = reflect_pairs(initial, target, mask)
    return a, b, mask


def augmented_train_step(model, optimizer, initial, target, theta, lag, axis,
                         generator, scale, augmentation_generator):
    a, b, mask = augment_pairs(initial, target, augmentation_generator)
    value, grad, diagnostics = train_step(
        model, optimizer, a, b, theta, lag, axis, generator, scale)
    return value, grad, diagnostics, int(mask.sum().item())
