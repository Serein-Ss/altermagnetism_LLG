"""Conditional local moments, with independent-ensemble cross-product loss."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def moments(x):
    if x.ndim != 4 or x.shape[1] != 2 or len(x) < 2:
        raise ValueError('two frames and independent branches required')
    d = x[:, 1] - x[:, 0]
    mean = d.mean(0)
    centered = d - mean
    cov = torch.einsum('bni,bnj->nij', centered, centered) / (len(x)-1)
    return mean, cov


def moment_loss(generated, reference):
    if len(generated) < 4 or len(generated) % 2:
        raise ValueError('two equal independent ensembles required')
    for x in (generated, reference):
        if not torch.allclose(x[:, 0], reference[:1, 0].expand_as(x[:, 0]), atol=1e-6, rtol=0):
            raise ValueError('all branches must share the same initial state')
    mr, cr = moments(reference.detach())
    ma, ca = moments(generated[:len(generated)//2])
    mb, cb = moments(generated[len(generated)//2:])
    scale = cr.diagonal(dim1=-2, dim2=-1).sum(-1).mean().clamp_min(.05)
    # Unbiased w.r.t. generated-ensemble noise, conditional on the finite TRAIN
    # target. Unlike squared noisy sample means, no extra variance-shrink term.
    lm = ((ma-mr)*(mb-mr)).sum(-1).mean()/scale
    lc = ((ca-cr)*(cb-cr)).sum((-1,-2)).mean()/scale.square()
    return lm+lc, dict(mean_loss=float(lm.detach()), covariance_loss=float(lc.detach()))


def sample(net, initial, z, steps=128, use_checkpoint=True):
    """Same normalized midpoint map as deployment, differentiable and anchored."""
    times = z.new_tensor([0., .1])
    def clamp(x):
        return torch.cat((initial[:, None], F.normalize(x[:, 1:], dim=-1)), 1)
    x = z
    for start in range(0, steps, 8):
        def segment(x, start=start):
            for k in range(start, min(start+8, steps)):
                v = net.velocity(x, x.new_full((len(x),), k/steps), initial, times)
                mid = clamp(x + .5/steps*v)
                v = net.velocity(mid, x.new_full((len(x),), (k+.5)/steps), initial, times)
                x = clamp(x + v/steps)
            return x
        x = checkpoint(segment, x, use_reentrant=False) if use_checkpoint else segment(x)
    return x
