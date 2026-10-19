"""Conditional local/spatial moments. Finite TRAIN targets are not population truth."""
import torch
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


LAGS=(1,2,4)


def cross_covariance(paths,lag):
    d=paths[:,1]-paths[:,0];d=d-d.mean(0)
    return torch.einsum('bni,bnj->nij',d[:,:-lag],d[:,lag:])/(len(d)-1)


def spatial_moment_loss(generated,reference,spatial):
    local,parts=moment_loss(generated,reference)
    if not spatial:return local,dict(**parts,spatial_loss=0.)
    if generated.shape[1:]!=reference.shape[1:] or generated.shape[2]<=max(LAGS):raise ValueError('matching shapes and at least five sites required')
    a,b=generated.chunk(2);reference=reference.detach();d=reference[:,1]-reference[:,0]
    scale=(d-d.mean(0)).square().sum(-1).sum(0).mean()/(len(d)-1);scale=scale.clamp_min(.05)
    values=[]
    for lag in LAGS:
        target=cross_covariance(reference,lag)
        values.append(((cross_covariance(a,lag)-target)*(cross_covariance(b,lag)-target)).sum((-1,-2)).mean()/scale.square())
    extra=torch.stack(values).mean()
    # Independent generated folds remove the noisy-square variance-shrink term.
    # Finite TRAIN targets still carry estimation error; do not clamp negatives.
    return local+extra,dict(**parts,spatial_loss=float(extra.detach()))
