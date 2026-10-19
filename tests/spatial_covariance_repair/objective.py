"""Conditional distance-resolved covariance, not energy conservation."""
import torch
from tests.conditional_moment_repair.objectives import sample,moment_loss

LAGS=(1,2,4)


def cross_covariance(paths,lag):
    d=paths[:,1]-paths[:,0];d=d-d.mean(0)
    return torch.einsum('bni,bnj->nij',d[:,:-lag],d[:,lag:])/(len(d)-1)


def loss(generated,reference,spatial):
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
