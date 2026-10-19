"""Strictly positive weights on auxiliary flow time, never physical time."""
import torch

def weight(tau,arm):
    if arm not in ('control','early_tau'):raise ValueError(arm)
    if not torch.isfinite(tau).all() or torch.any((tau<0)|(tau>1)):
        raise ValueError('flow time must be in [0,1]')
    if arm=='control':return torch.ones_like(tau)
    # Integral on U[0,1] equals one. No batch-dependent normalization.
    return (.25+4*(1-tau).square())/(.25+4/3)

def loss(pred,target,tau,arm):
    if pred.shape!=target.shape or len(tau)!=len(pred):raise ValueError('batch mismatch')
    per_path=(pred[:,1:]-target[:,1:]).square().sum(-1).flatten(1).mean(1)
    return (per_path*weight(tau,arm)).mean()
