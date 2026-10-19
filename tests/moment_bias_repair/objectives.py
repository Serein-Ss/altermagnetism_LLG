"""Separate moment objectives and cross-replicate squared-distance estimator."""
import torch
from tests.flow_training_repair.objectives import statistics


def cross_square(a,b,target):
    # Independent unbiased a,b => E[(a-t)(b-t)] = (E[a]-t)^2.
    # Can be negative for one batch; clamping would reintroduce bias.
    return ((a-target)*(b-target)).mean()


def loss_parts(generated,reference,mode):
    if mode not in ('variance','correlation','joint','cross'):raise ValueError(mode)
    if len(generated)<4 or len(generated)%2:raise ValueError('even independent generated branches >=4 required')
    if not torch.allclose(generated[:,0],reference[:1,0].expand_as(generated[:,0]),atol=1e-6,rtol=0):raise ValueError('different parents')
    if not torch.allclose(reference[:,0],reference[:1,0].expand_as(reference[:,0]),atol=1e-6,rtol=0):raise ValueError('mixed reference parents')
    vr,cr=statistics(reference.detach());sv=vr.mean().clamp_min(.05)
    mask=torch.triu(torch.ones_like(cr,dtype=torch.bool),diagonal=1)
    sc=(1-cr[mask]).mean().clamp_min(.05)
    if mode=='cross':
        va,ca=statistics(generated[:len(generated)//2]);vb,cb=statistics(generated[len(generated)//2:])
        lv=cross_square(va/sv,vb/sv,vr/sv)
        lc=cross_square(ca[mask]/sc,cb[mask]/sc,cr[mask]/sc)
    else:
        vg,cg=statistics(generated)
        lv=((vg-vr)/sv).square().mean();lc=((cg[mask]-cr[mask])/sc).square().mean()
    loss=lv if mode=='variance' else lc if mode=='correlation' else lv+lc
    return loss,dict(variance_loss=float(lv.detach()),correlation_loss=float(lc.detach()))
