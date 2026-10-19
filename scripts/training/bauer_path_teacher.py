"""Empirical conditional-velocity labels; never used at deployment."""
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from scripts.model.empirical_path_flow import conditional


@torch.no_grad()
def labels(z,y,tau,bank,generator):
    if len(bank)!=64 or y.shape[1]!=2:
        raise ValueError('64 TRAIN futures and two-frame targets required')
    if not torch.allclose(bank[:,0],bank[:1,0].expand_as(bank[:,0]),atol=1e-6,rtol=0):
        raise ValueError('bank must share one condition')
    if not torch.allclose(y[:,0],bank[:1,0].expand_as(y[:,0]),atol=1e-6,rtol=0):
        raise ValueError('wrong conditioning bank')
    if not torch.equal(z[:,0],y[:,0]):raise ValueError('wrong latent anchor')
    # Float64 avoids the inverse spherical contraction amplifying float32
    # interpolation errors. Both experimental arms use these same drafts.
    y=F.normalize(y.double(),dim=-1);z=F.normalize(z.double(),dim=-1)
    bank=F.normalize(bank.double(),dim=-1);z[:,0]=y[:,0]
    x,v,cut=path_interpolate(z,y,tau.double(),generator)
    if cut['unresolved_cut_count']:raise RuntimeError('unresolved antipode')
    rb=v.clone();mask=tau<=.9
    effective=0.
    if mask.any():
        mean,w,bad=conditional(x[mask],tau[mask].double(),bank,y[mask,0])
        if bad.any() or not torch.isfinite(mean).all():
            raise RuntimeError('invalid empirical teacher support; no silent nearest fallback')
        rb[mask]=mean
        effective=float((1/w.square().sum(-1)).mean())
    # At late transport times retain ordinary unbiased sampled labels. This is
    # not truncation of training/sampling to tau=.9 and not a physical time cut.
    return x.float(),v.float(),rb.float(),dict(teacher_count=int(mask.sum()),
        effective_modes=effective,target_residual_mse=float((v[:,1:]-rb[:,1:]).square().sum(-1).mean()))
