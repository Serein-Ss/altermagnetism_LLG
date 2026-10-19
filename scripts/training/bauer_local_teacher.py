"""TRAIN empirical patch-marginal teacher, not an exact whole-chain flow law."""
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from scripts.model.empirical_path_flow import conditional


@torch.no_grad()
def patch_conditional(x,tau,bank,initial,radius):
    if not isinstance(radius,int) or radius<0:raise ValueError('nonnegative integer patch radius')
    n=x.shape[2];pieces=[];effective=[]
    for site in range(n):
        start=max(0,site-radius);stop=min(n,site+radius+1)
        v,w,bad=conditional(x[:,:,start:stop],tau,bank[:,:,start:stop],initial[:,start:stop])
        if bad.any() or not torch.isfinite(v).all():raise RuntimeError('unsupported empirical patch, no nearest fallback')
        pieces.append(v[:,:,site-start:site-start+1]);effective.append(1/w.square().sum(-1))
    return torch.cat(pieces,2),torch.stack(effective,1)


@torch.no_grad()
def labels(z,y,tau,bank,generator,local=False):
    if len(bank)!=64 or y.shape[1]!=2:raise ValueError('64 TRAIN futures and two frames required')
    if not torch.allclose(bank[:,0],bank[:1,0].expand_as(bank[:,0]),atol=1e-6,rtol=0):raise ValueError('mixed bank anchors')
    if not torch.allclose(y[:,0],bank[:1,0].expand_as(y[:,0]),atol=1e-6,rtol=0):raise ValueError('wrong bank')
    if not torch.equal(z[:,0],y[:,0]):raise ValueError('wrong latent anchor')
    y=F.normalize(y.double(),dim=-1);z=F.normalize(z.double(),dim=-1);bank=F.normalize(bank.double(),dim=-1);z[:,0]=y[:,0]
    x,v,cut=path_interpolate(z,y,tau.double(),generator)
    if cut['unresolved_cut_count']:raise RuntimeError('unresolved antipode')
    full=v.clone();patch=v.clone();mask=tau<=.9;stats=dict(teacher_count=int(mask.sum()),whole_effective_modes=0.,patch_effective_modes=0.,target_difference=0.)
    if mask.any():
        fv,w,bad=conditional(x[mask],tau[mask].double(),bank,y[mask,0])
        if bad.any() or not torch.isfinite(fv).all():raise RuntimeError('unsupported whole teacher')
        pv,ess=patch_conditional(x[mask],tau[mask].double(),bank,y[mask,0],2)
        full[mask]=fv;patch[mask]=pv
        stats.update(whole_effective_modes=float((1/w.square().sum(-1)).mean()),patch_effective_modes=float(ess.mean()),
            target_difference=float((pv[:,1:]-fv[:,1:]).square().sum(-1).mean()))
    # Identical drafts/RNG and ordinary CFM at tau>.9 for both arms.
    return x.float(),v.float(),(patch if local else full).float(),stats
