"""Time-allocation control and training-only conditional ensemble objectives."""
import torch
from scripts.model.sphere import sphere_exp


def draw_tau(count,stop,mode,generator,device):
    if mode not in ('baseline','importance','early','joint','early_joint'):raise ValueError(mode)
    # Identical number of RNG draws in every arm; latent/data RNGs separate.
    u=torch.rand(count,device=device,generator=generator)
    choose=torch.rand(count,device=device,generator=generator)
    if mode in ('baseline','joint'):return u*stop,torch.ones_like(u)
    tau=torch.where(choose<.5,u*.1,u*stop)
    relative_density=.5+.5*stop/.1*(tau<.1).to(u)
    weight=1/relative_density if mode=='importance' else torch.ones_like(u)
    return tau,weight


def differentiable_sample(net,initial,times,z,steps=32):
    x=z
    for k in range(steps):
        v=net.velocity(x,x.new_full((len(x),),k/steps),initial,times)
        x=sphere_exp(x,v/steps)
        x=torch.cat((initial[:,None],x[:,1:]),1)
    return x


def statistics(s):
    if len(s)<2:raise ValueError('independent branches required')
    d=s[:,1:]-s[:,:-1]
    variance=d.var(0,unbiased=True).sum(-1).mean(-1)
    correlation=torch.einsum('btnc,bunc->tu',s,s)/(len(s)*s.shape[2])
    return variance,correlation


def joint_loss(generated,reference):
    # Targets are whole-branch ensemble statistics; no path-to-path matching.
    if not torch.allclose(generated[:,0],reference[:1,0].expand_as(generated[:,0]),atol=1e-6,rtol=0):
        raise ValueError('generated paths must share the reference parent')
    if not torch.allclose(reference[:,0],reference[:1,0].expand_as(reference[:,0]),atol=1e-6,rtol=0):
        raise ValueError('reference branches have different parents')
    vg,cg=statistics(generated);vr,cr=statistics(reference.detach())
    scale_v=vr.mean().clamp_min(.05)
    mask=torch.triu(torch.ones_like(cr,dtype=torch.bool),diagonal=1)
    scale_c=(1-cr[mask]).mean().clamp_min(.05)
    lv=((vg-vr)/scale_v).square().mean()
    lc=((cg[mask]-cr[mask])/scale_c).square().mean()
    return lv+lc,dict(variance_loss=float(lv.detach()),correlation_loss=float(lc.detach()))
