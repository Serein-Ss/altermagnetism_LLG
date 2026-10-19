"""Empirical whole-path oracle matching normalize(N(2*S0,I)) sources.

For u on S2 and b=mu.u, the projected-normal density is
exp(-|mu|^2/2)/(2*pi)^(3/2) times
I2(b)=(1+b^2)*sqrt(2*pi)*exp(b^2/2)*Phi(b)+b.
Under contraction toward target y by a=1-tau, the density Jacobian
is sinc(r/a)/sinc(r)/a^2, with r=angle(x,y)<a*pi.
The discrete targets are eight EMPIRICAL paths, not the true LLG law.
"""
import math
import torch
from torch.nn import functional as F


def log_source(u,initial,shift=2.):
    if not 0<=shift<=2:raise ValueError('validated shift range0..2')
    b=shift*(u*initial).sum(-1)
    integral=(1+b.square())*math.sqrt(2*math.pi)*torch.exp(b.square()/2)*torch.special.ndtr(b)+b
    return integral.log()-shift**2/2-1.5*math.log(2*math.pi)


def conditional(x,tau,targets,initial,shift=2.):
    if torch.any(tau<0) or torch.any(tau>=1):raise ValueError('oracle requires0<=tau<1')
    y=targets[None].expand(len(x),-1,-1,-1,-1)
    xx=x[:,None].expand_as(y)
    dot=(xx*y).sum(-1).clamp(-1,1)
    tangent=xx-dot[...,None]*y
    sine=tangent.norm(dim=-1);r=torch.atan2(sine,dot)
    a=(1-tau)[:,None,None,None];q=r/a
    z=q.cos()[...,None]*y+q.sin()[...,None]*tangent/sine.clamp_min(1e-15)[...,None]
    logp=log_source(z,initial[:,None,None],shift)
    logp+=torch.sinc(q/math.pi).clamp_min(1e-30).log()-torch.sinc(r/math.pi).clamp_min(1e-30).log()-2*a.log()
    valid=(q[:,:,1:]<math.pi).flatten(2).all(-1)
    logp=logp[:,:,1:].flatten(2).sum(-1).masked_fill(~valid,-torch.inf)
    bad=~valid.any(-1)
    weights=torch.softmax(torch.where(bad[:,None],torch.zeros_like(logp),logp),-1)
    nearest=r[:,:,1:].square().flatten(2).sum(-1).argmin(-1)
    weights=torch.where(bad[:,None],F.one_hot(nearest,len(targets)).to(x),weights)
    direction=y-dot[...,None]*xx
    norm=direction.norm(dim=-1)
    velocity=(r/a)[...,None]*direction/norm.clamp_min(1e-15)[...,None]
    v=(weights[:,:,None,None,None]*velocity).sum(1)
    v=torch.cat((torch.zeros_like(v[:,:1]),v[:,1:]),1)
    return v,weights,bad


def branch_summary(x,targets):
    distance=(x[:,None,1:]-targets[None,:,1:]).square().mean((2,3,4))
    nearest=distance.argmin(1);counts=torch.bincount(nearest,minlength=len(targets))
    return dict(nearest_counts=counts.cpu().tolist(),nearest_fractions=(counts/len(x)).cpu().tolist(),
                nearest_path_mse=float(distance.min(1).values.mean()),branches=len(x),
                interpretation='Nearest empirical path labels, NOT physical event classes or natural LLG probabilities')
