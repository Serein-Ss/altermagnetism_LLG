"""Native0.1ps transport and composition; steps are NOT physical LLG dt."""
import os
import math
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from scripts.model.sphere import sphere_exp

def correlate(noise,rho):
    if not math.isfinite(rho) or not 0<=rho<1:raise ValueError('rho must be in [0,1)')
    if rho==0:return noise
    result=noise.clone()
    # Frame zero will be clamped; frame one starts a stationary Gaussian chain.
    for t in range(2,noise.shape[1]):
        result[:,t]=rho*result[:,t-1]+math.sqrt(1-rho*rho)*noise[:,t]
    return result


def from_noise(initial,noise,rho):
    if noise.shape[0]!=len(initial) or noise.shape[2:]!=initial.shape[1:]:raise ValueError('shape mismatch')
    z=F.normalize(correlate(noise,rho)+2*initial[:,None],dim=-1)
    z[:,0]=initial
    return z


def draw(initial,frames,generator,rho):
    noise=torch.randn((len(initial),frames,*initial.shape[1:]),device=initial.device,dtype=initial.dtype,generator=generator)
    return from_noise(initial,noise,rho)


@torch.no_grad()
def integrate(field,x,start,stop,steps,method):
    if not 0<=start<stop<=1 or steps<1:raise ValueError('transport range')
    if method not in ('euler','midpoint'):raise ValueError(method)
    x=x.clone();initial=x[:,0].clone();dt=(stop-start)/steps
    for k in range(steps):
        tau=x.new_full((len(x),),start+k*dt);v=field(x,tau)
        if method=='euler':x=sphere_exp(x,dt*v)
        else:
            mid=F.normalize(x+.5*dt*v,dim=-1);mid[:,0]=initial
            x=F.normalize(x+dt*field(mid,tau+.5*dt),dim=-1)
        x[:,0]=initial
    return x


@torch.no_grad()
def rollout(net,initial,generator,steps,transitions=10):
    if not isinstance(transitions,int) or transitions<1:raise ValueError('positive transitions required')
    frames=[initial];state=initial;times=initial.new_tensor([0.,.1])
    for _ in range(transitions):
        z=draw(state,2,generator,0.)
        out=integrate(lambda x,t:net.velocity(x,t,state,times),z,0.,1.,steps,'midpoint')
        assert torch.equal(out[:,0],state)
        state=out[:,1];frames.append(state)
    return torch.stack(frames,1)

def differentiable_sample(net, initial, z, steps=128, use_checkpoint=True):
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


def require_device(device):
    if str(device).startswith('cuda') and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('CUDA requires a Slurm GPU allocation in partition rtx4090')
    return torch.device(device)
