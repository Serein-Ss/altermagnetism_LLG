"""Test-only two-stage geometry; no analytic field enters either neural network."""
import torch
from torch import nn
from torch.nn import functional as F
from scripts.model.sphere import sphere_exp,tangent_project
from tests.bauer_closed_loop_20260919.model import Block,hierarchy,neighbor,Net


class InitialNet(nn.Module):
    def __init__(self,width=64,depth=3):
        super().__init__();self.input=nn.Linear(26,width)
        self.blocks=nn.ModuleList([Block(width,False,True) for _ in range(depth)])
        self.output=nn.Linear(width,10)

    def forward(self,x,tau,axis=None):
        axis=x.new_tensor([0.,0.,1.]) if axis is None else axis
        x=x[:,None];a=axis.expand_as(x)
        vectors=torch.stack((x,neighbor(x),(x*a).sum(-1,keepdim=True)*a,hierarchy(x),a),-2)
        gram=(vectors@vectors.transpose(-1,-2)).flatten(-2)
        h=F.silu(self.input(torch.cat((gram,tau[:,None,None,None].expand(*x.shape[:-1],1)),-1)))
        cond=torch.stack((tau,torch.zeros_like(tau)),-1)
        for block in self.blocks:h=block(h,cond)
        direct,cross=self.output(h).chunk(2,-1)
        v=(direct[...,None]*vectors+cross[...,None]*torch.linalg.cross(x[...,None,:],vectors)).sum(-2)
        return tangent_project(x,v)[:,0]


def base_path(initial,frames,rng,variant,sigma=.7,rho=.8):
    shape=(len(initial),frames,initial.shape[1],3)
    noise=torch.randn(shape,device=initial.device,dtype=initial.dtype,generator=rng)
    if variant=='uniform':x=F.normalize(noise,dim=-1)
    elif variant=='anchored':
        for j in range(2,frames):noise[:,j]=rho*noise[:,j-1]+(1-rho*rho)**.5*noise[:,j]
        anchor=initial[:,None].expand(shape)
        x=sphere_exp(anchor,sigma*tangent_project(anchor,noise))
    else:raise ValueError(variant)
    x[:,0]=initial
    return x


@torch.no_grad()
def sample_initial(net,length,count,seed,steps,device='cpu'):
    rng=torch.Generator(device=device).manual_seed(seed)
    x=F.normalize(torch.randn(count,length,3,generator=rng,device=device),dim=-1)
    for k in range(steps):x=sphere_exp(x,net(x,x.new_full((count,),k/steps))/steps)
    return x


@torch.no_grad()
def sample_path(net,initial,times,seed,steps,variant,c):
    rng=torch.Generator(device=initial.device).manual_seed(seed)
    x=base_path(initial,len(times),rng,variant,c['anchored_sigma'],c['anchored_time_rho'])
    for k in range(steps):
        x=sphere_exp(x,net(x,x.new_full((len(x),),k/steps),initial,times,63.82484966960072)/steps)
        x[:,0]=initial
    return x
