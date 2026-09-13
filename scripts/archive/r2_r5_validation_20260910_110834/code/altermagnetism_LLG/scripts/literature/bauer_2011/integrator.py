"""Milstein--Tretyakov SINUM34 (1997), equations 5.9, 5.6, 4.5/4.6.

Specialized to Bauer's time-independent rotational Stratonovich noise.
epsilon²=2*lambda*theta, sigma_r(s)=-s cross e_r. The Ito correction b=-s;
L2 sigma_r=-sigma_r, and sum Lambda_i sigma_r Xi_ir=Xi@s-tr(Xi)*s.
The unprojected formula has the cited O(h4+epsilon²*h2) weak error.
Final normalization is an explicit selectable project extension, not part
of that theorem. The discrete weak draws are NOT Wiener increments.
"""
import math
import torch
from scripts.core.reduced_llg import ReducedLLG


class BauerWeakRK(ReducedLLG):
    def __init__(self,model,*,alpha,theta,project=True):
        super().__init__(model,alpha=alpha,theta=theta,equation_convention='bauer_ll')
        self.project=bool(project)

    def noise_increment(self,s,dt,generator):
        if not math.isfinite(dt) or dt<=0: raise ValueError('positive dt required')
        uniform=torch.rand(s.shape,dtype=s.dtype,device=s.device,generator=generator)
        xi=torch.where(uniform<1/6,-math.sqrt(3.),torch.where(uniform>5/6,math.sqrt(3.),0.))
        zeta=2*torch.randint(0,2,s.shape,device=s.device,generator=generator).to(s.dtype)-1
        return torch.stack((xi,zeta),-1)

    def step(self,s,dt,draws,*,method='milstein_tretyakov_5_9',**kwargs):
        if method!='milstein_tretyakov_5_9' or not math.isfinite(dt) or dt<=0:
            raise ValueError('this solver implements only MT equation 5.9')
        if self.theta:
            if draws.shape!=(*s.shape,2): raise ValueError('weak xi/zeta draws required, not Wiener increments')
            xi,zeta=draws.unbind(-1)
        else:
            xi=zeta=torch.zeros_like(s)
        eps=math.sqrt(2*self.alpha*self.theta)
        cross=torch.linalg.cross
        def drift(x):
            b=self.model.field(x)
            return -cross(x,b)-self.alpha*cross(x,cross(x,b))
        a=drift(s); correction=-s
        sigma=-cross(s,xi); noise=eps*math.sqrt(dt)*sigma
        k1=dt*a; k2=dt*drift(s+k1/2)
        k3=dt*drift(s+noise+k2/2)
        k4=dt*drift(s+noise+k3+3*eps**2*dt*correction)
        gamma=s.new_tensor(((1.,-1.,-1.),(1.,1.,-1.),(1.,1.,1.)))
        tensor=(xi[..., :,None]*xi[...,None,:]-gamma*zeta[..., :,None]*zeta[...,None,:])/2
        derivative_noise=(tensor@s[...,None]).squeeze(-1)-tensor.diagonal(dim1=-2,dim2=-1).sum(-1,keepdim=True)*s
        end=s+noise+dt*(a+eps**2*correction)
        raw=(s+noise+eps**2*dt*derivative_noise
             -eps*dt**1.5*cross(a,xi)/2-eps**3*dt**1.5*sigma/2
             +(k1+2*k2+2*k3+k4)/6+eps**2*dt*(correction-end)/2)
        error=(raw.norm(dim=-1)-1).abs().max()
        result=raw/raw.norm(dim=-1,keepdim=True) if self.project else raw
        return result,torch.stack((error,(result.norm(dim=-1)-1).abs().max()))
