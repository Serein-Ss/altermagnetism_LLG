"""Active tangent residual and controlled shared-source experiments, not production."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from scripts.model.sphere import tangent_project
from tests.bauer_p2_20260923.common import Model

ARMS = ['baseline', 'local_fixed', 'temporal_fixed', 'relational_global',
        'temporal_shared', 'relational_shared']


def source_rho(arm):
    if arm not in ARMS:
        raise ValueError('unknown repair arm')
    return .7 if arm.endswith('_shared') else 0.


def correlated_noise(independent, shared_vector, shared_scalar, initial, rho):
    """Each site's conditional covariance is I, even with a shared tangent draw.

    For unit s, Cov(g cross s + h*s) = (I-ss^T)+ss^T = I.
    Independent eps with nonzero weight preserves full joint Gaussian support.
    g,h are shared across sites/time WITHIN a branch, never across branches.
    """
    if not 0 <= rho < 1:
        raise ValueError('rho must preserve a nonzero independent component')
    anchor = initial[:, None]
    coherent = torch.linalg.cross(shared_vector.expand_as(anchor), anchor) + shared_scalar*anchor
    return math.sqrt(1-rho*rho)*independent + rho*coherent


def source_from_draws(independent, shared_vector, shared_scalar, initial, arm):
    noise = correlated_noise(independent, shared_vector, shared_scalar, initial, source_rho(arm))
    z = F.normalize(noise+2.*initial[:, None],dim=-1)
    z[:, 0] = initial
    return z


def draw_source(shape, initial, generator, arm):
    kwargs=dict(device=initial.device,dtype=initial.dtype,generator=generator)
    eps=torch.randn(shape,**kwargs)
    g=torch.randn((len(initial),1,1,3),**kwargs)
    h=torch.randn((len(initial),1,1,1),**kwargs)
    # Draw identical tensors in every arm, even when rho=0.
    source=source_from_draws(eps,g,h,initial,arm)
    fingerprint=torch.cat((eps.flatten(1),g.flatten(1),h.flatten(1)),1)
    return source,fingerprint


def relational_pool(x, anchor):
    """Equal-weight actual-site means; no padding, no unit normalization."""
    return torch.linalg.cross(anchor,x).mean(2,keepdim=True)


class RepairModel(Model):
    def __init__(self, config, arm):
        super().__init__(config)
        source_rho(arm)
        self.arm=arm
        self.kind={'baseline':'none','local_fixed':'local','temporal_fixed':'temporal',
                   'temporal_shared':'temporal','relational_global':'relational',
                   'relational_shared':'relational'}[arm]
        if self.kind!='none':
            width=config['width']
            self.residual_input=nn.Sequential(nn.Linear(9,width),nn.SiLU(),nn.LayerNorm(width))
            self.residual_qk=nn.Linear(width,2*width)
            self.residual_gate=nn.Linear(width,8)
            nn.init.zeros_(self.residual_gate.weight)
            nn.init.zeros_(self.residual_gate.bias)

    def get_extra_state(self):
        return dict(periodic=self.periodic,symmetry='SO(3)',version=3,arm=self.arm,
                    domain=self.config.get('domain','software_test'),time_unit=self.config.get('time_unit','ps'),
                    source='normalized_shifted_correlated_gaussian',source_shift=2.,rho=source_rho(self.arm))

    def residual(self,x,tau,initial,times,frame=None):
        if self.kind=='none':
            return torch.zeros_like(x)
        frame=torch.eye(3,device=x.device,dtype=x.dtype) if frame is None else frame
        anchor=initial[:,None].expand_as(x)
        meta=torch.stack((tau[:,None,None].expand(x.shape[:-1]),times[None,:,None].expand(x.shape[:-1])),-1)
        features=torch.cat(((x*anchor).sum(-1,keepdim=True),x@frame.T,anchor@frame.T,meta),-1)
        h=self.residual_input(features).transpose(1,2)
        if self.kind=='relational':
            h=h+h.mean(1,keepdim=True)
        q,k=self.residual_qk(h).chunk(2,-1)
        b,n,t,c=q.shape
        q=q.reshape(b,n,t,4,c//4).transpose(2,3)
        k=k.reshape(b,n,t,4,c//4).transpose(2,3)
        weights=torch.softmax((q@k.transpose(-1,-2))/(c//4)**.5,-1)
        # Anchor makes the local value generically NONPARALLEL to current x.
        values=.5*(x+anchor)
        if self.kind=='relational':
            values=values+relational_pool(x,anchor)
        if self.kind=='local':
            values=values.transpose(1,2)[:,:,None].expand(b,n,4,t,3)
            values=values*weights.diagonal(dim1=-2,dim2=-1)[...,None]
        else:
            values=weights@values.transpose(1,2)[:,:,None]
        values=values.permute(0,3,1,2,4)
        direct,cross=self.residual_gate(h.transpose(1,2)).chunk(2,-1)
        correction=(direct[...,None]*values+cross[...,None]*torch.linalg.cross(x[...,None,:],values)).sum(-2)
        correction=tangent_project(x,correction)
        return torch.cat((torch.zeros_like(correction[:,:1]),correction[:,1:]),1)

    def velocity(self,x,tau,initial,times,temperature=None,frame=None):
        return super().velocity(x,tau,initial,times,temperature,frame)+self.residual(x,tau,initial,times,frame)


def make_model(config,arm):
    return RepairModel(config,arm)
