"""Exact conditional velocity for uniform-sphere source and discrete whole-path modes.

For r=angle(x,y), a=1-tau, p_tau(x|y) is sin(r/a)/(4*pi*a*sin(r))
on r<a*pi. Product over future spins gives a WHOLE-PATH mode posterior.
We omit the common 1/(4*pi). Targets are deterministic synthetic rotations,
not LLG. Numerical trajectories outside BOTH supports are explicitly counted.
Endpoint is tau=.95 (non-singular target law), not relabelled as tau=1.
"""
import math
import torch
from torch.nn import functional as F
from scripts.model.sphere import sphere_exp, sphere_log
from tests.bauer_comprehensive_20260923.components import synthetic_parents
from tests.bauer_p2_20260923.common import cfg
from tests.bauer_repair_20260924.model import RepairModel
from .model import source

STOP = .95


def targets(initial, modes=2):
    time = torch.linspace(0,1,11,device=initial.device,dtype=initial.dtype)
    out=[]
    for sign in ([1] if modes==1 else [-1,1]):
        angle=sign*.8*time
        c,s=angle.cos()[None,:,None],angle.sin()[None,:,None]
        a=initial[:,None].expand(-1,11,-1,-1)
        y=torch.stack((c*a[...,0]-s*a[...,1],s*a[...,0]+c*a[...,1],a[...,2]),-1)
        out.append(y)
    return torch.stack(out,1)


def conditional(x,tau,ys):
    dot=(x[:,None]*ys).sum(-1).clamp(-1.,1.)
    r=torch.acos(dot.clamp(-1+1e-7,1-1e-7))
    a=(1-tau)[:,None,None,None]
    q=r/a
    valid=(q[:,:,1:] < math.pi).flatten(2).all(-1)
    # sinc form has a regular r=0 limit 1/a^2.
    logp=(torch.sinc(q/math.pi).clamp_min(1e-30).log()
          -torch.sinc(r/math.pi).clamp_min(1e-30).log()-2*a.log())[:,:,1:].flatten(2).sum(-1)
    logp=logp.masked_fill(~valid,-torch.inf)
    unsupported=~valid.any(-1)
    # Explicit emergency extension for numerical solver diagnostics only.
    nearest=r[:,:,1:].square().flatten(2).sum(-1).argmin(-1)
    fallback=F.one_hot(nearest,ys.shape[1]).to(x)
    weights=torch.softmax(torch.where(unsupported[:,None],torch.zeros_like(logp),logp),-1)
    weights=torch.where(unsupported[:,None],fallback,weights)
    velocities=sphere_log(x[:,None].expand_as(ys),ys)/a[...,None]
    v=(weights[:,:,None,None,None]*velocities).sum(1)
    v[:,0]=0
    return v,weights,unsupported


class OracleNetwork(RepairModel):
    def get_extra_state(self):
        state=super().get_extra_state()
        state.update(source='independent_uniform_spheres',source_shift=0.,rho=0.,transport_stop=STOP)
        return state


def make_net(kind,device):
    config=cfg();config.update(domain='analytic_whole_modes',time_unit='dimensionless',sizes=[16])
    return OracleNetwork(config,'relational_global' if kind=='global' else 'baseline').to(device)


@torch.no_grad()
def integrate(initial,ys,steps,generator,net=None):
    x=source(initial,11,generator,0.)
    times=torch.linspace(0,1,11,device=x.device)
    outside=0
    for k in range(steps):
        tau=x.new_full((len(x),),STOP*k/steps)
        if net is None:
            v,_,bad=conditional(x,tau,ys);outside+=int(bad.sum())
        else:v=net.velocity(x,tau,initial,times)
        x=sphere_exp(x,v*(STOP/steps));x[:,0]=initial
    _,weights,bad=conditional(x,x.new_full((len(x),),STOP),ys)
    return x,dict(outside_support_evaluations=outside,final_outside_support=int(bad.sum()),
                  transport_stop=STOP,oracle_positive_posterior=float(weights[:,-1].mean()))
