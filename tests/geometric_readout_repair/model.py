"""Learned spherical readout; no Hamiltonian, future or LLG torque input."""
import torch
from torch import nn
from scripts.model.sphere import tangent_project
from tests.transition_block_repair.model import TransitionNet
from tests.spin_path_transformer.model import fourier

def basis(x,initial):
    a=initial[:,None].expand_as(x)
    dot=(x*a).sum(-1,keepdim=True).clamp(-1,1)
    tangent=tangent_project(x,a)
    sine=tangent.norm(dim=-1,keepdim=True)
    if torch.any((sine<1e-7)&(dot<0)):
        raise ValueError('geodesic readout undefined at an antipode')
    angle=torch.atan2(sine,dot)
    log=tangent*(angle/sine.clamp_min(1e-7))
    log=torch.where(sine<1e-7,tangent,log)
    return log,torch.linalg.cross(x,a)

class GeometricNet(TransitionNet):
    def __init__(self):
        super().__init__()
        self.geometric_head=nn.Sequential(nn.Linear(9,32),nn.SiLU(),nn.Linear(32,2))
        nn.init.zeros_(self.geometric_head[-1].weight)
        nn.init.zeros_(self.geometric_head[-1].bias)

    def get_extra_state(self):
        state=super().get_extra_state()
        state.update(package='geometric_readout_repair',geometric_head='learned_tau_log_and_cross_S0',version=3)
        return state

    def velocity(self,x,tau,initial,times,temperature=None,frame=None):
        base=super().velocity(x,tau,initial,times,temperature,frame)
        log,cross=basis(x,initial)
        coef=self.geometric_head(fourier(tau))[:,None,None]
        extra=coef[...,0:1]*log+coef[...,1:2]*cross
        extra=tangent_project(x,extra)
        return base+torch.cat((torch.zeros_like(extra[:,:1]),extra[:,1:]),1)

def initialize(checkpoint,enabled):
    if not enabled:
        net=TransitionNet();net.load_state_dict(checkpoint['model']);return net
    net=GeometricNet()
    state={k:v for k,v in checkpoint['model'].items() if k!='_extra_state'}
    result=net.load_state_dict(state,strict=False)
    assert set(result.missing_keys)=={'_extra_state',*(f'geometric_head.{i}.{k}' for i in (0,2) for k in ('weight','bias'))}
    assert not result.unexpected_keys
    return net
