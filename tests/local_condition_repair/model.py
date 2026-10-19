"""Learned residual flow readout, no analytic Hamiltonian or LLG field input."""
import torch
from torch import nn
from scripts.model.sphere import tangent_project
from tests.transition_block_repair.model import TransitionNet
from tests.spin_path_transformer.model import geometry,neighbor_mean,fourier


class LocalReadout(TransitionNet):
    def __init__(self,mode):
        if mode not in ('pointwise','neighbor'):raise ValueError(mode)
        self.readout_mode=mode
        super().__init__()
        self.readout=nn.Sequential(nn.Linear(31,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,3))
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)
        self._readout_geometry={}

    def get_extra_state(self):
        state=super().get_extra_state()
        state.update(package='local_condition_repair',readout=self.readout_mode,readout_width=64,readout_version=1)
        return state

    def features(self,x,tau,initial,cell,frame,time):
        # Components in the jointly rotated crystal frame are invariant scalars.
        s=x.flatten(2,-2);body=s@frame.T
        anchor=(initial.flatten(1,-2)@frame.T)[:,None].expand_as(body)
        key=(cell.shape,cell.basis,tuple((a,b,d) for a,b,d,_ in cell.templates),tuple(cell.periodic),str(x.device),x.dtype)
        if key not in self._readout_geometry:self._readout_geometry[key]=geometry(cell,x.device,x.dtype)
        g=self._readout_geometry[key]
        if self.readout_mode=='neighbor':
            u=neighbor_mean(anchor,g['edges']);v=neighbor_mean(body,g['edges'])
        else:
            u=body*anchor;v=body-anchor
        b,t,n,_=body.shape
        physical=fourier(time)[None,:,None].expand(b,t,n,-1)
        transport=fourier(tau)[:,None,None].expand(b,t,n,-1)
        degree=g['degree'].log1p()[None,None,:,None].expand(b,t,n,1)
        return torch.cat((body,anchor,u,v,physical,transport,degree),-1)

    def forward(self,x,tau,initial,cell,frame,temperature,time):
        base=super().forward(x,tau,initial,cell,frame,temperature,time)
        correction=self.readout(self.features(x,tau,initial,cell,frame,time))@frame
        correction=tangent_project(x.flatten(2,-2),correction).reshape_as(x)
        correction=torch.cat((torch.zeros_like(correction[:,:1]),correction[:,1:]),1)
        return base+correction


def initialize(checkpoint,mode,seed,device='cuda'):
    # Adapter initialization is paired without changing native data RNG streams.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(949000+seed)
        net=TransitionNet() if mode=='baseline' else LocalReadout(mode)
    if mode=='baseline':net.load_state_dict(checkpoint['model'])
    else:
        assert checkpoint['model']['_extra_state']==TransitionNet().get_extra_state()
        weights={k:v for k,v in checkpoint['model'].items() if k!='_extra_state'}
        result=net.load_state_dict(weights,strict=False)
        assert not result.unexpected_keys and set(result.missing_keys)=={'_extra_state',*[f'readout.{i}.{s}' for i in (0,2,4) for s in ('weight','bias')]}
    net.to(device)
    original=[p for n,p in net.named_parameters() if not n.startswith('readout.')]
    opt=torch.optim.AdamW(original,lr=3e-5,weight_decay=1e-5);opt.load_state_dict(checkpoint['optimizer'])
    if mode!='baseline':opt.add_param_group(dict(params=list(net.readout.parameters()),lr=3e-5,weight_decay=1e-5))
    return net,opt
