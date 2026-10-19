"""Diagnostic capacity controls, not validated dynamical models."""
import torch
from torch import nn
from scripts.model.sphere import tangent_project


class VelocityTable(nn.Module):
    """Fixed-batch optimization control with no inference/generalization meaning."""
    def __init__(self, shape):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(shape))

    def forward(self, state, *args, **kwargs):
        if state.shape != self.value.shape:
            raise ValueError('fixed batch only')
        v = tangent_project(state, self.value)
        return torch.cat((torch.zeros_like(v[:, :1]), v[:, 1:]), 1)


class CartesianPathNet(nn.Module):
    """Bauer open-chain diagnostic; no Hamiltonian input or exact equivariance.

    Absolute normalized site coordinate and Fourier time features deliberately
    relax restrictions to test fitting capacity, not establish size transfer.
    """
    def __init__(self, width=64, blocks=4, temporal_dilations=None, global_context=False):
        super().__init__()
        dilations = (1,)*blocks if temporal_dilations is None else tuple(temporal_dilations)
        if len(dilations) != blocks or any(not isinstance(d, int) or d < 1 for d in dilations):
            raise ValueError('one positive integer temporal dilation per block required')
        self.input = nn.Conv2d(27, width, 1)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(width, width, 3, padding=(d,1), dilation=(d,1), padding_mode='replicate'), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=(d,1), dilation=(d,1), padding_mode='replicate')) for d in dilations])
        self.output = nn.Conv2d(width, 3, 1)
        self.context = nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(blocks)]) if global_context else None
        if self.context is not None:
            for layer in self.context:
                nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self, state, tau, initial, condition, time, graph, axis, *, tangent=True):
        b,f,a,n,ny,_ = state.shape
        if a != 1 or ny != 1:
            raise ValueError('Bauer chain only')
        x=state[:,:,0,:,0];anchor=initial[:,0,:,0][:,None].expand_as(x)
        scaled=(condition-condition.new_tensor([.12,.1]))/condition.new_tensor([.01,1.])
        t=time-time[:,:1]
        phase=t[...,None]*torch.arange(1,9,device=x.device,dtype=x.dtype)*torch.pi
        temporal=torch.cat((phase.sin(),phase.cos(),t[...,None]),-1)[:,:,None].expand(-1,-1,n,-1)
        site=torch.linspace(-1,1,n,device=x.device,dtype=x.dtype)[None,None,:,None].expand(b,f,-1,-1)
        features=torch.cat((x,anchor,scaled[:,None,None].expand(-1,f,n,-1),
                            tau[:,None,None,None].expand(-1,f,n,-1),temporal,site),-1)
        h=torch.nn.functional.silu(self.input(features.permute(0,3,1,2)))
        for index,block in enumerate(self.blocks):
            local=h+.1*block(h)
            h=local if self.context is None else local+.1*self.context[index](h.mean(dim=(2,3),keepdim=True))
        v=self.output(h).permute(0,2,3,1)[:,:,None,:,None]
        if tangent:v=tangent_project(state,v)
        return torch.cat((torch.zeros_like(v[:,:1]),v[:,1:]),1)
