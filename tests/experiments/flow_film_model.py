"""Zero-initialized flow-time FiLM candidate; not a validated physical model.

Retains Cartesian axes/site coordinates and replicate padding from the control.
This does NOT add rotational equivariance or establish size generalization.
"""
import torch
from torch import nn
from scripts.model.sphere import tangent_project
from tests.experiments.velocity_controls import CartesianPathNet


class FlowFiLMPathNet(CartesianPathNet):
    def __init__(self, width=64, blocks=4):
        super().__init__(width=width, blocks=blocks)
        self.flow_film = nn.ModuleList([nn.Linear(17, 2*width) for _ in range(blocks)])
        for layer in self.flow_film:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, state, tau, initial, condition, time, graph, axis, *, tangent=True):
        b,f,a,n,ny,_ = state.shape
        if a != 1 or ny != 1:
            raise ValueError('Bauer chain only')
        x=state[:,:,0,:,0]; anchor=initial[:,0,:,0][:,None].expand_as(x)
        scaled=(condition-condition.new_tensor([.12,.1]))/condition.new_tensor([.01,1.])
        t=time-time[:,:1]
        phase=t[...,None]*torch.arange(1,9,device=x.device,dtype=x.dtype)*torch.pi
        temporal=torch.cat((phase.sin(),phase.cos(),t[...,None]),-1)[:,:,None].expand(-1,-1,n,-1)
        site=torch.linspace(-1,1,n,device=x.device,dtype=x.dtype)[None,None,:,None].expand(b,f,-1,-1)
        features=torch.cat((x,anchor,scaled[:,None,None].expand(-1,f,n,-1),
                            tau[:,None,None,None].expand(-1,f,n,-1),temporal,site),-1)
        flow_phase=tau[:,None]*(2.**torch.arange(8,device=x.device,dtype=x.dtype))*2*torch.pi
        flow_features=torch.cat((tau[:,None],flow_phase.sin(),flow_phase.cos()),-1)
        h=torch.nn.functional.silu(self.input(features.permute(0,3,1,2)))
        for block,film in zip(self.blocks,self.flow_film):
            scale,shift=film(flow_features).chunk(2,-1)
            h=h+.1*block(h)
            h=h*(1+scale[:,:,None,None])+shift[:,:,None,None]
        v=self.output(h).permute(0,2,3,1)[:,:,None,:,None]
        if tangent:v=tangent_project(state,v)
        return torch.cat((torch.zeros_like(v[:,:1]),v[:,1:]),1)
