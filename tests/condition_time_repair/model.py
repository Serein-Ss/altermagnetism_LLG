"""Zero-initialized conditional/relative-time adapters to a frozen baseline design."""
import torch
from torch import nn
from tests.spin_path_transformer.model import fourier,neighbor_mean


class RelativeAttention(nn.Module):
    def __init__(self,attention):
        super().__init__();self.attention=attention;self.time=None
        self.bias=nn.Sequential(nn.Linear(18,32),nn.SiLU(),nn.Linear(32,attention.num_heads))
        nn.init.zeros_(self.bias[-1].weight);nn.init.zeros_(self.bias[-1].bias)

    def forward(self,query,key,value,need_weights=False):
        if self.time is None or len(self.time)!=query.shape[1]:raise ValueError('physical time required')
        dt=self.time[:,None]-self.time[None,:]
        bias=self.bias(torch.cat((fourier(dt),fourier(dt.abs())),-1)).permute(2,0,1)
        mask=bias[None].expand(query.shape[0],-1,-1,-1).reshape(-1,len(self.time),len(self.time))
        return self.attention(query,key,value,attn_mask=mask,need_weights=need_weights)


class AnchorBlock(nn.Module):
    def __init__(self,block,width):
        super().__init__();self.block=block;self.time=None
        self.film=nn.Sequential(nn.Linear(16,width),nn.SiLU(),nn.Linear(width,2*width))
        nn.init.zeros_(self.film[-1].weight);nn.init.zeros_(self.film[-1].bias)
        self.norm=nn.LayerNorm(width,elementwise_affine=False)

    def forward(self,h,cond,g,edge,raw):
        if self.time is None or len(self.time)!=h.shape[1]:raise ValueError('physical time required')
        anchor=raw[...,3:6];neighbors=neighbor_mean(anchor,g['edges'])
        t=fourier(self.time)[None,:,None].expand(*h.shape[:-1],9)
        features=torch.cat((anchor,neighbors,(anchor*neighbors).sum(-1,keepdim=True),t),-1)
        shift,scale=self.film(features).chunk(2,-1)
        h=h+shift+torch.tanh(scale)*self.norm(h)
        return self.block(h,cond,g,edge,raw)


class RepairedNet(nn.Module):
    def __init__(self,base,repair):
        super().__init__()
        if repair not in ('control','anchor','relative_time'):raise ValueError(repair)
        self.base=base;self.repair=repair;self.source_shift=base.source_shift;self.arm=base.arm
        for i,block in enumerate(base.blocks):
            if repair=='anchor':base.blocks[i]=AnchorBlock(block,base.width)
            elif repair=='relative_time':block.temporal=RelativeAttention(block.temporal)
        parameter=next(base.parameters());self.to(device=parameter.device,dtype=parameter.dtype)

    def get_extra_state(self):
        return dict(version=1,repair=self.repair,base=self.base.get_extra_state(),
            initial='actual supplied S0 only',relative_time_units='ps',future_conditioning=False)

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('repair checkpoint contract differs')

    def set_time(self,time):
        for block in self.base.blocks:
            if isinstance(block,AnchorBlock):block.time=time
            elif isinstance(block.temporal,RelativeAttention):block.temporal.time=time

    def forward(self,x,tau,initial,cell,frame,temperature,time):
        self.set_time(time)
        return self.base(x,tau,initial,cell,frame,temperature,time)

    def velocity(self,x,tau,initial,times,temperature=None,frame=None):
        self.set_time(times)
        return self.base.velocity(x,tau,initial,times,temperature,frame)
