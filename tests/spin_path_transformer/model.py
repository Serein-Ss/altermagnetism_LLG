"""Local/latent/hierarchical magnetic path Transformer.

Frame-relative scalar tokens and vector reconstruction guarantee JOINT SO(3)
covariance. The frame is the registered crystal frame, not a spin-only symmetry.
Fixed sites, explicit image-aware bonds, no Hamiltonian coefficients or fields.
Architecture inspired by MDGen/DiT and Perceiver IO; not their reproduction.
"""
import itertools
import math
import torch
from torch import nn
from torch.nn import functional as F
from scripts.core.cell_hamiltonian import CellHamiltonian
from scripts.model.sphere import tangent_project


def geometry(cell,device,dtype,region=4):
    ndim=len(cell.shape);basis=cell.basis
    offsets=torch.tensor(getattr(cell,'offsets',[(0.,0.,0.)]*basis),dtype=dtype,device=device)
    if basis>1 and not hasattr(cell,'offsets'):
        raise ValueError('multi-basis geometry requires explicit offsets')
    vectors=torch.eye(3,dtype=dtype,device=device)[:ndim].clone()
    if hasattr(cell,'offsets'):
        if type(cell).__name__!='TriangularCo' or ndim!=2:
            raise ValueError('unregistered multi-basis lattice vectors')
        vectors[1]*=math.sqrt(3.)
    coords=list(itertools.product(*(range(n) for n in cell.shape)))
    lookup={c:i for i,c in enumerate(coords)};edges=[];directions=[];groups=[]
    coarse_shape=tuple((n+region-1)//region for n in cell.shape)
    for c in coords:
        coarse=0
        for v,n in zip(c,coarse_shape):coarse=coarse*n+v//region
        groups.extend([coarse]*basis)
        for a,b,delta,_ in cell.templates:
            target=tuple(v+d for v,d in zip(c,delta))
            if any(not p and not 0<=v<n for v,n,p in zip(target,cell.shape,cell.periodic)):continue
            wrapped=tuple(v%n for v,n in zip(target,cell.shape))
            i,j=lookup[c]*basis+a,lookup[wrapped]*basis+b
            d=torch.tensor(delta,dtype=dtype,device=device)@vectors+offsets[b]-offsets[a]
            edges.extend([(i,j),(j,i)]);directions.extend([d,-d])
    ij=torch.tensor(edges,device=device,dtype=torch.long).T.contiguous()
    direction=torch.stack(directions)
    group=torch.tensor(groups,device=device,dtype=torch.long)
    count=torch.bincount(group).to(dtype)
    degree=torch.bincount(ij[0],minlength=len(groups)).to(dtype)
    coarse_edges=torch.unique(group[ij].T,dim=0).T
    coarse_edges=coarse_edges[:,coarse_edges[0]!=coarse_edges[1]]
    return dict(edges=ij,direction=direction,group=group,count=count,degree=degree,coarse_edges=coarse_edges)


def pool_mean(x,group,count):
    out=x.new_zeros(*x.shape[:-2],len(count),x.shape[-1]);out.index_add_(-2,group,x)
    return out/count.clamp_min(1)[:,None]


def neighbor_mean(x,edges):
    i,j=edges;out=torch.zeros_like(x);out.index_add_(-2,i,x[...,j,:])
    count=torch.bincount(i,minlength=x.shape[-2]).to(x).clamp_min(1)
    return out/count[:,None]


def fourier(x):
    frequency=x.new_tensor([1.,2.,4.,8.])*math.pi
    return torch.cat((x[...,None],(x[...,None]*frequency).sin(),(x[...,None]*frequency).cos()),-1)


class Block(nn.Module):
    def __init__(self,width,arm):
        super().__init__();self.arm=arm
        self.norms=nn.ModuleList([nn.LayerNorm(width,elementwise_affine=False) for _ in range(4)])
        self.modulation=nn.Sequential(nn.SiLU(),nn.Linear(width,12*width))
        self.edge=nn.Sequential(nn.Linear(2*width+6,width),nn.SiLU(),nn.Linear(width,width))
        self.temporal=nn.MultiheadAttention(width,4,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(width,4*width),nn.GELU(approximate='tanh'),nn.Linear(4*width,width))
        if arm=='dense':self.spatial=nn.MultiheadAttention(width,4,batch_first=True)
        if arm in ('latent','hierarchical'):
            self.queries=nn.Parameter(torch.randn(8,width)*.02)
            self.read=nn.MultiheadAttention(width,4,batch_first=True)
            self.process=nn.MultiheadAttention(width,4,batch_first=True)
            self.write=nn.MultiheadAttention(width,4,batch_first=True)
            self.latent_norm=nn.LayerNorm(width)
        if arm=='hierarchical':
            self.coarse=nn.Linear(width,width)
            self.coarse_geometry=nn.Linear(10,width)
        nn.init.zeros_(self.modulation[-1].weight);nn.init.zeros_(self.modulation[-1].bias)

    def forward(self,h,cond,g,edge_features,raw_vectors):
        b,t,n,c=h.shape;parts=self.modulation(cond).chunk(12,-1)
        def normalized(index):
            shift,scale,gate=parts[3*index:3*index+3]
            return self.norms[index](h)*(1+scale[:,None,None])+shift[:,None,None],gate[:,None,None]
        y,gate=normalized(0);i,j=g['edges']
        msg=self.edge(torch.cat((y[:,:,i],y[:,:,j],edge_features),-1))
        agg=torch.zeros_like(h);agg.index_add_(2,i,msg);agg=agg/g['degree'].clamp_min(1)[None,None,:,None]
        h=h+gate*agg
        y,gate=normalized(1);seq=y.transpose(1,2).reshape(b*n,t,c)
        temporal=self.temporal(seq,seq,seq,need_weights=False)[0].reshape(b,n,t,c).transpose(1,2)
        h=h+gate*temporal
        if self.arm!='local':
            y,gate=normalized(2)
            if self.arm=='dense':
                if n>512:raise ValueError('dense control restricted to <=512 sites')
                seq=y.reshape(b*t,n,c);z=self.spatial(seq,seq,seq,need_weights=False)[0].reshape_as(y)
            else:
                if self.arm=='hierarchical':
                    memory=pool_mean(y,g['group'],g['count'])
                    raw=pool_mean(raw_vectors,g['group'],g['count'])
                    counts=g['count'].log1p()[None,None,:,None].expand(b,t,-1,1)
                    memory=memory+self.coarse_geometry(torch.cat((raw,counts),-1))
                    memory=memory+self.coarse(neighbor_mean(memory,g['coarse_edges']))
                else:memory=y
                memory=memory.flatten(1,2)
                latent=self.queries[None].expand(b,-1,-1)
                latent=latent+self.read(latent,memory,memory,need_weights=False)[0]
                norm=self.latent_norm(latent)
                latent=latent+self.process(norm,norm,norm,need_weights=False)[0]
                # Fine queries retain every spin/frame. No low-resolution replacement.
                z=self.write(y.flatten(1,2),latent,latent,need_weights=False)[0].reshape_as(y)
                if self.arm=='hierarchical':z=z+memory.reshape(b,t,-1,c)[:,:,g['group']]
            h=h+gate*z
        y,gate=normalized(3);return h+gate*self.ff(y)


class SpinPathTransformer(nn.Module):
    def __init__(self,domain,arm,width=64,depth=4,periodic=None,source_shift=None):
        super().__init__()
        if arm not in ('local','latent','hierarchical','dense'):raise ValueError(arm)
        self.domain=domain;self.architecture=arm
        self.periodic=tuple(periodic if periodic is not None else ((False,False) if domain=='miltat' else (False,)))
        self.temperature_K=12.5 if domain=='miltat' else 63.824849668 if domain=='bauer' else 0.
        self.source_shift=(0. if domain=='toy' else 2.) if source_shift is None else source_shift
        # Compatibility with frozen sampler's source selector (architecture stored separately).
        self.arm='uniform' if self.source_shift==0 else 'shifted'
        self.input=nn.Linear(27,width);self.condition=nn.Sequential(nn.Linear(10,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList([Block(width,arm) for _ in range(depth)])
        self.final_norm=nn.LayerNorm(width);self.output=nn.Linear(width,6)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self._geometry={};self.width=width;self.depth=depth

    def get_extra_state(self):
        return dict(version=1,domain=self.domain,architecture=self.architecture,width=self.width,depth=self.depth,
            periodic=self.periodic,temperature_K=self.temperature_K,source_shift=self.source_shift,
            time_unit='dimensionless' if self.domain=='toy' else 'ps',frames=11,frame_interval=.1,
            transport_stop=.95 if self.domain=='toy' else 1.,crystal_frame='registered orthonormal triad',
            symmetry='joint SO(3), not reflection or spin-only symmetry',region_cells=4,latents=8)

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('checkpoint contract differs')

    def forward(self,x,tau,initial,cell,frame,temperature,time):
        if tuple(cell.periodic)!=self.periodic:raise ValueError('boundary contract differs')
        if tuple(x.shape[2:-1])!=(*cell.shape,cell.basis):raise ValueError('shape/geometry mismatch')
        if len(time)!=x.shape[1] or not torch.all(time[1:]>time[:-1]):raise ValueError('time grid')
        if self.domain!='software' and (len(time)!=11 or not torch.allclose(time,torch.arange(11,device=x.device,dtype=x.dtype)*.1,atol=1e-6,rtol=0)):
            raise ValueError('registered package requires 11 frames on its native 0.1 grid')
        if not torch.allclose(torch.as_tensor(temperature,device=x.device,dtype=x.dtype),x.new_tensor(0.)):raise ValueError('unsupported standardized temperature')
        key=(cell.shape,cell.basis,tuple((a,b,d) for a,b,d,_ in cell.templates),self.periodic,str(x.device),x.dtype)
        if key not in self._geometry:self._geometry[key]=geometry(cell,x.device,x.dtype)
        g=self._geometry[key];s=x.flatten(2,-2);a=initial.flatten(1,-2)[:,None].expand_as(s)
        body=s@frame.T;anchor=a@frame.T
        prev=torch.cat((body[:,:1],body[:,:-1]),1);nxt=torch.cat((body[:,1:],body[:,-1:]),1)
        cross=torch.linalg.cross(body,anchor)
        vec=torch.cat((body,anchor,prev,nxt,cross,(body*anchor).sum(-1,keepdim=True)),-1)
        b,t,n,_=s.shape
        meta=fourier(time)[None,:,None].expand(b,-1,n,-1)
        degree=g['degree'].log1p()[None,None,:,None].expand(b,t,-1,-1)
        types=(torch.arange(n,device=x.device)%cell.basis).to(x)/max(1,cell.basis-1)
        h=F.silu(self.input(torch.cat((vec,meta,degree,types[None,None,:,None].expand(b,t,-1,-1)),-1)))
        cond=self.condition(torch.cat((fourier(tau),torch.zeros_like(tau[:,None])),-1))
        i,j=g['edges'];d=g['direction'];length=d.norm(dim=-1,keepdim=True)
        d=d/length.clamp_min(1e-8);u,v=body[:,:,i],body[:,:,j]
        normal=x.new_tensor([0.,0.,1.]).expand_as(d)
        chiral=(torch.linalg.cross(u,v)*torch.linalg.cross(normal,d)[None,None]).sum(-1,keepdim=True)
        edge=torch.cat((d[None,None].expand(b,t,-1,-1),length[None,None].expand(b,t,-1,-1),(u*v).sum(-1,keepdim=True),chiral),-1)
        raw=torch.cat((body,anchor,cross),-1)
        for block in self.blocks:h=block(h,cond,g,edge,raw)
        direct,cross_coef=self.output(self.final_norm(h)).chunk(2,-1)
        v=direct@frame+torch.linalg.cross(s,cross_coef@frame)
        v=tangent_project(s,v).reshape_as(x)
        return torch.cat((torch.zeros_like(v[:,:1]),v[:,1:]),1)

    def velocity(self,x,tau,initial,times,temperature=None,frame=None):
        if temperature is not None and abs(float(temperature)-self.temperature_K)>1e-4:raise ValueError('unsupported temperature')
        cell=CellHamiltonian((x.shape[2],),1,[(0,0,(1,),1.)],periodic=self.periodic)
        frame=torch.eye(3,device=x.device,dtype=x.dtype) if frame is None else frame
        return self(x.unsqueeze(-2),tau,initial.unsqueeze(-2),cell,frame,0.,times).squeeze(-2)
