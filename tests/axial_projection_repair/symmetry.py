"""Bauer-only spin rotation about its fixed z easy axis; not arbitrary SO(3)."""
import torch


def rotate(x,angle):
    angle=torch.as_tensor(angle,device=x.device,dtype=x.dtype)
    if angle.ndim==1:
        if len(angle)!=len(x):raise ValueError('one angle per path')
        angle=angle.reshape(len(x),*([1]*(x.ndim-2)))
    elif angle.ndim!=0:raise ValueError('scalar or batch angles')
    c,s=angle.cos(),angle.sin()
    return torch.stack((c*x[...,0]-s*x[...,1],s*x[...,0]+c*x[...,1],x[...,2]),-1)


@torch.no_grad()
def probe(net):
    import numpy as np
    from tests.training_restart_repair.data import load_train,rows
    from tests.temporal_source_repair.source import draw
    from scripts.model.gate_f import path_interpolate
    bank,_=load_train();meta=rows();device=next(net.parameters()).device
    y=torch.nn.functional.normalize(torch.tensor(np.stack([b[0] for b in bank]),device=device),dim=-1)
    result=[]
    for tv in (.1,.5,.9):
        g=torch.Generator(device=device).manual_seed(952000)
        z=draw(y[:,0],2,g,0.);tau=torch.full((len(y),),tv,device=device);x,v,_=path_interpolate(z,y,tau,g)
        base=net.velocity(x,tau,y[:,0],y.new_tensor([0.,.1]))
        for angle in (.37,.7853981633974483,1.5707963267948966,3.141592653589793):
            rotated=net.velocity(rotate(x,angle),tau,rotate(y[:,0],angle),y.new_tensor([0.,.1]))
            expected=rotate(base,angle)
            for family in sorted({r['parent']['family'] for r in meta}):
                ids=[i for i,r in enumerate(meta) if r['parent']['family']==family]
                rms=lambda a:float(a[ids,1:].square().mean().sqrt())
                result.append(dict(family=family,tau=tv,angle=angle,absolute_rms=rms(rotated-expected),
                    relative_rms=rms(rotated-expected)/max(rms(base),1e-12)))
    return result

