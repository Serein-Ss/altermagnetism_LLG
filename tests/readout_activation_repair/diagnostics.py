"""Head intervention with fixed backbone and paired latent streams."""
import json
from contextlib import contextmanager
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.temporal_source_repair.source import draw
from tests.transition_block_repair.model import rollout
from tests.training_restart_repair.data import load_train,rows
from tests.intermediate_restart_repair.run import selected,initial


@contextmanager
def disabled(net):
    hook=net.readout.register_forward_hook(lambda module,args,out:torch.zeros_like(out))
    try:yield
    finally:hook.remove()


@torch.no_grad()
def probe(net):
    device=next(net.parameters()).device
    bank,_=load_train();meta=rows();result=[]
    y=F.normalize(torch.tensor(np.stack([x[0] for x in bank]),device=device),dim=-1)
    for tau_value in (.1,.5,.9):
        g=torch.Generator(device=device).manual_seed(950000)
        z=draw(y[:,0],2,g,0.);tau=torch.full((len(y),),tau_value,device=device)
        x,v,_=path_interpolate(z,y,tau,g)
        full=net.velocity(x,tau,y[:,0],y.new_tensor([0.,.1]))
        with disabled(net):base=net.velocity(x,tau,y[:,0],y.new_tensor([0.,.1]))
        for family in sorted({r['parent']['family'] for r in meta}):
            ids=[i for i,r in enumerate(meta) if r['parent']['family']==family]
            rms=lambda a:float(a[ids,1:].square().mean().sqrt())
            result.append(dict(family=family,tau=tau_value,correction_rms=rms(full-base),base_rms=rms(base),
                relative_rms=rms(full-base)/max(rms(base),1e-12),
                fm_full=float((full[ids,1:]-v[ids,1:]).square().sum(-1).mean()),
                fm_disabled=float((base[ids,1:]-v[ids,1:]).square().sum(-1).mean())))
    return result


def ablation_rows(smoke=False):
    data=[r for r in selected() if r['origin']=='llg' and r['parent']['L']==16 and r['branch']==0 and r['frame']==3]
    assert len(data)==6
    return data[:1] if smoke else data


@torch.no_grad()
def ablation(net,out,smoke):
    device=next(net.parameters()).device
    with h5py.File(out/'ablation.h5','x') as h:
        h.attrs.update(complete=False,scope='DEV L16 six fixed states;head off intervention,not a trained alternative')
        for row in ablation_rows(smoke):
            sid=row['index'];n=4 if smoke else 128
            a=F.normalize(torch.tensor(initial(row),dtype=torch.float32,device=device),dim=-1)[None].expand(n,-1,-1)
            def generate():
                g=torch.Generator(device=device).manual_seed(950100+sid)
                return rollout(net,a,g,8 if smoke else 128,1).cpu().numpy()
            on=generate()
            with disabled(net):off=generate()
            for x in (on,off):
                assert np.isfinite(x).all() and abs(np.linalg.norm(x,axis=-1)-1).max()<2e-6
                assert np.array_equal(x[:,0],a.cpu().numpy())
            group=h.create_group(f'state{sid:03d}');group.attrs['metadata']=json.dumps(row)
            group.create_dataset('on',data=on,compression='lzf');group.create_dataset('off',data=off,compression='lzf')
        h.attrs['complete']=True
