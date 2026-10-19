"""Measure the trained readout, not just parameter norms (CPU, no training)."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.temporal_source_repair.source import draw
from tests.local_condition_repair.model import LocalReadout
from tests.training_restart_repair.data import load_train, rows
from tests.bauer_p2_20260923.common import dump

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


@torch.no_grad()
def audit():
    torch.set_num_threads(1)
    bank,_=load_train();meta=rows();result=[]
    y=F.normalize(torch.tensor(np.stack([x[0] for x in bank])),dim=-1)
    for index in (1,2,4,5,7,8):
        ck=torch.load(ROOT/f'tests/local_condition_repair/train/{index:02d}/checkpoint.pt',map_location='cpu',weights_only=True)
        mode=ck['task']['repair'];net=LocalReadout(mode);net.load_state_dict(ck['model']);net.eval()
        for tau_value in (.1,.5,.9):
            g=torch.Generator().manual_seed(950000)
            z=draw(y[:,0],2,g,0.);tau=torch.full((len(y),),tau_value)
            x,v,_=path_interpolate(z,y,tau,g)
            full=net.velocity(x,tau,y[:,0],y.new_tensor([0.,.1]))
            # Bypass only the added head, leaving its trained backbone unchanged.
            hook=net.readout.register_forward_hook(lambda module,args,out:torch.zeros_like(out))
            try:base=net.velocity(x,tau,y[:,0],y.new_tensor([0.,.1]))
            finally:hook.remove()
            correction=full-base
            for family in sorted({r['parent']['family'] for r in meta}):
                ids=[i for i,r in enumerate(meta) if r['parent']['family']==family]
                rms=lambda a:float(a[ids,1:].square().mean().sqrt())
                result.append(dict(seed=ck['task']['seed'],mode=mode,tau=tau_value,family=family,
                    correction_rms=rms(correction),base_rms=rms(base),relative_rms=rms(correction)/max(rms(base),1e-12),
                    fm_full=float((full[ids,1:]-v[ids,1:]).square().sum(-1).mean()),
                    fm_disabled=float((base[ids,1:]-v[ids,1:]).square().sum(-1).mean())))
        del ck,net
    dump(HERE/'activation_audit.json',dict(rows=result,scope='TRAIN states, first recorded future only, fixed transport probes; not sampling-distribution certification'))
    for mode in ('pointwise','neighbor'):
        q=[r for r in result if r['mode']==mode]
        print(mode,'relative_rms range',min(r['relative_rms'] for r in q),max(r['relative_rms'] for r in q))
    old=json.loads((ROOT/'tests/local_condition_repair/assets/summary.json').read_text())
    for family in sorted({r['parent']['family'] for r in meta}):
        q=[x[5:] for x in old['intermediate'] if x[1]=='baseline' and x[3]==family]
        print('DEV',family,np.mean(q,axis=0).tolist())


if __name__=='__main__':audit()
