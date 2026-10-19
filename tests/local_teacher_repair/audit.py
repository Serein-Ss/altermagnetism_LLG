"""Bounded TRAIN-only posterior diagnostic and frozen prior-result summaries."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tests.bauer_p2_20260923.common import dump
from tests.deployment_moment_repair.run import banks
from tests.temporal_source_repair.source import draw
from scripts.model.gate_f import path_interpolate
from tests.empirical_transport_repair.oracle import conditional
from .target import patch_conditional

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    torch.set_num_threads(1);bank,hashes=banks();rows=[]
    for role in (0,2):
        for sid in (0,12,24,36):
            target=F.normalize(torch.tensor(bank[role][sid],dtype=torch.float64),dim=-1);y=target[:16]
            for tv in (0.,.1,.3,.5,.7,.9):
                gen=torch.Generator().manual_seed(984900+role*100+sid);z=draw(y[:,0],2,gen,0.)
                tau=torch.full((16,),tv,dtype=torch.float64);x,v,cut=path_interpolate(z,y,tau,gen)
                assert cut['unresolved_cut_count']==0
                whole,w,bad=conditional(x,tau,target,y[:,0]);assert not bad.any()
                for radius in (0,1,2,4):
                    local,ess=patch_conditional(x,tau,target,y[:,0],radius)
                    rows.append(dict(role=role,state=sid,tau=tv,radius=radius,whole_ess=float((1/w.square().sum(-1)).mean()),
                        patch_ess=float(ess.mean()),whole_near_single_mode_fraction=float(((1/w.square().sum(-1))<1.1).double().mean()),
                        patch_near_single_mode_fraction=float((ess<1.1).double().mean()),
                        whole_label_residual=float((v[:,1:]-whole[:,1:]).square().sum(-1).mean()),
                        patch_label_residual=float((v[:,1:]-local[:,1:]).square().sum(-1).mean()),
                        target_change=float((local[:,1:]-whole[:,1:]).square().sum(-1).mean())))
    previous=json.loads((ROOT/'tests/auxiliary_frequency_repair/assets/summary.json').read_text())
    prior=[]
    for arm in ('sparse64','dense8_same_mass','dense8_strong'):
        prior.append(dict(arm=arm,dev_mean=np.mean([r[2:] for r in previous['dev'] if r[1]==arm],0).tolist(),
            full_mean=np.mean([r[4:] for r in previous['full'] if r[1]==arm and r[3]=='midpoint128'],0).tolist(),
            train_seconds_mean=float(np.mean([r[3] for r in previous['cost'] if r[1]==arm]))))
    dump(HERE/'audit.json',dict(rows=rows,hashes=hashes,previous=prior,scientific_certification=False,
        scope='8 TRAIN states from two origins, first16 fixed branches; descriptive conditional posterior of finite library, not physical modes or independent preparation counts',
        interpretation='patch expectation is a projection on less information, not the exact whole-chain velocity; reduced empirical degeneracy does not prove joint physical accuracy'))
    print('tau, whole ESS, radius2 ESS',[(tv,float(np.mean([r['whole_ess'] for r in rows if r['tau']==tv and r['radius']==2])),float(np.mean([r['patch_ess'] for r in rows if r['tau']==tv and r['radius']==2]))) for tv in (0.,.1,.3,.5,.7,.9)])


if __name__=='__main__':main()
