"""Descriptive local-state coverage; replay actual native training conditions."""
import argparse
import hashlib
import json
from collections import Counter
import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tests.bauer_p2_20260923.common import cfg,dump
from tests.bauer_comprehensive_20260923.run import real_data,next_batch
from tests.branch_coverage_repair.run import pools,parents as train_parents
from tests.sampling_balance_repair.run import RefillWindows
from tests.transition_block_repair.model import pair_window
from tests.intermediate_restart_repair.run import selected,initial
from tests.training_restart_repair.data import rows,state
from .run import HERE,ROOT,verify


def descriptors(s,axis=None):
    s=np.asarray(s,dtype=np.float64)
    if s.ndim<2 or s.shape[-1]!=3 or s.shape[-2]<5 or not np.isfinite(s).all():raise ValueError('spin chain with at least5sites required')
    axis=np.array([0.,0.,1.]) if axis is None else np.asarray(axis)
    L=s.shape[-2];items=[]
    for lag in (1,2):
        left=np.zeros_like(s);right=np.zeros_like(s);left[...,lag:,:]=s[...,:-lag,:];right[...,:-lag,:]=s[...,lag:,:]
        degree=(np.arange(L)>=lag).astype(float)+(np.arange(L)<L-lag)
        v=(left+right)/degree[:,None]
        dot=(s*v).sum(-1)
        var=(((s*left).sum(-1)**2+(s*right).sum(-1)**2)/degree-dot**2)
        items.append((v,dot,var,degree))
    v,d,var,degree=items[0];v2,d2,_,_=items[1]
    edge=np.minimum(np.minimum(np.arange(L),L-1-np.arange(L)),3)/3
    return np.stack([s@axis,v@axis,d,(v*v).sum(-1),np.cross(s,v)@axis,var,v2@axis,d2,
                     np.broadcast_to(degree/2,s.shape[:-1]),np.broadcast_to(edge,s.shape[:-1])],-1)


def nearest(bank,query,scale):
    return cKDTree(bank/scale).query(query/scale,k=1,workers=1)[0]


def run(index,smoke):
    torch.set_num_threads(1);seed=(11,22,33)[index]
    ck=torch.load(ROOT/f'tests/transition_block_repair/train/{index*2+1:02d}/step12000.pt',map_location='cpu',weights_only=True)
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    refill=np.random.default_rng();refill.bit_generator.state=ck['refill_rng']
    pair_rng=np.random.default_rng();pair_rng.bit_generator.state=ck['pair_rng']
    _,new,_=pools(False);pool={r['parent_id']:x for r,x in zip(train_parents(),new)}
    audit=dict(counts=Counter(),hash=hashlib.sha256());config=cfg();data=[RefillWindows(d,pool,refill,True,audit) for d in real_data(config)]
    blocks=[]
    for step in range(16 if smoke else 6000):
        whole,_=next_batch(dict(domain='bauer'),config,data,rng,None,'cpu')
        starts=np.where(pair_rng.random(len(whole))<.5,0,pair_rng.integers(1,10,len(whole)))
        y=pair_window(whole,torch.as_tensor(starts));blocks.append(descriptors(y[:,0].numpy()).reshape(-1,10).astype('f4'))
    if not smoke:
        expected=json.loads((ROOT/f'tests/training_restart_repair/train/{index*3:02d}/complete.json').read_text())['schedule_hash']
        assert audit['hash'].hexdigest()==expected,'CPU replay changed parent/window/refill selection'
        end=torch.load(ROOT/f'tests/training_restart_repair/train/{index*3:02d}/checkpoint.pt',map_location='cpu',weights_only=True)
        assert pair_rng.bit_generator.state==end['pair_rng']
        assert rng.bit_generator.state==end['data_rng'] and refill.bit_generator.state==end['refill_rng']
    bank=np.concatenate(blocks);del blocks,data
    mean=bank.mean(0,dtype=np.float64);scale=np.maximum(bank.std(0,dtype=np.float64),.05)
    added=np.concatenate([descriptors(state(r)) for r in rows()])
    dev=[r for r in selected() if r['origin']=='llg'];dist={}
    query=np.concatenate([descriptors(initial(r)) for r in dev]);a=nearest(bank,query,scale);b=nearest(added,query,scale);aug=np.minimum(a,b)
    offset=0
    for r in dev:
        n=r['parent']['L'];dist[r['index']]=(a[offset:offset+n],aug[offset:offset+n]);offset+=n
    out=HERE/('smoke_coverage' if smoke else 'coverage')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    result=[];site_rows=[]
    for arm_index,arm in enumerate(('native','restart_fm','restart_moments')):
        old=json.loads((ROOT/f'tests/training_restart_repair/train/{index*3+arm_index:02d}/report.json').read_text())
        for r in old['intermediate']:
            meta=r['metadata'];sid=meta['index'];metric=r['metrics']['fine'];d0,d1=dist[sid]
            em=np.sqrt(np.mean(np.asarray(metric['increment_mean']['difference'])**2,axis=-1))
            ec=np.sqrt(np.mean(np.asarray(metric['increment_cov']['difference'])**2,axis=(-1,-2)))
            result.append(dict(seed=seed,arm=arm,state=sid,parent=meta['parent']['parent_id'],L=meta['parent']['L'],family=meta['parent']['family'],time_ps=meta['origin_time_ps'],
                native_distance=float(d0.mean()),augmented_distance=float(d1.mean()),mean_rmse=metric['increment_mean']['rmse'],covariance_rmse=metric['increment_cov']['rmse']))
            for j in range(len(d0)):site_rows.append([seed,arm,sid,meta['parent']['parent_id'],j,min(j,len(d0)-1-j),float(d0[j]),float(d1[j]),float(em[j]),float(ec[j])])
        del old
    dump(out/'summary.json',dict(seed=seed,rows=result,site_rows=site_rows,training_descriptors=len(bank),feature_mean=mean.tolist(),feature_scale=scale.tolist(),
        schedule_hash=audit['hash'].hexdigest(),smoke=smoke,limitations='Replay only this6000update continuation,not complete lifetime training support. Handchosen radius1/2features are not sufficient statistics. Correlated sites/states are descriptive,not independent samples or coverage certification. Scaling fits native train exposure only; no dev-driven threshold.'))
    fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for arm,color in zip(('native','restart_fm','restart_moments'),('#0072B2','#D55E00','#009E73')):
        rr=[r for r in result if r['arm']==arm]
        for axis,key,title in zip(ax,('native_distance','augmented_distance'),('Native training exposure','Native + restart states')):
            axis.scatter([r[key] for r in rr],[r['mean_rmse'] for r in rr],s=15,alpha=.65,color=color,label=arm)
            axis.set(xlabel='Mean standardized local nearest distance',ylabel='Conditional increment mean RMSE',title=title)
    ax[0].legend(fontsize=8);fig.savefig(out/'coverage.png',dpi=200);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args();verify();run(a.index,a.smoke)
