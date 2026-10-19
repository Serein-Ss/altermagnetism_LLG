"""Round-zero state diversity + unbiased conditional branches; CPU pilot only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
import yaml
from tests.bauer_fig2_10ns_20260917.kernel import ensemble

BASE=Path(__file__).resolve().parent
HBAR=0.6582119569


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def simulate(initial,seeds,cfg,dt_fs,duration_ps):
    p=cfg['physics']; n=cfg['numerics']; dt_ps=dt_fs/1000
    steps=round(duration_ps/dt_ps); stride=round(n['save_ps']/dt_ps)
    assert steps>0 and stride>0 and steps%stride==0
    assert np.isclose(steps*dt_ps,duration_ps) and np.isclose(stride*dt_ps,n['save_ps'])
    assert len(np.unique(seeds))==len(seeds)
    x,error=ensemble(initial,np.asarray(seeds,dtype=np.int64),dt_ps/(HBAR/p['exchange_meV']),
                     steps,stride,p['anisotropy_over_J'],p['damping'],p['theta'],0,True)
    assert np.isfinite(x).all() and np.max(abs(np.linalg.norm(x,axis=-1)-1))<1e-10
    assert np.array_equal(x[:,0],initial)
    return x,error


def descriptors(states):
    z=states[...,2]; bond=(states[:,:-1]*states[:,1:]).sum(-1)
    features=[np.quantile(z,[0,.1,.25,.5,.75,.9,1],axis=1).T,
              np.quantile(bond,[0,.1,.5,.9,1],axis=1).T,
              z[:,:3].mean(1)[:,None],z[:,-3:].mean(1)[:,None],
              ((states[:,1:]-states[:,:-1])**2).sum(-1).mean(1)[:,None]]
    for lag in [1,2,4]:
        features.append((z[:,:-lag]*z[:,lag:]).mean(1)[:,None])
    for part in np.array_split(z,4,axis=1): features.append(part.mean(1)[:,None])
    return np.concatenate(features,axis=1)


def fps(x,count):
    assert 0<count<=len(x)
    chosen=[int(np.argmax(((x-x.mean(0))**2).sum(1)))]
    d=((x-x[chosen[0]])**2).sum(1)
    while len(chosen)<count:
        d[chosen]=-1
        j=int(np.argmax(d)); chosen.append(j)
        d=np.minimum(d,((x-x[j])**2).sum(1))
    return np.asarray(chosen)


def candidate(rng,length,family,cfg):
    n=cfg['sampling']; u=np.arange(length,dtype=float); width=rng.uniform(*n['wall_width_sites'])
    phase=rng.uniform(0,2*np.pi); polarity=rng.choice([-1,1])
    if family in ['positive','negative']:
        z=np.full(length,1. if family=='positive' else -1.)
    elif family=='tilted':
        angle=rng.uniform(*n['tilt_radians'])+.15*np.sin(2*np.pi*u/length+rng.uniform(0,2*np.pi))
        z=polarity*np.cos(angle)
    elif family=='single_wall':
        z=polarity*np.tanh((u-rng.uniform(.2,.8)*length)/width)
    elif family=='edge_domain':
        z=polarity*np.tanh((u-rng.uniform(.1,.4)*length)/width)
        if rng.random()<.5: z=z[::-1]
    elif family=='interior_domain':
        left=rng.uniform(.15,.35)*length; right=rng.uniform(.65,.85)*length
        z=polarity*(1-np.tanh((u-left)/width)+np.tanh((u-right)/width))
        z=np.clip(z,-1,1)
    else: raise ValueError(family)
    a=np.sqrt(np.maximum(0,1-z*z)); s=np.stack((a*np.cos(phase),a*np.sin(phase),z),-1)
    noise=rng.normal(size=s.shape); noise-=s*(noise*s).sum(-1,keepdims=True)
    s+=rng.uniform(*n['jitter_range'])*noise
    return s/np.linalg.norm(s,axis=-1,keepdims=True)


def build(cfg,length,out):
    rows=[]; states=[]; seedbase=cfg['sampling']['seed']+length*100000
    # Each split gets independently generated parent states before any FPS or fitting.
    for split,count in cfg['sampling']['candidates_per_family'].items():
        for family in cfg['sampling']['families']:
            for _ in range(count):
                cid=len(rows); seed=seedbase+cid
                states.append(candidate(np.random.default_rng(seed),length,family,cfg))
                rows.append(dict(id=cid,parent_id=f'L{length}_parent{cid}',split=split,family=family,
                                 candidate_seed=seed,preparation_seed=seedbase+10000+cid))
    states=np.array(states)
    ids=[r['id'] for r in rows if r['family'] in ['positive','negative']]
    paths,err=simulate(states[ids],np.array([rows[i]['preparation_seed'] for i in ids]),cfg,
                      cfg['numerics']['dt_fs'],cfg['numerics']['preparation_ps'])
    states[ids]=paths[:,-1]
    with h5py.File(out/'natural_preparation.h5','x') as h:
        h.create_dataset('spins',data=paths.astype('f4'),compression='lzf')
        h.create_dataset('candidate_ids',data=ids)
        h.create_dataset('time_ps',data=np.arange(paths.shape[1])*cfg['numerics']['save_ps'])
        h.create_dataset('preprojection_error',data=err)
        h.attrs['protocol']='Independent perturbed +/-z parents; finite preparation, not equilibrium certified'
    d=descriptors(states); train=np.array([r['split']=='train' for r in rows])
    mean=d[train].mean(0); scale=d[train].std(0); scale[scale<1e-10]=1
    x=(d-mean)/scale; selected=[]; diagnostics=[]
    rng=np.random.default_rng(seedbase+20000)
    for split in ['train','dev','test']:
        for family in cfg['sampling']['families']:
            ix=np.array([r['id'] for r in rows if r['split']==split and r['family']==family])
            count=cfg['sampling']['selected_per_family'][split]
            pick=ix[fps(x[ix],count)] if split=='train' else rng.choice(ix,count,replace=False)
            selected.extend(pick.tolist())
            if split=='train':
                random_pick=rng.choice(ix,count,replace=False)
                for name,chosen in [('fps',pick),('random',random_pick)]:
                    distance=np.sqrt(((x[ix,None]-x[chosen][None])**2).sum(-1)).min(1)
                    diagnostics.append(dict(family=family,method=name,selected_ids=chosen.tolist(),
                                            mean_pool_distance=float(distance.mean()),max_pool_distance=float(distance.max())))
    fingerprints=[hashlib.sha256(s.tobytes()).hexdigest() for s in states]
    assert len(set(fingerprints))==len(states)
    for r,sha in zip(rows,fingerprints): r.update(state_sha256=sha,selected=r['id'] in selected)
    np.savez_compressed(out/'candidates.npz',spins=states,descriptors=d,train_mean=mean,train_scale=scale,selected_ids=selected)
    dump(out/'parents.json',rows); dump(out/'selection_diagnostics.json',diagnostics)
    return states,rows,selected


def generate(cfg,length,out,states,rows,selected):
    records=[]; windows=[]; seedbase=cfg['sampling']['seed']+length*100000
    for cid in selected:
        row=rows[cid]; split=row['split']; count=cfg['sampling']['branches'][split]
        initials=np.repeat(states[cid:cid+1],count,axis=0)
        arms=[('base',cfg['numerics']['dt_fs'])]
        if split==cfg['sampling']['fine_control_split']: arms.append(('fine',cfg['numerics']['fine_dt_fs']))
        for arm,dt in arms:
            seeds=seedbase+30000+cid*100+np.arange(count)+(50000 if arm=='fine' else 0)
            paths,err=simulate(initials,seeds,cfg,dt,cfg['numerics']['duration_ps'])
            name=f'{split}_parent{cid}_{arm}.h5'; target=out/name
            with h5py.File(target.with_suffix('.partial.h5'),'x') as h:
                h.attrs['complete']=False; h.attrs['parent_json']=json.dumps(row); h.attrs['config_json']=json.dumps(cfg)
                h.attrs['dt_fs']=dt; h.attrs['arm']=arm; h.attrs['distribution_certified']=False
                h.attrs['temperature_K']=cfg['physics']['theta']*cfg['physics']['exchange_meV']/.08617333262
                h.create_dataset('initial',data=states[cid]); h.create_dataset('seeds',data=seeds)
                h.create_dataset('time_ps',data=np.arange(paths.shape[1])*cfg['numerics']['save_ps'])
                h.create_dataset('spins',data=paths.astype('f4'),compression='lzf',chunks=(1,min(256,paths.shape[1]),length,3))
                mz=paths[...,2].mean(-1)
                energy=-(paths[:,:,:-1]*paths[:,:,1:]).sum((-1,-2))/length-cfg['physics']['anisotropy_over_J']*(paths[...,2]**2).mean(-1)
                h.create_dataset('mean_mz',data=mz); h.create_dataset('energy_per_spin_J',data=energy)
                h.create_dataset('preprojection_error',data=err); h.attrs['complete']=True
            target.with_suffix('.partial.h5').replace(target)
            records.append(dict(file=name,parent_id=row['parent_id'],split=split,family=row['family'],arm=arm,
                                branches=count,frames=paths.shape[1],dt_fs=dt,
                                sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
            for horizon in cfg['numerics']['diagnostic_windows_ps']:
                k=round(horizon/cfg['numerics']['save_ps']); end=paths[:,k]
                displacement=((end-initials)**2).sum(-1).mean(1)
                bond=(end[:,:-1]*end[:,1:]).sum(-1).mean(1)
                item=dict(parent_id=row['parent_id'],split=split,family=row['family'],arm=arm,time_ps=horizon,n=count,
                          mean_mz=float(mz[:,k].mean()),se_mz=float(mz[:,k].std(ddof=1)/np.sqrt(count)),
                          mean_energy=float(energy[:,k].mean()),mean_bond=float(bond.mean()),
                          mean_squared_displacement=float(displacement.mean()))
                if count>=16:
                    item['R1_R2_mz_difference']=float(mz[:8,k].mean()-mz[8:16,k].mean())
                else: item['R1_R2_mz_difference']=''
                windows.append(item)
            print(name,flush=True)
    dump(out/'manifest.json',records)
    with (out/'window_diagnostics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(windows[0])); w.writeheader(); w.writerows(windows)
    return records


def audit(out,records,cfg):
    seen=set(); parent_split={}; maximum=0.; total=0
    for r in records:
        assert parent_split.setdefault(r['parent_id'],r['split'])==r['split']
        with h5py.File(out/r['file']) as h:
            assert h.attrs['complete']
            seeds=h['seeds'][:].tolist(); assert not seen.intersection(seeds); seen.update(seeds)
            assert h['time_ps'][-1]==cfg['numerics']['duration_ps']
            for b in range(r['branches']):
                s=h['spins'][b]; assert np.isfinite(s).all()
                error=float(abs(np.linalg.norm(s,axis=-1)-1).max()); maximum=max(maximum,error)
                assert error<1e-6 and np.allclose(s[0],h['initial'][:],rtol=0,atol=1e-7)
            total+=r['branches']
    dump(out/'complete.json',dict(status='DATA_PILOT_COMPLETE_NOT_DISTRIBUTION_CERTIFIED',
          total_branches=total,parents=len(parent_split),max_float32_norm_error=maximum,
          finite=True,unique_future_seeds=True,parent_split_isolation=True,
          caveat='FPS diagnostic is only descriptor-pool coverage; it does not certify physical coverage or path fidelity.'))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--length',type=int,required=True); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); cfg=yaml.safe_load((BASE/'config.yaml').read_text())
    assert args.length in cfg['physics']['sizes']
    if args.smoke:
        cfg['numerics'].update(preparation_ps=.1,duration_ps=.2,diagnostic_windows_ps=[.1,.2])
        cfg['sampling'].update(candidates_per_family=dict(train=3,dev=1,test=1),
                              selected_per_family=dict(train=2,dev=1,test=1),branches=dict(train=2,dev=2,test=2))
    elif not os.getenv('SLURM_JOB_ID'): raise RuntimeError('formal pilot requires Slurm allocation')
    out=BASE/('smoke' if args.smoke else 'data')/f'L{args.length}'; out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); dump(out/'frozen_config.json',cfg)
    sources=[BASE/'run.py',BASE/'config.yaml',BASE.parent/'bauer_fig2_10ns_20260917/kernel.py']
    dump(out/'sources.json',{str(p.relative_to(BASE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    states,rows,selected=build(cfg,args.length,out)
    records=generate(cfg,args.length,out,states,rows,selected); audit(out,records,cfg)
    dump(out/'runtime.json',dict(seconds=time.monotonic()-started,job_id=os.getenv('SLURM_JOB_ID')))


if __name__=='__main__': main()
