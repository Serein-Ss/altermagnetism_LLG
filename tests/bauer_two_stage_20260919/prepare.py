"""Independent declared-protocol initials and held-out futures; no outcome selection."""
import argparse
import json
import time
import h5py
import numpy as np
from tests.bauer_active_data_20260918.run import simulate
from tests.bauer_closed_loop_20260919.run import records, reference
from tests.bauer_closed_loop_20260919.metrics import compare, summary
from .common import HERE, cfg, dump, allocation, old_cfg, seed_stream


def run(length):
    allocation();c=cfg();out=HERE/'data'/f'L{length}'
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    counts=c['initial_counts'] if length in c['train_sizes'] else {'test':64}
    manifest=[]
    for split,count in counts.items():
        offset={'train':1,'dev':2,'test':3}[split]
        seeds=np.array([seed_stream(910000+length*1000+offset,j) for j in range(count)],dtype='i8')
        initial=np.zeros((count,length,3));initial[...,2]=1.
        # Save only initial/final preparation; exact same integrator and physical duration.
        prep=dict(c,numerics=dict(c['numerics'],save_ps=c['numerics']['preparation_ps']))
        paths,error=simulate(initial,seeds,prep,c['numerics']['dt_fs'],c['numerics']['preparation_ps'])
        s=paths[:,-1]
        with h5py.File(out/f'{split}_initial.h5','x') as h:
            h.create_dataset('spins',data=s);h.create_dataset('seeds',data=seeds)
            h.attrs['protocol']=c['initial_protocol'];h.attrs['complete']=True
            h.attrs['config_json']=json.dumps(c);h.attrs['time_unit']='ps';h.attrs['preparation_ps']=c['numerics']['preparation_ps']
        manifest.extend(dict(id=f'L{length}_{split}_{j}',split=split,seed=int(seed),file=f'{split}_initial.h5',row=j)
                        for j,seed in enumerate(seeds))
        if split=='test':
            # Two independent groups of 16 per parent, each 100 ps. Never used to select model.
            for j in range(c['evaluation_parents']):
                init=np.repeat(s[j:j+1],2*c['branches'],0)
                future_seeds=np.array([seed_stream(920000+length*100+j,k) for k in range(len(init))],dtype='i8')
                x,err=simulate(init,future_seeds,c,c['numerics']['dt_fs'],max(c['reference_horizons_ps']))
                with h5py.File(out/f'test_future_{j}.h5','x') as h:
                    h.create_dataset('spins',data=x.astype('f4'),compression='lzf')
                    h.create_dataset('initial',data=s[j]);h.create_dataset('seeds',data=future_seeds)
                    h.create_dataset('time_ps',data=np.arange(x.shape[1])*c['numerics']['save_ps'])
                    h.attrs['config_json']=json.dumps(c);h.attrs['time_unit']='ps'
                    h.attrs['complete']=True
                print(length,'test_future',j,flush=True)
    dump(out/'manifest.json',manifest)
    if length in c['train_sizes']:
        for j,row in enumerate(r for r in records(old_cfg(),'dev') if r['L']==length):
            truth,times,initial=reference(row,old_cfg())
            seeds=np.array([seed_stream(930000+length*100+j,k) for k in range(c['calibration_branches'])],dtype='i8')
            x,err=simulate(np.repeat(initial[None],len(seeds),0),seeds,c,c['numerics']['dt_fs'],1.)
            np.savez_compressed(out/f'calibration_{row["parent_id"]}.npz',reference=x.astype('f4'),seeds=seeds)
    dump(out/'complete.json',dict(complete=True,length=length,counts=counts,seconds=time.perf_counter()-started,
                                protocol=c['initial_protocol'],distribution_certified=False))


def calibrate():
    c=cfg();oc=old_cfg();rows=[]
    for r in records(oc,'dev'):
        truth,_,_=reference(r,oc)
        with np.load(HERE/'data'/f'L{r["L"]}'/f'calibration_{r["parent_id"]}.npz') as z:x=z['reference'].astype('f8')
        scores=[compare(x[k:k+16],truth) for k in range(0,64,16)]
        item=dict(parent_id=r['parent_id'],L=r['L'],family=r['family'])
        item.update({key:float(np.mean([v[key] for v in scores])) for key in oc['margins']})
        rows.append(item)
    dump(HERE/'calibration.json',dict(parents=rows,statistics=summary(rows,oc['margins']),
         comparison='Independent new 16 versus existing 16, averaged over four new groups; reference group reused',
         gate='Raw tolerances retained; paired excess screening is separately reported, not physical equivalence',
         initial_protocol=c['initial_protocol']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--length',type=int);p.add_argument('--calibrate',action='store_true');a=p.parse_args()
    if a.calibrate:calibrate()
    else:run(a.length)
