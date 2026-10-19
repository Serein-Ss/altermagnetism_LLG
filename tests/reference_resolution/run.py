"""Frozen-checkpoint reference expansion and sampler validation; no retraining."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from numba import set_num_threads
from tests.bauer_active_data_20260918.run import simulate
from tests.bauer_p2_20260923.common import dump
from tests.bauer_comprehensive_20260923.run import require_gpu
from tests.conditional_refinement.run import tasks as old_tasks
from tests.spin_path_transformer.run import make_net
from tests.cross_dimension_20260924.model import source
from tests.transport_audit.run import integrate

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def parents():
    rows=json.loads((ROOT/'tests/bauer_p2_20260923/manifest_dev.json').read_text())
    return [r for r in rows if r['family'] in ('interior_domain','tilted','single_wall')]


def checkpoint(seed):
    i=next(i for i,r in enumerate(old_tasks()) if r['seed']==seed and r['repair']=='control')
    return ROOT/f'tests/conditional_refinement/train/{i:02d}/checkpoint.pt'


def gpu_tasks():
    return [dict(seed=s,L=n) for s in (11,22,33) for n in (16,25,32)]


def reference_inputs(row):
    with h5py.File(ROOT/row['path']) as h:
        assert h.attrs['complete'] and h.attrs['arm']=='base'
        initial=h['initial'][:];config=json.loads(h.attrs['config_json'])
        assert np.max(abs(initial-h['spins'][0,0]))<1e-7
        assert config['physics']['boundary']=='open'
        assert config['physics']['exchange_meV']==50 and config['physics']['theta']==.11
        assert config['physics']['anisotropy_over_J']==.1 and config['physics']['damping']==.1
        assert config['physics']['external_field']==0 and config['numerics']['save_ps']==.1
        assert config['numerics']['method']=='projected_Milstein_Tretyakov_5.9'
        return initial,config


def seeds_for(index,fine,count=128):
    return 900000000+index*10000+(1000 if fine else 0)+np.arange(count,dtype=np.int64)


def prepare():
    if (HERE/'source_lock.json').exists():raise RuntimeError('Already frozen')
    lock=json.loads((ROOT/'tests/conditional_refinement/source_lock.json').read_text())
    for name,digest in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    rows=parents();assert len(rows)==18
    used=set();fresh=set()
    for i,r in enumerate(rows):
        reference_inputs(r)
        for suffix in ('base','fine'):
            path=ROOT/r['path'].replace('_base.h5',f'_{suffix}.h5')
            with h5py.File(path) as h:used.update(map(int,h['seeds'][:]))
        for fine in (False,True):
            values=set(map(int,seeds_for(i,fine)));assert not fresh&values;fresh|=values
    assert not used&fresh
    dump(HERE/'protocol.json',dict(parents=rows,gpu_tasks=gpu_tasks(),
        scope='Bauer open, L16/25/32, T63.824849668K, J50meV,kappa0.1,alpha0.1,zero field. Existing prepared initial states,1ps,0.1ps output,11frames. No changed initial preparation or train split.',
        references='18 dev parents: both existing parents per size in tilted/interior_domain/single_wall. Each128 NEW MT5.9 base dt0.25fs and128 NEW fine dt0.125fs, independent RNG. Total4608 physical trajectories. Not Brownian-coupled timestep pairs. Preserve original16 paths; never add dev futures to training.',
        sampling='Freeze step18000 control checkpoints for seeds11/22/33, selected as unchanged baseline rather than retrospective best per metric. Per parent256 generated paths; same latent sample across Euler128/Euler512/projected-midpoint128/projected-midpoint256. Float32 network and sampler as existing deployed path. Midpoint reused verified candidate, not promoted.',
        evaluation='Independent model/reference whole-branch bootstrap100 with population ddof1. Compare original16 to new128 base, new128 base to new128 fine, model to BOTH new ensembles. Four disjoint16+16 subsets of base pool are descriptive noise controls, not independent parent replications or acceptance thresholds.',
        limitations='Budgeted precision diagnostic, no power guarantee. Reference-reference differences include sampling error. Reused parents are development, not final holdout. No automatic equivalence/pass; no all-parent/large-size claim. Flow steps do not change physical time.',
        decision='Only consider sampler replacement if paired latent path changes converge and physical metrics improve across seeds without tradeoffs. If model deviations persist against independent larger reference ensembles and numerical checks, next work targets learned conditional dynamics; do not continue optimizer-only expansion.',
        budget='18 fat reference tasks estimated2-8min each;9 rtx4090 GPU tasks estimated10-25min each;fat report10-30min. Submission only, no long monitoring.',
        skills='Karpathy isolated regression; experimental-design seed blocking and fixed-parent independent branches. Scientific Agent Skills https://doi.org/10.48550/arXiv.2609.00065'))
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json',ROOT/'tests/bauer_active_data_20260918/run.py',ROOT/'tests/bauer_fig2_10ns_20260917/kernel.py',ROOT/'tests/transport_audit/run.py']+[checkpoint(s) for s in (11,22,33)]
    files += [ROOT/r['path'] for r in rows]
    for f in files:lock[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


def reference(index,smoke=False):
    set_num_threads(4 if not smoke else 2);row=parents()[index];initial,config=reference_inputs(row)
    out=HERE/('smoke_reference' if smoke else 'reference')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    count=2 if smoke else 128
    tick=time.perf_counter()
    with h5py.File(out/'paths.h5','x') as h:
        h.attrs['complete']=False;h.attrs['parent']=json.dumps(row);h.attrs['config_json']=json.dumps(config)
        h.create_dataset('initial',data=initial);h.create_dataset('time_ps',data=np.arange(11)*.1)
        for fine in (False,True):
            seeds=seeds_for(index,fine,count);dt=.125 if fine else .25
            x,err=simulate(np.repeat(initial[None],count,axis=0),seeds,config,dt,1.)
            g=h.create_group('fine' if fine else 'base');g.attrs['dt_fs']=dt
            g.create_dataset('spins',data=x,compression='lzf');g.create_dataset('seeds',data=seeds);g.create_dataset('preprojection_error',data=err)
        h.attrs['complete']=True
    dump(out/'complete.json',dict(parent=row,branches_per_dt=count,seconds=time.perf_counter()-tick))


@torch.no_grad()
def generate(index,smoke=False):
    require_gpu('cuda');task=gpu_tasks()[index]
    ck=torch.load(checkpoint(task['seed']),map_location='cpu',weights_only=True)
    assert ck['step']==18000 and ck['task']['repair']=='control'
    net=make_net(ck['task']);net.load_state_dict(ck['model']);net.eval()
    out=HERE/('smoke_gpu' if smoke else 'gpu')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    rows=[(i,r) for i,r in enumerate(parents()) if r['L']==task['L']]
    if smoke:rows=rows[:1]
    records=[];tick=time.perf_counter()
    with h5py.File(out/'paths.h5','x') as h:
        h.attrs['complete']=False;h.attrs['task']=json.dumps(task)
        for i,row in rows:
            initial,_=reference_inputs(row);count=4 if smoke else 256
            init=F.normalize(torch.tensor(initial,dtype=torch.float32,device='cuda'),dim=-1)[None].expand(count,-1,-1)
            z=source(init,11,torch.Generator(device='cuda').manual_seed(932100+i),2.)
            times=torch.arange(11,device='cuda')*.1;group=h.create_group(f'parent{i:02d}');group.attrs['parent']=json.dumps(row)
            group.create_dataset('time_ps',data=np.arange(11)*.1)
            settings=[('euler',8),('midpoint',8)] if smoke else [('euler',128),('euler',512),('midpoint',128),('midpoint',256)]
            for method,steps in settings:
                torch.cuda.synchronize();start=time.perf_counter();chunks=[]
                for j in range(0,count,64):
                    a=init[j:j+64]
                    chunks.append(integrate(lambda x,t:net.velocity(x,t,a,times),z[j:j+64],0.,1.,steps,method))
                x=torch.cat(chunks);torch.cuda.synchronize();seconds=time.perf_counter()-start
                assert torch.isfinite(x).all() and (x.norm(dim=-1)-1).abs().max()<2e-6
                assert torch.equal(x[:,0],init)
                group.create_dataset(f'{method}{steps}',data=x.cpu().numpy(),compression='lzf')
                records.append(dict(parent_index=i,method=method,steps=steps,nfe=steps*(2 if method=='midpoint' else 1),seconds=seconds))
        h.attrs['complete']=True
    dump(out/'complete.json',dict(task=task,seconds=time.perf_counter()-tick,records=records,scientific_certification=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','reference','gpu','smoke_reference','smoke_gpu']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();return
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    if 'reference' in a.action:reference(a.index,a.action.startswith('smoke'))
    else:generate(a.index,a.action.startswith('smoke'))


if __name__=='__main__':main()
