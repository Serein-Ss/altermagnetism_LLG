"""Seen-parent unseen-noise audit and paired branch-coverage repair; FM only."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from numba import set_num_threads
from tests.bauer_p2_20260923.common import cfg, dump
from tests.bauer_active_data_20260918.run import simulate
from tests.bauer_comprehensive_20260923.run import real_data, next_batch, require_gpu
from tests.condition_time_repair.run import build, tasks as previous_tasks, expanded
from tests.curriculum_repair.run import fm_loss
from tests.reference_resolution.run import reference_inputs
from tests.moment_bias_repair.run import evaluate
from tests.cross_dimension_20260924.model import source
from tests.transport_audit.run import integrate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def parents():
    rows = json.loads((ROOT/'tests/bauer_p2_20260923/manifest_train.json').read_text())
    return [r for r in rows if r['L'] == 16 and r['family'] in ('tilted', 'interior_domain', 'single_wall')]


def tasks():
    rows = [dict(domain='bauer', arm='hierarchical', seed=s, repair=r, updates=6000)
            for s in (11, 22, 33) for r in ('repeat_old', 'expanded_branches')]
    np.random.default_rng(93401).shuffle(rows)
    return rows


def checkpoint(seed):
    i = next(i for i, t in enumerate(previous_tasks()) if t['seed'] == seed and t['repair'] == 'control')
    return ROOT/f'tests/condition_time_repair/train/{i:02d}/checkpoint.pt'


def branch_seeds(index, role, count):
    assert role in ('train', 'probe') and 0 <= index < 24 and 2 <= count <= 128
    return 950000000 + index*10000 + (1000 if role == 'probe' else 0) + np.arange(count, dtype=np.int64)


def prepare():
    assert not (HERE/'source_lock.json').exists(), 'Already frozen'
    rows = parents()
    assert len(rows) == 24 and all(sum(r['family'] == f for r in rows) == 8 for f in ('tilted','interior_domain','single_wall'))
    used = set()
    for split in ('train', 'dev'):
        for r in json.loads((ROOT/f'tests/bauer_p2_20260923/manifest_{split}.json').read_text()):
            with h5py.File(ROOT/r['path']) as h:
                used.update(map(int, h['seeds'][:]))
    for f in (ROOT/'tests/reference_resolution/reference').glob('*/paths.h5'):
        with h5py.File(f) as h:
            for role in ('base','fine'): used.update(map(int, h[f'{role}/seeds'][:]))
    fresh = set()
    for i, r in enumerate(rows):
        reference_inputs(r)
        for role, n in (('train',64),('probe',128)):
            seeds = set(map(int, branch_seeds(i,role,n)))
            assert not seeds & (used | fresh)
            fresh |= seeds
    dump(HERE/'protocol.json', dict(tasks=tasks(), parents=rows,
        question='Does eight-branch conditional coverage limit learned fluctuations, versus conditional model/optimization bias?',
        design='24 L16 training parents, all eight tilted/interior_domain/single_wall parents, selected by family not future outcome. Each 64 new training and 128 separately seeded diagnostic futures. No old manifest changed. Probe remains a TRAIN-parent conditional diagnostic, never independent-parent dev/test and never optimized.',
        physics='Original exact initial configurations and verified MT5.9 solver; J50meV,kappa0.1,alpha0.1,theta0.11,zero field,open. dt0.25fs,1ps duration,0.1ps saved grid. No temporal relabelling.',
        controls='Two arms x three paired seeds. Same control step24000 weights/Adam; 6000 additional FM-only cosine3e-4->3e-6 updates. Half batches original full data; half selected L16 initial windows. Same parent/mixture/quantile draws. Only pool differs: original8 versus original8+new64. No outcome selection or new architecture; both retain all original data.',
        evaluation='Before/after each24 training parent128 generated paired-latent Euler128/midpoint128. Compare original8 fixed empirical ddof0 and unseen128 population ddof1 separately. Independent-parent existing36dev64generated and expanded18dev256generated retained after training. Separate physical metrics, whole-branch bootstrap100; reused development evidence, not final certification.',
        interpretation='Improvement on new-noise train-parent probes but not dev indicates coverage/generalization issue; no improvement cannot alone prove architectural incapacity. Old8 fit but unseen128 bias indicates empirical overfit. Reference sampling and numerical error remain. No thresholds or automatic equivalence; no all-system claims.',
        budget='24 fat reference jobs estimated1-3min each;6 GPU continuations estimated20-40min each including sampling;fat report30-70min. Smoke estimated1-3min. No long-job monitoring.',
        skills='Karpathy surgical changes; experimental-design paired concurrent seed controls. Scientific Agent Skills https://doi.org/10.48550/arXiv.2609.00065'))
    lock = json.loads((ROOT/'tests/condition_time_repair/source_lock.json').read_text())
    for name, digest in lock.items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    files = list(HERE.glob('*.py')) + list(HERE.glob('*.sh')) + [HERE/'protocol.json']
    files += [checkpoint(s) for s in (11,22,33)] + [ROOT/r['path'] for r in rows]
    files += [ROOT/'tests/full_data_decay/metrics.py']
    for f in files: lock[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock)
    (HERE/'logs').mkdir(exist_ok=True)


def reference(index, smoke):
    set_num_threads(2 if smoke else 4)
    row = parents()[index]; initial, config = reference_inputs(row)
    config['numerics']['duration_ps'] = 1.
    out = HERE/('smoke_reference' if smoke else 'reference')/f'{index:02d}'
    out.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    for role, n in (('train',2 if smoke else 64), ('probe',4 if smoke else 128)):
        seeds=branch_seeds(index,role,n)
        x,err=simulate(np.repeat(initial[None],n,axis=0),seeds,config,.25,1.)
        with h5py.File(out/f'{role}.h5','x') as h:
            h.attrs.update(complete=False,role=role,split='train',parent=json.dumps(row),config_json=json.dumps(config),dt_fs=.25,duration_ps=1.)
            h.create_dataset('spins',data=x,compression='lzf');h.create_dataset('seeds',data=seeds)
            h.create_dataset('initial',data=initial);h.create_dataset('time_ps',data=np.arange(11)*.1)
            h.create_dataset('preprojection_error',data=err);h.attrs['complete']=True
    dump(out/'complete.json',dict(parent=row,seconds=time.perf_counter()-start,probe_role='seen-training-parent unseen-noise diagnostic only'))


def pools(smoke):
    """Only train.h5 is accessible here; probe.h5 is evaluation-only."""
    old=[];new=[];hashes={}
    for i,row in enumerate(parents()[:1] if smoke else parents()):
        with h5py.File(ROOT/row['path']) as h: a=h['spins'][:,:11]
        p=HERE/('smoke_reference' if smoke else 'reference')/f'{i:02d}/train.h5'
        with h5py.File(p) as h:
            assert h.attrs['complete'] and h.attrs['role']=='train' and h.attrs['split']=='train'
            assert json.loads(h.attrs['parent'])==row
            b=h['spins'][:].astype(np.float32)
            assert len(b)==(2 if smoke else 64) and np.max(abs(a[:1,0]-b[:,0]))<1e-7
            assert np.array_equal(h['seeds'][:],branch_seeds(i,'train',len(b)))
        old.append(a);new.append(np.concatenate([a,b]))
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return old,new,hashes


def targeted_batch(pool,rng,batch_size,device):
    # Draw uniforms independently of pool size so paired arms retain parent/RNG schedules.
    ids=rng.integers(len(pool),size=batch_size);u=rng.random(batch_size)
    x=np.stack([pool[i][min(int(v*len(pool[i])),len(pool[i])-1)] for i,v in zip(ids,u)])
    return F.normalize(torch.as_tensor(x,device=device),dim=-1),torch.arange(11,device=device)*.1,ids,u


@torch.no_grad()
def seen(net,out,smoke):
    with h5py.File(out/'seen.h5','x') as h:
        h.attrs['complete']=False
        for i,row in enumerate(parents()[:1] if smoke else parents()):
            initial,_=reference_inputs(row);n=4 if smoke else 128
            a=F.normalize(torch.tensor(initial,device='cuda',dtype=torch.float32),dim=-1)[None].expand(n,-1,-1)
            z=source(a,11,torch.Generator(device='cuda').manual_seed(934100+i),2.)
            t=torch.arange(11,device='cuda')*.1;g=h.create_group(f'parent{i:02d}');g.attrs['parent']=json.dumps(row)
            g.create_dataset('time_ps',data=np.arange(11)*.1)
            for method in ('euler','midpoint'):
                steps=8 if smoke else 128;chunks=[]
                for j in range(0,n,64):
                    init=a[j:j+64]
                    chunks.append(integrate(lambda x,tau:net.velocity(x,tau,init,t),z[j:j+64],0.,1.,steps,method))
                x=torch.cat(chunks)
                assert torch.isfinite(x).all() and (x.norm(dim=-1)-1).abs().max()<2e-6 and torch.equal(x[:,0],a)
                g.create_dataset(f'{method}{steps}',data=x.cpu().numpy(),compression='lzf')
        h.attrs['complete']=True


def train(index,smoke):
    require_gpu('cuda'); task=tasks()[index]
    base_task=dict(task,repair='control');net,opt,_=build(base_task)
    ck=torch.load(checkpoint(task['seed']),map_location='cpu',weights_only=True)
    assert ck['step']==24000 and ck['task']['repair']=='control'
    net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    mix=np.random.default_rng(934000+task['seed'])
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    old,new,hashes=pools(smoke);pool=old if task['repair']=='repeat_old' else new
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    before=out/'before';before.mkdir();net.eval();seen(net,before,smoke)
    config=cfg();data=real_data(config);steps=32 if smoke else task['updates'];history=[]
    schedule=hashlib.sha256();source_tau=hashlib.sha256();start=time.perf_counter();net.train()
    for step in range(1,steps+1):
        targeted=bool(mix.random()<.5);schedule.update(bytes([targeted]))
        if targeted:
            y,t,ids,u=targeted_batch(pool,mix,config['batch'],'cuda')
            schedule.update(ids.tobytes());schedule.update(u.tobytes())
        else:
            y,t=next_batch(task,config,data,rng,gs['target'],'cuda');schedule.update(y.cpu().numpy().tobytes())
        for k in ('source','tau'):source_tau.update(gs[k].get_state().numpy().tobytes())
        lr=3e-6+(.0003-3e-6)*.5*(1+math.cos(math.pi*(step-1)/max(1,steps-1)))
        for group in opt.param_groups:group['lr']=lr
        loss,_=fm_loss(net,y,t,gs);opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        assert torch.isfinite(loss) and torch.isfinite(grad)
        opt.step()
        if smoke or step%100==0:history.append(dict(step=24000+step,loss=float(loss.detach()),lr=lr))
        if step%500==0: print(task,step,flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=24000+steps,
        rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,mix_rng=mix.bit_generator.state,
        auxiliary_rng=ck['auxiliary_rng']),out/'checkpoint.pt')
    dump(out/'history.json',history);net.eval();seen(net,out,smoke)
    ev=evaluate(net,out,[],smoke);extra=expanded(net,out,smoke)
    dump(out/'complete.json',dict(task=task,train_seconds=elapsed,evaluation=ev,expanded=extra,
        data_hashes=hashes,schedule_hash=schedule.hexdigest(),source_tau_hash=source_tau.hexdigest(),
        device=torch.cuda.get_device_name(),scientific_certification=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','reference','smoke_reference','train','smoke']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();return
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    if 'reference' in a.action:reference(a.index,a.action.startswith('smoke'))
    else:train(a.index,a.action=='smoke')


if __name__=='__main__':main()
