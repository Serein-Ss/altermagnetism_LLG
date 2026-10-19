"""Conditional coverage and gradient-noise controls, without changing physics."""
import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_p2_20260923.common import cfg, dump
from tests.bauer_comprehensive_20260923.run import real_data, references, require_gpu
from tests.curriculum_repair.run import fm_loss
from tests.cross_dimension_20260924.model import source
from tests.full_data_decay.run import tasks as previous_tasks
from tests.moment_bias_repair.run import evaluate
from tests.spin_path_transformer.run import make_net

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def tasks():
    rows = [dict(domain='bauer', arm='hierarchical', seed=s, repair=a, updates=6000)
            for s in (11,22,33) for a in ('control','anchor90','accum4')]
    np.random.default_rng(93101).shuffle(rows)
    return rows


def checkpoint(seed):
    i = next(i for i,t in enumerate(previous_tasks()) if t['seed']==seed and t['repair']=='decay')
    return ROOT/f'tests/full_data_decay/train/{i:02d}/checkpoint.pt'


def indices(datasets, rng, count, anchor_probability):
    if not 0 <= anchor_probability <= 1: raise ValueError('anchor probability')
    ds = datasets[int(rng.integers(len(datasets)))]; selected = []
    for _ in range(count):
        i = int(rng.integers(len(ds)))
        if rng.random() < anchor_probability:
            g = int(np.searchsorted(ds.ends, i, side='right'))
            start = ds.ends[g-1] if g else 0
            nstarts = len(ds.records[g][2])
            i = start+((i-start)//nstarts)*nstarts
        selected.append(i)
    return ds, selected


def get_batch(data, rng, config, probability, device):
    ds, ids = indices(data, rng, config['batch'], probability)
    samples = [ds[i] for i in ids]
    y = F.normalize(torch.tensor(np.stack([s['spins'] for s in samples]), device=device), dim=-1)
    times = torch.arange(y.shape[1], device=device)*config['save_ps']
    return y, times, samples


def lr_at(step, total):
    return 3e-6+(1e-4-3e-6)*.5*(1+math.cos(math.pi*(step-1)/max(1,total-1)))


@torch.no_grad()
def probes(net, smoke=False):
    """Paired conditional-FM residuals; not errors against the unknown exact velocity."""
    result = []; net.eval(); times = torch.arange(11, device='cuda')*.1
    for split in ('train','dev'):
        for j,(row,raw) in enumerate(references('bauer', split)):
            if smoke and j: break
            y = F.normalize(torch.tensor(raw, device='cuda'), dim=-1)
            for repeat in range(1 if smoke else 2):
                z = source(y[:,0], 11, torch.Generator(device='cuda').manual_seed(931100+j*10+repeat), 2.)
                for tau in ((.25,) if smoke else (.02,.1,.25,.5,.9,.98)):
                    t = y.new_full((len(y),), tau)
                    x,v,_ = path_interpolate(z,y,t,torch.Generator(device='cuda').manual_seed(931900+repeat))
                    pred = net.velocity(x,t,y[:,0],times)
                    err = (pred[:,1:]-v[:,1:]).square().sum(-1).mean((0,2))
                    result.append(dict(split=split,parent=row,repeat=repeat,tau=tau,
                        mse_by_physical_frame=err.cpu().tolist(),target_speed2=float(v[:,1:].square().sum(-1).mean())))
    return result


def audit(data):
    rows=[]
    for ds in data:
        for path,row,starts,steps in ds.records:
            rows.append(dict(parent_id=row['parent_id'],family=row['family'],split=row['split'],
                sites=ds.cache[path][0].shape[2],branches=ds.cache[path][0].shape[0],
                windows_per_branch=len(starts),frame_steps=steps,
                baseline_anchor_probability=.5+.5/len(starts),
                treatment_anchor_probability=.9+.1/len(starts)))
    train_ids={r['parent_id'] for r in rows}
    dev_ids={r['parent_id'] for r,_ in references('bauer','dev')}
    assert not train_ids & dev_ids
    return dict(parents=rows,split_disjoint=True,
        counts=dict(Counter(f"L{r['sites']}:{r['family']}" for r in rows)),
        caveat='Counts are exposure/availability, not proof of local-state coverage. No new independent futures, no outcome selection. Anchor mix changes preparation weighting, not natural occupancy.')


def prepare():
    if (HERE/'source_lock.json').exists(): raise RuntimeError('Already frozen')
    lock=json.loads((ROOT/'tests/full_data_decay/source_lock.json').read_text())
    for name,digest in lock.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    dump(HERE/'data_audit.json', audit(real_data(cfg())))
    dump(HERE/'protocol.json',dict(tasks=tasks(),
        scope='Bauer open L16/25/32, 63.824849668K; same1ps/.1ps grid; existing data/splits, Hamiltonian, backbone, source and FM objective unchanged.',
        start='Each seed starts its full_data_decay cosine step12000 checkpoint; Adam/RNG restored. 6000 updates at cosine1e-4 to3e-6 in every arm. Fixed final step18000 primary; not a claim that more budget is sufficient.',
        control='Legacy .5 anchor override plus .5 uniform windows, batch8, one microbatch per optimizer step.',
        anchor90='.9 anchor override plus .1 uniform windows; same RNG draws/branch selections as control. All families retained; no future-outcome filtering. Effective anchor mass is p+(1-p)/number_of_windows.',
        accum4='Same baseline data distribution, average four batch8 gradients before clipping/Adam. Four times training exposure/compute, NOT equal-compute comparison and not pure noise-only identification. Microbatches may use different lattice sizes.',
        pairing='Three seed blocks. Control/anchor90 same base index and branch draws; windows intentionally differ. Control/accum4 share first microbatch stream prefix, but per-update streams differ. Log first32microbatch target/source/tau hashes and actual family/origin exposure.',
        diagnostics='Before/after paired CFM residual per train/dev parent, six transport times and physical frame. Finite empirical stochastic targets contain irreducible target variance: not exact population velocity error. Train18 calibration vsdev36; two probe latent draws.',
        evaluation='Final checkpoint64 paths per parent,128flow plus first train/dev32/256; population variance ddof1, whole-branch bootstrap100. Reused development data; no untouched final test, threshold or certification. Save every generated path. Family/size and temporal-sign diagnostics, not just global means.',
        budget='9GPU jobs estimated control/anchor8-15min,accum4 20-35min; report5-15min; short3arm smoke1-3min. Formal tasks left unmonitored.',
        limitations='Parent/window exposure versus gradient averaging are candidate mechanisms, not established root causes. No claim of independent late-window conditional ensembles, full state coverage, size transfer or universal architecture success.',
        skills='Karpathy isolated code/tests; experimental-design concurrent seed-blocked controls with explicit compute confounding; Scientific Agent Skills https://doi.org/10.48550/arXiv.2609.00065'))
    for f in list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json',HERE/'data_audit.json']+[checkpoint(s) for s in (11,22,33)]:
        lock[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock); (HERE/'logs').mkdir(exist_ok=True)


def train(task,out,smoke):
    ck=torch.load(checkpoint(task['seed']),map_location='cpu',weights_only=True)
    assert ck['step']==12000 and ck['task']['seed']==task['seed'] and ck['task']['repair']=='decay'
    net=make_net(task);net.load_state_dict(ck['model'])
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-5);opt.load_state_dict(ck['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    data=real_data(cfg());config=cfg();steps=32 if smoke else task['updates']
    accumulation=4 if task['repair']=='accum4' else 1
    probability=.9 if task['repair']=='anchor90' else .5
    before=probes(net,smoke);net.train();history=[];exposure=Counter();stream=hashlib.sha256();micro=0
    tick=time.perf_counter()
    for step in range(1,steps+1):
        lr=lr_at(step,steps)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);loss_value=0.
        for _ in range(accumulation):
            y,times,samples=get_batch(data,rng,config,probability,'cuda');tau_state=gs['tau'].get_state()
            loss,z=fm_loss(net,y,times,gs)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            (loss/accumulation).backward();loss_value+=float(loss.detach())/accumulation
            if micro<32:
                for v in (y,z,tau_state):stream.update(v.detach().cpu().numpy().tobytes())
            micro+=1
            for sample in samples:exposure[f"{sample['family']}:{'anchor' if sample['origin_ps']==0 else 'later'}"]+=1
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
        opt.step()
        if step%100==0 or smoke:history.append(dict(step=12000+step,loss=loss_value,gradient=float(grad),lr=lr))
        if step%500==0:print(task,step,flush=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-tick
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=12000+steps,
        rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,auxiliary_rng=ck['auxiliary_rng']),out/'checkpoint.pt')
    dump(out/'history.json',history);after=probes(net,smoke)
    dump(out/'transport_probes.json',dict(before=before,after=after))
    net.eval();ev=evaluate(net,out,[],smoke)
    return dict(task=task,train_seconds=seconds,microbatches=micro,exposure=dict(exposure),
        prefix_microbatches=min(32,micro),prefix_data_source_tau_hash=stream.hexdigest(),
        evaluation=ev,device=torch.cuda.get_device_name(),scientific_certification=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();return
    require_gpu('cuda')
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    out=HERE/f'{a.action}/{a.index:02d}';out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    r=train(tasks()[a.index],out,a.action=='smoke');r['total_seconds']=time.perf_counter()-tick
    dump(out/'complete.json',r)


if __name__=='__main__':main()
