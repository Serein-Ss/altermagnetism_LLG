"""Frozen gradient audit, then paired direction versus magnitude repair."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_p2_20260923.common import cfg, dump
from tests.bauer_comprehensive_20260923.run import require_gpu, real_data, next_batch
from tests.branch_coverage_repair.run import pools, parents as train_parents
from tests.sampling_balance_repair.run import RefillWindows
from tests.temporal_source_repair.source import draw
from tests.transition_block_repair.model import TransitionNet, pair_window
from tests.transition_block_repair.run import evaluate
from tests.conditional_moment_repair.run import intermediate
from tests.training_restart_repair.run import seen_restart
from tests.training_restart_repair.data import load_train, HERE as DATA
from tests.balanced_replay_repair.run import verify as verify_sources
from .gradients import merge, clone_grad

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ('sum', 'project', 'rescale')


def tasks():
    return [dict(domain='bauer', arm='hierarchical', seed=s, repair=r, updates=6000)
            for s in (11, 22, 33) for r in ARMS]


def source(seed):
    return ROOT / f'tests/balanced_replay_repair/train/{(11,22,33).index(seed)*3:02d}'


def selection(arm, step, rng, count=48):
    if arm not in ARMS:
        raise ValueError(arm)
    return rng.permutation(count), rng.integers(64, size=count), 1.


def prepare():
    assert not (HERE / 'source_lock.json').exists()
    verify_sources()
    assert json.loads((DATA / 'data_audit.json').read_text())['passed']
    _, hashes = load_train()
    protocol = dict(tasks=tasks(),
        question='Do native and restart FM gradients conflict, and does removing the conflicting component improve independent dynamics rather than only changing effective weight?',
        source='Each seed starts latest balanced_replay sparse_iid step24000 with retained Adam and native RNG. No architecture, data, physics, time-grid or reference changes.',
        audit='Freeze all weights;128paired TRAIN minibatches for each3seed. Native window batch8 plus all48restartstates one random TRAIN branch each. Report gradient dot,cosine,norm,projection fraction and successive same-objective batch gradient cosines as stochastic-gradient baseline. Negative dot alone does not establish harmful physical interference.',
        arms='6000updates at cosine3e-5->3e-6. Every arm same native batch and dense balanced restart batch. sum=g_native+g_restart; project=g_native+g_restart-min(dot,0)/norm_native_squared*g_native; rescale=g_native+ratio*g_restart,ratio=norm(projected_restart)/norm(restart). Equal-norm sham distinguishes direction from magnitude at a given gradient pair; after models diverge their ratios need not match.',
        limitations='One-sided Euclidean projection on raw parameter gradients, not a proof of descent under Adam,momentum,clipping or finite steps. Record actual update dot native/restart gradient. No guarantee that improved FM losses repair conditional physics. Do not describe projection as a new magnetic architecture or automatically activate production.',
        pairing='Restore data/refill/pair/source/tau/cut streams, common independent extra RNG. All3arms must have identical native schedule/input hashes and extra generator hashes. Same6000updates and batches, measured compute. Native supervision remains present every update. Probe/fine/dev never train.',
        physics='Bauer open chain,J50meV,kappa.1,alpha.1,theta.11,T63.824849668K; lag.1ps and1ps10step composition. Midpoint128/512 same physical grid and source. ActualS0anchored,unitnorm. No new LLG data.',
        evaluation='Reuse full frozen evaluator:48TRAIN-origin states64generated/probe128/fine128,18dev128paths1ps midpoint128/512,24oldseen64paths,72dev intermediate128futures. Before source after_restart reused with matched latent seeds. Fixed final30000 checkpoint. Means,covariance,variance,energy,spatial and temporal statistics,family,size and time strata; save all generated trajectories.',
        uncertainty='Fixed-parent whole-branch bootstrap100;both sample sets vary. Cross-state averages descriptive;24sourcegroups underlying48states are correlated. Three seed variation separately. Dev repeatedly reused,not finalholdout. No invented equivalence band or certification.',
        decisions='Audit first, then run controlled repair if conflicts occur consistently across seeds; this is a diagnostic selection, not a physical pass/fail threshold. If projection equals rescaling gains, direction mechanism unsupported. If loss improves without independent path improvement, optimization conflict is not sufficient explanation.',
        budget='Audit3GPUjobs estimated1-3min;smoke3GPUjobs1-3min;formal9rtx4090jobs2-3h each,9fat stats3-6min,web<3min. After formal submission no monitoring.',
        skills='karpathy-guidelines isolated tests-only edits;experimental-design paired controls and sham comparison. No automatic production promotion or remote push.')
    dump(HERE / 'protocol.json', protocol)
    files = list(HERE.glob('*.py')) + list(HERE.glob('*.sh')) + [HERE/'protocol.json', DATA/'source_lock.json', DATA/'data_audit.json']
    for seed in (11,22,33):
        files += [source(seed)/'checkpoint.pt', source(seed)/'complete.json', source(seed)/'after_restart.h5']
    lock = dict(hashes)
    for p in files:
        lock[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json', lock)
    (HERE/'logs').mkdir(exist_ok=True)


def verify():
    verify_sources()
    for name, digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name


def train(index, smoke, audit_only=False):
    require_gpu('cuda'); task=tasks()[index]; seed=task['seed']
    ck=torch.load(source(seed)/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert ck['step']==24000 and ck['task']['seed']==seed and ck['task']['repair']=='sparse_iid'
    net=TransitionNet().cuda();net.load_state_dict(ck['model'])
    opt=torch.optim.AdamW(net.parameters(),lr=3e-5,weight_decay=1e-5);opt.load_state_dict(ck['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    refill=np.random.default_rng();refill.bit_generator.state=ck['refill_rng']
    pair_rng=np.random.default_rng();pair_rng.bit_generator.state=ck['pair_rng']
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    extra_rng=np.random.default_rng(947000+seed)
    extra_g=torch.Generator(device='cuda').manual_seed(947100+seed)
    bank,bank_hashes=load_train(False);assert len(bank)==48
    _,new,hashes=pools(False);pool={r['parent_id']:x for r,x in zip(train_parents(),new)}
    audit=dict(counts=Counter(),hash=hashlib.sha256());config=cfg()
    data=[RefillWindows(d,pool,refill,True,audit) for d in real_data(config)]
    out=HERE/('audit' if audit_only else 'smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    total=128 if audit_only else 16 if smoke else 6000;history=[];stream=hashlib.sha256();noise_hash=hashlib.sha256()
    exposures=np.zeros((48,64),dtype=np.int64);tick=time.perf_counter();net.train()
    parameters=list(net.parameters());previous_native=None;previous_replay=None
    initial_model_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in parameters)).hexdigest()
    for step in range(1,total+1):
        whole,_=next_batch(task,config,data,rng,gs['target'],'cuda')
        stream.update(whole.detach().cpu().numpy().tobytes())
        for k in ('source','tau'):stream.update(gs[k].get_state().numpy().tobytes())
        starts=np.where(pair_rng.random(len(whole))<.5,0,pair_rng.integers(1,10,len(whole)))
        y=pair_window(whole,torch.as_tensor(starts,device='cuda'))
        z=draw(y[:,0],11,gs['source'],0.)[:,:2];tau=torch.rand(len(y),device='cuda',generator=gs['tau'])
        cut_state=gs['cut'].get_state();torch.randn(whole.shape,device='cuda',generator=gs['cut'])
        temp=torch.Generator(device='cuda').set_state(cut_state);x,v,_=path_interpolate(z,y,tau,temp)
        pred=net.velocity(x,tau,y[:,0],y.new_tensor([0.,.1]));fm=(pred[:,1:]-v[:,1:]).square().sum(-1).mean()
        opt.zero_grad(set_to_none=True);fm.backward();native=clone_grad(parameters)
        opt.zero_grad(set_to_none=True);record=dict(step=step,fm=float(fm.detach()))
        ids,branches,weight=selection(task['repair'],step,extra_rng)
        if len(ids):
            np.add.at(exposures,(ids,branches),1);noise_hash.update(extra_g.get_state().numpy().tobytes())
            yr=F.normalize(torch.tensor(np.stack([bank[i][j] for i,j in zip(ids,branches)]),device='cuda'),dim=-1)
            zr=draw(yr[:,0],2,extra_g,0.);tr=torch.rand(len(yr),device='cuda',generator=extra_g)
            xr,vr,_=path_interpolate(zr,yr,tr,extra_g)
            pr=net.velocity(xr,tr,yr[:,0],yr.new_tensor([0.,.1]));extra=(pr[:,1:]-vr[:,1:]).square().sum(-1).mean()
            (weight*extra).backward();assert torch.isfinite(extra);record.update(restart_fm=float(extra.detach()),weight=weight)
        replay=clone_grad(parameters);combined,stats=merge(native,replay,task['repair']);record.update(stats)
        if audit_only:
            if previous_native is not None:
                record['native_same_objective_cosine']=merge(previous_native,native,'sum')[1]['cosine']
                record['replay_same_objective_cosine']=merge(previous_replay,replay,'sum')[1]['cosine']
            previous_native=native;previous_replay=replay
        for parameter,gradient in zip(parameters,combined):parameter.grad=gradient
        lr=3e-6+(3e-5-3e-6)*.5*(1+math.cos(math.pi*(step-1)/max(1,total-1)))
        for group in opt.param_groups:group['lr']=lr
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(fm) and torch.isfinite(grad)
        if not audit_only:
            measured=smoke or step%100==0
            before=[p.detach().clone() for p in parameters] if measured else None
            opt.step()
            if measured:
                delta=[p.detach()-old for p,old in zip(parameters,before)]
                record['actual_update_dot_native']=float(sum((d*g).sum() for d,g in zip(delta,native) if g is not None))
                record['actual_update_dot_replay']=float(sum((d*g).sum() for d,g in zip(delta,replay) if g is not None))
        record.update(gradient=float(grad),lr=lr)
        if audit_only or smoke or step%100==0:history.append(record)
        if step%500==0:print(task,step,flush=True)
    torch.cuda.synchronize();training=time.perf_counter()-tick
    assert exposures.sum()==total*48 and (exposures.sum(1)==total).all()
    if audit_only:
        final_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in parameters)).hexdigest()
        assert final_hash==initial_model_hash
        dump(out/'history.json',history)
        dump(out/'complete.json',dict(task=task,batches=total,frozen_weights=True,model_hash=final_hash,seconds=training,scientific_certification=False))
        return
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=24000+total,
        rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state,
        pair_rng=pair_rng.bit_generator.state,extra_rng=extra_rng.bit_generator.state,extra_generator=extra_g.get_state()),out/'checkpoint.pt')
    dump(out/'history.json',history);dump(out/'exposures.json',exposures.tolist())
    net.eval();tick=time.perf_counter();seen_restart(net,out/'after_restart.h5',smoke)
    ev=evaluate(net,out,smoke);intermediate(net,out,smoke);torch.cuda.synchronize()
    dump(out/'complete.json',dict(task=task,step=24000+total,train_seconds=training,evaluation_seconds=time.perf_counter()-tick,
        paired_input_hash=stream.hexdigest(),extra_noise_hash=noise_hash.hexdigest(),schedule_hash=audit['hash'].hexdigest(),
        data_hashes=hashes,restart_data_hashes=bank_hashes,evaluation=ev,device=torch.cuda.get_device_name(),scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','audit','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:verify();train(a.index,a.action=='smoke',a.action=='audit')
