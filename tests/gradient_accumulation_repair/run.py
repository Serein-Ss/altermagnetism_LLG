"""Gradient accumulation with matched-exposure and matched-update controls."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ('batch1', 'acc8_equal_data', 'acc8_equal_updates')


def tasks():
    return [dict(domain='bauer',arm='hierarchical',seed=seed,repair=arm,
                 accumulation=1 if arm=='batch1' else 8,
                 updates=750 if arm=='acc8_equal_data' else 6000)
            for seed in (11,22,33) for arm in ARMS]


def budget(task,smoke=False):
    updates=(2 if task['repair']=='acc8_equal_data' else 16) if smoke else task['updates']
    return updates,task['accumulation'],updates*task['accumulation']


def backward_mean(loss, accumulation):
    (loss/accumulation).backward()


def source(seed):
    return ROOT / f'tests/balanced_replay_repair/train/{(11,22,33).index(seed)*3:02d}'


def selection(arm, step, rng, count=48):
    if arm not in ARMS:raise ValueError(arm)
    return rng.permutation(count),rng.integers(64,size=count),1.


def prepare():
    assert not (HERE / 'source_lock.json').exists()
    verify_sources()
    assert json.loads((DATA / 'data_audit.json').read_text())['passed']
    _, hashes = load_train()
    protocol = dict(tasks=tasks(),
        question='Does reducing minibatch gradient noise improve independent conditional dynamics, beyond only increasing data exposure?',
        evidence='Frozen8192batch audit: mean native/restart gradient cosine=.021702,.186808,.037804; persistent opposition not supported at these checkpoints. Native single-batch RMS much larger than mean gradient. Noise limitation is a hypothesis, not proven cause of path bias; do not activate projection.',
        controls='Start perseed balanced_replay sparse_iid step24000 weights AND Adam. Every microbatch original native8 plus balanced48restart weight1. batch1:6000updates/6000microbatches; acc8_equal_data:750updates/6000microbatches; acc8_equal_updates:6000updates/48000microbatches. Divide BOTH FM terms by8 before backward; zero once per group,clip10 once after accumulation,Adam step once pergroup. Different lattice sizes averaged as equal micro-objectives,not concatenated into a single size.',
        optimization='Cosine3e-5->3e-6 over each arms optimizer update count. Same-exposure arm has fewer Adam steps,decay applications and altered moment horizon; cannot isolate noise from all optimizer effects. Same-update arm has8x microbatch exposure/compute. No linear LR scaling or extra moment/energy loss. Same learned2frame.1ps package; unchanged network.',
        pairing='Restore native data/refill/pair/source/tau/cut states; shared extra seed. Common first6000microbatches checked using native,extra-input,schedule hashes; whole-run hashes differ intentionally for48000arm. Track actual update count and perstate exposures. Retained optimizer state verified at source; checkpoint contains update/microstep counters and all RNG.',
        data='Reuse existing train-only64futures per48TRAIN-origin states from24parents plus native windows and72branch refill. No probe/fine/dev samples in loss. No new LLG simulation,split mutation or physical parameter changes.',
        evaluation='Fixed final checkpoint.48train-origin states64generated compared with independent128probe/fine;18dev128full1ps paths midpoint128/512;24oldseen64paths;72trueLLGdev intermediate128futures. Before values from sourceafter_restart with matching latent seeds. Record all paths and independent physical means,covariance,variance,energy,space/time metrics,family and size strata.',
        uncertainty='Whole-branch bootstrap100 within fixedparent,model and reference sampling both included. Cross-state averages descriptive;48states are correlated by24sources. Reuseddev not finalholdout. Three seeds separately. No arbitrary pass thresholds or aggregate score.',
        decisions='Equal-data improvement supports usefulness under fixed exposure but fewer optimizer updates remain part of intervention. Improvement only with8xdata requires explicit compute disclosure. Lower FM or smoother optimization without independent path improvement is not repair. Keep oldanchor and1ps regressions visible. No production promotion.',
        physics='Bauer open,J50meV,kappa.1,alpha.1,theta.11,T63.824849668K;reference dt.25/.125fs,learnedlag.1ps,1ps10freshnoise transitions. No time relabeling.',
        budget='Smoke3GPU jobs1-4min.9formalrtx4090 jobs2-4h each including full evaluation;9fatstats3-6min;web<3min. Submit and stop monitoring long tasks.',
        skills='karpathy-guidelines surgical isolation;experimental-design paired exposure/update controls;matplotlib PNG only.')
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


def train(index, smoke):
    require_gpu('cuda'); task=tasks()[index]; seed=task['seed']
    ck=torch.load(source(seed)/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert ck['step']==24000 and ck['task']['seed']==seed and ck['task']['repair']=='sparse_iid'
    net=TransitionNet().cuda();net.load_state_dict(ck['model'])
    opt=torch.optim.AdamW(net.parameters(),lr=3e-5,weight_decay=1e-5);opt.load_state_dict(ck['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    refill=np.random.default_rng();refill.bit_generator.state=ck['refill_rng']
    pair_rng=np.random.default_rng();pair_rng.bit_generator.state=ck['pair_rng']
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    extra_rng=np.random.default_rng(948000+seed)
    extra_g=torch.Generator(device='cuda').manual_seed(948100+seed)
    bank,bank_hashes=load_train(False);assert len(bank)==48
    _,new,hashes=pools(False);pool={r['parent_id']:x for r,x in zip(train_parents(),new)}
    audit=dict(counts=Counter(),hash=hashlib.sha256());config=cfg()
    data=[RefillWindows(d,pool,refill,True,audit) for d in real_data(config)]
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    updates,accumulation,total=budget(task,smoke);history=[];stream=hashlib.sha256();noise_hash=hashlib.sha256()
    prefix_steps=16 if smoke else 6000;prefix={};actual_updates=0
    exposures=np.zeros((48,64),dtype=np.int64);tick=time.perf_counter();net.train()
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
        if (step-1)%accumulation==0:opt.zero_grad(set_to_none=True)
        backward_mean(fm,accumulation);record=dict(microstep=step,optimizer_step=(step-1)//accumulation+1,fm=float(fm.detach()))
        ids,branches,weight=selection(task['repair'],step,extra_rng)
        if len(ids):
            np.add.at(exposures,(ids,branches),1);noise_hash.update(extra_g.get_state().numpy().tobytes())
            yr=F.normalize(torch.tensor(np.stack([bank[i][j] for i,j in zip(ids,branches)]),device='cuda'),dim=-1)
            zr=draw(yr[:,0],2,extra_g,0.);tr=torch.rand(len(yr),device='cuda',generator=extra_g)
            xr,vr,_=path_interpolate(zr,yr,tr,extra_g)
            pr=net.velocity(xr,tr,yr[:,0],yr.new_tensor([0.,.1]));extra=(pr[:,1:]-vr[:,1:]).square().sum(-1).mean()
            backward_mean(weight*extra,accumulation);assert torch.isfinite(extra);record.update(restart_fm=float(extra.detach()),weight=weight)
        if step==prefix_steps:
            prefix=dict(native=stream.hexdigest(),extra=noise_hash.hexdigest(),schedule=audit['hash'].hexdigest())
        if step%accumulation==0:
            actual_updates+=1
            lr=3e-6+(3e-5-3e-6)*.5*(1+math.cos(math.pi*(actual_updates-1)/max(1,updates-1)))
            for group in opt.param_groups:group['lr']=lr
            grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(fm) and torch.isfinite(grad)
            opt.step();record.update(gradient=float(grad),lr=lr)
            if smoke or actual_updates%100==0:history.append(record)
            if actual_updates%500==0:print(task,actual_updates,step,flush=True)
    assert actual_updates==updates
    torch.cuda.synchronize();training=time.perf_counter()-tick
    assert exposures.sum()==total*48 and (exposures.sum(1)==total).all()
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=24000+updates,microsteps=total,optimizer_updates=updates,
        rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state,
        pair_rng=pair_rng.bit_generator.state,extra_rng=extra_rng.bit_generator.state,extra_generator=extra_g.get_state()),out/'checkpoint.pt')
    dump(out/'history.json',history);dump(out/'exposures.json',exposures.tolist())
    net.eval();tick=time.perf_counter();seen_restart(net,out/'after_restart.h5',smoke)
    ev=evaluate(net,out,smoke);intermediate(net,out,smoke);torch.cuda.synchronize()
    dump(out/'complete.json',dict(task=task,step=24000+updates,microsteps=total,optimizer_updates=updates,train_seconds=training,evaluation_seconds=time.perf_counter()-tick,
        paired_input_hash=stream.hexdigest(),extra_noise_hash=noise_hash.hexdigest(),schedule_hash=audit['hash'].hexdigest(),prefix_hashes=prefix,
        data_hashes=hashes,restart_data_hashes=bank_hashes,evaluation=ev,device=torch.cuda.get_device_name(),scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:verify();train(a.index,a.action=='smoke')
