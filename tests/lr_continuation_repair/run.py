"""Paired continuation schedules with frozen data and intermediate path diagnostics."""
import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from tests.sampling_balance_repair.run import RefillWindows
from tests.bauer_p2_20260923.common import cfg,dump
from tests.bauer_comprehensive_20260923.run import real_data,next_batch,require_gpu
from tests.condition_time_repair.run import build,expanded
from tests.branch_coverage_repair.run import checkpoint,pools,parents,seen
from tests.curriculum_repair.run import fm_loss
from tests.moment_bias_repair.run import evaluate

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
STAGES=(500,1500,3000,6000)


def tasks():
    return [dict(domain='bauer',arm='hierarchical',seed=s,repair=r,updates=6000)
            for s in (22,11,33) for r in ('restart_high','restart_low','continue_floor')]


def learning_rate(arm,step,total=6000):
    if not 1<=step<=total:raise ValueError('step outside schedule')
    if arm=='continue_floor':return 3e-6
    if arm not in ('restart_high','restart_low'):raise ValueError(arm)
    peak=3e-4 if arm=='restart_high' else 3e-5
    return 3e-6+(peak-3e-6)*.5*(1+math.cos(math.pi*(step-1)/max(1,total-1)))


def prepare():
    assert not (HERE/'source_lock.json').exists(),'Already frozen'
    audit=[]
    for seed in (11,22,33):
        ck=torch.load(checkpoint(seed),map_location='cpu',weights_only=True)
        lr=[g['lr'] for g in ck['optimizer']['param_groups']]
        assert ck['step']==24000 and lr==[3e-6]
        audit.append(dict(seed=seed,checkpoint=str(checkpoint(seed).relative_to(ROOT)),step=ck['step'],loaded_lr=lr,adam_states=len(ck['optimizer']['state'])))
    dump(HERE/'protocol.json',dict(tasks=tasks(),checkpoint_audit=audit,stages=STAGES,
        scope='Bauer open L16/25/32, same physics,1ps/.1ps grid, same network/FM-only loss. Reuse native_expanded data and all reference ensembles. No new data or production changes.',
        design='Three seed-blocked concurrent arms: cosine3e-4->3e-6,cosine3e-5->3e-6,constant3e-6. All start unchanged control step24000 weights and142 Adam states, not the degraded step30000 checkpoints. Same original size/parent/window sampling and same branch-refill, source and tau RNG. Only learning-rate schedule changes.6000 updates, not equal integrated learning-rate dose.',
        checkpoints='Save full model,Adam,explicit/global RNG and sampling state at +500,+1500,+3000,+6000; no adaptive early stopping. Evaluate18dev parents256generated per stage with paired latent Euler128 and midpoint128; preserve all paths. Final also24seen128paths and original36dev64paths. Intermediate snapshots do not consume training RNG.',
        baseline='Reuse independently stored step24000 expanded diagnostics from condition_time_repair. Intermediate comparisons use identical latent seeds and reference ensembles. Added evaluation cost reported separately.',
        interpretation='Fixed final6000 result primary; intermediate trajectory is diagnostic, not retroactive final-test selection. No automatic best-model promotion or equivalence tolerance. Low LR may only preserve baseline rather than improve it. Reused dev is not an untouched test; whole-branch bootstrap100 and seed/size-separated physical metrics retained.',
        budget='9GPU jobs estimated35-65min each;9fat per-run reports15-35min each;final aggregation<3min. Paired3-arm smoke1-3min, no long-job monitoring.',
        skills='Karpathy isolated schedule change; experimental-design concurrent paired controls. Scientific Agent Skills https://doi.org/10.48550/arXiv.2609.00065'))
    lock=json.loads((ROOT/'tests/sampling_balance_repair/source_lock.json').read_text())
    for name,digest in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for f in list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']:
        lock[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


def train(index,smoke):
    require_gpu('cuda');task=tasks()[index];net,opt,_=build(dict(task,repair='control'))
    ck=torch.load(checkpoint(task['seed']),map_location='cpu',weights_only=True)
    net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng'];refill=np.random.default_rng(935000+task['seed'])
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    _,new,hashes=pools(False);pool={r['parent_id']:x for r,x in zip(parents(),new)}
    audit=dict(counts=Counter(),hash=hashlib.sha256());config=cfg()
    data=[RefillWindows(d,pool,refill,True,audit) for d in real_data(config)]
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    steps=32 if smoke else 6000;stages=(16,32) if smoke else STAGES
    history=[];noise=hashlib.sha256();evaluation_seconds=0.;start=time.perf_counter();net.train()
    for step in range(1,steps+1):
        y,t=next_batch(task,config,data,rng,gs['target'],'cuda')
        for k in ('source','tau'):noise.update(gs[k].get_state().numpy().tobytes())
        lr=learning_rate(task['repair'],step,steps)
        for group in opt.param_groups:group['lr']=lr
        loss,_=fm_loss(net,y,t,gs);opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(loss) and torch.isfinite(grad);opt.step()
        if smoke or step%100==0:history.append(dict(step=24000+step,loss=float(loss.detach()),lr=lr,gradient=float(grad)))
        if step in stages:
            folder=out/f'step{step:05d}';folder.mkdir()
            torch.cuda.synchronize();tick=time.perf_counter()
            state=dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=24000+step,
                rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),auxiliary_rng=ck['auxiliary_rng'])
            torch.save(state,folder/'checkpoint.pt')
            # Evaluation uses its own generators and cannot perturb subsequent optimization.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                net.eval();ev=expanded(net,folder,smoke)
            net.train()
            for k,g in gs.items():assert torch.equal(g.get_state(),state['rng'][k]),k
            assert torch.equal(torch.get_rng_state(),state['torch_rng']) and torch.equal(torch.cuda.get_rng_state(),state['cuda_rng'])
            torch.cuda.synchronize();elapsed=time.perf_counter()-tick;evaluation_seconds+=elapsed
            dump(folder/'complete.json',dict(step=24000+step,evaluation=ev,seconds=elapsed,schedule_hash=audit['hash'].hexdigest(),source_tau_hash=noise.hexdigest()))
            print(task,step,flush=True)
    torch.cuda.synchronize();train_seconds=time.perf_counter()-start-evaluation_seconds
    dump(out/'history.json',history);net.eval();tick=time.perf_counter();seen(net,out,smoke);ev=evaluate(net,out,[],smoke)
    torch.cuda.synchronize();evaluation_seconds+=time.perf_counter()-tick
    dump(out/'complete.json',dict(task=task,train_seconds=train_seconds,evaluation_seconds=evaluation_seconds,evaluation=ev,
        data_hashes=hashes,counts=dict(audit['counts']),schedule_hash=audit['hash'].hexdigest(),source_tau_hash=noise.hexdigest(),
        stages=list(stages),device=torch.cuda.get_device_name(),scientific_certification=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','smoke']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();return
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    train(a.index,a.action=='smoke')


if __name__=='__main__':main()
