"""Keep native supervision while separating restart exposure from batch balance."""
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
from tests.training_restart_repair.data import load_train, verify_sources, HERE as DATA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ('sparse_iid', 'dense_iid', 'dense_balanced')


def tasks():
    return [dict(domain='bauer', arm='hierarchical', seed=s, repair=r, updates=6000)
            for s in (11, 22, 33) for r in ARMS]


def source(seed):
    return ROOT / f'tests/training_restart_repair/train/{(11,22,33).index(seed)*3+1:02d}'


def selection(arm, step, rng, count=48):
    if arm not in ARMS:
        raise ValueError(arm)
    if arm == 'sparse_iid' and step % 2:
        return np.array([], dtype=int), np.array([], dtype=int), 0.
    n = 8 if arm == 'sparse_iid' else count
    ids = rng.permutation(count) if arm == 'dense_balanced' else rng.integers(count, size=n)
    return ids, rng.integers(64, size=n), .5 if arm == 'sparse_iid' else 1.


def prepare():
    assert not (HERE / 'source_lock.json').exists()
    verify_sources()
    assert json.loads((DATA / 'data_audit.json').read_text())['passed']
    _, hashes = load_train()
    protocol = dict(tasks=tasks(),
        question='Can greater effective restart supervision improve independent-state dynamics while native training prevents forgetting? At matched dense exposure, does exact within-batch coverage help?',
        evidence='Fixed-state fitting improved conditional means but harmed tilted-state energy/variance. Prior 48-state sampler was uniform in expectation, not systematically biased. Radius1/2 local descriptor coverage was already nearby; it is not proof of full-state coverage.',
        controls='Each seed starts at its restart_fm step18000 weights AND Adam. Retain every original native batch and restored RNG streams. Continue6000updates cosine3e-5->3e-6. sparse_iid:batch8 every2updates weight.5; dense_iid:batch48 everyupdate weight1; dense_balanced:all48states once everyupdate weight1,one independent uniform TRAIN branch per state. Dense arms have identical expected objective, update count and batch size; only within-batch composition changes. Sparse versus dense confounds exposure, weight and compute deliberately; do not call it pure balance effect.',
        data='Reuse only64TRAIN futures per48TRAIN-origin restart states from24parents. Original native windows and72branch t0 refill preserved. No probe, fine or dev trajectories enter losses. Exact balancing does not balance branch counts or prove full configuration coverage.',
        pairing='Restore original data/refill/pair/source/tau/cut streams; verify equal native input and schedule hashes across3arms perseed. Extra RNG separate from native. Dense extra source/tau/interpolation generator state hashes must match, target identities intentionally differ. Record48x64 exposure counts and checkpoint RNG states.',
        physics='Unchanged Bauer open chain,J50meV,kappa.1,alpha.1,theta.11,T63.824849668K,zero field. Learned lag.1ps,reference dt.25/.125fs; evaluation1ps10fresh-noise transitions. No checkpoint time relabeling or architecture change.',
        evaluation='Fixed final24000 checkpoint.48TRAIN-origin states64generated futures mid128 compared with independent128probe/fine. Before values reuse source after_restart.h5 with identical latent seeds.18dev parents128full1ps paths midpoint128/512,24oldseen64paths mid128,72realLLG dev intermediate states128futures mid128. Save complete generated trajectories. Compare means,covariance,variance,energy,spatial and two-time statistics; family/time/size stratification exposes negative transfer.',
        uncertainty='Whole-branch bootstrap100 at fixed parent, both sample sets resampled. States from shared parents are correlated; cross-state means descriptive only. Three training seeds separately. Reused dev is not untouched finalholdout. Reference control includes finite-sampling and numerical difference; not automatic tolerance.',
        decisions='Improvement only on TRAIN-origin probes is fitting without generalization. Stronger restart fit plus old-anchor regression is a tradeoff, not success. Dense iid versus balanced difference isolates within-batch coverage, not enlarged support. No improvement suggests weighting insufficient, not proof of architecture impossibility. No certification, production promotion or speedup claim.',
        budget='GPU smoke3arms estimated1-4min each;9formal rtx4090 jobs estimated2-3h each including full evaluation;9fat stats estimated5-10min each;web<3min. Submit accepted long dependency chain then stop monitoring.',
        skills='karpathy-guidelines: isolated tests-only surgical changes; experimental-design: concurrent paired controls, predeclared estimands and explicit confounds.')
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
    assert ck['step']==18000 and ck['task']['seed']==seed and ck['task']['repair']=='restart_fm'
    net=TransitionNet().cuda();net.load_state_dict(ck['model'])
    opt=torch.optim.AdamW(net.parameters(),lr=3e-5,weight_decay=1e-5);opt.load_state_dict(ck['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    refill=np.random.default_rng();refill.bit_generator.state=ck['refill_rng']
    pair_rng=np.random.default_rng();pair_rng.bit_generator.state=ck['pair_rng']
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    extra_rng=np.random.default_rng(946000+seed)
    extra_g=torch.Generator(device='cuda').manual_seed(946100+seed)
    bank,bank_hashes=load_train(False);assert len(bank)==48
    _,new,hashes=pools(False);pool={r['parent_id']:x for r,x in zip(train_parents(),new)}
    audit=dict(counts=Counter(),hash=hashlib.sha256());config=cfg()
    data=[RefillWindows(d,pool,refill,True,audit) for d in real_data(config)]
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    total=16 if smoke else 6000;history=[];stream=hashlib.sha256();noise_hash=hashlib.sha256()
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
        opt.zero_grad(set_to_none=True);fm.backward();record=dict(step=step,fm=float(fm.detach()))
        ids,branches,weight=selection(task['repair'],step,extra_rng)
        if len(ids):
            np.add.at(exposures,(ids,branches),1);noise_hash.update(extra_g.get_state().numpy().tobytes())
            yr=F.normalize(torch.tensor(np.stack([bank[i][j] for i,j in zip(ids,branches)]),device='cuda'),dim=-1)
            zr=draw(yr[:,0],2,extra_g,0.);tr=torch.rand(len(yr),device='cuda',generator=extra_g)
            xr,vr,_=path_interpolate(zr,yr,tr,extra_g)
            pr=net.velocity(xr,tr,yr[:,0],yr.new_tensor([0.,.1]));extra=(pr[:,1:]-vr[:,1:]).square().sum(-1).mean()
            (weight*extra).backward();assert torch.isfinite(extra);record.update(restart_fm=float(extra.detach()),weight=weight)
        lr=3e-6+(3e-5-3e-6)*.5*(1+math.cos(math.pi*(step-1)/max(1,total-1)))
        for group in opt.param_groups:group['lr']=lr
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(fm) and torch.isfinite(grad)
        opt.step();record.update(gradient=float(grad),lr=lr)
        if smoke or step%100==0:history.append(record)
        if step%500==0:print(task,step,flush=True)
    torch.cuda.synchronize();training=time.perf_counter()-tick
    assert exposures.sum()==(total//2*8 if task['repair']=='sparse_iid' else total*48)
    if task['repair']=='dense_balanced':assert (exposures.sum(1)==total).all()
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=18000+total,
        rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state,
        pair_rng=pair_rng.bit_generator.state,extra_rng=extra_rng.bit_generator.state,extra_generator=extra_g.get_state()),out/'checkpoint.pt')
    dump(out/'history.json',history);dump(out/'exposures.json',exposures.tolist())
    net.eval();tick=time.perf_counter();seen_restart(net,out/'after_restart.h5',smoke)
    ev=evaluate(net,out,smoke);intermediate(net,out,smoke);torch.cuda.synchronize()
    dump(out/'complete.json',dict(task=task,step=18000+total,train_seconds=training,evaluation_seconds=time.perf_counter()-tick,
        paired_input_hash=stream.hexdigest(),extra_noise_hash=noise_hash.hexdigest(),schedule_hash=audit['hash'].hexdigest(),
        data_hashes=hashes,restart_data_hashes=bank_hashes,evaluation=ev,device=torch.cuda.get_device_name(),scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:verify();train(a.index,a.action=='smoke')
