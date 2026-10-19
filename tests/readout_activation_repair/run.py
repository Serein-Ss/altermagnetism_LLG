"""Matched readout learning-rate intervention; frozen source experiment unchanged."""
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
from tests.gradient_accumulation_repair.run import verify as verify_sources
from tests.local_condition_repair.model import initialize
from .diagnostics import probe, ablation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ('neighbor_low', 'neighbor_fast', 'pointwise_fast')


def tasks():
    return [dict(domain='bauer', arm='hierarchical', seed=s, repair=r, updates=6000)
            for s in (11, 22, 33) for r in ARMS]


def source(seed):
    return ROOT / f'tests/gradient_accumulation_repair/train/{(11,22,33).index(seed)*3:02d}'


def selection(arm, step, rng, count=48):
    if arm not in ARMS:raise ValueError(arm)
    return rng.permutation(count),rng.integers(64,size=count),1.


def set_rates(opt,lr,arm):
    if arm not in ARMS:raise ValueError(arm)
    assert len(opt.param_groups)==2
    opt.param_groups[0]['lr']=lr
    opt.param_groups[1]['lr']=lr*(10 if arm.endswith('fast') else 1)


def prepare():
    assert not (HERE/'source_lock.json').exists()
    verify_sources()
    assert json.loads((DATA/'data_audit.json').read_text())['passed']
    _,hashes=load_train()
    protocol=dict(tasks=tasks(),
        hypothesis='Previous heads changed TRAIN-probe velocity by only0.074-0.308percent. Test fresh-head optimization, not assume neighborhood information useless.',
        arms='neighbor_low readout3e-5->3e-6;neighbor_fast and pointwise_fast readout3e-4->3e-5. Backbone all3e-5->3e-6. Paired equal6403readoutparameters. No inference gain or post-hoc temperature change.',
        controls='Same gradient_accumulation batch1 step30000 source,originalAdamretained,newheadzerooutput/newAdam. Exact previous local_condition data/source/pair/tau streams,6000updates,native8+balanced48TRAINrestart weight1. NewheadLRonly changes neighborlow/fast. Fastpointwise controls localneighborfeatures at sameLR.',
        physics='Bauer open,J50meV,kappa.1,alpha.1,theta.11,T63.824849668K. Reference dt.25/.125fs;trainedlag.1ps,10freshnoise transitions1ps. No new data,loss,physics or time relabel.',
        diagnostics='Fixed TRAIN transport probes at tau.1/.5/.9 report head/base RMS and head-off FM. On/off generation on six L16 independent DEV states (two per family),same latent seeds;checkpoint backbone unchanged during ablation. Full existing72intermediate and18dev1ps128/512 evaluator retained;allpaths saved.',
        evaluation='Freeze final36000checkpoint,three trainingseeds. Wholebranchbootstrap100,reference and generated samplingerror;fixedparent diagnostics,reuseddev not finalholdout. Report family,size,time strata,energy/space/time separately. No arbitrary universal pass score.',
        limitations='Stronger head activity is manipulation evidence,not physicalaccuracy. If neither fastcandidate improves,do not declare general locality ineffective. HighLR may worsen variance or timecorrelation. Familystratification diagnoses failures but does not byitself identify causality;no specialized-family model claim.',
        budget='3smokeGPU estimated1-4min each.9formalGPU estimated2-3h each including evaluations;9fatstats3-6min;web<3min. Partitionrtx4090,zrs-mag. Stopmonitoring formaljobs afteracceptance.',
        skills='karpathy-guidelines;experimental-design pairedcontrols;matplotlib PNG. No productionpromotion.')
    dump(HERE/'protocol.json',protocol)
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json',HERE/'activation_audit.json',ROOT/'tests/local_condition_repair/model.py']
    for seed in (11,22,33):
        files += [source(seed)/'checkpoint.pt',source(seed)/'complete.json',source(seed)/'after_restart.h5']
    lock=dict(hashes)
    for p in files:lock[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


def verify():
    verify_sources()
    for name, digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name


def train(index, smoke):
    require_gpu('cuda'); task=tasks()[index]; seed=task['seed']
    ck=torch.load(source(seed)/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert ck['step']==30000 and ck['task']['seed']==seed and ck['task']['repair']=='batch1'
    net,opt=initialize(ck,task['repair'].split('_')[0],seed)
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    refill=np.random.default_rng();refill.bit_generator.state=ck['refill_rng']
    pair_rng=np.random.default_rng();pair_rng.bit_generator.state=ck['pair_rng']
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    extra_rng=np.random.default_rng(949100+seed)
    extra_g=torch.Generator(device='cuda').manual_seed(949200+seed)
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
        set_rates(opt,lr,task['repair'])
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(fm) and torch.isfinite(grad)
        opt.step();record.update(gradient=float(grad),lr=lr,readout_lr=opt.param_groups[1]['lr'])
        if smoke or step%100==0:history.append(record)
        if step%500==0:print(task,step,flush=True)
    torch.cuda.synchronize();training=time.perf_counter()-tick
    assert exposures.sum()==total*48 and (exposures.sum(1)==total).all()
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=30000+total,
        rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state,
        pair_rng=pair_rng.bit_generator.state,extra_rng=extra_rng.bit_generator.state,extra_generator=extra_g.get_state()),out/'checkpoint.pt')
    dump(out/'history.json',history);dump(out/'exposures.json',exposures.tolist())
    net.eval();dump(out/'activation.json',probe(net));tick=time.perf_counter();ablation(net,out,smoke);seen_restart(net,out/'after_restart.h5',smoke)
    ev=evaluate(net,out,smoke);intermediate(net,out,smoke);torch.cuda.synchronize()
    dump(out/'complete.json',dict(task=task,step=30000+total,train_seconds=training,evaluation_seconds=time.perf_counter()-tick,
        paired_input_hash=stream.hexdigest(),extra_noise_hash=noise_hash.hexdigest(),schedule_hash=audit['hash'].hexdigest(),
        data_hashes=hashes,restart_data_hashes=bank_hashes,evaluation=ev,device=torch.cuda.get_device_name(),parameter_count=sum(p.numel() for p in net.parameters()),scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:verify();train(a.index,a.action=='smoke')
