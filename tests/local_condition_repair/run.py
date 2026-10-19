"""Paired original backbone, pointwise readout, and neighbor-conditioned readout."""
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
from .model import initialize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ('baseline', 'pointwise', 'neighbor')


def tasks():
    return [dict(domain='bauer', arm='hierarchical', seed=s, repair=r, updates=6000)
            for s in (11, 22, 33) for r in ARMS]


def source(seed):
    return ROOT / f'tests/gradient_accumulation_repair/train/{(11,22,33).index(seed)*3:02d}'


def selection(arm, step, rng, count=48):
    if arm not in ARMS:raise ValueError(arm)
    return rng.permutation(count),rng.integers(64,size=count),1.


def prepare():
    assert not (HERE / 'source_lock.json').exists()
    verify_sources()
    assert json.loads((DATA / 'data_audit.json').read_text())['passed']
    _, hashes = load_train()
    protocol = dict(tasks=tasks(),
        question='Does a direct learned local-state residual readout improve conditional transfer beyond a parameter-matched pointwise readout?',
        audit='Current backbone already receives actualS0 and passes neighbor messages. There is no demonstrated missing-initial-state bug. Repeated training/weighting/noise interventions mainly improved seen states. This tests a representation shortcut, not proof that old hidden features discarded information.',
        model='Unchanged pretrained hierarchical two-frame backbone. baseline adds nothing. pointwise and neighbor each add31->64->64->3SiLU readout with6403parameters,zero final layer. Inputs:four3component feature blocks in registered crystal frame,nineFourierphysical-time,nineFouriertransport-time,log1pdegree. Both first blocks=current transport draft and actualS0. neighbor additional blocks=mean neighborS0 and mean neighbor draft;pointwise=componentwise draft*S0 and draft-S0. Output transformed back and tangent-projected,frame0zero. No Hamiltonian coefficients,analytic field,LLG torque or clean future condition.',
        limitation='Extra branch features include crystal-frame component invariants under JOINT rotations,not spin-only symmetry or reflection. Neighbor mean retains cancellation magnitude; topology uses existingregistered geometry. A readout shortcut does not establish a new multiscale theory or large-size physics. This Bauer package remains open-boundary and.1ps perstep.',
        controls='All start matching previous gradient_accumulation batch1 step30000 weights/Adam,not select best on dev.6000updates cosine3e-5->3e-6,original native batch8 plus balanced48TRAIN restart batch weight1 everyupdate. Baseline optimizer states retained;adapters new optimizer group,oldstates retained. Pointwise and neighbor initialization paired and parametercount equal. Allthree initialization velocities equal oldmodel byzerohead. Backbone alsofinetuned. Baseline smaller/faster;measuredcosts reported.',
        pairing='Identical native data/refill/pair/source/tau/cut and extra target/noise streams perseed;hashes checked. Only64TRAINfutures per48states enter restart loss. Sourcegroups/splits/frozenrecords unchanged. No newLLGdata.',
        evaluation='Reuse frozen full evaluator:48trainorigin64generated versus128probe/fine,before sourceafter_restart;18dev128paths1ps midpoint128/512,24oldseen64paths,72devrealintermediate128futures. Save all generatedpaths;report conditionalmeans,covariance,variance,energy,space,time,family/size/time strata. Fixed final36000checkpoint;no adaptive winner selection.',
        uncertainty='Wholebranch bootstrap100 atfixedparent;bothreference/generatedsampling,errorbars notphysicalbranchspread. Correlated source states averaged descriptively,three seedsseparate. Reuseddev notfinalholdout,no inventedthresholds.',
        decision='Neighbor better than pointwise supports useful direct neighborhood encoding underthisrecipe. Both similar gains suggest extra readout/capacity effect. Only trainorigin improvementsleavegeneralizationunfixed. If allfail,do notblindlyincreasebudget. Mixed versus specialized-family training remains a separate possible diagnostic,not runinthisexperiment. No productionpromotion.',
        tests='Zero-init identity;nonzerohead jointSO3,tangency,initialclamp,matched-latent samplingcovariance;openedgecount,nocrossboundary,cancellation;neighbor feature sensitivity versus pointwisecontrol;metadata rejection;oldoptimizerstepretention.',
        physics='Bauer open,J50meV,kappa.1,alpha.1,theta.11,T63.824849668K;reference dt.25/.125fs,savedlag.1ps,tenfreshnoise transitions1ps. No temporalrelabelling.',
        budget='3GPU smoke1-4min;9formalGPUjobs estimated2-3h eachincludingfullsampling,9fatstats3-6min,web<3min. AllGPUrtx4090,zrs-mag;no longmonitoring.',
        skills='karpathy-guidelines surgicaltests-only;experimental-design parameter-matchedcontrols;matplotlibPNGreport.')
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
    assert ck['step']==30000 and ck['task']['seed']==seed and ck['task']['repair']=='batch1'
    net,opt=initialize(ck,task['repair'],seed)
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
        for group in opt.param_groups:group['lr']=lr
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(fm) and torch.isfinite(grad)
        opt.step();record.update(gradient=float(grad),lr=lr)
        if smoke or step%100==0:history.append(record)
        if step%500==0:print(task,step,flush=True)
    torch.cuda.synchronize();training=time.perf_counter()-tick
    assert exposures.sum()==total*48 and (exposures.sum(1)==total).all()
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=30000+total,
        rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state,
        pair_rng=pair_rng.bit_generator.state,extra_rng=extra_rng.bit_generator.state,extra_generator=extra_g.get_state()),out/'checkpoint.pt')
    dump(out/'history.json',history);dump(out/'exposures.json',exposures.tolist())
    net.eval();tick=time.perf_counter();seen_restart(net,out/'after_restart.h5',smoke)
    ev=evaluate(net,out,smoke);intermediate(net,out,smoke);torch.cuda.synchronize()
    dump(out/'complete.json',dict(task=task,step=30000+total,train_seconds=training,evaluation_seconds=time.perf_counter()-tick,
        paired_input_hash=stream.hexdigest(),extra_noise_hash=noise_hash.hexdigest(),schedule_hash=audit['hash'].hexdigest(),
        data_hashes=hashes,restart_data_hashes=bank_hashes,evaluation=ev,device=torch.cuda.get_device_name(),parameter_count=sum(p.numel() for p in net.parameters()),scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:verify();train(a.index,a.action=='smoke')
