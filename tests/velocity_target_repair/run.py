"""Separate empirical conditional-field fit from stochastic supervision variance."""
import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.empirical_transport_repair.oracle import conditional
from tests.cross_dimension_20260924.model import source
from tests.flow_training_repair.objectives import draw_tau
from tests.sampling_balance_repair.run import RefillWindows
from tests.bauer_p2_20260923.common import cfg,dump
from tests.bauer_comprehensive_20260923.run import real_data,next_batch,require_gpu
from tests.condition_time_repair.run import build,expanded
from tests.branch_coverage_repair.run import checkpoint,pools,parents,seen
from tests.reference_resolution.run import parents as dev_parents
from tests.moment_bias_repair.run import evaluate

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def tasks():
    return [dict(domain='bauer',arm='hierarchical',seed=s,repair=r,updates=6000)
            for s in (22,11,33) for r in ('sampled_fm','conditional_mean')]


class TaggedWindows(RefillWindows):
    def __getitem__(self,index):
        item=super().__getitem__(index)
        g=int(np.searchsorted(self.ends,index,side='right'));_,row,starts,_=self.records[g]
        off=index-(self.ends[g-1] if g else 0);start=starts[off%len(starts)]
        self.audit['tags'].append(row['parent_id'] if start==0 and row['parent_id'] in self.pool else None)
        return item


@torch.no_grad()
def mean_target(x,tau,targets,initial):
    # Evaluate the already-tested finite-mixture oracle in float64 on the sphere.
    v,w,bad=conditional(F.normalize(x.double(),dim=-1),tau.double(),
                        F.normalize(targets.double(),dim=-1),F.normalize(initial.double(),dim=-1),2.)
    if bad.any():raise RuntimeError('Empirical oracle outside support: refusing nearest-target fallback')
    assert torch.isfinite(v).all()
    return v.to(x.dtype),w


@torch.no_grad()
def diagnose(net,out,smoke):
    old,new,_=pools(False);cases=[]
    for i,row in enumerate(parents()[:1] if smoke else parents()):
        cases.append(('train_pool72',row,new[i]))
        with h5py.File(ROOT/f'tests/branch_coverage_repair/reference/{i:02d}/probe.h5') as h:
            assert h.attrs['role']=='probe' and h.attrs['complete'];cases.append(('seen_unseen_noise128',row,h['spins'][:]))
    for i,row in enumerate(dev_parents()[:1] if smoke else dev_parents()):
        with h5py.File(ROOT/f'tests/reference_resolution/reference/{i:02d}/paths.h5') as h:
            assert h.attrs['complete'];cases.append(('dev_base128',row,h['base/spins'][:]))
    rows=[];times=torch.arange(11,device='cuda')*.1
    for i,(kind,parent,raw) in enumerate(cases):
        y=F.normalize(torch.tensor(raw,device='cuda',dtype=torch.float64),dim=-1);n=4 if smoke else 64
        gen=torch.Generator(device='cuda').manual_seed(936100+i)
        ids=torch.randint(len(y),(n,),generator=gen,device='cuda');target=y[ids];initial=target[:,0]
        z=source(initial,11,gen,2.)
        for tau in ((.05,.5) if smoke else (0.,.02,.05,.1,.25,.5,.75,.9)):
            tt=z.new_full((n,),tau);x,v,cut=path_interpolate(z,target,tt,gen)
            assert not cut['unresolved_cut_count'];truth,w=mean_target(x,tt,y,initial)
            pred=net.velocity(x.float(),tt.float(),initial.float(),times).double()
            def frame_mean(value):return value[:,1:].square().sum(-1).mean((0,2)).cpu().tolist()
            rows.append(dict(kind=kind,parent=parent,tau=tau,physical_time_ps=(np.arange(1,11)*.1).tolist(),
                sampled_target_mse=frame_mean(pred-v),empirical_field_mse=frame_mean(pred-truth),
                target_noise=frame_mean(v-truth),target_power=frame_mean(v),
                mean_posterior_effective_targets=float((1/w.square().sum(-1)).mean())))
    dump(out/'velocity_diagnostics.json',dict(rows=rows,
        warning='Exact only for each finite empirical path mixture and specified latent law; not the true LLG conditional velocity. Different pool sizes change teacher. Target noise is not physical thermal noise; no inference consumes future paths. Frames/sites are not independent replicates.'))


def prepare():
    assert not (HERE/'source_lock.json').exists(),'Already frozen'
    dump(HERE/'protocol.json',dict(tasks=tasks(),
        scope='Existing Bauer open L16/25/32,1ps/.1ps, unchanged physics/network/source. Start control24000; native_expanded sampler; no new data or dev training.',
        diagnosis='Before/after by physical frame and tau0,.02,.05,.1,.25,.5,.75,.9: sampled-target MSE, model versus finite-mixture conditional velocity, target residual noise, posterior effective target count.24train pools72,24training-parent unseen128,18independent dev base128.64 source/target draws per condition. Finite-mixture teachers are NOT true LLG population fields.',
        repair='Replace sampled velocity with its finite-training-mixture conditional mean ONLY for eligible original windows of24 L16 parents, tau<=0.9. Else ordinary FM. Uniform72 training branches exactly match existing refill sampler. No Hamiltonian teacher, no new weighting, no dev/probe supervision. Float64 oracle; outside-support error aborts rather than nearest extension. Same expected FM minimizer under exact finite-mixture arithmetic; finite precision and changed gradient variance are tested.',
        control='Sampled-FM and conditional-mean arms, three seeds; identical target/source/tau/parent/window draws,6000updates cosine3e-5->3e-6. Lower restart avoids previous100x restart, but not asserted optimal. Same updates not same compute; conditional oracle adds cost. Original inference unchanged.',
        evaluation='Fixed final checkpoint; original36dev64paths,expanded18dev256paths paired Euler128/midpoint128,24seen128paths. Whole-branch bootstrap100. Previous single-parent eight-path teacher experiments are not overwritten; this tests restricted variance-reduced supervision inside full data training.',
        limitations='Reduction in empirical velocity MSE does not establish true physical accuracy. No automatic certification or promotion. Conditional averaging cannot add missing physics or remedy all coverage deficits. No final-test claim; uncertainty and metric tradeoffs retained.',
        budget='Six rtx4090 jobs estimated25-50min each; six fat report jobs10-25min each; aggregation<3min. Three-family quick checks and paired GPU smoke<5min. No long monitoring.',
        skills='Karpathy isolated regression; experimental-design paired controls. Scientific Agent Skills https://doi.org/10.48550/arXiv.2609.00065'))
    lock=json.loads((ROOT/'tests/lr_continuation_repair/source_lock.json').read_text())
    for name,digest in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json',ROOT/'tests/empirical_transport_repair/oracle.py']
    for f in files:lock[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


def train(index,smoke):
    require_gpu('cuda');task=tasks()[index];net,opt,_=build(dict(task,repair='control'))
    ck=torch.load(checkpoint(task['seed']),map_location='cpu',weights_only=True);net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng'];refill=np.random.default_rng(935000+task['seed'])
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    _,new,hashes=pools(False);pool={r['parent_id']:x for r,x in zip(parents(),new)}
    teachers={k:F.normalize(torch.tensor(v,device='cuda'),dim=-1) for k,v in pool.items()}
    audit=dict(counts=Counter(),hash=hashlib.sha256(),tags=[]);config=cfg()
    data=[TaggedWindows(d,pool,refill,True,audit) for d in real_data(config)]
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    before=out/'before';before.mkdir();net.eval();diagnose(net,before,smoke);net.train()
    steps=32 if smoke else 6000;history=[];stream=hashlib.sha256();replaced=0;eligible=0;tick=time.perf_counter()
    for step in range(1,steps+1):
        audit['tags'].clear();y,t=next_batch(task,config,data,rng,gs['target'],'cuda')
        z=source(y[:,0],11,gs['source'],net.source_shift);tau,_=draw_tau(len(y),1.,'baseline',gs['tau'],'cuda')
        x,v,cut=path_interpolate(z,y,tau,gs['source'])
        for value in (y,z,tau):stream.update(value.detach().cpu().numpy().tobytes())
        for j,parent in enumerate(audit['tags']):
            if parent is None or tau[j]>.9:continue
            eligible+=1
            if task['repair']=='conditional_mean':
                target,_=mean_target(x[j:j+1],tau[j:j+1],teachers[parent],y[j:j+1,0])
                v[j:j+1]=target;replaced+=1
        pred=net.velocity(x,tau,y[:,0],t);loss=(pred[:,1:]-v[:,1:]).square().sum(-1).mean()
        lr=3e-6+(3e-5-3e-6)*.5*(1+math.cos(math.pi*(step-1)/max(1,steps-1)))
        for g in opt.param_groups:g['lr']=lr
        opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        assert torch.isfinite(loss) and torch.isfinite(grad);opt.step()
        if smoke or step%100==0:history.append(dict(step=24000+step,loss=float(loss.detach()),lr=lr))
        if step%500==0:print(task,step,flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-tick
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=24000+steps,rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state),out/'checkpoint.pt')
    dump(out/'history.json',history);net.eval();diagnose(net,out,smoke);seen(net,out,smoke);ev=evaluate(net,out,[],smoke);extra=expanded(net,out,smoke)
    dump(out/'complete.json',dict(task=task,train_seconds=elapsed,evaluation=ev,expanded=extra,data_hashes=hashes,
        input_hash=stream.hexdigest(),schedule_hash=audit['hash'].hexdigest(),counts=dict(audit['counts']),eligible=eligible,replaced=replaced,
        device=torch.cuda.get_device_name(),scientific_certification=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','smoke']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();return
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    train(a.index,a.action=='smoke')


if __name__=='__main__':main()
