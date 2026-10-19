"""Bounded optimizer response test; unchanged conditional teacher and sampler."""
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
from tests.training_restart_repair.data import load_train
from tests.conditional_teacher_repair.run import verify as verify_sources
from tests.axial_symmetry_repair.symmetry import rotate,probe
from tests.conditional_coupling_repair.coupling import schedule
from tests.conditional_teacher_repair.target import labels
from .diagnostic import optimizer, rate, risk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ('restore_low', 'reset_low', 'restore_high', 'reset_high')


def tasks():
    return [dict(domain='bauer', arm='hierarchical', seed=s, repair=r, updates=512)
            for s in (11, 22, 33) for r in ARMS]


def source(seed):
    return ROOT / f'tests/conditional_teacher_repair/train/{(11,22,33).index(seed)*2+1:02d}'


def augment(x,v,initial,generator,enabled):
    angles=torch.rand(len(x),device=x.device,dtype=x.dtype,generator=generator)*(2*math.pi)
    if not enabled:return x,v,initial
    return rotate(x,angles),rotate(v,angles),rotate(initial,angles)


def prepare():
    assert not (HERE/'source_lock.json').exists()
    verify_sources()
    _,hashes=load_train()
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']
    for seed in (11,22,33):files += [source(seed)/'checkpoint.pt',source(seed)/'after_restart.h5']
    for f in files:hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',hashes);(HERE/'logs').mkdir(exist_ok=True)


def verify():
    verify_sources()
    for name, digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name


def train(index, smoke):
    require_gpu('cuda'); task=tasks()[index]; seed=task['seed']
    ck=torch.load(source(seed)/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert ck['step']==54000 and ck['task']['seed']==seed and ck['task']['repair']=='conditional'
    net=TransitionNet().cuda();net.load_state_dict(ck['model'])
    opt=optimizer(net,ck['optimizer'],task['repair'])
    original=[p.detach().clone() for p in net.parameters()]
    optimizer_steps_before=sorted({int(v['step']) for v in opt.state.values()})
    aug_g=torch.Generator(device='cuda').manual_seed(956100+seed)
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    refill=np.random.default_rng();refill.bit_generator.state=ck['refill_rng']
    pair_rng=np.random.default_rng();pair_rng.bit_generator.state=ck['pair_rng']
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    extra_g=torch.Generator(device='cuda').manual_seed(956200+seed)
    bank,bank_hashes=load_train(False);assert len(bank)==48
    bank_gpu=[F.normalize(torch.tensor(b,device='cuda'),dim=-1) for b in bank]
    _,new,hashes=pools(False);pool={r['parent_id']:x for r,x in zip(train_parents(),new)}
    audit=dict(counts=Counter(),hash=hashlib.sha256());config=cfg()
    data=[RefillWindows(d,pool,refill,True,audit) for d in real_data(config)]
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    total=16 if smoke else 512;history=[];stream=hashlib.sha256();noise_hash=hashlib.sha256()
    state_ids,branch_ids=schedule(956300+seed,total)
    exposures=np.zeros((48,64),dtype=np.int64)
    before=risk(net,bank_gpu,smoke)
    tick=time.perf_counter();net.train()
    for step in range(1,total+1):
        whole,_=next_batch(task,config,data,rng,gs['target'],'cuda')
        stream.update(whole.detach().cpu().numpy().tobytes())
        for k in ('source','tau'):stream.update(gs[k].get_state().numpy().tobytes())
        starts=np.where(pair_rng.random(len(whole))<.5,0,pair_rng.integers(1,10,len(whole)))
        y=pair_window(whole,torch.as_tensor(starts,device='cuda'))
        z=draw(y[:,0],11,gs['source'],0.)[:,:2];tau=torch.rand(len(y),device='cuda',generator=gs['tau'])
        cut_state=gs['cut'].get_state();torch.randn(whole.shape,device='cuda',generator=gs['cut'])
        temp=torch.Generator(device='cuda').set_state(cut_state);x,v,_=path_interpolate(z,y,tau,temp)
        x,v,anchor=augment(x,v,y[:,0],aug_g,True)
        pred=net.velocity(x,tau,anchor,y.new_tensor([0.,.1]));fm=(pred[:,1:]-v[:,1:]).square().sum(-1).mean()
        opt.zero_grad(set_to_none=True);fm.backward();record=dict(step=step,fm=float(fm.detach()))
        ids,branches,weight=state_ids[step-1],branch_ids[step-1],1.
        if len(ids):
            np.add.at(exposures,(ids,branches),1);noise_hash.update(extra_g.get_state().numpy().tobytes())
            yr=F.normalize(torch.tensor(np.stack([bank[i][j] for i,j in zip(ids,branches)]),device='cuda'),dim=-1)
            zr=draw(yr[:,0],2,extra_g,0.)
            tr=torch.rand(len(yr),device='cuda',generator=extra_g)
            drafts=[];sampled=[];conditional=[];diagnostics=[]
            for start in (0,16,32):
                xx,vv,rr,dd=labels(zr[start:start+16],yr[start:start+16],tr[start:start+16],
                    bank_gpu[ids[start]],extra_g)
                drafts.append(xx);sampled.append(vv);conditional.append(rr);diagnostics.append(dd)
            xr=torch.cat(drafts);vr=torch.cat(conditional)
            record.update(teacher_count=sum(d['teacher_count'] for d in diagnostics),
                effective_modes=float(np.mean([d['effective_modes'] for d in diagnostics])),
                target_residual_mse=float(np.mean([d['target_residual_mse'] for d in diagnostics])))
            angles=torch.rand(3,device='cuda',generator=aug_g).repeat_interleave(16)*(2*math.pi)
            anchor=rotate(xr[:,0],angles)
            xr,vr=rotate(xr,angles),rotate(vr,angles)
            pr=net.velocity(xr,tr,anchor,yr.new_tensor([0.,.1]));extra=(pr[:,1:]-vr[:,1:]).square().sum(-1).mean()
            (weight*extra).backward();assert torch.isfinite(extra);record.update(restart_fm=float(extra.detach()),weight=weight)
        lr=rate(task['repair'],step,total)
        for group in opt.param_groups:group['lr']=lr
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(fm) and torch.isfinite(grad)
        opt.step();record.update(gradient=float(grad),lr=lr)
        if smoke or step==1 or step%32==0:history.append(record)
        if step%500==0:print(task,step,flush=True)
    torch.cuda.synchronize();training=time.perf_counter()-tick
    assert exposures.sum()==total*48 and (exposures.sum(1)==total).all()
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=54000+total,
        rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state,
        pair_rng=pair_rng.bit_generator.state,extra_generator=extra_g.get_state(),augmentation_generator=aug_g.get_state()),out/'checkpoint.pt')
    dump(out/'history.json',history);dump(out/'exposures.json',exposures.tolist())
    net.eval();after=risk(net,bank_gpu,smoke)
    change=float(torch.sqrt(sum((p.detach()-q).square().sum() for p,q in zip(net.parameters(),original))/sum(q.square().sum() for q in original)))
    dump(out/'response.json',dict(before=before,after=after,relative_parameter_change=change,
        optimizer_steps_before=optimizer_steps_before,optimizer_steps_after=sorted({int(v['step']) for v in opt.state.values()})))
    tick=time.perf_counter();seen_restart(net,out/'after_restart.h5',smoke)
    intermediate(net,out,smoke);torch.cuda.synchronize();ev={'scope':'0.1ps independent futures; no full1ps certification'}
    dump(out/'complete.json',dict(task=task,step=54000+total,train_seconds=training,evaluation_seconds=time.perf_counter()-tick,
        paired_input_hash=stream.hexdigest(),extra_noise_hash=noise_hash.hexdigest(),schedule_hash=audit['hash'].hexdigest(),
        group_schedule_hash=hashlib.sha256(state_ids.tobytes()+branch_ids.tobytes()).hexdigest(),data_hashes=hashes,restart_data_hashes=bank_hashes,evaluation=ev,device=torch.cuda.get_device_name(),parameter_count=sum(p.numel() for p in net.parameters()),scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:verify();train(a.index,a.action=='smoke')
