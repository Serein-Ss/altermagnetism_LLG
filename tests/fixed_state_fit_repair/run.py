"""Bounded fitting diagnostic for the current two-frame conditional model."""
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
from scripts.model.gate_f import path_interpolate
from tests.bauer_p2_20260923.common import dump
from tests.bauer_comprehensive_20260923.run import require_gpu
from tests.training_restart_repair.data import rows,load_train,state
from tests.transition_block_repair.model import TransitionNet,rollout
from tests.temporal_source_repair.source import draw

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def chosen():
    return [next(r for r in rows() if r['parent']['family']==f and r['frame']==8)
            for f in ('tilted','interior_domain','single_wall')]


def tasks():
    return [dict(seed=s,repair=r,updates=6000,lr=lr) for s in (11,22,33)
            for r,lr in (('low_lr',3e-5),('higher_lr',3e-4))]


def checkpoint(seed):
    return ROOT/f'tests/training_restart_repair/train/{(11,22,33).index(seed)*3+1:02d}/checkpoint.pt'


def prepare():
    assert not (HERE/'source_lock.json').exists()
    dump(HERE/'protocol.json',dict(tasks=tasks(),selected=chosen(),
        question='Can the current two-frame model fit three fixed TRAIN configurations and their64 empirical futures? Does higher optimization step size improve fit or just overfit finite branches?',
        history='Earlier empirical_transport_repair used one positive initial state,8whole11frame paths and tau<=.95,not these3nonuniform intermediate states,current2frame network,64branches or fulltau1. Preserve those results; this is a changed supported fitting task,not a repeated proof of capacity.',
        control='Same restart_fm18000 checkpoint and Adam state per seed. Restrict FM training to3states(IDs1,17,33),all64existing TRAIN futures. Uniform state and branch sampling,batch32,6000updates. Only LR differs: cosine3e-5 or3e-4 ->3e-6. Both retain identical RNG/target schedule. No moments or Hamiltonian loss; no new LLG simulation.',
        times='Bauer openL16,T63.824849668K,unchanged Hamiltonian. Physical0.1ps transition;2frames. Uniform transporttau[0,1],shiftednormal2S0 latent. Flowmidpoint128/512 do not change physical duration.',
        evaluation='Fixed snapshots0,500,1500,3000,6000;256new generated branches per state atmid128;final also512 paired latent. Compare finite empirical TRAIN law with ddof0 target covariance separately from untouched128probe/fine futures with population ddof1 and wholebranch bootstrap100. Final checkpoint fixed,not selected ondev. Save every generated sample/checkpoint/time.',
        limitations='This concentrated fit intentionally discards broad training distribution. Successful fit is not broad dynamics/generalization,production repair or size/time validation. Three configurations cannot certify all capacity; failure does not isolate architecture from optimization. Empirical covariance fitting and physical population fitting are distinct.',
        coverage='Independent CPU diagnostic replays native6000condition schedules for3seeds fromadjacent12000,including72branchrefill; extracts radius1/2 SO3invariants and edge distance. Train-only feature scaling,then nearest-neighbor distance to replaybank vs bank+48restartstates. Link to prior independentdev state mean/cov errors; descriptive associations only,not threshold,causation or exhaustive sufficient statistics.',
        budget='6GPU jobs estimated20-60min each;3fat coverage jobs10-30min;6fat fitstats2-8min;aggregate<3min. GPUrtx4090,jobzrs-mag;no long monitoring. Short smoke<5min.',
        skills='Karpathy surgical tests; experimental-design pairedLR controls; matplotlib PNG diagnostics. Scientific Agent Skills https://doi.org/10.48550/arXiv.2609.00065'))
    lock=json.loads((ROOT/'tests/training_restart_repair/source_lock.json').read_text())
    for name,digest in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    _,hashes=load_train();lock.update(hashes)
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']+[checkpoint(s) for s in (11,22,33)]
    for p in files:lock[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


def verify():
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


@torch.no_grad()
def evaluate(net,out,step,smoke,final):
    tick=time.perf_counter();records=[]
    with h5py.File(out/f'paths{step:05d}.h5','x') as h:
        h.attrs['complete']=False;h.attrs['physical_duration_ps']=.1
        for row in chosen():
            sid=row['index'];n=4 if smoke else 256
            a=F.normalize(torch.tensor(state(row),device='cuda',dtype=torch.float32),dim=-1)[None].expand(n,-1,-1)
            g=h.create_group(f'state{sid:03d}');g.attrs['metadata']=json.dumps(row);g.create_dataset('time_ps',data=[0.,.1])
            levels=(8,16) if smoke and final else ((8,) if smoke else ((128,512) if final else (128,)))
            for steps in levels:
                chunks=[]
                for b in range(0,n,64):
                    gen=torch.Generator(device='cuda').manual_seed(946000+sid+b*10000)
                    chunks.append(rollout(net,a[b:b+64],gen,steps,1).cpu().numpy())
                x=np.concatenate(chunks);norm=float(abs(np.linalg.norm(x,axis=-1)-1).max());anchor=float(abs(x[:,0]-a.cpu().numpy()).max())
                assert np.isfinite(x).all() and norm<2e-6 and anchor==0
                g.create_dataset(f'midpoint{steps}',data=x,compression='lzf');records.append(dict(state=sid,steps=steps,norm_error=norm,anchor_error=anchor))
        h.attrs['complete']=True
    return dict(step=step,seconds=time.perf_counter()-tick,records=records)


def train(index,smoke):
    require_gpu('cuda');task=tasks()[index];ck=torch.load(checkpoint(task['seed']),map_location='cpu',weights_only=True)
    assert ck['task']['repair']=='restart_fm' and ck['task']['seed']==task['seed'] and ck['step']==18000
    net=TransitionNet().cuda();net.load_state_dict(ck['model']);opt=torch.optim.AdamW(net.parameters(),lr=task['lr'],weight_decay=1e-5);opt.load_state_dict(ck['optimizer'])
    bank,hashes=load_train();bank=np.stack([bank[r['index']] for r in chosen()])
    rng=np.random.default_rng(945000+task['seed']);gs={k:torch.Generator(device='cuda').manual_seed(b+task['seed']) for k,b in [('source',945100),('tau',945200),('cut',945300)]}
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    total=16 if smoke else 6000;stops={total} if smoke else {500,1500,3000,6000};history=[];events=[];stream=hashlib.sha256();seconds=0.
    net.eval();events.append(evaluate(net,out,0,smoke,False));net.train();tick=time.perf_counter()
    for step in range(1,total+1):
        ids=rng.integers(3,size=32);branches=rng.integers(64,size=32)
        y=F.normalize(torch.tensor(bank[ids,branches],device='cuda'),dim=-1)
        stream.update(ids.tobytes());stream.update(branches.tobytes())
        for k in gs:stream.update(gs[k].get_state().numpy().tobytes())
        z=draw(y[:,0],2,gs['source'],0.);tau=torch.rand(len(y),device='cuda',generator=gs['tau'])
        x,v,_=path_interpolate(z,y,tau,gs['cut']);p=net.velocity(x,tau,y[:,0],y.new_tensor([0.,.1]))
        loss=(p[:,1:]-v[:,1:]).square().sum(-1).mean();lr=3e-6+(task['lr']-3e-6)*.5*(1+math.cos(math.pi*(step-1)/max(1,total-1)))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        assert torch.isfinite(loss) and torch.isfinite(grad);opt.step()
        if smoke or step%100==0:history.append(dict(step=step,loss=float(loss.detach()),gradient=float(grad),lr=lr))
        if step in stops:
            torch.cuda.synchronize();seconds+=time.perf_counter()-tick
            torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step,task=task,
                rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state),out/f'step{step:05d}.pt')
            net.eval();events.append(evaluate(net,out,step,smoke,step==total));net.train();tick=time.perf_counter();print(task,step,flush=True)
    dump(out/'history.json',history);dump(out/'complete.json',dict(task=task,train_seconds=seconds,evaluation=events,paired_hash=stream.hexdigest(),
        data_hashes=hashes,device=torch.cuda.get_device_name(),scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:verify();train(a.index,a.action=='smoke')
