"""Paired FM continuation versus conditional mean/covariance assistance."""
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
from tests.bauer_p2_20260923.common import cfg, dump
from tests.bauer_comprehensive_20260923.run import require_gpu, real_data, next_batch
from tests.branch_coverage_repair.run import pools, parents as train_parents
from tests.sampling_balance_repair.run import RefillWindows
from tests.temporal_source_repair.source import draw
from tests.transition_block_repair.model import TransitionNet, pair_window, rollout
from tests.transition_block_repair.run import evaluate
from tests.intermediate_restart_repair.run import selected, initial
from .objectives import sample, moment_loss

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]


def tasks():
    return [dict(domain='bauer',arm='hierarchical',seed=s,repair=r,updates=6000)
            for s in (11,22,33) for r in ('fm','moments')]


def source_index(seed):
    return (11,22,33).index(seed)*2+1


def prepare():
    assert not (HERE/'source_lock.json').exists()
    dump(HERE/'protocol.json',dict(tasks=tasks(),
        hypothesis='Does directly supervising local conditional mean and full3x3 covariance repair the real-state transition bias, without worsening unoptimized path statistics?',
        controls='Same adjacent step12000 checkpoint and Adam state per seed,6000updates cosine3e-5->3e-6. Same native windows,72train-branch refill,source/tau/pair streams. FM-only concurrent control. Same updates not equal compute.',
        auxiliary='Every16updates,one uniformly chosen training parent among24L16,all72 existing TRAIN futures sliced0->.1ps,64 independent generated branches split32+32. Midpoint128 differentiable exactly matching deployment,activation checkpointing. Weight ramps linearly to.1 over1000updates. Cross-products of independent estimates avoid generated sample-variance shrinkage from naive squared moment estimates; stochastic loss may be negative. Target remains noisy finite72 reference moments,not exact population moments.',
        limitation='Auxiliary support only original t0 TRAIN states; no intermediate dev data trained. Tests whether this supervision transfers to real intermediate states; lack of transfer would leave conditional coverage unresolved. No new data generation. No Hamiltonian or analytic torque in loss. Local3x3 covariance is not all intersite correlations.',
        history='Older flow_training_repair and curriculum_repair optimized full-path scalar increment variance/time correlation with sphereEuler32/64 and did not establish accuracy. This changes to two-frame local vector mean/full covariance, unbiased generated-ensemble cross-products and deployment midpoint128; cannot attribute benefit to each ingredient separately.',
        physics='Unchanged Bauer J50meV,kappa.1,alpha.1,theta.11,T63.824849668K,open; trained lag.1ps,reference dt.25/.125fs. Rollout1ps10steps; no physical time relabelling.',
        evaluation='Fixed final checkpoint only. Reuse full18dev128paths midpoint128/512 and24seen64paths midpoint128 evaluator. Additionally72fixed real-LLG intermediate dev states(.3/.8ps)128futures midpoint128. New independent latent seeds paired across arms. Reference base/fine from previous diagnostic,never training. Whole-branch bootstrap100; fixed-parent estimands,not216independent parents. Three training seeds separately; no arbitrary pass criterion.',
        decision='Compare first mean/covariance/variance jointly,real intermediate states,and unoptimized energy/spatial/two-time/full1ps statistics. Improvement on training-only but not independent parents means no repaired generalization. Do not call successful execution physical certification.',
        budget='Smoke estimated2-6min;6GPU jobs estimated2-4h each including full sampling;6fat statistics5-15min;aggregate<3min. rtx4090 GPU,job zrs-mag,no long monitoring.',
        skill_use='Karpathy surgical changes; experimental-design paired controls and fixed estimands. Scientific Agent Skills https://doi.org/10.48550/arXiv.2609.00065'))
    lock=json.loads((ROOT/'tests/intermediate_restart_repair/source_lock.json').read_text())
    for name,digest in lock.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    _,_,hashes=pools(False);lock.update(hashes)
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']
    for f in files:lock[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


@torch.no_grad()
def intermediate(net,out,smoke):
    rows=[r for r in selected() if r['origin']=='llg']; rows=rows[:1] if smoke else rows
    with h5py.File(out/'intermediate.h5','x') as h:
        h.attrs['complete']=False
        for row in rows:
            sid=row['index'];n=4 if smoke else 128
            a=F.normalize(torch.tensor(initial(row),device='cuda',dtype=torch.float32),dim=-1)[None].expand(n,-1,-1)
            chunks=[]
            for b in range(0,n,64):
                gen=torch.Generator(device='cuda').manual_seed(941100+sid+b*10000)
                chunks.append(rollout(net,a[b:b+64],gen,8 if smoke else 128,1).cpu().numpy())
            x=np.concatenate(chunks);assert np.isfinite(x).all() and abs(np.linalg.norm(x,axis=-1)-1).max()<2e-6
            assert np.array_equal(x[:,0],a.cpu().numpy())
            g=h.create_group(f'state{sid:03d}');g.attrs['metadata']=json.dumps(row);g.create_dataset('spins',data=x,compression='lzf')
        h.attrs['complete']=True


def train(index,smoke):
    require_gpu('cuda');task=tasks()[index]
    ck=torch.load(ROOT/f'tests/transition_block_repair/train/{source_index(task["seed"]):02d}/step12000.pt',map_location='cpu',weights_only=True)
    assert ck['task']['seed']==task['seed'] and ck['task']['repair']=='adjacent'
    net=TransitionNet().cuda();net.load_state_dict(ck['model'])
    opt=torch.optim.AdamW(net.parameters(),lr=3e-5,weight_decay=1e-5);opt.load_state_dict(ck['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    refill=np.random.default_rng();refill.bit_generator.state=ck['refill_rng']
    pair_rng=np.random.default_rng();pair_rng.bit_generator.state=ck['pair_rng']
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    aux_rng=np.random.default_rng(942000+task['seed']);aux_g=torch.Generator(device='cuda').manual_seed(942100+task['seed'])
    _,new,hashes=pools(False);pool={r['parent_id']:x for r,x in zip(train_parents(),new)}
    audit=dict(counts=Counter(),hash=hashlib.sha256());config=cfg();data=[RefillWindows(d,pool,refill,True,audit) for d in real_data(config)]
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    total=16 if smoke else 6000;history=[];stream=hashlib.sha256();tick=time.perf_counter();net.train()
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
        if step%16==0:
            pid=int(aux_rng.integers(len(new)))
            ref=F.normalize(torch.tensor(new[pid][:,:2],device='cuda',dtype=torch.float32),dim=-1)
            a=ref[:1,0].expand(64,-1,-1);zaux=draw(a,2,aux_g,0.)
            if task['repair']=='moments':
                gen=sample(net,a,zaux);loss,parts=moment_loss(gen,ref)
                weight=.1*min(1.,step/1000);(weight*loss).backward()
                assert torch.isfinite(loss);record.update(parts,auxiliary=float(loss.detach()),weight=weight,parent=pid)
        lr=3e-6+(3e-5-3e-6)*.5*(1+math.cos(math.pi*(step-1)/max(1,total-1)))
        for group in opt.param_groups:group['lr']=lr
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        assert torch.isfinite(fm) and torch.isfinite(grad);opt.step();record.update(gradient=float(grad),lr=lr)
        if smoke or step%16==0 or step%100==0:history.append(record)
        if step%500==0:print(task,step,flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-tick
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=12000+total,
        rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state,
        pair_rng=pair_rng.bit_generator.state,aux_rng=aux_rng.bit_generator.state,aux_generator=aux_g.get_state()),out/'checkpoint.pt')
    dump(out/'history.json',history);net.eval();tick=time.perf_counter();ev=evaluate(net,out,smoke);intermediate(net,out,smoke)
    torch.cuda.synchronize();dump(out/'complete.json',dict(task=task,train_seconds=elapsed,evaluation_seconds=time.perf_counter()-tick,
        paired_input_hash=stream.hexdigest(),schedule_hash=audit['hash'].hexdigest(),data_hashes=hashes,evaluation=ev,
        device=torch.cuda.get_device_name(),scientific_certification=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();return
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    train(a.index,a.action=='smoke')


if __name__=='__main__':main()
