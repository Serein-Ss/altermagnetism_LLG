"""Three paired controls separating added restart FM and moment assistance."""
import argparse
import hashlib
import json
import math
import time
from collections import Counter
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_p2_20260923.common import cfg,dump
from tests.bauer_comprehensive_20260923.run import require_gpu,real_data,next_batch
from tests.branch_coverage_repair.run import pools,parents as train_parents
from tests.sampling_balance_repair.run import RefillWindows
from tests.temporal_source_repair.source import draw
from tests.transition_block_repair.model import TransitionNet,pair_window,rollout
from tests.transition_block_repair.run import evaluate
from tests.conditional_moment_repair.run import source_index,intermediate
from tests.conditional_moment_repair.objectives import sample,moment_loss
from .data import HERE,ROOT,rows,state,load_train,verify_sources


def tasks():
    return [dict(domain='bauer',arm='hierarchical',seed=s,repair=r,updates=6000)
            for s in (11,22,33) for r in ('native','restart_fm','restart_moments')]


def restart_batch(bank,rng,n=8):
    ids=rng.integers(len(bank),size=n);branches=rng.integers(64,size=n)
    return np.stack([bank[i][j] for i,j in zip(ids,branches)]),ids,branches


@torch.no_grad()
def seen_restart(net,out,smoke):
    with h5py.File(out,'x') as h:
        h.attrs['complete']=False
        for row in rows(smoke):
            sid=row['index'];n=4 if smoke else 64
            a=F.normalize(torch.tensor(state(row),device='cuda',dtype=torch.float32),dim=-1)[None].expand(n,-1,-1)
            gen=torch.Generator(device='cuda').manual_seed(944000+sid)
            x=rollout(net,a,gen,8 if smoke else 128,1).cpu().numpy()
            assert np.isfinite(x).all() and abs(np.linalg.norm(x,axis=-1)-1).max()<2e-6 and np.array_equal(x[:,0],a.cpu().numpy())
            g=h.create_group(f'state{sid:03d}');g.attrs['metadata']=json.dumps(row);g.create_dataset('spins',data=x,compression='lzf')
        h.attrs['complete']=True


def train(index,smoke):
    require_gpu('cuda');task=tasks()[index];seed=task['seed']
    if not smoke:assert json.loads((HERE/'data_audit.json').read_text())['passed']
    ck=torch.load(ROOT/f'tests/transition_block_repair/train/{source_index(seed):02d}/step12000.pt',map_location='cpu',weights_only=True)
    assert ck['task']['seed']==seed and ck['task']['repair']=='adjacent' and ck['step']==12000
    net=TransitionNet().cuda();net.load_state_dict(ck['model'])
    opt=torch.optim.AdamW(net.parameters(),lr=3e-5,weight_decay=1e-5);opt.load_state_dict(ck['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    refill=np.random.default_rng();refill.bit_generator.state=ck['refill_rng']
    pair_rng=np.random.default_rng();pair_rng.bit_generator.state=ck['pair_rng']
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    extra_rng=np.random.default_rng(943000+seed);aux_rng=np.random.default_rng(943100+seed)
    extra_g=torch.Generator(device='cuda').manual_seed(943200+seed)
    aux_g=torch.Generator(device='cuda').manual_seed(943300+seed)
    bank,bank_hashes=load_train(smoke);_,new,hashes=pools(False)
    pool={r['parent_id']:x for r,x in zip(train_parents(),new)}
    audit=dict(counts=Counter(),hash=hashlib.sha256());config=cfg();data=[RefillWindows(d,pool,refill,True,audit) for d in real_data(config)]
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    net.eval();tick=time.perf_counter();seen_restart(net,out/'before_restart.h5',smoke);before_seconds=time.perf_counter()-tick
    total=16 if smoke else 6000;history=[];stream=hashlib.sha256();extra_hash=hashlib.sha256();tick=time.perf_counter();net.train()
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
        if step%2==0:
            raw,ids,branches=restart_batch(bank,extra_rng);extra_hash.update(raw.tobytes());extra_hash.update(extra_g.get_state().numpy().tobytes())
            yr=F.normalize(torch.tensor(raw,device='cuda'),dim=-1)
            zr=draw(yr[:,0],2,extra_g,0.);tr=torch.rand(len(yr),device='cuda',generator=extra_g)
            xr,vr,_=path_interpolate(zr,yr,tr,extra_g)
            if task['repair']!='native':
                pr=net.velocity(xr,tr,yr[:,0],yr.new_tensor([0.,.1]));extra=(pr[:,1:]-vr[:,1:]).square().sum(-1).mean()
                (.5*extra).backward();assert torch.isfinite(extra);record['restart_fm']=float(extra.detach())
        if step%16==0:
            sid=int(aux_rng.integers(len(bank)));ref=F.normalize(torch.tensor(bank[sid],device='cuda'),dim=-1)
            a=ref[:1,0].expand(64,-1,-1);za=draw(a,2,aux_g,0.)
            if task['repair']=='restart_moments':
                loss,parts=moment_loss(sample(net,a,za),ref);weight=.1*min(1.,step/1000)
                (weight*loss).backward();assert torch.isfinite(loss);record.update(parts,auxiliary=float(loss.detach()),weight=weight,state=sid)
        lr=3e-6+(3e-5-3e-6)*.5*(1+math.cos(math.pi*(step-1)/max(1,total-1)))
        for group in opt.param_groups:group['lr']=lr
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(fm) and torch.isfinite(grad)
        opt.step();record.update(gradient=float(grad),lr=lr)
        if smoke or step%16==0 or step%100==0:history.append(record)
        if step%500==0:print(task,step,flush=True)
    torch.cuda.synchronize();training=time.perf_counter()-tick
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=12000+total,
        rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state,pair_rng=pair_rng.bit_generator.state,
        extra_rng=extra_rng.bit_generator.state,aux_rng=aux_rng.bit_generator.state,extra_generator=extra_g.get_state(),aux_generator=aux_g.get_state()),out/'checkpoint.pt')
    dump(out/'history.json',history);net.eval();tick=time.perf_counter()
    seen_restart(net,out/'after_restart.h5',smoke);ev=evaluate(net,out,smoke);intermediate(net,out,smoke)
    torch.cuda.synchronize();dump(out/'complete.json',dict(task=task,train_seconds=training,before_evaluation_seconds=before_seconds,evaluation_seconds=time.perf_counter()-tick,
        paired_input_hash=stream.hexdigest(),extra_input_hash=extra_hash.hexdigest(),schedule_hash=audit['hash'].hexdigest(),
        data_hashes=hashes,restart_data_hashes=bank_hashes,evaluation=ev,device=torch.cuda.get_device_name(),scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    verify_sources();train(a.index,a.action=='smoke')
