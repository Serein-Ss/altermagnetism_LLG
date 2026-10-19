"""Current state-replay FM versus deployed-sampler conditional-moment supervision."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from collections import Counter
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
from tests.transition_block_repair.model import pair_window
from tests.transition_block_repair.run import evaluate
from tests.conditional_moment_repair.run import intermediate
from tests.training_restart_repair.run import seen_restart
from tests.training_restart_repair.data import load_train,rows as real_rows
from tests.axial_symmetry_repair.symmetry import rotate,probe
from tests.conditional_coupling_repair.coupling import schedule
from tests.conditional_teacher_repair.target import labels
from tests.training_restart_from_scratch.recipe import initialize
from tests.model_state_refresh_repair.run import verify as verify_previous
from tests.model_state_coverage_repair import data as stale
from tests.model_state_refresh_repair import data as visited
from tests.transition_block_repair.model import rollout
from tests.model_state_refresh_repair.recipe import replay_roles,rate
from tests.conditional_moment_repair.objectives import sample,moment_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ('control', 'moments_0p1', 'moments_0p5')


def tasks():
    return [dict(domain='bauer', arm='hierarchical', seed=s, repair=r, updates=6000)
            for s in (11, 22, 33) for r in ARMS]


def source(seed):
    return ROOT / f'tests/model_state_refresh_repair/train/{(11,22,33).index(seed)*2+1:02d}'


def augment(x,v,initial,generator,enabled):
    angles=torch.rand(len(x),device=x.device,dtype=x.dtype,generator=generator)*(2*math.pi)
    if not enabled:return x,v,initial
    return rotate(x,angles),rotate(v,angles),rotate(initial,angles)


def banks(smoke=False):
    real,hashes=load_train(smoke);old=stale.load_train(smoke);new=visited.load_train(smoke)
    for before,after in zip(real_rows(smoke),visited.rows(smoke)):
        assert before['parent']==after['parent'] and before['index']==after['index'] and before['parent']['split']=='train'
        for module in (stale,visited):
            sid=after['index'];path=module.HERE/'reference'/f"{after['parent_index']:02d}"/f'state{sid:03d}_train.h5'
            hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(real)==len(old)==len(new)==(2 if smoke else 48)
    return (real,old,new),hashes

def prepare():
    assert not (HERE/'source_lock.json').exists();verify_previous()
    _,hashes=banks(False)
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json',ROOT/'tests/conditional_moment_repair/objectives.py']
    for seed in (11,22,33):files+=[source(seed)/'checkpoint.pt',source(seed)/'report.json']
    for path in files:hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',hashes);(HERE/'logs').mkdir(exist_ok=True)

def verify():
    verify_previous()
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name

@torch.no_grad()
def model_states(net,out,smoke,filename='model_states.h5'):
    with h5py.File(out/filename,'x') as h:
        h.attrs['complete']=False
        for row in visited.rows(smoke):
            sid=row['index'];n=4 if smoke else 64
            a=F.normalize(torch.tensor(visited.state(row),device='cuda',dtype=torch.float32),dim=-1)[None].expand(n,-1,-1)
            gen=torch.Generator(device='cuda').manual_seed(966000+sid)
            x=rollout(net,a,gen,8 if smoke else 128,1).cpu().numpy()
            assert np.isfinite(x).all() and abs(np.linalg.norm(x,axis=-1)-1).max()<2e-6 and np.array_equal(x[:,0],a.cpu().numpy())
            g=h.create_group(f'state{sid:03d}');g.attrs['metadata']=json.dumps(row)
            g.create_dataset('spins',data=x,compression='lzf');g.create_dataset('time_ps',data=[0.,.1])
        h.attrs['complete']=True


def train(index, smoke):
    require_gpu('cuda'); task=tasks()[index]; seed=task['seed']
    ck=torch.load(source(seed)/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert ck['step']==84144 and ck['task']['seed']==seed and ck['task']['repair']=='refreshed_replay'
    net,opt=initialize(ck,'warm',seed,'cuda')
    assert {id(p) for group in opt.param_groups for p in group['params']}=={id(p) for p in net.parameters()}
    assert all(p.is_cuda for group in opt.param_groups for p in group['params'])
    assert not opt.state
    initial_hash=hashlib.sha256()
    for p in net.parameters():initial_hash.update(p.detach().cpu().numpy().tobytes())
    aug_g=torch.Generator(device='cuda').manual_seed(971100+seed)
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    refill=np.random.default_rng();refill.bit_generator.state=ck['refill_rng']
    pair_rng=np.random.default_rng();pair_rng.bit_generator.state=ck['pair_rng']
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    extra_g=torch.Generator(device='cuda').manual_seed(971200+seed)
    aux_g=torch.Generator(device='cuda').manual_seed(971400+seed);aux_rng=np.random.default_rng(971500+seed)
    auxiliary_hash=hashlib.sha256();auxiliary_records=[]
    if not smoke:assert json.loads((visited.HERE/'data_audit.json').read_text())['passed']
    bank,bank_hashes=banks(smoke)
    bank_gpu=[[F.normalize(torch.tensor(b,device='cuda'),dim=-1) for b in role] for role in bank]
    
    _,new,hashes=pools(False);pool={r['parent_id']:x for r,x in zip(train_parents(),new)}
    audit=dict(counts=Counter(),hash=hashlib.sha256());config=cfg()
    data=[RefillWindows(d,pool,refill,True,audit) for d in real_data(config)]
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    total=16 if smoke else 6000;history=[];stream=hashlib.sha256();noise_hash=hashlib.sha256()
    origin=84144
    state_ids,branch_ids=schedule(971300+seed,total)
    if smoke:state_ids=state_ids%2
    exposures=np.zeros((3,48,64),dtype=np.int64)
    net.eval();tick=time.perf_counter();model_states(net,out,smoke,'before_model_states.h5');before_seconds=time.perf_counter()-tick
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
            roles=replay_roles(step,'refreshed_replay')
            np.add.at(exposures,(roles,ids,branches),1);noise_hash.update(extra_g.get_state().numpy().tobytes())
            yr=F.normalize(torch.tensor(np.stack([bank[r][i][j] for r,i,j in zip(roles,ids,branches)]),device='cuda'),dim=-1)
            zr=draw(yr[:,0],2,extra_g,0.)
            tr=torch.rand(len(yr),device='cuda',generator=extra_g)
            drafts=[];sampled=[];conditional=[];diagnostics=[]
            for start in (0,16,32):
                xx,vv,rr,dd=labels(zr[start:start+16],yr[start:start+16],tr[start:start+16],
                    bank_gpu[roles[start]][ids[start]],extra_g)
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
        if step%(4 if smoke else 32)==0:
            role=2 if len(auxiliary_records)%2 else 0
            sid=int(aux_rng.integers(len(bank[role])));target=bank_gpu[role][sid].detach()
            initial=target[:1,0].expand(64,-1,-1)
            latent=draw(initial,2,aux_g,0.)
            auxiliary_hash.update(np.array([role,sid],dtype=np.int64).tobytes())
            auxiliary_hash.update(latent.detach().cpu().numpy().tobytes())
            active=task['repair']!='control'
            before=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in net.parameters()] if active else None
            torch.cuda.synchronize();began=time.perf_counter()
            with torch.set_grad_enabled(active):
                generated=sample(net,initial,latent,128,active)
                auxiliary,parts=moment_loss(generated,target)
            assert torch.isfinite(auxiliary)
            norm=float((generated.detach().norm(dim=-1)-1).abs().max())
            assert norm<2e-6 and torch.equal(generated[:,0],initial)
            weight=(.5 if task['repair']=='moments_0p5' else .1)*min(1.,step/(16 if smoke else 1000))
            gradient=0.
            if active:
                (weight*auxiliary).backward()
                gradient=float(torch.stack([((p.grad-before[i]) if p.grad is not None else -before[i]).square().sum() for i,p in enumerate(net.parameters())]).sum().sqrt())
                assert math.isfinite(gradient)
            torch.cuda.synchronize()
            auxiliary_records.append(dict(step=step,role=role,state=sid,active=active,weight=weight,
                loss=float(auxiliary.detach()),gradient_norm=gradient,seconds=time.perf_counter()-began,norm_error=norm,**parts))
            record.update(auxiliary=float(auxiliary.detach()),auxiliary_active=active,auxiliary_gradient=gradient)
        lr=rate(step,total)
        for group in opt.param_groups:group['lr']=lr
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(fm) and torch.isfinite(grad)
        opt.step();record.update(gradient=float(grad),lr=lr)
        if smoke or step%100==0:history.append(record)
        if step%500==0:print(task,step,flush=True)
    torch.cuda.synchronize();training=time.perf_counter()-tick
    assert exposures.sum()==total*48
    assert exposures[0].sum()==total*24
    assert exposures[1].sum()==0 and exposures[2].sum()==total*24
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=origin+total,
        training_updates=total,optimizer_reset=True,rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state,
        pair_rng=pair_rng.bit_generator.state,extra_generator=extra_g.get_state(),augmentation_generator=aug_g.get_state(),auxiliary_generator=aux_g.get_state(),auxiliary_rng=aux_rng.bit_generator.state),out/'checkpoint.pt')
    dump(out/'history.json',history);dump(out/'exposures.json',exposures.tolist());dump(out/'auxiliary.json',auxiliary_records)
    net.eval()
    optimizer_steps=[sorted({int(opt.state[p]['step']) for p in group['params'] if p in opt.state}) for group in opt.param_groups]
    dump(out/'symmetry.json',probe(net));tick=time.perf_counter();seen_restart(net,out/'after_restart.h5',smoke)
    ev=evaluate(net,out,smoke);intermediate(net,out,smoke);model_states(net,out,smoke);torch.cuda.synchronize()
    dump(out/'complete.json',dict(task=task,step=origin+total,train_seconds=training,evaluation_seconds=time.perf_counter()-tick,
        training_updates=total,optimizer_reset=True,before_model_state_seconds=before_seconds,initial_hash=initial_hash.hexdigest(),optimizer_steps_by_group=optimizer_steps,paired_input_hash=stream.hexdigest(),extra_noise_hash=noise_hash.hexdigest(),schedule_hash=audit['hash'].hexdigest(),
        auxiliary_input_hash=auxiliary_hash.hexdigest(),auxiliary_calls=len(auxiliary_records),
        auxiliary_seconds=sum(r['seconds'] for r in auxiliary_records),group_schedule_hash=hashlib.sha256(state_ids.tobytes()+branch_ids.tobytes()).hexdigest(),data_hashes=hashes,restart_data_hashes=bank_hashes,evaluation=ev,device=torch.cuda.get_device_name(),parameter_count=sum(p.numel() for p in net.parameters()),scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:verify();train(a.index,a.action=='smoke')
