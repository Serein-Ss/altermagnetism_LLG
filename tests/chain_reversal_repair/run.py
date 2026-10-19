import argparse,hashlib,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_p2_20260923.common import cfg,dump
from tests.deployment_moment_repair.run import banks,augment,model_states
from tests.continuous_axial_repair.run import verify as verify_previous
from tests.bauer_comprehensive_20260923.run import require_gpu,real_data,next_batch
from tests.branch_coverage_repair.run import pools,parents
from tests.sampling_balance_repair.run import RefillWindows
from tests.temporal_source_repair.source import draw
from tests.transition_block_repair.model import pair_window
from tests.transition_block_repair.run import evaluate
from tests.conditional_moment_repair.run import intermediate
from tests.training_restart_repair.run import seen_restart
from tests.axial_symmetry_repair.symmetry import rotate,probe
from tests.conditional_coupling_repair.coupling import schedule
from tests.conditional_teacher_repair.target import labels
from .model import initialize,ARMS,gauge_probe,reversal_probe
from tests.model_state_refresh_repair.recipe import replay_roles,rate
from tests.replay_weight_repair.recipe import add_replay

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def tasks():
    return [dict(domain='bauer',arm='hierarchical',seed=s,repair=r,updates=6000)
            for s in (11,22,33) for r in ARMS]


def source(seed):
    return ROOT/f'tests/continuous_axial_repair/train/{(11,22,33).index(seed)*2+1:02d}'


def prepare():
    assert not (HERE/'source_lock.json').exists();verify_previous()
    assert json.loads((HERE/'audit.json').read_text())['all_supported']
    _,hashes=banks()
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json',HERE/'audit.json',HERE/'initial_probe.json',ROOT/'tests/full_training_support/assets/summary.json',ROOT/'tests/joint_uncertainty_repair/metric.py']
    for seed in (11,22,33):files += [source(seed)/'checkpoint.pt',source(seed)/'report.json']
    for p in files:hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',hashes)


def verify():
    verify_previous()
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


def train(index,smoke):
    require_gpu('cuda');task=tasks()[index];seed=task['seed'];weight=.25;arm=task['repair']
    ck=torch.load(source(seed)/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert ck['step']==114144 and ck['task']['seed']==seed and ck['task']['repair']=='gauge2'
    net,opt=initialize(ck,arm,seed,'cuda');assert not opt.state
    initial_hash=hashlib.sha256()
    for p in net.parameters():initial_hash.update(p.detach().cpu().numpy().tobytes())
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    refill=np.random.default_rng();refill.bit_generator.state=ck['refill_rng']
    pair_rng=np.random.default_rng();pair_rng.bit_generator.state=ck['pair_rng']
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    aug=torch.Generator(device='cuda').manual_seed(981100+seed)
    extra_g=torch.Generator(device='cuda').manual_seed(981200+seed)
    bank,bank_hashes=banks(smoke);gpu_bank=[[F.normalize(torch.tensor(b,device='cuda'),dim=-1) for b in role] for role in bank]
    _,new,hashes=pools(False);pool={r['parent_id']:x for r,x in zip(parents(),new)}
    audit=dict(counts=Counter(),hash=hashlib.sha256());config=cfg()
    data=[RefillWindows(d,pool,refill,True,audit) for d in real_data(config)]
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    total=16 if smoke else 6000;ids,branches=schedule(981300+seed,total)
    if smoke:ids=ids%2
    stream=hashlib.sha256();noise=hashlib.sha256();history=[];gradient=[];exposures=np.zeros((3,48,64),dtype=np.int64)
    tick=time.perf_counter();net.train()
    for step in range(1,total+1):
        whole,_=next_batch(task,config,data,rng,gs['target'],'cuda');stream.update(whole.detach().cpu().numpy().tobytes())
        for k in ('source','tau'):stream.update(gs[k].get_state().numpy().tobytes())
        starts=np.where(pair_rng.random(len(whole))<.5,0,pair_rng.integers(1,10,len(whole)))
        y=pair_window(whole,torch.as_tensor(starts,device='cuda'));z=draw(y[:,0],11,gs['source'],0.)[:,:2]
        tau=torch.rand(len(y),device='cuda',generator=gs['tau'])
        cut=gs['cut'].get_state();torch.randn(whole.shape,device='cuda',generator=gs['cut'])
        x,v,_=path_interpolate(z,y,tau,torch.Generator(device='cuda').set_state(cut))
        x,v,anchor=augment(x,v,y[:,0],aug,True)
        pred=net.velocity(x,tau,anchor,y.new_tensor([0.,.1]));fm=(pred[:,1:]-v[:,1:]).square().sum(-1).mean()
        opt.zero_grad(set_to_none=True);fm.backward()
        roles=replay_roles(step,'refreshed_replay');ii,jj=ids[step-1],branches[step-1]
        np.add.at(exposures,(roles,ii,jj),1);noise.update(extra_g.get_state().numpy().tobytes())
        yr=F.normalize(torch.tensor(np.stack([bank[r][i][j] for r,i,j in zip(roles,ii,jj)]),device='cuda'),dim=-1)
        zr=draw(yr[:,0],2,extra_g,0.);tr=torch.rand(len(yr),device='cuda',generator=extra_g)
        drafts=[];targets=[]
        for start in (0,16,32):
            xx,_,vv,_=labels(zr[start:start+16],yr[start:start+16],tr[start:start+16],gpu_bank[roles[start]][ii[start]],extra_g)
            drafts.append(xx);targets.append(vv)
        xr=torch.cat(drafts);vr=torch.cat(targets)
        angles=torch.rand(3,device='cuda',generator=aug).repeat_interleave(16)*(2*np.pi)
        anchor=rotate(xr[:,0],angles);xr,vr=rotate(xr,angles),rotate(vr,angles)
        pr=net.velocity(xr,tr,anchor,yr.new_tensor([0.,.1]));extra=(pr[:,1:]-vr[:,1:]).square().sum(-1).mean()
        stats=add_replay(net.parameters(),extra,weight,smoke or step==1 or step%128==0)
        if stats is not None:gradient.append(dict(step=step,**stats))
        lr=rate(step,total)
        for group in opt.param_groups:group['lr']=lr
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        assert torch.isfinite(fm) and torch.isfinite(extra) and torch.isfinite(grad)
        opt.step()
        if smoke or step%100==0:history.append(dict(step=step,fm=float(fm.detach()),replay_fm=float(extra.detach()),weight=weight,gradient=float(grad),lr=lr))
        if step%500==0:print(task,step,flush=True)
    torch.cuda.synchronize();training=time.perf_counter()-tick
    assert exposures.sum()==total*48 and exposures[0].sum()==exposures[2].sum()==total*24 and exposures[1].sum()==0
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=114144+total,
        rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,refill_rng=refill.bit_generator.state,
        pair_rng=pair_rng.bit_generator.state,extra_generator=extra_g.get_state(),augmentation_generator=aug.get_state()),out/'checkpoint.pt')
    dump(out/'history.json',history);dump(out/'gradients.json',gradient);dump(out/'exposures.json',exposures.tolist())
    net.eval();dump(out/'symmetry.json',probe(net));dump(out/'gauge.json',gauge_probe(net));dump(out/'reversal.json',reversal_probe(net));tick=time.perf_counter()
    seen_restart(net,out/'after_restart.h5',smoke);ev=evaluate(net,out,smoke);intermediate(net,out,smoke);model_states(net,out,smoke)
    torch.cuda.synchronize()
    dump(out/'complete.json',dict(task=task,weight=weight,gauge_arm=arm,step=114144+total,optimizer_reset=True,training_updates=total,
        train_seconds=training,evaluation_seconds=time.perf_counter()-tick,initial_hash=initial_hash.hexdigest(),paired_input_hash=stream.hexdigest(),
        extra_noise_hash=noise.hexdigest(),schedule_hash=audit['hash'].hexdigest(),group_schedule_hash=hashlib.sha256(ids.tobytes()+branches.tobytes()).hexdigest(),
        data_hashes=hashes,restart_data_hashes=bank_hashes,native_counts=dict(audit['counts']),evaluation=ev,device=torch.cuda.get_device_name(),
        parameter_count=sum(p.numel() for p in net.parameters()),scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:verify();train(a.index,a.action=='smoke')
