"""Matched per-family exposure: mixed replay versus single-family continuation."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.bauer_p2_20260923.common import dump
from tests.bauer_comprehensive_20260923.run import require_gpu
from tests.temporal_source_repair.source import draw
from tests.transition_block_repair.model import TransitionNet
from tests.transition_block_repair.run import evaluate
from tests.conditional_moment_repair.run import intermediate
from tests.training_restart_repair.run import seen_restart
from tests.training_restart_repair.data import load_train,rows
from tests.local_condition_repair.run import verify as verify_sources

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ARMS=('mixed','tilted','interior_domain','single_wall')


def tasks():
    return [dict(domain='bauer',arm='hierarchical',seed=s,repair=a,updates=6000) for s in (11,22,33) for a in ARMS]


def source(seed):
    return ROOT/f'tests/local_condition_repair/train/{(11,22,33).index(seed)*3:02d}'


def mask(arm,metadata):
    if arm not in ARMS:raise ValueError(arm)
    return np.array([arm=='mixed' or r['parent']['family']==arm for r in metadata])


def loss(pred,target):
    # Keep the target family's contribution equal to its contribution in mixed.
    # Sum of the three specialist losses equals mixed at identical weights/data.
    return (pred[:,1:]-target[:,1:]).square().sum(-1).mean((1,2)).sum()/48


def prepare():
    assert not (HERE/'source_lock.json').exists()
    verify_sources();_,hashes=load_train();meta=rows()
    from tests.intermediate_restart_repair.run import selected
    dev=[r for r in selected() if r['origin']=='llg']
    assert not {r['parent']['parent_id'] for r in meta}&{r['parent']['parent_id'] for r in dev}
    assert len(meta)==48 and all(mask(a,meta).sum()==16 for a in ARMS[1:])
    dump(HERE/'protocol.json',dict(tasks=tasks(),
        hypothesis='Does removing other preparation families improve held-out conditional dynamics with matched target-family exposure?',
        source='All arms start per-seed local_condition baseline step36000, unchanged TransitionNet and restored Adam. No readout adapters, no best-dev source selection.',
        data='Existing48L16TRAINrestart states,16perfamily from8parents,each64TRAINfutures. Three families total24parents. No new LLG data, no outcome selection, no dev/fine/probe in loss. Family is preparation provenance, not inferred event label.',
        controls='Each update draws one future for all48states with paired latent/tau/interpolation RNG, then selects16or48. Each participating state appears6000times. Specialist loss=sum per-state FM/48, mixed=sum48/48; matched family contribution, not renormalized specialist mean. Old Adam restored,cos3e-5->3e-6,clip10,weightdecay1e-5. All arms omit native windows to isolate replay-family interference.',
        limits='Specialist removes two loss components; adaptive Adam/clipping changes effective updates. Not proof of persistent gradient conflict. Same updates and target exposure, NOT same total examples/compute. Specialist16*64=1024unique available futures,96000exposures;mixed3072unique,288000exposures. Pretraining remains mixed; not single-family from scratch. Family label never a neural input or inference router.',
        evaluation='Freeze final42000. All models evaluated on all families,not only their chosen family:48TRAINorigin independent probes,72LLG DEVstates,18DEV128paths1psmidpoint128/512,24oldseen64paths. L16 primary in-size, L25/32 reused diagnostic targets not fresh transfer certification. Wholebranchbootstrap100;descriptive family averages,three trainingseeds;no arbitrary acceptance threshold.',
        physics='Unchanged Bauer open,J50meV,kappa.1,alpha.1,theta.11,T63.824849668K. Learnedlag.1ps;reference dt.25/.125fs;1ps rollout10fresh-noise transitions. No Hamiltonian input or time relabel.',
        decision='Specialist helps held-out same-family without covariance/time failure: supports specialization benefit under this recipe,not unique cause. Seen-only gains leave generalization unresolved. No gains motivates representation/objective rather than another blind budget increase. Cross-family regression disclosed;no production promotion.',
        budget='Four GPU smoke1-4min each.12formalGPU2-3h each including evaluation;12fatstats3-6min;aggregate<3min. GPUrtx4090,zrs-mag. No long monitoring.',
        skill_use='karpathy-guidelines and experimental-design:paired target exposure and explicit unequal total compute.'))
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']
    for seed in (11,22,33):files += [source(seed)/'checkpoint.pt',source(seed)/'after_restart.h5']
    for p in files:hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',hashes);(HERE/'logs').mkdir(exist_ok=True)


def verify():
    verify_sources()
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


def train(index,smoke):
    require_gpu('cuda');task=tasks()[index];seed=task['seed']
    ck=torch.load(source(seed)/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert ck['step']==36000 and ck['task']['repair']=='baseline' and ck['task']['seed']==seed
    net=TransitionNet().cuda();net.load_state_dict(ck['model'])
    opt=torch.optim.AdamW(net.parameters(),lr=3e-5,weight_decay=1e-5);opt.load_state_dict(ck['optimizer'])
    bank,hashes=load_train();meta=rows();active=mask(task['repair'],meta);select=torch.tensor(active,device='cuda')
    rng=np.random.default_rng(951000+seed);g=torch.Generator(device='cuda').manual_seed(951100+seed)
    out=HERE/('smoke' if smoke else 'train')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    total=16 if smoke else 6000;history=[];exposures=np.zeros((48,64),dtype=np.int64)
    stream=hashlib.sha256();net.train();tick=time.perf_counter()
    for step in range(1,total+1):
        branches=rng.integers(64,size=48)
        y=F.normalize(torch.tensor(np.stack([b[j] for b,j in zip(bank,branches)]),device='cuda'),dim=-1)
        stream.update(branches.tobytes());stream.update(g.get_state().numpy().tobytes())
        z=draw(y[:,0],2,g,0.);tau=torch.rand(48,device='cuda',generator=g);x,v,_=path_interpolate(z,y,tau,g)
        np.add.at(exposures,(np.flatnonzero(active),branches[active]),1)
        opt.zero_grad(set_to_none=True)
        pred=net.velocity(x[select],tau[select],y[select,0],y.new_tensor([0.,.1]));fm=loss(pred,v[select]);fm.backward()
        lr=3e-6+(3e-5-3e-6)*.5*(1+math.cos(math.pi*(step-1)/max(1,total-1)))
        for group in opt.param_groups:group['lr']=lr
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.);assert torch.isfinite(fm) and torch.isfinite(grad)
        opt.step()
        if smoke or step%100==0:history.append(dict(step=step,weighted_fm=float(fm.detach()),family_mean_fm=float(fm.detach())*48/int(active.sum()),gradient=float(grad),lr=lr))
        if step%500==0:print(task,step,flush=True)
    torch.cuda.synchronize();training=time.perf_counter()-tick
    assert (exposures.sum(1)[active]==total).all() and not exposures[~active].any()
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=36000+total,
        data_rng=rng.bit_generator.state,generator=g.get_state(),specialization=task['repair']),out/'checkpoint.pt')
    dump(out/'history.json',history);dump(out/'exposures.json',exposures.tolist())
    net.eval();tick=time.perf_counter();seen_restart(net,out/'after_restart.h5',smoke)
    ev=evaluate(net,out,smoke);intermediate(net,out,smoke);torch.cuda.synchronize()
    dump(out/'complete.json',dict(task=task,step=36000+total,train_seconds=training,evaluation_seconds=time.perf_counter()-tick,
        paired_input_hash=stream.hexdigest(),data_hashes=hashes,evaluation=ev,active_states=int(active.sum()),
        device=torch.cuda.get_device_name(),parameter_count=sum(p.numel() for p in net.parameters()),scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:verify();train(a.index,a.action=='smoke')
