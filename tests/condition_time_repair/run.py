"""Paired model-level repair. Existing reference futures remain evaluation-only."""
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
from tests.bauer_p2_20260923.common import cfg,dump
from tests.bauer_comprehensive_20260923.run import real_data,next_batch,require_gpu
from tests.curriculum_repair.run import fm_loss
from tests.spin_path_transformer.run import make_net
from tests.moment_bias_repair.run import evaluate
from tests.reference_resolution.run import checkpoint,parents,reference_inputs
from tests.transport_audit.run import integrate
from tests.cross_dimension_20260924.model import source
from .model import RepairedNet

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def tasks():
    rows=[dict(domain='bauer',arm='hierarchical',seed=s,repair=r,updates=6000)
          for s in (11,22,33) for r in ('control','anchor','relative_time')]
    np.random.default_rng(93301).shuffle(rows);return rows


def build(task,device='cuda'):
    ck=torch.load(checkpoint(task['seed']),map_location='cpu',weights_only=True)
    assert ck['step']==18000 and ck['task']['repair']=='control'
    base=make_net(task,device);base.load_state_dict(ck['model'])
    opt=torch.optim.AdamW(base.parameters(),lr=.0003,weight_decay=1e-5);opt.load_state_dict(ck['optimizer'])
    old_ids={id(p) for p in base.parameters()}
    torch.manual_seed(task['seed']+30000)
    net=RepairedNet(base,task['repair']);added=[p for p in net.parameters() if id(p) not in old_ids]
    if added:opt.add_param_group(dict(params=added,lr=.0003))
    return net,opt,ck


@torch.no_grad()
def expanded(net,out,smoke):
    records=[]
    with h5py.File(out/'expanded.h5','x') as h:
        h.attrs['complete']=False
        for i,row in enumerate(parents()[:1] if smoke else parents()):
            initial,_=reference_inputs(row);n=4 if smoke else 256
            a=F.normalize(torch.tensor(initial,device='cuda',dtype=torch.float32),dim=-1)[None].expand(n,-1,-1)
            z=source(a,11,torch.Generator(device='cuda').manual_seed(932100+i),2.)
            times=torch.arange(11,device='cuda')*.1;group=h.create_group(f'parent{i:02d}');group.attrs['parent']=json.dumps(row)
            group.create_dataset('time_ps',data=np.arange(11)*.1)
            # Both solvers retained; midpoint is not silently promoted to sole evaluator.
            for method,steps in ([('euler',8),('midpoint',8)] if smoke else [('euler',128),('midpoint',128)]):
                torch.cuda.synchronize();tick=time.perf_counter();xs=[]
                for j in range(0,n,64):
                    init=a[j:j+64]
                    xs.append(integrate(lambda x,t:net.velocity(x,t,init,times),z[j:j+64],0.,1.,steps,method))
                x=torch.cat(xs);torch.cuda.synchronize()
                assert torch.isfinite(x).all() and (x.norm(dim=-1)-1).abs().max()<2e-6 and torch.equal(x[:,0],a)
                group.create_dataset(f'{method}{steps}',data=x.cpu().numpy(),compression='lzf')
                records.append(dict(parent=i,sampler=f'{method}{steps}',seconds=time.perf_counter()-tick))
        h.attrs['complete']=True
    return records


def prepare():
    if (HERE/'source_lock.json').exists():raise RuntimeError('Already frozen')
    lock=json.loads((ROOT/'tests/reference_resolution/source_lock.json').read_text())
    for name,digest in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    dump(HERE/'protocol.json',dict(tasks=tasks(),
        scope='Existing Bauer open L16/25/32, T63.824849668K,1ps/.1ps grid. FM-only. No new training data, no dev futures in training, no production changes.',
        design='Three seed-blocked arms: unchanged control, per-block anchor FiLM, relative physical-time attention bias. All start control step18000; existing Adam state restored, new parameters fresh state; identical data/source/tau streams.6000 updates cosine3e-4 to3e-6. Fixed final step24000 primary.',
        anchor='Each block gets actual S0, neighbor-mean S0, their dot product and physical frame-time Fourier features, in registered crystal-frame coordinates; zero-init shift/scale residual. No Hamiltonian or clean future added.',
        relative_time='Each existing temporal attention receives learned additive per-head bias from signed and absolute physical time differences (ps). Zero-init bias; no new attention/value backbone, no clean future input.',
        controls='Zero-initialization preserves starting function within floating-point tolerance. Added capacity/compute not parameter-matched; measure both. Positive findings are candidate-level, not proof mechanism independent of capacity. No combined arm before separate effects.',
        evaluation='Full18train/36dev64paths128Euler plus first32/256. Expanded18dev256paths with paired latent Euler128 and midpoint128 versus frozen128base+128fine references. Whole-branch bootstrap100. Reused dev, no untouched holdout or automatic certification.',
        budget='9GPU jobs estimated15-35min each including expanded sampling; fat report20-50min. Smoke three arms estimated1-3min. No long-job monitoring.',
        skills='Karpathy zero-init/regression/isolation; experimental-design concurrent paired seed controls. Scientific Agent Skills https://doi.org/10.48550/arXiv.2609.00065'))
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']
    files += [ROOT/f'tests/reference_resolution/reference/{i:02d}/paths.h5' for i in range(18)]
    for f in files:lock[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


def train(task,out,smoke):
    net,opt,ck=build(task);rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    gs={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    data=real_data(cfg());config=cfg();steps=32 if smoke else task['updates'];history=[];stream=hashlib.sha256()
    start=time.perf_counter();net.train()
    for step in range(1,steps+1):
        lr=3e-6+(.0003-3e-6)*.5*(1+math.cos(math.pi*(step-1)/max(1,steps-1)))
        for group in opt.param_groups:group['lr']=lr
        y,t=next_batch(task,config,data,rng,gs['target'],'cuda');tau=gs['tau'].get_state()
        loss,z=fm_loss(net,y,t,gs)
        for v in (y,z,tau):stream.update(v.detach().cpu().numpy().tobytes())
        opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('Nonfinite update')
        opt.step()
        if smoke or step%100==0:history.append(dict(step=18000+step,loss=float(loss.detach()),gradient=float(grad),lr=lr))
        if step%500==0:print(task,step,flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=18000+steps,
        rng={k:g.get_state() for k,g in gs.items()},data_rng=rng.bit_generator.state,auxiliary_rng=ck['auxiliary_rng']),out/'checkpoint.pt')
    dump(out/'history.json',history);net.eval();ev=evaluate(net,out,[],smoke);extra=expanded(net,out,smoke)
    return dict(task=task,train_seconds=elapsed,evaluation=ev,expanded=extra,
        stream_hash=stream.hexdigest(),parameters=sum(p.numel() for p in net.parameters()),device=torch.cuda.get_device_name(),scientific_certification=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();return
    require_gpu('cuda')
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    out=HERE/f'{a.action}/{a.index:02d}';out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    r=train(tasks()[a.index],out,a.action=='smoke');r['total_seconds']=time.perf_counter()-tick;dump(out/'complete.json',r)


if __name__=='__main__':main()
