"""Paired continuation and objective-curriculum diagnostic; no physics changes."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from scripts.model.gate_f import path_interpolate
from tests.flow_training_repair.run import training_anchors, tasks as old_tasks
from tests.flow_training_repair.objectives import draw_tau, differentiable_sample, joint_loss
from tests.spin_path_transformer.run import make_net, evaluate_bauer
from tests.bauer_p2_20260923.common import cfg, dump, generate
from tests.bauer_comprehensive_20260923.run import real_data, next_batch, require_gpu, references, diagnostic_comparison
from tests.cross_dimension_20260924.model import source

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'tests/flow_training_repair'
UPDATES=3000


def ramp(step):
    return .1*min(1.,max(0.,(step-500)/1000))


FIXED_WEIGHT=sum(ramp(s) for s in range(16,UPDATES+1,16))/len(range(16,UPDATES+1,16))


def weight(arm,step):
    if arm=='fm':return 0.
    if arm=='fixed32':return FIXED_WEIGHT
    if arm in ('ramp32','ramp64'):return ramp(step)
    raise ValueError(arm)


def tasks():
    rows=[dict(domain='bauer',arm='hierarchical',seed=s,repair=a,updates=UPDATES)
          for s in (11,22,33) for a in ('fm','fixed32','ramp32','ramp64')]
    np.random.default_rng(92601).shuffle(rows)
    return rows


def checkpoint(seed,repair='baseline'):
    index=next(i for i,t in enumerate(old_tasks()) if t['domain']=='bauer' and t['seed']==seed and t['repair']==repair)
    return OLD/f'run/{index:02d}/step6000.pt'


def load(task,repair='baseline'):
    ck=torch.load(checkpoint(task['seed'],repair),map_location='cpu',weights_only=True)
    assert ck['step']==6000 and ck['task']['seed']==task['seed'] and ck['task']['repair']==repair
    net=make_net(task);net.load_state_dict(ck['model']);return net,ck


def flat_gradient(loss,net):
    parameters=list(net.parameters())
    grads=torch.autograd.grad(loss,parameters,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p,g in zip(parameters,grads)])


def gradient_relation(a,b):
    na=a.norm();nb=b.norm();den=na*nb
    return dict(fm_norm=float(na),aux_norm=float(nb),
                cosine=float(a.dot(b)/den) if den>0 else None,
                aux_to_fm=float(nb/na) if na>0 else None)


def fm_loss(net,y,times,generators):
    z=source(y[:,0],11,generators['source'],net.source_shift)
    tau,_=draw_tau(len(y),1.,'baseline',generators['tau'],'cuda')
    x,v,_=path_interpolate(z,y,tau,generators['source'])
    pred=net.velocity(x,tau,y[:,0],times)
    return (pred[:,1:]-v[:,1:]).square().sum(-1).mean(),z


def diagnose(seed,out,smoke=False):
    task=dict(domain='bauer',arm='hierarchical',seed=seed)
    times=torch.arange(11,device='cuda')*.1;rows=[]
    parents=[(r,x) for r,x in references('bauer','train') if r['L']==16]
    with h5py.File(out/'paths.h5','x') as h:
        h.attrs['complete']=False
        for kind in ('baseline','joint'):
            net,_=load(task,kind)
            for j,(row,raw) in enumerate(parents[:1] if smoke else parents):
                reference=F.normalize(torch.tensor(raw,device='cuda'),dim=-1)
                for repeat in range(1 if smoke else 2):
                    gs={k:torch.Generator(device='cuda').manual_seed(926100+j*100+repeat*10+i)
                        for i,k in enumerate(('source','tau'))}
                    # Whole train branches, identical branch subsets and sources across checkpoints.
                    y=reference[repeat*8:repeat*8+8]
                    loss,_=fm_loss(net,y,times,gs);gf=flat_gradient(loss,net)
                    initial=reference[:1,0].expand(8,-1,-1)
                    z=source(initial,11,torch.Generator(device='cuda').manual_seed(926200+j*10+repeat),2.)
                    for steps in (32,64):
                        x=differentiable_sample(net,initial,times,z,steps)
                        aux,parts=joint_loss(x,reference);ga=flat_gradient(aux,net)
                        relation=gradient_relation(gf,ga)
                        if not torch.isfinite(ga).all() or not torch.isfinite(gf).all():raise RuntimeError('nonfinite diagnostic gradient')
                        rows.append(dict(checkpoint=kind,parent=row,repeat=repeat,steps=steps,
                                         fm_loss=float(loss.detach()),aux_loss=float(aux.detach()),**parts,**relation))
                        del x,aux,ga
                initial=reference[:1,0].expand(32,-1,-1)
                z=source(initial,11,torch.Generator(device='cuda').manual_seed(926300+j),2.)
                for steps in (32,128,256):
                    x=generate(net,initial,times,None,steps,source=z).cpu().numpy()
                    group=h.create_group(f'{kind}/parent{j}/steps{steps}')
                    group.attrs['parent']=json.dumps(row)
                    group.create_dataset('generated',data=x,compression='lzf')
                    group.create_dataset('reference',data=raw,compression='lzf')
                    group.create_dataset('time_ps',data=np.arange(11)*.1)
                    dump(out/f'{kind}_parent{j}_steps{steps}.json',diagnostic_comparison(x,raw,0,seed))
            del net
        h.attrs['complete']=True
    dump(out/'gradients.json',rows)
    return dict(seed=seed,gradient_rows=len(rows),scientific_certification=False)


def train(task,out,smoke=False):
    net,ck=load(task);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
    opt.load_state_dict(ck['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    arng=np.random.default_rng();arng.bit_generator.state=ck['auxiliary_rng']
    generators={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    config=cfg();data=real_data(config);anchors=training_anchors();history=[]
    digest=hashlib.sha256();start=time.perf_counter();steps_total=32 if smoke else UPDATES
    for step in range(1,steps_total+1):
        y,times=next_batch(task,config,data,rng,generators['target'],'cuda')
        loss,z=fm_loss(net,y,times,generators)
        if step<=32:
            for a in (y,z):digest.update(a.cpu().numpy().tobytes())
        opt.zero_grad(set_to_none=True);loss.backward();info={}
        if step%16==0:
            # Consume identical auxiliary parent/source streams even while weight is zero.
            reference=anchors[int(arng.integers(len(anchors)))].cuda()
            initial=reference[:1,0].expand(8,-1,-1)
            za=source(initial,11,generators['aux'],2.)
            w=weight(task['repair'],step if not smoke else step+1000)
            info['weight']=w
            if w:
                gf=torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1) for p in net.parameters()]).detach().clone()
                x=differentiable_sample(net,initial,times,za,64 if task['repair']=='ramp64' else 32)
                aux,parts=joint_loss(x,reference)
                ga=flat_gradient(aux,net);info.update(gradient_relation(gf,ga),**parts,aux_loss=float(aux.detach()))
                offset=0
                for p in net.parameters():
                    g=ga[offset:offset+p.numel()].view_as(p);offset+=p.numel()
                    if p.grad is None:p.grad=w*g.clone()
                    else:p.grad.add_(g,alpha=w)
                del x,aux,gf,ga
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('nonfinite training')
        opt.step()
        if info or step%500==0:history.append(dict(step=step,fm_loss=float(loss.detach()),gradient=float(grad),**info))
        if step%500==0:dump(out/'history.json',history);print(task,step,flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=6000+steps_total,
        rng={k:g.get_state() for k,g in generators.items()},data_rng=rng.bit_generator.state,
        auxiliary_rng=arng.bit_generator.state),out/'checkpoint.pt')
    dump(out/'history.json',history);net.eval();evaluation=evaluate_bauer(net,out,smoke)
    return dict(task=task,train_seconds=elapsed,evaluation=evaluation,first32_data_source_hash=digest.hexdigest(),
                device=torch.cuda.get_device_name(),peak_bytes=torch.cuda.max_memory_allocated(),scientific_certification=False)


def prepare():
    if (HERE/'source_lock.json').exists():raise RuntimeError('already frozen')
    protocol=dict(tasks=tasks(),start='old FM-only step6000 with optimizer and all RNG states restored',
        scope='Bauer open L16/25/32,63.824849668K,1ps;0.1ps output; no physical/data/backbone change',
        objectives='FM-only; fixed weak32; ramp32; ramp64. Every16 updates auxiliary8 generated vs16 train branches.',
        curriculum='First500 updates FM only; linear auxiliary0 to0.1 through1500; hold0.1 thereafter.',
        fixed_weight=FIXED_WEIGHT,matching='Exact sum of weights on scheduled auxiliary updates matched fixed32/ramp32. Same FM and auxiliary draws; NOT equal compute or equal effective gradient.',
        diagnosis='Frozen baseline/joint6000 for3seeds; six L16 train families;2 repeated draws; FM/aux gradients32/64; paired32/128/256 sampling.',
        evaluation='All36 existing development parents,64 generated/16 reference,128flow; firstparent convergence via frozen evaluator. Fixed final checkpoint primary; no new final holdout.',
        limitations='Finite-branch squared moment loss retains sampling bias. Ramping does not prove physics. No hard-path filtering, physical-time curriculum or automatic production promotion.',
        provenance='Karpathy: isolated changes/tests; experimental-design: seed blocks, paired streams, concurrent FM control. Scientific Agent Skills https://doi.org/10.48550/arXiv.2609.00065',
        estimates_minutes=dict(diagnose=[3,8],training=[10,25],report=[3,10]))
    dump(HERE/'protocol.json',protocol)
    lock=json.loads((OLD/'source_lock.json').read_text())
    for name,digest in lock.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']
    files += [checkpoint(s,k) for s in (11,22,33) for k in ('baseline','joint')]
    for file in files:lock[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock)
    (HERE/'logs').mkdir(exist_ok=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','diagnose','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();return
    require_gpu('cuda');torch.set_num_threads(1)
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    out=HERE/f'{a.action}/{a.index:02d}';out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    result=diagnose((11,22,33)[a.index],out) if a.action=='diagnose' else train(tasks()[a.index],out,a.action=='smoke')
    result['total_seconds']=time.perf_counter()-start;dump(out/'complete.json',result)


if __name__=='__main__':main()
