"""Empirical-fit controls and separated/debiased moment continuation."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from tests.curriculum_repair.run import load,checkpoint,fm_loss,flat_gradient,gradient_relation,FIXED_WEIGHT
from tests.flow_training_repair.run import training_anchors
from tests.flow_training_repair.objectives import differentiable_sample
from tests.bauer_p2_20260923.common import cfg,dump,generate
from tests.bauer_comprehensive_20260923.run import references,real_data,next_batch,require_gpu
from tests.cross_dimension_20260924.model import source
from .objectives import loss_parts

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def tasks():
    rows=[dict(domain='bauer',arm='hierarchical',seed=s,repair=a,updates=3000)
          for s in (11,22,33) for a in ('fm','variance','correlation','joint','cross','fit1','fit6')]
    np.random.default_rng(92802).shuffle(rows);return rows


def fit_parents():
    return [(r,F.normalize(torch.tensor(x),dim=-1)) for r,x in references('bauer','train') if r['L']==16]


@torch.no_grad()
def evaluate(net,out,fit_ids,smoke=False):
    times=torch.arange(11,device='cuda')*.1;rows=[]
    with h5py.File(out/'paths.h5','x') as h:
        h.attrs['complete']=False
        for split in ('train','dev'):
            for j,(row,raw) in enumerate(references('bauer',split)):
                if smoke and j:break
                initial=F.normalize(torch.tensor(raw[:1,0],device='cuda'),dim=-1).expand(2 if smoke else 64,-1,-1)
                seed=928300+(0 if split=='train' else 1000)+j
                z=source(initial,11,torch.Generator(device='cuda').manual_seed(seed),2.)
                levels=[16] if smoke else ([32,128,256] if j==0 else [128])
                for steps in levels:
                    start=time.perf_counter();x=generate(net,initial,times,None,steps,source=z)
                    torch.cuda.synchronize();seconds=time.perf_counter()-start
                    g=h.create_group(f'{split}_parent{j:03d}_steps{steps}')
                    g.attrs['parent']=json.dumps(row);g.attrs['split']=split
                    g.attrs['fit_parent']=row['parent_id'] in fit_ids
                    g.create_dataset('generated',data=x.cpu().numpy(),compression='lzf')
                    g.create_dataset('reference',data=raw,compression='lzf')
                    g.create_dataset('time_ps',data=np.arange(11)*.1)
                    rows.append(dict(split=split,parent=row,steps=steps,seconds=seconds,branches=len(x)))
        h.attrs['complete']=True
    return rows


def train(task,out,smoke=False):
    net,ck=load(task);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5);opt.load_state_dict(ck['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=ck['data_rng']
    arng=np.random.default_rng();arng.bit_generator.state=ck['auxiliary_rng']
    generators={k:torch.Generator(device='cuda').set_state(v) for k,v in ck['rng'].items()}
    config=cfg();small=fit_parents();mode=task['repair'];is_fit=mode in ('fit1','fit6')
    if mode=='fit1':small=small[:1]
    fit_ids=[r['parent_id'] for r,x in small] if is_fit else []
    data=None if is_fit else real_data(config);anchors=None if is_fit else training_anchors()
    times=torch.arange(11,device='cuda')*.1;history=[];stream=hashlib.sha256()
    # Same independent interpolation probes at start/end; CFM loss includes target variance.
    initial_probes=probes(net,small)
    start=time.perf_counter()
    for step in range(1,(32 if smoke else task['updates'])+1):
        if is_fit:
            row,y=small[int(rng.integers(len(small)))];y=y.cuda()
        else:y,times=next_batch(task,config,data,rng,generators['target'],'cuda')
        loss,z=fm_loss(net,y,times,generators)
        if step<=32:
            for val in (y,z):stream.update(val.cpu().numpy().tobytes())
        opt.zero_grad(set_to_none=True);loss.backward();info={}
        if not is_fit and step%16==0:
            reference=anchors[int(arng.integers(len(anchors)))].cuda()
            initial=reference[:1,0].expand(8,-1,-1)
            za=source(initial,11,generators['aux'],2.)
            if mode!='fm':
                gf=torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1) for p in net.parameters()]).detach().clone()
                x=differentiable_sample(net,initial,times,za,32)
                aux,parts=loss_parts(x,reference,mode);ga=flat_gradient(aux,net)
                info.update(**parts,**gradient_relation(gf,ga),aux_loss=float(aux.detach()),weight=FIXED_WEIGHT)
                offset=0
                for p in net.parameters():
                    g=ga[offset:offset+p.numel()].view_as(p);offset+=p.numel()
                    if p.grad is None:p.grad=FIXED_WEIGHT*g.clone()
                    else:p.grad.add_(g,alpha=FIXED_WEIGHT)
                del x,aux,gf,ga
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('nonfinite update')
        opt.step()
        if info or step%100==0 or smoke:history.append(dict(step=step,fm_loss=float(loss.detach()),gradient=float(grad),**info))
        if step%500==0:dump(out/'history.json',history);print(task,step,flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),task=task,step=6000+(32 if smoke else task['updates']),
        rng={k:g.get_state() for k,g in generators.items()},data_rng=rng.bit_generator.state,
        auxiliary_rng=arng.bit_generator.state,fit_ids=fit_ids),out/'checkpoint.pt')
    dump(out/'history.json',history);final_probes=probes(net,small);net.eval()
    evaluation=evaluate(net,out,fit_ids,smoke)
    return dict(task=task,train_seconds=elapsed,evaluation=evaluation,fit_ids=fit_ids,
        first32_data_source_hash=stream.hexdigest(),initial_probes=initial_probes,final_probes=final_probes,
        device=torch.cuda.get_device_name(),peak_bytes=torch.cuda.max_memory_allocated(),scientific_certification=False)


@torch.no_grad()
def probes(net,parents):
    times=torch.arange(11,device='cuda')*.1;rows=[]
    for j,(row,raw) in enumerate(parents):
        gs={k:torch.Generator(device='cuda').manual_seed(928500+j*10+i) for i,k in enumerate(('source','tau'))}
        loss,_=fm_loss(net,raw.cuda(),times,gs)
        rows.append(dict(parent=row,cfm_loss=float(loss)))
    return rows


def prepare():
    if (HERE/'source_lock.json').exists():raise RuntimeError('already frozen')
    train_rows=json.loads((ROOT/'tests/bauer_p2_20260923/manifest_train.json').read_text())
    dev_rows=json.loads((ROOT/'tests/bauer_p2_20260923/manifest_dev.json').read_text())
    assert not {r['parent_id'] for r in train_rows}&{r['parent_id'] for r in dev_rows}
    parents=fit_parents();assert len(parents)==6 and all(len(x)==8 for r,x in parents)
    assert all(len(a)==8 for a in training_anchors())
    dump(HERE/'protocol.json',dict(tasks=tasks(),fit_parents=[r for r,x in parents],
        scope='Bauer open L16/25/32,63.824849668K,1ps block,0.1ps grid; no reference or backbone change',
        start='Original FM-only6000 checkpoint per seed, optimizer/RNG restored,3000 more updates. Not continuation of chosen best curriculum arm.',
        controls='Full-data FM,var-only,corr-only,joint-plugin,joint-cross: shared data/latent/aux draws; fixed0.06684491978609611 every16updates,8 generated/8 train reference branches,32-step sampling.',
        estimator='Cross uses two disjoint independent generated groups of4. E[(mA-t)(mB-t)]=(E[m]-t)^2 conditional on fixed empirical reference t and its normalization. Removes generated-estimator variance penalty, not reference error or finite-step error. Can be negative; not clamped.',
        limitations='Splitting into4+4 can increase gradient variance. No claim of total unbiased physical loss; reference8 remains noisy, scales depend on reference. Component removal changes total auxiliary strength; gradients logged.',
        fit='One fixed L16 train parent or first6 L16 family parents,all8 empirical branches,FM only. Restricted fine-tuning/memorization diagnostic, not independent physical certification; failure at3000 updates does not prove architecture impossibility.',
        evaluation='Frozen final checkpoint.18 train calibration parents (one per size/family) +36 independent dev parents;64 generated each,128flow; first train/dev parent32/128/256. Fit parents labelled separately. No new final holdout.',
        metrics='Physical energy,magnetization,space,increment covariance,two-time statistics; whole-branch bootstrap100. Empirical train agreement cannot prove unseen-future physics.',
        budget='21GPU tasks (15 component/estimator controls+6 restricted-fit),3seeds, estimated10-20min/task; report5-15min. No monitoring after long submission.',
        skills='Karpathy isolation/regression; experimental-design seed blocks and paired controls. Scientific Agent Skills DOI10.48550/arXiv.2609.00065'))
    lock=json.loads((ROOT/'tests/curriculum_repair/source_lock.json').read_text())
    for name,digest in lock.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    for file in list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']:
        lock[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();return
    require_gpu('cuda');torch.set_num_threads(1)
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    out=HERE/f'{a.action}/{a.index:02d}';out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    result=train(tasks()[a.index],out,a.action=='smoke');result['total_seconds']=time.perf_counter()-start
    dump(out/'complete.json',result)


if __name__=='__main__':main()
