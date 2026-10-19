"""Finite empirical-law transport and optimizer controls; no physics changes."""
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
from tests.curriculum_repair.run import load,fm_loss
from tests.spin_path_transformer.run import make_net
from tests.moment_bias_repair.run import fit_parents
from tests.cross_dimension_20260924.model import source
from tests.transport_audit.run import integrate
from tests.bauer_comprehensive_20260923.run import require_gpu,diagnostic_comparison
from tests.bauer_p2_20260923.common import dump,comparison
from .oracle import conditional,branch_summary

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
STOP=.95


def tasks():
    rows=[dict(domain='bauer',arm='hierarchical',seed=s,repair=a,updates=6000)
          for s in (11,22,33) for a in ('fresh_constant','fresh_decay','warm_constant','warm_decay')]
    np.random.default_rng(92902).shuffle(rows);return rows


def rate(arm,step,total):
    return .001 if arm.endswith('constant') else .00003+.00097*.5*(1+math.cos(math.pi*(step-1)/max(1,total-1)))


def empirical(dtype=torch.float64):
    row,y=fit_parents()[0]
    return row,F.normalize(y.cuda().to(dtype),dim=-1)


@torch.no_grad()
def evaluate(net,out,smoke=False,oracle=False):
    row,y=empirical();n=16 if smoke else 256
    initial=y[:1,0].expand(n,-1,-1);times=torch.arange(11,device='cuda')*.1
    z=source(initial,11,torch.Generator(device='cuda').manual_seed(929100),2.)
    labels=torch.arange(n,device='cuda')%len(y)
    direct=path_interpolate(z,y[labels],z.new_full((n,),STOP),torch.Generator(device='cuda').manual_seed(929101))[0]
    rows=[];errors=[]
    with h5py.File(out/'paths.h5','x') as h:
        h.attrs['complete']=False;h.create_dataset('empirical_targets',data=y.cpu().numpy())
        h.create_dataset('direct_interpolation_tau095',data=direct.cpu().numpy(),compression='lzf')
        if not oracle:
            for t in (0.,.01,.03,.1,.25,.5,.75,.9,.95):
                x=path_interpolate(z,y[labels],z.new_full((n,),t),torch.Generator(device='cuda').manual_seed(929102))[0]
                truth,w,bad=conditional(x,x.new_full((n,),t),y,initial)
                pred=net.velocity(x.float(),x.new_full((n,),t).float(),initial.float(),times).double()
                errors.append(dict(tau=t,mse=float((pred[:,1:]-truth[:,1:]).square().sum(-1).mean()),
                    oracle_speed2=float(truth[:,1:].square().sum(-1).mean()),unsupported=int(bad.sum())))
        configs=[(0.,16,'euler',STOP)] if smoke else (
            [(0.,512,'euler',STOP),(0.,2048,'euler',STOP),(0.,1024,'midpoint',STOP)] if oracle else
            [(0.,128,'euler',STOP),(0.,512,'euler',STOP),(0.,256,'midpoint',STOP),
             *[(s,128,'euler',STOP) for s in (.03,.1,.25,.5,.75)],(0.,128,'euler',1.)])
        for start,steps,method,stop in configs:
            counts=[0]
            def field(x,t):
                if oracle:
                    v,_,bad=conditional(x,t,y,initial);counts[0]+=int(bad.sum());return v
                return net.velocity(x.float(),t.float(),initial.float(),times).double()
            begin=z if start==0 else path_interpolate(z,y[labels],z.new_full((n,),start),torch.Generator(device='cuda').manual_seed(929103))[0]
            tick=time.perf_counter();x=integrate(field,begin,start,stop,steps,method);torch.cuda.synchronize()
            name=f'start{start}_steps{steps}_{method}_stop{stop}';g=h.create_group(name)
            g.create_dataset('generated',data=x.cpu().numpy(),compression='lzf')
            g.create_dataset('time_ps',data=np.arange(11)*.1)
            r=direct if stop<1 else y
            _,w,bad=conditional(x,x.new_full((n,),min(stop,STOP)),y,initial) if stop<1 else (None,None,None)
            rows.append(dict(group=name,start_tau=start,stop_tau=stop,steps=steps,method=method,seconds=time.perf_counter()-tick,
                reference_informed_restart=start>0,**branch_summary(x,y),
                unsupported_field_evaluations=counts[0] if oracle else None,
                final_outside_empirical_support=int(bad.sum()) if bad is not None else None,
                physical=comparison(x.cpu().numpy(),r.cpu().numpy()),
                diagnostic=diagnostic_comparison(x.cpu().numpy(),r.cpu().numpy(),0,929),
                norm_error=float((x.norm(dim=-1)-1).abs().max()),anchor_error=float((x[:,0]-initial).abs().max())))
        h.attrs['complete']=True
    return dict(parent=row,rows=rows,velocity_errors=errors,reference='uniform empirical8; tau.95 smoothed law, not original physical target',scientific_certification=False)


def run(task,out,smoke):
    if task['repair'].startswith('warm'):net,ck=load(task)
    else:net=make_net(task);ck=None
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
    if ck is not None:opt.load_state_dict(ck['optimizer'])
    _,y=empirical(torch.float32);times=torch.arange(11,device='cuda')*.1
    gs={k:torch.Generator(device='cuda').manual_seed(base+task['seed']) for k,base in [('source',929200),('tau',929300)]}
    history=[];stream=hashlib.sha256();start=time.perf_counter()
    for step in range(1,(32 if smoke else task['updates'])+1):
        lr=rate(task['repair'],step,32 if smoke else task['updates'])
        for group in opt.param_groups:group['lr']=lr
        loss,z=fm_loss(net,y,times,gs)
        if step<=32:stream.update(z.cpu().numpy().tobytes())
        opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('nonfinite training')
        opt.step()
        if step%100==0 or smoke:history.append(dict(step=step,loss=float(loss.detach()),gradient=float(grad),lr=lr))
        if step%500==0:dump(out/'history.json',history);print(task,step,flush=True)
        if step in (3000,6000):torch.save(dict(model=net.state_dict(),task=task,step=step,optimizer=opt.state_dict(),rng={k:g.get_state() for k,g in gs.items()}),out/f'step{step}.pt')
    torch.cuda.synchronize();elapsed=time.perf_counter()-start;dump(out/'history.json',history);net.eval()
    result=evaluate(net,out,smoke);result.update(task=task,train_seconds=elapsed,first32_source_hash=stream.hexdigest(),
        peak_bytes=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name())
    return result


def prepare():
    if (HERE/'source_lock.json').exists():raise RuntimeError('already frozen')
    dump(HERE/'protocol.json',dict(tasks=tasks(),parent=fit_parents()[0][0],
        scope='Bauer L16 one fixed TRAIN parent,8 empirical future paths. Physical1ps/0.1ps unchanged. No auxiliary statistics losses. Not generalization or full LLG-law certification.',
        training='6000 CFM updates,uniform tau0..1,shifted normal source2*S0,8branches per batch. Fresh vs original full-dataFM6000 warm-start with Adam state; constant1e-3 vs cosine1e-3 to3e-5. Warm arms have extra prior compute/data. New paired source/tau RNG streams perseed, not full historical RNG continuation.',
        oracle='Projected-normal spherical source density times inverse-contraction Jacobian; sum all8 equal-weight whole-path modes. Numerically verified. No real future as extra network input. Oracle itself accesses targets only for diagnosis.',
        evaluation='256sources,empirical smoothed law at transport tau0.95; network128/512Euler and256midpoint. Teacher-law velocity error grid; restarts.03/.1/.25/.5/.75 use true targets and are NOT deployable. Separate tau1 output versus empirical8. Never relabel tau0.95 as fully generated original law.',
        numerical='Oracle512/2048Euler vs1024midpoint; float64 solver state/oracle with float32 network. Outside-support nearest extension counted, not silently certified. Seed-matched same sources across all arms.',
        uncertainty='Nearest-path counts are empirical labels, not physical mechanism weights. No hard physical thresholds. Directlaw stratified32perempiricalpath. Seed variation separate; no new holdout.',
        estimates_minutes=dict(training=[10,20],oracle=[1,5],report=[1,5]),
        skills='Karpathy isolated changes; experimental-design paired2x2 controls/3seeds; Scientific Agent Skills DOI10.48550/arXiv.2609.00065'))
    lock=json.loads((ROOT/'tests/moment_bias_repair/source_lock.json').read_text())
    for name,digest in lock.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    for file in list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']:
        lock[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','train','oracle']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();return
    require_gpu('cuda');torch.set_num_threads(1)
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError(name)
    out=HERE/f'{a.action}/{a.index:02d}';out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    result=evaluate(None,out,oracle=True) if a.action=='oracle' else run(tasks()[a.index],out,a.action=='smoke')
    result['total_seconds']=time.perf_counter()-start;dump(out/'complete.json',result)


if __name__=='__main__':main()
