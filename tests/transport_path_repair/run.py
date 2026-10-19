"""Frozen transport-path diagnosis; reference restarts are not deployable inference."""
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
from tests.cross_dimension_20260924.model import source
from tests.condition_time_repair.run import build
from tests.branch_coverage_repair.run import checkpoint
from tests.reference_resolution.run import parents,reference_inputs
from tests.bauer_comprehensive_20260923.run import require_gpu
from tests.bauer_p2_20260923.common import dump

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CUTS=(.125,.25,.5,.75)


def tasks():
    return [dict(seed=s,L=n) for s in (11,22,33) for n in (16,25,32)]


def grid(steps,late=False,start=0.):
    if steps<1 or not 0<=start<1:raise ValueError('transport grid')
    u=np.linspace(0.,1.,steps+1)
    return start+(1-start)*(1-(1-u)**2 if late else u)


@torch.no_grad()
def integrate(field,x,edges,trace=False):
    edges=np.asarray(edges,dtype=float)
    if len(edges)<2 or not np.isfinite(edges).all() or edges[0]<0 or edges[-1]>1 or not np.all(np.diff(edges)>0):
        raise ValueError('invalid transport edges')
    x=x.clone();initial=x[:,0].clone();states={}
    for lo,hi in zip(edges[:-1],edges[1:]):
        dt=float(hi-lo);tau=x.new_full((len(x),),float(lo))
        mid=F.normalize(x+.5*dt*field(x,tau),dim=-1);mid[:,0]=initial
        x=F.normalize(x+dt*field(mid,tau+.5*dt),dim=-1);x[:,0]=initial
        if trace and any(abs(hi-v)<1e-12 for v in CUTS):states[float(hi)]=x.clone()
    return x,states


def prepare():
    assert not (HERE/'source_lock.json').exists(),'Already frozen'
    dump(HERE/'protocol.json',dict(tasks=tasks(),
        scope='Frozen control step24000, seeds11/22/33. Bauer open L16/25/32, 18 existing dev parents;1ps,11frames,0.1ps saved grid. No training, no physics changes.',
        sampling='128 latent paths/parent. Paired projected midpoint uniform128, late128, uniform512, late512. late grid tau(u)=1-(1-u)^2. Same-step arms match network evaluations, not guaranteed wall cost. Model seed blocked; sampler order randomized per parent using seed937500+parent_index.',
        diagnosis='Save normal uniform128 transport states at tau=.125,.25,.5,.75 and geodesic source/base-reference interpolations at same tau. Restart from reference interpolation at .25,.5,.75 with ceil(128*(1-tau)) uniform midpoint steps. Reference restarts are oracle-informed diagnostics, NOT deployable generated samples. Shorter remaining transport and access to future confound improvement; cannot alone establish distribution-shift causality.',
        comparison='Final outputs compared separately to base128 and independent fine128; teacher uses base128. Intermediate states are transport geometry, NOT physical LLG-time marginals. Report S0 overlap, neighbor alignment, cross-frame alignment and whole-branch uncertainty. Normal final metrics: increment variance, two-time spin correlation, energy mean plus existing full evaluator. No paired model-to-LLG trajectory error.',
        uncertainty='Fixed-parent whole-branch bootstrap100, reference and model independent for normal inference; base comparison of teacher restart is descriptive only. Fine reference independent of teacher. Latent arms paired. Reused development parents, not untouched test. Reference-reference differences are not tolerances. No automatic pass or promotion.',
        decision='If nonuniform and uniform high-resolution outputs agree while physical error persists, integration is not sufficient explanation. If teacher-informed restarts improve, locate accumulated transport mismatch without claiming a deployable repair. Report tradeoffs across all seeds and sizes.',
        budget='Nine GPU tasks estimated10-25min each; nine fat statistics tasks5-15min each; aggregation<3min. Smoke<5min. Queue time excluded. No long monitoring.',
        skills='Karpathy isolated regression and experimental-design paired/blocked controls; Scientific Agent Skills https://doi.org/10.48550/arXiv.2609.00065'))
    lock=json.loads((ROOT/'tests/velocity_target_repair/source_lock.json').read_text())
    for name,digest in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']+[checkpoint(s) for s in (11,22,33)]
    for f in files:lock[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


@torch.no_grad()
def run(index,smoke=False):
    require_gpu('cuda');torch.set_num_threads(1);task=tasks()[index]
    net,_,_=build(dict(domain='bauer',arm='hierarchical',repair='control',seed=task['seed']))
    ck=torch.load(checkpoint(task['seed']),map_location='cpu',weights_only=True)
    assert ck['step']==24000;net.load_state_dict(ck['model']);net.eval()
    out=HERE/('smoke' if smoke else 'run')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False)
    selected=[(i,r) for i,r in enumerate(parents()) if r['L']==task['L']]
    if smoke:selected=selected[:1]
    count=4 if smoke else 128;steps=8 if smoke else 128;records=[]
    with h5py.File(out/'paths.h5','x') as h:
        h.attrs['complete']=False;h.attrs['task']=json.dumps(task)
        for i,row in selected:
            initial,_=reference_inputs(row)
            a=F.normalize(torch.tensor(initial,device='cuda',dtype=torch.float32),dim=-1)[None].expand(count,-1,-1)
            gen=torch.Generator(device='cuda').manual_seed(937100+i)
            z=source(a,11,gen,2.);times=torch.arange(11,device='cuda')*.1
            with h5py.File(ROOT/f'tests/reference_resolution/reference/{i:02d}/paths.h5') as ref:
                assert ref.attrs['complete'];y=F.normalize(torch.tensor(ref['base/spins'][:count],device='cuda',dtype=torch.float32),dim=-1)
            y[:,0]=a
            g=h.create_group(f'parent{i:02d}');g.attrs['parent']=json.dumps(row)
            g.create_dataset('time_ps',data=np.arange(11)*.1);g.create_dataset('latent',data=z.cpu().numpy(),compression='lzf')
            teacher={}
            for tau in CUTS:
                state,_,cut=path_interpolate(z,y,z.new_full((count,),tau),gen)
                assert not cut['unresolved_cut_count'];teacher[tau]=state
                g.create_dataset(f'teacher/tau{tau}',data=state.cpu().numpy(),compression='lzf')
            configs=[(f'uniform{steps}',grid(steps),z,True,False),
                     (f'late{steps}',grid(steps,True),z,False,False),
                     (f'uniform{steps*4}',grid(steps*4),z,False,False),
                     (f'late{steps*4}',grid(steps*4,True),z,False,False)]
            configs += [(f'teacher_restart{tau}',grid(int(steps*(1-tau)),start=tau),teacher[tau],False,True) for tau in (.25,.5,.75)]
            np.random.default_rng(937500+i).shuffle(configs)
            # Warmup is excluded from timing; reported times include transfer of saved snapshots.
            net.velocity(z[:1],z.new_zeros(1),a[:1],times);torch.cuda.synchronize()
            for name,edges,x,trace,informed in configs:
                torch.cuda.synchronize();tick=time.perf_counter();pieces=[];traces={v:[] for v in CUTS}
                for b in range(0,count,64):
                    init=a[b:b+64]
                    final,states=integrate(lambda v,t:net.velocity(v,t,init,times),x[b:b+64],edges,trace)
                    pieces.append(final.cpu().numpy())
                    for tau,state in states.items():traces[tau].append(state.cpu().numpy())
                torch.cuda.synchronize();seconds=time.perf_counter()-tick;arr=np.concatenate(pieces)
                norm=float(abs(np.linalg.norm(arr,axis=-1)-1).max());anchor=float(abs(arr[:,0]-a.cpu().numpy()).max())
                assert np.isfinite(arr).all() and norm<2e-6 and anchor==0
                ds=g.create_dataset(f'final/{name}',data=arr,compression='lzf');ds.attrs['reference_informed']=informed
                g.create_dataset(f'grids/{name}',data=edges)
                for tau,values in traces.items():
                    if values:g.create_dataset(f'on_policy/tau{tau}',data=np.concatenate(values),compression='lzf')
                records.append(dict(parent=i,sampler=name,seconds=seconds,network_evaluations_per_path=2*(len(edges)-1),norm_error=norm,anchor_error=anchor,reference_informed=informed))
            print(task,row['parent_id'],'complete',flush=True)
        h.attrs['complete']=True
    dump(out/'complete.json',dict(task=task,records=records,device=torch.cuda.get_device_name(),count=count,scientific_certification=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','smoke']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare();return
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    run(a.index,a.action=='smoke')


if __name__=='__main__':main()
