"""Matched-state independent LLG restarts; evaluation only, never training data."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from numba import set_num_threads
from tests.bauer_active_data_20260918.run import simulate
from tests.bauer_p2_20260923.common import dump,energy
from tests.reference_resolution.run import parents,reference_inputs
from tests.bauer_comprehensive_20260923.run import require_gpu
from tests.transition_block_repair.run import tasks
from tests.transition_block_repair.model import TransitionNet,rollout

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def seeds(state_index,fine,count):
    if not 0<=state_index<216 or count<1 or count>128:raise ValueError('seed allocation')
    return 1900000000+state_index*1000+(500 if fine else 0)+np.arange(count,dtype=np.int64)


def normalize_state(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[-1]!=3 or not np.isfinite(x).all() or np.any(np.linalg.norm(x,axis=-1)<.5):raise ValueError('invalid spin state')
    return x/np.linalg.norm(x,axis=-1,keepdims=True)


def prepare():
    assert not (HERE/'source_lock.json').exists(),'Already frozen'
    rows=[];sources=set();fingerprints=set()
    with h5py.File(HERE/'states.h5','x') as bank:
        bank.attrs['complete']=False;bank.attrs['role']='development_diagnostic_never_train'
        for i,parent in enumerate(parents()):
            _,config=reference_inputs(parent)
            for origin,path,key in (
                ('llg',ROOT/f'tests/reference_resolution/reference/{i:02d}/paths.h5','base/spins'),
                ('prefix_seed11',ROOT/'tests/transition_block_repair/train/00/paths.h5',f'dev/parent{i:02d}/midpoint128'),
                ('adjacent_seed11',ROOT/'tests/transition_block_repair/train/01/paths.h5',f'dev/parent{i:02d}/midpoint128')):
                sources.add(path)
                with h5py.File(path) as h:
                    assert h.attrs['complete'];raw=h[key]
                    for branch in (0,1):
                        for frame in (3,8):
                            initial=normalize_state(raw[branch,frame]);sid=len(rows)
                            sha=hashlib.sha256(initial.tobytes()).hexdigest();assert sha not in fingerprints;fingerprints.add(sha)
                            row=dict(index=sid,parent_index=i,parent=parent,origin=origin,source_path=str(path.relative_to(ROOT)),source_key=key,
                                     branch=branch,frame=frame,origin_time_ps=frame*.1,state_sha256=sha,
                                     config=config,energy_over_NJ=float(energy(initial)),mz=float(initial[:,2].mean()))
                            bank.create_dataset(f'state{sid:03d}',data=initial);rows.append(row)
        bank.attrs['complete']=True
    assert len(rows)==216
    # Check physical RNG separation from the reference/data sources used here.
    used=set()
    files=list((ROOT/'tests/bauer_active_data_20260918/data').glob('L*/*.h5'))
    files+=list((ROOT/'tests/branch_coverage_repair/reference').glob('*/*.h5'))
    files+=list((ROOT/'tests/reference_resolution/reference').glob('*/paths.h5'))
    for path in files:
        with h5py.File(path) as h:
            for key in ('seeds','base/seeds','fine/seeds'):
                if key in h:used.update(map(int,h[key][:]))
    fresh=set()
    for sid in range(216):
        for fine in (False,True):
            values=set(map(int,seeds(sid,fine,128)));assert not values&(used|fresh);fresh|=values
    dump(HERE/'manifest.json',rows)
    dump(HERE/'protocol.json',dict(
        scope='Frozen current prefix/adjacent two-frame models,3seeds; Bauer open L16/25/32 unchanged physics,0.1ps futures. No optimization or new training library.',
        selection='18existing dev parents x3origins(llg,prefix_seed11,adjacent_seed11) x2branches(indices0,1) x2physical times(.3,.8ps)=216fixed states. No selection by outcome/error. Same states for all6models. Seed11 chosen by fixed index, not best model; own-rollout claims for other model seeds not supported. Two states per origin/time are a limited descriptive slice, not population coverage certification.',
        reference='Each state128new base dt.25fs and128new fine dt.125fs, independent seeds; MT5.9,unchanged Hamiltonian,T,open boundary,save.1ps,duration.1ps.55,296short futures in18fat jobs. Same double-normalized initial for both references; float32 model input rounding explicitly measured. All new futures development-only, never training.',
        models='All six frozen step12000 checkpoints generate128independent futures perstate at midpoint128 and512. New latent seeds940100+state_id paired across models/solvers. No real future input, no teacher reset represented as free rollout, no relabelled physical time.',
        historical='Old bauer_teacher_state_20260920 bank uses L16/32/100 and0/1/5/9ps on different parents. Not substituted for these exact states. Reuse current paths to select states; new independent futures required since original each state has only one continuation.',
        metrics='Two-frame mean/covariance/variance,energy,mz,spatial andS0overlap using existing dedicated evaluator; both base/fine comparisons and base-vs-fine numerical/sampling control.100whole-future-branch bootstrap per fixed state; not site/frame replicas. Grouped summaries descriptive; same origin parents/times correlated, no216independent-parent claim. Reused development not final test. No pass thresholds or promotion.',
        decisions='Errors on real states imply conditional-transition bias remains. Errors mainly on generated states motivate coverage/accumulation repair, but source distributions differ so no sole-cause proof. Existing prefix versus adjacent training is the frozen repair contrast. This run repairs the missing diagnostic reference, not evidence of a repaired neural model.',
        budget='18fat reference jobs estimated3-10min each;6rtx4090 model jobs20-50min each;6fat stats5-20min each;aggregate<3min. Smoke<5min. Queue excluded; no long monitoring.',
        skill_use='Karpathy surgical isolation; experimental-design common-state blocked comparisons. Scientific Agent Skills https://doi.org/10.48550/arXiv.2609.00065'))
    lock=json.loads((ROOT/'tests/transition_block_repair/source_lock.json').read_text())
    for name,digest in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'states.h5',HERE/'manifest.json',HERE/'protocol.json']+list(sources)
    files += [ROOT/f'tests/transition_block_repair/train/{i:02d}/step12000.pt' for i in range(6)]
    for f in files:lock[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


def selected(parent=None,smoke=False):
    rows=json.loads((HERE/'manifest.json').read_text())
    if parent is not None:rows=[r for r in rows if r['parent_index']==parent]
    return rows[:1] if smoke else rows


def initial(row):
    with h5py.File(HERE/'states.h5') as h:
        assert h.attrs['complete'];a=h[f"state{row['index']:03d}"][:]
    assert hashlib.sha256(a.tobytes()).hexdigest()==row['state_sha256']
    return a


def reference(index,smoke):
    set_num_threads(2);rows=selected(index,smoke);count=4 if smoke else 128
    out=HERE/('smoke_reference' if smoke else 'reference')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    with h5py.File(out/'paths.h5','x') as h:
        h.attrs['complete']=False;h.attrs['role']='development_diagnostic_never_train'
        for row in rows:
            sid=row['index'];a=initial(row);config=row['config'];config['numerics']['duration_ps']=.1
            g=h.create_group(f'state{sid:03d}');g.attrs['metadata']=json.dumps(row);g.create_dataset('initial',data=a);g.create_dataset('time_ps',data=[0.,.1])
            for name,dt in (('base',.25),('fine',.125)):
                noise=seeds(sid,name=='fine',count);x,error=simulate(np.repeat(a[None],count,axis=0),noise,config,dt,.1)
                assert x.shape==(count,2,row['parent']['L'],3)
                part=g.create_group(name);part.attrs['dt_fs']=dt;part.create_dataset('spins',data=x,compression='lzf');part.create_dataset('seeds',data=noise)
                part.create_dataset('preprojection_error',data=error)
        h.attrs['complete']=True
    dump(out/'complete.json',dict(parent_index=index,states=len(rows),branches_per_dt=count,seconds=time.perf_counter()-tick))


@torch.no_grad()
def generate(index,smoke):
    require_gpu('cuda');task=tasks()[index];net=TransitionNet().cuda()
    ck=torch.load(ROOT/f'tests/transition_block_repair/train/{index:02d}/step12000.pt',map_location='cpu',weights_only=True)
    assert ck['task']==task and ck['step']==12000;net.load_state_dict(ck['model']);net.eval()
    out=HERE/('smoke_gpu' if smoke else 'gpu')/f'{index:02d}';out.mkdir(parents=True,exist_ok=False);records=[]
    with h5py.File(out/'paths.h5','x') as h:
        h.attrs['complete']=False;h.attrs['task']=json.dumps(task)
        for row in selected(smoke=smoke):
            sid=row['index'];count=4 if smoke else 128;a0=initial(row)
            a=F.normalize(torch.tensor(a0,device='cuda',dtype=torch.float32),dim=-1)[None].expand(count,-1,-1)
            delta=float(abs(a[0].cpu().numpy()-a0).max());assert delta<2e-7
            g=h.create_group(f'state{sid:03d}');g.attrs['metadata']=json.dumps(row);g.attrs['latent_seed']=940100+sid
            g.create_dataset('time_ps',data=[0.,.1]);g.create_dataset('model_initial',data=a[0].cpu().numpy())
            for steps in ((8,16) if smoke else (128,512)):
                pieces=[];torch.cuda.synchronize();tick=time.perf_counter()
                for b in range(0,count,64):
                    gen=torch.Generator(device='cuda').manual_seed(940100+sid+b*10000)
                    pieces.append(rollout(net,a[b:b+64],gen,steps,transitions=1).cpu().numpy())
                torch.cuda.synchronize();x=np.concatenate(pieces);elapsed=time.perf_counter()-tick
                norm=float(abs(np.linalg.norm(x,axis=-1)-1).max());anchor=float(abs(x[:,0]-a.cpu().numpy()).max())
                assert np.isfinite(x).all() and norm<2e-6 and anchor==0
                g.create_dataset(f'midpoint{steps}',data=x,compression='lzf')
                records.append(dict(state=sid,steps=steps,seconds=elapsed,norm_error=norm,anchor_error=anchor,reference_initial_rounding=delta))
        h.attrs['complete']=True
    dump(out/'complete.json',dict(task=task,records=records,device=torch.cuda.get_device_name(),scientific_certification=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','reference','generate']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare();return
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    if a.action=='reference':reference(a.index,a.smoke)
    else:generate(a.index,a.smoke)


if __name__=='__main__':main()
