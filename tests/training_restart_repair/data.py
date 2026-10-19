"""Train-origin intermediate states; new futures and diagnostic futures separated."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import h5py
import numpy as np
from numba import set_num_threads
from tests.branch_coverage_repair.run import parents
from tests.reference_resolution.run import reference_inputs
from tests.intermediate_restart_repair.run import normalize_state
from tests.bauer_active_data_20260918.run import simulate
from tests.bauer_p2_20260923.common import dump

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ROLES={'train':(64,.25,0),'probe':(128,.25,1000),'fine':(128,.125,2000)}


def seeds(sid,role):
    if not 0<=sid<48 or role not in ROLES:raise ValueError('state or role')
    n,_,offset=ROLES[role]
    return 1940000000+sid*10000+offset+np.arange(n,dtype=np.int64)


def rows(smoke=False):
    r=json.loads((HERE/'manifest.json').read_text())
    return r[:2] if smoke else r


def state(row):
    with h5py.File(HERE/'states.h5') as h:a=h[f"state{row['index']:03d}"][:]
    assert hashlib.sha256(a.tobytes()).hexdigest()==row['state_sha256']
    return a


def prepare():
    assert not (HERE/'source_lock.json').exists()
    train=parents();dev=json.loads((ROOT/'tests/bauer_p2_20260923/manifest_dev.json').read_text())
    assert len(train)==24 and not {r['parent_id'] for r in train}&{r['parent_id'] for r in dev}
    records=[];files=[]
    with h5py.File(HERE/'states.h5','x') as h:
        h.attrs['role']='train_origin_intermediate_states';h.attrs['complete']=False
        for i,parent in enumerate(train):
            assert parent['split']=='train';_,config=reference_inputs(parent);path=ROOT/parent['path'];files.append(path)
            with h5py.File(path) as src:
                for frame in (3,8):
                    a=normalize_state(src['spins'][0,frame]);sid=len(records)
                    h.create_dataset(f'state{sid:03d}',data=a)
                    records.append(dict(index=sid,parent_index=i,parent=parent,source_path=parent['path'],source_key='spins',
                        branch=0,frame=frame,source_time_ps=frame*.1,source_group=parent['parent_id'],
                        split='train',config=config,state_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
        h.attrs['complete']=True
    # Check the actual known source/reference seed banks, including prior restarts.
    used=set()
    paths=list((ROOT/'tests/bauer_active_data_20260918/data').glob('L*/*.h5'))
    paths+=list((ROOT/'tests/branch_coverage_repair/reference').glob('*/*.h5'))
    paths+=list((ROOT/'tests/reference_resolution/reference').glob('*/paths.h5'))
    paths+=list((ROOT/'tests/intermediate_restart_repair/reference').glob('*/paths.h5'))
    for p in paths:
        with h5py.File(p) as h:
            def visit(name,obj):
                if isinstance(obj,h5py.Dataset) and name.split('/')[-1]=='seeds':used.update(map(int,obj[:]))
            h.visititems(visit)
    fresh=set()
    for sid in range(48):
        for role in ROLES:
            values=set(map(int,seeds(sid,role)));assert not values&(fresh|used);fresh|=values
    dump(HERE/'manifest.json',records)
    dump(HERE/'protocol.json',dict(
        question='Does repeated conditional supervision at TRAIN-origin intermediate states improve drift and covariance, versus merely penalizing t0 statistics?',
        state_selection='24existing L16 training parents,8each tilted/interior_domain/single_wall;branch0 at frames3,8,selected by indices before future generation.48states from24source groups,not48independent preparations. No dev-state or dev-future training; no outcome selection.',
        data='Per state64train futures dt.25fs,128probe futures dt.25fs,128fine diagnostic futures dt.125fs. All0.1ps with2saved frames.15360short trajectories total;3072train. Probe/fine are unseen-noise at TRAIN-origin states,not independent-parent test. Seed streams disjoint and role-checked.',
        physics='Bauer open L16,exchange50meV,kappa.1,alpha.1,theta.11,T63.824849668K,zero field,unchanged MT5.9. Numerically normalized snapshot recorded with SHA. No interpolation,time relabelling,or relaxed reference dt.',
        controls='3arms x3seeds: native FM; native FM plus0.5restart FM every second update; same restart FM plus moment loss every16updates. Same adjacent step12000 weights/Adam,6000cosine3e-5->3e-6 updates; fixed final model. Native batch/source/tau/pair streams identical. Added batch8 uniformly sampled48states and64train futures uses separate paired streams. Extra weighting intentionally increases exposure to these L16states; does not isolate state coverage from that weighting. Same updates not equal compute.',
        moments='Reuse tested conditional_moment_repair local mean/full3x3 covariance independent32+32ensemble loss,64generated branches,midpoint128 differentiable. Uniform new state,all64TRAIN targets,weight ramps0->.1 over1000steps. Finite target error remains; loss can be negative. No Hamiltonian supervision. No architecture change.',
        evaluation='Before and after training:48TRAIN-origin states64generated futures mid128 against untouched probe/fine. After:reuse18dev128full1ps paths mid128/512,24oldseen64paths mid128,and72real-LLG dev intermediate states128futures mid128. Fixed state whole-branch bootstrap100. Separate seed and numerical variation; source groups correlated. No final holdout claim or automatic threshold.',
        decisions='If added restart FM improves both new-state probes and dev,coverage/weighting mattered. If only probes improve,generalization remains. If neither improves,inspect optimization/representation before enlarging data. Moment arm tests increment beyond identical added FM. Always report full1ps and oldanchor regressions; no promotion.',
        budget='First fat reference smoke(one parent,640futures) estimated1-3min;GPU smoke3arms1-3min each;formal23fat ref tasks1-3min each,9GPU jobs2-4h each,9fat reports5-15min,aggregate<3min. No long monitoring after submission.',
        skill_use='Karpathy surgical changes and experimental-design paired concurrent controls; skill provenance https://doi.org/10.48550/arXiv.2609.00065'))
    lock=json.loads((ROOT/'tests/conditional_moment_repair/source_lock.json').read_text())
    for name,digest in lock.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    files+=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json',HERE/'manifest.json',HERE/'states.h5']
    for p in files:lock[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


def verify_sources():
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


def reference(index):
    set_num_threads(2);out=HERE/'reference'/f'{index:02d}';out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    for row in [r for r in rows() if r['parent_index']==index]:
        sid=row['index'];a=state(row);config=json.loads(json.dumps(row['config']));config['numerics']['duration_ps']=.1
        for role,(n,dt,_) in ROLES.items():
            noise=seeds(sid,role);x,err=simulate(np.repeat(a[None],n,axis=0),noise,config,dt,.1)
            assert x.shape==(n,2,16,3) and np.isfinite(x).all() and abs(np.linalg.norm(x,axis=-1)-1).max()<1e-10
            with h5py.File(out/f'state{sid:03d}_{role}.h5','x') as h:
                h.attrs.update(complete=False,role=role,split='train',metadata=json.dumps(row),config_json=json.dumps(config),dt_fs=dt)
                h.create_dataset('spins',data=x,compression='lzf');h.create_dataset('seeds',data=noise)
                h.create_dataset('time_ps',data=[0.,.1]);h.create_dataset('preprojection_error',data=err);h.attrs['complete']=True
    dump(out/'complete.json',dict(parent_index=index,seconds=time.perf_counter()-tick,trajectories=640))


def load_train(smoke=False):
    bank=[];hashes={}
    for row in rows(smoke):
        sid=row['index'];p=HERE/'reference'/f"{row['parent_index']:02d}"/f'state{sid:03d}_train.h5'
        with h5py.File(p) as h:
            assert h.attrs['complete'] and h.attrs['role']=='train' and h.attrs['split']=='train'
            assert json.loads(h.attrs['metadata'])==row
            assert np.array_equal(h['seeds'][:],seeds(sid,'train')) and np.allclose(h['time_ps'][:],[0.,.1],atol=1e-12)
            x=h['spins'][:].astype(np.float32);assert x.shape==(64,2,16,3)
            assert abs(x[:,0]-state(row)).max()<1e-7;bank.append(x)
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return bank,hashes


def audit():
    bank,hashes=load_train();assert len(bank)==48
    for row in rows():
        sid=row['index']
        for role in ('probe','fine'):
            with h5py.File(HERE/'reference'/f"{row['parent_index']:02d}"/f'state{sid:03d}_{role}.h5') as h:
                assert h.attrs['complete'] and h.attrs['role']==role
                assert np.array_equal(h['seeds'][:],seeds(sid,role))
                assert h['spins'].shape==(128,2,16,3)
                assert np.isfinite(h['spins'][:]).all() and abs(h['spins'][:,0]-state(row)).max()<1e-10
    dump(HERE/'data_audit.json',dict(states=48,source_groups=24,training_futures=3072,diagnostic_futures=12288,hashes=hashes,passed=True,physical_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','reference','audit']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        verify_sources()
        if a.action=='reference':reference(a.index)
        else:audit()
