"""Current-policy TRAIN states; independent true-LLG labels, not model labels."""
import argparse,hashlib,json,time
from pathlib import Path
import h5py
import numpy as np
from numba import set_num_threads
from tests.bauer_p2_20260923.common import dump,energy
from tests.training_restart_repair.data import rows as real_rows
from tests.model_state_refresh_repair import data as old
from tests.intermediate_restart_repair.run import normalize_state
from tests.bauer_active_data_20260918.run import simulate
from tests.local_teacher_repair.run import verify as verify_previous

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SEEDS=(11,22,33)
ROLES={'train':(64,.25,0),'probe':(128,.25,1000),'fine':(128,.125,2000)}


def source(seed):
    return ROOT/f'tests/local_teacher_repair/train/{SEEDS.index(seed)*2:02d}'


def seeds(seed,sid,role):
    if seed not in SEEDS or not 0<=sid<48 or role not in ROLES:raise ValueError('unregistered state/seed/role')
    n,_,offset=ROLES[role]
    return 2020000000+SEEDS.index(seed)*1000000+sid*10000+offset+np.arange(n,dtype=np.int64)


def rows(seed,smoke=False):
    records=json.loads((HERE/'manifest.json').read_text());r=[x for x in records if x['model_seed']==seed]
    assert len(r)==48
    return r[:2] if smoke else r


def state(row):
    with h5py.File(HERE/'states.h5') as h:
        assert h.attrs['complete'];a=h[f"seed{row['model_seed']}/state{row['index']:03d}"][:]
    assert hashlib.sha256(a.tobytes()).hexdigest()==row['state_sha256']
    return a


def descriptor(a):
    bond=(a[:-1]*a[1:]).sum(-1)
    return np.r_[
        a[:,2].mean(),a[:,2].std(),energy(a),np.quantile(a[:,2],[.1,.5,.9]),np.quantile(bond,[.1,.5,.9]),a[0,2],a[-1,2],np.mean(np.diff(a,axis=0)**2)]


def prepare():
    assert not (HERE/'source_lock.json').exists();verify_previous()
    base=real_rows();dev=json.loads((ROOT/'tests/bauer_p2_20260923/manifest_dev.json').read_text())
    assert len(base)==48 and not {r['source_group'] for r in base}&{r['parent_id'] for r in dev}
    old_states=np.stack([old.state(r) for r in old.rows()]);features=np.stack([descriptor(a) for a in old_states])
    scale=features.std(0).clip(1e-6);records=[];coverage=[];sources=[]
    with h5py.File(HERE/'states.h5','x') as out:
        out.attrs.update(complete=False,split='train')
        for seed in SEEDS:
            path=source(seed)/'paths.h5';sources += [path,source(seed)/'checkpoint.pt',source(seed)/'report.json']
            with h5py.File(path) as src:
                assert src.attrs['complete']
                for row in base:
                    i,sid=row['parent_index'],row['index'];g=src[f'seen/parent{i:02d}']
                    assert json.loads(g.attrs['parent'])==row['parent'] and row['parent']['split']=='train'
                    assert np.allclose(g['time_ps'][:],np.arange(11)*.1,atol=1e-12)
                    a=normalize_state(g['midpoint128'][0,row['frame']]);anchor=a[[0,1,-2,-1]]
                    assert (anchor.reshape(2,2,3)[...,:2]**2).sum((1,2)).min()>1e-12
                    out.create_dataset(f'seed{seed}/state{sid:03d}',data=a)
                    item=dict(row,model_seed=seed,origin=f'current_seed{seed}_step130240',source_path=str(path.relative_to(ROOT)),
                        source_key=f'seen/parent{i:02d}/midpoint128',branch=0,state_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                        generating_checkpoint=str((source(seed)/'checkpoint.pt').relative_to(ROOT)))
                    records.append(item)
                    coverage.append(dict(model_seed=seed,state=sid,source_group=row['source_group'],family=row['parent']['family'],
                        source_time_ps=row['source_time_ps'],old_to_current_mean_angle=float(np.arccos(np.clip((a*old_states[sid]).sum(-1),-1,1)).mean()),
                        old_only_scaled_descriptor_distance=float(np.linalg.norm((features-descriptor(a))/scale,axis=1).min()),
                        energy_over_NJ=float(energy(a)),mz=float(a[:,2].mean())))
        out.attrs['complete']=True
    # Audit actual known ancestral physical seed banks, not only namespace prefixes.
    used=set();paths=list((ROOT/'tests/bauer_active_data_20260918/data').glob('L*/*.h5'))
    for folder in ('branch_coverage_repair','reference_resolution','intermediate_restart_repair','training_restart_repair','model_state_coverage_repair','model_state_refresh_repair'):
        paths+=list((ROOT/f'tests/{folder}/reference').glob('**/*.h5'))
    for p in paths:
        with h5py.File(p) as h:
            def visit(name,obj):
                if isinstance(obj,h5py.Dataset) and name.split('/')[-1]=='seeds':used.update(map(int,obj[:]))
            h.visititems(visit)
    fresh=set()
    for seed in SEEDS:
        for sid in range(48):
            for role in ROLES:
                values=set(map(int,seeds(seed,sid,role)));assert not values&(used|fresh);fresh|=values
    dump(HERE/'manifest.json',records);dump(HERE/'coverage.json',dict(rows=coverage,scope='fixed TRAIN snapshots, correlated by24 sources; axial-invariant old-only scaled descriptors are descriptive, not complete state coverage or a pass threshold'))
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json',HERE/'manifest.json',HERE/'states.h5',HERE/'coverage.json']+sources
    lock={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)


def verify():
    verify_previous()
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


def path(row,role):
    return HERE/f"reference/seed{row['model_seed']}/{row['parent_index']:02d}/state{row['index']:03d}_{role}.h5"


def reference(index):
    set_num_threads(2);tick=time.perf_counter();count=0
    for seed in SEEDS:
        for row in [r for r in rows(seed) if r['parent_index']==index]:
            a=state(row);config=json.loads(json.dumps(row['config']));config['numerics']['duration_ps']=.1
            for role,(n,dt,_) in ROLES.items():
                noise=seeds(seed,row['index'],role);x,error=simulate(np.repeat(a[None],n,axis=0),noise,config,dt,.1)
                assert x.shape==(n,2,16,3) and np.isfinite(x).all() and abs(np.linalg.norm(x,axis=-1)-1).max()<1e-10
                assert np.array_equal(x[:,0],np.repeat(a[None],n,axis=0))
                p=path(row,role);p.parent.mkdir(parents=True,exist_ok=True)
                with h5py.File(p,'x') as h:
                    h.attrs.update(complete=False,role=role,split='train',metadata=json.dumps(row),config_json=json.dumps(config),dt_fs=dt)
                    h.create_dataset('spins',data=x,compression='lzf');h.create_dataset('seeds',data=noise)
                    h.create_dataset('time_ps',data=[0.,.1]);h.create_dataset('preprojection_error',data=error);h.attrs['complete']=True
                count+=n
    dump(HERE/f'reference/complete_{index:02d}.json',dict(parent_index=index,seconds=time.perf_counter()-tick,trajectories=count))


def load_train(seed,smoke=False):
    bank=[];hashes={}
    for row in rows(seed,smoke):
        p=path(row,'train')
        with h5py.File(p) as h:
            assert h.attrs['complete'] and h.attrs['role']=='train' and h.attrs['split']=='train'
            assert json.loads(h.attrs['metadata'])==row and np.array_equal(h['seeds'][:],seeds(seed,row['index'],'train'))
            x=h['spins'][:];assert x.shape==(64,2,16,3) and np.array_equal(x[:,0],np.repeat(state(row)[None],64,axis=0))
            bank.append(x.astype(np.float32))
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return bank,hashes


def audit(smoke=False):
    hashes={};count=0
    for seed in SEEDS:
        _,training_hashes=load_train(seed,smoke);hashes.update(training_hashes)
        for row in rows(seed,smoke):
            for role,(n,dt,_) in ROLES.items():
                p=path(row,role)
                with h5py.File(p) as h:
                    assert h.attrs['complete'] and h.attrs['role']==role and h.attrs['dt_fs']==dt
                    assert json.loads(h.attrs['metadata'])==row and np.array_equal(h['seeds'][:],seeds(seed,row['index'],role))
                    x=h['spins'][:];assert x.shape==(n,2,16,3) and np.isfinite(x).all()
                    assert abs(np.linalg.norm(x,axis=-1)-1).max()<1e-10 and np.array_equal(x[:,0],np.repeat(state(row)[None],n,axis=0))
                    assert np.allclose(h['time_ps'][:],[0.,.1],atol=1e-12)
                    expected=json.loads(json.dumps(row['config']));expected['numerics']['duration_ps']=.1
                    assert json.loads(h.attrs['config_json'])==expected
                hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();count+=n
    dump(HERE/('smoke_data_audit.json' if smoke else 'data_audit.json'),dict(passed=True,smoke=smoke,states=6 if smoke else 144,source_groups=1 if smoke else 24,trajectories=count,hashes=hashes,scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','reference','audit']);p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        verify()
        reference(a.index) if a.action=='reference' else audit(a.smoke)
