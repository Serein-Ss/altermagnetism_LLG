"""New independent LLG futures at model-visited TRAIN-parent states."""
import argparse,hashlib,json,time
from pathlib import Path
import h5py
import numpy as np
from numba import set_num_threads
from tests.bauer_p2_20260923.common import dump,energy
from tests.training_restart_repair.data import rows as real_rows,state as real_state
from tests.intermediate_restart_repair.run import normalize_state
from tests.bauer_active_data_20260918.run import simulate
from tests.model_state_replay_repair.run import verify as verify_previous

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ROLES={'train':(64,.25,0),'probe':(128,.25,1000),'fine':(128,.125,2000)}

def seeds(sid,role):
    if not 0<=sid<48 or role not in ROLES:raise ValueError('invalid role/state')
    n,_,offset=ROLES[role]
    return 1980000000+sid*10000+offset+np.arange(n,dtype=np.int64)

def source(seed):
    return ROOT/f'tests/model_state_replay_repair/train/{(11,22,33).index(seed)*2+1:02d}'

def rows(smoke=False):
    r=json.loads((HERE/'manifest.json').read_text());return r[:2] if smoke else r

def state(row):
    with h5py.File(HERE/'states.h5') as h:
        assert h.attrs['complete'];a=h[f"state{row['index']:03d}"][:]
    assert hashlib.sha256(a.tobytes()).hexdigest()==row['state_sha256']
    return a

def prepare():
    assert not (HERE/'source_lock.json').exists();verify_previous()
    real=real_rows();dev=json.loads((ROOT/'tests/bauer_p2_20260923/manifest_dev.json').read_text())
    assert len(real)==48 and not {r['parent']['parent_id'] for r in real}&{r['parent_id'] for r in dev}
    path=source(11)/'paths.h5';records=[]
    with h5py.File(path) as src,h5py.File(HERE/'states.h5','x') as out:
        assert src.attrs['complete'];out.attrs.update(complete=False,split='train',role='model_visited_states')
        for old in real:
            i,sid=old['parent_index'],old['index'];g=src[f'seen/parent{i:02d}']
            assert json.loads(g.attrs['parent'])==old['parent'] and old['parent']['split']=='train'
            assert np.allclose(g['time_ps'][:],np.arange(11)*.1,atol=1e-12)
            a=normalize_state(g['midpoint128'][0,old['frame']]);out.create_dataset(f'state{sid:03d}',data=a)
            row=dict(old,origin='replay_seed11_step78144_model',source_path=str(path.relative_to(ROOT)),
                source_key=f'seen/parent{i:02d}/midpoint128',branch=0,
                state_sha256=hashlib.sha256(a.tobytes()).hexdigest(),generating_checkpoint=str((source(11)/'checkpoint.pt').relative_to(ROOT)))
            records.append(row)
        out.attrs['complete']=True
    # Check known physical seed banks instead of relying on the numeric prefix.
    used=set();paths=list((ROOT/'tests/bauer_active_data_20260918/data').glob('L*/*.h5'))
    for directory in ('branch_coverage_repair','reference_resolution','intermediate_restart_repair','training_restart_repair','model_state_coverage_repair'):
        paths+=list((ROOT/f'tests/{directory}/reference').glob('**/*.h5'))
    for p in paths:
        with h5py.File(p) as h:
            def visit(name,obj):
                if isinstance(obj,h5py.Dataset) and name.split('/')[-1]=='seeds':used.update(map(int,obj[:]))
            h.visititems(visit)
    fresh=set()
    for sid in range(48):
        for role in ROLES:
            values=set(map(int,seeds(sid,role)));assert not values&(used|fresh);fresh|=values
    dump(HERE/'manifest.json',records)
    coverage=[]
    for old,row in zip(real,records):
        for origin,a in (('llg',real_state(old)),('model',state(row))):
            coverage.append(dict(state=row['index'],family=row['parent']['family'],origin=origin,
                source_time_ps=row['source_time_ps'],energy_over_NJ=float(energy(a)),mz=float(a[:,2].mean()),
                neighbor_alignment=float((a[:-1]*a[1:]).sum(-1).mean())))
    dump(HERE/'state_comparison.json',coverage)
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json',HERE/'manifest.json',HERE/'states.h5',path]
    for seed in (11,22,33):files += [source(seed)/'checkpoint.pt',source(seed)/'after_restart.h5']
    lock={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    dump(HERE/'source_lock.json',lock);(HERE/'logs').mkdir(exist_ok=True)

def verify():
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name

def reference(index):
    set_num_threads(2);out=HERE/'reference'/f'{index:02d}';out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter()
    for row in [r for r in rows() if r['parent_index']==index]:
        sid=row['index'];a=state(row);config=json.loads(json.dumps(row['config']));config['numerics']['duration_ps']=.1
        for role,(n,dt,_) in ROLES.items():
            noise=seeds(sid,role);x,err=simulate(np.repeat(a[None],n,axis=0),noise,config,dt,.1)
            assert x.shape==(n,2,16,3) and np.isfinite(x).all() and abs(np.linalg.norm(x,axis=-1)-1).max()<1e-10
            assert np.array_equal(x[:,0],np.repeat(a[None],n,axis=0))
            with h5py.File(out/f'state{sid:03d}_{role}.h5','x') as h:
                h.attrs.update(complete=False,role=role,split='train',metadata=json.dumps(row),config_json=json.dumps(config),dt_fs=dt)
                h.create_dataset('spins',data=x,compression='lzf');h.create_dataset('seeds',data=noise)
                h.create_dataset('time_ps',data=[0.,.1]);h.create_dataset('preprojection_error',data=err);h.attrs['complete']=True
    dump(out/'complete.json',dict(parent_index=index,seconds=time.perf_counter()-tick,trajectories=640))

def load_train(smoke=False):
    bank=[]
    for row in rows(smoke):
        with h5py.File(HERE/'reference'/f"{row['parent_index']:02d}"/f"state{row['index']:03d}_train.h5") as h:
            assert h.attrs['complete'] and h.attrs['role']=='train' and h.attrs['split']=='train'
            assert json.loads(h.attrs['metadata'])==row
            assert np.array_equal(h['seeds'][:],seeds(row['index'],'train'))
            bank.append(h['spins'][:].astype(np.float32))
    return bank

def audit():
    load_train();counts=0;hashes={}
    for row in rows():
        for role,(n,dt,_) in ROLES.items():
            with h5py.File(HERE/'reference'/f"{row['parent_index']:02d}"/f"state{row['index']:03d}_{role}.h5") as h:
                assert h.attrs['complete'] and h.attrs['role']==role and h.attrs['dt_fs']==dt
                assert json.loads(h.attrs['metadata'])==row
                assert np.array_equal(h['seeds'][:],seeds(row['index'],role))
                assert h['spins'].shape==(n,2,16,3);counts+=n
                x=h['spins'][:];assert np.isfinite(x).all() and abs(np.linalg.norm(x,axis=-1)-1).max()<1e-10
                assert np.array_equal(x[:,0],np.repeat(state(row)[None],n,axis=0))
                assert np.allclose(h['time_ps'][:],[0.,.1],atol=1e-12)
                config=json.loads(h.attrs['config_json']);expected=json.loads(json.dumps(row['config']));expected['numerics']['duration_ps']=.1
                assert config==expected
                name=f"state{row['index']:03d}_{role}.h5";path=HERE/'reference'/f"{row['parent_index']:02d}"/name
                hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    dump(HERE/'data_audit.json',dict(passed=True,states=48,source_groups=24,trajectories=counts,train_futures=3072,
        probe_fine_futures=12288,hashes=hashes,physical_certification=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','reference','audit']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        verify()
        reference(a.index) if a.action=='reference' else audit()
