"""No training: matched latent tests of a Bauer-specific symmetry projection."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from tests.bauer_p2_20260923.common import dump
from tests.bauer_comprehensive_20260923.run import require_gpu
from tests.reference_resolution.run import parents
from tests.intermediate_restart_repair.run import selected, initial
from tests.transition_block_repair.model import TransitionNet, rollout
from .symmetry import probe
from .model import AxialProjection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def tasks():
    return [dict(seed=s, order=k) for s in (11,22,33) for k in (1,4,8)]


def source(seed):
    return ROOT / f'tests/axial_symmetry_repair/train/{(11,22,33).index(seed)*2+1:02d}'


def dev_rows():
    return [(i,p) for i,p in enumerate(parents()) if p['L']==16]


def middle_rows():
    return [r for r in selected() if r['origin']=='llg' and r['parent']['L']==16]


def prepare():
    assert not (HERE/'source_lock.json').exists()
    assert len(dev_rows())==6 and len(middle_rows())==24
    files = list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[HERE/'protocol.json']
    files += [ROOT/p for p in (
        'tests/axial_symmetry_repair/symmetry.py','tests/axial_symmetry_repair/reference_audit.json',
        'tests/axial_symmetry_repair/assets/summary.json','tests/transition_block_repair/model.py',
        'tests/spin_path_transformer/model.py','tests/temporal_source_repair/source.py',
        'tests/transport_audit/run.py','tests/reference_resolution/run.py',
        'tests/reference_resolution/report.py','tests/transition_block_repair/metrics.py',
        'tests/intermediate_restart_repair/run.py','tests/intermediate_restart_repair/manifest.json',
        'tests/intermediate_restart_repair/states.h5','tests/bauer_p2_20260923/common.py')]
    for seed in (11,22,33):
        files += [source(seed)/'checkpoint.pt',source(seed)/'complete.json']
        c=json.loads((source(seed)/'complete.json').read_text())
        assert c['step']==42000 and c['task']['repair']=='axial_aug' and c['task']['seed']==seed
    for i,_ in dev_rows():
        files += [ROOT/f'tests/reference_resolution/reference/{i:02d}/paths.h5',
                  ROOT/f'tests/intermediate_restart_repair/reference/{i:02d}/paths.h5']
    dump(HERE/'source_lock.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def verify():
    for name,digest in json.loads((HERE/'source_lock.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


@torch.no_grad()
def run(index,smoke):
    require_gpu('cuda');torch.set_num_threads(1)
    task=tasks()[index]
    ck=torch.load(source(task['seed'])/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert ck['step']==42000 and ck['task']['repair']=='axial_aug'
    base=TransitionNet().cuda().eval();base.load_state_dict(ck['model'])
    net=AxialProjection(base,task['order']).eval()
    folder=HERE/('smoke' if smoke else 'runs')/f'{index:02d}'
    folder.mkdir(parents=True,exist_ok=False)
    dump(folder/'symmetry.json',probe(net))
    records=[];stream=hashlib.sha256();tick=time.perf_counter()
    with h5py.File(folder/'paths.h5','x') as h:
        h.attrs['complete']=False;h.attrs['model_contract']=json.dumps(net.get_extra_state())
        for role,rows in (('dev',dev_rows()),('middle',middle_rows())):
            for row in rows[:1] if smoke else rows:
                if role=='dev':
                    i,meta=row
                    with h5py.File(ROOT/f'tests/reference_resolution/reference/{i:02d}/paths.h5') as ref:
                        assert ref.attrs['complete'];a0=ref['base/spins'][0,0]
                    name=f'parent{i:02d}';seed=939100+i;factor=100;transitions=10
                else:
                    meta=row;i=row['index'];a0=initial(row)
                    name=f'state{i:03d}';seed=941100+i;factor=10000;transitions=1
                n=4 if smoke else 128;steps=4 if smoke else 128
                a=F.normalize(torch.tensor(a0,device='cuda',dtype=torch.float32),dim=-1)[None].expand(n,-1,-1)
                g=h.create_group(f'{role}/{name}');g.attrs['metadata']=json.dumps(meta)
                g.create_dataset('time_ps',data=np.arange(transitions+1)*.1)
                for level in (steps,steps*4):
                    # Main sampler: complete 1ps; refined sampler: first0.1ps only.
                    count=transitions if level==steps else 1
                    pieces=[];states=[];torch.cuda.synchronize();start=time.perf_counter()
                    for b in range(0,n,64):
                        gen=torch.Generator(device='cuda').manual_seed(seed+b*factor)
                        stream.update(gen.get_state().cpu().numpy().tobytes())
                        stream.update(a[b:b+64].cpu().numpy().tobytes())
                        pieces.append(rollout(net,a[b:b+64],gen,level,transitions=count).cpu().numpy())
                        states.append(gen.get_state().cpu().numpy())
                    torch.cuda.synchronize();x=np.concatenate(pieces);elapsed=time.perf_counter()-start
                    norm=float(abs(np.linalg.norm(x,axis=-1)-1).max());anchor=float(abs(x[:,0]-a.cpu().numpy()).max())
                    assert np.isfinite(x).all() and norm<2e-6 and anchor==0
                    g.create_dataset(f'midpoint{level}',data=x,compression='lzf')
                    g.create_dataset(f'rng_final{level}',data=np.stack(states))
                    records.append(dict(role=role,id=i,steps=level,transitions=count,n=n,seconds=elapsed,
                        norm_error=norm,anchor_error=anchor,backbone_calls_per_path=2*level*count*task['order']))
                print(task,role,i,flush=True)
        h.attrs['complete']=True
    dump(folder/'complete.json',dict(task=task,source=str(source(task['seed']).relative_to(ROOT)),
        step=42000,paired_input_hash=stream.hexdigest(),records=records,
        seconds=time.perf_counter()-tick,device=torch.cuda.get_device_name(),scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','smoke','run'])
    p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:verify();run(a.index,a.action=='smoke')
