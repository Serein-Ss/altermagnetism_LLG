import json
import h5py
import numpy as np
from tests.mdgen_path_refactor_20260918.data import PathWindows
from tests.bauer_p2_20260923.common import cfg
from tests.deployment_moment_repair.run import banks
from .run import HERE,ROOT,dump,source,verify_previous


def main():
    verify_previous();config=cfg();records=[]
    def check(name,a):
        weight=np.asarray(a,dtype=np.float64)[...,:2]**2;total=weight.reshape(*weight.shape[:-2],2,2,2).sum((-1,-2));minimum=float(total.min())
        assert minimum>1e-12,(name,minimum)
        records.append(dict(source=name,states_including_related_duplicates=int(total.size),minimum_total_transverse=minimum))
    for L in config['sizes']:
        data=PathWindows([ROOT/config['data']/f'L{L}'],'train',1.,stride_frames=10)
        for path,row,starts,steps in data.records:
            with h5py.File(path) as h:check(str(path.relative_to(ROOT)),h['spins'][:,:1000][:,:,[0,1,-2,-1]])
    bank,_=banks()
    for role,values in enumerate(bank):
        for i,x in enumerate(values):check(f'train_bank_role{role}_state{i}',x[:,0][:,[0,1,-2,-1]])
    manifest=json.loads((ROOT/'tests/intermediate_restart_repair/manifest.json').read_text())
    with h5py.File(ROOT/'tests/intermediate_restart_repair/states.h5') as h:
        for r in manifest:
            if r['origin']=='llg':check(f'dev_intermediate{r["index"]}',h[f'state{r["index"]:03d}'][:][None][:,[0,1,-2,-1]])
    with h5py.File(source(11)/'paths.h5') as h:
        for name,g in h['dev'].items():check(f'dev_initial_{name}',g['midpoint128'][:1,0][:,[0,1,-2,-1]])
    dump(HERE/'audit.json',dict(all_supported=True,records=records,global_minimum_either_end=min(r['minimum_total_transverse'] for r in records),
        scope='Both ends checked separately. Eligible native TRAIN frame0:1000,all available extra TRAIN banks and reused DEV initial/LLG states; no untouched test inspected. Duplicate/related states are not independent coverage samples. Future rollout states checked at runtime.'))


if __name__=='__main__':main()
