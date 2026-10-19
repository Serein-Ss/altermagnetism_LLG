"""Check rotated weak-solver ensembles; no claim that MT discrete noise is SO(2)-exact."""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from numba import set_num_threads
from tests.readout_activation_repair.diagnostics import ablation_rows
from tests.intermediate_restart_repair.run import initial
from tests.bauer_active_data_20260918.run import simulate
from tests.bauer_p2_20260923.common import dump
from tests.transition_block_repair.metrics import compare_first
from tests.conditional_moment_repair.report import metrics
from .symmetry import rotate


if __name__=='__main__':
    set_num_threads(2);torch.set_num_threads(1);here=Path(__file__).resolve().parent;root=here.parents[1];result=[]
    with h5py.File(here/'rotated_reference.h5','x') as out:
        out.attrs['complete']=False
        for k,row in enumerate(ablation_rows()):
            a=initial(row);cfg=json.loads(json.dumps(row['config']));cfg['numerics']['duration_ps']=.1
            with h5py.File(root/f"tests/intermediate_restart_repair/reference/{row['parent_index']:02d}/paths.h5") as h:
                old=h[f"state{row['index']:03d}/fine/spins"][:]
            for j,angle in enumerate((np.pi/4,np.pi/2)):
                b=rotate(torch.tensor(a),angle).numpy();b=b/np.linalg.norm(b,axis=-1,keepdims=True)
                rotated_old=rotate(torch.tensor(old),angle).numpy();r={}
                for q,dt in enumerate((.25,.125)):
                    seeds=2100000000+k*10000+j*1000+q*200+np.arange(128)
                    x,_=simulate(np.repeat(b[None],128,axis=0),seeds,cfg,dt,.1);r[dt]=x
                    g=out.create_group(f'{k}/{j}/{q}');g.create_dataset('spins',data=x,compression='lzf');g.create_dataset('seeds',data=seeds)
                    g.attrs.update(metadata=json.dumps(row),angle=angle,dt_fs=dt)
                item=dict(parent=row['parent']['parent_id'],family=row['parent']['family'],angle=angle,
                    fine_vs_rotated_fine=compare_first(r[.125],rotated_old),base_vs_fine=compare_first(r[.25],r[.125]))
                result.append(item);print(k,j,metrics(item['fine_vs_rotated_fine']),flush=True)
        out.attrs['complete']=True
    dump(here/'reference_audit.json',dict(rows=result,scope='Six fixed DEV L16 states;independent128branch ensembles;MT noise not exactly rotation invariant at finite dt;wholebranchbootstrap100;not automatically an equivalence test'))
