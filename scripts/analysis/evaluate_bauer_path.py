"""Descriptive fixed-parent native-lag comparison, not an acceptance gate."""
import argparse,json
from pathlib import Path
import h5py
import numpy as np
from scripts.analysis.bauer_path_metrics import compare_first,compare_spatial


def read(path):
    with h5py.File(path) as h:
        if not h.attrs.get('complete',False):raise ValueError('incomplete trajectories')
        t=h['time_ps'][:]
        if len(t)<2 or not np.allclose(t[:2],[0.,.1],atol=1e-8,rtol=0):raise ValueError('native [0,0.1]ps required')
        x=h['spins'][:,:2]
    if x.ndim!=4 or x.shape[-1]!=3 or not np.isfinite(x).all():raise ValueError('invalid spin trajectories')
    if not np.allclose(np.linalg.norm(x,axis=-1),1.,atol=2e-6,rtol=0):raise ValueError('nonunit spins')
    return x


def evaluate(generated,reference,repeats=100):
    g,r=read(generated),read(reference)
    if g.shape[1:]!=r.shape[1:]:raise ValueError('matching observation task required')
    for x in (g,r):
        if not np.allclose(x[:,0],r[:1,0],atol=1e-6,rtol=0):raise ValueError('same actual initial state required')
    return dict(generated=str(generated),reference=str(reference),physical_times_ps=[0.,.1],
        branches=[len(g),len(r)],metrics=compare_first(g,r,repeats),spatial=compare_spatial(g,r),
        scientific_certification=False,scope='Fixed-parent descriptive native-lag statistics; whole-branch independent bootstrap, pointwise not simultaneous. Reference provenance and numerical convergence must be audited separately. Not full-path/transfer certification or an event acceptance gate.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generated',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True);p.add_argument('--bootstrap',type=int,default=100);a=p.parse_args()
    if a.bootstrap<0:raise ValueError('nonnegative bootstrap count')
    result=evaluate(a.generated,a.reference,a.bootstrap);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
