"""Parent-grouped lazy windows for future Miltat libraries, not preflight files.

One HDF5 per independent parent: spins[branch,time,nx,ny,2,3], initial[nx,ny,2,3],
time_ps, noise_seeds, positions_nm; metadata repeated in manifest and HDF5.
Storage validation does NOT certify the physics or authorize dataset release.
"""
import json
from pathlib import Path
import h5py
import numpy as np
from torch.utils.data import Dataset
from .geometry import CoreGeometry


def validate_manifest(root,allow_smoke=False):
    root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('smoke',False) and not allow_smoke:raise ValueError('Software smoke data require explicit opt-in')
    if manifest['schema']!='miltat_parent_v1':raise ValueError('Unsupported library schema')
    records=manifest['parents'];parents=set();files=set();groups={};seeds=set()
    if not records:raise ValueError('Empty library')
    for row in records:
        parent,split,group=row['parent_id'],row['split'],row['preparation_group']
        if split not in ['train','dev','test'] or not group:raise ValueError('Invalid split or preparation group')
        if parent in parents:raise ValueError('Duplicate parent')
        parents.add(parent)
        if group in groups and groups[group]!=split:raise ValueError('Preparation ancestry crosses splits')
        groups[group]=split
        path=(root/row['file']).resolve()
        if not path.is_relative_to(root) or path in files:raise ValueError('Duplicate or out-of-root file')
        files.add(path)
        with h5py.File(path) as h:
            if h.attrs.get('smoke',False) and not allow_smoke:raise ValueError('Software smoke file requires explicit opt-in')
            if not h.attrs.get('complete',False) or h.attrs.get('schema')!='miltat_parent_v1':
                raise ValueError('Incomplete or diagnostic-only file')
            for name in ['parent_id','split','preparation_group']:
                if h.attrs[name]!=row[name]:raise ValueError('Manifest and file metadata disagree')
            shape=h['spins'].shape
            if len(shape)!=6 or shape[-2:]!=(2,3) or shape[0]<2 or shape[1]<2:
                raise ValueError('Expected branched 2D triangular spin paths')
            if 'cell_shape' in row and tuple(row['cell_shape'])!=shape[2:4]:
                raise ValueError('Manifest geometry differs from stored trajectory')
            initial=h['initial'][:]
            if initial.shape!=shape[2:] or h['positions_nm'].shape!=shape[2:]:
                raise ValueError('Initial state or coordinates have wrong shape')
            if not np.isfinite(initial).all() or not np.allclose(np.linalg.norm(initial,axis=-1),1,atol=1e-6,rtol=0):
                raise ValueError('Invalid initial spins')
            if not np.allclose(h['spins'][:,0],initial[None],atol=1e-7,rtol=0):
                raise ValueError('Branches do not share one parent initial state')
            t=h['time_ps'][:]
            if len(t)!=shape[1] or not np.isfinite(t).all() or t[0]!=0 or t[1]<=0 or not np.allclose(np.diff(t),t[1],rtol=1e-7,atol=1e-10):
                raise ValueError('Invalid physical time grid')
            noise=h['noise_seeds'][:]
            if noise.shape!=(shape[0],) or noise.dtype.kind not in 'iu':raise ValueError('Invalid branch seeds')
            for seed in noise:
                seed=int(seed)
                if seed in seeds:raise ValueError('Reused thermal noise seed')
                seeds.add(seed)
            physics=json.loads(h.attrs['physical_parameters_json'])
            if physics['boundary']!='open' or physics['temperature_K']<0:
                raise ValueError('Unexpected physical regime')
            for field in ['J_meV','K_meV','D_meV','moment_muB','a_angstrom','gamma_per_second_tesla','alpha']:
                if not np.isfinite(physics[field]):raise ValueError('Nonfinite physical parameters')
            if min(physics[k] for k in ['J_meV','moment_muB','a_angstrom','gamma_per_second_tesla'])<=0 or physics['alpha']<0:
                raise ValueError('Invalid physical scale')
            positions=h['positions_nm'][:]
            if not np.isfinite(positions).all():raise ValueError('Invalid coordinates')
            geometry=CoreGeometry(positions,spacing_nm=physics['a_angstrom']*.1)
            if not np.any(geometry.degree==6):raise ValueError('No triangular-lattice interior')
    return records


class MiltatWindows(Dataset):
    def __init__(self,root,split,duration_ps,stride_frames=1,allow_smoke=False):
        if split not in ['train','dev','test'] or duration_ps<=0 or not isinstance(stride_frames,int) or stride_frames<1:
            raise ValueError('Invalid split, duration or stride')
        self.root=Path(root).resolve();self.records=[];self.ends=[];total=0
        for row in validate_manifest(self.root,allow_smoke=allow_smoke):
            if row['split']!=split:continue
            with h5py.File(self.root/row['file']) as h:
                dt=float(h['time_ps'][1]);shape=h['spins'].shape
                steps=round(duration_ps/dt)
                if steps<1 or steps>=shape[1] or not np.isclose(steps*dt,duration_ps):
                    raise ValueError('Requested duration is not on the stored grid')
            starts=range(0,shape[1]-steps,stride_frames)
            total+=shape[0]*len(starts);self.ends.append(total)
            self.records.append((row,starts,steps))
        if not self.records:raise ValueError('No parents in requested split')

    def __len__(self):return self.ends[-1]

    def __getitem__(self,index):
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        group=int(np.searchsorted(self.ends,index,side='right'))
        offset=index-(self.ends[group-1] if group else 0)
        row,starts,steps=self.records[group];branch,window=divmod(offset,len(starts));start=starts[window]
        with h5py.File(self.root/row['file']) as h:
            x=h['spins'][branch,start:start+steps+1]
            times=h['time_ps'][start:start+steps+1]
            seed=int(h['noise_seeds'][branch]);positions=h['positions_nm'][:]
        if not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x,axis=-1),1,atol=1e-6,rtol=0):
            raise ValueError('Invalid stored trajectory window')
        return dict(spins=x,initial=x[0].copy(),time_ps=times-times[0],origin_ps=float(times[0]),
                    positions_nm=positions,parent_id=row['parent_id'],split=row['split'],
                    preparation_group=row['preparation_group'],noise_seed=seed,branch=branch)
