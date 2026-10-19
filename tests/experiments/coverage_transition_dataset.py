"""Replace half the pilot paths, preserving pair count and temperature ordering."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest
from tests.experiments.fine_transition_dataset import FineTransitionDataset


def replacement_index(index):
    group,offset=divmod(index,256*8)
    path,window=divmod(offset,8)
    if path<128: return None
    family,new_path=divmod(path-128,32)
    return group*4+family,new_path,window


class CoverageTransitionDataset:
    def __init__(self):
        self.original=FineTransitionDataset('train')
        root=Path('tests/results/coverage_intervention_data')
        report=json.loads((root/'summary.json').read_text())
        assert report['status']=='COMPLETE_COVERAGE_INTERVENTION_DATA_NOT_CERTIFIED'
        assert digest(root/'manifest.json')==report['manifest_sha256']
        self.rows={r['anchor']:r for r in report['rows']}
        assert len(report['rows'])==24 and set(self.rows)==set(range(24))
        assert len(self.original)==12288 and list(self.original.starts)==list(range(0,32,4))
        self.cache_index=None; self.cache=None

    def __len__(self): return len(self.original)

    def __getitem__(self,index):
        if not 0<=index<len(self): raise IndexError(index)
        replacement=replacement_index(index)
        if replacement is None: return self.original[index]
        anchor,path,window=replacement; row=self.rows[anchor]
        group=index//2048
        assert row['state']['theta']==self.original.rows[group]['state']['theta']
        if self.cache_index!=anchor:
            assert digest(row['path'])==row['sha256']
            with np.load(row['path']) as z:
                spins=z['spins']; times=z['reduced_time']; seeds=z['seeds']
            assert spins.shape==(32,33,25,3) and np.isfinite(spins).all()
            assert np.max(abs(np.linalg.norm(spins,axis=-1)-1))<1e-10
            np.testing.assert_allclose(times,np.arange(33)*.0025,atol=1e-12,rtol=0)
            assert seeds.shape==(32,) and len(np.unique(seeds))==32
            self.cache=(spins,seeds); self.cache_index=anchor
        spins,seeds=self.cache; start=window*4
        return dict(initial=spins[path,start].copy(),target=spins[path,start+4].copy(),
            theta=float(row['state']['theta']),lag=.01,anchor=anchor,noise_seed=int(seeds[path]),
            start=start,source='coverage_intervention')
