"""Read-only data audit plus new experiment protocol; never modifies frozen data."""
import hashlib
import sys
import json
from collections import Counter
import h5py
import numpy as np
import yaml
from tests.miltat_library_20260920.data import validate_manifest
from tests.bauer_p2_20260923.common import dump
from .run import HERE,ROOT,DATA,tasks


def main():
    if (HERE/'submission.json').exists():raise RuntimeError('already submitted')
    rows=validate_manifest(DATA);physics=[]
    for row in rows:
        if row['split']=='test':continue
        with h5py.File(DATA/row['file']) as h:
            p=json.loads(h.attrs['physical_parameters_json'])
            assert p['temperature_K']==12.5 and p['alpha']==.5 and p['boundary']=='open'
            assert p['J_meV']==29 and p['D_meV']==-1.5 and p['K_meV']==.4
            assert np.isclose(h['time_ps'][1],.1)
            physics.append(dict(parent_id=row['parent_id'],physics=p,shape=list(h['spins'].shape),
                                source_bytes=(DATA/row['file']).stat().st_size))
    release=json.loads((DATA.parent/'release.json').read_text())
    for key,path in [('config',DATA.parent/'config.yaml'),('layout',DATA.parent/'layout.json'),
                     ('source',ROOT/'tests/literature_preflight_20260917/config.yaml')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==release[key+'_sha256']
    counts=Counter((tuple(r['cell_shape']),r['split']) for r in rows)
    dump(HERE/'data_audit.json',dict(counts={str(k):v for k,v in counts.items()},parents=physics,
         physical_release='engineering release, not blanket equivalence certification',
         reference_dt_fs=.3125,source_manifest_sha256=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
         source_config=yaml.safe_load((ROOT/'tests/literature_preflight_20260917/config.yaml').read_text())['systems']['miltat'],
         provenance_holds='Retain source gamma/moving-window/literature caveats; not a new claim of exact paper reproduction.'))
    protocol=dict(tasks=tasks(),two_d=dict(training_shape=[64,38],basis=2,temperature_K=12.5,
        alpha=.5,periodic=[False,False],reference_dt_fs=.3125,frame_dt_ps=.1,block_ps=1.,
        train_parents=12,reference_branches=8,dev_parents=4,generated_branches=16,
        objective='FM only; no Hamiltonian coefficients/field supplied to network; geometry supplied',
        controls='same graph width64 depth3: shifted source; uniform source; shifted plus global vector means',
        budget='6000 updates, batch1 full space, 50% first windows / 50% all windows; identical data/tau/noise streams within seed',
        checkpoint='fixed final update primary; 1500/3000 snapshots retained, not best-of-test',
        limitations='Global means are not completed hierarchical multiscale; short-window only; constant temperature; actual costs separately'),
        synthetic=dict(modes=[1,2],parents_train=24,parents_dev=4,seeds=[11,22,33],
            controls=['sample-target FM','same FM plus relational global residual','exact conditional-velocity supervision'],
            source='independent uniform spheres, fixed actual S0; different declared source from prior shifted repair',
            target='one or two equally weighted deterministic whole-path rotations, no local noise',
            time='dimensionless synthetic coordinate; NO ps meaning',transport_stop=.95,
            law='compare against exact geodesic-interpolated law at .95, not delta endpoint; exact oracle support violations reported',
            oracle_steps=[16,64,256,1024,4096],neural_steps=[16,64,128,512]),
        diagnoses=['input/data provenance and split separation','2D oriented geometry, tangent, norm, SO3, energy pair sum',
            'train versus dev, seeds separately, no pooled-mode certification','source versus global communication',
            'full conditional velocity oracle versus sampled pair targets','sampler convergence with same latent draws',
            'full energy/magnetization/increment covariance/two-time/spatial/low-q/topology diagnostics',
            'whole-branch bootstrap and descriptive reference-reference floor','two same-boundary larger sizes labelled exploratory'],
        scope='This is a bounded diagnostic matrix, not proof every conceivable cause has been excluded. No production promotion.',
        reporting='PNG, 2D GIF, downloadable self-contained HTML; scheduled afterany report shows failed/missing tasks too')
    dump(HERE/'protocol.json',protocol)
    lock={}
    for p in list(HERE.glob('*.py'))+list(HERE.glob('*.sh'))+[DATA/'manifest.json',DATA.parent/'config.yaml',HERE/'protocol.json']:
        lock[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            from pathlib import Path
            p=Path(name).resolve()
            if p.is_relative_to(ROOT) and p.suffix=='.py' and p.is_file():
                lock[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [ROOT/'tests/bauer_p2_20260923/config.yaml',ROOT/'tests/literature_preflight_20260917/config.yaml',DATA.parent/'release.json']:
        lock[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    dump(HERE/'source_lock.json',lock)
    print('audited',dict(counts),flush=True)


if __name__=='__main__':main()
