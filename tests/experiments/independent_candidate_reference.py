"""Independent candidate numerical references; no training or bank regeneration."""
import json
import os
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.independent_reference import generate
from tests.experiments.audit_independent_candidate import reference_seeds

CONFIG = Path("tests/conf/independent_candidate_validation.json")
ROOT = Path("tests/results/independent_candidate_reference")
RAW = Path("tests/data/independent_candidate_reference")

def run():
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('CPU Slurm required')
    manifest = json.loads((ROOT/'manifest.json').read_text())
    audit=json.loads((ROOT/'isolation_audit.json').read_text())
    assert audit['manifest_sha256']==digest(ROOT/'manifest.json')
    assert audit['status']=='NO_OVERLAP_IN_AUDITED_SOURCES'
    cfg = manifest['config']
    assert digest(CONFIG) == manifest['config_sha256']
    assert digest(RAW/'anchors.npz') == manifest['bank_sha256']
    for p, h in manifest['code_sha256'].items():
        assert digest(p) == h
    dest = ROOT/'summary.json'
    if dest.exists():
        raise FileExistsError('no silent retry')
    with np.load(RAW/'anchors.npz') as z:
        states = z['unit_initials']
    report = dict(status='RUNNING', rows=[], manifest_sha256=digest(ROOT/'manifest.json'), job_id=os.environ['SLURM_JOB_ID'],
        code_sha256={p:digest(p) for p in [__file__, 'tests/experiments/independent_reference.py', 'tests/experiments/audit_independent_candidate.py', 'scripts/core/bauer_fast.py']})
    save(dest, report)
    for ci, ai in manifest['schedule']:
        a = manifest['anchors'][ai]
        n = cfg['paths_per_anchor']
        seeds = reference_seeds(cfg, ci, ai, len(manifest['anchors']))
        initial = np.repeat(states[a['initial_id']:a['initial_id']+1], n, axis=0)
        paths, numerics = generate(initial, seeds, **cfg['cases'][ci], duration=cfg['duration'],
            save_dt=cfg['save_dt'], anisotropy=cfg['anisotropy'], damping=cfg['damping'], theta=a['theta'])
        p = RAW/f'case{ci}_anchor{ai}.npz'
        np.savez_compressed(p, spins=paths, seeds=seeds, reduced_time=np.arange(paths.shape[1])*cfg['save_dt'])
        report['rows'].append(dict(case=ci, anchor=ai, path=str(p), sha256=digest(p), numerics=numerics))
        save(dest, report)
    report['status'] = 'COMPLETE_INDEPENDENT_CANDIDATE_REFERENCE_NOT_CERTIFIED'
    save(dest, report)


if __name__ == '__main__':
    run()
