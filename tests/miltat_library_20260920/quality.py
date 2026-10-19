"""Bounded data-generation decision; never converts diagnostics into publication certification."""
import argparse
import hashlib
import json
import h5py
import numpy as np
import yaml
from tests.bauer_two_stage_20260919.common import dump
from .generate import HERE,ROOT


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def release():
    target=HERE/'release.json'
    if target.exists():raise RuntimeError('Release already exists')
    cfg=yaml.safe_load((HERE/'config.yaml').read_text())
    if cfg['dt_fs']!=.3125:raise RuntimeError('This review applies only to dt=0.3125fs')
    evidence=[]
    for branch in range(4):
        folder=HERE.parent/'miltat_refinement_20260920/data'/f'branch_{branch}'
        metrics=json.loads((folder/'metrics.json').read_text())
        with h5py.File(folder/'paths.h5') as h:
            assert h.attrs['complete'] and h.attrs['frames_written']==len(h['time_ps'])
            assert h.attrs['dt_fs']==cfg['dt_fs'] and h['time_ps'][-1]==10.
        assert metrics['replay_max_error_first_saved_frame']<2e-6
        assert all(r['stored_norm_error']<1e-10 and r['ambiguous_faces']==0 for r in metrics['rows'])
        evidence.append(dict(branch=branch,source=str(folder.relative_to(ROOT)),
            max_energy_difference_vs0625_J=float(max(abs(r['energy_difference_vs0625']) for r in metrics['rows'])),
            final_energy_difference_vs0625_J=metrics['rows'][-1]['energy_difference_vs0625'],
            final_spin_rmse_vs0625=metrics['rows'][-1]['spin_rmse_vs0625'],
            metrics_sha256=digest(folder/'metrics.json')))
    boundary=[]
    for index in range(5):
        p=HERE.parent/'miltat_source_audit_20260920/relaxation'/f'size_{index}'/'metrics.json'
        value=json.loads(p.read_text())
        boundary.append(dict(shape=value['shape'],final=value['rows'][-1],source=str(p.relative_to(ROOT)),sha256=digest(p)))
    dump(target,dict(approved_for_generation=True,scientific_certification=False,
        config_sha256=digest(HERE/'config.yaml'),layout_sha256=digest(HERE/'layout.json'),
        source_sha256=digest(ROOT/cfg['source_config']),step_evidence=evidence,boundary_evidence=boundary,
        decision='Explicit engineering release for the frozen first short-path small-size library; NOT a retrospective equivalence test.',
        required_next_gate='New deterministic cold preparation must preserve an isolated positive core for all selected sizes.',
        limitations=['Single temperature and preparation protocol; not an equilibrium or comprehensive state library.',
            'Finite-size changes in core size are retained, not tuned away.',
            'Gamma is a documented conventional value; no material-specific gamma measurement is asserted.',
            'DMI convention explicit; no independently verified 2018 in-plane chirality claim.',
            'Four coupled branches from one parent provide only bounded short-time numerical evidence.',
            'Keep all stochastic outcomes, including core loss; final data review still required.',
            'No long-time diffusion coefficient or model size-generalization certification.']))


def cold_status(reports):
    return len(reports)==3 and all(r['core']['status']=='ISOLATED_POSITIVE_CORE' and np.isfinite(r['torque_max'])
               and r['norm_error']<1e-10 and not r['smoke'] for r in reports)


def cold_gate():
    cfg=yaml.safe_load((HERE/'config.yaml').read_text());out=HERE/'data';reports=[];hashes={}
    for nx,ny in cfg['cell_shapes']:
        label=f'{nx}x{ny}'
        report=json.loads((out/f'cold_{label}.json').read_text())
        if list(report['shape'])!=[nx,ny]:raise RuntimeError('Cold geometry mismatch')
        reports.append(report);hashes[label]=digest(out/f'cold_{label}.pt')
    passed=cold_status(reports)
    dump(out/'cold_gate.json',dict(approved=passed,reports=reports,checkpoint_sha256=hashes,
         note='Initial protocol sanity only, not an equilibrium test or boundary-independence claim.'))
    if not passed:raise RuntimeError('At least one prescribed cold preparation lost its isolated core; stop, do not select another seed')


def status():
    out=HERE/'data';plan=json.loads((HERE/'layout.json').read_text());complete=[];pending=[]
    for row in plan['parents']:
        p=out/row['file']
        if p.exists():
            with h5py.File(p) as h:
                (complete if h.attrs.get('complete',False) else pending).append(row['parent_id'])
        else:pending.append(row['parent_id'])
    dump(HERE/'results'/'generation_status.json',dict(completed_parents=complete,pending_or_failed_parents=pending,
         all_parent_files_complete=not pending,scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['release','cold_gate','status']);a=p.parse_args()
    {'release':release,'cold_gate':cold_gate,'status':status}[a.phase]()
