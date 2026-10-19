"""Reject incomplete/mismatched ensembles before any distributional analysis."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import save,digest


def validate_generated_report(report,cfg,seed):
    expected={(i,t) for i in range(4) for t in cfg['theta']}
    rows=report['rows']
    if report['status']!='COMPLETE_GENERATED_ENSEMBLE_NOT_CERTIFIED' or report['training_seed']!=seed:
        raise ValueError('generated run incomplete or wrong seed')
    if len(rows)!=len(expected) or {(r['family'],r['theta']) for r in rows}!=expected:
        raise ValueError('missing or duplicated generated families')
    if any(r['paths']!=cfg['n_per_half'] or r['sampling_steps']!=cfg['generated_sampling_steps'] for r in rows):
        raise ValueError('generated sample count or integration mismatch')


def validate_spins(spins,initial,n,frames):
    a=np.asarray(spins)
    if a.shape!=(n,frames,*initial.shape) or not np.isfinite(a).all():raise ValueError('invalid path shape or values')
    expected=initial.astype(a.dtype)
    if not np.all(a[:,0]==expected):raise ValueError('initial mismatch')
    tolerance=1e-5 if a.dtype==np.float32 else 1e-10
    if np.abs(np.linalg.norm(a,axis=-1)-1).max()>tolerance:raise ValueError('unit spin violation')


def collect():
    root=Path('tests/results/stable_independent_reference')
    manifest_path=root/'manifest.json';manifest=json.loads(manifest_path.read_text());cfg=manifest['config']
    mhash=digest(manifest_path)
    if digest('tests/conf/stable_independent_reference.json')!=manifest['config_sha256'] or digest(cfg['initials'])!=cfg['initial_sha256']:
        raise ValueError('reference contract or initials changed')
    if digest(cfg['model_freeze'])!=manifest['model_freeze_sha256']:raise ValueError('candidate freeze changed')
    freeze=json.loads(Path(cfg['model_freeze']).read_text())
    records=[]
    # All completion/provenance checks precede any loading of future spin arrays.
    for case in manifest['cases']:
        report=json.loads((root/f"case_{case['case_id']}"/'summary.json').read_text())
        if report['status']!='COMPLETE_NUMERICAL_REFERENCE_NOT_CERTIFIED' or report['case']!=case or report['manifest_sha256']!=mhash:
            raise ValueError('reference run incomplete or mismatched')
        if report['r1_paths']!=cfg['n_per_half'] or report['r2_paths']!=cfg['n_per_half']:raise ValueError('reference sample count mismatch')
        records.append(dict(kind='reference',case=case,path=f"tests/data/stable_independent_reference/case_{case['case_id']}.npz",sha256=report['data_sha256']))
    for seed in (11,22,33):
        report=json.loads(Path(f'tests/results/stable_independent_sampling/{seed}/summary.json').read_text())
        validate_generated_report(report,cfg,seed)
        if report['reference_manifest_sha256']!=mhash or report['checkpoint_sha256']!=freeze['checkpoint_sha256'][str(seed)]:
            raise ValueError('model provenance mismatch')
        if report['sampling_config_sha256']!=digest('tests/conf/stable_independent_sampling.json'):raise ValueError('sampling configuration changed')
        for row in report['rows']:
            records.append(dict(kind='generated',seed=seed,family=row['family'],theta=row['theta'],batch_seeds=row['batch_seeds'],
                path=f"tests/data/stable_independent_sampling/{seed}/family_{row['family']}_T{row['theta']}.npz",sha256=row['data_sha256']))
    initials=np.load(cfg['initials']);n=cfg['n_per_half'];frames=round(cfg['duration']/cfg['save_dt'])+1
    noise=[]
    for record in records:
        if digest(record['path'])!=record['sha256']:raise ValueError('data hash mismatch')
        with np.load(record['path']) as d:
            is_ref=record['kind']=='reference';family=record['case']['family'] if is_ref else record['family']
            validate_spins(d['spins'],initials[family],2*n if is_ref else n,frames)
            if not np.array_equal(d['reduced_time'],np.arange(frames)*cfg['save_dt']):raise ValueError('time grid mismatch')
            if is_ref:
                expected=record['case']['noise_start']+np.arange(2*n)
                if not np.array_equal(d['seeds'],expected) or not np.array_equal(d['half'],np.repeat([0,1],n)):
                    raise ValueError('reference seeds or R1/R2 split mismatch')
                noise.extend(d['seeds'].tolist())
            elif not np.array_equal(d['batch_seeds'],record['batch_seeds']):raise ValueError('latent provenance mismatch')
    if len(set(noise))!=len(noise):raise ValueError('reference noise reused')
    return dict(status='COMPLETE_INPUT_INTEGRITY_ONLY',manifest_sha256=mhash,records=records,
        scope='Shapes, initial anchors, norm, time grids, sample counts and recorded provenance verified. No distributional or continuum certification.')


if __name__=='__main__':
    destination=Path('tests/results/independent_input_audit/summary.json')
    if destination.exists():raise FileExistsError(destination)
    result=collect();save(destination,result)
