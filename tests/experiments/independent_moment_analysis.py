"""Conditional moment diagnostics after the complete input-integrity gate."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import save,digest
from tests.experiments.path_observables import path_observables
from tests.experiments.independent_comparison_statistics import compare_ensembles


def select_records(records,family,theta):
    reference=[r for r in records if r['kind']=='reference' and r['case']['family']==family and r['case']['theta']==theta]
    primary=[r for r in reference if r['case']['role']=='primary']
    controls=[r for r in reference if r['case']['role']=='numerical_control']
    generated=[r for r in records if r['kind']=='generated' and r['family']==family and r['theta']==theta]
    if len(primary)!=1 or len(generated)!=3 or {r['seed'] for r in generated}!={11,22,33}:
        raise ValueError('incomplete conditional ensemble')
    if len(controls)!=(2 if family==0 else 0):raise ValueError('incomplete numerical controls')
    return primary[0],generated,controls


def load_spins(record):
    if digest(record['path'])!=record['sha256']:raise ValueError('data changed since input audit')
    with np.load(record['path']) as d:return d['spins']


def run(index):
    if not 0<=index<8:raise ValueError('unregistered conditional analysis')
    audit_path=Path('tests/results/independent_input_audit/summary.json')
    audit=json.loads(audit_path.read_text())
    manifest_path=Path('tests/results/stable_independent_reference/manifest.json')
    if audit['status']!='COMPLETE_INPUT_INTEGRITY_ONLY' or digest(manifest_path)!=audit['manifest_sha256']:
        raise ValueError('complete matching integrity audit required')
    cfg=json.loads(manifest_path.read_text())['config'];family=index//2;ti=index%2;theta=cfg['theta'][ti]
    primary,generated,controls=select_records(audit['records'],family,theta)
    out=Path(f'tests/results/independent_moment_analysis/family_{family}_T{theta}')
    if out.exists():raise FileExistsError('preserve analysis')
    truth=load_spins(primary);n=cfg['n_per_half'];r1,r2=truth[:n],truth[n:]
    comparisons=[(f"model_{r['seed']}",r,None) for r in generated]
    comparisons.extend((f"control_{r['case']['method']}_half{half}",r,half) for r in controls for half in (0,1))
    out.mkdir(parents=True)
    profiles={f'{name}_{key}':value for name,a in (('r1',r1),('r2',r2)) for key,value in path_observables(a).items()}
    np.savez_compressed(out/'reference_profiles.npz',**profiles,reduced_time=np.arange(201)*cfg['save_dt'])
    report=dict(status='RUNNING',family=family,theta=theta,n_per_ensemble=n,replicates=2000,
        audit_sha256=digest(audit_path),job_id=os.getenv('SLURM_JOB_ID'),
        scope='Conditional moment diagnostics only; percentile intervals are descriptive and not simultaneous or coverage-calibrated. No equivalence, event, across-initial or full-path-law certificate.',
        resampling='Independent R1/R2 weights shared across models; paired generated weights shared across model seeds and temperatures at each initial. Controls have independent weights.',
        code_sha256={p:digest(p) for p in [__file__,'tests/experiments/independent_comparison_statistics.py',
            'tests/experiments/trajectory_bootstrap.py','tests/experiments/path_observables.py']},rows=[])
    save(out/'summary.json',report)
    for name,record,half in comparisons:
        g=load_spins(record)
        if half is not None:g=g[half*n:(half+1)*n]
        reference_seed=860000000+family*100+ti*10
        generated_seed=860010000+family*100 if half is None else 860020000+record['case']['case_id']*10+half
        summary,draws=compare_ensembles(g,r1,r2,reference_seed=reference_seed,generated_seed=generated_seed)
        np.savez_compressed(out/f'{name}_bootstrap.npz',**draws)
        np.savez_compressed(out/f'{name}_profiles.npz',**path_observables(g))
        report['rows'].append(dict(comparison=name,source_sha256=record['sha256'],reference_seed=reference_seed,
            generated_seed=generated_seed,statistics=summary))
        save(out/'summary.json',report)
    report['status']='COMPLETE_CONDITIONAL_MOMENTS_NOT_CERTIFIED';save(out/'summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True)
    run(p.parse_args().index)
