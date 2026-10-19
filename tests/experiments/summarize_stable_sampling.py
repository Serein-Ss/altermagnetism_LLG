"""Separate paired-path sensitivity from conditional empirical moment changes."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import save,digest


def validate_report(report,steps):
    expected={(theta,step) for theta in (.11,.13) for step in steps}
    rows=report['rows']
    if report['status']!='COMPLETE_SAMPLING_DIAGNOSTIC' or len(rows)!=len(expected) or {(r['theta'],r['steps']) for r in rows}!=expected:
        raise ValueError('incomplete or duplicate sampling rows')
    if any(len(r['families'])!=2 or {f['family'] for f in r['families']}!={0,1} for r in rows):
        raise ValueError('incomplete initial families')
    if any(r.get('previous_evaluation_max_abs_difference',float('inf'))>1e-6 for r in rows if r['steps']==128):
        raise ValueError('128-step replay not verified')


def main(extended=False):
    name='stable_sampling_extended' if extended else 'stable_sampling_audit'
    root=Path('tests/results')/name
    steps=(128,256,512,1024) if extended else (32,64,128,256)
    output=root/'comparison.json'
    if output.exists():raise FileExistsError(output)
    reports=[]
    for seed in (11,22,33):
        path=root/str(seed)/'summary.json';report=json.loads(path.read_text())
        validate_report(report,steps);reports.append((seed,path,report))
    rows=[]
    for seed,path,report in reports:
        for theta in (.11,.13):
            for family in (0,1):
                previous=None;previous_step=None
                for step in steps:
                    profile=root/str(seed)/f'T{theta}_steps_{step}_family_{family}.npz'
                    with np.load(profile) as data:
                        current={k:data[k] for k in data.files}
                    if previous is not None:
                        keys=[k for k in current if k.startswith('generated_') and k!='generated_magnetization_per_path']
                        if any(not np.array_equal(current[k],previous[k]) for k in current if k.startswith('reference_')):
                            raise ValueError('reference changed between integration settings')
                        rows.append(dict(seed=seed,theta=theta,family=family,steps=[previous_step,step],
                            moment_change_rmse={k.removeprefix('generated_'):float(np.sqrt(np.square(current[k]-previous[k]).mean())) for k in keys},
                            reference_error_rmse={k.removeprefix('generated_'):float(np.sqrt(np.square(current[k]-current[k.replace('generated_','reference_',1)]).mean())) for k in keys},
                            profile_sha256=digest(profile)))
                    previous=current;previous_step=step
    save(output,dict(status='COMPLETE_DESCRIPTIVE_SAMPLING_COMPARISON',
        scope='Per-initial empirical moments,16 trajectories each; no uncertainty intervals, equivalence tolerances or physical certification. Paired-path sensitivity is distinct from distributional error. Reused development, not held-out testing.',
        input_sha256={str(path):digest(path) for _,path,_ in reports},
        paired_path_errors={str(seed):report['pairs'] for seed,_,report in reports},rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--extended',action='store_true')
    main(p.parse_args().extended)
