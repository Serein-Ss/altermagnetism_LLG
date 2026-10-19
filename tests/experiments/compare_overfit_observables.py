"""Descriptive training-subset diagnostics; never a held-out physics gate."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import save, digest
from tests.experiments.path_observables import path_observables


def collect_reports(dilation=False, global_context=False, weighted=False):
    if sum((dilation,global_context,weighted))>1:raise ValueError('choose one comparison')
    experiments = ('cartesian_overfit', 'temporal_dilation_overfit') if dilation else ('bauer_overfit', 'cartesian_overfit')
    if global_context:experiments=('cartesian_overfit','global_context_overfit')
    if weighted:experiments=('cartesian_overfit','weighted_flow_overfit')
    collected=[]
    for experiment in experiments:
        pattern='mixed_eight/*/summary.json' if dilation or global_context or weighted else '*/*/summary.json'
        reports=sorted(Path(f'tests/results/{experiment}').glob(pattern))
        if len(reports)!=(3 if dilation or global_context or weighted else 9):
            raise ValueError('missing expected runs')
        for path in reports:
            report=json.loads(path.read_text())
            if report['status']!='COMPLETE_EXPLORATORY_NOT_CERTIFIED':
                raise ValueError('training/evaluation not complete')
            expected={(step,theta) for step in report['config']['budgets'] for theta in report['config']['evaluation_theta']}
            actual=[(row['step'],row['theta']) for row in report['evaluation']]
            if len(actual)!=len(expected) or set(actual)!=expected:
                raise ValueError('missing or duplicate evaluations')
            collected.append((experiment,path,report))
    return collected


def main(dilation=False, global_context=False, weighted=False):
    destination=Path('tests/results/temporal_dilation_observables' if dilation else 'tests/results/overfit_path_observables')
    if global_context:destination=Path('tests/results/global_context_observables')
    if weighted:destination=Path('tests/results/weighted_flow_observables')
    if destination.exists():
        raise FileExistsError(destination)
    reports=collect_reports(dilation,global_context,weighted)
    rows=[]
    for experiment, report_path, report in reports:
            cfg=report['config']; arm=report_path.parent.parent.name; seed=report['seed']
            with np.load(cfg['data']) as d:
                reference=d['train'][cfg['train_indices']]
            # A deterministic one-path target has zero ensemble variance by definition.
            # Duplicating it here defines that target moment, not extra evidence.
            ref=path_observables(np.repeat(reference,2,axis=0) if len(reference)==1 else reference)
            for evaluation in report['evaluation']:
                step=evaluation['step']; theta=evaluation['theta']
                source=Path(cfg['raw'])/str(seed)/f'step_{step}_T{theta}.npy'
                generated=np.load(source)
                if not np.all(generated[:,0]==reference[0,0]):
                    raise ValueError('reference/generated initial mismatch')
                obs=path_observables(generated)
                target=destination/experiment/arm/str(seed)
                target.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target/f'step_{step}.npz',
                    **{f'reference_{k}':v for k,v in ref.items()},
                    **{f'generated_{k}':v for k,v in obs.items()},
                    reduced_time=np.arange(reference.shape[1])*report['base_config']['save_dt'])
                rows.append(dict(model=experiment,arm=arm,seed=seed,step=step,
                    unique_reference_paths=len(reference),generated_paths=len(generated),
                    generated_sha256=digest(source),reference_data_sha256=report['data_sha256'],
                    curve_rmse={k:float(np.sqrt(np.square(obs[k]-ref[k]).mean()))
                                for k in ref if k!='magnetization_per_path'}))
    save(destination/'summary.json',dict(
        status='COMPLETE_DESCRIPTIVE_NOT_CERTIFIED',
        scope='Selected training examples, no independent test. Curves retain start time; no stationarity assumption. Frames/sites are not independent samples. Single-reference duplication only defines deterministic target moments. No significance or equivalence decision.',
        code_sha256={p:digest(p) for p in [__file__,'tests/experiments/path_observables.py']},
        comparisons=rows))
    print(f'Saved {len(rows)} paired profile comparisons to {destination}')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--dilation',action='store_true')
    group.add_argument('--global-context',action='store_true')
    group.add_argument('--weighted',action='store_true')
    args=parser.parse_args();main(args.dilation,args.global_context,args.weighted)
