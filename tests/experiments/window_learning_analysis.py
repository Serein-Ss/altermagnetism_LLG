"""Matched-initial development profiles and chunk-boundary diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import save,digest
from tests.experiments.path_observables import path_observables


def boundary_increments(paths,stride=10):
    """Increment after each internal chunk endpoint versus other increments."""
    increment=np.square(np.diff(paths.astype('float64'),axis=1)).sum(-1).mean(axis=(0,2))
    index=np.arange(len(increment));boundary=(index>0)&(index%stride==0)
    if not boundary.any():raise ValueError('no internal chunk boundary')
    return dict(after_boundary=float(increment[boundary].mean()),
                elsewhere=float(increment[~boundary].mean()))


def prefix_errors(prediction,reference,dt):
    return {str(end):dict(reduced_time=end*dt,
        mean_magnetization_rmse=float(np.sqrt(np.square(prediction[:end+1]-reference[:end+1]).mean())),
        endpoint_mz_bias=float(prediction[end,2]-reference[end,2])) for end in (10,50,200)}


def main(conditional=False,stable=False):
    if conditional and stable:raise ValueError('choose one comparison')
    destination=Path('tests/results/conditional_source_observables' if conditional else 'tests/results/window_learning_observables')
    if stable:destination=Path('tests/results/stable_window_observables')
    if destination.exists():raise FileExistsError(destination)
    reports=[]
    comparisons=[('full','window_learning','full'),('window11','window_learning','window11')]
    if conditional:comparisons=[('uniform','window_learning','window11'),('conditional','conditional_source_learning','window11')]
    if stable:comparisons=[('baseline','window_learning','window11'),('stable','stable_window_learning','window11')]
    for arm,experiment,folder in comparisons:
        for seed in (11,22,33):
            path=Path(f'tests/results/{experiment}/{folder}/{seed}/summary.json')
            report=json.loads(path.read_text())
            expected={(s,t) for s in ((6000,20000,40000) if experiment=='stable_window_learning' else (600,3000,6000)) for t in (.11,.13)}
            if report['status']!='COMPLETE_EXPLORATORY_NOT_CERTIFIED' or len(report['evaluation'])!=6 or {(r['step'],r['theta']) for r in report['evaluation']}!=expected:
                raise ValueError('incomplete experiment')
            reports.append((arm,seed,report))
    rows=[]
    for arm,seed,report in reports:
        cfg=report['config'];base=report['base_config']
        with np.load(cfg['data']) as d:
            development=d['development'];conditions=d['development_condition'];training_initial=d['train'][:,0]
        for evaluation in report['evaluation']:
            step= evaluation['step'];theta=evaluation['theta']
            pool=development[np.isclose(conditions[:,0],theta)]
            path=Path(cfg['raw'])/str(seed)/f'step_{step}_T{theta}.npy'
            generated=np.load(path)
            if generated.shape[1:]!=pool.shape[1:] or len(generated)!=cfg['initials']*cfg['latents']:
                raise ValueError('trajectory shape mismatch')
            for family,indices in enumerate(evaluation['reference_indices']):
                reference=pool[indices]
                prediction=generated[family*cfg['latents']:(family+1)*cfg['latents']]
                if not np.all(prediction[:,0]==reference[0,0]):raise ValueError('initial mismatch')
                if np.any(np.all(training_initial==reference[0,0],axis=(1,2))):raise ValueError('development initial reused in training')
                ref=path_observables(reference);obs=path_observables(prediction)
                output=destination/arm/str(seed)/f'T{theta}'/str(family)
                output.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(output/f'step_{step}.npz',
                    **{f'reference_{k}':v for k,v in ref.items()},**{f'generated_{k}':v for k,v in obs.items()},
                    reduced_time=np.arange(pool.shape[1])*base['save_dt'])
                rows.append(dict(arm=arm,seed=seed,step=step,theta=theta,family=family,
                    generated_sha256=digest(path),reference_indices=indices,
                    boundary_reference=boundary_increments(reference),boundary_generated=boundary_increments(prediction),
                    prefixes=prefix_errors(obs['magnetization_mean'],ref['magnetization_mean'],base['save_dt']),
                    curve_rmse={k:float(np.sqrt(np.square(obs[k]-ref[k]).mean())) for k in ref if k!='magnetization_per_path'}))
    save(destination/'summary.json',dict(status='COMPLETE_DEVELOPMENT_DIAGNOSTIC',
        scope='Two development initial families per temperature, reused during development, not fresh final certification. Conditional profiles never pool initial families. Boundary index10 means increment from frame10 to11, not a duplicated anchor. Same boundary positions reported for whole-path baseline; no stationarity/equilibrium assumption or significance claim.',
        comparison_scope='Unequal optimization budgets; stable arm uses cosine schedule and EMA, no single-factor attribution.' if stable else 'See per-run training configurations.',
        code_sha256={p:digest(p) for p in [__file__,'tests/experiments/path_observables.py']},rows=rows))
    print(f'Saved {len(rows)} matched-initial comparisons')


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
    group.add_argument('--conditional-source',action='store_true');group.add_argument('--stable',action='store_true')
    args=parser.parse_args();main(args.conditional_source,args.stable)
