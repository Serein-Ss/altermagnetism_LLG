"""Exploratory paired-arm whole-path uncertainty, conditional on fixed initials."""
import itertools
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.flow_film_statistics import moment_errors, validate_rows
from tests.experiments.trajectory_bootstrap import bootstrap_weights, weighted_observables


def batched_errors(g, r):
    """Apply the already validated prefix convention independently to each draw."""
    n=len(g['magnetization_mean'])
    if set(g)!=set(r) or any(len(v)!=n for v in (*g.values(),*r.values())):
        raise ValueError('matching bootstrap draws required')
    rows=[moment_errors({k:v[i] for k,v in g.items()},
                        {k:v[i] for k,v in r.items()}) for i in range(n)]
    return {k:np.array([row[k] for row in rows]) for k in rows[0]}


def contrasts(draws):
    """Subtract errors at matched draw indices, never independently reorder arms."""
    out={}
    for a,b in (('control','frozen'),('flow_film','frozen'),('flow_film','control')):
        if set(draws[a])!=set(draws[b]): raise ValueError('different metric sets')
        for k in draws[a]:
            if draws[a][k].shape!=draws[b][k].shape: raise ValueError('unmatched draws')
            out[f'{a}_minus_{b}/{k}']=draws[a][k]-draws[b][k]
    return out


def main():
    cfg_path=Path('tests/conf/flow_film_uncertainty.json')
    cfg=json.loads(cfg_path.read_text())
    source=Path('tests/results/flow_film_statistics/summary.json')
    stats=json.loads(source.read_text())
    expected=set(itertools.product(('frozen','control','flow_film'),(11,22,33),(0,1),(.11,.13)))
    keys=[(r['arm'],r['seed'],r['family'],r['theta']) for r in stats['rows']]
    if stats['status']!='COMPLETE_DEVELOPMENT_STATISTICS_NOT_CERTIFIED' or len(keys)!=36 or set(keys)!=expected:
        raise ValueError('complete statistics required')
    reports={}
    for p,h in stats['source_reports'].items():
        if digest(p)!=h: raise ValueError('generation report changed')
        report=json.loads(Path(p).read_text()); validate_rows(report,report['arm'],report['seed'])
        reports[report['arm'],report['seed']]=report
    if set(reports)!=set(itertools.product(('frozen','control','flow_film'),(11,22,33))):
        raise ValueError('missing model report')
    train=json.loads(Path('tests/conf/flow_film_controlled.json').read_text())
    if digest(train['data'])!=train['data_sha256']: raise ValueError('reference changed')
    with np.load(train['data']) as d: dev=d['development']
    out=Path('tests/results/flow_film_uncertainty'); out.mkdir(exist_ok=False)
    result=dict(status='RUNNING',config=cfg,config_sha256=digest(cfg_path),
                source_sha256=digest(source),code_sha256=digest(__file__),rows=[])
    save(out/'summary.json',result)
    for family,theta in itertools.product((0,1),(.11,.13)):
        paths={}; indices=None; batch_seeds=None
        for key,report in reports.items():
            row=next(r for r in report['rows'] if (r['family'],r['theta'])==(family,theta))
            if digest(row['path'])!=row['sha256']: raise ValueError('raw paths changed')
            with np.load(row['path']) as d:
                g=d['spins']; ix=d['reference_indices']; bs=d['batch_seeds']
                if g.shape!=(128,201,25,3) or ix.shape!=(16,) or len(np.unique(ix))!=16:
                    raise ValueError('incorrect sample counts')
                if not np.array_equal(d['reduced_time'],np.arange(201)*80): raise ValueError('wrong time grid')
                if not np.array_equal(ix,row['reference_indices']): raise ValueError('reference mapping changed')
                if indices is not None and (not np.array_equal(ix,indices) or not np.array_equal(bs,batch_seeds)):
                    raise ValueError('unpaired ensembles')
                indices=ix.copy(); batch_seeds=bs.copy(); paths[key]=g
        ref=dev[indices]
        for g in paths.values():
            if not np.array_equal(g[0,0].astype('f4'),ref[0,0].astype('f4')): raise ValueError('wrong initial')
        # Family rank matches the sampler's sharing across theta; no pooled-theta inference.
        seed=cfg['bootstrap_seed']+family*100
        wg=bootstrap_weights(128,cfg['replicates'],seed)
        wr=bootstrap_weights(16,cfg['replicates'],seed+1)
        collected={key:[] for key in paths}
        for start in range(0,cfg['replicates'],cfg['chunk']):
            sl=slice(start,start+cfg['chunk']); ro=weighted_observables(ref,wr[sl])
            for key,g in paths.items():
                collected[key].append(batched_errors(weighted_observables(g,wg[sl]),ro))
        combined={key:{k:np.concatenate([part[k] for part in parts]) for k in parts[0]}
                  for key,parts in collected.items()}
        for trained_seed in (11,22,33):
            arms={arm:combined[arm,trained_seed] for arm in ('frozen','control','flow_film')}
            draws={f'{arm}/{k}':v for arm,values in arms.items() for k,v in values.items()}
            draws.update(contrasts(arms))
            if not all(np.isfinite(v).all() for v in draws.values()): raise ValueError('nonfinite bootstrap')
            dest=out/f'seed{trained_seed}_family{family}_T{theta}.npz'
            np.savez_compressed(dest,**draws)
            result['rows'].append(dict(seed=trained_seed,family=family,theta=theta,
                path=str(dest),sha256=digest(dest),n_reference=16,n_generated=128,
                percentile95={k:np.quantile(v,[.025,.975]).tolist() for k,v in draws.items()}))
            save(out/'summary.json',result)
    result['status']='COMPLETE_EXPLORATORY_UNCERTAINTY_NOT_CERTIFIED'
    save(out/'summary.json',result)


if __name__=='__main__': main()
