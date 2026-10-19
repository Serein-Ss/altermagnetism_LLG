"""Complete-batch descriptive development moments and sign events."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import save, digest
from tests.experiments.path_observables import path_observables
from tests.experiments.descriptive_path_events import sign_events


def moment_errors(g, r, ends=(10,200)):
    result={}
    if set(g)!=set(r): raise ValueError('different observable sets')
    for key in g:
        if key=='magnetization_per_path': continue  # Independent physical noises, unequal n.
        if g[key].shape!=r[key].shape: raise ValueError('different observable grids')
        lag=int(key.rsplit('lag',1)[1]) if key.startswith(('temporal_','increment_')) else 0
        for end in ends:
            if not 0<=end<len(g['magnetization_mean']): raise ValueError('invalid prefix')
            stop=end+1-lag
            if stop>0:
                result[f'frame{end}/{key}']=float(np.sqrt(np.square(g[key][:stop]-r[key][:stop]).mean()))
    return result


def validate_rows(report, arm, seed):
    if report['status']!='COMPLETE_DEVELOPMENT_GENERATION_NOT_CERTIFIED' or (report['arm'],report['seed'])!=(arm,seed):
        raise ValueError('complete matching generation required')
    keys=[(r['family'],r['theta']) for r in report['rows']]
    if len(keys)!=4 or set(keys)!={(f,t) for f in (0,1) for t in (.11,.13)}:
        raise ValueError('missing/duplicated initial-condition group')


def main():
    cfg_path=Path('tests/conf/local_pair_evaluation.json'); cfg=json.loads(cfg_path.read_text())
    train=json.loads(Path(cfg['training_config']).read_text())
    if digest(train['data'])!=train['data_sha256']: raise ValueError('reference data changed')
    reports=[]
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            p=Path(f'tests/results/local_pair_evaluation/{arm}/{seed}/summary.json')
            r=json.loads(p.read_text()); validate_rows(r,arm,seed)
            if r['config_sha256']!=digest(cfg_path) or r['data_sha256']!=train['data_sha256']:
                raise ValueError('evaluation configuration or data mismatch')
            reports.append((p,r))
    with np.load(train['data']) as d: dev=d['development']; conditions=d['development_condition']
    out=Path('tests/results/local_pair_statistics')
    if out.exists(): raise FileExistsError('preserve prior statistics')
    out.mkdir()
    result=dict(status='RUNNING',rows=[],
        scope='Development descriptive moment-curve RMSE only, generated n128/reference n16. No CIs, hypothesis/equivalence test, seed selection or final certification. Signs are not committed basin switches.',
        source_reports={str(p):digest(p) for p,_ in reports},code_sha256=digest(__file__))
    save(out/'summary.json',result)
    pairing={}
    for _,report in reports:
        for row in report['rows']:
            if digest(row['path'])!=row['sha256']: raise ValueError('generated data changed')
            with np.load(row['path']) as d:
                g=d['spins']; indices=d['reference_indices']; time=d['reduced_time']; seeds=d['batch_seeds']
            if g.shape!=(128,201,25,3) or indices.shape!=(16,) or len(np.unique(indices))!=16:
                raise ValueError('invalid sample counts')
            if not np.array_equal(time,np.arange(201)*80): raise ValueError('time grid changed')
            if not np.array_equal(indices,row['reference_indices']): raise ValueError('reference mapping changed')
            if not np.all(np.isclose(conditions[indices,0],row['theta'])): raise ValueError('wrong reference condition')
            key=(row['family'],row['theta'])
            if key in pairing:
                old_indices,old_seeds=pairing[key]
                if not np.array_equal(indices,old_indices) or not np.array_equal(seeds,old_seeds):
                    raise ValueError('unpaired model comparison')
            else: pairing[key]=(indices.copy(),seeds.copy())
            ref=dev[indices]
            if not np.array_equal(g[0,0].astype('f4'),ref[0,0].astype('f4')): raise ValueError('wrong initial')
            go,ro=path_observables(g),path_observables(ref)
            name=f"{report['arm']}_seed{report['seed']}_family{row['family']}_T{row['theta']}"
            profile=out/f'{name}.npz'
            np.savez_compressed(profile,reduced_time=time,
                **{f'{group}/{k}':v for group,obs in (('generated',go),('reference',ro)) for k,v in obs.items()})
            events={group:sign_events(obs['magnetization_per_path'][...,2],time)
                    for group,obs in (('generated',go),('reference',ro))}
            np.savez_compressed(out/f'{name}_events.npz',
                **{f'{group}/{k}':v for group,e in events.items() for k,v in e.items()})
            result['rows'].append(dict(arm=report['arm'],seed=report['seed'],family=row['family'],theta=row['theta'],
                profile_path=str(profile),profile_sha256=digest(profile),moment_curve_rmse=moment_errors(go,ro),
                events={group:dict(n=len(e['crossed_zero']),negative_endpoint=int(e['negative_endpoint'].sum()),
                                    crossed_zero=int(e['crossed_zero'].sum())) for group,e in events.items()}))
            save(out/'summary.json',result)
    result['status']='COMPLETE_DEVELOPMENT_STATISTICS_NOT_CERTIFIED'; save(out/'summary.json',result)


if __name__=='__main__': main()
