"""Read-only raw analysis; equivalence bands, paired dt tests and Holm CDF tests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT.parent))
from altermagnetism_LLG.scripts.core.literature_config import load_runtime, sha256
from altermagnetism_LLG.scripts.literature.nishino_miyashita_2015.model import FreeMoments
from altermagnetism_LLG.scripts.literature.nishino_miyashita_2015.run import PAPER,task_coordinates


def interval(x):
    x=np.asarray(x,dtype=float)
    if x.size<2 or not np.isfinite(x).all(): raise ValueError('insufficient/nonfinite independent samples')
    return float(x.mean()),float(stats.t.ppf(.975,x.size-1)*x.std(ddof=1)/np.sqrt(x.size))


def equivalent(x,target,tolerance):
    mean,half=interval(x)
    return {'mean':mean,'ci95_halfwidth':half,'target':float(target),
            'absolute_difference':abs(mean-target),'tolerance_abs':tolerance,
            'passed':bool(abs(mean-target)+half<=tolerance),
            'n_independent_moments':len(x),'sd':float(np.std(x,ddof=1)),
            'skewness':float(stats.skew(x)) if np.std(x)>0 else 0.,
            'inference':'Student-t CI of independent moment time-averages; bounded finite-variance CLT; no frame pseudoreplication'}


def holm(pvalues):
    p=np.asarray(pvalues);order=np.argsort(p);out=np.empty(len(p));previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,float(p[index])*(len(p)-rank)));out[index]=previous
    return out.tolist()


def collect(raw,config,phase,config_hash):
    ext=config.numerics['validation_extension'];grid=config.numerics['numerical_choice']['theta_grid']
    repeats=ext['convergence_repetitions'] if phase=='convergence' else ext['repetitions']
    groups={};hashes={};seen_seeds=set()
    for task in range(2*len(grid)*repeats):
        path=raw/f'task_{task:04d}.h5'
        if not path.exists(): raise ValueError(f'missing task {task}')
        with h5py.File(path,'r') as h5:
            meta=json.loads(h5.attrs['metadata'])
            expected=task_coordinates(config,phase,task)
            if (meta['case'],meta['theta_index'],meta['repeat'])!=expected or not h5.attrs['complete'] or not meta['formal_settings']:
                raise ValueError('wrong/incomplete/nonformal task '+str(path))
            if meta.get('phase')!=phase or meta.get('run_id')!=raw.name.removesuffix('.partial'):
                raise ValueError('phase or run identity mismatch')
            if meta.get('code_sha256')!=config.numerics.get('code_sha256'):
                raise ValueError('code identity mismatch')
            if meta['config_sha256']!=config_hash or meta['seed'] in seen_seeds:
                raise ValueError('configuration hash mismatch or reused independent seed')
            seen_seeds.add(meta['seed'])
            methods=ext['methods'] if phase=='convergence' else ['paper_midpoint']
            factors=ext['factors'] if phase=='convergence' else [config.numerics.get('selected_factor',1)]
            for method in methods:
                for factor in factors:
                    g=h5[f'{method}/factor_{factor}'];z=g['z'][:]
                    measurement=z[h5['tau'][:]>meta['burn_steps']*meta['dt']]
                    half=len(measurement)//2
                    entry={'mean':g['per_moment_mean'][:], 'final':g['final_spins'][:,2],
                        'stationarity':measurement[:half].mean(0)-measurement[-half:].mean(0),
                        'trace_mean':z.mean(1),'trace_std':z.std(1),'tau':h5['tau'][:],
                        'example_tau':h5['tau'][::20].tolist(),'example_z':z[::20,:16].tolist(),
                        'norm':float(g.attrs['norm_error']),'raw_norm':g['raw_norm_errors'][:].tolist()}
                    groups.setdefault((meta['case'],meta['theta_index'],method,factor),[]).append(entry)
        hashes[path.name]=sha256(path)
    return groups,hashes


def analyze(raw,config,phase,config_hash):
    groups,hashes=collect(raw,config,phase,config_hash)
    model=FreeMoments(config.reduced);ext=config.numerics['validation_extension'];grid=config.numerics['numerical_choice']['theta_grid']
    rows=[]
    for (case,index,method,factor),entries in sorted(groups.items()):
        theta=grid[index];averages=np.concatenate([e['mean'] for e in entries]);final=np.concatenate([e['final'] for e in entries])
        row={'case':case,'theta_index':index,'theta':theta,'method':method,'factor':factor,
            'mean_check':equivalent(averages,model.exact_mean(theta),ext['mean_tolerance_abs']),
            'stationarity_check':equivalent(np.concatenate([e['stationarity'] for e in entries]),0.,ext['mean_tolerance_abs']),
            'norm_error':max(e['norm'] for e in entries),
            'pre_normalization_max_errors':np.max([e['raw_norm'] for e in entries],axis=0).tolist(),
            'tau':entries[0]['tau'].tolist(),'mean_trace':np.mean([e['trace_mean'] for e in entries],axis=0).tolist(),
            'mean_within_replica_std_z':np.mean([e['trace_std'] for e in entries],axis=0).tolist(),
            'example_tau':entries[0]['example_tau'],'example_z':entries[0]['example_z']}
        x=model.field_h/theta
        def cdf(z): return np.expm1(x*(np.asarray(z)+1))/np.expm1(2*x)
        ks=stats.kstest(final,cdf)
        row['cdf_check']={'D':float(ks.statistic),'p_raw':float(ks.pvalue),
            'sampling':'one terminal observation per independent moment; no temporal pooling',
            'effect_tolerance':ext['cdf_effect_tolerance']}
        row['endpoint_histogram']={'counts':np.histogram(final,bins=40,range=(-1,1))[0].tolist(),
                                   'edges':np.linspace(-1,1,41).tolist()}
        rows.append(row)
    # One family per numerical method/factor; main case A/B temperatures tested together.
    for method,factor in sorted({(r['method'],r['factor']) for r in rows}):
        family=[r for r in rows if (r['method'],r['factor'])==(method,factor)]
        adjusted=holm([r['cdf_check']['p_raw'] for r in family])
        for row,p in zip(family,adjusted):
            row['cdf_check']['p_holm']=p
            row['cdf_check']['passed']=bool(p>=.05 and row['cdf_check']['D']<=ext['cdf_effect_tolerance'])
            row['passed']=bool(row['mean_check']['passed'] and row['stationarity_check']['passed'] and row['cdf_check']['passed'] and row['norm_error']<=ext['norm_tolerance'])
    comparisons=[];selected=None
    if phase=='convergence':
        for case in ('A','B'):
            for index in range(len(grid)):
                for factor in (1,2):
                    base=np.concatenate([e['mean'] for e in groups[case,index,'paper_midpoint',factor]])
                    fine=np.concatenate([e['mean'] for e in groups[case,index,'paper_midpoint',4]])
                    geometric=np.concatenate([e['mean'] for e in groups[case,index,'geometric_midpoint',4]])
                    for reference,values in [('fine_paper',fine),('fine_geometric',geometric)]:
                        comparisons.append({'case':case,'theta_index':index,'factor':factor,'reference':reference,
                            **equivalent(base-values,0.,ext['convergence_tolerance_abs'])})
        for factor in (1,2):
            required=[r for r in rows if (r['method']=='paper_midpoint' and r['factor'] in (factor,4)) or (r['method']=='geometric_midpoint' and r['factor']==4)]
            if all(r['passed'] for r in required) and all(r['passed'] for r in comparisons if r['factor']==factor):
                selected=factor;break
        passed=selected is not None
    else: passed=all(r['passed'] for r in rows)
    return {'status':'pass' if passed else 'fail','phase':phase,'selected_factor':selected,
        'scope':'Nishino Fig.1 only; no interacting-system or model-training production authorization',
        'rows':rows,'paired_convergence':comparisons,'raw_sha256':hashes,
        'production_enabled':False,'config_sha256':config_hash,
        'reduced':config.reduced,'numerics':config.numerics,
        'remaining':'R2-R5 not run; all-system certification remains unavailable'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--project-root',type=Path,default=ROOT)
    p.add_argument('--run-id',required=True);p.add_argument('--phase',choices=['primary','convergence'],required=True)
    args=p.parse_args();root=args.project_root;config=load_runtime(args.config)
    raw=root/'data/literature_reproduction'/PAPER/'raw'/args.run_id
    report=analyze(raw,config,args.phase,sha256(args.config));report['run_id']=args.run_id
    report['code_commit']=config.numerics.get('code_commit');report['code_sha256']=config.numerics.get('code_sha256')
    derived=root/'data/literature_reproduction'/PAPER/'derived'/args.run_id
    derived.mkdir(parents=True,exist_ok=True)
    with (derived/'summary.json').open('x') as stream: json.dump(report,stream,indent=2)
    output=root/'output/literature_reproduction'/PAPER/args.run_id
    output.mkdir(parents=True,exist_ok=True)
    with (output/'report.json').open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps({'status':report['status'],'report':str(output/'report.json')}),flush=True)
    if report['status']!='pass': raise SystemExit(2)


if __name__=='__main__': main()
