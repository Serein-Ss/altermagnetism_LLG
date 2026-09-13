"""Independent-chain, rank/folded Rhat and ESS static comparison gate.

This small-lattice gate only releases the wide-temperature equilibrium pilot;
it never releases paths, an initial pool, a Tc claim or a production dataset.
"""
import argparse
import json
from pathlib import Path
import sys
import arviz as az
import h5py
import numpy as np
from scipy.stats import t as student
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.core.literature_config import sha256


def observations(energy,neel,sites):
    return dict(energy_per_spin=energy/sites,neel_norm=np.linalg.norm(neel,axis=-1),
                neel_z_squared=neel[...,2]**2,signed_neel_z=neel[...,2])


def diagnostics(x,spacing):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[0]<4 or x.shape[1]<20 or not np.isfinite(x).all():
        raise ValueError('four finite chains and sufficient saved times required')
    raw_ess=float(az.ess(x,method='mean'))
    return dict(rank_folded_rhat=float(az.rhat(x,method='rank')),
        bulk_ess=float(az.ess(x,method='bulk')),tail_ess=float(az.ess(x,method='tail')),
        tau_int_reduced=(x.size/raw_ess)*spacing,
        tau_convention='1+2sum_rho; multichain ESS based; MC spacing is sweeps not physical time',
        chain_means=x.mean(1).tolist(),mean=float(x.mean()),sd=float(x.std(ddof=1)))


def compare(campaign):
    campaign=Path(campaign);m=json.loads((campaign/'manifest.json').read_text())
    acceptance=Path(m['acceptance']);a=yaml.safe_load(acceptance.read_text())['equilibrium']
    jobs=[j for j in m['tasks'] if j['kind']=='kernel_equilibrium']
    comparisons=len(jobs)*len(a['primary_observables'])
    # Family includes method-vs-MC and late-block drift for all methods/MC.
    critical_alpha=.05/(comparisons+3*(len(jobs)+2))
    all_metrics=[];report={};evidence={str(acceptance):sha256(acceptance)}
    def add(name,value,lo,hi):
        all_metrics.append(dict(name=name,value=float(value),lower=float(lo),upper=float(hi)))
    for job in jobs:
        path=Path(job['raw']);c=yaml.safe_load(Path(job['config']).read_text());theta=c['theta']
        mc=next(j for j in m['tasks'] if j['kind']=='static_mc' and j['theta']==theta)
        refpath=Path(mc['raw'])
        evidence[str(path)]=sha256(path);evidence[str(refpath)]=sha256(refpath)
        evidence[job['config']]=sha256(job['config']);evidence[mc['config']]=sha256(mc['config'])
        with h5py.File(path) as h:
            if not h.attrs['complete']: raise ValueError('incomplete LLG')
            add(f'{job["index"]}/maximum_norm_error',float(np.max(h['max_norm_error'][:])),0,1e-10)
            for frame in h['spins']:
                if not np.isfinite(frame).all(): raise ValueError('nonfinite stored LLG spins')
            obs=observations(h['energy'][:].T,h['neel'][:].transpose(1,0,2),2*c['shape'][0]*c['shape'][1])
            spacing=float(h['time'][1]-h['time'][0])
        with np.load(refpath,allow_pickle=False) as ref:
            robs=observations(ref['energy'],ref['neel'],2*c['shape'][0]*c['shape'][1])
            rspacing=float(ref['sweep'][1]-ref['sweep'][0])
        details={}
        for key,x in obs.items():
            x=x[:,int(x.shape[1]*a['burn_fraction']):]
            y=robs[key][:,int(robs[key].shape[1]*a['burn_fraction']):]
            dx,dy=diagnostics(x,spacing),diagnostics(y,rspacing)
            entry=dict(llg=dx,mc=dy,interpretation='signed mixing reported separately; not a basin-even gate')
            if key in a['primary_observables']:
                for label,d,data in [('llg',dx,x),('mc',dy,y)]:
                    add(f'{job["index"]}/{key}/{label}/rhat',d['rank_folded_rhat'],0,a['rhat_max'])
                    add(f'{job["index"]}/{key}/{label}/bulk_ess',d['bulk_ess'],a['bulk_ess_min'],1e100)
                    add(f'{job["index"]}/{key}/{label}/tail_ess',d['tail_ess'],a['tail_ess_min'],1e100)
                    half=data.shape[1]//2
                    drift=data[:,-half:].mean(1)-data[:,:half].mean(1)
                    upper=abs(drift.mean())+student.ppf(1-critical_alpha/2,3)*drift.std(ddof=1)/2
                    add(f'{job["index"]}/{key}/{label}/late_block_drift',upper,0,a['block_drift_absolute_bands'][key])
                mx,my=x.mean(1),y.mean(1);vx,vy=mx.var(ddof=1)/4,my.var(ddof=1)/4
                se=np.sqrt(vx+vy);df=(vx+vy)**2/(vx**2/3+vy**2/3) if se else 6
                halfwidth=student.ppf(1-critical_alpha/2,df)*se;difference=mx.mean()-my.mean()
                entry.update(difference=float(difference),simultaneous_ci=[float(difference-halfwidth),float(difference+halfwidth)],
                    independent_units='4 LLG chains versus 4 MC chains; not saved frames')
                add(f'{job["index"]}/{key}/equivalence',abs(difference)+halfwidth,0,a['absolute_equivalence_bands'][key])
            details[key]=entry
        report[str(job['index'])]=details
    valid=all(np.isfinite(row['value']) and row['lower']<=row['value']<=row['upper'] for row in all_metrics)
    result=dict(scope='small_periodic_gomonay_static_kernel',bindings=m['bindings'],
        evidence_sha256=evidence,metrics=all_metrics,status='pass' if valid else 'inconclusive',
        production_enabled=False,comparisons=report,acceptance_sha256=sha256(acceptance),
        precision_scope='preliminary absolute bands; not full P4 spatial/event/numerical certification',
        versions=dict(arviz=az.__version__,numpy=np.__version__))
    def finite(obj):
        if isinstance(obj,dict): return {k:finite(v) for k,v in obj.items()}
        if isinstance(obj,list): return [finite(v) for v in obj]
        if isinstance(obj,float) and not np.isfinite(obj): return None
        return obj
    destination=campaign/'equilibrium_gate.json'
    with destination.open('x') as f: json.dump(finite(result),f,indent=2,allow_nan=False)
    return valid


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);a=p.parse_args()
    if not compare(a.campaign): raise SystemExit(3)
