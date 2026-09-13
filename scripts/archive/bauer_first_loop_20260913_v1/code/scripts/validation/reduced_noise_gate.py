"""Fixed-seed >=1e6-component thermal bath gate; CPU fat-node preflight."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent))
from altermagnetism_LLG.scripts.core.literature_config import load_runtime, sha256
from altermagnetism_LLG.scripts.core.reduced_llg import ReducedLLG
from altermagnetism_LLG.scripts.literature.nishino_miyashita_2015.model import FreeMoments,case_alpha


def noise_gate(config):
    count=config.numerics['validation_extension']['noise_components']
    if count<1_000_000: raise ValueError('at least one million components required')
    count=math.ceil(count/3)*3
    state=torch.zeros(1,count//3,3,dtype=torch.float64)
    rows=[]
    for case in ('A','B'):
        for theta in (.5,2.,6.):
            alpha=case_alpha(config.reduced,case,theta);dt=.005
            solver=ReducedLLG(FreeMoments(config.reduced),alpha=alpha,theta=theta)
            dw=solver.noise_increment(state,dt,torch.Generator().manual_seed(20260909))
            field=(math.sqrt(2*alpha*theta)*dw/dt).numpy().reshape(-1,3)
            expected=2*alpha*theta/dt
            mean=field.mean()/math.sqrt(expected);variance=field.var()/expected
            cross=np.corrcoef(field,rowvar=False)-np.eye(3)
            # Consecutive independent draws in each component stand for successive
            # time samples; the generator has no time-specific sampling branch.
            lag=max(abs(np.corrcoef(field[:-1,k],field[1:,k])[0,1]) for k in range(3))
            mean_limit=6/math.sqrt(count);variance_limit=6*math.sqrt(2/(count-1))
            corr_limit=6/math.sqrt(len(field)-1)
            passed=abs(mean)<mean_limit and abs(variance-1)<variance_limit and abs(cross).max()<corr_limit and lag<corr_limit
            rows.append(dict(case=case,theta=theta,components=count,normalized_mean=float(mean),
                normalized_variance=float(variance),max_cross_correlation=float(abs(cross).max()),
                max_lag1_correlation=float(lag),expected_field_variance=expected,
                limits=dict(mean=mean_limit,variance_deviation=variance_limit,correlation=corr_limit),
                passed=bool(passed)))
    return dict(status='pass' if all(r['passed'] for r in rows) else 'fail',rows=rows,
                independence_note='Same frozen seed reused across scale checks, not six independent noise experiments.',
                tolerance_note='Predeclared six-standard-error thresholds for PRNG diagnostics; not an equilibrium certificate.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=noise_gate(load_runtime(a.config));result['config_sha256']=sha256(a.config)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
    if result['status']!='pass': raise SystemExit(2)


if __name__=='__main__': main()
