"""Fit fixed-lag diagnostic using the exact mixed training pairs, no validation."""
import json
import time
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest,save
from tests.experiments.coverage_transition_dataset import CoverageTransitionDataset
from tests.experiments.director_budget import training_arrays
from tests.experiments.sample_tangent_baseline import fit


def run():
    output=Path('tests/output/sample_tangent_baseline/model.json')
    if output.exists():raise FileExistsError('Preserve fitted baseline')
    started=time.monotonic()
    x,y,theta,_=training_arrays(CoverageTransitionDataset())
    assert len(x)==12288
    x/=np.linalg.norm(x,axis=-1,keepdims=True)
    y/=np.linalg.norm(y,axis=-1,keepdims=True)
    rows=[]
    for temperature in [.11,.13]:
        select=theta==temperature
        assert select.sum()==6144
        coefficients,sigma,diagnostics=fit(x[select],y[select],np.array([0.,0.,1.]))
        rows.append(dict(theta=temperature,coefficients=coefficients.tolist(),
            sigma=sigma,pairs=int(select.sum()),diagnostics=diagnostics))
    output.parent.mkdir(parents=True)
    save(output,dict(status='FITTED_TRAINING_ONLY_NOT_VALIDATED',rows=rows,lag=.01,
        axis=[0,0,1],seconds=time.monotonic()-started,
        sources={p:digest(p) for p in ['tests/results/coverage_intervention_data/verification.json',
        'tests/results/fine_transition_pilot/manifest.json']},
        code_sha256={p:digest(p) for p in [__file__,'tests/experiments/sample_tangent_baseline.py',
        'tests/experiments/coverage_transition_dataset.py','tests/experiments/director_budget.py']},
        scope='Restricted tangent Gaussian baseline with sample-learned mean coefficients and isotropic residual scale. '
        'Local nearest-neighbor and anisotropy-director features encode inductive structure, not supplied LLG force coefficients. '
        'Only trained temperatures and lag supported; no claim of non-Gaussian one-step fidelity, instantaneous generator, speedup or path-law certification.'))
    print(json.dumps(rows))


if __name__=='__main__':run()
