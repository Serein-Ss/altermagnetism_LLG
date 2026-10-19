"""Whole-path conditional restart summaries; no automatic pass threshold."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.conditional_restart import ROOT, inputs
from tests.experiments.director_uncertainty import weighted_profiles, differences
from tests.experiments.midpoint_mean_resolution import corrected_mean_mse


def run():
    m,states=inputs(); n=m['config']['paths']
    reports={}
    for label in ['reference','model11','model22','model33']:
        p=ROOT/f'{label}.json'; r=json.loads(p.read_text())
        expected='COMPLETE_CONDITIONAL_REFERENCE_NOT_CERTIFIED' if label=='reference' else 'COMPLETE_CONDITIONAL_MODEL_NOT_CERTIFIED'
        assert r['status']==expected and r['manifest_sha256']==digest(ROOT/'manifest.json')
        assert len(r['rows'])==(180 if label=='reference' else 60)
        reports[label]=r
    dest=ROOT/'analysis.json'
    if dest.exists(): raise FileExistsError(dest)
    result=dict(status='RUNNING',rows=[],replicates=2000,
        source_sha256={k:digest(ROOT/f'{k}.json') for k in reports},
        code_sha256={p:digest(p) for p in [__file__,'tests/experiments/director_uncertainty.py','tests/experiments/midpoint_mean_resolution.py']},
        scope='Exploratory fixed-state whole-path bootstrap; pointwise percentile intervals, not simultaneous intervals or zero-RMSE tests. Finite-sample corrected MSE retains negative estimates. States nested in original families/parent paths. No equivalence or continuum certificate.')
    for state in range(60):
        sources={f'case{c}':next(r for r in reports['reference']['rows'] if r['state_id']==state and r['case']==c) for c in range(3)}
        sources.update({k:next(r for r in reports[k]['rows'] if r['state_id']==state) for k in ['model11','model22','model33']})
        data={}
        for label,row in sources.items():
            assert digest(row['path'])==row['sha256']
            with np.load(row['path']) as z:
                p=z['spins'].astype(float)
                np.testing.assert_allclose(z['reduced_time'],[0,m['config']['lag']],atol=1e-12,rtol=0)
            assert p.shape==(n,2,25,3) and np.isfinite(p).all()
            np.testing.assert_allclose(p[:,0],np.repeat(states[state:state+1],n,axis=0),atol=2e-7,rtol=0)
            assert np.max(abs(np.linalg.norm(p,axis=-1)-1))<1e-5
            data[label]=p
        rng=np.random.default_rng(997000000+state)
        reference=data['case1']
        rp=weighted_profiles(reference,np.ones((1,n)))
        rb=weighted_profiles(reference,rng.multinomial(n,np.full(n,1/n),size=2000))
        for label in ['case0','case2','model11','model22','model33']:
            candidate=data[label]
            point=differences(weighted_profiles(candidate,np.ones((1,n))),rp)
            draws=differences(weighted_profiles(candidate,rng.multinomial(n,np.full(n,1/n),size=2000)),rb)
            raw,noise,corrected=corrected_mean_mse(candidate,reference)
            gc=np.cov(candidate[:,1].reshape(n,-1),rowvar=False)
            rc=np.cov(reference[:,1].reshape(n,-1),rowvar=False)
            path=ROOT/f'analysis_{label}_state{state}.npz'
            np.savez_compressed(path,**draws,raw_mse=raw,mean_noise=noise,corrected_mse=corrected,
                generated_covariance=gc,reference_covariance=rc)
            result['rows'].append(dict(label=label,state_id=state,state=m['rows'][state],
                path=str(path),sha256=digest(path),raw_mse=float(raw[0]),mean_noise=float(noise[0]),
                corrected_mse=float(corrected[0]),covariance_rmse=float(np.sqrt(np.mean((gc-rc)**2))),
                metrics={k:dict(estimate=float(point[k][0,0]),ci95=np.quantile(v[:,0],[.025,.975]).tolist()) for k,v in draws.items()}))
        save(dest,result)
    result['status']='COMPLETE_CONDITIONAL_RESTART_ANALYSIS_NOT_CERTIFIED'; save(dest,result)


if __name__=='__main__':
    run()
