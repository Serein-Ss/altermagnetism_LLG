"""Paired original-versus-coverage rollout comparison at unchanged budget."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest,save
from tests.experiments.director_budget_analysis import paired_statistics
from tests.experiments.fresh_reference_audit import FRAMES,path_energy
from tests.experiments.fresh_model_audit import temporal_covariance


def compare(data,rng,replicates=2000):
    arrays,metrics=paired_statistics(data,rng,replicates)
    mapping={'6000':'control','24000':'coverage','paired_change':'paired_change'}
    renamed={}
    for k,v in arrays.items():
        prefix=next(p for p in mapping if k.startswith(p+'_'))
        renamed[mapping[prefix]+k[len(prefix):]]=v
    return renamed,{mapping[k]:v for k,v in metrics.items()}


def run():
    rp=Path('tests/results/fresh_rollout_reference/summary.json'); reference=json.loads(rp.read_text())
    assert reference['status']=='COMPLETE_FRESH_ROLLOUT_REFERENCE_NOT_CERTIFIED'
    refs={r['anchor']:r for r in reference['rows'] if r['case']==1}
    reports={}
    for seed in [11,22,33]:
        npth=Path(f'tests/results/coverage_intervention_training/{seed}/summary.json')
        opth=Path(f'tests/results/fresh_rollout_sampling/{seed}/summary.json')
        new=json.loads(npth.read_text()); old=json.loads(opth.read_text())
        assert new['status']=='COMPLETE_COVERAGE_TRAINING_AND_ROLLOUT_NOT_CERTIFIED' and new['control_rng_matched']
        assert new['rollout_control_sha256']==digest(opth)
        assert len(new['rows'])==12 and {r['anchor'] for r in new['rows']}==set(range(12))
        assert new['config']==old['config'] or new['rollout_config']==old['config']
        reports[seed]=(new,old,digest(npth),digest(opth))
    root=Path('tests/results/coverage_intervention_analysis'); root.mkdir(exist_ok=False)
    result=dict(status='RUNNING',rows=[],replicates=2000,reference_sha256=digest(rp),
        code_sha256={p:digest(p) for p in [__file__,'tests/experiments/director_budget_analysis.py',
            'tests/experiments/director_uncertainty.py','tests/experiments/fresh_reference_audit.py',
            'tests/experiments/fresh_model_audit.py']},
        scope='Fixed-count data-coverage intervention at fixed optimization. Paired latent path indices across models, independent reference resampling. Pointwise intervals, no equivalence gate. Previously used diagnostic states are development, not final confirmation. Joint energy/covariance at five selected times; temporal covariance same-site site-averaged. No full path or size-transfer certificate.')
    for seed,(new,old,nh,oh) in reports.items():
        for anchor in range(12):
            a=next(r for r in old['rows'] if r['anchor']==anchor and r['steps']==32)
            b=next(r for r in new['rows'] if r['anchor']==anchor)
            assert a['latent_seed']==b['latent_seed'] and a['state']==b['state']
            data=[]; rngstates=[]
            for row in [a,b,refs[anchor]]:
                assert digest(row['path'])==row['sha256']
                with np.load(row['path']) as z:
                    p=z['spins'].astype(float); times=z['reduced_time']
                    if 'rng_state' in z: rngstates.append(z['rng_state'])
                assert p.shape==(256,81,25,3) and np.isfinite(p).all()
                assert np.max(abs(np.linalg.norm(p,axis=-1)-1))<1e-5
                np.testing.assert_allclose(times,np.arange(81)*.01,atol=1e-12,rtol=0)
                if data: np.testing.assert_allclose(p[:,0],data[0][:,0],atol=2e-7,rtol=0)
                data.append(p)
            np.testing.assert_array_equal(*rngstates)
            arrays,metrics=compare(data,np.random.default_rng(999000000+seed*1000+anchor))
            arrays['time']=times[1:]
            energy={}
            for label,p in zip(['control','coverage','reference'],data):
                arrays[label+'_selected_joint_covariance']=np.cov(p[:,FRAMES].reshape(256,-1),rowvar=False)
                arrays[label+'_temporal_covariance']=temporal_covariance(p)
                if label!='reference': energy[label]=path_energy(p[:,FRAMES],data[2][:,FRAMES])
            assert all(np.isfinite(v).all() for v in arrays.values())
            path=root/f'seed{seed}_anchor{anchor}.npz'; np.savez_compressed(path,**arrays)
            result['rows'].append(dict(seed=seed,anchor=anchor,state=b['state'],path=str(path),sha256=digest(path),
                source_sha256={'control':oh,'coverage':nh},metrics=metrics,joint_energy=energy))
            save(root/'summary.json',result)
    result['status']='COMPLETE_COVERAGE_COMPARISON_NOT_CERTIFIED'; save(root/'summary.json',result)


if __name__=='__main__': run()
