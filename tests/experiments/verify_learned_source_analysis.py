"""Recompute reflection augmentation comparison from saved trajectories."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest,save
from tests.experiments.verify_fresh_model_audit import profiles
from tests.experiments.fresh_reference_audit import FRAMES


def verify_paired_draws(archive,key):
    np.testing.assert_array_equal(archive['paired_change_'+key],
        archive['learned_source_'+key]-archive['control_'+key])


def run():
    root=Path('tests/results/learned_source_analysis')
    source=root/'summary.json'; summary=json.loads(source.read_text())
    assert summary['status']=='COMPLETE_LEARNED_SOURCE_COMPARISON_NOT_CERTIFIED'
    assert len(summary['rows'])==36
    assert {(r['seed'],r['anchor']) for r in summary['rows']}=={(s,a) for s in [11,22,33] for a in range(12)}
    for p,h in summary['code_sha256'].items(): assert digest(p)==h
    rp=Path('tests/results/fresh_rollout_reference/summary.json')
    assert digest(rp)==summary['reference_sha256']
    refs={r['anchor']:r for r in json.loads(rp.read_text())['rows'] if r['case']==1}
    rows=[]
    for row in summary['rows']:
        seed,anchor=row['seed'],row['anchor']
        paths={'control':Path(f'tests/results/reflection_augmentation_training/{seed}/summary.json'),
            'learned_source':Path(f'tests/results/learned_source_training/{seed}/summary.json')}
        data={}; reports={}; rngstates=[]
        for label,p in paths.items():
            assert digest(p)==row['source_sha256'][label]
            reports[label]=json.loads(p.read_text())
        items={label:next(r for r in report['rows'] if r['anchor']==anchor) for label,report in reports.items()}
        items['reference']=refs[anchor]
        for label,item in items.items():
            assert digest(item['path'])==item['sha256']
            with np.load(item['path']) as z:
                p=z['spins'].astype(float)
                np.testing.assert_allclose(z['reduced_time'],np.arange(81)*.01,atol=1e-12,rtol=0)
                if 'rng_state' in z: rngstates.append(z['rng_state'])
            assert p.shape==(256,81,25,3) and np.isfinite(p).all()
            assert np.max(abs(np.linalg.norm(p,axis=-1)-1))<1e-5
            data[label]=p
        np.testing.assert_array_equal(*rngstates)
        rm,rv,rb,rt=profiles(data['reference'])
        assert digest(row['path'])==row['sha256']
        with np.load(row['path']) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
            for label in ['control','learned_source']:
                m,v,b,t=profiles(data[label])
                estimates=dict(mean_rmse=np.sqrt(((m-rm)**2).mean((1,2))),variance_difference=v-rv,
                    bond_rmse=np.sqrt(((b-rb)**2).mean(1)),temporal_connected_difference=t-rt)
                for key,estimate in estimates.items():
                    np.testing.assert_allclose(estimate,row['metrics'][label][key]['estimate'],atol=1e-12,rtol=1e-9)
                np.testing.assert_allclose(v/rv,z[label+'_variance_ratio'],rtol=1e-9)
            for label,metrics in row['metrics'].items():
                for key,value in metrics.items():
                    draws=z[label+'_'+key]; assert draws.shape==(2000,80)
                    np.testing.assert_allclose(np.quantile(draws,[.025,.975],axis=0),value['ci95'],atol=1e-12)
                    if label=='paired_change':
                        verify_paired_draws(z,key)
                        np.testing.assert_allclose(value['estimate'],np.array(row['metrics']['learned_source'][key]['estimate'])-row['metrics']['control'][key]['estimate'],atol=1e-12)
            for label,p in data.items():
                flat=p[:,FRAMES].reshape(256,-1); c=flat-flat.mean(0)
                np.testing.assert_allclose(c.T@c/255,z[label+'_selected_joint_covariance'],atol=1e-12)
            rows.append(dict(seed=seed,anchor=anchor,state=row['state'],
                terminal={label:dict(mean_rmse=row['metrics'][label]['mean_rmse']['estimate'][-1],
                    bond_rmse=row['metrics'][label]['bond_rmse']['estimate'][-1],
                    variance_ratio=float(z[label+'_variance_ratio'][-1]),joint_energy=row['joint_energy'][label]) for label in ['control','learned_source']},
                paired_mean_ci95=np.asarray(row['metrics']['paired_change']['mean_rmse']['ci95'])[:,-1].tolist()))
    groups=[]
    for kind in ['aligned','tilted','wall']:
        for theta in [.11,.13]:
            selected=[r for r in rows if r['state']['kind']==kind and r['state']['theta']==theta]
            groups.append(dict(kind=kind,theta=theta,count=len(selected),
                terminal_means={label:{k:float(np.mean([r['terminal'][label][k] for r in selected])) for k in ['mean_rmse','bond_rmse','variance_ratio','joint_energy']} for label in ['control','learned_source']}))
    save(root/'verification.json',dict(status='VERIFIED_RAW_LEARNED_SOURCE_COMPARISON_NOT_CERTIFIED',
        source_sha256=digest(source),code_sha256={p:digest(p) for p in [__file__,'tests/experiments/verify_fresh_model_audit.py']},
        rows=rows,groups=groups,scope='Raw profiles and selected joint covariance independently recomputed. Stored bootstrap quantiles and paired differences verified, not bootstrap regenerated. Group means descriptive; no family independence, simultaneous intervals or equivalence claimed. Energy statistic carried from source, not independently recomputed here.'))
    print('Verified all 36 raw comparisons and saved paired bootstrap summaries.')


if __name__=='__main__': run()
