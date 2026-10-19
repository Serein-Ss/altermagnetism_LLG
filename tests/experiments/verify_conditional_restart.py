"""Independent point-statistic recomputation for all restart conditions."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.verify_fresh_model_audit import profiles


def run():
    root=Path('tests/results/conditional_restart')
    source=root/'analysis.json'; summary=json.loads(source.read_text())
    assert summary['status']=='COMPLETE_CONDITIONAL_RESTART_ANALYSIS_NOT_CERTIFIED'
    assert len(summary['rows'])==300
    assert {(r['label'],r['state_id']) for r in summary['rows']}=={(l,s) for l in ['case0','case2','model11','model22','model33'] for s in range(60)}
    reports={}
    for label,h in summary['source_sha256'].items():
        p=root/f'{label}.json'; assert digest(p)==h
        reports[label]=json.loads(p.read_text())
    for p,h in summary['code_sha256'].items(): assert digest(p)==h
    refs={ (r['case'],r['state_id']):r for r in reports['reference']['rows']}
    models={k:{r['state_id']:r for r in v['rows']} for k,v in reports.items() if k!='reference'}
    verified=[]
    for row in summary['rows']:
        label,state=row['label'],row['state_id']
        candidate=refs[int(label[-1]),state] if label.startswith('case') else models[label][state]
        data=[]
        for r in [candidate,refs[1,state]]:
            assert digest(r['path'])==r['sha256']
            with np.load(r['path']) as z: data.append(z['spins'].astype(float))
        a,b=data
        m,v,bond,t=profiles(a); rm,rv,rb,rt=profiles(b)
        point=dict(mean_rmse=np.sqrt(((m-rm)**2).mean((1,2))),variance_difference=v-rv,
            bond_rmse=np.sqrt(((bond-rb)**2).mean(1)),temporal_connected_difference=t-rt)
        da=a-a[:,:1]; db=b-b[:,:1]
        noise=(da.var(0,ddof=1)/len(a)+db.var(0,ddof=1)/len(b)).mean((1,2))[1]
        raw=float(point['mean_rmse'][0]**2)
        np.testing.assert_allclose([raw,noise,raw-noise],[row['raw_mse'],row['mean_noise'],row['corrected_mse']],atol=1e-14,rtol=1e-9)
        assert digest(row['path'])==row['sha256']
        with np.load(row['path']) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
            for k,estimate in point.items():
                np.testing.assert_allclose(estimate[0],row['metrics'][k]['estimate'],atol=1e-12,rtol=1e-9)
                assert z[k].shape==(2000,1)
                np.testing.assert_allclose(np.quantile(z[k][:,0],[.025,.975]),row['metrics'][k]['ci95'],atol=1e-12)
            for p,k in [(a,'generated_covariance'),(b,'reference_covariance')]:
                c=(p[:,1]-p[:,1].mean(0)).reshape(len(p),-1)
                np.testing.assert_allclose(c.T@c/(len(p)-1),z[k],atol=1e-14)
        verified.append(dict(label=label,state_id=state,state=row['state'],raw_noise_ratio=raw/noise,
            corrected_mse=raw-noise,variance_ratio=float(v[0]/rv[0]),bond_rmse=float(point['bond_rmse'][0]),
            covariance_rmse=row['covariance_rmse']))
    groups=[]
    for kind in ['aligned','tilted','wall']:
        for label in ['case0','case2','model11','model22','model33']:
            for frame in [0,20,40]:
                rows=[r for r in verified if r['label']==label and r['state']['state']['kind']==kind and r['state']['frame']==frame]
                groups.append(dict(kind=kind,label=label,frame=frame,count=len(rows),
                    means={k:float(np.mean([r[k] for r in rows])) for k in ['raw_noise_ratio','corrected_mse','variance_ratio','bond_rmse','covariance_rmse']}))
    save(root/'verified_analysis.json',dict(status='VERIFIED_300_RAW_POINT_COMPARISONS_NOT_CERTIFIED',
        source_sha256=digest(source),code_sha256=digest(__file__),rows=verified,groups=groups,
        scope='All point estimates and covariance matrices independently recomputed; intervals checked against saved draws, not regenerated. Group means descriptive; correlated states/temperatures and shared reference across models not independent replicates. Raw/noise ratio is not a pass threshold.'))
    print('Verified all 300 comparisons, point statistics, covariance matrices and stored interval quantiles.')


if __name__=='__main__': run()
