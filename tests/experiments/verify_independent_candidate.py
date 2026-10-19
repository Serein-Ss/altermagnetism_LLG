"""Recompute raw profiles and covariance; verify stored bootstrap intervals."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.verify_fresh_model_audit import profiles


def verify_metrics(candidate, reference, metrics, arrays):
    m,v,b,t = profiles(candidate); rm,rv,rb,rt = profiles(reference)
    points = dict(mean_rmse=np.sqrt(((m-rm)**2).mean((1,2))),
        variance_difference=v-rv, bond_rmse=np.sqrt(((b-rb)**2).mean(1)),
        temporal_connected_difference=t-rt)
    for key, value in points.items():
        np.testing.assert_allclose(value, metrics[key]['estimate'], atol=1e-12, rtol=1e-9)
        np.testing.assert_allclose(np.quantile(arrays[key],[.025,.975],axis=0),
                                   metrics[key]['ci95'], atol=1e-12)
    return dict(mean_rmse=float(points['mean_rmse'][-1]),
        bond_rmse=float(points['bond_rmse'][-1]), variance_ratio=float(v[-1]/rv[-1]))


def run():
    root = Path('tests/results/independent_candidate_analysis')
    source = root/'summary.json'; summary = json.loads(source.read_text())
    assert summary['status']=='COMPLETE_INDEPENDENT_CANDIDATE_ANALYSIS_NOT_CERTIFIED'
    labels = ['reference0','reference2','model11','model22','model33']
    assert len(summary['rows'])==120
    assert {(r['label'],r['anchor']) for r in summary['rows']}=={(l,a) for l in labels for a in range(24)}
    for p,h in {**summary['sources'],**summary['code_sha256']}.items():
        assert digest(p)==h
    rows=[]
    for row in summary['rows']:
        data=[]
        for key in ['candidate','reference']:
            item=row[key]; assert digest(item['path'])==item['sha256']
            with np.load(item['path']) as z:
                p=z['spins'].astype(float)
                np.testing.assert_allclose(z['reduced_time'],np.arange(81)*.01,atol=1e-12,rtol=0)
            assert p.shape==(512,81,25,3) and np.isfinite(p).all()
            assert np.max(abs(np.linalg.norm(p,axis=-1)-1))<1e-5
            data.append(p)
        assert digest(row['path'])==row['sha256']
        with np.load(row['path']) as arrays:
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
            for key in row['metrics']: assert arrays[key].shape==(2000,80)
            terminal=verify_metrics(*data,row['metrics'],arrays)
            selected=[]
            for label,p in zip(['candidate','reference'],data):
                x=p[:,[1,8,20,40,80]].reshape(512,-1); selected.append(x)
                c=x-x.mean(0)
                np.testing.assert_allclose(c.T@c/511,arrays[label+'_selected_covariance'],atol=1e-12)
                c=p.reshape(512,81,-1); c=c-c.mean(0)
                covariance=np.einsum('ntd,nsd->ts',c,c)/(511*25)
                np.testing.assert_allclose(covariance,arrays[label+'_temporal_covariance'],atol=1e-12)
        x,y=selected
        energy=(2*cdist(x,y).mean()-cdist(x,x).sum()/(512*511)-cdist(y,y).sum()/(512*511))/np.sqrt(x.shape[1])
        np.testing.assert_allclose(energy,row['joint_energy'],atol=1e-12)
        rows.append(dict(label=row['label'],anchor=row['anchor'],state=row['state'],
                         terminal=terminal,joint_energy=float(energy)))
    save(root/'verification.json',dict(status='VERIFIED_INDEPENDENT_CANDIDATE_RESULTS_NOT_CERTIFIED',
        source_sha256=digest(source),code_sha256={p:digest(p) for p in [__file__,
        'tests/experiments/verify_fresh_model_audit.py']},rows=rows,
        scope='Raw point profiles, selected covariance, same-site temporal covariance and energy recomputed. '
        'Stored bootstrap quantiles checked; bootstrap draws not regenerated. No equivalence or full-path certification.'))
    print('Verified 120 raw comparisons.')


if __name__=='__main__':
    run()
