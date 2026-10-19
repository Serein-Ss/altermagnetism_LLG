"""Independent raw-point recomputation and descriptive frozen rollout audit."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save


def profiles(p):
    d = p-p[:, :1]
    c = d-d.mean(0)
    return (d.mean(0)[1:], d.var(0, ddof=1).sum(-1).mean(-1)[1:],
            (p[:, :, :-1]*p[:, :, 1:]).sum(-1).mean(0)[1:],
            (c[:, 1:]*c[:, :-1]).sum(-1).sum(0).mean(-1)/(len(p)-1))


def run():
    root = Path('tests/results/fresh_model_audit')
    summary = json.loads((root/'summary.json').read_text())
    assert summary['status'] == 'COMPLETE_FRESH_MODEL_AUDIT_NOT_CERTIFIED'
    assert len(summary['rows']) == 36
    assert {(r['seed'], r['anchor']) for r in summary['rows']} == {(s,a) for s in [11,22,33] for a in range(12)}
    for p,h in summary['code_sha256'].items():
        assert digest(p) == h
    rp = Path('tests/results/fresh_rollout_reference/summary.json')
    assert digest(rp) == summary['reference_sha256']
    refs = {r['anchor']:r for r in json.loads(rp.read_text())['rows'] if r['case']==1}
    rows = []
    for r in summary['rows']:
        sp = Path(f"tests/results/fresh_rollout_sampling/{r['seed']}/summary.json")
        assert digest(sp) == r['source_sha256']
        samples = json.loads(sp.read_text())
        raw = []
        for source in [next(x for x in samples['rows'] if x['anchor']==r['anchor'] and x['steps']==s) for s in [32,64]]+[refs[r['anchor']]]:
            assert digest(source['path']) == source['sha256']
            with np.load(source['path']) as z:
                p = z['spins'].astype(float)
                np.testing.assert_allclose(z['reduced_time'], np.arange(81)*.01, atol=1e-12, rtol=0)
            assert p.shape == (256,81,25,3) and np.isfinite(p).all()
            assert np.max(abs(np.linalg.norm(p,axis=-1)-1)) < 1e-5
            raw.append(p)
        assert digest(r['path']) == r['sha256']
        with np.load(r['path']) as z:
            assert all(np.isfinite(z[k]).all() for k in z.files)
            for label,metrics in r['metrics'].items():
                for key,m in metrics.items():
                    draws = z[label+'_'+key]
                    assert draws.shape == (2000,80)
                    np.testing.assert_allclose(np.quantile(draws,[.025,.975],axis=0),m['ci95'],atol=1e-12)
                    if label == 'paired_change':
                        np.testing.assert_array_equal(draws,z['64_'+key]-z['32_'+key])
            rm,rv,rb,rt = profiles(raw[2])
            for label,p in zip(['32','64'],raw[:2]):
                m,v,b,t = profiles(p)
                estimates = dict(mean_rmse=np.sqrt(((m-rm)**2).mean((1,2))),
                    variance_difference=v-rv,bond_rmse=np.sqrt(((b-rb)**2).mean(1)),
                    temporal_connected_difference=t-rt)
                for k,value in estimates.items():
                    np.testing.assert_allclose(value,r['metrics'][label][k]['estimate'],rtol=1e-9,atol=1e-12)
                np.testing.assert_allclose(v/rv,z[label+'_variance_ratio'],rtol=1e-9)
            paired = np.sqrt(((raw[1]-raw[0])**2).mean((0,2,3)))
            np.testing.assert_allclose(paired,z['paired_path_rmse'],atol=1e-12)
            rows.append(dict(seed=r['seed'],anchor=r['anchor'],state=r['state'],
                terminal_mean_rmse=r['metrics']['32']['mean_rmse']['estimate'][-1],
                terminal_bond_rmse=r['metrics']['32']['bond_rmse']['estimate'][-1],
                terminal_variance_ratio=float(z['32_variance_ratio'][-1]),
                paired_path_rmse_max=float(paired.max()),joint_energy=r['joint_energy']['32']))
    groups = []
    for kind in ['aligned','tilted','wall']:
        for theta in [.11,.13]:
            subset = [r for r in rows if r['state']['kind']==kind and r['state']['theta']==theta]
            groups.append(dict(kind=kind,theta=theta,n_conditions_seeds=len(subset),
                descriptive={k:dict(mean=float(np.mean([r[k] for r in subset])),
                    min=min(r[k] for r in subset),max=max(r[k] for r in subset))
                    for k in ['terminal_mean_rmse','terminal_bond_rmse','terminal_variance_ratio','paired_path_rmse_max','joint_energy']}))
    result = dict(status='VERIFIED_RAW_POINTS_AND_STORED_INTERVALS_NOT_CERTIFIED',
        source_sha256=digest(root/'summary.json'),code_sha256=digest(__file__),rows=rows,groups=groups,
        scope='Independent raw recomputation of point profiles; interval quantiles checked against stored draws, not bootstrap regenerated. Six initial families crossed with temperature and three model seeds; grouped ranges descriptive, not confidence intervals. No equivalence decision or complete path-law certification.')
    save(root/'verified_aggregate.json',result)
    print(json.dumps(groups,indent=2))


if __name__ == '__main__':
    run()
