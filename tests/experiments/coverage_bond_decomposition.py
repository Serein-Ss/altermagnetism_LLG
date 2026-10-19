"""Exact empirical bond decomposition into mean and connected fluctuations."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest,save


def components(p):
    p=np.asarray(p,dtype=float)
    mean=p.mean(0); centered=p-mean
    total=(p[:,:,:-1]*p[:,:,1:]).sum(-1).mean(0)
    product=(mean[:,:-1]*mean[:,1:]).sum(-1)
    connected=(centered[:,:,:-1]*centered[:,:,1:]).sum(-1).mean(0)
    np.testing.assert_allclose(total,product+connected,atol=1e-12,rtol=0)
    return dict(total=total,mean_product=product,connected=connected)


def run():
    source=Path('tests/results/coverage_intervention_analysis/summary.json')
    summary=json.loads(source.read_text())
    verification=json.loads(source.with_name('verification.json').read_text())
    assert verification['source_sha256']==digest(source)
    assert verification['status']=='VERIFIED_RAW_COVERAGE_COMPARISON_NOT_CERTIFIED'
    rp=Path('tests/results/fresh_rollout_reference/summary.json')
    assert digest(rp)==summary['reference_sha256']
    refs={r['anchor']:r for r in json.loads(rp.read_text())['rows'] if r['case']==1}
    root=Path('tests/results/coverage_bond_decomposition'); root.mkdir(exist_ok=False)
    rows=[]
    for row in summary['rows']:
        seed,anchor=row['seed'],row['anchor']; items={}
        for label,path in [('control',f'tests/results/fresh_rollout_sampling/{seed}/summary.json'),
                           ('coverage',f'tests/results/coverage_intervention_training/{seed}/summary.json')]:
            assert digest(path)==row['source_sha256'][label]
            report=json.loads(Path(path).read_text())
            items[label]=next(r for r in report['rows'] if r['anchor']==anchor and (label=='coverage' or r['steps']==32))
        items['reference']=refs[anchor]; profiles={}; arrays={}
        for label,item in items.items():
            assert digest(item['path'])==item['sha256']
            with np.load(item['path']) as z: p=z['spins']; times=z['reduced_time']
            assert p.shape==(256,81,25,3)
            profiles[label]=components(p)
            arrays.update({label+'_'+key:value for key,value in profiles[label].items()})
        metrics={}
        for label in ['control','coverage']:
            errors={key:profiles[label][key]-profiles['reference'][key] for key in profiles[label]}
            metrics[label]={}
            for key,error in errors.items():
                arrays[label+'_'+key+'_error']=error
                metrics[label][key]=dict(rmse=np.sqrt((error**2).mean(-1)).tolist(),signed_mean=error.mean(-1).tolist())
            np.testing.assert_allclose(metrics[label]['total']['rmse'][1:],row['metrics'][label]['bond_rmse']['estimate'],atol=1e-12,rtol=1e-9)
            cross=2*(errors['mean_product']*errors['connected']).mean(-1)
            arrays[label+'_mse_cross_term']=cross
            np.testing.assert_allclose((errors['total']**2).mean(-1),
                (errors['mean_product']**2).mean(-1)+(errors['connected']**2).mean(-1)+cross,atol=1e-14)
            metrics[label]['mse_cross_term']=cross.tolist()
        path=root/f'seed{seed}_anchor{anchor}.npz'; np.savez_compressed(path,time=times,**arrays)
        rows.append(dict(seed=seed,anchor=anchor,state=row['state'],path=str(path),sha256=digest(path),metrics=metrics))
    groups=[]
    for kind in ['aligned','tilted','wall']:
        for theta in [.11,.13]:
            selected=[r for r in rows if r['state']['kind']==kind and r['state']['theta']==theta]
            group=dict(kind=kind,theta=theta,terminal={label:{key:float(np.mean([r['metrics'][label][key]['rmse'][-1] for r in selected])) for key in ['total','mean_product','connected']} for label in ['control','coverage']})
            groups.append(group); print(group)
    save(root/'summary.json',dict(status='COMPLETE_EMPIRICAL_BOND_DECOMPOSITION_NOT_CERTIFIED',source_sha256=digest(source),
        code_sha256=digest(__file__),rows=rows,groups=groups,
        scope='Exact empirical identity using connected covariance with denominator N, not unbiased N-1, so terms sum exactly to sample bond. Errors and MSE cross term retained: component RMSEs cannot be added, cancellation is possible. Descriptive decomposition without uncertainty or causal attribution; correlated conditions not independent replicates.'))


if __name__=='__main__': run()
