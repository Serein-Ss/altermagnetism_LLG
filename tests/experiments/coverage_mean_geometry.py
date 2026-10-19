"""Symmetric exact decomposition of mean-bond amplitude and direction errors."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest,save


def geometry(mean,reference):
    length=np.linalg.norm(mean,axis=-1); rl=np.linalg.norm(reference,axis=-1)
    if min(length.min(),rl.min())<1e-10:
        raise ValueError('mean direction undefined; do not interpret angular decomposition')
    unit=mean/length[...,None]; ru=reference/rl[...,None]
    amplitude=length[:,:-1]*length[:,1:]; ra=rl[:,:-1]*rl[:,1:]
    cosine=(unit[:,:-1]*unit[:,1:]).sum(-1); rc=(ru[:,:-1]*ru[:,1:]).sum(-1)
    magnitude=.5*(cosine+rc)*(amplitude-ra)
    direction=.5*(amplitude+ra)*(cosine-rc)
    total=(mean[:,:-1]*mean[:,1:]).sum(-1)-(reference[:,:-1]*reference[:,1:]).sum(-1)
    np.testing.assert_allclose(total,magnitude+direction,atol=1e-12,rtol=0)
    return dict(length_error=length-rl,direction_chord=np.linalg.norm(unit-ru,axis=-1),
        mean_bond_error=total,magnitude_contribution=magnitude,direction_contribution=direction,
        length=length,reference_length=rl,bond_cosine_error=cosine-rc)


def run():
    source=Path('tests/results/coverage_intervention_analysis/summary.json')
    summary=json.loads(source.read_text())
    verification=json.loads(source.with_name('verification.json').read_text())
    assert verification['source_sha256']==digest(source)
    assert verification['status']=='VERIFIED_RAW_COVERAGE_COMPARISON_NOT_CERTIFIED'
    rp=Path('tests/results/fresh_rollout_reference/summary.json')
    assert digest(rp)==summary['reference_sha256']
    refs={r['anchor']:r for r in json.loads(rp.read_text())['rows'] if r['case']==1}
    root=Path('tests/results/coverage_mean_geometry'); root.mkdir(exist_ok=False)
    rows=[]
    for row in summary['rows']:
        seed,anchor=row['seed'],row['anchor']; items={'reference':refs[anchor]}
        for label,path in [('control',f'tests/results/fresh_rollout_sampling/{seed}/summary.json'),
                           ('coverage',f'tests/results/coverage_intervention_training/{seed}/summary.json')]:
            assert digest(path)==row['source_sha256'][label]
            report=json.loads(Path(path).read_text())
            items[label]=next(r for r in report['rows'] if r['anchor']==anchor and (label=='coverage' or r['steps']==32))
        means={}; norm_errors={}
        for label,item in items.items():
            assert digest(item['path'])==item['sha256']
            with np.load(item['path']) as z: p=z['spins'].astype(float)
            assert p.shape==(256,81,25,3) and np.isfinite(p).all()
            means[label]=p.mean(0)
            # Exact empirical identity: E||s||^2 = ||E s||^2 + sum Var_N(s).
            np.testing.assert_allclose((means[label]**2).sum(-1)+p.var(0).sum(-1),(p*p).sum(-1).mean(0),atol=1e-12)
            norm_errors[label]=float(np.max(abs((p*p).sum(-1)-1)))
        arrays={label+'_mean':m for label,m in means.items()}; metrics={}
        for label in ['control','coverage']:
            g=geometry(means[label],means['reference']); arrays.update({label+'_'+k:v for k,v in g.items()})
            metrics[label]={k:float(np.sqrt(np.mean(g[k][-1]**2))) for k in ['length_error','direction_chord','mean_bond_error','magnitude_contribution','direction_contribution']}
            metrics[label]['mse_cross_term']=float(2*np.mean(g['magnitude_contribution'][-1]*g['direction_contribution'][-1]))
            metrics[label]['regions']={region:{k:float(np.sqrt(np.mean(g[k][-1,index]**2))) for k in ['length_error','direction_chord']} for region,index in [('ends',np.r_[0:3,22:25]),('center',np.arange(8,17)),('shoulders',np.r_[3:8,17:22])]}
        path=root/f'seed{seed}_anchor{anchor}.npz'; np.savez_compressed(path,**arrays)
        rows.append(dict(seed=seed,anchor=anchor,state=row['state'],path=str(path),sha256=digest(path),metrics=metrics,max_squared_norm_error=norm_errors))
    groups=[]
    for kind in ['aligned','tilted','wall']:
        for theta in [.11,.13]:
            selected=[r for r in rows if r['state']['kind']==kind and r['state']['theta']==theta]
            group=dict(kind=kind,theta=theta,terminal={label:{k:float(np.mean([r['metrics'][label][k] for r in selected])) for k in ['length_error','direction_chord','mean_bond_error','magnitude_contribution','direction_contribution','mse_cross_term']} for label in ['control','coverage']})
            groups.append(group); print(group)
    save(root/'summary.json',dict(status='COMPLETE_EXPLORATORY_MEAN_GEOMETRY_NOT_CERTIFIED',rows=rows,groups=groups,
        source_sha256=digest(source),code_sha256=digest(__file__),
        scope='Mean bond = product of mean lengths times cosine of neighbor mean directions. Symmetric amplitude/direction decomposition, cross term retained. Direction requires nonzero mean. For unit spins, mean length also encodes on-site variance; mean-product error is not solely deterministic error. Fixed descriptive site regions, no significance, uncertainty or causal claim.'))


if __name__=='__main__': run()
