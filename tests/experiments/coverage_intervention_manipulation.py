"""Equal-count coverage check of the actual intervention training inputs."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scripts.workflow.bauer_campaign import digest,save
from tests.experiments.local_state_coverage import descriptors
from tests.experiments.coverage_transition_dataset import CoverageTransitionDataset
from tests.experiments.conditional_restart import inputs


def run():
    vp=Path('tests/results/coverage_intervention_data/verification.json')
    verification=json.loads(vp.read_text())
    assert verification['status']=='VERIFIED_FIXED_COUNT_DATA_READY_FOR_CONTROLLED_TRAINING'
    for p,h in verification['code_sha256'].items(): assert digest(p)==h
    dataset=CoverageTransitionDataset(); manifest,states=inputs()
    query=descriptors(states)
    banks={}; temperatures={}
    for label,d in [('control',dataset.original),('coverage',dataset)]:
        samples=[d[i] for i in range(len(d))]
        banks[label]=descriptors(np.stack([r['initial'] for r in samples]))
        temperatures[label]=np.array([r['theta'] for r in samples])
    assert banks['control'].shape==banks['coverage'].shape==(12288,25,10)
    np.testing.assert_array_equal(temperatures['control'],temperatures['coverage'])
    distance={k:np.full((60,25),np.nan) for k in banks}
    brute_checks=[]
    for theta in [.11,.13]:
        ids=[i for i,r in enumerate(manifest['rows']) if r['state']['theta']==theta]
        for region,sites in [('left',[0]),('interior',list(range(1,24))),('right',[24])]:
            for label,features in banks.items():
                bank=features[temperatures[label]==theta][:,sites].reshape(-1,10)
                points=query[ids][:,sites].reshape(-1,10)
                nearest=cKDTree(bank).query(points,workers=1)[0]
                # Independent brute force check, one query for every arm/temperature/boundary stratum.
                brute=np.sqrt(((bank-points[0])**2).sum(-1)).min()
                np.testing.assert_allclose(brute,nearest[0],atol=1e-14)
                brute_checks.append(dict(label=label,theta=theta,region=region,distance=float(brute)))
                distance[label][np.ix_(ids,sites)]=nearest.reshape(len(ids),len(sites))
    assert all(np.isfinite(a).all() for a in distance.values())
    root=Path('tests/results/coverage_intervention_manipulation'); root.mkdir(exist_ok=False)
    path=root/'distances.npz'
    np.savez_compressed(path,**distance)
    rows=[]
    for i,state in enumerate(manifest['rows']):
        for region,sites in [('left',[0]),('interior',list(range(1,24))),('right',[24])]:
            rows.append(dict(state_id=i,state=state,region=region,
                medians={k:float(np.median(v[i,sites])) for k,v in distance.items()},
                fraction_closer=float(np.mean(distance['coverage'][i,sites]<distance['control'][i,sites]))))
    groups=[]
    for kind in ['aligned','tilted','wall']:
        for frame in [0,20,40]:
            selected=[r for r in rows if r['region']=='interior' and r['state']['state']['kind']==kind and r['state']['frame']==frame]
            group=dict(kind=kind,frame=frame,
                mean_median_distance={k:float(np.mean([r['medians'][k] for r in selected])) for k in banks},
                fraction_closer=float(np.mean([r['fraction_closer'] for r in selected])))
            groups.append(group); print(group)
    save(root/'summary.json',dict(status='COMPLETE_EQUAL_COUNT_COVERAGE_CHECK_NOT_CERTIFIED',
        verification_sha256=digest(vp),code_sha256={p:digest(p) for p in [__file__,
            'tests/experiments/local_state_coverage.py','tests/experiments/coverage_transition_dataset.py']},
        state_manifest_sha256=digest('tests/results/conditional_restart/manifest.json'),
        distance_path=str(path),distance_sha256=digest(path),rows=rows,groups=groups,brute_force_checks=brute_checks,
        scope='Equal 12288 training initial states in both arms, matched temperature and boundary type. '
        'Ten radius-one invariants, not full model receptive field/flow support. No fitted cutoff or significance test. '
        'Correlated sites/windows and parent families are not independent replicates. Smaller distance verifies a local coverage change, not model accuracy.'))


if __name__=='__main__': run()
