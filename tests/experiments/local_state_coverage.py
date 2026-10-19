"""Descriptive local-invariant coverage, not full model support certification."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.fine_transition_dataset import FineTransitionDataset
from tests.experiments.conditional_restart import inputs, ROOT


def descriptors(x, axis=np.array([0.,0.,1.])):
    x=np.asarray(x,dtype=float)
    left=np.zeros_like(x); right=np.zeros_like(x)
    left[:,1:]=x[:,:-1]; right[:,:-1]=x[:,1:]
    tri=np.stack([left,x,right],axis=-2)
    z=tri@np.asarray(axis)
    i,j=np.triu_indices(3)
    return np.concatenate([np.stack([(left*x).sum(-1),(x*right).sum(-1),
        (left*right).sum(-1)],axis=-1),z[...,i]*z[...,j],
        (x*np.cross(left,right)).sum(-1)[...,None]],axis=-1)


def run():
    dataset=FineTransitionDataset('train')
    manifest,states=inputs()
    root=Path('tests/results/local_state_coverage'); root.mkdir(exist_ok=False)
    train={}; sources=[]
    for row in dataset.rows:
        assert digest(row['path'])==row['sha256']
        with np.load(row['path']) as z:
            p=z['spins'][:,dataset.starts]
        assert p.shape==(256,8,25,3)
        theta=row['state']['theta']
        halves=[descriptors(half.reshape(-1,25,3)) for half in [p[:128],p[128:]]]
        train.setdefault(theta,[]).append(halves)
        sources.append(dict(path=row['path'],sha256=row['sha256'],anchor=row['anchor']))
    query=descriptors(states)
    rows=[]; baselines=[]
    for theta,pairs in train.items():
        bank=np.concatenate([p[0] for p in pairs]); held=np.concatenate([p[1] for p in pairs])
        for region,sites in [('left',[0]),('interior',list(range(1,24))),('right',[24])]:
            a=bank[:,sites].reshape(-1,10); b=held[:,sites].reshape(-1,10)
            tree=cKDTree(a)
            bd=tree.query(b,workers=1)[0]
            baseline_path=root/f'baseline_{theta}_{region}.npz'
            np.savez_compressed(baseline_path,distance=bd)
            baselines.append(dict(theta=theta,region=region,bank_points=len(a),held_points=len(b),
                quantiles=np.quantile(bd,[0,.5,.9,.95,1]).tolist(),path=str(baseline_path),sha256=digest(baseline_path)))
            for si,state in enumerate(manifest['rows']):
                if state['state']['theta']!=theta: continue
                distance,index=tree.query(query[si,sites],workers=1)
                path=root/f'state{si}_{region}.npz'
                np.savez_compressed(path,distance=distance,nearest_index=index,descriptors=query[si,sites])
                rows.append(dict(state_id=si,state=state,region=region,path=str(path),sha256=digest(path),
                    quantiles=np.quantile(distance,[0,.5,.9,.95,1]).tolist(),
                    above_within_family_q95=float(np.mean(distance>np.quantile(bd,.95)))))
    result=dict(status='COMPLETE_DESCRIPTIVE_LOCAL_COVERAGE_NOT_CERTIFIED',rows=rows,baselines=baselines,
        training_sources=sources,manifest_sha256=digest(ROOT/'manifest.json'),
        code_sha256={p:digest(p) for p in [__file__,'tests/experiments/fine_transition_dataset.py']},
        scope='Nearest-neighbor Euclidean distance in ten unscaled dimensionless radius-one invariants: three spin dot products, six director projection products and signed triple product. Temperature and left/interior/right boundary type matched. Reference bank first 128 training noise paths; baseline other 128, all eight transition starts. Correlated windows/sites and repeated initial states are not independent samples; baseline is within training parent family, not a held-out family. No fitted threshold, no significance test, no full receptive-field or latent-flow support certificate. Distances can suggest scarcity but do not establish causal model failure.')
    save(root/'summary.json',result)
    for kind in ['aligned','tilted','wall']:
        subset=[r for r in rows if r['region']=='interior' and r['state']['state']['kind']==kind]
        for frame in [0,20,40]:
            group=[r for r in subset if r['state']['frame']==frame]
            print(kind,frame,'median distance averaged over conditions',np.mean([r['quantiles'][1] for r in group]),
                'fraction above within-family q95',np.mean([r['above_within_family_q95'] for r in group]))


if __name__=='__main__':
    run()
