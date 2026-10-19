"""Gate fixed-count mixed dataset before any coverage-intervention training."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest,save
from tests.experiments.coverage_intervention_data import ROOT,RAW,CONFIG
from tests.experiments.coverage_transition_dataset import CoverageTransitionDataset,replacement_index


def run():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert manifest['config_sha256']==digest(CONFIG)
    assert manifest['old_source_sha256']==digest('tests/results/fine_transition_pilot/summary.json')
    assert manifest['bank_sha256']==digest(RAW/'initials.npz')
    with np.load(RAW/'initials.npz') as z: initials=z['unit_initials']
    dataset=CoverageTransitionDataset()
    old_seeds=[]; new_seeds=[]
    oldreport=json.loads(Path('tests/results/fine_transition_pilot/summary.json').read_text())
    for row in oldreport['rows']:
        with np.load(row['path']) as z: old_seeds.extend(z['seeds'].tolist())
    for row in dataset.rows.values():
        assert digest(row['path'])==row['sha256']
        with np.load(row['path']) as z:
            seeds=z['seeds']; p=z['spins']
        new_seeds.extend(seeds.tolist())
        original=initials[row['state']['initial_id']]
        np.testing.assert_array_equal(p[:,0],np.repeat(original[None],32,axis=0))
    assert len(new_seeds)==768 and len(set(new_seeds))==768
    assert not set(new_seeds).intersection(old_seeds)
    counts={'kept':0,'new':0}; theta=[]; identities=set()
    for i in range(len(dataset)):
        item=dataset[i]; base=dataset.original[i]
        assert item['theta']==base['theta'] and item['lag']==base['lag']
        theta.append(item['theta'])
        if replacement_index(i) is None:
            for key in ['initial','target']: np.testing.assert_array_equal(item[key],base[key])
            counts['kept']+=1
        else: counts['new']+=1
        identity=(item['noise_seed'],item['start'])
        assert identity not in identities; identities.add(identity)
    assert counts=={'kept':6144,'new':6144}
    np.testing.assert_allclose([np.mean(theta),np.std(theta)],[.12,.01],atol=1e-14)
    save(ROOT/'verification.json',dict(status='VERIFIED_FIXED_COUNT_DATA_READY_FOR_CONTROLLED_TRAINING',
        manifest_sha256=digest(ROOT/'manifest.json'),summary_sha256=digest(ROOT/'summary.json'),
        code_sha256={p:digest(p) for p in [__file__,'tests/experiments/coverage_transition_dataset.py']},
        counts=counts,total_noise_paths=1536,unique_pairs=len(identities),
        frozen_latent_scale=manifest['frozen_control'][0]['latent_scale'],
        scope='All mixed pairs checked; retained half bit-identical, temperature order unchanged, unique noise-window identities. Ready for controlled training, not a learned-model or continuum certification.'))
    print('Verified 12288 unique pairs: 6144 original + 6144 new; temperature order unchanged.')


if __name__=='__main__': run()
