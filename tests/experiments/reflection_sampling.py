"""Same checkpoints and latent paths, only impose chain reflection on flow."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from scripts.workflow.bauer_campaign import digest,save
from tests.experiments.director_transition import DirectorTransition
from tests.experiments.director_midpoint_sampling import rollout
from tests.experiments.reflection_projection import ReflectionProjection


def run(seed,device):
    if device!='cuda' or not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm CUDA allocation required')
    source=Path(f'tests/results/coverage_intervention_training/{seed}/summary.json')
    control=json.loads(source.read_text())
    assert control['status']=='COMPLETE_COVERAGE_TRAINING_AND_ROLLOUT_NOT_CERTIFIED'
    assert digest(control['checkpoint'])==control['checkpoint_sha256']
    for p,h in control['code_sha256'].items(): assert digest(p)==h
    cfg=control['rollout_config']
    assert cfg['generated_paths']==256 and cfg['rollout_steps']==80 and cfg['flow_steps']==32
    mp=Path('tests/results/fresh_rollout_reference/manifest.json'); m=json.loads(mp.read_text())
    assert digest(mp)==control['reference_manifest_sha256']
    bank=Path('tests/data/fresh_rollout_reference/anchors.npz'); assert digest(bank)==m['bank_sha256']
    with np.load(bank) as z: states=z['unit_initials']
    ck=torch.load(control['checkpoint'],map_location='cpu',weights_only=False)
    base=DirectorTransition(ck['theta_mean'],ck['theta_std'],cfg['width']).to(device)
    base.load_state_dict(ck['ema'],strict=True)
    model=ReflectionProjection(base).eval()
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    root=Path(f'tests/results/reflection_sampling/{seed}'); raw=Path(f'tests/data/reflection_sampling/{seed}')
    if root.exists() or raw.exists(): raise FileExistsError('preserve previous sampling')
    root.mkdir(parents=True); raw.mkdir(parents=True)
    report=dict(status='GENERATING',seed=seed,rows=[],config=cfg,source_sha256=digest(source),checkpoint_sha256=control['checkpoint_sha256'],
        job_id=os.environ['SLURM_JOB_ID'],code_sha256={p:digest(p) for p in [__file__,'tests/experiments/reflection_projection.py',
            'tests/experiments/director_midpoint_sampling.py','tests/experiments/projected_midpoint_sampler.py']},
        scope='Inference-only reflection group projection, no retraining. Same latent seeds and batch layout, midpoint32 and duration .8. Twice base director evaluations per flow call. Development comparison, no automatic physical certification.')
    save(root/'summary.json',report)
    for row in control['rows']:
        assert digest(row['path'])==row['sha256']
        a=row['state']; anchor=row['anchor']
        paths,rng=rollout(model,states[a['initial_id']],a['theta'],cfg,ck['latent_scale'],32,row['latent_seed'],device)
        assert paths.shape==(256,81,25,3) and np.isfinite(paths).all()
        assert np.max(abs(np.linalg.norm(paths,axis=-1)-1))<1e-5
        with np.load(row['path']) as z: np.testing.assert_array_equal(rng,z['rng_state'])
        path=raw/f'anchor{anchor}.npz'
        np.savez_compressed(path,spins=paths,rng_state=rng,reduced_time=np.arange(81)*cfg['lag'])
        report['rows'].append(dict(anchor=anchor,state=a,path=str(path),sha256=digest(path),latent_seed=row['latent_seed']))
        save(root/'summary.json',report)
    report['status']='COMPLETE_REFLECTION_SAMPLING_NOT_CERTIFIED'; save(root/'summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,choices=[11,22,33],required=True); p.add_argument('--device',required=True)
    a=p.parse_args(); run(a.seed,a.device)
