"""Fixed-data reflection augmentation, with unchanged inference and optimizer."""
import argparse
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from scripts.workflow.bauer_campaign import digest,save
from tests.experiments.coverage_transition_dataset import CoverageTransitionDataset
from tests.experiments.director_budget import training_arrays,config_for_budget
from tests.experiments.director_transition import DirectorTransition
from tests.experiments.director_midpoint_sampling import rollout
from tests.experiments.fresh_rollout_sampling import rollout_config
from tests.experiments.learned_tangent_source import LearnedTangentSource


def run(seed,device):
    if device!='cuda' or not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm CUDA allocation required')
    plan_path=Path('tests/conf/learned_source_plan.json')
    plan=json.loads(plan_path.read_text())
    assert seed in plan['training_seeds'] and plan['augmentation_probability']==.5
    prior_path=Path(plan['source_model'])
    prior_report=json.loads(prior_path.read_text())
    for p,h in {**prior_report['sources'],**prior_report['code_sha256']}.items(): assert digest(p)==h
    prior=LearnedTangentSource(prior_report).to(device)
    augmented_train_step=prior.train_step
    source=Path('tests/results/coverage_intervention_data')
    manifest=json.loads((source/'manifest.json').read_text())
    verification=json.loads((source/'verification.json').read_text())
    assert verification['status']=='VERIFIED_FIXED_COUNT_DATA_READY_FOR_CONTROLLED_TRAINING'
    assert digest(source/'manifest.json')==verification['manifest_sha256']
    assert digest(source/'summary.json')==verification['summary_sha256']
    for p,h in verification['code_sha256'].items(): assert digest(p)==h
    control=next(r for r in manifest['frozen_control'] if r['seed']==seed)
    assert digest(control['checkpoint'])==control['checkpoint_sha256']
    assert digest(control['report'])==control['report_sha256']
    old=json.loads(Path(control['report']).read_text())
    for p,h in old['code_sha256'].items(): assert digest(p)==h
    cfg=config_for_budget(24000); assert cfg==old['config']
    initial,target,theta,unused_scale=training_arrays(CoverageTransitionDataset())
    assert len(initial)==12288
    mean,std=float(theta.mean()),float(theta.std())
    assert mean==old['theta_mean'] and std==old['theta_std']
    scale=old['latent_scale']; assert scale==verification['frozen_latent_scale']
    root=Path(f'tests/results/learned_source_training/{seed}')
    output=Path(f'tests/output/learned_source/{seed}')
    raw=Path(f'tests/data/learned_source_rollout/{seed}')
    if any(p.exists() for p in [root,output,raw]): raise FileExistsError('preserve prior training')
    for p in [root,output,raw]: p.mkdir(parents=True)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model=DirectorTransition(mean,std,cfg['width']).to(device); ema=copy.deepcopy(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg['updates'],eta_min=cfg['final_learning_rate'])
    initial,target,theta=[torch.as_tensor(a,device=device,dtype=torch.float32) for a in [initial,target,theta]]
    initial=torch.nn.functional.normalize(initial,dim=-1); target=torch.nn.functional.normalize(target,dim=-1)
    rng=torch.Generator(device=device).manual_seed(950000000+seed)
    augmentation_rng=torch.Generator(device=device).manual_seed(plan['augmentation_rng_seed_base']+seed)
    reflected_count=0
    axis=torch.tensor(cfg['axis'],device=device).expand(cfg['batch_size'],-1)
    lag=theta.new_full((cfg['batch_size'],),cfg['lag'])
    report=dict(status='TRAINING',seed=seed,plan_sha256=digest(plan_path),source_model_sha256=digest(prior_path),config=cfg,history=[],rows=[],job_id=os.environ['SLURM_JOB_ID'],
        control_sha256=control['checkpoint_sha256'],verification_sha256=digest(source/'verification.json'),
        theta_mean=mean,theta_std=std,latent_scale=scale,mixed_scale_not_used=unused_scale,
        code_sha256={p:digest(p) for p in [__file__,'tests/experiments/learned_tangent_source.py','tests/experiments/director_budget.py',
            'tests/experiments/coverage_transition_dataset.py','tests/experiments/director_transition.py',
            'tests/experiments/local_equivariant_transition.py','tests/experiments/director_midpoint_sampling.py',
            'tests/experiments/projected_midpoint_sampler.py','tests/experiments/fresh_rollout_sampling.py']},
        scope='Sample-fitted conditional source intervention; reflection augmentation, backbone and optimization unchanged. Source sampling changes identically in training and inference. All control hyperparameters and latent scale frozen. '
        'Diagnostic fresh bank is development after using its results for design, not independent final validation.')
    save(root/'summary.json',report); started=time.monotonic()
    for step in range(1,cfg['updates']+1):
        ids=torch.randint(len(initial),(cfg['batch_size'],),device=device,generator=rng)
        value,grad,diagnostics,count=augmented_train_step(model,optimizer,initial[ids],target[ids],theta[ids],lag,axis,rng,scale,augmentation_rng)
        reflected_count+=count
        scheduler.step()
        if step==6000:
            state=torch.load(old['training_rng_at6000'],map_location='cpu',weights_only=True)
            assert torch.equal(rng.get_state().cpu(),state.cpu()),'control random-stream mismatch'
        with torch.no_grad():
            for ep,p in zip(ema.parameters(),model.parameters()): ep.lerp_(p,1-cfg['ema_decay'])
        if step==1 or step%100==0:
            report['history'].append(dict(step=step,loss=value,grad_norm=grad,diagnostics=diagnostics,reflected_count=reflected_count))
            save(root/'summary.json',report)
    torch.cuda.synchronize(); report['training_seconds']=time.monotonic()-started
    previous=torch.load(control['checkpoint'],map_location='cpu',weights_only=False)
    assert torch.equal(rng.get_state().cpu(),previous['rng'].cpu()),'final control random-stream mismatch'
    checkpoint=output/'final.pt'
    torch.save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),rng=rng.get_state(),augmentation_rng=augmentation_rng.get_state(),config=cfg,step=cfg['updates'],
        theta_mean=mean,theta_std=std,latent_scale=scale,source_model=prior_report,
        source_model_sha256=digest(prior_path)),checkpoint)
    report.update(status='GENERATING',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),control_rng_matched=True)
    save(root/'summary.json',report)
    ref=Path('tests/results/fresh_rollout_reference/manifest.json'); refm=json.loads(ref.read_text())
    assert digest('tests/data/fresh_rollout_reference/anchors.npz')==refm['bank_sha256']
    with np.load('tests/data/fresh_rollout_reference/anchors.npz') as z: states=z['unit_initials']
    cp=Path(f'tests/results/reflection_augmentation_training/{seed}/summary.json'); cr=json.loads(cp.read_text())
    assert cr['status']=='COMPLETE_REFLECTION_AUGMENTATION_TRAINING_AND_ROLLOUT_NOT_CERTIFIED'
    assert cr['control_sha256']==control['checkpoint_sha256'] and cr['reference_manifest_sha256']==digest(ref)
    assert cr['config']==cfg
    samplecfg=rollout_config(cfg,refm['config'],{'generation_batch':64})
    report.update(rollout_config=samplecfg,rollout_control_sha256=digest(cp),reference_manifest_sha256=digest(ref))
    ema.eval()
    for anchor,a in enumerate(refm['anchors']):
        oldrow=next(r for r in cr['rows'] if r['anchor']==anchor)
        assert digest(oldrow['path'])==oldrow['sha256']
        paths,state=rollout(ema,states[a['initial_id']],a['theta'],samplecfg,scale,32,oldrow['latent_seed'],device,sampler=prior.sample)
        assert paths.shape==(256,81,25,3) and np.isfinite(paths).all()
        assert np.max(abs(np.linalg.norm(paths,axis=-1)-1))<1e-5
        with np.load(oldrow['path']) as z: np.testing.assert_array_equal(state,z['rng_state'])
        path=raw/f'anchor{anchor}.npz'
        np.savez_compressed(path,spins=paths,rng_state=state,reduced_time=np.arange(81)*cfg['lag'])
        report['rows'].append(dict(anchor=anchor,state=a,path=str(path),sha256=digest(path),latent_seed=oldrow['latent_seed']))
        save(root/'summary.json',report)
    report['status']='COMPLETE_LEARNED_SOURCE_TRAINING_AND_ROLLOUT_NOT_CERTIFIED'; save(root/'summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True,choices=[11,22,33]); p.add_argument('--device',required=True)
    a=p.parse_args(); run(a.seed,a.device)
