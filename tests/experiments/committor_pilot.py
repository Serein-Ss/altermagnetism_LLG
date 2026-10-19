"""Bounded, sample-grounded basin/branch diagnostic; no infinite-time claims."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.independent_reference import generate

CONF=Path('tests/conf/committor_pilot.yaml')
RAW=Path('tests/data/committor_pilot')
OUT=Path('tests/results/committor_pilot')


def prepare():
    cfg=yaml.safe_load(CONF.read_text())
    oldpath=Path('tests/results/independent_candidate_reference/manifest.json')
    old=json.loads(oldpath.read_text())
    refpath=oldpath.with_name('summary.json'); refs=json.loads(refpath.read_text())
    assert refs['manifest_sha256']==digest(oldpath)
    rng=np.random.default_rng(cfg['selection_seed']); rows=[]; initials=[]
    for theta in cfg['theta']:
        for kind,count in cfg['shooting']['kinds'].items():
            candidates=[i for i,a in enumerate(old['anchors']) if a['theta']==theta and a['kind']==kind]
            for anchor in rng.choice(candidates,size=count,replace=False):
                r=next(r for r in refs['rows'] if r['case']==1 and r['anchor']==anchor)
                assert digest(r['path'])==r['sha256']
                trajectory=int(rng.integers(512)); frame=int(rng.choice(cfg['shooting']['source_frames']))
                with np.load(r['path']) as z:
                    x=z['spins'][trajectory,frame].copy()
                    parent_noise=int(z['seeds'][trajectory])
                initials.append(x)
                rows.append(dict(theta=theta,kind=kind,anchor=int(anchor),trajectory=trajectory,frame=frame,
                    parent_noise=parent_noise,source=r['path'],source_sha256=r['sha256'],state=old['anchors'][anchor]))
    assert len(rows)==12
    planned=[]
    c=cfg['calibration'];s=cfg['shooting']
    for i in range(4):planned.extend((c['seed_base']+i*10000+np.arange(c['paths'])).tolist())
    for i in range(24):planned.extend((s['seed_base']+i*10000+np.arange(s['paths'])).tolist())
    previous_noise=[v for r in refs['rows'] for v in r['numerics']['noise_seeds']]
    assert len(set(planned))==len(planned) and not set(planned).intersection(previous_noise)
    if RAW.exists() or OUT.exists(): raise FileExistsError('Preserve existing pilot')
    RAW.mkdir(parents=True);OUT.mkdir(parents=True)
    np.savez_compressed(RAW/'states.npz',spins=np.asarray(initials))
    baseline=Path('tests/output/sample_tangent_baseline/model.json')
    model=json.loads(baseline.read_text())
    for p,h in {**model['sources'],**model['code_sha256']}.items(): assert digest(p)==h
    for r in old['frozen_models']: assert digest(r['checkpoint'])==r['sha256']
    sources=['tests/experiments/committor_pilot.py','tests/experiments/committor_pilot_analysis.py',
        'tests/experiments/independent_reference.py','scripts/core/bauer_fast.py',
        'tests/experiments/director_midpoint_sampling.py','tests/experiments/projected_midpoint_sampler.py',
        'tests/experiments/director_transition.py','tests/experiments/local_equivariant_transition.py',
        'scripts/model/gate_f.py','tests/experiments/sample_tangent_baseline.py',
        'tests/experiments/sample_tangent_path_comparison.py']
    save(OUT/'manifest.json',dict(config=cfg,config_sha256=digest(CONF),rows=rows,
        states_sha256=digest(RAW/'states.npz'),baseline=str(baseline),baseline_sha256=digest(baseline),
        frozen_models=old['frozen_models'],source_report_sha256=digest(refpath),
        code_sha256={p:digest(p) for p in sources},status='PREPARED_DEVELOPMENT_ONLY',
        noise_audit=dict(new_reference_seeds=len(planned),overlap_with_independent_reference=0,
            scope='Audited named independent reference bank only; distinct seeds do not mathematically prove PRNG independence.')))


def smoke(device):
    import torch
    from tests.experiments.director_transition import DirectorTransition
    from tests.experiments.director_midpoint_sampling import rollout
    m,cfg=load()
    if device!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA allocation required')
    with np.load(RAW/'states.npz') as z:x=z['spins'][0]
    checks=[]
    for frozen in m['frozen_models']:
        assert digest(frozen['checkpoint'])==frozen['sha256']
        ck=torch.load(frozen['checkpoint'],map_location='cpu',weights_only=False)
        tc=dict(ck['config']);tc.update(rollout_steps=2,generated_paths=4,generation_batch=4)
        model=DirectorTransition(ck['theta_mean'],ck['theta_std'],tc['width']).to(device)
        model.load_state_dict(ck['ema'],strict=True);model.eval()
        p,_=rollout(model,x,m['rows'][0]['theta'],tc,ck['latent_scale'],32,2190000000+frozen['seed'],device)
        assert p.shape==(4,3,25,3) and np.isfinite(p).all()
        assert np.max(abs(np.linalg.norm(p,axis=-1)-1))<1e-5
        checks.append(frozen['seed'])
    p,_=generate(np.repeat(x[None],2,axis=0),np.array([2189000000,2189000001]),dt=.00125,
        duration=.02,save_dt=.01,anisotropy=.1,damping=.1,theta=m['rows'][0]['theta'],method='mt')
    assert p.shape==(2,3,25,3)
    save(OUT/'gpu_smoke.json',dict(status='PASS',seeds=checks,manifest_sha256=digest(OUT/'manifest.json')))


def load():
    m=json.loads((OUT/'manifest.json').read_text())
    assert digest(CONF)==m['config_sha256'] and digest(RAW/'states.npz')==m['states_sha256']
    for p,h in m['code_sha256'].items(): assert digest(p)==h
    return m,m['config']


def persist(name,paths,times,meta,started,seeds=None):
    if not np.isfinite(paths).all(): raise ValueError('numerical failure, not censoring')
    assert np.max(abs(np.linalg.norm(paths,axis=-1)-1))<1e-5
    p=RAW/(name+'.npz')
    if p.exists(): raise FileExistsError(p)
    fields=dict(spins=paths,reduced_time=times)
    if seeds is not None: fields['seeds']=seeds
    np.savez_compressed(p,**fields)
    save(OUT/(name+'.json'),dict(status='COMPLETE_NUMERICAL_ONLY',path=str(p),sha256=digest(p),
        metadata=meta,seconds_including_save=time.monotonic()-started,
        manifest_sha256=digest(OUT/'manifest.json'),job_id=os.environ.get('SLURM_JOB_ID')))


def calibration(index):
    _,cfg=load(); c=cfg['calibration']; ti,ci=divmod(index,2)
    n=c['paths']; x=np.zeros((n,cfg['sites'],3));x[:n//2,:,2]=1;x[n//2:,:,2]=-1
    seeds=c['seed_base']+index*10000+np.arange(n); start=time.monotonic()
    p,num=generate(x,seeds,**c['cases'][ci],duration=c['duration'],save_dt=c['save_dt'],
        anisotropy=cfg['anisotropy'],damping=cfg['damping'],theta=cfg['theta'][ti])
    persist(f'calibration_{index}',p,np.arange(p.shape[1])*c['save_dt'],num,start,seeds)


def branch(arm,index,device):
    m,cfg=load(); c=cfg['shooting']; n=c['paths']; steps=round(c['duration']/c['save_dt'])
    with np.load(RAW/'states.npz') as z: states=z['spins']
    model=None; checkpoint=None
    if arm=='flow':
        import torch
        from tests.experiments.director_transition import DirectorTransition
        if device!='cuda' or not torch.cuda.is_available(): raise RuntimeError('Slurm CUDA required')
        seed=c['model_seeds'][index]; frozen=next(r for r in m['frozen_models'] if r['seed']==seed)
        assert digest(frozen['checkpoint'])==frozen['sha256']
        previous=json.loads(Path(frozen['report']).read_text())
        assert digest(frozen['report'])==frozen['report_sha256']
        for p,h in previous['code_sha256'].items(): assert digest(p)==h
        checkpoint=torch.load(frozen['checkpoint'],map_location='cpu',weights_only=False)
        model=DirectorTransition(checkpoint['theta_mean'],checkpoint['theta_std'],checkpoint['config']['width']).to(device)
        model.load_state_dict(checkpoint['ema'],strict=True);model.eval()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if arm=='baseline':
        assert digest(m['baseline'])==m['baseline_sha256']
        baseline=json.loads(Path(m['baseline']).read_text())
    for ai,a in enumerate(m['rows']):
        start=time.monotonic(); seeds=None
        if arm=='reference':
            seeds=c['seed_base']+(index*len(states)+ai)*10000+np.arange(n)
            p,meta=generate(np.repeat(states[ai:ai+1],n,axis=0),seeds,**c['cases'][index],
                duration=c['duration'],save_dt=c['save_dt'],anisotropy=cfg['anisotropy'],
                damping=cfg['damping'],theta=a['theta'])
            name=f'reference{index}_state{ai}'
        elif arm=='baseline':
            from tests.experiments.sample_tangent_path_comparison import rollout
            assert baseline['lag']==c['save_dt']
            param=next(r for r in baseline['rows'] if r['theta']==a['theta'])
            latent=c['baseline_seed_base']+ai
            p=rollout(states[ai],param,np.asarray(baseline['axis']),n,steps,latent)
            meta=dict(latent_seed=latent,model_sha256=m['baseline_sha256']);name=f'baseline_state{ai}'
        else:
            from tests.experiments.director_midpoint_sampling import rollout
            tc=dict(checkpoint['config']);assert tc['lag']==c['save_dt']
            tc.update(rollout_steps=steps,generated_paths=n,generation_batch=c['generation_batch'])
            latent=c['flow_seed_base']+seed*1000+ai
            torch.cuda.synchronize()
            p,_=rollout(model,states[ai],a['theta'],tc,checkpoint['latent_scale'],c['flow_steps'],latent,device)
            torch.cuda.synchronize()
            meta=dict(latent_seed=latent,model_sha256=frozen['sha256']);name=f'flow{seed}_state{ai}'
        assert p.shape==(n,steps+1,cfg['sites'],3)
        np.testing.assert_allclose(p[:,0],np.broadcast_to(states[ai],p[:,0].shape),atol=2e-7,rtol=0)
        persist(name,p,np.arange(steps+1)*c['save_dt'],meta,start,seeds)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','calibration','reference','baseline','flow','smoke'])
    p.add_argument('--index',type=int,default=0);p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:
        if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
        if a.stage=='smoke':smoke(a.device)
        elif a.stage=='calibration':calibration(a.index)
        else:branch(a.stage,a.index,a.device)
