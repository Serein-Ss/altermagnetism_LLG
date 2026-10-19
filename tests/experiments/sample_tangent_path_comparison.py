"""Whole-path diagnostic: fitted tangent baseline versus frozen flow candidates."""
import json
import os
import time
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest,save
from tests.experiments.sample_tangent_baseline import sample
from tests.experiments.independent_candidate_analysis import compare
from tests.experiments.verify_independent_candidate import verify_metrics
from tests.experiments.fresh_reference_audit import FRAMES,path_energy
from tests.experiments.fresh_model_audit import temporal_covariance


def rollout(initial,parameter,axis,paths,steps,seed):
    rng=np.random.default_rng(seed)
    x=np.repeat(initial[None],paths,axis=0)
    frames=[x.copy()]
    for _ in range(steps):
        x=sample(x,axis,np.asarray(parameter['coefficients']),parameter['sigma'],rng)
        frames.append(x.copy())
    return np.stack(frames,axis=1)


def run():
    if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('CPU Slurm allocation required')
    mp=Path('tests/output/sample_tangent_baseline/model.json');model=json.loads(mp.read_text())
    for p,h in {**model['sources'],**model['code_sha256']}.items():assert digest(p)==h
    refp=Path('tests/results/fresh_rollout_reference/summary.json')
    reference_report=json.loads(refp.read_text())
    assert reference_report['status']=='COMPLETE_FRESH_ROLLOUT_REFERENCE_NOT_CERTIFIED'
    manifest_path=refp.parent/'manifest.json'
    assert digest(manifest_path)==reference_report['manifest_sha256']
    manifest=json.loads(manifest_path.read_text())
    reports={}
    for seed in [11,22,33]:
        p=Path(f'tests/results/reflection_augmentation_training/{seed}/summary.json')
        r=json.loads(p.read_text())
        assert r['status']=='COMPLETE_REFLECTION_AUGMENTATION_TRAINING_AND_ROLLOUT_NOT_CERTIFIED'
        reports[seed]=(r,digest(p))
    root=Path('tests/results/sample_tangent_path_comparison')
    raw=Path('tests/data/sample_tangent_path_comparison')
    if root.exists() or raw.exists():raise FileExistsError('Preserve previous comparison')
    root.mkdir();raw.mkdir()
    result=dict(status='RUNNING',rows=[],job_id=os.environ['SLURM_JOB_ID'],model_sha256=digest(mp),
        reference_sha256=digest(refp),flow_report_sha256={s:h for s,(_,h) in reports.items()},
        code_sha256={p:digest(p) for p in [__file__,'tests/experiments/sample_tangent_baseline.py',
            'tests/experiments/independent_candidate_analysis.py','tests/experiments/verify_independent_candidate.py',
            'tests/experiments/fresh_reference_audit.py','tests/experiments/fresh_model_audit.py']},
        scope='Development-only diagnostic, no retraining or tuning. One fitted Gaussian tangent baseline and three frozen flow candidates, '
        'each against the same fine-MT reference. Independent whole-path bootstrap, pointwise intervals only. '
        'The baseline has stronger local structural assumptions, not supplied force coefficients. '
        'Timing excludes file I/O and fitting, and is not a matched-hardware speedup test. No full-path certificate.')
    for anchor,state in enumerate(manifest['anchors']):
        ref=next(r for r in reference_report['rows'] if r['case']==1 and r['anchor']==anchor)
        assert digest(ref['path'])==ref['sha256']
        with np.load(ref['path']) as z:reference=z['spins'].astype(float)
        assert reference.shape==(256,81,25,3)
        parameter=next(r for r in model['rows'] if r['theta']==state['theta'])
        started=time.monotonic()
        paths=rollout(reference[0,0],parameter,np.asarray(model['axis']),256,80,1800000000+anchor)
        seconds=time.monotonic()-started
        assert paths.shape==reference.shape and np.isfinite(paths).all()
        assert np.max(abs(np.linalg.norm(paths,axis=-1)-1))<1e-12
        path=raw/f'anchor{anchor}.npz'
        np.savez_compressed(path,spins=paths,reduced_time=np.arange(81)*.01)
        candidates=[('baseline',dict(path=str(path),sha256=digest(path)))]
        for seed,(report,_) in reports.items():
            candidates.append((f'flow{seed}',next(r for r in report['rows'] if r['anchor']==anchor)))
        for index,(label,item) in enumerate(candidates):
            assert digest(item['path'])==item['sha256']
            with np.load(item['path']) as z:p=z['spins'].astype(float)
            np.testing.assert_allclose(p[:,0],reference[:,0],atol=2e-7,rtol=0)
            arrays,metrics=compare(p,reference,np.random.default_rng(1801000000+anchor*10+index))
            terminal=verify_metrics(p,reference,metrics,arrays)
            for name,trajectory in [('candidate',p),('reference',reference)]:
                arrays[name+'_selected_covariance']=np.cov(trajectory[:,FRAMES].reshape(256,-1),rowvar=False)
                arrays[name+'_temporal_covariance']=temporal_covariance(trajectory)
            out=root/f'{label}_anchor{anchor}.npz';np.savez_compressed(out,**arrays)
            result['rows'].append(dict(label=label,anchor=anchor,state=state,path=str(out),sha256=digest(out),
                candidate=item,reference=ref,metrics=metrics,terminal=terminal,
                joint_energy=path_energy(p[:,FRAMES],reference[:,FRAMES]),
                baseline_generation_seconds=seconds if label=='baseline' else None))
            save(root/'summary.json',result)
    assert len(result['rows'])==48
    result['status']='COMPLETE_RAW_POINTS_CHECKED_NOT_CERTIFIED';save(root/'summary.json',result)


if __name__=='__main__':run()
