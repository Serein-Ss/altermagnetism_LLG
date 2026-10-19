"""Frozen learned-flow reflection audit on actual conditional bridge states."""
import json
from pathlib import Path
import numpy as np
import torch
from scripts.workflow.bauer_campaign import digest,save
from scripts.model.gate_f import path_interpolate
from tests.experiments.director_transition import DirectorTransition
from tests.experiments.local_equivariant_transition import source


def reflected_field(model,x,tau,initial,theta,lag,axis):
    return model(x.flip(1),tau,initial.flip(1),theta,lag,axis).flip(1)


def run():
    torch.set_num_threads(1)
    rp=Path('tests/results/conditional_restart/reference.json')
    ref=json.loads(rp.read_text()); assert ref['status']=='COMPLETE_CONDITIONAL_REFERENCE_NOT_CERTIFIED'
    mp=rp.with_name('manifest.json'); manifest=json.loads(mp.read_text())
    assert digest(mp)==ref['manifest_sha256']
    initial=[]; target=[]; temperatures=[]
    for state in range(60):
        row=next(r for r in ref['rows'] if r['case']==1 and r['state_id']==state)
        assert digest(row['path'])==row['sha256']
        with np.load(row['path']) as z: p=z['spins'][:4].copy()
        initial.extend(p[:,0]); target.extend(p[:,1])
        temperatures.extend([manifest['rows'][state]['state']['theta']]*4)
    initial=torch.nn.functional.normalize(torch.tensor(np.asarray(initial),dtype=torch.float32),dim=-1)
    target=torch.nn.functional.normalize(torch.tensor(np.asarray(target),dtype=torch.float32),dim=-1)
    theta=torch.tensor(temperatures,dtype=torch.float32); lag=theta.new_full((240,),.01)
    axis=theta.new_tensor([0.,0.,1.]).expand(240,-1)
    root=Path('tests/results/chain_reflection_audit'); root.mkdir(exist_ok=False)
    rows=[]
    for arm in ['control','coverage']:
        for seed in [11,22,33]:
            report_path=Path(f'tests/results/director_budget/24000/{seed}/summary.json' if arm=='control' else f'tests/results/coverage_intervention_training/{seed}/summary.json')
            report=json.loads(report_path.read_text())
            assert report['status'] in ['COMPLETE_TRAINING_AND_GENERATION_NOT_CERTIFIED','COMPLETE_COVERAGE_TRAINING_AND_ROLLOUT_NOT_CERTIFIED']
            assert digest(report['checkpoint'])==report['checkpoint_sha256']
            ck=torch.load(report['checkpoint'],map_location='cpu',weights_only=False)
            model=DirectorTransition(ck['theta_mean'],ck['theta_std'],ck['config']['width'])
            model.load_state_dict(ck['ema'],strict=True); model.eval()
            rng=torch.Generator().manual_seed(1001000000)
            z=source(initial,rng,ck['latent_scale'])
            values=[]; mirrors=[]
            with torch.no_grad():
                for time in [.1,.5,.9]:
                    tau=theta.new_full((240,),time)
                    x,_,_=path_interpolate(torch.stack([initial,z],1),torch.stack([initial,target],1),tau,rng)
                    v=model(x[:,1],tau,initial,theta,lag,axis)
                    mirrored=reflected_field(model,x[:,1],tau,initial,theta,lag,axis)
                    values.append(v.numpy().reshape(60,4,25,3)); mirrors.append(mirrored.numpy().reshape(60,4,25,3))
            values=np.asarray(values); mirrors=np.asarray(mirrors)
            assert np.isfinite(values).all() and np.isfinite(mirrors).all()
            path=root/f'{arm}_seed{seed}.npz'
            np.savez_compressed(path,field=values,reflected_field=mirrors,tau=[.1,.5,.9])
            for state,metadata in enumerate(manifest['rows']):
                metrics={}
                for region,ids in [('all',np.arange(25)),('ends',np.r_[0:3,22:25]),('interior',np.arange(3,22))]:
                    v=values[:,state][:,:,ids]; mv=mirrors[:,state][:,:,ids]
                    metrics[region]=dict(reflection_rmse=float(np.sqrt(np.mean((v-mv)**2))),
                        field_rms=float(np.sqrt(np.mean((v*v+mv*mv)/2))))
                rows.append(dict(arm=arm,seed=seed,state_id=state,state=metadata,metrics=metrics,path=str(path),sha256=digest(path),checkpoint_sha256=report['checkpoint_sha256']))
    save(root/'summary.json',dict(status='COMPLETE_FROZEN_FLOW_REFLECTION_AUDIT_NOT_CERTIFIED',rows=rows,
        reference_sha256=digest(rp),code_sha256={p:digest(p) for p in [__file__,'tests/experiments/director_transition.py','tests/experiments/local_equivariant_transition.py','scripts/model/gate_f.py']},
        scope='Chain index reversal only; spin components and anisotropy axis unchanged. Four fixed true target paths per condition, common source noise, three flow times. Learned transport-field symmetry diagnostic, not comparison to physical LLG drift or proof of path-law error. Exact reference symmetry tested separately with explicitly reversed noise arrays. No significance or pass threshold.'))
    for arm in ['control','coverage']:
        for kind in ['aligned','tilted','wall']:
            selected=[r for r in rows if r['arm']==arm and r['state']['state']['kind']==kind]
            print(arm,kind,{region:{k:float(np.mean([r['metrics'][region][k] for r in selected])) for k in ['reflection_rmse','field_rms']} for region in ['ends','interior']})


if __name__=='__main__': run()
