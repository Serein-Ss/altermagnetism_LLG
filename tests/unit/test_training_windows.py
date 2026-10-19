import torch
import json
import numpy as np
import pytest
from tests.experiments.bauer_window import crop_training_windows,sample_chunks
from tests.experiments.velocity_controls import CartesianPathNet


def test_crops_preserve_parent_and_spacing_full_window_preserves_rng():
    parents=torch.arange(100)[:,None]*100+torch.arange(21)[None]
    gen=torch.Generator().manual_seed(4)
    windows=crop_training_windows(parents,11,gen)
    assert torch.equal(windows//100,torch.arange(100)[:,None].expand(-1,11))
    assert torch.all(windows[:,1:]-windows[:,:-1]==1)
    assert windows[:,0].remainder(100).min()==0
    assert windows[:,0].remainder(100).max()==10
    state=gen.get_state().clone()
    assert torch.equal(crop_training_windows(parents,21,gen),parents)
    assert torch.equal(gen.get_state(),state)


def test_short_model_rollout_shape_and_unit_norm():
    torch.manual_seed(32)
    net=CartesianPathNet(width=8,blocks=1)
    initial=torch.randn(2,1,5,1,3);initial/=initial.norm(dim=-1,keepdim=True)
    time=torch.linspace(0,.05,11)[None].expand(2,-1)
    pred=sample_chunks(net,initial,torch.tensor([[.11,.1],[.13,.1]]),time,None,None,
                       torch.Generator().manual_seed(73),chunks=2,steps=16)
    assert pred.shape==(2,21,1,5,1,3)
    assert torch.equal(pred[:,0],initial)
    assert torch.isfinite(pred).all()
    assert (pred.norm(dim=-1)-1).abs().max()<1e-6


@pytest.mark.parametrize('frames',[11,21])
@pytest.mark.parametrize('source_scale',[None,1.])
@pytest.mark.parametrize('stable',[False,True])
def test_window_trainer_end_to_end(tmp_path,frames,source_scale,stable):
    from tests.experiments.bauer_budget_comparison import run
    torch.set_num_threads(1)
    rng=np.random.default_rng(9)
    y=rng.normal(size=(4,21,5,3)).astype('float32');y/=np.linalg.norm(y,axis=-1,keepdims=True)
    y[:,0]=np.array([0.,0.,1.],dtype='float32')
    condition=np.tile(np.array([.13,.1],dtype='float32'),(4,1))
    # Deliberate reuse is a software fixture only, never a scientific split.
    np.savez(tmp_path/'data.npz',train=y,condition=condition,development=y[:2],development_condition=condition[:2])
    base=dict(frames=21,L=5,width=8,blocks=1,batch_size=2,learning_rate=.001,theta=[.13],save_dt=80.)
    (tmp_path/'base.json').write_text(json.dumps(base))
    cfg=dict(base_config=str(tmp_path/'base.json'),data=str(tmp_path/'data.npz'),budgets=[1],seeds=[11],
             model='cartesian',window_frames=frames,source_scale=source_scale,compare_legacy=False,loss_draws=1,
             initials=1,latents=2,sampling_steps=1,threshold=.6,dwell=160.,
             **{key:str(tmp_path/key) for key in ('output','raw','analysis')})
    if stable:cfg['optimization']=dict(min_lr=.00001,ema_decay=.5)
    (tmp_path/'config.json').write_text(json.dumps(cfg));run(tmp_path/'config.json',11,'cpu')
    report=json.loads((tmp_path/'analysis/11/summary.json').read_text())
    assert report['status']=='COMPLETE_EXPLORATORY_NOT_CERTIFIED'
    assert report['evaluation'][0]['loss']['development']['window_frames']==frames
    assert np.load(tmp_path/'raw/11/step_1_T0.13.npy').shape==(2,21,5,3)
    checkpoint=torch.load(tmp_path/'output/11/step_1.pt',weights_only=False)
    assert checkpoint['optimizer_weights']=='raw_model'
    assert checkpoint['inference_weights']==('ema' if stable else 'raw_model')
    if stable:
        assert any(not torch.equal(v,checkpoint['raw_model'][k]) for k,v in checkpoint['model'].items())
