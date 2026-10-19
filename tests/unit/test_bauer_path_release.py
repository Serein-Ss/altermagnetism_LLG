"""Release migration regression, not physical certification."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
torch.set_num_threads(1)


@pytest.mark.parametrize('seed',[11,22,33])
def test_checkpoint_and_velocity_exact(seed):
    from scripts.inference.bauer_path import load_checkpoint
    from tests.chain_reversal_repair.model import ReversalGauge
    net,ck=load_checkpoint(ROOT/f'output/development/bauer_path/step130240/seed{seed}/checkpoint.pt')
    old=ReversalGauge('reversal');old.load_state_dict(ck['model']);old.eval()
    rng=torch.Generator().manual_seed(seed)
    x=F.normalize(torch.randn(1,2,16,3,generator=rng),dim=-1);a=x[:,0];t=x.new_tensor([.4]);times=x.new_tensor([0.,.1])
    assert ck['step']==130240
    assert net.get_extra_state()==old.get_extra_state()
    with torch.no_grad():
        assert torch.equal(net.velocity(x,t,a,times),old.velocity(x,t,a,times))


def test_sampling_exact_and_resumable():
    from scripts.inference.bauer_path import load_checkpoint,generate
    from tests.transition_block_repair.model import rollout
    net,_=load_checkpoint(ROOT/'output/development/bauer_path/step130240/seed11/checkpoint.pt')
    a=F.normalize(torch.randn(1,16,3,generator=torch.Generator().manual_seed(32)),dim=-1)
    x,state=generate(net,a,.2,40,flow_steps=2)
    old=rollout(net,a,torch.Generator().manual_seed(40),2,2)
    assert torch.equal(x,old)
    first,s=generate(net,a,.1,40,2)
    second,s2=generate(net,first[:,-1],.1,0,2,torch.Generator().set_state(s))
    assert torch.equal(x,torch.cat((first,second[:,1:]),1))
    assert torch.equal(state,s2)
    assert torch.equal(x[:,0],a)
    assert torch.allclose(x.norm(dim=-1),torch.ones_like(x[...,0]),atol=2e-6)
    with pytest.raises(ValueError):generate(net,a,.15,40,2)


def test_loss_and_teacher_exact():
    from scripts.training.bauer_path_objectives import spatial_moment_loss
    from tests.spatial_covariance_repair.objective import loss
    from scripts.training.bauer_path_teacher import labels
    from tests.conditional_teacher_repair.target import labels as old_labels
    from scripts.model.path_sampling import draw
    rng=torch.Generator().manual_seed(7)
    bank=F.normalize(torch.randn(64,2,16,3,generator=rng),dim=-1);bank[:,0]=bank[0,0].clone()
    y=bank[:4];z=draw(y[:,0],2,rng,0.);tau=torch.tensor([.2,.4,.6,.95])
    g=rng.get_state()
    new=labels(z,y,tau,bank,torch.Generator().set_state(g))
    old=old_labels(z,y,tau,bank,torch.Generator().set_state(g))
    for a,b in zip(new[:3],old[:3]):assert torch.equal(a,b)
    assert new[3]==old[3]
    a=spatial_moment_loss(bank[:32],bank[32:],True);b=loss(bank[:32],bank[32:],True)
    assert torch.equal(a[0],b[0]);assert a[1]==b[1]


@pytest.mark.parametrize('size',[16,25,32])
def test_lazy_windows_match_frozen(size):
    from scripts.datasets.bauer_path_windows import PathWindows
    from tests.mdgen_path_refactor_20260918.data import PathWindows as OldWindows
    path=ROOT/f'data/research/datasets/bauer_open_chain/native/L{size}'
    a=PathWindows([path],'train',1.,10);b=OldWindows([path],'train',1.,10)
    assert len(a)==len(b)
    for index in (0,len(a)-1):
        aa,bb=a[index],b[index]
        for key in aa:
            if isinstance(aa[key],np.ndarray):np.testing.assert_array_equal(aa[key],bb[key])
            else:assert aa[key]==bb[key]


def test_energy_and_metrics_exact():
    from scripts.analysis.bauer_path_metrics import statistics,compare_spatial
    from tests.transition_block_repair.metrics import statistics as old_statistics
    from tests.spatial_covariance_repair.metrics import compare as old_compare
    rng=np.random.default_rng(9);x=rng.normal(size=(16,2,16,3));x/=np.linalg.norm(x,axis=-1,keepdims=True);x[:,0]=x[0,0]
    for k,v in statistics(x).items():np.testing.assert_array_equal(v,old_statistics(x)[k])
    assert compare_spatial(x[:8],x[8:])==old_compare(x[:8],x[8:])


def test_canonical_imports_have_no_test_dependency():
    import subprocess,sys
    modules=['scripts.inference.bauer_path','scripts.training.train_bauer_path','scripts.analysis.bauer_path_metrics']
    code='import importlib,sys; [importlib.import_module(x) for x in '+repr(modules)+']; assert not any(x=="tests" or x.startswith("tests.") for x in sys.modules)'
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


def test_differentiable_sampler_and_energy():
    from scripts.inference.bauer_path import load_checkpoint
    from scripts.model.path_sampling import draw,differentiable_sample
    from tests.conditional_moment_repair.objectives import sample
    from scripts.analysis.bauer_energy import energy
    net,_=load_checkpoint(ROOT/'output/development/bauer_path/step130240/seed11/checkpoint.pt')
    rng=torch.Generator().manual_seed(24);a=F.normalize(torch.randn(1,16,3,generator=rng),dim=-1);z=draw(a,2,rng,0.)
    x=differentiable_sample(net,a,z,2,True);y=sample(net,a,z,2,True)
    assert torch.equal(x,y)
    energy(x).mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    assert sum(float(p.grad.square().sum()) for p in net.parameters() if p.grad is not None)>0


def test_schedules_and_native_batch_exact():
    from scripts.training.bauer_path_recipe import schedule,roles,rate
    from tests.conditional_coupling_repair.coupling import schedule as old_schedule
    from tests.model_state_refresh_repair.recipe import replay_roles,rate as old_rate
    from scripts.training.train_bauer_path import native_batch
    from tests.bauer_closed_loop_20260919.run import batch
    a,b=schedule(12,16);c,d=old_schedule(12,16)
    np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
    for step in (1,8,16,64,200,2048):
        np.testing.assert_array_equal(roles(step),replay_roles(step,'refreshed_replay'))
        assert np.isclose(rate(step,2048),old_rate(step,2048),rtol=1e-15)
    class Dummy:
        ends=[20];records=[(None,None,range(10),10)]
        def __len__(self):return 20
        def __getitem__(self,i):
            x=np.random.default_rng(i).normal(size=(11,16,3)).astype(np.float32)
            return dict(spins=x)
    ds=[Dummy()];cfg=dict(batch=8,save_ps=.1)
    x=native_batch(ds,np.random.default_rng(3),8,'cpu')
    y,_=batch(ds,np.random.default_rng(3),cfg,'cpu','B')
    assert torch.equal(x,y)
