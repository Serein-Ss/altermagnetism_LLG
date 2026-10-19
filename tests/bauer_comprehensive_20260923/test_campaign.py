import numpy as np
import pytest
import torch
from .run import tasks, config_for, next_batch, train_step
from .components import make_model, synthetic_parents
from .report import finite_sample


@pytest.mark.parametrize('domain',['deterministic','gaussian','mixture'])
def test_training_smoke_and_common_stream(domain):
    torch.set_num_threads(1);data=synthetic_parents('train');streams=[]
    for arm in ('baseline','temporal','global_temporal','source','temporal_source_long'):
        task=dict(domain=domain,arm=arm,seed=11);config=config_for(task)
        torch.manual_seed(11);net=make_model(config,arm)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
        rng=np.random.default_rng(11);g=torch.Generator().manual_seed(1010);tg=torch.Generator().manual_seed(923641)
        y,t=next_batch(task,config,data,rng,tg,'cpu')
        info,noise,tau=train_step(net,optimizer,y,t,g,arm)
        assert np.isfinite(info['loss']) and np.isfinite(info['gradient_norm'])
        assert any(p.grad is not None and p.grad.abs().max()>0 for p in net.parameters())
        streams.append((y,noise,tau))
    for stream in streams[1:]:
        for a,b in zip(streams[0],stream):torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_task_balance_and_long_controls():
    rows=tasks();assert len(rows)==60
    assert len({(r['domain'],r['arm'],r['seed']) for r in rows})==len(rows)
    for row in rows:
        group=[r for r in rows if (r['domain'],r['arm'])==(row['domain'],row['arm'])]
        assert {r['seed'] for r in group}=={11,22,33}
        if row['arm'].endswith('_long'):assert row['updates']==24000


def test_finite_sample_zero_error_control():
    ref=np.zeros((16,11,5,3));ref[...,2]=1
    result=finite_sample(ref,ref,7)
    assert all(v['mean']==0 for v in result['reference_reference'].values())
    assert all(v['mean']==0 for v in result['model_reference'].values())
