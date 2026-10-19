import torch
from .run import tasks,weight,UPDATES,gradient_relation,flat_gradient


def test_matched_weight_and_warmup():
    points=range(16,UPDATES+1,16)
    assert abs(sum(weight('fixed32',s) for s in points)-sum(weight('ramp32',s) for s in points))<1e-12
    assert all(weight('ramp32',s)==0 for s in range(1,501))
    assert weight('ramp32',1000)==.05 and weight('ramp32',1500)==.1
    assert all(weight('ramp32',s)==weight('ramp64',s) for s in points)
    assert all(weight('fm',s)==0 for s in points)


def test_gradient_relation_and_unused_parameters():
    net=torch.nn.Linear(2,1)
    grad=flat_gradient(net.weight.square().sum(),net)
    assert grad.numel()==3 and grad[-1]==0
    assert abs(gradient_relation(grad,-grad)['cosine']+1)<1e-6
    assert gradient_relation(grad,torch.zeros_like(grad))['cosine'] is None


def test_three_seed_complete_blocks():
    rows=tasks();assert len(rows)==12
    for seed in (11,22,33):
        assert {t['repair'] for t in rows if t['seed']==seed}=={'fm','fixed32','ramp32','ramp64'}
