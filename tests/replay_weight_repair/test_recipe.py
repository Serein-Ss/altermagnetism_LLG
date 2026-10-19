import pytest
import torch
from .recipe import add_replay,WEIGHTS


@pytest.mark.parametrize('weight',[1.,.25,0.])
def test_weighted_sum_matches_direct_backward(weight):
    p=torch.nn.Parameter(torch.tensor([.7,-.3],dtype=torch.float64))
    native=(p-torch.tensor([1.,0.])).square().sum()
    replay=(p-torch.tensor([-1.,2.])).square().sum()
    native.backward();result=add_replay([p],replay,weight,True)
    q=p.detach().clone().requires_grad_()
    total=(q-torch.tensor([1.,0.])).square().sum()+weight*(q-torch.tensor([-1.,2.])).square().sum()
    total.backward()
    assert torch.allclose(p.grad,q.grad,atol=1e-12,rtol=0)
    assert result['native_norm']>0 and result['replay_norm']>0
    assert -1<=result['cosine']<=1


def test_task_matrix_and_unregistered_weight():
    from .run import tasks,source
    assert len(tasks())==9
    assert {t['seed'] for t in tasks()}=={11,22,33}
    assert {t['repair'] for t in tasks()}==set(WEIGHTS)
    assert [source(s).name for s in (11,22,33)]==['00','03','06']
    p=torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError):add_replay([p],p.sum(),.5)
