import torch
from scripts.workflow.bauer_campaign import graph
from scripts.model.gate_f import build_model, reference_path
from tests.experiments.physics_inputs import GeometryOnlyGraph
from tests.experiments.path_learning_audit import oracle_check


def test_oracle_sampler_and_derivative():
    r=oracle_check()
    assert max(x['max_error'] for x in r['sampler']) < 1e-8
    assert r['derivative_error'] < 1e-8
    assert r['tangent_error'] < 1e-10


def test_hamiltonian_inputs_are_removed():
    torch.manual_seed(7)
    a=graph(5,dtype=torch.float32);b=graph(5,dtype=torch.float32)
    b.weights.fill_(-9);b.kappa.fill_(7)
    x=torch.randn(1,1,5,1,3);x/=x.norm(dim=-1,keepdim=True)
    y=reference_path(x,5,torch.Generator().manual_seed(8))
    net=build_model('rfm',width=8,blocks=1)
    args=(y,torch.tensor([.3]),x,torch.tensor([[.13,.1]]),torch.linspace(0,1,5)[None])
    axis=torch.tensor([[0.,0.,1.]])
    assert torch.equal(net(*args,GeometryOnlyGraph(a),axis),net(*args,GeometryOnlyGraph(b),axis))
    assert not torch.allclose(net(*args,a,axis),net(*args,b,axis))
