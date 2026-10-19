import torch
import pytest
from scripts.model.gate_f import reference_path
from tests.experiments.conditional_source import conditional_source


def test_default_exact_and_centered_source_not_hemisphere_limited():
    initial=torch.tensor([0.,0.,1.]).expand(1000,1,3,1,3)
    a=torch.Generator().manual_seed(71);b=torch.Generator().manual_seed(71)
    assert torch.equal(reference_path(initial,4,a),conditional_source(initial,4,b))
    assert torch.equal(a.get_state(),b.get_state())
    z=conditional_source(initial,4,torch.Generator().manual_seed(71),1.)
    assert torch.equal(z[:,0],initial)
    assert (z.norm(dim=-1)-1).abs().max()<1e-6
    assert z[:,1:,...,2].mean()>.3
    assert (z[:,1:,...,2]<0).float().mean()>.05
    with pytest.raises(ValueError):conditional_source(initial,4,a,0.)
