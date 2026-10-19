import torch
from tests.experiments.reversal_risk_audit import reversal_risks


def test_pair_risk_identity_does_not_imply_improvement_over_original():
    target=torch.zeros(2,3,1,4,1,3,dtype=torch.float64)
    direct=target.clone(); rev=torch.ones_like(target)
    out=reversal_risks(direct,rev,target)
    torch.testing.assert_close(out['symmetric'],(out['original']+out['reversed'])/2-out['defect']/4)
    assert (out['symmetric']>out['original']).all()
    rev[:,0]=100  # Anchor is excluded.
    other=reversal_risks(direct,rev,target)
    for k in out: torch.testing.assert_close(out[k],other[k])
