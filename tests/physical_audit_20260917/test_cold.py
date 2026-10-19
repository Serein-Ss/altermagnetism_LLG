import pytest
import torch
from tests.physical_audit_20260917.cold import strong_rotation


@pytest.mark.parametrize('shape,basis',[((8,),1),((8,6),2),((6,6,6),4)])
def test_strong_rotation_is_large_norm_preserving_and_nonmutating(shape,basis):
    settings={'strong_rotation_center_degrees':35,'strong_rotation_modulation_degrees':15}
    s=torch.zeros((1,*shape,basis,3),dtype=torch.float64); s[...,2]=1
    rotated=strong_rotation(s,shape,settings)
    torch.testing.assert_close(rotated.norm(dim=-1),torch.ones_like(s[...,0]),atol=1e-14,rtol=0)
    angle=torch.acos((rotated*s).sum(-1))*180/torch.pi
    assert abs(angle.min().item()-20)<1e-10 and abs(angle.max().item()-50)<1e-10
    assert s[...,2].min()==1 and s[...,0].abs().max()==0
