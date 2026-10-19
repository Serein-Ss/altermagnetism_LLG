"""Initial-centered projected Gaussian source, not physical thermal noise."""
import math
import torch
from scripts.model.gate_f import reference_path


def conditional_source(initial,frames,generator,scale=None):
    if scale is None:return reference_path(initial,frames,generator)
    if frames<2 or not math.isfinite(scale) or scale<=0:raise ValueError('positive noise scale and >=2 frames required')
    noise=torch.randn((len(initial),frames,*initial.shape[1:]),device=initial.device,dtype=initial.dtype,generator=generator)
    z=initial[:,None]+scale*noise
    z=z/z.norm(dim=-1,keepdim=True)
    return torch.cat((initial[:,None],z[:,1:]),dim=1)
