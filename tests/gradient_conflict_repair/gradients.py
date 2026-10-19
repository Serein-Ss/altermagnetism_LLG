"""One-sided Euclidean projection; no claim of Adam descent guarantees."""
import torch


def merge(native, replay, mode):
    if mode not in ('sum', 'project', 'rescale'):
        raise ValueError(mode)
    pairs=[(a,b) for a,b in zip(native,replay) if a is not None or b is not None]
    exemplar=next(a if a is not None else b for a,b in pairs)
    zero=exemplar.new_zeros(())
    nn=sum((a.square().sum() for a,b in pairs if a is not None),zero)
    rr=sum((b.square().sum() for a,b in pairs if b is not None),zero)
    dot=sum(((a*b).sum() for a,b in pairs if a is not None and b is not None),zero)
    coefficient=torch.minimum(dot,zero)/nn.clamp_min(1e-30)
    projected_rr=(rr-2*coefficient*dot+coefficient.square()*nn).clamp_min(0)
    ratio=torch.sqrt(projected_rr/rr.clamp_min(1e-30)) if float(rr)>0 else zero+1
    combined=[]
    for a,b in zip(native,replay):
        if a is None and b is None:
            combined.append(None);continue
        a=torch.zeros_like(b) if a is None else a
        b=torch.zeros_like(a) if b is None else b
        contribution=b-coefficient*a if mode=='project' else b*ratio if mode=='rescale' else b
        combined.append(a+contribution)
    stats=dict(cosine=float(dot/(nn*rr).sqrt().clamp_min(1e-30)),dot=float(dot),
               native_norm=float(nn.sqrt()),replay_norm=float(rr.sqrt()),
               projected_replay_norm_ratio=float(ratio),conflict=bool(dot<0))
    return combined,stats


def clone_grad(parameters):
    return [None if p.grad is None else p.grad.detach().clone() for p in parameters]
