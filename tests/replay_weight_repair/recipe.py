import torch

WEIGHTS={'weight1':1.,'weight0p25':.25,'weight0':0.}


def add_replay(parameters, loss, weight, diagnose=False):
    """Native gradients already accumulated; obtain an unscaled replay gradient."""
    if weight not in WEIGHTS.values():raise ValueError('unregistered replay weight')
    parameters=list(parameters)
    grads=torch.autograd.grad(loss,parameters,allow_unused=True)
    if diagnose:
        native=torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().reshape(-1) for p in parameters])
        replay=torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p,g in zip(parameters,grads)])
        n,r=native.norm(),replay.norm()
        stats=dict(native_norm=float(n),replay_norm=float(r),
                   cosine=float(native.dot(replay)/(n*r)) if n>0 and r>0 else None,
                   weighted_ratio=float(weight*r/n) if n>0 else None)
    else:stats=None
    for p,g in zip(parameters,grads):
        if g is not None:
            if p.grad is None:p.grad=torch.zeros_like(p)
            p.grad.add_(g,alpha=weight)
    return stats
