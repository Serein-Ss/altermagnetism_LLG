"""Shadow AdamW step: clone parameters and optimizer, never alter checkpoint."""
import copy
import torch


def flat(grads,parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for g,p in zip(grads,parameters)])


def combine(a,b,weight):
    return [None if x is None and y is None else
        (0 if x is None else x)+(0 if y is None else weight*y) for x,y in zip(a,b)]


def shadow(parameters,optimizer_state,gradients):
    parameters=list(parameters)
    if len(parameters)!=len(gradients) or len(optimizer_state['param_groups'])!=1:raise ValueError('matching gradients and one registered AdamW group required')
    clones=[torch.nn.Parameter(p.detach().clone()) for p in parameters]
    opt=torch.optim.AdamW(clones);opt.load_state_dict(copy.deepcopy(optimizer_state))
    for p,g in zip(clones,gradients):p.grad=None if g is None else g.detach().clone()
    norm=torch.nn.utils.clip_grad_norm_(clones,10.)
    opt.step()
    delta=torch.cat([(p.detach()-before.detach()).reshape(-1) for p,before in zip(clones,parameters)])
    return delta,dict(preclip_norm=float(norm),clip_factor=min(1.,10./(float(norm)+1e-6)))


def cosine(a,b):
    den=a.norm()*b.norm()
    return None if float(den)==0 else float(a.dot(b)/den)
