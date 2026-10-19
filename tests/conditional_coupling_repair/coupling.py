"""Whole-configuration assignment within identical conditioning states only."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def schedule(seed, steps):
    if steps % 16:
        raise ValueError('complete 16-update balanced cycles required')
    rng = np.random.default_rng(seed)
    ids, branches = [], []
    for _ in range(steps // 16):
        order = rng.permutation(48).reshape(16, 3)
        for parents in order:
            ids.append(np.repeat(parents, 16))
            branches.append(np.concatenate([rng.choice(64, 16, replace=False) for _ in parents]))
    return np.asarray(ids), np.asarray(branches)


@torch.no_grad()
def match(z, y, group=16):
    if z.shape != y.shape or z.ndim != 4 or z.shape[1] != 2 or len(z) % group:
        raise ValueError('equal two-frame grouped ensembles required')
    if not torch.allclose(z[:,0], y[:,0], atol=1e-6, rtol=0):
        raise ValueError('latent and target must share initial states')
    indices=[];original=[];assigned=[]
    for start in range(0,len(z),group):
        a=z[start:start+group];b=y[start:start+group]
        if not torch.allclose(b[:,0],b[:1,0].expand_as(b[:,0]),atol=1e-6,rtol=0):
            raise ValueError('assignment cannot cross conditioning states')
        # Dot products in float64; cost sums squared geodesic distances over
        # all sites. A single permutation preserves spatial joint configurations.
        dot=torch.einsum('anc,bnc->abn',a[:,1].double(),b[:,1].double())
        cost=dot.clamp(-1,1).acos().square().mean(-1).cpu().numpy()
        rows,cols=linear_sum_assignment(cost)
        assert np.array_equal(rows,np.arange(group))
        indices.extend((cols+start).tolist())
        original.append(float(np.trace(cost)/group));assigned.append(float(cost[rows,cols].mean()))
    ix=torch.tensor(indices,device=y.device)
    return y[ix],dict(random_cost=float(np.mean(original)),assignment_cost=float(np.mean(assigned)),
                     permutation=indices)
