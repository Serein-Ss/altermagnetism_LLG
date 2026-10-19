import math
import numpy as np
import torch


def vectors(x):
    if x.ndim!=4 or x.shape[1]<2 or x.shape[-1]!=3 or len(x)<2:
        raise ValueError('independent spin path ensembles required')
    return x[:,1:].flatten(1)/math.sqrt(x.shape[2]*(x.shape[1]-1))


def distances(a,b):
    # Direct norm calculation handles coincident samples without cancellation.
    return torch.cdist(a,b,p=2,compute_mode='donot_use_mm_for_euclid_dist')


def joint_loss(generated,reference):
    if generated.shape[1:]!=reference.shape[1:] or generated.shape[1]!=2 or len(generated)%2:
        raise ValueError('matching two-frame paths and two independent generated folds required')
    for x in (generated,reference):
        if not torch.allclose(x[:,0],reference[:1,0].expand_as(x[:,0]),atol=1e-6,rtol=0):
            raise ValueError('conditional anchor differs')
    g=vectors(generated);r=vectors(reference.detach());a,b=g.chunk(2)
    gr=distances(a,r).mean()+distances(b,r).mean()
    gg=distances(a,b).mean();rr=distances(r,r).mean()
    # Energy-distance excess relative to the finite empirical TRAIN law.
    # Gen folds are independent; this avoids diagonal shrinkage of gen variance.
    loss=gr-gg-rr
    return loss,dict(generated_reference=float(gr.detach()),generated_generated=float(gg.detach()),reference_reference=float(rr.detach()))


def evaluate_distance(generated,reference):
    """Unbiased two-sample energy distance, may be negative; no pass threshold."""
    if generated.shape[1:]!=reference.shape[1:]:raise ValueError('matching observation grid required')
    g,r=vectors(generated),vectors(reference)
    gg,rr=distances(g,g),distances(r,r)
    return float(2*distances(g,r).mean()-gg.sum()/(len(g)*(len(g)-1))-rr.sum()/(len(r)*(len(r)-1)))


def distance_with_ci(generated,reference):
    g,r=[vectors(torch.as_tensor(x,dtype=torch.float64)) for x in (generated,reference)]
    gg,gr,rr=[x.numpy() for x in (distances(g,g),distances(g,r),distances(r,r))]
    def statistic(i,j):
        return float(2*gr[np.ix_(i,j)].mean()-gg[np.ix_(i,i)].sum()/(len(i)*(len(i)-1))-rr[np.ix_(j,j)].sum()/(len(j)*(len(j)-1)))
    rng=np.random.default_rng(976977)
    boot=[statistic(rng.integers(len(g),size=len(g)),rng.integers(len(r),size=len(r))) for _ in range(100)]
    return dict(estimate=statistic(np.arange(len(g)),np.arange(len(r))),pointwise_ci95=np.quantile(boot,[.025,.975]).tolist(),
                bootstrap_repeats=100,resampling_unit='independent_whole_branch_within_fixed_state',generated_branches=len(g),reference_branches=len(r))
