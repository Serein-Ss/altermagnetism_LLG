"""Versioned joint-distance uncertainty audit; no frozen evaluator changes."""
import numpy as np
import torch
from tests.joint_score_repair.objective import vectors,distances


def matrices(generated,reference):
    if generated.shape[1:]!=reference.shape[1:]:raise ValueError('matching observation grids required')
    g,r=[vectors(torch.as_tensor(x,dtype=torch.float64)) for x in (generated,reference)]
    return tuple(x.numpy() for x in (distances(g,g),distances(g,r),distances(r,r)))


def values(gg,gr,rr,cg,cr):
    n,m=gg.shape[0],rr.shape[0]
    return (2*np.einsum('bi,ij,bj->b',cg,gr,cr,optimize=True)/(n*m)
            -np.einsum('bi,ij,bj->b',cg,gg,cg,optimize=True)/(n*(n-1))
            -np.einsum('bi,ij,bj->b',cr,rr,cr,optimize=True)/(m*(m-1)))


def summarize(gg,gr,rr,repeats=999,permutations=999,seed=979100):
    n,m=len(gg),len(rr)
    if n<2 or m<2 or repeats<20 or permutations<20:raise ValueError('insufficient branches or resamples')
    observed=float(values(gg,gr,rr,np.ones((1,n)),np.ones((1,m)))[0])
    empirical=float(2*gr.mean()-gg.mean()-rr.mean())
    rng=np.random.default_rng(seed)
    cg=rng.multinomial(n,np.full(n,1/n),size=repeats).astype(float)
    cr=rng.multinomial(m,np.full(m,1/m),size=repeats).astype(float)
    boot=values(gg,gr,rr,cg,cr)
    # Conditional bootstrap mean is exactly the empirical V statistic, NOT U.
    # Reflect the centered bootstrap error, rather than taking its raw quantiles.
    lo,hi=np.quantile(boot-empirical,[.025,.975])
    pooled=np.block([[gg,gr],[gr.T,rr]])
    masks=np.zeros((permutations,n+m))
    for i in range(permutations):masks[i,rng.permutation(n+m)[:n]]=1
    other=1-masks;gd=masks@pooled;rd=other@pooled
    perm=(2*(gd*other).sum(1)/(n*m)-(gd*masks).sum(1)/(n*(n-1))-(rd*other).sum(1)/(m*(m-1)))
    p=float((1+np.count_nonzero(perm>=observed-1e-12))/(permutations+1))
    return dict(version=2,estimate=observed,empirical_v=empirical,
        exact_bootstrap_center_shift=empirical-observed,observed_bootstrap_center_shift=float(boot.mean()-observed),
        legacy_raw_percentile95=np.quantile(boot,[.025,.975]).tolist(),
        centered_basic_approx95=[float(observed-hi),float(observed-lo)],
        interval_scope='Nondegenerate-alternative approximation only; not certified near equal distributions. No clipping, equivalence or pass gate.',
        permutation_equality_p=p,permutation_repeats=permutations,bootstrap_repeats=repeats,seed=seed,
        permutation_scope='One-sided equality test under within-fixed-state independent branch exchangeability; nonrejection is not equivalence. Raw per-state p, multiple comparisons not used for acceptance.',
        resampling_unit='independent_whole_branch_within_fixed_state_both_ensembles',generated_branches=n,reference_branches=m)


def distance_with_uncertainty(generated,reference,**kwargs):
    return summarize(*matrices(generated,reference),**kwargs)
