"""Exact binomial interval sensitivity for planning, not observed power.

Every Bernoulli observation denotes one independent complete-window trajectory
under a fixed initial/condition and a separately frozen event definition.
Frames/sites are never counted as replicates. No equivalence tolerance is chosen.
"""
from pathlib import Path
import numpy as np
from scipy.stats import beta, binom
from scripts.workflow.bauer_campaign import save


def exact_interval(k,n,alpha=.05):
    if not isinstance(n,(int,np.integer)) or n<1 or not isinstance(k,(int,np.integer)) or not 0<=k<=n or not 0<alpha<1:
        raise ValueError('integer counts and alpha in (0,1) required')
    return (0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),
            1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k)))


def planning_row(n,comparisons=1,alpha=.05):
    if not isinstance(comparisons,int) or comparisons<1:raise ValueError('positive comparison count required')
    intervals=np.asarray([exact_interval(k,n,alpha/comparisons) for k in range(n+1)])
    # To cover a DIFFERENCE, both constituent probability intervals must cover.
    # Bonferroni across two probabilities and all registered differences.
    pair=np.asarray([exact_interval(k,n,alpha/(2*comparisons)) for k in range(n+1)])
    widths=intervals[:,1]-intervals[:,0]
    return dict(n_per_ensemble=n,comparisons=comparisons,family_alpha=alpha,
        worst_single_probability_interval_half_width=float(widths.max()/2),
        worst_difference_interval_half_width=float((pair[:,1]-pair[:,0]).max()),
        zero_events_one_sided_upper=float(1-(alpha/comparisons)**(1/n)),
        sensitivity=[dict(assumed_probability=p,
            expected_single_interval_width=float(binom.pmf(np.arange(n+1),n,p)@widths)) for p in (.01,.05,.1,.3,.5)],
        real_paths_per_initial=2*n,generated_paths_per_initial_per_training_seed=n)


def main():
    dest=Path('tests/results/reference_precision_plan/summary.json')
    if dest.exists():raise FileExistsError(dest)
    rows=[planning_row(n,m) for m in (1,24) for n in (16,32,64,128,256,512)]
    # 24 is a sensitivity scenario: 2 temperatures x 4 initials x 3 models
    # for ONE event. It is not the complete future path-feature test family.
    save(dest,dict(status='COMPLETE_PLANNING_SENSITIVITY_NOT_CERTIFICATION',
        scope='No sample size, event tolerance or final contract selected. Exact Clopper-Pearson probabilities; conservative difference intervals by endpoint subtraction. Precision, not power of an equivalence test. Finite-window events only, not unlimited-time switching.',
        comparison_scenario='m24 means2 temperatures x4 initial families x3 training seeds for one event only; extra events/features/look times need further accounting.',
        independent_unit='noise trajectory conditional on a fixed initial state; number of initials limits across-initial inference',
        sample_size_sensitivity=rows,
        unresolved=['scientifically justified event and mechanism tolerances','calibrated basin and observation/confirmation window',
            'simultaneous continuous path-feature uncertainty and coverage','independent initial/noise provenance freeze',
            'reference timestep convergence at the selected finite observation horizon','measured CPU generation cost and final allocation']))


if __name__=='__main__':main()
