# Auxiliary frequency repair

## Question and previous evidence

The completed shadow-Adam audit found clipping factors of 1. At weight 0.1,
local moments changed a single auxiliary-call Adam update by about 6.5–7.4%,
spatial-only moments by about 0.6–0.7%. These are local counterfactual probes,
not accumulated training effects or evidence of a unique failure source.

## Paired design

Start each seed 11/22/33 from its own spatial_covariance_repair **control**
checkpoint 126144. All three arms reset Adam identically and train 2048 updates.
Network, data, native FM, teacher weight 0.25, physical grid and sampler remain
fixed. All arms use local plus spatial moments; sparse64 calls every64 at0.1,
dense8_same_mass every8 at0.0125, dense8_strong every8 at0.1.
The ramp is constant within each64 block. Sparse and dense_same have exactly
equal scalar coefficient sums per block; this is NOT equal Adam, gradients,
state coverage or compute. Formal calls:32 versus256 versus256.
All arms draw inputs every8; sparse skips extra computation, not RNG consumption.
Common64-step inputs and the complete planned input stream are hashed.

## Validation and interpretation

CPU unit tests check counts, ramp budgets and source roles. A short GPU preflight
uses16 updates and two actual auxiliary calls per arm, reduced development
evaluation; it tests software only and does not reproduce formal frequency ratios.
Formal evaluation keeps the existing full0.1ps/1ps reference comparisons,128/512
sampling convergence, spatial tensor/crossfold diagnostics, energy and time
statistics, joint distances and symmetry probes. Known joint CI undercoverage
and reused-development limitations are retained. Three training seeds do not
turn correlated states or lattice sites into independent repetitions.

Sparse versus equal-budget dense tests scheduling/coverage; the two dense arms
test coefficient strength. Examine unoptimized full-path statistics and costs
alongside optimized moments, rather than reporting a best seed or lower loss.
No physical tolerance is invented; no production promotion is authorized.

## Execution

Use zrs-mag via ../run_zrs_mag.sh. GPU: rtx4090, name zrs-mag; statistics: fat.
Estimated formal runtime3–5.5h per GPU task,6h allocation; statistics3–10min.
Nine GPU array entries have no concurrency cap. Submit dependencies after short
preflight, then do not monitor long tasks. Outputs and checkpoints remain here.
