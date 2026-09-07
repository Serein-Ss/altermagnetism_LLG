# Standard finite-temperature LLG trajectory suite (schema v2)

## Intended use

This is the phase-1 dataset for conditional path-generation model development
and a strictly size-held-out test.  It contains unbiased stochastic-LLG paths
from the published Gomonay *et al.* d-wave altermagnet Hamiltonian
(DOI: 10.1038/s44306-024-00042-3).  No trajectory was selected or discarded by
its outcome, and every retained path has `sample_weight = 1`.

The suite-level certificate status is `ready_for_phase1_model_training`.
This means that the files, splits, conditioning, random seeds and phase-1
mechanism coverage passed the implemented checks.  It does not mean that the
sample count is sufficient for percent-level rare-event probabilities or that
the mechanism classifier is a converged physical phase label.

## Fixed simulation protocol

- Temperature: 5 K.
- Damping-like drive: 0.70, 0.78 and 0.80 T; field-like drive: 0 T.
- Integrator: stochastic Heun for the Stratonovich stochastic LLG equation.
- Thermal field: fluctuation-dissipation amplitude corresponding to
  `<B_mu(r,t) B_nu(r',t')> = 2 alpha k_B T/(gamma mu_s)` times the spatial,
  temporal and Cartesian Kronecker/Dirac deltas.
- Integration step: 0.05 fs; saved interval: 10 fs.
- Saved duration: 1 ps (101 frames including the initial frame).
- Equilibration: 0.2 ps; drive pulse: 0.5 ps.
- Boundary: periodic in both lattice directions.
- Initial-state family: collinear.

The 0.05 fs step was selected conservatively after the 0.2/0.1/0.05 fs scan.
The 0.1 and 0.05 fs switching estimates were statistically compatible, but the
finite ensemble did not resolve their absolute difference below 0.05.

## Files and splits

| File | Role | Paths | Train/validation/test | Size |
|---|---|---:|---:|---:|
| `train/d_wave_altermagnet_L16.h5` | small-system training | 90 | 72/9/9 | 16 x 16 x 2 |
| `train/d_wave_altermagnet_L32.h5` | small-system training | 90 | 72/9/9 | 32 x 32 x 2 |
| `train/d_wave_altermagnet_L64.h5` | nonuniform-mechanism training anchor | 90 | 72/9/9 | 64 x 64 x 2 |
| `test_large/d_wave_altermagnet_L96.h5` | held-out large-size test | 60 | 0/0/60 | 96 x 96 x 2 |

Totals: 216 training, 27 validation and 87 test trajectories (330 paths,
2,053,112,371 bytes).  Splits are assigned within each physical condition by
whole trajectory.  L96 is never used for fitting, checkpoint selection or
threshold selection.

## Observed outcome coverage

These historical counts use the endpoint-only geometric rule. A
`negative_endpoint` path ends with global Neel-z below zero; this is not the
same as a `committed_switch`, which now requires residence in a
data-calibrated reverse basin. The old mechanism table calls a negative endpoint
spatially nonuniform when the maximum spatial standard deviation of the local
Neel order during `|global order| < 0.5` is at least 0.25.

| Size | No crossing | Coherent-like negative endpoint | Nonuniform negative endpoint |
|---:|---:|---:|---:|
| 16 | 50 | 40 | 0 |
| 32 | 57 | 33 | 0 |
| 64 | 49 | 30 | 11 |
| 96 | 33 | 9 | 18 |

The L64 training data therefore expose the model to nonuniform paths, while
the independently held-out L96 file tests whether their size dependence is
generalized.  These descriptors are analysis labels, not supervised targets
and not asserted to be unique microscopic mechanisms.

## HDF5 layout

The root contains `time` and the `d_wave_altermagnet` group.  The group stores:

- `spins`: `(path, frame, sublattice, x, y, xyz)` full spin trajectories;
- `initial_spins`, `magnetization`, `neel` and `energy`;
- `physical_condition`, dimensionless `model_condition` and covariant
  `vector_condition`;
- `condition_id`, `trajectory_id`, `trajectory_seed`, `sample_weight` and
  `split` (`0/1/2 = train/validation/test`).

Use lazy HDF5 reads or the project trajectory loader; do not load all four
`spins` arrays into memory simultaneously.

## Certification and version identity

- `suite_certification.json` is the authoritative cross-file certificate and
  records exact SHA-256 values, byte counts and per-condition statistics.
- Each HDF5 has a sibling `.manifest.json`, `.certification.json` and
  `.mechanisms.json`.
- All 330 trajectory seeds are globally unique.  Their namespaces are derived
  deterministically by SHA-256 from the base seed, system and lattice size.
- A prior draft used overlapping integer seed ranges between sizes; the
  cross-suite audit detected this and all four files were regenerated.  The
  files listed here are the corrected version.

## Claim boundary and remaining gates

Permitted now: model implementation, debugging, conditional-distribution
learning, validation-set checkpoint selection and a held-out L96 size test.

Not yet supported: percent-level switching probabilities, rare-event tails,
temperature extrapolation, transfer to other Hamiltonians or initial-state
families, experimental material prediction, or a definitive mechanism phase
diagram.  Before such claims, increase independent path counts and complete
save-cadence convergence, mechanism-threshold sensitivity and larger-size
convergence tests.

No generative-model training was performed while constructing or certifying
this dataset.
