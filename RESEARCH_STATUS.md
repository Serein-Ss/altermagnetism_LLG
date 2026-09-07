# Research scope, novelty boundary and remaining certification

## Defensible innovation boundary

LLG, stochastic LLG, thermal fields, switching-time distributions, coherent
rotation, nucleation and domain-wall propagation are established methods and
phenomena. They are simulation sources and validation targets, not the proposed
novelty.

The candidate methodological contribution is a conditional generative model of
the **complete finite-temperature magnetic path distribution** that simultaneously
respects the spin sphere, periodic lattice and joint spin/field/crystal symmetry,
and is certified by equilibrium, frequency, switching-rate and spatial-mechanism
observables across ordinary, antiferromagnetic, noncollinear and altermagnetic
Hamiltonians. A final claim of priority requires a broader systematic novelty
search before manuscript submission.

The later physics contribution, if supported by larger simulations, is not that
multimodality exists. It is the change of spatial path mechanisms and their
probability weights with crystal direction and alternating exchange in an
altermagnet.

## Literature validation ladder

1. Nishino and Miyashita, PRB 91, 134411 (2015), DOI
   `10.1103/PhysRevB.91.134411`: stochastic LLG equilibrium agrees with the exact
   Langevin distribution and Monte Carlo. This project has reproduced the exact
   free-moment equilibrium observable.
2. Bauer et al., JPCM 23, 394204 (2011), DOI
   `10.1088/0953-8984/23/39/394204`: at one fixed condition their Fig. 2 contains
   successful reversals and failed attempts, caused by edge nucleation and
   domain-wall propagation. The paper supplies qualitative same-condition path
   classes and a quantitative domain-wall barrier, but not class probabilities,
   random seed or numerical time step.
3. Laliena et al., Scientific Reports 10, 18148 (2020), DOI
   `10.1038/s41598-020-76989-2`: published CrNb3S6 parameters give a zero-field
   helix period of about 48 nm. This replaces the earlier unstable imposed spiral.

An individual stochastic trace cannot be reproduced point by point without the
original random stream. Reproduction therefore targets distributions and
reported observables, not identical noise realizations.

## Work completed in the current extension

- Added the published CrNb3S6 monoaxial Hamiltonian to the common solver.
- Reproduced the 48 nm helix scale: the independent discrete parameter prediction
  is 48.46 nm, with 0.95% relative difference. The commensurate one-turn state
  has negligible torque and energy drift and is lower than tested winding 0/2.
- Added the Bauer open-chain Hamiltonian and its exact LL stochastic convention.
- Reproduced the continuum domain-wall energy with discrete optimized errors of
  0.083% for `K/J=0.01` and 0.861% for `K/J=0.1` at length 100.
- Added an unbiased accelerated path pilot and spatial mechanism descriptors.
  It contains 28 no-crossing, 5 crossing-return and 7 negative-endpoint paths.
  This pilot changes the published length and temperature and is not a Fig. 2
  quantitative reproduction.
- Applied the spatial descriptors to the existing 8x8 benchmark. Every switched
  AFM/altermagnet path is currently classified coherent-like, proving that the
  present lattice is too small to support the intended nucleation/domain-wall
  study.

## Generator issues now resolved in schema v2

- one independently replayable seed is stored per trajectory;
- initial states, physical conditions, dimensionless model conditions, SOT
  polarization and crystal-frame vectors are explicit;
- full paths use spatial/time HDF5 chunks and retain exact 8:1:1 whole-path splits;
- files are written atomically with SHA-256 manifests;
- a separate certifier prevents an integrity-valid but convergence-invalid file
  from being presented as production physics data.

## Current blockers before production use

### Physics and numerical certification

- The correlated AFM/altermagnet preparation time has not been certified by
  autocorrelation or independent equilibrium checks.
- The 8x8 lattice and short trajectory window suppress spatial reversal modes.
- SOT amplitudes are a numerical scan, not published material/device currents.
- Timestep, saved-frame cadence and finite-size convergence are incomplete for
  every production condition.
- The existing altermagnet dataset has only one collinear initial-state family;
  the newly validated helix is not yet included in the unified training HDF5.
- Fixed-length atomistic LLG is not automatically valid near a Curie point or
  after changing to coarse micromagnetic cells.

### Dataset and machine-learning interface

- Twenty paths per condition cannot estimate rare mechanism weights or train a
  high-dimensional generator.
- The current split tests interpolation only; it lacks held-out conditions and
  held-out Hamiltonians.
- Time-dependent protocols are currently represented by amplitude and pulse
  duration, not an arbitrary sampled waveform.
- DMI/bond vectors and geometry masks must be added when a Hamiltonian that
  actually contains them is admitted; zero placeholders are not invented.
- Mechanism labels are threshold-based geometric descriptors. Their threshold,
  lattice-size and saved-cadence robustness remain to be established.

## Next executable production gate

Before launching a large training set:

1. Repeat the Bauer chain at the published `L=100`, `K/J=0.1`, `kBT/J=0.11`,
   `lambda=0.1` condition for at least two converged time steps and enough total
   time to estimate lifetimes. The paper's missing timestep must be treated as a
   numerical choice, not a copied parameter.
2. Run the implemented spatial gate at 48/64/96 cells, full 1-ps-or-longer
   duration and at least 100 paths per condition; scan near-threshold drive and
   temperature until nonuniform events are observed, then repeat the final
   condition independently.
3. Promote a schema-v2 file from `generated_not_convergence_certified` to
   production only after dt, saved-cadence, size and mechanism-threshold checks
   all pass; choose final counts from confidence intervals on the rarest class.
