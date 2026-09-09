# Unified LLG benchmark: results and usage

## 1. What the spin-wave animation shows

The GIF is a deterministic, zero-temperature, zero-damping LLG trajectory from
the published double-layer Hamiltonian. It contains 201 stored states over
0--4 ps. The transverse spin components are only of order `1e-3` and are
therefore magnified by 1000 in the visualization. The final frame is not a
relaxed state: with `alpha = 0` it is simply a later phase of the persistent
spin-wave precession.

For the selected lattice wave vector, the complex spatial-mode amplitude is

`a_k(t) = mean_r [(S_1x(r,t) + i S_1y(r,t)) exp(-i k.r)]`.

Linearizing the LLG equation around the collinear state gives normal modes
`a_k(t) ~ exp(-i omega t)`. A temporal Fourier transform therefore places a
peak at each eigenfrequency. Using the complex transverse signal rather than a
real component preserves the sign of frequency and separates the two circular
polarizations. The 4 ps observation window gives a raw frequency-bin spacing
of `1 / 4 ps = 0.25 THz`; longer trajectories or spectral peak fitting would
reduce this finite-window uncertainty.

At mode `(2,2)` on the `16 x 16` lattice, the simulated peaks are 12.2494 and
13.9993 THz. Supplementary Eq. (S.9) predicts 12.3563 and 13.9056 THz, giving
relative errors of 0.87% and 0.67%.

## 2. Why the current reproduction is not every pixel of published Fig. 2

Published Fig. 2 combines three levels of evidence: DFT, atomistic spin-model
dispersion, and phenomenology in the wave-vector panel; a two-dimensional
angular splitting map and spin-precession schematics form the other panel. The
current reproduction checks the published analytic dispersion and angular
sign pattern, but the direct LLG spectral calculation is deliberately only one
wave vector. It does not rerun the paper's DFT calculation, a dense LLG
wave-vector sweep, or the illustrative spin-wave packets. Thus it reproduces
the Hamiltonian-level claim and one dynamical point, not the full publication
workflow.

Read the side-by-side figure as follows:

- left panel: the two branches split along the diagonal `[110]` direction;
- middle panel: the splitting vanishes on nodal directions and changes sign
  after a 90-degree crystal rotation;
- right panel: the complex LLG mode has peaks at opposite signed frequencies,
  identifying the two circular branches;
- published image: the corresponding full wave-vector/intensity map, angular
  map, and schematic interpretation.

## 3. Unified solver and benchmark systems

All systems now use the same unit-spin tensor contract
`[batch, sublattice, x, y, xyz]`, the same Gilbert form, the same optional SOT
interface, and the same Stratonovich stochastic-Heun integrator. A model
adapter supplies only the Hamiltonian field, magnetic moment, gyromagnetic
ratio, Boltzmann constant, units, and provenance.

The initial benchmark contains:

1. `ordinary_free_moments`: Nishino and Miyashita, PRB 91, 134411 (2015),
   Fig. 1 parameters `h=2`, `M=1`, `gamma=kB=1`, `alpha=0.05`, `dt=0.005`.
   Its equilibrium magnetization is independently known from the Langevin
   function. The source has an associated 2018 erratum
   (`10.1103/PhysRevB.97.019904`), which is recorded in the model metadata for
   future regenerations; the benchmark is judged by the exact stationary
   distribution rather than by visual agreement alone.
2. `d_wave_altermagnet`: the Gomonay et al. 2024 double-layer Hamiltonian with
   its published exchange, anisotropy, moment, and lattice parameters.
3. `conventional_afm_control`: the same published Hamiltonian with only the
   alternating exchange set to zero. This is a controlled ablation, not a
   claim about a separately parameterized material.

Certification is tiered. The ordinary system passes an analytic equilibrium
observable; the altermagnet passes the published spin-wave frequencies; the
AFM control passes common numerical checks but currently has no independent
material-specific literature curve. The collection is therefore a partially
literature-certified training benchmark, not a fully certified materials
database.

## 4. Trainable full-spatial data

The HDF5 file contains 180 complete trajectories: 3 systems, 3 conditions per
system, and 20 paths per condition. Every path has 51 stored spatial frames on
an `8 x 8` periodic lattice. Spins are stored as float32 in per-trajectory gzip
chunks; observables and conditions are stored alongside them. The loader opens
the file lazily, pads the one-sublattice ordinary system to two sublattices,
and returns a sublattice mask.

The split is exactly 16/2/2 within every system-condition, hence 144/18/18
globally. The split unit is the whole trajectory. Outcome labels are not used
to create the split. Each physical condition is generated from its own seeded
counter-based random stream, every tensor lane receives disjoint Gaussian
draws, every generated path is retained, and all paths within a condition have
equal statistical weight.

The condition vector is
`[temperature, alpha, damping_like_drive, static_field_z, dt, duration]`.
Raw values remain in their model's declared units, so training code should use
`system_id` and manifest metadata to normalize them rather than mixing SI and
dimensionless quantities blindly.

## 5. Current data conclusions

- The maximum stored spin-length error is `5.96e-8`; all values are finite.
- The ordinary-system mean magnetization differs from the exact Langevin value
  by 0.0101, 0.00187, and 0.0178 at temperatures 1, 2, and 5, respectively.
- At 0.8 T, the negative-final fractions are 0.45 for the AFM control and 0.35
  for the d-wave altermagnet. The trajectories occupy separated positive and
  negative basins even though their ensemble mean lies near zero. This is the
  failure mode of an average trajectory that motivates a path-distribution
  model.
- At 1.0 T, many paths cross zero and return before 1 ps, so switching
  probability is not monotonically encoded by a single fixed-time endpoint.
  Full path statistics are needed.
- Peak spatial incoherence is only about 0.011--0.012 for both two-sublattice
  systems. At this lattice size and temperature, the observed branching is
  predominantly coherent; the present data do not establish a nucleation or
  domain-wall mechanism or an altermagnet-specific effect.

## 6. Generative-model rules

A diffusion or flow-matching model should learn full paths, not independent
frames. It must enforce or test:

- unit spins on `S2` by tangent-space/Riemannian updates or explicit projection;
- periodic lattice translations;
- the published combined C4 rotation and sublattice-registry exchange for the
  d-wave model;
- consistent rotation of spins, SOT polarization, fields, and crystal tensors
  under any augmentation;
- trajectory-level splits and separate condition-holdout tests;
- comparisons of path-class weights, first-passage times, spatial correlations,
  energy/work distributions, and failure probabilities against held-out LLG.

The current HDF5 file is a runnable pilot for scripts/model/data-interface development.
Publication-scale physical claims require larger lattices, longer preparation,
timestep and size convergence, more trajectories, and preregistered held-out
conditions.
