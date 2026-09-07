# Periodic equivariant path-flow model specification

## Learning target

The model does not replace the deterministic right-hand side of LLG.  It learns
the conditional law of complete finite-temperature paths,

`p[S(0:T) | S0, Hamiltonian, T, alpha, drive, crystal frame, geometry]`.

Repeated sampling under one condition must reproduce path-class probabilities,
first-passage-time distributions and spatial mechanisms.  A single generated
path is not the model output used for scientific conclusions.

## Implemented first architecture

Use a **periodic equivariant Riemannian flow-matching model** (PE-RFM):

1. Represent every saved spin as a point on `S2`; a complete trajectory is a
   point on the product manifold `S2^(time*sublattice*sites)`.
2. Use a Brownian-bridge-like random reference path whose first frame is fixed
   to the supplied initial state.
3. Connect reference and data paths by shortest spherical geodesics and train a
   conditional velocity field with Riemannian conditional flow matching.
4. Parameterize the velocity with the PyTorch-only
   `model/network.py::PeriodicEquivariantFlowNet`. It is fully convolutional,
   circular in both spatial axes and accepts sizes not seen during training.
5. Form covariant vector bases from the initial state, spatial/temporal
   differences, the other sublattice, vector conditions, and the complete
   dimensionless Gomonay effective-field stencil. A periodic scalar network uses
   only dot products and squared norms to predict basis coefficients. This gives
   joint proper-rotation covariance without an `e3nn` dependency.
6. Project every predicted vector to the local tangent plane,
   `v_tan = v - (v dot S) S`, before evaluating the flow loss or integrating.
7. Architecture version 2 standardizes all ten scalar conditions from the
   training split, injects temperature, damping and both drive amplitudes into
   every residual block with FiLM, and normalizes channels independently at
   each lattice site and time rather than pooling statistics across space.

This is more appropriate than a plain 3-D U-Net: treating `(mx,my,mz)` as three
unrelated image channels breaks joint spin/field/crystal-frame rotation
covariance. Crystal anisotropy does not imply arbitrary spin-only rotation
invariance; spins, fields, DMI vectors and crystal axes must rotate together.

The default eight residual blocks have an 18-cell spatial receptive-field
radius, about two 4-nm walls at `a0=0.448 nm`. The implementation is not parity
equivariant and has no nonlocal magnetostatic operator; if dipolar interactions
are added to the Hamiltonian, an FFT/global field branch must also be added.

## Small-to-large protocol

- `16x16` and `32x32`: local thermal/spin-wave pretraining and finite-size
  controls only.
- `48x48`: smallest candidate that can contain a periodic wall pair with limited
  overlap; it is not accepted until the convergence gate passes.
- `64x64`: recommended production anchor for nonuniform mechanisms.
- `96x96` and `128x128`: completely held-out LLG and generated ensembles for
  size generalization.

Training exclusively on `8x8` or `16x16` cannot establish extrapolation of a
domain-wall mechanism that those boxes cannot represent. “Small-system
training” for that claim therefore means pretraining below 48 followed by at
least some converged 48/64 mechanism data, with 96/128 reserved for testing.

## Training sample

One item contains:

- `spins`: `[time, sublattice, nx, ny, 3]`, the target full path;
- `initial_spins`: `[sublattice, nx, ny, 3]`;
- `physical_time`: `[time]` and a valid-frame mask;
- scalar conditions: `T`, `alpha`, `gamma`, magnetic moment, normalized
  exchange/anisotropy/DMI parameters, pulse amplitudes and durations;
- vector conditions: external field, SOT polarization and time-dependent drive;
- geometry: lattice vectors, periodic axes, neighbor edges and displacements;
- symmetry data: crystal-frame vectors, sublattice identity and allowed joint
  crystal/sublattice operations;
- identifiers used only for provenance: system, condition, trajectory, seed and
  80:10:10 split.

Dimensionful scalars are converted to model-specific natural units before
standardization.  The physical values and conversion factors remain in the
manifest.

## Network input and output

At training flow time `tau in [0,1]`, the network input is the interpolated
noisy path `S_tau`, `tau`, the fixed first frame and all conditions above.  The
network output is one tangent vector for every generated spin and saved time:

`velocity: [time, sublattice, nx, ny, 3]`.

At inference, a random reference path is sampled and the learned flow is
integrated from `tau=0` to `tau=1`.  The returned artifact is one complete spin
trajectory with the same shape as the target data.  Repeating the inference
with different latent seeds produces the conditional path ensemble.

## Training loop

For each batch:

1. Load complete trajectories or contiguous time windows; never treat frames as
   independent samples.
2. Sample `tau` uniformly and draw a temporally correlated spherical reference
   path anchored at `S0`.
3. Construct the spherical geodesic interpolation and its analytic tangent
   velocity target.
4. Predict the velocity and minimize masked tangent-vector MSE. Weight sites,
   times and trajectories uniformly unless the sampling protocol declares a
   physical reweighting scheme.
5. Optionally add small auxiliary losses for global order, structure factor and
   temporal correlations. Do not impose a deterministic LLG residual on a
   coarse stochastic trajectory; that would penalize the unresolved thermal
   increment.

Start with 8:1:1 whole-trajectory splits for interpolation. Add a separate
condition-held-out test (unseen temperature/drive) and a system-held-out test
only after the interpolation model closes.

## Required evaluation

The acceptance test is distributional closure against held-out stochastic LLG:

- distinguish `negative_endpoint`, `crossed_zero`, `committed_switch`,
  `crossing_return` and `unresolved_transition`; infer stable-basin thresholds
  from real LLG terminal residence distributions and extend the observation
  window if unresolved paths remain common;

- path-class fractions with binomial/multinomial uncertainty;
- first-passage and residence-time distributions;
- energy, stochastic-work and order-parameter distributions;
- spatial correlation functions and structure factors;
- coherent/nucleation/domain-wall mechanism fractions;
- spin-wave frequency and linewidth where applicable;
- exact equilibrium observables for the Nishino benchmark;
- symmetry tests under periodic translations and valid joint crystal rotations.

Baselines are a deterministic conditional mean predictor, a non-equivariant
space-time U-Net flow model, and an autoregressive neural-SDE model.  Better MSE
alone is not evidence that the path probability law is correct.
