# Altermagnetic finite-temperature LLG paths

This directory is an independent research workspace for literature-anchored
altermagnetic spin dynamics.  It does not modify the historical
`micromagnetics/` outputs.

## What is being built

The physical target is the conditional path distribution

`p[s_1, s_2](0:T | temperature, crystal direction, damping, drive protocol)`

for a two-sublattice d-wave altermagnet.  The intended question is whether
finite-temperature switching contains distinct path classes, crystal-direction
signatures, and low-probability failures that disappear after averaging LLG
trajectories.

The first reference model is the double-layer RuO2-type discrete Hamiltonian in
Gomonay et al., *npj Spintronics* **2**, 35 (2024), including its Supplementary
Eq. (S.1) and Table I:

- inter-layer AFM exchange `J1 = 11.1 meV`;
- intra-layer FM exchange `J2 = 1.88 meV`;
- alternating diagonal exchange `J_tilde = 0.8 meV`;
- domain-wall anisotropy `K = 0.047 meV`;
- lattice constant `a0 = 0.448 nm` and atomic moment `mu_s = 1 mu_B`.

This is a literature-constrained model, not a claim of parameter-free
quantitative prediction for bulk RuO2.

## Directory contract (restructured)

See [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md) for the current layout,
data inventory, entry points and provenance rules. Historical scientific status
below is not a new certification. GUIDE plans and historical report payloads
retain their original text; use DIRECTORY_MIGRATION.json for old path lookup.

- `GUIDE/`: user research/work plans, including the V3 requirements.
- `scripts/`: all live code by responsibility; `archive/` is frozen evidence only.
- `data/`: literature, research, datasets, audit and generated trajectory arrays.
- `assets/literature/`, `assets/research/`: plots and animations by task.
- `output/smoke/`, `output/production/`, `output/diagnostics/`: checkpoints and metrics.
- `slurm/`: current submission scripts; `archive/` retains historical batch snapshots.
- `logs/`: scheduler/process logs and environment captures.

## Research stages and gates

1. **Hamiltonian gate**: analytic field agrees with `-dH/(mu_s ds)`, energy is
   invariant under translations and under a 90-degree lattice rotation followed
   by sublattice exchange.
2. **Zero-temperature LLG gate**: spin length is conserved, undriven damped
   dynamics lowers energy, and the simulated spin-wave peaks agree with the
   published dispersion.
3. **Finite-temperature gate**: stochastic Heun uses the Stratonovich
   predictor-corrector convention and the fluctuation-dissipation noise
   amplitude.  Equilibrium statistics, timestep convergence and `T -> 0` must be
   checked before production.
4. **Path-data gate**: all successes and failures are retained; no outcome
   balancing or post-selection is allowed in the unbiased set.  Rare-event data
   must be stored separately with statistical weights.
5. **Generative-model gate**: train/validation/test splits are by whole
   trajectory and held-out physical condition.  The future model must respect
   periodic translations, the combined C4/sublattice symmetry, and unit spins
   on S2.

## Commands

Run the tests:

```bash
bash run_zrs_mag.sh -m pytest altermagnetism_LLG/scripts/tests -q
```

Generate the deterministic spin-wave validation data:

```bash
bash run_zrs_mag.sh altermagnetism_LLG/scripts/literature/validate_spinwave.py
```

Plot the validation result:

```bash
bash run_zrs_mag.sh altermagnetism_LLG/scripts/visualization/plot_literature_validation.py
```

Animate the actual stored LLG spin-wave trajectory:

```bash
bash run_zrs_mag.sh altermagnetism_LLG/scripts/visualization/animate_spinwave_trajectory.py
```

Generate and certify the compact full-spatial training benchmark:

```bash
bash run_zrs_mag.sh altermagnetism_LLG/scripts/generation/generate_benchmark_hdf5.py
bash run_zrs_mag.sh altermagnetism_LLG/scripts/validation/certify_benchmark_dataset.py
bash run_zrs_mag.sh altermagnetism_LLG/scripts/visualization/plot_benchmark_dataset.py
bash run_zrs_mag.sh altermagnetism_LLG/scripts/datasets/example_load_dataset.py
```

The stochastic generator deliberately requires an explicit temperature and
protocol.  Do not launch a large production run until those choices and the
switching basins have been fixed.

## Current verified status

- The Hamiltonian/field, periodic-translation, combined C4/sublattice,
  zero-temperature damping, thermal-amplitude, independent trajectory RNG,
  variable-size, lazy-batching, deterministic-baseline and joint-rotation tests pass (`33 passed`).
- For the `[110]` Fourier mode `(2, 2)` on a `16 x 16` lattice, the two LLG
  peaks are `12.2494 THz` and `13.9993 THz`; Supplementary Eq. (S.9) predicts
  `12.3563 THz` and `13.9056 THz`.  The relative errors are `0.87%` and
  `0.67%`, with relative energy drift `2.17e-15`.
- `data/research/unbiased_smoke/` contains four equal-weight, unselected `10 K` paths
  only for end-to-end pipeline QA.  Its `4 x 4` lattice, short preparation and
  four trajectories are not an equilibrium certificate and must not be used
  for scientific switching claims.

The production problem will be finite-temperature SOT-driven 180-degree Neel
switching, including both coherent and nucleation/domain-wall paths.  Drive
thresholds will first be measured for this Hamiltonian rather than copied from
a different material model.  Conditions will include crystal direction and an
AFM control with `J_tilde = 0`; every trajectory will be retained with equal
weight.  The resulting path distribution, not only its mean, is the learning
target.

## Multimodality validation checkpoint

At the independently identified near-separatrix condition (`T = 5 K`,
`H_DL = 0.8 T`, `0.5 ps` pulse, `1 ps` observation, `8 x 8` periodic lattice),
all 256 equal-weight paths were retained.  The pre-registered classes contain
102 no-crossing paths, 15 crossing-and-return paths and 139 paths that are
negative at `1 ps`.  A GMM fitted to PCA coordinates of the continuous
`n_z(t)` paths, without using those labels, improves BIC over one component by
698.4.  Halving the timestep to `0.05 fs` with a new seed gives fractions
`50/128`, `8/128`, `70/128` and a BIC gain of 227.2.

This demonstrates path multimodality for the stated finite-size condition.  It
does not yet demonstrate that multimodality itself is unique to altermagnets:
the `J_tilde = 0` AFM control has statistically indistinguishable class
fractions (`chi-square p = 0.866`).  The next scientific test must therefore
compare spatial path mechanisms and their weights across crystal directions,
not merely repeat the near-threshold endpoint split.

See `legacy_code_comparison.md` for the physical and numerical differences
from `example/微磁学.txt`.

## Unified benchmark checkpoint

`data/datasets/training_benchmark/llg_spatial_paths.h5` now stores 180 complete
trajectories from an analytically checkable ordinary-spin system, a controlled
conventional-AFM ablation, and the published d-wave altermagnet. Each of nine
system-condition groups contains 20 paths split 16/2/2, so the global
train/validation/test counts are 144/18/18. The file is 11.6 MB and is read
lazily by `scripts/datasets/hdf5_trajectory_dataset.py`.

The ordinary benchmark agrees with the exact Langevin magnetization at three
temperatures with maximum absolute error 0.0178. All dataset checks pass and
the maximum stored spin-norm error is `5.96e-8`. This is a partially
literature-certified pilot benchmark: the Gomonay model has a published
spin-wave certificate, while the `J_tilde=0` AFM is a numerical control rather
than a separately published material parameterization. See
`RESULTS_AND_USAGE.md` for the frequency derivation, figure-reading guide,
data schema, and interpretation limits.

## Finite-temperature noise convention

The production stochastic LLG uses an isotropic, zero-mean magnetic field
`B_th` with

`<B_mu,i(t) B_nu,j(t')> = [2 alpha k_B T / (gamma mu_s)]
delta_mu,nu delta_i,j delta(t-t')`.

For a time step `dt`, every Cartesian component is therefore sampled with
standard deviation
`sqrt[2 alpha k_B T / (gamma mu_s dt)]`.  If the energy-like random variable
is defined as `R = mu_s B_th`, the same convention becomes

`<R_mu,i(t) R_nu,j(t')> = 2 alpha (mu_s/gamma) k_B T
delta_mu,nu delta_i,j delta(t-t')`.

The continuum factor `a^2 delta(r-r')` reduces to `delta_i,j` for one spin per
cell of area `a^2`.  The commonly written coefficient `2 alpha hbar k_B T a^2`
is recovered only when the angular momentum per normalized lattice spin obeys
`mu_s/gamma = hbar`; it must not be substituted independently of the equation
of motion and magnetic-moment convention.  The numerical sampler was already
using the field form above, so existing trajectories do not require
regeneration.

`scripts/validation/validate_configuration_noise_response.py` is a diagnostic
that resolves the one-step response into global x, y and z noise directions.
It is not a production thermal bath: production data always uses all three
isotropic components.  The diagnostic output is stored in
`data/research/configuration_noise_response.json`.

## Next-stage literature path validation

The stable noncollinear benchmark is now the published monoaxial CrNb3S6 model,
not the earlier imposed spiral. Run:

```bash
bash run_zrs_mag.sh altermagnetism_LLG/scripts/literature/validate_crnb3s6_helix.py
bash run_zrs_mag.sh altermagnetism_LLG/scripts/literature/validate_bauer2011_chain.py
bash run_zrs_mag.sh altermagnetism_LLG/scripts/literature/generate_bauer2011_paths.py
bash run_zrs_mag.sh altermagnetism_LLG/scripts/analysis/analyze_spatial_path_mechanisms.py altermagnetism_LLG/data/literature/path_literature_validation/bauer2011_accelerated_pilot_paths.h5
bash run_zrs_mag.sh altermagnetism_LLG/scripts/visualization/plot_next_stage_validation.py
```

The CrNb3S6 parameter-predicted discrete helix period is 48.46 nm versus the
published 48 nm. The Bauer-chain optimized domain-wall energies agree with its
continuum reference to 0.083% (`K/J=0.01`) and 0.861% (`K/J=0.1`). The included
40-path Bauer file is deliberately an accelerated mechanism pilot; it is not a
quantitative reproduction of the paper's 750000 hbar/J Fig. 2 trace.

See `RESEARCH_STATUS.md` for the novelty boundary and unresolved production
gates, and `MODEL_SPEC.md` for the concrete periodic equivariant flow-matching
architecture.

## Size-scalable data and model smoke test

`scripts/generation/generate_scalable_hdf5.py` is the schema-v2 generator. It
stores independently replayable trajectory seeds, explicit initial states,
dimensionless scalar conditions, vector/crystal-frame conditions, spatial/time
chunks and exact 8:1:1 whole-trajectory splits. Use
`scripts/validation/certify_scalable_dataset.py` as the final data gate.

`data/research/scalable_paths/software_smoke_16x16.h5` is retained as a lightweight
interface fixture.  The current CUDA smoke gate is
`slurm/gpu_smoke.sbatch`; it tests both model types and an L96 inference pass
inside a Slurm GPU allocation.  Smoke results are not physical evidence.

`data/research/size_convergence/pilot_spatial_convergence.json` records a 0.1-ps,
10-path pilot over 8/16/32/48 cells and three time steps. It passes the tested
dt and late-size observable tolerances but observes no reversal or nonuniform
path and has too few paths, so its status is correctly `not_certified`.

## Phase-1 standard dataset

The corrected schema-v2 suite is under `data/datasets/standard_v2/`.  It contains 330
unbiased 1-ps paths at 5 K and 0.70/0.78/0.80 T: L16/L32/L64 use condition-wise
8:1:1 splits, and all L96 paths are reserved for the large-size test.  The
cross-file certificate is `data/datasets/standard_v2/suite_certification.json`; its
status is `ready_for_phase1_model_training`.  It verifies exact file hashes,
global seed uniqueness, protocol consistency, split isolation and nonuniform
path coverage in both L64 training data and L96 test data.

Read `data/datasets/standard_v2/DATASET_CARD.md` before training.  The suite is suitable
for phase-1 model development, not yet for percent-level rare-event estimates
or a definitive physical mechanism phase diagram.

## Path-distribution evaluation and v2 training

Endpoint sign is no longer reported as a completed reversal. The common evaluator
in `scripts/analysis/evaluate.py` reports `negative_endpoint`, `crossed_zero`,
`committed_switch`, `crossing_return` and `unresolved_transition`.
Stable-basin thresholds are fitted from the terminal residence distribution of
real train/validation LLG paths. A condition with more than 20% unresolved real
paths fails the observation-window gate rather than receiving a forced label.
Each evaluation stores every generated spin path in HDF5 together with endpoint
histograms, each per-path `n_z(t)` curve with its ensemble mean, per-path spatial
standard deviations and animations.

The original checkpoint fails already on the L16 full test distribution:
all generated size-condition groups have zero committed switches and are
unresolved at the endpoint. The 16/32/64/128-step diagnostic also shows that
32 integration steps are not converged; 64 is closer and formal evaluation uses
128. At fixed initial state and latent seed the generated paths are effectively
unchanged across 0.70/0.78/0.80 T, while 16 latent seeds remain diverse. This
locates the primary failure in condition use/training rather than L96-only size
generalization or simple latent mode collapse.

Architecture version 2 standardizes all ten scalar conditions using the training
split, applies temperature/damping/drive FiLM modulation in every residual block,
and replaces spatial GroupNorm with per-site channel normalization. Its training
schedule uses complete L16 and L32 paths, 48x48 L64 crops, batch size one and
gradient accumulation. Formal runs use three independent training seeds and
common evaluation latent seeds; summary tables report the sample mean and sample
standard deviation. The current data
contain only one temperature and damping value, so they cannot establish
temperature- or damping-generalization even though those variables are wired
into the model.

## Methodological tooling

Scientific-figure integrity and accessibility review followed Kassis et al.,
*Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents* (2026), [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
