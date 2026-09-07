# Size-scalable path flow model

This folder contains the first executable model for conditional generation of full
finite-temperature spin trajectories. It is fully convolutional in time and space,
uses circular padding on the periodic lattice, and predicts tangent velocities on
the product of spin spheres.

Each frame also receives its normalized physical time and a pulse-active mask
constructed from `pulse_fraction`.  The model can therefore distinguish the driven
and post-pulse relaxation portions of a trajectory; both are rotation-invariant
scalars and do not weaken the SO(3) covariance described below.

The vector basis includes the exact dimensionless effective-field stencil of the
stored Gomonay Hamiltonian (four inter-sublattice bonds, axial `J2`, signed diagonal
`Jtilde`, and crystal-axis anisotropy). The default eight residual blocks give an
18-cell scalar receptive-field radius, about two published wall widths.

`PeriodicEquivariantFlowNet` is exactly covariant under joint proper rotations of
spins and vector conditions because it combines covariant vector bases using scalar
invariant coefficients. It is also periodic-translation equivariant and accepts any
`Nx x Ny` at inference. It is not reflection-equivariant and it does not yet contain
a global magnetostatic operator. Those limitations are explicit because the current
published Gomonay Hamiltonian is local and does not include dipolar interactions.

Recommended workflow:

1. Run `scripts/validation/scan_spatial_convergence.py` and do not train mechanism
   claims until its gate passes with at least 100 paths per condition.
2. Generate separate schema-v2 files with `scripts/datasets/generate_scalable_hdf5.py`.
3. Pretrain local fluctuations on 16/32-cell crops, but include converged 48/64-cell
   trajectories before claiming nonuniform mechanism learning.
4. Train with `bash run_zrs_mag.sh -m altermagnetism_LLG.model.train --input FILES...`.
5. Generate 96/128-cell held-out samples with `bash run_zrs_mag.sh -m altermagnetism_LLG.model.sample`.
6. Compare distributions against new LLG ensembles at exactly the same large sizes.

The software can run on a larger lattice without retraining; this is a software
property, not evidence of physical generalization. Physical generalization requires
held-out large-size LLG closure for path classes, first-passage times, correlations,
structure factors, and wall statistics.

## Server workflow

Variable lattice sizes are loaded lazily and grouped into homogeneous-shape
batches.  Passing `--crop-size` intentionally allows source sizes to mix after
the same periodic crop has been applied.

Run the end-to-end CUDA smoke gate through Slurm:

```bash
sbatch altermagnetism_LLG/slurm/gpu_smoke.sbatch
```

After it passes, submit the deterministic baseline, conditional Riemannian flow
model, in-distribution test and held-out L96 evaluation:

```bash
sbatch altermagnetism_LLG/slurm/train_and_evaluate.sbatch
```

