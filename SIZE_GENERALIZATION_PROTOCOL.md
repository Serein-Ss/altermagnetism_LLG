# Small-to-large LLG path protocol

## What “minimum size” means

There are three different minima. The two-sublattice Hamiltonian can be evaluated
on a tiny periodic cell, a 3x3 cell avoids aliasing the plus/minus neighbor stencil,
but neither is large enough to represent the target reversal physics. Gomonay et
al. report a domain-wall width of about 4 nm; with `a0=0.448 nm` this is 8.93
lattice spacings. A periodic cell contains a wall pair, and requiring roughly two
wall widths of bulk domain per wall gives `L >= 4 delta`, or 36 cells. Therefore
48x48 is the first candidate and 64x64 is the recommended production anchor.

## Required ladder

| Role | Sizes | Allowed conclusion |
|---|---:|---|
| software/local pretraining | 16, 32 | local fluctuations and size-agnostic code work |
| mechanism training | 48, 64 | only after all convergence gates pass |
| strict extrapolation test | 96, 128 | no fine-tuning or condition reselection |

An 8/16-only model may execute at 128, but cannot be credited with discovering a
wall mechanism absent from its training support.

## Convergence gate

For each final physical condition:

1. Generate the same physical duration at `dt`, `dt/2`, and a coarser screening
   step. Require switching-fraction change <=0.05 and peak-spatial-std relative
   change <=10%; use at least 100 paths because this is weak convergence.
2. Save a fine-cadence path once, subsample it by factors two and four, and require
   identical mechanism labels plus first-passage error no larger than one coarse
   saved interval.
3. Compare consecutive sizes, at minimum 48/64/96. Require switching-fraction
   change <=0.05 and peak-spatial-std relative change <=10% for the final pair.
4. Require at least three nonuniform switched paths in the screening ensemble and
   then determine the production count from a Wilson/binomial interval for the
   rarest class. Three events is only a detection gate, not a precise probability.
5. Repeat the accepted condition with an independent master seed and scan the
   spatial-std threshold around 0.25 before freezing labels.

The current short pilot fails steps 3-5 because it used ten paths, stopped at
48x48 and 0.1 ps, and observed no switch. This is a useful negative result: the
existing protocol must not be silently promoted into a mechanism-training set.

## Production commands

Run the convergence scan first, then plot it in a separate process:

```bash
bash run_zrs_mag.sh altermagnetism_LLG/scripts/validation/scan_spatial_convergence.py --sizes 48 64 96 --paths 100 --steps 10000 --save-every 50
bash run_zrs_mag.sh altermagnetism_LLG/scripts/visualization/plot_spatial_convergence.py
```

After the JSON says `certified_for_nonuniform_path_training`, generate one file
per size so a failed large run cannot corrupt smaller completed data:

```bash
bash run_zrs_mag.sh altermagnetism_LLG/scripts/datasets/generate_scalable_hdf5.py --size 48 48 --paths-per-condition 100
bash run_zrs_mag.sh altermagnetism_LLG/scripts/datasets/generate_scalable_hdf5.py --size 64 64 --paths-per-condition 100
```

Train with fixed periodic crops or same-size batches and reserve 96/128 for direct
comparison against newly generated LLG ensembles. All model checkpoints and samples
belong under `outputs/`; all solver trajectories remain under `data/`.
