# Reassembling split Git LFS data

The eight completed HDF5 files listed in `LFS_SPLIT_MANIFEST.json` are stored
losslessly as 1800 MiB Git LFS parts. Unsplit originals remain on the source
server and are ignored by Git. The manifest records source and part SHA-256
checksums and byte counts. Download LFS content before reassembly:

```bash
git lfs pull
```

From the repository root, reconstruct only a missing original using the
corresponding command below. These commands write the original file path.

```bash
cat data/datasets/standard_v2_extended_2ps/test_large/d_wave_altermagnet_L96.h5.part-* > data/datasets/standard_v2_extended_2ps/test_large/d_wave_altermagnet_L96.h5
cat data/generated/diagnostics/diagnostics_v2/current_checkpoint/L96/generated_trajectories.h5.part-* > data/generated/diagnostics/diagnostics_v2/current_checkpoint/L96/generated_trajectories.h5
cat data/generated/diagnostics/diagnostics_v2/current_checkpoint_128/L96/generated_trajectories.h5.part-* > data/generated/diagnostics/diagnostics_v2/current_checkpoint_128/L96/generated_trajectories.h5
cat data/datasets/standard_v2_extended_3ps/test_large/d_wave_altermagnet_L96.h5.part-* > data/datasets/standard_v2_extended_3ps/test_large/d_wave_altermagnet_L96.h5
cat data/datasets/standard_v2_extended_3ps/train/d_wave_altermagnet_L64.h5.part-* > data/datasets/standard_v2_extended_3ps/train/d_wave_altermagnet_L64.h5
cat data/generated/production/production_v2/seed_20260908/evaluation/L96/generated_trajectories.h5.part-* > data/generated/production/production_v2/seed_20260908/evaluation/L96/generated_trajectories.h5
cat data/generated/production/production_v2/seed_20260909/evaluation/L96/generated_trajectories.h5.part-* > data/generated/production/production_v2/seed_20260909/evaluation/L96/generated_trajectories.h5
cat data/generated/production/production_v2/seed_20260910/evaluation/L96/generated_trajectories.h5.part-* > data/generated/production/production_v2/seed_20260910/evaluation/L96/generated_trajectories.h5
```

Verify the reconstructed files against each `source_sha256` in the manifest
(e.g. `sha256sum PATH.h5`). Each part also has its own checksum. The split
representation changes storage only and is byte-for-byte reversible.

The new 2026-09-09 files are the L64 three-picosecond training set and the L96
generated ensembles for all three V2 training seeds. Smaller HDF5 audit files,
checkpoints and figures are tracked directly through Git LFS.
