# Reassembling split Git LFS data

Four completed HDF5 files exceed the 2 GB per-object limit of GitHub Free and
Pro. They are stored losslessly as 1800 MiB Git LFS parts. From the repository
root, reconstruct them with:

```bash
cat data/standard_v2_extended_3ps/test_large/d_wave_altermagnet_L96.h5.part-* \
  > data/standard_v2_extended_3ps/test_large/d_wave_altermagnet_L96.h5
cat data/standard_v2_extended_2ps/test_large/d_wave_altermagnet_L96.h5.part-* \
  > data/standard_v2_extended_2ps/test_large/d_wave_altermagnet_L96.h5
cat outputs/diagnostics_v2/current_checkpoint/L96/generated_trajectories.h5.part-* \
  > outputs/diagnostics_v2/current_checkpoint/L96/generated_trajectories.h5
cat outputs/diagnostics_v2/current_checkpoint_128/L96/generated_trajectories.h5.part-* \
  > outputs/diagnostics_v2/current_checkpoint_128/L96/generated_trajectories.h5
```

Verify the reconstructed files and individual parts against
`LFS_SPLIT_MANIFEST.json`. The unsplit originals remain on the source server;
the split representation changes storage only and is byte-for-byte reversible.
