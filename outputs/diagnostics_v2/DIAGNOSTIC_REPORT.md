# Current-checkpoint diagnostic report

This report uses the existing architecture-v1 checkpoint only. No result below
comes from retraining.

## Outcome semantics

- `negative_endpoint`: the final global Neel-z is below zero.
- `crossed_zero`: the path crossed zero at least once.
- `committed_switch`: after crossing zero, the path stayed inside the
  data-calibrated basin opposite its initial basin throughout the final 0.10 ps.
- `crossing_return`: after crossing zero, the path stayed inside its original
  data-calibrated basin throughout the final 0.10 ps.
- `unresolved_transition`: the path crossed or left its original basin but was
  not resident in either resolved basin at the observation endpoint.

The 1 ps real train/validation terminal distribution selects a three-component
fit with thresholds -0.6833 and +0.7802.

## Full test distributions at 128 integration steps

| L | H (T) | ref negative endpoint | flow negative endpoint | ref committed | flow committed | ref unresolved | flow unresolved | endpoint W1 | path RMSE | spatial RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.70 | .333 | .333 | .333 | .000 | .000 | 1.000 | .520 | .374 | .118 |
| 16 | 0.78 | .333 | .667 | .333 | .000 | .000 | 1.000 | .815 | .259 | .109 |
| 16 | 0.80 | .667 | .333 | .667 | .000 | .333 | 1.000 | .412 | .233 | .112 |
| 32 | 0.70 | .000 | .667 | .000 | .000 | .000 | 1.000 | 1.002 | .653 | .160 |
| 32 | 0.78 | 1.000 | 1.000 | .667 | .000 | .333 | 1.000 | .585 | .330 | .150 |
| 32 | 0.80 | 1.000 | 1.000 | 1.000 | .000 | .000 | 1.000 | .526 | .462 | .157 |
| 64 | 0.70 | .000 | 1.000 | .000 | .000 | .000 | 1.000 | 1.057 | .636 | .195 |
| 64 | 0.78 | .000 | 1.000 | .000 | .000 | 1.000 | 1.000 | .754 | .289 | .084 |
| 64 | 0.80 | .667 | 1.000 | .000 | .000 | 1.000 | 1.000 | .394 | .159 | .084 |
| 96 | 0.70 | .000 | 1.000 | .000 | .000 | .000 | 1.000 | 1.054 | .613 | .195 |
| 96 | 0.78 | .400 | 1.000 | .100 | .000 | .800 | 1.000 | .496 | .161 | .044 |
| 96 | 0.80 | .950 | 1.000 | .750 | .000 | .250 | 1.000 | .621 | .329 | .103 |

L16/L32/L64 have three held-out paths per field and therefore noisy empirical
fractions; L96 has twenty per field. The failure is nevertheless categorical:
the flow produces no committed switch and leaves every generated path unresolved
in every size-field group. Failure therefore starts at L16 and cannot be
attributed to L96-only size generalization.

## Three pretraining diagnostics

- Integration: paired differences from 128 steps are endpoint/path
  0.1947/0.0910 at 16 steps, 0.1066/0.0485 at 32, and 0.0414/0.0186 at 64.
  Thirty-two steps are not converged; formal evaluation uses 128.
- Condition sensitivity: with fixed initial state and latent seed, changing
  0.70/0.78/0.80 T gives endpoint span 0.000441 and mean pairwise path RMSE
  0.000151. The old model effectively ignores drive.
- Same-initial diversity: sixteen latent seeds give endpoint standard deviation
  0.3806 and mean pairwise path RMSE 0.2599. Simple latent mode collapse is not
  detected.

The primary diagnosis is a conditional-model/training-target failure, with
sampling discretization error as a secondary issue. Receptive-field effects may
still worsen larger sizes, but they are not the first failure.
