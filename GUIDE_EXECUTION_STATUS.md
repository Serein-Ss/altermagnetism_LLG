# GUIDE execution status — 2026-09-09

> Historical submission record. Its old three-plan basis has been superseded by `GUIDE/STRICT_LITERATURE_REPRODUCTION_PLAN.md`. Current R0 workflow and artifact paths: [Literature reproduction](LITERATURE_REPRODUCTION.md). Old Nishino numerical outputs are subject to the new plan's explicit cleanup; this record is not a current pass certificate.

The three GUIDE plans are **not fully implemented or completed**. This submission
executes the prerequisite noise and free-moment integrator certification stage.
No new production dataset or model-training job is submitted by this workflow.

## Verified before submission

- 44 tests passed in a Slurm fat CPU allocation, including midpoint residual and
  norm preservation, Brownian coupling, original-Heun equivalence, chain diagnostics,
  JSON/HDF5 writing, and missing-evidence refusal.
- Twenty million-component noise conditions passed the mean, variance,
  cross-component/site/consecutive-time independence, replay, dt-scaling and zero-T
  checks through the actual solver noise interface.
- The initial reduced Nishino I/O test exposed a NumPy-bool serialization error.
  The error was fixed and covered by an end-to-end regression test. Reduced data
  remain explicitly non-certifiable; failed pilot artifacts were retained.
- Slurm batch syntax checked. Environment: zrs-mag. CPU only.

## Submitted jobs

All jobs are named `zrs_data`, partition `fat`, 20 CPUs and 192 GiB per allocation.

| Job | Work | Dependency |
|---|---|---|
| 668983 | Snapshot tests, environment capture, thermal-noise audit | none |
| 668984 | First full Nishino run; measured runtime budget gate | afterok 668983 |
| 668985, array 1–6, concurrency 1 | Remaining 119 independent runs; up to 20 one-thread processes per allocation | afterok 668984 |
| 668986 | Separate all-run completeness/hash/statistical audit | afterany 668985 |

The first full run measures 10,000 coarse steps and refuses continuation when a
1.5 runtime margin exceeds the two-day allocation. Subsequent runs apply the same
check. A failed prerequisite cancels invalid dependents; there is no automatic
resubmission. A successful executable exit is not a physical certificate.

Nishino settings follow GUIDE: T=0.5,1.0,...,6.0, ten independent runs per T,
1000 free moments per run, total dimensionless duration 400, preparation 200,
dt=0.005/0.0025/0.00125, Heun and implicit midpoint, coupled Brownian increments.
Nishino's nonzero static field is the analytic thermal-benchmark condition; it is
not an external drive in a new altermagnet training dataset.

Outputs and code snapshot:
`data/audit/20260909_3b3b824_guide_thermal/`.
`submission.json` records job IDs, source hashes, base commit and dirty state.
The jobs execute the snapshot, not subsequent working-tree changes.
The initial queue check confirmed acceptance. At the user's subsequent status
request on 2026-09-09, Slurm showed all generation tasks completed successfully.
Job 668986 exited 2: the separate ensemble gate did not pass. The existing report
contains 120 runs, no failed marginal-distribution/stationarity/norm checks, and
24 failed paired comparisons out of 72. `production_enabled` remains false.
No continuing monitoring is configured. These are status/report observations,
not a completed diagnosis of the failed comparison criteria.

## Other code added, not production-certified

- `scan_neel_temperature.py`: four-chain zero-field temperature pilot, complete
  saved spatial states and equilibrium observables. Requires a passed thermal gate.
- `analyze_neel_temperature.py`: even-observable rank/folded split Rhat and ESS,
  Binder/susceptibility/heat-capacity summaries; does not self-certify TN.
- `build_equilibrium_initial_pool.py`: decorrelated positive-basin candidate
  extraction and initial-group split. Does not self-certify the equilibrium pool.

## Unfinished gates and implementation

1. Interacting-Hamiltonian coupled-dt and geometric-integrator audits, independent
   parameter/neighbor/term-wise field certificates, complete zero-T controls.
2. Refined TN grid with block-bootstrap Binder intersections and uncertainty;
   certified equilibrium pool, basin-distribution thresholds and size convergence.
3. Zero-field fixed-initial-state/reversal generator, committed-event detector,
   critical-disordering discrimination, survival statistics, mechanism validation,
   cadence/duration convergence and frozen production protocol.
4. Full Gomonay multi-wavevector/window convergence, Bauer timed full-lifetime
   campaign and censored event statistics; Mn2Au parameter/3D-neighbor audit and
   implementation remain blocked, not replaced by Jtilde=0.
5. Independent production metadata/split/hash audit, all requested figures and
   animations, V3 model-loader adaptation and eventual model training.

No TN, production temperature, basin threshold or observation window has been
invented from an unfinished pilot. Resolve the differing GUIDE/old-V3 burn-in and
censoring rules in the eventual versioned protocol; retain strict certification
boundaries meanwhile. Existing V2 jobs and existing data were not changed here.
