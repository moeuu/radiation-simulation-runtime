# Shared-runtime paired all-64 phase-space native integration

The paired phase-space facility is a dedicated calibration/acceptance path. It
must not be exposed by `main.py`, the standard Geant4 application config, or
the standard observation runtime. Reusing one upstream bank preserves each
shield pair's marginal transport mean, but the 64 counterfactual results share
upstream histories and therefore are not 64 independent observations.

The dependency-free native core is:

- `native/geant4_sidecar/paired_all64_phase_space.hpp`
- `native/geant4_sidecar/paired_all64_phase_space.cpp`

The Geant4-facing integration is:

- `native/geant4_sidecar/paired_all64_geant4.hpp`
- `native/geant4_sidecar/paired_all64_geant4.cpp`

It implements the material-free parallel-world boundary, worker-local capture,
event-grouped replay primaries, and complete original-history score matrices.
The dependency-free core implements canonical bank serialization, SHA-256
authentication, pair- and history-stable replay seeds, a full-world replay
schedule, and exact stratified original-history covariance. It deliberately does not
alter the standard sidecar until the Geant4 capture and replay wiring below is
complete.

The current native contract is **Option B: an analog, shared, unit-weight
phase-space bank**. It is not composable with forced-collision branch weights.
Forced collision may be used by a separate estimator, but its weighted
branches cannot be captured into this bank or replayed as if they were analog
histories.

## Capture geometry

Register a `G4VUserParallelWorld` containing only a spherical detector-boundary
surface and add `G4ParallelWorldPhysics` to the same `FTFP_BERT` +
`G4EmStandardPhysics_option4` physics list used by calibration transport. A
parallel-world boundary avoids changing the material geometry or adding a mass
boundary that would perturb ordinary transport.

The capture run must contain the full static environment but omit the detector
and both movable shields. Before `BeamOn`, fail unless:

- the sphere fully encloses every possible detector and shield solid,
- every source point lies strictly outside the sphere,
- the sphere lies inside the Geant4 world,
- full secondary transport is active,
- primary sampling and track weights are exactly one,
- detector response sampling, background, and dead time are disabled.

Do not use a mass-world sphere or a reduced environment.

## Track and lineage state

Extend `TransportTrackInformation` in
`native/geant4_sidecar/geant4_sidecar.cpp` with:

- source-line index,
- branch lineage ID and parent lineage ID,
- generation,
- gamma interaction count.

The existing primary history ID, source index, interaction flag, and secondary
lineage flag are not sufficient: a zero-crossing history still needs its
source-line label, and multiple sibling branches from one primary must remain
distinct. Assign a positive branch ID at track creation and copy the original
history/source/line identity in `TransportTrackingAction` when secondaries are
created.

At `BeginOfEvent`, call
`CaptureAccumulator::RegisterHistory(history_id, source_index, line_index,
angle_stratum_index, angle_stratum_count, estimator_coefficient)` even if the
event will have no boundary crossing. The final three fields come from the
immutable fixed-quota source schedule. The coefficient is external estimator
metadata; it is authenticated in the bank and must never be assigned to a
Geant4 track. Create one `CaptureAccumulator` per Geant4 worker.

For every source-line schedule, all declared angle strata must be present and
must contain the same number of original histories. Histories are grouped by
the exact `(source, line, angle_stratum)` key. A source line, angle stratum, or
coefficient may not be reconstructed from a crossing after transport.

## First inward crossing

In `TransportSteppingAction`, inspect the parallel-world ghost pre/post step.
On a `fGeomBoundary` transition from outside to inside:

1. construct `paired_all64::Crossing` from the post-step state, including
   particle name, PDG code, dynamic mass, dynamic charge, proper time,
   polarization, and kinetic energy,
2. call `RecordFirstInwardCrossing`,
3. on `kCaptured`, set the physical track to `fStopAndKill`,
4. on `kAlreadyCaptured`, abort because a captured branch should already have
   been killed.

Ignore outward crossings during capture. Do not kill sibling tracks. Every
transported particle species crossing inward must be captured: rejecting
electrons or positrons would remove paths that emit bremsstrahlung inside the
shield assembly. A non-unit track weight, missing particle definition,
incomplete dynamic particle state, missing lineage, or source-line mismatch
must abort the entire bank rather than being omitted. The core independently
validates that the position is on the sphere, the direction is inward, the
restart state is finite, and the lineage is self-consistent.

At end of run, call `Finalize` for each worker and
`MergeWorkerBanks` once on the master. Serialize with `WriteBank` and record
`BankPayloadSha256` in the Python bank metadata contract.

## Full-world replay

Replay must reconstruct the complete original environment, detector, and
selected Pb/Fe shields. The capture sphere is only a scoring surface; it must
not become an absorbing boundary during replay.

Create one `ReplaySchedule` for one shield pair. For schedule event `i`:

- create exactly one `G4Event`,
- set the event random stream from `ReplayEvent.random_seed`,
- add every `ReplayEvent.primaries` crossing as its recorded Geant4 particle
  species and restore dynamic mass, charge, proper time, polarization, and
  kinetic energy,
- preserve position, direction, polarization, energy, time, source/line, and
  interaction lineage,
- copy source, line, angle-stratum identity, and the external estimator
  coefficient to `ReplayEventInformation`,
- permit the event to have zero primaries.

All crossings from one original primary history must stay in one `G4Event`.
Run normal full secondary transport after injection. Never kill outward
crossings or later reentries; a replay branch may leave the sphere, scatter in
the environment, and return. `ReplaySchedule::FullWorldReplayRequired()` is
always true and `KillOutwardCrossings()` is always false.

The sidecar must run each pair with the pair seed authenticated by the Python
manifest. Pair iteration order must not affect seeds or output identities.

## Exact stratified original-history covariance

Detector scoring must retain the original history ID for every replay deposit.
For each shield pair, build one dense row-major
`(history_count, feature_count)` score matrix, including explicit zero rows,
and call `PairedScoreAccumulator::SubmitPairScores` once. There is no scalar
insertion API, so an incomplete pair cannot silently look like a physical
zero.

After all 64 pairs are submitted, call `FinalizeExact`. Let group `g` be one
exact `(source, line, angle_stratum)` group, let `n_g` be its fixed quota, let
`a_g` be its authenticated external coefficient, and let `y_h` be one
history's raw replay score. The stored factor row is

```text
a_g * sqrt(n_g / (n_g - 1)) * (y_h - mean_g).
```

Therefore `F.T @ F` is the unbiased original-history covariance of
`sum_g a_g * sum_{h in g} y_h`. The artifact stores the per-group first sums,
the coefficient-weighted estimate, every centered original-history factor,
and the 64-by-64 outer-product covariance of total feature scores. Every bank
history has a factor row, including histories with no crossing and histories
whose replay score is exactly zero. Heterogeneous source, line, or angle
strata are never pooled.

The exact artifact and its SHA-256 are byte-for-byte equivalent to
`spectrum.paired_all64_phase_space.aggregate_cross_pair_stratified_covariance`.
Persist `SerializeCovarianceArtifact` beside the 64 pair results.

`FinalizeApproximateBlockDiagnostic` remains available only for coarse
convergence plots. It is labeled
`approximate_pooled_hash_block_diagnostic_v1`, may pool heterogeneous groups,
is not serializable as the exact artifact, and must never be consumed as model
covariance.

Source-line quota and branching coefficients are estimator metadata, not
Geant4 track weights. Capture crossings and replay primaries remain exactly
unit weight. A weighted boundary crossing aborts the bank.

## Minimum standard-sidecar hooks

The dedicated implementation intentionally does not edit
`geant4_sidecar.cpp`. Enabling it later requires only these explicit hooks:

1. Register `CaptureParallelWorld` on the capture detector construction and
   register `G4ParallelWorldPhysics(kParallelWorldName, false)` on the existing
   `FTFP_BERT` plus `G4EmStandardPhysics_option4` physics list.
2. Implement `CaptureIdentityProvider` from the immutable primary schedule.
   It must return source-line, angle-stratum, and external coefficient
   metadata. Extend `TransportTrackInformation` with source-line,
   branch/parent IDs, generation, and interaction count; never infer a missing
   identity and never copy the estimator coefficient into track weight.
3. Use a capture mass world with the static environment but without detector or
   movable shields. Call `ValidateCapturePreflight` before `Initialize`.
4. For replay, use `ReplayPrimaryGeneratorAction`. In
   `TransportTrackingAction`, initialize the normal track-information object
   from `ReplayPrimaryInformation` for each injected primary, then propagate it
   to secondaries exactly as normal transport does.
5. Keep the complete original environment, detector, and selected shields.
   Call `ValidateReplayPreflight`; do not register an absorbing capture surface
   in replay.
6. At the existing `EventStore` end-event boundary, submit one complete feature
   vector, including explicit zeros, to
   `ReplayPairScoreMatrixCollector`. Submit each completed pair to
   `All64ReplayScoreCoordinator`.

No hook is present in `main.py` or the standard Geant4 application config.
The implementation in this document therefore remains non-selectable by the
standard runtime.

## Sources inside the capture sphere

The first integration fails preflight when a source lies inside the boundary;
it never drops the source or records a false zero. A production calibration
orchestrator must use a hybrid exact schedule:

- outside-source histories use the shared boundary bank;
- inside-source histories run ordinary unit-weight full-world transport for
  every shield pair;
- the two original-history score matrices are concatenated before covariance;
- the manifest records shared-bank and per-pair fallback history counts.

Until that hybrid orchestration and its manifest fields exist, the dedicated
profile remains non-selectable rather than silently biasing random-surface
environments.

## Required integration tests

Before enabling the dedicated profile, add native Geant4 tests that prove:

- analog transport and capture/replay agree within declared Monte Carlo
  uncertainty for each tested pair,
- one-thread and multi-thread bank payloads have identical statistical
  semantics and complete history sets,
- a source inside the sphere fails preflight (or enters the future exact
  per-pair fallback), every transported species round-trips, and any weighted
  track fails before an artifact is marked complete,
- outward environment scattering and reentry contribute in replay,
- shuffled pair execution produces identical per-pair seeds and results,
- all 64 pair matrices are required before covariance finalization,
- exact factors match an independent within-stratum oracle, zero histories
  remain present, and heterogeneous strata are never pooled,
- the external estimator coefficient survives bank/replay authentication while
  every Geant4 crossing and replay primary remains unit weight,
- the standard runtime still has no route to this profile.
