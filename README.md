# Rotating-shield simulation runtime

This repository is the common simulation and observation-generation boundary for
the rotating-shield estimators. It owns Geant4, environments, source realization,
detector and Fe/Pb shield geometry, spectrum generation, and raw MeasurementLog
publication. It contains no PF, MLE, or hybrid estimator.

```text
Rotating-shield-simulation-runtime
  SimulationCommand -> Geant4 -> raw integer spectrum -> MeasurementLog
                                           |
                  +------------------------+------------------------+
                  |                        |                        |
      Rotating-shield-particle-filter  radiation-surface-mle-estimator  estimator orchestrator
```

Each estimator-controlled mission gets a separate causal acquisition session so its
controller can choose the next position and shield posture from the observations seen
so far. This repository supplies the same simulator implementation and physics
configuration to every live session.

## Commands

```bash
uv sync --extra test
uv run rotating-shield-sim validate-log PATH
uv run rotating-shield-sim serve --config configs/geant4/variance_reduction_external_no_isaac_32threads.json
uv run rotating-shield-sim run-adaptive-session PRIVATE_SCENARIO.json
uv run rotating-shield-sim serve-adaptive-session-socket PRIVATE_SCENARIO.json \
  --socket-path /run/user/1000/rotating-shield/run-001.sock
uv run rotating-shield-sim generate-scenario PRIVATE_SCENARIO.json \
  --truth-manifest-output /private/truth/run-001.json \
  --measurement-log-output /private/logs/run-001 \
  --run-id run-001 \
  --experiment-profile multi_isotope_surface_search \
  --scene-variant mix9
uv run rotating-shield-sim calibrate-discrepancy CALIBRATION_ROWS.npz \
  discrepancy-calibration.json --calibration-id ral-independent
```

The shared geometry-conditioned spectrum model exposes exact NumPy and Torch
predictive samplers. Its Torch path keeps renewal/Gamma-Poisson totals, calibrated
energy marks, and integer spectra on the caller's device and supports canonical
per-action seeds, allowing PF planning to continue directly into the Torch cross
likelihood without a bulk device-to-host round trip.

`run-adaptive-session` is the closed-loop boundary. Its private scenario contains
the realized sources, environment/obstacles, physical runtime configuration, and
MeasurementLog destination, but no action array, station count, view count, route,
or shield program. It publishes reachable truth-free candidates, accepts exactly one
controller selection over JSON lines, durably stages that observation, and repeats
until the controller requests finalization.
The private scene variant is read and validated entirely inside the runtime; variant
IDs, scene seeds/RNG provenance, counts, and realized source data are not included in
estimator-visible events or MeasurementLog. The public experiment profile and its
truth-free acquisition contract are included so every controller uses identical
station, view, dwell, separation, and coverage limits. The runtime rejects dwell,
station-order, view-boundary, and measurement-limit drift before transport and
publishes only nonempty logs ending at a complete station boundary.
`serve-adaptive-session-socket` exposes only an owner-accessible Unix socket, so an
estimator process does not receive the private scenario path.

Adaptive acquisition is fresh-run only. Any controller disconnect, simulator error,
or explicit abort closes transport, removes the private write-ahead stage, and
publishes no partial MeasurementLog. There is no production resume or prefix-replay
entrypoint; a failed acquisition must start again with a new output destination.
Estimator clients attach only to the opaque socket with
`AdaptiveRuntimeClient.connect(...)`.

`generate-scenario` creates that private, action-free scenario and a separate
owner-readable truth manifest joined to completed estimator output by `run_id`.
Omitting
`--scene-seed` creates a fresh environment and source realization; an explicit seed
is reserved for reproducing a previously declared validation scene. The command
publishes the runtime-owned acquisition contract but does not choose a shield policy,
estimator setting, or statistical stopping rule. The command has no default
environment. Selecting `--experiment-profile cs4_co3_surface_search` with
`--scene-variant cs4-co3` realizes exactly four Cs-137 and three Co-60 sources and
exposes only those two candidate isotopes. The separately selected
`multi_isotope_surface_search` / `mix9` experiment adds two Eu-154 sources.

## Common adaptive workspace

Adaptive candidate generation is estimator-neutral. The runtime draws a nested,
scrambled Sobol sequence directly in collision-free `(x, y, z)` free volume; it is
not a floor grid with a list of heights attached. The initial pose and requested
same-XY height anchors are retained, and a controller may ask the runtime to refine
the neighborhood of selected candidate indices. Local refinement is still generated,
collision-checked, and timed by the runtime—the controller supplies only seeds.

The free-volume check models the detector head, mast, and base separately. Motion
costs separately account for horizontal travel, retract/extend vertical travel,
settling time, shield angular actuation, and dwell. Candidate events therefore expose
only reachable poses and common physical costs; PF and MLE remain responsible for
their own ranking objectives. Increasing `candidate_count` preserves the earlier
Sobol prefix, which makes candidate-density convergence tests reproducible.

The runtime does **not** contain a PF candidate generator and an MLE candidate
generator. It has one physical workspace and one acquisition protocol shared by all
estimators. Likewise, it never contains estimator likelihoods, posterior state,
regularization, stopping rules, or planner scores.

This is an interface commonality, not a requirement that estimators use the same
configuration or take the same actions. A PF-controlled and an MLE-controlled run
normally create separate causal sessions and may use different station programs,
budgets, and stopping times while relying on the same runtime implementation.

## Shared estimator-neutral Python APIs

Estimator repositories consume runtime-owned wire and artifact mechanics without
moving their algorithms into this package:

- `ResolvedForwardContext.from_log(...)` and `.from_run_context(...)` authenticate
  and construct the environment, obstacles, spectrum model, and observation model
  once. PF particles and MLE surfaces remain in their own repositories.
- `AdaptiveRuntimeClient.connect(...)` exposes typed `handshake()`, `acquire()`,
  `refine_candidates()`, and `finalize_log()` calls. It is a context manager with
  bounded termination and an optional immutable protocol observer. This client is
  estimator-facing and exposes no CUI-truth request method. A retired private-overlay
  frame on the estimator stream is a fatal protocol error rather than a compatibility
  path.
- `candidate_index_for_pose(...)` requires an `AdaptiveCandidateSnapshot`, so raw
  unvalidated candidate mappings cannot enter controller decisions.
- `AdaptiveCandidateSnapshot.quote_shield_program_time_s(...)` quotes the exact
  sequential Fe/Pb actuation time from the current shield state before a station is
  executed, using the same parallel-actuator timing function as acquisition.
- `MeasurementLog.view()`, `MeasurementLog.prefix_view(...)`, and
  `MeasurementLogView.from_records(...)` provide read-only array and station views
  without synthesizing manifests. File-backed logs additionally expose an
  authenticated artifact inventory. Live record prefixes bind their covered records
  to an algorithm-qualified digest without publishing a second log bundle.
- `CUIRoute`, `CUIScene`, and the responsive CUI shell standardize route,
  obstacle, URL, and page structure. Each estimator supplies its own result panel
  specifications and renders particles, grids, density surfaces, or combined plots
  inside those panels.
- `CUITruthOverlaySocketServer` is a separate, mode-`0600`, single-response owner
  channel for an evaluation renderer. It is enabled only by an explicit
  `--cui-truth-overlay-socket-path`; its payload never enters adaptive events,
  MeasurementLog, or `AdaptiveRuntimeClient`.
- `cui_scene_from_run_context(...)` resolves embedded or root-confined file-backed
  obstacle geometry from a truth-free `RunContext` without constructing a spectral
  response model.
- `AtomicBundlePublisher`, `DurableJSONLWriter`, `ArtifactInventory`, strict JSON,
  digest identities, and CLI JSON framing provide versioned mechanical contracts.

Legacy bare SHA-256 fields remain readable, but new cross-repository data should
carry the corresponding algorithm identifier so different historical digest
normalizations cannot be compared as if they were equivalent.

## Shared discrepancy calibration contract

`calibrate-discrepancy` fits a versioned, estimator-neutral spectral discrepancy
artifact from independent calibration environments. Its strict NPZ input contains
exactly:

- `observed_counts` and corresponding physics-model `expected_counts` with shape
  `N x B`;
- `energy_bin_edges_keV` with shape `B + 1`;
- `environment_ids` with at least two distinct independent realizations;
- `shield_pair_ids` covering every Fe/Pb pair `0..63`.

The published JSON contains background/scatter bases, low-dimensional all-pair shield
leakage features, station-rate and low-rank residual families, gain/resolution drift
derivatives, shrinkage strengths, and calibrated negative-binomial overdispersion.
Incomplete pair coverage or non-independent environments fail closed. The artifact
is common physical/calibration input: an estimator chooses whether and how to include
these nuisance columns in its own likelihood. Runtime calibration never imports or
calls PF or MLE code.

## Citation and license

If this software contributes to research, use the metadata in
[`CITATION.cff`](CITATION.cff) to cite the exact software repository. Citation is
a scholarly request, not an additional license condition. Repository-authored
software and documentation are released under the [MIT License](LICENSE);
third-party dependencies and externally sourced data retain their own terms.
