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

For a fair same-observation comparison, run acquisition once and replay the exact
same MeasurementLog with every estimator. For estimator-controlled closed-loop
missions, each estimator gets a separate causal acquisition session, while this
repository still supplies the identical simulator implementation and physics
configuration.

## Commands

```bash
uv sync --extra test
uv run rotating-shield-sim validate-log PATH
uv run rotating-shield-sim serve --config configs/geant4/variance_reduction_external_no_isaac_32threads.json
uv run rotating-shield-sim run-plan PLAN.json
uv run rotating-shield-sim run-adaptive-session PRIVATE_SCENARIO.json \
  --private-scene-profile ral-mix9
uv run rotating-shield-sim run-adaptive-session PRIVATE_SCENARIO.json \
  --resume-stage /private/logs/.run-001.stream-1234
uv run rotating-shield-sim generate-ral-scenario PRIVATE_SCENARIO.json \
  --measurement-log-output /private/logs/run-001 \
  --run-id run-001 \
  --runtime-config configs/geant4/variance_reduction_external_no_isaac_32threads.json \
  --source-profile ral-mix9
uv run rotating-shield-sim calibrate-discrepancy CALIBRATION_ROWS.npz \
  discrepancy-calibration.json --calibration-id ral-independent
```

`run-plan` reads source truth only from the private plan, durably records each raw
observation before returning it to a controller, and never writes source truth into
MeasurementLog.

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
The optional `ral-mix9` private-scene profile validates Cs-137 x4, Co-60 x3, and
Eu-154 x2 entirely inside the runtime; the counts and realized source data are not
included in estimator-visible events.

An interrupted adaptive acquisition can resume from its hidden stream stage with
`--resume-stage`. The runtime authenticates the static acquisition identity and
every durable record shard, discards an incomplete station tail, copies through the
last `station_complete` boundary, restores pose/yaw/Fe/Pb state, and returns the
adopted truth-free prefix in a canonical ready event. The original stage is never
modified. Resume under a different runtime commit fails closed unless
`--resume-compatibility COMPATIBILITY.json` supplies explicit provenance. The same
surface is available to Python callers through `AdaptiveRuntimeSession.resume(...)`
and `AdaptiveRuntimeClient(..., resume_stage_path=...)`.

`generate-ral-scenario` creates that private, action-free scenario. Omitting
`--scene-seed` creates a fresh environment and source realization; an explicit seed
is reserved for exact replay or paired estimator comparisons. The command does not
choose station count, view count, shield programs, estimator settings, or a stopping
rule. Those remain private to the estimator or experiment harness controlling the
session. `--source-profile ral-cs4-co3-eu0` defines both the private truth and the
truth-free candidate contract as Cs-137 plus Co-60: it realizes exactly four
Cs-137 and three Co-60 sources, and Eu-154 is not offered to the estimator. Use
`ral-mix9` when Eu-154 must remain in the candidate set.

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
