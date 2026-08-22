# Cross-repository API and CUI consolidation roadmap

## Scope and ownership constraints

The simulation runtime remains the only production owner of environment and
obstacle generation, detector and shield geometry, transport physics,
observation generation, adaptive acquisition, and MeasurementLog publication.
It must not acquire PF, MLE, hybrid-control, posterior, or estimator-planning
logic.

The shared surface between the runtime and estimators should therefore contain
only:

- immutable acquisition records and run context;
- versioned adaptive-session messages;
- truth-isolated evaluation overlays;
- deterministic serialization and atomic artifact publication;
- estimator-neutral CUI serving, route extraction, and panel metadata.

Particle rendering, surface-density rendering, posterior summaries, solver
state, and estimator-specific planning remain in their owning repositories.

## Implemented foundation

The current worktree implements the estimator-neutral portions of phases 1–3:

- `runtime.artifacts` owns strict atomic bytes, text, JSON, and file-copy
  publication;
- legacy `canonical_json_bytes` remains byte-compatible for existing digests,
  while new schemas use fail-closed `strict_canonical_json_bytes` and
  `strict_sha256_json`;
- `runtime.measurement_log.write_deterministic_npz` is a public, atomic,
  byte-stable NPZ API;
- `runtime.defaults` is packaged, including the legacy `runtime_defaults`
  wheel shim and shared PF-style CUI host/port/path values;
- `runtime.cui` owns strict configuration, browser URLs, managed safe static
  serving, the immutable truth-free `CUIRoute`, correctly ordered `CUIScene`
  geometry, and the PF-reference five-panel HTML shell;
- `ResolvedForwardContext` owns authenticated physical-input reconstruction for
  replay and live consumers, without owning any estimator state;
- adaptive ready, record, candidate, refinement, abort, and publication messages
  are frozen DTOs, and the client owns bounded lifecycle plus transcript observing;
- `MeasurementLogView.from_records` owns live array/station conversion, while
  file-backed logs expose generic inventories and pathless prefix views;
- digest identities and golden fixtures distinguish the runtime v2 record digest
  from the orchestrator's historical compatibility normalization;
- bundle publication, durable JSONL append, strict JSON/config identity, forward
  fixture parsing, and prefixed CLI JSON framing are shared mechanical APIs.

No phase-1 extraction changes transport physics, detector response, source
truth boundaries, observation statistics, or estimator algorithms.

## Starting duplication and target owner

| Concern | Current duplication | Target owner |
| --- | --- | --- |
| Canonical JSON and repository provenance | PF now delegates; MLE/orchestrator retain repository adapters | runtime public API, with repository-root adapters |
| Atomic bytes, JSON, latest-image, and bundle publication | Consumers retain only format-specific byte encoders | runtime public API |
| Deterministic NPZ writing | PF now uses the public API; orchestrator still has legacy formats | runtime public API |
| Static HTTP serving and browser URL construction | PF, MLE, and orchestrator delegate | runtime CUI API |
| Public-host discovery and occupied-port handling | PF, MLE, and orchestrator delegate | runtime CUI API |
| Route, station-view, waypoint, scene, and page shell | all CUI consumers use PF-compatible contracts | runtime truth-free CUI DTOs |
| Adaptive event dictionaries and process lifecycle | typed consumers retain only controller decisions | typed runtime adaptive-session API |
| MeasurementLog v2 validation and prefix writing | runtime and orchestrator | runtime; orchestrator keeps only legacy readers |
| PF/MLE solver and posterior state | PF/MLE repositories and orchestrator | estimator owners; no runtime dependency reversal |

## Canonical CUI contract

The live Particle Filter split view is the compatibility baseline. A CUI
implementation should provide a directly clickable URL and refresh completed
image files every two seconds. The standard filenames and ordering are:

1. `latest_experiment_overview.png`
2. `latest_robot_2d.png`
3. the estimator-specific 3-D result (`latest_pf_3d.png` or
   `latest_mle_3d.png`)
4. an optional labeled or secondary estimator panel
5. `latest_spectrum.png`

All implementations should use the same dark, responsive split-view shell,
metric coordinate limits, isotope colors, obstacle rendering, runtime-provided
travel waypoints, station visit labels, and current robot marker. A combined
orchestrator view may add navigation and both estimator panels without changing
the shared filenames.

The human-readable URL event is:

```text
CUI split visualization URL: http://<browser-host>:<port>/index.html
```

Machine-readable commands should additionally emit a versioned `cui_ready`
event to stderr or a dedicated event stream so JSON result output on stdout
remains valid.

Unknown services on an occupied preferred port must never be treated as the
current CUI. The server selects the next available port and reports the actual
one. Explicit public hosts and IPv6 literals must produce valid browser URLs.

## Truth isolation

CUI truth has three explicit modes:

- `hidden`: never request or render evaluation truth;
- `evaluation_live`: request the private overlay for live evaluation only;
- `post_run`: request the private overlay only after estimator decisions are
  complete and render a final evaluation frame.

`post_run` is the default. Truth overlay objects must not be stored in estimator
state, planner inputs, estimator result manifests, or MeasurementLog artifacts.
The runtime private CUI channel remains the only live source of realized truth.

## Migration phases

### Phase 1: compatible infrastructure extraction

- Publish runtime atomic file helpers and deterministic NPZ writing.
- Publish packaged runtime defaults.
- Publish the CUI server, URL, and route APIs.
- Keep existing PF/MLE/orchestrator entry points as thin compatibility wrappers.
- Make every `latest_*` update atomic and retain the PF five-panel result set.

### Phase 2: typed adaptive session

- Replace raw ready, candidate, record, refine, resume, and finalized dictionaries
  with frozen, versioned DTOs.
- Add high-level `handshake`, `acquire_station`, `refine_candidates`, and
  `finalize_log` client operations.
- Preserve a strict JSON wire representation and reject unknown fields.

### Phase 3: MeasurementLog convergence

- Add a public read-only array view and artifact inventory to the runtime
  `MeasurementLog` API.
- Delegate orchestrator MeasurementLog v2 validation and prefix publication to
  runtime APIs.
- Retain orchestrator-local schema-v1 readers only for historical artifacts.
- Freeze byte-level conformance fixtures before replacing legacy digest code.

### Phase 4: estimator service contracts

PF and MLE backends use a small, separately versioned estimator-service wire
contract containing capabilities, execution requests/responses, artifact
references, and controller-owned execution receipts. It contains neither
simulation physics nor realized truth. Each service adapter invokes the solver
inside its owning estimator repository; the orchestrator uses a subprocess
client and never imports or copies either solver.

## Compatibility and acceptance gates

- Existing CLI commands and result filenames remain readable during migration.
- Contract changes are backward-compatible or increment a schema version.
- Canonical JSON, deterministic NPZ, MeasurementLog, and prefix digests have
  byte-level conformance tests.
- CUI tests cover IPv4, IPv6, explicit and discovered public hosts, occupied
  ports, custom roots, actual HTTP GETs, route waypoints, repeated station
  visits, atomic multi-file publication, and truth isolation.
- Runtime physics and the event-level observation distribution are unchanged by
  every phase in this roadmap.
