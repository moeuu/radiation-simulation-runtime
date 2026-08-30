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
- every runtime artifact uses the same fail-closed canonical JSON contract;
  the public serializer names reject non-string keys, NumPy scalars, paths,
  arbitrary objects, and non-finite values instead of stringifying them;
- `runtime.measurement_log.write_deterministic_npz` is a public, atomic,
  byte-stable NPZ API;
- `runtime.defaults` is the sole packaged defaults namespace and owns shared
  PF-style CUI host/port/path values;
- `runtime.cui` owns strict configuration, browser URLs, managed safe static
  serving, the immutable truth-free `CUIRoute`, correctly ordered `CUIScene`
  geometry, and the responsive panel HTML shell. Estimator repositories own the
  number, identity, and rendering of result panels;
- `ResolvedForwardContext` owns authenticated physical-input reconstruction for
  live consumers and external evaluators, without owning any estimator state;
- adaptive ready, record, candidate, refinement, abort, and publication messages
  are frozen DTOs, and the client owns bounded lifecycle plus transcript observing;
- `MeasurementLogView.from_records` owns live array/station conversion, while
  file-backed logs expose generic inventories and pathless prefix views;
- digest identities and golden fixtures bind each current algorithm explicitly;
  historical normalization is not reachable from live artifact authoring;
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
| MeasurementLog v2 validation, views, and record-prefix digests | runtime and orchestrator | runtime; orchestrator keeps only legacy readers |
| PF/MLE solver and posterior state | PF/MLE repositories and orchestrator | estimator owners; no runtime dependency reversal |

## Canonical CUI contract

The live Particle Filter split view is the compatibility baseline for URL,
responsive layout, and shared acquisition context. A CUI implementation should
provide a directly clickable URL and periodically refresh completed image files.
The shared structure is:

1. `latest_experiment_overview.png`
2. `latest_robot_2d.png`
3. one or more estimator-owned result panels
4. `latest_spectrum.png`

The result-panel identifiers, filenames, titles, count, and renderers are not a
cross-estimator contract. PF keeps its particle and labeled-particle views; MLE
keeps its grid, likelihood-surface, or hotspot views. The shared shell accepts
these owner-defined panels without interpreting or transforming estimator state.

All implementations should use the same dark, responsive split-view shell,
metric coordinate limits, isotope colors, obstacle rendering, runtime-provided
travel waypoints, station visit labels, and current robot marker. A combined
orchestrator view may add navigation and compose both sets of estimator-owned
result panels without renaming their artifacts.

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
- `evaluation_live`: an external evaluation renderer may request the private
  overlay for live evaluation only;
- `post_run`: an external evaluator may request the private overlay only after
  estimator decisions are complete and render a final evaluation frame.

`post_run` is the default. Truth overlay objects must not be stored in estimator
state, planner inputs, estimator result manifests, or MeasurementLog artifacts.
`AdaptiveRuntimeClient` is estimator-facing and has no truth-overlay request API;
the retired overlay request is rejected as an unknown adaptive request, and a
retired private-overlay frame on its event stream is fatal.

Live evaluation uses a distinct owner-only Unix socket. The runtime creates it only
when `--cui-truth-overlay-socket-path` is explicit, refuses to replace an existing
path, applies mode `0600`, accepts one exact schema-version-1 request, serves one
validated response, and removes the endpoint on shutdown. The asynchronous CUI
renderer child connects to this socket directly before acknowledging readiness and
adds isotope-local source numbers plus an XYZ inventory to the live figures. PF
frames and the adaptive estimator stream remain truth-free. The
`CUITruthDisplayMode` enum describes renderer policy; it does not add a truth method
to an estimator client.

## Migration phases

### Phase 1: compatible infrastructure extraction

- Publish runtime atomic file helpers and deterministic NPZ writing.
- Publish packaged runtime defaults.
- Publish the CUI server, URL, and route APIs.
- Keep existing PF/MLE/orchestrator entry points as thin compatibility wrappers.
- Make every `latest_*` update atomic and retain the shared page structure while
  letting each estimator define and render its own result panels.

### Phase 2: typed adaptive session

- Replace raw ready, candidate, record, refine, and finalized dictionaries
  with frozen, versioned DTOs.
- Add high-level `handshake`, `acquire_station`, `refine_candidates`, and
  `finalize_log` client operations.
- Preserve a strict JSON wire representation and reject unknown fields.

### Phase 3: MeasurementLog convergence

- Add a public read-only array view and artifact inventory to the runtime
  `MeasurementLog` API.
- Delegate orchestrator MeasurementLog v2 validation, read-only views, and record
  prefix digests to runtime APIs.
- Retain orchestrator-local schema-v1 readers only for historical artifacts.
- Keep byte-level conformance fixtures for JSON-native historical artifacts;
  lossy historical authoring is intentionally unsupported.

## Compatibility and acceptance gates

- Current CLI commands and result filenames remain readable when their schemas
  are exact; retired aliases and lossy authoring paths fail immediately.
- Semantic contract changes use an explicit schema version or a deliberate
  clean break rather than a compatibility fallback.
- Canonical JSON, deterministic NPZ, MeasurementLog, and prefix digests have
  byte-level conformance tests.
- CUI tests cover IPv4, IPv6, explicit and discovered public hosts, occupied
  ports, custom roots, actual HTTP GETs, route waypoints, repeated station
  visits, atomic multi-file publication, and truth isolation.
- Runtime physics and the event-level observation distribution are unchanged by
  every phase in this roadmap.
