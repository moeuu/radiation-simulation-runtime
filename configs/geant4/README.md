# Geant4 Runtime Configs

Use these top-level configs for current simulations:

Standard top-level configs use `primary_sampling_fraction=1.0`. The Python
Geant4 application rejects fractional primary-history sampling unless the
nonstandard accelerated mode is selected explicitly, and it checks native
response provenance before returning any observation to an estimator.
Production adaptive sessions accept only complete, self-contained schema-1
documents with the canonical field set. Unknown fields, missing fields,
estimator-owned controls, implicit type coercion, and `extends` inheritance are
rejected before simulator construction. Diagnostic tooling may still use the
generic inherited-config loader, but that loader is not a production entrypoint.
The live CUI lifecycle and all `cui_split_view_*` settings are PF-owned; Geant4
production configs reject those keys instead of accepting ignored duplicates.

- `variance_reduction_external_no_isaac_32threads.json`: standard no-GUI
  standard no-GUI Geant4 acquisition config for the shared runtime,
  `--cui`, and `--full-simulation`. It requires one authenticated
  profile-selected geometry-conditioned joint full-spectrum model with a
  globally fitted, nonnegative additive noncollided transport component. The
  standard RA-L profile uses a training-only, view-conditioned count and mark
  discrepancy; the rejected deterministic low-rank mean correction is not in
  the runtime path.
- `variance_reduction_external_gui_32threads.json`: standard Geant4 acquisition
  simulation config plus an Isaac Sim sidecar for visualization. It is a
  complete standalone production document; the intended runtime difference is
  Isaac Sim startup only.
- `high_fidelity_external_no_isaac.json`: full-transport verification config.
  It is intentionally slower and should be selected explicitly.
- `external_gui_scene.json`: explicit USD-backed Manchester Drum Store scene.
- `shield_validation_scene.json`: material/shield validation config.

The profile registry hash proves which exact catalog-derived model asset was
loaded. Production startup requires either literal all-64 application approval
for that exact model or a schema-v7 catalog-independent approval transferred
from the canonical validated model. Transfer is allowed only when detector,
transport, background, dead-time, likelihood uncertainty, and execution
contracts are identical and every catalog line is inside the validated
continuous detector-energy domain. The provenance continues to name the
isotopes that received end-to-end all-64 validation, so a transferred profile
does not masquerade as application-validated. Changes to the catalog-independent
algorithm contract fail closed; adding canonical in-domain lines does not force
another multi-hour acceptance run. Training-only models with neither form of
approval remain rejected. The evidence also binds the native executable,
resolved dynamic-library bytes, explicit Geant4 environment and complete physics
data trees, and the Python implementation bundle. Production requires
`auto_start_sidecar=true`, refuses an already occupied TCP endpoint, validates all
bundle digests in the reset handshake, and revalidates the launch inputs immediately
before starting the native process.

## Versioned asset retention

Only model assets referenced by the current runtime, registry, builders, tests,
or documentation are kept in this production repository. Superseded training
and profile generations remain recoverable from Git history instead of living
beside the active assets. The profile directory is intentionally limited to the
seven `physics_only` files authenticated by
`isotope_profile_model_registry.json`.

Generation suffixes are omitted when only one active asset exists. Explicit
serialized schema numbers and stable contract identifiers remain versioned
when their values participate in authentication or exact-identity checks.
MeasurementLog publication is fixed at schema version 2 and has no schema-1
writer or loader.
