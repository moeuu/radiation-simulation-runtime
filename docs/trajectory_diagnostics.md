# Native trajectory diagnostics

`scripts/export_ral_geant4_trajectories.py` records primary-gamma step endpoints
from small Geant4 diagnostic reruns at a selected acquired station. The isotropic
sample shows outward transport; the fixed-quota detector-cone sample provides
detector-entering tracks. These are newly simulated paths, not recovered paths
from the original acquisition. Neither sample is a production observation or
new independent model-acceptance result.

Build the diagnostic tool separately:

```bash
uv run python scripts/build_geant4_sidecar.py \
  --trajectory-diagnostic \
  --output build/geant4_trajectory_sidecar \
  --metadata-output build/geant4_trajectory_sidecar.build.json
```

The ordinary build omits trajectory storage and recording hooks. The diagnostic
build refuses the production executable and build-metadata destinations. Its
default PGO directory is also separate. The approved production executable and
its acceptance evidence remain in place.

Select the run and station explicitly:

```bash
uv run python scripts/export_ral_geant4_trajectories.py \
  --run-id RUN_ID --station-index 5
```

Default input paths follow that run under `private_runs/ral_ablation/`; the
MeasurementLog comes from the sibling PF repository's matching run. Override
`--runtime-config`, `--scenario`, `--truth-manifest`, or `--observations` when
needed. The observation file must belong to a complete, authenticated
MeasurementLog with the same run ID as the private inputs.

Each invocation selects a fresh directory under
`private_runs/ral_ablation/figure_tracks/`, with a timestamp and unique suffix.
It contains `trajectories.json` and the associated native files under `raw/`.
`--output` can select an explicit new destination; existing JSON or raw outputs
are never replaced. Interrupted exports stay in their own directory for review.
All trajectory data contains private source information and stays outside Git.

The recorder caps retained tracks and points without killing or changing any
transport history. A truncated track retains native endpoints but omits
intermediate steps; inspect `points_truncated` before treating connecting
segments as a complete physical path. The artifact records the two sampling
modes, seeds, selected pose, native metadata, and source/executable hashes.
Native tests compare recording on/off and ordinary/diagnostic builds using the
same seeds, requiring identical spectra, variances, and primary counts.

Only immutable executables are shared between native tests. Every scene,
request, response, and trajectory output is owned by a disposable test directory.
