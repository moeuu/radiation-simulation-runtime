# Artifact and Git policy

Track source, native build inputs, reusable configurations, tests, documentation,
dependency locks, and canonical model assets under `configs/` and `src/`.
Generated observations, raw calibration corpora, private truth, logs, local
datasets, caches, and distributions stay outside Git. Do not bypass `.gitignore`
with `git add -f` for these files.

## Output ownership

| Purpose | Repository-relative location |
| --- | --- |
| Runtime acquisitions | `results/runs/<run-id>/` |
| Private scenes and truth | `private_runs/<run-id>/` or the existing `private_runs/ral_ablation/` layout |
| Detector construction | `results/detector_green_construction/<run-id>/` |
| Detector validation | `results/detector_green_validation/<run-id>/` |
| Calibration and all-64 acceptance | `results/mean_calibration/<run-id>/`, `results/full_spectrum_all64_acceptance/<run-id>/` |
| Other diagnostics and timing | `results/diagnostics/<run-id>/`, `results/benchmarks/<run-id>/` |
| Decay comparison | `results/decay_cascade_comparison/<run-id>/` |
| Environment previews | `results/previews/<run-id>/` |
| Generated environments | `results/blender_environments/<run-id>/` |
| Console logs | The run's `logs/` directory, or `logs/<run-id>/` |
| Disposable files and test operators | Automatically cleaned temporary directories under `tmp/` |
| Original Manchester assets | `data/manchester_nuclear_assets/` |

Keep commands, resolved configuration, revisions, start/end status, logs, and
raw evidence together for a run. Start new experiments with a fresh directory.
The acceptance CLI requires `--output-root`; resuming or advancing
a phase must explicitly name that same recorded directory. Never overwrite a
completed run just because the software or contract hash is unchanged.

Use a timestamp or run ID for new output directories, rather than implementation
suffixes such as `v7`. Existing evidence paths remain immutable references.
Do not restore retired model implementations or their scene-fitting launchers;
follow the [current model policy](current_model.md).

Sidecar defaults create a fresh `logs/<backend>-<port>-<id>/bridge.log` for each
launch. Set `sidecar_log_path` or `isaacsim_sidecar_log_path` to keep it within an
acquisition bundle. Environment preview defaults also create a fresh run
directory. Explicit output paths remain caller-controlled.

The existing `results/full_spectrum_all64_acceptance/` is the retained canonical
approval bundle. Keep its evidence and associated logs together while the current
approved model depends on it. Existing `ral_isaac_figures/` and
`ral_supplementary_video/` tools own fixed figure filenames; these exceptions are
not general storage for unrelated output.

## Retention and verification

Keep current detector construction/validation evidence, approved acceptance
corpora, current RA-L observations/private truth/evaluation/figure evidence, and
Manchester data. Review obsolete data by exact run identity and references
before moving it to the trash. Never automatically delete by age. Remove
disposable previews after review; `tmp/` is not an archive. Tests must use pytest
temporary paths or process-owned `TemporaryDirectory` objects.

Build with `uv build`. Hatchling packages current source directly, including
`runtime_environment.py` and the canonical detector operator. The source
distribution also includes native source, configuration, and layouts. Physical
execution still needs a configured runtime workspace and native executable;
wheels do not embed Geant4 or Manchester data. Keep only needed distributions
in ignored `dist/`. Preserve active `build/geant4_sidecar*` executables;
obsolete `build/lib/` copies must not return.

Run `uv run python scripts/audit_artifacts.py --check` before committing. This
read-only check detects force-added ignored files, unexpected output categories,
loose outputs, and retired directories without hashing datasets or running
physics. Document and register intentional new output categories in that script.
Use related tests during iteration and full verification for core changes;
artifact housekeeping does not require full scientific acquisitions.
