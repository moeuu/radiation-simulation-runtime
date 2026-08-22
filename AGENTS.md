# Repository instructions

## Repository role

This repository is the sole production owner of rotating-shield simulation,
environment generation, detector and shield geometry, Geant4 transport, raw
observation generation, and MeasurementLog publication.

- Do not add PF, MLE, hybrid inference, estimator planning, posterior summaries,
  or estimator-specific rescue logic.
- Estimators connect through `SimulationCommand` and immutable MeasurementLog
  artifacts. They must not receive realized source truth.
- Keep Geant4 physics fidelity and the event-level observation distribution
  unchanged when optimizing runtime performance.
- Use Python 3.12 and `uv`; run `uv run pytest` after changes.
- Follow PEP 8. Every function must have an English docstring.
- Commit only to `main`. Do not create a pull request unless explicitly asked.

## Physical semantics

- `intensity_cps_1m` is expected pre-dead-time detector pulse rate at 1 m for
  the configured detector and response processing, not total gamma emission.
- Production spectra are exact nonnegative int64 unit-weight event histograms.
- Background and non-paralyzable dead time are applied exactly once.
- Runtime shortcuts that change physics or observation statistics are forbidden.
