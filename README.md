# Rotating-shield simulation runtime

This repository is the common simulation and observation-generation boundary for
the rotating-shield estimators. It owns Geant4, environments, source realization,
detector and Fe/Pb shield geometry, spectrum generation, and raw MeasurementLog v2
publication. It contains no PF, MLE, or hybrid estimator.

```text
Rotating-shield-simulation-runtime
  SimulationCommand -> Geant4 -> raw integer spectrum -> MeasurementLog v2
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
```

`run-plan` reads source truth only from the private plan, durably records each raw
observation before returning it to a controller, and never writes source truth into
MeasurementLog v2.

## License

Released under the [MIT License](LICENSE).
