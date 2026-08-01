# Architecture and ownership

The simulation runtime is the only repository allowed to generate production
observations. Estimator repositories own only inference, estimator-specific
planning, summaries, and evaluation adapters.

The process boundary is intentional. Estimators submit a versioned
`SimulationCommand`; the runtime returns a raw `SimulationObservation` and writes
the corresponding MeasurementLog record first. Complete-log replay and live
closed-loop operation therefore share the same physics and serialization path.

MeasurementLog contains detector poses, timing, Fe/Pb orientation indices, the raw
integer spectrum, energy edges, environment geometry, and immutable forward-model
identity. Realized sources and evaluation truth remain outside the estimator-visible
bundle.

Estimator configuration is not part of the acquisition identity. PF particle count,
MLE regularization, DSS policy, and hybrid settings belong to their estimator repos
and must never be inserted into the shared runtime config.

