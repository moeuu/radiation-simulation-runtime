"""Runtime contracts shared by live acquisition and estimator replay."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "AdaptiveRuntimeSession": "runtime.adaptive",
    "serve_adaptive_session": "runtime.adaptive",
    "AdaptiveResumePrefix": "runtime.adaptive_client",
    "AdaptiveRuntimeClient": "runtime.adaptive_client",
    "parse_adaptive_resume_prefix": "runtime.adaptive_client",
    "simulation_runtime_root": "runtime.assets",
    "standard_geant4_config_path": "runtime.assets",
    "CANONICAL_UNITS": "runtime.forward_model_manifest",
    "line_energy_weight_by_isotope": "runtime.forward_model_manifest",
    "production_line_mu_by_isotope": "runtime.forward_model_manifest",
    "MEASUREMENT_LOG_SCHEMA_VERSION": "runtime.measurement_log",
    "MeasurementLog": "runtime.measurement_log",
    "MeasurementLogRecord": "runtime.measurement_log",
    "MeasurementLogStreamWriter": "runtime.measurement_log",
    "MeasurementLogValidationError": "runtime.measurement_log",
    "build_forward_model_manifest": "runtime.measurement_log",
    "load_measurement_log": "runtime.measurement_log",
    "measurement_log_sha256": "runtime.measurement_log",
    "validate_forward_model_manifest": "runtime.measurement_log",
    "write_measurement_log": "runtime.measurement_log",
    "MeasurementLogPrefix": "runtime.prefix",
    "covered_station_boundaries_sha256": "runtime.prefix",
    "materialize_measurement_log_prefix": "runtime.prefix",
    "measurement_records_sha256": "runtime.prefix",
}


def __getattr__(name: str) -> Any:
    """Load public runtime contracts without eager package side effects."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eager and lazy public package attributes."""
    return sorted(set(globals()) | set(_EXPORT_MODULES))

__all__ = [
    "CANONICAL_UNITS",
    "AdaptiveResumePrefix",
    "AdaptiveRuntimeClient",
    "AdaptiveRuntimeSession",
    "MEASUREMENT_LOG_SCHEMA_VERSION",
    "MeasurementLog",
    "MeasurementLogRecord",
    "MeasurementLogStreamWriter",
    "MeasurementLogValidationError",
    "MeasurementLogPrefix",
    "build_forward_model_manifest",
    "covered_station_boundaries_sha256",
    "load_measurement_log",
    "line_energy_weight_by_isotope",
    "measurement_log_sha256",
    "materialize_measurement_log_prefix",
    "measurement_records_sha256",
    "production_line_mu_by_isotope",
    "parse_adaptive_resume_prefix",
    "serve_adaptive_session",
    "simulation_runtime_root",
    "standard_geant4_config_path",
    "validate_forward_model_manifest",
    "write_measurement_log",
]
