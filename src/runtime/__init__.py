"""Runtime contracts shared by live acquisition and estimator replay."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "ArtifactInventory": "runtime.artifacts",
    "AdaptiveAbortedEvent": "runtime.adaptive_protocol",
    "AdaptiveBootstrap": "runtime.adaptive_protocol",
    "AdaptiveCandidateSnapshot": "runtime.adaptive_protocol",
    "AdaptiveCandidatesEvent": "runtime.adaptive_protocol",
    "AdaptivePublishedEvent": "runtime.adaptive_protocol",
    "AdaptiveReadyEvent": "runtime.adaptive_protocol",
    "AdaptiveRecordEvent": "runtime.adaptive_protocol",
    "AdaptiveRefineRequest": "runtime.adaptive_protocol",
    "AdaptiveRuntimeSession": "runtime.adaptive",
    "AdaptiveSessionEvent": "runtime.adaptive_protocol",
    "AdaptiveStepRequest": "runtime.adaptive_protocol",
    "parse_adaptive_event": "runtime.adaptive_protocol",
    "serve_adaptive_session": "runtime.adaptive",
    "AdaptiveResumePrefix": "runtime.adaptive_client",
    "AdaptiveRuntimeClient": "runtime.adaptive_client",
    "parse_adaptive_resume_prefix": "runtime.adaptive_client",
    "atomic_copy_file": "runtime.artifacts",
    "atomic_write_bytes": "runtime.artifacts",
    "atomic_write_json": "runtime.artifacts",
    "atomic_write_text": "runtime.artifacts",
    "build_artifact_inventory": "runtime.artifacts",
    "simulation_runtime_root": "runtime.assets",
    "standard_geant4_config_path": "runtime.assets",
    "CUIDashboardConfig": "runtime.cui",
    "CUIRoute": "runtime.cui",
    "CUIServerHandle": "runtime.cui",
    "CUI_URL_MESSAGE_PREFIX": "runtime.cui",
    "cui_browser_url": "runtime.cui",
    "cui_route_from_records": "runtime.cui",
    "resolve_cui_public_host": "runtime.cui",
    "start_cui_server": "runtime.cui",
    "CANONICAL_UNITS": "runtime.forward_model_manifest",
    "line_energy_weight_by_isotope": "runtime.forward_model_manifest",
    "production_line_mu_by_isotope": "runtime.forward_model_manifest",
    "MEASUREMENT_LOG_SCHEMA_VERSION": "runtime.measurement_log",
    "MeasurementLog": "runtime.measurement_log",
    "MeasurementLogArrayView": "runtime.measurement_log",
    "MeasurementLogStationView": "runtime.measurement_log",
    "MeasurementLogRecord": "runtime.measurement_log",
    "MeasurementStation": "runtime.measurement_log",
    "MeasurementLogStreamWriter": "runtime.measurement_log",
    "MeasurementLogValidationError": "runtime.measurement_log",
    "build_forward_model_manifest": "runtime.measurement_log",
    "load_measurement_log": "runtime.measurement_log",
    "measurement_log_artifact_inventory": "runtime.measurement_log",
    "measurement_log_sha256": "runtime.measurement_log",
    "measurement_records_content_sha256": "runtime.measurement_log",
    "validate_forward_model_manifest": "runtime.measurement_log",
    "write_deterministic_npz": "runtime.measurement_log",
    "write_measurement_log": "runtime.measurement_log",
    "MeasurementLogPrefix": "runtime.prefix",
    "covered_station_boundaries_sha256": "runtime.prefix",
    "materialize_measurement_log_prefix": "runtime.prefix",
    "measurement_records_sha256": "runtime.prefix",
    "strict_canonical_json_bytes": "runtime.provenance",
    "strict_sha256_json": "runtime.provenance",
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
    "ArtifactInventory",
    "AdaptiveAbortedEvent",
    "AdaptiveBootstrap",
    "AdaptiveCandidateSnapshot",
    "AdaptiveCandidatesEvent",
    "AdaptivePublishedEvent",
    "AdaptiveReadyEvent",
    "AdaptiveRecordEvent",
    "AdaptiveRefineRequest",
    "AdaptiveResumePrefix",
    "AdaptiveRuntimeClient",
    "AdaptiveRuntimeSession",
    "AdaptiveSessionEvent",
    "AdaptiveStepRequest",
    "CANONICAL_UNITS",
    "CUIDashboardConfig",
    "CUIRoute",
    "CUIServerHandle",
    "CUI_URL_MESSAGE_PREFIX",
    "MEASUREMENT_LOG_SCHEMA_VERSION",
    "MeasurementLog",
    "MeasurementLogArrayView",
    "MeasurementLogStationView",
    "MeasurementLogRecord",
    "MeasurementLogStreamWriter",
    "MeasurementLogValidationError",
    "MeasurementLogPrefix",
    "MeasurementStation",
    "atomic_copy_file",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "build_artifact_inventory",
    "build_forward_model_manifest",
    "covered_station_boundaries_sha256",
    "cui_browser_url",
    "cui_route_from_records",
    "load_measurement_log",
    "line_energy_weight_by_isotope",
    "measurement_log_artifact_inventory",
    "measurement_log_sha256",
    "measurement_records_content_sha256",
    "materialize_measurement_log_prefix",
    "measurement_records_sha256",
    "production_line_mu_by_isotope",
    "parse_adaptive_resume_prefix",
    "parse_adaptive_event",
    "resolve_cui_public_host",
    "serve_adaptive_session",
    "simulation_runtime_root",
    "standard_geant4_config_path",
    "start_cui_server",
    "strict_canonical_json_bytes",
    "strict_sha256_json",
    "validate_forward_model_manifest",
    "write_deterministic_npz",
    "write_measurement_log",
]
