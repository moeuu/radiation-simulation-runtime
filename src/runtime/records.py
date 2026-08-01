"""Read-only record views exported by the shared MeasurementLog v2 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from runtime.measurement_log import (
    MEASUREMENT_LOG_SCHEMA_VERSION,
    MeasurementLog,
    MeasurementLogRecord,
    _validate_truth_free_payload,
)
from runtime.provenance import canonical_json_bytes, sha256_json


MeasurementRecord = MeasurementLogRecord


def canonical_json_sha256(value: object) -> str:
    """Return the canonical JSON digest used by shared runtime artifacts."""
    return sha256_json(value)


def validate_truth_free_estimator_input(
    value: object,
    *,
    path: str = "estimator_input",
) -> None:
    """Reject realized source truth in estimator-visible JSON payloads."""
    _validate_truth_free_payload(value, location=path)


@dataclass(frozen=True)
class RunContext:
    """Expose estimator-neutral run metadata derived from MeasurementLog v2."""

    repository_commit: str
    runtime_config: Mapping[str, Any]
    environment: Mapping[str, Any]
    sim_backend: str
    spectrum_count_method: str
    isotopes: tuple[str, ...]
    obstacle_layout_path: str | None
    source_layout_path: None
    source_rate_model: str
    metadata: Mapping[str, Any]
    run_id: str
    source_rate_semantics: Mapping[str, Any]
    forward_model_manifest: Mapping[str, Any]
    runtime_config_sha256: str
    schema_version: int = MEASUREMENT_LOG_SCHEMA_VERSION

    @classmethod
    def from_measurement_log(cls, log: MeasurementLog) -> "RunContext":
        """Build a read-only context from one validated shared log."""
        manifest = log.run_manifest
        return cls(
            repository_commit=str(manifest["repository_commit"]),
            runtime_config=MappingProxyType(dict(log.runtime_config)),
            environment=MappingProxyType(dict(log.environment)),
            sim_backend=str(manifest["sim_backend"]),
            spectrum_count_method="joint_full_spectrum_generative",
            isotopes=tuple(str(value) for value in manifest["isotopes"]),
            obstacle_layout_path=(
                None
                if manifest.get("obstacle_layout_path") is None
                else str(manifest["obstacle_layout_path"])
            ),
            source_layout_path=None,
            source_rate_model=str(manifest["source_rate_model"]),
            metadata=MappingProxyType(dict(manifest.get("metadata", {}))),
            run_id=str(manifest["run_id"]),
            source_rate_semantics=MappingProxyType(
                dict(manifest["source_rate_semantics"])
            ),
            forward_model_manifest=MappingProxyType(
                dict(log.forward_model_manifest)
            ),
            runtime_config_sha256=str(manifest["resolved_config_sha256"]),
            schema_version=int(manifest["schema_version"]),
        )


__all__ = [
    "MEASUREMENT_LOG_SCHEMA_VERSION",
    "MeasurementRecord",
    "RunContext",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "validate_truth_free_estimator_input",
]
