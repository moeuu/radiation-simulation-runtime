"""Read-only record views exported by the shared MeasurementLog runtime."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np

from runtime.measurement_log import (
    MEASUREMENT_LOG_SCHEMA_VERSION,
    MeasurementLog,
    MeasurementLogRecord,
    _validate_truth_free_payload,
)
from runtime.provenance import canonical_json_bytes, sha256_json


MeasurementRecord = MeasurementLogRecord

_MEASUREMENT_RECORD_FIELDS = frozenset(
    {
        "step_id",
        "action_id",
        "station_id",
        "detector_pose_xyz",
        "detector_quat_wxyz",
        "fe_orientation_index",
        "pb_orientation_index",
        "live_time_s",
        "travel_time_s",
        "shield_actuation_time_s",
        "energy_bin_edges_keV",
        "spectrum_counts",
        "metadata",
    }
)
_RUN_CONTEXT_FIELDS = frozenset(
    {
        "repository_commit",
        "runtime_config",
        "environment",
        "sim_backend",
        "spectrum_count_method",
        "isotopes",
        "obstacle_layout_path",
        "source_rate_model",
        "metadata",
        "run_id",
        "source_rate_semantics",
        "forward_model_manifest",
        "runtime_config_sha256",
        "schema_version",
    }
)


def _freeze_json(value: object, *, path: str) -> object:
    """Return a recursively immutable copy of one strict JSON value."""
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings.")
            frozen[key] = _freeze_json(nested, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(nested, path=f"{path}[{index}]")
            for index, nested in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(
        value,
        (bool, np.bool_),
    ):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError(f"{path} must not contain non-finite numbers.")
        return parsed
    raise TypeError(f"{path} must contain only strict JSON values.")


def _thaw_json(value: object) -> object:
    """Return a mutable JSON-compatible copy of one frozen JSON value."""
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(nested) for nested in value]
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    """Return one nonempty protocol string."""
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string.")
    return value


def _require_exact_fields(
    payload: Mapping[str, object],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    """Require an exact set of fields in one wire payload."""
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{name} fields disagree with the runtime schema: "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )


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
    """Expose estimator-neutral run metadata derived from a MeasurementLog."""

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

    def __post_init__(self) -> None:
        """Validate scalar fields and freeze every nested JSON mapping."""
        for name in (
            "repository_commit",
            "sim_backend",
            "spectrum_count_method",
            "source_rate_model",
            "run_id",
            "runtime_config_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty_string(getattr(self, name), name=name),
            )
        if self.obstacle_layout_path is not None and not isinstance(
            self.obstacle_layout_path,
            str,
        ):
            raise TypeError("obstacle_layout_path must be a string or null.")
        if self.source_layout_path is not None:
            raise ValueError("source_layout_path must remain null.")
        if isinstance(self.schema_version, (bool, np.bool_)) or not isinstance(
            self.schema_version,
            (int, np.integer),
        ):
            raise TypeError("schema_version must be an integer.")
        schema_version = int(self.schema_version)
        if schema_version <= 0:
            raise ValueError("schema_version must be positive.")
        object.__setattr__(self, "schema_version", schema_version)
        if not isinstance(self.isotopes, (list, tuple)) or any(
            not isinstance(value, str) or not value for value in self.isotopes
        ):
            raise TypeError("isotopes must contain nonempty strings.")
        isotopes = tuple(self.isotopes)
        if not isotopes or len(set(isotopes)) != len(isotopes):
            raise ValueError("isotopes must be nonempty and unique.")
        object.__setattr__(self, "isotopes", isotopes)
        for name in (
            "runtime_config",
            "environment",
            "metadata",
            "source_rate_semantics",
            "forward_model_manifest",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be an object.")
            object.__setattr__(self, name, _freeze_json(value, path=name))

    def to_payload(self) -> dict[str, object]:
        """Serialize this truth-free context to its adaptive wire payload."""
        payload: dict[str, object] = {
            "repository_commit": self.repository_commit,
            "runtime_config": _thaw_json(self.runtime_config),
            "environment": _thaw_json(self.environment),
            "sim_backend": self.sim_backend,
            "spectrum_count_method": self.spectrum_count_method,
            "isotopes": list(self.isotopes),
            "obstacle_layout_path": self.obstacle_layout_path,
            "source_rate_model": self.source_rate_model,
            "metadata": _thaw_json(self.metadata),
            "run_id": self.run_id,
            "source_rate_semantics": _thaw_json(self.source_rate_semantics),
            "forward_model_manifest": _thaw_json(self.forward_model_manifest),
            "runtime_config_sha256": self.runtime_config_sha256,
            "schema_version": self.schema_version,
        }
        validate_truth_free_estimator_input(payload, path="adaptive.context")
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "RunContext":
        """Parse one exact truth-free adaptive context payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("Adaptive runtime context must be an object.")
        validate_truth_free_estimator_input(payload, path="adaptive.context")
        _require_exact_fields(payload, _RUN_CONTEXT_FIELDS, name="adaptive context")
        return cls(
            repository_commit=payload["repository_commit"],
            runtime_config=payload["runtime_config"],
            environment=payload["environment"],
            sim_backend=payload["sim_backend"],
            spectrum_count_method=payload["spectrum_count_method"],
            isotopes=tuple(payload["isotopes"]),
            obstacle_layout_path=payload["obstacle_layout_path"],
            source_layout_path=None,
            source_rate_model=payload["source_rate_model"],
            metadata=payload["metadata"],
            run_id=payload["run_id"],
            source_rate_semantics=payload["source_rate_semantics"],
            forward_model_manifest=payload["forward_model_manifest"],
            runtime_config_sha256=payload["runtime_config_sha256"],
            schema_version=payload["schema_version"],
        )

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


def measurement_record_to_payload(
    record: MeasurementLogRecord,
) -> dict[str, object]:
    """Serialize one truth-free measurement record for adaptive transport."""
    payload: dict[str, object] = {
        "step_id": record.step_id,
        "action_id": record.action_id,
        "station_id": record.station_id,
        "detector_pose_xyz": list(record.detector_pose_xyz),
        "detector_quat_wxyz": list(record.detector_quat_wxyz),
        "fe_orientation_index": record.fe_orientation_index,
        "pb_orientation_index": record.pb_orientation_index,
        "live_time_s": record.live_time_s,
        "travel_time_s": record.travel_time_s,
        "shield_actuation_time_s": record.shield_actuation_time_s,
        "energy_bin_edges_keV": record.energy_bin_edges_keV.tolist(),
        "spectrum_counts": record.spectrum_counts.tolist(),
        "metadata": _thaw_json(record.metadata),
    }
    validate_truth_free_estimator_input(payload, path="adaptive.record")
    return payload


def measurement_record_from_payload(
    payload: Mapping[str, object],
) -> MeasurementLogRecord:
    """Parse one exact truth-free adaptive measurement record payload."""
    if not isinstance(payload, Mapping):
        raise TypeError("Adaptive record must be an object.")
    validate_truth_free_estimator_input(payload, path="adaptive.record")
    _require_exact_fields(payload, _MEASUREMENT_RECORD_FIELDS, name="adaptive record")
    raw_counts = np.asarray(payload["spectrum_counts"])
    if raw_counts.ndim != 1 or not np.issubdtype(raw_counts.dtype, np.integer):
        raise TypeError("Adaptive spectrum_counts must contain exact integers.")
    return MeasurementLogRecord(
        step_id=payload["step_id"],
        action_id=payload["action_id"],
        station_id=payload["station_id"],
        detector_pose_xyz=payload["detector_pose_xyz"],
        detector_quat_wxyz=payload["detector_quat_wxyz"],
        fe_orientation_index=payload["fe_orientation_index"],
        pb_orientation_index=payload["pb_orientation_index"],
        live_time_s=payload["live_time_s"],
        travel_time_s=payload["travel_time_s"],
        shield_actuation_time_s=payload["shield_actuation_time_s"],
        energy_bin_edges_keV=np.asarray(
            payload["energy_bin_edges_keV"],
            dtype=np.float64,
        ),
        spectrum_counts=np.asarray(raw_counts, dtype=np.int64),
        metadata=payload["metadata"],
    )


__all__ = [
    "MEASUREMENT_LOG_SCHEMA_VERSION",
    "MeasurementRecord",
    "RunContext",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "measurement_record_from_payload",
    "measurement_record_to_payload",
    "validate_truth_free_estimator_input",
]
