"""Resolve estimator-neutral physical context from runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from measurement.continuous_kernels import ContinuousKernel
from measurement.model import EnvironmentConfig
from measurement.observation_model import (
    RuntimeObservationModel,
    build_runtime_observation_model,
    continuous_kernel_from_observation_model,
)
from measurement.obstacles import ObstacleGrid
from runtime.forward_model_manifest import (
    SOURCE_RATE_MODEL,
    SOURCE_RATE_SEMANTICS,
    forward_model_component_payloads,
    resolve_file_backed_model_asset,
)
from runtime.measurement_log import (
    MEASUREMENT_LOG_SCHEMA_VERSION,
    MeasurementLog,
    MeasurementLogValidationError,
    _canonical_isotope_names,
    _validate_environment_payload,
    _validate_run_identity,
    _validate_runtime_observation_contract,
    validate_forward_model_manifest,
)
from runtime.provenance import sha256_json, strict_canonical_json_bytes
from runtime.records import RunContext, validate_truth_free_estimator_input
from spectrum.transport_spectral import (
    GeometryConditionedSpectralModel,
    geometry_conditioned_model_from_runtime_config,
)


def _json_object(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    """Return an independent strict-JSON object from a frozen mapping."""
    try:
        decoded = json.loads(strict_canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MeasurementLogValidationError(
            f"{field_name} must contain only strict finite JSON data."
        ) from exc
    if not isinstance(decoded, dict):
        raise MeasurementLogValidationError(f"{field_name} must be an object.")
    return decoded


def _freeze_json_mapping(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return a recursively immutable copy of one strict JSON mapping."""

    def freeze(item: object) -> object:
        """Freeze one already validated JSON subtree."""
        if isinstance(item, dict):
            return MappingProxyType(
                {str(key): freeze(nested) for key, nested in item.items()}
            )
        if isinstance(item, list):
            return tuple(freeze(nested) for nested in item)
        return item

    normalized = _json_object(value, field_name="context mapping")
    result = freeze(normalized)
    if not isinstance(result, Mapping):  # pragma: no cover - defensive invariant
        raise TypeError("Frozen context payload must remain a mapping.")
    return result


def _validated_run_root(value: str | Path, *, field_name: str) -> Path:
    """Return an existing absolute directory for file-backed asset resolution."""
    supplied = Path(value)
    if not supplied.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path.")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{field_name} does not exist: {supplied}.") from exc
    if not resolved.is_dir():
        raise ValueError(f"{field_name} must identify an existing directory.")
    return resolved


def _environment_config(payload: Mapping[str, Any]) -> EnvironmentConfig:
    """Construct the validated runtime-owned room geometry."""
    return EnvironmentConfig(
        size_x=payload["size_x"],
        size_y=payload["size_y"],
        size_z=payload["size_z"],
        detector_position=payload["detector_position"],
    )


def _environment_bounds(
    environment: EnvironmentConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return immutable lower and upper Cartesian room bounds."""
    lower = np.zeros(3, dtype=np.float64)
    upper = np.asarray(
        [environment.size_x, environment.size_y, environment.size_z],
        dtype=np.float64,
    )
    lower.setflags(write=False)
    upper.setflags(write=False)
    return lower, upper


def _obstacle_grid(
    environment_payload: Mapping[str, Any],
    *,
    obstacle_layout_path: str | None,
    run_root: Path | None,
) -> tuple[ObstacleGrid | None, Path | None]:
    """Resolve embedded geometry first, then an authenticated local asset."""
    embedded = environment_payload["obstacle_grid"]
    if embedded is not None:
        if not isinstance(embedded, Mapping):
            raise MeasurementLogValidationError(
                "environment.obstacle_grid must be an object or null."
            )
        return ObstacleGrid.from_dict(dict(embedded)), None
    if obstacle_layout_path is None:
        return None, None
    resolved = resolve_file_backed_model_asset(
        obstacle_layout_path,
        field_name="obstacle_layout_path",
        run_root=run_root,
    )
    return ObstacleGrid.load(resolved), resolved


def _asset_identities(
    *,
    runtime_config: Mapping[str, Any],
    environment_payload: Mapping[str, Any],
    obstacle_layout_path: str | None,
    isotopes: tuple[str, ...],
    run_root: Path | None,
) -> Mapping[str, Mapping[str, str]]:
    """Return immutable portable identities for every file-backed input."""
    components = forward_model_component_payloads(
        runtime_config=runtime_config,
        environment=environment_payload,
        obstacle_layout_path=obstacle_layout_path,
        isotopes=isotopes,
        run_root=run_root,
    )
    identities: dict[str, Mapping[str, str]] = {}
    obstacle_asset = components["obstacle"].get("layout_asset")
    if isinstance(obstacle_asset, Mapping):
        identities["obstacle_layout_path"] = MappingProxyType(
            {
                "component": "obstacle",
                "path": str(obstacle_asset["path"]),
                "sha256": str(obstacle_asset["sha256"]),
            }
        )
    for component in ("detector", "transport", "spectrum"):
        file_assets = components[component].get("file_assets", {})
        if not isinstance(file_assets, Mapping):
            raise MeasurementLogValidationError(
                f"Resolved {component} file assets must be an object."
            )
        for field_path, raw_identity in file_assets.items():
            if not isinstance(raw_identity, Mapping):
                raise MeasurementLogValidationError(
                    f"Resolved asset identity {field_path!s} must be an object."
                )
            identities[f"runtime_config.{field_path!s}"] = MappingProxyType(
                {
                    "component": component,
                    "path": str(raw_identity["path"]),
                    "sha256": str(raw_identity["sha256"]),
                }
            )
    return MappingProxyType(dict(sorted(identities.items())))


def _model_identifiers(
    manifest: Mapping[str, Any],
) -> Mapping[str, Mapping[str, str]]:
    """Return immutable identifiers from one validated forward manifest."""
    raw = manifest["model_identifiers"]
    if not isinstance(raw, Mapping):  # pragma: no cover - validated upstream
        raise MeasurementLogValidationError("model_identifiers must be an object.")
    return MappingProxyType(
        {
            str(component): MappingProxyType(
                {
                    "id": str(identity["id"]),
                    "sha256": str(identity["sha256"]),
                }
            )
            for component, identity in raw.items()
            if isinstance(identity, Mapping)
        }
    )


def _resolve_context(
    context: RunContext,
    *,
    run_root: Path | None,
) -> ResolvedForwardContext:
    """Validate one truth-free context and resolve each physical object once."""
    validate_truth_free_estimator_input(
        context.to_payload(),
        path="resolved_forward_context",
    )
    if context.schema_version != MEASUREMENT_LOG_SCHEMA_VERSION:
        raise MeasurementLogValidationError(
            "Resolved forward context requires MeasurementLog schema version "
            f"{MEASUREMENT_LOG_SCHEMA_VERSION}."
        )
    if context.spectrum_count_method != "joint_full_spectrum_generative":
        raise MeasurementLogValidationError(
            "Resolved forward context requires joint full-spectrum observations."
        )
    if context.source_rate_model != SOURCE_RATE_MODEL:
        raise MeasurementLogValidationError(
            f"source_rate_model must be {SOURCE_RATE_MODEL!r}."
        )
    if dict(context.source_rate_semantics) != SOURCE_RATE_SEMANTICS:
        raise MeasurementLogValidationError(
            "source_rate_semantics differs from the runtime contract."
        )
    _validate_run_identity(context.run_id, context.repository_commit)
    isotopes = _canonical_isotope_names(
        context.isotopes,
        location="RunContext isotopes",
    )

    runtime_config = _json_object(
        context.runtime_config,
        field_name="runtime_config",
    )
    environment_payload = _json_object(
        context.environment,
        field_name="environment",
    )
    forward_manifest = _json_object(
        context.forward_model_manifest,
        field_name="forward_model_manifest",
    )
    _validate_environment_payload(environment_payload)
    _validate_runtime_observation_contract(
        runtime_config,
        isotopes=isotopes,
    )
    if runtime_config["sim_backend"] != context.sim_backend:
        raise MeasurementLogValidationError(
            "RunContext sim_backend differs from runtime_config."
        )
    if sha256_json(runtime_config) != context.runtime_config_sha256:
        raise MeasurementLogValidationError(
            "RunContext runtime_config_sha256 does not authenticate runtime_config."
        )
    validated_manifest = validate_forward_model_manifest(
        forward_manifest,
        runtime_config=runtime_config,
        environment=environment_payload,
        obstacle_layout_path=context.obstacle_layout_path,
        isotopes=isotopes,
        repository_commit=context.repository_commit,
        resolved_config_sha256=context.runtime_config_sha256,
        source_rate_model=context.source_rate_model,
        run_root=run_root,
    )

    environment = _environment_config(environment_payload)
    bounds_xyz = _environment_bounds(environment)
    obstacle_grid, resolved_obstacle_path = _obstacle_grid(
        environment_payload,
        obstacle_layout_path=context.obstacle_layout_path,
        run_root=run_root,
    )
    spectral_model = geometry_conditioned_model_from_runtime_config(
        runtime_config,
        run_root=run_root,
    )
    spectral_model.require_runtime_ready()
    spectral_model.require_environment_applicable(environment_payload)
    model_isotopes = {
        str(row["isotope"])
        for row in spectral_model.line_identity
        if isinstance(row, Mapping) and isinstance(row.get("isotope"), str)
    }
    if not set(isotopes).issubset(model_isotopes):
        raise MeasurementLogValidationError(
            "Authenticated spectrum model does not cover every run isotope."
        )
    observation_model = build_runtime_observation_model(
        runtime_config,
        isotopes=isotopes,
        authenticated_full_spectrum_model=spectral_model,
    )
    return ResolvedForwardContext(
        runtime_config=_freeze_json_mapping(runtime_config),
        environment_payload=_freeze_json_mapping(environment_payload),
        isotopes=isotopes,
        run_root=run_root,
        environment=environment,
        bounds_xyz=bounds_xyz,
        obstacle_grid=obstacle_grid,
        resolved_obstacle_path=resolved_obstacle_path,
        obstacle_attenuation_enabled=runtime_config[
            "obstacle_attenuation_enabled"
        ],
        spectral_model=spectral_model,
        observation_model=observation_model,
        model_identifiers=_model_identifiers(validated_manifest),
        asset_identities=_asset_identities(
            runtime_config=runtime_config,
            environment_payload=environment_payload,
            obstacle_layout_path=context.obstacle_layout_path,
            isotopes=isotopes,
            run_root=run_root,
        ),
    )


@dataclass(frozen=True, eq=False, slots=True)
class ResolvedForwardContext:
    """Hold authenticated physical inputs shared by estimator consumers."""

    runtime_config: Mapping[str, Any]
    environment_payload: Mapping[str, Any]
    isotopes: tuple[str, ...]
    run_root: Path | None
    environment: EnvironmentConfig
    bounds_xyz: tuple[NDArray[np.float64], NDArray[np.float64]]
    obstacle_grid: ObstacleGrid | None
    resolved_obstacle_path: Path | None
    obstacle_attenuation_enabled: bool
    spectral_model: GeometryConditionedSpectralModel = field(repr=False)
    observation_model: RuntimeObservationModel = field(repr=False)
    model_identifiers: Mapping[str, Mapping[str, str]]
    asset_identities: Mapping[str, Mapping[str, str]]

    @classmethod
    def from_log(cls, log: MeasurementLog) -> ResolvedForwardContext:
        """Resolve authenticated physical inputs from one validated log."""
        if not isinstance(log, MeasurementLog):
            raise TypeError("log must be a MeasurementLog.")
        run_root = (
            None
            if log.path is None
            else _validated_run_root(log.path, field_name="log.path")
        )
        resolved = _resolve_context(log.context, run_root=run_root)
        if not isinstance(resolved, cls):  # pragma: no cover - subclass guard
            raise TypeError("Resolved context has an incompatible class.")
        return resolved

    @classmethod
    def from_run_context(
        cls,
        context: RunContext,
        *,
        run_root: str | Path,
    ) -> ResolvedForwardContext:
        """Resolve authenticated live inputs under an explicit asset root."""
        if not isinstance(context, RunContext):
            raise TypeError("context must be a RunContext.")
        root = _validated_run_root(run_root, field_name="run_root")
        resolved = _resolve_context(context, run_root=root)
        if not isinstance(resolved, cls):  # pragma: no cover - subclass guard
            raise TypeError("Resolved context has an incompatible class.")
        return resolved

    def build_continuous_kernel(
        self,
        *,
        use_gpu: bool,
        gpu_device: str,
        gpu_dtype: str,
    ) -> ContinuousKernel:
        """Build one physical kernel with explicit execution settings."""
        if not isinstance(use_gpu, bool):
            raise TypeError("use_gpu must be a boolean.")
        if not isinstance(gpu_device, str) or not gpu_device:
            raise TypeError("gpu_device must be a nonempty string.")
        if gpu_dtype not in {"float32", "float64"}:
            raise ValueError("gpu_dtype must be 'float32' or 'float64'.")
        return continuous_kernel_from_observation_model(
            self.observation_model,
            obstacle_grid=(
                self.obstacle_grid
                if self.obstacle_attenuation_enabled
                else None
            ),
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            gpu_dtype=gpu_dtype,
        )


__all__ = ["ResolvedForwardContext"]
