"""Estimator-neutral acquisition sessions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from measurement.observation_model import require_production_model_approval
from runtime.contracts import FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY
from runtime.measurement_log import (
    MeasurementLog,
    MeasurementLogRecord,
    MeasurementLogStreamWriter,
)
from runtime.provenance import strict_json_loads
from sim.geant4_app.execution_environment import (
    native_execution_environment_bundle_sha256,
)
from sim.protocol import SimulationCommand, SimulationObservation
from sim.runtime import (
    SimulationRuntime,
    validate_production_runtime_config,
)
from spectrum.isotope_profiles import resolve_profile_model_runtime_config
from spectrum.full_spectrum_acceptance_runner import (
    acceptance_implementation_bundle_sha256,
)
from spectrum.transport_spectral import (
    GeometryConditionedSpectralModel,
    geometry_conditioned_model_from_runtime_config,
    with_catalog_independent_production_approval,
)


_ESTIMATOR_ONLY_PREFIXES = (
    "adaptive_mission_",
    "baseline_",
    "dss_",
    "evaluation_",
    "history_",
    "joint_",
    "measurement_pose_",
    "measurement_route_",
    "mission_stop_",
    "orientation_",
    "path_",
    "pf_",
    "pose_selection_",
    "posterior_",
    "structural_",
    "surface_diagnostic_",
    "surface_observability_",
)
_ESTIMATOR_ONLY_KEYS = frozenset(
    {
        "cui_truth_display_mode",
        "estimator_profile",
        "gpu_device",
        "gpu_dtype",
        "history_estimate_interval",
        "max_temper_steps",
        "min_delta_beta",
        "num_particles",
        "python_worker_count",
        "pure_pf_schema_version",
        "target_ess_ratio",
        "use_gpu",
        "variable_cardinality",
    }
)
_TRANSPORT_PROVENANCE_KEYS = frozenset(
    {
        "accelerated_weighted_transport_enable",
        "background_cps",
        "background_spectrum_model_id",
        "dead_time_observed_scale",
        "dead_time_tau_s",
        "detector_response_applied_in_native",
        "detector_response_boundary_state",
        "detector_response_coincidence_pulse_count",
        "detector_response_coincidence_semantics",
        "detector_response_conditioning",
        "detector_response_incident_entry_count",
        "detector_response_multi_entry_pulse_count",
        "detector_response_operator_binary_sha256",
        "detector_response_registered_entry_count",
        "detector_response_sampling_contract_sha256",
        "detector_response_sampling_mode",
        "detector_response_sampling_model",
        "detector_scoring_mode",
        "dwell_time_s",
        "emission_model",
        "engine_mode",
        "history_thinning_enabled",
        "intensity_cps_1m_definition",
        "line_intensities_normalized",
        "multithreaded_run_manager",
        "num_primaries",
        "physics_profile",
        "primary_history_weight",
        "primary_sampling_fraction",
        "requested_threads",
        "scene_hash",
        "secondary_transport_mode",
        "source_rate_model",
        "spectrum_bin_count",
        "spectrum_bin_width_keV",
        "spectrum_energy_max_keV",
        "spectrum_energy_min_keV",
        "theory_tvl_attenuation",
        "transport_history_mode",
        "weighted_transport",
    }
)
_DETECTOR_QUATERNION_ABSOLUTE_TOLERANCE = 1.0e-10
_PRODUCTION_BACKEND = "geant4"
_RUNTIME_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_APPLICATION_APPROVAL_MODEL = (
    _RUNTIME_REPOSITORY_ROOT
    / "configs/geant4/models/profiles/unconditioned_cs_co.json"
)


def _attach_canonical_algorithm_approval(
    model: GeometryConditionedSpectralModel,
) -> GeometryConditionedSpectralModel:
    """Authorize an in-domain profile from the canonical all-64 evidence."""
    if model.production_ready:
        return model
    path = _CANONICAL_APPLICATION_APPROVAL_MODEL.absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise RuntimeError(
            "Canonical full-spectrum application approval must be one regular "
            "repository file without symlink traversal."
        )
    raw_bytes = path.read_bytes()
    payload = strict_json_loads(raw_bytes)
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            "Canonical full-spectrum application approval is not a JSON object."
        )
    approved_source = GeometryConditionedSpectralModel.from_manifest_payload(
        payload,
        detector_green_operator=model.detector_green_operator,
    )
    return with_catalog_independent_production_approval(
        model,
        approved_source=approved_source,
    )


def _native_executable_sha256(
    runtime_config: Mapping[str, Any],
) -> str:
    """Hash the exact regular executable selected by production runtime."""
    raw_path = runtime_config.get("executable_path")
    if type(raw_path) is not str or not raw_path:
        raise TypeError("Production executable_path must be a nonempty JSON string.")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = _RUNTIME_REPOSITORY_ROOT / candidate
    candidate = candidate.absolute()
    if candidate.is_symlink():
        raise ValueError(
            f"Production native Geant4 executable cannot be a symlink: {candidate}."
        )
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Production native Geant4 executable is missing: {candidate}."
        ) from exc
    if resolved != candidate or not candidate.is_file():
        raise ValueError(
            "Production native Geant4 executable must be one exact regular "
            f"file without symlink traversal: {candidate}."
        )
    status_before = candidate.stat()
    if status_before.st_mode & 0o111 == 0:
        raise PermissionError(
            f"Production native Geant4 executable is not executable: {candidate}."
        )
    digest = sha256(candidate.read_bytes()).hexdigest()
    status_after = candidate.stat()
    if (
        status_after.st_dev != status_before.st_dev
        or status_after.st_ino != status_before.st_ino
        or status_after.st_size != status_before.st_size
        or status_after.st_mtime_ns != status_before.st_mtime_ns
    ):
        raise RuntimeError(
            "Production native Geant4 executable changed during preflight."
        )
    return digest


def _require_approved_execution_bundle(
    runtime_config: Mapping[str, Any],
    *,
    model: GeometryConditionedSpectralModel,
) -> None:
    """Bind production execution to the exact independently approved build."""
    validation = model.validation_manifest
    if not isinstance(validation, Mapping):
        raise RuntimeError(
            "Approved model is missing independent execution provenance."
        )
    expected_native = validation.get("native_executable_sha256")
    expected_runtime_config = validation.get("runtime_config_sha256")
    expected_native_environment = validation.get("native_execution_environment_sha256")
    expected_implementation = validation.get("implementation_bundle_sha256")
    for field_name, value in (
        ("runtime_config_sha256", expected_runtime_config),
        ("native_executable_sha256", expected_native),
        (
            "native_execution_environment_sha256",
            expected_native_environment,
        ),
        ("implementation_bundle_sha256", expected_implementation),
    ):
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(
                f"Approved model has invalid execution provenance field {field_name}."
            )
    actual_native = _native_executable_sha256(runtime_config)
    if actual_native != expected_native:
        raise RuntimeError(
            "Production native Geant4 executable SHA-256 differs from the "
            "independently approved validation build."
        )
    raw_executable_path = runtime_config["executable_path"]
    if type(raw_executable_path) is not str:
        raise TypeError("Production executable_path must be a JSON string.")
    executable_path = Path(raw_executable_path).expanduser()
    if not executable_path.is_absolute():
        executable_path = _RUNTIME_REPOSITORY_ROOT / executable_path
    actual_native_environment = native_execution_environment_bundle_sha256(
        executable_path.absolute()
    )
    if actual_native_environment != expected_native_environment:
        raise RuntimeError(
            "Production native Geant4 execution-environment SHA-256 differs "
            "from the independently approved validation environment."
        )
    actual_implementation = acceptance_implementation_bundle_sha256(
        _RUNTIME_REPOSITORY_ROOT
    )
    if actual_implementation != expected_implementation:
        raise RuntimeError(
            "Production implementation bundle SHA-256 differs from the "
            "independently approved validation implementation."
        )


def _commanded_detector_quaternion_wxyz(yaw_rad: float) -> np.ndarray:
    """Return the unit WXYZ quaternion commanded for a base Z-axis yaw."""
    half_yaw = 0.5 * float(yaw_rad)
    return np.asarray(
        [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
        dtype=np.float64,
    )


def _require_commanded_detector_orientation(
    quaternion_wxyz: Sequence[float],
    *,
    target_base_yaw_rad: float,
) -> None:
    """Reject a simulator orientation that differs from the commanded yaw."""
    observed = np.asarray(quaternion_wxyz, dtype=np.float64)
    if observed.shape != (4,) or np.any(~np.isfinite(observed)):
        raise RuntimeError(
            "Simulator response detector quaternion is not a finite WXYZ vector."
        )
    observed_norm = float(np.linalg.norm(observed))
    if not np.isclose(
        observed_norm,
        1.0,
        rtol=0.0,
        atol=_DETECTOR_QUATERNION_ABSOLUTE_TOLERANCE,
    ):
        raise RuntimeError(
            "Simulator response detector quaternion is not unit normalized."
        )
    expected = _commanded_detector_quaternion_wxyz(target_base_yaw_rad)
    same_sign = np.allclose(
        observed,
        expected,
        rtol=0.0,
        atol=_DETECTOR_QUATERNION_ABSOLUTE_TOLERANCE,
    )
    opposite_sign = np.allclose(
        observed,
        -expected,
        rtol=0.0,
        atol=_DETECTOR_QUATERNION_ABSOLUTE_TOLERANCE,
    )
    if not same_sign and not opposite_sign:
        raise RuntimeError(
            "Simulator response detector orientation differs from its command."
        )


@dataclass(frozen=True)
class AcquisitionAction:
    """Bind one simulator command to its causal station boundary."""

    station_id: int
    command: SimulationCommand
    station_complete: bool


def estimator_neutral_physical_runtime_config(
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject estimator-owned fields instead of silently removing them."""
    if not isinstance(runtime_config, Mapping) or any(
        not isinstance(key, str) for key in runtime_config
    ):
        raise TypeError("Runtime configuration must be a string-keyed object.")
    retired = sorted(
        key
        for key in runtime_config
        if key in _ESTIMATOR_ONLY_KEYS or key.startswith(_ESTIMATOR_ONLY_PREFIXES)
    )
    if retired:
        raise ValueError(
            "Production runtime configuration contains estimator-owned or "
            f"retired fields: {retired}."
        )
    return dict(runtime_config)


def require_production_runtime_preflight(
    runtime_config: Mapping[str, Any],
    *,
    requested_backend: object,
) -> GeometryConditionedSpectralModel:
    """Authenticate the canonical Geant4 config and approved model asset."""
    validate_production_runtime_config(runtime_config)
    configured_backend = runtime_config["backend"]
    if requested_backend != _PRODUCTION_BACKEND:
        raise ValueError(
            "Production acquisition requires backend='geant4'; "
            f"got {requested_backend!r}."
        )
    if configured_backend != requested_backend:
        raise ValueError(
            "Production acquisition backend differs from runtime config: "
            f"requested={requested_backend!r}, configured={configured_backend!r}."
        )
    required_values = {
        "auto_start_sidecar": True,
        "author_obstacle_prims": True,
        "author_room_boundary_prims": True,
        "detector_scoring_mode": "incident_gamma_energy",
        "line_resolved_shield_attenuation": True,
        "obstacle_attenuation_enabled": True,
        "primary_emission_model": "independent_gamma_lines",
        "primary_sampling_fraction": 1.0,
        "sample_detector_response": True,
        "secondary_transport_mode": "full_transport",
        "source_bias_cone_policy": "detector_covering",
        "source_bias_isotropic_fraction": 1.0,
        "source_bias_mode": "detector_cone",
        "source_rate_model": "detector_cps_1m",
    }
    mismatches = {
        name: runtime_config[name]
        for name, expected in required_values.items()
        if runtime_config[name] != expected
        or type(runtime_config[name]) is not type(expected)
    }
    if mismatches:
        raise ValueError(
            "Production full-spectrum transport invariants differ from the "
            f"canonical values: {mismatches}."
        )
    from sim.geant4_app.app import Geant4AppConfig

    Geant4AppConfig.from_dict(dict(runtime_config))
    model = geometry_conditioned_model_from_runtime_config(
        runtime_config,
        run_root=_RUNTIME_REPOSITORY_ROOT,
    )
    model = _attach_canonical_algorithm_approval(model)
    require_production_model_approval(model)
    background_cps = runtime_config["background_cps"]
    if (
        isinstance(background_cps, bool)
        or not isinstance(background_cps, (int, float))
        or not np.isfinite(float(background_cps))
        or float(background_cps) < 0.0
    ):
        raise TypeError("Production background_cps must be finite and nonnegative.")
    manifest = model.manifest_payload()
    if float(background_cps) != float(model.background_rate_cps):
        raise ValueError(
            "Production background_cps differs from the approved model rate."
        )
    if runtime_config["background_spectrum_model_id"] != manifest.get(
        "background_model"
    ):
        raise ValueError(
            "Production background_spectrum_model_id differs from the approved "
            "model manifest."
        )
    _require_approved_execution_bundle(runtime_config, model=model)
    return model


def estimator_neutral_runtime_config(
    runtime_config: Mapping[str, Any],
    *,
    backend: str,
    isotopes: Sequence[str],
    run_root: Path | None = None,
) -> dict[str, Any]:
    """Return logged physical configuration without estimator-owned settings."""
    if run_root is not None and Path(run_root).resolve() != _RUNTIME_REPOSITORY_ROOT:
        raise ValueError(
            "Production model assets must resolve from the runtime repository root."
        )
    physical_config = estimator_neutral_physical_runtime_config(runtime_config)
    model = require_production_runtime_preflight(
        physical_config,
        requested_backend=backend,
    )
    profile_resolved = resolve_profile_model_runtime_config(
        physical_config,
        run_root=_RUNTIME_REPOSITORY_ROOT,
    )
    resolved = estimator_neutral_physical_runtime_config(profile_resolved)
    resolved.pop("full_spectrum_generative_model_path", None)
    resolved.pop("full_spectrum_generative_model_file_sha256", None)
    resolved.pop("full_spectrum_model_registry_path", None)
    resolved.pop("full_spectrum_model_registry_file_sha256", None)
    resolved.pop("isotope_experiment_profile", None)
    resolved["full_spectrum_generative_model"] = model.manifest_payload()
    resolved["full_spectrum_contract_hash_sha256"] = model.contract_hash_sha256
    resolved["simulation_runtime_schema_version"] = 1
    resolved["sim_backend"] = str(backend)
    resolved["candidate_isotopes"] = sorted(str(value) for value in isotopes)
    obstacle_attenuation = runtime_config["obstacle_attenuation_enabled"]
    if type(obstacle_attenuation) is not bool:
        raise TypeError("obstacle_attenuation_enabled must be an exact JSON boolean.")
    resolved["obstacle_attenuation_enabled"] = obstacle_attenuation
    return resolved


def production_energy_bin_edges_keV(
    logged_runtime_config: Mapping[str, Any],
) -> np.ndarray:
    """Return the exact production energy edges from the approved model."""
    manifest = logged_runtime_config.get("full_spectrum_generative_model")
    if not isinstance(manifest, Mapping):
        raise RuntimeError(
            "Logged runtime config has no approved full-spectrum model manifest."
        )
    raw_count = manifest.get("energy_bin_count")
    if isinstance(raw_count, bool) or not isinstance(raw_count, int):
        raise RuntimeError("Approved model energy_bin_count must be an integer.")
    count = int(raw_count)
    if count <= 0:
        raise RuntimeError("Approved model energy_bin_count must be positive.")
    numeric_fields: dict[str, float] = {}
    for name in ("energy_min_keV", "energy_max_keV", "bin_width_keV"):
        raw = manifest.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RuntimeError(f"Approved model {name} must be numeric.")
        value = float(raw)
        if not np.isfinite(value):
            raise RuntimeError(f"Approved model {name} must be finite.")
        numeric_fields[name] = value
    width = numeric_fields["bin_width_keV"]
    if width <= 0.0:
        raise RuntimeError("Approved model bin_width_keV must be positive.")
    edges = numeric_fields["energy_min_keV"] + width * np.arange(
        count + 1,
        dtype=np.float64,
    )
    if edges[-2] != numeric_fields["energy_max_keV"]:
        raise RuntimeError(
            "Approved model energy range does not match its bin count and width."
        )
    edges.setflags(write=False)
    return edges


def production_native_execution_digests(
    logged_runtime_config: Mapping[str, Any],
) -> tuple[str, str, str]:
    """Return approved executable, environment, and Python bundle digests."""
    model_manifest = logged_runtime_config.get("full_spectrum_generative_model")
    if not isinstance(model_manifest, Mapping):
        raise RuntimeError(
            "Logged runtime config has no approved full-spectrum model manifest."
        )
    validation = model_manifest.get("validation")
    if not isinstance(validation, Mapping):
        raise RuntimeError("Approved full-spectrum model has no validation provenance.")
    values: list[str] = []
    for field_name in (
        "native_executable_sha256",
        "native_execution_environment_sha256",
        "implementation_bundle_sha256",
    ):
        value = validation.get(field_name)
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"Approved model has invalid {field_name} provenance.")
        values.append(value)
    return values[0], values[1], values[2]


class ObservationSession:
    """Execute and durably record observations before estimator ingestion."""

    def __init__(
        self,
        *,
        simulation_runtime: SimulationRuntime,
        writer: MeasurementLogStreamWriter,
        full_spectrum_contract_hash_sha256: str,
        energy_bin_edges_keV: np.ndarray,
    ) -> None:
        """Retain the sole simulator and log-writer handles for one run."""
        if (
            not isinstance(full_spectrum_contract_hash_sha256, str)
            or len(full_spectrum_contract_hash_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in full_spectrum_contract_hash_sha256
            )
        ):
            raise ValueError("Full-spectrum contract hash must be a SHA-256 digest.")
        expected_edges = np.asarray(energy_bin_edges_keV)
        if (
            expected_edges.dtype != np.dtype(np.float64)
            or expected_edges.ndim != 1
            or expected_edges.size < 2
            or np.any(~np.isfinite(expected_edges))
            or np.any(np.diff(expected_edges) <= 0.0)
        ):
            raise ValueError(
                "Production energy_bin_edges_keV must be exact increasing float64."
            )
        self.simulation_runtime = simulation_runtime
        self.writer = writer
        self.contract_hash = full_spectrum_contract_hash_sha256
        self.energy_bin_edges_keV = expected_edges.copy()
        self.energy_bin_edges_keV.setflags(write=False)
        self._closed = False

    def reset(self, scene_payload: Mapping[str, Any]) -> None:
        """Reset the private simulator scene without exposing truth to estimators."""
        if self._closed:
            raise RuntimeError("Cannot reset a closed acquisition session.")
        self.simulation_runtime.reset(dict(scene_payload))

    def step(self, action: AcquisitionAction) -> SimulationObservation:
        """Run, persist, and then return exactly one raw observation."""
        if self._closed:
            raise RuntimeError("Cannot step a closed acquisition session.")
        expected_step = len(self.writer.records)
        if action.command.step_id != expected_step:
            raise ValueError(
                f"command step_id {action.command.step_id} does not equal "
                f"causal action index {expected_step}."
            )
        observation = self.simulation_runtime.step(action.command)
        if observation.step_id != action.command.step_id:
            raise RuntimeError("Simulator response step_id differs from its command.")
        if tuple(observation.detector_pose_xyz) != action.command.target_pose_xyz:
            raise RuntimeError(
                "Simulator response detector pose differs from its command."
            )
        _require_commanded_detector_orientation(
            observation.detector_quat_wxyz,
            target_base_yaw_rad=action.command.target_base_yaw_rad,
        )
        if (
            observation.fe_orientation_index != action.command.fe_orientation_index
            or observation.pb_orientation_index != action.command.pb_orientation_index
        ):
            raise RuntimeError(
                "Simulator response shield orientations differ from its command."
            )
        response_dwell = observation.metadata.get("dwell_time_s")
        if (
            isinstance(response_dwell, bool)
            or not isinstance(response_dwell, (int, float))
            or not np.isfinite(float(response_dwell))
            or float(response_dwell) != action.command.dwell_time_s
        ):
            raise RuntimeError(
                "Simulator response dwell_time_s differs from its command."
            )
        response_edges = np.asarray(observation.energy_bin_edges_keV)
        if response_edges.dtype != np.dtype(np.float64) or not np.array_equal(
            response_edges, self.energy_bin_edges_keV
        ):
            raise RuntimeError(
                "Simulator response energy axis differs from the approved model."
            )
        raw_spectrum = np.asarray(observation.spectrum_counts)
        if (
            raw_spectrum.ndim != 1
            or not np.issubdtype(raw_spectrum.dtype, np.integer)
            or np.any(raw_spectrum < 0)
            or np.any(raw_spectrum > np.iinfo(np.int64).max)
        ):
            raise RuntimeError(
                "Production observation must be an exact nonnegative integer spectrum."
            )
        metadata = {
            key: observation.metadata[key]
            for key in sorted(_TRANSPORT_PROVENANCE_KEYS)
            if key in observation.metadata
        }
        metadata[FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY] = self.contract_hash
        if action.command.travel_waypoints_xyz:
            metadata["travel_waypoints_xyz"] = [
                [float(value) for value in waypoint]
                for waypoint in action.command.travel_waypoints_xyz
            ]
        self.writer.append_before_update(
            MeasurementLogRecord(
                step_id=expected_step,
                action_id=expected_step,
                station_id=action.station_id,
                detector_pose_xyz=observation.detector_pose_xyz,
                detector_quat_wxyz=observation.detector_quat_wxyz,
                fe_orientation_index=observation.fe_orientation_index,
                pb_orientation_index=observation.pb_orientation_index,
                live_time_s=action.command.dwell_time_s,
                travel_time_s=action.command.travel_time_s,
                shield_actuation_time_s=action.command.shield_actuation_time_s,
                energy_bin_edges_keV=np.asarray(
                    observation.energy_bin_edges_keV,
                    dtype=np.float64,
                ),
                spectrum_counts=np.asarray(raw_spectrum, dtype=np.int64),
                metadata=metadata,
            )
        )
        if action.station_complete:
            self.writer.mark_station_complete_before_update(action.station_id)
        return observation

    def finalize(self) -> MeasurementLog:
        """Publish only after transport confirms a graceful clean shutdown."""
        if self._closed:
            raise RuntimeError("Acquisition session is already closed.")
        shutdown_failure: BaseException | None = None
        try:
            self.simulation_runtime.close()
        except BaseException as failure:
            shutdown_failure = failure
        if shutdown_failure is not None:
            try:
                self.writer.abort()
            except BaseException as cleanup_failure:
                shutdown_failure.add_note(
                    "MeasurementLog WAL cleanup also failed: "
                    f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                )
            self._closed = True
            raise shutdown_failure
        try:
            return self.writer.finalize()
        except BaseException as publication_failure:
            try:
                self.writer.abort()
            except BaseException as cleanup_failure:
                publication_failure.add_note(
                    "MeasurementLog WAL cleanup also failed: "
                    f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                )
            raise
        finally:
            self._closed = True

    def close(self) -> None:
        """Abort an incomplete run and remove its private MeasurementLog WAL."""
        if self._closed:
            return
        runtime_failure: BaseException | None = None
        try:
            self.simulation_runtime.close()
        except BaseException as exc:
            runtime_failure = exc
        try:
            self.writer.abort()
        except BaseException as cleanup_exc:
            if runtime_failure is not None:
                runtime_failure.add_note(
                    "MeasurementLog WAL cleanup also failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
                raise runtime_failure
            raise
        finally:
            self._closed = True
        if runtime_failure is not None:
            raise runtime_failure


def _open_owned_observation_session(
    *,
    simulation_runtime: SimulationRuntime,
    output_dir: str | Path,
    writer_arguments: Mapping[str, Any],
    full_spectrum_contract_hash_sha256: str,
    energy_bin_edges_keV: np.ndarray,
) -> ObservationSession:
    """Transfer a runtime and a new WAL writer into one owning session."""
    writer: MeasurementLogStreamWriter | None = None
    try:
        writer = MeasurementLogStreamWriter(output_dir, **dict(writer_arguments))
        return ObservationSession(
            simulation_runtime=simulation_runtime,
            writer=writer,
            full_spectrum_contract_hash_sha256=(full_spectrum_contract_hash_sha256),
            energy_bin_edges_keV=energy_bin_edges_keV,
        )
    except BaseException as startup_failure:
        cleanup_failures: list[BaseException] = []
        try:
            simulation_runtime.close()
        except BaseException as cleanup_failure:
            cleanup_failures.append(cleanup_failure)
        if writer is not None:
            try:
                writer.abort()
            except BaseException as cleanup_failure:
                cleanup_failures.append(cleanup_failure)
        for cleanup_failure in cleanup_failures:
            startup_failure.add_note(
                "Acquisition startup cleanup also failed: "
                f"{type(cleanup_failure).__name__}: {cleanup_failure}"
            )
        raise


def _close_observation_session_after_failure(
    session: ObservationSession,
    failure: BaseException,
) -> None:
    """Close an incomplete session without replacing its primary failure."""
    try:
        session.close()
    except BaseException as cleanup_failure:
        failure.add_note(
            "Acquisition cleanup also failed: "
            f"{type(cleanup_failure).__name__}: {cleanup_failure}"
        )


__all__ = [
    "AcquisitionAction",
    "ObservationSession",
    "estimator_neutral_physical_runtime_config",
    "estimator_neutral_runtime_config",
    "production_energy_bin_edges_keV",
    "production_native_execution_digests",
    "require_production_runtime_preflight",
]
