"""Estimator-neutral acquisition sessions and fixed-plan execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from measurement.kernels import ShieldParams
from runtime.contracts import FULL_SPECTRUM_CONTRACT_HASH_METADATA_KEY
from runtime.forward_model_manifest import build_forward_model_manifest
from runtime.measurement_log import (
    MeasurementLog,
    MeasurementLogRecord,
    MeasurementLogStreamWriter,
)
from runtime.provenance import (
    canonical_json_bytes,
    repository_commit,
    repository_source_snapshot_sha256,
)
from sim.isaacsim_app.scene_builder import build_scene_description
from sim.protocol import SimulationCommand, SimulationObservation
from sim.runtime import (
    SimulationRuntime,
    create_simulation_runtime,
    load_runtime_config,
)
from spectrum.isotope_profiles import resolve_profile_model_runtime_config
from spectrum.transport_spectral import geometry_conditioned_model_from_runtime_config


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


@dataclass(frozen=True)
class AcquisitionAction:
    """Bind one simulator command to its causal station boundary."""

    station_id: int
    command: SimulationCommand
    station_complete: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AcquisitionAction":
        """Parse one strict fixed-plan action."""
        expected = {"station_id", "station_complete", "command"}
        if set(value) != expected:
            raise ValueError(
                "Acquisition action fields disagree with the schema; "
                f"missing={sorted(expected - set(value))}, "
                f"unknown={sorted(set(value) - expected)}."
            )
        station_id = value["station_id"]
        station_complete = value["station_complete"]
        command = value["command"]
        if isinstance(station_id, bool) or not isinstance(station_id, int):
            raise TypeError("station_id must be a JSON integer.")
        if station_id < 0:
            raise ValueError("station_id must be nonnegative.")
        if not isinstance(station_complete, bool):
            raise TypeError("station_complete must be a JSON boolean.")
        if not isinstance(command, dict):
            raise TypeError("command must be a JSON object.")
        return cls(
            station_id=station_id,
            command=SimulationCommand.from_dict(command),
            station_complete=station_complete,
        )


def estimator_neutral_physical_runtime_config(
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return physical runtime fields without estimator-owned settings."""
    return {
        str(key): value
        for key, value in runtime_config.items()
        if str(key) not in _ESTIMATOR_ONLY_KEYS
        and not str(key).startswith(_ESTIMATOR_ONLY_PREFIXES)
    }


def estimator_neutral_runtime_config(
    runtime_config: Mapping[str, Any],
    *,
    backend: str,
    isotopes: Sequence[str],
    run_root: Path,
) -> dict[str, Any]:
    """Return logged physical configuration without estimator-owned settings."""
    physical_config = estimator_neutral_physical_runtime_config(runtime_config)
    model = geometry_conditioned_model_from_runtime_config(
        physical_config,
        run_root=run_root,
    )
    profile_resolved = resolve_profile_model_runtime_config(
        physical_config,
        run_root=run_root,
    )
    resolved = estimator_neutral_physical_runtime_config(profile_resolved)
    resolved.pop("full_spectrum_generative_model_path", None)
    resolved.pop("full_spectrum_generative_model_file_sha256", None)
    resolved.pop("full_spectrum_model_registry_path", None)
    resolved.pop("full_spectrum_model_registry_file_sha256", None)
    resolved.pop("isotope_experiment_profile", None)
    resolved.pop("full_spectrum_profile_calibration_status", None)
    resolved["full_spectrum_generative_model"] = model.manifest_payload()
    resolved["full_spectrum_contract_hash_sha256"] = model.contract_hash_sha256
    resolved["simulation_runtime_schema_version"] = 1
    resolved["sim_backend"] = str(backend)
    resolved["candidate_isotopes"] = sorted(str(value) for value in isotopes)
    resolved["obstacle_attenuation_enabled"] = bool(
        runtime_config.get("obstacle_attenuation_enabled", True)
    )
    return resolved


class ObservationSession:
    """Execute and durably record observations before estimator ingestion."""

    def __init__(
        self,
        *,
        simulation_runtime: SimulationRuntime,
        writer: MeasurementLogStreamWriter,
        full_spectrum_contract_hash_sha256: str,
    ) -> None:
        """Retain the sole simulator and log-writer handles for one run."""
        self.simulation_runtime = simulation_runtime
        self.writer = writer
        self.contract_hash = full_spectrum_contract_hash_sha256
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
        """Publish the immutable MeasurementLog and close simulator resources."""
        if self._closed:
            raise RuntimeError("Acquisition session is already closed.")
        try:
            return self.writer.finalize()
        finally:
            self.simulation_runtime.close()
            self._closed = True

    def close(self) -> None:
        """Close simulator resources without publishing an incomplete run."""
        if not self._closed:
            self.simulation_runtime.close()
            self._closed = True


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one finite JSON object from a plan-owned file."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def run_acquisition_plan(plan_path: str | Path) -> MeasurementLog:
    """Execute a fixed action plan through the same live-session boundary."""
    path = Path(plan_path).expanduser().resolve()
    plan = _load_json_object(path)
    expected = {
        "schema_version",
        "run_id",
        "backend",
        "runtime_config_path",
        "output_dir",
        "environment",
        "scene",
        "isotopes",
        "actions",
        "metadata",
        "obstacle_layout_path",
    }
    if set(plan) != expected or plan.get("schema_version") != 1:
        raise ValueError("Acquisition plan must match schema version 1 exactly.")
    base = path.parent
    config_path = (base / str(plan["runtime_config_path"])).resolve()
    output_dir = (base / str(plan["output_dir"])).resolve()
    raw_config = load_runtime_config(config_path)
    isotopes = tuple(sorted(str(value) for value in plan["isotopes"]))
    if not isotopes or len(set(isotopes)) != len(isotopes):
        raise ValueError("Plan isotopes must be nonempty and unique.")
    backend = str(plan["backend"])
    logged_config = estimator_neutral_runtime_config(
        raw_config,
        backend=backend,
        isotopes=isotopes,
        run_root=config_path.parents[2],
    )
    environment = plan["environment"]
    scene = plan["scene"]
    if not isinstance(environment, dict) or not isinstance(scene, dict):
        raise TypeError("Plan environment and scene must be JSON objects.")
    commit = repository_commit(Path(__file__).resolve().parents[2])
    if len(commit) != 40:
        raise RuntimeError("Acquisition runtime must execute from a Git commit.")
    source_snapshot = repository_source_snapshot_sha256(
        Path(__file__).resolve().parents[2]
    )
    run_metadata = dict(plan["metadata"])
    run_metadata["repository_source_snapshot_sha256"] = source_snapshot
    resolved_hash = sha256(canonical_json_bytes(logged_config)).hexdigest()
    forward = build_forward_model_manifest(
        runtime_config=logged_config,
        environment=environment,
        obstacle_layout_path=plan["obstacle_layout_path"],
        isotopes=isotopes,
        repository_commit=commit,
        resolved_config_sha256=resolved_hash,
        repository_root=Path(__file__).resolve().parents[2],
    )
    writer = MeasurementLogStreamWriter(
        output_dir,
        run_id=str(plan["run_id"]),
        repository_commit=commit,
        runtime_config=logged_config,
        environment=environment,
        forward_model_manifest=forward,
        isotopes=isotopes,
        metadata=run_metadata,
        obstacle_layout_path=plan["obstacle_layout_path"],
        source_layout_path=None,
    )
    scene_description = build_scene_description(scene)
    simulation_runtime = create_simulation_runtime(
        backend,
        sources=scene_description.to_point_sources(),
        mu_by_isotope={},
        shield_params=ShieldParams(),
        runtime_config=raw_config,
        runtime_config_path=config_path,
    )
    session = ObservationSession(
        simulation_runtime=simulation_runtime,
        writer=writer,
        full_spectrum_contract_hash_sha256=str(
            logged_config["full_spectrum_contract_hash_sha256"]
        ),
    )
    try:
        session.reset(scene)
        actions = plan["actions"]
        if not isinstance(actions, list) or not actions:
            raise ValueError("Acquisition plan requires nonempty actions.")
        for raw_action in actions:
            if not isinstance(raw_action, dict):
                raise TypeError("Every acquisition action must be an object.")
            session.step(AcquisitionAction.from_mapping(raw_action))
        return session.finalize()
    except BaseException:
        session.close()
        raise


__all__ = [
    "AcquisitionAction",
    "ObservationSession",
    "estimator_neutral_physical_runtime_config",
    "estimator_neutral_runtime_config",
    "run_acquisition_plan",
]
