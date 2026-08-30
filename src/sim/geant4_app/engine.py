"""Geant4-side observation engine backed by an external Geant4 executable."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import selectors
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

from measurement.shielding import (
    SHIELD_POSE_CONTRACT_ID,
    SHIELD_POSE_CONTRACT_SHA256,
    octant_index_from_normal,
    physical_shield_normal_from_orientation_index,
)
from measurement.source_boundary import (
    activity_surface_source_runtime_contract_sha256,
    surface_source_runtime_contract_sha256,
)
from sim.geant4_app.io_format import (
    NATIVE_ACTION_IDENTITY_CONTRACT_ID,
    read_response_file,
    write_request_file,
    write_scene_file,
)
from sim.geant4_app.execution_environment import (
    require_native_execution_bundle,
)
from sim.geant4_app.scene_export import ExportedGeant4Scene
from sim.radiation_visualization import (
    RadiationVisualizationConfig,
    build_visualization_metadata_from_scene,
)
from sim.shield_geometry import shield_normal_from_quaternion_wxyz
from spectrum.detector_green_operator import DetectorGreenOperator
from spectrum.library import Nuclide, default_library, nuclide_catalog_sha256


def validate_native_scene_identity(
    metadata: dict[str, Any],
    scene: ExportedGeant4Scene,
    *,
    expected_nuclide_catalog_sha256: str | None = None,
    expected_detector_green_reference_efficiency_by_isotope: (
        Mapping[str, float] | None
    ) = None,
) -> None:
    """Authenticate the exact scene and source payload parsed by native Geant4."""
    activity_sources = [
        source for source in scene.sources if source.activity_bq is not None
    ]
    if activity_sources and len(activity_sources) != len(scene.sources):
        raise RuntimeError(
            "Native scene identity cannot mix activity and detector-cps sources."
        )
    strength_key = "activity_bq" if activity_sources else "intensity_cps_1m"
    contract_function = (
        activity_surface_source_runtime_contract_sha256
        if activity_sources
        else surface_source_runtime_contract_sha256
    )
    expected_entries = [
        {
            "isotope": source.isotope,
            "position": list(source.anchor_position_xyz),
            "transport_position": list(source.position_xyz),
            strength_key: getattr(source, strength_key),
            "surface_chart_id": source.surface_chart_id,
            "surface_uv": list(source.surface_uv),
            "surface_normal": list(source.surface_normal_xyz),
            "surface_emission_policy_sha256": (source.surface_emission_policy_sha256),
        }
        for source in scene.sources
    ]
    expected_efficiencies = (
        None
        if expected_detector_green_reference_efficiency_by_isotope is None
        else dict(expected_detector_green_reference_efficiency_by_isotope)
    )
    expected_isotopes = {str(entry["isotope"]) for entry in expected_entries}
    if expected_efficiencies is not None and (
        activity_sources
        or set(expected_efficiencies) != expected_isotopes
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) <= 0.0
            or float(value) > 1.0
            for value in expected_efficiencies.values()
        )
    ):
        raise RuntimeError(
            "Expected detector Green source-rate efficiencies are invalid."
        )
    expected_source_hash = contract_function(expected_entries)
    expected = {
        "backend": "geant4",
        "engine_mode": "external",
        "scene_hash": scene.scene_hash,
        "surface_source_contract_sha256": expected_source_hash,
        "nuclide_catalog_sha256": (
            nuclide_catalog_sha256()
            if expected_nuclide_catalog_sha256 is None
            else expected_nuclide_catalog_sha256
        ),
    }
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual != expected_value:
            raise RuntimeError(
                "Native Geant4 did not authenticate the exported scene "
                f"identity for {key}: expected {expected_value!r}, "
                f"got {actual!r}."
            )
    coincidence_window = metadata.get("detector_coincidence_window_s")
    if (
        isinstance(coincidence_window, bool)
        or not isinstance(coincidence_window, (int, float))
        or not np.isfinite(float(coincidence_window))
        or not np.isclose(
            float(coincidence_window),
            float(scene.detector_model.coincidence_window_s),
            rtol=0.0,
            atol=1.0e-15,
        )
    ):
        raise RuntimeError(
            "Native Geant4 detector coincidence window does not match the "
            "exported detector model."
        )
    native_source_count = metadata.pop(
        "native_surface_source_count",
        None,
    )
    if (
        isinstance(native_source_count, bool)
        or not isinstance(native_source_count, int)
        or native_source_count != len(expected_entries)
    ):
        raise RuntimeError(
            "Native Geant4 did not report the exact parsed source count."
        )
    native_entries: list[dict[str, object]] = []
    scalar_fields = (
        "isotope",
        strength_key,
        "surface_chart_id",
        "surface_emission_policy_sha256",
    )
    vector_fields = {
        "position": ("anchor_x", "anchor_y", "anchor_z"),
        "transport_position": (
            "transport_x",
            "transport_y",
            "transport_z",
        ),
        "surface_uv": ("surface_u", "surface_v"),
        "surface_normal": (
            "surface_normal_x",
            "surface_normal_y",
            "surface_normal_z",
        ),
    }
    for source_index in range(native_source_count):
        prefix = f"native_surface_source_{source_index}_"
        entry = {field: metadata.pop(prefix + field, None) for field in scalar_fields}
        entry.update(
            {
                field: [
                    metadata.pop(prefix + component, None) for component in components
                ]
                for field, components in vector_fields.items()
            }
        )
        if expected_efficiencies is not None:
            isotope = str(expected_entries[source_index]["isotope"])
            actual_efficiency = metadata.pop(
                prefix + "detector_green_reference_efficiency",
                None,
            )
            if (
                isinstance(actual_efficiency, bool)
                or not isinstance(actual_efficiency, (int, float))
                or not np.isfinite(float(actual_efficiency))
                or not np.isclose(
                    float(actual_efficiency),
                    float(expected_efficiencies[isotope]),
                    rtol=1.0e-14,
                    atol=1.0e-15,
                )
            ):
                raise RuntimeError(
                    "Native detector Green reference efficiency differs from "
                    f"the authenticated operator/catalog value for {isotope!r}."
                )
        native_entries.append(entry)
    try:
        native_source_hash = contract_function(native_entries)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Native Geant4 parsed-source identity payload is invalid."
        ) from exc
    unexpected_native_fields = sorted(
        key for key in metadata if key.startswith("native_surface_source_")
    )
    if unexpected_native_fields:
        raise RuntimeError(
            "Native Geant4 parsed-source identity contains unexpected fields: "
            f"{unexpected_native_fields}."
        )
    if native_source_hash != expected_source_hash:
        raise RuntimeError(
            "Native Geant4 parsed source strengths or transport positions "
            "differ from the exported scene."
        )


def validate_native_shield_pose_identity(
    metadata: dict[str, Any],
    request: Geant4StepRequest,
) -> None:
    """Prove native shield placement matches the shared octant contract."""
    fe_index, pb_index = request.resolved_orientation_indices()
    if metadata.get("shield_pose_contract_id") != SHIELD_POSE_CONTRACT_ID:
        raise RuntimeError("Native Geant4 shield-pose contract is incompatible.")
    if metadata.get("shield_pose_contract_sha256") != SHIELD_POSE_CONTRACT_SHA256:
        raise RuntimeError("Native Geant4 shield-pose hash is incompatible.")
    for kind, index in (("fe", fe_index), ("pb", pb_index)):
        try:
            native_index = int(metadata[f"{kind}_orientation_index"])
            native_normal = np.asarray(
                [
                    float(metadata[f"{kind}_shield_normal_{axis}"])
                    for axis in ("x", "y", "z")
                ],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Native Geant4 omitted valid {kind} shield-pose identity."
            ) from exc
        expected = physical_shield_normal_from_orientation_index(index)
        if native_index != index or not np.allclose(
            native_normal,
            expected,
            rtol=0.0,
            atol=2.0e-6,
        ):
            raise RuntimeError(
                f"Native Geant4 {kind} shield placement disagrees with "
                "the requested octant orientation."
            )


def _required_native_action_number(
    metadata: Mapping[str, Any],
    key: str,
) -> float:
    """Return one finite non-boolean native action metadata number."""
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Native Geant4 omitted action field {key!r}.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise RuntimeError(f"Native Geant4 action field {key!r} is nonfinite.")
    return parsed


def _required_native_action_integer(
    metadata: Mapping[str, Any],
    key: str,
) -> int:
    """Return one exact non-boolean native action metadata integer."""
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(
            f"Native Geant4 action field {key!r} is not an exact integer."
        )
    return value


def validate_native_action_identity(
    metadata: Mapping[str, Any],
    request: Geant4StepRequest,
) -> None:
    """Require the native spectrum to echo the complete requested action."""
    if metadata.get("native_action_contract_id") != (
        NATIVE_ACTION_IDENTITY_CONTRACT_ID
    ):
        raise RuntimeError("Native Geant4 action contract is incompatible.")
    if metadata.get("native_action_sha256") != request.action_identity_sha256():
        raise RuntimeError("Native Geant4 action digest differs from the request.")
    integer_fields = {
        "native_action_step_id": int(request.step_id),
        "native_action_seed": int(request.seed),
        "native_action_fe_orientation_index": (
            request.resolved_orientation_indices()[0]
        ),
        "native_action_pb_orientation_index": (
            request.resolved_orientation_indices()[1]
        ),
    }
    for key, expected in integer_fields.items():
        actual = _required_native_action_integer(metadata, key)
        if actual != expected:
            raise RuntimeError(
                f"Native Geant4 action field {key!r} differs from the request."
            )
    numeric_fields = {
        "native_action_dwell_time_s": float(request.dwell_time_s),
    }
    for prefix, position, quaternion in (
        ("detector", request.detector_pose_xyz, request.detector_quat_wxyz),
        ("fe_shield", request.fe_shield_pose_xyz, request.fe_shield_quat_wxyz),
        ("pb_shield", request.pb_shield_pose_xyz, request.pb_shield_quat_wxyz),
    ):
        numeric_fields.update(
            {
                f"native_action_{prefix}_pose_{axis}": float(value)
                for axis, value in zip("xyz", position, strict=True)
            }
        )
        numeric_fields.update(
            {
                f"native_action_{prefix}_quat_{axis}": float(value)
                for axis, value in zip("wxyz", quaternion, strict=True)
            }
        )
    for key, expected in numeric_fields.items():
        actual = _required_native_action_number(metadata, key)
        if actual != expected:
            raise RuntimeError(
                f"Native Geant4 action field {key!r} differs from the request."
            )


@dataclass(frozen=True)
class Geant4StepRequest:
    """Describe a single Geant4 step request."""

    step_id: int
    dwell_time_s: float
    seed: int
    detector_pose_xyz: tuple[float, float, float]
    detector_quat_wxyz: tuple[float, float, float, float]
    fe_shield_pose_xyz: tuple[float, float, float]
    fe_shield_quat_wxyz: tuple[float, float, float, float]
    pb_shield_pose_xyz: tuple[float, float, float]
    pb_shield_quat_wxyz: tuple[float, float, float, float]
    shield_pose_contract_id: str = SHIELD_POSE_CONTRACT_ID
    shield_pose_contract_sha256: str = SHIELD_POSE_CONTRACT_SHA256
    fe_orientation_index: int | None = None
    pb_orientation_index: int | None = None

    def resolved_orientation_indices(self) -> tuple[int, int]:
        """Return indices proven consistent with both physical quaternions."""
        if self.shield_pose_contract_id != SHIELD_POSE_CONTRACT_ID:
            raise ValueError("Geant4 request shield-pose contract is incompatible.")
        if self.shield_pose_contract_sha256 != SHIELD_POSE_CONTRACT_SHA256:
            raise ValueError("Geant4 request shield-pose hash is incompatible.")
        resolved: list[int] = []
        for kind, quaternion, declared_index in (
            ("fe", self.fe_shield_quat_wxyz, self.fe_orientation_index),
            ("pb", self.pb_shield_quat_wxyz, self.pb_orientation_index),
        ):
            physical_normal = np.asarray(
                shield_normal_from_quaternion_wxyz(quaternion),
                dtype=float,
            )
            inferred_index = octant_index_from_normal(-physical_normal)
            index = inferred_index if declared_index is None else int(declared_index)
            expected = physical_shield_normal_from_orientation_index(index)
            if not np.allclose(
                physical_normal,
                expected,
                rtol=0.0,
                atol=1.0e-8,
            ):
                raise ValueError(
                    f"{kind} shield quaternion does not place local (+X,+Y,+Z) "
                    f"material at orientation index {index}."
                )
            resolved.append(index)
        return int(resolved[0]), int(resolved[1])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable request payload."""
        fe_index, pb_index = self.resolved_orientation_indices()
        return {
            "step_id": int(self.step_id),
            "dwell_time_s": float(self.dwell_time_s),
            "seed": int(self.seed),
            "detector_pose_xyz": list(self.detector_pose_xyz),
            "detector_quat_wxyz": list(self.detector_quat_wxyz),
            "fe_shield_pose_xyz": list(self.fe_shield_pose_xyz),
            "fe_shield_quat_wxyz": list(self.fe_shield_quat_wxyz),
            "pb_shield_pose_xyz": list(self.pb_shield_pose_xyz),
            "pb_shield_quat_wxyz": list(self.pb_shield_quat_wxyz),
            "shield_pose_contract_id": self.shield_pose_contract_id,
            "shield_pose_contract_sha256": self.shield_pose_contract_sha256,
            "fe_orientation_index": fe_index,
            "pb_orientation_index": pb_index,
        }

    def action_identity_sha256(self) -> str:
        """Hash the complete action payload before native transport starts."""
        payload = {
            "schema_version": 1,
            "contract_id": NATIVE_ACTION_IDENTITY_CONTRACT_ID,
            **self.to_dict(),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Geant4EngineConfig:
    """Collect external Geant4 engine settings."""

    physics_profile: str = "balanced"
    thread_count: int = 1
    random_seed_base: int = 123
    dead_time_tau_s: float = 5.813e-9
    scatter_gain: float = 0.0
    executable_path: str | None = None
    executable_args: tuple[str, ...] = ()
    timeout_s: float = 120.0
    persistent_process: bool = False
    source_rate_model: str = "detector_cps_1m"
    primary_emission_model: str = "independent_gamma_lines"
    source_bias_mode: str = "detector_cone"
    source_bias_cone_policy: str = "detector_covering"
    source_bias_isotropic_fraction: float = 1.0
    detector_scoring_mode: str = "full_transport"
    secondary_transport_mode: str = "full_transport"
    primary_sampling_fraction: float = 1.0
    target_sampled_primaries: int | None = None
    mean_calibration_histories_per_source_line: int | None = None
    mean_calibration_angle_strata_mu: int = 1
    mean_calibration_angle_strata_phi: int = 1
    mean_calibration_forced_collision: bool = False
    background_cps: float = 0.0
    sample_detector_response: bool = False
    detector_green_operator_path: str | None = None
    detector_green_operator_binary_sha256: str | None = None
    detector_green_operator_contract_sha256: str | None = None
    validation_entry_class_spectra: bool = False
    expected_native_executable_sha256: str | None = None
    expected_native_execution_environment_sha256: str | None = None
    expected_implementation_bundle_sha256: str | None = None
    radiation_visualization: RadiationVisualizationConfig = field(
        default_factory=RadiationVisualizationConfig
    )


class Geant4Engine(ABC):
    """Define the Geant4 engine interface used by the sidecar app."""

    @abstractmethod
    def load_scene(self, scene: ExportedGeant4Scene) -> bool:
        """Load a scene and return whether a cached world was reused."""

    @abstractmethod
    def simulate(self, request: Geant4StepRequest) -> tuple[np.ndarray, dict[str, Any]]:
        """Run one transport step and return a spectrum plus metadata."""

    @abstractmethod
    def close(self) -> None:
        """Release engine-owned resources."""


def _force_stop_persistent_process(
    process: subprocess.Popen[str],
) -> list[BaseException]:
    """Best-effort terminate then kill one ungraceful native child."""
    failures: list[BaseException] = []
    try:
        running = process.poll() is None
    except BaseException as failure:
        failures.append(failure)
        running = True
    if running:
        try:
            process.terminate()
        except BaseException as failure:
            failures.append(failure)
    needs_kill = False
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        needs_kill = True
    except BaseException as failure:
        failures.append(failure)
        needs_kill = True
    if needs_kill:
        try:
            still_running = process.poll() is None
        except BaseException as failure:
            failures.append(failure)
            still_running = True
        if still_running:
            try:
                process.kill()
            except BaseException as failure:
                failures.append(failure)
        try:
            process.wait(timeout=5.0)
        except BaseException as failure:
            failures.append(failure)
    return failures


class ExternalCommandGeant4Engine(Geant4Engine):
    """Delegate transport to an external executable or persistent native process."""

    def __init__(
        self,
        config: Geant4EngineConfig,
        *,
        nuclide_library: Mapping[str, Nuclide] | None = None,
        detector_green_operator: DetectorGreenOperator | None = None,
    ) -> None:
        """Store launch configuration and its explicit nuclide-line mapping."""
        if config.executable_path in (None, ""):
            raise ValueError(
                "executable_path is required for the external Geant4 engine."
            )
        green_values = (
            config.detector_green_operator_path,
            config.detector_green_operator_binary_sha256,
            config.detector_green_operator_contract_sha256,
        )
        if config.sample_detector_response:
            if any(
                not isinstance(value, str) or not value for value in green_values
            ) or any(
                len(str(value)) != 64
                or any(character not in "0123456789abcdef" for character in str(value))
                for value in green_values[1:]
            ):
                raise ValueError(
                    "Detector-response sampling requires one authenticated "
                    "detector Green binary and contract."
                )
            green_path = Path(str(green_values[0])).resolve()
            if not green_path.is_file():
                raise FileNotFoundError(
                    f"Detector Green binary does not exist: {green_path}."
                )
            actual_binary_sha256 = hashlib.sha256(green_path.read_bytes()).hexdigest()
            if actual_binary_sha256 != green_values[1]:
                raise ValueError("Detector Green binary hash is stale.")
            if not isinstance(detector_green_operator, DetectorGreenOperator):
                raise ValueError(
                    "Detector-response sampling requires its authenticated "
                    "DetectorGreenOperator object."
                )
            detector_green_operator.require_runtime_ready()
            if (
                detector_green_operator.binary_sha256 != green_values[1]
                or detector_green_operator.contract_hash_sha256 != green_values[2]
            ):
                raise ValueError(
                    "Detector Green object differs from the configured binary."
                )
        elif any(value is not None for value in green_values):
            raise ValueError(
                "Detector Green settings are forbidden when response "
                "sampling is disabled."
            )
        elif detector_green_operator is not None:
            raise ValueError(
                "Detector Green object is forbidden when response sampling "
                "is disabled."
            )
        self.config = config
        self.detector_green_operator = detector_green_operator
        self.scene: ExportedGeant4Scene | None = None
        self._last_cache_hit = False
        self.library = (
            default_library() if nuclide_library is None else dict(nuclide_library)
        )
        self.nuclide_catalog_sha256 = nuclide_catalog_sha256(self.library)
        self._persistent_process: subprocess.Popen[str] | None = None
        self._persistent_tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._persistent_scene_path: Path | None = None
        self._persistent_scene_hash: str | None = None

    def load_scene(self, scene: ExportedGeant4Scene) -> bool:
        """Store the latest scene for the next external simulation call."""
        cache_hit = self.scene is not None and self.scene.scene_hash == scene.scene_hash
        self.scene = scene
        self._last_cache_hit = cache_hit
        if not cache_hit:
            self._persistent_scene_hash = None
            self._persistent_scene_path = None
        return cache_hit

    def simulate(self, request: Geant4StepRequest) -> tuple[np.ndarray, dict[str, Any]]:
        """Call the configured external executable and parse its response."""
        if self.scene is None:
            raise RuntimeError("Geant4 scene was not loaded before simulate().")
        if self.config.persistent_process:
            spectrum, metadata = self._simulate_persistent(request)
        else:
            spectrum, metadata = self._simulate_one_shot(request)
        validate_native_scene_identity(
            metadata,
            self.scene,
            expected_nuclide_catalog_sha256=(self.nuclide_catalog_sha256),
            expected_detector_green_reference_efficiency_by_isotope=(
                self._detector_green_reference_efficiencies()
            ),
        )
        validate_native_shield_pose_identity(metadata, request)
        validate_native_action_identity(metadata, request)
        metadata["cache_hit"] = bool(self._last_cache_hit)
        metadata["seed"] = int(metadata["native_action_seed"])
        metadata.update(
            build_visualization_metadata_from_scene(
                self.scene,
                request,
                seed=int(request.seed),
                config=self.config.radiation_visualization,
                library=self.library,
                mode="geant4-external-representative",
                scatter_gain=self.config.scatter_gain,
            )
        )
        return spectrum, metadata

    def _detector_green_reference_efficiencies(
        self,
    ) -> dict[str, float] | None:
        """Return exact catalog/operator efficiencies expected from native."""
        operator = self.detector_green_operator
        scene = self.scene
        if operator is None or scene is None:
            return None
        if any(source.activity_bq is not None for source in scene.sources):
            return None
        radius = float(
            scene.detector_model.crystal_radius_m
            + scene.detector_model.housing_thickness_m
        )
        result: dict[str, float] = {}
        for isotope in sorted({source.isotope for source in scene.sources}):
            nuclide = self.library.get(isotope)
            if nuclide is None:
                raise RuntimeError(
                    f"Detector Green normalization lacks catalog isotope {isotope!r}."
                )
            result[isotope] = operator.catalog_weighted_reference_efficiency(
                nuclide,
                detector_target_radius_m=radius,
            )
        return result

    def _simulate_one_shot(
        self, request: Geant4StepRequest
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run one request by launching a fresh native executable process."""
        if self.scene is None:
            raise RuntimeError("Geant4 scene was not loaded before simulate().")
        self._require_approved_native_launch_environment()
        with tempfile.TemporaryDirectory(prefix="geant4_sidecar_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            scene_path = tmp_path / "scene.txt"
            request_path = tmp_path / "request.txt"
            response_path = tmp_path / "response.txt"
            write_scene_file(
                self.scene,
                scene_path,
                nuclide_library=self.library,
            )
            write_request_file(request, request_path)
            command = [
                str(self.config.executable_path),
                "--scene",
                scene_path.as_posix(),
                "--request",
                request_path.as_posix(),
                "--response",
                response_path.as_posix(),
                "--physics-profile",
                self.config.physics_profile,
                "--threads",
                str(self.config.thread_count),
                "--dead-time-tau-s",
                str(self.config.dead_time_tau_s),
                *self._source_bias_args(),
                *self._observation_args(),
                *self.config.executable_args,
            ]
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=self.config.timeout_s,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "External Geant4 executable failed: "
                    f"returncode={result.returncode} stderr={result.stderr.strip()}"
                )
            spectrum, metadata = read_response_file(response_path)
        return spectrum, metadata

    def _simulate_persistent(
        self, request: Geant4StepRequest
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run one request through a persistent native executable process."""
        if self.scene is None:
            raise RuntimeError("Geant4 scene was not loaded before simulate().")
        return self._simulate_persistent_once(request)

    def _simulate_persistent_once(
        self,
        request: Geant4StepRequest,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run one request through the current persistent native process."""
        if self.scene is None:
            raise RuntimeError("Geant4 scene was not loaded before simulate().")
        process = self._ensure_persistent_process()
        tmp_path = self._persistent_tmp_path()
        scene_path = self._ensure_persistent_scene_file(tmp_path)
        request_path = tmp_path / f"request_{int(request.step_id)}.txt"
        response_path = tmp_path / f"response_{int(request.step_id)}.txt"
        write_request_file(request, request_path)
        response_path.unlink(missing_ok=True)
        command = (
            "RUN "
            f"scene={_encode_token(scene_path.as_posix())} "
            f"request={_encode_token(request_path.as_posix())} "
            f"response={_encode_token(response_path.as_posix())}\n"
        )
        if process.stdin is None:
            raise RuntimeError("Persistent Geant4 process does not expose stdin.")
        process.stdin.write(command)
        process.stdin.flush()
        self._wait_for_persistent_ok(process, response_path)
        return read_response_file(response_path)

    def _ensure_persistent_process(self) -> subprocess.Popen[str]:
        """Start the persistent native process if it is not already running."""
        if self._persistent_process is not None:
            returncode = self._persistent_process.poll()
            if returncode is None:
                return self._persistent_process
            raise RuntimeError(
                "Persistent Geant4 executable exited unexpectedly and cannot "
                f"be restarted: returncode={returncode}."
            )
        self._require_approved_native_launch_environment()
        persistent_tmpdir = tempfile.TemporaryDirectory(prefix="geant4_persistent_")
        command = [
            str(self.config.executable_path),
            "--persistent",
            "--physics-profile",
            self.config.physics_profile,
            "--threads",
            str(self.config.thread_count),
            "--dead-time-tau-s",
            str(self.config.dead_time_tau_s),
            *self._source_bias_args(),
            *self._observation_args(),
            *self.config.executable_args,
        ]
        try:
            self._persistent_process = subprocess.Popen(
                command,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
        except BaseException:
            persistent_tmpdir.cleanup()
            raise
        self._persistent_tmpdir = persistent_tmpdir
        return self._persistent_process

    def _require_approved_native_launch_environment(self) -> None:
        """Reauthenticate executable, libraries, and physics data before Popen."""
        self._require_detector_green_binary()
        expected_executable = self.config.expected_native_executable_sha256
        expected_environment = self.config.expected_native_execution_environment_sha256
        expected_implementation = self.config.expected_implementation_bundle_sha256
        expected_digests = (
            expected_executable,
            expected_environment,
            expected_implementation,
        )
        if all(digest is None for digest in expected_digests):
            return
        if any(digest is None for digest in expected_digests):
            raise RuntimeError(
                "Native launch provenance requires executable, execution-"
                "environment, and Python implementation bundle digests."
            )
        executable_path = self.config.executable_path
        if executable_path is None:
            raise RuntimeError("Native launch has no executable path.")
        require_native_execution_bundle(
            executable_path,
            expected_executable_sha256=expected_executable,
            expected_environment_sha256=expected_environment,
        )
        from spectrum.full_spectrum_acceptance_runner import (
            acceptance_implementation_bundle_sha256,
        )

        actual_implementation = acceptance_implementation_bundle_sha256(
            Path(__file__).resolve().parents[3]
        )
        if actual_implementation != expected_implementation:
            raise RuntimeError(
                "Python implementation bundle changed after provenance validation."
            )

    def _require_detector_green_binary(self) -> None:
        """Reauthenticate the immutable detector operator before each launch."""
        if not self.config.sample_detector_response:
            return
        path_value = self.config.detector_green_operator_path
        expected = self.config.detector_green_operator_binary_sha256
        if not isinstance(path_value, str) or not isinstance(expected, str):
            raise RuntimeError("Detector Green launch contract is incomplete.")
        path = Path(path_value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Detector Green binary does not exist: {path}.")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Detector Green binary changed before launch.")

    def _persistent_tmp_path(self) -> Path:
        """Return the persistent process temporary directory path."""
        if self._persistent_tmpdir is None:
            raise RuntimeError(
                "Persistent Geant4 temporary directory is not initialized."
            )
        return Path(self._persistent_tmpdir.name)

    def _ensure_persistent_scene_file(self, tmp_path: Path) -> Path:
        """Write the scene file once per scene hash for the persistent process."""
        if self.scene is None:
            raise RuntimeError("Geant4 scene was not loaded before simulate().")
        if (
            self._persistent_scene_path is not None
            and self._persistent_scene_hash == self.scene.scene_hash
            and self._persistent_scene_path.exists()
        ):
            return self._persistent_scene_path
        scene_path = tmp_path / f"scene_{self.scene.scene_hash[:16]}.txt"
        write_scene_file(
            self.scene,
            scene_path,
            nuclide_library=self.library,
        )
        self._persistent_scene_hash = self.scene.scene_hash
        self._persistent_scene_path = scene_path
        return scene_path

    def _wait_for_persistent_ok(
        self,
        process: subprocess.Popen[str],
        response_path: Path,
    ) -> None:
        """Wait until the persistent process reports request completion."""
        if process.stdout is None:
            raise RuntimeError("Persistent Geant4 process does not expose stdout.")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + float(self.config.timeout_s)
        output_lines: list[str] = []
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    remaining = process.stdout.read() or ""
                    raise RuntimeError(
                        "Persistent Geant4 executable exited unexpectedly: "
                        f"returncode={process.returncode} output={(remaining or '').strip()}"
                    )
                timeout = max(0.0, min(0.25, deadline - time.monotonic()))
                events = selector.select(timeout)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    stripped = line.strip()
                    output_lines.append(stripped)
                    if stripped.startswith("SIMBRIDGE_OK"):
                        if not response_path.exists():
                            raise RuntimeError(
                                "Persistent Geant4 completed without writing response file."
                            )
                        return
                    if stripped.startswith("SIMBRIDGE_ERR"):
                        raise RuntimeError(
                            f"Persistent Geant4 executable failed: {stripped}"
                        )
        finally:
            selector.close()
        tail = "\n".join(output_lines[-20:])
        raise TimeoutError(
            "Timed out waiting for persistent Geant4 response. "
            f"Recent native output:\n{tail}"
        )

    def close(self) -> None:
        """Release the cached scene reference."""
        try:
            self._close_persistent_process()
        finally:
            self.scene = None

    def _close_persistent_process(self) -> None:
        """Require graceful native shutdown and remove every temporary handle."""
        process = self._persistent_process
        self._persistent_process = None
        failures: list[BaseException] = []
        if process is not None:
            try:
                initial_returncode = process.poll()
            except BaseException as poll_failure:
                failures.append(poll_failure)
                initial_returncode = None
            forced_stop = False
            shutdown_write_failed = False
            if initial_returncode is not None:
                failures.append(
                    RuntimeError(
                        "Persistent Geant4 exited before graceful shutdown: "
                        f"returncode={initial_returncode}."
                    )
                )
            else:
                try:
                    if process.stdin is None:
                        raise RuntimeError(
                            "Persistent Geant4 process does not expose stdin "
                            "for graceful shutdown."
                        )
                    process.stdin.write("SHUTDOWN\n")
                    process.stdin.flush()
                except BaseException as failure:
                    shutdown_write_failed = True
                    failures.append(failure)
                if not shutdown_write_failed:
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        forced_stop = True
                    except BaseException as wait_failure:
                        failures.append(wait_failure)
                        forced_stop = True
                if shutdown_write_failed or forced_stop:
                    failures.extend(_force_stop_persistent_process(process))
                if forced_stop:
                    failures.append(
                        RuntimeError(
                            "Persistent Geant4 required forced termination after "
                            "the shutdown deadline."
                        )
                    )
                try:
                    final_returncode = process.poll()
                except BaseException as poll_failure:
                    failures.append(poll_failure)
                    final_returncode = None
                if (
                    not shutdown_write_failed
                    and not forced_stop
                    and final_returncode != 0
                ):
                    failures.append(
                        RuntimeError(
                            "Persistent Geant4 exited with nonzero status after "
                            f"shutdown: returncode={final_returncode}."
                        )
                    )
            for stream in (process.stdin, process.stdout):
                if stream is None:
                    continue
                try:
                    stream.close()
                except BaseException as cleanup_failure:
                    failures.append(cleanup_failure)
        if self._persistent_tmpdir is not None:
            try:
                self._persistent_tmpdir.cleanup()
            except BaseException as cleanup_failure:
                failures.append(cleanup_failure)
        self._persistent_tmpdir = None
        self._persistent_scene_path = None
        self._persistent_scene_hash = None
        if failures:
            primary = failures[0]
            for cleanup_failure in failures[1:]:
                primary.add_note(
                    "Persistent Geant4 cleanup also failed: "
                    f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                )
            raise primary

    def _source_bias_args(self) -> list[str]:
        """Return native executable arguments for the configured source bias mode."""
        arguments = [
            "--source-rate-model",
            str(self.config.source_rate_model),
            "--primary-emission-model",
            str(self.config.primary_emission_model),
            "--source-bias-cone-policy",
            str(self.config.source_bias_cone_policy),
        ]
        if str(self.config.source_rate_model) != "detector_cps_1m":
            arguments.extend(
                [
                    "--source-bias-mode",
                    str(self.config.source_bias_mode),
                    "--source-bias-isotropic-fraction",
                    str(float(self.config.source_bias_isotropic_fraction)),
                ]
            )
        arguments.extend(
            [
                "--detector-scoring-mode",
                str(self.config.detector_scoring_mode),
                "--secondary-transport-mode",
                str(self.config.secondary_transport_mode),
                "--primary-sampling-fraction",
                str(float(self.config.primary_sampling_fraction)),
            ]
        )
        if self.config.target_sampled_primaries is not None:
            arguments.extend(
                [
                    "--target-sampled-primaries",
                    str(int(self.config.target_sampled_primaries)),
                ]
            )
        if self.config.mean_calibration_histories_per_source_line is not None:
            arguments.extend(
                [
                    "--mean-calibration-histories-per-source-line",
                    str(int(self.config.mean_calibration_histories_per_source_line)),
                    "--mean-calibration-angle-strata-mu",
                    str(int(self.config.mean_calibration_angle_strata_mu)),
                    "--mean-calibration-angle-strata-phi",
                    str(int(self.config.mean_calibration_angle_strata_phi)),
                ]
            )
            if self.config.mean_calibration_forced_collision:
                arguments.append("--mean-calibration-forced-collision")
        return arguments

    def _observation_args(self) -> list[str]:
        """Return native arguments for typed background and spectrum marking."""
        arguments = [
            "--background-cps",
            str(float(self.config.background_cps)),
        ]
        if self.config.sample_detector_response:
            arguments.extend(
                [
                    "--sample-detector-response",
                    "--detector-green-operator-path",
                    str(self.config.detector_green_operator_path),
                    "--detector-green-operator-binary-sha256",
                    str(self.config.detector_green_operator_binary_sha256),
                    "--detector-green-operator-contract-sha256",
                    str(self.config.detector_green_operator_contract_sha256),
                ]
            )
        if self.config.validation_entry_class_spectra:
            arguments.append("--validation-entry-class-spectra")
        return arguments


def build_geant4_engine(
    config: Geant4EngineConfig,
    *,
    engine_mode: str,
    nuclide_library: Mapping[str, Nuclide] | None = None,
    detector_green_operator: DetectorGreenOperator | None = None,
) -> Geant4Engine:
    """Instantiate the requested Geant4 engine implementation."""
    normalized = engine_mode.strip().lower()
    if normalized == "external":
        return ExternalCommandGeant4Engine(
            config,
            nuclide_library=nuclide_library,
            detector_green_operator=detector_green_operator,
        )
    raise ValueError(
        f"Unsupported Geant4 engine mode: {engine_mode}. "
        "Only 'external' native Geant4 transport is supported."
    )


def _encode_token(value: str) -> str:
    """Encode a whitespace-free line-protocol token."""
    return str(value).replace(" ", "%20")
