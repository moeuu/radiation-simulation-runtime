"""Simulation runtime abstractions for analytic and sidecar backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
import atexit
from collections.abc import Mapping
import json
import math
from numbers import Real
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

from measurement.model import PointSource
from runtime.provenance import strict_sha256_json
from sim.protocol import (
    SimulationCommand,
    SimulationObservation,
    decode_message,
    encode_message,
    normalize_json_payload,
)
from sim.approx.python_transport import PythonTransportSpectrumModel
from spectrum.detector_green_operator import (
    DETECTOR_GREEN_COINCIDENCE_SEMANTICS,
    DETECTOR_GREEN_SAMPLING_MODE,
)


def _reject_nonfinite_json_constant(value: str) -> None:
    """Reject non-standard non-finite constants in runtime JSON."""
    raise ValueError(
        f"Runtime configuration must use finite standard-JSON numbers; found {value}."
    )


def _runtime_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Build one runtime JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Runtime configuration contains duplicate key {key!r}.")
        result[key] = value
    return result


def _config_bool(
    config: dict[str, Any],
    key: str,
    default: bool,
) -> bool:
    """Return one exact JSON boolean from a runtime configuration."""
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a JSON boolean.")
    return value


def _config_integer(
    config: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Return one exact JSON integer inside inclusive bounds."""
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be a JSON integer.")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{key} must be at least {minimum}.")
    if maximum is not None and resolved > maximum:
        raise ValueError(f"{key} must be at most {maximum}.")
    return resolved


def _config_number(
    config: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float,
    strict_minimum: bool = False,
) -> float:
    """Return one finite JSON number above a configured lower bound."""
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a JSON number.")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{key} must be finite.")
    below = resolved <= minimum if strict_minimum else resolved < minimum
    if below:
        relation = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{key} must be {relation} {minimum}.")
    return resolved


def _config_string(
    config: dict[str, Any],
    key: str,
    default: str,
) -> str:
    """Return one exact nonempty JSON string from runtime configuration."""
    value = config.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty JSON string.")
    return value


class SimulationRuntime(ABC):
    """Define the runtime interface used by the live acquisition loop."""

    @abstractmethod
    def reset(self, payload: dict[str, Any] | None = None) -> None:
        """Reset simulator state for a new episode."""

    @abstractmethod
    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Execute one step and return the resulting observation."""

    @abstractmethod
    def close(self) -> None:
        """Release runtime resources."""


class AnalyticSimulationRuntime(SimulationRuntime):
    """Provide approximate observations using the Python transport debug model."""

    def __init__(
        self,
        *,
        sources: list[PointSource],
        mu_by_isotope: dict[str, object],
        shield_params: Any,
        rng_seed: int = 123,
        obstacle_height_m: float = 2.0,
        obstacle_material: str = "concrete",
        scatter_gain: float = 0.03,
        dead_time_s: float = 0.0,
        background_rate_cps: float = 0.0,
        detector_model: dict[str, Any] | None = None,
    ) -> None:
        """Store simulator inputs for analytic observation generation."""
        self.transport_model = PythonTransportSpectrumModel(
            sources=sources,
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
            obstacle_height_m=float(obstacle_height_m),
            obstacle_material=str(obstacle_material),
            scatter_gain=float(scatter_gain),
            rng_seed=int(rng_seed),
            dead_time_s=float(dead_time_s),
            background_rate_cps=float(background_rate_cps),
            detector_model=detector_model,
        )

    def reset(self, payload: dict[str, Any] | None = None) -> None:
        """Reset sources and static scene geometry from the episode payload."""
        self.transport_model.reset_from_payload(payload)

    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Generate a spectrum at the requested pose and shield orientation."""
        half_yaw = 0.5 * float(command.target_base_yaw_rad)
        return self.transport_model.observe(
            command,
            detector_pose_xyz=command.target_pose_xyz,
            detector_quat_wxyz=(
                math.cos(half_yaw),
                0.0,
                0.0,
                math.sin(half_yaw),
            ),
            backend_label="analytic",
        )

    def close(self) -> None:
        """Closing the analytic runtime is a no-op."""
        return None


class TCPSidecarClientRuntime(SimulationRuntime):
    """Send simulator commands to a remote sidecar over TCP."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float = 10.0,
        *,
        close_on_close: bool = True,
    ) -> None:
        """Store sidecar connection parameters."""
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.close_on_close = bool(close_on_close)

    def _round_trip(self, message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a single request and return the response payload."""
        wire_payload = normalize_json_payload(payload)
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout_s
        ) as conn:
            conn.sendall(encode_message(message_type, wire_payload))
            conn.shutdown(socket.SHUT_WR)
            chunks: list[bytes] = []
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        if not chunks:
            raise RuntimeError("Simulator sidecar returned an empty response.")
        response_type, response_payload = decode_message(b"".join(chunks).strip())
        if response_type == "error":
            raise RuntimeError(
                str(response_payload.get("message", "Unknown sidecar error."))
            )
        if response_type != "ok":
            raise RuntimeError(f"Unexpected sidecar response type: {response_type}")
        return response_payload

    def reset(self, payload: dict[str, Any] | None = None) -> None:
        """Reset the remote sidecar episode state."""
        self._round_trip("reset", payload or {})

    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Execute one remote step and parse the resulting observation."""
        payload = self._round_trip("step", command.to_dict())
        return SimulationObservation.from_dict(payload["observation"])

    def close(self) -> None:
        """Require an acknowledged clean shutdown from the remote sidecar."""
        if not self.close_on_close:
            return None
        response = self._round_trip("shutdown", {})
        if response.get("status") != "shutdown":
            raise RuntimeError(
                "Simulator sidecar did not acknowledge a clean shutdown."
            )


class IsaacSimTCPClientRuntime(TCPSidecarClientRuntime):
    """TCP client for the Isaac Sim visualization sidecar."""

    def visualize_observation(self, observation: SimulationObservation) -> None:
        """Send observation metadata to Isaac Sim for stage visualization."""
        self._round_trip("visualize", {"observation": observation.to_dict()})

    def visualize_estimator_state(self, payload: dict[str, Any]) -> None:
        """Send estimator particle and estimate markers to Isaac Sim for visualization."""
        self._round_trip("visualize_estimator", payload)


class Geant4TCPClientRuntime(TCPSidecarClientRuntime):
    """TCP client for the Geant4 sidecar."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float = 10.0,
        *,
        expected_primary_sampling_fraction: float = 1.0,
        expected_target_sampled_primaries: int | None = None,
        accelerated_weighted_transport_enable: bool = False,
        expected_source_rate_model: str | None = None,
        expected_thread_count: int | None = None,
        expected_physics_profile: str | None = None,
        expected_detector_scoring_mode: str | None = None,
        expected_secondary_transport_mode: str | None = None,
        expected_source_bias_mode: str | None = None,
        expected_background_cps: float | None = None,
        expected_dead_time_tau_s: float | None = None,
        expected_detector_response_sampling: bool = False,
        expected_detector_green_operator_contract_sha256: str | None = None,
        expected_detector_green_operator_binary_sha256: str | None = None,
        expected_runtime_config_sha256: str | None = None,
        expected_native_executable_sha256: str | None = None,
        expected_native_execution_environment_sha256: str | None = None,
        expected_implementation_bundle_sha256: str | None = None,
    ) -> None:
        """Store connection parameters and expected sidecar fidelity."""
        from sim.geant4_app.app import require_primary_sampling_fraction

        super().__init__(host=host, port=port, timeout_s=timeout_s)
        if not isinstance(accelerated_weighted_transport_enable, bool):
            raise ValueError("accelerated_weighted_transport_enable must be a boolean.")
        self.accelerated_weighted_transport_enable = bool(
            accelerated_weighted_transport_enable
        )
        if expected_target_sampled_primaries is None:
            self.expected_target_sampled_primaries = None
        else:
            if isinstance(expected_target_sampled_primaries, bool) or not isinstance(
                expected_target_sampled_primaries,
                int,
            ):
                raise ValueError(
                    "expected_target_sampled_primaries must be a positive integer."
                )
            if expected_target_sampled_primaries <= 0:
                raise ValueError(
                    "expected_target_sampled_primaries must be a positive integer."
                )
            self.expected_target_sampled_primaries = int(
                expected_target_sampled_primaries
            )
        self.expected_primary_sampling_fraction = require_primary_sampling_fraction(
            expected_primary_sampling_fraction,
            accelerated_weighted_transport_enable=(
                self.accelerated_weighted_transport_enable
            ),
            target_sampled_primaries=self.expected_target_sampled_primaries,
        )
        weighted_requested = (
            self.expected_primary_sampling_fraction < 1.0
            or self.expected_target_sampled_primaries is not None
        )
        if weighted_requested and not self.accelerated_weighted_transport_enable:
            raise ValueError(
                "Weighted Geant4 sampling requires "
                "accelerated_weighted_transport_enable=true."
            )
        if self.accelerated_weighted_transport_enable and not weighted_requested:
            raise ValueError(
                "accelerated_weighted_transport_enable=true requires a reduced "
                "primary_sampling_fraction or target_sampled_primaries."
            )
        self.expected_source_rate_model = (
            None
            if expected_source_rate_model is None
            else str(expected_source_rate_model)
        )
        if expected_thread_count is None:
            self.expected_thread_count = None
        else:
            if isinstance(expected_thread_count, bool) or not isinstance(
                expected_thread_count,
                int,
            ):
                raise ValueError(
                    "expected_thread_count must be a positive JSON integer."
                )
            if expected_thread_count <= 0:
                raise ValueError(
                    "expected_thread_count must be a positive JSON integer."
                )
            self.expected_thread_count = expected_thread_count
        self.expected_physics_profile = (
            None if expected_physics_profile is None else str(expected_physics_profile)
        )
        self.expected_detector_scoring_mode = (
            None
            if expected_detector_scoring_mode is None
            else str(expected_detector_scoring_mode)
        )
        self.expected_secondary_transport_mode = (
            None
            if expected_secondary_transport_mode is None
            else str(expected_secondary_transport_mode)
        )
        self.expected_source_bias_mode = (
            None
            if expected_source_bias_mode is None
            else str(expected_source_bias_mode)
        )
        self.expected_background_cps = (
            None if expected_background_cps is None else float(expected_background_cps)
        )
        self.expected_dead_time_tau_s = (
            None
            if expected_dead_time_tau_s is None
            else float(expected_dead_time_tau_s)
        )
        if self.expected_dead_time_tau_s is not None and (
            not math.isfinite(self.expected_dead_time_tau_s)
            or self.expected_dead_time_tau_s < 0.0
        ):
            raise ValueError("expected_dead_time_tau_s must be finite and nonnegative.")
        if not isinstance(expected_detector_response_sampling, bool):
            raise ValueError("expected_detector_response_sampling must be a boolean.")
        self.expected_detector_response_sampling = bool(
            expected_detector_response_sampling
        )
        green_hashes = (
            expected_detector_green_operator_contract_sha256,
            expected_detector_green_operator_binary_sha256,
        )
        valid_green_hashes = all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in green_hashes
        )
        if self.expected_detector_response_sampling != valid_green_hashes or (
            not self.expected_detector_response_sampling
            and any(value is not None for value in green_hashes)
        ):
            raise ValueError(
                "Detector Green hashes are required exactly when detector "
                "response sampling is enabled."
            )
        self.expected_detector_green_operator_contract_sha256 = (
            expected_detector_green_operator_contract_sha256
        )
        self.expected_detector_green_operator_binary_sha256 = (
            expected_detector_green_operator_binary_sha256
        )
        if expected_runtime_config_sha256 is not None and (
            not isinstance(expected_runtime_config_sha256, str)
            or len(expected_runtime_config_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_runtime_config_sha256
            )
        ):
            raise ValueError(
                "expected_runtime_config_sha256 must be a lowercase SHA-256 string."
            )
        self.expected_runtime_config_sha256 = expected_runtime_config_sha256
        for field_name, value in (
            (
                "expected_native_executable_sha256",
                expected_native_executable_sha256,
            ),
            (
                "expected_native_execution_environment_sha256",
                expected_native_execution_environment_sha256,
            ),
            (
                "expected_implementation_bundle_sha256",
                expected_implementation_bundle_sha256,
            ),
        ):
            if value is not None and (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 string.")
        self.expected_native_executable_sha256 = expected_native_executable_sha256
        self.expected_native_execution_environment_sha256 = (
            expected_native_execution_environment_sha256
        )
        self.expected_implementation_bundle_sha256 = (
            expected_implementation_bundle_sha256
        )
        self.expected_surface_source_contract_sha256: str | None = None
        self.expected_scene_hash: str | None = None

    def reset(self, payload: dict[str, Any] | None = None) -> None:
        """Reset the sidecar only after validating its fidelity handshake."""
        from measurement.source_boundary import (
            surface_source_runtime_contract_sha256,
        )

        reset_payload = payload or {}
        sources = reset_payload.get("sources")
        if not isinstance(sources, list) or not sources:
            raise RuntimeError(
                "Geant4 reset requires a nonempty canonical surface-source payload."
            )
        self.expected_surface_source_contract_sha256 = (
            surface_source_runtime_contract_sha256(sources)
        )
        response = self._round_trip("reset", reset_payload)
        self._validate_fidelity_handshake(response)

    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Execute one step and validate native fidelity before estimator ingestion."""
        from sim.geant4_app.app import validate_transport_metadata

        payload = self._round_trip("step", command.to_dict())
        observation = SimulationObservation.from_dict(payload["observation"])
        validate_transport_metadata(
            observation.metadata,
            expected_primary_sampling_fraction=(
                self.expected_primary_sampling_fraction
            ),
            expected_target_sampled_primaries=(self.expected_target_sampled_primaries),
            accelerated_weighted_transport_enable=(
                self.accelerated_weighted_transport_enable
            ),
            expected_source_rate_model=self.expected_source_rate_model,
            expected_thread_count=self.expected_thread_count,
            expected_physics_profile=self.expected_physics_profile,
            expected_detector_scoring_mode=self.expected_detector_scoring_mode,
            expected_secondary_transport_mode=(self.expected_secondary_transport_mode),
            expected_source_bias_mode=self.expected_source_bias_mode,
            expected_background_cps=self.expected_background_cps,
            expected_dead_time_tau_s=self.expected_dead_time_tau_s,
            expected_detector_response_sampling=(
                self.expected_detector_response_sampling
            ),
            expected_detector_green_operator_contract_sha256=(
                self.expected_detector_green_operator_contract_sha256
            ),
            expected_detector_green_operator_binary_sha256=(
                self.expected_detector_green_operator_binary_sha256
            ),
            expected_surface_source_contract_sha256=(
                self.expected_surface_source_contract_sha256
            ),
            expected_scene_hash=self.expected_scene_hash,
        )
        return observation

    @staticmethod
    def _required_handshake_bool(
        fidelity: dict[str, Any],
        key: str,
    ) -> bool:
        """Return a strict boolean from a reset handshake or fail closed."""
        value = fidelity.get(key)
        if not isinstance(value, bool):
            raise RuntimeError(
                f"Geant4 sidecar fidelity handshake is missing valid {key}."
            )
        return value

    @staticmethod
    def _required_handshake_integer(
        fidelity: dict[str, Any],
        key: str,
    ) -> int:
        """Return an exact JSON integer from a reset handshake or fail closed."""
        value = fidelity.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                f"Geant4 sidecar fidelity handshake is missing valid {key}."
            )
        return value

    @staticmethod
    def _required_handshake_number(
        fidelity: dict[str, Any],
        key: str,
    ) -> float:
        """Return a finite JSON number from a reset handshake or fail closed."""
        value = fidelity.get(key)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RuntimeError(
                f"Geant4 sidecar fidelity handshake is missing valid {key}."
            )
        resolved = float(value)
        if not math.isfinite(resolved):
            raise RuntimeError(
                f"Geant4 sidecar fidelity handshake is missing valid {key}."
            )
        return resolved

    def _validate_fidelity_handshake(self, response: dict[str, Any]) -> None:
        """Reject stale or mismatched Geant4 bridge processes."""
        from measurement.source_boundary import (
            SURFACE_EMISSION_EPSILON_M,
            surface_emission_policy_sha256,
        )

        fidelity = response.get("runtime_fidelity")
        if not isinstance(fidelity, dict):
            raise RuntimeError(
                "Geant4 sidecar reset did not return a runtime_fidelity "
                "handshake; refusing to reuse an unverified process."
            )
        if self.expected_runtime_config_sha256 is not None:
            actual_runtime_config_sha256 = fidelity.get(
                "production_runtime_config_sha256"
            )
            if actual_runtime_config_sha256 != self.expected_runtime_config_sha256:
                raise RuntimeError(
                    "Geant4 sidecar production runtime-config hash mismatch: "
                    f"expected {self.expected_runtime_config_sha256}, got "
                    f"{actual_runtime_config_sha256 or 'missing'}."
                )
        for field_name, expected_value in (
            (
                "native_executable_sha256",
                self.expected_native_executable_sha256,
            ),
            (
                "native_execution_environment_sha256",
                self.expected_native_execution_environment_sha256,
            ),
            (
                "implementation_bundle_sha256",
                self.expected_implementation_bundle_sha256,
            ),
        ):
            if (
                expected_value is not None
                and fidelity.get(field_name) != expected_value
            ):
                raise RuntimeError(
                    "Geant4 sidecar native execution provenance mismatch for "
                    f"{field_name}: expected {expected_value}, got "
                    f"{fidelity.get(field_name) or 'missing'}."
                )
        expected_float_fields = {
            "primary_sampling_fraction": self.expected_primary_sampling_fraction,
            "requested_primary_sampling_fraction": (
                self.expected_primary_sampling_fraction
            ),
        }
        if self.expected_target_sampled_primaries is None:
            expected_float_fields["primary_history_weight"] = (
                1.0 / self.expected_primary_sampling_fraction
            )
        if self.expected_dead_time_tau_s is not None:
            expected_float_fields["dead_time_tau_s"] = self.expected_dead_time_tau_s
        for key, expected_value in expected_float_fields.items():
            value = self._required_handshake_number(fidelity, key)
            if not math.isclose(
                value,
                expected_value,
                rel_tol=1.0e-12,
                abs_tol=1.0e-18,
            ):
                raise RuntimeError(
                    f"Geant4 sidecar requires {key}={expected_value}, got {value}."
                )
        if (
            fidelity.get("source_position_semantics") != "air_side_native_emission_xyz"
            or fidelity.get("source_anchor_semantics")
            != "exact_surface_chart_uv_evaluation_truth"
            or not self._required_handshake_bool(
                fidelity,
                "all_sources_surface_bound",
            )
            or fidelity.get("surface_emission_policy_sha256")
            != surface_emission_policy_sha256()
        ):
            raise RuntimeError(
                "Geant4 sidecar source-boundary fidelity handshake is invalid."
            )
        if fidelity.get("intensity_cps_1m_definition") != (
            "pre_dead_time_detector_pulse_rate_at_1m"
        ):
            raise RuntimeError(
                "Geant4 sidecar source-strength fidelity handshake is invalid."
            )
        source_epsilon_m = self._required_handshake_number(
            fidelity,
            "surface_emission_epsilon_m",
        )
        if not math.isclose(
            source_epsilon_m,
            SURFACE_EMISSION_EPSILON_M,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise RuntimeError(
                "Geant4 sidecar source epsilon differs from the shared contract."
            )
        source_contract_hash = fidelity.get("surface_source_contract_sha256")
        scene_hash = fidelity.get("scene_hash")
        for key, value in (
            ("surface_source_contract_sha256", source_contract_hash),
            ("scene_hash", scene_hash),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise RuntimeError(
                    f"Geant4 sidecar fidelity handshake has invalid {key}."
                )
        if (
            self.expected_surface_source_contract_sha256 is not None
            and source_contract_hash != self.expected_surface_source_contract_sha256
        ):
            raise RuntimeError(
                "Geant4 sidecar source strength/position contract hash differs "
                "from the reset payload."
            )
        self.expected_scene_hash = str(scene_hash)
        actual_target_sampled_primaries = self._required_handshake_integer(
            fidelity,
            "target_sampled_primaries",
        )
        expected_target_sampled_primaries = (
            0
            if self.expected_target_sampled_primaries is None
            else self.expected_target_sampled_primaries
        )
        if actual_target_sampled_primaries != expected_target_sampled_primaries:
            raise RuntimeError(
                "Geant4 sidecar target-sampled-primary mismatch: expected "
                f"{expected_target_sampled_primaries}, got "
                f"{actual_target_sampled_primaries}."
            )
        actual_budget_enabled = self._required_handshake_bool(
            fidelity,
            "primary_sampling_budget_enabled",
        )
        expected_budget_enabled = self.expected_target_sampled_primaries is not None
        if actual_budget_enabled != expected_budget_enabled:
            raise RuntimeError(
                "Geant4 sidecar primary-sampling budget mismatch: expected "
                f"{expected_budget_enabled}, got {actual_budget_enabled}."
            )
        expected_fraction_resolution = (
            "per_observation_pending" if expected_budget_enabled else "fixed_fraction"
        )
        actual_fraction_resolution = str(
            fidelity.get("primary_sampling_fraction_resolution", "")
        )
        if actual_fraction_resolution != expected_fraction_resolution:
            raise RuntimeError(
                "Geant4 sidecar primary-sampling fraction resolution mismatch: "
                f"expected {expected_fraction_resolution}, got "
                f"{actual_fraction_resolution or 'missing'}."
            )
        if expected_budget_enabled:
            actual_history_resolution = str(
                fidelity.get("history_thinning_resolution", "")
            )
            if actual_history_resolution != "per_observation_pending":
                raise RuntimeError(
                    "Geant4 sidecar history-thinning resolution mismatch: "
                    "expected per_observation_pending."
                )
            premature_history_fields = sorted(
                key
                for key in (
                    "history_thinning_enabled",
                    "transport_history_mode",
                )
                if key in fidelity
            )
            if premature_history_fields:
                raise RuntimeError(
                    "Budgeted Geant4 reset handshake must not report unresolved "
                    "per-observation history state: "
                    f"{premature_history_fields}."
                )
        actual_accelerated = self._required_handshake_bool(
            fidelity,
            "accelerated_weighted_transport_enable",
        )
        if actual_accelerated != self.accelerated_weighted_transport_enable:
            raise RuntimeError(
                "Geant4 sidecar accelerated weighted-transport mismatch: "
                f"expected {self.accelerated_weighted_transport_enable}, "
                f"got {actual_accelerated}."
            )
        actual_response_sampling = self._required_handshake_bool(
            fidelity,
            "sample_detector_response",
        )
        if actual_response_sampling != self.expected_detector_response_sampling:
            raise RuntimeError(
                "Geant4 sidecar detector-response sampling mismatch: expected "
                f"{self.expected_detector_response_sampling}, got "
                f"{actual_response_sampling}."
            )
        if self.expected_detector_response_sampling:
            from spectrum.response_matrix import (
                NATIVE_GEANT4_BACKGROUND_MODEL_ID,
                NATIVE_GEANT4_BIN_COUNT,
                NATIVE_GEANT4_BIN_WIDTH_KEV,
                NATIVE_GEANT4_ENERGY_MAX_KEV,
                NATIVE_GEANT4_ENERGY_MIN_KEV,
            )

            expected_response_text = {
                "detector_response_sampling_model": (
                    "isotope_independent_full_detector_green_operator_v3"
                ),
                "detector_response_sampling_mode": DETECTOR_GREEN_SAMPLING_MODE,
                "detector_response_sampling_contract_sha256": (
                    self.expected_detector_green_operator_contract_sha256
                ),
                "detector_response_operator_binary_sha256": (
                    self.expected_detector_green_operator_binary_sha256
                ),
                "detector_response_boundary_state": (
                    "normalized_impact_parameter_at_detector_housing_entry_v1"
                ),
                "detector_response_conditioning": (
                    "registered_pulse_subprobability_given_housing_incident_gamma_v1"
                ),
                "detector_response_coincidence_semantics": (
                    DETECTOR_GREEN_COINCIDENCE_SEMANTICS
                ),
                "detector_cps_green_reference_normalization": (
                    "catalog_branching_weighted_absolute_detection_efficiency_at_1m_v1"
                ),
                "background_spectrum_model_id": (NATIVE_GEANT4_BACKGROUND_MODEL_ID),
            }
            for key, expected_value in expected_response_text.items():
                actual_value = str(fidelity.get(key, ""))
                if actual_value != expected_value:
                    raise RuntimeError(
                        "Geant4 sidecar full-spectrum handshake mismatch for "
                        f"{key}: expected {expected_value}, got "
                        f"{actual_value or 'missing'}."
                    )
            expected_response_float = {
                "spectrum_energy_min_keV": NATIVE_GEANT4_ENERGY_MIN_KEV,
                "spectrum_energy_max_keV": NATIVE_GEANT4_ENERGY_MAX_KEV,
                "spectrum_bin_width_keV": NATIVE_GEANT4_BIN_WIDTH_KEV,
            }
            for key, expected_value in expected_response_float.items():
                actual_value = self._required_handshake_number(
                    fidelity,
                    key,
                )
                if not math.isclose(
                    actual_value,
                    float(expected_value),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    raise RuntimeError(
                        "Geant4 sidecar full-spectrum handshake mismatch for "
                        f"{key}: expected {expected_value}, got {actual_value}."
                    )
            actual_bin_count = self._required_handshake_integer(
                fidelity,
                "spectrum_bin_count",
            )
            if actual_bin_count != int(NATIVE_GEANT4_BIN_COUNT):
                raise RuntimeError(
                    "Geant4 sidecar full-spectrum handshake has the wrong "
                    f"spectrum_bin_count: expected {NATIVE_GEANT4_BIN_COUNT}, "
                    f"got {actual_bin_count}."
                )
        if self.expected_source_rate_model is not None:
            actual_model = str(fidelity.get("source_rate_model", ""))
            if actual_model != self.expected_source_rate_model:
                raise RuntimeError(
                    "Geant4 sidecar source-rate model mismatch: "
                    f"expected {self.expected_source_rate_model}, "
                    f"got {actual_model or 'missing'}."
                )
        if self.expected_thread_count is not None:
            actual_threads = self._required_handshake_integer(
                fidelity,
                "requested_threads",
            )
            if actual_threads != self.expected_thread_count:
                raise RuntimeError(
                    "Geant4 sidecar thread-count mismatch: "
                    f"expected {self.expected_thread_count}, got {actual_threads}."
                )
        expected_text_fields = {
            "physics_profile": self.expected_physics_profile,
            "detector_scoring_mode": self.expected_detector_scoring_mode,
            "secondary_transport_mode": self.expected_secondary_transport_mode,
            "source_bias_mode": self.expected_source_bias_mode,
        }
        for key, expected_value in expected_text_fields.items():
            if expected_value is None:
                continue
            actual_value = str(fidelity.get(key, ""))
            if actual_value != expected_value:
                raise RuntimeError(
                    f"Geant4 sidecar {key} mismatch: expected {expected_value}, "
                    f"got {actual_value or 'missing'}."
                )
        if self.expected_background_cps is not None:
            actual_background_cps = self._required_handshake_number(
                fidelity,
                "background_cps",
            )
            if actual_background_cps != self.expected_background_cps:
                raise RuntimeError(
                    "Geant4 sidecar background rate mismatch: "
                    f"expected {self.expected_background_cps}, "
                    f"got {actual_background_cps}."
                )


class Geant4WithIsaacSimRuntime(SimulationRuntime):
    """Route robot motion to Isaac Sim while using Geant4 observations."""

    def __init__(
        self,
        *,
        geant4_runtime: SimulationRuntime,
        isaacsim_runtime: SimulationRuntime,
    ) -> None:
        """Store the paired runtimes."""
        self.geant4_runtime = geant4_runtime
        self.isaacsim_runtime = isaacsim_runtime

    def reset(self, payload: dict[str, Any] | None = None) -> None:
        """Reset both simulators with the same scene payload."""
        scene_payload = payload or {}
        self.isaacsim_runtime.reset(scene_payload)
        self.geant4_runtime.reset(scene_payload)

    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Move the Isaac Sim robot and return the Geant4 observation."""
        self.isaacsim_runtime.step(command)
        observation = self.geant4_runtime.step(command)
        visualizer = getattr(self.isaacsim_runtime, "visualize_observation", None)
        if visualizer is not None:
            visualizer(observation)
        return observation

    def visualize_estimator_state(self, payload: dict[str, Any]) -> None:
        """Forward estimator particle visualization to the Isaac Sim companion runtime."""
        visualizer = getattr(self.isaacsim_runtime, "visualize_estimator_state", None)
        if visualizer is not None:
            visualizer(payload)

    def close(self) -> None:
        """Close both runtimes, preserving the first close error if any."""
        first_error: BaseException | None = None
        for runtime in (self.geant4_runtime, self.isaacsim_runtime):
            try:
                runtime.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    first_error.add_note(
                        "Companion runtime cleanup also failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
        if first_error is not None:
            raise first_error


def _force_stop_owned_process(
    process: subprocess.Popen[str],
) -> list[BaseException]:
    """Best-effort terminate then kill, returning every cleanup failure."""
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


def _finish_managed_sidecar_close(
    *,
    process: subprocess.Popen[str],
    log_handle: object | None,
    temp_config_path: Path | None,
    shutdown_failure: BaseException | None,
) -> None:
    """Verify graceful child exit and release every owned sidecar handle."""
    failures: list[BaseException] = []
    if shutdown_failure is not None:
        failures.append(shutdown_failure)
    forced_stop = False
    try:
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            forced_stop = True
        except BaseException as wait_failure:
            failures.append(wait_failure)
            forced_stop = True
        if forced_stop:
            failures.append(
                RuntimeError("Sidecar required forced termination after shutdown.")
            )
            failures.extend(_force_stop_owned_process(process))
        returncode = process.poll()
        if not forced_stop and returncode != 0:
            failures.append(
                RuntimeError(
                    "Sidecar exited with nonzero status after shutdown: "
                    f"returncode={returncode}."
                )
            )
    except BaseException as cleanup_failure:
        failures.append(cleanup_failure)
    if log_handle is not None:
        try:
            close = getattr(log_handle, "close", None)
            if close is not None:
                close()
        except BaseException as cleanup_failure:
            failures.append(cleanup_failure)
    if temp_config_path is not None:
        try:
            temp_config_path.unlink(missing_ok=True)
        except BaseException as cleanup_failure:
            failures.append(cleanup_failure)
    if failures:
        primary = failures[0]
        for cleanup_failure in failures[1:]:
            primary.add_note(
                "Sidecar close also failed: "
                f"{type(cleanup_failure).__name__}: {cleanup_failure}"
            )
        raise primary


class ManagedIsaacSimTCPClientRuntime(IsaacSimTCPClientRuntime):
    """Isaac Sim TCP client that owns an auto-started sidecar process."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float,
        *,
        process: subprocess.Popen[str],
        log_handle: object | None = None,
        temp_config_path: Path | None = None,
        close_on_close: bool = True,
    ) -> None:
        """Store the client parameters and owned process handles."""
        super().__init__(
            host=host, port=port, timeout_s=timeout_s, close_on_close=close_on_close
        )
        self.process = process
        self.log_handle = log_handle
        self.temp_config_path = temp_config_path
        self._closed = False
        self._atexit_callback = self._close_at_interpreter_exit
        atexit.register(self._atexit_callback)

    def close(self) -> None:
        """Require clean shutdown and release the owned sidecar process."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self._atexit_callback)
        shutdown_failure: BaseException | None = None
        try:
            super().close()
        except BaseException as failure:
            shutdown_failure = failure
        if not self.close_on_close:
            if self.log_handle is not None:
                close = getattr(self.log_handle, "close", None)
                if close is not None:
                    close()
            if self.temp_config_path is not None:
                self.temp_config_path.unlink(missing_ok=True)
            if shutdown_failure is not None:
                raise shutdown_failure
            return
        _finish_managed_sidecar_close(
            process=self.process,
            log_handle=self.log_handle,
            temp_config_path=self.temp_config_path,
            shutdown_failure=shutdown_failure,
        )

    def _close_at_interpreter_exit(self) -> None:
        """Best-effort cleanup if ownership outlives the acquisition loop."""
        try:
            self.close()
        except BaseException:
            return


class ManagedGeant4TCPClientRuntime(Geant4TCPClientRuntime):
    """Geant4 TCP client that owns an auto-started sidecar process."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float,
        *,
        process: subprocess.Popen[str],
        log_handle: object | None = None,
        temp_config_path: Path | None = None,
        expected_primary_sampling_fraction: float = 1.0,
        expected_target_sampled_primaries: int | None = None,
        accelerated_weighted_transport_enable: bool = False,
        expected_source_rate_model: str | None = None,
        expected_thread_count: int | None = None,
        expected_physics_profile: str | None = None,
        expected_detector_scoring_mode: str | None = None,
        expected_secondary_transport_mode: str | None = None,
        expected_source_bias_mode: str | None = None,
        expected_background_cps: float | None = None,
        expected_dead_time_tau_s: float | None = None,
        expected_detector_response_sampling: bool = False,
        expected_detector_green_operator_contract_sha256: str | None = None,
        expected_detector_green_operator_binary_sha256: str | None = None,
        expected_runtime_config_sha256: str | None = None,
        expected_native_executable_sha256: str | None = None,
        expected_native_execution_environment_sha256: str | None = None,
        expected_implementation_bundle_sha256: str | None = None,
    ) -> None:
        """Store the client parameters and owned process handles."""
        super().__init__(
            host=host,
            port=port,
            timeout_s=timeout_s,
            expected_primary_sampling_fraction=(expected_primary_sampling_fraction),
            expected_target_sampled_primaries=(expected_target_sampled_primaries),
            accelerated_weighted_transport_enable=(
                accelerated_weighted_transport_enable
            ),
            expected_source_rate_model=expected_source_rate_model,
            expected_thread_count=expected_thread_count,
            expected_physics_profile=expected_physics_profile,
            expected_detector_scoring_mode=expected_detector_scoring_mode,
            expected_secondary_transport_mode=expected_secondary_transport_mode,
            expected_source_bias_mode=expected_source_bias_mode,
            expected_background_cps=expected_background_cps,
            expected_dead_time_tau_s=expected_dead_time_tau_s,
            expected_detector_response_sampling=(expected_detector_response_sampling),
            expected_detector_green_operator_contract_sha256=(
                expected_detector_green_operator_contract_sha256
            ),
            expected_detector_green_operator_binary_sha256=(
                expected_detector_green_operator_binary_sha256
            ),
            expected_runtime_config_sha256=expected_runtime_config_sha256,
            expected_native_executable_sha256=(expected_native_executable_sha256),
            expected_native_execution_environment_sha256=(
                expected_native_execution_environment_sha256
            ),
            expected_implementation_bundle_sha256=(
                expected_implementation_bundle_sha256
            ),
        )
        self.process = process
        self.log_handle = log_handle
        self.temp_config_path = temp_config_path
        self._closed = False
        self._atexit_callback = self._close_at_interpreter_exit
        atexit.register(self._atexit_callback)

    def close(self) -> None:
        """Require clean shutdown and release the owned sidecar process."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self._atexit_callback)
        shutdown_failure: BaseException | None = None
        try:
            super().close()
        except BaseException as failure:
            shutdown_failure = failure
        _finish_managed_sidecar_close(
            process=self.process,
            log_handle=self.log_handle,
            temp_config_path=self.temp_config_path,
            shutdown_failure=shutdown_failure,
        )

    def _close_at_interpreter_exit(self) -> None:
        """Best-effort cleanup for failures before the live-loop ``finally``."""
        try:
            self.close()
        except Exception:
            return


def load_runtime_config(path: str | Path | None) -> dict[str, Any]:
    """Load a JSON runtime configuration file with optional inheritance."""
    return _load_runtime_config(path, seen=set())


PRODUCTION_RUNTIME_CONFIG_FIELDS = frozenset(
    {
        "absorbing_transport_groups",
        "author_obstacle_prims",
        "author_room_boundary_prims",
        "auto_start_sidecar",
        "backend",
        "background_cps",
        "background_spectrum_model_id",
        "bin_width_keV",
        "dead_time_tau_s",
        "detector_aperture_samples",
        "detector_aperture_sampling",
        "detector_height_m",
        "detector_green_operator_manifest",
        "detector_model",
        "detector_scoring_mode",
        "energy_bin_count",
        "energy_max_keV",
        "energy_min_keV",
        "engine_mode",
        "executable_path",
        "full_spectrum_model_registry_file_sha256",
        "full_spectrum_model_registry_path",
        "headless",
        "host",
        "isotope_experiment_profile",
        "line_resolved_shield_attenuation",
        "obstacle_attenuation_enabled",
        "obstacle_height_m",
        "persistent_process",
        "physics_profile",
        "port",
        "primary_emission_model",
        "primary_sampling_fraction",
        "random_seed_base",
        "renderer",
        "sample_detector_response",
        "secondary_transport_mode",
        "shield_transmission_target",
        "simulation_runtime_schema_version",
        "source_bias_cone_policy",
        "source_bias_isotropic_fraction",
        "source_bias_mode",
        "source_extent_radius_m",
        "source_extent_samples",
        "source_rate_model",
        "stage_material_rules",
        "start_isaacsim_sidecar_with_geant4",
        "thread_count",
        "timeout_s",
        "usd_path",
        "use_mock_stage",
    }
)
PRODUCTION_GUI_RUNTIME_CONFIG_FIELDS = frozenset(
    {
        "isaacsim_keep_sidecar_alive",
        "isaacsim_sidecar_config_path",
        "isaacsim_sidecar_python_env",
        "isaacsim_sidecar_startup_timeout_s",
    }
)
_PRODUCTION_DETECTOR_MODEL_FIELDS = frozenset(
    {
        "crystal_length_m",
        "crystal_material",
        "crystal_radius_m",
        "crystal_shape",
        "coincidence_window_s",
        "housing_material",
        "housing_thickness_m",
    }
)
_PRODUCTION_BOOLEAN_FIELDS = frozenset(
    {
        "author_obstacle_prims",
        "author_room_boundary_prims",
        "auto_start_sidecar",
        "headless",
        "line_resolved_shield_attenuation",
        "obstacle_attenuation_enabled",
        "persistent_process",
        "sample_detector_response",
        "start_isaacsim_sidecar_with_geant4",
        "use_mock_stage",
    }
)


def validate_production_runtime_config(config: Mapping[str, Any]) -> None:
    """Require the complete canonical production transport configuration."""
    if not isinstance(config, Mapping) or any(
        not isinstance(key, str) for key in config
    ):
        raise TypeError("Production runtime config must be a string-keyed object.")
    actual = frozenset(config)
    gui_enabled = config.get("start_isaacsim_sidecar_with_geant4") is True
    expected = PRODUCTION_RUNTIME_CONFIG_FIELDS | (
        PRODUCTION_GUI_RUNTIME_CONFIG_FIELDS if gui_enabled else frozenset()
    )
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            "Production runtime config fields differ from the canonical schema: "
            f"missing={missing}, unknown_or_retired={unknown}."
        )
    schema_version = config["simulation_runtime_schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError(
            "Production runtime config requires simulation_runtime_schema_version=1."
        )
    invalid_booleans = sorted(
        name for name in _PRODUCTION_BOOLEAN_FIELDS if type(config[name]) is not bool
    )
    if invalid_booleans:
        raise TypeError(
            "Production runtime Boolean fields require exact JSON booleans: "
            f"{invalid_booleans}."
        )
    detector = config["detector_model"]
    if not isinstance(detector, Mapping):
        raise TypeError("Production detector_model must be an object.")
    detector_actual = frozenset(detector)
    detector_missing = sorted(_PRODUCTION_DETECTOR_MODEL_FIELDS - detector_actual)
    detector_unknown = sorted(detector_actual - _PRODUCTION_DETECTOR_MODEL_FIELDS)
    if detector_missing or detector_unknown:
        raise ValueError(
            "Production detector_model fields differ from the canonical schema: "
            f"missing={detector_missing}, unknown={detector_unknown}."
        )
    if config["backend"] != "geant4":
        raise ValueError("Production runtime config requires backend='geant4'.")
    expected_mock_stage = not config["start_isaacsim_sidecar_with_geant4"]
    if config["use_mock_stage"] is not expected_mock_stage:
        raise ValueError(
            "Production use_mock_stage must be true for no-Isaac in-memory scene "
            "authoring and false when the Isaac Sim sidecar is enabled."
        )
    if gui_enabled:
        if type(config["isaacsim_keep_sidecar_alive"]) is not bool:
            raise TypeError(
                "Production isaacsim_keep_sidecar_alive requires an exact JSON boolean."
            )
        if config["isaacsim_keep_sidecar_alive"] is not False:
            raise ValueError(
                "Production GUI requires isaacsim_keep_sidecar_alive=false so "
                "the fresh Isaac Sim sidecar has one exact acquisition owner."
            )


def load_production_runtime_config(path: str | Path) -> dict[str, Any]:
    """Load one self-contained canonical production runtime configuration."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Simulation config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(
            handle,
            object_pairs_hook=_runtime_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    if not isinstance(data, dict):
        raise ValueError("Production simulation config must be a JSON object.")
    if "extends" in data:
        raise ValueError(
            "Production runtime config cannot use retired 'extends' inheritance."
        )
    validate_production_runtime_config(data)
    usd_path = data["usd_path"]
    if not isinstance(usd_path, str) or not usd_path.strip():
        raise TypeError("Production usd_path must be a nonempty JSON string.")
    resolved_usd_path = Path(usd_path).expanduser()
    if not resolved_usd_path.is_absolute():
        resolved_usd_path = (
            _repo_root() / "configs" / "geant4" / resolved_usd_path
        ).resolve()
    data["usd_path"] = resolved_usd_path.as_posix()
    from sim.geant4_app.app import Geant4AppConfig

    Geant4AppConfig.from_dict(data)
    return data


def production_runtime_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash one normalized exact production runtime configuration."""
    validate_production_runtime_config(config)
    usd_path = config["usd_path"]
    if not isinstance(usd_path, str) or not Path(usd_path).is_absolute():
        raise ValueError(
            "Production runtime config must be loaded and canonicalized before hashing."
        )
    from sim.geant4_app.app import Geant4AppConfig

    Geant4AppConfig.from_dict(dict(config))
    return strict_sha256_json(dict(config))


def load_production_runtime_config_with_digest(
    path: str | Path,
) -> tuple[dict[str, Any], str]:
    """Load one canonical production config and return its sole approved digest."""
    config = load_production_runtime_config(path)
    return config, production_runtime_config_sha256(config)


def _load_runtime_config(
    path: str | Path | None,
    *,
    seen: set[Path],
) -> dict[str, Any]:
    """Load a runtime config and recursively merge an ``extends`` parent."""
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Simulation config not found: {config_path}")
    config_path = config_path.resolve()
    if config_path in seen:
        raise ValueError(f"Cyclic runtime config inheritance at {config_path}")
    seen.add(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(
            handle,
            object_pairs_hook=_runtime_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    if not isinstance(data, dict):
        raise ValueError("Simulation config must be a JSON object.")
    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data
    if not isinstance(parent_ref, str) or not parent_ref.strip():
        raise ValueError("Runtime config 'extends' must be a nonempty JSON string.")
    parent_path = Path(parent_ref).expanduser()
    if not parent_path.is_absolute():
        parent_path = (config_path.parent / parent_path).resolve()
    parent = _load_runtime_config(parent_path, seen=seen)
    merged = dict(parent)
    merged.update(data)
    return merged


def _repo_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[2]


def _tcp_server_available(host: str, port: int, timeout_s: float = 0.25) -> bool:
    """Return True when a TCP server is already accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _write_temp_sidecar_config(config: dict[str, Any]) -> Path:
    """Write an ephemeral sidecar config file and return its path."""
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="geant4_sidecar_",
        delete=False,
        encoding="utf-8",
    )
    with handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    return Path(handle.name)


def _merged_config_from_path(
    config_path: Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a config file and apply optional overrides."""
    loaded = load_runtime_config(config_path)
    merged = dict(loaded)
    if overrides:
        merged.update(overrides)
    return merged


def _resolve_executable_path(path_value: str) -> str:
    """Expand shell variables and user home markers in an executable path."""
    return Path(os.path.expandvars(path_value)).expanduser().as_posix()


def _local_isaacsim_python_candidates() -> list[Path]:
    """Return likely Isaac Sim Python launchers installed on this machine."""
    candidates: list[Path] = []
    home = Path.home()
    local_root = home / ".local" / "isaacsim"
    if local_root.exists():
        candidates.extend(sorted(local_root.glob("*/python.sh"), reverse=True))
    candidates.extend(
        [
            home / ".local" / "isaacsim" / "python.sh",
            home / "isaacsim" / "python.sh",
            Path("/opt/isaacsim/python.sh"),
        ]
    )
    return candidates


def _resolve_local_isaacsim_python() -> str | None:
    """Return a local Isaac Sim Python launcher if one exists."""
    for candidate in _local_isaacsim_python_candidates():
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.as_posix()
    return None


def _config_requires_isaacsim_python(config: dict[str, Any]) -> bool:
    """Return True when a sidecar must be launched with Isaac Sim's Python."""
    if _config_bool(config, "requires_isaacsim_python", False):
        return True
    if str(config.get("mode", "")).strip().lower() == "real":
        return True
    return config.get("use_mock_stage") is False


def _resolve_sidecar_python(config: dict[str, Any], sidecar_name: str) -> str:
    """Resolve the Python executable used to launch a sidecar process."""
    configured = config.get("sidecar_python")
    if configured not in (None, ""):
        return _resolve_executable_path(str(configured))

    env_names: list[str] = ["SIMBRIDGE_SIDECAR_PYTHON"]
    configured_env = config.get("sidecar_python_env")
    if configured_env not in (None, ""):
        env_names.insert(0, str(configured_env))
    if _config_requires_isaacsim_python(config) and "ISAACSIM_PYTHON" not in env_names:
        env_names.append("ISAACSIM_PYTHON")
    for env_name in env_names:
        env_value = os.environ.get(env_name)
        if env_value:
            return _resolve_executable_path(env_value)

    if _config_requires_isaacsim_python(config):
        local_python = _resolve_local_isaacsim_python()
        if local_python is not None:
            return local_python
        raise RuntimeError(
            f"{sidecar_name} sidecar requires Isaac Sim Python. Set "
            "ISAACSIM_PYTHON=/path/to/isaacsim/python.sh or set "
            "sidecar_python in the config."
        )
    return sys.executable


def _start_sidecar_process(
    *,
    script_path: Path,
    config_path: Path,
    config: dict[str, Any],
    host: str,
    port: int,
    timeout_s: float,
    log_path: Path,
    sidecar_name: str,
    extra_args: list[str] | None = None,
) -> tuple[subprocess.Popen[str], object]:
    """Start a sidecar subprocess and wait until it accepts TCP connections."""
    root = _repo_root()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    python_executable = _resolve_sidecar_python(config, sidecar_name)
    command = [
        python_executable,
        script_path.as_posix(),
        "--config",
        config_path.as_posix(),
    ]
    if extra_args:
        command.extend(extra_args)
    log_handle: object | None = None
    process: subprocess.Popen[str] | None = None
    try:
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=root.as_posix(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Auto-started {sidecar_name} sidecar exited before "
                    f"accepting connections. See log: {log_path}"
                )
            if _tcp_server_available(host, port):
                print(
                    f"Auto-started {sidecar_name} sidecar on {host}:{port} "
                    f"(log: {log_path})"
                )
                return process, log_handle
            time.sleep(0.1)
        raise TimeoutError(
            f"Timed out waiting for auto-started {sidecar_name} sidecar on "
            f"{host}:{port}. See log: {log_path}"
        )
    except BaseException as startup_failure:
        _cleanup_failed_sidecar_process(
            process=process,
            log_handle=log_handle,
            failure=startup_failure,
        )
        raise


def _cleanup_failed_sidecar_process(
    *,
    process: subprocess.Popen[str] | None,
    log_handle: object | None,
    failure: BaseException,
) -> None:
    """Terminate startup-owned process handles without masking the failure."""
    cleanup_failures: list[BaseException] = []
    if process is not None:
        try:
            running = process.poll() is None
        except BaseException as cleanup_failure:
            cleanup_failures.append(cleanup_failure)
            running = True
        if running:
            cleanup_failures.extend(_force_stop_owned_process(process))
    if log_handle is not None:
        try:
            close = getattr(log_handle, "close", None)
            if close is not None:
                close()
        except BaseException as cleanup_failure:
            cleanup_failures.append(cleanup_failure)
    for cleanup_failure in cleanup_failures:
        failure.add_note(
            "Sidecar startup cleanup also failed: "
            f"{type(cleanup_failure).__name__}: {cleanup_failure}"
        )


def _resolve_geant4_sidecar_config_path(
    config: dict[str, Any],
    runtime_config_path: str | Path | None,
) -> tuple[Path, Path | None]:
    """Freeze the validated mapping for the child without rereading its source."""
    del runtime_config_path
    temp_path = _write_temp_sidecar_config(config)
    return temp_path, temp_path


def _configured_detector_green_hashes(
    config: object,
) -> tuple[str | None, str | None]:
    """Load the exact configured detector Green identity or fail closed."""
    from spectrum.detector_green_operator import DetectorGreenOperator

    sample_response = bool(getattr(config, "sample_detector_response"))
    manifest_value = getattr(config, "detector_green_operator_manifest")
    if sample_response != (manifest_value is not None):
        raise ValueError(
            "Detector Green manifest is required exactly when response "
            "sampling is enabled."
        )
    if not sample_response:
        return None, None
    manifest_path = Path(str(manifest_value)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = _repo_root() / manifest_path
    operator = DetectorGreenOperator.from_artifact(manifest_path)
    operator.require_runtime_ready()
    return operator.contract_hash_sha256, operator.binary_sha256


def _start_geant4_sidecar(
    config: dict[str, Any],
    *,
    host: str,
    port: int,
    runtime_config_path: str | Path | None,
    expected_runtime_config_sha256: str | None = None,
    expected_native_executable_sha256: str | None = None,
    expected_native_execution_environment_sha256: str | None = None,
    expected_implementation_bundle_sha256: str | None = None,
) -> ManagedGeant4TCPClientRuntime:
    """Start a Geant4 bridge sidecar subprocess and return its managed client."""
    from sim.geant4_app.app import Geant4AppConfig

    validated_geant4_config = Geant4AppConfig.from_dict(config)
    green_contract_hash, green_binary_hash = _configured_detector_green_hashes(
        validated_geant4_config
    )
    root = _repo_root()
    script_path = root / "scripts" / "run_geant4_bridge.py"
    config_path, temp_config_path = _resolve_geant4_sidecar_config_path(
        config,
        runtime_config_path,
    )
    sidecar_geant4_config = Geant4AppConfig.from_dict(
        load_production_runtime_config(config_path)
    )
    if sidecar_geant4_config != validated_geant4_config:
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)
        raise ValueError(
            "runtime_config and the Geant4 sidecar config path resolve to "
            "different Geant4 application settings."
        )
    log_path = Path(
        str(
            config.get(
                "sidecar_log_path",
                root / "results" / "sidecars" / f"geant4_bridge_{port}.log",
            )
        )
    ).expanduser()
    if not log_path.is_absolute():
        log_path = (root / log_path).resolve()
    startup_timeout_s = _config_number(
        config,
        "sidecar_startup_timeout_s",
        30.0,
        minimum=0.0,
        strict_minimum=True,
    )
    try:
        process, log_handle = _start_sidecar_process(
            script_path=script_path,
            config_path=config_path,
            config=config,
            host=host,
            port=port,
            timeout_s=startup_timeout_s,
            log_path=log_path,
            sidecar_name="Geant4",
            extra_args=None,
        )
    except BaseException:
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)
        raise
    try:
        return ManagedGeant4TCPClientRuntime(
            host=host,
            port=port,
            timeout_s=_config_number(
                config,
                "timeout_s",
                120.0,
                minimum=0.0,
                strict_minimum=True,
            ),
            process=process,
            log_handle=log_handle,
            temp_config_path=temp_config_path,
            expected_primary_sampling_fraction=(
                validated_geant4_config.primary_sampling_fraction
            ),
            expected_target_sampled_primaries=(
                validated_geant4_config.target_sampled_primaries
            ),
            accelerated_weighted_transport_enable=(
                validated_geant4_config.accelerated_weighted_transport_enable
            ),
            expected_source_rate_model=validated_geant4_config.source_rate_model,
            expected_thread_count=validated_geant4_config.thread_count,
            expected_physics_profile=validated_geant4_config.physics_profile,
            expected_detector_scoring_mode=(
                validated_geant4_config.detector_scoring_mode
            ),
            expected_secondary_transport_mode=(
                validated_geant4_config.secondary_transport_mode
            ),
            expected_source_bias_mode=validated_geant4_config.source_bias_mode,
            expected_background_cps=validated_geant4_config.background_cps,
            expected_dead_time_tau_s=validated_geant4_config.dead_time_tau_s,
            expected_detector_response_sampling=(
                validated_geant4_config.sample_detector_response
            ),
            expected_detector_green_operator_contract_sha256=(green_contract_hash),
            expected_detector_green_operator_binary_sha256=(green_binary_hash),
            expected_runtime_config_sha256=expected_runtime_config_sha256,
            expected_native_executable_sha256=(expected_native_executable_sha256),
            expected_native_execution_environment_sha256=(
                expected_native_execution_environment_sha256
            ),
            expected_implementation_bundle_sha256=(
                expected_implementation_bundle_sha256
            ),
        )
    except BaseException as startup_failure:
        _cleanup_failed_sidecar_process(
            process=process,
            log_handle=log_handle,
            failure=startup_failure,
        )
        if temp_config_path is not None:
            try:
                temp_config_path.unlink(missing_ok=True)
            except BaseException as cleanup_failure:
                startup_failure.add_note(
                    "Sidecar config cleanup also failed: "
                    f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                )
        raise


def _resolve_isaacsim_sidecar_config_path(
    config: dict[str, Any],
    runtime_config_path: str | Path | None = None,
    *,
    direct_config: bool = False,
) -> tuple[Path, Path | None, dict[str, Any]]:
    """Return an Isaac Sim sidecar config path and loaded config."""
    root = _repo_root()
    if direct_config:
        loaded = dict(config)
        if runtime_config_path is not None:
            return Path(runtime_config_path).expanduser().resolve(), None, loaded
        temp_path = _write_temp_sidecar_config(loaded)
        return temp_path, temp_path, loaded
    configured = config.get("isaacsim_sidecar_config_path")
    config_path = (
        Path(str(configured)).expanduser().resolve()
        if configured not in (None, "")
        else root / "configs" / "isaacsim" / "real_scene.json"
    )
    overrides = config.get("isaacsim_sidecar_config", {})
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("isaacsim_sidecar_config must be a JSON object when provided.")
    loaded = _merged_config_from_path(config_path, overrides)
    for src_key, dst_key in (
        ("isaacsim_host", "host"),
        ("isaacsim_port", "port"),
        ("isaacsim_timeout_s", "timeout_s"),
        ("isaacsim_mode", "mode"),
    ):
        if src_key in config:
            loaded[dst_key] = config[src_key]
    override_keys = (
        "isaacsim_host",
        "isaacsim_port",
        "isaacsim_timeout_s",
        "isaacsim_mode",
    )
    if overrides or any(key in config for key in override_keys):
        temp_path = _write_temp_sidecar_config(loaded)
        return temp_path, temp_path, loaded
    return config_path, None, loaded


def _start_isaacsim_sidecar(
    config: dict[str, Any],
    runtime_config_path: str | Path | None = None,
    *,
    direct_config: bool = False,
    require_fresh_owned_process: bool = False,
) -> IsaacSimTCPClientRuntime:
    """Start an Isaac sidecar, optionally forbidding all endpoint reuse."""
    config_path, temp_config_path, isaac_config = _resolve_isaacsim_sidecar_config_path(
        config,
        runtime_config_path=runtime_config_path,
        direct_config=direct_config,
    )
    host = _config_string(isaac_config, "host", "127.0.0.1")
    port = _config_integer(
        isaac_config,
        "port",
        5555,
        minimum=1,
        maximum=65535,
    )
    timeout_s = _config_number(
        isaac_config,
        "timeout_s",
        10.0,
        minimum=0.0,
        strict_minimum=True,
    )
    keep_alive = (
        _config_bool(
            config,
            "isaacsim_keep_sidecar_alive",
            False,
        )
        if "isaacsim_keep_sidecar_alive" in config
        else _config_bool(isaac_config, "keep_sidecar_alive", False)
    )
    if require_fresh_owned_process and keep_alive:
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)
        raise ValueError(
            "Production Isaac Sim requires isaacsim_keep_sidecar_alive=false "
            "so every GUI sidecar remains owned by one acquisition."
        )
    if _tcp_server_available(host, port):
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)
        if require_fresh_owned_process:
            raise RuntimeError(
                "Production Isaac Sim refuses to attach to an existing TCP "
                f"endpoint at {host}:{port}."
            )
        return IsaacSimTCPClientRuntime(
            host=host,
            port=port,
            timeout_s=timeout_s,
            close_on_close=not keep_alive,
        )
    root = _repo_root()
    script_path = root / "scripts" / "run_isaacsim_bridge.py"
    default_log_path = root / "results" / "sidecars" / f"isaacsim_bridge_{port}.log"
    log_path = Path(
        str(config.get("isaacsim_sidecar_log_path", default_log_path))
    ).expanduser()
    if not log_path.is_absolute():
        log_path = (root / log_path).resolve()
    startup_timeout_s = _config_number(
        config,
        "isaacsim_sidecar_startup_timeout_s",
        60.0,
        minimum=0.0,
        strict_minimum=True,
    )
    process_config = dict(isaac_config)
    isaac_python = config.get(
        "isaacsim_sidecar_python",
        isaac_config.get("sidecar_python"),
    )
    if isaac_python not in (None, ""):
        process_config["sidecar_python"] = str(isaac_python)
    isaac_python_env = config.get(
        "isaacsim_sidecar_python_env",
        isaac_config.get("sidecar_python_env"),
    )
    if isaac_python_env not in (None, ""):
        process_config["sidecar_python_env"] = str(isaac_python_env)
    try:
        process, log_handle = _start_sidecar_process(
            script_path=script_path,
            config_path=config_path,
            config=process_config,
            host=host,
            port=port,
            timeout_s=startup_timeout_s,
            log_path=log_path,
            sidecar_name="Isaac Sim",
            extra_args=(
                ["--mock"]
                if _config_bool(config, "isaacsim_sidecar_mock", False)
                else None
            ),
        )
    except BaseException:
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)
        raise
    try:
        return ManagedIsaacSimTCPClientRuntime(
            host=host,
            port=port,
            timeout_s=timeout_s,
            process=process,
            log_handle=log_handle,
            temp_config_path=temp_config_path,
            close_on_close=not keep_alive,
        )
    except BaseException as startup_failure:
        _cleanup_failed_sidecar_process(
            process=process,
            log_handle=log_handle,
            failure=startup_failure,
        )
        if temp_config_path is not None:
            try:
                temp_config_path.unlink(missing_ok=True)
            except BaseException as cleanup_failure:
                startup_failure.add_note(
                    "Sidecar config cleanup also failed: "
                    f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                )
        raise


def _maybe_pair_geant4_with_isaacsim(
    config: dict[str, Any],
    geant4_runtime: SimulationRuntime,
    *,
    require_fresh_isaacsim: bool = False,
) -> SimulationRuntime:
    """Wrap Geant4 with an Isaac Sim companion runtime when configured."""
    if not _config_bool(
        config,
        "start_isaacsim_sidecar_with_geant4",
        True,
    ):
        return geant4_runtime
    try:
        isaacsim_runtime = _start_isaacsim_sidecar(
            config,
            require_fresh_owned_process=require_fresh_isaacsim,
        )
    except BaseException as startup_failure:
        try:
            geant4_runtime.close()
        except BaseException as cleanup_failure:
            startup_failure.add_note(
                "Geant4 cleanup after Isaac Sim startup failure also failed: "
                f"{type(cleanup_failure).__name__}: {cleanup_failure}"
            )
        raise
    return Geant4WithIsaacSimRuntime(
        geant4_runtime=geant4_runtime,
        isaacsim_runtime=isaacsim_runtime,
    )


def create_simulation_runtime(
    backend: str,
    *,
    sources: list[PointSource],
    mu_by_isotope: dict[str, object],
    shield_params: Any,
    runtime_config: dict[str, Any] | None = None,
    runtime_config_path: str | Path | None = None,
    expected_runtime_config_sha256: str | None = None,
    expected_native_executable_sha256: str | None = None,
    expected_native_execution_environment_sha256: str | None = None,
    expected_implementation_bundle_sha256: str | None = None,
) -> SimulationRuntime:
    """Instantiate the requested simulation runtime."""
    config = {} if runtime_config is None else dict(runtime_config)
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("Simulation backend must be a nonempty string.")
    normalized = backend.strip().lower()
    if normalized == "analytic":
        rng_seed = _config_integer(
            config,
            "rng_seed",
            123,
            minimum=0,
        )
        return AnalyticSimulationRuntime(
            sources=sources,
            mu_by_isotope=mu_by_isotope,
            shield_params=shield_params,
            rng_seed=rng_seed,
            obstacle_height_m=_config_number(
                config,
                "obstacle_height_m",
                2.0,
                minimum=0.0,
            ),
            obstacle_material=_config_string(
                config,
                "obstacle_material",
                "concrete",
            ),
            scatter_gain=_config_number(
                config,
                "scatter_gain",
                0.03,
                minimum=0.0,
            ),
            dead_time_s=_config_number(
                config,
                "dead_time_tau_s",
                0.0,
                minimum=0.0,
            ),
            background_rate_cps=_config_number(
                config,
                "background_rate_cps",
                0.0,
                minimum=0.0,
            ),
            detector_model=(
                dict(config["detector_model"])
                if isinstance(config.get("detector_model"), dict)
                else None
            ),
        )
    if normalized == "isaacsim":
        host = _config_string(config, "host", "127.0.0.1")
        port = _config_integer(
            config,
            "port",
            5555,
            minimum=1,
            maximum=65535,
        )
        timeout_s = _config_number(
            config,
            "timeout_s",
            10.0,
            minimum=0.0,
            strict_minimum=True,
        )
        keep_alive = _config_bool(config, "keep_sidecar_alive", False)
        auto_start = _config_bool(config, "auto_start_sidecar", True)
        if auto_start and not _tcp_server_available(host, port):
            return _start_isaacsim_sidecar(
                config,
                runtime_config_path=runtime_config_path,
                direct_config=True,
            )
        return IsaacSimTCPClientRuntime(
            host=host,
            port=port,
            timeout_s=timeout_s,
            close_on_close=not keep_alive,
        )
    if normalized == "geant4":
        from sim.geant4_app.app import Geant4AppConfig

        gui_enabled = config.get("start_isaacsim_sidecar_with_geant4") is True
        production_fields = PRODUCTION_RUNTIME_CONFIG_FIELDS | (
            PRODUCTION_GUI_RUNTIME_CONFIG_FIELDS if gui_enabled else frozenset()
        )
        production_config = frozenset(config) == production_fields
        if production_config:
            actual_runtime_config_sha256 = production_runtime_config_sha256(config)
            if expected_runtime_config_sha256 is None:
                expected_runtime_config_sha256 = actual_runtime_config_sha256
            elif expected_runtime_config_sha256 != actual_runtime_config_sha256:
                raise ValueError(
                    "Expected production runtime-config hash does not authenticate "
                    "the supplied canonical configuration."
                )
            if (
                expected_native_executable_sha256 is None
                or expected_native_execution_environment_sha256 is None
                or expected_implementation_bundle_sha256 is None
            ):
                raise ValueError(
                    "Production Geant4 startup requires approved native "
                    "executable and execution-environment digests."
                )
        validated_geant4_config = Geant4AppConfig.from_dict(config)
        green_contract_hash, green_binary_hash = _configured_detector_green_hashes(
            validated_geant4_config
        )
        host = _config_string(config, "host", "127.0.0.1")
        port = _config_integer(
            config,
            "port",
            5556,
            minimum=1,
            maximum=65535,
        )
        timeout_s = _config_number(
            config,
            "timeout_s",
            120.0,
            minimum=0.0,
            strict_minimum=True,
        )
        auto_start = _config_bool(config, "auto_start_sidecar", True)
        endpoint_occupied = _tcp_server_available(host, port)
        if production_config and auto_start is not True:
            raise ValueError("Production Geant4 requires auto_start_sidecar=true.")
        if production_config and endpoint_occupied:
            raise RuntimeError(
                "Production Geant4 refuses to attach to an existing TCP "
                f"endpoint at {host}:{port}."
            )
        if auto_start and not endpoint_occupied:
            geant4_runtime = _start_geant4_sidecar(
                config,
                host=host,
                port=port,
                runtime_config_path=runtime_config_path,
                expected_runtime_config_sha256=(expected_runtime_config_sha256),
                expected_native_executable_sha256=(expected_native_executable_sha256),
                expected_native_execution_environment_sha256=(
                    expected_native_execution_environment_sha256
                ),
                expected_implementation_bundle_sha256=(
                    expected_implementation_bundle_sha256
                ),
            )
        else:
            geant4_runtime = Geant4TCPClientRuntime(
                host=host,
                port=port,
                timeout_s=timeout_s,
                expected_primary_sampling_fraction=(
                    validated_geant4_config.primary_sampling_fraction
                ),
                expected_target_sampled_primaries=(
                    validated_geant4_config.target_sampled_primaries
                ),
                accelerated_weighted_transport_enable=(
                    validated_geant4_config.accelerated_weighted_transport_enable
                ),
                expected_source_rate_model=validated_geant4_config.source_rate_model,
                expected_thread_count=validated_geant4_config.thread_count,
                expected_physics_profile=validated_geant4_config.physics_profile,
                expected_detector_scoring_mode=(
                    validated_geant4_config.detector_scoring_mode
                ),
                expected_secondary_transport_mode=(
                    validated_geant4_config.secondary_transport_mode
                ),
                expected_source_bias_mode=validated_geant4_config.source_bias_mode,
                expected_background_cps=validated_geant4_config.background_cps,
                expected_dead_time_tau_s=validated_geant4_config.dead_time_tau_s,
                expected_detector_response_sampling=(
                    validated_geant4_config.sample_detector_response
                ),
                expected_detector_green_operator_contract_sha256=(green_contract_hash),
                expected_detector_green_operator_binary_sha256=(green_binary_hash),
                expected_runtime_config_sha256=(expected_runtime_config_sha256),
                expected_native_executable_sha256=(expected_native_executable_sha256),
                expected_native_execution_environment_sha256=(
                    expected_native_execution_environment_sha256
                ),
                expected_implementation_bundle_sha256=(
                    expected_implementation_bundle_sha256
                ),
            )
        return _maybe_pair_geant4_with_isaacsim(
            config,
            geant4_runtime,
            require_fresh_isaacsim=production_config,
        )
    raise ValueError(f"Unknown simulation backend: {backend}")
