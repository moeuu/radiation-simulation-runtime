"""Estimator-neutral client for one adaptive acquisition subprocess."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from runtime.adaptive import ADAPTIVE_CUI_OVERLAY_PREFIX, ADAPTIVE_EVENT_PREFIX
from runtime.measurement_log import MeasurementLogRecord
from runtime.records import RunContext, validate_truth_free_estimator_input


@dataclass(frozen=True, slots=True)
class AdaptiveResumePrefix:
    """Store the verified completed-station prefix returned during resume."""

    records: tuple[MeasurementLogRecord, ...]
    next_station_id: int


def _strict_fields(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    name: str,
) -> None:
    """Require one exact protocol object schema."""
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{name} fields disagree with the adaptive protocol: "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )


def parse_adaptive_record(payload: object) -> MeasurementLogRecord:
    """Parse one truth-free durable record returned by the runtime."""
    if not isinstance(payload, dict):
        raise TypeError("Adaptive record must be an object.")
    validate_truth_free_estimator_input(payload, path="adaptive.record")
    expected = {
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
    _strict_fields(payload, expected, name="adaptive record")
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


def parse_adaptive_resume_prefix(payload: object) -> AdaptiveResumePrefix:
    """Parse and validate one truth-free adaptive resume handshake prefix."""
    if not isinstance(payload, dict):
        raise TypeError("Adaptive resume prefix must be an object.")
    validate_truth_free_estimator_input(payload, path="adaptive.resume")
    _strict_fields(
        payload,
        {"record_count", "records", "next_station_id"},
        name="adaptive resume prefix",
    )
    raw_count = payload["record_count"]
    raw_next_station = payload["next_station_id"]
    raw_records = payload["records"]
    if isinstance(raw_count, bool) or not isinstance(raw_count, int):
        raise TypeError("Adaptive resume record_count must be an integer.")
    if isinstance(raw_next_station, bool) or not isinstance(raw_next_station, int):
        raise TypeError("Adaptive resume next_station_id must be an integer.")
    if not isinstance(raw_records, list) or not raw_records:
        raise TypeError("Adaptive resume records must be a nonempty list.")
    records = tuple(parse_adaptive_record(record) for record in raw_records)
    if raw_count != len(records):
        raise ValueError("Adaptive resume record_count disagrees with records.")
    station_ids = [int(record.station_id) for record in records]
    if [int(record.step_id) for record in records] != list(range(len(records))):
        raise ValueError("Adaptive resume step_id must equal causal record order.")
    if [int(record.action_id) for record in records] != list(range(len(records))):
        raise ValueError("Adaptive resume action_id must equal causal record order.")
    if sorted(set(station_ids)) != list(range(station_ids[-1] + 1)):
        raise ValueError(
            "Adaptive resume stations must be contiguous and zero based."
        )
    for index, record in enumerate(records):
        station_end = index + 1 == len(records) or (
            station_ids[index + 1] != station_ids[index]
        )
        if (record.metadata.get("station_complete") is True) is not station_end:
            raise ValueError(
                "Adaptive resume records must end every station exactly once."
            )
    expected_next_station = station_ids[-1] + 1
    if raw_next_station != expected_next_station:
        raise ValueError(
            "Adaptive resume next_station_id disagrees with the durable prefix."
        )
    return AdaptiveResumePrefix(
        records=records,
        next_station_id=expected_next_station,
    )


def parse_run_context(payload: object) -> RunContext:
    """Parse the truth-free runtime handshake context."""
    if not isinstance(payload, dict):
        raise TypeError("Adaptive runtime context must be an object.")
    validate_truth_free_estimator_input(payload, path="adaptive.context")
    expected = {
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
    _strict_fields(payload, expected, name="adaptive context")
    return RunContext(
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


def parse_candidate_snapshot(payload: object) -> dict[str, object]:
    """Validate one runtime-owned reachable candidate snapshot."""
    if not isinstance(payload, dict):
        raise TypeError("Adaptive candidates must be an object.")
    validate_truth_free_estimator_input(payload, path="adaptive.candidates")
    _strict_fields(
        payload,
        {
            "candidate_poses_xyz",
            "travel_costs",
            "allowed_pair_ids",
            "current_pair_id",
        },
        name="adaptive candidates",
    )
    poses = np.asarray(payload["candidate_poses_xyz"], dtype=np.float64)
    costs = np.asarray(payload["travel_costs"], dtype=np.float64)
    pair_ids = np.asarray(payload["allowed_pair_ids"])
    if poses.ndim != 2 or poses.shape[1:] != (3,) or not len(poses):
        raise ValueError("Runtime candidate poses must have nonempty shape (C, 3).")
    if costs.shape != (len(poses),) or np.any(~np.isfinite(costs)) or np.any(costs < 0.0):
        raise ValueError("Runtime travel costs must align with candidate poses.")
    if pair_ids.shape != (64,) or not np.array_equal(
        np.sort(pair_ids.astype(np.int64, copy=False)),
        np.arange(64, dtype=np.int64),
    ):
        raise ValueError("Adaptive runtime must expose every Fe/Pb pair 0..63.")
    return dict(payload)


def parse_cui_overlay_payload(payload: object) -> dict[str, object]:
    """Parse private CUI overlay data that must not enter estimators."""
    if not isinstance(payload, dict):
        raise TypeError("CUI overlay payload must be an object.")
    _strict_fields(
        payload,
        {"type", "schema_version", "truth"},
        name="adaptive CUI overlay",
    )
    if payload.get("type") != "cui_overlay" or payload.get("schema_version") != 1:
        raise ValueError("Adaptive CUI overlay schema is incompatible.")
    truth = payload.get("truth")
    if truth is None:
        return dict(payload)
    if not isinstance(truth, dict):
        raise TypeError("CUI truth overlay must be an object or null.")
    _strict_fields(
        truth,
        {"schema_version", "semantics", "true_sources", "true_strengths"},
        name="adaptive CUI truth overlay",
    )
    if truth.get("schema_version") != 1:
        raise ValueError("CUI truth overlay schema is incompatible.")
    sources = truth["true_sources"]
    strengths = truth["true_strengths"]
    if not isinstance(sources, dict) or not isinstance(strengths, dict):
        raise TypeError("CUI truth sources and strengths must be objects.")
    if set(sources) != set(strengths):
        raise ValueError("CUI truth sources and strengths isotope sets differ.")
    for isotope, raw_positions in sources.items():
        if not isinstance(isotope, str):
            raise TypeError("CUI truth isotope keys must be strings.")
        positions = np.asarray(raw_positions, dtype=np.float64)
        if positions.size == 0:
            positions = positions.reshape((0, 3))
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or np.any(~np.isfinite(positions))
        ):
            raise ValueError("CUI truth positions must have finite shape (N, 3).")
        raw_strengths = np.asarray(strengths[isotope], dtype=np.float64).reshape(-1)
        if (
            raw_strengths.shape != (positions.shape[0],)
            or np.any(~np.isfinite(raw_strengths))
        ):
            raise ValueError("CUI truth strengths must align with positions.")
    return dict(payload)


def adaptive_step_request(
    *,
    candidate_index: int,
    fe_orientation_index: int,
    pb_orientation_index: int,
    dwell_time_s: float,
    station_id: int,
    station_complete: bool,
) -> dict[str, object]:
    """Build one schema-v1 adaptive observation request."""
    return {
        "type": "step",
        "candidate_index": int(candidate_index),
        "fe_orientation_index": int(fe_orientation_index),
        "pb_orientation_index": int(pb_orientation_index),
        "dwell_time_s": float(dwell_time_s),
        "station_id": int(station_id),
        "station_complete": bool(station_complete),
    }


def candidate_index_for_pose(
    candidates: Mapping[str, object],
    pose_xyz: Sequence[float],
) -> int:
    """Locate one exact retained pose in a new runtime candidate snapshot."""
    poses = np.asarray(candidates["candidate_poses_xyz"], dtype=np.float64)
    target = np.asarray(pose_xyz, dtype=np.float64)
    matches = np.flatnonzero(
        np.all(np.isclose(poses, target[None, :], rtol=0.0, atol=1.0e-10), axis=1)
    )
    if matches.size != 1:
        raise RuntimeError(
            "The runtime candidate domain did not preserve the selected station pose."
        )
    return int(matches[0])


class AdaptiveRuntimeClient:
    """Drive the shared runtime without opening its private physical scenario."""

    def __init__(
        self,
        scenario_path: str | Path,
        *,
        runtime_root: str | Path,
        private_scene_profile: str | None = None,
        resume_stage_path: str | Path | None = None,
        resume_compatibility_path: str | Path | None = None,
        output_hook: Callable[[str], None] = print,
    ) -> None:
        """Start one persistent runtime-owned adaptive subprocess."""
        scenario = Path(scenario_path).expanduser().resolve()
        if not scenario.is_file():
            raise FileNotFoundError(f"Private adaptive scenario is missing: {scenario}")
        root = Path(runtime_root).expanduser().resolve()
        command = [
            "uv",
            "run",
            "--project",
            root.as_posix(),
            "rotating-shield-sim",
            "run-adaptive-session",
            scenario.as_posix(),
        ]
        if private_scene_profile is not None:
            command.extend(("--private-scene-profile", private_scene_profile))
        if resume_compatibility_path is not None and resume_stage_path is None:
            raise ValueError(
                "resume_compatibility_path requires resume_stage_path."
            )
        if resume_stage_path is not None:
            stage = Path(resume_stage_path).expanduser().resolve()
            if not stage.is_dir():
                raise FileNotFoundError(f"Adaptive resume stage is missing: {stage}")
            command.extend(("--resume-stage", stage.as_posix()))
        if resume_compatibility_path is not None:
            compatibility = Path(resume_compatibility_path).expanduser().resolve()
            if not compatibility.is_file():
                raise FileNotFoundError(
                    "Adaptive resume compatibility file is missing: "
                    f"{compatibility}"
                )
            command.extend(("--resume-compatibility", compatibility.as_posix()))
        self.command = command
        self.output_hook = output_hook
        self.process = subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.input: TextIO | None = self.process.stdin
        self.output: TextIO | None = self.process.stdout
        if self.input is None or self.output is None:
            self.process.kill()
            raise RuntimeError("Shared runtime did not expose adaptive pipes.")

    def read_event(self) -> dict[str, Any]:
        """Read the next framed event while relaying runtime diagnostics."""
        assert self.output is not None
        for raw_line in self.output:
            line = raw_line.rstrip("\n")
            if line.startswith(ADAPTIVE_CUI_OVERLAY_PREFIX):
                raise RuntimeError(
                    "Shared runtime emitted a private CUI overlay during an "
                    "estimator-visible read."
                )
            if not line.startswith(ADAPTIVE_EVENT_PREFIX):
                self.output_hook(line)
                continue
            payload = json.loads(line.removeprefix(ADAPTIVE_EVENT_PREFIX))
            if not isinstance(payload, dict):
                raise TypeError("Adaptive runtime event must be an object.")
            validate_truth_free_estimator_input(payload, path="adaptive.event")
            return payload
        return_code = self.process.poll()
        raise RuntimeError(
            "Shared adaptive runtime closed before its next event; "
            f"return_code={return_code}."
        )

    def request(self, payload: Mapping[str, object]) -> dict[str, Any]:
        """Send one causal controller decision and wait for its response."""
        if self.input is None:
            raise RuntimeError("Adaptive runtime input is closed.")
        validate_truth_free_estimator_input(payload, path="adaptive.request")
        self.input.write(json.dumps(dict(payload), allow_nan=False) + "\n")
        self.input.flush()
        return self.read_event()

    def request_cui_overlay(self, *, include_truth: bool) -> dict[str, object]:
        """Request private CUI overlay data outside the estimator protocol."""
        if self.input is None:
            raise RuntimeError("Adaptive runtime input is closed.")
        if not isinstance(include_truth, bool):
            raise TypeError("include_truth must be a boolean.")
        payload = {
            "type": "cui_overlay",
            "include_truth": include_truth,
        }
        self.input.write(json.dumps(payload, allow_nan=False) + "\n")
        self.input.flush()
        assert self.output is not None
        for raw_line in self.output:
            line = raw_line.rstrip("\n")
            if line.startswith(ADAPTIVE_CUI_OVERLAY_PREFIX):
                payload = json.loads(
                    line.removeprefix(ADAPTIVE_CUI_OVERLAY_PREFIX)
                )
                return parse_cui_overlay_payload(payload)
            if line.startswith(ADAPTIVE_EVENT_PREFIX):
                raise RuntimeError(
                    "Shared runtime emitted an estimator event during a CUI "
                    "overlay request."
                )
            self.output_hook(line)
        return_code = self.process.poll()
        raise RuntimeError(
            "Shared adaptive runtime closed before its CUI overlay event; "
            f"return_code={return_code}."
        )

    def finalize(self) -> dict[str, Any]:
        """Finalize the runtime log and require a clean process exit."""
        event = self.request({"type": "finalize"})
        if self.input is not None:
            self.input.close()
            self.input = None
        return_code = self.process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, self.command)
        return event

    def abort(self) -> None:
        """Best-effort close of an incomplete acquisition session."""
        if self.process.poll() is not None:
            return
        try:
            self.request({"type": "abort"})
        except (BrokenPipeError, OSError, RuntimeError, ValueError):
            self.process.terminate()
        finally:
            try:
                self.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


__all__ = [
    "AdaptiveResumePrefix",
    "AdaptiveRuntimeClient",
    "adaptive_step_request",
    "candidate_index_for_pose",
    "parse_adaptive_record",
    "parse_adaptive_resume_prefix",
    "parse_candidate_snapshot",
    "parse_cui_overlay_payload",
    "parse_run_context",
]
