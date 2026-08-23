"""Estimator-neutral client for one adaptive acquisition subprocess."""

from __future__ import annotations

import json
import socket
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, TextIO

import numpy as np

from runtime.adaptive_protocol import (
    ADAPTIVE_CUI_OVERLAY_PREFIX,
    ADAPTIVE_CUI_OVERLAY_FRAMING,
    ADAPTIVE_EVENT_PREFIX,
    ADAPTIVE_EVENT_FRAMING,
    AdaptiveAbortedEvent,
    AdaptiveBootstrap,
    AdaptiveCandidateSnapshot,
    AdaptiveCandidatesEvent,
    AdaptivePublishedEvent,
    AdaptiveReadyEvent,
    AdaptiveRecordEvent,
    AdaptiveRefineRequest,
    AdaptiveResumePrefix,
    AdaptiveSessionEvent,
    AdaptiveStepRequest,
    parse_adaptive_event,
)
from runtime.measurement_log import MeasurementLogRecord
from runtime.provenance import strict_canonical_json_bytes, strict_json_loads
from runtime.records import (
    RunContext,
    measurement_record_from_payload,
    validate_truth_free_estimator_input,
)


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
    return measurement_record_from_payload(payload)


def parse_adaptive_resume_prefix(payload: object) -> AdaptiveResumePrefix:
    """Parse and validate one truth-free adaptive resume handshake prefix."""
    if not isinstance(payload, dict):
        raise TypeError("Adaptive resume prefix must be an object.")
    return AdaptiveResumePrefix.from_payload(payload)


def parse_run_context(payload: object) -> RunContext:
    """Parse the truth-free runtime handshake context."""
    if not isinstance(payload, dict):
        raise TypeError("Adaptive runtime context must be an object.")
    return RunContext.from_payload(payload)


def parse_candidate_snapshot(payload: object) -> dict[str, object]:
    """Validate one runtime-owned reachable candidate snapshot."""
    if not isinstance(payload, dict):
        raise TypeError("Adaptive candidates must be an object.")
    return AdaptiveCandidateSnapshot.from_payload(payload).to_payload()


def _parse_truth_free_cui_overlay_payload(payload: object) -> dict[str, object]:
    """Parse a CUI overlay only when it contains no realized truth."""
    if not isinstance(payload, dict):
        raise TypeError("CUI overlay payload must be an object.")
    _strict_fields(
        payload,
        {"type", "schema_version", "truth"},
        name="adaptive CUI overlay",
    )
    if payload.get("type") != "cui_overlay" or payload.get("schema_version") != 1:
        raise ValueError("Adaptive CUI overlay schema is incompatible.")
    if payload.get("truth") is not None:
        raise ValueError(
            "Estimator-facing AdaptiveRuntimeClient cannot receive realized truth."
        )
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
    candidates: Mapping[str, object] | AdaptiveCandidateSnapshot,
    pose_xyz: Sequence[float],
) -> int:
    """Locate one exact retained pose in a mapping or typed snapshot."""
    if isinstance(candidates, AdaptiveCandidateSnapshot):
        raw_poses = candidates.candidate_poses_xyz
    elif isinstance(candidates, Mapping):
        raw_poses = candidates["candidate_poses_xyz"]
    else:
        raise TypeError("candidates must be a mapping or AdaptiveCandidateSnapshot.")
    poses = np.asarray(raw_poses, dtype=np.float64)
    if (
        poses.ndim != 2
        or poses.shape[0] == 0
        or poses.shape[1] != 3
        or np.any(~np.isfinite(poses))
    ):
        raise ValueError("candidate_poses_xyz must have finite nonempty shape (C, 3).")
    target = np.asarray(pose_xyz, dtype=np.float64)
    if target.shape != (3,) or np.any(~np.isfinite(target)):
        raise ValueError("pose_xyz must contain exactly three finite coordinates.")
    matches = np.flatnonzero(
        np.all(np.isclose(poses, target[None, :], rtol=0.0, atol=1.0e-10), axis=1)
    )
    if matches.size != 1:
        raise RuntimeError(
            "The runtime candidate domain did not preserve the selected station pose."
        )
    return int(matches[0])


class AdaptiveProtocolDirection(StrEnum):
    """Identify one estimator-visible direction on the adaptive wire."""

    REQUEST = "request"
    EVENT = "event"


def _freeze_protocol_value(value: object) -> object:
    """Recursively freeze already strict JSON data for observer delivery."""
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_protocol_value(nested) for key, nested in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_protocol_value(nested) for nested in value)
    return value


def _thaw_protocol_value(value: object) -> object:
    """Return mutable JSON data from an immutable observer payload."""
    if isinstance(value, Mapping):
        return {str(key): _thaw_protocol_value(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_protocol_value(nested) for nested in value]
    return value


@dataclass(frozen=True, slots=True)
class AdaptiveProtocolObservation:
    """Store one ordered truth-free request or event for transcript observers."""

    sequence_id: int
    direction: AdaptiveProtocolDirection
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        """Validate ordering and retain an immutable strict JSON payload."""
        if (
            isinstance(self.sequence_id, bool)
            or not isinstance(self.sequence_id, int)
            or self.sequence_id < 0
        ):
            raise ValueError("Adaptive observation sequence_id must be nonnegative.")
        if not isinstance(self.direction, AdaptiveProtocolDirection):
            raise TypeError("Adaptive observation direction is invalid.")
        if not isinstance(self.payload, Mapping):
            raise TypeError("Adaptive observation payload must be an object.")
        normalized = strict_json_loads(strict_canonical_json_bytes(self.payload))
        if not isinstance(normalized, dict):  # pragma: no cover - mapping invariant
            raise TypeError("Adaptive observation payload must remain an object.")
        validate_truth_free_estimator_input(
            normalized,
            path="adaptive.observation",
        )
        frozen = _freeze_protocol_value(normalized)
        if not isinstance(frozen, Mapping):  # pragma: no cover - defensive
            raise TypeError("Adaptive observation payload must remain a mapping.")
        object.__setattr__(self, "payload", frozen)

    def to_payload(self) -> dict[str, object]:
        """Return strict JSON data suitable for a durable transcript."""
        return {
            "schema_version": 1,
            "sequence_id": self.sequence_id,
            "direction": self.direction.value,
            "payload": _thaw_protocol_value(self.payload),
        }


class AdaptiveRuntimeClient:
    """Drive the shared runtime without opening its private physical scenario."""

    def __init__(
        self,
        scenario_path: str | Path,
        *,
        runtime_root: str | Path,
        resume_stage_path: str | Path | None = None,
        resume_compatibility_path: str | Path | None = None,
        output_hook: Callable[[str], None] = print,
        protocol_observer: Callable[[AdaptiveProtocolObservation], None] | None = None,
        terminate_timeout_s: float = 10.0,
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
        if resume_compatibility_path is not None and resume_stage_path is None:
            raise ValueError("resume_compatibility_path requires resume_stage_path.")
        if resume_stage_path is not None:
            stage = Path(resume_stage_path).expanduser().resolve()
            if not stage.is_dir():
                raise FileNotFoundError(f"Adaptive resume stage is missing: {stage}")
            command.extend(("--resume-stage", stage.as_posix()))
        if resume_compatibility_path is not None:
            compatibility = Path(resume_compatibility_path).expanduser().resolve()
            if not compatibility.is_file():
                raise FileNotFoundError(
                    f"Adaptive resume compatibility file is missing: {compatibility}"
                )
            command.extend(("--resume-compatibility", compatibility.as_posix()))
        self.command = command
        self.output_hook = output_hook
        if protocol_observer is not None and not callable(protocol_observer):
            raise TypeError("protocol_observer must be callable or null.")
        if (
            isinstance(terminate_timeout_s, bool)
            or not isinstance(terminate_timeout_s, (int, float))
            or not np.isfinite(float(terminate_timeout_s))
            or float(terminate_timeout_s) <= 0.0
        ):
            raise ValueError("terminate_timeout_s must be finite and positive.")
        self.protocol_observer = protocol_observer
        self.terminate_timeout_s = float(terminate_timeout_s)
        self._observation_sequence = 0
        self._closed = False
        self._finalized = False
        self._socket: socket.socket | None = None
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

    @classmethod
    def connect(
        cls,
        socket_path: str | Path,
        *,
        output_hook: Callable[[str], None] = print,
        protocol_observer: Callable[[AdaptiveProtocolObservation], None] | None = None,
        connect_timeout_s: float = 30.0,
        terminate_timeout_s: float = 10.0,
    ) -> "AdaptiveRuntimeClient":
        """Connect to a runtime-owned private session through an opaque socket."""
        endpoint = Path(socket_path).expanduser().resolve()
        for name, value in (
            ("connect_timeout_s", connect_timeout_s),
            ("terminate_timeout_s", terminate_timeout_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        if protocol_observer is not None and not callable(protocol_observer):
            raise TypeError("protocol_observer must be callable or null.")
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        deadline = time.monotonic() + float(connect_timeout_s)
        while True:
            try:
                connection.connect(endpoint.as_posix())
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    connection.close()
                    raise TimeoutError(
                        f"Adaptive runtime socket was not ready: {endpoint}"
                    ) from None
                time.sleep(0.05)
        instance = cls.__new__(cls)
        instance.command = ["adaptive-session-socket", endpoint.as_posix()]
        instance.output_hook = output_hook
        instance.protocol_observer = protocol_observer
        instance.terminate_timeout_s = float(terminate_timeout_s)
        instance._observation_sequence = 0
        instance._closed = False
        instance._finalized = False
        instance._socket = connection
        instance.process = None
        instance.input = connection.makefile("w", encoding="utf-8", buffering=1)
        instance.output = connection.makefile("r", encoding="utf-8")
        return instance

    def _observe(
        self,
        direction: AdaptiveProtocolDirection,
        payload: Mapping[str, object],
    ) -> None:
        """Deliver one ordered immutable protocol observation when configured."""
        observer = getattr(self, "protocol_observer", None)
        sequence = int(getattr(self, "_observation_sequence", 0))
        observation = AdaptiveProtocolObservation(
            sequence_id=sequence,
            direction=direction,
            payload=payload,
        )
        self._observation_sequence = sequence + 1
        if observer is not None:
            observer(observation)

    def _write_request(self, payload: Mapping[str, object]) -> None:
        """Validate, observe, and flush one estimator-visible request."""
        if self.input is None:
            raise RuntimeError("Adaptive runtime input is closed.")
        validate_truth_free_estimator_input(payload, path="adaptive.request")
        self._observe(AdaptiveProtocolDirection.REQUEST, payload)
        self.input.write(json.dumps(dict(payload), allow_nan=False) + "\n")
        self.input.flush()

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
            payload = ADAPTIVE_EVENT_FRAMING.parse(line)
            validate_truth_free_estimator_input(payload, path="adaptive.event")
            self._observe(AdaptiveProtocolDirection.EVENT, payload)
            return payload
        return_code = None if self.process is None else self.process.poll()
        raise RuntimeError(
            "Shared adaptive runtime closed before its next event; "
            f"return_code={return_code}."
        )

    def read_session_event(self) -> AdaptiveSessionEvent:
        """Read and parse the next event through the typed protocol API."""
        return parse_adaptive_event(self.read_event())

    def read_ready_event(self) -> AdaptiveReadyEvent:
        """Read and require the initial typed adaptive handshake event."""
        event = self.read_session_event()
        if not isinstance(event, AdaptiveReadyEvent):
            raise RuntimeError(
                "Shared adaptive runtime did not emit a ready handshake."
            )
        return event

    def handshake(self) -> AdaptiveReadyEvent:
        """Read the typed initial handshake under the concise public API."""
        return self.read_ready_event()

    def request(self, payload: Mapping[str, object]) -> dict[str, Any]:
        """Send one causal controller decision and wait for its response."""
        self._write_request(payload)
        return self.read_event()

    def request_step(self, request: AdaptiveStepRequest) -> AdaptiveRecordEvent:
        """Execute one typed acquisition request and parse its record event."""
        if not isinstance(request, AdaptiveStepRequest):
            raise TypeError("request must be an AdaptiveStepRequest.")
        event = parse_adaptive_event(self.request(request.to_payload()))
        if not isinstance(event, AdaptiveRecordEvent):
            raise RuntimeError("Shared adaptive runtime did not emit a record event.")
        return event

    def acquire(self, request: AdaptiveStepRequest) -> AdaptiveRecordEvent:
        """Acquire one typed observation through the concise public API."""
        return self.request_step(request)

    def request_refinement(
        self,
        request: AdaptiveRefineRequest,
    ) -> AdaptiveCandidatesEvent:
        """Execute one typed refinement request and parse its candidate event."""
        if not isinstance(request, AdaptiveRefineRequest):
            raise TypeError("request must be an AdaptiveRefineRequest.")
        event = parse_adaptive_event(self.request(request.to_payload()))
        if not isinstance(event, AdaptiveCandidatesEvent):
            raise RuntimeError(
                "Shared adaptive runtime did not emit a candidates event."
            )
        return event

    def refine_candidates(
        self,
        request: AdaptiveRefineRequest,
    ) -> AdaptiveCandidatesEvent:
        """Refine runtime-owned candidates through the concise public API."""
        return self.request_refinement(request)

    def request_cui_overlay(self, *, include_truth: bool) -> dict[str, object]:
        """Request only a truth-free CUI overlay for estimator-owned rendering."""
        if self.input is None:
            raise RuntimeError("Adaptive runtime input is closed.")
        if not isinstance(include_truth, bool):
            raise TypeError("include_truth must be a boolean.")
        if include_truth:
            raise ValueError(
                "AdaptiveRuntimeClient is estimator-facing and cannot request "
                "realized truth."
            )
        payload = {
            "type": "cui_overlay",
            "include_truth": False,
        }
        self.input.write(json.dumps(payload, allow_nan=False) + "\n")
        self.input.flush()
        assert self.output is not None
        for raw_line in self.output:
            line = raw_line.rstrip("\n")
            if line.startswith(ADAPTIVE_CUI_OVERLAY_PREFIX):
                payload = ADAPTIVE_CUI_OVERLAY_FRAMING.parse(line)
                return _parse_truth_free_cui_overlay_payload(payload)
            if line.startswith(ADAPTIVE_EVENT_PREFIX):
                raise RuntimeError(
                    "Shared runtime emitted an estimator event during a CUI "
                    "overlay request."
                )
            self.output_hook(line)
        return_code = None if self.process is None else self.process.poll()
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
        if self.process is not None:
            return_code = self.process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, self.command)
        self._finalized = True
        self._closed = True
        if self.output is not None:
            self.output.close()
            self.output = None
        active_socket = getattr(self, "_socket", None)
        if active_socket is not None:
            active_socket.close()
            self._socket = None
        return event

    def finalize_event(self) -> AdaptivePublishedEvent:
        """Finalize the runtime and parse its typed publication event."""
        event = parse_adaptive_event(self.finalize())
        if not isinstance(event, AdaptivePublishedEvent):
            raise RuntimeError(
                "Shared adaptive runtime did not emit a published event."
            )
        return event

    def finalize_log(self) -> AdaptivePublishedEvent:
        """Finalize and return the typed published-log event."""
        return self.finalize_event()

    def terminate(self, timeout: float | None = None) -> None:
        """End an incomplete session within a bounded graceful shutdown window."""
        if bool(getattr(self, "_closed", False)):
            return
        configured = float(getattr(self, "terminate_timeout_s", 10.0))
        timeout_s = configured if timeout is None else timeout
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not np.isfinite(float(timeout_s))
            or float(timeout_s) <= 0.0
        ):
            raise ValueError("terminate timeout must be finite and positive.")
        timeout_value = float(timeout_s)
        if self.process is None:
            try:
                event = self.request({"type": "abort"})
                if event != {"type": "aborted"}:
                    raise RuntimeError(
                        "Adaptive socket did not acknowledge session abort."
                    )
            except (BrokenPipeError, OSError, RuntimeError, ValueError):
                pass
        elif self.process.poll() is None:
            try:
                self._write_request({"type": "abort"})
            except (BrokenPipeError, OSError, RuntimeError, ValueError):
                pass
            if self.input is not None:
                try:
                    self.input.close()
                finally:
                    self.input = None
            try:
                self.process.wait(timeout=timeout_value)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=min(timeout_value, 2.0))
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if self.input is not None:
            self.input.close()
            self.input = None
        if self.output is not None:
            self.output.close()
            self.output = None
        active_socket = getattr(self, "_socket", None)
        if active_socket is not None:
            active_socket.close()
            self._socket = None
        self._closed = True

    def close(self) -> None:
        """Close this client using its configured bounded termination policy."""
        self.terminate()

    def __enter__(self) -> "AdaptiveRuntimeClient":
        """Return this client for deterministic session ownership."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Terminate any session that was not already finalized."""
        del exc_type, exc, traceback
        self.close()

    def abort(self) -> None:
        """Best-effort close of an incomplete acquisition session."""
        self.terminate()


__all__ = [
    "AdaptiveAbortedEvent",
    "AdaptiveBootstrap",
    "AdaptiveCandidateSnapshot",
    "AdaptiveCandidatesEvent",
    "AdaptivePublishedEvent",
    "AdaptiveProtocolDirection",
    "AdaptiveProtocolObservation",
    "AdaptiveReadyEvent",
    "AdaptiveRecordEvent",
    "AdaptiveRefineRequest",
    "AdaptiveResumePrefix",
    "AdaptiveRuntimeClient",
    "AdaptiveSessionEvent",
    "AdaptiveStepRequest",
    "adaptive_step_request",
    "candidate_index_for_pose",
    "parse_adaptive_event",
    "parse_adaptive_record",
    "parse_adaptive_resume_prefix",
    "parse_candidate_snapshot",
    "parse_run_context",
]
