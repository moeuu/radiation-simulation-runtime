"""Estimator-neutral client for one opaque adaptive-session socket."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from runtime.adaptive_protocol import (
    ADAPTIVE_CUI_OVERLAY_PREFIX,
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


def parse_run_context(payload: object) -> RunContext:
    """Parse the truth-free runtime handshake context."""
    if not isinstance(payload, dict):
        raise TypeError("Adaptive runtime context must be an object.")
    return RunContext.from_payload(payload)


def adaptive_step_request(
    *,
    action_id: int,
    candidate_index: int,
    fe_orientation_index: int,
    pb_orientation_index: int,
    dwell_time_s: float,
    station_id: int,
    station_complete: bool,
) -> dict[str, object]:
    """Build one exact adaptive observation request without coercion."""
    return AdaptiveStepRequest(
        action_id=action_id,
        candidate_index=candidate_index,
        fe_orientation_index=fe_orientation_index,
        pb_orientation_index=pb_orientation_index,
        dwell_time_s=dwell_time_s,
        station_id=station_id,
        station_complete=station_complete,
    ).to_payload()


def candidate_index_for_pose(
    candidates: AdaptiveCandidateSnapshot,
    pose_xyz: Sequence[float],
) -> int:
    """Locate one exact retained pose in a validated typed snapshot."""
    if not isinstance(candidates, AdaptiveCandidateSnapshot):
        raise TypeError("candidates must be an AdaptiveCandidateSnapshot.")
    poses = np.asarray(candidates.candidate_poses_xyz, dtype=np.float64)
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

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Reject direct construction that could expose a private scenario path."""
        raise TypeError(
            "AdaptiveRuntimeClient must connect to a runtime-owned opaque socket "
            "through AdaptiveRuntimeClient.connect()."
        )

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
        raise RuntimeError("Shared adaptive runtime closed before its next event.")

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

    def finalize(self) -> dict[str, Any]:
        """Finalize the runtime log and close the opaque session socket."""
        event = self.request({"type": "finalize"})
        if self.input is not None:
            self.input.close()
            self.input = None
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

    def _terminate(
        self,
        timeout: float | None,
        *,
        suppress_abort_error: bool,
    ) -> None:
        """Close transport after an abort request, optionally preserving errors."""
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
        active_socket = getattr(self, "_socket", None)
        if active_socket is not None:
            active_socket.settimeout(float(timeout_s))
        abort_error: BaseException | None = None
        try:
            event = self.request({"type": "abort"})
            if event != {"type": "aborted"}:
                raise RuntimeError("Adaptive socket did not acknowledge session abort.")
        except (BrokenPipeError, OSError, RuntimeError, ValueError) as exc:
            abort_error = exc
        finally:
            if self.input is not None:
                self.input.close()
                self.input = None
            if self.output is not None:
                self.output.close()
                self.output = None
            if active_socket is not None:
                active_socket.close()
                self._socket = None
            self._closed = True
        if abort_error is not None and not suppress_abort_error:
            raise abort_error

    def terminate(self, timeout: float | None = None) -> None:
        """Best-effort close within a bounded graceful shutdown window."""
        self._terminate(timeout, suppress_abort_error=True)

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
        """Abort an incomplete acquisition and require runtime acknowledgement."""
        self._terminate(None, suppress_abort_error=False)


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
    "AdaptiveRuntimeClient",
    "AdaptiveSessionEvent",
    "AdaptiveStepRequest",
    "adaptive_step_request",
    "candidate_index_for_pose",
    "parse_adaptive_event",
    "parse_adaptive_record",
    "parse_run_context",
]
