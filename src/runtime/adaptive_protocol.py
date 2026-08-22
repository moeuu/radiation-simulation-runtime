"""Typed data-transfer objects for the adaptive-session JSON protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from runtime.measurement_log import MeasurementLogRecord
from runtime.records import (
    RunContext,
    measurement_record_from_payload,
    measurement_record_to_payload,
    validate_truth_free_estimator_input,
)

ADAPTIVE_EVENT_PREFIX = "adaptive-session "
ADAPTIVE_CUI_OVERLAY_PREFIX = "adaptive-cui-overlay "


def _strict_fields(
    payload: Mapping[str, object],
    expected: set[str] | frozenset[str],
    *,
    name: str,
) -> None:
    """Require one exact adaptive protocol object schema."""
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{name} fields disagree with the adaptive protocol: "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )


def _exact_nonnegative_integer(value: object, *, name: str) -> int:
    """Return one nonnegative protocol integer without boolean coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed


def _finite_nonnegative_number(value: object, *, name: str) -> float:
    """Return one finite nonnegative protocol number."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return parsed


@dataclass(frozen=True, slots=True)
class AdaptiveCandidateSnapshot:
    """Expose reachable truth-free poses and runtime-owned motion costs."""

    candidate_poses_xyz: tuple[tuple[float, float, float], ...]
    travel_costs: tuple[float, ...]
    allowed_pair_ids: tuple[int, ...]
    current_pair_id: int

    def __post_init__(self) -> None:
        """Validate and normalize every candidate snapshot field."""
        raw_poses = self.candidate_poses_xyz
        if not isinstance(raw_poses, (list, tuple)) or not raw_poses:
            raise ValueError(
                "Runtime candidate poses must have finite nonempty shape (C, 3)."
            )
        poses: list[tuple[float, float, float]] = []
        for raw_pose in raw_poses:
            if not isinstance(raw_pose, (list, tuple)) or len(raw_pose) != 3:
                raise ValueError(
                    "Runtime candidate poses must have finite nonempty shape "
                    "(C, 3)."
                )
            pose: list[float] = []
            for value in raw_pose:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                    raise TypeError("Runtime candidate poses must contain numbers.")
                parsed = float(value)
                if not np.isfinite(parsed):
                    raise ValueError("Runtime candidate poses must be finite.")
                pose.append(parsed)
            poses.append((pose[0], pose[1], pose[2]))
        raw_costs = self.travel_costs
        if not isinstance(raw_costs, (list, tuple)) or len(raw_costs) != len(poses):
            raise ValueError("Runtime travel costs must align with candidate poses.")
        costs = tuple(
            _finite_nonnegative_number(value, name="travel_costs item")
            for value in raw_costs
        )
        raw_pair_ids = self.allowed_pair_ids
        if not isinstance(raw_pair_ids, (list, tuple)) or len(raw_pair_ids) != 64:
            raise ValueError("Adaptive runtime must expose every Fe/Pb pair 0..63.")
        pair_ids = tuple(
            _exact_nonnegative_integer(value, name="allowed_pair_ids item")
            for value in raw_pair_ids
        )
        if sorted(pair_ids) != list(range(64)):
            raise ValueError("Adaptive runtime must expose every Fe/Pb pair 0..63.")
        current_pair_id = _exact_nonnegative_integer(
            self.current_pair_id,
            name="current_pair_id",
        )
        if current_pair_id > 63:
            raise ValueError("current_pair_id must lie in [0, 63].")
        object.__setattr__(self, "candidate_poses_xyz", tuple(poses))
        object.__setattr__(self, "travel_costs", costs)
        object.__setattr__(self, "allowed_pair_ids", pair_ids)
        object.__setattr__(self, "current_pair_id", current_pair_id)

    def to_payload(self) -> dict[str, object]:
        """Serialize this snapshot to its existing wire representation."""
        return {
            "candidate_poses_xyz": [list(pose) for pose in self.candidate_poses_xyz],
            "travel_costs": list(self.travel_costs),
            "allowed_pair_ids": list(self.allowed_pair_ids),
            "current_pair_id": self.current_pair_id,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the legacy dictionary representation of this snapshot."""
        return self.to_payload()

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "AdaptiveCandidateSnapshot":
        """Parse one runtime-owned reachable candidate snapshot."""
        if not isinstance(payload, Mapping):
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
        return cls(
            candidate_poses_xyz=payload["candidate_poses_xyz"],
            travel_costs=payload["travel_costs"],
            allowed_pair_ids=payload["allowed_pair_ids"],
            current_pair_id=payload["current_pair_id"],
        )


@dataclass(frozen=True, slots=True)
class AdaptiveBootstrap:
    """Describe the runtime-selected initial candidate and shield pair."""

    candidate_index: int
    fe_orientation_index: int
    pb_orientation_index: int

    def __post_init__(self) -> None:
        """Validate and normalize the bootstrap candidate and shield pair."""
        candidate_index = _exact_nonnegative_integer(
            self.candidate_index,
            name="candidate_index",
        )
        fe_index = _exact_nonnegative_integer(
            self.fe_orientation_index,
            name="fe_orientation_index",
        )
        pb_index = _exact_nonnegative_integer(
            self.pb_orientation_index,
            name="pb_orientation_index",
        )
        if fe_index > 7 or pb_index > 7:
            raise ValueError("Bootstrap Fe/Pb orientation indices must lie in [0, 7].")
        object.__setattr__(self, "candidate_index", candidate_index)
        object.__setattr__(self, "fe_orientation_index", fe_index)
        object.__setattr__(self, "pb_orientation_index", pb_index)

    def to_payload(self) -> dict[str, object]:
        """Serialize this bootstrap selection to the existing wire schema."""
        return {
            "candidate_index": self.candidate_index,
            "fe_orientation_index": self.fe_orientation_index,
            "pb_orientation_index": self.pb_orientation_index,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AdaptiveBootstrap":
        """Parse one adaptive bootstrap selection."""
        if not isinstance(payload, Mapping):
            raise TypeError("Adaptive bootstrap must be an object.")
        _strict_fields(
            payload,
            {
                "candidate_index",
                "fe_orientation_index",
                "pb_orientation_index",
            },
            name="adaptive bootstrap",
        )
        return cls(
            payload["candidate_index"],
            payload["fe_orientation_index"],
            payload["pb_orientation_index"],
        )


@dataclass(frozen=True, slots=True)
class AdaptiveResumePrefix:
    """Store the verified completed-station prefix returned during resume."""

    records: tuple[MeasurementLogRecord, ...]
    next_station_id: int

    def __post_init__(self) -> None:
        """Validate and normalize one completed causal resume prefix."""
        if not isinstance(self.records, (list, tuple)) or not self.records:
            raise TypeError("Adaptive resume records must be a nonempty sequence.")
        records = tuple(self.records)
        if any(not isinstance(record, MeasurementLogRecord) for record in records):
            raise TypeError(
                "Adaptive resume records must contain MeasurementLogRecord values."
            )
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
        next_station_id = _exact_nonnegative_integer(
            self.next_station_id,
            name="Adaptive resume next_station_id",
        )
        if next_station_id != station_ids[-1] + 1:
            raise ValueError(
                "Adaptive resume next_station_id disagrees with the durable prefix."
            )
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "next_station_id", next_station_id)

    def to_payload(self) -> dict[str, object]:
        """Serialize this durable prefix to the existing resume wire schema."""
        return {
            "record_count": len(self.records),
            "records": [
                measurement_record_to_payload(record) for record in self.records
            ],
            "next_station_id": self.next_station_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AdaptiveResumePrefix":
        """Parse and validate one truth-free adaptive resume prefix."""
        if not isinstance(payload, Mapping):
            raise TypeError("Adaptive resume prefix must be an object.")
        validate_truth_free_estimator_input(payload, path="adaptive.resume")
        _strict_fields(
            payload,
            {"record_count", "records", "next_station_id"},
            name="adaptive resume prefix",
        )
        raw_count = _exact_nonnegative_integer(
            payload["record_count"],
            name="Adaptive resume record_count",
        )
        raw_records = payload["records"]
        if not isinstance(raw_records, list) or not raw_records:
            raise TypeError("Adaptive resume records must be a nonempty list.")
        records = tuple(
            measurement_record_from_payload(record) for record in raw_records
        )
        if raw_count != len(records):
            raise ValueError("Adaptive resume record_count disagrees with records.")
        return cls(records=records, next_station_id=payload["next_station_id"])


@dataclass(frozen=True, slots=True)
class AdaptiveStepRequest:
    """Represent one typed adaptive observation request."""

    candidate_index: int
    fe_orientation_index: int
    pb_orientation_index: int
    dwell_time_s: float
    station_id: int
    station_complete: bool

    def __post_init__(self) -> None:
        """Validate and normalize every typed step request field."""
        candidate_index = _exact_nonnegative_integer(
            self.candidate_index,
            name="candidate_index",
        )
        fe_index = _exact_nonnegative_integer(
            self.fe_orientation_index,
            name="fe_orientation_index",
        )
        pb_index = _exact_nonnegative_integer(
            self.pb_orientation_index,
            name="pb_orientation_index",
        )
        if fe_index > 7 or pb_index > 7:
            raise ValueError("Step Fe/Pb orientation indices must lie in [0, 7].")
        dwell_time_s = _finite_nonnegative_number(
            self.dwell_time_s,
            name="dwell_time_s",
        )
        if dwell_time_s <= 0.0:
            raise ValueError("dwell_time_s must be positive.")
        station_id = _exact_nonnegative_integer(self.station_id, name="station_id")
        if not isinstance(self.station_complete, bool):
            raise TypeError("station_complete must be a boolean.")
        object.__setattr__(self, "candidate_index", candidate_index)
        object.__setattr__(self, "fe_orientation_index", fe_index)
        object.__setattr__(self, "pb_orientation_index", pb_index)
        object.__setattr__(self, "dwell_time_s", dwell_time_s)
        object.__setattr__(self, "station_id", station_id)

    def to_payload(self) -> dict[str, object]:
        """Serialize this request without changing the schema-v1 wire fields."""
        return {
            "type": "step",
            "candidate_index": self.candidate_index,
            "fe_orientation_index": self.fe_orientation_index,
            "pb_orientation_index": self.pb_orientation_index,
            "dwell_time_s": self.dwell_time_s,
            "station_id": self.station_id,
            "station_complete": self.station_complete,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AdaptiveStepRequest":
        """Parse one exact schema-v1 adaptive observation request."""
        if not isinstance(payload, Mapping):
            raise TypeError("Adaptive step request must be an object.")
        _strict_fields(
            payload,
            {
                "type",
                "candidate_index",
                "fe_orientation_index",
                "pb_orientation_index",
                "dwell_time_s",
                "station_id",
                "station_complete",
            },
            name="adaptive step request",
        )
        if payload["type"] != "step":
            raise ValueError("Adaptive step request type must be 'step'.")
        return cls(
            candidate_index=payload["candidate_index"],
            fe_orientation_index=payload["fe_orientation_index"],
            pb_orientation_index=payload["pb_orientation_index"],
            dwell_time_s=payload["dwell_time_s"],
            station_id=payload["station_id"],
            station_complete=payload["station_complete"],
        )


@dataclass(frozen=True, slots=True)
class AdaptiveRefineRequest:
    """Represent one typed adaptive candidate-refinement request."""

    candidate_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate and normalize candidate refinement seed indices."""
        if not isinstance(self.candidate_indices, (list, tuple)) or not self.candidate_indices:
            raise TypeError("candidate_indices must be a nonempty sequence.")
        indices = tuple(
            _exact_nonnegative_integer(value, name="candidate_indices item")
            for value in self.candidate_indices
        )
        object.__setattr__(self, "candidate_indices", indices)

    def to_payload(self) -> dict[str, object]:
        """Serialize this request to the existing refinement wire fields."""
        return {
            "type": "refine",
            "candidate_indices": list(self.candidate_indices),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AdaptiveRefineRequest":
        """Parse one exact adaptive candidate-refinement request."""
        if not isinstance(payload, Mapping):
            raise TypeError("Adaptive refine request must be an object.")
        _strict_fields(
            payload,
            {"type", "candidate_indices"},
            name="adaptive refine request",
        )
        if payload["type"] != "refine":
            raise ValueError("Adaptive refine request type must be 'refine'.")
        raw_indices = payload["candidate_indices"]
        if not isinstance(raw_indices, list) or not raw_indices:
            raise TypeError("candidate_indices must be a nonempty JSON list.")
        return cls(tuple(raw_indices))

    @classmethod
    def from_indices(
        cls,
        candidate_indices: Sequence[int],
    ) -> "AdaptiveRefineRequest":
        """Build a typed refinement request from an application sequence."""
        return cls.from_payload(
            {"type": "refine", "candidate_indices": list(candidate_indices)}
        )


@dataclass(frozen=True, slots=True)
class AdaptiveReadyEvent:
    """Represent a fresh or resumed adaptive-session handshake."""

    schema_version: int
    context: RunContext
    candidates: AdaptiveCandidateSnapshot
    bootstrap: AdaptiveBootstrap | None = None
    resume: AdaptiveResumePrefix | None = None

    def __post_init__(self) -> None:
        """Require one internally consistent fresh or resumed handshake."""
        schema_version = _exact_nonnegative_integer(
            self.schema_version,
            name="schema_version",
        )
        if not isinstance(self.context, RunContext):
            raise TypeError("context must be a RunContext.")
        if not isinstance(self.candidates, AdaptiveCandidateSnapshot):
            raise TypeError("candidates must be an AdaptiveCandidateSnapshot.")
        if schema_version == 1:
            if not isinstance(self.bootstrap, AdaptiveBootstrap) or self.resume is not None:
                raise ValueError(
                    "Schema-v1 ready events require only a bootstrap selection."
                )
        elif schema_version == 2:
            if not isinstance(self.resume, AdaptiveResumePrefix) or self.bootstrap is not None:
                raise ValueError(
                    "Schema-v2 ready events require only a resume prefix."
                )
        else:
            raise ValueError("Adaptive ready event schema is incompatible.")
        object.__setattr__(self, "schema_version", schema_version)

    def to_payload(self) -> dict[str, object]:
        """Serialize this handshake without changing its versioned wire schema."""
        payload: dict[str, object] = {
            "type": "ready",
            "schema_version": self.schema_version,
            "context": self.context.to_payload(),
            "candidates": self.candidates.to_payload(),
        }
        if self.schema_version == 1 and self.bootstrap is not None:
            payload["bootstrap"] = self.bootstrap.to_payload()
        elif self.schema_version == 2 and self.resume is not None:
            payload["resume"] = self.resume.to_payload()
        else:
            raise AssertionError("Validated ready event state became inconsistent.")
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AdaptiveReadyEvent":
        """Parse one exact versioned adaptive-session handshake."""
        if not isinstance(payload, Mapping):
            raise TypeError("Adaptive ready event must be an object.")
        if payload.get("type") != "ready":
            raise ValueError("Adaptive ready event type must be 'ready'.")
        schema_version = payload.get("schema_version")
        if schema_version == 1:
            _strict_fields(
                payload,
                {"type", "schema_version", "context", "candidates", "bootstrap"},
                name="adaptive ready event",
            )
            return cls(
                schema_version=1,
                context=RunContext.from_payload(payload["context"]),
                candidates=AdaptiveCandidateSnapshot.from_payload(
                    payload["candidates"]
                ),
                bootstrap=AdaptiveBootstrap.from_payload(payload["bootstrap"]),
            )
        if schema_version == 2:
            _strict_fields(
                payload,
                {"type", "schema_version", "context", "candidates", "resume"},
                name="adaptive ready event",
            )
            return cls(
                schema_version=2,
                context=RunContext.from_payload(payload["context"]),
                candidates=AdaptiveCandidateSnapshot.from_payload(
                    payload["candidates"]
                ),
                resume=AdaptiveResumePrefix.from_payload(payload["resume"]),
            )
        raise ValueError("Adaptive ready event schema is incompatible.")


@dataclass(frozen=True, slots=True)
class AdaptiveRecordEvent:
    """Represent one durable measurement and the next candidate snapshot."""

    record: MeasurementLogRecord
    candidates: AdaptiveCandidateSnapshot

    def __post_init__(self) -> None:
        """Require typed immutable record-event members."""
        if not isinstance(self.record, MeasurementLogRecord):
            raise TypeError("record must be a MeasurementLogRecord.")
        if not isinstance(self.candidates, AdaptiveCandidateSnapshot):
            raise TypeError("candidates must be an AdaptiveCandidateSnapshot.")

    def to_payload(self) -> dict[str, object]:
        """Serialize this record event to the existing wire schema."""
        return {
            "type": "record",
            "record": measurement_record_to_payload(self.record),
            "candidates": self.candidates.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AdaptiveRecordEvent":
        """Parse one exact adaptive measurement record event."""
        if not isinstance(payload, Mapping):
            raise TypeError("Adaptive record event must be an object.")
        _strict_fields(
            payload,
            {"type", "record", "candidates"},
            name="adaptive record event",
        )
        if payload["type"] != "record":
            raise ValueError("Adaptive record event type must be 'record'.")
        return cls(
            record=measurement_record_from_payload(payload["record"]),
            candidates=AdaptiveCandidateSnapshot.from_payload(
                payload["candidates"]
            ),
        )


@dataclass(frozen=True, slots=True)
class AdaptiveCandidatesEvent:
    """Represent a runtime-refined candidate snapshot."""

    candidates: AdaptiveCandidateSnapshot

    def __post_init__(self) -> None:
        """Require one typed immutable candidate snapshot."""
        if not isinstance(self.candidates, AdaptiveCandidateSnapshot):
            raise TypeError("candidates must be an AdaptiveCandidateSnapshot.")

    def to_payload(self) -> dict[str, object]:
        """Serialize this candidate event to the existing wire schema."""
        return {"type": "candidates", "candidates": self.candidates.to_payload()}

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "AdaptiveCandidatesEvent":
        """Parse one exact adaptive candidate-refinement event."""
        if not isinstance(payload, Mapping):
            raise TypeError("Adaptive candidates event must be an object.")
        _strict_fields(
            payload,
            {"type", "candidates"},
            name="adaptive candidates event",
        )
        if payload["type"] != "candidates":
            raise ValueError("Adaptive candidates event type must be 'candidates'.")
        return cls(
            candidates=AdaptiveCandidateSnapshot.from_payload(
                payload["candidates"]
            )
        )


@dataclass(frozen=True, slots=True)
class AdaptivePublishedEvent:
    """Represent successful immutable MeasurementLog publication."""

    path: str
    record_count: int

    def __post_init__(self) -> None:
        """Validate and normalize immutable publication metadata."""
        if not isinstance(self.path, str) or not self.path:
            raise TypeError("Adaptive published path must be a nonempty string.")
        object.__setattr__(
            self,
            "record_count",
            _exact_nonnegative_integer(self.record_count, name="record_count"),
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize this publication event to the existing wire schema."""
        return {
            "type": "published",
            "path": self.path,
            "record_count": self.record_count,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AdaptivePublishedEvent":
        """Parse one exact adaptive publication event."""
        if not isinstance(payload, Mapping):
            raise TypeError("Adaptive published event must be an object.")
        _strict_fields(
            payload,
            {"type", "path", "record_count"},
            name="adaptive published event",
        )
        if payload["type"] != "published":
            raise ValueError("Adaptive published event type must be 'published'.")
        return cls(path=payload["path"], record_count=payload["record_count"])


@dataclass(frozen=True, slots=True)
class AdaptiveAbortedEvent:
    """Represent successful closure without MeasurementLog publication."""

    def to_payload(self) -> dict[str, object]:
        """Serialize this abort acknowledgement to the existing wire schema."""
        return {"type": "aborted"}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AdaptiveAbortedEvent":
        """Parse one exact adaptive abort acknowledgement."""
        if not isinstance(payload, Mapping):
            raise TypeError("Adaptive aborted event must be an object.")
        _strict_fields(payload, {"type"}, name="adaptive aborted event")
        if payload["type"] != "aborted":
            raise ValueError("Adaptive aborted event type must be 'aborted'.")
        return cls()


AdaptiveSessionEvent = (
    AdaptiveReadyEvent
    | AdaptiveRecordEvent
    | AdaptiveCandidatesEvent
    | AdaptivePublishedEvent
    | AdaptiveAbortedEvent
)


def parse_adaptive_event(payload: Mapping[str, object]) -> AdaptiveSessionEvent:
    """Parse one truth-free event from the adaptive-session wire protocol."""
    if not isinstance(payload, Mapping):
        raise TypeError("Adaptive runtime event must be an object.")
    validate_truth_free_estimator_input(payload, path="adaptive.event")
    event_type = payload.get("type")
    if event_type == "ready":
        return AdaptiveReadyEvent.from_payload(payload)
    if event_type == "record":
        return AdaptiveRecordEvent.from_payload(payload)
    if event_type == "candidates":
        return AdaptiveCandidatesEvent.from_payload(payload)
    if event_type == "published":
        return AdaptivePublishedEvent.from_payload(payload)
    if event_type == "aborted":
        return AdaptiveAbortedEvent.from_payload(payload)
    raise ValueError(f"Unknown adaptive event type: {event_type!r}.")


__all__ = [
    "ADAPTIVE_CUI_OVERLAY_PREFIX",
    "ADAPTIVE_EVENT_PREFIX",
    "AdaptiveAbortedEvent",
    "AdaptiveBootstrap",
    "AdaptiveCandidateSnapshot",
    "AdaptiveCandidatesEvent",
    "AdaptivePublishedEvent",
    "AdaptiveReadyEvent",
    "AdaptiveRecordEvent",
    "AdaptiveRefineRequest",
    "AdaptiveResumePrefix",
    "AdaptiveSessionEvent",
    "AdaptiveStepRequest",
    "parse_adaptive_event",
]
