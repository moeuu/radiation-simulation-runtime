"""Typed data-transfer objects for the adaptive-session JSON protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from runtime.cli_events import CLIJSONEventFraming
from runtime.measurement_log import MeasurementLogRecord
from runtime.records import (
    RunContext,
    measurement_record_from_payload,
    measurement_record_to_payload,
    validate_truth_free_estimator_input,
)
from runtime.shield_timing import (
    shield_program_actuation_time_s,
)

ADAPTIVE_EVENT_PREFIX = "adaptive-session "
ADAPTIVE_CUI_OVERLAY_PREFIX = "adaptive-cui-overlay "
ADAPTIVE_EVENT_FRAMING = CLIJSONEventFraming(ADAPTIVE_EVENT_PREFIX)


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

    current_pose_xyz: tuple[float, float, float]
    candidate_poses_xyz: tuple[tuple[float, float, float], ...]
    travel_costs: tuple[float, ...]
    allowed_pair_ids: tuple[int, ...]
    current_pair_id: int
    shield_angular_speed_rad_s: float
    horizontal_travel_times_s: tuple[float, ...]
    mast_vertical_times_s: tuple[float, ...]
    settling_times_s: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate and normalize every candidate snapshot field."""
        raw_current_pose = self.current_pose_xyz
        if (
            not isinstance(raw_current_pose, (list, tuple))
            or len(raw_current_pose) != 3
        ):
            raise ValueError("Runtime current pose must contain exactly three values.")
        current_pose: list[float] = []
        for value in raw_current_pose:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError("Runtime current pose must contain numbers.")
            parsed = float(value)
            if not np.isfinite(parsed):
                raise ValueError("Runtime current pose must be finite.")
            current_pose.append(parsed)
        normalized_current_pose = (
            current_pose[0],
            current_pose[1],
            current_pose[2],
        )
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
        if len(set(poses)) != len(poses):
            raise ValueError("Runtime candidate poses must be unique.")
        raw_costs = self.travel_costs
        if not isinstance(raw_costs, (list, tuple)) or len(raw_costs) != len(poses):
            raise ValueError("Runtime travel costs must align with candidate poses.")
        costs = tuple(
            _finite_nonnegative_number(value, name="travel_costs item")
            for value in raw_costs
        )
        raw_components = (
            self.horizontal_travel_times_s,
            self.mast_vertical_times_s,
            self.settling_times_s,
        )
        component_names = (
            "horizontal_travel_times_s",
            "mast_vertical_times_s",
            "settling_times_s",
        )
        normalized_components: list[tuple[float, ...]] = []
        for component_name, raw_component in zip(
            component_names,
            raw_components,
            strict=True,
        ):
            if (
                not isinstance(raw_component, (list, tuple))
                or len(raw_component) != len(poses)
            ):
                raise ValueError(
                    f"Runtime {component_name} must align with candidate poses."
                )
            normalized_components.append(
                tuple(
                    _finite_nonnegative_number(
                        value,
                        name=f"{component_name} item",
                    )
                    for value in raw_component
                )
            )
        horizontal_times, mast_times, settling_times = normalized_components
        component_totals = np.asarray(horizontal_times, dtype=np.float64)
        component_totals += np.asarray(mast_times, dtype=np.float64)
        component_totals += np.asarray(settling_times, dtype=np.float64)
        if not np.array_equal(
            component_totals,
            np.asarray(costs, dtype=np.float64),
        ):
            raise ValueError(
                "Runtime motion-time components must sum to travel_costs."
            )
        current_rows = [
            index
            for index, pose in enumerate(poses)
            if pose == normalized_current_pose
        ]
        if len(current_rows) != 1:
            raise ValueError(
                "Runtime candidates must contain current_pose_xyz exactly once."
            )
        current_index = current_rows[0]
        if any(
            value != 0.0
            for value in (
                costs[current_index],
                horizontal_times[current_index],
                mast_times[current_index],
                settling_times[current_index],
            )
        ):
            raise ValueError(
                "Runtime current_pose_xyz candidate must be the exact zero-motion row."
            )
        if any(
            index != current_index and costs[index] == 0.0
            for index in range(len(costs))
        ):
            raise ValueError(
                "Runtime candidates must contain only one zero-motion row."
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
        angular_speed = self.shield_angular_speed_rad_s
        if isinstance(angular_speed, (bool, np.bool_)) or not isinstance(
            angular_speed,
            Real,
        ):
            raise TypeError(
                "shield_angular_speed_rad_s must be a finite number."
            )
        parsed_angular_speed = float(angular_speed)
        if not np.isfinite(parsed_angular_speed) or parsed_angular_speed <= 0.0:
            raise ValueError(
                "shield_angular_speed_rad_s must be finite and positive."
            )
        object.__setattr__(self, "current_pose_xyz", normalized_current_pose)
        object.__setattr__(self, "candidate_poses_xyz", tuple(poses))
        object.__setattr__(self, "travel_costs", costs)
        object.__setattr__(
            self,
            "horizontal_travel_times_s",
            tuple(horizontal_times),
        )
        object.__setattr__(
            self,
            "mast_vertical_times_s",
            tuple(mast_times),
        )
        object.__setattr__(self, "settling_times_s", tuple(settling_times))
        object.__setattr__(self, "allowed_pair_ids", pair_ids)
        object.__setattr__(self, "current_pair_id", current_pair_id)
        object.__setattr__(
            self,
            "shield_angular_speed_rad_s",
            parsed_angular_speed,
        )

    def quote_shield_program_time_s(self, pair_ids: Sequence[int]) -> float:
        """Quote exact sequential shield actuation time from current state."""
        return shield_program_actuation_time_s(
            self.current_pair_id,
            pair_ids,
            shield_angular_speed_rad_s=self.shield_angular_speed_rad_s,
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize this snapshot to the current strict wire schema."""
        return {
            "current_pose_xyz": list(self.current_pose_xyz),
            "candidate_poses_xyz": [list(pose) for pose in self.candidate_poses_xyz],
            "travel_costs": list(self.travel_costs),
            "allowed_pair_ids": list(self.allowed_pair_ids),
            "current_pair_id": self.current_pair_id,
            "shield_angular_speed_rad_s": self.shield_angular_speed_rad_s,
            "horizontal_travel_times_s": list(
                self.horizontal_travel_times_s
            ),
            "mast_vertical_times_s": list(self.mast_vertical_times_s),
            "settling_times_s": list(self.settling_times_s),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "AdaptiveCandidateSnapshot":
        """Parse one runtime-owned reachable candidate snapshot."""
        if not isinstance(payload, Mapping):
            raise TypeError("Adaptive candidates must be an object.")
        validate_truth_free_estimator_input(payload, path="adaptive.candidates")
        component_fields = {
            "current_pose_xyz",
            "candidate_poses_xyz",
            "travel_costs",
            "allowed_pair_ids",
            "current_pair_id",
            "shield_angular_speed_rad_s",
            "horizontal_travel_times_s",
            "mast_vertical_times_s",
            "settling_times_s",
        }
        _strict_fields(
            payload,
            component_fields,
            name="adaptive candidates",
        )
        return cls(
            current_pose_xyz=payload["current_pose_xyz"],
            candidate_poses_xyz=payload["candidate_poses_xyz"],
            travel_costs=payload["travel_costs"],
            allowed_pair_ids=payload["allowed_pair_ids"],
            current_pair_id=payload["current_pair_id"],
            shield_angular_speed_rad_s=payload["shield_angular_speed_rad_s"],
            horizontal_travel_times_s=payload["horizontal_travel_times_s"],
            mast_vertical_times_s=payload["mast_vertical_times_s"],
            settling_times_s=payload["settling_times_s"],
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
class AdaptiveStepRequest:
    """Represent one typed adaptive observation request."""

    action_id: int
    candidate_index: int
    fe_orientation_index: int
    pb_orientation_index: int
    dwell_time_s: float
    station_id: int
    station_complete: bool

    def __post_init__(self) -> None:
        """Validate and normalize every typed step request field."""
        action_id = _exact_nonnegative_integer(
            self.action_id,
            name="action_id",
        )
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
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "candidate_index", candidate_index)
        object.__setattr__(self, "fe_orientation_index", fe_index)
        object.__setattr__(self, "pb_orientation_index", pb_index)
        object.__setattr__(self, "dwell_time_s", dwell_time_s)
        object.__setattr__(self, "station_id", station_id)

    def to_payload(self) -> dict[str, object]:
        """Serialize this request to the current strict wire fields."""
        return {
            "type": "step",
            "action_id": self.action_id,
            "candidate_index": self.candidate_index,
            "fe_orientation_index": self.fe_orientation_index,
            "pb_orientation_index": self.pb_orientation_index,
            "dwell_time_s": self.dwell_time_s,
            "station_id": self.station_id,
            "station_complete": self.station_complete,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AdaptiveStepRequest":
        """Parse one exact current adaptive observation request."""
        if not isinstance(payload, Mapping):
            raise TypeError("Adaptive step request must be an object.")
        _strict_fields(
            payload,
            {
                "type",
                "action_id",
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
            action_id=payload["action_id"],
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
    """Represent the sole fresh adaptive-session handshake."""

    schema_version: int
    context: RunContext
    candidates: AdaptiveCandidateSnapshot
    bootstrap: AdaptiveBootstrap

    def __post_init__(self) -> None:
        """Require one internally consistent fresh-run handshake."""
        schema_version = _exact_nonnegative_integer(
            self.schema_version,
            name="schema_version",
        )
        if schema_version != 1:
            raise ValueError("Adaptive ready events support fresh schema version 1 only.")
        if not isinstance(self.context, RunContext):
            raise TypeError("context must be a RunContext.")
        if not isinstance(self.candidates, AdaptiveCandidateSnapshot):
            raise TypeError("candidates must be an AdaptiveCandidateSnapshot.")
        if not isinstance(self.bootstrap, AdaptiveBootstrap):
            raise TypeError("bootstrap must be an AdaptiveBootstrap.")
        if self.bootstrap.candidate_index >= len(
            self.candidates.candidate_poses_xyz
        ):
            raise ValueError(
                "Adaptive bootstrap candidate_index lies outside candidates."
            )
        bootstrap_pair_id = (
            self.bootstrap.fe_orientation_index * 8
            + self.bootstrap.pb_orientation_index
        )
        if (
            self.candidates.current_pair_id != bootstrap_pair_id
            or self.candidates.candidate_poses_xyz[
                self.bootstrap.candidate_index
            ]
            != self.candidates.current_pose_xyz
        ):
            raise ValueError(
                "Adaptive bootstrap must identify the candidate snapshot's "
                "current pose and shield pair."
            )
        object.__setattr__(self, "schema_version", schema_version)

    def to_payload(self) -> dict[str, object]:
        """Serialize the sole fresh-run handshake schema."""
        return {
            "type": "ready",
            "schema_version": self.schema_version,
            "context": self.context.to_payload(),
            "candidates": self.candidates.to_payload(),
            "bootstrap": self.bootstrap.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "AdaptiveReadyEvent":
        """Parse the sole fresh-run adaptive-session handshake."""
        if not isinstance(payload, Mapping):
            raise TypeError("Adaptive ready event must be an object.")
        _strict_fields(
            payload,
            {"type", "schema_version", "context", "candidates", "bootstrap"},
            name="adaptive ready event",
        )
        if payload.get("type") != "ready":
            raise ValueError("Adaptive ready event type must be 'ready'.")
        schema_version = payload["schema_version"]
        if type(schema_version) is not int or schema_version != 1:
            raise ValueError("Adaptive ready events support fresh schema version 1 only.")
        return cls(
            schema_version=schema_version,
            context=RunContext.from_payload(payload["context"]),
            candidates=AdaptiveCandidateSnapshot.from_payload(payload["candidates"]),
            bootstrap=AdaptiveBootstrap.from_payload(payload["bootstrap"]),
        )


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
        record_pair_id = (
            int(self.record.fe_orientation_index) * 8
            + int(self.record.pb_orientation_index)
        )
        if (
            self.candidates.current_pair_id != record_pair_id
            or self.candidates.current_pose_xyz
            != tuple(self.record.detector_pose_xyz)
        ):
            raise ValueError(
                "Adaptive record candidates disagree with the durable current "
                "pose or shield pair."
            )

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
    "AdaptiveSessionEvent",
    "AdaptiveStepRequest",
    "parse_adaptive_event",
]
