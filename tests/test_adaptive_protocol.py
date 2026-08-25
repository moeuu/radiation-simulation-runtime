"""Tests for typed adaptive-session protocol data-transfer objects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import io
import math
from typing import Any

import numpy as np
import pytest

import runtime.adaptive_client as adaptive_client_module
from runtime.adaptive_client import (
    AdaptiveProtocolDirection,
    AdaptiveProtocolObservation,
    AdaptiveRuntimeClient,
    adaptive_step_request,
    candidate_index_for_pose,
    parse_run_context,
)
from runtime.adaptive_protocol import (
    ADAPTIVE_EVENT_PREFIX,
    AdaptiveAbortedEvent,
    AdaptiveBootstrap,
    AdaptiveCandidateSnapshot,
    AdaptiveCandidatesEvent,
    AdaptivePublishedEvent,
    AdaptiveReadyEvent,
    AdaptiveRecordEvent,
    AdaptiveRefineRequest,
    AdaptiveStepRequest,
    parse_adaptive_event,
)
from runtime.measurement_log import MeasurementLogRecord
from runtime.records import (
    RunContext,
    measurement_record_from_payload,
    measurement_record_to_payload,
)


def _context_payload() -> dict[str, object]:
    """Return one complete truth-free adaptive context payload."""
    return {
        "repository_commit": "a" * 40,
        "runtime_config": {"transport": {"threads": 4, "modes": ["full"]}},
        "environment": {"size_x": 3.0, "detector_position": [0.5, 0.5, 0.5]},
        "sim_backend": "test",
        "spectrum_count_method": "joint_full_spectrum_generative",
        "isotopes": ["Cs-137"],
        "obstacle_layout_path": None,
        "source_rate_model": "detector_cps_1m",
        "metadata": {"campaign": {"name": "typed-protocol"}},
        "run_id": "adaptive-protocol-test",
        "source_rate_semantics": {"unit": "counts_per_second"},
        "forward_model_manifest": {"schema_version": 1},
        "runtime_config_sha256": "b" * 64,
        "schema_version": 2,
    }


def _candidate_payload() -> dict[str, object]:
    """Return one valid candidate snapshot wire payload."""
    return {
        "current_pose_xyz": [0.5, 0.5, 0.5],
        "candidate_poses_xyz": [[0.5, 0.5, 0.5], [1.0, 0.5, 0.5]],
        "travel_costs": [0.0, 0.5],
        "allowed_pair_ids": list(range(64)),
        "current_pair_id": 0,
        "shield_angular_speed_rad_s": math.pi / 4.0,
        "horizontal_travel_times_s": [0.0, 0.5],
        "mast_vertical_times_s": [0.0, 0.0],
        "settling_times_s": [0.0, 0.0],
    }


def _record() -> MeasurementLogRecord:
    """Return one valid durable adaptive measurement record."""
    return MeasurementLogRecord(
        step_id=0,
        action_id=0,
        station_id=0,
        detector_pose_xyz=(0.5, 0.5, 0.5),
        detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fe_orientation_index=0,
        pb_orientation_index=0,
        live_time_s=1.0,
        travel_time_s=0.0,
        shield_actuation_time_s=0.0,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0]),
        spectrum_counts=np.asarray([3, 4], dtype=np.int64),
        metadata={
            "full_spectrum_contract_hash_sha256": "c" * 64,
            "station_complete": True,
        },
    )


def test_run_context_payload_is_deeply_immutable_and_round_trips() -> None:
    """Parsed context state must not alias mutable wire dictionaries."""
    payload = _context_payload()
    context = RunContext.from_payload(payload)
    payload["runtime_config"]["transport"]["modes"][0] = "changed"

    assert context.to_payload() == _context_payload()
    with pytest.raises(TypeError):
        context.runtime_config["new"] = "value"
    with pytest.raises(TypeError):
        context.runtime_config["transport"]["threads"] = 8
    with pytest.raises(FrozenInstanceError):
        context.run_id = "changed"


def test_run_context_rejects_numpy_scalar_coercion() -> None:
    """The live context boundary must accept only already-native JSON values."""
    payload = _context_payload()
    payload["runtime_config"]["transport"]["threads"] = np.int64(4)

    with pytest.raises(TypeError, match="strict JSON"):
        RunContext.from_payload(payload)


def test_record_codec_preserves_the_existing_wire_payload() -> None:
    """The shared record codec must round-trip exact integer histograms."""
    record = _record()
    payload = measurement_record_to_payload(record)
    parsed = measurement_record_from_payload(payload)

    assert measurement_record_to_payload(parsed) == payload
    assert parsed.spectrum_counts.dtype == np.int64
    np.testing.assert_array_equal(parsed.spectrum_counts, record.spectrum_counts)


@pytest.mark.parametrize(
    "energy_axis",
    (
        [0.0, "1.0", 2.0],
        [0.0, np.float64(1.0), 2.0],
        (0.0, 1.0, 2.0),
        [0.0, True, 2.0],
    ),
)
def test_record_codec_rejects_non_json_native_energy_axes(
    energy_axis: object,
) -> None:
    """The wire codec must reject energy-axis coercion before NumPy sees it."""
    payload = measurement_record_to_payload(_record())
    payload["energy_bin_edges_keV"] = energy_axis

    with pytest.raises(TypeError, match="energy_bin_edges_keV"):
        measurement_record_from_payload(payload)


@pytest.mark.parametrize(
    "spectrum_counts",
    (
        [3, "4"],
        [np.int64(3), 4],
        (3, 4),
        [True, 4],
    ),
)
def test_record_codec_rejects_non_json_native_spectrum_counts(
    spectrum_counts: object,
) -> None:
    """Adjacent histogram parsing must also reject implicit integer coercion."""
    payload = measurement_record_to_payload(_record())
    payload["spectrum_counts"] = spectrum_counts

    with pytest.raises(TypeError, match="spectrum_counts"):
        measurement_record_from_payload(payload)


def test_candidate_snapshot_is_frozen_and_wire_compatible() -> None:
    """Candidate DTOs must own immutable tuples and emit current JSON lists."""
    payload = _candidate_payload()
    snapshot = AdaptiveCandidateSnapshot.from_payload(payload)
    payload["candidate_poses_xyz"][0][0] = 99.0

    assert snapshot.to_payload() == _candidate_payload()
    assert isinstance(snapshot.candidate_poses_xyz, tuple)
    assert not hasattr(snapshot, "to_dict")
    with pytest.raises(FrozenInstanceError):
        snapshot.current_pair_id = 1


def test_candidate_snapshot_rejects_duplicate_poses() -> None:
    """Candidate identity must not depend on first-match duplicate handling."""
    payload = _candidate_payload()
    payload["candidate_poses_xyz"][1] = payload["candidate_poses_xyz"][0]
    payload["travel_costs"][1] = 0.0
    payload["horizontal_travel_times_s"][1] = 0.0

    with pytest.raises(ValueError, match="unique"):
        AdaptiveCandidateSnapshot.from_payload(payload)


def test_candidate_snapshot_quotes_sequential_shield_program_time() -> None:
    """A quote must advance Fe/Pb state through every planned pair."""
    payload = _candidate_payload()
    payload["shield_angular_speed_rad_s"] = 2.0
    snapshot = AdaptiveCandidateSnapshot.from_payload(payload)
    octant_step_angle = math.acos(1.0 / 3.0)

    assert snapshot.quote_shield_program_time_s((1, 9, 9)) == pytest.approx(
        2.0 * octant_step_angle / 2.0
    )
    assert snapshot.quote_shield_program_time_s((0,)) == 0.0


@pytest.mark.parametrize(
    "missing_field",
    [
        "current_pose_xyz",
        "shield_angular_speed_rad_s",
        "horizontal_travel_times_s",
        "mast_vertical_times_s",
        "settling_times_s",
    ],
)
def test_candidate_snapshot_rejects_incomplete_old_payloads(
    missing_field: str,
) -> None:
    """The current protocol must reject every incomplete historical shape."""
    payload = _candidate_payload()
    del payload[missing_field]

    with pytest.raises(ValueError, match=f"missing=.*{missing_field}"):
        AdaptiveCandidateSnapshot.from_payload(payload)


def test_candidate_snapshot_round_trips_explicit_motion_time_components() -> None:
    """Motion components must remain explicit and exactly sum to total costs."""
    payload = _candidate_payload()
    payload["horizontal_travel_times_s"] = [0.0, 0.2]
    payload["mast_vertical_times_s"] = [0.0, 0.2]
    payload["settling_times_s"] = [0.0, 0.1]

    snapshot = AdaptiveCandidateSnapshot.from_payload(payload)

    assert snapshot.to_payload() == payload


def test_candidate_snapshot_rejects_inconsistent_motion_time_components() -> None:
    """Published motion components must reproduce every total exactly."""
    payload = _candidate_payload()
    payload.update(
        {
            "horizontal_travel_times_s": [0.0, 0.2],
            "mast_vertical_times_s": [0.0, 0.2],
            "settling_times_s": [0.0, 0.2],
        }
    )

    with pytest.raises(ValueError, match="sum to travel_costs"):
        AdaptiveCandidateSnapshot.from_payload(payload)


def test_candidate_snapshot_rejects_nonzero_declared_current_pose() -> None:
    """The explicit current pose must identify the exact zero-motion row."""
    payload = _candidate_payload()
    payload["current_pose_xyz"] = [1.0, 0.5, 0.5]

    with pytest.raises(ValueError, match="zero-motion row"):
        AdaptiveCandidateSnapshot.from_payload(payload)


def test_candidate_snapshot_rejects_another_zero_motion_row() -> None:
    """A second zero-cost row would make the runtime motion anchor ambiguous."""
    payload = _candidate_payload()
    payload["travel_costs"] = [0.0, 0.0]
    payload["horizontal_travel_times_s"] = [0.0, 0.0]

    with pytest.raises(ValueError, match="only one zero-motion row"):
        AdaptiveCandidateSnapshot.from_payload(payload)


@pytest.mark.parametrize("mismatch", ["current_pose_xyz", "current_pair_id"])
def test_record_event_rejects_candidate_state_not_anchored_to_record(
    mismatch: str,
) -> None:
    """Every next snapshot must be anchored to the durable response state."""
    candidates = _candidate_payload()
    if mismatch == "current_pose_xyz":
        candidates["current_pose_xyz"] = [1.0, 0.5, 0.5]
        candidates["travel_costs"] = [0.5, 0.0]
        candidates["horizontal_travel_times_s"] = [0.5, 0.0]
    else:
        candidates["current_pair_id"] = 1

    with pytest.raises(ValueError, match="durable current"):
        AdaptiveRecordEvent.from_payload(
            {
                "type": "record",
                "record": measurement_record_to_payload(_record()),
                "candidates": candidates,
            }
        )


@pytest.mark.parametrize("pair_ids", [(), (True,), (64,), ("1",)])
def test_candidate_snapshot_rejects_invalid_shield_programs(
    pair_ids: tuple[object, ...],
) -> None:
    """Shield quotes must reject empty, coercible, and out-of-domain programs."""
    snapshot = AdaptiveCandidateSnapshot.from_payload(_candidate_payload())

    with pytest.raises((TypeError, ValueError)):
        snapshot.quote_shield_program_time_s(pair_ids)


def test_candidate_index_requires_typed_snapshot() -> None:
    """Pose lookup must reject unvalidated raw candidate mappings."""
    payload = _candidate_payload()
    snapshot = AdaptiveCandidateSnapshot.from_payload(payload)

    assert candidate_index_for_pose(snapshot, (1.0, 0.5, 0.5)) == 1
    with pytest.raises(TypeError, match="AdaptiveCandidateSnapshot"):
        candidate_index_for_pose(payload, (0.5, 0.5, 0.5))  # type: ignore[arg-type]


def test_candidate_index_rejects_invalid_target_pose() -> None:
    """Pose lookup must fail clearly before attempting malformed broadcasting."""
    snapshot = AdaptiveCandidateSnapshot.from_payload(_candidate_payload())

    with pytest.raises(ValueError, match="three finite"):
        candidate_index_for_pose(snapshot, (1.0, 0.5))


def test_estimator_facing_exports_exclude_truth_overlay_parser() -> None:
    """Public client exports must expose pose lookup but no truth parser."""
    assert adaptive_client_module.candidate_index_for_pose is candidate_index_for_pose
    assert "parse_cui_overlay_payload" not in adaptive_client_module.__all__
    assert not hasattr(adaptive_client_module, "parse_cui_overlay_payload")


def test_public_dto_constructors_reject_coercible_invalid_values() -> None:
    """Direct DTO construction must never hide invalid values by coercion."""
    with pytest.raises(TypeError, match="candidate_index"):
        AdaptiveStepRequest(
            action_id=0,
            candidate_index="1",
            fe_orientation_index=0,
            pb_orientation_index=0,
            dwell_time_s=1.0,
            station_id=0,
            station_complete=True,
        )
    with pytest.raises(TypeError, match="station_complete"):
        AdaptiveStepRequest(
            action_id=0,
            candidate_index=1,
            fe_orientation_index=0,
            pb_orientation_index=0,
            dwell_time_s=1.0,
            station_id=0,
            station_complete=1,
        )
    with pytest.raises(TypeError, match="allowed_pair_ids"):
        AdaptiveCandidateSnapshot(
            current_pose_xyz=(0.5, 0.5, 0.5),
            candidate_poses_xyz=((0.5, 0.5, 0.5),),
            travel_costs=(0.0,),
            allowed_pair_ids=tuple(float(value) for value in range(64)),
            current_pair_id=0,
            shield_angular_speed_rad_s=math.pi / 4.0,
            horizontal_travel_times_s=(0.0,),
            mast_vertical_times_s=(0.0,),
            settling_times_s=(0.0,),
        )
    with pytest.raises(TypeError, match="fe_orientation_index"):
        AdaptiveBootstrap(
            candidate_index=0,
            fe_orientation_index="0",
            pb_orientation_index=0,
        )
    with pytest.raises(TypeError, match="record_count"):
        AdaptivePublishedEvent(path="/tmp/log", record_count=True)


def test_public_dto_constructors_canonicalize_valid_numeric_values() -> None:
    """Direct DTO construction must normalize accepted numeric scalar types."""
    request = AdaptiveStepRequest(
        action_id=np.int64(0),
        candidate_index=np.int64(1),
        fe_orientation_index=np.int64(2),
        pb_orientation_index=np.int64(3),
        dwell_time_s=np.float64(4.0),
        station_id=np.int64(0),
        station_complete=True,
    )
    refinement = AdaptiveRefineRequest([np.int64(0), np.int64(1)])

    assert request.to_payload() == {
        "type": "step",
        "action_id": 0,
        "candidate_index": 1,
        "fe_orientation_index": 2,
        "pb_orientation_index": 3,
        "dwell_time_s": 4.0,
        "station_id": 0,
        "station_complete": True,
    }
    assert refinement.candidate_indices == (0, 1)


def test_ready_event_round_trips_without_wire_schema_changes() -> None:
    """A fresh ready event must preserve all schema-v1 field names and values."""
    payload = {
        "type": "ready",
        "schema_version": 1,
        "context": _context_payload(),
        "candidates": _candidate_payload(),
        "bootstrap": {
            "candidate_index": 0,
            "fe_orientation_index": 0,
            "pb_orientation_index": 0,
        },
    }

    event = parse_adaptive_event(payload)

    assert isinstance(event, AdaptiveReadyEvent)
    assert event.to_payload() == payload
    assert parse_run_context(payload["context"]).to_payload() == payload["context"]


@pytest.mark.parametrize("schema_version", (True, 1.0, "1", np.int64(1)))
def test_ready_event_requires_an_exact_json_integer_schema(
    schema_version: object,
) -> None:
    """A ready handshake must not coerce schema values equal to integer one."""
    payload = {
        "type": "ready",
        "schema_version": schema_version,
        "context": _context_payload(),
        "candidates": _candidate_payload(),
        "bootstrap": {
            "candidate_index": 0,
            "fe_orientation_index": 0,
            "pb_orientation_index": 0,
        },
    }

    with pytest.raises(ValueError, match="schema version 1"):
        parse_adaptive_event(payload)


def test_ready_event_rejects_retired_resume_schema() -> None:
    """The production protocol must reject the removed resume schema."""
    payload = {
        "type": "ready",
        "schema_version": 2,
        "context": _context_payload(),
        "candidates": _candidate_payload(),
        "resume": {
            "record_count": 1,
            "records": [measurement_record_to_payload(_record())],
            "next_station_id": 1,
        },
    }

    with pytest.raises(ValueError, match="fields disagree|schema version 1"):
        parse_adaptive_event(payload)


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        (
            {
                "type": "record",
                "record": measurement_record_to_payload(_record()),
                "candidates": _candidate_payload(),
            },
            AdaptiveRecordEvent,
        ),
        (
            {"type": "candidates", "candidates": _candidate_payload()},
            AdaptiveCandidatesEvent,
        ),
        (
            {"type": "published", "path": "/tmp/log", "record_count": 1},
            AdaptivePublishedEvent,
        ),
        ({"type": "aborted"}, AdaptiveAbortedEvent),
    ],
)
def test_event_dispatch_round_trips_wire_payloads(
    payload: dict[str, object],
    expected_type: type[object],
) -> None:
    """The typed dispatcher must cover every normal adaptive response event."""
    event = parse_adaptive_event(payload)

    assert isinstance(event, expected_type)
    assert event.to_payload() == payload


def test_typed_client_methods_reuse_raw_transport_without_changing_it() -> None:
    """High-level methods must submit old dictionaries and return typed events."""
    client = AdaptiveRuntimeClient.__new__(AdaptiveRuntimeClient)
    sent: list[dict[str, object]] = []
    candidate_payload = _candidate_payload()
    responses = iter(
        [
            {
                "type": "record",
                "record": measurement_record_to_payload(_record()),
                "candidates": candidate_payload,
            },
            {"type": "candidates", "candidates": candidate_payload},
        ]
    )

    def request(payload: dict[str, object]) -> dict[str, Any]:
        """Capture one raw payload and return the next protocol response."""
        sent.append(payload)
        return next(responses)

    client.request = request
    step = AdaptiveStepRequest.from_payload(
        adaptive_step_request(
            action_id=0,
            candidate_index=1,
            fe_orientation_index=2,
            pb_orientation_index=3,
            dwell_time_s=4.0,
            station_id=0,
            station_complete=True,
        )
    )
    record_event = client.request_step(step)
    candidates_event = client.request_refinement(
        AdaptiveRefineRequest.from_indices([0, 1])
    )

    assert isinstance(record_event, AdaptiveRecordEvent)
    assert isinstance(candidates_event, AdaptiveCandidatesEvent)
    assert sent == [
        adaptive_step_request(
            action_id=0,
            candidate_index=1,
            fe_orientation_index=2,
            pb_orientation_index=3,
            dwell_time_s=4.0,
            station_id=0,
            station_complete=True,
        ),
        {"type": "refine", "candidate_indices": [0, 1]},
    ]


def test_typed_client_ready_and_finalize_methods_require_expected_events() -> None:
    """Typed lifecycle helpers must reject unexpected raw response kinds."""
    client = AdaptiveRuntimeClient.__new__(AdaptiveRuntimeClient)
    ready_payload = {
        "type": "ready",
        "schema_version": 1,
        "context": _context_payload(),
        "candidates": _candidate_payload(),
        "bootstrap": {
            "candidate_index": 0,
            "fe_orientation_index": 0,
            "pb_orientation_index": 0,
        },
    }

    def read_event() -> dict[str, Any]:
        """Return one raw ready handshake."""
        return ready_payload

    def finalize() -> dict[str, Any]:
        """Return one raw publication acknowledgement."""
        return {"type": "published", "path": "/tmp/log", "record_count": 1}

    client.read_event = read_event
    assert isinstance(client.read_ready_event(), AdaptiveReadyEvent)
    client.finalize = finalize
    assert client.finalize_event() == AdaptivePublishedEvent("/tmp/log", 1)


def test_concise_typed_client_aliases_delegate_without_new_wire_shapes() -> None:
    """Public lifecycle vocabulary must remain a thin typed transport layer."""
    client = AdaptiveRuntimeClient.__new__(AdaptiveRuntimeClient)
    ready = AdaptiveReadyEvent.from_payload(
        {
            "type": "ready",
            "schema_version": 1,
            "context": _context_payload(),
            "candidates": _candidate_payload(),
            "bootstrap": {
                "candidate_index": 0,
                "fe_orientation_index": 0,
                "pb_orientation_index": 0,
            },
        }
    )
    record = AdaptiveRecordEvent.from_payload(
        {
            "type": "record",
            "record": measurement_record_to_payload(_record()),
            "candidates": _candidate_payload(),
        }
    )
    candidates = AdaptiveCandidatesEvent.from_payload(
        {"type": "candidates", "candidates": _candidate_payload()}
    )
    published = AdaptivePublishedEvent("/tmp/log", 1)
    client.read_ready_event = lambda: ready
    client.request_step = lambda request: record
    client.request_refinement = lambda request: candidates
    client.finalize_event = lambda: published

    assert client.handshake() is ready
    assert client.acquire(
        AdaptiveStepRequest(0, 0, 0, 0, 1.0, 0, True)
    ) is record
    assert client.refine_candidates(
        AdaptiveRefineRequest.from_indices([0])
    ) is candidates
    assert client.finalize_log() is published


def test_protocol_observer_receives_ordered_immutable_truth_free_payloads() -> None:
    """Transcript wiring should observe exact request/event pairs in order."""
    client = AdaptiveRuntimeClient.__new__(AdaptiveRuntimeClient)
    observations: list[AdaptiveProtocolObservation] = []
    client.protocol_observer = observations.append
    client._observation_sequence = 0
    client.input = io.StringIO()
    client.output = io.StringIO(
        ADAPTIVE_EVENT_PREFIX
        + '{"type":"published","path":"/tmp/log","record_count":1}\n'
    )
    response = client.request({"type": "finalize"})

    assert response["type"] == "published"
    assert [item.sequence_id for item in observations] == [0, 1]
    assert [item.direction for item in observations] == [
        AdaptiveProtocolDirection.REQUEST,
        AdaptiveProtocolDirection.EVENT,
    ]
    assert observations[0].to_payload()["payload"] == {"type": "finalize"}
    with pytest.raises(TypeError):
        observations[1].payload["type"] = "changed"


def test_context_manager_sends_abort_with_configured_socket_timeout() -> None:
    """Leaving an unfinished client scope must bound the socket abort."""
    class Pipe:
        """Record writes and closure without discarding buffered test data."""

        def __init__(self, lines: tuple[str, ...] = ()) -> None:
            """Initialize an open pipe with optional response lines."""
            self.writes: list[str] = []
            self.lines = lines
            self.closed = False

        def __iter__(self) -> object:
            """Iterate over configured runtime response lines."""
            return iter(self.lines)

        def write(self, value: str) -> int:
            """Record one write and return its character count."""
            self.writes.append(value)
            return len(value)

        def flush(self) -> None:
            """Accept one no-op flush."""

        def close(self) -> None:
            """Mark the pipe closed."""
            self.closed = True

    class Socket:
        """Record bounded socket shutdown state."""

        def __init__(self) -> None:
            """Initialize an open socket and empty timeout record."""
            self.timeouts: list[float] = []
            self.closed = False

        def settimeout(self, timeout: float) -> None:
            """Record the abort timeout."""
            self.timeouts.append(timeout)

        def close(self) -> None:
            """Mark the socket closed."""
            self.closed = True

    client = AdaptiveRuntimeClient.__new__(AdaptiveRuntimeClient)
    input_pipe = Pipe()
    output_pipe = Pipe((f'{ADAPTIVE_EVENT_PREFIX}{{"type":"aborted"}}\n',))
    active_socket = Socket()
    observations: list[AdaptiveProtocolObservation] = []
    client.input = input_pipe
    client.output = output_pipe
    client._socket = active_socket
    client.command = ["runtime"]
    client.protocol_observer = observations.append
    client._observation_sequence = 0
    client.terminate_timeout_s = 0.75
    client._closed = False
    client._finalized = False

    with client:
        pass

    assert input_pipe.writes == ['{"type": "abort"}\n']
    assert input_pipe.closed and output_pipe.closed
    assert active_socket.timeouts == [0.75]
    assert active_socket.closed
    assert observations[0].direction is AdaptiveProtocolDirection.REQUEST
    assert observations[0].payload["type"] == "abort"
    client.close()
    assert active_socket.timeouts == [0.75]


def test_typed_parser_rejects_unknown_or_truth_bearing_event_fields() -> None:
    """Typed events must retain exact-field and estimator-boundary validation."""
    with pytest.raises(ValueError, match="unknown"):
        parse_adaptive_event(
            {
                "type": "published",
                "path": "/tmp/log",
                "record_count": 1,
                "extra": True,
            }
        )
    with pytest.raises(ValueError, match="realized truth"):
        parse_adaptive_event(
            {
                "type": "published",
                "path": "/tmp/log",
                "record_count": 1,
                "source_positions": [[1.0, 1.0, 1.0]],
            }
        )
