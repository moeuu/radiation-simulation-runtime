"""Tests for typed adaptive-session protocol data-transfer objects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from runtime.adaptive_client import (
    AdaptiveRuntimeClient,
    adaptive_step_request,
    parse_candidate_snapshot,
    parse_run_context,
)
from runtime.adaptive_protocol import (
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
        "candidate_poses_xyz": [[0.5, 0.5, 0.5], [1.0, 0.5, 0.5]],
        "travel_costs": [0.0, 0.5],
        "allowed_pair_ids": list(range(64)),
        "current_pair_id": 0,
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


def test_record_codec_preserves_the_existing_wire_payload() -> None:
    """The shared record codec must round-trip exact integer histograms."""
    record = _record()
    payload = measurement_record_to_payload(record)
    parsed = measurement_record_from_payload(payload)

    assert measurement_record_to_payload(parsed) == payload
    assert parsed.spectrum_counts.dtype == np.int64
    np.testing.assert_array_equal(parsed.spectrum_counts, record.spectrum_counts)


def test_candidate_snapshot_is_frozen_and_wire_compatible() -> None:
    """Candidate DTOs must own immutable tuples and emit legacy JSON lists."""
    payload = _candidate_payload()
    snapshot = AdaptiveCandidateSnapshot.from_payload(payload)
    payload["candidate_poses_xyz"][0][0] = 99.0

    assert snapshot.to_payload() == _candidate_payload()
    assert isinstance(snapshot.candidate_poses_xyz, tuple)
    with pytest.raises(FrozenInstanceError):
        snapshot.current_pair_id = 1
    assert parse_candidate_snapshot(_candidate_payload()) == _candidate_payload()


def test_public_dto_constructors_reject_coercible_invalid_values() -> None:
    """Direct DTO construction must never hide invalid values by coercion."""
    with pytest.raises(TypeError, match="candidate_index"):
        AdaptiveStepRequest(
            candidate_index="1",
            fe_orientation_index=0,
            pb_orientation_index=0,
            dwell_time_s=1.0,
            station_id=0,
            station_complete=True,
        )
    with pytest.raises(TypeError, match="station_complete"):
        AdaptiveStepRequest(
            candidate_index=1,
            fe_orientation_index=0,
            pb_orientation_index=0,
            dwell_time_s=1.0,
            station_id=0,
            station_complete=1,
        )
    with pytest.raises(TypeError, match="allowed_pair_ids"):
        AdaptiveCandidateSnapshot(
            candidate_poses_xyz=((0.5, 0.5, 0.5),),
            travel_costs=(0.0,),
            allowed_pair_ids=tuple(float(value) for value in range(64)),
            current_pair_id=0,
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


def test_resume_ready_event_round_trips_completed_prefix() -> None:
    """A resumed ready event must preserve the schema-v2 durable prefix."""
    record_payload = measurement_record_to_payload(_record())
    payload = {
        "type": "ready",
        "schema_version": 2,
        "context": _context_payload(),
        "candidates": _candidate_payload(),
        "resume": {
            "record_count": 1,
            "records": [record_payload],
            "next_station_id": 1,
        },
    }

    event = parse_adaptive_event(payload)

    assert isinstance(event, AdaptiveReadyEvent)
    assert event.resume is not None
    assert event.resume.next_station_id == 1
    assert event.to_payload() == payload


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
