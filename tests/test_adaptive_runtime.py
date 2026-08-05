"""Tests for estimator-neutral interactive runtime acquisition."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from measurement.obstacles import ObstacleGrid
from runtime.adaptive import (
    ADAPTIVE_CUI_OVERLAY_PREFIX,
    ADAPTIVE_EVENT_PREFIX,
    AdaptiveCandidateProvider,
    AdaptiveRuntimeSession,
    _pose_is_clear,
    _validate_private_scene_profile,
    serve_adaptive_session,
)
from runtime.adaptive_client import AdaptiveRuntimeClient
from runtime.measurement_log import MeasurementLogRecord
from runtime.records import RunContext


def _environment() -> dict[str, object]:
    """Return a small truth-free room and detector start pose."""
    return {
        "size_x": 3.0,
        "size_y": 2.0,
        "size_z": 2.0,
        "detector_position": [0.25, 0.25, 0.5],
        "adaptive_measurement": {
            "candidate_count": 32,
            "local_refinement_count": 16,
            "settling_time_s": 0.75,
        },
    }


def test_open_room_candidates_include_runtime_start_and_motion_costs() -> None:
    """Candidate generation should be runtime-owned and start-pose causal."""
    provider = AdaptiveCandidateProvider(_environment(), None)

    snapshot = provider.snapshot(provider.initial_pose, current_pair_id=19)

    assert snapshot.candidate_poses_xyz[0] == (0.25, 0.25, 0.5)
    assert snapshot.travel_costs[0] == 0.0
    assert len(snapshot.candidate_poses_xyz) == 32
    start_xy = provider.initial_pose[:2]
    same_xy_heights = {
        pose[2] for pose in snapshot.candidate_poses_xyz if pose[:2] == start_xy
    }
    assert len(same_xy_heights) >= 3
    assert len({round(pose[2], 6) for pose in snapshot.candidate_poses_xyz}) > 3
    assert snapshot.allowed_pair_ids == tuple(range(64))
    assert snapshot.current_pair_id == 19


def test_obstacle_candidates_exclude_disconnected_free_cells() -> None:
    """The estimator must never receive an unreachable measurement pose."""
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=1.0,
        grid_shape=(3, 1),
        blocked_cells=((1, 0),),
    )
    environment = {
        "size_x": 3.0,
        "size_y": 1.0,
        "size_z": 2.0,
        "detector_position": [0.25, 0.25, 0.5],
        "adaptive_measurement": {"candidate_count": 16},
    }
    provider = AdaptiveCandidateProvider(environment, grid)

    snapshot = provider.snapshot(provider.initial_pose, current_pair_id=0)

    assert all(pose[0] < 1.0 for pose in snapshot.candidate_poses_xyz)
    assert snapshot.candidate_poses_xyz[0] == (0.25, 0.25, 0.5)
    assert snapshot.travel_costs[0] == 0.0
    assert len({round(pose[2], 6) for pose in snapshot.candidate_poses_xyz}) > 1


def test_candidate_density_uses_one_nested_sobol_prefix() -> None:
    """Increasing candidate density must preserve the collision-free prefix."""
    low_environment = _environment()
    high_environment = _environment()
    low_environment["adaptive_measurement"] = {
        "candidate_count": 16,
        "candidate_seed": 17,
    }
    high_environment["adaptive_measurement"] = {
        "candidate_count": 32,
        "candidate_seed": 17,
    }

    low = AdaptiveCandidateProvider(low_environment, None)
    high = AdaptiveCandidateProvider(high_environment, None)

    assert high.all_poses[: len(low.all_poses)] == low.all_poses


def test_candidate_density_converges_monotonically_to_interior_target() -> None:
    """Nested candidate sets cannot lose the best approximation to a target pose."""
    target = np.asarray([2.4, 1.6, 1.6], dtype=float)
    best_distances: list[float] = []
    for candidate_count in (8, 16, 32, 64):
        environment = _environment()
        environment["adaptive_measurement"] = {
            "candidate_count": candidate_count,
            "candidate_seed": 23,
        }
        provider = AdaptiveCandidateProvider(environment, None)
        poses = np.asarray(provider.all_poses, dtype=float)
        best_distances.append(float(np.min(np.linalg.norm(poses - target, axis=1))))

    assert np.all(np.diff(best_distances) <= 1.0e-12)
    assert best_distances[-1] < best_distances[0]


def test_candidates_clear_base_mast_and_detector_head_boxes() -> None:
    """Runtime candidates must honor every detector-assembly collision volume."""
    environment = _environment()
    grid = ObstacleGrid(
        origin=(0.0, 0.0),
        cell_size=0.5,
        grid_shape=(6, 4),
        blocked_cells=(),
        collision_boxes_m=((1.1, 0.6, 0.0, 1.7, 1.4, 1.8),),
    )
    provider = AdaptiveCandidateProvider(environment, grid)
    snapshot = provider.snapshot(provider.initial_pose, current_pair_id=0)
    reachable = set(provider._reachable_cells(provider.initial_pose) or {})

    assert snapshot.candidate_poses_xyz
    assert all(
        _pose_is_clear(
            pose,
            environment,
            grid,
            provider.motion,
            reachable,
        )
        for pose in snapshot.candidate_poses_xyz
    )


def test_motion_time_separates_vertical_horizontal_and_settling_costs() -> None:
    """Vertical and horizontal actuator speeds must have separate semantics."""
    environment = _environment()
    environment["adaptive_measurement"] = {
        "candidate_count": 16,
        "horizontal_speed_m_s": 0.5,
        "vertical_speed_m_s": 0.25,
        "settling_time_s": 2.0,
        "transport_height_m": 0.4,
    }
    provider = AdaptiveCandidateProvider(environment, None)
    start = provider.initial_pose

    same_xy = provider.motion_time_s(start, (start[0], start[1], 1.0))
    translated = provider.motion_time_s(start, (1.25, start[1], 1.0))

    assert same_xy == pytest.approx(0.5 / 0.25 + 2.0)
    expected_vertical = abs(0.5 - 0.4) + abs(1.0 - 0.4)
    assert translated == pytest.approx(1.0 / 0.5 + expected_vertical / 0.25 + 2.0)


def test_travel_waypoints_follow_runtime_motion_model() -> None:
    """Selected actions should expose the same detector route used for CUI."""
    environment = _environment()
    environment["adaptive_measurement"] = {
        "candidate_count": 16,
        "transport_height_m": 0.4,
    }
    provider = AdaptiveCandidateProvider(environment, None)
    start = provider.initial_pose
    target = (1.25, start[1], 1.0)

    waypoints = provider.travel_waypoints_xyz(start, target)

    assert waypoints == (
        start,
        (start[0], start[1], 0.4),
        (target[0], target[1], 0.4),
        target,
    )


def test_local_refinement_is_runtime_filtered_and_adds_candidates() -> None:
    """Local Sobol refinement must add only runtime-validated 3-D poses."""
    provider = AdaptiveCandidateProvider(_environment(), None)
    coarse = provider.snapshot(provider.initial_pose, current_pair_id=0)

    refined = provider.refine(
        provider.initial_pose,
        current_pair_id=0,
        seed_poses=(coarse.candidate_poses_xyz[4], coarse.candidate_poses_xyz[5]),
    )

    assert len(refined.candidate_poses_xyz) > len(coarse.candidate_poses_xyz)
    assert set(coarse.candidate_poses_xyz).issubset(refined.candidate_poses_xyz)


def test_ral_mix9_profile_is_checked_only_inside_private_runtime() -> None:
    """RAL source cardinality should be enforced without entering estimator data."""
    sources = [SimpleNamespace(isotope="Cs-137") for _ in range(4)]
    sources.extend(SimpleNamespace(isotope="Co-60") for _ in range(3))
    sources.extend(SimpleNamespace(isotope="Eu-154") for _ in range(2))
    scene = SimpleNamespace(sources=sources)

    _validate_private_scene_profile(scene, "ral-mix9")

    scene.sources.pop()
    with pytest.raises(ValueError, match="exactly"):
        _validate_private_scene_profile(scene, "ral-mix9")


def test_ral_cs4_co3_eu0_profile_accepts_explicit_absence() -> None:
    """The Eu-zero profile must validate without exposing truth downstream."""
    sources = [SimpleNamespace(isotope="Cs-137") for _ in range(4)]
    sources.extend(SimpleNamespace(isotope="Co-60") for _ in range(3))
    scene = SimpleNamespace(sources=sources)

    _validate_private_scene_profile(scene, "ral-cs4-co3-eu0")

    scene.sources.append(SimpleNamespace(isotope="Eu-154"))
    with pytest.raises(ValueError, match="exactly"):
        _validate_private_scene_profile(scene, "ral-cs4-co3-eu0")


class _FakeObservationSession:
    """Persist selected actions as deterministic records for session tests."""

    def __init__(self) -> None:
        """Initialize a writer-shaped record list."""
        self.writer = SimpleNamespace(records=[])
        self.actions: list[Any] = []
        self.closed = False

    def step(self, action: Any) -> None:
        """Persist one selected action using the runtime record contract."""
        self.actions.append(action)
        metadata: dict[str, object] = {
            "full_spectrum_contract_hash_sha256": "a" * 64,
        }
        if action.station_complete:
            metadata["station_complete"] = True
        command = action.command
        self.writer.records.append(
            MeasurementLogRecord(
                step_id=command.step_id,
                action_id=command.step_id,
                station_id=action.station_id,
                detector_pose_xyz=command.target_pose_xyz,
                detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                fe_orientation_index=command.fe_orientation_index,
                pb_orientation_index=command.pb_orientation_index,
                live_time_s=command.dwell_time_s,
                travel_time_s=command.travel_time_s,
                shield_actuation_time_s=command.shield_actuation_time_s,
                energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0]),
                spectrum_counts=np.asarray([2, 3], dtype=np.int64),
                metadata=metadata,
            )
        )

    def close(self) -> None:
        """Mark the fake observation session closed."""
        self.closed = True


def _context() -> RunContext:
    """Return one minimal truth-free live context."""
    return RunContext(
        repository_commit="a" * 40,
        runtime_config={},
        environment=_environment(),
        sim_backend="test",
        spectrum_count_method="joint_full_spectrum_generative",
        isotopes=("Cs-137",),
        obstacle_layout_path=None,
        source_layout_path=None,
        source_rate_model="detector_cps_1m",
        metadata={},
        run_id="adaptive-test",
        source_rate_semantics={},
        forward_model_manifest={},
        runtime_config_sha256="b" * 64,
    )


def test_runtime_executes_only_the_current_selected_observation() -> None:
    """A request must resolve to one durable command, never a future action list."""
    observation = _FakeObservationSession()
    provider = AdaptiveCandidateProvider(_environment(), None)
    session = AdaptiveRuntimeSession(
        observation,  # type: ignore[arg-type]
        _context(),
        provider,
    )

    event = session.step(
        {
            "type": "step",
            "candidate_index": 0,
            "fe_orientation_index": 3,
            "pb_orientation_index": 6,
            "dwell_time_s": 30.0,
            "station_id": 0,
            "station_complete": True,
        }
    )

    assert event["type"] == "record"
    assert len(observation.writer.records) == 1
    record = observation.writer.records[0]
    assert record.fe_orientation_index == 3
    assert record.pb_orientation_index == 6
    assert record.shield_actuation_time_s > 0.0
    assert record.metadata["station_complete"] is True


def test_refined_current_pose_remains_available_for_same_station_views() -> None:
    """A locally refined pose must survive for subsequent shield-only actions."""
    observation = _FakeObservationSession()
    provider = AdaptiveCandidateProvider(_environment(), None)
    session = AdaptiveRuntimeSession(
        observation,  # type: ignore[arg-type]
        _context(),
        provider,
    )
    refined_event = session.refine({"type": "refine", "candidate_indices": [4, 5]})
    refined_poses = refined_event["candidates"]["candidate_poses_xyz"]
    base = set(provider.all_poses)
    refined_index = next(
        index for index, pose in enumerate(refined_poses) if tuple(pose) not in base
    )
    selected_pose = tuple(refined_poses[refined_index])

    event = session.step(
        {
            "type": "step",
            "candidate_index": refined_index,
            "fe_orientation_index": 0,
            "pb_orientation_index": 1,
            "dwell_time_s": 10.0,
            "station_id": 2,
            "station_complete": False,
        }
    )

    assert tuple(event["candidates"]["candidate_poses_xyz"][0]) == selected_pose
    assert event["candidates"]["travel_costs"][0] == 0.0


def test_same_station_views_preserve_the_arrival_base_yaw() -> None:
    """Shield-only views at one pose must retain an identical quaternion."""
    observation = _FakeObservationSession()
    provider = AdaptiveCandidateProvider(_environment(), None)
    session = AdaptiveRuntimeSession(
        observation,  # type: ignore[arg-type]
        _context(),
        provider,
    )
    initial = provider.snapshot(provider.initial_pose, current_pair_id=0)
    moved_index = next(
        index
        for index, pose in enumerate(initial.candidate_poses_xyz)
        if pose[:2] != provider.initial_pose[:2]
    )
    session.step(
        {
            "type": "step",
            "candidate_index": moved_index,
            "fe_orientation_index": 0,
            "pb_orientation_index": 1,
            "dwell_time_s": 10.0,
            "station_id": 0,
            "station_complete": False,
        }
    )
    session.step(
        {
            "type": "step",
            "candidate_index": 0,
            "fe_orientation_index": 1,
            "pb_orientation_index": 2,
            "dwell_time_s": 10.0,
            "station_id": 0,
            "station_complete": True,
        }
    )

    first_yaw = observation.actions[0].command.target_base_yaw_rad
    second_yaw = observation.actions[1].command.target_base_yaw_rad
    assert first_yaw != 0.0
    assert second_yaw == pytest.approx(first_yaw)
    assert observation.actions[0].command.travel_waypoints_xyz is not None
    assert observation.actions[1].command.travel_waypoints_xyz is None


class _FakeAdaptiveSession:
    """Expose a deterministic session surface for protocol tests."""

    def __init__(self) -> None:
        """Initialize request capture and close state."""
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    def ready_payload(self) -> dict[str, object]:
        """Return a minimal truth-free handshake."""
        return {
            "type": "ready",
            "schema_version": 1,
            "context": {"run_id": "test"},
            "candidates": {"candidate_poses_xyz": [[0.0, 0.0, 0.5]]},
            "bootstrap": {
                "candidate_index": 0,
                "fe_orientation_index": 0,
                "pb_orientation_index": 0,
            },
        }

    def step(self, request: dict[str, Any]) -> dict[str, object]:
        """Capture one controller-selected action."""
        self.requests.append(dict(request))
        return {"type": "record", "record": {"step_id": 0}, "candidates": {}}

    def refine(self, request: dict[str, Any]) -> dict[str, object]:
        """Capture one estimator-ranked local-refinement request."""
        self.requests.append(dict(request))
        return {"type": "candidates", "candidates": {}}

    def cui_overlay(self, request: dict[str, Any]) -> dict[str, object]:
        """Return private CUI overlay data without entering normal events."""
        self.requests.append(dict(request))
        return {
            "type": "cui_overlay",
            "schema_version": 1,
            "truth": {
                "schema_version": 1,
                "semantics": "evaluation_cui_overlay_only_not_estimator_input",
                "true_sources": {"Cs-137": [[1.0, 1.0, 1.0]]},
                "true_strengths": {"Cs-137": [300000.0]},
            },
        }

    def finalize(self) -> tuple[object, dict[str, object]]:
        """Return a published-log event."""
        self.closed = True
        return object(), {"type": "published", "path": "/tmp/log", "record_count": 1}

    def close(self) -> None:
        """Mark the fake session closed."""
        self.closed = True


def test_adaptive_protocol_accepts_actions_incrementally(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The runtime protocol should receive one selected action, never a plan."""
    fake = _FakeAdaptiveSession()
    monkeypatch.setattr(
        AdaptiveRuntimeSession,
        "open",
        classmethod(lambda cls, path, private_scene_profile=None: fake),
    )
    step = {
        "type": "step",
        "candidate_index": 0,
        "fe_orientation_index": 1,
        "pb_orientation_index": 2,
        "dwell_time_s": 30.0,
        "station_id": 0,
        "station_complete": True,
    }
    input_stream = StringIO(
        json.dumps(step) + "\n" + json.dumps({"type": "finalize"}) + "\n"
    )
    output_stream = StringIO()

    status = serve_adaptive_session(
        tmp_path / "private-scenario.json",
        input_stream=input_stream,
        output_stream=output_stream,
    )

    events = [
        json.loads(line.removeprefix(ADAPTIVE_EVENT_PREFIX))
        for line in output_stream.getvalue().splitlines()
    ]
    assert status == 0
    assert fake.requests == [step]
    assert [event["type"] for event in events] == ["ready", "record", "published"]
    assert all("actions" not in event for event in events)


def test_adaptive_protocol_supports_estimator_ranked_runtime_refinement(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Refinement requests must carry indices rather than estimator internals."""
    fake = _FakeAdaptiveSession()
    monkeypatch.setattr(
        AdaptiveRuntimeSession,
        "open",
        classmethod(lambda cls, path, private_scene_profile=None: fake),
    )
    refine = {"type": "refine", "candidate_indices": [0]}
    input_stream = StringIO(
        json.dumps(refine) + "\n" + json.dumps({"type": "abort"}) + "\n"
    )
    output_stream = StringIO()

    status = serve_adaptive_session(
        tmp_path / "private-scenario.json",
        input_stream=input_stream,
        output_stream=output_stream,
    )

    events = [
        json.loads(line.removeprefix(ADAPTIVE_EVENT_PREFIX))
        for line in output_stream.getvalue().splitlines()
    ]
    assert status == 0
    assert fake.requests == [refine]
    assert [event["type"] for event in events] == ["ready", "candidates", "aborted"]


def test_adaptive_protocol_uses_private_prefix_for_cui_overlay(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Evaluation truth must not be emitted as an estimator-visible event."""
    fake = _FakeAdaptiveSession()
    monkeypatch.setattr(
        AdaptiveRuntimeSession,
        "open",
        classmethod(lambda cls, path, private_scene_profile=None: fake),
    )
    overlay = {"type": "cui_overlay", "include_truth": True}
    input_stream = StringIO(
        json.dumps(overlay) + "\n" + json.dumps({"type": "abort"}) + "\n"
    )
    output_stream = StringIO()

    status = serve_adaptive_session(
        tmp_path / "private-scenario.json",
        input_stream=input_stream,
        output_stream=output_stream,
    )

    lines = output_stream.getvalue().splitlines()
    normal_events = [
        json.loads(line.removeprefix(ADAPTIVE_EVENT_PREFIX))
        for line in lines
        if line.startswith(ADAPTIVE_EVENT_PREFIX)
    ]
    private_events = [
        json.loads(line.removeprefix(ADAPTIVE_CUI_OVERLAY_PREFIX))
        for line in lines
        if line.startswith(ADAPTIVE_CUI_OVERLAY_PREFIX)
    ]
    assert status == 0
    assert fake.requests == [overlay]
    assert [event["type"] for event in normal_events] == ["ready", "aborted"]
    assert len(private_events) == 1
    assert private_events[0]["truth"]["true_sources"]["Cs-137"] == [
        [1.0, 1.0, 1.0]
    ]


def test_adaptive_client_requests_truth_only_on_private_cui_protocol() -> None:
    """The client must keep CUI truth outside estimator-validated events."""
    response = {
        "type": "cui_overlay",
        "schema_version": 1,
        "truth": {
            "schema_version": 1,
            "semantics": "evaluation_cui_overlay_only_not_estimator_input",
            "true_sources": {"Cs-137": [[1.0, 1.0, 1.0]]},
            "true_strengths": {"Cs-137": [300000.0]},
        },
    }
    client = AdaptiveRuntimeClient.__new__(AdaptiveRuntimeClient)
    client.input = StringIO()
    client.output = StringIO(
        ADAPTIVE_CUI_OVERLAY_PREFIX + json.dumps(response) + "\n"
    )
    client.output_hook = lambda message: None
    client.process = SimpleNamespace(poll=lambda: None)

    payload = client.request_cui_overlay(include_truth=True)

    assert json.loads(client.input.getvalue()) == {
        "type": "cui_overlay",
        "include_truth": True,
    }
    assert payload == response
