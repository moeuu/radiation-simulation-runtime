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
    _adaptive_resume_compatibility,
    _pose_is_clear,
    _validate_private_scene_profile,
    serve_adaptive_session,
)
from runtime.adaptive_client import AdaptiveRuntimeClient
from runtime.adaptive_client import parse_adaptive_resume_prefix
from runtime.measurement_log import MeasurementLogRecord
from runtime.records import RunContext
from runtime.scenarios import build_random_ral_mix9_scenario, write_private_scenario
from sim.protocol import SimulationCommand, SimulationObservation
from sim.runtime import SimulationRuntime


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
    assert snapshot.shield_angular_speed_rad_s == pytest.approx(np.pi / 4.0)


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


def test_cross_commit_resume_requires_explicit_compatibility_provenance() -> None:
    """Runtime code changes must fail closed without a durable review record."""
    with pytest.raises(ValueError, match="across runtime commits"):
        _adaptive_resume_compatibility(
            prefix_repository_commit="a" * 40,
            resume_execution_commit="b" * 40,
            supplied=None,
        )

    payload = _adaptive_resume_compatibility(
        prefix_repository_commit="a" * 40,
        resume_execution_commit="b" * 40,
        supplied={"compatibility_review": "physics-contract-unchanged"},
    )

    assert payload["compatibility_mode"] == "explicit_cross_commit"
    assert payload["repository_commits_match"] is False
    assert payload["compatibility_review"] == "physics-contract-unchanged"


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


class _DurableFakeRuntime(SimulationRuntime):
    """Return production-axis spectra while exercising the real stream writer."""

    def __init__(self) -> None:
        """Initialize reset and close state."""
        self.reset_payload: dict[str, Any] | None = None
        self.closed = False

    def reset(self, payload: dict[str, Any] | None = None) -> None:
        """Retain one private reset payload without exposing it."""
        self.reset_payload = dict(payload or {})

    def step(self, command: SimulationCommand) -> SimulationObservation:
        """Return one exact deterministic spectrum at the commanded pose."""
        half_yaw = 0.5 * float(command.target_base_yaw_rad)
        return SimulationObservation(
            step_id=command.step_id,
            detector_pose_xyz=command.target_pose_xyz,
            detector_quat_wxyz=(
                float(np.cos(half_yaw)),
                0.0,
                0.0,
                float(np.sin(half_yaw)),
            ),
            fe_orientation_index=command.fe_orientation_index,
            pb_orientation_index=command.pb_orientation_index,
            spectrum_counts=np.ones(851, dtype=np.int64).tolist(),
            energy_bin_edges_keV=np.linspace(0.0, 1702.0, 852).tolist(),
            metadata={
                "detector_response_sampling_mode": (
                    "multinomial_marking_with_nonparalyzable_event_time"
                )
            },
        )

    def close(self) -> None:
        """Mark the fake runtime closed."""
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
    quoted_time_s = session._candidate_snapshot.quote_shield_program_time_s((30,))

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
    assert record.shield_actuation_time_s == pytest.approx(quoted_time_s)
    assert record.metadata["station_complete"] is True


def test_shield_program_quote_equals_sequential_executed_record_times() -> None:
    """One pre-station quote must equal every later pair transition combined."""
    environment = _environment()
    environment["adaptive_measurement"] = {
        "candidate_count": 16,
        "shield_angular_speed_rad_s": 2.0,
    }
    observation = _FakeObservationSession()
    provider = AdaptiveCandidateProvider(environment, None)
    session = AdaptiveRuntimeSession(
        observation,  # type: ignore[arg-type]
        _context(),
        provider,
    )
    program = (1, 9, 63)
    quoted_time_s = session._candidate_snapshot.quote_shield_program_time_s(
        program
    )

    for index, pair_id in enumerate(program):
        fe_index, pb_index = divmod(pair_id, 8)
        session.step(
            {
                "type": "step",
                "candidate_index": 0,
                "fe_orientation_index": fe_index,
                "pb_orientation_index": pb_index,
                "dwell_time_s": 1.0,
                "station_id": 0,
                "station_complete": index == len(program) - 1,
            }
        )

    assert sum(
        record.shield_actuation_time_s for record in observation.writer.records
    ) == pytest.approx(quoted_time_s)


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


def test_resumed_session_restores_pose_pair_yaw_and_completed_prefix() -> None:
    """A resumed handshake must restore causal motion state and prior records."""
    yaw = 0.75
    record = MeasurementLogRecord(
        step_id=0,
        action_id=0,
        station_id=0,
        detector_pose_xyz=(1.25, 0.75, 0.5),
        detector_quat_wxyz=(
            float(np.cos(0.5 * yaw)),
            0.0,
            0.0,
            float(np.sin(0.5 * yaw)),
        ),
        fe_orientation_index=3,
        pb_orientation_index=4,
        live_time_s=10.0,
        travel_time_s=2.0,
        shield_actuation_time_s=1.0,
        energy_bin_edges_keV=np.asarray([0.0, 1.0, 2.0]),
        spectrum_counts=np.asarray([2, 3], dtype=np.int64),
        metadata={
            "full_spectrum_contract_hash_sha256": "a" * 64,
            "station_complete": True,
        },
    )
    observation = _FakeObservationSession()
    observation.writer.records.append(record)
    provider = AdaptiveCandidateProvider(_environment(), None)
    session = AdaptiveRuntimeSession(
        observation,  # type: ignore[arg-type]
        _context(),
        provider,
        resume_records=(record,),
    )

    ready = session.ready_payload()
    prefix = parse_adaptive_resume_prefix(ready["resume"])
    current = ready["candidates"]
    event = session.step(
        {
            "type": "step",
            "candidate_index": 0,
            "fe_orientation_index": 3,
            "pb_orientation_index": 5,
            "dwell_time_s": 10.0,
            "station_id": 1,
            "station_complete": True,
        }
    )

    assert ready["schema_version"] == 2
    assert len(prefix.records) == 1
    assert prefix.records[0].step_id == record.step_id
    np.testing.assert_array_equal(
        prefix.records[0].spectrum_counts,
        record.spectrum_counts,
    )
    assert prefix.next_station_id == 1
    assert tuple(current["candidate_poses_xyz"][0]) == record.detector_pose_xyz
    assert current["current_pair_id"] == 28
    assert event["record"]["step_id"] == 1
    assert observation.actions[-1].command.target_base_yaw_rad == pytest.approx(yaw)


def test_public_resume_adopts_verified_stage_and_continues_step_ids(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The public API must copy a completed prefix and publish its continuation."""
    runtime_config = (
        Path(__file__).resolve().parents[1]
        / "configs/geant4/variance_reduction_external_no_isaac_32threads.json"
    )
    scenario = build_random_ral_mix9_scenario(
        scene_seed=31415,
        runtime_config_path=runtime_config,
        measurement_log_output_dir=tmp_path / "measurement-log",
        run_id="adaptive-resume-integration",
        candidate_count=16,
    )
    scenario_path = write_private_scenario(tmp_path / "scenario.json", scenario)
    runtimes: list[_DurableFakeRuntime] = []

    def create_runtime(*args: object, **kwargs: object) -> _DurableFakeRuntime:
        """Return a separately observable fake runtime for each open."""
        runtime = _DurableFakeRuntime()
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr("runtime.adaptive.create_simulation_runtime", create_runtime)
    first = AdaptiveRuntimeSession.open(scenario_path)
    first.step(
        {
            "type": "step",
            "candidate_index": 0,
            "fe_orientation_index": 2,
            "pb_orientation_index": 3,
            "dwell_time_s": 1.0,
            "station_id": 0,
            "station_complete": True,
        }
    )
    source_stage = first.observation_session.writer.stage_dir
    first.close()

    resumed = AdaptiveRuntimeSession.resume(
        scenario_path,
        stage_dir=source_stage,
    )
    ready = resumed.ready_payload()
    resumed_event = resumed.step(
        {
            "type": "step",
            "candidate_index": 0,
            "fe_orientation_index": 4,
            "pb_orientation_index": 5,
            "dwell_time_s": 1.0,
            "station_id": 1,
            "station_complete": True,
        }
    )
    log, published = resumed.finalize()

    assert ready["schema_version"] == 2
    assert ready["resume"]["record_count"] == 1
    assert resumed_event["record"]["step_id"] == 1
    assert published["record_count"] == 2
    assert [record.step_id for record in log.records] == [0, 1]
    assert log.records[1].metadata["resume_prefix_record_count"] == 1
    assert source_stage.is_dir()
    assert len(runtimes) == 2
    assert all(runtime.closed for runtime in runtimes)


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


def test_adaptive_protocol_routes_resume_stage_through_public_session_api(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The server entrypoint must adopt a stage through the public resume API."""
    fake = _FakeAdaptiveSession()
    captured: dict[str, object] = {}

    def resume(
        cls: type[AdaptiveRuntimeSession],
        path: Path,
        *,
        stage_dir: Path,
        resume_compatibility: dict[str, object] | None = None,
        private_scene_profile: str | None = None,
    ) -> _FakeAdaptiveSession:
        """Capture public resume arguments and return a fake live session."""
        captured.update(
            {
                "path": path,
                "stage_dir": stage_dir,
                "resume_compatibility": resume_compatibility,
                "private_scene_profile": private_scene_profile,
            }
        )
        return fake

    monkeypatch.setattr(AdaptiveRuntimeSession, "resume", classmethod(resume))
    output_stream = StringIO()
    status = serve_adaptive_session(
        tmp_path / "private-scenario.json",
        input_stream=StringIO(json.dumps({"type": "abort"}) + "\n"),
        output_stream=output_stream,
        private_scene_profile="ral-mix9",
        resume_stage_dir=tmp_path / ".measurement-log.stream-7",
        resume_compatibility={"review": "approved"},
    )

    assert status == 0
    assert captured == {
        "path": tmp_path / "private-scenario.json",
        "stage_dir": tmp_path / ".measurement-log.stream-7",
        "resume_compatibility": {"review": "approved"},
        "private_scene_profile": "ral-mix9",
    }


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


def test_adaptive_client_rejects_truth_before_writing_cui_request() -> None:
    """The estimator-facing client must not open the runtime truth channel."""
    response = {
        "type": "cui_overlay",
        "schema_version": 1,
        "truth": None,
    }
    client = AdaptiveRuntimeClient.__new__(AdaptiveRuntimeClient)
    client.input = StringIO()
    client.output = StringIO(
        ADAPTIVE_CUI_OVERLAY_PREFIX + json.dumps(response) + "\n"
    )
    client.output_hook = lambda message: None
    client.process = SimpleNamespace(poll=lambda: None)

    with pytest.raises(ValueError, match="cannot request realized truth"):
        client.request_cui_overlay(include_truth=True)

    assert client.input.getvalue() == ""


def test_adaptive_client_accepts_only_truth_free_cui_response() -> None:
    """The estimator-facing CUI channel must require a null truth member."""
    response = {
        "type": "cui_overlay",
        "schema_version": 1,
        "truth": None,
    }
    client = AdaptiveRuntimeClient.__new__(AdaptiveRuntimeClient)
    client.input = StringIO()
    client.output = StringIO(
        ADAPTIVE_CUI_OVERLAY_PREFIX + json.dumps(response) + "\n"
    )
    client.output_hook = lambda message: None
    client.process = SimpleNamespace(poll=lambda: None)

    payload = client.request_cui_overlay(include_truth=False)

    assert json.loads(client.input.getvalue()) == {
        "type": "cui_overlay",
        "include_truth": False,
    }
    assert payload == response


def test_adaptive_client_rejects_unexpected_truth_cui_response() -> None:
    """A runtime response cannot inject realized truth into the client."""
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

    with pytest.raises(ValueError, match="cannot receive realized truth"):
        client.request_cui_overlay(include_truth=False)
