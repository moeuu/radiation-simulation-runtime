"""Tests for estimator-neutral interactive runtime acquisition."""

from __future__ import annotations

import json
from dataclasses import replace
from io import StringIO
from pathlib import Path
from threading import Thread
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
    _validate_private_scene_variant,
    _validate_production_scenario,
    serve_adaptive_session,
    serve_adaptive_session_socket,
)
from runtime.adaptive_client import AdaptiveRuntimeClient
from runtime.experiment_profiles import (
    DEFAULT_EXPERIMENT_PROFILE_ID,
    STANDARD_EXPERIMENT_PROFILE,
    AcquisitionContract,
    ExperimentProfile,
)
from runtime.measurement_log import MeasurementLogRecord
from runtime.records import RunContext
from runtime.scenarios import build_random_surface_scenario, write_private_scenario
from sim.protocol import SimulationCommand, SimulationObservation
from sim.runtime import SimulationRuntime
from tests.runtime_test_support import runtime_config


def _adaptive_measurement(**overrides: object) -> dict[str, object]:
    """Return one complete explicit adaptive-motion test contract."""
    payload: dict[str, object] = {
        "candidate_count": 32,
        "candidate_seed": 0,
        "detector_height_min_m": 0.2 + 0.0395,
        "detector_height_max_m": 2.0 - 0.0395,
        "local_refinement_count": 16,
        "local_refinement_radius_m": 0.5,
        "base_radius_m": 0.2,
        "base_height_m": 0.2,
        "mast_radius_m": 0.03,
        "head_radius_m": 0.0395,
        "transport_height_m": 0.2 + 0.0395,
        "horizontal_speed_m_s": 0.5,
        "vertical_speed_m_s": 0.25,
        "settling_time_s": 0.75,
        "shield_angular_speed_rad_s": float(np.pi / 4.0),
    }
    payload.update(overrides)
    return payload


def _environment() -> dict[str, object]:
    """Return a small truth-free room and detector start pose."""
    return {
        "size_x": 3.0,
        "size_y": 2.0,
        "size_z": 2.0,
        "detector_position": [0.25, 0.25, 0.5],
        "adaptive_measurement": _adaptive_measurement(),
    }


def test_adaptive_motion_requires_the_complete_explicit_schema() -> None:
    """Missing and unknown motion fields must fail before candidate generation."""
    missing_container = _environment()
    del missing_container["adaptive_measurement"]
    with pytest.raises(ValueError, match="adaptive_measurement is required"):
        AdaptiveCandidateProvider(missing_container, None)

    missing_field = _environment()
    missing_payload = dict(missing_field["adaptive_measurement"])
    del missing_payload["candidate_seed"]
    missing_field["adaptive_measurement"] = missing_payload
    with pytest.raises(ValueError, match=r"missing=\['candidate_seed'\]"):
        AdaptiveCandidateProvider(missing_field, None)

    unknown_field = _environment()
    unknown_payload = dict(unknown_field["adaptive_measurement"])
    unknown_payload["candidate_sead"] = 7
    unknown_field["adaptive_measurement"] = unknown_payload
    with pytest.raises(ValueError, match=r"unknown=\['candidate_sead'\]"):
        AdaptiveCandidateProvider(unknown_field, None)


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
        "adaptive_measurement": _adaptive_measurement(candidate_count=16),
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
    low_environment["adaptive_measurement"] = _adaptive_measurement(
        candidate_count=16,
        candidate_seed=17,
    )
    high_environment["adaptive_measurement"] = _adaptive_measurement(
        candidate_count=32,
        candidate_seed=17,
    )

    low = AdaptiveCandidateProvider(low_environment, None)
    high = AdaptiveCandidateProvider(high_environment, None)

    assert high.all_poses[: len(low.all_poses)] == low.all_poses


def test_candidate_density_converges_monotonically_to_interior_target() -> None:
    """Nested candidate sets cannot lose the best approximation to a target pose."""
    target = np.asarray([2.4, 1.6, 1.6], dtype=float)
    best_distances: list[float] = []
    for candidate_count in (8, 16, 32, 64):
        environment = _environment()
        environment["adaptive_measurement"] = _adaptive_measurement(
            candidate_count=candidate_count,
            candidate_seed=23,
        )
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
    environment["adaptive_measurement"] = _adaptive_measurement(
        candidate_count=16,
        horizontal_speed_m_s=0.5,
        vertical_speed_m_s=0.25,
        settling_time_s=2.0,
        transport_height_m=0.4,
    )
    provider = AdaptiveCandidateProvider(environment, None)
    start = provider.initial_pose

    same_xy = provider.motion_time_s(start, (start[0], start[1], 1.0))
    translated = provider.motion_time_s(start, (1.25, start[1], 1.0))
    translated_components = provider.motion_time_components_s(
        start,
        (1.25, start[1], 1.0),
    )

    assert same_xy == pytest.approx(0.5 / 0.25 + 2.0)
    expected_vertical = abs(0.5 - 0.4) + abs(1.0 - 0.4)
    assert translated == pytest.approx(1.0 / 0.5 + expected_vertical / 0.25 + 2.0)
    assert translated_components == pytest.approx(
        (1.0 / 0.5, expected_vertical / 0.25, 2.0)
    )


def test_candidate_snapshot_publishes_motion_time_components() -> None:
    """Runtime snapshots must publish components that sum to every total cost."""
    provider = AdaptiveCandidateProvider(_environment(), None)

    snapshot = provider.snapshot(provider.initial_pose, current_pair_id=0)

    assert snapshot.horizontal_travel_times_s is not None
    assert snapshot.mast_vertical_times_s is not None
    assert snapshot.settling_times_s is not None
    totals = (
        np.asarray(snapshot.horizontal_travel_times_s)
        + np.asarray(snapshot.mast_vertical_times_s)
        + np.asarray(snapshot.settling_times_s)
    )
    np.testing.assert_allclose(totals, np.asarray(snapshot.travel_costs))
    payload = snapshot.to_payload()
    assert "horizontal_travel_times_s" in payload
    assert "mast_vertical_times_s" in payload
    assert "settling_times_s" in payload


def test_travel_waypoints_follow_runtime_motion_model() -> None:
    """Selected actions should expose the same detector route used for CUI."""
    environment = _environment()
    environment["adaptive_measurement"] = _adaptive_measurement(
        candidate_count=16,
        transport_height_m=0.4,
    )
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


def test_default_scene_variant_is_checked_only_inside_private_runtime() -> None:
    """Source cardinality should be enforced without entering estimator data."""
    sources = [SimpleNamespace(isotope="Cs-137") for _ in range(4)]
    sources.extend(SimpleNamespace(isotope="Co-60") for _ in range(3))
    sources.extend(SimpleNamespace(isotope="Eu-154") for _ in range(2))
    scene = SimpleNamespace(sources=sources)

    _validate_private_scene_variant(
        scene,
        DEFAULT_EXPERIMENT_PROFILE_ID,
        "mix9",
    )

    scene.sources.pop()
    with pytest.raises(ValueError, match="exactly"):
        _validate_private_scene_variant(
            scene,
            DEFAULT_EXPERIMENT_PROFILE_ID,
            "mix9",
        )


def test_cs4_co3_eu0_variant_accepts_explicit_absence() -> None:
    """The Eu-zero profile must validate without exposing truth downstream."""
    sources = [SimpleNamespace(isotope="Cs-137") for _ in range(4)]
    sources.extend(SimpleNamespace(isotope="Co-60") for _ in range(3))
    scene = SimpleNamespace(sources=sources)

    _validate_private_scene_variant(
        scene,
        DEFAULT_EXPERIMENT_PROFILE_ID,
        "cs4-co3-eu0",
    )

    scene.sources.append(SimpleNamespace(isotope="Eu-154"))
    with pytest.raises(ValueError, match="exactly"):
        _validate_private_scene_variant(
            scene,
            DEFAULT_EXPERIMENT_PROFILE_ID,
            "cs4-co3-eu0",
        )


class _FakeObservationSession:
    """Persist selected actions as deterministic records for session tests."""

    def __init__(self) -> None:
        """Initialize a writer-shaped record list."""
        self.writer = SimpleNamespace(records=[])
        self.actions: list[Any] = []
        self.closed = False
        self.finalized = False

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

    def finalize(self) -> object:
        """Return one log-shaped sentinel after recording publication."""
        self.finalized = True
        return SimpleNamespace(
            path=Path("/tmp/adaptive-contract-test"),
            records=tuple(self.writer.records),
        )


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
                ),
                "dwell_time_s": command.dwell_time_s,
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


def _experiment_profile(
    *,
    max_stations: int,
    views_per_station: int,
    live_time_s: float,
) -> ExperimentProfile:
    """Return one valid compact acquisition profile for live-session tests."""
    return replace(
        STANDARD_EXPERIMENT_PROFILE,
        acquisition=AcquisitionContract(
            max_stations=max_stations,
            views_per_station=views_per_station,
            live_time_s=live_time_s,
            max_measurements=max_stations * views_per_station,
            min_station_separation_m=3.0,
            coverage_radius_m=3.0,
        ),
    )


def test_runtime_executes_only_the_current_selected_observation() -> None:
    """A request must resolve to one durable command, never a future action list."""
    observation = _FakeObservationSession()
    provider = AdaptiveCandidateProvider(_environment(), None)
    session = AdaptiveRuntimeSession(
        observation,  # type: ignore[arg-type]
        _context(),
        provider,
        _experiment_profile(max_stations=2, views_per_station=1, live_time_s=30.0),
    )
    quoted_time_s = session._candidate_snapshot.quote_shield_program_time_s((30,))

    event = session.step(
        {
            "type": "step",
            "action_id": 0,
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


def test_runtime_rejects_stale_action_id_before_transport() -> None:
    """A replayed or skipped action id must not execute a simulator command."""
    observation = _FakeObservationSession()
    provider = AdaptiveCandidateProvider(_environment(), None)
    session = AdaptiveRuntimeSession(
        observation,  # type: ignore[arg-type]
        _context(),
        provider,
        _experiment_profile(max_stations=2, views_per_station=1, live_time_s=1.0),
    )

    with pytest.raises(ValueError, match="expected 0, got 1"):
        session.step(
            {
                "type": "step",
                "action_id": 1,
                "candidate_index": 0,
                "fe_orientation_index": 0,
                "pb_orientation_index": 0,
                "dwell_time_s": 1.0,
                "station_id": 0,
                "station_complete": True,
            }
        )

    assert observation.actions == []
    assert observation.writer.records == []


def test_shield_program_quote_equals_sequential_executed_record_times() -> None:
    """One pre-station quote must equal every later pair transition combined."""
    environment = _environment()
    environment["adaptive_measurement"] = _adaptive_measurement(
        candidate_count=16,
        shield_angular_speed_rad_s=2.0,
    )
    observation = _FakeObservationSession()
    provider = AdaptiveCandidateProvider(environment, None)
    session = AdaptiveRuntimeSession(
        observation,  # type: ignore[arg-type]
        _context(),
        provider,
        _experiment_profile(max_stations=2, views_per_station=3, live_time_s=1.0),
    )
    program = (1, 9, 63)
    quoted_time_s = session._candidate_snapshot.quote_shield_program_time_s(program)

    for index, pair_id in enumerate(program):
        fe_index, pb_index = divmod(pair_id, 8)
        session.step(
            {
                "type": "step",
                "action_id": index,
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
        _experiment_profile(max_stations=2, views_per_station=2, live_time_s=10.0),
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
            "action_id": 0,
            "candidate_index": refined_index,
            "fe_orientation_index": 0,
            "pb_orientation_index": 1,
            "dwell_time_s": 10.0,
            "station_id": 0,
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
        _experiment_profile(max_stations=2, views_per_station=2, live_time_s=10.0),
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
            "action_id": 0,
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
            "action_id": 1,
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


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"dwell_time_s": 9.0}, "dwell_time_s differs"),
        ({"station_id": 1}, "next contract station"),
        ({"station_complete": True}, "exactly the final view"),
    ),
)
def test_live_session_rejects_contract_drift_before_transport(
    override: dict[str, object],
    message: str,
) -> None:
    """Dwell and station-boundary drift must fail before observation."""
    observation = _FakeObservationSession()
    provider = AdaptiveCandidateProvider(_environment(), None)
    session = AdaptiveRuntimeSession(
        observation,  # type: ignore[arg-type]
        _context(),
        provider,
        _experiment_profile(max_stations=2, views_per_station=2, live_time_s=10.0),
    )
    request: dict[str, object] = {
        "type": "step",
        "action_id": 0,
        "candidate_index": 0,
        "fe_orientation_index": 0,
        "pb_orientation_index": 0,
        "dwell_time_s": 10.0,
        "station_id": 0,
        "station_complete": False,
    }
    request.update(override)

    with pytest.raises(ValueError, match=message):
        session.step(request)

    assert observation.actions == []


def test_live_session_requires_exact_views_and_one_pose_per_station() -> None:
    """Each station must contain its exact ordered views at one detector pose."""
    observation = _FakeObservationSession()
    provider = AdaptiveCandidateProvider(_environment(), None)
    session = AdaptiveRuntimeSession(
        observation,  # type: ignore[arg-type]
        _context(),
        provider,
        _experiment_profile(max_stations=2, views_per_station=2, live_time_s=10.0),
    )
    session.step(
        {
            "type": "step",
            "action_id": 0,
            "candidate_index": 0,
            "fe_orientation_index": 0,
            "pb_orientation_index": 0,
            "dwell_time_s": 10.0,
            "station_id": 0,
            "station_complete": False,
        }
    )
    moved_index = next(
        index
        for index, pose in enumerate(
            session._candidate_snapshot.candidate_poses_xyz
        )
        if pose != session.current_pose
    )
    with pytest.raises(ValueError, match="station's first detector pose"):
        session.step(
            {
                "type": "step",
                "action_id": 1,
                "candidate_index": moved_index,
                "fe_orientation_index": 0,
                "pb_orientation_index": 1,
                "dwell_time_s": 10.0,
                "station_id": 0,
                "station_complete": True,
            }
        )
    assert len(observation.actions) == 1

    session.step(
        {
            "type": "step",
            "action_id": 1,
            "candidate_index": 0,
            "fe_orientation_index": 0,
            "pb_orientation_index": 1,
            "dwell_time_s": 10.0,
            "station_id": 0,
            "station_complete": True,
        }
    )
    session.step(
        {
            "type": "step",
            "action_id": 2,
            "candidate_index": 0,
            "fe_orientation_index": 1,
            "pb_orientation_index": 1,
            "dwell_time_s": 10.0,
            "station_id": 1,
            "station_complete": False,
        }
    )

    assert [record.station_id for record in observation.writer.records] == [0, 0, 1]


def test_live_session_rejects_finalize_without_complete_station() -> None:
    """Zero-record and partial-station logs must never be published."""
    observation = _FakeObservationSession()
    provider = AdaptiveCandidateProvider(_environment(), None)
    session = AdaptiveRuntimeSession(
        observation,  # type: ignore[arg-type]
        _context(),
        provider,
        _experiment_profile(max_stations=2, views_per_station=2, live_time_s=10.0),
    )

    with pytest.raises(RuntimeError, match="zero records"):
        session.finalize()
    session.step(
        {
            "type": "step",
            "action_id": 0,
            "candidate_index": 0,
            "fe_orientation_index": 0,
            "pb_orientation_index": 0,
            "dwell_time_s": 10.0,
            "station_id": 0,
            "station_complete": False,
        }
    )
    with pytest.raises(RuntimeError, match="current station"):
        session.finalize()

    assert observation.finalized is False


def test_live_session_allows_only_finalize_or_abort_after_exact_limit() -> None:
    """The server must not accept any further selection after its hard limit."""
    observation = _FakeObservationSession()
    provider = AdaptiveCandidateProvider(_environment(), None)
    session = AdaptiveRuntimeSession(
        observation,  # type: ignore[arg-type]
        _context(),
        provider,
        _experiment_profile(max_stations=1, views_per_station=1, live_time_s=10.0),
    )
    request = {
        "type": "step",
        "action_id": 0,
        "candidate_index": 0,
        "fe_orientation_index": 0,
        "pb_orientation_index": 0,
        "dwell_time_s": 10.0,
        "station_id": 0,
        "station_complete": True,
    }
    session.step(request)

    with pytest.raises(RuntimeError, match="measurement limit"):
        session.step({**request, "action_id": 1, "station_id": 1})
    with pytest.raises(RuntimeError, match="measurement limit"):
        session.refine({"type": "refine", "candidate_indices": [0]})
    with pytest.raises(RuntimeError, match="measurement limit"):
        session.cui_overlay({"type": "cui_overlay", "include_truth": False})

    _, published = session.finalize()
    assert published["record_count"] == 1
    assert observation.finalized is True


def test_unapproved_model_fails_before_simulator_creation(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Production preflight must reject the model before transport is opened."""
    scenario = build_random_surface_scenario(
        scene_seed=27182,
        measurement_log_output_dir=tmp_path / "measurement-log",
        run_id="adaptive-production-gate-test",
    )
    scenario_path = write_private_scenario(tmp_path / "scenario.json", scenario)
    simulator_created = False

    def create_runtime(*args: object, **kwargs: object) -> _DurableFakeRuntime:
        """Record any forbidden simulator construction after failed preflight."""
        nonlocal simulator_created
        simulator_created = True
        return _DurableFakeRuntime()

    monkeypatch.setattr("runtime.adaptive.create_simulation_runtime", create_runtime)

    with pytest.raises(RuntimeError, match="independent all-64 holdout"):
        AdaptiveRuntimeSession.open(scenario_path)

    assert simulator_created is False


def test_analytic_scenario_backend_fails_before_runtime_or_writer(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A private scenario cannot select the approximate debug backend."""
    scenario = build_random_surface_scenario(
        scene_seed=16180,
        measurement_log_output_dir=tmp_path / "measurement-log",
        run_id="adaptive-analytic-backend-test",
    )
    scenario["backend"] = "analytic"
    scenario_path = write_private_scenario(tmp_path / "scenario.json", scenario)
    simulator_created = False

    def create_runtime(*args: object, **kwargs: object) -> _DurableFakeRuntime:
        """Record any forbidden simulator construction after failed preflight."""
        nonlocal simulator_created
        simulator_created = True
        return _DurableFakeRuntime()

    monkeypatch.setattr("runtime.adaptive.create_simulation_runtime", create_runtime)

    with pytest.raises(ValueError, match="backend must equal 'geant4'"):
        AdaptiveRuntimeSession.open(scenario_path)

    assert simulator_created is False
    assert not (tmp_path / "measurement-log").exists()


def test_adaptive_runtime_startup_failure_precedes_wal_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An adaptive sidecar startup failure must not create a private WAL."""
    scenario = build_random_surface_scenario(
        scene_seed=16181,
        measurement_log_output_dir=tmp_path / "measurement-log",
        run_id="adaptive-startup-failure-test",
    )
    scenario_path = write_private_scenario(tmp_path / "scenario.json", scenario)
    isotope_names = tuple(sorted(scenario["isotopes"]))
    profile_id = scenario["metadata"]["experiment_profile_id"]
    logged_config = runtime_config()
    writer_started = False

    class FakeSceneDescription:
        """Expose only the source conversion required before runtime startup."""

        def to_point_sources(self) -> list[object]:
            """Return an empty source list for this startup-only test."""
            return []

    def fake_writer(*args: object, **kwargs: object) -> object:
        """Record any forbidden WAL creation before transport startup."""
        nonlocal writer_started
        del args, kwargs
        writer_started = True
        return object()

    def fail_runtime(*args: object, **kwargs: object) -> SimulationRuntime:
        """Inject one native sidecar startup failure."""
        del args, kwargs
        raise RuntimeError("injected native startup failure")

    profile = SimpleNamespace(
        profile_id=profile_id,
        candidate_isotopes=isotope_names,
    )
    monkeypatch.setattr(
        "runtime.adaptive.load_production_runtime_config",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "runtime.adaptive._validate_production_environment",
        lambda _environment, _config: (profile, None, ()),
    )
    monkeypatch.setattr(
        "runtime.adaptive._validate_production_scene",
        lambda *_args, **_kwargs: FakeSceneDescription(),
    )
    monkeypatch.setattr(
        "runtime.adaptive._validate_private_scene_variant",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "runtime.adaptive.estimator_neutral_runtime_config",
        lambda *_args, **_kwargs: logged_config,
    )
    monkeypatch.setattr("runtime.adaptive.repository_commit", lambda _root: "a" * 40)
    monkeypatch.setattr(
        "runtime.adaptive.repository_source_snapshot_sha256",
        lambda _root: "b" * 64,
    )
    monkeypatch.setattr(
        "runtime.adaptive.build_forward_model_manifest",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        "runtime.adaptive.production_native_execution_digests",
        lambda _config: ("c" * 64, "d" * 64, "e" * 64),
    )
    monkeypatch.setattr(
        "runtime.adaptive.production_runtime_config_sha256",
        lambda _config: "f" * 64,
    )
    monkeypatch.setattr("runtime.adaptive.create_simulation_runtime", fail_runtime)
    monkeypatch.setattr("runtime.session.MeasurementLogStreamWriter", fake_writer)

    with pytest.raises(RuntimeError, match="injected native startup failure"):
        AdaptiveRuntimeSession.open(scenario_path)

    assert writer_started is False
    assert not tuple(tmp_path.glob(".measurement-log.stream-*"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("environment_unknown", "environment fields differ"),
        ("environment_missing", "environment fields differ"),
        ("obstacle_grid_missing", "environment.obstacle_grid fields differ"),
        ("scene_unknown", "scene fields differ"),
        ("scene_missing", "scene fields differ"),
        ("usd_fallback", "use_config_usd_fallback=false"),
        ("author_mismatch", "author_obstacle_prims differs"),
        ("room_mismatch", "room_size_xyz differs"),
        ("obstacle_mismatch", "obstacle geometry differs"),
        ("external_obstacle_layout", "embedded obstacle_grid"),
    ),
)
def test_production_scenario_rejects_nested_defaults_and_schema_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """Production scenarios must fail before model or simulator preflight."""
    scenario = build_random_surface_scenario(
        scene_seed=17320,
        measurement_log_output_dir=tmp_path / "measurement-log",
        run_id=f"adaptive-strict-scenario-{mutation}",
    )
    environment = scenario["environment"]
    scene = scenario["scene"]
    assert isinstance(environment, dict)
    assert isinstance(scene, dict)
    if mutation == "environment_unknown":
        environment["num_particels"] = 4096
    elif mutation == "environment_missing":
        del environment["environment_model_id"]
    elif mutation == "obstacle_grid_missing":
        obstacle_grid = environment["obstacle_grid"]
        assert isinstance(obstacle_grid, dict)
        del obstacle_grid["transport_boxes_m"]
    elif mutation == "scene_unknown":
        scene["legacy_room_size"] = scene["room_size_xyz"]
    elif mutation == "scene_missing":
        del scene["obstacle_material"]
    elif mutation == "usd_fallback":
        scene["use_config_usd_fallback"] = True
    elif mutation == "author_mismatch":
        scene["author_obstacle_prims"] = False
    elif mutation == "room_mismatch":
        scene["room_size_xyz"] = [9.0, 15.0, 5.0]
    elif mutation == "obstacle_mismatch":
        scene["obstacle_cell_size_m"] = 0.5
    elif mutation == "external_obstacle_layout":
        scenario["obstacle_layout_path"] = "ignored-grid.json"
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(f"Unhandled mutation {mutation!r}.")
    scenario_path = write_private_scenario(tmp_path / "scenario.json", scenario)

    with pytest.raises(ValueError, match=message):
        AdaptiveRuntimeSession.open(scenario_path)

    assert not (tmp_path / "measurement-log").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("boolean_schema", "schema_version must be exact integer 1"),
        ("numeric_run_id", "run_id must be a nonempty JSON string"),
        ("wrong_backend", "backend must equal 'geant4'"),
        (
            "numeric_config_path",
            "runtime_config_path must be a nonempty JSON string",
        ),
        ("empty_output_dir", "output_dir must be a nonempty JSON string"),
        ("string_isotope_list", "isotopes must be a nonempty JSON array"),
        ("numeric_isotope", "isotopes must contain nonempty JSON strings"),
        ("duplicate_isotope", "isotopes must be unique"),
        ("metadata_array", "metadata must be a JSON object"),
    ),
)
def test_production_scenario_rejects_top_level_type_coercion(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """Production scenario values must retain their exact JSON types."""
    scenario = build_random_surface_scenario(
        scene_seed=34120,
        measurement_log_output_dir=tmp_path / "measurement-log",
        run_id=f"adaptive-strict-envelope-{mutation}",
    )
    if mutation == "boolean_schema":
        scenario["schema_version"] = True
    elif mutation == "numeric_run_id":
        scenario["run_id"] = 42
    elif mutation == "wrong_backend":
        scenario["backend"] = "analytic"
    elif mutation == "numeric_config_path":
        scenario["runtime_config_path"] = 7
    elif mutation == "empty_output_dir":
        scenario["output_dir"] = "   "
    elif mutation == "string_isotope_list":
        scenario["isotopes"] = "Co-60,Cs-137,Eu-154"
    elif mutation == "numeric_isotope":
        scenario["isotopes"] = [60, "Cs-137", "Eu-154"]
    elif mutation == "duplicate_isotope":
        scenario["isotopes"] = ["Co-60", "Co-60", "Eu-154"]
    elif mutation == "metadata_array":
        scenario["metadata"] = []
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(f"Unhandled mutation {mutation!r}.")
    scenario_path = write_private_scenario(tmp_path / "scenario.json", scenario)

    with pytest.raises((TypeError, ValueError), match=message):
        AdaptiveRuntimeSession.open(scenario_path)

    assert not (tmp_path / "measurement-log").exists()


@pytest.mark.parametrize(
    "invalid_value",
    (
        Path("/not/json-native"),
        np.int64(3),
        object(),
    ),
)
def test_production_scenario_rejects_non_json_metadata_values(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    """Private metadata must not coerce paths, NumPy scalars, or objects."""
    scenario = build_random_surface_scenario(
        scene_seed=51230,
        measurement_log_output_dir=tmp_path / "measurement-log",
        run_id="adaptive-strict-private-metadata",
        metadata={"invalid": invalid_value},
    )

    with pytest.raises(TypeError, match="exact finite JSON-native values"):
        _validate_production_scenario(scenario)


def test_copied_production_config_resolves_assets_from_runtime_repository(
    tmp_path: Path,
) -> None:
    """Moving a canonical config must not relocate repository model assets."""
    scenario = build_random_surface_scenario(
        scene_seed=14142,
        measurement_log_output_dir=tmp_path / "measurement-log",
        run_id="adaptive-moved-config-test",
    )
    source = Path(str(scenario["runtime_config_path"]))
    copied = tmp_path / "private_runs/ral_ablation/runtime_configs/runtime.json"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(source.read_bytes())
    scenario["runtime_config_path"] = copied.as_posix()
    scenario_path = write_private_scenario(tmp_path / "scenario.json", scenario)

    with pytest.raises(RuntimeError, match="independent all-64 holdout"):
        AdaptiveRuntimeSession.open(scenario_path)


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


@pytest.mark.parametrize(
    "invalid_request",
    (
        '{"type":"abort","type":"abort"}\n',
        '{"type":"step","dwell_time_s":NaN}\n',
    ),
)
def test_adaptive_protocol_rejects_non_strict_json_requests(
    monkeypatch: Any,
    invalid_request: str,
) -> None:
    """Duplicate keys and non-finite constants cannot cross the live protocol."""
    fake = _FakeAdaptiveSession()
    monkeypatch.setattr(
        AdaptiveRuntimeSession,
        "open",
        classmethod(lambda cls, path: fake),
    )

    with pytest.raises(ValueError, match="Strict JSON"):
        serve_adaptive_session(
            Path("scenario.json"),
            input_stream=StringIO(invalid_request),
            output_stream=StringIO(),
        )

    assert fake.closed is True


def test_adaptive_protocol_accepts_actions_incrementally(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The runtime protocol should receive one selected action, never a plan."""
    fake = _FakeAdaptiveSession()
    monkeypatch.setattr(
        AdaptiveRuntimeSession,
        "open",
        classmethod(lambda cls, path: fake),
    )
    step = {
        "type": "step",
        "action_id": 0,
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
        classmethod(lambda cls, path: fake),
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


def test_adaptive_socket_hides_private_scenario_from_client_arguments(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A socket client must receive events without a private scenario path."""
    fake = _FakeAdaptiveSession()
    monkeypatch.setattr(
        AdaptiveRuntimeSession,
        "open",
        classmethod(lambda cls, path: fake),
    )
    endpoint = tmp_path / "adaptive.sock"
    outcomes: list[int] = []
    failures: list[BaseException] = []

    def serve() -> None:
        """Run the blocking socket server for one client interaction."""
        try:
            outcomes.append(
                serve_adaptive_session_socket(
                    tmp_path / "private-scenario.json",
                    socket_path=endpoint,
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            failures.append(exc)

    thread = Thread(target=serve)
    thread.start()
    client = AdaptiveRuntimeClient.connect(endpoint, connect_timeout_s=2.0)
    ready = client.read_event()
    client.abort()
    thread.join(timeout=2.0)

    assert ready["context"]["run_id"] == "test"
    assert client.command == ["adaptive-session-socket", endpoint.as_posix()]
    assert "private-scenario" not in " ".join(client.command)
    assert outcomes == [0]
    assert failures == []
    assert not endpoint.exists()


def test_adaptive_client_cannot_open_a_private_scenario() -> None:
    """Estimator clients must use only the opaque runtime-owned socket."""
    with pytest.raises(TypeError, match=r"AdaptiveRuntimeClient\.connect"):
        AdaptiveRuntimeClient(
            Path("/private/scenario.json"),
            runtime_root=Path("/runtime"),
        )


def test_adaptive_client_abort_surfaces_missing_cleanup_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A production abort must not hide failure to clean the runtime WAL."""
    client = AdaptiveRuntimeClient.__new__(AdaptiveRuntimeClient)
    client._closed = False
    client._socket = None
    client.terminate_timeout_s = 1.0
    client.input = StringIO()
    client.output = StringIO()

    def fail_abort(payload: object) -> object:
        """Model a runtime that closes before acknowledging WAL deletion."""
        del payload
        raise RuntimeError("runtime cleanup failed")

    monkeypatch.setattr(client, "request", fail_abort)

    with pytest.raises(RuntimeError, match="runtime cleanup failed"):
        client.abort()
    assert client._closed is True
    assert client.input is None
    assert client.output is None


def test_adaptive_protocol_uses_private_prefix_for_cui_overlay(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Evaluation truth must not be emitted as an estimator-visible event."""
    fake = _FakeAdaptiveSession()
    monkeypatch.setattr(
        AdaptiveRuntimeSession,
        "open",
        classmethod(lambda cls, path: fake),
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
    assert private_events[0]["truth"]["true_sources"]["Cs-137"] == [[1.0, 1.0, 1.0]]


def test_adaptive_client_rejects_truth_before_writing_cui_request() -> None:
    """The estimator-facing client must not open the runtime truth channel."""
    response = {
        "type": "cui_overlay",
        "schema_version": 1,
        "truth": None,
    }
    client = AdaptiveRuntimeClient.__new__(AdaptiveRuntimeClient)
    client.input = StringIO()
    client.output = StringIO(ADAPTIVE_CUI_OVERLAY_PREFIX + json.dumps(response) + "\n")
    client.output_hook = lambda message: None

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
    client.output = StringIO(ADAPTIVE_CUI_OVERLAY_PREFIX + json.dumps(response) + "\n")
    client.output_hook = lambda message: None

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
    client.output = StringIO(ADAPTIVE_CUI_OVERLAY_PREFIX + json.dumps(response) + "\n")
    client.output_hook = lambda message: None

    with pytest.raises(ValueError, match="cannot receive realized truth"):
        client.request_cui_overlay(include_truth=False)
