"""End-to-end checks for the shared Python/native octant-pose contract."""

import numpy as np
import pytest

from measurement.shielding import (
    SHIELD_POSE_CONTRACT_ID,
    SHIELD_POSE_CONTRACT_SHA256,
    physical_shield_normal_from_orientation_index,
)
from sim.geant4_app.engine import (
    Geant4StepRequest,
    validate_native_action_identity,
)
from sim.geant4_app.io_format import NATIVE_ACTION_IDENTITY_CONTRACT_ID
from sim.isaacsim_app.robot_controller import (
    RobotController,
    _movement_yaw_rad,
    _rotation_between_vectors_wxyz,
)
from sim.isaacsim_app.scene_builder import StagePrimPaths
from sim.isaacsim_app.stage_backend import FakeStageBackend
from sim.protocol import SimulationCommand
from sim.shield_geometry import shield_normal_from_quaternion_wxyz


@pytest.mark.parametrize(
    "invalid",
    ((0.0, 0.0, 0.0), (np.nan, 0.0, 1.0), (np.inf, 0.0, 1.0)),
)
def test_robot_rotation_rejects_degenerate_directions(
    invalid: tuple[float, float, float],
) -> None:
    """Isaac shield placement must not substitute a direction for invalid input."""
    with pytest.raises(ValueError, match="Direction vectors"):
        _rotation_between_vectors_wxyz(invalid, (1.0, 0.0, 0.0))


def test_vertical_robot_motion_preserves_the_commanded_yaw() -> None:
    """A vertical move has no planar heading and keeps its explicit command yaw."""
    assert _movement_yaw_rad(
        np.asarray((1.0, 2.0, 0.5)),
        np.asarray((1.0, 2.0, 1.5)),
        preserved_yaw_rad=0.75,
    ) == pytest.approx(0.75)


@pytest.mark.parametrize("base_yaw_rad", [0.0, 0.37, -1.2, np.pi])
def test_all_64_robot_pairs_have_world_fixed_physical_normals(
    base_yaw_rad: float,
) -> None:
    """Every pair must retain its world octant under arbitrary robot yaw."""
    stage = FakeStageBackend()
    stage.open_stage()
    paths = StagePrimPaths()
    controller = RobotController(stage, paths, animation_time_scale=0.0)
    controller.reset()
    for pair_id in range(64):
        fe_index = pair_id // 8
        pb_index = pair_id % 8
        controller.apply_command(
            SimulationCommand(
                step_id=pair_id,
                target_pose_xyz=(1.0, 1.0, 0.5),
                target_base_yaw_rad=float(base_yaw_rad),
                fe_orientation_index=fe_index,
                pb_orientation_index=pb_index,
                dwell_time_s=1.0,
            )
        )
        for kind, path, index in (
            ("fe", paths.fe_shield_path, fe_index),
            ("pb", paths.pb_shield_path, pb_index),
        ):
            pose = stage.get_world_pose(path)
            actual = np.asarray(
                shield_normal_from_quaternion_wxyz(pose.orientation_wxyz),
                dtype=float,
            )
            expected = physical_shield_normal_from_orientation_index(index)
            assert np.allclose(actual, expected, rtol=0.0, atol=1.0e-8), kind


def test_final_yaw_change_recomputes_child_shield_rotations() -> None:
    """Final robot yaw must not rotate the commanded world shield octants."""
    stage = FakeStageBackend()
    stage.open_stage()
    paths = StagePrimPaths()
    controller = RobotController(stage, paths, animation_time_scale=0.0)
    controller.reset()
    controller.apply_command(
        SimulationCommand(
            step_id=0,
            target_pose_xyz=(4.0, 1.0, 1.5),
            target_base_yaw_rad=1.1,
            fe_orientation_index=2,
            pb_orientation_index=5,
            dwell_time_s=1.0,
        )
    )
    for path, index in (
        (paths.fe_shield_path, 2),
        (paths.pb_shield_path, 5),
    ):
        pose = stage.get_world_pose(path)
        actual = np.asarray(
            shield_normal_from_quaternion_wxyz(pose.orientation_wxyz),
            dtype=float,
        )
        expected = physical_shield_normal_from_orientation_index(index)
        assert np.allclose(actual, expected, rtol=0.0, atol=1.0e-8)


def test_geant4_request_rejects_wrong_pair_rotation() -> None:
    """A pair ID and a physically different quaternion must fail closed."""
    request = Geant4StepRequest(
        step_id=0,
        dwell_time_s=1.0,
        seed=1,
        detector_pose_xyz=(0.0, 0.0, 0.0),
        detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fe_shield_pose_xyz=(0.0, 0.0, 0.0),
        fe_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        pb_shield_pose_xyz=(0.0, 0.0, 0.0),
        pb_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        shield_pose_contract_id=SHIELD_POSE_CONTRACT_ID,
        fe_orientation_index=0,
        pb_orientation_index=0,
    )
    with pytest.raises(ValueError, match="does not place local"):
        request.resolved_orientation_indices()


def test_geant4_request_infers_identity_as_incoming_octant_seven() -> None:
    """Identity placement has physical +++ and incoming --- semantics."""
    request = Geant4StepRequest(
        step_id=0,
        dwell_time_s=1.0,
        seed=1,
        detector_pose_xyz=(0.0, 0.0, 0.0),
        detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fe_shield_pose_xyz=(0.0, 0.0, 0.0),
        fe_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        pb_shield_pose_xyz=(0.0, 0.0, 0.0),
        pb_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    assert request.resolved_orientation_indices() == (7, 7)


def test_geant4_request_rejects_wrong_contract_hash() -> None:
    """A matching name with different contract content must fail closed."""
    request = Geant4StepRequest(
        step_id=0,
        dwell_time_s=1.0,
        seed=1,
        detector_pose_xyz=(0.0, 0.0, 0.0),
        detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fe_shield_pose_xyz=(0.0, 0.0, 0.0),
        fe_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        pb_shield_pose_xyz=(0.0, 0.0, 0.0),
        pb_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        shield_pose_contract_id=SHIELD_POSE_CONTRACT_ID,
        shield_pose_contract_sha256="0" * len(SHIELD_POSE_CONTRACT_SHA256),
    )
    with pytest.raises(ValueError, match="hash"):
        request.resolved_orientation_indices()


def _native_action_metadata(request: Geant4StepRequest) -> dict[str, object]:
    """Return one complete native action echo for validator tests."""
    fe_index, pb_index = request.resolved_orientation_indices()
    metadata: dict[str, object] = {
        "native_action_contract_id": NATIVE_ACTION_IDENTITY_CONTRACT_ID,
        "native_action_sha256": request.action_identity_sha256(),
        "native_action_step_id": request.step_id,
        "native_action_seed": request.seed,
        "native_action_dwell_time_s": request.dwell_time_s,
        "native_action_fe_orientation_index": fe_index,
        "native_action_pb_orientation_index": pb_index,
    }
    for prefix, position, quaternion in (
        ("detector", request.detector_pose_xyz, request.detector_quat_wxyz),
        ("fe_shield", request.fe_shield_pose_xyz, request.fe_shield_quat_wxyz),
        ("pb_shield", request.pb_shield_pose_xyz, request.pb_shield_quat_wxyz),
    ):
        metadata.update(
            {
                f"native_action_{prefix}_pose_{axis}": value
                for axis, value in zip("xyz", position, strict=True)
            }
        )
        metadata.update(
            {
                f"native_action_{prefix}_quat_{axis}": value
                for axis, value in zip("wxyz", quaternion, strict=True)
            }
        )
    return metadata


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("native_action_sha256", "0" * 64),
        ("native_action_step_id", 8),
        ("native_action_detector_pose_x", 9.0),
        ("native_action_fe_shield_quat_w", None),
    ),
)
def test_native_action_identity_rejects_every_mismatch(
    field: str,
    replacement: object,
) -> None:
    """A response cannot be rebound to another step, pose, or action digest."""
    request = Geant4StepRequest(
        step_id=7,
        dwell_time_s=20.0,
        seed=(1 << 62) + 91,
        detector_pose_xyz=(1.0, 2.0, 3.0),
        detector_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        fe_shield_pose_xyz=(1.0, 2.0, 3.0),
        fe_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        pb_shield_pose_xyz=(1.0, 2.0, 3.0),
        pb_shield_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    metadata = _native_action_metadata(request)
    validate_native_action_identity(metadata, request)
    if replacement is None:
        metadata.pop(field)
    else:
        metadata[field] = replacement

    with pytest.raises(RuntimeError, match="Native Geant4 .*action"):
        validate_native_action_identity(metadata, request)
