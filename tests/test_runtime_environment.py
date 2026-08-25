"""Tests for shared runtime obstacle environment setup."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime_environment import (
    attach_random_manchester_transport_geometry,
    build_runtime_obstacle_environment,
    normalize_environment_mode,
)
from measurement.obstacle_assets import validate_component_transport_contract
from measurement.geometry_family import (
    geometry_family_descriptor,
    randomized_training_geometry_parameters,
    validate_geometry_family_descriptor,
)
from measurement.obstacles import ObstacleGrid
from spectrum.transport_spectral import (
    DESIGNATED_HOLDOUT_SCENE_SEEDS,
    DESIGNATED_TRAINING_SCENE_SEEDS,
)


def test_random_runtime_obstacle_environment_is_in_memory(tmp_path: Path) -> None:
    """Random mode should build obstacles without writing the fixed-layout path."""
    obstacle_path = tmp_path / "random_unused.json"

    environment = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=obstacle_path,
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=7,
        passage_width_m=1.0,
    )

    assert environment.mode == "random"
    assert environment.grid is not None
    assert environment.grid.blocked_cells
    assert not obstacle_path.exists()
    assert environment.message is not None
    assert "passage_width_m=1.00" in environment.message


def test_random_runtime_environment_can_attach_transport_model(tmp_path: Path) -> None:
    """Runtime random obstacles should optionally expose PF transport boxes."""
    environment = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=tmp_path / "random_unused.json",
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=9,
        attach_known_transport=True,
        obstacle_height_m=2.0,
    )

    assert environment.grid is not None
    assert environment.known_obstacle_instances is not None
    assert environment.grid.transport_boxes_m
    assert environment.grid.collision_boxes_m
    summary = environment.asset_summary()
    assert summary is not None
    assert "nominal_min_transmission=" in summary


def test_component_transport_contract_rejects_solid_envelope(
    tmp_path: Path,
) -> None:
    """A hollow obstacle footprint must never replace its material components."""
    environment = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=tmp_path / "random_unused.json",
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=91,
        attach_known_transport=True,
        obstacle_height_m=2.0,
    )

    assert environment.grid is not None
    assert environment.known_obstacle_instances is not None
    first = environment.known_obstacle_instances[0]
    x0, x1, y0, y1 = first.footprint_xy
    solid_envelope = (x0, y0, 0.0, x1, y1, 2.0)
    invalid_grid = environment.grid.with_collision_model(
        boxes_m=(solid_envelope,),
    )

    with pytest.raises(ValueError, match="authored component boxes"):
        validate_component_transport_contract(
            invalid_grid,
            environment.known_obstacle_instances,
        )


def test_component_transport_contract_rejects_material_mu_drift(
    tmp_path: Path,
) -> None:
    """Serialized attenuation must match every authored component material."""
    environment = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=tmp_path / "random_unused.json",
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=92,
        attach_known_transport=True,
        obstacle_height_m=2.0,
    )

    assert environment.grid is not None
    assert environment.known_obstacle_instances is not None
    payload = environment.grid.to_dict()
    payload["transport_mu_by_isotope"]["Cs-137"][0] *= 0.5
    invalid_grid = ObstacleGrid.from_dict(payload)

    with pytest.raises(ValueError, match="component materials"):
        validate_component_transport_contract(
            invalid_grid,
            environment.known_obstacle_instances,
        )


def test_randomized_geometry_family_is_deterministic_and_in_domain(
    tmp_path: Path,
) -> None:
    """Training geometry variation must be reproducible and OOD-checkable."""
    room = (10.0, 20.0, 10.0)
    first = randomized_training_geometry_parameters(2026072701, room_size_xyz=room)
    second = randomized_training_geometry_parameters(2026072701, room_size_xyz=room)
    different = randomized_training_geometry_parameters(2026072702, room_size_xyz=room)
    assert first == second
    assert first != different
    environment = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=tmp_path / "unused.json",
        room_size_xyz=room,
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=2026072701,
        blocked_fraction=float(first["blocked_fraction"]),
        passage_width_m=float(first["passage_width_m"]),
        attach_known_transport=True,
        obstacle_height_m=float(first["obstacle_height_m"]),
    )
    assert environment.grid is not None
    assert environment.known_obstacle_instances is not None
    descriptor = geometry_family_descriptor(
        environment.grid,
        environment.known_obstacle_instances,
        room_size_xyz=room,
        passage_width_m=float(first["passage_width_m"]),
        target_blocked_fraction=float(first["blocked_fraction"]),
        obstacle_height_limit_m=float(first["obstacle_height_m"]),
    )
    validate_geometry_family_descriptor(descriptor, require_in_domain=True)
    outside = dict(descriptor)
    outside["passage_width_m"] = 10.0
    with pytest.raises(ValueError, match="outside"):
        validate_geometry_family_descriptor(outside, require_in_domain=True)


@pytest.mark.parametrize(
    "scene_seed",
    DESIGNATED_TRAINING_SCENE_SEEDS + DESIGNATED_HOLDOUT_SCENE_SEEDS,
)
def test_randomized_geometry_family_retains_obstacles_and_xy_access(
    tmp_path: Path,
    scene_seed: int,
) -> None:
    """Wide navigation corridors must not erase every physical obstacle."""
    room = (10.0, 20.0, 10.0)
    parameters = randomized_training_geometry_parameters(
        scene_seed,
        room_size_xyz=room,
    )
    start = (5.0, 10.0)
    environment = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=tmp_path / f"unused_{scene_seed}.json",
        room_size_xyz=room,
        detector_position_xy=start,
        obstacle_seed=scene_seed,
        blocked_fraction=float(parameters["blocked_fraction"]),
        passage_width_m=float(parameters["passage_width_m"]),
        attach_known_transport=True,
        obstacle_height_m=float(parameters["obstacle_height_m"]),
    )

    assert environment.grid is not None
    assert environment.known_obstacle_instances
    assert environment.grid.blocked_cells
    for goal in ((0.5, 0.5), (9.5, 0.5), (0.5, 19.5), (9.5, 19.5)):
        assert environment.grid.has_free_path(start, goal)


def test_shared_transport_attachment_matches_runtime_builder(tmp_path: Path) -> None:
    """Shared attachment should reproduce the runtime component union exactly."""
    base = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=tmp_path / "random_unused.json",
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=17,
        obstacle_height_m=2.5,
    )
    runtime = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=tmp_path / "random_unused.json",
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=17,
        attach_known_transport=True,
        obstacle_height_m=2.5,
        include_room_boundaries=True,
        room_boundary_thickness_m=0.2,
    )

    assert base.grid is not None
    shared_grid, shared_instances = attach_random_manchester_transport_geometry(
        base.grid,
        room_size_xyz=(10.0, 20.0, 10.0),
        obstacle_height_m=2.5,
        rng_seed=17,
        include_room_boundaries=True,
        room_boundary_thickness_m=0.2,
    )

    assert runtime.grid is not None
    assert runtime.known_obstacle_instances == shared_instances
    assert runtime.grid.transport_boxes_m == shared_grid.transport_boxes_m
    assert runtime.grid.transport_mu_by_isotope == (
        shared_grid.transport_mu_by_isotope
    )
    assert runtime.grid.collision_boxes_m == shared_grid.collision_boxes_m


def test_random_runtime_environment_can_attach_room_boundary_transport(
    tmp_path: Path,
) -> None:
    """Runtime transport model should include authored room boundaries when requested."""
    base = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=tmp_path / "random_unused.json",
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=10,
        attach_known_transport=True,
        obstacle_height_m=2.0,
        include_room_boundaries=False,
    )
    with_boundaries = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=tmp_path / "random_unused.json",
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=10,
        attach_known_transport=True,
        obstacle_height_m=2.0,
        include_room_boundaries=True,
        room_boundary_thickness_m=0.1,
    )

    assert base.grid is not None
    assert with_boundaries.grid is not None
    assert len(with_boundaries.grid.transport_boxes_m) == (
        len(base.grid.transport_boxes_m) + 6
    )
    assert with_boundaries.grid.collision_boxes_m == base.grid.collision_boxes_m
    assert any(box[2] < 0.0 for box in with_boundaries.grid.transport_boxes_m)
    assert any(box[5] > 10.0 for box in with_boundaries.grid.transport_boxes_m)


def test_fixed_runtime_obstacle_environment_uses_layout_file(tmp_path: Path) -> None:
    """Fixed mode should load or create the requested obstacle layout file."""
    obstacle_path = tmp_path / "fixed.json"

    environment = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="fixed",
        obstacle_layout_path=obstacle_path,
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
        obstacle_seed=11,
    )

    assert environment.mode == "fixed"
    assert environment.grid is not None
    assert obstacle_path.exists()
    assert environment.layout_path == obstacle_path


def test_runtime_obstacle_environment_can_be_disabled(tmp_path: Path) -> None:
    """A None obstacle path should preserve the explicit no-obstacle behavior."""
    environment = build_runtime_obstacle_environment(
        root=tmp_path,
        environment_mode="random",
        obstacle_layout_path=None,
        room_size_xyz=(10.0, 20.0, 10.0),
        detector_position_xy=(1.0, 1.0),
    )

    assert environment.mode == "random"
    assert environment.grid is None
    assert environment.message is None


def test_invalid_runtime_environment_mode_is_rejected() -> None:
    """Unknown obstacle environment modes should fail early."""
    with pytest.raises(ValueError, match="Unknown environment_mode"):
        normalize_environment_mode("unsupported")
