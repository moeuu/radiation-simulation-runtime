"""Versioned randomized geometry-family and applicability contracts."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from measurement.obstacle_assets import KnownObstacleInstance
    from measurement.obstacles import ObstacleGrid


GEOMETRY_FAMILY_ID = "random_manchester_hollow_components_v2"
GEOMETRY_FAMILY_SCHEMA_VERSION = 2
GEOMETRY_GENERATOR_ALGORITHM_ID = (
    "hollow_templates_material_components_xy_backbone_corridor_v2"
)
TRAINING_TARGET_BLOCKED_FRACTION_RANGE = (0.25, 0.55)
TRAINING_REALIZED_BLOCKED_FRACTION_RANGE = (0.005, 0.55)
TRAINING_PASSAGE_WIDTH_M_RANGE = (1.5, 3.5)
TRAINING_OBSTACLE_HEIGHT_LIMIT_FRACTION_RANGE = (0.15, 0.95)
TRAINING_REALIZED_COMPONENT_HEIGHT_FRACTION_RANGE = (0.05, 0.25)
ALLOWED_COMPONENT_MATERIALS = (
    "aluminum",
    "concrete",
    "steel",
    "water",
)


def _canonical_sha256(payload: object) -> str:
    """Return a deterministic SHA-256 for one JSON-compatible payload."""
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def randomized_training_geometry_parameters(
    scene_seed: int,
    *,
    room_size_xyz: Sequence[float],
) -> Mapping[str, float]:
    """Draw deterministic geometry-family parameters for one training scene."""
    room = tuple(float(value) for value in room_size_xyz)
    if len(room) != 3 or any(not np.isfinite(value) or value <= 0.0 for value in room):
        raise ValueError("Geometry-family room dimensions must be positive.")
    seed_material = hashlib.sha256(
        f"{GEOMETRY_FAMILY_ID}:{int(scene_seed)}".encode("ascii")
    ).digest()
    rng = np.random.Generator(
        np.random.Philox(int.from_bytes(seed_material[:8], "little"))
    )
    blocked = rng.uniform(*TRAINING_TARGET_BLOCKED_FRACTION_RANGE)
    passage = rng.uniform(*TRAINING_PASSAGE_WIDTH_M_RANGE)
    height_fraction = rng.uniform(
        *TRAINING_OBSTACLE_HEIGHT_LIMIT_FRACTION_RANGE
    )
    return {
        "blocked_fraction": float(blocked),
        "passage_width_m": float(passage),
        "obstacle_height_m": float(height_fraction * room[2]),
    }


def geometry_family_applicability_contract() -> Mapping[str, object]:
    """Return the predeclared applicability domain for trained corrections."""
    return {
        "schema_version": GEOMETRY_FAMILY_SCHEMA_VERSION,
        "geometry_family_id": GEOMETRY_FAMILY_ID,
        "generator_algorithm_id": GEOMETRY_GENERATOR_ALGORITHM_ID,
        "transport_representation": "explicit_material_component_boxes",
        "target_blocked_fraction_range": list(
            TRAINING_TARGET_BLOCKED_FRACTION_RANGE
        ),
        "realized_blocked_fraction_range": list(
            TRAINING_REALIZED_BLOCKED_FRACTION_RANGE
        ),
        "passage_width_m_range": list(TRAINING_PASSAGE_WIDTH_M_RANGE),
        "obstacle_height_limit_fraction_range": list(
            TRAINING_OBSTACLE_HEIGHT_LIMIT_FRACTION_RANGE
        ),
        "realized_component_height_fraction_range": list(
            TRAINING_REALIZED_COMPONENT_HEIGHT_FRACTION_RANGE
        ),
        "allowed_component_materials": list(ALLOWED_COMPONENT_MATERIALS),
        "out_of_domain_policy": "reject_empirical_discrepancy",
    }


GEOMETRY_FAMILY_APPLICABILITY_SHA256 = _canonical_sha256(
    geometry_family_applicability_contract()
)


def geometry_family_descriptor(
    grid: ObstacleGrid,
    instances: Sequence[KnownObstacleInstance],
    *,
    room_size_xyz: Sequence[float],
    passage_width_m: float,
    target_blocked_fraction: float,
    obstacle_height_limit_m: float,
) -> Mapping[str, object]:
    """Describe one realized component environment without truth information."""
    room = tuple(float(value) for value in room_size_xyz)
    realized_instances = tuple(instances)
    materials = sorted(
        {
            str(component.material)
            for instance in realized_instances
            for component in instance.components
        }
    )
    template_names = sorted({instance.template for instance in realized_instances})
    component_boxes = [
        [float(value) for value in component.box_m]
        for instance in realized_instances
        for component in instance.components
    ]
    maximum_top = max((box[5] for box in component_boxes), default=0.0)
    descriptor: dict[str, object] = {
        "schema_version": GEOMETRY_FAMILY_SCHEMA_VERSION,
        "geometry_family_id": GEOMETRY_FAMILY_ID,
        "generator_algorithm_id": GEOMETRY_GENERATOR_ALGORITHM_ID,
        "transport_representation": "explicit_material_component_boxes",
        "room_size_xyz_m": list(room),
        "cell_size_m": float(grid.cell_size),
        "target_blocked_fraction": float(target_blocked_fraction),
        "realized_blocked_fraction": float(grid.blocked_fraction),
        "passage_width_m": float(passage_width_m),
        "obstacle_height_limit_fraction": float(
            obstacle_height_limit_m / room[2]
        ),
        "realized_max_component_height_fraction": (
            float(maximum_top / room[2]) if room[2] > 0.0 else 0.0
        ),
        "instance_count": len(realized_instances),
        "transport_component_count": len(component_boxes),
        "template_names": template_names,
        "component_materials": materials,
        "component_geometry_sha256": _canonical_sha256(component_boxes),
        "applicability_contract_sha256": (
            GEOMETRY_FAMILY_APPLICABILITY_SHA256
        ),
    }
    return descriptor


def validate_geometry_family_descriptor(
    descriptor: Mapping[str, object],
    *,
    require_in_domain: bool,
) -> None:
    """Validate one realized descriptor and optionally require applicability."""
    if (
        not isinstance(descriptor, Mapping)
        or descriptor.get("schema_version") != GEOMETRY_FAMILY_SCHEMA_VERSION
        or descriptor.get("geometry_family_id") != GEOMETRY_FAMILY_ID
        or descriptor.get("generator_algorithm_id")
        != GEOMETRY_GENERATOR_ALGORITHM_ID
        or descriptor.get("transport_representation")
        != "explicit_material_component_boxes"
        or descriptor.get("applicability_contract_sha256")
        != GEOMETRY_FAMILY_APPLICABILITY_SHA256
    ):
        raise ValueError("Geometry-family descriptor identity is invalid.")
    try:
        target_blocked = float(descriptor["target_blocked_fraction"])
        realized_blocked = float(descriptor["realized_blocked_fraction"])
        passage = float(descriptor["passage_width_m"])
        height_limit_fraction = float(
            descriptor["obstacle_height_limit_fraction"]
        )
        realized_height_fraction = float(
            descriptor["realized_max_component_height_fraction"]
        )
        instance_count = int(descriptor["instance_count"])
        component_count = int(descriptor["transport_component_count"])
        materials = tuple(str(value) for value in descriptor["component_materials"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Geometry-family descriptor fields are invalid.") from exc
    if (
        not np.isfinite(target_blocked)
        or not np.isfinite(realized_blocked)
        or not np.isfinite(passage)
        or not np.isfinite(height_limit_fraction)
        or not np.isfinite(realized_height_fraction)
        or instance_count <= 0
        or component_count < instance_count
        or not materials
        or not set(materials).issubset(ALLOWED_COMPONENT_MATERIALS)
    ):
        raise ValueError("Geometry-family descriptor values are invalid.")
    if not require_in_domain:
        return
    checks = (
        TRAINING_TARGET_BLOCKED_FRACTION_RANGE[0]
        <= target_blocked
        <= TRAINING_TARGET_BLOCKED_FRACTION_RANGE[1],
        TRAINING_REALIZED_BLOCKED_FRACTION_RANGE[0]
        <= realized_blocked
        <= TRAINING_REALIZED_BLOCKED_FRACTION_RANGE[1],
        TRAINING_PASSAGE_WIDTH_M_RANGE[0]
        <= passage
        <= TRAINING_PASSAGE_WIDTH_M_RANGE[1],
        TRAINING_OBSTACLE_HEIGHT_LIMIT_FRACTION_RANGE[0]
        <= height_limit_fraction
        <= TRAINING_OBSTACLE_HEIGHT_LIMIT_FRACTION_RANGE[1],
        TRAINING_REALIZED_COMPONENT_HEIGHT_FRACTION_RANGE[0]
        <= realized_height_fraction
        <= TRAINING_REALIZED_COMPONENT_HEIGHT_FRACTION_RANGE[1],
    )
    if not all(checks):
        raise ValueError(
            "Environment is outside the randomized geometry-family training "
            "domain; empirical discrepancy must not be applied."
        )
