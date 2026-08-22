"""Estimator-neutral parsing for forward-response conformance fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path

from measurement.obstacle_assets import material_mu_cm_inv
from measurement.obstacles import ObstacleGrid

from runtime.provenance import load_strict_json


FORWARD_CONFORMANCE_SCHEMA_VERSION = 1
FORWARD_CONFORMANCE_CASE_ORDER = (
    "isotope",
    "detector_pose",
    "fe_orientation",
    "pb_orientation",
    "source_point",
    "obstacle",
)
FORWARD_CONFORMANCE_UNITS = {
    "distance": "m",
    "live_time": "s",
    "source_strength": "detector_cps_1m",
}


class ForwardConformanceFixtureError(ValueError):
    """Report an invalid provider-neutral forward-response fixture."""


def _mapping(value: object, *, name: str) -> dict[str, object]:
    """Return one string-keyed JSON object with a path-specific error."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ForwardConformanceFixtureError(f"{name} must be a JSON object.")
    return dict(value)


def _sequence(value: object, *, name: str, allow_empty: bool = False) -> tuple[object, ...]:
    """Return one non-string JSON array, optionally allowing no entries."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ForwardConformanceFixtureError(f"{name} must be a JSON array.")
    result = tuple(value)
    if not result and not allow_empty:
        raise ForwardConformanceFixtureError(f"{name} must not be empty.")
    return result


def _identifier(value: object, *, name: str) -> str:
    """Return one nonempty identifier that cannot corrupt canonical case IDs."""
    if not isinstance(value, str) or not value or value != value.strip() or "|" in value:
        raise ForwardConformanceFixtureError(
            f"{name} must be a nonempty trimmed string without '|'."
        )
    return value


def _xyz(value: object, *, name: str) -> tuple[float, float, float]:
    """Return one exact finite XYZ triple in metres."""
    values = _sequence(value, name=name)
    if len(values) != 3:
        raise ForwardConformanceFixtureError(f"{name} must contain exactly three values.")
    result: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ForwardConformanceFixtureError(f"{name} values must be JSON numbers.")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ForwardConformanceFixtureError(f"{name} values must be finite.")
        result.append(numeric)
    return (result[0], result[1], result[2])


def _unique(values: Sequence[str], *, name: str) -> None:
    """Reject duplicate identifiers along one fixture axis."""
    if len(values) != len(set(values)):
        raise ForwardConformanceFixtureError(f"{name} identifiers must be unique.")


@dataclass(frozen=True, slots=True)
class ForwardConformanceDetectorPose:
    """Describe one detector pose and exposure duration."""

    pose_id: str
    xyz: tuple[float, float, float]
    live_time_s: float


@dataclass(frozen=True, slots=True)
class ForwardConformanceSourcePoint:
    """Describe one unit-strength continuous source position."""

    source_id: str
    xyz: tuple[float, float, float]
    surface_kind: str | None


@dataclass(frozen=True, slots=True)
class ForwardConformanceObstacleBox:
    """Describe one strictly positive-volume material transport box."""

    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]
    material: str


@dataclass(frozen=True, slots=True)
class ForwardConformanceObstacle:
    """Describe one alternative complete obstacle environment."""

    obstacle_id: str
    boxes: tuple[ForwardConformanceObstacleBox, ...]


@dataclass(frozen=True, slots=True)
class ForwardConformanceFixture:
    """Store immutable canonical axes shared by every estimator provider."""

    fixture_id: str
    isotopes: tuple[str, ...]
    detector_poses: tuple[ForwardConformanceDetectorPose, ...]
    fe_orientation_indices: tuple[int, ...]
    pb_orientation_indices: tuple[int, ...]
    source_points: tuple[ForwardConformanceSourcePoint, ...]
    obstacles: tuple[ForwardConformanceObstacle, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ForwardConformanceFixture":
        """Parse one exact schema-v1 fixture into immutable typed axes."""
        root = _mapping(payload, name="fixture")
        required = {
            "schema_version",
            "units",
            "isotopes",
            "detector_poses",
            "shield_program",
            "source_points",
            "obstacles",
            "required_case_order",
        }
        optional = {"fixture_id"}
        if set(root) - required - optional or required - set(root):
            raise ForwardConformanceFixtureError(
                "Fixture fields do not match the forward-conformance v1 contract."
            )
        if root["schema_version"] != FORWARD_CONFORMANCE_SCHEMA_VERSION:
            raise ForwardConformanceFixtureError("schema_version must be 1.")
        fixture_id = root.get("fixture_id", "forward-response-conformance-v1")
        if fixture_id != "forward-response-conformance-v1":
            raise ForwardConformanceFixtureError(
                "fixture_id must be forward-response-conformance-v1."
            )
        if _mapping(root["units"], name="units") != FORWARD_CONFORMANCE_UNITS:
            raise ForwardConformanceFixtureError(
                f"units must be exactly {FORWARD_CONFORMANCE_UNITS}."
            )
        case_order = tuple(
            _identifier(value, name="required_case_order[]")
            for value in _sequence(
                root["required_case_order"],
                name="required_case_order",
            )
        )
        if case_order != FORWARD_CONFORMANCE_CASE_ORDER:
            raise ForwardConformanceFixtureError(
                "required_case_order does not match the canonical axis order."
            )
        isotopes = tuple(
            _identifier(value, name="isotopes[]")
            for value in _sequence(root["isotopes"], name="isotopes")
        )
        _unique(isotopes, name="isotope")
        poses = _parse_detector_poses(root["detector_poses"])
        fe_indices, pb_indices = _parse_shield_program(root["shield_program"])
        sources = _parse_source_points(root["source_points"])
        obstacles = _parse_obstacles(root["obstacles"], isotopes=isotopes)
        return cls(
            fixture_id=str(fixture_id),
            isotopes=isotopes,
            detector_poses=poses,
            fe_orientation_indices=fe_indices,
            pb_orientation_indices=pb_indices,
            source_points=sources,
            obstacles=obstacles,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "ForwardConformanceFixture":
        """Load strict JSON and parse one forward-response fixture file."""
        payload = load_strict_json(path)
        if not isinstance(payload, Mapping):
            raise ForwardConformanceFixtureError("Fixture root must be a JSON object.")
        return cls.from_payload(payload)

    def obstacle_grid(self, obstacle_id: str) -> ObstacleGrid | None:
        """Build the runtime-owned transport grid for one obstacle variant."""
        matches = tuple(
            obstacle for obstacle in self.obstacles if obstacle.obstacle_id == obstacle_id
        )
        if len(matches) != 1:
            raise KeyError(f"Unknown conformance obstacle_id {obstacle_id!r}.")
        boxes = matches[0].boxes
        if not boxes:
            return None
        transport_boxes = tuple(
            (*box.min_xyz, *box.max_xyz)
            for box in boxes
        )
        materials = tuple(box.material for box in boxes)
        return ObstacleGrid(
            origin=(0.0, 0.0),
            cell_size=1.0,
            grid_shape=(0, 0),
            blocked_cells=(),
            transport_boxes_m=transport_boxes,
            transport_mu_by_isotope={
                isotope: tuple(
                    float(material_mu_cm_inv(material, isotope))
                    for material in materials
                )
                for isotope in self.isotopes
            },
            collision_boxes_m=transport_boxes,
        )

    def case_ids(self) -> tuple[str, ...]:
        """Return every canonical case ID in the required nested axis order."""
        return tuple(
            f"{isotope}|pose={pose.pose_id}|fe={fe_index:02d}|"
            f"pb={pb_index:02d}|source={source.source_id}|"
            f"obstacle={obstacle.obstacle_id}"
            for isotope in self.isotopes
            for pose in self.detector_poses
            for fe_index in self.fe_orientation_indices
            for pb_index in self.pb_orientation_indices
            for source in self.source_points
            for obstacle in self.obstacles
        )


def _parse_detector_poses(value: object) -> tuple[ForwardConformanceDetectorPose, ...]:
    """Parse detector-pose objects and reject duplicate identifiers."""
    result: list[ForwardConformanceDetectorPose] = []
    for index, raw in enumerate(_sequence(value, name="detector_poses")):
        pose = _mapping(raw, name=f"detector_poses[{index}]")
        if set(pose) != {"pose_id", "xyz", "live_time_s"}:
            raise ForwardConformanceFixtureError(
                f"detector_poses[{index}] has incompatible fields."
            )
        live_time = pose["live_time_s"]
        if isinstance(live_time, bool) or not isinstance(live_time, (int, float)):
            raise ForwardConformanceFixtureError("Detector live_time_s must be numeric.")
        duration = float(live_time)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ForwardConformanceFixtureError(
                "Detector live_time_s must be finite and positive."
            )
        result.append(
            ForwardConformanceDetectorPose(
                pose_id=_identifier(
                    pose["pose_id"],
                    name=f"detector_poses[{index}].pose_id",
                ),
                xyz=_xyz(pose["xyz"], name=f"detector_poses[{index}].xyz"),
                live_time_s=duration,
            )
        )
    _unique(tuple(item.pose_id for item in result), name="detector pose")
    return tuple(result)


def _parse_shield_program(value: object) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Parse ordered unique Fe/Pb orientation subsets in the supported range."""
    program = _mapping(value, name="shield_program")
    if set(program) != {
        "pairing",
        "fe_orientation_indices",
        "pb_orientation_indices",
    } or program["pairing"] != "cartesian_product":
        raise ForwardConformanceFixtureError(
            "shield_program must declare cartesian_product and Fe/Pb indices."
        )

    def indices(name: str) -> tuple[int, ...]:
        """Parse one exact orientation-index sequence."""
        result: list[int] = []
        for value in _sequence(program[name], name=f"shield_program.{name}"):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 7:
                raise ForwardConformanceFixtureError(
                    f"shield_program.{name} values must be integers in 0..7."
                )
            result.append(value)
        if len(result) != len(set(result)):
            raise ForwardConformanceFixtureError(
                f"shield_program.{name} values must be unique."
            )
        return tuple(result)

    return indices("fe_orientation_indices"), indices("pb_orientation_indices")


def _parse_source_points(value: object) -> tuple[ForwardConformanceSourcePoint, ...]:
    """Parse source-point objects with an optional descriptive surface kind."""
    result: list[ForwardConformanceSourcePoint] = []
    for index, raw in enumerate(_sequence(value, name="source_points")):
        source = _mapping(raw, name=f"source_points[{index}]")
        if set(source) not in (
            {"source_id", "xyz"},
            {"source_id", "xyz", "surface_kind"},
        ):
            raise ForwardConformanceFixtureError(
                f"source_points[{index}] has incompatible fields."
            )
        surface_kind = (
            None
            if "surface_kind" not in source
            else _identifier(
                source["surface_kind"],
                name=f"source_points[{index}].surface_kind",
            )
        )
        result.append(
            ForwardConformanceSourcePoint(
                source_id=_identifier(
                    source["source_id"],
                    name=f"source_points[{index}].source_id",
                ),
                xyz=_xyz(source["xyz"], name=f"source_points[{index}].xyz"),
                surface_kind=surface_kind,
            )
        )
    _unique(tuple(item.source_id for item in result), name="source point")
    return tuple(result)


def _parse_obstacles(
    value: object,
    *,
    isotopes: tuple[str, ...],
) -> tuple[ForwardConformanceObstacle, ...]:
    """Parse obstacle alternatives and authenticate every material/isotope pair."""
    result: list[ForwardConformanceObstacle] = []
    for obstacle_index, raw in enumerate(_sequence(value, name="obstacles")):
        obstacle = _mapping(raw, name=f"obstacles[{obstacle_index}]")
        if set(obstacle) != {"obstacle_id", "boxes"}:
            raise ForwardConformanceFixtureError(
                f"obstacles[{obstacle_index}] has incompatible fields."
            )
        boxes: list[ForwardConformanceObstacleBox] = []
        for box_index, raw_box in enumerate(
            _sequence(
                obstacle["boxes"],
                name=f"obstacles[{obstacle_index}].boxes",
                allow_empty=True,
            )
        ):
            box = _mapping(
                raw_box,
                name=f"obstacles[{obstacle_index}].boxes[{box_index}]",
            )
            if set(box) != {"min_xyz", "max_xyz", "material"}:
                raise ForwardConformanceFixtureError(
                    "Obstacle boxes require min_xyz, max_xyz, and material."
                )
            lower = _xyz(box["min_xyz"], name="obstacle box min_xyz")
            upper = _xyz(box["max_xyz"], name="obstacle box max_xyz")
            if any(high <= low for low, high in zip(lower, upper, strict=True)):
                raise ForwardConformanceFixtureError(
                    "Obstacle box max_xyz must exceed min_xyz on every axis."
                )
            material = _identifier(box["material"], name="obstacle box material")
            try:
                for isotope in isotopes:
                    material_mu_cm_inv(material, isotope)
            except (KeyError, TypeError, ValueError) as exc:
                raise ForwardConformanceFixtureError(
                    f"Obstacle material {material!r} is unavailable for the fixture isotopes."
                ) from exc
            boxes.append(
                ForwardConformanceObstacleBox(
                    min_xyz=lower,
                    max_xyz=upper,
                    material=material,
                )
            )
        result.append(
            ForwardConformanceObstacle(
                obstacle_id=_identifier(
                    obstacle["obstacle_id"],
                    name=f"obstacles[{obstacle_index}].obstacle_id",
                ),
                boxes=tuple(boxes),
            )
        )
    _unique(tuple(item.obstacle_id for item in result), name="obstacle")
    return tuple(result)


__all__ = [
    "FORWARD_CONFORMANCE_CASE_ORDER",
    "FORWARD_CONFORMANCE_SCHEMA_VERSION",
    "FORWARD_CONFORMANCE_UNITS",
    "ForwardConformanceDetectorPose",
    "ForwardConformanceFixture",
    "ForwardConformanceFixtureError",
    "ForwardConformanceObstacle",
    "ForwardConformanceObstacleBox",
    "ForwardConformanceSourcePoint",
]
