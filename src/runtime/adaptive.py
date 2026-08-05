"""Estimator-neutral interactive acquisition over a private runtime scenario."""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from scipy.stats import qmc

from measurement.kernels import ShieldParams
from measurement.obstacles import ObstacleGrid
from measurement.shielding import generate_octant_orientations
from runtime.forward_model_manifest import (
    SOURCE_RATE_SEMANTICS,
    build_forward_model_manifest,
)
from runtime.measurement_log import (
    MEASUREMENT_LOG_SCHEMA_VERSION,
    MeasurementLog,
    MeasurementLogRecord,
    MeasurementLogStreamWriter,
)
from runtime.provenance import (
    canonical_json_bytes,
    repository_commit,
    repository_source_snapshot_sha256,
)
from runtime.records import RunContext, validate_truth_free_estimator_input
from runtime.session import (
    AcquisitionAction,
    ObservationSession,
    estimator_neutral_runtime_config,
)
from sim.isaacsim_app.scene_builder import build_scene_description
from sim.protocol import SimulationCommand
from sim.runtime import create_simulation_runtime, load_runtime_config

ADAPTIVE_EVENT_PREFIX = "adaptive-session "
ADAPTIVE_CUI_OVERLAY_PREFIX = "adaptive-cui-overlay "
_SCENARIO_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "backend",
        "runtime_config_path",
        "output_dir",
        "environment",
        "scene",
        "isotopes",
        "metadata",
        "obstacle_layout_path",
    }
)
_STEP_FIELDS = frozenset(
    {
        "type",
        "candidate_index",
        "fe_orientation_index",
        "pb_orientation_index",
        "dwell_time_s",
        "station_id",
        "station_complete",
    }
)
_REFINE_FIELDS = frozenset({"type", "candidate_indices"})
_CUI_OVERLAY_FIELDS = frozenset({"type", "include_truth"})
_PRIVATE_SCENE_PROFILE_COUNTS = {
    "ral-mix9": {"Co-60": 3, "Cs-137": 4, "Eu-154": 2},
    "ral-cs4-co3-eu0": {"Co-60": 3, "Cs-137": 4, "Eu-154": 0},
}
_ADAPTIVE_MEASUREMENT_FIELDS = frozenset(
    {
        "candidate_count",
        "candidate_seed",
        "detector_height_min_m",
        "detector_height_max_m",
        "local_refinement_count",
        "local_refinement_radius_m",
        "base_radius_m",
        "base_height_m",
        "mast_radius_m",
        "head_radius_m",
        "transport_height_m",
        "horizontal_speed_m_s",
        "vertical_speed_m_s",
        "settling_time_s",
        "shield_angular_speed_rad_s",
    }
)


def _finite_number(
    value: object,
    *,
    field_name: str,
    minimum: float | None = None,
    positive: bool = False,
) -> float:
    """Return one finite real value satisfying the requested lower bound."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    if positive and parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return parsed


def _exact_integer(value: object, *, field_name: str, minimum: int) -> int:
    """Return one exact JSON integer at or above a lower bound."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return value


@dataclass(frozen=True, slots=True)
class AdaptiveMotionConfig:
    """Describe runtime-owned detector assembly, sampling, and motion timing."""

    candidate_count: int
    candidate_seed: int
    detector_height_min_m: float
    detector_height_max_m: float
    local_refinement_count: int
    local_refinement_radius_m: float
    base_radius_m: float
    base_height_m: float
    mast_radius_m: float
    head_radius_m: float
    transport_height_m: float
    horizontal_speed_m_s: float
    vertical_speed_m_s: float
    settling_time_s: float
    shield_angular_speed_rad_s: float

    @classmethod
    def from_inputs(
        cls,
        environment: Mapping[str, Any],
        runtime_config: Mapping[str, Any] | None,
    ) -> "AdaptiveMotionConfig":
        """Resolve a strict motion configuration from truth-free runtime inputs."""
        raw = environment.get("adaptive_measurement", {})
        if not isinstance(raw, Mapping):
            raise TypeError("environment.adaptive_measurement must be an object.")
        unknown = sorted(set(raw) - _ADAPTIVE_MEASUREMENT_FIELDS)
        if unknown:
            raise ValueError(
                f"environment.adaptive_measurement contains unknown fields: {unknown}."
            )
        runtime = {} if runtime_config is None else runtime_config
        detector = runtime.get("detector_model", {})
        if not isinstance(detector, Mapping):
            raise TypeError("runtime detector_model must be an object.")
        crystal_radius = _finite_number(
            detector.get("crystal_radius_m", 0.038),
            field_name="detector_model.crystal_radius_m",
            positive=True,
        )
        housing = _finite_number(
            detector.get("housing_thickness_m", 0.0015),
            field_name="detector_model.housing_thickness_m",
            minimum=0.0,
        )
        head_radius = _finite_number(
            raw.get("head_radius_m", crystal_radius + housing),
            field_name="adaptive_measurement.head_radius_m",
            positive=True,
        )
        base_height = _finite_number(
            raw.get("base_height_m", 0.2),
            field_name="adaptive_measurement.base_height_m",
            positive=True,
        )
        size_z = _finite_number(
            environment.get("size_z", 0.0),
            field_name="environment.size_z",
            positive=True,
        )
        minimum_height = _finite_number(
            raw.get("detector_height_min_m", base_height + head_radius),
            field_name="adaptive_measurement.detector_height_min_m",
            minimum=base_height + head_radius,
        )
        maximum_height = _finite_number(
            raw.get("detector_height_max_m", size_z - head_radius),
            field_name="adaptive_measurement.detector_height_max_m",
            minimum=minimum_height,
        )
        if maximum_height + head_radius > size_z + 1.0e-12:
            raise ValueError("Maximum detector height places the head above the room.")
        transport_height = _finite_number(
            raw.get("transport_height_m", minimum_height),
            field_name="adaptive_measurement.transport_height_m",
            minimum=minimum_height,
        )
        if transport_height > maximum_height:
            raise ValueError("transport_height_m must not exceed the height maximum.")
        base_radius = _finite_number(
            raw.get("base_radius_m", 0.2),
            field_name="adaptive_measurement.base_radius_m",
            positive=True,
        )
        mast_radius = _finite_number(
            raw.get("mast_radius_m", 0.03),
            field_name="adaptive_measurement.mast_radius_m",
            minimum=0.0,
        )
        if mast_radius > base_radius:
            raise ValueError("mast_radius_m must not exceed base_radius_m.")
        return cls(
            candidate_count=_exact_integer(
                raw.get("candidate_count", 256),
                field_name="adaptive_measurement.candidate_count",
                minimum=8,
            ),
            candidate_seed=_exact_integer(
                raw.get("candidate_seed", 0),
                field_name="adaptive_measurement.candidate_seed",
                minimum=0,
            ),
            detector_height_min_m=minimum_height,
            detector_height_max_m=maximum_height,
            local_refinement_count=_exact_integer(
                raw.get("local_refinement_count", 64),
                field_name="adaptive_measurement.local_refinement_count",
                minimum=0,
            ),
            local_refinement_radius_m=_finite_number(
                raw.get("local_refinement_radius_m", 0.5),
                field_name="adaptive_measurement.local_refinement_radius_m",
                minimum=0.0,
            ),
            base_radius_m=base_radius,
            base_height_m=base_height,
            mast_radius_m=mast_radius,
            head_radius_m=head_radius,
            transport_height_m=transport_height,
            horizontal_speed_m_s=_finite_number(
                raw.get("horizontal_speed_m_s", 0.5),
                field_name="adaptive_measurement.horizontal_speed_m_s",
                positive=True,
            ),
            vertical_speed_m_s=_finite_number(
                raw.get("vertical_speed_m_s", 0.25),
                field_name="adaptive_measurement.vertical_speed_m_s",
                positive=True,
            ),
            settling_time_s=_finite_number(
                raw.get("settling_time_s", 1.0),
                field_name="adaptive_measurement.settling_time_s",
                minimum=0.0,
            ),
            shield_angular_speed_rad_s=_finite_number(
                raw.get("shield_angular_speed_rad_s", math.pi / 4.0),
                field_name="adaptive_measurement.shield_angular_speed_rad_s",
                positive=True,
            ),
        )


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one strict JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def _validate_private_scene_profile(scene: object, profile: str | None) -> None:
    """Validate a named source-cardinality profile inside the private runtime."""
    if profile is None:
        return
    if profile not in _PRIVATE_SCENE_PROFILE_COUNTS:
        raise ValueError(f"Unknown adaptive private-scene profile: {profile!r}.")
    counts: dict[str, int] = {}
    for source in scene.sources:
        counts[source.isotope] = counts.get(source.isotope, 0) + 1
    expected = _PRIVATE_SCENE_PROFILE_COUNTS[profile]
    normalized = {
        isotope: int(counts.get(isotope, 0))
        for isotope in expected
    }
    unknown = set(counts) - set(expected)
    if normalized != expected or unknown:
        raise ValueError(
            f"{profile} private scene must contain exactly "
            + ", ".join(
                f"{isotope} x{count}"
                for isotope, count in sorted(expected.items())
            )
            + "."
        )


def _cui_truth_overlay(scene: object) -> dict[str, object]:
    """Return evaluation-only source truth for private CUI rendering."""
    true_sources: dict[str, list[list[float]]] = {}
    true_strengths: dict[str, list[float]] = {}
    for source in getattr(scene, "sources", []):
        isotope = str(source.isotope)
        position = np.asarray(source.position_xyz, dtype=float).reshape(3)
        if np.any(~np.isfinite(position)):
            raise ValueError("CUI truth overlay requires finite source positions.")
        true_sources.setdefault(isotope, []).append(
            [float(value) for value in position]
        )
        true_strengths.setdefault(isotope, []).append(
            float(source.intensity_cps_1m)
        )
    return {
        "schema_version": 1,
        "semantics": "evaluation_cui_overlay_only_not_estimator_input",
        "true_sources": true_sources,
        "true_strengths": true_strengths,
    }


def cui_truth_overlay_from_scene(scene: object) -> dict[str, object]:
    """Return private evaluation truth for CUI callers only."""
    return _cui_truth_overlay(scene)


def _initial_detector_pose(
    environment: Mapping[str, Any],
) -> tuple[float, float, float]:
    """Resolve the truth-free detector starting pose from the environment."""
    raw = environment.get("detector_position")
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(
            "Adaptive environment requires detector_position with 3 values."
        )
    pose = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) for value in pose):
        raise ValueError("environment.detector_position must be finite.")
    return pose


def _obstacle_grid(
    environment: Mapping[str, Any],
    obstacle_layout_path: object,
    *,
    runtime_root: Path,
) -> ObstacleGrid | None:
    """Load estimator-neutral traversability geometry owned by the runtime."""
    embedded = environment.get("obstacle_grid")
    if embedded is not None:
        if not isinstance(embedded, dict):
            raise TypeError("environment.obstacle_grid must be an object or null.")
        return ObstacleGrid.from_dict(embedded)
    if obstacle_layout_path is None:
        return None
    if not isinstance(obstacle_layout_path, str) or not obstacle_layout_path:
        raise TypeError("obstacle_layout_path must be a nonempty string or null.")
    path = Path(obstacle_layout_path)
    if not path.is_absolute():
        path = runtime_root / path
    return ObstacleGrid.load(path.resolve())


def _cylinder_intersects_box(
    x: float,
    y: float,
    z_lower: float,
    z_upper: float,
    radius: float,
    box: Sequence[float],
) -> bool:
    """Return whether one finite vertical cylinder intersects an AABB."""
    if z_upper < float(box[2]) or z_lower > float(box[5]):
        return False
    closest_x = min(max(x, float(box[0])), float(box[3]))
    closest_y = min(max(y, float(box[1])), float(box[4]))
    return (x - closest_x) ** 2 + (y - closest_y) ** 2 <= radius**2


def _sphere_intersects_box(
    center: tuple[float, float, float],
    radius: float,
    box: Sequence[float],
) -> bool:
    """Return whether one detector-head sphere intersects an AABB."""
    closest = np.minimum(
        np.maximum(np.asarray(center, dtype=float), np.asarray(box[:3], dtype=float)),
        np.asarray(box[3:], dtype=float),
    )
    return bool(np.sum((np.asarray(center, dtype=float) - closest) ** 2) <= radius**2)


def _pose_is_clear(
    pose: tuple[float, float, float],
    environment: Mapping[str, Any],
    grid: ObstacleGrid | None,
    motion: AdaptiveMotionConfig,
    reachable_cells: set[tuple[int, int]] | None,
) -> bool:
    """Validate room, traversability, base, mast, and detector-head clearance."""
    x, y, z = pose
    size_x = float(environment["size_x"])
    size_y = float(environment["size_y"])
    size_z = float(environment["size_z"])
    if not (
        motion.base_radius_m <= x <= size_x - motion.base_radius_m
        and motion.base_radius_m <= y <= size_y - motion.base_radius_m
        and motion.detector_height_min_m <= z <= motion.detector_height_max_m
        and motion.head_radius_m <= z <= size_z - motion.head_radius_m
    ):
        return False
    if grid is not None:
        cell = grid.cell_index(pose)
        if (
            cell is None
            or not grid.is_cell_free(cell)
            or (reachable_cells is not None and cell not in reachable_cells)
        ):
            return False
    boxes = () if grid is None else grid.collision_boxes_m
    base_upper = motion.base_height_m
    mast_upper = max(z, base_upper)
    for box in boxes:
        if _cylinder_intersects_box(
            x,
            y,
            0.0,
            base_upper,
            motion.base_radius_m,
            box,
        ):
            return False
        if _cylinder_intersects_box(
            x,
            y,
            base_upper,
            mast_upper,
            motion.mast_radius_m,
            box,
        ):
            return False
        if _sphere_intersects_box(pose, motion.head_radius_m, box):
            return False
    return True


def _sobol_points(
    lower: np.ndarray,
    upper: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    """Return the nested prefix of one scrambled three-dimensional Sobol design."""
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    exponent = int(math.ceil(math.log2(count)))
    sampler = qmc.Sobol(d=3, scramble=True, seed=int(seed))
    unit = sampler.random_base2(exponent)[:count]
    return np.asarray(qmc.scale(unit, lower, upper), dtype=np.float64)


def _unique_poses(
    poses: Sequence[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], ...]:
    """Deduplicate candidate poses without changing deterministic order."""
    seen: set[tuple[float, float, float]] = set()
    result: list[tuple[float, float, float]] = []
    for pose in poses:
        key = tuple(round(float(value), 12) for value in pose)
        if key in seen:
            continue
        seen.add(key)
        result.append(tuple(float(value) for value in pose))
    return tuple(result)


def _continuous_candidate_poses(
    environment: Mapping[str, Any],
    grid: ObstacleGrid | None,
    initial_pose: tuple[float, float, float],
    motion: AdaptiveMotionConfig,
) -> tuple[tuple[float, float, float], ...]:
    """Generate a collision-free nested Sobol design over detector XYZ."""
    size_x = _finite_number(
        environment.get("size_x", 0.0),
        field_name="environment.size_x",
        positive=True,
    )
    size_y = _finite_number(
        environment.get("size_y", 0.0),
        field_name="environment.size_y",
        positive=True,
    )
    lower = np.asarray(
        [
            motion.base_radius_m,
            motion.base_radius_m,
            motion.detector_height_min_m,
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            size_x - motion.base_radius_m,
            size_y - motion.base_radius_m,
            motion.detector_height_max_m,
        ],
        dtype=np.float64,
    )
    if np.any(upper < lower):
        raise ValueError("Detector assembly does not fit inside adaptive room bounds.")
    reachable: set[tuple[int, int]] | None = None
    if grid is not None:
        reachable = set(_grid_distances(grid, initial_pose))
    if not _pose_is_clear(initial_pose, environment, grid, motion, reachable):
        raise ValueError(
            "The initial detector pose is not assembly-clear and reachable."
        )
    anchors = [
        initial_pose,
        (initial_pose[0], initial_pose[1], motion.detector_height_min_m),
        (initial_pose[0], initial_pose[1], motion.detector_height_max_m),
        (
            initial_pose[0],
            initial_pose[1],
            0.5 * (motion.detector_height_min_m + motion.detector_height_max_m),
        ),
    ]
    sampled = _sobol_points(
        lower,
        upper,
        max(4 * motion.candidate_count, motion.candidate_count),
        motion.candidate_seed,
    )
    candidates = list(anchors)
    candidates.extend(tuple(float(value) for value in row) for row in sampled)
    clear = [
        pose
        for pose in _unique_poses(candidates)
        if _pose_is_clear(pose, environment, grid, motion, reachable)
    ]
    if len(clear) < 2:
        raise RuntimeError("Adaptive 3-D candidate generation found no usable motion.")
    return tuple(clear[: motion.candidate_count])


def _grid_distances(
    grid: ObstacleGrid,
    start_pose: tuple[float, float, float],
) -> dict[tuple[int, int], int]:
    """Return shortest four-connected free-cell distances from one pose."""
    start = grid.cell_index(start_pose)
    if start is None or not grid.is_cell_free(start):
        raise ValueError("Current detector pose is outside traversable free space.")
    distances = {start: 0}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        ix, iy = queue.popleft()
        for neighbor in ((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)):
            if neighbor in distances or not grid.is_cell_free(neighbor):
                continue
            distances[neighbor] = distances[(ix, iy)] + 1
            queue.append(neighbor)
    return distances


@dataclass(frozen=True, slots=True)
class AdaptiveCandidateSnapshot:
    """Expose reachable truth-free poses and runtime-owned motion costs."""

    candidate_poses_xyz: tuple[tuple[float, float, float], ...]
    travel_costs: tuple[float, ...]
    allowed_pair_ids: tuple[int, ...]
    current_pair_id: int

    def to_dict(self) -> dict[str, object]:
        """Return the estimator-visible candidate payload."""
        return {
            "candidate_poses_xyz": [list(pose) for pose in self.candidate_poses_xyz],
            "travel_costs": list(self.travel_costs),
            "allowed_pair_ids": list(self.allowed_pair_ids),
            "current_pair_id": int(self.current_pair_id),
        }


class AdaptiveCandidateProvider:
    """Generate reachable poses without exposing realized source truth."""

    def __init__(
        self,
        environment: Mapping[str, Any],
        obstacle_grid: ObstacleGrid | None,
        *,
        runtime_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the static collision-free continuous three-dimensional domain."""
        initial_pose = _initial_detector_pose(environment)
        motion = AdaptiveMotionConfig.from_inputs(environment, runtime_config)
        self.initial_pose = initial_pose
        self.environment = dict(environment)
        self.obstacle_grid = obstacle_grid
        self.motion = motion
        self.all_poses = _continuous_candidate_poses(
            environment,
            obstacle_grid,
            initial_pose,
            motion,
        )

    def _reachable_cells(
        self,
        current_pose: tuple[float, float, float],
    ) -> dict[tuple[int, int], int] | None:
        """Return current base-grid reachability, or none for an open room."""
        if self.obstacle_grid is None:
            return None
        return _grid_distances(self.obstacle_grid, current_pose)

    def _horizontal_distance_m(
        self,
        current_pose: tuple[float, float, float],
        target_pose: tuple[float, float, float],
        distances: Mapping[tuple[int, int], int] | None,
    ) -> float | None:
        """Return a reachable horizontal base-path length for one target."""
        if self.obstacle_grid is None:
            return float(
                np.linalg.norm(
                    np.asarray(target_pose[:2], dtype=float)
                    - np.asarray(current_pose[:2], dtype=float)
                )
            )
        assert distances is not None
        cell = self.obstacle_grid.cell_index(target_pose)
        if cell is None or cell not in distances:
            return None
        return float(distances[cell]) * float(self.obstacle_grid.cell_size)

    def motion_time_s(
        self,
        current_pose: tuple[float, float, float],
        target_pose: tuple[float, float, float],
        *,
        distances: Mapping[tuple[int, int], int] | None = None,
    ) -> float | None:
        """Return retract-translate-extend and settling time for one target."""
        horizontal = self._horizontal_distance_m(
            current_pose,
            target_pose,
            distances,
        )
        if horizontal is None:
            return None
        if horizontal <= 1.0e-12:
            vertical = abs(float(target_pose[2]) - float(current_pose[2]))
        else:
            transport = float(self.motion.transport_height_m)
            vertical = abs(float(current_pose[2]) - transport) + abs(
                float(target_pose[2]) - transport
            )
        changed = horizontal > 1.0e-12 or vertical > 1.0e-12
        return float(
            horizontal / float(self.motion.horizontal_speed_m_s)
            + vertical / float(self.motion.vertical_speed_m_s)
            + (float(self.motion.settling_time_s) if changed else 0.0)
        )

    def _shortest_cell_path(
        self,
        current_pose: tuple[float, float, float],
        target_pose: tuple[float, float, float],
    ) -> list[tuple[int, int]]:
        """Return one deterministic shortest free-cell path between poses."""
        if self.obstacle_grid is None:
            return []
        start = self.obstacle_grid.cell_index(current_pose)
        goal = self.obstacle_grid.cell_index(target_pose)
        if start is None or goal is None:
            raise ValueError("Travel route endpoints must lie inside the grid.")
        distances = self._reachable_cells(current_pose)
        if distances is None or goal not in distances:
            raise ValueError("Travel route target is not reachable from current pose.")
        if start == goal:
            return [start]
        path = [goal]
        cursor = goal
        while cursor != start:
            cursor_distance = int(distances[cursor])
            neighbors = sorted(
                (
                    (cursor[0] - 1, cursor[1]),
                    (cursor[0] + 1, cursor[1]),
                    (cursor[0], cursor[1] - 1),
                    (cursor[0], cursor[1] + 1),
                )
            )
            predecessor = next(
                (
                    neighbor
                    for neighbor in neighbors
                    if distances.get(neighbor) == cursor_distance - 1
                ),
                None,
            )
            if predecessor is None:
                raise RuntimeError("Reachability distances are not connected.")
            path.append(predecessor)
            cursor = predecessor
        path.reverse()
        return path

    def _cell_center_pose(
        self,
        cell: tuple[int, int],
        height_m: float,
    ) -> tuple[float, float, float]:
        """Return the detector waypoint at the center of one grid cell."""
        if self.obstacle_grid is None:
            raise RuntimeError("A cell-center pose requires an obstacle grid.")
        return (
            float(self.obstacle_grid.origin[0])
            + (float(cell[0]) + 0.5) * float(self.obstacle_grid.cell_size),
            float(self.obstacle_grid.origin[1])
            + (float(cell[1]) + 0.5) * float(self.obstacle_grid.cell_size),
            float(height_m),
        )

    def travel_waypoints_xyz(
        self,
        current_pose: tuple[float, float, float],
        target_pose: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], ...]:
        """Return the runtime-owned detector route for one selected action."""
        current = np.asarray(current_pose, dtype=float).reshape(3)
        target = np.asarray(target_pose, dtype=float).reshape(3)
        if np.any(~np.isfinite(current)) or np.any(~np.isfinite(target)):
            raise ValueError("Travel route poses must be finite.")
        if float(np.linalg.norm(target - current)) <= 1.0e-12:
            return ()
        horizontal = float(np.linalg.norm(target[:2] - current[:2]))
        transport = float(self.motion.transport_height_m)
        points: list[tuple[float, float, float]] = [
            tuple(float(value) for value in current)
        ]
        if horizontal > 1.0e-12:
            points.append((float(current[0]), float(current[1]), transport))
            if self.obstacle_grid is None:
                points.append((float(target[0]), float(target[1]), transport))
            else:
                points.extend(
                    self._cell_center_pose(cell, transport)
                    for cell in self._shortest_cell_path(
                        tuple(float(value) for value in current),
                        tuple(float(value) for value in target),
                    )
                )
                points.append((float(target[0]), float(target[1]), transport))
        points.append(tuple(float(value) for value in target))
        deduplicated: list[tuple[float, float, float]] = []
        for point in points:
            arr = np.asarray(point, dtype=float).reshape(3)
            if (
                deduplicated
                and float(
                    np.linalg.norm(
                        arr - np.asarray(deduplicated[-1], dtype=float)
                    )
                )
                <= 1.0e-9
            ):
                continue
            deduplicated.append(tuple(float(value) for value in arr))
        return tuple(deduplicated)

    def refine(
        self,
        current_pose: tuple[float, float, float],
        current_pair_id: int,
        seed_poses: Sequence[Sequence[float]],
    ) -> AdaptiveCandidateSnapshot:
        """Return runtime-validated local Sobol refinements around ranked seeds."""
        parsed: list[tuple[float, float, float]] = []
        for index, value in enumerate(seed_poses):
            array = np.asarray(value, dtype=float)
            if array.shape != (3,) or np.any(~np.isfinite(array)):
                raise ValueError(f"seed_poses[{index}] must be one finite XYZ pose.")
            parsed.append(tuple(float(item) for item in array))
        if not parsed or self.motion.local_refinement_count == 0:
            return self.snapshot(current_pose, current_pair_id)
        count_per_seed = max(
            1,
            int(math.ceil(self.motion.local_refinement_count / len(parsed))),
        )
        room_lower = np.asarray(
            [
                self.motion.base_radius_m,
                self.motion.base_radius_m,
                self.motion.detector_height_min_m,
            ],
            dtype=float,
        )
        room_upper = np.asarray(
            [
                float(self.environment["size_x"]) - self.motion.base_radius_m,
                float(self.environment["size_y"]) - self.motion.base_radius_m,
                self.motion.detector_height_max_m,
            ],
            dtype=float,
        )
        distances = self._reachable_cells(current_pose)
        reachable = None if distances is None else set(distances)
        candidates = list(self.all_poses)
        radius = float(self.motion.local_refinement_radius_m)
        for seed_index, seed in enumerate(parsed):
            center = np.asarray(seed, dtype=float)
            lower = np.maximum(room_lower, center - radius)
            upper = np.minimum(room_upper, center + radius)
            samples = _sobol_points(
                lower,
                upper,
                count_per_seed,
                self.motion.candidate_seed + 104729 * (seed_index + 1),
            )
            candidates.extend(
                tuple(float(item) for item in sample) for sample in samples
            )
        clear = [
            pose
            for pose in _unique_poses(candidates)
            if _pose_is_clear(
                pose,
                self.environment,
                self.obstacle_grid,
                self.motion,
                reachable,
            )
        ]
        return self._snapshot_from_poses(
            current_pose,
            current_pair_id,
            clear,
            distances=distances,
        )

    def _snapshot_from_poses(
        self,
        current_pose: tuple[float, float, float],
        current_pair_id: int,
        candidates: Sequence[tuple[float, float, float]],
        *,
        distances: Mapping[tuple[int, int], int] | None,
    ) -> AdaptiveCandidateSnapshot:
        """Build one reachable candidate snapshot with time-valued costs."""
        selected: list[tuple[float, float, float]] = []
        costs: list[float] = []
        for pose in candidates:
            cost = self.motion_time_s(
                current_pose,
                pose,
                distances=distances,
            )
            if cost is None:
                continue
            selected.append(pose)
            costs.append(cost)
        if not selected:
            raise RuntimeError("No reachable adaptive measurement pose remains.")
        return AdaptiveCandidateSnapshot(
            candidate_poses_xyz=tuple(selected),
            travel_costs=tuple(costs),
            allowed_pair_ids=tuple(range(64)),
            current_pair_id=int(current_pair_id),
        )

    def snapshot(
        self,
        current_pose: tuple[float, float, float],
        current_pair_id: int,
    ) -> AdaptiveCandidateSnapshot:
        """Return candidates while preserving the current validated pose."""
        return self._snapshot_from_poses(
            current_pose,
            current_pair_id,
            _unique_poses((current_pose, *self.all_poses)),
            distances=self._reachable_cells(current_pose),
        )


def _context_payload(context: RunContext) -> dict[str, object]:
    """Serialize the estimator-neutral live run context."""
    payload = {
        "repository_commit": context.repository_commit,
        "runtime_config": dict(context.runtime_config),
        "environment": dict(context.environment),
        "sim_backend": context.sim_backend,
        "spectrum_count_method": context.spectrum_count_method,
        "isotopes": list(context.isotopes),
        "obstacle_layout_path": context.obstacle_layout_path,
        "source_rate_model": context.source_rate_model,
        "metadata": dict(context.metadata),
        "run_id": context.run_id,
        "source_rate_semantics": dict(context.source_rate_semantics),
        "forward_model_manifest": dict(context.forward_model_manifest),
        "runtime_config_sha256": context.runtime_config_sha256,
        "schema_version": context.schema_version,
    }
    validate_truth_free_estimator_input(payload, path="adaptive.context")
    return payload


def _record_payload(record: MeasurementLogRecord) -> dict[str, object]:
    """Serialize one durably staged truth-free raw measurement record."""
    payload = {
        "step_id": record.step_id,
        "action_id": record.action_id,
        "station_id": record.station_id,
        "detector_pose_xyz": list(record.detector_pose_xyz),
        "detector_quat_wxyz": list(record.detector_quat_wxyz),
        "fe_orientation_index": record.fe_orientation_index,
        "pb_orientation_index": record.pb_orientation_index,
        "live_time_s": record.live_time_s,
        "travel_time_s": record.travel_time_s,
        "shield_actuation_time_s": record.shield_actuation_time_s,
        "energy_bin_edges_keV": record.energy_bin_edges_keV.tolist(),
        "spectrum_counts": record.spectrum_counts.tolist(),
        "metadata": dict(record.metadata),
    }
    validate_truth_free_estimator_input(payload, path="adaptive.record")
    return payload


class AdaptiveRuntimeSession:
    """Own one private scene while executing estimator-selected actions."""

    def __init__(
        self,
        observation_session: ObservationSession,
        context: RunContext,
        candidates: AdaptiveCandidateProvider,
        cui_truth_overlay: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize live action resolution at the environment start pose."""
        self.observation_session = observation_session
        self.context = context
        self.candidates = candidates
        self.cui_truth_overlay = (
            {}
            if cui_truth_overlay is None
            else json.loads(json.dumps(dict(cui_truth_overlay), allow_nan=False))
        )
        self.current_pose = candidates.initial_pose
        self.current_base_yaw_rad = 0.0
        self.current_pair_id = 0
        self._candidate_snapshot = candidates.snapshot(self.current_pose, 0)
        self._closed = False

    @classmethod
    def open(
        cls,
        scenario_path: str | Path,
        *,
        private_scene_profile: str | None = None,
    ) -> AdaptiveRuntimeSession:
        """Open a private scenario that contains no acquisition action list."""
        path = Path(scenario_path).expanduser().resolve()
        scenario = _load_json_object(path)
        if set(scenario) != _SCENARIO_FIELDS or scenario.get("schema_version") != 1:
            raise ValueError("Adaptive scenario must match schema version 1 exactly.")
        base = path.parent
        config_path = (base / str(scenario["runtime_config_path"])).resolve()
        output_dir = (base / str(scenario["output_dir"])).resolve()
        raw_config = load_runtime_config(config_path)
        isotopes = tuple(sorted(str(value) for value in scenario["isotopes"]))
        if not isotopes or len(set(isotopes)) != len(isotopes):
            raise ValueError("Scenario isotopes must be nonempty and unique.")
        backend = str(scenario["backend"])
        runtime_root = Path(__file__).resolve().parents[2]
        logged_config = estimator_neutral_runtime_config(
            raw_config,
            backend=backend,
            isotopes=isotopes,
            run_root=config_path.parents[2],
        )
        environment = scenario["environment"]
        scene = scenario["scene"]
        if not isinstance(environment, dict) or not isinstance(scene, dict):
            raise TypeError("Scenario environment and scene must be JSON objects.")
        commit = repository_commit(runtime_root)
        if len(commit) != 40:
            raise RuntimeError("Acquisition runtime must execute from a Git commit.")
        run_metadata = dict(scenario["metadata"])
        run_metadata["repository_source_snapshot_sha256"] = (
            repository_source_snapshot_sha256(runtime_root)
        )
        resolved_hash = sha256(canonical_json_bytes(logged_config)).hexdigest()
        forward = build_forward_model_manifest(
            runtime_config=logged_config,
            environment=environment,
            obstacle_layout_path=scenario["obstacle_layout_path"],
            isotopes=isotopes,
            repository_commit=commit,
            resolved_config_sha256=resolved_hash,
            repository_root=runtime_root,
        )
        scene_description = build_scene_description(scene)
        _validate_private_scene_profile(scene_description, private_scene_profile)
        writer = MeasurementLogStreamWriter(
            output_dir,
            run_id=str(scenario["run_id"]),
            repository_commit=commit,
            runtime_config=logged_config,
            environment=environment,
            forward_model_manifest=forward,
            isotopes=isotopes,
            metadata=run_metadata,
            obstacle_layout_path=scenario["obstacle_layout_path"],
            source_layout_path=None,
        )
        simulation_runtime = create_simulation_runtime(
            backend,
            sources=scene_description.to_point_sources(),
            mu_by_isotope={},
            shield_params=ShieldParams(),
            runtime_config=raw_config,
            runtime_config_path=config_path,
        )
        observation = ObservationSession(
            simulation_runtime=simulation_runtime,
            writer=writer,
            full_spectrum_contract_hash_sha256=str(
                logged_config["full_spectrum_contract_hash_sha256"]
            ),
        )
        try:
            observation.reset(scene)
            obstacle = _obstacle_grid(
                environment,
                scenario["obstacle_layout_path"],
                runtime_root=runtime_root,
            )
            provider = AdaptiveCandidateProvider(
                environment,
                obstacle,
                runtime_config=raw_config,
            )
            context = RunContext(
                repository_commit=commit,
                runtime_config=logged_config,
                environment=environment,
                sim_backend=backend,
                spectrum_count_method="joint_full_spectrum_generative",
                isotopes=isotopes,
                obstacle_layout_path=scenario["obstacle_layout_path"],
                source_layout_path=None,
                source_rate_model="detector_cps_1m",
                metadata=run_metadata,
                run_id=str(scenario["run_id"]),
                source_rate_semantics=SOURCE_RATE_SEMANTICS,
                forward_model_manifest=forward,
                runtime_config_sha256=resolved_hash,
                schema_version=MEASUREMENT_LOG_SCHEMA_VERSION,
            )
            return cls(
                observation,
                context,
                provider,
                cui_truth_overlay=_cui_truth_overlay(scene_description),
            )
        except BaseException:
            observation.close()
            raise

    def ready_payload(self) -> dict[str, object]:
        """Return the initial truth-free handshake."""
        return {
            "type": "ready",
            "schema_version": 1,
            "context": _context_payload(self.context),
            "candidates": self._candidate_snapshot.to_dict(),
            "bootstrap": {
                "candidate_index": int(
                    np.argmin(self._candidate_snapshot.travel_costs)
                ),
                "fe_orientation_index": 0,
                "pb_orientation_index": 0,
            },
        }

    def step(self, request: Mapping[str, Any]) -> dict[str, object]:
        """Execute one estimator selection and return its durable record."""
        if self._closed:
            raise RuntimeError("Adaptive runtime session is closed.")
        if set(request) != _STEP_FIELDS or request.get("type") != "step":
            raise ValueError("Adaptive step request fields disagree with schema 1.")
        candidate_index = request["candidate_index"]
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            raise TypeError("candidate_index must be an integer.")
        if not 0 <= candidate_index < len(self._candidate_snapshot.candidate_poses_xyz):
            raise ValueError("candidate_index is outside the current runtime snapshot.")
        target = self._candidate_snapshot.candidate_poses_xyz[candidate_index]
        travel_time = self._candidate_snapshot.travel_costs[candidate_index]
        travel_waypoints = self.candidates.travel_waypoints_xyz(
            self.current_pose,
            target,
        )
        requested_pair_id = int(request["fe_orientation_index"]) * 8 + int(
            request["pb_orientation_index"]
        )
        shield_actuation_time = self._shield_actuation_time_s(requested_pair_id)
        delta_x = target[0] - self.current_pose[0]
        delta_y = target[1] - self.current_pose[1]
        yaw = (
            self.current_base_yaw_rad
            if delta_x == 0.0 and delta_y == 0.0
            else math.atan2(delta_y, delta_x)
        )
        action = AcquisitionAction(
            station_id=request["station_id"],
            station_complete=request["station_complete"],
            command=SimulationCommand(
                step_id=len(self.observation_session.writer.records),
                target_pose_xyz=target,
                target_base_yaw_rad=yaw,
                fe_orientation_index=request["fe_orientation_index"],
                pb_orientation_index=request["pb_orientation_index"],
                dwell_time_s=request["dwell_time_s"],
                travel_time_s=travel_time,
                shield_actuation_time_s=shield_actuation_time,
                travel_waypoints_xyz=travel_waypoints or None,
            ),
        )
        self.observation_session.step(action)
        record = self.observation_session.writer.records[-1]
        self.current_pose = target
        self.current_base_yaw_rad = float(yaw)
        self.current_pair_id = int(record.fe_orientation_index) * 8 + int(
            record.pb_orientation_index
        )
        self._candidate_snapshot = self.candidates.snapshot(
            self.current_pose,
            self.current_pair_id,
        )
        return {
            "type": "record",
            "record": _record_payload(record),
            "candidates": self._candidate_snapshot.to_dict(),
        }

    def refine(self, request: Mapping[str, Any]) -> dict[str, object]:
        """Refine runtime-owned candidates around estimator-ranked seed indices."""
        if self._closed:
            raise RuntimeError("Adaptive runtime session is closed.")
        if set(request) != _REFINE_FIELDS or request.get("type") != "refine":
            raise ValueError("Adaptive refine request fields disagree with schema 1.")
        raw_indices = request["candidate_indices"]
        if not isinstance(raw_indices, list) or not raw_indices:
            raise TypeError("candidate_indices must be a nonempty JSON list.")
        indices: list[int] = []
        for value in raw_indices:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("candidate_indices must contain integers.")
            if not 0 <= value < len(self._candidate_snapshot.candidate_poses_xyz):
                raise ValueError("A candidate refinement index is out of range.")
            if value not in indices:
                indices.append(value)
        if len(indices) > 32:
            raise ValueError("At most 32 candidate seeds may be refined at once.")
        seeds = [
            self._candidate_snapshot.candidate_poses_xyz[index] for index in indices
        ]
        self._candidate_snapshot = self.candidates.refine(
            self.current_pose,
            self.current_pair_id,
            seeds,
        )
        return {
            "type": "candidates",
            "candidates": self._candidate_snapshot.to_dict(),
        }

    def cui_overlay(self, request: Mapping[str, Any]) -> dict[str, object]:
        """Return private CUI overlay data outside estimator-visible events."""
        if self._closed:
            raise RuntimeError("Adaptive runtime session is closed.")
        if set(request) != _CUI_OVERLAY_FIELDS or request.get("type") != "cui_overlay":
            raise ValueError("CUI overlay request fields disagree with schema 1.")
        include_truth = request["include_truth"]
        if not isinstance(include_truth, bool):
            raise TypeError("include_truth must be a boolean.")
        return {
            "type": "cui_overlay",
            "schema_version": 1,
            "truth": self.cui_truth_overlay if include_truth else None,
        }

    def _shield_actuation_time_s(self, target_pair_id: int) -> float:
        """Return parallel Fe/Pb actuator time from physical octant angles."""
        if not 0 <= int(target_pair_id) < 64:
            raise ValueError("target_pair_id must lie in [0, 63].")
        orientations = np.asarray(generate_octant_orientations(), dtype=float)
        current_fe, current_pb = divmod(int(self.current_pair_id), 8)
        target_fe, target_pb = divmod(int(target_pair_id), 8)

        def angle(first: int, second: int) -> float:
            """Return the shortest angular displacement between two normals."""
            cosine = float(
                np.clip(
                    np.dot(orientations[first], orientations[second]),
                    -1.0,
                    1.0,
                )
            )
            return float(math.acos(cosine))

        displacement = max(
            angle(current_fe, target_fe),
            angle(current_pb, target_pb),
        )
        return displacement / float(self.candidates.motion.shield_angular_speed_rad_s)

    def finalize(self) -> tuple[MeasurementLog, dict[str, object]]:
        """Publish the immutable log and close the live session."""
        if self._closed:
            raise RuntimeError("Adaptive runtime session is already closed.")
        log = self.observation_session.finalize()
        self._closed = True
        return log, {
            "type": "published",
            "path": log.path.resolve().as_posix(),
            "record_count": len(log.records),
        }

    def close(self) -> None:
        """Close without publishing when the controller aborts."""
        if not self._closed:
            self.observation_session.close()
            self._closed = True


def _write_event(stream: TextIO, payload: Mapping[str, object]) -> None:
    """Write one distinguishable, flushed JSON protocol event."""
    validate_truth_free_estimator_input(payload, path="adaptive.event")
    encoded = json.dumps(dict(payload), allow_nan=False, sort_keys=True)
    stream.write(f"{ADAPTIVE_EVENT_PREFIX}{encoded}\n")
    stream.flush()


def _write_cui_overlay_event(stream: TextIO, payload: Mapping[str, object]) -> None:
    """Write one private CUI overlay event without estimator validation."""
    encoded = json.dumps(dict(payload), allow_nan=False, sort_keys=True)
    stream.write(f"{ADAPTIVE_CUI_OVERLAY_PREFIX}{encoded}\n")
    stream.flush()


def serve_adaptive_session(
    scenario_path: str | Path,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    private_scene_profile: str | None = None,
) -> int:
    """Serve one private acquisition session over JSON lines."""
    session = AdaptiveRuntimeSession.open(
        scenario_path,
        private_scene_profile=private_scene_profile,
    )
    try:
        _write_event(output_stream, session.ready_payload())
        for line in input_stream:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise TypeError("Adaptive session request must be an object.")
            request_type = request.get("type")
            if request_type == "step":
                _write_event(output_stream, session.step(request))
            elif request_type == "refine":
                _write_event(output_stream, session.refine(request))
            elif request_type == "cui_overlay":
                _write_cui_overlay_event(
                    output_stream,
                    session.cui_overlay(request),
                )
            elif request_type == "finalize":
                if set(request) != {"type"}:
                    raise ValueError("finalize request has unknown fields.")
                _, payload = session.finalize()
                _write_event(output_stream, payload)
                return 0
            elif request_type == "abort":
                if set(request) != {"type"}:
                    raise ValueError("abort request has unknown fields.")
                session.close()
                _write_event(output_stream, {"type": "aborted"})
                return 0
            else:
                raise ValueError(f"Unknown adaptive request type: {request_type!r}.")
        raise EOFError("Adaptive controller disconnected before finalize or abort.")
    except BaseException:
        session.close()
        raise


__all__ = [
    "ADAPTIVE_CUI_OVERLAY_PREFIX",
    "ADAPTIVE_EVENT_PREFIX",
    "AdaptiveCandidateProvider",
    "AdaptiveCandidateSnapshot",
    "AdaptiveMotionConfig",
    "AdaptiveRuntimeSession",
    "cui_truth_overlay_from_scene",
    "serve_adaptive_session",
]
