"""Author estimator particle and estimate markers into an Isaac Sim stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sim.isaacsim_app.stage_backend import StageBackend


ISOTOPE_COLORS: dict[str, tuple[float, float, float]] = {
    "Cs-137": (1.0, 0.05, 0.05),
    "Co-60": (0.05, 0.45, 1.0),
    "Eu-154": (0.0, 0.75, 0.25),
    "Eu-155": (0.0, 0.75, 0.25),
}
_ESTIMATOR_FRAME_FIELDS = frozenset(
    {
        "sample_positions",
        "sample_weights",
        "estimated_sources",
        "estimated_strengths",
    }
)


@dataclass(frozen=True)
class EstimatorSceneVisualizationConfig:
    """Collect visual-only estimator marker settings for Isaac Sim."""

    enabled: bool = True
    max_particles_per_isotope: int = 800
    particle_radius_m: float = 0.025
    estimate_radius_m: float = 0.13
    estimate_cross_size_m: float = 0.35
    estimate_cross_width_m: float = 0.035
    min_weight_fraction: float = 0.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "EstimatorSceneVisualizationConfig":
        """Build a config from an application config mapping."""
        payload = {} if data is None else dict(data)
        return cls(
            enabled=bool(payload.get("estimator_visualization_enabled", True)),
            max_particles_per_isotope=max(
                0,
                int(payload.get("estimator_visual_max_samples_per_isotope", 800)),
            ),
            particle_radius_m=max(
                1.0e-4,
                float(payload.get("estimator_visual_sample_radius_m", 0.025)),
            ),
            estimate_radius_m=max(
                1.0e-4,
                float(payload.get("estimator_visual_estimate_radius_m", 0.13)),
            ),
            estimate_cross_size_m=max(
                1.0e-4,
                float(payload.get("estimator_visual_estimate_cross_size_m", 0.35)),
            ),
            estimate_cross_width_m=max(
                1.0e-4,
                float(payload.get("estimator_visual_estimate_cross_width_m", 0.035)),
            ),
            min_weight_fraction=max(
                0.0,
                float(payload.get("estimator_visual_min_weight_fraction", 0.0)),
            ),
        )


class EstimatorSceneVisualizer:
    """Render estimator particles and estimates as visual-only Isaac Sim prims."""

    def __init__(
        self,
        stage_backend: StageBackend,
        *,
        config: EstimatorSceneVisualizationConfig | None = None,
        root_path: str = "/World/SimBridge/EstimatorVisualization",
    ) -> None:
        """Store the backend and marker roots."""
        self.stage_backend = stage_backend
        self.config = config or EstimatorSceneVisualizationConfig()
        self.root_path = str(root_path)
        self.particles_root = f"{self.root_path}/Particles"
        self.estimates_root = f"{self.root_path}/Estimates"

    def update_from_payload(self, payload: dict[str, Any]) -> None:
        """Replace estimator markers from a serialized estimator frame payload."""
        if not self.config.enabled:
            return
        if not isinstance(payload, dict) or set(payload) != _ESTIMATOR_FRAME_FIELDS:
            raise ValueError(
                "Estimator visualization frame must contain exactly "
                f"{sorted(_ESTIMATOR_FRAME_FIELDS)}."
            )
        particles = _require_point_mapping(
            payload["sample_positions"],
            location="sample_positions",
        )
        weights = _require_raw_mapping(
            payload["sample_weights"],
            location="sample_weights",
        )
        estimates = _require_point_mapping(
            payload["estimated_sources"],
            location="estimated_sources",
        )
        strengths = _require_raw_mapping(
            payload["estimated_strengths"],
            location="estimated_strengths",
        )
        if set(weights) != set(particles):
            raise ValueError("sample_weights isotope keys must match sample_positions.")
        if set(strengths) != set(estimates):
            raise ValueError(
                "estimated_strengths isotope keys must match estimated_sources."
            )
        self.stage_backend.remove_prim(self.root_path)
        self.stage_backend.ensure_xform(self.root_path)
        self.stage_backend.ensure_xform(self.particles_root)
        self.stage_backend.ensure_xform(self.estimates_root)
        for isotope, positions in sorted(particles.items()):
            weight_arr = _require_vector(
                weights[isotope],
                positions.shape[0],
                location=f"sample_weights.{isotope}",
            )
            selected_positions, selected_weights = self._select_particles(
                positions,
                weight_arr,
            )
            self._author_particles(isotope, selected_positions, selected_weights)
        for isotope, positions in sorted(estimates.items()):
            strength_arr = _require_vector(
                strengths[isotope],
                positions.shape[0],
                location=f"estimated_strengths.{isotope}",
            )
            self._author_estimates(isotope, positions, strength_arr)
        self.stage_backend.step()

    def _select_particles(
        self,
        positions: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return a top-weight bounded particle subset for GUI rendering."""
        if positions.size == 0 or weights.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        valid = np.isfinite(positions).all(axis=1) & np.isfinite(weights)
        if self.config.min_weight_fraction > 0.0 and np.any(valid):
            max_weight = float(np.max(np.abs(weights[valid])))
            valid &= weights >= max_weight * self.config.min_weight_fraction
        positions = positions[valid]
        weights = weights[valid]
        if positions.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        max_particles = int(self.config.max_particles_per_isotope)
        if max_particles > 0 and positions.shape[0] > max_particles:
            order = np.argsort(weights)[::-1][:max_particles]
            positions = positions[order]
            weights = weights[order]
        return positions, weights

    def _author_particles(
        self,
        isotope: str,
        positions: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> None:
        """Author small estimator particle spheres for one isotope."""
        isotope_token = _sanitize_token(isotope)
        root = f"{self.particles_root}/{isotope_token}"
        self.stage_backend.ensure_xform(root)
        color = _isotope_color(isotope)
        radii = _particle_radii(
            weights,
            base_radius=float(self.config.particle_radius_m),
        )
        for index, (position, radius_m) in enumerate(zip(positions, radii)):
            self.stage_backend.ensure_sphere(
                f"{root}/Particle_{index:04d}",
                radius_m=float(radius_m),
                translation_xyz=_tuple3(position),
                color_rgb=color,
                material="air",
            )

    def _author_estimates(
        self,
        isotope: str,
        positions: NDArray[np.float64],
        strengths: NDArray[np.float64],
    ) -> None:
        """Author estimate markers and cross-hairs for one isotope."""
        isotope_token = _sanitize_token(isotope)
        root = f"{self.estimates_root}/{isotope_token}"
        self.stage_backend.ensure_xform(root)
        color = _isotope_color(isotope)
        for index, position in enumerate(positions):
            if not np.isfinite(position).all():
                continue
            marker_root = f"{root}/Estimate_{index:02d}"
            strength_scale = _estimate_strength_scale(strengths, index)
            radius = float(self.config.estimate_radius_m) * strength_scale
            self.stage_backend.ensure_sphere(
                f"{marker_root}/Center",
                radius_m=radius,
                translation_xyz=_tuple3(position),
                color_rgb=color,
                material="air",
            )
            self._author_cross(marker_root, position, color)

    def _author_cross(
        self,
        root: str,
        center: NDArray[np.float64],
        color: tuple[float, float, float],
    ) -> None:
        """Author three short cross-hair curves centered on an estimate."""
        half = 0.5 * float(self.config.estimate_cross_size_m)
        width = float(self.config.estimate_cross_width_m)
        axes = (
            ((-half, 0.0, 0.0), (half, 0.0, 0.0), "X"),
            ((0.0, -half, 0.0), (0.0, half, 0.0), "Y"),
            ((0.0, 0.0, -half), (0.0, 0.0, half), "Z"),
        )
        center_arr = np.asarray(center, dtype=float)
        for start_offset, end_offset, axis_name in axes:
            start = center_arr + np.asarray(start_offset, dtype=float)
            end = center_arr + np.asarray(end_offset, dtype=float)
            self.stage_backend.ensure_polyline(
                f"{root}/Cross_{axis_name}",
                points_xyz=(_tuple3(start), _tuple3(end)),
                color_rgb=color,
                width_m=width,
            )


def _require_point_mapping(
    value: Any,
    *,
    location: str,
) -> dict[str, NDArray[np.float64]]:
    """Return exact finite isotope-keyed XYZ arrays."""
    if not isinstance(value, dict):
        raise TypeError(f"{location} must be an object.")
    output: dict[str, NDArray[np.float64]] = {}
    for isotope, raw in value.items():
        if not isinstance(isotope, str) or not isotope:
            raise TypeError(f"{location} keys must be nonempty strings.")
        arr = np.asarray(raw)
        if arr.size == 0:
            if type(raw) is not list or raw:
                raise ValueError(f"{location}.{isotope} must have shape (N, 3).")
            output[isotope] = np.zeros((0, 3), dtype=float)
            continue
        if (
            arr.ndim != 2
            or arr.shape[1] != 3
            or not np.issubdtype(arr.dtype, np.number)
            or np.issubdtype(arr.dtype, np.bool_)
        ):
            raise ValueError(f"{location}.{isotope} must be a numeric (N, 3) array.")
        parsed = np.asarray(arr, dtype=float)
        if np.any(~np.isfinite(parsed)):
            raise ValueError(f"{location}.{isotope} must contain finite values.")
        output[isotope] = parsed
    return output


def _require_raw_mapping(value: Any, *, location: str) -> dict[str, Any]:
    """Return one exact nonempty-string-keyed vector mapping."""
    if not isinstance(value, dict):
        raise TypeError(f"{location} must be an object.")
    if any(not isinstance(key, str) or not key for key in value):
        raise TypeError(f"{location} keys must be nonempty strings.")
    return dict(value)


def _require_vector(value: Any, size: int, *, location: str) -> NDArray[np.float64]:
    """Return one exact finite nonnegative numeric vector of the required size."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("Visualization vector size must be a nonnegative integer.")
    arr = np.asarray(value)
    if (
        arr.ndim != 1
        or arr.size != size
        or not np.issubdtype(arr.dtype, np.number)
        or np.issubdtype(arr.dtype, np.bool_)
    ):
        raise ValueError(f"{location} must be a numeric vector of length {size}.")
    parsed = np.asarray(arr, dtype=float)
    if np.any(~np.isfinite(parsed)) or np.any(parsed < 0.0):
        raise ValueError(f"{location} must contain finite nonnegative values.")
    return parsed


def _particle_radii(
    weights: NDArray[np.float64],
    *,
    base_radius: float,
) -> NDArray[np.float64]:
    """Scale particle marker radii by relative posterior weight."""
    if weights.size == 0:
        return np.zeros(0, dtype=float)
    finite = np.asarray(weights, dtype=float)
    finite = np.where(np.isfinite(finite), np.maximum(finite, 0.0), 0.0)
    max_weight = float(np.max(finite)) if finite.size else 0.0
    if max_weight <= 0.0:
        return np.full(finite.shape, base_radius, dtype=float)
    relative = np.sqrt(np.clip(finite / max_weight, 0.0, 1.0))
    return base_radius * (0.7 + 1.2 * relative)


def _estimate_strength_scale(strengths: NDArray[np.float64], index: int) -> float:
    """Return a mild display scale based on relative estimated strength."""
    if strengths.size == 0 or index >= strengths.size:
        return 1.0
    finite = np.where(np.isfinite(strengths), np.maximum(strengths, 0.0), 0.0)
    max_strength = float(np.max(finite)) if finite.size else 0.0
    if max_strength <= 0.0:
        return 1.0
    relative = float(np.sqrt(np.clip(finite[index] / max_strength, 0.0, 1.0)))
    return float(0.85 + 0.55 * relative)


def _isotope_color(isotope: str) -> tuple[float, float, float]:
    """Return the configured visualization color for an isotope."""
    return ISOTOPE_COLORS.get(str(isotope), (1.0, 0.8, 0.05))


def _sanitize_token(value: str) -> str:
    """Return a USD path token safe enough for generated marker names."""
    chars = [char if char.isalnum() else "_" for char in str(value)]
    token = "".join(chars).strip("_")
    if not token:
        return "Isotope"
    if token[0].isdigit():
        return f"Isotope_{token}"
    return token


def _tuple3(value: NDArray[np.float64]) -> tuple[float, float, float]:
    """Convert a length-three vector to a float tuple."""
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 3:
        raise ValueError("Expected a three-dimensional position.")
    return (float(arr[0]), float(arr[1]), float(arr[2]))
