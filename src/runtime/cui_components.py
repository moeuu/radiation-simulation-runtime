"""Shared PF-reference scene, status, and five-panel CUI components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
import html
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from measurement.model import EnvironmentConfig
from measurement.obstacles import ObstacleGrid
from runtime.artifacts import atomic_write_json, atomic_write_text
from runtime.cui import CUIRoute


PF_REFERENCE_PANEL_ORDER = (
    "overview",
    "robot",
    "estimator",
    "estimator-labeled",
    "spectrum",
)


def _readonly_vector(value: object, *, name: str) -> NDArray[np.float64]:
    """Return one finite immutable float64 vector."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite vector.") from exc
    if array.ndim != 1 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector.")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


class CUITruthDisplayMode(StrEnum):
    """Declare whether realized truth may appear in evaluation-only output."""

    HIDDEN = "hidden"
    EVALUATION_LIVE = "evaluation_live"
    POST_RUN = "post_run"


@dataclass(frozen=True, slots=True)
class CUIPanelSpec:
    """Describe one image panel in the shared PF-reference dashboard shell."""

    panel_id: str
    title: str
    image_filename: str
    column_span: int = 1

    def __post_init__(self) -> None:
        """Validate browser-safe identity, text, filename, and grid span."""
        allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
        if (
            not isinstance(self.panel_id, str)
            or not self.panel_id
            or any(character not in allowed for character in self.panel_id)
        ):
            raise ValueError("CUI panel_id must be a lowercase browser identifier.")
        if not isinstance(self.title, str) or not self.title.strip():
            raise TypeError("CUI panel title must be a nonempty string.")
        if not isinstance(self.image_filename, str):
            raise TypeError("CUI panel image_filename must be a string.")
        path = Path(self.image_filename)
        if (
            path.is_absolute()
            or len(path.parts) != 1
            or path.name.startswith(".")
            or path.suffix.lower() != ".png"
        ):
            raise ValueError("CUI panel image_filename must be one visible PNG name.")
        if isinstance(self.column_span, bool) or self.column_span not in {1, 2}:
            raise ValueError("CUI panel column_span must be 1 or 2.")


def pf_reference_panel_specs(
    *,
    estimator_title: str = "Particle filter 3D",
    estimator_filename: str = "latest_pf_3d.png",
    labeled_estimator_title: str = "Particle filter 3D with source labels",
    labeled_estimator_filename: str = "latest_pf_3d_labeled.png",
) -> tuple[CUIPanelSpec, ...]:
    """Return the canonical five-panel order with estimator-specific 3-D slots."""
    return (
        CUIPanelSpec(
            "overview",
            "RA-L experiment overview",
            "latest_experiment_overview.png",
            2,
        ),
        CUIPanelSpec("robot", "Robot position 2D", "latest_robot_2d.png"),
        CUIPanelSpec("estimator", estimator_title, estimator_filename),
        CUIPanelSpec(
            "estimator-labeled",
            labeled_estimator_title,
            labeled_estimator_filename,
            2,
        ),
        CUIPanelSpec(
            "spectrum",
            "Raw native full spectrum",
            "latest_spectrum.png",
            2,
        ),
    )


@dataclass(frozen=True, slots=True)
class CUIStatus:
    """Store estimator-neutral dashboard progress without posterior content."""

    phase: str
    message: str
    step_id: int | None = None
    station_id: int | None = None

    def __post_init__(self) -> None:
        """Validate bounded status text and optional nonnegative identifiers."""
        if not isinstance(self.phase, str) or not self.phase or len(self.phase) > 64:
            raise ValueError("CUI status phase must be a bounded nonempty string.")
        if not isinstance(self.message, str) or len(self.message) > 512:
            raise ValueError("CUI status message must be a bounded string.")
        for name, value in (
            ("step_id", self.step_id),
            ("station_id", self.station_id),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"CUI status {name} must be nonnegative or null.")

    def to_payload(self) -> dict[str, object]:
        """Return strict JSON data for browser-side status display."""
        return {
            "schema_version": 1,
            "phase": self.phase,
            "message": self.message,
            "step_id": self.step_id,
            "station_id": self.station_id,
        }


@dataclass(frozen=True, slots=True)
class CUIScene:
    """Store immutable room bounds and correctly ordered obstacle geometry."""

    bounds_min_xyz: NDArray[np.float64]
    bounds_max_xyz: NDArray[np.float64]
    obstacle_boxes_xyz: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate finite increasing bounds and canonical XYZXYZ boxes."""
        lower = _readonly_vector(self.bounds_min_xyz, name="bounds_min_xyz")
        upper = _readonly_vector(self.bounds_max_xyz, name="bounds_max_xyz")
        if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
            raise ValueError("CUI scene bounds must be increasing XYZ vectors.")
        try:
            raw_boxes = np.asarray(self.obstacle_boxes_xyz, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("CUI obstacle boxes must have shape (N, 6).") from exc
        if raw_boxes.size == 0:
            raw_boxes = np.zeros((0, 6), dtype=np.float64)
        if (
            raw_boxes.ndim != 2
            or raw_boxes.shape[1:] != (6,)
            or np.any(~np.isfinite(raw_boxes))
            or np.any(raw_boxes[:, 3:] <= raw_boxes[:, :3])
        ):
            raise ValueError("CUI obstacle boxes must be increasing XYZXYZ rows.")
        boxes = np.array(raw_boxes, dtype=np.float64, copy=True)
        boxes.setflags(write=False)
        object.__setattr__(self, "bounds_min_xyz", lower)
        object.__setattr__(self, "bounds_max_xyz", upper)
        object.__setattr__(self, "obstacle_boxes_xyz", boxes)

    @classmethod
    def from_environment(
        cls,
        environment: EnvironmentConfig,
        obstacle_grid: ObstacleGrid | None,
        *,
        obstacle_height_m: float = 2.0,
    ) -> "CUIScene":
        """Build one scene while preserving runtime obstacle (x, y) ordering."""
        if not isinstance(environment, EnvironmentConfig):
            raise TypeError("environment must be an EnvironmentConfig.")
        if isinstance(obstacle_height_m, bool) or not isinstance(
            obstacle_height_m,
            (int, float),
        ):
            raise TypeError("obstacle_height_m must be numeric.")
        height = float(obstacle_height_m)
        if not np.isfinite(height) or height <= 0.0:
            raise ValueError("obstacle_height_m must be finite and positive.")
        boxes: tuple[tuple[float, ...], ...] = ()
        if obstacle_grid is not None:
            if not isinstance(obstacle_grid, ObstacleGrid):
                raise TypeError("obstacle_grid must be an ObstacleGrid or null.")
            boxes = tuple(
                tuple(float(value) for value in box)
                for box in obstacle_grid.attenuation_boxes(
                    z_min=0.0,
                    z_max=height,
                )
            )
        return cls(
            bounds_min_xyz=np.zeros(3, dtype=np.float64),
            bounds_max_xyz=np.asarray(
                [environment.size_x, environment.size_y, environment.size_z],
                dtype=np.float64,
            ),
            obstacle_boxes_xyz=np.asarray(boxes, dtype=np.float64).reshape(-1, 6),
        )

    @property
    def obstacle_footprints_xy(self) -> tuple[NDArray[np.float64], ...]:
        """Return four-corner XY polygons without swapping grid axes."""
        polygons: list[NDArray[np.float64]] = []
        for x0, y0, _z0, x1, y1, _z1 in self.obstacle_boxes_xyz:
            polygon = np.asarray(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                dtype=np.float64,
            )
            polygon.setflags(write=False)
            polygons.append(polygon)
        return tuple(polygons)

    def to_payload(self) -> dict[str, object]:
        """Return strict JSON geometry shared by estimator renderers."""
        return {
            "schema_version": 1,
            "bounds_min_xyz": self.bounds_min_xyz.tolist(),
            "bounds_max_xyz": self.bounds_max_xyz.tolist(),
            "obstacle_boxes_xyz": self.obstacle_boxes_xyz.tolist(),
        }


@dataclass(frozen=True, slots=True)
class CUIAcquisitionFrame:
    """Bind one cumulative truth-free route to shared dashboard status."""

    route: CUIRoute
    status: CUIStatus
    truth_display_mode: CUITruthDisplayMode = CUITruthDisplayMode.HIDDEN

    def __post_init__(self) -> None:
        """Require exact shared DTOs and an explicit truth-display mode."""
        if not isinstance(self.route, CUIRoute):
            raise TypeError("CUI acquisition frame route must be CUIRoute.")
        if not isinstance(self.status, CUIStatus):
            raise TypeError("CUI acquisition frame status must be CUIStatus.")
        if not isinstance(self.truth_display_mode, CUITruthDisplayMode):
            raise TypeError("truth_display_mode must be CUITruthDisplayMode.")

    def to_payload(self) -> dict[str, object]:
        """Return browser-safe frame data without any realized truth values."""
        return {
            "schema_version": 1,
            "route": self.route.to_payload(),
            "status": self.status.to_payload(),
            "truth_display_mode": self.truth_display_mode.value,
        }


def write_cui_index(
    root: str | Path,
    panels: Sequence[CUIPanelSpec],
    *,
    title: str = "Rotating-shield radiation estimation",
    refresh_interval_ms: int = 1200,
    index_filename: str = "index.html",
) -> Path:
    """Publish the common PF-reference responsive five-panel HTML shell."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    panel_values = tuple(panels)
    if len(panel_values) != 5 or any(
        not isinstance(panel, CUIPanelSpec) for panel in panel_values
    ):
        raise ValueError("The shared CUI shell requires exactly five panel specs.")
    if tuple(panel.panel_id for panel in panel_values) != PF_REFERENCE_PANEL_ORDER:
        raise ValueError("CUI panels must follow the PF reference panel order.")
    if isinstance(refresh_interval_ms, bool) or not isinstance(
        refresh_interval_ms,
        int,
    ):
        raise TypeError("refresh_interval_ms must be an integer.")
    if refresh_interval_ms < 250:
        raise ValueError("refresh_interval_ms must be at least 250.")
    index_path = Path(index_filename)
    if (
        index_path.is_absolute()
        or len(index_path.parts) != 1
        or index_path.name.startswith(".")
        or index_path.suffix != ".html"
    ):
        raise ValueError("index_filename must be one visible HTML filename.")
    sections = "\n".join(
        (
            f'<section class="panel span-{panel.column_span}">'
            f"<h2>{html.escape(panel.title)}</h2>"
            f'<img id="{html.escape(panel.panel_id)}" '
            f'src="{html.escape(panel.image_filename)}" '
            f'alt="{html.escape(panel.title)}">'
            "</section>"
        )
        for panel in panel_values
    )
    refresh_lines = "\n".join(
        (
            f'document.getElementById({json.dumps(panel.panel_id)}).src = '
            f'{json.dumps(panel.image_filename)} + "?t=" + token;'
        )
        for panel in panel_values
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin: 0; background: #0d1117; color: #e6edf3; }}
header {{ padding: 12px 16px 4px; }}
h1 {{ margin: 0; font-size: 18px; }}
main {{ display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 10px; padding: 10px; }}
.panel {{ min-width: 0; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 8px; }}
.span-2 {{ grid-column: 1 / -1; }}
h2 {{ margin: 0 0 6px; font-size: 14px; font-weight: 600; }}
img {{ display: block; width: 100%; height: auto; background: #fff; }}
@media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} .span-2 {{ grid-column: auto; }} }}
</style></head><body><header><h1>{html.escape(title)}</h1></header><main>
{sections}
</main><script>
function refreshPanels() {{ const token = Date.now(); {refresh_lines} }}
setInterval(refreshPanels, {refresh_interval_ms});
</script></body></html>
"""
    return atomic_write_text(root_path / index_path, document)


def write_cui_status(path: str | Path, status: CUIStatus) -> Path:
    """Atomically publish one estimator-neutral dashboard status document."""
    if not isinstance(status, CUIStatus):
        raise TypeError("status must be CUIStatus.")
    return atomic_write_json(path, status.to_payload())


__all__ = [
    "PF_REFERENCE_PANEL_ORDER",
    "CUIAcquisitionFrame",
    "CUIPanelSpec",
    "CUIScene",
    "CUIStatus",
    "CUITruthDisplayMode",
    "pf_reference_panel_specs",
    "write_cui_index",
    "write_cui_status",
]
