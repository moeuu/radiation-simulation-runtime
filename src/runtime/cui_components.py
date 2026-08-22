"""Estimator-neutral scene, status, panel, and CUI shell components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from runtime.forward_model_manifest import resolve_file_backed_model_asset
from runtime.provenance import load_strict_json
from runtime.records import RunContext, validate_truth_free_estimator_input


_SHARED_CONTEXT_PANEL_IDS = frozenset({"overview", "robot", "spectrum"})


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
    """Describe one owner-rendered image in the shared dashboard shell."""

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


def shared_cui_panel_specs(
    result_panels: Sequence[CUIPanelSpec],
) -> tuple[CUIPanelSpec, ...]:
    """Place owner-defined result panels in the shared dashboard structure."""
    result_values = tuple(result_panels)
    if not result_values:
        raise ValueError("The shared CUI structure requires a result panel.")
    if any(not isinstance(panel, CUIPanelSpec) for panel in result_values):
        raise TypeError("result_panels must contain CUIPanelSpec values.")
    result_ids = tuple(panel.panel_id for panel in result_values)
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("CUI result panel identifiers must be unique.")
    if _SHARED_CONTEXT_PANEL_IDS.intersection(result_ids):
        raise ValueError(
            "CUI result panel identifiers must not replace shared context panels."
        )
    return (
        CUIPanelSpec(
            "overview",
            "RA-L experiment overview",
            "latest_experiment_overview.png",
            2,
        ),
        CUIPanelSpec("robot", "Robot position 2D", "latest_robot_2d.png"),
        *result_values,
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

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CUIScene":
        """Parse one exact schema-v1 scene saved in a live manifest."""
        if not isinstance(payload, Mapping):
            raise TypeError("CUI scene payload must be an object.")
        expected = {
            "schema_version",
            "bounds_min_xyz",
            "bounds_max_xyz",
            "obstacle_boxes_xyz",
        }
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                "CUI scene payload fields disagree with schema 1: "
                f"missing={sorted(expected - actual)}, "
                f"unknown={sorted(actual - expected)}."
            )
        schema_version = payload["schema_version"]
        if (
            isinstance(schema_version, (bool, np.bool_))
            or not isinstance(schema_version, (int, np.integer))
            or int(schema_version) != 1
        ):
            raise ValueError("CUI scene schema_version must be exactly 1.")
        return cls(
            bounds_min_xyz=payload["bounds_min_xyz"],
            bounds_max_xyz=payload["bounds_max_xyz"],
            obstacle_boxes_xyz=payload["obstacle_boxes_xyz"],
        )


def _cui_runtime_asset_root(value: str | Path) -> Path:
    """Return one explicit absolute directory for CUI asset resolution."""
    supplied = Path(value)
    if not supplied.is_absolute():
        raise ValueError("runtime_asset_root must be an absolute path.")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ValueError("runtime_asset_root must exist.") from exc
    if not resolved.is_dir():
        raise ValueError("runtime_asset_root must be an existing directory.")
    return resolved


def _cui_obstacle_grid(
    context: RunContext,
    *,
    runtime_asset_root: Path,
) -> ObstacleGrid | None:
    """Resolve embedded or root-confined file-backed obstacle geometry."""
    raw_embedded = context.environment.get("obstacle_grid")
    if raw_embedded is not None:
        if not isinstance(raw_embedded, Mapping):
            raise ValueError("environment.obstacle_grid must be an object or null.")
        return ObstacleGrid.from_dict(dict(raw_embedded))
    if context.obstacle_layout_path is None:
        return None
    resolved = resolve_file_backed_model_asset(
        context.obstacle_layout_path,
        field_name="obstacle_layout_path",
        repository_root=runtime_asset_root,
    )
    payload = load_strict_json(resolved)
    if not isinstance(payload, dict):
        raise ValueError("File-backed obstacle layout must be a JSON object.")
    return ObstacleGrid.from_dict(payload)


def cui_scene_from_run_context(
    context: RunContext,
    *,
    runtime_asset_root: str | Path,
) -> CUIScene:
    """Build a truth-free CUI scene without constructing spectral models."""
    if not isinstance(context, RunContext):
        raise TypeError("context must be a RunContext.")
    validate_truth_free_estimator_input(
        context.to_payload(),
        path="cui.run_context",
    )
    asset_root = _cui_runtime_asset_root(runtime_asset_root)
    environment_payload = context.environment
    required = {"size_x", "size_y", "size_z", "detector_position"}
    missing = sorted(required - set(environment_payload))
    if missing:
        raise ValueError(
            "CUI environment is missing required fields: " + ", ".join(missing)
        )
    environment = EnvironmentConfig(
        size_x=environment_payload["size_x"],
        size_y=environment_payload["size_y"],
        size_z=environment_payload["size_z"],
        detector_position=environment_payload["detector_position"],
    )
    obstacle_grid = _cui_obstacle_grid(
        context,
        runtime_asset_root=asset_root,
    )
    return CUIScene.from_environment(
        environment,
        obstacle_grid,
        obstacle_height_m=environment.size_z,
    )


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
    asset_base_href: str | None = None,
) -> Path:
    """Publish a responsive shell for owner-defined image panels."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    panel_values = tuple(panels)
    if not panel_values:
        raise ValueError("The shared CUI shell requires at least one panel spec.")
    if any(not isinstance(panel, CUIPanelSpec) for panel in panel_values):
        raise TypeError("panels must contain CUIPanelSpec values.")
    panel_ids = tuple(panel.panel_id for panel in panel_values)
    if len(set(panel_ids)) != len(panel_ids):
        raise ValueError("CUI panel identifiers must be unique.")
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
    base_element = ""
    if asset_base_href is not None:
        if (
            not isinstance(asset_base_href, str)
            or not asset_base_href
            or not asset_base_href.endswith("/")
            or asset_base_href.startswith("/")
            or "\\" in asset_base_href
            or "\x00" in asset_base_href
            or "?" in asset_base_href
            or "#" in asset_base_href
            or ":" in asset_base_href
        ):
            raise ValueError(
                "asset_base_href must be a safe relative directory reference."
            )
        components = asset_base_href[:-1].split("/")
        if any(
            not component
            or component == "."
            or (
                component != ".."
                and any(
                    character
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
                    for character in component
                )
            )
            for component in components
        ):
            raise ValueError(
                "asset_base_href must be a safe relative directory reference."
            )
        base_element = f'<base href="{html.escape(asset_base_href)}">\n'
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
{base_element}<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="icon" href="data:,">
<style>
:root {{ color-scheme: dark; font-family: sans-serif; }}
body {{ margin: 0; background: #111; color: #eee; }}
header {{ padding: 10px 16px; background: #1d1d1d; border-bottom: 1px solid #333; }}
h1 {{ margin: 0; font-size: 16px; font-weight: 600; }}
main {{ display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 10px; padding: 10px; }}
.panel {{ min-width: 0; background: #181818; border: 1px solid #333; padding: 8px; }}
.span-2 {{ grid-column: 1 / -1; }}
h2 {{ margin: 0 0 8px; font-size: 16px; font-weight: 600; }}
img {{ display: block; width: 100%; height: calc(50vh - 70px); object-fit: contain; background: #fff; }}
@media (max-width: 900px) {{
  main {{ grid-template-columns: 1fr; }}
  .span-2 {{ grid-column: auto; }}
  img {{ height: auto; min-height: 240px; }}
}}
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
    "CUIAcquisitionFrame",
    "CUIPanelSpec",
    "CUIScene",
    "CUIStatus",
    "CUITruthDisplayMode",
    "cui_scene_from_run_context",
    "shared_cui_panel_specs",
    "write_cui_index",
    "write_cui_status",
]
