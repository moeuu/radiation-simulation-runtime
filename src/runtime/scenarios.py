"""Author private physical scenarios for estimator-neutral acquisition."""

from __future__ import annotations

import json
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from measurement.model import EnvironmentConfig
from measurement.obstacle_assets import obstacle_instances_to_dicts
from measurement.obstacles import build_obstacle_grid
from measurement.source_surfaces import generate_surface_sources
from runtime.randomness import (
    named_random_generator,
    named_rng_provenance,
    named_stream_seed,
    normalize_random_seed,
)
from runtime_environment import attach_random_manchester_transport_geometry
from sim.runtime import load_runtime_config

RAL_MIX9_ISOTOPE_SEQUENCE = (
    "Cs-137",
    "Cs-137",
    "Cs-137",
    "Cs-137",
    "Co-60",
    "Co-60",
    "Co-60",
    "Eu-154",
    "Eu-154",
)
RAL_CS4_CO3_EU0_ISOTOPE_SEQUENCE = (
    "Cs-137",
    "Cs-137",
    "Cs-137",
    "Cs-137",
    "Co-60",
    "Co-60",
    "Co-60",
)
RAL_MIX9_ISOTOPES = tuple(sorted(set(RAL_MIX9_ISOTOPE_SEQUENCE)))
RAL_PRIVATE_SOURCE_PROFILES = {
    "ral-mix9": RAL_MIX9_ISOTOPE_SEQUENCE,
    "ral-cs4-co3-eu0": RAL_CS4_CO3_EU0_ISOTOPE_SEQUENCE,
}
RAL_PRIVATE_CANDIDATE_ISOTOPES = {
    profile: tuple(sorted(set(isotope_sequence)))
    for profile, isotope_sequence in RAL_PRIVATE_SOURCE_PROFILES.items()
}
_MAX_FRESH_SEED = (1 << 48) - 18


def generate_fresh_scene_seed() -> int:
    """Return a fresh JSON-safe seed for one independent physical scene."""
    return 1 + secrets.randbelow(_MAX_FRESH_SEED)


def _source_payload(source: object) -> dict[str, object]:
    """Serialize one generated continuous-surface source contract."""
    return {
        "isotope": str(source.isotope),
        "position": [float(value) for value in source.position],
        "transport_position": [
            float(value) for value in source.transport_position
        ],
        "intensity_cps_1m": float(source.intensity_cps_1m),
        "surface_chart_id": int(source.surface_chart_id),
        "surface_uv": [float(value) for value in source.surface_uv],
        "surface_normal": [float(value) for value in source.surface_normal],
        "surface_emission_policy_sha256": str(
            source.surface_emission_policy_sha256
        ),
    }


def build_random_ral_mix9_scenario(
    *,
    scene_seed: int,
    runtime_config_path: str | Path,
    measurement_log_output_dir: str | Path,
    run_id: str,
    intensity_cps_1m: float | Sequence[float] = (300_000.0, 2_000_000.0),
    candidate_count: int = 256,
    passage_width_m: float = 2.0,
    blocked_fraction: float = 0.4,
    same_isotope_min_distance_m: float = 3.0,
    source_profile: str = "ral-mix9",
    metadata: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build one action-free RA-L private scenario from runtime physics.

    The returned object contains realized physical truth and therefore belongs
    outside estimator repositories. It intentionally contains no station,
    route, view-count, shield-program, or stopping-policy field.
    """
    seed = normalize_random_seed(scene_seed)
    if source_profile not in RAL_PRIVATE_SOURCE_PROFILES:
        raise ValueError(
            f"Unknown RA-L source profile: {source_profile!r}."
        )
    isotope_sequence = RAL_PRIVATE_SOURCE_PROFILES[source_profile]
    candidate_isotopes = RAL_PRIVATE_CANDIDATE_ISOTOPES[source_profile]
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a nonempty string.")
    config_path = Path(runtime_config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Runtime configuration is missing: {config_path}")
    config = load_runtime_config(config_path)
    obstacle_height_m = float(config.get("obstacle_height_m", 2.0))
    chart_max_edge_m = float(
        config.get("structural_rj_surface_chart_max_edge_m", 1.0)
    )
    include_room_boundaries = bool(config.get("author_room_boundary_prims", False))
    room_boundary_thickness_m = float(
        config.get("room_boundary_thickness_m", 0.1)
    )
    environment_model_id = str(
        config.get(
            "environment_model_id",
            "random_manchester_component_union_v1",
        )
    )
    environment = EnvironmentConfig(
        size_x=10.0,
        size_y=20.0,
        size_z=10.0,
        detector_position=(1.0, 1.0, 0.5),
    )
    obstacle_seed = named_stream_seed(seed, "physical_obstacle_environment")
    candidate_seed = named_stream_seed(seed, "adaptive_candidate_workspace")
    grid = build_obstacle_grid(
        mode="random",
        path=None,
        size_x=environment.size_x,
        size_y=environment.size_y,
        cell_size=1.0,
        blocked_fraction=float(blocked_fraction),
        rng_seed=obstacle_seed,
        keep_free_points=[
            (environment.detector_position[0], environment.detector_position[1])
        ],
        passage_width_m=float(passage_width_m),
    )
    grid, obstacle_instances = attach_random_manchester_transport_geometry(
        grid,
        room_size_xyz=(
            environment.size_x,
            environment.size_y,
            environment.size_z,
        ),
        obstacle_height_m=obstacle_height_m,
        rng_seed=obstacle_seed,
        isotopes=candidate_isotopes,
        include_room_boundaries=include_room_boundaries,
        room_boundary_thickness_m=room_boundary_thickness_m,
    )
    sources = generate_surface_sources(
        env=environment,
        obstacle_grid=grid,
        isotopes=isotope_sequence,
        intensity_cps_1m=intensity_cps_1m,
        rng=named_random_generator(seed, "physical_surface_sources"),
        count=len(isotope_sequence),
        obstacle_height_m=obstacle_height_m,
        chart_max_edge_m=chart_max_edge_m,
        same_isotope_min_distance_m=float(same_isotope_min_distance_m),
    )
    obstacle_payload = obstacle_instances_to_dicts(obstacle_instances)
    environment_payload: dict[str, object] = {
        "environment_model_id": environment_model_id,
        "size_x": float(environment.size_x),
        "size_y": float(environment.size_y),
        "size_z": float(environment.size_z),
        "detector_position": [
            float(value) for value in environment.detector_position
        ],
        "obstacle_grid": grid.to_dict(),
        "obstacle_instances": obstacle_payload,
        "adaptive_measurement": {
            "candidate_count": int(candidate_count),
            "candidate_seed": int(candidate_seed),
        },
    }
    scene_payload: dict[str, object] = {
        "room_size_xyz": [
            float(environment.size_x),
            float(environment.size_y),
            float(environment.size_z),
        ],
        "sources": [_source_payload(source) for source in sources],
        "obstacle_origin_xy": [float(value) for value in grid.origin],
        "obstacle_cell_size_m": float(grid.cell_size),
        "obstacle_grid_shape": [int(value) for value in grid.grid_shape],
        "obstacle_cells": [list(cell) for cell in grid.blocked_cells],
        "collision_boxes_m": [list(box) for box in grid.collision_boxes_m],
        "transport_boxes_m": [list(box) for box in grid.transport_boxes_m],
        "transport_mu_by_isotope": {
            isotope: [float(value) for value in values]
            for isotope, values in grid.transport_mu_by_isotope.items()
        },
        "transport_line_mu_by_isotope": {
            isotope: [[float(value) for value in row] for row in rows]
            for isotope, rows in grid.transport_line_mu_by_isotope.items()
        },
        "transport_line_compton_mu_by_isotope": {
            isotope: [[float(value) for value in row] for row in rows]
            for isotope, rows in grid.transport_line_compton_mu_by_isotope.items()
        },
        "obstacle_instances": obstacle_payload,
        "author_obstacle_prims": True,
        "author_room_boundary_prims": include_room_boundaries,
        "use_config_usd_fallback": True,
    }
    run_metadata = dict(metadata or {})
    run_metadata.update(
        {
            "scenario_family": (
                "ral_random_physical_surface_v1:" + source_profile
            ),
            "private_source_profile": source_profile,
            "scene_seed": int(seed),
            "scene_rng_provenance": named_rng_provenance(
                seed,
                (
                    "physical_obstacle_environment",
                    "physical_surface_sources",
                    "adaptive_candidate_workspace",
                ),
            ),
            "same_isotope_min_distance_m": float(
                same_isotope_min_distance_m
            ),
            "measurement_actions_precomputed": False,
        }
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "backend": "geant4",
        "runtime_config_path": config_path.as_posix(),
        "output_dir": (
            Path(measurement_log_output_dir)
            .expanduser()
            .resolve()
            .as_posix()
        ),
        "environment": environment_payload,
        "scene": scene_payload,
        "isotopes": list(candidate_isotopes),
        "metadata": run_metadata,
        "obstacle_layout_path": None,
    }


def write_private_scenario(
    path: str | Path,
    scenario: Mapping[str, object],
) -> Path:
    """Write one private scenario without replacing an existing artifact."""
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to replace private scenario {target}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            dict(scenario),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


__all__ = [
    "RAL_CS4_CO3_EU0_ISOTOPE_SEQUENCE",
    "RAL_MIX9_ISOTOPES",
    "RAL_MIX9_ISOTOPE_SEQUENCE",
    "RAL_PRIVATE_CANDIDATE_ISOTOPES",
    "RAL_PRIVATE_SOURCE_PROFILES",
    "build_random_ral_mix9_scenario",
    "generate_fresh_scene_seed",
    "write_private_scenario",
]
