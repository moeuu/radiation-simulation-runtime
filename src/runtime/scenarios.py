"""Author private physical scenarios for estimator-neutral acquisition."""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from measurement.obstacle_assets import obstacle_instances_to_dicts
from measurement.obstacles import build_obstacle_grid
from measurement.source_surfaces import generate_surface_sources
from runtime.randomness import (
    named_random_generator,
    named_rng_provenance,
    named_stream_seed,
    normalize_random_seed,
)
from runtime.experiment_profiles import (
    require_experiment_profile,
    require_private_scene_variant,
)
from runtime_environment import attach_random_manchester_transport_geometry
from sim.runtime import load_production_runtime_config

_MAX_FRESH_SEED = (1 << 48) - 18


def generate_fresh_scene_seed() -> int:
    """Return a fresh JSON-safe seed for one independent physical scene."""
    return 1 + secrets.randbelow(_MAX_FRESH_SEED)


def _source_payload(source: object) -> dict[str, object]:
    """Serialize one generated continuous-surface source contract."""
    return {
        "isotope": str(source.isotope),
        "position": [float(value) for value in source.position],
        "transport_position": [float(value) for value in source.transport_position],
        "intensity_cps_1m": float(source.intensity_cps_1m),
        "surface_chart_id": int(source.surface_chart_id),
        "surface_uv": [float(value) for value in source.surface_uv],
        "surface_normal": [float(value) for value in source.surface_normal],
        "surface_emission_policy_sha256": str(source.surface_emission_policy_sha256),
    }


def _adaptive_measurement_payload(
    config: Mapping[str, object],
    *,
    size_z: float,
    candidate_count: int,
    candidate_seed: int,
) -> dict[str, object]:
    """Return the complete explicit standard adaptive-motion contract."""
    detector = config["detector_model"]
    if not isinstance(detector, Mapping):
        raise TypeError("Production detector_model must be an object.")
    crystal_radius = detector["crystal_radius_m"]
    housing_thickness = detector["housing_thickness_m"]
    for name, value in (
        ("crystal_radius_m", crystal_radius),
        ("housing_thickness_m", housing_thickness),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"detector_model.{name} must be a JSON number.")
    head_radius = float(crystal_radius) + float(housing_thickness)
    base_height = 0.2
    detector_height_min = base_height + head_radius
    return {
        "candidate_count": int(candidate_count),
        "candidate_seed": int(candidate_seed),
        "detector_height_min_m": detector_height_min,
        "detector_height_max_m": float(size_z) - head_radius,
        "local_refinement_count": 64,
        "local_refinement_radius_m": 0.5,
        "base_radius_m": 0.2,
        "base_height_m": base_height,
        "mast_radius_m": 0.03,
        "head_radius_m": head_radius,
        "transport_height_m": detector_height_min,
        "horizontal_speed_m_s": 0.5,
        "vertical_speed_m_s": 0.25,
        "settling_time_s": 1.0,
        "shield_angular_speed_rad_s": float(np.pi / 4.0),
    }


def build_random_surface_scenario(
    *,
    scene_seed: int,
    measurement_log_output_dir: str | Path,
    run_id: str,
    experiment_profile_id: str,
    scene_variant_id: str,
    runtime_config_path: str | Path | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build one action-free private scenario from a runtime experiment.

    The returned object contains realized physical truth and therefore belongs
    outside estimator repositories. It intentionally contains no station,
    route, view-count, shield-program, or stopping-policy field.
    """
    seed = normalize_random_seed(scene_seed)
    profile = require_experiment_profile(experiment_profile_id)
    scene_variant = require_private_scene_variant(
        profile.profile_id,
        scene_variant_id,
    )
    isotope_sequence = scene_variant.isotope_sequence
    candidate_isotopes = profile.candidate_isotopes
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a nonempty string.")
    runtime_root = Path(__file__).resolve().parents[2]
    config_path = (
        (runtime_root / profile.runtime_config_relative_path).resolve()
        if runtime_config_path is None
        else Path(runtime_config_path).expanduser().resolve()
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Runtime configuration is missing: {config_path}")
    config = load_production_runtime_config(config_path)
    obstacle_height_m = float(config["obstacle_height_m"])
    chart_max_edge_m = float(profile.surface_chart_max_edge_m)
    include_room_boundaries = config["author_room_boundary_prims"]
    room_boundary_thickness_m = float(profile.room_boundary_thickness_m)
    environment_model_id = profile.environment_model_id
    environment = profile.environment
    obstacle_seed = named_stream_seed(seed, "physical_obstacle_environment")
    candidate_seed = named_stream_seed(seed, "adaptive_candidate_workspace")
    grid = build_obstacle_grid(
        mode="random",
        path=None,
        size_x=environment.size_x,
        size_y=environment.size_y,
        cell_size=1.0,
        blocked_fraction=float(profile.blocked_fraction),
        rng_seed=obstacle_seed,
        keep_free_points=[
            (environment.detector_position[0], environment.detector_position[1])
        ],
        passage_width_m=float(profile.passage_width_m),
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
        intensity_cps_1m=profile.intensity_cps_1m,
        rng=named_random_generator(seed, "physical_surface_sources"),
        count=len(isotope_sequence),
        obstacle_height_m=obstacle_height_m,
        chart_max_edge_m=chart_max_edge_m,
        same_isotope_min_distance_m=float(profile.same_isotope_min_distance_m),
    )
    obstacle_payload = obstacle_instances_to_dicts(obstacle_instances)
    environment_payload: dict[str, object] = {
        **profile.public_environment_fields(),
        "environment_model_id": environment_model_id,
        "size_x": float(environment.size_x),
        "size_y": float(environment.size_y),
        "size_z": float(environment.size_z),
        "detector_position": [float(value) for value in environment.detector_position],
        "obstacle_grid": grid.to_dict(),
        "obstacle_instances": obstacle_payload,
        "adaptive_measurement": _adaptive_measurement_payload(
            config,
            size_z=float(environment.size_z),
            candidate_count=int(profile.candidate_count),
            candidate_seed=int(candidate_seed),
        ),
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
        "absorber_transport_group": grid.absorber_transport_group,
        "absorber_transport_boxes_m": [
            list(box) for box in grid.absorber_transport_boxes_m
        ],
        "absorber_transport_contract_sha256": (grid.absorber_transport_contract_sha256),
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
        "obstacle_material": profile.obstacle_material,
        "author_obstacle_prims": config["author_obstacle_prims"],
        "author_room_boundary_prims": include_room_boundaries,
        "usd_path": config["usd_path"],
        "use_config_usd_fallback": False,
    }
    run_metadata = dict(metadata or {})
    run_metadata.update(
        {
            "scenario_family": (
                "random_physical_surface_v1:"
                f"{profile.profile_id}:{scene_variant.variant_id}"
            ),
            "experiment_profile_id": profile.profile_id,
            "private_scene_variant_id": scene_variant.variant_id,
            "scene_seed": int(seed),
            "scene_rng_provenance": named_rng_provenance(
                seed,
                (
                    "physical_obstacle_environment",
                    "physical_surface_sources",
                    "adaptive_candidate_workspace",
                ),
            ),
            "same_isotope_min_distance_m": float(profile.same_isotope_min_distance_m),
            "measurement_actions_precomputed": False,
        }
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "backend": "geant4",
        "runtime_config_path": config_path.as_posix(),
        "output_dir": (
            Path(measurement_log_output_dir).expanduser().resolve().as_posix()
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
    return _write_private_json_exclusive(
        target,
        scenario,
        artifact_name="private scenario",
    )


def _write_private_json_exclusive(
    target: Path,
    payload: Mapping[str, object],
    *,
    artifact_name: str,
) -> Path:
    """Create one owner-only JSON file without an overwrite race."""
    serialized = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.parent.chmod(0o700)
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        raise FileExistsError(
            f"Refusing to replace {artifact_name} {target}."
        ) from None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target


def build_private_truth_manifest(
    scenario: Mapping[str, object],
) -> dict[str, object]:
    """Return private evaluation truth joined to estimator output by run ID."""
    if not isinstance(scenario, Mapping):
        raise TypeError("scenario must be a mapping.")
    run_id = scenario.get("run_id")
    scene = scenario.get("scene")
    metadata = scenario.get("metadata")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("scenario.run_id must be a nonempty string.")
    if not isinstance(scene, Mapping) or not isinstance(scene.get("sources"), list):
        raise ValueError("scenario.scene.sources must be a JSON array.")
    if not isinstance(metadata, Mapping):
        raise ValueError("scenario.metadata must be a JSON object.")
    required_metadata = (
        "experiment_profile_id",
        "private_scene_variant_id",
        "scene_seed",
        "scene_rng_provenance",
    )
    missing = [key for key in required_metadata if key not in metadata]
    if missing:
        raise ValueError(
            "Private scenario metadata lacks truth-manifest fields: "
            + ", ".join(missing)
        )
    return json.loads(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "experiment_profile_id": metadata["experiment_profile_id"],
                "scene_variant_id": metadata["private_scene_variant_id"],
                "scene_seed": metadata["scene_seed"],
                "scene_rng_provenance": metadata["scene_rng_provenance"],
                "sources": scene["sources"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


def write_private_truth_manifest(
    path: str | Path,
    manifest: Mapping[str, object],
) -> Path:
    """Publish one immutable private truth manifest outside estimator artifacts."""
    target = Path(path).expanduser().resolve()
    return _write_private_json_exclusive(
        target,
        manifest,
        artifact_name="private truth manifest",
    )


__all__ = [
    "build_private_truth_manifest",
    "build_random_surface_scenario",
    "generate_fresh_scene_seed",
    "write_private_scenario",
    "write_private_truth_manifest",
]
