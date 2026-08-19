"""Tests for runtime-owned private physical scenario authoring."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from measurement.obstacles import ObstacleGrid
from runtime.adaptive import AdaptiveCandidateProvider
from runtime.scenarios import (
    build_random_ral_mix9_scenario,
    write_private_scenario,
)


def _runtime_config() -> Path:
    """Return the production Geant4 configuration used by scenario tests."""
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )


def _scenario(tmp_path: Path, *, seed: int = 123) -> dict[str, object]:
    """Build one deterministic action-free private scenario."""
    return build_random_ral_mix9_scenario(
        scene_seed=seed,
        runtime_config_path=_runtime_config(),
        measurement_log_output_dir=tmp_path / "measurement-log",
        run_id=f"scenario-{seed}",
        candidate_count=32,
    )


def test_ral_scenario_contains_physics_but_no_estimator_plan(
    tmp_path: Path,
) -> None:
    """A runtime scenario must not precompute estimator actions or budgets."""
    scenario = _scenario(tmp_path)

    def field_names(value: object) -> set[str]:
        """Return every object field name in a nested JSON-like value."""
        if isinstance(value, dict):
            return set(value).union(
                *(field_names(item) for item in value.values())
            )
        if isinstance(value, list):
            return set().union(*(field_names(item) for item in value))
        return set()

    assert set(scenario) == {
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
    fields = field_names(scenario)
    for forbidden in (
        "actions",
        "station_count",
        "view_count",
        "shield_program",
        "max_measurements",
        "stopping_rule",
        "num_particles",
        "dss_pp",
        "mle_config",
        "planning_config",
        "estimator_profile",
    ):
        assert forbidden not in fields
    assert scenario["metadata"]["measurement_actions_precomputed"] is False


def test_ral_scenario_has_exact_mix9_and_resolvable_same_isotope_spacing(
    tmp_path: Path,
) -> None:
    """Runtime truth must satisfy the fixed RA-L counts and spacing contract."""
    scenario = _scenario(tmp_path)
    sources = scenario["scene"]["sources"]
    counts = Counter(source["isotope"] for source in sources)

    assert counts == {"Co-60": 3, "Cs-137": 4, "Eu-154": 2}
    for isotope in counts:
        positions = np.asarray(
            [
                source["position"]
                for source in sources
                if source["isotope"] == isotope
            ],
            dtype=float,
        )
        if len(positions) < 2:
            continue
        distances = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :],
            axis=2,
        )
        distances[np.eye(len(positions), dtype=bool)] = np.inf
        assert float(np.min(distances)) >= 3.0 - 1.0e-12


def test_ral_cs4_co3_eu0_uses_cs_co_candidate_contract(
    tmp_path: Path,
) -> None:
    """The explicit Cs/Co experiment must not infer an excluded Eu isotope."""
    scenario = build_random_ral_mix9_scenario(
        scene_seed=321,
        runtime_config_path=_runtime_config(),
        measurement_log_output_dir=tmp_path / "measurement-log",
        run_id="scenario-cs4-co3-eu0",
        candidate_count=32,
        source_profile="ral-cs4-co3-eu0",
    )
    sources = scenario["scene"]["sources"]
    counts = Counter(source["isotope"] for source in sources)

    assert counts == {"Co-60": 3, "Cs-137": 4}
    assert set(scenario["isotopes"]) == {"Co-60", "Cs-137"}
    assert set(
        scenario["scene"]["transport_mu_by_isotope"]
    ) == {"Co-60", "Cs-137"}
    assert scenario["metadata"]["private_source_profile"] == (
        "ral-cs4-co3-eu0"
    )


def test_ral_scenario_seed_is_deterministic_and_candidates_are_reachable(
    tmp_path: Path,
) -> None:
    """One scene seed must reproduce truth and a valid runtime workspace."""
    first = _scenario(tmp_path / "first")
    second = _scenario(tmp_path / "second")

    assert first["scene"] == second["scene"]
    assert first["environment"] == second["environment"]
    environment = first["environment"]
    grid = ObstacleGrid.from_dict(environment["obstacle_grid"])
    provider = AdaptiveCandidateProvider(environment, grid)
    snapshot = provider.snapshot(provider.initial_pose, current_pair_id=0)

    assert len(snapshot.candidate_poses_xyz) == 32
    assert snapshot.allowed_pair_ids == tuple(range(64))
    assert all(grid.is_free(pose) for pose in snapshot.candidate_poses_xyz)


def test_private_scenario_writer_refuses_overwrite(tmp_path: Path) -> None:
    """Private truth artifacts must be immutable after publication."""
    scenario = _scenario(tmp_path)
    target = write_private_scenario(tmp_path / "scenario.json", scenario)

    assert target.is_file()
    try:
        write_private_scenario(target, scenario)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Private scenario overwrite was not rejected.")
