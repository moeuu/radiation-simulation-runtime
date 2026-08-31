"""Tests for runtime-owned private physical scenario authoring."""

from __future__ import annotations

from collections import Counter
import json
import stat
from pathlib import Path

import numpy as np
import pytest

from measurement.obstacles import ObstacleGrid
from runtime.adaptive import AdaptiveCandidateProvider
from runtime.cli import _build_parser, main as runtime_cli_main
from runtime.experiment_profiles import (
    CS_CO_SURFACE_SEARCH_PROFILE,
    MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE,
)
from runtime.scenarios import (
    build_private_truth_manifest,
    build_random_surface_scenario,
    write_private_scenario,
    write_private_truth_manifest,
)


def _scenario(tmp_path: Path, *, seed: int = 123) -> dict[str, object]:
    """Build one deterministic action-free private scenario."""
    return build_random_surface_scenario(
        scene_seed=seed,
        measurement_log_output_dir=tmp_path / "measurement-log",
        run_id=f"scenario-{seed}",
        experiment_profile_id=MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.profile_id,
        scene_variant_id="mix9",
    )


def test_scenario_contains_physics_and_runtime_contract_but_no_estimator_plan(
    tmp_path: Path,
) -> None:
    """A runtime scenario must not precompute estimator actions or budgets."""
    scenario = _scenario(tmp_path)

    def field_names(value: object) -> set[str]:
        """Return every object field name in a nested JSON-like value."""
        if isinstance(value, dict):
            return set(value).union(*(field_names(item) for item in value.values()))
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
        "stopping_rule",
        "num_particles",
        "dss_pp",
        "mle_config",
        "planning_config",
        "estimator_profile",
    ):
        assert forbidden not in fields
    assert scenario["metadata"]["measurement_actions_precomputed"] is False
    assert scenario["scene"]["obstacle_material"] == (
        MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.obstacle_material
    )
    assert scenario["scene"]["use_config_usd_fallback"] is False
    assert scenario["scene"]["usd_path"]


def test_scenario_derives_every_room_payload_from_one_profile(
    tmp_path: Path,
) -> None:
    """The runtime-owned profile must drive all published room bounds."""
    scenario = _scenario(tmp_path)
    expected = (
        MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.environment.size_x,
        MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.environment.size_y,
        MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.environment.size_z,
    )
    environment = scenario["environment"]

    assert expected == (10.0, 15.0, 5.0)
    assert (
        environment["size_x"],
        environment["size_y"],
        environment["size_z"],
    ) == expected
    assert tuple(scenario["scene"]["room_size_xyz"]) == expected
    assert environment["environment_model_id"] == (
        MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.environment_model_id
    )
    assert environment["acquisition_contract"] == (
        MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.acquisition.to_payload()
    )
    assert set(environment["adaptive_measurement"]) == {
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


def test_explicit_mix9_scenario_has_exact_counts_and_same_isotope_spacing(
    tmp_path: Path,
) -> None:
    """Runtime truth must satisfy the private counts and spacing contract."""
    scenario = _scenario(tmp_path)
    sources = scenario["scene"]["sources"]
    counts = Counter(source["isotope"] for source in sources)

    assert counts == {"Co-60": 3, "Cs-137": 4, "Eu-154": 2}
    for isotope in counts:
        positions = np.asarray(
            [source["position"] for source in sources if source["isotope"] == isotope],
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


def test_explicit_cs_co_profile_has_no_unrequested_eu_candidate(
    tmp_path: Path,
) -> None:
    """The requested Cs/Co environment exposes only its two candidates."""
    scenario = build_random_surface_scenario(
        scene_seed=321,
        measurement_log_output_dir=tmp_path / "measurement-log",
        run_id="scenario-cs4-co3",
        experiment_profile_id=CS_CO_SURFACE_SEARCH_PROFILE.profile_id,
        scene_variant_id="cs4-co3",
    )
    sources = scenario["scene"]["sources"]
    counts = Counter(source["isotope"] for source in sources)

    assert counts == {"Co-60": 3, "Cs-137": 4}
    assert set(scenario["isotopes"]) == {"Co-60", "Cs-137"}
    assert set(scenario["scene"]["transport_mu_by_isotope"]) == {
        "Co-60",
        "Cs-137",
    }
    assert scenario["metadata"]["private_scene_variant_id"] == "cs4-co3"


def test_scenario_seed_is_deterministic_and_candidates_are_reachable(
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

    assert len(snapshot.candidate_poses_xyz) == 256
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


def test_scenario_cli_selects_only_runtime_owned_profiles() -> None:
    """The scenario CLI must not expose duplicated physical-value overrides."""
    parser = _build_parser()
    parsed = parser.parse_args(
        [
            "generate-scenario",
            "/private/scenario.json",
            "--truth-manifest-output",
            "/private/truth.json",
            "--measurement-log-output",
            "/private/measurement-log",
            "--run-id",
            "run-001",
            "--experiment-profile",
            CS_CO_SURFACE_SEARCH_PROFILE.profile_id,
            "--scene-variant",
            "cs4-co3",
        ]
    )

    assert parsed.experiment_profile == (CS_CO_SURFACE_SEARCH_PROFILE.profile_id)
    assert parsed.scene_variant == "cs4-co3"
    assert not hasattr(parsed, "candidate_count")
    with pytest.raises(SystemExit):
        parser.parse_args(["generate-ral-scenario"])


@pytest.mark.parametrize("omitted", ("experiment_profile", "scene_variant"))
def test_scenario_cli_has_no_implicit_environment_selection(omitted: str) -> None:
    """Scenario authoring must require the requested profile and source variant."""
    arguments = [
        "generate-scenario",
        "/private/scenario.json",
        "--truth-manifest-output",
        "/private/truth.json",
        "--measurement-log-output",
        "/private/measurement-log",
        "--run-id",
        "run-001",
    ]
    if omitted != "experiment_profile":
        arguments.extend(
            [
                "--experiment-profile",
                MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.profile_id,
            ]
        )
    if omitted != "scene_variant":
        arguments.extend(["--scene-variant", "mix9"])

    with pytest.raises(SystemExit):
        _build_parser().parse_args(arguments)


def test_adaptive_cli_rejects_retired_resume_flags() -> None:
    """Production adaptive commands must provide no resume entrypoint."""
    parser = _build_parser()
    for command in ("run-adaptive-session", "serve-adaptive-session-socket"):
        arguments = [command, "/private/scenario.json"]
        if command == "serve-adaptive-session-socket":
            arguments.extend(["--socket-path", "/private/session.sock"])
        arguments.extend(["--resume-stage", "/private/.measurement-log.stream-7"])
        with pytest.raises(SystemExit):
            parser.parse_args(arguments)


def test_production_cli_rejects_retired_fixed_run_plan() -> None:
    """No predeclared batch plan may publish a production MeasurementLog."""
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["run-plan", "/private/plan.json"])


def test_serve_cli_rejects_unknown_runtime_config_before_binding(
    tmp_path: Path,
) -> None:
    """The public bridge server must use the exact production loader."""
    runtime_root = Path(__file__).resolve().parents[1]
    config_path = (
        runtime_root / MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.runtime_config_relative_path
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["thred_count"] = payload["thread_count"]
    invalid_path = tmp_path / "invalid-runtime.json"
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown_or_retired.*thred_count"):
        runtime_cli_main(["serve", "--config", str(invalid_path)])


def test_serve_cli_rejects_invalid_model_registry_before_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public server must resolve registry identity before binding a port."""
    runtime_root = Path(__file__).resolve().parents[1]
    config_path = (
        runtime_root / MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.runtime_config_relative_path
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["full_spectrum_model_registry_path"] = "missing-registry.json"
    payload["full_spectrum_model_registry_file_sha256"] = "0" * 64
    invalid_path = tmp_path / "invalid-registry.json"
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")
    bound = False

    def bind_server(*args: object, **kwargs: object) -> None:
        """Record any forbidden server bind after failed model preflight."""
        nonlocal bound
        bound = True

    monkeypatch.setattr("runtime.cli.serve_forever", bind_server)

    with pytest.raises(FileNotFoundError, match="missing-registry"):
        runtime_cli_main(["serve", "--config", str(invalid_path)])

    assert bound is False


def test_serve_cli_accepts_catalog_independent_profile_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-domain profile with transferable approval may start a server."""
    runtime_root = Path(__file__).resolve().parents[1]
    config_path = (
        runtime_root / MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.runtime_config_relative_path
    )
    bound = False

    def bind_server(*args: object, **kwargs: object) -> None:
        """Record server bind after successful model approval."""
        nonlocal bound
        bound = True

    monkeypatch.setattr("runtime.cli.serve_forever", bind_server)

    assert runtime_cli_main(["serve", "--config", str(config_path)]) == 0

    assert bound is True


def test_private_truth_manifest_is_separate_and_joined_by_run_id(
    tmp_path: Path,
) -> None:
    """Evaluation truth must publish privately without entering MeasurementLog."""
    scenario = _scenario(tmp_path)
    manifest = build_private_truth_manifest(scenario)
    target = write_private_truth_manifest(tmp_path / "truth.json", manifest)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["run_id"] == scenario["run_id"]
    assert payload["sources"] == scenario["scene"]["sources"]
    assert payload["experiment_profile_id"] == (
        MULTI_ISOTOPE_SURFACE_SEARCH_PROFILE.profile_id
    )
    assert payload["scene_variant_id"] == "mix9"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        write_private_truth_manifest(target, manifest)
