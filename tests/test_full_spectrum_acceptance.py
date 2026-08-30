"""Tests for fail-closed physics-only all-64 validation aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from measurement.source_boundary import (
    SURFACE_EMISSION_EPSILON_M,
    surface_emission_policy_sha256,
)
import spectrum.full_spectrum_acceptance as acceptance
from spectrum.full_spectrum_acceptance import (
    ACCEPTANCE_SCENE_ARTIFACT_SCHEMA_VERSION,
    SURFACE_BOUNDARY_GATE_SCHEMA_VERSION,
    SURFACE_BOUNDARY_NATIVE_POSITION_VARIANTS,
    SURFACE_BOUNDARY_PROBE_DWELL_TIME_S,
    build_independent_validation_manifest,
)
from spectrum.transport_spectral import (
    ACCEPTANCE_METRIC_CONTRACT,
    DESIGNATED_VALIDATION_SCENE_SEEDS,
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    VALIDATION_SCENARIO_IDS,
)
from tests.green_test_support import (
    synthetic_detector_green_operator,
    synthetic_detector_green_validation_manifest,
)


@pytest.fixture(autouse=True)
def _canonical_test_operator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind aggregation tests to one explicit synthetic Green operator."""
    operator = synthetic_detector_green_operator()
    monkeypatch.setattr(
        acceptance,
        "_canonical_detector_green_operator",
        lambda: operator,
    )


def _operator():
    """Return the deterministic operator used by the autouse fixture."""
    return synthetic_detector_green_operator()


def _metric_values(*, passing: bool = True) -> dict[str, float]:
    """Return deterministic per-scene metrics at fixed thresholds."""
    values = {
        metric_id: float(threshold)
        for metric_id, (_, threshold) in ACCEPTANCE_METRIC_CONTRACT.items()
    }
    if not passing:
        values["background_pairwise_95_coverage_fraction"] = 0.0
    return values


def _write_payload(path: Path, payload: object) -> None:
    """Write one canonical immutable acceptance fixture."""
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_artifact(
    root: Path,
    *,
    scene_seed: int,
    metrics: dict[str, float] | None = None,
) -> Path:
    """Write one complete synthetic validation-scene artifact."""
    operator = _operator()
    payload = {
        "schema_version": ACCEPTANCE_SCENE_ARTIFACT_SCHEMA_VERSION,
        "acceptance_contract_sha256": (FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256),
        "acceptance_run_contract_sha256": "c" * 64,
        "runtime_config_sha256": "7" * 64,
        "native_executable_sha256": "2" * 64,
        "native_execution_environment_sha256": "3" * 64,
        "implementation_bundle_sha256": "4" * 64,
        "scene_seed": scene_seed,
        "split": "validation",
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "shield_pair_ids": list(range(64)),
        "pair_ids_by_scenario": {
            scenario: list(range(64)) for scenario in VALIDATION_SCENARIO_IDS
        },
        "observation_count_by_scenario": {
            scenario: 64 for scenario in VALIDATION_SCENARIO_IDS
        },
        "approved_model_contract_sha256": "a" * 64,
        "detector_green_operator_contract_sha256": (operator.contract_hash_sha256),
        "detector_green_operator_binary_sha256": operator.binary_sha256,
        "additive_scatter_contract_sha256": "b" * 64,
        "surface_emission_policy_sha256": (surface_emission_policy_sha256()),
        "scene_hash_by_scenario": {
            scenario: f"{scene_seed + index:064x}"
            for index, scenario in enumerate(VALIDATION_SCENARIO_IDS)
        },
        "surface_source_contract_sha256_by_scenario": {
            scenario: (f"{scene_seed + len(VALIDATION_SCENARIO_IDS) + index:064x}")
            for index, scenario in enumerate(VALIDATION_SCENARIO_IDS)
        },
        "surface_boundary_gate": {
            "schema_version": SURFACE_BOUNDARY_GATE_SCHEMA_VERSION,
            "surface_emission_policy_sha256": (surface_emission_policy_sha256()),
            "surface_emission_epsilon_m": SURFACE_EMISSION_EPSILON_M,
            "probe_dwell_time_s": SURFACE_BOUNDARY_PROBE_DWELL_TIME_S,
            "native_position_variants": list(SURFACE_BOUNDARY_NATIVE_POSITION_VARIANTS),
            "exact_anchor_vs_air_gate_passed": True,
            "solid_minus_air_gate_passed": True,
            "passed": True,
        },
        "metrics": _metric_values() if metrics is None else metrics,
    }
    path = root / f"scene_{scene_seed}.json"
    _write_payload(path, payload)
    return path


def _all_artifacts(
    root: Path,
    *,
    failing_validation: bool = False,
) -> list[Path]:
    """Write exactly the five predeclared independent validation scenes."""
    paths: list[Path] = []
    for seed in DESIGNATED_VALIDATION_SCENE_SEEDS:
        paths.append(
            _write_artifact(
                root,
                scene_seed=seed,
                metrics=(
                    _metric_values(passing=False)
                    if failing_validation
                    and seed == DESIGNATED_VALIDATION_SCENE_SEEDS[0]
                    else _metric_values()
                ),
            )
        )
    return paths


def _green_validation() -> dict[str, object]:
    """Return passing generic monoenergetic validation evidence."""
    return synthetic_detector_green_validation_manifest(_operator())


def _aggregate(paths: list[Path]) -> dict[str, object]:
    """Aggregate fixtures with their exact synthetic Green evidence."""
    return build_independent_validation_manifest(
        paths,
        detector_green_validation_manifest=_green_validation(),
    )


def test_manifest_is_validation_only_and_contains_no_scene_fit(
    tmp_path: Path,
) -> None:
    """Approval must record zero calibration and no candidate selection."""
    manifest = _aggregate(_all_artifacts(tmp_path))

    assert manifest["schema_version"] == 6
    assert manifest["validation_scene_seeds"] == list(DESIGNATED_VALIDATION_SCENE_SEEDS)
    assert manifest["candidate_selection"] == ("none_predeclared_physics_only")
    assert manifest["scene_calibration_count"] == 0
    assert manifest["metric_split"] == "independent_validation_only"
    assert manifest["metric_aggregation"] == (
        "validation_scene_conservative_worst_case"
    )
    assert not any("training" in key for key in manifest)
    assert manifest["all_passed"] is True


def test_one_failed_validation_scene_blocks_approval(tmp_path: Path) -> None:
    """A failed independent scene must survive conservative aggregation."""
    manifest = _aggregate(_all_artifacts(tmp_path, failing_validation=True))

    result = manifest["metrics"]["background_pairwise_95_coverage_fraction"]
    assert result["value"] == 0.0
    assert result["passed"] is False
    assert manifest["all_passed"] is False


def test_scene_artifact_requires_all_64_pairs(tmp_path: Path) -> None:
    """A nonempty but incomplete shield-pair set must fail closed."""
    paths = _all_artifacts(tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["shield_pair_ids"] = list(range(63))
    _write_payload(paths[0], payload)

    with pytest.raises(ValueError, match="all-64"):
        _aggregate(paths)


def test_every_scenario_requires_all_64_pairs(tmp_path: Path) -> None:
    """A global pair list cannot hide one incomplete scenario."""
    paths = _all_artifacts(tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["pair_ids_by_scenario"][VALIDATION_SCENARIO_IDS[1]] = list(range(63))
    _write_payload(paths[0], payload)

    with pytest.raises(ValueError, match="independently cover all 64"):
        _aggregate(paths)


def test_missing_fixed_metric_fails_closed(tmp_path: Path) -> None:
    """Validation cannot silently omit an unfavorable fixed metric."""
    paths = _all_artifacts(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    del payload["metrics"]["absent_isotope_k_positive_decision_rate_at_p0p95"]
    _write_payload(paths[-1], payload)

    with pytest.raises(ValueError, match="fixed contract"):
        _aggregate(paths)


def test_unknown_scene_seed_has_no_fallback_split(tmp_path: Path) -> None:
    """An arbitrary historical seed cannot enter formal validation."""
    paths = _all_artifacts(tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["scene_seed"] = 2026072791
    _write_payload(paths[0], payload)

    with pytest.raises(ValueError, match="outside the predeclared"):
        _aggregate(paths)


def test_native_signed_epsilon_gate_is_mandatory(tmp_path: Path) -> None:
    """A scene without the solid-side counterfactual cannot be approved."""
    paths = _all_artifacts(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["surface_boundary_gate"]["solid_minus_air_gate_passed"] = False
    _write_payload(paths[-1], payload)

    with pytest.raises(ValueError, match="signed-epsilon"):
        _aggregate(paths)


@pytest.mark.parametrize(
    "field_name",
    (
        "acceptance_run_contract_sha256",
        "runtime_config_sha256",
        "native_executable_sha256",
        "native_execution_environment_sha256",
        "implementation_bundle_sha256",
    ),
)
def test_all_scenes_require_one_native_execution_contract(
    tmp_path: Path,
    field_name: str,
) -> None:
    """One scene from another native build cannot enter model approval."""
    paths = _all_artifacts(tmp_path)
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload[field_name] = "0" * 64
    _write_payload(paths[-1], payload)

    with pytest.raises(ValueError, match="native execution contracts"):
        _aggregate(paths)


def test_green_validation_must_match_application_provenance(
    tmp_path: Path,
) -> None:
    """Generic detector validation from another binary cannot authorize use."""
    paths = _all_artifacts(tmp_path)
    evidence = _green_validation()
    evidence["native_executable_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="differs from construction"):
        build_independent_validation_manifest(
            paths,
            detector_green_validation_manifest=evidence,
        )


def test_green_and_application_runtime_configs_are_separate(
    tmp_path: Path,
) -> None:
    """Detector and application runs may use purpose-specific configs."""
    paths = _all_artifacts(tmp_path)
    evidence = synthetic_detector_green_validation_manifest(
        _operator(),
        runtime_config_sha256="8" * 64,
    )
    assert evidence["runtime_config_sha256"] != "7" * 64

    manifest = build_independent_validation_manifest(
        paths,
        detector_green_validation_manifest=evidence,
    )

    assert manifest["runtime_config_sha256"] == "7" * 64
    assert manifest["detector_green_validation"]["runtime_config_sha256"] == (
        evidence["runtime_config_sha256"]
    )


def test_scene_green_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    """A scene generated with another detector operator cannot be aggregated."""
    paths = _all_artifacts(tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["detector_green_operator_contract_sha256"] = "0" * 64
    _write_payload(paths[0], payload)

    with pytest.raises(ValueError, match="physics contracts"):
        _aggregate(paths)
