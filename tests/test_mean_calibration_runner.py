"""Tests for the isolated fixed-quota Geant4 calibration runner."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from runtime.experiment_profiles import STANDARD_ACQUISITION_LIVE_TIME_S
from scripts.run_mean_calibration import (
    _design as cli_design,
    _parser as calibration_cli_parser,
)
from spectrum.mean_calibration import parse_mean_calibration_metadata
from spectrum.mean_calibration_runner import (
    MeanCalibrationLayout,
    MeanCalibrationTrainingRow,
    build_mean_calibration_app_payload,
    build_mean_calibration_completion_manifest,
    canonical_json_sha256,
    build_predeclared_mean_calibration_design,
    fit_additive_scatter_training_rows,
    freeze_mean_calibration_completion_manifest,
    freeze_mean_calibration_scene_manifest,
    freeze_runtime_ready_model,
    initialize_mean_calibration_layout,
    load_mean_calibration_pair_artifact,
    validate_predeclared_mean_calibration_design,
    write_mean_calibration_pair_artifact,
)
from spectrum.transport_spectral import (
    DESIGNATED_TRAINING_SCENE_SEEDS,
    GeometryConditionedSpectralModel,
)


def _design() -> dict[str, object]:
    """Return a small valid quota with the full immutable training identity."""
    return build_predeclared_mean_calibration_design(
        histories_per_source_line=256,
        angle_strata_mu=8,
        angle_strata_phi=2,
    )


def _metadata() -> dict[str, object]:
    """Return two angular strata for one synthetic source line."""
    payload: dict[str, object] = {
        "mean_calibration_enabled": True,
        "primary_schedule_mode": (
            "fixed_source_line_stratified_mean_calibration"
        ),
        "transport_history_mode": (
            "fixed_source_line_stratified_weighted_mean"
        ),
        "transport_tally_weighted": True,
        "history_thinning_enabled": False,
        "mean_calibration_forced_collision": False,
        "mean_calibration_history_weight_semantics": (
            "expected_source_line_mean_divided_by_fixed_quota"
        ),
        "mean_calibration_covariance_semantics": (
            "independent_mu_phi_stratum_sample_mean_cluster_"
            "sufficient_statistics_v1"
        ),
        "spectrum_variance_semantics": (
            "stratified_fixed_quota_sample_mean_covariance"
        ),
        "spectrum_bin_count": 3,
        "mean_calibration_histories_per_source_line": 4,
        "mean_calibration_angle_strata_mu": 2,
        "mean_calibration_angle_strata_phi": 1,
        "mean_calibration_angle_stratum_count": 2,
        "primary_history_batch_count": 2,
    }
    rows = (
        ("0:1", "1:1", "-", 0),
        ("-", "1:1", "-", 1),
    )
    for index, (uncollided, interacted, secondary, stratum) in enumerate(rows):
        prefix = f"mean_calibration_batch_{index}_"
        payload.update(
            {
                prefix + "source_token": "src0_Cs-137",
                prefix + "line_token": "src0_Cs-137_e662p0",
                prefix + "expected_unthinned_histories": 4.0,
                prefix + "sampled_histories": 2,
                prefix + "history_weight": 2.0,
                prefix + "angle_stratum_index": stratum,
                prefix
                + "sparse_entry_histogram_uncollided_primary": uncollided,
                prefix
                + "sparse_entry_histogram_interacted_primary": interacted,
                prefix + "sparse_entry_histogram_secondary": secondary,
            }
        )
    return payload


def test_design_is_predeclared_training_only_and_holdout_optional() -> None:
    """Training design must be exact and must not consume holdout evidence."""
    design = _design()

    assert validate_predeclared_mean_calibration_design(design) == design
    assert design["holdout_consumed_by_training"] is False
    assert (
        design["all64_holdout_role"]
        == "optional_independent_release_evidence"
    )
    assert design["forced_collision"] is False

    corrupted = dict(design)
    corrupted["scenario_ids"] = ["one_known_replay"]
    with pytest.raises(ValueError, match="exact predeclared"):
        validate_predeclared_mean_calibration_design(corrupted)


def test_calibration_payload_isolated_from_standard_runtime() -> None:
    """Calibration overrides must not mutate or weaken the runtime config."""
    repository_root = Path(__file__).resolve().parents[1]
    config_path = (
        repository_root
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    standard = json.loads(config_path.read_text(encoding="utf-8"))
    before = copy.deepcopy(standard)

    payload = build_mean_calibration_app_payload(
        standard,
        repository_root=repository_root,
        design=_design(),
    )

    assert standard == before
    assert standard["sample_detector_response"] is True
    assert standard["background_cps"] > 0.0
    assert standard["dead_time_tau_s"] > 0.0
    assert payload["sample_detector_response"] is False
    assert payload["background_cps"] == 0.0
    assert payload["dead_time_tau_s"] == 0.0
    assert payload["detector_scoring_mode"] == "incident_gamma_energy"
    assert payload["secondary_transport_mode"] == "full_transport"
    assert payload["primary_sampling_fraction"] == 1.0
    assert payload["target_sampled_primaries"] is None
    assert payload["accelerated_weighted_transport_enable"] is False
    assert payload["mean_calibration_histories_per_source_line"] == 256
    assert payload["mean_calibration_angle_strata_mu"] == 8
    assert payload["mean_calibration_angle_strata_phi"] == 2

    missing_executable = dict(standard)
    del missing_executable["executable_path"]
    with pytest.raises(ValueError, match="explicit executable_path"):
        build_mean_calibration_app_payload(
            missing_executable,
            repository_root=repository_root,
            design=_design(),
        )


def test_forced_collision_is_selected_only_by_calibration_design() -> None:
    """Only the immutable calibration builder may opt into forced collision."""
    repository_root = Path(__file__).resolve().parents[1]
    config_path = (
        repository_root
        / "configs"
        / "geant4"
        / "variance_reduction_external_no_isaac_32threads.json"
    )
    standard = json.loads(config_path.read_text(encoding="utf-8"))
    design = build_predeclared_mean_calibration_design(
        histories_per_source_line=16,
        angle_strata_mu=2,
        angle_strata_phi=2,
        forced_collision=True,
        training_scene_seeds=(2026072901,),
        scenario_ids=("single_line_source_resolved",),
        shield_pair_ids=(3, 11),
    )

    payload = build_mean_calibration_app_payload(
        standard,
        repository_root=repository_root,
        design=design,
    )

    assert design["forced_collision"] is True
    assert payload["mean_calibration_forced_collision"] is True
    assert "mean_calibration_forced_collision" not in standard


def test_cli_predeclares_forced_collision_and_small_training_scope() -> None:
    """CLI flags must become immutable design fields before acquisition."""
    arguments = calibration_cli_parser().parse_args(
        [
            "init",
            "--histories-per-source-line",
            "16",
            "--angle-strata-mu",
            "2",
            "--angle-strata-phi",
            "2",
            "--forced-collision",
            "--design-scene-seed",
            "2026072901",
            "--design-scenario-id",
            "single_line_source_resolved",
            "--design-pair-id",
            "7",
        ]
    )

    design = cli_design(arguments)

    assert design["forced_collision"] is True
    assert design["training_scene_seeds"] == [2026072901]
    assert design["scenario_ids"] == ["single_line_source_resolved"]
    assert design["shield_pair_ids"] == [7]


def test_smaller_predeclared_design_seals_only_declared_artifacts(
    tmp_path: Path,
) -> None:
    """Sealing must require exactly declared pairs, not an implicit all-64 set."""
    scene_seed = 2026072901
    scenario_id = "single_line_source_resolved"
    pair_id = 7
    design = build_predeclared_mean_calibration_design(
        histories_per_source_line=4,
        angle_strata_mu=2,
        angle_strata_phi=1,
        training_scene_seeds=(scene_seed,),
        scenario_ids=(scenario_id,),
        shield_pair_ids=(pair_id,),
    )
    layout = MeanCalibrationLayout(tmp_path / "small")
    initialize_mean_calibration_layout(layout=layout, design=design)
    calibration = parse_mean_calibration_metadata(
        _metadata(),
        bin_count=3,
    )
    covariance = calibration.raw_covariance()
    write_mean_calibration_pair_artifact(
        directory=layout.pair_directory(
            scene_seed=scene_seed,
            scenario_id=scenario_id,
            shield_pair_id=pair_id,
        ),
        provenance={
            "design_sha256": canonical_json_sha256(design),
            "scene_seed": scene_seed,
            "scenario_id": scenario_id,
            "shield_pair_id": pair_id,
            "holdout_artifacts_consumed": False,
        },
        calibration=calibration,
        native_spectrum=calibration.raw_mean(),
        native_spectrum_variance=np.diag(covariance),
        response_operator_br=np.eye(3, dtype=np.float64),
        geometry={},
    )

    scene_manifest = freeze_mean_calibration_scene_manifest(
        layout=layout,
        design=design,
        scene_seed=scene_seed,
    )
    completion = freeze_mean_calibration_completion_manifest(
        layout=layout,
        design=design,
    )
    scene_payload = json.loads(scene_manifest.read_text(encoding="utf-8"))
    completion_payload = json.loads(completion.read_text(encoding="utf-8"))

    assert scene_payload["shield_pair_ids"] == [pair_id]
    assert set(
        scene_payload["pair_manifest_sha256_by_scenario"][scenario_id]
    ) == {str(pair_id)}
    assert completion_payload["training_scene_seeds"] == [scene_seed]
    assert completion_payload["scenario_ids"] == [scenario_id]
    assert completion_payload["shield_pair_ids"] == [pair_id]


def test_pair_artifact_saves_rao_blackwell_moments_and_covariance(
    tmp_path: Path,
) -> None:
    """Pair artifacts must bind sparse statistics and all moment arrays."""
    calibration = parse_mean_calibration_metadata(
        _metadata(),
        bin_count=3,
    )
    response = np.asarray(
        [
            [1.0, 0.25, 0.0],
            [0.0, 0.75, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    raw_covariance = calibration.raw_covariance()

    manifest = write_mean_calibration_pair_artifact(
        directory=tmp_path / "pair",
        provenance={
            "scene_seed": DESIGNATED_TRAINING_SCENE_SEEDS[0],
            "scenario_id": "single_line_source_resolved",
            "shield_pair_id": 0,
            "dwell_time_s": STANDARD_ACQUISITION_LIVE_TIME_S,
            "sources": [],
        },
        calibration=calibration,
        native_spectrum=calibration.raw_mean(),
        native_spectrum_variance=np.diag(raw_covariance),
        response_operator_br=response,
        geometry={},
    )
    payload, arrays = load_mean_calibration_pair_artifact(manifest)

    np.testing.assert_allclose(
        arrays["marked_mean"],
        response @ arrays["raw_mean"],
    )
    np.testing.assert_allclose(
        arrays["marked_covariance"],
        response @ raw_covariance @ response.T,
    )
    assert payload["sufficient_statistics"]["batches"]
    assert (
        payload["sampling_covariance_scope"]
        == "fixed_quota_transport_mean_not_station_count_dispersion"
    )

    covariance_path = manifest.parent / "marked_covariance.npz"
    data = bytearray(covariance_path.read_bytes())
    data[-1] ^= 1
    covariance_path.write_bytes(data)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_mean_calibration_pair_artifact(manifest)


def test_pair_artifact_rejects_laundered_shield_contract(
    tmp_path: Path,
) -> None:
    """A model input cannot gain a new shield contract by editing one JSON."""
    calibration = parse_mean_calibration_metadata(_metadata(), bin_count=3)
    response = np.eye(3, dtype=np.float64)
    raw_covariance = calibration.raw_covariance()
    manifest = write_mean_calibration_pair_artifact(
        directory=tmp_path / "pair",
        provenance={
            "scene_seed": DESIGNATED_TRAINING_SCENE_SEEDS[0],
            "scenario_id": "single_line_source_resolved",
            "shield_pair_id": 0,
            "dwell_time_s": STANDARD_ACQUISITION_LIVE_TIME_S,
            "sources": [],
        },
        calibration=calibration,
        native_spectrum=calibration.raw_mean(),
        native_spectrum_variance=np.diag(raw_covariance),
        response_operator_br=response,
        geometry={},
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["shield_pose_contract_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest is invalid"):
        load_mean_calibration_pair_artifact(manifest)


def test_additive_fit_accepts_only_complete_training_scene_identity() -> None:
    """Physical fitting must bind all designated training-scene manifests."""
    rows: list[MeanCalibrationTrainingRow] = []
    coefficients = np.linspace(0.02, 0.08, 7)
    for scene_index, scene_seed in enumerate(
        DESIGNATED_TRAINING_SCENE_SEEDS
    ):
        scale = 1.0 + 0.01 * scene_index
        for feature_index in range(7):
            basis = np.zeros(7, dtype=np.float64)
            basis[feature_index] = scale
            rows.append(
                MeanCalibrationTrainingRow(
                    scene_id=str(scene_seed),
                    feature_basis=tuple(float(value) for value in basis),
                    scatter_fraction=float(
                        np.dot(coefficients, basis)
                    ),
                    sample_weight=1000.0,
                )
            )
    hashes = {
        str(seed): f"{index + 1:064x}"
        for index, seed in enumerate(DESIGNATED_TRAINING_SCENE_SEEDS)
    }

    response = fit_additive_scatter_training_rows(
        rows,
        scene_manifest_sha256_by_seed=hashes,
    )

    assert response.training_ready is True
    assert np.any(np.asarray(response.coefficients) > 0.0)
    with pytest.raises(ValueError, match="Complete authenticated"):
        fit_additive_scatter_training_rows(
            rows,
            scene_manifest_sha256_by_seed={
                str(DESIGNATED_TRAINING_SCENE_SEEDS[0]): "1" * 64
            },
        )


def test_additive_fit_accepts_declared_subset_without_all64_claim() -> None:
    """A complete small design may fit while declaring only measured pairs."""
    scene_seeds = (101, 202)
    rows: list[MeanCalibrationTrainingRow] = []
    coefficients = np.linspace(0.02, 0.08, 7)
    for scene_index, scene_seed in enumerate(scene_seeds):
        scale = 1.0 + 0.1 * scene_index
        for feature_index in range(7):
            basis = np.zeros(7, dtype=np.float64)
            basis[feature_index] = scale
            rows.append(
                MeanCalibrationTrainingRow(
                    scene_id=str(scene_seed),
                    feature_basis=tuple(float(value) for value in basis),
                    scatter_fraction=float(
                        np.dot(coefficients, basis)
                    ),
                    sample_weight=1000.0,
                )
            )

    response = fit_additive_scatter_training_rows(
        rows,
        scene_manifest_sha256_by_seed={
            "101": "1" * 64,
            "202": "2" * 64,
        },
        training_scene_seeds=scene_seeds,
        scenario_ids=("single_line_source_resolved",),
        shield_pair_ids=(7,),
    )

    assert response.training_ready is True
    assert response.training_manifest["training_scene_seeds"] == [101, 202]
    assert response.training_manifest["scenario_ids"] == [
        "single_line_source_resolved"
    ]
    assert response.training_manifest["pair_ids_by_scene"] == {
        "101": [7],
        "202": [7],
    }


def test_incomplete_calibration_cannot_create_runtime_ready_model(
    tmp_path: Path,
) -> None:
    """Mean-only or incomplete artifacts must fail closed at runtime freeze."""
    layout = MeanCalibrationLayout(tmp_path / "calibration")
    initialize_mean_calibration_layout(layout=layout, design=_design())

    with pytest.raises(FileNotFoundError):
        build_mean_calibration_completion_manifest(
            layout=layout,
            design=_design(),
        )

    rows: list[MeanCalibrationTrainingRow] = []
    for scene_seed in DESIGNATED_TRAINING_SCENE_SEEDS:
        for feature_index in range(7):
            basis = np.zeros(7, dtype=np.float64)
            basis[feature_index] = 1.0
            rows.append(
                MeanCalibrationTrainingRow(
                    scene_id=str(scene_seed),
                    feature_basis=tuple(float(value) for value in basis),
                    scatter_fraction=0.05,
                    sample_weight=100.0,
                )
            )
    additive = fit_additive_scatter_training_rows(
        rows,
        scene_manifest_sha256_by_seed={
            str(seed): f"{index + 1:064x}"
            for index, seed in enumerate(DESIGNATED_TRAINING_SCENE_SEEDS)
        },
    )
    mean_only_model = GeometryConditionedSpectralModel.standard_native(
        ("Co-60", "Cs-137", "Eu-154"),
        dead_time_tau_s=0.0,
        background_rate_cps=0.0,
        additive_scatter_response=additive,
    )

    assert mean_only_model.exact_physical_statistics_ready is True
    assert mean_only_model.runtime_ready is True
    with pytest.raises(RuntimeError, match="complete predeclared"):
        freeze_runtime_ready_model(
            output_path=tmp_path / "runtime_model.json",
            model=mean_only_model,
            additive_response=additive,
            layout=layout,
            design=_design(),
        )
    assert not (tmp_path / "runtime_model.json").exists()
