"""Fit a randomized-family physical-component discrepancy candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import stats

from measurement.geometry_family import (
    GEOMETRY_FAMILY_APPLICABILITY_SHA256,
    validate_geometry_family_descriptor,
)
from spectrum.additive_scatter import scatter_basis_from_stored_geometry_numpy
from spectrum.full_spectrum_acceptance_runner import (
    ACCEPTANCE_ISOTOPES,
    canonical_json_bytes,
    file_sha256,
    line_identity_contract_sha256,
    load_acceptance_pair,
)
from spectrum.mean_calibration_runner import (
    _training_rows_from_pair,
    fit_additive_scatter_training_rows,
    load_mean_calibration_pair_artifact,
)
from spectrum.transport_spectral import (
    ACCEPTANCE_METRIC_CONTRACT,
    DESIGNATED_TRAINING_SCENE_SEEDS,
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    MARK_CONCENTRATION_GRID,
    VALIDATION_SCENARIO_IDS,
    GeometryConditionedSpectralModel,
    LowRankSpectralMeanCorrection,
    PhysicalComponentDiscrepancy,
)
try:
    from scripts.build_low_rank_spectral_mean_correction import (
        _MAXIMUM_ABS_LOG_CORRECTION,
        _declared_pair_manifests,
        _descriptor_order,
        _fit_model,
        _training_row,
    )
except ModuleNotFoundError:
    from build_low_rank_spectral_mean_correction import (
        _MAXIMUM_ABS_LOG_CORRECTION,
        _declared_pair_manifests,
        _descriptor_order,
        _fit_model,
        _training_row,
    )


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TRAINING_ROOT = (
    _ROOT / "results" / "randomized_component_training"
)
_DEFAULT_BASE_MODEL = (
    _ROOT
    / "configs"
    / "geant4"
    / "models"
    / "geometry_conditioned_full_spectrum_randomized_mean.json"
)
_DEFAULT_MEAN_CORRECTION = (
    _ROOT
    / "configs"
    / "geant4"
    / "models"
    / "low_rank_spectral_mean_correction_randomized.json"
)
_DEFAULT_MEAN_TRAINING_ROOT = (
    _ROOT / "results" / "mean_calibration" / "randomized_geometry_family"
)
_DEFAULT_OUTPUT_MODEL = (
    _ROOT
    / "configs"
    / "geant4"
    / "models"
    / "geometry_conditioned_full_spectrum_ral_eu154_component.json"
)
_DEFAULT_SELECTION = (
    _DEFAULT_TRAINING_ROOT / "component_discrepancy_selection.json"
)
COMPONENT_TRAINING_PAIR_IDS = (
    0,
    1,
    8,
    15,
    22,
    23,
    29,
    30,
    36,
    37,
    43,
    44,
    50,
    51,
    57,
    58,
)
COMPONENT_CONCENTRATION_GRID = tuple(
    float(value)
    for value in (
        *MARK_CONCENTRATION_GRID,
        300_000.0,
        1_000_000.0,
        3_000_000.0,
        10_000_000.0,
        100_000_000.0,
        10_000_000_000.0,
    )
)
COMPONENT_LOG_RATIO_REGULARIZATION = 0.01
COMPONENT_MARK_TAIL_PROBABILITY_THRESHOLD = 0.01
COMPONENT_MARK_COVERAGE_THRESHOLD = float(
    ACCEPTANCE_METRIC_CONTRACT[
        "pairwise_mark_tail_ge_0p01_fraction"
    ][1]
)


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit direct/scatter latent concentrations from predeclared "
            "randomized training scenes without reading any holdout or PF run."
        )
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=_DEFAULT_TRAINING_ROOT,
    )
    parser.add_argument("--base-model", type=Path, default=_DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--mean-correction",
        type=Path,
        default=_DEFAULT_MEAN_CORRECTION,
    )
    parser.add_argument(
        "--mean-training-root",
        type=Path,
        default=_DEFAULT_MEAN_TRAINING_ROOT,
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=_DEFAULT_OUTPUT_MODEL,
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=_DEFAULT_SELECTION,
    )
    return parser


def _load_base_model(path: Path) -> GeometryConditionedSpectralModel:
    """Load a physical mean whose additive response is training-authenticated."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = GeometryConditionedSpectralModel.from_manifest_payload(payload)
    additive = model.additive_scatter_response
    if additive is None or not additive.training_ready:
        raise ValueError(
            "Physical-component fitting requires a training-ready additive "
            "transport response."
        )
    training = additive.training_manifest
    if tuple(training.get("training_scene_seeds", ())) != tuple(
        DESIGNATED_TRAINING_SCENE_SEEDS
    ):
        raise ValueError(
            "Additive transport mean was not fitted on every designated "
            "randomized training scene."
        )
    if tuple(training.get("scenario_ids", ())) != tuple(
        VALIDATION_SCENARIO_IDS
    ):
        raise ValueError(
            "Additive transport mean does not cover every training scenario."
        )
    return model


def _attach_mean_correction(
    model: GeometryConditionedSpectralModel,
    path: Path,
) -> GeometryConditionedSpectralModel:
    """Attach one authenticated training-only spectral mean correction."""
    correction = LowRankSpectralMeanCorrection.from_payload(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if not correction.training_ready:
        raise ValueError("Spectral mean correction is not training-authenticated.")
    corrected = GeometryConditionedSpectralModel.standard_native(
        ACCEPTANCE_ISOTOPES,
        dead_time_tau_s=model.dead_time_tau_s,
        background_rate_cps=model.background_rate_cps,
        additive_scatter_response=model.additive_scatter_response,
        low_rank_spectral_mean_correction=correction,
    )
    corrected.require_runtime_ready()
    return corrected


def _training_paths(root: Path) -> dict[tuple[int, str], tuple[Path, ...]]:
    """Return the exact predeclared randomized training pair paths."""
    result: dict[tuple[int, str], tuple[Path, ...]] = {}
    for seed in DESIGNATED_TRAINING_SCENE_SEEDS:
        for scenario in VALIDATION_SCENARIO_IDS:
            paths = tuple(
                root
                / "training"
                / f"scene_{seed}"
                / scenario
                / f"pair_{pair_id:02d}.json"
                for pair_id in COMPONENT_TRAINING_PAIR_IDS
            )
            missing = [path for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Randomized component training is incomplete; first "
                    f"missing artifact: {missing[0]}."
                )
            result[(int(seed), str(scenario))] = paths
    return result


def _group_arrays(
    records: Sequence[object],
    *,
    model: GeometryConditionedSpectralModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack one source-cardinality-homogeneous training group."""
    additive = model.additive_scatter_response
    if additive is None:
        raise RuntimeError("Base model lacks additive transport response.")
    observed = np.stack(
        [record.observed_spectrum_counts for record in records],
        axis=0,
    ).astype(np.float64)
    unattenuated = np.concatenate(
        [record.unattenuated_vsl for record in records],
        axis=0,
    )
    uncollided = np.concatenate(
        [record.uncollided_vsl for record in records],
        axis=0,
    )
    features = np.concatenate(
        [record.features_vslf for record in records],
        axis=0,
    )
    stored_scatter_basis = np.concatenate(
        [record.scatter_basis_vslf for record in records],
        axis=0,
    )
    scatter_basis = scatter_basis_from_stored_geometry_numpy(
        stored_basis=stored_scatter_basis,
        transport_features=features,
        line_identity=model.line_identity,
        target_semantics=additive.feature_basis_semantics,
        detector_radius_m=getattr(additive, "detector_radius_m", None),
        fe_scatter_distance_m=getattr(
            additive,
            "fe_scatter_distance_m",
            None,
        ),
        pb_scatter_distance_m=getattr(
            additive,
            "pb_scatter_distance_m",
            None,
        ),
    )
    total = additive.total_kernel_numpy(
        unattenuated,
        uncollided,
        scatter_basis,
    )
    uncollided = additive.corrected_uncollided_kernel_numpy(
        uncollided,
        scatter_basis,
    )
    return observed, total, uncollided, features


def _candidate_score(
    *,
    arrays: Sequence[
        tuple[
            GeometryConditionedSpectralModel,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ]
    ],
    count_direct: float,
    count_scatter: float,
    mark_direct: float,
    mark_scatter: float,
) -> float:
    """Return mean training log density with a fixed smoothness penalty."""
    component = PhysicalComponentDiscrepancy(
        count_uncollided_concentration=count_direct,
        count_scatter_concentration=count_scatter,
        mark_uncollided_concentration=mark_direct,
        mark_scatter_concentration=mark_scatter,
    )
    score = 0.0
    observation_count = 0
    for fold_model, (observed, total, uncollided, features) in arrays:
        candidate = GeometryConditionedSpectralModel.standard_native(
            ACCEPTANCE_ISOTOPES,
            dead_time_tau_s=fold_model.dead_time_tau_s,
            background_rate_cps=fold_model.background_rate_cps,
            physical_component_discrepancy=component,
            additive_scatter_response=fold_model.additive_scatter_response,
            low_rank_spectral_mean_correction=(
                fold_model.low_rank_spectral_mean_correction
            ),
        )
        likelihood = candidate.log_likelihood_numpy(
            observed,
            total[np.newaxis, ...],
            uncollided[np.newaxis, ...],
            features[np.newaxis, ...],
            np.full(observed.shape[0], 30.0, dtype=np.float64),
        )
        score += float(likelihood[0])
        observation_count += int(observed.shape[0])
    if observation_count <= 0:
        raise RuntimeError("Component fitting has no training observations.")
    regularizer = COMPONENT_LOG_RATIO_REGULARIZATION * (
        math.log(count_direct / count_scatter) ** 2
        + math.log(mark_direct / mark_scatter) ** 2
    )
    return score / float(observation_count) - regularizer


def _candidate_pairwise_mark_coverage(
    *,
    arrays: Sequence[
        tuple[
            GeometryConditionedSpectralModel,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ]
    ],
    mark_direct: float,
    mark_scatter: float,
) -> float:
    """Return leave-one-geometry-out pairwise conditional-mark coverage.

    The calculation uses the same Dirichlet-multinomial variance as the
    runtime innovation diagnostic.  Every row is predicted by a mean model
    fitted without that row's geometry seed.  No holdout observation enters
    this training-only calibration constraint.
    """
    covered: list[np.ndarray] = []
    exact_count = 1.0e15
    component = PhysicalComponentDiscrepancy(
        count_uncollided_concentration=exact_count,
        count_scatter_concentration=exact_count,
        mark_uncollided_concentration=mark_direct,
        mark_scatter_concentration=mark_scatter,
    )
    for fold_model, (observed, total, uncollided, features) in arrays:
        candidate = GeometryConditionedSpectralModel.standard_native(
            ACCEPTANCE_ISOTOPES,
            dead_time_tau_s=fold_model.dead_time_tau_s,
            background_rate_cps=fold_model.background_rate_cps,
            physical_component_discrepancy=component,
            additive_scatter_response=fold_model.additive_scatter_response,
            low_rank_spectral_mean_correction=(
                fold_model.low_rank_spectral_mean_correction
            ),
        )
        live_times = np.full(observed.shape[0], 30.0, dtype=np.float64)
        source_mean, background_mean = candidate.pre_dead_time_components_numpy(
            total,
            uncollided,
            features,
            live_times,
        )
        pre_mean = source_mean + background_mean
        pre_total = np.sum(pre_mean, axis=-1)
        probabilities = np.divide(
            pre_mean,
            pre_total[:, np.newaxis],
            out=np.zeros_like(pre_mean),
            where=pre_total[:, np.newaxis] > 0.0,
        )
        observed_total = np.sum(observed, axis=-1)
        expected = observed_total[:, np.newaxis] * probabilities
        pearson = np.sum(
            np.square(observed - expected) / np.maximum(expected, 1.0),
            axis=-1,
        )
        degrees = np.sum(expected >= 1.0, axis=-1) - 1
        source_fraction = np.divide(
            np.sum(source_mean, axis=-1),
            pre_total,
            out=np.zeros_like(pre_total),
            where=pre_total > 0.0,
        )
        base_concentration = candidate._base_mark_concentration_numpy(
            total,
            uncollided,
        )
        concentration = base_concentration / np.maximum(
            np.square(source_fraction),
            1.0e-12,
        )
        dispersion = np.where(
            source_fraction > 0.0,
            (observed_total + concentration) / (1.0 + concentration),
            1.0,
        )
        tail = stats.chi2.sf(
            pearson / dispersion,
            np.maximum(degrees, 1),
        )
        covered.append(
            (degrees > 0)
            & (tail >= COMPONENT_MARK_TAIL_PROBABILITY_THRESHOLD)
        )
    if not covered:
        raise RuntimeError("Component fitting has no mark-calibration rows.")
    return float(np.mean(np.concatenate(covered, axis=0)))


def _best_calibrated_pair(
    *,
    scores: Mapping[str, float],
    coverages: Mapping[str, float],
) -> tuple[float, float]:
    """Return the best predictive pair satisfying predeclared coverage."""
    eligible = tuple(
        key
        for key, coverage in coverages.items()
        if coverage + 1.0e-12 >= COMPONENT_MARK_COVERAGE_THRESHOLD
    )
    if not eligible:
        raise RuntimeError(
            "No physical-component mark concentration satisfies the "
            "predeclared cross-fitted pairwise coverage contract."
        )
    best = max(scores[key] for key in eligible)
    tied = []
    for key in eligible:
        if scores[key] >= best - 1.0e-12:
            fields = dict(item.split("=") for item in key.split(";"))
            tied.append((float(fields["direct"]), float(fields["scatter"])))
    return max(tied, key=lambda values: (values[0], values[1]))


def _select_component_pair(
    *,
    arrays: Sequence[
        tuple[
            GeometryConditionedSpectralModel,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ]
    ],
    component: str,
) -> tuple[tuple[float, float], dict[str, float], dict[str, float]]:
    """Select one direct/scatter pair while the other law is effectively exact."""
    scores: dict[str, float] = {}
    coverages: dict[str, float] = {}
    exact = 1.0e15
    for direct in COMPONENT_CONCENTRATION_GRID:
        for scatter in COMPONENT_CONCENTRATION_GRID:
            if direct < scatter:
                continue
            kwargs = {
                "count_direct": exact,
                "count_scatter": exact,
                "mark_direct": exact,
                "mark_scatter": exact,
            }
            kwargs[f"{component}_direct"] = direct
            kwargs[f"{component}_scatter"] = scatter
            key = f"direct={direct:.12g};scatter={scatter:.12g}"
            scores[key] = _candidate_score(
                arrays=arrays,
                **kwargs,
            )
            if component == "mark":
                coverages[key] = _candidate_pairwise_mark_coverage(
                    arrays=arrays,
                    mark_direct=direct,
                    mark_scatter=scatter,
                )
    if component == "mark":
        return _best_calibrated_pair(
            scores=scores,
            coverages=coverages,
        ), scores, coverages
    best = max(scores.values())
    tied: list[tuple[float, float]] = []
    for key, score in scores.items():
        if score >= best - 1.0e-12:
            fields = dict(item.split("=") for item in key.split(";"))
            tied.append((float(fields["direct"]), float(fields["scatter"])))
    selected = max(tied, key=lambda values: (values[0], values[1]))
    return selected, scores, coverages


def build_candidate(
    *,
    training_root: Path,
    base_model: GeometryConditionedSpectralModel,
    cross_fitted_mean_models: Mapping[int, GeometryConditionedSpectralModel],
) -> tuple[dict[str, object], GeometryConditionedSpectralModel]:
    """Build one randomized-family physical-component runtime candidate."""
    paths_by_group = _training_paths(training_root)
    line_hash = line_identity_contract_sha256(base_model)
    arrays: list[
        tuple[
            GeometryConditionedSpectralModel,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ]
    ] = []
    artifact_hashes: dict[str, dict[str, dict[str, str]]] = {}
    geometry_hashes: dict[str, str] = {}
    for (seed, scenario), paths in sorted(paths_by_group.items()):
        records = tuple(
            load_acceptance_pair(
                path,
                expected_line_identity_sha256=line_hash,
            )
            for path in paths
        )
        if any(
            record.scene_seed != seed
            or record.split != "training"
            or record.scenario_id != scenario
            or record.geometry_family is None
            for record in records
        ):
            raise RuntimeError("Randomized training pair identity is invalid.")
        for record in records:
            validate_geometry_family_descriptor(
                record.geometry_family,
                require_in_domain=True,
            )
            descriptor_hash = str(
                record.geometry_family["component_geometry_sha256"]
            )
            previous = geometry_hashes.setdefault(str(seed), descriptor_hash)
            if previous != descriptor_hash:
                raise RuntimeError(
                    "One training seed resolved to multiple geometries."
                )
        fold_model = cross_fitted_mean_models.get(int(seed))
        if fold_model is None:
            raise RuntimeError(
                f"No cross-fitted spectral mean exists for scene {seed}."
            )
        arrays.append((fold_model, _group_arrays(records, model=fold_model)))
        artifact_hashes.setdefault(str(seed), {})[scenario] = {
            str(record.shield_pair_id): file_sha256(record.path)
            for record in records
        }
    (
        (count_direct, count_scatter),
        count_scores,
        _,
    ) = _select_component_pair(
        arrays=arrays,
        component="count",
    )
    (
        (mark_direct, mark_scatter),
        mark_scores,
        mark_coverages,
    ) = _select_component_pair(
        arrays=arrays,
        component="mark",
    )
    component = PhysicalComponentDiscrepancy(
        count_uncollided_concentration=count_direct,
        count_scatter_concentration=count_scatter,
        mark_uncollided_concentration=mark_direct,
        mark_scatter_concentration=mark_scatter,
    )
    selected = {
        "count_uncollided_concentration": count_direct,
        "count_scatter_concentration": count_scatter,
        "mark_uncollided_concentration": mark_direct,
        "mark_scatter_concentration": mark_scatter,
        "count_scope": component.count_scope,
    }
    additive = base_model.additive_scatter_response
    mean_correction = base_model.low_rank_spectral_mean_correction
    if additive is None or mean_correction is None:
        raise RuntimeError("Component fitting requires the complete mean model.")
    manifest: dict[str, object] = {
        "schema_version": 5,
        "training_policy": (
            "randomized_geometry_family_cross_fitted_component_v4"
        ),
        "base_additive_response_contract_sha256": (
            additive.contract_hash_sha256
        ),
        "low_rank_mean_correction_contract_sha256": (
            mean_correction.contract_hash_sha256
        ),
        "feature_basis_semantics": additive.feature_basis_semantics,
        "acceptance_contract_sha256": (
            FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        ),
        "geometry_family_applicability_sha256": (
            GEOMETRY_FAMILY_APPLICABILITY_SHA256
        ),
        "training_scene_seeds": list(DESIGNATED_TRAINING_SCENE_SEEDS),
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "artifact_sha256_by_scene_and_scenario": artifact_hashes,
        "component_family": "uncollided_scatter_component_latents_v1",
        "selected_concentrations": selected,
        "selection_objective": (
            "leave_one_geometry_out_log_predictive_density_regularized_"
            "subject_to_predeclared_pairwise_mark_coverage"
        ),
        "mark_tail_probability_threshold": (
            COMPONENT_MARK_TAIL_PROBABILITY_THRESHOLD
        ),
        "mark_cross_fitted_coverage_threshold": (
            COMPONENT_MARK_COVERAGE_THRESHOLD
        ),
        "selected_mark_cross_fitted_coverage": float(
            mark_coverages[
                f"direct={mark_direct:.12g};scatter={mark_scatter:.12g}"
            ]
        ),
        "selection_completed": True,
        "holdout_artifacts_consumed": False,
    }
    model = GeometryConditionedSpectralModel.standard_native(
        ACCEPTANCE_ISOTOPES,
        dead_time_tau_s=base_model.dead_time_tau_s,
        background_rate_cps=base_model.background_rate_cps,
        physical_component_discrepancy=component,
        discrepancy_training_manifest=manifest,
        additive_scatter_response=base_model.additive_scatter_response,
        low_rank_spectral_mean_correction=(
            base_model.low_rank_spectral_mean_correction
        ),
    )
    model.require_runtime_ready()
    selection = {
        "schema_version": 2,
        "training_policy": manifest["training_policy"],
        "training_scene_seeds": list(DESIGNATED_TRAINING_SCENE_SEEDS),
        "scenario_ids": list(VALIDATION_SCENARIO_IDS),
        "shield_pair_ids": list(COMPONENT_TRAINING_PAIR_IDS),
        "geometry_sha256_by_scene": geometry_hashes,
        "component_concentration_grid": list(COMPONENT_CONCENTRATION_GRID),
        "log_ratio_regularization": COMPONENT_LOG_RATIO_REGULARIZATION,
        "count_candidate_scores": count_scores,
        "mark_candidate_scores": mark_scores,
        "mark_candidate_cross_fitted_coverages": mark_coverages,
        "mark_tail_probability_threshold": (
            COMPONENT_MARK_TAIL_PROBABILITY_THRESHOLD
        ),
        "mark_cross_fitted_coverage_threshold": (
            COMPONENT_MARK_COVERAGE_THRESHOLD
        ),
        "selected_concentrations": selected,
        "model_contract_hash_sha256": model.contract_hash_sha256,
        "holdout_artifacts_consumed": False,
    }
    return selection, model


def _cross_fitted_mean_models(
    *,
    uncorrected_model: GeometryConditionedSpectralModel,
    final_correction: LowRankSpectralMeanCorrection,
    mean_training_root: Path,
) -> Mapping[int, GeometryConditionedSpectralModel]:
    """Return additive and spectral means excluding the scene they score."""
    full_additive = uncorrected_model.additive_scatter_response
    if full_additive is None:
        raise RuntimeError("Uncorrected model lacks its additive response.")
    feature_basis_semantics = full_additive.feature_basis_semantics
    manifests = _declared_pair_manifests(mean_training_root)
    payloads = tuple(
        load_mean_calibration_pair_artifact(path)[0] for path in manifests
    )
    completion = json.loads(
        (mean_training_root / "training_complete.json").read_text("utf-8")
    )
    scene_hashes = completion["scene_manifest_sha256_by_seed"]
    rank = int(final_correction.basis_kb.shape[0])
    ridge = float(final_correction.training_manifest["selected_ridge_lambda"])
    models: dict[int, GeometryConditionedSpectralModel] = {}
    for holdout_seed in DESIGNATED_TRAINING_SCENE_SEEDS:
        fit_seeds = tuple(
            seed
            for seed in DESIGNATED_TRAINING_SCENE_SEEDS
            if seed != holdout_seed
        )
        additive_rows = tuple(
            row
            for payload in payloads
            if int(payload["provenance"]["scene_seed"]) in fit_seeds
            for row in _training_rows_from_pair(
                payload,
                model=uncorrected_model,
                feature_basis_semantics=feature_basis_semantics,
            )
        )
        fold_additive = fit_additive_scatter_training_rows(
            additive_rows,
            scene_manifest_sha256_by_seed={
                str(seed): str(scene_hashes[str(seed)]) for seed in fit_seeds
            },
            training_scene_seeds=fit_seeds,
            scenario_ids=VALIDATION_SCENARIO_IDS,
            shield_pair_ids=COMPONENT_TRAINING_PAIR_IDS,
            feature_basis_semantics=feature_basis_semantics,
        )
        fold_uncorrected = GeometryConditionedSpectralModel.standard_native(
            ACCEPTANCE_ISOTOPES,
            dead_time_tau_s=uncorrected_model.dead_time_tau_s,
            background_rate_cps=uncorrected_model.background_rate_cps,
            additive_scatter_response=fold_additive,
        )
        rows = tuple(
            _training_row(path, model=fold_uncorrected)
            for path in manifests
        )
        seeds = np.asarray([row[0] for row in rows], dtype=np.int64)
        training_mask = seeds != int(holdout_seed)
        descriptors = np.stack([row[1] for row in rows], axis=0)
        residuals = np.stack([row[2] for row in rows], axis=0)
        center, scale, regression, basis = _fit_model(
            descriptors[training_mask],
            residuals[training_mask],
            rank=rank,
            ridge_lambda=ridge,
        )
        correction = LowRankSpectralMeanCorrection(
            descriptor_order=_descriptor_order(uncorrected_model),
            descriptor_center_d=center,
            descriptor_scale_d=scale,
            regression_qk=regression,
            basis_kb=basis,
            maximum_abs_log_correction=_MAXIMUM_ABS_LOG_CORRECTION,
            training_manifest={},
        )
        models[int(holdout_seed)] = GeometryConditionedSpectralModel.standard_native(
            ACCEPTANCE_ISOTOPES,
            dead_time_tau_s=uncorrected_model.dead_time_tau_s,
            background_rate_cps=uncorrected_model.background_rate_cps,
            additive_scatter_response=fold_additive,
            low_rank_spectral_mean_correction=correction,
        )
    return models


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write one generated canonical JSON artifact atomically."""
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(destination)


def main(argv: Sequence[str] | None = None) -> int:
    """Fit and write the component candidate without reading holdout data."""
    arguments = _parser().parse_args(argv)
    uncorrected_model = _load_base_model(arguments.base_model.resolve())
    base_model = _attach_mean_correction(
        uncorrected_model,
        arguments.mean_correction.resolve(),
    )
    final_correction = base_model.low_rank_spectral_mean_correction
    if final_correction is None:
        raise RuntimeError("Corrected base model lacks its spectral correction.")
    selection, model = build_candidate(
        training_root=arguments.training_root.resolve(),
        base_model=base_model,
        cross_fitted_mean_models=_cross_fitted_mean_models(
            uncorrected_model=uncorrected_model,
            final_correction=final_correction,
            mean_training_root=arguments.mean_training_root.resolve(),
        ),
    )
    _write_json(arguments.selection_output, selection)
    _write_json(arguments.output_model, model.manifest_payload())
    print(arguments.output_model.resolve())
    print(model.contract_hash_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
