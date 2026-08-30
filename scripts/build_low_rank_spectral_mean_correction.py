"""Fit a training-only low-rank correction to native spectral means."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from spectrum.additive_scatter import scatter_basis_from_stored_geometry_numpy
from spectrum.mean_calibration_runner import (
    load_mean_calibration_pair_artifact,
)
from spectrum.full_spectrum_acceptance_runner import (
    canonical_detector_green_operator,
)
from spectrum.transport_spectral import (
    DESIGNATED_TRAINING_SCENE_SEEDS,
    VALIDATION_SCENARIO_IDS,
    GeometryConditionedSpectralModel,
    LowRankSpectralMeanCorrection,
    low_rank_spectral_mean_descriptor_numpy,
)


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TRAINING_ROOT = (
    _ROOT
    / "results"
    / "mean_calibration"
    / "randomized_geometry_family"
)
_DEFAULT_BASE_MODEL = (
    _ROOT
    / "configs"
    / "geant4"
    / "models"
    / "geometry_conditioned_full_spectrum_randomized_mean.json"
)
_DEFAULT_OUTPUT = (
    _ROOT
    / "configs"
    / "geant4"
    / "models"
    / "low_rank_spectral_mean_correction_randomized.json"
)
_DEFAULT_SELECTION = (
    _ROOT
    / "results"
    / "randomized_component_training"
    / "low_rank_mean_selection.json"
)
_TRAINING_SCENE_SEEDS = DESIGNATED_TRAINING_SCENE_SEEDS
_TRAINING_SCENARIOS = tuple(
    scenario
    for scenario in VALIDATION_SCENARIO_IDS
    if scenario != "background_only"
)
_RANK_GRID = (1, 2, 4, 6, 8)
_RIDGE_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)
_MAXIMUM_ABS_LOG_CORRECTION = 2.0


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit a regularized spectral-mean correction from fixed-quota "
            "training scenes without reading PF failures or holdouts."
        )
    )
    parser.add_argument("--training-root", type=Path, default=_DEFAULT_TRAINING_ROOT)
    parser.add_argument("--base-model", type=Path, default=_DEFAULT_BASE_MODEL)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--selection-output", type=Path, default=_DEFAULT_SELECTION)
    return parser


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one immutable input file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    """Return canonical JSON bytes for generated model artifacts."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write one canonical JSON artifact atomically."""
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(_canonical_bytes(payload))
    temporary.replace(destination)


def _load_base_model(path: Path) -> GeometryConditionedSpectralModel:
    """Load the authenticated uncorrected physical mean model."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = GeometryConditionedSpectralModel.from_manifest_payload(
        payload,
        detector_green_operator=canonical_detector_green_operator(),
    )
    model.require_runtime_ready()
    if model.low_rank_spectral_mean_correction is not None:
        raise ValueError("Mean-correction training requires an uncorrected base model.")
    return model


def _descriptor_order(
    model: GeometryConditionedSpectralModel,
) -> tuple[str, ...]:
    """Return stable names for every physical descriptor column."""
    line_names = tuple(
        "line_fraction::"
        f"{item['isotope']}::"
        f"{int(item['transport_line_index'])}"
        for item in model.line_identity
    )
    feature_names = tuple(
        f"transport_mean::{name}" for name in model.transport_feature_order
    )
    return (
        "log_total_source_counts",
        *line_names,
        "uncollided_fraction",
        *feature_names,
    )


def _declared_pair_manifests(root: Path) -> tuple[Path, ...]:
    """Return the complete predeclared training-only pair set."""
    design = json.loads((root / "training_design.json").read_text("utf-8"))
    if (
        design.get("schema_version") != 1
        or tuple(design.get("training_scene_seeds", ()))
        != _TRAINING_SCENE_SEEDS
        or design.get("holdout_consumed_by_training") is not False
    ):
        raise ValueError("Fixed-quota mean-training design is incompatible.")
    declared_pairs = tuple(int(value) for value in design["shield_pair_ids"])
    paths: list[Path] = []
    for seed in _TRAINING_SCENE_SEEDS:
        for scenario in _TRAINING_SCENARIOS:
            for pair_id in declared_pairs:
                manifest = (
                    root
                    / "pairs"
                    / f"scene_{seed}"
                    / scenario
                    / f"pair_{pair_id:02d}"
                    / "manifest.json"
                )
                if not manifest.is_file():
                    raise FileNotFoundError(manifest)
                paths.append(manifest)
    return tuple(paths)


def _training_row(
    manifest_path: Path,
    *,
    model: GeometryConditionedSpectralModel,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, str, int]:
    """Return one authenticated descriptor, residual, and spectral weight row."""
    payload, arrays = load_mean_calibration_pair_artifact(manifest_path)
    provenance = payload.get("provenance")
    geometry = payload.get("geometry")
    if not isinstance(provenance, Mapping) or not isinstance(geometry, Mapping):
        raise TypeError("Mean-calibration pair lacks authenticated geometry.")
    seed = int(provenance["scene_seed"])
    scenario = str(provenance["scenario_id"])
    pair_id = int(provenance["shield_pair_id"])
    if (
        seed not in _TRAINING_SCENE_SEEDS
        or scenario not in _TRAINING_SCENARIOS
        or provenance.get("holdout_artifacts_consumed") is not False
    ):
        raise ValueError("Mean-correction training row is contaminated.")
    unattenuated = np.asarray(
        geometry["unattenuated_source_line_rate_sl"],
        dtype=np.float64,
    )
    uncollided = np.asarray(
        geometry["uncollided_source_line_rate_sl"],
        dtype=np.float64,
    )
    features = np.asarray(
        geometry["transport_features_slf"],
        dtype=np.float64,
    )
    stored_scatter_basis = np.asarray(
        geometry["additive_scatter_basis_slf"],
        dtype=np.float64,
    )
    additive = model.additive_scatter_response
    if additive is None:
        raise RuntimeError("Base model lacks its authenticated scatter response.")
    scatter_basis = scatter_basis_from_stored_geometry_numpy(
        stored_basis=stored_scatter_basis,
        transport_features=features,
        line_identity=model.line_identity,
        target_semantics=additive.feature_basis_semantics,
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
    live_time = float(provenance["dwell_time_s"])
    source_mean, _ = model.pre_dead_time_components_numpy(
        total[np.newaxis, np.newaxis, ...],
        uncollided[np.newaxis, np.newaxis, ...],
        features[np.newaxis, np.newaxis, ...],
        np.asarray([live_time], dtype=np.float64),
    )
    predicted = np.asarray(source_mean[0, 0], dtype=np.float64)
    target = np.asarray(arrays["marked_mean"], dtype=np.float64)
    target_total = float(np.sum(target))
    predicted_total = float(np.sum(predicted))
    if target_total <= 0.0 or predicted_total <= 0.0:
        raise ValueError("Positive-source training scenarios must have nonzero means.")
    target_probability = target / target_total
    predicted_probability = predicted / predicted_total
    floor = 1.0e-12 / target.size
    residual = np.log(
        (target_probability + floor) / (predicted_probability + floor)
    )
    residual = np.clip(
        residual,
        -_MAXIMUM_ABS_LOG_CORRECTION,
        _MAXIMUM_ABS_LOG_CORRECTION,
    )
    spectral_weight = target + target_total * 1.0e-8 / target.size
    spectral_weight /= np.sum(spectral_weight)
    descriptor = low_rank_spectral_mean_descriptor_numpy(
        (total * live_time)[np.newaxis, np.newaxis, ...],
        (uncollided * live_time)[np.newaxis, np.newaxis, ...],
        features[np.newaxis, np.newaxis, ...],
    )[0, 0]
    return (
        seed,
        descriptor,
        residual,
        spectral_weight,
        scenario,
        pair_id,
    )


def _fit_model(
    descriptor_rd: np.ndarray,
    residual_rb: np.ndarray,
    *,
    rank: int,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one SVD basis and ridge map on a designated row subset."""
    center = np.mean(descriptor_rd, axis=0)
    scale = np.std(descriptor_rd, axis=0)
    scale = np.maximum(scale, 1.0e-8)
    standardized = (descriptor_rd - center) / scale
    design = np.concatenate(
        (np.ones((standardized.shape[0], 1)), standardized),
        axis=1,
    )
    _, _, right = np.linalg.svd(residual_rb, full_matrices=False)
    basis = right[:rank]
    scores = residual_rb @ basis.T
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    regression = np.linalg.solve(
        design.T @ design + float(ridge_lambda) * penalty,
        design.T @ scores,
    )
    return center, scale, regression, basis


def _predict_residual(
    descriptor_rd: np.ndarray,
    *,
    center_d: np.ndarray,
    scale_d: np.ndarray,
    regression_qk: np.ndarray,
    basis_kb: np.ndarray,
) -> np.ndarray:
    """Predict bounded log-mean residuals for one batched descriptor matrix."""
    standardized = (descriptor_rd - center_d) / scale_d
    design = np.concatenate(
        (np.ones((standardized.shape[0], 1)), standardized),
        axis=1,
    )
    result = design @ regression_qk @ basis_kb
    return np.clip(
        result,
        -_MAXIMUM_ABS_LOG_CORRECTION,
        _MAXIMUM_ABS_LOG_CORRECTION,
    )


def build_correction(
    *,
    training_root: Path,
    base_model: GeometryConditionedSpectralModel,
) -> tuple[LowRankSpectralMeanCorrection, dict[str, object]]:
    """Select by leave-one-scene-out validation and fit all training rows."""
    manifests = _declared_pair_manifests(training_root)
    rows = tuple(
        _training_row(path, model=base_model) for path in manifests
    )
    seeds = np.asarray([row[0] for row in rows], dtype=np.int64)
    descriptors = np.stack([row[1] for row in rows], axis=0)
    residuals = np.stack([row[2] for row in rows], axis=0)
    weights = np.stack([row[3] for row in rows], axis=0)
    scores: dict[str, float] = {}
    for rank in _RANK_GRID:
        for ridge_lambda in _RIDGE_GRID:
            fold_losses: list[float] = []
            for holdout_seed in _TRAINING_SCENE_SEEDS:
                training_mask = seeds != holdout_seed
                holdout_mask = ~training_mask
                center, scale, regression, basis = _fit_model(
                    descriptors[training_mask],
                    residuals[training_mask],
                    rank=rank,
                    ridge_lambda=ridge_lambda,
                )
                predicted = _predict_residual(
                    descriptors[holdout_mask],
                    center_d=center,
                    scale_d=scale,
                    regression_qk=regression,
                    basis_kb=basis,
                )
                squared = np.square(predicted - residuals[holdout_mask])
                fold_losses.extend(
                    np.sum(weights[holdout_mask] * squared, axis=1).tolist()
                )
            scores[f"rank={rank};ridge={ridge_lambda:.12g}"] = float(
                np.mean(fold_losses)
            )
    selected_rank, selected_ridge = min(
        (
            (rank, ridge_lambda)
            for rank in _RANK_GRID
            for ridge_lambda in _RIDGE_GRID
        ),
        key=lambda item: (
            scores[f"rank={item[0]};ridge={item[1]:.12g}"],
            item[0],
            -item[1],
        ),
    )
    center, scale, regression, basis = _fit_model(
        descriptors,
        residuals,
        rank=selected_rank,
        ridge_lambda=selected_ridge,
    )
    complete = json.loads(
        (training_root / "training_complete.json").read_text("utf-8")
    )
    artifact_hashes = {
        str(seed): str(complete["scene_manifest_sha256_by_seed"][str(seed)])
        for seed in _TRAINING_SCENE_SEEDS
    }
    pair_ids_by_scene = {
        str(seed): sorted(
            {row[5] for row in rows if row[0] == seed}
        )
        for seed in _TRAINING_SCENE_SEEDS
    }
    additive = base_model.additive_scatter_response
    if additive is None:
        raise RuntimeError("Base model lacks its authenticated scatter response.")
    training_manifest: dict[str, object] = {
        "schema_version": 2,
        "training_policy": (
            "randomized_geometry_family_loso_low_rank_log_mean_v3"
        ),
        "base_additive_response_contract_sha256": (
            additive.contract_hash_sha256
        ),
        "feature_basis_semantics": additive.feature_basis_semantics,
        "training_scene_seeds": list(_TRAINING_SCENE_SEEDS),
        "scenario_ids": list(_TRAINING_SCENARIOS),
        "pair_ids_by_scene": pair_ids_by_scene,
        "artifact_sha256_by_scene": artifact_hashes,
        "rank_grid": list(_RANK_GRID),
        "ridge_lambda_grid": list(_RIDGE_GRID),
        "selected_rank": int(selected_rank),
        "selected_ridge_lambda": float(selected_ridge),
        "selection_objective": (
            "leave_one_scene_out_target_probability_weighted_log_mse"
        ),
        "selected_validation_score": float(
            scores[
                f"rank={selected_rank};ridge={selected_ridge:.12g}"
            ]
        ),
        "selection_completed": True,
        "holdout_artifacts_consumed": False,
    }
    correction = LowRankSpectralMeanCorrection(
        descriptor_order=_descriptor_order(base_model),
        descriptor_center_d=center,
        descriptor_scale_d=scale,
        regression_qk=regression,
        basis_kb=basis,
        maximum_abs_log_correction=_MAXIMUM_ABS_LOG_CORRECTION,
        training_manifest=training_manifest,
    )
    if not correction.training_ready:
        raise RuntimeError("Low-rank correction training contract is incomplete.")
    selection: dict[str, object] = {
        **training_manifest,
        "candidate_scores": scores,
        "row_count": len(rows),
        "descriptor_count": descriptors.shape[1],
        "energy_bin_count": residuals.shape[1],
        "correction_contract_sha256": correction.contract_hash_sha256,
        "training_complete_file_sha256": _file_sha256(
            training_root / "training_complete.json"
        ),
    }
    return correction, selection


def main(argv: Sequence[str] | None = None) -> int:
    """Build one authenticated training-only spectral-mean correction."""
    arguments = _parser().parse_args(argv)
    base_model = _load_base_model(arguments.base_model.resolve())
    correction, selection = build_correction(
        training_root=arguments.training_root.resolve(),
        base_model=base_model,
    )
    _write_json(arguments.output, correction.to_payload())
    _write_json(arguments.selection_output, selection)
    print(arguments.output.resolve())
    print(correction.contract_hash_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
