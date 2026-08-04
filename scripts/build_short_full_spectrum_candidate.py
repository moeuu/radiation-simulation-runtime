"""Fit a training-only short diagnostic full-spectrum discrepancy model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from spectrum.additive_scatter import scatter_basis_from_stored_geometry_numpy
from spectrum.full_spectrum_acceptance_runner import (
    ACCEPTANCE_ISOTOPES,
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    line_identity_contract_sha256,
    load_acceptance_pair,
)
from spectrum.mean_calibration_runner import (
    load_mean_calibration_pair_artifact,
)
from spectrum.transport_spectral import (
    FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256,
    MARK_CONCENTRATION_GRID,
    RATE_SCALE_HALF_WIDTH_GRID,
    GeometryConditionedSpectralModel,
    LowRankSpectralMeanCorrection,
)


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TRAINING_ROOT = _ROOT / "results" / "full_spectrum_all64_acceptance"
_DEFAULT_BASE_MODEL = (
    _ROOT
    / "configs"
    / "geant4"
    / "models"
    / "geometry_conditioned_full_spectrum_exact_v1.json"
)
_DEFAULT_OUTPUT_MODEL = (
    _ROOT
    / "configs"
    / "geant4"
    / "models"
    / "geometry_conditioned_full_spectrum_ral_eu154_training_v4.json"
)
_DEFAULT_SELECTION = (
    _ROOT
    / "results"
    / "full_spectrum_short_diagnostic"
    / "training_selection_view_conditioned_v4.json"
)
_DEFAULT_MEAN_CORRECTION = (
    _ROOT
    / "configs"
    / "geant4"
    / "models"
    / "low_rank_spectral_mean_correction_v1.json"
)
_DEFAULT_MEAN_TRAINING_ROOT = (
    _ROOT
    / "results"
    / "mean_calibration"
    / "standard_native_exact_analog65536_20260729"
)
_TRAINING_SCENE_SEEDS = (2026072701,)
_TRAINING_SCENARIOS = (
    "single_line_source_resolved",
    "dominant_plus_absent_isotope",
)
_MEAN_TRAINING_SCENARIOS = (
    "dominant_plus_absent_isotope",
    "multi_isotope_superposition",
    "continuous_surface_perturbation_ranking",
)
_MARK_CALIBRATION_LOWER_QUANTILE = 0.05


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Select a global discrepancy from declared training artifacts "
            "without reading any holdout or failed PF replay."
        )
    )
    parser.add_argument("--training-root", type=Path, default=_DEFAULT_TRAINING_ROOT)
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
        "--disable-mean-correction",
        action="store_true",
        help=(
            "Build the zero-shift physical mean after an independent holdout "
            "has rejected a deterministic correction."
        ),
    )
    parser.add_argument("--output-model", type=Path, default=_DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--selection-output", type=Path, default=_DEFAULT_SELECTION)
    return parser


def _load_base_model(path: Path) -> GeometryConditionedSpectralModel:
    """Load the authenticated physical-mean model used by standard runtime."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = GeometryConditionedSpectralModel.from_manifest_payload(payload)
    model.require_runtime_ready()
    return model


def _load_mean_correction(path: Path) -> LowRankSpectralMeanCorrection:
    """Load and authenticate the independent fixed-quota mean correction."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    correction = LowRankSpectralMeanCorrection.from_payload(payload)
    if not correction.training_ready:
        raise ValueError("Configured spectral-mean correction is not training-ready.")
    return correction


def _declared_training_paths(root: Path) -> dict[tuple[int, str], tuple[Path, ...]]:
    """Return every available pair from the declared training groups."""
    groups: dict[tuple[int, str], tuple[Path, ...]] = {}
    for seed in _TRAINING_SCENE_SEEDS:
        for scenario in _TRAINING_SCENARIOS:
            directory = (
                root
                / "training"
                / f"scene_{seed}"
                / scenario
            )
            paths = tuple(sorted(directory.glob("pair_*.json")))
            if len(paths) < 16:
                raise RuntimeError(
                    f"Short discrepancy training group is incomplete: {directory}."
                )
            groups[(seed, scenario)] = paths
    return groups


def _group_arrays(
    records: Sequence[object],
    *,
    model: GeometryConditionedSpectralModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack one arbitrary-size training group into likelihood tensors."""
    additive = model.additive_scatter_response
    if additive is None:
        raise RuntimeError("Base model lacks the fitted additive response.")
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
    )
    total = additive.total_kernel_numpy(
        unattenuated,
        uncollided,
        scatter_basis,
    )
    return observed, total, uncollided, features


def _predicted_training_source_mean(
    model: GeometryConditionedSpectralModel,
    payload: Mapping[str, object],
) -> np.ndarray:
    """Return the source-only marked mean for one fixed-quota artifact."""
    geometry = payload.get("geometry")
    provenance = payload.get("provenance")
    if not isinstance(geometry, Mapping) or not isinstance(provenance, Mapping):
        raise TypeError("Mean-training pair lacks geometry or provenance.")
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
        raise RuntimeError("Mean-training model lacks additive scatter response.")
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
    live_time = float(provenance["dwell_time_s"])
    source, _ = model.pre_dead_time_components_numpy(
        total[np.newaxis, np.newaxis, ...],
        uncollided[np.newaxis, np.newaxis, ...],
        features[np.newaxis, np.newaxis, ...],
        np.asarray([live_time], dtype=np.float64),
    )
    return np.asarray(source[0, 0], dtype=np.float64)


def _select_mark_concentration_from_mean_training(
    root: Path,
    *,
    model: GeometryConditionedSpectralModel,
) -> tuple[float, float, dict[str, object]]:
    """Select one cardinality-safe dispersion from training-only residuals.

    A higher aggregate-source concentration would assume that independent
    component model errors cancel as source cardinality grows.  The runtime
    model does not explicitly represent those component-level latent errors,
    so both single- and multi-source states use the most conservative declared
    training-scenario lower quantile.
    """
    design_path = root / "training_design.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    seeds = tuple(int(value) for value in design["training_scene_seeds"])
    pair_ids = tuple(int(value) for value in design["shield_pair_ids"])
    scenarios = tuple(str(value) for value in design["scenario_ids"])
    if (
        seeds != (2026072701, 2026072702)
        or design.get("holdout_consumed_by_training") is not False
        or any(value not in scenarios for value in _MEAN_TRAINING_SCENARIOS)
        or len(pair_ids) < 16
    ):
        raise RuntimeError("Mean-discrepancy training design is invalid.")
    concentrations_by_scenario: dict[str, list[float]] = {
        scenario: [] for scenario in _MEAN_TRAINING_SCENARIOS
    }
    artifact_hashes: dict[str, dict[str, dict[str, str]]] = {}
    for seed in seeds:
        for scenario in _MEAN_TRAINING_SCENARIOS:
            for pair_id in pair_ids:
                manifest_path = (
                    root
                    / "pairs"
                    / f"scene_{seed}"
                    / scenario
                    / f"pair_{pair_id:02d}"
                    / "manifest.json"
                )
                payload, arrays = load_mean_calibration_pair_artifact(
                    manifest_path
                )
                target = np.asarray(arrays["marked_mean"], dtype=np.float64)
                predicted = _predicted_training_source_mean(model, payload)
                target_total = float(np.sum(target))
                predicted_total = float(np.sum(predicted))
                if target_total <= 0.0 or predicted_total <= 0.0:
                    raise RuntimeError("Mean mark calibration requires signal.")
                target_probability = target / target_total
                predicted_probability = predicted / predicted_total
                squared_error = float(
                    np.sum(
                        np.square(
                            target_probability - predicted_probability
                        )
                    )
                )
                numerator = 1.0 - float(
                    np.sum(np.square(predicted_probability))
                )
                concentrations_by_scenario[scenario].append(
                    max(numerator / max(squared_error, 1.0e-30) - 1.0, 0.0)
                )
                artifact_hashes.setdefault(str(seed), {}).setdefault(
                    scenario,
                    {},
                )[str(pair_id)] = file_sha256(manifest_path)
    lower_by_scenario = {
        scenario: float(
            np.quantile(
                np.asarray(values, dtype=np.float64),
                _MARK_CALIBRATION_LOWER_QUANTILE,
                method="linear",
            )
        )
        for scenario, values in concentrations_by_scenario.items()
    }

    def _grid_floor(value: float) -> float:
        """Return the largest predeclared concentration not above a bound."""
        eligible = [item for item in MARK_CONCENTRATION_GRID if item <= value]
        return float(
            min(MARK_CONCENTRATION_GRID) if not eligible else max(eligible)
        )

    selected = _grid_floor(min(lower_by_scenario.values()))
    selected_multi = selected
    calibration = {
        "method": (
            "training_mean_dirichlet_moment_lower_quantile_"
            "cardinality_conservative_v2"
        ),
        "lower_quantile": _MARK_CALIBRATION_LOWER_QUANTILE,
        "lower_quantile_moment_concentration_by_scenario": (
            lower_by_scenario
        ),
        "selected_concentration": float(selected),
        "selected_multi_isotope_concentration": float(selected_multi),
        "training_scene_seeds": list(seeds),
        "scenario_ids": list(_MEAN_TRAINING_SCENARIOS),
        "pair_ids": list(pair_ids),
        "artifact_sha256_by_scene_and_scenario": artifact_hashes,
        "design_sha256": file_sha256(design_path),
        "holdout_artifacts_consumed": False,
    }
    return float(selected), float(selected_multi), calibration


def build_short_candidate(
    *,
    training_root: Path,
    base_model: GeometryConditionedSpectralModel,
    mean_correction: LowRankSpectralMeanCorrection | None,
    mean_training_root: Path,
) -> tuple[dict[str, object], GeometryConditionedSpectralModel]:
    """Select and return one short training-only discrepancy candidate."""
    paths_by_group = _declared_training_paths(training_root)
    line_hash = line_identity_contract_sha256(base_model)
    arrays_by_group: dict[
        tuple[int, str],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    pair_ids_by_scene_and_scenario: dict[str, dict[str, list[int]]] = {}
    artifact_hashes: dict[str, dict[str, dict[str, str]]] = {}
    for (seed, scenario), paths in sorted(paths_by_group.items()):
        records = tuple(
            load_acceptance_pair(
                path,
                expected_line_identity_sha256=line_hash,
            )
            for path in paths
        )
        if any(
            record.split != "training"
            or record.scene_seed != seed
            or record.scenario_id != scenario
            for record in records
        ):
            raise RuntimeError("Training pair identity is contaminated.")
        pair_ids = [int(record.shield_pair_id) for record in records]
        if len(set(pair_ids)) != len(pair_ids):
            raise RuntimeError("Training group contains duplicate shield pairs.")
        arrays_by_group[(seed, scenario)] = _group_arrays(
            records,
            model=base_model,
        )
        pair_ids_by_scene_and_scenario.setdefault(str(seed), {})[
            scenario
        ] = pair_ids
        artifact_hashes.setdefault(str(seed), {})[scenario] = {
            str(record.shield_pair_id): file_sha256(path)
            for record, path in zip(records, paths, strict=True)
        }
    additive = base_model.additive_scatter_response
    if additive is None:
        raise RuntimeError("Base model lacks the fitted additive response.")
    (
        selected_mark_concentration,
        selected_multi_mark_concentration,
        mark_calibration,
    ) = (
        _select_mark_concentration_from_mean_training(
            mean_training_root,
            model=base_model,
        )
    )
    candidate_scores: dict[str, float] = {}
    candidate_specs = [
        (
            float(width),
            None if float(width) == 0.0 else "view_independent",
        )
        for width in RATE_SCALE_HALF_WIDTH_GRID
    ]
    for width, scope in candidate_specs:
        count_concentration = (
            None if width == 0.0 else 3.0 / float(width**2)
        )
        candidate = GeometryConditionedSpectralModel.standard_native(
            ACCEPTANCE_ISOTOPES,
            dead_time_tau_s=base_model.dead_time_tau_s,
            background_rate_cps=base_model.background_rate_cps,
            count_discrepancy_concentration=count_concentration,
            count_discrepancy_scope=scope,
            mark_concentration_source=selected_mark_concentration,
            mark_concentration_multi_isotope=(
                selected_multi_mark_concentration
            ),
            additive_scatter_response=additive,
            low_rank_spectral_mean_correction=mean_correction,
        )
        score = 0.0
        for observed, total, uncollided, features in arrays_by_group.values():
            value = candidate.log_likelihood_numpy(
                observed,
                total[np.newaxis, ...],
                uncollided[np.newaxis, ...],
                features[np.newaxis, ...],
                np.full(observed.shape[0], 30.0, dtype=np.float64),
            )
            score += float(value[0])
        candidate_scores[
            f"rate_half_width={width:.12g};scope={scope or 'none'}"
        ] = score
    best_score = max(candidate_scores.values())
    tied: list[tuple[float, str | None]] = []
    for width, scope in candidate_specs:
        key = f"rate_half_width={width:.12g};scope={scope or 'none'}"
        if candidate_scores[key] >= best_score - 1.0e-12:
            tied.append((float(width), scope))
    selected_width, selected_scope = min(
        tied,
        key=lambda item: (
            item[0],
            0 if item[1] == "station_shared" else 1,
        ),
    )
    selected_concentration = selected_mark_concentration
    score_payload: dict[str, object] = {
        "schema_version": 1,
        "training_policy": (
            "declared_runtime_training_no_holdout_feedback_v2"
        ),
        "training_scene_seeds": list(_TRAINING_SCENE_SEEDS),
        "scenario_ids": list(_TRAINING_SCENARIOS),
        "pair_ids_by_scene_and_scenario": pair_ids_by_scene_and_scenario,
        "artifact_sha256_by_scene_and_scenario": artifact_hashes,
        "rate_scale_half_width_grid": list(RATE_SCALE_HALF_WIDTH_GRID),
        "mark_concentration_grid": list(MARK_CONCENTRATION_GRID),
        "candidate_scores": candidate_scores,
        "mark_calibration": mark_calibration,
        "selected_rate_scale_half_width": selected_width,
        "selected_count_discrepancy_scope": selected_scope,
        "selected_mark_concentration_source": selected_concentration,
        "selected_mark_concentration_multi_isotope": (
            selected_multi_mark_concentration
        ),
        "selected_training_log_predictive_density": best_score,
        "holdout_artifacts_consumed": False,
    }
    selection_hash = canonical_json_sha256(score_payload)
    manifest: dict[str, object] = {
        "schema_version": 2,
        "training_policy": score_payload["training_policy"],
        "acceptance_contract_sha256": (
            FULL_SPECTRUM_ACCEPTANCE_CONTRACT_SHA256
        ),
        "training_scene_seeds": list(_TRAINING_SCENE_SEEDS),
        "scenario_ids": list(_TRAINING_SCENARIOS),
        "pair_ids_by_scene_and_scenario": pair_ids_by_scene_and_scenario,
        "artifact_sha256_by_scene_and_scenario": artifact_hashes,
        "rate_scale_family": (
            "view_conditioned_gamma_poisson_recorded_count_mean_one"
        ),
        "mark_family": "source_fraction_dirichlet_multinomial",
        "mark_calibration": mark_calibration,
        "selection_objective": "maximum_joint_training_log_predictive_density",
        "selected_rate_scale_half_width": selected_width,
        "selected_count_discrepancy_scope": selected_scope,
        "selected_mark_concentration_source": selected_concentration,
        "selected_mark_concentration_multi_isotope": (
            selected_multi_mark_concentration
        ),
        "candidate_count": (
            len(candidate_specs)
            + 2 * len(MARK_CONCENTRATION_GRID)
        ),
        "selected_training_log_predictive_density": best_score,
        "selection_artifact_sha256": selection_hash,
        "selection_completed": True,
        "holdout_artifacts_consumed": False,
    }
    selected_count_concentration = (
        None
        if selected_width == 0.0
        else 3.0 / float(selected_width**2)
    )
    selected_model = GeometryConditionedSpectralModel.standard_native(
        ACCEPTANCE_ISOTOPES,
        dead_time_tau_s=base_model.dead_time_tau_s,
        background_rate_cps=base_model.background_rate_cps,
        count_discrepancy_concentration=selected_count_concentration,
        count_discrepancy_scope=selected_scope,
        mark_concentration_source=selected_concentration,
        mark_concentration_multi_isotope=(
            selected_multi_mark_concentration
        ),
        discrepancy_training_manifest=manifest,
        additive_scatter_response=additive,
        low_rank_spectral_mean_correction=mean_correction,
    )
    if not selected_model.runtime_ready or selected_model.production_ready:
        raise RuntimeError(
            "Short candidate readiness contract is invalid: "
            f"discrepancy={selected_model.discrepancy_training_ready}, "
            f"runtime={selected_model.runtime_ready}, "
            f"production={selected_model.production_ready}."
        )
    selection = {
        **score_payload,
        "selection_artifact_sha256": selection_hash,
        "selected_model_contract_sha256": selected_model.contract_hash_sha256,
        "discrepancy_training_manifest": manifest,
    }
    return selection, selected_model


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write one canonical generated artifact atomically."""
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(destination)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the short candidate without consuming holdout artifacts."""
    arguments = _parser().parse_args(argv)
    base_model = _load_base_model(arguments.base_model.resolve())
    mean_correction = (
        None
        if arguments.disable_mean_correction
        else _load_mean_correction(arguments.mean_correction.resolve())
    )
    selection, model = build_short_candidate(
        training_root=arguments.training_root.resolve(),
        base_model=base_model,
        mean_correction=mean_correction,
        mean_training_root=arguments.mean_training_root.resolve(),
    )
    _write_json(arguments.selection_output, selection)
    _write_json(arguments.output_model, model.manifest_payload())
    print(arguments.output_model.resolve())
    print(model.contract_hash_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
