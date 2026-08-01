"""Evaluate one frozen spectral-mean model on a short unused Geant4 scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from spectrum.mean_calibration_runner import (
    canonical_json_bytes,
    load_mean_calibration_pair_artifact,
)
from spectrum.transport_spectral import GeometryConditionedSpectralModel


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BASE_MODEL = (
    _ROOT
    / "configs"
    / "geant4"
    / "models"
    / "geometry_conditioned_full_spectrum_exact_v1.json"
)
_DEFAULT_CANDIDATE_MODEL = (
    _ROOT
    / "configs"
    / "geant4"
    / "models"
    / "geometry_conditioned_full_spectrum_ral_eu154_training_v4.json"
)


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=_DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--candidate-model",
        type=Path,
        default=_DEFAULT_CANDIDATE_MODEL,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_model(path: Path) -> GeometryConditionedSpectralModel:
    """Load and authenticate one file-backed full-spectrum model."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = GeometryConditionedSpectralModel.from_manifest_payload(payload)
    model.require_runtime_ready()
    return model


def _predicted_source_components(
    model: GeometryConditionedSpectralModel,
    payload: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Return marked means and transport line contributions for one pair."""
    geometry = payload.get("geometry")
    provenance = payload.get("provenance")
    if not isinstance(geometry, Mapping) or not isinstance(provenance, Mapping):
        raise TypeError("Holdout pair lacks geometry or provenance.")
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
    scatter_basis = np.asarray(
        geometry["additive_scatter_basis_slf"],
        dtype=np.float64,
    )
    additive = model.additive_scatter_response
    if additive is None:
        raise RuntimeError("Holdout model lacks an additive scatter response.")
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
    return (
        np.asarray(source[0, 0], dtype=np.float64),
        np.asarray(total, dtype=np.float64),
    )


def _discrepancy_envelope_metrics(
    model: GeometryConditionedSpectralModel,
    target: np.ndarray,
    predicted: np.ndarray,
    transport_total_vsl: np.ndarray,
) -> dict[str, float | bool | None]:
    """Test one unused pair against the frozen count and mark discrepancy."""
    target_total = float(np.sum(target))
    predicted_total = float(np.sum(predicted))
    count_concentration = model.count_discrepancy_concentration
    if count_concentration is None:
        count_standardized = None
        count_covered = None
    else:
        fractional_sigma = 1.0 / np.sqrt(float(count_concentration))
        count_standardized = (
            target_total - predicted_total
        ) / (predicted_total * fractional_sigma)
        count_covered = bool(abs(count_standardized) <= 2.5758293035489004)

    target_probability = target / target_total
    predicted_probability = predicted / predicted_total
    squared_mark_error = float(
        np.sum(np.square(target_probability - predicted_probability))
    )
    probability_variance = float(
        1.0 - np.sum(np.square(predicted_probability))
    )
    if squared_mark_error <= 0.0:
        moment_concentration = float("inf")
    else:
        moment_concentration = max(
            probability_variance / squared_mark_error - 1.0,
            0.0,
        )
    if model.mark_concentration_source is None:
        configured_mark_concentration = None
        mark_covered = None
    else:
        configured_mark = model._base_mark_concentration_numpy(
            transport_total_vsl[np.newaxis, np.newaxis, ...]
        )
        configured_mark_concentration = float(configured_mark[0, 0])
        mark_covered = bool(
            moment_concentration >= configured_mark_concentration
        )
    return {
        "count_standardized_model_error": (
            None
            if count_standardized is None
            else float(count_standardized)
        ),
        "count_within_two_sided_99_percent_envelope": count_covered,
        "mark_moment_concentration": float(moment_concentration),
        "configured_mark_concentration": configured_mark_concentration,
        "mark_discrepancy_is_conservative": mark_covered,
        "joint_discrepancy_envelope_passed": bool(
            count_covered is True and mark_covered is True
        ),
    }


def _mean_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return total and conditional-mark errors for one mean spectrum."""
    target_total = float(np.sum(target))
    predicted_total = float(np.sum(predicted))
    if target_total <= 0.0 or predicted_total <= 0.0:
        raise ValueError("Short mean holdout requires positive source spectra.")
    target_probability = target / target_total
    predicted_probability = predicted / predicted_total
    cosine_denominator = float(
        np.linalg.norm(target) * np.linalg.norm(predicted)
    )
    floor = max(target_total, predicted_total) * 1.0e-12 / target.size
    return {
        "predicted_to_target_total_ratio": predicted_total / target_total,
        "absolute_total_relative_error": abs(predicted_total / target_total - 1.0),
        "conditional_mark_total_variation": float(
            0.5 * np.sum(np.abs(target_probability - predicted_probability))
        ),
        "spectral_cosine_similarity": float(
            np.dot(target, predicted) / cosine_denominator
        ),
        "target_probability_weighted_log_mse": float(
            np.sum(
                target_probability
                * np.square(
                    np.log((predicted + floor) / (target + floor))
                )
            )
        ),
    }


def _aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Aggregate each metric without changing the predeclared pair set."""
    keys = tuple(rows[0])
    return {
        f"mean_{key}": float(np.mean([row[key] for row in rows]))
        for key in keys
    } | {
        f"maximum_{key}": float(np.max([row[key] for row in rows]))
        for key in keys
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate the frozen candidate once, without selection or refitting."""
    arguments = _parser().parse_args(argv)
    root = arguments.holdout_root.resolve()
    design = json.loads((root / "training_design.json").read_text("utf-8"))
    seeds = tuple(int(value) for value in design["training_scene_seeds"])
    scenarios = tuple(str(value) for value in design["scenario_ids"])
    pair_ids = tuple(int(value) for value in design["shield_pair_ids"])
    if (
        len(seeds) != 1
        or seeds[0] in (2026072701, 2026072702)
        or design.get("holdout_consumed_by_training") is not False
    ):
        raise ValueError("Short holdout must be one unused non-training scene.")
    models = {
        "uncorrected_physical_mean": _load_model(
            arguments.base_model.resolve()
        ),
        "frozen_candidate": _load_model(arguments.candidate_model.resolve()),
    }
    pair_results: list[dict[str, object]] = []
    metric_rows: dict[str, list[dict[str, float]]] = {
        name: [] for name in models
    }
    for scenario in scenarios:
        for pair_id in pair_ids:
            manifest = (
                root
                / "pairs"
                / f"scene_{seeds[0]}"
                / scenario
                / f"pair_{pair_id:02d}"
                / "manifest.json"
            )
            payload, arrays = load_mean_calibration_pair_artifact(manifest)
            target = np.asarray(arrays["marked_mean"], dtype=np.float64)
            metrics_by_model: dict[
                str,
                dict[str, float | bool | None],
            ] = {}
            for name, model in models.items():
                predicted, transport_total = _predicted_source_components(
                    model,
                    payload,
                )
                metrics = _mean_metrics(
                    target,
                    predicted,
                )
                discrepancy = _discrepancy_envelope_metrics(
                    model,
                    target,
                    predicted,
                    transport_total,
                )
                metrics_by_model[name] = metrics | discrepancy
                metric_rows[name].append(metrics)
            pair_results.append(
                {
                    "scenario_id": scenario,
                    "shield_pair_id": pair_id,
                    "metrics": metrics_by_model,
                }
            )
    payload = {
        "schema_version": 1,
        "evaluation": "short_unused_scene_spectral_mean_holdout",
        "fit_selection_or_tuning_performed": False,
        "scene_seed": seeds[0],
        "scenario_ids": list(scenarios),
        "shield_pair_ids": list(pair_ids),
        "models": {
            name: {
                "contract_hash_sha256": model.contract_hash_sha256,
                "runtime_ready": model.runtime_ready,
                "production_ready": model.production_ready,
                "aggregate_metrics": _aggregate(metric_rows[name]),
                "all_pairs_pass_discrepancy_envelope": all(
                    pair["metrics"][name][
                        "joint_discrepancy_envelope_passed"
                    ]
                    is True
                    for pair in pair_results
                ),
            }
            for name, model in models.items()
        },
        "pairs": pair_results,
    }
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(payload))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
