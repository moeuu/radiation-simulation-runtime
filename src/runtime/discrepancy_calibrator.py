"""Fit estimator-neutral structured spectral discrepancy from holdout runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np
from numpy.typing import NDArray

from runtime.discrepancy_calibration import DiscrepancyCalibration


_INPUT_KEYS = frozenset(
    {
        "observed_counts",
        "expected_counts",
        "energy_bin_edges_keV",
        "environment_ids",
        "shield_pair_ids",
    }
)


def _normalized_nonnegative(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a nonnegative spectrum, falling back to a uniform shape."""
    result = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(np.sum(result))
    if total <= 0.0:
        result = np.ones_like(result)
        total = float(result.size)
    return result / total


def _shield_features() -> NDArray[np.float64]:
    """Return low-dimensional periodic features for all 64 Fe/Pb pairs."""
    features = []
    for pair_id in range(64):
        fe, pb = divmod(pair_id, 8)
        fe_angle = 2.0 * np.pi * fe / 8.0
        pb_angle = 2.0 * np.pi * pb / 8.0
        features.append(
            [
                1.0,
                0.5 * (1.0 + np.cos(fe_angle)),
                0.5 * (1.0 + np.sin(fe_angle)),
                0.5 * (1.0 + np.cos(pb_angle)),
                0.5 * (1.0 + np.sin(pb_angle)),
                0.5 * (1.0 + np.cos(fe_angle - pb_angle)),
            ]
        )
    return np.asarray(features, dtype=np.float64)


def _low_rank_nonnegative_basis(
    residual: NDArray[np.float64],
    rank: int,
) -> NDArray[np.float64]:
    """Represent signed right-singular modes with positive/negative columns."""
    centered = residual - np.mean(residual, axis=0, keepdims=True)
    _left, _singular, right = np.linalg.svd(centered, full_matrices=False)
    basis: list[NDArray[np.float64]] = []
    for vector in right[: min(int(rank), right.shape[0])]:
        for part in (np.maximum(vector, 0.0), np.maximum(-vector, 0.0)):
            if np.any(part > 0.0):
                basis.append(_normalized_nonnegative(part))
    if not basis:
        basis.append(_normalized_nonnegative(np.ones(residual.shape[1])))
    return np.vstack(basis)


def calibrate_discrepancy(
    input_path: str | Path,
    output_path: str | Path,
    *,
    calibration_id: str,
    residual_rank: int = 3,
) -> DiscrepancyCalibration:
    """Fit and save a strict calibration from independent all-pair spectra."""
    if isinstance(residual_rank, bool) or int(residual_rank) < 1:
        raise ValueError("residual_rank must be a positive integer.")
    source = Path(input_path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != _INPUT_KEYS:
            raise ValueError(
                "Discrepancy calibration NPZ fields disagree with the strict input schema."
            )
        observed = np.asarray(archive["observed_counts"], dtype=np.float64)
        expected = np.asarray(archive["expected_counts"], dtype=np.float64)
        edges = np.asarray(archive["energy_bin_edges_keV"], dtype=np.float64)
        environment_ids = np.asarray(archive["environment_ids"]).astype(str)
        pair_ids = np.asarray(archive["shield_pair_ids"], dtype=np.int64)
    if observed.ndim != 2 or expected.shape != observed.shape:
        raise ValueError("observed_counts and expected_counts must share shape (N, B).")
    if edges.shape != (observed.shape[1] + 1,) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy_bin_edges_keV must align with the spectra.")
    if environment_ids.shape != (observed.shape[0],) or pair_ids.shape != (
        observed.shape[0],
    ):
        raise ValueError("Calibration row metadata must align with spectra.")
    environments = tuple(sorted(set(environment_ids.tolist())))
    if len(environments) < 2:
        raise ValueError("Calibration requires at least two independent environments.")
    if set(pair_ids.tolist()) != set(range(64)):
        raise ValueError("Calibration input must cover all 64 shield pairs.")
    required_pairs = set(range(64))
    for environment_id in environments:
        covered = set(pair_ids[environment_ids == environment_id].tolist())
        if covered != required_pairs:
            raise ValueError(
                "Every independent environment must cover all 64 shield pairs."
            )
    if np.any(~np.isfinite(observed)) or np.any(~np.isfinite(expected)):
        raise ValueError("Calibration spectra must contain finite values.")
    if np.any(observed < 0.0) or np.any(expected < 0.0):
        raise ValueError("Calibration spectra must be nonnegative.")
    residual = observed - expected
    positive_mean = _normalized_nonnegative(np.mean(np.maximum(residual, 0.0), axis=0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    low_energy_weight = np.exp(-centers / max(float(edges[-1]) * 0.25, 1.0))
    scatter = _normalized_nonnegative(positive_mean * low_energy_weight)
    low_rank = _low_rank_nonnegative_basis(residual, int(residual_rank))
    leakage_rows = []
    for pair_id in range(64):
        pair_residual = residual[pair_ids == pair_id]
        leakage_rows.append(np.mean(np.maximum(pair_residual, 0.0), axis=0))
    leakage = _normalized_nonnegative(np.mean(leakage_rows, axis=0))[None, :]
    mean_expected = np.mean(expected, axis=0)
    gain_derivative = np.gradient(mean_expected)[None, :]
    resolution_derivative = np.gradient(np.gradient(mean_expected))[None, :]
    group_residuals = []
    for pair_id in range(64):
        values = observed[pair_ids == pair_id]
        if values.shape[0] >= 2:
            group_residuals.append(values - np.mean(values, axis=0, keepdims=True))
    if not group_residuals:
        raise ValueError("Every-pair calibration needs replicated independent rows.")
    empirical_variance = np.mean(
        np.vstack(group_residuals) ** 2,
        axis=0,
    )
    mean_count = np.maximum(np.mean(observed, axis=0), 1.0)
    alpha = np.maximum((empirical_variance - mean_count) / mean_count**2, 0.0)
    finite_cap = float(np.quantile(alpha, 0.99)) if alpha.size else 0.0
    alpha = np.minimum(alpha, finite_cap)
    payload = {
        "schema_version": 1,
        "calibration_id": str(calibration_id),
        "independent_environment_ids": list(environments),
        "shield_pair_ids": list(range(64)),
        "energy_bin_edges_keV": edges.tolist(),
        "background_basis": [positive_mean.tolist()],
        "scatter_basis": [scatter.tolist()],
        "shield_pair_feature_basis": _shield_features().tolist(),
        "shield_leakage_basis": leakage.tolist(),
        "low_rank_spectral_residual_basis": low_rank.tolist(),
        "gain_derivative_basis": gain_derivative.tolist(),
        "resolution_derivative_basis": resolution_derivative.tolist(),
        "overdispersion": {
            "family": "negative_binomial",
            "alpha_by_bin": alpha.tolist(),
        },
        "shrinkage_l2_by_family": {
            "background": 1.0,
            "scatter": 1.0,
            "shield_leakage": 10.0,
            "station_rate": 10.0,
            "low_rank_residual": 10.0,
            "gain_drift": 100.0,
            "resolution_drift": 100.0,
        },
    }
    calibration = DiscrepancyCalibration.from_mapping(payload)
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return calibration


__all__ = ["calibrate_discrepancy"]
