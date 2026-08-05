"""Estimator-neutral calibration contract for structured model discrepancy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


DISCREPANCY_CALIBRATION_SCHEMA_VERSION = 1
_FIELDS = frozenset(
    {
        "schema_version",
        "calibration_id",
        "independent_environment_ids",
        "shield_pair_ids",
        "energy_bin_edges_keV",
        "background_basis",
        "scatter_basis",
        "shield_pair_feature_basis",
        "shield_leakage_basis",
        "low_rank_spectral_residual_basis",
        "gain_derivative_basis",
        "resolution_derivative_basis",
        "overdispersion",
        "shrinkage_l2_by_family",
    }
)


def _matrix(
    values: ArrayLike,
    *,
    name: str,
    columns: int,
    non_negative: bool,
) -> NDArray[np.float64]:
    """Return one finite two-dimensional calibration matrix."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise ValueError(f"{name} must have shape (K, {columns}).")
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite values.")
    if non_negative and np.any(matrix < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    matrix = np.array(matrix, dtype=np.float64, copy=True)
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True, slots=True)
class DiscrepancyCalibration:
    """Store shared spectral bases calibrated outside an evaluation run."""

    calibration_id: str
    independent_environment_ids: tuple[str, ...]
    energy_bin_edges_keV: NDArray[np.float64]
    background_basis: NDArray[np.float64]
    scatter_basis: NDArray[np.float64]
    shield_pair_feature_basis: NDArray[np.float64]
    shield_leakage_basis: NDArray[np.float64]
    low_rank_spectral_residual_basis: NDArray[np.float64]
    gain_derivative_basis: NDArray[np.float64]
    resolution_derivative_basis: NDArray[np.float64]
    overdispersion_family: str
    overdispersion_alpha_by_bin: NDArray[np.float64]
    shrinkage_l2_by_family: Mapping[str, float]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DiscrepancyCalibration":
        """Validate one strict all-pair independent calibration payload."""
        if set(payload) != _FIELDS:
            raise ValueError(
                "Discrepancy calibration fields disagree with schema version 1."
            )
        if payload.get("schema_version") != DISCREPANCY_CALIBRATION_SCHEMA_VERSION:
            raise ValueError("Unsupported discrepancy calibration schema_version.")
        calibration_id = str(payload.get("calibration_id", "")).strip()
        if not calibration_id:
            raise ValueError("calibration_id must be non-empty.")
        raw_environments = payload.get("independent_environment_ids")
        if not isinstance(raw_environments, list):
            raise TypeError("independent_environment_ids must be a JSON list.")
        environments = tuple(str(value).strip() for value in raw_environments)
        if len(environments) < 2 or any(not value for value in environments):
            raise ValueError(
                "At least two independent calibration environments are required."
            )
        if len(set(environments)) != len(environments):
            raise ValueError("independent_environment_ids must be unique.")
        pair_ids = payload.get("shield_pair_ids")
        if pair_ids != list(range(64)):
            raise ValueError("Calibration must cover all shield pair IDs 0 through 63.")
        edges = np.asarray(payload.get("energy_bin_edges_keV"), dtype=np.float64)
        if (
            edges.ndim != 1
            or edges.size < 2
            or np.any(~np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError("energy_bin_edges_keV must be finite and increasing.")
        bin_count = edges.size - 1
        pair_features = np.asarray(
            payload.get("shield_pair_feature_basis"),
            dtype=np.float64,
        )
        if pair_features.ndim != 2 or pair_features.shape[0] != 64:
            raise ValueError("shield_pair_feature_basis must have shape (64, K).")
        if np.any(~np.isfinite(pair_features)) or np.any(pair_features < 0.0):
            raise ValueError(
                "shield_pair_feature_basis must be finite and non-negative."
            )
        overdispersion = payload.get("overdispersion")
        if not isinstance(overdispersion, Mapping) or set(overdispersion) != {
            "family",
            "alpha_by_bin",
        }:
            raise ValueError("overdispersion must contain family and alpha_by_bin.")
        family = str(overdispersion["family"])
        if family not in {"poisson", "negative_binomial"}:
            raise ValueError("Unsupported overdispersion family.")
        alpha = np.asarray(overdispersion["alpha_by_bin"], dtype=np.float64)
        if (
            alpha.shape != (bin_count,)
            or np.any(~np.isfinite(alpha))
            or np.any(alpha < 0.0)
        ):
            raise ValueError(
                "overdispersion.alpha_by_bin must be finite and non-negative."
            )
        if family == "poisson" and np.any(alpha != 0.0):
            raise ValueError("Poisson calibration must have zero alpha_by_bin.")
        shrinkage = payload.get("shrinkage_l2_by_family")
        expected_families = {
            "background",
            "scatter",
            "shield_leakage",
            "station_rate",
            "low_rank_residual",
            "gain_drift",
            "resolution_drift",
        }
        if not isinstance(shrinkage, Mapping) or set(shrinkage) != expected_families:
            raise ValueError(
                "shrinkage_l2_by_family must contain every structured basis family."
            )
        parsed_shrinkage = {str(key): float(value) for key, value in shrinkage.items()}
        if any(
            not np.isfinite(value) or value < 0.0 for value in parsed_shrinkage.values()
        ):
            raise ValueError("Shrinkage weights must be finite and non-negative.")
        edges = np.array(edges, dtype=np.float64, copy=True)
        pair_features = np.array(pair_features, dtype=np.float64, copy=True)
        alpha = np.array(alpha, dtype=np.float64, copy=True)
        edges.setflags(write=False)
        pair_features.setflags(write=False)
        alpha.setflags(write=False)
        return cls(
            calibration_id=calibration_id,
            independent_environment_ids=environments,
            energy_bin_edges_keV=edges,
            background_basis=_matrix(
                payload.get("background_basis"),
                name="background_basis",
                columns=bin_count,
                non_negative=True,
            ),
            scatter_basis=_matrix(
                payload.get("scatter_basis"),
                name="scatter_basis",
                columns=bin_count,
                non_negative=True,
            ),
            shield_pair_feature_basis=pair_features,
            shield_leakage_basis=_matrix(
                payload.get("shield_leakage_basis"),
                name="shield_leakage_basis",
                columns=bin_count,
                non_negative=True,
            ),
            low_rank_spectral_residual_basis=_matrix(
                payload.get("low_rank_spectral_residual_basis"),
                name="low_rank_spectral_residual_basis",
                columns=bin_count,
                non_negative=True,
            ),
            gain_derivative_basis=_matrix(
                payload.get("gain_derivative_basis"),
                name="gain_derivative_basis",
                columns=bin_count,
                non_negative=False,
            ),
            resolution_derivative_basis=_matrix(
                payload.get("resolution_derivative_basis"),
                name="resolution_derivative_basis",
                columns=bin_count,
                non_negative=False,
            ),
            overdispersion_family=family,
            overdispersion_alpha_by_bin=alpha,
            shrinkage_l2_by_family=parsed_shrinkage,
        )

    def validate_energy_axis(self, edges_keV: ArrayLike) -> None:
        """Fail closed when observations use a different energy discretization."""
        edges = np.asarray(edges_keV, dtype=np.float64)
        if not np.array_equal(edges, self.energy_bin_edges_keV):
            raise ValueError(
                "Discrepancy calibration energy bins differ from the observations."
            )


def load_discrepancy_calibration(path: str | Path) -> DiscrepancyCalibration:
    """Load one estimator-neutral structured-discrepancy calibration JSON."""
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Discrepancy calibration root must be a JSON object.")
    return DiscrepancyCalibration.from_mapping(payload)


__all__ = [
    "DISCREPANCY_CALIBRATION_SCHEMA_VERSION",
    "DiscrepancyCalibration",
    "load_discrepancy_calibration",
]
