"""Tests for estimator-neutral structured-discrepancy calibration contracts."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from runtime.discrepancy_calibration import DiscrepancyCalibration


def _payload() -> dict[str, object]:
    """Return one minimal independent all-pair calibration payload."""
    return {
        "schema_version": 1,
        "calibration_id": "independent-geant4-all-pairs-v1",
        "independent_environment_ids": ["calibration-a", "calibration-b"],
        "shield_pair_ids": list(range(64)),
        "energy_bin_edges_keV": [0.0, 100.0, 200.0],
        "background_basis": [[0.7, 0.3]],
        "scatter_basis": [[0.4, 0.6]],
        "shield_pair_feature_basis": [
            [1.0, float(pair_id) / 63.0] for pair_id in range(64)
        ],
        "shield_leakage_basis": [[0.2, 0.8], [0.8, 0.2]],
        "low_rank_spectral_residual_basis": [[0.5, 0.5]],
        "gain_derivative_basis": [[-0.2, 0.2]],
        "resolution_derivative_basis": [[0.1, -0.1]],
        "overdispersion": {
            "family": "negative_binomial",
            "alpha_by_bin": [0.02, 0.03],
        },
        "shrinkage_l2_by_family": {
            "background": 1.0,
            "scatter": 1.0,
            "shield_leakage": 4.0,
            "station_rate": 10.0,
            "low_rank_residual": 8.0,
            "gain_drift": 20.0,
            "resolution_drift": 20.0,
        },
    }


def test_calibration_requires_independent_environments_and_all_64_pairs() -> None:
    """A partial or same-environment fit must never enter production inference."""
    calibration = DiscrepancyCalibration.from_mapping(_payload())

    assert calibration.shield_pair_feature_basis.shape == (64, 2)
    calibration.validate_energy_axis(np.asarray([0.0, 100.0, 200.0]))

    missing_pair = copy.deepcopy(_payload())
    missing_pair["shield_pair_ids"] = list(range(63))
    with pytest.raises(ValueError, match="all shield pair"):
        DiscrepancyCalibration.from_mapping(missing_pair)
    one_environment = copy.deepcopy(_payload())
    one_environment["independent_environment_ids"] = ["calibration-a"]
    with pytest.raises(ValueError, match="At least two independent"):
        DiscrepancyCalibration.from_mapping(one_environment)


def test_calibration_energy_axis_fails_closed() -> None:
    """Basis coefficients cannot be silently interpolated onto another spectrum."""
    calibration = DiscrepancyCalibration.from_mapping(_payload())

    with pytest.raises(ValueError, match="energy bins differ"):
        calibration.validate_energy_axis(np.asarray([0.0, 50.0, 200.0]))
