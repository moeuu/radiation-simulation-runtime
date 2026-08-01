"""Tests for fixed-quota Geant4 calibration sufficient statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spectrum.mean_calibration import parse_mean_calibration_metadata


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


def _forced_metadata() -> tuple[
    dict[str, object],
    tuple[np.ndarray, np.ndarray],
]:
    """Return non-one-hot v2 branch clusters and their original histories."""
    payload = _metadata()
    payload["mean_calibration_forced_collision"] = True
    payload["mean_calibration_covariance_semantics"] = (
        "independent_mu_phi_stratum_original_history_branch_cluster_"
        "sufficient_statistics_v2"
    )
    histories = (
        np.asarray(
            [
                [0.25, 0.75, 0.0],
                [0.0, 0.4, 0.1],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    rows = (
        {
            "cluster_first": "0:0.25,4:1.15,8:0.1",
            "cluster_outer": (
                "0:0:0.0625,0:4:0.1875,4:4:0.7225,"
                "4:8:0.04,8:8:0.01"
            ),
            "combined_first": "0:0.25,1:1.15,2:0.1",
            "combined_outer": (
                "0:0:0.0625,0:1:0.1875,1:1:0.7225,"
                "1:2:0.04,2:2:0.01"
            ),
            "class_first": ("0:0.25", "1:1.15", "2:0.1"),
        },
        {
            "cluster_first": "0:0.5,1:0.2,3:0.5,7:0.3",
            "cluster_outer": (
                "0:0:0.25,0:3:0.25,1:1:0.04,1:7:0.06,"
                "3:3:0.25,7:7:0.09"
            ),
            "combined_first": "0:1,1:0.5",
            "combined_outer": "0:0:1,1:1:0.25",
            "class_first": ("0:0.5,1:0.2", "0:0.5", "1:0.3"),
        },
    )
    for batch_index, row in enumerate(rows):
        prefix = f"mean_calibration_batch_{batch_index}_"
        for entry_class in (
            "uncollided_primary",
            "interacted_primary",
            "secondary",
        ):
            del payload[
                prefix + "sparse_entry_histogram_" + entry_class
            ]
        payload.update(
            {
                prefix
                + "cluster_coordinate_semantics": (
                    "entry_class_major_then_energy_bin"
                ),
                prefix
                + "cluster_score_semantics": (
                    "sum_branch_relative_bias_weight_one_hot_per_"
                    "original_history"
                ),
                prefix + "sparse_cluster_first_sum": row["cluster_first"],
                prefix + "sparse_cluster_sum_outer": row["cluster_outer"],
                prefix
                + "sparse_combined_bin_first_sum": row["combined_first"],
                prefix
                + "sparse_combined_bin_sum_outer": row["combined_outer"],
            }
        )
        for entry_class, value in zip(
            (
                "uncollided_primary",
                "interacted_primary",
                "secondary",
            ),
            row["class_first"],
            strict=True,
        ):
            payload[
                prefix + "sparse_cluster_first_sum_" + entry_class
            ] = value
    return payload, histories


def test_fixed_quota_statistics_reconstruct_mean_and_covariance() -> None:
    """Sparse histories must reproduce the exact stratified estimator."""
    calibration = parse_mean_calibration_metadata(_metadata(), bin_count=3)

    np.testing.assert_array_equal(
        calibration.raw_mean(),
        np.asarray([2.0, 4.0, 0.0]),
    )
    expected_covariance = np.asarray(
        [
            [4.0, -4.0, 0.0],
            [-4.0, 8.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(
        calibration.raw_covariance(),
        expected_covariance,
    )
    calibration.validate_native_arrays(
        [2.0, 4.0, 0.0],
        np.diag(expected_covariance),
    )


def test_detector_response_is_rao_blackwellized() -> None:
    """Response marking must transform moments without categorical sampling."""
    calibration = parse_mean_calibration_metadata(_metadata(), bin_count=3)
    response = np.asarray(
        [
            [1.0, 0.25, 0.0],
            [0.0, 0.75, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(
        calibration.marked_mean(response),
        response @ calibration.raw_mean(),
    )
    np.testing.assert_allclose(
        calibration.marked_covariance(response),
        response @ calibration.raw_covariance() @ response.T,
    )


def test_entry_class_line_totals_preserve_label_uncertainty() -> None:
    """Per-class source-line labels must retain their sampling variance."""
    calibration = parse_mean_calibration_metadata(_metadata(), bin_count=3)

    totals = calibration.entry_class_line_totals()[
        "src0_Cs-137_e662p0"
    ]
    assert totals["uncollided_primary"] == pytest.approx((2.0, 4.0))
    assert totals["interacted_primary"] == pytest.approx((4.0, 8.0))
    assert totals["secondary"] == pytest.approx((0.0, 0.0))


def test_forced_collision_clusters_reconstruct_full_moments() -> None:
    """Branch clusters must retain cross-bin covariance without one-hot logic."""
    metadata, histories = _forced_metadata()
    calibration = parse_mean_calibration_metadata(metadata, bin_count=3)
    expected_mean = np.zeros(3, dtype=np.float64)
    expected_covariance = np.zeros((3, 3), dtype=np.float64)
    for batch_histories in histories:
        first = np.sum(batch_histories, axis=0)
        second = batch_histories.T @ batch_histories
        expected_mean += 2.0 * first
        expected_covariance += 8.0 * (
            second - np.outer(first, first) / 2.0
        )

    assert calibration.statistics_version == 2
    np.testing.assert_allclose(calibration.raw_mean(), expected_mean)
    np.testing.assert_allclose(
        calibration.raw_covariance(),
        expected_covariance,
    )
    response = np.asarray(
        [
            [1.0, 0.25, 0.0],
            [0.0, 0.75, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(
        calibration.marked_mean(response),
        response @ expected_mean,
    )
    np.testing.assert_allclose(
        calibration.marked_covariance(response),
        response @ expected_covariance @ response.T,
    )
    totals = calibration.entry_class_line_totals()[
        "src0_Cs-137_e662p0"
    ]
    assert totals["uncollided_primary"] == pytest.approx((1.9, 0.61))
    assert totals["interacted_primary"] == pytest.approx((3.3, 1.49))
    assert totals["secondary"] == pytest.approx((0.8, 0.4))
    calibration.validate_native_arrays(
        expected_mean,
        np.diag(expected_covariance),
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (
            "mean_calibration_batch_0_sparse_entry_histogram_secondary",
            "0:1",
        ),
        (
            "mean_calibration_batch_0_sparse_combined_bin_first_sum",
            "0:0.25,1:1.14,2:0.1",
        ),
        (
            "mean_calibration_batch_0_sparse_cluster_sum_outer",
            "0:0:0.001",
        ),
    ],
)
def test_forced_collision_statistics_fail_closed(
    key: str,
    value: object,
) -> None:
    """Mixed or algebraically inconsistent v2 moments must be rejected."""
    metadata, _ = _forced_metadata()
    metadata[key] = value

    with pytest.raises((TypeError, ValueError)):
        parse_mean_calibration_metadata(metadata, bin_count=3)


def test_actual_native_forced_collision_response_reconstructs() -> None:
    """A captured native v2 response must reproduce its dense mean and variance."""
    fixture_path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "force_collision_thin_forced_v2.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert (
        fixture["source_sha256"]
        == "b89df6a345ca503adc4633eedafc331167bca6b81025b6fa98c4b1ab2e3cb2f6"
    )
    bin_count = int(fixture["metadata"]["spectrum_bin_count"])
    spectrum = np.zeros(bin_count, dtype=np.float64)
    variance = np.zeros(bin_count, dtype=np.float64)
    for bin_index, value in fixture["spectrum_sparse"]:
        spectrum[int(bin_index)] = float(value)
    for bin_index, value in fixture["variance_sparse"]:
        variance[int(bin_index)] = float(value)

    calibration = parse_mean_calibration_metadata(
        fixture["metadata"],
        bin_count=bin_count,
    )

    assert calibration.statistics_version == 2
    calibration.validate_native_arrays(spectrum, variance)
    np.testing.assert_allclose(
        np.diag(calibration.raw_covariance()),
        variance,
        rtol=2.0e-10,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mean_calibration_enabled", False),
        ("history_thinning_enabled", True),
        ("transport_tally_weighted", False),
        ("mean_calibration_forced_collision", True),
        ("mean_calibration_angle_stratum_count", 3),
    ],
)
def test_mean_calibration_metadata_fails_closed(
    field: str,
    value: object,
) -> None:
    """Any semantic drift in weighted calibration metadata must be rejected."""
    payload = _metadata()
    payload[field] = value

    with pytest.raises((TypeError, ValueError)):
        parse_mean_calibration_metadata(payload, bin_count=3)
