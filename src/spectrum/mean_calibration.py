"""Typed sufficient statistics for fixed-quota Geant4 mean calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


MEAN_CALIBRATION_SCHEMA_VERSION = 1
MEAN_CALIBRATION_ENTRY_CLASSES = (
    "uncollided_primary",
    "interacted_primary",
    "secondary",
)
MEAN_CALIBRATION_ANALOG_COVARIANCE_SEMANTICS = (
    "independent_mu_phi_stratum_sample_mean_cluster_"
    "sufficient_statistics_v1"
)
MEAN_CALIBRATION_FORCED_COLLISION_COVARIANCE_SEMANTICS = (
    "independent_mu_phi_stratum_original_history_branch_cluster_"
    "sufficient_statistics_v2"
)
MEAN_CALIBRATION_CLUSTER_COORDINATE_SEMANTICS = (
    "entry_class_major_then_energy_bin"
)
# The native token names one-hot leaf scores.  Their original-history sum is a
# general sparse vector, so no consumer may apply a one-hot cluster shortcut.
MEAN_CALIBRATION_CLUSTER_SCORE_SEMANTICS = (
    "sum_branch_relative_bias_weight_one_hot_per_original_history"
)


def _strict_integer(value: object, *, field_name: str) -> int:
    """Return one JSON-style integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be an integer.")
    return int(value)


def _strict_float(value: object, *, field_name: str) -> float:
    """Return one finite JSON-style floating-point value."""
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{field_name} must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field_name} must be finite.")
    return result


def _parse_sparse_histogram(
    value: object,
    *,
    field_name: str,
    bin_count: int,
) -> tuple[tuple[int, int], ...]:
    """Parse one canonical ``bin:count`` detector-entry histogram."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if value == "-":
        return ()
    result: list[tuple[int, int]] = []
    previous_bin = -1
    for item in value.split(","):
        pieces = item.split(":")
        if len(pieces) != 2:
            raise ValueError(f"{field_name} contains a malformed item.")
        try:
            bin_index = int(pieces[0])
            count = int(pieces[1])
        except ValueError as exc:
            raise ValueError(
                f"{field_name} contains a non-integer item."
            ) from exc
        if (
            bin_index <= previous_bin
            or bin_index < 0
            or bin_index >= bin_count
            or count <= 0
        ):
            raise ValueError(
                f"{field_name} must contain sorted positive in-range counts."
            )
        result.append((bin_index, count))
        previous_bin = bin_index
    return tuple(result)


def _parse_sparse_first_sum(
    value: object,
    *,
    field_name: str,
    coordinate_count: int,
) -> tuple[tuple[int, float], ...]:
    """Parse one canonical sparse first-moment vector."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if value == "-":
        return ()
    result: list[tuple[int, float]] = []
    previous_index = -1
    for item in value.split(","):
        pieces = item.split(":")
        if len(pieces) != 2:
            raise ValueError(f"{field_name} contains a malformed item.")
        try:
            index = int(pieces[0])
            moment = float(pieces[1])
        except ValueError as exc:
            raise ValueError(
                f"{field_name} contains a nonnumeric item."
            ) from exc
        if (
            str(index) != pieces[0]
            or index <= previous_index
            or index < 0
            or index >= coordinate_count
            or not np.isfinite(moment)
            or moment <= 0.0
        ):
            raise ValueError(
                f"{field_name} must contain sorted positive finite moments."
            )
        result.append((index, moment))
        previous_index = index
    return tuple(result)


def _parse_sparse_sum_outer(
    value: object,
    *,
    field_name: str,
    coordinate_count: int,
) -> tuple[tuple[int, int, float], ...]:
    """Parse one canonical sparse upper-triangular second-moment matrix."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if value == "-":
        return ()
    result: list[tuple[int, int, float]] = []
    previous_pair = (-1, -1)
    for item in value.split(","):
        pieces = item.split(":")
        if len(pieces) != 3:
            raise ValueError(f"{field_name} contains a malformed item.")
        try:
            left = int(pieces[0])
            right = int(pieces[1])
            moment = float(pieces[2])
        except ValueError as exc:
            raise ValueError(
                f"{field_name} contains a nonnumeric item."
            ) from exc
        pair = (left, right)
        if (
            str(left) != pieces[0]
            or str(right) != pieces[1]
            or pair <= previous_pair
            or left < 0
            or right < left
            or right >= coordinate_count
            or not np.isfinite(moment)
            or moment <= 0.0
        ):
            raise ValueError(
                f"{field_name} must contain sorted positive upper moments."
            )
        result.append((left, right, moment))
        previous_pair = pair
    return tuple(result)


def _sparse_vector(
    items: Sequence[tuple[int, float]],
    *,
    size: int,
) -> NDArray[np.float64]:
    """Materialize one validated sparse vector."""
    result = np.zeros(size, dtype=np.float64)
    for index, value in items:
        result[index] = value
    return result


def _sparse_symmetric_matrix(
    items: Sequence[tuple[int, int, float]],
    *,
    size: int,
) -> NDArray[np.float64]:
    """Materialize one validated sparse upper-triangular matrix."""
    result = np.zeros((size, size), dtype=np.float64)
    for left, right, value in items:
        result[left, right] = value
        result[right, left] = value
    return result


@dataclass(frozen=True)
class MeanCalibrationBatch:
    """Store one source-line and angular-stratum sufficient statistic."""

    source_token: str
    line_token: str
    expected_unthinned_histories: float
    sampled_histories: int
    history_weight: float
    angle_stratum_index: int
    statistics_version: int
    entry_histograms: tuple[tuple[tuple[int, int], ...], ...]
    cluster_first_sum: tuple[tuple[int, float], ...] = ()
    cluster_sum_outer: tuple[tuple[int, int, float], ...] = ()
    combined_bin_first_sum: tuple[tuple[int, float], ...] = ()
    combined_bin_sum_outer: tuple[tuple[int, int, float], ...] = ()

    def combined_histogram(self) -> tuple[tuple[int, int], ...]:
        """Return the mutually exclusive entry classes as one histogram."""
        if self.statistics_version != 1:
            raise RuntimeError(
                "Branch-cluster statistics do not define a count histogram."
            )
        counts: dict[int, int] = {}
        for histogram in self.entry_histograms:
            for bin_index, count in histogram:
                counts[bin_index] = counts.get(bin_index, 0) + count
        return tuple(sorted(counts.items()))

    def entry_count(self) -> int:
        """Return the number of histories that reached the detector."""
        if self.statistics_version != 1:
            raise RuntimeError(
                "Branch-cluster scores are not one-hot history counts."
            )
        return sum(
            count
            for histogram in self.entry_histograms
            for _, count in histogram
        )

    def combined_first_items(self) -> tuple[tuple[int, float], ...]:
        """Return original-history combined-bin first moments."""
        if self.statistics_version == 1:
            return tuple(
                (bin_index, float(count))
                for bin_index, count in self.combined_histogram()
            )
        return self.combined_bin_first_sum

    def combined_outer_items(self) -> tuple[tuple[int, int, float], ...]:
        """Return original-history combined-bin upper second moments."""
        if self.statistics_version == 1:
            return tuple(
                (bin_index, bin_index, float(count))
                for bin_index, count in self.combined_histogram()
            )
        return self.combined_bin_sum_outer

    def cluster_first_items(
        self,
        *,
        bin_count: int,
    ) -> tuple[tuple[int, float], ...]:
        """Return entry-class-major first moments."""
        if self.statistics_version == 2:
            return self.cluster_first_sum
        result: list[tuple[int, float]] = []
        for entry_index, histogram in enumerate(self.entry_histograms):
            result.extend(
                (
                    entry_index * bin_count + bin_index,
                    float(count),
                )
                for bin_index, count in histogram
            )
        return tuple(sorted(result))

    def cluster_outer_items(
        self,
        *,
        bin_count: int,
    ) -> tuple[tuple[int, int, float], ...]:
        """Return entry-class-major upper second moments."""
        if self.statistics_version == 2:
            return self.cluster_sum_outer
        return tuple(
            (coordinate, coordinate, value)
            for coordinate, value in self.cluster_first_items(
                bin_count=bin_count
            )
        )

    def to_payload(self) -> dict[str, object]:
        """Return one compact JSON-compatible sufficient-statistic row."""
        payload: dict[str, object] = {
            "source_token": self.source_token,
            "line_token": self.line_token,
            "expected_unthinned_histories": (
                self.expected_unthinned_histories
            ),
            "sampled_histories": self.sampled_histories,
            "history_weight": self.history_weight,
            "angle_stratum_index": self.angle_stratum_index,
        }
        if self.statistics_version == 1:
            payload["entry_histograms"] = {
                entry_class: [
                    [bin_index, count]
                    for bin_index, count in histogram
                ]
                for entry_class, histogram in zip(
                    MEAN_CALIBRATION_ENTRY_CLASSES,
                    self.entry_histograms,
                    strict=True,
                )
            }
            return payload
        payload.update(
            {
                "statistics_version": 2,
                "cluster_coordinate_semantics": (
                    MEAN_CALIBRATION_CLUSTER_COORDINATE_SEMANTICS
                ),
                "cluster_score_semantics": (
                    MEAN_CALIBRATION_CLUSTER_SCORE_SEMANTICS
                ),
                "cluster_first_sum": [
                    [index, value]
                    for index, value in self.cluster_first_sum
                ],
                "cluster_sum_outer": [
                    [left, right, value]
                    for left, right, value in self.cluster_sum_outer
                ],
                "combined_bin_first_sum": [
                    [index, value]
                    for index, value in self.combined_bin_first_sum
                ],
                "combined_bin_sum_outer": [
                    [left, right, value]
                    for left, right, value in self.combined_bin_sum_outer
                ],
            }
        )
        return payload


@dataclass(frozen=True)
class StratifiedMeanCalibration:
    """Represent an authenticated fixed-quota Geant4 calibration result."""

    bin_count: int
    histories_per_source_line: int
    angle_strata_mu: int
    angle_strata_phi: int
    batches: tuple[MeanCalibrationBatch, ...]

    @property
    def angle_stratum_count(self) -> int:
        """Return the number of equal-solid-angle strata."""
        return self.angle_strata_mu * self.angle_strata_phi

    @property
    def statistics_version(self) -> int:
        """Return the common native sufficient-statistic version."""
        versions = {batch.statistics_version for batch in self.batches}
        if len(versions) != 1:
            raise RuntimeError(
                "Mean calibration contains mixed statistic semantics."
            )
        return next(iter(versions))

    def raw_mean(self) -> NDArray[np.float64]:
        """Return the weighted mean incident spectrum."""
        mean = np.zeros(self.bin_count, dtype=np.float64)
        for batch in self.batches:
            for bin_index, first_sum in batch.combined_first_items():
                mean[bin_index] += batch.history_weight * first_sum
        return mean

    def raw_covariance(self) -> NDArray[np.float64]:
        """Materialize the exact stratified sample-mean covariance."""
        covariance = np.zeros(
            (self.bin_count, self.bin_count),
            dtype=np.float64,
        )
        for batch in self.batches:
            first_sum = _sparse_vector(
                batch.combined_first_items(),
                size=self.bin_count,
            )
            second_sum = _sparse_symmetric_matrix(
                batch.combined_outer_items(),
                size=self.bin_count,
            )
            sample_count = float(batch.sampled_histories)
            correction = (
                batch.history_weight**2
                * sample_count
                / (sample_count - 1.0)
            )
            covariance += correction * (
                second_sum
                - np.outer(first_sum, first_sum) / sample_count
            )
        return 0.5 * (covariance + covariance.T)

    def marked_mean(
        self,
        response_operator_br: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Rao–Blackwellize detector-response marking without extra noise."""
        response = self._validated_response_operator(response_operator_br)
        return np.asarray(response @ self.raw_mean(), dtype=np.float64)

    def marked_covariance(
        self,
        response_operator_br: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return covariance after analytic detector-response integration."""
        response = self._validated_response_operator(response_operator_br)
        covariance = response @ self.raw_covariance() @ response.T
        covariance = 0.5 * (covariance + covariance.T)
        return covariance

    def entry_class_line_totals(
        self,
    ) -> dict[str, dict[str, tuple[float, float]]]:
        """Return per-line class means and sampling variances."""
        result: dict[str, dict[str, list[float]]] = {}
        for batch in self.batches:
            line = result.setdefault(
                batch.line_token,
                {
                    entry_class: [0.0, 0.0]
                    for entry_class in MEAN_CALIBRATION_ENTRY_CLASSES
                },
            )
            sample_count = float(batch.sampled_histories)
            correction = (
                batch.history_weight**2
                * sample_count
                / (sample_count - 1.0)
            )
            cluster_first = batch.cluster_first_items(
                bin_count=self.bin_count
            )
            cluster_outer = batch.cluster_outer_items(
                bin_count=self.bin_count
            )
            for entry_index, entry_class in enumerate(
                MEAN_CALIBRATION_ENTRY_CLASSES
            ):
                first_sum = sum(
                    value
                    for coordinate, value in cluster_first
                    if coordinate // self.bin_count == entry_index
                )
                second_sum = sum(
                    value * (2.0 if left != right else 1.0)
                    for left, right, value in cluster_outer
                    if (
                        left // self.bin_count == entry_index
                        and right // self.bin_count == entry_index
                    )
                )
                line[entry_class][0] += (
                    batch.history_weight * first_sum
                )
                centered = (
                    second_sum
                    - first_sum * first_sum / sample_count
                )
                line[entry_class][1] += correction * max(centered, 0.0)
        return {
            line_token: {
                entry_class: (float(values[0]), float(values[1]))
                for entry_class, values in classes.items()
            }
            for line_token, classes in result.items()
        }

    def entry_class_line_bin_variances(
        self,
    ) -> dict[str, dict[str, NDArray[np.float64]]]:
        """Return per-line class-coordinate sampling variances."""
        result: dict[str, dict[str, NDArray[np.float64]]] = {}
        for batch in self.batches:
            line = result.setdefault(
                batch.line_token,
                {
                    entry_class: np.zeros(
                        self.bin_count,
                        dtype=np.float64,
                    )
                    for entry_class in MEAN_CALIBRATION_ENTRY_CLASSES
                },
            )
            first = dict(
                batch.cluster_first_items(bin_count=self.bin_count)
            )
            diagonal = {
                left: value
                for left, right, value in batch.cluster_outer_items(
                    bin_count=self.bin_count
                )
                if left == right
            }
            sample_count = float(batch.sampled_histories)
            correction = (
                batch.history_weight**2
                * sample_count
                / (sample_count - 1.0)
            )
            for coordinate, first_sum in first.items():
                entry_index, bin_index = divmod(
                    coordinate,
                    self.bin_count,
                )
                second_sum = diagonal.get(coordinate, 0.0)
                centered = (
                    second_sum
                    - first_sum * first_sum / sample_count
                )
                line[
                    MEAN_CALIBRATION_ENTRY_CLASSES[entry_index]
                ][bin_index] += correction * max(centered, 0.0)
        return result

    def validate_native_arrays(
        self,
        spectrum: Sequence[float],
        spectrum_variance: Sequence[float],
    ) -> None:
        """Verify native dense arrays against the sparse sufficient statistics."""
        observed_mean = np.asarray(spectrum, dtype=np.float64)
        observed_variance = np.asarray(
            spectrum_variance,
            dtype=np.float64,
        )
        if (
            observed_mean.shape != (self.bin_count,)
            or observed_variance.shape != (self.bin_count,)
            or np.any(~np.isfinite(observed_mean))
            or np.any(~np.isfinite(observed_variance))
            or np.any(observed_mean < 0.0)
            or np.any(observed_variance < 0.0)
        ):
            raise ValueError("Native calibration arrays are invalid.")
        expected_mean = self.raw_mean()
        expected_variance = np.diag(self.raw_covariance())
        if not np.allclose(
            observed_mean,
            expected_mean,
            rtol=2.0e-10,
            atol=1.0e-9,
        ):
            raise ValueError(
                "Native calibration spectrum disagrees with sparse histories."
            )
        if not np.allclose(
            observed_variance,
            expected_variance,
            rtol=2.0e-10,
            atol=1.0e-8,
        ):
            raise ValueError(
                "Native calibration variance disagrees with sparse histories."
            )

    def to_payload(self) -> dict[str, object]:
        """Return a versioned JSON-compatible calibration artifact."""
        payload = {
            "schema_version": MEAN_CALIBRATION_SCHEMA_VERSION,
            "model": "geant4_fixed_source_line_stratified_mean_v1",
            "bin_count": self.bin_count,
            "histories_per_source_line": self.histories_per_source_line,
            "angle_strata_mu": self.angle_strata_mu,
            "angle_strata_phi": self.angle_strata_phi,
            "covariance_semantics": (
                "primary_history_stratified_sample_mean_sufficient_statistics"
            ),
            "batches": [batch.to_payload() for batch in self.batches],
        }
        if self.statistics_version == 2:
            payload["native_statistics_version"] = 2
            payload["covariance_semantics"] = (
                "original_history_branch_cluster_sample_mean_"
                "sufficient_statistics"
            )
        return payload

    def _validated_response_operator(
        self,
        value: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return a finite count-preserving detector-response operator."""
        response = np.asarray(value, dtype=np.float64)
        if (
            response.shape != (self.bin_count, self.bin_count)
            or np.any(~np.isfinite(response))
            or np.any(response < 0.0)
            or not np.allclose(
                np.sum(response, axis=0),
                1.0,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ):
            raise ValueError("Detector-response operator is invalid.")
        return response


def _require_close_sparse_values(
    observed: Mapping[object, float],
    expected: Mapping[object, float],
    *,
    field_name: str,
) -> None:
    """Require two sparse moment maps to agree including their support."""
    keys = set(observed) | set(expected)
    for key in keys:
        if not np.isclose(
            observed.get(key, 0.0),
            expected.get(key, 0.0),
            rtol=2.0e-12,
            atol=1.0e-14,
        ):
            raise ValueError(
                f"{field_name} disagrees with branch-cluster moments."
            )


def _validate_forced_collision_moments(
    *,
    bin_count: int,
    sampled_histories: int,
    cluster_first: tuple[tuple[int, float], ...],
    cluster_outer: tuple[tuple[int, int, float], ...],
    combined_first: tuple[tuple[int, float], ...],
    combined_outer: tuple[tuple[int, int, float], ...],
    class_first: Sequence[tuple[tuple[int, float], ...]],
) -> None:
    """Validate redundant v2 projections without assuming one-hot histories."""
    cluster_first_map = dict(cluster_first)
    cluster_outer_map = {
        (left, right): value
        for left, right, value in cluster_outer
    }
    for left, right in cluster_outer_map:
        if left not in cluster_first_map or right not in cluster_first_map:
            raise ValueError(
                "Branch-cluster second moments reference a zero first moment."
            )
    for coordinate, first_sum in cluster_first_map.items():
        second_sum = cluster_outer_map.get(
            (coordinate, coordinate),
            0.0,
        )
        centered = (
            second_sum
            - first_sum * first_sum / float(sampled_histories)
        )
        tolerance = 2.0e-12 * max(
            1.0,
            abs(second_sum)
            + first_sum * first_sum / float(sampled_histories),
        )
        if centered < -tolerance:
            raise ValueError(
                "Branch-cluster moments imply negative coordinate variance."
            )
    diagonal = {
        left: value
        for (left, right), value in cluster_outer_map.items()
        if left == right
    }
    for (left, right), cross in cluster_outer_map.items():
        cauchy_limit = diagonal.get(left, 0.0) * diagonal.get(right, 0.0)
        if cross * cross > cauchy_limit + 2.0e-12 * max(
            1.0,
            cross * cross,
            cauchy_limit,
        ):
            raise ValueError(
                "Branch-cluster second moments violate Cauchy bounds."
            )
    projected_class_first: list[dict[int, float]] = [
        {} for _ in MEAN_CALIBRATION_ENTRY_CLASSES
    ]
    projected_combined_first: dict[int, float] = {}
    for coordinate, value in cluster_first:
        entry_index, bin_index = divmod(coordinate, bin_count)
        projected_class_first[entry_index][bin_index] = (
            projected_class_first[entry_index].get(bin_index, 0.0) + value
        )
        projected_combined_first[bin_index] = (
            projected_combined_first.get(bin_index, 0.0) + value
        )
    for entry_index, observed_items in enumerate(class_first):
        _require_close_sparse_values(
            dict(observed_items),
            projected_class_first[entry_index],
            field_name=(
                "Class first sum "
                + MEAN_CALIBRATION_ENTRY_CLASSES[entry_index]
            ),
        )
        first_total = sum(projected_class_first[entry_index].values())
        second_total = sum(
            value * (2.0 if left != right else 1.0)
            for (left, right), value in cluster_outer_map.items()
            if (
                left // bin_count == entry_index
                and right // bin_count == entry_index
            )
        )
        centered = (
            second_total
            - first_total * first_total / float(sampled_histories)
        )
        tolerance = 2.0e-12 * max(
            1.0,
            abs(second_total)
            + first_total * first_total / float(sampled_histories),
        )
        if centered < -tolerance:
            raise ValueError(
                "Branch-cluster moments imply negative class variance."
            )
    _require_close_sparse_values(
        dict(combined_first),
        projected_combined_first,
        field_name="Combined-bin first sum",
    )

    projected_combined_outer: dict[tuple[int, int], float] = {}
    for (left, right), value in cluster_outer_map.items():
        left_bin = left % bin_count
        right_bin = right % bin_count
        pair = tuple(sorted((left_bin, right_bin)))
        factor = 2.0 if left != right and left_bin == right_bin else 1.0
        projected_combined_outer[pair] = (
            projected_combined_outer.get(pair, 0.0) + factor * value
        )
    _require_close_sparse_values(
        {
            (left, right): value
            for left, right, value in combined_outer
        },
        projected_combined_outer,
        field_name="Combined-bin sum outer",
    )

    combined_first_map = dict(combined_first)
    combined_outer_map = {
        (left, right): value
        for left, right, value in combined_outer
    }
    for bin_index, first_sum in combined_first_map.items():
        second_sum = combined_outer_map.get(
            (bin_index, bin_index),
            0.0,
        )
        centered = (
            second_sum
            - first_sum * first_sum / float(sampled_histories)
        )
        tolerance = 2.0e-12 * max(
            1.0,
            abs(second_sum)
            + first_sum * first_sum / float(sampled_histories),
        )
        if centered < -tolerance:
            raise ValueError(
                "Combined branch-cluster moments imply negative variance."
            )


def parse_mean_calibration_metadata(
    metadata: Mapping[str, Any],
    *,
    bin_count: int,
) -> StratifiedMeanCalibration:
    """Parse and authenticate native fixed-quota calibration metadata."""
    if not isinstance(metadata, Mapping):
        raise TypeError("Mean-calibration metadata must be a mapping.")
    if bin_count <= 0:
        raise ValueError("bin_count must be positive.")
    forced_collision = metadata.get("mean_calibration_forced_collision")
    if not isinstance(forced_collision, bool):
        raise TypeError(
            "mean_calibration_forced_collision must be a JSON boolean."
        )
    expected_covariance_semantics = (
        MEAN_CALIBRATION_FORCED_COLLISION_COVARIANCE_SEMANTICS
        if forced_collision
        else MEAN_CALIBRATION_ANALOG_COVARIANCE_SEMANTICS
    )
    required_exact = {
        "mean_calibration_enabled": True,
        "primary_schedule_mode": (
            "fixed_source_line_stratified_mean_calibration"
        ),
        "transport_history_mode": (
            "fixed_source_line_stratified_weighted_mean"
        ),
        "transport_tally_weighted": True,
        "history_thinning_enabled": False,
        "mean_calibration_forced_collision": forced_collision,
        "mean_calibration_history_weight_semantics": (
            "expected_source_line_mean_divided_by_fixed_quota"
        ),
        "mean_calibration_covariance_semantics": (
            expected_covariance_semantics
        ),
        "spectrum_variance_semantics": (
            "stratified_fixed_quota_sample_mean_covariance"
        ),
    }
    for key, expected in required_exact.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Native mean calibration has invalid {key}: "
                f"{metadata.get(key)!r}."
            )
    native_bin_count = _strict_integer(
        metadata.get("spectrum_bin_count"),
        field_name="spectrum_bin_count",
    )
    if native_bin_count != bin_count:
        raise ValueError("Native calibration bin count is inconsistent.")
    quota = _strict_integer(
        metadata.get("mean_calibration_histories_per_source_line"),
        field_name="mean_calibration_histories_per_source_line",
    )
    mu_count = _strict_integer(
        metadata.get("mean_calibration_angle_strata_mu"),
        field_name="mean_calibration_angle_strata_mu",
    )
    phi_count = _strict_integer(
        metadata.get("mean_calibration_angle_strata_phi"),
        field_name="mean_calibration_angle_strata_phi",
    )
    stratum_count = _strict_integer(
        metadata.get("mean_calibration_angle_stratum_count"),
        field_name="mean_calibration_angle_stratum_count",
    )
    batch_count = _strict_integer(
        metadata.get("primary_history_batch_count"),
        field_name="primary_history_batch_count",
    )
    if (
        quota <= 0
        or mu_count <= 0
        or phi_count <= 0
        or stratum_count != mu_count * phi_count
        or quota % stratum_count != 0
        or quota // stratum_count < 2
        or batch_count <= 0
        or batch_count % stratum_count != 0
    ):
        raise ValueError("Native mean-calibration quota/strata are invalid.")
    batches: list[MeanCalibrationBatch] = []
    for batch_index in range(batch_count):
        prefix = f"mean_calibration_batch_{batch_index}_"
        source_token = metadata.get(prefix + "source_token")
        line_token = metadata.get(prefix + "line_token")
        if (
            not isinstance(source_token, str)
            or not source_token
            or not isinstance(line_token, str)
            or not line_token
        ):
            raise TypeError("Native calibration tokens must be nonempty strings.")
        sampled_histories = _strict_integer(
            metadata.get(prefix + "sampled_histories"),
            field_name=prefix + "sampled_histories",
        )
        angle_stratum_index = _strict_integer(
            metadata.get(prefix + "angle_stratum_index"),
            field_name=prefix + "angle_stratum_index",
        )
        expected = _strict_float(
            metadata.get(prefix + "expected_unthinned_histories"),
            field_name=prefix + "expected_unthinned_histories",
        )
        history_weight = _strict_float(
            metadata.get(prefix + "history_weight"),
            field_name=prefix + "history_weight",
        )
        if forced_collision:
            if any(
                prefix + "sparse_entry_histogram_" + entry_class
                in metadata
                for entry_class in MEAN_CALIBRATION_ENTRY_CLASSES
            ):
                raise ValueError(
                    "Forced-collision metadata mixes v1 histogram semantics."
                )
            if (
                metadata.get(prefix + "cluster_coordinate_semantics")
                != MEAN_CALIBRATION_CLUSTER_COORDINATE_SEMANTICS
                or metadata.get(prefix + "cluster_score_semantics")
                != MEAN_CALIBRATION_CLUSTER_SCORE_SEMANTICS
            ):
                raise ValueError(
                    "Forced-collision branch-cluster semantics are invalid."
                )
            cluster_first = _parse_sparse_first_sum(
                metadata.get(prefix + "sparse_cluster_first_sum"),
                field_name=prefix + "sparse_cluster_first_sum",
                coordinate_count=3 * bin_count,
            )
            cluster_outer = _parse_sparse_sum_outer(
                metadata.get(prefix + "sparse_cluster_sum_outer"),
                field_name=prefix + "sparse_cluster_sum_outer",
                coordinate_count=3 * bin_count,
            )
            combined_first = _parse_sparse_first_sum(
                metadata.get(
                    prefix + "sparse_combined_bin_first_sum"
                ),
                field_name=prefix + "sparse_combined_bin_first_sum",
                coordinate_count=bin_count,
            )
            combined_outer = _parse_sparse_sum_outer(
                metadata.get(
                    prefix + "sparse_combined_bin_sum_outer"
                ),
                field_name=prefix + "sparse_combined_bin_sum_outer",
                coordinate_count=bin_count,
            )
            class_first = tuple(
                _parse_sparse_first_sum(
                    metadata.get(
                        prefix
                        + "sparse_cluster_first_sum_"
                        + entry_class
                    ),
                    field_name=(
                        prefix
                        + "sparse_cluster_first_sum_"
                        + entry_class
                    ),
                    coordinate_count=bin_count,
                )
                for entry_class in MEAN_CALIBRATION_ENTRY_CLASSES
            )
            _validate_forced_collision_moments(
                bin_count=bin_count,
                sampled_histories=sampled_histories,
                cluster_first=cluster_first,
                cluster_outer=cluster_outer,
                combined_first=combined_first,
                combined_outer=combined_outer,
                class_first=class_first,
            )
            histograms: tuple[
                tuple[tuple[int, int], ...],
                ...,
            ] = tuple(() for _ in MEAN_CALIBRATION_ENTRY_CLASSES)
            statistics_version = 2
        else:
            if any(
                isinstance(key, str)
                and (
                    key.startswith(prefix + "sparse_cluster_")
                    or key.startswith(prefix + "sparse_combined_")
                    or key
                    in {
                        prefix + "cluster_coordinate_semantics",
                        prefix + "cluster_score_semantics",
                    }
                )
                for key in metadata
            ):
                raise ValueError(
                    "Analog metadata mixes v2 branch-cluster semantics."
                )
            histograms = tuple(
                _parse_sparse_histogram(
                    metadata.get(
                        prefix + "sparse_entry_histogram_" + entry_class
                    ),
                    field_name=(
                        prefix + "sparse_entry_histogram_" + entry_class
                    ),
                    bin_count=bin_count,
                )
                for entry_class in MEAN_CALIBRATION_ENTRY_CLASSES
            )
            cluster_first = ()
            cluster_outer = ()
            combined_first = ()
            combined_outer = ()
            statistics_version = 1
        batch = MeanCalibrationBatch(
            source_token=source_token,
            line_token=line_token,
            expected_unthinned_histories=expected,
            sampled_histories=sampled_histories,
            history_weight=history_weight,
            angle_stratum_index=angle_stratum_index,
            statistics_version=statistics_version,
            entry_histograms=histograms,
            cluster_first_sum=cluster_first,
            cluster_sum_outer=cluster_outer,
            combined_bin_first_sum=combined_first,
            combined_bin_sum_outer=combined_outer,
        )
        if (
            statistics_version == 1
            and batch.entry_count() > sampled_histories
        ):
            raise ValueError(
                "Native calibration histogram exceeds its history count."
            )
        if (
            expected <= 0.0
            or history_weight <= 0.0
            or sampled_histories != quota // stratum_count
            or not 0 <= angle_stratum_index < stratum_count
            or not np.isclose(
                history_weight * sampled_histories,
                expected,
                rtol=2.0e-12,
                atol=1.0e-12,
            )
        ):
            raise ValueError("Native calibration batch contract is invalid.")
        batches.append(batch)
    grouped: dict[str, list[MeanCalibrationBatch]] = {}
    for batch in batches:
        grouped.setdefault(batch.line_token, []).append(batch)
    for line_token, group in grouped.items():
        stratum_indices = [
            batch.angle_stratum_index for batch in group
        ]
        if len(set(stratum_indices)) != len(stratum_indices):
            raise ValueError(
                "Native mean calibration duplicated an angular stratum."
            )
        if (
            len(group) != stratum_count
            or set(stratum_indices)
            != set(range(stratum_count))
            or len({batch.source_token for batch in group}) != 1
            or any(
                not batch.line_token.startswith(
                    batch.source_token + "_"
                )
                for batch in group
            )
            or len(
                {
                    round(
                        batch.expected_unthinned_histories * stratum_count,
                        12,
                    )
                    for batch in group
                }
            )
            != 1
        ):
            raise ValueError(
                f"Native calibration line group is incomplete: {line_token}."
            )
    return StratifiedMeanCalibration(
        bin_count=bin_count,
        histories_per_source_line=quota,
        angle_strata_mu=mu_count,
        angle_strata_phi=phi_count,
        batches=tuple(batches),
    )


__all__ = [
    "MEAN_CALIBRATION_ANALOG_COVARIANCE_SEMANTICS",
    "MEAN_CALIBRATION_CLUSTER_COORDINATE_SEMANTICS",
    "MEAN_CALIBRATION_CLUSTER_SCORE_SEMANTICS",
    "MEAN_CALIBRATION_ENTRY_CLASSES",
    "MEAN_CALIBRATION_FORCED_COLLISION_COVARIANCE_SEMANTICS",
    "MEAN_CALIBRATION_SCHEMA_VERSION",
    "MeanCalibrationBatch",
    "StratifiedMeanCalibration",
    "parse_mean_calibration_metadata",
]
