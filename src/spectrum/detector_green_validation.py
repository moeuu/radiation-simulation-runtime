"""Independent monoenergetic validation for a detector Green operator."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Final

import numpy as np

from spectrum.detector_green_construction import gaussian_resolution_operator
from spectrum.detector_green_operator import (
    DetectorGreenOperator,
    canonical_json_bytes,
)
from spectrum.library import default_library
from spectrum.response_matrix import (
    NATIVE_GEANT4_BIN_COUNT,
    NATIVE_GEANT4_BIN_WIDTH_KEV,
)


DETECTOR_GREEN_VALIDATION_SCHEMA_VERSION: Final = 2
DETECTOR_GREEN_VALIDATION_RAW_SCHEMA_VERSION: Final = 2
DETECTOR_GREEN_VALIDATION_ID: Final = (
    "catalog_independent_monoenergetic_detector_green_holdout_v2"
)
DETECTOR_GREEN_VALIDATION_RAW_BASENAME: Final = "raw_corpus.json"
DETECTOR_GREEN_VALIDATION_MANIFEST_BASENAME: Final = "manifest.json"
DETECTOR_GREEN_HOLDOUT_ENERGY_COUNT: Final = 16
DETECTOR_GREEN_HOLDOUT_MINIMUM_KEV: Final = 20.0
DETECTOR_GREEN_HOLDOUT_MAXIMUM_KEV: Final = 1680.0
DETECTOR_GREEN_HOLDOUT_EXCLUSION_KEV: Final = 0.5
DETECTOR_GREEN_VALIDATION_MINIMUM_HISTORIES_PER_ENERGY: Final = 100_000
DETECTOR_GREEN_VALIDATION_MINIMUM_PULSES_PER_CELL: Final = 100
DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT: Final = {
    "maximum_grouped_total_variation": ("le", 0.20),
    "maximum_cdf_distance": ("le", 0.15),
    "maximum_conditional_mean_scaled_error": ("le", 0.08),
    "maximum_detection_probability_absolute_error": ("le", 0.08),
}


def _is_sha256(value: object) -> bool:
    """Return whether a value is one lowercase SHA-256 digest."""
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: object, *, field_name: str) -> float:
    """Return one finite non-boolean JSON number."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be a finite JSON number.")
    return float(value)


def _positive_integer(value: object, *, field_name: str) -> int:
    """Return one positive non-boolean JSON integer."""
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive JSON integer.")
    return int(value)


def _forbidden_energies_keV(
    operator: DetectorGreenOperator,
) -> np.ndarray:
    """Return construction and catalog energies excluded from holdout design."""
    catalog = [
        float(line.energy_keV)
        for nuclide in default_library().values()
        for line in nuclide.lines
    ]
    return np.unique(
        np.concatenate(
            (
                operator.energy_nodes_keV,
                np.asarray(catalog, dtype=np.float64),
            )
        )
    )


def detector_green_holdout_energies_keV(
    design_seed: int,
    *,
    operator: DetectorGreenOperator,
) -> np.ndarray:
    """Return deterministic stratified energies excluded from model design."""
    seed = _positive_integer(design_seed, field_name="design_seed")
    if seed >= 2_147_483_647:
        raise ValueError("design_seed must fit one positive signed 32-bit integer.")
    forbidden = _forbidden_energies_keV(operator)
    edges = np.linspace(
        DETECTOR_GREEN_HOLDOUT_MINIMUM_KEV,
        DETECTOR_GREEN_HOLDOUT_MAXIMUM_KEV,
        DETECTOR_GREEN_HOLDOUT_ENERGY_COUNT + 1,
        dtype=np.float64,
    )
    generator = np.random.Generator(np.random.Philox(seed))
    selected: list[float] = []
    for index in range(DETECTOR_GREEN_HOLDOUT_ENERGY_COUNT):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        value = math.nan
        for _ in range(1_024):
            candidate = float(generator.uniform(lower, upper))
            if np.min(np.abs(forbidden - candidate)) > (
                DETECTOR_GREEN_HOLDOUT_EXCLUSION_KEV
            ):
                value = candidate
                break
        if not math.isfinite(value):
            raise RuntimeError("Could not construct an independent holdout energy.")
        selected.append(value)
    result = np.asarray(sorted(selected), dtype=np.float64)
    if (
        np.any(np.diff(result) <= 0.0)
        or result[0] < DETECTOR_GREEN_HOLDOUT_MINIMUM_KEV
        or result[-1] > DETECTOR_GREEN_HOLDOUT_MAXIMUM_KEV
    ):
        raise RuntimeError("Detector Green holdout design is invalid.")
    return result


def detector_green_validation_contract_sha256() -> str:
    """Return the immutable generic validation protocol digest."""
    payload = {
        "schema_version": DETECTOR_GREEN_VALIDATION_SCHEMA_VERSION,
        "validation": DETECTOR_GREEN_VALIDATION_ID,
        "energy_design": "philox_stratified_continuous_domain_v1",
        "energy_count": DETECTOR_GREEN_HOLDOUT_ENERGY_COUNT,
        "energy_domain_keV": [
            DETECTOR_GREEN_HOLDOUT_MINIMUM_KEV,
            DETECTOR_GREEN_HOLDOUT_MAXIMUM_KEV,
        ],
        "catalog_and_construction_exclusion_keV": (
            DETECTOR_GREEN_HOLDOUT_EXCLUSION_KEV
        ),
        "minimum_histories_per_energy": (
            DETECTOR_GREEN_VALIDATION_MINIMUM_HISTORIES_PER_ENERGY
        ),
        "minimum_pulses_per_cell": (DETECTOR_GREEN_VALIDATION_MINIMUM_PULSES_PER_CELL),
        "group_width_keV": 20.0,
        "metrics": {
            name: {"comparison": comparison, "threshold": threshold}
            for name, (comparison, threshold) in (
                DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT.items()
            )
        },
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


DETECTOR_GREEN_VALIDATION_CONTRACT_SHA256: Final = (
    detector_green_validation_contract_sha256()
)


def validate_detector_green_validation_raw_corpus(
    payload: object,
    *,
    operator: DetectorGreenOperator,
) -> dict[str, object]:
    """Validate a strict isotope-free full-detector holdout corpus."""
    if not isinstance(payload, Mapping):
        raise TypeError("Detector Green validation corpus must be an object.")
    expected_keys = {
        "schema_version",
        "validation",
        "validation_contract_sha256",
        "operator_contract_sha256",
        "operator_binary_sha256",
        "operator_construction_seed",
        "design_seed",
        "transport_seed",
        "histories_per_energy",
        "impact_parameter_edges_fraction",
        "holdout_energies_keV",
        "output_energy_axis",
        "cells",
        "runtime_config_sha256",
        "native_executable_sha256",
        "native_execution_environment_sha256",
        "detector_implementation_bundle_sha256",
        "detector_model_sha256",
        "geant4_physics_contract_sha256",
        "energy_resolution_contract_sha256",
        "reference_scene_sha256",
        "reference_source_contract_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("Detector Green validation corpus schema is incompatible.")
    construction = operator.construction
    if construction is None:
        raise RuntimeError("Detector Green operator has no construction provenance.")
    if (
        payload["schema_version"] != DETECTOR_GREEN_VALIDATION_RAW_SCHEMA_VERSION
        or payload["validation"] != DETECTOR_GREEN_VALIDATION_ID
        or payload["validation_contract_sha256"]
        != DETECTOR_GREEN_VALIDATION_CONTRACT_SHA256
        or payload["operator_contract_sha256"] != operator.contract_hash_sha256
        or payload["operator_binary_sha256"] != operator.binary_sha256
        or payload["operator_construction_seed"] != construction["construction_seed"]
    ):
        raise ValueError("Detector Green validation corpus identity is stale.")
    for field_name in (
        "runtime_config_sha256",
        "native_executable_sha256",
        "native_execution_environment_sha256",
        "detector_implementation_bundle_sha256",
        "detector_model_sha256",
        "geant4_physics_contract_sha256",
        "energy_resolution_contract_sha256",
        "reference_scene_sha256",
        "reference_source_contract_sha256",
    ):
        if not _is_sha256(payload[field_name]):
            raise ValueError(f"Detector Green validation {field_name} is invalid.")
    for field_name in (
        "native_executable_sha256",
        "native_execution_environment_sha256",
        "detector_implementation_bundle_sha256",
        "detector_model_sha256",
        "geant4_physics_contract_sha256",
        "energy_resolution_contract_sha256",
    ):
        if payload[field_name] != construction[field_name]:
            raise ValueError(
                f"Detector Green validation {field_name} differs from construction."
            )
    design_seed = _positive_integer(payload["design_seed"], field_name="design_seed")
    transport_seed = _positive_integer(
        payload["transport_seed"],
        field_name="transport_seed",
    )
    histories = _positive_integer(
        payload["histories_per_energy"],
        field_name="histories_per_energy",
    )
    if (
        histories < DETECTOR_GREEN_VALIDATION_MINIMUM_HISTORIES_PER_ENERGY
        or len({design_seed, transport_seed, int(construction["construction_seed"])})
        != 3
    ):
        raise ValueError("Detector Green validation histories or seeds are invalid.")
    expected_energies = detector_green_holdout_energies_keV(
        design_seed,
        operator=operator,
    )
    energies = np.asarray(payload["holdout_energies_keV"], dtype=np.float64)
    edges = np.asarray(
        payload["impact_parameter_edges_fraction"],
        dtype=np.float64,
    )
    if (
        not np.array_equal(energies, expected_energies)
        or not np.array_equal(
            edges,
            operator.impact_parameter_edges_fraction,
        )
        or histories % (edges.size - 1) != 0
    ):
        raise ValueError("Detector Green validation energy or phase design drifted.")
    output_axis = payload["output_energy_axis"]
    if output_axis != {
        "minimum_keV": 0.0,
        "bin_width_keV": NATIVE_GEANT4_BIN_WIDTH_KEV,
        "bin_count": NATIVE_GEANT4_BIN_COUNT,
    }:
        raise ValueError("Detector Green validation output axis is incompatible.")
    cells = payload["cells"]
    expected_cell_count = energies.size * (edges.size - 1)
    if not isinstance(cells, list) or len(cells) != expected_cell_count:
        raise ValueError("Detector Green validation cells are incomplete.")
    cell_keys = {
        "energy_index",
        "impact_bin_index",
        "sampled_histories",
        "registered_pulses",
        "sparse_raw_deposit_histogram",
    }
    per_cell_histories = histories // (edges.size - 1)
    for flat_index, cell in enumerate(cells):
        energy_index = flat_index // (edges.size - 1)
        impact_index = flat_index % (edges.size - 1)
        if (
            not isinstance(cell, Mapping)
            or set(cell) != cell_keys
            or cell["energy_index"] != energy_index
            or cell["impact_bin_index"] != impact_index
            or cell["sampled_histories"] != per_cell_histories
            or not isinstance(cell["registered_pulses"], int)
            or cell["registered_pulses"]
            < DETECTOR_GREEN_VALIDATION_MINIMUM_PULSES_PER_CELL
            or cell["registered_pulses"] > per_cell_histories
        ):
            raise ValueError("Detector Green validation cell metadata is invalid.")
        sparse = cell["sparse_raw_deposit_histogram"]
        if not isinstance(sparse, list):
            raise ValueError("Detector Green validation histogram is invalid.")
        previous = -1
        total = 0
        for entry in sparse:
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or not all(isinstance(value, int) for value in entry)
                or entry[0] <= previous
                or entry[0] < 0
                or entry[0] >= NATIVE_GEANT4_BIN_COUNT
                or entry[1] <= 0
            ):
                raise ValueError("Detector Green sparse histogram is malformed.")
            previous = entry[0]
            total += entry[1]
        if total != cell["registered_pulses"]:
            raise ValueError("Detector Green sparse histogram total is stale.")
    return json.loads(json.dumps(dict(payload), allow_nan=False))


def _dense_raw_histograms(
    corpus: Mapping[str, object],
) -> np.ndarray:
    """Materialize validated sparse holdout cells as one dense tensor."""
    energies = corpus["holdout_energies_keV"]
    edges = corpus["impact_parameter_edges_fraction"]
    result = np.zeros(
        (len(energies), len(edges) - 1, NATIVE_GEANT4_BIN_COUNT),
        dtype=np.float64,
    )
    for cell in corpus["cells"]:
        for bin_index, count in cell["sparse_raw_deposit_histogram"]:
            result[
                int(cell["energy_index"]),
                int(cell["impact_bin_index"]),
                int(bin_index),
            ] = float(count)
    return result


def evaluate_detector_green_validation_raw_corpus(
    payload: object,
    *,
    operator: DetectorGreenOperator,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]], bool]:
    """Recompute cell metrics from immutable full-detector evidence."""
    corpus = validate_detector_green_validation_raw_corpus(
        payload,
        operator=operator,
    )
    energies = np.asarray(corpus["holdout_energies_keV"], dtype=np.float64)
    raw = _dense_raw_histograms(corpus)
    resolution = gaussian_resolution_operator(
        output_bin_count=NATIVE_GEANT4_BIN_COUNT,
        output_bin_width_keV=NATIVE_GEANT4_BIN_WIDTH_KEV,
    )
    observed_counts = np.einsum("br,ecr->ecb", resolution, raw, optimize=True)
    observed = observed_counts / np.sum(
        observed_counts,
        axis=-1,
        keepdims=True,
    )
    candidate_cbe, _ = operator.phase_response_for_axis(energies)
    candidate = np.transpose(candidate_cbe, (2, 0, 1))
    candidate_detection_ce = operator.phase_detection_probability_for_axis(energies)
    candidate_detection = candidate_detection_ce.T
    histories_per_cell = int(corpus["histories_per_energy"]) // raw.shape[1]
    observed_detection = np.sum(raw, axis=-1) / float(histories_per_cell)
    group_size = int(round(20.0 / NATIVE_GEANT4_BIN_WIDTH_KEV))
    pad = (-NATIVE_GEANT4_BIN_COUNT) % group_size
    observed_grouped = (
        np.pad(observed, ((0, 0), (0, 0), (0, pad)))
        .reshape(
            observed.shape[0],
            observed.shape[1],
            -1,
            group_size,
        )
        .sum(axis=-1)
    )
    candidate_grouped = (
        np.pad(candidate, ((0, 0), (0, 0), (0, pad)))
        .reshape(
            candidate.shape[0],
            candidate.shape[1],
            -1,
            group_size,
        )
        .sum(axis=-1)
    )
    grouped_tv = 0.5 * np.sum(
        np.abs(observed_grouped - candidate_grouped),
        axis=-1,
    )
    cdf_distance = np.max(
        np.abs(np.cumsum(observed, axis=-1) - np.cumsum(candidate, axis=-1)),
        axis=-1,
    )
    axis = np.arange(NATIVE_GEANT4_BIN_COUNT, dtype=np.float64) * (
        NATIVE_GEANT4_BIN_WIDTH_KEV
    )
    observed_mean = np.einsum("ecb,b->ec", observed, axis, optimize=True)
    candidate_mean = np.einsum("ecb,b->ec", candidate, axis, optimize=True)
    scaled_mean_error = np.abs(observed_mean - candidate_mean) / np.maximum(
        energies[:, np.newaxis],
        50.0,
    )
    detection_error = np.abs(observed_detection - candidate_detection)
    cell_results: list[dict[str, object]] = []
    for energy_index, energy in enumerate(energies):
        for impact_index in range(raw.shape[1]):
            values = {
                "grouped_total_variation": float(
                    grouped_tv[energy_index, impact_index]
                ),
                "cdf_distance": float(cdf_distance[energy_index, impact_index]),
                "conditional_mean_scaled_error": float(
                    scaled_mean_error[energy_index, impact_index]
                ),
                "detection_probability_absolute_error": float(
                    detection_error[energy_index, impact_index]
                ),
            }
            passed = bool(
                values["grouped_total_variation"]
                <= DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT[
                    "maximum_grouped_total_variation"
                ][1]
                and values["cdf_distance"]
                <= DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT["maximum_cdf_distance"][1]
                and values["conditional_mean_scaled_error"]
                <= DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT[
                    "maximum_conditional_mean_scaled_error"
                ][1]
                and values["detection_probability_absolute_error"]
                <= DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT[
                    "maximum_detection_probability_absolute_error"
                ][1]
            )
            cell_results.append(
                {
                    "energy_index": energy_index,
                    "impact_bin_index": impact_index,
                    "energy_keV": float(energy),
                    **values,
                    "observed_conditional_mean_keV": float(
                        observed_mean[energy_index, impact_index]
                    ),
                    "candidate_conditional_mean_keV": float(
                        candidate_mean[energy_index, impact_index]
                    ),
                    "observed_detection_probability": float(
                        observed_detection[energy_index, impact_index]
                    ),
                    "candidate_detection_probability": float(
                        candidate_detection[energy_index, impact_index]
                    ),
                    "passed": passed,
                }
            )
    aggregate_values = {
        "maximum_grouped_total_variation": float(np.max(grouped_tv)),
        "maximum_cdf_distance": float(np.max(cdf_distance)),
        "maximum_conditional_mean_scaled_error": float(np.max(scaled_mean_error)),
        "maximum_detection_probability_absolute_error": float(np.max(detection_error)),
    }
    metrics = {
        name: {
            "value": aggregate_values[name],
            "comparison": comparison,
            "threshold": float(threshold),
            "passed": bool(aggregate_values[name] <= threshold),
        }
        for name, (comparison, threshold) in (
            DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT.items()
        )
    }
    all_passed = bool(
        all(result["passed"] is True for result in cell_results)
        and all(result["passed"] is True for result in metrics.values())
    )
    return cell_results, metrics, all_passed


def build_detector_green_validation_manifest(
    raw_corpus: object,
    *,
    operator: DetectorGreenOperator,
) -> dict[str, object]:
    """Build a derived generic validation manifest from raw evidence."""
    corpus = validate_detector_green_validation_raw_corpus(
        raw_corpus,
        operator=operator,
    )
    cell_results, metrics, all_passed = evaluate_detector_green_validation_raw_corpus(
        corpus,
        operator=operator,
    )
    return {
        "schema_version": DETECTOR_GREEN_VALIDATION_SCHEMA_VERSION,
        "validation": DETECTOR_GREEN_VALIDATION_ID,
        "validation_contract_sha256": (DETECTOR_GREEN_VALIDATION_CONTRACT_SHA256),
        "operator_contract_sha256": operator.contract_hash_sha256,
        "operator_binary_sha256": operator.binary_sha256,
        "raw_corpus_sha256": hashlib.sha256(canonical_json_bytes(corpus)).hexdigest(),
        "design_seed": corpus["design_seed"],
        "transport_seed": corpus["transport_seed"],
        "holdout_energies_keV": corpus["holdout_energies_keV"],
        "impact_parameter_edges_fraction": corpus["impact_parameter_edges_fraction"],
        "histories_per_energy": corpus["histories_per_energy"],
        "runtime_config_sha256": corpus["runtime_config_sha256"],
        "native_executable_sha256": corpus["native_executable_sha256"],
        "native_execution_environment_sha256": corpus[
            "native_execution_environment_sha256"
        ],
        "detector_implementation_bundle_sha256": corpus[
            "detector_implementation_bundle_sha256"
        ],
        "cell_results": cell_results,
        "metrics": metrics,
        "all_passed": all_passed,
    }


def validate_detector_green_validation_manifest(
    payload: object,
    *,
    operator: DetectorGreenOperator,
    require_passed: bool = True,
) -> dict[str, object]:
    """Validate one derived generic operator-validation manifest."""
    if not isinstance(payload, Mapping):
        raise TypeError("Detector Green validation manifest must be an object.")
    expected_keys = {
        "schema_version",
        "validation",
        "validation_contract_sha256",
        "operator_contract_sha256",
        "operator_binary_sha256",
        "raw_corpus_sha256",
        "design_seed",
        "transport_seed",
        "holdout_energies_keV",
        "impact_parameter_edges_fraction",
        "histories_per_energy",
        "runtime_config_sha256",
        "native_executable_sha256",
        "native_execution_environment_sha256",
        "detector_implementation_bundle_sha256",
        "cell_results",
        "metrics",
        "all_passed",
    }
    if set(payload) != expected_keys:
        raise ValueError("Detector Green validation manifest schema is incompatible.")
    if (
        payload["schema_version"] != DETECTOR_GREEN_VALIDATION_SCHEMA_VERSION
        or payload["validation"] != DETECTOR_GREEN_VALIDATION_ID
        or payload["validation_contract_sha256"]
        != DETECTOR_GREEN_VALIDATION_CONTRACT_SHA256
        or payload["operator_contract_sha256"] != operator.contract_hash_sha256
        or payload["operator_binary_sha256"] != operator.binary_sha256
        or not isinstance(payload["all_passed"], bool)
    ):
        raise ValueError("Detector Green validation manifest identity is stale.")
    for field_name in (
        "raw_corpus_sha256",
        "runtime_config_sha256",
        "native_executable_sha256",
        "native_execution_environment_sha256",
        "detector_implementation_bundle_sha256",
    ):
        if not _is_sha256(payload[field_name]):
            raise ValueError(f"Detector Green validation {field_name} is invalid.")
    construction = operator.construction
    if construction is None:
        raise RuntimeError("Detector Green operator has no construction provenance.")
    for field_name in (
        "native_executable_sha256",
        "native_execution_environment_sha256",
        "detector_implementation_bundle_sha256",
    ):
        if payload[field_name] != construction[field_name]:
            raise ValueError(
                f"Detector Green validation {field_name} differs from construction."
            )
    expected_energies = detector_green_holdout_energies_keV(
        int(payload["design_seed"]),
        operator=operator,
    )
    if not np.array_equal(
        np.asarray(payload["holdout_energies_keV"], dtype=np.float64),
        expected_energies,
    ) or not np.array_equal(
        np.asarray(
            payload["impact_parameter_edges_fraction"],
            dtype=np.float64,
        ),
        operator.impact_parameter_edges_fraction,
    ):
        raise ValueError("Detector Green validation holdout design is stale.")
    _positive_integer(payload["transport_seed"], field_name="transport_seed")
    histories = _positive_integer(
        payload["histories_per_energy"],
        field_name="histories_per_energy",
    )
    if histories < DETECTOR_GREEN_VALIDATION_MINIMUM_HISTORIES_PER_ENERGY:
        raise ValueError("Detector Green validation histories are insufficient.")
    cell_results = payload["cell_results"]
    expected_count = DETECTOR_GREEN_HOLDOUT_ENERGY_COUNT * (
        operator.impact_parameter_edges_fraction.size - 1
    )
    if not isinstance(cell_results, list) or len(cell_results) != expected_count:
        raise ValueError("Detector Green validation cell results are incomplete.")
    cell_keys = {
        "energy_index",
        "impact_bin_index",
        "energy_keV",
        "grouped_total_variation",
        "cdf_distance",
        "conditional_mean_scaled_error",
        "detection_probability_absolute_error",
        "observed_conditional_mean_keV",
        "candidate_conditional_mean_keV",
        "observed_detection_probability",
        "candidate_detection_probability",
        "passed",
    }
    impact_count = operator.impact_parameter_edges_fraction.size - 1
    for flat_index, result in enumerate(cell_results):
        energy_index = flat_index // impact_count
        impact_index = flat_index % impact_count
        if not isinstance(result, Mapping) or set(result) != cell_keys:
            raise ValueError("Detector Green validation cell result is malformed.")
        if (
            result["energy_index"] != energy_index
            or result["impact_bin_index"] != impact_index
            or _finite_number(
                result["energy_keV"],
                field_name="cell_results.energy_keV",
            )
            != float(expected_energies[energy_index])
            or not isinstance(result["passed"], bool)
        ):
            raise ValueError("Detector Green validation cell identity is stale.")
        grouped_tv = _finite_number(
            result["grouped_total_variation"],
            field_name="cell_results.grouped_total_variation",
        )
        cdf_distance = _finite_number(
            result["cdf_distance"],
            field_name="cell_results.cdf_distance",
        )
        mean_error = _finite_number(
            result["conditional_mean_scaled_error"],
            field_name="cell_results.conditional_mean_scaled_error",
        )
        detection_error = _finite_number(
            result["detection_probability_absolute_error"],
            field_name=("cell_results.detection_probability_absolute_error"),
        )
        observed_mean = _finite_number(
            result["observed_conditional_mean_keV"],
            field_name="cell_results.observed_conditional_mean_keV",
        )
        candidate_mean = _finite_number(
            result["candidate_conditional_mean_keV"],
            field_name="cell_results.candidate_conditional_mean_keV",
        )
        observed_detection = _finite_number(
            result["observed_detection_probability"],
            field_name="cell_results.observed_detection_probability",
        )
        candidate_detection = _finite_number(
            result["candidate_detection_probability"],
            field_name="cell_results.candidate_detection_probability",
        )
        expected_cell_pass = bool(
            grouped_tv
            <= DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT[
                "maximum_grouped_total_variation"
            ][1]
            and cdf_distance
            <= DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT["maximum_cdf_distance"][1]
            and mean_error
            <= DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT[
                "maximum_conditional_mean_scaled_error"
            ][1]
            and detection_error
            <= DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT[
                "maximum_detection_probability_absolute_error"
            ][1]
        )
        if (
            min(grouped_tv, cdf_distance, mean_error, detection_error) < 0.0
            or min(observed_mean, candidate_mean) < 0.0
            or not 0.0 <= observed_detection <= 1.0
            or not 0.0 <= candidate_detection <= 1.0
            or result["passed"] is not expected_cell_pass
        ):
            raise ValueError("Detector Green validation cell metrics are stale.")
    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != set(
        DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT
    ):
        raise ValueError("Detector Green validation metrics are incomplete.")
    for name, (
        comparison,
        threshold,
    ) in DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT.items():
        result = metrics[name]
        if not isinstance(result, Mapping) or set(result) != {
            "value",
            "comparison",
            "threshold",
            "passed",
        }:
            raise ValueError(f"Detector Green validation metric {name} is malformed.")
        value = _finite_number(result["value"], field_name=name)
        if (
            result["comparison"] != comparison
            or result["threshold"] != threshold
            or result["passed"] is not (value <= threshold)
        ):
            raise ValueError(f"Detector Green validation metric {name} is stale.")
    expected_all_passed = bool(
        all(
            isinstance(result, Mapping) and result.get("passed") is True
            for result in cell_results
        )
        and all(result["passed"] is True for result in metrics.values())
    )
    if payload["all_passed"] is not expected_all_passed:
        raise ValueError("Detector Green validation aggregate status is stale.")
    if require_passed and not expected_all_passed:
        raise ValueError("Detector Green validation did not pass.")
    return json.loads(json.dumps(dict(payload), allow_nan=False))


def load_detector_green_validation_manifest(
    path: str | Path,
    *,
    operator: DetectorGreenOperator,
    require_passed: bool = True,
) -> dict[str, object]:
    """Load canonical validation evidence and recompute every metric."""
    manifest_path = Path(path).resolve()
    encoded_manifest = manifest_path.read_bytes()
    manifest = validate_detector_green_validation_manifest(
        json.loads(encoded_manifest),
        operator=operator,
        require_passed=require_passed,
    )
    if encoded_manifest != canonical_json_bytes(manifest):
        raise ValueError("Detector Green validation manifest is not canonical.")
    corpus_path = manifest_path.parent / DETECTOR_GREEN_VALIDATION_RAW_BASENAME
    encoded_corpus = corpus_path.read_bytes()
    if hashlib.sha256(encoded_corpus).hexdigest() != manifest["raw_corpus_sha256"]:
        raise ValueError("Detector Green validation raw corpus hash is stale.")
    corpus = validate_detector_green_validation_raw_corpus(
        json.loads(encoded_corpus),
        operator=operator,
    )
    if encoded_corpus != canonical_json_bytes(corpus):
        raise ValueError("Detector Green validation raw corpus is not canonical.")
    if (
        build_detector_green_validation_manifest(
            corpus,
            operator=operator,
        )
        != manifest
    ):
        raise ValueError("Detector Green validation metrics do not reconstruct.")
    return manifest


def detector_green_validation_manifest_sha256(
    payload: object,
    *,
    operator: DetectorGreenOperator,
) -> str:
    """Return the canonical digest of one passing validation manifest."""
    validated = validate_detector_green_validation_manifest(
        payload,
        operator=operator,
    )
    return hashlib.sha256(canonical_json_bytes(validated)).hexdigest()


__all__ = [
    "DETECTOR_GREEN_VALIDATION_CONTRACT_SHA256",
    "DETECTOR_GREEN_VALIDATION_MANIFEST_BASENAME",
    "DETECTOR_GREEN_VALIDATION_METRIC_CONTRACT",
    "DETECTOR_GREEN_VALIDATION_MINIMUM_HISTORIES_PER_ENERGY",
    "DETECTOR_GREEN_VALIDATION_RAW_BASENAME",
    "build_detector_green_validation_manifest",
    "detector_green_holdout_energies_keV",
    "detector_green_validation_manifest_sha256",
    "evaluate_detector_green_validation_raw_corpus",
    "load_detector_green_validation_manifest",
    "validate_detector_green_validation_manifest",
    "validate_detector_green_validation_raw_corpus",
]
