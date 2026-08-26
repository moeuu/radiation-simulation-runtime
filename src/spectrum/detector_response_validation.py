"""Independent full-detector validation for the analytic response matrix."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from runtime.experiment_profiles import STANDARD_ACQUISITION_LIVE_TIME_S
from spectrum.geant4_physics import (
    GEANT4_PHYSICS_CONTRACT_SHA256,
    geant4_physics_contract_payload,
)
from spectrum.library import default_library
from spectrum.native_metadata import native_source_line_token
from spectrum.response_matrix import (
    NATIVE_GEANT4_BIN_COUNT,
    NATIVE_GEANT4_BIN_WIDTH_KEV,
    NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256,
    build_native_geant4_detector_response_matrix,
)


DETECTOR_RESPONSE_VALIDATION_SCHEMA_VERSION = 2
DETECTOR_RESPONSE_RAW_CORPUS_SCHEMA_VERSION = 1
DETECTOR_RESPONSE_VALIDATION_CONTRACT_ID = (
    "independent_native_full_detector_energy_deposition_v1"
)
DETECTOR_RESPONSE_REFERENCE_GEOMETRY_ID = (
    "shield_free_near_field_detector_cone_v1"
)
DETECTOR_RESPONSE_REFERENCE_SOURCE_DISTANCE_M = 0.05
DETECTOR_RESPONSE_MINIMUM_HISTORIES_PER_ENERGY = 100_000
DETECTOR_RESPONSE_RAW_CORPUS_BASENAME = "raw_corpus.json"
DETECTOR_RESPONSE_VALIDATION_METRIC_CONTRACT = {
    "maximum_total_variation": ("le", 0.10),
    "maximum_photopeak_fraction_absolute_error": ("le", 0.05),
    "maximum_conditional_mean_relative_error": ("le", 0.05),
}


def canonical_json_bytes(payload: object) -> bytes:
    """Return deterministic strict JSON bytes with one trailing newline."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _required_line_descriptors() -> tuple[tuple[float, str, int, str], ...]:
    """Return the exact energy, isotope, source index, and native line token."""
    library = default_library()
    source_isotopes = ("Cs-137", "Co-60", "Eu-154")
    descriptors = [
        (
            float(line.energy_keV),
            isotope,
            source_index,
            native_source_line_token(
                source_index=source_index,
                isotope=isotope,
                energy_keV=float(line.energy_keV),
            ),
        )
        for source_index, isotope in enumerate(source_isotopes)
        for line in library[isotope].lines
        if float(line.intensity) > 0.0
    ]
    return tuple(sorted(descriptors))


def required_detector_response_validation_energies_keV() -> tuple[float, ...]:
    """Return every unique positive line energy used by formal acceptance."""
    return tuple(descriptor[0] for descriptor in _required_line_descriptors())


def detector_response_reference_geometry_payload() -> dict[str, object]:
    """Return the immutable shield-free detector-response geometry contract."""
    return {
        "geometry_id": DETECTOR_RESPONSE_REFERENCE_GEOMETRY_ID,
        "source_distance_from_detector_center_m": (
            DETECTOR_RESPONSE_REFERENCE_SOURCE_DISTANCE_M
        ),
        "static_transport_volume_count": 0,
        "shield_count": 0,
        "source_bias_mode": "detector_cone",
        "source_position_semantics": "air_side_native_emission_xyz",
        "reference_detector_scoring_mode": "full_transport",
        "candidate_detector_scoring_mode": "incident_gamma_energy",
    }


DETECTOR_RESPONSE_REFERENCE_GEOMETRY_SHA256 = hashlib.sha256(
    json.dumps(
        detector_response_reference_geometry_payload(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def _validation_contract_payload() -> dict[str, object]:
    """Return the immutable detector-response validation specification."""
    return {
        "schema_version": DETECTOR_RESPONSE_VALIDATION_SCHEMA_VERSION,
        "raw_corpus_schema_version": (
            DETECTOR_RESPONSE_RAW_CORPUS_SCHEMA_VERSION
        ),
        "contract_id": DETECTOR_RESPONSE_VALIDATION_CONTRACT_ID,
        "response_contract_sha256": (
            NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
        ),
        "geant4_physics_contract_sha256": (
            GEANT4_PHYSICS_CONTRACT_SHA256
        ),
        "reference_geometry_sha256": (
            DETECTOR_RESPONSE_REFERENCE_GEOMETRY_SHA256
        ),
        "dwell_time_s": STANDARD_ACQUISITION_LIVE_TIME_S,
        "required_energy_keV": list(
            required_detector_response_validation_energies_keV()
        ),
        "minimum_histories_per_energy": (
            DETECTOR_RESPONSE_MINIMUM_HISTORIES_PER_ENERGY
        ),
        "metric_contract": {
            metric: {
                "comparison": comparison,
                "threshold": threshold,
            }
            for metric, (comparison, threshold) in (
                DETECTOR_RESPONSE_VALIDATION_METRIC_CONTRACT.items()
            )
        },
    }


DETECTOR_RESPONSE_VALIDATION_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        _validation_contract_payload(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def _is_sha256(value: object) -> bool:
    """Return whether a value is one lowercase SHA-256 digest."""
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(
    payload: Mapping[str, object],
    field_name: str,
    expected: str | None = None,
) -> str:
    """Return one exact digest or reject missing and stale provenance."""
    value = payload.get(field_name)
    if not _is_sha256(value) or (expected is not None and value != expected):
        raise ValueError(f"Detector-response {field_name} is invalid or stale.")
    return str(value)


def _finite_number(value: object, *, field_name: str) -> float:
    """Return one finite JSON number without accepting booleans."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be a finite JSON number.")
    return float(value)


def _positive_integer(value: object, *, field_name: str) -> int:
    """Return one positive JSON integer without accepting booleans."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(f"{field_name} must be a positive JSON integer.")
    return int(value)


def _native_physics_provenance() -> dict[str, object]:
    """Return the exact native fields sealed into the raw response corpus."""
    payload = geant4_physics_contract_payload()
    return {
        "geant4_physics_contract_id": payload["contract_id"],
        "geant4_physics_contract_sha256": (
            GEANT4_PHYSICS_CONTRACT_SHA256
        ),
        **{
            key: value
            for key, value in payload.items()
            if key != "contract_id"
        },
    }


def validate_detector_response_raw_corpus(
    payload: object,
    *,
    expected_native_executable_sha256: str | None = None,
    expected_native_execution_environment_sha256: str | None = None,
    expected_implementation_bundle_sha256: str | None = None,
    expected_runtime_config_sha256: str | None = None,
) -> dict[str, object]:
    """Validate one canonical native full-detector raw corpus."""
    if not isinstance(payload, Mapping):
        raise TypeError("Detector-response raw corpus must be an object.")
    expected_keys = {
        "schema_version",
        "validation_contract_id",
        "validation_contract_sha256",
        "detector_response_contract_sha256",
        "geant4_physics_contract_sha256",
        "reference_geometry_sha256",
        "runtime_config_sha256",
        "native_executable_sha256",
        "native_execution_environment_sha256",
        "implementation_bundle_sha256",
        "reference_scene_sha256",
        "reference_source_contract_sha256",
        "reference_detector_model_sha256",
        "reference_detector_scoring_mode",
        "candidate_detector_scoring_mode",
        "dwell_time_s",
        "transport_seed",
        "histories_per_energy",
        "energy_axis_keV",
        "line_corpora",
        "native_physics_provenance",
    }
    if set(payload) != expected_keys:
        raise ValueError("Detector-response raw corpus schema is incompatible.")
    if (
        payload["schema_version"]
        != DETECTOR_RESPONSE_RAW_CORPUS_SCHEMA_VERSION
        or payload["validation_contract_id"]
        != DETECTOR_RESPONSE_VALIDATION_CONTRACT_ID
        or payload["validation_contract_sha256"]
        != DETECTOR_RESPONSE_VALIDATION_CONTRACT_SHA256
        or payload["detector_response_contract_sha256"]
        != NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
        or payload["geant4_physics_contract_sha256"]
        != GEANT4_PHYSICS_CONTRACT_SHA256
        or payload["reference_geometry_sha256"]
        != DETECTOR_RESPONSE_REFERENCE_GEOMETRY_SHA256
        or payload["reference_detector_scoring_mode"] != "full_transport"
        or payload["candidate_detector_scoring_mode"]
        != "incident_gamma_energy"
    ):
        raise ValueError("Detector-response raw corpus contract is incompatible.")
    for field_name, expected in (
        ("native_executable_sha256", expected_native_executable_sha256),
        (
            "native_execution_environment_sha256",
            expected_native_execution_environment_sha256,
        ),
        ("implementation_bundle_sha256", expected_implementation_bundle_sha256),
        ("runtime_config_sha256", expected_runtime_config_sha256),
        ("reference_scene_sha256", None),
        ("reference_source_contract_sha256", None),
        ("reference_detector_model_sha256", None),
    ):
        _require_sha256(payload, field_name, expected)
    if _finite_number(payload["dwell_time_s"], field_name="dwell_time_s") != (
        STANDARD_ACQUISITION_LIVE_TIME_S
    ):
        raise ValueError("Detector-response dwell time is not the live standard.")
    _positive_integer(payload["transport_seed"], field_name="transport_seed")
    histories = _positive_integer(
        payload["histories_per_energy"],
        field_name="histories_per_energy",
    )
    if histories < DETECTOR_RESPONSE_MINIMUM_HISTORIES_PER_ENERGY:
        raise ValueError("Detector-response validation histories are insufficient.")
    expected_axis = (
        np.arange(NATIVE_GEANT4_BIN_COUNT, dtype=np.float64)
        * NATIVE_GEANT4_BIN_WIDTH_KEV
    )
    raw_axis = payload["energy_axis_keV"]
    if not isinstance(raw_axis, list):
        raise ValueError("Detector-response energy axis must be a JSON array.")
    try:
        axis = np.asarray(raw_axis, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Detector-response energy axis is not numeric.") from exc
    if not np.array_equal(axis, expected_axis):
        raise ValueError("Detector-response energy axis is incompatible.")
    if payload["native_physics_provenance"] != _native_physics_provenance():
        raise ValueError("Detector-response native physics provenance drifted.")
    line_corpora = payload["line_corpora"]
    descriptors = _required_line_descriptors()
    if not isinstance(line_corpora, list) or len(line_corpora) != len(
        descriptors
    ):
        raise ValueError("Detector-response raw corpus does not cover all lines.")
    line_keys = {
        "energy_keV",
        "isotope",
        "source_index",
        "line_token",
        "sampled_histories",
        "history_weight",
        "observed_weighted_spectrum",
        "pulse_count_weighted",
    }
    for index, (entry, descriptor) in enumerate(zip(line_corpora, descriptors)):
        energy, isotope, source_index, line_token = descriptor
        if not isinstance(entry, Mapping) or set(entry) != line_keys:
            raise ValueError(f"Detector-response line corpus {index} is malformed.")
        if (
            _finite_number(entry["energy_keV"], field_name="energy_keV")
            != energy
            or entry["isotope"] != isotope
            or entry["source_index"] != source_index
            or entry["line_token"] != line_token
            or entry["sampled_histories"] != histories
        ):
            raise ValueError(f"Detector-response line corpus {index} is stale.")
        weight = _finite_number(
            entry["history_weight"],
            field_name="history_weight",
        )
        if weight <= 0.0:
            raise ValueError("Detector-response history weight must be positive.")
        spectrum_payload = entry["observed_weighted_spectrum"]
        if not isinstance(spectrum_payload, list):
            raise ValueError("Detector-response spectrum must be a JSON array.")
        try:
            spectrum = np.asarray(spectrum_payload, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Detector-response spectrum is not numeric.") from exc
        pulse_count = _finite_number(
            entry["pulse_count_weighted"],
            field_name="pulse_count_weighted",
        )
        if (
            spectrum.shape != (NATIVE_GEANT4_BIN_COUNT,)
            or np.any(~np.isfinite(spectrum))
            or np.any(spectrum < 0.0)
            or pulse_count < 0.0
            or not math.isclose(
                float(np.sum(spectrum)),
                pulse_count,
                rel_tol=1.0e-12,
                abs_tol=1.0e-9,
            )
            or pulse_count > weight * histories * (1.0 + 1.0e-9)
        ):
            raise ValueError("Detector-response weighted spectrum is invalid.")
    return json.loads(json.dumps(dict(payload), allow_nan=False))


def evaluate_detector_response_raw_corpus(
    payload: object,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]], bool]:
    """Evaluate immutable metrics directly from one validated raw corpus."""
    corpus = validate_detector_response_raw_corpus(payload)
    axis = np.asarray(corpus["energy_axis_keV"], dtype=np.float64)
    response = build_native_geant4_detector_response_matrix(
        axis,
        NATIVE_GEANT4_BIN_WIDTH_KEV,
    )
    histories = int(corpus["histories_per_energy"])
    line_results: list[dict[str, object]] = []
    for entry in corpus["line_corpora"]:
        energy = float(entry["energy_keV"])
        input_index = int(math.floor(energy / NATIVE_GEANT4_BIN_WIDTH_KEV))
        candidate = np.asarray(response[:, input_index], dtype=np.float64)
        observed_weighted = np.asarray(
            entry["observed_weighted_spectrum"],
            dtype=np.float64,
        )
        observed_total = float(np.sum(observed_weighted))
        candidate_total = float(np.sum(candidate))
        if observed_total > 0.0 and candidate_total > 0.0:
            observed = observed_weighted / observed_total
            candidate = candidate / candidate_total
            total_variation = 0.5 * float(np.sum(np.abs(observed - candidate)))
            center = float(axis[input_index])
            sigma = max(0.5 * math.sqrt(center) - 1.5, 0.5)
            peak_mask = np.abs(axis - center) <= 3.0 * sigma
            observed_peak = float(np.sum(observed[peak_mask]))
            candidate_peak = float(np.sum(candidate[peak_mask]))
            observed_mean = float(np.dot(axis, observed))
            candidate_mean = float(np.dot(axis, candidate))
            mean_relative_error = abs(observed_mean - candidate_mean) / max(
                candidate_mean,
                np.finfo(np.float64).tiny,
            )
        else:
            total_variation = 1.0
            observed_peak = 0.0
            candidate_peak = 0.0
            observed_mean = 0.0
            candidate_mean = 0.0
            mean_relative_error = 1.0
        peak_error = abs(observed_peak - candidate_peak)
        passed = bool(
            total_variation
            <= DETECTOR_RESPONSE_VALIDATION_METRIC_CONTRACT[
                "maximum_total_variation"
            ][1]
            and peak_error
            <= DETECTOR_RESPONSE_VALIDATION_METRIC_CONTRACT[
                "maximum_photopeak_fraction_absolute_error"
            ][1]
            and mean_relative_error
            <= DETECTOR_RESPONSE_VALIDATION_METRIC_CONTRACT[
                "maximum_conditional_mean_relative_error"
            ][1]
        )
        pulse_fraction = float(entry["pulse_count_weighted"]) / (
            float(entry["history_weight"]) * histories
        )
        line_results.append(
            {
                "energy_keV": energy,
                "isotope": str(entry["isotope"]),
                "total_variation": total_variation,
                "observed_photopeak_fraction": observed_peak,
                "candidate_photopeak_fraction": candidate_peak,
                "photopeak_fraction_absolute_error": peak_error,
                "observed_conditional_mean_keV": observed_mean,
                "candidate_conditional_mean_keV": candidate_mean,
                "conditional_mean_relative_error": mean_relative_error,
                "pulse_detection_fraction": pulse_fraction,
                "passed": passed,
            }
        )
    metric_values = {
        "maximum_total_variation": max(
            float(result["total_variation"]) for result in line_results
        ),
        "maximum_photopeak_fraction_absolute_error": max(
            float(result["photopeak_fraction_absolute_error"])
            for result in line_results
        ),
        "maximum_conditional_mean_relative_error": max(
            float(result["conditional_mean_relative_error"])
            for result in line_results
        ),
    }
    metrics = {
        metric: {
            "value": float(metric_values[metric]),
            "comparison": comparison,
            "threshold": float(threshold),
            "passed": bool(metric_values[metric] <= threshold),
        }
        for metric, (comparison, threshold) in (
            DETECTOR_RESPONSE_VALIDATION_METRIC_CONTRACT.items()
        )
    }
    all_passed = bool(
        all(result["passed"] is True for result in line_results)
        and all(result["passed"] is True for result in metrics.values())
    )
    return line_results, metrics, all_passed


def build_detector_response_validation_manifest(
    raw_corpus: object,
) -> dict[str, object]:
    """Build one derived manifest whose metrics come only from raw evidence."""
    corpus = validate_detector_response_raw_corpus(raw_corpus)
    line_results, metrics, all_passed = evaluate_detector_response_raw_corpus(
        corpus
    )
    return {
        "schema_version": DETECTOR_RESPONSE_VALIDATION_SCHEMA_VERSION,
        "validation_contract_id": DETECTOR_RESPONSE_VALIDATION_CONTRACT_ID,
        "validation_contract_sha256": (
            DETECTOR_RESPONSE_VALIDATION_CONTRACT_SHA256
        ),
        "detector_response_contract_sha256": (
            NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
        ),
        "geant4_physics_contract_sha256": GEANT4_PHYSICS_CONTRACT_SHA256,
        "reference_geometry_sha256": (
            DETECTOR_RESPONSE_REFERENCE_GEOMETRY_SHA256
        ),
        "runtime_config_sha256": corpus["runtime_config_sha256"],
        "native_executable_sha256": corpus["native_executable_sha256"],
        "native_execution_environment_sha256": (
            corpus["native_execution_environment_sha256"]
        ),
        "implementation_bundle_sha256": (
            corpus["implementation_bundle_sha256"]
        ),
        "reference_scene_sha256": corpus["reference_scene_sha256"],
        "reference_source_contract_sha256": (
            corpus["reference_source_contract_sha256"]
        ),
        "reference_detector_model_sha256": (
            corpus["reference_detector_model_sha256"]
        ),
        "reference_detector_scoring_mode": "full_transport",
        "candidate_detector_scoring_mode": "incident_gamma_energy",
        "evaluated_energy_keV": list(
            required_detector_response_validation_energies_keV()
        ),
        "histories_per_energy": int(corpus["histories_per_energy"]),
        "transport_seed": int(corpus["transport_seed"]),
        "dwell_time_s": float(corpus["dwell_time_s"]),
        "raw_corpus_sha256": hashlib.sha256(
            canonical_json_bytes(corpus)
        ).hexdigest(),
        "line_results": line_results,
        "metrics": metrics,
        "all_passed": all_passed,
    }


def validate_detector_response_validation_manifest(
    payload: object,
    *,
    expected_native_executable_sha256: str | None = None,
    expected_native_execution_environment_sha256: str | None = None,
    expected_implementation_bundle_sha256: str | None = None,
    expected_runtime_config_sha256: str | None = None,
    require_passed: bool = True,
) -> dict[str, object]:
    """Validate one derived response manifest without trusting loose fields."""
    if not isinstance(payload, Mapping):
        raise TypeError("Detector-response validation manifest must be an object.")
    expected_keys = {
        "schema_version",
        "validation_contract_id",
        "validation_contract_sha256",
        "detector_response_contract_sha256",
        "geant4_physics_contract_sha256",
        "reference_geometry_sha256",
        "runtime_config_sha256",
        "native_executable_sha256",
        "native_execution_environment_sha256",
        "implementation_bundle_sha256",
        "reference_scene_sha256",
        "reference_source_contract_sha256",
        "reference_detector_model_sha256",
        "reference_detector_scoring_mode",
        "candidate_detector_scoring_mode",
        "evaluated_energy_keV",
        "histories_per_energy",
        "transport_seed",
        "dwell_time_s",
        "raw_corpus_sha256",
        "line_results",
        "metrics",
        "all_passed",
    }
    if set(payload) != expected_keys:
        raise ValueError(
            "Detector-response validation manifest has an incompatible schema."
        )
    if (
        payload["schema_version"]
        != DETECTOR_RESPONSE_VALIDATION_SCHEMA_VERSION
        or payload["validation_contract_id"]
        != DETECTOR_RESPONSE_VALIDATION_CONTRACT_ID
        or payload["validation_contract_sha256"]
        != DETECTOR_RESPONSE_VALIDATION_CONTRACT_SHA256
        or payload["detector_response_contract_sha256"]
        != NATIVE_GEANT4_DETECTOR_RESPONSE_CONTRACT_SHA256
        or payload["geant4_physics_contract_sha256"]
        != GEANT4_PHYSICS_CONTRACT_SHA256
        or payload["reference_geometry_sha256"]
        != DETECTOR_RESPONSE_REFERENCE_GEOMETRY_SHA256
        or payload["reference_detector_scoring_mode"] != "full_transport"
        or payload["candidate_detector_scoring_mode"]
        != "incident_gamma_energy"
        or not isinstance(payload["all_passed"], bool)
    ):
        raise ValueError("Detector-response validation identity is invalid.")
    for field_name, expected in (
        ("native_executable_sha256", expected_native_executable_sha256),
        (
            "native_execution_environment_sha256",
            expected_native_execution_environment_sha256,
        ),
        ("implementation_bundle_sha256", expected_implementation_bundle_sha256),
        ("runtime_config_sha256", expected_runtime_config_sha256),
        ("reference_scene_sha256", None),
        ("reference_source_contract_sha256", None),
        ("reference_detector_model_sha256", None),
        ("raw_corpus_sha256", None),
    ):
        _require_sha256(payload, field_name, expected)
    energies = payload["evaluated_energy_keV"]
    required_energies = required_detector_response_validation_energies_keV()
    if (
        not isinstance(energies, list)
        or tuple(energies) != required_energies
        or any(type(value) is not float for value in energies)
    ):
        raise ValueError(
            "Detector-response validation must cover every formal line energy."
        )
    histories = _positive_integer(
        payload["histories_per_energy"],
        field_name="histories_per_energy",
    )
    if histories < DETECTOR_RESPONSE_MINIMUM_HISTORIES_PER_ENERGY:
        raise ValueError("Detector-response validation histories are insufficient.")
    _positive_integer(payload["transport_seed"], field_name="transport_seed")
    if _finite_number(payload["dwell_time_s"], field_name="dwell_time_s") != (
        STANDARD_ACQUISITION_LIVE_TIME_S
    ):
        raise ValueError("Detector-response validation dwell time drifted.")
    line_results = payload["line_results"]
    expected_line_keys = {
        "energy_keV",
        "isotope",
        "total_variation",
        "observed_photopeak_fraction",
        "candidate_photopeak_fraction",
        "photopeak_fraction_absolute_error",
        "observed_conditional_mean_keV",
        "candidate_conditional_mean_keV",
        "conditional_mean_relative_error",
        "pulse_detection_fraction",
        "passed",
    }
    if not isinstance(line_results, list) or len(line_results) != len(
        required_energies
    ):
        raise ValueError("Detector-response line results are incomplete.")
    for result, descriptor in zip(line_results, _required_line_descriptors()):
        if (
            not isinstance(result, Mapping)
            or set(result) != expected_line_keys
            or result["energy_keV"] != descriptor[0]
            or result["isotope"] != descriptor[1]
            or not isinstance(result["passed"], bool)
        ):
            raise ValueError("Detector-response line result is malformed.")
        numeric_fields = expected_line_keys - {"isotope", "passed"}
        values = {
            field: _finite_number(result[field], field_name=field)
            for field in numeric_fields
        }
        expected_line_pass = bool(
            values["total_variation"]
            <= DETECTOR_RESPONSE_VALIDATION_METRIC_CONTRACT[
                "maximum_total_variation"
            ][1]
            and values["photopeak_fraction_absolute_error"]
            <= DETECTOR_RESPONSE_VALIDATION_METRIC_CONTRACT[
                "maximum_photopeak_fraction_absolute_error"
            ][1]
            and values["conditional_mean_relative_error"]
            <= DETECTOR_RESPONSE_VALIDATION_METRIC_CONTRACT[
                "maximum_conditional_mean_relative_error"
            ][1]
        )
        if result["passed"] is not expected_line_pass:
            raise ValueError("Detector-response line pass flag is inconsistent.")
    metrics = payload["metrics"]
    if (
        not isinstance(metrics, Mapping)
        or set(metrics) != set(DETECTOR_RESPONSE_VALIDATION_METRIC_CONTRACT)
    ):
        raise ValueError("Detector-response validation metrics are incomplete.")
    for metric, (comparison, threshold) in (
        DETECTOR_RESPONSE_VALIDATION_METRIC_CONTRACT.items()
    ):
        result = metrics[metric]
        if not isinstance(result, Mapping) or set(result) != {
            "value",
            "comparison",
            "threshold",
            "passed",
        }:
            raise ValueError(f"Detector-response metric {metric} is malformed.")
        value = _finite_number(result["value"], field_name=metric)
        if (
            result["comparison"] != comparison
            or result["threshold"] != threshold
            or result["passed"] is not (value <= threshold)
        ):
            raise ValueError(f"Detector-response metric {metric} is inconsistent.")
    expected_all_passed = bool(
        all(result["passed"] is True for result in line_results)
        and all(result["passed"] is True for result in metrics.values())
    )
    if payload["all_passed"] is not expected_all_passed:
        raise ValueError("Detector-response aggregate status is inconsistent.")
    if require_passed and not expected_all_passed:
        raise ValueError("Detector-response validation did not pass.")
    return json.loads(json.dumps(dict(payload), allow_nan=False))


def load_detector_response_validation_manifest(
    path: str | Path,
    *,
    expected_native_executable_sha256: str | None = None,
    expected_native_execution_environment_sha256: str | None = None,
    expected_implementation_bundle_sha256: str | None = None,
    expected_runtime_config_sha256: str | None = None,
    require_passed: bool = True,
) -> dict[str, object]:
    """Load a manifest, authenticate its raw corpus, and recompute all metrics."""
    manifest_path = Path(path).resolve()
    encoded_manifest = manifest_path.read_bytes()
    try:
        raw_manifest = json.loads(encoded_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Detector-response manifest is not valid JSON.") from exc
    manifest = validate_detector_response_validation_manifest(
        raw_manifest,
        expected_native_executable_sha256=expected_native_executable_sha256,
        expected_native_execution_environment_sha256=(
            expected_native_execution_environment_sha256
        ),
        expected_implementation_bundle_sha256=(
            expected_implementation_bundle_sha256
        ),
        expected_runtime_config_sha256=expected_runtime_config_sha256,
        require_passed=require_passed,
    )
    if encoded_manifest != canonical_json_bytes(manifest):
        raise ValueError("Detector-response manifest is not canonical JSON.")
    corpus_path = manifest_path.parent / DETECTOR_RESPONSE_RAW_CORPUS_BASENAME
    encoded_corpus = corpus_path.read_bytes()
    if hashlib.sha256(encoded_corpus).hexdigest() != manifest["raw_corpus_sha256"]:
        raise ValueError("Detector-response raw corpus hash is stale.")
    try:
        raw_corpus = json.loads(encoded_corpus)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Detector-response raw corpus is not valid JSON.") from exc
    corpus = validate_detector_response_raw_corpus(
        raw_corpus,
        expected_native_executable_sha256=expected_native_executable_sha256,
        expected_native_execution_environment_sha256=(
            expected_native_execution_environment_sha256
        ),
        expected_implementation_bundle_sha256=(
            expected_implementation_bundle_sha256
        ),
        expected_runtime_config_sha256=expected_runtime_config_sha256,
    )
    if encoded_corpus != canonical_json_bytes(corpus):
        raise ValueError("Detector-response raw corpus is not canonical JSON.")
    recomputed = build_detector_response_validation_manifest(corpus)
    if recomputed != manifest:
        raise ValueError(
            "Detector-response manifest metrics do not match the raw corpus."
        )
    return manifest


def detector_response_validation_manifest_sha256(payload: object) -> str:
    """Return the canonical hash of one already validated passing manifest."""
    validated = validate_detector_response_validation_manifest(payload)
    return hashlib.sha256(canonical_json_bytes(validated)).hexdigest()


__all__ = [
    "DETECTOR_RESPONSE_MINIMUM_HISTORIES_PER_ENERGY",
    "DETECTOR_RESPONSE_RAW_CORPUS_BASENAME",
    "DETECTOR_RESPONSE_REFERENCE_GEOMETRY_SHA256",
    "DETECTOR_RESPONSE_VALIDATION_CONTRACT_ID",
    "DETECTOR_RESPONSE_VALIDATION_CONTRACT_SHA256",
    "DETECTOR_RESPONSE_VALIDATION_METRIC_CONTRACT",
    "DETECTOR_RESPONSE_VALIDATION_SCHEMA_VERSION",
    "build_detector_response_validation_manifest",
    "detector_response_reference_geometry_payload",
    "detector_response_validation_manifest_sha256",
    "evaluate_detector_response_raw_corpus",
    "load_detector_response_validation_manifest",
    "required_detector_response_validation_energies_keV",
    "validate_detector_response_raw_corpus",
    "validate_detector_response_validation_manifest",
]
